# backend/app/services/rag_query_service.py

import os
import re
import gc
import glob
import hashlib
import json
from threading import Thread
from operator import itemgetter
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
    pipeline,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from FlagEmbedding import FlagReranker

import backend.app.core.database as db


class RAGQueryService:
    def __init__(self):
        # --- 1. Detect optimal device and compute dtype
        print(f"Loading embedding model: {db.EMBEDDING_MODEL_NAME}...")
        self.device, self.compute_dtype = self._get_optimal_device_info()

        # --- 2. Initialize embedding model
        print(f"Loading embedding model on {self.device}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=db.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": self.device},
        )

        # --- 3. Initialize core components
        self.vector_store = None
        self.retriever = None
        self.model = None
        self.llm = None
        self.reranker = None
        self.qa_chain = None
        self.tokenizer = None
        # Prompt template for streaming mode
        self.template = """You are a highly precise Financial Data Analysis Expert. 
        Your task is to extract information from the provided context with absolute accuracy.
        
        ### OPERATIONAL RULES:
        1. **ENTITY-ACTION MAPPING**: You must strictly map actions, roles, and dates to the correct specific entities (individuals, companies, or departments). Do NOT swap attributes between different entities.
        2. **TRANSITION LOGIC**: For any changes, transitions, or chronological events (e.g., personnel changes, financial growth, policy updates), clearly distinguish between the "PREVIOUS" state and the "NEW" state. Identify specifically WHO is leaving/changing and WHO is arriving/taking over.
        3. **SUBJECT-PREDICATE INTEGRITY**: Ensure that the relationship between the subject and the predicate is preserved exactly as stated in the context.
        4. **NO ASSUMPTIONS**: Only use the provided information. If the context is ambiguous, state the ambiguity instead of guessing.
        5. {language_instruction}
        
        6. **EXPERT RESPONSE FORMAT**: Provide a concise but complete sentence. For example, instead of just a name, say "Based on [Source], [Name] will [Action]...". Ensure the answer directly addresses the core of the question.
        7. CROSS-SOURCE VALIDATION: If the context contains both official filings (e.g., Form 10-K) and news reports, prioritize official filings for formal organizational changes, but include news for recent strategic shifts. State clearly if sources provide conflicting or different information.
        8. NUMERICAL RIGOR: Pay extra attention to years and dates. Ensure the year (e.g., 2025, 2026) is extracted exactly as written in the source text.
        
        ### CONTEXT:
        {context}
        
        ### QUESTION:
        {question}
        
        ### FINAL ANSWER:"""     
        
        # --- 4. Initialize LLM
        print(f" Pre-loading LLM on {self.device}...")
        self._prepare_llm()  

        # --- 5. Load existing vector store if available
        if os.path.exists(db.VECTOR_DB_DIR) and os.listdir(db.VECTOR_DB_DIR):
            print(f"✅ Found existing vector store at {db.VECTOR_DB_DIR}, loading...")
            self._init_vector_store()
        else:
            print(
                f"⚠️ No vector store found at {db.VECTOR_DB_DIR}. "
                "Please run 'ingest_data_into_vector_store()' first."
            )

    def _get_optimal_device_info(self):
        """Determine the optimal device and torch dtype."""
        if torch.cuda.is_available():
            # if bf16 cannot work, need to set dtype = torch.float16
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return "cuda", dtype
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    def _init_vector_store(self):
        """Initialize the Chroma vector store and retriever."""
        try:
            self.vector_store = Chroma(
                persist_directory=db.VECTOR_DB_DIR,
                embedding_function=self.embeddings,
            )
            print(f"Vector store initialized at {db.VECTOR_DB_DIR}")

            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs = {
                    "k": 30
                }

            )
            print("Chroma DB loaded and retriever created.")
        except Exception as e:
            print(
                f"❌ Error loading Chroma DB: {e}. "
                "Ensure DataPipelineService has completed successfully."
            )
            self.vector_store = None

    def _prepare_llm(self):
        """
        Lazily initialize the LLM.
        Large models are only loaded upon the first query.
        """
        if self.llm is not None:
            return

        print(
            f"First query detected. Initializing LLM on {self.device} "
            f"(dtype: {self.compute_dtype})..."
        )

        try:
            if self.device == "mps":
                self.model = self.model.to("mps")

            print(f" start to initialize LLM with 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

            print(f"Loading tokenizer and model: {db.LLM_MODEL_NAME}...")
            self.tokenizer = AutoTokenizer.from_pretrained(db.LLM_MODEL_NAME)

            self.model = AutoModelForCausalLM.from_pretrained(
                db.LLM_MODEL_NAME,
                device_map="auto" if self.device == "cuda" else None,
                quantization_config=bnb_config,
                dtype=self.compute_dtype,
                low_cpu_mem_usage=True,
            )

            self.reranker = FlagReranker(
                db.RERANKER_MODEL_NAME,
                device=self.device,
                use_fp16=(self.device == "cuda")
            )
            print(f" {db.RERANKER_MODEL_NAME} loaded successfully. ")

            pipe = pipeline(
                task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.1,  # Low temperature for focused responses
                top_p=0.95,
                repetition_penalty=1.1,
                return_full_text=False,
            )

            self.llm = HuggingFacePipeline(pipeline=pipe)

            # Build QA chain
            prompt = ChatPromptTemplate.from_template(self.template)

            self.qa_chain = (
                {
                    "context": itemgetter("context"), 
                    "question": RunnablePassthrough(), 
                    "language_instruction": itemgetter("lang_instruction")
                    }
                | prompt
                | self.llm.bind(stop=["Question:", "User:", "###", "\n\n\n"])
            )

            print("LLM pipeline initialized successfully.")
        except Exception as e:
            print(f"❌ Error initializing LLM pipeline: {e}")
            print("❌ Falling back to None. RAG queries will fail.")
    
    
    def _get_reranked_context(self, question: str):
        """"""
        # 1. rough rank ( retriever set k=30)
        docs = self.retriever.invoke(question)
        if not self.reranker or not docs:
            return docs[:7]
        
        # 2. Rerank
        pairs = [[question, d.page_content] for d in docs]
        scores = self.reranker.compute_score(pairs)

        # 3. add weight and sort
        scored_list = []
        for doc, score in zip(docs, scores):
            source = doc.metadata.get('source', '').lower()
            # 10-k with more weight
            boosted_score = score + 0.8 if ("edgar" in source or "10k" in source) else score
            scored_list.append((doc, boosted_score))

        # sorted by score desc
        scored_list.sort(key=lambda x: x[1], reverse=True)

        # Debug
        print("\n[Rerank Result]")
        for i, (d, s) in enumerate(scored_list[:5]):
            print(f"Rank {i}: Score {s:.2f} | Source: {d.metadata.get('source')}")
        
        return [d for d, s in scored_list[:7]]
    
    def _get_language_instruction(self, text: str) -> str:
        """Extract the language instruction from the text."""
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return "请使用**中文**回答。直接给出结果，不要显示分析步骤。"
        else:
            return "Please respond in **English**. Provide the answer directly without showing your thought process."



    def _clean_output_chunk(self, text: str) -> str:
        """Clean model output by removing special tokens and prefixes."""
        stop_tokens = ["<|endoftext|>", "<|im_end|>", "</s>", "<pad>"]
        for token in stop_tokens:
            text = text.replace(token, "")

        text = re.sub(r"^(回答|答|Answer|Output):\s*", "", text)
        return text

    def ingest_data_into_vector_store(self):
        """
        Ingest data into the vector store.
        Supports mixed input formats defined in SOURCES_CONFIG.
        """
        print("--- Launching data ingestion pipeline ---")
        all_chunks = []

        for key, config in db.SOURCES_CONFIG.items():
            print(f"Processing data source: {key}...")

            if config.get("format") == "json":
                chunks = self.load_json_chunks(key)
            elif config.get("format") == "txt":
                chunks = self.load_and_chunk_txt_data(key)
            else:
                print(
                    f"⚠️ Unknown format '{config.get('format')}' "
                    f"for key '{key}'. Skipping."
                )
                continue

            all_chunks.extend(chunks)

        if not all_chunks:
            print("⚠️ No data chunks to ingest.")
            return
        
        # chunk_id is the unique tag for all documents
        all_ids = [doc.metadata["chunk_id"] for doc in all_chunks]

        if not self.vector_store:

            self.vector_store = Chroma.from_documents(
                documents=all_chunks,
                embedding=self.embeddings,
                persist_directory=db.VECTOR_DB_DIR,
                ids=all_ids
            )
        else:
            self.vector_store.add_documents(all_chunks)

        print(
            f"🎉 Data ingestion completed. "
            f"{len(all_chunks)} chunks stored in {db.VECTOR_DB_DIR}."
        )

    def load_and_chunk_txt_data(self, config_key):
        """Load and chunk TXT files based on configuration."""
        config = db.SOURCES_CONFIG[config_key]
        documents = []

        # 1. initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            separators=["\n\n", "\n", ".", " ", ""],
        )

        # 2. traverse files in the specified directory
        for file_path in glob.glob(os.path.join(config["path"], "*.txt")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                    text = re.sub(r'\n+', '\n', text)

                source_name = os.path.basename(file_path)

                # 3. split text into chunks
                chunks = text_splitter.split_text(text)

                # 4. transform chunks into Document objects with metadata
                for i, chunk_data in enumerate(chunks):

                    content_hash = hashlib.md5(chunk_data.encode('utf-8')).hexdigest()
                    unique_id = f"{source_name}_{i+1}_{content_hash[:8]}"

                    doc = Document(
                        page_content=chunk_data,
                        metadata={
                            "source": source_name,
                            "type": config["doc_type"],
                            "chunk_id": unique_id,
                            "chunk_size": len(chunk_data),
                        },
                    )
                    documents.append(doc)

                print(f" -> {source_name}: {len(chunks)} chunks processed.")
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")

        return documents

    def load_json_chunks(self, config_key):
        """Load pre-chunked JSON data."""
        config = db.SOURCES_CONFIG[config_key]
        documents = []

        print(f"Loading pre-processed JSON data from: {config['path']}")
        # Iterating through a JSON file
        for file_path in glob.glob(os.path.join(config["path"], "*.json")):
            file_name = os.path.basename(file_path)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    # Assume the JSON file is a list containing dictionaries of RAG blocks.
                    json_data = json.load(f)

                for i, chunk_data in enumerate(json_data):
                    # Check if the required fields exist.
                    if 'text' not in chunk_data or not chunk_data['text'].strip():
                        continue

                    raw_text = chunk_data.pop('text')
                    section_title = chunk_data.get("section_title", "")
                    doc_title = chunk_data.get("document_title", "")
                    
                    combined_content = f"Item Title: {section_title}\nDocument: {doc_title}\nDescription: {raw_text}"


                    content_hash = hashlib.md5(combined_content.encode('utf-8')).hexdigest()
                    unique_id = f"json_{file_name}_{i}_{content_hash[:8]}"

                    # Merge additional metadata
                    metadata = {
                        "source": chunk_data.get("source_file", file_name),
                        "type": config["doc_type"],
                        "chunk_id": unique_id,
                        **chunk_data,
                    }

                    documents.append(
                        Document(page_content=combined_content, metadata=metadata)
                    )

                print(f" -> {file_name}: {len(json_data)} documents loaded.")
            except Exception as e:
                print(f"❌ Error processing JSON file {file_path}: {e}")

        return documents

    def query(self, question: str):
        """Execute a standard (non-streaming) RAG query."""
        import time
        start_time = time.time()
        print(f"DEBUG: 开始检索耗时 {start_time}")
        
              
        if not self.vector_store:
            return "Error: Vector store is not initialized."

        if self.model is None or self.tokenizer is None:
            return " Error: LLM Model is not initialized. Check Initialization."


        lang_inst = self._get_language_instruction(question)

        final_docs = self._get_reranked_context(question)

        context_text = "\n\n".join([f"Source: {d.metadata.get('source')}\n{d.page_content}" for d in final_docs])
        
        
        print(f"DEBUG: 检索完成耗时 {time.time() - start_time}")
        mid_time = time.time()

        if self.qa_chain:
            print(f"\n ----- Querying RAG system with question: {question} -----")
            response = self.qa_chain.invoke({
                "context": context_text,
                "question": question,
                "lang_instruction": lang_inst
            })
            
            print(f"DEBUG: LLM生成耗时 {time.time() - mid_time}")

            clean_response = response.split("Question:")[0].split("###")[0].strip()
            return clean_response

        return "Error: QA chain is not initialized."
    
    async def query_stream(self, question: str):
        """Execute a streaming RAG query."""
        if not self.vector_store:
            yield " Error: Vector store is not initialized."
            return
        
        if self.model is None or self.tokenizer is None:
            yield " Error: LLM Model is not initialized. Check Initialization."
            return

        lang_inst = self._get_language_instruction(question)

        # 1. checking stage (k=10)
        final_docs = self._get_reranked_context(question)
        context_text = "\n\n".join([f"Source: {d.metadata.get('source')}\n{d.page_content}" for d in final_docs])
        context_parts = []
        for i, d in enumerate(final_docs):
            source_raw = d.metadata.get('source', 'Unknown')
            source_label = source_raw.split('__')[0]

            doc_str = f"[DOCUMENT {i+1}]\nSOURCE: {source_label}\nCONTENT: {d.page_content}"
            context_parts.append(doc_str)
        context_text = "\n\n".join(context_parts)


        # 2. Prompt
        full_prompt = self.template.format(context=context_text, question=question, language_instruction = lang_inst)

        # 3. set streamer
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.1,
            "top_p": 0.9,
            "repetition_penalty": 1.2
        }

        # 4. generate
        print(f"\n ----- Querying RAG system with question: {question} -----")
        def threaded_generate():
            with torch.no_grad():
                self.model.generate(**generation_kwargs)
        
        thread = Thread(target=threaded_generate)
        thread.start()

        # 5. output
        first_chunk = True
        try:
            for new_text in streamer:
                if new_text:
                    # 只有在第一帧或者还没开始输出正文时才清理前缀
                    if first_chunk:
                        # 这里的正则去掉了 \s*，改为手动处理或更精确的匹配
                        cleaned_text = re.sub(r"^(回答|答|Answer|Output):", "", new_text).lstrip()
                        if cleaned_text:
                            first_chunk = False # 只要清理出内容了，就标记为非第一帧
                        yield cleaned_text
                    else:
                        # 后续片段直接清理停止符，不再清理前缀和空格
                        stop_tokens = ["<|endoftext|>", "<|im_end|>", "</s>", "<pad>"]
                        for token in stop_tokens:
                            new_text = new_text.replace(token, "")
                        yield new_text
        # try:
        #     for new_text in streamer:
        #         if new_text:
        #         # clean output by removing special tokens and prefixes（such as <|im_end|>）
        #             cleaned_text = self._clean_output_chunk(new_text)
        #             if cleaned_text:
        #                 yield cleaned_text
        finally:
            thread.join()

    def clear_gpu_memory(self):
        """Manually release GPU and model memory."""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Memory cleared.")

    # async def query(self, text):
    #     # 这里只做纯粹的检索逻辑，不加载模型
    #     vector = self.model.encode(text)
    #     results = self.vector_db.search(vector)
    #     return results


# if __name__ == "__main__":
#     service = RAGQueryService()
# #     #service.ingest_data_into_vector_store()

#     test_question = "what is 10-k report?"
#     #test_question = "根据财报和最近的新闻，苹果的财务领导层和 AI 团队分别有什么最新的人事变动？"
# # #     test_question = "分别查阅 10-K 财报和最新新闻，总结苹果财务团队和 AI 团队的变动。"
#     #print(service.query(test_question))

#     for chunk in service.query_stream(test_question):
#         print(chunk,end="", flush=True)
