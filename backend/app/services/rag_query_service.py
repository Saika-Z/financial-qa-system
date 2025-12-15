# backend/app/services/rag_query_service.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_core.documents import Document
#from langchain.chains import RAG
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
#from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser

import os
import glob
import backend.app.core.database as db
import json


class RAGQueryService:
    def __init__(self):
        # 1. initialize embedding model
        print(f"Loading embedding model: {db.EMBEDDING_MODEL_NAME}...")
        # HuggingFaceEmbeddings will download the model if not present locally
        self.embeddings = HuggingFaceEmbeddings(model_name=db.EMBEDDING_MODEL_NAME)
        
        # 2. initialize vector store
        print("initializing vector store...")
        # Chroma.from_documents will create the vector store if not exists
        try:
            self.vector_store = Chroma(
            persist_directory=db.VECTOR_DB_DIR, 
            embedding_function=self.embeddings
            )
            print(f"vector store initialized at {db.VECTOR_DB_DIR}")

            # 3. create accessor method for querying
            # k = 4 means retrieve top 4 similar documents
            self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            print("Chroma DB loaded from {VECTOR_DB_DIR} and retriever created.")

        except Exception as e:
            print(f"   ❌ Error loading Chroma DB: {e}. Ensure DataPipelineService has run.")
            self.vector_store = None

        # 4. ----- initialize LLM model for RAG -----
        print("Initializing HuggingFace Pipeline with model: db.{LLM_MODEL_NAME}...")
        try:
            # load auto model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(db.LLM_MODEL_NAME)

            # Dynamically check for the presence of a GPU and set loading parameters.
            device = "cuda" if torch.cuda.is_available() else "cpu"

            model = AutoModelForCausalLM.from_pretrained(
                db.LLM_MODEL_NAME,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )

            pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.1,  # low temperature for more focused responses
                top_p=0.95,
                return_full_text=False,
                device=device
            )
            self.llm = HuggingFacePipeline(pipeline=pipeline)
            print("LLM Pipeline initialized successfully. on {device}. ")

        except Exception as e:
            print(f"   ❌ Error initializing LLM Pipeline: {e}.")
            print("   ❌ Falling back to None. RAG queries will fail.")
            self.llm = None

        # ----- LLM initialization ends -----

        if self.retriever and self.llm:
            # 5. create QA chain
            #self.qa_chain = RAG.from_llm(llm=self.llm, retriever=self.retriever)
            prompt = ChatPromptTemplate.from_template(
                "Answer the question based on the following context:\n\n{context}\n\nQuestion: {question}"
                )
            self.qa_chain = (
                RunnableParallel({"context": self.retriever, "question": RunnablePassthrough()})
                | prompt
                | self.llm
                )
        else:
            self.qa_chain = None 

    def ingest_data_into_vector_store(self):
        """
        main method to ingest data into the vector store, supporting mixed formats.
        """
        print("--- launching data ingestion pipeline ---")
        all_chunks = []

        for key, config in db.SOURCES_CONFIG.items():
            print(f"Solving data source: {key}...")
            
            # Different loaders are called based on the 'format' field in the configuration.
            if config.get('format') == 'json':
                chunks = self.load_json_chunks(key)
            elif config.get('format') == 'txt':
                # Ensure that `load_and_chunk_data` is adapted for processing TXT files.
                chunks = self.load_and_chunk_txt_data(key) 
            else:
                print(f"   ⚠️ WARNING: Unknown format '{config.get('format')}' for key '{key}'. Skipping.")
                continue

            all_chunks.extend(chunks)
        if not all_chunks:
            print("warning: no data chunks to ingest.")
            return

        # 5. Embed and store in vector DB
        print(f"\nstarting to embed and store {len(all_chunks)} chunks into vector store...")
        # Chroma.add_documents will handle embedding and storage
        self.vector_store.add_documents(all_chunks)
        
        # 6. Persist the vector store to disk
        print(f" 🎉 Data ingestion completed and vector store persisted at {db.VECTOR_DB_DIR}.")
    
    def load_and_chunk_txt_data(self, config_key):
        """
        Load and chunk text data from files based on the provided configuration.
        """
        config = db.SOURCES_CONFIG[config_key]
        all_documents = []
        
        # 1. initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap'],
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        # 2. traverse files in the specified directory
        for file_path in glob.glob(os.path.join(config['path'], '*.txt')):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                source_name = os.path.basename(file_path)
                
                # 3. split text into chunks
                chunks = text_splitter.split_text(text)
                
                # 4. transform chunks into Document objects with metadata
                for i, chunk in enumerate(chunks):
                    # ensure chunk is not empty
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": source_name,
                            "type": config['doc_type'],
                            "chunk_id": f"{source_name}_{i+1}",
                            "chunk_size": len(chunk),
                            # additional metadata can be added here
                            # "original_file": file_path,
                        }
                    )
                    all_documents.append(doc)
                
                print(f" -> {source_name}: {len(chunks)} chunks processed.")

            except Exception as e:
                print(f"   ❌ Error processing {file_path}: {e}")
        
        return all_documents
    
    def load_json_chunks(self, config_key):
        """
        Load pre-chunked data from JSON files.
        """
        config = db.SOURCES_CONFIG[config_key]
        all_documents = []
        
        print(f"Loading pre-processed JSON data from: {config['path']}")

        # Iterating through a JSON file
        for file_path in glob.glob(os.path.join(config['path'], '*.json')):
            file_name = os.path.basename(file_path)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Assume the JSON file is a list containing dictionaries of RAG blocks.
                    json_data = json.load(f)

                for chunk_data in json_data:
                    # Check if the required fields exist.
                    if 'text' not in chunk_data or not chunk_data['text'].strip():
                        continue 

                    # Extract the text as `page_content`, and the rest as `metadata`.
                    page_content = chunk_data.pop('text')
                    
                    # Merge additional metadata
                    metadata = {
                        "source": chunk_data.get("source_file", file_name),
                        "type": config['doc_type'],
                        "section_title": chunk_data.get("section_title", "N/A"),
                        **chunk_data # Use the other key-value pairs in the JSON as metadata.
                    }

                    doc = Document(page_content=page_content, metadata=metadata)
                    all_documents.append(doc)

                print(f" -> {file_name}: {len(json_data)} pre-chunked documents loaded.")

            except Exception as e:
                print(f"   ❌ Error processing JSON file {file_path}: {e}")
                
        return all_documents
    
    def query(self, question: str, top_k: int = 5):
        """
        Query the vector store with a question and retrieve top_k relevant documents.
        """
        if not self.qa_chain:
            return "RAG System not properly initialized. check LLM and Vector DB status."
        
        print(f"\n ----- Querying RAG system with question: {question} -----")

        # RAG system to get answer
        response = self.qa_chain.invoke(question)
        return response

    

if __name__ == "__main__":
    # ensure data directories exist in backend/data/
    # for key in db.SOURCES_CONFIG.keys():
    #     os.makedirs(db.SOURCES_CONFIG[key]['path'], exist_ok=True)
    
    # # ensure kb directory exists in backend/data/kb/
    # os.makedirs(os.path.join(os.getcwd(), 'data', 'kb'), exist_ok=True)
    
    # # cumsume the cleaned TXT f
    # # Assumption: You have already placed the cleaned TXT files in the appropriate directory.
    
    # pipeline = RAGQueryService()
    # pipeline.ingest_data_into_vector_store()
    question = RAGQueryService()
    anwer = question.query("Form 10-k includes which main sections?")
    print(f"\nRAG Answer: {anwer}")