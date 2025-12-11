from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import os
import glob
import core.database as db


class DataPipelineService:
    def __init__(self):
        # 1. 初始化嵌入模型 (Embedding Model)
        print(f"正在加载嵌入模型: {db.EMBEDDING_MODEL_NAME}...")
        # HuggingFaceEmbeddings 包装了 sentence-transformers 库
        self.embeddings = HuggingFaceEmbeddings(model_name=db.EMBEDDING_MODEL_NAME)
        
        # 2. 初始化向量数据库连接 (使用 ChromaDB 作为本地数据库)
        # Chroma.from_documents 是导入数据的标准方式
        self.vector_store = Chroma(
            persist_directory=db.VECTOR_DB_DIR, 
            embedding_function=self.embeddings
        )
        print(f"向量数据库存储路径: {db.VECTOR_DB_DIR}")

    def load_and_chunk_data(self, config_key):
        """
        加载指定数据源的所有文件，并根据配置进行分块。
        """
        config = db.SOURCES_CONFIG[config_key]
        all_documents = []
        
        # 1. 初始化递归文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap'],
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        # 2. 遍历目录下所有 TXT 文件
        for file_path in glob.glob(os.path.join(config['path'], '*.txt')):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                source_name = os.path.basename(file_path)
                
                # 3. 执行分块
                chunks = text_splitter.split_text(text)
                
                # 4. 转换为 LangChain Document 格式并添加元数据
                for i, chunk in enumerate(chunks):
                    # 确保每个块都附带关键元数据
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": source_name,
                            "type": config['doc_type'],
                            "chunk_id": f"{source_name}_{i+1}",
                            "chunk_size": len(chunk),
                            # 您可以解析文件内容头部的 TITLE, URL 等信息，然后在此处添加
                            # 示例：'title': "P/E Ratio Definition" 
                        }
                    )
                    all_documents.append(doc)
                
                print(f"   -> {source_name} 分块完成，共 {len(chunks)} 个块。")

            except Exception as e:
                print(f"   ❌ 处理文件 {file_path} 失败: {e}")
        
        return all_documents

    def ingest_data_into_vector_store(self):
        """
        主函数：统一处理所有数据源并导入向量数据库。
        """
        print("--- 启动 RAG 知识库导入流程 ---")
        all_chunks = []

        for key in db.SOURCES_CONFIG.keys():
            print(f"处理数据源: {key.upper()}...")
            chunks = self.load_and_chunk_data(key)
            all_chunks.extend(chunks)

        if not all_chunks:
            print("警告：未找到任何可处理的文本块。请检查路径和文件。")
            return

        # 5. 嵌入 (Embedding) 并存储到向量数据库
        print(f"\n开始嵌入和存储 {len(all_chunks)} 个文本块...")
        # Chroma.add_documents 会自动调用 self.embeddings 将文本块转换为向量
        self.vector_store.add_documents(all_chunks)
        
        # 6. 持久化存储
        self.vector_store.persist()
        print(f"🎉 知识库构建完成。总计 {len(all_chunks)} 个块已存储。")
        
        # 

# --- 脚本执行示例 ---
if __name__ == "__main__":
    # 注意：在实际项目中，您应该确保所有清洗后的文件已存在于 DATA_BASE_DIR 下
    # 模拟创建目录以便代码运行
    for key in db.SOURCES_CONFIG.keys():
        os.makedirs(db.SOURCES_CONFIG[key]['path'], exist_ok=True)
    
    # 确保 data/kb 存在
    os.makedirs(os.path.join(os.getcwd(), 'data', 'kb'), exist_ok=True)
    
    # 假设：您已经将清洗后的 TXT 文件放入相应的目录中
    
    pipeline = DataPipelineService()
    pipeline.ingest_data_into_vector_store()