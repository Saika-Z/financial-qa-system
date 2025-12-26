
import os
# --- Configuration ---
DATA_BASE_DIR = os.path.join(os.getcwd(),'data', 'processed')
VECTOR_DB_DIR = os.path.join(os.getcwd(), 'data', 'kb', 'finance_vector_db')

# Use an efficient and commonly used Sentence Transformer model for embedding.
#EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# testing with a larger LLM model
#LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Reranker model
RERANKER_MODEL_NAME="BAAI/bge-reranker-v2-m3"

# --- Data source configuration (defining different partitioning strategies) ---
SOURCES_CONFIG = {
    'concept': {
        'path': os.path.join(DATA_BASE_DIR, 'investopedia'),
        'chunk_size': 400, # Ignore this value, as the JSON file is pre-chunked.
        'chunk_overlap': 50,
        'doc_type': 'Investopedia_Concept',
        'format': 'json'  # <-- Indicates that the file has been preprocessed and stored as JSON blocks.
    },
    'report': {
        'path': os.path.join(DATA_BASE_DIR, 'company_reports'),
        'chunk_size': 1500,
        'chunk_overlap': 400,
        'doc_type': 'SEC_MDA_Report',
        'format': 'txt'   # <-- Indicates that the file needs to be processed using a Text Splitter.
    },
    'news': {
        'path': os.path.join(DATA_BASE_DIR, 'company_history_news'),
        'chunk_size': 600,
        'chunk_overlap': 100,
        'doc_type': 'Historical_News_Event',
        'format': 'txt'
    }
}