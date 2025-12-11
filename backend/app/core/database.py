'''
 # Author: Wenqing Zhao
 # Date: 2025-12-05 19:36:38
 # LastEditTime: 2025-12-11 14:09:47
 # Description: 
 # FilePath: /financial-qa-system/backend/app/core/database.py
'''
import os
# --- 配置 ---
DATA_BASE_DIR = os.path.join(os.getcwd(), 'data', 'processed')
VECTOR_DB_DIR = os.path.join(os.getcwd(), 'data', 'kb', 'finance_vector_db')
# 使用一个高效且常用的 Sentence Transformer 模型进行嵌入
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 

# --- 数据源配置 (定义不同的分块策略) ---
SOURCES_CONFIG = {
    'concept': {
        'path': os.path.join(DATA_BASE_DIR, 'clean_concepts'),
        'chunk_size': 400,
        'chunk_overlap': 50,
        'doc_type': 'Investopedia_Concept'
    },
    'report': {
        'path': os.path.join(DATA_BASE_DIR, 'clean_reports'),
        'chunk_size': 1500,
        'chunk_overlap': 200,
        'doc_type': 'SEC_MDA_Report'
    },
    'news': {
        'path': os.path.join(DATA_BASE_DIR, 'clean_news'),
        'chunk_size': 600,
        'chunk_overlap': 100,
        'doc_type': 'Historical_News_Event'
    }
}