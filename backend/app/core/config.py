# backend/app/core/config.py 示例

import os
import pathlib

# 获取项目根目录 (假设运行在 backend/app/ 或其子目录)
CURRENT_FILE_PATH = pathlib.Path(__file__).resolve()
# 向上跳两级，从 app -> backend -> project_root
#PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
PROJECT_ROOT = CURRENT_FILE_PATH.parents[2]

class Settings:
    # --- 服务配置 ---
    API_V1_STR: str = "/api/v1"
    # ... 其他服务相关的配置

    # --- 模型配置 ---
    # 定义模型保存的相对路径，并结合 PROJECT_ROOT 得到绝对路径
    MODEL_SUBDIR = 'models/sentiment_model'
    #MODEL_PATH: str = os.path.join(PROJECT_ROOT, MODEL_SUBDIR)
    MODEL_PATH: str = str(PROJECT_ROOT / MODEL_SUBDIR)
    
    # 因为 tokenizer 和 model 保存在同一路径，所以路径相同
    TOKENIZER_PATH: str = MODEL_PATH
    # 打印路径进行调试
    print(f"DEBUG: Calculated PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DEBUG: Model Path attempting to load: {MODEL_PATH}")
    
settings = Settings()