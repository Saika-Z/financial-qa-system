
# backend/app/core/config.py 示例

import os
import pathlib

# the absolute path of the config.py
CURRENT_FILE_PATH = pathlib.Path(__file__).resolve()
# Jump up two levels.，从 app -> backend -> project_root
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


class Config:
    # --- Service Configuration ---
    API_V1_STR: str = "/api/v1"
    # ... Other service-related configurations

    # --- Model Configuration ---
    # Define the absolute path for saving the model\
    # \financial-qa-system\models\
    MODEL_PATH: str = os.path.join(PROJECT_PATH, 'models' )
    print(f"DEBUG: Model Path attempting to load: {MODEL_PATH}")

    BERT_PATH_NAME = "sentiment_intention_bert"
    BASE_MODE_NAME = "bert-base-multilingual-cased"

    LOCAL_BERT_PATH = os.path.join(MODEL_PATH, BERT_PATH_NAME)
    
    # Because the tokenizer and the model are saved in the same path, the paths are identical.
    TOKENIZER_PATH: str = LOCAL_BERT_PATH

    # fiance_service.py tag to contol the development mod, if it is False, it will use yfinance
    DEV_MODE: bool = True

    TICKER_MAP = {
            "苹果": "AAPL", "apple": "AAPL",
            "特斯拉": "TSLA", "tesla": "TSLA",
            "英伟达": "NVDA", "nvidia": "NVDA",
            "微软": "MSFT", "microsoft": "MSFT",
            "谷歌": "GOOGL", "google": "GOOGL",
            "亚马逊": "AMZN", "amazon": "AMZN",
            "阿里巴巴": "BABA", "alibaba": "BABA",
            "腾讯": "700.HK", "tencent": "700.HK"
        }
    
config = Config()