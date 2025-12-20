'''
Author: Zhao
Date: 2025-12-08 18:03:48
LastEditors: Please set LastEditors
LastEditTime: 2025-12-20 18:43:58
FilePath: config.py
Description: 

'''
# backend/app/core/config.py 示例

import os
import pathlib

# the absolute path of the config.py
CURRENT_FILE_PATH = pathlib.Path(__file__).resolve()
# Jump up two levels.，从 app -> backend -> project_root
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


class Settings:
    # --- Service Configuration ---
    API_V1_STR: str = "/api/v1"
    # ... Other service-related configurations

    # --- Model Configuration ---
    # Define the relative path for saving the model, and combine it with PROJECT_ROOT to obtain the absolute path.
    MODEL_SUBDIR = 'sentiment_model'

    # Define the absolute path for saving the model\
    # \financial-qa-system\models\sentiment_model
    MODEL_PATH: str = os.path.join(PROJECT_PATH, 'models' ,MODEL_SUBDIR)
    print(f"DEBUG: Model Path attempting to load: {MODEL_PATH}")
    
    # Because the tokenizer and the model are saved in the same path, the paths are identical.
    TOKENIZER_PATH: str = MODEL_PATH
    # Print the path for debugging.
    #print(f"DEBUG: Calculated PROJECT_ROOT: {PROJECT_ROOT}")
    #print(f"DEBUG: Model Path attempting to load: {MODEL_PATH}")

    # fiance_service.py tag to contol the development mod, if it is False, it will use yfinance
    DEV_MODE: bool = True
    
settings = Settings()