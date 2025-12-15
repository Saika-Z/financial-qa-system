# backend/app/core/config.py 示例

import os
import pathlib

# Get the project root directory (assuming it's running in backend/app/ or one of its subdirectories).
CURRENT_FILE_PATH = pathlib.Path(__file__).resolve()
# Jump up two levels.，从 app -> backend -> project_root
#PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
PROJECT_ROOT = CURRENT_FILE_PATH.parents[2]

class Settings:
    # --- Service Configuration ---
    API_V1_STR: str = "/api/v1"
    # ... Other service-related configurations

    # --- Model Configuration ---
    # Define the relative path for saving the model, and combine it with PROJECT_ROOT to obtain the absolute path.
    MODEL_SUBDIR = 'models/sentiment_model'
    #MODEL_PATH: str = os.path.join(PROJECT_ROOT, MODEL_SUBDIR)
    MODEL_PATH: str = str(PROJECT_ROOT / MODEL_SUBDIR)
    
    # Because the tokenizer and the model are saved in the same path, the paths are identical.
    TOKENIZER_PATH: str = MODEL_PATH
    # Print the path for debugging.
    print(f"DEBUG: Calculated PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DEBUG: Model Path attempting to load: {MODEL_PATH}")
    
settings = Settings()