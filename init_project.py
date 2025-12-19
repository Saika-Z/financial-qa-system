'''
Author: Zhao
Date: 2025-12-17 23:12:59
LastEditors: Please set LastEditors
LastEditTime: 2025-12-19 22:39:07
FilePath: init_project.py
Description: 

'''
import os
from huggingface_hub import snapshot_download

def init_project():
    # --- 1. Dynamically get the root directory of the current script ---
    # __file__ is the path of the current script
    # os.path.dirname(os.path.abspath(__file__)) gets the absolute path of financial-qa-system
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # --- 2. Define relative paths based on the root directory ---
    # This will automatically concatenate the correct path regardless of whether your project is on D drive or in Linux's /home
    MODEL_DIR = os.path.join(BASE_DIR,"backend", "models", "sentiment_model")
    KB_DIR = os.path.join(BASE_DIR, "backend", "data")

    # Define Hugging Face repositories (keep as is, this is the remote URL)
    MODEL_REPO = "Saika-Zh/bert-finance-qa-model" 
    DATA_REPO = "Saika-Zh/financial-qa-data"


    # --- 3. Automatically create directories ---
    for path in [MODEL_DIR, KB_DIR]:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Directory ready: {path}")

    # --- 4. Execute download (example: downloading the model) ---
    try:
        print(f"Synchronizing model to: {MODEL_DIR}")
        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=MODEL_DIR,
            repo_type="model",
            local_dir_use_symlinks=False # It's recommended to turn off symbolic links and directly download files to this directory
        )
        
        print(f"Synchronizing knowledge base to: {KB_DIR}")
        snapshot_download(
            repo_id=DATA_REPO,
            local_dir=KB_DIR,
            repo_type="dataset",
            local_dir_use_symlinks=False
        )
        print("✅ Assets synchronization completed!")
    except Exception as e:
        print(f"❌ Synchronization failed: {e}")

if __name__ == "__main__":
    init_project()
