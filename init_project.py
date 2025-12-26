import os
import sys
import subprocess
import platform
from huggingface_hub import snapshot_download

def run_command(command):
    """ execute command and print result """
    try:
        subprocess.check_call([sys.executable, "-m"] + command)
    except subprocess.CalledProcessError as e:
        print(f"❌ command failed: {e}")

def install_dependencies():
    """ Automated dependency installation logic """
    print("📦 Step 1: checking env and installing dependencies ...")

    # 1. install basic dependencies
    run_command(["pip", "install", "-r", "requirements.txt"])

    # System-specific PyTorch installation
    system = platform.system()
    if system == "Darwin":
        print("🍎 Mac OS detected, installing MPS-optimized Torch...")
        run_command(["pip", "install", "torch", "torchvision", "torchaudio"])
    elif system == "Linux" or system == "Windows":
        print("💻 PC system (Linux/Windows) detected, attempting to install CUDA 12.1 optimized Torch...")
        # Default to installation with CUDA support
        run_command(["pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"])

def download_assets():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    MODELS_ROOT = os.path.join(BASE_DIR, "models")
    SENTIMENT_DIR = os.path.join(MODELS_ROOT, "sentiment_model")
    RERANKER_DIR = os.path.join(MODELS_ROOT, "bge-reranker-v2-m3")
    KB_DIR = os.path.join(BASE_DIR, "data")

    SENTIMENT_REPO = "Saika-Zh/bert-finance-qa-model" 
    DATA_REPO = "Saika-Zh/financial-qa-data"
    RERANKER_REPO = "BAAI/bge-reranker-v2-m3"

    assets = [
        {"name": "Sentiment Analysis Model", "repo": SENTIMENT_REPO, "path": SENTIMENT_DIR, "type": "model"},
        {"name": "Reranker Model", "repo": RERANKER_REPO, "path": RERANKER_DIR, "type": "model"},
        {"name": "Knowledge Base Data", "repo": DATA_REPO, "path": KB_DIR, "type": "dataset"},
    ]

    print("🚀 Step 2: Syncing models and assets (this may take a while)...")

    for asset in assets:
        try:
            print(f"Syncing {asset['name']}...")
            snapshot_download(
                repo_id=asset['repo'],
                local_dir=asset['path'],
                repo_type=asset['type'],
                local_dir_use_symlinks=False,
                ignore_patterns=["*.msgpack", "*.h5"] # Optimization: ignore unnecessary formats to save space
            )
        except Exception as e:
            print(f"❌ Failed to sync {asset['name']}: {e}")

def init_project():
    # execute in order：env -> dependencies -> assets
    install_dependencies()
    download_assets()
    print("\n✨ project init success! Now can run run.py to start the server")

if __name__ == "__main__":
    init_project()