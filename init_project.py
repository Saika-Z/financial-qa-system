import os
import sys
import subprocess
import platform
import shutil
from huggingface_hub import snapshot_download

def run_command(command):
    """ execute command and print result """
    try:
        full_command = [sys.executable, "-m"] + command if is_python else command
        print(f"executing: {' '.join(full_command)}")
        subprocess.check_call(full_command, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"❌ command failed: {e}")


def install_python_dependencies():
    print("\n 📦 Step 1: installing python dependencies ...")
    run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    system = platform.system()
    if system == "Darwin":
        print("🍎 Mac OS detected, installing MPS-optimized Torch...")
        run_command(["pip", "install", "torch", "torchvision", "torchaudio"], is_python=True)
    else:
        print("💻 PC system (Linux/Windows) detected, attempting to install CUDA 12.1 optimized Torch...")
        run_command([
            "pip", "install", "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ], is_python=True)

def install_frontend_assets():
    """ Automated dependency installation logic """
    print("📦 Step 2: and installing frontend dependencies ...")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

    if not os.path.exists(FRONTEND_DIR):
        print(f"⚠️ cannot find frontend : {FRONTEND_DIR}，skipping。")
        return
    
    npm_path = shutil.which("npm")
    if not npm_path:
        print("❌ Error: cannot find npm。Please install Node.js before proceeding。")
        return
    
    print(f"📥 npm installing (path: {FRONTEND_DIR})...")
    run_command([npm_path, "install"], cwd=FRONTEND_DIR)

    # 2. 确保 vue 和 markdown-it 存在 (防止 package.json 没写)
    print("📥 making sur vue and markdown-it status ...")
    run_command([npm_path, "install", "vue", "markdown-it"], cwd=FRONTEND_DIR)

def download_hf_assets():
    print("\n🚀 Step 3: downloading model and data from Hugging Face (this may take a while) ...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    MODELS_ROOT = os.path.join(BASE_DIR, "models")

    assets = [
        {
            "name": "Sentiment Model", 
            "repo": "Saika-Zh/bert-kb-fin-qa",
            "path": os.path.join(MODELS_ROOT, "sentiment_model"), 
            "type": "model"
        },
        {
            "name": "Reranker Model", 
            "repo": "BAAI/bge-reranker-v2-m3", 
            "path": os.path.join(MODELS_ROOT, "bge-reranker-v2-m3"), 
            "type": "model"
        },
        {
            "name": "Knowledge Base Data", 
            "repo": "Saika-Zh/financial-qa-data", 
            "path": os.path.join(BASE_DIR, "data"), 
            "type": "dataset"
        },
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
    print("="*30)
    print("=== Project Init ===")
    install_python_dependencies()
    install_frontend_assets()
    download_hf_assets()
    print("\n" + "="*30)
    print("\n✨ project init success! Now can run run.py to start the server")
    print("\n Run 'python run.py' to start the server")
    print("\n" + "="*30)

if __name__ == "__main__":
    init_project()