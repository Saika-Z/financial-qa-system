
import os
from huggingface_hub import HfApi

REPO_ID = "Saika-Zh/bert-kb-fin-qa"
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','models'))

def upload_model():
    api = HfApi()

    print(f"正在从本地目录 {MODEL_DIR} 上传文件到 {REPO_ID}...")
    
    api.upload_folder(
        folder_path=MODEL_DIR, # 使用绝对路径更稳妥
        repo_id=REPO_ID,
        repo_type="model",
        # 排除脚本本身和不必要的文件
        ignore_patterns=[
            os.path.basename(__file__), # 动态排除脚本名
            ".git*", 
            "__pycache__/*", 
            "*.DS_Store",
            "upload_to_hf.py"
        ], 
    )
    
    print("✅ 上传完成！")

if __name__ == "__main__":
    upload_model()