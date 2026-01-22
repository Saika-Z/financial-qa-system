import os
from huggingface_hub import HfApi

REPO_ID = "Saika-Zh/financial-qa-data"
#current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))

def upload_data():
    api = HfApi()

    print(f"正在从本地目录 {DATA_DIR} 上传文件到 {REPO_ID}...")
    
    api.upload_folder(
        folder_path=DATA_DIR, # 使用绝对路径更稳妥
        repo_id=REPO_ID,
        repo_type="dataset",
        delete_patterns="*",
        # 排除脚本本身和不必要的文件
        ignore_patterns=[
            os.path.basename(__file__), # 动态排除脚本名
            ".git*", 
            "__pycache__/*", 
            "*.DS_Store",
            "upload_data_hf.py",
            ".cache/*",
            "*.log",
            "tmp/*"
        ], 
    )
    
    print("✅ 上传完成！")

if __name__ == "__main__":
    upload_data()