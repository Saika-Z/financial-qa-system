'''
 # Author: Wenqing Zhao
 # Date: 2025-12-10 10:20:25
 # LastEditTime: 2025-12-10 10:32:53
 # Description: 
 # FilePath: /financial-qa-system/backend/data/raw/getEdgarData.py
'''
from sec_edgar_downloader import Downloader
import os

# ----------------------------------------------------
# 替换成您自己的信息。这些信息仅用于遵守 SEC 规定进行用户代理识别。
COMPANY_NAME = "Data Science Project" # 例如: "Data Science Project"
EMAIL_ADDRESS = "vincent0814cn@outlook.com" # 例如: "jane.doe@personal.com"
# ----------------------------------------------------


download_dir = "backend/data/raw/company_reports/"

if not os.path.exists(download_dir):
    os.makedirs(download_dir)


dl = Downloader(
    company_name=COMPANY_NAME,
    email_address=EMAIL_ADDRESS,
    download_folder = download_dir
    )
    

try:
    # download the latest 10-K report for Apple Inc. (AAPL)
    dl.get(
        "10-K", # form type
        "AAPL", # Apple Inc. ticker symbol
        limit=1 # only download the latest report
        )
    print("Successfully downloaded the latest 10-K report for AAPL. at {download_dir} directory")
except Exception as e:
    print(f"An error occurred while downloading: {e}")