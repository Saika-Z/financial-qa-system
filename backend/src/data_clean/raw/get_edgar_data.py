
from sec_edgar_downloader import Downloader
import os

# ----------------------------------------------------
# exchange this with your own information
COMPANY_NAME = "Data Science Project" # example: "Data Science Project"
EMAIL_ADDRESS = "vincent0814cn@outlook.com" # example: "vincent0814cn@outlook.com"
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