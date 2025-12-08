'''
 # Author: Wenqing Zhao
 # Date: 2025-12-05 19:37:13
 # LastEditTime: 2025-12-05 19:56:03
 # Description: 
 # FilePath: /financial-qa-system/backend/app/main.py
'''
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}