'''
 # Author: Wenqing Zhao
 # Date: 2025-12-09 11:14:24
 # LastEditTime: 2025-12-09 11:14:28
 # Description: 
 # FilePath: /financial-qa-system/backend/app/api/v1/endpoints/1.py
'''
# backend/app/api/v1/endpoints/__init__.py

from fastapi import APIRouter
from .finance import router as finance_router

router = APIRouter()

router.include_router(finance_router, prefix="/finance", tags=["finance"])
