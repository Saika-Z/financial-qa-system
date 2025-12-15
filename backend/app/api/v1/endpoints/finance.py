'''
 # Author: Wenqing Zhao
 # Date: 2025-12-09 11:10:52
 # LastEditTime: 2025-12-09 11:14:09
 # Description: 
 # FilePath: /financial-qa-system/backend/app/api/v1/endpoints/finance.py
'''
# backend/app/api/v1/endpoints/finance.py

from fastapi import APIRouter
from backend.app.services.finance_service import FinanceService

router = APIRouter()

# Instantiate FinanceService
finance_service = FinanceService()

@router.get("/stock/{ticker}/data")
async def get_stock_data(ticker: str, period: str = "1d", interval: str = "1m"):
    """
    Get real-time data for a specific stock.
    """
    data = finance_service.get_stock_data(ticker, period, interval)
    return data.to_dict()  # 将数据转换为字典返回

@router.get("/stock/{ticker}/info")
async def get_stock_info(ticker: str):
    """
    Get basic information about a particular stock.
    """
    info = finance_service.get_stock_info(ticker)
    return info

@router.get("/stock/{ticker}/news")
async def get_stock_news(ticker: str):
    """
    Get news about a specific stock.
    """
    news = finance_service.get_news(ticker)
    return news
