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

# 实例化 FinanceService
finance_service = FinanceService()

@router.get("/stock/{ticker}/data")
async def get_stock_data(ticker: str, period: str = "1d", interval: str = "1m"):
    """
    获取某只股票的实时数据
    """
    data = finance_service.get_stock_data(ticker, period, interval)
    return data.to_dict()  # 将数据转换为字典返回

@router.get("/stock/{ticker}/info")
async def get_stock_info(ticker: str):
    """
    获取某只股票的基本信息
    """
    info = finance_service.get_stock_info(ticker)
    return info

@router.get("/stock/{ticker}/news")
async def get_stock_news(ticker: str):
    """
    获取某只股票的新闻
    """
    news = finance_service.get_news(ticker)
    return news
