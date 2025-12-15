'''
Author: Zhao
Date: 2025-12-08 18:03:48
LastEditors: Please set LastEditors
LastEditTime: 2025-12-15 12:55:26
FilePath: qa.py
Description: 

'''
from fastapi import APIRouter
from backend.app.services.finance_service import FinanceService

router = APIRouter()

# Instantiate FinanceService
finance_service = FinanceService()

@router.get("/rag/query")
async def rag_query(ticker: str):
    """
    using RAG to answer finance-related questions about a specific stock ticker
    """
    try:
        answer = finance_service.answer_question_with_rag(ticker)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}

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
