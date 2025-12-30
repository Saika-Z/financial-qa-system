
# backend/app/api/endpoints/qa.py

from fastapi import APIRouter
from backend.app.services.rag_query_service import RAGQueryService
from pydantic import BaseModel

router = APIRouter()

# Instantiate FinanceService
rag_service = RAGQueryService()

@router.post("/rag/query")
async def rag_query(ticker: str):
    """
    using RAG to answer finance-related questions about a specific stock ticker
    """
    try:
        answer = rag_service.query(ticker)
        return answer
    except Exception as e:
        return {"error": str(e)}
    

# if __name__ == "__main__":
#     question = "根据财报和最近的新闻，苹果的财务领导层和 AI 团队分别有什么最新的人事变动？"
#     try:
#         result = rag_service.query(question)
#         print(f"查询成功: {result}")
#     except Exception as e:
#         print(f"查询失败: {e}")