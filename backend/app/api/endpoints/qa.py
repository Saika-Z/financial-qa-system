
# backend/app/api/endpoints/qa.py

from fastapi import APIRouter
from backend.app.services.sentiment_service import SentimentService

router = APIRouter()

# Instantiate FinanceService
sentiment_service = SentimentService()

@router.get("/rag/query")
async def rag_query(ticker: str):
    """
    using RAG to answer finance-related questions about a specific stock ticker
    """
    try:
        answer = sentiment_service.answer_question_with_rag(ticker)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
    

if __name__ == "__main__":
    print(sentiment_service.answer_question_with_rag("AAPL"))