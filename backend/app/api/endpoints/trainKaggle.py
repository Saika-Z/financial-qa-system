
# backend/app/api/endpoints/trainKaggle.py

from fastapi import APIRouter
from backend.app.services.sentiment_service import get_sentiment_service
from pydantic import BaseModel

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str

# Instantiate FinanceService
sentiment_service = get_sentiment_service()

@router.post("/query")
async def sentiment_query(item: QuestionRequest):
    """
    using Kaggle data to answer sentiment-related questions about a specific stock ticker
    """
    print(f" Received Question: {item.question}")
    try:
        answer = sentiment_service.predict_sentiment(item.question)
        return {"Answer": answer}
    except Exception as e:
        return {"error": str(e)}
    

# if __name__ == "__main__":
#     print(sentiment_service.predict_sentiment("Apple's earnings report exceeded expectations."))