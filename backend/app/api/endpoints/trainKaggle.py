
# backend/app/api/endpoints/trainKaggle.py

from fastapi import APIRouter
from backend.app.services.inference import FinancialPredictor
from pydantic import BaseModel
import backend.src.config.settings as settings

router = APIRouter()

# Instantiate FinanceService
predictor = FinancialPredictor(settings.MODEL_SAVE_PATH, settings.MODEL_NAME)

class QuestionRequest(BaseModel):
    question: str



@router.post("/query")
async def sentiment_query(item: QuestionRequest):
    """
    using Kaggle data to answer sentiment-related questions about a specific stock ticker
    """
    print(f" Received Question: {item.question}")
    try:
        answer = predictor.predict(item.question)
        return {"Answer": answer}
    except Exception as e:
        return {"error": str(e)}
    

# if __name__ == "__main__":
#     print(sentiment_service.predict_sentiment("Apple's earnings report exceeded expectations."))