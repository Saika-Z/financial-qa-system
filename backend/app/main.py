'''
 # Author: Wenqing Zhao
 # Date: 2025-12-05 19:37:13
 # LastEditTime: 2025-12-08 22:38:36
 # Description: 
 # FilePath: /financial-qa-system/backend/app/main.py
'''
from fastapi import FastAPI
from backend.app.services.sentiment_service import get_sentiment_service
from backend.app.core.config import settings
from fastapi import APIRouter
from pydantic import BaseModel

# --- 1. Define the request body data structure. ---
# Define it in main.py or a separate schemas.py file.
class SentimentRequest(BaseModel):
    """定义POST请求体的数据结构"""
    text: str

# --- 2. Create API routes ---
api_router = APIRouter()

@api_router.post("/predict", tags=["Sentiment"])
def predict_sentiment_endpoint(request_data: SentimentRequest):
    """Receives text and returns the predicted sentiment result."""
    
    # Obtain a service instance (the model is loaded on the first call).
    sentiment_service = get_sentiment_service()
    
    # Calling the core prediction method
    predicted_sentiment = sentiment_service.predict_sentiment(request_data.text)
    
    return {"text": request_data.text, "sentiment": predicted_sentiment}


# --- 3. Initialize the FastAPI application. ---
def create_app():
    # Attempt to load the service; if it fails, throw an error and exit.
    try:
        # Pre-load the model to ensure it is ready when the service starts.
        get_sentiment_service() 
        print("✅ Sentiment Service loaded successfully.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Failed to load Sentiment Service: {e}")
        # In a real-world deployment, it might be necessary to handle this more gracefully or directly raise an error to prevent the service from starting.
        raise RuntimeError("Service initialization failed.") from e

    app = FastAPI(
        title="Financial QA/Sentiment API",
        version="1.0.0",
    )
    
    # Includes your route
    app.include_router(api_router)
    return app

app = create_app()

# Note: The command to run is no longer `python -m backend.app.main`, but `uvicorn`.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# # uvicorn backend.app.main:api_router --host 0.0.0.0 --port 8000 --reload
# curl -X POST "http://localhost:8000/predict" \
# -H "Content-Type: application/json" \
# -d '{"text": "Apple earnings exceeded all market expectations and the stock price soared."}'