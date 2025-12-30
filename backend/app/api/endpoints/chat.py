

# backend/app/api/endpoints/chat.py

from fastapi import APIRouter, HTTPException, Request
from backend.app.core.config import config
from backend.app.services.rag_query_service import RAGQueryService
from backend.app.services.sentiment_service import get_sentiment_service
from backend.app.services.finance_service import FinanceService
from pydantic import BaseModel

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    text: str
    sentiment: str
    intent: str
    data: dict = None

@router.post("/chat")
async def chat_endpoint(request: Request, chat_req: ChatRequest):
    predictor = request.app.state.predictor
    rag_service = request.app.state.rag_service
    
    user_input = chat_req.message

    if not user_input:
        raise HTTPException(status_code=400, detail="Empty message")

    try:
        # --- 1. Bert intention ---
        pred = predictor.predict(user_input)
        intent = pred["intent"]
        sentiment = pred["sentiment"]

        data_result = {}
        final_text = ""
        
        # --- 2. According to intent, execute service ---
        if "FINANCE" in intent:
            # ticker
            ticker = predictor.extract_ticker(user_input)
            data_result = await predictor.get_stock_data(ticker)

            if ticker == "UNKNOWN":
                final_text = " Sorry we can't find this ticker. "

            else:
                prefix = "📈 市场信心十足：" if "Positive" in sentiment else "⚠️ 市场情绪谨慎："
                final_text = f"{prefix}为您查到 {ticker} 的最新报价为 {data_result['price']}。"

        elif "SENTIMENT" in intent:
            final_text = f"according to your question: {sentiment}"
            data_result = {"confidence": pred.get("sent_confidence")}

        elif "RAG" in intent:
            
            final_text = " Start searching from database ..."
            rag_data = await rag_service.query(user_input)
            data_result = {"source": "VectorDB", "content": rag_data}

        else:
            final_text = " Sorry, I don't understand your question."

        # --- 3. Combine response ---
        return ChatResponse(
            text=final_text,
            sentiment=sentiment,
            intent=intent,
            data=data_result
        )

    except Exception as e:
        print(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     print(finance_service.get_stock_data("AAPL"))