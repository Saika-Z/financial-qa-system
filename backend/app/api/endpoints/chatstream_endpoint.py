# backend/app/api/endpoints/chatstream_endpoint.py

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import logging

logging.basicConfig(
    level=logging.DEBUG,  # set logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # set log message format
)

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    text: str
    sentiment: str
    intent: str
    data: dict = None
    
@router.post("/chat/stream")
async def chat_stream_endpoint(request: Request, chat_req: ChatRequest):
    user_input = chat_req.message

    rag_service = request.app.state.rag_service
    predictor = request.app.state.predictor

    ticker_tool = request.app.state.ticker_tool
    finance_client = request.app.state.finance_client
    
    # logic：intent ->  rag_service.query_stream -> return StreamingResponse
    prediction = await run_in_threadpool(predictor.predict, user_input)
    intent = prediction["intent"]
    sentiment = prediction["sentiment"]

    ticker = "UNKNOWN"
    if "FINANCE" in intent:
        ticker = await run_in_threadpool(ticker_tool.extract_ticker, user_input)

    async def event_generator():
        import json
        meta_data = json.dumps({"intent": intent, "sentiment": sentiment})
        yield f"event: metadata\ndata: {meta_data}\n\n"

        if intent == "FINANCE" and ticker != "UNKNOWN":

            stock_res = await run_in_threadpool(finance_client.get_price, ticker)
            price_data = json.dumps(stock_res)
            yield f"event: finance\ndata: {price_data}\n\n"
        else:
            async for chunk in rag_service.query_stream(user_input):
                yield f"event: message\ndata: {chunk}\n\n"

        yield "event: end\ndata: done\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
# if __name__ == "__main__":
#     print(finance_service.get_stock_data("AAPL"))