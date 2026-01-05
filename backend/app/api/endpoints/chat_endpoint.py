# backend/app/api/endpoints/chat_endpoint.py

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

@router.post("/chat")
async def chat_endpoint(request: Request, chat_req: ChatRequest):
    import time
    t_start = time.time()

    predictor = request.app.state.predictor
    rag_service = request.app.state.rag_service

    t_intent = time.time()
    logging.info(f">>> [时间分析] 意图识别结束: {t_intent - t_start:.2f}s")
    
    user_input = chat_req.message

    if not user_input:
        raise HTTPException(status_code=400, detail="Empty message")

    try:
        # --- 1. Bert intention ---
        pred = await run_in_threadpool(predictor.predict, user_input)
        intent = pred["intent"]
        sentiment = pred["sentiment"]

        data_result = {}
        final_text = ""
        
        logging.debug(f" intent: {intent}")
        logging.debug(f" sentiment: {sentiment}")   
        
        # --- 2. According to intent, execute service ---
        if "FINANCE" in intent:
            # ticker
            ticker = predictor.extract_ticker(user_input)
            
            if ticker == "UNKNOWN" or not ticker:
                try:
                    data_result = await predictor.get_stock_data(ticker)
                    prefix = "📈 市场信心十足：" if "Positive" in sentiment else "⚠️ 市场情绪谨慎："
                    final_text = f"{prefix}为您查到 {ticker} 的最新报价为 {data_result['price']}。"
                except Exception as e:
                    final_text = f"Sorry, there was an error retrieving data for {ticker}. Please try again later."
                

        elif "SENTIMENT" in intent:
            final_text = f"according to your question: {sentiment}"
            data_result = {"confidence": pred.get("sent_confidence")}

        elif "RAG" in intent:
            logging.info(f">>> [时间分析] RAG 逻辑开始: {t_intent - t_start:.2f}s")

            rag_data = await run_in_threadpool(rag_service.query_stream, user_input)

            t_rag = time.time()
            
            logging.info(f">>> [时间分析] RAG 逻辑结束: {t_rag - t_start:.2f}s")
            
            final_text = rag_data
            logging.info(f">>> [时间分析] RAG 结果类型: {type(rag_data)}")    
            logging.info(f">>> [时间分析] 总耗时: {t_rag - t_start:.2f}s")
        
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
    
@router.post("/chat/stream")
async def chat_stream_endpoint(request: Request, chat_req: ChatRequest):
    rag_service = request.app.state.rag_service
    user_input = chat_req.message
    
    # 逻辑：识别意图 -> 调用 rag_service.query_stream -> 返回 StreamingResponse
    return StreamingResponse(
        rag_service.query_stream(user_input),
        media_type="text/event-stream"
    )

# if __name__ == "__main__":
#     print(finance_service.get_stock_data("AAPL"))