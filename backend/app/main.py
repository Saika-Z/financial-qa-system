
# backend/app/main.py

import uvicorn
from fastapi import APIRouter, FastAPI
from backend.app.api.endpoints.chat_endpoint import router as chat_router
from backend.app.api.endpoints.chatstream_endpoint import router as chatstream_router
from backend.app.services.tools.finance_client import FinanceClient
from backend.app.services.tools.ticker_tool import TickerExtractor
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi import FastAPI
from backend.app.services.inference import FinancialPredictor
from backend.app.services.rag_query_service_new import RAGQueryService
from backend.app.core.config import config
import torch
import gc
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@asynccontextmanager
async def lifespan(app: FastAPI):
    gc.collect()
    torch.cuda.empty_cache()
    # --- setup ---
    print(f"Loading Model: {config.LOCAL_BERT_PATH}...")
    # load model
    app.state.predictor = FinancialPredictor(config.LOCAL_BERT_PATH, config.BASE_MODE_NAME)
    
    app.state.rag_service = RAGQueryService()

    app.state.ticker_tool = TickerExtractor()

    app.state.finance_client = FinanceClient()

    print("✅ All models loaded.")

    
    yield  # this is where the app runs
    
    # --- shutdown ---
    print("Shutting down... releasing resources.")
    del app.state.predictor
    del app.state.rag_service
    del app.state.finance_client
    del app.state.ticker_tool

def create_app() -> FastAPI:
    app = FastAPI(
        title="Financial QA/Sentiment API",
        version="1.0.0",
        lifespan=lifespan
    )
    # development environment allows all origins, 
    # in production, you must specify the vue url
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  
        allow_credentials=True,
        # allowed HTTP methods（GET, POST, etc）
        allow_methods=["*"],  
        allow_headers=["*"],
    )
    base_router = APIRouter()
    @base_router.get("/")
    async def root():
        return {"message": "Server is running"}
    
    app.include_router(base_router, tags=["core"])

    # chat Api
    #app.include_router(chat_router, prefix="/chat", tags=["chat"]) # test

    app.include_router(chatstream_router, prefix="/api", tags=["chatstream"])

    return app

# Note: The command to run is no longer `python -m backend.app.main`, but `uvicorn`.
if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
