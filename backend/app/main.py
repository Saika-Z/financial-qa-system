
# backend/app/main.py

import uvicorn
from fastapi import APIRouter, FastAPI
from backend.app.api.endpoints.trainKaggle import router as sentiment_router
from backend.app.api.endpoints.finance import router as finance_router
from backend.app.api.endpoints.qa import router as qa_router
from backend.app.api.endpoints.chat import router as chat_router

from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi import FastAPI
from backend.app.services.inference import FinancialPredictor
from backend.app.services.rag_query_service import RAGQueryService
from backend.app.core.config import config

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 【启动时执行】 ---
    print(f"Loading Model: {config.LOCAL_BERT_PATH}...")
    # 将加载好的模型挂载到 app.state 中
    app.state.predictor = FinancialPredictor(config.LOCAL_BERT_PATH, config.BASE_MODE_NAME)
    
    app.state.rag_service = RAGQueryService()

    print("✅ All models loaded.")

    
    yield  # 这里是应用运行的时间点
    
    # --- 【关闭时执行】 ---
    print("Shutting down... releasing resources.")
    del app.state.predictor
    del app.state.rag_service

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

    # Stock Api
    app.include_router(finance_router, prefix="/finance", tags=["finance"])

    # sentiment Api
    app.include_router(sentiment_router, prefix="/sentiment", tags=["sentiment"])

    # QA Api
    app.include_router(qa_router, prefix="/qa", tags=["qa"])

    # chat Api
    app.include_router(chat_router, prefix="/chat", tags=["chat"])

    return app

# Note: The command to run is no longer `python -m backend.app.main`, but `uvicorn`.
if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
