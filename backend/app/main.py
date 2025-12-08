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

# --- 1. 定义请求体数据结构 ---
# 在 main.py 或单独的 schemas.py 中定义
class SentimentRequest(BaseModel):
    """定义POST请求体的数据结构"""
    text: str

# --- 2. 创建 API 路由 ---
api_router = APIRouter()

@api_router.post("/predict", tags=["Sentiment"])
def predict_sentiment_endpoint(request_data: SentimentRequest):
    """接收文本并返回预测的情感结果"""
    
    # 获取服务实例 (第一次调用时加载模型)
    sentiment_service = get_sentiment_service()
    
    # 调用核心预测方法
    predicted_sentiment = sentiment_service.predict_sentiment(request_data.text)
    
    return {"text": request_data.text, "sentiment": predicted_sentiment}


# --- 3. 初始化 FastAPI 应用 ---
def create_app():
    # 尝试加载服务，如果失败则抛出错误并退出
    try:
        # 预先加载模型，确保服务启动时模型已就绪
        get_sentiment_service() 
        print("✅ Sentiment Service loaded successfully.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Failed to load Sentiment Service: {e}")
        # 在实际部署中，可能需要更优雅地处理或直接 raise 阻止服务启动
        raise RuntimeError("Service initialization failed.") from e

    app = FastAPI(
        title="Financial QA/Sentiment API",
        version="1.0.0",
    )
    
    # 包含您的路由
    app.include_router(api_router)
    return app

app = create_app()

# 注意：运行命令不再是 python -m backend.app.main，而是 uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# # uvicorn backend.app.main:api_router --host 0.0.0.0 --port 8000 --reload
# curl -X POST "http://localhost:8000/predict" \
# -H "Content-Type: application/json" \
# -d '{"text": "Apple earnings exceeded all market expectations and the stock price soared."}'