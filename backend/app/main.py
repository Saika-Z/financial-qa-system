# backend/app/main.py

import uvicorn
from fastapi import APIRouter, FastAPI
from backend.app.api.endpoints.trainKaggle import router as sentiment_router
from backend.app.api.endpoints.finance import router as finance_router

from fastapi.middleware.cors import CORSMiddleware



def create_app() -> FastAPI:
    app = FastAPI(
        title="Financial QA/Sentiment API",
        version="1.0.0",
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

    return app

# Note: The command to run is no longer `python -m backend.app.main`, but `uvicorn`.
if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
