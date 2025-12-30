
# backend/app/api/v1/endpoints/finance.py

from fastapi import APIRouter, HTTPException
from backend.app.services.finance_service import FinanceService
from backend.app.core.config import config

router = APIRouter()

# local test set env=development to avoid rate limits


@router.get("/stock/{ticker}/all")
async def get_stock_all_in_one(ticker: str):
    """
    Get all data for a specific stock in 1 request: price, change, history.
    """
    data = finance_service.get_stock_data(ticker)

    if not data:
        raise HTTPException(status_code=500, detail="Internal Server Error: No data received")
    
    if isinstance(data, dict) and "error" in data:
        raise HTTPException(status_code=429, detail=data["error"])
    
    return data

# if __name__ == "__main__":
#     print(finance_service.get_stock_data("AAPL"))