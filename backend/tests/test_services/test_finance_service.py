'''
 # Author: Wenqing Zhao
 # Date: 2025-12-09 11:14:50
 # LastEditTime: 2025-12-09 11:14:55
 # Description: 
 # FilePath: /financial-qa-system/backend/tests/test_services/test_finance_service.py
'''
# backend/tests/test_services/test_finance_service.py

from backend.app.services.finance_service import FinanceService

def test_get_stock_data():
    finance_service = FinanceService()
    data = finance_service.get_stock_data("AAPL", period="1d", interval="5m")
    assert data is not None
    assert "Open" in data.columns

def test_get_stock_info():
    finance_service = FinanceService()
    info = finance_service.get_stock_info("AAPL")
    assert info is not None
    assert "symbol" in info
