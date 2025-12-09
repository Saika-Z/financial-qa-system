'''
 # Author: Wenqing Zhao
 # Date: 2025-12-09 11:06:03
 # LastEditTime: 2025-12-09 11:13:57
 # Description: 
 # FilePath: /financial-qa-system/backend/app/services/finance_service.py
'''
import yfinance as yf

class FinanceService:
    def __init__(self):
        pass

    def get_stock_data(self, ticker: str, period: str = "1d", interval: str = "1m"):
        """
        get real-time stock price data using yfinance
        """
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        return data
    
    def get_stock_info(self, ticker: str):
        """
        get stock information
        """
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    
    def get_news(self, ticker: str):
        """
        get latest news related to the stock
        """
        stock = yf.Ticker(ticker)
        news = stock.news
        return news