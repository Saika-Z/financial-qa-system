import yfinance as yf
from functools import lru_cache
import time
import pandas as pd
import backend.app.core.config as config
from rapidfuzz import process, fuzz

class FinanceService:

    def __init__(self, dev_mode=False):
        # if True will use mock data not from yfinance
        self.dev_mode = dev_mode
        self.ticker_map = config.TICKER_MAP

    @lru_cache(maxsize=128)
    def _fetch_from_yahoo(self, ticker_str, timestamp_bin):
       # use mock data to test the service
       if self.dev_mode:
           print(f" DEBUG: using mock data for {ticker_str}")
           mock_history = {
               "Close": {
                   "09:30": 150.00,"09:35": 151.20,"09:40": 150.80,"09:45": 152.30,"09:50": 151.90,"09:55": 153.10,
                   "10:00": 152.50,"10:05": 154.00,"10:10": 153.80,"10:15": 155.00
                   }
                }
           mock_news = [
               {"uuid": "n1","title": f"{ticker_str} publishes Q2 results, earnings beat expectations","link": "https://finance.yahoo.com","publisher": "MarketWatch"},
               {"uuid": "n2","title": f"analyst: {ticker_str} target price at $200","link": "https://finance.yahoo.com","publisher": "Reuters"},
               {"uuid": "n3","title": f" {ticker_str} grows 10% in Q2","link": "https://finance.yahoo.com","publisher": "Bloomberg"}
           ]
           return{
               "symbol": ticker_str,"price": 155.00,"change": 3.33,"history": mock_history,"news": mock_news
               }
       print(f" Connecting to yfinance for latest data for {ticker_str}")
       ticker = yf.Ticker(ticker_str)
       fast_info = ticker.fast_info

       hist = ticker.history(period = "1d", inteval = "1m")

       #hist_dict = hist['Close'].to_dict() if not hist.empty else {}

       chart_data = {
        "timestamps": [t.strftime('%H:%M') for t in hist.index],
        "prices": [round(p, 2) for p in hist['Close'].tolist()]
        }

       return{
           "symbol": ticker_str,
           "price": round(fast_info.last_price,2),
           "change": round(fast_info.day_change_percent,2),
           "history": chart_data,
           "news": ticker.news[:5]

       }

    def get_stock_data(self, ticker: str, period: str = "1d", interval: str = "1m"):
        """
        get real-time stock price data using yfinance
        """
        timestamp_bin = int(time.time() / 60)
        try:
            return self._fetch_from_yahoo(ticker, timestamp_bin)
    
        except Exception as e:
            return {"error": "Rate limit exceeded, please try later."}
        
    
    def extract_ticker(self, text: str) -> str:
        """
        get Ticker from text
        """
        text = text.lower()

        # strategy A: directly match (if user inputs like AAPL )
        potential_tickers = re.findall(r'\b[a-zA-Z]{1,5}\b', text)
        for t in potential_tickers:
            if t.upper() in self.TICKER_MAP.values():
                return t.upper()

        # strategy B: fuzzy match (sloving, such as “特斯拉”、“特拉斯”, brief words)
        # extractOne return (match, score, index)
        result = process.extractOne(
            text, 
            self.TICKER_MAP.keys(), 
            scorer=fuzz.partial_ratio
        )
        
        if result and result[1] > 70:  # score > 70 as success 
            matched_name = result[0]
            return self.TICKER_MAP[matched_name]

        return "UNKNOWN"
    
    async def get_stock_data(self, ticker: str):
        """
        return data (support mock data)
        """
        if self.dev_mode or ticker == "UNKNOWN":
            return {
                "ticker": ticker if ticker != "UNKNOWN" else "Unknown Asset",
                "price": "185.20",
                "change": "+1.5%",
                "status": "MOCK_DATA"
            }
        #TODO : yfinance logic needed to add here

        return {"ticker": ticker, "price": "100.00", "change": "0.0%"}
    
# if __name__ == "__main__":
#     finance_service = FinanceService(dev_mode=True)
#     print(finance_service.get_stock_data("AAPL"))