# /backend/app/services/tools/finance_client.py
import sqlite3
import time
import yfinance as yf
from backend.app.core.config import config
import logging

logger = logging.getLogger(__name__)
class FinanceClient:
    def __init__(self, db_path="data/finance_cache.db"):
        self.db_path = db_path
        self._init_db()
        self._sync_with_config()

    def _init_db(self):
        # 初始化表结构，只在第一次运行
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS stock_cache (
                    ticker TEXT PRIMARY KEY, 
                    name TEXT,
                    price REAL, 
                    timestamp INTEGER
                )
            ''')
            conn.commit()

    def _sync_with_config(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for name, ticker in config.TICKER_MAP.items():
                # if not exist, insert(price=0.0, timestamp=0)
                cursor.execute("""
                    INSERT OR IGNORE INTO stock_cache (ticker, name, price, timestamp) 
                    VALUES (?, ?, ?, ?)
                """, (ticker, name, 0.0, 0))
            conn.commit()

    def get_price(self, ticker_symbol):
        now = int(time.time())
        ticker_symbol = ticker_symbol.upper()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT price, timestamp FROM stock_cache WHERE ticker=?", (ticker_symbol,))
            row = cursor.fetchone()

            # strategy：10 minute (600 seconds) directly use cache
            if row and row[0] > 0 and (now - row[1] < 600):
                return {"price": row[0], "status": "cached", "last_update": row[1]}

            # otherwise, call yfinance
            return self._fetch_and_update(ticker_symbol)
        
    def _fetch_and_update(self, ticker):
        try:
            logger.info(f"trying to get the latest price of {ticker} from yfinance ...")
            stock = yf.Ticker(ticker)
            # use fast_info to reduce the risk of ip blocking
            new_price = round(stock.fast_info['last_price'], 2)
            now = int(time.time())

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE stock_cache SET price=?, timestamp=? WHERE ticker=?", 
                    (new_price, now, ticker)
                )
                conn.commit()
            return {"price": new_price, "status": "updated", "last_update": now}
        except Exception as e:
            logger.error(f"yfinance service failed: {e}")
            # if failed, use cached data (if exists)
            return {"error": "API_LIMIT", "detail": str(e)}