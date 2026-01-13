# /backend/app/services/tools/ticker_tool.py
from rapidfuzz import process, fuzz
import regex as re
from backend.app.core.config import config

class TickerExtractor:
    def __init__(self):
        self.ticker_map = config.TICKER_MAP

    def extract_ticker(self, text: str) -> str:
        text = text.lower()
        # 策略 A: 正则匹配
        potential_tickers = re.findall(r'\b[a-zA-Z]{1,5}\b', text)
        for t in potential_tickers:
            if t.upper() in self.ticker_map.values():
                return t.upper()

        # 策略 B: 模糊匹配
        result = process.extractOne(
            text, self.ticker_map.keys(), scorer=fuzz.partial_ratio
        )
        return self.ticker_map[result[0]] if result and result[1] > 70 else "UNKNOWN"