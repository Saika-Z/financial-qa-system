
#  /backend/app/services/inference.py
import torch    
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from rapidfuzz import process, fuzz
import os
import regex as re
from backend.app.core.config import config


class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_sentiment=3, num_intent=3):
        super(MultiTaskModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.sentiment_head = nn.Linear(768, num_sentiment)
        self.intent_head = nn.Linear(768, num_intent)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        sent_logits = self.sentiment_head(pooled_output)
        intent_logits = self.intent_head(pooled_output)
        return sent_logits, intent_logits

# --- 2. Predict ---    
class FinancialPredictor:
    def __init__(self, model_path, model_name, device=None):
        #self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.tokenizer = BertTokenizer.from_pretrained(model_path) # 从保存目录加载 tokenizer
        
        # init model
        self.model = MultiTaskModel(model_name)
        # load .pth 
        state_dict = torch.load(os.path.join(model_path, "best_model.pth"), map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # mapping, same as training
        self.ticker_map = config.TICKER_MAP
        self.id2sent = {0: "Neutral", 1: "Negative", 2: "Positive"}
        self.id2intent = {0: "FINANCE", 1: "SENTIMENT", 2: "RAG"}

    def _has_ticker(self, text):
        """正则匹配常见的股票代码格式"""
        # 匹配: $AAPL, 600519, 00700.HK, 腾讯控股 等
        patterns = [
            r'[A-Z]{2,5}',        # 美股 Ticker
            r'\d{5,6}',           # A股/港股 代码
            r'\$+[A-Za-z]+'       # $AAPL 格式
        ]
        return any(re.search(p, text) for p in patterns)

    def predict(self, text):
        # data preprocess
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding='max_length', 
            truncation=True, 
            max_length=128
        ).to(self.device)
        

        with torch.no_grad():
            # forward
            sent_logits, intent_logits = self.model(inputs['input_ids'], inputs['attention_mask'])

            # intent prediction
            intent_probs_all = torch.softmax(intent_logits, dim=1)
            intent_conf, intent_idx = torch.max(intent_probs_all, dim=1)
            intent_idx = intent_idx.item()
            intent_label = self.id2intent[intent_idx]
            intent_conf = intent_conf.item()

            sent_probs_all = torch.softmax(sent_logits, dim=1)
            _, sent_idx_tensor = torch.max(sent_probs_all, dim=1)
            sent_idx = sent_idx_tensor.item()

            # RAG detection
            finance_keywords = ["价格", "股价", "现价", "多少钱", "走势", "行情", "实时"]
            rag_keywords = [
                "解释", "定义", "什么是", "为什么", "逻辑", "变动", 
            "总结", "分析", "报告", "财报", "10-K", "意见", "看法",
            "趋势", "区别", "比较", "业绩"
            ]
            is_long_query = len(text) > 18

            has_finance_word = any(k in text for k in finance_keywords)
            has_rag_word = any(k in text for k in rag_keywords)

            #has_ticker = self._has_ticker(text)

            final_intent = intent_label

            if has_rag_word or is_long_query:
                if not (has_finance_word and len(text) < 15):
                    final_intent = "RAG"
                    #intent_conf = 0.99
            
            elif intent_label == "FINANCE" and not has_finance_word:
                final_intent = "RAG"
                #intent_conf = 0.99
            
            if has_finance_word and len(text) < 10:
                final_intent = "FINANCE"

            # # if Finance intent, but confidence is low, or contains obvious search keywords, force to RAG
            # if intent_label == "FINANCE":
            #     if (intent_conf < 0.85 and is_long_query) or has_rag_word:
            #         intent_label = "RAG"
            #         intent_conf = 0.99

            # # sentiment prediction
            # sent_probs_all = torch.softmax(sent_logits, dim=1)
            # sent_conf, sent_idx = torch.max(sent_probs_all, dim=1)
            # sent_idx = sent_idx.item()
            # sent_conf = sent_conf.item()
        
        return {
            "text": text,
            "sentiment": self.id2sent[sent_idx],
            #"sent_confidence": f"{sent_conf:.2%}",
            "intent": final_intent,
            #"intent_confidence": f"{intent_conf:.2%}",
            "is_fallback": final_intent != intent_label
        }

if __name__ == "__main__":
    predictor = FinancialPredictor(config.LOCAL_BERT_PATH, config.BASE_MODE_NAME)

    # example
    test_cases = [
        "Apple's stock price plummeted after the news.", # 情感应为负面，意图应为 FINANCE
        "What is the definition of Price-to-Earnings ratio?", # 意图应为 RAG
        "I am very happy with the market growth!", # 情感应为正面，意图应为 SENTIMENT
        "特斯拉最近的业绩报告出炉了吗？", # 中文测试：意图应为 FINANCE
        "帮我解释一下什么是量化宽松", # 中文测试：意图应为 RAG
        "苹果今年的财报真烂，我想知道详情。",
        "苹果今年的财报太棒了，我想知道详情。",
        "我想知道今天苹果股价。",
        "$FB trending nicely, intraday."
    ]
    for f in test_cases:
        print(predictor.predict(f))

