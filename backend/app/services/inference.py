
#  /backend/app/services/inference.py
import torch    
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import os
import backend.app.core.config as config


#---- This script is for valadation the model ----

# --- 1. Define Model must be same as training ---
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
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_path) # 从保存目录加载 tokenizer
        
        # init model
        self.model = MultiTaskModel(model_name)
        # load .pth 
        state_dict = torch.load(os.path.join(model_path, "best_model.pth"), map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # mapping, same as training
        self.id2sent = {0: "Negative 🔴", 1: "Neutral ⚪", 2: "Positive 🟢"}
        self.id2intent = {0: "FINANCE (股价查询) 💰", 1: "SENTIMENT (情绪分析) 📊", 2: "RAG (知识百科) 📖"}

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
            
            # get prediction
            sent_idx = torch.argmax(sent_logits, dim=1).item()
            intent_idx = torch.argmax(intent_logits, dim=1).item()
            
            # get probability
            sent_probs = torch.softmax(sent_logits, dim=1)[0][sent_idx].item()
            intent_probs = torch.softmax(intent_logits, dim=1)[0][intent_idx].item()

        return {
            "text": text,
            "sentiment": self.id2sent[sent_idx],
            "sent_confidence": f"{sent_probs:.2%}",
            "intent": self.id2intent[intent_idx],
            "intent_confidence": f"{intent_probs:.2%}"
        }

# --- 3. test ---
if __name__ == "__main__":
    MODEL_DIR = config.LOCAL_BERT_PATH
    BASE_MODEL = config.BASE_MODE_NAME # 训练时用的底座名

    predictor = FinancialPredictor(MODEL_DIR, BASE_MODEL)

    # example
    test_cases = [
        "Apple's stock price plummeted after the news.", # 情感应为负面，意图应为 FINANCE
        "What is the definition of Price-to-Earnings ratio?", # 意图应为 RAG
        "I am very happy with the market growth!", # 情感应为正面，意图应为 SENTIMENT
        "特斯拉最近的业绩报告出炉了吗？", # 中文测试：意图应为 FINANCE
        "帮我解释一下什么是量化宽松" # 中文测试：意图应为 RAG
    ]

    print("\n--- Model Inference Test ---")
    for text in test_cases:
        res = predictor.predict(text)
        print("-" * 50)
        print(f"Input: {res['text']}")
        print(f"Result: {res['sentiment']} (Conf: {res['sent_confidence']})")
        print(f"Intent: {res['intent']} (Conf: {res['intent_confidence']})")