'''
 # Author: Wenqing Zhao
 # Date: 2025-12-08 18:23:25
 # LastEditTime: 2025-12-09 16:12:29
 # Description: 
 # FilePath: /financial-qa-system/backend/app/services/sentiment_service.py
'''
from backend.app.core.config import settings
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import os
import pathlib

class SentimentService:
    def __init__(self, model_path, tokenizer_path):
        # 确定设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 使用传入的路径加载模型和分词器
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

        # 将模型移动到设备
        self.model.to(self.device)
        self.model.eval() # 确保模型处于评估模式

    def predict_sentiment(self, text):
        # 处理输入文本
        texts = [text] if isinstance(text, str) else text

        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

        # 将输入数据移动到设备
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
        
        # ensure the mapping matches .data/processed/process_data.py 'positive': 2, 'negative': 1, 'neutral': 0
        sentiment = {2: "positive", 1: "negative", 0: "neutral"}
        return sentiment[predicted_class]

# 初始化推理服务
# sentiment_service_instance = SentimentService(
#     model_path=settings.MODEL_PATH, 
#     tokenizer_path=settings.TOKENIZER_PATH
# )
sentiment_service_instance: SentimentService = None # 使用 None 初始化

def get_sentiment_service() -> SentimentService:
    """返回 SentimentService 的单例实例，只有在第一次调用时才加载模型。"""
    global sentiment_service_instance
    if sentiment_service_instance is None:

        model_path = settings.MODEL_PATH

        model_dir = pathlib.Path(model_path)

        if not model_dir.is_dir():
             # 打印 Uvicorn 子进程的工作目录，帮助最终定位问题
             current_cwd = os.getcwd() 
             raise FileNotFoundError(
                 f"Model directory not found at: {model_path}. "
                 f"CWD: {current_cwd}. " # 打印工作目录
                 f"Is it really a directory? {model_dir.exists()}." # 再次确认存在性
             )

        sentiment_service_instance = SentimentService(
            model_path=settings.MODEL_PATH, 
            tokenizer_path=settings.TOKENIZER_PATH
        )
    return sentiment_service_instance



# # 推理例子
# result = sentiment_service_instance.predict_sentiment("Apple's earnings report exceeded expectations.")
# print(result)  # 输出: positive
