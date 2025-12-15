'''
 # Author: Wenqing Zhao
 # Date: 2025-12-08 18:23:25
LastEditTime: 2025-12-15 12:57:41
 # Description: 
FilePath: sentiment_service.py
'''
from backend.app.core.config import settings
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import os
import pathlib

class SentimentService:
    def __init__(self, model_path, tokenizer_path):
        # Identify the device.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the model and tokenizer using the provided path.
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

        # Move the model to the device.
        self.model.to(self.device)
        self.model.eval() # Ensure the model is in evaluation mode.

    def predict_sentiment(self, text):
        # Processing the input text
        texts = [text] if isinstance(text, str) else text

        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

        # Move the input data to the device.
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
        
        # ensure the mapping matches .data/processed/process_data.py 'positive': 2, 'negative': 1, 'neutral': 0
        sentiment = {2: "positive", 1: "negative", 0: "neutral"}
        return sentiment[predicted_class]

# Initializing inference service
# sentiment_service_instance = SentimentService(
#     model_path=settings.MODEL_PATH, 
#     tokenizer_path=settings.TOKENIZER_PATH
# )
sentiment_service_instance: SentimentService = None # Initialize with None.

def get_sentiment_service() -> SentimentService:
    """Returns the singleton instance of SentimentService; the model is loaded only on the first call."""
    global sentiment_service_instance
    if sentiment_service_instance is None:

        model_path = settings.MODEL_PATH

        model_dir = pathlib.Path(model_path)

        if not model_dir.is_dir():
             # Print the working directory of the Uvicorn subprocess to help pinpoint the problem.
             current_cwd = os.getcwd() 
             raise FileNotFoundError(
                 f"Model directory not found at: {model_path}. "
                 f"CWD: {current_cwd}. " # Print the working directory.
                 f"Is it really a directory? {model_dir.exists()}." # Second confirmation of existence
             )

        sentiment_service_instance = SentimentService(
            model_path=settings.MODEL_PATH, 
            tokenizer_path=settings.TOKENIZER_PATH
        )
    return sentiment_service_instance



# # Reasoning examples
# result = sentiment_service_instance.predict_sentiment("Apple's earnings report exceeded expectations.")
# print(result)  # Output: positive
