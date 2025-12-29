
import torch.nn as nn
from transformers import BertModel
from backend.src.config import settings

class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_sentiment, num_intent):
        super(MultiTaskModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(settings.DROPOUT)
        self.sentiment_head = nn.Linear(settings.BERT_HIDDEN_SIZE, num_sentiment)
        self.intent_head = nn.Linear(settings.BERT_HIDDEN_SIZE, num_intent)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        sentiment_logits = self.sentiment_head(pooled_output)
        intent_logits = self.intent_head(pooled_output)
        return sentiment_logits, intent_logits