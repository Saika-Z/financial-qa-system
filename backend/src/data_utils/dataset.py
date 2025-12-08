import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer

class SentimentDataset(Dataset):
    """
    dataset for sentiment analysis
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return{
            'input_ids': self.encodings['input_ids'][item],
            'attention_mask': self.encodings['attention_mask'][item],
            'labels': torch.tensor(self.labels[item])
        }

def get_dataloaders(train_path, val_path, test_path, model_name, batch_size, max_seq_len):
    """
    load data from csv files and create DataLoader for train, val, test sets
    """

    # 1. load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # 2. tokenization
    tokenizer = BertTokenizer.from_pretrained(model_name)
    params = {'padding': True, 'truncation': True, 'max_length': max_seq_len, 'return_tensors': "pt"}

    # 3. encode the datasets
    train_encodings = tokenizer(train_df['Sentence'].tolist(), **params)
    val_encodings = tokenizer(val_df['Sentence'].tolist(), **params)
    test_encodings = tokenizer(test_df['Sentence'].tolist(), **params)

    # 4. create Dataset instances
    train_dataset = SentimentDataset(train_encodings, train_df['label'].tolist())
    val_dataset = SentimentDataset(val_encodings, val_df['label'].tolist())
    test_dataset = SentimentDataset(test_encodings, test_df['label'].tolist())

    # 5. create DataLoader instances
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader
