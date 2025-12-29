
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import WeightedRandomSampler

class SentimentDataset(Dataset):
    """
    dataset for sentiment analysis
    """
    def __init__(self, encodings, labels, task_ids):
        self.encodings = encodings
        self.labels = labels
        self.task_ids = task_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return{
            'input_ids': self.encodings['input_ids'][item].squeeze(0),
            'attention_mask': self.encodings['attention_mask'][item].squeeze(0),
            'labels': torch.tensor(self.labels[item], dtype=torch.long),
            'task_id': torch.tensor(self.task_ids[item],dtype=torch.long)
        }

def get_dataloaders(sentiment_paths, intent_path, model_name, batch_size, max_seq_len):
    """
    load data from csv files and create DataLoader for train, val, test sets
    """

    # 1. load data
    # sentence file
    s_train = pd.read_csv(sentiment_paths['train'])
    s_val = pd.read_csv(sentiment_paths['val'])
    s_test = pd.read_csv(sentiment_paths['test'])
    
    # intention file
    i_df = pd.read_csv(intent_path)
    # simple split, only get 90% of data for training
    i_train = i_df.sample(frac=0.9, random_state=42)
    i_val = i_df.drop(i_train.index)


    # 2. tokenization
    tokenizer = BertTokenizer.from_pretrained(model_name)
    params = {'padding': 'max_length', 'truncation': True, 'max_length': max_seq_len, 'return_tensors': "pt"}

    # 3. define a inner function: tagging and encoding
    def encode_data(df, text_col, task_id):
        texts = df[text_col].tolist()
        labels = df['label'].tolist()
        task_ids = [task_id] * len(df)
        encodings = tokenizer(texts, **params)
        return SentimentDataset(encodings, labels, task_ids)

    # 4. create train and val dataset
    train_dataset = torch.utils.data.ConcatDataset([
        encode_data(s_train, 'Sentence', task_id = 0),
        encode_data(i_train, 'Text', task_id = 1)
    ])

    val_dataset = torch.utils.data.ConcatDataset([
        encode_data(s_val, 'Sentence', task_id = 0),
        encode_data(i_val, 'Text', task_id = 1)
    ])

    # intention files are limited so use val as test
    test_dataset = torch.utils.data.ConcatDataset([
        encode_data(s_test, 'Sentence', task_id = 0),
        encode_data(i_val, 'Text', task_id = 1)
    ])

    # 5. weights and create DataLoader
    num_s = len(s_train)
    num_i = len(i_train)

    weights = [1.0 / num_s] * num_s + [1.0 / num_i] * num_i
    sampler = WeightedRandomSampler(weights, num_samples=num_s+num_i, replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    
    return train_loader, val_loader, test_loader, tokenizer
