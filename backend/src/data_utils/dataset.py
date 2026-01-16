
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import WeightedRandomSampler

class SentimentDataset(Dataset):
    """
    dataset for sentiment analysis
    """
    def __init__(self, encodings, sentiment_labels, intent_labels, task_ids):
        self.encodings = encodings
        self.sentiment_labels = sentiment_labels
        self.intent_labels = intent_labels
        self.task_ids = task_ids

    def __len__(self):
        return len(self.sentiment_labels)

    def __getitem__(self, item):
        return{
            'input_ids': self.encodings['input_ids'][item].squeeze(0),
            'attention_mask': self.encodings['attention_mask'][item].squeeze(0),
            'sentiment_labels': torch.tensor(self.sentiment_labels[item], dtype=torch.long),
            'intent_labels': torch.tensor(self.intent_labels[item], dtype=torch.long),
            'task_id': torch.tensor(self.task_ids[item],dtype=torch.long)
        }

def get_dataloaders(sentiment_paths, intent_path, model_name, batch_size, max_seq_len):
    """
    load data from csv files and create DataLoader for train, val, test sets
    """

    # 1. load data
    # sentence file
    s_train_df = pd.read_csv(sentiment_paths['train'])
    s_val_df = pd.read_csv(sentiment_paths['val'])
    s_test_df = pd.read_csv(sentiment_paths['test'])

    # intention file
    i_df = pd.read_csv(intent_path)
    i_train_df = i_df.sample(frac=0.9, random_state=42)
    i_val_df = i_df.drop(i_train_df.index)


    # 2. tokenization
    tokenizer = BertTokenizer.from_pretrained(model_name)
    params = {'padding': 'max_length', 'truncation': True, 'max_length': max_seq_len, 'return_tensors': "pt"}

    # 3. define a inner function: tagging and encoding
    def encode_data(df, text_col, task_id):
        texts = df[text_col].tolist()
        raw_labels = df['label'].tolist()
        sentiment_labels = []
        intent_labels = []

        if task_id == 0:
            # kaggle data. intent_labels is -100 to ignore
            sentiment_labels = raw_labels
            intent_labels = [-100] * len(df)
        else:
            sentiment_labels = [0] * len(df)
            intent_labels = raw_labels

        task_ids = [task_id] * len(df)
        encodings = tokenizer(texts, **params)

        return SentimentDataset(encodings, sentiment_labels, intent_labels, task_ids)

    # 4. create train and val dataset
    # -- train --
    ds_s_train = encode_data(s_train_df, 'Sentence', task_id=0)
    ds_i_train = encode_data(i_train_df, 'Text', task_id=1)
    train_dataset = torch.utils.data.ConcatDataset([ds_s_train, ds_i_train])

    # -- valdation --
    ds_s_val = encode_data(s_val_df, 'Sentence', task_id=0)
    ds_i_val = encode_data(i_val_df, 'Text', task_id=1)
    val_dataset = torch.utils.data.ConcatDataset([ds_s_val, ds_i_val])

    # -- test --
    ds_s_test = encode_data(s_test_df, 'Sentence', task_id=0)
    test_dataset = torch.utils.data.ConcatDataset([ds_s_test, ds_i_val])

    # 5. weights and create DataLoader
    num_s = len(ds_s_train)
    num_i = len(ds_i_train)

    weights = [1.0 / num_s] * num_s + [1.0 / num_i] * num_i
    sampler = WeightedRandomSampler(
        weights=weights, 
        num_samples=len(train_dataset), 
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    # valadation and test don't need sampler
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"DEBUG: num_s={num_s}, num_i={num_i}, concat_len={len(train_dataset)}")
    
    return train_loader, val_loader, test_loader, tokenizer
