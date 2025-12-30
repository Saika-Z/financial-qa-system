'''
Author: Zhao
Date: 2025-12-29 10:58:27
LastEditors: 
LastEditTime: 2025-12-30 17:52:44
FilePath: clean_intention_bert.py
Description: 

'''

# /backend/src/data_clean/processed/clean_intention_bert.py
import os
import json
from backend.src.config import settings
import pandas as pd
import re

def clean_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

        intent_map = {"finance": 0, "sentiment": 1, "rag": 2}

        if isinstance(data, list):
            df = pd.DataFrame(data)
           
        elif isinstance(data, dict):
            all_items = []
            for v in data.values():
                if isinstance(v, list):
                    all_items.extend(v)
            df = pd.DataFrame(all_items)
        else:
            return []
        
        # if 'Text' in df.columns:
        #     df['Text'] = df['Text'].apply(clean_text)
        if 'Label' in df.columns:
            df['label'] = df['Label'].str.lower().map(intent_map)
        if 'Ticker' in df.columns:
            df['ticker'] = df['Ticker'].fillna('N/A')
        else:
            df['ticker'] = 'N/A'
    result_df = df[['Text', 'label', 'ticker']]
    return result_df.to_dict(orient='records')
    

def prepare_directory(dir_path, output_path):
    all_file_list = []

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for root, _, files in os.walk(dir_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            # add all file to a list
            all_file_list.extend(clean_json_file(file_path))

    df = pd.DataFrame(all_file_list)

    # write to csv for BERT and FastSet
    bert_file = os.path.join(output_path, 'intention_bert.csv')
    fastset_file = os.path.join(output_path, 'intention_fastset.csv')

    df.to_csv(bert_file, index=False, encoding='utf-8', mode='w')
    df.to_csv(fastset_file, index=False, encoding='utf-8', mode='w')

    print(f"Cleaned data saved to {bert_file}")
    print(f"Cleaned data saved to {fastset_file}")



def clean_text(text):
    # 用正则替换所有不符合标准字符的字符
    text = re.sub(r'[’‘“”]', "'", text)  # 将不同类型的引号都替换为标准单引号
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # 去掉非 ASCII 字符
    return text

if __name__ == "__main__":
    raw_path = settings.INTENTION_RAW_PATH
    output_path = settings.INTENTION_CLEAN_PATH
    prepare_directory(raw_path, output_path)