'''
 # Author: Wenqing Zhao
 # Date: 2025-12-06 20:29:42
 # LastEditTime: 2025-12-08 15:13:48
 # Description: 
 # FilePath: /financial-qa-system/backend/data/processed/process_data.py
'''
import pandas as pd
from sklearn.model_selection import train_test_split

# --- 1. 数据加载与初始划分 (训练集 + 剩余集合) ---

kaggle_data = "backend/data/kaggle_data.csv"
data = pd.read_csv(kaggle_data)

# 标签编码（假设 'positive': 2, 'neutral': 1, 'negative': 0）
label_map = {'positive': 2, 'negative': 1, 'neutral': 0}
data['label'] = data['Sentiment'].map(label_map)
#print(data.head())

# 第一次划分：先分离出测试集 (例如 10%)
# stratify=data['label'] 确保每个集合中积极、消极、中性的比例一致
train_val_df, test_df = train_test_split(
    data, 
    test_size=0.10, 
    random_state=42, 
    stratify=data['label']
)

# --- 2. 第二次划分 (训练集 + 验证集) ---
# 从剩余的 90% 中分离出验证集 (例如 10% / 0.9 = 约 11.11%)
train_df, val_df = train_test_split(
    train_val_df, 
    test_size=(0.10/0.90), # 确保验证集在原始数据中仍占 10%
    random_state=42, 
    stratify=train_val_df['label']
)

print(f"训练集大小: {len(train_df)}") # 约 80%
print(f"验证集大小: {len(val_df)}")   # 约 10%
print(f"测试集大小: {len(test_df)}")   # 约 10%

# 1. 划分后的保存
train_df.to_csv("backend/data/processed/train.csv", index=False)
val_df.to_csv("backend/data/processed/val.csv", index=False)
test_df.to_csv("backend/data/processed/test.csv", index=False)