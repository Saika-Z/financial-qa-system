# /backend/src/data_clean/processed/kaggle_data_split.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split

# --- 1. Data loading and initial splitting (training set + remaining set) ---

kaggle_data = "data/raw/kaggle/kaggle_data.csv"
data = pd.read_csv(kaggle_data)
output_path = "data/processed/kaggle_split"


# Label encoding (assuming 'positive': 2, 'negative': 1, 'neutral': 0)
label_map = {'positive': 2, 'negative': 1, 'neutral': 0}
data['label'] = data['Sentiment'].map(label_map)
#print(data.head())

# First division: Separate out the test set first (e.g., 10%).
# `stratify=data['label']` ensures that the proportion of positive, negative, and neutral samples is consistent in each subset.
train_val_df, test_df = train_test_split(
    data, 
    test_size=0.10, 
    random_state=42, 
    stratify=data['label']
)

# --- 2. Second division (training set + validation set) ---
# Separate the validation set from the remaining 90% (e.g., 10% / 0.9 = approximately 11.11%).
train_df, val_df = train_test_split(
    train_val_df, 
    test_size=(0.10/0.90), # Ensure that the validation set still accounts for 10% of the original data.
    random_state=42, 
    stratify=train_val_df['label']
)

print(f"Training set size: {len(train_df)}") # Approximately 80%
print(f"Validation set size: {len(val_df)}")   # Approximately 10%
print(f"Test set size: {len(test_df)}")   # Approximately 10%


if not os.path.exists(output_path):
    os.makedirs(output_path)

out_train_path = os.path.join(output_path, "train.csv")
out_val_path = os.path.join(output_path, "val.csv")
out_test_path = os.path.join(output_path, "test.csv")  

# 3. Saving after partitioning
train_df.to_csv(out_train_path, index=False)
val_df.to_csv(out_val_path, index=False)
test_df.to_csv(out_test_path, index=False)