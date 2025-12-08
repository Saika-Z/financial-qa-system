'''
 # Author: Wenqing Zhao
 # Date: 2025-12-06 20:30:07
 # LastEditTime: 2025-12-08 17:58:35
 # Description: 
 # FilePath: /financial-qa-system/backend/app/services/train_sentiment_model.py
'''
# backend/src/train_sentiment_model.py

import torch
from transformers import BertForSequenceClassification, AdamW
import os

# 导入自定义模块
from config import settings
from data_utils.dataset import get_dataloaders
from training.trainer import SentimentTrainer


def main():
    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. data loaders
    train_dl, val_dl, test_dl, tokenizer = get_dataloaders(
        train_path=settings.TRAIN_DATA_PATH,
        val_path=settings.VAL_DATA_PATH,
        test_path=settings.TEST_DATA_PATH,
        model_name=settings.MODEL_NAME,
        batch_size=settings.BATCH_SIZE,
        max_seq_len=settings.MAX_SEQ_LEN
    )

    # 2. model initialization
    model = BertForSequenceClassification.from_pretrained(
        settings.MODEL_NAME,
        num_labels=settings.NUM_LABELS
    )
    model.to(device)

    # 3. optimizer and trainer setup
    optimizer = AdamW(model.parameters(), lr=settings.LEARNING_RATE)
    
    trainer = SentimentTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        save_path=settings.MODEL_SAVE_PATH,
        tokenizer=tokenizer
    )

    # 4. training loop
    trainer.train(train_dl, val_dl, settings.EPOCHS)
    
    # 5. final evaluation on test set
    print("\nFinal evaluation on Test Set...")
    test_loss, test_accuracy = trainer.evaluate(test_dl)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


if __name__ == '__main__':
    """
    Ensure that when running the script, the current working directory is set to the project root
    so that relative paths in settings.py are correctly resolved.
    For example: If you run the script under backend/src/, you may need to adjust the path or run it from the project root.
    """
    
    # temporarily change working directory to project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(current_dir, '..')) # move up to project root
    
    # ensure src is in sys.path
    import sys
    sys.path.insert(0, current_dir)
    
    main()