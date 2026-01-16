
# backend/src/train_sentiment_model.py
import torch
from torch.optim import AdamW
from transformers import BertForSequenceClassification
import os

from backend.src.training.model import MultiTaskModel
from backend.src.config import settings
from backend.src.data_utils.dataset import get_dataloaders
from backend.src.training.trainer import MultiTaskTrainer


def main():
    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. data loaders
    train_dl, val_dl, test_dl, tokenizer = get_dataloaders(
        sentiment_paths={
            "train": settings.KAGGLE_CLEAN_TRAIN,
            "val": settings.KAGGLE_CLEAN_VAL,
            "test": settings.KAGGLE_CLEAN_TEST
        },
        intent_path=settings.INTENTION_BERT_DATA,
        model_name=settings.MODEL_NAME,
        batch_size=settings.BATCH_SIZE,
        max_seq_len=settings.MAX_SEQ_LEN
    )
    for batch in train_dl:
        s_labels = batch['sentiment_labels']
        i_labels = batch['intent_labels']
        # 过滤掉 -100 后，检查最大值
        valid_s = s_labels[s_labels != -100]
        valid_i = i_labels[i_labels != -100]
        if (valid_s >= 3).any() or (valid_i >= 3).any():
            print("发现非法标签！标签值必须在 0-2 之间。")
        break

    # 2. model initialization
    model = MultiTaskModel(
        settings.MODEL_NAME,
        num_sentiment=settings.NUM_SENTIMENT,
        num_intent=settings.NUM_INTENT
    )
    model.to(device)
    print(f" Model loaded from {settings.MODEL_NAME} successfully.")

    # 3. optimizer and trainer setup
    optimizer = AdamW(model.parameters(), lr=settings.LEARNING_RATE)
    
    trainer = MultiTaskTrainer(
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
    avg_loss, acc_s, acc_i = trainer.evaluate(test_dl)
    print(f"test loss: {avg_loss:.4f} | sentiment acc: {acc_s:.4f} | intent acc: {acc_i:.4f}")


if __name__ == '__main__':
    
    main()