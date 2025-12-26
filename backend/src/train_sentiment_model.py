
# backend/src/train_sentiment_model.py
import torch
from torch.optim import AdamW
from transformers import BertForSequenceClassification
import os

from backend.src.config import settings
from backend.src.data_utils.dataset import get_dataloaders
from backend.src.training.trainer import SentimentTrainer


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
    
    main()