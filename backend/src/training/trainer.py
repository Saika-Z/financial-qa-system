# backend/src/training/trainer.py

import torch
from tqdm.auto import tqdm
import time
import os

class SentimentTrainer:
    def __init__(self, model, optimizer, device, save_path, tokenizer):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.tokenizer = tokenizer
        self.best_val_accuracy = 0.0

    def train_epoch(self, data_loader):
        """single epoch training"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(data_loader, desc="Training"):
            self.optimizer.zero_grad()
            
            # move data to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
        return total_loss / len(data_loader)

    def evaluate(self, data_loader):
        """evaluate model performance"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                _, preds = torch.max(outputs.logits, dim=1)
                correct_predictions += torch.sum(preds == labels).item()
            
        avg_loss = total_loss / len(data_loader)
        accuracy = correct_predictions / len(data_loader.dataset)
        return avg_loss, accuracy

    def save_checkpoint(self, val_accuracy):
        """save the best model checkpoint"""
        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            os.makedirs(self.save_path, exist_ok=True)
            self.model.save_pretrained(self.save_path)
            self.tokenizer.save_pretrained(self.save_path)
            print(f"Model saved to {self.save_path} with Acc: {val_accuracy:.4f}")
            return True
        return False
    
    def train(self, train_dataloader, val_dataloader, epochs):
        """full training loop"""
        print("\nStarting Training...")
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # training
            train_loss = self.train_epoch(train_dataloader)
            
            # evaluation
            val_loss, val_accuracy = self.evaluate(val_dataloader)

            end_time = time.time()
            
            print(f"\n--- Epoch {epoch}/{epochs} ---")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print(f"Time taken: {(end_time - start_time):.2f}s")
            
            # save best model
            self.save_checkpoint(val_accuracy)
            
        print("\nTraining complete!")
        print(f"Best Validation Accuracy achieved: {self.best_val_accuracy:.4f}")