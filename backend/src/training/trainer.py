# backend/src/training/trainer.py

import torch
from tqdm.auto import tqdm
import time
import os

class MultiTaskTrainer:
    def __init__(self, model, optimizer, device, save_path, tokenizer):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.tokenizer = tokenizer
        self.best_val_accuracy = 0.0
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_epoch(self, data_loader):
        """single epoch training"""
        self.model.train()
        running_loss = 0.0
        
        for batch in tqdm(data_loader, desc="Training"):
            self.optimizer.zero_grad()
            
            # move data to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            task_ids = batch['task_id'].to(self.device)
            
            # forward
            sent_logits, intent_logits = self.model(input_ids, attention_mask)
            
            # calculate loss
            batch_loss = 0
            if (task_ids == 0).any():
                batch_loss += self.criterion(sent_logits[task_ids == 0], labels[task_ids == 0])
            if (task_ids == 1).any():
                batch_loss += self.criterion(intent_logits[task_ids == 1], labels[task_ids == 1])
            
            if torch.is_tensor(batch_loss):
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                running_loss += batch_loss.item()
        
        return running_loss / len(data_loader)

    def evaluate(self, data_loader):
        """evaluate model performance with loss and multi-task accuracy"""
        self.model.eval()
        total_loss = 0
        results = {
            0: {"correct": 0, "total": 0},
            1: {"correct": 0, "total": 0},
        }
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                task_ids = batch['task_id'].to(self.device)


                sent_logits, intent_logits = self.model(input_ids, attention_mask)
                
                # --- calculate loss for each task ---
                batch_loss = 0
                if(task_ids == 0).any():
                    batch_loss = self.criterion(sent_logits[task_ids == 0], labels[task_ids == 0])
                if(task_ids == 1).any():
                    batch_loss = self.criterion(intent_logits[task_ids == 1], labels[task_ids == 1])
                if torch.is_tensor(batch_loss):
                    total_loss += batch_loss.item()

                # --- calculate accuracy for each task ---
                if (task_ids == 0).any():
                    _, preds = torch.max(sent_logits[task_ids == 0], dim=1)
                    results[0]["correct"] += (preds == labels[task_ids == 0]).sum().item()
                    results[0]["total"] += (task_ids == 0).sum().item()

                if (task_ids == 1).any():
                    _, preds = torch.max(intent_logits[task_ids == 1], dim=1)
                    results[1]["correct"] += (preds == labels[task_ids == 1]).sum().item()
                    results[1]["total"] += (task_ids == 1).sum().item()

        avg_loss = total_loss / len(data_loader)
        acc_s = results[0]["correct"] / results[0]["total"] if results[0]["total"] > 0 else 0
        acc_i = results[1]["correct"] / results[1]["total"] if results[1]["total"] > 0 else 0
        return avg_loss, acc_s, acc_i
        
    def save_checkpoint(self, val_accuracy):
        """save the best model checkpoint"""
        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            os.makedirs(self.save_path, exist_ok=True)

            checkpoint_path = os.path.join(self.save_path, "best_model.pth")
            torch.save(self.model.state_dict(), checkpoint_path)

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
            val_loss, acc_s, acc_i = self.evaluate(val_dataloader)

            # A. sentiment 0.7 weight, intent 0.3 weight
            #combined_acc = (acc_s * 0.7) + (acc_i * 0.3)

            # B. insure sentiment analysis doesn't drop, while intent recognition achieves the goal
            combined_acc = (acc_s + acc_i) / 2

            end_time = time.time()
            
            print(f"\n--- Epoch {epoch}/{epochs} ---")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Sentiment Acc: {acc_s:.4f} | Intent Acc: {acc_i:.4f} | Combined Acc: {combined_acc:.4f}")
            print(f"Time taken: {(end_time - start_time):.2f}s")
            
            # use sentiment accuracy as metric, due to its quality and stability
            self.save_checkpoint(acc_s)
            
        print("\nTraining complete!")
        print(f"Best Validation Accuracy achieved: {self.best_val_accuracy:.4f}")
