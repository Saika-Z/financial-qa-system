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
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        #self.criterion = torch.nn.CrossEntropyLoss()

    def train_epoch(self, data_loader):
        """single epoch training"""
        self.model.train()
        running_loss = 0.0
        
        for batch in tqdm(data_loader, desc="Training"):
            self.optimizer.zero_grad()
            
            # move data to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            sent_labels = batch['sentiment_labels'].to(self.device)
            intent_labels = batch['intent_labels'].to(self.device)
            
            # forward
            sent_logits, intent_logits = self.model(input_ids, attention_mask)
            
            # calculate loss
            loss_sent = self.criterion(sent_logits, sent_labels)
            loss_intent = self.criterion(intent_logits, intent_labels)

            batch_loss = torch.tensor(0.0, device=self.device)

            if not torch.isnan(loss_sent):
                batch_loss += loss_sent
            
            if not torch.isnan(loss_intent):
                batch_loss += loss_intent

            if batch_loss > 0:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                running_loss += batch_loss.item()
            else:
                continue
                
        return running_loss / len(data_loader)

        #     batch_loss.backward()
        #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        #     self.optimizer.step()
        #     running_loss += batch_loss.item()
        
        # return running_loss / len(data_loader)

    def evaluate(self, data_loader):
        """evaluate model performance with loss and multi-task accuracy"""
        self.model.eval()
        total_loss = 0
        valid_steps = 0
        results = {
            "sent": {"correct": 0, "total": 0},
            "intent": {"correct": 0, "total": 0},
        }
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                sent_labels = batch['sentiment_labels'].to(self.device)
                intent_labels = batch['intent_labels'].to(self.device)


                sent_logits, intent_logits = self.model(input_ids, attention_mask)
                
                # --- calculate loss for each task ---
                loss_sent = self.criterion(sent_logits, sent_labels)
                loss_intent = self.criterion(intent_logits, intent_labels)

                current_batch_loss = 0.0
                has_valid_loss = False

                if not torch.isnan(loss_sent):
                    current_batch_loss += loss_sent.item()
                    has_valid_loss = True
                if not torch.isnan(loss_intent):
                    current_batch_loss += loss_intent.item()
                    has_valid_loss = True
                
                if has_valid_loss:
                    total_loss += current_batch_loss
                    valid_steps += 1
                
                sent_mask = (sent_labels != -100)
                if sent_mask.any():
                    _, sent_preds = torch.max(sent_logits[sent_mask], dim=1)
                    results["sent"]["correct"] += (sent_preds == sent_labels[sent_mask]).sum().item()
                    results["sent"]["total"] += sent_mask.sum().item()

                intent_mask = (intent_labels != -100)
                if intent_mask.any():
                    _, intent_preds = torch.max(intent_logits[intent_mask], dim=1)
                    results["intent"]["correct"] += (intent_preds == intent_labels[intent_mask]).sum().item()
                    results["intent"]["total"] += intent_mask.sum().item()

        avg_loss = total_loss / valid_steps if valid_steps > 0 else 0
        acc_s = results["sent"]["correct"] / results["sent"]["total"] if results["sent"]["total"] > 0 else 0
        acc_i = results["intent"]["correct"] / results["intent"]["total"] if results["intent"]["total"] > 0 else 0
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
