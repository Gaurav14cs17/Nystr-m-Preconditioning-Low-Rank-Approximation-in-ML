import torch
import torch.nn as nn


class TransformerTrainer:
    def __init__(self, model, train_loader, val_loader, lr=1e-3, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, num_epochs=10):
        history = {'train_loss': [], 'val_acc': []}
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            for input_ids, labels in self.train_loader:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                logits, _ = self.model(input_ids)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            val_acc = self.evaluate()
            history['train_loss'].append(avg_loss)
            history['val_acc'].append(val_acc)

        return history

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        for input_ids, labels in self.val_loader:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            logits, _ = self.model(input_ids)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return correct / max(total, 1)

    @torch.no_grad()
    def get_attention_maps(self, batch):
        self.model.eval()
        input_ids, labels = batch
        input_ids = input_ids.to(self.device)
        _, all_weights = self.model(input_ids, return_weights=True)
        return all_weights
