"""
Trainers for pretraining, LoRA fine-tuning, and full fine-tuning.

PretrainTrainer  — trains the base model on the pretraining dataset.
LoRAFinetuner    — trains only the LoRA adapter parameters (A, B).
FullFinetuner    — unfreezes all parameters for a full fine-tuning baseline.
"""

import torch
import torch.nn as nn


class PretrainTrainer:
    def __init__(self, model: nn.Module, dataloader, lr: float = 1e-3):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train(self, num_epochs: int = 20) -> list[float]:
        self.model.train()
        loss_history = []
        for _ in range(num_epochs):
            epoch_loss = 0.0
            n = 0
            for X, Y in self.dataloader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X), Y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                n += 1
            loss_history.append(epoch_loss / n)
        return loss_history


class LoRAFinetuner:
    """Only updates the LoRA adapter parameters (requires_grad=True)."""

    def __init__(self, lora_model: nn.Module, dataloader, lr: float = 1e-3):
        self.model = lora_model
        self.dataloader = dataloader
        trainable = [p for p in lora_model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=lr)
        self.criterion = nn.MSELoss()

    def train(self, num_epochs: int = 10) -> list[float]:
        self.model.train()
        loss_history = []
        for _ in range(num_epochs):
            epoch_loss = 0.0
            n = 0
            for X, Y in self.dataloader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X), Y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                n += 1
            loss_history.append(epoch_loss / n)
        return loss_history

    def evaluate(self, test_loader) -> dict:
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for X, Y in test_loader:
                total_loss += self.criterion(self.model(X), Y).item()
                n += 1
        return {"test_loss": total_loss / n}


class FullFinetuner:
    """Unfreezes all parameters for a full fine-tuning baseline comparison."""

    def __init__(self, model: nn.Module, dataloader, lr: float = 1e-3):
        self.model = model
        self.dataloader = dataloader
        for p in model.parameters():
            p.requires_grad = True
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train(self, num_epochs: int = 10) -> list[float]:
        self.model.train()
        loss_history = []
        for _ in range(num_epochs):
            epoch_loss = 0.0
            n = 0
            for X, Y in self.dataloader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X), Y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                n += 1
            loss_history.append(epoch_loss / n)
        return loss_history

    def evaluate(self, test_loader) -> dict:
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for X, Y in test_loader:
                total_loss += self.criterion(self.model(X), Y).item()
                n += 1
        return {"test_loss": total_loss / n}
