"""
LoRA fine-tuning datasets using real sklearn Digits data (64 features from 8x8 images).
- Pretrain: multi-output regression on ALL digits (learn general representations)
- Finetune: DIFFERENT regression targets (requires model adaptation)
No download needed — sklearn bundles this dataset.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class PretrainDataset(Dataset):
    """Pretrain on sklearn Digits: 64 pixel features → 8 nonlinear target outputs.
    Uses real handwritten digit pixel data as input features."""

    def __init__(self, n_samples=2000, seed=42):
        digits = load_digits()
        X = digits.data.astype(np.float32)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        rng = np.random.RandomState(seed)
        n = min(n_samples, len(X))
        idx = rng.choice(len(X), n, replace=(n > len(X)))
        X = X[idx]

        W_target = rng.randn(64, 8).astype(np.float32) * 0.3
        Y = np.tanh(X @ W_target) + rng.randn(n, 8).astype(np.float32) * 0.05

        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class FinetuneDataset(Dataset):
    """Finetune on sklearn Digits: DIFFERENT target function (requires adaptation).
    Same real pixel inputs but different nonlinear mapping to learn."""

    def __init__(self, n_samples=200, seed=123):
        digits = load_digits()
        X = digits.data.astype(np.float32)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        rng = np.random.RandomState(seed)
        n = min(n_samples, len(X))
        idx = rng.choice(len(X), n, replace=(n > len(X)))
        X = X[idx]

        W_target = rng.randn(64, 8).astype(np.float32) * 0.5
        b_target = rng.randn(8).astype(np.float32) * 0.2
        Y = np.sin(X @ W_target + b_target) + rng.randn(n, 8).astype(np.float32) * 0.05

        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def get_pretrain_loader(batch_size=32, n_samples=2000, seed=42):
    dataset = PretrainDataset(n_samples=n_samples, seed=seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def get_finetune_loader(batch_size=16, n_samples=200, seed=123):
    dataset = FinetuneDataset(n_samples=n_samples, seed=seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
