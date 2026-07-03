"""
Sequence classification dataset using sklearn's handwritten Digits dataset.
Each 8x8 digit image is flattened to a 64-length sequence of pixel-intensity tokens (0-16).
Classification task: identify which digit (0-9). Requires spatial attention.
No download needed — sklearn bundles this dataset.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class DigitSequenceDataset(Dataset):
    """Treats 8x8 digit images as sequences of 64 tokens for transformer classification.
    Each pixel value (0-16) becomes a token ID. The model must learn spatial
    relationships between pixel positions to classify digits."""

    def __init__(self, split='train', test_size=0.2, seed=42):
        digits = load_digits()
        X, y = digits.data, digits.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

        if split == 'train':
            self.X = torch.tensor(X_train, dtype=torch.long)
            self.y = torch.tensor(y_train, dtype=torch.long)
        else:
            self.X = torch.tensor(X_test, dtype=torch.long)
            self.y = torch.tensor(y_test, dtype=torch.long)

        self.seq_len = 64
        self.vocab_size = 17  # pixel values 0-16
        self.num_classes = 10

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloader(batch_size=32, split='train', **kwargs):
    """Returns a DataLoader for digit sequence classification.
    kwargs are accepted for API compatibility but ignored (data is fixed)."""
    dataset = DigitSequenceDataset(split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
