"""
Character-level language model dataset using TinyShakespeare.
Downloads the text (~1MB) on first run from Karpathy's GitHub.
"""

import os
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _download_tinyshakespeare():
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "tinyshakespeare.txt")
    if not os.path.exists(path):
        print(f"  Downloading TinyShakespeare to {path}...")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


class CharDataset(Dataset):
    """Character-level dataset from TinyShakespeare.
    Each sample is a (input_ids, target_ids) pair of length seq_len,
    where target_ids = input_ids shifted by 1."""

    def __init__(self, num_samples=500, seq_len=32, split='train'):
        self.seq_len = seq_len
        self.vocab_size = 256

        text = _download_tinyshakespeare()
        data = [ord(c) % 256 for c in text]

        split_idx = int(len(data) * 0.9)
        if split == 'train':
            data = data[:split_idx]
        else:
            data = data[split_idx:]

        self.data = data
        self.num_samples = min(num_samples, len(data) - seq_len - 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * (len(self.data) // self.num_samples)
        start = min(start, len(self.data) - self.seq_len - 1)
        chunk = self.data[start:start + self.seq_len + 1]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)
        return input_ids, target_ids


def get_dataloader(batch_size=16, seq_len=32, num_samples=500, shuffle=True, split='train'):
    dataset = CharDataset(num_samples=num_samples, seq_len=seq_len, split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
