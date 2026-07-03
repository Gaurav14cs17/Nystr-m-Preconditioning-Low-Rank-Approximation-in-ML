"""
Diffusion dataset using synthetic digit-like patterns (no download needed).
Generates 28x28 grayscale images with circles, lines, and blobs.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class DiffusionDataset(Dataset):
    """Synthetic 28x28 patterns for diffusion training.
    Returns images normalized to [-1, 1]."""

    def __init__(self, num_samples=1000, train=True, data_dir='./data'):
        rng = np.random.RandomState(42 if train else 123)
        self.images = []
        for i in range(num_samples):
            img = np.zeros((28, 28), dtype=np.float32)
            pattern = i % 5
            if pattern == 0:
                cx, cy = rng.randint(8, 20, 2)
                r = rng.randint(3, 8)
                yy, xx = np.ogrid[-cx:28-cx, -cy:28-cy]
                mask = xx**2 + yy**2 <= r**2
                img[mask] = 1.0
            elif pattern == 1:
                thickness = rng.randint(1, 4)
                start = rng.randint(2, 12)
                img[start:start+thickness, 4:24] = 1.0
                img[10:20, start:start+thickness] = 1.0
            elif pattern == 2:
                for _ in range(rng.randint(2, 6)):
                    x, y = rng.randint(2, 26, 2)
                    s = rng.randint(1, 4)
                    img[max(0,x-s):x+s, max(0,y-s):y+s] = rng.uniform(0.5, 1.0)
            elif pattern == 3:
                for r in range(2, 10, 2):
                    yy, xx = np.ogrid[-14:14, -14:14]
                    ring = (xx**2 + yy**2 >= (r-1)**2) & (xx**2 + yy**2 <= r**2)
                    img[ring] = 1.0 - r*0.08
            else:
                angle = rng.uniform(0, np.pi)
                for t in np.linspace(-12, 12, 50):
                    x = int(14 + t * np.cos(angle))
                    y = int(14 + t * np.sin(angle))
                    if 0 <= x < 28 and 0 <= y < 28:
                        img[x, y] = 1.0
            img += rng.randn(28, 28).astype(np.float32) * 0.05
            img = np.clip(img, 0, 1)
            img = img * 2 - 1  # normalize to [-1, 1]
            self.images.append(torch.tensor(img).unsqueeze(0))  # (1, 28, 28)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


def get_dataloader(batch_size=32, num_samples=1000, seed=42, data_dir='./data'):
    dataset = DiffusionDataset(num_samples=num_samples, data_dir=data_dir)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, generator=generator
    )
