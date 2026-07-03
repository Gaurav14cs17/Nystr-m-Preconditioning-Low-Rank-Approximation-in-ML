import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np


class SyntheticClassificationDataset(Dataset):
    """
    Ill-conditioned binary classification with stretched feature scales.
    Decision boundary is nonlinear (based on radius in the first two features),
    and feature scales span orders of magnitude to create Hessian ill-conditioning.
    """

    def __init__(self, n_samples=600, d_features=10, n_classes=2, seed=42):
        rng = np.random.RandomState(seed)
        X = rng.randn(n_samples, d_features).astype(np.float32)

        r = X[:, 0] ** 2 + X[:, 1] ** 2
        y = (r > np.median(r)).astype(np.int64)

        X[:, 2] += 0.3 * X[:, 0] * X[:, 1]

        scales = np.logspace(1.5, 0, d_features).astype(np.float32)
        X *= scales[None, :]

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SyntheticRegressionDataset(Dataset):
    """
    Ill-conditioned regression: y = sin(2x₀) + 0.5·x₁² − x₂·x₃ + 0.3·cos(x₄) + ε,
    with stretched feature scales creating Hessian ill-conditioning.
    """

    def __init__(self, n_samples=600, d_features=10, seed=42):
        rng = np.random.RandomState(seed)
        X = rng.randn(n_samples, d_features).astype(np.float32)

        y = (
            np.sin(2 * X[:, 0])
            + 0.5 * X[:, 1] ** 2
            - X[:, 2] * X[:, 3]
            + 0.3 * np.cos(X[:, 4])
            + 0.1 * rng.randn(n_samples)
        ).astype(np.float32)

        scales = np.logspace(1.5, 0, d_features).astype(np.float32)
        X *= scales[None, :]

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(task="classification", batch_size=32, n_train=500, n_val=100,
                    d_features=10, seed=42):
    n_total = n_train + n_val
    if task == "classification":
        dataset = SyntheticClassificationDataset(n_total, d_features, seed=seed)
    else:
        dataset = SyntheticRegressionDataset(n_total, d_features, seed=seed)

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
