"""
Synthetic regression datasets for GP benchmarks.
"""

import numpy as np


def make_1d_regression(n_train=200, n_test=100, noise=0.1, seed=42):
    rng = np.random.RandomState(seed)
    X_train = np.sort(rng.uniform(-5, 5, n_train))[:, None]
    y_train = np.sin(X_train.ravel()) + 0.3 * np.cos(3 * X_train.ravel()) + rng.randn(n_train) * noise
    X_test = np.linspace(-5.5, 5.5, n_test)[:, None]
    y_test = np.sin(X_test.ravel()) + 0.3 * np.cos(3 * X_test.ravel())
    return X_train, y_train, X_test, y_test


def make_2d_regression(n_train=500, n_test=200, noise=0.1, seed=42):
    rng = np.random.RandomState(seed)
    X_train = rng.uniform(-3, 3, (n_train, 2))
    y_train = np.sin(X_train[:, 0]) * np.cos(X_train[:, 1]) + rng.randn(n_train) * noise
    grid = np.linspace(-3.5, 3.5, int(np.sqrt(n_test)))
    xx, yy = np.meshgrid(grid, grid)
    X_test = np.column_stack([xx.ravel(), yy.ravel()])
    y_test = np.sin(X_test[:, 0]) * np.cos(X_test[:, 1])
    return X_train, y_train, X_test, y_test


def make_scaling_data(n, d=1, noise=0.1, seed=42):
    """Create n-sample regression data for scaling experiments."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = np.sin(X @ np.ones(d)) + rng.randn(n) * noise
    return X, y
