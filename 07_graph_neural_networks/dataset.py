"""
Synthetic graph datasets for GNN benchmarks.
"""

import numpy as np


def make_community_graph(n_nodes=200, n_communities=4, p_in=0.3, p_out=0.02, seed=42):
    """Stochastic block model: communities with high intra-, low inter-connectivity."""
    rng = np.random.RandomState(seed)
    labels = np.repeat(np.arange(n_communities), n_nodes // n_communities)
    n = len(labels)

    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if labels[i] == labels[j] else p_out
            if rng.rand() < p:
                W[i, j] = W[j, i] = 1.0

    features = rng.randn(n, 8) * 0.3
    for c in range(n_communities):
        mask = labels == c
        features[mask] += rng.randn(8) * 1.5

    return W, features, labels


def make_point_cloud_graph(n_points=300, k_neighbors=8, seed=42):
    """Points on concentric circles → KNN graph."""
    rng = np.random.RandomState(seed)
    n1 = n_points // 2
    n2 = n_points - n1

    theta1 = rng.uniform(0, 2 * np.pi, n1)
    r1 = 1.0 + rng.randn(n1) * 0.1
    X1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])

    theta2 = rng.uniform(0, 2 * np.pi, n2)
    r2 = 3.0 + rng.randn(n2) * 0.1
    X2 = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])

    X = np.vstack([X1, X2])
    labels = np.array([0] * n1 + [1] * n2)

    from models import build_knn_graph
    W = build_knn_graph(X, k=k_neighbors)

    return W, X, labels
