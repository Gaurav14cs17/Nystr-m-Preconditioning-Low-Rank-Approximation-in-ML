"""
Nyström analysis tools for graph Laplacian.
"""

import numpy as np
import time


def laplacian_spectrum(L):
    """Eigenvalue analysis of graph Laplacian."""
    eigvals = np.sort(np.linalg.eigvalsh(L))
    nonzero = eigvals[eigvals > 1e-10]
    cumul = np.cumsum(eigvals ** 2) / np.sum(eigvals ** 2)
    r90 = int(np.searchsorted(cumul, 0.9) + 1)
    r99 = int(np.searchsorted(cumul, 0.99) + 1)
    return {
        'eigenvalues': eigvals,
        'cumulative_energy': cumul,
        'rank_90': r90,
        'rank_99': r99,
        'algebraic_connectivity': nonzero[0] if len(nonzero) > 0 else 0,
        'spectral_gap': nonzero[0] / nonzero[-1] if len(nonzero) > 1 else 0,
    }


def nystrom_eigenvector_error(L, rank, n_trials=5):
    """Compare Nyström approximate eigenvectors to exact ones."""
    n = L.shape[0]
    m = min(rank, n)
    exact_vals, exact_vecs = np.linalg.eigh(L)

    errors = []
    for _ in range(n_trials):
        idx = np.random.choice(n, m, replace=False)
        L_mm = L[np.ix_(idx, idx)]
        L_nm = L[:, idx]

        eigvals, eigvecs = np.linalg.eigh(L_mm)
        eigvals = np.maximum(eigvals, 1e-8)

        approx_vecs = L_nm @ eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
        norms = np.linalg.norm(approx_vecs, axis=0, keepdims=True)
        approx_vecs /= np.maximum(norms, 1e-10)

        P_exact = exact_vecs[:, :m] @ exact_vecs[:, :m].T
        P_approx = approx_vecs @ approx_vecs.T
        err = np.linalg.norm(P_exact - P_approx, 'fro') / np.linalg.norm(P_exact, 'fro')
        errors.append(err)

    return np.mean(errors), np.std(errors)
