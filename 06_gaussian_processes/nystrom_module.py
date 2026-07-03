"""
Nyström kernel analysis utilities for GP benchmarks.
"""

import numpy as np
import time


def kernel_spectrum(K):
    """Eigenvalue analysis of kernel matrix."""
    eigvals = np.sort(np.abs(np.linalg.eigvalsh(K)))[::-1]
    cumul = np.cumsum(eigvals ** 2) / np.sum(eigvals ** 2)
    r90 = int(np.searchsorted(cumul, 0.9) + 1)
    r99 = int(np.searchsorted(cumul, 0.99) + 1)
    return {
        'eigenvalues': eigvals,
        'cumulative_energy': cumul,
        'rank_90': r90,
        'rank_99': r99,
        'condition_number': eigvals[0] / (eigvals[-1] + 1e-15),
    }


def nystrom_kernel_error(K_full, n_landmarks, n_trials=5):
    """Measure Nyström approximation error for a kernel matrix."""
    n = K_full.shape[0]
    m = min(n_landmarks, n)
    errors = []
    for _ in range(n_trials):
        idx = np.random.choice(n, m, replace=False)
        K_mm = K_full[np.ix_(idx, idx)] + 1e-6 * np.eye(m)
        K_nm = K_full[:, idx]
        K_approx = K_nm @ np.linalg.solve(K_mm, K_nm.T)
        err = np.linalg.norm(K_full - K_approx, 'fro') / np.linalg.norm(K_full, 'fro')
        errors.append(err)
    return np.mean(errors), np.std(errors)


def compare_gp_methods(exact_gp, nystrom_gp, precond_gp, X_train, y_train, X_test, y_test):
    """Compare 3 GP methods: exact, Nyström approximate, Nyström-preconditioned CG."""
    results = {}

    # Exact GP
    t0 = time.perf_counter()
    exact_gp.fit(X_train, y_train)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred_exact = exact_gp.predict(X_test)
    t_pred = time.perf_counter() - t0
    rmse = np.sqrt(np.mean((y_pred_exact - y_test) ** 2))
    results['Exact GP'] = {
        'fit_ms': t_fit * 1000, 'pred_ms': t_pred * 1000,
        'rmse': rmse, 'memory': f'O(N²)', 'complexity': f'O(N³)',
    }

    # Nyström GP
    t0 = time.perf_counter()
    nystrom_gp.fit(X_train, y_train)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred_ny = nystrom_gp.predict(X_test)
    t_pred = time.perf_counter() - t0
    rmse = np.sqrt(np.mean((y_pred_ny - y_test) ** 2))
    results[f'Nyström GP (r={nystrom_gp.rank})'] = {
        'fit_ms': t_fit * 1000, 'pred_ms': t_pred * 1000,
        'rmse': rmse, 'memory': f'O(Nm)', 'complexity': f'O(Nm²)',
    }

    # Preconditioned CG GP
    t0 = time.perf_counter()
    precond_gp.fit(X_train, y_train)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred_pcg = precond_gp.predict(X_test)
    t_pred = time.perf_counter() - t0
    rmse = np.sqrt(np.mean((y_pred_pcg - y_test) ** 2))
    results[f'PCG GP (r={precond_gp.rank})'] = {
        'fit_ms': t_fit * 1000, 'pred_ms': t_pred * 1000,
        'rmse': rmse, 'memory': f'O(Nm)', 'complexity': f'O(N·iter)',
        'cg_iters': len(precond_gp.residuals) - 1,
    }

    return results
