"""
Graph models with spectral and Nyström-approximated graph Laplacian operations.
"""

import numpy as np
from scipy.spatial.distance import cdist


def build_knn_graph(X, k=5):
    """Build k-nearest-neighbor adjacency matrix."""
    n = X.shape[0]
    dists = cdist(X, X, 'sqeuclidean')
    W = np.zeros((n, n))
    for i in range(n):
        neighbors = np.argsort(dists[i])[1:k+1]
        for j in neighbors:
            W[i, j] = np.exp(-dists[i, j])
            W[j, i] = W[i, j]
    return W


def graph_laplacian(W, normalized=True):
    """Compute graph Laplacian from adjacency matrix.
    L = D - W (unnormalized) or L = I - D^{-1/2} W D^{-1/2} (normalized)."""
    D = np.diag(W.sum(axis=1))
    if normalized:
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(W.sum(axis=1), 1e-10)))
        L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        L = D - W
    return L


def spectral_embedding(L, k):
    """Compute first k eigenvectors of Laplacian (spectral coordinates)."""
    eigvals, eigvecs = np.linalg.eigh(L)
    return eigvals[:k], eigvecs[:, :k]


class SpectralGNNConv:
    """Spectral graph convolution: g(L) = Σ θ_k λ_k u_k u_k^T.
    Full eigendecomposition: O(N³)."""

    def __init__(self, L, n_filters=16):
        self.eigvals, self.eigvecs = np.linalg.eigh(L)
        self.n_filters = n_filters

    def forward(self, X, theta):
        """Apply spectral filter. theta: filter coefficients for each eigenvalue."""
        g_lambda = np.zeros(len(self.eigvals))
        for k, t in enumerate(theta):
            g_lambda += t * self.eigvals ** k
        return self.eigvecs @ np.diag(g_lambda) @ self.eigvecs.T @ X


def heat_kernel(L, t=1.0):
    """Graph heat kernel: K = exp(-tL). This IS low-rank for large t."""
    eigvals, eigvecs = np.linalg.eigh(L)
    return eigvecs @ np.diag(np.exp(-t * eigvals)) @ eigvecs.T


class NystromGNNConv:
    """Nyström-approximated spectral convolution using graph KERNEL (not Laplacian).
    The heat kernel exp(-tL) has rapid eigenvalue decay → Nyström works well."""

    def __init__(self, L, rank=20, heat_t=2.0):
        n = L.shape[0]
        self.rank = min(rank, n)
        m = self.rank
        self.L = L

        # Nyström on the heat kernel K = exp(-tL), which IS low-rank
        K = heat_kernel(L, t=heat_t)
        idx = np.random.choice(n, m, replace=False)
        K_mm = K[np.ix_(idx, idx)] + 1e-6 * np.eye(m)
        K_nm = K[:, idx]

        eigvals, eigvecs = np.linalg.eigh(K_mm)
        eigvals = np.maximum(eigvals, 1e-8)

        self.approx_eigvals = eigvals
        self.approx_eigvecs = K_nm @ eigvecs @ np.diag(1.0 / np.sqrt(eigvals))

        norms = np.linalg.norm(self.approx_eigvecs, axis=0, keepdims=True)
        self.approx_eigvecs /= np.maximum(norms, 1e-10)

    def forward(self, X, theta=None):
        """Apply kernel-based smoothing filter."""
        return self.approx_eigvecs @ (self.approx_eigvecs.T @ X)


class NystromPreconditionedLaplacian:
    """Nyström preconditioner for solving (L + μI)x = b.
    Graph smoothing / semi-supervised learning requires solving Laplacian systems."""

    def __init__(self, L, mu=0.01, rank=20):
        n = L.shape[0]
        m = min(rank, n)
        self.mu = mu
        A = L + mu * np.eye(n)

        idx = np.random.choice(n, m, replace=False)
        Omega = np.random.randn(n, m)
        Y = A @ Omega
        Q, _ = np.linalg.qr(Y)
        B = Q.T @ A @ Q
        eigvals, eigvecs = np.linalg.eigh(B)
        eigvals = np.maximum(eigvals, 1e-8)

        self.U = Q @ eigvecs
        self.correction = 1.0 / (eigvals + mu) - 1.0 / mu

    def apply(self, r):
        Ur = self.U.T @ r
        return r / self.mu + self.U @ (self.correction * Ur)

    @staticmethod
    def pcg_solve(A, b, precond_fn=None, tol=1e-10, maxiter=500):
        x = np.zeros_like(b)
        r = b.copy(); z = precond_fn(r) if precond_fn else r.copy()
        p = z.copy(); b_norm = np.linalg.norm(b)
        residuals = [1.0]
        for _ in range(maxiter):
            Ap = A @ p; rz = r @ z
            alpha = rz / (p @ Ap + 1e-30)
            x += alpha * p; r -= alpha * Ap
            rel = np.linalg.norm(r) / b_norm; residuals.append(rel)
            if rel < tol: break
            z_new = precond_fn(r) if precond_fn else r.copy()
            beta = (r @ z_new) / (rz + 1e-30)
            p = z_new + beta * p; z = z_new
        return x, residuals
