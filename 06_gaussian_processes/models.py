"""
Gaussian Process models with exact and Nyström-approximated kernel matrices.
"""

import numpy as np
from scipy.spatial.distance import cdist


class RBFKernel:
    """Radial Basis Function (Gaussian) kernel: k(x,y) = σ² exp(-||x-y||²/(2l²))"""

    def __init__(self, length_scale=1.0, variance=1.0):
        self.length_scale = length_scale
        self.variance = variance

    def __call__(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        dists = cdist(X1, X2, 'sqeuclidean')
        return self.variance * np.exp(-dists / (2 * self.length_scale ** 2))


class ExactGP:
    """Exact Gaussian Process regression: O(N³) training, O(N²) prediction."""

    def __init__(self, kernel, noise=0.1):
        self.kernel = kernel
        self.noise = noise
        self.X_train = None
        self.alpha = None
        self.K = None

    def fit(self, X, y):
        self.X_train = X
        self.K = self.kernel(X) + self.noise ** 2 * np.eye(len(X))
        self.L = np.linalg.cholesky(self.K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
        return self

    def predict(self, X_test, return_std=False):
        K_star = self.kernel(X_test, self.X_train)
        mean = K_star @ self.alpha
        if return_std:
            v = np.linalg.solve(self.L, K_star.T)
            K_ss = self.kernel(X_test)
            var = np.diag(K_ss) - np.sum(v ** 2, axis=0)
            return mean, np.sqrt(np.maximum(var, 1e-10))
        return mean

    def log_marginal_likelihood(self):
        n = len(self.X_train)
        return -0.5 * (self.alpha @ (self.K @ self.alpha) +
                       2 * np.sum(np.log(np.diag(self.L))) +
                       n * np.log(2 * np.pi))


class NystromGP:
    """Nyström-approximated GP: O(Nm²) training, O(Nm) prediction."""

    def __init__(self, kernel, noise=0.1, rank=50):
        self.kernel = kernel
        self.noise = noise
        self.rank = rank

    def fit(self, X, y):
        self.X_train = X
        n = len(X)
        m = min(self.rank, n)

        idx = np.random.choice(n, m, replace=False)
        self.X_land = X[idx]

        # Nyström matrices
        K_mm = self.kernel(self.X_land) + 1e-6 * np.eye(m)
        K_nm = self.kernel(X, self.X_land)

        # Eigendecomposition of K_mm
        eigvals, eigvecs = np.linalg.eigh(K_mm)
        eigvals = np.maximum(eigvals, 1e-8)

        # Nyström features: Φ = K_nm @ U_m @ Λ_m^{-1/2}
        self.Phi = K_nm @ eigvecs @ np.diag(1.0 / np.sqrt(eigvals))

        # Solve (Φ^T Φ + σ²I)β = Φ^T y
        PhiTPhi = self.Phi.T @ self.Phi + self.noise ** 2 * np.eye(m)
        self.beta = np.linalg.solve(PhiTPhi, self.Phi.T @ y)

        self.K_mm = K_mm
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        return self

    def predict(self, X_test, return_std=False):
        K_test_m = self.kernel(X_test, self.X_land)
        Phi_test = K_test_m @ self.eigvecs @ np.diag(1.0 / np.sqrt(self.eigvals))
        mean = Phi_test @ self.beta
        if return_std:
            PhiTPhi = self.Phi.T @ self.Phi + self.noise ** 2 * np.eye(len(self.beta))
            PhiTPhi_inv = np.linalg.inv(PhiTPhi)
            var = self.noise ** 2 * (1 + np.sum((Phi_test @ PhiTPhi_inv) * Phi_test, axis=1))
            return mean, np.sqrt(np.maximum(var, 1e-10))
        return mean


class NystromPreconditionedGP:
    """GP with Nyström preconditioner for CG-based kernel solve.
    Uses Nyström to build M^{-1} ≈ (K + σ²I)^{-1}, then solves via PCG."""

    def __init__(self, kernel, noise=0.1, rank=50, cg_tol=1e-8, cg_maxiter=200):
        self.kernel = kernel
        self.noise = noise
        self.rank = rank
        self.cg_tol = cg_tol
        self.cg_maxiter = cg_maxiter

    def fit(self, X, y):
        self.X_train = X
        n = len(X)
        m = min(self.rank, n)

        self.K_full = self.kernel(X) + self.noise ** 2 * np.eye(n)

        idx = np.random.choice(n, m, replace=False)
        K_mm = self.kernel(X[idx]) + 1e-6 * np.eye(m)
        K_nm = self.kernel(X, X[idx])

        eigvals, eigvecs = np.linalg.eigh(K_mm)
        eigvals = np.maximum(eigvals, 1e-8)

        U_nys = K_nm @ eigvecs @ np.diag(1.0 / np.sqrt(eigvals))

        sigma_nys = np.sum(U_nys ** 2, axis=0) + self.noise ** 2
        correction = 1.0 / (sigma_nys) - 1.0 / self.noise ** 2

        def precond(r):
            Ur = U_nys.T @ r
            return r / self.noise ** 2 + U_nys @ (correction * Ur)

        self.alpha, self.residuals = self._pcg(self.K_full, y, precond)
        return self

    def _pcg(self, A, b, precond_fn):
        x = np.zeros_like(b)
        r = b.copy()
        z = precond_fn(r)
        p = z.copy()
        b_norm = np.linalg.norm(b)
        residuals = [np.linalg.norm(r) / b_norm]

        for _ in range(self.cg_maxiter):
            Ap = A @ p
            rz = r @ z
            alpha = rz / (p @ Ap + 1e-30)
            x += alpha * p
            r -= alpha * Ap
            rel = np.linalg.norm(r) / b_norm
            residuals.append(rel)
            if rel < self.cg_tol:
                break
            z_new = precond_fn(r)
            beta = (r @ z_new) / (rz + 1e-30)
            p = z_new + beta * p
            z = z_new
        return x, residuals

    def predict(self, X_test):
        K_star = self.kernel(X_test, self.X_train)
        return K_star @ self.alpha
