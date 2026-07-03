import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class NystromPreconditionedCG:
    """Nyström preconditioner for regularized least-squares: (A^T A + lambda*I) x = b."""

    def __init__(self, AtA, lam, rank=20):
        self.n = AtA.shape[0]
        self.lam = lam
        self.rank = rank
        self._build(AtA)

    def _build(self, AtA):
        n, rank = self.n, self.rank
        # Randomized SVD-style: multiply by random probe, QR, project
        Omega = np.random.randn(n, rank)
        Y = AtA @ Omega
        Q, _ = np.linalg.qr(Y)
        B = Q.T @ AtA @ Q
        eigvals, eigvecs = np.linalg.eigh(B)
        eigvals = np.maximum(eigvals, 1e-10)
        self.U = Q @ eigvecs
        self.eig = eigvals
        # Preconditioner: deflate the top eigenspace of (AtA + lam*I)
        # P^{-1} x = (1/lam) x + U diag(1/(sigma_i+lam) - 1/lam) U^T x
        self.correction_scale = 1.0 / (eigvals + self.lam) - 1.0 / self.lam

    def apply(self, r):
        Utr = self.U.T @ r
        return r / self.lam + self.U @ (self.correction_scale * Utr)

    @staticmethod
    def cg_solve(A, b, precond_fn=None, tol=1e-10, maxiter=2000):
        x = np.zeros_like(b)
        r = b.copy()
        z = precond_fn(r) if precond_fn else r.copy()
        p = z.copy()
        b_norm = np.linalg.norm(b)
        residuals = [np.linalg.norm(r) / b_norm]

        for _ in range(maxiter):
            Ap = A @ p
            rz = r @ z
            alpha = rz / (p @ Ap + 1e-30)
            x += alpha * p
            r -= alpha * Ap
            rel = np.linalg.norm(r) / b_norm
            residuals.append(rel)
            if rel < tol:
                break
            z_new = precond_fn(r) if precond_fn else r.copy()
            beta = (r @ z_new) / (rz + 1e-30)
            p = z_new + beta * p
            z = z_new
        return x, residuals


def compare_preconditioners(A, b, lambda_reg, nystrom_rank=20):
    """Compare CG convergence: no preconditioner, Jacobi, Nyström, ILU."""
    n = A.shape[0]
    AtA = A.T @ A + lambda_reg * np.eye(n)
    rhs = AtA @ b
    kappa = np.linalg.cond(AtA)

    solve = NystromPreconditionedCG.cg_solve
    results = {}

    # Plain CG
    t0 = time.perf_counter()
    _, res_cg = solve(AtA, rhs)
    t_cg = time.perf_counter() - t0
    results['CG'] = {'iters': len(res_cg) - 1, 'time_ms': t_cg * 1000, 'residuals': res_cg}

    # Jacobi
    diag = np.diag(AtA)
    jac_fn = lambda r: r / diag
    t0 = time.perf_counter()
    _, res_jac = solve(AtA, rhs, precond_fn=jac_fn)
    t_jac = time.perf_counter() - t0
    results['Jacobi'] = {'iters': len(res_jac) - 1, 'time_ms': t_jac * 1000, 'residuals': res_jac}

    # Nyström
    npc = NystromPreconditionedCG(AtA - lambda_reg * np.eye(n), lambda_reg, rank=nystrom_rank)
    t0 = time.perf_counter()
    _, res_ny = solve(AtA, rhs, precond_fn=npc.apply)
    t_ny = time.perf_counter() - t0
    results[f'Nystrom-{nystrom_rank}'] = {'iters': len(res_ny) - 1, 'time_ms': t_ny * 1000, 'residuals': res_ny}

    # ILU
    ilu = spla.spilu(sp.csc_matrix(AtA))
    ilu_fn = lambda r: ilu.solve(r)
    t0 = time.perf_counter()
    _, res_ilu = solve(AtA, rhs, precond_fn=ilu_fn)
    t_ilu = time.perf_counter() - t0
    results['ILU'] = {'iters': len(res_ilu) - 1, 'time_ms': t_ilu * 1000, 'residuals': res_ilu}

    return results, kappa
