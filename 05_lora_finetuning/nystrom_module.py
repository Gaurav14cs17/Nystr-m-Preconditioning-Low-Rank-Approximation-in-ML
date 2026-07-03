"""
Nyström ↔ LoRA connection module.

Establishes the formal correspondence:
  Full matrix : ΔW (weight update)            ↔  A (PDE operator)
  Low-rank    : BA (LoRA, rank r)              ↔  C W† C^T (Nyström, rank r)
  Why it works: singular value decay of ΔW     ↔  eigenvalue decay of A

Provides quantitative comparison of SVD-optimal and Nyström (column-sampling)
approximations applied to the Gram matrix G = ΔW ΔW^T (symmetric PSD),
which has eigenvalues σ_i² and directly connects to standard Nyström theory.
"""

import numpy as np
import torch
import torch.nn as nn


class NystromLoRAConnection:
    """Quantitative analysis of the LoRA–Nyström correspondence."""

    @staticmethod
    def analyze_weight_updates(model_before: nn.Module, model_after: nn.Module,
                               ranks=(1, 2, 4, 8, 16)):
        """Compute SVD of each layer's ΔW and report reconstruction error
        at each rank.  Returns dict[layer_name → stats]."""
        before = {n: p.detach().cpu().numpy()
                  for n, p in model_before.named_parameters()}
        after = {n: p.detach().cpu().numpy()
                 for n, p in model_after.named_parameters()}

        results = {}
        for name in before:
            if "weight" not in name or before[name].ndim != 2:
                continue
            dW = after[name] - before[name]
            full_norm = float(np.linalg.norm(dW, "fro"))
            if full_norm < 1e-12:
                continue

            U, S, Vt = np.linalg.svd(dW, full_matrices=False)
            cumul = np.cumsum(S ** 2) / np.sum(S ** 2)

            rank_data = {}
            for r in ranks:
                if r > min(dW.shape):
                    continue
                dW_r = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
                err = np.linalg.norm(dW - dW_r, "fro") / full_norm
                rank_data[r] = {
                    "svd_relative_error": float(err),
                    "energy_captured": float(cumul[r - 1]),
                }

            results[name] = {
                "shape": list(dW.shape),
                "singular_values": S.tolist(),
                "full_norm": full_norm,
                "rank_data": rank_data,
            }
        return results

    @staticmethod
    def nystrom_weight_approximation(delta_W, rank, n_trials: int = 20):
        """Compare Nyström approximation of the Gram matrix G = ΔW ΔW^T
        (symmetric PSD) with the eigendecomposition-optimal rank-r
        approximation.

        Standard Nyström is well-defined on symmetric PSD matrices; the
        Gram matrix's eigenvalues are σ_i² of ΔW, so both LoRA and
        Nyström exploit the same spectral decay.
        """
        if isinstance(delta_W, torch.Tensor):
            delta_W = delta_W.detach().cpu().numpy()

        m, n = delta_W.shape
        G = delta_W @ delta_W.T                       # m × m, symmetric PSD
        full_norm = np.linalg.norm(G, "fro")
        if full_norm < 1e-12:
            return {"svd_error": 0.0, "nystrom_error_mean": 0.0,
                    "nystrom_error_std": 0.0, "singular_values": []}

        eigvals_all, eigvecs_all = np.linalg.eigh(G)
        idx = np.argsort(eigvals_all)[::-1]
        eigvals_all = eigvals_all[idx]
        eigvecs_all = eigvecs_all[:, idx]

        actual_rank = min(rank, m)
        G_svd = (eigvecs_all[:, :actual_rank]
                 @ np.diag(eigvals_all[:actual_rank])
                 @ eigvecs_all[:, :actual_rank].T)
        svd_err = float(np.linalg.norm(G - G_svd, "fro") / full_norm)

        nystrom_errors = []
        for _ in range(n_trials):
            col_idx = np.random.choice(m, size=actual_rank, replace=False)
            C = G[:, col_idx]                              # m × rank
            W_block = G[np.ix_(col_idx, col_idx)]          # rank × rank
            reg = 1e-8 * np.trace(W_block) / actual_rank
            W_pinv = np.linalg.pinv(W_block + reg * np.eye(actual_rank))
            G_nys = C @ W_pinv @ C.T
            nys_err = np.linalg.norm(G - G_nys, "fro") / full_norm
            nystrom_errors.append(nys_err)

        _, S, _ = np.linalg.svd(delta_W, full_matrices=False)

        return {
            "svd_error": svd_err,
            "nystrom_error_mean": float(np.mean(nystrom_errors)),
            "nystrom_error_std": float(np.std(nystrom_errors)),
            "singular_values": S.tolist(),
        }
