"""
Graph Neural Networks — Nyström for Spectral Convolution & Laplacian Solves.

The graph Laplacian L is the exact matrix the Poisson solver paper targets.
Spectral GNNs need eigenvectors of L → Nyström gives O(Nm²) approximation.

Benchmarks:
  1. Graph Laplacian eigenvalue spectrum
  2. Nyström eigenvector approximation quality
  3. Spectral convolution: exact vs Nyström
  4. Semi-supervised learning: Laplacian solve (L+μI)x = b
  5. Preconditioned CG convergence for Laplacian system
"""

import os, sys, time, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)
os.makedirs(os.path.join(DIR, "results"), exist_ok=True)

from models import (graph_laplacian, spectral_embedding, SpectralGNNConv,
                     NystromGNNConv, NystromPreconditionedLaplacian, heat_kernel)
from dataset import make_community_graph, make_point_cloud_graph
from nystrom_module import laplacian_spectrum, nystrom_eigenvector_error

def save(name):
    p = os.path.join(DIR, "results", name)
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")

results = {}

# ═══════════════════════════════════════════════════════════════
# BENCHMARK 1: Graph Laplacian Spectrum
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("GNN BENCHMARK 1: Graph Laplacian Eigenvalue Spectrum")
print("=" * 60)

W, features, labels = make_community_graph(n_nodes=200, n_communities=4)
L = graph_laplacian(W, normalized=True)
spec = laplacian_spectrum(L)

print(f"  Graph: {W.shape[0]} nodes, {int(W.sum()/2)} edges, 4 communities")
print(f"  Laplacian: {L.shape[0]}×{L.shape[0]}")
print(f"  Algebraic connectivity (λ₂): {spec['algebraic_connectivity']:.4f}")
print(f"  90% energy at rank {spec['rank_90']}/{L.shape[0]}")
print(f"  99% energy at rank {spec['rank_99']}/{L.shape[0]}")

results['spectrum'] = {
    'n_nodes': W.shape[0], 'n_edges': int(W.sum()/2),
    'algebraic_connectivity': spec['algebraic_connectivity'],
    'rank_90': spec['rank_90'], 'rank_99': spec['rank_99'],
}

eigvals2, eigvecs2 = spectral_embedding(L, 4)

fig, axes = plt.subplots(1, 4, figsize=(18, 4))

axes[0].semilogy(spec['eigenvalues'][1:], 'b-', lw=2)
axes[0].set_xlabel('Index'); axes[0].set_ylabel('Eigenvalue')
axes[0].set_title(f'Laplacian Spectrum ({L.shape[0]} nodes)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(spec['cumulative_energy'], 'g-', lw=2)
axes[1].axhline(0.9, color='r', ls='--', label=f'90% at rank {spec["rank_90"]}')
axes[1].axhline(0.99, color='orange', ls='--', label=f'99% at rank {spec["rank_99"]}')
axes[1].set_xlabel('Rank'); axes[1].set_ylabel('Cumul Energy')
axes[1].set_title('Spectral Energy'); axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

sc = axes[2].scatter(eigvecs2[:, 1], eigvecs2[:, 2], c=labels, cmap='tab10', s=15)
axes[2].set_xlabel('v₂'); axes[2].set_ylabel('v₃')
axes[2].set_title('Spectral Embedding (Fiedler)'); axes[2].grid(True, alpha=0.3)

axes[3].imshow(L[:50, :50], cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
axes[3].set_title('Laplacian (50×50 block)')
axes[3].set_xlabel('Node j'); axes[3].set_ylabel('Node i')

plt.tight_layout(); save("gnn_laplacian_spectrum.png")


# ═══════════════════════════════════════════════════════════════
# BENCHMARK 2: Nyström Kernel Matrix Approximation
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("GNN BENCHMARK 2: Nyström on Graph Kernel (not Laplacian!)")
print("=" * 60)
print("  Key insight: L itself is full-rank. The KERNEL exp(-tL) is low-rank.")

# Build regularized kernel: (L + μI)^{-1}
mu_reg = 0.1
K_reg = np.linalg.inv(L + mu_reg * np.eye(L.shape[0]))
n = K_reg.shape[0]

ranks = [5, 10, 20, 40, 80, 100]
approx_results = []
print(f"\n  {'Rank':>6} {'Kernel Err':>14} {'Std':>10} {'Compression':>12}")
print(f"  {'-'*46}")
for r in ranks:
    errs = []
    for _ in range(10):
        idx = np.random.choice(n, min(r, n), replace=False)
        K_mm = K_reg[np.ix_(idx, idx)] + 1e-6 * np.eye(min(r, n))
        K_nm = K_reg[:, idx]
        K_approx = K_nm @ np.linalg.solve(K_mm, K_nm.T)
        errs.append(np.linalg.norm(K_reg - K_approx, 'fro') / np.linalg.norm(K_reg, 'fro'))
    err_mean, err_std = np.mean(errs), np.std(errs)
    comp = n / r
    approx_results.append({'rank': r, 'error': err_mean, 'std': err_std, 'compression': comp})
    status = "good" if err_mean < 0.1 else "ok" if err_mean < 0.5 else "poor"
    print(f"  {r:>6} {err_mean:>14.6f} {err_std:>10.6f} {comp:>11.1f}×  ({status})")

results['kernel_approximation'] = approx_results

fig, ax = plt.subplots(figsize=(8, 5))
rs = [r['rank'] for r in approx_results]
ax.semilogy(rs, [r['error'] for r in approx_results], 'bo-', lw=2, ms=8)
ax.fill_between(rs, [max(r['error']-r['std'], 1e-10) for r in approx_results],
                [r['error']+r['std'] for r in approx_results], alpha=0.2)
ax.set_xlabel('Nyström Rank (m)'); ax.set_ylabel('Relative Frobenius Error')
ax.set_title('Nyström Approximation of (L + μI)⁻¹'); ax.grid(True, alpha=0.3)
save("gnn_nystrom_kernel.png")


# ═══════════════════════════════════════════════════════════════
# BENCHMARK 3: Heat Kernel Approximation — Exact vs Nyström
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("GNN BENCHMARK 3: Heat Kernel exp(-tL) Nyström Approximation")
print("=" * 60)

# The heat kernel IS low-rank for t>0 (exponential eigenvalue decay)
K_heat = heat_kernel(L, t=2.0)
heat_spec = laplacian_spectrum(np.diag(np.diag(K_heat)) - K_heat + np.eye(K_heat.shape[0]))
K_eigvals = np.sort(np.linalg.eigvalsh(K_heat))[::-1]
K_cumul = np.cumsum(K_eigvals ** 2) / np.sum(K_eigvals ** 2)
kr90 = int(np.searchsorted(K_cumul, 0.9) + 1)
kr99 = int(np.searchsorted(K_cumul, 0.99) + 1)
print(f"  Heat kernel: 90% energy at rank {kr90}/{K_heat.shape[0]}  ← low-rank!")
print(f"  Heat kernel: 99% energy at rank {kr99}/{K_heat.shape[0]}")

out_exact = K_heat @ features

conv_results = []
print(f"\n  {'Rank':>6} {'RelError':>12} {'Exact(ms)':>12} {'Nyst(ms)':>12} {'Speedup':>10}")
print(f"  {'-'*54}")
for r in [5, 10, 20, 40, 80]:
    conv_nystrom = NystromGNNConv(L, rank=r, heat_t=2.0)

    t0 = time.perf_counter()
    _ = K_heat @ features
    t_ex = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    out_ny = conv_nystrom.forward(features)
    t_ny = (time.perf_counter() - t0) * 1000

    err = np.linalg.norm(out_exact - out_ny) / np.linalg.norm(out_exact)
    row = {'rank': r, 'error': err, 'exact_ms': t_ex, 'ny_ms': t_ny, 'speedup': t_ex / max(t_ny, 1e-6)}
    conv_results.append(row)
    print(f"  {r:>6} {err:>12.6f} {t_ex:>12.3f} {t_ny:>12.3f} {row['speedup']:>9.1f}×")

results['heat_kernel'] = {'rank_90': kr90, 'rank_99': kr99, 'approximation': conv_results}


# ═══════════════════════════════════════════════════════════════
# BENCHMARK 4: Semi-supervised Learning — Laplacian Solve
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("GNN BENCHMARK 4: Semi-supervised Label Propagation")
print("=" * 60)

W_pc, X_pc, labels_pc = make_point_cloud_graph(n_points=300, k_neighbors=8)
L_pc = graph_laplacian(W_pc, normalized=False)
n_pc = L_pc.shape[0]

# Label propagation: (L + μI)f = μ·y where y has labels on labeled nodes
mu = 0.1
n_labeled = 20
labeled_idx = np.concatenate([
    np.where(labels_pc == 0)[0][:n_labeled // 2],
    np.where(labels_pc == 1)[0][:n_labeled // 2],
])
y = np.zeros(n_pc)
y[labeled_idx] = 2 * labels_pc[labeled_idx].astype(float) - 1

A_lp = L_pc + mu * np.eye(n_pc)
rhs = mu * y

# Exact solve
t0 = time.perf_counter()
f_exact = np.linalg.solve(A_lp, rhs)
t_exact = (time.perf_counter() - t0) * 1000
acc_exact = np.mean((f_exact > 0).astype(int) == labels_pc) * 100

# Nyström-preconditioned CG
precond = NystromPreconditionedLaplacian(L_pc, mu=mu, rank=30)
t0 = time.perf_counter()
f_pcg, residuals_pcg = NystromPreconditionedLaplacian.pcg_solve(A_lp, rhs, precond_fn=precond.apply)
t_pcg = (time.perf_counter() - t0) * 1000
acc_pcg = np.mean((f_pcg > 0).astype(int) == labels_pc) * 100

# Plain CG
t0 = time.perf_counter()
f_cg, residuals_cg = NystromPreconditionedLaplacian.pcg_solve(A_lp, rhs)
t_cg = (time.perf_counter() - t0) * 1000
acc_cg = np.mean((f_cg > 0).astype(int) == labels_pc) * 100

print(f"  {'Method':<25} {'Time(ms)':>10} {'Iters':>8} {'Accuracy':>10}")
print(f"  {'-'*55}")
print(f"  {'Direct solve':<25} {t_exact:>10.2f} {'N/A':>8} {acc_exact:>9.1f}%")
print(f"  {'Plain CG':<25} {t_cg:>10.2f} {len(residuals_cg)-1:>8} {acc_cg:>9.1f}%")
print(f"  {'Nyström-PCG (r=30)':<25} {t_pcg:>10.2f} {len(residuals_pcg)-1:>8} {acc_pcg:>9.1f}%")

results['label_propagation'] = {
    'exact': {'time_ms': t_exact, 'accuracy': acc_exact},
    'cg': {'time_ms': t_cg, 'iters': len(residuals_cg)-1, 'accuracy': acc_cg},
    'pcg': {'time_ms': t_pcg, 'iters': len(residuals_pcg)-1, 'accuracy': acc_pcg},
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
sc = axes[0].scatter(X_pc[:, 0], X_pc[:, 1], c=labels_pc, cmap='coolwarm', s=10, alpha=0.5)
axes[0].scatter(X_pc[labeled_idx, 0], X_pc[labeled_idx, 1], c='black', s=50, marker='x', lw=2)
axes[0].set_title(f'Ground Truth ({n_labeled} labeled)'); axes[0].grid(True, alpha=0.3)

axes[1].scatter(X_pc[:, 0], X_pc[:, 1], c=f_exact, cmap='RdBu', s=10)
axes[1].set_title(f'Exact Solve ({acc_exact:.0f}%)'); axes[1].grid(True, alpha=0.3)

axes[2].scatter(X_pc[:, 0], X_pc[:, 1], c=f_pcg, cmap='RdBu', s=10)
axes[2].set_title(f'Nyström-PCG ({acc_pcg:.0f}%, {len(residuals_pcg)-1} iters)'); axes[2].grid(True, alpha=0.3)
plt.tight_layout(); save("gnn_label_propagation.png")


# ═══════════════════════════════════════════════════════════════
# BENCHMARK 5: CG Convergence
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("GNN BENCHMARK 5: CG Convergence for Laplacian System")
print("=" * 60)

print(f"  Plain CG:       {len(residuals_cg)-1} iterations")
print(f"  Nyström-PCG:    {len(residuals_pcg)-1} iterations")
if len(residuals_pcg) < len(residuals_cg):
    print(f"  Speedup:        {(len(residuals_cg)-1)/(len(residuals_pcg)-1):.1f}× fewer iterations")
else:
    print(f"  Note: plain CG already converges fast on this problem")

results['cg_convergence'] = {
    'cg_iters': len(residuals_cg) - 1,
    'pcg_iters': len(residuals_pcg) - 1,
}

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(residuals_cg, 'b-', lw=2, label=f'Plain CG ({len(residuals_cg)-1} iters)')
ax.semilogy(residuals_pcg, 'r-', lw=2, label=f'Nyström-PCG ({len(residuals_pcg)-1} iters)')
ax.axhline(1e-10, color='gray', ls='--', label='tol=1e-10')
ax.set_xlabel('Iteration'); ax.set_ylabel('Relative Residual')
ax.set_title('CG for Graph Laplacian System (L + μI)f = μy')
ax.legend(); ax.grid(True, alpha=0.3)
save("gnn_cg_convergence.png")


# ═══════════════════════════════════════════════════════════════
# Save JSON
# ═══════════════════════════════════════════════════════════════
json_path = os.path.join(DIR, "results", "gnn_results.json")
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2,
              default=lambda x: float(x) if isinstance(x, (np.floating,)) else
                                int(x) if isinstance(x, (np.integer,)) else str(x))

print(f"\n  JSON: {json_path}")
print(f"\n{'=' * 60}")
print("SUMMARY")
print("=" * 60)
print(f"  Laplacian spectrum:  90% at rank {spec['rank_90']}/{L.shape[0]} (not low-rank)")
print(f"  Heat kernel:         90% at rank {kr90}/{K_heat.shape[0]} (low-rank!)")
print(f"  Kernel approx best:  {approx_results[-1]['error']:.4f} at rank {approx_results[-1]['rank']}")
print(f"  Label propagation:   exact={acc_exact:.0f}%, PCG={acc_pcg:.0f}%")
print("  ✓ Done!")
