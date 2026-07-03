"""
Gaussian Processes — THE canonical Nyström application in ML.
Kernel matrix K has rapid eigenvalue decay → Nyström gives O(Nm²) instead of O(N³).

Benchmarks:
  1. Kernel matrix eigenvalue spectrum (RBF kernel)
  2. Nyström kernel approximation quality vs rank
  3. Exact GP vs Nyström GP vs Nyström-Preconditioned CG GP
  4. Scaling comparison (N = 100 to 2000)
  5. Preconditioned CG convergence
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

from models import RBFKernel, ExactGP, NystromGP, NystromPreconditionedGP
from dataset import make_1d_regression, make_scaling_data
from nystrom_module import kernel_spectrum, nystrom_kernel_error, compare_gp_methods

def save(name):
    p = os.path.join(DIR, "results", name)
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")

results = {}

# ═══════════════════════════════════════════════════════════════
# BENCHMARK 1: Kernel Matrix Eigenvalue Spectrum
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("GP BENCHMARK 1: RBF Kernel Eigenvalue Spectrum")
print("=" * 60)

X_train, y_train, X_test, y_test = make_1d_regression(n_train=300, noise=0.1)
kernel = RBFKernel(length_scale=1.0, variance=1.0)
K = kernel(X_train)

spec = kernel_spectrum(K)
print(f"  Kernel matrix: {K.shape[0]}×{K.shape[0]}")
print(f"  Condition number: {spec['condition_number']:.0f}")
print(f"  90% energy at rank {spec['rank_90']}/{K.shape[0]}  ← low-rank!")
print(f"  99% energy at rank {spec['rank_99']}/{K.shape[0]}")

results['spectrum'] = {
    'n': K.shape[0], 'kappa': spec['condition_number'],
    'rank_90': spec['rank_90'], 'rank_99': spec['rank_99'],
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].semilogy(spec['eigenvalues'], 'b-', lw=2)
axes[0].set_xlabel('Index'); axes[0].set_ylabel('|Eigenvalue|')
axes[0].set_title(f'RBF Kernel Spectrum ({K.shape[0]}×{K.shape[0]})')
axes[0].axhline(spec['eigenvalues'][0] * 0.01, color='r', ls='--', label='1% of max')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(spec['cumulative_energy'], 'g-', lw=2)
axes[1].axhline(0.9, color='r', ls='--', label=f'90% at rank {spec["rank_90"]}')
axes[1].axhline(0.99, color='orange', ls='--', label=f'99% at rank {spec["rank_99"]}')
axes[1].set_xlabel('Rank'); axes[1].set_ylabel('Cumulative Energy')
axes[1].set_title('Spectral Energy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].imshow(K[:60, :60], cmap='viridis', aspect='auto')
axes[2].set_title('Kernel Matrix (60×60 block)')
axes[2].set_xlabel('Sample j'); axes[2].set_ylabel('Sample i')
plt.colorbar(axes[2].images[0], ax=axes[2])
plt.tight_layout(); save("gp_kernel_spectrum.png")


# ═══════════════════════════════════════════════════════════════
# BENCHMARK 2: Nyström Kernel Approximation Quality
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("GP BENCHMARK 2: Nyström Kernel Approximation Quality")
print("=" * 60)

ranks = [5, 10, 20, 40, 80, 150]
approx_results = []
print(f"\n  {'Rank':>6} {'Mean Error':>12} {'Std':>10} {'Compression':>12}")
print(f"  {'-'*44}")
for r in ranks:
    err_mean, err_std = nystrom_kernel_error(K, r, n_trials=10)
    comp = K.shape[0] / r
    approx_results.append({'rank': r, 'error': err_mean, 'std': err_std, 'compression': comp})
    status = "✓" if err_mean < 0.1 else "~" if err_mean < 0.5 else "✗"
    print(f"  {r:>6} {err_mean:>12.6f} {err_std:>10.6f} {comp:>11.1f}× {status}")

results['approximation'] = approx_results

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
rs = [r['rank'] for r in approx_results]
ax1.semilogy(rs, [r['error'] for r in approx_results], 'bo-', lw=2, ms=8)
ax1.fill_between(rs, [r['error'] - r['std'] for r in approx_results],
                 [r['error'] + r['std'] for r in approx_results], alpha=0.2)
ax1.set_xlabel('Nyström Rank (m)'); ax1.set_ylabel('Relative Frobenius Error')
ax1.set_title('Kernel Approximation Error'); ax1.grid(True, alpha=0.3)

ax2.bar(range(len(rs)), [r['compression'] for r in approx_results], color='green', alpha=0.7)
ax2.set_xticks(range(len(rs))); ax2.set_xticklabels([str(r) for r in rs])
ax2.set_xlabel('Nyström Rank'); ax2.set_ylabel('Compression (N/m)')
ax2.set_title('Memory Reduction'); ax2.grid(True, alpha=0.3)
plt.tight_layout(); save("gp_nystrom_approximation.png")


# ═══════════════════════════════════════════════════════════════
# BENCHMARK 3: GP Prediction Comparison
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("GP BENCHMARK 3: Exact vs Nyström vs PCG GP Prediction")
print("=" * 60)

exact = ExactGP(kernel, noise=0.1)
nystrom = NystromGP(kernel, noise=0.1, rank=30)
precond = NystromPreconditionedGP(kernel, noise=0.1, rank=30)

gp_results = compare_gp_methods(exact, nystrom, precond, X_train, y_train, X_test, y_test)

print(f"\n  {'Method':<25} {'Fit(ms)':>10} {'Pred(ms)':>10} {'RMSE':>10} {'Complexity':>12}")
print(f"  {'-'*70}")
for name, data in gp_results.items():
    extra = f"  CG:{data['cg_iters']}it" if 'cg_iters' in data else ""
    print(f"  {name:<25} {data['fit_ms']:>10.2f} {data['pred_ms']:>10.2f} "
          f"{data['rmse']:>10.6f} {data['complexity']:>12}{extra}")

results['prediction'] = {k: {kk: vv for kk, vv in v.items()} for k, v in gp_results.items()}

# Prediction plot
exact.fit(X_train, y_train)
nystrom.fit(X_train, y_train)
precond.fit(X_train, y_train)

y_exact, std_exact = exact.predict(X_test, return_std=True)
y_ny, std_ny = nystrom.predict(X_test, return_std=True)
y_pcg = precond.predict(X_test)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, name, y_pred, color in [
    (axes[0], 'Exact GP', y_exact, 'blue'),
    (axes[1], f'Nyström GP (r=30)', y_ny, 'red'),
    (axes[2], f'PCG GP (r=30)', y_pcg, 'green'),
]:
    ax.plot(X_test.ravel(), y_test, 'k--', lw=1, alpha=0.5, label='True')
    ax.scatter(X_train.ravel(), y_train, s=5, c='gray', alpha=0.3, label='Train')
    ax.plot(X_test.ravel(), y_pred, color=color, lw=2, label=name)
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    ax.set_title(f'{name}\nRMSE = {rmse:.4f}')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlabel('x'); ax.set_ylabel('y')
plt.tight_layout(); save("gp_prediction_comparison.png")


# ═══════════════════════════════════════════════════════════════
# BENCHMARK 4: Scaling Comparison
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("GP BENCHMARK 4: Scaling — Exact O(N³) vs Nyström O(Nm²)")
print("=" * 60)

sizes = [100, 200, 400, 800, 1500]
rank_fixed = 30
scaling = []

print(f"\n  {'N':>6} {'Exact(ms)':>12} {'Nyst(ms)':>12} {'PCG(ms)':>12} {'Speedup':>10} {'ΔRMSE':>10}")
print(f"  {'-'*64}")

for n in sizes:
    X, y = make_scaling_data(n, d=1, noise=0.1)
    X_te, y_te = make_scaling_data(50, d=1, noise=0.0, seed=99)

    t0 = time.perf_counter()
    gp_ex = ExactGP(kernel, noise=0.1).fit(X, y)
    t_exact = (time.perf_counter() - t0) * 1000
    y_ex = gp_ex.predict(X_te)
    rmse_ex = np.sqrt(np.mean((y_ex - y_te) ** 2))

    t0 = time.perf_counter()
    gp_ny = NystromGP(kernel, noise=0.1, rank=rank_fixed).fit(X, y)
    t_ny = (time.perf_counter() - t0) * 1000
    y_ny = gp_ny.predict(X_te)
    rmse_ny = np.sqrt(np.mean((y_ny - y_te) ** 2))

    t0 = time.perf_counter()
    gp_pcg = NystromPreconditionedGP(kernel, noise=0.1, rank=rank_fixed).fit(X, y)
    t_pcg = (time.perf_counter() - t0) * 1000
    y_pcg = gp_pcg.predict(X_te)
    rmse_pcg = np.sqrt(np.mean((y_pcg - y_te) ** 2))

    row = {'n': n, 'exact_ms': t_exact, 'ny_ms': t_ny, 'pcg_ms': t_pcg,
           'speedup': t_exact / t_ny, 'rmse_exact': rmse_ex,
           'rmse_ny': rmse_ny, 'rmse_pcg': rmse_pcg}
    scaling.append(row)
    print(f"  {n:>6} {t_exact:>12.1f} {t_ny:>12.1f} {t_pcg:>12.1f} "
          f"{t_exact/t_ny:>9.1f}× {abs(rmse_ny - rmse_ex):>10.6f}")

results['scaling'] = scaling

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ns = [r['n'] for r in scaling]
ax1.loglog(ns, [r['exact_ms'] for r in scaling], 'bo-', lw=2, ms=8, label='Exact O(N³)')
ax1.loglog(ns, [r['ny_ms'] for r in scaling], 'rs-', lw=2, ms=8, label=f'Nyström O(Nm²)')
ax1.loglog(ns, [r['pcg_ms'] for r in scaling], 'g^-', lw=2, ms=8, label=f'PCG O(N·iter)')
ax1.set_xlabel('N (training samples)'); ax1.set_ylabel('Time (ms)')
ax1.set_title('GP Training Time Scaling'); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(ns, [r['rmse_exact'] for r in scaling], 'bo-', lw=2, ms=8, label='Exact')
ax2.plot(ns, [r['rmse_ny'] for r in scaling], 'rs-', lw=2, ms=8, label='Nyström')
ax2.plot(ns, [r['rmse_pcg'] for r in scaling], 'g^-', lw=2, ms=8, label='PCG')
ax2.set_xlabel('N'); ax2.set_ylabel('RMSE')
ax2.set_title('Prediction Quality vs N'); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout(); save("gp_scaling_comparison.png")


# ═══════════════════════════════════════════════════════════════
# BENCHMARK 5: Preconditioned CG Convergence
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("GP BENCHMARK 5: CG vs Nyström-Preconditioned CG Convergence")
print("=" * 60)

X_cg, y_cg = make_scaling_data(500, d=1, noise=0.1)
K_cg = kernel(X_cg) + 0.01 * np.eye(500)

# Plain CG
from models import NystromPreconditionedGP
x_cg = np.zeros_like(y_cg)
r = y_cg.copy(); p = r.copy()
b_norm = np.linalg.norm(y_cg)
cg_residuals = [1.0]
for _ in range(200):
    Ap = K_cg @ p; rr = r @ r; alpha = rr / (p @ Ap + 1e-30)
    x_cg += alpha * p; r -= alpha * Ap
    rel = np.linalg.norm(r) / b_norm; cg_residuals.append(rel)
    if rel < 1e-8: break
    beta = (r @ r) / (rr + 1e-30); p = r + beta * p

# Nyström-preconditioned CG
pcg = NystromPreconditionedGP(kernel, noise=0.1, rank=30)
pcg.fit(X_cg, y_cg)

print(f"  Plain CG iterations: {len(cg_residuals) - 1}")
print(f"  Nyström-PCG iterations: {len(pcg.residuals) - 1}")
print(f"  Speedup: {(len(cg_residuals)-1)/(len(pcg.residuals)-1):.1f}×")

results['cg_convergence'] = {
    'cg_iters': len(cg_residuals) - 1,
    'pcg_iters': len(pcg.residuals) - 1,
}

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(cg_residuals, 'b-', lw=2, label=f'Plain CG ({len(cg_residuals)-1} iters)')
ax.semilogy(pcg.residuals, 'r-', lw=2, label=f'Nyström-PCG ({len(pcg.residuals)-1} iters)')
ax.axhline(1e-8, color='gray', ls='--', label='tol=1e-8')
ax.set_xlabel('Iteration'); ax.set_ylabel('Relative Residual')
ax.set_title('CG Convergence for Kernel System (K + σ²I)α = y')
ax.legend(); ax.grid(True, alpha=0.3)
save("gp_cg_convergence.png")


# ═══════════════════════════════════════════════════════════════
# Save JSON
# ═══════════════════════════════════════════════════════════════
json_path = os.path.join(DIR, "results", "gp_results.json")
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2,
              default=lambda x: float(x) if isinstance(x, (np.floating,)) else
                                int(x) if isinstance(x, (np.integer,)) else str(x))

print(f"\n  JSON: {json_path}")
print(f"\n{'=' * 60}")
print("SUMMARY")
print("=" * 60)
print(f"  Kernel spectrum:     90% at rank {spec['rank_90']}/{K.shape[0]}")
print(f"  Best approx (r=150): error {approx_results[-1]['error']:.6f}")
print(f"  PCG speedup:         {(len(cg_residuals)-1)/(len(pcg.residuals)-1):.1f}× fewer iterations")
print(f"  Scaling at N=1500:   {scaling[-1]['speedup']:.1f}× faster than exact")
print("  ✓ Done!")
