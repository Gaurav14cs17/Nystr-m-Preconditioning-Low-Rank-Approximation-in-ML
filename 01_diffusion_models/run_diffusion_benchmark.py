"""
Diffusion Models — Nyström Attention & Preconditioning Benchmark

1. Train a tiny DDPM on synthetic 28x28 patterns
2. Verify NystromAttentionBlock vs full attention
3. Inverse-problem preconditioning comparison (CG convergence)
4. Save results + plots to results/
"""

import os
import sys
import json
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import UNet, GaussianDiffusion, NystromAttentionBlock
from dataset import get_dataloader
from trainer import DiffusionTrainer
from nystrom_module import compare_preconditioners

torch.manual_seed(42)
np.random.seed(42)

DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(DIR, "results")
os.makedirs(RESULTS, exist_ok=True)

def save_fig(name):
    path = os.path.join(RESULTS, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# BENCHMARK 1: Training a DDPM
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("BENCHMARK 1: Train DDPM on MNIST 28x28 digits")
print("=" * 60)

dataloader = get_dataloader(batch_size=16, num_samples=256, seed=42)
unet = UNet(in_ch=1, channels=(32, 64), time_dim=64, num_landmarks=8)
diffusion = GaussianDiffusion(unet, timesteps=200)

param_count = sum(p.numel() for p in unet.parameters())
print(f"  UNet parameters: {param_count:,}")

trainer = DiffusionTrainer(unet, diffusion, dataloader, lr=1e-3, device='cpu')
t0 = time.time()
losses = trainer.train(num_epochs=3)
train_time = time.time() - t0
print(f"  Total training time: {train_time:.1f}s")

eval_mse = trainer.evaluate()
print(f"  Evaluation noise-prediction MSE: {eval_mse:.4f}")

samples = trainer.sample(n_samples=4)

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i in range(4):
    axes[i].imshow(samples[i, 0].numpy(), cmap='gray', vmin=-1, vmax=1)
    axes[i].axis('off')
    axes[i].set_title(f'Sample {i+1}')
plt.suptitle(f'DDPM Samples (3 epochs, loss={losses[-1]:.4f})')
plt.tight_layout()
save_fig("diffusion_ddpm_samples.png")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(1, len(losses) + 1), losses, 'b-o', lw=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('DDPM Training Loss')
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_fig("diffusion_training_loss.png")


# ═══════════════════════════════════════════════════════════════
# BENCHMARK 2: Nyström Attention vs Full Attention
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("BENCHMARK 2: NystromAttentionBlock vs Full Attention")
print("=" * 60)

attention_results = []
for landmarks_m in [4, 8, 16, 32]:
    attn_block = NystromAttentionBlock(channels=64, num_landmarks=landmarks_m, num_heads=1)
    attn_block.eval()

    x_test = torch.randn(2, 64, 7, 7)

    with torch.no_grad():
        out_nystrom = attn_block(x_test)
        out_full = attn_block.full_attention(x_test)

    # Both outputs include the residual connection, so the difference is only in the attention part
    rel_err = torch.norm(out_nystrom - out_full) / torch.norm(out_full)

    t_ny, t_full = 0.0, 0.0
    n_trials = 50
    with torch.no_grad():
        for _ in range(n_trials):
            t0 = time.perf_counter()
            attn_block(x_test)
            t_ny += time.perf_counter() - t0

            t0 = time.perf_counter()
            attn_block.full_attention(x_test)
            t_full += time.perf_counter() - t0

    row = {
        'landmarks': landmarks_m,
        'rel_error': rel_err.item(),
        'nystrom_ms': t_ny / n_trials * 1000,
        'full_ms': t_full / n_trials * 1000,
    }
    attention_results.append(row)

print(f"\n  {'m':>4} {'RelError':>10} {'NystromMS':>10} {'FullMS':>10} {'MemRatio':>10}")
print(f"  {'-' * 48}")
N_seq = 7 * 7
for r in attention_results:
    mem_ratio = N_seq / r['landmarks']
    print(f"  {r['landmarks']:>4} {r['rel_error']:>10.4f} {r['nystrom_ms']:>10.3f} "
          f"{r['full_ms']:>10.3f} {mem_ratio:>9.1f}x")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ms = [r['landmarks'] for r in attention_results]
errs = [r['rel_error'] for r in attention_results]
ax1.plot(ms, errs, 'bo-', lw=2, ms=8)
ax1.set_xlabel('Landmarks (m)')
ax1.set_ylabel('Relative Error')
ax1.set_title('Nyström vs Full Attention (7x7 feature map)')
ax1.grid(True, alpha=0.3)

ax2.bar(range(len(ms)), [N_seq / m for m in ms], color='steelblue', alpha=0.7)
ax2.set_xticks(range(len(ms)))
ax2.set_xticklabels([str(m) for m in ms])
ax2.set_xlabel('Landmarks (m)')
ax2.set_ylabel('Memory Savings (N/m)')
ax2.set_title('Attention Memory Reduction')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
save_fig("diffusion_attention_spectrum.png")


# ═══════════════════════════════════════════════════════════════
# BENCHMARK 3: Eigenvalue spectrum of attention matrix
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("BENCHMARK 3: Attention Matrix Eigenvalue Spectrum")
print("=" * 60)

seq_len, d = 128, 32
pos = np.linspace(0, 2 * np.pi, seq_len)[:, None]
freqs = np.arange(1, d // 2 + 1)[None, :]
features = np.concatenate([np.sin(pos * freqs), np.cos(pos * freqs)], axis=1)[:, :d]
Q = (features + np.random.randn(seq_len, d) * 0.15).astype(np.float32)
K = (features + np.random.randn(seq_len, d) * 0.15).astype(np.float32)

scores = Q @ K.T / np.sqrt(d)
exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))
A_full = exp_s / exp_s.sum(axis=1, keepdims=True)

eigvals = np.sort(np.abs(np.linalg.eigvals(A_full)))[::-1]
cumul = np.cumsum(eigvals ** 2) / np.sum(eigvals ** 2)
r90 = int(np.searchsorted(cumul, 0.9) + 1)
r99 = int(np.searchsorted(cumul, 0.99) + 1)

print(f"  Matrix: {seq_len}x{seq_len}")
print(f"  Top eigenvalue: {eigvals[0]:.4f},  Bottom: {eigvals[-1]:.2e}")
print(f"  Ratio: {eigvals[0]/(eigvals[-1]+1e-15):.0f}x")
print(f"  90% energy at rank {r90}/{seq_len}")
print(f"  99% energy at rank {r99}/{seq_len}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].semilogy(eigvals, 'b-', lw=2)
axes[0].set_xlabel('Index')
axes[0].set_ylabel('|Eigenvalue|')
axes[0].set_title(f'Attention Eigenvalue Decay ({seq_len}x{seq_len})')
axes[0].axhline(eigvals[0] * 0.01, color='r', ls='--', label='1% of max')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(cumul, 'g-', lw=2)
axes[1].axhline(0.9, color='r', ls='--', label=f'90% at rank {r90}')
axes[1].axhline(0.99, color='orange', ls='--', label=f'99% at rank {r99}')
axes[1].set_xlabel('Rank')
axes[1].set_ylabel('Cumulative Energy')
axes[1].set_title('Spectral Energy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].imshow(A_full[:64, :64], cmap='hot', aspect='auto')
axes[2].set_title('Attention Matrix (64x64 block)')
axes[2].set_xlabel('Key')
axes[2].set_ylabel('Query')
plt.tight_layout()
save_fig("diffusion_nystrom_approximation.png")

spectrum_results = {
    'matrix_size': seq_len,
    'top_eig': float(eigvals[0]),
    'bottom_eig': float(eigvals[-1]),
    'ratio': float(eigvals[0] / (eigvals[-1] + 1e-15)),
    'rank_90pct': r90,
    'rank_99pct': r99,
}


# ═══════════════════════════════════════════════════════════════
# BENCHMARK 4: Inverse Problem — Deblurring with Preconditioned CG
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("BENCHMARK 4: Deblurring — Preconditioned CG Convergence")
print("=" * 60)

N_inv = 128
kernel = np.exp(-np.arange(N_inv) ** 2 / (2 * 10 ** 2))
kernel /= kernel.sum()
A_blur = toeplitz(kernel)
A_blur /= np.linalg.norm(A_blur, 2)

x_true = np.sin(np.linspace(0, 4 * np.pi, N_inv)) + 0.5 * np.cos(np.linspace(0, 8 * np.pi, N_inv))

cg_results = []
for lam in [0.001, 0.01, 0.1]:
    precond_results, kappa = compare_preconditioners(A_blur, x_true, lam, nystrom_rank=20)

    row = {'lambda': lam, 'kappa': f"{kappa:.0f}"}
    print(f"\n  lambda={lam}, kappa={kappa:.0f}")
    print(f"  {'Method':<18} {'Iters':>8} {'Time(ms)':>10}")
    print(f"  {'-' * 38}")
    for name, data in precond_results.items():
        print(f"  {name:<18} {data['iters']:>8} {data['time_ms']:>10.2f}")
        row[f'{name}_iters'] = data['iters']
        row[f'{name}_time_ms'] = round(data['time_ms'], 2)
    cg_results.append(row)

# Convergence plot for lambda=0.001
precond_results_plot, _ = compare_preconditioners(A_blur, x_true, 0.001, nystrom_rank=20)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
colors = {'CG': 'b', 'Jacobi': 'g', 'Nystrom-20': 'r', 'ILU': 'm'}
for name, data in precond_results_plot.items():
    ax1.semilogy(data['residuals'], colors.get(name, 'k') + '-', lw=2,
                 label=f"{name} ({data['iters']} iters)")
ax1.axhline(1e-10, color='gray', ls='--', label='tol=1e-10')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Relative Residual')
ax1.set_title('Deblurring CG Convergence (lambda=0.001)')
ax1.legend()
ax1.grid(True, alpha=0.3)

from nystrom_module import NystromPreconditionedCG
AtA = A_blur.T @ A_blur + 0.001 * np.eye(N_inv)
rhs = AtA @ x_true
npc = NystromPreconditionedCG(AtA - 0.001 * np.eye(N_inv), 0.001, rank=20)
x_recon, _ = NystromPreconditionedCG.cg_solve(AtA, rhs, precond_fn=npc.apply)

ax2.plot(x_true, 'b-', lw=2, label='True signal')
ax2.plot(A_blur @ x_true, 'r--', lw=1, alpha=0.5, label='Blurred')
ax2.plot(x_recon, 'g-', lw=1.5, label='Reconstructed (Nystrom CG)')
ax2.set_xlabel('Index')
ax2.set_ylabel('Value')
ax2.set_title('Signal Reconstruction')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
save_fig("diffusion_inverse_problem_cg.png")


# ═══════════════════════════════════════════════════════════════
# Save all results to JSON
# ═══════════════════════════════════════════════════════════════
all_results = {
    'training': {
        'param_count': param_count,
        'epochs': 3,
        'final_loss': losses[-1],
        'eval_mse': eval_mse,
        'train_time_s': round(train_time, 1),
        'loss_history': losses,
    },
    'attention_comparison': attention_results,
    'attention_spectrum': spectrum_results,
    'inverse_problem': cg_results,
}

json_path = os.path.join(RESULTS, "diffusion_results.json")
with open(json_path, 'w') as f:
    json.dump(all_results, f, indent=2,
              default=lambda x: float(x) if isinstance(x, (np.floating,)) else
                                int(x) if isinstance(x, (np.integer,)) else x)

print(f"\n  JSON: {json_path}")
print(f"\n{'=' * 60}")
print("SUMMARY")
print("=" * 60)
print(f"  UNet params:          {param_count:,}")
print(f"  Training loss:        {losses[0]:.4f} -> {losses[-1]:.4f}")
print(f"  Eval MSE:             {eval_mse:.4f}")
print(f"  Attention spectrum:   90% energy at rank {r90}/{seq_len}")
print(f"  Best CG (lam=0.001): Nystrom-20 = {precond_results_plot['Nystrom-20']['iters']} iters")
print("  Done.")
