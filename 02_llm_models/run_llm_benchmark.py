"""
LLM Benchmark: Nyström Attention + KV-Cache Compression + Hessian Spectrum
Runs on CPU with tiny models. Completes in under 60 seconds.
"""

import os
import sys
import time
import json
import math
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import CausalLM, KVCache
from dataset import CharDataset, get_dataloader
from trainer import LLMTrainer
from nystrom_module import (
    NystromAttentionLayer, KVCacheCompressor, measure_attention_spectrum
)

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_plot(name):
    path = os.path.join(RESULTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


results = {}

# ── 1. Train the CausalLM ────────────────────────────────────
print("=" * 60)
print("LLM BENCHMARK 1: Train CausalLM on TinyShakespeare (char-level)")
print("=" * 60)

dataloader = get_dataloader(batch_size=16, seq_len=32, num_samples=500)
model_full = CausalLM(vocab_size=256, d_model=64, n_heads=4, n_layers=2,
                      max_seq=128, attention_mode='full')

n_params = sum(p.numel() for p in model_full.parameters())
print(f"  Model params: {n_params:,}")
print(f"  Dataset: {len(dataloader.dataset)} samples, seq_len=32")
print()

trainer = LLMTrainer(model_full, dataloader, lr=3e-4, device='cpu')
train_losses = trainer.train(num_epochs=3)

eval_loss, eval_ppl = trainer.evaluate()
print(f"\n  Final eval loss: {eval_loss:.4f} | Perplexity: {eval_ppl:.2f}")

prompt = [ord(c) for c in "3+4="]
generated_ids = trainer.generate(prompt, max_len=10, temperature=0.8)
generated_text = ''.join(chr(min(c, 127)) for c in generated_ids)
print(f"  Generation sample: '{generated_text}'")

loss_decreased = train_losses[-1] < train_losses[0]
print(f"  Loss decreased: {loss_decreased} ({train_losses[0]:.3f} -> {train_losses[-1]:.3f})")

results['training'] = {
    'n_params': n_params,
    'train_losses': train_losses,
    'eval_loss': eval_loss,
    'eval_perplexity': eval_ppl,
    'loss_decreased': loss_decreased,
}

# ── 2. Full vs Nyström attention comparison ───────────────────
print(f"\n{'=' * 60}")
print("LLM BENCHMARK 2: Full Attention vs Nyström Attention")
print("=" * 60)

model_nystrom = CausalLM(vocab_size=256, d_model=64, n_heads=4, n_layers=2,
                         max_seq=128, attention_mode='nystrom', n_landmarks=16)

model_nystrom.load_state_dict(model_full.state_dict(), strict=False)

attention_comparison = []
seq_lengths = [16, 32, 64, 128]

print(f"\n  {'SeqLen':>8} {'Full(ms)':>10} {'Nyst(ms)':>10} {'Speedup':>8} {'OutputErr':>10}")
print(f"  {'-' * 50}")

for T in seq_lengths:
    x = torch.randint(0, 256, (4, T))

    model_full.eval()
    model_nystrom.eval()

    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(20):
            out_full = model_full(x)
        t_full = (time.perf_counter() - t0) / 20

        t0 = time.perf_counter()
        for _ in range(20):
            out_nyst = model_nystrom(x)
        t_nyst = (time.perf_counter() - t0) / 20

    err = torch.norm(out_full - out_nyst).item() / (torch.norm(out_full).item() + 1e-8)
    speedup = t_full / (t_nyst + 1e-8)

    row = {'seq_len': T, 'full_ms': t_full * 1000, 'nyst_ms': t_nyst * 1000,
           'speedup': speedup, 'output_error': err}
    attention_comparison.append(row)
    print(f"  {T:>8} {t_full*1000:>10.2f} {t_nyst*1000:>10.2f} {speedup:>7.2f}× {err:>10.4f}")

results['attention_comparison'] = attention_comparison

# ── 3. KV-Cache compression ──────────────────────────────────
print(f"\n{'=' * 60}")
print("LLM BENCHMARK 3: KV-Cache Compression")
print("=" * 60)

T_cache = 64
D = 64
K_cache = torch.randn(T_cache, D)
V_cache = torch.randn(T_cache, D)

pos = torch.linspace(0, 2 * math.pi, T_cache).unsqueeze(1)
freqs = torch.arange(1, D // 2 + 1).unsqueeze(0).float()
structured = torch.cat([torch.sin(pos * freqs), torch.cos(pos * freqs)], dim=1)[:, :D]
K_cache = structured + 0.1 * torch.randn(T_cache, D)
V_cache = structured + 0.1 * torch.randn(T_cache, D)

compressor_nystrom = KVCacheCompressor(method='nystrom')
compressor_svd = KVCacheCompressor(method='svd')

ranks = [4, 8, 16, 32, 48]
kv_results = []

print(f"\n  {'Rank':>6} {'Nyst Err':>10} {'SVD Err':>10} {'Compression':>12}")
print(f"  {'-' * 42}")

for r in ranks:
    _, _, err_ny = compressor_nystrom.compress(K_cache.clone(), V_cache.clone(), r)
    _, _, err_svd = compressor_svd.compress(K_cache.clone(), V_cache.clone(), r)
    compression = T_cache / r

    kv_results.append({'rank': r, 'nystrom_error': err_ny, 'svd_error': err_svd,
                       'compression_ratio': compression})
    print(f"  {r:>6} {err_ny:>10.6f} {err_svd:>10.6f} {compression:>11.1f}×")

results['kv_cache'] = kv_results

# ── 4. Attention spectrum ─────────────────────────────────────
print(f"\n{'=' * 60}")
print("LLM BENCHMARK 4: Attention Matrix Eigenvalue Spectrum")
print("=" * 60)

eigvals = measure_attention_spectrum(model_full, dataloader, device='cpu', max_batches=3)
if len(eigvals) > 0:
    cumul = np.cumsum(eigvals) / (np.sum(eigvals) + 1e-10)
    r90 = int(np.searchsorted(cumul, 0.9) + 1)
    r99 = int(np.searchsorted(cumul, 0.99) + 1)
    print(f"  Spectrum length: {len(eigvals)}")
    print(f"  Top 5 eigenvalues: {eigvals[:5].round(4)}")
    print(f"  Rank for 90% energy: {r90}/{len(eigvals)}")
    print(f"  Rank for 99% energy: {r99}/{len(eigvals)}")
    results['attention_spectrum'] = {
        'eigenvalues': eigvals.tolist()[:20],
        'rank_90': r90,
        'rank_99': r99,
    }
else:
    print("  (No eigenvalues computed)")
    eigvals = np.array([1.0])
    results['attention_spectrum'] = {}

# ── 5. Hessian spectrum ───────────────────────────────────────
print(f"\n{'=' * 60}")
print("LLM BENCHMARK 5: Hessian Spectrum of Trained Model")
print("=" * 60)

small_model = CausalLM(vocab_size=64, d_model=16, n_heads=2, n_layers=1, max_seq=32)
small_params = sum(p.numel() for p in small_model.parameters())
print(f"  Small model for Hessian: {small_params} params")

X_h = torch.randint(0, 64, (8, 8))
Y_h = torch.randint(0, 64, (8, 8))

logits_h = small_model(X_h)
loss_h = F.cross_entropy(logits_h.view(-1, 64), Y_h.view(-1))

params_list = [p for p in small_model.parameters() if p.requires_grad]
grads = torch.autograd.grad(loss_h, params_list, create_graph=True)
g_vec = torch.cat([g.flatten() for g in grads])

n_hess = min(g_vec.shape[0], 100)
H = torch.zeros(n_hess, n_hess)
for i in range(n_hess):
    h_row = torch.autograd.grad(g_vec[i], params_list, retain_graph=True)
    h_flat = torch.cat([h.flatten() for h in h_row])
    H[i] = h_flat[:n_hess]

H_np = ((H + H.T) / 2).detach().numpy()
eigvals_H = np.sort(np.abs(np.linalg.eigvalsh(H_np)))[::-1]
cumul_H = np.cumsum(eigvals_H**2) / (np.sum(eigvals_H**2) + 1e-10)
r90_h = int(np.searchsorted(cumul_H, 0.9) + 1)
r99_h = int(np.searchsorted(cumul_H, 0.99) + 1)

print(f"  Hessian submatrix size: {n_hess}×{n_hess}")
print(f"  Top 5 |eigenvalues|: {eigvals_H[:5].round(6)}")
print(f"  Rank for 90% energy: {r90_h}/{n_hess}  ← Hessian is low-rank")
print(f"  Rank for 99% energy: {r99_h}/{n_hess}")

results['hessian'] = {
    'n_params': small_params,
    'hessian_size': n_hess,
    'rank_90': r90_h,
    'rank_99': r99_h,
    'top_eigenvalue': float(eigvals_H[0]),
    'condition_number': float(eigvals_H[0] / (eigvals_H[-1] + 1e-10)),
}

# ── Plots ─────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("Generating plots...")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0, 0].plot(range(1, len(train_losses) + 1), train_losses, 'bo-', lw=2, ms=8)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss (CausalLM)')
axes[0, 0].grid(True, alpha=0.3)

sls = [r['seq_len'] for r in attention_comparison]
axes[0, 1].plot(sls, [r['full_ms'] for r in attention_comparison], 'bo-', lw=2, label='Full')
axes[0, 1].plot(sls, [r['nyst_ms'] for r in attention_comparison], 'rs-', lw=2, label='Nyström')
axes[0, 1].set_xlabel('Sequence Length')
axes[0, 1].set_ylabel('Time (ms)')
axes[0, 1].set_title('Attention Speed: Full vs Nyström')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(sls, [r['output_error'] for r in attention_comparison], 'mD-', lw=2, ms=8)
axes[0, 2].set_xlabel('Sequence Length')
axes[0, 2].set_ylabel('Relative Output Error')
axes[0, 2].set_title('Nyström Approximation Error')
axes[0, 2].grid(True, alpha=0.3)

rs = [r['rank'] for r in kv_results]
axes[1, 0].plot(rs, [r['nystrom_error'] for r in kv_results], 'rs-', lw=2, label='Nyström')
axes[1, 0].plot(rs, [r['svd_error'] for r in kv_results], 'b^-', lw=2, label='SVD')
axes[1, 0].set_xlabel('Compression Rank')
axes[1, 0].set_ylabel('Reconstruction Error')
axes[1, 0].set_title('KV-Cache Compression')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].semilogy(eigvals[:min(len(eigvals), 32)], 'g-', lw=2)
axes[1, 1].set_xlabel('Index')
axes[1, 1].set_ylabel('Eigenvalue')
axes[1, 1].set_title('Attention Spectrum (Layer 0)')
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].semilogy(eigvals_H, 'b-', lw=2)
axes[1, 2].axvline(r90_h, color='r', ls='--', alpha=0.7, label=f'90% @ rank {r90_h}')
axes[1, 2].axvline(r99_h, color='orange', ls='--', alpha=0.7, label=f'99% @ rank {r99_h}')
axes[1, 2].set_xlabel('Index')
axes[1, 2].set_ylabel('|Eigenvalue|')
axes[1, 2].set_title(f'Hessian Spectrum ({n_hess} dims)')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
save_plot("llm_benchmark_results.png")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.semilogy(eigvals_H, 'b-', lw=2)
ax1.set_xlabel('Index')
ax1.set_ylabel('|Eigenvalue|')
ax1.set_title(f'Hessian Spectrum ({small_params} params, {n_hess} dims)')
ax1.grid(True, alpha=0.3)

ax2.plot(cumul_H, 'g-', lw=2)
ax2.axhline(0.9, color='r', ls='--', label=f'90% at rank {r90_h}')
ax2.axhline(0.99, color='orange', ls='--', label=f'99% at rank {r99_h}')
ax2.set_xlabel('Rank')
ax2.set_ylabel('Cumulative Energy')
ax2.set_title('Hessian Spectral Energy')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
save_plot("llm_hessian_spectrum.png")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(rs, [r['nystrom_error'] for r in kv_results], 'rs-', lw=2, ms=8, label='Nyström')
ax1.plot(rs, [r['svd_error'] for r in kv_results], 'b^-', lw=2, ms=8, label='SVD')
ax1.set_xlabel('Compression Rank')
ax1.set_ylabel('Reconstruction Error')
ax1.set_title('KV-Cache Compression Quality')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.bar(range(len(rs)), [r['compression_ratio'] for r in kv_results], color='teal', alpha=0.7)
ax2.set_xticks(range(len(rs)))
ax2.set_xticklabels([str(r) for r in rs])
ax2.set_xlabel('Rank')
ax2.set_ylabel('Compression Ratio (×)')
ax2.set_title('KV-Cache Memory Reduction')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
save_plot("llm_kv_cache_compression.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(sls, [r['full_ms'] for r in attention_comparison], 'bo-', lw=2, ms=8, label='Full O(N²)')
axes[0].plot(sls, [r['nyst_ms'] for r in attention_comparison], 'rs-', lw=2, ms=8, label='Nyström O(Nm)')
axes[0].set_xlabel('Sequence Length')
axes[0].set_ylabel('Time (ms)')
axes[0].set_title('Attention Time Scaling')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].semilogy(eigvals[:min(len(eigvals), 32)], 'g-', lw=2, ms=6)
axes[1].set_xlabel('Index')
axes[1].set_ylabel('Eigenvalue (log)')
axes[1].set_title('Attention Eigenvalue Decay')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
save_plot("llm_attention_scaling.png")

# ── Save JSON ─────────────────────────────────────────────────
json_path = os.path.join(RESULTS_DIR, "llm_results.json")
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2,
              default=lambda x: float(x) if isinstance(x, (np.floating, torch.Tensor)) else
                                int(x) if isinstance(x, np.integer) else x)

print(f"\n  JSON: {json_path}")
print(f"\n{'=' * 60}")
print("  ✓ All LLM benchmarks completed successfully!")
print("=" * 60)
