"""
Transformer Attention Benchmark — Full vs Nyström vs Linear
============================================================
Trains small TransformerClassifiers on a positional pattern detection task,
compares accuracy, speed, attention error, and spectral properties.
"""

import os
import sys
import time
import json

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import TransformerClassifier
from dataset import get_dataloader
from trainer import TransformerTrainer
from nystrom_module import NystromAttentionAnalyzer

torch.manual_seed(42)
np.random.seed(42)

DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
VOCAB_SIZE = 17   # sklearn digits: pixel values 0-16
MAX_SEQ = 64      # sklearn digits: 8x8 = 64 pixels flattened
N_CLASSES = 10    # digits 0-9
N_LANDMARKS = 16
NUM_EPOCHS = 25
LR = 3e-3
BATCH_SIZE = 64


def save_fig(name):
    path = os.path.join(RESULTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_model(attn_type, train_loader, val_loader):
    model = TransformerClassifier(
        vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ, d_model=D_MODEL,
        n_heads=N_HEADS, n_layers=N_LAYERS, n_classes=N_CLASSES,
        attn_type=attn_type, n_landmarks=N_LANDMARKS
    )
    trainer = TransformerTrainer(model, train_loader, val_loader, lr=LR)
    t0 = time.perf_counter()
    history = trainer.train(num_epochs=NUM_EPOCHS)
    train_time = time.perf_counter() - t0
    final_acc = trainer.evaluate()
    return model, trainer, history, train_time, final_acc


print("=" * 65)
print("TRANSFORMER ATTENTION BENCHMARK")
print(f"  Task: Digit classification (sklearn Digits, 8x8 → 64 tokens)")
print(f"  Config: d_model={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}, "
      f"seq_len={MAX_SEQ}, vocab={VOCAB_SIZE}, classes={N_CLASSES}")
print("=" * 65)

train_loader = get_dataloader(batch_size=BATCH_SIZE, split='train')
val_loader = get_dataloader(batch_size=BATCH_SIZE, split='test')

results = {}
models_trained = {}
histories = {}

print("\n── Phase 1: Training Transformers ─────────────────────────────")
for attn_type in ['full', 'nystrom', 'linear']:
    print(f"\n  Training [{attn_type.upper()}] attention...")
    model, trainer, history, train_time, final_acc = train_model(
        attn_type, train_loader, val_loader
    )
    models_trained[attn_type] = (model, trainer)
    histories[attn_type] = history
    n_params = count_params(model)
    results[attn_type] = {
        'final_accuracy': final_acc,
        'train_time_s': train_time,
        'n_params': n_params,
        'final_loss': history['train_loss'][-1],
    }
    print(f"    Accuracy: {final_acc:.3f} | Loss: {history['train_loss'][-1]:.4f} "
          f"| Time: {train_time:.2f}s | Params: {n_params}")

print("\n── Phase 2: Attention Comparison ──────────────────────────────")
analyzer = NystromAttentionAnalyzer()

model_full, trainer_full = models_trained['full']
model_nystrom, _ = models_trained['nystrom']

comparison = analyzer.compare_attention_outputs(model_full, model_nystrom, val_loader)
results['attention_comparison'] = comparison
print(f"  Full vs Nyström relative attention error: {comparison['relative_attn_error']:.4f}")
print(f"  Mean logit difference: {comparison['mean_logit_diff']:.4f}")

print("\n── Phase 3: Eigenvalue Analysis ───────────────────────────────")
sample_batch = next(iter(val_loader))
attn_maps = trainer_full.get_attention_maps(sample_batch)

if attn_maps and attn_maps[0] is not None:
    spectrum_info = analyzer.eigenvalue_analysis(attn_maps[0])
    results['attention_spectrum'] = {
        'rank_90': spectrum_info['rank_90'],
        'effective_rank': spectrum_info['effective_rank'],
        'condition_ratio': spectrum_info['condition_ratio'],
    }
    print(f"  Attention rank (90% energy): {spectrum_info['rank_90']}/{MAX_SEQ}")
    print(f"  Effective rank: {spectrum_info['effective_rank']:.1f}")
    print(f"  Condition ratio: {spectrum_info['condition_ratio']:.1f}")
else:
    spectrum_info = None
    print("  [Skipped - no attention weights available]")

print("\n── Phase 4: Poisson vs Attention Spectrum ─────────────────────")
poisson_cmp = analyzer.poisson_vs_attention_spectrum(MAX_SEQ)
results['poisson_vs_attention'] = {
    'poisson_condition': poisson_cmp['poisson_condition'],
    'attn_condition': poisson_cmp['attn_condition'],
}
print(f"  Poisson condition number: {poisson_cmp['poisson_condition']:.1f}")
print(f"  Attention condition number: {poisson_cmp['attn_condition']:.1f}")

print("\n── Phase 5: Speed Benchmark ───────────────────────────────────")
seq_lengths = [16, 32, 64]
speed_results = analyzer.benchmark_speed(seq_lengths, d_model=D_MODEL,
                                         n_landmarks=N_LANDMARKS, n_heads=N_HEADS)
results['speed_benchmark'] = speed_results
print(f"  {'SeqLen':>8} {'Full(ms)':>10} {'Nystrom(ms)':>12} {'Speedup':>8}")
print(f"  {'-'*42}")
for r in speed_results:
    print(f"  {r['seq_len']:>8} {r['full_ms']:>10.2f} {r['nystrom_ms']:>12.2f} "
          f"{r['speedup']:>7.2f}×")

# ── Plotting ────────────────────────────────────────────────────────────────
print("\n── Generating Plots ──────────────────────────────────────────")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

colors = {'full': '#2196F3', 'nystrom': '#FF9800', 'linear': '#4CAF50'}

ax = axes[0, 0]
for attn_type, hist in histories.items():
    ax.plot(hist['train_loss'], color=colors[attn_type], lw=2, label=attn_type.capitalize())
ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss')
ax.set_title('Training Loss Curves')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for attn_type, hist in histories.items():
    ax.plot(hist['val_acc'], color=colors[attn_type], lw=2, marker='o', label=attn_type.capitalize())
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Accuracy')
ax.set_title('Validation Accuracy')
ax.axhline(0.1, color='gray', ls='--', alpha=0.5, label='Chance (10-class)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
names = list(results.keys())[:3]
accs = [results[k]['final_accuracy'] for k in ['full', 'nystrom', 'linear']]
bars = ax.bar(['Full', 'Nyström', 'Linear'], accs,
              color=[colors['full'], colors['nystrom'], colors['linear']], alpha=0.8)
ax.set_ylabel('Final Accuracy')
ax.set_title('Final Accuracy Comparison')
ax.set_ylim(0, 1.05)
ax.axhline(0.1, color='gray', ls='--', alpha=0.5)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{acc:.2f}', ha='center', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1, 0]
if spectrum_info is not None:
    eigvals = spectrum_info['eigenvalues']
    ax.semilogy(eigvals / eigvals[0], 'b-', lw=2, marker='o', ms=3)
    ax.axvline(spectrum_info['rank_90'], color='r', ls='--',
               label=f"90% energy at rank {spectrum_info['rank_90']}")
    ax.set_xlabel('Index')
    ax.set_ylabel('Normalized |λ|')
    ax.set_title('Learned Attention Spectrum')
    ax.legend()
else:
    ax.text(0.5, 0.5, 'No attention weights', ha='center', va='center',
            transform=ax.transAxes)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
pe = poisson_cmp['poisson_eigvals']
ae = poisson_cmp['attn_eigvals']
ax.semilogy(pe / pe[0], 'b-', lw=2, label=f"Poisson (κ={poisson_cmp['poisson_condition']:.0f})")
ax.semilogy(ae / ae[0], 'r-', lw=2, label=f"Attention (κ={poisson_cmp['attn_condition']:.0f})")
ax.set_xlabel('Index')
ax.set_ylabel('Normalized eigenvalue')
ax.set_title('Poisson vs Attention Spectrum')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 2]
sls = [r['seq_len'] for r in speed_results]
ax.plot(sls, [r['full_ms'] for r in speed_results], 'b-o', lw=2, label='Full')
ax.plot(sls, [r['nystrom_ms'] for r in speed_results], 'r-s', lw=2, label='Nyström')
ax.set_xlabel('Sequence Length')
ax.set_ylabel('Time (ms)')
ax.set_title('Inference Time Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Transformer Attention: Full vs Nyström vs Linear', fontsize=14, y=1.01)
plt.tight_layout()
save_fig("transformer_attention_benchmark.png")

# ── Save JSON ──────────────────────────────────────────────────────────
json_path = os.path.join(RESULTS_DIR, "transformer_results.json")
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2,
              default=lambda x: float(x) if isinstance(x, (np.floating, torch.Tensor)) else
                                int(x) if isinstance(x, np.integer) else str(x))
print(f"\n  Results JSON: {json_path}")
print("\n  ✓ Benchmark complete!")
