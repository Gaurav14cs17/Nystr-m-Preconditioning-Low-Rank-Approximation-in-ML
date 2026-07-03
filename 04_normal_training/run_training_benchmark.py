"""
Normal Training — Nyström Preconditioning Benchmark

Demonstrates that Nyström-preconditioned SGD accelerates neural network
training by approximating the Hessian inverse — the same technique that
speeds up PDE solvers.

Benchmarks:
  1. Optimizer convergence comparison (SGD, Adam, SGD+Nyström, Adam+Nyström)
  2. MLP Hessian spectrum analysis (eigenvalue decay + effective rank)
  3. Condition number before / after Nyström preconditioning
  4. Regression task with preconditioning
"""

import os
import sys
import time
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from copy import deepcopy

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)

from models import MLP, NystromPreconditioner
from dataset import get_dataloaders
from trainer import Trainer
from nystrom_module import HessianApproximator, NystromOptimizer, compare_optimizers

np.random.seed(42)
torch.manual_seed(42)

RES = os.path.join(DIR, "results")
os.makedirs(RES, exist_ok=True)
results = {}


def save_fig(name):
    path = os.path.join(RES, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")
    return path


t_start = time.perf_counter()

# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 1: Optimizer Comparison — Ill-Conditioned Classification
# ═══════════════════════════════════════════════════════════════════
print("=" * 65)
print("BENCHMARK 1: Optimizer Comparison (Ill-Conditioned Classification)")
print("=" * 65)

train_loader, val_loader = get_dataloaders(
    task="classification", batch_size=64, n_train=500, n_val=100, d_features=10, seed=42
)

model_fn = lambda: MLP(d_in=10, d_hidden=32, d_out=2, n_layers=3)
loss_fn = nn.CrossEntropyLoss()

n_params = sum(p.numel() for p in model_fn().parameters())
print(f"  Model: MLP(10→32→32→2), {n_params} parameters")
print(f"  Data:  500 train / 100 val, 10-dim features (scales 31.6× to 1×)")
print(f"  Training SGD, Adam, SGD+Nyström, Adam+Nyström for 30 epochs ...")

comparison = compare_optimizers(
    model_fn, train_loader, val_loader, loss_fn, n_epochs=30, device="cpu"
)

print(f"\n  {'Method':<20s} {'Final Loss':>12s} {'Val Acc':>10s}")
print(f"  {'-'*44}")
for name, data in comparison.items():
    acc_str = f"{data['final_acc']:.3f}" if data["final_acc"] is not None else "N/A"
    print(f"  {name:<20s} {data['final_loss']:>12.4f} {acc_str:>10s}")

results["optimizer_comparison"] = {
    name: {
        "final_loss": float(d["final_loss"]),
        "final_acc": float(d["final_acc"]) if d["final_acc"] is not None else None,
        "train_loss": [float(v) for v in d["train_loss"]],
        "val_accuracy": [float(v) if v is not None else None for v in d["val_accuracy"]],
    }
    for name, d in comparison.items()
}

colors = {
    "SGD": "#2196F3",
    "Adam": "#4CAF50",
    "SGD + Nyström": "#F44336",
    "Adam + Nyström": "#FF9800",
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for name, data in comparison.items():
    ax1.semilogy(data["train_loss"], color=colors[name], lw=2, label=name)
    if data["val_accuracy"][0] is not None:
        ax2.plot([v if v is not None else 0 for v in data["val_accuracy"]],
                 color=colors[name], lw=2, label=name)

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training Loss (log)")
ax1.set_title("Convergence: Training Loss")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel("Epoch")
ax2.set_ylabel("Validation Accuracy")
ax2.set_title("Convergence: Validation Accuracy")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.suptitle(
    "Nyström-Preconditioned SGD vs Standard Optimizers (MLP Classification)", fontsize=13
)
plt.tight_layout()
save_fig("training_convergence_comparison.png")


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 2: Hessian Spectrum + Condition Number Analysis
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print("BENCHMARK 2: MLP Hessian Spectrum Analysis")
print("=" * 65)

torch.manual_seed(42)
model_hess = MLP(d_in=10, d_hidden=32, d_out=2, n_layers=3)
X_batch, y_batch = next(iter(train_loader))

print(f"  Computing full {n_params}×{n_params} Hessian ...")
params_h = [p for p in model_hess.parameters() if p.requires_grad]
loss_h = loss_fn(model_hess(X_batch), y_batch)
grads_h = torch.autograd.grad(loss_h, params_h, create_graph=True)
flat_grad_h = torch.cat([g.reshape(-1) for g in grads_h])
H_full = torch.zeros(n_params, n_params)
for i in range(n_params):
    row = torch.autograd.grad(flat_grad_h[i], params_h, retain_graph=True)
    H_full[i] = torch.cat([r.reshape(-1) for r in row])
H_full = 0.5 * (H_full + H_full.T)
H_full = H_full.detach()

eigs = np.sort(np.abs(torch.linalg.eigvalsh(H_full).numpy()))[::-1].copy()

cumul = np.cumsum(eigs) / np.sum(eigs)
r90 = int(np.searchsorted(cumul, 0.9) + 1)
r99 = int(np.searchsorted(cumul, 0.99) + 1)

threshold = 1e-3 * eigs[0]
effective_eigs = eigs[eigs > threshold]
kappa_eff = float(effective_eigs[0] / effective_eigs[-1]) if len(effective_eigs) > 1 else float("inf")

print(f"  Top eigenvalue:     {eigs[0]:.4f}")
print(f"  Smallest significant (>{threshold:.2e}): {effective_eigs[-1]:.4e}")
print(f"  Effective κ (λ > 10⁻³·λ_max): {kappa_eff:.1f}")
print(f"  Effective rank (90% energy): {r90} / {n_params}")
print(f"  Effective rank (99% energy): {r99} / {n_params}")

results["hessian_spectrum"] = {
    "n_params": n_params,
    "condition_number": float(kappa_eff),
    "top_eig": float(eigs[0]),
    "bottom_significant_eig": float(effective_eigs[-1]),
    "rank_90": r90,
    "rank_99": r99,
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.semilogy(eigs + 1e-15, "b-", lw=2)
ax1.axhline(threshold, color="r", ls="--", alpha=0.5, label=f"10⁻³·λ_max = {threshold:.2e}")
ax1.set_xlabel("Index i")
ax1.set_ylabel("|λ_i|")
ax1.set_title(f"MLP Hessian Spectrum ({n_params} params, κ_eff ≈ {kappa_eff:.0f})")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.plot(cumul, "g-", lw=2)
ax2.axhline(0.9, color="r", ls="--", label=f"90 % at rank {r90}")
ax2.axhline(0.99, color="orange", ls="--", label=f"99 % at rank {r99}")
ax2.set_xlabel("Rank")
ax2.set_ylabel("Cumulative Energy")
ax2.set_title("Hessian Spectral Energy")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
save_fig("training_hessian_spectrum.png")


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 3: Condition Number Before / After Preconditioning
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print("BENCHMARK 3: Condition Number Before / After Preconditioning")
print("=" * 65)

U_nys, sigma_nys = HessianApproximator.nystrom_hessian(
    model_hess, loss_fn, (X_batch, y_batch), rank=15
)

# Eigenvalues of the preconditioned system (H_nys + λI)^{-1} H
# via Woodbury: P^{-1} H = (1/λ)(H − U diag(σ/(σ+λ)) U^T H)
damping = 1.0
UtH = U_nys.T @ H_full
scale = torch.diag(sigma_nys / (sigma_nys + damping))
precond_H = (H_full - U_nys @ scale @ UtH) / damping
eigs_precond = torch.linalg.eigvalsh(precond_H).numpy()
eigs_precond_abs = np.sort(np.abs(eigs_precond))[::-1]

max_lr_original = 2.0 / eigs[0]
max_lr_precond = 2.0 / max(eigs_precond_abs[0], 1e-15)
spectral_range_orig = eigs[0] - effective_eigs[-1]
spectral_range_precond = eigs_precond_abs[0] - eigs_precond_abs[min(len(effective_eigs) - 1, len(eigs_precond_abs) - 1)]

print(f"  Original: λ_max = {eigs[0]:.2f},  max stable lr = {max_lr_original:.4f}")
print(f"  Preconditioned: λ_max = {eigs_precond_abs[0]:.2f},  max stable lr = {max_lr_precond:.4f}")
print(f"  Allowed step-size increase: {max_lr_precond / max_lr_original:.1f}×")
print(f"  Original κ_eff:              {kappa_eff:.1f}")

results["conditioning"] = {
    "kappa_original": float(kappa_eff),
    "lambda_max_original": float(eigs[0]),
    "lambda_max_preconditioned": float(eigs_precond_abs[0]),
    "max_lr_original": float(max_lr_original),
    "max_lr_preconditioned": float(max_lr_precond),
    "step_size_increase": float(max_lr_precond / max_lr_original),
    "damping": damping,
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

nys_sorted = np.sort(sigma_nys.numpy())[::-1]
ax1.semilogy(eigs + 1e-15, "b-", lw=2, label=f"Full Hessian (λ_max = {eigs[0]:.1f})")
ax1.semilogy(nys_sorted + 1e-15, "ro-", lw=2, ms=5,
             label=f"Nyström top-15")
ax1.set_xlabel("Index")
ax1.set_ylabel("Eigenvalue")
ax1.set_title("Full vs Nyström Hessian Spectrum")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.semilogy(eigs + 1e-15, "b--", lw=1.5, alpha=0.5,
             label=f"Original (λ_max = {eigs[0]:.1f})")
ax2.semilogy(eigs_precond_abs + 1e-15, "r-", lw=2,
             label=f"Preconditioned (λ_max = {eigs_precond_abs[0]:.2f})")
ax2.axhline(1.0, color="green", ls=":", alpha=0.7, label="Ideal: all λ = 1")
ax2.set_xlabel("Index")
ax2.set_ylabel("|Eigenvalue|")
ax2.set_title(
    f"Preconditioning compresses top eigenvalues → "
    f"{max_lr_precond / max_lr_original:.0f}× larger stable lr"
)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.suptitle(
    f"Nyström Preconditioning: λ_max {eigs[0]:.1f} → {eigs_precond_abs[0]:.2f} "
    f"(allows {max_lr_precond / max_lr_original:.0f}× larger learning rate)",
    fontsize=13,
)
plt.tight_layout()
save_fig("training_condition_numbers.png")


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 4: Regression Task with Preconditioning
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print("BENCHMARK 4: Regression Task with Preconditioning")
print("=" * 65)

torch.manual_seed(42)
train_loader_reg, val_loader_reg = get_dataloaders(
    task="regression", batch_size=64, n_train=500, n_val=100, d_features=10, seed=42
)
model_fn_reg = lambda: MLP(d_in=10, d_hidden=32, d_out=1, n_layers=3)
loss_fn_reg = nn.MSELoss()

reg_configs = [
    ("SGD", lambda m: torch.optim.SGD(m.parameters(), lr=0.01)),
    ("Adam", lambda m: torch.optim.Adam(m.parameters(), lr=0.005)),
    (
        "SGD + Nyström",
        lambda m: NystromOptimizer(
            m, torch.optim.SGD(m.parameters(), lr=0.05),
            loss_fn_reg, rank=10, damping=1.0, rebuild_every=5,
        ),
    ),
    (
        "Adam + Nyström",
        lambda m: NystromOptimizer(
            m, torch.optim.Adam(m.parameters(), lr=0.003),
            loss_fn_reg, rank=10, damping=1.0, rebuild_every=5,
        ),
    ),
]

comparison_reg = compare_optimizers(
    model_fn_reg, train_loader_reg, val_loader_reg, loss_fn_reg,
    n_epochs=30, device="cpu", configs=reg_configs,
)

print(f"\n  {'Method':<20s} {'Final Loss':>12s}")
print(f"  {'-'*34}")
for name, data in comparison_reg.items():
    print(f"  {name:<20s} {data['final_loss']:>12.4f}")

results["regression"] = {
    name: {
        "final_loss": float(d["final_loss"]),
        "train_loss": [float(v) for v in d["train_loss"]],
    }
    for name, d in comparison_reg.items()
}

fig, ax = plt.subplots(figsize=(10, 6))
for name, data in comparison_reg.items():
    ax.semilogy(data["train_loss"], color=colors[name], lw=2, label=name)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss (log)")
ax.set_title("Regression: Convergence with Different Optimizers")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_fig("training_regression_convergence.png")


# ═══════════════════════════════════════════════════════════════════
# Save JSON results
# ═══════════════════════════════════════════════════════════════════
t_total = time.perf_counter() - t_start

results["meta"] = {
    "total_time_seconds": round(t_total, 2),
    "device": "cpu",
    "n_params": n_params,
}

json_path = os.path.join(RES, "training_results.json")
with open(json_path, "w") as f:
    json.dump(
        results, f, indent=2,
        default=lambda x: (
            float(x)
            if isinstance(x, (np.floating,))
            else int(x) if isinstance(x, (np.integer,)) else x
        ),
    )

print(f"\n{'=' * 65}")
print(f"All benchmarks complete in {t_total:.1f}s")
print(f"Results → {json_path}")
print("=" * 65)
