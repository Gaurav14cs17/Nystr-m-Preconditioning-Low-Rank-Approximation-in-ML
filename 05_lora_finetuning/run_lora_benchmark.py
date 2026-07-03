"""
LoRA Fine-tuning — Low-Rank = Nyström Preconditioning Benchmark

Proper pretrain → fine-tune pipeline demonstrating:
  1. Pretraining a base model on multi-output regression
  2. LoRA fine-tuning (ranks 2, 4, 8, 16) on a different downstream task
  3. Full fine-tuning comparison
  4. SVD analysis of weight updates (low-rank structure)
  5. Nyström ↔ LoRA connection (quantitative)
"""

import os, sys, time, json, copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

np.random.seed(42)
torch.manual_seed(42)

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)
os.makedirs(os.path.join(DIR, "results"), exist_ok=True)

from models import BaseModel, LoRAModel, NystromLoRAAnalyzer
from dataset import get_pretrain_loader, get_finetune_loader, FinetuneDataset
from trainer import PretrainTrainer, LoRAFinetuner, FullFinetuner
from nystrom_module import NystromLoRAConnection


def save(name):
    p = os.path.join(DIR, "results", name)
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")
    return p


def default_ser(x):
    if isinstance(x, (np.floating, np.float64, np.float32)):
        return float(x)
    if isinstance(x, (np.integer, np.int64, np.int32)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


results = {}
t_start = time.perf_counter()

# ── 1. Pretrain the base model ────────────────────────────────
print("=" * 60)
print("LORA BENCHMARK 1: Pretraining the Base Model")
print("=" * 60)

torch.manual_seed(42)
base_model = BaseModel(d_in=64, d_hidden=128, d_out=8)
pretrain_loader = get_pretrain_loader(batch_size=32, n_samples=2000)

total_base_params = sum(p.numel() for p in base_model.parameters())
print(f"  Base model params: {total_base_params:,}")
print(f"  Architecture: 64 → 128 → 128 → 8 (3-layer MLP)")

trainer = PretrainTrainer(base_model, pretrain_loader, lr=1e-3)
pretrain_losses = trainer.train(num_epochs=100)

print(f"  Pretrain loss:  {pretrain_losses[0]:.4f} → {pretrain_losses[-1]:.4f}")

pretrained_model = copy.deepcopy(base_model)

finetune_loader = get_finetune_loader(batch_size=16, n_samples=200)
test_set = FinetuneDataset(n_samples=300, seed=999)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

base_model.eval()
with torch.no_grad():
    baseline_loss = sum(
        torch.nn.MSELoss()(base_model(X), Y).item()
        for X, Y in test_loader
    ) / len(test_loader)
print(f"  Pretrained model on finetune test: {baseline_loss:.4f}  ← needs adaptation")

results["pretrain"] = {
    "params": total_base_params,
    "loss_history": pretrain_losses,
    "finetune_baseline_loss": baseline_loss,
}


# ── 2. LoRA fine-tuning at different ranks ────────────────────
print(f"\n{'=' * 60}")
print("LORA BENCHMARK 2: LoRA Fine-Tuning (ranks 2, 4, 8, 16)")
print("=" * 60)

lora_ranks = [2, 4, 8, 16]
lora_results = {}

print(f"\n  {'Rank':>6} {'LoRA Params':>12} {'% of Base':>10} "
      f"{'Final Loss':>12} {'Test Loss':>12}")
print(f"  {'-' * 58}")

for rank in lora_ranks:
    torch.manual_seed(42)
    lora_model = LoRAModel(pretrained_model, rank=rank)
    ft_loader = get_finetune_loader(batch_size=16, n_samples=200)
    finetuner = LoRAFinetuner(lora_model, ft_loader, lr=1e-3)
    losses = finetuner.train(num_epochs=40)
    test_metrics = finetuner.evaluate(test_loader)

    n_lora = lora_model.trainable_params()
    pct = 100.0 * n_lora / total_base_params

    lora_results[rank] = {
        "trainable_params": n_lora,
        "pct_of_base": pct,
        "loss_history": losses,
        "test_loss": test_metrics["test_loss"],
    }

    print(f"  {rank:>6} {n_lora:>12,} {pct:>9.1f}% "
          f"{losses[-1]:>12.4f} {test_metrics['test_loss']:>12.4f}")

results["lora_finetune"] = {
    r: {k: v for k, v in d.items() if k != "loss_history"}
    for r, d in lora_results.items()
}


# ── 3. Full fine-tuning comparison ────────────────────────────
print(f"\n{'=' * 60}")
print("LORA BENCHMARK 3: Full Fine-Tuning (all params unfrozen)")
print("=" * 60)

torch.manual_seed(42)
full_model = copy.deepcopy(pretrained_model)
ft_loader = get_finetune_loader(batch_size=16, n_samples=200)
full_finetuner = FullFinetuner(full_model, ft_loader, lr=1e-3)
full_losses = full_finetuner.train(num_epochs=40)
full_test = full_finetuner.evaluate(test_loader)

full_trainable = sum(p.numel() for p in full_model.parameters())
print(f"  Full FT params: {full_trainable:,} (100%)")
print(f"  Final loss:     {full_losses[-1]:.4f}")
print(f"  Test loss:      {full_test['test_loss']:.4f}")

results["full_finetune"] = {
    "trainable_params": full_trainable,
    "final_loss": full_losses[-1],
    "test_loss": full_test["test_loss"],
}

# Summary table
print(f"\n  {'Method':<18} {'Params':>10} {'Train Loss':>12} {'Test Loss':>12}")
print(f"  {'-' * 54}")
print(f"  {'No finetune':<18} {'0':>10} {'—':>12} {baseline_loss:>12.4f}")
for rank in lora_ranks:
    r = lora_results[rank]
    print(f"  {'LoRA r=' + str(rank):<18} {r['trainable_params']:>10,} "
          f"{r['loss_history'][-1]:>12.4f} {r['test_loss']:>12.4f}")
print(f"  {'Full FT':<18} {full_trainable:>10,} "
      f"{full_losses[-1]:>12.4f} {full_test['test_loss']:>12.4f}")


# ── Plot 1: Training curves + parameter comparison ───────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(pretrain_losses, "k-", lw=2, label="Pretrain")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].set_title("Phase 1: Pretraining")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(lora_ranks)))
for i, rank in enumerate(lora_ranks):
    axes[1].plot(lora_results[rank]["loss_history"], "-", color=colors[i],
                 lw=2, label=f"LoRA r={rank}")
axes[1].plot(full_losses, "r--", lw=2, label="Full FT")
axes[1].axhline(baseline_loss, color="gray", ls=":", lw=1, label="No finetune")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MSE Loss")
axes[1].set_title("Phase 2: Fine-Tuning Comparison")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

bar_labels = [f"r={r}" for r in lora_ranks] + ["Full"]
bar_params = [lora_results[r]["trainable_params"] for r in lora_ranks] + [full_trainable]
bar_losses = [lora_results[r]["test_loss"] for r in lora_ranks] + [full_test["test_loss"]]
bar_colors = list(colors) + ["red"]

x_pos = np.arange(len(bar_labels))
bars = axes[2].bar(x_pos, bar_params, color=bar_colors, alpha=0.7, edgecolor="black")
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(bar_labels)
axes[2].set_ylabel("Trainable Parameters")
axes[2].set_title("Parameter Efficiency")
axes[2].grid(True, alpha=0.3, axis="y")

ax2t = axes[2].twinx()
ax2t.plot(x_pos, bar_losses, "k^-", ms=8, lw=2, label="Test Loss")
ax2t.set_ylabel("Test Loss")
ax2t.legend(loc="upper left")

plt.tight_layout()
save("lora_training_comparison.png")


# ── 4. SVD analysis of weight updates ─────────────────────────
print(f"\n{'=' * 60}")
print("LORA BENCHMARK 4: Weight Update SVD Analysis")
print("=" * 60)

svd_stats = NystromLoRAAnalyzer.analyze_rank_structure(
    pretrained_model, full_model
)

print(f"\n  {'Layer':<18} {'Shape':>12} {'‖ΔW‖_F':>10} "
      f"{'r@90%':>8} {'r@95%':>8} {'r@99%':>8} {'σ₁/σₙ':>10}")
print(f"  {'-' * 78}")

for name, stats in svd_stats.items():
    er = stats["effective_ranks"]
    shape_str = f"{len(stats['singular_values'])}×{len(stats['singular_values'])}"
    print(f"  {name:<18} {shape_str:>12} {stats['full_frobenius_norm']:>10.4f} "
          f"{er['90%']:>8} {er['95%']:>8} {er['99%']:>8} "
          f"{stats['top_sv_ratio']:>10.0f}")

results["svd_analysis"] = {
    name: {
        "effective_ranks": s["effective_ranks"],
        "full_norm": s["full_frobenius_norm"],
        "top_sv_ratio": s["top_sv_ratio"],
        "top_10_sv": s["singular_values"][:10],
    }
    for name, s in svd_stats.items()
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
layer_names = list(svd_stats.keys())

for i, name in enumerate(layer_names):
    sv = np.array(svd_stats[name]["singular_values"])
    sv_norm = sv / (sv[0] + 1e-15)
    axes[0].semilogy(sv_norm, "o-", ms=4, lw=2, label=name.replace(".weight", ""))

axes[0].set_xlabel("Singular Value Index")
axes[0].set_ylabel("σᵢ / σ₁")
axes[0].set_title("Weight Update SVD: Rapid Decay → LoRA is Justified")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

for i, name in enumerate(layer_names):
    cumul = np.array(svd_stats[name]["cumulative_energy"])
    er = svd_stats[name]["effective_ranks"]
    axes[1].plot(cumul, "-", lw=2,
                 label=f"{name.replace('.weight', '')} (90% @ r={er['90%']})")

axes[1].axhline(0.90, color="gray", ls="--", alpha=0.5)
axes[1].axhline(0.95, color="gray", ls=":", alpha=0.5)
axes[1].set_xlabel("Rank r")
axes[1].set_ylabel("Cumulative Energy")
axes[1].set_title("Energy vs Rank (justifies low-rank LoRA)")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

test_ranks_bar = [2, 4, 8, 16]
fc2_sv = svd_stats.get("fc2.weight", list(svd_stats.values())[1])
fc2_cumul = np.array(fc2_sv["cumulative_energy"])
fc2_energies = [fc2_cumul[min(r - 1, len(fc2_cumul) - 1)] * 100
                for r in test_ranks_bar]
fc2_name = "fc2" if "fc2.weight" in svd_stats else list(svd_stats.keys())[1]
bar_c = plt.cm.viridis(np.linspace(0.2, 0.8, len(test_ranks_bar)))
axes[2].bar(range(len(test_ranks_bar)), fc2_energies, color=bar_c,
            edgecolor="black", alpha=0.8)
axes[2].set_xticks(range(len(test_ranks_bar)))
axes[2].set_xticklabels([f"r={r}" for r in test_ranks_bar])
axes[2].set_ylabel("Energy Captured (%)")
axes[2].set_title(f"LoRA Rank vs Energy ({fc2_name})")
axes[2].axhline(90, color="red", ls="--", alpha=0.5, label="90%")
axes[2].axhline(95, color="orange", ls="--", alpha=0.5, label="95%")
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis="y")
axes[2].set_ylim(0, 105)

plt.tight_layout()
save("lora_svd_analysis.png")


# ── 5. Nyström ↔ LoRA connection ──────────────────────────────
print(f"\n{'=' * 60}")
print("LORA BENCHMARK 5: Nyström ↔ LoRA Quantitative Connection")
print("=" * 60)

nystrom_conn = NystromLoRAConnection()
nystrom_analysis = nystrom_conn.analyze_weight_updates(
    pretrained_model, full_model, ranks=[1, 2, 4, 8, 16]
)

target_layer = "fc2.weight"
if target_layer not in nystrom_analysis:
    target_layer = list(nystrom_analysis.keys())[1]

print(f"\n  Layer: {target_layer}")
print(f"  {'Rank':>6} {'SVD Error':>12} {'Energy':>10}")
print(f"  {'-' * 30}")
for r, rd in sorted(nystrom_analysis[target_layer]["rank_data"].items()):
    print(f"  {r:>6} {rd['svd_relative_error']:>12.4f} "
          f"{rd['energy_captured'] * 100:>9.1f}%")

dW_target = (
    dict(full_model.named_parameters())[target_layer].detach().cpu().numpy()
    - dict(pretrained_model.named_parameters())[target_layer].detach().cpu().numpy()
)

test_ranks = [1, 2, 4, 8, 16, 32]
svd_errors = []
nys_errors_mean = []
nys_errors_std = []

np.random.seed(42)
for r in test_ranks:
    if r > min(dW_target.shape):
        continue
    res = nystrom_conn.nystrom_weight_approximation(dW_target, r, n_trials=20)
    svd_errors.append(res["svd_error"])
    nys_errors_mean.append(res["nystrom_error_mean"])
    nys_errors_std.append(res["nystrom_error_std"])

valid_ranks = test_ranks[: len(svd_errors)]

print(f"\n  Nyström vs eigendecomp of Gram matrix G = ΔW·ΔWᵀ ({target_layer}):")
print(f"  {'Rank':>6} {'Eig Error':>12} {'Nyström Mean':>14} {'Nyström Std':>14}")
print(f"  {'-' * 48}")
for i, r in enumerate(valid_ranks):
    print(f"  {r:>6} {svd_errors[i]:>12.4f} {nys_errors_mean[i]:>14.4f} "
          f"{nys_errors_std[i]:>14.4f}")

results["nystrom_connection"] = {
    "layer": target_layer,
    "ranks": valid_ranks,
    "svd_errors": svd_errors,
    "nystrom_errors_mean": nys_errors_mean,
    "nystrom_errors_std": nys_errors_std,
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].semilogy(valid_ranks, svd_errors, "b^-", ms=8, lw=2,
                 label="Eigendecomp (optimal)")
axes[0].errorbar(valid_ranks, nys_errors_mean, yerr=nys_errors_std,
                 fmt="rs-", ms=8, lw=2, capsize=4,
                 label="Nyström (column sampling)")
axes[0].set_xlabel("Rank r")
axes[0].set_ylabel("Relative Frobenius Error")
axes[0].set_title(f"Gram Matrix G = ΔW·ΔWᵀ ({target_layer.replace('.weight', '')})")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

sv = np.array(nystrom_analysis[target_layer]["singular_values"])
axes[1].semilogy(sv / sv[0], "k-o", ms=3, lw=2)
for r in [2, 4, 8]:
    if r < len(sv):
        axes[1].axvline(r, color="gray", ls="--", alpha=0.4)
        axes[1].annotate(f"r={r}", (r, sv[r] / sv[0]), fontsize=8)
axes[1].set_xlabel("Index i")
axes[1].set_ylabel("σᵢ / σ₁")
axes[1].set_title("Singular Value Decay of ΔW")
axes[1].grid(True, alpha=0.3)

analogy = [
    ["Full matrix", "A ∈ R^{N×N} (PDE op.)", "ΔW ∈ R^{d×d} (wt. update)"],
    ["Low-rank", "CW†C^T (Nyström)", "BA (LoRA)"],
    ["Why works", "Eigenvalue decay", "Singular value decay"],
    ["Cost: full", "O(N²)", "O(d²)"],
    ["Cost: low-rank", "O(Nr)", "O(dr)"],
    ["Landmarks", "Column samples", "Learned rank dirs"],
]
axes[2].axis("off")
table = axes[2].table(
    cellText=analogy,
    colLabels=["Concept", "Nyström (PDE)", "LoRA (Fine-tuning)"],
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.6)
for key, cell in table.get_celld().items():
    if key[0] == 0:
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("#D6E4F0" if key[0] % 2 == 1 else "#EDF2F9")
axes[2].set_title("Formal Nyström ↔ LoRA Correspondence", pad=20)

plt.tight_layout()
save("lora_nystrom_connection.png")

# ── Save JSON ─────────────────────────────────────────────────
elapsed = time.perf_counter() - t_start
results["elapsed_seconds"] = round(elapsed, 2)

json_path = os.path.join(DIR, "results", "lora_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2, default=default_ser)

print(f"\n{'=' * 60}")
print(f"  JSON: {json_path}")
print(f"  Total time: {elapsed:.1f}s")
print(f"  Done!")
print(f"{'=' * 60}")
