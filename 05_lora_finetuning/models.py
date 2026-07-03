"""
LoRA model components: LoRALinear, BaseModel, LoRAModel, NystromLoRAAnalyzer.

Implements proper LoRA fine-tuning with frozen base weights and trainable
low-rank adapters (A, B), plus SVD analysis of weight updates.
"""

import copy
import numpy as np
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a frozen base weight and
    trainable low-rank adapter: forward(x) = x @ (W + B @ A)^T + bias."""

    def __init__(self, base_linear: nn.Linear, rank: int = 4):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank

        self.weight = nn.Parameter(base_linear.weight.data.clone())
        self.weight.requires_grad = False

        self.bias = None
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.data.clone())
            self.bias.requires_grad = False

        self.A = nn.Parameter(torch.randn(rank, self.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(self.out_features, rank))

    def forward(self, x):
        W_eff = self.weight + self.B @ self.A
        out = x @ W_eff.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def merge(self):
        """Fold the LoRA adapter into the base weight (irreversible)."""
        self.weight.data += self.B.data @ self.A.data
        self.A.data.zero_()
        self.B.data.zero_()

    def lora_params(self) -> int:
        return self.A.numel() + self.B.numel()


class BaseModel(nn.Module):
    """3-layer MLP used as the 'pretrained' base model.

    Architecture: Linear(d_in→d_hidden) → ReLU → Linear(d_hidden→d_hidden) → ReLU → Linear(d_hidden→d_out)
    """

    def __init__(self, d_in: int = 64, d_hidden: int = 128, d_out: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False


class LoRAModel(nn.Module):
    """Wraps a BaseModel, replacing specified Linear layers with LoRALinear.

    Only LoRA adapter parameters (A, B) are trainable; all base weights
    are frozen.
    """

    def __init__(self, base_model: BaseModel, rank: int = 4, target_layers=None):
        super().__init__()
        self.model = copy.deepcopy(base_model)
        self.model.freeze()
        self.lora_layers: dict[str, LoRALinear] = {}

        if target_layers is None:
            target_layers = [
                name for name, mod in self.model.named_modules()
                if isinstance(mod, nn.Linear)
            ]

        for name in target_layers:
            base_linear = getattr(self.model, name)
            lora_linear = LoRALinear(base_linear, rank=rank)
            setattr(self.model, name, lora_linear)
            self.lora_layers[name] = lora_linear

    def forward(self, x):
        return self.model(x)

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class NystromLoRAAnalyzer:
    """Analyze the rank structure of weight updates ΔW = W_finetuned − W_pretrained.

    Shows that full fine-tuning weight updates are approximately low-rank
    (rapid singular value decay), which justifies LoRA's rank-r constraint.
    """

    @staticmethod
    def analyze_rank_structure(model_before: nn.Module, model_after: nn.Module):
        """Compute SVD of ΔW for each weight matrix and return decay statistics.

        Returns a dict mapping layer names to {singular_values, cumulative_energy,
        effective_ranks, full_frobenius_norm}.
        """
        before_params = {n: p.detach().cpu().numpy()
                         for n, p in model_before.named_parameters()}
        after_params = {n: p.detach().cpu().numpy()
                        for n, p in model_after.named_parameters()}

        results = {}
        for name in before_params:
            if "weight" not in name or before_params[name].ndim != 2:
                continue

            dW = after_params[name] - before_params[name]
            full_norm = float(np.linalg.norm(dW, "fro"))
            if full_norm < 1e-12:
                continue

            _, S, _ = np.linalg.svd(dW, full_matrices=False)
            cumul = np.cumsum(S ** 2) / np.sum(S ** 2)

            r90 = int(np.searchsorted(cumul, 0.90) + 1)
            r95 = int(np.searchsorted(cumul, 0.95) + 1)
            r99 = int(np.searchsorted(cumul, 0.99) + 1)

            results[name] = {
                "singular_values": S.tolist(),
                "cumulative_energy": cumul.tolist(),
                "full_frobenius_norm": full_norm,
                "effective_ranks": {"90%": r90, "95%": r95, "99%": r99},
                "top_sv_ratio": float(S[0] / (S[-1] + 1e-15)),
            }

        return results
