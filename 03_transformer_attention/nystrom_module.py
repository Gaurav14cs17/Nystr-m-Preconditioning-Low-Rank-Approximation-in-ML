import time
import numpy as np
import torch
import scipy.sparse as sp

from models import TransformerClassifier


class NystromAttentionAnalyzer:
    def __init__(self, device='cpu'):
        self.device = device

    @torch.no_grad()
    def compare_attention_outputs(self, model_full, model_nystrom, data_loader):
        model_full.eval()
        model_nystrom.eval()

        total_error = 0.0
        total_norm = 0.0
        logit_errors = []

        for input_ids, _ in data_loader:
            input_ids = input_ids.to(self.device)
            logits_full, weights_full = model_full(input_ids, return_weights=True)
            logits_nystrom, weights_nystrom = model_nystrom(input_ids, return_weights=True)

            logit_diff = (logits_full - logits_nystrom).norm().item()
            logit_errors.append(logit_diff)

            if weights_full and weights_nystrom:
                for wf, wn in zip(weights_full, weights_nystrom):
                    total_error += (wf - wn).norm().item()
                    total_norm += wf.norm().item()

        return {
            'relative_attn_error': total_error / max(total_norm, 1e-8),
            'mean_logit_diff': float(np.mean(logit_errors)),
        }

    @torch.no_grad()
    def eigenvalue_analysis(self, attention_weights):
        """Compute eigenvalue spectrum and effective rank of attention matrices."""
        if isinstance(attention_weights, torch.Tensor):
            A = attention_weights.cpu().numpy()
        else:
            A = np.array(attention_weights)

        if A.ndim == 4:
            A = A[0, 0]
        elif A.ndim == 3:
            A = A[0]

        eigvals = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
        eigvals_normalized = eigvals / (eigvals[0] + 1e-15)

        cumulative_energy = np.cumsum(eigvals ** 2) / (np.sum(eigvals ** 2) + 1e-15)
        rank_90 = int(np.searchsorted(cumulative_energy, 0.9) + 1)

        p = eigvals / (eigvals.sum() + 1e-15)
        p = p[p > 1e-15]
        effective_rank = float(np.exp(-np.sum(p * np.log(p))))

        return {
            'eigenvalues': eigvals,
            'normalized': eigvals_normalized,
            'cumulative_energy': cumulative_energy,
            'rank_90': rank_90,
            'effective_rank': effective_rank,
            'condition_ratio': float(eigvals[0] / (eigvals[-1] + 1e-15)),
        }

    def poisson_vs_attention_spectrum(self, seq_len):
        """Side-by-side spectral comparison of 1D Poisson matrix vs attention kernel."""
        n = seq_len
        e = np.ones(n)
        L = sp.diags([-e[1:], 2 * e, -e[1:]], [-1, 0, 1]).toarray()
        poisson_eigvals = np.sort(np.linalg.eigvalsh(L))[::-1]

        pos = np.linspace(0, 2 * np.pi, n)[:, None]
        freqs = np.arange(1, 17)[None, :]
        features = np.concatenate([np.sin(pos * freqs), np.cos(pos * freqs)], axis=1)
        d = features.shape[1]
        scores = features @ features.T / np.sqrt(d)
        e_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        attn_matrix = e_scores / e_scores.sum(axis=1, keepdims=True)
        attn_eigvals = np.sort(np.abs(np.linalg.eigvals(attn_matrix)))[::-1]

        return {
            'poisson_eigvals': poisson_eigvals,
            'attn_eigvals': attn_eigvals,
            'poisson_condition': float(poisson_eigvals[0] / (poisson_eigvals[-1] + 1e-15)),
            'attn_condition': float(attn_eigvals[0] / (attn_eigvals[-1] + 1e-15)),
        }

    def benchmark_speed(self, seq_lengths, d_model=64, n_landmarks=8, n_heads=4, n_repeats=5):
        """Timing comparison between full and Nyström attention for various sequence lengths."""
        results = []
        for N in seq_lengths:
            model_full = TransformerClassifier(
                d_model=d_model, n_heads=n_heads, max_seq_len=N,
                n_layers=1, attn_type='full'
            ).to(self.device)
            model_nystrom = TransformerClassifier(
                d_model=d_model, n_heads=n_heads, max_seq_len=N,
                n_layers=1, attn_type='nystrom', n_landmarks=n_landmarks
            ).to(self.device)

            x = torch.randint(0, 128, (4, N))

            model_full.eval()
            model_nystrom.eval()
            with torch.no_grad():
                model_full(x)
                model_nystrom(x)

                t0 = time.perf_counter()
                for _ in range(n_repeats):
                    model_full(x)
                t_full = (time.perf_counter() - t0) / n_repeats

                t0 = time.perf_counter()
                for _ in range(n_repeats):
                    model_nystrom(x)
                t_nystrom = (time.perf_counter() - t0) / n_repeats

            results.append({
                'seq_len': N,
                'full_ms': t_full * 1000,
                'nystrom_ms': t_nystrom * 1000,
                'speedup': t_full / max(t_nystrom, 1e-9),
            })

        return results
