"""
Nyström attention layer and KV-cache compression for LLMs.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NystromAttentionLayer(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_landmarks=16, max_seq=128):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_landmarks = n_landmarks
        self.max_seq = max_seq

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        mask = torch.tril(torch.ones(max_seq, max_seq))
        self.register_buffer('causal_mask', mask.view(1, 1, max_seq, max_seq))

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        m = min(self.n_landmarks, T)
        seg = max(1, T // m)
        scale = 1.0 / math.sqrt(self.head_dim)

        indices = [slice(i * seg, min((i + 1) * seg, T)) for i in range(m)]
        q_land = torch.stack([q[:, :, sl, :].mean(dim=2) for sl in indices], dim=2)
        k_land = torch.stack([k[:, :, sl, :].mean(dim=2) for sl in indices], dim=2)

        F1 = F.softmax((q @ k_land.transpose(-2, -1)) * scale, dim=-1)
        A_tilde = F.softmax((q_land @ k_land.transpose(-2, -1)) * scale, dim=-1)
        F2 = F.softmax((q_land @ k.transpose(-2, -1)) * scale, dim=-1)

        A_inv = torch.linalg.pinv(A_tilde)
        for _ in range(3):
            eye = torch.eye(m, device=x.device).unsqueeze(0).unsqueeze(0)
            A_inv = A_inv @ (2 * eye - A_tilde @ A_inv)

        approx = F1 @ A_inv @ F2
        causal = self.causal_mask[:, :, :T, :T]
        approx = approx * causal
        approx = approx / (approx.sum(dim=-1, keepdim=True) + 1e-8)

        out = (approx @ v).transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out_proj(out)


class KVCacheCompressor:
    def __init__(self, method='nystrom'):
        self.method = method

    def compress(self, K, V, rank):
        T, D = K.shape
        if T <= rank:
            return K, V, 0.0

        if self.method == 'nystrom':
            return self._compress_nystrom(K, V, rank)
        else:
            return self._compress_svd(K, V, rank)

    def _compress_nystrom(self, K, V, rank):
        T, D = K.shape
        seg = max(1, T // rank)
        indices = [slice(i * seg, min((i + 1) * seg, T)) for i in range(rank)]

        K_compressed = torch.stack([K[sl].mean(dim=0) for sl in indices])
        V_compressed = torch.stack([V[sl].mean(dim=0) for sl in indices])

        scores_full = K @ K.T
        scores_approx = K @ K_compressed.T @ torch.linalg.pinv(K_compressed @ K_compressed.T) @ K_compressed @ K.T
        error = torch.norm(scores_full - scores_approx).item() / (torch.norm(scores_full).item() + 1e-8)

        return K_compressed, V_compressed, error

    def _compress_svd(self, K, V, rank):
        T, D = K.shape
        U_k, S_k, Vh_k = torch.linalg.svd(K, full_matrices=False)
        U_v, S_v, Vh_v = torch.linalg.svd(V, full_matrices=False)

        K_approx = (U_k[:, :rank] * S_k[:rank].unsqueeze(0)) @ Vh_k[:rank, :]
        V_approx = (U_v[:, :rank] * S_v[:rank].unsqueeze(0)) @ Vh_v[:rank, :]

        k_err = torch.norm(K - K_approx).item() / (torch.norm(K).item() + 1e-8)
        v_err = torch.norm(V - V_approx).item() / (torch.norm(V).item() + 1e-8)

        K_compressed = (U_k[:rank, :rank] * S_k[:rank].unsqueeze(0)) @ Vh_k[:rank, :]
        V_compressed = (U_v[:rank, :rank] * S_v[:rank].unsqueeze(0)) @ Vh_v[:rank, :]

        return K_compressed, V_compressed, (k_err + v_err) / 2

    def measure_reconstruction_error(self, K, V, ranks):
        results = []
        for r in ranks:
            _, _, err = self.compress(K, V, r)
            results.append({'rank': r, 'error': err})
        return results


def measure_attention_spectrum(model, dataloader, device='cpu', max_batches=5):
    model.eval()
    all_eigenvalues = []

    with torch.no_grad():
        for batch_idx, (input_ids, _) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            input_ids = input_ids.to(device)
            B, T = input_ids.shape

            pos = torch.arange(0, T, device=device).unsqueeze(0)
            x = model.tok_emb(input_ids) + model.pos_emb(pos)

            block = model.blocks[0]
            x_normed = block.ln1(x)

            if hasattr(block.attn, 'qkv'):
                qkv = block.attn.qkv(x_normed)
                qkv = qkv.reshape(B, T, 3, block.attn.n_heads, block.attn.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k = qkv[0], qkv[1]
            else:
                q = block.attn.q_proj(x_normed).view(B, T, block.attn.n_heads, block.attn.head_dim).transpose(1, 2)
                k = block.attn.k_proj(x_normed).view(B, T, block.attn.n_heads, block.attn.head_dim).transpose(1, 2)

            scale = 1.0 / math.sqrt(block.attn.head_dim)
            A = (q @ k.transpose(-2, -1)) * scale
            A = F.softmax(A, dim=-1)

            A_mean = A.mean(dim=(0, 1)).cpu().numpy()
            eigvals = np.sort(np.abs(np.linalg.eigvalsh(A_mean)))[::-1]
            all_eigenvalues.append(eigvals)

    return np.mean(all_eigenvalues, axis=0) if all_eigenvalues else np.array([])
