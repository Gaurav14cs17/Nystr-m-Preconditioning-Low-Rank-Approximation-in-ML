"""
Causal Language Model with Full and Nyström attention variants.
Tiny but correct implementation for benchmarking preconditioning methods.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4, max_seq=128, mode='full', n_landmarks=16):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq = max_seq
        self.mode = mode
        self.n_landmarks = n_landmarks

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        mask = torch.tril(torch.ones(max_seq, max_seq))
        self.register_buffer('causal_mask', mask.view(1, 1, max_seq, max_seq))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.mode == 'full':
            out = self._full_attention(q, k, v, T)
        else:
            out = self._nystrom_attention(q, k, v, T)

        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out_proj(out)

    def _full_attention(self, q, k, v, T):
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        return attn @ v

    def _nystrom_attention(self, q, k, v, T):
        m = min(self.n_landmarks, T)
        seg = max(1, T // m)

        q_land = q.unfold(2, seg, seg).mean(dim=-1)[:, :, :m, :]
        k_land = k.unfold(2, seg, seg).mean(dim=-1)[:, :, :m, :]

        scale = 1.0 / math.sqrt(self.head_dim)

        F1 = F.softmax((q @ k_land.transpose(-2, -1)) * scale, dim=-1)
        A_tilde = F.softmax((q_land @ k_land.transpose(-2, -1)) * scale, dim=-1)
        F2 = F.softmax((q_land @ k.transpose(-2, -1)) * scale, dim=-1)

        A_inv = torch.linalg.pinv(A_tilde)
        for _ in range(3):
            A_inv = A_inv @ (2 * torch.eye(m, device=q.device).unsqueeze(0).unsqueeze(0) - A_tilde @ A_inv)

        causal = self.causal_mask[:, :, :T, :T]
        approx = F1 @ A_inv @ F2
        approx = approx * causal
        approx = approx / (approx.sum(dim=-1, keepdim=True) + 1e-8)

        return approx @ v


class NystromCausalAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4, max_seq=128, n_landmarks=16):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq = max_seq
        self.n_landmarks = n_landmarks

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

        out = approx @ v
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, max_seq=128, attention_mode='full', n_landmarks=16):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        if attention_mode == 'nystrom':
            self.attn = NystromCausalAttention(d_model, n_heads, max_seq, n_landmarks)
        else:
            self.attn = CausalSelfAttention(d_model, n_heads, max_seq, mode=attention_mode, n_landmarks=n_landmarks)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class CausalLM(nn.Module):
    def __init__(self, vocab_size=256, d_model=64, n_heads=4, n_layers=2,
                 max_seq=128, attention_mode='full', n_landmarks=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq = max_seq

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq, attention_mode, n_landmarks)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.tok_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        B, T = input_ids.shape
        assert T <= self.max_seq

        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


class KVCache:
    def __init__(self):
        self.keys = None
        self.values = None

    def update(self, new_k, new_v):
        if self.keys is None:
            self.keys = new_k
            self.values = new_v
        else:
            self.keys = torch.cat([self.keys, new_k], dim=2)
            self.values = torch.cat([self.values, new_v], dim=2)
        return self.keys, self.values

    @property
    def seq_len(self):
        return 0 if self.keys is None else self.keys.shape[2]

    def compress(self, rank):
        if self.keys is None or self.keys.shape[2] <= rank:
            return self.keys, self.values

        B, H, T, D = self.keys.shape
        k_flat = self.keys.reshape(B * H, T, D)
        v_flat = self.values.reshape(B * H, T, D)

        U_k, S_k, Vh_k = torch.linalg.svd(k_flat, full_matrices=False)
        U_v, S_v, Vh_v = torch.linalg.svd(v_flat, full_matrices=False)

        k_compressed = (U_k[:, :rank, :] * S_k[:, :rank].unsqueeze(1)) @ Vh_k[:, :rank, :]
        v_compressed = (U_v[:, :rank, :] * S_v[:, :rank].unsqueeze(1)) @ Vh_v[:, :rank, :]

        self.keys = k_compressed.reshape(B, H, rank, D)
        self.values = v_compressed.reshape(B, H, rank, D)
        return self.keys, self.values

    def compress_nystrom(self, rank):
        if self.keys is None or self.keys.shape[2] <= rank:
            return self.keys, self.values

        B, H, T, D = self.keys.shape
        seg = max(1, T // rank)

        indices = [slice(i * seg, min((i + 1) * seg, T)) for i in range(rank)]
        k_land = torch.stack([self.keys[:, :, sl, :].mean(dim=2) for sl in indices], dim=2)
        v_land = torch.stack([self.values[:, :, sl, :].mean(dim=2) for sl in indices], dim=2)

        self.keys = k_land
        self.values = v_land
        return self.keys, self.values
