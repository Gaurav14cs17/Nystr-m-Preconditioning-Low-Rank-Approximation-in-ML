import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FullAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, return_weights=False):
        B, N, _ = x.shape
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.W_o(out)

        if return_weights:
            return out, attn_weights
        return out, None


class NystromAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_landmarks=8, n_iter=3):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)
        self.n_landmarks = n_landmarks
        self.n_iter = n_iter
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def _segment_means(self, x, m):
        B, H, N, D = x.shape
        seg_len = N // m
        usable = seg_len * m
        x_trunc = x[:, :, :usable, :].reshape(B, H, m, seg_len, D)
        return x_trunc.mean(dim=3)

    def _iterative_pinv(self, A, n_iter):
        I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
        I = I.unsqueeze(0).unsqueeze(0).expand_as(A)
        Z = A.transpose(-2, -1) / (torch.norm(A, dim=(-2, -1), keepdim=True) ** 2 + 1e-6)
        for _ in range(n_iter):
            Z = Z @ (2 * I - A @ Z)
        return Z

    def forward(self, x, return_weights=False):
        B, N, _ = x.shape
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        m = min(self.n_landmarks, N)
        Q_landmarks = self._segment_means(Q, m)
        K_landmarks = self._segment_means(K, m)

        F1 = F.softmax(torch.matmul(Q, K_landmarks.transpose(-2, -1)) / self.scale, dim=-1)
        F2 = F.softmax(torch.matmul(Q_landmarks, K.transpose(-2, -1)) / self.scale, dim=-1)
        A_tilde = F.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-2, -1)) / self.scale, dim=-1)

        A_tilde_inv = self._iterative_pinv(A_tilde, self.n_iter)
        out = torch.matmul(F1, torch.matmul(A_tilde_inv, torch.matmul(F2, V)))

        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.W_o(out)

        if return_weights:
            approx_weights = torch.matmul(F1, torch.matmul(A_tilde_inv, F2))
            return out, approx_weights
        return out, None


class LinearAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def _elu_feature_map(self, x):
        return F.elu(x) + 1

    def forward(self, x, return_weights=False):
        B, N, _ = x.shape
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        Q = self._elu_feature_map(Q)
        K = self._elu_feature_map(K)

        KV = torch.matmul(K.transpose(-2, -1), V)
        Z = 1.0 / (torch.matmul(Q, K.sum(dim=-2, keepdim=True).transpose(-2, -1)) + 1e-6)
        out = torch.matmul(Q, KV) * Z

        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.W_o(out)

        if return_weights:
            attn = torch.matmul(Q, K.transpose(-2, -1))
            attn = attn * (1.0 / (attn.sum(dim=-1, keepdim=True) + 1e-6))
            return out, attn
        return out, None


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, ff_dim=128, attn_type='full', n_landmarks=8):
        super().__init__()
        if attn_type == 'full':
            self.attn = FullAttention(d_model, n_heads)
        elif attn_type == 'nystrom':
            self.attn = NystromAttention(d_model, n_heads, n_landmarks=n_landmarks)
        elif attn_type == 'linear':
            self.attn = LinearAttention(d_model, n_heads)
        else:
            raise ValueError(f"Unknown attention type: {attn_type}")

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )

    def forward(self, x, return_weights=False):
        residual = x
        x = self.norm1(x)
        attn_out, weights = self.attn(x, return_weights=return_weights)
        x = residual + attn_out

        residual = x
        x = residual + self.ffn(self.norm2(x))
        return x, weights


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size=128, max_seq_len=64, d_model=64, n_heads=4,
                 n_layers=2, n_classes=2, attn_type='full', n_landmarks=8):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, ff_dim=d_model * 2,
                                   attn_type=attn_type, n_landmarks=n_landmarks)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, n_classes)
        self.d_model = d_model

    def forward(self, input_ids, return_weights=False):
        B, N = input_ids.shape
        positions = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, N)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        all_weights = []
        for layer in self.layers:
            x, w = layer(x, return_weights=return_weights)
            if return_weights:
                all_weights.append(w)

        pooled = x.mean(dim=1)
        logits = self.head(pooled)

        if return_weights:
            return logits, all_weights
        return logits, None
