import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class NystromAttentionBlock(nn.Module):
    """Nyströmformer attention: segment-mean landmarks, 3-matrix decomposition."""

    def __init__(self, channels, num_landmarks=8, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_landmarks = num_landmarks
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.scale = self.head_dim ** -0.5

    def _segment_means(self, x, m):
        """Compute segment-mean landmarks by splitting sequence into m segments."""
        B, H, N, D = x.shape
        seg = max(1, N // m)
        actual_m = min(m, N)
        landmarks = []
        for i in range(actual_m):
            start = i * seg
            end = min((i + 1) * seg, N)
            landmarks.append(x[:, :, start:end, :].mean(dim=2))
        return torch.stack(landmarks, dim=2)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        residual = x
        x = self.norm(x)
        x = x.view(B, C, N)

        qkv = self.qkv(x).view(B, 3, self.num_heads, self.head_dim, N)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.permute(0, 1, 3, 2)  # B, H, N, D
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        m = min(self.num_landmarks, N)
        q_land = self._segment_means(q, m)
        k_land = self._segment_means(k, m)

        # 3-matrix Nyström decomposition
        F1 = F.softmax(q @ k_land.transpose(-1, -2) * self.scale, dim=-1)       # B,H,N,m
        A_tilde = F.softmax(q_land @ k_land.transpose(-1, -2) * self.scale, dim=-1)  # B,H,m,m
        F2 = F.softmax(q_land @ k.transpose(-1, -2) * self.scale, dim=-1)       # B,H,m,N

        # Iterative pseudo-inverse for stability
        A_inv = torch.linalg.pinv(A_tilde)
        for _ in range(3):
            A_inv = A_inv @ (2 * torch.eye(m, device=x.device) - A_tilde @ A_inv)

        out = F1 @ (A_inv @ (F2 @ v))
        out = out.permute(0, 1, 3, 2).reshape(B, C, N)
        out = self.proj(out).view(B, C, H, W)
        return out + residual

    def full_attention(self, x):
        """Standard O(N^2) attention for comparison."""
        B, C, H, W = x.shape
        N = H * W
        x_flat = self.norm(x).view(B, C, N)

        qkv = self.qkv(x_flat).view(B, 3, self.num_heads, self.head_dim, N)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        attn = F.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1)
        out = (attn @ v).permute(0, 1, 3, 2).reshape(B, C, N)
        out = self.proj(out).view(B, C, H, W)
        return out + x


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNet(nn.Module):
    """Small UNet for 1-channel 28x28 images. channels=[32,64], attention at 7x7."""

    def __init__(self, in_ch=1, channels=(32, 64), time_dim=64, num_landmarks=8):
        super().__init__()
        self.time_dim = time_dim
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv2d(in_ch, channels[0], 3, padding=1)

        # Encoder: 28x28 -> 14x14 -> 7x7
        self.enc1 = ResBlock(channels[0], channels[0], time_dim)
        self.down1 = Downsample(channels[0])
        self.enc2 = ResBlock(channels[0], channels[1], time_dim)
        self.down2 = Downsample(channels[1])

        # Middle at 7x7
        self.mid1 = ResBlock(channels[1], channels[1], time_dim)
        self.mid_attn = NystromAttentionBlock(channels[1], num_landmarks=num_landmarks)
        self.mid2 = ResBlock(channels[1], channels[1], time_dim)

        # Decoder: 7x7 -> 14x14 -> 28x28
        self.up2 = Upsample(channels[1])
        self.dec2 = ResBlock(channels[1] * 2, channels[0], time_dim)
        self.up1 = Upsample(channels[0])
        self.dec1 = ResBlock(channels[0] * 2, channels[0], time_dim)

        self.out_norm = nn.GroupNorm(min(8, channels[0]), channels[0])
        self.out_conv = nn.Conv2d(channels[0], in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        h = self.input_conv(x)

        # Encoder
        h1 = self.enc1(h, t_emb)         # 28x28
        h = self.down1(h1)               # 14x14
        h2 = self.enc2(h, t_emb)         # 14x14
        h = self.down2(h2)               # 7x7

        # Middle
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        # Decoder with skip connections
        h = self.up2(h)                  # 14x14
        h = self.dec2(torch.cat([h, h2], dim=1), t_emb)
        h = self.up1(h)                  # 28x28
        h = self.dec1(torch.cat([h, h1], dim=1), t_emb)

        return self.out_conv(F.silu(self.out_norm(h)))


class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps=200):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('sqrt_alpha_bar', torch.sqrt(alpha_bar))
        self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1.0 - alpha_bar))

    def q_sample(self, x_0, t, noise=None):
        """Forward process: q(x_t | x_0) = N(sqrt(alpha_bar_t)*x_0, (1-alpha_bar_t)*I)"""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_1_ab = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return sqrt_ab * x_0 + sqrt_1_ab * noise

    def training_loss(self, x_0):
        B = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_t, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, shape, device='cpu'):
        """Reverse process: iteratively denoise from x_T ~ N(0, I)."""
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            pred_noise = self.model(x, t)

            alpha = self.alphas[i]
            alpha_bar = self.alpha_bar[i]
            beta = self.betas[i]

            # mu_theta(x_t, t) = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta)
            mean = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise)

            if i > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x)
            else:
                x = mean
        return x
