import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron with configurable depth, width, and optional BatchNorm."""

    def __init__(self, d_in=10, d_hidden=32, d_out=2, n_layers=3, use_batchnorm=False):
        super().__init__()
        layers = []
        prev = d_in
        for i in range(n_layers - 1):
            layers.append(nn.Linear(prev, d_hidden))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(d_hidden))
            layers.append(nn.ReLU())
            prev = d_hidden
        layers.append(nn.Linear(prev, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """Small convolutional network for 8×8 or 16×16 image classification."""

    def __init__(self, in_channels=1, img_size=8, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class NystromPreconditioner:
    """
    Approximates the Hessian inverse via a rank-r Nyström sketch.

    Uses Hessian-vector products (autograd) to build the sketch, then applies
    (H_nys + λI)^{-1} to gradients via the Woodbury identity in O(d·r).
    """

    def __init__(self, rank=20, damping=1.0):
        self.rank = rank
        self.damping = damping
        self.U = None
        self.sigma = None
        self._built = False

    def build(self, model, loss_fn, data_batch):
        """Compute low-rank Nyström approximation of the Hessian."""
        params = [p for p in model.parameters() if p.requires_grad]
        d = sum(p.numel() for p in params)
        r = min(self.rank, d)

        x_data, y_data = data_batch
        loss = loss_fn(model(x_data), y_data)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grad = torch.cat([g.reshape(-1) for g in grads])

        Omega = torch.randn(d, r)
        Omega, _ = torch.linalg.qr(Omega)

        Y = torch.zeros(d, r)
        for i in range(r):
            gv = flat_grad @ Omega[:, i]
            hvp = torch.autograd.grad(gv, params, retain_graph=True)
            Y[:, i] = torch.cat([h.reshape(-1) for h in hvp]).detach()

        if not torch.isfinite(Y).all():
            return

        B = Omega.T @ Y
        B = 0.5 * (B + B.T)

        if not torch.isfinite(B).all():
            return

        U_b, S_b, _ = torch.linalg.svd(B)
        S_b = torch.clamp(S_b, min=1e-8)

        Z = Y @ U_b @ torch.diag(1.0 / torch.sqrt(S_b))
        U, S, _ = torch.linalg.svd(Z, full_matrices=False)
        self.U = U.detach()
        self.sigma = (S ** 2).detach()
        self._built = True

    def apply(self, gradient_flat):
        """Apply (H_nys + λI)^{-1} to a gradient vector via Woodbury identity."""
        if not self._built:
            return gradient_flat.clone()
        lam = self.damping
        Ut_g = self.U.T @ gradient_flat
        correction = self.U @ ((self.sigma / (self.sigma + lam)) * Ut_g)
        return (gradient_flat - correction) / lam

    @property
    def spectrum(self):
        if not self._built:
            return None
        return self.sigma.numpy().copy()


class PreconditionedSGD(torch.optim.Optimizer):
    """SGD with Nyström Hessian preconditioning: θ -= lr · (H_nys + λI)^{-1} · ∇L."""

    def __init__(self, params, lr=0.01, preconditioner=None):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.preconditioner = preconditioner

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        all_params = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    all_params.append(p)

        if not all_params:
            return loss

        if self.preconditioner is not None and self.preconditioner._built:
            flat_grad = torch.cat([p.grad.reshape(-1) for p in all_params])
            step_dir = self.preconditioner.apply(flat_grad)
            offset = 0
            for group in self.param_groups:
                lr = group["lr"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    n = p.numel()
                    p.add_(step_dir[offset : offset + n].reshape(p.shape), alpha=-lr)
                    offset += n
        else:
            for group in self.param_groups:
                lr = group["lr"]
                for p in group["params"]:
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-lr)

        return loss
