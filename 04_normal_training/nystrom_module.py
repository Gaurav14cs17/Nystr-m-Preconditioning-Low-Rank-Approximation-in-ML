import os
import sys

import numpy as np
import torch
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import NystromPreconditioner  # noqa: E402
from trainer import Trainer  # noqa: E402


class HessianApproximator:
    """Hessian analysis tools using autograd Hessian-vector products."""

    @staticmethod
    def _hvp_batch(model, loss_fn, data_batch, V):
        """Compute H @ V for V ∈ R^{d×r} using a shared computation graph."""
        params = [p for p in model.parameters() if p.requires_grad]
        d = sum(p.numel() for p in params)

        x_data, y_data = data_batch
        loss = loss_fn(model(x_data), y_data)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grad = torch.cat([g.reshape(-1) for g in grads])

        r = V.shape[1]
        Y = torch.zeros(d, r)
        for i in range(r):
            gv = flat_grad @ V[:, i]
            hvp = torch.autograd.grad(gv, params, retain_graph=True)
            Y[:, i] = torch.cat([h.reshape(-1) for h in hvp]).detach()
        return Y

    @staticmethod
    def randomized_svd(model, loss_fn, data, rank=20):
        """Low-rank Hessian approximation H ≈ U diag(S) U^T via randomized SVD."""
        params = [p for p in model.parameters() if p.requires_grad]
        d = sum(p.numel() for p in params)
        r = min(rank, d)

        Omega = torch.randn(d, r)
        Omega, _ = torch.linalg.qr(Omega)

        Y = HessianApproximator._hvp_batch(model, loss_fn, data, Omega)
        Q, _ = torch.linalg.qr(Y)

        HQ = HessianApproximator._hvp_batch(model, loss_fn, data, Q)
        B = Q.T @ HQ
        B = 0.5 * (B + B.T)

        U_b, S_b, _ = torch.linalg.svd(B)
        U = Q @ U_b
        return U.detach(), S_b.detach(), U.detach()

    @staticmethod
    def nystrom_hessian(model, loss_fn, data, rank=20):
        """Nyström approximation returning (U, σ) with H ≈ U diag(σ) U^T."""
        params = [p for p in model.parameters() if p.requires_grad]
        d = sum(p.numel() for p in params)
        r = min(rank, d)

        Omega = torch.randn(d, r)
        Omega, _ = torch.linalg.qr(Omega)

        Y = HessianApproximator._hvp_batch(model, loss_fn, data, Omega)

        B = Omega.T @ Y
        B = 0.5 * (B + B.T)

        U_b, S_b, _ = torch.linalg.svd(B)
        S_b = torch.clamp(S_b, min=1e-8)

        Z = Y @ U_b @ torch.diag(1.0 / torch.sqrt(S_b))
        U, S, _ = torch.linalg.svd(Z, full_matrices=False)
        return U.detach(), (S ** 2).detach()

    @staticmethod
    def condition_number(sigma):
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.numpy()
        pos = np.abs(sigma)
        pos = pos[pos > 1e-10]
        if len(pos) < 2:
            return float("inf")
        return float(pos.max() / pos.min())


class NystromOptimizer:
    """
    Wraps any base optimizer with periodic Nyström Hessian preconditioning.

    Every `rebuild_every` steps, recomputes the rank-r Nyström sketch of the
    Hessian, then replaces raw gradients with preconditioned gradients
    (H_nys + λI)^{-1} g before handing off to the base optimizer.
    """

    def __init__(self, model, base_optimizer, loss_fn,
                 rank=15, damping=1.0, rebuild_every=5):
        self.model = model
        self.base_optimizer = base_optimizer
        self.loss_fn = loss_fn
        self.preconditioner = NystromPreconditioner(rank=rank, damping=damping)
        self.rebuild_every = rebuild_every
        self.step_count = 0
        self._current_batch = None

    def set_batch(self, batch):
        self._current_batch = batch

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def step(self, closure=None):
        if self._current_batch is not None and self.step_count % self.rebuild_every == 0:
            self.preconditioner.build(self.model, self.loss_fn, self._current_batch)

        params = [
            p for p in self.model.parameters()
            if p.requires_grad and p.grad is not None
        ]

        if params and self.preconditioner._built:
            flat_grad = torch.cat([p.grad.reshape(-1) for p in params])
            precond_grad = self.preconditioner.apply(flat_grad)

            if torch.isfinite(precond_grad).all():
                grad_norm = flat_grad.norm()
                precond_norm = precond_grad.norm()
                if grad_norm > 0 and precond_norm > 10.0 * grad_norm:
                    precond_grad = precond_grad * (10.0 * grad_norm / precond_norm)

                offset = 0
                for p in params:
                    n = p.numel()
                    p.grad.copy_(precond_grad[offset : offset + n].reshape(p.shape))
                    offset += n

        self.base_optimizer.step(closure)
        self.step_count += 1

    def state_dict(self):
        return self.base_optimizer.state_dict()


def compare_optimizers(model_fn, train_loader, val_loader, loss_fn,
                       n_epochs=20, device="cpu", configs=None):
    """
    Train identical models with different optimizers and return convergence curves.

    Parameters
    ----------
    model_fn : callable
        Zero-argument function that returns a fresh nn.Module.
    configs : list of (name, optimizer_factory) tuples, optional
        Each optimizer_factory takes a model and returns an optimizer.
        Defaults to SGD, Adam, SGD+Nyström, Adam+Nyström.

    Returns
    -------
    dict : {name: {train_loss, val_accuracy, final_loss, final_acc}}
    """
    ref_model = model_fn()
    init_state = deepcopy(ref_model.state_dict())

    if configs is None:
        configs = [
            ("SGD", lambda m: torch.optim.SGD(m.parameters(), lr=0.05)),
            ("Adam", lambda m: torch.optim.Adam(m.parameters(), lr=0.01)),
            (
                "SGD + Nyström",
                lambda m: NystromOptimizer(
                    m,
                    torch.optim.SGD(m.parameters(), lr=0.3),
                    loss_fn,
                    rank=15,
                    damping=1.0,
                    rebuild_every=3,
                ),
            ),
            (
                "Adam + Nyström",
                lambda m: NystromOptimizer(
                    m,
                    torch.optim.Adam(m.parameters(), lr=0.005),
                    loss_fn,
                    rank=15,
                    damping=1.0,
                    rebuild_every=5,
                ),
            ),
        ]

    results = {}
    for name, make_opt in configs:
        model = model_fn()
        model.load_state_dict(deepcopy(init_state))
        model = model.to(device)

        opt = make_opt(model)
        trainer = Trainer(model, train_loader, val_loader, opt, loss_fn, device)
        loss_hist, val_acc_hist = trainer.train(num_epochs=n_epochs)

        results[name] = {
            "train_loss": loss_hist,
            "val_accuracy": val_acc_hist,
            "final_loss": loss_hist[-1],
            "final_acc": val_acc_hist[-1] if val_acc_hist else None,
        }

    return results
