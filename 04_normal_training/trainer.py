import torch
import numpy as np


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, device="cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train(self, num_epochs=20):
        loss_history = []
        val_accuracy_history = []

        for _epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)

                if hasattr(self.optimizer, "set_batch"):
                    self.optimizer.set_batch((X, y))

                self.optimizer.zero_grad()
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            loss_history.append(epoch_loss / max(n_batches, 1))
            _, val_acc = self.evaluate()
            val_accuracy_history.append(val_acc)

        return loss_history, val_accuracy_history

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        is_classification = False

        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                total_loss += loss.item() * X.size(0)
                total += X.size(0)

                if y.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                    is_classification = True
                    correct += (pred.argmax(dim=-1) == y).sum().item()

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1) if is_classification else None
        return avg_loss, accuracy

    def compute_hessian_spectrum(self, n_samples=None):
        """Compute full Hessian eigenvalues (feasible only for small models)."""
        self.model.eval()
        params = [p for p in self.model.parameters() if p.requires_grad]
        d = sum(p.numel() for p in params)

        X, y = next(iter(self.train_loader))
        X, y = X.to(self.device), y.to(self.device)

        loss = self.loss_fn(self.model(X), y)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grad = torch.cat([g.reshape(-1) for g in grads])

        H = torch.zeros(d, d)
        for i in range(d):
            row = torch.autograd.grad(flat_grad[i], params, retain_graph=True)
            H[i] = torch.cat([r.reshape(-1) for r in row])

        H = 0.5 * (H + H.T)
        eigenvalues = torch.linalg.eigvalsh(H).detach().numpy()
        return np.sort(np.abs(eigenvalues))[::-1].copy()

    def compute_condition_number(self):
        eigs = self.compute_hessian_spectrum()
        pos = eigs[eigs > 1e-10]
        if len(pos) < 2:
            return float("inf")
        return float(pos[0] / pos[-1])
