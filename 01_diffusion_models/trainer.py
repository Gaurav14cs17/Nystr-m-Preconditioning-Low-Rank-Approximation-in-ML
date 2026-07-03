import time
import torch
import numpy as np


class DiffusionTrainer:
    def __init__(self, model, diffusion, dataloader, lr=1e-3, device='cpu'):
        self.model = model.to(device)
        self.diffusion = diffusion.to(device)
        self.dataloader = dataloader
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_history = []

    def train(self, num_epochs=5):
        self.model.train()
        for epoch in range(num_epochs):
            epoch_losses = []
            t0 = time.time()
            for batch in self.dataloader:
                x_0 = batch.to(self.device)
                loss = self.diffusion.training_loss(x_0)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
            avg_loss = np.mean(epoch_losses)
            self.loss_history.append(avg_loss)
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1}/{num_epochs}  loss={avg_loss:.4f}  ({elapsed:.1f}s)")
        return self.loss_history

    @torch.no_grad()
    def sample(self, n_samples=4):
        self.model.eval()
        shape = (n_samples, 1, 28, 28)
        samples = self.diffusion.sample(shape, device=self.device)
        return samples.cpu()

    @torch.no_grad()
    def evaluate(self):
        """Compute reconstruction MSE: noise a real image, denoise one step, measure error."""
        self.model.eval()
        total_mse = 0.0
        count = 0
        for batch in self.dataloader:
            x_0 = batch.to(self.device)
            B = x_0.shape[0]
            # Use a mid-range timestep for evaluation
            t = torch.full((B,), self.diffusion.timesteps // 4, device=self.device, dtype=torch.long)
            noise = torch.randn_like(x_0)
            x_t = self.diffusion.q_sample(x_0, t, noise)
            pred_noise = self.model(x_t, t)
            mse = torch.mean((pred_noise - noise) ** 2).item()
            total_mse += mse * B
            count += B
        return total_mse / count
