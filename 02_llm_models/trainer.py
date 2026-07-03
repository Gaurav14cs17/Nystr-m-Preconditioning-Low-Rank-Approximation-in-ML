"""
Training, evaluation, and generation for the CausalLM.
"""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .models import CausalLM, KVCache
except ImportError:
    from models import CausalLM, KVCache


class LLMTrainer:
    def __init__(self, model, dataloader, lr=3e-4, device='cpu'):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_losses = []

    def train(self, num_epochs=5):
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for input_ids, target_ids in self.dataloader:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                logits = self.model(input_ids)
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.train_losses.append(avg_loss)
            ppl = math.exp(min(avg_loss, 20))
            print(f"  Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Perplexity: {ppl:.2f}")
        return self.train_losses

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for input_ids, target_ids in self.dataloader:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                logits = self.model(input_ids)
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                total_loss += loss.item()
                n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        perplexity = math.exp(min(avg_loss, 20))
        return avg_loss, perplexity

    @torch.no_grad()
    def generate(self, prompt_ids, max_len=20, temperature=0.8):
        self.model.eval()
        if isinstance(prompt_ids, list):
            prompt_ids = torch.tensor([prompt_ids], dtype=torch.long)
        prompt_ids = prompt_ids.to(self.device)

        generated = prompt_ids.clone()
        for _ in range(max_len):
            if generated.shape[1] >= self.model.max_seq:
                generated = generated[:, -self.model.max_seq:]
            logits = self.model(generated)
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated[0].tolist()

    @torch.no_grad()
    def generate_with_kv_cache(self, prompt_ids, max_len=20, temperature=0.8, compress_rank=None):
        self.model.eval()
        if isinstance(prompt_ids, list):
            prompt_ids = torch.tensor([prompt_ids], dtype=torch.long)
        prompt_ids = prompt_ids.to(self.device)

        generated = prompt_ids[0].tolist()
        context = prompt_ids

        for step in range(max_len):
            logits = self.model(context)
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token.item())
            context = torch.cat([context, next_token], dim=1)
            if context.shape[1] > self.model.max_seq:
                context = context[:, -self.model.max_seq:]

        return generated
