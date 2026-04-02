"""Conditional Flow Matching training and ODE inference."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CFMConfig:
    """Hyperparameters for Conditional Flow Matching."""

    lr: float = 1e-3
    weight_decay: float = 1e-3
    epochs: int = 300
    patience: int = 30
    label_smoothing: float = 0.025  # labels become 0.025 and 0.975
    n_inference_samples: int = 32
    n_ode_steps: int = 5
    sigma_min: float = 1e-4  # small noise floor to avoid exact interpolation at t=1


class ConditionalFlowMatcher:
    """Optimal-transport Conditional Flow Matching trainer.

    The flow interpolates between Gaussian noise (t=0) and the target (t=1)
    via straight-line paths: x_t = (1 - t) * x_0 + t * x_1.
    The model learns the velocity field v(x_t, t, c) = x_1 - x_0.
    """

    def __init__(self, model: nn.Module, config: CFMConfig | None = None) -> None:
        self.model = model
        self.config = config or CFMConfig()

    def compute_loss(
        self,
        x_1: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the CFM training loss for a batch.

        Args:
            x_1: target logit values, shape (B, D)
            features: conditioning features, shape (B, F)

        Returns:
            scalar loss (MSE on velocity prediction)
        """
        B, D = x_1.shape
        device = x_1.device

        # Sample noise and time
        x_0 = torch.randn_like(x_1)
        t = torch.rand(B, device=device)

        # Straight-line interpolation: x_t = (1-t)*x_0 + t*x_1
        t_expand = t.unsqueeze(-1)  # (B, 1)
        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1

        # Target velocity is the direction from noise to data
        target_v = x_1 - x_0

        # Predict velocity
        pred_v = self.model(x_t, t, features)

        # MSE loss on velocity
        return torch.mean((pred_v - target_v) ** 2)

    def train_epoch(
        self,
        x_1: torch.Tensor,
        features: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Run one training epoch (full-batch for small datasets).

        Returns:
            epoch loss (float)
        """
        self.model.train()
        optimizer.zero_grad()
        loss = self.compute_loss(x_1, features)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def sample(
        self,
        features: torch.Tensor,
        n_samples: int | None = None,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        """Generate probability samples via ODE integration.

        Args:
            features: conditioning features, shape (B, F) or (F,)
            n_samples: number of noise samples per input (default from config)
            n_steps: number of Euler steps (default from config)

        Returns:
            sampled logits, shape (B, n_samples, D) or (n_samples, D)
        """
        self.model.eval()
        n_samples = n_samples or self.config.n_inference_samples
        n_steps = n_steps or self.config.n_ode_steps

        squeeze = features.dim() == 1
        if squeeze:
            features = features.unsqueeze(0)

        B, F = features.shape
        D = 1  # target dimensionality (1D logit for MVE)

        # Repeat features for n_samples
        # (B, F) -> (B * n_samples, F)
        feat_expanded = features.unsqueeze(1).expand(B, n_samples, F).reshape(B * n_samples, F)

        # Start from noise
        z = torch.randn(B * n_samples, D, device=features.device)

        # Euler integration from t=0 to t=1
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_val = i * dt
            t = torch.full((B * n_samples,), t_val, device=features.device)
            v = self.model(z, t, feat_expanded)
            z = z + dt * v

        # Reshape: (B * n_samples, D) -> (B, n_samples, D)
        z = z.view(B, n_samples, D)

        if squeeze:
            z = z.squeeze(0)

        return z


class ODESolver:
    """Standalone ODE solver for more control over the integration process."""

    def __init__(self, model: nn.Module, n_steps: int = 5) -> None:
        self.model = model
        self.n_steps = n_steps

    @torch.no_grad()
    def solve(
        self,
        z_0: torch.Tensor,
        features: torch.Tensor,
        return_trajectory: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Integrate the ODE from t=0 to t=1.

        Args:
            z_0: initial noise, shape (B, D)
            features: conditioning, shape (B, F)
            return_trajectory: if True, return list of all intermediate states

        Returns:
            final state (B, D) or list of states [(B, D), ...]
        """
        self.model.eval()
        dt = 1.0 / self.n_steps
        z = z_0
        trajectory = [z.clone()] if return_trajectory else []

        for i in range(self.n_steps):
            t = torch.full((z.shape[0],), i * dt, device=z.device)
            v = self.model(z, t, features)
            z = z + dt * v
            if return_trajectory:
                trajectory.append(z.clone())

        return trajectory if return_trajectory else z


def logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Safe logit transform: p -> log(p / (1-p))."""
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid: logit -> probability."""
    return torch.sigmoid(x)
