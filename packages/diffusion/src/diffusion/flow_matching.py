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
    sigma_min: float = 1e-4
    use_market_prior: bool = True  # start ODE from market price, not pure noise
    market_price_feature_idx: int = 0  # index of market_price in feature vector


def _market_prior_sigma(market_price: torch.Tensor) -> torch.Tensor:
    """Compute noise scale for market-price-informed prior.

    When price is extreme (near 0 or 1), sigma is small -- we trust the market.
    When price is near 0.5, sigma is large -- uncertain, let the model work.

    sigma = 4 * p * (1 - p), peaks at 1.0 when p=0.5, ~0 at extremes.
    """
    p = market_price.clamp(0.01, 0.99)
    return 4.0 * p * (1.0 - p)


class ConditionalFlowMatcher:
    """Optimal-transport Conditional Flow Matching trainer.

    With market-price prior: the flow starts from logit(market_price) + noise
    instead of pure N(0,1). This means the model learns to *refine* the market
    price rather than discover the answer from scratch.
    """

    def __init__(self, model: nn.Module, config: CFMConfig | None = None) -> None:
        self.model = model
        self.config = config or CFMConfig()

    def _sample_prior(
        self, shape: tuple, features: torch.Tensor, device: torch.device,
    ) -> torch.Tensor:
        """Sample from the source distribution (t=0).

        If use_market_prior: N(logit(market_price), sigma(market_price))
        Otherwise: N(0, 1)
        """
        if self.config.use_market_prior and features.shape[-1] > self.config.market_price_feature_idx:
            # Extract market_price from features (it's a passthrough feature, not normalized)
            mp = features[:, self.config.market_price_feature_idx].clamp(0.01, 0.99)
            mu = logit(mp).unsqueeze(-1)  # (B, 1)
            sigma = _market_prior_sigma(mp).unsqueeze(-1)  # (B, 1)
            # Ensure sigma has a floor so we still explore
            sigma = sigma.clamp(min=0.1)
            return mu + sigma * torch.randn(shape, device=device)
        return torch.randn(shape, device=device)

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

        # Sample from source distribution and time
        x_0 = self._sample_prior((B, D), features, device)
        t = torch.rand(B, device=device)

        # Straight-line interpolation: x_t = (1-t)*x_0 + t*x_1
        t_expand = t.unsqueeze(-1)
        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1

        # Target velocity: direction from source to target
        target_v = x_1 - x_0

        # Predict velocity
        pred_v = self.model(x_t, t, features)

        return torch.mean((pred_v - target_v) ** 2)

    def train_epoch(
        self,
        x_1: torch.Tensor,
        features: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Run one training epoch (full-batch for small datasets)."""
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
            n_samples: number of noise samples per input
            n_steps: number of Euler steps

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
        D = 1

        # Expand features for n_samples
        feat_expanded = features.unsqueeze(1).expand(B, n_samples, F).reshape(B * n_samples, F)

        # Start from market-price-informed prior (or pure noise)
        z = self._sample_prior((B * n_samples, D), feat_expanded, features.device)

        # Euler integration from t=0 to t=1
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_val = i * dt
            t = torch.full((B * n_samples,), t_val, device=features.device)
            v = self.model(z, t, feat_expanded)
            z = z + dt * v

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
