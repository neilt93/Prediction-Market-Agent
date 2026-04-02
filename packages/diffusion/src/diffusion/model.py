"""Neural network components for the Conditional Flow Matching denoiser."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding, mapping scalar t in [0,1] to a vector."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.dim = dim
        # Precompute frequency bands (not trainable)
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed scalar time steps.

        Args:
            t: shape (B,) or (B, 1), values in [0, 1]

        Returns:
            shape (B, dim)
        """
        t = t.view(-1, 1).float()  # (B, 1)
        args = t * self.freqs.unsqueeze(0)  # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class DenoisingMLP(nn.Module):
    """Small MLP denoiser for 1D Conditional Flow Matching.

    Takes noisy logit z_t, time embedding, and conditioning features.
    Predicts the velocity field v(z_t, t, c) for the ODE.
    """

    def __init__(
        self,
        target_dim: int = 1,
        feature_dim: int = 13,
        time_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)

        input_dim = target_dim + time_dim + feature_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field.

        Args:
            z_t: noisy logit state, shape (B, target_dim)
            t: time step, shape (B,)
            features: conditioning features, shape (B, feature_dim)

        Returns:
            predicted velocity, shape (B, target_dim)
        """
        t_emb = self.time_embed(t)  # (B, time_dim)
        x = torch.cat([z_t, t_emb, features], dim=-1)  # (B, input_dim)
        return self.net(x)


class FiLMDenoisingMLP(nn.Module):
    """Denoiser with FiLM conditioning for Phase 2 (evidence-conditioned).

    Uses Feature-wise Linear Modulation: the condition vector produces
    per-layer scale and shift parameters, allowing richer conditioning
    than simple concatenation.
    """

    def __init__(
        self,
        target_dim: int = 2,
        condition_dim: int = 256,
        time_dim: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        self.input_proj = nn.Linear(target_dim + time_dim, hidden_dim)

        # FiLM generators: condition -> (scale, shift) per layer
        self.film_generators = nn.ModuleList([
            nn.Linear(condition_dim, hidden_dim * 2) for _ in range(n_layers)
        ])

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            ))

        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, target_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize FiLM to identity transform (scale=1, shift=0)
        for film in self.film_generators:
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)
            # Set scale bias to 1 (first half of output)
            film.bias.data[: film.bias.shape[0] // 2] = 1.0

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field with FiLM conditioning.

        Args:
            z_t: noisy state, shape (B, target_dim)
            t: time step, shape (B,)
            condition: condition vector, shape (B, condition_dim)

        Returns:
            predicted velocity, shape (B, target_dim)
        """
        t_emb = self.time_embed(t)
        h = self.input_proj(torch.cat([z_t, t_emb], dim=-1))

        for layer, film_gen in zip(self.layers, self.film_generators):
            h = layer(h)
            # FiLM: modulate with condition-dependent scale and shift
            film_params = film_gen(condition)
            scale, shift = film_params.chunk(2, dim=-1)
            h = scale * h + shift
            h = torch.nn.functional.silu(h)

        h = self.dropout(h)
        return self.output_proj(h)


class ConditionEncoder(nn.Module):
    """Encodes heterogeneous conditioning signals into a single vector.

    For Phase 2: combines evidence embeddings (384D), title embedding (384D),
    and numerical features (13D) into a unified condition vector.
    """

    def __init__(
        self,
        evidence_dim: int = 384,
        context_dim: int = 384,
        feature_dim: int = 13,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        input_dim = evidence_dim + context_dim + feature_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        evidence_embedding: torch.Tensor,
        context_embedding: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode all conditioning signals.

        Args:
            evidence_embedding: aggregated evidence, shape (B, 384)
            context_embedding: market title embedding, shape (B, 384)
            features: numerical features, shape (B, 13)

        Returns:
            condition vector, shape (B, output_dim)
        """
        x = torch.cat([evidence_embedding, context_embedding, features], dim=-1)
        return self.net(x)
