"""DiffusionCalibrator: drop-in replacement for the LightGBM Calibrator."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diffusion.dataset import DatasetStats, FEATURE_COLUMNS
from diffusion.flow_matching import ConditionalFlowMatcher, CFMConfig
from diffusion.model import DenoisingMLP


class CalibratedOutput:
    """Matches the interface from calibration.calibrator.CalibratedOutput."""

    def __init__(
        self,
        calibrated_probability: float,
        predicted_edge_bps: int,
        uncertainty_low: float,
        uncertainty_high: float,
    ) -> None:
        self.calibrated_probability = calibrated_probability
        self.predicted_edge_bps = predicted_edge_bps
        self.uncertainty_low = uncertainty_low
        self.uncertainty_high = uncertainty_high


class DiffusionCalibrator:
    """Flow-matching calibrator with the same interface as the LightGBM Calibrator.

    Produces calibrated probabilities by sampling from a learned
    conditional distribution, giving both point estimates and
    uncertainty bounds.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: DenoisingMLP | None = None
        self.cfm: ConditionalFlowMatcher | None = None
        self.stats: DatasetStats | None = None
        self.version = "v0-untrained"

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def predict(
        self,
        features: dict[str, float],
        market_price: float | None = None,
    ) -> CalibratedOutput:
        """Predict calibrated probability using flow matching.

        Args:
            features: dict of feature name -> value (same as FEATURE_COLUMNS)
            market_price: optional market price for edge computation

        Returns:
            CalibratedOutput with calibrated_probability, edge, uncertainty
        """
        if self.model is None or self.stats is None:
            # Fallback: return raw probability with wide uncertainty
            raw = features.get("raw_probability", 0.5)
            return CalibratedOutput(
                calibrated_probability=raw,
                predicted_edge_bps=0,
                uncertainty_low=max(0, raw - 0.15),
                uncertainty_high=min(1, raw + 0.15),
            )

        # Build feature vector in the same order as training
        x = np.array(
            [features.get(c, 0.0) for c in self.stats.feature_names],
            dtype=np.float32,
        )
        # Normalize using training stats
        x = (x - self.stats.mean) / self.stats.std
        feat_tensor = torch.tensor(x, device=self.device).unsqueeze(0)  # (1, F)

        # Sample from the flow model
        logit_samples = self.cfm.sample(feat_tensor, n_samples=32)  # (1, 32, 1)
        prob_samples = torch.sigmoid(logit_samples).squeeze(-1).squeeze(0)  # (32,)
        prob_samples = prob_samples.cpu().numpy()

        # Aggregate
        prob = float(np.mean(prob_samples))
        prob = max(0.01, min(0.99, prob))
        p10 = float(np.percentile(prob_samples, 10))
        p90 = float(np.percentile(prob_samples, 90))

        edge_bps = 0
        if market_price is not None and market_price > 0:
            edge_bps = int((prob - market_price) * 10000)

        return CalibratedOutput(
            calibrated_probability=prob,
            predicted_edge_bps=edge_bps,
            uncertainty_low=max(0.01, p10),
            uncertainty_high=min(0.99, p90),
        )

    def save(self, path: str) -> None:
        """Save model weights, config, and normalization stats."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            torch.save(self.model.state_dict(), str(p))

        if self.stats is not None:
            self.stats.save(str(p.with_suffix(".stats.npz")))

        meta = {"version": self.version, "device": self.device}
        with open(str(p.with_suffix(".meta.json")), "w") as f:
            json.dump(meta, f)

    def load(self, path: str) -> None:
        """Load a saved model."""
        p = Path(path)

        # Load stats first to know feature count
        stats_path = p.with_suffix(".stats.npz")
        if stats_path.exists():
            self.stats = DatasetStats.load(str(stats_path))
        else:
            raise FileNotFoundError(f"Stats file not found: {stats_path}")

        feature_dim = len(self.stats.feature_names)
        self.model = DenoisingMLP(
            target_dim=1,
            feature_dim=feature_dim,
        ).to(self.device)
        self.model.load_state_dict(torch.load(str(p), map_location=self.device, weights_only=True))
        self.model.eval()

        self.cfm = ConditionalFlowMatcher(self.model, CFMConfig())

        meta_path = p.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                self.version = meta.get("version", "unknown")
