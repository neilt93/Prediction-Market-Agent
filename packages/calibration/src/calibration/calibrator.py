"""Calibration service using LightGBM to correct raw LLM probabilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()

FEATURE_COLUMNS = [
    "market_price",
    "spread_bps",
    "vol_24h",
    "time_to_close_sec",
    "ambiguity_score",
    "freshness_score",
    "source_agreement_score",
    "official_source_present",
    "llm_confidence",
    "retrieval_count",
    "price_momentum_1h",
    "price_momentum_24h",
    "raw_probability",
]


class CalibratedOutput:
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


class Calibrator:
    """LightGBM-based calibrator for forecast probabilities."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model: lgb.Booster | None = None
        self.version = "v0-untrained"
        if model_path and Path(model_path).exists():
            self.load(model_path)

    def train(self, features_df: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
        """Train the calibrator on historical data.

        Args:
            features_df: DataFrame with columns from FEATURE_COLUMNS + raw_probability
            labels: Series of 0/1 resolved outcomes
        """
        available_cols = [c for c in FEATURE_COLUMNS if c in features_df.columns]
        X = features_df[available_cols].fillna(0).values
        y = labels.values

        train_data = lgb.Dataset(X, label=y, feature_name=available_cols)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[train_data],
            callbacks=[lgb.log_evaluation(period=50)],
        )

        # Compute training metrics
        preds = self.model.predict(X)
        from sklearn.metrics import brier_score_loss, log_loss

        metrics = {
            "brier_score": float(brier_score_loss(y, preds)),
            "log_loss": float(log_loss(y, preds)),
            "n_samples": len(y),
        }
        self.version = f"v1-lgb-{len(y)}samples"
        logger.info("calibrator_trained", **metrics)
        return metrics

    def predict(
        self,
        features: dict[str, float],
        market_price: float | None = None,
    ) -> CalibratedOutput:
        """Predict calibrated probability from features."""
        if self.model is None:
            # Fallback: return raw probability with wide uncertainty
            raw = features.get("raw_probability", 0.5)
            return CalibratedOutput(
                calibrated_probability=raw,
                predicted_edge_bps=0,
                uncertainty_low=max(0, raw - 0.15),
                uncertainty_high=min(1, raw + 0.15),
            )

        available_cols = [c for c in FEATURE_COLUMNS if c in features]
        X = np.array([[features.get(c, 0) for c in available_cols]])
        prob = float(self.model.predict(X)[0])
        prob = max(0.01, min(0.99, prob))

        # Compute edge vs market price
        edge_bps = 0
        if market_price is not None and market_price > 0:
            edge_bps = int((prob - market_price) * 10000)

        # Simple uncertainty from model variance (using leaf predictions)
        uncertainty = 0.1  # default
        return CalibratedOutput(
            calibrated_probability=prob,
            predicted_edge_bps=edge_bps,
            uncertainty_low=max(0.01, prob - uncertainty),
            uncertainty_high=min(0.99, prob + uncertainty),
        )

    def save(self, path: str) -> None:
        if self.model:
            self.model.save_model(path)
            meta_path = path + ".meta.json"
            with open(meta_path, "w") as f:
                json.dump({"version": self.version}, f)
            logger.info("calibrator_saved", path=path)

    def load(self, path: str) -> None:
        self.model = lgb.Booster(model_file=path)
        meta_path = path + ".meta.json"
        if Path(meta_path).exists():
            with open(meta_path) as f:
                meta = json.load(f)
                self.version = meta.get("version", "unknown")
        logger.info("calibrator_loaded", path=path, version=self.version)
