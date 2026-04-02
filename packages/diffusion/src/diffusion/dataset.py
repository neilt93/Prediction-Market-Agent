"""Dataset construction from the prediction market database."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from calibration.calibrator import FEATURE_COLUMNS


@dataclass
class DatasetStats:
    """Normalization statistics for features."""

    mean: np.ndarray
    std: np.ndarray
    feature_names: list[str]

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, std=self.std, feature_names=self.feature_names)

    @classmethod
    def load(cls, path: str) -> DatasetStats:
        data = np.load(path, allow_pickle=True)
        return cls(
            mean=data["mean"],
            std=data["std"],
            feature_names=list(data["feature_names"]),
        )


class ForecastDataset(Dataset):
    """PyTorch dataset of (features, label) pairs from resolved markets.

    Loads data from a pandas DataFrame (extracted from DB by the trainer).
    Handles normalization and logit transformation.
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
        stats: DatasetStats | None = None,
        label_smoothing: float = 0.025,
        device: str = "cpu",
    ) -> None:
        available_cols = [c for c in FEATURE_COLUMNS if c in features_df.columns]
        X = features_df[available_cols].fillna(0).values.astype(np.float32)
        y = labels.values.astype(np.float32)

        # Compute or apply normalization
        if stats is None:
            self.stats = DatasetStats(
                mean=X.mean(axis=0),
                std=X.std(axis=0) + 1e-8,
                feature_names=available_cols,
            )
        else:
            self.stats = stats

        X = (X - self.stats.mean) / self.stats.std

        # Label smoothing: 0 -> label_smoothing, 1 -> 1 - label_smoothing
        y_smooth = y * (1.0 - 2 * label_smoothing) + label_smoothing

        # Transform to logit space
        y_logit = np.log(y_smooth / (1.0 - y_smooth))

        self.features = torch.tensor(X, device=device)
        self.targets = torch.tensor(y_logit, device=device).unsqueeze(-1)  # (N, 1)
        self.labels_raw = torch.tensor(y, device=device)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


def build_dataset_from_db(
    db_session: Any,
    market_ids: set | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Extract features, labels, and close_times from the database.

    Returns data sorted by close_time (temporal ordering).
    """
    from sqlalchemy import func
    from schemas.models.forecast import Forecast, ForecastFeature
    from schemas.models.market import Market, MarketOutcome

    query = (
        db_session.query(
            ForecastFeature,
            Forecast.raw_probability,
            MarketOutcome.resolved_label,
            Market.close_time,
        )
        .join(Forecast, ForecastFeature.forecast_id == Forecast.id)
        .join(MarketOutcome, Forecast.market_id == MarketOutcome.market_id)
        .join(Market, Forecast.market_id == Market.id)
        .filter(MarketOutcome.resolved_label.isnot(None))
        .order_by(Market.close_time.asc())
    )
    if market_ids:
        query = query.filter(Forecast.market_id.in_(list(market_ids)))

    results = query.all()
    if not results:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype="datetime64[ns]")

    rows = []
    labels = []
    close_times = []
    for feature, raw_prob, label, close_time in results:
        row = {
            "market_price": float(feature.market_price or 0),
            "spread_bps": feature.spread_bps or 0,
            "vol_24h": float(feature.vol_24h or 0),
            "time_to_close_sec": feature.time_to_close_sec or 0,
            "ambiguity_score": float(feature.ambiguity_score or 0),
            "freshness_score": float(feature.freshness_score or 0),
            "source_agreement_score": float(feature.source_agreement_score or 0),
            "official_source_present": 1.0 if feature.official_source_present else 0.0,
            "llm_confidence": float(feature.llm_confidence or 0),
            "retrieval_count": feature.retrieval_count or 0,
            "price_momentum_1h": float(feature.price_momentum_1h or 0),
            "price_momentum_24h": float(feature.price_momentum_24h or 0),
            "raw_probability": float(raw_prob or 0.5),
        }
        rows.append(row)
        labels.append(label)
        close_times.append(close_time)

    return (
        pd.DataFrame(rows),
        pd.Series(labels, dtype=float),
        pd.Series(close_times, dtype="datetime64[ns]"),
    )


def temporal_cv_splits(
    n_samples: int,
    n_folds: int = 5,
    min_train_size: int = 30,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window temporal cross-validation splits.

    Data must already be sorted by time. Each fold uses all prior data
    for training and the next chunk for validation.

    Returns:
        List of (train_indices, val_indices) tuples
    """
    fold_size = (n_samples - min_train_size) // n_folds
    if fold_size < 5:
        # Too few samples for CV -- use a single 80/20 split
        split = int(n_samples * 0.8)
        return [(np.arange(split), np.arange(split, n_samples))]

    splits = []
    for i in range(n_folds):
        val_start = min_train_size + i * fold_size
        val_end = val_start + fold_size if i < n_folds - 1 else n_samples
        if val_start >= n_samples:
            break
        train_idx = np.arange(val_start)
        val_idx = np.arange(val_start, val_end)
        splits.append((train_idx, val_idx))

    return splits


def augment_with_counterfactuals(
    features_df: pd.DataFrame,
    labels: pd.Series,
    noise_scale: float = 0.1,
    n_augmented: int = 3,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create counterfactual augmented training data.

    For each sample, creates n_augmented copies with:
    - Small Gaussian noise on numerical features
    - Flipped labels with correspondingly perturbed raw_probability
      (simulates a world where the outcome was different)

    This forces the model to learn from feature patterns rather than
    memorizing market-outcome correlations.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    augmented_rows = []
    augmented_labels = []

    feature_stds = features_df.std().fillna(0).values

    for i in range(len(features_df)):
        row = features_df.iloc[i].values.astype(np.float32)
        label = labels.iloc[i]

        for _ in range(n_augmented):
            # Add noise to features
            noise = rng.normal(0, noise_scale, size=row.shape) * feature_stds
            new_row = row + noise.astype(np.float32)

            # Half the augmentations flip the label (counterfactual)
            if rng.random() < 0.5:
                new_label = 1.0 - label
                # Perturb raw_probability toward the flipped label
                raw_prob_idx = list(features_df.columns).index("raw_probability") if "raw_probability" in features_df.columns else -1
                if raw_prob_idx >= 0:
                    new_row[raw_prob_idx] = np.clip(
                        1.0 - new_row[raw_prob_idx] + rng.normal(0, 0.1), 0.01, 0.99
                    )
            else:
                new_label = label

            augmented_rows.append(new_row)
            augmented_labels.append(new_label)

    aug_df = pd.DataFrame(augmented_rows, columns=features_df.columns)
    aug_labels = pd.Series(augmented_labels, dtype=float)

    # Concatenate original + augmented
    combined_df = pd.concat([features_df, aug_df], ignore_index=True)
    combined_labels = pd.concat([labels, aug_labels], ignore_index=True)

    return combined_df, combined_labels
