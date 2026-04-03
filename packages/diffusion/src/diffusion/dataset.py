"""Dataset construction from the prediction market database."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Duplicated from calibration.calibrator to avoid hard dependency on lightgbm
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

# Features already on [0,1] scale that should NOT be z-score normalized.
# These are the dominant predictive signals and normalizing them washes out
# the direct correspondence between input and output probability space.
PASSTHROUGH_FEATURES = {"market_price", "raw_probability", "llm_confidence",
                        "ambiguity_score", "freshness_score", "source_agreement_score",
                        "official_source_present"}


@dataclass
class DatasetStats:
    """Normalization statistics for features."""

    mean: np.ndarray
    std: np.ndarray
    feature_names: list[str]
    passthrough_mask: np.ndarray = field(default_factory=lambda: np.array([]))

    def save(self, path: str) -> None:
        np.savez(
            path, mean=self.mean, std=self.std,
            feature_names=self.feature_names,
            passthrough_mask=self.passthrough_mask,
        )

    @classmethod
    def load(cls, path: str) -> DatasetStats:
        data = np.load(path, allow_pickle=True)
        return cls(
            mean=data["mean"],
            std=data["std"],
            feature_names=list(data["feature_names"]),
            passthrough_mask=data.get("passthrough_mask", np.array([])),
        )


class ForecastDataset(Dataset):
    """PyTorch dataset of (features, label) pairs from resolved markets.

    Features on [0,1] scale (market_price, raw_probability, etc.) are passed
    through without normalization. Other features are z-score normalized.
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

        # Build passthrough mask: True for features that skip normalization
        passthrough = np.array([c in PASSTHROUGH_FEATURES for c in available_cols])

        if stats is None:
            self.stats = DatasetStats(
                mean=X.mean(axis=0),
                std=X.std(axis=0) + 1e-8,
                feature_names=available_cols,
                passthrough_mask=passthrough,
            )
        else:
            self.stats = stats
            passthrough = stats.passthrough_mask

        # Selective normalization: only z-score non-passthrough features
        X_norm = X.copy()
        normalize_mask = ~passthrough
        if normalize_mask.any():
            X_norm[:, normalize_mask] = (
                (X[:, normalize_mask] - self.stats.mean[normalize_mask])
                / self.stats.std[normalize_mask]
            )

        # Label smoothing: 0 -> eps, 1 -> 1-eps
        y_smooth = y * (1.0 - 2 * label_smoothing) + label_smoothing
        y_logit = np.log(y_smooth / (1.0 - y_smooth))

        self.features = torch.tensor(X_norm, device=device)
        self.targets = torch.tensor(y_logit, device=device).unsqueeze(-1)
        self.labels_raw = torch.tensor(y, device=device)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


def build_dataset_from_db(
    db_session: Any,
    market_ids: set | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Extract features, labels, close_times, and titles from the database.

    Returns data sorted by close_time (temporal ordering).
    """
    from schemas.models.forecast import Forecast, ForecastFeature
    from schemas.models.market import Market, MarketOutcome

    query = (
        db_session.query(
            ForecastFeature,
            Forecast.raw_probability,
            MarketOutcome.resolved_label,
            Market.close_time,
            Market.title,
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
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype="datetime64[ns]"), []

    rows = []
    labels = []
    close_times = []
    titles = []
    for feature, raw_prob, label, close_time, title in results:
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
        titles.append(title or "")

    return (
        pd.DataFrame(rows),
        pd.Series(labels, dtype=float),
        pd.Series(close_times).dt.tz_localize(None),
        titles,
    )


def temporal_cv_splits(
    n_samples: int,
    n_folds: int = 5,
    min_train_size: int = 30,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window temporal cross-validation splits.

    Data must already be sorted by time. Each fold uses all prior data
    for training and the next chunk for validation.
    """
    fold_size = (n_samples - min_train_size) // n_folds
    if fold_size < 5:
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
    noise_scale: float = 0.05,
    n_augmented: int = 3,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create augmented training data with smart counterfactuals.

    Improvements over v1:
    - Only flips labels on uncertain markets (market_price between 0.2 and 0.8)
    - When flipping, also flips market_price (1 - price) for consistency
    - Reduces noise scale to avoid destroying signal
    - Non-flipped augmentations just add small noise (keeps label)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    cols = list(features_df.columns)
    mp_idx = cols.index("market_price") if "market_price" in cols else -1
    rp_idx = cols.index("raw_probability") if "raw_probability" in cols else -1

    augmented_rows = []
    augmented_labels = []

    # Compute stds only for non-passthrough features
    feature_stds = features_df.std().fillna(0).values

    for i in range(len(features_df)):
        row = features_df.iloc[i].values.astype(np.float32)
        label = labels.iloc[i]
        market_price = row[mp_idx] if mp_idx >= 0 else 0.5

        for _ in range(n_augmented):
            new_row = row.copy()

            # Add small noise to non-probability features
            noise = rng.normal(0, noise_scale, size=row.shape) * feature_stds
            new_row += noise.astype(np.float32)

            # Only flip labels on uncertain markets (price between 0.2-0.8)
            is_uncertain = 0.2 < market_price < 0.8
            if is_uncertain and rng.random() < 0.4:
                new_label = 1.0 - label
                # Flip the probability features to match
                if mp_idx >= 0:
                    new_row[mp_idx] = np.clip(1.0 - market_price + rng.normal(0, 0.05), 0.01, 0.99)
                if rp_idx >= 0:
                    new_row[rp_idx] = np.clip(1.0 - row[rp_idx] + rng.normal(0, 0.05), 0.01, 0.99)
            else:
                new_label = label
                # Keep probability features close to original
                if mp_idx >= 0:
                    new_row[mp_idx] = np.clip(row[mp_idx] + rng.normal(0, 0.02), 0.01, 0.99)
                if rp_idx >= 0:
                    new_row[rp_idx] = np.clip(row[rp_idx] + rng.normal(0, 0.02), 0.01, 0.99)

            augmented_rows.append(new_row)
            augmented_labels.append(new_label)

    aug_df = pd.DataFrame(augmented_rows, columns=features_df.columns)
    aug_labels = pd.Series(augmented_labels, dtype=float)

    combined_df = pd.concat([features_df, aug_df], ignore_index=True)
    combined_labels = pd.concat([labels, aug_labels], ignore_index=True)

    return combined_df, combined_labels
