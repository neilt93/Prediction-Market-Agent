"""Evaluation utilities: metrics, reliability diagrams, bootstrap significance."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class EvalMetrics:
    """Evaluation metrics for a set of predictions."""

    brier_score: float
    log_loss: float
    accuracy: float  # directional accuracy (pred > 0.5 matches label)
    n_samples: int
    mean_uncertainty: float  # avg width of 80% prediction interval
    interval_coverage: float  # fraction of labels within [p10, p90]

    def __str__(self) -> str:
        return (
            f"Brier: {self.brier_score:.4f} | LogLoss: {self.log_loss:.4f} | "
            f"Accuracy: {self.accuracy:.1%} | N: {self.n_samples} | "
            f"Uncertainty: {self.mean_uncertainty:.3f} | "
            f"80% Coverage: {self.interval_coverage:.1%}"
        )


def compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    prob_samples: np.ndarray | None = None,
) -> EvalMetrics:
    """Compute evaluation metrics.

    Args:
        probs: point probability estimates, shape (N,)
        labels: ground truth 0/1 labels, shape (N,)
        prob_samples: if available, shape (N, K) samples for uncertainty

    Returns:
        EvalMetrics dataclass
    """
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    n = len(labels)

    brier = float(np.mean((probs - labels) ** 2))
    log_loss = float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))
    accuracy = float(np.mean((probs > 0.5) == (labels > 0.5)))

    if prob_samples is not None and prob_samples.ndim == 2:
        p10 = np.percentile(prob_samples, 10, axis=1)
        p90 = np.percentile(prob_samples, 90, axis=1)
        mean_uncertainty = float(np.mean(p90 - p10))
        # For binary labels (0/1): check if the interval includes the correct side
        # label=1 is covered if p90 > 0.5, label=0 is covered if p10 < 0.5
        # More precisely: the interval should contain the "correct" probability direction
        covered = ((labels == 1) & (p90 > 0.5)) | ((labels == 0) & (p10 < 0.5))
        interval_coverage = float(np.mean(covered))
    else:
        mean_uncertainty = 0.0
        interval_coverage = 0.0

    return EvalMetrics(
        brier_score=brier,
        log_loss=log_loss,
        accuracy=accuracy,
        n_samples=n,
        mean_uncertainty=mean_uncertainty,
        interval_coverage=interval_coverage,
    )


def reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    save_path: str | None = None,
    title: str = "Reliability Diagram",
) -> dict[str, np.ndarray]:
    """Compute and optionally plot a reliability (calibration) diagram.

    Args:
        probs: predicted probabilities, shape (N,)
        labels: ground truth 0/1, shape (N,)
        n_bins: number of probability bins
        save_path: if provided, save plot to this path
        title: plot title

    Returns:
        dict with bin_centers, bin_means, bin_counts
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_means = []
    bin_counts = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        count = mask.sum()
        if count > 0:
            bin_centers.append((lo + hi) / 2)
            bin_means.append(labels[mask].mean())
            bin_counts.append(int(count))

    result = {
        "bin_centers": np.array(bin_centers),
        "bin_means": np.array(bin_means),
        "bin_counts": np.array(bin_counts),
    }

    if save_path:
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})

            # Calibration plot
            ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
            ax1.bar(
                result["bin_centers"],
                result["bin_means"],
                width=1.0 / n_bins * 0.8,
                alpha=0.7,
                label="Model",
            )
            ax1.set_xlabel("Predicted probability")
            ax1.set_ylabel("Observed frequency")
            ax1.set_title(title)
            ax1.legend()
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)

            # Histogram of predictions
            ax2.hist(probs, bins=n_bins, range=(0, 1), alpha=0.7)
            ax2.set_xlabel("Predicted probability")
            ax2.set_ylabel("Count")

            plt.tight_layout()
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        except ImportError:
            pass

    return result


def bootstrap_brier_test(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Paired bootstrap test for Brier score difference.

    Tests H0: Brier(A) = Brier(B) vs H1: Brier(A) != Brier(B).

    Args:
        probs_a: predictions from model A, shape (N,)
        probs_b: predictions from model B, shape (N,)
        labels: ground truth, shape (N,)
        n_bootstrap: number of bootstrap resamples

    Returns:
        dict with observed_diff, p_value, ci_low, ci_high
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(labels)
    brier_a = (probs_a - labels) ** 2
    brier_b = (probs_b - labels) ** 2
    observed_diff = brier_a.mean() - brier_b.mean()

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diff = brier_a[idx].mean() - brier_b[idx].mean()
        diffs.append(diff)

    diffs = np.array(diffs)
    # Two-sided p-value
    p_value = float(np.mean(np.abs(diffs) >= np.abs(observed_diff)))

    return {
        "observed_diff": float(observed_diff),
        "p_value": p_value,
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "a_better": bool(observed_diff < 0),
    }
