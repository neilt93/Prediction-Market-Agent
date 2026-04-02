"""Expected Calibration Error (ECE) computation.

ECE measures how well-calibrated the forecaster is: when it says 70%,
does the event happen ~70% of the time?

Used as a dynamic trading threshold: only trade when Edge > ECE.
This retains nearly all upside while avoiding the loss-making tail.
"""
from __future__ import annotations

import numpy as np


def compute_ece(
    probabilities: list[float],
    outcomes: list[int],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    Args:
        probabilities: Model's predicted probabilities (0-1)
        outcomes: Actual binary outcomes (0 or 1)
        n_bins: Number of bins for calibration curve

    Returns:
        ECE as a float (0 = perfectly calibrated)
    """
    if not probabilities or not outcomes:
        return 0.5  # Default high ECE when no data

    probs = np.array(probabilities)
    labels = np.array(outcomes)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(probs)

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])

        bin_count = mask.sum()
        if bin_count == 0:
            continue

        avg_confidence = probs[mask].mean()
        avg_accuracy = labels[mask].mean()
        ece += (bin_count / total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def compute_ece_from_db(session) -> dict[str, float]:
    """Compute ECE from postmortem data, overall and per-niche.

    Returns dict like:
        {"overall": 0.08, "geopolitics": 0.05, "crypto": 0.12, "tech": 0.03}
    """
    from schemas.models.forecast import Forecast
    from schemas.models.postmortem import Postmortem
    from schemas.models.market import Market

    rows = (
        session.query(
            Forecast.raw_probability,
            Postmortem.resolved_label,
            Market.title,
            Market.category,
        )
        .join(Postmortem, Postmortem.forecast_id == Forecast.id)
        .join(Market, Market.id == Postmortem.market_id)
        .filter(Postmortem.resolved_label.isnot(None))
        .all()
    )

    if not rows:
        return {"overall": 0.5}

    # Niche classification (same keywords as run_live.py)
    geo_kw = ["trump", "iran", "ukraine", "russia", "china", "fed ", "interest rate",
              "election", "president", "congress", "senate", "pope", "netanyahu"]
    tech_kw = ["tesla", "nvidia", "apple", "google", "openai", "ai ", "spacex",
               "elon musk", "anthropic", "meta ", "microsoft"]
    crypto_kw = ["bitcoin", "btc", "ethereum", "eth", "solana", "crypto", "defi"]

    def classify(title: str) -> str:
        t = title.lower()
        for kw in crypto_kw:
            if kw in t:
                return "crypto"
        for kw in geo_kw:
            if kw in t:
                return "geopolitics"
        for kw in tech_kw:
            if kw in t:
                return "tech"
        return "other"

    # Collect by niche
    by_niche: dict[str, tuple[list[float], list[int]]] = {}
    all_probs = []
    all_labels = []

    for prob, label, title, _cat in rows:
        p = float(prob)
        l = int(label)
        all_probs.append(p)
        all_labels.append(l)

        niche = classify(title)
        if niche not in by_niche:
            by_niche[niche] = ([], [])
        by_niche[niche][0].append(p)
        by_niche[niche][1].append(l)

    result = {"overall": compute_ece(all_probs, all_labels)}
    for niche, (probs, labels) in by_niche.items():
        if len(probs) >= 10:  # Need enough samples
            result[niche] = compute_ece(probs, labels)

    return result
