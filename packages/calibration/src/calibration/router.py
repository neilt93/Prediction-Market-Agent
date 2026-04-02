"""Category-routed calibrator ensemble.

Trains separate LightGBM calibrators for each niche (geopolitics, crypto, tech)
and routes predictions to the appropriate specialist. Falls back to the general
calibrator for unknown categories.

Research basis: MoLE (Mixture of Linear Experts) — training separate expert
models per domain and routing outperforms a single model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from calibration.calibrator import Calibrator, CalibratedOutput, FEATURE_COLUMNS

logger = structlog.get_logger()

MODEL_DIR = Path("data/models")


class CalibratorRouter:
    """Routes calibration to niche-specific models."""

    def __init__(self, model_dir: str | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.calibrators: dict[str, Calibrator] = {}
        self.fallback = Calibrator()

        # Load any existing niche models
        for niche in ["geopolitics", "crypto", "tech"]:
            path = self.model_dir / f"calibrator_{niche}.lgb"
            if path.exists():
                self.calibrators[niche] = Calibrator(model_path=str(path))
                logger.info(f"Loaded {niche} calibrator from {path}")

        # Load general fallback
        general_path = self.model_dir / "calibrator_latest.lgb"
        if general_path.exists():
            self.fallback = Calibrator(model_path=str(general_path))

    def predict(
        self,
        features: dict[str, float],
        market_price: float | None = None,
        niche: str | None = None,
    ) -> CalibratedOutput:
        """Route to niche calibrator or fallback."""
        if niche and niche in self.calibrators:
            return self.calibrators[niche].predict(features, market_price)
        return self.fallback.predict(features, market_price)

    def train_all(self, session: Any) -> dict[str, Any]:
        """Train niche-specific calibrators from DB data."""
        from schemas.models.forecast import Forecast, ForecastFeature
        from schemas.models.market import Market, MarketOutcome
        from schemas.models.postmortem import Postmortem

        # Niche keywords
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

        # Get all training data
        rows = (
            session.query(
                ForecastFeature,
                Forecast.raw_probability,
                MarketOutcome.resolved_label,
                Market.title,
            )
            .join(Forecast, ForecastFeature.forecast_id == Forecast.id)
            .join(MarketOutcome, Forecast.market_id == MarketOutcome.market_id)
            .join(Market, Market.id == Forecast.market_id)
            .filter(MarketOutcome.resolved_label.isnot(None))
            .all()
        )

        if not rows:
            return {"status": "no_data"}

        # Split by niche
        niche_data: dict[str, tuple[list[dict], list[int]]] = {}
        for feature, raw_prob, label, title in rows:
            niche = classify(title)
            if niche not in niche_data:
                niche_data[niche] = ([], [])

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
            niche_data[niche][0].append(row)
            niche_data[niche][1].append(int(label))

        # Train each niche
        self.model_dir.mkdir(parents=True, exist_ok=True)
        results = {}

        for niche, (feature_rows, labels) in niche_data.items():
            n_pos = sum(labels)
            n_neg = len(labels) - n_pos
            if len(feature_rows) < 20 or n_pos < 3 or n_neg < 3:
                results[niche] = {"status": "skipped", "n_samples": len(feature_rows),
                                  "reason": f"need 20+ samples with 3+ of each class (got {n_pos}Y/{n_neg}N)"}
                continue

            df = pd.DataFrame(feature_rows)
            series = pd.Series(labels, dtype=float)

            cal = Calibrator()
            try:
                metrics = cal.train(df, series)
            except Exception as e:
                results[niche] = {"status": "error", "error": str(e)}
                continue

            path = str(self.model_dir / f"calibrator_{niche}.lgb")
            cal.save(path)
            self.calibrators[niche] = cal

            results[niche] = {
                "status": "trained",
                "n_samples": len(feature_rows),
                **metrics,
            }

        logger.info("niche_calibrators_trained", results=results)
        return results
