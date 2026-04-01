"""Training service that retrains the calibrator from resolved postmortems."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sqlalchemy.orm import Session

from schemas.models.forecast import CalibratedForecast, Forecast, ForecastFeature
from schemas.models.market import MarketOutcome
from schemas.models.postmortem import Postmortem
from calibration.calibrator import Calibrator, FEATURE_COLUMNS

logger = structlog.get_logger()

MODEL_DIR = Path("data/models")


class CalibrationTrainer:
    """Retrains the calibrator on accumulated postmortem data."""

    def __init__(self, db_session: Session) -> None:
        self.db = db_session

    def build_training_dataset(self) -> tuple[pd.DataFrame, pd.Series]:
        """Build training dataset from forecasts with known outcomes."""
        # Join forecasts -> features -> outcomes
        results = (
            self.db.query(
                ForecastFeature,
                Forecast.raw_probability,
                MarketOutcome.resolved_label,
                Postmortem.training_weight,
            )
            .join(Forecast, ForecastFeature.forecast_id == Forecast.id)
            .join(MarketOutcome, Forecast.market_id == MarketOutcome.market_id)
            .outerjoin(Postmortem, Forecast.market_id == Postmortem.market_id)
            .filter(MarketOutcome.resolved_label.isnot(None))
            .all()
        )

        if not results:
            return pd.DataFrame(), pd.Series(dtype=float)

        rows = []
        labels = []
        for feature, raw_prob, label, weight in results:
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

        return pd.DataFrame(rows), pd.Series(labels, dtype=float)

    def retrain(self, save_path: str | None = None) -> dict[str, Any]:
        """Retrain the calibrator and optionally save it."""
        features_df, labels = self.build_training_dataset()

        if len(features_df) < 20:
            logger.warning("insufficient_training_data", n_samples=len(features_df))
            return {"status": "skipped", "reason": "insufficient_data", "n_samples": len(features_df)}

        calibrator = Calibrator()
        metrics = calibrator.train(features_df, labels)

        if save_path is None:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            save_path = str(MODEL_DIR / "calibrator_latest.lgb")

        calibrator.save(save_path)

        return {
            "status": "success",
            "model_path": save_path,
            "version": calibrator.version,
            **metrics,
        }

    def get_mistake_summary(self) -> dict[str, int]:
        """Summarize error buckets from postmortems."""
        postmortems = self.db.query(Postmortem).filter(Postmortem.error_bucket.isnot(None)).all()
        summary: dict[str, int] = {}
        for pm in postmortems:
            bucket = pm.error_bucket
            summary[bucket] = summary.get(bucket, 0) + 1
        return summary
