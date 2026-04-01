"""Paper trading simulator that runs the full pipeline end-to-end.

This is the main entry point for running a trading simulation against live
market data (from Polymarket for research, Kalshi demo for execution).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import structlog
from sqlalchemy.orm import Session

from schemas.models.market import Market, MarketSnapshot
from schemas.models.forecast import Forecast, ForecastFeature, CalibratedForecast
from schemas.models.execution import Order, Position
from schemas.models.rules import RuleParse
from schemas.models.postmortem import Postmortem
from schemas.models.market import MarketOutcome

from rules.parser import RuleParser
from forecasting.forecaster import Forecaster, ForecastOutput
from calibration.calibrator import Calibrator, CalibratedOutput
from execution.policy import ExecutionPolicy, TradeDecision, PolicyConfig

logger = structlog.get_logger()


class TradingSimulator:
    """End-to-end trading simulation pipeline."""

    def __init__(
        self,
        db_session: Session,
        forecaster: Forecaster,
        calibrator: Calibrator,
        policy: ExecutionPolicy | None = None,
    ) -> None:
        self.db = db_session
        self.forecaster = forecaster
        self.calibrator = calibrator
        self.policy = policy or ExecutionPolicy()
        self.rule_parser = RuleParser()

    async def run_cycle(self) -> dict[str, Any]:
        """Run one full trading cycle: forecast -> calibrate -> decide -> execute."""
        markets = (
            self.db.query(Market)
            .filter(Market.status == "open")
            .all()
        )

        stats = {
            "markets_evaluated": 0,
            "forecasts_made": 0,
            "trades_attempted": 0,
            "trades_executed": 0,
            "abstains": 0,
        }

        for market in markets:
            try:
                result = await self._process_market(market)
                stats["markets_evaluated"] += 1
                if result.get("abstained"):
                    stats["abstains"] += 1
                if result.get("forecast_made"):
                    stats["forecasts_made"] += 1
                if result.get("trade_attempted"):
                    stats["trades_attempted"] += 1
                if result.get("trade_executed"):
                    stats["trades_executed"] += 1
            except Exception as e:
                logger.error("market_processing_failed", market_id=str(market.id), error=str(e))

        self.db.commit()
        logger.info("trading_cycle_complete", **stats)
        return stats

    async def _process_market(self, market: Market) -> dict[str, Any]:
        result: dict[str, Any] = {}

        # 1. Get latest snapshot
        snapshot = (
            self.db.query(MarketSnapshot)
            .filter(MarketSnapshot.market_id == market.id)
            .order_by(MarketSnapshot.ts.desc())
            .first()
        )

        market_price = float(snapshot.mid_yes) if snapshot and snapshot.mid_yes else None
        spread_bps = snapshot.spread_bps if snapshot else None
        liquidity = float(snapshot.liquidity_proxy) if snapshot and snapshot.liquidity_proxy else None
        time_to_close = snapshot.time_to_close_sec if snapshot else None

        # 2. Parse rules
        parsed_rule = self.rule_parser.parse(market.title, market.rules_text)

        # Store rule parse
        rule_parse = RuleParse(
            **self.rule_parser.to_db_dict(parsed_rule, market.id)
        )
        self.db.add(rule_parse)

        # 3. Forecast
        time_to_close_hours = time_to_close / 3600 if time_to_close else None

        forecast_output = await self.forecaster.forecast(
            title=market.title,
            rules_text=market.rules_text,
            parsed_rules=parsed_rule.model_dump(),
            market_price=market_price,
            time_to_close_hours=time_to_close_hours,
        )
        result["forecast_made"] = True

        if forecast_output.abstain:
            result["abstained"] = True

        # Store forecast
        forecast_dict = self.forecaster.to_db_dict(forecast_output, market.id)
        forecast = Forecast(**forecast_dict)
        self.db.add(forecast)
        self.db.flush()

        # 4. Build features for calibrator
        features = {
            "market_price": market_price or 0.5,
            "spread_bps": spread_bps or 0,
            "vol_24h": float(snapshot.volume_24h) if snapshot and snapshot.volume_24h else 0,
            "time_to_close_sec": time_to_close or 0,
            "ambiguity_score": parsed_rule.ambiguity_score,
            "freshness_score": 0.5,
            "source_agreement_score": 0.5,
            "official_source_present": 1.0 if parsed_rule.source_of_truth else 0.0,
            "llm_confidence": forecast_output.confidence,
            "retrieval_count": 0,
            "price_momentum_1h": 0.0,
            "price_momentum_24h": 0.0,
            "raw_probability": forecast_output.raw_probability,
        }

        # Store features
        feature_record = ForecastFeature(forecast_id=forecast.id, **{
            k: v for k, v in features.items()
            if k not in ("raw_probability",)  # raw_prob is already on the forecast
        })
        self.db.add(feature_record)

        # 5. Calibrate
        calibrated = self.calibrator.predict(features, market_price=market_price)

        # Store calibrated forecast
        cal_record = CalibratedForecast(
            forecast_id=forecast.id,
            calibrator_version=self.calibrator.version,
            calibrated_probability=calibrated.calibrated_probability,
            predicted_edge_bps=calibrated.predicted_edge_bps,
            uncertainty_low=calibrated.uncertainty_low,
            uncertainty_high=calibrated.uncertainty_high,
        )
        self.db.add(cal_record)

        # 6. Execution policy
        current_position = (
            self.db.query(Position)
            .filter(Position.market_id == market.id)
            .first()
        )
        current_qty = current_position.net_qty if current_position else 0

        decision = self.policy.evaluate(
            calibrated_probability=calibrated.calibrated_probability,
            market_price=market_price or 0.5,
            confidence=forecast_output.confidence,
            ambiguity_score=parsed_rule.ambiguity_score,
            spread_bps=spread_bps,
            liquidity=liquidity,
            current_position_qty=current_qty,
            category=market.category,
            abstain_flag=forecast_output.abstain,
        )

        if decision.should_trade:
            result["trade_attempted"] = True
            # In simulation mode: record order as filled immediately
            order = Order(
                market_id=market.id,
                forecast_id=forecast.id,
                platform=market.platform,
                env="demo",
                side=decision.side,
                order_type=decision.order_type,
                price=Decimal(str(decision.limit_price)),
                qty=decision.quantity,
                status="filled",
                submitted_at=datetime.now(tz=timezone.utc),
                filled_at=datetime.now(tz=timezone.utc),
                avg_fill_price=Decimal(str(decision.limit_price)),
                fees=Decimal("0"),
                slippage_bps=0,
            )
            self.db.add(order)

            # Update or create position
            if current_position:
                if decision.side == "buy_yes":
                    current_position.net_qty += decision.quantity
                else:
                    current_position.net_qty -= decision.quantity
                current_position.mark_price = Decimal(str(market_price or 0.5))
            else:
                qty = decision.quantity if decision.side == "buy_yes" else -decision.quantity
                position = Position(
                    market_id=market.id,
                    net_qty=qty,
                    avg_cost=Decimal(str(decision.limit_price)),
                    mark_price=Decimal(str(market_price or 0.5)),
                    realized_pnl=Decimal("0"),
                )
                self.db.add(position)

            result["trade_executed"] = True
            logger.info(
                "trade_executed",
                market=market.title[:60],
                side=decision.side,
                price=decision.limit_price,
                edge_bps=decision.edge_bps,
            )
        else:
            logger.debug(
                "trade_skipped",
                market=market.title[:60],
                reason=decision.reason,
            )

        return result

    def generate_postmortems(self) -> int:
        """Generate postmortems for all resolved markets with forecasts."""
        resolved = (
            self.db.query(Market)
            .filter(Market.status == "resolved")
            .all()
        )

        count = 0
        for market in resolved:
            # Check if postmortem already exists
            existing = (
                self.db.query(Postmortem)
                .filter(Postmortem.market_id == market.id)
                .first()
            )
            if existing:
                continue

            # Get outcome
            outcome = (
                self.db.query(MarketOutcome)
                .filter(MarketOutcome.market_id == market.id)
                .first()
            )
            if not outcome or outcome.resolved_label is None:
                continue

            # Get best forecast (highest confidence non-abstain)
            forecast = (
                self.db.query(Forecast)
                .filter(
                    Forecast.market_id == market.id,
                    Forecast.abstain_flag == False,
                )
                .order_by(Forecast.confidence.desc())
                .first()
            )
            if not forecast:
                continue

            # Compute metrics
            prob = float(forecast.raw_probability)
            label = outcome.resolved_label
            brier = (prob - label) ** 2
            import math
            eps = 1e-7
            log_loss_val = -(label * math.log(prob + eps) + (1 - label) * math.log(1 - prob + eps))

            # Get trading PnL
            position = (
                self.db.query(Position)
                .filter(Position.market_id == market.id)
                .first()
            )
            trading_pnl = float(position.realized_pnl) if position else None

            # Classify error
            error_bucket = self._classify_error(prob, label, forecast, market)

            postmortem = Postmortem(
                forecast_id=forecast.id,
                market_id=market.id,
                resolved_label=label,
                brier=Decimal(str(round(brier, 6))),
                log_loss=Decimal(str(round(log_loss_val, 6))),
                trading_pnl=Decimal(str(round(trading_pnl, 4))) if trading_pnl is not None else None,
                error_bucket=error_bucket,
                error_notes=None,
                human_reviewed=False,
                training_weight=Decimal("1.0"),
            )
            self.db.add(postmortem)
            count += 1

        self.db.commit()
        logger.info("postmortems_generated", count=count)
        return count

    @staticmethod
    def _classify_error(prob: float, label: int, forecast: Forecast, market: Market) -> str | None:
        error = abs(prob - label)
        if error < 0.15:
            return None  # Good forecast

        if forecast.confidence > 0.8 and error > 0.5:
            return "bad_calibration"
        if market.rules_text and len(market.rules_text) > 500:
            return "rule_misread"
        return "bad_calibration"
