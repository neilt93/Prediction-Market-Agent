"""Backtest the agent on historical resolved Polymarket markets.

1. Fetch resolved events from Polymarket (known outcomes)
2. For each market, reconstruct the state BEFORE resolution
3. Run the forecaster as if we didn't know the outcome
4. Compare forecast vs actual outcome
5. Generate postmortems with Brier scores
6. Retrain the calibrator on accumulated data
7. Print a performance report
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

# Add package paths
for pkg in ["shared", "schemas", "market_ingest", "rules", "forecasting", "calibration", "execution", "training"]:
    sys.path.insert(0, str(Path(__file__).parent / "packages" / pkg / "src"))

from shared.config import BaseAppSettings
from shared.logging import setup_logging
from shared.db import create_sync_engine, create_sync_session_factory

import structlog
logger = structlog.get_logger()


def get_db():
    settings = BaseAppSettings()
    engine = create_sync_engine(settings.database_url_sync)
    factory = create_sync_session_factory(engine)
    return factory()


async def fetch_resolved_markets(session, max_events: int = 100):
    """Fetch resolved Polymarket events with known outcomes."""
    from market_ingest.clients.polymarket.gamma_client import GammaClient
    from market_ingest.clients.polymarket.clob_client import ClobClient
    from market_ingest.mappers.polymarket_mapper import PolymarketMapper
    from schemas.models.market import Market, MarketOutcome, MarketSnapshot

    gamma = GammaClient()
    clob = ClobClient()
    mapper = PolymarketMapper()
    ingested = 0

    try:
        # Fetch resolved events across multiple pages for diversity
        offset = 0
        all_events = []
        pages_fetched = 0
        while len(all_events) < max_events and pages_fetched < 5:
            events = await gamma.list_events(
                closed=True, limit=100, offset=offset, order="volume", ascending=False,
            )
            if not events:
                break
            all_events.extend(events)
            offset += 100
            pages_fetched += 1

        logger.info(f"Fetched {len(all_events)} resolved events")

        for event in all_events[:max_events]:
            if not event.markets:
                continue

            for pm in event.markets:
                if not pm.enable_order_book:
                    continue
                if not (pm.closed or pm.archived):
                    continue

                # Determine outcome
                outcome_data = mapper.market_to_outcome(pm)
                if not outcome_data or outcome_data["resolved_label"] is None:
                    continue

                cid = pm.condition_id or pm.id
                existing = (
                    session.query(Market)
                    .filter(Market.platform == "polymarket", Market.platform_market_id == cid)
                    .first()
                )
                if existing:
                    continue

                # Create market record
                market_data = mapper.market_to_db(pm, event)
                market_data["status"] = "resolved"
                market_data["metadata_json"] = {
                    "clob_token_ids": pm.clob_token_ids,
                    "outcomes": pm.outcomes,
                    "outcome_prices": pm.outcome_prices,
                }
                market = Market(**market_data)
                session.add(market)
                session.flush()

                # Store outcome
                outcome_data["market_id"] = market.id
                session.add(MarketOutcome(**outcome_data))

                # Try to get historical price for a synthetic "pre-resolution" snapshot
                if pm.clob_token_ids:
                    try:
                        history = await clob.get_prices_history(
                            pm.clob_token_ids[0], interval="max", fidelity=60,
                        )
                        if history and len(history) > 10:
                            # Use a price from ~75% through the market's life
                            # This simulates having seen most evidence but not the final outcome
                            idx = int(len(history) * 0.75)
                            mid_point = history[idx]
                            # Also grab some earlier points for context
                            early_point = history[int(len(history) * 0.25)]

                            snap = MarketSnapshot(
                                market_id=market.id,
                                ts=datetime.fromtimestamp(mid_point.t, tz=timezone.utc),
                                mid_yes=Decimal(str(round(mid_point.p, 6))),
                                last_yes=Decimal(str(round(mid_point.p, 6))),
                                best_bid_yes=Decimal(str(round(max(0.01, mid_point.p - 0.02), 6))),
                                best_ask_yes=Decimal(str(round(min(0.99, mid_point.p + 0.02), 6))),
                                spread_bps=400,  # Synthetic 4-cent spread
                                liquidity_proxy=Decimal("100"),
                                orderbook_imbalance=Decimal("0.5"),
                            )
                            session.add(snap)
                    except Exception as e:
                        logger.debug(f"Price history unavailable for {cid}: {e}")

                ingested += 1
                if ingested % 25 == 0:
                    session.commit()
                    logger.info(f"Ingested {ingested} resolved markets...")

        session.commit()
        logger.info(f"Total resolved markets ingested: {ingested}")
    finally:
        await gamma.close()
        await clob.close()

    return ingested


async def run_backtest_forecasts(session):
    """Run the forecaster on resolved markets and score against outcomes."""
    from forecasting.forecaster import Forecaster
    from calibration.calibrator import Calibrator
    from execution.policy import ExecutionPolicy, PolicyConfig
    from rules.parser import RuleParser
    from schemas.models.market import Market, MarketSnapshot, MarketOutcome
    from schemas.models.forecast import Forecast, ForecastFeature, CalibratedForecast
    from schemas.models.postmortem import Postmortem

    rule_parser = RuleParser()
    calibrator = Calibrator(model_path="data/models/calibrator_latest.lgb")
    forecaster = Forecaster(
        api_url="http://localhost:11434/v1",
        model="llama3.1:8b",
    )
    policy = ExecutionPolicy(PolicyConfig(
        min_edge_bps=300,
        min_confidence=0.3,
        max_spread_bps=15000,
        max_position_per_market=3,
    ))

    # Get resolved markets with snapshots and outcomes (only those with price history)
    from sqlalchemy import exists
    import random
    all_candidates = (
        session.query(Market, MarketOutcome)
        .join(MarketOutcome, MarketOutcome.market_id == Market.id)
        .filter(
            Market.status == "resolved",
            MarketOutcome.resolved_label.isnot(None),
            exists().where(MarketSnapshot.market_id == Market.id),
        )
        .all()
    )

    # Diversify: pick at most 5 markets per event/subtitle, shuffle for variety
    seen_subtitles: dict[str, int] = {}
    diverse = []
    random.shuffle(all_candidates)
    for m, o in all_candidates:
        key = m.subtitle or m.title[:30]
        seen_subtitles[key] = seen_subtitles.get(key, 0) + 1
        if seen_subtitles[key] <= 5:
            diverse.append((m, o))
    markets_with_outcomes = diverse[:50]

    logger.info(f"Backtesting on {len(markets_with_outcomes)} diverse resolved markets (from {len(all_candidates)} total)")

    results = []
    for market, outcome in markets_with_outcomes:
        # Skip if already forecasted
        existing_forecast = session.query(Forecast).filter(Forecast.market_id == market.id).first()
        if existing_forecast:
            continue

        # Get snapshot (our "pre-resolution" view)
        snapshot = (
            session.query(MarketSnapshot)
            .filter(MarketSnapshot.market_id == market.id)
            .order_by(MarketSnapshot.ts.desc())
            .first()
        )

        market_price = float(snapshot.mid_yes) if snapshot and snapshot.mid_yes else None
        spread_bps = snapshot.spread_bps if snapshot else None
        liquidity = float(snapshot.liquidity_proxy) if snapshot and snapshot.liquidity_proxy else None
        time_to_close = snapshot.time_to_close_sec if snapshot else None

        # Parse rules
        parsed = rule_parser.parse(market.title, market.rules_text)

        # Forecast — the LLM sees market title, rules, market price, but NOT the outcome
        forecast_output = await forecaster.forecast(
            title=market.title,
            rules_text=market.rules_text,
            parsed_rules=parsed.model_dump(),
            market_price=market_price,
            time_to_close_hours=time_to_close / 3600 if time_to_close else None,
        )

        # Store forecast
        forecast = Forecast(
            market_id=market.id,
            ts=datetime.now(tz=timezone.utc),
            model_name="llama3.1:8b",
            raw_probability=Decimal(str(round(forecast_output.raw_probability, 6))),
            confidence=Decimal(str(round(forecast_output.confidence, 6))),
            abstain_flag=forecast_output.abstain,
            reasoning_summary=forecast_output.reasoning_summary,
        )
        session.add(forecast)
        session.flush()

        # Store features
        features = {
            "market_price": market_price or 0.5,
            "spread_bps": spread_bps or 0,
            "vol_24h": 0,
            "time_to_close_sec": time_to_close or 0,
            "ambiguity_score": parsed.ambiguity_score,
            "freshness_score": 0.5,
            "source_agreement_score": 0.5,
            "official_source_present": 1.0 if parsed.source_of_truth else 0.0,
            "llm_confidence": forecast_output.confidence,
            "retrieval_count": 0,
            "price_momentum_1h": 0.0,
            "price_momentum_24h": 0.0,
            "raw_probability": forecast_output.raw_probability,
        }
        feat_cols = {k: v for k, v in features.items() if k != "raw_probability"}
        session.add(ForecastFeature(forecast_id=forecast.id, **feat_cols))

        # Calibrate
        cal_output = calibrator.predict(features, market_price=market_price)
        session.add(CalibratedForecast(
            forecast_id=forecast.id,
            calibrator_version=calibrator.version,
            calibrated_probability=Decimal(str(round(cal_output.calibrated_probability, 6))),
            predicted_edge_bps=cal_output.predicted_edge_bps,
            uncertainty_low=Decimal(str(round(cal_output.uncertainty_low, 6))),
            uncertainty_high=Decimal(str(round(cal_output.uncertainty_high, 6))),
        ))

        # Score against reality
        label = outcome.resolved_label
        prob = forecast_output.raw_probability
        brier = (prob - label) ** 2
        eps = 1e-7
        log_loss_val = -(label * math.log(prob + eps) + (1 - label) * math.log(1 - prob + eps))

        # Would-have-traded analysis
        decision = policy.evaluate(
            calibrated_probability=cal_output.calibrated_probability,
            market_price=market_price or 0.5,
            confidence=forecast_output.confidence,
            ambiguity_score=parsed.ambiguity_score,
            spread_bps=spread_bps,
            liquidity=liquidity,
            abstain_flag=forecast_output.abstain,
        )

        # Compute hypothetical PnL if we'd traded
        hyp_pnl = None
        if decision.should_trade and market_price is not None:
            if decision.side == "buy_yes":
                # Bought YES at market_price, settled at label (0 or 1)
                hyp_pnl = round((label - market_price) * decision.quantity, 4)
            else:
                # Bought NO at (1 - market_price), settled at (1 - label)
                hyp_pnl = round(((1 - label) - (1 - market_price)) * decision.quantity, 4)

        # Classify error
        error_bucket = _classify_error(prob, label, forecast_output.confidence, parsed.ambiguity_score)

        # Store postmortem
        session.add(Postmortem(
            forecast_id=forecast.id,
            market_id=market.id,
            resolved_label=label,
            brier=Decimal(str(round(brier, 6))),
            log_loss=Decimal(str(round(log_loss_val, 6))),
            trading_pnl=Decimal(str(hyp_pnl)) if hyp_pnl is not None else None,
            error_bucket=error_bucket,
            human_reviewed=False,
            training_weight=Decimal("1.0"),
        ))

        result = {
            "title": market.title[:60],
            "market_price": market_price,
            "forecast": round(prob, 3),
            "actual": label,
            "brier": round(brier, 4),
            "would_trade": decision.should_trade,
            "trade_side": decision.side if decision.should_trade else None,
            "hyp_pnl": hyp_pnl,
            "error_bucket": error_bucket,
        }
        results.append(result)

        direction = "OK" if (prob > 0.5 and label == 1) or (prob < 0.5 and label == 0) else "XX"
        pnl_str = f"  PnL: ${hyp_pnl:+.2f}" if hyp_pnl is not None else ""
        logger.info(
            f"{direction} {market.title[:55]:55s} | "
            f"forecast={prob:.2f} actual={label} brier={brier:.3f}{pnl_str}"
        )

    session.commit()
    await forecaster.close()
    return results


def _classify_error(prob: float, label: int, confidence: float, ambiguity: float) -> str | None:
    """Classify the type of error for learning."""
    error = abs(prob - label)
    if error < 0.15:
        return None  # Good forecast

    wrong_direction = (prob > 0.5 and label == 0) or (prob < 0.5 and label == 1)

    if confidence > 0.7 and wrong_direction:
        return "bad_calibration"
    if ambiguity > 0.3:
        return "ambiguous_should_have_abstained"
    if error > 0.4:
        return "missed_official_source"
    return "bad_calibration"


def retrain_calibrator(session):
    """Retrain the calibrator using backtest postmortems."""
    from training.trainer import CalibrationTrainer

    trainer = CalibrationTrainer(session)
    features_df, labels = trainer.build_training_dataset()

    if len(features_df) < 10:
        logger.warning(f"Only {len(features_df)} samples — need more data for meaningful training")
        if len(features_df) >= 5:
            logger.info("Training anyway with small dataset for demo...")
        else:
            return None

    result = trainer.retrain(save_path="data/models/calibrator_latest.lgb")
    logger.info(f"Calibrator retrained: {result}")

    mistakes = trainer.get_mistake_summary()
    if mistakes:
        logger.info(f"Mistake breakdown: {mistakes}")

    return result


def print_report(results: list[dict]):
    """Print a performance report."""
    if not results:
        logger.info("No results to report")
        return

    n = len(results)
    briers = [r["brier"] for r in results]
    avg_brier = sum(briers) / n

    # Directional accuracy
    correct = sum(
        1 for r in results
        if (r["forecast"] > 0.5 and r["actual"] == 1) or (r["forecast"] < 0.5 and r["actual"] == 0)
    )
    accuracy = correct / n

    # Market baseline (what if we just used the market price as our forecast?)
    market_briers = [
        (r["market_price"] - r["actual"]) ** 2
        for r in results
        if r["market_price"] is not None
    ]
    avg_market_brier = sum(market_briers) / len(market_briers) if market_briers else None

    # Trading PnL
    trades = [r for r in results if r["hyp_pnl"] is not None]
    total_pnl = sum(r["hyp_pnl"] for r in trades) if trades else 0
    winners = sum(1 for r in trades if r["hyp_pnl"] > 0)
    losers = sum(1 for r in trades if r["hyp_pnl"] < 0)

    # Error buckets
    from collections import Counter
    error_counts = Counter(r["error_bucket"] for r in results if r["error_bucket"])

    print("\n" + "=" * 70)
    print("              BACKTEST PERFORMANCE REPORT")
    print("=" * 70)
    print(f"\n  Markets evaluated:     {n}")
    print(f"  Directional accuracy:  {correct}/{n} ({accuracy:.1%})")
    print(f"\n  Avg Brier score:       {avg_brier:.4f}  (lower = better, 0 = perfect)")
    if avg_market_brier is not None:
        print(f"  Market baseline Brier: {avg_market_brier:.4f}")
        edge = avg_market_brier - avg_brier
        print(f"  Edge vs market:        {edge:+.4f}  {'(BEATING market)' if edge > 0 else '(losing to market)'}")

    if trades:
        print(f"\n  --- Hypothetical Trading ---")
        print(f"  Trades taken:          {len(trades)}")
        print(f"  Winners:               {winners}")
        print(f"  Losers:                {losers}")
        print(f"  Win rate:              {winners/len(trades):.1%}" if trades else "")
        print(f"  Total PnL:             ${total_pnl:+.2f}")
    else:
        print(f"\n  No trades met risk gates")

    if error_counts:
        print(f"\n  --- Mistake Taxonomy ---")
        for bucket, count in error_counts.most_common():
            print(f"  {bucket:40s} {count}")

    print("\n" + "=" * 70)


async def main():
    setup_logging(log_level="INFO")
    session = get_db()

    try:
        print("\n" + "=" * 70)
        print("  STEP 1: Fetching resolved markets with known outcomes")
        print("=" * 70)
        n_resolved = await fetch_resolved_markets(session, max_events=30)

        print("\n" + "=" * 70)
        print("  STEP 2: Running forecaster on historical markets")
        print("  (the LLM sees market title + rules + market price, NOT the outcome)")
        print("=" * 70)
        results = await run_backtest_forecasts(session)

        print("\n" + "=" * 70)
        print("  STEP 3: Scoring and generating postmortems")
        print("=" * 70)
        print_report(results)

        print("\n" + "=" * 70)
        print("  STEP 4: Retraining calibrator on results")
        print("=" * 70)
        retrain_result = retrain_calibrator(session)

        if retrain_result:
            print(f"\n  Calibrator retrained and saved to data/models/calibrator_latest.lgb")
            print(f"  Version: {retrain_result.get('version')}")
            print(f"  Training samples: {retrain_result.get('n_samples')}")
            print(f"  Training Brier: {retrain_result.get('brier_score', 'N/A')}")
        else:
            print(f"\n  Not enough data to retrain yet — need more resolved markets")

    finally:
        session.close()


if __name__ == "__main__":
    asyncio.run(main())
