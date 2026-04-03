"""Backtest the agent on historical resolved Polymarket markets.

Full pipeline:
1. Fetch hundreds of resolved events with known outcomes
2. For each market, reconstruct pre-resolution state
3. Gather evidence (news, crypto prices, Wikipedia)
4. Run LLM forecaster with evidence
5. Score against actual outcomes
6. Generate postmortems with error classification
7. Retrain calibrator on accumulated data
"""
from __future__ import annotations

import asyncio
from collections import Counter
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

for pkg in ["shared", "schemas", "market_ingest", "rules", "forecasting",
            "calibration", "execution", "training", "evidence", "diffusion"]:
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


EVAL_LOOKBACK_HOURS = (24, 12, 6, 2)


def _select_history_point(history: list[Any], close_time: datetime | None) -> Any | None:
    if close_time is None:
        return None
    close_ts = int(close_time.timestamp())
    for hours in EVAL_LOOKBACK_HOURS:
        cutoff_ts = close_ts - int(hours * 3600)
        candidates = [point for point in history if point.t <= cutoff_ts]
        if candidates:
            return candidates[-1]
    return None


def _select_backtest_snapshot(session, market) -> Any | None:
    from schemas.models.market import MarketSnapshot

    query = session.query(MarketSnapshot).filter(MarketSnapshot.market_id == market.id)
    if market.close_time is None:
        return query.order_by(MarketSnapshot.ts.asc()).first()

    for hours in EVAL_LOOKBACK_HOURS:
        cutoff = market.close_time - timedelta(hours=hours)
        snapshot = (
            query.filter(MarketSnapshot.ts <= cutoff)
            .order_by(MarketSnapshot.ts.desc())
            .first()
        )
        if snapshot:
            return snapshot
    return None


# ─── STEP 1: Ingest resolved markets ───────────────────────────────────────

async def fetch_resolved_markets(session, max_pages: int = 5):
    from market_ingest.clients.polymarket.gamma_client import GammaClient
    from market_ingest.clients.polymarket.clob_client import ClobClient
    from market_ingest.mappers.polymarket_mapper import PolymarketMapper
    from schemas.models.market import Market, MarketOutcome, MarketSnapshot

    gamma = GammaClient()
    clob = ClobClient()
    mapper = PolymarketMapper()
    ingested = 0

    try:
        all_events = []
        for page in range(max_pages):
            events = await gamma.list_events(
                closed=True, limit=100, offset=page * 100,
                order="volume", ascending=False,
            )
            if not events:
                break
            all_events.extend(events)
        logger.info(f"Fetched {len(all_events)} resolved events across {max_pages} pages")

        for event in all_events:
            if not event.markets:
                continue
            for pm in event.markets:
                if not pm.enable_order_book or not (pm.closed or pm.archived):
                    continue
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

                outcome_data["market_id"] = market.id
                session.add(MarketOutcome(**outcome_data))

                # Build a snapshot from the last observable price at least a few
                # hours before close so the backtest does not peek near resolution.
                if pm.clob_token_ids:
                    try:
                        history = await clob.get_prices_history(
                            pm.clob_token_ids[0], interval="max", fidelity=60,
                        )
                        if history:
                            pt = _select_history_point(history, market_data["close_time"])
                            if pt is not None:
                                snapshot_ts = datetime.fromtimestamp(pt.t, tz=timezone.utc)
                                time_to_close_sec = None
                                if market_data["close_time"] is not None:
                                    time_to_close_sec = max(
                                        0,
                                        int((market_data["close_time"] - snapshot_ts).total_seconds()),
                                    )
                                snap = MarketSnapshot(
                                    market_id=market.id,
                                    ts=snapshot_ts,
                                    mid_yes=Decimal(str(round(pt.p, 6))),
                                    last_yes=Decimal(str(round(pt.p, 6))),
                                    best_bid_yes=Decimal(str(round(max(0.01, pt.p - 0.02), 6))),
                                    best_ask_yes=Decimal(str(round(min(0.99, pt.p + 0.02), 6))),
                                    spread_bps=400,
                                    liquidity_proxy=Decimal("100"),
                                    orderbook_imbalance=Decimal("0.5"),
                                    time_to_close_sec=time_to_close_sec,
                                )
                                session.add(snap)
                    except Exception:
                        pass

                ingested += 1
                if ingested % 50 == 0:
                    session.commit()
                    logger.info(f"  ...ingested {ingested} resolved markets")

        session.commit()
        logger.info(f"Total resolved markets ingested: {ingested}")
    finally:
        await gamma.close()
        await clob.close()
    return ingested


# ─── STEP 2: Backtest with evidence ────────────────────────────────────────

async def run_backtest(session, max_markets: int = 50):
    from forecasting.forecaster import Forecaster
    from calibration.calibrator import Calibrator
    from execution.policy import ExecutionPolicy, PolicyConfig
    from rules.parser import RuleParser
    from evidence.retriever import EvidenceRetriever
    from schemas.models.market import Market, MarketSnapshot, MarketOutcome
    from schemas.models.forecast import Forecast, ForecastFeature, CalibratedForecast
    from schemas.models.postmortem import Postmortem
    from sqlalchemy import exists

    settings = BaseAppSettings()
    rule_parser = RuleParser()
    calibrator = Calibrator()
    forecaster = Forecaster(api_url=settings.llm_api_url, model=settings.llm_model)
    evidence = EvidenceRetriever()
    policy = ExecutionPolicy(PolicyConfig(
        min_edge_bps=300, min_confidence=0.3, max_spread_bps=15000,
        max_position_per_market=3, daily_max_loss=50.0,
    ))

    # Get diverse resolved markets with price history
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

    # Diversify — max 3 per event subtitle
    seen: dict[str, int] = {}
    diverse = []
    random.shuffle(all_candidates)
    for m, o in all_candidates:
        key = (m.subtitle or "")[:40]
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= 3:
            diverse.append((m, o))
    candidates = diverse[:max_markets]

    logger.info(f"Backtesting {len(candidates)} markets (from {len(all_candidates)} with price history)")

    results = []
    for i, (market, outcome) in enumerate(candidates):
        # Skip already forecasted
        if session.query(Forecast).filter(Forecast.market_id == market.id).first():
            continue

        snapshot = _select_backtest_snapshot(session, market)
        if snapshot is None:
            continue
        market_price = float(snapshot.mid_yes) if snapshot and snapshot.mid_yes else None
        spread_bps = snapshot.spread_bps if snapshot else None
        liquidity = float(snapshot.liquidity_proxy) if snapshot and snapshot.liquidity_proxy else None
        time_to_close = snapshot.time_to_close_sec if snapshot else None
        if time_to_close is None and snapshot and market.close_time:
            time_to_close = max(0, int((market.close_time - snapshot.ts).total_seconds()))

        # Parse rules
        parsed = rule_parser.parse(market.title, market.rules_text)

        # Gather evidence
        bundle = await evidence.gather(
            title=market.title,
            entity=parsed.entity,
            category=market.category,
            as_of=snapshot.ts,
        )

        # Forecast with evidence
        forecast_output = await forecaster.forecast(
            title=market.title,
            rules_text=market.rules_text,
            parsed_rules=parsed.model_dump(),
            evidence_snippets=bundle.top_snippets(8),
            market_price=market_price,
            time_to_close_hours=time_to_close / 3600 if time_to_close else None,
        )

        # Store forecast
        forecast = Forecast(
            market_id=market.id,
            ts=datetime.now(tz=timezone.utc),
            model_name=settings.llm_model,
            raw_probability=Decimal(str(round(forecast_output.raw_probability, 6))),
            confidence=Decimal(str(round(forecast_output.confidence, 6))),
            abstain_flag=forecast_output.abstain,
            reasoning_summary=forecast_output.reasoning_summary[:500] if forecast_output.reasoning_summary else None,
        )
        session.add(forecast)
        session.flush()

        # Features
        features = {
            "market_price": market_price or 0.5,
            "spread_bps": spread_bps or 0,
            "vol_24h": 0,
            "time_to_close_sec": time_to_close or 0,
            "ambiguity_score": parsed.ambiguity_score,
            "freshness_score": min(1.0, bundle.count / 10),
            "source_agreement_score": 0.5,
            "official_source_present": 1.0 if parsed.source_of_truth else 0.0,
            "llm_confidence": forecast_output.confidence,
            "retrieval_count": bundle.count,
            "price_momentum_1h": 0.0,
            "price_momentum_24h": 0.0,
            "raw_probability": forecast_output.raw_probability,
        }
        feat_cols = {k: v for k, v in features.items() if k != "raw_probability"}
        session.add(ForecastFeature(forecast_id=forecast.id, **feat_cols))

        # Calibrate
        cal = calibrator.predict(features, market_price=market_price)
        session.add(CalibratedForecast(
            forecast_id=forecast.id,
            calibrator_version=calibrator.version,
            calibrated_probability=Decimal(str(round(cal.calibrated_probability, 6))),
            predicted_edge_bps=cal.predicted_edge_bps,
            uncertainty_low=Decimal(str(round(cal.uncertainty_low, 6))),
            uncertainty_high=Decimal(str(round(cal.uncertainty_high, 6))),
        ))

        # Score
        label = outcome.resolved_label
        prob = forecast_output.raw_probability
        brier = (prob - label) ** 2
        eps = 1e-7
        ll = -(label * math.log(prob + eps) + (1 - label) * math.log(1 - prob + eps))

        # Would-have-traded
        decision = policy.evaluate(
            calibrated_probability=cal.calibrated_probability,
            market_price=market_price or 0.5,
            confidence=forecast_output.confidence,
            ambiguity_score=parsed.ambiguity_score,
            spread_bps=spread_bps, liquidity=liquidity,
            abstain_flag=forecast_output.abstain,
        )
        hyp_pnl = None
        if decision.should_trade and market_price is not None:
            if decision.side == "buy_yes":
                hyp_pnl = round((label - market_price) * decision.quantity, 4)
            else:
                hyp_pnl = round(((1 - label) - (1 - market_price)) * decision.quantity, 4)

        error_bucket = _classify_error(prob, label, forecast_output.confidence, parsed.ambiguity_score)

        session.add(Postmortem(
            forecast_id=forecast.id, market_id=market.id,
            resolved_label=label,
            brier=Decimal(str(round(brier, 6))),
            log_loss=Decimal(str(round(ll, 6))),
            trading_pnl=Decimal(str(hyp_pnl)) if hyp_pnl is not None else None,
            error_bucket=error_bucket,
            human_reviewed=False, training_weight=Decimal("1.0"),
        ))

        results.append({
            "title": market.title[:60],
            "market_price": market_price,
            "forecast": round(prob, 3),
            "actual": label,
            "brier": round(brier, 4),
            "would_trade": decision.should_trade,
            "trade_side": decision.side if decision.should_trade else None,
            "hyp_pnl": hyp_pnl,
            "error_bucket": error_bucket,
            "evidence_count": bundle.count,
        })

        ok = "OK" if (prob > 0.5 and label == 1) or (prob < 0.5 and label == 0) else "XX"
        pnl_str = f"  PnL: ${hyp_pnl:+.2f}" if hyp_pnl is not None else ""
        evid = f"[{bundle.count} sources]"
        logger.info(
            f"  {ok} {market.title[:50]:50s} | "
            f"p={prob:.2f} actual={label} brier={brier:.3f} {evid}{pnl_str}"
        )

        if (i + 1) % 10 == 0:
            session.commit()

    session.commit()
    await forecaster.close()
    await evidence.close()
    return results


def _classify_error(prob: float, label: int, confidence: float, ambiguity: float) -> str | None:
    error = abs(prob - label)
    if error < 0.15:
        return None
    wrong = (prob > 0.5 and label == 0) or (prob < 0.5 and label == 1)
    if ambiguity > 0.3:
        return "ambiguous_should_have_abstained"
    if confidence > 0.7 and wrong:
        return "bad_calibration"
    if error > 0.4 and wrong:
        return "missed_official_source"
    if error > 0.3:
        return "bad_calibration"
    return "stale_evidence"


# ─── STEP 3: Retrain calibrator ────────────────────────────────────────────

def retrain_calibrator(session):
    from training.trainer import CalibrationTrainer
    trainer = CalibrationTrainer(session)
    features_df, labels = trainer.build_training_dataset()

    if len(features_df) < 10:
        logger.warning(f"Only {len(features_df)} samples for training")
        return None

    Path("data/models").mkdir(parents=True, exist_ok=True)
    result = trainer.retrain(save_path="data/models/calibrator_latest.lgb")
    mistakes = trainer.get_mistake_summary()
    return result, mistakes


# ─── STEP 4: Report ────────────────────────────────────────────────────────

def print_report(results: list[dict]):
    if not results:
        print("No results.")
        return

    n = len(results)
    briers = [r["brier"] for r in results]
    avg_brier = sum(briers) / n

    correct = sum(
        1 for r in results
        if (r["forecast"] > 0.5 and r["actual"] == 1) or
           (r["forecast"] < 0.5 and r["actual"] == 0) or
           (r["forecast"] == 0.5)
    )

    market_briers = [
        (r["market_price"] - r["actual"]) ** 2
        for r in results if r["market_price"] is not None
    ]
    avg_mkt_brier = sum(market_briers) / len(market_briers) if market_briers else None

    trades = [r for r in results if r["hyp_pnl"] is not None]
    total_pnl = sum(r["hyp_pnl"] for r in trades) if trades else 0
    winners = sum(1 for r in trades if r["hyp_pnl"] > 0)
    losers = sum(1 for r in trades if r["hyp_pnl"] < 0)
    flat = sum(1 for r in trades if r["hyp_pnl"] == 0)

    error_counts = Counter(r["error_bucket"] for r in results if r["error_bucket"])
    avg_evidence = sum(r["evidence_count"] for r in results) / n

    print("\n" + "=" * 70)
    print("              BACKTEST PERFORMANCE REPORT")
    print("=" * 70)
    print(f"\n  Markets evaluated:     {n}")
    print(f"  Avg evidence per mkt:  {avg_evidence:.1f} sources")
    print(f"  Directional accuracy:  {correct}/{n} ({correct/n:.1%})")
    print(f"\n  Avg Brier score:       {avg_brier:.4f}  (lower = better)")
    if avg_mkt_brier is not None:
        print(f"  Market baseline Brier: {avg_mkt_brier:.4f}")
        edge = avg_mkt_brier - avg_brier
        if edge > 0:
            print(f"  Edge vs market:        {edge:+.4f}  ** BEATING THE MARKET **")
        else:
            print(f"  Edge vs market:        {edge:+.4f}  (market is better)")

    if trades:
        print(f"\n  --- Hypothetical Trading ---")
        print(f"  Trades taken:          {len(trades)}")
        print(f"  Winners / Losers:      {winners}W / {losers}L / {flat}flat")
        if len(trades) > 0:
            print(f"  Win rate:              {winners/len(trades):.1%}")
        print(f"  Total PnL:             ${total_pnl:+.2f}")

    # Show best and worst calls
    sorted_by_brier = sorted(results, key=lambda r: r["brier"])
    print(f"\n  --- Best Calls (lowest Brier) ---")
    for r in sorted_by_brier[:5]:
        print(f"    {r['title'][:55]:55s} brier={r['brier']:.3f} p={r['forecast']:.2f} actual={r['actual']}")

    worst = sorted(results, key=lambda r: r["brier"], reverse=True)
    print(f"\n  --- Worst Calls (highest Brier) ---")
    for r in worst[:5]:
        print(f"    {r['title'][:55]:55s} brier={r['brier']:.3f} p={r['forecast']:.2f} actual={r['actual']}")

    if error_counts:
        print(f"\n  --- Mistake Taxonomy ---")
        for bucket, count in error_counts.most_common():
            pct = count / n * 100
            print(f"    {bucket:40s} {count:3d} ({pct:.0f}%)")

    print("\n" + "=" * 70)


# ─── Main ──────────────────────────────────────────────────────────────────

async def main():
    setup_logging(log_level="INFO")
    session = get_db()

    try:
        print("\n" + "=" * 70)
        print("  STEP 1: Fetching resolved markets (5 pages of events)")
        print("=" * 70)
        await fetch_resolved_markets(session, max_pages=5)

        print("\n" + "=" * 70)
        print("  STEP 2: Running forecaster with evidence on resolved markets")
        print("  (LLM sees title + rules + market price + news — NOT the outcome)")
        print("=" * 70)
        results = await run_backtest(session, max_markets=50)

        print("\n" + "=" * 70)
        print("  STEP 3: Performance report")
        print("=" * 70)
        print_report(results)

        print("\n" + "=" * 70)
        print("  STEP 4: Retraining calibrator on results")
        print("=" * 70)
        retrain_out = retrain_calibrator(session)
        if retrain_out:
            result, mistakes = retrain_out
            print(f"\n  Calibrator retrained: {result.get('version')}")
            print(f"  Samples: {result.get('n_samples')}")
            print(f"  Brier: {result.get('brier_score', 'N/A'):.4f}")
            if mistakes:
                print(f"  Mistake breakdown: {dict(mistakes)}")
        else:
            print("  Not enough data to retrain")

    finally:
        session.close()


if __name__ == "__main__":
    asyncio.run(main())
