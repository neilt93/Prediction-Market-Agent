"""Run multiple backtest passes to accumulate calibrator training data.

Each pass:
1. Picks 50 new diverse resolved markets (avoiding already-forecasted ones)
2. Gathers evidence + runs LLM forecasts
3. Scores against outcomes, generates postmortems
4. Retrains calibrator on ALL accumulated data

After all passes: prints niche-filtered analysis.
"""
from __future__ import annotations

import asyncio
import math
import sys
import random
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from collections import Counter

for pkg in ["shared", "schemas", "market_ingest", "rules", "forecasting",
            "calibration", "execution", "training", "evidence"]:
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


CRYPTO_KEYWORDS = ["bitcoin", "btc", "ethereum", "eth", "solana", "sol", "crypto",
                   "doge", "xrp", "cardano", "ada", "bnb", "defi", "nft", "token"]
GEOPOLITICS_KEYWORDS = ["trump", "iran", "ukraine", "russia", "china", "nato", "ceasefire",
                        "war", "sanctions", "tariff", "election", "president", "fed ",
                        "interest rate", "congress", "senate", "supreme court", "xi ",
                        "netanyahu", "putin", "zelenskyy", "biden", "government shutdown"]
TECH_KEYWORDS = ["tesla", "nvidia", "apple", "google", "openai", "ai ", "spacex",
                 "elon musk", "model ", "gpt", "anthropic", "meta ", "microsoft"]
SKIP_KEYWORDS = ["nba", "nfl", "nhl", "mlb", "ncaa", "spread:", "o/u ",
                 "rebounds", "assists", "points", "rushing", "passing",
                 "touchdown", "strikeout", "goals", "premier league",
                 "la liga", "champions league", "world cup", "f1 ", "formula",
                 "masters", "grand slam", "wimbledon", "stanley cup",
                 "super bowl", "tweets from", "tweet"]


def classify_market(title: str) -> str:
    t = title.lower()
    for kw in SKIP_KEYWORDS:
        if kw in t:
            return "skip"
    for kw in CRYPTO_KEYWORDS:
        if kw in t:
            return "crypto"
    for kw in GEOPOLITICS_KEYWORDS:
        if kw in t:
            return "geopolitics"
    for kw in TECH_KEYWORDS:
        if kw in t:
            return "tech"
    return "other"


async def run_one_pass(session, pass_num: int, per_pass: int = 50, niche_only: bool = False):
    """Run one backtest pass on un-forecasted markets."""
    from forecasting.forecaster import Forecaster
    from calibration.calibrator import Calibrator
    from execution.policy import ExecutionPolicy, PolicyConfig
    from rules.parser import RuleParser
    from evidence.retriever import EvidenceRetriever
    from schemas.models.market import Market, MarketSnapshot, MarketOutcome
    from schemas.models.forecast import Forecast, ForecastFeature, CalibratedForecast
    from schemas.models.postmortem import Postmortem
    from sqlalchemy import exists, and_

    rule_parser = RuleParser()
    calibrator = Calibrator(model_path="data/models/calibrator_latest.lgb")
    forecaster = Forecaster(api_url="http://localhost:11434/v1", model="llama3.1:8b")
    evidence_retriever = EvidenceRetriever()
    policy = ExecutionPolicy(PolicyConfig(
        min_edge_bps=300, min_confidence=0.3, max_spread_bps=15000,
        max_position_per_market=3, daily_max_loss=50.0,
    ))

    # Get un-forecasted resolved markets with price history
    already_forecasted = session.query(Forecast.market_id).subquery()
    candidates = (
        session.query(Market, MarketOutcome)
        .join(MarketOutcome, MarketOutcome.market_id == Market.id)
        .filter(
            Market.status == "resolved",
            MarketOutcome.resolved_label.isnot(None),
            exists().where(MarketSnapshot.market_id == Market.id),
            ~Market.id.in_(session.query(already_forecasted)),
        )
        .all()
    )

    # Filter by niche if requested
    if niche_only:
        candidates = [(m, o) for m, o in candidates if classify_market(m.title) in ("crypto", "geopolitics", "tech")]

    # Diversify
    seen: dict[str, int] = {}
    diverse = []
    random.shuffle(candidates)
    for m, o in candidates:
        key = (m.subtitle or "")[:40]
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= 3:
            diverse.append((m, o))
    batch = diverse[:per_pass]

    logger.info(f"Pass {pass_num}: {len(batch)} markets to forecast (from {len(candidates)} available)")
    if not batch:
        logger.info("No more un-forecasted markets available")
        return []

    results = []
    for i, (market, outcome) in enumerate(batch):
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

        parsed = rule_parser.parse(market.title, market.rules_text)
        bundle = await evidence_retriever.gather(market.title, parsed.entity, market.category)

        forecast_output = await forecaster.forecast(
            title=market.title,
            rules_text=market.rules_text,
            parsed_rules=parsed.model_dump(),
            evidence_snippets=bundle.top_snippets(8),
            market_price=market_price,
            time_to_close_hours=time_to_close / 3600 if time_to_close else None,
        )

        forecast = Forecast(
            market_id=market.id, ts=datetime.now(tz=timezone.utc),
            model_name="llama3.1:8b",
            raw_probability=Decimal(str(round(forecast_output.raw_probability, 6))),
            confidence=Decimal(str(round(forecast_output.confidence, 6))),
            abstain_flag=forecast_output.abstain,
            reasoning_summary=(forecast_output.reasoning_summary or "")[:500],
        )
        session.add(forecast)
        session.flush()

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

        cal = calibrator.predict(features, market_price=market_price)
        session.add(CalibratedForecast(
            forecast_id=forecast.id, calibrator_version=calibrator.version,
            calibrated_probability=Decimal(str(round(cal.calibrated_probability, 6))),
            predicted_edge_bps=cal.predicted_edge_bps,
            uncertainty_low=Decimal(str(round(cal.uncertainty_low, 6))),
            uncertainty_high=Decimal(str(round(cal.uncertainty_high, 6))),
        ))

        label = outcome.resolved_label
        prob = forecast_output.raw_probability
        brier = (prob - label) ** 2
        eps = 1e-7
        ll = -(label * math.log(prob + eps) + (1 - label) * math.log(1 - prob + eps))

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

        error = abs(prob - label)
        error_bucket = None
        if error >= 0.15:
            niche = classify_market(market.title)
            if niche == "skip":
                error_bucket = "bad_calibration"
            elif parsed.ambiguity_score > 0.3:
                error_bucket = "ambiguous_should_have_abstained"
            elif error > 0.4:
                error_bucket = "missed_official_source"
            elif error > 0.25:
                error_bucket = "bad_calibration"
            else:
                error_bucket = "stale_evidence"

        session.add(Postmortem(
            forecast_id=forecast.id, market_id=market.id,
            resolved_label=label,
            brier=Decimal(str(round(brier, 6))),
            log_loss=Decimal(str(round(ll, 6))),
            trading_pnl=Decimal(str(hyp_pnl)) if hyp_pnl is not None else None,
            error_bucket=error_bucket,
            human_reviewed=False, training_weight=Decimal("1.0"),
        ))

        niche = classify_market(market.title)
        results.append({
            "title": market.title[:60],
            "niche": niche,
            "market_price": market_price,
            "forecast": round(prob, 3),
            "actual": label,
            "brier": round(brier, 4),
            "would_trade": decision.should_trade,
            "trade_side": decision.side if decision.should_trade else None,
            "hyp_pnl": hyp_pnl,
            "error_bucket": error_bucket,
        })

        ok = "OK" if (prob > 0.5 and label == 1) or (prob < 0.5 and label == 0) else "XX"
        pnl_str = f"  ${hyp_pnl:+.2f}" if hyp_pnl is not None else ""
        logger.info(f"  {ok} [{niche:5s}] {market.title[:45]:45s} p={prob:.2f} act={label} B={brier:.3f}{pnl_str}")

        if (i + 1) % 10 == 0:
            session.commit()

    session.commit()
    await forecaster.close()
    await evidence_retriever.close()
    return results


def retrain(session):
    from training.trainer import CalibrationTrainer
    trainer = CalibrationTrainer(session)
    features_df, labels = trainer.build_training_dataset()
    if len(features_df) < 15:
        return None, {}
    Path("data/models").mkdir(parents=True, exist_ok=True)
    result = trainer.retrain(save_path="data/models/calibrator_latest.lgb")
    mistakes = trainer.get_mistake_summary()
    return result, mistakes


def print_niche_report(all_results: list[dict]):
    if not all_results:
        return

    print("\n" + "=" * 75)
    print("           CUMULATIVE BACKTEST REPORT — BY NICHE")
    print("=" * 75)

    # Overall
    n = len(all_results)
    correct = sum(1 for r in all_results if (r["forecast"] > 0.5 and r["actual"] == 1) or (r["forecast"] < 0.5 and r["actual"] == 0))
    trades = [r for r in all_results if r["hyp_pnl"] is not None]
    total_pnl = sum(r["hyp_pnl"] for r in trades)
    avg_brier = sum(r["brier"] for r in all_results) / n
    mkt_brier = [((r["market_price"] or 0.5) - r["actual"]) ** 2 for r in all_results]
    avg_mkt = sum(mkt_brier) / len(mkt_brier)

    print(f"\n  OVERALL: {n} markets | accuracy={correct}/{n} ({correct/n:.0%}) | Brier={avg_brier:.4f} vs market={avg_mkt:.4f} | PnL=${total_pnl:+.2f}")

    # By niche
    niches = {}
    for r in all_results:
        niche = r["niche"]
        if niche not in niches:
            niches[niche] = []
        niches[niche].append(r)

    print(f"\n  {'Niche':<14s} {'N':>4s} {'Acc':>6s} {'Brier':>7s} {'MktBri':>7s} {'Edge':>7s} {'Trades':>6s} {'W/L':>7s} {'PnL':>8s}")
    print("  " + "-" * 70)

    for niche in ["crypto", "geopolitics", "tech", "other", "skip"]:
        if niche not in niches:
            continue
        rs = niches[niche]
        nn = len(rs)
        nc = sum(1 for r in rs if (r["forecast"] > 0.5 and r["actual"] == 1) or (r["forecast"] < 0.5 and r["actual"] == 0))
        nb = sum(r["brier"] for r in rs) / nn
        nm = sum(((r["market_price"] or 0.5) - r["actual"]) ** 2 for r in rs) / nn
        nt = [r for r in rs if r["hyp_pnl"] is not None]
        npnl = sum(r["hyp_pnl"] for r in nt)
        nw = sum(1 for r in nt if r["hyp_pnl"] > 0)
        nl = sum(1 for r in nt if r["hyp_pnl"] < 0)
        edge = nm - nb

        edge_str = f"{edge:+.4f}" if edge > 0 else f"{edge:+.4f}"
        tag = " **" if edge > 0 else ""

        print(f"  {niche:<14s} {nn:4d} {nc/nn:5.0%}  {nb:7.4f} {nm:7.4f} {edge_str:>7s}{tag} {len(nt):5d}  {nw}W/{nl}L  ${npnl:+7.2f}")

    # Error analysis
    errors = Counter(r["error_bucket"] for r in all_results if r["error_bucket"])
    if errors:
        print(f"\n  Mistake taxonomy (total):")
        for bucket, count in errors.most_common():
            print(f"    {bucket:40s} {count:3d}")

    # Best and worst calls
    sorted_res = sorted(all_results, key=lambda r: r["brier"])
    print(f"\n  Top 5 best calls:")
    for r in sorted_res[:5]:
        print(f"    [{r['niche']:5s}] {r['title'][:50]:50s} B={r['brier']:.3f} p={r['forecast']:.2f} act={r['actual']}")

    worst = sorted(all_results, key=lambda r: r["brier"], reverse=True)
    print(f"\n  Top 5 worst calls:")
    for r in worst[:5]:
        print(f"    [{r['niche']:5s}] {r['title'][:50]:50s} B={r['brier']:.3f} p={r['forecast']:.2f} act={r['actual']}")

    print("\n" + "=" * 75)


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--passes", type=int, default=4, help="Number of backtest passes")
    parser.add_argument("--per-pass", type=int, default=50, help="Markets per pass")
    parser.add_argument("--niche-only", action="store_true", help="Only test crypto/geopolitics/tech")
    args = parser.parse_args()

    setup_logging(log_level="INFO")
    session = get_db()

    all_results = []
    try:
        for p in range(1, args.passes + 1):
            print(f"\n{'='*75}")
            print(f"  PASS {p}/{args.passes}")
            print(f"{'='*75}")

            results = await run_one_pass(session, p, per_pass=args.per_pass, niche_only=args.niche_only)
            all_results.extend(results)

            if not results:
                logger.info("No more markets — stopping early")
                break

            # Retrain calibrator after each pass
            retrain_out, mistakes = retrain(session)
            if retrain_out:
                logger.info(f"Calibrator retrained: {retrain_out.get('version')} "
                           f"({retrain_out.get('n_samples')} samples, "
                           f"Brier={retrain_out.get('brier_score', 0):.4f})")

        # Final niche-filtered report
        print_niche_report(all_results)

    finally:
        session.close()


if __name__ == "__main__":
    asyncio.run(main())
