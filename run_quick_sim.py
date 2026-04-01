"""Quick simulation script - discovers a small batch, snapshots, and forecasts."""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

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


async def quick_discover(session):
    """Discover just the top 50 active Polymarket markets."""
    from market_ingest.clients.polymarket.gamma_client import GammaClient
    from market_ingest.mappers.polymarket_mapper import PolymarketMapper
    from schemas.models.market import Market

    gamma = GammaClient()
    mapper = PolymarketMapper()
    count = 0

    try:
        # Fetch top events by volume (most traded = most liquid individual markets)
        events = await gamma.list_events(active=True, limit=100, offset=0, order="volume", ascending=False)
        logger.info(f"Fetched {len(events)} events from Polymarket")

        for event in events:
            if not event.markets:
                continue
            for pm in event.markets:
                if not pm.enable_order_book:
                    continue
                market_data = mapper.market_to_db(pm, event)
                # Store clob_token_ids for snapshot lookups
                market_data["metadata_json"] = {
                    "clob_token_ids": pm.clob_token_ids,
                    "outcomes": pm.outcomes,
                    "outcome_prices": pm.outcome_prices,
                }

                existing = (
                    session.query(Market)
                    .filter(
                        Market.platform == market_data["platform"],
                        Market.platform_market_id == market_data["platform_market_id"],
                    )
                    .first()
                )
                if existing:
                    existing.metadata_json = market_data["metadata_json"]
                else:
                    session.add(Market(**market_data))
                count += 1
        session.commit()
        logger.info(f"Discovered {count} new markets")
    finally:
        await gamma.close()

    return count


async def quick_snapshot(session):
    """Snapshot order books for discovered markets using CLOB API directly."""
    from market_ingest.clients.polymarket.clob_client import ClobClient
    from schemas.models.market import Market, MarketSnapshot

    clob = ClobClient()
    count = 0

    try:
        # Get ALL open markets with clob token IDs, prioritize binary markets
        all_markets = (
            session.query(Market)
            .filter(
                Market.status == "open",
                Market.metadata_json.isnot(None),
            )
            .all()
        )
        has_tokens = [m for m in all_markets if (m.metadata_json or {}).get("clob_token_ids")]
        # Binary first, then multi
        binary = [m for m in has_tokens if m.market_type == "binary"]
        multi = [m for m in has_tokens if m.market_type != "binary"]
        import random
        random.shuffle(multi)
        markets = binary + multi[:200]
        logger.info(f"Snapshotting {len(markets)} markets")

        for market in markets:
            try:
                # Use metadata_json to get clob token IDs, or skip
                token_ids = (market.metadata_json or {}).get("clob_token_ids")
                if not token_ids:
                    continue
                ob = await clob.get_orderbook(token_ids[0])
                bids = ob.bid_levels
                asks = ob.ask_levels

                best_bid = bids[0][0] if bids else None
                best_ask = asks[0][0] if asks else None
                mid = (best_bid + best_ask) / 2 if best_bid and best_ask else None
                spread_bps = int((best_ask - best_bid) * 10000) if best_bid and best_ask else None
                bid_depth = sum(l[1] for l in bids[:5]) if bids else 0
                ask_depth = sum(l[1] for l in asks[:5]) if asks else 0
                total = bid_depth + ask_depth
                imbalance = bid_depth / total if total > 0 else 0.5

                snap = MarketSnapshot(
                    market_id=market.id,
                    ts=datetime.now(tz=timezone.utc),
                    best_bid_yes=Decimal(str(best_bid)) if best_bid else None,
                    best_ask_yes=Decimal(str(best_ask)) if best_ask else None,
                    mid_yes=Decimal(str(mid)) if mid else None,
                    last_yes=Decimal(str(ob.last_trade_price)) if ob.last_trade_price else None,
                    spread_bps=spread_bps,
                    liquidity_proxy=Decimal(str(total)) if total else None,
                    orderbook_imbalance=Decimal(str(round(imbalance, 6))),
                )
                session.add(snap)
                count += 1
            except Exception as e:
                logger.warning(f"Snapshot failed for {market.title[:40]}: {e}")

        session.commit()
        logger.info(f"Captured {count} snapshots")
    finally:
        await clob.close()

    return count


async def quick_forecast(session):
    """Run forecaster on markets with snapshots (using Claude API or mock)."""
    from forecasting.forecaster import Forecaster, ForecastOutput
    from calibration.calibrator import Calibrator
    from execution.policy import ExecutionPolicy, PolicyConfig
    from rules.parser import RuleParser
    from schemas.models.market import Market, MarketSnapshot
    from schemas.models.forecast import Forecast, ForecastFeature, CalibratedForecast
    from schemas.models.execution import Order, Position

    rule_parser = RuleParser()
    calibrator = Calibrator()
    policy = ExecutionPolicy(PolicyConfig(
        min_edge_bps=300,
        min_confidence=0.3,
        max_spread_bps=15000,  # Relaxed for Polymarket sub-markets (tighten for Kalshi)
        max_position_per_market=3,
        daily_max_loss=50.0,
    ))

    # Try to connect to a local LLM, fall back to mock forecasts
    forecaster = Forecaster(
        api_url="http://localhost:11434/v1",
        model="llama3.1:8b",
    )

    # Forecast markets with snapshots, ordered by tightest spread
    from sqlalchemy import func
    markets = (
        session.query(Market)
        .join(MarketSnapshot, MarketSnapshot.market_id == Market.id)
        .filter(Market.status == "open", MarketSnapshot.spread_bps.isnot(None))
        .group_by(Market.id)
        .order_by(func.min(MarketSnapshot.spread_bps))
        .limit(15)
        .all()
    )
    logger.info(f"Found {len(markets)} markets to forecast (ordered by spread)")

    stats = {"forecasts": 0, "trades": 0, "abstains": 0}

    for market in markets:
        snapshot = (
            session.query(MarketSnapshot)
            .filter(MarketSnapshot.market_id == market.id)
            .order_by(MarketSnapshot.ts.desc())
            .first()
        )

        market_price = float(snapshot.mid_yes) if snapshot and snapshot.mid_yes else 0.5
        spread_bps = snapshot.spread_bps if snapshot else None
        liquidity = float(snapshot.liquidity_proxy) if snapshot and snapshot.liquidity_proxy else None
        time_to_close = snapshot.time_to_close_sec if snapshot else None

        # Parse rules
        parsed = rule_parser.parse(market.title, market.rules_text)

        # Forecast with LLM (falls back gracefully inside Forecaster if it errors)
        forecast_output = await forecaster.forecast(
            title=market.title,
            rules_text=market.rules_text,
            parsed_rules=parsed.model_dump(),
            market_price=market_price,
            time_to_close_hours=time_to_close / 3600 if time_to_close else None,
        )

        if forecast_output.abstain:
            stats["abstains"] += 1
            continue

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
        stats["forecasts"] += 1

        # Calibrate (passthrough since untrained)
        features = {
            "market_price": market_price,
            "raw_probability": forecast_output.raw_probability,
            "llm_confidence": forecast_output.confidence,
            "spread_bps": spread_bps or 0,
            "ambiguity_score": parsed.ambiguity_score,
        }
        cal_output = calibrator.predict(features, market_price=market_price)

        cal_record = CalibratedForecast(
            forecast_id=forecast.id,
            calibrator_version=calibrator.version,
            calibrated_probability=Decimal(str(round(cal_output.calibrated_probability, 6))),
            predicted_edge_bps=cal_output.predicted_edge_bps,
            uncertainty_low=Decimal(str(round(cal_output.uncertainty_low, 6))),
            uncertainty_high=Decimal(str(round(cal_output.uncertainty_high, 6))),
        )
        session.add(cal_record)

        # Execution policy
        decision = policy.evaluate(
            calibrated_probability=cal_output.calibrated_probability,
            market_price=market_price,
            confidence=forecast_output.confidence,
            ambiguity_score=parsed.ambiguity_score,
            spread_bps=spread_bps,
            liquidity=liquidity,
            abstain_flag=forecast_output.abstain,
        )

        if decision.should_trade:
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
            session.add(order)

            existing_pos = session.query(Position).filter(Position.market_id == market.id).first()
            if existing_pos:
                if decision.side == "buy_yes":
                    existing_pos.net_qty += decision.quantity
                else:
                    existing_pos.net_qty -= decision.quantity
                existing_pos.mark_price = Decimal(str(market_price))
            else:
                qty = decision.quantity if decision.side == "buy_yes" else -decision.quantity
                session.add(Position(
                    market_id=market.id,
                    net_qty=qty,
                    avg_cost=Decimal(str(decision.limit_price)),
                    mark_price=Decimal(str(market_price)),
                    realized_pnl=Decimal("0"),
                ))
            stats["trades"] += 1
            logger.info(
                f"TRADE: {decision.side} {market.title[:50]} @ {decision.limit_price:.2f} "
                f"(edge={decision.edge_bps}bps)"
            )
        else:
            logger.info(f"SKIP: {market.title[:50]} - {decision.reason}")

    session.commit()
    await forecaster.close()
    return stats


async def main():
    setup_logging(log_level="INFO")
    session = get_db()

    try:
        logger.info("=== STEP 1: Discovering markets ===")
        n_markets = await quick_discover(session)

        logger.info("=== STEP 2: Taking order book snapshots ===")
        n_snapshots = await quick_snapshot(session)

        logger.info("=== STEP 3: Forecasting and trading ===")
        stats = await quick_forecast(session)

        # Print summary
        from schemas.models.market import Market, MarketSnapshot
        from schemas.models.forecast import Forecast
        from schemas.models.execution import Order

        total_markets = session.query(Market).count()
        total_snapshots = session.query(MarketSnapshot).count()
        total_forecasts = session.query(Forecast).count()
        total_orders = session.query(Order).count()

        logger.info("=" * 60)
        logger.info(f"SIMULATION SUMMARY")
        logger.info(f"  Markets in DB:    {total_markets}")
        logger.info(f"  Snapshots:        {total_snapshots}")
        logger.info(f"  Forecasts:        {total_forecasts}")
        logger.info(f"  Orders:           {total_orders}")
        logger.info(f"  This cycle stats: {stats}")
        logger.info("=" * 60)
    finally:
        session.close()


if __name__ == "__main__":
    asyncio.run(main())
