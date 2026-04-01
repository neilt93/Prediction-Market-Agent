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
        # Just one page of events
        events = await gamma.list_events(active=True, limit=20, offset=0)
        logger.info(f"Fetched {len(events)} events from Polymarket")

        for event in events:
            if not event.markets:
                continue
            for pm in event.markets:
                if not pm.enable_order_book:
                    continue
                market_data = mapper.market_to_db(pm, event)

                existing = (
                    session.query(Market)
                    .filter(
                        Market.platform == market_data["platform"],
                        Market.platform_market_id == market_data["platform_market_id"],
                    )
                    .first()
                )
                if not existing:
                    session.add(Market(**market_data))
                    count += 1
        session.commit()
        logger.info(f"Discovered {count} new markets")
    finally:
        await gamma.close()

    return count


async def quick_snapshot(session):
    """Snapshot order books for discovered markets."""
    from market_ingest.clients.polymarket.clob_client import ClobClient
    from market_ingest.clients.polymarket.gamma_client import GammaClient
    from market_ingest.mappers.polymarket_mapper import PolymarketMapper
    from schemas.models.market import Market, MarketSnapshot

    clob = ClobClient()
    gamma = GammaClient()
    mapper = PolymarketMapper()
    count = 0

    try:
        markets = session.query(Market).filter(Market.status == "open").limit(20).all()
        logger.info(f"Snapshotting {len(markets)} markets")

        for market in markets:
            try:
                gamma_market = await gamma.get_market(market.platform_market_id)
                ob = None
                if gamma_market.clob_token_ids:
                    try:
                        ob = await clob.get_orderbook(gamma_market.clob_token_ids[0])
                    except Exception:
                        pass

                snapshot_data = mapper.to_snapshot(gamma_market, ob)
                snapshot_data["market_id"] = market.id
                session.add(MarketSnapshot(**snapshot_data))
                count += 1
            except Exception as e:
                logger.warning(f"Snapshot failed for {market.title[:40]}: {e}")

        session.commit()
        logger.info(f"Captured {count} snapshots")
    finally:
        await clob.close()
        await gamma.close()

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
        min_confidence=0.5,
        max_position_per_market=3,
        daily_max_loss=50.0,
    ))

    # Try to connect to a local LLM, fall back to mock forecasts
    forecaster = Forecaster(
        api_url="http://localhost:11434/v1",
        model="llama3.1:8b",
    )

    markets = (
        session.query(Market)
        .filter(Market.status == "open")
        .limit(10)
        .all()
    )

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

        # Try LLM forecast, fall back to mock
        try:
            forecast_output = await forecaster.forecast(
                title=market.title,
                rules_text=market.rules_text,
                parsed_rules=parsed.model_dump(),
                market_price=market_price,
                time_to_close_hours=time_to_close / 3600 if time_to_close else None,
            )
        except Exception:
            # Mock forecast if no LLM available
            import random
            forecast_output = ForecastOutput(
                raw_probability=max(0.05, min(0.95, market_price + random.gauss(0, 0.1))),
                confidence=random.uniform(0.4, 0.9),
                abstain=False,
                reasoning_summary="Mock forecast (no LLM available)",
            )

        if forecast_output.abstain:
            stats["abstains"] += 1
            continue

        # Store forecast
        forecast = Forecast(
            market_id=market.id,
            ts=datetime.now(tz=timezone.utc),
            model_name="mock-v1",
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

            position = Position(
                market_id=market.id,
                net_qty=decision.quantity if decision.side == "buy_yes" else -decision.quantity,
                avg_cost=Decimal(str(decision.limit_price)),
                mark_price=Decimal(str(market_price)),
                realized_pnl=Decimal("0"),
            )
            session.merge(position)
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
