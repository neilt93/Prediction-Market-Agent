"""Main entry point to run the full trading simulation pipeline.

Usage:
    uv run python run_simulation.py [--mode discover|forecast|simulate|backfill|postmortem|retrain|full]
"""
from __future__ import annotations

import asyncio
import argparse
import sys
from pathlib import Path

import structlog

# Add package paths
for pkg in ["shared", "schemas", "market_ingest", "rules", "forecasting", "calibration", "execution", "training"]:
    sys.path.insert(0, str(Path(__file__).parent / "packages" / pkg / "src"))
sys.path.insert(0, str(Path(__file__).parent / "apps" / "api" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "apps" / "worker" / "src"))

from shared.config import BaseAppSettings
from shared.logging import setup_logging
from shared.db import create_sync_engine, create_sync_session_factory

logger = structlog.get_logger()


def get_db_session():
    settings = BaseAppSettings()
    engine = create_sync_engine(settings.database_url_sync)
    factory = create_sync_session_factory(engine)
    return factory()


async def run_discover():
    """Discover markets from Kalshi and Polymarket."""
    from market_ingest.clients.kalshi.client import KalshiClient
    from market_ingest.clients.kalshi.config import KalshiEnvironment
    from market_ingest.clients.polymarket.gamma_client import GammaClient
    from market_ingest.orchestration.discovery import MarketDiscoverer

    session = get_db_session()
    kalshi = KalshiClient(env=KalshiEnvironment.DEMO)
    gamma = GammaClient()

    try:
        discoverer = MarketDiscoverer(kalshi, gamma, session)
        # Polymarket public data works without auth
        poly_count = await discoverer.discover_polymarket()
        logger.info("discovery_complete", polymarket=poly_count)

        # Kalshi requires API keys - try if configured
        try:
            kalshi_count = await discoverer.discover_kalshi()
            logger.info("kalshi_discovery", count=kalshi_count)
        except Exception as e:
            logger.warning("kalshi_discovery_skipped", reason=str(e))
    finally:
        await kalshi.close()
        await gamma.close()
        session.close()


async def run_snapshot():
    """Snapshot order books for active markets."""
    from market_ingest.clients.kalshi.client import KalshiClient
    from market_ingest.clients.kalshi.config import KalshiEnvironment
    from market_ingest.clients.polymarket.clob_client import ClobClient
    from market_ingest.clients.polymarket.gamma_client import GammaClient
    from market_ingest.orchestration.snapshotter import MarketSnapshotter

    session = get_db_session()
    kalshi = KalshiClient(env=KalshiEnvironment.DEMO)
    clob = ClobClient()
    gamma = GammaClient()

    try:
        snapshotter = MarketSnapshotter(kalshi, clob, gamma, session)
        count = await snapshotter.snapshot_all()
        logger.info("snapshot_complete", count=count)
    finally:
        await kalshi.close()
        await clob.close()
        await gamma.close()
        session.close()


async def run_forecast():
    """Run forecaster on all active markets."""
    from forecasting.forecaster import Forecaster
    from calibration.calibrator import Calibrator
    from execution.policy import ExecutionPolicy
    from execution.simulator import TradingSimulator

    session = get_db_session()
    forecaster = Forecaster()
    calibrator = Calibrator(model_path="data/models/calibrator_latest.lgb")
    policy = ExecutionPolicy()

    try:
        simulator = TradingSimulator(session, forecaster, calibrator, policy)
        stats = await simulator.run_cycle()
        logger.info("forecast_cycle_complete", **stats)
    finally:
        await forecaster.close()
        session.close()


async def run_simulate():
    """Full simulation: discover -> snapshot -> forecast -> trade."""
    logger.info("=== Starting full simulation cycle ===")

    logger.info("Step 1: Discovering markets...")
    await run_discover()

    logger.info("Step 2: Taking snapshots...")
    await run_snapshot()

    logger.info("Step 3: Forecasting and trading...")
    await run_forecast()

    logger.info("=== Simulation cycle complete ===")


async def run_backfill():
    """Backfill historical data."""
    from market_ingest.clients.kalshi.client import KalshiClient
    from market_ingest.clients.kalshi.config import KalshiEnvironment
    from market_ingest.clients.polymarket.gamma_client import GammaClient
    from market_ingest.clients.polymarket.clob_client import ClobClient
    from market_ingest.orchestration.backfiller import HistoricalBackfiller

    session = get_db_session()
    kalshi = KalshiClient(env=KalshiEnvironment.DEMO)
    gamma = GammaClient()
    clob = ClobClient()

    try:
        backfiller = HistoricalBackfiller(kalshi, gamma, clob, session)
        p_count = await backfiller.backfill_polymarket(max_markets=200)
        logger.info("backfill_complete", polymarket=p_count)
    finally:
        await kalshi.close()
        await gamma.close()
        await clob.close()
        session.close()


def run_postmortem():
    """Generate postmortems for resolved markets."""
    from forecasting.forecaster import Forecaster
    from calibration.calibrator import Calibrator
    from execution.simulator import TradingSimulator

    session = get_db_session()
    forecaster = Forecaster()
    calibrator = Calibrator()

    try:
        simulator = TradingSimulator(session, forecaster, calibrator)
        count = simulator.generate_postmortems()
        logger.info("postmortems_done", count=count)
    finally:
        session.close()


def run_retrain():
    """Retrain the calibrator on accumulated data."""
    from training.trainer import CalibrationTrainer

    session = get_db_session()
    try:
        trainer = CalibrationTrainer(session)
        result = trainer.retrain()
        logger.info("retrain_complete", **result)

        mistakes = trainer.get_mistake_summary()
        if mistakes:
            logger.info("mistake_summary", **mistakes)
    finally:
        session.close()


async def run_full():
    """Run the complete pipeline: simulate -> postmortem -> retrain."""
    await run_simulate()
    run_postmortem()
    run_retrain()


def main():
    parser = argparse.ArgumentParser(description="Prediction Market Agent")
    parser.add_argument(
        "--mode",
        choices=["discover", "snapshot", "forecast", "simulate", "backfill", "postmortem", "retrain", "full"],
        default="simulate",
        help="Mode to run (default: simulate)",
    )
    args = parser.parse_args()

    setup_logging(log_level="INFO")

    mode_map = {
        "discover": run_discover,
        "snapshot": run_snapshot,
        "forecast": run_forecast,
        "simulate": run_simulate,
        "backfill": run_backfill,
        "full": run_full,
    }

    sync_modes = {
        "postmortem": run_postmortem,
        "retrain": run_retrain,
    }

    if args.mode in sync_modes:
        sync_modes[args.mode]()
    else:
        asyncio.run(mode_map[args.mode]())


if __name__ == "__main__":
    main()
