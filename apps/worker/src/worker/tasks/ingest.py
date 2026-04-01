from __future__ import annotations

import asyncio

import structlog

from worker.main import celery_app
from worker.config import WorkerSettings
from shared.db import create_sync_engine, create_sync_session_factory
from market_ingest.clients.kalshi.client import KalshiClient
from market_ingest.clients.kalshi.config import KalshiEnvironment
from market_ingest.clients.polymarket.gamma_client import GammaClient
from market_ingest.clients.polymarket.clob_client import ClobClient
from market_ingest.orchestration.discovery import MarketDiscoverer
from market_ingest.orchestration.snapshotter import MarketSnapshotter
from market_ingest.orchestration.resolution_detector import ResolutionDetector

logger = structlog.get_logger()


def _get_db_session():
    settings = WorkerSettings()
    engine = create_sync_engine(settings.database_url_sync)
    factory = create_sync_session_factory(engine)
    return factory()


def _get_kalshi_client() -> KalshiClient:
    settings = WorkerSettings()
    env = KalshiEnvironment(settings.kalshi_env)
    return KalshiClient(
        env=env,
        api_key_id=settings.kalshi_api_key_id,
        private_key_path=settings.kalshi_private_key_path,
    )


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def discover_markets(self):
    """Discover new markets from Kalshi and Polymarket."""
    logger.info("discover_markets.start")
    session = _get_db_session()
    try:
        kalshi = _get_kalshi_client()
        gamma = GammaClient()

        discoverer = MarketDiscoverer(kalshi, gamma, session)
        loop = asyncio.new_event_loop()
        try:
            kalshi_count = loop.run_until_complete(discoverer.discover_kalshi())
            poly_count = loop.run_until_complete(discoverer.discover_polymarket())
        finally:
            loop.run_until_complete(kalshi.close())
            loop.run_until_complete(gamma.close())
            loop.close()

        logger.info(
            "discover_markets.complete",
            kalshi=kalshi_count,
            polymarket=poly_count,
        )
    except Exception as exc:
        session.rollback()
        logger.error("discover_markets.failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        session.close()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def snapshot_markets(self):
    """Take order book snapshots for all active markets."""
    logger.info("snapshot_markets.start")
    session = _get_db_session()
    try:
        kalshi = _get_kalshi_client()
        clob = ClobClient()
        gamma = GammaClient()

        snapshotter = MarketSnapshotter(kalshi, clob, gamma, session)
        loop = asyncio.new_event_loop()
        try:
            count = loop.run_until_complete(snapshotter.snapshot_all())
        finally:
            loop.run_until_complete(kalshi.close())
            loop.run_until_complete(clob.close())
            loop.run_until_complete(gamma.close())
            loop.close()

        logger.info("snapshot_markets.complete", count=count)
    except Exception as exc:
        session.rollback()
        logger.error("snapshot_markets.failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        session.close()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def detect_resolutions(self):
    """Detect newly resolved markets."""
    logger.info("detect_resolutions.start")
    session = _get_db_session()
    try:
        kalshi = _get_kalshi_client()
        gamma = GammaClient()

        detector = ResolutionDetector(kalshi, gamma, session)
        loop = asyncio.new_event_loop()
        try:
            count = loop.run_until_complete(detector.detect_all())
        finally:
            loop.run_until_complete(kalshi.close())
            loop.run_until_complete(gamma.close())
            loop.close()

        logger.info("detect_resolutions.complete", count=count)
    except Exception as exc:
        session.rollback()
        logger.error("detect_resolutions.failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        session.close()


@celery_app.task(bind=True, max_retries=1, default_retry_delay=300)
def backfill_historical(self):
    """Backfill historical market data for backtesting."""
    logger.info("backfill_historical.start")
    session = _get_db_session()
    try:
        kalshi = _get_kalshi_client()
        gamma = GammaClient()
        clob = ClobClient()

        from market_ingest.orchestration.backfiller import HistoricalBackfiller

        backfiller = HistoricalBackfiller(kalshi, gamma, clob, session)
        loop = asyncio.new_event_loop()
        try:
            k_count = loop.run_until_complete(backfiller.backfill_kalshi())
            p_count = loop.run_until_complete(backfiller.backfill_polymarket())
        finally:
            loop.run_until_complete(kalshi.close())
            loop.run_until_complete(gamma.close())
            loop.run_until_complete(clob.close())
            loop.close()

        logger.info(
            "backfill_historical.complete",
            kalshi=k_count,
            polymarket=p_count,
        )
    except Exception as exc:
        session.rollback()
        logger.error("backfill_historical.failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        session.close()
