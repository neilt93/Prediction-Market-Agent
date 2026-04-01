from __future__ import annotations

import structlog
from sqlalchemy import text
from sqlalchemy.orm import Session

from market_ingest.clients.kalshi.client import KalshiClient
from market_ingest.clients.polymarket.gamma_client import GammaClient
from market_ingest.mappers.kalshi_mapper import KalshiMapper
from market_ingest.mappers.polymarket_mapper import PolymarketMapper
from schemas.models.market import Market

logger = structlog.get_logger()


class MarketDiscoverer:
    """Discovers and upserts new markets from Kalshi and Polymarket."""

    def __init__(
        self,
        kalshi_client: KalshiClient,
        gamma_client: GammaClient,
        db_session: Session,
    ) -> None:
        self.kalshi = kalshi_client
        self.gamma = gamma_client
        self.db = db_session
        self.kalshi_mapper = KalshiMapper()
        self.poly_mapper = PolymarketMapper()

    async def discover_kalshi(self) -> int:
        """Fetch active Kalshi events with markets and upsert."""
        count = 0
        async for event in self.kalshi.get_events(status="open"):
            if not event.markets:
                continue
            for km in event.markets:
                market_data = self.kalshi_mapper.market_to_db(km, event)
                self._upsert_market(market_data)
                count += 1
        self.db.commit()
        logger.info("kalshi_discovery_complete", count=count)
        return count

    async def discover_polymarket(self) -> int:
        """Fetch active Polymarket events and upsert markets."""
        count = 0
        events = await self.gamma.list_all_active_events()
        for event in events:
            if not event.markets:
                continue
            for pm in event.markets:
                if not pm.enable_order_book:
                    continue
                market_data = self.poly_mapper.market_to_db(pm, event)
                self._upsert_market(market_data)
                count += 1
        self.db.commit()
        logger.info("polymarket_discovery_complete", count=count)
        return count

    def _upsert_market(self, data: dict) -> None:
        """Insert or update market by platform + platform_market_id."""
        existing = (
            self.db.query(Market)
            .filter(
                Market.platform == data["platform"],
                Market.platform_market_id == data["platform_market_id"],
            )
            .first()
        )
        if existing:
            for key, value in data.items():
                if value is not None:
                    setattr(existing, key, value)
        else:
            self.db.add(Market(**data))
