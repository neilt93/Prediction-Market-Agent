from __future__ import annotations

import structlog
from sqlalchemy.orm import Session

from market_ingest.clients.kalshi.client import KalshiClient
from market_ingest.clients.polymarket.gamma_client import GammaClient
from market_ingest.mappers.kalshi_mapper import KalshiMapper
from market_ingest.mappers.polymarket_mapper import PolymarketMapper
from schemas.models.market import Market, MarketOutcome

logger = structlog.get_logger()


class ResolutionDetector:
    """Detects markets that have newly resolved."""

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

    async def detect_all(self) -> int:
        count = 0
        count += await self.detect_kalshi_resolutions()
        count += await self.detect_polymarket_resolutions()
        self.db.commit()
        return count

    async def detect_kalshi_resolutions(self) -> int:
        open_markets = (
            self.db.query(Market)
            .filter(Market.platform == "kalshi", Market.status.in_(["open", "closed"]))
            .all()
        )
        if not open_markets:
            return 0

        ticker_map = {m.platform_market_id: m for m in open_markets}
        count = 0

        async for event in self.kalshi.get_events(status="settled"):
            if not event.markets:
                continue
            for km in event.markets:
                if km.ticker not in ticker_map:
                    continue
                market = ticker_map[km.ticker]
                outcome_data = self.kalshi_mapper.market_to_outcome(km)
                if outcome_data is None:
                    continue

                market.status = "resolved"
                existing = (
                    self.db.query(MarketOutcome)
                    .filter(MarketOutcome.market_id == market.id)
                    .first()
                )
                if not existing:
                    outcome_data["market_id"] = market.id
                    self.db.add(MarketOutcome(**outcome_data))
                    count += 1

        logger.info("kalshi_resolution_detection", count=count)
        return count

    async def detect_polymarket_resolutions(self) -> int:
        open_markets = (
            self.db.query(Market)
            .filter(Market.platform == "polymarket", Market.status.in_(["open", "closed"]))
            .all()
        )
        if not open_markets:
            return 0

        cid_map = {m.platform_market_id: m for m in open_markets}
        count = 0

        offset = 0
        while True:
            resolved_markets = await self.gamma.list_markets(
                closed=True, limit=100, offset=offset
            )
            if not resolved_markets:
                break

            for pm in resolved_markets:
                cid = pm.condition_id or pm.id
                if cid not in cid_map:
                    continue
                if not (pm.closed or pm.archived):
                    continue

                market = cid_map[cid]
                outcome_data = self.poly_mapper.market_to_outcome(pm)
                if outcome_data is None:
                    continue

                market.status = "resolved"
                existing = (
                    self.db.query(MarketOutcome)
                    .filter(MarketOutcome.market_id == market.id)
                    .first()
                )
                if not existing:
                    outcome_data["market_id"] = market.id
                    self.db.add(MarketOutcome(**outcome_data))
                    count += 1

            if len(resolved_markets) < 100:
                break
            offset += 100

        logger.info("polymarket_resolution_detection", count=count)
        return count
