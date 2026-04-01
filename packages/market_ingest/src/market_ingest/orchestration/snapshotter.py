from __future__ import annotations

import structlog
from sqlalchemy.orm import Session

from market_ingest.clients.kalshi.client import KalshiClient
from market_ingest.clients.polymarket.clob_client import ClobClient
from market_ingest.clients.polymarket.gamma_client import GammaClient
from market_ingest.mappers.kalshi_mapper import KalshiMapper
from market_ingest.mappers.polymarket_mapper import PolymarketMapper
from schemas.models.market import Market, MarketSnapshot

logger = structlog.get_logger()


class MarketSnapshotter:
    """Collects order book snapshots for all tracked active markets."""

    def __init__(
        self,
        kalshi_client: KalshiClient,
        clob_client: ClobClient,
        gamma_client: GammaClient,
        db_session: Session,
    ) -> None:
        self.kalshi = kalshi_client
        self.clob = clob_client
        self.gamma = gamma_client
        self.db = db_session
        self.kalshi_mapper = KalshiMapper()
        self.poly_mapper = PolymarketMapper()

    async def snapshot_all(self) -> int:
        """Snapshot all active markets."""
        markets = self.db.query(Market).filter(Market.status == "open").all()
        kalshi_markets = [m for m in markets if m.platform == "kalshi"]
        poly_markets = [m for m in markets if m.platform == "polymarket"]

        count = 0
        count += await self._snapshot_kalshi(kalshi_markets)
        count += await self._snapshot_polymarket(poly_markets)
        self.db.commit()
        logger.info("snapshot_complete", count=count)
        return count

    async def _snapshot_kalshi(self, markets: list[Market]) -> int:
        if not markets:
            return 0
        tickers = [m.platform_market_id for m in markets]
        orderbooks = await self.kalshi.get_orderbooks_batch(tickers)
        ob_map = {ob.ticker: ob for ob in orderbooks}

        count = 0
        for market in markets:
            ob = ob_map.get(market.platform_market_id)
            if not ob:
                continue
            km_data = await self.kalshi.get_market(market.platform_market_id)
            snapshot_data = self.kalshi_mapper.orderbook_to_snapshot(ob, km_data)
            snapshot_data["market_id"] = market.id
            self.db.add(MarketSnapshot(**snapshot_data))
            count += 1
        return count

    async def _snapshot_polymarket(self, markets: list[Market]) -> int:
        if not markets:
            return 0
        count = 0
        for market in markets:
            try:
                gamma_market = await self.gamma.get_market(market.platform_market_id)
                ob = None
                if gamma_market.clob_token_ids:
                    ob = await self.clob.get_orderbook(gamma_market.clob_token_ids[0])
                snapshot_data = self.poly_mapper.to_snapshot(gamma_market, ob)
                snapshot_data["market_id"] = market.id
                self.db.add(MarketSnapshot(**snapshot_data))
                count += 1
            except Exception as e:
                logger.warning(
                    "poly_snapshot_failed",
                    market_id=str(market.id),
                    error=str(e),
                )
        return count
