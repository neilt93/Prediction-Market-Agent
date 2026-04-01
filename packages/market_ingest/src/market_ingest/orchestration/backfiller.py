from __future__ import annotations

from datetime import datetime, timezone

import structlog
from sqlalchemy.orm import Session

from market_ingest.clients.kalshi.client import KalshiClient
from market_ingest.clients.polymarket.clob_client import ClobClient
from market_ingest.clients.polymarket.gamma_client import GammaClient
from market_ingest.mappers.kalshi_mapper import KalshiMapper
from market_ingest.mappers.polymarket_mapper import PolymarketMapper
from schemas.models.market import Market, MarketOutcome, MarketSnapshot

logger = structlog.get_logger()


class HistoricalBackfiller:
    """Backfills historical market data for backtesting."""

    def __init__(
        self,
        kalshi_client: KalshiClient,
        gamma_client: GammaClient,
        clob_client: ClobClient,
        db_session: Session,
    ) -> None:
        self.kalshi = kalshi_client
        self.gamma = gamma_client
        self.clob = clob_client
        self.db = db_session
        self.kalshi_mapper = KalshiMapper()
        self.poly_mapper = PolymarketMapper()

    async def backfill_kalshi(self, max_markets: int = 500) -> int:
        """Backfill historical Kalshi markets."""
        count = 0
        async for km in self.kalshi.get_historical_markets(limit=200):
            if count >= max_markets:
                break

            existing = (
                self.db.query(Market)
                .filter(
                    Market.platform == "kalshi",
                    Market.platform_market_id == km.ticker,
                )
                .first()
            )
            if existing:
                continue

            market_data = self.kalshi_mapper.market_to_db(km)
            market_data["status"] = "resolved"
            market = Market(**market_data)
            self.db.add(market)
            self.db.flush()

            outcome_data = self.kalshi_mapper.market_to_outcome(km)
            if outcome_data:
                outcome_data["market_id"] = market.id
                self.db.add(MarketOutcome(**outcome_data))

            count += 1
            if count % 50 == 0:
                self.db.commit()
                logger.info("kalshi_backfill_progress", count=count)

        self.db.commit()
        logger.info("kalshi_backfill_complete", count=count)
        return count

    async def backfill_polymarket(self, max_markets: int = 500) -> int:
        """Backfill historical Polymarket markets."""
        count = 0
        offset = 0

        while count < max_markets:
            events = await self.gamma.list_events(
                closed=True, archived=True, limit=100, offset=offset
            )
            if not events:
                break

            for event in events:
                if not event.markets:
                    continue
                for pm in event.markets:
                    if count >= max_markets:
                        break
                    cid = pm.condition_id or pm.id
                    existing = (
                        self.db.query(Market)
                        .filter(
                            Market.platform == "polymarket",
                            Market.platform_market_id == cid,
                        )
                        .first()
                    )
                    if existing:
                        continue

                    market_data = self.poly_mapper.market_to_db(pm, event)
                    market_data["status"] = "resolved"
                    market = Market(**market_data)
                    self.db.add(market)
                    self.db.flush()

                    outcome_data = self.poly_mapper.market_to_outcome(pm)
                    if outcome_data:
                        outcome_data["market_id"] = market.id
                        self.db.add(MarketOutcome(**outcome_data))

                    # Try to get price history for synthetic snapshots
                    if pm.clob_token_ids:
                        try:
                            history = await self.clob.get_prices_history(
                                pm.clob_token_ids[0], interval="max", fidelity=60
                            )
                            for point in history:
                                snap = MarketSnapshot(
                                    market_id=market.id,
                                    ts=datetime.fromtimestamp(point.t, tz=timezone.utc),
                                    mid_yes=point.p,
                                    last_yes=point.p,
                                )
                                self.db.add(snap)
                        except Exception as e:
                            logger.warning(
                                "poly_price_history_failed",
                                market_id=cid,
                                error=str(e),
                            )

                    count += 1

            if count % 50 == 0:
                self.db.commit()
                logger.info("polymarket_backfill_progress", count=count)

            if len(events) < 100:
                break
            offset += 100

        self.db.commit()
        logger.info("polymarket_backfill_complete", count=count)
        return count
