from __future__ import annotations

from typing import Any

from market_ingest.clients.base import BaseHttpClient
from market_ingest.clients.polymarket.config import DATA_BASE_URL
from market_ingest.clients.polymarket.models import PolyTrade


class DataClient(BaseHttpClient):
    """Client for Polymarket Data API (trades and activity)."""

    def __init__(self) -> None:
        super().__init__(base_url=DATA_BASE_URL, rate_limit=15.0)

    async def get_trades(
        self,
        market: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[PolyTrade]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if market:
            params["market"] = market

        data = await self.get("/trades", params=params)
        trades = data if isinstance(data, list) else data.get("data", [])
        return [PolyTrade.model_validate(t) for t in trades]
