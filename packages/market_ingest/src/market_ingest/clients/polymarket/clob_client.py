from __future__ import annotations

from typing import Any

from market_ingest.clients.base import BaseHttpClient
from market_ingest.clients.polymarket.config import CLOB_BASE_URL
from market_ingest.clients.polymarket.models import PolyClobOrderbook, PolyPricePoint


class ClobClient(BaseHttpClient):
    """Client for Polymarket CLOB API (order books and pricing)."""

    def __init__(self) -> None:
        super().__init__(base_url=CLOB_BASE_URL, rate_limit=80.0)

    async def get_orderbook(self, token_id: str) -> PolyClobOrderbook:
        data = await self.get("/book", params={"token_id": token_id})
        return PolyClobOrderbook.model_validate(data)

    async def get_orderbooks_batch(self, token_ids: list[str]) -> list[PolyClobOrderbook]:
        data = await self.post("/books", json=token_ids)
        if isinstance(data, list):
            return [PolyClobOrderbook.model_validate(ob) for ob in data]
        return [PolyClobOrderbook.model_validate(ob) for ob in data.get("data", [])]

    async def get_midpoint(self, token_id: str) -> float:
        data = await self.get("/midpoint", params={"token_id": token_id})
        return float(data.get("mid", 0))

    async def get_spread(self, token_id: str) -> float:
        data = await self.get("/spread", params={"token_id": token_id})
        return float(data.get("spread", 0))

    async def get_price(self, token_id: str, side: str = "BUY") -> float:
        data = await self.get("/price", params={"token_id": token_id, "side": side})
        return float(data.get("price", 0))

    async def get_last_trade_price(self, token_id: str) -> float:
        data = await self.get("/last-trade-price", params={"token_id": token_id})
        return float(data.get("price", 0))

    async def get_prices_history(
        self,
        token_id: str,
        interval: str = "max",
        fidelity: int = 60,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[PolyPricePoint]:
        params: dict[str, Any] = {
            "market": token_id,
            "interval": interval,
            "fidelity": fidelity,
        }
        if start_ts:
            params["startTs"] = start_ts
        if end_ts:
            params["endTs"] = end_ts

        data = await self.get("/prices-history", params=params)
        history = data.get("history", data) if isinstance(data, dict) else data
        if isinstance(history, list):
            return [PolyPricePoint.model_validate(p) for p in history]
        return []
