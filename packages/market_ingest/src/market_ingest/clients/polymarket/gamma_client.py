from __future__ import annotations

from typing import Any

from market_ingest.clients.base import BaseHttpClient
from market_ingest.clients.polymarket.config import GAMMA_BASE_URL
from market_ingest.clients.polymarket.models import PolyEvent, PolyGammaMarket


class GammaClient(BaseHttpClient):
    """Client for Polymarket Gamma API (market metadata)."""

    def __init__(self) -> None:
        super().__init__(base_url=GAMMA_BASE_URL, rate_limit=40.0)

    async def list_events(
        self,
        active: bool | None = None,
        closed: bool | None = None,
        archived: bool | None = None,
        limit: int = 100,
        offset: int = 0,
        order: str = "volume",
        ascending: bool = False,
    ) -> list[PolyEvent]:
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "order": order,
            "ascending": str(ascending).lower(),
        }
        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()
        if archived is not None:
            params["archived"] = str(archived).lower()

        data = await self.get("/events", params=params)
        if isinstance(data, list):
            return [PolyEvent.model_validate(e) for e in data]
        return [PolyEvent.model_validate(e) for e in data.get("events", data.get("data", []))]

    async def get_event(self, event_id: str) -> PolyEvent:
        data = await self.get(f"/events/{event_id}")
        return PolyEvent.model_validate(data)

    async def list_markets(
        self,
        active: bool | None = None,
        closed: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[PolyGammaMarket]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()

        data = await self.get("/markets", params=params)
        if isinstance(data, list):
            return [PolyGammaMarket.model_validate(m) for m in data]
        return [PolyGammaMarket.model_validate(m) for m in data.get("data", [])]

    async def get_market(self, market_id: str) -> PolyGammaMarket:
        data = await self.get(f"/markets/{market_id}")
        return PolyGammaMarket.model_validate(data)

    async def list_all_active_events(self) -> list[PolyEvent]:
        """Paginate through all active events."""
        all_events: list[PolyEvent] = []
        offset = 0
        while True:
            batch = await self.list_events(active=True, limit=100, offset=offset)
            all_events.extend(batch)
            if len(batch) < 100:
                break
            offset += 100
        return all_events
