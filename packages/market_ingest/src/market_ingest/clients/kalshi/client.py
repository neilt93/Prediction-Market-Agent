from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse

import structlog

from market_ingest.clients.base import BaseHttpClient
from market_ingest.clients.kalshi.auth import KalshiAuthenticator
from market_ingest.clients.kalshi.config import KALSHI_BASE_URLS, KalshiEnvironment
from market_ingest.clients.kalshi.models import (
    KalshiCandlestick,
    KalshiEvent,
    KalshiMarket,
    KalshiOrder,
    KalshiOrderbook,
    KalshiOrderRequest,
    KalshiPosition,
    KalshiTrade,
)

logger = structlog.get_logger()


class KalshiClient(BaseHttpClient):
    """Kalshi REST API client with authentication and pagination."""

    def __init__(
        self,
        env: KalshiEnvironment = KalshiEnvironment.DEMO,
        api_key_id: str = "",
        private_key_pem: str = "",
        private_key_path: str = "",
    ) -> None:
        base_url = KALSHI_BASE_URLS[env]
        super().__init__(base_url=base_url, rate_limit=10.0)
        self.env = env

        if private_key_path and api_key_id:
            self._auth = KalshiAuthenticator.from_key_file(api_key_id, private_key_path)
        elif private_key_pem and api_key_id:
            self._auth = KalshiAuthenticator(api_key_id, private_key_pem)
        else:
            self._auth = None

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        if self._auth is None:
            return {}
        return self._auth.sign_request(method, path)

    def _full_path(self, path: str) -> str:
        """Get the full URL path for signing."""
        parsed = urlparse(self.base_url)
        base = parsed.path.rstrip("/")
        return f"{base}/{path.lstrip('/')}"

    async def _auth_request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Make an authenticated request — signs AFTER rate limiting."""
        await self.rate_limiter.acquire()
        client = await self._get_client()
        full_path = self._full_path(path)
        headers = self._auth_headers(method, full_path)
        resp = await client.request(method, path, headers=headers, **kwargs)
        resp.raise_for_status()
        return resp.json()

    async def _auth_get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._auth_request("GET", path, params=params)

    async def _auth_post(self, path: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._auth_request("POST", path, json=json)

    async def _auth_delete(self, path: str) -> dict[str, Any]:
        return await self._auth_request("DELETE", path)

    # --- Market Discovery ---

    async def get_events(
        self,
        status: str | None = None,
        with_nested_markets: bool = True,
        limit: int = 200,
    ) -> AsyncIterator[KalshiEvent]:
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"limit": limit}
            if status:
                params["status"] = status
            if with_nested_markets:
                params["with_nested_markets"] = "true"
            if cursor:
                params["cursor"] = cursor

            data = await self.get("/events", params=params)
            events = data.get("events", [])
            for ev in events:
                yield KalshiEvent.model_validate(ev)

            cursor = data.get("cursor")
            if not cursor or len(events) < limit:
                break

    async def get_market(self, ticker: str) -> KalshiMarket:
        data = await self.get(f"/markets/{ticker}")
        return KalshiMarket.model_validate(data.get("market", data))

    async def get_markets(
        self,
        status: str | None = None,
        event_ticker: str | None = None,
        limit: int = 200,
    ) -> AsyncIterator[KalshiMarket]:
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"limit": limit}
            if status:
                params["status"] = status
            if event_ticker:
                params["event_ticker"] = event_ticker
            if cursor:
                params["cursor"] = cursor

            data = await self.get("/markets", params=params)
            markets = data.get("markets", [])
            for m in markets:
                yield KalshiMarket.model_validate(m)

            cursor = data.get("cursor")
            if not cursor or len(markets) < limit:
                break

    # --- Order Book ---

    async def get_orderbook(self, ticker: str, depth: int | None = None) -> KalshiOrderbook:
        params: dict[str, Any] = {}
        if depth:
            params["depth"] = depth
        data = await self.get(f"/markets/{ticker}/orderbook", params=params)
        return KalshiOrderbook.model_validate(data.get("orderbook", data))

    async def get_orderbooks_batch(self, tickers: list[str]) -> list[KalshiOrderbook]:
        """Batch up to 100 tickers per call."""
        results: list[KalshiOrderbook] = []
        for i in range(0, len(tickers), 100):
            batch = tickers[i : i + 100]
            data = await self.get(
                "/markets/orderbooks", params={"tickers": ",".join(batch)}
            )
            for ob in data.get("orderbooks", {}).values():
                results.append(KalshiOrderbook.model_validate(ob))
        return results

    # --- Trades & Candlesticks ---

    async def get_trades(
        self, ticker: str, min_ts: int | None = None, limit: int = 100
    ) -> list[KalshiTrade]:
        params: dict[str, Any] = {"ticker": ticker, "limit": limit}
        if min_ts:
            params["min_ts"] = min_ts
        data = await self.get("/markets/trades", params=params)
        return [KalshiTrade.model_validate(t) for t in data.get("trades", [])]

    async def get_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        period_interval: int = 60,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[KalshiCandlestick]:
        params: dict[str, Any] = {"period_interval": period_interval}
        if start_ts:
            params["start_ts"] = start_ts
        if end_ts:
            params["end_ts"] = end_ts
        data = await self.get(
            f"/series/{series_ticker}/markets/{ticker}/candlesticks",
            params=params,
        )
        return [KalshiCandlestick.model_validate(c) for c in data.get("candlesticks", [])]

    # --- Historical ---

    async def get_historical_markets(
        self,
        status: str | None = None,
        min_close_ts: int | None = None,
        max_close_ts: int | None = None,
        limit: int = 200,
    ) -> AsyncIterator[KalshiMarket]:
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"limit": limit}
            if status:
                params["status"] = status
            if min_close_ts:
                params["min_close_ts"] = min_close_ts
            if max_close_ts:
                params["max_close_ts"] = max_close_ts
            if cursor:
                params["cursor"] = cursor

            data = await self.get("/historical/markets", params=params)
            markets = data.get("markets", [])
            for m in markets:
                yield KalshiMarket.model_validate(m)

            cursor = data.get("cursor")
            if not cursor or len(markets) < limit:
                break

    # --- Portfolio (auth required) ---

    async def get_balance(self) -> int:
        data = await self._auth_get("/portfolio/balance")
        return data.get("balance", 0)

    async def get_positions(self, ticker: str | None = None) -> list[KalshiPosition]:
        params: dict[str, Any] = {}
        if ticker:
            params["ticker"] = ticker
        data = await self._auth_get("/portfolio/positions", params=params)
        positions = data.get("market_positions", [])
        return [KalshiPosition.model_validate(p) for p in positions]

    async def create_order(self, order: KalshiOrderRequest) -> KalshiOrder:
        data = await self._auth_post("/portfolio/orders", json=order.model_dump())
        return KalshiOrder.model_validate(data.get("order", data))

    async def cancel_order(self, order_id: str) -> None:
        await self._auth_delete(f"/portfolio/orders/{order_id}")

    async def get_fills(self, ticker: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if ticker:
            params["ticker"] = ticker
        data = await self._auth_get("/portfolio/fills", params=params)
        return data.get("fills", [])
