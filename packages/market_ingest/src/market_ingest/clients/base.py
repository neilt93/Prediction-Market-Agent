from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


class TokenBucket:
    """Simple token-bucket rate limiter."""

    def __init__(self, rate: float, capacity: float | None = None) -> None:
        self.rate = rate
        self.capacity = capacity or rate * 2
        self.tokens = self.capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self._last_refill = now

            if self.tokens < 1:
                wait = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait)
                self.tokens = 0
            else:
                self.tokens -= 1


class BaseHttpClient:
    """Base async HTTP client with retry and rate limiting."""

    def __init__(
        self,
        base_url: str,
        rate_limit: float = 10.0,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.rate_limiter = TokenBucket(rate_limit)
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        await self.rate_limiter.acquire()
        client = await self._get_client()

        for attempt in range(max_retries):
            try:
                resp = await client.request(
                    method,
                    path,
                    params=params,
                    json=json,
                    headers=headers,
                )

                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", 2))
                    logger.warning(
                        "rate_limited", path=path, retry_after=retry_after
                    )
                    await asyncio.sleep(retry_after)
                    continue

                if resp.status_code >= 500:
                    wait = 2**attempt
                    logger.warning(
                        "server_error",
                        path=path,
                        status=resp.status_code,
                        retry_in=wait,
                    )
                    await asyncio.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()

            except httpx.HTTPStatusError:
                raise
            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2**attempt
                logger.warning("connection_error", path=path, error=str(e), retry_in=wait)
                await asyncio.sleep(wait)

        raise RuntimeError(f"Max retries exceeded for {method} {path}")

    async def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return await self._request("GET", path, params=params, headers=headers)

    async def post(
        self,
        path: str,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return await self._request("POST", path, json=json, headers=headers)

    async def delete(
        self,
        path: str,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return await self._request("DELETE", path, headers=headers)
