from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from market_ingest.clients.kalshi.models import (
    KalshiCandlestick,
    KalshiEvent,
    KalshiMarket,
    KalshiOrderbook,
)


def _parse_ts(val: str | int | None) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, int):
        return datetime.fromtimestamp(val, tz=timezone.utc)
    try:
        return datetime.fromisoformat(val.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _seconds_until(ts_str: str | None) -> int | None:
    dt = _parse_ts(ts_str)
    if dt is None:
        return None
    now = datetime.now(tz=timezone.utc)
    return max(0, int((dt - now).total_seconds()))


STATUS_MAP = {
    "initialized": "pending",
    "inactive": "pending",
    "active": "open",
    "closed": "closed",
    "determined": "resolved",
    "disputed": "disputed",
    "amended": "resolved",
    "finalized": "resolved",
    "settled": "resolved",
}


class KalshiMapper:
    """Maps Kalshi API responses to unified DB schema dicts."""

    PLATFORM = "kalshi"

    def market_to_db(
        self, km: KalshiMarket, event: KalshiEvent | None = None
    ) -> dict[str, Any]:
        return {
            "platform": self.PLATFORM,
            "platform_market_id": km.ticker,
            "title": (event.title if event else km.title) or km.ticker,
            "subtitle": event.sub_title if event else km.subtitle,
            "category": event.category if event else None,
            "status": STATUS_MAP.get(km.status, km.status),
            "market_type": "binary",
            "open_time": _parse_ts(km.open_time),
            "close_time": _parse_ts(km.close_time),
            "resolve_time": None,
            "resolution_source_text": None,
            "rules_text": km.rules_primary,
        }

    def orderbook_to_snapshot(
        self,
        ob: KalshiOrderbook,
        km: KalshiMarket,
        candlesticks: list[KalshiCandlestick] | None = None,
    ) -> dict[str, Any]:
        best_bid_yes = Decimal(ob.yes[0][0]) / 100 if ob.yes else None
        best_ask_yes = (100 - Decimal(ob.no[0][0])) / 100 if ob.no else None
        mid = (best_bid_yes + best_ask_yes) / 2 if best_bid_yes and best_ask_yes else None
        spread_bps = (
            int((best_ask_yes - best_bid_yes) * 10000)
            if best_bid_yes is not None and best_ask_yes is not None
            else None
        )
        last_yes = Decimal(km.last_price or 0) / 100 if km.last_price else None

        bid_depth = sum(level[1] for level in ob.yes[:5]) if ob.yes else 0
        ask_depth = sum(level[1] for level in ob.no[:5]) if ob.no else 0
        total = bid_depth + ask_depth
        imbalance = Decimal(bid_depth) / Decimal(total) if total > 0 else Decimal("0.5")

        vol_24h = None
        if candlesticks:
            vol_24h = sum(c.volume for c in candlesticks[-24:])

        return {
            "ts": datetime.now(tz=timezone.utc),
            "best_bid_yes": float(best_bid_yes) if best_bid_yes is not None else None,
            "best_ask_yes": float(best_ask_yes) if best_ask_yes is not None else None,
            "mid_yes": float(mid) if mid is not None else None,
            "last_yes": float(last_yes) if last_yes is not None else None,
            "spread_bps": spread_bps,
            "volume_1h": candlesticks[-1].volume if candlesticks else None,
            "volume_24h": vol_24h,
            "liquidity_proxy": float(bid_depth + ask_depth),
            "orderbook_imbalance": float(imbalance),
            "recent_volatility": self._compute_volatility(candlesticks),
            "time_to_close_sec": _seconds_until(km.close_time),
        }

    def market_to_outcome(self, km: KalshiMarket) -> dict[str, Any] | None:
        if km.status not in ("determined", "finalized", "settled"):
            return None
        resolved_label = None
        if km.result == "yes":
            resolved_label = 1
        elif km.result == "no":
            resolved_label = 0
        elif km.settlement_value is not None:
            resolved_label = 1 if km.settlement_value > 50 else 0

        return {
            "resolved_label": resolved_label,
            "resolved_at": datetime.now(tz=timezone.utc),
            "resolution_notes": f"result={km.result}, settlement_value={km.settlement_value}",
            "source_url": f"https://kalshi.com/markets/{km.ticker}",
        }

    @staticmethod
    def _compute_volatility(candlesticks: list[KalshiCandlestick] | None) -> float | None:
        if not candlesticks or len(candlesticks) < 2:
            return None
        closes = [c.close for c in candlesticks[-24:]]
        mean = sum(closes) / len(closes)
        variance = sum((c - mean) ** 2 for c in closes) / len(closes)
        return variance**0.5
