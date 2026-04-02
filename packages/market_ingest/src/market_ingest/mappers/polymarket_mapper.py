from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from market_ingest.clients.polymarket.models import (
    PolyClobOrderbook,
    PolyEvent,
    PolyGammaMarket,
    PolyTrade,
)


def _parse_iso(val: str | None) -> datetime | None:
    if val is None:
        return None
    try:
        return datetime.fromisoformat(val.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _seconds_until(ts_str: str | None) -> int | None:
    dt = _parse_iso(ts_str)
    if dt is None:
        return None
    now = datetime.now(tz=timezone.utc)
    return max(0, int((dt - now).total_seconds()))


class PolymarketMapper:
    """Maps Polymarket API responses to unified DB schema dicts."""

    PLATFORM = "polymarket"

    def market_to_db(
        self, pm: PolyGammaMarket, event: PolyEvent | None = None
    ) -> dict[str, Any]:
        return {
            "platform": self.PLATFORM,
            "platform_market_id": pm.condition_id or pm.id,
            "title": pm.question,
            "subtitle": event.title if event and event.title != pm.question else None,
            "category": pm.category or (event.category if event else None),
            "status": self._derive_status(pm),
            "market_type": self._market_type(pm, event),
            "open_time": _parse_iso(pm.start_date),
            "close_time": _parse_iso(pm.end_date),
            "resolve_time": None,
            "resolution_source_text": pm.resolution_source,
            "rules_text": pm.description,
        }

    def to_snapshot(
        self,
        pm: PolyGammaMarket,
        ob: PolyClobOrderbook | None = None,
        trades: list[PolyTrade] | None = None,
    ) -> dict[str, Any]:
        bid_levels = ob.bid_levels if ob else []
        ask_levels = ob.ask_levels if ob else []
        best_bid_yes = bid_levels[0][0] if bid_levels else pm.best_bid
        best_ask_yes = ask_levels[0][0] if ask_levels else pm.best_ask
        mid = (
            (best_bid_yes + best_ask_yes) / 2
            if best_bid_yes is not None and best_ask_yes is not None
            else None
        )
        spread_bps = (
            int((best_ask_yes - best_bid_yes) * 10000)
            if best_bid_yes is not None and best_ask_yes is not None
            else None
        )
        last_yes = ob.last_trade_price if ob and ob.last_trade_price else None
        if last_yes is None and pm.outcome_prices:
            try:
                last_yes = float(pm.outcome_prices[0])
            except (ValueError, IndexError):
                pass

        bid_depth = sum(level[1] for level in bid_levels[:5]) if bid_levels else 0
        ask_depth = sum(level[1] for level in ask_levels[:5]) if ask_levels else 0
        total = bid_depth + ask_depth
        imbalance = bid_depth / total if total > 0 else 0.5

        vol_1h = self._compute_volume_window(trades, hours=1)
        vol_24h = float(pm.volume_num) if pm.volume_num else self._compute_volume_window(trades, hours=24)

        return {
            "ts": datetime.now(tz=timezone.utc),
            "best_bid_yes": best_bid_yes,
            "best_ask_yes": best_ask_yes,
            "mid_yes": mid,
            "last_yes": last_yes,
            "spread_bps": spread_bps,
            "volume_1h": vol_1h,
            "volume_24h": vol_24h,
            "liquidity_proxy": float(pm.liquidity_num) if pm.liquidity_num else float(total),
            "orderbook_imbalance": float(imbalance),
            "recent_volatility": None,
            "time_to_close_sec": _seconds_until(pm.end_date),
        }

    def market_to_outcome(self, pm: PolyGammaMarket) -> dict[str, Any] | None:
        if not (pm.closed or pm.archived):
            return None
        resolved_label = None
        if pm.outcome_prices:
            try:
                yes_price = float(pm.outcome_prices[0])
                if yes_price >= 0.99:
                    resolved_label = 1
                elif yes_price <= 0.01:
                    resolved_label = 0
            except (ValueError, IndexError):
                pass

        return {
            "resolved_label": resolved_label,
            "resolved_at": _parse_iso(pm.end_date),
            "resolution_notes": pm.uma_resolution_status,
            "source_url": f"https://polymarket.com/event/{pm.slug}",
        }

    @staticmethod
    def _derive_status(pm: PolyGammaMarket) -> str:
        if pm.archived:
            return "resolved"
        if pm.closed:
            return "closed"
        if pm.active:
            return "open"
        return "pending"

    @staticmethod
    def _market_type(pm: PolyGammaMarket, event: PolyEvent | None) -> str:
        if event and event.markets and len(event.markets) > 1:
            return "multi"
        return "binary"

    @staticmethod
    def _compute_volume_window(trades: list[PolyTrade] | None, hours: int) -> float | None:
        if not trades:
            return None
        cutoff = datetime.now(tz=timezone.utc).timestamp() - (hours * 3600)
        total = 0.0
        for t in trades:
            try:
                ts = datetime.fromisoformat(t.timestamp.replace("Z", "+00:00")).timestamp()
                if ts >= cutoff:
                    total += t.size * t.price
            except (ValueError, AttributeError):
                continue
        return total if total > 0 else None
