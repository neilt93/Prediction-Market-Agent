from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from schemas.pydantic.common import OrmBase


class MarketCreate(OrmBase):
    platform: str
    platform_market_id: str
    title: str
    subtitle: str | None = None
    category: str | None = None
    status: str = "open"
    market_type: str = "binary"
    open_time: datetime | None = None
    close_time: datetime | None = None
    resolution_source_text: str | None = None
    rules_text: str | None = None


class MarketRead(OrmBase):
    id: uuid.UUID
    platform: str
    platform_market_id: str
    title: str
    subtitle: str | None
    category: str | None
    status: str
    market_type: str
    open_time: datetime | None
    close_time: datetime | None
    resolve_time: datetime | None
    created_at: datetime
    updated_at: datetime


class MarketSnapshotRead(OrmBase):
    id: uuid.UUID
    market_id: uuid.UUID
    ts: datetime
    best_bid_yes: Decimal | None
    best_ask_yes: Decimal | None
    mid_yes: Decimal | None
    last_yes: Decimal | None
    spread_bps: int | None
    volume_24h: Decimal | None


class MarketOutcomeRead(OrmBase):
    id: uuid.UUID
    market_id: uuid.UUID
    resolved_label: int | None
    resolved_at: datetime | None
    resolution_notes: str | None
    source_url: str | None
