from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from schemas.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from schemas.enums import MarketPlatform, MarketStatus, MarketType


class Market(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "markets"

    platform: Mapped[str] = mapped_column(String(20), nullable=False)
    platform_market_id: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    subtitle: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20), default=MarketStatus.OPEN.value)
    market_type: Mapped[str] = mapped_column(String(20), default=MarketType.BINARY.value)
    open_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    close_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolve_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolution_source_text: Mapped[str | None] = mapped_column(Text)
    rules_text: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB)
    title_embedding: Mapped[list | None] = mapped_column(Vector(384))

    snapshots: Mapped[list[MarketSnapshot]] = relationship(back_populates="market")
    outcomes: Mapped[list[MarketOutcome]] = relationship(back_populates="market")

    __table_args__ = (
        Index("ix_markets_platform_id", "platform", "platform_market_id", unique=True),
        Index("ix_markets_platform_status", "platform", "status"),
        Index("ix_markets_close_time", "close_time"),
    )


class MarketSnapshot(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "market_snapshots"

    market_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("markets.id"), nullable=False
    )
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    best_bid_yes: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    best_ask_yes: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    mid_yes: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    last_yes: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    spread_bps: Mapped[int | None] = mapped_column(Integer)
    volume_1h: Mapped[Decimal | None] = mapped_column(Numeric(18, 2))
    volume_24h: Mapped[Decimal | None] = mapped_column(Numeric(18, 2))
    liquidity_proxy: Mapped[Decimal | None] = mapped_column(Numeric(18, 2))
    orderbook_imbalance: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    recent_volatility: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    time_to_close_sec: Mapped[int | None] = mapped_column(Integer)

    market: Mapped[Market] = relationship(back_populates="snapshots")

    __table_args__ = (
        Index("ix_snapshots_market_time", "market_id", "ts"),
    )


class MarketOutcome(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "market_outcomes"

    market_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("markets.id"), nullable=False
    )
    resolved_label: Mapped[int | None] = mapped_column(Integer)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolution_notes: Mapped[str | None] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text)

    market: Mapped[Market] = relationship(back_populates="outcomes")
