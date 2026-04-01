from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Index, Integer, Numeric, String
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from schemas.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Order(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "orders"

    market_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("markets.id"), nullable=False
    )
    forecast_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("forecasts.id")
    )
    platform: Mapped[str] = mapped_column(String(20), nullable=False)
    env: Mapped[str] = mapped_column(String(10), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    order_type: Mapped[str] = mapped_column(String(20), nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    qty: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    filled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    avg_fill_price: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    fees: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    slippage_bps: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        Index("ix_orders_market", "market_id"),
        Index("ix_orders_status", "status"),
    )


class Position(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "positions"

    market_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("markets.id"), nullable=False, unique=True
    )
    net_qty: Mapped[int] = mapped_column(Integer, default=0)
    avg_cost: Mapped[Decimal] = mapped_column(Numeric(10, 6), default=0)
    mark_price: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    unrealized_pnl: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    realized_pnl: Mapped[Decimal] = mapped_column(Numeric(10, 4), default=0)
    max_risk: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
