from __future__ import annotations

import uuid
from decimal import Decimal

from sqlalchemy import Boolean, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from schemas.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Postmortem(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "postmortems"

    forecast_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("forecasts.id")
    )
    market_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("markets.id"), nullable=False
    )
    resolved_label: Mapped[int] = mapped_column(Integer, nullable=False)
    brier: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    log_loss: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    trading_pnl: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    error_bucket: Mapped[str | None] = mapped_column(String(50))
    error_notes: Mapped[str | None] = mapped_column(Text)
    human_reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    training_weight: Mapped[Decimal] = mapped_column(Numeric(5, 4), default=1.0)

    __table_args__ = (
        Index("ix_postmortems_market", "market_id"),
        Index("ix_postmortems_error_bucket", "error_bucket"),
    )
