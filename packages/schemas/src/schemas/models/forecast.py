from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from schemas.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Forecast(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "forecasts"

    market_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("markets.id"), nullable=False
    )
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    adapter_name: Mapped[str | None] = mapped_column(String(100))
    prompt_version: Mapped[str | None] = mapped_column(String(50))
    raw_probability: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    abstain_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    reasoning_summary: Mapped[str | None] = mapped_column(Text)
    counterfactual_trigger_json: Mapped[dict | None] = mapped_column(JSONB)

    features: Mapped[list[ForecastFeature]] = relationship(back_populates="forecast")
    calibrated: Mapped[list[CalibratedForecast]] = relationship(back_populates="forecast")

    __table_args__ = (
        Index("ix_forecasts_market", "market_id"),
        Index("ix_forecasts_ts", "ts"),
    )


class ForecastFeature(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "forecast_features"

    forecast_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("forecasts.id"), nullable=False
    )
    market_price: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    spread_bps: Mapped[int | None] = mapped_column(Integer)
    vol_24h: Mapped[Decimal | None] = mapped_column(Numeric(18, 2))
    time_to_close_sec: Mapped[int | None] = mapped_column(Integer)
    ambiguity_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    freshness_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    source_agreement_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    official_source_present: Mapped[bool | None] = mapped_column(Boolean)
    category: Mapped[str | None] = mapped_column(String(100))
    platform: Mapped[str | None] = mapped_column(String(20))
    llm_confidence: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    retrieval_count: Mapped[int | None] = mapped_column(Integer)
    price_momentum_1h: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    price_momentum_24h: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))

    forecast: Mapped[Forecast] = relationship(back_populates="features")


class CalibratedForecast(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "calibrated_forecasts"

    forecast_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("forecasts.id"), nullable=False
    )
    calibrator_version: Mapped[str] = mapped_column(String(50), nullable=False)
    calibrated_probability: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    predicted_edge_bps: Mapped[int | None] = mapped_column(Integer)
    uncertainty_low: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    uncertainty_high: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))

    forecast: Mapped[Forecast] = relationship(back_populates="calibrated")
