from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Index, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from schemas.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class RuleParse(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "rule_parses"

    market_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("markets.id"), nullable=False
    )
    parsed_json: Mapped[dict | None] = mapped_column(JSONB)
    deadline_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    timezone: Mapped[str | None] = mapped_column(String(50))
    comparator: Mapped[str | None] = mapped_column(String(20))
    threshold_value: Mapped[str | None] = mapped_column(Text)
    entity: Mapped[str | None] = mapped_column(Text)
    source_of_truth: Mapped[str | None] = mapped_column(Text)
    ambiguity_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    parser_version: Mapped[str | None] = mapped_column(String(50))

    __table_args__ = (Index("ix_rule_parses_market", "market_id"),)
