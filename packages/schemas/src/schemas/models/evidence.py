from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Index, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from schemas.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class EvidenceItem(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "evidence_items"

    market_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("markets.id"), nullable=False
    )
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_type: Mapped[str] = mapped_column(String(20), nullable=False)
    source_domain: Mapped[str | None] = mapped_column(String(255))
    url: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    title: Mapped[str | None] = mapped_column(Text)
    snippet: Mapped[str | None] = mapped_column(Text)
    content_embedding: Mapped[list | None] = mapped_column(Vector(384))
    freshness_hours: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    reliability_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    stance_hint: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    dedupe_hash: Mapped[str | None] = mapped_column(String(64))

    __table_args__ = (
        Index("ix_evidence_market", "market_id"),
        Index("ix_evidence_dedupe", "dedupe_hash"),
    )
