from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db
from schemas.models.market import Market, MarketOutcome, MarketSnapshot
from schemas.pydantic.market import MarketOutcomeRead, MarketRead, MarketSnapshotRead

router = APIRouter()


@router.get("/")
async def list_markets(
    platform: str | None = None,
    status: str | None = None,
    category: str | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> dict:
    query = select(Market)
    count_query = select(func.count(Market.id))

    if platform:
        query = query.where(Market.platform == platform)
        count_query = count_query.where(Market.platform == platform)
    if status:
        query = query.where(Market.status == status)
        count_query = count_query.where(Market.status == status)
    if category:
        query = query.where(Market.category == category)
        count_query = count_query.where(Market.category == category)

    query = query.order_by(Market.updated_at.desc()).offset(offset).limit(limit)

    total = (await db.execute(count_query)).scalar_one()
    results = (await db.execute(query)).scalars().all()

    return {
        "items": [MarketRead.model_validate(m) for m in results],
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@router.get("/{market_id}")
async def get_market(market_id: UUID, db: AsyncSession = Depends(get_db)) -> MarketRead:
    market = await db.get(Market, market_id)
    if not market:
        raise HTTPException(status_code=404, detail="Market not found")
    return MarketRead.model_validate(market)


@router.get("/{market_id}/snapshots")
async def get_market_snapshots(
    market_id: UUID,
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
) -> list[MarketSnapshotRead]:
    query = (
        select(MarketSnapshot)
        .where(MarketSnapshot.market_id == market_id)
        .order_by(MarketSnapshot.ts.desc())
        .limit(limit)
    )
    results = (await db.execute(query)).scalars().all()
    return [MarketSnapshotRead.model_validate(s) for s in results]


@router.get("/{market_id}/outcome")
async def get_market_outcome(
    market_id: UUID, db: AsyncSession = Depends(get_db)
) -> MarketOutcomeRead:
    query = select(MarketOutcome).where(MarketOutcome.market_id == market_id)
    result = (await db.execute(query)).scalar_one_or_none()
    if not result:
        raise HTTPException(status_code=404, detail="Outcome not found")
    return MarketOutcomeRead.model_validate(result)
