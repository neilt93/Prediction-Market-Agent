from __future__ import annotations

from collections.abc import AsyncGenerator
from functools import lru_cache

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from api.config import ApiSettings


@lru_cache
def get_settings() -> ApiSettings:
    return ApiSettings()


def get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    return request.app.state.session_factory


async def get_db(
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> AsyncGenerator[AsyncSession, None]:
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
