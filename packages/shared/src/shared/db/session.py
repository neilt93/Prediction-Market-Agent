from sqlalchemy import create_engine as _create_sync_engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker


def create_engine(database_url: str, **kwargs: object) -> AsyncEngine:
    return create_async_engine(
        database_url,
        echo=bool(kwargs.pop("echo", False)),
        pool_size=int(kwargs.pop("pool_size", 10)),
        max_overflow=int(kwargs.pop("max_overflow", 20)),
        pool_pre_ping=True,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


def create_sync_engine(database_url: str, **kwargs: object) -> _create_sync_engine:
    """For use in Celery workers."""
    return _create_sync_engine(
        database_url,
        echo=bool(kwargs.pop("echo", False)),
        pool_pre_ping=True,
    )


def create_sync_session_factory(engine: object) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)
