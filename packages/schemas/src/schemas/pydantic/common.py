from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class OrmBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class PaginatedResponse[T](BaseModel):
    items: list[T]
    total: int
    offset: int
    limit: int


class ErrorResponse(BaseModel):
    detail: str
    error_code: str | None = None
