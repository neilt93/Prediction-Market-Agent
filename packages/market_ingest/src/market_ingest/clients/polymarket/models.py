from __future__ import annotations

import json

from pydantic import BaseModel, Field, field_validator


def _parse_json_list(v: object) -> list[str]:
    """Parse a JSON string list or return as-is if already a list."""
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except (json.JSONDecodeError, TypeError):
            pass
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return []


class PolyGammaMarket(BaseModel):
    id: str = ""
    question: str = ""
    condition_id: str = Field("", alias="conditionId")
    slug: str = ""
    category: str | None = None
    outcomes: list[str] = []
    outcome_prices: list[str] = Field([], alias="outcomePrices")
    active: bool = False
    closed: bool = False
    archived: bool = False
    start_date: str | None = Field(None, alias="startDate")
    end_date: str | None = Field(None, alias="endDate")
    description: str | None = None
    resolution_source: str | None = Field(None, alias="resolutionSource")
    volume: str | None = None
    volume_num: float | None = Field(None, alias="volumeNum")
    liquidity: str | None = None
    liquidity_num: float | None = Field(None, alias="liquidityNum")
    best_bid: float | None = Field(None, alias="bestBid")
    best_ask: float | None = Field(None, alias="bestAsk")
    market_type: str | None = Field(None, alias="marketType")
    uma_resolution_status: str | None = Field(None, alias="umaResolutionStatus")
    enable_order_book: bool = Field(True, alias="enableOrderBook")
    clob_token_ids: list[str] | None = Field(None, alias="clobTokenIds")

    model_config = {"populate_by_name": True}

    @field_validator("outcomes", "outcome_prices", mode="before")
    @classmethod
    def parse_json_string_list(cls, v: object) -> list[str]:
        return _parse_json_list(v)

    @field_validator("clob_token_ids", mode="before")
    @classmethod
    def parse_clob_token_ids(cls, v: object) -> list[str] | None:
        if v is None:
            return None
        return _parse_json_list(v)


class PolyEvent(BaseModel):
    id: str = ""
    title: str = ""
    slug: str = ""
    description: str | None = None
    start_date: str | None = Field(None, alias="startDate")
    end_date: str | None = Field(None, alias="endDate")
    category: str | None = None
    active: bool = False
    closed: bool = False
    archived: bool = False
    liquidity: float | None = None
    volume: float | None = None
    markets: list[PolyGammaMarket] | None = None

    model_config = {"populate_by_name": True}


class PolyClobLevel(BaseModel):
    price: str = "0"
    size: str = "0"


class PolyClobOrderbook(BaseModel):
    market: str = ""
    asset_id: str = ""
    bids: list[PolyClobLevel] = []
    asks: list[PolyClobLevel] = []
    timestamp: str = ""
    last_trade_price: float | None = None
    tick_size: float | None = None

    @field_validator("last_trade_price", "tick_size", mode="before")
    @classmethod
    def parse_optional_float(cls, v: object) -> float | None:
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    @property
    def bid_levels(self) -> list[tuple[float, float]]:
        return [(float(b.price), float(b.size)) for b in self.bids]

    @property
    def ask_levels(self) -> list[tuple[float, float]]:
        return [(float(a.price), float(a.size)) for a in self.asks]


class PolyPricePoint(BaseModel):
    t: int = 0
    p: float = 0.0


class PolyTrade(BaseModel):
    id: str = ""
    market: str = ""
    side: str = ""
    price: float = 0.0
    size: float = 0.0
    timestamp: str = ""


class PolySpread(BaseModel):
    spread: float = 0.0
