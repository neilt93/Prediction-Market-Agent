from __future__ import annotations

from pydantic import BaseModel, Field


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


class PolyClobOrderbook(BaseModel):
    market: str = ""
    asset_id: str = ""
    bids: list[list[float]] = []
    asks: list[list[float]] = []
    timestamp: str = ""
    last_trade_price: float | None = None
    tick_size: float | None = None


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
