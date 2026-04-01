from __future__ import annotations

from pydantic import BaseModel, Field


class KalshiEvent(BaseModel):
    event_ticker: str
    series_ticker: str = ""
    title: str = ""
    sub_title: str | None = None
    category: str | None = None
    mutually_exclusive: bool = False
    status: str = ""
    markets: list[KalshiMarket] | None = None
    strike_date: str | None = None


class KalshiMarket(BaseModel):
    ticker: str
    event_ticker: str = ""
    status: str = ""
    market_type: str = "binary"
    title: str | None = None
    subtitle: str | None = None
    yes_bid: int | None = Field(None, alias="yes_bid")
    yes_ask: int | None = Field(None, alias="yes_ask")
    last_price: int | None = None
    volume: int | None = None
    open_interest: int | None = None
    open_time: str | None = None
    close_time: str | None = None
    settlement_timer_seconds: int | None = None
    result: str | None = None
    rules_primary: str | None = None
    settlement_value: int | None = None

    model_config = {"populate_by_name": True}


class KalshiOrderbook(BaseModel):
    ticker: str = ""
    yes: list[list[int]] = []
    no: list[list[int]] = []


class KalshiTrade(BaseModel):
    ticker: str = ""
    trade_id: str = ""
    count: int = 0
    yes_price: int = 0
    no_price: int = 0
    created_time: str = ""
    taker_side: str = ""


class KalshiCandlestick(BaseModel):
    ticker: str = ""
    period_begin: str = ""
    open: float = 0
    high: float = 0
    low: float = 0
    close: float = 0
    volume: int = 0


class KalshiOrderRequest(BaseModel):
    ticker: str
    action: str  # "buy" or "sell"
    side: str  # "yes" or "no"
    type: str = "limit"
    count: int = 1
    yes_price: int | None = None
    no_price: int | None = None
    expiration_ts: int | None = None


class KalshiOrder(BaseModel):
    order_id: str = ""
    ticker: str = ""
    status: str = ""
    action: str = ""
    side: str = ""
    type: str = ""
    yes_price: int = 0
    no_price: int = 0
    created_time: str = ""
    remaining_count: int = 0
    place_count: int = 0


class KalshiPosition(BaseModel):
    ticker: str = ""
    market_exposure: int = 0
    resting_orders_count: int = 0
    total_traded: int = 0
    realized_pnl: int = 0
