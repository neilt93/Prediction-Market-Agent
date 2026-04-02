from __future__ import annotations

from types import SimpleNamespace

from market_ingest.clients.polymarket.models import PolyClobLevel, PolyClobOrderbook, PolyGammaMarket
from market_ingest.mappers.polymarket_mapper import PolymarketMapper
from run_live import SafetyState, filled_contract_count


TEST_SAFETY_CONFIG = {
    "min_edge_bps": 500,
    "min_confidence": 0.7,
    "max_ambiguity": 0.25,
    "max_spread_bps": 500,
    "max_per_trade_cents": 300,
    "max_total_exposure_cents": 1500,
    "daily_loss_limit": 5.0,
    "weekly_loss_limit": 10.0,
    "max_trades_per_day": 10,
    "consecutive_loss_pause": 5,
}


def test_polymarket_snapshot_accepts_pydantic_orderbook_levels() -> None:
    mapper = PolymarketMapper()
    market = PolyGammaMarket(question="Will it rain?", bestBid=0.4, bestAsk=0.6, liquidityNum=0.0)
    orderbook = PolyClobOrderbook(
        bids=[PolyClobLevel(price="0.45", size="10")],
        asks=[PolyClobLevel(price="0.55", size="8")],
    )

    snapshot = mapper.to_snapshot(market, orderbook)

    assert snapshot["best_bid_yes"] == 0.45
    assert snapshot["best_ask_yes"] == 0.55
    assert snapshot["liquidity_proxy"] == 18.0


def test_filled_contract_count_requires_an_actual_fill() -> None:
    resting_order = SimpleNamespace(place_count=1, remaining_count=1, status="resting")
    filled_order = SimpleNamespace(place_count=1, remaining_count=0, status="filled")

    assert filled_contract_count(resting_order, 1) == 0
    assert filled_contract_count(filled_order, 1) == 1


def test_safety_state_marks_losses_and_flags_large_drawdown() -> None:
    safety = SafetyState(TEST_SAFETY_CONFIG)
    safety.record_trade("TEST", "yes", 1, 60, "tech", "EVT")

    safety.mark_position("TEST", 40)
    assert safety._daily_pnl == -0.2

    safety.mark_position("TEST", 25)
    safety.check_position_health()

    assert safety._flagged_for_review is True
