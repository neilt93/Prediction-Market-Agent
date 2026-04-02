from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from evidence.retriever import EvidenceRetriever
from run_backtest import _select_history_point


def test_select_history_point_uses_latest_price_before_cutoff() -> None:
    close_time = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)
    history = [
        SimpleNamespace(t=int((close_time - timedelta(hours=30)).timestamp()), p=0.31),
        SimpleNamespace(t=int((close_time - timedelta(hours=20)).timestamp()), p=0.44),
        SimpleNamespace(t=int((close_time - timedelta(hours=5)).timestamp()), p=0.87),
    ]

    selected = _select_history_point(history, close_time)

    assert selected is history[0]


def test_historical_evidence_filters_future_and_undated_items() -> None:
    as_of = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)
    stale_item = {"snippet": "old", "published_at": as_of - timedelta(hours=3)}
    future_item = {"snippet": "future", "published_at": as_of + timedelta(hours=1)}
    undated_item = {"snippet": "undated"}

    assert EvidenceRetriever._should_include(stale_item, as_of) is True
    assert EvidenceRetriever._should_include(future_item, as_of) is False
    assert EvidenceRetriever._should_include(undated_item, as_of) is False
