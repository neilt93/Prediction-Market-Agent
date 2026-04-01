"""Execution policy that decides whether and how to trade."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class TradeDecision:
    """Output of the execution policy."""
    should_trade: bool
    side: str = ""  # "buy_yes" or "buy_no"
    order_type: str = "limit"
    limit_price: float = 0.0
    quantity: int = 0
    reason: str = ""
    edge_bps: int = 0
    confidence: float = 0.0


@dataclass
class PolicyConfig:
    """Configurable trade policy thresholds."""
    min_edge_bps: int = 400  # 4 cents
    min_confidence: float = 0.65
    max_ambiguity: float = 0.25
    max_spread_bps: int = 1000  # 10 cents
    min_liquidity: float = 50.0
    max_position_per_market: int = 5  # $5
    daily_max_loss: float = 20.0  # $20
    approved_categories: list[str] | None = None


class ExecutionPolicy:
    """Decides whether to trade based on forecast, calibration, and risk gates."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()
        self._daily_pnl = 0.0
        self._daily_reset_date: str = ""

    def evaluate(
        self,
        calibrated_probability: float,
        market_price: float,
        confidence: float,
        ambiguity_score: float,
        spread_bps: int | None,
        liquidity: float | None,
        current_position_qty: int = 0,
        category: str | None = None,
        abstain_flag: bool = False,
    ) -> TradeDecision:
        """Evaluate all risk gates and return a trade decision."""

        # Reset daily PnL tracker
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            self._daily_pnl = 0.0
            self._daily_reset_date = today

        # Gate 1: Abstain flag
        if abstain_flag:
            return TradeDecision(should_trade=False, reason="Model abstained")

        # Gate 2: Confidence
        if confidence < self.config.min_confidence:
            return TradeDecision(
                should_trade=False,
                reason=f"Confidence {confidence:.2f} < {self.config.min_confidence}",
            )

        # Gate 3: Ambiguity
        if ambiguity_score > self.config.max_ambiguity:
            return TradeDecision(
                should_trade=False,
                reason=f"Ambiguity {ambiguity_score:.2f} > {self.config.max_ambiguity}",
            )

        # Gate 4: Edge
        edge_bps = int((calibrated_probability - market_price) * 10000)
        abs_edge = abs(edge_bps)
        if abs_edge < self.config.min_edge_bps:
            return TradeDecision(
                should_trade=False,
                reason=f"Edge {abs_edge}bps < {self.config.min_edge_bps}bps",
            )

        # Gate 5: Spread
        if spread_bps is not None and spread_bps > self.config.max_spread_bps:
            return TradeDecision(
                should_trade=False,
                reason=f"Spread {spread_bps}bps > {self.config.max_spread_bps}bps",
            )

        # Gate 6: Liquidity
        if liquidity is not None and liquidity < self.config.min_liquidity:
            return TradeDecision(
                should_trade=False,
                reason=f"Liquidity {liquidity:.0f} < {self.config.min_liquidity}",
            )

        # Gate 7: Position size
        if abs(current_position_qty) >= self.config.max_position_per_market:
            return TradeDecision(
                should_trade=False,
                reason=f"Position {current_position_qty} at max {self.config.max_position_per_market}",
            )

        # Gate 8: Daily loss limit
        if self._daily_pnl <= -self.config.daily_max_loss:
            return TradeDecision(
                should_trade=False,
                reason=f"Daily loss {self._daily_pnl:.2f} hit limit",
            )

        # Gate 9: Category check
        if self.config.approved_categories and category:
            if category.lower() not in [c.lower() for c in self.config.approved_categories]:
                return TradeDecision(
                    should_trade=False,
                    reason=f"Category '{category}' not approved",
                )

        # All gates passed - determine trade direction and sizing
        if edge_bps > 0:
            side = "buy_yes"
            limit_price = min(calibrated_probability, market_price + 0.02)
        else:
            side = "buy_no"
            limit_price = max(1 - calibrated_probability, (1 - market_price) + 0.02)

        # Simple sizing: 1 contract per trade, capped by position limit
        remaining_capacity = self.config.max_position_per_market - abs(current_position_qty)
        quantity = min(1, remaining_capacity)

        return TradeDecision(
            should_trade=True,
            side=side,
            order_type="limit",
            limit_price=round(limit_price, 2),
            quantity=quantity,
            reason=f"Edge {abs_edge}bps, conf {confidence:.2f}",
            edge_bps=edge_bps,
            confidence=confidence,
        )

    def record_fill(self, pnl: float) -> None:
        """Record a fill's PnL for daily tracking."""
        self._daily_pnl += pnl
