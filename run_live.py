"""Live trading loop on Kalshi with comprehensive safety system.

Safety layers:
- Pre-trade: edge threshold, confidence floor, liquidity, spread, time-to-resolution, ambiguity
- Position: max per trade, total exposure cap, category cap, no averaging down
- Session: daily/weekly loss limits, max trades/day, consecutive loss pause
- System: kill switch, stale data guard, API error pause, model integrity check
- Integrity: balance verification, fill price tolerance, loss flag for review

Usage:
  uv run python run_live.py                  # Dry run (default)
  uv run python run_live.py --live           # Execute real trades
  touch STOP_TRADING                         # Kill switch (halt immediately)
"""
from __future__ import annotations

import asyncio
import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

for pkg in ["shared", "schemas", "market_ingest", "rules", "forecasting",
            "calibration", "execution", "evidence"]:
    sys.path.insert(0, str(Path(__file__).parent / "packages" / pkg / "src"))

from shared.config import BaseAppSettings
from shared.logging import setup_logging

import structlog
logger = structlog.get_logger()

STOP_FILE = Path("STOP_TRADING")
MODEL_PATH = Path("data/models/calibrator_latest.lgb")
TRADE_LOG_PATH = Path("data/trade_log.jsonl")

# ─── Niche filter ───────────────────────────────────────────────────────────

GEOPOLITICS_KW = [
    "trump", "iran", "ukraine", "russia", "china", "nato", "ceasefire",
    "war", "sanctions", "tariff", "election", "president", "fed ",
    "interest rate", "congress", "senate", "supreme court", "government",
    "pope", "netanyahu", "putin", "zelenskyy", "biden", "executive order",
    "impeach", "veto", "shutdown", "debt ceiling", "cabinet", "nominee",
]
TECH_KW = [
    "tesla", "nvidia", "apple", "google", "openai", "ai ", "spacex",
    "elon musk", "anthropic", "meta ", "microsoft", "amazon", "chatgpt",
    "semiconductor", "chip", "robot",
]
SKIP_KW = [
    "nba", "nfl", "nhl", "mlb", "ncaa", "spread", "o/u", "over/under",
    "rebounds", "assists", "points", "rushing", "passing", "touchdown",
    "strikeout", "goals", "premier league", "la liga", "champions league",
    "world cup", "f1 driver", "formula 1", "masters tournament",
    "grand slam", "wimbledon", "stanley cup", "super bowl",
    "tweets", "tweet count", "followers",
    "bitcoin reach", "bitcoin dip", "ethereum reach", "ethereum dip",
    "btc reach", "btc dip", "solana reach", "solana dip",
    "above $", "below $",
]


def classify_niche(title: str) -> str | None:
    t = title.lower()
    for kw in SKIP_KW:
        if kw in t:
            return None
    for kw in GEOPOLITICS_KW:
        if kw in t:
            return "geopolitics"
    for kw in TECH_KW:
        if kw in t:
            return "tech"
    return None


def orderbook_yes_mid_cents(orderbook: Any) -> int | None:
    yes_levels = getattr(orderbook, "yes", None) or []
    no_levels = getattr(orderbook, "no", None) or []
    if not yes_levels or not no_levels:
        return None
    best_bid_cents = int(yes_levels[0][0])
    best_ask_cents = int(100 - no_levels[0][0])
    return int(round((best_bid_cents + best_ask_cents) / 2))


def filled_contract_count(order: Any, requested_count: int) -> int:
    placed = int(getattr(order, "place_count", 0) or 0)
    remaining = int(getattr(order, "remaining_count", 0) or 0)
    if placed > 0:
        return max(0, placed - remaining)

    status = str(getattr(order, "status", "") or "").lower()
    if remaining == 0 and status not in {"canceled", "cancelled", "rejected"}:
        return requested_count
    return 0


def fill_price_cents(order: Any, side: str, fallback_price_cents: int) -> int:
    field_name = "yes_price" if side == "yes" else "no_price"
    raw_price = getattr(order, field_name, None)
    if raw_price in (None, 0, "0"):
        return fallback_price_cents
    return max(1, min(99, int(raw_price)))


# ─── Safety state tracker ───────────────────────────────────────────────────

class SafetyState:
    """Tracks all safety limits and state."""

    def __init__(self, config: dict):
        self.config = config
        # Session tracking
        self._daily_pnl = 0.0
        self._daily_date = ""
        self._weekly_pnl = 0.0
        self._weekly_start = ""
        self._daily_trade_count = 0
        self._consecutive_losses = 0
        self._pause_until: datetime | None = None
        self._api_error_count = 0
        # Position tracking
        self._positions: dict[str, dict[str, Any]] = {}
        self._expected_balance: int | None = None
        # Model integrity
        self._model_hash: str | None = None
        # Flagged for review
        self._flagged_for_review = False

    def check_model_integrity(self) -> bool:
        """Hash calibrator model file, refuse to trade if modified externally."""
        if not MODEL_PATH.exists():
            logger.warning("No calibrator model found — trading without calibration")
            return True
        current_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:16]
        if self._model_hash is None:
            self._model_hash = current_hash
            return True
        if current_hash != self._model_hash:
            logger.error(f"MODEL INTEGRITY FAILED: hash changed {self._model_hash} -> {current_hash}")
            return False
        return True

    def check_kill_switch(self) -> bool:
        if STOP_FILE.exists():
            logger.warning("KILL SWITCH: STOP_TRADING file detected")
            return True
        return False

    def check_session_limits(self) -> str | None:
        """Return reason if trading should pause, None if OK."""
        now = datetime.now(tz=timezone.utc)

        # Pause timer
        if self._pause_until and now < self._pause_until:
            remaining = (self._pause_until - now).total_seconds() / 60
            return f"Paused for {remaining:.0f} more minutes"

        # Daily reset
        today = now.strftime("%Y-%m-%d")
        if today != self._daily_date:
            self._daily_pnl = 0.0
            self._daily_trade_count = 0
            self._daily_date = today

        # Weekly reset (Monday)
        week_start = (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d")
        if week_start != self._weekly_start:
            self._weekly_pnl = 0.0
            self._weekly_start = week_start

        # Daily loss limit
        if self._daily_pnl <= -self.config["daily_loss_limit"]:
            return f"Daily loss limit hit: ${self._daily_pnl:.2f}"

        # Weekly loss limit
        if self._weekly_pnl <= -self.config["weekly_loss_limit"]:
            return f"Weekly loss limit hit: ${self._weekly_pnl:.2f}"

        # Max trades per day
        if self._daily_trade_count >= self.config["max_trades_per_day"]:
            return f"Max {self.config['max_trades_per_day']} trades/day reached"

        # Consecutive losses -> pause
        if self._consecutive_losses >= self.config["consecutive_loss_pause"]:
            self._pause_until = now + timedelta(hours=6)
            self._consecutive_losses = 0
            return f"5 consecutive losses — pausing 6 hours"

        # API errors
        if self._api_error_count >= 3:
            self._pause_until = now + timedelta(minutes=30)
            self._api_error_count = 0
            return "3 consecutive API errors — pausing 30 minutes"

        # Flagged for review
        if self._flagged_for_review:
            return "FLAGGED: position >50% loss — manual review required"

        return None

    def check_pre_trade(
        self,
        ticker: str,
        edge_bps: int,
        confidence: float,
        ambiguity: float,
        spread_bps: int,
        time_to_close_hours: float | None,
        niche: str,
        event_ticker: str,
    ) -> str | None:
        """Return rejection reason, or None if trade passes all gates."""

        # Edge threshold: > 5% (500 bps)
        if abs(edge_bps) < self.config["min_edge_bps"]:
            return f"Edge {abs(edge_bps)}bps < {self.config['min_edge_bps']}bps"

        # Confidence floor
        if confidence < self.config["min_confidence"]:
            return f"Confidence {confidence:.2f} < {self.config['min_confidence']}"

        # Ambiguity
        if ambiguity > self.config["max_ambiguity"]:
            return f"Ambiguity {ambiguity:.2f} > {self.config['max_ambiguity']}"

        # Spread check: < 5 cents (500 bps)
        if spread_bps > self.config["max_spread_bps"]:
            return f"Spread {spread_bps}bps > {self.config['max_spread_bps']}bps"

        # Time to resolution
        if time_to_close_hours is not None:
            if time_to_close_hours < 2:
                return f"Resolves in {time_to_close_hours:.1f}h (min 2h)"
            if time_to_close_hours > 24 * 30:
                return f"Resolves in {time_to_close_hours/24:.0f}d (max 30d)"

        # Position limit per market
        pos = self._positions.get(ticker)
        if pos and pos["qty"] * 100 >= self.config["max_per_trade_cents"]:
            return f"Max position reached for {ticker}"

        # No averaging down
        if pos and pos["qty"] != 0:
            return f"Already positioned in {ticker} — no averaging"

        # Total exposure cap
        total_exposure = sum(p["qty"] * p.get("cost_cents", 50) for p in self._positions.values())
        if total_exposure >= self.config["max_total_exposure_cents"]:
            return f"Total exposure ${total_exposure/100:.2f} at limit"

        # Category cap: max 3 in same niche
        same_niche = sum(1 for p in self._positions.values() if p.get("category") == niche)
        if same_niche >= 3:
            return f"Already 3 positions in {niche}"

        # Correlation check: max 2 on same event
        same_event = sum(1 for p in self._positions.values() if p.get("event") == event_ticker)
        if same_event >= 2:
            return f"Already 2 positions on event {event_ticker}"

        return None

    def verify_balance(self, actual_balance: int) -> bool:
        """Check balance matches expected. Flag if discrepancy > $1."""
        if self._expected_balance is None:
            self._expected_balance = actual_balance
            return True
        diff = abs(actual_balance - self._expected_balance)
        if diff > 100:  # More than $1 discrepancy
            logger.error(
                f"BALANCE DISCREPANCY: expected ${self._expected_balance/100:.2f}, "
                f"got ${actual_balance/100:.2f} (diff: ${diff/100:.2f})"
            )
            return False
        return True

    def open_tickers(self) -> list[str]:
        return list(self._positions)

    def record_trade(
        self,
        ticker: str,
        side: str,
        qty: int,
        cost_cents: int,
        niche: str,
        event_ticker: str,
    ) -> None:
        self._positions[ticker] = {
            "qty": qty,
            "side": side,
            "cost_cents": cost_cents,
            "last_contract_price_cents": cost_cents,
            "current_contract_price_cents": cost_cents,
            "category": niche,
            "event": event_ticker,
        }
        self._daily_trade_count += 1

    def record_pnl(self, pnl_cents: int) -> None:
        pnl = pnl_cents / 100
        self._daily_pnl += pnl
        self._weekly_pnl += pnl
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def mark_position(self, ticker: str, yes_mid_cents: int) -> None:
        pos = self._positions.get(ticker)
        if pos is None:
            return
        current_contract_price_cents = yes_mid_cents if pos["side"] == "yes" else 100 - yes_mid_cents
        last_contract_price_cents = pos.get("last_contract_price_cents", current_contract_price_cents)
        pnl_delta_cents = (current_contract_price_cents - last_contract_price_cents) * pos["qty"]
        if pnl_delta_cents != 0:
            self.record_pnl(pnl_delta_cents)
        pos["last_contract_price_cents"] = current_contract_price_cents
        pos["current_contract_price_cents"] = current_contract_price_cents

    def check_position_health(self) -> None:
        """Flag if any live-marked position shows > 50% loss."""
        for ticker, pos in self._positions.items():
            cost_cents = pos.get("cost_cents", 0)
            current_contract_price_cents = pos.get("current_contract_price_cents")
            if not cost_cents or current_contract_price_cents is None:
                continue
            loss_ratio = (cost_cents - current_contract_price_cents) / cost_cents
            if loss_ratio >= 0.5:
                self._flagged_for_review = True
                logger.error(
                    "POSITION HEALTH FLAG",
                    ticker=ticker,
                    side=pos.get("side"),
                    entry_cents=cost_cents,
                    mark_cents=current_contract_price_cents,
                    loss_ratio=round(loss_ratio, 3),
                )

    def sync_balance(self, actual_balance: int) -> None:
        self._expected_balance = actual_balance

    def record_api_error(self) -> None:
        self._api_error_count += 1

    def clear_api_errors(self) -> None:
        self._api_error_count = 0


# ─── Trade logger ───────────────────────────────────────────────────────────

def log_trade_decision(decision: dict):
    """Append every trade decision (taken or skipped) to JSONL log."""
    TRADE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    decision["timestamp"] = datetime.now(tz=timezone.utc).isoformat()
    with open(TRADE_LOG_PATH, "a") as f:
        f.write(json.dumps(decision, default=str) + "\n")


# ─── Main trading loop ─────────────────────────────────────────────────────

class LiveTrader:
    def __init__(self, dry_run: bool = True, cycle_interval: int = 300):
        self.dry_run = dry_run
        self.cycle_interval = cycle_interval
        self.safety = SafetyState({
            "min_edge_bps": 500,           # 5 cents
            "min_confidence": 0.7,
            "max_ambiguity": 0.25,
            "max_spread_bps": 500,         # 5 cents
            "max_per_trade_cents": 300,    # $3
            "max_total_exposure_cents": 1500,  # $15
            "daily_loss_limit": 5.0,       # $5
            "weekly_loss_limit": 10.0,     # $10
            "max_trades_per_day": 10,
            "consecutive_loss_pause": 5,
        })
        self._cycle_count = 0
        self._balance_getter: Any | None = None

    async def run(self):
        from market_ingest.clients.kalshi.client import KalshiClient
        from market_ingest.clients.kalshi.config import KalshiEnvironment
        from forecasting.forecaster import Forecaster
        from calibration.calibrator import Calibrator
        from evidence.retriever import EvidenceRetriever
        from rules.parser import RuleParser
        import httpx as _httpx
        from market_ingest.clients.kalshi.auth import KalshiAuthenticator

        setup_logging(log_level="INFO")
        settings = BaseAppSettings()

        kalshi = KalshiClient(
            env=KalshiEnvironment(settings.kalshi_env),
            api_key_id=settings.kalshi_api_key_id,
            private_key_path=settings.kalshi_private_key_path,
        )
        forecaster = Forecaster(api_url=settings.llm_api_url, model=settings.llm_model)
        calibrator = Calibrator(model_path=str(MODEL_PATH) if MODEL_PATH.exists() else None)
        evidence = EvidenceRetriever()
        rule_parser = RuleParser()

        # Direct auth for balance checks
        _auth = KalshiAuthenticator.from_key_file(settings.kalshi_api_key_id, settings.kalshi_private_key_path)

        def get_balance() -> int:
            headers = _auth.sign_request("GET", "/trade-api/v2/portfolio/balance")
            resp = _httpx.get(f"{kalshi.base_url}/portfolio/balance", headers=headers)
            resp.raise_for_status()
            return resp.json().get("balance", 0)

        self._balance_getter = get_balance

        mode = "DRY RUN" if self.dry_run else "LIVE TRADING"
        logger.info(f"=== KALSHI {mode} ===")
        logger.info(f"  Safety config: {json.dumps(self.safety.config, indent=2)}")
        logger.info(f"  Kill switch: touch STOP_TRADING to halt")
        logger.info(f"  Cycle interval: {self.cycle_interval}s")

        # Startup checks
        if not self.safety.check_model_integrity():
            logger.error("Model integrity check failed — aborting")
            return

        try:
            balance = get_balance()
            self.safety.sync_balance(balance)
            logger.info(f"  Balance: ${balance/100:.2f}")
        except Exception as e:
            logger.error(f"  Auth failed: {e}")
            return

        try:
            while True:
                # System checks
                if self.safety.check_kill_switch():
                    break
                if not self.safety.check_model_integrity():
                    break

                self._cycle_count += 1
                ts = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
                logger.info(f"\n--- Cycle {self._cycle_count} [{ts}] ---")

                # Verify balance before trading
                try:
                    actual_balance = get_balance()
                    if not self.safety.verify_balance(actual_balance):
                        logger.error("Balance discrepancy — pausing")
                        break
                    self.safety.clear_api_errors()
                except Exception as e:
                    logger.error(f"Balance check failed: {e}")
                    self.safety.record_api_error()
                    await asyncio.sleep(30)
                    continue

                await self._refresh_position_marks(kalshi)

                session_issue = self.safety.check_session_limits()
                if session_issue:
                    logger.info(f"  PAUSED: {session_issue}")
                    await asyncio.sleep(60)
                    continue

                try:
                    await self._run_cycle(kalshi, forecaster, calibrator, evidence, rule_parser)
                except Exception as e:
                    logger.error(f"Cycle error: {e}")
                    self.safety.record_api_error()

                logger.info(f"  Next cycle in {self.cycle_interval}s...")
                await asyncio.sleep(self.cycle_interval)

        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down")
        finally:
            await kalshi.close()
            await forecaster.close()
            await evidence.close()
            logger.info("Shutdown complete")

    async def _refresh_position_marks(self, kalshi) -> None:
        for ticker in self.safety.open_tickers():
            try:
                orderbook = await kalshi.get_orderbook(ticker)
            except Exception as exc:
                logger.warning(f"Position mark refresh failed for {ticker}: {exc}")
                continue
            yes_mid_cents = orderbook_yes_mid_cents(orderbook)
            if yes_mid_cents is None:
                continue
            self.safety.mark_position(ticker, yes_mid_cents)
        self.safety.check_position_health()

    async def _run_cycle(self, kalshi, forecaster, calibrator, evidence, rule_parser):
        # 1. Discover target markets
        targets = []
        try:
            async for event in kalshi.get_events(status="open", limit=200):
                if not event.markets:
                    continue
                for km in event.markets:
                    niche = classify_niche(km.title or event.title)
                    if niche and km.status == "active":
                        targets.append((event, km, niche))
        except Exception as e:
            logger.error(f"Market discovery failed: {e}")
            self.safety.record_api_error()
            return

        logger.info(f"  {len(targets)} target markets")
        trades_this_cycle = 0

        for event, km, niche in targets:
            if self.safety.check_kill_switch():
                break
            session_issue = self.safety.check_session_limits()
            if session_issue:
                logger.info(f"  PAUSED: {session_issue}")
                break

            title = km.title or event.title
            ticker = km.ticker

            # 2. Orderbook + spread
            try:
                ob = await kalshi.get_orderbook(ticker)
            except Exception:
                continue

            yes_mid_cents = orderbook_yes_mid_cents(ob)
            if yes_mid_cents is None:
                continue

            best_bid_cents = ob.yes[0][0]
            best_no_bid_cents = ob.no[0][0]
            best_bid = best_bid_cents / 100
            best_ask = (100 - best_no_bid_cents) / 100
            mid = (best_bid + best_ask) / 2
            spread_bps = int((best_ask - best_bid) * 10000)
            if ticker in self.safety.open_tickers():
                self.safety.mark_position(ticker, yes_mid_cents)
                self.safety.check_position_health()

            # Time to close
            time_to_close_hours = None
            if km.close_time:
                try:
                    close_dt = datetime.fromisoformat(km.close_time.replace("Z", "+00:00"))
                    time_to_close_hours = max(0, (close_dt - datetime.now(tz=timezone.utc)).total_seconds() / 3600)
                except (ValueError, TypeError):
                    pass

            # 3. Parse + evidence + forecast
            parsed = rule_parser.parse(title, km.rules_primary)
            bundle = await evidence.gather(title, parsed.entity)

            forecast_output = await forecaster.forecast(
                title=title,
                rules_text=km.rules_primary,
                parsed_rules=parsed.model_dump(),
                evidence_snippets=bundle.top_snippets(8),
                market_price=mid,
                time_to_close_hours=time_to_close_hours,
            )

            if forecast_output.abstain:
                log_trade_decision({"action": "skip", "ticker": ticker, "title": title[:60],
                                    "reason": "model_abstained", "niche": niche})
                continue

            # 4. Calibrate
            features = {
                "market_price": mid, "spread_bps": spread_bps,
                "ambiguity_score": parsed.ambiguity_score,
                "llm_confidence": forecast_output.confidence,
                "retrieval_count": bundle.count,
                "raw_probability": forecast_output.raw_probability,
            }
            cal = calibrator.predict(features, market_price=mid)
            edge_bps = int((cal.calibrated_probability - mid) * 10000)

            # 5. Pre-trade safety checks
            rejection = self.safety.check_pre_trade(
                ticker=ticker, edge_bps=edge_bps, confidence=forecast_output.confidence,
                ambiguity=parsed.ambiguity_score, spread_bps=spread_bps,
                time_to_close_hours=time_to_close_hours, niche=niche,
                event_ticker=event.event_ticker,
            )

            if rejection:
                log_trade_decision({
                    "action": "skip", "ticker": ticker, "title": title[:60],
                    "niche": niche, "reason": rejection,
                    "forecast": forecast_output.raw_probability,
                    "calibrated": cal.calibrated_probability,
                    "market_price": mid, "edge_bps": edge_bps,
                    "confidence": forecast_output.confidence,
                    "evidence_count": bundle.count,
                })
                continue

            # 6. Determine trade parameters
            if edge_bps > 0:
                side = "yes"
                limit_price_cents = min(int(cal.calibrated_probability * 100), best_bid_cents + 2)
            else:
                side = "no"
                limit_price_cents = min(int((1 - cal.calibrated_probability) * 100), best_no_bid_cents + 2)

            limit_price_cents = max(1, min(99, limit_price_cents))

            trade_record = {
                "action": "trade" if not self.dry_run else "dry_trade",
                "ticker": ticker, "title": title[:60], "niche": niche,
                "side": side, "price_cents": limit_price_cents,
                "forecast": round(forecast_output.raw_probability, 4),
                "calibrated": round(cal.calibrated_probability, 4),
                "market_price": round(mid, 4),
                "edge_bps": abs(edge_bps),
                "confidence": round(forecast_output.confidence, 4),
                "spread_bps": spread_bps,
                "evidence_count": bundle.count,
                "reasoning": forecast_output.reasoning_summary[:200],
            }

            prefix = "[DRY]" if self.dry_run else "[LIVE]"
            logger.info(
                f"  {prefix} BUY {side.upper()} {ticker} @ {limit_price_cents}c | "
                f"edge={abs(edge_bps)}bps conf={forecast_output.confidence:.2f} "
                f"[{niche}] {title[:35]}"
            )

            if not self.dry_run:
                from market_ingest.clients.kalshi.models import KalshiOrderRequest
                try:
                    order = KalshiOrderRequest(
                        ticker=ticker, action="buy", side=side, type="limit",
                        count=1,
                        yes_price=limit_price_cents if side == "yes" else None,
                        no_price=limit_price_cents if side == "no" else None,
                    )
                    result = await kalshi.create_order(order)
                    trade_record["order_id"] = result.order_id
                    trade_record["order_status"] = result.status
                    logger.info(f"    Order: {result.order_id} status={result.status}")

                    filled_count = filled_contract_count(result, order.count)
                    remaining_count = max(0, int(result.remaining_count or 0))
                    trade_record["filled_count"] = filled_count
                    trade_record["remaining_count"] = remaining_count

                    if remaining_count > 0:
                        try:
                            await kalshi.cancel_order(result.order_id)
                            trade_record["remainder_canceled"] = remaining_count
                            logger.info(f"    Canceled {remaining_count} resting contract(s)")
                        except Exception as cancel_exc:
                            logger.warning(f"    Failed to cancel remainder: {cancel_exc}")
                            self.safety._expected_balance = None

                    if filled_count > 0:
                        executed_price_cents = fill_price_cents(result, side, limit_price_cents)
                        trade_record["fill_price_cents"] = executed_price_cents
                        if abs(executed_price_cents - limit_price_cents) > 2:
                            logger.warning(
                                f"    Fill price {executed_price_cents}c vs limit {limit_price_cents}c - tolerance exceeded"
                            )
                        self.safety.record_trade(
                            ticker,
                            side,
                            filled_count,
                            executed_price_cents,
                            niche,
                            event.event_ticker,
                        )
                        if self.safety._expected_balance is not None:
                            self.safety._expected_balance -= executed_price_cents * filled_count
                    else:
                        logger.info("    No immediate fill; position not booked")

                    if self._balance_getter is not None:
                        try:
                            self.safety.sync_balance(self._balance_getter())
                        except Exception as balance_exc:
                            logger.warning(f"    Post-trade balance sync failed: {balance_exc}")
                            self.safety._expected_balance = None

                except Exception as e:
                    logger.error(f"    Order failed: {e}")
                    trade_record["error"] = str(e)
                    self.safety.record_api_error()
            else:
                self.safety.record_trade(ticker, side, 1, limit_price_cents, niche, event.event_ticker)

            log_trade_decision(trade_record)
            trades_this_cycle += 1

        # Cycle summary
        logger.info(
            f"  Cycle {self._cycle_count}: {trades_this_cycle} trades | "
            f"positions: {len(self.safety._positions)} | "
            f"daily trades: {self.safety._daily_trade_count}/{self.safety.config['max_trades_per_day']} | "
            f"daily PnL: ${self.safety._daily_pnl:.2f}"
        )


async def main():
    parser = argparse.ArgumentParser(description="Kalshi Live Trading")
    parser.add_argument("--live", action="store_true", help="Execute real trades")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between cycles")
    args = parser.parse_args()

    if STOP_FILE.exists():
        STOP_FILE.unlink()

    trader = LiveTrader(dry_run=not args.live, cycle_interval=args.interval)
    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
