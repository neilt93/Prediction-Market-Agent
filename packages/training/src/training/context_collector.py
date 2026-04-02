"""Frozen context collector for DPO training data.

Scrapes unresolved Kalshi markets, saves current web search results at time of
scraping, and packages as (question, frozen_context, resolution) once resolved.

This avoids information leakage and creates clean training data for future
DPO/GRPO fine-tuning runs.

Usage:
    collector = FrozenContextCollector(db_session)
    await collector.collect_contexts(kalshi_client, evidence_retriever)
    # Later, after markets resolve:
    pairs = collector.build_dpo_pairs()
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy.orm import Session

logger = structlog.get_logger()

CONTEXT_DIR = Path("data/frozen_contexts")


class FrozenContextCollector:
    """Collects and stores frozen evidence contexts for future training."""

    def __init__(self, db_session: Session) -> None:
        self.db = db_session
        CONTEXT_DIR.mkdir(parents=True, exist_ok=True)

    async def collect_contexts(
        self,
        kalshi_client: Any,
        evidence_retriever: Any,
        rule_parser: Any,
        max_markets: int = 50,
    ) -> int:
        """Scrape current Kalshi markets and freeze their evidence context."""
        from schemas.models.market import Market

        count = 0
        async for event in kalshi_client.get_events(status="open", limit=100):
            if not event.markets:
                continue
            for km in event.markets:
                if km.status != "active" or count >= max_markets:
                    continue

                title = km.title or event.title
                ticker = km.ticker

                # Skip if already collected
                context_path = CONTEXT_DIR / f"{ticker}.json"
                if context_path.exists():
                    continue

                # Gather evidence NOW (frozen at this point in time)
                parsed = rule_parser.parse(title, km.rules_primary)
                bundle = await evidence_retriever.gather(title, parsed.entity)

                # Save frozen context
                context = {
                    "ticker": ticker,
                    "title": title,
                    "event_ticker": event.event_ticker,
                    "rules_text": km.rules_primary,
                    "collected_at": datetime.now(tz=timezone.utc).isoformat(),
                    "close_time": km.close_time,
                    "market_price": {
                        "yes_bid": km.yes_bid_dollars,
                        "yes_ask": km.yes_ask_dollars,
                    },
                    "parsed_rules": parsed.model_dump(),
                    "evidence_snippets": bundle.snippets,
                    "evidence_sources": bundle.sources,
                    "evidence_count": bundle.count,
                    "resolution": None,  # Filled in later
                }

                context_path.write_text(json.dumps(context, indent=2, default=str))
                count += 1

        logger.info("frozen_contexts_collected", count=count)
        return count

    def resolve_contexts(self, kalshi_client_sync: Any = None) -> int:
        """Check collected contexts against resolved markets and fill in outcomes."""
        resolved = 0
        for path in CONTEXT_DIR.glob("*.json"):
            context = json.loads(path.read_text())
            if context.get("resolution") is not None:
                continue

            # Check if market resolved in DB
            from schemas.models.market import Market, MarketOutcome
            ticker = context["ticker"]

            # Look for resolution in outcomes table
            market = (
                self.db.query(Market)
                .filter(Market.platform == "kalshi", Market.platform_market_id == ticker)
                .first()
            )
            if not market:
                continue

            outcome = (
                self.db.query(MarketOutcome)
                .filter(MarketOutcome.market_id == market.id)
                .first()
            )
            if outcome and outcome.resolved_label is not None:
                context["resolution"] = {
                    "label": outcome.resolved_label,
                    "resolved_at": outcome.resolved_at.isoformat() if outcome.resolved_at else None,
                    "notes": outcome.resolution_notes,
                }
                path.write_text(json.dumps(context, indent=2, default=str))
                resolved += 1

        logger.info("frozen_contexts_resolved", count=resolved)
        return resolved

    def build_dpo_pairs(self) -> list[dict[str, Any]]:
        """Build DPO preference pairs from resolved frozen contexts.

        For each resolved market:
        - Generate the 'chosen' response (close to actual outcome)
        - Generate the 'rejected' response (far from actual outcome)
        """
        pairs = []
        for path in CONTEXT_DIR.glob("*.json"):
            context = json.loads(path.read_text())
            if context.get("resolution") is None:
                continue

            label = context["resolution"]["label"]
            title = context["title"]
            evidence = "\n".join(f"- {s}" for s in context.get("evidence_snippets", []))

            # Chosen: reasoning that leads to correct answer
            correct_prob = 0.90 if label == 1 else 0.10
            chosen_reasoning = (
                f"Based on the evidence, the probability of '{title}' resolving YES "
                f"is approximately {correct_prob:.0%}."
            )

            # Rejected: reasoning that leads to wrong answer
            wrong_prob = 0.10 if label == 1 else 0.90
            rejected_reasoning = (
                f"Based on the evidence, the probability of '{title}' resolving YES "
                f"is approximately {wrong_prob:.0%}."
            )

            pairs.append({
                "prompt": f"Market: {title}\nRules: {context.get('rules_text', '')}\nEvidence:\n{evidence}",
                "chosen": chosen_reasoning,
                "rejected": rejected_reasoning,
                "label": label,
                "ticker": context["ticker"],
            })

        logger.info("dpo_pairs_built", count=len(pairs))
        return pairs

    def export_for_training(self, output_path: str = "data/dpo_training.jsonl") -> int:
        """Export DPO pairs as JSONL for training with TRL."""
        pairs = self.build_dpo_pairs()
        with open(output_path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
        logger.info("dpo_data_exported", path=output_path, count=len(pairs))
        return len(pairs)
