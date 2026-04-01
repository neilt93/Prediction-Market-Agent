"""Rule parser that converts market text into structured resolution objects."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

import structlog

logger = structlog.get_logger()


class ParsedRule(BaseModel):
    """Structured representation of market resolution rules."""
    target_event: str = ""
    entity: str | None = None
    threshold_value: str | None = None
    comparator: str | None = None  # >, <, >=, <=, ==
    deadline_ts: datetime | None = None
    timezone: str = "UTC"
    source_of_truth: str | None = None
    hard_constraints: list[str] = []
    ambiguous_phrases: list[str] = []
    ambiguity_score: float = 0.0


# Common patterns for rule extraction
THRESHOLD_PATTERN = re.compile(
    r"(above|below|over|under|at least|more than|less than|exceed)\s+"
    r"\$?([\d,]+\.?\d*[kmb]?)",
    re.IGNORECASE,
)

DEADLINE_PATTERN = re.compile(
    r"(by|before|on|at)\s+"
    r"(\d{1,2}[:/]\d{2}\s*(AM|PM)?\s*(ET|PT|CT|MT|UTC)?)\s*"
    r"(on\s+)?(January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    r"(\d{1,2}),?\s*(\d{4})?",
    re.IGNORECASE,
)

DATE_PATTERN = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    r"(\d{1,2}),?\s*(\d{4})?",
    re.IGNORECASE,
)

COMPARATOR_MAP = {
    "above": ">",
    "over": ">",
    "more than": ">",
    "exceed": ">",
    "at least": ">=",
    "below": "<",
    "under": "<",
    "less than": "<",
}

AMBIGUOUS_TRIGGERS = [
    "approximately",
    "around",
    "roughly",
    "about",
    "may",
    "might",
    "could",
    "possibly",
    "generally",
    "typically",
    "usually",
    "in general",
    "subject to",
    "at the discretion",
]


class RuleParser:
    """Parses market resolution rules into structured objects."""

    def parse(self, title: str, rules_text: str | None = None) -> ParsedRule:
        text = f"{title} {rules_text or ''}"
        result = ParsedRule(target_event=title)

        # Extract threshold
        threshold_match = THRESHOLD_PATTERN.search(text)
        if threshold_match:
            direction = threshold_match.group(1).lower()
            value = threshold_match.group(2).replace(",", "")
            result.comparator = COMPARATOR_MAP.get(direction, ">")
            result.threshold_value = value

        # Extract entity (crypto tickers, company names, etc.)
        result.entity = self._extract_entity(title)

        # Extract deadline
        date_match = DATE_PATTERN.search(text)
        if date_match:
            try:
                month_str = date_match.group(1)
                day = int(date_match.group(2))
                year = int(date_match.group(3)) if date_match.group(3) else datetime.now().year
                month = datetime.strptime(month_str, "%B").month
                result.deadline_ts = datetime(year, month, day, tzinfo=timezone.utc)
            except (ValueError, TypeError):
                pass

        # Extract source of truth
        result.source_of_truth = self._extract_source(text)

        # Check ambiguity
        text_lower = text.lower()
        result.ambiguous_phrases = [
            phrase for phrase in AMBIGUOUS_TRIGGERS
            if phrase in text_lower
        ]
        result.ambiguity_score = min(1.0, len(result.ambiguous_phrases) * 0.2)

        return result

    @staticmethod
    def _extract_entity(title: str) -> str | None:
        crypto_pattern = re.compile(r"\b(BTC|ETH|SOL|DOGE|XRP|ADA|DOT|MATIC|AVAX|LINK)\b", re.IGNORECASE)
        match = crypto_pattern.search(title)
        if match:
            return match.group(1).upper()
        return None

    @staticmethod
    def _extract_source(text: str) -> str | None:
        source_patterns = [
            (r"according to\s+(\w+[\w\s]*)", None),
            (r"as reported by\s+(\w+[\w\s]*)", None),
            (r"CoinGecko|CoinMarketCap|Bloomberg|Reuters|Yahoo Finance", None),
            (r"official\s+(\w+)", None),
        ]
        for pattern, _ in source_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        return None

    def to_db_dict(self, parsed: ParsedRule, market_id: Any) -> dict[str, Any]:
        """Convert ParsedRule to dict for DB insertion."""
        return {
            "market_id": market_id,
            "parsed_json": parsed.model_dump(),
            "deadline_ts": parsed.deadline_ts,
            "timezone": parsed.timezone,
            "comparator": parsed.comparator,
            "threshold_value": parsed.threshold_value,
            "entity": parsed.entity,
            "source_of_truth": parsed.source_of_truth,
            "ambiguity_score": parsed.ambiguity_score,
            "parser_version": "v1",
        }
