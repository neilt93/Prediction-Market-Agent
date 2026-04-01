"""Forecaster service that produces structured probability estimates.

Uses an LLM (via HTTP API - compatible with Ollama, vLLM, or any OpenAI-compatible endpoint)
to generate forecasts from market data, rules, and evidence.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import httpx
import structlog

from pydantic import BaseModel

logger = structlog.get_logger()


class ForecastOutput(BaseModel):
    """Structured forecast from the LLM."""
    raw_probability: float
    confidence: float
    abstain: bool = False
    reasoning_summary: str = ""
    supporting_factors: list[str] = []
    counterarguments: list[str] = []
    what_would_change_mind: list[str] = []


FORECAST_SYSTEM_PROMPT = """You are a specialist prediction market forecaster. You analyze market data, resolution rules, and evidence to produce calibrated probability estimates.

RULES:
1. Output ONLY valid JSON matching the schema below. No other text.
2. Your probability should reflect your genuine best estimate, not the market price.
3. Set abstain=true if you cannot make a meaningful estimate (ambiguous rules, no evidence, etc.)
4. Confidence reflects how certain you are of your probability estimate (0.0-1.0).
5. Be well-calibrated: when you say 70%, the event should happen ~70% of the time.

Output JSON schema:
{
  "raw_probability": <float 0.0-1.0>,
  "confidence": <float 0.0-1.0>,
  "abstain": <bool>,
  "reasoning_summary": "<1-2 sentence summary>",
  "supporting_factors": ["<factor1>", "<factor2>"],
  "counterarguments": ["<counter1>", "<counter2>"],
  "what_would_change_mind": ["<trigger1>", "<trigger2>"]
}"""


class Forecaster:
    """LLM-based forecaster that calls an OpenAI-compatible API."""

    def __init__(
        self,
        api_url: str = "http://localhost:11434/v1",
        model: str = "llama3.1:8b",
        api_key: str = "ollama",
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=120.0)

    async def close(self) -> None:
        await self._client.aclose()

    def _build_user_prompt(
        self,
        title: str,
        rules_text: str | None,
        parsed_rules: dict | None,
        evidence_snippets: list[str],
        market_price: float | None,
        time_to_close_hours: float | None,
    ) -> str:
        parts = [f"## Market\n{title}"]

        if rules_text:
            parts.append(f"## Resolution Rules\n{rules_text}")

        if parsed_rules:
            parts.append(f"## Parsed Rules\n{json.dumps(parsed_rules, indent=2, default=str)}")

        if market_price is not None:
            parts.append(f"## Current Market Price\nYES: {market_price:.2f}")

        if time_to_close_hours is not None:
            if time_to_close_hours < 24:
                parts.append(f"## Time Remaining\n{time_to_close_hours:.1f} hours")
            else:
                parts.append(f"## Time Remaining\n{time_to_close_hours/24:.1f} days")

        if evidence_snippets:
            evidence_text = "\n".join(f"- {s}" for s in evidence_snippets[:10])
            parts.append(f"## Evidence\n{evidence_text}")

        parts.append("\nProvide your probability estimate as JSON.")
        return "\n\n".join(parts)

    async def forecast(
        self,
        title: str,
        rules_text: str | None = None,
        parsed_rules: dict | None = None,
        evidence_snippets: list[str] | None = None,
        market_price: float | None = None,
        time_to_close_hours: float | None = None,
    ) -> ForecastOutput:
        user_prompt = self._build_user_prompt(
            title=title,
            rules_text=rules_text,
            parsed_rules=parsed_rules,
            evidence_snippets=evidence_snippets or [],
            market_price=market_price,
            time_to_close_hours=time_to_close_hours,
        )

        try:
            resp = await self._client.post(
                f"{self.api_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": FORECAST_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1024,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            # Parse JSON from response (handle markdown code blocks)
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0]

            forecast_data = json.loads(content)
            output = ForecastOutput.model_validate(forecast_data)

            # Clamp probability
            output.raw_probability = max(0.01, min(0.99, output.raw_probability))
            output.confidence = max(0.0, min(1.0, output.confidence))

            return output

        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error("forecast_failed", error=str(e), title=title[:80])
            return ForecastOutput(
                raw_probability=0.5,
                confidence=0.0,
                abstain=True,
                reasoning_summary=f"Forecast failed: {e}",
            )

    def to_db_dict(
        self,
        output: ForecastOutput,
        market_id: Any,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        return {
            "market_id": market_id,
            "ts": datetime.now(tz=timezone.utc),
            "model_name": model_name or self.model,
            "adapter_name": None,
            "prompt_version": "v1",
            "raw_probability": output.raw_probability,
            "confidence": output.confidence,
            "abstain_flag": output.abstain,
            "reasoning_summary": output.reasoning_summary,
            "counterfactual_trigger_json": {
                "what_would_change_mind": output.what_would_change_mind,
            },
        }
