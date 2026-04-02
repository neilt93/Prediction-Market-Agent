"""Forecaster service with sub-question decomposition and selective debate.

Pipeline:
1. Decompose market question into 3-5 researchable sub-questions
2. Gather evidence for each sub-question
3. Synthesize into a structured forecast
4. If confidence is uncertain (0.35-0.65), run devil's advocate re-evaluation
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
    decomposition_used: bool = False
    debate_used: bool = False


FORECAST_SYSTEM_PROMPT = """You are a superforecaster — one of the top 2% of predictors in the world. You analyze market data, resolution rules, and evidence to produce calibrated probability estimates.

CALIBRATION RULES:
1. Output ONLY valid JSON matching the schema below. No other text.
2. Your probability should reflect your genuine best estimate, not the market price.
3. First think of a probability RANGE, then pick your point estimate from within it.
4. Avoid extreme probabilities (>0.95 or <0.05) unless you have overwhelming evidence.
5. When you say 70%, the event should happen ~70% of the time. When uncertain, stay near 0.50.
6. Set abstain=true if rules are ambiguous or evidence is contradictory/absent.

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

DECOMPOSE_PROMPT = """Break this prediction market question into 3-5 specific, researchable sub-questions that would help determine the probability. Each sub-question should be independently answerable.

Question: {title}

Output ONLY a JSON array of strings, no other text:
["sub-question 1", "sub-question 2", "sub-question 3"]"""

DEBATE_PROMPT = """You are a devil's advocate analyst. A forecaster predicted {probability:.0%} probability for this market. Argue the OPPOSITE case as strongly as possible, then give your revised probability estimate.

Market: {title}
Original forecast: {probability:.0%} YES
Original reasoning: {reasoning}

Evidence:
{evidence}

After making the strongest case against the original forecast, output ONLY this JSON:
{{"revised_probability": <float 0.0-1.0>, "counterargument": "<your strongest argument>"}}"""


class Forecaster:
    """LLM-based forecaster with decomposition and selective debate."""

    def __init__(
        self,
        api_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:14b",
        api_key: str = "ollama",
        enable_decomposition: bool = True,
        enable_debate: bool = True,
        debate_uncertainty_range: tuple[float, float] = (0.35, 0.65),
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.enable_decomposition = enable_decomposition
        self.enable_debate = enable_debate
        self.debate_range = debate_uncertainty_range
        self._client = httpx.AsyncClient(timeout=120.0)

    async def close(self) -> None:
        await self._client.aclose()

    async def _llm_call(self, system: str, user: str, temperature: float = 0.3) -> str:
        """Make a single LLM call and return the content string."""
        resp = await self._client.post(
            f"{self.api_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": 1024,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        # Strip markdown code blocks
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0]
        return content.strip()

    async def decompose(self, title: str) -> list[str]:
        """Break a market question into researchable sub-questions."""
        try:
            content = await self._llm_call(
                "You break prediction questions into sub-questions. Output only JSON.",
                DECOMPOSE_PROMPT.format(title=title),
                temperature=0.5,
            )
            subs = json.loads(content)
            if isinstance(subs, list) and all(isinstance(s, str) for s in subs):
                return subs[:5]
        except Exception as e:
            logger.debug("decomposition_failed", error=str(e))
        return []

    async def _debate(
        self, title: str, probability: float, reasoning: str, evidence_text: str
    ) -> float | None:
        """Run devil's advocate and return revised probability, or None on failure."""
        try:
            content = await self._llm_call(
                "You are a devil's advocate analyst. Output only JSON.",
                DEBATE_PROMPT.format(
                    title=title,
                    probability=probability,
                    reasoning=reasoning,
                    evidence=evidence_text,
                ),
                temperature=0.4,
            )
            data = json.loads(content)
            revised = float(data.get("revised_probability", probability))
            return max(0.01, min(0.99, revised))
        except Exception as e:
            logger.debug("debate_failed", error=str(e))
        return None

    def _build_user_prompt(
        self,
        title: str,
        rules_text: str | None,
        parsed_rules: dict | None,
        evidence_snippets: list[str],
        sub_question_evidence: dict[str, list[str]] | None,
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

        # Sub-question research (richer than flat evidence)
        if sub_question_evidence:
            sq_parts = []
            for q, snippets in sub_question_evidence.items():
                if snippets:
                    sq_parts.append(f"**{q}**\n" + "\n".join(f"  - {s}" for s in snippets[:3]))
            if sq_parts:
                parts.append("## Research by Sub-Question\n" + "\n\n".join(sq_parts))

        # Flat evidence (fallback or additional)
        if evidence_snippets:
            evidence_text = "\n".join(f"- {s}" for s in evidence_snippets[:10])
            parts.append(f"## Additional Evidence\n{evidence_text}")

        parts.append("\nFirst consider a probability RANGE, then provide your point estimate as JSON.")
        return "\n\n".join(parts)

    async def forecast(
        self,
        title: str,
        rules_text: str | None = None,
        parsed_rules: dict | None = None,
        evidence_snippets: list[str] | None = None,
        sub_question_evidence: dict[str, list[str]] | None = None,
        market_price: float | None = None,
        time_to_close_hours: float | None = None,
    ) -> ForecastOutput:
        user_prompt = self._build_user_prompt(
            title=title,
            rules_text=rules_text,
            parsed_rules=parsed_rules,
            evidence_snippets=evidence_snippets or [],
            sub_question_evidence=sub_question_evidence,
            market_price=market_price,
            time_to_close_hours=time_to_close_hours,
        )

        try:
            content = await self._llm_call(FORECAST_SYSTEM_PROMPT, user_prompt)
            forecast_data = json.loads(content)
            output = ForecastOutput.model_validate(forecast_data)
            output.raw_probability = max(0.01, min(0.99, output.raw_probability))
            output.confidence = max(0.0, min(1.0, output.confidence))
            output.decomposition_used = sub_question_evidence is not None

            # Selective debate: if uncertain, run devil's advocate
            if (
                self.enable_debate
                and not output.abstain
                and self.debate_range[0] <= output.raw_probability <= self.debate_range[1]
            ):
                evidence_text = "\n".join(evidence_snippets or [])
                revised = await self._debate(
                    title, output.raw_probability, output.reasoning_summary, evidence_text
                )
                if revised is not None:
                    # Weighted average: 60% original, 40% debate
                    blended = 0.6 * output.raw_probability + 0.4 * revised
                    output.raw_probability = max(0.01, min(0.99, blended))
                    output.debate_used = True

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
            "prompt_version": "v2-decompose-debate",
            "raw_probability": output.raw_probability,
            "confidence": output.confidence,
            "abstain_flag": output.abstain,
            "reasoning_summary": output.reasoning_summary,
            "counterfactual_trigger_json": {
                "what_would_change_mind": output.what_would_change_mind,
                "decomposition_used": output.decomposition_used,
                "debate_used": output.debate_used,
            },
        }
