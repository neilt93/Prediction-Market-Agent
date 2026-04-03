"""DPO fine-tuning pipeline for forecasting models.

Generates preference pairs from resolved prediction markets:
- For each market, generate multiple reasoning traces at different temperatures
- After resolution, rank traces by proximity to actual outcome (Brier score)
- Create (chosen, rejected) pairs for DPO training
- Fine-tune with QLoRA on RTX 4080

Based on: "Outcome-based RL on Polymarket data" approach.
Uses TRL's DPOTrainer with QLoRA (rank 16, alpha 32, 4-bit).

Usage:
    python -m training.dpo_trainer --generate   # Generate preference pairs
    python -m training.dpo_trainer --train      # Run DPO fine-tuning
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

DATA_DIR = Path("data/dpo")
MODEL_OUTPUT_DIR = Path("data/models/dpo")


async def generate_preference_pairs(
    session: Any,
    forecaster_api_url: str = "http://localhost:11434/v1",
    forecaster_model: str = "qwen2.5:14b",
    n_traces: int = 4,
    max_markets: int = 200,
) -> list[dict]:
    """Generate multiple reasoning traces per resolved market, rank by Brier.

    For each market:
    1. Generate n_traces forecasts at temperature=1.0 (diverse reasoning)
    2. After resolution, compute Brier score for each trace
    3. Best trace = chosen, worst trace = rejected
    """
    import httpx
    from schemas.models.market import Market, MarketSnapshot, MarketOutcome
    from schemas.models.forecast import Forecast
    from sqlalchemy import exists

    markets = (
        session.query(Market, MarketOutcome)
        .join(MarketOutcome, MarketOutcome.market_id == Market.id)
        .filter(
            Market.status == "resolved",
            MarketOutcome.resolved_label.isnot(None),
            exists().where(MarketSnapshot.market_id == Market.id),
        )
        .limit(max_markets)
        .all()
    )

    logger.info(f"Generating preference pairs for {len(markets)} markets")

    client = httpx.AsyncClient(timeout=120.0)
    pairs = []

    system_prompt = (
        "You are a forecaster. Analyze this prediction market and estimate the probability "
        "of YES. Think step by step, then output your final probability as a number between 0 and 1. "
        "Format: first your reasoning, then on the last line: PROBABILITY: 0.XX"
    )

    for i, (market, outcome) in enumerate(markets):
        snapshot = (
            session.query(MarketSnapshot)
            .filter(MarketSnapshot.market_id == market.id)
            .order_by(MarketSnapshot.ts.desc())
            .first()
        )
        market_price = float(snapshot.mid_yes) if snapshot and snapshot.mid_yes else 0.5
        label = outcome.resolved_label

        user_prompt = f"Market: {market.title}\n"
        if market.rules_text:
            user_prompt += f"Rules: {market.rules_text[:500]}\n"
        user_prompt += f"Current market price: {market_price:.2f}\n"
        user_prompt += "What is the probability this resolves YES?"

        # Generate diverse traces
        traces = []
        for t in range(n_traces):
            try:
                resp = await client.post(
                    f"{forecaster_api_url}/chat/completions",
                    json={
                        "model": forecaster_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 1.0,  # High temp for diversity
                        "max_tokens": 512,
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]

                # Extract probability from response
                prob = _extract_probability(content)
                if prob is not None:
                    brier = (prob - label) ** 2
                    traces.append({
                        "content": content,
                        "probability": prob,
                        "brier": brier,
                    })
            except Exception as e:
                logger.debug(f"Trace generation failed: {e}")

        if len(traces) < 2:
            continue

        # Sort by Brier score (best first)
        traces.sort(key=lambda t: t["brier"])
        chosen = traces[0]
        rejected = traces[-1]

        pair = {
            "prompt": user_prompt,
            "chosen": chosen["content"],
            "rejected": rejected["content"],
            "chosen_prob": chosen["probability"],
            "rejected_prob": rejected["probability"],
            "chosen_brier": chosen["brier"],
            "rejected_brier": rejected["brier"],
            "actual_label": label,
            "market_title": market.title[:100],
            "market_price": market_price,
        }
        pairs.append(pair)

        if (i + 1) % 25 == 0:
            logger.info(f"  Generated {len(pairs)} pairs from {i+1} markets")

    await client.aclose()

    # Save pairs
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "preference_pairs.jsonl"
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    logger.info(f"Generated {len(pairs)} preference pairs -> {output_path}")
    return pairs


def _extract_probability(text: str) -> float | None:
    """Extract probability from LLM response text."""
    import re
    # Look for "PROBABILITY: 0.XX" pattern
    match = re.search(r"PROBABILITY:\s*(0?\.\d+|1\.0?|0)", text, re.IGNORECASE)
    if match:
        return max(0.01, min(0.99, float(match.group(1))))
    # Fallback: look for any float between 0 and 1 near the end
    matches = re.findall(r"\b(0\.\d+)\b", text[-200:])
    if matches:
        return max(0.01, min(0.99, float(matches[-1])))
    return None


def setup_dpo_training(
    base_model: str = "Qwen/Qwen2.5-14B-Instruct",
    data_path: str = "data/dpo/preference_pairs.jsonl",
    output_dir: str = "data/models/dpo",
) -> dict[str, Any]:
    """Set up DPO training config. Returns config dict.

    Requires: pip install trl peft bitsandbytes
    Run on RTX 4080 with QLoRA (4-bit quantization).
    """
    config = {
        "base_model": base_model,
        "data_path": data_path,
        "output_dir": output_dir,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "quantization": "4bit",
        "learning_rate": 5e-5,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_length": 1024,
        "max_prompt_length": 512,
        "beta": 0.1,  # DPO beta parameter
        "fp16": True,
    }

    # Save config
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    config_path = Path(output_dir) / "dpo_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"DPO config saved to {config_path}")

    # Generate training script
    script = f'''"""Auto-generated DPO training script. Run with: uv run python {output_dir}/train_dpo.py"""
import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer
import torch

# Load config
config = json.loads(Path("{config_path}").read_text())

# Load data
pairs = []
with open(config["data_path"]) as f:
    for line in f:
        pairs.append(json.loads(line))

dataset = Dataset.from_list([{{
    "prompt": p["prompt"],
    "chosen": p["chosen"],
    "rejected": p["rejected"],
}} for p in pairs])

print(f"Training on {{len(dataset)}} preference pairs")

# Load model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    config["base_model"],
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
tokenizer.pad_token = tokenizer.eos_token

# LoRA config
lora_config = LoraConfig(
    r=config["lora_r"],
    lora_alpha=config["lora_alpha"],
    lora_dropout=config["lora_dropout"],
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# DPO training config
training_args = DPOConfig(
    output_dir=config["output_dir"],
    num_train_epochs=config["num_train_epochs"],
    per_device_train_batch_size=config["per_device_train_batch_size"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    learning_rate=config["learning_rate"],
    beta=config["beta"],
    max_length=config["max_length"],
    max_prompt_length=config["max_prompt_length"],
    fp16=config["fp16"],
    logging_steps=10,
    save_steps=50,
    remove_unused_columns=False,
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
)

print("Starting DPO training...")
trainer.train()
trainer.save_model(config["output_dir"] + "/final")
print(f"Model saved to {{config['output_dir']}}/final")
'''

    script_path = Path(output_dir) / "train_dpo.py"
    script_path.write_text(script)
    logger.info(f"DPO training script saved to {script_path}")

    return config


if __name__ == "__main__":
    import argparse

    for pkg in ["shared", "schemas", "market_ingest", "rules", "forecasting",
                "calibration", "execution", "evidence"]:
        sys.path.insert(0, str(Path(__file__).parents[4] / "packages" / pkg / "src"))

    from shared.config import BaseAppSettings
    from shared.db import create_sync_engine, create_sync_session_factory

    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Generate preference pairs")
    parser.add_argument("--setup", action="store_true", help="Set up DPO training config+script")
    parser.add_argument("--max-markets", type=int, default=200)
    args = parser.parse_args()

    settings = BaseAppSettings()
    engine = create_sync_engine(settings.database_url_sync)
    session = create_sync_session_factory(engine)()

    if args.generate:
        asyncio.run(generate_preference_pairs(
            session,
            forecaster_api_url=settings.llm_api_url,
            forecaster_model=settings.llm_model,
            max_markets=args.max_markets,
        ))
    elif args.setup:
        setup_dpo_training()
    else:
        parser.print_help()

    session.close()
