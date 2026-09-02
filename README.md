# Prediction Market Agent

Local-first forecasting and trading agent for prediction markets (Kalshi, Polymarket). An open-weights LLM (Qwen 2.5 14B via Ollama) produces probability forecasts, a LightGBM model calibrates them, and an execution policy decides whether the edge over the market price is worth trading. Everything runs on a single machine: Postgres, Redis, Ollama, one GPU.

## Architecture

```
Market -> RuleParser -> EvidenceRetriever -> LLM Forecaster -> Calibrator -> ExecutionPolicy
                                                                   |
                                                            Postmortem (after resolution)
                                                                   |
                                                            Calibrator retrain
```

| Stage | What it does |
|---|---|
| Rule parsing | Extracts entity, threshold, and an ambiguity score from market rules |
| Evidence retrieval | Google News, DuckDuckGo, CoinGecko, Wikipedia. In backtests, only evidence with an explicit timestamp at or before the forecast time is kept |
| Forecasting | Qwen 2.5 14B decomposes the question into sub-questions, answers them, then forecasts. If the result lands in the uncertain band (0.35 to 0.65), a devil's advocate pass argues the opposite case and the two estimates are blended |
| Calibration | LightGBM maps 13 features (raw probability, market price, spread, time to close, evidence stats, etc.) to a calibrated probability. Per-niche calibrators for geopolitics, crypto, and tech, with a general fallback |
| Position sizing | Trades only when predicted edge exceeds the model's own Expected Calibration Error, measured on recent resolved forecasts. High ECE throttles trading automatically |
| Execution | Kalshi order placement with a safety layer: dry-run default, daily loss cap, position limits, spread and liquidity gates, and a `STOP_TRADING` kill-switch file |
| Learning loop | After resolution, each forecast gets a postmortem (Brier, log loss, error class) and the calibrator retrains on accumulated postmortems |

There is also a research package (`packages/diffusion/`) implementing Conditional Flow Matching as an alternative calibrator that outputs a probability distribution instead of a point estimate.

## Results

A multi-pass backtest (commit `2919f72`, April 2026) ran Qwen 2.5 14B over 229 resolved markets in 6 passes. Recorded results:

| Metric | Value | What it actually measures |
|---|---|---|
| Directional accuracy | 80% | Raw LLM forecast on the correct side of 0.5 |
| Calibrator Brier | 0.0033 | In-sample training Brier. See caveats |
| Best niche | Tech: 100% acc, 0.007 Brier | Small per-niche sample sizes |
| Simulated PnL | -$1.08 | Negative. The market was more accurate than the agent |

### Caveats, in plain terms

These numbers overstate skill. Read them with the following in mind:

- **The Brier 0.0033 is not out-of-sample.** At that commit the calibrator (unregularized LightGBM, 31 leaves, 200 rounds, no validation split) was trained on all 229 forecasts and scored on the same 229 rows, with the market price as an input feature. That is a training-set fit metric, not a forecast metric. The calibrator was later regularized (7 leaves, L2, early stopping) with a temporal train/validation split, and now reports validation Brier; no comparable headline number has been recorded since.
- **The LLM sees the market price.** The current market price is in the forecast prompt, so directional accuracy partly reflects reading the market, not beating it.
- **Forecasts were made 2 to 24 hours before market close** on already-resolved markets. Prices that close to resolution are often near 0 or 1, which makes both accuracy and Brier look strong for any method that tracks price.
- **Evidence leakage was possible in that run.** The retriever at that commit included undated live search results (DuckDuckGo, Wikipedia, CoinGecko) fetched at backtest time, after the markets had resolved. Undated sources are now excluded in backtests, but the 229-market run predates that fix.
- **The honest baseline comparison is in the harness.** `run_multi_backtest.py` scores "always predict the market price" alongside the agent, per market and per niche. In the recorded run the market's Brier was better overall, which is why simulated PnL was negative.
- **Not reproducible from this repo alone.** Per-market results live in a local Postgres database that is not committed. The numbers above come from the commit log and the committed calibrator artifact, not from checked-in output files.

The takeaway is the negative result: a 14B local model with retrieval reads near-close markets well but did not beat the market price on edge. The calibration, ECE gating, and postmortem loop exist to measure that honestly rather than hide it.

## Quick Start

```bash
uv sync
docker compose -f infra/docker-compose.yml up -d
cd infra/migrations && uv run alembic upgrade head
uv run uvicorn api.main:app --reload
```

## Running

```bash
# Backtest (requires Postgres with ingested resolved markets)
python run_backtest.py
python run_multi_backtest.py --passes 4 --per-pass 50

# Live trading loop (dry-run by default; touch STOP_TRADING to halt)
python run_live.py

# Diffusion calibrator experiment
python run_diffusion_mve.py
```

## Repo layout

```
packages/
  market_ingest/  # Polymarket + Kalshi API clients
  rules/          # Market rule parsing
  evidence/       # Evidence retrieval with temporal cutoffs
  forecasting/    # LLM forecaster (decomposition + debate)
  calibration/    # LightGBM calibrator, ECE, niche router
  training/       # Calibrator retraining from postmortems
  execution/      # Trade policy + Kalshi order placement
  diffusion/      # Conditional Flow Matching calibrator (research)
apps/             # FastAPI service + Celery worker
infra/            # Docker compose, Alembic migrations
```

## Limitations

- Backtests select markets that already resolved and forecast near close; there is no walk-forward evaluation at longer horizons yet.
- Calibrator training data is small (tens to low hundreds of samples per niche).
- Live trading has only been run in dry-run and small-stake modes.
