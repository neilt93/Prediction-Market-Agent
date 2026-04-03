# Prediction Market Agent

## Project Structure

This is a monorepo using `uv` workspaces. All packages live in `packages/`:

```
packages/
  shared/         # Config (BaseAppSettings), DB helpers, logging
  schemas/        # SQLAlchemy ORM models (Market, Forecast, Evidence, Postmortem, etc.)
  market_ingest/  # Polymarket (Gamma + CLOB) and Kalshi API clients + mappers
  rules/          # Market rule parsing (entity extraction, ambiguity scoring)
  evidence/       # Evidence retrieval (DuckDuckGo, Google News, CoinGecko, Wikipedia)
  forecasting/    # LLM forecaster (Qwen 2.5 14B via Ollama OpenAI-compat API)
  calibration/    # LightGBM calibrator (13 features -> calibrated probability)
  training/       # Calibrator retraining from postmortems
  execution/      # Trade execution policy + Kalshi order placement
  backtest/       # Backtest utilities
  diffusion/      # [NEW] Conditional Flow Matching for probabilistic forecasting (research)
```

## Database

PostgreSQL 16 + pgvector at `localhost:5433` (Docker via `infra/docker-compose.yml`).
User: `predagent`, DB: `predagent`. Migrations via Alembic in `infra/migrations/`.

Key tables: `markets`, `market_snapshots`, `market_outcomes`, `forecasts`, `forecast_features`,
`calibrated_forecasts`, `postmortems`, `evidence_items`, `rule_parses`, `orders`, `positions`.

## The Forecasting Pipeline

```
Market -> RuleParser -> EvidenceRetriever -> LLM Forecaster -> Calibrator -> ExecutionPolicy
                                                                   |
                                                            Postmortem (after resolution)
                                                                   |
                                                            Calibrator Retrain
```

1. Fetch market from Polymarket/Kalshi
2. Parse rules: extract entity, threshold, ambiguity_score
3. Gather evidence: news, prices, Wikipedia (respects temporal cutoffs for backtesting)
4. LLM forecast: Qwen 2.5 14B produces raw_probability + confidence + reasoning
5. Calibration: LightGBM takes 13 features, outputs calibrated_probability + edge
6. Execution: gates on edge, confidence, spread, liquidity -> trade decision
7. Post-resolution: Brier score, log loss, error classification, calibrator retraining

## The Diffusion Research Package (packages/diffusion/)

A separate Claude instance built this as a research exploration. It implements
**Conditional Flow Matching (CFM)** as an alternative to the LightGBM calibrator.

### What it does

Instead of LightGBM producing a point estimate, CFM learns a velocity field that
transports Gaussian noise to the target probability distribution. At inference,
it runs a 5-step ODE to generate 32 probability samples, giving both a point
estimate (mean) and uncertainty bounds (10th/90th percentile).

### Architecture

**Phase 1 (MVE, built):**
- `DenoisingMLP`: 27K-param MLP. Input = [noisy_logit (1D), time_embedding (64D), features (13D)].
  Two hidden layers (128 units), LayerNorm, SiLU, Dropout(0.2). Predicts velocity field.
- Operates in logit space (unbounded), sigmoid maps back to probability.
- Same 13 FEATURE_COLUMNS as the LightGBM calibrator.

**Phase 2 (designed, not wired up yet):**
- `FiLMDenoisingMLP`: 611K params. 2D output (probability + uncertainty).
  FiLM conditioning from a 256D condition vector.
- `ConditionEncoder`: merges evidence_embedding (384D from pgvector),
  title_embedding (384D), and 13 numerical features into 256D condition vector.

### Key files

- `model.py` -- Neural network components (DenoisingMLP, FiLMDenoisingMLP, ConditionEncoder)
- `flow_matching.py` -- CFM training loop, ODE solver, loss computation
- `dataset.py` -- Data loading from DB, temporal CV splits, counterfactual augmentation
- `evaluate.py` -- Brier score, reliability diagrams, bootstrap significance tests
- `inference.py` -- `DiffusionCalibrator` class (drop-in for `Calibrator`)
- `run_diffusion_mve.py` -- MVE experiment script (root level)

### Interface

`DiffusionCalibrator.predict(features, market_price)` returns the same
`CalibratedOutput(calibrated_probability, predicted_edge_bps, uncertainty_low, uncertainty_high)`
as the LightGBM `Calibrator`. It's a drop-in replacement.

### Do not modify

The diffusion package is a research exploration. If you need to change the calibration
interface, update `calibration/calibrator.py` and the diffusion package will adapt
(it duplicates `FEATURE_COLUMNS` and `CalibratedOutput` to avoid a hard dependency on lightgbm).

## Calibrator Changes (Phase 0)

The LightGBM calibrator was regularized to fix overfitting:
- `num_leaves`: 31 -> 7
- `num_boost_round`: 200 -> 100 (with early stopping, patience 20)
- Added `min_data_in_leaf=15`, `lambda_l2=5.0`
- Added temporal train/val split (last 20% as validation set)
- Now reports both train and val Brier scores (previously only train)
- Version string changed from `v1-lgb-*` to `v2-lgb-*`

The `train()` method signature added an optional `val_fraction` parameter (default 0.2).
Data must be passed in temporal order (oldest first) for the split to be correct.

## Environment

- Python 3.12+, uv workspaces
- PostgreSQL 16 + pgvector (Docker)
- Redis 7 (Celery broker)
- Ollama for LLM serving (localhost:11434)
- RTX 4080 (PyTorch 2.7.1 + CUDA 12.8)

## Running

```bash
# Start infra
docker compose -f infra/docker-compose.yml up -d

# Backtest
python run_backtest.py
python run_multi_backtest.py

# Diffusion MVE (requires DB + resolved markets)
python run_diffusion_mve.py
python run_diffusion_mve.py --augment  # with counterfactual data augmentation
```
