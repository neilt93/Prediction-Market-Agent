# Prediction Market Agent

Local-first specialist forecasting and trading agent for prediction markets.

## Quick Start

```bash
uv sync
docker compose -f infra/docker-compose.yml up -d
cd infra/migrations && uv run alembic upgrade head
uv run uvicorn api.main:app --reload
```
