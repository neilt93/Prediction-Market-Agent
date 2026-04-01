from shared.config import BaseAppSettings


class WorkerSettings(BaseAppSettings):
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    ingest_interval_seconds: int = 300
    snapshot_interval_seconds: int = 60
    resolution_interval_seconds: int = 600

    # Kalshi
    kalshi_env: str = "demo"
    kalshi_api_key_id: str = ""
    kalshi_private_key_path: str = ""
