from celery import Celery

from worker.config import WorkerSettings

settings = WorkerSettings()

celery_app = Celery(
    "prediction_agent",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

celery_app.autodiscover_tasks(["worker.tasks"])

celery_app.conf.beat_schedule = {
    "discover-markets": {
        "task": "worker.tasks.ingest.discover_markets",
        "schedule": settings.ingest_interval_seconds,
    },
    "snapshot-markets": {
        "task": "worker.tasks.ingest.snapshot_markets",
        "schedule": settings.snapshot_interval_seconds,
    },
    "detect-resolutions": {
        "task": "worker.tasks.ingest.detect_resolutions",
        "schedule": settings.resolution_interval_seconds,
    },
}
