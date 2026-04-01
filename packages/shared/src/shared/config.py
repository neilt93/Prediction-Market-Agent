from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseAppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = (
        "postgresql+asyncpg://predagent:predagent_dev@localhost:5433/predagent"
    )
    database_url_sync: str = (
        "postgresql+psycopg2://predagent:predagent_dev@localhost:5433/predagent"
    )
    redis_url: str = "redis://localhost:6379/0"
    debug: bool = False
    log_level: str = "INFO"
