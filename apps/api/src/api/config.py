from shared.config import BaseAppSettings


class ApiSettings(BaseAppSettings):
    app_name: str = "Prediction Agent API"
    api_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000"]
