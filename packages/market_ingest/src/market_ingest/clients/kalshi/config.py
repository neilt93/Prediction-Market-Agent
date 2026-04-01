import enum


class KalshiEnvironment(str, enum.Enum):
    DEMO = "demo"
    PROD = "prod"


KALSHI_BASE_URLS = {
    KalshiEnvironment.DEMO: "https://api.elections.kalshi.com/trade-api/v2",
    KalshiEnvironment.PROD: "https://api.elections.kalshi.com/trade-api/v2",
}
