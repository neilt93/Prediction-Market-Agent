import enum


class MarketPlatform(str, enum.Enum):
    KALSHI = "kalshi"
    POLYMARKET = "polymarket"


class MarketStatus(str, enum.Enum):
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    RESOLVED = "resolved"
    DISPUTED = "disputed"


class MarketType(str, enum.Enum):
    BINARY = "binary"
    MULTI = "multi"


class SourceType(str, enum.Enum):
    OFFICIAL = "official"
    NEWS = "news"
    ONCHAIN = "onchain"
    SOCIAL = "social"
    FORUM = "forum"


class OrderSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, enum.Enum):
    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(str, enum.Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class Environment(str, enum.Enum):
    DEMO = "demo"
    PROD = "prod"


class ErrorBucket(str, enum.Enum):
    RULE_MISREAD = "rule_misread"
    STALE_EVIDENCE = "stale_evidence"
    WEAK_SOURCE_OVERWEIGHTED = "weak_source_overweighted"
    MISSED_OFFICIAL_SOURCE = "missed_official_source"
    BAD_CALIBRATION = "bad_calibration"
    OVERTRADED_SMALL_EDGE = "overtraded_small_edge"
    SLIPPAGE_UNDERMODELED = "slippage_undermodeled"
    MISSED_MARKET_REGIME_SHIFT = "missed_market_regime_shift"
    GOOD_FORECAST_BAD_TIMING = "good_forecast_bad_timing"
    AMBIGUOUS_SHOULD_HAVE_ABSTAINED = "ambiguous_should_have_abstained"
