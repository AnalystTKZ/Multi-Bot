from __future__ import annotations

from typing import List

import pytz
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Broker
    BROKER_TYPE: str = "capital"
    CAPITAL_API_KEY: str = ""
    CAPITAL_IDENTIFIER: str = ""
    CAPITAL_PASSWORD: str = ""
    CAPITAL_ENV: str = "demo"

    # Trading mode
    PAPER_TRADING: bool = True
    ML_ENABLED: bool = True

    # Symbols (matches existing system exactly)
    TRADING_PAIRS: List[str] = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"]

    # Capital allocation for the unified ML trader (% of total account)
    ACCOUNT_BALANCE: float = 10000.0       # default; overridden by broker at startup
    CAPITAL_PER_TRADER: float = 1.00       # unified trader uses the full account by default
    RISK_PER_TRADE: float = 0.01           # 1% per trade

    # Risk limits
    MAX_DAILY_LOSS_PCT: float = 0.02
    MAX_DRAWDOWN_PCT: float = 0.08
    MAX_CONCURRENT_POSITIONS: int = 2

    # ML thresholds
    ML_QUALITY_THRESHOLD_DEFAULT: float = 0.55
    ML_DIRECTION_THRESHOLD: float = 0.58
    RL_ACTION_CONFIDENCE_MIN: float = 0.45

    # Session windows (UTC hours, inclusive start / exclusive end)
    ASIAN_SESSION_START: int = 0
    ASIAN_SESSION_END: int = 7
    ASIAN_TRADE_START: int = 2
    ASIAN_TRADE_END_HARD: int = 6
    ASIAN_HARD_CLOSE_MINUTE: int = 45      # 06:45 UTC
    LONDON_SESSION_START: int = 7
    LONDON_SESSION_END: int = 12
    NY_SESSION_START: int = 13
    NY_SESSION_END: int = 18
    DEAD_ZONE_START: int = 12
    DEAD_ZONE_END: int = 13

    # Redis
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0

    # Health server
    HEALTH_PORT: int = 8000

    # Logging
    LOG_LEVEL: str = "INFO"

    # Retrain schedule
    RETRAIN_DAY: str = "sunday"
    RETRAIN_HOUR: int = 2

    # News API (optional)
    NEWS_API_KEY: str = ""

    # Telegram (optional)
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
UTC = pytz.utc
