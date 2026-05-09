from __future__ import annotations

from typing import List

import pytz
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Broker ────────────────────────────────────────────────────────────────
    BROKER_TYPE: str = "capital"
    CAPITAL_API_KEY: str = ""
    CAPITAL_IDENTIFIER: str = ""
    CAPITAL_PASSWORD: str = ""
    CAPITAL_ENV: str = "demo"

    # ── Trading mode ──────────────────────────────────────────────────────────
    PAPER_TRADING: bool = True
    ML_ENABLED: bool = True
    RL_ENABLED: bool = False     # RL is dormant; isolated behind this flag
    SIMPLIFIED_ML_ENABLED: bool = False
    SIMPLIFIED_USE_QUALITY: bool = False

    # ── Symbols ───────────────────────────────────────────────────────────────
    TRADING_PAIRS: List[str] = ["XAUUSD", "EURUSD", "USDJPY", "EURJPY", "GBPJPY", "GBPUSD"]

    # ── Capital / account ─────────────────────────────────────────────────────
    ACCOUNT_BALANCE: float = 10000.0
    CAPITAL_PER_TRADER: float = 1.00

    # ── Risk management ───────────────────────────────────────────────────────
    RISK_PER_TRADE: float = 0.005           # 0.5% fixed fractional (no ML scaling)
    MAX_DAILY_LOSS_PCT: float = 0.02        # 2% daily circuit breaker
    MAX_WEEKLY_LOSS_PCT: float = 0.05       # 5% weekly circuit breaker
    MAX_DRAWDOWN_PCT: float = 0.15          # 15% portfolio halt
    MAX_CONCURRENT_POSITIONS: int = 2       # max simultaneous open trades
    MAX_CORRELATED_POSITIONS: int = 1       # max trades in same currency-direction group
    MAX_CONSECUTIVE_LOSSES: int = 3         # losses in a row before cooldown
    CONSECUTIVE_LOSS_COOLDOWN_BARS: int = 10  # bars to wait after streak

    # ── ML signal thresholds ──────────────────────────────────────────────────
    ML_DIRECTION_THRESHOLD: float = 0.65   # minimum P(direction)
    MIN_EXPECTED_R: float = 1.30           # minimum probability-weighted expected R
    MAX_UNCERTAINTY: float = 0.15          # maximum GRU predicted variance
    MIN_REWARD_TO_RISK: float = 1.50       # minimum geometric RR at entry
    ML_QUALITY_THRESHOLD_DEFAULT: float = 0.55
    RL_ACTION_CONFIDENCE_MIN: float = 0.45
    PM_MIN_CONFIDENCE: float = 0.50

    # ── ATR-based stop/target multipliers ─────────────────────────────────────
    ATR_STOP_MULTIPLIER: float = 1.5
    ATR_TARGET_MULTIPLIER: float = 2.5
    # Gold requires stricter volatility controls
    GOLD_ATR_STOP_MULTIPLIER: float = 2.0
    GOLD_ATR_TARGET_MULTIPLIER: float = 3.5

    # ── Maximum spread per symbol (pips) ──────────────────────────────────────
    MAX_SPREAD_EURUSD: float = 2.0
    MAX_SPREAD_GBPUSD: float = 2.5
    MAX_SPREAD_USDJPY: float = 2.0
    MAX_SPREAD_EURJPY: float = 2.5
    MAX_SPREAD_GBPJPY: float = 3.0
    MAX_SPREAD_XAUUSD: float = 30.0        # XAUUSD pips are 0.1; 30 pips = $3.0 spread

    # ── Fractional Kelly sizing (optional overlay — off by default) ───────────
    # Full Kelly maximises long-run geometric growth but is too aggressive.
    # Quarter-Kelly (0.25) reduces variance substantially at modest cost to growth.
    # NEVER scale size up with ML confidence — this is a fixed-fractional system.
    KELLY_ENABLED: bool = False
    KELLY_FRACTION: float = 0.25

    # ── Session windows (UTC hours, inclusive start / exclusive end) ───────────
    ASIAN_SESSION_START: int = 0
    ASIAN_SESSION_END: int = 7
    ASIAN_TRADE_START: int = 2
    ASIAN_TRADE_END_HARD: int = 6
    ASIAN_HARD_CLOSE_MINUTE: int = 45
    LONDON_SESSION_START: int = 7
    LONDON_SESSION_END: int = 12
    NY_SESSION_START: int = 13
    NY_SESSION_END: int = 18
    DEAD_ZONE_START: int = 12
    DEAD_ZONE_END: int = 13

    # ── Infrastructure ────────────────────────────────────────────────────────
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0
    HEALTH_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    RETRAIN_DAY: str = "sunday"
    RETRAIN_HOUR: int = 2
    NEWS_API_KEY: str = ""
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
UTC = pytz.utc
