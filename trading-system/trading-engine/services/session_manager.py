"""
session_manager.py — UTC-based session detection. Single source of truth.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd


# Symbols that trade in Asian session (2–6 UTC). Matches run_backtest.py default.
# Overrideable via ASIAN_SESSION_SYMBOLS env var (comma-separated).
ASIAN_SESSION_SYMBOLS: frozenset[str] = frozenset(
    s.strip().upper()
    for s in os.getenv(
        "ASIAN_SESSION_SYMBOLS",
        "EURUSD,USDJPY,EURJPY,GBPJPY,GBPUSD",
    ).split(",")
    if s.strip()
)

# Hard close times per trader_id: (hour, minute)
_HARD_CLOSE = {
    "trader_1": (17, 0),
    "trader_2": (18, 0),
    "trader_3": (12, 0),
    "trader_4": (23, 59),
    "trader_5": (6, 45),
}


class SessionManager:
    """UTC-based session detection. All traders delegate here."""

    def get_current_session(self, dt: Optional[datetime] = None) -> str:
        """Returns: 'ASIAN' | 'LONDON' | 'NY' | 'DEAD' | 'INACTIVE'"""
        now = dt or datetime.now(timezone.utc)
        h = now.hour

        if 2 <= h < 6:
            return "ASIAN"
        elif 6 <= h < 12:
            return "LONDON"
        elif h == 12:
            return "DEAD"
        elif 13 <= h < 20:
            return "NY"
        else:
            return "INACTIVE"

    def is_dead_zone(self, dt: Optional[datetime] = None) -> bool:
        return self.get_current_session(dt) == "DEAD"

    def is_hard_close_time(self, trader_id: str, dt: Optional[datetime] = None) -> bool:
        """Returns True if this trader's hard close time has passed this session."""
        now = dt or datetime.now(timezone.utc)
        close = _HARD_CLOSE.get(trader_id)
        if close is None:
            return False
        close_h, close_m = close
        return (now.hour > close_h) or (now.hour == close_h and now.minute >= close_m)

    def get_session_open_price(
        self, df: pd.DataFrame, session: str
    ) -> Optional[float]:
        """Returns first close of given session on same date as df.index[-1]."""
        if df is None or len(df) == 0:
            return None
        if not hasattr(df.index, "hour"):
            return None

        session_hours = {
            "ASIAN": range(2, 6),
            "LONDON": range(6, 12),
            "NY": range(13, 20),
        }
        hours = session_hours.get(session.upper())
        if hours is None:
            return None

        today = df.index[-1].date()
        bars = df[(df.index.date == today) & (df.index.hour.isin(hours))]
        if len(bars) == 0:
            return None
        return float(bars["close"].iloc[0])

    def is_active_for_symbol(self, symbol: str, dt: Optional[datetime] = None) -> bool:
        """Returns True if this symbol should be traded at the given UTC time.

        Forex pairs in ASIAN_SESSION_SYMBOLS are active 2–20 UTC.
        All other symbols (gold, etc.) are active 6–20 UTC only.
        """
        now = dt or datetime.now(timezone.utc)
        h = now.hour
        if symbol.upper() in ASIAN_SESSION_SYMBOLS:
            return 2 <= h < 20
        return 6 <= h < 20

    def should_trade(self, trader_id: str, dt: Optional[datetime] = None) -> bool:
        """Combined session + dead zone check for a specific trader."""
        now = dt or datetime.now(timezone.utc)
        if self.is_dead_zone(now):
            return False
        if self.is_hard_close_time(trader_id, now):
            return False
        return True
