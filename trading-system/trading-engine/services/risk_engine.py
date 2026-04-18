"""
risk_engine.py — System-wide risk controls (complete replacement).
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RiskEngine:
    """System-wide risk controls. All checks run before order submission."""

    def __init__(self, settings):
        self._settings = settings

    def check_pre_trade(
        self, signal: dict, portfolio_state: dict
    ) -> Tuple[bool, str]:
        """
        Returns (allowed, reason_if_blocked).
        Checks:
          - daily_loss_pct < 2% (circuit breaker)
          - drawdown_pct < 8% (max drawdown)
          - open_positions < 2 (max concurrent)
          - one position per symbol (no stacking)
          - spread within tolerance
        """
        settings = self._settings

        # Daily loss circuit breaker
        daily_loss_pct = float(portfolio_state.get("daily_loss_pct", 0.0))
        if daily_loss_pct > settings.MAX_DAILY_LOSS_PCT:
            return (False, f"Daily loss {daily_loss_pct:.2%} > limit {settings.MAX_DAILY_LOSS_PCT:.2%}")

        # Max drawdown
        drawdown_pct = float(portfolio_state.get("drawdown_pct", 0.0))
        if drawdown_pct > settings.MAX_DRAWDOWN_PCT:
            return (False, f"Drawdown {drawdown_pct:.2%} > limit {settings.MAX_DRAWDOWN_PCT:.2%}")

        # Max concurrent positions
        open_pos = int(portfolio_state.get("open_positions", 0))
        if open_pos >= settings.MAX_CONCURRENT_POSITIONS:
            return (False, f"Max concurrent positions ({settings.MAX_CONCURRENT_POSITIONS}) reached")

        # One position per symbol (no stacking)
        symbol = signal.get("symbol", "")
        open_symbols = set(portfolio_state.get("open_symbols", []))
        if symbol in open_symbols:
            return (False, f"Position already open for {symbol}")

        # Spread check
        spread = float(signal.get("signal_metadata", {}).get("spread_pips", 0))
        if symbol == "XAUUSD" and spread > 50:
            return (False, f"XAUUSD spread {spread} pips too high")
        elif symbol != "XAUUSD" and spread > 3:
            return (False, f"Spread {spread} pips too high for {symbol}")

        return (True, "")

    def compute_position_size(
        self,
        symbol: str,
        entry: float,
        stop_loss: float,
        equity: float,
        quality_score: float = 0.5,
        ml_enabled: bool = False,
    ) -> float:
        """
        Base: equity × 0.01 / pip_distance
        ML scaling: base × (0.75 + quality_score × 0.5) if ml_enabled
        Clamped: [0.5× base, 1.5× base]
        """
        pip_distance = abs(entry - stop_loss)
        if pip_distance < 1e-9:
            return 0.01

        base = (equity * 0.01) / pip_distance

        if ml_enabled:
            scale = 0.75 + float(np.clip(quality_score, 0, 1)) * 0.5
            sized = base * scale
            sized = float(np.clip(sized, base * 0.5, base * 1.5))
        else:
            sized = base

        return round(max(sized, 0.01), 2)
