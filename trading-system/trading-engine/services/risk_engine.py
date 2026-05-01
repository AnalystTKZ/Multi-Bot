"""
risk_engine.py — Mathematically disciplined, statistically grounded risk controls.

Fixed-fractional position sizing.  No martingale.  No ML-confidence scaling.
All circuit breakers checked in strict order before any order submission.

Position sizing formula:
    risk_amount   = equity × RISK_PER_TRADE
    position_size = risk_amount / |entry - stop_loss|

Optional fractional-Kelly overlay (disabled by default):
    f* = (p × R − q) / R        p=win_prob, R=reward/risk, q=1−p
    kelly_size = f* × KELLY_FRACTION × equity / |entry − stop_loss|
    final_size = min(fixed_fractional, kelly_size)   ← always the lower of the two

Weekly loss tracking requires callers to invoke update_weekly_state() on every
bar and record_trade_result() after each closed trade.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Currency-direction correlation groups used for the correlated-exposure filter.
# Positions in the same group pointing the same direction represent stacked USD exposure.
_USD_SHORT_GROUP: frozenset[str] = frozenset({"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "EURGBP"})
_USD_LONG_GROUP:  frozenset[str] = frozenset({"USDJPY", "USDCAD", "USDCHF"})
_CROSS_GROUP:     frozenset[str] = frozenset({"EURJPY", "GBPJPY"})
_GOLD_GROUP:      frozenset[str] = frozenset({"XAUUSD"})


def _corr_group(symbol: str) -> Optional[str]:
    if symbol in _USD_SHORT_GROUP:
        return "usd_short"
    if symbol in _USD_LONG_GROUP:
        return "usd_long"
    if symbol in _CROSS_GROUP:
        return "cross"
    if symbol in _GOLD_GROUP:
        return "gold"
    return None


class RiskEngine:
    """
    System-wide risk controls.  All checks run before order submission.

    State tracked across bars:
      _weekly_loss_pct       — cumulative fractional loss this ISO week
      _week_start_equity     — equity at Monday open (reset every new ISO week)
      _last_week_number      — ISO week number of last update
      _consecutive_losses    — count of consecutive closed losing trades
      _cooldown_remaining    — bars remaining in post-streak cooldown
    """

    def __init__(self, settings) -> None:
        self._settings = settings
        self._weekly_loss_pct: float = 0.0
        self._week_start_equity: float = getattr(settings, "ACCOUNT_BALANCE", 10000.0)
        self._last_week_number: int = -1
        self._consecutive_losses: int = 0
        self._cooldown_remaining: int = 0

    # ── State maintenance ─────────────────────────────────────────────────────

    def update_weekly_state(self, equity: float, current_dt: Optional[datetime] = None) -> None:
        """
        Call once per bar.  Resets weekly loss counter on ISO week boundary.
        Uses UTC week number to be consistent with broker timestamps.
        """
        if current_dt is None:
            current_dt = datetime.now(timezone.utc)
        week_num = current_dt.isocalendar()[1]
        if week_num != self._last_week_number:
            self._weekly_loss_pct = 0.0
            self._week_start_equity = equity
            self._last_week_number = week_num

    def record_trade_result(self, pnl: float, equity: float) -> None:
        """
        Call after each closed trade.  Updates consecutive-loss streak and
        triggers cooldown when MAX_CONSECUTIVE_LOSSES is breached.

        The weekly loss accumulator uses pnl as a fraction of current equity
        so it is comparable to MAX_WEEKLY_LOSS_PCT.
        """
        if pnl < 0:
            self._consecutive_losses += 1
            self._weekly_loss_pct += abs(pnl) / max(equity, 1.0)
            max_consec = getattr(self._settings, "MAX_CONSECUTIVE_LOSSES", 3)
            if self._consecutive_losses >= max_consec:
                cooldown = getattr(self._settings, "CONSECUTIVE_LOSS_COOLDOWN_BARS", 10)
                self._cooldown_remaining = cooldown
                logger.warning(
                    "RiskEngine: %d consecutive losses — cooldown %d bars",
                    self._consecutive_losses, cooldown,
                )
        else:
            self._consecutive_losses = 0

    def tick_cooldown(self) -> None:
        """Decrement cooldown counter by one bar. Call once per bar after update_weekly_state."""
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

    # ── Pre-trade gate ────────────────────────────────────────────────────────

    def check_pre_trade(
        self,
        signal: dict,
        portfolio_state: dict,
    ) -> Tuple[bool, str]:
        """
        Returns (allowed, reason_if_blocked).

        Gate order (fail-fast):
          1. Daily loss circuit breaker
          2. Weekly loss circuit breaker
          3. Portfolio drawdown halt
          4. Consecutive-loss cooldown
          5. Max concurrent positions
          6. One position per symbol (no stacking)
          7. Correlated-exposure cap
          8. Spread filter (per-symbol configurable limit)
        """
        s = self._settings

        # 1. Daily loss
        daily_loss_pct = float(portfolio_state.get("daily_loss_pct", 0.0))
        max_daily = getattr(s, "MAX_DAILY_LOSS_PCT", 0.02)
        if daily_loss_pct > max_daily:
            return False, f"daily_loss {daily_loss_pct:.2%} > limit {max_daily:.2%}"

        # 2. Weekly loss
        max_weekly = getattr(s, "MAX_WEEKLY_LOSS_PCT", 0.05)
        if self._weekly_loss_pct > max_weekly:
            return False, f"weekly_loss {self._weekly_loss_pct:.2%} > limit {max_weekly:.2%}"

        # 3. Drawdown halt
        drawdown_pct = float(portfolio_state.get("drawdown_pct", 0.0))
        max_dd = getattr(s, "MAX_DRAWDOWN_PCT", 0.15)
        if drawdown_pct > max_dd:
            return False, f"drawdown {drawdown_pct:.2%} > limit {max_dd:.2%}"

        # 4. Consecutive-loss cooldown
        if self._cooldown_remaining > 0:
            return False, f"consecutive_loss_cooldown {self._cooldown_remaining} bars remaining"

        # 5. Max concurrent positions
        open_pos = int(portfolio_state.get("open_positions", 0))
        max_pos = getattr(s, "MAX_CONCURRENT_POSITIONS", 2)
        if open_pos >= max_pos:
            return False, f"max_concurrent_positions ({max_pos}) reached"

        # 6. One position per symbol
        symbol = signal.get("symbol", "")
        open_symbols: set = set(portfolio_state.get("open_symbols", []))
        if symbol in open_symbols:
            return False, f"position already open for {symbol}"

        # 7. Correlated-exposure cap
        side = str(signal.get("side", "buy")).lower()
        max_corr = getattr(s, "MAX_CORRELATED_POSITIONS", 1)
        group = _corr_group(symbol)
        if group is not None:
            open_positions: List[dict] = portfolio_state.get("positions", [])
            corr_count = sum(
                1 for p in open_positions
                if _corr_group(str(p.get("symbol", ""))) == group
                and str(p.get("side", "")).lower() == side
            )
            if corr_count >= max_corr:
                return False, (
                    f"correlated_exposure limit ({max_corr}) reached for "
                    f"{symbol} {side} [group={group}]"
                )

        # 8. Spread filter (configurable per symbol)
        spread = float((signal.get("signal_metadata") or {}).get("spread_pips", 0.0))
        max_spread_key = f"MAX_SPREAD_{symbol.upper()}"
        max_spread_default = 30.0 if symbol == "XAUUSD" else 3.0
        max_spread = float(getattr(s, max_spread_key, max_spread_default))
        if spread > max_spread:
            return False, f"spread {spread:.1f}pips > limit {max_spread:.1f} for {symbol}"

        return True, ""

    # ── Position sizing ───────────────────────────────────────────────────────

    def compute_position_size(
        self,
        symbol: str,
        entry: float,
        stop_loss: float,
        equity: float,
        win_rate: Optional[float] = None,
        avg_rr: Optional[float] = None,
    ) -> float:
        """
        Fixed-fractional sizing.  No ML-confidence scaling.  No martingale.

        risk_amount   = equity × RISK_PER_TRADE
        position_size = risk_amount / |entry − stop_loss|

        Optional fractional-Kelly overlay (KELLY_ENABLED=True):
          f* = (p×R − q) / R        (Kelly criterion)
          kelly_size = f* × KELLY_FRACTION × equity / stop_distance
          Uses the MORE CONSERVATIVE of fixed-fractional vs Kelly.

        Caller must supply win_rate and avg_rr for Kelly to activate;
        if either is None the overlay is skipped regardless of KELLY_ENABLED.
        """
        s = self._settings
        risk_pct: float = float(getattr(s, "RISK_PER_TRADE", 0.005))
        risk_amount = equity * risk_pct
        stop_dist = abs(entry - stop_loss)
        if stop_dist < 1e-9:
            return 0.01

        fixed_size = risk_amount / stop_dist

        # Optional fractional-Kelly overlay
        kelly_enabled = getattr(s, "KELLY_ENABLED", False)
        if kelly_enabled and win_rate is not None and avg_rr is not None:
            p = float(np.clip(win_rate, 0.05, 0.95))
            q = 1.0 - p
            R = float(max(avg_rr, 0.1))
            kelly_f = (p * R - q) / R          # optimal fraction of capital
            if kelly_f > 0:
                frac = float(getattr(s, "KELLY_FRACTION", 0.25))
                kelly_size = (kelly_f * frac * equity) / stop_dist
                fixed_size = min(fixed_size, kelly_size)

        return round(max(fixed_size, 0.01), 2)

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def compute_expected_r(p_win: float, reward_to_risk: float) -> float:
        """
        Probability-weighted expected R multiple.

        E[R] = P(win) × RR − P(loss) × 1.0

        A trade is only worth taking when E[R] > 0 AND exceeds MIN_EXPECTED_R.
        """
        p_win = float(np.clip(p_win, 0.0, 1.0))
        p_loss = 1.0 - p_win
        rr = float(max(reward_to_risk, 0.0))
        return p_win * rr - p_loss * 1.0

    @staticmethod
    def compute_break_even_win_rate(reward_to_risk: float) -> float:
        """
        Minimum win rate required for positive expectancy at a given RR.

        break_even_p = 1 / (1 + RR)
        """
        rr = float(max(reward_to_risk, 1e-9))
        return 1.0 / (1.0 + rr)

    @staticmethod
    def fractional_kelly(p_win: float, reward_to_risk: float, fraction: float = 0.25) -> float:
        """
        Fractional Kelly fraction of capital to risk on a single trade.

        f* = (p×R − q) / R
        fractional_f = f* × fraction

        Returns 0.0 if the trade has negative or zero Kelly fraction (negative EV).
        """
        p = float(np.clip(p_win, 0.05, 0.95))
        q = 1.0 - p
        R = float(max(reward_to_risk, 0.1))
        kelly_f = (p * R - q) / R
        if kelly_f <= 0:
            return 0.0
        return float(np.clip(kelly_f * fraction, 0.0, 0.20))  # hard cap at 20% per trade
