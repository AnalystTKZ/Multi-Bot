"""
portfolio_manager.py — Unified trade lifecycle and portfolio risk manager.

Responsibilities
────────────────
1. Dynamic R:R — TP1/TP2 targets scaled by signal confidence (must be >1.0).
2. TP1 partial close (50%) → SL to break-even.
3. TP2 full close of remainder.
4. Trailing stop after TP1 (1×ATR behind price).
5. Correlation filter — max 2 positions per directional currency group.
6. Session-end forced close.
7. Daily risk budget per trader.
8. Volatility-adjusted position sizing (ATR scalar, clamped [0.5, 1.25]).
9. Streak scaling — 3 losses → 0.5×, 4+ → 0.35×, recover on 2 wins.
10. Max concurrent correlated positions cap.

Usage
─────
    pm = PortfolioManager(settings, bar_date="2024-01-15")  # backtest mode
    pm = PortfolioManager(settings)                          # live mode

    enriched = pm.enrich_signal(signal, portfolio_state, atr=atr)
    if enriched is None:
        continue

    actions = pm.manage_open_positions(open_positions, current_prices)
    pm.record_outcome(trader_id, pnl)
    pm.notify_date(bar_date_str)   # call each new bar in backtest
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─── Currency correlation groups ──────────────────────────────────────────────
# Directional group = base_group + "_" + which-side-is-risk-on
# EURUSD buy  = long EUR / short USD → group "usd_short_buy"
# USDCAD buy  = long USD / short CAD → group "usd_long_buy"
_CURRENCY_GROUPS: Dict[str, str] = {
    "EURUSD": "usd_short",
    "GBPUSD": "usd_short",
    "AUDUSD": "usd_short",
    "NZDUSD": "usd_short",
    "USDCAD": "usd_long",
    "USDCHF": "usd_long",
    "USDJPY": "usd_long",
    "XAUUSD": "risk_off",
    "EURGBP": "eur_cross",
    "EURJPY": "eur_cross",
    "GBPJPY": "gbp_cross",
}

_MAX_SAME_GROUP = 2   # max positions in the same directional group


class PortfolioManager:
    """
    Central trade lifecycle and portfolio risk manager.

    Two operating modes:
      - Live mode: daily budget resets on real wall-clock date change.
      - Backtest mode: call notify_date(date_str) each bar so daily state
        resets at the correct historical date boundary.
    """

    def __init__(self, settings, bar_date: Optional[str] = None):
        self._settings = settings

        # Per-trader streak: list of bool (True=win)
        self._outcome_history: Dict[str, List[bool]] = defaultdict(list)

        # Per-trader daily P&L accumulator
        self._daily_pnl: Dict[str, float] = defaultdict(float)

        # Date of last daily reset — set externally in backtest via notify_date()
        self._current_date: Optional[str] = bar_date

        # Live position state (used by manage_open_positions / force_close_all)
        self._be_moved:     Dict[str, bool] = {}
        self._tp1_done:     Dict[str, bool] = {}
        self._trail_active: Dict[str, bool] = {}   # True once price clears TP1 + 0.5×ATR

    # ─── Backtest date sync ───────────────────────────────────────────────────

    def notify_date(self, date_str: str) -> None:
        """
        Call at the start of each new bar in backtest mode.
        Resets daily P&L when the date changes, mirroring what happens in live
        trading at midnight UTC.
        """
        if date_str != self._current_date:
            self._current_date = date_str
            self._daily_pnl.clear()
            logger.debug("PM: daily P&L reset → %s", date_str)

    # ─── Public: enrich signal ────────────────────────────────────────────────

    def enrich_signal(
        self,
        signal: dict,
        portfolio_state: dict,
        atr: float = 0.0,
    ) -> Optional[dict]:
        """
        Validates and enriches a signal. Returns enriched dict or None (rejected).

        Checks (in order):
          1. Daily loss budget for this trader
          2. Currency correlation cap
          3. Dynamic R:R (TP1 mult must be > 1.0)
          4. Volatility-adjusted sizing
          5. Streak-based size scaling
        """
        trader_id  = signal.get("trader_id", "unknown")
        symbol     = signal.get("symbol", "")
        side       = signal.get("side", "buy")
        confidence = float(signal.get("confidence", 0.60))
        entry      = float(signal.get("entry", 0.0))
        stop_loss  = float(signal.get("stop_loss", entry))

        # 1. Daily loss budget
        account    = float(getattr(self._settings, "ACCOUNT_BALANCE", 10000))
        alloc      = account * float(getattr(self._settings, "CAPITAL_PER_TRADER", 0.20))
        budget     = alloc * float(getattr(self._settings, "MAX_DAILY_LOSS_PCT", 0.02))
        daily_loss = abs(min(0.0, self._daily_pnl[trader_id]))
        if daily_loss >= budget:
            logger.info("PM: %s daily budget exhausted (%.2f/%.2f) — skip",
                        trader_id, daily_loss, budget)
            return None

        # 2. Correlation cap
        if not self._passes_correlation_check(symbol, side, portfolio_state):
            logger.info("PM: %s %s %s — correlation cap hit", trader_id, side, symbol)
            return None

        # 3. Dynamic R:R
        sl_dist = abs(entry - stop_loss)
        if sl_dist < 1e-9:
            logger.warning("PM: %s zero SL distance — skip", trader_id)
            return None

        tp1_mult, tp2_mult = self._dynamic_rr_multipliers(confidence)
        if tp1_mult < 2.0:
            logger.info("PM: %s conf=%.2f → R:R=%.2f < 2.0 (min 1:2) — skip",
                        trader_id, confidence, tp1_mult)
            return None

        if side == "buy":
            tp1 = entry + sl_dist * tp1_mult
            tp2 = entry + sl_dist * tp2_mult
        else:
            tp1 = entry - sl_dist * tp1_mult
            tp2 = entry - sl_dist * tp2_mult

        # 4. Volatility-adjusted sizing (risk 1% of allocated capital per trade)
        equity   = float(portfolio_state.get("equity", account))
        risk_pct = float(getattr(self._settings, "RISK_PER_TRADE", 0.01))
        allocated = equity * float(getattr(self._settings, "CAPITAL_PER_TRADER", 0.20))
        base_size = (allocated * risk_pct) / sl_dist
        base_size = max(round(base_size, 2), 0.01)

        vol_scalar    = self._volatility_scalar(atr, signal)
        streak_scalar = self._streak_scalar(trader_id)
        size = round(max(base_size * vol_scalar * streak_scalar, 0.01), 2)

        logger.info(
            "PM: %s %s %s conf=%.2f R:R %.2f/%.2f size=%.2f "
            "(base=%.2f vol=%.2f streak=%.2f daily_left=%.2f)",
            trader_id, side, symbol, confidence,
            tp1_mult, tp2_mult, size,
            base_size, vol_scalar, streak_scalar, budget - daily_loss,
        )

        return {
            **signal,
            "tp1":        round(tp1, 6),
            "tp2":        round(tp2, 6),
            "take_profit": round(tp2, 6),
            "rr_ratio":   round(tp1_mult, 2),
            "rr_to_tp2":  round(tp2_mult, 2),
            "size":       size,
            "size_full":  size,
            "confidence": confidence,
            "portfolio_manager": {
                "tp1_mult":           round(tp1_mult, 2),
                "tp2_mult":           round(tp2_mult, 2),
                "vol_scalar":         round(vol_scalar, 3),
                "streak_scalar":      round(streak_scalar, 3),
                "daily_loss_remaining": round(budget - daily_loss, 2),
            },
        }

    # ─── Public: manage open positions ────────────────────────────────────────

    def manage_open_positions(
        self,
        open_positions: List[dict],
        current_prices: Dict[str, float],
    ) -> List[dict]:
        """
        Called on every bar (live/paper trading). Returns list of action dicts:
          {"ticket": ..., "action": "partial_close"|"move_sl"|"close", ...}

        Note: in backtest the two-phase logic lives in _simulate_trade_pm()
        which operates on bar high/low for accurate touch detection.
        """
        actions = []
        for pos in open_positions:
            ticket    = str(pos.get("ticket", ""))
            symbol    = str(pos.get("symbol", ""))
            side      = str(pos.get("side", "buy"))
            entry     = float(pos.get("entry", 0.0))
            sl        = float(pos.get("stop_loss", entry))
            tp1       = float(pos.get("tp1", pos.get("take_profit", entry)))
            tp2       = float(pos.get("take_profit", entry))
            size_full = float(pos.get("size_full", pos.get("size", 0.01)))
            current   = float(current_prices.get(symbol, 0.0))
            if current <= 0:
                continue

            tp1_done     = self._tp1_done.get(ticket, False)
            be_done      = self._be_moved.get(ticket, False)
            trail_active = self._trail_active.get(ticket, False)
            pip          = self._pip_size(symbol)
            atr          = float(pos.get("signal_metadata", {}).get(
                               "atr_at_entry", pos.get("atr_at_entry", current * 0.005)))
            # Trail activates only once price clears TP1 + 0.5×ATR buffer
            activate_level_buy  = tp1 + 0.5 * atr
            activate_level_sell = tp1 - 0.5 * atr

            if side == "buy":
                if current <= sl:
                    actions.append({"ticket": ticket, "action": "close",
                                    "reason": "sl", "price": current})
                    self._cleanup(ticket)
                    continue
                if not tp1_done and current >= tp1:
                    actions.append({"ticket": ticket, "action": "partial_close",
                                    "size": round(size_full * 0.5, 2),
                                    "reason": "tp1", "price": tp1})
                    self._tp1_done[ticket] = True
                    if not be_done:
                        actions.append({"ticket": ticket, "action": "move_sl",
                                        "new_sl": round(entry + pip, 6),
                                        "reason": "breakeven"})
                        self._be_moved[ticket] = True
                if tp1_done and current >= tp2:
                    actions.append({"ticket": ticket, "action": "close",
                                    "reason": "tp2", "price": tp2})
                    self._cleanup(ticket)
                    continue
                if tp1_done:
                    # Activate trail once price clears TP1 + 0.5×ATR
                    if not trail_active and current >= activate_level_buy:
                        self._trail_active[ticket] = True
                        trail_active = True
                    if trail_active:
                        trail = self._trail_stop(pos, current, side)
                        if trail is not None and trail > sl:
                            actions.append({"ticket": ticket, "action": "move_sl",
                                            "new_sl": round(trail, 6), "reason": "trailing"})
            else:
                if current >= sl:
                    actions.append({"ticket": ticket, "action": "close",
                                    "reason": "sl", "price": current})
                    self._cleanup(ticket)
                    continue
                if not tp1_done and current <= tp1:
                    actions.append({"ticket": ticket, "action": "partial_close",
                                    "size": round(size_full * 0.5, 2),
                                    "reason": "tp1", "price": tp1})
                    self._tp1_done[ticket] = True
                    if not be_done:
                        actions.append({"ticket": ticket, "action": "move_sl",
                                        "new_sl": round(entry - pip, 6),
                                        "reason": "breakeven"})
                        self._be_moved[ticket] = True
                if tp1_done and current <= tp2:
                    actions.append({"ticket": ticket, "action": "close",
                                    "reason": "tp2", "price": tp2})
                    self._cleanup(ticket)
                    continue
                if tp1_done:
                    if not trail_active and current <= activate_level_sell:
                        self._trail_active[ticket] = True
                        trail_active = True
                    if trail_active:
                        trail = self._trail_stop(pos, current, side)
                        if trail is not None and trail < sl:
                            actions.append({"ticket": ticket, "action": "move_sl",
                                            "new_sl": round(trail, 6), "reason": "trailing"})
        return actions

    def force_close_all(
        self,
        open_positions: List[dict],
        current_prices: Dict[str, float],
        reason: str = "session_end",
    ) -> List[dict]:
        """Hard-close all positions (session boundary or circuit breaker)."""
        actions = []
        for pos in open_positions:
            ticket  = str(pos.get("ticket", ""))
            symbol  = str(pos.get("symbol", ""))
            current = float(current_prices.get(symbol, pos.get("entry", 0.0)))
            actions.append({"ticket": ticket, "action": "close",
                             "reason": reason, "price": current})
            self._cleanup(ticket)
        return actions

    # ─── Public: record outcome ───────────────────────────────────────────────

    def record_outcome(self, trader_id: str, pnl: float, rr_achieved: float = 0.0) -> None:
        """Update streak history and daily P&L after a trade closes."""
        self._daily_pnl[trader_id] += pnl
        self._outcome_history[trader_id].append(pnl > 0)
        if len(self._outcome_history[trader_id]) > 20:
            self._outcome_history[trader_id].pop(0)
        logger.debug("PM outcome: %s pnl=%.4f rr=%.2f %s",
                     trader_id, pnl, rr_achieved, "W" if pnl > 0 else "L")

    # ─── Dynamic R:R ──────────────────────────────────────────────────────────

    def _dynamic_rr_multipliers(self, confidence: float) -> tuple[float, float]:
        """
        Returns (tp1_mult, tp2_mult) as multiples of SL distance.
        Minimum enforced R:R is 1:2 (tp1_mult ≥ 2.0).
        Signals with confidence < 0.70 are rejected before reaching here.

        0.70 → 2.0 / 3.5   (min accepted: 1:2 to TP1, 1:3.5 to TP2)
        0.75 → 2.5 / 4.0
        0.80 → 3.0 / 5.0
        0.85 → 3.5 / 5.5
        0.90 → 4.0 / 6.0   (max confidence: 1:4 to TP1, 1:6 to TP2)
        """
        # Minimum accepted R:R is 1:2 (tp1_mult = 2.0).
        # Floor anchor is 0.70 (aligned with MIN_CONFIDENCE in run_backtest.py).
        # Signals below 0.70 are already rejected before reaching here.
        # TP2 is always ≥ 1.5× TP1 distance (second leg of the move).
        anchors = [
            (0.70, 2.0, 3.5),   # min conf → min R:R 1:2, TP2 at 3.5×
            (0.75, 2.5, 4.0),
            (0.80, 3.0, 5.0),
            (0.85, 3.5, 5.5),
            (0.90, 4.0, 6.0),
        ]
        c = float(np.clip(confidence, 0.70, 0.90))
        for i in range(len(anchors) - 1):
            c0, t1_0, t2_0 = anchors[i]
            c1, t1_1, t2_1 = anchors[i + 1]
            if c0 <= c <= c1:
                t = (c - c0) / (c1 - c0)
                return round(t1_0 + t * (t1_1 - t1_0), 3), round(t2_0 + t * (t2_1 - t2_0), 3)
        return 3.0, 5.0

    # ─── Volatility scalar ────────────────────────────────────────────────────

    def _volatility_scalar(self, atr: float, signal: dict) -> float:
        """nominal_atr / current_atr, clamped [0.5, 1.25]."""
        nominal = float(signal.get("signal_metadata", {}).get("atr_nominal", 0.0))
        if nominal < 1e-9 or atr < 1e-9:
            return 1.0
        return float(np.clip(nominal / atr, 0.5, 1.25))

    # ─── Streak scalar ────────────────────────────────────────────────────────

    def _streak_scalar(self, trader_id: str) -> float:
        """
        3 consecutive losses → 0.50×
        4+ consecutive losses → 0.35×
        2 consecutive wins after a loss → full restore to 1.0×
        """
        hist = self._outcome_history[trader_id]
        if not hist:
            return 1.0

        # Count trailing consecutive losses
        consec_losses = 0
        for outcome in reversed(hist):
            if not outcome:
                consec_losses += 1
            else:
                break

        if consec_losses >= 4:
            return 0.35
        if consec_losses >= 3:
            return 0.50

        # Recovery: 2 wins after a prior loss streak
        if len(hist) >= 3 and hist[-1] and hist[-2] and not hist[-3]:
            return 1.0

        return 1.0

    # ─── Correlation filter ───────────────────────────────────────────────────

    def _passes_correlation_check(
        self, symbol: str, side: str, portfolio_state: dict
    ) -> bool:
        """Block if adding this trade exceeds _MAX_SAME_GROUP in any directional group."""
        open_positions = portfolio_state.get("open_positions_detail", [])
        if not open_positions:
            return True
        new_group = self._directional_group(symbol, side)
        if new_group is None:
            return True
        count = sum(
            1 for pos in open_positions
            if self._directional_group(pos.get("symbol", ""), pos.get("side", "")) == new_group
        )
        if count >= _MAX_SAME_GROUP:
            logger.debug("PM: group=%s at cap (%d) — block", new_group, count)
            return False
        return True

    def _directional_group(self, symbol: str, side: str) -> Optional[str]:
        base = _CURRENCY_GROUPS.get(symbol)
        if base is None:
            return None
        return f"{base}_{side}"

    # ─── Trailing stop ────────────────────────────────────────────────────────

    def _trail_stop(self, pos: dict, current: float, side: str) -> Optional[float]:
        """
        Trail at 1.5×ATR behind current price after TP1.

        Uses 1.5× (not 1×) to prevent normal post-TP1 oscillation from
        stopping the remainder out before TP2. The activation buffer
        (0.5×ATR past TP1 before trail starts) is enforced by the caller
        in manage_open_positions.
        """
        atr = float(
            pos.get("signal_metadata", {}).get("atr_at_entry",
            pos.get("atr_at_entry", current * 0.005))
        )
        return (current - 1.5 * atr) if side == "buy" else (current + 1.5 * atr)

    # ─── Pip size ─────────────────────────────────────────────────────────────

    @staticmethod
    def _pip_size(symbol: str) -> float:
        if symbol == "XAUUSD":
            return 0.10
        if "JPY" in symbol:
            return 0.01
        return 0.0001

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _cleanup(self, ticket: str) -> None:
        self._tp1_done.pop(ticket, None)
        self._be_moved.pop(ticket, None)
        self._trail_active.pop(ticket, None)
