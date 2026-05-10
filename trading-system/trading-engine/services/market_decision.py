"""
Shared HTF/LTF market-decision gate for live and backtest signal generation.

The regime classifiers describe context; this module decides whether that context
is tradable for the proposed side. Keeping this rule in one place prevents
backtest/live drift.
"""

from __future__ import annotations

from typing import Any

from services.regime_scores import classify_entry_readiness, classify_trade_regime


def _bar_value(bar: Any, key: str, default: Any = None) -> Any:
    try:
        return bar.get(key, default)
    except AttributeError:
        return default


def _as_bool(value: Any) -> bool:
    try:
        if value != value:
            return False
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def combined_market_decision(
    *,
    htf_bias: str,
    ltf_behaviour: str,
    side: str,
    confidence: float,
    bar: Any,
    neutral_threshold: float = 0.60,
    volatile_threshold: float = 0.70,
    block_consolidating: bool = True,
    require_range: bool = True,
    htf_confidence: float = 1.0,
    regime_scores: dict[str, float] | None = None,
    trade_regime: str | None = None,
) -> tuple[bool, str]:
    """
    Return (allowed, reason) for a proposed trade side.

    Decision matrix:
      - BIAS_UP: buy only; TRENDING needs bullish structure, RANGING needs buy
        range-boundary entry, VOLATILE needs high confidence plus structure.
      - BIAS_DOWN: sell only; mirror of BIAS_UP.
      - BIAS_NEUTRAL: only range-boundary trades are allowed, and only when the
        LTF classifier explicitly says RANGING.

    htf_confidence: HTF regime classifier's softmax confidence for its prediction.
      The HTF confidence gate (HTF_MIN_REGIME_CONFIDENCE) is applied in the caller
      before this function is invoked, so htf_confidence here is available for any
      additional per-regime logic that needs it.
    """
    htf = str(htf_bias or "BIAS_NEUTRAL").upper()
    ltf = str(ltf_behaviour or "RANGING").upper()
    side = str(side or "").lower()
    conf = float(confidence)

    score_state = str(trade_regime or "").upper()
    if not score_state and regime_scores:
        try:
            score_state = classify_trade_regime(regime_scores).upper()
        except Exception:
            score_state = ""

    # New directional tradeability states (from classify_tradeability_directional)
    if score_state == "TRADEABLE_UP":
        if side != "buy":
            return False, "tradeability_direction_conflict"
    elif score_state == "TRADEABLE_DOWN":
        if side != "sell":
            return False, "tradeability_direction_conflict"
    elif score_state == "WAIT_PULLBACK":
        return False, "wait_pullback"
    elif score_state == "NO_TRADE_UNCERTAIN":
        return False, "no_trade_uncertain"
    # Legacy regime states (backward-compatible)
    elif score_state in {"NO_TRADE_CHOP", "NO_TRADE_EXTREME_VOL", "UNCERTAIN"}:
        return False, score_state.lower()
    elif score_state == "CONSOLIDATION" and block_consolidating:
        return False, "blocked_consolidation"
    elif score_state in {"TRADEABLE_TREND", "TRADEABLE_TREND_HIGH_VOL"}:
        ltf = "TRENDING"
    elif score_state == "RANGE":
        ltf = "RANGING"
    if score_state == "TRADEABLE_TREND_HIGH_VOL" and conf < volatile_threshold:
        return False, "trend_high_vol_weak_conf"

    range_valid = _as_bool(_bar_value(bar, "range_valid", False))
    range_side = str(_bar_value(bar, "range_side", "") or "").lower()
    range_width_atr = _as_float(_bar_value(bar, "range_width_atr", 0.0), 0.0)

    pullback_valid = _as_bool(_bar_value(bar, "pullback_valid", False))
    pullback_side = str(_bar_value(bar, "pullback_side", "") or "").lower()
    bos_bull = _as_bool(_bar_value(bar, "bos_bull", False))
    bos_bear = _as_bool(_bar_value(bar, "bos_bear", False))
    fvg_bull = _as_bool(_bar_value(bar, "fvg_bull", False))
    fvg_bear = _as_bool(_bar_value(bar, "fvg_bear", False))
    mss_bull = _as_bool(_bar_value(bar, "mss_bull", False))
    mss_bear = _as_bool(_bar_value(bar, "mss_bear", False))
    adx = _as_float(_bar_value(bar, "adx_14", _bar_value(bar, "adx", 20.0)), 20.0)

    bullish_structure = (
        (pullback_valid and pullback_side == "buy")
        or bos_bull
        or fvg_bull
        or mss_bull
    )
    bearish_structure = (
        (pullback_valid and pullback_side == "sell")
        or bos_bear
        or fvg_bear
        or mss_bear
    )
    side_structure = bullish_structure if side == "buy" else bearish_structure

    if score_state in {"TRADEABLE_UP", "TRADEABLE_DOWN"}:
        entry_state = classify_entry_readiness(
            score_state,
            side,
            bar,
            ltf_behaviour=ltf,
            require_range=require_range,
        )
        if entry_state == "WAIT_PULLBACK":
            return False, "wait_pullback"
        if entry_state != score_state:
            return False, entry_state.lower()

    if ltf == "CONSOLIDATING" and block_consolidating:
        return False, "blocked_consolidating"

    if htf == "BIAS_UP" and side != "buy":
        return False, "htf_bias_conflict"
    if htf == "BIAS_DOWN" and side != "sell":
        return False, "htf_bias_conflict"

    # MSS-against-bias: structure just broke in the direction OPPOSITE to HTF bias.
    # A bearish MSS during BIAS_UP means sell-side structure is now controlling —
    # trust the reversal signal over the stale bias label.
    if htf == "BIAS_UP" and mss_bear:
        return False, "mss_against_bias"
    if htf == "BIAS_DOWN" and mss_bull:
        return False, "mss_against_bias"

    if htf == "BIAS_NEUTRAL":
        if conf < neutral_threshold:
            return False, "neutral_bias_weak_conf"
        if ltf != "RANGING":
            return False, "neutral_requires_ltf_ranging"
        if not range_valid:
            return False, "neutral_range_missing"
        if range_side != side:
            return False, "neutral_range_side_conflict"
        if adx > 22:
            return False, "neutral_adx_too_high"
        if range_width_atr < 2.5:
            return False, "neutral_range_too_narrow"
        return True, "neutral_range_entry"

    if ltf == "TRENDING":
        if not side_structure:
            return False, "trend_structure_missing"
        return True, "trend_structure_entry"

    if ltf == "RANGING":
        if require_range and not range_valid:
            return False, "range_missing"
        if range_valid and range_side and range_side != side:
            return False, "range_side_conflict"
        return True, "range_entry"

    if ltf == "VOLATILE":
        if conf < volatile_threshold:
            return False, "volatile_weak_conf"
        if not side_structure:
            return False, "volatile_structure_missing"
        return True, "volatile_structure_entry"

    return False, "unknown_ltf_behaviour"
