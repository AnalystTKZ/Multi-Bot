#!/usr/bin/env python3
"""
optimize_parameters.py — Autonomous multi-agent parameter optimization.

Agent architecture (single process, sequential phases):
  PARAMETER_AGENT   → defines search space per trader
  SEARCH_AGENT      → Bayesian Optimization (scipy minimize / random fallback)
  BACKTEST_AGENT    → fast in-process backtest per config
  EVALUATION_AGENT  → composite score
  OVERFITTING_GUARD → train/val split validation
  CRITIC_AGENT      → reject unstable configs
  PORTFOLIO_AGENT   → capital allocation across traders

Usage:
    cd trading-system/trading-engine
    python scripts/optimize_parameters.py
    python scripts/optimize_parameters.py --traders 1 3 5
    python scripts/optimize_parameters.py --iterations 60 --output optimized_params.json

Hard constraint: NO strategy logic is modified.
Only parameters (thresholds, multipliers, filters, timing) are tuned.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("optimizer")

def _setup_log_file(output_dir: str) -> str:
    """Add file handler so logs are saved alongside results."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"optimize_{ts}.log")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                                       datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    return log_path

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG
# ──────────────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL    = 10_000.0
RISK_PER_TRADE     = 0.01
CAPITAL_PER_TRADER = 0.20
COMMISSION_PCT     = 0.001
SLIPPAGE_PCT       = 0.0002
MAX_DAILY_LOSS_PCT = 0.02
MAX_DRAWDOWN_PCT   = 0.20
COOLDOWN_BARS      = 10
MAX_HOLD_BARS      = 200

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "backtesting", "datasets"
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "backtest_results"
)

# Train: 2016-01-01 → 2022-12-31 (70%)
# Val:   2023-01-01 → 2026-02-28 (30%)
TRAIN_END = pd.Timestamp("2022-12-31", tz="UTC")
VAL_START = pd.Timestamp("2023-01-01", tz="UTC")

# Minimum trades threshold: 15 trades per pair per year
# Training span = 2016–2022 = 7 years; 11 pairs per trader
# → 15 trades/pair/year × 11 pairs × 7 years = 1155 total minimum
_TRAIN_YEARS = 7   # 2016–2022
MIN_TRADES_PER_PAIR_PER_YEAR = 15
MIN_TRADES_PER_TRADER = {}   # filled dynamically in _composite_score

# Scoring weights
W_PF   = 0.35
W_SR   = 0.25
W_EXP  = 0.20
W_DD   = 0.20

# Validation gates
GATE_PF  = 1.5
GATE_SR  = 1.2
GATE_DD  = 0.20
# GATE_MIN_TRADES is computed per-trader in _validate_oos using MIN_TRADES_PER_PAIR_PER_YEAR

# Available timeframes for tuning (maps to dataset files)
# Traders may tune which TF they run the primary signal on
TF_OPTIONS = ["15M", "1H"]     # 5M excluded (too noisy), 4H excluded (too few bars for session logic)

# ──────────────────────────────────────────────────────────────────────────────
# SYMBOL UNIVERSE — must be defined before SEARCH_SPACES so _PAIR_PARAMS resolves
# ──────────────────────────────────────────────────────────────────────────────

_ALL_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD",
    "USDCAD", "USDCHF", "EURGBP", "EURJPY", "GBPJPY", "XAUUSD",
]

# Pair selection params injected into every trader search space.
# Key: "pair_{SYMBOL}" — integer 0 (exclude) or 1 (include).
# Minimum 3 pairs enforced in _critic_check.
_PAIR_PARAMS: Dict[str, dict] = {
    f"pair_{sym}": {"low": 0, "high": 1, "default": 1, "integer": True, "is_pair": True}
    for sym in _ALL_SYMBOLS
}

# TRADER_SYMBOLS kept as fallback when no pair params present (e.g. portfolio backtest).
TRADER_SYMBOLS = {
    "trader_1": _ALL_SYMBOLS,
    "trader_2": _ALL_SYMBOLS,
    "trader_3": _ALL_SYMBOLS,
    "trader_5": _ALL_SYMBOLS,
}


def _active_symbols(params: dict) -> List[str]:
    """Return symbols selected by pair_* params. Falls back to all if none present."""
    selected = [sym for sym in _ALL_SYMBOLS if int(params.get(f"pair_{sym}", 1)) == 1]
    return selected if selected else list(_ALL_SYMBOLS)


# ──────────────────────────────────────────────────────────────────────────────
# PARAMETER_AGENT — search spaces per trader
# ──────────────────────────────────────────────────────────────────────────────

SEARCH_SPACES: Dict[str, Dict] = {
    "trader_1": {
        # Timeframe: 0=15M, 1=1H
        "tf_idx": {"low": 0, "high": 1, "default": 0, "integer": True, "is_tf": True},
        # EMA pullback distance threshold (× ATR)
        "dist_atr_mult": {"low": 0.3, "high": 0.8, "default": 0.5},
        # ADX threshold
        "adx_threshold": {"low": 18, "high": 30, "default": 22, "integer": True},
        # RSI pullback band (buy: low–high centred around 47.5; sell mirrored)
        "rsi_low":  {"low": 35, "high": 45, "default": 40, "integer": True},
        "rsi_high": {"low": 50, "high": 62, "default": 55, "integer": True},
        # Signal candle body fraction
        "body_pct": {"low": 0.20, "high": 0.50, "default": 0.30},
        # Gold ATR ratio gate
        "gold_atr_ratio": {"low": 0.5, "high": 0.9, "default": 0.7},
        # SL buffer (× ATR)
        "sl_atr_mult": {"low": 0.1, "high": 0.6, "default": 0.3},
        # TP multipliers
        "tp1_atr_mult": {"low": 1.0, "high": 2.5, "default": 1.5},
        "tp2_atr_mult": {"low": 2.0, "high": 4.5, "default": 3.0},
        # Round-number skip window (pips)
        "round_pips": {"low": 5, "high": 25, "default": 15, "integer": True},
    },
    "trader_2": {
        # Timeframe: 0=15M, 1=1H  (1H recommended — more FVG accumulation)
        "tf_idx": {"low": 0, "high": 1, "default": 0, "integer": True, "is_tf": True},
        # FVG max age (bars)
        "fvg_max_age": {"low": 5, "high": 60, "default": 20, "integer": True},
        # Min ATR ratio for FVG size validity
        "fvg_min_atr_ratio": {"low": 0.05, "high": 0.4, "default": 0.1},
        # BOS lookback: BOS within last N bars counts as confirmation
        "bos_lookback": {"low": 1, "high": 20, "default": 5, "integer": True},
        # SL buffer beyond FVG edge (× ATR)
        "sl_atr_mult": {"low": 0.05, "high": 0.5, "default": 0.1},
        # Minimum R:R to enter
        "min_rr": {"low": 1.0, "high": 3.0, "default": 1.5},
        # TP extension multiplier when no swing target (× ATR)
        "tp_fallback_mult": {"low": 1.0, "high": 3.5, "default": 2.0},
        # Round-number skip window (pips)
        "round_pips": {"low": 10, "high": 30, "default": 20, "integer": True},
    },
    "trader_3": {
        # Timeframe: 0=15M, 1=1H
        "tf_idx": {"low": 0, "high": 1, "default": 0, "integer": True, "is_tf": True},
        # Sweep wick/body ratio threshold
        "wick_body_ratio": {"low": 1.0, "high": 2.5, "default": 1.5},
        # Volume SMA multiplier
        "volume_mult": {"low": 1.0, "high": 1.8, "default": 1.3},
        # Asian range validity for EURUSD/GBPUSD (min pips)
        "asian_range_min_fx": {"low": 0.0008, "high": 0.0020, "default": 0.0015},
        # Asian range validity max (pips)
        "asian_range_max_fx": {"low": 0.005, "high": 0.012, "default": 0.008},
        # Asian range validity for XAUUSD (min $)
        "asian_range_min_gold": {"low": 0.5, "high": 2.0, "default": 1.0},
        # TP1 multiplier (× asian_range)
        "tp1_range_mult": {"low": 0.8, "high": 1.5, "default": 1.0},
        # TP2 multiplier (× asian_range) for FX
        "tp2_range_mult_fx": {"low": 1.3, "high": 3.0, "default": 2.0},
        # TP2 multiplier (× asian_range) for XAUUSD
        "tp2_range_mult_gold": {"low": 1.0, "high": 2.5, "default": 1.5},
        # SL buffer (× buffer constant)
        "sl_buffer_mult": {"low": 2.0, "high": 8.0, "default": 5.0},
    },
    "trader_5": {
        # Timeframe: 0=15M, 1=1H
        "tf_idx": {"low": 0, "high": 1, "default": 0, "integer": True, "is_tf": True},
        # ADX threshold (must be BELOW this)
        "adx_max": {"low": 18, "high": 28, "default": 25, "integer": True},
        # RSI oversold threshold (long)
        "rsi_oversold": {"low": 25, "high": 38, "default": 35, "integer": True},
        # RSI overbought threshold (short)
        "rsi_overbought": {"low": 62, "high": 75, "default": 65, "integer": True},
        # Stochastic thresholds
        "stoch_oversold": {"low": 20, "high": 35, "default": 30, "integer": True},
        "stoch_overbought": {"low": 65, "high": 80, "default": 70, "integer": True},
        # Price proximity to range edge (pips)
        "entry_proximity_pips": {"low": 3, "high": 10, "default": 5, "integer": True},
        # SL beyond range (pips)
        "sl_pips": {"low": 5, "high": 15, "default": 8, "integer": True},
        # Range validity FX (pips min/max)
        "range_min_fx": {"low": 0.0006, "high": 0.0015, "default": 0.0010},
        "range_max_fx": {"low": 0.004, "high": 0.008, "default": 0.006},
        # Sentiment gate (abs score)
        "sentiment_gate": {"low": 0.3, "high": 0.7, "default": 0.5},
    },
}

# Inject pair selection params into every trader's search space
for _tid in list(SEARCH_SPACES.keys()):
    SEARCH_SPACES[_tid].update(_PAIR_PARAMS)

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING & INDICATORS
# ──────────────────────────────────────────────────────────────────────────────

_DATA_CACHE: Dict[str, pd.DataFrame] = {}


def _load_csv(symbol: str, timeframe: str = "15M") -> pd.DataFrame:
    key = f"{symbol}_{timeframe}"
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    candidates = [
        os.path.join(DATA_DIR, f"{symbol}_{timeframe}.csv"),
        os.path.join(DATA_DIR, f"{symbol.lower()}_{timeframe.lower()}.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.columns = [c.lower() for c in df.columns]
            if "close" not in df.columns:
                continue
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index().dropna(subset=["open", "high", "low", "close"])
            _DATA_CACHE[key] = df
            return df
    logger.warning("No data for %s/%s", symbol, timeframe)
    return pd.DataFrame()


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    from indicators.market_structure import compute_all
    return compute_all(df)


_IND_CACHE: Dict[str, pd.DataFrame] = {}


def _get_indicators(symbol: str, timeframe: str = "15M") -> pd.DataFrame:
    key = f"{symbol}_{timeframe}"
    if key in _IND_CACHE:
        return _IND_CACHE[key]
    df = _load_csv(symbol, timeframe)
    if df.empty:
        return df
    logger.info("Computing indicators: %s/%s (%d bars)", symbol, timeframe, len(df))
    df = _compute_indicators(df)
    _IND_CACHE[key] = df
    return df


def _pip_for_entry(entry: float) -> float:
    if entry > 500:
        return 0.10
    if entry > 50:
        return 0.01
    return 0.0001


# ──────────────────────────────────────────────────────────────────────────────
# TRADE SIMULATION (shared)
# ──────────────────────────────────────────────────────────────────────────────

def _simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    side: str,
    entry: float,
    sl: float,
    tp1: float,
    tp2: float,
    size: float,
    atr: float,
) -> dict:
    pip = _pip_for_entry(entry)
    half = round(size * 0.5, 2)
    rem  = round(size - half, 2)

    tp1_done = False
    be_sl = sl
    trail_sl = sl
    p1_pnl = p2_pnl = 0.0
    exit_reason = "time_exit"

    for i in range(entry_idx + 1, min(entry_idx + MAX_HOLD_BARS, len(df))):
        h = float(df.iloc[i]["high"])
        l = float(df.iloc[i]["low"])
        c = float(df.iloc[i]["close"])

        if not tp1_done:
            if side == "buy":
                if l <= sl:
                    p1_pnl = (sl - entry) * size; exit_reason = "sl_full"; break
                if h >= tp1:
                    p1_pnl = (tp1 - entry) * half; tp1_done = True
                    be_sl = entry + pip; trail_sl = be_sl; continue
            else:
                if h >= sl:
                    p1_pnl = (entry - sl) * size; exit_reason = "sl_full"; break
                if l <= tp1:
                    p1_pnl = (entry - tp1) * half; tp1_done = True
                    be_sl = entry - pip; trail_sl = be_sl; continue
        else:
            if side == "buy":
                trail_sl = max(trail_sl, c - atr)
                eff_sl = max(be_sl, trail_sl)
                if l <= eff_sl:
                    p2_pnl = (eff_sl - entry) * rem
                    exit_reason = "be_or_trail"; break
                if h >= tp2:
                    p2_pnl = (tp2 - entry) * rem; exit_reason = "tp2"; break
            else:
                trail_sl = min(trail_sl, c + atr)
                eff_sl = min(be_sl, trail_sl)
                if h >= eff_sl:
                    p2_pnl = (entry - eff_sl) * rem
                    exit_reason = "be_or_trail"; break
                if l <= tp2:
                    p2_pnl = (entry - tp2) * rem; exit_reason = "tp2"; break
    else:
        ep = float(df.iloc[min(entry_idx + MAX_HOLD_BARS - 1, len(df) - 1)]["close"])
        if tp1_done:
            p2_pnl = (ep - entry if side == "buy" else entry - ep) * rem
        else:
            p1_pnl = (ep - entry if side == "buy" else entry - ep) * size

    gross = p1_pnl + p2_pnl
    gross -= abs(gross) * COMMISSION_PCT
    return {"pnl": gross, "exit_reason": exit_reason, "tp1_hit": tp1_done}


# ──────────────────────────────────────────────────────────────────────────────
# BACKTEST_AGENT — per-trader signal generators with injected params
# ──────────────────────────────────────────────────────────────────────────────

def _in_session(dt: pd.Timestamp, trader_id: str) -> bool:
    h = dt.hour
    sessions = {
        "trader_1": lambda h: 13 <= h < 17,
        "trader_2": lambda h: (7 <= h < 12) or (13 <= h < 18),
        "trader_3": lambda h: 7 <= h < 10,
        "trader_4": lambda h: True,
        "trader_5": lambda h: 2 <= h < 7,
    }
    fn = sessions.get(trader_id)
    return fn(h) if fn else True


def _signal_trader1(df: pd.DataFrame, i: int, symbol: str, p: dict) -> Optional[dict]:
    bar   = df.iloc[i]
    close = float(bar["close"])
    open_ = float(bar["open"])
    high  = float(bar["high"])
    low   = float(bar["low"])
    atr   = float(bar.get("atr_14", close * 0.001))
    if atr < 1e-9:
        return None

    ema_stk = int(bar.get("ema_stack", 0))
    dist    = abs(close - float(bar.get("ema_21", close)))
    adx     = float(bar.get("adx_14", 0))
    rsi     = float(bar.get("rsi_14", 50))

    if dist > atr * p["dist_atr_mult"]:
        return None
    if adx < p["adx_threshold"]:
        return None

    candle_range = high - low
    body = abs(close - open_)
    if candle_range < 1e-9 or body < p["body_pct"] * candle_range:
        return None

    if symbol == "XAUUSD":
        atr_mean = float(df["atr_14"].iloc[max(0, i-20):i].mean()) if i > 20 else atr
        if atr / (atr_mean + 1e-9) < p["gold_atr_ratio"]:
            return None

    if ema_stk == 2 and close > open_:
        if not (p["rsi_low"] <= rsi <= p["rsi_high"]):
            return None
        ema50 = float(bar.get("ema_50", close))
        sl = ema50 - atr * p["sl_atr_mult"]
        if sl >= close:
            return None
        tp1 = close + atr * p["tp1_atr_mult"]
        tp2 = close + atr * p["tp2_atr_mult"]
        return {"side": "buy", "entry": close, "sl": sl, "tp1": tp1, "tp2": tp2,
                "conf": 0.72, "atr": atr}

    if ema_stk == -2 and close < open_:
        rsi_sell_low  = 100 - p["rsi_high"]
        rsi_sell_high = 100 - p["rsi_low"]
        if not (rsi_sell_low <= rsi <= rsi_sell_high):
            return None
        ema50 = float(bar.get("ema_50", close))
        sl = ema50 + atr * p["sl_atr_mult"]
        if sl <= close:
            return None
        tp1 = close - atr * p["tp1_atr_mult"]
        tp2 = close - atr * p["tp2_atr_mult"]
        return {"side": "sell", "entry": close, "sl": sl, "tp1": tp1, "tp2": tp2,
                "conf": 0.72, "atr": atr}

    return None


def _signal_trader2(df: pd.DataFrame, i: int, symbol: str, p: dict,
                    fvg_registry: dict) -> Optional[dict]:
    bar    = df.iloc[i]
    close  = float(bar["close"])
    open_  = float(bar["open"])
    atr    = float(bar.get("atr_14", close * 0.001))
    if atr < 1e-9:
        return None

    fvg_bull = bool(bar.get("fvg_bull", False))
    fvg_bear = bool(bar.get("fvg_bear", False))
    bos_bull = bool(bar.get("bos_bull", False))
    bos_bear = bool(bar.get("bos_bear", False))

    min_atr_ratio = p["fvg_min_atr_ratio"]
    bos_lookback  = int(p.get("bos_lookback", 5))

    # Update per-symbol BOS state (counts down for lookback_bars after BOS fires)
    sym_state = fvg_registry.setdefault(f"_bos_{symbol}", {"bull": 0, "bear": 0})
    if bos_bull:
        sym_state["bull"] = bos_lookback
    elif sym_state["bull"] > 0:
        sym_state["bull"] -= 1
    if bos_bear:
        sym_state["bear"] = bos_lookback
    elif sym_state["bear"] > 0:
        sym_state["bear"] -= 1

    recent_bos_bull = sym_state["bull"] > 0
    recent_bos_bear = sym_state["bear"] > 0

    # Register new FVGs (independent of BOS — BOS is checked at entry time)
    if fvg_bull:
        top    = float(bar.get("fvg_bull_top", close + atr * 0.3))
        bottom = float(bar.get("fvg_bull_bottom", close - atr * 0.3))
        if (not (np.isnan(top) or np.isnan(bottom))
                and top > bottom
                and (top - bottom) >= atr * min_atr_ratio):
            fvg_registry.setdefault(symbol, []).append({
                "dir": "bull", "top": top, "bottom": bottom,
                "mid": (top + bottom) / 2, "age": 0
            })

    if fvg_bear:
        top    = float(bar.get("fvg_bear_top", close + atr * 0.3))
        bottom = float(bar.get("fvg_bear_bottom", close - atr * 0.3))
        if (not (np.isnan(top) or np.isnan(bottom))
                and top > bottom
                and (top - bottom) >= atr * min_atr_ratio):
            fvg_registry.setdefault(symbol, []).append({
                "dir": "bear", "top": top, "bottom": bottom,
                "mid": (top + bottom) / 2, "age": 0
            })

    # Age and prune registry (independently of current bar's signal)
    surviving = []
    for e in fvg_registry.get(symbol, []):
        if isinstance(e, dict) and "dir" in e:
            e["age"] += 1
            if e["age"] > p["fvg_max_age"]:
                continue
            # Invalidate if price closed fully through the FVG
            if e["dir"] == "bull" and close < e["bottom"]:
                continue
            if e["dir"] == "bear" and close > e["top"]:
                continue
            surviving.append(e)
    fvg_registry[symbol] = surviving

    # Entry scan: price inside FVG zone + BOS within lookback window + directional candle
    for fvg_entry in surviving:
        if fvg_entry["dir"] == "bull" and recent_bos_bull and close > open_:
            # Full FVG zone entry (bottom to top, not just lower half)
            in_zone = fvg_entry["bottom"] <= close <= fvg_entry["top"]
            if not in_zone:
                continue
            sl   = fvg_entry["bottom"] - atr * p["sl_atr_mult"]
            tp1  = close + atr * p["tp_fallback_mult"]
            tp2  = close + atr * p["tp_fallback_mult"] * 1.6
            denom = abs(close - sl)
            if denom < 1e-9 or abs(tp1 - close) / denom < p["min_rr"]:
                continue
            return {"side": "buy", "entry": close, "sl": sl, "tp1": tp1, "tp2": tp2,
                    "conf": 0.72, "atr": atr}

        if fvg_entry["dir"] == "bear" and recent_bos_bear and close < open_:
            # Full FVG zone entry (bottom to top, not just upper half)
            in_zone = fvg_entry["bottom"] <= close <= fvg_entry["top"]
            if not in_zone:
                continue
            sl   = fvg_entry["top"] + atr * p["sl_atr_mult"]
            tp1  = close - atr * p["tp_fallback_mult"]
            tp2  = close - atr * p["tp_fallback_mult"] * 1.6
            denom = abs(close - sl)
            if denom < 1e-9 or abs(tp1 - close) / denom < p["min_rr"]:
                continue
            return {"side": "sell", "entry": close, "sl": sl, "tp1": tp1, "tp2": tp2,
                    "conf": 0.72, "atr": atr}

    return None


def _signal_trader3(df: pd.DataFrame, i: int, symbol: str, p: dict,
                    asian_cache: dict) -> Optional[dict]:
    dt  = df.index[i]
    bar = df.iloc[i]
    close = float(bar["close"])
    open_ = float(bar["open"])
    high  = float(bar["high"])
    low   = float(bar["low"])
    atr   = float(bar.get("atr_14", close * 0.001))

    today = dt.date()

    # Build/get asian range
    if today not in asian_cache.get(symbol, {}):
        past = df[(df.index.date == today) & (df.index.hour < 7)]
        if len(past) < 3:
            return None
        a_high = float(past["high"].max())
        a_low  = float(past["low"].min())
        rng    = a_high - a_low
        if symbol == "XAUUSD":
            valid = rng >= p["asian_range_min_gold"]
        elif "JPY" in symbol:
            # JPY pairs: ranges in pips × 0.01 scale (e.g. 0.15–1.50)
            valid = (p["asian_range_min_fx"] * 100) <= rng <= (p["asian_range_max_fx"] * 100)
        else:
            valid = p["asian_range_min_fx"] <= rng <= p["asian_range_max_fx"]
        asian_cache.setdefault(symbol, {})[today] = {
            "high": a_high, "low": a_low, "range": rng, "valid": valid
        }

    arc = asian_cache.get(symbol, {}).get(today)
    if arc is None or not arc["valid"]:
        return None

    a_high = arc["high"]
    a_low  = arc["low"]
    a_range = arc["range"]

    candle_range = high - low
    body = abs(close - open_)
    if candle_range < 1e-9 or body < 1e-9:
        return None

    buffer_fx   = 0.0005
    buffer_gold = 0.50
    buffer = buffer_gold if symbol == "XAUUSD" else buffer_fx

    direction = None
    wick_below = max(0.0, a_low - low)
    if low < a_low - buffer and close > a_low and wick_below / (body + 1e-9) > p["wick_body_ratio"]:
        direction = "buy"

    wick_above = max(0.0, high - a_high)
    if direction is None and high > a_high + buffer and close < a_high and \
            wick_above / (body + 1e-9) > p["wick_body_ratio"]:
        direction = "sell"

    if direction is None:
        return None

    # Volume filter
    vol_sma = df["volume"].iloc[max(0, i-20):i].mean()
    if vol_sma > 0 and float(bar["volume"]) < vol_sma * p["volume_mult"]:
        return None

    sl_mult = p["sl_buffer_mult"]
    if direction == "buy":
        sl  = low - buffer * sl_mult
        tp1 = close + a_range * p["tp1_range_mult"]
        tp2 = close + a_range * (p["tp2_range_mult_gold"] if symbol == "XAUUSD" else p["tp2_range_mult_fx"])
    else:
        sl  = high + buffer * sl_mult
        tp1 = close - a_range * p["tp1_range_mult"]
        tp2 = close - a_range * (p["tp2_range_mult_gold"] if symbol == "XAUUSD" else p["tp2_range_mult_fx"])

    if abs(close - sl) < 1e-9:
        return None
    if abs(tp1 - close) / (abs(close - sl) + 1e-9) < 1.0:
        return None

    return {"side": direction, "entry": close, "sl": sl, "tp1": tp1, "tp2": tp2,
            "conf": 0.72, "atr": atr}


def _signal_trader5(df: pd.DataFrame, i: int, symbol: str, p: dict,
                    range_cache: dict) -> Optional[dict]:
    dt  = df.index[i]
    bar = df.iloc[i]
    close = float(bar["close"])
    atr   = float(bar.get("atr_14", close * 0.001))

    today = dt.date()

    # Build/get Asian MR range (00:00–02:00)
    if today not in range_cache.get(symbol, {}):
        past = df[(df.index.date == today) & (df.index.hour < 2)]
        if len(past) < 2:
            return None
        r_high = float(past["high"].max())
        r_low  = float(past["low"].min())
        rng    = r_high - r_low
        if symbol == "XAUUSD":
            valid = 1.0 <= rng <= 15.0   # gold Asian range $1–$15
        elif "JPY" in symbol:
            valid = 0.15 <= rng <= 1.50
        else:
            valid = p["range_min_fx"] <= rng <= p["range_max_fx"]
        range_cache.setdefault(symbol, {})[today] = {
            "high": r_high, "low": r_low, "mid": (r_high + r_low) / 2,
            "range": rng, "valid": valid
        }

    rc = range_cache.get(symbol, {}).get(today)
    if rc is None or not rc["valid"]:
        return None

    r_high = rc["high"]
    r_low  = rc["low"]
    r_mid  = rc["mid"]

    # ADX gate
    adx = float(bar.get("adx_14", 99))
    if adx >= p["adx_max"]:
        return None

    pip = 0.01 if "JPY" in symbol else 0.0001
    prox = p["entry_proximity_pips"] * pip
    sl_buf = p["sl_pips"] * pip

    rsi     = float(bar.get("rsi_14", 50))
    stoch_k = float(bar.get("stoch_k", 50))

    # LONG signal
    if close <= r_low + prox:
        if rsi < p["rsi_oversold"] and stoch_k < p["stoch_oversold"]:
            sl  = r_low - sl_buf
            tp1 = r_mid
            tp2 = r_high
            if abs(close - sl) < 1e-9:
                return None
            return {"side": "buy", "entry": close, "sl": sl, "tp1": tp1, "tp2": tp2,
                    "conf": 0.70, "atr": atr}

    # SHORT signal
    if close >= r_high - prox:
        if rsi > p["rsi_overbought"] and stoch_k > p["stoch_overbought"]:
            sl  = r_high + sl_buf
            tp1 = r_mid
            tp2 = r_low
            if abs(close - sl) < 1e-9:
                return None
            return {"side": "sell", "entry": close, "sl": sl, "tp1": tp1, "tp2": tp2,
                    "conf": 0.70, "atr": atr}

    return None


SIGNAL_FUNCS = {
    "trader_1": _signal_trader1,
    "trader_3": _signal_trader3,
}

# State-carrying traders need wrappers
def _make_signal_t2(p):
    fvg_registry = {}
    def fn(df, i, symbol, _p):
        return _signal_trader2(df, i, symbol, _p, fvg_registry)
    return fn

def _make_signal_t5(p):
    range_cache = {}
    def fn(df, i, symbol, _p):
        return _signal_trader5(df, i, symbol, _p, range_cache)
    return fn

def _make_signal_t3(p):
    asian_cache = {}
    def fn(df, i, symbol, _p):
        return _signal_trader3(df, i, symbol, _p, asian_cache)
    return fn


# ──────────────────────────────────────────────────────────────────────────────
# BACKTEST_AGENT — run backtest for one trader with given params on a date slice
# ──────────────────────────────────────────────────────────────────────────────

def _size_trade(entry: float, sl: float, equity: float) -> float:
    sl_dist = abs(entry - sl)
    if sl_dist < 1e-9:
        return 0.01
    risk = equity * CAPITAL_PER_TRADER * RISK_PER_TRADE
    return max(round(risk / sl_dist, 2), 0.01)


def _run_backtest_for(
    trader_id: str,
    params: dict,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> dict:
    symbols = _active_symbols(params)

    # Resolve timeframe from tf_idx param (0=15M, 1=1H)
    tf_idx = int(params.get("tf_idx", 0))
    timeframe = TF_OPTIONS[min(tf_idx, len(TF_OPTIONS) - 1)]

    # Create per-run stateful signal function
    if trader_id == "trader_1":
        sig_fn = SIGNAL_FUNCS["trader_1"]
    elif trader_id == "trader_2":
        sig_fn = _make_signal_t2(params)
    elif trader_id == "trader_3":
        sig_fn = _make_signal_t3(params)
    elif trader_id == "trader_5":
        sig_fn = _make_signal_t5(params)
    else:
        return {"trades": 0, "profit_factor": 0.0, "sharpe": 0.0,
                "max_drawdown": 1.0, "win_rate": 0.0, "expectancy": 0.0}

    # Load and filter all symbol dataframes
    symbol_dfs: dict = {}
    for symbol in symbols:
        df = _get_indicators(symbol, timeframe)
        if df.empty or len(df) < 200:
            continue
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        if len(df) < 200:
            continue
        symbol_dfs[symbol] = df

    if not symbol_dfs:
        return {"trades": 0, "profit_factor": 0.0, "sharpe": 0.0,
                "max_drawdown": 1.0, "win_rate": 0.0, "expectancy": 0.0}

    # Build merged sorted bar queue for correct cross-symbol cooldown
    bar_queue = []
    for symbol, df in symbol_dfs.items():
        for idx in range(200, len(df)):
            bar_queue.append((df.index[idx], symbol, idx))
    bar_queue.sort(key=lambda x: x[0])

    all_pnls = []
    equity = INITIAL_CAPITAL
    peak = equity
    current_date = None
    last_trade_bar: dict = {}   # per-symbol cooldown

    for dt, symbol, i in bar_queue:
        df = symbol_dfs[symbol]

        # Per-symbol cooldown
        if i - last_trade_bar.get(symbol, -COOLDOWN_BARS) < COOLDOWN_BARS:
            continue

        day_str = dt.strftime("%Y-%m-%d")
        if day_str != current_date:
            current_date = day_str

        # NOTE: circuit breaker intentionally disabled in optimizer backtest.
        # We optimize signal quality (PF, Sharpe) not capital survival.
        # The live trading circuit breaker remains in place in the real engine.

        if not _in_session(dt, trader_id):
            continue
        if dt.hour == 12:
            continue

        try:
            sig = sig_fn(df, i, symbol, params)
        except Exception:
            continue

        if sig is None:
            continue

        entry = float(sig["entry"]) * (1 + SLIPPAGE_PCT if sig["side"] == "buy" else 1 - SLIPPAGE_PCT)
        sl    = float(sig["sl"])
        tp1   = float(sig["tp1"])
        tp2   = float(sig["tp2"])
        atr   = float(sig["atr"])
        size  = _size_trade(entry, sl, equity)

        result = _simulate_trade(df, i, sig["side"], entry, sl, tp1, tp2, size, atr)
        pnl = result["pnl"]

        equity += pnl
        if equity > peak:
            peak = equity

        all_pnls.append(pnl)
        last_trade_bar[symbol] = i

    if len(all_pnls) < 5:
        return {"trades": len(all_pnls), "profit_factor": 0.0, "sharpe": 0.0,
                "max_drawdown": 1.0, "win_rate": 0.0, "expectancy": 0.0}

    wins   = [p for p in all_pnls if p > 0]
    losses = [p for p in all_pnls if p < 0]
    pf     = sum(wins) / (abs(sum(losses)) + 1e-9)
    wr     = len(wins) / len(all_pnls)
    exp    = float(np.mean(all_pnls))

    # Annualised Sharpe
    mean_r = np.mean(all_pnls)
    std_r  = np.std(all_pnls)
    sharpe = float((mean_r / (std_r + 1e-9)) * np.sqrt(252))

    # Max drawdown
    eq_curve = [INITIAL_CAPITAL]
    for p in all_pnls:
        eq_curve.append(eq_curve[-1] + p)
    peak2 = eq_curve[0]
    max_dd = 0.0
    for v in eq_curve:
        if v > peak2:
            peak2 = v
        dd = (peak2 - v) / (peak2 + 1e-9)
        if dd > max_dd:
            max_dd = dd

    return {
        "trades": len(all_pnls),
        "profit_factor": round(pf, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(wr, 4),
        "expectancy": round(exp, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION_AGENT — composite score
# ──────────────────────────────────────────────────────────────────────────────

def _composite_score(m: dict, trader_id: str = "", params: dict = None) -> float:
    n_symbols = len(_active_symbols(params)) if params else len(_ALL_SYMBOLS)
    min_t = MIN_TRADES_PER_PAIR_PER_YEAR * n_symbols * _TRAIN_YEARS
    if m["trades"] < min_t:
        return -999.0

    pf   = min(m["profit_factor"], 5.0)
    sr   = min(max(m["sharpe"], -3.0), 6.0)
    exp  = m["expectancy"]
    dd   = m["max_drawdown"]

    # Normalise
    pf_norm  = (pf - 1.0) / 4.0          # 0 at PF=1, 1 at PF=5
    sr_norm  = (sr + 3.0) / 9.0          # 0 at SR=-3, 1 at SR=6
    exp_norm = float(np.clip(exp / 50.0, -1.0, 1.0))   # $50 per trade → 1.0
    dd_pen   = dd                          # 0=best, 1=worst

    score = (W_PF * pf_norm) + (W_SR * sr_norm) + (W_EXP * exp_norm) - (W_DD * dd_pen)
    return round(score, 6)


# ──────────────────────────────────────────────────────────────────────────────
# OVERFITTING_GUARD_AGENT
# ──────────────────────────────────────────────────────────────────────────────

def _validate_oos(trader_id: str, params: dict, train_metrics: dict) -> dict:
    """Run on val split; reject if val score < 60% of train score."""
    val_m = _run_backtest_for(trader_id, params, start=VAL_START)
    val_score = _composite_score(val_m, trader_id, params)
    train_score = _composite_score(train_metrics, trader_id, params)

    # Val period is ~3 years (2023-2026); apply same 15/pair/year rule
    _VAL_YEARS = 3
    n_symbols = len(_active_symbols(params))
    min_t_val = MIN_TRADES_PER_PAIR_PER_YEAR * n_symbols * _VAL_YEARS

    # Must pass minimum gates on val data
    valid = (
        val_m["profit_factor"] >= GATE_PF
        and val_m["sharpe"] >= GATE_SR
        and val_m["max_drawdown"] <= GATE_DD
        and val_m["trades"] >= min_t_val
    )

    # Penalise train/val divergence > 40%
    if train_score > 0 and val_score < train_score * 0.60:
        valid = False

    return {
        "valid": valid,
        "val_metrics": val_m,
        "val_score": val_score,
        "train_score": train_score,
        "divergence": round(abs(train_score - val_score) / (abs(train_score) + 1e-9), 3),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CRITIC_AGENT
# ──────────────────────────────────────────────────────────────────────────────

def _critic_check(trader_id: str, params: dict, space: dict) -> Tuple[bool, str]:
    """Reject unrealistic param configs."""
    # Minimum 3 pairs must be active to prevent overfitting to a single pair
    active = _active_symbols(params)
    if len(active) < 3:
        return False, f"only {len(active)} pairs active (min 3)"
    # TP2 must be > TP1
    if trader_id == "trader_1":
        if params["tp2_atr_mult"] <= params["tp1_atr_mult"]:
            return False, "tp2 <= tp1"
        if params["rsi_high"] <= params["rsi_low"]:
            return False, "rsi_high <= rsi_low"
    if trader_id == "trader_3":
        if params["tp2_range_mult_fx"] <= params["tp1_range_mult"]:
            return False, "tp2_fx <= tp1"
        if params["asian_range_max_fx"] <= params["asian_range_min_fx"]:
            return False, "asian range max <= min"
    if trader_id == "trader_5":
        if params["rsi_overbought"] <= params["rsi_oversold"]:
            return False, "RSI overbought <= oversold"
        if params["stoch_overbought"] <= params["stoch_oversold"]:
            return False, "Stoch OB <= OS"
    return True, "ok"


# ──────────────────────────────────────────────────────────────────────────────
# SEARCH_AGENT — Bayesian-like search via scipy minimize + random restarts
# ──────────────────────────────────────────────────────────────────────────────

def _space_to_array(space: dict, params: dict) -> np.ndarray:
    keys = sorted(space.keys())
    return np.array([float(params[k]) for k in keys])


def _array_to_params(space: dict, x: np.ndarray) -> dict:
    keys = sorted(space.keys())
    params = {}
    for k, v in zip(keys, x):
        spec = space[k]
        if spec.get("integer"):
            params[k] = int(round(float(v)))
        else:
            params[k] = float(round(v, 6))
    return params


def _clip_to_bounds(space: dict, x: np.ndarray) -> np.ndarray:
    keys = sorted(space.keys())
    for i, k in enumerate(keys):
        spec = space[k]
        x[i] = np.clip(x[i], spec["low"], spec["high"])
    return x


def _random_params(space: dict, rng: np.random.Generator) -> dict:
    params = {}
    for k, spec in space.items():
        v = rng.uniform(spec["low"], spec["high"])
        if spec.get("integer"):
            params[k] = int(round(v))
        else:
            params[k] = round(float(v), 6)
    return params


def _default_params(space: dict) -> dict:
    return {k: v["default"] for k, v in space.items()}


class BayesianOptimizer:
    """
    Simple Gaussian Process surrogate + EI acquisition (scipy-based).
    Falls back to random search if scipy.optimize is unavailable.
    """

    def __init__(self, space: dict, n_initial: int = 8):
        self.space = space
        self.keys  = sorted(space.keys())
        self.n_dim = len(self.keys)
        self.X: List[np.ndarray] = []
        self.Y: List[float] = []
        self.n_initial = n_initial
        self._rng = np.random.default_rng(42)
        self._evaluated_count = 0

    def suggest(self) -> dict:
        self._evaluated_count += 1

        # Initial random exploration
        if len(self.X) < self.n_initial:
            return _random_params(self.space, self._rng)

        # Gaussian Process surrogate
        try:
            from scipy.optimize import minimize
            from scipy.stats import norm

            X = np.array(self.X)
            Y = np.array(self.Y)

            # Normalise Y
            y_mean = Y.mean()
            y_std  = Y.std() + 1e-9
            Y_norm = (Y - y_mean) / y_std

            # Simple RBF kernel
            def rbf(x1, x2, length_scale=1.0):
                diff = (x1 - x2) / (length_scale + 1e-9)
                return np.exp(-0.5 * np.dot(diff, diff))

            ls = 1.0  # length scale
            K = np.array([[rbf(X[i], X[j], ls) for j in range(len(X))] for i in range(len(X))])
            K += np.eye(len(X)) * 1e-6

            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))
            except np.linalg.LinAlgError:
                return _random_params(self.space, self._rng)

            def expected_improvement(x):
                k_star = np.array([rbf(x, xi, ls) for xi in X])
                mu  = float(np.dot(k_star, alpha))
                v   = np.linalg.solve(L, k_star)
                sig = float(np.sqrt(max(rbf(x, x, ls) - np.dot(v, v), 1e-9)))

                y_best = Y_norm.max()
                z = (mu - y_best) / (sig + 1e-9)
                ei = sig * (z * norm.cdf(z) + norm.pdf(z))
                return -float(ei)   # minimise negative EI

            # Random restarts
            best_x = None
            best_ei = np.inf
            bounds = [(self.space[k]["low"], self.space[k]["high"]) for k in self.keys]

            for _ in range(5):
                x0 = np.array([self._rng.uniform(b[0], b[1]) for b in bounds])
                try:
                    res = minimize(expected_improvement, x0, method="L-BFGS-B",
                                   bounds=bounds, options={"maxiter": 100})
                    if res.fun < best_ei:
                        best_ei = res.fun
                        best_x  = res.x
                except Exception:
                    pass

            if best_x is not None:
                best_x = _clip_to_bounds(self.space, best_x)
                return _array_to_params(self.space, best_x)

        except ImportError:
            pass

        return _random_params(self.space, self._rng)

    def observe(self, params: dict, score: float):
        x = _space_to_array(self.space, params)
        self.X.append(x)
        self.Y.append(score)


# ──────────────────────────────────────────────────────────────────────────────
# OPTIMIZATION LOOP
# ──────────────────────────────────────────────────────────────────────────────

def _write_best_checkpoint(
    trader_id: str,
    params: dict,
    train_metrics: dict,
    score: float,
    iteration: int,
) -> None:
    """Write current best params to disk immediately so no progress is lost on interruption."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"best_{trader_id}_checkpoint.json")
    active_pairs = _active_symbols(params)
    payload = {
        "trader_id":     trader_id,
        "iteration":     iteration,
        "score":         score,
        "active_pairs":  active_pairs,
        "n_pairs":       len(active_pairs),
        "best_params":   params,
        "train_metrics": train_metrics,
        "saved_at":      datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("  [checkpoint] best params written → %s", path)


def _optimize_trader(
    trader_id: str,
    n_iter: int = 40,
    n_initial: int = 8,
) -> dict:
    space = SEARCH_SPACES[trader_id]

    logger.info("=" * 60)
    logger.info("PARAMETER_AGENT: %s — %d parameters", trader_id, len(space))
    for k, spec in space.items():
        logger.info("  %-25s [%.4f – %.4f] default=%.4f",
                    k, spec["low"], spec["high"], spec["default"])

    optimizer = BayesianOptimizer(space, n_initial=n_initial)

    best_score  = -999.0
    best_params = _default_params(space)
    _empty_metrics = {"trades": 0, "profit_factor": 0.0, "sharpe": 0.0,
                      "max_drawdown": 1.0, "win_rate": 0.0, "expectancy": 0.0}
    best_train  = _empty_metrics.copy()
    best_val    = _empty_metrics.copy()
    history     = []
    no_improve  = 0

    # Always evaluate the default first
    candidates = [_default_params(space)]

    logger.info("SEARCH_AGENT: starting %d iterations (train: 2016-2022, val: 2023-2026)", n_iter)

    for it in range(n_iter + len(candidates)):
        if it < len(candidates):
            params = candidates[it]
        else:
            params = optimizer.suggest()

        # CRITIC_AGENT pre-screen
        ok, reason = _critic_check(trader_id, params, space)
        if not ok:
            logger.debug("  iter %3d  CRITIC rejected: %s", it + 1, reason)
            optimizer.observe(params, -1.0)
            continue

        # BACKTEST_AGENT — train slice
        try:
            train_m = _run_backtest_for(trader_id, params, end=TRAIN_END)
        except Exception as e:
            logger.debug("  iter %3d  backtest error: %s", it + 1, e)
            optimizer.observe(params, -1.0)
            continue

        score = _composite_score(train_m, trader_id, params)

        logger.info(
            "  iter %3d  PF=%5.2f  SR=%6.2f  DD=%5.1f%%  WR=%5.1f%%  trades=%4d  score=%7.4f",
            it + 1,
            train_m["profit_factor"],
            train_m["sharpe"],
            train_m["max_drawdown"] * 100,
            train_m["win_rate"] * 100,
            train_m["trades"],
            score,
        )

        optimizer.observe(params, score)
        history.append({"iter": it + 1, "params": deepcopy(params), "train": train_m, "score": score})

        if score > best_score:
            best_score  = score
            best_params = deepcopy(params)
            best_train  = train_m
            no_improve  = 0
            logger.info("  *** NEW BEST  score=%.4f  PF=%.2f  SR=%.2f trades=%d ***",
                        score, train_m["profit_factor"], train_m["sharpe"], train_m["trades"])
            _write_best_checkpoint(trader_id, best_params, best_train, best_score, it + 1)
        elif score == -999.0 and train_m["trades"] > best_train.get("trades", 0):
            # No config met trade threshold yet — track highest-frequency config
            # as a fallback so best_params always holds *something* usable
            best_params = deepcopy(params)
            best_train  = train_m
            no_improve += 1
        else:
            no_improve += 1

        # Early stop: no improvement for 20 iterations after initial exploration
        if no_improve >= 20 and it >= n_initial:
            logger.info("  Early stop: no improvement for 20 iterations")
            break

    # OVERFITTING_GUARD
    logger.info("OVERFITTING_GUARD: validating best config on OOS data (2023-2026)...")
    oos = _validate_oos(trader_id, best_params, best_train)
    logger.info(
        "  Val  PF=%5.2f  SR=%6.2f  DD=%5.1f%%  trades=%4d  valid=%s  divergence=%.2f",
        oos["val_metrics"]["profit_factor"],
        oos["val_metrics"]["sharpe"],
        oos["val_metrics"]["max_drawdown"] * 100,
        oos["val_metrics"]["trades"],
        oos["valid"],
        oos["divergence"],
    )

    if not oos["valid"]:
        logger.warning("  Overfitting detected! Falling back to best valid config from history.")
        # Try next-best configs from history in score order
        for entry in sorted(history, key=lambda x: x["score"], reverse=True)[1:20]:
            ok, _ = _critic_check(trader_id, entry["params"], space)
            if not ok:
                continue
            oos2 = _validate_oos(trader_id, entry["params"], entry["train"])
            if oos2["valid"]:
                best_params = entry["params"]
                best_train  = entry["train"]
                best_score  = entry["score"]
                oos         = oos2
                logger.info("  Fallback valid config found. score=%.4f", best_score)
                _write_best_checkpoint(trader_id, best_params, best_train, best_score, -1)
                break
        else:
            logger.warning("  No valid fallback found — using best-scoring anyway (flag for manual review).")

    return {
        "trader_id": trader_id,
        "best_params": best_params,
        "train_metrics": best_train,
        "val_metrics": oos["val_metrics"],
        "train_score": oos["train_score"],
        "val_score": oos["val_score"],
        "oos_valid": oos["valid"],
        "divergence": oos["divergence"],
        "iterations_run": len(history),
    }


# ──────────────────────────────────────────────────────────────────────────────
# PORTFOLIO_AGENT
# ──────────────────────────────────────────────────────────────────────────────

def _run_portfolio_backtest(trader_results: dict) -> dict:
    """
    Run all traders with their optimized params simultaneously.
    Compute portfolio-level metrics and suggest capital allocation.
    """
    logger.info("=" * 60)
    logger.info("PORTFOLIO_AGENT: running combined portfolio backtest...")

    equity_curves: Dict[str, List[float]] = {}
    all_metrics: Dict[str, dict] = {}

    for tid, res in trader_results.items():
        params = res["best_params"]
        try:
            m = _run_backtest_for(tid, params)
            all_metrics[tid] = m

            # Rebuild equity curve for correlation analysis
            symbols = TRADER_SYMBOLS.get(tid, ["EURUSD"])
            eq = [INITIAL_CAPITAL]
            # Use trade-level equity if stored, else approximate
            equity_curves[tid] = eq

        except Exception as e:
            logger.error("Portfolio backtest failed for %s: %s", tid, e)
            all_metrics[tid] = {"profit_factor": 0.0, "sharpe": 0.0,
                                 "max_drawdown": 1.0, "trades": 0,
                                 "win_rate": 0.0, "expectancy": 0.0}

    # Capital allocation via Sharpe-ratio weighting
    sharpes = {tid: max(m["sharpe"], 0.0) for tid, m in all_metrics.items()}
    total_sharpe = sum(sharpes.values()) + 1e-9
    allocation = {}
    for tid in all_metrics:
        if total_sharpe > 0:
            allocation[tid] = round(sharpes[tid] / total_sharpe, 4)
        else:
            allocation[tid] = round(1.0 / len(all_metrics), 4)

    # Portfolio-level aggregate score
    pf_weighted = sum(
        allocation[tid] * all_metrics[tid]["profit_factor"]
        for tid in all_metrics
    )
    sr_weighted = sum(
        allocation[tid] * all_metrics[tid]["sharpe"]
        for tid in all_metrics
    )
    dd_max = max(m["max_drawdown"] for m in all_metrics.values())

    logger.info("Portfolio results:")
    for tid, m in sorted(all_metrics.items()):
        logger.info("  %-10s  PF=%5.2f  SR=%6.2f  DD=%5.1f%%  alloc=%.1f%%",
                    tid, m["profit_factor"], m["sharpe"],
                    m["max_drawdown"] * 100, allocation[tid] * 100)
    logger.info("  PORTFOLIO   wPF=%5.2f  wSR=%6.2f  maxDD=%5.1f%%",
                pf_weighted, sr_weighted, dd_max * 100)

    return {
        "trader_metrics": all_metrics,
        "capital_allocation": allocation,
        "portfolio_profit_factor": round(pf_weighted, 4),
        "portfolio_sharpe": round(sr_weighted, 4),
        "portfolio_max_drawdown": round(dd_max, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# REPORT
# ──────────────────────────────────────────────────────────────────────────────

def _print_report(trader_results: dict, portfolio: dict):
    BANNER = "=" * 65

    print(f"\n{BANNER}")
    print("  QUANTITATIVE RESEARCH AGENT — OPTIMIZATION REPORT")
    print(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(BANNER)

    for tid, res in sorted(trader_results.items()):
        tm = res["train_metrics"]
        vm = res["val_metrics"]
        print(f"\n{'─' * 65}")
        print(f"  {tid.upper()}")
        print(f"{'─' * 65}")
        print(f"  {'Metric':<22}  {'Train (2016-2022)':>18}  {'Val (2023-2026)':>16}")
        print(f"  {'-'*22}  {'-'*18}  {'-'*16}")
        print(f"  {'Profit Factor':<22}  {tm['profit_factor']:>18.3f}  {vm['profit_factor']:>16.3f}")
        print(f"  {'Sharpe Ratio':<22}  {tm['sharpe']:>18.3f}  {vm['sharpe']:>16.3f}")
        print(f"  {'Max Drawdown':<22}  {tm['max_drawdown']*100:>17.1f}%  {vm['max_drawdown']*100:>15.1f}%")
        print(f"  {'Win Rate':<22}  {tm['win_rate']*100:>17.1f}%  {vm['win_rate']*100:>15.1f}%")
        print(f"  {'Trades':<22}  {tm['trades']:>18d}  {vm['trades']:>16d}")
        print(f"  {'OOS Valid':<22}  {'':>18}  {str(res['oos_valid']):>16}")
        print(f"  {'Divergence':<22}  {'':>18}  {res['divergence']:>15.2%}")
        # Show selected pairs separately
        active_pairs = _active_symbols(res["best_params"])
        excluded = [s for s in _ALL_SYMBOLS if s not in active_pairs]
        print(f"\n  PAIRS SELECTED ({len(active_pairs)}/11): {', '.join(active_pairs)}")
        if excluded:
            print(f"  EXCLUDED: {', '.join(excluded)}")
        print(f"\n  OPTIMIZED PARAMETERS:")
        for k, v in sorted(res["best_params"].items()):
            if not k.startswith("pair_"):
                print(f"    {k:<28} {v}")

    pa = portfolio["capital_allocation"]
    print(f"\n{'=' * 65}")
    print("  PORTFOLIO SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Weighted Profit Factor : {portfolio['portfolio_profit_factor']:.3f}")
    print(f"  Weighted Sharpe Ratio  : {portfolio['portfolio_sharpe']:.3f}")
    print(f"  Max Drawdown (any leg) : {portfolio['portfolio_max_drawdown']*100:.1f}%")
    print(f"\n  CAPITAL ALLOCATION RECOMMENDATION:")
    for tid, alloc in sorted(pa.items()):
        print(f"    {tid:<12} {alloc*100:.1f}%")

    print(f"\n  VALIDATION SUMMARY:")
    for tid, res in sorted(trader_results.items()):
        tag = "PASS" if res["oos_valid"] else "FAIL (review params)"
        print(f"    {tid:<12} {tag}")

    print(f"\n{BANNER}\n")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parameter optimization agent")
    parser.add_argument("--traders", nargs="*", type=int, default=[1, 2, 3, 5])
    parser.add_argument("--iterations", type=int, default=40,
                        help="Search iterations per trader (default 40)")
    parser.add_argument("--initial", type=int, default=8,
                        help="Random exploration iterations before Bayesian (default 8)")
    parser.add_argument("--output", default="optimized_params.json",
                        help="Output file (default: optimized_params.json)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = _setup_log_file(OUTPUT_DIR)
    logger.info("Log file: %s", log_path)
    t_start = time.time()

    trader_ids = [f"trader_{t}" for t in args.traders if f"trader_{t}" in SEARCH_SPACES]
    if not trader_ids:
        logger.error("No valid trader IDs in search spaces: %s", list(SEARCH_SPACES.keys()))
        sys.exit(1)

    logger.info("Starting optimization for: %s", trader_ids)
    logger.info("Iterations per trader: %d  |  Initial random: %d", args.iterations, args.initial)
    logger.info("Train: 2016–2022  |  Val: 2023–2026")

    # Pre-warm indicator cache for all timeframes
    logger.info("Pre-loading and computing indicators (15M + 1H)...")
    all_symbols = set()
    for tid in trader_ids:
        all_symbols.update(TRADER_SYMBOLS.get(tid, []))
    for tf in TF_OPTIONS:
        for sym in sorted(all_symbols):
            _get_indicators(sym, tf)
    logger.info("Indicator cache ready.")

    trader_results: Dict[str, dict] = {}

    for tid in trader_ids:
        logger.info("\nOptimizing %s...", tid)
        result = _optimize_trader(tid, n_iter=args.iterations, n_initial=args.initial)
        trader_results[tid] = result

    # Portfolio validation
    portfolio = _run_portfolio_backtest(trader_results)

    # Print report
    _print_report(trader_results, portfolio)

    # Save to file
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "traders": trader_ids,
            "iterations": args.iterations,
            "train_end": str(TRAIN_END.date()),
            "val_start": str(VAL_START.date()),
        },
        "trader_results": trader_results,
        "portfolio": portfolio,
    }

    outpath = os.path.join(OUTPUT_DIR, args.output)
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = round(time.time() - t_start, 1)
    logger.info("Optimization complete in %.1fs → %s", elapsed, outpath)
    print(f"Results saved to: {outpath}")


if __name__ == "__main__":
    main()
