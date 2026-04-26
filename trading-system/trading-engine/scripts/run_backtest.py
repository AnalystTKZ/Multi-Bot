#!/usr/bin/env python3
"""
run_backtest.py — ML-native backtest. Single unified ml_trader across all 11 symbols.

ICT conditions are encoded as numeric features (SEQUENCE_FEATURES). The GRU/EV stack
decides direction, sizing, and entry — no rule-based trader branches.

Usage:
    python run_backtest.py
    python run_backtest.py --start 2022-01-01 --end 2024-12-31

Output: backtest_results/backtest_YYYYMMDD_HHMMSS.json
"""

from __future__ import annotations

import argparse
from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import csv
import gc
import hashlib
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

_ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _ENGINE_DIR)
# Models use relative "weights/..." paths — must run from trading-engine/
os.chdir(_ENGINE_DIR)

# env_config resolves processed_data/ to the Kaggle dataset mount on Kaggle,
# or to trading-system/processed_data/ locally. Without this, DATA_DIR pointed
# at trading-system/processed_data/histdata/ which doesn't exist on Kaggle
# (data is mounted at /kaggle/input/<slug>/processed_data/histdata/).
_TS_DIR = os.path.join(_ENGINE_DIR, "..")
sys.path.insert(0, _TS_DIR)
try:
    from env_config import get_env as _get_env
    _ENV = _get_env()
    _DATA_DIR_RESOLVED = str(_ENV["processed"] / "histdata")
except Exception:
    _DATA_DIR_RESOLVED = os.path.join(_TS_DIR, "processed_data", "histdata")

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backtest")

# ── GPU/CPU performance flags ──────────────────────────────────────────────────
try:
    import torch as _bt_torch
    if _bt_torch.cuda.is_available():
        _bt_torch.backends.cudnn.benchmark        = True
        _bt_torch.backends.cuda.matmul.allow_tf32 = True
        _bt_torch.backends.cudnn.allow_tf32        = True
    _bt_n_cpu = int(os.getenv("RETRAIN_CPU_WORKERS", "4"))
    _bt_torch.set_num_threads(_bt_n_cpu)
    _bt_torch.set_num_interop_threads(max(1, _bt_n_cpu // 2))
except Exception:
    pass

# ─── Config ───────────────────────────────────────────────────────────────────
INITIAL_CAPITAL    = 10000.0
RISK_PER_TRADE     = 0.01       # 1% per trade
CAPITAL_PER_TRADER = 0.20       # 20% of account allocated per trader
COMMISSION_PCT     = 0.001
SLIPPAGE_PCT       = 0.0002
MAX_DAILY_LOSS_PCT = 0.02       # 2% daily circuit breaker
MAX_DRAWDOWN_PCT   = 0.20       # 20% portfolio halt
COOLDOWN_BARS      = 10         # bars between signals per symbol
MIN_CONFIDENCE     = 0.70       # minimum signal confidence — PM R:R gate also enforces this
MAX_HOLD_BARS      = 200        # max bars before time-exit
DATA_DIR           = _DATA_DIR_RESOLVED
OUTPUT_DIR         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backtest_results")
ML_CACHE_DIR       = os.path.join(OUTPUT_DIR, "ml_cache")
_BACKTEST_WARMUP = {
    "5M": pd.Timedelta(days=7),
    "15M": pd.Timedelta(days=14),
    "1H": pd.Timedelta(days=60),
    "4H": pd.Timedelta(days=180),
    "1D": pd.Timedelta(days=400),
}

# PM settings object — mirrors what the live engine passes to PortfolioManager
_PM_SETTINGS = SimpleNamespace(
    ACCOUNT_BALANCE=INITIAL_CAPITAL,
    CAPITAL_PER_TRADER=CAPITAL_PER_TRADER,
    RISK_PER_TRADE=RISK_PER_TRADE,
    MAX_DAILY_LOSS_PCT=MAX_DAILY_LOSS_PCT,
)

_ALL_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD",
    "USDCAD", "USDCHF", "EURGBP", "EURJPY", "GBPJPY", "XAUUSD",
]

# Single unified ML trader replaces all 5 ICT rule branches.
# ICT conditions are encoded as numeric features; the GRU/EV stack decides trades.
TRADER_SYMBOLS = {
    "ml_trader": _ALL_SYMBOLS,
}

TRADER_NAMES = {
    "ml_trader": "ML-Native Execution (GRU + EV)",
}

_HTF_REGIME_NAMES = np.array(["BIAS_UP", "BIAS_DOWN", "BIAS_NEUTRAL"], dtype=object)
_LTF_REGIME_NAMES = np.array(["TRENDING", "RANGING", "CONSOLIDATING", "VOLATILE"], dtype=object)


def _env_ml_enabled() -> bool:
    """Avoid importing config.settings (pydantic) — Kaggle images can ship mismatched pydantic_core."""
    v = os.getenv("ML_ENABLED", "true").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return True


def _csv_columns(path: str) -> list[str]:
    with open(path, newline="") as fh:
        return next(csv.reader(fh), [])


def _utc_ts(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _filter_date_range(df: pd.DataFrame, start_ts: pd.Timestamp | None, end_ts: pd.Timestamp | None) -> pd.DataFrame:
    if start_ts is not None:
        df = df.loc[df.index >= start_ts]
    if end_ts is not None:
        df = df.loc[df.index <= end_ts]
    return df


def _start_with_warmup(start: str | None, timeframe: str) -> str | None:
    start_ts = _utc_ts(start)
    if start_ts is None:
        return start
    warmup = _BACKTEST_WARMUP.get(timeframe.upper())
    if warmup is None:
        return start
    return (start_ts - warmup).strftime("%Y-%m-%d")


def _weight_token(path_str: str | None) -> str:
    if not path_str:
        return "missing"
    path = Path(path_str)
    if not path.exists():
        return "missing"
    if path.is_file():
        st = path.stat()
        return f"{path.name}:{st.st_mtime_ns}:{st.st_size}"
    mtimes = []
    sizes = 0
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        st = child.stat()
        mtimes.append(str(st.st_mtime_ns))
        sizes += st.st_size
    payload = "|".join(mtimes) if mtimes else "empty"
    return f"{path.name}:{sizes}:{payload}"


def _ml_cache_file(symbol: str, start: str, end: str, ml_models: dict) -> str:
    tokens = [symbol, start or "", end or ""]
    relevant = {"gru_lstm", "regime", "regime_htf", "regime_4h", "regime_ltf", "regime_1h"}
    for name in sorted(k for k in (ml_models or {}).keys() if k in relevant):
        model = ml_models[name]
        tokens.append(f"{name}={_weight_token(getattr(model, 'weight_path', None))}")
    digest = hashlib.sha1("||".join(tokens).encode("utf-8")).hexdigest()[:16]
    os.makedirs(ML_CACHE_DIR, exist_ok=True)
    return os.path.join(ML_CACHE_DIR, f"{symbol}_{start}_{end}_{digest}.npz")


def _save_ml_cache(path: str, cache: dict[str, np.ndarray]) -> None:
    tmp = f"{path}.tmp.npz"
    np.savez_compressed(tmp, **cache)
    os.replace(tmp, path)


def _load_ml_cache(path: str, expected_len: int) -> dict[str, np.ndarray] | None:
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            cache = {k: payload[k] for k in payload.files}
        if not cache:
            return None
        for arr in cache.values():
            if len(arr) != expected_len:
                return None
        return cache
    except Exception as exc:
        logger.warning("ML cache load failed %s: %s", path, exc)
        return None


def _ml_cache_entry(cache: dict[str, np.ndarray] | None, idx: int) -> dict:
    if not cache:
        return {}
    out: dict = {}
    p_bull = float(cache["p_bull"][idx])
    if not np.isnan(p_bull):
        out["p_bull"] = p_bull
        out["p_bear"] = float(cache["p_bear"][idx])
        out["expected_variance"] = float(cache["expected_variance"][idx])
    regime_id = int(cache["regime"][idx])
    if regime_id >= 0:
        out["regime"] = str(_HTF_REGIME_NAMES[regime_id])
    regime_ltf_id = int(cache["regime_ltf"][idx])
    if regime_ltf_id >= 0:
        out["regime_ltf"] = str(_LTF_REGIME_NAMES[regime_ltf_id])
    return out


def _read_market_csv_chunked(
    path: str,
    usecols: list[str],
    index_col: str,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
    *,
    engine: str | None = None,
) -> pd.DataFrame:
    read_kwargs = {
        "usecols": usecols,
        "index_col": index_col,
        "parse_dates": [index_col],
        "chunksize": 50_000,
    }
    if engine is not None:
        read_kwargs["engine"] = engine
    else:
        read_kwargs["dtype"] = {col: "float32" for col in usecols[1:]}

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, **read_kwargs):
        chunk.columns = [c.lower() for c in chunk.columns]
        numeric_cols = [c for c in ("open", "high", "low", "close", "volume") if c in chunk.columns]
        if numeric_cols:
            chunk[numeric_cols] = chunk[numeric_cols].apply(pd.to_numeric, errors="coerce")
        chunk.index = pd.to_datetime(chunk.index, utc=True, errors="coerce")
        chunk = chunk.loc[~chunk.index.isna()]
        chunk = _filter_date_range(chunk, start_ts, end_ts)
        if chunk.empty:
            continue
        chunks.append(chunk.dropna(subset=["open", "high", "low", "close"]))

    if not chunks:
        return pd.DataFrame(columns=[c.lower() for c in usecols[1:]])
    return pd.concat(chunks).sort_index()


def _read_market_csv_streaming(
    path: str,
    usecols: list[str],
    index_col: str,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
) -> pd.DataFrame:
    timestamps = array("q")
    col_buffers = {
        col.lower(): array("f")
        for col in usecols[1:]
    }

    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header:
            return pd.DataFrame(columns=[c.lower() for c in usecols[1:]])

        header_positions = {name: idx for idx, name in enumerate(header)}
        required_positions = {}
        for col in usecols:
            if col not in header_positions:
                raise ValueError(f"Missing expected column {col!r} in {path}")
            required_positions[col] = header_positions[col]

        for raw_row in reader:
            if not raw_row:
                continue
            try:
                ts = _utc_ts(raw_row[required_positions[index_col]])
            except (IndexError, ValueError, TypeError):
                continue
            if ts is None:
                continue

            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue

            valid = True
            parsed_values: dict[str, float] = {}
            for col in usecols[1:]:
                try:
                    value = float(raw_row[required_positions[col]])
                except (IndexError, ValueError, TypeError):
                    valid = False
                    break
                parsed_values[col.lower()] = value
            if valid:
                timestamps.append(ts.value)
                for col_name, value in parsed_values.items():
                    col_buffers[col_name].append(value)

    if not timestamps:
        return pd.DataFrame(columns=[c.lower() for c in usecols[1:]])

    data = {
        col_name: np.fromiter(values, dtype=np.float32, count=len(values))
        for col_name, values in col_buffers.items()
    }
    index = pd.to_datetime(np.fromiter(timestamps, dtype=np.int64, count=len(timestamps)), utc=True)
    df = pd.DataFrame(data, index=index)
    return df.dropna(subset=["open", "high", "low", "close"])


def _load_csv(symbol: str, timeframe: str = "15M", start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Load OHLCV from processed_data/histdata/{SYM}_{TF}.parquet (step0 pipeline output).
    Only the requested date window is loaded into RAM — pyarrow row-group filtering
    avoids reading the full 9-year file when start/end are specified.
    No CSV fallbacks: run pipeline/step0_resample.py first if parquet is missing.
    """
    tf_upper = timeframe.upper()
    parquet_path = os.path.join(DATA_DIR, f"{symbol}_{tf_upper}.parquet")
    if not os.path.exists(parquet_path):
        logger.warning("Missing parquet %s — run pipeline/step0_resample.py first", parquet_path)
        return pd.DataFrame()

    try:
        start_ts = _utc_ts(_start_with_warmup(start, tf_upper))
        end_ts   = _utc_ts(end)

        # Push date range into the parquet read when both bounds are known.
        # pyarrow uses row-group min/max statistics to skip groups outside the
        # filter range — no full file scan required.
        if start_ts is not None or end_ts is not None:
            try:
                import pyarrow.parquet as _pq
                _filters = []
                if start_ts is not None:
                    _filters.append(("index", ">=", start_ts))
                if end_ts is not None:
                    _filters.append(("index", "<=", end_ts))
                _tbl = _pq.read_table(parquet_path, filters=_filters)
                df = _tbl.to_pandas()
                del _tbl
            except Exception:
                # Fallback: full read + slice
                df = pd.read_parquet(parquet_path)
                df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
                df = _filter_date_range(df, start_ts, end_ts)
        else:
            df = pd.read_parquet(parquet_path)

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df[df.index.notna()].sort_index()
        df.columns = [c.lower() for c in df.columns]
        ohlcv = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[ohlcv].dropna(subset=["open", "high", "low", "close"])

        logger.info("Loaded parquet %s/%s: %d bars (%s → %s)",
                    symbol, tf_upper, len(df),
                    df.index.min().date() if len(df) else "?",
                    df.index.max().date() if len(df) else "?")
        return df
    except Exception as exc:
        logger.error("Failed to load parquet %s/%s: %s", symbol, tf_upper, exc)
        return pd.DataFrame()


def _model_ready(model) -> bool:
    return bool(getattr(model, "_loaded", False) and getattr(model, "_model", None) is not None)


def _ensure_mpl_config_dir() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return
    mpl_dir = os.path.join("/tmp", f"matplotlib-{os.getuid()}")
    os.makedirs(mpl_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = mpl_dir



def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    from indicators.market_structure import compute_all
    return compute_all(df)


def _load_htf_context(symbol: str, start: str, end: str) -> dict:
    """
    Load and precompute HTF context frames for a symbol.

    Returns a dict with keys "5M", "1H", "4H", "1D" — all TFs used by ML models
    for MTF feature computation (regime adx/ema_stack/atr/bb per TF, GRU cross-TF features).

    All filtered to [start, end] and indexed by UTC timestamp for asof lookup.
    """
    out = {}
    for tf in ("5M", "1H", "4H", "1D"):
        df = _load_csv(symbol, tf, start=start, end=end)
        if df is None or df.empty or len(df) < 14:
            out[tf] = pd.DataFrame()
            continue
        df = _compute_indicators(df)
        # Pre-compute 1H EMA21 slope: positive = rising, negative = falling
        if tf == "1H" and "ema_21" in df.columns:
            df["ema_21_slope"] = df["ema_21"].diff(3)   # change over 3 bars (3H)
        out[tf] = df
        logger.info("HTF context %s/%s: %d bars", symbol, tf, len(df))
    return out


def _build_htf_aligned(base_index: pd.DatetimeIndex, htf: dict) -> dict:
    """
    Forward-fill all HTF lookup columns onto the 15M base index once, returning
    plain numpy arrays indexed by bar position.  Replaces per-bar searchsorted +
    iloc calls (~50-100 µs each) with O(1) array reads.

    Returned dict keys (all np.ndarray aligned to base_index):
      1h_adx, 1h_ema_stack, 1h_ema_21_slope, 1h_rsi
      4h_ema_stack, 4h_bos_bull, 4h_bos_bear, 4h_fvg_bull, 4h_fvg_bear,
      4h_last_swing_high, 4h_last_swing_low, 4h_atr, 4h_rsi,
      4h_recent_bos_bull, 4h_recent_bos_bear   (rolling-3-bar any)
      1d_ema_stack
    """
    def _align(tf_df: pd.DataFrame, col: str, default=0.0) -> np.ndarray:
        if tf_df is None or tf_df.empty or col not in tf_df.columns:
            return np.full(len(base_index), default, dtype=np.float32)
        s = tf_df[col].reindex(base_index, method="ffill").fillna(default)
        return s.to_numpy(dtype=np.float32)

    df_1h = htf.get("1H", pd.DataFrame())
    df_4h = htf.get("4H", pd.DataFrame())
    df_1d = htf.get("1D", pd.DataFrame())

    arr: dict[str, np.ndarray] = {
        "1h_adx":          _align(df_1h, "adx_14",      50.0),
        "1h_ema_stack":    _align(df_1h, "ema_stack",    0.0),
        "1h_ema_21_slope": _align(df_1h, "ema_21_slope", 0.0),
        "1h_rsi":          _align(df_1h, "rsi_14",       50.0),
        "4h_ema_stack":    _align(df_4h, "ema_stack",    0.0),
        "4h_bos_bull":     _align(df_4h, "bos_bull",     0.0),
        "4h_bos_bear":     _align(df_4h, "bos_bear",     0.0),
        "4h_fvg_bull":     _align(df_4h, "fvg_bull",     0.0),
        "4h_fvg_bear":     _align(df_4h, "fvg_bear",     0.0),
        "4h_last_swing_high": _align(df_4h, "last_swing_high", np.nan),
        "4h_last_swing_low":  _align(df_4h, "last_swing_low",  np.nan),
        "4h_atr":          _align(df_4h, "atr_14",       0.001),
        "4h_rsi":          _align(df_4h, "rsi_14",       50.0),
        "1d_ema_stack":    _align(df_1d, "ema_stack",    0.0),
    }

    # Rolling-3-bar 4H BOS: did any of the last 3 4H bars have a BOS?
    # Computed on the 4H frame then forward-filled — avoids per-bar slice.
    for side in ("bull", "bear"):
        col = f"bos_{side}"
        if df_4h is not None and not df_4h.empty and col in df_4h.columns:
            roll = df_4h[col].astype(float).rolling(3, min_periods=1).max()  # max==any for 0/1
            arr[f"4h_recent_bos_{side}"] = (
                roll.reindex(base_index, method="ffill").fillna(0).to_numpy(dtype=np.float32)
            )
        else:
            arr[f"4h_recent_bos_{side}"] = np.zeros(len(base_index), dtype=np.float32)

    return arr


def _htf_bias(htf: dict, dt: pd.Timestamp) -> int:
    """
    Derive directional bias from 1D and 4H frames.

    Returns:
      +1  both 1D and 4H are bullish (ema_stack == 2 on the most recent bar <= dt)
      -1  both bearish (ema_stack == -2)
       0  mixed or insufficient data — trades require additional confluence
    """
    bias_1d = 0
    bias_4h = 0

    df_1d = htf.get("1D", pd.DataFrame())
    if not df_1d.empty and "ema_stack" in df_1d.columns:
        idx = df_1d.index.searchsorted(dt, side="right") - 1
        if idx >= 0:
            bias_1d = int(df_1d.iloc[idx]["ema_stack"])

    df_4h = htf.get("4H", pd.DataFrame())
    if not df_4h.empty and "ema_stack" in df_4h.columns:
        idx = df_4h.index.searchsorted(dt, side="right") - 1
        if idx >= 0:
            bias_4h = int(df_4h.iloc[idx]["ema_stack"])

    if bias_1d == 2 and bias_4h == 2:
        return 1
    if bias_1d == -2 and bias_4h == -2:
        return -1
    # One timeframe aligned is enough for T3 (sweep reversal against intraday momentum)
    if bias_1d == 2 or bias_4h == 2:
        return 1 if bias_1d + bias_4h > 0 else 0
    if bias_1d == -2 or bias_4h == -2:
        return -1 if bias_1d + bias_4h < 0 else 0
    return 0


def _get_4h_structure(htf: dict, dt: pd.Timestamp) -> dict:
    """
    Return the most recent 4H bar's structure signals as of dt.
    Used by T1 (SL anchor from 4H ATR) and T2 (4H BOS/FVG confirmation).
    """
    df_4h = htf.get("4H", pd.DataFrame())
    if df_4h.empty:
        return {}
    idx = df_4h.index.searchsorted(dt, side="right") - 1
    if idx < 0:
        return {}
    bar = df_4h.iloc[idx]
    return {
        "bos_bull":        bool(bar.get("bos_bull", False)),
        "bos_bear":        bool(bar.get("bos_bear", False)),
        "fvg_bull":        bool(bar.get("fvg_bull", False)),
        "fvg_bear":        bool(bar.get("fvg_bear", False)),
        "last_swing_high": float(bar.get("last_swing_high", np.nan)),
        "last_swing_low":  float(bar.get("last_swing_low", np.nan)),
        "ema_stack":       int(bar.get("ema_stack", 0)),
        "atr_14":          float(bar.get("atr_14", np.nan)),
        "rsi_14":          float(bar.get("rsi_14", 50)),
    }


def _get_1h_context(htf: dict, dt: pd.Timestamp) -> dict:
    """Return most-recent 1H bar context as of dt: adx, ema_stack, ema_21_slope."""
    df_1h = htf.get("1H", pd.DataFrame())
    if df_1h.empty:
        return {"adx_14": 50.0, "ema_stack": 0, "ema_21_slope": 0.0, "rsi_14": 50.0}
    idx = df_1h.index.searchsorted(dt, side="right") - 1
    if idx < 0:
        return {"adx_14": 50.0, "ema_stack": 0, "ema_21_slope": 0.0, "rsi_14": 50.0}
    bar = df_1h.iloc[idx]
    return {
        "adx_14":      float(bar.get("adx_14", 50.0)),
        "ema_stack":   int(bar.get("ema_stack", 0)),
        "ema_21_slope": float(bar.get("ema_21_slope", 0.0)),
        "rsi_14":      float(bar.get("rsi_14", 50.0)),
    }


def _pip_for_entry(entry: float) -> float:
    """Derive pip size from price magnitude: XAUUSD ~1800-3000, JPY ~100-160, others ~0.6-1.4."""
    if entry > 500:   # XAUUSD (gold trades 1800-3000)
        return 0.10
    if entry > 50:    # JPY pairs (100-160)
        return 0.01
    return 0.0001


def _simulate_trade_pm(
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
    """
    Vectorised walk-forward simulation honouring TP1/TP2 two-phase logic.

    Uses NumPy array slices instead of a Python bar-by-bar loop.
    Equivalent logic to the original implementation — same TP1 partial close,
    break-even, deferred trailing stop, and TP2 targets.

    Phase 1 (before TP1):
      - SL hit → full loss on full size
      - TP1 hit → close 50%, move SL to break-even (+/- 1 pip), enter Phase 2

    Phase 2 (after TP1):
      - Trailing SL activated once price clears TP1 + 0.5×ATR
      - Effective SL = max(BE, trail) for buys / min(BE, trail) for sells
      - TP2 hit → close remainder at TP2
      - Time exit → close remainder at market
    """
    pip       = _pip_for_entry(entry)
    half_size = round(size * 0.50, 2)
    rem_size  = round(size - half_size, 2)
    TRAIL_MULT            = 1.5
    TRAIL_ACTIVATE_BUFFER = 0.5

    end_idx   = min(entry_idx + MAX_HOLD_BARS, len(df))
    look_end  = end_idx  # exclusive

    # Extract price arrays for the trade window — one allocation, no per-bar overhead
    highs  = df["high"].values[entry_idx + 1 : look_end].astype(np.float64)
    lows   = df["low"].values[entry_idx + 1 : look_end].astype(np.float64)
    closes = df["close"].values[entry_idx + 1 : look_end].astype(np.float64)
    n_bars = len(highs)

    if n_bars == 0:
        return {
            "pnl": 0.0, "exit_reason": "time_exit",
            "tp1_hit": False, "tp1_bar": None,
            "bars_held": 0, "phase1_pnl": 0.0, "phase2_pnl": 0.0,
        }

    is_buy     = (side == "buy")
    phase1_pnl = 0.0
    phase2_pnl = 0.0
    exit_reason = "time_exit"
    tp1_hit    = False
    tp1_bar    = None
    bars_held  = n_bars  # default: full hold (time exit)

    # ── Phase 1: scan for first SL or TP1 hit ────────────────────────────────
    # Vectorised: find the first bar where either level is breached.
    if is_buy:
        sl_hit_mask  = lows  <= sl
        tp1_hit_mask = highs >= tp1
    else:
        sl_hit_mask  = highs >= sl
        tp1_hit_mask = lows  <= tp1

    sl_hits  = np.where(sl_hit_mask)[0]
    tp1_hits = np.where(tp1_hit_mask)[0]
    first_sl  = int(sl_hits[0])  if len(sl_hits)  > 0 else n_bars
    first_tp1 = int(tp1_hits[0]) if len(tp1_hits) > 0 else n_bars

    if first_sl <= first_tp1:
        # SL hit before (or at same bar as) TP1
        bars_held  = first_sl + 1
        phase1_pnl = (sl - entry if is_buy else entry - sl) * size
        exit_reason = "sl_full"
        gross_pnl  = phase1_pnl
        gross_pnl -= abs(gross_pnl) * COMMISSION_PCT
        return {
            "pnl": gross_pnl, "exit_reason": exit_reason,
            "tp1_hit": False, "tp1_bar": None,
            "bars_held": bars_held,
            "phase1_pnl": round(phase1_pnl, 6),
            "phase2_pnl": 0.0,
        }

    if first_tp1 < n_bars:
        # TP1 hit — record Phase 1 PnL, set BE, enter Phase 2
        tp1_hit    = True
        tp1_bar    = entry_idx + 1 + first_tp1
        phase1_pnl = (tp1 - entry if is_buy else entry - tp1) * half_size
        be_sl      = entry + pip if is_buy else entry - pip
        trail_sl   = be_sl

        # ── Phase 2: scan remaining bars for trail/TP2 ────────────────────────
        p2_start   = first_tp1 + 1      # phase 2 starts the bar AFTER TP1
        p2_highs   = highs[p2_start:]
        p2_lows    = lows[p2_start:]
        p2_closes  = closes[p2_start:]
        n_p2       = len(p2_highs)

        if n_p2 == 0:
            # Time exit immediately after TP1
            exit_price  = float(closes[first_tp1])
            phase2_pnl  = (exit_price - entry if is_buy else entry - exit_price) * rem_size
            exit_reason = "time_exit"
            bars_held   = first_tp1 + 1
        else:
            trail_trigger = tp1 + TRAIL_ACTIVATE_BUFFER * atr if is_buy else tp1 - TRAIL_ACTIVATE_BUFFER * atr

            # ── Vectorised Phase 2 trailing stop ──────────────────────────────
            # Step 1: find first bar where trail activates
            if is_buy:
                trail_on_mask = p2_closes >= trail_trigger
            else:
                trail_on_mask = p2_closes <= trail_trigger
            trail_start = int(np.argmax(trail_on_mask)) if trail_on_mask.any() else n_p2

            # Step 2: compute trailing SL level at each bar after activation
            # Before activation: effective_sl = be_sl
            # After activation: trail = cummax(close) - mult*atr (buy) or cummin + mult*atr (sell)
            eff_sl = np.full(n_p2, be_sl, dtype=np.float64)
            if trail_start < n_p2:
                if is_buy:
                    running_trail = np.maximum.accumulate(p2_closes[trail_start:]) - TRAIL_MULT * atr
                    # Trail SL starts at be_sl, only ratchets up
                    running_trail = np.maximum(running_trail, be_sl)
                    eff_sl[trail_start:] = running_trail
                else:
                    running_trail = np.minimum.accumulate(p2_closes[trail_start:]) + TRAIL_MULT * atr
                    running_trail = np.minimum(running_trail, be_sl)
                    eff_sl[trail_start:] = running_trail

            # Step 3: find first exit event (SL breach or TP2 hit)
            if is_buy:
                sl_hit  = p2_lows  <= eff_sl
                tp2_hit = p2_highs >= tp2
            else:
                sl_hit  = p2_highs >= eff_sl
                tp2_hit = p2_lows  <= tp2

            sl_bars  = np.where(sl_hit)[0]
            tp2_bars = np.where(tp2_hit)[0]
            first_sl  = int(sl_bars[0])  if sl_bars.size  else n_p2
            first_tp2 = int(tp2_bars[0]) if tp2_bars.size else n_p2

            if first_sl < first_tp2 and first_sl < n_p2:
                hit_sl      = float(eff_sl[first_sl])
                phase2_pnl  = (hit_sl - entry if is_buy else entry - hit_sl) * rem_size
                exit_reason = "be_or_trail" if (hit_sl > entry if is_buy else hit_sl < entry) else "sl_after_tp1"
                exit_p2_bar = first_sl
            elif first_tp2 < n_p2:
                phase2_pnl  = (tp2 - entry if is_buy else entry - tp2) * rem_size
                exit_reason = "tp2"
                exit_p2_bar = first_tp2
            else:
                exit_price  = float(p2_closes[-1])
                phase2_pnl  = (exit_price - entry if is_buy else entry - exit_price) * rem_size
                exit_reason = "time_exit"
                exit_p2_bar = n_p2 - 1

            bars_held = first_tp1 + 1 + exit_p2_bar + 1

    else:
        # Neither SL nor TP1 hit within MAX_HOLD_BARS — time exit
        exit_price  = float(closes[-1])
        phase1_pnl  = (exit_price - entry if is_buy else entry - exit_price) * size
        exit_reason = "time_exit"
        bars_held   = n_bars

    gross_pnl = phase1_pnl + phase2_pnl
    gross_pnl -= abs(gross_pnl) * COMMISSION_PCT

    return {
        "pnl": gross_pnl,
        "exit_reason": exit_reason,
        "tp1_hit": tp1_hit,
        "tp1_bar": tp1_bar,
        "bars_held": bars_held,
        "phase1_pnl": round(phase1_pnl, 6),
        "phase2_pnl": round(phase2_pnl, 6),
    }


def _run_bar_ml(ml_models: dict, df: pd.DataFrame, i: int, symbol: str = "",
                htf: dict = None) -> dict:
    """
    Single-bar ML inference — used only when pre-computed cache is unavailable.
    Returns preds dict with regime, quality_score, p_bull, p_bear.
    Returns {} if ml_models is empty (ML disabled).
    htf: full {tf: DataFrame} dict — all TFs sliced to current bar for no-lookahead MTF features.
    """
    if not ml_models:
        return {}
    preds: dict = {}
    cur_ts = df.index[i]
    window = df.iloc[max(0, i - 199): i + 1]

    # Build no-lookahead MTF dict: slice every HTF df up to current bar timestamp
    htf_slice: dict = {}
    if htf:
        for tf_key, tf_df in htf.items():
            if tf_df is not None and not tf_df.empty:
                sliced = tf_df[tf_df.index <= cur_ts]
                if len(sliced) >= 2:
                    htf_slice[tf_key] = sliced
    # Include 15M window itself so models can reference it as "15M"
    htf_slice["15M"] = window

    gru = ml_models.get("gru_lstm")
    if gru:
        r = gru.predict(window, symbol=symbol, df_htf=htf_slice)
        preds.update(r)

    regime_model = ml_models.get("regime")
    if regime_model:
        r = regime_model.predict(window, symbol=symbol, df_htf=htf_slice)
        preds["regime"] = r.get("regime")

    qs = ml_models.get("quality")
    if qs:
        bar = df.iloc[i]
        neutral_signal = {"trader_id": "", "side": "buy", "rr_ratio": 1.5}
        from services.feature_engine import FeatureEngine, QUALITY_FEATURES
        _fe = FeatureEngine()
        features = _fe.get_quality_features(neutral_signal, preds, bar)
        feat_dict = dict(zip(QUALITY_FEATURES, features))
        _qs_result = qs.predict(feat_dict)
        preds["quality_score"] = float(_qs_result["quality_score"])
        preds["ev"] = float(_qs_result["ev"])

    return preds


def _precompute_ml_cache(
    df: pd.DataFrame,
    symbol: str,
    htf: dict,
    ml_models: dict,
    batch_size: int = 512,
    cache_file: str | None = None,
) -> dict[str, np.ndarray]:
    """
    GPU-batched ML pre-computation for a full symbol DataFrame.

    Instead of launching one GPU forward pass per bar (O(N) kernel launches,
    each with ~1ms CUDA overhead), this function:
      1. Builds all sequence feature arrays for every bar in one NumPy pass.
      2. Runs the GRU model in batches of `batch_size` on GPU → one kernel
         launch per batch instead of one per bar.
      3. Runs the Regime classifier in a single batch sklearn/LightGBM call.
      4. Returns a dict {bar_index → preds_dict} for O(1) per-bar lookup.

    Speedup on T4: ~40–80× vs per-bar inference for a 50k-bar symbol.
    Falls back to empty dict (triggering per-bar inference) on any error.
    """
    import time as _time
    _t0_cache = _time.perf_counter()

    if not ml_models:
        return {}

    if cache_file:
        cache = _load_ml_cache(cache_file, len(df))
        if cache is not None:
            logger.info("ML cache restored for %s from %s (%.1fs)",
                        symbol, os.path.basename(cache_file), _time.perf_counter() - _t0_cache)
            return cache

    gru_model    = ml_models.get("gru_lstm")
    regime_4h    = ml_models.get("regime_htf") or ml_models.get("regime_4h") or ml_models.get("regime")
    regime_1h    = ml_models.get("regime_ltf") or ml_models.get("regime_1h")
    qs_model     = ml_models.get("quality")

    if not (gru_model or regime_4h or regime_1h):
        return {}


    try:
        import torch
        from services.feature_engine import FeatureEngine, SEQUENCE_FEATURES, REGIME_FEATURES, QUALITY_FEATURES
        fe = FeatureEngine()

        n = len(df)
        SEQUENCE_LENGTH = 30

        # ── Regime inference ──────────────────────────────────────────────────────
        # Build feature matrix once per source df (serially), slice per model.
        # predict_batch (GPU/DataParallel) runs serially — one model at a time.
        from models.regime_classifier import RegimeClassifier as _RC

        _regime_htf_series = None
        _regime_htf_conf   = None
        _regime_ltf_series = None
        _regime_ltf_conf   = None

        _do_4h = bool(regime_4h and regime_4h.is_trained and regime_4h._model is not None)
        _do_1h = bool(regime_1h and regime_1h.is_trained and regime_1h._model is not None)

        _df_src_4h = (htf.get("4H") or htf.get("H4")) if _do_4h else None
        if _df_src_4h is None or len(_df_src_4h) < 50:
            _df_src_4h = df if _do_4h else None
        _df_src_1h = (htf.get("1H") or htf.get("H1")) if _do_1h else None
        if _df_src_1h is None or len(_df_src_1h) < 50:
            _df_src_1h = df if _do_1h else None

        _X_4h = _X_1h = None
        try:
            if _do_4h:
                _t_fm = _time.perf_counter()
                _X_4h = _RC._build_feature_matrix(_df_src_4h, htf, symbol)
                logger.info("ML cache [%s]: 4H feature matrix %.1fs (%d bars)",
                            symbol, _time.perf_counter() - _t_fm, len(_df_src_4h))
        except Exception as _exc:
            logger.error("ML cache: regime 4H feature build failed %s: %s", symbol, _exc)
        try:
            if _do_1h:
                # Reuse matrix when both models fall back to the same source df
                if _do_4h and _df_src_1h is _df_src_4h and _X_4h is not None:
                    _X_1h = _X_4h
                else:
                    _t_fm = _time.perf_counter()
                    _X_1h = _RC._build_feature_matrix(_df_src_1h, htf, symbol)
                    logger.info("ML cache [%s]: 1H feature matrix %.1fs (%d bars)",
                                symbol, _time.perf_counter() - _t_fm, len(_df_src_1h))
        except Exception as _exc:
            logger.error("ML cache: regime 1H feature build failed %s: %s", symbol, _exc)

        def _infer_regime(rc_model, X_feat, df_src):
            _mode = getattr(rc_model, "_mode", "ltf_behaviour")
            if _mode == "htf_bias":
                _classes, _default_id = ["BIAS_UP", "BIAS_DOWN", "BIAS_NEUTRAL"], 2
            else:
                _classes, _default_id = ["TRENDING", "RANGING", "CONSOLIDATING", "VOLATILE"], 1
            _n    = len(df_src)
            _step = max(1, _n // 20_000)
            try:
                row_idx = list(range(50, _n, _step))
                X = X_feat[row_idx]
                if len(X) == 0:
                    return None, None, {}
                ids, conf = rc_model.predict_batch(X)
                del X
                arr = np.full(_n, -1, dtype=np.int8)
                arr[row_idx] = ids.astype(np.int8)
                filled = (pd.Series(arr.astype(float))
                          .replace(-1, np.nan).ffill().fillna(_default_id).astype(int).values)
                c_arr = np.full(_n, np.nan, dtype=np.float32)
                c_arr[row_idx] = conf
                filled_conf = pd.Series(c_arr, index=df_src.index).ffill()
                return pd.Series(filled, index=df_src.index, dtype=int), filled_conf
            except Exception as _e:
                logger.error("ML cache: regime infer failed %s: %s", symbol, _e)
                return None, None

        if _do_4h and _X_4h is not None:
            _t_ri = _time.perf_counter()
            _r4h, _c4h = _infer_regime(regime_4h, _X_4h, _df_src_4h)
            logger.info("ML cache [%s]: HTF regime infer %.1fs", symbol, _time.perf_counter() - _t_ri)
            del _X_4h
            if _r4h is not None:
                if _df_src_4h is not df:
                    _regime_htf_series = _r4h.reindex(df.index, method="ffill").fillna(2).astype(int)
                    _regime_htf_conf   = _c4h.reindex(df.index, method="ffill").fillna(1/3)
                    logger.info("ML cache: HTF regime done %s (%d 4H bars → %d 15M bars)",
                                symbol, len(_df_src_4h), n)
                else:
                    _regime_htf_series, _regime_htf_conf = _r4h, _c4h
                    logger.info("ML cache: HTF regime on 15M fallback for %s", symbol)
            gc.collect()

        if _do_1h and _X_1h is not None:
            _t_ri = _time.perf_counter()
            _r1h, _c1h = _infer_regime(regime_1h, _X_1h, _df_src_1h)
            logger.info("ML cache [%s]: LTF regime infer %.1fs", symbol, _time.perf_counter() - _t_ri)
            del _X_1h
            if _r1h is not None:
                if _df_src_1h is not df:
                    _regime_ltf_series = _r1h.reindex(df.index, method="ffill").fillna(1).astype(int)
                    _regime_ltf_conf   = _c1h.reindex(df.index, method="ffill").fillna(0.25)
                    logger.info("ML cache: LTF regime done %s (%d 1H bars → %d 15M bars)",
                                symbol, len(_df_src_1h), n)
                else:
                    _regime_ltf_series, _regime_ltf_conf = _r1h, _c1h
            gc.collect()

        # Regime distribution diagnostic
        if _regime_htf_series is not None:
            from collections import Counter
            logger.info("ML cache: 4H regime distribution %s: %s",
                        symbol, dict(Counter(_HTF_REGIME_NAMES[np.clip(_regime_htf_series.values, 0, 2)])))

        # ── Build sequence features (with dual-regime context) ────────────────
        _t_sf = _time.perf_counter()
        logger.info("ML cache: building sequence features for %s (%d bars)...", symbol, n)
        try:
            feat_df = fe._build_sequence_df(
                df, htf, symbol=symbol,
                regime_4h_series=_regime_htf_series,
                regime_4h_conf_series=_regime_htf_conf,
                regime_1h_series=_regime_ltf_series,
                regime_1h_conf_series=_regime_ltf_conf,
            )
            seq_arr = feat_df[SEQUENCE_FEATURES].to_numpy(dtype=np.float32, copy=False)
            seq_arr = np.nan_to_num(seq_arr, nan=0.0, posinf=0.0, neginf=0.0)
            del feat_df
            logger.info("ML cache [%s]: sequence features %.1fs", symbol, _time.perf_counter() - _t_sf)
        except Exception as exc:
            logger.error("ML cache: sequence feature build failed for %s: %s", symbol, exc)
            raise

        # ── Batched GRU inference ──────────────────────────────────────────────
        _t_gru = _time.perf_counter()
        cache = {
            "p_bull": np.full(n, np.nan, dtype=np.float32),
            "p_bear": np.full(n, np.nan, dtype=np.float32),
            "expected_variance": np.full(n, np.nan, dtype=np.float32),
            "regime": np.full(n, -1, dtype=np.int8),
            "regime_ltf": np.full(n, -1, dtype=np.int8),
        }
        if gru_model and seq_arr is not None and gru_model.is_trained and gru_model._model is not None:
            try:
                from models.gru_lstm_predictor import DEVICE, SEQUENCE_LENGTH as SEQ_LEN
                import torch

                n_valid = n - SEQ_LEN + 1   # number of complete sequences
                n_feat  = seq_arr.shape[1]

                if n_valid > 0:
                    m = gru_model._model.module if hasattr(gru_model._model, "module") else gru_model._model
                    m.eval()
                    all_p_bull  = np.empty(n_valid, dtype=np.float32)
                    all_mag     = np.empty(n_valid, dtype=np.float32)
                    all_log_var = np.empty(n_valid, dtype=np.float32)

                    # Stream batches directly from seq_arr — no full (N,SEQ,F) tensor.
                    # sliding_window_view of the whole array would be ~1 GB per symbol;
                    # slicing batch-sized windows keeps peak RAM to batch_size×SEQ_LEN×F.
                    _T = getattr(gru_model, "_temperature", 1.0)
                    with torch.no_grad():
                        for b_start in range(0, n_valid, batch_size):
                            b_end = min(b_start + batch_size, n_valid)
                            # seq_arr[b_start : b_end+SEQ_LEN-1] is a contiguous view;
                            # stride_tricks over just these rows — O(batch) not O(N).
                            batch_raw = np.lib.stride_tricks.sliding_window_view(
                                seq_arr[b_start: b_end + SEQ_LEN - 1], (SEQ_LEN, n_feat)
                            ).reshape(b_end - b_start, SEQ_LEN, n_feat)
                            xb = torch.from_numpy(batch_raw.copy()).to(DEVICE)
                            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                                dl, mp, lv = m(xb)
                            all_p_bull[b_start:b_end]  = torch.sigmoid(dl / _T).cpu().numpy()
                            all_mag[b_start:b_end]     = torch.relu(mp).cpu().numpy()
                            all_log_var[b_start:b_end] = lv.cpu().numpy()
                            del xb, dl, mp, lv, batch_raw

                    # Vectorised post-processing
                    var_vals    = np.log1p(np.exp(all_log_var)) + 1e-6   # softplus
                    p_bear_arr  = np.clip(1.0 - all_p_bull, 0.0, 1.0)
                    entry_depth = np.clip(all_mag * 100.0, 0.0, 1.0)
                    vol_arr     = np.sqrt(var_vals)

                    # bar_idx for sequence i = i + SEQ_LEN - 1
                    bar_indices = np.arange(n_valid, dtype=np.int32) + (SEQ_LEN - 1)

                    cache["p_bull"][bar_indices] = all_p_bull
                    cache["p_bear"][bar_indices] = p_bear_arr
                    cache["expected_variance"][bar_indices] = var_vals.astype(np.float32, copy=False)
                    del all_p_bull, all_mag, all_log_var, var_vals, p_bear_arr, entry_depth, vol_arr
                    logger.info("ML cache [%s]: GRU batch inference %.1fs (%d bars)",
                                symbol, _time.perf_counter() - _t_gru, n_valid)
            except Exception as exc:
                logger.error("ML cache: GRU batch failed for %s: %s", symbol, exc)
                del seq_arr
                raise
            finally:
                if 'seq_arr' in dir() and seq_arr is not None:
                    del seq_arr  # ~(N×F) float32 — free before regime allocates X_all

        # ── Build LTF per-bar lookup (behaviour class name) ──────────────────
        if _regime_ltf_series is not None:
            cache["regime_ltf"] = np.clip(_regime_ltf_series.values, 0, 3).astype(np.int8, copy=False)
        if _regime_htf_series is not None:
            cache["regime"] = np.clip(_regime_htf_series.values, 0, 2).astype(np.int8, copy=False)

        n_gru = int(np.count_nonzero(~np.isnan(cache["p_bull"])))
        n_htf = int(np.count_nonzero(cache["regime"] >= 0))
        n_ltf = int(np.count_nonzero(cache["regime_ltf"] >= 0))
        logger.info("ML cache [%s]: DONE %.1fs — %d bars (gru=%d htf=%d ltf=%d)",
                    symbol, _time.perf_counter() - _t0_cache, n, n_gru, n_htf, n_ltf)

        # Diagnostic: sample GRU output distribution so 0-trade causes are visible
        _valid = ~np.isnan(cache["p_bull"])
        if _valid.any():
            _pb = cache["p_bull"][_valid][:5000]
            _ev = cache["expected_variance"][_valid][:5000]
            logger.info(
                "ML cache GRU sample %s: p_bull mean=%.3f std=%.3f "
                "pct_above_55=%.1f%% | variance mean=%.3f pct_above_80=%.1f%%",
                symbol, _pb.mean(), _pb.std(),
                100 * (_pb >= 0.55).mean(),
                _ev.mean(), 100 * (_ev >= 0.80).mean(),
            )

        if cache_file:
            try:
                _save_ml_cache(cache_file, cache)
            except Exception as exc:
                logger.warning("ML cache save failed %s: %s", cache_file, exc)
        gc.collect()
        return cache

    except Exception as exc:
        logger.error("_precompute_ml_cache failed for %s: %s", symbol, exc)
        gc.collect()
        raise


def _compute_backtest_signal(symbol: str, ml_preds: dict, bar: pd.Series) -> dict | None:
    """
    ML-native signal generator gated on HTF bias + LTF behaviour + GRU direction.

    Gate order:
      1. ATR sanity
      2. GRU variance ≤ MAX_UNCERTAINTY
      3. GRU direction ≥ ML_DIRECTION_THRESHOLD  →  determines side (buy/sell)
      4. HTF bias must AGREE with GRU side
      5. LTF behaviour must PERMIT entry:
           TRENDING   → by default, GRU+HTF gates only; if a swing pullback is
                        detected, require it to match GRU side (set
                        REQUIRE_TRENDING_PULLBACK=1 to require a pullback for every
                        TRENDING bar — that mode is very sparse)
           VOLATILE   → high GRU confidence only
           RANGING    → significant range + correct boundary
           CONSOLIDATING → blocked

    EV gate runs after PM enrichment in _backtest_trader (needs rr_ratio).
    """
    close = float(bar["close"])
    atr   = float(bar.get("atr_14", close * 0.001))
    if atr < 1e-9:
        return None

    if not ml_preds:
        return None

    # ── Gate 2: GRU uncertainty ───────────────────────────────────────────────
    _uncertainty = float(ml_preds.get("expected_variance", 0.0))
    if _uncertainty > float(os.getenv("MAX_UNCERTAINTY", "2.0")):
        return None

    # ── Gate 3: GRU direction ─────────────────────────────────────────────────
    p_bull = float(ml_preds.get("p_bull", 0.5))
    p_bear = float(ml_preds.get("p_bear", 0.5))
    # Default 0.58 matches config.settings.ML_DIRECTION_THRESHOLD and published docs
    _dir_thresh = float(os.getenv("ML_DIRECTION_THRESHOLD", "0.58"))
    if p_bull >= p_bear and p_bull >= _dir_thresh:
        side = "buy"
        conf = p_bull
    elif p_bear > p_bull and p_bear >= _dir_thresh:
        side = "sell"
        conf = p_bear
    else:
        return None

    # ── Gate 4: HTF bias alignment ────────────────────────────────────────────
    # BIAS_UP   → only buys allowed (HTF trend is up)
    # BIAS_DOWN → only sells allowed (HTF trend is down)
    # BIAS_NEUTRAL → require at least the same bar as the direction threshold
    # (0.65 was stricter than the direction gate and filtered almost all NEUTRAL bars)
    _htf_bias = str(ml_preds.get("regime", "BIAS_NEUTRAL"))
    _neutral_thresh = float(os.getenv("NEUTRAL_BIAS_THRESHOLD", "0.58"))

    if _htf_bias == "BIAS_UP" and side == "sell":
        return None   # HTF is bullish — no sells
    if _htf_bias == "BIAS_DOWN" and side == "buy":
        return None   # HTF is bearish — no buys
    if _htf_bias == "BIAS_NEUTRAL" and conf < _neutral_thresh:
        return None   # no structural bias — require stronger GRU conviction

    # ── Gate 5: LTF behaviour filter ─────────────────────────────────────────
    _ltf_behaviour = str(ml_preds.get("regime_ltf", "TRENDING"))
    # Default tracks direction threshold — 0.72 blocked almost all VOLATILE bars
    _volatile_thresh = float(os.getenv("VOLATILE_ENTRY_THRESHOLD", str(_dir_thresh)))

    _range_valid    = bool(bar.get("range_valid", False))
    _range_side     = str(bar.get("range_side", ""))
    _pullback_valid = bool(bar.get("pullback_valid", False))
    _pullback_side  = str(bar.get("pullback_side", ""))
    _strict_trend_pb = str(os.getenv("REQUIRE_TRENDING_PULLBACK", "0")).lower() in (
        "1", "true", "yes",
    )

    if _ltf_behaviour == "CONSOLIDATING":
        if str(os.getenv("BLOCK_LTF_CONSOLIDATING", "0")).lower() in ("1", "true", "yes"):
            return None

    if _ltf_behaviour == "VOLATILE" and conf < _volatile_thresh:
        return None

    if _ltf_behaviour == "TRENDING":
        if _strict_trend_pb:
            if not _pullback_valid:
                return None
            if str(_pullback_side or "") != side:
                return None
        else:
            # When a retest is detected, it must not contradict the GRU side; when
            # none is detected, rely on direction + HTF gates (REQUIRE_TRENDING_PULLBACK=1
            # restores the old "pullback on every TRENDING bar" rule, which is ~0 signals).
            if _pullback_valid and str(_pullback_side or "") and str(_pullback_side) != side:
                return None

    if _ltf_behaviour == "RANGING":
        _strict_rng = str(os.getenv("RANGING_REQUIRE_RANGE", "0")).lower() in ("1", "true", "yes")
        if _strict_rng:
            if not _range_valid:
                return None
            if str(_range_side or "") != side:
                return None
        else:
            if _range_valid and str(_range_side or "") and str(_range_side) != side:
                return None

    # ── ATR-based entry / SL / TP ─────────────────────────────────────────────
    # For RANGING entries: TP is the far wall of the range rather than a fixed
    # ATR multiple, provided the far wall gives better R:R than the default.
    _sl_mult    = float(os.getenv("SL_ATR_MULT", "1.5"))
    _rr_default = float(os.getenv("RR_DEFAULT", "2.0"))
    sl_dist = atr * _sl_mult

    if _ltf_behaviour == "RANGING" and _range_valid:
        # SL: just beyond the near boundary (boundary acts as stop anchor)
        # TP: the far wall of the range
        if side == "buy":
            stop_loss   = float(bar.get("range_support", close - sl_dist)) - atr * 0.3
            far_wall    = float(bar.get("range_resist",  close + sl_dist * _rr_default))
            take_profit = far_wall
        else:
            stop_loss   = float(bar.get("range_resist",  close + sl_dist)) + atr * 0.3
            far_wall    = float(bar.get("range_support", close - sl_dist * _rr_default))
            take_profit = far_wall
        # Fall back to ATR-based TP if the range wall gives worse R:R
        default_tp = (close + sl_dist * _rr_default) if side == "buy" else (close - sl_dist * _rr_default)
        actual_rr  = abs(take_profit - close) / (abs(close - stop_loss) + 1e-9)
        if actual_rr < 1.5:
            stop_loss   = (close - sl_dist) if side == "buy" else (close + sl_dist)
            take_profit = default_tp
    else:
        if side == "buy":
            stop_loss   = close - sl_dist
            take_profit = close + sl_dist * _rr_default
        else:
            stop_loss   = close + sl_dist
            take_profit = close - sl_dist * _rr_default

    _range_width = float(bar.get("range_width_atr", 0.0)) if _ltf_behaviour == "RANGING" else 0.0

    return {
        "side":        side,
        "entry":       close,
        "stop_loss":   stop_loss,
        "take_profit": take_profit,
        "confidence":  round(float(conf), 3),
        "trader_id":   "ml_trader",
        "symbol":      symbol,
        "signal_metadata": {
            "regime":            _htf_bias,
            "regime_ltf":        _ltf_behaviour,
            "expected_variance": _uncertainty,
            "p_bull":            p_bull,
            "p_bear":            p_bear,
            "atr_at_entry":      atr,
            "range_width_atr":   _range_width,
            "pullback_valid":    _pullback_valid,
            "pullback_level":    float(bar.get("pullback_level", float("nan"))),
        },
    }

def _backtest_trader(
    trader_id: str,
    symbols: list,
    pm,
    start: str,
    end: str,
    ml_models: dict = None,
    shared_ml_cache: dict = None,
) -> dict:
    """
    Run backtest for one trader across all its symbols.

    Architecture:
    - All symbols are loaded and iterated bar-by-bar in date order via a
      merged sorted index. This ensures:
        a) Cross-symbol cooldown is enforced correctly (no jumping from last bar
           of EURUSD to bar 200 of GBPUSD without cooldown).
        b) Daily circuit breakers fire on calendar-day boundaries regardless
           of which symbol triggered them.
        c) PM.notify_date() is called every bar so the PM's daily budget
           resets at the correct historical date, not wall-clock today.

    Circuit breakers (ordered, checked every bar):
      1. 20% portfolio drawdown halt (hard stop for trader)
      2. 2% daily loss cap (skip rest of day, resets next calendar day)
      3. Session filter (trader only trades in its defined hours)
      4. NY/London dead zone 12:00–13:00 UTC
      5. Cooldown — 10 bars since last trade on the same symbol
      6. PM R:R gate — confidence must yield tp1_mult > 1.0
      7. PM correlation cap — max 2 positions per directional group
      8. PM streak gate — size scaled down after 3-4 consecutive losses
    """
    # ── Load + filter all symbol dataframes ──────────────────────────────────
    symbol_dfs: dict[str, pd.DataFrame] = {}
    symbol_htf: dict[str, dict] = {}   # HTF context per symbol (1D + 4H)
    symbol_start_idx: dict[str, int] = {}
    start_ts = _utc_ts(start)

    for symbol in symbols:
        df = _load_csv(symbol, "15M", start=start, end=end)
        if len(df) < 200:
            logger.warning("Skipping %s/%s — only %d bars", trader_id, symbol, len(df))
            continue
        df = _compute_indicators(df)
        first_live_idx = int(df.index.searchsorted(start_ts, side="left")) if start_ts is not None else 0
        if len(df) - first_live_idx < 200:
            logger.warning(
                "Skipping %s/%s — only %d bars inside requested window",
                trader_id, symbol, max(0, len(df) - first_live_idx),
            )
            continue
        symbol_dfs[symbol] = df
        symbol_htf[symbol] = _load_htf_context(symbol, start, end)
        symbol_start_idx[symbol] = max(200, first_live_idx)
        logger.info("Loaded %s %s: %d bars (%s → %s), tradable from idx=%d",
                    trader_id, symbol, len(df),
                    df.index[0].date(), df.index[-1].date(),
                    symbol_start_idx[symbol])

    if not symbol_dfs:
        return {"trades": 0, "win_rate": 0.0, "total_return": 0.0,
                "profit_factor": 0.0, "max_drawdown": 0.0, "sharpe": 0.0,
                "tp1_rate": 0.0, "tp2_rate": 0.0, "trade_log": []}


    # ── GPU-batched ML pre-computation ────────────────────────────────────────
    # Use shared cross-trader cache when available — avoids recomputing GRU/Regime
    # inference for every (trader × symbol) pair (55 calls → 11 calls).
    # Symbols are processed ONE AT A TIME to stay within the 25 GB RAM budget:
    # each symbol holds ~2–3 GB of feature/GRU tensors during build; running 4
    # in parallel would peak at 8–12 GB on top of the already-loaded model weights
    # and training dataset — causing OOM (rc=-9). The inner 2-thread regime
    # feature-matrix parallelism inside _precompute_ml_cache is preserved.
    symbol_ml_cache: dict[str, dict[str, np.ndarray]] = {}
    # 4 CPUs available; BLAS/numpy threads already capped at 1–4 per worker via env vars.
    # 2 was conservative to avoid OOM, but with OMP/MKL capped each worker stays ~2–3 GB.
    _cache_workers = max(1, min(4, int(os.getenv("MAX_PARALLEL_CACHE_BUILDS", "4"))))

    def _build_cache_sym(sym):
        cache_file = _ml_cache_file(sym, start, end, ml_models or {})
        return sym, _precompute_ml_cache(
            symbol_dfs[sym], sym, symbol_htf.get(sym, {}), ml_models, cache_file=cache_file
        )

    if shared_ml_cache is not None:
        missing = [sym for sym in symbol_dfs if sym not in shared_ml_cache and ml_models]
        hits    = [sym for sym in symbol_dfs if sym in shared_ml_cache]
        for sym in hits:
            symbol_ml_cache[sym] = shared_ml_cache[sym]
        if missing:
            logger.info("%s: ML cache miss for %d symbols — building with %d worker(s)", trader_id, len(missing), _cache_workers)
            with ThreadPoolExecutor(max_workers=_cache_workers) as ex:
                futures = {ex.submit(_build_cache_sym, sym): sym for sym in missing}
                for fut in as_completed(futures):
                    sym = futures[fut]
                    try:
                        _, c = fut.result()
                        shared_ml_cache[sym] = c
                        symbol_ml_cache[sym] = c
                    except Exception as exc:
                        logger.warning("ML cache build failed for %s: %s", sym, exc)
    elif ml_models:
        logger.info("%s: pre-computing ML inference cache for %d symbols with %d worker(s)...",
                    trader_id, len(symbol_dfs), _cache_workers)
        with ThreadPoolExecutor(max_workers=_cache_workers) as ex:
            futures = {ex.submit(_build_cache_sym, sym): sym for sym in symbol_dfs}
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    _, c = fut.result()
                    symbol_ml_cache[sym] = c
                except Exception as exc:
                    logger.warning("ML cache build failed for %s: %s", sym, exc)

    # ── Pre-extract bar arrays as numpy (eliminates df.iloc[i] per bar) ─────────
    # ~10-20 µs per df.iloc[] call × 473k bars = 5-10 min saved in pure Python.
    import heapq
    _bar_cols = ["open", "high", "low", "close", "atr_14", "adx_14", "rsi_14",
                 "ema_stack", "bb_width", "bos_bull", "bos_bear", "fvg_bull", "fvg_bear",
                 "range_valid", "range_side", "range_support", "range_resist",
                 "range_width_atr", "pullback_valid", "pullback_side", "pullback_level",
                 "last_swing_high", "last_swing_low", "regime_duration", "vol_slope_seq",
                 "volume_ratio", "ema_21"]
    _sym_arrays: dict[str, dict] = {}
    _sym_timestamps: dict[str, np.ndarray] = {}
    for _sym, _df in symbol_dfs.items():
        _arrs: dict[str, np.ndarray] = {}
        for _col in _bar_cols:
            if _col in _df.columns:
                _arrs[_col] = _df[_col].to_numpy()
            else:
                _arrs[_col] = None
        _sym_arrays[_sym] = _arrs
        _sym_timestamps[_sym] = _df.index.view(np.int64)  # nanoseconds

    # ── Build merged bar iterator via heapq.merge — no flat list, no sort ─────
    # Each per-symbol generator yields (timestamp_ns, symbol, bar_index) in order.
    # heapq.merge merges N already-sorted generators in O(N log K) time, K=11.
    def _sym_iter(sym: str, df: pd.DataFrame):
        ts_arr = _sym_timestamps[sym]
        for idx in range(symbol_start_idx[sym], len(df)):
            yield ts_arr[idx], sym, idx

    bar_iter = heapq.merge(*[_sym_iter(s, d) for s, d in symbol_dfs.items()])

    # ── State ─────────────────────────────────────────────────────────────────
    trades: list[dict] = []
    equity      = INITIAL_CAPITAL
    peak_equity = INITIAL_CAPITAL

    # Fixed daily budget base (does not float with equity)
    daily_budget    = INITIAL_CAPITAL * MAX_DAILY_LOSS_PCT
    daily_loss      = 0.0
    current_date    = None
    daily_halt      = False   # True when daily budget is exhausted for today

    # Per-symbol cooldown: symbol → last bar index where a trade was taken
    last_trade_bar: dict[str, int] = {}

    # Trade density tracking: rolling 96-bar window (~24h at 15M) per symbol.
    # Used for soft exponential confidence penalty — not a hard cap.
    # DENSITY_LAMBDA env controls suppression strength (default 0.12).
    _DENSITY_WINDOW = 96
    recent_trade_bars: dict[str, deque[int]] = {s: deque() for s in symbol_dfs}

    # Pre-instantiate QualityScorer helper (avoid per-signal construction cost)
    _qs_fe    = None
    _qs_model = (ml_models or {}).get("quality") if ml_models else None
    _qs_infer = None  # fast CPU inference fn: (feat_array) -> (ev, quality_score)
    if _qs_model is not None:
        from services.feature_engine import FeatureEngine, QUALITY_FEATURES
        _qs_fe = FeatureEngine()
        try:
            # Single-sample GPU forward pass costs ~500 µs of kernel launch overhead.
            # CPU copy of the 20→128→64→32→1 MLP runs in ~3–5 µs — 100× faster.
            _qs_infer = _qs_model.get_cpu_inference_fn()
            logger.info("QualityScorer: using CPU inference path for backtest loop (fast single-sample)")
        except Exception as _e:
            logger.warning("QualityScorer CPU path unavailable (%s) — falling back to GPU predict()", _e)
            _qs_infer = None

    # Hoist env-var reads outside the 473k-bar loop — os.getenv does a dict lookup
    # + string compare on every call; at 473k bars this adds measurable overhead.
    _cfg_density_lambda   = float(os.getenv("DENSITY_LAMBDA",             "0.12"))
    _cfg_dir_thresh       = float(os.getenv("ML_DIRECTION_THRESHOLD",     "0.58"))
    _cfg_min_conf         = float(os.getenv("MIN_CONFIDENCE",             "0.0"))
    _cfg_min_ev           = float(os.getenv("MIN_EV_THRESHOLD",           "0.0"))
    _cfg_max_uncertainty  = float(os.getenv("MAX_UNCERTAINTY",            "2.0"))
    _cfg_neutral_thresh   = float(os.getenv("NEUTRAL_BIAS_THRESHOLD",     "0.58"))
    _cfg_volatile_thresh  = float(os.getenv("VOLATILE_ENTRY_THRESHOLD",   _cfg_dir_thresh.__str__()))
    _cfg_block_consol     = os.getenv("BLOCK_LTF_CONSOLIDATING",          "0").lower() in ("1","true","yes")
    _cfg_strict_trend_pb  = os.getenv("REQUIRE_TRENDING_PULLBACK",        "0").lower() in ("1","true","yes")
    _cfg_strict_rng       = os.getenv("RANGING_REQUIRE_RANGE",            "0").lower() in ("1","true","yes")
    _cfg_sl_mult          = float(os.getenv("SL_ATR_MULT",               "1.5"))
    _cfg_rr_default       = float(os.getenv("RR_DEFAULT",                "2.0"))

    # ── Main loop ─────────────────────────────────────────────────────────────
    import time as _time
    halted = False  # portfolio drawdown halt
    _dbg = {"total": 0, "dd_halt": 0, "daily": 0, "session": 0,
            "cooldown": 0, "density": 0, "no_signal": 0, "pm_reject": 0, "quality_block": 0}
    _loop_t0 = _time.perf_counter()
    _loop_last_log = _loop_t0
    _LOOP_LOG_INTERVAL = 50_000  # log progress every 50k bars

    for ts_ns, symbol, i in bar_iter:
        if halted:
            break

        _dbg["total"] += 1
        if _dbg["total"] % _LOOP_LOG_INTERVAL == 0:
            _now = _time.perf_counter()
            _rate = _LOOP_LOG_INTERVAL / max(0.01, _now - _loop_last_log)
            logger.info(
                "%s loop progress: %d bars | %.0f bars/s | trades=%d | "
                "session_skip=%d quality_block=%d no_signal=%d",
                trader_id, _dbg["total"], _rate,
                len(trades),
                _dbg["session"], _dbg["quality_block"], _dbg["no_signal"],
            )
            _loop_last_log = _now
        df  = symbol_dfs[symbol]
        dt  = pd.Timestamp(ts_ns, tz="UTC")

        # ── Circuit breaker 1: portfolio drawdown ─────────────────────────────
        dd = (peak_equity - equity) / (peak_equity + 1e-9)
        if dd >= MAX_DRAWDOWN_PCT:
            logger.warning("%s: portfolio drawdown %.1f%% — halting all trading",
                           trader_id, dd * 100)
            halted = True
            break

        # ── Date boundary: reset daily state, sync PM ─────────────────────────
        day_str = dt.strftime("%Y-%m-%d")
        if day_str != current_date:
            current_date = day_str
            daily_loss   = 0.0
            daily_halt   = False
            pm.notify_date(day_str)   # PM resets its internal daily P&L

        # ── Circuit breaker 2: daily loss cap ─────────────────────────────────
        if daily_halt:
            _dbg["daily"] += 1
            continue
        if abs(daily_loss) >= daily_budget:
            daily_halt = True
            _dbg["daily"] += 1
            logger.info("%s: daily loss cap hit on %s — skipping rest of day",
                        trader_id, day_str)
            continue

        # ── Circuit breaker 3: active hours (London open → NY close) ─────────
        if not (7 <= dt.hour < 18):
            _dbg["session"] += 1
            continue

        # ── Circuit breaker 5: per-symbol cooldown ────────────────────────────
        if i - last_trade_bar.get(symbol, -COOLDOWN_BARS) < COOLDOWN_BARS:
            _dbg["cooldown"] += 1
            continue

        # ── Trade density: prune window, compute current count ───────────────
        _recent = recent_trade_bars[symbol]
        while _recent and i - _recent[0] >= _DENSITY_WINDOW:
            _recent.popleft()
        _density_count = len(_recent)

        # ── ML inference — use pre-computed GPU cache, fall back to per-bar ────
        if symbol_ml_cache:
            ml_preds = _ml_cache_entry(symbol_ml_cache.get(symbol), i)
            if not ml_preds and ml_models:
                # Cache miss (bar too early for sequence window) — fall back
                ml_preds = _run_bar_ml(ml_models, df, i, symbol=symbol,
                                       htf=symbol_htf.get(symbol))
        else:
            ml_preds = _run_bar_ml(ml_models or {}, df, i, symbol=symbol,
                                   htf=symbol_htf.get(symbol)) if ml_models else {}

        # ── Signal generation ─────────────────────────────────────────────────
        # Build bar dict lazily — only for bars that pass session + cooldown gates.
        # Avoids 473k df.iloc[] allocations; ~95% of bars exit before this point.
        _sarr = _sym_arrays[symbol]
        bar = {
            "close":          float(_sarr["close"][i]) if _sarr["close"] is not None else 0.0,
            "atr_14":         float(_sarr["atr_14"][i]) if _sarr["atr_14"] is not None else 0.001,
            "range_valid":    bool(_sarr["range_valid"][i] == 1) if _sarr["range_valid"] is not None else False,
            "range_side":     (str(_sarr["range_side"][i]) if (
                                  _sarr["range_side"] is not None and
                                  _sarr["range_side"][i] == _sarr["range_side"][i])
                              else ""),
            "range_support":  float(_sarr["range_support"][i]) if _sarr["range_support"] is not None else 0.0,
            "range_resist":   float(_sarr["range_resist"][i]) if _sarr["range_resist"] is not None else 0.0,
            "range_width_atr": float(_sarr["range_width_atr"][i]) if _sarr["range_width_atr"] is not None else 0.0,
            "pullback_valid": bool(_sarr["pullback_valid"][i] == 1) if _sarr["pullback_valid"] is not None else False,
            "pullback_side":  (str(_sarr["pullback_side"][i]) if (
                                  _sarr["pullback_side"] is not None and
                                  _sarr["pullback_side"][i] == _sarr["pullback_side"][i])
                              else ""),
            "pullback_level": float(_sarr["pullback_level"][i]) if _sarr["pullback_level"] is not None else float("nan"),
            "adx_14":         float(_sarr["adx_14"][i]) if _sarr["adx_14"] is not None else 20.0,
            "rsi_14":         float(_sarr["rsi_14"][i]) if _sarr["rsi_14"] is not None else 50.0,
            "ema_stack":      int(_sarr["ema_stack"][i]) if _sarr["ema_stack"] is not None else 0,
            "bb_width":       float(_sarr["bb_width"][i]) if _sarr["bb_width"] is not None else 0.0,
            "bos_bull":       bool(_sarr["bos_bull"][i] == 1) if _sarr["bos_bull"] is not None else False,
            "bos_bear":       bool(_sarr["bos_bear"][i] == 1) if _sarr["bos_bear"] is not None else False,
            "fvg_bull":       bool(_sarr["fvg_bull"][i] == 1) if _sarr["fvg_bull"] is not None else False,
            "fvg_bear":       bool(_sarr["fvg_bear"][i] == 1) if _sarr["fvg_bear"] is not None else False,
            "last_swing_high": float(_sarr["last_swing_high"][i]) if _sarr["last_swing_high"] is not None else float("nan"),
            "last_swing_low":  float(_sarr["last_swing_low"][i]) if _sarr["last_swing_low"] is not None else float("nan"),
            "regime_duration": float(_sarr["regime_duration"][i]) if _sarr["regime_duration"] is not None else 0.5,
            "vol_slope_seq":   float(_sarr["vol_slope_seq"][i]) if _sarr["vol_slope_seq"] is not None else 0.0,
            "volume_ratio":    float(_sarr["volume_ratio"][i]) if _sarr["volume_ratio"] is not None else 1.0,
        }
        raw_signal = _compute_backtest_signal(symbol, ml_preds, bar)
        if raw_signal is None:
            _dbg["no_signal"] += 1
            continue

        # ── Soft density penalty: exp decay on confidence ─────────────────────
        _density_mult = math.exp(-_cfg_density_lambda * _density_count)
        raw_signal["confidence"] = float(raw_signal["confidence"]) * _density_mult

        if _cfg_min_conf > 0 and raw_signal["confidence"] < _cfg_min_conf:
            _dbg["density"] += 1
            continue
        if raw_signal["confidence"] < _cfg_dir_thresh:
            _dbg["density"] += 1
            continue

        # Apply slippage
        entry_raw = float(raw_signal["entry"])
        raw_signal["entry"] = (entry_raw * (1 + SLIPPAGE_PCT)
                               if raw_signal["side"] == "buy"
                               else entry_raw * (1 - SLIPPAGE_PCT))

        atr = float(bar.get("atr_14", entry_raw * 0.001))

        # ── Circuit breakers 6-8: PM gates (R:R, correlation, streak) ─────────
        portfolio_state = {"equity": equity, "open_positions_detail": []}
        enriched = pm.enrich_signal(raw_signal, portfolio_state, atr=atr)
        if enriched is None:
            _dbg["pm_reject"] += 1
            continue

        entry    = float(enriched["entry"])
        sl       = float(enriched["stop_loss"])
        tp1      = float(enriched["tp1"])
        tp2      = float(enriched["tp2"])
        size     = float(enriched["size"])
        rr_ratio = float(enriched["rr_ratio"])

        # ── EV gate: run QualityScorer now that we have actual signal context ──
        # QualityScorer needs rr_ratio, strategy_id, side — only known post-signal.
        if _qs_model is not None and _qs_fe is not None and ml_preds:
            _sig_ctx = {
                "trader_id": trader_id,
                "side":      enriched["side"],
                "rr_ratio":  rr_ratio,
            }
            # Cache stores p_bull/p_bear; feature engine expects p_bull_gru/p_bear_gru.
            # Augment ml_preds with aliased keys + bar-level features before scoring.
            _qs_preds = dict(ml_preds)
            _qs_preds.setdefault("p_bull_gru", _qs_preds.get("p_bull", 0.5))
            _qs_preds.setdefault("p_bear_gru", _qs_preds.get("p_bear", 0.5))
            _qs_preds.setdefault("regime_duration", float(bar.get("regime_duration", 0.5)))
            _qs_preds.setdefault("vol_slope", float(bar.get("vol_slope_seq", 0.0)))
            _qs_preds.setdefault("volume_ratio", float(bar.get("volume_ratio", 1.0)))
            _qs_feats = _qs_fe.get_quality_features(_sig_ctx, _qs_preds, bar)
            if _qs_infer is not None:
                # CPU path: ~3–5 µs vs ~500 µs GPU kernel launch for batch=1
                _ev, _qs = _qs_infer(_qs_feats)
            else:
                _qs_dict   = dict(zip(QUALITY_FEATURES, _qs_feats))
                _qs_result = _qs_model.predict(_qs_dict)
                _ev, _qs   = float(_qs_result["ev"]), float(_qs_result["quality_score"])
            ml_preds["ev"]            = _ev
            ml_preds["quality_score"] = _qs
            # Block trades where predicted EV < threshold. 0.0 = only take trades the
            # model believes are positive EV. Set MIN_EV_THRESHOLD env var to override.
            # Previously defaulted to -1.0 (never blocks) — raised to 0.0 so the quality
            # gate actually filters predicted-negative-EV setups.
            if _ev < _cfg_min_ev:
                _dbg["quality_block"] += 1
                continue
        elif ml_preds and ml_preds.get("ev") is not None:
            # Cache had ev (e.g. from _run_bar_ml fallback path)
            if float(ml_preds["ev"]) < _cfg_min_ev:
                _dbg["quality_block"] += 1
                continue

        # ── Simulate trade ────────────────────────────────────────────────────
        result = _simulate_trade_pm(df, i, enriched["side"], entry, sl, tp1, tp2, size, atr)
        pnl = result["pnl"]

        equity += pnl
        daily_loss += min(0.0, pnl)
        if equity > peak_equity:
            peak_equity = equity

        # Update PM streak + daily accumulator
        sl_dist     = abs(entry - sl)
        rr_achieved = abs(pnl / (sl_dist * size + 1e-9))
        pm.record_outcome(trader_id, pnl, rr_achieved)

        last_trade_bar[symbol] = i
        recent_trade_bars[symbol].append(i)

        meta = raw_signal.get("signal_metadata", {})
        peak_eq = peak_equity if peak_equity > 0 else 1.0
        trades.append({
            "timestamp":   dt.isoformat(),
            "trader":      trader_id,
            "symbol":      symbol,
            "side":        enriched["side"],
            "size":        size,
            "entry":       round(entry, 6),
            "stop_loss":   round(sl, 6),
            "tp1":         round(tp1, 6),
            "tp2":         round(tp2, 6),
            "confidence":  enriched.get("confidence", 0.0),
            "rr_ratio":    rr_ratio,
            "pnl":         round(pnl, 4),
            "exit_reason": result["exit_reason"],
            "tp1_hit":     result["tp1_hit"],
            "bars_held":   result["bars_held"],
            "equity":      round(equity, 2),
            "pm_info":     enriched.get("portfolio_manager", {}),
            # RL state fields — used by step6_backtest to build state_at_entry
            "p_bull":       float((ml_preds or meta).get("p_bull", 0.5)),
            "p_bear":       float((ml_preds or meta).get("p_bear", 0.5)),
            "quality_score": float((ml_preds or meta).get("ev", 0.0)),
            "ev":           float((ml_preds or meta).get("ev", 0.0)),
            "expected_variance": float((ml_preds or meta).get("expected_variance", 0.0)),
            "regime":       str(ml_preds.get("regime") or meta.get("regime", "RANGING")),
            "adx":          float(bar.get("adx_14", 20.0)),
            "atr":          float(bar.get("atr_14", atr)),
            "ema_stack":    int(bar.get("ema_stack", 0)),
            "bb_width":     float(bar.get("bb_width", 0.0)),
            "rsi":          float(bar.get("rsi_14", 50.0)),
            "bos_bull":     bool(bar.get("bos_bull", False)),
            "bos_bear":     bool(bar.get("bos_bear", False)),
            "fvg_bull":     bool(bar.get("fvg_bull", False)),
            "fvg_bear":     bool(bar.get("fvg_bear", False)),
            "drawdown":     float((peak_eq - equity) / peak_eq),
            "equity_pct":   float(equity / INITIAL_CAPITAL),
            # Diagnostics fields
            "session_hour":  dt.hour,
            "regime_duration": float(bar.get("regime_duration", 0.5)),
            "session_weight":  float(meta.get("session_weight", 1.0)),
            "regime_weight":   float(meta.get("regime_weight", 1.0)),
            "age_weight":      float(meta.get("age_weight", 1.0)),
            "realized_rr":   float(abs(pnl) / (abs(entry - sl) * size + 1e-9)) * (1 if pnl > 0 else -1),
            "density_count": _density_count,
        })

    # ─── Metrics ────────────────────────────────────────────────────────────
    _loop_elapsed = _time.perf_counter() - _loop_t0
    _total_bars = _dbg["total"]
    logger.info(
        "%s loop COMPLETE: %.1fs | %d bars | %.0f bars/s | %d trades",
        trader_id, _loop_elapsed, _total_bars,
        _total_bars / max(0.01, _loop_elapsed), len(trades),
    )
    logger.info(
        "%s diagnostics — bars=%d session_skip=%d daily_skip=%d cooldown=%d "
        "density_suppressed=%d quality_blocked=%d no_signal=%d pm_reject=%d",
        trader_id, _dbg["total"], _dbg["session"], _dbg["daily"],
        _dbg["cooldown"], _dbg["density"], _dbg["quality_block"],
        _dbg["no_signal"], _dbg["pm_reject"]
    )
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "tp1_rate": 0.0,
            "tp2_rate": 0.0,
            "trade_log": [],
            "gate_diagnostics": dict(_dbg),
        }

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    win_rate = len(wins) / len(pnls)
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / (gross_loss + 1e-9)
    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL

    tp1_hits = sum(1 for t in trades if t["tp1_hit"])
    tp2_hits = sum(1 for t in trades if t["exit_reason"] == "tp2")

    # Max drawdown from equity curve
    eq_curve = [INITIAL_CAPITAL]
    for p in pnls:
        eq_curve.append(eq_curve[-1] + p)
    peak = eq_curve[0]
    max_dd = 0.0
    for v in eq_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / (peak + 1e-9)
        if dd > max_dd:
            max_dd = dd

    # Sharpe (annualized)
    sharpe = 0.0
    if len(pnls) >= 2:
        mean_r = np.mean(pnls)
        std_r = np.std(pnls)
        sharpe = float((mean_r / (std_r + 1e-9)) * np.sqrt(252))

    # Avg bars held
    avg_bars = float(np.mean([t["bars_held"] for t in trades]))

    return {
        "trades": len(trades),
        "win_rate": round(win_rate, 4),
        "total_return": round(total_return, 4),
        "profit_factor": round(profit_factor, 4),
        "max_drawdown": round(max_dd, 4),
        "sharpe": round(sharpe, 4),
        "tp1_rate": round(tp1_hits / len(trades), 4),
        "tp2_rate": round(tp2_hits / len(trades), 4),
        "avg_bars_held": round(avg_bars, 1),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "trade_log": trades,
        "gate_diagnostics": dict(_dbg),
    }


def _run_trader_worker(args_tuple: tuple) -> tuple:
    """
    Top-level worker function for parallel backtest execution.
    Each worker runs in a separate process and loads its own model instances
    so there is no shared GPU/CPU state between traders.

    GPU assignment: with N GPUs available, worker index w uses GPU (w % N).
    This spreads 5 traders across 2 T4s: traders 1,3,5 → GPU 0; traders 2,4 → GPU 1.

    Returns (trader_id, result_dict, trade_log_list).
    """
    tid, bt_start, bt_end, ml_enabled, engine_dir, worker_idx, all_symbols = args_tuple

    # Re-insert path so imports work in the subprocess
    sys.path.insert(0, engine_dir)
    os.chdir(engine_dir)

    # Pin this worker to a specific GPU — round-robin across all visible GPUs.
    # Must happen BEFORE any CUDA context is created (before torch import).
    import torch as _torch_probe
    n_gpu = _torch_probe.cuda.device_count() if _torch_probe.cuda.is_available() else 0
    if n_gpu > 0:
        gpu_id = worker_idx % n_gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info("Worker %d pinned to GPU %d", worker_idx, gpu_id)

    # Each worker loads its own model copies — avoids pickling GPU tensors
    worker_ml_models: dict = {}
    if ml_enabled:
        try:
            _ensure_mpl_config_dir()
            from models.regime_classifier import RegimeClassifier
            from models.quality_scorer import QualityScorer
            from models.gru_lstm_predictor import GRULSTMPredictor
            r = RegimeClassifier()
            if _model_ready(r):
                worker_ml_models["regime"] = r
            q = QualityScorer()
            if _model_ready(q):
                worker_ml_models["quality"] = q
            g = GRULSTMPredictor()
            if _model_ready(g):
                worker_ml_models["gru_lstm"] = g
        except Exception as exc:
            logger.error("Worker ML model load failed: %s", exc)
            raise

    from monitors.portfolio_manager import PortfolioManager
    pm = PortfolioManager(_PM_SETTINGS, bar_date=bt_start)

    symbols = TRADER_SYMBOLS.get("ml_trader", _ALL_SYMBOLS)
    result = _backtest_trader("ml_trader", symbols, pm, bt_start, bt_end,
                               ml_models=worker_ml_models)
    trade_log = result.pop("trade_log", [])
    return "ml_trader", result, trade_log


def _load_split_window(split: str) -> tuple[str, str]:
    """
    Read split_summary.json produced by pipeline/step5_split.py and return
    (start, end) date strings for the requested split name.

    Falls back to a 2-year rolling window ending today if the file is missing.
    """
    # Resolve the JSON relative to this script's location so it works both
    # locally and on Kaggle (where env_config may redirect paths).
    _candidates = [
        # Kaggle: remote clone or working copy
        Path(_ENGINE_DIR) / ".." / "ml_training" / "datasets" / "split_summary.json",
        # env_config path
        Path(_DATA_DIR_RESOLVED).parent.parent / "ml_training" / "datasets" / "split_summary.json",
    ]
    if "_ENV" in globals() and _ENV is not None:
        _candidates.insert(0, _ENV.get("ml_training", Path()) / "datasets" / "split_summary.json")

    for _p in _candidates:
        try:
            _p = Path(_p).resolve()
            if _p.exists():
                with open(_p) as _f:
                    _summary = json.load(_f)
                _dr = _summary["date_ranges"][split]
                return _dr["start"][:10], _dr["end"][:10]
        except Exception:
            continue

    # Fallback: rolling 2-year window ending today
    _today = datetime.now(timezone.utc)
    return (
        _today.replace(year=_today.year - 2).strftime("%Y-%m-%d"),
        _today.strftime("%Y-%m-%d"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Backtest with ML-native execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Date windows are read from split_summary.json (pipeline/step5_split.py):\n"
            "  default (--split val) → validation window  (Round 1 backtest, SAFE)\n"
            "  --split test          → test/blind window   (Round 2 only — do not overfit)\n"
            "  --split train         → training window     (sanity check only)\n"
            "  --start/--end         → manual override\n"
        ),
    )
    parser.add_argument(
        "--split",
        choices=["validation", "val", "test", "train"],
        default="validation",
        help="Which split window to backtest (default: validation = Round 1)",
    )
    parser.add_argument("--start", default=None,
                        help="Override split start date (YYYY-MM-DD)")
    parser.add_argument("--end",   default=None,
                        help="Override split end date (YYYY-MM-DD)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run confidence calibration report after backtest")
    args = parser.parse_args()

    # Resolve the date window: manual override > split file > rolling 2-yr fallback
    _split_key = "validation" if args.split == "val" else args.split
    _split_start, _split_end = _load_split_window(_split_key)
    bt_start = args.start or _split_start
    bt_end   = args.end   or _split_end
    logger.info(
        "Backtest window: %s → %s  (split=%s%s)",
        bt_start, bt_end, _split_key,
        " [BLIND — test set]" if _split_key == "test" else "",
    )

    from monitors.portfolio_manager import PortfolioManager

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    run_at = datetime.now(timezone.utc)
    results = {}
    all_trade_logs = []

    # ── Load ML models (if ML_ENABLED) ───────────────────────────────────────
    ml_models: dict = {}
    if _env_ml_enabled():
        logger.info("ML_ENABLED=True — loading models for backtest inference...")
        _ensure_mpl_config_dir()

        try:
            from models.regime_classifier import RegimeClassifier
            # Load HTF bias classifier (regime_htf.pkl) — 3-class BIAS_UP/DOWN/NEUTRAL
            regime_htf = RegimeClassifier(timeframe="4H", mode="htf_bias")
            if _model_ready(regime_htf):
                ml_models["regime_htf"] = regime_htf
                ml_models["regime_4h"] = regime_htf   # legacy alias
                logger.info("  [OK] RegimeClassifier[HTF bias 3-class] loaded")
            else:
                logger.warning("  [SKIP] RegimeClassifier[HTF] unavailable (weights missing)")
            # Load LTF behaviour classifier (regime_ltf.pkl) — 4-class TRENDING/RANGING/CONSOLIDATING/VOLATILE
            regime_ltf = RegimeClassifier(timeframe="1H", mode="ltf_behaviour")
            if _model_ready(regime_ltf):
                ml_models["regime_ltf"] = regime_ltf
                ml_models["regime_1h"] = regime_ltf   # legacy alias
                logger.info("  [OK] RegimeClassifier[LTF behaviour 4-class] loaded")
            else:
                logger.warning("  [SKIP] RegimeClassifier[LTF] unavailable (weights missing)")
            # Legacy key for backwards compat (points to HTF)
            if "regime_htf" in ml_models:
                ml_models["regime"] = ml_models["regime_htf"]
        except Exception as exc:
            logger.warning("  [SKIP] RegimeClassifier load failed: %s", exc)

        try:
            from models.quality_scorer import QualityScorer
            qs = QualityScorer()
            if _model_ready(qs):
                ml_models["quality"] = qs
                logger.info("  [OK] QualityScorer loaded")
            else:
                logger.warning("  [SKIP] QualityScorer unavailable (weights missing or load failed)")
        except Exception as exc:
            logger.warning("  [SKIP] QualityScorer load failed: %s", exc)

        try:
            from models.gru_lstm_predictor import GRULSTMPredictor
            gru = GRULSTMPredictor()
            if _model_ready(gru):
                ml_models["gru_lstm"] = gru
                logger.info("  [OK] GRU-LSTM loaded")
            else:
                logger.warning("  [SKIP] GRU-LSTM unavailable (weights missing or load failed)")
        except Exception as exc:
            logger.warning("  [SKIP] GRU-LSTM load failed: %s", exc)

        if not ml_models:
            raise RuntimeError(
                "ML_ENABLED=True but no models loaded. "
                "Run: python scripts/retrain_incremental.py --model all"
            )
        logger.info("ML models loaded: %s", list(ml_models.keys()))
    else:
        logger.info("ML_ENABLED=False — running rule-only backtest")

    # Initialise CandidateLogger for backtest-mode candidate logging
    try:
        from services.candidate_logger import CandidateLogger
        from services.quant_analytics import EVFilter
        _bt_candidate_logger = CandidateLogger(csv_path="logs/backtest_candidate_log.csv")
        _bt_ev_filter = EVFilter(candidate_log_path="logs/backtest_candidate_log.csv")
        logger.info("CandidateLogger and EVFilter initialised for backtest")
    except Exception as _e:
        _bt_candidate_logger = None
        _bt_ev_filter = None
        logger.warning("Could not init CandidateLogger/EVFilter: %s", _e)

    # ── Run single ML trader ─────────────────────────────────────────────────
    symbols = TRADER_SYMBOLS["ml_trader"]
    logger.info("Running ml_trader on %d symbols: %s", len(symbols), symbols)
    pm = PortfolioManager(_PM_SETTINGS, bar_date=bt_start)
    shared_ml_cache: dict = {} if ml_models else None
    result = _backtest_trader("ml_trader", symbols, pm, bt_start, bt_end,
                              ml_models=ml_models,
                              shared_ml_cache=shared_ml_cache)
    trade_log = result.pop("trade_log", [])
    results["ml_trader"] = result
    all_trade_logs.extend(trade_log)

    # Log backtest trades as candidates with outcomes for calibration
    for trader_id, result in results.items():
        trade_log = [t for t in all_trade_logs if t.get("trader") == trader_id]
        if _bt_candidate_logger is not None:
            for tr in trade_log:
                cid = _bt_candidate_logger.log_candidate(
                    trader_id=trader_id,
                    symbol=tr.get("symbol", ""),
                    side=tr.get("side", "buy"),
                    features={
                        "rr_ratio":   tr.get("rr_ratio", 1.5),
                        "confidence": tr.get("confidence", 0.7),
                        "p_win":      tr.get("confidence", 0.7),
                    },
                    executed=True,
                )
                tp_hit = tr.get("tp1_hit", False) or tr.get("exit_reason", "") == "tp2"
                sl_hit = tr.get("exit_reason", "") == "sl"
                _bt_candidate_logger.mark_outcome(
                    candidate_id=cid,
                    tp_hit=bool(tp_hit),
                    sl_hit=bool(sl_hit),
                    pnl=float(tr.get("pnl", 0.0)),
                    exit_reason=str(tr.get("exit_reason", "")),
                )

        logger.info(
            "%s: %d trades | WR=%.1f%% | PF=%.2f | Return=%.1f%% | "
            "TP1=%.1f%% TP2=%.1f%% | DD=%.1f%% | Sharpe=%.2f",
            trader_id,
            result["trades"],
            result["win_rate"] * 100,
            result["profit_factor"],
            result["total_return"] * 100,
            result["tp1_rate"] * 100,
            result["tp2_rate"] * 100,
            result["max_drawdown"] * 100,
            result["sharpe"],
        )

    # ── Calibration report ────────────────────────────────────────────────────
    calibration_report = {}
    if args.calibrate or True:  # always generate when candidate data exists
        try:
            from services.quant_analytics import ConfidenceCalibrator
            cal = ConfidenceCalibrator(candidate_log_path="logs/backtest_candidate_log.csv")
            calibration_report = cal.run_all_traders()
            logger.info("Calibration report generated: %s", ConfidenceCalibrator.REPORT_PATH)
        except Exception as _ce:
            logger.warning("Calibration report failed: %s", _ce)

    output = {
        "run_at": run_at.isoformat(),
        "start": bt_start,
        "end":   bt_end,
        "split": _split_key,
        "config": {
            "trader": "ml_trader",
            "initial_capital": INITIAL_CAPITAL,
            "risk_per_trade": RISK_PER_TRADE,
            "max_daily_loss_pct": MAX_DAILY_LOSS_PCT,
            "max_drawdown_halt": MAX_DRAWDOWN_PCT,
            "portfolio_manager": "enabled",
        },
        "results": results,
        "trade_log": all_trade_logs,
        "calibration": calibration_report,
    }

    filename = f"backtest_{run_at.strftime('%Y%m%d_%H%M%S')}.json"
    outpath = os.path.join(OUTPUT_DIR, filename)
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("Backtest complete → %s", outpath)
    print(f"\nBacktest results → {outpath}")
    print(f"{'Trader':<40} {'Trades':>6} {'WR':>7} {'PF':>6} {'Return':>8} {'TP1%':>6} {'TP2%':>6} {'DD':>7} {'Sharpe':>7}")
    print("-" * 105)
    for tid, r in results.items():
        name = TRADER_NAMES.get(tid, tid)
        print(
            f"{name:<40} {r['trades']:>6} {r['win_rate']:>6.1%} {r['profit_factor']:>6.2f}"
            f" {r['total_return']:>7.1%} {r['tp1_rate']:>5.1%} {r['tp2_rate']:>5.1%}"
            f" {r['max_drawdown']:>6.1%} {r['sharpe']:>7.2f}"
        )
        gd = r.get("gate_diagnostics") or {}
        if gd:
            print(
                f"  gate_diagnostics: bars={gd.get('total')} no_signal={gd.get('no_signal')} "
                f"quality_block={gd.get('quality_block')} session_skip={gd.get('session')} "
                f"density={gd.get('density')} pm_reject={gd.get('pm_reject')} "
                f"daily_skip={gd.get('daily')} cooldown={gd.get('cooldown')}"
            )

    # Print calibration summary
    if calibration_report.get("traders"):
        print("\nCalibration Summary:")
        for tid, cal_result in calibration_report["traders"].items():
            reliable = cal_result.get("reliable")
            note = cal_result.get("note", "")
            status = "OK" if reliable else ("WARN" if reliable is not None else "N/A")
            print(f"  {tid:<12} [{status}] {note[:80]}")


if __name__ == "__main__":
    main()
