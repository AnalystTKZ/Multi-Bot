"""
feature_engine.py — Pure feature computation for all ML models and RL state.

No ML logic here — pure feature math. No lookahead. No side effects.
All outputs: numpy float32. No NaN. No Inf.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from indicators.market_structure import compute_atr, compute_ema

logger = logging.getLogger(__name__)


def _vec_autocorr(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Vectorized lag-1 autocorrelation over a rolling window.

    Replaces rolling().apply(pd.Series.autocorr) — that calls Python once per
    row, making it O(N×window) in Python. This uses a strided matrix to compute
    the Pearson formula in NumPy entirely, giving ~100-500× speedup on 240k rows.

    Returns float32 array aligned to arr; NaN-padded for the first (window-1) bars.
    """
    n = len(arr)
    out = np.full(n, 0.0, dtype=np.float32)
    if window < 3 or n < window:
        return out
    # Build (N-window+1, window) strided view — no data copy
    shape   = (n - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])
    wins    = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    # Pearson lag-1: cov(x[:-1], x[1:]) / (std(x[:-1]) * std(x[1:]))
    x0 = wins[:, :-1].astype(np.float64)   # shape (M, window-1)
    x1 = wins[:, 1:].astype(np.float64)
    m0 = x0.mean(axis=1, keepdims=True)
    m1 = x1.mean(axis=1, keepdims=True)
    d0 = x0 - m0
    d1 = x1 - m1
    cov  = (d0 * d1).mean(axis=1)
    std0 = d0.std(axis=1) + 1e-9
    std1 = d1.std(axis=1) + 1e-9
    ac   = np.clip(cov / (std0 * std1), -1.0, 1.0).astype(np.float32)
    out[window - 1:] = ac
    return out


def _vec_atr_pctile(atr_vals: np.ndarray, window: int = 42, min_periods: int = 14) -> np.ndarray:
    """
    Vectorized rolling ATR percentile rank.

    Replaces rolling().apply(lambda x: searchsorted(sort(x[:-1]), x[-1])) which
    calls Python once per row (O(N×W×logW) in Python). This builds a strided
    matrix and uses np.argsort to compute all ranks in one C-level call.

    Returns float32 [0, 1] array; 0.5-filled for the first (min_periods-1) bars.
    """
    n = len(atr_vals)
    out = np.full(n, 0.5, dtype=np.float32)
    arr = np.asarray(atr_vals, dtype=np.float64)
    eff_window = min(window, n)
    if n < min_periods or eff_window < 2:
        return out
    # Pad left so every row gets a full window — pad value is -inf so it ranks below all real values
    pad  = eff_window - 1
    padded = np.concatenate([np.full(pad, -np.inf), arr])
    shape   = (n, eff_window)
    strides = (padded.strides[0], padded.strides[0])
    wins = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)  # (N, W)
    # For each row: rank of wins[:, -1] (current bar) among wins[:, :-1] (history)
    # Use argsort rank: rank = number of historical values < current
    cur  = wins[:, -1:].copy()          # (N, 1) — must copy out of strided view
    hist = wins[:, :-1].copy()          # (N, W-1)
    ranks = (hist < cur).sum(axis=1).astype(np.float32)  # (N,)
    denom = np.clip((hist != -np.inf).sum(axis=1).astype(np.float32), 1.0, None)
    pctile = np.clip(ranks / denom, 0.0, 1.0)
    # Zero out rows that don't meet min_periods
    valid_from = min_periods - 1
    out[valid_from:] = pctile[valid_from:]
    return out

# ─── Feature name lists (contract: order and length fixed) ────────────────────

# Fixed list of 17 index names — must match training_data/indices/*_1d.csv files.
# NEVER discover dynamically: count instability breaks saved model weights.
INDEX_NAMES = [
    "asx200", "cac40", "dax", "djia", "dxy",
    "eurostoxx", "ftse", "gold_fut", "hsi", "nasdaq",
    "nikkei", "oil_fut", "spx", "us10y", "us30y",
    "us3m", "vix",
]
INDEX_FEATURES = [f"idx_{name}_ret" for name in INDEX_NAMES]

MACRO_FEATURES = INDEX_FEATURES + [
    "macro_vix_level",
    "macro_yield_spread",
]

SEQUENCE_FEATURES = [
    # ── 15M execution price action ───────────────────────────────────────────
    "log_return",
    "high_low_range",
    "close_vs_open",
    "atr_normalized",
    "rsi_14",
    "ema21_dist",
    "ema50_dist",
    "bb_position",
    "is_asian",
    "is_london",
    "is_ny",
    # ── Cross-timeframe technical context ────────────────────────────────────
    "mtf_5m_rsi",
    "mtf_5m_ema21_dist",
    "mtf_1h_adx",
    "mtf_1h_ema21_dist",
    "mtf_1h_ema50_dist",
    "mtf_4h_ema21_ema50_diff",
    "mtf_4h_adx",
    "mtf_4h_rsi",
    "mtf_1d_ema21_dist",
    "mtf_1d_ema_stack",
    # ── Volatility and time ──────────────────────────────────────────────────
    "vol_slope_seq",
    "time_sin",
    "time_cos",
    "mins_since_london_open",
    "mins_since_ny_open",
    # ── EMA / market structure ───────────────────────────────────────────────
    "ema_pullback_zone",
    "ema21_slope_15m",
    "ema21_slope_1h",
    "ema_stack_15m",
    "hh_hl_structure",
    "lh_ll_structure",
    # ── BOS / FVG / sweeps ───────────────────────────────────────────────────
    "bos_bull_flag",
    "bos_bear_flag",
    "bos_bull_bars_ago",
    "bos_bear_bars_ago",
    "bos_bull_strength",
    "bos_bear_strength",
    "fvg_bull_open",
    "fvg_bear_open",
    "fvg_bull_dist_atr",
    "fvg_bear_dist_atr",
    "fvg_bull_fill_ratio",
    "fvg_bear_fill_ratio",
    "sweep_wick_depth_atr",
    "body_recovery_ratio",
    # ── Liquidity / candle execution context ─────────────────────────────────
    "dist_to_recent_high_atr",
    "dist_to_recent_low_atr",
    "asian_range_width_atr",
    "price_vs_asian_high_atr",
    "price_vs_asian_low_atr",
    "candle_body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "adx_15m",
    "vol_expansion",
    "atr_pctile",
    "vwap_dist_atr",
    "wick_auction_ratio",
    # ── MSS / CHoCH (market structure shift) ─────────────────────────────────
    "mss_bull_flag",            # 1 at bar where bullish MSS/CHoCH occurred
    "mss_bear_flag",            # 1 at bar where bearish MSS/CHoCH occurred
    "mss_bull_bars_ago",        # normalised bars since last bullish MSS (0=recent, 1=cap)
    "mss_bear_bars_ago",        # normalised bars since last bearish MSS
    "bars_since_mss",           # normalised bars since any MSS/CHoCH
    # ── External (major-swing) structure ─────────────────────────────────────
    "external_trend_direction", # +1 bullish major swing sequence, -1 bearish
    "external_structure_score", # +1 = external HH, -1 = external LH (normalised to [-1,1])
    "internal_structure_state", # +1 internal HH/HL, -1 internal LH/LL
    "swing_sequence_score",     # weighted internal/external structure agreement
    "position_in_external_range",# price position inside [ext_swing_low, ext_swing_high] in [0,1]
    "dist_to_external_high_atr",# (ext_swing_high - close) / ATR — clipped [-5, 5]
    "dist_to_external_low_atr", # (close - ext_swing_low)  / ATR — clipped [-5, 5]
]
# Total: technical execution/structure features.

# ─── 4H BIAS classifier features ─────────────────────────────────────────────
# Trained on 4H data. Only HTF-appropriate features: no 5M/15M noise.
# Macro indices belong here — they operate at daily/weekly resolution, match 4H bias.
REGIME_4H_FEATURES = [
    # Lean HTF directional-bias set. Raw macro/index columns were noisy and
    # path-dependent, so bias learns from price structure and normalised context.
    "adx_14_base",
    "ema_stack_score",
    "mtf_1d_adx",
    "mtf_1d_ema_stack",
    "mtf_1d_atr_ratio",
    "efficiency_ratio",
    "plus_di",
    "minus_di",
    "ema_50_slope",
    "ema_200_slope",
    "ema_50_dist_atr",
    "ema_200_dist_atr",
    "atr_percentile_500",
    "rolling_range_percentile",
    "hh_hl_structure",
    "lh_ll_structure",
    "external_trend_direction",
    "external_structure_score", # major-swing direction from larger swing window
    "internal_structure_state",
    "swing_sequence_score",
    "bars_since_mss",
    "mss_bull_bars_ago",        # bars since last bullish MSS at 4H resolution
    "mss_bear_bars_ago",        # bars since last bearish MSS at 4H resolution
    "symbol_group_code",
]

# ─── 1H STRUCTURE classifier features ────────────────────────────────────────
# Trained on 1H data. No macro indices — too coarse for 1H structure decisions.
REGIME_1H_FEATURES = [
    # Lean LTF behaviour set: trend/range/chop/volatility/consolidation scores
    # should learn structure, not dozens of duplicated volatility variants.
    "adx_14_base",
    "mtf_4h_adx",
    "mtf_4h_ema_stack",
    "session_code",
    "efficiency_ratio",
    "plus_di",
    "minus_di",
    "ema_50_slope",
    "ema_50_dist_atr",
    "atr_percentile_500",
    "rolling_vol_percentile",
    "bb_width_percentile",
    "rolling_range_percentile",
    "range_expansion_zscore",
    "wick_ratio",
    "hh_hl_structure",
    "lh_ll_structure",
    "swing_hh_hl_count",
    "liquidity_sweep_24h",
    "vol_slope",
    "mss_bull_flag",
    "mss_bear_flag",
    "mss_bull_bars_ago",
    "mss_bear_bars_ago",
    "bars_since_mss",
    "external_trend_direction",
    "external_structure_score",
    "internal_structure_state",
    "swing_sequence_score",
    "position_in_external_range",
    "symbol_group_code",
]

QUALITY_FEATURES = [
    "strategy_id",             # 0  numeric strategy code, stable across train/inference
    "signal_direction",        # 1  buy=1, sell=0
    "rr_ratio",                # 2  planned geometric reward/risk
    "p_win_gru",               # 3  GRU probability aligned to the candidate side
    "gru_edge",                # 4  side probability minus opposite-side probability
    "expected_move",           # 5  short-horizon move magnitude score from GRU
    "gru_uncertainty",         # 6  expected_variance from GRU variance head
    "trade_regime_code",       # 7  compact LTF trade-regime suitability code (RANGE=0.55, TREND=1.0)
    "expected_r_gross",        # 8  probability-weighted EV before costs: p_win×rr − (1−p_win)×1
                               #    replaces htf_bias_alignment (always 1.0 after tradeability gate)
    "volatility_percentile",   # 9  regime volatility percentile, not raw volatility
    "chop_score",              # 10 noisy/two-way behaviour score
    "adx_at_signal",           # 11 directional strength at candidate bar
    "atr_ratio_at_signal",     # 12 symbol-normalised ATR ratio at candidate bar
    "spread_at_signal",        # 13 execution friction in raw pips
    "session_at_signal",       # 14 compact session code
    "news_in_30min",           # 15 near-news execution risk
    "strategy_win_rate_5",     # 16 prior short-term strategy outcome context
    "strategy_win_rate_20",    # 17 prior medium-term strategy outcome context
    "strategy_win_rate_50",    # 18 prior longer-term strategy outcome context
    "vol_slope_at_signal",     # 19 expanding vs contracting volatility
]  # 20 features — contract fixed; add features only by replacing degenerate ones

# RL state dimension layout (total = 43):
# [0-5]   GRU execution forecast (p_bull, p_bear, strength, entry_depth, expected_move, variance)
# [6-8]   HTF directional-bias probabilities
# [9-14]  LTF score-based regime context
# [15-16] Regime confidence
# [17-26] Bar structure, SMC / auction context, and execution friction
# [27-30] Time/session/news context
# [31-36] Portfolio context
# [37-40] Instrument one-hot
# [41-42] Setup validity flags (range, pullback)
RL_STATE_DIM = 43

_INSTRUMENT_IDX = {"EURUSD": 0, "GBPUSD": 1, "USDJPY": 2, "XAUUSD": 3}

_MACRO_CACHE: Dict[str, pd.Series] = {}
_MACRO_MAP_CACHE: Dict[str, Any] | None = None
_MACRO_MAP_MTIME: float = 0.0
# Aligned macro DataFrame cache: keyed by (symbol, index_id) so that
# _build_macro_frame only does the reindex+ffill once per symbol per run.
# Reindexing 19 daily series onto a 300k-bar 15M index is O(N) pandas work —
# caching avoids repeating it for every call from _build_feature_matrix.
_MACRO_ALIGNED_CACHE: Dict[tuple, "pd.DataFrame"] = {}

_QUALITY_STRATEGY_MAP = {
    "ml_trader": 0.0,
    "trader_1": 1.0,
    "trader_2": 2.0,
    "trader_3": 3.0,
    "trader_4": 4.0,
    "trader_5": 5.0,
}

_QUALITY_SESSION_MAP = {
    "INACTIVE": 0.0,
    "DEAD": 0.0,
    "ASIAN": 0.25,
    "LONDON": 0.75,
    "NY": 1.0,
}

_TRADE_REGIME_CODE = {
    # No-trade states — lowest code (0 = no edge)
    "NO_TRADE_CHOP":         0.0,
    "NO_TRADE_EXTREME_VOL":  0.0,
    "NO_TRADE_UNCERTAIN":    0.0,
    "UNCERTAIN":             0.10,
    "CONSOLIDATION":         0.25,
    # Tradeable states
    "RANGE":                 0.55,
    "TRADEABLE_TREND_HIGH_VOL": 0.75,
    "TRADEABLE_TREND":       1.0,
    # New directional states — same quality tier as their trend equivalents
    "TRADEABLE_UP":          1.0,
    "TRADEABLE_DOWN":        1.0,
}


def _quality_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def _quality_strategy_code(trader_id: Any) -> float:
    return float(_QUALITY_STRATEGY_MAP.get(str(trader_id or ""), 0.0))


def _quality_session_code(session: Any) -> float:
    return float(_QUALITY_SESSION_MAP.get(str(session or "INACTIVE").upper(), 0.0))


def _quality_regime_scores(source: dict) -> dict:
    scores = source.get("regime_scores", {}) if isinstance(source, dict) else {}
    out = dict(scores) if isinstance(scores, dict) else {}
    if isinstance(source, dict):
        for key in (
            "trend_score",
            "range_score",
            "chop_score",
            "volatility_percentile",
            "consolidation_score",
        ):
            if key in source and key not in out:
                out[key] = source[key]
    return out


def _quality_trade_regime_code(trade_regime: Any, regime_scores: Optional[dict] = None) -> float:
    regime = str(trade_regime or "").upper()
    if regime in _TRADE_REGIME_CODE:
        return float(_TRADE_REGIME_CODE[regime])

    scores = regime_scores if isinstance(regime_scores, dict) else {}
    trend = _quality_float(scores.get("trend_score", scores.get("TREND_SCORE", 0.0)), 0.0)
    range_score = _quality_float(scores.get("range_score", scores.get("RANGE_SCORE", 0.0)), 0.0)
    chop = _quality_float(scores.get("chop_score", scores.get("CHOP_SCORE", 0.0)), 0.0)
    vol = _quality_float(
        scores.get("volatility_percentile", scores.get("VOLATILITY_PERCENTILE", 0.5)),
        0.5,
    )
    consolidation = _quality_float(
        scores.get("consolidation_score", scores.get("CONSOLIDATION_SCORE", 0.0)),
        0.0,
    )
    if vol > 0.90 or chop > 0.65:
        return 0.0
    if trend > 0.65:
        return 0.75 if vol > 0.75 else 1.0
    if range_score > 0.65 and trend < 0.35:
        return 0.55
    if consolidation > 0.60 or (vol < 0.25 and trend < 0.35):
        return 0.25
    return 0.10


def _quality_expected_r_gross(p_win: float, rr_ratio: float) -> float:
    """
    Probability-weighted expected R before execution costs.
    EV = p_win × rr_ratio − (1 − p_win) × 1.0

    This replaces htf_bias_alignment (feature 8), which became degenerate after
    the directional tradeability gate: all passing trades have alignment=1.0, so
    it contributed zero discriminative signal.  expected_r_gross directly encodes
    how much edge this setup carries according to GRU before quality refinement.
    """
    ev = float(p_win) * float(rr_ratio) - (1.0 - float(p_win)) * 1.0
    return float(np.clip(ev, -2.0, 6.0))


def _quality_side_probabilities(source: dict, side: Any) -> tuple[float, float]:
    p_bull = np.clip(
        _quality_float(source.get("p_bull_gru", source.get("p_bull", 0.5)), 0.5),
        0.0,
        1.0,
    )
    p_bear = np.clip(
        _quality_float(source.get("p_bear_gru", source.get("p_bear", 1.0 - p_bull)), 1.0 - p_bull),
        0.0,
        1.0,
    )
    if str(side or "").lower() == "sell":
        return float(p_bear), float(p_bull)
    return float(p_bull), float(p_bear)


def _first_present_frame(
    frames: Optional[Dict[str, pd.DataFrame]],
    *keys: str,
    default: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """Return the first non-None frame for the given keys without truth-testing DataFrames."""
    if not isinstance(frames, dict):
        return default
    for key in keys:
        frame = frames.get(key)
        if frame is not None:
            return frame
    return default


def _load_series(path: Path, date_col: str, value_col: str) -> "pd.Series | None":
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if date_col not in df.columns or value_col not in df.columns:
        return None
    idx = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    vals = pd.to_numeric(df[value_col], errors="coerce")
    s = pd.Series(vals.values, index=idx).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _load_macro_cache() -> Dict[str, pd.Series]:
    global _MACRO_CACHE
    if _MACRO_CACHE:
        return _MACRO_CACHE
    base = Path("training_data")
    idx_dir = base / "indices"
    fund_dir = base / "fundamental"
    _MACRO_CACHE = {}
    if idx_dir.exists():
        for f in sorted(idx_dir.glob("*_1d.csv")):
            name = f.stem.replace("_1d", "").lower()
            _MACRO_CACHE[name] = _load_series(f, "Datetime", "close")
    # fundamentals for yield spread
    _MACRO_CACHE["us10y_fred"] = _load_series(fund_dir / "treasury_10yr.csv", "Date", "DGS10")
    _MACRO_CACHE["us2y_fred"] = _load_series(fund_dir / "treasury_2yr.csv", "Date", "DGS2")
    _MACRO_CACHE = {k: v for k, v in _MACRO_CACHE.items() if v is not None and len(v) > 10}
    return _MACRO_CACHE


def _load_macro_map() -> Dict[str, Any]:
    global _MACRO_MAP_CACHE, _MACRO_MAP_MTIME
    for path in (Path("training_data") / "macro_correlations.json",
                 Path("models") / "weights" / "macro_correlations.json"):
        if not path.exists():
            continue
        mtime = path.stat().st_mtime
        if _MACRO_MAP_CACHE is not None and mtime == _MACRO_MAP_MTIME:
            return _MACRO_MAP_CACHE
        data = json.loads(path.read_text())
        _MACRO_MAP_CACHE = data.get("symbols", {})
        _MACRO_MAP_MTIME = mtime
        return _MACRO_MAP_CACHE
    return {}


def _ema_stack_series(df: pd.DataFrame) -> pd.Series:
    """Return EMA stack score series for df."""
    from indicators.market_structure import compute_ema_stack_score
    return compute_ema_stack_score(df).astype(float)


def _mtf_regime_features(htf_df: Optional[pd.DataFrame], base_atr: float = 1.0) -> tuple:
    """Return (adx, ema_stack, atr_ratio, bb_width) scalars for a single TF df.
    Returns (0,0,0,0) if df is None or too short.
    """
    if htf_df is None or len(htf_df) < 14:
        return 0.0, 0.0, 0.0, 0.0
    from indicators.market_structure import (
        compute_adx, compute_atr, compute_ema_stack_score, compute_bollinger_bands,
    )
    adx_s   = compute_adx(htf_df, 14)
    atr_s   = compute_atr(htf_df, 14)
    stack_s = _ema_stack_series(htf_df)
    bb_u, bb_m, bb_l = compute_bollinger_bands(htf_df["close"])
    bb_w_s  = (bb_u - bb_l) / (bb_m + 1e-9)

    adx_v   = float(adx_s.iloc[-1])   if not pd.isna(adx_s.iloc[-1])   else 0.0
    atr_v   = float(atr_s.iloc[-1])   if not pd.isna(atr_s.iloc[-1])   else 0.0
    close_v = float(htf_df["close"].iloc[-1])
    stk_v   = float(stack_s.iloc[-1]) if not pd.isna(stack_s.iloc[-1]) else 0.0
    bb_v    = float(bb_w_s.iloc[-1])  if not pd.isna(bb_w_s.iloc[-1])  else 0.0

    return (
        float(np.clip(adx_v, 0, 100)),
        float(np.clip(stk_v, -2, 2)),
        float(np.clip(atr_v / (close_v + 1e-9) * 1000, 0, 10)),
        float(np.clip(bb_v,  0, 0.1)),
    )


class FeatureEngine:
    """
    Computes all feature vectors consumed by ML models and the RL state builder.
    Inject a TradeJournal instance if strategy_win_rate_20 is needed.
    """

    def __init__(self, trade_journal=None):
        self._journal = trade_journal

    # ─── Sequence features (GRU-LSTM input) ──────────────────────────────────

    def get_sequence(
        self, df: pd.DataFrame, length: int = 30,
        df_htf: Optional[Dict[str, pd.DataFrame]] = None,
        symbol: Optional[str] = None,
    ) -> np.ndarray:
        """
        Returns shape (length, N) float32.
        GRU sequence inputs are technical-only; regime/sentiment context is combined
        later by the decision layer.
        """
        if df is None or len(df) == 0:
            raise ValueError("FeatureEngine.get_sequence: df cannot be empty")
        if len(df) < length:
            raise ValueError(
                f"FeatureEngine.get_sequence: need at least {length} bars, got {len(df)}"
            )

        feat = self._build_sequence_df(df, df_htf, symbol=symbol)
        arr = feat[SEQUENCE_FEATURES].values.astype(np.float32)

        # Replace NaN / Inf
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        return arr[-length:]

    def _build_sequence_df(
        self, df: pd.DataFrame,
        df_htf: Optional[Dict[str, pd.DataFrame]],
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """Add technical GRU sequence feature columns to a copy of df.

        df_htf: dict of {tf_key: DataFrame} with keys "5M", "1H", "4H", "1D".
        Missing HTF context raises; this model should not learn from fabricated
        zero-filled cross-timeframe features.
        """
        from indicators.market_structure import (
            compute_rsi, compute_adx, compute_bollinger_bands,
            detect_break_of_structure, detect_fair_value_gaps,
            compute_market_structure_scores,
        )

        out = df.copy(deep=False)
        required_cols = {"open", "high", "low", "close", "volume"}
        missing_cols = sorted(required_cols - set(out.columns))
        if missing_cols:
            raise ValueError(f"_build_sequence_df: missing required columns: {missing_cols}")

        atr = out.get("atr_14", compute_atr(df, 14)).astype(float)
        atr = atr.replace([np.inf, -np.inf], np.nan)
        initial_range = (out["high"] - out["low"]).abs().replace(0.0, np.nan)
        atr = atr.fillna(initial_range).ffill()
        if atr.isna().any():
            raise ValueError("_build_sequence_df: ATR warmup produced non-finite values")

        # ── Base TF features ─────────────────────────────────────────────────
        out["log_return"]    = np.log(out["close"] / (out["close"].shift(1) + 1e-9)).clip(-0.1, 0.1).fillna(0.0)
        out["high_low_range"]= (out["high"] - out["low"]) / (atr + 1e-9)
        out["close_vs_open"] = (out["close"] - out["open"]) / (atr + 1e-9)
        out["atr_normalized"]= atr / (out["close"] + 1e-9)

        if "rsi_14" not in out.columns:
            out["rsi_14"] = compute_rsi(out["close"], 14)
        out["rsi_14"] = (out["rsi_14"].fillna(50.0) - 50.0) / 50.0

        if "ema_21" not in out.columns:
            out["ema_21"] = compute_ema(out["close"], 21)
        if "ema_50" not in out.columns:
            out["ema_50"] = compute_ema(out["close"], 50)

        out["ema21_dist"] = (out["close"] - out["ema_21"]) / (atr + 1e-9)
        out["ema50_dist"] = (out["close"] - out["ema_50"]) / (atr + 1e-9)

        if "bb_upper" not in out.columns:
            bb_u, bb_m, bb_l = compute_bollinger_bands(out["close"])
            out["bb_upper"], out["bb_mid"], out["bb_lower"] = bb_u, bb_m, bb_l
        bb_range = out["bb_upper"] - out["bb_lower"]
        out["bb_position"] = (
            (out["close"] - out["bb_lower"]) / (bb_range + 1e-9)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.5)

        if hasattr(out.index, "hour"):
            hour = out.index.hour
            out["is_asian"]  = ((hour >= 0) & (hour < 7)).astype(float)
            out["is_london"] = ((hour >= 7) & (hour < 12)).astype(float)
            out["is_ny"]     = ((hour >= 13) & (hour < 18)).astype(float)
        else:
            out["is_asian"] = out["is_london"] = out["is_ny"] = 0.0

        if "bos_bull" not in out.columns:
            bos = detect_break_of_structure(df)
            out["bos_bull"] = bos["bos_bull"]
            out["bos_bear"] = bos["bos_bear"]
        out["bos_bull_flag"] = out["bos_bull"].astype(float)
        out["bos_bear_flag"] = out["bos_bear"].astype(float)

        if "fvg_bull" not in out.columns:
            fvg = detect_fair_value_gaps(df)
            out["fvg_bull"]        = fvg["fvg_bull"]
            out["fvg_bear"]        = fvg["fvg_bear"]
            out["fvg_bull_top"]    = fvg["fvg_bull_top"]
            out["fvg_bull_bottom"] = fvg["fvg_bull_bottom"]
            out["fvg_bear_top"]    = fvg["fvg_bear_top"]
            out["fvg_bear_bottom"] = fvg["fvg_bear_bottom"]
        out["fvg_bull_open"] = out["fvg_bull"].astype(float)
        out["fvg_bear_open"] = out["fvg_bear"].astype(float)

        structure_cols = {
            "mss_bull", "mss_bear", "mss_bull_flag", "mss_bear_flag",
            "mss_bull_bars_ago", "mss_bear_bars_ago", "bars_since_mss",
            "external_trend_direction", "external_structure_score",
            "internal_structure_state", "swing_sequence_score",
            "position_in_external_range", "dist_to_external_high_atr",
            "dist_to_external_low_atr",
        }
        if not structure_cols.issubset(out.columns):
            structure = compute_market_structure_scores(out)
            for col in structure_cols:
                out[col] = structure[col]
        out["mss_bull"] = out["mss_bull"].fillna(False).astype(bool)
        out["mss_bear"] = out["mss_bear"].fillna(False).astype(bool)
        out["mss_bull_flag"] = out["mss_bull_flag"].fillna(0.0).astype(float)
        out["mss_bear_flag"] = out["mss_bear_flag"].fillna(0.0).astype(float)
        _structure_defaults = {
            "mss_bull_bars_ago": 1.0,
            "mss_bear_bars_ago": 1.0,
            "bars_since_mss": 1.0,
            "position_in_external_range": 0.5,
        }
        for col in (
            "mss_bull_bars_ago", "mss_bear_bars_ago", "bars_since_mss",
            "external_trend_direction", "external_structure_score",
            "internal_structure_state", "swing_sequence_score",
            "position_in_external_range", "dist_to_external_high_atr",
            "dist_to_external_low_atr",
        ):
            out[col] = pd.to_numeric(out[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).fillna(_structure_defaults.get(col, 0.0)).astype(float)

        # ── Helper: reindex a required HTF series onto base df index ──────────
        def _htf_series(
            htf_df: Optional[pd.DataFrame],
            tf_name: str,
            compute_fn,
            default_value: float = 0.0,
        ) -> pd.Series:
            """Compute an HTF series and align it to the base index without lookahead."""
            if htf_df is None:
                raise ValueError(f"_build_sequence_df: missing required HTF frame {tf_name}")
            if len(htf_df) < 14:
                raise ValueError(
                    f"_build_sequence_df: HTF frame {tf_name} has {len(htf_df)} rows, need >= 14"
                )
            htf_missing = sorted(required_cols - set(htf_df.columns))
            if htf_missing:
                raise ValueError(
                    f"_build_sequence_df: HTF frame {tf_name} missing columns: {htf_missing}"
                )
            htf_df = htf_df.sort_index()
            if htf_df.index.has_duplicates:
                htf_df = htf_df[~htf_df.index.duplicated(keep="last")]
            s = pd.to_numeric(compute_fn(htf_df), errors="coerce").replace([np.inf, -np.inf], np.nan)
            s = s.sort_index().ffill()
            aligned = s.reindex(out.index, method="ffill")
            missing = aligned.isna()
            if missing.any():
                # Leading base bars can precede the first finite HTF value after
                # calendar slicing or indicator warmup. Fill those with the
                # neutral value for that feature; never backfill from the future.
                logger.warning(
                    "_build_sequence_df: HTF frame %s filled %d warmup/alignment gaps with %.3f",
                    tf_name,
                    int(missing.sum()),
                    default_value,
                )
                aligned = aligned.fillna(float(default_value))
            if not np.isfinite(aligned.to_numpy(dtype=np.float64, copy=False)).all():
                raise ValueError(
                    f"_build_sequence_df: HTF frame {tf_name} has non-finite values after alignment"
                )
            return aligned.astype(float)

        def _safe_htf_atr(htf_df: pd.DataFrame) -> pd.Series:
            htf_atr = compute_atr(htf_df, 14).replace([np.inf, -np.inf], np.nan)
            htf_range = (htf_df["high"] - htf_df["low"]).abs().replace(0.0, np.nan)
            htf_atr = htf_atr.fillna(htf_range).ffill()
            first_valid = htf_atr.first_valid_index()
            if first_valid is None:
                raise ValueError("_build_sequence_df: HTF ATR warmup produced no finite values")
            if htf_atr.loc[first_valid:].isna().any():
                raise ValueError("_build_sequence_df: HTF ATR contains non-finite values after warmup")
            return htf_atr

        htf = df_htf if isinstance(df_htf, dict) else {}

        df_5m = _first_present_frame(htf, "5M", "5m")
        df_1h = _first_present_frame(htf, "1H", "H1")
        df_4h = _first_present_frame(htf, "4H", "H4")
        df_1d = _first_present_frame(htf, "1D", "D1")

        # ── 5M cross-TF ──────────────────────────────────────────────────────
        out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
            lambda d: (compute_rsi(d["close"], 14).fillna(50.0) - 50.0) / 50.0)
        out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
            lambda d: (d["close"] - compute_ema(d["close"], 21)) / (_safe_htf_atr(d) + 1e-9))

        # ── 1H cross-TF ──────────────────────────────────────────────────────
        out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
            lambda d: compute_adx(d, 14).fillna(0.0) / 100.0)
        out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
            lambda d: (d["close"] - compute_ema(d["close"], 21)) / (_safe_htf_atr(d) + 1e-9))
        out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
            lambda d: (d["close"] - compute_ema(d["close"], 50)) / (_safe_htf_atr(d) + 1e-9))

        # ── 4H cross-TF ──────────────────────────────────────────────────────
        out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
            lambda d: (compute_ema(d["close"], 21) - compute_ema(d["close"], 50)) / (d["close"] + 1e-9))
        out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
            lambda d: compute_adx(d, 14).fillna(0.0) / 100.0)
        out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
            lambda d: (compute_rsi(d["close"], 14).fillna(50.0) - 50.0) / 50.0)

        # ── 1D cross-TF ──────────────────────────────────────────────────────
        out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
            lambda d: (d["close"] - compute_ema(d["close"], 21)) / (_safe_htf_atr(d) + 1e-9))
        out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
            lambda d: _ema_stack_series(d) / 2.0)

        # ── Late additions — all collected into `extra`, concat'd once ──────────
        from indicators.market_structure import compute_adx as _compute_adx
        extra: dict[str, np.ndarray] = {}
        n = len(out)
        _close = out["close"].to_numpy(dtype=np.float64)
        _high  = out["high"].to_numpy(dtype=np.float64)
        _low   = out["low"].to_numpy(dtype=np.float64)
        _open  = out["open"].to_numpy(dtype=np.float64)
        _atr   = atr.to_numpy(dtype=np.float64)

        # ── Volatility dynamics ───────────────────────────────────────────────
        _rel_vol = atr / (out["close"] + 1e-9)
        extra["vol_slope_seq"] = np.clip(
            _rel_vol.diff(14).fillna(0.0).to_numpy() * 1000, -5.0, 5.0
        ).astype(np.float32)

        # ── Cyclic time encoding ──────────────────────────────────────────────
        if hasattr(out.index, "hour"):
            _hour = out.index.hour.astype(np.float32)
            extra["time_sin"] = np.sin(2.0 * np.pi * _hour / 24.0).astype(np.float32)
            extra["time_cos"] = np.cos(2.0 * np.pi * _hour / 24.0).astype(np.float32)
        else:
            extra["time_sin"] = np.zeros(n, dtype=np.float32)
            extra["time_cos"] = np.zeros(n, dtype=np.float32)

        # ── EMA structure features ────────────────────────────────────────────
        _ema21 = out["ema_21"].to_numpy(dtype=np.float64)
        _ema50 = out["ema_50"].to_numpy(dtype=np.float64)
        # Price in EMA21-50 band: positive if inside (above EMA21 for bull), clamped
        _band_mid  = (_ema21 + _ema50) * 0.5
        _band_half = np.abs(_ema21 - _ema50) * 0.5 + 1e-9
        extra["ema_pullback_zone"] = np.clip(
            (_close - _band_mid) / (_band_half + _atr * 0.5 + 1e-9), -2.0, 2.0
        ).astype(np.float32)
        # EMA21 slope on 15M: (EMA21[t] - EMA21[t-5]) / ATR
        _ema21_s = pd.Series(_ema21, index=out.index)
        extra["ema21_slope_15m"] = np.clip(
            (_ema21_s - _ema21_s.shift(5)).fillna(0.0).to_numpy() / (_atr + 1e-9),
            -5.0, 5.0
        ).astype(np.float32)
        # 1H EMA21 slope (forward-filled to 15M)
        extra["ema21_slope_1h"] = _htf_series(df_1h, "1H",
            lambda d: (compute_ema(d["close"], 21) - compute_ema(d["close"], 21).shift(3)).fillna(0.0)
                      / (_safe_htf_atr(d) + 1e-9)
        ).clip(-5.0, 5.0).to_numpy(dtype=np.float32)
        # 15M EMA stack score
        from indicators.market_structure import compute_ema_stack_score as _ema_stack_fn
        extra["ema_stack_15m"] = np.clip(
            _ema_stack_fn(out).fillna(0.0).to_numpy() / 2.0, -1.0, 1.0
        ).astype(np.float32)

        # Higher-high/higher-low and lower-high/lower-low directional structure.
        _structure_window = 20
        _high_s = pd.Series(_high, index=out.index)
        _low_s = pd.Series(_low, index=out.index)
        _hh = _high_s > _high_s.shift(1).rolling(
            _structure_window, min_periods=max(3, _structure_window // 2)
        ).max()
        _hl = _low_s > _low_s.shift(_structure_window // 2).fillna(_low_s)
        _ll = _low_s < _low_s.shift(1).rolling(
            _structure_window, min_periods=max(3, _structure_window // 2)
        ).min()
        _lh = _high_s < _high_s.shift(_structure_window // 2).fillna(_high_s)
        _signed_structure = (
            _hh.astype(float) + _hl.astype(float) - _ll.astype(float) - _lh.astype(float)
        ).rolling(_structure_window, min_periods=1).mean().clip(-1.0, 1.0)
        extra["hh_hl_structure"] = _signed_structure.clip(lower=0.0, upper=1.0).to_numpy(dtype=np.float32)
        extra["lh_ll_structure"] = (-_signed_structure).clip(lower=0.0, upper=1.0).to_numpy(dtype=np.float32)

        # ── BOS age + strength ────────────────────────────────────────────────
        # bos_bull/bos_bear already computed above in out["bos_bull"], out["bos_bear"]
        _bos_bull = out["bos_bull"].to_numpy(dtype=bool)
        _bos_bear = out["bos_bear"].to_numpy(dtype=bool)
        _cap = 40.0
        # Vectorized "bars since last event" + "strength at that event bar"
        idx = np.arange(n, dtype=np.float32)
        _atr_safe = np.where(_atr > 0, _atr, 1e-9)
        _bull_move = np.where(_bos_bull, np.maximum(_close - _open, 0.0), 0.0)
        _bear_move = np.where(_bos_bear, np.maximum(_open - _close, 0.0), 0.0)

        def _last_event_arrays(mask: np.ndarray, move_vals: np.ndarray, fill_idx: int):
            # Fully vectorized "last event bar + value" — no Python loop.
            # event_idx[i] = index of the most recent True in mask[0..i].
            event_positions = np.where(mask)[0]
            bar_range = np.arange(n, dtype=np.float32)
            if len(event_positions) == 0:
                return np.full(n, float(fill_idx), dtype=np.float32), np.zeros(n, dtype=np.float32)
            # For each bar, searchsorted gives the index into event_positions of the
            # last event at or before that bar — O(N log E) fully in C.
            grp = np.searchsorted(event_positions, bar_range, side="right") - 1
            # Bars before the first event get fill_idx / 0.0
            last_bar  = np.where(grp >= 0, event_positions[np.clip(grp, 0, len(event_positions) - 1)].astype(np.float32), float(fill_idx))
            last_move = np.where(grp >= 0, move_vals[np.clip(grp, 0, len(event_positions) - 1)], 0.0).astype(np.float32)
            return last_bar, last_move

        _bull_last_bar, _bull_last_move = _last_event_arrays(_bos_bull, _bull_move, -int(_cap))
        _bear_last_bar, _bear_last_move = _last_event_arrays(_bos_bear, _bear_move, -int(_cap))
        bos_bull_ago = np.clip((idx - _bull_last_bar) / _cap, 0.0, 1.0).astype(np.float32)
        bos_bear_ago = np.clip((idx - _bear_last_bar) / _cap, 0.0, 1.0).astype(np.float32)
        bos_bull_str = np.clip(_bull_last_move / _atr_safe, 0.0, 5.0).astype(np.float32)
        bos_bear_str = np.clip(_bear_last_move / _atr_safe, 0.0, 5.0).astype(np.float32)
        extra["bos_bull_bars_ago"]  = bos_bull_ago
        extra["bos_bear_bars_ago"]  = bos_bear_ago
        extra["bos_bull_strength"]  = bos_bull_str
        extra["bos_bear_strength"]  = bos_bear_str

        # ── MSS / CHoCH + external structure ─────────────────────────────────
        # compute_market_structure_scores may have already run early in this
        # method (lines ~658-681) to populate out; reuse those columns when
        # present so we don't recompute the swing arrays a second time.
        _mss_feature_cols = (
            "mss_bull_flag", "mss_bear_flag",
            "mss_bull_bars_ago", "mss_bear_bars_ago",
            "external_structure_score", "position_in_external_range",
            "dist_to_external_high_atr", "dist_to_external_low_atr",
        )
        if all(c in out.columns for c in _mss_feature_cols):
            _mss_src = out
        else:
            from indicators.market_structure import compute_market_structure_scores as _mss_scores_fn
            _mss_src = _mss_scores_fn(out)
        for _mc in _mss_feature_cols:
            extra[_mc] = _mss_src[_mc].to_numpy(dtype=np.float32)

        # ── FVG distance + fill ratio ─────────────────────────────────────────
        # fvg_bull_top/bottom carry the FVG boundaries at the bar it formed; forward-fill
        # to get the most recent open FVG levels. Reset when fvg_bear forms (directional flip).
        _fvg_bull_top    = out.get("fvg_bull_top",    pd.Series(np.nan, index=out.index)).to_numpy(dtype=np.float64)
        _fvg_bull_bottom = out.get("fvg_bull_bottom", pd.Series(np.nan, index=out.index)).to_numpy(dtype=np.float64)
        _fvg_bear_top    = out.get("fvg_bear_top",    pd.Series(np.nan, index=out.index)).to_numpy(dtype=np.float64)
        _fvg_bear_bottom = out.get("fvg_bear_bottom", pd.Series(np.nan, index=out.index)).to_numpy(dtype=np.float64)
        # Forward-fill FVG boundaries then compute distances vectorized
        _atr_safe2 = np.where(_atr > 1e-12, _atr, 1e-9)
        _bt = pd.Series(_fvg_bull_top,    index=out.index).ffill().to_numpy(dtype=np.float64)
        _bb = pd.Series(_fvg_bull_bottom, index=out.index).ffill().to_numpy(dtype=np.float64)
        _ft = pd.Series(_fvg_bear_top,    index=out.index).ffill().to_numpy(dtype=np.float64)
        _fb = pd.Series(_fvg_bear_bottom, index=out.index).ffill().to_numpy(dtype=np.float64)
        _bull_valid = ~np.isnan(_bt)
        _bear_valid = ~np.isnan(_ft)
        _bull_gap = np.where(_bull_valid, _bt - _bb, 1.0)
        _bear_gap = np.where(_bear_valid, _ft - _fb, 1.0)
        fvg_bull_dist = np.where(_bull_valid,
            np.clip((_close - _bt) / _atr_safe2, -5.0, 5.0), 0.0).astype(np.float32)
        fvg_bull_fill = np.where(_bull_valid & (_bull_gap > 1e-9),
            np.clip((_close - _bb) / (_bull_gap + 1e-9), 0.0, 1.0), 0.0).astype(np.float32)
        fvg_bear_dist = np.where(_bear_valid,
            np.clip((_fb - _close) / _atr_safe2, -5.0, 5.0), 0.0).astype(np.float32)
        fvg_bear_fill = np.where(_bear_valid & (_bear_gap > 1e-9),
            np.clip((_ft - _close) / (_bear_gap + 1e-9), 0.0, 1.0), 0.0).astype(np.float32)
        extra["fvg_bull_dist_atr"]   = fvg_bull_dist
        extra["fvg_bear_dist_atr"]   = fvg_bear_dist
        extra["fvg_bull_fill_ratio"] = fvg_bull_fill
        extra["fvg_bear_fill_ratio"] = fvg_bear_fill

        # ── Sweep / liquidity ─────────────────────────────────────────────────
        from indicators.market_structure import detect_liquidity_sweeps as _det_sweeps
        _sweeps = _det_sweeps(out)
        _sw_bull_wick = _sweeps["sweep_bull_wick"].fillna(0.0).to_numpy(dtype=np.float64)
        _sw_bear_wick = _sweeps["sweep_bear_wick"].fillna(0.0).to_numpy(dtype=np.float64)
        _sw_bull = _sweeps["sweep_bull"].to_numpy(dtype=bool)
        _sw_bear = _sweeps["sweep_bear"].to_numpy(dtype=bool)
        _hl = _high - _low
        _body = np.abs(_close - _open)
        # Vectorized: for each bar take the nearest sweep signal within last 3 bars.
        # Build arrays of wick/body at sweep bars, forward-fill with a 3-bar window cap.
        _atr_safe3 = np.where(_atr > 1e-12, _atr, 1e-9)
        _sweep_any  = _sw_bull | _sw_bear
        _wick_at    = np.where(_sw_bull, _sw_bull_wick, _sw_bear_wick)
        _body_at    = np.where(_hl + 1e-9 > 0, _body / (_hl + 1e-9), 0.0)
        # Forward-fill then zero out values older than 3 bars
        _wick_ff   = pd.Series(_wick_at, index=out.index).where(_sweep_any).ffill(limit=3).fillna(0.0).to_numpy(dtype=np.float64)
        _bodyf_ff  = pd.Series(_body_at, index=out.index).where(_sweep_any).ffill(limit=3).fillna(0.0).to_numpy(dtype=np.float64)
        sweep_wick = np.clip(_wick_ff / _atr_safe3, 0.0, 5.0).astype(np.float32)
        body_rec   = np.clip(_bodyf_ff, 0.0, 1.0).astype(np.float32)
        extra["sweep_wick_depth_atr"] = sweep_wick
        extra["body_recovery_ratio"]  = body_rec

        # ── Liquidity proximity ───────────────────────────────────────────────
        _high20 = pd.Series(_high, index=out.index).rolling(20, min_periods=1).max().to_numpy(dtype=np.float64)
        _low20  = pd.Series(_low,  index=out.index).rolling(20, min_periods=1).min().to_numpy(dtype=np.float64)
        extra["dist_to_recent_high_atr"] = np.clip(
            (_high20 - _close) / (_atr + 1e-9), 0.0, 10.0
        ).astype(np.float32)
        extra["dist_to_recent_low_atr"]  = np.clip(
            (_close - _low20) / (_atr + 1e-9), 0.0, 10.0
        ).astype(np.float32)

        # ── Asian range context ───────────────────────────────────────────────
        # Asian session: 01:00–05:00 UTC. Per bar, look back to find session high/low.
        asian_high_arr = np.full(n, np.nan, dtype=np.float64)
        asian_low_arr  = np.full(n, np.nan, dtype=np.float64)
        if hasattr(out.index, "hour") and hasattr(out.index, "minute"):
            _ts_minutes = out.index.hour * 60 + out.index.minute
            _is_asian_window = (_ts_minutes >= 60) & (_ts_minutes < 300)  # 01:00–05:00
            # Fully vectorized per-day cummax/cummin — no groupby, no Python loop.
            # Strategy: assign Asian-window values; mark day-start bars so we can
            # "reset" running max/min at each boundary using cumsum group IDs.
            _idx      = out.index
            _day_norm = _idx.normalize().astype(np.int64)   # ns, changes at UTC midnight
            _day_chg  = np.empty(n, dtype=bool)
            _day_chg[0] = True
            _day_chg[1:] = _day_norm[1:] != _day_norm[:-1]
            _group_id = np.cumsum(_day_chg)                 # integer day group per bar
            # Within each group, running max/min of Asian-only values.
            _ah_vals = np.where(_is_asian_window, _high, np.nan)
            _al_vals = np.where(_is_asian_window, _low,  np.nan)
            # Use pandas groupby on the pre-computed integer group — single C-level pass.
            _s_ah = pd.Series(_ah_vals)
            _s_al = pd.Series(_al_vals)
            _grp  = pd.Series(_group_id)
            asian_high_arr = (
                _s_ah.groupby(_grp).transform(lambda g: g.expanding().max())
            ).to_numpy(dtype=np.float64)
            asian_low_arr = (
                _s_al.groupby(_grp).transform(lambda g: g.expanding().min())
            ).to_numpy(dtype=np.float64)
        _asian_range = np.where(
            ~np.isnan(asian_high_arr) & ~np.isnan(asian_low_arr),
            asian_high_arr - asian_low_arr, 0.0
        )
        extra["asian_range_width_atr"] = np.clip(
            _asian_range / (_atr + 1e-9), 0.0, 10.0
        ).astype(np.float32)
        extra["price_vs_asian_high_atr"] = np.where(
            ~np.isnan(asian_high_arr),
            np.clip((_close - asian_high_arr) / (_atr + 1e-9), -5.0, 5.0), 0.0
        ).astype(np.float32)
        extra["price_vs_asian_low_atr"] = np.where(
            ~np.isnan(asian_low_arr),
            np.clip((_close - asian_low_arr) / (_atr + 1e-9), -5.0, 5.0), 0.0
        ).astype(np.float32)

        # ── Candle structure ──────────────────────────────────────────────────
        _range = _high - _low + 1e-9
        extra["candle_body_ratio"] = np.clip(_body / _range, 0.0, 1.0).astype(np.float32)
        extra["upper_wick_ratio"]  = np.clip(
            (_high - np.maximum(_close, _open)) / _range, 0.0, 1.0
        ).astype(np.float32)
        extra["lower_wick_ratio"]  = np.clip(
            (np.minimum(_close, _open) - _low) / _range, 0.0, 1.0
        ).astype(np.float32)

        # ── ADX on 15M ────────────────────────────────────────────────────────
        extra["adx_15m"] = np.clip(
            _compute_adx(out, 14).fillna(0.0).to_numpy(dtype=np.float64) / 100.0, 0.0, 1.0
        ).astype(np.float32)

        # ── VWAP / wick auction structure ─────────────────────────────────────
        from indicators.market_structure import (
            compute_vwap as _vwap_fn,
            compute_wick_ratio as _wr_fn,
        )
        _vwap_df = _vwap_fn(out)
        extra["vwap_dist_atr"] = np.clip(
            _vwap_df["vwap_dist_atr"].fillna(0.0).to_numpy(dtype=np.float64), -3.0, 3.0
        ).astype(np.float32)

        _wr_df = _wr_fn(out)
        extra["wick_auction_ratio"] = (
            _wr_df["wick_auction_ratio"].fillna(0.5).to_numpy(dtype=np.float32)
        )

        # ── Volatility expansion ──────────────────────────────────────────────
        _atr_s = pd.Series(_atr, index=out.index)
        _atr_lag10 = _atr_s.shift(10).replace(0, np.nan).bfill().fillna(1e-9)
        extra["vol_expansion"] = np.clip(
            (_atr_s / _atr_lag10).fillna(1.0).to_numpy(dtype=np.float64), 0.5, 3.0
        ).astype(np.float32)

        # ── ATR percentile rank (consolidation signal) ────────────────────────
        # Low value = ATR at multi-period low = compression / consolidation building.
        # High value = ATR expanding = breakout / volatility regime.
        # Window: 42 bars (≈ 10.5 hours at 15M, ≈ 1 week at 4H).
        extra["atr_pctile"] = _vec_atr_pctile(_atr_s.to_numpy(dtype=np.float64), window=42, min_periods=14)

        # ── Session timing — continuous ───────────────────────────────────────
        if hasattr(out.index, "hour") and hasattr(out.index, "minute"):
            _mins_in_day = out.index.hour * 60 + out.index.minute
            # London: 07:00 = 420 min, window 8h = 480 min
            _london_mins = (_mins_in_day - 420).astype(np.float32)
            extra["mins_since_london_open"] = np.where(
                _london_mins >= 0, np.clip(_london_mins / 480.0, 0.0, 1.0), -1.0
            ).astype(np.float32)
            # NY: 13:00 = 780 min, window 5h = 300 min
            _ny_mins = (_mins_in_day - 780).astype(np.float32)
            extra["mins_since_ny_open"] = np.where(
                _ny_mins >= 0, np.clip(_ny_mins / 300.0, 0.0, 1.0), -1.0
            ).astype(np.float32)
        else:
            extra["mins_since_london_open"] = np.full(n, -1.0, dtype=np.float32)
            extra["mins_since_ny_open"]     = np.full(n, -1.0, dtype=np.float32)

        out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)
        # Drop duplicate columns — input df may already have pre-computed indicators.
        # Keep last occurrence so that the normalised values computed above always win.
        if out.columns.duplicated().any():
            out = out.loc[:, ~out.columns.duplicated(keep="last")]
        missing_features = [name for name in SEQUENCE_FEATURES if name not in out.columns]
        if missing_features:
            raise ValueError(f"_build_sequence_df: missing sequence features: {missing_features}")
        seq_values = out[SEQUENCE_FEATURES].to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(seq_values).all():
            bad_mask = ~np.isfinite(seq_values)
            bad_cols = [SEQUENCE_FEATURES[i] for i in np.where(bad_mask.any(axis=0))[0]]
            raise ValueError(f"_build_sequence_df: non-finite sequence features: {bad_cols}")
        return out

    def _macro_mask(self, symbol: Optional[str]) -> set[str]:
        if os.getenv("MACRO_USE_ALL_INDICES", "true").lower() == "true":
            return set()
        if not symbol:
            return set()
        macro_map = _load_macro_map()
        if not macro_map or symbol not in macro_map:
            return set()
        return set(macro_map.get(symbol, {}).get("selected", []))

    def _build_macro_frame(self, index: pd.Index, symbol: Optional[str]) -> pd.DataFrame:
        # Cache keyed by (symbol, first_ts, last_ts, len) — reindexing 19 daily series
        # onto a 300k-row 15M index takes seconds; skip if we've done it already.
        _cache_key = (
            symbol,
            index[0] if len(index) else None,
            index[-1] if len(index) else None,
            len(index),
        )
        if _cache_key in _MACRO_ALIGNED_CACHE:
            return _MACRO_ALIGNED_CACHE[_cache_key]

        macro = _load_macro_cache()
        mask = self._macro_mask(symbol)

        def _align(series: Optional[pd.Series]) -> pd.Series:
            if series is None or len(series) == 0:
                return pd.Series(0.0, index=index)
            # ffill only — bfill would pull future values back into earlier bars (lookahead)
            s = series.reindex(index, method="ffill").fillna(0.0)
            return s

        us10y = _align(macro.get("us10y_fred"))
        us2y = _align(macro.get("us2y_fred"))
        yield_spread = (us10y - us2y).clip(-2.0, 4.0) / 10.0

        # Build index returns for every index in the directory
        data = {}
        for name in INDEX_NAMES:
            series = _align(macro.get(name))
            if name in {"us10y", "us30y", "us3m"}:
                ret = series.diff().fillna(0.0)
            else:
                ret = series.pct_change().fillna(0.0)
            ret = ret.clip(-0.05, 0.05)
            key = f"idx_{name}_ret"
            if not mask or name in mask:
                data[key] = ret
            else:
                data[key] = pd.Series(0.0, index=index)

        vix = _align(macro.get("vix"))
        vix_level = (vix / 50.0).clip(0.0, 2.0)

        data["macro_vix_level"] = vix_level
        data["macro_yield_spread"] = yield_spread

        result_df = pd.DataFrame(data, index=index)
        # Store in aligned cache — cap at 30 entries to avoid unbounded RAM growth
        if len(_MACRO_ALIGNED_CACHE) < 30:
            _MACRO_ALIGNED_CACHE[_cache_key] = result_df
        return result_df

    def get_macro_snapshot(self, symbol: str, timestamp: pd.Timestamp) -> Dict[str, float]:
        try:
            idx = pd.DatetimeIndex([timestamp])
            macro_df = self._build_macro_frame(idx, symbol)
            row = macro_df.iloc[-1]
            return {k: float(row[k]) for k in MACRO_FEATURES}
        except Exception:
            return {k: 0.0 for k in MACRO_FEATURES}

    # ─── Retired regime feature API ──────────────────────────────────────────

    def get_regime_features(
        self,
        df: pd.DataFrame,
        df_htf: Optional[Dict[str, pd.DataFrame]] = None,
        df_h4: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
    ) -> np.ndarray:
        """Retired: regime models now require explicit 4H or 1H contracts."""
        raise RuntimeError(
            "FeatureEngine.get_regime_features is retired. Use "
            "RegimeClassifier._build_feature_matrix(..., feature_names=REGIME_4H_FEATURES) "
            "or REGIME_1H_FEATURES."
        )

    # ─── Quality features (XGBoost input) ────────────────────────────────────

    def get_quality_features(self, signal: dict, ml_base: dict, bar: pd.Series) -> np.ndarray:
        """
        Returns shape (N,) float32 where N = len(QUALITY_FEATURES).
        QualityScorer scores a concrete candidate trade. It consumes the side,
        planned RR, side-conditioned GRU edge, compact regime context, execution
        friction, and prior strategy outcomes. It intentionally does not repeat
        macro/sentiment placeholders or raw flat regime classes.
        """
        feats = np.zeros(len(QUALITY_FEATURES), dtype=np.float32)

        side = signal.get("side", "")
        regime_scores = _quality_regime_scores(ml_base)
        p_win, p_loss = _quality_side_probabilities(ml_base, side)
        expected_move = _quality_float(
            ml_base.get("expected_move", ml_base.get("entry_depth", 0.0)),
            0.0,
        )

        _rr_val = float(np.clip(_quality_float(signal.get("rr_ratio", 1.5), 1.5), 0.0, 10.0))
        feats[0] = _quality_strategy_code(signal.get("trader_id", ""))
        feats[1] = 1.0 if str(side).lower() == "buy" else 0.0
        feats[2] = _rr_val
        feats[3] = float(np.clip(p_win, 0.0, 1.0))
        feats[4] = float(np.clip(p_win - p_loss, -1.0, 1.0))
        feats[5] = float(np.clip(expected_move, 0.0, 1.0))
        feats[6] = float(np.clip(_quality_float(ml_base.get("expected_variance", 0.1), 0.1), 0.0, 5.0))
        feats[7] = _quality_trade_regime_code(ml_base.get("trade_regime", ""), regime_scores)
        feats[8] = _quality_expected_r_gross(p_win, _rr_val)   # p_win×rr − (1-p_win)×1
        feats[9] = float(np.clip(_quality_float(
            regime_scores.get("volatility_percentile", ml_base.get("volatility_percentile", 0.5)),
            0.5,
        ), 0.0, 1.0))
        feats[10] = float(np.clip(_quality_float(
            regime_scores.get("chop_score", ml_base.get("chop_score", 0.0)),
            0.0,
        ), 0.0, 1.0))

        feats[11] = float(np.clip(bar.get("adx_14", 20.0) if hasattr(bar, "get") else 20.0, 0.0, 100.0))
        feats[12] = float(np.clip(
            bar.get("atr_14", 0.001) / (bar.get("close", 1.0) + 1e-9) * 1000
            if hasattr(bar, "get") else 1.0, 0.0, 20.0
        ))
        feats[13] = float(np.clip(_quality_float(
            ml_base.get("spread_pips", ml_base.get("spread_at_signal", 1.0)),
            1.0,
        ), 0.0, 20.0))
        feats[14] = _quality_session_code(ml_base.get("session", "INACTIVE"))
        feats[15] = 1.0 if ml_base.get("news_in_30min", False) else 0.0

        win_rate_20 = 0.5
        win_rate_5  = 0.5
        win_rate_50 = 0.5
        if self._journal is not None:
            tid = signal.get("trader_id", "")
            stats_20 = self._journal.get_rolling_stats(tid, n=20)
            stats_5  = self._journal.get_rolling_stats(tid, n=5)
            stats_50 = self._journal.get_rolling_stats(tid, n=50)
            win_rate_20 = stats_20.get("win_rate", 0.5)
            win_rate_5  = stats_5.get("win_rate",  0.5)
            win_rate_50 = stats_50.get("win_rate", 0.5)
        feats[16] = float(np.clip(win_rate_5, 0.0, 1.0))
        feats[17] = float(np.clip(win_rate_20, 0.0, 1.0))
        feats[18] = float(np.clip(win_rate_50, 0.0, 1.0))
        feats[19] = float(np.clip(_quality_float(
            ml_base.get("vol_slope", ml_base.get("vol_slope_at_signal", 0.0)),
            0.0,
        ), -5.0, 5.0))

        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats.astype(np.float32)

    # ─── RL state vector ──────────────────────────────────────────────────────

    def get_rl_state(
        self,
        bar: Any,
        portfolio: dict,
        signals: dict,
        ml_preds: dict,
        symbol: str = "",
    ) -> np.ndarray:
        """
        Returns the canonical 43-dim RL state vector, clamped to [-10, 10].

        RL should tune threshold/risk from aligned context. It should not be a
        second quality model, a sentiment model, or a duplicate flat-regime
        classifier.
        """
        portfolio = portfolio or {}
        signals = signals or {}
        ml_preds = ml_preds or {}
        state = np.zeros(RL_STATE_DIM, dtype=np.float32)

        def _safe(d: dict, key: str, default: float = 0.0) -> float:
            try:
                v = d.get(key, default)
                return float(v) if v is not None else default
            except Exception:
                return default

        def _flag(key: str) -> float:
            return 1.0 if _bar(key, 0.0) > 0.0 else 0.0

        def _probabilities() -> tuple[list[float], float]:
            raw = ml_preds.get("regime_proba")
            if raw is not None:
                vals = list(raw)[:3]
                vals += [0.0] * (3 - len(vals))
                vals = [float(np.clip(_quality_float(v, 0.0), 0.0, 1.0)) for v in vals]
                return vals, max(vals)
            regime = str(ml_preds.get("regime", "BIAS_NEUTRAL")).upper()
            if regime == "BIAS_UP":
                return [1.0, 0.0, 0.0], _safe(ml_preds, "regime_conf", 1.0)
            if regime == "BIAS_DOWN":
                return [0.0, 1.0, 0.0], _safe(ml_preds, "regime_conf", 1.0)
            return [0.0, 0.0, 1.0], _safe(ml_preds, "regime_conf", 1.0)

        def _ltf_confidence(scores: dict) -> float:
            raw = ml_preds.get("regime_ltf_conf")
            if raw is None:
                return float(np.clip(max([
                    _quality_float(scores.get("trend_score", 0.0), 0.0),
                    _quality_float(scores.get("range_score", 0.0), 0.0),
                    _quality_float(scores.get("chop_score", 0.0), 0.0),
                    _quality_float(scores.get("consolidation_score", 0.0), 0.0),
                ]), 0.0, 1.0))
            if isinstance(raw, (list, tuple, np.ndarray)):
                vals = [_quality_float(v, 0.0) for v in list(raw)]
                return float(np.clip(max(vals) if vals else 0.0, 0.0, 1.0))
            return float(np.clip(_quality_float(raw, 0.0), 0.0, 1.0))

        def _bar(key: str, default: float = 0.0) -> float:
            try:
                if hasattr(bar, "get"):
                    v = bar.get(key, default)
                elif hasattr(bar, key):
                    v = getattr(bar, key)
                else:
                    return default
                return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else default
            except Exception:
                return default

        htf_p, htf_conf = _probabilities()
        regime_scores = _quality_regime_scores(ml_preds)
        trend_score = _quality_float(
            regime_scores.get("trend_score", ml_preds.get("trend_score", 0.0)),
            0.0,
        )
        range_score = _quality_float(
            regime_scores.get("range_score", ml_preds.get("range_score", 0.0)),
            0.0,
        )
        chop_score = _quality_float(
            regime_scores.get("chop_score", ml_preds.get("chop_score", 0.0)),
            0.0,
        )
        vol_pct = _quality_float(
            regime_scores.get("volatility_percentile", ml_preds.get("volatility_percentile", 0.5)),
            0.5,
        )
        consolidation_score = _quality_float(
            regime_scores.get("consolidation_score", ml_preds.get("consolidation_score", 0.0)),
            0.0,
        )
        trade_code = _quality_trade_regime_code(ml_preds.get("trade_regime", ""), regime_scores)

        p_bull = float(np.clip(_safe(ml_preds, "p_bull", 0.5), 0.0, 1.0))
        p_bear = float(np.clip(_safe(ml_preds, "p_bear", 0.5), 0.0, 1.0))
        expected_move = _safe(ml_preds, "expected_move", _safe(ml_preds, "entry_depth", 0.0))
        close = _bar("close", 1.0)
        atr = _bar("atr_14", 0.001)
        atr_ratio = float(np.clip(atr / (close + 1e-9) * 1000.0, 0.0, 10.0))

        if hasattr(bar, "name") and hasattr(bar.name, "hour"):
            h = bar.name.hour
        else:
            h = _quality_float(ml_preds.get("hour", 0), 0.0)
        session = str(ml_preds.get("session") or signals.get("session") or "").upper()
        if not session:
            if 2 <= h < 7:
                session = "ASIAN"
            elif 7 <= h < 12:
                session = "LONDON"
            elif 13 <= h < 18:
                session = "NY"
            else:
                session = "INACTIVE"

        # [0-5] GRU execution forecast
        state[0] = p_bull
        state[1] = p_bear
        state[2] = float(np.clip(max(p_bull, p_bear) * 2.0 - 1.0, -1.0, 1.0))
        state[3] = float(np.clip(_safe(ml_preds, "entry_depth", 0.0), 0.0, 1.0))
        state[4] = float(np.clip(expected_move, 0.0, 1.0))
        state[5] = float(np.clip(_safe(ml_preds, "expected_variance", 0.1), 0.0, 5.0))

        # [6-8] HTF directional bias probabilities
        state[6:9] = np.asarray(htf_p, dtype=np.float32)

        # [9-14] LTF score-based regime context
        state[9] = float(np.clip(trend_score, 0.0, 1.0))
        state[10] = float(np.clip(range_score, 0.0, 1.0))
        state[11] = float(np.clip(chop_score, 0.0, 1.0))
        state[12] = float(np.clip(vol_pct, 0.0, 1.0))
        state[13] = float(np.clip(consolidation_score, 0.0, 1.0))
        state[14] = float(np.clip(trade_code, 0.0, 1.0))

        # [15-26] regime confidence, market structure, and execution friction
        state[15] = float(np.clip(htf_conf, 0.0, 1.0))
        state[16] = _ltf_confidence(regime_scores)
        state[17] = float(np.clip(_bar("adx_14", 20.0) / 100.0, 0.0, 1.0))
        state[18] = float(np.clip(_bar("ema_stack", 0.0) / 2.0, -1.0, 1.0))
        state[19] = atr_ratio
        state[20] = float(np.clip(_safe(ml_preds, "spread_pips", 1.0) / 5.0, 0.0, 1.0))
        state[21] = _flag("bos_bull")
        state[22] = _flag("bos_bear")
        state[23] = _flag("fvg_bull")
        state[24] = _flag("fvg_bear")
        state[25] = float(np.clip(_safe(ml_preds, "vol_slope", _bar("vol_slope_seq", 0.0)), -1.0, 1.0))
        state[26] = float(np.clip(_bar("vwap_dist_atr", 0.0) / 3.0, -1.0, 1.0))

        # [27-30] time/session/news context
        state[27] = 1.0 if ml_preds.get("news_in_30min", False) else 0.0
        state[28] = float(np.sin(2.0 * np.pi * float(h) / 24.0))
        state[29] = float(np.cos(2.0 * np.pi * float(h) / 24.0))
        state[30] = _quality_session_code(session)

        # [31-36] portfolio context
        equity = _quality_float(portfolio.get("equity", portfolio.get("balance", 1000.0)), 1000.0)
        state[31] = float(np.clip(portfolio.get("open_positions", 0), 0, 10)) / 10.0
        state[32] = float(np.clip(portfolio.get("drawdown_pct", 0.0), 0.0, 0.2)) / 0.2
        state[33] = float(np.clip(_quality_float(portfolio.get("daily_pnl", 0.0), 0.0) / (equity + 1e-9), -0.10, 0.10))
        state[34] = float(np.clip(portfolio.get("trades_today", 0), 0, 20)) / 20.0
        state[35] = float(np.clip(portfolio.get("win_rate_10", 0.5), 0.0, 1.0))
        state[36] = float(np.clip(portfolio.get("equity_norm", 1.0), 0.0, 2.0))

        # [37-40] instrument one-hot
        idx = _INSTRUMENT_IDX.get(symbol, -1)
        if 0 <= idx <= 3:
            state[37 + idx] = 1.0

        # [41-42] setup validity flags consumed by the decision layer
        state[41] = 1.0 if bool(_bar("range_valid", 0.0)) else 0.0
        state[42] = 1.0 if bool(_bar("pullback_valid", 0.0)) else 0.0

        if state.shape[0] != RL_STATE_DIM:
            raise ValueError(f"RL state dimension mismatch: got {state.shape[0]}, expected {RL_STATE_DIM}")
        state = np.clip(state, -10.0, 10.0)
        state = np.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0)
        return state.astype(np.float32)
