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
    # ── Base TF (15M execution) ──────────────────────────────────────────────
    "log_return",           # 0
    "high_low_range",       # 1
    "close_vs_open",        # 2
    "atr_normalized",       # 3
    "rsi_14",               # 4
    "ema21_dist",           # 5
    "ema50_dist",           # 6
    "bb_position",          # 7
    "volume_ratio",         # 8
    "is_asian",             # 9
    "is_london",            # 10
    "is_ny",                # 11
    "bos_bull_flag",        # 12
    "bos_bear_flag",        # 13
    "fvg_bull_open",        # 14
    "fvg_bear_open",        # 15
    # ── 5M cross-TF features (execution precision) ───────────────────────────
    "mtf_5m_rsi",           # 16
    "mtf_5m_ema21_dist",    # 17
    # ── 1H cross-TF features (intraday trend) ────────────────────────────────
    "mtf_1h_adx",           # 18
    "mtf_1h_ema21_dist",    # 19
    "mtf_1h_ema50_dist",    # 20
    # ── 4H cross-TF features (structural context) ────────────────────────────
    "mtf_4h_ema21_ema50_diff",  # 21
    "mtf_4h_adx",           # 22
    "mtf_4h_rsi",           # 23
    # ── 1D cross-TF features (macro structure) ───────────────────────────────
    "mtf_1d_ema21_dist",    # 24
    "mtf_1d_ema_stack",     # 25
    # ── HTF bias context (4H classifier: 3-class) ───────────────────────────
    "htf_bias_up",          # 26  one-hot: 4H bias is BIAS_UP
    "htf_bias_down",        # 27  one-hot: 4H bias is BIAS_DOWN
    "htf_bias_neutral",     # 28  one-hot: 4H bias is BIAS_NEUTRAL
    "htf_bias_conf",        # 29  4H classifier confidence (max softmax)
    # ── LTF behaviour context (1H classifier: 4-class) ───────────────────────
    "ltf_trending",         # 30  one-hot: 1H behaviour is TRENDING
    "ltf_ranging",          # 31  one-hot: 1H behaviour is RANGING
    "ltf_consolidating",    # 32  one-hot: 1H behaviour is CONSOLIDATING
    "ltf_volatile",         # 33  one-hot: 1H behaviour is VOLATILE
    "ltf_conf",             # 34  1H classifier confidence
    # ── HTF/LTF alignment + duration (3 slots, indices 35–37) ───────────────
    "htf_ltf_align",        # 35  1.0 if HTF bias is directional AND LTF is TRENDING
    "htf_regime_dur",       # 36  bars since last HTF bias change / 100 (capped 1.0)
    "ltf_regime_dur",       # 37  bars since last LTF behaviour change / 100 (capped 1.0)
    # ── Volatility dynamics ───────────────────────────────────────────────────
    "vol_slope_seq",        # 38  Δ(ATR/close) over 14 bars × 1000
    # ── Time encoding (cyclic) ────────────────────────────────────────────────
    "time_sin",             # 39  sin(2π × hour / 24)
    "time_cos",             # 40  cos(2π × hour / 24)
    # ── ICT structural features ───────────────────────────────────────────────
    # EMA structure
    "ema_pullback_zone",    # 41  price in EMA21-50 band normalised by ATR (0=outside, ±1=inside)
    "ema21_slope_15m",      # 42  EMA21 slope / ATR over 5 bars
    "ema21_slope_1h",       # 43  1H EMA21 slope / ATR (ffill to 15M)
    "ema_stack_15m",        # 44  15M EMA stack score / 2
    # BOS — age + strength
    "bos_bull_bars_ago",    # 45  bars since last bull BOS / 40 (1.0 = no recent BOS)
    "bos_bear_bars_ago",    # 46  bars since last bear BOS / 40
    "bos_bull_strength",    # 47  bull BOS move / ATR at signal bar (0 = no recent BOS)
    "bos_bear_strength",    # 48  bear BOS move / ATR at signal bar
    # FVG — distance + fill ratio
    "fvg_bull_dist_atr",    # 49  distance from close to nearest open bull FVG / ATR
    "fvg_bear_dist_atr",    # 50  distance from close to nearest open bear FVG / ATR
    "fvg_bull_fill_ratio",  # 51  how far price has moved into nearest bull FVG [0,1]
    "fvg_bear_fill_ratio",  # 52  how far price has moved into nearest bear FVG [0,1]
    # Sweep / liquidity
    "sweep_wick_depth_atr", # 53  wick beyond recent range extreme / ATR (last 3 bars, else 0)
    "body_recovery_ratio",  # 54  |close-open| / (high-low) of sweep candle (0 if no sweep)
    # Liquidity proximity
    "dist_to_recent_high_atr",  # 55  (20-bar high - close) / ATR
    "dist_to_recent_low_atr",   # 56  (close - 20-bar low) / ATR
    # Asian range context
    "asian_range_width_atr",    # 57  Asian session range width / ATR
    "price_vs_asian_high_atr",  # 58  (close - asian_high) / ATR
    "price_vs_asian_low_atr",   # 59  (close - asian_low) / ATR
    # Candle structure
    "candle_body_ratio",    # 60  |close-open| / (high-low)
    "upper_wick_ratio",     # 61  upper wick / (high-low)
    "lower_wick_ratio",     # 62  lower wick / (high-low)
    # Oscillators
    "rsi_extreme",          # 63  (rsi - 50) / 50  — distance from neutral
    "stoch_k",              # 64  stoch %K / 100
    "stoch_k_vs_d",         # 65  (stoch K - stoch D) / 100
    # ADX on 15M
    "adx_15m",              # 66  ADX / 100
    # Regime dynamics
    "regime_duration",      # 67  bars in current regime / 100 (capped at 1.0)
    "vol_expansion",        # 68  ATR_t / ATR_{t-10} (clipped 0.5–3.0, centred at 1.0)
    "atr_pctile",           # 69  ATR percentile rank in own 42-bar history [0→1]
                            #     low = compression/consolidation; high = breakout expansion
    # Session timing — continuous
    "mins_since_london_open",   # 70  minutes since 07:00 UTC / 480 (-1 if before open)
    "mins_since_ny_open",       # 71  minutes since 13:00 UTC / 300 (-1 if before open)
    # ── Execution-relevant macro context (daily, ffill'd to 15M) ─────────────
    # Only risk/liquidity signals: 17 index returns omitted (daily resolution
    # adds noise at 15M execution; they belong in the 4H bias classifier).
    "macro_vix_level",          # 72  VIX percentile — risk-off context for sizing
    "macro_yield_spread",       # 73  10Y-2Y spread — macro regime signal
]
# Total: 74 features (26 base/MTF + 12 regime slots [3 HTF + 1 HTF_conf + 4 LTF + 1 LTF_conf + 3 align/dur]
#                     + 36 ICT/macro)

# ─── 4H BIAS classifier features ─────────────────────────────────────────────
# Trained on 4H data. Only HTF-appropriate features: no 5M/15M noise.
# Macro indices belong here — they operate at daily/weekly resolution, match 4H bias.
REGIME_4H_FEATURES = [
    # ── 4H base structural features ───────────────────────────────────────────
    "adx_14_base",          # 0   ADX on 4H df
    "ema_stack_score",      # 1   EMA stack score on 4H
    "atr_ratio",            # 2   ATR/close * 1000 on 4H
    "bb_width_pct",         # 3   BB width on 4H
    "realized_vol_20",      # 4   20-bar realised vol on 4H
    # ── HTF context: 1D features ──────────────────────────────────────────────
    "mtf_1d_adx",           # 5
    "mtf_1d_ema_stack",     # 6
    "mtf_1d_atr_ratio",     # 7
    "mtf_1d_bb_width",      # 8
    # ── Regime dynamics ────────────────────────────────────────────────────────
    "vol_slope",            # 9   Δ(ATR/close) over 14 bars — bias expanding/contracting
    "regime_duration",      # 10  bars since last regime change (capped 50, /50)
    "atr_pctile",           # 11  ATR percentile rank in own 3×n_bar history — consolidation signal
    # ── Time-series discriminators (BIAS_NEUTRAL vs BIAS_UP/DOWN) ─────────────
    # These three features are used to *label* regimes in the GMM but were never
    # given to the MLP — so the classifier had no signal to separate BIAS_NEUTRAL
    # (low autocorr, low efficiency) from weak-trend bars (same ADX, different dynamics).
    "efficiency_ratio",     # 12  |net n-bar move| / sum(|bar moves|) [0→1, 1=clean trend]
    "autocorr_lag1",        # 13  lag-1 autocorrelation of log-returns — trending>0, ranging≈0
    "hurst_proxy",          # 14  R/S Hurst proxy — H>1=trending, H<1=mean-reverting
] + INDEX_FEATURES + [
    "macro_vix_level",      # macro risk-off/on
    "macro_yield_spread",   # yield curve regime signal
]  # 15 base + 1D context + regime + dynamics + 19 macro = 34 features

# ─── 1H STRUCTURE classifier features ────────────────────────────────────────
# Trained on 1H data. No macro indices — too coarse for 1H structure decisions.
REGIME_1H_FEATURES = [
    # ── 1H base structural features ───────────────────────────────────────────
    "adx_14_base",          # 0   ADX on 1H df
    "ema_stack_score",      # 1   EMA stack score on 1H
    "atr_ratio",            # 2   ATR/close * 1000 on 1H
    "bb_width_pct",         # 3   BB width on 1H
    "realized_vol_20",      # 4   20-bar realised vol on 1H
    # ── Session context ──────────────────────────────────────────────────────
    "session_code",         # 5   0=inactive 1=asian 2=london 3=ny 4=dead
    # ── Intraday market structure ─────────────────────────────────────────────
    "swing_hh_hl_count",    # 6   BOS count in last 40 bars
    "liquidity_sweep_24h",  # 7   sweep count in last 48 bars
    # ── 4H context (zoom-out from 1H) ─────────────────────────────────────────
    "mtf_4h_adx",           # 8
    "mtf_4h_ema_stack",     # 9
    "mtf_4h_atr_ratio",     # 10
    "mtf_4h_bb_width",      # 11
    # ── Regime dynamics ────────────────────────────────────────────────────────
    "vol_slope",            # 12  Δ(ATR/close) over 14 bars
    "regime_duration",      # 13  bars since last regime change
    "atr_pctile",           # 14  ATR percentile rank — consolidation signal
    # ── Time-series discriminators (RANGING vs TRENDING — the key missing signal)
    # RANGING has near-zero autocorr, low efficiency ratio, Hurst≈0.5.
    # TRENDING has high autocorr, high efficiency, Hurst>0.5.
    # Without these the MLP sees identical ADX/ATR for both and collapses RANGING to 0%.
    "efficiency_ratio",     # 15  |net n-bar move| / sum(|bar moves|) [0→1]
    "autocorr_lag1",        # 16  lag-1 autocorrelation of log-returns
    "hurst_proxy",          # 17  R/S Hurst proxy
]  # 18 features total

# ─── Legacy REGIME_FEATURES (shared contract kept for backwards compat) ───────
# Used by _build_feature_matrix which builds ALL columns regardless of which
# subset each classifier uses. Classifiers then index into the relevant columns.
REGIME_FEATURES = [
    # ── Base structural features (computed on the input df TF) ───────────────
    "adx_14_base",          # 0   ADX of input df (TF-agnostic label)
    "ema_stack_score",      # 1   EMA stack score of input df
    "atr_ratio",            # 2   ATR/close * 1000
    "bb_width_pct",         # 3   BB width
    "realized_vol_20",      # 4   20-bar realised volatility
    "session_code",         # 5   0=inactive 1=asian 2=london 3=ny 4=dead
    "swing_hh_hl_count",    # 6   BOS count in last 40 bars
    "liquidity_sweep_24h",  # 7   sweep count in last 48 bars
    # ── 5M TF features ───────────────────────────────────────────────────────
    "mtf_5m_adx",           # 8
    "mtf_5m_ema_stack",     # 9
    "mtf_5m_atr_ratio",     # 10
    "mtf_5m_bb_width",      # 11
    # ── 15M TF features ──────────────────────────────────────────────────────
    "mtf_15m_adx",          # 12
    "mtf_15m_ema_stack",    # 13
    "mtf_15m_atr_ratio",    # 14
    "mtf_15m_bb_width",     # 15
    # ── 1H TF features ───────────────────────────────────────────────────────
    "mtf_1h_adx",           # 16
    "mtf_1h_ema_stack",     # 17
    "mtf_1h_atr_ratio",     # 18
    "mtf_1h_bb_width",      # 19
    # ── 4H TF features ───────────────────────────────────────────────────────
    "mtf_4h_adx",           # 20
    "mtf_4h_ema_stack",     # 21
    "mtf_4h_atr_ratio",     # 22
    "mtf_4h_bb_width",      # 23
    # ── 1D TF features ───────────────────────────────────────────────────────
    "mtf_1d_adx",           # 24
    "mtf_1d_ema_stack",     # 25
    "mtf_1d_atr_ratio",     # 26
    "mtf_1d_bb_width",      # 27
    # ── S/R and Supply/Demand zone features ──────────────────────────────────
    "sr_dist_resist_atr",   # 28
    "sr_dist_support_atr",  # 29
    "sr_in_supply_zone",    # 30
    "sr_in_demand_zone",    # 31
    "sr_resist_strength",   # 32
    "sr_support_strength",  # 33
    # ── Regime dynamics ──────────────────────────────────────────────────────
    "vol_slope",            # 34
    "regime_duration",      # 35
    "atr_pctile",           # 36  ATR percentile rank — consolidation signal
    # ── Regime memory ─────────────────────────────────────────────────────────
    # HTF bias prev_regime (3 slots)
    "prev_regime_htf_up",       # 37  one-hot: previous HTF regime was BIAS_UP
    "prev_regime_htf_down",     # 38  one-hot: previous HTF regime was BIAS_DOWN
    "prev_regime_htf_neutral",  # 39  one-hot: previous HTF regime was BIAS_NEUTRAL
    # LTF behaviour prev_regime (4 slots)
    "prev_regime_ltf_trending",     # 40  one-hot: previous LTF regime was TRENDING
    "prev_regime_ltf_ranging",      # 41  one-hot: previous LTF regime was RANGING
    "prev_regime_ltf_consolidating",# 42  one-hot: previous LTF regime was CONSOLIDATING
    "prev_regime_ltf_volatile",     # 43  one-hot: previous LTF regime was VOLATILE
    "regime_confidence",    # 44
    # ── Time-series discriminators ────────────────────────────────────────────
    # These were only used for GMM labeling — never passed to the MLP.
    # Without them, RANGING (autocorr≈0, eff≈0, Hurst≈0.5) is indistinguishable
    # from weak TRENDING in ADX/ATR space, causing the model to predict RANGING=0%.
    "efficiency_ratio",     # 45  |net n-bar move| / sum(|bar moves|) [0→1]
    "autocorr_lag1",        # 46  lag-1 autocorrelation of log-returns [-1→1]
    "hurst_proxy",          # 47  R/S Hurst proxy [0.2→3.0, normalised to 0→1]
] + INDEX_FEATURES + [
    "macro_vix_level",
    "macro_yield_spread",
]  # base + MTF + S/R + regime dynamics + memory + ts-discriminators + indices + macro

QUALITY_FEATURES = [
    "strategy_id",          # 0
    "signal_direction",     # 1
    "rr_ratio",             # 2
    "p_bull_gru",           # 3
    "p_bear_gru",           # 4
    "regime_class",         # 5
    "sentiment_score",      # 6
    "adx_at_signal",        # 7
    "atr_ratio_at_signal",  # 8
    "volume_ratio",         # 9
    "spread_at_signal",     # 10
    "session_at_signal",    # 11
    "news_in_30min",        # 12
    "strategy_win_rate_20", # 13
    "gru_uncertainty",      # 14  expected_variance from GRU variance head
    "regime_duration",      # 15  normalised bars since last regime change [0,1]
    "vol_slope_at_signal",  # 16  Δ(ATR/close)*1000 — expanding vs contracting vol
]  # 17 features

# RL state dimension layout (total = 42):
# [0-5]   ML predictions  (p_bull, p_bear, entry_depth, regime_id, sentiment, quality)
# [6-13]  Market structure (adx, ema_stack, atr_ratio, bb_width, bos_bull, bos_bear, fvg_bull, fvg_bear)
# [14-18] Session context  (asian, london, ny, dead, news_proximity)
# [19-23] Strategy signals (S1–S5 booleans)
# [24-29] Portfolio state  (open_pos, drawdown, daily_pnl, trades_today, last_result, equity_norm)
# [30-33] Instrument one-hot (EURUSD, GBPUSD, USDJPY, XAUUSD)
# [34-41] ATR history ratios (8 lags: 1,4,8,24,48,96,168,336 bars)
RL_STATE_DIM = 43

_INSTRUMENT_IDX = {"EURUSD": 0, "GBPUSD": 1, "USDJPY": 2, "XAUUSD": 3}
_ATR_LAGS = [1, 4, 8, 24, 48, 96, 168, 336]

_MACRO_CACHE: Dict[str, pd.Series] = {}
_MACRO_MAP_CACHE: Dict[str, Any] | None = None
_MACRO_MAP_MTIME: float = 0.0
# Aligned macro DataFrame cache: keyed by (symbol, index_id) so that
# _build_macro_frame only does the reindex+ffill once per symbol per run.
# Reindexing 19 daily series onto a 300k-bar 15M index is O(N) pandas work —
# caching avoids repeating it for every call from _build_feature_matrix.
_MACRO_ALIGNED_CACHE: Dict[tuple, "pd.DataFrame"] = {}


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
        regime_series: Optional["pd.Series"] = None,
        regime_conf_series: Optional["pd.Series"] = None,
        # New canonical names
        regime_htf_series: Optional["pd.Series"] = None,
        regime_htf_conf_series: Optional["pd.Series"] = None,
        regime_ltf_series: Optional["pd.Series"] = None,
        regime_ltf_conf_series: Optional["pd.Series"] = None,
        # Legacy backward-compat aliases (maps to htf/ltf)
        regime_4h_series: Optional["pd.Series"] = None,
        regime_4h_conf_series: Optional["pd.Series"] = None,
        regime_1h_series: Optional["pd.Series"] = None,
        regime_1h_conf_series: Optional["pd.Series"] = None,
    ) -> np.ndarray:
        """
        Returns shape (length, N) float32.
        Pads with zeros at start if len(df) < length. Never raises IndexError.
        regime_htf_series: int labels (0-2) from HTF bias classifier (3-class).
        regime_ltf_series: int labels (0-3) from LTF behaviour classifier (4-class).
        regime_4h/1h_series: legacy aliases for htf/ltf.
        regime_series/regime_conf_series: legacy params kept for backwards compat,
            maps to regime_htf if regime_htf_series is None.
        """
        if df is None or len(df) == 0:
            return np.zeros((length, len(SEQUENCE_FEATURES)), dtype=np.float32)

        # Resolve aliases: new canonical > legacy 4h/1h > oldest regime_series
        _r4h = regime_htf_series or regime_4h_series or regime_series
        _c4h = regime_htf_conf_series or regime_4h_conf_series or regime_conf_series
        _r1h = regime_ltf_series or regime_1h_series
        _c1h = regime_ltf_conf_series or regime_1h_conf_series

        feat = self._build_sequence_df(
            df, df_htf, symbol=symbol,
            regime_4h_series=_r4h,
            regime_4h_conf_series=_c4h,
            regime_1h_series=_r1h,
            regime_1h_conf_series=_c1h,
        )
        arr = feat[SEQUENCE_FEATURES].values.astype(np.float32)

        # Replace NaN / Inf
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if len(arr) >= length:
            return arr[-length:]
        else:
            pad = np.zeros((length - len(arr), arr.shape[1]), dtype=np.float32)
            return np.vstack([pad, arr])

    def _build_sequence_df(
        self, df: pd.DataFrame,
        df_htf: Optional[Dict[str, pd.DataFrame]],
        symbol: Optional[str] = None,
        regime_series: Optional["pd.Series"] = None,
        regime_conf_series: Optional["pd.Series"] = None,
        regime_4h_series: Optional["pd.Series"] = None,
        regime_4h_conf_series: Optional["pd.Series"] = None,
        regime_1h_series: Optional["pd.Series"] = None,
        regime_1h_conf_series: Optional["pd.Series"] = None,
    ) -> pd.DataFrame:
        """Add all sequence feature columns to a copy of df.

        df_htf: dict of {tf_key: DataFrame} with keys "5M", "1H", "4H", "1D".
        regime_4h_series: int labels (0-2) from HTF bias classifier — BIAS_UP/DOWN/NEUTRAL.
        regime_1h_series: int labels (0-3) from LTF behaviour classifier — TRENDING/RANGING/CONSOLIDATING/VOLATILE.
        regime_series/regime_conf_series: legacy compat, treated as HTF if _4h not provided.
        """
        from indicators.market_structure import (
            compute_rsi, compute_adx, compute_bollinger_bands,
            detect_break_of_structure, detect_fair_value_gaps,
        )

        out = df.copy(deep=False)
        atr = out.get("atr_14", compute_atr(df, 14))

        # ── Base TF features ─────────────────────────────────────────────────
        out["log_return"]    = np.log(out["close"] / (out["close"].shift(1) + 1e-9)).clip(-0.1, 0.1)
        out["high_low_range"]= (out["high"] - out["low"]) / (atr + 1e-9)
        out["close_vs_open"] = (out["close"] - out["open"]) / (atr + 1e-9)
        out["atr_normalized"]= atr / (out["close"] + 1e-9)

        if "rsi_14" not in out.columns:
            out["rsi_14"] = compute_rsi(out["close"], 14)
        out["rsi_14"] = (out["rsi_14"] - 50.0) / 50.0

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
        out["bb_position"] = (out["close"] - out["bb_lower"]) / (bb_range + 1e-9)

        vol_sma = out["volume"].rolling(20).mean()
        out["volume_ratio"] = out["volume"] / (vol_sma + 1e-9)

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

        # ── Helper: reindex a HTF series onto base df index ──────────────────
        def _htf_series(htf_df: Optional[pd.DataFrame], compute_fn, fallback=0.0) -> pd.Series:
            """Compute a series on htf_df, forward-fill to out.index."""
            if htf_df is None or len(htf_df) < 14:
                return pd.Series(fallback, index=out.index)
            s = compute_fn(htf_df)
            return s.reindex(out.index, method="ffill").fillna(fallback)

        htf = df_htf if isinstance(df_htf, dict) else {}

        df_5m = _first_present_frame(htf, "5M", "5m")
        df_1h = _first_present_frame(htf, "1H", "H1")
        df_4h = _first_present_frame(htf, "4H", "H4")
        df_1d = _first_present_frame(htf, "1D", "D1")

        # ── 5M cross-TF ──────────────────────────────────────────────────────
        out["mtf_5m_rsi"]      = _htf_series(df_5m,
            lambda d: (compute_rsi(d["close"], 14) - 50.0) / 50.0)
        out["mtf_5m_ema21_dist"] = _htf_series(df_5m,
            lambda d: (d["close"] - compute_ema(d["close"], 21)) / (compute_atr(d, 14) + 1e-9))

        # ── 1H cross-TF ──────────────────────────────────────────────────────
        out["mtf_1h_adx"]      = _htf_series(df_1h,
            lambda d: compute_adx(d, 14) / 100.0)
        out["mtf_1h_ema21_dist"] = _htf_series(df_1h,
            lambda d: (d["close"] - compute_ema(d["close"], 21)) / (compute_atr(d, 14) + 1e-9))
        out["mtf_1h_ema50_dist"] = _htf_series(df_1h,
            lambda d: (d["close"] - compute_ema(d["close"], 50)) / (compute_atr(d, 14) + 1e-9))

        # ── 4H cross-TF ──────────────────────────────────────────────────────
        out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h,
            lambda d: (compute_ema(d["close"], 21) - compute_ema(d["close"], 50)) / (d["close"] + 1e-9))
        out["mtf_4h_adx"]      = _htf_series(df_4h,
            lambda d: compute_adx(d, 14) / 100.0)
        out["mtf_4h_rsi"]      = _htf_series(df_4h,
            lambda d: (compute_rsi(d["close"], 14) - 50.0) / 50.0)

        # ── 1D cross-TF ──────────────────────────────────────────────────────
        out["mtf_1d_ema21_dist"] = _htf_series(df_1d,
            lambda d: (d["close"] - compute_ema(d["close"], 21)) / (compute_atr(d, 14) + 1e-9))
        out["mtf_1d_ema_stack"]  = _htf_series(df_1d,
            lambda d: _ema_stack_series(d) / 2.0)

        # ── Late additions — all collected into `extra`, concat'd once ──────────
        from indicators.market_structure import compute_stochastic, compute_adx as _compute_adx
        # Execution macro: only VIX and yield spread — daily-resolution index returns
        # are not informative at 15M execution precision and add noise.
        macro_df = self._build_macro_frame(out.index, symbol)
        extra: dict[str, np.ndarray] = {
            "macro_vix_level":   macro_df["macro_vix_level"].to_numpy(dtype=np.float32),
            "macro_yield_spread": macro_df["macro_yield_spread"].to_numpy(dtype=np.float32),
        }
        n = len(out)
        _close = out["close"].to_numpy(dtype=np.float64)
        _high  = out["high"].to_numpy(dtype=np.float64)
        _low   = out["low"].to_numpy(dtype=np.float64)
        _open  = out["open"].to_numpy(dtype=np.float64)
        _atr   = atr.to_numpy(dtype=np.float64)

        # ── Dual-regime context (4H bias + 1H behaviour) ─────────────────────
        # Legacy: if old regime_series provided but new _4h not, treat as 4H
        _r4h = regime_4h_series if regime_4h_series is not None else regime_series
        _c4h = regime_4h_conf_series if regime_4h_conf_series is not None else regime_conf_series

        # HTF bias (3-class: BIAS_UP=0, BIAS_DOWN=1, BIAS_NEUTRAL=2)
        if _r4h is not None:
            cur_htf = _r4h.reindex(out.index, method="ffill").fillna(2).astype(int)
            extra["htf_bias_up"]      = (cur_htf == 0).to_numpy(dtype=np.float32)
            extra["htf_bias_down"]    = (cur_htf == 1).to_numpy(dtype=np.float32)
            extra["htf_bias_neutral"] = (cur_htf == 2).to_numpy(dtype=np.float32)
        else:
            cur_htf = pd.Series(2, index=out.index, dtype=int)
            extra["htf_bias_up"]      = np.zeros(n, dtype=np.float32)
            extra["htf_bias_down"]    = np.zeros(n, dtype=np.float32)
            extra["htf_bias_neutral"] = np.ones(n, dtype=np.float32)
        extra["htf_bias_conf"] = (
            _c4h.reindex(out.index, method="ffill").fillna(0.33).clip(0, 1).to_numpy(dtype=np.float32)
            if _c4h is not None else np.full(n, 0.33, dtype=np.float32)
        )

        # LTF behaviour (4-class: TRENDING=0, RANGING=1, CONSOLIDATING=2, VOLATILE=3)
        if regime_1h_series is not None:
            cur_ltf = regime_1h_series.reindex(out.index, method="ffill").fillna(1).astype(int)
            extra["ltf_trending"]      = (cur_ltf == 0).to_numpy(dtype=np.float32)
            extra["ltf_ranging"]       = (cur_ltf == 1).to_numpy(dtype=np.float32)
            extra["ltf_consolidating"] = (cur_ltf == 2).to_numpy(dtype=np.float32)
            extra["ltf_volatile"]      = (cur_ltf == 3).to_numpy(dtype=np.float32)
        else:
            cur_ltf = pd.Series(1, index=out.index, dtype=int)
            extra["ltf_trending"]      = np.zeros(n, dtype=np.float32)
            extra["ltf_ranging"]       = np.ones(n, dtype=np.float32)
            extra["ltf_consolidating"] = np.zeros(n, dtype=np.float32)
            extra["ltf_volatile"]      = np.zeros(n, dtype=np.float32)
        extra["ltf_conf"] = (
            regime_1h_conf_series.reindex(out.index, method="ffill").fillna(0.25).clip(0, 1).to_numpy(dtype=np.float32)
            if regime_1h_conf_series is not None else np.full(n, 0.25, dtype=np.float32)
        )

        # HTF/LTF alignment: 1.0 when HTF has directional bias AND LTF is TRENDING
        _htf_a = cur_htf  # already aligned to out.index
        _ltf_a = cur_ltf  # already aligned to out.index
        extra["htf_ltf_align"] = ((_htf_a != 2) & (_ltf_a == 0)).to_numpy(dtype=np.float32)

        # HTF regime duration: bars since last HTF bias change / 100 (capped 1.0)
        _htf_dur_arr = np.zeros(n, dtype=np.float32)
        _cnt = 0
        _prev = -1
        for _i in range(n):
            _v = int(_htf_a.iloc[_i])
            if _v != _prev:
                _cnt = 0
                _prev = _v
            _cnt += 1
            _htf_dur_arr[_i] = min(_cnt / 100.0, 1.0)
        extra["htf_regime_dur"] = _htf_dur_arr

        # LTF regime duration: bars since last LTF behaviour change / 100 (capped 1.0)
        _ltf_dur_arr = np.zeros(n, dtype=np.float32)
        _cnt2 = 0
        _prev2 = -1
        for _i in range(n):
            _v2 = int(_ltf_a.iloc[_i])
            if _v2 != _prev2:
                _cnt2 = 0
                _prev2 = _v2
            _cnt2 += 1
            _ltf_dur_arr[_i] = min(_cnt2 / 100.0, 1.0)
        extra["ltf_regime_dur"] = _ltf_dur_arr

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
        extra["ema21_slope_1h"] = _htf_series(df_1h,
            lambda d: (compute_ema(d["close"], 21) - compute_ema(d["close"], 21).shift(3)).fillna(0.0)
                      / (compute_atr(d, 14) + 1e-9)
        ).clip(-5.0, 5.0).to_numpy(dtype=np.float32)
        # 15M EMA stack score
        from indicators.market_structure import compute_ema_stack_score as _ema_stack_fn
        extra["ema_stack_15m"] = np.clip(
            _ema_stack_fn(out).fillna(0.0).to_numpy() / 2.0, -1.0, 1.0
        ).astype(np.float32)

        # ── BOS age + strength ────────────────────────────────────────────────
        # bos_bull/bos_bear already computed above in out["bos_bull"], out["bos_bear"]
        _bos_bull = out["bos_bull"].to_numpy(dtype=bool)
        _bos_bear = out["bos_bear"].to_numpy(dtype=bool)
        _cap = 40.0
        bos_bull_ago   = np.full(n, _cap, dtype=np.float32)
        bos_bear_ago   = np.full(n, _cap, dtype=np.float32)
        bos_bull_str   = np.zeros(n, dtype=np.float32)
        bos_bear_str   = np.zeros(n, dtype=np.float32)
        _last_bull_bar = -int(_cap)
        _last_bear_bar = -int(_cap)
        _last_bull_move = 0.0
        _last_bear_move = 0.0
        for _i in range(n):
            if _bos_bull[_i]:
                _last_bull_bar  = _i
                # move = close - open as proxy for BOS impulse size
                _last_bull_move = max(float(_close[_i] - _open[_i]), 0.0)
            if _bos_bear[_i]:
                _last_bear_bar  = _i
                _last_bear_move = max(float(_open[_i] - _close[_i]), 0.0)
            _atr_i = float(_atr[_i]) if _atr[_i] > 0 else 1e-9
            bos_bull_ago[_i] = min((_i - _last_bull_bar), _cap) / _cap
            bos_bear_ago[_i] = min((_i - _last_bear_bar), _cap) / _cap
            bos_bull_str[_i] = min(_last_bull_move / _atr_i, 5.0) if _last_bull_bar >= 0 else 0.0
            bos_bear_str[_i] = min(_last_bear_move / _atr_i, 5.0) if _last_bear_bar >= 0 else 0.0
        extra["bos_bull_bars_ago"]  = bos_bull_ago
        extra["bos_bear_bars_ago"]  = bos_bear_ago
        extra["bos_bull_strength"]  = bos_bull_str
        extra["bos_bear_strength"]  = bos_bear_str

        # ── FVG distance + fill ratio ─────────────────────────────────────────
        # fvg_bull_top/bottom carry the FVG boundaries at the bar it formed; forward-fill
        # to get the most recent open FVG levels. Reset when fvg_bear forms (directional flip).
        _fvg_bull_top    = out.get("fvg_bull_top",    pd.Series(np.nan, index=out.index)).to_numpy(dtype=np.float64)
        _fvg_bull_bottom = out.get("fvg_bull_bottom", pd.Series(np.nan, index=out.index)).to_numpy(dtype=np.float64)
        _fvg_bear_top    = out.get("fvg_bear_top",    pd.Series(np.nan, index=out.index)).to_numpy(dtype=np.float64)
        _fvg_bear_bottom = out.get("fvg_bear_bottom", pd.Series(np.nan, index=out.index)).to_numpy(dtype=np.float64)
        fvg_bull_dist = np.zeros(n, dtype=np.float32)
        fvg_bear_dist = np.zeros(n, dtype=np.float32)
        fvg_bull_fill = np.zeros(n, dtype=np.float32)
        fvg_bear_fill = np.zeros(n, dtype=np.float32)
        _cur_bull_top = _cur_bull_bot = _cur_bear_top = _cur_bear_bot = np.nan
        for _i in range(n):
            if not np.isnan(_fvg_bull_top[_i]):
                _cur_bull_top = _fvg_bull_top[_i]
                _cur_bull_bot = _fvg_bull_bottom[_i]
            if not np.isnan(_fvg_bear_top[_i]):
                _cur_bear_top = _fvg_bear_top[_i]
                _cur_bear_bot = _fvg_bear_bottom[_i]
            _atr_i = float(_atr[_i]) if _atr[_i] > 1e-12 else 1e-9
            _c = float(_close[_i])
            if not np.isnan(_cur_bull_bot):
                _gap = _cur_bull_top - _cur_bull_bot
                fvg_bull_dist[_i] = float(np.clip((_c - _cur_bull_top) / _atr_i, -5.0, 5.0))
                fvg_bull_fill[_i] = float(np.clip(
                    (_c - _cur_bull_bot) / (_gap + 1e-9), 0.0, 1.0
                )) if _gap > 1e-9 else 0.0
            if not np.isnan(_cur_bear_top):
                _gap = _cur_bear_top - _cur_bear_bot
                fvg_bear_dist[_i] = float(np.clip((_cur_bear_bot - _c) / _atr_i, -5.0, 5.0))
                fvg_bear_fill[_i] = float(np.clip(
                    (_cur_bear_top - _c) / (_gap + 1e-9), 0.0, 1.0
                )) if _gap > 1e-9 else 0.0
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
        sweep_wick = np.zeros(n, dtype=np.float32)
        body_rec   = np.zeros(n, dtype=np.float32)
        # Look back 3 bars for a sweep signal
        for _i in range(n):
            _atr_i = float(_atr[_i]) if _atr[_i] > 1e-12 else 1e-9
            for _lag in range(3):
                _j = _i - _lag
                if _j < 0:
                    break
                if _sw_bull[_j] or _sw_bear[_j]:
                    _wick = float(_sw_bull_wick[_j] if _sw_bull[_j] else _sw_bear_wick[_j])
                    sweep_wick[_i] = float(np.clip(_wick / _atr_i, 0.0, 5.0))
                    _hl_j = float(_hl[_j])
                    body_rec[_i] = float(np.clip(_body[_j] / (_hl_j + 1e-9), 0.0, 1.0))
                    break
        extra["sweep_wick_depth_atr"] = sweep_wick
        extra["body_recovery_ratio"]  = body_rec

        # ── Liquidity proximity ───────────────────────────────────────────────
        _high20 = pd.Series(_high, index=out.index).rolling(20).max().to_numpy(dtype=np.float64)
        _low20  = pd.Series(_low,  index=out.index).rolling(20).min().to_numpy(dtype=np.float64)
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
            _asian_h = np.where(_is_asian_window, _high, np.nan)
            _asian_l = np.where(_is_asian_window, _low,  np.nan)
            # Forward-fill session high/low: reset at 05:00 UTC each day
            _cur_ah = _cur_al = np.nan
            _prev_date = None
            for _i in range(n):
                _date = out.index[_i].date()
                if _date != _prev_date:
                    _cur_ah = _cur_al = np.nan
                    _prev_date = _date
                if _is_asian_window[_i]:
                    _cur_ah = _high[_i] if np.isnan(_cur_ah) else max(_cur_ah, _high[_i])
                    _cur_al = _low[_i]  if np.isnan(_cur_al) else min(_cur_al, _low[_i])
                asian_high_arr[_i] = _cur_ah
                asian_low_arr[_i]  = _cur_al
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

        # ── Oscillators ───────────────────────────────────────────────────────
        from indicators.market_structure import compute_rsi as _rsi_fn
        _rsi_raw = _rsi_fn(out["close"], 14).fillna(50.0).to_numpy(dtype=np.float64)
        extra["rsi_extreme"] = np.clip((_rsi_raw - 50.0) / 50.0, -1.0, 1.0).astype(np.float32)
        _stoch_k, _stoch_d = compute_stochastic(out)
        extra["stoch_k"]     = np.clip(
            _stoch_k.fillna(50.0).to_numpy(dtype=np.float64) / 100.0, 0.0, 1.0
        ).astype(np.float32)
        extra["stoch_k_vs_d"] = np.clip(
            (_stoch_k - _stoch_d).fillna(0.0).to_numpy(dtype=np.float64) / 100.0, -1.0, 1.0
        ).astype(np.float32)

        # ── ADX on 15M ────────────────────────────────────────────────────────
        extra["adx_15m"] = np.clip(
            _compute_adx(out, 14).fillna(0.0).to_numpy(dtype=np.float64) / 100.0, 0.0, 1.0
        ).astype(np.float32)

        # ── Regime duration ───────────────────────────────────────────────────
        # Bars since last regime change; computed from regime_series if provided.
        if regime_series is not None:
            _reg = regime_series.reindex(out.index, method="ffill").fillna(2).to_numpy(dtype=np.int8)
            _dur = np.zeros(n, dtype=np.float32)
            _cnt = 0
            for _i in range(n):
                if _i > 0 and _reg[_i] != _reg[_i - 1]:
                    _cnt = 0
                _cnt += 1
                _dur[_i] = min(_cnt / 100.0, 1.0)
            extra["regime_duration"] = _dur
        else:
            extra["regime_duration"] = np.full(n, 0.5, dtype=np.float32)

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
        _atr_pctile = _atr_s.rolling(42, min_periods=14).apply(
            lambda x: float(np.searchsorted(np.sort(x[:-1]), x[-1])) / max(len(x) - 1, 1)
            if len(x) > 1 else 0.5, raw=True
        ).clip(0.0, 1.0).fillna(0.5)
        extra["atr_pctile"] = _atr_pctile.to_numpy(dtype=np.float32)

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
        # Drop duplicate columns — input df may already have pre-computed indicators
        # (e.g. stoch_k from _compute_indicators). Keep last occurrence so that the
        # normalised values computed above always win.
        if out.columns.duplicated().any():
            out = out.loc[:, ~out.columns.duplicated(keep="last")]
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

    # ─── Regime features (LightGBM input) ────────────────────────────────────

    def get_regime_features(
        self,
        df: pd.DataFrame,
        df_htf: Optional[Dict[str, pd.DataFrame]] = None,
        # Legacy keyword kept for backward compat — folded into df_htf["4H"]
        df_h4: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
    ) -> np.ndarray:
        """Returns shape (N,) float32. No NaN.

        df_htf: dict with keys "5M", "15M", "1H", "4H", "1D".
                Each TF contributes 4 features: adx, ema_stack, atr_ratio, bb_width.
                df is used for base structural features (session, swing count, sweeps, vol).
                df_h4 is accepted for backward compat and merged into df_htf["4H"].
        """
        if df is None or len(df) < 20:
            return np.zeros(len(REGIME_FEATURES), dtype=np.float32)

        # Normalise HTF dict — accept both dict and legacy single-df form
        htf: Dict[str, Optional[pd.DataFrame]] = {}
        if isinstance(df_htf, dict):
            htf.update(df_htf)
        # Legacy df_h4 fills in "4H" if not already present
        if df_h4 is not None and "4H" not in htf and "H4" not in htf:
            htf["4H"] = df_h4

        df_5m  = _first_present_frame(htf, "5M", "5m")
        df_15m = _first_present_frame(htf, "15M", "15m", default=df)   # default to input df
        df_1h  = _first_present_frame(htf, "1H", "H1")
        df_h4_ = _first_present_frame(htf, "4H", "H4")
        df_1d  = _first_present_frame(htf, "1D", "D1")

        feats = np.zeros(len(REGIME_FEATURES), dtype=np.float32)

        from indicators.market_structure import (
            compute_adx, compute_atr, compute_ema_stack_score,
            compute_bollinger_bands, detect_break_of_structure,
            detect_liquidity_sweeps,
        )

        # ── Base features (input df) ──────────────────────────────────────
        atr = compute_atr(df, 14)
        atr_val = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
        close   = float(df["close"].iloc[-1])

        adx = df["adx_14"] if "adx_14" in df.columns else compute_adx(df, 14)
        feats[0] = float(np.clip(adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0.0, 0, 100))

        stk = df["ema_stack"] if "ema_stack" in df.columns else _ema_stack_series(df)
        feats[1] = float(np.clip(stk.iloc[-1] if not pd.isna(stk.iloc[-1]) else 0.0, -2, 2))

        feats[2] = float(np.clip(atr_val / (close + 1e-9) * 1000, 0, 10))

        bb_width = df["bb_width"] if "bb_width" in df.columns else \
            (lambda u, m, l: (u - l) / (m + 1e-9))(*compute_bollinger_bands(df["close"]))
        feats[3] = float(np.clip(bb_width.iloc[-1] if not pd.isna(bb_width.iloc[-1]) else 0.0, 0, 0.1))

        returns = df["close"].pct_change()
        feats[4] = float(np.clip(returns.rolling(20).std().iloc[-1] * 100, 0, 5))

        if hasattr(df.index, "hour"):
            h = df.index[-1].hour
            feats[5] = 1.0 if 2 <= h < 7 else 2.0 if 7 <= h < 12 else \
                       4.0 if h == 12 else 3.0 if 13 <= h < 18 else 0.0

        if len(df) >= 20:
            bos = detect_break_of_structure(df.iloc[-40:] if len(df) >= 40 else df)
            feats[6] = float(np.clip(bos["bos_bull"].sum() + bos["bos_bear"].sum(), 0, 10))

        if len(df) >= 24:
            sw = detect_liquidity_sweeps(df.iloc[-48:] if len(df) >= 48 else df)
            feats[7] = float(np.clip(sw["sweep_bull"].sum() + sw["sweep_bear"].sum(), 0, 5))

        # ── Per-TF MTF features (4 features each: adx, ema_stack, atr_ratio, bb_width) ─
        # Offsets: 5M=8, 15M=12, 1H=16, 4H=20, 1D=24
        for offset, tf_df in [(8, df_5m), (12, df_15m), (16, df_1h), (20, df_h4_), (24, df_1d)]:
            adx_v, stk_v, atr_v, bb_v = _mtf_regime_features(tf_df)
            feats[offset]     = adx_v
            feats[offset + 1] = stk_v
            feats[offset + 2] = atr_v
            feats[offset + 3] = bb_v

        # ── S/R and Supply/Demand zone features (indices 28–33) ──────────
        # detect_sr_zones uses rolling(center=True) — lookahead. Zeroed until
        # the indicator is made strictly causal.
        # feats[28:34] already zero from np.zeros init above.

        # ── Regime dynamics (indices 34–36) ──────────────────────────────
        if len(atr) >= 15:
            rel_vol_now  = atr.iloc[-1]  / (df["close"].iloc[-1]  + 1e-9)
            rel_vol_prev = atr.iloc[-15] / (df["close"].iloc[-15] + 1e-9)
            feats[34] = float(np.clip((rel_vol_now - rel_vol_prev) * 1000, -5, 5))

        if len(df) >= 10:
            recent_close = df["close"].iloc[-50:] if len(df) >= 50 else df["close"]
            direction = np.sign(recent_close.diff().fillna(0))
            sign_changes = (direction != direction.shift(1)).sum()
            stability = float(np.clip(len(recent_close) - sign_changes, 0, 50))
            feats[35] = stability / 50.0

        # atr_pctile: ATR percentile rank in own 42-bar history [0→1]
        if len(atr) >= 14:
            _atr_window = atr.iloc[-42:].values if len(atr) >= 42 else atr.values
            _cur_atr = float(atr.iloc[-1])
            _sorted = np.sort(_atr_window[:-1])
            feats[36] = float(np.searchsorted(_sorted, _cur_atr)) / max(len(_sorted), 1)

        # ── Regime memory (indices 37–41) — prev_regime one-hot ─────────
        # Populated externally by callers that track regime state over time.
        # Defaults to RANGING (class 2) = [0, 0, 1, 0, 0] when not set.

        # ── Macro features (indices 43–61) ────────────────────────────────
        macro_df = self._build_macro_frame(df.index, symbol)
        base_macro = 43  # after 8 base + 5×4 MTF + 6 S/R + 3 regime dynamics + 5 prev_regime + 1 confidence
        for i, name in enumerate(INDEX_NAMES):
            feats[base_macro + i] = float(np.clip(macro_df[f"idx_{name}_ret"].iloc[-1] * 100, -5, 5))
        feats[base_macro + len(INDEX_NAMES)]     = float(np.clip(macro_df["macro_vix_level"].iloc[-1], 0, 2))
        feats[base_macro + len(INDEX_NAMES) + 1] = float(np.clip(macro_df["macro_yield_spread"].iloc[-1], -0.2, 0.4))

        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats.astype(np.float32)

    # ─── Quality features (XGBoost input) ────────────────────────────────────

    def get_quality_features(self, signal: dict, ml_base: dict, bar: pd.Series) -> np.ndarray:
        """
        Returns shape (N,) float32 where N = len(QUALITY_FEATURES).
        signal: dict with keys matching QUALITY_FEATURES where relevant.
        ml_base: dict with p_bull_gru, p_bear_gru, regime_class, sentiment_score,
                 expected_variance (GRU uncertainty), regime_duration, vol_slope.
        """
        feats = np.zeros(len(QUALITY_FEATURES), dtype=np.float32)

        strategy_map = {"trader_1": 0, "trader_2": 1, "trader_3": 2,
                        "trader_4": 3, "trader_5": 4}
        feats[0] = float(strategy_map.get(signal.get("trader_id", ""), 0))
        feats[1] = 1.0 if signal.get("side") == "buy" else 0.0
        feats[2] = float(np.clip(signal.get("rr_ratio", 1.5), 0, 10))
        feats[3] = float(np.clip(ml_base.get("p_bull_gru", 0.5), 0, 1))
        feats[4] = float(np.clip(ml_base.get("p_bear_gru", 0.5), 0, 1))

        regime_map = {
            # LTF behaviour classes
            "TRENDING": 0, "RANGING": 1, "CONSOLIDATING": 2, "VOLATILE": 3,
            # HTF bias classes (map to neutral/default)
            "BIAS_UP": 0, "BIAS_DOWN": 0, "BIAS_NEUTRAL": 1,
        }
        feats[5] = float(regime_map.get(ml_base.get("regime", "RANGING"), 1))
        feats[6] = float(np.clip(ml_base.get("sentiment_score", 0.0), -1, 1))

        feats[7] = float(np.clip(bar.get("adx_14", 20.0) if hasattr(bar, "get") else 20.0, 0, 100))
        feats[8] = float(np.clip(
            bar.get("atr_14", 0.001) / (bar.get("close", 1.0) + 1e-9) * 1000
            if hasattr(bar, "get") else 1.0, 0, 20
        ))
        feats[9] = float(np.clip(ml_base.get("volume_ratio", 1.0), 0, 5))
        feats[10] = float(np.clip(ml_base.get("spread_pips", 1.0), 0, 20))

        session_map = {"ASIAN": 1, "LONDON": 2, "NY": 3, "DEAD": 4, "INACTIVE": 0}
        feats[11] = float(session_map.get(ml_base.get("session", "INACTIVE"), 0))
        feats[12] = 1.0 if ml_base.get("news_in_30min", False) else 0.0

        win_rate = 0.5
        if self._journal is not None:
            stats = self._journal.get_rolling_stats(signal.get("trader_id", ""), n=20)
            win_rate = stats.get("win_rate", 0.5)
        feats[13] = float(np.clip(win_rate, 0, 1))

        # New features (indices 14–16)
        feats[14] = float(np.clip(ml_base.get("expected_variance", 0.1), 0.0, 5.0))
        feats[15] = float(np.clip(ml_base.get("regime_duration", 0.5), 0.0, 1.0))
        feats[16] = float(np.clip(ml_base.get("vol_slope", 0.0), -5.0, 5.0))

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
        Returns shape (42,) float32, clamped to [-10, 10].
        Never raises. All missing values default to 0.
        """
        state = np.zeros(RL_STATE_DIM, dtype=np.float32)

        def _safe(d: dict, key: str, default: float = 0.0) -> float:
            try:
                v = d.get(key, default)
                return float(v) if v is not None else default
            except Exception:
                return default

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

        # [0-5] ML predictions
        state[0] = _safe(ml_preds, "p_bull", 0.5)
        state[1] = _safe(ml_preds, "p_bear", 0.5)
        state[2] = _safe(ml_preds, "entry_depth", 0.3)
        # HTF bias encoding at state[3]: BIAS_UP=0.0, BIAS_DOWN=1.0, BIAS_NEUTRAL=2.0 → /2.0
        # LTF behaviour encoding at state[42]: TRENDING=0, RANGING=1, CONSOLIDATING=2, VOLATILE=3 → /3.0
        htf_regime_map = {
            "BIAS_UP": 0.0, "BIAS_DOWN": 1.0, "BIAS_NEUTRAL": 2.0,
        }
        state[3] = htf_regime_map.get(ml_preds.get("regime", "BIAS_NEUTRAL"), 2.0) / 2.0
        state[4] = _safe(ml_preds, "sentiment_score", 0.0)
        state[5] = _safe(ml_preds, "quality_score", 0.5)

        # [6-13] Market structure
        state[6] = _bar("adx_14", 20.0) / 100.0
        state[7] = _bar("ema_stack", 0.0) / 2.0
        close = _bar("close", 1.0)
        atr = _bar("atr_14", 0.001)
        state[8] = float(np.clip(atr / (close + 1e-9) * 1000, 0, 10))
        state[9] = _bar("bb_width", 0.0)
        state[10] = float(_bar("bos_bull", 0.0) > 0)
        state[11] = float(_bar("bos_bear", 0.0) > 0)
        state[12] = float(_bar("fvg_bull", 0.0) > 0)
        state[13] = float(_bar("fvg_bear", 0.0) > 0)

        # [14-18] Session context
        if hasattr(bar, "name") and hasattr(bar.name, "hour"):
            h = bar.name.hour
        else:
            h = -1
        state[14] = float(2 <= h < 7)    # asian
        state[15] = float(7 <= h < 12)   # london
        state[16] = float(13 <= h < 18)  # ny
        state[17] = float(h == 12)        # dead
        state[18] = _safe(ml_preds, "news_proximity", 0.0)

        # [19-23] Strategy signals (S1–S5 booleans)
        for i, tid in enumerate(["trader_1", "trader_2", "trader_3", "trader_4", "trader_5"]):
            state[19 + i] = 1.0 if signals.get(tid) else 0.0

        # [24-29] Portfolio state
        state[24] = float(np.clip(portfolio.get("open_positions", 0), 0, 10)) / 10.0
        state[25] = float(np.clip(portfolio.get("drawdown_pct", 0.0), 0, 0.2)) / 0.2
        state[26] = float(np.clip(portfolio.get("daily_pnl", 0.0), -500, 500)) / 500.0
        state[27] = float(np.clip(portfolio.get("trades_today", 0), 0, 20)) / 20.0
        state[28] = float(np.clip(portfolio.get("last_result", 0.0), -3, 3)) / 3.0
        state[29] = float(np.clip(portfolio.get("equity_norm", 1.0), 0, 2))

        # [30-33] Instrument one-hot
        idx = _INSTRUMENT_IDX.get(symbol, -1)
        if 0 <= idx <= 3:
            state[30 + idx] = 1.0

        # [34-41] ATR history ratios (8 lags — filled as 0 when not available from bar)
        # These are populated by SignalPipeline when it has historical df context
        # Default to 1.0 (neutral ratio)
        for i in range(8):
            state[34 + i] = float(np.clip(
                ml_preds.get(f"atr_lag_{_ATR_LAGS[i]}", 1.0), 0, 5
            ))

        # Clamp all values to [-10, 10]
        state = np.clip(state, -10.0, 10.0)
        state = np.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0)
        return state.astype(np.float32)
