"""
Causal regime score utilities shared by training, live inference, and tests.

The score frame intentionally uses only current and past bars. Supervised
training may combine these causal features with forward outcomes to create
targets, but these functions are safe for live feature generation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from indicators.market_structure import compute_atr, compute_bollinger_bands, compute_ema


LTF_SCORE_COLUMNS = [
    "trend_score",
    "range_score",
    "chop_score",
    "volatility_percentile",
    "consolidation_score",
]

REGIME_PRIMITIVE_COLUMNS = [
    "plus_di",
    "minus_di",
    "ema_20_slope",
    "ema_50_slope",
    "ema_200_slope",
    "ema_50_dist_atr",
    "ema_200_dist_atr",
    "atr_percentile_500",
    "rolling_vol_percentile",
    "bb_width_percentile",
    "rolling_range_percentile",
    "candle_body_ratio",
    "wick_ratio",
    "range_expansion_zscore",
    "hh_hl_structure",
    "lh_ll_structure",
    "symbol_group_code",
]

REGIME_SCORE_COLUMNS = REGIME_PRIMITIVE_COLUMNS + [
    "efficiency_ratio_20",
    "directional_bias_score",
    "bias_up_score",
    "bias_down_score",
    *LTF_SCORE_COLUMNS,
    "volatility_score",
]


def _clip01(value: Any) -> Any:
    return np.clip(value, 0.0, 1.0)


def symbol_group_code(symbol: str | None) -> float:
    """Stable scalar encoding for broad symbol group normalisation."""
    sym = (symbol or "").upper()
    if "XAU" in sym or "GOLD" in sym:
        return 1.00
    if "JPY" in sym:
        return 0.75
    if "USD" in sym:
        return 0.25
    if sym:
        return 0.50
    return 0.0


def efficiency_ratio(close: pd.Series, window: int = 20) -> pd.Series:
    """abs(net movement) / sum(abs(path movement)); high means clean direction."""
    close = pd.Series(close, dtype=float)
    net_change = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window, min_periods=window).sum()
    er = net_change / path.replace(0.0, np.nan)
    return er.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)


def rolling_percentile(
    series: pd.Series,
    window: int = 500,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Causal rolling percentile rank of the current value versus prior values.

    The current bar is ranked against the previous bars in the rolling window,
    not future bars. Initial rows return 0.5 until enough history exists.
    """
    s = pd.Series(series, dtype=float)
    arr = s.to_numpy(dtype=np.float64)
    n = len(arr)
    if n == 0:
        return pd.Series(dtype=float, index=s.index)
    min_p = min_periods if min_periods is not None else min(window, 50)
    min_p = max(2, int(min_p))
    eff_window = min(int(window), n)
    out = np.full(n, 0.5, dtype=np.float32)
    if n < min_p or eff_window < 2:
        return pd.Series(out, index=s.index)

    safe = (
        pd.Series(arr)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    pad = eff_window - 1
    padded = np.concatenate([np.full(pad, np.nan), safe])
    shape = (n, eff_window)
    strides = (padded.strides[0], padded.strides[0])
    wins = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    hist = wins[:, :-1]
    cur = wins[:, -1:]
    valid = np.isfinite(hist)
    denom = np.maximum(valid.sum(axis=1), 1)
    ranks = ((hist < cur) & valid).sum(axis=1)
    pct = np.clip(ranks / denom, 0.0, 1.0).astype(np.float32)
    out[min_p - 1:] = pct[min_p - 1:]
    return pd.Series(out, index=s.index)


def compute_adx_components(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Return ADX, +DI, and -DI with the same smoothing as compute_adx."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = df["close"].astype(float).shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm_s = pd.Series(plus_dm, index=df.index)
    minus_dm_s = pd.Series(minus_dm, index=df.index)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_smooth = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100.0 * plus_dm_s.ewm(span=period, adjust=False).mean() / (atr_smooth + 1e-9)
    minus_di = 100.0 * minus_dm_s.ewm(span=period, adjust=False).mean() / (atr_smooth + 1e-9)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx = dx.ewm(span=period, adjust=False).mean()
    return pd.DataFrame(
        {
            "adx_14": adx.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "plus_di": plus_di.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "minus_di": minus_di.replace([np.inf, -np.inf], np.nan).fillna(0.0),
        },
        index=df.index,
    )


def _ema_slope(close: pd.Series, span: int, atr: pd.Series, window: int) -> pd.Series:
    ema = compute_ema(close, span)
    return ((ema - ema.shift(window)) / (atr * max(window, 1) + 1e-9)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0).clip(-3.0, 3.0)


def build_regime_score_frame(
    df: pd.DataFrame,
    *,
    symbol: str | None = None,
    window: int = 20,
    percentile_window: int = 500,
) -> pd.DataFrame:
    """Build causal regime primitives, scores, and final score inputs."""
    if df is None or len(df) == 0:
        raise ValueError("build_regime_score_frame requires a non-empty OHLC dataframe")
    missing = {"open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"build_regime_score_frame missing required OHLC columns: {sorted(missing)}")

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)

    atr = compute_atr(df, 14).astype(float).replace(0.0, np.nan).ffill().bfill().fillna(1e-9)
    adx_df = compute_adx_components(df, 14)
    adx = adx_df["adx_14"]
    plus_di = adx_df["plus_di"]
    minus_di = adx_df["minus_di"]

    ema20 = compute_ema(close, 20)
    ema50 = compute_ema(close, 50)
    ema200 = compute_ema(close, 200)
    ema20_slope = _ema_slope(close, 20, atr, max(3, window // 4))
    ema50_slope = _ema_slope(close, 50, atr, max(5, window // 2))
    ema200_slope = _ema_slope(close, 200, atr, window)

    atr_pct = (atr / (close.abs() + 1e-9)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    atr_percentile = rolling_percentile(atr_pct, percentile_window, min_periods=min(100, percentile_window))
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rolling_vol = ret.rolling(window, min_periods=max(3, window // 2)).std().fillna(0.0)
    rolling_vol_percentile = rolling_percentile(
        rolling_vol, percentile_window, min_periods=min(100, percentile_window)
    )

    bb_u, bb_m, bb_l = compute_bollinger_bands(close)
    bb_width = ((bb_u - bb_l) / (bb_m.abs() + 1e-9)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    bb_width_percentile = rolling_percentile(bb_width, percentile_window, min_periods=min(100, percentile_window))

    rolling_range = (
        (high.rolling(window, min_periods=max(3, window // 2)).max()
         - low.rolling(window, min_periods=max(3, window // 2)).min())
        / (atr + 1e-9)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rolling_range_percentile = rolling_percentile(
        rolling_range, percentile_window, min_periods=min(100, percentile_window)
    )

    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0)
    tr_mean = true_range.rolling(50, min_periods=10).mean()
    tr_std = true_range.rolling(50, min_periods=10).std().replace(0.0, np.nan)
    range_expansion_z = ((true_range - tr_mean) / (tr_std + 1e-9)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0).clip(-5.0, 5.0)

    candle_range = (high - low).abs() + 1e-9
    body_ratio = ((close - open_).abs() / candle_range).clip(0.0, 1.0)
    upper_wick = (high - np.maximum(close, open_)).clip(lower=0.0)
    lower_wick = (np.minimum(close, open_) - low).clip(lower=0.0)
    wick_ratio = ((upper_wick + lower_wick) / candle_range).clip(0.0, 1.0)

    hh = high > high.shift(1).rolling(window, min_periods=max(3, window // 2)).max()
    hl = low > low.shift(window // 2).fillna(low)
    ll = low < low.shift(1).rolling(window, min_periods=max(3, window // 2)).min()
    lh = high < high.shift(window // 2).fillna(high)
    signed_structure = (
        (hh.astype(float) + hl.astype(float) - ll.astype(float) - lh.astype(float))
        .rolling(window, min_periods=1)
        .mean()
        .clip(-1.0, 1.0)
    )
    hh_hl_structure = signed_structure.clip(lower=0.0, upper=1.0)
    lh_ll_structure = (-signed_structure).clip(lower=0.0, upper=1.0)

    er = efficiency_ratio(close, window)
    di_total = plus_di + minus_di + 1e-9
    di_spread = ((plus_di - minus_di).abs() / di_total).clip(0.0, 1.0)
    adx_strength = ((adx - 18.0) / 14.0).clip(0.0, 1.0)
    adx_weak = (1.0 - (adx / 18.0).clip(0.0, 1.0)).clip(0.0, 1.0)
    er_trend = ((er - 0.25) / 0.35).clip(0.0, 1.0)
    er_chop = (1.0 - (er / 0.28).clip(0.0, 1.0)).clip(0.0, 1.0)
    ema_slope_strength = (ema50_slope.abs() / 0.08).clip(0.0, 1.0)
    ema_flat = (1.0 - (ema50_slope.abs() / 0.035).clip(0.0, 1.0)).clip(0.0, 1.0)
    ema_dist50 = ((close - ema50) / (atr + 1e-9)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10, 10)
    ema_dist200 = ((close - ema200) / (atr + 1e-9)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-20, 20)
    stack_direction = np.sign(ema_dist50) + np.sign(ema_dist200) + np.sign(ema50_slope)
    structure_strength = np.maximum(hh_hl_structure, lh_ll_structure)

    trend_score = _clip01(
        0.30 * adx_strength
        + 0.30 * er_trend
        + 0.20 * ema_slope_strength
        + 0.10 * di_spread
        + 0.10 * structure_strength
    )
    range_score = _clip01(
        0.30 * adx_weak
        + 0.30 * er_chop
        + 0.20 * ema_flat
        + 0.20 * (1.0 - (atr_percentile - 0.50).abs() / 0.50).clip(0.0, 1.0)
    )
    consolidation_score = _clip01(
        0.40 * (1.0 - (atr_percentile / 0.35).clip(0.0, 1.0))
        + 0.35 * (1.0 - (bb_width_percentile / 0.35).clip(0.0, 1.0))
        + 0.25 * (1.0 - (rolling_range_percentile / 0.35).clip(0.0, 1.0))
    )
    volatility_percentile = _clip01(
        0.40 * atr_percentile
        + 0.35 * rolling_vol_percentile
        + 0.25 * ((range_expansion_z - 1.0) / 2.0).clip(0.0, 1.0)
    )
    chop_score = _clip01(
        0.35 * er_chop
        + 0.25 * (1.0 - di_spread)
        + 0.20 * wick_ratio
        + 0.20 * (1.0 - ema_slope_strength)
    )

    plus_pressure = (plus_di / di_total).clip(0.0, 1.0)
    minus_pressure = (minus_di / di_total).clip(0.0, 1.0)
    price_above = ((ema_dist50 > 0).astype(float) + (ema_dist200 > 0).astype(float)) / 2.0
    price_below = ((ema_dist50 < 0).astype(float) + (ema_dist200 < 0).astype(float)) / 2.0
    up_slope = (ema50_slope > 0).astype(float)
    down_slope = (ema50_slope < 0).astype(float)
    bias_up_score = _clip01(
        0.30 * plus_pressure
        + 0.25 * adx_strength
        + 0.20 * price_above
        + 0.15 * up_slope
        + 0.10 * er_trend
    )
    bias_down_score = _clip01(
        0.30 * minus_pressure
        + 0.25 * adx_strength
        + 0.20 * price_below
        + 0.15 * down_slope
        + 0.10 * er_trend
    )
    directional_bias_score = (bias_up_score - bias_down_score).clip(-1.0, 1.0)

    out = pd.DataFrame(
        {
            "adx_14": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "ema_20_slope": ema20_slope,
            "ema_50_slope": ema50_slope,
            "ema_200_slope": ema200_slope,
            "ema_50_dist_atr": ema_dist50,
            "ema_200_dist_atr": ema_dist200,
            "atr_percentile_500": atr_percentile,
            "rolling_vol_percentile": rolling_vol_percentile,
            "bb_width_percentile": bb_width_percentile,
            "rolling_range_percentile": rolling_range_percentile,
            "candle_body_ratio": body_ratio,
            "wick_ratio": wick_ratio,
            "range_expansion_zscore": range_expansion_z,
            "hh_hl_structure": hh_hl_structure,
            "lh_ll_structure": lh_ll_structure,
            "symbol_group_code": np.full(len(df), symbol_group_code(symbol), dtype=np.float32),
            "efficiency_ratio_20": er,
            "directional_bias_score": directional_bias_score,
            "bias_up_score": bias_up_score,
            "bias_down_score": bias_down_score,
            "trend_score": trend_score,
            "range_score": range_score,
            "chop_score": chop_score,
            "volatility_percentile": volatility_percentile,
            "volatility_score": volatility_percentile,
            "consolidation_score": consolidation_score,
        },
        index=df.index,
    )
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


TRADEABILITY_STATES = frozenset({
    "TRADEABLE_UP",
    "TRADEABLE_DOWN",
    "WAIT_PULLBACK",
    "NO_TRADE_CHOP",
    "NO_TRADE_EXTREME_VOL",
    "NO_TRADE_UNCERTAIN",
})


def classify_tradeability_directional(
    ltf_trade_regime: str,
    htf_bias: str,
    *,
    side: str | None = None,
    directional_confidence: float | None = None,
    neutral_strong_confidence: float = 0.70,
) -> str:
    """
    Map (ltf_trade_regime, htf_bias) → direction-aware tradeability state.

    ltf_trade_regime: output of classify_trade_regime()
      e.g. TRADEABLE_TREND, TRADEABLE_TREND_HIGH_VOL, RANGE,
           CONSOLIDATION, UNCERTAIN, NO_TRADE_CHOP, NO_TRADE_EXTREME_VOL
    htf_bias: output of HTF regime classifier
      e.g. BIAS_UP, BIAS_DOWN, BIAS_NEUTRAL

    Returns one of:
      TRADEABLE_UP            — HTF up + LTF tradeable/range → buy-only
      TRADEABLE_DOWN          — HTF down + LTF tradeable/range → sell-only
      NO_TRADE_CHOP           — low-momentum choppy market
      NO_TRADE_EXTREME_VOL    — volatility too high to trade
      NO_TRADE_UNCERTAIN      — regime unclear without a strong LTF/GRU override
    """
    tr = str(ltf_trade_regime or "").upper()
    bias = str(htf_bias or "BIAS_NEUTRAL").upper()

    # Hard no-trade: volatility and chop override everything
    if tr == "NO_TRADE_EXTREME_VOL":
        return "NO_TRADE_EXTREME_VOL"
    if tr == "NO_TRADE_CHOP":
        return "NO_TRADE_CHOP"

    # Consolidation and unclear LTF → no edge
    if tr in ("CONSOLIDATION", "UNCERTAIN", ""):
        return "NO_TRADE_UNCERTAIN"

    # Neutral HTF bias should not be a hard veto. If LTF says the market is
    # tradeable-trending and the GRU side is strong, let the execution gate
    # validate structure/pullback instead of collapsing to NO_TRADE_UNCERTAIN.
    if bias == "BIAS_NEUTRAL":
        if tr in ("TRADEABLE_TREND", "TRADEABLE_TREND_HIGH_VOL"):
            side_s = str(side or "").lower()
            try:
                dir_conf = float(directional_confidence)
            except (TypeError, ValueError):
                dir_conf = 0.0
            if dir_conf >= float(neutral_strong_confidence):
                if side_s == "buy":
                    return "TRADEABLE_UP"
                if side_s == "sell":
                    return "TRADEABLE_DOWN"
        return "NO_TRADE_UNCERTAIN"

    # Trending LTF + aligned HTF bias → directional trade
    if tr in ("TRADEABLE_TREND", "TRADEABLE_TREND_HIGH_VOL"):
        if bias == "BIAS_UP":
            return "TRADEABLE_UP"
        if bias == "BIAS_DOWN":
            return "TRADEABLE_DOWN"

    # Range LTF + HTF bias → trade only in the bias direction
    # (buy range bottoms in uptrend context; sell range tops in downtrend context)
    if tr == "RANGE":
        if bias == "BIAS_UP":
            return "TRADEABLE_UP"
        if bias == "BIAS_DOWN":
            return "TRADEABLE_DOWN"

    return "NO_TRADE_UNCERTAIN"


def _entry_bar_value(bar: Any, key: str, default: Any = None) -> Any:
    try:
        return bar.get(key, default)
    except AttributeError:
        return default


def _entry_truthy(value: Any) -> bool:
    try:
        if value != value:
            return False
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def has_aligned_entry_structure(bar: Any, side: str) -> bool:
    """Return true when the current bar has causal structure for the entry side."""
    side_s = str(side or "").lower()
    if side_s not in {"buy", "sell"}:
        return False
    pullback_valid = _entry_truthy(_entry_bar_value(bar, "pullback_valid", False))
    pullback_side = str(_entry_bar_value(bar, "pullback_side", "") or "").lower()
    if side_s == "buy":
        return (
            (pullback_valid and pullback_side == "buy")
            or _entry_truthy(_entry_bar_value(bar, "bos_bull", False))
            or _entry_truthy(_entry_bar_value(bar, "fvg_bull", False))
            or _entry_truthy(_entry_bar_value(bar, "mss_bull", False))
        )
    return (
        (pullback_valid and pullback_side == "sell")
        or _entry_truthy(_entry_bar_value(bar, "bos_bear", False))
        or _entry_truthy(_entry_bar_value(bar, "fvg_bear", False))
        or _entry_truthy(_entry_bar_value(bar, "mss_bear", False))
    )


def upgrade_uncertain_trade_regime(
    trade_regime: str,
    *,
    side: str,
    confidence: float,
    bar: Any,
    regime_scores: Optional[dict[str, Any]] = None,
    min_confidence: float = 0.70,
    min_trend_score: float = 0.55,
    max_chop_score: float = 0.55,
    max_volatility_score: float = 0.85,
) -> str:
    """
    Promote borderline LTF uncertainty only when price action confirms the side.

    This is deliberately narrow: it does not rescue chop, extreme volatility,
    range, or consolidation. It only handles the common case where the LTF score
    head returns UNCERTAIN despite a trend-like score mix and a fresh aligned
    BOS/FVG/MSS/pullback bar.
    """
    tr = str(trade_regime or "").upper()
    if tr not in {"UNCERTAIN", ""}:
        return tr
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        conf = 0.0
    if conf < float(min_confidence):
        return tr or "UNCERTAIN"
    if not has_aligned_entry_structure(bar, side):
        return tr or "UNCERTAIN"

    scores = regime_scores or {}
    try:
        trend_score = float(scores.get("trend_score", 0.0))
        chop_score = float(scores.get("chop_score", 0.0))
        vol_score = float(scores.get("volatility_percentile", scores.get("volatility_score", 0.0)))
    except (TypeError, ValueError):
        return tr or "UNCERTAIN"
    if (
        trend_score >= float(min_trend_score)
        and chop_score <= float(max_chop_score)
        and vol_score <= float(max_volatility_score)
    ):
        return "TRADEABLE_TREND"
    return tr or "UNCERTAIN"


def classify_entry_readiness(
    tradeability: str,
    side: str,
    bar: Any,
    *,
    ltf_behaviour: str = "TRENDING",
    require_range: bool = True,
) -> str:
    """
    Convert direction-aware tradeability into entry readiness for the current bar.

    WAIT_PULLBACK means the broader context is tradable in this direction, but
    the current candle is not at an acceptable structure/range/pullback entry.
    """
    state = str(tradeability or "NO_TRADE_UNCERTAIN").upper()
    if state not in {"TRADEABLE_UP", "TRADEABLE_DOWN"}:
        return state if state in TRADEABILITY_STATES else "NO_TRADE_UNCERTAIN"

    side_s = str(side or "").lower()
    if state == "TRADEABLE_UP" and side_s != "buy":
        return "NO_TRADE_UNCERTAIN"
    if state == "TRADEABLE_DOWN" and side_s != "sell":
        return "NO_TRADE_UNCERTAIN"

    ltf = str(ltf_behaviour or "TRENDING").upper()
    if ltf in {"CONSOLIDATION", "CONSOLIDATING", "UNCERTAIN"}:
        return "NO_TRADE_UNCERTAIN"

    range_valid = _entry_truthy(_entry_bar_value(bar, "range_valid", False))
    range_side = str(_entry_bar_value(bar, "range_side", "") or "").lower()
    side_structure = has_aligned_entry_structure(bar, side_s)

    if ltf == "RANGING":
        if require_range and not range_valid:
            return "WAIT_PULLBACK"
        if range_valid and range_side and range_side != side_s:
            return "WAIT_PULLBACK"
        return state

    if ltf in {"TRENDING", "VOLATILE"} and not side_structure:
        return "WAIT_PULLBACK"

    return state


def classify_trade_regime(row: pd.Series | dict[str, Any]) -> str:
    """Convert score outputs into a final trading filter state."""
    get = row.get if hasattr(row, "get") else lambda key, default=0.0: default
    trend_score = float(get("trend_score", 0.0))
    range_score = float(get("range_score", 0.0))
    chop_score = float(get("chop_score", 0.0))
    vol_score = float(get("volatility_percentile", get("volatility_score", 0.0)))
    vol_pct = float(get("atr_percentile_500", get("atr_percentile", vol_score)))
    er = float(get("efficiency_ratio_20", get("efficiency_ratio", 0.0)))
    consolidation_score = float(get("consolidation_score", 0.0))

    if vol_pct > 0.90 or (vol_score > 0.90 and trend_score < 0.65):
        return "NO_TRADE_EXTREME_VOL"
    if chop_score > 0.65 and trend_score < 0.65:
        return "NO_TRADE_CHOP"
    if trend_score > 0.65 and er > 0.35:
        if vol_pct > 0.75 or vol_score > 0.75:
            return "TRADEABLE_TREND_HIGH_VOL"
        return "TRADEABLE_TREND"
    if range_score > 0.40 and trend_score < 0.45:
        return "RANGE"
    if consolidation_score > 0.65 or (vol_pct < 0.25 and er < 0.25):
        return "CONSOLIDATION"
    return "UNCERTAIN"


def legacy_ltf_label_from_scores(row: pd.Series | dict[str, Any]) -> str:
    """Backward-compatible 4-class behaviour label derived from score outputs."""
    state = classify_trade_regime(row)
    if state in {"TRADEABLE_TREND", "TRADEABLE_TREND_HIGH_VOL"}:
        return "TRENDING"
    if state == "RANGE":
        return "RANGING"
    if state == "CONSOLIDATION":
        return "CONSOLIDATING"
    if state == "NO_TRADE_EXTREME_VOL":
        return "VOLATILE"

    get = row.get if hasattr(row, "get") else lambda key, default=0.0: default
    scores = {
        "TRENDING": float(get("trend_score", 0.0)),
        "RANGING": float(get("range_score", 0.0)),
        "CONSOLIDATING": float(get("consolidation_score", 0.0)),
        "VOLATILE": float(get("volatility_percentile", get("volatility_score", 0.0))),
    }
    return max(scores, key=scores.get)


def legacy_ltf_labels_from_scores(score_df: pd.DataFrame) -> pd.Series:
    """Vector-friendly wrapper for deriving legacy LTF labels."""
    if score_df.empty:
        return pd.Series(dtype=object, index=score_df.index)
    return score_df.apply(legacy_ltf_label_from_scores, axis=1)
