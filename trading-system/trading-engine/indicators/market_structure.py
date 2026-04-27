"""
market_structure.py — Vectorized market structure indicators for Forex + Gold.

All functions accept a pandas DataFrame with columns: open, high, low, close, volume.
All functions are pure (no side effects). No .at[i] integer indexing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple

try:
    from numba import njit as _njit
    _NUMBA = True
except ImportError:
    def _njit(fn=None, **kw):  # type: ignore[return]
        return fn if fn is not None else (lambda f: f)
    _NUMBA = False


# ---------------------------------------------------------------------------
# Numba-accelerated helpers (fall back to plain Python when numba absent)
# ---------------------------------------------------------------------------

@_njit(cache=True)
def _cluster_zones_nb(levels: np.ndarray, merge_tol: float):
    """
    Greedy zone clustering.  Modifies levels in-place via output arrays.
    Returns (zone_levels, zone_counts) as 1-D float64 arrays.
    levels must be a 1-D float64 array with no NaNs.
    """
    max_zones = len(levels)
    z_lvl = np.empty(max_zones, dtype=np.float64)
    z_cnt = np.zeros(max_zones, dtype=np.int64)
    nz = 0
    for k in range(len(levels)):
        lvl = levels[k]
        placed = False
        for j in range(nz):
            if abs(lvl - z_lvl[j]) <= merge_tol:
                z_lvl[j] = (z_lvl[j] * z_cnt[j] + lvl) / (z_cnt[j] + 1)
                z_cnt[j] += 1
                placed = True
                break
        if not placed:
            z_lvl[nz] = lvl
            z_cnt[nz] = 1
            nz += 1
    return z_lvl[:nz], z_cnt[:nz]


@_njit(cache=True)
def _sr_zones_loop(
    n: int,
    start0: int,
    atr: np.ndarray,
    close: np.ndarray,
    swing_highs: np.ndarray,
    swing_lows: np.ndarray,
    lookback: int,
    atr_merge_ratio: float,
    nearest_resist: np.ndarray,
    nearest_support: np.ndarray,
    dist_resist: np.ndarray,
    dist_support: np.ndarray,
    in_supply: np.ndarray,
    in_demand: np.ndarray,
    resist_strength: np.ndarray,
    support_strength: np.ndarray,
):
    for i in range(start0, n):
        atr_i = atr[i]
        if atr_i != atr_i or atr_i <= 0:
            atr_i = 1e-5
        merge_tol = atr_i * atr_merge_ratio
        c = close[i]
        start = i - lookback if i - lookback > 0 else 0

        # collect valid swing highs / lows in window
        raw_h = swing_highs[start:i]
        raw_l = swing_lows[start:i]
        nh = 0
        nl = 0
        for v in raw_h:
            if v == v:  # not nan
                nh += 1
        for v in raw_l:
            if v == v:
                nl += 1
        vh = np.empty(nh, dtype=np.float64)
        vl = np.empty(nl, dtype=np.float64)
        ki = 0
        for v in raw_h:
            if v == v:
                vh[ki] = v
                ki += 1
        ki = 0
        for v in raw_l:
            if v == v:
                vl[ki] = v
                ki += 1

        r_lvl, r_cnt = _cluster_zones_nb(vh, merge_tol)
        s_lvl, s_cnt = _cluster_zones_nb(vl, merge_tol)

        # nearest resistance above close
        best_r = np.inf
        best_rc = 1
        for j in range(len(r_lvl)):
            if r_lvl[j] > c and r_lvl[j] < best_r:
                best_r = r_lvl[j]
                best_rc = r_cnt[j]
        if best_r < np.inf:
            nearest_resist[i] = best_r
            d = (best_r - c) / (atr_i + 1e-9)
            dist_resist[i] = d if d < 10 else 10.0
            resist_strength[i] = float(best_rc if best_rc < 5 else 5)
            if (best_r - c) <= merge_tol:
                in_supply[i] = 1.0

        # nearest support below close
        best_s = -np.inf
        best_sc = 1
        for j in range(len(s_lvl)):
            if s_lvl[j] < c and s_lvl[j] > best_s:
                best_s = s_lvl[j]
                best_sc = s_cnt[j]
        if best_s > -np.inf:
            nearest_support[i] = best_s
            d = (c - best_s) / (atr_i + 1e-9)
            dist_support[i] = d if d < 10 else 10.0
            support_strength[i] = float(best_sc if best_sc < 5 else 5)
            if (c - best_s) <= merge_tol:
                in_demand[i] = 1.0


@_njit(cache=True)
def _range_loop(
    n: int,
    start0: int,
    atr: np.ndarray,
    close: np.ndarray,
    swing_highs: np.ndarray,
    swing_lows: np.ndarray,
    lookback: int,
    min_width_atr: float,
    min_touches: int,
    boundary_atr: float,
    range_valid: np.ndarray,
    range_side: np.ndarray,   # int8: 0=none, 1=buy, -1=sell
    range_support: np.ndarray,
    range_resist: np.ndarray,
    range_width_atr: np.ndarray,
):
    merge_tol_ratio = 0.5
    for i in range(start0, n):
        atr_i = atr[i]
        if atr_i != atr_i or atr_i <= 1e-9:
            continue
        c = close[i]
        start = i - lookback if i - lookback > 0 else 0
        merge_tol = atr_i * merge_tol_ratio

        raw_h = swing_highs[start:i]
        raw_l = swing_lows[start:i]
        nh = 0
        nl = 0
        for v in raw_h:
            if v == v:
                nh += 1
        for v in raw_l:
            if v == v:
                nl += 1
        if nh == 0 or nl == 0:
            continue

        vh = np.empty(nh, dtype=np.float64)
        vl = np.empty(nl, dtype=np.float64)
        ki = 0
        for v in raw_h:
            if v == v:
                vh[ki] = v
                ki += 1
        ki = 0
        for v in raw_l:
            if v == v:
                vl[ki] = v
                ki += 1

        r_lvl, r_cnt = _cluster_zones_nb(vh, merge_tol)
        s_lvl, s_cnt = _cluster_zones_nb(vl, merge_tol)

        # nearest qualified resistance above close
        best_r = np.inf
        for j in range(len(r_lvl)):
            if r_lvl[j] > c and r_cnt[j] >= min_touches and r_lvl[j] < best_r:
                best_r = r_lvl[j]
        if best_r == np.inf:
            continue

        # nearest qualified support below close
        best_s = -np.inf
        for j in range(len(s_lvl)):
            if s_lvl[j] < c and s_cnt[j] >= min_touches and s_lvl[j] > best_s:
                best_s = s_lvl[j]
        if best_s == -np.inf:
            continue

        width_atr = (best_r - best_s) / atr_i
        if width_atr < min_width_atr:
            continue

        near_r = (best_r - c) <= boundary_atr * atr_i
        near_s = (c - best_s) <= boundary_atr * atr_i
        if not (near_r or near_s):
            continue

        range_valid[i] = 1
        range_support[i] = best_s
        range_resist[i] = best_r
        range_width_atr[i] = width_atr
        if near_s:
            range_side[i] = 1    # buy
        else:
            range_side[i] = -1   # sell


@_njit(cache=True)
def _pullback_state_loop(
    n: int,
    start0: int,
    atr: np.ndarray,
    close: np.ndarray,
    swing_high_arr: np.ndarray,
    swing_low_arr: np.ndarray,
    pullback_atr: float,
    retest_atr: float,
    pullback_valid: np.ndarray,
    pullback_side: np.ndarray,    # int8: 0=none, 1=buy, -1=sell
    pullback_level: np.ndarray,
    pb_swing_high: np.ndarray,
    pb_swing_low: np.ndarray,
):
    last_sh = np.nan
    last_sl = np.nan
    prev_sh = np.nan
    prev_sl = np.nan
    for i in range(start0, n):
        sh = swing_high_arr[i]
        sl = swing_low_arr[i]
        if sh == sh:  # not nan
            prev_sh = last_sh
            last_sh = sh
        if sl == sl:
            prev_sl = last_sl
            last_sl = sl

        pb_swing_high[i] = last_sh
        pb_swing_low[i] = last_sl

        atr_i = atr[i]
        if atr_i != atr_i or atr_i <= 1e-9:
            continue
        c = close[i]

        lsh = last_sh; psh = prev_sh; lsl = last_sl; psl = prev_sl
        if (lsh == lsh and psh == psh and lsl == lsl and psl == psl
                and lsh > psh and lsl > psl
                and c < lsh
                and abs(c - lsl) <= pullback_atr * atr_i
                and c > lsl - retest_atr * atr_i):
            pullback_valid[i] = 1
            pullback_side[i] = 1   # buy
            pullback_level[i] = lsl
        elif (lsl == lsl and psl == psl and lsh == lsh and psh == psh
                and lsl < psl and lsh < psh
                and c > lsl
                and abs(c - lsh) <= pullback_atr * atr_i
                and c < lsh + retest_atr * atr_i):
            pullback_valid[i] = 1
            pullback_side[i] = -1  # sell
            pullback_level[i] = lsh


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing (ewm span=period)."""
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    prev_close = df["close"].shift(1).to_numpy()

    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ),
    ).astype(np.float32)

    return pd.Series(tr, index=df.index).ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI via Wilder's smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ADX — computes +DI, -DI, and returns ADX series."""
    high = df["high"]
    low = df["low"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = df["close"].shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=df.index)
    minus_dm_s = pd.Series(minus_dm, index=df.index)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_smooth = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm_s.ewm(span=period, adjust=False).mean() / (atr_smooth + 1e-9)
    minus_di = 100 * minus_dm_s.ewm(span=period, adjust=False).mean() / (atr_smooth + 1e-9)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx


def compute_stochastic(
    df: pd.DataFrame, k: int = 14, smooth_k: int = 3, d: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """Stochastic %K and %D. Returns (stoch_k, stoch_d)."""
    low_min = df["low"].rolling(k).min()
    high_max = df["high"].rolling(k).max()

    raw_k = 100.0 * (df["close"] - low_min) / (high_max - low_min + 1e-9)
    stoch_k = raw_k.rolling(smooth_k).mean()
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d


def compute_bollinger_bands(
    series: pd.Series, period: int = 20, std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, middle, lower)."""
    middle = series.rolling(period).mean()
    rolling_std = series.rolling(period).std()
    upper = middle + std * rolling_std
    lower = middle - std * rolling_std
    return upper, middle, lower


def compute_ema_stack_score(df: pd.DataFrame) -> pd.Series:
    """
    EMA alignment score.
    +2: ema21 > ema50 > ema200 (bull trend)
    -2: ema21 < ema50 < ema200 (bear trend)
     0: mixed / transitioning
    """
    ema21 = compute_ema(df["close"], 21)
    ema50 = compute_ema(df["close"], 50)
    ema200 = compute_ema(df["close"], 200)

    bull = (ema21 > ema50) & (ema50 > ema200)
    bear = (ema21 < ema50) & (ema50 < ema200)

    score = pd.Series(0, index=df.index, dtype=int)
    score = score.where(~bull, 2)
    score = score.where(~bear, -2)
    return score


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD line, signal line, and histogram.
    Returns (macd_line, signal_line, histogram).
    histogram > 0 → bullish momentum; < 0 → bearish momentum.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _confirmed_swing_arrays(
    high: np.ndarray,
    low: np.ndarray,
    swing_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return swing levels at the bar where they become knowable.

    A swing at bar j needs swing_n bars to its right before it is confirmed.
    Therefore the level is emitted at j + swing_n, never at j. Downstream
    consumers see only confirmed, causal structure.
    """
    n = len(high)
    swing_high = np.full(n, np.nan, dtype=np.float64)
    swing_low = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return swing_high, swing_low

    swing_n = max(1, int(swing_n))
    window = 2 * swing_n + 1
    high_s = pd.Series(high, dtype=np.float64)
    low_s = pd.Series(low, dtype=np.float64)

    roll_high = high_s.rolling(window, min_periods=window).max()
    roll_low = low_s.rolling(window, min_periods=window).min()
    mid_high = high_s.shift(swing_n)
    mid_low = low_s.shift(swing_n)

    confirmed_high = mid_high.where(mid_high == roll_high)
    confirmed_low = mid_low.where(mid_low == roll_low)
    swing_high[:] = confirmed_high.to_numpy(dtype=np.float64)
    swing_low[:] = confirmed_low.to_numpy(dtype=np.float64)
    return swing_high, swing_low


def detect_fair_value_gaps(df: pd.DataFrame, min_atr_ratio: float = 0.25) -> pd.DataFrame:
    """
    Detect 3-candle Fair Value Gaps (vectorized).
    Bullish FVG:  candle[i-2].high < candle[i].low  (gap above)
    Bearish FVG:  candle[i-2].low  > candle[i].high (gap below)

    Returns DataFrame with columns:
      fvg_bull (bool), fvg_bear (bool),
      fvg_bull_top (float), fvg_bull_bottom (float),
      fvg_bear_top (float), fvg_bear_bottom (float)
    """
    atr = compute_atr(df, 14)
    min_gap = atr * min_atr_ratio

    high_2 = df["high"].shift(2)   # candle[i-2].high
    low_2 = df["low"].shift(2)     # candle[i-2].low

    fvg_bull = (high_2 < df["low"]) & ((df["low"] - high_2) > min_gap)
    fvg_bear = (low_2 > df["high"]) & ((low_2 - df["high"]) > min_gap)

    result = pd.DataFrame(index=df.index)
    result["fvg_bull"] = fvg_bull.fillna(False)
    result["fvg_bear"] = fvg_bear.fillna(False)
    result["fvg_bull_bottom"] = high_2.where(fvg_bull)
    result["fvg_bull_top"] = df["low"].where(fvg_bull)
    result["fvg_bear_top"] = low_2.where(fvg_bear)
    result["fvg_bear_bottom"] = df["high"].where(fvg_bear)
    return result


def detect_break_of_structure(df: pd.DataFrame, swing_n: int = 5) -> pd.DataFrame:
    """
    Vectorized BOS detection using causally confirmed swing highs/lows.

    Returns DataFrame with:
      swing_high (float, NaN where not swing), swing_low (float, NaN),
      bos_bull (bool), bos_bear (bool),
      last_swing_high (float, forward-filled), last_swing_low (float, forward-filled)
    """
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    swing_high_arr, swing_low_arr = _confirmed_swing_arrays(high, low, swing_n)

    swing_high = pd.Series(swing_high_arr, index=df.index)
    swing_low = pd.Series(swing_low_arr, index=df.index)

    last_swing_high = swing_high.ffill()
    last_swing_low = swing_low.ffill()

    # BOS bull: close breaks above last swing high
    bos_bull = (df["close"] > last_swing_high.shift(1)) & \
               (df["close"].shift(1) <= last_swing_high.shift(1))
    # BOS bear: close breaks below last swing low
    bos_bear = (df["close"] < last_swing_low.shift(1)) & \
               (df["close"].shift(1) >= last_swing_low.shift(1))

    result = pd.DataFrame(index=df.index)
    result["swing_high"] = swing_high
    result["swing_low"] = swing_low
    result["bos_bull"] = bos_bull.fillna(False)
    result["bos_bear"] = bos_bear.fillna(False)
    result["last_swing_high"] = last_swing_high
    result["last_swing_low"] = last_swing_low
    return result


def detect_liquidity_sweeps(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Detect high-quality stop hunts: price sweeps a key level then closes
    with a strong recovery body (≥50% of the wick recovered).

    Quality filter: the close must recover at least halfway back from the
    swept low to the bar's open. A 1-pip recovery above the level is not
    enough — we need a genuine reversal body to filter out weak continuation
    candles that barely close above the level before pushing lower next bar.

    Returns DataFrame with:
      sweep_bull (bool): swept below recent low, close recovered ≥50% of wick
      sweep_bear (bool): swept above recent high, close dropped ≥50% of wick
      sweep_low_level (float): level swept
      sweep_high_level (float): level swept
      sweep_bull_wick (float): total wick size (close - low)
      sweep_bear_wick (float): total wick size (high - close)
    """
    recent_high = df["high"].rolling(lookback).max().shift(1).astype("float32")
    recent_low = df["low"].rolling(lookback).min().shift(1).astype("float32")

    # Bullish sweep: wick pierces below recent_low, close back above it
    low = df["low"].to_numpy(dtype=np.float32, copy=False)
    high = df["high"].to_numpy(dtype=np.float32, copy=False)
    close = df["close"].to_numpy(dtype=np.float32, copy=False)
    recent_low_arr = recent_low.to_numpy(dtype=np.float32, copy=False)
    recent_high_arr = recent_high.to_numpy(dtype=np.float32, copy=False)

    wick_pierced_bull = low < recent_low_arr
    close_above_level = close > recent_low_arr

    # Quality: close must recover ≥50% of the bar's range back toward open
    # Recovery ratio = (close - low) / (high - low), must be ≥ 0.50
    bar_range = np.clip(high - low, 1e-9, None).astype(np.float32, copy=False)
    recovery_bull = (close - low) / bar_range
    recovery_bear = (high - close) / bar_range

    sweep_bull = wick_pierced_bull & close_above_level & (recovery_bull >= 0.50)

    # Bearish sweep: wick pierces above recent_high, close back below it
    wick_pierced_bear = high > recent_high_arr
    close_below_level = close < recent_high_arr
    sweep_bear = wick_pierced_bear & close_below_level & (recovery_bear >= 0.50)

    result = pd.DataFrame(index=df.index)
    sweep_bull = np.asarray(sweep_bull, dtype=bool)
    sweep_bear = np.asarray(sweep_bear, dtype=bool)
    result["sweep_bull"] = sweep_bull
    result["sweep_bear"] = sweep_bear

    sweep_low_level = np.full(len(df), np.nan, dtype=np.float32)
    sweep_low_level[sweep_bull] = recent_low_arr[sweep_bull]
    result["sweep_low_level"] = sweep_low_level

    sweep_high_level = np.full(len(df), np.nan, dtype=np.float32)
    sweep_high_level[sweep_bear] = recent_high_arr[sweep_bear]
    result["sweep_high_level"] = sweep_high_level

    sweep_bull_wick = np.full(len(df), np.nan, dtype=np.float32)
    sweep_bull_wick[sweep_bull] = (close - low)[sweep_bull]
    result["sweep_bull_wick"] = sweep_bull_wick

    sweep_bear_wick = np.full(len(df), np.nan, dtype=np.float32)
    sweep_bear_wick[sweep_bear] = (high - close)[sweep_bear]
    result["sweep_bear_wick"] = sweep_bear_wick
    return result


def detect_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Last opposing candle before an impulse move (vectorized).

    Bullish OB: last bearish candle before a strong bullish move (close[i+1] >> open[i])
    Bearish OB: last bullish candle before a strong bearish move

    Returns: ob_bull (bool), ob_bear (bool), ob_high (float), ob_low (float)
    """
    atr = compute_atr(df, 14)
    body = (df["close"] - df["open"]).abs()

    # Use next bar to confirm impulse but shift(-1) is lookahead — use
    # a 2-bar forward confirmed version that's also safe: if current bar
    # itself is a strong impulse candle, it is an OB candidate on the
    # *prior* bar, so we shift the impulse signal by +1 bar (backwards).
    impulse_up = (df["close"] - df["open"]) > atr * 1.0
    impulse_dn = (df["open"] - df["close"]) > atr * 1.0
    # OB is the bar *before* the impulse
    impulse_up = impulse_up.shift(1, fill_value=False).astype(bool)
    impulse_dn = impulse_dn.shift(1, fill_value=False).astype(bool)

    bearish_candle = df["close"] < df["open"]
    bullish_candle = df["close"] > df["open"]

    ob_bull = bearish_candle & impulse_up & (body > atr * 0.3)
    ob_bear = bullish_candle & impulse_dn & (body > atr * 0.3)

    result = pd.DataFrame(index=df.index)
    result["ob_bull"] = ob_bull.fillna(False)
    result["ob_bear"] = ob_bear.fillna(False)
    result["ob_high"] = df["high"].where(ob_bull | ob_bear)
    result["ob_low"] = df["low"].where(ob_bull | ob_bear)
    return result


def detect_sr_zones(
    df: pd.DataFrame,
    swing_n: int = 5,
    atr_merge_ratio: float = 0.5,
    lookback: int = 200,
) -> pd.DataFrame:
    """
    Identify significant Support/Resistance and Supply/Demand zones.

    Method: cluster swing highs (resistance/supply) and swing lows (support/demand)
    within ATR-distance tolerance. Each zone is characterised by:
      - how many times price has touched/respected it (strength)
      - how recently it was last tested (recency)
      - whether price is currently inside it

    Returns DataFrame with one row per input bar:
      sr_nearest_resist      (float) — price of nearest resistance above close
      sr_nearest_support     (float) — price of nearest support below close
      sr_dist_resist_atr     (float) — distance to nearest resistance in ATR units
      sr_dist_support_atr    (float) — distance to nearest support in ATR units
      sr_in_supply_zone      (float) — 1.0 if close is inside a supply/OB zone, else 0
      sr_in_demand_zone      (float) — 1.0 if close is inside a demand/OB zone, else 0
      sr_resist_strength     (float) — touch count of nearest resistance (capped 1-5)
      sr_support_strength    (float) — touch count of nearest support (capped 1-5)
    """
    result = pd.DataFrame(index=df.index)
    n = len(df)

    atr = compute_atr(df, 14).to_numpy(dtype=np.float32)
    close = df["close"].to_numpy(dtype=np.float32)
    high = df["high"].to_numpy(dtype=np.float32)
    low = df["low"].to_numpy(dtype=np.float32)

    swing_highs, swing_lows = _confirmed_swing_arrays(
        high.astype(np.float64),
        low.astype(np.float64),
        swing_n,
    )

    # Output arrays (float64 for numba; cast to float32 at assignment)
    nearest_resist = np.full(n, np.nan, dtype=np.float64)
    nearest_support = np.full(n, np.nan, dtype=np.float64)
    dist_resist = np.zeros(n, dtype=np.float64)
    dist_support = np.zeros(n, dtype=np.float64)
    in_supply = np.zeros(n, dtype=np.float64)
    in_demand = np.zeros(n, dtype=np.float64)
    resist_strength = np.zeros(n, dtype=np.float64)
    support_strength = np.zeros(n, dtype=np.float64)

    _sr_zones_loop(
        n, swing_n * 2,
        atr.astype(np.float64), close.astype(np.float64),
        swing_highs.astype(np.float64), swing_lows.astype(np.float64),
        lookback, atr_merge_ratio,
        nearest_resist, nearest_support,
        dist_resist, dist_support,
        in_supply, in_demand,
        resist_strength, support_strength,
    )

    result["sr_nearest_resist"]   = nearest_resist
    result["sr_nearest_support"]  = nearest_support
    result["sr_dist_resist_atr"]  = dist_resist
    result["sr_dist_support_atr"] = dist_support
    result["sr_in_supply_zone"]   = in_supply
    result["sr_in_demand_zone"]   = in_demand
    result["sr_resist_strength"]  = resist_strength
    result["sr_support_strength"] = support_strength
    return result


def detect_trending_pullback(
    df: pd.DataFrame,
    swing_n: int = 5,
    pullback_atr: float = 1.5,
    retest_atr: float = 0.75,
) -> pd.DataFrame:
    """
    Detect valid pullback/retest zones for TRENDING entries.

    Uptrend structure (HH + HL): price has made a Higher High followed by a Higher
    Low. A buy pullback is valid when price retraces back toward the last confirmed
    HL and is within `pullback_atr` × ATR of that level — i.e., retesting the HL
    before the next leg up.

    Downtrend structure (LH + LL): price has made a Lower High followed by a Lower
    Low. A sell pullback is valid when price retraces back up toward the last
    confirmed LH and is within `pullback_atr` × ATR of that level.

    Both cases are tightened by a `retest_atr` confirmation: the last swing must
    have been established at least `swing_n` bars ago (not the current bar) so the
    retest is genuinely causal.

    Returns DataFrame with columns:
      pullback_valid  (bool)  — price is in a valid pullback zone
      pullback_side   (str)   — "buy" (at HL retest) or "sell" (at LH retest), "" otherwise
      pullback_level  (float) — the HL or LH level being retested
      pb_swing_high   (float) — most recent confirmed swing high (for context)
      pb_swing_low    (float) — most recent confirmed swing low (for context)
    """
    n = len(df)
    atr = compute_atr(df, 14).to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    high  = df["high"].to_numpy(dtype=np.float64)
    low   = df["low"].to_numpy(dtype=np.float64)

    # Causal swing detection: vectorized via rolling — swing high at mid of window
    # confirmed only after swing_n bars have passed (no center lookahead).
    _w = swing_n * 2
    _high_s = pd.Series(high, dtype=np.float64)
    _low_s  = pd.Series(low,  dtype=np.float64)
    _roll_max = _high_s.rolling(_w).max().to_numpy()   # max of [i-_w+1 .. i]
    _roll_min = _low_s.rolling(_w).min().to_numpy()
    # mid bar = i - swing_n; confirmed at i = mid + swing_n
    _mid_high = np.roll(high, swing_n)                 # high[i - swing_n]
    _mid_low  = np.roll(low,  swing_n)
    # mark positions where mid bar was the window max/min
    swing_high_arr = np.where(_mid_high == _roll_max, _mid_high, np.nan).astype(np.float64)
    swing_low_arr  = np.where(_mid_low  == _roll_min, _mid_low,  np.nan).astype(np.float64)
    # first _w positions are unreliable
    swing_high_arr[:_w] = np.nan
    swing_low_arr[:_w]  = np.nan

    _pb_valid_nb = np.zeros(n, dtype=np.int8)
    _pb_side_nb  = np.zeros(n, dtype=np.int8)
    pullback_level = np.full(n, np.nan, dtype=np.float64)
    pb_swing_high  = np.full(n, np.nan, dtype=np.float64)
    pb_swing_low   = np.full(n, np.nan, dtype=np.float64)

    _pullback_state_loop(
        n, _w,
        atr, close,
        swing_high_arr, swing_low_arr,
        pullback_atr, retest_atr,
        _pb_valid_nb, _pb_side_nb,
        pullback_level, pb_swing_high, pb_swing_low,
    )

    _side_map = {0: "", 1: "buy", -1: "sell"}
    pullback_side = np.array([_side_map[int(v)] for v in _pb_side_nb], dtype=object)

    result = pd.DataFrame(index=df.index)
    result["pullback_valid"]  = _pb_valid_nb.astype(bool)
    result["pullback_side"]   = pullback_side
    result["pullback_level"]  = pullback_level
    result["pb_swing_high"]   = pb_swing_high
    result["pb_swing_low"]    = pb_swing_low
    return result


def detect_significant_range(
    df: pd.DataFrame,
    swing_n: int = 5,
    lookback: int = 100,
    min_width_atr: float = 2.0,
    min_touches: int = 2,
    boundary_atr: float = 0.5,
) -> pd.DataFrame:
    """
    Identify bars where price is inside a significant, clearly-defined range and
    sitting at one of its boundaries — the only condition under which RANGING entries
    are valid.

    A range is "significant and clear" when:
      1. There is a resistance level above AND a support level below within `lookback` bars
      2. Both levels have been touched at least `min_touches` times (clustered swings)
      3. The gap between them is at least `min_width_atr` × ATR  (room for a trade)
      4. Price is within `boundary_atr` × ATR of either level (entry near the wall, not the middle)

    Returns DataFrame with columns:
      range_valid    (bool)   — all four conditions satisfied
      range_side     (str)    — "buy" (at support), "sell" (at resistance), "" (middle/invalid)
      range_support  (float)  — support level of the detected range
      range_resist   (float)  — resistance level of the detected range
      range_width_atr (float) — range width in ATR units
    """
    n = len(df)
    atr = compute_atr(df, 14).to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    high  = df["high"].to_numpy(dtype=np.float64)
    low   = df["low"].to_numpy(dtype=np.float64)

    # Confirm swing levels only after swing_n right-side bars have closed.
    swing_highs, swing_lows = _confirmed_swing_arrays(high, low, swing_n)

    _range_valid_nb  = np.zeros(n, dtype=np.int8)
    _range_side_nb   = np.zeros(n, dtype=np.int8)
    range_support    = np.full(n, np.nan, dtype=np.float64)
    range_resist     = np.full(n, np.nan, dtype=np.float64)
    range_width_atr  = np.zeros(n, dtype=np.float64)

    _range_loop(
        n, swing_n * 2,
        atr, close,
        swing_highs.astype(np.float64), swing_lows.astype(np.float64),
        lookback, min_width_atr, min_touches, boundary_atr,
        _range_valid_nb, _range_side_nb,
        range_support, range_resist, range_width_atr,
    )

    _side_map = {0: "", 1: "buy", -1: "sell"}
    range_side_arr = np.array([_side_map[int(v)] for v in _range_side_nb], dtype=object)

    result = pd.DataFrame(index=df.index)
    result["range_valid"]     = _range_valid_nb.astype(bool)
    result["range_side"]      = range_side_arr
    result["range_support"]   = range_support
    result["range_resist"]    = range_resist
    result["range_width_atr"] = range_width_atr
    return result


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function. Adds all indicator columns to df and returns it.

    Column naming convention:
      ema_21, ema_50, ema_200, atr_14, rsi_14, adx_14,
      stoch_k, stoch_d, bb_upper, bb_mid, bb_lower, bb_width,
      macd_line, macd_signal, macd_hist,
      ema_stack, fvg_bull, fvg_bear, bos_bull, bos_bear,
      sweep_bull, sweep_bear, ob_bull, ob_bear
    """
    out = df.copy()

    out["ema_21"] = compute_ema(df["close"], 21)
    out["ema_50"] = compute_ema(df["close"], 50)
    out["ema_200"] = compute_ema(df["close"], 200)
    out["atr_14"] = compute_atr(df, 14)
    out["rsi_14"] = compute_rsi(df["close"], 14)
    out["adx_14"] = compute_adx(df, 14)

    stoch_k, stoch_d = compute_stochastic(df)
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_d

    bb_upper, bb_mid, bb_lower = compute_bollinger_bands(df["close"])
    out["bb_upper"] = bb_upper
    out["bb_mid"] = bb_mid
    out["bb_lower"] = bb_lower
    out["bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-9)

    macd_line, macd_sig, macd_hist = compute_macd(df["close"])
    out["macd_line"]   = macd_line
    out["macd_signal"] = macd_sig
    out["macd_hist"]   = macd_hist

    out["ema_stack"] = compute_ema_stack_score(df)

    fvg = detect_fair_value_gaps(df)
    out["fvg_bull"] = fvg["fvg_bull"]
    out["fvg_bear"] = fvg["fvg_bear"]
    out["fvg_bull_top"] = fvg["fvg_bull_top"]
    out["fvg_bull_bottom"] = fvg["fvg_bull_bottom"]
    out["fvg_bear_top"] = fvg["fvg_bear_top"]
    out["fvg_bear_bottom"] = fvg["fvg_bear_bottom"]

    bos = detect_break_of_structure(df)
    out["bos_bull"] = bos["bos_bull"]
    out["bos_bear"] = bos["bos_bear"]
    out["swing_high"] = bos["swing_high"]
    out["swing_low"] = bos["swing_low"]
    out["last_swing_high"] = bos["last_swing_high"]
    out["last_swing_low"] = bos["last_swing_low"]

    sweeps = detect_liquidity_sweeps(df)
    out["sweep_bull"] = sweeps["sweep_bull"]
    out["sweep_bear"] = sweeps["sweep_bear"]
    out["sweep_low_level"] = sweeps["sweep_low_level"]
    out["sweep_high_level"] = sweeps["sweep_high_level"]

    obs = detect_order_blocks(df)
    out["ob_bull"] = obs["ob_bull"]
    out["ob_bear"] = obs["ob_bear"]
    out["ob_high"] = obs["ob_high"]
    out["ob_low"] = obs["ob_low"]

    sr = detect_sr_zones(df)
    out["sr_nearest_resist"]   = sr["sr_nearest_resist"]
    out["sr_nearest_support"]  = sr["sr_nearest_support"]
    out["sr_dist_resist_atr"]  = sr["sr_dist_resist_atr"]
    out["sr_dist_support_atr"] = sr["sr_dist_support_atr"]
    out["sr_in_supply_zone"]   = sr["sr_in_supply_zone"]
    out["sr_in_demand_zone"]   = sr["sr_in_demand_zone"]
    out["sr_resist_strength"]  = sr["sr_resist_strength"]
    out["sr_support_strength"] = sr["sr_support_strength"]

    rng = detect_significant_range(df)
    out["range_valid"]     = rng["range_valid"]
    out["range_side"]      = rng["range_side"]
    out["range_support"]   = rng["range_support"]
    out["range_resist"]    = rng["range_resist"]
    out["range_width_atr"] = rng["range_width_atr"]

    pb = detect_trending_pullback(df)
    out["pullback_valid"]  = pb["pullback_valid"]
    out["pullback_side"]   = pb["pullback_side"]
    out["pullback_level"]  = pb["pullback_level"]
    out["pb_swing_high"]   = pb["pb_swing_high"]
    out["pb_swing_low"]    = pb["pb_swing_low"]

    # Downcast float64 → float32 to halve per-symbol memory footprint.
    # Boolean columns are left as-is; OHLCV source columns keep their dtype.
    float_cols = out.select_dtypes(include="float64").columns
    out[float_cols] = out[float_cols].astype("float32")

    return out


def _warmup_numba() -> None:
    """Pre-compile numba JIT functions with a tiny synthetic dataframe."""
    if not _NUMBA:
        return
    _n = 30
    _rng = np.random.default_rng(0)
    _df = pd.DataFrame({
        "open":   _rng.uniform(1.0, 1.1, _n),
        "high":   _rng.uniform(1.1, 1.2, _n),
        "low":    _rng.uniform(0.9, 1.0, _n),
        "close":  _rng.uniform(1.0, 1.1, _n),
        "volume": _rng.uniform(100, 1000, _n),
    })
    detect_sr_zones(_df)
    detect_significant_range(_df)
    detect_trending_pullback(_df)


# Trigger JIT compilation at import time (pays ~2s once per process).
_warmup_numba()
