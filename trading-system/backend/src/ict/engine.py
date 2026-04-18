"""
Institutional-grade ICT entry model.
Deterministic, multi-timeframe, liquidity-focused logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import median
from typing import Dict, List, Optional, Tuple

from .models import Candle, MarketSnapshot, IctSignal


@dataclass
class SwingPoint:
    index: int
    price: float
    kind: str  # 'high' or 'low'
    strength: str = "internal"  # 'internal' or 'external'


@dataclass
class LiquidityPool:
    level: float
    count: int
    kind: str  # 'high' or 'low'
    last_index: int
    strength: str  # 'internal' or 'external'


@dataclass
class SweepEvent:
    index: int
    level: float
    direction: str  # 'up' (buyside) or 'down' (sellside)
    wick_size: float


@dataclass
class OrderBlock:
    index: int
    direction: str  # 'bullish' or 'bearish'
    low: float
    high: float
    displacement_index: int
    mitigated: bool
    score: float


@dataclass
class FvgZone:
    index: int
    direction: str  # 'bullish' or 'bearish'
    low: float
    high: float
    size: float
    fill_pct: float


LONDON_SESSION = (7, 10)  # UTC
NY_SESSION = (12, 15)  # UTC


def generate_ict_signal(snapshot: MarketSnapshot, score_threshold: int = 75) -> IctSignal:
    symbol = snapshot.symbol
    reasoning: List[str] = []

    tf_map = snapshot.timeframes
    if not tf_map:
        return IctSignal(
            symbol=symbol,
            direction="none",
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            confluence_score=0,
            reasoning=["No timeframe data supplied"],
        )

    tf_norm = {key.upper(): key for key in tf_map.keys()}
    htf_daily = tf_map.get(tf_norm.get("1D", "1D"))
    htf_4h = tf_map.get(tf_norm.get("4H", "4H"))
    if not htf_daily or not htf_4h:
        return IctSignal(
            symbol=symbol,
            direction="none",
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            confluence_score=0,
            reasoning=["Missing required HTF data (1D and 4H)"],
        )

    ltf_key = _select_ltf_key(tf_map)
    if not ltf_key:
        return IctSignal(
            symbol=symbol,
            direction="none",
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            confluence_score=0,
            reasoning=["Missing LTF data for entry logic"],
        )

    ltf = tf_map[ltf_key]
    if len(ltf) < 20:
        return IctSignal(
            symbol=symbol,
            direction="none",
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            confluence_score=0,
            reasoning=["Insufficient LTF candles for ICT logic"],
        )

    htf_bias = _htf_bias(htf_daily, htf_4h)
    if htf_bias == "neutral":
        return IctSignal(
            symbol=symbol,
            direction="none",
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            confluence_score=0,
            reasoning=["HTF bias is neutral"],
        )

    dealing_range = _dealing_range(htf_4h)
    if not dealing_range:
        return IctSignal(
            symbol=symbol,
            direction="none",
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            confluence_score=0,
            reasoning=["Unable to compute dealing range"],
        )

    ltf_candles = _normalize_candles(ltf)
    swings = _detect_swings(ltf_candles)
    _classify_swings(swings)

    external_highs = [s for s in swings if s.kind == "high" and s.strength == "external"]
    external_lows = [s for s in swings if s.kind == "low" and s.strength == "external"]
    internal_highs = [s for s in swings if s.kind == "high" and s.strength == "internal"]
    internal_lows = [s for s in swings if s.kind == "low" and s.strength == "internal"]

    avg_range = _avg_range(ltf_candles, 50)
    tol = max(avg_range * 0.25, ltf_candles[-1]["close"] * 0.0005)

    external_high_pools = _cluster_levels(external_highs, "high", tol)
    external_low_pools = _cluster_levels(external_lows, "low", tol)
    internal_high_pools = _cluster_levels(internal_highs, "high", tol)
    internal_low_pools = _cluster_levels(internal_lows, "low", tol)
    session_pools = _session_pools(ltf_candles)
    for pool in session_pools:
        if pool.kind == "high":
            external_high_pools.append(pool)
        else:
            external_low_pools.append(pool)

    sweep = None
    if htf_bias == "bullish":
        sweep = _detect_sweep(ltf_candles, external_low_pools, "down", tol)
    else:
        sweep = _detect_sweep(ltf_candles, external_high_pools, "up", tol)

    inducement = _detect_inducement(
        ltf_candles,
        internal_low_pools if htf_bias == "bullish" else internal_high_pools,
        external_low_pools if htf_bias == "bullish" else external_high_pools,
        tol,
    )

    obs = _detect_order_blocks(ltf_candles)
    fvgs = _detect_fvgs(ltf_candles)

    last_close = ltf_candles[-1]["close"]

    trend = _trend_state(swings)
    bos = _detect_bos(ltf_candles, swings, trend)
    choch = None
    if sweep:
        choch = _detect_choch_after(ltf_candles, swings, sweep.index, htf_bias)

    zone_hit, ob_used, fvg_used = _find_return_to_zone(
        last_close,
        obs,
        fvgs,
        htf_bias,
        dealing_range,
    )

    confirmation = _is_displacement(ltf_candles, len(ltf_candles) - 1, htf_bias)

    reasons = []
    if htf_bias == "bullish":
        reasons.append("HTF bias bullish (1D + 4H)")
    else:
        reasons.append("HTF bias bearish (1D + 4H)")

    if sweep:
        reasons.append("Liquidity sweep detected on LTF")
    else:
        reasons.append("No valid LTF liquidity sweep")

    if inducement:
        reasons.append("Inducement pattern detected")

    if zone_hit:
        reasons.append("Price returned to OB/FVG zone")
    else:
        reasons.append("No valid OB/FVG retrace")

    if bos:
        reasons.append("Break of structure (BOS) on LTF")
    if choch:
        reasons.append("Market structure shift (CHoCH) after sweep")
    else:
        reasons.append("No CHoCH after sweep")

    if confirmation:
        reasons.append("Confirmation displacement candle")
    else:
        reasons.append("No confirmation displacement candle")

    score = _score_signal(
        htf_bias=htf_bias,
        sweep=sweep,
        ob=ob_used,
        fvg=fvg_used,
        in_session=_in_killzone(ltf_candles[-1]),
    )

    all_conditions = all([
        htf_bias in ("bullish", "bearish"),
        sweep is not None,
        zone_hit,
        choch is not None,
        confirmation,
    ])

    feature_payload = _build_feature_payload(
        ob_used=ob_used,
        fvg_used=fvg_used,
        sweep=sweep,
        htf_bias=htf_bias,
        in_session=_in_killzone(ltf_candles[-1]),
        entry_price=None,
        stop_loss=None,
        take_profit=None,
        avg_range=avg_range,
        last_close=last_close,
    )

    if not all_conditions or score < score_threshold:
        return IctSignal(
            symbol=symbol,
            direction="none",
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            confluence_score=int(score),
            reasoning=reasons,
            feature_payload=feature_payload,
        )

    entry_price, stop_loss, take_profit = _build_trade_levels(
        last_close=last_close,
        bias=htf_bias,
        sweep=sweep,
        ob=ob_used,
        fvg=fvg_used,
        pools=external_high_pools if htf_bias == "bullish" else external_low_pools,
        avg_range=avg_range,
    )

    direction = "buy" if htf_bias == "bullish" else "sell"
    feature_payload = _build_feature_payload(
        ob_used=ob_used,
        fvg_used=fvg_used,
        sweep=sweep,
        htf_bias=htf_bias,
        in_session=_in_killzone(ltf_candles[-1]),
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        avg_range=avg_range,
        last_close=last_close,
    )
    return IctSignal(
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confluence_score=int(score),
        reasoning=reasons,
        feature_payload=feature_payload,
    )


def _select_ltf_key(tf_map: Dict[str, List[Candle]]) -> Optional[str]:
    candidates = []
    for key in tf_map.keys():
        minutes = _timeframe_to_minutes(key)
        if minutes is None:
            continue
        candidates.append((minutes, key))
    if not candidates:
        return None
    candidates.sort()
    for minutes, key in candidates:
        if key.upper() not in ("1D", "4H"):
            return key
    return None


def _timeframe_to_minutes(tf: str) -> Optional[int]:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        try:
            return int(tf[:-1])
        except ValueError:
            return None
    if tf.endswith("h"):
        try:
            return int(tf[:-1]) * 60
        except ValueError:
            return None
    if tf.endswith("d"):
        try:
            return int(tf[:-1]) * 24 * 60
        except ValueError:
            return None
    return None


def _normalize_candles(candles: List[Candle]) -> List[Dict[str, float]]:
    return [
        {
            "ts": c.ts,
            "open": float(c.open),
            "high": float(c.high),
            "low": float(c.low),
            "close": float(c.close),
            "volume": float(c.volume) if c.volume is not None else None,
            "dt": _parse_ts(c.ts),
        }
        for c in candles
    ]


def _parse_ts(ts: str) -> Optional[datetime]:
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _avg_range(candles: List[Dict[str, float]], lookback: int) -> float:
    if not candles:
        return 0.0
    window = candles[-lookback:] if len(candles) > lookback else candles
    ranges = [c["high"] - c["low"] for c in window]
    return sum(ranges) / max(len(ranges), 1)


def _avg_body(candles: List[Dict[str, float]], lookback: int) -> float:
    if not candles:
        return 0.0
    window = candles[-lookback:] if len(candles) > lookback else candles
    bodies = [abs(c["close"] - c["open"]) for c in window]
    return sum(bodies) / max(len(bodies), 1)


def _detect_swings(candles: List[Dict[str, float]], left: int = 2, right: int = 2) -> List[SwingPoint]:
    swings: List[SwingPoint] = []
    if len(candles) < left + right + 1:
        return swings
    for i in range(left, len(candles) - right):
        high = candles[i]["high"]
        low = candles[i]["low"]
        is_high = True
        is_low = True
        for j in range(i - left, i + right + 1):
            if candles[j]["high"] > high:
                is_high = False
            if candles[j]["low"] < low:
                is_low = False
            if not is_high and not is_low:
                break
        if is_high:
            swings.append(SwingPoint(index=i, price=high, kind="high"))
        if is_low:
            swings.append(SwingPoint(index=i, price=low, kind="low"))
    return swings


def _classify_swings(swings: List[SwingPoint]) -> None:
    sizes: List[float] = []
    for idx, swing in enumerate(swings):
        prev = next(
            (s for s in reversed(swings[:idx]) if s.kind != swing.kind),
            None,
        )
        if prev:
            sizes.append(abs(swing.price - prev.price))
    threshold = median(sizes) if sizes else 0.0
    for idx, swing in enumerate(swings):
        prev = next(
            (s for s in reversed(swings[:idx]) if s.kind != swing.kind),
            None,
        )
        if not prev:
            swing.strength = "internal"
            continue
        size = abs(swing.price - prev.price)
        swing.strength = "external" if size >= threshold else "internal"


def _trend_state(swings: List[SwingPoint]) -> str:
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return "neutral"
    hi1, hi2 = highs[-1].price, highs[-2].price
    lo1, lo2 = lows[-1].price, lows[-2].price
    if hi1 > hi2 and lo1 > lo2:
        return "bullish"
    if hi1 < hi2 and lo1 < lo2:
        return "bearish"
    return "neutral"


def _htf_bias(daily: List[Candle], h4: List[Candle]) -> str:
    daily_swings = _detect_swings(_normalize_candles(daily))
    _classify_swings(daily_swings)
    h4_swings = _detect_swings(_normalize_candles(h4))
    _classify_swings(h4_swings)
    daily_trend = _trend_state(daily_swings)
    h4_trend = _trend_state(h4_swings)
    if daily_trend == h4_trend and daily_trend in ("bullish", "bearish"):
        return daily_trend
    return "neutral"


def _dealing_range(candles: List[Candle]) -> Optional[Tuple[float, float, float]]:
    if len(candles) < 5:
        return None
    norm = _normalize_candles(candles)
    high = max(c["high"] for c in norm)
    low = min(c["low"] for c in norm)
    mid = (high + low) / 2.0
    return low, high, mid


def _premium_discount_zone(price: float, dealing_range: Tuple[float, float, float]) -> str:
    _, _, mid = dealing_range
    return "discount" if price <= mid else "premium"


def _cluster_levels(swings: List[SwingPoint], kind: str, tol: float) -> List[LiquidityPool]:
    clusters: List[LiquidityPool] = []
    for swing in swings:
        if swing.kind != kind:
            continue
        matched = False
        for cluster in clusters:
            if abs(swing.price - cluster.level) <= tol:
                cluster.level = (cluster.level * cluster.count + swing.price) / (cluster.count + 1)
                cluster.count += 1
                cluster.last_index = swing.index
                matched = True
                break
        if not matched:
            clusters.append(
                LiquidityPool(
                    level=swing.price,
                    count=1,
                    kind=kind,
                    last_index=swing.index,
                    strength=swing.strength,
                )
            )
    return clusters


def _session_pools(candles: List[Dict[str, float]]) -> List[LiquidityPool]:
    last_dt = next((c["dt"] for c in reversed(candles) if c.get("dt")), None)
    if not last_dt:
        return []
    date = last_dt.date()
    pools: List[LiquidityPool] = []
    for start, end in (LONDON_SESSION, NY_SESSION):
        session_candles = [
            c for c in candles
            if c.get("dt") and c["dt"].date() == date and start <= c["dt"].hour < end
        ]
        if not session_candles:
            continue
        high = max(c["high"] for c in session_candles)
        low = min(c["low"] for c in session_candles)
        high_index = next((i for i, c in enumerate(candles) if c["dt"] and c["dt"].date() == date and c["high"] == high), len(candles) - 1)
        low_index = next((i for i, c in enumerate(candles) if c["dt"] and c["dt"].date() == date and c["low"] == low), len(candles) - 1)
        pools.append(LiquidityPool(level=high, count=1, kind="high", last_index=high_index, strength="external"))
        pools.append(LiquidityPool(level=low, count=1, kind="low", last_index=low_index, strength="external"))
    return pools


def _detect_sweep(
    candles: List[Dict[str, float]],
    pools: List[LiquidityPool],
    direction: str,
    tol: float,
    lookback: int = 20,
) -> Optional[SweepEvent]:
    if not pools:
        return None
    recent = candles[-lookback:] if len(candles) > lookback else candles
    best: Optional[SweepEvent] = None
    for idx, candle in enumerate(recent):
        global_index = len(candles) - len(recent) + idx
        for pool in pools:
            if direction == "down":
                if candle["low"] < pool.level - tol and candle["close"] > pool.level:
                    wick = pool.level - candle["low"]
                    best = SweepEvent(index=global_index, level=pool.level, direction="down", wick_size=wick)
            else:
                if candle["high"] > pool.level + tol and candle["close"] < pool.level:
                    wick = candle["high"] - pool.level
                    best = SweepEvent(index=global_index, level=pool.level, direction="up", wick_size=wick)
    return best


def _detect_bos(
    candles: List[Dict[str, float]],
    swings: List[SwingPoint],
    trend: str,
) -> Optional[Dict[str, float]]:
    if not swings or not candles:
        return None
    last_close = candles[-1]["close"]
    if trend == "bullish":
        last_high = next((s for s in reversed(swings) if s.kind == "high"), None)
        if last_high and last_close > last_high.price:
            return {"direction": "bullish", "level": last_high.price}
    if trend == "bearish":
        last_low = next((s for s in reversed(swings) if s.kind == "low"), None)
        if last_low and last_close < last_low.price:
            return {"direction": "bearish", "level": last_low.price}
    return None


def _detect_inducement(
    candles: List[Dict[str, float]],
    internal_pools: List[LiquidityPool],
    external_pools: List[LiquidityPool],
    tol: float,
    lookback: int = 30,
) -> bool:
    if not internal_pools or not external_pools:
        return False
    recent = candles[-lookback:] if len(candles) > lookback else candles
    internal_sweep = _detect_sweep(recent, internal_pools, "down", tol) or _detect_sweep(recent, internal_pools, "up", tol)
    if not internal_sweep:
        return False
    external_sweep = _detect_sweep(recent, external_pools, "down", tol) or _detect_sweep(recent, external_pools, "up", tol)
    if not external_sweep:
        return False
    return external_sweep.index > internal_sweep.index


def _detect_choch_after(
    candles: List[Dict[str, float]],
    swings: List[SwingPoint],
    sweep_index: int,
    bias: str,
) -> Optional[Dict[str, float]]:
    if not swings:
        return None
    if bias == "bullish":
        last_high = next((s for s in reversed(swings) if s.kind == "high" and s.index < sweep_index), None)
        if not last_high:
            return None
        for i in range(sweep_index + 1, len(candles)):
            if candles[i]["close"] > last_high.price:
                return {"index": i, "level": last_high.price}
    else:
        last_low = next((s for s in reversed(swings) if s.kind == "low" and s.index < sweep_index), None)
        if not last_low:
            return None
        for i in range(sweep_index + 1, len(candles)):
            if candles[i]["close"] < last_low.price:
                return {"index": i, "level": last_low.price}
    return None


def _detect_order_blocks(candles: List[Dict[str, float]]) -> List[OrderBlock]:
    obs: List[OrderBlock] = []
    if len(candles) < 5:
        return obs
    avg_body = _avg_body(candles, 50)
    avg_range = _avg_range(candles, 50)
    for i in range(1, len(candles) - 1):
        candle = candles[i]
        body = abs(candle["close"] - candle["open"])
        displacement = body > avg_body * 1.5 and (candle["high"] - candle["low"]) > avg_range * 1.2
        if not displacement:
            continue
        if candle["close"] > candle["open"] and candle["close"] > candles[i - 1]["high"]:
            ob_index = _find_last_opposite(candles, i, "bearish")
            if ob_index is None:
                continue
            ob = _build_order_block(candles, ob_index, "bullish", i, avg_body)
            obs.append(ob)
        if candle["close"] < candle["open"] and candle["close"] < candles[i - 1]["low"]:
            ob_index = _find_last_opposite(candles, i, "bullish")
            if ob_index is None:
                continue
            ob = _build_order_block(candles, ob_index, "bearish", i, avg_body)
            obs.append(ob)
    return obs


def _find_last_opposite(candles: List[Dict[str, float]], start_index: int, direction: str) -> Optional[int]:
    limit = max(0, start_index - 10)
    for j in range(start_index - 1, limit - 1, -1):
        if direction == "bearish" and candles[j]["close"] < candles[j]["open"]:
            return j
        if direction == "bullish" and candles[j]["close"] > candles[j]["open"]:
            return j
    return None


def _build_order_block(
    candles: List[Dict[str, float]],
    ob_index: int,
    direction: str,
    displacement_index: int,
    avg_body: float,
) -> OrderBlock:
    low = candles[ob_index]["low"]
    high = candles[ob_index]["high"]
    mitigated = _is_mitigated(candles, ob_index, low, high)
    disp = candles[displacement_index]
    disp_strength = min(1.0, abs(disp["close"] - disp["open"]) / max(avg_body, 1e-9))
    volume_score = _volume_spike_score(candles, displacement_index)
    imbalance_score = 1.0 if _has_fvg_around(candles, displacement_index) else 0.0
    freshness = _freshness_score(len(candles) - 1 - ob_index)
    score = 100.0 * (0.35 * freshness + 0.3 * disp_strength + 0.2 * volume_score + 0.15 * imbalance_score)
    if mitigated:
        score *= 0.6
    return OrderBlock(
        index=ob_index,
        direction=direction,
        low=low,
        high=high,
        displacement_index=displacement_index,
        mitigated=mitigated,
        score=score,
    )


def _is_mitigated(candles: List[Dict[str, float]], ob_index: int, low: float, high: float) -> bool:
    for i in range(ob_index + 1, len(candles)):
        if candles[i]["low"] <= high and candles[i]["high"] >= low:
            return True
    return False


def _freshness_score(age: int, max_age: int = 100) -> float:
    return max(0.0, 1.0 - (age / max_age))


def _volume_spike_score(candles: List[Dict[str, float]], index: int) -> float:
    vols = [c["volume"] for c in candles if c["volume"] is not None]
    if not vols or candles[index]["volume"] is None:
        return 0.4
    avg_vol = sum(vols) / len(vols)
    return min(1.0, candles[index]["volume"] / max(avg_vol, 1e-9))


def _has_fvg_around(candles: List[Dict[str, float]], index: int) -> bool:
    if index < 1 or index >= len(candles) - 1:
        return False
    prev = candles[index - 1]
    nxt = candles[index + 1]
    return nxt["low"] > prev["high"] or nxt["high"] < prev["low"]


def _detect_fvgs(candles: List[Dict[str, float]]) -> List[FvgZone]:
    fvgs: List[FvgZone] = []
    if len(candles) < 3:
        return fvgs
    for i in range(1, len(candles) - 1):
        prev = candles[i - 1]
        nxt = candles[i + 1]
        if nxt["low"] > prev["high"]:
            low = prev["high"]
            high = nxt["low"]
            size = high - low
            fill_pct = _fvg_fill_pct(candles, i + 1, low, high, "bullish")
            fvgs.append(FvgZone(index=i, direction="bullish", low=low, high=high, size=size, fill_pct=fill_pct))
        elif nxt["high"] < prev["low"]:
            low = nxt["high"]
            high = prev["low"]
            size = high - low
            fill_pct = _fvg_fill_pct(candles, i + 1, low, high, "bearish")
            fvgs.append(FvgZone(index=i, direction="bearish", low=low, high=high, size=size, fill_pct=fill_pct))
    return fvgs


def _fvg_fill_pct(
    candles: List[Dict[str, float]],
    start_index: int,
    low: float,
    high: float,
    direction: str,
) -> float:
    if start_index >= len(candles):
        return 0.0
    if direction == "bullish":
        lowest = min(c["low"] for c in candles[start_index:])
        if lowest >= high:
            return 0.0
        return min(1.0, max(0.0, (high - lowest) / max(high - low, 1e-9)))
    highest = max(c["high"] for c in candles[start_index:])
    if highest <= low:
        return 0.0
    return min(1.0, max(0.0, (highest - low) / max(high - low, 1e-9)))


def _find_return_to_zone(
    price: float,
    obs: List[OrderBlock],
    fvgs: List[FvgZone],
    bias: str,
    dealing_range: Tuple[float, float, float],
) -> Tuple[bool, Optional[OrderBlock], Optional[FvgZone]]:
    zone = _premium_discount_zone(price, dealing_range)
    if bias == "bullish" and zone != "discount":
        return False, None, None
    if bias == "bearish" and zone != "premium":
        return False, None, None

    ob_candidates = [ob for ob in obs if ob.direction == ("bullish" if bias == "bullish" else "bearish")]
    fvg_candidates = [fvg for fvg in fvgs if fvg.direction == ("bullish" if bias == "bullish" else "bearish")]

    best_ob = max(ob_candidates, key=lambda ob: ob.score, default=None)
    best_fvg = max(fvg_candidates, key=lambda fvg: fvg.size, default=None)

    in_ob = False
    if best_ob:
        in_ob = best_ob.low <= price <= best_ob.high
    in_fvg = False
    if best_fvg:
        in_fvg = best_fvg.low <= price <= best_fvg.high

    if in_ob:
        return True, best_ob, None
    if in_fvg:
        return True, None, best_fvg
    return False, best_ob, best_fvg


def _is_displacement(candles: List[Dict[str, float]], index: int, bias: str) -> bool:
    if index < 1:
        return False
    avg_body = _avg_body(candles, 50)
    candle = candles[index]
    body = abs(candle["close"] - candle["open"])
    if body < avg_body * 1.5:
        return False
    if bias == "bullish":
        return candle["close"] > candle["open"] and candle["close"] > candles[index - 1]["high"]
    return candle["close"] < candle["open"] and candle["close"] < candles[index - 1]["low"]


def _score_signal(
    htf_bias: str,
    sweep: Optional[SweepEvent],
    ob: Optional[OrderBlock],
    fvg: Optional[FvgZone],
    in_session: bool,
) -> float:
    score = 0.0
    if htf_bias in ("bullish", "bearish"):
        score += 30.0
    if sweep:
        strength = min(1.0, sweep.wick_size / max(sweep.level * 0.002, 1e-9))
        score += 20.0 * strength
    if ob:
        score += 20.0 * min(1.0, ob.score / 100.0)
    if fvg:
        size_factor = min(1.0, fvg.size / max(fvg.high * 0.002, 1e-9))
        align_factor = 1.0 - fvg.fill_pct
        score += 15.0 * (0.6 * size_factor + 0.4 * align_factor)
    score += 15.0 if in_session else 5.0
    return min(100.0, score)


def _in_killzone(candle: Dict[str, float]) -> bool:
    dt = candle.get("dt")
    if not dt:
        return False
    hour = dt.hour
    in_london = LONDON_SESSION[0] <= hour < LONDON_SESSION[1]
    in_ny = NY_SESSION[0] <= hour < NY_SESSION[1]
    return in_london or in_ny


def _build_trade_levels(
    last_close: float,
    bias: str,
    sweep: Optional[SweepEvent],
    ob: Optional[OrderBlock],
    fvg: Optional[FvgZone],
    pools: List[LiquidityPool],
    avg_range: float,
) -> Tuple[float, float, float]:
    buffer = avg_range * 0.1
    entry = last_close

    if ob:
        entry = (ob.low + ob.high) / 2.0
    elif fvg:
        entry = (fvg.low + fvg.high) / 2.0

    if bias == "bullish":
        stop = (sweep.level if sweep else (ob.low if ob else entry - avg_range)) - buffer
        target = _nearest_pool_above(entry, pools)
        if target is None:
            target = entry + 2.0 * (entry - stop)
    else:
        stop = (sweep.level if sweep else (ob.high if ob else entry + avg_range)) + buffer
        target = _nearest_pool_below(entry, pools)
        if target is None:
            target = entry - 2.0 * (stop - entry)
    return entry, stop, target


def _nearest_pool_above(price: float, pools: List[LiquidityPool]) -> Optional[float]:
    candidates = [p.level for p in pools if p.level > price]
    return min(candidates) if candidates else None


def _nearest_pool_below(price: float, pools: List[LiquidityPool]) -> Optional[float]:
    candidates = [p.level for p in pools if p.level < price]
    return max(candidates) if candidates else None


def _build_feature_payload(
    ob_used: Optional[OrderBlock],
    fvg_used: Optional[FvgZone],
    sweep: Optional[SweepEvent],
    htf_bias: str,
    in_session: bool,
    entry_price: Optional[float],
    stop_loss: Optional[float],
    take_profit: Optional[float],
    avg_range: float,
    last_close: float,
) -> Dict[str, float]:
    ob_strength = ob_used.score if ob_used else 0.0
    fvg_size = fvg_used.size if fvg_used else 0.0
    liquidity_sweep = 1.0 if sweep else 0.0
    htf_alignment = 1.0 if htf_bias in ("bullish", "bearish") else 0.0
    session_timing = 1.0 if in_session else 0.0

    rr_ratio = 0.0
    distance_to_liquidity = 0.0
    if entry_price is not None and stop_loss is not None and take_profit is not None:
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        if risk > 0:
            rr_ratio = reward / risk
        distance_to_liquidity = abs(take_profit - entry_price)
    else:
        distance_to_liquidity = avg_range if avg_range > 0 else 0.0

    return {
        "ob_strength": float(ob_strength),
        "fvg_size": float(fvg_size),
        "liquidity_sweep": float(liquidity_sweep),
        "htf_alignment": float(htf_alignment),
        "session_timing": float(session_timing),
        "rr_ratio": float(rr_ratio),
        "distance_to_liquidity": float(distance_to_liquidity),
        "last_close": float(last_close),
    }
