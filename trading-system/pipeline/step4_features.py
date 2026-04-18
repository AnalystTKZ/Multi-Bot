#!/usr/bin/env python3
"""
Step 4: Feature Engineering — strictly in-place, no extra copies.
Reads aligned parquet, adds features directly to the frame, writes back.
gc.collect() between each pass. No intermediate copies held in memory.
Output: processed_data/feature_engineered.parquet
"""
from __future__ import annotations
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("step4_features")

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "trading-engine"))
OUTPUT_DIR = BASE / "processed_data"

from indicators.market_structure import (
    compute_all, compute_atr,
    detect_fair_value_gaps, detect_break_of_structure,
    detect_liquidity_sweeps, detect_order_blocks,
)


def _apply_technical(df: pd.DataFrame) -> None:
    """Add technical indicators in-place. Uses compute_all on a minimal OHLCV slice."""
    ohlcv = df[["open", "high", "low", "close", "volume"]].copy()
    ind = compute_all(ohlcv)
    del ohlcv
    gc.collect()

    for col in ind.columns:
        df[col] = ind[col]
    del ind
    gc.collect()

    atr = df["atr_14"]
    bb_range = df["bb_upper"] - df["bb_lower"]

    df["bb_position"]    = (df["close"] - df["bb_lower"]) / (bb_range + 1e-9)
    df["atr_normalized"] = atr / (df["close"] + 1e-9)
    df["log_return"]     = np.log(df["close"] / (df["close"].shift(1) + 1e-9)).clip(-0.1, 0.1)
    df["high_low_range"] = (df["high"] - df["low"]) / (atr + 1e-9)
    df["close_vs_open"]  = (df["close"] - df["open"]) / (atr + 1e-9)
    df["ema21_dist"]     = (df["close"] - df.get("ema_21", df["close"])) / (atr + 1e-9)
    df["ema50_dist"]     = (df["close"] - df.get("ema_50", df["close"])) / (atr + 1e-9)
    df["ema200_dist"]    = (df["close"] - df.get("ema_200", df["close"])) / (atr + 1e-9)
    df["ema_20"]         = df.get("ema_21", df["close"])

    vol_sma = df["volume"].rolling(20).mean()
    df["volume_ratio"]    = df["volume"] / (vol_sma + 1e-9)
    df["realized_vol_20"] = df["close"].pct_change().rolling(20).std().mul(100).clip(0, 5)

    df["atr_ratio"]    = (atr / (df["close"] + 1e-9) * 1000).clip(0, 10)
    df["bb_width_pct"] = df.get("bb_width", bb_range / (df["bb_mid"] + 1e-9))
    df["adx_14_h1"]    = df.get("adx_14", pd.Series(0.0, index=df.index)).fillna(0)
    df["adx_14_h4"]    = df["adx_14_h1"]
    df["ema_stack_score"] = df.get("ema_stack", pd.Series(0.0, index=df.index))


def _apply_smc_rolling(df: pd.DataFrame) -> None:
    """Add rolling SMC counters in-place. SMC flags already added by compute_all."""
    for col in ["bos_bull", "bos_bear", "fvg_bull", "fvg_bear", "sweep_bull", "sweep_bear"]:
        if col in df.columns:
            df[f"{col}_count_24"] = df[col].rolling(24).sum().fillna(0)

    bos_sum   = df.get("bos_bull",   pd.Series(0.0, index=df.index)) + \
                df.get("bos_bear",   pd.Series(0.0, index=df.index))
    sweep_sum = df.get("sweep_bull", pd.Series(0.0, index=df.index)) + \
                df.get("sweep_bear", pd.Series(0.0, index=df.index))
    df["swing_hh_hl_count"]   = bos_sum.rolling(40,   min_periods=1).sum().clip(0, 10)
    df["liquidity_sweep_24h"] = sweep_sum.rolling(48, min_periods=1).sum().clip(0, 5)


def _apply_cross_asset(df: pd.DataFrame) -> None:
    """Add cross-asset features in-place."""
    if "DXY_close" in df.columns:
        df["dxy_eurusd_ratio"] = df["DXY_close"] / (df["close"] + 1e-9)
        df["dxy_1h_return"]    = ((df["DXY_close"] / (df["DXY_close"].shift(4) + 1e-9)) - 1).mul(100).clip(-2, 2)
    elif "dxy_strength" in df.columns:
        df["dxy_1h_return"] = df["dxy_strength"]
    else:
        df["dxy_1h_return"] = 0.0

    xau = "XAUUSD_close"
    if xau in df.columns:
        df["gold_usd_ratio"]  = df[xau] / (df["close"] * 1000 + 1e-9)
        df["gold_atr_ratio"]  = (df["atr_14"] / (df[xau] + 1e-9) * 100).clip(0, 5)
    else:
        df["gold_atr_ratio"] = (df["atr_14"] / (df["close"] + 1e-9) * 100).clip(0, 5)

    for idx in ["SPX", "DAX", "NIKKEI"]:
        rc = f"{idx}_ret1d"
        if rc in df.columns:
            df[f"{idx}_trend_strong"] = (df[rc].abs() > 0.01).astype(float)

    if "risk_on_score" not in df.columns:
        ret_cols = [c for c in df.columns if c.endswith("_ret1d")][:3]
        df["risk_on_score"] = df[ret_cols].mean(axis=1).fillna(0) if ret_cols else 0.0


def _apply_fundamental(df: pd.DataFrame) -> None:
    """Add fundamental features in-place."""
    if "fedfunds" in df.columns and "ecbdfr" in df.columns:
        df["rate_diff_us_eu"] = df["fedfunds"] - df["ecbdfr"]
    elif "dff" in df.columns:
        df["rate_diff_us_eu"] = df["dff"].ffill()

    for col in ["cpiaucsl_mom", "cpilfesl_mom"]:
        if col in df.columns:
            df[f"{col}_direction"] = np.sign(df[col])

    for col in ["fed_decision_flag", "ecb_decision_flag", "boj_decision_flag",
                "election_flag", "fed_speech_flag"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(float)

    if "macro_surprise_composite" not in df.columns:
        sc = [c for c in df.columns if "surprise_norm" in c]
        df["macro_surprise_composite"] = df[sc].mean(axis=1).fillna(0) if sc else 0.0

    if "yield_curve_2_10" not in df.columns and "dgs10" in df.columns and "dgs2" in df.columns:
        df["yield_curve_2_10"] = df["dgs10"] - df["dgs2"]
    if "yield_curve_2_10" in df.columns:
        df["yield_curve_inverted"] = (df["yield_curve_2_10"] < 0).astype(float)


def _apply_session(df: pd.DataFrame) -> None:
    """Add session flags in-place."""
    if not hasattr(df.index, "hour"):
        return
    h = df.index.hour
    if "asia_session"   not in df.columns: df["asia_session"]   = ((h >= 0)  & (h < 7)).astype(float)
    if "london_session" not in df.columns: df["london_session"] = ((h >= 7)  & (h < 12)).astype(float)
    if "ny_session"     not in df.columns: df["ny_session"]     = ((h >= 13) & (h < 18)).astype(float)
    df["dead_zone"] = (h == 12).astype(float)
    if "hour"        not in df.columns: df["hour"]        = h
    if "day_of_week" not in df.columns: df["day_of_week"] = df.index.dayofweek
    if "month"       not in df.columns: df["month"]       = df.index.month
    if "hour_sin"    not in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * h / 24)
        df["hour_cos"] = np.cos(2 * np.pi * h / 24)

    # Session code for regime classifier
    df["session_code"] = np.select(
        [(h >= 2) & (h < 7), (h >= 7) & (h < 12), h == 12, (h >= 13) & (h < 18)],
        [1.0, 2.0, 4.0, 3.0], default=0.0,
    )


def _apply_rl_features(df: pd.DataFrame) -> None:
    """Add RL-state compatible features — batched via pd.concat to avoid fragmentation."""
    atr = df["atr_14"]
    new_cols = {}
    for lag in [1, 4, 8, 24, 48, 96, 168, 336]:
        col = f"atr_lag_{lag}"
        if col not in df.columns:
            new_cols[col] = ((atr + 1e-9) / (atr.shift(lag) + 1e-9)).clip(0, 5).fillna(1.0)

    if "hours_to_next_event" in df.columns:
        new_cols["news_proximity"] = (1.0 / df["hours_to_next_event"].clip(1, 48)).fillna(0)
    else:
        new_cols["news_proximity"] = 0.0

    new_cols["entry_depth"] = 0.3

    if new_cols:
        import pandas as _pd
        extra = _pd.DataFrame(new_cols, index=df.index)
        for col, series in extra.items():
            df[col] = series.values


def main():
    in_path = OUTPUT_DIR / "aligned_multi_asset.parquet"
    if not in_path.exists():
        logger.error("aligned_multi_asset.parquet not found — run step3 first")
        sys.exit(1)

    logger.info("Loading aligned data...")
    df = pd.read_parquet(in_path)
    logger.info("Loaded: %d bars, %d columns", len(df), len(df.columns))

    logger.info("Pass 1/6: technical indicators...")
    _apply_technical(df)
    gc.collect()

    logger.info("Pass 2/6: SMC rolling features...")
    _apply_smc_rolling(df)
    gc.collect()

    logger.info("Pass 3/6: cross-asset features...")
    _apply_cross_asset(df)
    gc.collect()

    logger.info("Pass 4/6: fundamental features...")
    _apply_fundamental(df)
    gc.collect()

    logger.info("Pass 5/6: session features...")
    _apply_session(df)
    gc.collect()

    logger.info("Pass 6/6: RL-state features...")
    _apply_rl_features(df)
    gc.collect()

    logger.info("Final cleanup...")
    # Defragment accumulated column insertions from all 6 passes before cleanup
    df = df.copy()
    gc.collect()
    # Replace inf/nan without inplace to avoid pandas 3.0 chained-assignment warnings
    float_cols = [c for c in df.columns if df[c].dtype.kind == 'f']
    df[float_cols] = df[float_cols].replace([np.inf, -np.inf], np.nan)
    df = df.ffill(limit=4).fillna(0)

    logger.info("Writing %d bars, %d features...", len(df), len(df.columns))
    out_path = OUTPUT_DIR / "feature_engineered.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(df, nthreads=1)
    pq.write_table(table, out_path, compression="snappy")
    del table

    manifest = {
        "total_features": len(df.columns),
        "feature_names": list(df.columns),
        "bars": len(df),
        "date_start": str(df.index.min()),
        "date_end":   str(df.index.max()),
    }
    del df
    gc.collect()

    with open(OUTPUT_DIR / "feature_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\n=== FEATURE ENGINEERING COMPLETE ===")
    print(f"  Total bars: {manifest['bars']:,}")
    print(f"  Total features: {manifest['total_features']}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
