#!/usr/bin/env python3
"""
Step 0: Resample M1/tick histdata → 5M, 15M, 1H, 4H, 1D, 1W, 1MN OHLCV parquet files.

Multi-timeframe outputs per symbol:
  5M  — scalping / signal refinement
  15M — primary entry timeframe (trader sessions)
  1H  — confirmation / regime detection
  4H  — structural bias (BOS, FVG direction)
  1D  — macro trend / daily range context
  1W  — weekly structure / major S&R
  1MN — monthly macro context

Symbols processed (13 total):
  Forex M1 from training_data/forex/*_m1_histdata.csv (2016-2026):
    AUDUSD, EURGBP, EURJPY, EURUSD, GBPJPY, GBPUSD,
    NZDUSD, USDCAD, USDCHF, USDJPY
  Gold M1 from training_data/_histdata_tmp/XAUUSD_m1_histdata.csv (2009-2026,
    semicolon-delimited). If missing, falls back to forex folder XAUUSD file.
  NZDUSD supplement: training_data/_histdata_tmp/DAT_ASCII_NZDUSD_M1_2025.zip
    is merged with the existing NZDUSD_m1_histdata.csv to extend coverage.

Output layout:
  processed_data/histdata/{SYMBOL}_{TF}.parquet   (TF = 5M/15M/1H/4H/1D/1W/1MN)

Each step is fully resumable — existing parquet files are skipped.
"""
from __future__ import annotations
import gc
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("step0_resample")

BASE        = Path(__file__).resolve().parent.parent
FOREX_M1    = BASE / "training_data" / "forex"
ENGINE_HIST = BASE / "trading-engine" / "training_data" / "_histdata_tmp"
OUT_DIR     = BASE / "processed_data" / "histdata"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_ROWS = 300_000   # ~25 MB per chunk — safe for 7.6 GB RAM machine

# All timeframes to produce from each M1 source
# (pandas resample rule, output suffix)
TIMEFRAMES = [
    ("5min",  "5M"),
    ("15min", "15M"),
    ("1h",    "1H"),
    ("4h",    "4H"),
    ("1D",    "1D"),
    ("W",     "1W"),
    ("MS",    "1MN"),
]

OHLCV_AGG = {
    "open":   "first",
    "high":   "max",
    "low":    "min",
    "close":  "last",
    "volume": "sum",
}


def _write_all_timeframes(df_1m: pd.DataFrame, symbol: str, existing: set[str]) -> dict[str, Path]:
    """Resample a full 1M DataFrame into all timeframes and write parquet files."""
    produced = {}
    for rule, suffix in TIMEFRAMES:
        out = OUT_DIR / f"{symbol}_{suffix}.parquet"
        if out.name in existing:
            logger.info("  SKIP %s_%s (exists)", symbol, suffix)
            produced[suffix] = out
            continue
        resampled = df_1m.resample(rule).agg(OHLCV_AGG).dropna(subset=["close"])
        # Merge cross-chunk boundary duplicates
        resampled = resampled.groupby(resampled.index).agg(OHLCV_AGG)
        resampled.to_parquet(out, compression="snappy")
        logger.info("  Wrote %s_%s: %d bars", symbol, suffix, len(resampled))
        produced[suffix] = out
        del resampled; gc.collect()
    return produced


# ── Standard OHLCV 1M CSV (training_data/forex/*_m1_histdata.csv) ────────────
# Header: Datetime,Open,High,Low,Close,Volume

def resample_standard_m1(src: Path, symbol: str) -> dict[str, Path]:
    # Check which timeframes already exist
    existing = {f"{symbol}_{s}.parquet" for _, s in TIMEFRAMES
                if (OUT_DIR / f"{symbol}_{s}.parquet").exists()}
    if len(existing) == len(TIMEFRAMES):
        logger.info("SKIP %s — all timeframes exist", symbol)
        return {s: OUT_DIR / f"{symbol}_{s}.parquet" for _, s in TIMEFRAMES}

    logger.info("Resampling M1 → MTF: %s  (%.0f MB)", symbol, src.stat().st_size / 1e6)
    buckets: list[pd.DataFrame] = []

    for chunk in pd.read_csv(src, index_col=0, parse_dates=True,
                              chunksize=CHUNK_ROWS, low_memory=False):
        chunk.columns = [c.lower() for c in chunk.columns]
        chunk.index = pd.to_datetime(chunk.index, utc=True, errors="coerce")
        chunk = chunk[chunk.index.notna()]
        for col in ["open", "high", "low", "close"]:
            if col not in chunk.columns:
                chunk[col] = chunk.get("close", 0.0)
        if "volume" not in chunk.columns:
            chunk["volume"] = 0.0
        chunk = chunk[["open", "high", "low", "close", "volume"]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna(subset=["close"])

        # Pre-aggregate to 1M within chunk (already 1M, just deduplicate)
        chunk = chunk.groupby(chunk.index).agg(OHLCV_AGG)
        buckets.append(chunk)
        del chunk; gc.collect()

    if not buckets:
        logger.warning("No data for %s", symbol)
        return {}

    df_1m = pd.concat(buckets).sort_index()
    del buckets; gc.collect()
    df_1m = df_1m.groupby(df_1m.index).agg(OHLCV_AGG)

    logger.info("%s M1: %d bars  %s → %s",
                symbol, len(df_1m), df_1m.index.min().date(), df_1m.index.max().date())

    produced = _write_all_timeframes(df_1m, symbol, existing)
    del df_1m; gc.collect()
    return produced


# ── Semicolon-delimited 1M (XAUUSD_m1_histdata.csv) ─────────────────────────
# Format: "20090315 170000;929.6;929.6;929.6;929.6;0"

def resample_xauusd_m1(src: Path) -> dict[str, Path]:
    symbol = "XAUUSD"
    existing = {f"{symbol}_{s}.parquet" for _, s in TIMEFRAMES
                if (OUT_DIR / f"{symbol}_{s}.parquet").exists()}
    if len(existing) == len(TIMEFRAMES):
        logger.info("SKIP XAUUSD — all timeframes exist")
        return {s: OUT_DIR / f"{symbol}_{s}.parquet" for _, s in TIMEFRAMES}

    logger.info("Resampling XAUUSD M1 → MTF  (%.0f MB)", src.stat().st_size / 1e6)
    buckets: list[pd.DataFrame] = []

    for chunk in pd.read_csv(src, header=None, chunksize=CHUNK_ROWS, low_memory=False):
        raw = chunk[0].astype(str)
        parts = raw.str.split(";", expand=True)
        if parts.shape[1] < 5:
            del chunk, raw, parts; gc.collect()
            continue
        parts.columns = ["datetime", "open", "high", "low", "close", "volume"][:parts.shape[1]]
        parts["datetime"] = pd.to_datetime(
            parts["datetime"].str[:15], format="%Y%m%d %H%M%S", utc=True, errors="coerce"
        )
        for col in ["open", "high", "low", "close", "volume"]:
            if col in parts.columns:
                parts[col] = pd.to_numeric(parts[col], errors="coerce")
        if "volume" not in parts.columns:
            parts["volume"] = 0.0
        parts = parts.dropna(subset=["datetime", "close"]).set_index("datetime")
        parts = parts.groupby(parts.index).agg(OHLCV_AGG)
        buckets.append(parts)
        del chunk, raw, parts; gc.collect()

    if not buckets:
        return {}

    df_1m = pd.concat(buckets).sort_index()
    del buckets; gc.collect()
    df_1m = df_1m.groupby(df_1m.index).agg(OHLCV_AGG)

    logger.info("XAUUSD M1: %d bars  %s → %s",
                len(df_1m), df_1m.index.min().date(), df_1m.index.max().date())

    produced = _write_all_timeframes(df_1m, symbol, existing)
    del df_1m; gc.collect()
    return produced


# ── Tick volume extraction — DISABLED (uncomment to enable) ──────────────────
# TODO: Re-enable when order-flow / volume-profile features are needed.
# Processes 44M–545M rows per symbol streamed in CHUNK_ROWS chunks.
# Adds ~30–90 min total runtime. Output: {SYM}_tick_vol.parquet per symbol.
#
# def extract_tick_volume(src: Path, symbol: str) -> Path | None:
#     out = OUT_DIR / f"{symbol}_tick_vol.parquet"
#     if out.exists():
#         logger.info("SKIP tick volume %s", symbol)
#         return out
#     base_15m = OUT_DIR / f"{symbol}_15M.parquet"
#     if not base_15m.exists():
#         return None
#     logger.info("Extracting tick volume: %s (%.1f GB)", symbol, src.stat().st_size / 1e9)
#     buckets = []
#     for chunk in pd.read_csv(src, header=None, names=["raw"],
#                               chunksize=CHUNK_ROWS, low_memory=False):
#         dt = pd.to_datetime(chunk["raw"].str[:15], format="%Y%m%d %H%M%S",
#                             utc=True, errors="coerce").dropna()
#         if dt.empty:
#             del chunk, dt; gc.collect(); continue
#         vol = pd.Series(1, index=dt).resample("15min").sum()
#         buckets.append(vol)
#         del chunk, dt, vol; gc.collect()
#     if not buckets:
#         return None
#     result = pd.concat(buckets).groupby(level=0).sum().rename("tick_volume")
#     idx_15m = pd.read_parquet(base_15m, columns=["close"]).index
#     result = result.reindex(idx_15m, fill_value=0).to_frame()
#     result.to_parquet(out, compression="snappy")
#     logger.info("Tick volume %s: %d bars", symbol, len(result))
#     del result, idx_15m, buckets; gc.collect()
#     return out


def main():
    all_produced: dict[str, dict] = {}

    # ── 1. Forex M1 (2016-2026) ───────────────────────────────────────────────
    for src in sorted(FOREX_M1.glob("*_m1_histdata.csv")):
        sym = src.stem.replace("_m1_histdata", "").upper()
        produced = resample_standard_m1(src, sym)
        if produced:
            all_produced[sym] = {tf: str(p) for tf, p in produced.items()}

    # ── 2. XAUUSD M1 (2009-2026) ─────────────────────────────────────────────
    xau_src = ENGINE_HIST / "XAUUSD_m1_histdata.csv"
    if xau_src.exists():
        produced = resample_xauusd_m1(xau_src)
        if produced:
            all_produced["XAUUSD"] = {tf: str(p) for tf, p in produced.items()}

    # ── 3. Tick volume extraction — DISABLED ─────────────────────────────────
    # Uncomment the block below and the extract_tick_volume function above to enable.
    # for src in sorted(ENGINE_HIST.glob("*_tick_histdata.csv")):
    #     sym = src.stem.replace("_tick_histdata", "").upper()
    #     extract_tick_volume(src, sym)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {"symbols": all_produced, "total_symbols": len(all_produced),
               "timeframes": [s for _, s in TIMEFRAMES]}
    with open(BASE / "processed_data" / "histdata_resample_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== RESAMPLE COMPLETE: {len(all_produced)} symbols ===")
    print(f"  Timeframes: {[s for _, s in TIMEFRAMES]}")
    for sym, tfs in all_produced.items():
        ref = list(tfs.values())[0]
        df = pd.read_parquet(ref, columns=["close"])
        tf_label = list(tfs.keys())[0]
        print(f"  {sym}: {len(df):,} bars @ {tf_label}  "
              f"({df.index.min().date()} → {df.index.max().date()})")
        del df; gc.collect()


if __name__ == "__main__":
    main()
