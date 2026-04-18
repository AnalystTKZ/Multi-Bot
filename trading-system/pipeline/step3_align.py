#!/usr/bin/env python3
"""
Step 3: Alignment — builds aligned_multi_asset.parquet one symbol at a time.
Anchor = EURUSD (carries all macro columns).
Non-anchor symbols: only OHLCV columns are added, prefixed with symbol name.
Daily context series are loaded one at a time, reindexed, then deleted.

Peak RAM: one symbol DataFrame at a time (~5600 rows × 50 cols ≈ few MB).
"""
from __future__ import annotations
import gc
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("step3")

BASE = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE / "processed_data" / "clean"
OUTPUT_DIR = BASE / "processed_data"
TRAINING_DIR = BASE / "training_data"

PRIMARY = [
    "AUDUSD", "EURGBP", "EURJPY", "EURUSD", "GBPJPY",
    "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
    "XAUUSD",
]
ANCHOR  = "EURUSD"

# All 17 available daily indices for correlation context
CONTEXT = [
    "ASX200", "CAC40", "DAX", "DJIA", "DXY",
    "EUROSTOXX", "FTSE", "GOLD_FUT", "HSI", "NASDAQ",
    "NIKKEI", "OIL_FUT", "SPX", "US10Y", "US30Y",
    "US3M", "VIX",
]

# Fundamental series: (filename_stem, column_alias)
FUNDAMENTALS = [
    ("treasury_10yr",  "fund_us10y"),
    ("treasury_2yr",   "fund_us2y"),
    ("fed_funds_rate", "fund_fedfunds"),
    ("vix",            "fund_vix"),
]


def _load_clean(sym: str) -> pd.DataFrame | None:
    p = CLEAN_DIR / f"{sym}_15M.parquet"
    if not p.exists():
        logger.warning("Clean file missing: %s", p)
        return None
    df = pd.read_parquet(p)
    df.sort_index(inplace=True)
    return df


def _load_daily_close(sym: str) -> pd.Series | None:
    for sub in ["indices", "stocks"]:
        p = TRAINING_DIR / sub / f"{sym}_1d.csv"
        if p.exists():
            try:
                df = pd.read_csv(p, index_col=0, parse_dates=True, low_memory=False)
                df.columns = [c.lower() for c in df.columns]
                df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
                df = df[df.index.notna()].sort_index()
                if "close" in df.columns:
                    return df["close"]
            except Exception as e:
                logger.warning("Daily load %s/%s: %s", sub, sym, e)
    return None


def _load_fundamental(stem: str) -> pd.Series | None:
    """Load a fundamental CSV from training_data/fundamental/. Returns a daily Series."""
    p = TRAINING_DIR / "fundamental" / f"{stem}.csv"
    if not p.exists():
        logger.info("Fundamental file not found: %s", p)
        return None
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True, low_memory=False)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[df.index.notna()].sort_index()
        # Use first numeric column
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            return None
        return df[num_cols[0]]
    except Exception as e:
        logger.warning("Fundamental load %s: %s", stem, e)
    return None


def main():
    # ── 1. Load anchor ─────────────────────────────────────────────────────────
    logger.info("Loading anchor: %s", ANCHOR)
    anchor = _load_clean(ANCHOR)
    if anchor is None or len(anchor) < 200:
        logger.error("Cannot load anchor %s", ANCHOR)
        sys.exit(1)

    # Find common end date across all primary symbols to avoid stale ffill bias
    min_end = anchor.index.max()
    for sym in PRIMARY:
        if sym == ANCHOR:
            continue
        df_check = _load_clean(sym)
        if df_check is not None:
            min_end = min(min_end, df_check.index.max())
            del df_check

    # Clip anchor to common end date
    anchor = anchor[anchor.index <= min_end]
    master_idx = anchor.index
    logger.info("Master index: %d bars (%s → %s) [clipped to common end %s]",
                len(master_idx), master_idx.min().date(), master_idx.max().date(), min_end.date())

    # Start with anchor as the base aligned frame
    aligned = anchor
    del anchor; gc.collect()

    # ── 2. Add non-anchor symbols one at a time ─────────────────────────────
    for sym in PRIMARY:
        if sym == ANCHOR:
            continue
        df = _load_clean(sym)
        if df is None:
            logger.warning("Skipping %s — no clean data", sym)
            continue

        # Reindex to master, ffill up to 4 bars, keep only OHLCV
        df_ri = df[["open", "high", "low", "close", "volume"]].reindex(
            master_idx, method="ffill", limit=4
        )
        del df; gc.collect()

        for col in df_ri.columns:
            aligned[f"{sym}_{col}"] = df_ri[col]
        del df_ri; gc.collect()
        logger.info("Added %s OHLCV columns", sym)

    # ── 3. Add daily context series one at a time ───────────────────────────
    for sym in CONTEXT:
        close = _load_daily_close(sym)
        if close is None:
            logger.info("No daily data for context symbol %s — skipping", sym)
            continue

        # Only add if not already in aligned (unified file may have it)
        ret_col = f"{sym}_ret1d"
        if ret_col not in aligned.columns:
            close_ri = close.reindex(master_idx, method="ffill")
            aligned[f"{sym}_close"] = close_ri
            aligned[ret_col] = close.pct_change(1).reindex(master_idx, method="ffill")
            del close_ri
        del close; gc.collect()
        logger.info("Added context: %s", sym)

    # ── 4. Add fundamental series ───────────────────────────────────────────
    for stem, alias in FUNDAMENTALS:
        series = _load_fundamental(stem)
        if series is None:
            continue
        if alias not in aligned.columns:
            aligned[alias] = series.reindex(master_idx, method="ffill")
            logger.info("Added fundamental: %s → %s", stem, alias)
        del series; gc.collect()

    # ── 5. Final fill and write ─────────────────────────────────────────────
    aligned.ffill(limit=8, inplace=True)
    aligned = aligned[aligned["close"].notna()]

    n_bars = len(aligned)
    n_cols = len(aligned.columns)
    logger.info("Aligned frame: %d bars, %d features", n_bars, n_cols)

    out_path = OUTPUT_DIR / "aligned_multi_asset.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq
    _tbl = pa.Table.from_pandas(aligned, nthreads=1)
    pq.write_table(_tbl, out_path, compression="snappy")
    del _tbl, aligned; gc.collect()

    summary = {
        "bars": n_bars, "features": n_cols,
        "symbols": PRIMARY, "context": CONTEXT,
        "fundamentals": [alias for _, alias in FUNDAMENTALS],
        "output": str(out_path),
    }
    with open(OUTPUT_DIR / "alignment_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n=== ALIGNMENT COMPLETE ===")
    print(f"  Bars: {n_bars:,}  Features: {n_cols}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
