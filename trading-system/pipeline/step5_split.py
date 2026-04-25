#!/usr/bin/env python3
"""
Step 5: Time-based train/val/test split anchored to calendar years.

Split logic (calendar-based):
  test  = last 2 years of data  (blind — never touched until Round 2 backtest)
  val   = 2 years before test   (Round 1 backtest window)
  train = everything before val

Requires at least 5 years of data (1yr train + 2yr val + 2yr test minimum).
Exits with an error if the data is too short — no fallbacks.

Input:  processed_data/feature_engineered.parquet
Output: ml_training/datasets/train.parquet, validation.parquet, test.parquet
        ml_training/datasets/split_summary.json
"""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("step5_split")

# Use env_config so outputs go to the correct root on Kaggle (remote clone when present).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env_config import get_env
_ENV = get_env()

BASE       = _ENV["base"]
OUTPUT_DIR = _ENV["processed"]
ML_DIR     = _ENV["ml_training"] / "datasets"
ML_DIR.mkdir(parents=True, exist_ok=True)

TEST_YEARS = 2
VAL_YEARS  = 2
MIN_TRAIN_YEARS = 1


def main():
    in_path = OUTPUT_DIR / "feature_engineered.parquet"
    if not in_path.exists():
        logger.error("feature_engineered.parquet not found — run step4 first")
        sys.exit(1)

    logger.info("Loading feature-engineered data...")
    df = pd.read_parquet(in_path)
    df = df.sort_index()
    n = len(df)
    logger.info("Loaded %d rows, %d features", n, len(df.columns))

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    data_start  = df.index[0]
    data_end    = df.index[-1]
    total_years = (data_end - data_start).days / 365.25
    required    = MIN_TRAIN_YEARS + VAL_YEARS + TEST_YEARS

    logger.info("Data span: %s → %s  (%.1f years)", data_start.date(), data_end.date(), total_years)

    if total_years < required:
        logger.error(
            "Insufficient data: %.1f years available, need at least %d "
            "(train=%dyr + val=%dyr + test=%dyr). "
            "Add more historical data and re-run step0.",
            total_years, required, MIN_TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
        )
        sys.exit(1)

    # Anchor cutoffs to the data end
    test_start_dt = data_end - relativedelta(years=TEST_YEARS)
    val_start_dt  = test_start_dt - relativedelta(years=VAL_YEARS)

    # searchsorted: never include the boundary bar in the wrong split
    test_start_pos = df.index.searchsorted(test_start_dt, side="left")
    val_start_pos  = df.index.searchsorted(val_start_dt,  side="left")

    if val_start_pos == 0:
        logger.error(
            "val_start_dt %s is at or before the first bar %s — "
            "calendar cutoff leaves no training data.",
            val_start_dt.date(), data_start.date(),
        )
        sys.exit(1)

    if test_start_pos <= val_start_pos:
        logger.error(
            "test_start_pos (%d) <= val_start_pos (%d) — "
            "val window is empty. Check TEST_YEARS/VAL_YEARS constants.",
            test_start_pos, val_start_pos,
        )
        sys.exit(1)

    split_meta = {
        "train": {
            "rows":  val_start_pos,
            "start": df.index[0],
            "end":   df.index[val_start_pos - 1],
            "slice": slice(0, val_start_pos),
            "path":  ML_DIR / "train.parquet",
        },
        "validation": {
            "rows":  test_start_pos - val_start_pos,
            "start": df.index[val_start_pos],
            "end":   df.index[test_start_pos - 1],
            "slice": slice(val_start_pos, test_start_pos),
            "path":  ML_DIR / "validation.parquet",
        },
        "test": {
            "rows":  n - test_start_pos,
            "start": df.index[test_start_pos],
            "end":   df.index[-1],
            "slice": slice(test_start_pos, n),
            "path":  ML_DIR / "test.parquet",
        },
    }

    for name, meta in split_meta.items():
        logger.info(
            "%-12s %7d bars  %s → %s",
            name.capitalize() + ":",
            meta["rows"],
            meta["start"].date(),
            meta["end"].date(),
        )

    # Verify no overlap
    assert split_meta["train"]["end"] < split_meta["validation"]["start"], "Train/val overlap!"
    assert split_meta["validation"]["end"] < split_meta["test"]["start"],   "Val/test overlap!"
    logger.info("No leakage confirmed: train < val < test timestamps")

    rows = {}
    date_ranges = {}
    import pyarrow as pa
    import pyarrow.parquet as pq
    for split_name, meta in split_meta.items():
        split_df   = df.iloc[meta["slice"]]
        split_path = meta["path"]
        tbl = pa.Table.from_pandas(split_df, nthreads=1)
        pq.write_table(tbl, split_path, compression="snappy")
        del tbl
        rows[split_name] = len(split_df)
        date_ranges[split_name] = {
            "start": str(meta["start"]),
            "end":   str(meta["end"]),
        }

    summary = {
        "split_method": "calendar",
        "test_years":   TEST_YEARS,
        "val_years":    VAL_YEARS,
        "split_ratios": {
            "train":      round(rows["train"] / n, 4),
            "validation": round(rows["validation"] / n, 4),
            "test":       round(rows["test"] / n, 4),
        },
        "rows":        rows,
        "date_ranges": date_ranges,
        "features":    len(df.columns),
        "leakage_check": "PASS",
    }
    with open(ML_DIR / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n=== SPLIT COMPLETE (CALENDAR, no shuffling) ===")
    print(f"  Train:      {rows['train']:>7,} bars  {date_ranges['train']['start'][:10]} → {date_ranges['train']['end'][:10]}")
    print(f"  Validation: {rows['validation']:>7,} bars  {date_ranges['validation']['start'][:10]} → {date_ranges['validation']['end'][:10]}  ← Round 1 backtest")
    print(f"  Test:       {rows['test']:>7,} bars  {date_ranges['test']['start'][:10]} → {date_ranges['test']['end'][:10]}  ← Blind / Round 2 backtest")
    print(f"  Features:   {len(df.columns)}")
    print(f"  Leakage check: PASS")
    return rows


if __name__ == "__main__":
    main()
