#!/usr/bin/env python3
"""
Step 5: Strict time-based train/val/test split (70/15/15). No shuffling.
Input:  processed_data/feature_engineered.parquet
Output: ml_training/datasets/train.parquet, validation.parquet, test.parquet
"""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("step5_split")

BASE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE / "processed_data"
ML_DIR = BASE / "ml_training" / "datasets"
ML_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15


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
    if n < 3:
        logger.error("Need at least 3 rows to create train/validation/test splits, got %d", n)
        sys.exit(1)

    # Strict time-based cut points
    train_end_idx = int(n * TRAIN_RATIO)
    val_end_idx   = int(n * (TRAIN_RATIO + VAL_RATIO))
    if train_end_idx <= 0 or val_end_idx <= train_end_idx or val_end_idx >= n:
        logger.error(
            "Invalid split boundaries for %d rows: train_end=%d val_end=%d",
            n, train_end_idx, val_end_idx,
        )
        sys.exit(1)

    train_start, train_end = 0, train_end_idx
    val_start, val_end = train_end_idx, val_end_idx
    test_start, test_end = val_end_idx, n

    split_meta = {
        "train": {
            "rows": train_end - train_start,
            "start": df.index[train_start],
            "end": df.index[train_end - 1],
            "slice": slice(train_start, train_end),
            "path": ML_DIR / "train.parquet",
        },
        "validation": {
            "rows": val_end - val_start,
            "start": df.index[val_start],
            "end": df.index[val_end - 1],
            "slice": slice(val_start, val_end),
            "path": ML_DIR / "validation.parquet",
        },
        "test": {
            "rows": test_end - test_start,
            "start": df.index[test_start],
            "end": df.index[test_end - 1],
            "slice": slice(test_start, test_end),
            "path": ML_DIR / "test.parquet",
        },
    }

    logger.info("Train: %d bars (%s → %s)", split_meta["train"]["rows"], split_meta["train"]["start"].date(), split_meta["train"]["end"].date())
    logger.info("Val:   %d bars (%s → %s)", split_meta["validation"]["rows"], split_meta["validation"]["start"].date(), split_meta["validation"]["end"].date())
    logger.info("Test:  %d bars (%s → %s)", split_meta["test"]["rows"], split_meta["test"]["start"].date(), split_meta["test"]["end"].date())

    # Verify no overlap
    assert split_meta["train"]["end"] < split_meta["validation"]["start"], "Train/val overlap!"
    assert split_meta["validation"]["end"] < split_meta["test"]["start"], "Val/test overlap!"
    logger.info("No leakage confirmed: train < val < test timestamps")

    rows = {}
    date_ranges = {}
    for split_name, meta in split_meta.items():
        split_df = df.iloc[meta["slice"]]
        split_path = meta["path"]
        import pyarrow as pa
        import pyarrow.parquet as pq
        _tbl = pa.Table.from_pandas(split_df, nthreads=1)
        pq.write_table(_tbl, split_path, compression="snappy")
        del _tbl
        rows[split_name] = len(split_df)
        date_ranges[split_name] = {
            "start": str(meta["start"]),
            "end": str(meta["end"]),
        }

    summary = {
        "split_ratios": {"train": TRAIN_RATIO, "validation": VAL_RATIO, "test": TEST_RATIO},
        "rows": rows,
        "date_ranges": date_ranges,
        "features": len(df.columns),
        "leakage_check": "PASS",
    }
    with open(ML_DIR / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n=== SPLIT COMPLETE (no shuffling, time-based) ===")
    print(f"  Train:      {rows['train']:>7,} bars  {date_ranges['train']['start'][:10]} → {date_ranges['train']['end'][:10]}")
    print(f"  Validation: {rows['validation']:>7,} bars  {date_ranges['validation']['start'][:10]} → {date_ranges['validation']['end'][:10]}")
    print(f"  Test:       {rows['test']:>7,} bars  {date_ranges['test']['start'][:10]} → {date_ranges['test']['end'][:10]}")
    print(f"  Features: {len(df.columns)}")
    print(f"  Leakage check: PASS")
    return rows


if __name__ == "__main__":
    main()
