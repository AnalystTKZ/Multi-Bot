#!/usr/bin/env python3
"""
Step 5: Expanding 2-year train / 1-year internal-validation folds, plus train
tail and blind-test windows.

Split logic:
  test  = final 2 years of data, never used by training/validation
  train = all data before the final 2-year blind test
  train_tail = final 2 years inside train; Round 1 backtests this seen data
  folds = expanding train window starting at the first available bar, with a
          2-year minimum train span and the following 1-year internal validation
          window over pre-test history

Example:
  fold_000:   train 2016-01-01..2017-12-31, val 2018-01-01..2018-12-31
  fold_001:   train 2016-01-01..2018-12-31, val 2019-01-01..2019-12-31
  train_tail: final 2 pre-test years, used for Round 1 seen backtest

train.parquet contains all pre-test data for model fitting. validation.parquet
is an internal training-validation alias only; it is not the Round 1 backtest
window. All outputs are isolated from the final blind test.
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

BASE = _ENV["base"]
OUTPUT_DIR = _ENV["processed"]
ML_DIR = _ENV["ml_training"] / "datasets"
ML_DIR.mkdir(parents=True, exist_ok=True)

MIN_TRAIN_YEARS = 2
VAL_YEARS = 1
TRAIN_TAIL_YEARS = 2
TEST_YEARS = 2
FOLD_STEP_YEARS = 1


def _date_str(ts) -> str:
    return str(ts)[:10]


def _slice_meta(df: pd.DataFrame, start_pos: int, end_pos: int, path: Path) -> dict:
    """Build metadata for a half-open iloc slice [start_pos, end_pos)."""
    if end_pos <= start_pos:
        raise ValueError("empty split slice")
    return {
        "rows": end_pos - start_pos,
        "start": df.index[start_pos],
        "end": df.index[end_pos - 1],
        "slice": slice(start_pos, end_pos),
        "path": path,
    }


def main():
    in_path = OUTPUT_DIR / "feature_engineered.parquet"
    if not in_path.exists():
        logger.error("feature_engineered.parquet not found - run step4 first")
        sys.exit(1)

    logger.info("Loading feature-engineered data...")
    df = pd.read_parquet(in_path)
    df = df.sort_index()
    n = len(df)
    logger.info("Loaded %d rows, %d features", n, len(df.columns))

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    data_start = df.index[0]
    data_end = df.index[-1]
    total_years = (data_end - data_start).days / 365.25
    required = MIN_TRAIN_YEARS + VAL_YEARS + TEST_YEARS

    logger.info("Data span: %s -> %s  (%.1f years)", data_start.date(), data_end.date(), total_years)
    if total_years < required:
        logger.error(
            "Insufficient data: %.1f years available, need at least %d "
            "(min_train=%dyr + val=%dyr + test=%dyr). Add more historical data and re-run step0.",
            total_years,
            required,
            MIN_TRAIN_YEARS,
            VAL_YEARS,
            TEST_YEARS,
        )
        sys.exit(1)

    test_start_dt = data_end - relativedelta(years=TEST_YEARS)
    test_start_pos = df.index.searchsorted(test_start_dt, side="left")
    if test_start_pos <= 0 or test_start_pos >= n:
        logger.error("Invalid test_start_dt %s for data span", test_start_dt.date())
        sys.exit(1)

    fold_meta: list[dict] = []
    train_start_dt = data_start
    val_start_dt = data_start + relativedelta(years=MIN_TRAIN_YEARS)
    fold_idx = 0

    def _append_fold(idx: int, train_end_dt, val_end_dt) -> bool:
        train_start_pos = df.index.searchsorted(train_start_dt, side="left")
        train_end_pos = df.index.searchsorted(train_end_dt, side="left")
        val_start_pos = train_end_pos
        val_end_pos = df.index.searchsorted(val_end_dt, side="left")

        if train_end_pos <= train_start_pos or val_end_pos <= val_start_pos:
            return False

        train_path = ML_DIR / f"train_fold_{idx:03d}.parquet"
        val_path = ML_DIR / f"validation_fold_{idx:03d}.parquet"
        train_meta = _slice_meta(df, train_start_pos, train_end_pos, train_path)
        val_meta = _slice_meta(df, val_start_pos, val_end_pos, val_path)
        fold_meta.append(
            {
                "fold_id": f"fold_{idx:03d}",
                "index": idx,
                "train": train_meta,
                "validation": val_meta,
            }
        )
        logger.info(
            "Fold %03d train %s -> %s (%d bars), val %s -> %s (%d bars)",
            idx,
            train_meta["start"].date(),
            train_meta["end"].date(),
            train_meta["rows"],
            val_meta["start"].date(),
            val_meta["end"].date(),
            val_meta["rows"],
        )
        return True

    while True:
        train_end_dt = val_start_dt
        val_end_dt = val_start_dt + relativedelta(years=VAL_YEARS)
        if val_end_dt > test_start_dt:
            break

        if _append_fold(fold_idx, train_end_dt, val_end_dt):
            fold_idx += 1

        val_start_dt = val_start_dt + relativedelta(years=FOLD_STEP_YEARS)

    final_train_end_dt = test_start_dt - relativedelta(years=VAL_YEARS)
    if final_train_end_dt > train_start_dt + relativedelta(years=MIN_TRAIN_YEARS):
        latest_val_end = fold_meta[-1]["validation"]["end"] if fold_meta else None
        final_val_end_pos = df.index.searchsorted(test_start_dt, side="left")
        final_val_end = df.index[final_val_end_pos - 1] if final_val_end_pos > 0 else None
        if latest_val_end is None or final_val_end is None or latest_val_end < final_val_end:
            if _append_fold(fold_idx, final_train_end_dt, test_start_dt):
                fold_idx += 1

    if not fold_meta:
        logger.error(
            "No expanding train/validation folds fit before blind test start %s. Need at least %d pre-test years.",
            test_start_dt.date(),
            MIN_TRAIN_YEARS + VAL_YEARS,
        )
        sys.exit(1)

    latest_fold = fold_meta[-1]
    test_meta = _slice_meta(df, test_start_pos, n, ML_DIR / "test.parquet")
    train_meta = _slice_meta(df, 0, test_start_pos, ML_DIR / "train.parquet")
    train_tail_start_dt = test_start_dt - relativedelta(years=TRAIN_TAIL_YEARS)
    train_tail_start_pos = df.index.searchsorted(train_tail_start_dt, side="left")
    train_tail_meta = _slice_meta(df, train_tail_start_pos, test_start_pos, ML_DIR / "train_tail.parquet")

    split_meta = {
        "train": train_meta,
        "validation": {**latest_fold["validation"], "path": ML_DIR / "validation.parquet"},
        "train_tail": train_tail_meta,
        "test": test_meta,
    }

    assert split_meta["train"]["end"] < split_meta["test"]["start"], "Train/test overlap!"
    assert split_meta["train_tail"]["end"] < split_meta["test"]["start"], "Train-tail/test overlap!"
    for fold in fold_meta:
        assert fold["train"]["end"] < fold["validation"]["start"], "Fold train/val overlap!"
        assert fold["validation"]["end"] < test_meta["start"], "Fold validation touches blind test!"
    logger.info("No leakage confirmed: train/train_tail/internal folds end before final 2-year blind test")

    rows = {}
    date_ranges = {}
    import pyarrow as pa
    import pyarrow.parquet as pq

    def _write_split(meta: dict, path: Path) -> int:
        split_df = df.iloc[meta["slice"]]
        tbl = pa.Table.from_pandas(split_df, nthreads=1)
        pq.write_table(tbl, path, compression="snappy")
        del tbl
        return len(split_df)

    for fold in fold_meta:
        _write_split(fold["train"], fold["train"]["path"])
        _write_split(fold["validation"], fold["validation"]["path"])

    for split_name, meta in split_meta.items():
        rows[split_name] = _write_split(meta, meta["path"])
        date_ranges[split_name] = {
            "start": str(meta["start"]),
            "end": str(meta["end"]),
        }

    summary_folds = []
    for fold in fold_meta:
        summary_folds.append(
            {
                "fold_id": fold["fold_id"],
                "index": fold["index"],
                "min_train_years": MIN_TRAIN_YEARS,
                "train_window": "expanding",
                "validation_years": VAL_YEARS,
                "date_ranges": {
                    "train": {
                        "start": str(fold["train"]["start"]),
                        "end": str(fold["train"]["end"]),
                    },
                    "validation": {
                        "start": str(fold["validation"]["start"]),
                        "end": str(fold["validation"]["end"]),
                    },
                    "test": {
                        "start": str(test_meta["start"]),
                        "end": str(test_meta["end"]),
                    },
                },
                "rows": {
                    "train": int(fold["train"]["rows"]),
                    "validation": int(fold["validation"]["rows"]),
                    "test": int(test_meta["rows"]),
                },
                "paths": {
                    "train": str(fold["train"]["path"]),
                    "validation": str(fold["validation"]["path"]),
                    "test": str(test_meta["path"]),
                },
            }
        )

    summary = {
        "split_method": "expanding_calendar",
        "train_window": "expanding",
        "min_train_years": MIN_TRAIN_YEARS,
        "val_years": VAL_YEARS,
        "train_tail_years": TRAIN_TAIL_YEARS,
        "test_years": TEST_YEARS,
        "fold_step_years": FOLD_STEP_YEARS,
        "selected_fold": latest_fold["fold_id"],
        "fold_count": len(fold_meta),
        "folds": summary_folds,
        "split_ratios": {
            "train": round(rows["train"] / n, 4),
            "validation": round(rows["validation"] / n, 4),
            "train_tail": round(rows["train_tail"] / n, 4),
            "test": round(rows["test"] / n, 4),
        },
        "rows": rows,
        "date_ranges": date_ranges,
        "features": len(df.columns),
        "leakage_check": "PASS",
        "blind_test_policy": "final_2_years_excluded_from_train_train_tail_and_internal_validation_folds",
    }
    with open(ML_DIR / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== SPLIT COMPLETE (EXPANDING CALENDAR, no shuffling) ===")
    print(f"  Folds:      {len(fold_meta):>7} expanding folds (min {MIN_TRAIN_YEARS}y train + {VAL_YEARS}y val, step={FOLD_STEP_YEARS}y)")
    print(f"  Selected:   {latest_fold['fold_id']} for internal validation alias")
    print(f"  Train:      {rows['train']:>7,} bars  {_date_str(date_ranges['train']['start'])} -> {_date_str(date_ranges['train']['end'])}  <- model fitting")
    print(f"  TrainTail:  {rows['train_tail']:>7,} bars  {_date_str(date_ranges['train_tail']['start'])} -> {_date_str(date_ranges['train_tail']['end'])}  <- Round 1 seen backtest")
    print(f"  Validation: {rows['validation']:>7,} bars  {_date_str(date_ranges['validation']['start'])} -> {_date_str(date_ranges['validation']['end'])}  <- internal only")
    print(f"  Test:       {rows['test']:>7,} bars  {_date_str(date_ranges['test']['start'])} -> {_date_str(date_ranges['test']['end'])}  <- Blind / Round 2")
    print(f"  Features:   {len(df.columns)}")
    print("  Leakage check: PASS")
    return rows


if __name__ == "__main__":
    main()
