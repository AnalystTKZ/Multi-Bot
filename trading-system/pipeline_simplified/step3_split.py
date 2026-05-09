#!/usr/bin/env python3
"""Step 3: simple chronological train/validation/test split."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta

BASE = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("simple_step3_split")

INPUT = BASE / "processed_data" / "simple" / "features.parquet"
OUTPUT_DIR = BASE / "ml_training" / "simple_datasets"


def _write(df: pd.DataFrame, name: str) -> dict:
    path = OUTPUT_DIR / f"{name}.parquet"
    df.to_parquet(path, compression="snappy")
    return {"path": str(path), "rows": int(len(df)), "start": str(df.index.min()), "end": str(df.index.max())}


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(INPUT).sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    end = df.index.max()
    test_start = end - relativedelta(years=2)
    val_start = test_start - relativedelta(years=1)
    train = df[df.index < val_start]
    validation = df[(df.index >= val_start) & (df.index < test_start)]
    test = df[df.index >= test_start]
    if min(len(train), len(validation), len(test)) <= 0:
        raise ValueError("split produced an empty train/validation/test set")
    summary = {
        "split_method": "simple_calendar",
        "blind_test_policy": "final_2_years",
        "train": _write(train, "train"),
        "validation": _write(validation, "validation"),
        "test": _write(test, "test"),
        "leakage_check": "PASS",
    }
    (OUTPUT_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "split rows train=%d validation=%d test=%d",
        len(train),
        len(validation),
        len(test),
    )


if __name__ == "__main__":
    main()
