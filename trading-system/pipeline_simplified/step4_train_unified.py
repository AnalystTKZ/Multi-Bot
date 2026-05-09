#!/usr/bin/env python3
"""Step 4: train the unified direction + regime model."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "trading-engine"))

from models.unified_direction_regime import UnifiedDirectionRegimePredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("simple_step4_train_unified")

DATASET_DIR = BASE / "ml_training" / "datasets" / "simple_datasets"
METRICS_DIR = BASE / "ml_training" / "simple_metrics"


def _cap_per_symbol(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    if cap <= 0 or "symbol" not in df.columns:
        return df
    parts = []
    for _symbol, group in df.groupby("symbol", sort=False):
        if len(group) > cap:
            parts.append(group.tail(cap))
        else:
            parts.append(group)
    return pd.concat(parts, axis=0).sort_index()


def main() -> None:
    train_path = DATASET_DIR / "train.parquet"
    val_path = DATASET_DIR / "validation.parquet"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("simple train/validation parquet files are missing")
    train = pd.read_parquet(train_path).sort_index()
    validation = pd.read_parquet(val_path).sort_index()
    train_cap = int(os.getenv("UNIFIED_MAX_TRAIN_ROWS_PER_SYMBOL", "0"))
    val_cap = int(os.getenv("UNIFIED_MAX_VAL_ROWS_PER_SYMBOL", "0"))
    train = _cap_per_symbol(train, train_cap)
    validation = _cap_per_symbol(validation, val_cap)
    logger.info("training rows=%d validation rows=%d", len(train), len(validation))
    model = UnifiedDirectionRegimePredictor(load_existing=False)
    history = model.train_from_frames(
        train,
        validation,
        horizon=int(os.getenv("UNIFIED_HORIZON_BARS", "4")),
        epochs=int(os.getenv("UNIFIED_EPOCHS", "8")),
        batch_size=int(os.getenv("UNIFIED_BATCH_SIZE", "512")),
        learning_rate=float(os.getenv("UNIFIED_LR", "0.0003")),
    )
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "unified_training_summary.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    logger.info("unified model trained: %s", history)


if __name__ == "__main__":
    main()
