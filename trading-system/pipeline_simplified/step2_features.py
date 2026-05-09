#!/usr/bin/env python3
"""Step 2: build the unified model feature table from engine FeatureEngine."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "trading-engine"))

from services.feature_engine import FeatureEngine, SEQUENCE_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("simple_step2_features")

INPUT_DIR = BASE / "processed_data" / "simple" / "ohlcv"
OUTPUT_DIR = BASE / "processed_data" / "simple"
DEFAULT_SYMBOLS = "XAUUSD,EURUSD,USDJPY,EURJPY,GBPJPY,GBPUSD"
SYMBOLS = [s.strip().upper() for s in os.getenv("SIMPLE_SYMBOLS", DEFAULT_SYMBOLS).split(",") if s.strip()]


def _read(symbol: str, timeframe: str) -> pd.DataFrame:
    path = INPUT_DIR / f"{symbol}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def _trim_to_context(base: pd.DataFrame, htf: dict[str, pd.DataFrame]) -> pd.DataFrame:
    starts = [base.index.min()]
    ends = [base.index.max()]
    for frame in htf.values():
        starts.append(frame.index.min())
        ends.append(frame.index.max())
    start = max(starts) + pd.Timedelta(days=3)
    end = min(ends)
    trimmed = base[(base.index >= start) & (base.index <= end)].copy()
    if len(trimmed) < 500:
        raise ValueError(f"insufficient common MTF overlap after trim: {len(trimmed)} rows")
    return trimmed


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fe = FeatureEngine()
    frames = []
    for idx, symbol in enumerate(SYMBOLS):
        base = _read(symbol, "15M")
        htf = {
            "5M": _read(symbol, "5M"),
            "1H": _read(symbol, "1H"),
            "4H": _read(symbol, "4H"),
            "1D": _read(symbol, "1D"),
        }
        base = _trim_to_context(base, htf)
        features = fe._build_sequence_df(base, htf, symbol=symbol)
        keep = ["open", "high", "low", "close", "volume", "atr_14", *SEQUENCE_FEATURES]
        keep = list(dict.fromkeys([col for col in keep if col in features.columns]))
        out = features[keep].copy()
        out["symbol"] = symbol
        out["symbol_id"] = float(idx)
        frames.append(out)
        logger.info("%s features rows=%d cols=%d", symbol, len(out), len(out.columns))
    combined = pd.concat(frames, axis=0).sort_index()
    combined = combined.replace([float("inf"), float("-inf")], pd.NA).ffill().fillna(0.0)
    out_path = OUTPUT_DIR / "features.parquet"
    combined.to_parquet(out_path, compression="snappy")
    manifest = {
        "path": str(out_path),
        "rows": int(len(combined)),
        "columns": list(combined.columns),
        "sequence_features": list(SEQUENCE_FEATURES),
        "symbols": SYMBOLS,
        "start": str(combined.index.min()),
        "end": str(combined.index.max()),
    }
    (OUTPUT_DIR / "feature_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("wrote %s rows=%d cols=%d", out_path, len(combined), len(combined.columns))


if __name__ == "__main__":
    main()
