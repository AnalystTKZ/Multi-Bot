#!/usr/bin/env python3
"""Step 1: prepare a small symbol/timeframe subset for simplify-v2."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "trading-engine"))

from indicators.market_structure import compute_atr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("simple_step1_data_prep")

INPUT_DIR = BASE / "processed_data" / "histdata"
OUTPUT_DIR = BASE / "processed_data" / "simple" / "ohlcv"
DEFAULT_SYMBOLS = "XAUUSD,EURUSD,USDJPY,EURJPY,GBPJPY,GBPUSD"
SYMBOLS = [s.strip().upper() for s in os.getenv("SIMPLE_SYMBOLS", DEFAULT_SYMBOLS).split(",") if s.strip()]
TIMEFRAMES = [s.strip().upper() for s in os.getenv("SIMPLE_TIMEFRAMES", "5M,15M,1H,4H,1D").split(",") if s.strip()]
START_DATE = os.getenv("SIMPLE_START_DATE", "2020-01-01")


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}")
    out = df.loc[df.index >= pd.Timestamp(START_DATE, tz="UTC"), required].copy()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out[required] = out[required].apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    out["volume"] = out["volume"].fillna(0.0)
    out["atr_14"] = compute_atr(out, 14).fillna((out["high"] - out["low"]).abs())
    if hasattr(out.index, "hour"):
        h = out.index.hour
        out["is_asian"] = ((h >= 0) & (h < 7)).astype(float)
        out["is_london"] = ((h >= 7) & (h < 12)).astype(float)
        out["is_ny"] = ((h >= 13) & (h < 18)).astype(float)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"symbols": SYMBOLS, "timeframes": TIMEFRAMES, "files": []}
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            src = INPUT_DIR / f"{symbol}_{timeframe}.parquet"
            if not src.exists():
                raise FileNotFoundError(f"{src} not found; run pipeline/step0_resample.py first")
            df = _normalise(pd.read_parquet(src))
            dst = OUTPUT_DIR / f"{symbol}_{timeframe}.parquet"
            df.to_parquet(dst, compression="snappy")
            manifest["files"].append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "path": str(dst),
                    "rows": int(len(df)),
                    "start": str(df.index.min()),
                    "end": str(df.index.max()),
                }
            )
            logger.info("wrote %s %s rows=%d", symbol, timeframe, len(df))
    (OUTPUT_DIR.parent / "ohlcv_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
