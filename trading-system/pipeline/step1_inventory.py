#!/usr/bin/env python3
"""
Step 1: Data Discovery — scan training_data/ and write raw_inventory.json.
Processes one file at a time. Never holds more than one DataFrame in memory.
"""
from __future__ import annotations
import csv
import gc
import json
import logging
from datetime import timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("step1")

BASE = Path(__file__).resolve().parent.parent
TRAINING_DIR = BASE / "training_data"
OUTPUT_DIR = BASE / "processed_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _classify(category: str) -> str:
    m = {"forex": "forex", "commodities": "commodity", "indices": "index",
         "stocks": "index", "crypto": "crypto", "fundamental": "fundamental"}
    return m.get(category.lower(), "unknown")


def _parse_tf(stem: str) -> str:
    for part in reversed(stem.split("_")):
        p = part.lower()
        if p in ("15m", "1h", "4h", "1d", "5m", "m1", "1min"):
            return p
        if "histdata" in p:
            return "m1"
        if (p.endswith("m") or p.endswith("h") or p.endswith("d")) and p[:-1].isdigit():
            return p
    return "unknown"


def inspect_file(path: Path) -> dict:
    """Read only first and last rows + header — minimal RAM."""
    result = {"status": "ok", "rows": 0, "file_size_kb": round(path.stat().st_size / 1024, 1)}
    try:
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader, [])
        cols = [c.lower().strip() for c in header]
        result["columns"] = len(cols)
        result["is_ohlcv"] = all(c in cols for c in ["open", "high", "low", "close"])
        result["has_volume"] = "volume" in cols
        result["column_names"] = cols[:20]

        # Count rows without loading into pandas
        with open(path) as f:
            n = sum(1 for _ in f) - 1
        result["rows"] = n

        # Read only first and last row for date range
        df_head = pd.read_csv(path, index_col=0, parse_dates=True, nrows=1)
        df_tail = pd.read_csv(path, index_col=0, parse_dates=True, skiprows=range(1, max(1, n - 1)))
        try:
            result["date_start"] = str(pd.to_datetime(df_head.index[0], utc=True))
            result["date_end"]   = str(pd.to_datetime(df_tail.index[-1], utc=True))
        except Exception:
            result["date_start"] = result["date_end"] = None
        del df_head, df_tail
        gc.collect()

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    return result


def main():
    inventory = {
        "scan_date": pd.Timestamp.now(tz=timezone.utc).isoformat(),
        "training_data_root": str(TRAINING_DIR),
        "files": [],
        "summary": {},
    }

    by_cat: dict[str, int] = {}
    by_type: dict[str, int] = {}
    symbols: set[str] = set()
    timeframes: set[str] = set()

    for cat_dir in sorted(d for d in TRAINING_DIR.iterdir() if d.is_dir()):
        category = cat_dir.name
        for csv_path in sorted(cat_dir.glob("*.csv")):
            stem = csv_path.stem
            parts = stem.split("_")
            tf = _parse_tf(stem)
            symbol = parts[0].upper()
            asset_type = _classify(category)

            meta = inspect_file(csv_path)
            meta.update({
                "category": category, "symbol": symbol,
                "timeframe": tf, "asset_type": asset_type,
                "filename": csv_path.name,
                "path": str(csv_path.relative_to(BASE)),
            })
            inventory["files"].append(meta)

            by_cat[category] = by_cat.get(category, 0) + 1
            by_type[asset_type] = by_type.get(asset_type, 0) + 1
            symbols.add(symbol)
            timeframes.add(tf)

            logger.info("Scanned %s/%s: %d rows", category, csv_path.name, meta["rows"])

    inventory["summary"] = {
        "total_files": len(inventory["files"]),
        "by_category": by_cat,
        "by_asset_type": by_type,
        "unique_symbols": sorted(symbols),
        "unique_timeframes": sorted(timeframes),
        "total_symbols": len(symbols),
    }

    out = OUTPUT_DIR / "raw_inventory.json"
    with open(out, "w") as f:
        json.dump(inventory, f, indent=2, default=str)

    print(f"\n=== DATA INVENTORY ===")
    print(f"  Total files: {len(inventory['files'])}")
    for k, v in by_cat.items():
        print(f"  {k}: {v} files")
    print(f"  Symbols: {len(symbols)}, Timeframes: {sorted(timeframes)}")
    print(f"  Output: {out}")


if __name__ == "__main__":
    main()
