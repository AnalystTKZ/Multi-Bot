#!/usr/bin/env python3
"""
Step 2: Data Cleaning
Source: processed_data/histdata/{SYM}_15M.parquet — M1-resampled by step0_resample.py.
No CSV fallbacks. Run step0 first if histdata/ is empty.

Macro columns from training_data/unified/ are forward-merged onto the M1 data where available.
Output: processed_data/clean/{SYMBOL}_15M.parquet
"""
from __future__ import annotations
import csv as _csv
import gc
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("step2")

BASE        = Path(__file__).resolve().parent.parent
HIST_DIR    = BASE / "processed_data" / "histdata"   # MTF parquet outputs from step 0
UNIFIED_DIR = BASE / "training_data" / "unified"
OUTPUT_DIR  = BASE / "processed_data" / "clean"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY = [
    "AUDUSD", "EURGBP", "EURJPY", "EURUSD", "GBPJPY",
    "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
    "XAUUSD",
]

# Macro columns to pull from unified and merge onto M1 data
MACRO_COLS = {
    "fedfunds", "dff", "ecbdfr", "dgs2", "dgs10",
    "cpiaucsl", "cpilfesl", "unrate", "payems", "vixcls",
    "yield_curve_2_10", "t10y2y",
    "dff_hike_flag", "dff_cut_flag",
    "macro_surprise_composite",
    "fed_decision_flag", "ecb_decision_flag", "boj_decision_flag",
    "election_flag", "fed_speech_flag",
    "event_impact_score", "hours_since_last_event", "hours_to_next_event",
    "spx_ret1d", "dxy_ret1d", "vix_ret1d", "gold_fut_ret1d",
    "risk_on_score", "dxy_strength", "us10y_yield",
    "asia_session", "london_session", "ny_session",
    "hour", "day_of_week", "month",
    "hour_sin", "hour_cos",
    "1h_open", "1h_high", "1h_low", "1h_close",
    "4h_open", "4h_high", "4h_low", "4h_close",
    "1d_open", "1d_high", "1d_low", "1d_close",
}


def _load_macro_from_unified(sym: str) -> pd.DataFrame | None:
    """Load only macro columns from unified CSV. Returns None if not available."""
    path = UNIFIED_DIR / f"{sym}_15m_unified.csv"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            orig_cols = next(_csv.reader(f))
        keep = [orig_cols[0]] + [c for c in orig_cols[1:] if c.lower() in MACRO_COLS]
        if len(keep) <= 1:
            return None
        df = pd.read_csv(path, usecols=keep, index_col=0, parse_dates=True, low_memory=False)
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[df.index.notna()].sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df
    except Exception as e:
        logger.warning("Macro load failed %s: %s", sym, e)
        return None


def clean_symbol(sym: str) -> dict | None:
    out_path = OUTPUT_DIR / f"{sym}_15M.parquet"
    if out_path.exists():
        df_check = pd.read_parquet(out_path, columns=["close"])
        n = len(df_check)
        del df_check; gc.collect()
        logger.info("SKIP %s (already cleaned, %d bars)", sym, n)
        return {"symbol": sym, "rows": n, "status": "cached"}

    # ── Load from step-0 M1-resampled histdata (the only source) ─────────────
    # Run step0_resample.py first if this file is missing.
    hist_path = HIST_DIR / f"{sym}_15M.parquet"
    if not hist_path.exists():
        logger.error(
            "MISSING: %s — run step0_resample.py first to resample M1 → all TFs", hist_path
        )
        return None

    df = pd.read_parquet(hist_path)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[df.index.notna()].sort_index()
    logger.info("Loaded M1-resampled %s: %d bars (%s → %s)",
                sym, len(df), df.index.min().date(), df.index.max().date())

    # ── Normalize OHLCV ───────────────────────────────────────────────────────
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            df[col] = df.get("close", 1.0)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df[~df.index.duplicated(keep="last")]
    df = df[df["close"].notna() & (df["close"] > 0)]
    df[["open", "high", "low", "close", "volume"]] = \
        df[["open", "high", "low", "close", "volume"]].ffill(limit=5)

    if len(df) < 100:
        logger.warning("Too few rows after cleaning %s: %d", sym, len(df))
        del df; gc.collect()
        return None

    # ── Merge HTF OHLCV columns from step-0 MTF outputs ──────────────────────
    # Gives downstream steps ready-made 1H/4H/1D context without resampling
    for tf_suffix, col_prefix in [("1H", "htf_1h"), ("4H", "htf_4h"), ("1D", "htf_1d")]:
        htf_path = HIST_DIR / f"{sym}_{tf_suffix}.parquet"
        if htf_path.exists():
            htf = pd.read_parquet(htf_path, columns=["open", "high", "low", "close"])
            htf.columns = [f"{col_prefix}_{c}" for c in htf.columns]
            htf.index = pd.to_datetime(htf.index, utc=True, errors="coerce")
            for col in htf.columns:
                df[col] = htf[col].reindex(df.index, method="ffill")
            del htf; gc.collect()

    # ── Merge tick volume if available ────────────────────────────────────────
    tick_vol_path = HIST_DIR / f"{sym}_tick_vol.parquet"
    if tick_vol_path.exists():
        tv = pd.read_parquet(tick_vol_path)
        tv.index = pd.to_datetime(tv.index, utc=True, errors="coerce")
        tv_ri = tv["tick_volume"].reindex(df.index, fill_value=0)
        df["tick_volume"] = tv_ri
        del tv, tv_ri; gc.collect()

    # ── Merge macro columns from unified (forward-filled onto full M1 range) ──
    macro = _load_macro_from_unified(sym)
    if macro is not None and len(macro) > 0:
        # Only add columns not already in df
        new_cols = [c for c in macro.columns if c not in df.columns]
        if new_cols:
            macro_ri = macro[new_cols].reindex(df.index, method="ffill")
            for col in new_cols:
                df[col] = macro_ri[col]
            logger.info("Merged %d macro columns onto %s", len(new_cols), sym)
        del macro, macro_ri; gc.collect()

    start = str(df.index.min().date())
    end   = str(df.index.max().date())
    n     = len(df)
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(df, nthreads=1)
    pq.write_table(table, out_path, compression="snappy")
    del table
    logger.info("Saved %s: %d bars, %d cols (%s → %s)", sym, n, len(df.columns), start, end)
    del df; gc.collect()
    return {"symbol": sym, "rows": n, "start": start, "end": end, "status": "ok"}


def main():
    summary = {"assets": {}, "total_bars": 0}

    for sym in PRIMARY:
        result = clean_symbol(sym)
        if result:
            summary["assets"][sym] = result
            summary["total_bars"] += result.get("rows", 0)

    with open(BASE / "processed_data" / "cleaning_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n=== CLEANING COMPLETE ===")
    for sym, info in summary["assets"].items():
        print(f"  {sym}: {info.get('rows', 0):,} bars  "
              f"{info.get('start','')} → {info.get('end','')}  [{info.get('status')}]")
    print(f"  Total bars: {summary['total_bars']:,}")


if __name__ == "__main__":
    main()
