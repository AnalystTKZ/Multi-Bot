#!/usr/bin/env python3
"""
retrain_incremental.py — Incremental model retraining.

Usage:
    python retrain_incremental.py              # retrain all models
    python retrain_incremental.py --model gru
    python retrain_incremental.py --model regime
    python retrain_incremental.py --model quality
    python retrain_incremental.py --model rl
    python retrain_incremental.py --model sentiment  # no-op (pre-trained)
    python retrain_incremental.py --dry-run          # validate without saving
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Environment abstraction — resolves paths for both local and Kaggle
_env_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "env_config.py")
sys.path.insert(0, os.path.dirname(os.path.abspath(_env_config_path)))
from env_config import get_env, ensure_output_dirs
_ENV = get_env()
ensure_output_dirs(_ENV)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("retrain")
logger.info("retrain environment: %s", "KAGGLE" if _ENV["on_kaggle"] else "LOCAL")

import os as _os
# Unblock both T4 GPUs — remove any empty CUDA_VISIBLE_DEVICES mask
if _os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
    del _os.environ["CUDA_VISIBLE_DEVICES"]

try:
    import torch as _torch
    if _torch.cuda.is_available():
        _n_gpu = _torch.cuda.device_count()
        _DEVICE = "cuda"
        logger.info("Device: CUDA (%d GPU(s))", _n_gpu)
        for _i in range(_n_gpu):
            logger.info("  GPU %d: %s (%.1f GB)",
                        _i, _torch.cuda.get_device_name(_i),
                        _torch.cuda.get_device_properties(_i).total_memory / 1e9)
        # ── GPU performance flags ──────────────────────────────────────────────
        # cuDNN autotuner: picks fastest conv algorithm for fixed input shapes.
        _torch.backends.cudnn.benchmark = True
        # TF32: ~3× faster matmul on Ampere+ (T4 supports it); negligible accuracy loss.
        _torch.backends.cuda.matmul.allow_tf32 = True
        _torch.backends.cudnn.allow_tf32       = True
        logger.info("cuDNN benchmark=True, TF32 matmul=True")
    else:
        _DEVICE = "cpu"
        logger.info("Device: CPU")
        if _ENV["on_kaggle"]:
            raise RuntimeError(
                "retrain_incremental: CUDA not available on Kaggle — "
                "enable GPU accelerator in notebook settings."
            )
    # ── CPU thread config: use all 4 Kaggle CPUs ──────────────────────────────
    _n_cpu = int(os.getenv("RETRAIN_CPU_WORKERS", "4"))
    _torch.set_num_threads(_n_cpu)
    _torch.set_num_interop_threads(max(1, _n_cpu // 2))
    logger.info("PyTorch CPU threads: %d intra / %d interop", _n_cpu, max(1, _n_cpu // 2))
except RuntimeError:
    raise

# All paths resolved through env — no hardcoded absolute paths
DATA_DIR        = str(_ENV["data"])
JOURNAL_PATH    = str(_ENV["engine"] / "logs" / "trade_journal_detailed.jsonl")
WEIGHTS_DIR     = str(_ENV["weights"])
BACKUP_DIR      = str(_ENV["weights"] / "backups")
MAX_BACKUPS = 5
MONTHS_OF_DATA = int(os.getenv("RETRAIN_MONTHS", "0"))  # 0 = use all available data
GRU_EPOCHS = int(os.getenv("GRU_EPOCHS", "50"))
# 1024 per GPU × 2 GPUs (DataParallel) = 2048 effective batch; grad_accum×4 = 8192 logical.
# Overrideable via env var for memory-constrained runs.
GRU_BATCH_SIZE = int(os.getenv("GRU_BATCH_SIZE", "1024"))
MAJOR_SYMBOLS = [
    "AUDUSD", "EURGBP", "EURJPY", "EURUSD", "GBPJPY",
    "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY", "XAUUSD",
]
# All training timeframes — derived from step0 pipeline outputs
ALL_TIMEFRAMES = ["5M", "15M", "1H", "4H", "1D", "1W", "1MN"]
# GRU uses short-bar timeframes for sequence patterns
GRU_TIMEFRAMES = ["5M", "15M", "1H", "4H"]
# Hierarchical regime cascade:
# HTF (4H): 3-class bias (BIAS_UP/DOWN/NEUTRAL) — trained with mode="htf_bias"
# LTF (1H): 4-class behaviour (TRENDING/RANGING/CONSOLIDATING/VOLATILE) — trained with mode="ltf_behaviour"
REGIME_HTF_TF = ["4H"]         # HTF bias classifier source timeframe (was REGIME_TF_4H)
REGIME_LTF_TF = ["1H"]         # LTF behaviour classifier source timeframe (was REGIME_TF_1H)
REGIME_TF_4H = REGIME_HTF_TF   # backward compat alias
REGIME_TF_1H = REGIME_LTF_TF   # backward compat alias
REGIME_TIMEFRAMES = ["1H", "4H"]  # kept for backwards compat (covers both)
# Root of the pipeline processed data
_PROCESSED_DIR = str(_ENV["processed"] / "histdata")
MACRO_CORR_PATH = str(_ENV["weights"] / "macro_correlations.json")
# Fixed list — must match feature_engine.INDEX_NAMES exactly so macro correlations
# and REGIME_FEATURES use the same 17 indices regardless of directory contents.
INDEX_KEYS = [
    "asx200", "cac40", "dax", "djia", "dxy",
    "eurostoxx", "ftse", "gold_fut", "hsi", "nasdaq",
    "nikkei", "oil_fut", "spx", "us10y", "us30y",
    "us3m", "vix",
]
MACRO_KEYS = INDEX_KEYS

_SYMBOL_TO_GROUP = {
    "EURUSD": "dollar", "GBPUSD": "dollar", "USDJPY": "dollar",
    "USDCHF": "dollar", "USDCAD": "dollar", "AUDUSD": "dollar", "NZDUSD": "dollar",
    "EURGBP": "cross",  "EURJPY": "cross",  "GBPJPY": "cross",
    "XAUUSD": "gold",
}


def _group_for_symbol(sym: str) -> str:
    return _SYMBOL_TO_GROUP.get(sym.upper(), "dollar")


def _path_has_artifact(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    if p.is_file():
        return p.stat().st_size > 0
    return any(child.is_file() and child.stat().st_size > 0 for child in p.rglob("*"))


def _gru_artifact_exists() -> bool:
    return _path_has_artifact(os.path.join(WEIGHTS_DIR, "gru_lstm", "model.pt"))


def _regime_artifact_exists() -> bool:
    return _htf_regime_artifact_exists() and _ltf_regime_artifact_exists()


def _htf_regime_artifact_exists() -> bool:
    """Check if the new HTF bias regime weights exist (regime_htf.pkl)."""
    return _path_has_artifact(os.path.join(WEIGHTS_DIR, "regime_htf.pkl"))


def _ltf_regime_artifact_exists() -> bool:
    """Check if the new LTF behaviour regime weights exist (regime_ltf.pkl)."""
    return _path_has_artifact(os.path.join(WEIGHTS_DIR, "regime_ltf.pkl"))


def _quality_artifact_exists() -> bool:
    return _path_has_artifact(os.path.join(WEIGHTS_DIR, "quality_scorer.pkl"))


def _rl_artifact_exists() -> bool:
    return (
        _path_has_artifact(os.path.join(WEIGHTS_DIR, "rl_ppo", "model.zip"))
        or _path_has_artifact(os.path.join(WEIGHTS_DIR, "rl_ppo", "model"))
        or _path_has_artifact(os.path.join(WEIGHTS_DIR, "rl_ppo", "policy.pkl"))
    )


def _get_symbols(env_name: str, default: list[str]) -> list[str]:
    raw = os.getenv(env_name, "")
    if not raw.strip():
        return default
    return [s.strip().upper() for s in raw.split(",") if s.strip()]



def _load_macro_series() -> dict:
    import pandas as pd

    base = DATA_DIR
    idx_dir = os.path.join(base, "indices")
    fund_dir = os.path.join(base, "fundamental")

    def _load(path: str, date_col: str, value_col: str) -> "pd.Series | None":
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        if date_col not in df.columns or value_col not in df.columns:
            return None
        s = pd.to_datetime(df[date_col], utc=True, errors="coerce")
        v = pd.to_numeric(df[value_col], errors="coerce")
        out = pd.Series(v.values, index=s).dropna()
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out

    series = {}
    if os.path.isdir(idx_dir):
        for name in sorted(os.listdir(idx_dir)):
            if not name.endswith("_1d.csv"):
                continue
            key = name.replace("_1d.csv", "").lower()
            series[key] = _load(os.path.join(idx_dir, name), "Datetime", "close")
    series["us10y_fred"] = _load(os.path.join(fund_dir, "treasury_10yr.csv"), "Date", "DGS10")
    series["us2y_fred"] = _load(os.path.join(fund_dir, "treasury_2yr.csv"), "Date", "DGS2")
    return {k: v for k, v in series.items() if v is not None and len(v) > 10}


def _update_macro_correlations(symbols: list[str]) -> None:
    import pandas as pd

    macro = _load_macro_series()
    if not macro:
        return

    result: dict = {"updated_at": datetime.now(timezone.utc).isoformat(), "symbols": {}}

    for sym in symbols:
        df = _load_ohlcv(sym, "1H")
        if df is None:
            df = _load_ohlcv(sym, "15M")
        if df is None or len(df) < 200:
            continue
        daily = df["close"].resample("1D").last().dropna()
        if len(daily) < 30:
            continue
        sym_ret = daily.pct_change().dropna()
        corrs = {}
        for key in MACRO_KEYS:
            s = macro.get(key)
            if s is None or len(s) < 30:
                continue
            if key in {"us10y", "us30y", "us3m"}:
                mret = s.diff().dropna()
            else:
                mret = s.pct_change().dropna()
            aligned = pd.concat([sym_ret, mret], axis=1, join="inner").dropna()
            if len(aligned) < 30:
                continue
            corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            if not pd.isna(corr):
                corrs[key] = corr
        if not corrs:
            continue
        top = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        result["symbols"][sym] = {
            "selected": [k for k, _ in top],
            "correlations": {k: round(v, 4) for k, v in top},
        }

    if result["symbols"]:
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        with open(MACRO_CORR_PATH, "w") as f:
            f.write(json.dumps(result, indent=2))


_SPLIT_BOUNDARIES: dict | None = None
_PARQUET_CACHE: dict = {}  # (symbol, tf) -> raw DataFrame, populated once on first read


def _load_split_boundaries() -> dict:
    """
    Load train/val/test date boundaries from ml_training/datasets/split_summary.json.
    Returns dict with keys 'train_end', 'val_end' as UTC Timestamps, or empty dict
    if the file doesn't exist (falls back to full-data training).
    """
    global _SPLIT_BOUNDARIES
    if _SPLIT_BOUNDARIES is not None:
        return _SPLIT_BOUNDARIES

    import pandas as pd

    split_path = os.path.join(
        str(_ENV["ml_training"]), "datasets", "split_summary.json"
    )
    if not os.path.exists(split_path):
        logger.warning("split_summary.json not found — training on full history")
        _SPLIT_BOUNDARIES = {}
        return _SPLIT_BOUNDARIES

    try:
        with open(split_path) as f:
            summary = json.load(f)
        ranges = summary.get("date_ranges", {})
        _SPLIT_BOUNDARIES = {
            "train_end": pd.Timestamp(ranges["train"]["end"], tz="UTC"),
            "val_end":   pd.Timestamp(ranges["validation"]["end"], tz="UTC"),
            "test_start": pd.Timestamp(ranges["test"]["start"], tz="UTC"),
            "test_end":   pd.Timestamp(ranges["test"]["end"], tz="UTC"),
        }
        logger.info(
            "Split boundaries loaded — train≤%s  val≤%s  test≤%s",
            _SPLIT_BOUNDARIES["train_end"].date(),
            _SPLIT_BOUNDARIES["val_end"].date(),
            _SPLIT_BOUNDARIES["test_end"].date(),
        )
    except Exception as exc:
        logger.warning("Failed to parse split_summary.json: %s", exc)
        _SPLIT_BOUNDARIES = {}
    return _SPLIT_BOUNDARIES


def _load_ohlcv(symbol: str, timeframe: str = "15M",
                split: str = "train") -> "pd.DataFrame | None":
    """Load OHLCV from processed_data/histdata/{SYM}_{TF}.parquet (step0 output).
    No CSV fallbacks — all data must be resampled from M1 by step0_resample.py.

    split: one of 'train', 'val', 'test', 'all'
      - 'train'  → rows up to and including train_end
      - 'val'    → rows after train_end up to val_end
      - 'test'   → rows after val_end (held-out, never used for fitting)
      - 'all'    → full history (used for regime HTF context slices)
    """
    import pandas as pd

    tf_upper = timeframe.upper()
    cache_key = (symbol, tf_upper)
    if cache_key in _PARQUET_CACHE:
        df = _PARQUET_CACHE[cache_key].copy()
    else:
        parquet_path = os.path.join(_PROCESSED_DIR, f"{symbol}_{tf_upper}.parquet")
        if not os.path.exists(parquet_path):
            logger.warning(
                "Missing parquet %s — run pipeline/step0_resample.py first", parquet_path
            )
            return None
        try:
            raw = pd.read_parquet(parquet_path)
            raw.index = pd.to_datetime(raw.index, utc=True, errors="coerce")
            keep = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
            raw = raw[keep].dropna(subset=["close"])
            raw = raw[raw.index.notna()].sort_index()
            _PARQUET_CACHE[cache_key] = raw
            logger.debug("Cached parquet %s/%s (%d bars)", symbol, tf_upper, len(raw))
        except Exception as exc:
            logger.error("Failed to load parquet %s/%s: %s", symbol, tf_upper, exc)
            return None
        df = _PARQUET_CACHE[cache_key].copy()

    # Apply temporal split boundaries
    if split != "all":
        bounds = _load_split_boundaries()
        if bounds:
            if split == "train":
                df = df[df.index <= bounds["train_end"]]
            elif split == "val":
                df = df[(df.index > bounds["train_end"]) & (df.index <= bounds["val_end"])]
            elif split == "test":
                df = df[df.index > bounds["val_end"]]

    if len(df) == 0:
        logger.warning("_load_ohlcv: %s/%s split=%s yielded 0 rows", symbol, tf_upper, split)
        return None

    logger.info("Loaded %s/%s split=%s: %d bars (%s → %s)",
                symbol, tf_upper, split, len(df),
                df.index.min().date(), df.index.max().date())
    return df


def _backup_weights(path: str) -> None:
    os.makedirs(BACKUP_DIR, exist_ok=True)
    if os.path.exists(path):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        name = os.path.basename(path)
        dest = os.path.join(BACKUP_DIR, f"{name}_{ts}")
        if os.path.isdir(path):
            shutil.copytree(path, dest)
        else:
            shutil.copy2(path, dest)
        logger.info("Backed up %s → %s", path, dest)
        # Prune old backups
        all_bk = sorted([
            f for f in os.listdir(BACKUP_DIR)
            if f.startswith(name)
        ])
        while len(all_bk) > MAX_BACKUPS:
            old = os.path.join(BACKUP_DIR, all_bk.pop(0))
            if os.path.isdir(old):
                shutil.rmtree(old)
            else:
                os.remove(old)


def _retrain_gru_multi(model, symbols: list) -> dict:
    """
    Kaggle-optimised GRU training: build segments for ALL symbols across each TF,
    concatenate per TF, and train in one pass per TF via model.train_multi().
    Avoids 44 sequential save/reload cycles. Caps each TF group at 2M sequences
    (~10GB) to stay within 30GB RAM safely.
    """
    import gc
    from models.gru_lstm_predictor import GRULSTMPredictor

    logger.info("GRU multi-symbol training (Kaggle mode): %d symbols × %s", len(symbols), GRU_TIMEFRAMES)
    _backup_weights(str(_ENV["weights"] / "gru_lstm"))

    # Delete stale weights if feature contract changed since last save.
    _stale_pt   = str(_ENV["weights"] / "gru_lstm" / "model.pt")
    _weight_dir = str(_ENV["weights"] / "gru_lstm")
    try:
        from models.weights_manifest import WeightsManifest
        from services.feature_engine import SEQUENCE_FEATURES, REGIME_4H_FEATURES, REGIME_1H_FEATURES, QUALITY_FEATURES
        from models.gru_lstm_predictor import N_FEATURES
        compat = WeightsManifest(_weight_dir).check(
            gru_features=list(SEQUENCE_FEATURES),
            regime_4h_features=list(REGIME_4H_FEATURES),
            regime_1h_features=list(REGIME_1H_FEATURES),
            quality_features=list(QUALITY_FEATURES),
        )
        if not compat:
            logger.info("GRU weights stale (%s) — deleting for full retrain", compat.reason)
            WeightsManifest.delete_stale([_stale_pt], compat.reason)
        else:
            # No manifest or contract matches — verify the saved weights actually have
            # the right input shape before declaring "incremental retrain is safe".
            if os.path.exists(_stale_pt):
                try:
                    import torch as _torch
                    _sd = _torch.load(_stale_pt, map_location="cpu", weights_only=True)
                    _saved_input = _sd.get("gru.weight_ih_l0", _sd.get("module.gru.weight_ih_l0"))
                    if _saved_input is not None and _saved_input.shape[1] != N_FEATURES:
                        logger.info(
                            "GRU weights have input_size=%d but current N_FEATURES=%d — "
                            "deleting stale weights for full retrain",
                            _saved_input.shape[1], N_FEATURES,
                        )
                        os.remove(_stale_pt)
                    else:
                        logger.info("GRU feature contract unchanged (input_size=%d) — incremental retrain", N_FEATURES)
                except Exception as _se:
                    logger.warning("GRU shape check failed (%s) — deleting to be safe", _se)
                    os.remove(_stale_pt)
    except Exception as _me:
        logger.warning("Manifest check failed (%s) — deleting weights to be safe", _me)
        if os.path.exists(_stale_pt):
            os.remove(_stale_pt)

    segments = []
    samples_total = 0

    for sym in symbols:
        all_htf = {tf: _load_ohlcv(sym, tf, split="all") for tf in ("5M", "1H", "4H", "1D")}

        # Rule-based regime labels (consistent with what the regime classifier trains on)
        # HTF: 3-class bias labels; LTF: 4-class behaviour labels
        _sym_regime_htf: "pd.Series | None" = None
        _sym_regime_ltf: "pd.Series | None" = None
        try:
            from models.regime_classifier import RegimeClassifier as _RC_sym
            _df4h_sym = _load_ohlcv(sym, "4H", split="train")
            if _df4h_sym is not None and len(_df4h_sym) > 50:
                _sym_regime_htf = _RC_sym.create_rule_labels(
                    _df4h_sym, timeframe="4H", mode="htf_bias")
            _df1h_sym = _load_ohlcv(sym, "1H", split="train")
            if _df1h_sym is not None and len(_df1h_sym) > 50:
                _sym_regime_ltf = _RC_sym.create_rule_labels(
                    _df1h_sym, timeframe="1H", mode="ltf_behaviour")
        except Exception as _e:
            logger.debug("GRU multi: rule regime labels for %s failed (%s)", sym, _e)

        for tf in GRU_TIMEFRAMES:
            df = _load_ohlcv(sym, tf, split="train")
            if df is None or len(df) <= 200:
                continue

            labels = model.create_labels(df)
            valid_idx = labels.dropna(subset=["direction_up", "move_magnitude", "volatility_target"]).index
            df_train = df.loc[valid_idx]
            labels_train = labels.loc[valid_idx]

            if len(df_train) < 500:
                del df, labels, df_train, labels_train
                continue

            end_ts = df_train.index[-1]
            htf_train = {}
            for htf_tf, htf_df in all_htf.items():
                if htf_df is not None and len(htf_df) >= 14:
                    trimmed = htf_df[htf_df.index <= end_ts]
                    if len(trimmed) >= 14:
                        htf_train[htf_tf] = trimmed
            if tf == "15M":
                htf_train["15M"] = df_train
            elif tf not in htf_train:
                htf_train[tf] = df_train

            # Forward-fill both 4H and 1H regime labels to training TF index
            def _ffill_regime(s, end_ts, target_idx):
                if s is None:
                    return None
                return s[s.index <= end_ts].reindex(target_idx, method="ffill").fillna(2).astype(int)

            segments.append({
                "df": df_train,
                "labels": labels_train,
                "df_htf": htf_train,
                "symbol": sym,
                "timeframe": tf,
                # Canonical new names
                "regime_htf_series": _ffill_regime(_sym_regime_htf, end_ts, df_train.index),
                "regime_ltf_series": _ffill_regime(_sym_regime_ltf, end_ts, df_train.index),
                # Legacy compat aliases (some callers may check these keys)
                "regime_series":    _ffill_regime(_sym_regime_htf, end_ts, df_train.index),
                "regime_4h_series": _ffill_regime(_sym_regime_htf, end_ts, df_train.index),
                "regime_1h_series": _ffill_regime(_sym_regime_ltf, end_ts, df_train.index),
            })
            samples_total += len(df_train)

            del df, labels
            gc.collect()

    if not segments:
        return {"error": "No valid segments for multi-symbol GRU training"}

    logger.info("train_multi: %d segments, ~%d total bars", len(segments), samples_total)
    history = model.train_multi(
        segments,
        epochs=GRU_EPOCHS,
        batch_size=GRU_BATCH_SIZE,
    )

    if history.get("error"):
        return {"error": history["error"]}

    log_retrain("gru_lstm", {
        "status": "complete",
        "mode": "multi_symbol",
        "segments": len(segments),
        "samples": samples_total,
        "groups_trained": history.get("groups_trained", 0),
        "val_loss_points": len(history.get("val_loss", [])),
    })
    return {"trained": True, "segments": len(segments), "samples": samples_total}


def retrain_gru(dry_run: bool = False) -> dict:
    """GRU-LSTM: train on all GRU_TIMEFRAMES (5M, 15M, 1H, 4H) across all symbols.

    On Kaggle (large RAM): uses train_multi() to concatenate all symbols per TF into
    one combined dataset — keeps GPU fed continuously, avoids 44 save/reload cycles.
    Locally: falls back to per-symbol loop to keep RAM usage low.
    """
    import time as _time
    _t0_gru = _time.perf_counter()
    logger.info("=== GRU-LSTM retrain (timeframes: %s) ===", GRU_TIMEFRAMES)
    from models.gru_lstm_predictor import GRULSTMPredictor

    model = GRULSTMPredictor()
    symbols = _get_symbols("RETRAIN_SYMBOLS_GRU", MAJOR_SYMBOLS)
    trained = 0
    samples_total = 0

    if not dry_run:
        _t_macro = _time.perf_counter()
        _update_macro_correlations(symbols)
        logger.info("GRU phase macro_correlations: %.1fs", _time.perf_counter() - _t_macro)

    # On Kaggle use combined multi-symbol training (30GB RAM available)
    if _ENV["on_kaggle"] and not dry_run:
        return _retrain_gru_multi(model, symbols)

    backup_done = False
    for sym in symbols:
        # HTF context loaded with split="all" — regime/structure features need full
        # history for context, actual training data is sliced separately below
        all_htf = {tf: _load_ohlcv(sym, tf, split="all") for tf in ("5M", "1H", "4H", "1D")}

        for tf in GRU_TIMEFRAMES:
            df = _load_ohlcv(sym, tf, split="train")
            if df is None or len(df) <= 200:
                logger.warning("GRU: skipping %s/%s (insufficient data)", sym, tf)
                continue

            labels = model.create_labels(df)
            valid_idx = labels.dropna(subset=["direction_up", "move_magnitude", "volatility_target"]).index
            df_train = df.loc[valid_idx]
            labels_train = labels.loc[valid_idx]

            if len(df_train) < 500:
                logger.warning("GRU: %s/%s has only %d samples — skipping", sym, tf, len(df_train))
                del df, labels, df_train, labels_train
                continue

            # Build HTF dict trimmed to end of training data (no future leakage)
            end_ts = df_train.index[-1]
            htf_train = {}
            for htf_tf, htf_df in all_htf.items():
                if htf_df is not None and len(htf_df) >= 14:
                    trimmed = htf_df[htf_df.index <= end_ts]
                    if len(trimmed) >= 14:
                        htf_train[htf_tf] = trimmed
            # Include self as "15M" slot when training on 15M
            if tf == "15M":
                htf_train["15M"] = df_train
            elif tf not in htf_train:
                htf_train[tf] = df_train  # self-reference for the execution TF

            if dry_run:
                logger.info("DRY RUN: GRU-LSTM %s/%s — %d samples (htf TFs: %s)",
                            sym, tf, len(df_train), list(htf_train.keys()))
                samples_total += len(df_train)
                del df, labels, df_train, labels_train
                continue

            if not backup_done:
                _backup_weights(os.path.join(WEIGHTS_DIR, "gru_lstm"))
                # Delete stale weights — SEQUENCE_FEATURES count changed (53→83).
                # Old model.pt would cause a shape mismatch on first train() call.
                _stale_pt = os.path.join(WEIGHTS_DIR, "gru_lstm", "model.pt")
                if os.path.exists(_stale_pt):
                    os.remove(_stale_pt)
                    logger.info("Deleted stale GRU weights (%s) — full retrain from scratch", _stale_pt)
                backup_done = True

            _t_sym = _time.perf_counter()
            history = model.train(
                df_train,
                labels_train,
                epochs=GRU_EPOCHS,
                batch_size=GRU_BATCH_SIZE,
                symbol=sym,
                df_htf=htf_train,
            )
            logger.info("GRU phase train %s/%s: %.1fs", sym, tf, _time.perf_counter() - _t_sym)
            if history.get("error"):
                logger.error("GRU-LSTM train failed on %s/%s: %s", sym, tf, history["error"])
                log_retrain("gru_lstm", {
                    "error": history["error"],
                    "symbol": sym,
                    "timeframe": tf,
                    "status": "symbol_tf_failed",
                })
                del df, labels, df_train, labels_train
                import gc; gc.collect()
                continue
            if not _gru_artifact_exists():
                err = "GRU weights were not created"
                logger.error("GRU-LSTM train failed on %s/%s: %s", sym, tf, err)
                log_retrain("gru_lstm", {
                    "error": err,
                    "symbol": sym,
                    "timeframe": tf,
                    "status": "symbol_tf_failed",
                })
                del df, labels, df_train, labels_train
                import gc; gc.collect()
                continue
            logger.info("GRU-LSTM trained on %s/%s. Val loss points: %d",
                        sym, tf, len(history.get("val_loss", [])))
            log_retrain("gru_lstm", {
                "symbol": sym, "timeframe": tf,
                "status": "symbol_tf_complete",
                "samples": len(df_train),
                "val_loss_points": len(history.get("val_loss", [])),
            })
            trained += 1
            samples_total += len(df_train)

            del df, labels, df_train, labels_train
            import gc; gc.collect()

    logger.info("GRU retrain total: %.1fs (%d combos, %d samples)",
                _time.perf_counter() - _t0_gru, trained, samples_total)
    if dry_run:
        return {"dry_run": True, "combos": trained, "samples": samples_total}
    if trained == 0:
        return {"error": "No GRU symbol/timeframe produced trained weights"}
    return {"trained": True, "combos": trained, "samples": samples_total}


def _build_regime_dataset(symbols: list, source_tf: str, label_tf: str,
                           group_gmms: dict, dry_run: bool = False,
                           mode: str = "ltf_behaviour") -> tuple:
    """
    Build (X_all, y_all, sw_all) for one regime classifier.

    source_tf: the TF we build feature matrices on (e.g. "4H" or "1H").
    label_tf:  source TF for rule labels.
    mode: "htf_bias" → 3-class labels for HTF classifier.
          "ltf_behaviour" → 4-class labels for LTF classifier.
    Returns (X, y, sample_weight, n_samples).
    sample_weight: float32 array (N,) — per-bar confidence from create_rule_labels.
    """
    from models.regime_classifier import RegimeClassifier as _RC
    import gc as _gc
    import numpy as _np

    _tmp_model = _RC(mode=mode)
    X_parts:  list = []
    y_parts:  list = []
    sw_parts: list = []
    samples = 0
    _default_label = 2 if mode == "htf_bias" else 1  # BIAS_NEUTRAL or RANGING

    # Cache label_tf dfs per symbol
    _label_cache: dict = {}
    for sym in symbols:
        df_l = _load_ohlcv(sym, label_tf, split="train")
        if df_l is not None and len(df_l) > 50:
            _label_cache[sym] = df_l

    for sym in symbols:
        all_htf = {tf: _load_ohlcv(sym, tf, split="all")
                   for tf in ("5M", "15M", "1H", "4H", "1D")}
        grp = _group_for_symbol(sym)
        gmm_grp, scaler_grp, cl_grp = group_gmms.get(grp, (None, None, None))
        df_label_sym = _label_cache.get(sym)

        df = _load_ohlcv(sym, source_tf, split="train")
        if df is None or len(df) <= 200:
            logger.warning("Regime[%s mode=%s]: skipping %s (insufficient data)", source_tf, mode, sym)
            continue

        if dry_run:
            logger.info("DRY RUN: Regime[%s mode=%s] %s — %d bars", source_tf, mode, sym, len(df))
            samples += len(df)
            del df
            continue

        end_ts = df.index[-1]
        htf_train: dict = {}
        for htf_tf, htf_df in all_htf.items():
            if htf_df is not None and len(htf_df) >= 14:
                trimmed = htf_df[htf_df.index <= end_ts]
                if len(trimmed) >= 14:
                    htf_train[htf_tf] = trimmed
        if source_tf not in htf_train:
            htf_train[source_tf] = df

        try:
            X_sym = _RC._build_feature_matrix(df, htf_train, sym)

            # Mode-aware rule-based labels with confidence.
            # HTF: label on source_tf df with htf_bias mode.
            # LTF: label on source_tf df with ltf_behaviour mode.
            if mode == "htf_bias":
                labels, conf = _RC.create_rule_labels(df, timeframe=source_tf, mode="htf_bias",
                                                      return_confidence=True)
            elif mode == "ltf_behaviour":
                labels, conf = _RC.create_rule_labels(df, timeframe=source_tf, mode="ltf_behaviour",
                                                      return_confidence=True)
            else:
                # Fallback GMM for any other mode — uniform confidence
                if gmm_grp is not None:
                    _lbl_nbar = 50 if label_tf == "4H" else 24
                    labels = _tmp_model.create_labels_with_gmm(df, gmm_grp, scaler_grp, cl_grp,
                                                               n_bar=_lbl_nbar)
                else:
                    labels = _tmp_model.create_labels(df)
                conf = _np.ones(len(df), dtype=_np.float32)
                conf = type(labels)(conf, index=df.index)  # align index

            n = len(df)
            step = max(1, (n - 50) // 100_000)
            idx  = _np.arange(50, n, step)
            X_parts.append(X_sym[idx])
            y_parts.append(labels.iloc[idx].values.astype(_np.int64))
            sw_parts.append(conf.iloc[idx].values.astype(_np.float32))
            samples += len(idx)
            logger.info("Regime[%s mode=%s]: collected %s — %d samples (group=%s)", source_tf, mode, sym, len(idx), grp)
        except Exception as exc:
            logger.error("Regime[%s mode=%s]: feature build failed %s: %s", source_tf, mode, sym, exc)
        finally:
            del df
            _gc.collect()

    if dry_run:
        return None, None, None, samples

    if not X_parts:
        return None, None, None, 0

    import numpy as _np2
    X_all  = _np2.concatenate(X_parts,  axis=0)
    y_all  = _np2.concatenate(y_parts,  axis=0)
    sw_all = _np2.concatenate(sw_parts, axis=0)
    del X_parts, y_parts, sw_parts
    _gc.collect()
    return X_all, y_all, sw_all, samples


def _regime_diagnostics(model, group_gmms: dict, symbols: list, source_tf: str) -> None:
    """Log persistence and return-separation using the same rule labels as training."""
    try:
        from models.regime_classifier import RegimeClassifier as _RC_diag
        _diag_sym = symbols[-1] if symbols else None
        _diag_df  = _load_ohlcv(_diag_sym, source_tf, split="train") if _diag_sym else None
        if _diag_df is None or len(_diag_df) < 200:
            return
        _mode = "htf_bias" if str(source_tf).upper() == "4H" else "ltf_behaviour"
        _classes = (
            ["BIAS_UP", "BIAS_DOWN", "BIAS_NEUTRAL"]
            if _mode == "htf_bias"
            else ["TRENDING", "RANGING", "CONSOLIDATING", "VOLATILE"]
        )
        _min_conf = float(os.getenv("REGIME_MIN_LABEL_CONFIDENCE", "0.4"))
        _lbl, _conf = _RC_diag.create_rule_labels(
            _diag_df, timeframe=source_tf, mode=_mode, return_confidence=True,
        )
        _run_id = (_lbl != _lbl.shift()).cumsum()
        _runs = _lbl.groupby(_run_id).agg(["count", "first"])
        _persistence = _runs.groupby("first")["count"].mean()
        _persistence_named = {
            _classes[int(k)]: float(v)
            for k, v in _persistence.to_dict().items()
            if int(k) < len(_classes)
        }
        logger.info("Regime[%s mode=%s] persistence (avg bars/run) on %s %s:\n%s",
                    source_tf, _mode, _diag_sym, source_tf, _persistence_named)
        _nr = _diag_df["close"].pct_change().shift(-1)
        _sep_all = _nr.groupby(_lbl).agg(["count", "mean", "std"])
        _clean = _conf >= _min_conf
        _sep_clean = _nr[_clean].groupby(_lbl[_clean]).agg(["count", "mean", "std"])

        def _named_sep(_df):
            out = {}
            for k, row in _df.iterrows():
                k_i = int(k)
                if k_i >= len(_classes):
                    continue
                _std = float(row.get("std", 0.0) or 0.0)
                out[_classes[k_i]] = {
                    "n": int(row.get("count", 0)),
                    "mean": float(row.get("mean", 0.0) or 0.0),
                    "mean_over_std": float((row.get("mean", 0.0) or 0.0) / (_std + 1e-12)),
                }
            return out

        logger.info("Regime[%s mode=%s] return separation on %s %s (all labels):\n%s",
                    source_tf, _mode, _diag_sym, source_tf, _named_sep(_sep_all))
        logger.info("Regime[%s mode=%s] return separation on %s %s (clean labels conf>=%.2f):\n%s",
                    source_tf, _mode, _diag_sym, source_tf, _min_conf, _named_sep(_sep_clean))
    except Exception as _e:
        logger.warning("Regime[%s] diagnostics failed: %s", source_tf, _e)


def retrain_regime(dry_run: bool = False) -> dict:
    """
    Train hierarchical regime cascade:
      1. HTF classifier (regime_htf.pkl) — 3-class bias (BIAS_UP/DOWN/NEUTRAL).
         mode="htf_bias", trained on 4H bars, labels by drift direction.
         GPU-parallel across both T4s via DataParallel.
      2. LTF classifier (regime_ltf.pkl) — 4-class behaviour (TRENDING/RANGING/CONSOLIDATING/VOLATILE).
         mode="ltf_behaviour", trained on 1H bars, direction-agnostic labels.
         Same DataParallel setup — both GPUs stay hot across both trains.

    Both classifiers share the same REGIME_FEATURES contract (same feature matrix),
    so the same _build_feature_matrix() call produces valid input for both.
    """
    import time as _time
    _t0_regime = _time.perf_counter()
    logger.info("=== RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===")
    from models.regime_classifier import RegimeClassifier as _RC
    import gc as _gc

    symbols = _get_symbols("RETRAIN_SYMBOLS_REGIME", MAJOR_SYMBOLS)

    if not dry_run:
        _t_macro = _time.perf_counter()
        _update_macro_correlations(symbols)
        logger.info("Regime phase macro_correlations: %.1fs", _time.perf_counter() - _t_macro)

    # Fit per-group GMMs on 4H data for HTF bias — separate GMM for LTF on 1H data.
    logger.info("Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...")
    group_dfs_4h: dict = {"dollar": [], "cross": [], "gold": []}
    group_dfs_1h: dict = {"dollar": [], "cross": [], "gold": []}
    for sym in symbols:
        df_4h = _load_ohlcv(sym, "4H", split="train")
        if df_4h is not None and len(df_4h) > 200:
            group_dfs_4h[_group_for_symbol(sym)].append(df_4h)
        df_1h = _load_ohlcv(sym, "1H", split="train")
        if df_1h is not None and len(df_1h) > 200:
            group_dfs_1h[_group_for_symbol(sym)].append(df_1h)

    group_gmms_htf: dict = {}
    _t_gmm_htf = _time.perf_counter()
    for grp, dfs in group_dfs_4h.items():
        if dfs:
            _t_grp = _time.perf_counter()
            gmm, scaler, cluster_labels = _RC.fit_global_gmm(dfs, timeframe="4H", mode="htf_bias")
            group_gmms_htf[grp] = (gmm, scaler, cluster_labels)
            logger.info("Regime HTF GMM '%s' fitted on %d 4H dfs (3-class bias) in %.1fs",
                        grp, len(dfs), _time.perf_counter() - _t_grp)
        else:
            logger.warning("Regime: no 4H data for group '%s'", grp)
    logger.info("Regime phase GMM HTF total: %.1fs", _time.perf_counter() - _t_gmm_htf)

    group_gmms_ltf: dict = {}
    _t_gmm_ltf = _time.perf_counter()
    for grp, dfs in group_dfs_1h.items():
        if dfs:
            _t_grp = _time.perf_counter()
            gmm, scaler, cluster_labels = _RC.fit_global_gmm(dfs, timeframe="1H", mode="ltf_behaviour")
            group_gmms_ltf[grp] = (gmm, scaler, cluster_labels)
            logger.info("Regime LTF GMM '%s' fitted on %d 1H dfs (4-class behaviour) in %.1fs",
                        grp, len(dfs), _time.perf_counter() - _t_grp)
        else:
            logger.warning("Regime LTF: no 1H data for group '%s'", grp)
    logger.info("Regime phase GMM LTF total: %.1fs", _time.perf_counter() - _t_gmm_ltf)

    # For backward compat, pass htf GMMs to _build_regime_dataset (which uses group_gmms)
    group_gmms = group_gmms_htf
    del group_dfs_4h, group_dfs_1h
    _gc.collect()

    results: dict = {}
    total_samples = 0

    # ── HTF bias classifier (3-class) ────────────────────────────────────────
    logger.info("Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...")
    _t_htf_ds = _time.perf_counter()
    X_4h, y_4h, sw_4h, n_4h = _build_regime_dataset(
        symbols, source_tf="4H", label_tf="4H",
        group_gmms=group_gmms_htf, dry_run=dry_run, mode="htf_bias",
    )
    logger.info("Regime phase HTF dataset build: %.1fs (%d samples)", _time.perf_counter() - _t_htf_ds, n_4h)
    total_samples += n_4h
    if not dry_run and X_4h is not None:
        _backup_weights(os.path.join(WEIGHTS_DIR, "regime_htf.pkl"))
        model_htf = _RC(timeframe="4H", mode="htf_bias")
        _t_htf_train = _time.perf_counter()
        res_4h = model_htf.train_on_arrays(X_4h, y_4h, sample_weight=sw_4h)
        logger.info("Regime phase HTF train: %.1fs", _time.perf_counter() - _t_htf_train)
        del X_4h, y_4h, sw_4h; _gc.collect()
        if res_4h.get("error"):
            logger.error("Regime HTF training failed: %s", res_4h["error"])
            results["HTF"] = res_4h
        else:
            logger.info("Regime HTF complete: acc=%.3f, n=%d per_class=%s",
                        res_4h.get("accuracy", 0), n_4h, res_4h.get("per_class_accuracy", {}))
            _regime_diagnostics(model_htf, group_gmms_htf, symbols, "4H")
            results["HTF"] = res_4h
            log_retrain("regime_classifier_htf", {**res_4h, "status": "complete"})
        if not _htf_regime_artifact_exists():
            logger.error("Regime HTF weights were not created at regime_htf.pkl")

    # ── LTF behaviour classifier (4-class) ───────────────────────────────────
    logger.info("Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...")
    _t_ltf_ds = _time.perf_counter()
    X_1h, y_1h, sw_1h, n_1h = _build_regime_dataset(
        symbols, source_tf="1H", label_tf="1H",  # labels sourced from 1H directly (ltf_behaviour mode)
        group_gmms=group_gmms_ltf, dry_run=dry_run, mode="ltf_behaviour",
    )
    logger.info("Regime phase LTF dataset build: %.1fs (%d samples)", _time.perf_counter() - _t_ltf_ds, n_1h)
    total_samples += n_1h
    if not dry_run and X_1h is not None:
        _backup_weights(os.path.join(WEIGHTS_DIR, "regime_ltf.pkl"))
        model_ltf = _RC(timeframe="1H", mode="ltf_behaviour")
        _t_ltf_train = _time.perf_counter()
        res_1h = model_ltf.train_on_arrays(X_1h, y_1h, sample_weight=sw_1h)
        logger.info("Regime phase LTF train: %.1fs", _time.perf_counter() - _t_ltf_train)
        del X_1h, y_1h, sw_1h; _gc.collect()
        if res_1h.get("error"):
            logger.error("Regime LTF training failed: %s", res_1h["error"])
            results["LTF"] = res_1h
        else:
            logger.info("Regime LTF complete: acc=%.3f, n=%d per_class=%s",
                        res_1h.get("accuracy", 0), n_1h, res_1h.get("per_class_accuracy", {}))
            _regime_diagnostics(model_ltf, group_gmms_ltf, symbols, "1H")
            results["LTF"] = res_1h
            log_retrain("regime_classifier_ltf", {**res_1h, "status": "complete"})
        if not _ltf_regime_artifact_exists():
            logger.error("Regime LTF weights were not created at regime_ltf.pkl")

    logger.info("Regime retrain total: %.1fs (%d samples)", _time.perf_counter() - _t0_regime, total_samples)
    if dry_run:
        return {"dry_run": True, "samples": total_samples}  # n_4h + n_1h already accumulated

    any_error = any(r.get("error") for r in results.values())
    if any_error:
        return {"error": "One or more regime TF trainings failed", "details": results}

    return {
        "trained": True,
        "samples": total_samples,
        "results": {
            tf: {
                "accuracy": r.get("accuracy", 0),
                "per_class_accuracy": r.get("per_class_accuracy", {}),
            }
            for tf, r in results.items()
        },
    }


def retrain_quality(dry_run: bool = False) -> dict:
    """XGBoost quality scorer: load journal, TP1/SL labels, retrain."""
    import time as _time
    _t0 = _time.perf_counter()
    logger.info("=== QualityScorer retrain ===")
    from models.quality_scorer import QualityScorer

    if not os.path.exists(JOURNAL_PATH):
        return {"error": f"Journal not found: {JOURNAL_PATH}"}

    model = QualityScorer()
    _t_labels = _time.perf_counter()
    labeled_df = model.create_labels(JOURNAL_PATH)
    logger.info("Quality phase label creation: %.1fs (%d trades)",
                _time.perf_counter() - _t_labels,
                len(labeled_df) if labeled_df is not None else 0)
    if labeled_df is None or len(labeled_df) < 20:
        return {"error": f"Only {len(labeled_df) if labeled_df is not None else 0} labeled trades — need ≥20"}

    if dry_run:
        logger.info("DRY RUN: would train QualityScorer on %d journal trades", len(labeled_df))
        return {"dry_run": True, "samples": len(labeled_df)}

    _backup_weights(os.path.join(WEIGHTS_DIR, "quality_scorer.pkl"))
    _t_train = _time.perf_counter()
    result = model.train(JOURNAL_PATH)
    logger.info("Quality phase train: %.1fs | total: %.1fs",
                _time.perf_counter() - _t_train, _time.perf_counter() - _t0)
    if result.get("error"):
        return result
    if not _quality_artifact_exists():
        return {"error": "QualityScorer weights were not created"}
    return result


def retrain_rl(dry_run: bool = False) -> dict:
    """PPO RL: retrain from journal episodes."""
    import time as _time
    _t0 = _time.perf_counter()
    logger.info("=== RLAgent (PPO) retrain ===")
    from models.rl_agent import RLAgent

    if not os.path.exists(JOURNAL_PATH):
        return {"error": f"Journal not found: {JOURNAL_PATH}"}

    if dry_run:
        episodes = RLAgent()._load_journal_episodes(JOURNAL_PATH)
        logger.info("DRY RUN: %d RL episodes in journal", len(episodes))
        return {"dry_run": True, "episodes": len(episodes)}

    _backup_weights(os.path.join(WEIGHTS_DIR, "rl_ppo"))
    agent = RLAgent()
    _t_ep = _time.perf_counter()
    episodes = agent._load_journal_episodes(JOURNAL_PATH)
    logger.info("RL phase episode loading: %.1fs (%d episodes)", _time.perf_counter() - _t_ep, len(episodes))
    _t_train = _time.perf_counter()
    result = agent.retrain_from_journal(JOURNAL_PATH, n_epochs=10)
    logger.info("RL phase PPO train: %.1fs | total: %.1fs",
                _time.perf_counter() - _t_train, _time.perf_counter() - _t0)
    if result.get("error"):
        return result
    if not _rl_artifact_exists():
        return {"error": "RL weights were not created"}
    return result


def retrain_sentiment(dry_run: bool = False) -> dict:
    """FinBERT is pre-trained — skip with log message."""
    logger.info("=== SentimentModel: FinBERT pre-trained — skipping retrain ===")
    return {"skipped": True, "reason": "FinBERT is pre-trained via HuggingFace"}


def _index_embeddings_post_train(symbols: list[str], dry_run: bool = False) -> None:
    """
    Build VectorStore indices from trained weights after GRU + Regime training.

    Three indices populated:
      trade_patterns    (74-dim) — per-bar SEQUENCE_FEATURES snapshot, all training bars
      market_structures (34-dim) — REGIME_4H_FEATURES subset, all training bars
      regime_embeddings (64-dim) — GRU shared-layer encoding, sampled every 4 bars

    Runs after training so it never slows down the training loop itself.
    Saves to weights/vector_store/ for use by live trading and backtest.
    """
    if dry_run:
        logger.info("VectorStore: skipping indexing in dry-run mode")
        return

    try:
        import gc
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from models.vector_store import VectorStore
        from models.gru_lstm_predictor import GRULSTMPredictor
        from models.regime_classifier import RegimeClassifier as _RC
        from services.feature_engine import FeatureEngine, SEQUENCE_FEATURES, REGIME_FEATURES, REGIME_4H_FEATURES

        import time as _time
        _t0_vs = _time.perf_counter()
        logger.info("=== VectorStore: building similarity indices (parallel feature build) ===")
        store = VectorStore()
        gru_model = GRULSTMPredictor()
        fe = FeatureEngine()

        MAX_BARS_PER_SYMBOL = 50_000
        _col_idx = [REGIME_FEATURES.index(f) for f in REGIME_4H_FEATURES if f in REGIME_FEATURES]
        _n_workers = int(os.getenv("RETRAIN_CPU_WORKERS", "4"))

        def _build_sym_vectors(sym: str):
            """CPU-only: load data + build all three feature arrays for one symbol.
            Returns (sym, tp_vecs, tp_metas, ms_vecs, ms_metas, emb_seqs, emb_metas_idx, df_index)
            or raises on failure."""
            df = _load_ohlcv(sym, "15M", split="train")
            if df is None or len(df) < 200:
                return None

            result = {"sym": sym, "df_index": df.index}

            # trade_patterns
            try:
                feat_df = fe._build_sequence_df(df, None, symbol=sym)
                sq = feat_df[SEQUENCE_FEATURES].to_numpy(dtype="float32", copy=False)
                sq = sq[~np.isnan(sq).any(axis=1)]
                n = min(len(sq), MAX_BARS_PER_SYMBOL)
                step = max(1, len(sq) // n)
                vecs = sq[::step][:n]
                metas = [{"symbol": sym, "timeframe": "15M",
                          "ts": str(df.index[min(i * step, len(df) - 1)])}
                         for i in range(len(vecs))]
                result["tp"] = (vecs, metas)
                del feat_df, sq
            except Exception as exc:
                logger.warning("VectorStore trade_patterns failed for %s: %s", sym, exc)

            # market_structures
            try:
                all_htf = {tf: _load_ohlcv(sym, tf, split="all")
                           for tf in ("5M", "15M", "1H", "4H", "1D")}
                X_all = _RC._build_feature_matrix(df, all_htf, sym)
                X_htf = X_all[:, _col_idx]
                step = max(1, len(df) // MAX_BARS_PER_SYMBOL)
                idx = np.arange(50, len(df), step)
                idx = idx[idx < len(X_htf)]
                rvecs = X_htf[idx].astype("float32")
                rmetas = [{"symbol": sym, "timeframe": "15M", "ts": str(df.index[i])} for i in idx]
                result["ms"] = (rvecs, rmetas)
                del all_htf, X_all, X_htf
            except Exception as exc:
                logger.warning("VectorStore market_structures failed for %s: %s", sym, exc)

            # regime_embeddings: build sequences (GPU call happens in main thread)
            if gru_model.is_trained:
                try:
                    feat_df2 = fe._build_sequence_df(df, None, symbol=sym)
                    sq2 = feat_df2[SEQUENCE_FEATURES].to_numpy(dtype="float32", copy=False)
                    sq2 = sq2[~np.isnan(sq2).any(axis=1)]
                    n_seq = len(sq2) - 30
                    if n_seq > 0:
                        step4 = max(1, n_seq // (MAX_BARS_PER_SYMBOL // 4))
                        indices = list(range(0, n_seq, step4))
                        seqs = np.stack([sq2[i:i + 30] for i in indices], axis=0)
                        result["emb_seqs"] = seqs
                        result["emb_idx"] = indices
                    del feat_df2, sq2
                except Exception as exc:
                    logger.warning("VectorStore regime_embeddings prep failed for %s: %s", sym, exc)

            del df
            return result

        # Phase 1: parallel CPU feature build across all symbols
        _t_p1 = _time.perf_counter()
        sym_results = {}
        with ThreadPoolExecutor(max_workers=_n_workers) as pool:
            futures = {pool.submit(_build_sym_vectors, sym): sym for sym in symbols}
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    r = fut.result()
                    if r is not None:
                        sym_results[sym] = r
                except Exception as exc:
                    logger.warning("VectorStore feature build failed for %s: %s", sym, exc)
        logger.info("VectorStore phase 1 (parallel feature build, %d workers): %.1fs for %d symbols",
                    _n_workers, _time.perf_counter() - _t_p1, len(sym_results))

        # Phase 2: serial GPU add_batch (FAISS GPU index is not thread-safe)
        _t_p2 = _time.perf_counter()
        for sym in symbols:
            r = sym_results.get(sym)
            if r is None:
                continue

            if "tp" in r:
                vecs, metas = r["tp"]
                store.add_batch("trade_patterns", vecs, metas)
                logger.info("VectorStore trade_patterns: +%d vectors for %s", len(vecs), sym)

            if "ms" in r:
                rvecs, rmetas = r["ms"]
                store.add_batch("market_structures", rvecs, rmetas)
                logger.info("VectorStore market_structures: +%d vectors (34-dim 4H) for %s", len(rvecs), sym)

            if "emb_seqs" in r and gru_model.is_trained:
                seqs = r["emb_seqs"]
                indices = r["emb_idx"]
                df_index = r["df_index"]
                embs = gru_model.get_embedding_batch(seqs)
                if embs is not None:
                    emb_metas = [
                        {"symbol": sym, "timeframe": "15M",
                         "ts": str(df_index[min(i + 30, len(df_index) - 1)])}
                        for i in indices
                    ]
                    store.add_batch("regime_embeddings", embs, emb_metas)
                    logger.info("VectorStore regime_embeddings: +%d vectors for %s", len(embs), sym)

            gc.collect()

        logger.info("VectorStore phase 2 (serial GPU add): %.1fs", _time.perf_counter() - _t_p2)
        store.save()
        logger.info("VectorStore saved: %s | total indexing: %.1fs",
                    store.sizes(), _time.perf_counter() - _t0_vs)

    except Exception as exc:
        logger.error("_index_embeddings_post_train failed (non-fatal): %s", exc)


def validate_only() -> dict:
    """Check that all model files exist and imports work."""
    results = {}
    try:
        from models.gru_lstm_predictor import GRULSTMPredictor
        gru = GRULSTMPredictor()
        results["gru_lstm"] = {"is_trained": gru.is_trained}
    except Exception as exc:
        results["gru_lstm"] = {"error": str(exc)}

    try:
        from models.regime_classifier import RegimeClassifier
        rc_htf = RegimeClassifier(timeframe="4H", mode="htf_bias")
        rc_ltf = RegimeClassifier(timeframe="1H", mode="ltf_behaviour")
        results["regime_htf"] = {"is_trained": rc_htf.is_trained}
        results["regime_ltf"] = {"is_trained": rc_ltf.is_trained}
        results["regime"] = {
            "is_trained": bool(rc_htf.is_trained and rc_ltf.is_trained),
            "components": {
                "htf": bool(rc_htf.is_trained),
                "ltf": bool(rc_ltf.is_trained),
            },
        }
    except Exception as exc:
        results["regime"] = {"error": str(exc)}

    try:
        from models.quality_scorer import QualityScorer
        qs = QualityScorer()
        results["quality"] = {"is_trained": qs.is_trained}
    except Exception as exc:
        results["quality"] = {"error": str(exc)}

    try:
        from models.sentiment_model import SentimentModel
        sm = SentimentModel()
        results["sentiment"] = {"bert_available": sm._bert_available}
    except Exception as exc:
        results["sentiment"] = {"error": str(exc)}

    try:
        from models.rl_agent import RLAgent
        rl = RLAgent()
        results["rl"] = {"is_trained": rl.is_trained}
    except Exception as exc:
        results["rl"] = {"error": str(exc)}

    return results


def log_retrain(model_name: str, result: dict) -> None:
    os.makedirs("logs", exist_ok=True)
    path = "logs/retrain_history.jsonl"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        **result,
    }
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Incremental model retraining")
    parser.add_argument("--model", choices=["gru", "regime", "quality", "rl", "sentiment", "all"],
                        default="all", help="Model to retrain")
    parser.add_argument("--dry-run", action="store_true", help="Validate without saving")
    args = parser.parse_args()

    import time as _time
    _t0_main = _time.perf_counter()

    dry = args.dry_run
    model = args.model
    any_failure = False

    _gru_trained    = False
    _regime_trained = False

    if model in ("all", "gru"):
        result = retrain_gru(dry)
        log_retrain("gru_lstm", result)
        if result.get("error"):
            any_failure = True
        else:
            _gru_trained = True

    if model in ("all", "regime"):
        result = retrain_regime(dry)
        log_retrain("regime_classifier", result)
        if result.get("error"):
            any_failure = True
        else:
            _regime_trained = True

    # Build VectorStore indices after GRU + Regime are trained.
    # Runs even if only one of the two succeeded — partial indexing is still useful.
    if _gru_trained or _regime_trained:
        _index_symbols = _get_symbols("RETRAIN_SYMBOLS_GRU", MAJOR_SYMBOLS)
        _index_embeddings_post_train(_index_symbols, dry_run=dry)

    if model in ("all", "quality"):
        result = retrain_quality(dry)
        log_retrain("quality_scorer", result)
        if result.get("error"):
            any_failure = True

    if model in ("all", "rl"):
        result = retrain_rl(dry)
        log_retrain("rl_agent", result)
        if result.get("error"):
            any_failure = True

    if model in ("all", "sentiment"):
        result = retrain_sentiment(dry)
        log_retrain("sentiment_model", result)

    if dry:
        logger.info("=== DRY RUN COMPLETE — validation results ===")
        results = validate_only()
        for k, v in results.items():
            logger.info("  %s: %s", k, v)

    logger.info("Retrain complete. Total wall-clock: %.1fs", _time.perf_counter() - _t0_main)
    if any_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
