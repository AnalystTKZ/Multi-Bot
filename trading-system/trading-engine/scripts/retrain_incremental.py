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
import hashlib
import json
import logging
import os
import shutil
import subprocess
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
SPLIT_SUMMARY_PATH = str(_ENV["ml_training"] / "datasets" / "split_summary.json")
MAX_BACKUPS = 5
MONTHS_OF_DATA = int(os.getenv("RETRAIN_MONTHS", "0"))  # 0 = use all available data
RETRAIN_DATA_SPLIT = os.getenv("RETRAIN_DATA_SPLIT", "train").strip().lower()
if RETRAIN_DATA_SPLIT not in {"train", "val", "test", "all"}:
    raise ValueError(
        f"Invalid RETRAIN_DATA_SPLIT={RETRAIN_DATA_SPLIT!r}; expected train, val, test, or all"
    )
_ALLOW_NONTRAIN_RETRAIN = os.getenv("ALLOW_NONTRAIN_RETRAIN", "0").strip().lower() in {
    "1", "true", "yes", "on",
}
if RETRAIN_DATA_SPLIT != "train" and not _ALLOW_NONTRAIN_RETRAIN:
    raise PermissionError(
        "RETRAIN_DATA_SPLIT=%s requested without ALLOW_NONTRAIN_RETRAIN=1; "
        "refusing to train on validation/test data to preserve blind evaluation."
        % RETRAIN_DATA_SPLIT
    )
logger.info("Retrain data split: %s", RETRAIN_DATA_SPLIT)
RETRAIN_ROLLING_FOLD = os.getenv("RETRAIN_ROLLING_FOLD", "latest").strip().lower()
logger.info("Retrain rolling fold selector: %s", RETRAIN_ROLLING_FOLD)
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
# GRU is a 15M execution model. Regime and higher-timeframe context are combined
# later by the decision engine, not learned as GRU labels/features.
GRU_TIMEFRAMES = ["15M"]
# Hierarchical regime cascade:
# HTF (4H): 3-class bias (BIAS_UP/DOWN/NEUTRAL) — trained with mode="htf_bias"
# LTF (1H): five behaviour scores (trend/range/chop/volatility/consolidation) — trained with mode="ltf_behaviour"
REGIME_HTF_TF = ["4H"]         # HTF bias classifier source timeframe (was REGIME_TF_4H)
REGIME_LTF_TF = ["1H"]         # LTF behaviour classifier source timeframe (was REGIME_TF_1H)
REGIME_TF_4H = REGIME_HTF_TF   # backward compat alias
REGIME_TF_1H = REGIME_LTF_TF   # backward compat alias
REGIME_TIMEFRAMES = ["1H", "4H"]  # kept for backwards compat (covers both)
# Root of the pipeline processed data
_PROCESSED_DIR = str(_ENV["processed"] / "histdata")
MACRO_CORR_PATH = str(_ENV["weights"] / "macro_correlations.json")
# Fixed list — must match feature_engine.INDEX_NAMES exactly for macro correlations.
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


_SPLIT_BOUNDARIES: dict[str, dict] = {}
_PARQUET_CACHE: dict = {}  # (symbol, tf) -> raw DataFrame, populated once on first read


def _load_split_boundaries(fold_id: str | int | None = None) -> dict:
    """
    Load train/val/test date boundaries from ml_training/datasets/split_summary.json.
    Rolling summaries contain multiple train/validation folds over the non-test
    history. fold_id selects one fold; "latest" is the production default.
    """
    global _SPLIT_BOUNDARIES
    requested = str(fold_id if fold_id is not None else RETRAIN_ROLLING_FOLD).strip().lower()
    if requested in _SPLIT_BOUNDARIES:
        return _SPLIT_BOUNDARIES[requested]

    import pandas as pd

    def _ts(value):
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    split_path = os.path.join(
        str(_ENV["ml_training"]), "datasets", "split_summary.json"
    )
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Required rolling split summary not found: {split_path}. Run pipeline/step5_split.py first."
        )

    with open(split_path) as f:
        summary = json.load(f)
    folds = summary.get("folds") or summary.get("rolling_folds") or []
    if not folds:
        raise ValueError(
            f"{split_path} does not contain rolling folds; regenerate it with pipeline/step5_split.py"
        )

    selected_fold = None
    if requested in {"", "latest", "last"}:
        selected_fold = folds[-1]
    elif requested == "all":
        raise ValueError(
            "RETRAIN_ROLLING_FOLD='all' is only valid for REGIME_ROLLING_FOLDS; "
            "single dataset loads require a concrete fold id or 'latest'."
        )
    else:
        for fold in folds:
            candidates = {
                str(fold.get("fold_id", "")).lower(),
                str(fold.get("id", "")).lower(),
                str(fold.get("index", "")).lower(),
            }
            if requested in candidates:
                selected_fold = fold
                break
        if selected_fold is None:
            try:
                selected_fold = folds[int(requested)]
            except (IndexError, ValueError) as exc:
                valid = [
                    str(fold.get("fold_id", fold.get("id", i)))
                    for i, fold in enumerate(folds)
                ]
                raise ValueError(
                    f"Unknown RETRAIN_ROLLING_FOLD={requested!r}; valid folds are {valid}"
                ) from exc

    ranges = selected_fold.get("date_ranges", selected_fold)
    fold_name = str(
        selected_fold.get(
            "fold_id",
            selected_fold.get("id", selected_fold.get("index", "latest")),
        )
    )

    _SPLIT_BOUNDARIES[requested] = {
        "fold_id": fold_name,
        "train_start": _ts(ranges["train"]["start"]),
        "train_end":   _ts(ranges["train"]["end"]),
        "val_start":   _ts(ranges["validation"]["start"]),
        "val_end":     _ts(ranges["validation"]["end"]),
        "test_start":  _ts(ranges["test"]["start"]),
        "test_end":    _ts(ranges["test"]["end"]),
        "fold_count":  len(folds),
    }
    logger.info(
        "Split boundaries loaded fold=%s/%s — train %s→%s  val %s→%s  test %s→%s",
        _SPLIT_BOUNDARIES[requested]["fold_id"],
        max(1, _SPLIT_BOUNDARIES[requested]["fold_count"]),
        _SPLIT_BOUNDARIES[requested]["train_start"].date(),
        _SPLIT_BOUNDARIES[requested]["train_end"].date(),
        _SPLIT_BOUNDARIES[requested]["val_start"].date(),
        _SPLIT_BOUNDARIES[requested]["val_end"].date(),
        _SPLIT_BOUNDARIES[requested]["test_start"].date(),
        _SPLIT_BOUNDARIES[requested]["test_end"].date(),
    )
    return _SPLIT_BOUNDARIES[requested]


def _available_rolling_folds() -> list[str]:
    split_path = os.path.join(str(_ENV["ml_training"]), "datasets", "split_summary.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Required rolling split summary not found: {split_path}. Run pipeline/step5_split.py first."
        )
    with open(split_path) as f:
        summary = json.load(f)
    folds = summary.get("folds") or summary.get("rolling_folds") or []
    if not folds:
        raise ValueError(
            f"{split_path} does not contain rolling folds; regenerate it with pipeline/step5_split.py"
        )
    out = []
    for i, fold in enumerate(folds):
        out.append(str(fold.get("fold_id", fold.get("id", i))))
    return out


def _load_ohlcv(symbol: str, timeframe: str = "15M",
                split: str = "train",
                fold_id: str | int | None = None) -> "pd.DataFrame | None":
    """Load OHLCV from processed_data/histdata/{SYM}_{TF}.parquet (step0 output).
    No CSV fallbacks — all data must be resampled from M1 by step0_resample.py.

    split: one of 'train', 'val', 'test', 'all'
      - 'train'  → selected fold train_start through train_end
      - 'val'    → selected fold val_start through val_end
      - 'test'   → final blind test_start through test_end
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
            raise FileNotFoundError(
                f"Missing parquet {parquet_path}; run pipeline/step0_resample.py first"
            )
        try:
            raw = pd.read_parquet(parquet_path)
            raw.index = pd.to_datetime(raw.index, utc=True, errors="coerce")
            keep = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
            missing = {"open", "high", "low", "close"} - set(keep)
            if missing:
                raise ValueError(f"{parquet_path} missing required OHLC columns: {sorted(missing)}")
            raw = raw[keep].dropna(subset=["open", "high", "low", "close"])
            raw = raw[raw.index.notna()].sort_index()
            _PARQUET_CACHE[cache_key] = raw
            logger.debug("Cached parquet %s/%s (%d bars)", symbol, tf_upper, len(raw))
        except Exception as exc:
            raise RuntimeError(f"Failed to load parquet {symbol}/{tf_upper}: {exc}") from exc
        df = _PARQUET_CACHE[cache_key].copy()

    # Apply temporal split boundaries
    if split not in {"train", "val", "test", "all"}:
        raise ValueError(f"Invalid split={split!r}; expected train, val, test, or all")
    if split != "all":
        bounds = _load_split_boundaries(fold_id=fold_id)
        if split == "train":
            df = df[(df.index >= bounds["train_start"]) & (df.index <= bounds["train_end"])]
        elif split == "val":
            df = df[(df.index >= bounds["val_start"]) & (df.index <= bounds["val_end"])]
        elif split == "test":
            df = df[(df.index >= bounds["test_start"]) & (df.index <= bounds["test_end"])]

    if len(df) == 0:
        raise ValueError(f"_load_ohlcv: {symbol}/{tf_upper} split={split} yielded 0 rows")

    _fold_label = fold_id if fold_id is not None else RETRAIN_ROLLING_FOLD
    logger.info("Loaded %s/%s split=%s fold=%s: %d bars (%s → %s)",
                symbol, tf_upper, split, _fold_label, len(df),
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
    Kaggle-optimised GRU training: build 15M execution segments for all symbols
    and train in one pass via model.train_multi().
    Avoids sequential save/reload cycles. Caps the combined group at 2M sequences
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

        for tf in GRU_TIMEFRAMES:
            df = _load_ohlcv(sym, tf, split=RETRAIN_DATA_SPLIT)
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

            segments.append({
                "df": df_train,
                "labels": labels_train,
                "df_htf": htf_train,
                "symbol": sym,
                "timeframe": tf,
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
    """GRU-LSTM: train the 15M execution model across all symbols.

    On Kaggle (large RAM): uses train_multi() to concatenate all symbols into one
    combined 15M dataset and keep the GPU fed continuously.
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

    # On Kaggle use combined multi-symbol training (30GB RAM available)
    if _ENV["on_kaggle"] and not dry_run:
        return _retrain_gru_multi(model, symbols)

    backup_done = False
    for sym in symbols:
        # HTF context loaded with split="all" — regime/structure features need full
        # history for context, actual training data is sliced separately below
        all_htf = {tf: _load_ohlcv(sym, tf, split="all") for tf in ("5M", "1H", "4H", "1D")}

        for tf in GRU_TIMEFRAMES:
            df = _load_ohlcv(sym, tf, split=RETRAIN_DATA_SPLIT)
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
                # Delete stale weights when SEQUENCE_FEATURES changes.
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
                           mode: str = "ltf_behaviour",
                           data_split: str = RETRAIN_DATA_SPLIT,
                           fold_id: str | int | None = None) -> tuple:
    """
    Build (X_all, y_all, sw_all) for one regime classifier.

    source_tf: the TF we build feature matrices on (e.g. "4H" or "1H").
    label_tf:  source TF for rule labels.
    mode: "htf_bias" → 3-class labels for HTF classifier.
          "ltf_behaviour" → 5-column score targets for LTF behaviour head.
    Returns (X, y, sample_weight, n_samples).
    sample_weight: float32 array (N,) — per-bar confidence from create_rule_labels.
    """
    from models.regime_classifier import RegimeClassifier as _RC
    from services.feature_engine import REGIME_4H_FEATURES, REGIME_1H_FEATURES
    import gc as _gc
    import numpy as _np

    X_parts:  list = []
    y_parts:  list = []
    sw_parts: list = []
    samples = 0
    _score_mode = mode == "ltf_behaviour"
    if mode not in {"htf_bias", "ltf_behaviour"}:
        raise ValueError(f"Unsupported regime mode={mode!r}; expected htf_bias or ltf_behaviour")
    _classes = (
        ["BIAS_UP", "BIAS_DOWN", "BIAS_NEUTRAL"]
        if mode == "htf_bias"
        else ["TRENDING", "RANGING", "CONSOLIDATING", "VOLATILE"]
    )
    _n_classes = len(_classes)
    _group_counts: dict[str, _np.ndarray] = {}
    _year_counts: dict[int, _np.ndarray] = {}
    _score_outputs = []
    _group_score_sums: dict[str, _np.ndarray] = {}
    _group_score_counts: dict[str, int] = {}
    _year_score_sums: dict[int, _np.ndarray] = {}
    _year_score_counts: dict[int, int] = {}
    if _score_mode:
        from models.regime_classifier import LTF_SCORE_OUTPUTS as _LTF_SCORE_OUTPUTS

        _score_outputs = list(_LTF_SCORE_OUTPUTS)
    _feature_names = list(REGIME_1H_FEATURES if _score_mode else REGIME_4H_FEATURES)
    _need_bos = "swing_hh_hl_count" in _feature_names
    _need_sweep = "liquidity_sweep_24h" in _feature_names

    # Cache label_tf dfs per symbol
    _label_cache: dict = {}
    for sym in symbols:
        df_l = _load_ohlcv(sym, label_tf, split=data_split, fold_id=fold_id)
        if len(df_l) <= 50:
            raise ValueError(
                f"Regime[{source_tf} mode={mode} split={data_split} fold={fold_id}] "
                f"{sym}/{label_tf} has only {len(df_l)} label bars"
            )
        _label_cache[sym] = df_l

    for sym in symbols:
        all_htf = {
            tf: _load_ohlcv(sym, tf, split="all")
            for tf in ("5M", "15M", "1H", "4H", "1D")
        }
        grp = _group_for_symbol(sym)

        df = _load_ohlcv(sym, source_tf, split=data_split, fold_id=fold_id)
        if len(df) <= 200:
            raise ValueError(
                f"Regime[{source_tf} mode={mode} split={data_split} fold={fold_id}] "
                f"{sym} has only {len(df)} source bars"
            )
        df = _RC._ensure_structure_columns(
            df,
            require_bos=_need_bos,
            require_sweep=_need_sweep,
        )

        if dry_run:
            logger.info("DRY RUN: Regime[%s mode=%s] %s — %d bars", source_tf, mode, sym, len(df))
            samples += len(df)
            del df
            continue

        end_ts = df.index[-1]
        htf_train: dict = {}
        for htf_tf, htf_df in all_htf.items():
            if len(htf_df) < 14:
                raise ValueError(f"Regime[{source_tf} mode={mode}] {sym}/{htf_tf} has <14 context bars")
            trimmed = htf_df[htf_df.index <= end_ts]
            if len(trimmed) < 14:
                raise ValueError(
                    f"Regime[{source_tf} mode={mode}] {sym}/{htf_tf} has <14 context bars before {end_ts}"
                )
            htf_train[htf_tf] = trimmed
        if source_tf not in htf_train:
            htf_train[source_tf] = df

        X_sym = _RC._build_feature_matrix(
            df,
            htf_train,
            sym,
            feature_names=_feature_names,
        )

        # HTF uses outcome-aware structural labels. LTF behaviour uses causal
        # multi-output score targets instead of a forced class.
        if mode == "htf_bias":
            labels, conf = _RC.create_structural_labels(
                df, timeframe=source_tf, mode="htf_bias", return_confidence=True, symbol=sym,
            )
        else:
            labels, conf = _RC.create_behaviour_score_targets(
                df, timeframe=source_tf, symbol=sym, return_confidence=True,
            )

        n = len(df)
        step = max(1, (n - 50) // 100_000)
        idx  = _np.arange(50, n, step)
        if len(idx) == 0:
            raise ValueError(
                f"Regime[{source_tf} mode={mode} split={data_split} fold={fold_id}] "
                f"{sym} produced zero sampled rows"
            )
        X_parts.append(X_sym[idx])
        if _score_mode:
            y_sample = labels.iloc[idx].to_numpy(dtype=_np.float32)
        else:
            y_sample = labels.iloc[idx].values.astype(_np.int64)
        y_parts.append(y_sample)
        sw_parts.append(conf.iloc[idx].values.astype(_np.float32))
        samples += len(idx)
        if _score_mode:
            means = _np.nanmean(y_sample, axis=0)
            _group_score_sums.setdefault(grp, _np.zeros(len(_score_outputs), dtype=_np.float64))
            _group_score_counts[grp] = _group_score_counts.get(grp, 0) + len(y_sample)
            _group_score_sums[grp] += _np.nansum(y_sample, axis=0)
            if hasattr(labels.index, "year"):
                years = labels.index[idx].year
                for _yr in _np.unique(years):
                    _mask = years == _yr
                    _year_score_sums.setdefault(int(_yr), _np.zeros(len(_score_outputs), dtype=_np.float64))
                    _year_score_counts[int(_yr)] = _year_score_counts.get(int(_yr), 0) + int(_mask.sum())
                    _year_score_sums[int(_yr)] += _np.nansum(y_sample[_mask], axis=0)
            logger.info(
                "Regime[%s mode=%s split=%s fold=%s]: collected %s — %d samples (group=%s) score_means=%s",
                source_tf,
                mode,
                data_split,
                fold_id if fold_id is not None else RETRAIN_ROLLING_FOLD,
                sym,
                len(idx),
                grp,
                {_score_outputs[i]: round(float(means[i]), 4) for i in range(len(_score_outputs))},
            )
        else:
            clean_mask = conf.iloc[idx].values.astype(_np.float32) >= float(os.getenv("REGIME_MIN_LABEL_CONFIDENCE", "0.4"))
            dist = _np.bincount(y_sample, minlength=_n_classes)[:_n_classes]
            clean_dist = _np.bincount(y_sample[clean_mask], minlength=_n_classes)[:_n_classes]
            _group_counts.setdefault(grp, _np.zeros(_n_classes, dtype=_np.int64))
            _group_counts[grp] += dist
            if hasattr(labels.index, "year"):
                years = labels.index[idx].year
                for _yr in _np.unique(years):
                    _mask = years == _yr
                    _year_counts.setdefault(int(_yr), _np.zeros(_n_classes, dtype=_np.int64))
                    _year_counts[int(_yr)] += _np.bincount(y_sample[_mask], minlength=_n_classes)[:_n_classes]
            logger.info(
                "Regime[%s mode=%s split=%s fold=%s]: collected %s — %d samples (group=%s) labels=%s clean=%s",
                source_tf,
                mode,
                data_split,
                fold_id if fold_id is not None else RETRAIN_ROLLING_FOLD,
                sym,
                len(idx),
                grp,
                {_classes[i]: int(dist[i]) for i in range(_n_classes)},
                {_classes[i]: int(clean_dist[i]) for i in range(_n_classes)},
            )
        del df
        _gc.collect()

    if dry_run:
        return None, None, None, samples

    if not X_parts:
        raise RuntimeError(
            f"Regime[{source_tf} mode={mode} split={data_split} fold={fold_id}] produced no training parts"
        )

    import numpy as _np2
    X_all  = _np2.concatenate(X_parts,  axis=0)
    y_all  = _np2.concatenate(y_parts,  axis=0)
    sw_all = _np2.concatenate(sw_parts, axis=0)
    if _group_counts:
        logger.info(
            "Regime[%s mode=%s] label distribution by symbol group: %s",
            source_tf,
            mode,
            {
                grp: {_classes[i]: int(vals[i]) for i in range(_n_classes)}
                for grp, vals in sorted(_group_counts.items())
            },
        )
    if _group_score_sums:
        logger.info(
            "Regime[%s mode=%s] score means by symbol group: %s",
            source_tf,
            mode,
            {
                grp: {
                    _score_outputs[i]: round(float(vals[i] / max(_group_score_counts.get(grp, 1), 1)), 4)
                    for i in range(len(_score_outputs))
                }
                for grp, vals in sorted(_group_score_sums.items())
            },
        )
    if _year_counts:
        logger.info(
            "Regime[%s mode=%s] label distribution by year: %s",
            source_tf,
            mode,
            {
                yr: {_classes[i]: int(vals[i]) for i in range(_n_classes)}
                for yr, vals in sorted(_year_counts.items())
            },
        )
    if _year_score_sums:
        logger.info(
            "Regime[%s mode=%s] score means by year: %s",
            source_tf,
            mode,
            {
                yr: {
                    _score_outputs[i]: round(float(vals[i] / max(_year_score_counts.get(yr, 1), 1)), 4)
                    for i in range(len(_score_outputs))
                }
                for yr, vals in sorted(_year_score_sums.items())
            },
        )
    del X_parts, y_parts, sw_parts
    _gc.collect()
    return X_all, y_all, sw_all, samples


def _regime_diagnostics(model, group_gmms: dict, symbols: list, source_tf: str,
                        fold_id: str | int | None = None) -> None:
    """Log persistence and return-separation using the same labels as training."""
    from models.regime_classifier import RegimeClassifier as _RC_diag
    _diag_sym = symbols[-1] if symbols else None
    if not _diag_sym:
        raise ValueError("Regime diagnostics require at least one symbol")
    _diag_df = _load_ohlcv(_diag_sym, source_tf, split=RETRAIN_DATA_SPLIT, fold_id=fold_id)
    if len(_diag_df) < 200:
        raise ValueError(f"Regime diagnostics {source_tf} has only {len(_diag_df)} bars for {_diag_sym}")
    _mode = "htf_bias" if str(source_tf).upper() == "4H" else "ltf_behaviour"
    if _mode == "ltf_behaviour":
        from models.regime_classifier import LTF_SCORE_OUTPUTS as _LTF_SCORE_OUTPUTS

        _scores, _conf = _RC_diag.create_behaviour_score_targets(
            _diag_df, timeframe=source_tf, symbol=_diag_sym, return_confidence=True,
        )
        _summary = {}
        for _name in _LTF_SCORE_OUTPUTS:
            _series = _scores[_name]
            _summary[_name] = {
                "mean": round(float(_series.mean()), 4),
                "q10": round(float(_series.quantile(0.10)), 4),
                "q50": round(float(_series.quantile(0.50)), 4),
                "q90": round(float(_series.quantile(0.90)), 4),
            }
        logger.info(
            "Regime[%s mode=%s fold=%s] LTF score diagnostics on %s:\n%s",
            source_tf,
            _mode,
            fold_id if fold_id is not None else RETRAIN_ROLLING_FOLD,
            _diag_sym,
            _summary,
        )
        return
    _classes = ["BIAS_UP", "BIAS_DOWN", "BIAS_NEUTRAL"]
    _min_conf = float(os.getenv("REGIME_MIN_LABEL_CONFIDENCE", "0.4"))
    _lbl, _conf = _RC_diag.create_structural_labels(
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


def retrain_regime(dry_run: bool = False) -> dict:
    """
    Train hierarchical regime cascade:
      1. HTF classifier (regime_htf.pkl) — 3-class bias (BIAS_UP/DOWN/NEUTRAL).
         mode="htf_bias", trained on 4H bars with realised forward-path bias labels.
         GPU-parallel across both T4s via DataParallel.
      2. LTF score head (regime_ltf.pkl) — five independent behaviour scores.
         mode="ltf_behaviour", trained on 1H bars with causal score targets.
         Same DataParallel setup — both GPUs stay hot across both trains.

    Each classifier now builds only its own compact feature contract:
    REGIME_4H_FEATURES for HTF bias and REGIME_1H_FEATURES for LTF behaviour.
    """
    import time as _time
    _t0_regime = _time.perf_counter()
    logger.info("=== RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 5-score behaviour) ===")
    from models.regime_classifier import RegimeClassifier as _RC
    import gc as _gc

    symbols = _get_symbols("RETRAIN_SYMBOLS_REGIME", MAJOR_SYMBOLS)

    group_gmms_htf: dict = {}
    group_gmms_ltf: dict = {}
    if os.getenv("REGIME_FIT_GMM", "0").strip().lower() in {"1", "true", "yes", "on"}:
        logger.info("Regime: fitting optional per-group GMMs for diagnostics...")
        group_dfs_4h: dict = {"dollar": [], "cross": [], "gold": []}
        group_dfs_1h: dict = {"dollar": [], "cross": [], "gold": []}
        for sym in symbols:
            df_4h = _load_ohlcv(sym, "4H", split=RETRAIN_DATA_SPLIT)
            if len(df_4h) <= 200:
                raise ValueError(f"REGIME_FIT_GMM=1 but {sym}/4H has only {len(df_4h)} bars")
            group_dfs_4h[_group_for_symbol(sym)].append(df_4h)
            df_1h = _load_ohlcv(sym, "1H", split=RETRAIN_DATA_SPLIT)
            if len(df_1h) <= 200:
                raise ValueError(f"REGIME_FIT_GMM=1 but {sym}/1H has only {len(df_1h)} bars")
            group_dfs_1h[_group_for_symbol(sym)].append(df_1h)

        _t_gmm_htf = _time.perf_counter()
        for grp, dfs in group_dfs_4h.items():
            if dfs:
                _t_grp = _time.perf_counter()
                gmm, scaler, cluster_labels = _RC.fit_global_gmm(dfs, timeframe="4H", mode="htf_bias")
                group_gmms_htf[grp] = (gmm, scaler, cluster_labels)
                logger.info("Regime HTF GMM '%s' fitted on %d 4H dfs in %.1fs",
                            grp, len(dfs), _time.perf_counter() - _t_grp)
        logger.info("Regime phase GMM HTF total: %.1fs", _time.perf_counter() - _t_gmm_htf)

        _t_gmm_ltf = _time.perf_counter()
        for grp, dfs in group_dfs_1h.items():
            if dfs:
                _t_grp = _time.perf_counter()
                gmm, scaler, cluster_labels = _RC.fit_global_gmm(dfs, timeframe="1H", mode="ltf_behaviour")
                group_gmms_ltf[grp] = (gmm, scaler, cluster_labels)
                logger.info("Regime LTF GMM '%s' fitted on %d 1H dfs in %.1fs",
                            grp, len(dfs), _time.perf_counter() - _t_grp)
        logger.info("Regime phase GMM LTF total: %.1fs", _time.perf_counter() - _t_gmm_ltf)
        del group_dfs_4h, group_dfs_1h
    else:
        logger.info("Regime: skipping GMM fit; structural forward-path labels are the default target")

    _gc.collect()

    fold_selector = os.getenv("REGIME_ROLLING_FOLDS", "all").strip().lower()
    available_folds = _available_rolling_folds()
    if fold_selector in {"all", "rolling", "cv"}:
        active_folds = available_folds
    elif fold_selector in {"", "latest", "last"}:
        active_folds = [available_folds[-1]]
    else:
        active_folds = [fold_selector]
    logger.info("Regime rolling folds selected: %s", active_folds)

    results: dict = {}
    total_samples = 0
    backed_up: set[str] = set()

    for fold_i, fold_id in enumerate(active_folds):
        fold_key = str(fold_id)
        fold_results: dict = {}
        logger.info("=== Regime rolling fold %s/%s: %s ===",
                    fold_i + 1, len(active_folds), fold_key)

        # ── HTF bias classifier (3-class) ────────────────────────────────────
        logger.info("Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...")
        _t_htf_ds = _time.perf_counter()
        X_4h, y_4h, sw_4h, n_4h = _build_regime_dataset(
            symbols, source_tf="4H", label_tf="4H",
            group_gmms=group_gmms_htf, dry_run=dry_run, mode="htf_bias",
            data_split="train", fold_id=fold_id,
        )
        X_4h_val, y_4h_val, sw_4h_val, n_4h_val = _build_regime_dataset(
            symbols, source_tf="4H", label_tf="4H",
            group_gmms=group_gmms_htf, dry_run=dry_run, mode="htf_bias",
            data_split="val", fold_id=fold_id,
        )
        logger.info(
            "Regime phase HTF dataset build fold=%s: %.1fs (train=%d val=%d)",
            fold_key, _time.perf_counter() - _t_htf_ds, n_4h, n_4h_val,
        )
        total_samples += n_4h + n_4h_val
        if not dry_run:
            htf_path = os.path.join(WEIGHTS_DIR, "regime_htf.pkl")
            if htf_path not in backed_up:
                _backup_weights(htf_path)
                backed_up.add(htf_path)
            model_htf = _RC(timeframe="4H", mode="htf_bias")
            model_htf._model = None
            model_htf._loaded = False
            _t_htf_train = _time.perf_counter()
            res_4h = model_htf.train_on_arrays(
                X_4h, y_4h, sample_weight=sw_4h,
                X_val=X_4h_val, y_val=y_4h_val, sample_weight_val=sw_4h_val,
            )
            logger.info("Regime phase HTF train fold=%s: %.1fs", fold_key, _time.perf_counter() - _t_htf_train)
            del X_4h, y_4h, sw_4h, X_4h_val, y_4h_val, sw_4h_val; _gc.collect()
            if res_4h.get("error"):
                raise RuntimeError(f"Regime HTF training failed fold={fold_key}: {res_4h['error']}")
            else:
                logger.info("Regime HTF complete fold=%s: acc=%.3f, train=%d val=%d per_class=%s",
                            fold_key, res_4h.get("accuracy", 0), n_4h, n_4h_val,
                            res_4h.get("per_class_accuracy", {}))
                _regime_diagnostics(model_htf, group_gmms_htf, symbols, "4H", fold_id=fold_id)
                fold_results["HTF"] = res_4h
                log_retrain("regime_classifier_htf", {**res_4h, "status": "complete", "fold_id": fold_key})
            if not _htf_regime_artifact_exists():
                raise RuntimeError("Regime HTF weights were not created at regime_htf.pkl")

        # ── LTF behaviour score head (5 outputs) ─────────────────────────────
        logger.info("Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...")
        _t_ltf_ds = _time.perf_counter()
        X_1h, y_1h, sw_1h, n_1h = _build_regime_dataset(
            symbols, source_tf="1H", label_tf="1H",
            group_gmms=group_gmms_ltf, dry_run=dry_run, mode="ltf_behaviour",
            data_split="train", fold_id=fold_id,
        )
        X_1h_val, y_1h_val, sw_1h_val, n_1h_val = _build_regime_dataset(
            symbols, source_tf="1H", label_tf="1H",
            group_gmms=group_gmms_ltf, dry_run=dry_run, mode="ltf_behaviour",
            data_split="val", fold_id=fold_id,
        )
        logger.info(
            "Regime phase LTF dataset build fold=%s: %.1fs (train=%d val=%d)",
            fold_key, _time.perf_counter() - _t_ltf_ds, n_1h, n_1h_val,
        )
        total_samples += n_1h + n_1h_val
        if not dry_run:
            ltf_path = os.path.join(WEIGHTS_DIR, "regime_ltf.pkl")
            if ltf_path not in backed_up:
                _backup_weights(ltf_path)
                backed_up.add(ltf_path)
            model_ltf = _RC(timeframe="1H", mode="ltf_behaviour")
            model_ltf._model = None
            model_ltf._loaded = False
            _t_ltf_train = _time.perf_counter()
            res_1h = model_ltf.train_on_arrays(
                X_1h, y_1h, sample_weight=sw_1h,
                X_val=X_1h_val, y_val=y_1h_val, sample_weight_val=sw_1h_val,
            )
            logger.info("Regime phase LTF train fold=%s: %.1fs", fold_key, _time.perf_counter() - _t_ltf_train)
            del X_1h, y_1h, sw_1h, X_1h_val, y_1h_val, sw_1h_val; _gc.collect()
            if res_1h.get("error"):
                raise RuntimeError(f"Regime LTF training failed fold={fold_key}: {res_1h['error']}")
            else:
                logger.info("Regime LTF complete fold=%s: score_accuracy=%.3f, train=%d val=%d mae=%s",
                            fold_key, res_1h.get("accuracy", 0), n_1h, n_1h_val,
                            res_1h.get("score_mae", {}))
                _regime_diagnostics(model_ltf, group_gmms_ltf, symbols, "1H", fold_id=fold_id)
                fold_results["LTF"] = res_1h
                log_retrain("regime_classifier_ltf", {**res_1h, "status": "complete", "fold_id": fold_key})
            if not _ltf_regime_artifact_exists():
                raise RuntimeError("Regime LTF weights were not created at regime_ltf.pkl")

        results[fold_key] = fold_results

    logger.info("Regime retrain total: %.1fs (%d train+val samples)",
                _time.perf_counter() - _t0_regime, total_samples)
    if dry_run:
        return {"dry_run": True, "samples": total_samples, "folds": active_folds}

    any_error = any(
        isinstance(r, dict) and r.get("error")
        for fold_result in results.values()
        for r in fold_result.values()
    )
    if any_error:
        return {"error": "One or more regime fold trainings failed", "details": results}

    return {
        "trained": True,
        "samples": total_samples,
        "folds": active_folds,
        "results": results,
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
      trade_patterns    (len(SEQUENCE_FEATURES)) — per-bar technical GRU snapshot, all training bars
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
        from services.feature_engine import FeatureEngine, SEQUENCE_FEATURES, REGIME_4H_FEATURES

        import time as _time
        _t0_vs = _time.perf_counter()
        logger.info("=== VectorStore: building similarity indices (parallel feature build) ===")
        store = VectorStore()
        gru_model = GRULSTMPredictor()
        fe = FeatureEngine()

        MAX_BARS_PER_SYMBOL = 50_000
        _n_workers = int(os.getenv("RETRAIN_CPU_WORKERS", "4"))

        def _build_sym_vectors(sym: str):
            """CPU-only: load data + build all three feature arrays for one symbol.
            Returns (sym, tp_vecs, tp_metas, ms_vecs, ms_metas, emb_seqs, emb_metas_idx, df_index)
            or raises on failure."""
            df = _load_ohlcv(sym, "15M", split=RETRAIN_DATA_SPLIT)
            if df is None or len(df) < 200:
                return None

            result = {"sym": sym, "df_index": df.index}
            all_htf = {tf: _load_ohlcv(sym, tf, split="all")
                       for tf in ("5M", "15M", "1H", "4H", "1D")}

            # trade_patterns
            try:
                feat_df = fe._build_sequence_df(df, all_htf, symbol=sym)
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
                X_htf = _RC._build_feature_matrix(
                    df,
                    all_htf,
                    sym,
                    feature_names=REGIME_4H_FEATURES,
                )
                step = max(1, len(df) // MAX_BARS_PER_SYMBOL)
                idx = np.arange(50, len(df), step)
                idx = idx[idx < len(X_htf)]
                rvecs = X_htf[idx].astype("float32")
                rmetas = [{"symbol": sym, "timeframe": "15M", "ts": str(df.index[i])} for i in idx]
                result["ms"] = (rvecs, rmetas)
                del X_htf
            except Exception as exc:
                logger.warning("VectorStore market_structures failed for %s: %s", sym, exc)

            # regime_embeddings: build sequences (GPU call happens in main thread)
            if gru_model.is_trained:
                try:
                    feat_df2 = fe._build_sequence_df(df, all_htf, symbol=sym)
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
                logger.info(
                    "VectorStore market_structures: +%d vectors (%d-dim 4H) for %s",
                    len(rvecs), len(REGIME_4H_FEATURES), sym,
                )

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


def _file_sha256(path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _code_commit() -> str:
    try:
        root = Path(_ENV["base"]).resolve().parent
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def _artifact_hashes(model_name: str) -> dict:
    candidates = {
        "gru_lstm": [
            os.path.join(WEIGHTS_DIR, "gru_lstm", "model.pt"),
            os.path.join(WEIGHTS_DIR, "gru_lstm", "weights_manifest.json"),
        ],
        "regime_classifier": [
            os.path.join(WEIGHTS_DIR, "regime_htf.pkl"),
            os.path.join(WEIGHTS_DIR, "regime_ltf.pkl"),
        ],
        "quality_scorer": [
            os.path.join(WEIGHTS_DIR, "quality_scorer.pkl"),
        ],
        "rl_agent": [
            os.path.join(WEIGHTS_DIR, "rl_ppo", "model.zip"),
        ],
    }.get(model_name, [])
    return {
        os.path.relpath(path, WEIGHTS_DIR): _file_sha256(path)
        for path in candidates
        if os.path.exists(path)
    }


def log_retrain(model_name: str, result: dict) -> None:
    os.makedirs("logs", exist_ok=True)
    path = "logs/retrain_history.jsonl"
    split_hash = _file_sha256(SPLIT_SUMMARY_PATH)
    record = {
        "run_id": os.getenv("RUN_ID", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "retrain_data_split": RETRAIN_DATA_SPLIT,
        "retrain_rolling_fold": RETRAIN_ROLLING_FOLD,
        "regime_rolling_folds": os.getenv("REGIME_ROLLING_FOLDS", "all"),
        "split_summary_hash": split_hash,
        "journal_sha256": _file_sha256(JOURNAL_PATH),
        "artifact_hashes": _artifact_hashes(model_name),
        "code_commit": _code_commit(),
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
