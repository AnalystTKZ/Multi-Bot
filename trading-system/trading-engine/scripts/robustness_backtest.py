#!/usr/bin/env python3
"""
robustness_backtest.py — Out-of-sample robustness test on symbols the model was NEVER trained on.

Tests the 5 HistData M1 currency pairs completely outside the 11-symbol training universe:
  USDZAR, USDMXN, EURAUD, GBPCHF, AUDJPY

Date range: 2016-01-01 → 2025-12-31 (9 years)
Split into 3 equal 3-year windows:
  Epoch 1: 2016-01-01 → 2018-12-31  (Pre-COVID)
  Epoch 2: 2019-01-01 → 2021-12-31  (COVID & Recovery)
  Epoch 3: 2022-01-01 → 2025-12-31  (Rate Hike Cycle)

Checkpoint-resume:
  Each (symbol, epoch) result is saved immediately after completion to
  backtest_results/robustness_checkpoint.json and pushed to GitHub.
  On re-run, already-completed combinations are skipped automatically.
  Delete robustness_checkpoint.json to start fresh.

Usage (Kaggle — run from trading-engine/ directory):
    python scripts/robustness_backtest.py
    python scripts/robustness_backtest.py --epoch 2        # single epoch
    python scripts/robustness_backtest.py --symbol USDZAR  # single symbol
    python scripts/robustness_backtest.py --reset          # clear checkpoint, start over

Required env var for GitHub push:
    GITHUB_TOKEN  — personal access token with repo write scope
"""

from __future__ import annotations

import argparse
from collections import deque
import gc
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

_ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_TS_DIR     = os.path.join(_ENGINE_DIR, "..")
sys.path.insert(0, _ENGINE_DIR)
sys.path.insert(0, _TS_DIR)
os.chdir(_ENGINE_DIR)  # models use relative "weights/..." paths

# Must be set before any model module is imported — they call _get_device() at
# module level. Bypasses the Kaggle CUDA training guard for inference-only runs.
os.environ.setdefault("INFERENCE_ONLY", "1")

import numpy as np
import pandas as pd

# GPU / CPU performance flags (safe to apply before torch models load)
try:
    import torch as _torch
    if _torch.cuda.is_available():
        _torch.backends.cudnn.benchmark        = True
        _torch.backends.cuda.matmul.allow_tf32 = True
        _torch.backends.cudnn.allow_tf32       = True
    _n_cpu = int(os.getenv("ROBUSTNESS_CPU_WORKERS", "1"))
    _torch.set_num_threads(_n_cpu)
    _torch.set_num_interop_threads(max(1, _n_cpu // 2))
except Exception:
    pass

_ROBUSTNESS_LOG_DIR = os.path.join(_ENGINE_DIR, "logs")
os.makedirs(_ROBUSTNESS_LOG_DIR, exist_ok=True)
_ROBUSTNESS_LOG_FILE = os.path.join(
    _ROBUSTNESS_LOG_DIR,
    f"robustness_backtest_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log",
)
_CONSOLE_VERBOSE = str(os.getenv("BACKTEST_CONSOLE_VERBOSE", "0")).lower() in ("1", "true", "yes", "on")
_CONSOLE_LEVEL = logging.INFO if _CONSOLE_VERBOSE else logging.WARNING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(_ROBUSTNESS_LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
for _handler in logging.getLogger().handlers:
    if isinstance(_handler, logging.StreamHandler) and not isinstance(_handler, logging.FileHandler):
        _handler.setLevel(_CONSOLE_LEVEL)
    elif isinstance(_handler, logging.FileHandler):
        _handler.setLevel(logging.INFO)
logger = logging.getLogger("robustness_bt")

# ─── Config (mirrors run_backtest.py exactly) ─────────────────────────────────
INITIAL_CAPITAL    = 10_000.0
RISK_PER_TRADE     = 0.01
CAPITAL_PER_TRADER = 1.00
COMMISSION_PCT     = 0.001
SLIPPAGE_PCT       = 0.0002
MAX_DAILY_LOSS_PCT = 0.02
MAX_DRAWDOWN_PCT   = 0.20
COOLDOWN_BARS      = 10
MAX_HOLD_BARS      = 200
MIN_CONFIDENCE     = float(os.getenv("MIN_CONFIDENCE", "0.0"))
_BACKTEST_WARMUP = {
    "5M": pd.Timedelta(days=7),
    "15M": pd.Timedelta(days=14),
    "1H": pd.Timedelta(days=60),
    "4H": pd.Timedelta(days=180),
    "1D": pd.Timedelta(days=400),
}
_HTF_REGIME_NAMES = np.array(["BIAS_UP", "BIAS_DOWN", "BIAS_NEUTRAL"], dtype=object)
_LTF_REGIME_NAMES = np.array(["TRENDING", "RANGING", "CONSOLIDATING", "VOLATILE"], dtype=object)
_PM_DISABLED = str(os.getenv("BACKTEST_DISABLE_PM", "0")).lower() in ("1", "true", "yes", "on")

_PM_SETTINGS = SimpleNamespace(
    ACCOUNT_BALANCE=INITIAL_CAPITAL,
    CAPITAL_PER_TRADER=CAPITAL_PER_TRADER,
    RISK_PER_TRADE=RISK_PER_TRADE,
    MAX_DAILY_LOSS_PCT=MAX_DAILY_LOSS_PCT,
)

# Path resolution via env_config (same layer used by all pipeline scripts)
try:
    from env_config import get_env as _get_env
    _ENV = _get_env()
except Exception:
    _ENV = None

if _ENV is not None:
    DATA_DIR   = str(_ENV["processed"] / "histdata")
    OUTPUT_DIR = str(_ENV["engine"] / "backtest_results")
    _REPO_ROOT = _ENV["repo"]
else:
    # Fallback if env_config import fails — derive from this file's location
    _engine_path = Path(_ENGINE_DIR).resolve()
    _REPO_ROOT   = next(
        (p for p in [_engine_path, *_engine_path.parents] if (p / ".git").exists()),
        _engine_path.parent.parent,
    )
    DATA_DIR   = os.path.join(_ENGINE_DIR, "..", "..", "processed_data", "histdata")
    OUTPUT_DIR = os.path.join(_ENGINE_DIR, "backtest_results")

CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "robustness_checkpoint.json")
SUMMARY_PATH    = os.path.join(OUTPUT_DIR, "robustness_summary.txt")

# ─── GitHub push config ───────────────────────────────────────────────────────
GITHUB_TOKEN  = os.getenv("GITHUB_TOKEN",  "")
GITHUB_REPO   = os.getenv("GITHUB_REPO",   "AnalystTKZ/Multi-Bot")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
GITHUB_USER   = os.getenv("GITHUB_USER",   "Kaggle Robustness Bot")
GITHUB_EMAIL  = os.getenv("GITHUB_EMAIL",  "bot@kaggle.local")

# ─── Symbols (NEVER seen during training) ─────────────────────────────────────
ROBUSTNESS_SYMBOLS = ["USDZAR", "USDMXN", "EURAUD", "GBPCHF", "AUDJPY"]

# ─── Epochs — mirror pipeline/step5_split.py calendar split ───────────────────
def _load_epochs() -> dict:
    """
    Read epoch windows from split_summary.json so robustness epochs align exactly
    with the in-sample train/val/test split.
      Epoch 1 → train era   (generalisation check during training period)
      Epoch 2 → val era     (Round 1 window — safe to iterate)
      Epoch 3 → test era    (blind window — use sparingly)
    Falls back to hardcoded dates if the file is missing.
    """
    _candidates = [
        Path(_ENGINE_DIR) / ".." / "ml_training" / "datasets" / "split_summary.json",
    ]
    if _ENV is not None:
        _ml = _ENV.get("ml_training")
        if _ml:
            _candidates.insert(0, Path(_ml) / "datasets" / "split_summary.json")

    for _p in _candidates:
        try:
            _p = Path(_p).resolve()
            if not _p.exists():
                continue
            with open(_p) as _f:
                _s = json.load(_f)
            _dr = _s["date_ranges"]
            epochs = {
                1: (
                    _dr["train"]["start"][:10],
                    _dr["train"]["end"][:10],
                    f"Train era ({_dr['train']['start'][:4]}–{_dr['train']['end'][:4]})",
                ),
                2: (
                    _dr["validation"]["start"][:10],
                    _dr["validation"]["end"][:10],
                    f"Val era / Round 1 ({_dr['validation']['start'][:4]}–{_dr['validation']['end'][:4]})",
                ),
                3: (
                    _dr["test"]["start"][:10],
                    _dr["test"]["end"][:10],
                    f"Test era / Blind ({_dr['test']['start'][:4]}–{_dr['test']['end'][:4]})",
                ),
            }
            logger.info(
                "Epochs from split_summary — E1: %s→%s  E2: %s→%s  E3: %s→%s",
                epochs[1][0], epochs[1][1],
                epochs[2][0], epochs[2][1],
                epochs[3][0], epochs[3][1],
            )
            return epochs
        except Exception as _exc:
            logger.warning("split_summary.json load failed (%s): %s — using fallback", _p, _exc)

    logger.warning("split_summary.json not found — using hardcoded epoch dates")
    return {
        1: ("2016-01-01", "2022-12-31", "Pre-val era (2016-2022)"),
        2: ("2023-01-01", "2024-08-27", "Val era / Round 1 (2023-2024)"),
        3: ("2024-08-27", "2026-12-31", "Test era / Blind (2024-2026)"),
    }


EPOCHS = _load_epochs()


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def _checkpoint_key(symbol: str, epoch: int) -> str:
    return f"{symbol}_e{epoch}"


def _load_checkpoint() -> dict:
    """Load existing checkpoint. Returns dict of completed results keyed by symbol_eN."""
    if not os.path.exists(CHECKPOINT_PATH):
        return {}
    try:
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        completed = data.get("completed", {})
        logger.info(
            "Checkpoint loaded: %d/%d combinations already done: %s",
            len(completed), len(ROBUSTNESS_SYMBOLS) * len(EPOCHS),
            list(completed.keys()),
        )
        return completed
    except Exception as exc:
        logger.warning("Checkpoint load failed (%s) — starting fresh", exc)
        return {}


def _save_checkpoint(completed: dict) -> None:
    """Persist current completed results to checkpoint file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    payload = {
        "saved_at":  datetime.now(timezone.utc).isoformat(),
        "completed": completed,
    }
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, CHECKPOINT_PATH)  # atomic write
    logger.info("Checkpoint saved (%d combinations done)", len(completed))


# ─── GitHub push ──────────────────────────────────────────────────────────────

def _git(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), check=check, capture_output=True, text=True)


def _push_checkpoint_to_github(label: str) -> None:
    """Push backtest_results/ to GitHub.

    OUTPUT_DIR already points into the remote git clone (via env_config) so no
    file-copy step is needed — just git-add + commit + push.
    """
    if not GITHUB_TOKEN:
        logger.warning("GITHUB_TOKEN not set — skipping GitHub push after %s", label)
        return
    if not _REPO_ROOT.exists():
        logger.warning("Repo clone not found at %s — skipping push", _REPO_ROOT)
        return

    try:
        remote_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"
        _git(["git", "config", "user.name",  GITHUB_USER],  _REPO_ROOT)
        _git(["git", "config", "user.email", GITHUB_EMAIL], _REPO_ROOT)
        _git(["git", "remote", "set-url", "origin", remote_url], _REPO_ROOT)
        _git(["git", "pull", "--ff-only", "origin", GITHUB_BRANCH], _REPO_ROOT, check=False)

        # Files are written directly into the clone — just stage the directory.
        # Fallback copy: if OUTPUT_DIR is NOT inside the clone (env_config unavailable),
        # copy robustness* files into the expected clone location.
        _clone_results = _REPO_ROOT / "trading-system" / "trading-engine" / "backtest_results"
        _src = Path(OUTPUT_DIR)
        if _src.resolve() != _clone_results.resolve() and _clone_results != _src:
            _clone_results.mkdir(parents=True, exist_ok=True)
            for f in _src.glob("robustness*"):
                shutil.copy2(f, _clone_results / f.name)

        _git(["git", "add", "trading-system/trading-engine/backtest_results/"], _REPO_ROOT)

        status = _git(["git", "status", "--porcelain"], _REPO_ROOT)
        if not status.stdout.strip():
            logger.info("GitHub push: nothing changed after %s", label)
            return

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        _git(["git", "commit", "-m",
              f"chore(robustness): checkpoint after {label} — {ts}"], _REPO_ROOT)
        _git(["git", "push", "origin", f"HEAD:{GITHUB_BRANCH}"], _REPO_ROOT)
        logger.info("GitHub push: checkpoint pushed after %s", label)

    except subprocess.CalledProcessError as exc:
        logger.error("GitHub push failed after %s: %s\n%s", label, exc, exc.stderr)
    except Exception as exc:
        logger.error("GitHub push error after %s: %s", label, exc)


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_parquet(symbol: str, tf: str, start: str, end: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{symbol}_{tf}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No parquet for {symbol} {tf}: {path}")
    warmup = _BACKTEST_WARMUP.get(tf.upper())
    start_ts = pd.Timestamp(start, tz="UTC")
    if warmup is not None:
        start_ts = start_ts - warmup
    end_ts   = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)
    # Push the date filter into the parquet read — only the epoch window is
    # loaded into RAM, not the full 9-year file.
    try:
        import pyarrow.parquet as _pq
        import pyarrow.dataset as _ds
        _tbl = _pq.read_table(
            path,
            filters=[
                ("index", ">=", start_ts),
                ("index", "<",  end_ts),
            ],
        )
        df = _tbl.to_pandas()
        del _tbl
    except Exception:
        # Fallback: full read + slice (pyarrow filter may fail on non-standard index column name)
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.loc[(df.index >= start_ts) & (df.index < end_ts)]

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.columns = [c.lower() for c in df.columns]
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """ATR-14, range/pullback flags — mirrors run_backtest._compute_indicators."""
    hi = df["high"].values
    lo = df["low"].values
    cl = df["close"].values
    n  = len(df)

    prev_cl    = np.empty(n, dtype=np.float64)
    prev_cl[0] = cl[0]
    prev_cl[1:] = cl[:-1]
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_cl), np.abs(lo - prev_cl)))

    atr       = np.empty(n, dtype=np.float64)
    atr[:14]  = tr[:14].mean()
    alpha     = 1.0 / 14.0
    for i in range(14, n):
        atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha

    df = df.copy()
    df["atr_14"] = atr

    _window = 20
    roll_hi = df["high"].rolling(_window, min_periods=1).max()
    roll_lo = df["low"].rolling(_window, min_periods=1).min()
    df["range_resist"]    = roll_hi
    df["range_support"]   = roll_lo
    df["range_width_atr"] = (roll_hi - roll_lo) / (df["atr_14"] + 1e-9)

    at_top = (roll_hi - cl) / (df["atr_14"] + 1e-9) < 0.5
    at_bot = (cl - roll_lo) / (df["atr_14"] + 1e-9) < 0.5
    df["range_valid"] = (at_top | at_bot) & (df["range_width_atr"] > 1.5)
    df["range_side"]  = np.where(at_bot, "buy", np.where(at_top, "sell", ""))

    ema     = df["close"].ewm(span=21, adjust=False).mean()
    near    = (df["close"] - ema).abs() / (df["atr_14"] + 1e-9) < 0.4
    up      = df["close"] > ema
    df["pullback_valid"] = near
    df["pullback_side"]  = np.where(near & up, "buy", np.where(near & ~up, "sell", ""))
    df["pullback_level"] = ema.values
    return df


def _load_htf(symbol: str, start: str, end: str) -> dict:
    htf = {}
    for tf in ("5M", "1H", "4H", "1D"):
        try:
            htf[tf] = _load_parquet(symbol, tf, start, end)
        except FileNotFoundError:
            logger.warning("Missing HTF %s %s — will use None", symbol, tf)
    return htf


def _ml_cache_entry(cache: dict[str, np.ndarray] | None, idx: int) -> dict:
    if not cache:
        return {}
    out: dict = {}
    p_bull = float(cache["p_bull"][idx])
    if not np.isnan(p_bull):
        out["p_bull"] = p_bull
        out["p_bear"] = float(cache["p_bear"][idx])
        out["expected_variance"] = float(cache["expected_variance"][idx])
    regime_id = int(cache["regime"][idx])
    if regime_id >= 0:
        out["regime"] = str(_HTF_REGIME_NAMES[regime_id])
    regime_ltf_id = int(cache["regime_ltf"][idx])
    if regime_ltf_id >= 0:
        out["regime_ltf"] = str(_LTF_REGIME_NAMES[regime_ltf_id])
    return out


# ─── ML cache (mirrors _precompute_ml_cache from run_backtest.py) ─────────────

def _build_ml_cache(
    df: pd.DataFrame,
    symbol: str,
    htf: dict,
    ml_models: dict,
    batch_size: int = 1024,
) -> dict[str, np.ndarray]:
    gru_model = ml_models.get("gru")
    regime_4h = ml_models.get("regime_htf") or ml_models.get("regime_4h")
    regime_1h = ml_models.get("regime_ltf") or ml_models.get("regime_1h")

    if not (gru_model or regime_4h or regime_1h):
        return {}

    try:
        import torch
        from services.feature_engine import FeatureEngine, SEQUENCE_FEATURES

        fe = FeatureEngine()
        n  = len(df)

        _regime_htf_series = _regime_htf_conf = _regime_ltf_series = _regime_ltf_conf = None

        # Build feature matrix once per source df (serially), slice per model.
        # predict_batch (GPU/DataParallel) runs serially — one model at a time.
        from models.regime_classifier import RegimeClassifier as _RC

        _do_4h = bool(regime_4h and getattr(regime_4h, "is_trained", False)
                      and regime_4h._model is not None)
        _do_1h = bool(regime_1h and getattr(regime_1h, "is_trained", False)
                      and regime_1h._model is not None)

        _df_src_4h = htf.get("4H") if (_do_4h and htf.get("4H") is not None
                                        and len(htf.get("4H")) >= 50) else df
        _df_src_1h = htf.get("1H") if (_do_1h and htf.get("1H") is not None
                                        and len(htf.get("1H")) >= 50) else df

        _X_4h = _X_1h = None
        try:
            if _do_4h:
                _X_4h = _RC._build_feature_matrix(_df_src_4h, htf, symbol)
        except Exception as _exc:
            logger.error("regime 4H feature build failed %s: %s", symbol, _exc)
        try:
            if _do_1h:
                if _do_4h and _df_src_1h is _df_src_4h and _X_4h is not None:
                    _X_1h = _X_4h
                else:
                    _X_1h = _RC._build_feature_matrix(_df_src_1h, htf, symbol)
        except Exception as _exc:
            logger.error("regime 1H feature build failed %s: %s", symbol, _exc)

        # Phase 2: GPU inference (serial — one model at a time on CUDA).
        def _infer_regime(rc_model, X_feat, df_src):
            _mode = getattr(rc_model, "_mode", "ltf_behaviour")
            if _mode == "htf_bias":
                _classes, _default = ["BIAS_UP", "BIAS_DOWN", "BIAS_NEUTRAL"], 2
            else:
                _classes, _default = ["TRENDING", "RANGING", "CONSOLIDATING", "VOLATILE"], 1
            _n    = len(df_src)
            _step = max(1, _n // 20_000)
            try:
                row_idx = list(range(50, _n, _step))
                X = X_feat[row_idx]
                if len(X) == 0:
                    return None, None, {}
                ids, conf = rc_model.predict_batch(X); del X
                arr = np.full(_n, -1, dtype=np.int8)
                arr[row_idx] = ids.astype(np.int8)
                filled = (pd.Series(arr.astype(float))
                          .replace(-1, np.nan).ffill().fillna(_default).astype(int).values)
                c_arr = np.full(_n, np.nan, dtype=np.float32)
                c_arr[row_idx] = conf
                filled_conf = pd.Series(c_arr, index=df_src.index).ffill()
                return pd.Series(filled, index=df_src.index, dtype=int), filled_conf
            except Exception as exc:
                logger.error("regime infer failed %s: %s", symbol, exc)
                return None, None

        if _do_4h and _X_4h is not None:
            _r4h, _c4h = _infer_regime(regime_4h, _X_4h, _df_src_4h)
            del _X_4h
            if _r4h is not None:
                if _df_src_4h is not df:
                    _regime_htf_series = _r4h.reindex(df.index, method="ffill").fillna(2).astype(int)
                    _regime_htf_conf   = _c4h.reindex(df.index, method="ffill").fillna(1/3)
                else:
                    _regime_htf_series, _regime_htf_conf = _r4h, _c4h
            gc.collect()

        if _do_1h and _X_1h is not None:
            _r1h, _c1h = _infer_regime(regime_1h, _X_1h, _df_src_1h)
            del _X_1h
            if _r1h is not None:
                if _df_src_1h is not df:
                    _regime_ltf_series = _r1h.reindex(df.index, method="ffill").fillna(1).astype(int)
                    _regime_ltf_conf   = _c1h.reindex(df.index, method="ffill").fillna(0.25)
                else:
                    _regime_ltf_series, _regime_ltf_conf = _r1h, _c1h
            gc.collect()

        logger.info("Building sequence features %s (%d bars)...", symbol, n)
        feat_df = fe._build_sequence_df(
            df, htf, symbol=symbol,
            regime_4h_series=_regime_htf_series,
            regime_4h_conf_series=_regime_htf_conf,
            regime_1h_series=_regime_ltf_series,
            regime_1h_conf_series=_regime_ltf_conf,
        )
        seq_arr = feat_df[SEQUENCE_FEATURES].to_numpy(dtype=np.float32, copy=False)
        seq_arr = np.nan_to_num(seq_arr, nan=0.0, posinf=0.0, neginf=0.0)
        del feat_df

        cache = {
            "p_bull": np.full(n, np.nan, dtype=np.float32),
            "p_bear": np.full(n, np.nan, dtype=np.float32),
            "expected_variance": np.full(n, np.nan, dtype=np.float32),
            "regime": np.full(n, -1, dtype=np.int8),
            "regime_ltf": np.full(n, -1, dtype=np.int8),
        }
        if gru_model and getattr(gru_model, "is_trained", False) and gru_model._model is not None:
            from models.gru_lstm_predictor import DEVICE, SEQUENCE_LENGTH as SEQ_LEN
            n_valid = n - SEQ_LEN + 1
            n_feat  = seq_arr.shape[1]
            if n_valid > 0:
                m  = gru_model._model.module if hasattr(gru_model._model, "module") else gru_model._model
                m.eval()
                all_p_bull  = np.empty(n_valid, dtype=np.float32)
                all_mag     = np.empty(n_valid, dtype=np.float32)
                all_log_var = np.empty(n_valid, dtype=np.float32)
                _T = getattr(gru_model, "_temperature", 1.0)
                with torch.no_grad():
                    for b_start in range(0, n_valid, batch_size):
                        b_end     = min(b_start + batch_size, n_valid)
                        batch_raw = np.lib.stride_tricks.sliding_window_view(
                            seq_arr[b_start: b_end + SEQ_LEN - 1], (SEQ_LEN, n_feat)
                        ).reshape(b_end - b_start, SEQ_LEN, n_feat)
                        xb = torch.from_numpy(batch_raw.copy()).to(DEVICE)
                        with torch.amp.autocast("cuda", enabled=(str(DEVICE) == "cuda")):
                            dl, mp, lv = m(xb)
                        all_p_bull[b_start:b_end]  = torch.sigmoid(dl / _T).cpu().numpy()
                        all_mag[b_start:b_end]     = torch.relu(mp).cpu().numpy()
                        all_log_var[b_start:b_end] = lv.cpu().numpy()
                        del xb, dl, mp, lv, batch_raw

                var_vals   = np.log1p(np.exp(all_log_var)) + 1e-6
                p_bear_arr = np.clip(1.0 - all_p_bull, 0.0, 1.0)
                entry_dep  = np.clip(all_mag * 100.0, 0.0, 1.0)
                bar_idxs   = np.arange(n_valid, dtype=np.int32) + (SEQ_LEN - 1)
                cache["p_bull"][bar_idxs] = all_p_bull
                cache["p_bear"][bar_idxs] = p_bear_arr
                cache["expected_variance"][bar_idxs] = var_vals.astype(np.float32, copy=False)
                del all_p_bull, all_mag, all_log_var, var_vals, p_bear_arr, entry_dep
                logger.info("GRU done %s (%d bars)", symbol, n_valid)
        del seq_arr
        gc.collect()

        if _regime_ltf_series is not None:
            cache["regime_ltf"] = np.clip(_regime_ltf_series.values, 0, 3).astype(np.int8, copy=False)
        if _regime_htf_series is not None:
            cache["regime"] = np.clip(_regime_htf_series.values, 0, 2).astype(np.int8, copy=False)

        logger.info(
            "ML cache done %s: n=%d gru=%d htf=%d ltf=%d",
            symbol,
            n,
            int(np.count_nonzero(~np.isnan(cache["p_bull"]))),
            int(np.count_nonzero(cache["regime"] >= 0)),
            int(np.count_nonzero(cache["regime_ltf"] >= 0)),
        )
        gc.collect()
        return cache

    except Exception as exc:
        logger.error("_build_ml_cache failed %s: %s", symbol, exc)
        gc.collect()
        raise


# ─── Signal generation (exact copy of _compute_backtest_signal) ───────────────

# Signal config — resolved once at module level (env vars don't change mid-run).
_SCFG = {
    "max_uncertainty": float(os.getenv("MAX_UNCERTAINTY",          "2.0")),
    "dir_thresh":      float(os.getenv("ML_DIRECTION_THRESHOLD",   "0.58")),
    "neutral_thresh":  float(os.getenv("NEUTRAL_BIAS_THRESHOLD",   "0.58")),
    "vol_thr":         float(os.getenv("VOLATILE_ENTRY_THRESHOLD",
                                       os.getenv("ML_DIRECTION_THRESHOLD", "0.58"))),
    "block_consol":    os.getenv("BLOCK_LTF_CONSOLIDATING", "0").lower() in ("1","true","yes"),
    "sl_mult":         float(os.getenv("SL_ATR_MULT", "1.5")),
    "rr":              float(os.getenv("RR_DEFAULT",  "2.0")),
}


def _compute_signal(symbol: str, ml_preds: dict, bar: dict) -> dict | None:
    close = float(bar["close"])
    atr   = float(bar.get("atr_14", close * 0.001))
    if atr < 1e-9 or not ml_preds:
        return None

    if float(ml_preds.get("expected_variance", 0.0)) > _SCFG["max_uncertainty"]:
        return None

    p_bull      = float(ml_preds.get("p_bull", 0.5))
    p_bear      = float(ml_preds.get("p_bear", 0.5))
    _dir_thresh = _SCFG["dir_thresh"]

    if   p_bull >= p_bear and p_bull >= _dir_thresh: side, conf = "buy",  p_bull
    elif p_bear >  p_bull and p_bear >= _dir_thresh: side, conf = "sell", p_bear
    else: return None

    _htf_bias = str(ml_preds.get("regime", "BIAS_NEUTRAL"))
    _neutral  = _SCFG["neutral_thresh"]
    if _htf_bias == "BIAS_UP"      and side == "sell":    return None
    if _htf_bias == "BIAS_DOWN"    and side == "buy":     return None
    if _htf_bias == "BIAS_NEUTRAL" and conf < _neutral:   return None

    _ltf     = str(ml_preds.get("regime_ltf", "TRENDING"))
    _vol_thr = _SCFG["vol_thr"]
    _rv      = bool(bar.get("range_valid",    False))
    _rs      = str(bar.get("range_side",     ""))
    _pv      = bool(bar.get("pullback_valid", False))
    _ps      = str(bar.get("pullback_side",  ""))

    if _ltf == "CONSOLIDATING" and _SCFG["block_consol"]:  return None
    if _ltf == "VOLATILE"  and conf < _vol_thr:             return None
    if _ltf == "TRENDING"  and _pv and _ps and _ps != side: return None
    if _ltf == "RANGING"   and _rv and _rs and _rs != side: return None

    _sl_mult = _SCFG["sl_mult"]
    _rr      = _SCFG["rr"]
    sl_dist  = atr * _sl_mult

    if _ltf == "RANGING" and _rv:
        if side == "buy":
            sl  = float(bar.get("range_support", close - sl_dist)) - atr * 0.3
            tp  = float(bar.get("range_resist",  close + sl_dist * _rr))
        else:
            sl  = float(bar.get("range_resist",  close + sl_dist)) + atr * 0.3
            tp  = float(bar.get("range_support", close - sl_dist * _rr))
        if abs(tp - close) / (abs(close - sl) + 1e-9) < 1.5:
            sl = (close - sl_dist) if side == "buy" else (close + sl_dist)
            tp = (close + sl_dist * _rr) if side == "buy" else (close - sl_dist * _rr)
    else:
        sl = (close - sl_dist) if side == "buy" else (close + sl_dist)
        tp = (close + sl_dist * _rr) if side == "buy" else (close - sl_dist * _rr)

    return {
        "side": side, "entry": close, "stop_loss": sl, "take_profit": tp,
        "confidence": round(float(conf), 3), "trader_id": "ml_trader", "symbol": symbol,
        "signal_metadata": {
            "regime": _htf_bias, "regime_ltf": _ltf,
            "expected_variance": float(ml_preds.get("expected_variance", 0.0)),
            "p_bull": p_bull, "p_bear": p_bear, "atr_at_entry": atr,
        },
    }


# ─── Trade simulation ─────────────────────────────────────────────────────────

def _simulate_trade(
    df: pd.DataFrame, entry_i: int, side: str,
    entry: float, sl: float, tp1: float, tp2: float,
    size: float, atr: float,
) -> dict:
    """Vectorised TP1/TP2 two-phase simulation — no per-bar Python loop."""
    is_buy    = (side == "buy")
    direction = 1 if is_buy else -1
    end_i     = min(entry_i + MAX_HOLD_BARS, len(df))
    highs  = df["high"].values[entry_i + 1 : end_i].astype(np.float64)
    lows   = df["low"].values[entry_i + 1 : end_i].astype(np.float64)
    closes = df["close"].values[entry_i + 1 : end_i].astype(np.float64)
    n      = len(highs)

    if n == 0:
        return {"pnl": 0.0, "exit_reason": "timeout", "bars_held": 0,
                "exit_price": entry, "phase1_pnl": 0.0, "phase2_pnl": 0.0}

    # Phase 1: find first SL or TP1 hit
    sl_hits  = np.where(lows <= sl)[0]  if is_buy else np.where(highs >= sl)[0]
    tp1_hits = np.where(highs >= tp1)[0] if is_buy else np.where(lows <= tp1)[0]
    first_sl  = int(sl_hits[0])  if len(sl_hits)  else n
    first_tp1 = int(tp1_hits[0]) if len(tp1_hits) else n

    if first_sl <= first_tp1:
        pnl = direction * (sl - entry) * size - COMMISSION_PCT * abs(entry) * size
        return {"pnl": pnl, "exit_reason": "sl", "bars_held": first_sl + 1,
                "exit_price": sl, "phase1_pnl": pnl, "phase2_pnl": 0.0}

    if first_tp1 >= n:
        # Neither hit — time exit
        last_close = float(closes[-1])
        pnl = direction * (last_close - entry) * size - COMMISSION_PCT * abs(entry) * size
        return {"pnl": pnl, "exit_reason": "timeout", "bars_held": n,
                "exit_price": last_close, "phase1_pnl": pnl, "phase2_pnl": 0.0}

    # TP1 hit — Phase 2
    phase1_pnl = direction * (tp1 - entry) * size * 0.5
    be_stop    = entry + direction * atr * 0.1

    p2_start = first_tp1 + 1
    p2_highs = highs[p2_start:]
    p2_lows  = lows[p2_start:]
    p2_closes = closes[p2_start:]
    n2 = len(p2_highs)

    if n2 == 0:
        exit_price = float(closes[first_tp1])
        phase2_pnl = direction * (exit_price - entry) * size * 0.5
        pnl = phase1_pnl + phase2_pnl - COMMISSION_PCT * abs(entry) * size
        return {"pnl": pnl, "exit_reason": "timeout", "bars_held": first_tp1 + 1,
                "exit_price": exit_price, "phase1_pnl": phase1_pnl, "phase2_pnl": phase2_pnl}

    be_hits  = np.where(p2_lows <= be_stop)[0]  if is_buy else np.where(p2_highs >= be_stop)[0]
    tp2_hits = np.where(p2_highs >= tp2)[0]     if is_buy else np.where(p2_lows <= tp2)[0]
    first_be  = int(be_hits[0])  if len(be_hits)  else n2
    first_tp2 = int(tp2_hits[0]) if len(tp2_hits) else n2

    if first_be < first_tp2 and first_be < n2:
        phase2_pnl = direction * (be_stop - entry) * size * 0.5
        pnl = phase1_pnl + phase2_pnl - COMMISSION_PCT * abs(entry) * size
        return {"pnl": pnl, "exit_reason": "be_or_trail",
                "bars_held": first_tp1 + 1 + first_be + 1,
                "exit_price": be_stop, "phase1_pnl": phase1_pnl, "phase2_pnl": phase2_pnl}

    if first_tp2 < n2:
        phase2_pnl = direction * (tp2 - entry) * size * 0.5
        pnl = phase1_pnl + phase2_pnl - COMMISSION_PCT * abs(entry) * size
        return {"pnl": pnl, "exit_reason": "tp2",
                "bars_held": first_tp1 + 1 + first_tp2 + 1,
                "exit_price": tp2, "phase1_pnl": phase1_pnl, "phase2_pnl": phase2_pnl}

    last_close = float(p2_closes[-1])
    phase2_pnl = direction * (last_close - entry) * size * 0.5
    pnl = phase1_pnl + phase2_pnl - COMMISSION_PCT * abs(entry) * size
    return {"pnl": pnl, "exit_reason": "timeout",
            "bars_held": first_tp1 + 1 + n2,
            "exit_price": last_close, "phase1_pnl": phase1_pnl, "phase2_pnl": phase2_pnl}


def _enrich_signal_no_pm(raw_signal: dict, equity: float) -> dict | None:
    entry = float(raw_signal["entry"])
    sl = float(raw_signal["stop_loss"])
    tp = float(raw_signal["take_profit"])
    sl_dist = abs(entry - sl)
    if sl_dist < 1e-9:
        return None
    risk_amount = max(equity * RISK_PER_TRADE, 1e-6)
    size = max(round(risk_amount / sl_dist, 2), 0.01)
    rr_ratio = abs(tp - entry) / (sl_dist + 1e-9)
    return {
        **raw_signal,
        "tp1": round(tp, 6),
        "tp2": round(tp, 6),
        "take_profit": round(tp, 6),
        "rr_ratio": round(rr_ratio, 2),
        "rr_to_tp2": round(rr_ratio, 2),
        "size": size,
        "size_full": size,
        "portfolio_manager": {"disabled": True},
    }


# ─── Per-symbol epoch backtest ────────────────────────────────────────────────

def _run_symbol_epoch(
    symbol: str, epoch_num: int, start: str, end: str,
    ml_models: dict,
    bundle: dict | None = None,
) -> dict:
    logger.info("=== %s Epoch %d (%s → %s) ===", symbol, epoch_num, start, end)
    pm = None
    if not _PM_DISABLED:
        from monitors.portfolio_manager import PortfolioManager
        pm = PortfolioManager(_PM_SETTINGS)

    try:
        if bundle is None:
            df_full = _load_parquet(symbol, "15M", start, end)
            if len(df_full) < 200:
                logger.warning("Not enough bars %s epoch %d (%d bars)", symbol, epoch_num, len(df_full))
                return {"symbol": symbol, "epoch": epoch_num, "trades": 0, "note": "insufficient_data", "trade_log": []}
            df_full = _compute_indicators(df_full)
            htf = _load_htf(symbol, start, end)
            ml_cache_full: dict[str, np.ndarray] = {}
            if ml_models:
                try:
                    ml_cache_full = _build_ml_cache(df_full, symbol, htf, ml_models)
                except Exception as exc:
                    logger.error("ML cache failed %s epoch %d: %s", symbol, epoch_num, exc)
        else:
            df_full = bundle["df"]
            ml_cache_full = bundle.get("ml_cache", {})
    except FileNotFoundError as exc:
        logger.error("Data missing for %s: %s", symbol, exc)
        return {"symbol": symbol, "epoch": epoch_num, "error": str(exc), "trades": 0, "trade_log": []}

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    warmup_start = start_ts - _BACKTEST_WARMUP["15M"]
    slice_start = int(df_full.index.searchsorted(warmup_start, side="left"))
    slice_end = int(df_full.index.searchsorted(end_ts, side="left"))
    df = df_full.iloc[slice_start:slice_end]
    if len(df) < 200:
        logger.warning("Not enough sliced bars %s epoch %d (%d bars)", symbol, epoch_num, len(df))
        return {"symbol": symbol, "epoch": epoch_num, "trades": 0, "note": "insufficient_data", "trade_log": []}

    ml_cache = {
        k: v[slice_start:slice_end]
        for k, v in (ml_cache_full or {}).items()
    }

    # Pre-extract bar arrays as numpy — eliminates df.iloc[i] per bar (~10-20 µs each).
    _rb_cols = ["close", "high", "low", "atr_14", "range_valid", "range_side",
                "range_support", "range_resist", "range_width_atr",
                "pullback_valid", "pullback_side", "pullback_level"]
    _rb_arrs: dict[str, np.ndarray | None] = {
        c: (df[c].to_numpy() if c in df.columns else None) for c in _rb_cols
    }
    _rb_ts = df.index.view(np.int64)  # nanoseconds

    def _rb_bar(idx: int) -> dict:
        _a = _rb_arrs
        def _s(col, default):
            arr = _a[col]
            if arr is None:
                return default
            v = arr[idx]
            if col in ("range_side", "pullback_side"):
                return str(v) if v == v else ""
            if col in ("range_valid", "pullback_valid"):
                return bool(v == 1)
            return float(v) if v == v else default
        return {
            "close":          _s("close",          0.0),
            "high":           _s("high",           0.0),
            "low":            _s("low",            0.0),
            "atr_14":         _s("atr_14",         0.001),
            "range_valid":    _s("range_valid",    False),
            "range_side":     _s("range_side",     ""),
            "range_support":  _s("range_support",  0.0),
            "range_resist":   _s("range_resist",   0.0),
            "range_width_atr": _s("range_width_atr", 0.0),
            "pullback_valid": _s("pullback_valid", False),
            "pullback_side":  _s("pullback_side",  ""),
            "pullback_level": _s("pullback_level", float("nan")),
        }

    trades:      list[dict] = []
    equity       = INITIAL_CAPITAL
    peak_equity  = INITIAL_CAPITAL
    daily_budget = INITIAL_CAPITAL * MAX_DAILY_LOSS_PCT
    daily_loss   = 0.0
    current_date = None
    daily_halt   = False
    last_trade_i = -COOLDOWN_BARS
    recent_bars: deque[int] = deque()
    halted       = False
    first_live_idx = max(200, int(df.index.searchsorted(start_ts, side="left")))

    _dir_thresh     = float(os.getenv("ML_DIRECTION_THRESHOLD", "0.58"))
    _density_lambda = float(os.getenv("DENSITY_LAMBDA", "0.12"))

    for i in range(first_live_idx, len(df)):
        if halted:
            break
        dt  = pd.Timestamp(_rb_ts[i], tz="UTC")

        dd = (peak_equity - equity) / (peak_equity + 1e-9)
        if dd >= MAX_DRAWDOWN_PCT:
            logger.warning("%s epoch %d: drawdown halt %.1f%%", symbol, epoch_num, dd * 100)
            halted = True
            break

        day_str = dt.strftime("%Y-%m-%d")
        if day_str != current_date:
            current_date = day_str
            daily_loss   = 0.0
            daily_halt   = False
            if pm is not None:
                pm.notify_date(day_str)

        if daily_halt or abs(daily_loss) >= daily_budget:
            daily_halt = True
            continue

        if not (7 <= dt.hour < 18):
            continue

        if i - last_trade_i < COOLDOWN_BARS:
            continue

        ml_preds   = _ml_cache_entry(ml_cache, i)
        bar        = _rb_bar(i)
        raw_signal = _compute_signal(symbol, ml_preds, bar)
        if raw_signal is None:
            continue

        while recent_bars and i - recent_bars[0] >= 96:
            recent_bars.popleft()
        raw_signal["confidence"] = float(raw_signal["confidence"]) * math.exp(
            -_density_lambda * len(recent_bars)
        )
        if raw_signal["confidence"] < _dir_thresh:
            continue
        if MIN_CONFIDENCE > 0 and raw_signal["confidence"] < MIN_CONFIDENCE:
            continue

        entry_raw = float(raw_signal["entry"])
        raw_signal["entry"] = (entry_raw * (1 + SLIPPAGE_PCT) if raw_signal["side"] == "buy"
                               else entry_raw * (1 - SLIPPAGE_PCT))
        atr = float(bar.get("atr_14", entry_raw * 0.001))

        if _PM_DISABLED:
            enriched = _enrich_signal_no_pm(raw_signal, equity)
            if enriched is None:
                continue
        else:
            enriched = pm.enrich_signal(raw_signal, {"equity": equity, "open_positions_detail": []}, atr=atr)
            if enriched is None:
                continue

        result = _simulate_trade(
            df, i, enriched["side"],
            float(enriched["entry"]), float(enriched["stop_loss"]),
            float(enriched["tp1"]),   float(enriched["tp2"]),
            float(enriched["size"]),  atr,
        )
        pnl = result["pnl"]
        equity      += pnl
        peak_equity  = max(peak_equity, equity)
        daily_loss  += min(0.0, pnl)
        last_trade_i = i
        recent_bars.append(i)

        trades.append({
            "symbol":       symbol,
            "epoch":        epoch_num,
            "entry_time":   dt.isoformat(),
            "side":         enriched["side"],
            "entry":        round(float(enriched["entry"]),      5),
            "exit":         round(result["exit_price"],          5),
            "sl":           round(float(enriched["stop_loss"]),  5),
            "tp1":          round(float(enriched["tp1"]),        5),
            "tp2":          round(float(enriched["tp2"]),        5),
            "size":         round(float(enriched["size"]),       4),
            "pnl":          round(pnl,                           4),
            "exit_reason":  result["exit_reason"],
            "bars_held":    result["bars_held"],
            "rr_ratio":     round(float(enriched["rr_ratio"]),   2),
            "confidence":   round(raw_signal["confidence"],      3),
            "regime":       ml_preds.get("regime",     "UNKNOWN"),
            "regime_ltf":   ml_preds.get("regime_ltf", "UNKNOWN"),
            "p_bull":       round(ml_preds.get("p_bull", 0.5),   3),
            "p_bear":       round(ml_preds.get("p_bear", 0.5),   3),
            "equity_after": round(equity,                        2),
        })

    return _compute_metrics(symbol, epoch_num, start, end, trades, equity, peak_equity)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _compute_metrics(
    symbol: str, epoch_num: int, start: str, end: str,
    trades: list[dict], final_equity: float, peak_equity: float,
) -> dict:
    n = len(trades)
    if n == 0:
        return {"symbol": symbol, "epoch": epoch_num, "start": start, "end": end,
                "trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "total_return_pct": 0.0, "max_drawdown_pct": 0.0,
                "sharpe": 0.0, "avg_rr": 0.0, "tp1_rate": 0.0, "tp2_rate": 0.0,
                "trade_log": []}

    pnls  = np.array([t["pnl"] for t in trades])
    wins  = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gp    = float(wins.sum())   if len(wins)   else 0.0
    gl    = float(losses.sum()) if len(losses) else 0.0

    eq    = np.empty(n + 1); eq[0] = INITIAL_CAPITAL
    for k, t in enumerate(trades): eq[k + 1] = t["equity_after"]
    rets  = np.diff(eq) / (eq[:-1] + 1e-9)
    sharpe = (rets.mean() / (rets.std() + 1e-9)) * math.sqrt(252 * 26)
    rm    = np.maximum.accumulate(eq)
    max_dd = float(((rm - eq) / (rm + 1e-9)).max())

    from collections import Counter
    return {
        "symbol":            symbol,
        "epoch":             epoch_num,
        "start":             start,
        "end":               end,
        "trades":            n,
        "win_rate":          round(len(wins) / n,                   4),
        "profit_factor":     round(gp / (abs(gl) + 1e-9),          3),
        "total_return_pct":  round((final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2),
        "max_drawdown_pct":  round(max_dd * 100,                    2),
        "sharpe":            round(float(sharpe),                   3),
        "avg_rr":            round(float(np.mean([t["rr_ratio"] for t in trades])), 3),
        "tp1_rate":          round(sum(1 for t in trades if t["exit_reason"] == "tp1") / n, 4),
        "tp2_rate":          round(sum(1 for t in trades if t["exit_reason"] == "tp2") / n, 4),
        "gross_profit":      round(gp,          2),
        "gross_loss":        round(gl,          2),
        "final_equity":      round(float(eq[-1]), 2),
        "htf_regime_dist":   dict(Counter(t["regime"]     for t in trades)),
        "ltf_regime_dist":   dict(Counter(t["regime_ltf"] for t in trades)),
        "side_distribution": dict(Counter(t["side"]       for t in trades)),
        "trade_log":         trades,
    }


# ─── Summary table ────────────────────────────────────────────────────────────

def _print_summary(all_results: list[dict]) -> str:
    hdr = (f"  {'Symbol':<10} {'Epoch':<6} {'Period':<30} {'Trades':>7} {'WR%':>6} "
           f"{'PF':>6} {'Return%':>8} {'MaxDD%':>8} {'Sharpe':>7} {'TP1%':>6} {'TP2%':>6}")
    lines = ["", "=" * 100,
             "  ROBUSTNESS BACKTEST — Out-of-sample symbols (model never trained on these)",
             "=" * 100, hdr, "-" * 100]

    for res in all_results:
        lbl = EPOCHS[res["epoch"]][2]
        if res.get("error") or res.get("trades", 0) == 0:
            lines.append(f"  {res['symbol']:<10} {res['epoch']:<6} {lbl:<30}  -- {res.get('error','no_trades')}")
            continue
        lines.append(
            f"  {res['symbol']:<10} {res['epoch']:<6} {lbl:<30} "
            f"{res['trades']:>7} {res['win_rate']*100:>5.1f}% {res['profit_factor']:>6.2f} "
            f"{res['total_return_pct']:>7.1f}% {res['max_drawdown_pct']:>7.1f}% "
            f"{res['sharpe']:>7.2f} {res['tp1_rate']*100:>5.1f}% {res['tp2_rate']*100:>5.1f}%"
        )

    lines.append("-" * 100)
    for ep in sorted(EPOCHS):
        ep_res = [r for r in all_results if r.get("epoch") == ep and r.get("trades", 0) > 0]
        if not ep_res: continue
        lines.append(
            f"  {'[AVG]':<10} {ep:<6} {EPOCHS[ep][2]:<30} "
            f"{sum(r['trades'] for r in ep_res):>7} "
            f"{sum(r['win_rate'] for r in ep_res)/len(ep_res)*100:>5.1f}% "
            f"{sum(r['profit_factor'] for r in ep_res)/len(ep_res):>6.2f} "
            f"{sum(r['total_return_pct'] for r in ep_res)/len(ep_res):>7.1f}% "
            f"{sum(r['max_drawdown_pct'] for r in ep_res)/len(ep_res):>7.1f}% "
            f"{sum(r['sharpe'] for r in ep_res)/len(ep_res):>7.2f}"
        )

    valid = [r for r in all_results if r.get("trades", 0) > 0]
    if valid:
        lines.append("=" * 100)
        lines.append(
            f"  {'OVERALL':<10} {'ALL':<6} {'All epochs / all symbols':<30} "
            f"{sum(r['trades'] for r in valid):>7} "
            f"{sum(r['win_rate'] for r in valid)/len(valid)*100:>5.1f}% "
            f"{sum(r['profit_factor'] for r in valid)/len(valid):>6.2f} "
            f"{sum(r['total_return_pct'] for r in valid)/len(valid):>7.1f}% "
            f"{sum(r['max_drawdown_pct'] for r in valid)/len(valid):>7.1f}% "
            f"{sum(r['sharpe'] for r in valid)/len(valid):>7.2f}"
        )
    lines += ["=" * 100, ""]
    text = "\n".join(lines)
    return text


# ─── Model loading ────────────────────────────────────────────────────────────

def _load_models() -> dict:
    models: dict = {}
    try:
        from models.regime_classifier import RegimeClassifier
        for tf, mode, key in [("4H", "htf_bias", "regime_htf"), ("1H", "ltf_behaviour", "regime_ltf")]:
            m = RegimeClassifier(timeframe=tf, mode=mode)
            if getattr(m, "is_trained", False):
                models[key] = m
                logger.info("[OK] RegimeClassifier %s loaded", key)
    except Exception as exc:
        logger.warning("[SKIP] RegimeClassifier: %s", exc)
    try:
        from models.quality_scorer import QualityScorer
        qs = QualityScorer()
        if getattr(qs, "is_trained", False):
            models["quality"] = qs
            logger.info("[OK] QualityScorer loaded")
    except Exception as exc:
        logger.warning("[SKIP] QualityScorer: %s", exc)
    try:
        from models.gru_lstm_predictor import GRULSTMPredictor
        gru = GRULSTMPredictor()
        if getattr(gru, "is_trained", False):
            models["gru"] = gru
            logger.info("[OK] GRULSTMPredictor loaded")
    except Exception as exc:
        logger.warning("[SKIP] GRULSTMPredictor: %s", exc)
    return models


def _prepare_symbol_bundle(symbol: str, start: str, end: str, ml_models: dict) -> dict:
    logger.info("Preparing shared robustness bundle for %s (%s → %s)", symbol, start, end)
    df = _load_parquet(symbol, "15M", start, end)
    if len(df) < 200:
        raise RuntimeError(f"{symbol}: insufficient data in shared bundle ({len(df)} bars)")
    df = _compute_indicators(df)
    htf = _load_htf(symbol, start, end)
    ml_cache: dict[str, np.ndarray] = {}
    if ml_models:
        ml_cache = _build_ml_cache(df, symbol, htf, ml_models)
    return {"df": df, "ml_cache": ml_cache}


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness backtest on unseen symbols")
    parser.add_argument("--epoch",  type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--symbol", type=str, choices=ROBUSTNESS_SYMBOLS, default=None)
    parser.add_argument("--no-ml",  action="store_true", help="Skip ML inference")
    parser.add_argument("--reset",  action="store_true", help="Clear checkpoint and start over")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.reset and os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        logger.info("Checkpoint cleared — starting fresh")

    epochs_to_run  = [args.epoch]  if args.epoch  else list(EPOCHS.keys())
    symbols_to_run = [args.symbol] if args.symbol else ROBUSTNESS_SYMBOLS

    logger.info("Robustness log: %s", _ROBUSTNESS_LOG_FILE)
    logger.info("Robustness backtest starting")
    logger.info("  Symbols : %s", symbols_to_run)
    logger.info("  Epochs  : %s", epochs_to_run)
    logger.info("  Data dir: %s", DATA_DIR)

    # Load checkpoint — skip already-done combinations
    completed = _load_checkpoint()

    ml_models = {} if args.no_ml else _load_models()
    if not ml_models:
        logger.warning("No ML models loaded — backtest will produce 0 signals")

    if _PM_DISABLED:
        logger.info("PortfolioManager disabled for robustness backtest")
    else:
        try:
            from monitors.portfolio_manager import PortfolioManager  # noqa: F401
            logger.info("[OK] PortfolioManager loaded")
        except Exception as exc:
            logger.error("PortfolioManager failed: %s", exc)
            sys.exit(1)

    shared_bundles: dict[str, dict] = {}
    overall_start = min(EPOCHS[e][0] for e in epochs_to_run)
    overall_end = max(EPOCHS[e][1] for e in epochs_to_run)
    _prep_workers = max(1, min(len(symbols_to_run), 4, int(os.getenv("ROBUSTNESS_PARALLEL_SYMBOL_PREP", "4"))))
    if _prep_workers == 1:
        for symbol in symbols_to_run:
            try:
                shared_bundles[symbol] = _prepare_symbol_bundle(symbol, overall_start, overall_end, ml_models)
            except Exception as exc:
                logger.error("Shared bundle prep failed for %s: %s", symbol, exc)
    else:
        logger.info("Preparing %d robustness symbol bundles with %d worker(s)", len(symbols_to_run), _prep_workers)
        with ThreadPoolExecutor(max_workers=_prep_workers) as ex:
            futures = {
                ex.submit(_prepare_symbol_bundle, symbol, overall_start, overall_end, ml_models): symbol
                for symbol in symbols_to_run
            }
            for fut in as_completed(futures):
                symbol = futures[fut]
                try:
                    shared_bundles[symbol] = fut.result()
                except Exception as exc:
                    logger.error("Shared bundle prep failed for %s: %s", symbol, exc)

    all_results: list[dict] = list(completed.values())  # seed with prior results

    for epoch_num in epochs_to_run:
        start, end, label = EPOCHS[epoch_num]
        logger.info("── Epoch %d: %s (%s → %s) ──", epoch_num, label, start, end)

        for symbol in symbols_to_run:
            key = _checkpoint_key(symbol, epoch_num)

            if key in completed:
                logger.info("  SKIP %s epoch %d (already in checkpoint)", symbol, epoch_num)
                continue

            bundle = shared_bundles.get(symbol)
            result = _run_symbol_epoch(symbol, epoch_num, start, end, ml_models, bundle=bundle)
            result_meta = {k: v for k, v in result.items() if k != "trade_log"}

            logger.info(
                "  %s epoch %d done: %d trades, WR=%.1f%%, PF=%.2f, Return=%.1f%%",
                symbol, epoch_num,
                result.get("trades", 0),
                result.get("win_rate", 0) * 100,
                result.get("profit_factor", 0),
                result.get("total_return_pct", 0),
            )

            # Save full result (with trade_log) into completed dict and checkpoint
            completed[key] = result
            all_results.append(result)
            _save_checkpoint(completed)

            # Save full JSON (all completed results so far)
            ts              = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            _latest_path    = os.path.join(OUTPUT_DIR, "robustness_latest.json")
            with open(_latest_path, "w") as f:
                json.dump({
                    "timestamp": ts,
                    "completed": len(completed),
                    "total":     len(epochs_to_run) * len(symbols_to_run),
                    "results":   all_results,
                }, f, indent=2, default=str)

            # Print running summary
            summary_meta = [{k: v for k, v in r.items() if k != "trade_log"} for r in all_results]
            summary_text = _print_summary(summary_meta)
            with open(SUMMARY_PATH, "w") as f:
                f.write(summary_text)

            # Push to GitHub after every completed combination
            _push_checkpoint_to_github(f"{symbol}_epoch{epoch_num}")

            gc.collect()

        gc.collect()

    # Final timestamped archive copy
    out_path = os.path.join(OUTPUT_DIR, "robustness_latest.json")
    ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    arc_path = os.path.join(OUTPUT_DIR, f"robustness_{ts}.json")
    if os.path.exists(out_path):
        shutil.copy2(out_path, arc_path)
        logger.info("Final results → %s", arc_path)
    else:
        logger.info("No new results written (all combinations were already checkpointed)")
    logger.info("Summary table → %s", SUMMARY_PATH)

    final_summary_meta = [{k: v for k, v in r.items() if k != "trade_log"} for r in all_results]
    if final_summary_meta:
        print(_print_summary(final_summary_meta))

    # Final push with the archived filename
    _push_checkpoint_to_github("FINAL")


if __name__ == "__main__":
    main()
