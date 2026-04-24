#!/usr/bin/env python3
"""
fit_temperature.py — Post-hoc temperature scaling for the GRU-LSTM direction head.

Usage:
    python scripts/fit_temperature.py
    python scripts/fit_temperature.py --symbols EURUSD GBPUSD --timeframe 15M
    python scripts/fit_temperature.py --n-bins 15

Loads trained GRU weights, runs a forward pass over the validation split to
collect raw direction logits, fits a scalar temperature T that minimises NLL
on that set, and saves temperature.pt alongside model.pt.

Prints before/after ECE (Expected Calibration Error) so the improvement can
be verified before the sidecar is used in inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

_env_config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "env_config.py"
)
sys.path.insert(0, os.path.dirname(os.path.abspath(_env_config_path)))
from env_config import get_env, ensure_output_dirs

_ENV = get_env()
ensure_output_dirs(_ENV)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fit_temperature")

# ── paths ─────────────────────────────────────────────────────────────────────
_PROCESSED_DIR = str(_ENV["processed"] / "histdata")
_WEIGHTS_DIR   = str(_ENV["weights"])
_SPLIT_PATH    = os.path.join(str(_ENV["ml_training"]), "datasets", "split_summary.json")

MAJOR_SYMBOLS = [
    "AUDUSD", "EURGBP", "EURJPY", "EURUSD", "GBPJPY",
    "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY", "XAUUSD",
]
DEFAULT_TIMEFRAME = "15M"


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_split_boundaries() -> dict:
    """Return train_end / val_end timestamps from split_summary.json, or {}."""
    import pandas as pd
    if not os.path.exists(_SPLIT_PATH):
        logger.warning("split_summary.json not found at %s — using last 20%% of data as val", _SPLIT_PATH)
        return {}
    try:
        with open(_SPLIT_PATH) as f:
            summary = json.load(f)
        ranges = summary.get("date_ranges", {})
        return {
            "train_end": pd.Timestamp(ranges["train"]["end"], tz="UTC"),
            "val_end":   pd.Timestamp(ranges["validation"]["end"], tz="UTC"),
        }
    except Exception as exc:
        logger.warning("Failed to parse split_summary.json: %s", exc)
        return {}


def _load_val_df(symbol: str, timeframe: str, bounds: dict):
    """Load OHLCV for the validation split of a symbol/timeframe."""
    import pandas as pd

    parquet_path = os.path.join(_PROCESSED_DIR, f"{symbol}_{timeframe.upper()}.parquet")
    if not os.path.exists(parquet_path):
        logger.warning("Missing parquet: %s", parquet_path)
        return None
    try:
        raw = pd.read_parquet(parquet_path)
        raw.index = pd.to_datetime(raw.index, utc=True, errors="coerce")
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
        raw = raw[keep].dropna(subset=["close"])
        raw = raw[raw.index.notna()].sort_index()
    except Exception as exc:
        logger.error("Failed to load %s: %s", parquet_path, exc)
        return None

    if bounds:
        df = raw[(raw.index > bounds["train_end"]) & (raw.index <= bounds["val_end"])]
    else:
        # Fallback: last 20% of available data
        n = len(raw)
        df = raw.iloc[int(n * 0.8):]

    if len(df) == 0:
        logger.warning("Val split for %s/%s is empty", symbol, timeframe)
        return None

    logger.info("Val split %s/%s: %d bars (%s → %s)",
                symbol, timeframe, len(df),
                df.index.min().date(), df.index.max().date())
    return df


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (equal-width bins).

    probs  : (N,) predicted P(bull=1)
    labels : (N,) binary ground truth (0 or 1)
    """
    probs  = np.asarray(probs,  dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    mask   = ~np.isnan(labels)
    probs, labels = probs[mask], labels[mask]

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n   = len(labels)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (probs >= lo) & (probs < hi)
        cnt = in_bin.sum()
        if cnt == 0:
            continue
        acc  = labels[in_bin].mean()
        conf = probs[in_bin].mean()
        ece += (cnt / n) * abs(acc - conf)
    return float(ece)


# ── main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    import torch

    # ── load model ────────────────────────────────────────────────────────────
    from models.gru_lstm_predictor import GRULSTMPredictor, DEVICE, SEQUENCE_LENGTH, WEIGHT_DIR
    from services.feature_engine import FeatureEngine, SEQUENCE_FEATURES

    predictor = GRULSTMPredictor()
    if not predictor.is_trained or predictor._model is None:
        logger.error(
            "GRU weights not found — run: python scripts/retrain_incremental.py --model gru"
        )
        sys.exit(1)

    m = predictor._model.module if hasattr(predictor._model, "module") else predictor._model
    m.eval()

    fe   = FeatureEngine()
    SEQ  = SEQUENCE_LENGTH
    n_feat = len(SEQUENCE_FEATURES)

    bounds = _load_split_boundaries()

    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    symbols    = args.symbols or MAJOR_SYMBOLS
    timeframe  = args.timeframe

    for sym in symbols:
        # HTF context slices — needed by feature engine for cross-TF features
        try:
            import pandas as pd
            htf_all = {
                tf: _load_val_df(sym, tf, {})  # full history for HTF context
                for tf in ("5M", "1H", "4H", "1D")
            }
        except Exception as exc:
            logger.warning("HTF load failed for %s: %s", sym, exc)
            htf_all = {}

        val_df = _load_val_df(sym, timeframe, bounds)
        if val_df is None or len(val_df) < SEQ + 1:
            logger.warning("Skipping %s/%s — insufficient val data", sym, timeframe)
            continue

        # Build feature array
        try:
            feat_df  = fe._build_sequence_df(val_df, htf_all, symbol=sym)
            feat_arr = feat_df[SEQUENCE_FEATURES].to_numpy(dtype=np.float32, copy=False)
            feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
            del feat_df
        except Exception as exc:
            logger.warning("Feature build failed for %s/%s: %s", sym, timeframe, exc)
            continue

        # Build direction labels for the val window
        try:
            label_df = predictor.create_labels(val_df)
            y_dir    = label_df["direction_up"].values.astype(np.float32)[SEQ:]
        except Exception as exc:
            logger.warning("Label build failed for %s/%s: %s", sym, timeframe, exc)
            continue

        n_valid = len(feat_arr) - SEQ
        n_valid = min(n_valid, len(y_dir))
        if n_valid <= 0:
            continue

        # Batched forward pass — collect raw logits only
        batch_size = args.batch_size
        seg_logits = np.empty(n_valid, dtype=np.float32)

        with torch.no_grad():
            for b_start in range(0, n_valid, batch_size):
                b_end    = min(b_start + batch_size, n_valid)
                seg_len  = b_end - b_start
                batch_raw = np.lib.stride_tricks.sliding_window_view(
                    feat_arr[b_start: b_start + seg_len + SEQ - 1],
                    (SEQ, n_feat),
                ).reshape(seg_len, SEQ, n_feat)
                xb = torch.from_numpy(batch_raw.copy()).to(DEVICE)
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    dl, _mp, _lv = m(xb)
                seg_logits[b_start:b_end] = dl.float().cpu().numpy()
                del xb, dl, _mp, _lv, batch_raw

        all_logits.append(seg_logits[:n_valid])
        all_labels.append(y_dir[:n_valid])
        logger.info("  Collected %d sequences from %s/%s", n_valid, sym, timeframe)

    if not all_logits:
        logger.error("No validation sequences collected — nothing to calibrate.")
        sys.exit(1)

    logits_all = np.concatenate(all_logits, axis=0)
    labels_all = np.concatenate(all_labels, axis=0)
    logger.info("Total calibration sequences: %d", len(logits_all))

    # ── ECE before temperature scaling ────────────────────────────────────────
    probs_uncal = 1.0 / (1.0 + np.exp(-logits_all.astype(np.float64)))
    ece_before  = _compute_ece(probs_uncal, labels_all, n_bins=args.n_bins)
    logger.info("ECE before temperature scaling: %.4f", ece_before)

    # ── fit temperature ───────────────────────────────────────────────────────
    T_opt = predictor.fit_temperature(logits_all, labels_all)

    # ── ECE after temperature scaling ─────────────────────────────────────────
    probs_cal  = 1.0 / (1.0 + np.exp(-logits_all.astype(np.float64) / T_opt))
    ece_after  = _compute_ece(probs_cal, labels_all, n_bins=args.n_bins)
    logger.info("ECE after  temperature scaling: %.4f", ece_after)

    print(
        f"\n=== Temperature Scaling Results ===\n"
        f"  Samples calibrated : {len(logits_all):,}\n"
        f"  Optimal temperature: {T_opt:.4f}\n"
        f"  ECE before         : {ece_before:.4f}\n"
        f"  ECE after          : {ece_after:.4f}\n"
        f"  ECE improvement    : {ece_before - ece_after:+.4f}\n"
        f"  Saved to           : {os.path.join(WEIGHT_DIR, 'temperature.pt')}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit temperature scaling for GRU direction head")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        metavar="SYM",
        help="Symbols to use for calibration (default: all major symbols)",
    )
    parser.add_argument(
        "--timeframe",
        default=DEFAULT_TIMEFRAME,
        metavar="TF",
        help=f"Timeframe for LTF sequences (default: {DEFAULT_TIMEFRAME})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        metavar="N",
        help="Forward-pass batch size (default: 512)",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        metavar="N",
        help="Number of bins for ECE computation (default: 10)",
    )
    main(parser.parse_args())
