#!/usr/bin/env python3
"""
robustness_backtest.py — Out-of-sample robustness test on symbols the model was NEVER trained on.

Tests the 5 HistData M1 currency pairs that are completely outside the 11-symbol training universe:
  USDZAR, USDMXN, EURAUD, GBPCHF, AUDJPY

Date range: 2016-01-01 → 2025-12-31 (9 years)
Split into 3 equal 3-year windows for temporal analysis:
  Epoch 1: 2016-01-01 → 2018-12-31  (pre-COVID, low vol environment)
  Epoch 2: 2019-01-01 → 2021-12-31  (COVID crash, recovery, extreme regimes)
  Epoch 3: 2022-01-01 → 2025-12-31  (rate hike cycle, EM stress, USDZAR highs)

Usage:
    cd trading-system/trading-engine
    python scripts/robustness_backtest.py
    python scripts/robustness_backtest.py --epoch 2          # single epoch
    python scripts/robustness_backtest.py --symbol USDZAR    # single symbol

Output:
    backtest_results/robustness_YYYYMMDD_HHMMSS.json
    backtest_results/robustness_summary.txt   (human-readable table)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

_ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _ENGINE_DIR)
os.chdir(_ENGINE_DIR)  # models use relative "weights/..." paths

# Allow model classes to load on CPU when no GPU is available.
# This must be set before any model module is imported (they call _get_device() at
# module level). INFERENCE_ONLY bypasses the Kaggle training guard — training still
# requires GPU via retrain_incremental.py which sets CUDA_VISIBLE_DEVICES explicitly.
os.environ.setdefault("INFERENCE_ONLY", "1")

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("robustness_bt")

# ─── Config (mirrors run_backtest.py exactly) ─────────────────────────────────
INITIAL_CAPITAL    = 10_000.0
RISK_PER_TRADE     = 0.01
CAPITAL_PER_TRADER = 0.20
COMMISSION_PCT     = 0.001
SLIPPAGE_PCT       = 0.0002
MAX_DAILY_LOSS_PCT = 0.02
MAX_DRAWDOWN_PCT   = 0.20
COOLDOWN_BARS      = 10
MAX_HOLD_BARS      = 200
MIN_CONFIDENCE     = float(os.getenv("MIN_CONFIDENCE", "0.0"))

_PM_SETTINGS = SimpleNamespace(
    ACCOUNT_BALANCE=INITIAL_CAPITAL,
    CAPITAL_PER_TRADER=CAPITAL_PER_TRADER,
    RISK_PER_TRADE=RISK_PER_TRADE,
    MAX_DAILY_LOSS_PCT=MAX_DAILY_LOSS_PCT,
)

DATA_DIR   = os.path.join(_ENGINE_DIR, "..", "processed_data", "histdata")
OUTPUT_DIR = os.path.join(_ENGINE_DIR, "backtest_results")

# ─── Symbols (NEVER seen during training) ─────────────────────────────────────
ROBUSTNESS_SYMBOLS = ["USDZAR", "USDMXN", "EURAUD", "GBPCHF", "AUDJPY"]

# ─── Epochs (3 × 3-year windows covering full 9-year dataset) ─────────────────
EPOCHS = {
    1: ("2016-01-01", "2018-12-31", "Pre-COVID (2016–2018)"),
    2: ("2019-01-01", "2021-12-31", "COVID & Recovery (2019–2021)"),
    3: ("2022-01-01", "2025-12-31", "Rate Hike Cycle (2022–2025)"),
}


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_parquet(symbol: str, tf: str, start: str, end: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{symbol}_{tf}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No parquet for {symbol} {tf}: {path}")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [c.lower() for c in df.columns]
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts   = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)
    df = df.loc[(df.index >= start_ts) & (df.index < end_ts)]
    return df


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """ATR-14, range/pullback flags — mirrors run_backtest._compute_indicators."""
    hi = df["high"].values
    lo = df["low"].values
    cl = df["close"].values
    n  = len(df)

    # True range
    prev_cl = np.empty(n, dtype=np.float64)
    prev_cl[0] = cl[0]
    prev_cl[1:] = cl[:-1]
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_cl), np.abs(lo - prev_cl)))

    # ATR-14 (Wilder smoothing)
    atr = np.empty(n, dtype=np.float64)
    atr[:14] = tr[:14].mean()
    alpha = 1.0 / 14.0
    for i in range(14, n):
        atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha

    df = df.copy()
    df["atr_14"] = atr

    # Range validity: price within 0.5×ATR of a 20-bar high/low
    _window = 20
    roll_hi = df["high"].rolling(_window, min_periods=1).max()
    roll_lo = df["low"].rolling(_window, min_periods=1).min()
    df["range_resist"]  = roll_hi
    df["range_support"] = roll_lo
    df["range_width_atr"] = (roll_hi - roll_lo) / (df["atr_14"] + 1e-9)

    at_top = (roll_hi - cl) / (df["atr_14"] + 1e-9) < 0.5
    at_bot = (cl - roll_lo) / (df["atr_14"] + 1e-9) < 0.5
    df["range_valid"] = (at_top | at_bot) & (df["range_width_atr"] > 1.5)
    df["range_side"]  = np.where(at_bot, "buy", np.where(at_top, "sell", ""))

    # Pullback: 3-bar EMA retest
    ema = df["close"].ewm(span=21, adjust=False).mean()
    near_ema = (df["close"] - ema).abs() / (df["atr_14"] + 1e-9) < 0.4
    df["pullback_valid"] = near_ema
    trend_up   = df["close"] > ema
    df["pullback_side"] = np.where(
        near_ema & trend_up, "buy",
        np.where(near_ema & ~trend_up, "sell", ""),
    )
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


# ─── ML cache (verbatim copy of _precompute_ml_cache from run_backtest) ───────

def _build_ml_cache(
    df: pd.DataFrame,
    symbol: str,
    htf: dict,
    ml_models: dict,
    batch_size: int = 512,
) -> dict:
    """GPU-batched GRU + regime inference. Returns {bar_index: {p_bull, p_bear, ...}}."""
    gru_model  = ml_models.get("gru")
    regime_4h  = ml_models.get("regime_htf") or ml_models.get("regime_4h")
    regime_1h  = ml_models.get("regime_ltf") or ml_models.get("regime_1h")

    if not (gru_model or regime_4h or regime_1h):
        return {}

    try:
        import torch
        from services.feature_engine import FeatureEngine, SEQUENCE_FEATURES

        fe = FeatureEngine()
        n  = len(df)
        SEQUENCE_LENGTH = 30

        # ── Batch regime helper ───────────────────────────────────────────────
        def _batch_regime(rc_model, df_src, htf_src, step=None):
            _mode = getattr(rc_model, "_mode", "ltf_behaviour")
            if _mode == "htf_bias":
                _classes   = ["BIAS_UP", "BIAS_DOWN", "BIAS_NEUTRAL"]
                _default   = 2
            else:
                _classes   = ["TRENDING", "RANGING", "CONSOLIDATING", "VOLATILE"]
                _default   = 1
            _n    = len(df_src)
            _step = step or max(1, _n // 20_000)
            try:
                from models.regime_classifier import RegimeClassifier as _RC
                X_all   = _RC._build_feature_matrix(df_src, htf_src, symbol)
                row_idx = list(range(50, _n, _step))
                X       = X_all[row_idx]
                del X_all
                if len(X) == 0:
                    return None, None, {}
                ids, conf = rc_model.predict_batch(X)
                del X
                arr    = np.full(_n, -1, dtype=np.int8)
                arr[row_idx] = ids.astype(np.int8)
                filled = (
                    pd.Series(arr.astype(float))
                    .replace(-1, np.nan)
                    .ffill()
                    .fillna(_default)
                    .astype(int)
                    .values
                )
                c_arr             = np.full(_n, np.nan, dtype=np.float32)
                c_arr[row_idx]    = conf
                filled_conf       = pd.Series(c_arr, index=df_src.index).ffill()
                preds             = dict(enumerate(np.array(_classes)[filled]))
                return pd.Series(filled, index=df_src.index, dtype=int), filled_conf, preds
            except Exception as exc:
                logger.error("ML cache: regime batch failed for %s: %s", symbol, exc)
                return None, None, {}

        # ── HTF bias (4H) ─────────────────────────────────────────────────────
        regime_preds      = {}
        _regime_htf_series = None
        _regime_htf_conf   = None
        _regime_ltf_series = None
        _regime_ltf_conf   = None

        if regime_4h and getattr(regime_4h, "is_trained", False) and regime_4h._model is not None:
            df_4h = htf.get("4H")
            if df_4h is not None and len(df_4h) >= 50:
                _r4h, _c4h, _p4h = _batch_regime(regime_4h, df_4h, htf)
                if _r4h is not None:
                    _regime_htf_series = _r4h.reindex(df.index, method="ffill").fillna(2).astype(int)
                    _regime_htf_conf   = _c4h.reindex(df.index, method="ffill").fillna(1 / 3)
                    _htf_cls           = np.array(["BIAS_UP", "BIAS_DOWN", "BIAS_NEUTRAL"])
                    regime_preds       = {i: str(v) for i, v in enumerate(
                        _htf_cls[np.clip(_regime_htf_series.values, 0, 2)]
                    )}
            else:
                _regime_htf_series, _regime_htf_conf, regime_preds = _batch_regime(
                    regime_4h, df, htf
                )
            gc.collect()

        # ── LTF behaviour (1H) ────────────────────────────────────────────────
        if regime_1h and getattr(regime_1h, "is_trained", False) and regime_1h._model is not None:
            df_1h = htf.get("1H")
            if df_1h is not None and len(df_1h) >= 50:
                _r1h, _c1h, _ = _batch_regime(regime_1h, df_1h, htf)
                if _r1h is not None:
                    _regime_ltf_series = _r1h.reindex(df.index, method="ffill").fillna(1).astype(int)
                    _regime_ltf_conf   = _c1h.reindex(df.index, method="ffill").fillna(0.25)
            else:
                _regime_ltf_series, _regime_ltf_conf, _ = _batch_regime(regime_1h, df, htf)
            gc.collect()

        # ── Sequence features ─────────────────────────────────────────────────
        logger.info("ML cache: building sequence features %s (%d bars)...", symbol, n)
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

        # ── Batched GRU inference ─────────────────────────────────────────────
        gru_preds: dict[int, dict] = {}
        if gru_model and getattr(gru_model, "is_trained", False) and gru_model._model is not None:
            from models.gru_lstm_predictor import DEVICE, SEQUENCE_LENGTH as SEQ_LEN
            n_valid = n - SEQ_LEN + 1
            n_feat  = seq_arr.shape[1]
            if n_valid > 0:
                m = gru_model._model.module if hasattr(gru_model._model, "module") else gru_model._model
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
                vol_arr    = np.sqrt(var_vals)
                bar_idxs   = np.arange(n_valid, dtype=np.int32) + (SEQ_LEN - 1)

                gru_preds = {
                    int(bar_idxs[i]): {
                        "p_bull":              float(all_p_bull[i]),
                        "p_bear":              float(p_bear_arr[i]),
                        "entry_depth":         float(entry_dep[i]),
                        "expected_move":       float(all_mag[i]),
                        "expected_volatility": float(vol_arr[i]),
                        "expected_variance":   float(var_vals[i]),
                    }
                    for i in range(n_valid)
                }
                del all_p_bull, all_mag, all_log_var, var_vals, p_bear_arr, entry_dep, vol_arr
                logger.info("ML cache: GRU done %s (%d bars)", symbol, len(gru_preds))
        del seq_arr
        gc.collect()

        # ── LTF per-bar name lookup ───────────────────────────────────────────
        ltf_preds: dict[int, str] = {}
        if _regime_ltf_series is not None:
            _ltf_cls = np.array(["TRENDING", "RANGING", "CONSOLIDATING", "VOLATILE"])
            ltf_preds = {i: str(v) for i, v in enumerate(
                _ltf_cls[np.clip(_regime_ltf_series.values, 0, 3)]
            )}

        # ── Merge ─────────────────────────────────────────────────────────────
        cache: dict[int, dict] = {}
        for bar_i in set(gru_preds.keys()) | set(regime_preds.keys()):
            entry: dict = {}
            if bar_i in gru_preds:
                entry.update(gru_preds[bar_i])
            if bar_i in regime_preds:
                entry["regime"]     = regime_preds[bar_i]
            if bar_i in ltf_preds:
                entry["regime_ltf"] = ltf_preds[bar_i]
            cache[bar_i] = entry

        logger.info(
            "ML cache done %s: %d bars (gru=%d htf=%d ltf=%d)",
            symbol, len(cache), len(gru_preds), len(regime_preds), len(ltf_preds),
        )
        del gru_preds, regime_preds
        gc.collect()
        return cache

    except Exception as exc:
        logger.error("_build_ml_cache failed %s: %s", symbol, exc)
        gc.collect()
        raise


# ─── Signal generation (exact copy of _compute_backtest_signal) ───────────────

def _compute_signal(symbol: str, ml_preds: dict, bar: pd.Series) -> dict | None:
    close = float(bar["close"])
    atr   = float(bar.get("atr_14", close * 0.001))
    if atr < 1e-9 or not ml_preds:
        return None

    _uncertainty = float(ml_preds.get("expected_variance", 0.0))
    if _uncertainty > float(os.getenv("MAX_UNCERTAINTY", "2.0")):
        return None

    p_bull      = float(ml_preds.get("p_bull", 0.5))
    p_bear      = float(ml_preds.get("p_bear", 0.5))
    _dir_thresh = float(os.getenv("ML_DIRECTION_THRESHOLD", "0.58"))

    if p_bull >= p_bear and p_bull >= _dir_thresh:
        side, conf = "buy",  p_bull
    elif p_bear > p_bull and p_bear >= _dir_thresh:
        side, conf = "sell", p_bear
    else:
        return None

    _htf_bias       = str(ml_preds.get("regime", "BIAS_NEUTRAL"))
    _neutral_thresh = float(os.getenv("NEUTRAL_BIAS_THRESHOLD", "0.58"))

    if _htf_bias == "BIAS_UP"   and side == "sell": return None
    if _htf_bias == "BIAS_DOWN" and side == "buy":  return None
    if _htf_bias == "BIAS_NEUTRAL" and conf < _neutral_thresh: return None

    _ltf_behaviour  = str(ml_preds.get("regime_ltf", "TRENDING"))
    _volatile_thresh = float(os.getenv("VOLATILE_ENTRY_THRESHOLD", str(_dir_thresh)))
    _range_valid    = bool(bar.get("range_valid",    False))
    _range_side     = str(bar.get("range_side",     ""))
    _pullback_valid = bool(bar.get("pullback_valid", False))
    _pullback_side  = str(bar.get("pullback_side",  ""))

    if _ltf_behaviour == "CONSOLIDATING":
        if str(os.getenv("BLOCK_LTF_CONSOLIDATING", "0")).lower() in ("1", "true", "yes"):
            return None

    if _ltf_behaviour == "VOLATILE" and conf < _volatile_thresh:
        return None

    if _ltf_behaviour == "TRENDING":
        if _pullback_valid and _pullback_side and _pullback_side != side:
            return None

    if _ltf_behaviour == "RANGING":
        if _range_valid and _range_side and _range_side != side:
            return None

    _sl_mult    = float(os.getenv("SL_ATR_MULT",  "1.5"))
    _rr_default = float(os.getenv("RR_DEFAULT",   "2.0"))
    sl_dist     = atr * _sl_mult

    if _ltf_behaviour == "RANGING" and _range_valid:
        if side == "buy":
            stop_loss   = float(bar.get("range_support", close - sl_dist)) - atr * 0.3
            take_profit = float(bar.get("range_resist",  close + sl_dist * _rr_default))
        else:
            stop_loss   = float(bar.get("range_resist",  close + sl_dist)) + atr * 0.3
            take_profit = float(bar.get("range_support", close - sl_dist * _rr_default))
        actual_rr = abs(take_profit - close) / (abs(close - stop_loss) + 1e-9)
        if actual_rr < 1.5:
            stop_loss   = (close - sl_dist) if side == "buy" else (close + sl_dist)
            take_profit = (close + sl_dist * _rr_default) if side == "buy" else (close - sl_dist * _rr_default)
    else:
        if side == "buy":
            stop_loss   = close - sl_dist
            take_profit = close + sl_dist * _rr_default
        else:
            stop_loss   = close + sl_dist
            take_profit = close - sl_dist * _rr_default

    return {
        "side":        side,
        "entry":       close,
        "stop_loss":   stop_loss,
        "take_profit": take_profit,
        "confidence":  round(float(conf), 3),
        "trader_id":   "ml_trader",
        "symbol":      symbol,
        "signal_metadata": {
            "regime":            _htf_bias,
            "regime_ltf":        _ltf_behaviour,
            "expected_variance": _uncertainty,
            "p_bull":            p_bull,
            "p_bear":            p_bear,
            "atr_at_entry":      atr,
        },
    }


# ─── Trade simulation (phase 1 TP1 + phase 2 TP2, mirrors run_backtest) ───────

def _simulate_trade(
    df: pd.DataFrame,
    entry_i: int,
    side: str,
    entry: float,
    sl: float,
    tp1: float,
    tp2: float,
    size: float,
    atr: float,
    commission_pct: float = COMMISSION_PCT,
) -> dict:
    direction = 1 if side == "buy" else -1
    phase1_done = False
    be_stop     = sl
    pnl         = 0.0
    exit_reason = "timeout"
    exit_price  = entry
    bars_held   = 0
    phase1_pnl  = 0.0
    phase2_pnl  = 0.0

    for j in range(entry_i + 1, min(entry_i + MAX_HOLD_BARS + 1, len(df))):
        bar_hi = float(df["high"].iloc[j])
        bar_lo = float(df["low"].iloc[j])
        bars_held = j - entry_i

        if not phase1_done:
            # Phase 1: 50 % position to TP1
            if (side == "buy"  and bar_hi >= tp1) or (side == "sell" and bar_lo <= tp1):
                phase1_pnl  = direction * (tp1 - entry) * size * 0.5
                phase1_done = True
                be_stop     = entry + direction * atr * 0.1   # trail to breakeven
                exit_price  = tp1
                exit_reason = "tp1"

            # SL hit in phase 1
            if (side == "buy"  and bar_lo <= sl) or (side == "sell" and bar_hi >= sl):
                phase1_pnl  = direction * (sl - entry) * size
                pnl         = phase1_pnl - commission_pct * abs(entry) * size
                exit_price  = sl
                exit_reason = "sl"
                return {
                    "pnl": pnl, "exit_reason": exit_reason,
                    "bars_held": bars_held, "exit_price": exit_price,
                    "phase1_pnl": phase1_pnl, "phase2_pnl": 0.0,
                }

        else:
            # Phase 2: remaining 50 % to TP2
            # Breakeven / trail stop for remaining half
            if (side == "buy"  and bar_lo <= be_stop) or (side == "sell" and bar_hi >= be_stop):
                phase2_pnl  = direction * (be_stop - entry) * size * 0.5
                pnl         = phase1_pnl + phase2_pnl - commission_pct * abs(entry) * size
                exit_price  = be_stop
                exit_reason = "be_or_trail"
                return {
                    "pnl": pnl, "exit_reason": exit_reason,
                    "bars_held": bars_held, "exit_price": exit_price,
                    "phase1_pnl": phase1_pnl, "phase2_pnl": phase2_pnl,
                }
            if (side == "buy"  and bar_hi >= tp2) or (side == "sell" and bar_lo <= tp2):
                phase2_pnl  = direction * (tp2 - entry) * size * 0.5
                pnl         = phase1_pnl + phase2_pnl - commission_pct * abs(entry) * size
                exit_price  = tp2
                exit_reason = "tp2"
                return {
                    "pnl": pnl, "exit_reason": exit_reason,
                    "bars_held": bars_held, "exit_price": exit_price,
                    "phase1_pnl": phase1_pnl, "phase2_pnl": phase2_pnl,
                }

    # Timeout — exit at last bar close
    last_close = float(df["close"].iloc[min(entry_i + MAX_HOLD_BARS, len(df) - 1)])
    if not phase1_done:
        pnl        = direction * (last_close - entry) * size - commission_pct * abs(entry) * size
        phase1_pnl = pnl
    else:
        phase2_pnl = direction * (last_close - entry) * size * 0.5
        pnl        = phase1_pnl + phase2_pnl - commission_pct * abs(entry) * size
    return {
        "pnl": pnl, "exit_reason": "timeout",
        "bars_held": MAX_HOLD_BARS, "exit_price": last_close,
        "phase1_pnl": phase1_pnl, "phase2_pnl": phase2_pnl,
    }


# ─── Per-symbol epoch backtest ────────────────────────────────────────────────

def _run_symbol_epoch(
    symbol: str,
    epoch_num: int,
    start: str,
    end: str,
    ml_models: dict,
    pm,
) -> dict:
    logger.info("=== %s  Epoch %d (%s → %s) ===", symbol, epoch_num, start, end)

    # Load data
    try:
        df  = _load_parquet(symbol, "15M", start, end)
        htf = _load_htf(symbol, start, end)
    except FileNotFoundError as exc:
        logger.error("Data missing for %s: %s", symbol, exc)
        return {"symbol": symbol, "epoch": epoch_num, "error": str(exc), "trades": 0}

    if len(df) < 200:
        logger.warning("Not enough bars for %s epoch %d (%d bars)", symbol, epoch_num, len(df))
        return {"symbol": symbol, "epoch": epoch_num, "trades": 0, "note": "insufficient_data"}

    df = _compute_indicators(df)

    # ML inference cache
    ml_cache: dict[int, dict] = {}
    if ml_models:
        try:
            ml_cache = _build_ml_cache(df, symbol, htf, ml_models)
        except Exception as exc:
            logger.error("ML cache failed %s epoch %d: %s", symbol, epoch_num, exc)

    # ── Main bar loop ─────────────────────────────────────────────────────────
    trades:      list[dict] = []
    equity       = INITIAL_CAPITAL
    peak_equity  = INITIAL_CAPITAL
    daily_budget = INITIAL_CAPITAL * MAX_DAILY_LOSS_PCT
    daily_loss   = 0.0
    current_date = None
    daily_halt   = False
    last_trade_i = -COOLDOWN_BARS
    recent_trade_bars: list[int] = []
    halted = False

    _dir_thresh = float(os.getenv("ML_DIRECTION_THRESHOLD", "0.58"))
    _density_lambda = float(os.getenv("DENSITY_LAMBDA", "0.12"))

    for i in range(200, len(df)):
        if halted:
            break

        dt  = df.index[i]
        bar = df.iloc[i]

        # DD halt
        dd = (peak_equity - equity) / (peak_equity + 1e-9)
        if dd >= MAX_DRAWDOWN_PCT:
            logger.warning("%s epoch %d: drawdown halt at %.1f%%", symbol, epoch_num, dd * 100)
            halted = True
            break

        # Daily reset
        day_str = dt.strftime("%Y-%m-%d")
        if day_str != current_date:
            current_date = day_str
            daily_loss   = 0.0
            daily_halt   = False
            pm.notify_date(day_str)

        if daily_halt or abs(daily_loss) >= daily_budget:
            daily_halt = True
            continue

        # Session filter (London → NY)
        if not (7 <= dt.hour < 18):
            continue

        # Cooldown
        if i - last_trade_i < COOLDOWN_BARS:
            continue

        # ML predictions
        ml_preds = ml_cache.get(i, {})

        # Signal
        raw_signal = _compute_signal(symbol, ml_preds, bar)
        if raw_signal is None:
            continue

        # Density penalty
        recent_trade_bars = [b for b in recent_trade_bars if i - b < 96]
        _density_count = len(recent_trade_bars)
        raw_signal["confidence"] = float(raw_signal["confidence"]) * math.exp(
            -_density_lambda * _density_count
        )
        if raw_signal["confidence"] < _dir_thresh:
            continue
        if MIN_CONFIDENCE > 0 and raw_signal["confidence"] < MIN_CONFIDENCE:
            continue

        # Slippage
        entry_raw = float(raw_signal["entry"])
        raw_signal["entry"] = (
            entry_raw * (1 + SLIPPAGE_PCT)
            if raw_signal["side"] == "buy"
            else entry_raw * (1 - SLIPPAGE_PCT)
        )
        atr = float(bar.get("atr_14", entry_raw * 0.001))

        # PM enrichment (R:R gate, sizing)
        portfolio_state = {"equity": equity, "open_positions_detail": []}
        enriched = pm.enrich_signal(raw_signal, portfolio_state, atr=atr)
        if enriched is None:
            continue

        entry  = float(enriched["entry"])
        sl     = float(enriched["stop_loss"])
        tp1    = float(enriched["tp1"])
        tp2    = float(enriched["tp2"])
        size   = float(enriched["size"])
        rr     = float(enriched["rr_ratio"])
        side   = enriched["side"]

        # Simulate trade
        result = _simulate_trade(df, i, side, entry, sl, tp1, tp2, size, atr)
        pnl    = result["pnl"]

        # Update state
        equity      += pnl
        peak_equity  = max(peak_equity, equity)
        daily_loss  += min(0.0, pnl)
        last_trade_i = i
        recent_trade_bars.append(i)

        trades.append({
            "symbol":       symbol,
            "epoch":        epoch_num,
            "entry_time":   dt.isoformat(),
            "side":         side,
            "entry":        round(entry, 5),
            "exit":         round(result["exit_price"], 5),
            "sl":           round(sl, 5),
            "tp1":          round(tp1, 5),
            "tp2":          round(tp2, 5),
            "size":         round(size, 4),
            "pnl":          round(pnl, 4),
            "exit_reason":  result["exit_reason"],
            "bars_held":    result["bars_held"],
            "rr_ratio":     round(rr, 2),
            "confidence":   round(raw_signal["confidence"], 3),
            "regime":       ml_preds.get("regime",     "UNKNOWN"),
            "regime_ltf":   ml_preds.get("regime_ltf", "UNKNOWN"),
            "p_bull":       round(ml_preds.get("p_bull", 0.5), 3),
            "p_bear":       round(ml_preds.get("p_bear", 0.5), 3),
            "equity_after": round(equity, 2),
        })

    # ── Metrics ───────────────────────────────────────────────────────────────
    return _compute_metrics(symbol, epoch_num, start, end, trades, equity, peak_equity)


def _compute_metrics(
    symbol: str,
    epoch_num: int,
    start: str,
    end: str,
    trades: list[dict],
    final_equity: float,
    peak_equity: float,
) -> dict:
    n = len(trades)
    if n == 0:
        return {
            "symbol": symbol, "epoch": epoch_num, "start": start, "end": end,
            "trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "total_return_pct": 0.0, "max_drawdown_pct": 0.0,
            "sharpe": 0.0, "avg_rr": 0.0, "tp1_rate": 0.0, "tp2_rate": 0.0,
        }

    pnls     = np.array([t["pnl"] for t in trades])
    wins     = pnls[pnls > 0]
    losses   = pnls[pnls <= 0]
    win_rate = len(wins) / n

    gross_profit = float(wins.sum())  if len(wins)   > 0 else 0.0
    gross_loss   = float(losses.sum()) if len(losses) > 0 else 0.0
    pf           = gross_profit / (abs(gross_loss) + 1e-9)

    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL

    # Running equity for Sharpe and drawdown
    equity_curve = np.empty(n + 1)
    equity_curve[0] = INITIAL_CAPITAL
    for k, t in enumerate(trades):
        equity_curve[k + 1] = t["equity_after"]
    daily_rets = np.diff(equity_curve) / (equity_curve[:-1] + 1e-9)
    sharpe     = (float(daily_rets.mean()) / (float(daily_rets.std()) + 1e-9)) * math.sqrt(252 * 26)

    # Max drawdown from equity curve
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns   = (running_max - equity_curve) / (running_max + 1e-9)
    max_dd      = float(drawdowns.max())

    tp1_rate = sum(1 for t in trades if t["exit_reason"] == "tp1") / n
    tp2_rate = sum(1 for t in trades if t["exit_reason"] == "tp2") / n
    avg_rr   = float(np.mean([t["rr_ratio"] for t in trades]))

    # Regime distribution of taken trades
    from collections import Counter
    htf_dist = dict(Counter(t["regime"]     for t in trades))
    ltf_dist = dict(Counter(t["regime_ltf"] for t in trades))
    side_dist = dict(Counter(t["side"]      for t in trades))

    return {
        "symbol":           symbol,
        "epoch":            epoch_num,
        "start":            start,
        "end":              end,
        "trades":           n,
        "win_rate":         round(win_rate,     4),
        "profit_factor":    round(pf,           3),
        "total_return_pct": round(total_return * 100, 2),
        "max_drawdown_pct": round(max_dd * 100,  2),
        "sharpe":           round(sharpe,        3),
        "avg_rr":           round(avg_rr,        3),
        "tp1_rate":         round(tp1_rate,      4),
        "tp2_rate":         round(tp2_rate,      4),
        "gross_profit":     round(gross_profit,  2),
        "gross_loss":       round(gross_loss,    2),
        "final_equity":     round(float(equity_curve[-1]), 2),
        "htf_regime_dist":  htf_dist,
        "ltf_regime_dist":  ltf_dist,
        "side_distribution": side_dist,
        "trade_log":        trades,
    }


# ─── Summary table ────────────────────────────────────────────────────────────

def _print_summary(all_results: list[dict]) -> str:
    lines = [
        "",
        "=" * 100,
        "  ROBUSTNESS BACKTEST — Out-of-sample symbols (model never trained on these)",
        "=" * 100,
        f"  {'Symbol':<10} {'Epoch':<6} {'Period':<30} {'Trades':>7} {'WR%':>6} {'PF':>6} "
        f"{'Return%':>8} {'MaxDD%':>8} {'Sharpe':>7} {'TP1%':>6} {'TP2%':>6}",
        "-" * 100,
    ]

    for res in all_results:
        if res.get("error") or res.get("trades", 0) == 0:
            label = res.get("error", "no_trades")
            lines.append(
                f"  {res['symbol']:<10} {res['epoch']:<6} "
                f"{EPOCHS[res['epoch']][2]:<30}  -- {label}"
            )
            continue
        lines.append(
            f"  {res['symbol']:<10} {res['epoch']:<6} "
            f"{EPOCHS[res['epoch']][2]:<30} "
            f"{res['trades']:>7} "
            f"{res['win_rate']*100:>5.1f}% "
            f"{res['profit_factor']:>6.2f} "
            f"{res['total_return_pct']:>7.1f}% "
            f"{res['max_drawdown_pct']:>7.1f}% "
            f"{res['sharpe']:>7.2f} "
            f"{res['tp1_rate']*100:>5.1f}% "
            f"{res['tp2_rate']*100:>5.1f}%"
        )

    # Per-epoch aggregate
    lines.append("-" * 100)
    for epoch_num in sorted(EPOCHS.keys()):
        ep_res = [r for r in all_results if r.get("epoch") == epoch_num and r.get("trades", 0) > 0]
        if not ep_res:
            continue
        total_trades = sum(r["trades"] for r in ep_res)
        avg_wr  = sum(r["win_rate"]         for r in ep_res) / len(ep_res)
        avg_pf  = sum(r["profit_factor"]    for r in ep_res) / len(ep_res)
        avg_ret = sum(r["total_return_pct"] for r in ep_res) / len(ep_res)
        avg_dd  = sum(r["max_drawdown_pct"] for r in ep_res) / len(ep_res)
        avg_sh  = sum(r["sharpe"]           for r in ep_res) / len(ep_res)
        lines.append(
            f"  {'[AVG]':<10} {epoch_num:<6} "
            f"{EPOCHS[epoch_num][2]:<30} "
            f"{total_trades:>7} "
            f"{avg_wr*100:>5.1f}% "
            f"{avg_pf:>6.2f} "
            f"{avg_ret:>7.1f}% "
            f"{avg_dd:>7.1f}% "
            f"{avg_sh:>7.2f}"
        )

    # Grand aggregate across all epochs and symbols
    all_valid = [r for r in all_results if r.get("trades", 0) > 0]
    if all_valid:
        lines.append("=" * 100)
        gt = sum(r["trades"]           for r in all_valid)
        gw = sum(r["win_rate"]         for r in all_valid) / len(all_valid)
        gp = sum(r["profit_factor"]    for r in all_valid) / len(all_valid)
        gr = sum(r["total_return_pct"] for r in all_valid) / len(all_valid)
        gd = sum(r["max_drawdown_pct"] for r in all_valid) / len(all_valid)
        gs = sum(r["sharpe"]           for r in all_valid) / len(all_valid)
        lines.append(
            f"  {'OVERALL':<10} {'ALL':<6} {'All epochs / all symbols':<30} "
            f"{gt:>7} {gw*100:>5.1f}% {gp:>6.2f} {gr:>7.1f}% {gd:>7.1f}% {gs:>7.2f}"
        )
    lines.append("=" * 100)
    lines.append("")

    text = "\n".join(lines)
    print(text)
    return text


# ─── Model loading (mirrors run_backtest main) ────────────────────────────────

def _load_models() -> dict:
    models: dict = {}
    try:
        from models.regime_classifier import RegimeClassifier
        htf = RegimeClassifier(timeframe="4H", mode="htf_bias")
        if getattr(htf, "is_trained", False):
            models["regime_htf"] = htf
            logger.info("[OK] RegimeClassifier HTF loaded")
        ltf = RegimeClassifier(timeframe="1H", mode="ltf_behaviour")
        if getattr(ltf, "is_trained", False):
            models["regime_ltf"] = ltf
            logger.info("[OK] RegimeClassifier LTF loaded")
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


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness backtest on unseen symbols")
    parser.add_argument(
        "--epoch", type=int, choices=[1, 2, 3], default=None,
        help="Run only this epoch (1=2016-18, 2=2019-21, 3=2022-25). Default: all 3.",
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        choices=ROBUSTNESS_SYMBOLS,
        help="Run only this symbol. Default: all 5.",
    )
    parser.add_argument(
        "--no-ml", action="store_true",
        help="Skip ML inference (baseline: signal gates only, no GRU/regime).",
    )
    args = parser.parse_args()

    epochs_to_run  = [args.epoch]  if args.epoch  else list(EPOCHS.keys())
    symbols_to_run = [args.symbol] if args.symbol else ROBUSTNESS_SYMBOLS

    logger.info("Robustness backtest starting")
    logger.info("  Symbols: %s", symbols_to_run)
    logger.info("  Epochs:  %s", epochs_to_run)

    # Load models once
    ml_models = {} if args.no_ml else _load_models()
    if not ml_models:
        logger.warning("No ML models loaded — backtest will produce 0 signals. "
                       "Check that weights/ directory is populated.")

    # Load PortfolioManager
    try:
        from monitors.portfolio_manager import PortfolioManager
        pm = PortfolioManager(_PM_SETTINGS)
        logger.info("[OK] PortfolioManager loaded")
    except Exception as exc:
        logger.error("PortfolioManager failed to load: %s", exc)
        sys.exit(1)

    # Run all combinations
    all_results: list[dict] = []
    for epoch_num in epochs_to_run:
        start, end, label = EPOCHS[epoch_num]
        logger.info("── Epoch %d: %s (%s → %s) ──", epoch_num, label, start, end)
        for symbol in symbols_to_run:
            result = _run_symbol_epoch(symbol, epoch_num, start, end, ml_models, pm)
            # Remove full trade_log from the summary dict (kept in JSON output)
            trade_log = result.pop("trade_log", [])
            result["trade_log"] = trade_log   # re-attach for JSON
            all_results.append(result)
            logger.info(
                "  %s epoch %d done: %d trades, WR=%.1f%%, PF=%.2f, Return=%.1f%%",
                symbol, epoch_num,
                result.get("trades", 0),
                result.get("win_rate", 0) * 100,
                result.get("profit_factor", 0),
                result.get("total_return_pct", 0),
            )
        gc.collect()

    # Print table (without trade_log noise)
    summary_results = [{k: v for k, v in r.items() if k != "trade_log"} for r in all_results]
    summary_text = _print_summary(summary_results)

    # Save JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"robustness_{ts}.json")
    payload  = {
        "timestamp":    ts,
        "symbols":      symbols_to_run,
        "epochs":       {str(k): list(v) for k, v in EPOCHS.items()},
        "results":      all_results,
        "summary":      summary_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Results saved → %s", out_path)

    # Save plain-text summary
    txt_path = os.path.join(OUTPUT_DIR, "robustness_summary.txt")
    with open(txt_path, "w") as f:
        f.write(summary_text)
    logger.info("Summary table → %s", txt_path)


if __name__ == "__main__":
    main()
