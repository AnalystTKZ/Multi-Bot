#!/usr/bin/env python3
"""
Step 7: Model training using existing scripts/retrain_incremental.py
- Generates synthetic trade journal if empty (enables quality/RL training)
- Runs retrain for gru, regime, quality, rl models
- Weights saved to trading-engine/weights/ (canonical) — trading-system/weights/ symlinks to it
"""
from __future__ import annotations
import json
import logging
import os
import subprocess
import sys
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("step7_train")

# Use env_config so outputs go to the remote clone on Kaggle when present.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env_config import get_env
_ENV = get_env()

BASE       = _ENV["base"]
ENGINE_DIR = _ENV["engine"]
ML_DIR     = _ENV["ml_training"]
ML_METRICS = ML_DIR / "metrics"
ML_LOGS    = ML_DIR / "logs"
for d in [ML_METRICS, ML_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

JOURNAL_PATH     = ENGINE_DIR / "logs" / "trade_journal_detailed.jsonl"
JOURNAL_CSV_PATH = ENGINE_DIR / "logs" / "trade_journal.csv"


def generate_synthetic_journal(n_trades: int = 200) -> None:
    """
    Generate synthetic trade journal entries.
    Loads only 6 columns from the feature parquet to keep RAM low.
    """
    logger.info("Generating synthetic trade journal (%d trades)...", n_trades)

    fe_path = BASE / "processed_data" / "feature_engineered.parquet"
    if not fe_path.exists():
        logger.error("feature_engineered.parquet not found")
        return

    # Only load the columns we actually use — read schema without loading data
    import pyarrow.parquet as pq
    schema_cols = set(pq.read_schema(fe_path).names)
    needed = ["close", "high", "low", "open", "volume", "atr_14",
              "rsi_14", "adx_14", "ema_21", "ema_50",
              "bos_bull", "bos_bear", "fvg_bull", "fvg_bear",
              "sweep_bull", "sweep_bear", "bb_position", "volume_ratio",
              "log_return", "atr_normalized"]
    available = [c for c in needed if c in schema_cols]

    df = pd.read_parquet(fe_path, columns=available)
    df = df.sort_index()

    # Sample from training portion only (no test leakage)
    train_end = int(len(df) * 0.70)
    df_train = df.iloc[:train_end]

    rng = np.random.default_rng(42)

    traders = ["trader_1", "trader_2", "trader_3", "trader_4", "trader_5"]
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "AUDUSD", "USDCAD"]
    trader_symbols = {
        "trader_1": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"],
        "trader_2": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"],
        "trader_3": ["EURUSD", "GBPUSD", "XAUUSD"],
        "trader_4": ["EURUSD", "GBPUSD", "XAUUSD"],
        "trader_5": ["USDJPY", "EURUSD", "AUDUSD"],
    }

    # Sample indices from training data
    sample_idx = rng.choice(len(df_train) - 200, size=n_trades, replace=False)
    sample_idx = sorted(sample_idx)

    csv_rows = []
    jsonl_rows = []

    for i, idx in enumerate(sample_idx):
        bar = df_train.iloc[idx]
        ts = df_train.index[idx]

        trader = traders[i % len(traders)]
        sym_list = trader_symbols[trader]
        symbol = sym_list[i % len(sym_list)]

        close = float(bar.get("close", 1.0))
        atr = float(bar.get("atr_14", close * 0.001))
        if atr < 1e-9:
            atr = close * 0.001

        side = "buy" if rng.random() > 0.5 else "sell"
        size = round(rng.uniform(0.01, 0.5), 2)

        # Entry / SL / TP
        sl_dist = atr * rng.uniform(0.8, 1.5)
        tp_dist = sl_dist * rng.uniform(1.5, 3.0)
        if side == "buy":
            entry = close
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            entry = close
            sl = entry + sl_dist
            tp = entry - tp_dist

        rr = round(tp_dist / (sl_dist + 1e-9), 2)

        # Outcome distribution matching real backtest:
        #   tp2         ~25%  full TP hit
        #   tp1         ~20%  partial TP, held on
        #   be_or_trail ~10%  TP1 then trailed/stopped
        #   sl          ~40%  stopped out
        #   time_exit    ~5%  timed out
        outcome_roll = rng.random()
        if outcome_roll < 0.25:
            exit_reason = "tp2"
            pnl = tp_dist * size
        elif outcome_roll < 0.45:
            exit_reason = "tp1"
            pnl = tp_dist * 0.75 * size
        elif outcome_roll < 0.55:
            exit_reason = "be_or_trail"
            pnl = tp_dist * 0.4 * size
        elif outcome_roll < 0.95:
            exit_reason = "sl"
            pnl = -sl_dist * size
        else:
            exit_reason = "time_exit"
            # Small residual PnL for timed-out trades (can be positive or negative)
            pnl = float(rng.uniform(-sl_dist * 0.3, tp_dist * 0.2)) * size

        commission = abs(pnl) * 0.001
        pnl_net = round(pnl - commission, 4)

        confidence = round(rng.uniform(0.55, 0.95), 3)

        # Fake exit timestamp (1-24 hours later)
        exit_offset = pd.Timedelta(hours=rng.uniform(1, 24))
        exit_ts = ts + exit_offset

        # 42-dim RL state (derived from bar features)
        state_vec = []
        for feat in ["close", "high", "low", "open", "volume",
                     "rsi_14", "adx_14", "atr_14", "ema_21", "ema_50",
                     "bos_bull", "bos_bear", "fvg_bull", "fvg_bear",
                     "sweep_bull", "sweep_bear", "bb_position", "volume_ratio",
                     "log_return", "atr_normalized"]:
            v = bar.get(feat, 0.0)
            state_vec.append(float(v) if pd.notna(v) else 0.0)
        # Pad to 42
        while len(state_vec) < 42:
            state_vec.append(0.0)
        state_vec = state_vec[:42]

        # CSV row
        csv_rows.append({
            "timestamp": ts.isoformat(),
            "trader": trader,
            "symbol": symbol,
            "side": side,
            "size": size,
            "entry": round(entry, 6),
            "stop_loss": round(sl, 6),
            "take_profit": round(tp, 6),
            "rr_ratio": rr,
            "confidence": confidence,
            "pnl": pnl_net,
            "commission": round(commission, 4),
        })

        # JSONL row (full Contract 4 format)
        jsonl_rows.append({
            "timestamp": ts.isoformat(),
            "exit_timestamp": exit_ts.isoformat(),
            "trader": trader,
            "symbol": symbol,
            "side": side,
            "size": size,
            "entry": round(entry, 6),
            "stop_loss": round(sl, 6),
            "take_profit": round(tp, 6),
            "rr_ratio": rr,
            "confidence": confidence,
            "pnl": pnl_net,
            "commission": round(commission, 4),
            "exit_reason": exit_reason,
            "strategy": trader,
            "session": "london" if 7 <= ts.hour < 12 else ("ny" if 13 <= ts.hour < 18 else "asian"),
            "correlation_id": f"synth_{i:05d}",
            "state_at_entry": state_vec,
            "rl_action": int(traders.index(trader)) + 1,
            "ml_model_scores": {
                "p_bull": round(rng.uniform(0.3, 0.8), 3),
                "p_bear": round(rng.uniform(0.2, 0.7), 3),
                "quality_score": round(rng.uniform(0.4, 0.9), 3),
                "regime": rng.choice(["BIAS_UP", "BIAS_DOWN", "BIAS_NEUTRAL", "TRENDING", "RANGING", "CONSOLIDATING", "VOLATILE"]),
                "sentiment_score": round(rng.uniform(-0.5, 0.5), 3),
            },
            "signal_metadata": {
                "atr": round(atr, 6),
                # Bar-derived values consumed by quality_scorer.create_labels()
                "adx_at_signal":        float(bar.get("adx_14", 20.0)),
                "atr_ratio_at_signal":  float(bar.get("atr_normalized", 1.0))
                                        if bar.get("atr_normalized") else
                                        round(atr / (close * 0.001 + 1e-12), 4),
                "volume_ratio":         float(bar.get("volume_ratio", 1.0)),
                "spread_at_signal":     round(atr * rng.uniform(0.05, 0.20), 6),
                "news_in_30min":        int(rng.random() < 0.10),   # ~10% chance of nearby news
                "strategy_win_rate_20": round(float(rng.uniform(0.40, 0.70)), 3),
                "ema_stack":            float(bar.get("ema_stack", 0)),
            },
        })

    # Write CSV (append if exists, else create with header)
    csv_df = pd.DataFrame(csv_rows)
    if JOURNAL_CSV_PATH.exists() and JOURNAL_CSV_PATH.stat().st_size > 50:
        existing = pd.read_csv(JOURNAL_CSV_PATH)
        if len(existing) > 1:  # already has real data
            logger.info("Real journal CSV exists with %d rows — not overwriting", len(existing))
        else:
            csv_df.to_csv(JOURNAL_CSV_PATH, index=False)
            logger.info("Wrote synthetic journal CSV: %d rows", len(csv_df))
    else:
        csv_df.to_csv(JOURNAL_CSV_PATH, index=False)
        logger.info("Wrote synthetic journal CSV: %d rows", len(csv_df))

    # Write JSONL
    if JOURNAL_PATH.exists() and JOURNAL_PATH.stat().st_size > 100:
        logger.info("Real journal JSONL exists — not overwriting")
    else:
        JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(JOURNAL_PATH, "w") as f:
            for row in jsonl_rows:
                f.write(json.dumps(row, default=str) + "\n")
        logger.info("Wrote synthetic journal JSONL: %d entries", len(jsonl_rows))

    # Release all lists and the sampled df before returning
    del df, csv_rows, jsonl_rows, csv_df
    import gc; gc.collect()


def run_retrain(model: str, dry_run: bool = False) -> dict:
    """Execute retrain_incremental.py for a given model."""
    script = ENGINE_DIR / "scripts" / "retrain_incremental.py"
    if not script.exists():
        return {"error": f"script not found: {script}"}

    cmd = [sys.executable, str(script), "--model", model]
    if dry_run:
        cmd.append("--dry-run")

    env = os.environ.copy()
    # PYTHONPATH must include engine dir for imports
    env["PYTHONPATH"] = str(ENGINE_DIR)
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"

    logger.info("Running retrain --model %s%s", model, " --dry-run" if dry_run else "")
    try:
        # Stream stdout/stderr directly — do NOT use capture_output=True.
        # LightGBM/GRU training emits thousands of log lines; the 64 KB OS pipe
        # buffer fills and the child blocks waiting for the parent to drain, but
        # the parent is blocked in subprocess.run() waiting for the child to exit
        # → permanent deadlock. Streaming lets output go straight to the terminal.
        result = subprocess.run(
            cmd,
            cwd=str(ENGINE_DIR),
            env=env,
            timeout=14400,  # 4 hrs per model
        )
        if result.returncode != 0:
            logger.error("retrain %s failed (exit %d)", model, result.returncode)
            return {"error": f"exit {result.returncode}", "returncode": result.returncode, "model": model}

        return {"success": True, "model": model}

    except subprocess.TimeoutExpired:
        return {"error": "timeout", "model": model}
    except Exception as e:
        return {"error": str(e), "model": model}


def collect_model_artifacts():
    """Verify trained weights exist in the canonical location: trading-engine/weights/.

    trading-system/weights/ symlinks to trading-engine/weights/ — no copying needed.
    ml_training/models/ is NOT a weights store; weights live only in trading-engine/weights/.
    """
    weights_dir = ENGINE_DIR / "weights"
    if not weights_dir.exists():
        logger.warning("Canonical weights dir not found: %s", weights_dir)
        return

    found, missing = [], []
    checks = {
        "gru_lstm":             weights_dir / "gru_lstm" / "model.pt",
        "regime_classifier":    weights_dir / "regime_classifier.pkl",
        "quality_scorer":       weights_dir / "quality_scorer.pkl",
        "rl_ppo":               weights_dir / "rl_ppo" / "model.zip",
    }
    for name, path in checks.items():
        if path.exists():
            found.append(name)
            logger.info("  [OK] %s → %s", name, path)
        else:
            missing.append(name)
            logger.warning("  [MISSING] %s → %s", name, path)

    if missing:
        logger.warning("Missing weights: %s — run retrain_incremental.py for each", missing)
    else:
        logger.info("All weights present in canonical location: %s", weights_dir)


def collect_metrics():
    """Parse retrain_history.jsonl and save to ml_training/metrics/."""
    history_path = ENGINE_DIR / "logs" / "retrain_history.jsonl"
    if not history_path.exists():
        logger.warning("No retrain_history.jsonl found")
        return {}

    records = []
    with open(history_path) as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except Exception:
                pass

    if records:
        with open(ML_METRICS / "retrain_history.json", "w") as f:
            json.dump(records, f, indent=2, default=str)
        logger.info("Saved %d retrain records to metrics/", len(records))

    return {"records": len(records)}


def main():
    """
    Step 7a — train GRU + Regime only.
    Quality and RL are trained in step7b AFTER step6 backtest generates a real journal.
    No synthetic journal fallback — backtest must run first.
    """
    logger.info("=== STEP 7a: GRU + REGIME TRAINING ===")

    metrics = {}

    for model_name in ["regime", "gru"]:
        logger.info("--- Training %s ---", model_name)
        result = run_retrain(model_name)
        metrics[model_name] = result

        if result.get("error"):
            logger.error("Model %s failed: %s", model_name, result["error"])
            with open(ML_LOGS / f"retrain_{model_name}_error.log", "w") as f:
                f.write(str(result))
        else:
            logger.info("Model %s: SUCCESS", model_name)

    # Collect artifacts
    collect_model_artifacts()
    history = collect_metrics()
    metrics["history"] = history

    # Save training summary
    summary = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "models": metrics,
        "journal_path": str(JOURNAL_PATH),
    }
    with open(ML_METRICS / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n=== TRAINING COMPLETE ===")
    for model_name, result in metrics.items():
        if model_name == "history":
            continue
        status = "SUCCESS" if result.get("success") else f"FAILED: {result.get('error', '')[:80]}"
        print(f"  {model_name}: {status}")

    return metrics


if __name__ == "__main__":
    main()
