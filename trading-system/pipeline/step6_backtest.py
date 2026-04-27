#!/usr/bin/env python3
"""
Step 6: Backtest + Reinforced Training Loop

Blind-backtest pipeline:
  Train   — train window (train_start → train_end): Quality/RL label source
  Round 1 — validation window:                      seen evaluation window
  Round 2 — test window  (test_start → test_end):   fully blind OOS evaluation
  Round 3 — last 3yr     (3yr_before_end → end):    post-incremental-retrain evaluation

Control via BT_WINDOW env var:
  BT_WINDOW=train    — train period only (Quality/RL label generation)
  BT_WINDOW=round1   — val period only   (default; protects test set)
  BT_WINDOW=round2   — test period only  (blind backtest)
  BT_WINDOW=round3   — last 3yr          (post-full-retrain evaluation)

Date windows are derived from split_summary.json produced by step5_split.py.

Data source: processed_data/histdata/{SYM}_{TF}.parquet (step0 outputs, full history).
No CSV intermediaries. Run step0_resample.py first if histdata/ is empty.

Output:
  backtesting/results/backtest_round_{N}.json    — raw backtest output per round
  backtesting/results/latest_summary.json        — final round summary
  backtesting/results/reinforcement_log.json     — per-round metric progression
  trading-engine/logs/trade_journal_detailed.jsonl — cumulative backtest journal
"""
from __future__ import annotations
import gc
import csv
import json
import logging
import os
import subprocess
import sys
import shutil
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("step6_backtest")

# Use env_config so paths resolve correctly on Kaggle vs local.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env_config import get_env
_ENV = get_env()

BASE         = _ENV["base"]
ENGINE_DIR   = _ENV["engine"]
HIST_DIR     = _ENV["processed"] / "histdata"
BT_DIR       = BASE / "backtesting"
BT_RESULTS   = BT_DIR / "results"
BT_LOGS      = BT_DIR / "logs"
JOURNAL_PATH = ENGINE_DIR / "logs" / "trade_journal_detailed.jsonl"
JOURNAL_CSV  = ENGINE_DIR / "logs" / "trade_journal.csv"
N_ROUNDS     = 3
JOURNAL_CSV_COLUMNS = [
    "run_id", "timestamp", "exit_timestamp", "trader", "symbol", "side",
    "size", "entry", "stop_loss", "take_profit", "rr_ratio", "confidence",
    "pnl", "commission", "exit_reason", "source", "source_split",
    "bt_start", "bt_end", "split_summary_hash", "correlation_id",
]

for d in [BT_RESULTS, BT_LOGS, ENGINE_DIR / "logs"]:
    d.mkdir(parents=True, exist_ok=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ENGINE_DIR)
    env["PYTHONUNBUFFERED"] = "1"
    # We parallelize by symbol at the script level, so cap inner BLAS/OpenMP
    # libraries to 1 thread to avoid 4 workers x 4 library threads oversubscription.
    _n = "1"
    env["OMP_NUM_THREADS"]          = _n
    env["OPENBLAS_NUM_THREADS"]     = _n
    env["MKL_NUM_THREADS"]          = _n
    env["NUMEXPR_NUM_THREADS"]      = _n
    env["VECLIB_MAXIMUM_THREADS"]   = _n
    env["ARROW_NUM_THREADS"]        = _n
    env["RAYON_NUM_THREADS"]        = _n
    env.setdefault("RETRAIN_CPU_WORKERS", _n)
    env.setdefault("MAX_PARALLEL_SYMBOL_LOADS", "4")
    env.setdefault("MAX_PARALLEL_CACHE_BUILDS", "4")
    # High-trade research mode is the default for fresh weight generation.
    # Set BACKTEST_ENFORCE_DAILY_HALT=1 when validating production promotion.
    env.setdefault("BACKTEST_ENFORCE_DAILY_HALT", "0")
    env.setdefault("BACKTEST_COMPOUND_EQUITY", "0")
    env.setdefault("MAX_DAILY_LOSS_PCT", "1.0")
    env.setdefault("MAX_DRAWDOWN_PCT", "1.0")
    env.setdefault("MAX_CONCURRENT_POSITIONS", "25")
    env.setdefault("PM_MIN_CONFIDENCE", "0.50")
    env.setdefault("ML_DIRECTION_THRESHOLD", "0.55")
    env.setdefault("NEUTRAL_BIAS_THRESHOLD", "0.60")
    env.setdefault("VOLATILE_ENTRY_THRESHOLD", "0.70")
    env.setdefault("BLOCK_LTF_CONSOLIDATING", "1")
    env.setdefault("RANGING_REQUIRE_RANGE", "1")
    env.setdefault("COOLDOWN_BARS", "1")
    return env


def _load_split_summary() -> dict:
    split_path = _ENV["ml_training"] / "datasets" / "split_summary.json"
    if not split_path.exists():
        logger.error(
            "split_summary.json not found at %s — run step5_split.py first", split_path
        )
        sys.exit(1)
    return json.loads(split_path.read_text())


def _split_summary_hash() -> str:
    split_path = _ENV["ml_training"] / "datasets" / "split_summary.json"
    try:
        return hashlib.sha256(split_path.read_bytes()).hexdigest()
    except Exception:
        return ""


def _source_split_for_round(round_num: int) -> str:
    if round_num == 0:
        return "train"
    if round_num == 1:
        return "validation"
    if round_num == 2:
        return "test"
    return "combined_eval"


def _run_split_for_round(round_num: int) -> str:
    if round_num == 0:
        return "train"
    return "test" if round_num in (2, 3) else "validation"


def _resolve_bt_window(bt_window: str) -> tuple[str, str]:
    """
    Return (bt_start, bt_end) for the requested backtest window.

    train  — train window: train_start → train_end (clean Quality/RL labels)
    round1 — val window: val_start → val_end   (test set protected)
    round2 — test window: test_start → test_end (blind OOS)
    round3 — last 3yr: (test_end - 3yr) → test_end (post-incremental-retrain eval)
    """
    if not HIST_DIR.exists() or not list(HIST_DIR.glob("*_15M.parquet")):
        logger.error("processed_data/histdata/ is empty — run step0_resample.py first")
        sys.exit(1)

    summary = _load_split_summary()
    dr = summary["date_ranges"]

    train_start = dr["train"]["start"][:10]
    train_end   = dr["train"]["end"][:10]
    val_start   = dr["validation"]["start"][:10]
    val_end    = dr["validation"]["end"][:10]
    test_start = dr["test"]["start"][:10]
    test_end   = dr["test"]["end"][:10]

    if bt_window == "train":
        bt_start, bt_end = train_start, train_end
        logger.info("BT_WINDOW=train — train-window backtest: %s → %s (clean Quality/RL labels)", bt_start, bt_end)
    elif bt_window == "round2":
        bt_start, bt_end = test_start, test_end
        logger.info("BT_WINDOW=round2 — BLIND backtest: %s → %s (test set)", bt_start, bt_end)
    elif bt_window == "round3":
        from dateutil.relativedelta import relativedelta
        end_dt   = pd.Timestamp(test_end)
        bt_start = (end_dt - relativedelta(years=3)).strftime("%Y-%m-%d")
        bt_end   = test_end
        logger.info("BT_WINDOW=round3 — post-retrain eval: %s → %s (last 3yr)", bt_start, bt_end)
    elif bt_window == "round1":
        bt_start, bt_end = val_start, val_end
        logger.info("BT_WINDOW=round1 — val-window backtest: %s → %s (test set protected)", bt_start, bt_end)
    else:
        logger.error(
            "Unknown BT_WINDOW=%r — must be one of: train, round1, round2, round3", bt_window
        )
        sys.exit(1)

    return bt_start, bt_end


# ─── Backtest runner ──────────────────────────────────────────────────────────

def run_backtest(bt_start: str, bt_end: str, round_num: int) -> dict:
    """Run run_backtest.py (single ML trader) and return parsed results."""
    script = ENGINE_DIR / "scripts" / "run_backtest.py"
    if not script.exists():
        logger.error("run_backtest.py not found: %s", script)
        return {"error": "script not found"}

    cmd = [
        sys.executable, str(script),
        "--split", _run_split_for_round(round_num),
        "--window-label", _source_split_for_round(round_num),
        "--start", bt_start,
        "--end", bt_end,
    ]
    logger.info("Round %d — running backtest: %s → %s (ml_trader, shared ML cache)",
                round_num, bt_start, bt_end)

    try:
        # Both stdout and stderr inherit from parent — no pipe, no deadlock, no
        # silent capture. run_backtest.py writes its logging to stderr AND to its
        # own timestamped log file (trading-engine/logs/backtest_*.log), so every
        # diagnostic line is visible in the notebook and persisted to disk.
        result = subprocess.run(
            cmd,
            cwd=str(ENGINE_DIR),
            env=_build_env(),
            stdout=None,   # inherit
            stderr=None,   # inherit — backtest logs are visible immediately
            timeout=14400,
        )
        if result.returncode != 0:
            logger.error("Backtest failed (rc=%d) — check trading-engine/logs/backtest_*.log",
                         result.returncode)
            return {"error": f"backtest exited {result.returncode}", "returncode": result.returncode}

        return {"success": True, "returncode": 0}

    except subprocess.TimeoutExpired:
        logger.error("Backtest timed out")
        return {"error": "timeout"}
    except Exception as exc:
        logger.error("Backtest exception: %s", exc)
        return {"error": str(exc)}


def _latest_backtest_json() -> Path | None:
    """Return the most recently written backtest JSON from trading-engine/backtest_results/."""
    engine_bt = ENGINE_DIR / "backtest_results"
    if not engine_bt.exists():
        return None
    jsons = sorted(engine_bt.glob("backtest_*.json"))
    return jsons[-1] if jsons else None


def _summarise(result_path: Path) -> dict:
    """Parse backtest JSON → aggregate metrics."""
    with open(result_path) as f:
        data = json.load(f)

    results = data.get("results", {})
    total_trades = sum(r.get("trades", 0) for r in results.values())
    if not results:
        return {"total_trades": 0, "avg_win_rate": 0.0, "avg_profit_factor": 0.0, "avg_sharpe": 0.0, "traders": {}}

    def _primary(m: dict) -> dict:
        return m.get("primary_metrics") or {
            "basis": "legacy_pnl",
            "profit_factor": m.get("profit_factor", 0.0),
            "total_return": m.get("total_return", 0.0),
            "max_drawdown": m.get("max_drawdown", 0.0),
            "sharpe": m.get("sharpe", 0.0),
            "expectancy_r": m.get("expectancy_r", 0.0),
        }

    avg_wr = np.mean([r.get("win_rate", 0.0) for r in results.values()])
    avg_pf = np.mean([_primary(r).get("profit_factor", 0.0) for r in results.values()])
    avg_sh = np.mean([_primary(r).get("sharpe", 0.0) for r in results.values()])
    traders = {}
    for tid, m in results.items():
        pm = _primary(m)
        traders[tid] = {
            "trades": m.get("trades", 0),
            "win_rate": round(m.get("win_rate", 0.0), 4),
            "metric_basis": m.get("primary_metric_basis", pm.get("basis", "legacy_pnl")),
            "profit_factor": round(pm.get("profit_factor", 0.0), 3),
            "total_return": round(pm.get("total_return", 0.0), 4),
            "max_drawdown": round(pm.get("max_drawdown", 0.0), 4),
            "sharpe": round(pm.get("sharpe", 0.0), 3),
            "expectancy_r": round(pm.get("expectancy_r", 0.0), 4),
        }
    return {
        "total_trades": total_trades,
        "avg_win_rate": round(float(avg_wr), 4),
        "avg_profit_factor": round(float(avg_pf), 3),
        "avg_sharpe": round(float(avg_sh), 3),
        "primary_metric_basis": "fixed_risk_r_multiple",
        "traders": traders,
    }


# ─── Journal conversion ───────────────────────────────────────────────────────

def trade_log_to_journal(result_path: Path, round_num: int) -> int:
    """
    Convert trade_log entries from a backtest JSON to JSONL journal format
    that retrain_incremental.py consumes. Appends to the existing journal
    so each round accumulates more signal history.

    Returns number of entries written.
    """
    with open(result_path) as f:
        data = json.load(f)

    trade_log = data.get("trade_log", [])
    if not trade_log:
        logger.warning("Round %d: trade_log is empty — nothing to journal", round_num)
        return 0

    rng = np.random.default_rng(round_num)
    written = 0
    source_split = _source_split_for_round(round_num)
    split_hash = _split_summary_hash()
    bt_start = str(data.get("start", ""))
    bt_end = str(data.get("end", ""))
    run_id = str(data.get("run_id") or f"round{round_num}_{Path(result_path).stem}")

    with open(JOURNAL_PATH, "a") as jf, open(JOURNAL_CSV, "a", newline="") as cf:
        csv_writer = csv.DictWriter(cf, fieldnames=JOURNAL_CSV_COLUMNS)
        # Write CSV header only on first write
        if JOURNAL_CSV.stat().st_size == 0:
            csv_writer.writeheader()

        for tr in trade_log:
            trader  = tr.get("trader", tr.get("trader_id", "ml_trader"))
            symbol  = tr.get("symbol", "EURUSD")
            side    = tr.get("side", "buy")
            entry   = float(tr.get("entry", 1.0))
            sl      = float(tr.get("stop_loss", tr.get("sl", entry * 0.999)))
            tp      = float(tr.get("tp1", entry * 1.002))
            size    = float(tr.get("size", 0.1))
            pnl     = float(tr.get("pnl", 0.0))
            rr      = float(tr.get("rr_ratio", 1.5))
            conf    = float(tr.get("confidence", 0.7))
            ts_str  = str(tr.get("entry_time", tr.get("timestamp",
                          datetime.now(timezone.utc).isoformat())))
            exit_ts = str(tr.get("exit_time", tr.get("exit_timestamp", ts_str)))
            try:
                ts_parsed = pd.Timestamp(ts_str)
                if ts_parsed.year < 2000:
                    raise ValueError(f"timestamp before 2000: {ts_str}")
            except Exception as exc:
                raise RuntimeError(
                    f"Round {round_num}: refusing to journal invalid trade timestamp "
                    f"{ts_str!r} from {result_path}"
                ) from exc
            # Preserve granular exit reason — _compute_ev_label uses tiered EV values.
            # be_or_trail → partial win (trailed after TP1), tp1 → TP1 hit, tp2 → full TP.
            exit_reason = str(tr.get("exit_reason", "sl")).lower()

            commission = abs(pnl) * 0.001
            pnl_net    = round(pnl - commission, 4)

            # Build real 43-dim RL state from backtest trade fields
            # HTF bias encoding: BIAS_UP=0, BIAS_DOWN=1, BIAS_NEUTRAL=2 → /2.0
            regime_map = {
                "BIAS_UP": 0.0, "BIAS_DOWN": 1.0, "BIAS_NEUTRAL": 2.0,
                # LTF classes also map (approximate)
                "TRENDING": 0.0, "RANGING": 2.0, "VOLATILE": 2.0, "CONSOLIDATING": 2.0,
            }
            _INSTRUMENT_IDX = {"EURUSD": 0, "GBPUSD": 1, "USDJPY": 2, "XAUUSD": 3}
            trader_idx = int(trader.split("_")[1]) - 1 if trader.startswith("trader_") else 0  # 0-4

            state_vec = [0.0] * 43
            # [0-5] ML predictions
            state_vec[0] = float(tr.get("p_bull", 0.5))
            state_vec[1] = float(tr.get("p_bear", 0.5))
            state_vec[2] = float(np.clip(conf, 0.0, 1.0))            # entry_depth proxy
            state_vec[3] = float(regime_map.get(tr.get("regime", "BIAS_NEUTRAL"), 2.0) / 2.0)
            state_vec[4] = 0.0                                        # sentiment — unavailable
            state_vec[5] = float(tr.get("quality_score", 0.5))
            # [6-13] Market structure
            state_vec[6]  = float(np.clip(tr.get("adx", 20.0), 0, 100) / 100.0)
            state_vec[7]  = float(np.clip(tr.get("ema_stack", 0), -2, 2) / 2.0)
            _atr  = float(tr.get("atr", 0.001))
            _close = float(tr.get("entry", 1.0))
            state_vec[8]  = float(np.clip(_atr / (_close + 1e-9) * 1000, 0, 10))
            state_vec[9]  = float(tr.get("bb_width", 0.0))
            state_vec[10] = float(bool(tr.get("bos_bull", False)))
            state_vec[11] = float(bool(tr.get("bos_bear", False)))
            state_vec[12] = float(bool(tr.get("fvg_bull", False)))
            state_vec[13] = float(bool(tr.get("fvg_bear", False)))
            # [14-18] Session context from timestamp
            state_vec[14] = float(2 <= pd.Timestamp(ts_str).hour < 7)
            state_vec[15] = float(7 <= pd.Timestamp(ts_str).hour < 12)
            state_vec[16] = float(13 <= pd.Timestamp(ts_str).hour < 18)
            state_vec[17] = float(pd.Timestamp(ts_str).hour == 12)
            state_vec[18] = 0.0                                       # news_proximity
            # [19-23] Strategy signal one-hot
            if 0 <= trader_idx <= 4:
                state_vec[19 + trader_idx] = 1.0
            # [24-29] Portfolio state
            state_vec[24] = 0.0                                       # open_positions
            state_vec[25] = float(np.clip(tr.get("drawdown", 0.0), 0, 0.2) / 0.2)
            state_vec[26] = 0.0                                       # daily_pnl
            state_vec[27] = 0.0                                       # trades_today
            state_vec[28] = 0.0                                       # last_result
            state_vec[29] = float(np.clip(tr.get("equity_pct", 1.0), 0, 2))
            # [30-33] Instrument one-hot
            inst_idx = _INSTRUMENT_IDX.get(symbol, -1)
            if 0 <= inst_idx <= 3:
                state_vec[30 + inst_idx] = 1.0
            # [34-41] ATR lag ratios — not available in backtest, default neutral
            for _i in range(8):
                state_vec[34 + _i] = 1.0
            state_vec = [float(np.clip(v, -10.0, 10.0)) for v in state_vec]

            # JSONL entry (full format compatible with retrain_incremental)
            entry_session = "london"
            try:
                h = pd.Timestamp(ts_str).hour
                if 0 <= h < 7:
                    entry_session = "asian"
                elif 13 <= h < 18:
                    entry_session = "ny"
            except Exception:
                pass

            source = "backtest_train" if round_num == 0 else f"backtest_round_{round_num}"
            correlation_id = str(tr.get("correlation_id") or f"bt_r{round_num}_{written:06d}")
            trade_source_split = str(tr.get("source_split") or source_split)
            trade_split_hash = str(tr.get("split_summary_hash") or split_hash)
            trade_run_id = str(tr.get("run_id") or run_id)

            record = {
                "run_id":          trade_run_id,
                "timestamp":       ts_str,
                "exit_timestamp":  exit_ts,
                "trader":          trader,
                "symbol":          symbol,
                "side":            side,
                "size":            size,
                "entry":           round(entry, 6),
                "stop_loss":       round(sl, 6),
                "take_profit":     round(tp, 6),
                "rr_ratio":        round(rr, 3),
                "confidence":      round(conf, 3),
                "pnl":             pnl_net,
                "commission":      round(commission, 4),
                "exit_reason":     exit_reason,
                "strategy":        trader,
                "session":         entry_session,
                "source":          source,
                "source_split":    trade_source_split,
                "bt_start":        bt_start,
                "bt_end":          bt_end,
                "split_summary_hash": trade_split_hash,
                "correlation_id":  correlation_id,
                "state_at_entry":  state_vec,
                "rl_action":       int(trader.split("_")[1]) if trader.startswith("trader_") else 1,
                "ml_model_scores": {
                    "p_bull":              float(tr.get("p_bull", 0.5)),
                    "p_bear":              float(tr.get("p_bear", 0.5)),
                    "quality_score":       float(tr.get("quality_score", 0.5)),
                    "regime":              str(tr.get("regime", "RANGING")),
                    "expected_variance":   float(tr.get("expected_variance", 0.1)),
                    "regime_duration":     float(tr.get("regime_duration", 0.5)),
                    "vol_slope":           float(tr.get("vol_slope", 0.0)),
                    "sentiment_score":     0.0,
                    "ensemble_score":      float(tr.get("confidence", 0.5)),
                    "sentiment_label":     "neutral",
                    "sentiment_backend":   "neutral",
                    "sentiment_confidence": 0.0,
                },
                "signal_metadata": {
                    "run_id":    trade_run_id,
                    "atr":       float(tr.get("atr", 0.001)),
                    "adx":       float(tr.get("adx", 20.0)),
                    "ema_stack": float(tr.get("ema_stack", 0)),
                    "directional_group": str(tr.get("directional_group", "")),
                    "same_timestamp_entries": int(tr.get("same_timestamp_entries", 1)),
                    "same_timestamp_group_entries": int(tr.get("same_timestamp_group_entries", 1)),
                },
            }
            jf.write(json.dumps(record, default=str) + "\n")

            # CSV row
            csv_writer.writerow({
                "run_id":             trade_run_id,
                "timestamp":          ts_str,
                "exit_timestamp":     exit_ts,
                "trader":             trader,
                "symbol":             symbol,
                "side":               side,
                "size":               size,
                "entry":              round(entry, 6),
                "stop_loss":          round(sl, 6),
                "take_profit":        round(tp, 6),
                "rr_ratio":           round(rr, 3),
                "confidence":         round(conf, 3),
                "pnl":                pnl_net,
                "commission":         round(commission, 4),
                "exit_reason":        exit_reason,
                "source":             source,
                "source_split":       trade_source_split,
                "bt_start":           bt_start,
                "bt_end":             bt_end,
                "split_summary_hash": trade_split_hash,
                "correlation_id":     correlation_id,
            })
            written += 1

    logger.info("Round %d: wrote %d journal entries (total in file: %d)",
                round_num, written,
                sum(1 for _ in open(JOURNAL_PATH)))
    return written


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    bt_window = os.getenv("BT_WINDOW", "").lower().strip()
    if not bt_window:
        logger.error("BT_WINDOW env var not set — must be one of: train, round1, round2, round3")
        sys.exit(1)
    logger.info("=== STEP 6: BACKTEST (%s) ===", bt_window)

    bt_start, bt_end = _resolve_bt_window(bt_window)
    round_num_map = {"train": 0, "round1": 1, "round2": 2, "round3": 3}
    round_num = round_num_map[bt_window]

    # Clear journal before train-label generation and before round 1 evaluation
    # so Quality/RL training data and validation/test diagnostics stay separate.
    if bt_window in {"train", "round1"} and JOURNAL_PATH.exists():
        JOURNAL_PATH.unlink()
        logger.info("Cleared existing journal for fresh %s run", bt_window)
    if bt_window in {"train", "round1"} and JOURNAL_CSV.exists():
        JOURNAL_CSV.unlink()

    sep = "=" * 64
    logger.info("%s\n  ROUND %d / %d\n%s", sep, round_num, N_ROUNDS, sep)

    bt_result = run_backtest(bt_start, bt_end, round_num)
    if bt_result.get("error"):
        logger.error("Round %d backtest failed: %s", round_num, bt_result["error"])
        with open(BT_LOGS / f"backtest_r{round_num}_error.log", "w") as f:
            f.write(str(bt_result))
        sys.exit(1)

    latest_json = _latest_backtest_json()
    if latest_json is None:
        logger.error("Round %d: no backtest JSON found", round_num)
        sys.exit(1)

    round_json = BT_RESULTS / f"backtest_round_{round_num}.json"
    shutil.copy2(latest_json, round_json)

    summary = _summarise(round_json)
    summary["round"] = round_num
    summary["bt_window"] = bt_window
    summary["bt_start"] = bt_start
    summary["bt_end"] = bt_end

    logger.info(
        "Round %d backtest — %d trades | avg WR=%.1f%% | avg PF=%.2f | avg Sharpe=%.2f",
        round_num,
        summary["total_trades"],
        summary["avg_win_rate"] * 100,
        summary["avg_profit_factor"],
        summary["avg_sharpe"],
    )
    for tid, m in summary["traders"].items():
        logger.info(
            "  %s: %d trades | WR=%.1f%% | fixed PF=%.2f | Return=%.1f%% | ExpR=%.3f | DD=%.1f%% | Sharpe=%.2f",
            tid, m["trades"], m["win_rate"]*100, m["profit_factor"],
            m["total_return"]*100, m.get("expectancy_r", 0.0), m["max_drawdown"]*100, m["sharpe"],
        )

    try:
        _diag_script = ENGINE_DIR / "scripts" / "analyze_backtest.py"
        subprocess.run(
            [sys.executable, str(_diag_script), "--file", str(round_json)],
            cwd=str(ENGINE_DIR),
            timeout=120,
        )
    except Exception as _de:
        logger.warning("Round %d diagnostics failed (non-fatal): %s", round_num, _de)

    n_written = trade_log_to_journal(round_json, round_num)
    if n_written == 0:
        logger.warning("Round %d: no trades to journal", round_num)

    with open(BT_RESULTS / "latest_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  BACKTEST COMPLETE  (round {round_num} / window={bt_window})")
    print(f"{'='*70}")
    print(f"  {'Round':<8} {'Trades':>7} {'WR':>8} {'PF*':>7} {'Sharpe*':>8}")
    print(f"  {'-'*42}")
    print(f"  Round {summary['round']:<3}  {summary['total_trades']:>7}  "
          f"{summary['avg_win_rate']*100:>7.1f}%  {summary['avg_profit_factor']:>7.3f}  "
          f"{summary['avg_sharpe']:>8.3f}")
    print()


if __name__ == "__main__":
    main()
