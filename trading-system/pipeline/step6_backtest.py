#!/usr/bin/env python3
"""
Step 6: Backtest + Reinforced Training Loop

Blind-backtest pipeline (3 rounds):
  Round 1 — train window (data_start → val_end):   val set as the "seen" evaluation window
  Round 2 — test window  (test_start → test_end):  fully blind OOS evaluation
  Round 3 — last 3yr     (3yr_before_end → end):   post-incremental-retrain evaluation

Control via BT_WINDOW env var:
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
import json
import logging
import os
import subprocess
import sys
import shutil
from datetime import datetime, timezone
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

for d in [BT_RESULTS, BT_LOGS, ENGINE_DIR / "logs"]:
    d.mkdir(parents=True, exist_ok=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ENGINE_DIR)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    return env


def _load_split_summary() -> dict:
    split_path = _ENV["ml_training"] / "datasets" / "split_summary.json"
    if not split_path.exists():
        logger.error(
            "split_summary.json not found at %s — run step5_split.py first", split_path
        )
        sys.exit(1)
    return json.loads(split_path.read_text())


def _resolve_bt_window(bt_window: str) -> tuple[str, str]:
    """
    Return (bt_start, bt_end) for the requested backtest window.

    round1 — val window: val_start → val_end   (test set protected)
    round2 — test window: test_start → test_end (blind OOS)
    round3 — last 3yr: (test_end - 3yr) → test_end (post-incremental-retrain eval)
    """
    if not HIST_DIR.exists() or not list(HIST_DIR.glob("*_15M.parquet")):
        logger.error("processed_data/histdata/ is empty — run step0_resample.py first")
        sys.exit(1)

    summary = _load_split_summary()
    dr = summary["date_ranges"]

    val_start  = dr["validation"]["start"][:10]
    val_end    = dr["validation"]["end"][:10]
    test_start = dr["test"]["start"][:10]
    test_end   = dr["test"]["end"][:10]

    if bt_window == "round2":
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
            "Unknown BT_WINDOW=%r — must be one of: round1, round2, round3", bt_window
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
        "--start", bt_start,
        "--end", bt_end,
    ]
    logger.info("Round %d — running backtest: %s → %s (ml_trader, shared ML cache)",
                round_num, bt_start, bt_end)

    try:
        # stdout streams directly to terminal — capture_output=True deadlocks when
        # backtest verbose output (5 traders × 15yr × 11 symbols) exceeds the 64 KB
        # OS pipe buffer. stderr is captured separately (only errors, stays small).
        result = subprocess.run(
            cmd,
            cwd=str(ENGINE_DIR),
            env=_build_env(),
            stdout=None,          # inherit parent stdout — no pipe, no deadlock
            stderr=subprocess.PIPE,
            text=True,
            timeout=14400,  # 4 hours — 2021–2024 range, sequential, QualityScorer per-signal
        )
        if result.returncode != 0:
            logger.error("Backtest failed (rc=%d):\n%s", result.returncode, result.stderr[-3000:])
            return {"error": result.stderr[-2000:], "returncode": result.returncode}

        if result.stderr:
            logger.debug("Backtest stderr:\n%s", result.stderr[-1000:])

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

    avg_wr = np.mean([r.get("win_rate", 0.0) for r in results.values()])
    avg_pf = np.mean([r.get("profit_factor", 0.0) for r in results.values()])
    avg_sh = np.mean([r.get("sharpe", 0.0) for r in results.values()])
    traders = {}
    for tid, m in results.items():
        traders[tid] = {
            "trades": m.get("trades", 0),
            "win_rate": round(m.get("win_rate", 0.0), 4),
            "profit_factor": round(m.get("profit_factor", 0.0), 3),
            "total_return": round(m.get("total_return", 0.0), 4),
            "max_drawdown": round(m.get("max_drawdown", 0.0), 4),
            "sharpe": round(m.get("sharpe", 0.0), 3),
        }
    return {
        "total_trades": total_trades,
        "avg_win_rate": round(float(avg_wr), 4),
        "avg_profit_factor": round(float(avg_pf), 3),
        "avg_sharpe": round(float(avg_sh), 3),
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

    with open(JOURNAL_PATH, "a") as jf, open(JOURNAL_CSV, "a") as cf:
        # Write CSV header only on first write
        if JOURNAL_CSV.stat().st_size == 0:
            cf.write("timestamp,trader,symbol,side,size,entry,stop_loss,take_profit,"
                     "rr_ratio,confidence,pnl,commission\n")

        for tr in trade_log:
            trader  = tr.get("trader_id", "trader_1")
            symbol  = tr.get("symbol", "EURUSD")
            side    = tr.get("side", "buy")
            entry   = float(tr.get("entry", 1.0))
            sl      = float(tr.get("sl", entry * 0.999))
            tp      = float(tr.get("tp1", entry * 1.002))
            size    = float(tr.get("size", 0.1))
            pnl     = float(tr.get("pnl", 0.0))
            rr      = float(tr.get("rr_ratio", 1.5))
            conf    = float(tr.get("confidence", 0.7))
            ts_str  = str(tr.get("entry_time", tr.get("timestamp",
                          datetime.now(timezone.utc).isoformat())))
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
            trader_idx = int(trader.split("_")[1]) - 1 if "_" in trader else 0  # 0-4

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

            record = {
                "timestamp":       ts_str,
                "exit_timestamp":  str(tr.get("exit_time", ts_str)),
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
                "source":          f"backtest_round_{round_num}",
                "correlation_id":  f"bt_r{round_num}_{written:06d}",
                "state_at_entry":  state_vec,
                "rl_action":       int(trader.split("_")[1]) if "_" in trader else 1,
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
                    "atr":       float(tr.get("atr", 0.001)),
                    "adx":       float(tr.get("adx", 20.0)),
                    "ema_stack": float(tr.get("ema_stack", 0)),
                },
            }
            jf.write(json.dumps(record, default=str) + "\n")

            # CSV row
            cf.write(
                f"{ts_str},{trader},{symbol},{side},{size},"
                f"{round(entry,6)},{round(sl,6)},{round(tp,6)},"
                f"{round(rr,3)},{round(conf,3)},{pnl_net},{round(commission,4)}\n"
            )
            written += 1

    logger.info("Round %d: wrote %d journal entries (total in file: %d)",
                round_num, written,
                sum(1 for _ in open(JOURNAL_PATH)))
    return written


# ─── Retrain runner ───────────────────────────────────────────────────────────

def run_retrain(round_num: int) -> dict:
    """Retrain all models using the accumulated journal. Returns retrain results."""
    script = ENGINE_DIR / "scripts" / "retrain_incremental.py"
    if not script.exists():
        return {"error": f"retrain script not found: {script}"}

    # GRU excluded: trained on 7.4M sequences — fine-tuning on ~3k trades causes catastrophic forgetting.
    # Regime and quality warm-start from existing weights (low LR), building on what was learned
    # from 7 years of data rather than re-initialising from scratch each round.
    results = {}
    for model in ["regime", "quality", "rl"]:
        logger.info("Round %d — retraining %s...", round_num, model)
        cmd = [sys.executable, str(script), "--model", model]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(ENGINE_DIR),
                env=_build_env(),
                stdout=None,           # stream to terminal — avoids pipe-buffer deadlock
                stderr=subprocess.PIPE,
                text=True,
                timeout=7200,
            )
            if proc.returncode != 0:
                logger.error("Retrain %s failed (rc=%d):\n%s",
                             model, proc.returncode, proc.stderr[-2000:])
                results[model] = {"error": proc.stderr[-500:]}
            else:
                logger.info("Retrain %s: OK", model)
                results[model] = {"success": True}
        except subprocess.TimeoutExpired:
            logger.error("Retrain %s timed out", model)
            results[model] = {"error": "timeout"}
        except Exception as exc:
            logger.error("Retrain %s exception: %s", model, exc)
            results[model] = {"error": str(exc)}
        finally:
            gc.collect()

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    bt_window = os.getenv("BT_WINDOW", "").lower().strip()
    if not bt_window:
        logger.error("BT_WINDOW env var not set — must be one of: round1, round2, round3")
        sys.exit(1)
    logger.info("=== STEP 6: BACKTEST + REINFORCED TRAINING (%d rounds, window=%s) ===",
                N_ROUNDS, bt_window)

    bt_start, bt_end = _resolve_bt_window(bt_window)

    # Clear journal before round 1 so we accumulate only backtest-sourced trades
    if JOURNAL_PATH.exists():
        JOURNAL_PATH.unlink()
        logger.info("Cleared existing journal for fresh reinforced training run")
    if JOURNAL_CSV.exists():
        JOURNAL_CSV.unlink()

    round_metrics: list[dict] = []

    for round_num in range(1, N_ROUNDS + 1):
        sep = "=" * 64
        logger.info("%s\n  ROUND %d / %d\n%s", sep, round_num, N_ROUNDS, sep)

        # ── 1. Backtest ───────────────────────────────────────────────────────
        bt_result = run_backtest(bt_start, bt_end, round_num)
        if bt_result.get("error"):
            logger.error("Round %d backtest failed — stopping reinforcement loop: %s",
                         round_num, bt_result["error"])
            with open(BT_LOGS / f"backtest_r{round_num}_error.log", "w") as f:
                f.write(str(bt_result))
            break

        latest_json = _latest_backtest_json()
        if latest_json is None:
            logger.error("Round %d: no backtest JSON found", round_num)
            break

        # Copy to named round file
        round_json = BT_RESULTS / f"backtest_round_{round_num}.json"
        shutil.copy2(latest_json, round_json)

        summary = _summarise(round_json)
        summary["round"] = round_num
        round_metrics.append(summary)

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
                "  %s: %d trades | WR=%.1f%% | PF=%.2f | Return=%.1f%% | DD=%.1f%% | Sharpe=%.2f",
                tid, m["trades"], m["win_rate"]*100, m["profit_factor"],
                m["total_return"]*100, m["max_drawdown"]*100, m["sharpe"],
            )

        # ── 1b. Run diagnostics on this round's backtest ──────────────────────
        try:
            _diag_script = ENGINE_DIR / "scripts" / "analyze_backtest.py"
            subprocess.run(
                [sys.executable, str(_diag_script), "--file", str(round_json)],
                cwd=str(ENGINE_DIR),
                timeout=120,
            )
        except Exception as _de:
            logger.warning("Round %d diagnostics failed (non-fatal): %s", round_num, _de)

        # ── 2. Convert trade log → journal ────────────────────────────────────
        n_written = trade_log_to_journal(round_json, round_num)
        if n_written == 0:
            logger.warning("Round %d: no trades to journal — skipping retrain", round_num)
            continue

        # ── 3. Retrain on accumulated journal ─────────────────────────────────
        if round_num < N_ROUNDS:
            # Retrain before next round so the next backtest uses improved models
            retrain_result = run_retrain(round_num)
            any_failed = any(r.get("error") for r in retrain_result.values())
            if any_failed:
                logger.warning("Round %d: some models failed to retrain — continuing anyway", round_num)
        else:
            # Final round: still retrain so weights are ready for live trading
            logger.info("Round %d (final): retraining after last backtest...", round_num)
            retrain_result = run_retrain(round_num)

    # ── Save improvement log ──────────────────────────────────────────────────
    if round_metrics:
        improvement_log = {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "n_rounds": len(round_metrics),
            "bt_start": bt_start,
            "bt_end": bt_end,
            "rounds": round_metrics,
        }

        # Compute metric deltas round-over-round
        if len(round_metrics) > 1:
            improvement_log["improvement"] = {
                "win_rate_delta":      round(round_metrics[-1]["avg_win_rate"]      - round_metrics[0]["avg_win_rate"],      4),
                "profit_factor_delta": round(round_metrics[-1]["avg_profit_factor"] - round_metrics[0]["avg_profit_factor"], 3),
                "sharpe_delta":        round(round_metrics[-1]["avg_sharpe"]        - round_metrics[0]["avg_sharpe"],        3),
            }
            imp = improvement_log["improvement"]
            logger.info(
                "Improvement round 1 → %d: WR %+.1f%% | PF %+.3f | Sharpe %+.3f",
                len(round_metrics),
                imp["win_rate_delta"] * 100,
                imp["profit_factor_delta"],
                imp["sharpe_delta"],
            )

        with open(BT_RESULTS / "reinforcement_log.json", "w") as f:
            json.dump(improvement_log, f, indent=2, default=str)

        # latest_summary.json = last round summary (what run_pipeline.py checks)
        with open(BT_RESULTS / "latest_summary.json", "w") as f:
            json.dump(round_metrics[-1], f, indent=2, default=str)

    # ── Print final table ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  BACKTEST + REINFORCED TRAINING COMPLETE  ({len(round_metrics)} rounds)")
    print(f"{'='*70}")
    print(f"  {'Round':<8} {'Trades':>7} {'WR':>8} {'PF':>7} {'Sharpe':>8}")
    print(f"  {'-'*42}")
    for m in round_metrics:
        print(f"  Round {m['round']:<3}  {m['total_trades']:>7}  "
              f"{m['avg_win_rate']*100:>7.1f}%  {m['avg_profit_factor']:>7.3f}  "
              f"{m['avg_sharpe']:>8.3f}")
    print()

    if len(round_metrics) > 1:
        imp = improvement_log.get("improvement", {})
        wr_d  = imp.get("win_rate_delta", 0) * 100
        pf_d  = imp.get("profit_factor_delta", 0)
        sh_d  = imp.get("sharpe_delta", 0)
        print(f"  Net improvement (round 1 → {len(round_metrics)}):")
        print(f"    Win rate:      {wr_d:+.1f}%")
        print(f"    Profit factor: {pf_d:+.3f}")
        print(f"    Sharpe:        {sh_d:+.3f}")
        print()


if __name__ == "__main__":
    main()
