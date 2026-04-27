"""
trade_journal.py — Trade journal (complete replacement).

Contract 4 fields:
  CSV:  timestamp,trader,symbol,side,size,entry,stop_loss,take_profit,
        rr_ratio,confidence,pnl,commission
  JSONL: all CSV + strategy,timeframe,session,smc_score,ict_conditions,
         ml_model_scores,entry_reason,exit_reason,correlation_id,
         signal_metadata,state_at_entry,rl_action
"""

from __future__ import annotations

import csv
import json
import logging
import os
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

_CSV_PATH = "logs/trade_journal.csv"
_JSONL_PATH = "logs/trade_journal_detailed.jsonl"
_CSV_COLUMNS = [
    "run_id", "timestamp", "exit_timestamp", "trader", "symbol", "side",
    "size", "entry", "stop_loss", "take_profit", "rr_ratio", "confidence",
    "pnl", "commission", "exit_reason", "source", "source_split",
    "bt_start", "bt_end", "split_summary_hash", "correlation_id",
]


class TradeJournal:
    """Writes both CSV and JSONL journals. Provides rolling stats and RL episodes."""

    def __init__(self, rl_agent=None):
        self._rl_agent = rl_agent
        self._rolling: Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=200))
        os.makedirs("logs", exist_ok=True)
        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        if not os.path.exists(_CSV_PATH):
            with open(_CSV_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
                writer.writeheader()

    def log_trade(self, trade: dict) -> None:
        """Write to both CSV (clean) and JSONL (detailed)."""
        timestamp = trade.get("timestamp") or datetime.now(timezone.utc).isoformat()
        exit_timestamp = trade.get("exit_timestamp", "")
        trader = str(trade.get("trader") or trade.get("trader_id", ""))
        symbol = str(trade.get("symbol", ""))
        side = str(trade.get("side", ""))
        size = float(trade.get("size", 0.0))
        entry = float(trade.get("entry", 0.0))
        stop_loss = float(trade.get("stop_loss", 0.0))
        take_profit = float(trade.get("take_profit", 0.0))
        rr_ratio = float(trade.get("rr_ratio", 0.0))
        confidence = float(trade.get("confidence", 0.0))
        pnl = float(trade.get("pnl", 0.0))
        commission = float(trade.get("commission", 0.0))

        meta = trade.get("signal_metadata", {}) or {}
        state_at_entry = trade.get("state_at_entry", [0.0] * 42)
        if len(state_at_entry) != 42:
            state_at_entry = ([0.0] * 42)[:42]

        rl_action = int(meta.get("rl_action", 0))

        # ─── CSV ─────────────────────────────────────────────────────────────
        csv_row = {
            "run_id": str(trade.get("run_id") or meta.get("run_id", "")),
            "timestamp": timestamp,
            "exit_timestamp": exit_timestamp,
            "trader": trader,
            "symbol": symbol,
            "side": side,
            "size": size,
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "rr_ratio": rr_ratio,
            "confidence": confidence,
            "pnl": pnl,
            "commission": commission,
            "exit_reason": str(trade.get("exit_reason", "")),
            "source": str(trade.get("source") or meta.get("source", "")),
            "source_split": str(trade.get("source_split") or meta.get("source_split", "")),
            "bt_start": str(trade.get("bt_start") or meta.get("bt_start", "")),
            "bt_end": str(trade.get("bt_end") or meta.get("bt_end", "")),
            "split_summary_hash": str(
                trade.get("split_summary_hash") or meta.get("split_summary_hash", "")
            ),
            "correlation_id": str(trade.get("correlation_id", "")),
        }
        try:
            with open(_CSV_PATH, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
                writer.writerow(csv_row)
        except Exception as exc:
            logger.error("TradeJournal CSV write failed: %s", exc)

        # ─── JSONL ───────────────────────────────────────────────────────────
        jsonl_record = {
            # CSV fields
            "run_id": str(trade.get("run_id") or meta.get("run_id", "")),
            "timestamp": timestamp,
            "exit_timestamp": exit_timestamp,
            "trader": trader,
            "symbol": symbol,
            "side": side,
            "size": size,
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "rr_ratio": rr_ratio,
            "confidence": confidence,
            "pnl": pnl,
            "commission": commission,
            # Extended fields (Contract 4)
            "strategy": str(meta.get("strategy", "")),
            "timeframe": str(trade.get("timeframe", "15M")),
            "session": str(meta.get("session", "")),
            "smc_score": float(meta.get("smc_score", 0.0)),
            "ict_conditions": meta.get("ict_conditions", {}),
            "ml_model_scores": {
                "p_bull": float(meta.get("p_bull", 0.5)),
                "p_bear": float(meta.get("p_bear", 0.5)),
                "quality_score": float(meta.get("quality_score", 0.5)),
                "ensemble_score": float(meta.get("ensemble_score", 0.0)),
                "regime": str(meta.get("regime", "")),
                "sentiment_score": float(meta.get("sentiment_score", 0.0)),
                "sentiment_label": str(meta.get("sentiment_label", "neutral")),
                "sentiment_backend": str(meta.get("sentiment_backend", "neutral")),
                "sentiment_confidence": float(meta.get("sentiment_confidence", 0.0)),
                "rl_action": rl_action,
            },
            "entry_reason": str(trade.get("entry_reason", "")),
            "exit_reason": str(trade.get("exit_reason", "")),
            "correlation_id": str(trade.get("correlation_id", "")),
            "signal_metadata": meta,
            "state_at_entry": state_at_entry,
            "rl_action": rl_action,
            "source": str(trade.get("source") or meta.get("source", "")),
            "source_split": str(trade.get("source_split") or meta.get("source_split", "")),
            "bt_start": str(trade.get("bt_start") or meta.get("bt_start", "")),
            "bt_end": str(trade.get("bt_end") or meta.get("bt_end", "")),
            "split_summary_hash": str(
                trade.get("split_summary_hash") or meta.get("split_summary_hash", "")
            ),
        }
        try:
            with open(_JSONL_PATH, "a") as f:
                f.write(json.dumps(jsonl_record) + "\n")
        except Exception as exc:
            logger.error("TradeJournal JSONL write failed: %s", exc)

        # Update rolling stats
        self._rolling[trader].append({
            "pnl": pnl,
            "rr_ratio": rr_ratio,
            "win": pnl > 0,
        })

        # Notify RL agent
        if self._rl_agent is not None:
            try:
                self._rl_agent.record_outcome({
                    "pnl": pnl,
                    "rr_ratio": rr_ratio,
                    "confidence": confidence,
                    "rl_action": rl_action,
                    "state_at_entry": state_at_entry,
                    "session": meta.get("session", ""),
                    "trades_today": 0,
                })
            except Exception as exc:
                logger.debug("TradeJournal: RL record_outcome failed: %s", exc)

    def get_rolling_stats(self, trader_id: str, n: int = 20) -> dict:
        """Returns rolling win_rate, profit_factor, avg_rr from last n closed trades."""
        trades = list(self._rolling[trader_id])[-n:]
        if not trades:
            return {"win_rate": 0.5, "profit_factor": 1.0, "avg_rr": 1.5}

        wins = [t for t in trades if t.get("win")]
        win_rate = len(wins) / len(trades)

        gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
        profit_factor = gross_profit / (gross_loss + 1e-9)

        rr_vals = [t.get("rr_ratio", 1.5) for t in trades]
        avg_rr = sum(rr_vals) / len(rr_vals)

        return {
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "avg_rr": round(avg_rr, 4),
        }

    def get_rl_episodes(self) -> List[dict]:
        """Returns all closed trades with state_at_entry for RL retraining."""
        episodes = []
        if not os.path.exists(_JSONL_PATH):
            return episodes
        try:
            with open(_JSONL_PATH) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if "state_at_entry" in rec:
                            episodes.append(rec)
                    except json.JSONDecodeError:
                        continue
        except Exception as exc:
            logger.error("get_rl_episodes failed: %s", exc)
        return episodes

    @staticmethod
    def _compute_rl_reward(pnl: float, rr_ratio: float, confidence: float) -> float:
        """Simple reward shaping used by old system — preserved for compatibility."""
        import numpy as np
        base = float(np.clip(pnl / 100.0, -3, 4))
        rr_bonus = float(np.clip((rr_ratio - 1.5) * 0.2, -0.5, 0.5))
        conf_penalty = -0.1 if confidence < 0.55 else 0.0
        return base + rr_bonus + conf_penalty
