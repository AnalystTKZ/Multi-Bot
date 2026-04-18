"""
candidate_logger.py — Logs ALL trade candidates (pre-execution) for statistical analysis.

Records every signal that passes rule-based checks BEFORE the EV / confidence / RL gates.
Outcomes are written back once a trade closes via mark_outcome().

File layout (CSV):
  logs/candidate_log.csv  — one row per candidate
  logs/candidate_log.parquet — same, rewritten on flush (optional)

Usage:
    logger = CandidateLogger()
    cid = logger.log_candidate(trader_id, symbol, side, features)
    ...later...
    logger.mark_outcome(cid, tp_hit=True, sl_hit=False, pnl=50.0)
"""

from __future__ import annotations

import csv
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_CSV_PATH = "logs/candidate_log.csv"

_FIELDNAMES = [
    "candidate_id",
    "trader_id",
    "symbol",
    "side",
    "timestamp",
    # Market state
    "atr",
    "adx",
    "ema_stack_score",
    "regime",
    # Trader-specific features
    "rsi",
    "pullback_depth",
    "candle_body_ratio",
    "range_size",
    "breakout_strength",
    "distance_from_mean_atr",
    "volatility_stable",
    "range_age_bars",
    "fakeout_prob",
    # Model scores
    "p_win",           # quality_score at candidate time
    "p_bull",
    "p_bear",
    "sentiment_score",
    "ensemble_score",
    "ev",
    "rr_ratio",
    "confidence",
    # Outcome (filled later)
    "tp_hit",
    "sl_hit",
    "pnl",
    "exit_reason",
    "outcome_ts",
    "executed",        # 1 = passed all gates and was executed
]


class CandidateLogger:
    """
    Thread-safe CSV candidate logger.
    All candidates are appended immediately; outcomes are rewritten on flush.
    """

    def __init__(self, csv_path: str = _CSV_PATH):
        self._path = csv_path
        self._lock = threading.Lock()
        self._pending: dict[str, dict] = {}   # candidate_id → row dict (awaiting outcome)
        os.makedirs(os.path.dirname(self._path) if os.path.dirname(self._path) else ".", exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not os.path.exists(self._path):
            with open(self._path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_FIELDNAMES, extrasaction="ignore")
                writer.writeheader()

    def log_candidate(
        self,
        trader_id: str,
        symbol: str,
        side: str,
        features: dict,
        executed: bool = False,
    ) -> str:
        """
        Append a candidate row.  Returns candidate_id for later outcome linking.

        features should contain as many of the feature keys as available:
          atr, adx, ema_stack_score, regime, rsi, pullback_depth,
          candle_body_ratio, range_size, breakout_strength,
          distance_from_mean_atr, volatility_stable, range_age_bars,
          fakeout_prob, p_win, p_bull, p_bear, sentiment_score,
          ensemble_score, ev, rr_ratio, confidence
        """
        cid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        row: dict = {k: "" for k in _FIELDNAMES}
        row.update({
            "candidate_id": cid,
            "trader_id": trader_id,
            "symbol": symbol,
            "side": side,
            "timestamp": now,
            "executed": 1 if executed else 0,
            # outcome columns default empty until mark_outcome
            "tp_hit": "",
            "sl_hit": "",
            "pnl": "",
            "exit_reason": "",
            "outcome_ts": "",
        })

        # Copy available feature values
        for key in _FIELDNAMES:
            if key in features and row[key] == "":
                v = features[key]
                row[key] = round(float(v), 6) if isinstance(v, (int, float)) else str(v)

        with self._lock:
            with open(self._path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_FIELDNAMES, extrasaction="ignore")
                writer.writerow(row)
            # Keep row in pending so we can rewrite outcome
            self._pending[cid] = row

        return cid

    def mark_outcome(
        self,
        candidate_id: str,
        tp_hit: bool,
        sl_hit: bool,
        pnl: float,
        exit_reason: str = "",
    ) -> None:
        """Update outcome for a previously logged candidate (appends an outcome row)."""
        with self._lock:
            if candidate_id not in self._pending:
                return
            self._pending[candidate_id].update({
                "tp_hit": 1 if tp_hit else 0,
                "sl_hit": 1 if sl_hit else 0,
                "pnl": round(pnl, 4),
                "exit_reason": exit_reason,
                "outcome_ts": datetime.now(timezone.utc).isoformat(),
            })
            # Rewrite as outcome row (append update marker row)
            outcome_row = dict(self._pending[candidate_id])
            outcome_row["candidate_id"] = f"{candidate_id}__outcome"
            with open(self._path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_FIELDNAMES, extrasaction="ignore")
                writer.writerow(outcome_row)

    def load_with_outcomes(self) -> "Optional[Any]":
        """
        Load the log as a pandas DataFrame, merging candidates with their outcomes.
        Returns None if pandas not available or file not found.
        """
        try:
            import pandas as pd
            if not os.path.exists(self._path):
                return pd.DataFrame(columns=_FIELDNAMES)

            df = pd.read_csv(self._path)
            # Split original rows from outcome rows
            outcomes = df[df["candidate_id"].str.endswith("__outcome")].copy()
            candidates = df[~df["candidate_id"].str.endswith("__outcome")].copy()

            if len(outcomes) == 0:
                return candidates

            # Map outcome back to candidate
            outcomes["candidate_id"] = outcomes["candidate_id"].str.replace("__outcome", "", regex=False)
            outcomes = outcomes.set_index("candidate_id")[["tp_hit", "sl_hit", "pnl", "exit_reason", "outcome_ts"]]
            candidates = candidates.set_index("candidate_id")
            candidates.update(outcomes)
            candidates = candidates.reset_index()
            return candidates
        except Exception as exc:
            logger.warning("CandidateLogger.load_with_outcomes failed: %s", exc)
            return None
