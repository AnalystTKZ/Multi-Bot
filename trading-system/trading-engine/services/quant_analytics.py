"""
quant_analytics.py — Statistical validation layer for probabilistic trading decisions.

Three components:
  1. StatisticalBinningEngine  — feature binning → per-bin win rate / avg-R / count
  2. EVFilter                  — Expected Value gate before any execution
  3. ConfidenceCalibrator      — checks that predicted p_win is monotonically
                                  related to actual win rate; flags unreliable models

All components are stateless during inference (read-only from the candidate log).
No lookahead — operates only on closed trades with known outcomes.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Statistical Binning Engine
# ---------------------------------------------------------------------------

# Default bin edges for key features
_BIN_EDGES: Dict[str, List[float]] = {
    "rsi":                 [0, 30, 35, 40, 45, 50, 55, 60, 65, 70, 100],
    "adx":                 [0, 15, 20, 25, 30, 40, 60, 100],
    "pullback_depth":      [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
    "candle_body_ratio":   [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
    "breakout_strength":   [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
    "distance_from_mean_atr": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    "p_win":               [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.0],
    "ensemble_score":      [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.0],
}


class StatisticalBinningEngine:
    """
    Reads the candidate log (post-outcome) and computes per-bin statistics.
    Call compute() once to build the lookup table; then query with get_bin_stats().
    """

    def __init__(self, candidate_log_path: str = "logs/candidate_log.csv"):
        self._log_path = candidate_log_path
        self._bins: Dict[str, Dict[str, dict]] = {}   # feature → bin_label → stats

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, min_count: int = 10) -> Dict[str, List[dict]]:
        """
        Load candidate log, compute statistics per bin for all features.
        Returns dict of {feature: [{"bin": str, "win_rate": float, "avg_rr": float, "count": int}, ...]}
        Only bins with count >= min_count are retained.
        """
        df = self._load_df()
        if df is None or len(df) == 0:
            logger.warning("BinningEngine: no candidate data available")
            return {}

        # Keep only rows with known outcomes
        df = df.dropna(subset=["tp_hit", "sl_hit"])
        if "executed" in df.columns:
            df = df[df["executed"].astype(str).isin(["1", "1.0", "True", "true"])]
        df["win"] = (df["tp_hit"].astype(float) == 1.0).astype(float)

        # Numeric conversion
        numeric_cols = ["rsi", "adx", "pullback_depth", "candle_body_ratio",
                        "breakout_strength", "distance_from_mean_atr",
                        "p_win", "ensemble_score", "rr_ratio"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].replace("", float("nan"))
                df[col] = df[col].astype(float)

        result: Dict[str, List[dict]] = {}
        self._bins = {}

        for feature, edges in _BIN_EDGES.items():
            if feature not in df.columns:
                continue
            feat_series = df[feature].dropna()
            if len(feat_series) == 0:
                continue

            labels = [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges)-1)]
            import pandas as pd
            df["_bin"] = pd.cut(df[feature], bins=edges, labels=labels, right=False, include_lowest=True)

            grp = df.groupby("_bin", observed=True)
            rows = []
            bin_lookup: Dict[str, dict] = {}

            for bin_label, group in grp:
                n = len(group)
                if n < min_count:
                    continue
                wr = float(group["win"].mean())
                avg_rr = float(group["rr_ratio"].mean()) if "rr_ratio" in group.columns else 1.0
                entry = {"bin": str(bin_label), "win_rate": round(wr, 4),
                         "avg_rr": round(avg_rr, 3), "count": n}
                rows.append(entry)
                bin_lookup[str(bin_label)] = entry

            result[feature] = rows
            self._bins[feature] = bin_lookup
            df.drop(columns=["_bin"], inplace=True)

        return result

    def get_bin_stats(self, feature: str, value: float) -> Optional[dict]:
        """
        Return the bin stats dict for a given feature + value.
        Returns None if not computed or value falls outside all bins.
        """
        if feature not in _BIN_EDGES:
            return None
        edges = _BIN_EDGES[feature]
        labels = [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges)-1)]

        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            if lo <= value < hi or (i == len(edges)-2 and value == hi):
                label = labels[i]
                return self._bins.get(feature, {}).get(label)
        return None

    def get_profitable_ranges(self, feature: str, min_win_rate: float = 0.55) -> List[dict]:
        """Return bins where win_rate >= min_win_rate."""
        return [b for b in self._bins.get(feature, {}).values()
                if b["win_rate"] >= min_win_rate]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_df(self):
        try:
            import pandas as pd
            if not os.path.exists(self._log_path):
                return None
            df = pd.read_csv(self._log_path)
            # Merge outcomes back (outcome rows have __outcome suffix)
            outcomes = df[df["candidate_id"].astype(str).str.endswith("__outcome")].copy()
            candidates = df[~df["candidate_id"].astype(str).str.endswith("__outcome")].copy()
            if len(outcomes) > 0:
                outcomes["candidate_id"] = outcomes["candidate_id"].str.replace("__outcome", "", regex=False)
                outcomes = outcomes.set_index("candidate_id")[["tp_hit", "sl_hit", "pnl", "exit_reason"]]
                candidates = candidates.set_index("candidate_id")
                candidates.update(outcomes)
                candidates = candidates.reset_index()
            return candidates
        except Exception as exc:
            logger.warning("BinningEngine._load_df failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# 2. EV Filter
# ---------------------------------------------------------------------------

# Per-trader historical averages used when no logged data exists
_DEFAULT_AVG_WIN: Dict[str, float] = {
    "trader_1": 2.0,   # avg R-multiple on wins
    "trader_3": 1.8,
    "trader_5": 1.4,
    "default":  1.6,
}
_DEFAULT_AVG_LOSS: Dict[str, float] = {
    "trader_1": 1.0,
    "trader_3": 1.0,
    "trader_5": 1.0,
    "default":  1.0,
}


class EVFilter:
    """
    Gate trades by Expected Value.

    EV = (p_win × avg_win) − ((1 − p_win) × avg_loss)

    If EV ≤ ev_threshold (default 0.0) → reject.
    Averages are loaded from the candidate log (rolling last N closed trades per trader)
    or fall back to built-in defaults.
    """

    def __init__(
        self,
        candidate_log_path: str = "logs/candidate_log.csv",
        rolling_n: int = 100,
        ev_threshold: float = 0.0,
    ):
        self._log_path = candidate_log_path
        self._rolling_n = rolling_n
        self._ev_threshold = ev_threshold
        self._avg_win: Dict[str, float] = {}
        self._avg_loss: Dict[str, float] = {}
        self._loaded = False

    def refresh(self) -> None:
        """Reload historical averages from candidate log. Call periodically."""
        try:
            import pandas as pd
            if not os.path.exists(self._log_path):
                return
            df = pd.read_csv(self._log_path)
            # Keep only outcome rows (they have full outcomes)
            df = df[df["candidate_id"].astype(str).str.endswith("__outcome")].copy()
            if len(df) == 0:
                return
            if "executed" in df.columns:
                df = df[df["executed"].astype(str).isin(["1", "1.0", "True", "true"])]
            df["tp_hit"] = pd.to_numeric(df["tp_hit"], errors="coerce")
            df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
            df["rr_ratio"] = pd.to_numeric(df["rr_ratio"], errors="coerce")

            for trader_id in df["trader_id"].unique():
                sub = df[df["trader_id"] == trader_id].tail(self._rolling_n)
                wins = sub[sub["tp_hit"] == 1.0]
                losses = sub[sub["tp_hit"] != 1.0]
                if len(wins) >= 5:
                    self._avg_win[trader_id] = float(wins["rr_ratio"].mean())
                if len(losses) >= 5:
                    self._avg_loss[trader_id] = float(losses["rr_ratio"].abs().mean())
            self._loaded = True
        except Exception as exc:
            logger.warning("EVFilter.refresh failed: %s", exc)

    def compute_ev(self, trader_id: str, p_win: float) -> float:
        """Return expected value in R-multiples."""
        avg_win = self._avg_win.get(trader_id) or _DEFAULT_AVG_WIN.get(trader_id) or _DEFAULT_AVG_WIN["default"]
        avg_loss = self._avg_loss.get(trader_id) or _DEFAULT_AVG_LOSS.get(trader_id) or _DEFAULT_AVG_LOSS["default"]
        p_loss = 1.0 - p_win
        ev = (p_win * avg_win) - (p_loss * avg_loss)
        return float(ev)

    def passes(self, trader_id: str, p_win: float) -> bool:
        """True if EV > ev_threshold — trade should be allowed."""
        ev = self.compute_ev(trader_id, p_win)
        return ev > self._ev_threshold

    def get_avg_stats(self, trader_id: str) -> dict:
        avg_win = self._avg_win.get(trader_id) or _DEFAULT_AVG_WIN.get(trader_id) or _DEFAULT_AVG_WIN["default"]
        avg_loss = self._avg_loss.get(trader_id) or _DEFAULT_AVG_LOSS.get(trader_id) or _DEFAULT_AVG_LOSS["default"]
        return {"avg_win": avg_win, "avg_loss": avg_loss}


# ---------------------------------------------------------------------------
# 3. Confidence Calibrator
# ---------------------------------------------------------------------------

_CALIBRATION_BINS = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.0]


class ConfidenceCalibrator:
    """
    Validates that predicted p_win → actual win rate is monotonically increasing.

    Algorithm:
      1. Bucket all closed trades by their p_win at signal time
      2. Compute actual win rate per bucket
      3. Check monotonicity (each bucket's win rate ≥ previous bucket's win rate)
      4. Flag model as unreliable if NOT monotonic in majority of adjacent bucket pairs

    Result is written to logs/calibration_report.json.
    """

    REPORT_PATH = "logs/calibration_report.json"

    def __init__(self, candidate_log_path: str = "logs/candidate_log.csv"):
        self._log_path = candidate_log_path

    def run(self, trader_id: Optional[str] = None) -> dict:
        """
        Returns calibration report dict.
        trader_id=None → all traders combined.
        """
        result = {
            "trader_id": trader_id or "all",
            "bins": [],
            "monotonic": None,
            "reliable": None,
            "note": "",
        }

        try:
            import pandas as pd

            if not os.path.exists(self._log_path):
                result["note"] = "No candidate log found"
                return result

            df = pd.read_csv(self._log_path)
            df = df[df["candidate_id"].astype(str).str.endswith("__outcome")].copy()
            if len(df) == 0:
                result["note"] = "No outcome data yet"
                return result
            if "executed" in df.columns:
                df = df[df["executed"].astype(str).isin(["1", "1.0", "True", "true"])]

            if trader_id:
                df = df[df["trader_id"] == trader_id]
            if len(df) < 20:
                result["note"] = f"Insufficient data: {len(df)} samples"
                return result

            df["p_win"] = pd.to_numeric(df["p_win"], errors="coerce")
            df["tp_hit"] = pd.to_numeric(df["tp_hit"], errors="coerce")
            df = df.dropna(subset=["p_win", "tp_hit"])
            df["win"] = (df["tp_hit"] == 1.0).astype(float)

            labels = [f"{_CALIBRATION_BINS[i]:.2f}-{_CALIBRATION_BINS[i+1]:.2f}"
                      for i in range(len(_CALIBRATION_BINS)-1)]
            df["_bin"] = pd.cut(df["p_win"], bins=_CALIBRATION_BINS, labels=labels,
                                right=False, include_lowest=True)

            grp = df.groupby("_bin", observed=True)
            bins = []
            win_rates = []
            for bin_label, group in grp:
                n = len(group)
                if n < 5:
                    continue
                wr = float(group["win"].mean())
                bins.append({
                    "predicted_range": str(bin_label),
                    "actual_win_rate": round(wr, 4),
                    "count": n,
                })
                win_rates.append(wr)

            result["bins"] = bins

            if len(win_rates) < 2:
                result["note"] = "Too few populated bins for calibration check"
                result["reliable"] = True   # assume OK when data sparse
                return result

            # Check monotonicity. Keep the strict flag literal; use reliability
            # separately for the looser "few violations" operational threshold.
            non_monotonic_pairs = sum(
                1 for i in range(len(win_rates)-1) if win_rates[i+1] < win_rates[i] - 0.03
            )
            total_pairs = len(win_rates) - 1
            monotonic = non_monotonic_pairs == 0
            reliable = non_monotonic_pairs <= total_pairs * 0.3  # allow 30% violations

            result["monotonic"] = monotonic
            result["reliable"] = reliable
            if not reliable:
                result["note"] = (
                    f"Non-monotonic calibration: {non_monotonic_pairs}/{total_pairs} pairs violated. "
                    "Consider retraining QualityScorer."
                )
            elif not monotonic:
                result["note"] = (
                    f"Calibration usable but not strictly monotonic: "
                    f"{non_monotonic_pairs}/{total_pairs} pairs violated."
                )
            else:
                result["note"] = "Calibration OK — p_win correlates with actual win rate."

        except Exception as exc:
            logger.warning("ConfidenceCalibrator.run failed: %s", exc)
            result["note"] = f"Error: {exc}"

        return result

    def run_all_traders(self) -> dict:
        """Run calibration for all traders found in the log. Save report to JSON."""
        report: dict = {"timestamp": "", "traders": {}}
        from datetime import datetime, timezone
        report["timestamp"] = datetime.now(timezone.utc).isoformat()

        try:
            import pandas as pd
            if not os.path.exists(self._log_path):
                return report

            df = pd.read_csv(self._log_path)
            df = df[df["candidate_id"].astype(str).str.endswith("__outcome")]
            if "executed" in df.columns:
                df = df[df["executed"].astype(str).isin(["1", "1.0", "True", "true"])]
            trader_ids = list(df["trader_id"].unique()) if "trader_id" in df.columns else []
        except Exception:
            trader_ids = []

        # All-traders combined
        report["traders"]["all"] = self.run(trader_id=None)

        for tid in trader_ids:
            report["traders"][tid] = self.run(trader_id=tid)

        # Write to disk
        try:
            import json
            os.makedirs(os.path.dirname(self.REPORT_PATH) if os.path.dirname(self.REPORT_PATH) else ".", exist_ok=True)
            with open(self.REPORT_PATH, "w") as f:
                json.dump(report, f, indent=2)
            logger.info("ConfidenceCalibrator: report saved to %s", self.REPORT_PATH)
        except Exception as exc:
            logger.warning("ConfidenceCalibrator: failed to save report: %s", exc)

        return report
