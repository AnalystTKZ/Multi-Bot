#!/usr/bin/env python3
"""
retrain_scheduler.py — Model retraining scheduler.

Schedule (all times UTC):
  Quality  — every Sunday 02:00          (weekly, adapts to recent trade outcomes)
  RL       — every Sunday 02:30          (weekly, after quality so it sees updated EV)
  Regime   — 1st Sunday of the month 03:00  (monthly, fine-tune on new 4H structure)
  GRU      — 1st Sunday of the month 03:30  (monthly, fine-tune on new sequences)

All models warm-start from existing weights — lower LR preserves learned structure.
Full cold-start retraining (step7a on Kaggle) should be triggered manually when:
  - New historical data added (3+ months)
  - Feature contract changed (SEQUENCE_FEATURES / REGIME_FEATURES modified)
  - Model accuracy degrades significantly in live monitoring

Env vars:
  RETRAIN_QUALITY_DAY    weekday name, default: sunday
  RETRAIN_QUALITY_HOUR   UTC hour,     default: 2
  RETRAIN_RL_HOUR        UTC hour,     default: 2  (offset by 30 min in practice — same hour)
  RETRAIN_MONTHLY_HOUR   UTC hour,     default: 3
  CHECK_INTERVAL         seconds,      default: 900 (check every 15 min)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("retrain_scheduler")

# ── Config ────────────────────────────────────────────────────────────────────

_HERE       = Path(__file__).resolve().parent
SCRIPT      = _HERE / "retrain_incremental.py"
JOURNAL     = _HERE.parent / "logs" / "trade_journal_detailed.jsonl"

WEEKLY_DAY   = os.environ.get("RETRAIN_QUALITY_DAY", "sunday").lower()
WEEKLY_HOUR  = int(os.environ.get("RETRAIN_QUALITY_HOUR", "2"))
MONTHLY_HOUR = int(os.environ.get("RETRAIN_MONTHLY_HOUR", "3"))
CHECK_SECS   = int(os.environ.get("CHECK_INTERVAL", "900"))

_DAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

MIN_JOURNAL_ENTRIES_QUALITY = int(os.environ.get("MIN_JOURNAL_QUALITY", "200"))
MIN_JOURNAL_ENTRIES_RL      = int(os.environ.get("MIN_JOURNAL_RL",      "500"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _journal_size() -> int:
    """Count entries in the live journal. Returns 0 if file missing."""
    try:
        return sum(1 for _ in open(JOURNAL))
    except FileNotFoundError:
        return 0


def _is_first_sunday(now: datetime) -> bool:
    """True if today is the first Sunday of the month."""
    return now.weekday() == 6 and now.day <= 7


def _is_weekly_window(now: datetime) -> bool:
    target = _DAY_MAP.get(WEEKLY_DAY, 6)
    return now.weekday() == target and now.hour == WEEKLY_HOUR


def _is_monthly_window(now: datetime) -> bool:
    return _is_first_sunday(now) and now.hour == MONTHLY_HOUR


def _run(model: str, label: str) -> bool:
    """Run retrain_incremental.py --model <model>. Returns True on success."""
    logger.info("=== Scheduled retrain: %s ===", label)
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--model", model],
            cwd=str(_HERE.parent),
            env={**os.environ, "PYTHONPATH": str(_HERE.parent)},
            timeout=7200,
        )
        if result.returncode == 0:
            logger.info("Retrain %s: OK", label)
            return True
        else:
            logger.error("Retrain %s: FAILED (rc=%d)", label, result.returncode)
            return False
    except subprocess.TimeoutExpired:
        logger.error("Retrain %s: TIMED OUT", label)
        return False
    except Exception as exc:
        logger.error("Retrain %s: ERROR — %s", label, exc)
        return False


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info(
        "Retrain scheduler started\n"
        "  Quality + RL : every %s at %02d:00 UTC (warm-start, weekly)\n"
        "  Regime + GRU : 1st Sunday of month at %02d:00 UTC (warm-start, monthly)\n"
        "  Journal path : %s\n"
        "  Check interval: %ds",
        WEEKLY_DAY, WEEKLY_HOUR, MONTHLY_HOUR, JOURNAL, CHECK_SECS,
    )

    last_weekly_day  = -1   # calendar day number of last weekly fire
    last_monthly_day = -1   # calendar day number of last monthly fire

    while True:
        now = datetime.now(timezone.utc)
        n_journal = _journal_size()

        # ── Weekly: Quality + RL ──────────────────────────────────────────────
        if _is_weekly_window(now) and now.day != last_weekly_day:
            last_weekly_day = now.day

            if n_journal >= MIN_JOURNAL_ENTRIES_QUALITY:
                _run("quality", "QualityScorer (weekly warm-start)")
            else:
                logger.info(
                    "Quality retrain skipped — only %d journal entries (need %d)",
                    n_journal, MIN_JOURNAL_ENTRIES_QUALITY,
                )

            if n_journal >= MIN_JOURNAL_ENTRIES_RL:
                _run("rl", "RLAgent (weekly warm-start)")
            else:
                logger.info(
                    "RL retrain skipped — only %d journal entries (need %d)",
                    n_journal, MIN_JOURNAL_ENTRIES_RL,
                )

        # ── Monthly: Regime + GRU ─────────────────────────────────────────────
        if _is_monthly_window(now) and now.day != last_monthly_day:
            last_monthly_day = now.day
            _run("regime", "RegimeClassifier (monthly warm-start)")
            _run("gru",    "GRU (monthly warm-start)")

        time.sleep(CHECK_SECS)


if __name__ == "__main__":
    main()
