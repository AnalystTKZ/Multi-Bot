#!/usr/bin/env python3
"""
<<<<<<< HEAD
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
=======
retrain_scheduler.py — Pure Python weekly scheduler.

Fires retrain_incremental.py every Sunday at 02:00 UTC.
Runs as the model-retrainer container entrypoint.
>>>>>>> c4064229d51d2ab2277d986e3e1dcc6150d219ea
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
<<<<<<< HEAD
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
=======

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scheduler")

RETRAIN_DAY = os.environ.get("RETRAIN_DAY", "sunday").lower()
RETRAIN_HOUR = int(os.environ.get("RETRAIN_HOUR", "2"))
CHECK_INTERVAL_SECONDS = 3600  # check every hour
>>>>>>> c4064229d51d2ab2277d986e3e1dcc6150d219ea

_DAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

<<<<<<< HEAD
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
=======

def _should_retrain(now: datetime) -> bool:
    """True if it's the scheduled retrain day and hour."""
    target_day = _DAY_MAP.get(RETRAIN_DAY, 6)
    return now.weekday() == target_day and now.hour == RETRAIN_HOUR


def _run_retrain() -> None:
    logger.info("Starting scheduled retrain (all models)")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrain_incremental.py")
    try:
        result = subprocess.run(
            [sys.executable, script, "--model", "all"],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode == 0:
            logger.info("Retrain succeeded:\n%s", result.stdout[-2000:])
        else:
            logger.error("Retrain failed (rc=%d):\n%s", result.returncode, result.stderr[-2000:])
    except subprocess.TimeoutExpired:
        logger.error("Retrain timed out after 1 hour")
    except Exception as exc:
        logger.error("Retrain error: %s", exc)


def main():
    logger.info(
        "Retrain scheduler started — will fire every %s at %02d:00 UTC",
        RETRAIN_DAY, RETRAIN_HOUR,
    )
    last_fired_day = -1

    while True:
        now = datetime.now(timezone.utc)

        if _should_retrain(now) and now.day != last_fired_day:
            _run_retrain()
            last_fired_day = now.day

        time.sleep(CHECK_INTERVAL_SECONDS)
>>>>>>> c4064229d51d2ab2277d986e3e1dcc6150d219ea


if __name__ == "__main__":
    main()
