#!/usr/bin/env python3
"""
Step 7b: Quality + RL training — runs after a real train/live/paper journal exists.
Requires trade_journal_detailed.jsonl with >= MIN_JOURNAL_STEP7B real entries
(default 50).
No synthetic fallback — if journal is missing or too small, step fails loudly.
Evaluation backtest journals are excluded by default so validation/test windows
remain clean. Set ALLOW_ROUND_JOURNAL_TRAINING=1 and widen
JOURNAL_ALLOWED_SPLITS only for research-only feedback loops.
"""
from __future__ import annotations
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("step7b_train")

# Use env_config so paths resolve correctly on Kaggle vs local.
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

JOURNAL_PATH         = ENGINE_DIR / "logs" / "trade_journal_detailed.jsonl"
MIN_JOURNAL_ENTRIES = int(os.getenv("MIN_JOURNAL_STEP7B", "50"))
ROUND_JOURNAL_SPLITS = os.getenv(
    "ROUND_JOURNAL_ALLOWED_SPLITS",
    "train,live,paper,production",
)


def _allowed_splits() -> set[str]:
    raw = os.getenv("JOURNAL_ALLOWED_SPLITS", ROUND_JOURNAL_SPLITS)
    return {s.strip().lower() for s in raw.split(",") if s.strip()}


def _record_allowed(record: dict, allowed: set[str]) -> bool:
    split = str(record.get("source_split", "")).strip().lower()
    return bool(split and split in allowed)


def run_retrain(model: str) -> dict:
    script = ENGINE_DIR / "scripts" / "retrain_incremental.py"
    if not script.exists():
        return {"error": f"script not found: {script}"}

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ENGINE_DIR)
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    env.setdefault("ALLOW_ROUND_JOURNAL_TRAINING", "0")
    env.setdefault("JOURNAL_ALLOWED_SPLITS", ROUND_JOURNAL_SPLITS)

    logger.info(
        "Running retrain --model %s with JOURNAL_ALLOWED_SPLITS=%s",
        model,
        env.get("JOURNAL_ALLOWED_SPLITS", ""),
    )
    try:
        # Stream output directly — capture_output=True causes pipe-buffer deadlock
        # when training verbose output fills the 64 KB OS pipe buffer.
        result = subprocess.run(
            [sys.executable, str(script), "--model", model],
            cwd=str(ENGINE_DIR),
            env=env,
            timeout=14400,
        )
        if result.returncode != 0:
            logger.error("retrain %s failed (exit %d)", model, result.returncode)
            return {"error": f"exit {result.returncode}", "model": model}
        return {"success": True, "model": model}
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "model": model}
    except Exception as e:
        return {"error": str(e), "model": model}


def main():
    logger.info("=== STEP 7b: QUALITY + RL TRAINING ===")

    # Require real journal from backtest — no synthetic fallback.
    # Exit 0 (not error) when the journal is missing or too small: this happens
    # on the very first run before any backtest trades exist. kaggle_train.py
    # treats a non-zero exit as a hard failure that aborts the pipeline, so we
    # must exit gracefully here rather than crashing the whole run.
    if not JOURNAL_PATH.exists() or JOURNAL_PATH.stat().st_size < 100:
        logger.warning(
            "Journal missing or empty at %s — backtest produced no trades yet. "
            "Skipping Quality+RL training (will train after first successful backtest).",
            JOURNAL_PATH,
        )
        sys.exit(0)

    allowed_splits = _allowed_splits()
    n_entries = 0
    n_allowed = 0
    with open(JOURNAL_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            n_entries += 1
            try:
                if _record_allowed(json.loads(line), allowed_splits):
                    n_allowed += 1
            except Exception:
                continue

    logger.info("Journal entries: %d total, %d allowed for training (%s)",
                n_entries, n_allowed, sorted(allowed_splits))
    if n_allowed < MIN_JOURNAL_ENTRIES:
        logger.warning(
            "Journal has only %d allowed entries (need %d) — not enough clean "
            "Quality/RL training data. Check step6 logs or collect live/paper data.",
            n_allowed, MIN_JOURNAL_ENTRIES,
        )
        sys.exit(0)

    metrics = {}
    for model_name in ["quality", "rl"]:
        logger.info("--- Training %s ---", model_name)
        result = run_retrain(model_name)
        metrics[model_name] = result

        if result.get("error"):
            logger.error("Model %s failed: %s", model_name, result["error"])
            with open(ML_LOGS / f"retrain_{model_name}_error.log", "w") as f:
                f.write(str(result))
        else:
            logger.info("Model %s: SUCCESS", model_name)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": "7b",
        "journal_entries": n_entries,
        "allowed_journal_entries": n_allowed,
        "allowed_splits": sorted(allowed_splits),
        "models": metrics,
    }
    out = ML_METRICS / "training_7b_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Step 7b complete — summary: %s", out)


if __name__ == "__main__":
    main()
