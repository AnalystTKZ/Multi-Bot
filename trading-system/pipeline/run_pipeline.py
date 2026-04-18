#!/usr/bin/env python3
"""
Master pipeline runner.

Step order (training before backtesting):
  1  Data Discovery          → processed_data/raw_inventory.json
  2  Data Cleaning           → processed_data/clean/{SYM}_15M.parquet
  3  Multi-Asset Alignment   → processed_data/aligned_multi_asset.parquet
  4  Feature Engineering     → processed_data/feature_engineered.parquet
  5  Train/Val/Test Split    → ml_training/datasets/{train,validation,test}.parquet
  6  Model Training          → trading-engine/weights/{regime,quality,rl_ppo,gru_lstm}
  7  Backtesting             → backtesting/results/latest_summary.json
  8  Validation + Critic     → ml_training/metrics/critic_report.json

Steps whose primary output already exists on disk are SKIPPED automatically.
Use --force to bypass skip logic and re-run every requested step.

Usage:
    python pipeline/run_pipeline.py                      # run all, skip completed
    python pipeline/run_pipeline.py --force              # run all, skip nothing
    python pipeline/run_pipeline.py --steps 4 5 6        # specific steps
    python pipeline/run_pipeline.py --start-from 6       # resume from step N
    python pipeline/run_pipeline.py --steps 7 --force    # force re-backtest
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pipeline")

BASE = Path(__file__).resolve().parent.parent
PIPELINE_DIR = BASE / "pipeline"
ENGINE_DIR   = BASE / "trading-engine"

# ─── Memory limit (4 GB virtual) ─────────────────────────────────────────────
# Applied per subprocess — prevents OOM kills on constrained hardware.
_MEM_LIMIT_KB = 4_000_000   # 4 GB


def _apply_memory_limit() -> None:
    """Set virtual memory limit in the current process (inherited by children)."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_limit = _MEM_LIMIT_KB * 1024
        if hard == resource.RLIM_INFINITY or new_limit <= hard:
            resource.setrlimit(resource.RLIMIT_AS, (new_limit, hard))
            logger.info("Memory limit set to %d MB", _MEM_LIMIT_KB // 1024)
        else:
            logger.warning("Cannot set memory limit above hard limit %d bytes", hard)
    except Exception as exc:
        logger.warning("Could not set memory limit: %s", exc)


def _clean_pipeline_outputs() -> None:
    """
    Remove derived pipeline outputs so the run starts fresh.
    Does NOT delete:
      - training_data/     — raw M1 inputs
      - processed_data/histdata/  — M1-resampled parquets (expensive, inputs unchanged)
    Use --force with step 2 to re-run step0_resample if M1 data changed.
    """
    # Remove downstream dirs but preserve histdata/
    for sub in ["clean", "aligned_multi_asset.parquet", "feature_engineered.parquet",
                "cleaning_summary.json", "alignment_summary.json",
                "histdata_resample_summary.json", "raw_inventory.json"]:
        path = BASE / "processed_data" / sub
        try:
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
        except Exception as exc:
            logger.warning("Cleanup failed for %s: %s", path, exc)

    for path in [
        BASE / "ml_training",
        BASE / "backtesting" / "results",
        ENGINE_DIR / "weights",
        ENGINE_DIR / "models" / "weights",
        ENGINE_DIR / "logs" / "retrain_history.jsonl",
        ENGINE_DIR / "logs" / "trade_journal_detailed.jsonl",
        ENGINE_DIR / "logs" / "trade_journal.csv",
    ]:
        try:
            if path.is_symlink() or path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
        except Exception as exc:
            logger.warning("Cleanup failed for %s: %s", path, exc)


# ─── Step registry ────────────────────────────────────────────────────────────
# Each entry:  (display_name, script_basename, done_check_paths)
# done_check_paths: list of Path objects — step is SKIPPED when ALL exist.
# An empty list means the step never auto-skips.

_STEPS: dict[int, tuple[str, str, list[Path]]] = {
    1: (
        "Data Discovery",
        "step1_inventory",
        [BASE / "processed_data" / "raw_inventory.json"],
    ),
    2: (
        "M1 Resample → All TFs",
        "step0_resample",
        [
            BASE / "processed_data" / "histdata" / "EURUSD_15M.parquet",
            BASE / "processed_data" / "histdata" / "EURUSD_1H.parquet",
            BASE / "processed_data" / "histdata" / "EURUSD_4H.parquet",
            BASE / "processed_data" / "histdata" / "EURUSD_1D.parquet",
            BASE / "processed_data" / "histdata" / "EURUSD_1W.parquet",
            BASE / "processed_data" / "histdata" / "EURUSD_1MN.parquet",
            BASE / "processed_data" / "histdata" / "XAUUSD_15M.parquet",
        ],
    ),
    3: (
        "Data Cleaning",
        "step2_clean",
        [
            BASE / "processed_data" / "clean" / "EURUSD_15M.parquet",
            BASE / "processed_data" / "clean" / "EURGBP_15M.parquet",
            BASE / "processed_data" / "clean" / "EURJPY_15M.parquet",
            BASE / "processed_data" / "clean" / "GBPJPY_15M.parquet",
            BASE / "processed_data" / "clean" / "USDCHF_15M.parquet",
            BASE / "processed_data" / "clean" / "XAUUSD_15M.parquet",
        ],
    ),
    4: (
        "Multi-Asset Alignment",
        "step3_align",
        [BASE / "processed_data" / "aligned_multi_asset.parquet"],
    ),
    5: (
        "Feature Engineering",
        "step4_features",
        [BASE / "processed_data" / "feature_engineered.parquet"],
    ),
    6: (
        "Train/Val/Test Split",
        "step5_split",
        [
            BASE / "ml_training" / "datasets" / "train.parquet",
            BASE / "ml_training" / "datasets" / "validation.parquet",
            BASE / "ml_training" / "datasets" / "test.parquet",
        ],
    ),
    7: (
        "Model Training",
        "step7_train",
        [
            ENGINE_DIR / "weights" / "regime_classifier.pkl",
            ENGINE_DIR / "weights" / "quality_scorer.pkl",
            ENGINE_DIR / "weights" / "rl_ppo" / "model.zip",
            ENGINE_DIR / "weights" / "gru_lstm" / "model.pt",
        ],
    ),
    8: (
        "Backtest + Reinforced Training",
        "step6_backtest",
        [BASE / "backtesting" / "results" / "latest_summary.json"],
    ),
    9: (
        "Validation + Critic",
        "step8_validate",
        [BASE / "ml_training" / "metrics" / "critic_report.json"],
    ),
}

# Human-readable step table (for --list)
_STEP_DESCRIPTIONS = {
    1: "Scan training_data/ → processed_data/raw_inventory.json",
    2: "Resample M1 → 5M/15M/1H/4H/1D/1W/1MN parquets for all 11 symbols",
    3: "Clean M1-resampled parquets → processed_data/clean/*.parquet",
    4: "Align all symbols to shared timeline → aligned_multi_asset.parquet",
    5: "Engineer features → feature_engineered.parquet",
    6: "Time-based 70/15/15 split → ml_training/datasets/",
    7: "Train GRU-LSTM, Regime, Quality, RL models → trading-engine/weights/",
    8: "Backtest (3 rounds) + reinforced retraining → backtesting/results/",
    9: "Validate models + generate critic report → ml_training/metrics/",
}


# ─── Done-check ───────────────────────────────────────────────────────────────

def _is_done(step_num: int) -> bool:
    """Return True if all done-check paths exist for this step."""
    paths = _STEPS[step_num][2]
    if not paths:
        return False
    return all(p.exists() for p in paths)


def _done_summary(step_num: int) -> str:
    paths = _STEPS[step_num][2]
    if not paths:
        return "(no done-check configured)"
    existing = [p for p in paths if p.exists()]
    return f"{len(existing)}/{len(paths)} outputs present"


# ─── Step runner ──────────────────────────────────────────────────────────────

def _build_env() -> dict:
    """Build subprocess environment with correct PYTHONPATH."""
    env = os.environ.copy()
    parts = [
        str(BASE),
        str(PIPELINE_DIR),
        str(ENGINE_DIR),
    ]
    existing = env.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["PYTHONUNBUFFERED"] = "1"
    # Conservative thread caps to reduce memory spikes in BLAS/numexpr/arrow
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("ARROW_NUM_THREADS", "1")
    env.setdefault("RAYON_NUM_THREADS", "1")
    return env


def run_step(step_num: int, force: bool = False) -> bool:
    """
    Run a single pipeline step as a subprocess.

    Returns True on success, False on failure.
    Skips automatically when all done-check outputs exist (unless force=True).
    """
    display_name, script_basename, _ = _STEPS[step_num]
    sep = "=" * 64

    if not force and _is_done(step_num):
        logger.info(
            "%s\n  STEP %d: %-28s [SKIP — %s]\n%s",
            sep, step_num, display_name, _done_summary(step_num), sep,
        )
        return True

    logger.info("%s\n  STEP %d: %s\n%s", sep, step_num, display_name, sep)

    script_path = PIPELINE_DIR / f"{script_basename}.py"
    if not script_path.exists():
        logger.error("Script not found: %s", script_path)
        return False

    # Force garbage collection in parent before forking to give child more headroom
    gc.collect()

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BASE),
            env=_build_env(),
            check=False,
        )
        elapsed = time.time() - t0

        # GC after child exits to reclaim any shared pages
        gc.collect()

        if result.returncode == 0:
            logger.info("STEP %d COMPLETE in %.1fs", step_num, elapsed)
            return True

        logger.error("STEP %d FAILED (exit %d) after %.1fs", step_num, result.returncode, elapsed)
        return False

    except Exception as exc:
        elapsed = time.time() - t0
        logger.error("STEP %d FAILED after %.1fs: %s", step_num, elapsed, exc, exc_info=True)
        gc.collect()
        return False


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ML + backtest pipeline (training before backtesting)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Step order:
  1  Data Discovery          (skip if: raw_inventory.json exists)
  2  M1 Resample → All TFs   (skip if: histdata/ parquets exist)
  3  Data Cleaning           (skip if: clean/*.parquet exist)
  4  Multi-Asset Alignment   (skip if: aligned_multi_asset.parquet exists)
  5  Feature Engineering     (skip if: feature_engineered.parquet exists)
  6  Train/Val/Test Split    (skip if: train/val/test.parquet exist)
  7  Model Training          (skip if: all 4 weight outputs exist)
  8  Backtest + Reinforced   (skip if: latest_summary.json exists)
  9  Validation + Critic     (skip if: critic_report.json exists)
Note: outputs are cleaned by default; pass --no-clean to keep prior outputs.
""",
    )
    parser.add_argument(
        "--steps", nargs="+", type=int, metavar="N",
        help="Run only these steps (e.g. --steps 4 5 6)",
    )
    parser.add_argument(
        "--start-from", type=int, default=1, metavar="N",
        help="Start from step N, run through to the end (default: 1)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore done-check — re-run steps even if outputs exist",
    )
    parser.add_argument(
        "--no-clean", action="store_true",
        help="Do not delete prior pipeline outputs before running",
    )
    parser.add_argument(
        "--stop-on-failure", action="store_true",
        help="Abort remaining steps if any step fails",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print step descriptions and current done-status, then exit",
    )
    args = parser.parse_args()

    # --list: show status and exit
    if args.list:
        print(f"\n{'Step':<6} {'Status':<10} {'Description'}")
        print("-" * 80)
        for n in sorted(_STEPS):
            status = "DONE" if _is_done(n) else "pending"
            desc = _STEP_DESCRIPTIONS.get(n, "")
            print(f"  {n:<4} {status:<10} {desc}")
        print()
        return

    # Determine steps to run
    if args.steps:
        steps_to_run = sorted(set(args.steps))
    else:
        steps_to_run = list(range(args.start_from, max(_STEPS) + 1))

    invalid = [n for n in steps_to_run if n not in _STEPS]
    if invalid:
        parser.error(f"Invalid step numbers: {invalid}. Valid range: 1–{max(_STEPS)}")

    # Apply memory limit before running any step
    _apply_memory_limit()

    # Default behavior: clean outputs to ensure a fresh run
    if not args.no_clean:
        logger.info("Cleaning previous pipeline outputs...")
        _clean_pipeline_outputs()

    results: dict[int, str] = {}
    pipeline_start = time.time()

    for step_num in steps_to_run:
        success = run_step(step_num, force=args.force)
        results[step_num] = "SUCCESS" if success else "FAILED"

        if not success and args.stop_on_failure:
            logger.error("--stop-on-failure: aborting pipeline at step %d", step_num)
            break

        # Brief pause between steps to let OS reclaim memory from previous subprocess
        time.sleep(0.5)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - pipeline_start
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  PIPELINE COMPLETE in {total_elapsed:.0f}s")
    print(f"{sep}")
    for step_num, status in results.items():
        name = _STEPS[step_num][0]
        icon = "✓" if status == "SUCCESS" else "✗"
        skipped = not args.force and _is_done(step_num) and status == "SUCCESS"
        tag = " (skipped — already done)" if skipped else ""
        print(f"  {icon} Step {step_num}: {name} — {status}{tag}")

    failures = [k for k, v in results.items() if v == "FAILED"]
    if failures:
        print(f"\n  Failed steps: {failures}")
        print("  Re-run failed steps with: python pipeline/run_pipeline.py --steps "
              + " ".join(str(f) for f in failures) + " --force")
        sys.exit(1)
    else:
        print("\n  All steps passed.")


if __name__ == "__main__":
    main()
