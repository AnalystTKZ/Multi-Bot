#!/usr/bin/env python3
"""Run the simplify-v2 parallel pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PIPELINE_DIR = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_simple_pipeline")

STEPS = {
    1: ("Data Prep", "step1_data_prep", [BASE / "processed_data" / "simple" / "ohlcv_manifest.json"]),
    2: ("Feature Engineering", "step2_features", [BASE / "processed_data" / "simple" / "features.parquet"]),
    3: ("Split", "step3_split", [BASE / "ml_training" / "datasets" / "simple_datasets" / "split_summary.json"]),
    4: ("Unified Model Training", "step4_train_unified", [BASE / "trading-engine" / "weights" / "unified_direction_regime" / "model.pt"]),
}


def _env() -> dict:
    env = os.environ.copy()
    parts = [str(BASE), str(BASE / "trading-engine"), str(PIPELINE_DIR)]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _done(step: int) -> bool:
    return all(path.exists() for path in STEPS[step][2])


def run_step(step: int, force: bool = False) -> bool:
    name, module, _paths = STEPS[step]
    if _done(step) and not force:
        logger.info("Step %d %s skipped; outputs already exist", step, name)
        return True
    script = PIPELINE_DIR / f"{module}.py"
    logger.info("Step %d %s", step, name)
    result = subprocess.run([sys.executable, str(script)], cwd=str(BASE), env=_env(), check=False)
    if result.returncode != 0:
        logger.error("Step %d failed with exit %d", step, result.returncode)
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simplify-v2 pipeline")
    parser.add_argument("--steps", nargs="+", type=int)
    parser.add_argument("--start-from", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for step in sorted(STEPS):
            print(f"{step}: {STEPS[step][0]} [{'DONE' if _done(step) else 'pending'}]")
        return

    steps = sorted(set(args.steps)) if args.steps else list(range(args.start_from, max(STEPS) + 1))
    invalid = [step for step in steps if step not in STEPS]
    if invalid:
        parser.error(f"invalid steps: {invalid}")

    started = time.time()
    for step in steps:
        if not run_step(step, force=args.force):
            sys.exit(1)
    logger.info("Simplified pipeline complete in %.1fs", time.time() - started)


if __name__ == "__main__":
    main()
