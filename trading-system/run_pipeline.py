import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Environment abstraction — works on both local and Kaggle
sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_config import get_env, ensure_output_dirs

_env = get_env()
BASE_DIR = _env["base"]
ensure_output_dirs(_env)

print(f"run_pipeline: environment = {'KAGGLE' if _env['on_kaggle'] else 'LOCAL'}, base = {BASE_DIR}")

# Each entry: (step_name, script_path, done_check_path)
# done_check_path: file/dir that must exist to consider the step already done.
PIPELINE_STEPS = [
    ("Step 0 - Resample",  "pipeline/step0_resample.py",   "processed_data/histdata/XAUUSD_1D.parquet"),
    ("Step 1 - Inventory", "pipeline/step1_inventory.py",  "processed_data/raw_inventory.json"),
    ("Step 2 - Cleaning",  "pipeline/step2_clean.py",      "processed_data/clean/XAUUSD_15M.parquet"),
    ("Step 3 - Alignment", "pipeline/step3_align.py",      "processed_data/aligned_multi_asset.parquet"),
    ("Step 4 - Features",  "pipeline/step4_features.py",   "processed_data/feature_engineered.parquet"),
    ("Step 5 - Split",     "pipeline/step5_split.py",      "ml_training/datasets/train.parquet"),
    ("Step 6 - Backtest",  "pipeline/step6_backtest.py",   "backtesting/results/latest_summary.json"),
    ("Step 7 - Training",  "pipeline/step7_train.py",      "ml_training/metrics/training_summary.json"),
    ("Step 8 - Validation","pipeline/step8_validate.py",   "ml_training/metrics/critic_report.json"),
]

# Memory cap (~3GB as per your runbook)
MEMORY_LIMIT = "ulimit -v 4000000"

# Log file
LOG_FILE = BASE_DIR / "pipeline_run.log"


# =========================
# UTIL FUNCTIONS
# =========================

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_step(step_name, script_path, done_check=None):
    # Skip if output already exists
    if done_check and (BASE_DIR / done_check).exists():
        log(f"⏭  SKIP {step_name} — already done ({done_check})")
        return

    full_path = BASE_DIR / script_path

    if not full_path.exists():
        log(f"❌ {step_name} FAILED — Script not found: {full_path}")
        sys.exit(1)

    log(f"🚀 STARTING {step_name}")

    command = f"""
    cd {BASE_DIR} &&
    export PYTHONPATH="{BASE_DIR}:{BASE_DIR}/trading-engine" &&
    {MEMORY_LIMIT} &&
    python3 {script_path}
    """

    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        log(f"❌ {step_name} FAILED (exit code {result.returncode})")
        sys.exit(result.returncode)

    log(f"✅ COMPLETED {step_name}\n")


def check_output(path, description):
    if not Path(path).exists():
        log(f"⚠️ WARNING: Expected output missing → {description} ({path})")
    else:
        log(f"✔ Verified: {description}")


# =========================
# MAIN PIPELINE
# =========================

def main():
    log("=" * 60)
    log("STARTING FULL PIPELINE RUN")
    log("=" * 60)

    for step_name, script, done_check in PIPELINE_STEPS:
        run_step(step_name, script, done_check)

        # Optional validation checks after key steps
        if "Resample" in step_name:
            check_output(BASE_DIR / "processed_data/histdata", "Histdata MTF folder")

        elif "Inventory" in step_name:
            check_output(BASE_DIR / "processed_data/raw_inventory.json", "Inventory file")

        elif "Cleaning" in step_name:
            check_output(BASE_DIR / "processed_data/clean", "Cleaned data folder")

        elif "Alignment" in step_name:
            check_output(BASE_DIR / "processed_data/aligned_multi_asset.parquet", "Aligned dataset")

        elif "Features" in step_name:
            check_output(BASE_DIR / "processed_data/feature_engineered.parquet", "Feature dataset")

        elif "Split" in step_name:
            check_output(BASE_DIR / "ml_training/datasets/train.parquet", "Train dataset")

        elif "Backtest" in step_name:
            check_output(BASE_DIR / "backtesting/results/latest_summary.json", "Backtest summary")

        elif "Training" in step_name:
            check_output(BASE_DIR / "ml_training/metrics/training_summary.json", "Training summary")

        elif "Validation" in step_name:
            check_output(BASE_DIR / "ml_training/metrics/critic_report.json", "Critic report")

    log("=" * 60)
    log("🎉 PIPELINE COMPLETED SUCCESSFULLY")
    log("=" * 60)


if __name__ == "__main__":
    main()
