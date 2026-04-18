"""
kaggle_train.py — Kaggle training entrypoint.

== KAGGLE SETUP (run this cell first) ==

    import os, subprocess
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    subprocess.run(
        f"git clone https://{token}@github.com/AnalystTKZ/Multi-Bot.git /kaggle/working/Multi-Bot",
        shell=True, check=True
    )

Then in the next cell:

    %run /kaggle/working/Multi-Bot/trading-system/kaggle_train.py

== ON KAGGLE, ONLY STEPS 5-8 RUN ==
Steps 0-4 produce processed_data/ which is already committed to the repo.
The pipeline skips any step whose done-check output already exists.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Anchor to this file's directory (trading-system/)
HERE = Path(__file__).resolve().parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))

from env_config import get_env, ensure_output_dirs

env = get_env()
ensure_output_dirs(env)

# Always re-run training and backtest on Kaggle — delete stale done-checks so
# steps 7 and 6 never skip even if a previous session left their outputs behind.
if env["on_kaggle"]:
    for stale in [
        env["ml_training"] / "metrics" / "training_summary.json",
        env["ml_training"] / "metrics" / "training_7b_summary.json",
        env["base"] / "backtesting" / "results" / "latest_summary.json",
    ]:
        if stale.exists():
            stale.unlink()
            print(f"  Cleared done-check: {stale.name}")

label = "KAGGLE" if env["on_kaggle"] else "LOCAL"
print(f"Environment : {label}")
print(f"  base      -> {env['base']}")
print(f"  processed -> {env['processed']}")
print(f"  ml_train  -> {env['ml_training']}")
print(f"  weights   -> {env['weights']}")
print(f"  output    -> {env['output']}")

# Verify pipeline scripts exist
pipeline_dir = env["pipeline"]
required_scripts = [
    "step1_inventory.py", "step2_clean.py", "step3_align.py",
    "step4_features.py", "step5_split.py",
    "step7_train.py", "step6_backtest.py", "step7b_train.py",
]
missing_scripts = [s for s in required_scripts if not (pipeline_dir / s).exists()]
if missing_scripts:
    raise FileNotFoundError(f"Missing pipeline scripts: {missing_scripts}")

# Verify required input data
required_data = [
    env["processed"] / "histdata" / "EURUSD_5M.parquet",
    env["processed"] / "histdata" / "XAUUSD_1D.parquet",
    env["ml_training"] / "datasets" / "train.parquet",
    env["base"] / "training_data" / "indices" / "VIX_1d.csv",
    env["base"] / "training_data" / "fundamental" / "macro_releases.csv",
]
missing_data = [p for p in required_data if not p.exists()]
if missing_data:
    print("\nMissing required data:")
    for p in missing_data:
        print(f"  {p}")
    raise FileNotFoundError("Required data files not found.")

print("\nAll scripts and inputs verified.")

# Training order:
#   7a (GRU + Regime) → 6 (Backtest → real journal) → 7b (Quality + RL on real journal)
# Quality and RL require a real journal — no synthetic fallback.
# Test set (2024-08 → 2026) is never touched by backtest.
PIPELINE_STEPS = [
    ("Step 0 - Resample",    "step0_resample.py",  env["processed"] / "histdata" / "XAUUSD_1D.parquet"),
    ("Step 1 - Inventory",   "step1_inventory.py", env["processed"] / "raw_inventory.json"),
    ("Step 2 - Cleaning",    "step2_clean.py",      env["processed"] / "clean" / "XAUUSD_15M.parquet"),
    ("Step 3 - Alignment",   "step3_align.py",      env["processed"] / "aligned_multi_asset.parquet"),
    ("Step 4 - Features",    "step4_features.py",   env["processed"] / "feature_engineered.parquet"),
    ("Step 5 - Split",       "step5_split.py",      env["ml_training"] / "datasets" / "train.parquet"),
    ("Step 7a - GRU+Regime", "step7_train.py",      env["ml_training"] / "metrics" / "training_summary.json"),
    ("Step 6 - Backtest",    "step6_backtest.py",   env["base"] / "backtesting" / "results" / "latest_summary.json"),
    ("Step 7b - Quality+RL", "step7b_train.py",     env["ml_training"] / "metrics" / "training_7b_summary.json"),
]


def run_step(name: str, script: str, done_check: Path) -> None:
    script_path = pipeline_dir / script
    if not script_path.exists():
        raise FileNotFoundError(f"{name}: script not found at {script_path}")
    if done_check.exists():
        print(f"  SKIP  {name}")
        return
    print(f"  START {name}")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(env["base"]),
        env={
            **os.environ,
            "PYTHONPATH": f"{env['base']}:{env['base'] / 'trading-engine'}",
        },
    )
    if result.returncode != 0:
        raise RuntimeError(f"{name} FAILED (exit {result.returncode})")
    print(f"  DONE  {name}")


print("\n=== Running pipeline ===")
for step_name, script_file, done_check in PIPELINE_STEPS:
    run_step(step_name, script_file, done_check)

print("\n=== Pipeline complete ===")

# Copy weights to /kaggle/working for download after session
output_weights = env["output"] / "trained_weights"
if env["weights"].exists():
    shutil.copytree(env["weights"], output_weights, dirs_exist_ok=True)
    print(f"Weights saved to: {output_weights}")
else:
    print("No weights directory found after training.")

# ── Step 8: Push training outputs to GitHub ───────────────────────────────────
push_script = env["base"] / "step8_push_to_github.py"
if os.getenv("GITHUB_TOKEN") and push_script.exists():
    print("\n=== STEP 8: Pushing outputs to GitHub ===")
    result = subprocess.run(
        [sys.executable, str(push_script)],
        cwd=str(env["base"]),
        env={
            **os.environ,
            "PYTHONPATH": f"{env['base']}:{env['base'] / 'trading-engine'}",
        },
    )
    if result.returncode != 0:
        print("WARNING: GitHub push failed (non-fatal — weights still saved locally)")
    else:
        print("=== GitHub push complete ===")
