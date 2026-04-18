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

== DATA SOURCES (split-dataset setup) ==
Two Kaggle datasources are merged at runtime:
  1. GitHub clone (this file)   — code only, no raw data
  2. Kaggle dataset             — training_data/ + processed_data/ (read-only mount)

Steps 0-5 are skipped when their outputs already exist in the mounted dataset.
Steps 7a, 6, 7b always re-run (training + backtest + quality/RL retraining).

All constructed outputs (weights, logs, journals, backtest results) are saved to
trading-engine/weights/ and trading-engine/logs/ inside the working clone, then
copied back to /kaggle/working/outputs/processed_data/ for dataset update.
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
print(f"  data      -> {env['data']}")
print(f"  processed -> {env['processed']}")
print(f"  ml_train  -> {env['ml_training']}")
print(f"  weights   -> {env['weights']}")
print(f"  output    -> {env['output']}")
if env["on_kaggle"]:
    import os as _os
    kaggle_input = env["base"].parent.parent.parent / "input"  # /kaggle/input
    if not kaggle_input.exists():
        kaggle_input = Path("/kaggle/input")
    print(f"  kaggle/input -> {kaggle_input}")
    if kaggle_input.exists():
        for d in sorted(kaggle_input.iterdir()):
            print(f"    dataset: {d.name}  (has training_data={( d / 'training_data').exists()}, processed_data={(d / 'processed_data').exists()})")

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

# Verify required input data — histdata parquets are the only hard requirement.
# train.parquet is built by step 5 (not pre-existing). Macro CSVs are optional.
required_data = [
    env["processed"] / "histdata" / "EURUSD_5M.parquet",
    env["processed"] / "histdata" / "XAUUSD_1D.parquet",
]
missing_data = [p for p in required_data if not p.exists()]
if missing_data:
    print("\nMissing required histdata parquets (provide via Kaggle dataset):")
    for p in missing_data:
        print(f"  {p}")
    raise FileNotFoundError("Required histdata parquets not found.")

# Optional data — warn but continue if absent
optional_data = [
    env["data"] / "indices" / "VIX_1d.csv",
    env["data"] / "fundamental" / "macro_releases.csv",
]
for p in optional_data:
    if not p.exists():
        print(f"  WARNING: optional file missing (macro features reduced): {p}")

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

# ── Save outputs to dataset update directory ─────────────────────────────────
# Copy weights + logs into /kaggle/working/outputs/processed_data/
# so they can be uploaded as a new version of the Kaggle dataset.
if env["on_kaggle"]:
    save_root = env["output"] / "outputs" / "processed_data"
    save_root.mkdir(parents=True, exist_ok=True)

    if env["weights"].exists():
        shutil.copytree(str(env["weights"]), str(save_root / "weights"), dirs_exist_ok=True)
        print(f"Weights saved to: {save_root / 'weights'}")

    engine_logs = env["engine"] / "logs"
    if engine_logs.exists():
        shutil.copytree(str(engine_logs), str(save_root / "logs"), dirs_exist_ok=True)
        print(f"Logs saved to: {save_root / 'logs'}")

    print(f"\nDataset update bundle: {save_root}")
    print("Download /kaggle/working/outputs/ and re-upload as the dataset version.")
else:
    # Local: just report weights location
    if env["weights"].exists():
        print(f"Weights at: {env['weights']}")

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
