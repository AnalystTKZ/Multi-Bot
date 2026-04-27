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
Step 7a always re-runs (GRU + Regime full training on train set).

== BLIND BACKTEST PIPELINE ==

  Round 1 — backtest on val window (last 2yr of training data)
             → retrain Quality + RL on Round 1 journal

  Round 2 — backtest on test window (unseen 2yr — BLIND)
             → retrain Quality + RL on Round 1+2 journal

  Round 3 — incremental retrain
             → backtest on last 3yr (val+test overlap) as a stability check
             → retrain Quality + RL on Round 1+2+3 journal for the next run

Data split (step5_split.py):
  train      = data_start → (test_end - 4yr)      models train on this
  validation = 2yr before test                     Round 1 backtest
  test       = last 2yr of available data          Round 2 backtest (blind)
"""
from __future__ import annotations
import json
import os
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
# steps 7a, 6, and 7b never skip even if a previous session left outputs behind.
if env["on_kaggle"]:
    for stale in [
        env["ml_training"] / "metrics" / "training_summary.json",
        env["ml_training"] / "metrics" / "training_7b_r1_summary.json",
        env["ml_training"] / "metrics" / "training_7b_r2_summary.json",
        env["ml_training"] / "metrics" / "training_7b_r3_summary.json",
        env["base"] / "backtesting" / "results" / "latest_summary.json",
        env["base"] / "backtesting" / "results" / "round1_summary.json",
        env["base"] / "backtesting" / "results" / "round2_summary.json",
        env["base"] / "backtesting" / "results" / "round3_summary.json",
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
    kaggle_input = env["base"].parent.parent.parent / "input"
    if not kaggle_input.exists():
        kaggle_input = Path("/kaggle/input")
    print(f"  kaggle/input -> {kaggle_input}")
    if kaggle_input.exists():
        for d in sorted(kaggle_input.iterdir()):
            print(f"    dataset: {d.name}  "
                  f"(has training_data={( d / 'training_data').exists()}, "
                  f"processed_data={(d / 'processed_data').exists()})")

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
]
missing_data = [p for p in required_data if not p.exists()]
if missing_data:
    print("\nMissing required histdata parquets (provide via Kaggle dataset):")
    for p in missing_data:
        print(f"  {p}")
    raise FileNotFoundError("Required histdata parquets not found.")

optional_data = [
    env["data"] / "indices" / "VIX_1d.csv",
    env["data"] / "fundamental" / "macro_releases.csv",
]
for p in optional_data:
    if not p.exists():
        print(f"  WARNING: optional file missing (macro features reduced): {p}")

print("\nAll scripts and inputs verified.")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _base_env() -> dict:
    return {
        **os.environ,
        "PYTHONPATH": f"{env['base']}:{env['base'] / 'trading-engine'}",
    }


def _round_journal_training_env() -> dict:
    """
    Permit Quality/RL to train from the journals produced by the staged
    backtest rounds. This is intentionally scoped to explicit source_split
    names instead of ALLOW_NONTRAIN_JOURNAL_TRAINING=1, which would accept any
    journal provenance.
    """
    return {
        "ALLOW_ROUND_JOURNAL_TRAINING": "1",
        "JOURNAL_ALLOWED_SPLITS": "train,validation,test,combined_eval,live,paper,production",
    }


def run_step(name: str, script: str, done_check: Path, extra_env: dict | None = None) -> None:
    script_path = pipeline_dir / script
    if not script_path.exists():
        raise FileNotFoundError(f"{name}: script not found at {script_path}")
    if done_check.exists():
        print(f"  SKIP  {name}")
        return
    print(f"  START {name}")
    step_env = {**_base_env(), **(extra_env or {})}
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(env["base"]),
        env=step_env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{name} FAILED (exit {result.returncode})")
    print(f"  DONE  {name}")


def run_retrain(model: str, label: str = "", extra_env: dict | None = None) -> None:
    """Run retrain_incremental.py for a single model."""
    script = env["base"] / "trading-engine" / "scripts" / "retrain_incremental.py"
    tag = f" [{label}]" if label else ""
    print(f"  START Retrain {model}{tag}")
    retrain_env = {**_base_env(), **(extra_env or {})}
    result = subprocess.run(
        [sys.executable, str(script), "--model", model],
        cwd=str(env["base"] / "trading-engine"),
        env=retrain_env,
    )
    if result.returncode != 0:
        print(f"  WARN  Retrain {model} failed (exit {result.returncode}) — continuing")
    else:
        print(f"  DONE  Retrain {model}{tag}")


def _journal_path() -> Path:
    # env["engine"] resolves to the remote clone's trading-engine/ on Kaggle
    # when the clone exists — same root step6_backtest.py writes the journal to.
    return env["engine"] / "logs" / "trade_journal_detailed.jsonl"


def _journal_count() -> int:
    j = _journal_path()
    if not j.exists():
        return 0
    return sum(1 for _ in open(j))


def _copy_bt_result(src_label: str, dest_name: str) -> None:
    """Save the latest parsed step6 summary to a named round summary file."""
    import shutil
    dest = env["base"] / "backtesting" / "results" / dest_name
    dest.parent.mkdir(parents=True, exist_ok=True)

    latest_summary = env["base"] / "backtesting" / "results" / "latest_summary.json"
    if latest_summary.exists():
        shutil.copy2(latest_summary, dest)
    else:
        engine_bt = env["base"] / "trading-engine" / "backtest_results"
        if not engine_bt.exists():
            return
        jsons = sorted(engine_bt.glob("backtest_*.json"))
        if not jsons:
            return
        shutil.copy2(jsons[-1], dest)
    print(f"  Saved {src_label} result → {dest.name}")


def _print_split_info() -> None:
    sp = env["ml_training"] / "datasets" / "split_summary.json"
    if not sp.exists():
        return
    try:
        s = json.loads(sp.read_text())
        dr = s.get("date_ranges", {})
        print(f"\n  Data split ({s.get('split_method','?')}):")
        for name in ("train", "validation", "test"):
            w = dr.get(name, {})
            rows = s.get("rows", {}).get(name, "?")
            print(f"    {name:<12} {rows:>7} bars  {w.get('start','?')[:10]} → {w.get('end','?')[:10]}")
    except Exception:
        pass


# ─── Phase 0-5: Data preparation (skip if outputs exist) ─────────────────────

PREP_STEPS = [
    ("Step 0 - Resample",  "step0_resample.py", env["processed"] / "histdata" / "XAUUSD_1D.parquet"),
    ("Step 1 - Inventory", "step1_inventory.py", env["processed"] / "raw_inventory.json"),
    ("Step 2 - Cleaning",  "step2_clean.py",      env["processed"] / "clean" / "XAUUSD_15M.parquet"),
    ("Step 3 - Alignment", "step3_align.py",      env["processed"] / "aligned_multi_asset.parquet"),
    ("Step 4 - Features",  "step4_features.py",   env["processed"] / "feature_engineered.parquet"),
    ("Step 5 - Split",     "step5_split.py",      env["ml_training"] / "datasets" / "train.parquet"),
]

print("\n=== Phase 0-5: Data preparation ===")
for step_name, script_file, done_check in PREP_STEPS:
    run_step(step_name, script_file, done_check)

_print_split_info()

# ─── Phase 7a: Train GRU + Regime on TRAIN set ───────────────────────────────
# Models train only on data before the val split cutoff.
# BT_START_FLOOR is irrelevant here — retrain_incremental uses the train.parquet
# produced by step5, which is already capped at val_start.

print("\n=== Phase 7a: Train GRU + Regime (train set only) ===")
run_step(
    "Step 7a - GRU+Regime",
    "step7_train.py",
    env["ml_training"] / "metrics" / "training_summary.json",
)

# ─── Round 1: Backtest on val window ─────────────────────────────────────────
# Covers the last 2 years of the training window (val set).
# Tests whether trained models generalise beyond the training period.
# Journal is cleared before Round 1 so it accumulates only backtest trades.

print("\n=== Round 1: Backtest on validation window (last 2yr of training data) ===")

j = _journal_path()
if j.exists():
    j.unlink()
    print("  Cleared journal for fresh Round 1 run")

run_step(
    "Round 1 - Backtest (val)",
    "step6_backtest.py",
    env["base"] / "backtesting" / "results" / "round1_summary.json",
    extra_env={"BT_WINDOW": "round1"},
)
_copy_bt_result("Round 1", "round1_summary.json")
print(f"  Journal after Round 1: {_journal_count()} entries")

# ─── Train Quality + RL on Round 1 journal ───────────────────────────────────
print("\n=== Round 1 → Retrain Quality + RL on Round 1 journal ===")
run_step(
    "Round 1 - Quality+RL retrain",
    "step7b_train.py",
    env["ml_training"] / "metrics" / "training_7b_r1_summary.json",
    extra_env=_round_journal_training_env(),
)
# Rename done-check so Round 2's retrain gets a fresh marker
r1_done = env["ml_training"] / "metrics" / "training_7b_summary.json"
r1_dest = env["ml_training"] / "metrics" / "training_7b_r1_summary.json"
if r1_done.exists() and not r1_dest.exists():
    r1_done.rename(r1_dest)

# ─── Round 2: BLIND backtest on test window ───────────────────────────────────
# This is the true out-of-sample evaluation.
# Models were trained on data up to val_start; they have NEVER seen this data.

print("\n=== Round 2: BLIND backtest on test window (unseen 2yr) ===")
run_step(
    "Round 2 - Blind backtest (test)",
    "step6_backtest.py",
    env["base"] / "backtesting" / "results" / "round2_summary.json",
    extra_env={"BT_WINDOW": "round2"},
)
_copy_bt_result("Round 2", "round2_summary.json")
print(f"  Journal after Round 2: {_journal_count()} entries")

# ─── Train Quality + RL on Round 1+2 combined journal ────────────────────────
print("\n=== Round 2 → Retrain Quality + RL on Round 1+2 journal ===")
r2_done = env["ml_training"] / "metrics" / "training_7b_summary.json"
if r2_done.exists():
    r2_done.unlink()
run_step(
    "Round 2 - Quality+RL retrain",
    "step7b_train.py",
    env["ml_training"] / "metrics" / "training_7b_r2_summary.json",
    extra_env=_round_journal_training_env(),
)
r2_dest = env["ml_training"] / "metrics" / "training_7b_r2_summary.json"
if r2_done.exists() and not r2_dest.exists():
    r2_done.rename(r2_dest)

# ─── Round 3: Incremental retrain ────────────────────────────────────────────
# GRU/Regime continue to fit only on the train split. Quality/RL train on the
# accumulated round journals because they are the feedback learners for this
# staged research loop.

print("\n=== Round 3: Incremental retrain ===")
for model in ["gru", "regime"]:
    run_retrain(model, label="train-split retrain")
for model in ["quality", "rl"]:
    run_retrain(
        model,
        label="round-journal retrain",
        extra_env=_round_journal_training_env(),
    )

# ─── Round 3: Backtest on last 3yr (val+test overlap) ────────────────────────
# Evaluates the fully-retrained models over the most recent 3 years of data.
# This is not blind (models have seen some of this period), but it validates
# that incremental learning did not degrade performance.

print("\n=== Round 3: Backtest on last 3yr (post-retrain evaluation) ===")
run_step(
    "Round 3 - Post-retrain backtest (last 3yr)",
    "step6_backtest.py",
    env["base"] / "backtesting" / "results" / "round3_summary.json",
    extra_env={"BT_WINDOW": "round3"},
)
_copy_bt_result("Round 3", "round3_summary.json")
print(f"  Journal after Round 3: {_journal_count()} entries")

# ─── Train Quality + RL on Round 1+2+3 combined journal ──────────────────────
print("\n=== Round 3 → Retrain Quality + RL on Round 1+2+3 journal ===")
r3_done = env["ml_training"] / "metrics" / "training_7b_summary.json"
if r3_done.exists():
    r3_done.unlink()
run_step(
    "Round 3 - Quality+RL retrain",
    "step7b_train.py",
    env["ml_training"] / "metrics" / "training_7b_r3_summary.json",
    extra_env=_round_journal_training_env(),
)
r3_dest = env["ml_training"] / "metrics" / "training_7b_r3_summary.json"
if r3_done.exists() and not r3_dest.exists():
    r3_done.rename(r3_dest)

# ─── Final summary ────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  BLIND BACKTEST PIPELINE COMPLETE")
print("=" * 70)
for rnd, fname in [("Round 1 (val window)", "round1_summary.json"),
                   ("Round 2 (blind test)", "round2_summary.json"),
                   ("Round 3 (last 3yr)",   "round3_summary.json")]:
    p = env["base"] / "backtesting" / "results" / fname
    if p.exists():
        try:
            s = json.loads(p.read_text())
            trades = s.get("total_trades", s.get("trades", "?"))
            wr     = s.get("avg_win_rate", s.get("win_rate", 0.0))
            pf     = s.get("avg_profit_factor", s.get("profit_factor", 0.0))
            sh     = s.get("avg_sharpe", s.get("sharpe", 0.0))
            print(f"  {rnd:<28}  trades={trades}  WR={wr*100:.1f}%  PF={pf:.3f}  Sharpe={sh:.3f}")
        except Exception:
            print(f"  {rnd:<28}  (could not parse result)")
    else:
        print(f"  {rnd:<28}  (no result file)")
print()

# Mark the overall pipeline done (latest_summary.json checked by run_pipeline.py)
import shutil
r3 = env["base"] / "backtesting" / "results" / "round3_summary.json"
latest = env["base"] / "backtesting" / "results" / "latest_summary.json"
if r3.exists():
    shutil.copy2(r3, latest)

# ── Step 8: Push training outputs to GitHub ───────────────────────────────────
push_script = env["base"] / "step8_push_to_github.py"
github_token = os.getenv("GITHUB_TOKEN", "")
if not github_token:
    print("\nWARNING: GITHUB_TOKEN not set — skipping GitHub push")
elif not push_script.exists():
    print(f"\nWARNING: step8 script not found at {push_script} — skipping GitHub push")
else:
    print("\n=== STEP 8: Pushing outputs to GitHub ===")
    result = subprocess.run(
        [sys.executable, str(push_script)],
        cwd=str(env["base"]),
        env={
            **os.environ,
            "GITHUB_TOKEN": github_token,
            "PYTHONPATH": f"{env['base']}:{env['base'] / 'trading-engine'}",
        },
    )
    if result.returncode != 0:
        print("WARNING: GitHub push failed (non-fatal — weights still saved locally)")
    else:
        print("=== GitHub push complete ===")
