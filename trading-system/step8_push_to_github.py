"""
step8_push_to_github.py — Push training outputs back to GitHub.

Setup:
  - Pipeline runs in /kaggle/working/Multi-Bot (copied from input dataset, no git history)
  - Notebook cell 0 clones the repo fresh to /kaggle/working/remote/Multi-Bot
  - This script copies new training artifacts into that clone and pushes to GitHub

This script:
  1. Sets committer identity + injects GITHUB_TOKEN into the remote URL
  2. Copies new training artifacts into the clone
  3. Stages + commits + pushes to main

Required env var:
    GITHUB_TOKEN  — personal access token with repo write scope

Optional env vars:
    GITHUB_REPO   — full repo name, default: AnalystTKZ/Multi-Bot
    GITHUB_BRANCH — target branch,   default: main
    GITHUB_USER   — git committer name,  default: Kaggle Training Bot
    GITHUB_EMAIL  — git committer email, default: bot@kaggle.local
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

GITHUB_TOKEN  = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO   = os.getenv("GITHUB_REPO",   "AnalystTKZ/Multi-Bot")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
GITHUB_USER   = os.getenv("GITHUB_USER",   "Kaggle Training Bot")
GITHUB_EMAIL  = os.getenv("GITHUB_EMAIL",  "bot@kaggle.local")

# ── Paths ─────────────────────────────────────────────────────────────────────

try:
    _HERE = Path(__file__).resolve().parent   # trading-system/
except NameError:
    _HERE = Path("/kaggle/working/Multi-Bot/trading-system")

# On Kaggle: fresh clone is at /kaggle/working/remote/Multi-Bot
# Locally: two levels up from trading-system/
_KAGGLE_CLONE = Path("/kaggle/working/remote/Multi-Bot")
REPO_ROOT = _KAGGLE_CLONE if _KAGGLE_CLONE.exists() else _HERE.parent

ENGINE = _HERE / "trading-engine"
ML_DIR = _HERE / "ml_training"

# ── Artifact manifest ─────────────────────────────────────────────────────────
# (source_path, dest_path relative to REPO_ROOT)
# Missing sources are silently skipped.

ARTIFACTS: list[tuple[Path, Path]] = [
    (ENGINE / "weights" / "gru_lstm" / "model.pt",
     Path("trading-system/trading-engine/weights/gru_lstm/model.pt")),

    (ENGINE / "weights" / "regime_4h.pkl",
     Path("trading-system/trading-engine/weights/regime_4h.pkl")),

    (ENGINE / "weights" / "regime_1h.pkl",
     Path("trading-system/trading-engine/weights/regime_1h.pkl")),

    (ENGINE / "weights" / "quality_scorer.pkl",
     Path("trading-system/trading-engine/weights/quality_scorer.pkl")),

    (ENGINE / "weights" / "rl_ppo" / "model.zip",
     Path("trading-system/trading-engine/weights/rl_ppo/model.zip")),

    (ENGINE / "weights" / "macro_correlations.json",
     Path("trading-system/trading-engine/weights/macro_correlations.json")),

    (ENGINE / "weights" / "gru_lstm" / "weights_manifest.json",
     Path("trading-system/trading-engine/weights/gru_lstm/weights_manifest.json")),

    (ML_DIR / "metrics" / "training_summary.json",
     Path("trading-system/ml_training/metrics/training_summary.json")),

    (ML_DIR / "metrics" / "training_7b_summary.json",
     Path("trading-system/ml_training/metrics/training_7b_summary.json")),

    (ENGINE / "logs" / "backtest_diagnostics.csv",
     Path("trading-system/trading-engine/logs/backtest_diagnostics.csv")),

    (ENGINE / "logs" / "trade_journal_detailed.jsonl",
     Path("trading-system/trading-engine/logs/trade_journal_detailed.jsonl")),

    (_HERE / "backtesting" / "results" / "latest_summary.json",
     Path("trading-system/backtesting/results/latest_summary.json")),
]

ARTIFACT_DIRS: list[tuple[Path, Path]] = [
    (ENGINE / "backtest_results",
     Path("trading-system/trading-engine/backtest_results")),

    (ML_DIR / "logs",
     Path("trading-system/ml_training/logs")),
]


def _run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    logger.debug("$ %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(cwd), check=check, capture_output=True, text=True)


def _copy_artifact(src: Path, dst_rel: Path) -> bool:
    if not src.exists():
        logger.debug("skip (missing): %s", src)
        return False
    dst = REPO_ROOT / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    logger.info("  copied: %s → %s", src.name, dst_rel)
    return True


def _copy_dir(src: Path, dst_rel: Path) -> int:
    if not src.exists() or not src.is_dir():
        return 0
    count = 0
    for f in src.rglob("*"):
        if f.is_file():
            _copy_artifact(f, dst_rel / f.relative_to(src))
            count += 1
    return count


def main() -> None:
    if not GITHUB_TOKEN:
        logger.error("GITHUB_TOKEN not set — cannot push to GitHub")
        sys.exit(1)

    logger.info("=== STEP 8: PUSH TRAINING OUTPUTS TO GITHUB ===")
    logger.info("Repo:   %s", GITHUB_REPO)
    logger.info("Branch: %s", GITHUB_BRANCH)
    logger.info("Root:   %s", REPO_ROOT)

    remote_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"

    _run(["git", "config", "user.name",  GITHUB_USER],  cwd=REPO_ROOT)
    _run(["git", "config", "user.email", GITHUB_EMAIL], cwd=REPO_ROOT)
    _run(["git", "remote", "set-url", "origin", remote_url], cwd=REPO_ROOT)

    logger.info("Pulling latest from origin/%s ...", GITHUB_BRANCH)
    _run(["git", "pull", "--ff-only", "origin", GITHUB_BRANCH], cwd=REPO_ROOT)

    # ── Copy new artifacts into the working tree ──────────────────────────────
    logger.info("Collecting artifacts ...")
    n_files = 0
    for src, dst_rel in ARTIFACTS:
        if _copy_artifact(src, dst_rel):
            n_files += 1
    for src_dir, dst_rel in ARTIFACT_DIRS:
        n_files += _copy_dir(src_dir, dst_rel)
    logger.info("Collected %d file(s)", n_files)

    # ── Stage, commit, push ───────────────────────────────────────────────────
    _run(["git", "add",
          "trading-system/trading-engine/weights/",
          "trading-system/trading-engine/backtest_results/",
          "trading-system/trading-engine/logs/",
          "trading-system/ml_training/",
          "trading-system/backtesting/results/",
          ], cwd=REPO_ROOT)

    status = _run(["git", "status", "--porcelain"], cwd=REPO_ROOT)
    if not status.stdout.strip():
        logger.info("Nothing changed — skipping push")
        return

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    _run(["git", "commit", "-m", f"chore(kaggle): training outputs {ts}"], cwd=REPO_ROOT)

    logger.info("Pushing to origin/%s ...", GITHUB_BRANCH)
    _run(["git", "push", "origin", f"HEAD:{GITHUB_BRANCH}"], cwd=REPO_ROOT)

    logger.info("=== PUSH COMPLETE ===")


if __name__ == "__main__":
    main()
