"""
step8_push_to_github.py — Push training outputs back to GitHub.

Setup:
  - Pipeline runs in /kaggle/working/Multi-Bot/trading-system (working copy, no git history)
  - Notebook cell 0 clones the repo to /kaggle/working/remote/Multi-Bot (git clone)
  - env_config.py routes all weight/log writes directly into the git clone when it exists,
    so by the time this script runs the clone's working tree is already up-to-date.
  - This script handles fallback copying (if the clone wasn't present at training time),
    then stages, commits, and pushes to GitHub.

This script:
  1. Sets committer identity + injects GITHUB_TOKEN into the remote URL
  2. Falls back to copying artifacts from the working copy if they weren't written directly
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
    _WORK_TS = Path(__file__).resolve().parent   # working copy trading-system/
except NameError:
    _WORK_TS = Path("/kaggle/working/Multi-Bot/trading-system")

# Remote git clone — outputs are written here directly by env_config when present.
_REMOTE_CLONE = Path("/kaggle/working/remote/Multi-Bot")
REPO_ROOT = _REMOTE_CLONE if _REMOTE_CLONE.exists() else _WORK_TS.parent

# The remote clone's trading-system/ — this is where weights actually live after training.
_REMOTE_TS = _REMOTE_CLONE / "trading-system" if _REMOTE_CLONE.exists() else _WORK_TS

# Working copy engine/ml dirs — used as fallback source if remote clone wasn't present
# during training (e.g. GITHUB_TOKEN wasn't set, so env_config fell back to working copy).
_WORK_ENGINE = _WORK_TS / "trading-engine"
_WORK_ML     = _WORK_TS / "ml_training"

# Remote clone engine/ml dirs — primary artifact location.
_REMOTE_ENGINE = _REMOTE_TS / "trading-engine"
_REMOTE_ML     = _REMOTE_TS / "ml_training"


# ── Artifact manifest ─────────────────────────────────────────────────────────
# Fallback copy list: (working_copy_source, dest_path_relative_to_REPO_ROOT)
# Only copied when the file is missing from the remote clone (i.e. env_config
# fell back to the working copy because the clone didn't exist at training time).
# If outputs were written directly to the clone, these copies are no-ops.

_FALLBACK_ARTIFACTS: list[tuple[Path, Path]] = [
    (_WORK_ENGINE / "weights" / "gru_lstm" / "model.pt",
     Path("trading-system/trading-engine/weights/gru_lstm/model.pt")),

    (_WORK_ENGINE / "weights" / "regime_4h.pkl",
     Path("trading-system/trading-engine/weights/regime_4h.pkl")),

    (_WORK_ENGINE / "weights" / "regime_1h.pkl",
     Path("trading-system/trading-engine/weights/regime_1h.pkl")),

    (_WORK_ENGINE / "weights" / "quality_scorer.pkl",
     Path("trading-system/trading-engine/weights/quality_scorer.pkl")),

    (_WORK_ENGINE / "weights" / "rl_ppo" / "model.zip",
     Path("trading-system/trading-engine/weights/rl_ppo/model.zip")),

    (_WORK_ENGINE / "weights" / "macro_correlations.json",
     Path("trading-system/trading-engine/weights/macro_correlations.json")),

    (_WORK_ENGINE / "weights" / "gru_lstm" / "weights_manifest.json",
     Path("trading-system/trading-engine/weights/gru_lstm/weights_manifest.json")),

    (_WORK_ML / "metrics" / "training_summary.json",
     Path("trading-system/ml_training/metrics/training_summary.json")),

    (_WORK_ML / "metrics" / "training_7b_summary.json",
     Path("trading-system/ml_training/metrics/training_7b_summary.json")),

    (_WORK_ENGINE / "logs" / "backtest_diagnostics.csv",
     Path("trading-system/trading-engine/logs/backtest_diagnostics.csv")),

    (_WORK_ENGINE / "logs" / "trade_journal_detailed.jsonl",
     Path("trading-system/trading-engine/logs/trade_journal_detailed.jsonl")),

    (_WORK_TS / "backtesting" / "results" / "latest_summary.json",
     Path("trading-system/backtesting/results/latest_summary.json")),
]

_FALLBACK_DIRS: list[tuple[Path, Path]] = [
    (_WORK_ENGINE / "backtest_results",
     Path("trading-system/trading-engine/backtest_results")),

    (_WORK_ML / "logs",
     Path("trading-system/ml_training/logs")),
]


def _run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    logger.debug("$ %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(cwd), check=check, capture_output=True, text=True)


def _copy_artifact(src: Path, dst_rel: Path) -> bool:
    """Copy src → REPO_ROOT/dst_rel only when dst doesn't already exist (direct-write case)."""
    if not src.exists():
        logger.debug("skip (missing source): %s", src)
        return False
    dst = REPO_ROOT / dst_rel
    # If the destination already exists and has the same content (direct-write path), skip.
    if dst.exists() and dst.stat().st_size == src.stat().st_size and dst.samefile(src) is False:
        # File is present but came from a different path — still copy to keep in sync.
        pass
    if dst.exists() and dst.resolve() == src.resolve():
        logger.debug("skip (same file): %s", dst_rel)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    logger.info("  fallback copied: %s → %s", src.name, dst_rel)
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
    logger.info("Repo:        %s", GITHUB_REPO)
    logger.info("Branch:      %s", GITHUB_BRANCH)
    logger.info("Repo root:   %s", REPO_ROOT)
    logger.info("Remote TS:   %s", _REMOTE_TS)
    logger.info("Direct-write:%s", "YES" if _REMOTE_CLONE.exists() else "NO (fallback copy mode)")

    remote_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"

    _run(["git", "config", "user.name",  GITHUB_USER],  cwd=REPO_ROOT)
    _run(["git", "config", "user.email", GITHUB_EMAIL], cwd=REPO_ROOT)
    _run(["git", "remote", "set-url", "origin", remote_url], cwd=REPO_ROOT)

    logger.info("Pulling latest from origin/%s ...", GITHUB_BRANCH)
    _run(["git", "pull", "--ff-only", "origin", GITHUB_BRANCH], cwd=REPO_ROOT)

    # ── Fallback copy: only needed when clone wasn't present during training ──
    # When env_config routed writes directly to the clone, these are no-ops
    # (dst already exists at same path as src, or dst is already newer).
    n_copied = 0
    for src, dst_rel in _FALLBACK_ARTIFACTS:
        if _copy_artifact(src, dst_rel):
            n_copied += 1
    for src_dir, dst_rel in _FALLBACK_DIRS:
        n_copied += _copy_dir(src_dir, dst_rel)
    if n_copied:
        logger.info("Fallback: copied %d file(s) from working copy", n_copied)

    # ── Stage, commit, push ───────────────────────────────────────────────────
    _run(["git", "add",
          "trading-system/trading-engine/weights/",
          "trading-system/trading-engine/backtest_results/",
          "trading-system/trading-engine/logs/",
          "trading-system/ml_training/",
          "trading-system/backtesting/results/",
          ".gitignore",
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
