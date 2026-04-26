"""
step8_push_to_github.py — Push training outputs back to GitHub.

The repo lives at /kaggle/working/Multi-Bot/ (cloned at notebook start).
All pipeline outputs are written directly into that clone, so this script
only needs to stage, commit, and push — no intermediate copy step.

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

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_config import get_env as _get_env
_ENV = _get_env()

_TS       = _ENV["base"]    # trading-system/
REPO_ROOT = _ENV["repo"]    # git repo root — auto-detected, no hardcoded path

def _rel(*parts: str) -> str:
    return str((_TS / Path(*parts)).relative_to(REPO_ROOT))

_STAGE_PATHS = [
    _rel("trading-engine", "weights"),
    _rel("trading-engine", "backtest_results"),
    _rel("trading-engine", "logs"),
    _rel("ml_training"),
    _rel("backtesting", "results"),
]


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    logger.debug("$ %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(REPO_ROOT), check=check,
                          capture_output=True, text=True)


def main() -> None:
    if not GITHUB_TOKEN:
        logger.error("GITHUB_TOKEN not set — cannot push to GitHub")
        sys.exit(1)

    logger.info("=== STEP 8: PUSH TRAINING OUTPUTS TO GITHUB ===")
    logger.info("Repo:      %s", GITHUB_REPO)
    logger.info("Branch:    %s", GITHUB_BRANCH)
    logger.info("Repo root: %s", REPO_ROOT)

    remote_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"

    _run(["git", "config", "user.name",  GITHUB_USER])
    _run(["git", "config", "user.email", GITHUB_EMAIL])
    _run(["git", "remote", "set-url", "origin", remote_url])

    logger.info("Pulling latest from origin/%s ...", GITHUB_BRANCH)
    _run(["git", "pull", "--ff-only", "origin", GITHUB_BRANCH])

    # Stage all training outputs (vector_store/ is gitignored)
    _run(["git", "add", *_STAGE_PATHS])

    status = _run(["git", "status", "--porcelain"])
    if not status.stdout.strip():
        logger.info("Nothing changed — skipping push")
        return

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    _run(["git", "commit", "-m", f"chore(kaggle): training outputs {ts}"])

    logger.info("Pushing to origin/%s ...", GITHUB_BRANCH)
    _run(["git", "push", "origin", f"HEAD:{GITHUB_BRANCH}"])

    logger.info("=== PUSH COMPLETE ===")


if __name__ == "__main__":
    main()
