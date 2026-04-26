"""
env_config.py — Environment abstraction layer.

Resolves all paths from the actual filesystem — no hardcoded Kaggle paths
outside this file. Every other script imports get_env() and uses the keys.

Keys returned by get_env():
    base        trading-system/ root
    repo        git repo root (parent of trading-system/, auto-detected via .git walk)
    data        training_data/
    processed   processed_data/
    ml_training ml_training/
    weights     trading-engine/weights/
    engine      trading-engine/
    pipeline    pipeline/
    output      writable output root (/kaggle/working on Kaggle, base locally)
    on_kaggle   bool
"""
from __future__ import annotations
import os
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    """Walk up from start until we find a .git directory, then return that dir."""
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent
    # No .git found — fall back to start's parent (best-effort)
    return start.parent


def _find_kaggle_dataset() -> Path | None:
    """
    Find the Kaggle dataset directory that contains training_data/ or processed_data/.

    Resolution order:
      1. KAGGLE_DATA_PATH env var (set in notebook: DATA_PATH)
      2. KAGGLE_DATASET_SLUG env var  → /kaggle/input/<slug>/
      3. Auto-scan /kaggle/input/ recursively (handles nested dataset paths)
    """
    # 1. Explicit full path from notebook
    data_path_env = os.getenv("KAGGLE_DATA_PATH", "") or os.getenv("DATA_PATH", "")
    if data_path_env:
        candidate = Path(data_path_env)
        if candidate.exists():
            return candidate

    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return None

    # 2. Slug-based lookup
    slug = os.getenv("KAGGLE_DATASET_SLUG", "")
    if slug:
        candidate = kaggle_input / slug
        if candidate.exists():
            return candidate

    # 3. Auto-scan: check direct children first, then one level deeper
    def _has_data(p: Path) -> bool:
        return (p / "training_data").exists() or (p / "processed_data").exists()

    try:
        for entry in sorted(kaggle_input.iterdir()):
            if entry.is_dir():
                if _has_data(entry):
                    return entry
                # Check one level deeper (e.g. /kaggle/input/datasets/user/name/)
                for sub in sorted(entry.rglob("training_data")):
                    candidate = sub.parent
                    if _has_data(candidate):
                        return candidate
    except PermissionError:
        pass

    return None


def get_env() -> dict[str, Path]:
    on_kaggle = os.path.exists("/kaggle/input")

    # base is always the trading-system/ directory this file lives in.
    # On Kaggle that resolves to wherever the repo was cloned — no hardcoded path.
    base   = Path(__file__).resolve().parent
    repo   = _find_repo_root(base)
    # output: writable scratch root. On Kaggle this is the parent of the repo
    # clone (e.g. /kaggle/working/); locally it's trading-system/ itself.
    output = repo.parent if on_kaggle else base

    dataset = _find_kaggle_dataset() if on_kaggle else None
    if dataset is not None:
        data_dir = dataset / "training_data"
        proc_dir = dataset / "processed_data"
    else:
        data_dir = base / "training_data"
        proc_dir = base / "processed_data"

    return {
        "base":        base,
        "repo":        repo,
        "data":        data_dir,
        "processed":   proc_dir,
        "ml_training": base / "ml_training",
        "weights":     base / "trading-engine" / "weights",
        "engine":      base / "trading-engine",
        "pipeline":    base / "pipeline",
        "output":      output,
        "on_kaggle":   on_kaggle,  # type: ignore[dict-item]
    }


def ensure_output_dirs(env: dict) -> None:
    """Create all writable output directories if they don't exist."""
    dirs = [
        env["ml_training"] / "datasets",
        env["ml_training"] / "metrics",
        env["ml_training"] / "logs",
        env["weights"] / "gru_lstm",
        env["weights"] / "rl_ppo",
        env["weights"] / "vector_store",
        env["engine"] / "logs",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    env = get_env()
    label = "KAGGLE" if env["on_kaggle"] else "LOCAL"
    print(f"Running on: {label}")
    for k, v in env.items():
        if k != "on_kaggle":
            print(f"  {k:12s} → {v}")
