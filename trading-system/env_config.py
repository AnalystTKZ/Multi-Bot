"""
env_config.py — Environment abstraction layer.

Resolves all root paths based on where the code is running:
  - Kaggle (split-dataset setup):
      Code clone: /kaggle/working/Multi-Bot/  (no training_data or processed_data)
      Dataset:    /kaggle/input/<slug>/        (training_data/ + processed_data/, read-only)
      Outputs:    written back to base/ paths (weights, logs, ml_training as before)
  - Local: everything relative to this file's parent (trading-system/)

Usage (in any pipeline script or model script):
    from env_config import get_env

    env = get_env()
    BASE       = env["base"]        # trading-system/ root (code)
    DATA_PATH  = env["data"]        # training_data/  (dataset source on Kaggle)
    PROC_DATA  = env["processed"]   # processed_data/ (dataset source on Kaggle)
    OUTPUT     = env["output"]      # writable output root
    WEIGHTS    = env["weights"]     # trading-engine/weights/
    ML_DIR     = env["ml_training"] # ml_training/
"""
from __future__ import annotations
import os
from pathlib import Path


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

    if on_kaggle:
        base = Path("/kaggle/working/Multi-Bot/trading-system")
        output = Path("/kaggle/working")
        dataset = _find_kaggle_dataset()
        if dataset is not None:
            # Data sources come from the read-only dataset mount
            data_dir = dataset / "training_data"
            proc_dir = dataset / "processed_data"
        else:
            # Fallback: old single-repo layout where data lived in the clone
            data_dir = base / "training_data"
            proc_dir = base / "processed_data"
    else:
        base     = Path(__file__).resolve().parent
        output   = base
        data_dir = base / "training_data"
        proc_dir = base / "processed_data"

    return {
        "base":        base,
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
