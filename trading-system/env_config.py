"""
env_config.py — Environment abstraction layer.

Resolves all root paths based on where the code is running:
  - Kaggle:  code in /kaggle/working/Multi-Bot, outputs to /kaggle/working
  - Local:   everything relative to this file's parent (trading-system/)

Usage (in any pipeline script or model script):
    from env_config import get_env

    env = get_env()
    BASE       = env["base"]        # trading-system/ root
    DATA_PATH  = env["data"]        # training_data/
    OUTPUT     = env["output"]      # writable output root
    WEIGHTS    = env["weights"]     # trading-engine/weights/
    ML_DIR     = env["ml_training"] # ml_training/
    PROC_DATA  = env["processed"]   # processed_data/
"""
from __future__ import annotations
import os
from pathlib import Path


def get_env() -> dict[str, Path]:
    on_kaggle = os.path.exists("/kaggle/input")

    if on_kaggle:
        # Code is cloned to /kaggle/working/Multi-Bot via notebook setup cell
        base = Path("/kaggle/working/Multi-Bot/trading-system")
        output = Path("/kaggle/working")
    else:
        # Local: resolve relative to this file
        base = Path(__file__).resolve().parent
        output = base

    return {
        "base":        base,
        "data":        base / "training_data",
        "processed":   base / "processed_data",
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
        env["processed"] / "histdata",
        env["processed"] / "clean",
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
