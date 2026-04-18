"""
System, Backtest, Training, and ML API routes
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from routes.auth import get_current_user
from services.event_bus import publish_event
from services.redis_client import get_redis_client

router = APIRouter(dependencies=[Depends(get_current_user)])

# Resolve paths — in Docker the volumes are mounted under /app/
_APP_DIR = Path("/app")
_BACKTEST_DIR = _APP_DIR / "backtest_results"
_ML_WEIGHTS_DIR = _APP_DIR / "trading-engine" / "models" / "weights"
_ML_LOGS_DIR = _APP_DIR / "trading-engine" / "logs"
_RETRAIN_LOG = _ML_LOGS_DIR / "retrain_history.jsonl"


def _coerce_value(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
    return value


async def _read_redis_object(redis_client, key: str) -> Dict[str, Any] | None:
    data = await redis_client.get(key)
    if data:
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            pass

    mapping = await redis_client.hgetall(key)
    if mapping:
        return {_coerce_value(k): _coerce_value(v) for k, v in mapping.items()}

    return None


def _tail_jsonl(path: Path, limit: int = 50) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        from collections import deque
        buf: deque[Dict[str, Any]] = deque(maxlen=limit)
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    buf.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return list(buf)
    except Exception:
        return []


# ─────────────────────────── SYSTEM ───────────────────────────

@router.get("/system/status")
async def system_status():
    return {
        "status": "operational",
        "paper_trading": os.getenv("PAPER_TRADING", "true").lower() == "true",
        "trading_enabled": os.getenv("TRADING_ENABLED", "true").lower() == "true",
        "mode": "paper" if os.getenv("PAPER_TRADING", "true").lower() == "true" else "live",
    }


class ModePayload(BaseModel):
    mode: str  # 'paper' | 'live'


@router.post("/system/mode")
async def set_mode(payload: ModePayload):
    if payload.mode not in ("paper", "live", "backtest", "training"):
        raise HTTPException(status_code=400, detail="mode must be 'paper', 'live', 'backtest', or 'training'")
    await publish_event("system_mode_change", {"mode": payload.mode})
    return {"message": f"Mode change to '{payload.mode}' requested"}


# ─────────────────────────── BACKTEST ─────────────────────────

@router.get("/backtest/config")
async def get_backtest_config():
    return {
        "symbols": os.getenv("TRADING_PAIRS", "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,XAUUSD").split(","),
        "default_initial_capital": float(os.getenv("TOTAL_CAPITAL", "100000")),
        "default_commission_pct": 0.0002,
        "default_slippage_pct": 0.0001,
    }


@router.get("/backtest/results")
async def list_backtest_results():
    if not _BACKTEST_DIR.exists():
        return {"results": []}
    files = sorted(_BACKTEST_DIR.glob("backtest_*.json"), reverse=True)
    results: List[Dict[str, Any]] = []
    for f in files[:20]:
        try:
            data = json.loads(f.read_text())
            results.append({
                "id": f.stem,
                "timestamp": data.get("run_at", f.stem),
                "total_return": data.get("results", {}).get("total_return"),
                "win_rate": data.get("results", {}).get("win_rate"),
                "total_trades": data.get("results", {}).get("total_trades"),
                "net_pnl": data.get("results", {}).get("net_pnl"),
            })
        except Exception:
            continue
    return {"results": results}


@router.get("/backtest/results/{result_id}")
async def get_backtest_result(result_id: str):
    path = _BACKTEST_DIR / f"{result_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    return json.loads(path.read_text())


@router.post("/backtest/run")
async def run_backtest(config: Dict[str, Any]):
    await publish_event("backtest_requested", config)
    return {"message": "Backtest queued — results will appear when complete", "status": "queued"}


# ─────────────────────────── TRAINING ─────────────────────────

@router.get("/training/status")
async def training_status():
    redis_client = get_redis_client()
    models = []
    events = _tail_jsonl(_RETRAIN_LOG, limit=50)
    # Read live status from Redis first (written by trading engine state_manager)
    async for key in redis_client.scan_iter(match="ml:model:*"):
        info = await _read_redis_object(redis_client, key)
        if not info:
            continue
        models.append({
            "name": info.get("name", key.split(":")[-1]),
            "status": info.get("status"),
            "accuracy": info.get("accuracy"),
            "last_trained": info.get("last_trained"),
        })
    # Fall back to filesystem scan if Redis has no ML state yet
    if not models and _ML_WEIGHTS_DIR.exists():
        for f in _ML_WEIGHTS_DIR.glob("*.pkl"):
            models.append({"name": f.stem, "size_kb": round(f.stat().st_size / 1024, 1)})
    return {"status": "idle", "models": models, "events": events}


@router.post("/training/start")
async def start_training(config: Dict[str, Any]):
    await publish_event("training_requested", config)
    return {"message": "Training job queued", "status": "queued"}


@router.post("/training/upload")
async def upload_training_data():
    return {"message": "Upload endpoint not yet implemented", "status": "not_implemented"}


# ─────────────────────────── ML ───────────────────────────────

@router.get("/ml/models")
async def get_ml_models():
    redis_client = get_redis_client()
    ml_enabled = os.getenv("ML_ENABLED", "false").lower() == "true"
    models = []
    # Read live model status from Redis first
    async for key in redis_client.scan_iter(match="ml:model:*"):
        info = await _read_redis_object(redis_client, key)
        if not info:
            continue
        model_id = info.get("model_id") or key.split(":")[-1]
        models.append({
            "id": model_id,
            "name": info.get("name", model_id.replace("_", " ").title()),
            "status": info.get("status"),
            "accuracy": info.get("accuracy"),
            "last_trained": info.get("last_trained"),
            "enabled": ml_enabled,
        })
    # Fall back to filesystem scan if Redis has no ML state yet
    if not models and _ML_WEIGHTS_DIR.exists():
        for f in _ML_WEIGHTS_DIR.glob("*.pkl"):
            models.append({
                "id": f.stem,
                "name": f.stem.replace("_", " ").title(),
                "size_kb": round(f.stat().st_size / 1024, 1),
                "enabled": ml_enabled,
            })
    return {"models": models}


@router.get("/ml/models/{model_id}")
async def get_ml_model(model_id: str):
    redis_client = get_redis_client()
    ml_enabled = os.getenv("ML_ENABLED", "false").lower() == "true"
    # Try Redis first
    info = await _read_redis_object(redis_client, f"ml:model:{model_id}")
    if info:
        return {
            "id": model_id,
            "name": info.get("name", model_id.replace("_", " ").title()),
            "status": info.get("status"),
            "accuracy": info.get("accuracy"),
            "last_trained": info.get("last_trained"),
            "enabled": ml_enabled,
        }
    # Fall back to filesystem
    path = _ML_WEIGHTS_DIR / f"{model_id}.pkl"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    return {
        "id": model_id,
        "name": model_id.replace("_", " ").title(),
        "size_kb": round(path.stat().st_size / 1024, 1),
        "enabled": ml_enabled,
    }


@router.post("/ml/models/{model_id}/retrain")
async def retrain_model(model_id: str):
    await publish_event("retrain_requested", {"model_id": model_id})
    return {"message": f"Retrain of '{model_id}' queued"}


@router.get("/ml/rl-agent")
async def get_rl_agent():
    redis_client = get_redis_client()
    ml_enabled = os.getenv("ML_ENABLED", "false").lower() == "true"
    # Read live RL agent state from Redis (written by engine state_manager)
    info = await _read_redis_object(redis_client, "ml:rl_agent")
    if info:
        return {
            "enabled": ml_enabled,
            "status": "active" if ml_enabled else "idle",
            "algorithm": info.get("algorithm", "PPO"),
            "episodes": info.get("episodes", 0),
            "avg_reward": info.get("avg_reward", 0.0),
        }
    return {
        "enabled": ml_enabled,
        "status": "idle",
        "episodes": 0,
        "avg_reward": 0.0,
    }
