from __future__ import annotations

import json
from typing import Any, Dict, List

from fastapi import APIRouter, Depends

from routes.auth import get_current_user
from services.redis_client import get_redis_client
from services import state_reader

router = APIRouter(dependencies=[Depends(get_current_user)])


async def _read_list(key: str, limit: int = 200) -> List[Dict[str, Any]]:
    redis_client = get_redis_client()
    values = await redis_client.lrange(key, 0, max(limit - 1, 0))
    parsed: List[Dict[str, Any]] = []
    for raw in values:
        try:
            parsed.append(json.loads(raw))
        except Exception:
            continue
    return parsed


@router.get("/events")
async def get_recent_events(limit: int = 200):
    return {"events": await _read_list("debug:events", limit=limit)}


@router.get("/trades")
async def get_trade_lifecycle_logs(limit: int = 200):
    return {"trades": await _read_list("debug:trades", limit=limit)}


@router.get("/state")
async def get_current_state():
    positions = await state_reader.get_positions()
    portfolio = await state_reader.get_portfolio_state()
    allocations = await state_reader.get_strategy_allocations()
    return {"positions": positions, "portfolio": portfolio, "strategy_allocations": allocations}
