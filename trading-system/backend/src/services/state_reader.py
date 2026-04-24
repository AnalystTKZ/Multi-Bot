"""
Read-only access to trading-engine Redis state.

Contract (from CLAUDE.md "Engine writes"):
  positions:open              — JSON list of open positions
  positions:{ticket}:data     — individual position JSON (legacy/extended)
  strategy_allocations:{id}   — allocation per trader
  portfolio                   — portfolio snapshot
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from services.redis_client import get_redis_client


async def get_positions() -> List[Dict[str, Any]]:
    redis_client = get_redis_client()

    # Primary: read the bulk positions:open key (JSON list)
    data = await redis_client.get("positions:open")
    if data:
        try:
            parsed = json.loads(data)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    # Fallback: scan for individual positions:{ticket}:data keys
    positions: List[Dict[str, Any]] = []
    async for key in redis_client.scan_iter(match="positions:*:data"):
        item = await redis_client.get(key)
        if not item:
            continue
        try:
            positions.append(json.loads(item))
        except json.JSONDecodeError:
            continue
    return positions


async def get_portfolio_state() -> Optional[Dict[str, Any]]:
    redis_client = get_redis_client()
    data = await redis_client.get("portfolio")
    if not data:
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


async def get_market_contexts() -> List[Dict[str, Any]]:
    redis_client = get_redis_client()
    contexts: List[Dict[str, Any]] = []
    async for key in redis_client.scan_iter(match="context:*"):
        data = await redis_client.get(key)
        if not data:
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        symbol = key.split(":", 1)[-1]
        contexts.append({
            "symbol": symbol,
            "regime": payload.get("regime"),
            "bias": payload.get("bias"),
            "confidence": payload.get("confidence"),
        })
    return sorted(contexts, key=lambda item: item.get("symbol") or "")


_KNOWN_TRADERS = {"trader_1", "trader_2", "trader_3", "trader_4", "trader_5", "ml_trader"}

_DEFAULT_ALLOCATIONS = [
    {"strategy_id": "ml_trader", "is_active": True, "allocated_capital": None, "used_capital": None, "current_risk": None},
]


async def get_strategy_allocations() -> List[Dict[str, Any]]:
    redis_client = get_redis_client()
    allocations: List[Dict[str, Any]] = []
    async for key in redis_client.scan_iter(match="strategy_allocations:*"):
        data = await redis_client.get(key)
        if not data:
            continue
        try:
            allocation = json.loads(data)
        except json.JSONDecodeError:
            continue
        # Skip stale Redis keys for traders that no longer exist
        if allocation.get("strategy_id") not in _KNOWN_TRADERS:
            continue
        allocations.append(allocation)
    # Fall back to defaults when the trading engine hasn't populated Redis yet
    if not allocations:
        return _DEFAULT_ALLOCATIONS
    return allocations
