"""
Helpers for reading persisted event logs from Redis.

Events are stored as a Redis list at key "debug:events" (LPUSH/LTRIM, max 500).
Trade-specific events are also mirrored to "debug:trades".
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from services.redis_client import get_redis_client


async def fetch_events(event_type: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    redis_client = get_redis_client()
    raw_items = await redis_client.lrange("debug:events", 0, limit - 1)
    events: List[Dict[str, Any]] = []
    wanted_type = (event_type or "").lower()
    for item in raw_items:
        try:
            event = json.loads(item)
        except json.JSONDecodeError:
            continue
        current_type = str(event.get("event_type") or "").lower()
        if wanted_type and current_type != wanted_type:
            continue
        if current_type:
            event["event_type"] = current_type
        events.append(event)
    events.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
    return events
