"""
Backend event bus publisher for Redis Pub/Sub.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from services.redis_client import get_redis_client
from utils.observability import ensure_correlation_id, log_event


@dataclass
class Event:
    event_type: str
    payload: Dict[str, Any]
    source: str
    event_id: str
    timestamp: str
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "source": self.source,
            "event_id": self.event_id,
            "correlation_id": self.correlation_id,
        }


async def publish_event(event_type: str, payload: Dict[str, Any], source: str = "backend", correlation_id: Optional[str] = None) -> Event:
    correlation_id = ensure_correlation_id(correlation_id)
    event = Event(
        event_type=event_type,
        payload=payload,
        source=source,
        event_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        correlation_id=correlation_id,
    )
    redis_client = get_redis_client()
    channel = f"events:{event_type}"
    await redis_client.publish(channel, json.dumps(event.to_dict()))
    await redis_client.lpush("debug:events", json.dumps(event.to_dict()))
    await redis_client.ltrim("debug:events", 0, 499)
    if event_type in {"trade_requested", "trade_executed", "trade_failed", "signal_generated"}:
        await redis_client.lpush("debug:trades", json.dumps(event.to_dict()))
        await redis_client.ltrim("debug:trades", 0, 499)
    log_event(
        logging.getLogger(__name__),
        "info",
        module="backend_event_bus",
        event_type=event_type,
        message="Backend event published",
        correlation_id=correlation_id,
        strategy_id=payload.get("strategy_id"),
        data={"source": source, "channel": channel, "event_id": event.event_id},
    )
    return event
