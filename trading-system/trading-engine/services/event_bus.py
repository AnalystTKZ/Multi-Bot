"""
event_bus.py — Redis pub/sub event bus with EventType enum.

Contract 1 event schemas are enforced here.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    MARKET_DATA = "MARKET_DATA"
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    TRADE_EXECUTED = "TRADE_EXECUTED"
    TRADE_CLOSED = "TRADE_CLOSED"
    TRADE_REQUESTED = "TRADE_REQUESTED"
    ML_SIGNAL_GENERATED = "ML_SIGNAL_GENERATED"
    ENGINE_STATUS = "ENGINE_STATUS"
    START_TRADER = "start_trader"
    STOP_TRADER = "stop_trader"


class EventBus:
    """Redis pub/sub wrapper."""

    def __init__(self, redis_client):
        self._r = redis_client
        self._pubsub = None
        self._handlers: Dict[str, list] = {}

    def publish(self, event_type: str, payload: dict) -> None:
        try:
            self._r.publish(event_type, json.dumps(payload))
        except Exception as exc:
            logger.error("EventBus.publish(%s) failed: %s", event_type, exc)

    def subscribe(self, event_type: str, handler: Callable[[dict], None]) -> None:
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def start_listening(self) -> None:
        """Block-and-dispatch loop. Run in a dedicated thread."""
        self._pubsub = self._r.pubsub()
        channels = list(self._handlers.keys())
        if not channels:
            logger.warning("EventBus: no channels to listen on")
            return
        self._pubsub.subscribe(*channels)
        logger.info("EventBus: subscribed to %s", channels)

        for message in self._pubsub.listen():
            if message["type"] != "message":
                continue
            channel = message["channel"]
            if isinstance(channel, bytes):
                channel = channel.decode()
            try:
                data = json.loads(message["data"])
            except (json.JSONDecodeError, TypeError):
                continue

            for handler in self._handlers.get(channel, []):
                try:
                    handler(data)
                except Exception as exc:
                    logger.error("EventBus handler error on %s: %s", channel, exc)

    def stop(self) -> None:
        if self._pubsub:
            self._pubsub.unsubscribe()
            self._pubsub.close()
