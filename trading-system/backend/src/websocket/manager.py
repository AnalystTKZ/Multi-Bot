"""
WebSocket connection manager and Redis event bridge.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Set

from fastapi import WebSocket

from routes.auth import AUTH_COOKIE_NAME, validate_access_token
from services.redis_client import get_redis_client
from utils.observability import log_event

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()
        self._listener_task: asyncio.Task | None = None

    async def connect(self, websocket: WebSocket, token: str | None = None) -> bool:
        resolved_token = token or websocket.query_params.get("token") or websocket.cookies.get(AUTH_COOKIE_NAME)
        if not resolved_token:
            log_event(
                logger,
                "warning",
                module="websocket_manager",
                event_type="WS_CONNECTION_DENIED",
                message="WebSocket rejected: missing token",
                correlation_id=None,
                data={"reason": "missing_token"},
            )
            await websocket.close(code=1008)
            return False
        try:
            user = validate_access_token(resolved_token)
        except Exception:
            log_event(
                logger,
                "warning",
                module="websocket_manager",
                event_type="WS_CONNECTION_DENIED",
                message="WebSocket rejected: invalid token",
                correlation_id=None,
                data={"reason": "invalid_token"},
            )
            await websocket.close(code=1008)
            return False
        await websocket.accept()
        websocket.scope["auth_user"] = user
        self.active_connections.add(websocket)
        log_event(
            logger,
            "info",
            module="websocket_manager",
            event_type="WS_CONNECTION_STATUS",
            message="WebSocket client connected",
            correlation_id=None,
            data={"connections": len(self.active_connections), "username": user.get("username")},
        )
        return True

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.discard(websocket)
        log_event(
            logger,
            "info",
            module="websocket_manager",
            event_type="WS_CONNECTION_STATUS",
            message="WebSocket client disconnected",
            correlation_id=None,
            data={"connections": len(self.active_connections)},
        )

    async def broadcast(self, message: Dict[str, Any]) -> None:
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as exc:
                log_event(
                    logger,
                    "error",
                    module="websocket_manager",
                    event_type="WS_ERROR",
                    message="WebSocket send error",
                    correlation_id=message.get("correlation_id"),
                    strategy_id=message.get("payload", {}).get("strategy_id"),
                    data={"error": str(exc)},
                )
                disconnected.add(connection)
        for connection in disconnected:
            self.disconnect(connection)

    async def start_event_listener(self) -> None:
        redis_client = get_redis_client()
        pubsub = redis_client.pubsub()
        backend_channels = [
            "events:signal_generated",
            "events:trade_executed",
            "events:trade_failed",
            "events:portfolio_updated",
            "events:risk_violation",
            "events:risk_update",
            "events:system_alert",
            "events:position_updated",
            "events:strategy_status",
            "events:market_data",
        ]
        engine_channels = [
            "SIGNAL_GENERATED",
            "TRADE_EXECUTED",
            "TRADE_CLOSED",
            "ENGINE_STATUS",
            "MARKET_DATA",
        ]
        channels = backend_channels + engine_channels
        await pubsub.subscribe(*channels)
        logger.info("Subscribed to Redis channels: %s", channels)

        try:
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                channel = message.get("channel")
                if isinstance(channel, bytes):
                    channel = channel.decode()
                try:
                    data = json.loads(message.get("data"))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed event payload")
                    continue
                event = self._normalize_event(channel, data)
                log_event(
                    logger,
                    "info",
                    module="websocket_manager",
                    event_type="WS_MESSAGE_SENT",
                    message="Forwarding event to frontend clients",
                    correlation_id=event.get("correlation_id"),
                    strategy_id=event.get("payload", {}).get("strategy_id"),
                    data={"event_type": event.get("event_type"), "connections": len(self.active_connections)},
                )
                if channel in engine_channels:
                    await self._persist_engine_event(redis_client, event)
                await self.broadcast(event)
        except asyncio.CancelledError:
            pass
        finally:
            await pubsub.unsubscribe(*channels)
            await pubsub.close()

    def _normalize_event(self, channel: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if "event_type" in data and "payload" in data:
            event_type = str(data.get("event_type", "")).lower()
            return {**data, "event_type": event_type}

        event_type = str(channel or "").replace("events:", "").lower()
        correlation_id = data.get("correlation_id") or data.get("request_id")
        payload = data if isinstance(data, dict) else {"value": data}
        return {
            "event_type": event_type,
            "payload": payload,
            "timestamp": payload.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            "source": "trading-engine",
            "event_id": payload.get("event_id") or correlation_id or f"{event_type}:{payload.get('symbol', 'system')}",
            "correlation_id": correlation_id,
        }

    async def _persist_engine_event(self, redis_client, event: Dict[str, Any]) -> None:
        serialised = json.dumps(event)
        await redis_client.lpush("debug:events", serialised)
        await redis_client.ltrim("debug:events", 0, 499)
        if event.get("event_type") in {"signal_generated", "trade_executed", "trade_closed"}:
            await redis_client.lpush("debug:trades", serialised)
            await redis_client.ltrim("debug:trades", 0, 499)

    def run_listener(self) -> None:
        if self._listener_task is None or self._listener_task.done():
            self._listener_task = asyncio.create_task(self.start_event_listener())

    async def stop_listener(self) -> None:
        if self._listener_task:
            self._listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener_task
            self._listener_task = None


manager = ConnectionManager()
