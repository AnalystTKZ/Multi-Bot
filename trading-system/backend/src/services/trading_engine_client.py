"""
Trading Engine HTTP client for backend API gateway.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class TradingEngineClient:
    """Async client for the Trading Engine internal API."""

    def __init__(self) -> None:
        base_url = os.getenv("TRADING_ENGINE_URL", "http://localhost:8000")
        self._client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    async def get_traders(self) -> Dict[str, Any]:
        response = await self._client.get("/traders")
        response.raise_for_status()
        return response.json()

    async def get_trader(self, strategy_id: str) -> Dict[str, Any]:
        response = await self._client.get(f"/traders/{strategy_id}")
        response.raise_for_status()
        return response.json()

    async def enable_trader(self, strategy_id: str) -> Dict[str, Any]:
        response = await self._client.post(f"/traders/{strategy_id}/enable")
        response.raise_for_status()
        return response.json()

    async def disable_trader(self, strategy_id: str) -> Dict[str, Any]:
        response = await self._client.post(f"/traders/{strategy_id}/disable")
        response.raise_for_status()
        return response.json()

    async def get_portfolio(self) -> Dict[str, Any]:
        response = await self._client.get("/portfolio")
        response.raise_for_status()
        return response.json()

    async def get_risk(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        params = {"strategy_id": strategy_id} if strategy_id else None
        response = await self._client.get("/risk", params=params)
        response.raise_for_status()
        return response.json()

    async def get_alerts(self, limit: int = 100) -> Dict[str, Any]:
        response = await self._client.get("/alerts", params={"limit": limit})
        response.raise_for_status()
        return response.json()

    async def get_system_status(self) -> Dict[str, Any]:
        response = await self._client.get("/status")
        response.raise_for_status()
        return response.json()

    async def get_metrics_text(self) -> str:
        response = await self._client.get("/metrics")
        response.raise_for_status()
        return response.text

    async def close(self) -> None:
        await self._client.aclose()


_client: Optional[TradingEngineClient] = None


def get_trading_engine_client() -> TradingEngineClient:
    """Get or create a singleton TradingEngineClient."""
    global _client
    if _client is None:
        _client = TradingEngineClient()
    return _client
