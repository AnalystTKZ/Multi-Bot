"""
broker_connector.py — Capital.com REST API connector.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

_DEMO_BASE = "https://api-demo.capital.com"
_LIVE_BASE = "https://api-capital.backend.gbooking.co.uk"
_SESSION_REFRESH_SECONDS = 590  # ~9m50s


class CapitalComSession:
    def __init__(self, api_key: str, identifier: str, password: str, env: str = "demo"):
        self._api_key = api_key
        self._identifier = identifier
        self._password = password
        self._base = _DEMO_BASE if env == "demo" else _LIVE_BASE
        self._cst: Optional[str] = None
        self._x_security: Optional[str] = None
        self._last_auth: float = 0.0
        self._session = requests.Session()

    def _headers(self) -> dict:
        h = {"X-CAP-API-KEY": self._api_key, "Content-Type": "application/json"}
        if self._cst:
            h["CST"] = self._cst
        if self._x_security:
            h["X-SECURITY-TOKEN"] = self._x_security
        return h

    def authenticate(self) -> bool:
        try:
            resp = self._session.post(
                f"{self._base}/api/v1/session",
                json={"identifier": self._identifier, "password": self._password},
                headers={"X-CAP-API-KEY": self._api_key, "Content-Type": "application/json"},
                timeout=15,
            )
            if resp.status_code == 200:
                self._cst = resp.headers.get("CST")
                self._x_security = resp.headers.get("X-SECURITY-TOKEN")
                self._last_auth = time.time()
                logger.info("Capital.com: authenticated")
                return True
            logger.error("Capital.com auth failed: %s %s", resp.status_code, resp.text[:200])
            return False
        except Exception as exc:
            logger.error("Capital.com auth error: %s", exc)
            return False

    def _ensure_session(self) -> None:
        if time.time() - self._last_auth > _SESSION_REFRESH_SECONDS:
            self.authenticate()

    def get(self, path: str, params: dict = None) -> Optional[dict]:
        self._ensure_session()
        try:
            resp = self._session.get(
                f"{self._base}{path}",
                headers=self._headers(),
                params=params or {},
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error("Capital.com GET %s failed: %s", path, exc)
            return None

    def post(self, path: str, body: dict) -> Optional[dict]:
        self._ensure_session()
        try:
            resp = self._session.post(
                f"{self._base}{path}",
                json=body,
                headers=self._headers(),
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error("Capital.com POST %s failed: %s", path, exc)
            return None


class BrokerConnector:
    """Capital.com live order execution."""

    def __init__(self, session: CapitalComSession, event_bus, state_manager=None):
        self._session = session
        self._bus = event_bus
        self._state = state_manager

    def execute(self, signal: dict, portfolio_state: dict) -> Optional[dict]:
        import uuid
        from services.event_bus import EventType

        symbol = signal.get("symbol", "")
        side = signal.get("side", "buy")
        size = float(signal.get("size", 0.01))
        stop_loss = signal.get("stop_loss")
        take_profit = signal.get("take_profit")
        trader_id = signal.get("trader_id", "")
        confidence = float(signal.get("confidence", 0.6))
        correlation_id = signal.get("correlation_id", str(uuid.uuid4()))
        meta = signal.get("signal_metadata", {}) or {}

        direction = "BUY" if side == "buy" else "SELL"
        body = {
            "epic": symbol,
            "direction": direction,
            "size": str(size),
            "guaranteedStop": False,
            "stopLevel": stop_loss,
            "profitLevel": take_profit,
        }

        result = self._session.post("/api/v1/positions", body)
        if result is None:
            return None

        ticket = result.get("dealId", str(uuid.uuid4()))
        entry = float(result.get("level", signal.get("entry", 0.0)))

        rr_num = (take_profit - entry) if side == "buy" else (entry - take_profit)
        rr_den = (entry - stop_loss) if side == "buy" else (stop_loss - entry)
        rr_ratio = round(rr_num / (rr_den + 1e-9), 2)

        trade_event = {
            "ticket": ticket,
            "trader_id": trader_id,
            "symbol": symbol,
            "side": side,
            "size": size,
            "entry": round(entry, 6),
            "stop_loss": round(float(stop_loss or 0), 6),
            "take_profit": round(float(take_profit or 0), 6),
            "rr_ratio": rr_ratio,
            "confidence": confidence,
            "strategy_id": trader_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pnl": None,
            "signal_metadata": meta,
            "commission": 0.0,
            "correlation_id": correlation_id,
        }

        self._bus.publish(EventType.TRADE_EXECUTED, trade_event)
        if self._state:
            self._state.set_open_position(ticket, {**trade_event, "status": "open"})

        return trade_event
