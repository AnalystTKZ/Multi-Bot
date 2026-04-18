"""
state_manager.py — Redis state writer.

Writes all Contract 2 Redis keys so backend state_reader.py can read them.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger(__name__)


class StateManager:
    """Wraps Redis client and writes all Contract 2 key patterns."""

    def __init__(self, redis_client):
        self._r = redis_client

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ─── Engine state ─────────────────────────────────────────────────────────

    def set_engine_status(self, status: str) -> None:
        """engine:status = 'running' | 'stopped' | 'error'"""
        try:
            self._r.set("engine:status", status)
            self._r.set("engine:last_heartbeat", self._now_iso())
        except Exception as exc:
            logger.debug("StateManager.set_engine_status: %s", exc)

    def set_engine_mode(self, mode: str) -> None:
        """engine:mode = 'paper' | 'live'"""
        try:
            self._r.set("engine:mode", mode)
        except Exception as exc:
            logger.debug("StateManager.set_engine_mode: %s", exc)

    def heartbeat(self) -> None:
        try:
            self._r.set("engine:last_heartbeat", self._now_iso())
        except Exception as exc:
            logger.debug("StateManager.heartbeat: %s", exc)

    # ─── Trader state ─────────────────────────────────────────────────────────

    def set_trader_state(self, trader_id: str, state: dict) -> None:
        """trader:{trader_id}:state — used by GET /api/traders/{id}"""
        try:
            self._r.hset(f"trader:{trader_id}:state", mapping={
                "status": str(state.get("status", "active")),
                "trades_today": int(state.get("trades_today", 0)),
                "pnl_today": float(state.get("pnl_today", 0.0)),
                "win_rate": float(state.get("win_rate", 0.0)),
                "last_signal": str(state.get("last_signal", self._now_iso())),
            })
        except Exception as exc:
            logger.debug("StateManager.set_trader_state: %s", exc)

    def set_trader_performance(self, trader_id: str, perf: dict) -> None:
        """trader:{trader_id}:performance — used by GET /api/traders/{id}/performance"""
        try:
            self._r.hset(f"trader:{trader_id}:performance", mapping={
                "trader_id": trader_id,
                "monthly_pnl": float(perf.get("monthly_pnl", 0.0)),
                "profit_factor": float(perf.get("profit_factor", 0.0)),
                "total_trades": int(perf.get("total_trades", 0)),
                "win_rate": float(perf.get("win_rate", 0.0)),
                "avg_rr": float(perf.get("avg_rr", 0.0)),
                "max_drawdown": float(perf.get("max_drawdown", 0.0)),
            })
        except Exception as exc:
            logger.debug("StateManager.set_trader_performance: %s", exc)

    def set_strategy_allocation(self, trader_id: str, allocation: dict) -> None:
        """strategy_allocations:{trader_id} — read by backend state_reader.py"""
        try:
            self._r.set(
                f"strategy_allocations:{trader_id}",
                json.dumps({
                    "strategy_id": trader_id,
                    "is_active": bool(allocation.get("is_active", True)),
                    "allocated_capital": allocation.get("allocated_capital"),
                    "used_capital": allocation.get("used_capital"),
                    "current_risk": allocation.get("current_risk"),
                }),
            )
        except Exception as exc:
            logger.debug("StateManager.set_strategy_allocation: %s", exc)

    # ─── Positions ───────────────────────────────────────────────────────────

    def set_open_position(self, ticket: str, position: dict) -> None:
        """positions:open hash + positions:{ticket} key (state_reader scans positions:*:*)"""
        try:
            pos_json = json.dumps(position)
            self._r.hset("positions:open", ticket, pos_json)
            # state_reader scans "positions:*:*" — write individual keys too
            self._r.set(f"positions:{ticket}:data", pos_json)
        except Exception as exc:
            logger.debug("StateManager.set_open_position: %s", exc)

    def remove_open_position(self, ticket: str) -> None:
        try:
            self._r.hdel("positions:open", ticket)
            self._r.delete(f"positions:{ticket}:data")
        except Exception as exc:
            logger.debug("StateManager.remove_open_position: %s", exc)

    # ─── ML models ───────────────────────────────────────────────────────────

    def set_ml_model_status(self, model_id: str, info: dict) -> None:
        """ml:model:{model_id} — used by GET /api/ml/models"""
        try:
            self._r.hset(f"ml:model:{model_id}", mapping={
                "model_id": model_id,
                "name": str(info.get("name", model_id)),
                "status": str(info.get("status", "active")),
                "accuracy": float(info.get("accuracy", 0.0)),
                "last_trained": str(info.get("last_trained", self._now_iso())),
            })
        except Exception as exc:
            logger.debug("StateManager.set_ml_model_status: %s", exc)

    def set_rl_agent_state(self, info: dict) -> None:
        """ml:rl_agent — used by GET /api/ml/rl-agent"""
        try:
            self._r.hset("ml:rl_agent", mapping={
                "algorithm": "PPO",
                "state_dim": 42,
                "action_dim": 6,
                "episodes": int(info.get("episodes", 0)),
                "avg_reward": float(info.get("avg_reward", 0.0)),
                "last_updated": str(info.get("last_updated", self._now_iso())),
            })
        except Exception as exc:
            logger.debug("StateManager.set_rl_agent_state: %s", exc)

    # ─── Context / regime ────────────────────────────────────────────────────

    def set_context(self, symbol: str, regime: str, bias: str, confidence: float) -> None:
        """context:{symbol} — optional dashboard visibility"""
        try:
            self._r.set(
                f"context:{symbol}",
                json.dumps({"regime": regime, "bias": bias, "confidence": confidence}),
            )
        except Exception as exc:
            logger.debug("StateManager.set_context: %s", exc)
