"""
paper_trading_service.py — Full paper trading simulation.

Simulates slippage (0.01%–0.5%), commission (0.1%), execution delay (0.1–2s).
Publishes TRADE_EXECUTED event (Contract 1).
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from services.event_bus import EventBus, EventType

logger = logging.getLogger(__name__)

_COMMISSION_PCT = 0.001     # 0.1%
_SLIPPAGE_MIN = 0.0001      # 0.01%
_SLIPPAGE_MAX = 0.005       # 0.5%


class PaperTradingService:
    """Simulates order execution with realistic slippage and commission."""

    def __init__(self, event_bus: EventBus, state_manager=None):
        self._bus = event_bus
        self._state = state_manager

    def execute(self, signal: dict, portfolio_state: dict) -> Optional[dict]:
        """
        Simulate order execution.
        Returns TRADE_EXECUTED payload (Contract 1).
        """
        # Simulate execution delay
        delay = random.uniform(0.1, 2.0)
        time.sleep(min(delay, 0.5))  # cap at 0.5s to avoid blocking

        entry = float(signal.get("entry", 0.0))
        stop_loss = float(signal.get("stop_loss", entry))
        take_profit = float(signal.get("take_profit", entry))
        side = str(signal.get("side", "buy"))
        symbol = str(signal.get("symbol", ""))
        trader_id = str(signal.get("trader_id", ""))
        confidence = float(signal.get("confidence", 0.60))
        size = float(signal.get("size", 0.01))
        correlation_id = str(signal.get("correlation_id", str(uuid.uuid4())))

        # Slippage
        slippage_pct = random.uniform(_SLIPPAGE_MIN, _SLIPPAGE_MAX)
        if side == "buy":
            fill_price = entry * (1 + slippage_pct)
        else:
            fill_price = entry * (1 - slippage_pct)

        # Commission
        commission = fill_price * size * _COMMISSION_PCT

        # R:R
        if side == "buy":
            rr_num = take_profit - fill_price
            rr_den = fill_price - stop_loss
        else:
            rr_num = fill_price - take_profit
            rr_den = stop_loss - fill_price
        rr_ratio = round(rr_num / (rr_den + 1e-9), 2)

        ticket = str(uuid.uuid4())
        meta = signal.get("signal_metadata", {}) or {}
        now_iso = datetime.now(timezone.utc).isoformat()

        trade_event = {
            "ticket": ticket,
            "trader_id": trader_id,
            "symbol": symbol,
            "side": side,
            "size": size,
            "entry": round(fill_price, 6),
            "stop_loss": round(stop_loss, 6),
            "take_profit": round(take_profit, 6),
            "rr_ratio": rr_ratio,
            "confidence": confidence,
            "strategy_id": trader_id,
            "timestamp": now_iso,
            "pnl": None,        # null on open
            "signal_metadata": meta,
            "commission": round(commission, 4),
            "correlation_id": correlation_id,
        }

        # Publish Contract 1 TRADE_EXECUTED event
        self._bus.publish(EventType.TRADE_EXECUTED, trade_event)

        # Update positions in Redis
        if self._state is not None:
            position_dict = {
                **trade_event,
                "status": "open",
                "opened_at": now_iso,
            }
            self._state.set_open_position(ticket, position_dict)

        logger.info(
            "PAPER_TRADE %s %s %s @ %.5f sl=%.5f tp=%.5f rr=%.2f conf=%.2f",
            trader_id, side, symbol, fill_price, stop_loss, take_profit, rr_ratio, confidence,
        )
        return trade_event
