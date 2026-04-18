"""
order_executor.py — ExecutionEngine with ExecutionRequest dataclass.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from services.event_bus import EventBus, EventType

logger = logging.getLogger(__name__)


@dataclass
class ExecutionRequest:
    """Contract: carries all fields needed for execution and journaling."""
    symbol: str
    side: str
    size: float
    entry: float
    stop_loss: float
    take_profit: float
    trader_id: str
    confidence: float = 0.60
    rr_ratio: float = 1.5
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signal_metadata: Dict[str, Any] = field(default_factory=dict)
    state_at_entry: list = field(default_factory=lambda: [0.0] * 42)


class ExecutionEngine:
    """Routes ExecutionRequests to paper or live execution."""

    def __init__(
        self,
        event_bus: EventBus,
        paper_trading_service=None,
        broker_connector=None,
        paper_trading: bool = True,
    ):
        self._bus = event_bus
        self._paper = paper_trading_service
        self._broker = broker_connector
        self._paper_mode = paper_trading

    def execute_trade(self, request: ExecutionRequest, portfolio_state: dict) -> Optional[dict]:
        """
        Convert ExecutionRequest → signal dict → route to paper or live.
        Publishes TRADE_REQUESTED event before execution.
        """
        signal = {
            "symbol": request.symbol,
            "side": request.side,
            "size": request.size,
            "entry": request.entry,
            "stop_loss": request.stop_loss,
            "take_profit": request.take_profit,
            "trader_id": request.trader_id,
            "confidence": request.confidence,
            "rr_ratio": request.rr_ratio,
            "correlation_id": request.correlation_id,
            "signal_metadata": request.signal_metadata,
            "state_at_entry": request.state_at_entry,
        }

        # Publish TRADE_REQUESTED
        self._bus.publish(EventType.TRADE_REQUESTED, signal)

        if self._paper_mode and self._paper is not None:
            return self._paper.execute(signal, portfolio_state)
        elif not self._paper_mode and self._broker is not None:
            return self._broker.execute(signal, portfolio_state)
        else:
            logger.error("ExecutionEngine: no execution service available (paper=%s)", self._paper_mode)
            return None
