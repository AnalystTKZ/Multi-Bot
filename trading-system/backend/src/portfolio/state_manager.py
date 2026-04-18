"""In-memory state manager for portfolio optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional


@dataclass
class StateManager:
    portfolio_risk_cap: float = 1.0
    rebalance_interval_seconds: int = 3600
    last_rebalance: Optional[datetime] = None
    last_allocations: Dict[str, float] = field(default_factory=dict)

    def should_rebalance(self, now: Optional[datetime] = None, interval_seconds: Optional[int] = None) -> bool:
        if now is None:
            now = datetime.now(timezone.utc)
        interval = interval_seconds or self.rebalance_interval_seconds
        if self.last_rebalance is None:
            return True
        return (now - self.last_rebalance).total_seconds() >= interval

    def update_allocations(self, allocations: Dict[str, float], now: Optional[datetime] = None) -> None:
        if now is None:
            now = datetime.now(timezone.utc)
        self.last_rebalance = now
        self.last_allocations = allocations


GLOBAL_STATE = StateManager()
