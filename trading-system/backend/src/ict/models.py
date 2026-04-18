"""
ICT/SMC data models for signal generation.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class Candle(BaseModel):
    ts: str = Field(..., description="ISO-8601 timestamp or exchange time label")
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


class MarketSnapshot(BaseModel):
    symbol: str
    timeframes: Dict[str, List[Candle]] = Field(
        ..., description="Mapping of timeframe (e.g. '1D','4H','15m') to candles"
    )


class IctSignal(BaseModel):
    symbol: str
    direction: str  # 'buy', 'sell', or 'none'
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confluence_score: int
    reasoning: List[str]
    feature_payload: Optional[Dict[str, float]] = None
