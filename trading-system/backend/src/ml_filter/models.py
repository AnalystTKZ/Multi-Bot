"""Pydantic models for ML-based ICT signal filtering."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from ict.models import IctSignal, MarketSnapshot


class MarketFeatures(BaseModel):
    volatility: Optional[float] = Field(None, description="Annualized or per-period volatility")
    spread: Optional[float] = Field(None, description="Current bid/ask spread")
    volume: Optional[float] = Field(None, description="Current volume")
    session_timing: Optional[float] = Field(None, description="1 if in killzone, else 0")


class TradeFeatures(BaseModel):
    rr_ratio: Optional[float] = Field(None, description="Reward-to-risk ratio")
    distance_to_liquidity: Optional[float] = Field(None, description="Distance to nearest liquidity target")


class LabeledTrade(BaseModel):
    signal: IctSignal
    market_features: MarketFeatures
    trade_features: Optional[TradeFeatures] = None
    label: int = Field(..., ge=0, le=1)


class TrainRequest(BaseModel):
    trades: List[LabeledTrade]
    retrain: bool = True


class TrainResult(BaseModel):
    trained: bool
    samples: int
    feature_importance: Dict[str, float]
    message: str


class FilterRequest(BaseModel):
    original_signal: IctSignal
    market_features: MarketFeatures
    trade_features: Optional[TradeFeatures] = None
    threshold: float = Field(0.7, ge=0.0, le=1.0)


class FilteredSignalRequest(BaseModel):
    snapshot: MarketSnapshot
    market_features: MarketFeatures
    trade_features: Optional[TradeFeatures] = None
    threshold: float = Field(0.7, ge=0.0, le=1.0)


class FilterResponse(BaseModel):
    original_signal: IctSignal
    probability_score: float
    decision: str
    feature_importance: Dict[str, float]
    used_fallback: bool
