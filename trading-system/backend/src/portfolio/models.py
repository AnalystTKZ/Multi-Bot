"""Pydantic models for portfolio optimization."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class StrategyReturns(BaseModel):
    name: str
    returns: List[float] = Field(..., description="Per-period returns for the strategy")
    risk_limit: Optional[float] = Field(
        None, description="Max risk allocation for the strategy (0-1)"
    )
    kelly_multiplier: Optional[float] = Field(
        None, description="Fractional Kelly multiplier (0-1)"
    )
    target_volatility: Optional[float] = Field(
        None, description="Target annualized volatility for scaling"
    )


class PortfolioOptimizationRequest(BaseModel):
    strategies: List[StrategyReturns]
    risk_free_rate: float = 0.0
    periods_per_year: int = 252
    portfolio_risk_cap: float = 1.0
    correlation_penalty_factor: float = 1.0
    kelly_multiplier_default: float = 0.5
    target_volatility_default: float = 0.15
    rebalance_interval_seconds: int = 3600
    enable_rebalance: bool = True


class StrategyMetrics(BaseModel):
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    expectancy: float
    volatility: float


class PortfolioOptimizationResult(BaseModel):
    strategy_allocations: Dict[str, float]
    risk_distribution: Dict[str, float]
    expected_return: float
    portfolio_drawdown: float
    correlation_matrix: Dict[str, Dict[str, float]]
    strategy_metrics: Dict[str, StrategyMetrics]
