"""Portfolio optimization engine for multi-strategy allocation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import sqrt
from typing import Dict, List

from .models import (
    PortfolioOptimizationRequest,
    PortfolioOptimizationResult,
    StrategyMetrics,
)
from .state_manager import StateManager


@dataclass
class StrategyStats:
    name: str
    returns: List[float]
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    expectancy: float
    volatility: float


def optimize_portfolio(
    request: PortfolioOptimizationRequest,
    state: StateManager,
) -> PortfolioOptimizationResult:
    if not request.strategies:
        return PortfolioOptimizationResult(
            strategy_allocations={},
            risk_distribution={},
            expected_return=0.0,
            portfolio_drawdown=0.0,
            correlation_matrix={},
            strategy_metrics={},
        )

    stats = [_compute_metrics(s, request.risk_free_rate, request.periods_per_year) for s in request.strategies]
    corr = _correlation_matrix([s.returns for s in request.strategies], [s.name for s in request.strategies])

    base_weights = _performance_weights(stats, corr, request.correlation_penalty_factor)

    if request.enable_rebalance and state.should_rebalance(interval_seconds=request.rebalance_interval_seconds):
        base_weights = _apply_rebalance_penalties(base_weights, stats)

    size_adjustments = _size_adjustments(request, stats)
    adjusted_weights = {k: base_weights.get(k, 0.0) * size_adjustments.get(k, 1.0) for k in base_weights}

    final_weights = _normalize_weights(adjusted_weights)
    final_weights = _apply_risk_limits(final_weights, request)

    risk_distribution = {k: v * request.portfolio_risk_cap for k, v in final_weights.items()}

    expected_return = _portfolio_expected_return(final_weights, stats)
    portfolio_drawdown = _portfolio_drawdown(final_weights, stats)

    state.update_allocations(final_weights, now=datetime.now(timezone.utc))

    metrics_map = {
        s.name: StrategyMetrics(
            sharpe_ratio=s.sharpe,
            sortino_ratio=s.sortino,
            max_drawdown=s.max_drawdown,
            win_rate=s.win_rate,
            expectancy=s.expectancy,
            volatility=s.volatility,
        )
        for s in stats
    }

    return PortfolioOptimizationResult(
        strategy_allocations=final_weights,
        risk_distribution=risk_distribution,
        expected_return=expected_return,
        portfolio_drawdown=portfolio_drawdown,
        correlation_matrix=corr,
        strategy_metrics=metrics_map,
    )


def _compute_metrics(strategy, risk_free_rate: float, periods_per_year: int) -> StrategyStats:
    returns = strategy.returns
    name = strategy.name
    if not returns:
        return StrategyStats(name, [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    mean_r = sum(returns) / len(returns)
    variance = _variance(returns)
    stdev = sqrt(variance) if variance > 0 else 0.0
    downside = [min(0.0, r) for r in returns]
    downside_var = _variance(downside)
    downside_dev = sqrt(downside_var) if downside_var > 0 else 0.0

    rf_per_period = risk_free_rate / max(periods_per_year, 1)
    sharpe = 0.0
    if stdev > 0:
        sharpe = (mean_r - rf_per_period) / stdev * sqrt(periods_per_year)
    sortino = 0.0
    if downside_dev > 0:
        sortino = (mean_r - rf_per_period) / downside_dev * sqrt(periods_per_year)

    max_dd = _max_drawdown(returns)

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    win_rate = len(wins) / len(returns) if returns else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    volatility = stdev * sqrt(periods_per_year)

    return StrategyStats(
        name=name,
        returns=returns,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        win_rate=win_rate,
        expectancy=expectancy,
        volatility=volatility,
    )


def _variance(series: List[float]) -> float:
    if len(series) < 2:
        return 0.0
    mean = sum(series) / len(series)
    return sum((x - mean) ** 2 for x in series) / (len(series) - 1)


def _max_drawdown(returns: List[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= 1.0 + r
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak if peak > 0 else 0.0
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd


def _correlation_matrix(series: List[List[float]], names: List[str]) -> Dict[str, Dict[str, float]]:
    matrix: Dict[str, Dict[str, float]] = {name: {} for name in names}
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i == j:
                matrix[name_i][name_j] = 1.0
                continue
            corr = _correlation(series[i], series[j])
            matrix[name_i][name_j] = corr
    return matrix


def _correlation(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a = a[-n:]
    b = b[-n:]
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n)) / (n - 1)
    var_a = _variance(a)
    var_b = _variance(b)
    if var_a <= 0 or var_b <= 0:
        return 0.0
    return max(-1.0, min(1.0, cov / sqrt(var_a * var_b)))


def _performance_weights(
    stats: List[StrategyStats],
    corr: Dict[str, Dict[str, float]],
    penalty_factor: float,
) -> Dict[str, float]:
    raw: Dict[str, float] = {}
    for s in stats:
        correlations = [abs(corr[s.name][other]) for other in corr[s.name] if other != s.name]
        avg_corr = sum(correlations) / len(correlations) if correlations else 0.0
        penalty = 1.0 + penalty_factor * avg_corr
        score = max(0.0, s.sharpe * s.expectancy)
        raw[s.name] = score / penalty if penalty > 0 else 0.0

    return _normalize_weights(raw)


def _apply_rebalance_penalties(weights: Dict[str, float], stats: List[StrategyStats]) -> Dict[str, float]:
    penalized = weights.copy()
    for s in stats:
        if s.sharpe < 0 or s.expectancy <= 0:
            penalized[s.name] = penalized.get(s.name, 0.0) * 0.5
    return _normalize_weights(penalized)


def _size_adjustments(
    request: PortfolioOptimizationRequest,
    stats: List[StrategyStats],
) -> Dict[str, float]:
    adjustments: Dict[str, float] = {}
    for s, strat in zip(stats, request.strategies):
        variance = _variance(s.returns)
        kelly = 0.0
        if variance > 0:
            kelly = s.expectancy / variance
        kelly_multiplier = strat.kelly_multiplier if strat.kelly_multiplier is not None else request.kelly_multiplier_default
        fractional_kelly = max(0.0, min(1.0, kelly * kelly_multiplier))

        target_vol = strat.target_volatility if strat.target_volatility is not None else request.target_volatility_default
        vol_scaler = target_vol / s.volatility if s.volatility > 0 else 0.0
        adjustments[s.name] = fractional_kelly * vol_scaler

    return adjustments


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        if not weights:
            return {}
        equal = 1.0 / len(weights)
        return {k: equal for k in weights}
    return {k: v / total for k, v in weights.items()}


def _apply_risk_limits(
    weights: Dict[str, float],
    request: PortfolioOptimizationRequest,
) -> Dict[str, float]:
    caps = {s.name: s.risk_limit for s in request.strategies if s.risk_limit is not None}
    if not caps:
        return weights

    remaining = weights.copy()
    fixed: Dict[str, float] = {}
    for name, cap in caps.items():
        if name in remaining and remaining[name] > cap:
            fixed[name] = cap
            remaining[name] = 0.0

    leftover = 1.0 - sum(fixed.values())
    if leftover < 0:
        return _normalize_weights(fixed)

    eligible = {k: v for k, v in remaining.items() if v > 0}
    if not eligible:
        return _normalize_weights(fixed)

    scaled = _normalize_weights(eligible)
    for k in scaled:
        scaled[k] *= leftover

    merged = {**fixed, **scaled}
    return _normalize_weights(merged)


def _portfolio_expected_return(weights: Dict[str, float], stats: List[StrategyStats]) -> float:
    total = 0.0
    for s in stats:
        total += weights.get(s.name, 0.0) * (sum(s.returns) / len(s.returns) if s.returns else 0.0)
    return total


def _portfolio_drawdown(weights: Dict[str, float], stats: List[StrategyStats]) -> float:
    if not stats:
        return 0.0
    lengths = [len(s.returns) for s in stats if s.returns]
    if not lengths:
        return 0.0
    min_len = min(lengths)
    if min_len <= 1:
        return 0.0

    combined_returns = []
    for i in range(-min_len, 0):
        combined = 0.0
        for s in stats:
            combined += weights.get(s.name, 0.0) * s.returns[i]
        combined_returns.append(combined)

    return _max_drawdown(combined_returns)
