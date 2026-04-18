"""
Analytics API routes
"""

from collections import Counter, defaultdict
from datetime import datetime
import logging
import os
from math import sqrt
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from portfolio.engine import optimize_portfolio
from portfolio.models import PortfolioOptimizationRequest, PortfolioOptimizationResult
from portfolio.state_manager import GLOBAL_STATE
from routes.auth import get_current_user
from services import state_reader
from services.event_bus import publish_event
from utils.event_log import fetch_events

router = APIRouter(dependencies=[Depends(get_current_user)])
logger = logging.getLogger(__name__)

REGIME_SCORE_MAP = {
    "TRENDING_DOWN": -1,
    "RANGING": 0,
    "TRENDING_UP": 1,
    "VOLATILE": 2,
    "UNKNOWN": 0,
}


def _positions_from_portfolio(portfolio: Dict[str, Any]) -> List[Dict[str, Any]]:
    return portfolio.get("positions", []) if portfolio else []


def _to_float(value: Optional[Any]) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _equity_curve_from_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    points = []
    for event in events:
        payload = event.get("payload", {}) or {}
        equity = _to_float(payload.get("total_equity"))
        ts = event.get("timestamp")
        if equity and ts:
            points.append({"timestamp": ts, "equity": equity})
    points.sort(key=lambda item: item["timestamp"])
    return points


def _equity_curve_from_trades(trade_events: List[Dict[str, Any]], starting_equity: float) -> List[Dict[str, Any]]:
    points = []
    running_equity = starting_equity
    for event in sorted(trade_events, key=lambda item: item.get("timestamp", "")):
        pnl = _to_float((event.get("payload", {}) or {}).get("pnl"))
        running_equity += pnl
        points.append({"timestamp": event.get("timestamp"), "equity": running_equity})
    return points


def _compute_returns(points: List[Dict[str, Any]]) -> List[float]:
    returns = []
    for idx in range(1, len(points)):
        prev = points[idx - 1]["equity"]
        curr = points[idx]["equity"]
        if prev > 0:
            returns.append((curr - prev) / prev)
    return returns


def _max_drawdown(points: List[Dict[str, Any]]) -> float:
    if not points:
        return 0.0
    peak = points[0]["equity"]
    max_dd = 0.0
    for point in points:
        equity = point["equity"]
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak if peak > 0 else 0.0
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd


def _sharpe_ratio(returns: List[float], periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    stdev = sqrt(variance) if variance > 0 else 0.0
    if stdev == 0:
        return 0.0
    return (mean_r / stdev) * sqrt(periods_per_year)


def _max_drawdown_from_pnls(pnls: List[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        if equity > peak:
            peak = equity
        max_dd = max(max_dd, peak - equity)
    return max_dd


def _extract_trade_metrics(event: Dict[str, Any]) -> Dict[str, Any]:
    payload = event.get("payload", {}) or {}
    meta = payload.get("signal_metadata", {}) or {}
    return {
        "timestamp": event.get("timestamp"),
        "trader_id": payload.get("strategy_id") or payload.get("trader_id") or "system",
        "symbol": payload.get("symbol"),
        "side": payload.get("side") or payload.get("direction"),
        "pnl": _to_float(payload.get("pnl") or payload.get("profit")),
        "rr_ratio": _to_float(payload.get("rr_ratio") or meta.get("rr_ratio")),
        "confidence": _to_float(payload.get("confidence")),
    }


def _build_trade_summary(trade_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    records = [_extract_trade_metrics(event) for event in trade_events]
    pnls = [record["pnl"] for record in records]
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]
    rr_values = [record["rr_ratio"] for record in records if record["rr_ratio"] > 0]

    by_trader: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_trader[record["trader_id"]].append(record)

    trader_stats = {}
    for trader_id, trader_records in by_trader.items():
        trader_pnls = [record["pnl"] for record in trader_records]
        trader_wins = [pnl for pnl in trader_pnls if pnl > 0]
        trader_rrs = [record["rr_ratio"] for record in trader_records if record["rr_ratio"] > 0]
        trader_stats[trader_id] = {
            "total_trades": len(trader_records),
            "net_pnl": round(sum(trader_pnls), 4),
            "win_rate": len(trader_wins) / len(trader_records) if trader_records else 0.0,
            "avg_rr": round(sum(trader_rrs) / len(trader_rrs), 4) if trader_rrs else 0.0,
            "max_drawdown": round(_max_drawdown_from_pnls(trader_pnls), 4),
        }

    return {
        "records": records,
        "total_pnl": round(sum(pnls), 4),
        "total_trades": len(records),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": len(wins) / len(records) if records else 0.0,
        "avg_rr": round(sum(rr_values) / len(rr_values), 4) if rr_values else 0.0,
        "by_trader": trader_stats,
    }


def _build_position_breakdown(positions: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Any]] = {}
    for position in positions:
        label = str(position.get(key) or "Unknown")
        bucket = buckets.setdefault(label, {"label": label, "positions": 0, "volume": 0.0, "pnl": 0.0})
        bucket["positions"] += 1
        bucket["volume"] += _to_float(position.get("volume") or position.get("quantity"))
        bucket["pnl"] += _to_float(position.get("pnl") or position.get("profit"))
    return sorted(buckets.values(), key=lambda item: item["positions"], reverse=True)


def _extract_patterns(meta: Dict[str, Any]) -> List[str]:
    patterns: List[str] = []
    strategy = str(meta.get("strategy") or "").strip()
    if strategy:
        patterns.append(strategy.replace("_", " ").title())
    if meta.get("fvg_top") or meta.get("fvg_bottom"):
        patterns.append("Fair Value Gap")
    if meta.get("bos_present") or meta.get("bos_bull") or meta.get("bos_bear"):
        patterns.append("Break Of Structure")
    if meta.get("sweep_present") or meta.get("sweep_detected"):
        patterns.append("Liquidity Sweep")
    if _to_float(meta.get("fakeout_prob")) >= 0.55:
        patterns.append("Fakeout Risk")
    if meta.get("session"):
        patterns.append(f"{str(meta['session']).upper()} Session")
    return list(dict.fromkeys(patterns))


def _build_dashboard_payload(
    portfolio: Dict[str, Any],
    positions: List[Dict[str, Any]],
    trade_summary: Dict[str, Any],
    portfolio_points: List[Dict[str, Any]],
    signal_events: List[Dict[str, Any]],
    pair_contexts: List[Dict[str, Any]],
    selected_symbol: Optional[str],
    limit: int,
) -> Dict[str, Any]:
    context_symbols = [context.get("symbol") for context in pair_contexts if context.get("symbol")]
    all_symbols = sorted(
        {
            event.get("payload", {}).get("symbol")
            for event in signal_events
            if event.get("payload", {}).get("symbol")
        }
        | {record["symbol"] for record in trade_summary["records"] if record["symbol"]}
        | set(context_symbols)
    )
    active_symbol = selected_symbol or (all_symbols[0] if all_symbols else (context_symbols[0] if context_symbols else "XAUUSD"))

    filtered_signals = [
        event
        for event in signal_events
        if event.get("payload", {}).get("symbol") == active_symbol
    ]
    filtered_signals.sort(key=lambda item: item.get("timestamp", ""))
    filtered_signals = filtered_signals[-limit:]

    portfolio_curve = []
    peak = portfolio_points[0]["equity"] if portfolio_points else _to_float(portfolio.get("total_equity"))
    for point in portfolio_points[-limit:]:
        peak = max(peak, point["equity"])
        drawdown = ((peak - point["equity"]) / peak) if peak > 0 else 0.0
        portfolio_curve.append(
            {
                "timestamp": point["timestamp"],
                "equity": round(point["equity"], 4),
                "drawdown": round(drawdown, 4),
            }
        )

    pattern_counter: Counter[str] = Counter()
    regime_history = []
    prediction_history = []
    latest_prediction = None

    for event in filtered_signals:
        payload = event.get("payload", {}) or {}
        meta = payload.get("signal_metadata", {}) or {}
        regime = str(meta.get("regime") or payload.get("regime") or "UNKNOWN").upper()
        confidence = _to_float(payload.get("confidence"))
        quality_score = _to_float(meta.get("quality_score") or payload.get("quality_score"))
        p_bull = _to_float(meta.get("p_bull") or payload.get("p_bull"))
        p_bear = _to_float(meta.get("p_bear") or payload.get("p_bear"))
        entry_price = _to_float(payload.get("entry") or payload.get("entry_price") or payload.get("price"))
        take_profit = _to_float(payload.get("take_profit") or payload.get("tp"))
        predicted_move = take_profit - entry_price if entry_price and take_profit else 0.0
        patterns = _extract_patterns(meta)
        pattern_counter.update(patterns)

        regime_history.append(
            {
                "timestamp": event.get("timestamp"),
                "regime": regime,
                "regime_score": REGIME_SCORE_MAP.get(regime, 0),
                "quality_score": round(quality_score, 4),
                "confidence": round(confidence, 4),
            }
        )
        history_point = {
            "timestamp": event.get("timestamp"),
            "price_target": round(take_profit, 6) if take_profit else None,
            "entry_price": round(entry_price, 6) if entry_price else None,
            "predicted_move": round(predicted_move, 6),
            "predicted_move_pct": round((predicted_move / entry_price), 4) if entry_price else 0.0,
            "p_bull": round(p_bull, 4),
            "p_bear": round(p_bear, 4),
            "quality_score": round(quality_score, 4),
            "confidence": round(confidence, 4),
            "direction": payload.get("side") or payload.get("direction") or "buy",
            "regime": regime,
            "patterns": patterns,
        }
        prediction_history.append(history_point)
        latest_prediction = history_point

    if not latest_prediction:
        latest_prediction = {
            "timestamp": datetime.utcnow().isoformat(),
            "price_target": None,
            "entry_price": None,
            "predicted_move": 0.0,
            "predicted_move_pct": 0.0,
            "p_bull": 0.5,
            "p_bear": 0.5,
            "quality_score": 0.0,
            "confidence": 0.0,
            "direction": "buy",
            "regime": "UNKNOWN",
            "patterns": [],
        }

    current_pair_regime = next(
        (context for context in pair_contexts if context.get("symbol") == active_symbol),
        None,
    )
    if current_pair_regime and not regime_history:
        regime = str(current_pair_regime.get("regime") or "UNKNOWN").upper()
        regime_history = [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "regime": regime,
                "regime_score": REGIME_SCORE_MAP.get(regime, 0),
                "quality_score": 0.0,
                "confidence": _to_float(current_pair_regime.get("confidence")),
            }
        ]

    return {
        "symbol": active_symbol,
        "symbols": all_symbols,
        "portfolio_overview": {
            "portfolio_value": _to_float(portfolio.get("total_equity")),
            "cash_balance": _to_float(portfolio.get("cash_balance") or portfolio.get("cash")),
            "open_positions": len(positions),
            "total_pnl": trade_summary["total_pnl"],
            "total_trades": trade_summary["total_trades"],
        },
        "portfolio_curve": portfolio_curve,
        "pair_regimes": pair_contexts,
        "current_pair_regime": current_pair_regime,
        "regime_history": regime_history,
        "prediction_history": prediction_history,
        "latest_prediction": latest_prediction,
        "exposure_by_symbol": _build_position_breakdown(positions, "symbol"),
        "exposure_by_trader": _build_position_breakdown(positions, "trader_id"),
        "pattern_snapshot": [
            {"pattern": pattern, "count": count}
            for pattern, count in pattern_counter.most_common(6)
        ],
    }


@router.get("/performance")
async def get_performance_metrics(
    request: Request,
    period: str = Query("30d", min_length=2, max_length=12),
):
    """Get overall system performance derived from portfolio and events."""
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    try:
        await publish_event("get_metrics", {"scope": "performance", "period": period})
        portfolio = await state_reader.get_portfolio_state() or {}
        positions = await state_reader.get_positions()
        trade_events = await fetch_events("trade_executed", limit=1000)
        trade_summary = _build_trade_summary(trade_events)

        portfolio_events = await fetch_events("portfolio_updated", limit=500)
        equity_points = _equity_curve_from_events(portfolio_events)
        if not equity_points:
            starting_equity = _to_float(portfolio.get("total_equity")) or float(os.getenv("TOTAL_CAPITAL", "100000"))
            equity_points = _equity_curve_from_trades(trade_events, starting_equity)
        if not equity_points and portfolio.get("total_equity") is not None:
            equity_points = [
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "equity": _to_float(portfolio.get("total_equity")),
                }
            ]

        returns = _compute_returns(equity_points)
        max_drawdown = _max_drawdown(equity_points)
        sharpe = _sharpe_ratio(returns)
        total_return = 0.0
        if len(equity_points) >= 2 and equity_points[0]["equity"] > 0:
            total_return = (equity_points[-1]["equity"] - equity_points[0]["equity"]) / equity_points[0]["equity"]

        metrics = {
            "balance": _to_float(portfolio.get("total_equity") or portfolio.get("cash_balance") or 10000),
            "monthly_pnl": trade_summary["total_pnl"],
            "total_pnl": trade_summary["total_pnl"],
            "total_return": total_return,
            "total_trades": trade_summary["total_trades"],
            "winning_trades": trade_summary["winning_trades"],
            "losing_trades": trade_summary["losing_trades"],
            "win_rate": trade_summary["win_rate"],
            "avg_rr": trade_summary["avg_rr"],
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "current_drawdown": max_drawdown,
            "portfolio_value": _to_float(portfolio.get("total_equity")),
            "open_positions": len(_positions_from_portfolio(portfolio)) or len(positions),
        }
        return {"metrics": metrics, "by_trader": trade_summary["by_trader"]}
    except Exception as exc:
        logger.error("get_performance_metrics error: %s", exc, extra={"correlation_id": correlation_id})
        return JSONResponse(status_code=500, content={"error": str(exc), "correlation_id": correlation_id})


@router.get("/equity-curve")
async def get_equity_curve(
    request: Request,
    period: str = Query("30d", min_length=2, max_length=12),
    resolution: str = Query("1d", min_length=2, max_length=12),
):
    """Get equity curve data from portfolio snapshots if available."""
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    try:
        await publish_event("get_portfolio", {"scope": "equity_curve", "period": period, "resolution": resolution})
        portfolio = await state_reader.get_portfolio_state() or {}
        events = await fetch_events("portfolio_updated", limit=500)
        points = _equity_curve_from_events(events)
        if not points and portfolio:
            points = [{"timestamp": datetime.utcnow().isoformat(), "equity": _to_float(portfolio.get("total_equity"))}]
        return {
            "equity_curve": {
                "dates": [p["timestamp"] for p in points],
                "equity": [p["equity"] for p in points],
            }
        }
    except Exception as exc:
        logger.error("get_equity_curve error: %s", exc, extra={"correlation_id": correlation_id})
        return JSONResponse(status_code=500, content={"error": str(exc), "correlation_id": correlation_id})


@router.get("/dashboard")
async def get_dashboard_analytics(
    request: Request,
    symbol: Optional[str] = Query(default=None, min_length=3, max_length=20),
    limit: int = Query(default=60, ge=10, le=240),
):
    """Return dashboard-oriented analytics for portfolio, regime, and prediction views."""
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    try:
        await publish_event("get_metrics", {"scope": "dashboard", "symbol": symbol, "limit": limit})
        portfolio = await state_reader.get_portfolio_state() or {}
        positions = await state_reader.get_positions()
        trade_events = await fetch_events("trade_executed", limit=1000)
        trade_summary = _build_trade_summary(trade_events)
        portfolio_points = _equity_curve_from_events(await fetch_events("portfolio_updated", limit=500))
        if not portfolio_points and portfolio.get("total_equity") is not None:
            portfolio_points = [
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "equity": _to_float(portfolio.get("total_equity")),
                }
            ]
        signal_events = await fetch_events("signal_generated", limit=1000)
        if not signal_events:
            signal_events = trade_events
        return _build_dashboard_payload(
            portfolio=portfolio,
            positions=positions,
            trade_summary=trade_summary,
            portfolio_points=portfolio_points,
            signal_events=signal_events,
            pair_contexts=await state_reader.get_market_contexts(),
            selected_symbol=symbol.upper() if symbol else None,
            limit=limit,
        )
    except Exception as exc:
        logger.error("get_dashboard_analytics error: %s", exc, extra={"correlation_id": correlation_id})
        return JSONResponse(status_code=500, content={"error": str(exc), "correlation_id": correlation_id})


@router.get("/trades")
async def get_trades_history(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0, le=10000),
    trader_id: Optional[str] = Query(default=None, min_length=1, max_length=64),
):
    """Get trades history from the event log."""
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    try:
        events = await fetch_events("trade_executed", limit=min(limit + offset, 2000))
        trades = []
        for event in events:
            trade = _extract_trade_metrics(event)
            if trader_id and trade["trader_id"] != trader_id:
                continue
            payload = event.get("payload", {}) or {}
            trades.append(
                {
                    "id": event.get("event_id"),
                    "symbol": trade["symbol"],
                    "side": trade["side"],
                    "type": trade["side"],
                    "pnl": trade["pnl"],
                    "profit": trade["pnl"],
                    "timestamp": trade["timestamp"],
                    "opened_at": trade["timestamp"],
                    "closed_at": trade["timestamp"],
                    "trader_id": trade["trader_id"],
                    "trader": trade["trader_id"],
                    "rr_ratio": trade["rr_ratio"],
                    "confidence": trade["confidence"],
                    "price": _to_float(payload.get("average_price") or payload.get("price")),
                }
            )
        sliced = trades[offset : offset + limit]
        return {"trades": sliced, "pagination": {"limit": limit, "offset": offset}}
    except Exception as exc:
        logger.error("get_trades_history error: %s", exc, extra={"correlation_id": correlation_id})
        return JSONResponse(status_code=500, content={"error": str(exc), "correlation_id": correlation_id})


@router.get("/monthly-returns")
async def get_monthly_returns(request: Request):
    """Aggregate trade PnLs into monthly buckets."""
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    try:
        events = await fetch_events("trade_executed", limit=2000)
        buckets: dict = {}
        for event in events:
            ts = event.get("timestamp", "")
            pnl = _to_float(event.get("payload", {}).get("pnl"))
            month = ts[:7] if ts else None
            if month:
                buckets[month] = buckets.get(month, 0.0) + pnl
        data = [{"month": k, "return": round(v, 4)} for k, v in sorted(buckets.items())]
        return {"data": data}
    except Exception as exc:
        logger.error("get_monthly_returns error: %s", exc, extra={"correlation_id": correlation_id})
        return JSONResponse(status_code=500, content={"error": str(exc), "correlation_id": correlation_id})


@router.post("/portfolio-optimization", response_model=PortfolioOptimizationResult)
async def optimize_multi_strategy_portfolio(request: PortfolioOptimizationRequest):
    """Optimize multi-strategy allocations using performance, risk, and correlation."""
    return optimize_portfolio(request, GLOBAL_STATE)
