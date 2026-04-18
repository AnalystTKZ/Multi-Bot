"""
Traders API routes
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
from pydantic import BaseModel

from services.event_bus import publish_event
from services import state_reader
from utils.event_log import fetch_events
from routes.auth import get_current_user

router = APIRouter(dependencies=[Depends(get_current_user)])


class TraderControl(BaseModel):
    action: str  # 'start', 'stop', 'enable', 'disable'


def _sharpe_ratio(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    stdev = variance ** 0.5
    return (mean / stdev) if stdev > 0 else 0.0


def _max_drawdown_from_pnls(pnls: List[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd


def _metrics_from_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    pnls: List[float] = []
    for event in events:
        payload = event.get("payload", {}) or {}
        pnl_val = payload.get("pnl")
        if pnl_val is None:
            pnl_val = payload.get("profit")
        try:
            pnls.append(float(pnl_val or 0))
        except (TypeError, ValueError):
            pnls.append(0.0)

    total_trades = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    win_rate = len(wins) / total_trades if total_trades else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    total_pnl = sum(pnls)

    return {
        "total_trades": total_trades,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "sharpe_ratio": _sharpe_ratio(pnls),
        "max_drawdown": _max_drawdown_from_pnls(pnls),
    }


def _normalize_trader(strategy_id: str, allocation: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    is_active = allocation.get("is_active", True)
    return {
        "trader_id": strategy_id,
        "name": strategy_id,
        "strategy": strategy_id,
        "status": "running" if is_active else "stopped",
        "is_enabled": is_active,
        "total_trades": metrics.get("total_trades", 0),
        "win_rate": metrics.get("win_rate", 0),
        "total_pnl": metrics.get("total_pnl", 0),
        "allocated_capital": allocation.get("allocated_capital"),
        "used_capital": allocation.get("used_capital"),
        "current_risk": allocation.get("current_risk"),
        "metrics": metrics,
    }


@router.get("/", response_model=Dict[str, Any])
async def get_traders():
    """Get all traders status"""
    allocations = await state_reader.get_strategy_allocations()
    trade_events = await fetch_events("trade_executed", limit=1000)
    traders: List[Dict[str, Any]] = []

    for allocation in allocations:
        strategy_id = allocation.get("strategy_id")
        if not strategy_id:
            continue
        events = [e for e in trade_events if e.get("payload", {}).get("strategy_id") == strategy_id]
        metrics = _metrics_from_events(events)
        traders.append(_normalize_trader(strategy_id, allocation, metrics))

    return {"traders": traders}


@router.get("/{trader_id}", response_model=Dict[str, Any])
async def get_trader(trader_id: str):
    """Get specific trader status"""
    allocations = await state_reader.get_strategy_allocations()
    allocation = next((item for item in allocations if item.get("strategy_id") == trader_id), None)
    if not allocation:
        raise HTTPException(status_code=404, detail="Trader not found")
    events = await fetch_events("trade_executed", limit=500)
    metrics = _metrics_from_events([e for e in events if e.get("payload", {}).get("strategy_id") == trader_id])
    return {"trader": _normalize_trader(trader_id, allocation, metrics)}


@router.post("/{trader_id}/control")
async def control_trader(
    trader_id: str,
    control: TraderControl,
):
    """Control trader (start/stop/enable/disable)"""
    action = control.action.lower()
    if action in {"start", "enable"}:
        await publish_event("start_trader", {"strategy_id": trader_id})
    elif action in {"stop", "disable"}:
        await publish_event("stop_trader", {"strategy_id": trader_id})
    else:
        raise HTTPException(status_code=400, detail="Invalid action")
    return {"message": f"Trader {trader_id} {action} successful"}


@router.patch("/{trader_id}/status")
async def update_trader_status(
    trader_id: str,
    control: TraderControl,
):
    """Pause/Resume trader"""
    return await control_trader(trader_id, control)


@router.get("/{trader_id}/performance")
async def get_trader_performance(trader_id: str):
    """Get trader performance metrics derived from strategy metrics."""
    allocations = await state_reader.get_strategy_allocations()
    allocation = next((item for item in allocations if item.get("strategy_id") == trader_id), None)
    if not allocation:
        raise HTTPException(status_code=404, detail="Trader not found")

    events = await fetch_events("trade_executed", limit=500)
    metrics = _metrics_from_events([e for e in events if e.get("payload", {}).get("strategy_id") == trader_id])
    return {
        "trader_id": trader_id,
        "daily_pnl": metrics.get("total_pnl", 0),
        "weekly_pnl": metrics.get("total_pnl", 0),
        "monthly_pnl": metrics.get("total_pnl", 0),
        "sharpe_ratio": metrics.get("sharpe_ratio", 0),
        "max_drawdown": metrics.get("max_drawdown", 0),
        "win_rate": metrics.get("win_rate", 0),
        "avg_win": metrics.get("avg_win", 0),
        "avg_loss": metrics.get("avg_loss", 0),
        "profit_factor": metrics.get("avg_win", 0) / metrics.get("avg_loss", 1) if metrics.get("avg_loss", 0) else 0,
        "trader": _normalize_trader(trader_id, allocation, metrics),
    }


@router.get("/{trader_id}/signals")
async def get_trader_signals(trader_id: str, limit: int = 10):
    """Get recent signals from event log (trade_executed as proxy)."""
    events = await fetch_events("trade_executed", limit=limit * 3)
    filtered = [e for e in events if e.get("payload", {}).get("strategy_id") == trader_id]
    return {"signals": filtered[:limit]}


@router.get("/{trader_id}/trades")
async def get_trader_trades(trader_id: str, limit: int = 50):
    """Get recent trades from event log."""
    events = await fetch_events("trade_executed", limit=limit * 3)
    filtered = [e for e in events if e.get("payload", {}).get("strategy_id") == trader_id]
    return {"trades": filtered[:limit]}


@router.get("/{trader_id}/history")
async def get_trader_history(trader_id: str, period: str = "30d"):
    """Return event history snapshot based on trade events."""
    events = await fetch_events("trade_executed", limit=200)
    filtered = [e for e in events if e.get("payload", {}).get("strategy_id") == trader_id]
    return {"period": period, "events": filtered}
