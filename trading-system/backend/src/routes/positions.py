"""
Positions API routes
"""

import logging
import uuid
from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any
from pydantic import BaseModel

from services.event_bus import publish_event
from services import state_reader
from routes.auth import get_current_user

router = APIRouter(dependencies=[Depends(get_current_user)])
logger = logging.getLogger(__name__)


class ClosePositionRequest(BaseModel):
    reason: str = "manual"


def _normalize_position(position: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": position.get("order_id") or position.get("id") or position.get("ticket"),
        "ticket": position.get("order_id") or position.get("ticket") or position.get("id"),
        "symbol": position.get("symbol"),
        "type": position.get("side") or position.get("type"),
        "volume": float(position.get("quantity") or position.get("volume") or 0),
        "price_open": float(position.get("entry_price") or position.get("price_open") or 0),
        "price_current": float(position.get("current_price") or position.get("price_current") or 0),
        "profit": float(position.get("pnl") or position.get("unrealized_pnl") or position.get("profit") or 0),
        "sl": position.get("sl"),
        "tp": position.get("tp"),
        "trader_id": position.get("strategy_id") or position.get("trader_id"),
    }


@router.get("/", response_model=Dict[str, Any])
async def get_positions(
    request: Request,
    status: str = "open",
    limit: int = 50,
):
    """Get positions from trading engine portfolio snapshot."""
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    try:
        await publish_event("get_positions", {"status": status, "limit": limit})
        positions = await state_reader.get_positions()
        normalized = [_normalize_position(pos) for pos in positions]
        if status == "open":
            filtered = normalized[:limit]
        elif status == "closed":
            filtered = []
        else:
            filtered = normalized[:limit]

        summary = {
            "total_positions": len(filtered),
            "total_profit": sum(pos.get("profit", 0) for pos in filtered),
        }
        return {"positions": filtered, "summary": summary}
    except Exception as exc:
        logger.error("get_positions error: %s", exc, extra={"correlation_id": correlation_id})
        return JSONResponse(status_code=500, content={"error": str(exc), "correlation_id": correlation_id})


@router.get("/summary")
async def get_positions_summary(request: Request):
    """Get positions summary based on current portfolio."""
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    try:
        await publish_event("get_positions", {"summary": True})
        positions_state = await state_reader.get_positions()
        positions = [_normalize_position(pos) for pos in positions_state]
        total_positions = len(positions)
        total_profit = sum(pos.get("profit", 0) for pos in positions)
        winning_positions = sum(1 for pos in positions if pos.get("profit", 0) > 0)
        return {
            "total_positions": total_positions,
            "total_profit": total_profit,
            "winning_positions": winning_positions,
            "losing_positions": total_positions - winning_positions,
            "win_rate": winning_positions / total_positions if total_positions > 0 else 0,
        }
    except Exception as exc:
        logger.error("get_positions_summary error: %s", exc, extra={"correlation_id": correlation_id})
        return JSONResponse(status_code=500, content={"error": str(exc), "correlation_id": correlation_id})


@router.get("/locked-assets")
async def get_locked_assets(request: Request):
    """Return locked assets based on active positions."""
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    try:
        await publish_event("get_positions", {"locked_assets": True})
        positions = await state_reader.get_positions()
        symbols = sorted({pos.get("symbol") for pos in positions if pos.get("symbol")})
        return {"locked_assets": symbols}
    except Exception as exc:
        logger.error("get_locked_assets error: %s", exc, extra={"correlation_id": correlation_id})
        return JSONResponse(status_code=500, content={"error": str(exc), "correlation_id": correlation_id})


@router.get("/metrics")
async def get_position_metrics(request: Request):
    """Return position metrics based on portfolio snapshot."""
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    try:
        await publish_event("get_metrics", {"scope": "positions"})
        positions_state = await state_reader.get_positions()
        positions = [_normalize_position(pos) for pos in positions_state]
        total_profit = sum(pos.get("profit", 0) for pos in positions)
        portfolio_state = await state_reader.get_portfolio_state() or {}
        return {
            "positions_count": len(positions),
            "total_profit": total_profit,
            "portfolio_value": portfolio_state.get("total_equity"),
        }
    except Exception as exc:
        logger.error("get_position_metrics error: %s", exc, extra={"correlation_id": correlation_id})
        return JSONResponse(status_code=500, content={"error": str(exc), "correlation_id": correlation_id})


@router.post("/{ticket}/close")
async def close_position(
    ticket: str,
    request: Request,
    payload: ClosePositionRequest | None = Body(default=None),
):
    """Publish a close position command to the event bus."""
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    try:
        reason = payload.reason if payload else "manual"
        await publish_event("close_position", {"ticket": ticket, "reason": reason})
        return {"message": "Close request submitted", "ticket": ticket}
    except Exception as exc:
        logger.error("close_position error: %s", exc, extra={"correlation_id": correlation_id})
        return JSONResponse(status_code=500, content={"error": str(exc), "correlation_id": correlation_id})
