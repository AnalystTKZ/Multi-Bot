"""
Monitors API routes
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any

from ict.engine import generate_ict_signal
from ict.models import MarketSnapshot, IctSignal
from ml_filter.engine import filter_signal, train_filter_model
from ml_filter.models import (
    FilterRequest,
    FilterResponse,
    FilteredSignalRequest,
    TrainRequest,
    TrainResult,
)
from utils.event_log import fetch_events
from routes.auth import get_current_user

router = APIRouter(dependencies=[Depends(get_current_user)])


@router.get("/")
async def get_monitors_status():
    """Get monitors status from recent health check events."""
    events = await fetch_events("health_check", limit=100)
    status_map: Dict[str, Dict[str, Any]] = {}
    for event in events:
        source = event.get("source") or "monitor"
        if source not in status_map:
            status_map[source] = {
                "monitor_id": source,
                "name": source,
                "status": "ok",
                "last_check": event.get("timestamp"),
            }
    return list(status_map.values())


@router.get("/alerts")
async def get_recent_alerts(limit: int = 50):
    """Get recent alerts from event log."""
    events = await fetch_events("system_alert", limit=limit)
    alerts = []
    for event in events:
        payload = event.get("payload", {})
        alerts.append({
            "id": event.get("event_id"),
            "type": payload.get("alert_type", "system"),
            "level": "warning",
            "message": payload.get("message"),
            "timestamp": event.get("timestamp"),
        })
    return alerts


@router.post("/ict-signal", response_model=IctSignal)
async def get_ict_signal(snapshot: MarketSnapshot):
    """Generate ICT/SMC signal from multi-timeframe candles."""
    return generate_ict_signal(snapshot)


@router.post("/ict-signal/filter", response_model=FilterResponse)
async def filter_ict_signal(request: FilterRequest):
    """Apply AI filter to an existing ICT signal."""
    return filter_signal(request)


@router.post("/ict-signal/filtered", response_model=FilterResponse)
async def get_filtered_ict_signal(request: FilteredSignalRequest):
    """Generate ICT signal then filter it through the AI model."""
    signal = generate_ict_signal(request.snapshot)
    filter_request = FilterRequest(
        original_signal=signal,
        market_features=request.market_features,
        trade_features=request.trade_features,
        threshold=request.threshold,
    )
    return filter_signal(filter_request)


@router.post("/ict-signal/train", response_model=TrainResult)
async def train_ict_filter(request: TrainRequest):
    """Train or update the AI filter with labeled trades."""
    return train_filter_model(request)
