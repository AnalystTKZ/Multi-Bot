"""
FastAPI backend for Multi-Bot Trading System
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from starlette.middleware.trustedhost import TrustedHostMiddleware

# Load environment variables
load_dotenv()

logger = logging.getLogger("backend")

# Import routes
from routes import analytics, auth, debug, monitors, positions, system, traders
from services.redis_client import close_redis_client
from services.trading_engine_client import get_trading_engine_client
from utils.observability import configure_json_logging, ensure_correlation_id, log_event
from websocket.manager import manager

configure_json_logging(logging.INFO)


def _csv_env_list(name: str, fallback: str) -> list[str]:
    raw_value = os.getenv(name, fallback)
    return [item.strip() for item in raw_value.split(",") if item.strip()]


ALLOWED_ORIGINS = _csv_env_list(
    "FRONTEND_ALLOWED_ORIGINS",
    os.getenv("FRONTEND_URL", "http://localhost:3001"),
)
TRUSTED_HOSTS = _csv_env_list("TRUSTED_HOSTS", "*")

if "*" in ALLOWED_ORIGINS and os.getenv("ENV", "development") == "production":
    logger.warning(
        "SECURITY: FRONTEND_ALLOWED_ORIGINS contains '*' in production — "
        "set FRONTEND_ALLOWED_ORIGINS to the actual frontend origin"
    )


def _origin_allowed(origin: str | None, request_host: str | None = None) -> bool:
    if not origin:
        return True
    if "*" in ALLOWED_ORIGINS:
        return True
    if origin in ALLOWED_ORIGINS:
        return True

    try:
        origin_host = urlparse(origin).netloc
    except Exception:
        origin_host = ""
    if request_host and origin_host and origin_host == request_host:
        return True
    return False


def _resolve_websocket_token(websocket: WebSocket) -> str | None:
    auth_header = websocket.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header.split(" ", 1)[1].strip()
    cookie_token = websocket.cookies.get(auth.AUTH_COOKIE_NAME)
    if cookie_token:
        return cookie_token
    query_token = websocket.query_params.get("token")
    if query_token:
        return query_token
    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.run_listener()
    yield
    await close_redis_client()

# Create FastAPI app
app = FastAPI(
    title="Multi-Bot Trading System API",
    description="API for managing automated trading bots with ICT/SM strategies",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Correlation-Id"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=TRUSTED_HOSTS + ["testserver"])

@app.middleware("http")
async def request_logger(request: Request, call_next):
    start = time.perf_counter()
    correlation_id = ensure_correlation_id(request.headers.get("x-correlation-id"))
    request.state.correlation_id = correlation_id
    response = await call_next(request)
    response.headers["x-correlation-id"] = correlation_id
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault(
        "Permissions-Policy",
        "accelerometer=(), autoplay=(), camera=(), geolocation=(), microphone=(), payment=(), usb=()",
    )
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    duration_ms = (time.perf_counter() - start) * 1000
    log_event(
        logger,
        "info",
        module="backend_main",
        event_type="HTTP_REQUEST",
        message="Incoming API request processed",
        correlation_id=correlation_id,
        data={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
        },
    )
    return response

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(traders.router, prefix="/api/traders", tags=["Traders"])
app.include_router(positions.router, prefix="/api/positions", tags=["Positions"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(monitors.router, prefix="/api/monitors", tags=["Monitors"])
app.include_router(debug.router, prefix="/debug", tags=["Debug"])
app.include_router(system.router, prefix="/api", tags=["System"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Multi-Bot Trading System API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/api/health")
async def api_health_check():
    """API health alias for same-origin frontend calls."""
    return {"status": "healthy"}

@app.get("/status")
async def system_status():
    """Get system status — delegates to /api/system/status for real state."""
    from services.redis_client import get_redis_client
    import os
    redis_ok = False
    try:
        redis_client = get_redis_client()
        await redis_client.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    client = get_trading_engine_client()
    engine_ok = False
    try:
        await client.get_system_status()
        engine_ok = True
    except Exception:
        engine_ok = False

    return {
        "status": "operational" if (redis_ok and engine_ok) else "degraded",
        "services": {
            "redis": "connected" if redis_ok else "unavailable",
            "trading_engine": "running" if engine_ok else "unavailable",
        },
    }

@app.get("/metrics")
async def metrics():
    """Proxy Prometheus metrics from trading engine."""
    client = get_trading_engine_client()
    try:
        payload = await client.get_metrics_text()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Trading engine unavailable: {exc}") from exc
    return Response(content=payload, media_type="text/plain; version=0.0.4")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time events."""
    request_host = websocket.headers.get("x-forwarded-host") or websocket.headers.get("host")
    if not _origin_allowed(websocket.headers.get("origin"), request_host=request_host):
        await websocket.close(code=1008)
        return

    token = _resolve_websocket_token(websocket)
    if not token:
        await websocket.close(code=1008)
        return

    try:
        auth.validate_access_token(token)
    except HTTPException:
        await websocket.close(code=1008)
        return

    if not await manager.connect(websocket, token=token):
        return
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 3000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True if os.getenv("ENV") == "development" else False
    )
