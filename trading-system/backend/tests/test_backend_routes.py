from types import SimpleNamespace

import pytest
from fastapi import Response

from routes import analytics as analytics_routes
from routes import auth, positions as positions_routes
from services import state_reader


def _fake_request(correlation_id: str = "test-correlation-id"):
    return SimpleNamespace(state=SimpleNamespace(correlation_id=correlation_id))


@pytest.fixture(autouse=True)
def _disable_external_side_effects(monkeypatch):
    async def fake_publish_event(event_type, payload, source="backend", correlation_id=None):
        return None

    monkeypatch.setattr(analytics_routes, "publish_event", fake_publish_event)
    monkeypatch.setattr(positions_routes, "publish_event", fake_publish_event)


@pytest.mark.asyncio
async def test_positions_endpoint(monkeypatch):
    async def fake_positions():
        return [
            {
                "order_id": "order-1",
                "symbol": "EURUSD",
                "side": "buy",
                "quantity": "1.0",
                "entry_price": "1.1000",
                "current_price": "1.1000",
                "pnl": "0",
                "strategy_id": "strategy-1",
            }
        ]

    async def fake_portfolio():
        return {"total_equity": "10000"}

    monkeypatch.setattr(state_reader, "get_positions", fake_positions)
    monkeypatch.setattr(state_reader, "get_portfolio_state", fake_portfolio)

    payload = await positions_routes.get_positions(_fake_request())
    assert payload["positions"]
    assert payload["positions"][0]["symbol"] == "EURUSD"


@pytest.mark.asyncio
async def test_analytics_performance(monkeypatch):
    async def fake_positions():
        return []

    async def fake_portfolio():
        return {"total_equity": "10000"}

    async def fake_events(event_type=None, limit=200):
        if event_type == "trade_executed":
            return [
                {
                    "payload": {"pnl": 100, "rr_ratio": 1.5, "strategy_id": "trader_1"},
                    "timestamp": "2025-01-01T00:00:00Z",
                },
                {
                    "payload": {"pnl": -50, "rr_ratio": 1.0, "strategy_id": "trader_1"},
                    "timestamp": "2025-01-02T00:00:00Z",
                },
            ]
        if event_type == "portfolio_updated":
            return [
                {"payload": {"total_equity": 10000}, "timestamp": "2025-01-01T00:00:00Z"},
                {"payload": {"total_equity": 10050}, "timestamp": "2025-01-02T00:00:00Z"},
            ]
        return []

    monkeypatch.setattr(state_reader, "get_positions", fake_positions)
    monkeypatch.setattr(state_reader, "get_portfolio_state", fake_portfolio)
    monkeypatch.setattr(analytics_routes, "fetch_events", fake_events)

    payload = await analytics_routes.get_performance_metrics(_fake_request())
    assert payload["metrics"]["total_trades"] == 2
    assert payload["metrics"]["total_pnl"] == 50
    assert payload["by_trader"]["trader_1"]["total_trades"] == 2


@pytest.mark.asyncio
async def test_auth_login_profile_and_logout(monkeypatch):
    monkeypatch.setattr(auth, "JWT_SECRET", "test-secret")
    monkeypatch.setattr(auth, "ADMIN_USERNAME", "admin")
    monkeypatch.setattr(auth, "ADMIN_PASSWORD", "password")
    monkeypatch.setattr(auth, "ADMIN_EMAIL", "admin@example.com")

    login_response = Response()
    login_payload = auth.LoginRequest(username="admin", password="password")
    login_result = await auth.login(login_payload, login_response)

    assert login_result["authenticated"] is True
    assert login_result["user"]["username"] == "admin"
    assert "set-cookie" in login_response.headers
    assert f"{auth.AUTH_COOKIE_NAME}=" in login_response.headers["set-cookie"]

    profile_response = Response()
    profile_result = await auth.profile(profile_response, {"username": "admin"})
    assert profile_result["authenticated"] is True
    assert profile_result["user"]["email"] == "admin@example.com"

    logout_response = Response()
    logout_result = await auth.logout(logout_response)
    assert logout_result["message"] == "Logged out successfully"
    assert "set-cookie" in logout_response.headers
    assert f"{auth.AUTH_COOKIE_NAME}=" in logout_response.headers["set-cookie"]


@pytest.mark.asyncio
async def test_close_position_uses_request_body(monkeypatch):
    captured = {}

    async def fake_publish(event_type, payload, source="backend", correlation_id=None):
        captured["event_type"] = event_type
        captured["payload"] = payload
        return None

    monkeypatch.setattr(positions_routes, "publish_event", fake_publish)

    response = await positions_routes.close_position(
        "order-1",
        _fake_request(),
        positions_routes.ClosePositionRequest(reason="manual"),
    )

    assert response["ticket"] == "order-1"
    assert captured["event_type"] == "close_position"
    assert captured["payload"]["reason"] == "manual"


@pytest.mark.asyncio
async def test_dashboard_snapshot(monkeypatch):
    async def fake_positions():
        return [
            {
                "order_id": "order-1",
                "symbol": "EURUSD",
                "side": "buy",
                "quantity": "1.0",
                "entry_price": "1.1000",
                "current_price": "1.1100",
                "pnl": "100",
                "strategy_id": "trader_1",
                "timestamp": "2025-01-02T00:00:00Z",
            }
        ]

    async def fake_portfolio():
        return {"total_equity": "10100", "cash": "5000"}

    async def fake_events(event_type=None, limit=200):
        if event_type == "signal_generated":
            return [
                {
                    "event_id": "sig-1",
                    "timestamp": "2025-01-02T00:00:00Z",
                    "payload": {
                        "symbol": "EURUSD",
                        "trader_id": "trader_1",
                        "side": "buy",
                        "confidence": 0.82,
                        "entry_price": 1.1,
                        "take_profit": 1.12,
                        "signal_metadata": {
                            "strategy": "trader_1",
                            "regime": "TRENDING_UP",
                            "quality_score": 0.74,
                            "p_bull": 0.8,
                            "p_bear": 0.2,
                        },
                    },
                }
            ]
        if event_type == "trade_executed":
            return [
                {
                    "event_id": "evt-1",
                    "timestamp": "2025-01-02T00:00:00Z",
                    "payload": {
                        "pnl": 100,
                        "rr_ratio": 1.5,
                        "strategy_id": "trader_1",
                        "symbol": "EURUSD",
                        "side": "buy",
                        "signal_metadata": {"regime": "TRENDING_UP"},
                    },
                }
            ]
        if event_type == "portfolio_updated":
            return [
                {"payload": {"total_equity": 10000}, "timestamp": "2025-01-01T00:00:00Z"},
                {"payload": {"total_equity": 10100}, "timestamp": "2025-01-02T00:00:00Z"},
            ]
        return []

    async def fake_contexts():
        return [{"symbol": "EURUSD", "regime": "TRENDING_UP", "bias": "bullish", "confidence": 0.8}]

    monkeypatch.setattr(state_reader, "get_positions", fake_positions)
    monkeypatch.setattr(state_reader, "get_portfolio_state", fake_portfolio)
    monkeypatch.setattr(state_reader, "get_market_contexts", fake_contexts)
    monkeypatch.setattr(analytics_routes, "fetch_events", fake_events)

    payload = await analytics_routes.get_dashboard_analytics(_fake_request(), symbol=None, limit=60)

    assert payload["portfolio_overview"]["open_positions"] == 1
    assert payload["symbol"] == "EURUSD"
    assert payload["regime_history"][0]["regime"] == "TRENDING_UP"
    assert payload["prediction_history"]
