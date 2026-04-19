"""
main.py — ProductionTradingEngine

New engine. Same external interface as old engine:
  - Health server on port 8000
  - Redis event publisher/subscriber
  - Honour ML_ENABLED and PAPER_TRADING flags
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Optional

# Ensure trading-engine root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import redis

from config.settings import settings
from services.event_bus import EventBus, EventType
from services.session_manager import SessionManager
from services.state_manager import StateManager
from services.news_service import NewsService
from services.data_fetcher import DataFetcher
from services.risk_engine import RiskEngine
from services.trade_journal import TradeJournal
from services.signal_pipeline import SignalPipeline
from services.order_executor import ExecutionEngine, ExecutionRequest
from services.paper_trading_service import PaperTradingService
from services.feature_engine import FeatureEngine
from monitors.portfolio_manager import PortfolioManager

# ML models — only imported when ML_ENABLED=true
if settings.ML_ENABLED:
    from models.gru_lstm_predictor import GRULSTMPredictor
    from models.regime_classifier import RegimeClassifier
    from models.quality_scorer import QualityScorer
    from models.sentiment_model import SentimentModel
    from models.rl_agent import RLAgent

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("engine.main")

_TIMEFRAMES = ["15M", "1H", "4H"]
_HTF_MAP = {"1H": "H1", "4H": "H4"}


def _model_ready(model) -> bool:
    if hasattr(model, "_loaded") or hasattr(model, "_model"):
        return bool(getattr(model, "_loaded", False) and getattr(model, "_model", None) is not None)
    return bool(getattr(model, "is_trained", False))


def _ensure_mpl_config_dir() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return
    mpl_dir = os.path.join("/tmp", f"matplotlib-{os.getuid()}")
    os.makedirs(mpl_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = mpl_dir


class ProductionTradingEngine:
    """
    New engine. 5 traders, health server on port 8000.
    ML/RL active only when ML_ENABLED=true.
    """

    def __init__(self):
        self._running = False
        self._init_redis()
        self._init_state_manager()
        self._init_event_bus()
        self._init_feature_engine()
        self._init_ml_models()
        self._init_session_manager()
        self._init_news_service()
        self._init_signal_pipeline()
        self._init_risk_engine()
        self._init_trade_journal()
        self._init_execution_engine()
        self._init_portfolio_manager()
        self._init_health_server()
        self._ohlcv_cache: Dict[str, Dict[str, object]] = {}
        # Track open positions for portfolio manager lifecycle calls
        self._open_positions: List[dict] = []
        self._daily_pnl: float = 0.0
        self._trades_today: int = 0

        self._state_mgr.set_strategy_allocation("ml_trader", {
            "is_active": True,
            "allocated_capital": settings.ACCOUNT_BALANCE,
        })

    def _init_redis(self) -> None:
        self._redis = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD or None,
            db=settings.REDIS_DB,
            decode_responses=True,
        )
        try:
            self._redis.ping()
            logger.info("Redis connected at %s:%s", settings.REDIS_HOST, settings.REDIS_PORT)
        except Exception as exc:
            logger.error("Redis connection failed: %s", exc)

    def _init_state_manager(self) -> None:
        self._state_mgr = StateManager(self._redis)

    def _init_event_bus(self) -> None:
        self._bus = EventBus(self._redis)

    def _init_feature_engine(self) -> None:
        self._feature_engine = FeatureEngine()

    def _init_ml_models(self) -> None:
        if not settings.ML_ENABLED:
            logger.info("ML_ENABLED=false — all ML models skipped")
            self._ml_models: dict = {}
            return

        logger.info("ML_ENABLED=true — loading models")
        _ensure_mpl_config_dir()
        model_factories = [
            ("regime", RegimeClassifier),
            ("quality", QualityScorer),
            ("gru_lstm", GRULSTMPredictor),
            ("sentiment", SentimentModel),
            ("rl", RLAgent),
        ]
        self._ml_models = {}
        for model_id, factory in model_factories:
            try:
                model = factory()
            except Exception as exc:
                logger.warning("Skipping %s: init failed: %s", model_id, exc)
                continue
            if _model_ready(model):
                self._ml_models[model_id] = model
                logger.info("Loaded %s (%s)", model_id, model.__class__.__name__)
            else:
                logger.warning("Skipping %s: weights missing or load failed", model_id)
        # Publish model status to Redis
        for model_id in ["regime", "quality", "gru_lstm", "sentiment", "rl"]:
            model = self._ml_models.get(model_id)
            self._state_mgr.set_ml_model_status(model_id, {
                "name": model.__class__.__name__ if model is not None else "",
                "status": "active" if model is not None else "untrained",
                "accuracy": 0.0,
            })
        if "rl" in self._ml_models:
            self._state_mgr.set_rl_agent_state({"episodes": 0, "avg_reward": 0.0})

    def _init_session_manager(self) -> None:
        self._session_mgr = SessionManager()

    def _init_news_service(self) -> None:
        self._news = NewsService(redis_client=self._redis, news_api_key=settings.NEWS_API_KEY)
        self._news.start()

    def _init_signal_pipeline(self) -> None:
        self._pipeline = SignalPipeline(
            ml_models=self._ml_models,
            feature_engine=self._feature_engine,
            session_manager=self._session_mgr,
            news_service=self._news,
            settings=settings,
            event_bus=self._bus,
        )

    def _init_risk_engine(self) -> None:
        self._risk = RiskEngine(settings)

    def _init_portfolio_manager(self) -> None:
        self._portfolio_mgr = PortfolioManager(settings)
        logger.info("PortfolioManager initialised")

    def _init_trade_journal(self) -> None:
        rl = self._ml_models.get("rl") if settings.ML_ENABLED else None
        self._journal = TradeJournal(rl_agent=rl)
        # Inject journal into feature engine for rolling stats
        self._feature_engine._journal = self._journal

    def _init_execution_engine(self) -> None:
        broker = None
        paper_svc = PaperTradingService(self._bus, self._state_mgr)

        if not settings.PAPER_TRADING:
            from services.broker_connector import CapitalComSession, BrokerConnector
            session = CapitalComSession(
                api_key=settings.CAPITAL_API_KEY,
                identifier=settings.CAPITAL_IDENTIFIER,
                password=settings.CAPITAL_PASSWORD,
                env=settings.CAPITAL_ENV,
            )
            if session.authenticate():
                broker = BrokerConnector(session, self._bus, self._state_mgr)
                logger.info("Live trading enabled — Capital.com connected")
            else:
                logger.error("Capital.com auth failed — falling back to paper trading")
                settings.PAPER_TRADING = True

        self._executor = ExecutionEngine(
            event_bus=self._bus,
            paper_trading_service=paper_svc,
            broker_connector=broker,
            paper_trading=settings.PAPER_TRADING,
        )
        mode = "paper" if settings.PAPER_TRADING else "live"
        self._state_mgr.set_engine_mode(mode)
        logger.info("Engine mode: %s", mode)

    def _init_health_server(self) -> None:
        engine_ref = self

        class HealthHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/health", "/"):
                    payload = {
                        "status": "ok",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "mode": "paper" if settings.PAPER_TRADING else "live",
                    }
                    body = json.dumps(payload).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", len(body))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *args):
                pass  # suppress access log noise

        self._health_server = HTTPServer(("0.0.0.0", settings.HEALTH_PORT), HealthHandler)
        t = threading.Thread(target=self._health_server.serve_forever, daemon=True)
        t.start()
        logger.info("Health server on port %s", settings.HEALTH_PORT)

    # ─── Market data handler ─────────────────────────────────────────────────

    def on_market_data(self, event: dict) -> None:
        """Handler for MARKET_DATA Redis event (Contract 1)."""
        symbol = event.get("symbol", "")
        timeframe = event.get("timeframe", "15M")
        if not symbol:
            return

        # Update OHLCV store from incoming bar
        # In production the DataFetcher fetches full history; here we append
        self._pipeline.update_ohlcv(symbol, timeframe, self._get_ohlcv(symbol, timeframe))

        df = self._pipeline.get_ohlcv(symbol, timeframe)
        if df is None or len(df) < 20:
            df = self._get_ohlcv(symbol, timeframe)
        if df is None or len(df) < 10:
            return

        df_htf = {
            "5M":  self._get_ohlcv(symbol, "5M"),
            "15M": df,                              # already loaded above
            "1H":  self._get_ohlcv(symbol, "1H"),
            "4H":  self._get_ohlcv(symbol, "4H"),
            "1D":  self._get_ohlcv(symbol, "1D"),
        }

        # Run pipeline (async → sync bridge)
        try:
            loop = asyncio.new_event_loop()
            signals = loop.run_until_complete(
                self._pipeline.process_bar(symbol, df, df_htf)
            )
            loop.close()
        except Exception as exc:
            logger.error("Pipeline error for %s: %s", symbol, exc)
            return

        # Portfolio manager: lifecycle actions on open positions each bar
        current_prices = {symbol: float(df["close"].iloc[-1])} if df is not None and len(df) > 0 else {}
        pm_actions = self._portfolio_mgr.manage_open_positions(self._open_positions, current_prices)
        for action in pm_actions:
            logger.info("PortfolioManager action: %s", action)

        # For each approved signal: risk check → PM enrich → execute → journal
        portfolio = self._get_portfolio_state()
        for sig in signals:
            try:
                allowed, reason = self._risk.check_pre_trade(sig, portfolio)
                if not allowed:
                    logger.debug("Signal blocked: %s", reason)
                    continue

                atr = float(sig.get("signal_metadata", {}).get("atr", 0.0))
                enriched = self._portfolio_mgr.enrich_signal(sig, portfolio, atr=atr)
                if enriched is None:
                    logger.debug("Signal rejected by PortfolioManager: %s %s", sig.get("trader_id"), sig.get("symbol"))
                    continue

                req = ExecutionRequest(
                    symbol=enriched["symbol"],
                    side=enriched["side"],
                    size=float(enriched.get("size", 0.01)),
                    entry=float(enriched["entry"]),
                    stop_loss=float(enriched["stop_loss"]),
                    take_profit=float(enriched["take_profit"]),
                    trader_id=enriched["trader_id"],
                    confidence=float(enriched.get("confidence", 0.6)),
                    rr_ratio=float(enriched.get("rr_ratio", 1.5)),
                    correlation_id=enriched.get("correlation_id", ""),
                    signal_metadata=enriched.get("signal_metadata", {}),
                    state_at_entry=enriched.get("state_at_entry", [0.0] * 42),
                )
                trade = self._executor.execute_trade(req, portfolio)
                if trade:
                    pnl = float(trade.get("pnl", 0.0))
                    self._daily_pnl += pnl
                    self._trades_today += 1
                    self._portfolio_mgr.record_outcome("ml_trader", pnl)
                    self._open_positions.append({**trade, "tp1": enriched.get("tp1"), "size_full": enriched.get("size_full")})
                    self._journal.log_trade({
                        **trade,
                        "trader": trade.get("trader_id"),
                        "entry_reason": str(enriched.get("signal_metadata", {}).get("strategy", "")),
                    })
            except Exception as exc:
                logger.error("Execution error for %s: %s", sig.get("symbol"), exc)

        # Update Redis state
        self._state_mgr.heartbeat()

    def _get_ohlcv(self, symbol: str, timeframe: str):
        """Fetch OHLCV with caching (re-fetch every bar)."""
        try:
            if not hasattr(self, "_data_fetcher"):
                session = None
                if not settings.PAPER_TRADING:
                    from services.broker_connector import CapitalComSession
                    session = CapitalComSession(
                        api_key=settings.CAPITAL_API_KEY,
                        identifier=settings.CAPITAL_IDENTIFIER,
                        password=settings.CAPITAL_PASSWORD,
                        env=settings.CAPITAL_ENV,
                    )
                self._data_fetcher = DataFetcher(capital_session=session, settings=settings)

            return self._data_fetcher.get_ohlcv(symbol, timeframe, bars=300)
        except Exception as exc:
            logger.debug("DataFetcher error %s/%s: %s", symbol, timeframe, exc)
            return None

    def _get_portfolio_state(self) -> dict:
        open_symbols = list({p.get("symbol") for p in self._open_positions if p.get("symbol")})
        return {
            "daily_loss_pct": max(0, -self._daily_pnl) / (settings.ACCOUNT_BALANCE + 1e-9),
            "drawdown_pct": 0.0,
            "open_positions": len(self._open_positions),
            "open_symbols": open_symbols,
            "open_positions_detail": self._open_positions,
            "equity": settings.ACCOUNT_BALANCE + self._daily_pnl,
            "daily_pnl": self._daily_pnl,
            "trades_today": self._trades_today,
        }

    # ─── Main loop ───────────────────────────────────────────────────────────

    def run(self) -> None:
        """Main loop. Subscribes to MARKET_DATA. Runs heartbeat."""
        self._running = True
        self._state_mgr.set_engine_status("running")
        logger.info("Engine starting — ML=%s, PAPER=%s",
                    settings.ML_ENABLED, settings.PAPER_TRADING)

        self._bus.subscribe(EventType.MARKET_DATA, self.on_market_data)

        # Heartbeat thread
        def heartbeat_loop():
            while self._running:
                self._state_mgr.heartbeat()
                time.sleep(30)

        threading.Thread(target=heartbeat_loop, daemon=True).start()

        # Main event loop
        try:
            self._bus.start_listening()
        except KeyboardInterrupt:
            logger.info("Engine interrupted")
        except Exception as exc:
            logger.error("Engine error: %s", exc)
            self._state_mgr.set_engine_status("error")
        finally:
            self._running = False
            self._state_mgr.set_engine_status("stopped")
            logger.info("Engine stopped")

    def stop(self) -> None:
        self._running = False
        self._bus.stop()
        self._health_server.shutdown()


if __name__ == "__main__":
    engine = ProductionTradingEngine()
    engine.run()
