# Multi-Bot ICT/Smart Money Trading System

Updated 2026-04-12.

A production-grade, fully containerised automated trading system implementing Inner Circle Trader (ICT) and Smart Money Concepts (SMC) across Forex and Gold. Five independent strategy bots run concurrently against a Capital.com account with a Hybrid ML + RL signal layer.

---

## What It Does

- **5 trading bots** each running a distinct session-aware strategy (NY trend, FVG/BOS, London breakout, news momentum, Asian mean reversion)
- **Paper trading mode** with realistic slippage, commission, and execution delay simulation
- **PortfolioManager** — per-trade TP1/TP2 targets, SL→breakeven, trailing stop, correlation cap, streak-based size scaling
- **PPO RL agent** (stable-baselines3) that selects which strategy to activate per bar
- **5 ML models** that hot-reload weekly without container restarts (GRU-LSTM PyTorch, LightGBM Regime, XGBoost Quality, FinBERT Sentiment, PPO RL)
- **Weekly automatic retraining** via the `model-retrainer` service (Sunday 02:00 UTC)
- **Two trade journals** — a clean CSV and a detailed JSONL — that feed back into the ML training pipeline
- **Event-driven architecture** over Redis pub/sub (zero shared state between bots)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Trading Engine | Python 3.11, asyncio |
| Backend API | FastAPI |
| Frontend | React 18, Vite, Redux Toolkit, MUI |
| Broker | Capital.com REST API (demo + live) |
| Data fallback | yfinance |
| Message bus | Redis pub/sub |
| Persistence | PostgreSQL 15, Redis 7 |
| ML — time series | PyTorch CPU (GRU-LSTM) |
| ML — tabular | LightGBM (regime), XGBoost (quality) |
| ML — NLP | FinBERT `ProsusAI/finbert` primary, VADER fallback |
| ML — RL | stable-baselines3 PPO |
| Infrastructure | Docker Compose |

---

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Capital.com account (demo or live)
- Optional: NewsAPI key (improves Trader 4 news awareness)

### 1. Configure environment

```bash
cd trading-system
cp .env.example .env
# Fill in credentials — see Environment Variables below
```

### 2. Build and start

```bash
docker compose -f docker-compose.dev.yml up -d --build
```

This starts the full production stack: postgres, redis, backend, trading-engine, frontend, model-retrainer.

### 3. Access

| Interface | URL |
|---|---|
| Frontend dashboard | http://localhost:3001 |
| Backend API | http://localhost:3000 |
| API docs (Swagger) | http://localhost:3000/docs |

---

## Environment Variables

All services read from a single `.env` file in `trading-system/`.

```env
# Capital.com broker
CAPITAL_API_KEY=your_key
CAPITAL_IDENTIFIER=your_email_or_account_id
CAPITAL_PASSWORD=your_password
CAPITAL_ENV=demo                  # change to live when ready

# Trading mode
PAPER_TRADING=true                # set false only after live verification
ML_ENABLED=true                   # requires trained weights in trading-engine/weights/
TRADING_PAIRS=EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,XAUUSD

# Database
DB_USER=trading
DB_PASSWORD=changeme
DB_NAME=trading_db

# Redis
REDIS_PASSWORD=changeme

# Auth
JWT_SECRET=changeme
ADMIN_USERNAME=admin
ADMIN_PASSWORD=changeme

# Optional
NEWS_API_KEY=                     # improves Trader 4 event awareness
TELEGRAM_BOT_TOKEN=               # trade alerts
TELEGRAM_CHAT_ID=

# Retrainer schedule (defaults: sunday 02:00 UTC)
RETRAIN_DAY=sunday
RETRAIN_HOUR=2
```

> **ML_ENABLED=true is the default.** All 5 model weights must be present in `trading-engine/weights/` before enabling. Models raise `ModelNotTrainedError` if weights are missing — there is no silent fallback.

---

## Services

`docker-compose.dev.yml` runs 6 containers:

| Container | Image | Port | Purpose |
|---|---|---|---|
| `trading_postgres` | postgres:15-alpine | 5432 | Trade journal, state |
| `trading_redis` | redis:7-alpine | 6379 | Event bus, state cache |
| `trading_backend` | Custom build | 3000 | FastAPI — REST + WebSocket |
| `trading_engine_main` | Custom build | 8000 (internal) | 5 traders, ML inference |
| `trading_frontend` | Custom build | 3001 | React dashboard (Nginx) |
| `trading_model_retrainer` | Same as engine | — | Weekly model retraining |

---

## Architecture

```
Capital.com REST API → DataFetcher.get_ohlcv(symbol, tf, bars=300)
        │
        ▼  MARKET_DATA (Redis pub/sub)
  SignalPipeline.process_bar()
        │
        ├── Step 1: ML inference (if ML_ENABLED)
        │     GRULSTMPredictor  → p_bull, p_bear, entry_depth
        │     RegimeClassifier  → TRENDING_UP/DOWN/RANGING/VOLATILE
        │     QualityScorer     → P(signal hits TP)
        │     SentimentModel    → score [-1,1], label, backend
        │
        ├── Step 2: All 5 traders → BaseTrader.analyze_market()
        │     8 guards → _compute_signal() → ML quality gate → RL gate
        │
        ├── Step 3: Ensemble score filter (≥ 0.5)
        │     score = (ict × 0.5 + ml × 0.5 + sentiment_bonus) × regime_mult
        │
        ├── Step 4: Confidence gate (≥ 0.55)
        └── Step 5: SIGNAL_GENERATED → Redis pub/sub
                │
                ▼
        RiskEngine.check_pre_trade()
                │
                ▼
        PortfolioManager.enrich_signal()
          — TP1/TP2, R:R ≥ 2.0, correlation cap, vol scalar, streak scalar
                │
                ▼
        ExecutionEngine
          → PaperTradingService (slippage + commission simulation)
          → BrokerConnector (live Capital.com orders)
                │
                ▼  TRADE_EXECUTED
        PortfolioManager.manage_open_positions() (per bar)
          — partial close @ TP1, SL→breakeven, trailing stop @ TP2
                │
        TradeJournal
          ├── logs/trade_journal.csv           (clean 12-column)
          └── logs/trade_journal_detailed.jsonl (full: ML scores, RL state, sentiment backend)

Redis pub/sub → backend WebSocket → frontend Redux store
```

---

## Trading Strategies

See [strategies.md](strategies.md) for full details.

| Bot | Strategy | Session (UTC) | Instruments |
|---|---|---|---|
| T1 | NY EMA Trend Pullback | 13:00–17:00 | EURUSD, GBPUSD, USDJPY, XAUUSD |
| T2 | Structure Break + FVG Continuation | London + NY | EURUSD, GBPUSD, USDJPY, XAUUSD |
| T3 | London Breakout + Liquidity Sweep | 07:00–10:00 | EURUSD, GBPUSD, XAUUSD |
| T4 | News Structural Breakout | Any (news-driven) | All pairs |
| T5 | Asian Range Mean Reversion | 02:00–06:45 | USDJPY, EURJPY, AUDJPY, EURUSD, AUDUSD |

**Dead zone**: 12:00–13:00 UTC — all traders blocked every day.

### Risk Controls

All traders enforce via `BaseTrader` 8-guard chain:

| Guard | Check |
|---|---|
| 0 | Symbol exclusion list per trader |
| 1 | Session window (UTC start/end) |
| 2 | Dead zone 12:00–13:00 UTC |
| 3 | Per-symbol cooldown |
| 4 | Session trade cap |
| 5 | Circuit breaker — 2% daily loss or 8% drawdown halts all trading |
| 6 | Spread gate — Forex > 3 pips, Gold > 50 pips |
| 7 | ML quality gate — QualityScorer + directional probability threshold |
| 8 | RL gate — PPO action must approve this trader |

Capital isolation: 20% per trader, 1% risk per trade.

---

## ML / AI Layer

See [models.md](models.md) for full details.

| Model | Algorithm | Role |
|---|---|---|
| GRULSTMPredictor | GRU(20)→LSTM(128)→Linear(64)→Linear(3) — PyTorch CPU | Directional probability: p_bull, p_bear, entry_depth |
| RegimeClassifier | LightGBM 4-class + 3-bar hysteresis | Market regime gate |
| QualityScorer | XGBoost binary (journal outcomes) | P(signal hits TP) |
| SentimentModel | FinBERT `ProsusAI/finbert` + VADER fallback | News/macro bias; Gold inverts USD score |
| RLAgent | PPO (stable-baselines3), 42-dim state, 16 actions | Strategy selector + confidence threshold |

**Ensemble**: `(ict_score × 0.5 + ml_score × 0.5 + sentiment_bonus) × regime_mult`

Every journal entry records `sentiment_backend` (`finbert` / `vader` / `neutral`) and the full ensemble breakdown so every decision is explainable.

### Weekly Retraining

The `model-retrainer` container runs `retrain_scheduler.py` which fires `retrain_incremental.py --model all` every Sunday at 02:00 UTC. Schedule is configurable via `.env`:

```env
RETRAIN_DAY=sunday
RETRAIN_HOUR=2
```

To retrain manually:

```bash
docker exec trading_model_retrainer python /app/scripts/retrain_incremental.py --model gru
docker exec trading_model_retrainer python /app/scripts/retrain_incremental.py --model all
docker exec trading_model_retrainer python /app/scripts/retrain_incremental.py --dry-run

To keep RAM stable on large datasets, retraining runs one symbol at a time and supports symbol scoping via env vars:
```bash
docker exec -e RETRAIN_SYMBOLS_GRU="EURUSD,GBPUSD,USDJPY,XAUUSD" \
  -e RETRAIN_SYMBOLS_REGIME="EURUSD,USDJPY,XAUUSD" \
  trading_model_retrainer python /app/scripts/retrain_incremental.py --model gru
```
```

Hot-reload: `BaseModel.reload_if_updated()` checks weight file mtime every 5 min — no container restart needed after retraining.

---

## Trade Journals

Every executed trade writes to two logs in `trading-engine/logs/`:

**`trade_journal.csv`** — clean one-liner:
```
timestamp, trader, symbol, side, size, entry, stop_loss, take_profit, rr_ratio, confidence, pnl, commission
```

**`trade_journal_detailed.jsonl`** — full decision record:
```json
{
  "trader": "trader_2",
  "symbol": "XAUUSD",
  "side": "buy",
  "rr_ratio": 2.8,
  "confidence": 0.74,
  "pnl": 142.50,
  "exit_reason": "tp1",
  "session": "LONDON",
  "ml_model_scores": {
    "p_bull": 0.71,
    "p_bear": 0.18,
    "quality_score": 0.68,
    "ensemble_score": 0.72,
    "regime": "TRENDING_UP",
    "sentiment_score": 0.31,
    "sentiment_label": "positive",
    "sentiment_backend": "finbert",
    "sentiment_confidence": 0.84,
    "rl_action": 2
  },
  "state_at_entry": [0.12, -0.03, "...42 dims"],
  "signal_metadata": {"strategy": "fvg_continuation", "ict_score": 0.68}
}
```

Journals feed into:
- `retrain_incremental.py` — QualityScorer labels (TP hit = 1, SL hit = 0)
- `RLAgent.record_outcome()` — online PPO reward buffer

---

## Operations

```bash
# Start production stack
cd trading-system
docker compose -f docker-compose.dev.yml up -d --build

# Rebuild a single service after code changes
docker compose -f docker-compose.dev.yml build trading-engine
docker compose -f docker-compose.dev.yml up -d trading-engine

# View live trade journal
tail -f trading-engine/logs/trade_journal.csv

# Engine logs
docker compose -f docker-compose.dev.yml logs trading-engine --tail=50 -f

# Health check
docker exec trading_backend curl -s http://trading-engine:8000/health

# Run backtest (all 5 traders)
docker exec trading_engine_main python /app/scripts/run_backtest.py

# Manual retrain
docker exec trading_model_retrainer python /app/scripts/retrain_incremental.py --model gru
docker exec trading_model_retrainer python /app/scripts/retrain_incremental.py --model all

# Full reset (clears all volumes)
docker compose -f docker-compose.dev.yml down --volumes
docker compose -f docker-compose.dev.yml up -d --build
```

---

## Project Structure

```
trading-system/
├── .env                              ← Single source of truth for all services
├── docker-compose.dev.yml            ← Production stack (6 services)
├── docs/
│   ├── README.md                     ← This file
│   ├── CLAUDE.md                     ← Full Claude Code reference
│   ├── models.md                     ← ML model details
│   ├── strategies.md                 ← Trading strategy details
│   └── system_assessment.md
├── backend/                          ← FastAPI (port 3000)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── routes/                   ← auth, traders, positions, analytics, system
│       ├── services/                 ← redis_client, event_bus, state_reader
│       └── websocket/manager.py      ← Redis pub/sub → WebSocket broadcast
├── frontend/                         ← React/Nginx dashboard (port 3001)
│   ├── Dockerfile
│   └── src/
│       ├── pages/                    ← Dashboard, Traders, Analytics, ML, TradeHistory
│       ├── store/slices/             ← auth, traders, positions, analytics
│       └── services/                 ← api, traderService, analyticsService
└── trading-engine/                   ← Core engine (port 8000 internal)
    ├── main.py                       ← ProductionTradingEngine + health server
    ├── Dockerfile
    ├── requirements.txt
    ├── config/settings.py            ← Pydantic BaseSettings
    ├── traders/                      ← trader_1 through trader_5
    ├── models/                       ← GRULSTMPredictor, RegimeClassifier, QualityScorer,
    │                                    SentimentModel, RLAgent
    ├── monitors/portfolio_manager.py ← TP1/TP2, trailing stop, correlation, streak scaling
    ├── services/                     ← SignalPipeline, EventBus, TradeJournal,
    │                                    RiskEngine, PortfolioManager, DataFetcher
    ├── indicators/market_structure.py← Vectorized ICT indicators (FVG, BOS, CHoCH, OB)
    ├── scripts/                      ← run_backtest, retrain_incremental, retrain_scheduler
    ├── weights/                      ← Trained model weights (volume-mounted)
    └── logs/                         ← trade_journal.csv, trade_journal_detailed.jsonl
```

---

## Disclaimer

This software is for educational and research purposes. Automated trading carries substantial risk of financial loss. Past backtest performance does not guarantee future results. Always test thoroughly in paper trading mode before enabling live execution.
