# Multi-Bot ICT/Smart Money Trading System

Updated 2026-04-19.

A production-grade, fully containerised automated trading system implementing Inner Circle Trader (ICT) and Smart Money Concepts (SMC) across Forex and Gold. Five independent strategy bots run concurrently against a Capital.com account with a Hybrid ML + RL signal layer trained on Kaggle T4 × 2 GPUs.

---

## What It Does

- **5 trading bots** each running a distinct session-aware strategy (NY trend, FVG/BOS, London breakout, news momentum, Asian mean reversion)
- **Paper trading mode** with realistic slippage, commission, and execution delay simulation
- **PortfolioManager** — per-trade TP1/TP2 targets, SL→breakeven, trailing stop, correlation cap, streak-based size scaling
- **PPO RL agent** (stable-baselines3) that selects which strategy to activate per bar — runs on CPU (MLP policy is faster on CPU per SB3 guidance)
- **4 ML models** that hot-reload without container restarts (GRU-LSTM PyTorch, PyTorch RegimeClassifier, PyTorch QualityScorer EV regressor, PPO RL)
- **Tiered retraining** via `retrain_scheduler.py`: Quality + RL weekly (Sun 02:00 UTC), Regime + GRU monthly (1st Sun 03:00 UTC)
- **Two trade journals** — a clean CSV and a detailed JSONL — that feed back into the ML training pipeline
- **Event-driven architecture** over Redis pub/sub (zero shared state between bots)
- **Kaggle split-dataset** training: code cloned from GitHub, OHLCV data from a separate Kaggle dataset, weights pushed back to GitHub post-training

---

## Tech Stack

| Layer | Technology |
|---|---|
| Trading Engine | Python 3.12, asyncio |
| Backend API | FastAPI |
| Frontend | React 18, Vite, Redux Toolkit, MUI |
| Broker | Capital.com REST API (demo + live) |
| Data fallback | yfinance |
| Message bus | Redis pub/sub |
| Persistence | PostgreSQL 15, Redis 7 |
| ML — time series | PyTorch GPU (GRU-LSTM, 256 hidden, 2 layers, 3 heads) |
| ML — tabular | PyTorch GPU (RegimeClassifier MLP, QualityScorer EV regressor) |
| ML — RL | stable-baselines3 PPO (CPU) |
| Training hardware | Kaggle T4 × 2 GPUs (DataParallel) |
| Infrastructure | Docker Compose |

> LightGBM and XGBoost have been fully replaced by PyTorch models.

---

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Capital.com account (demo or live)
- Trained model weights in `trading-engine/weights/` (see [TRAINING_AND_BACKTEST.md](TRAINING_AND_BACKTEST.md))

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

### 3. Access

| Interface | URL |
|---|---|
| Frontend dashboard | http://localhost:3001 |
| Backend API | http://localhost:3000 |
| API docs (Swagger) | http://localhost:3000/docs |

---

## Environment Variables

```env
# Capital.com broker
CAPITAL_API_KEY=your_key
CAPITAL_IDENTIFIER=your_email_or_account_id
CAPITAL_PASSWORD=your_password
CAPITAL_ENV=demo                  # change to live when ready

# Trading mode
PAPER_TRADING=true
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

# EV and uncertainty gates (Guard 7)
MIN_EV_THRESHOLD=0.10
MAX_UNCERTAINTY=0.80

# Retrainer schedule
RETRAIN_QUALITY_DAY=sunday
RETRAIN_QUALITY_HOUR=2
RETRAIN_MONTHLY_HOUR=3
MIN_JOURNAL_QUALITY=200
MIN_JOURNAL_RL=500
```

> **`ML_ENABLED=true` is the default.** All model weights must be present in `trading-engine/weights/` before starting the engine. Models raise `ModelNotTrainedError` (no silent fallback) if weights are missing.

---

## Services

| Container | Port | Purpose |
|---|---|---|
| `trading_postgres` | 5432 | Trade journal, state |
| `trading_redis` | 6379 | Event bus, state cache |
| `trading_backend` | 3000 | FastAPI — REST + WebSocket |
| `trading_engine_main` | 8000 (internal) | 5 traders, ML inference |
| `trading_frontend` | 3001 | React dashboard (Nginx) |
| `trading_model_retrainer` | — | `retrain_scheduler.py` |

---

## Architecture

```
Capital.com REST API → DataFetcher.get_ohlcv(symbol, tf, bars=300)
        │
        ▼  MARKET_DATA (Redis pub/sub)
  SignalPipeline.process_bar()
        │
        ├── Step 1: ML inference (load-bearing order)
        │     RegimeClassifier  → TRENDING_UP/DOWN/RANGING/VOLATILE
        │     GRULSTMPredictor  → p_bull, p_bear, entry_depth, expected_variance
        │     QualityScorer     → ev (R-multiples), quality_score
        │     SentimentModel    → score [-1,1], label, backend
        │
        ├── Step 2: All 5 traders → BaseTrader.analyze_market()
        │     8 guards → _compute_signal() → EV gate → RL gate
        │
        ├── Step 3: Ensemble score
        │     score = (ict × 0.5 + ml × 0.5 + sentiment_bonus) × regime_mult
        │
        ├── Step 4: Confidence gate ≥ 0.55
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
        ExecutionEngine → PaperTradingService / BrokerConnector
                │
                ▼  TRADE_EXECUTED
        PortfolioManager.manage_open_positions()
          — partial close @ TP1, SL→breakeven, trailing stop @ TP2
                │
        TradeJournal
          ├── logs/trade_journal.csv
          └── logs/trade_journal_detailed.jsonl
                │
                ▼  feedback loop
        retrain_scheduler.py
          ├── Weekly  (Sun 02:00 UTC): Quality + RL  (≥200/500 journal entries)
          └── Monthly (1st Sun 03:00 UTC): Regime + GRU  (OHLCV-based)

Redis pub/sub → backend WebSocket → frontend Redux store
```

---

## Trading Strategies

| Bot | Strategy | Session (UTC) | Regime Requirement |
|---|---|---|---|
| T1 | NY EMA Trend Pullback | 13:00–17:00 | TRENDING_UP or DOWN required |
| T2 | Structure Break + FVG Continuation | London + NY | Any |
| T3 | London Breakout + Liquidity Sweep | 07:00–10:00 | TRENDING only |
| T4 | News Structural Breakout | Any | Any (sentiment ≥ 0.65 hard gate) |
| T5 | Asian Range Mean Reversion | 02:00–06:45 | RANGING required |

**Dead zone:** 12:00–13:00 UTC — all traders blocked.

### Risk Controls (BaseTrader 8 guards)

| Guard | Check |
|---|---|
| 0 | Symbol exclusion list per trader |
| 1 | Session window |
| 2 | Dead zone 12:00–13:00 UTC |
| 3 | Per-symbol cooldown |
| 4 | Session trade cap |
| 5 | Circuit breaker (2% daily loss / 8% drawdown) |
| 6 | Spread gate (Forex > 3 pips, Gold > 50 pips) |
| 7 | EV gate: `ev ≥ 0.10` AND `expected_variance ≤ 0.80` AND `p_dir ≥ threshold` |
| 8 | RL gate: PPO approves trader + sets EV selectivity tier |

---

## ML / AI Layer

| Model | Algorithm | Role |
|---|---|---|
| RegimeClassifier | PyTorch MLP 59→128→64→4, BatchNorm, GELU, residual | TRENDING_UP/DOWN/RANGING/VOLATILE |
| GRULSTMPredictor | GRU(256, 2L) + LayerNorm → 3 heads | p_bull, p_bear, magnitude, uncertainty |
| QualityScorer | PyTorch MLP 17→64→32→1, Huber loss | Expected value in R-multiples |
| SentimentModel | FinBERT + VADER fallback | News/macro directional bias |
| RLAgent | PPO (stable-baselines3), CPU, 42-dim state, 16 actions | Strategy selector + EV threshold |

**EV label tiers** (QualityScorer training):

| Exit | EV label |
|---|---|
| `tp2` | `+rr_ratio` (best — model aims here) |
| `tp1` | `+rr × 0.75` |
| `be_or_trail` | `+rr × 0.4` |
| `sl_*` | `-1.0` |
| `time_exit` | skipped |

**Regime persistence filter:** short regime runs (<20 bars at 4H, <48 bars at 1H) are zero-weighted during training to prevent label noise from short-lived flips.

**Retraining schedule:**

| Cadence | Models | Min data |
|---|---|---|
| Weekly (Sun 02:00 UTC) | Quality + RL | 200 (Quality), 500 (RL) journal entries |
| Monthly (1st Sun 03:00 UTC) | Regime + GRU | OHLCV only |

All retrains warm-start from existing weights at 5× lower LR. GRU is excluded from the per-round reinforcement loop (catastrophic forgetting risk).

---

## Latest Backtest Results

Period: Jan 2021 – Aug 2024 · Capital: $10,000 · 3 reinforcement rounds

| Round | Trades | WR | PF | Sharpe | Max DD | Return |
|---|---|---|---|---|---|---|
| 1 | 2,571 | 45.4% | 2.08 | 3.77 | 2.8% | 1,549% |
| 2 | 2,610 | 44.8% | 2.04 | 3.70 | 3.6% | 1,458% |
| 3 | 2,586 | 45.3% | 2.05 | 3.71 | 2.8% | 1,491% |

All 4 regime classes active in backtest. Calibration OK across all symbols.

---

## Trade Journals

**`trade_journal.csv`** — 12-column clean log:
```
timestamp, trader, symbol, side, size, entry, stop_loss, take_profit, rr_ratio, confidence, pnl, commission
```

**`trade_journal_detailed.jsonl`** — full decision record including:
- `ev`, `quality_score`, `expected_variance`
- `ml_model_scores`: `p_bull`, `p_bear`, `regime`, `sentiment_score`, `sentiment_backend`
- `state_at_entry` (42-dim RL state vector)
- `rl_action`, `exit_reason`, `session`, `source`

Journal exit reasons and their meaning:
- `tp2` — full TP hit (best)
- `tp1` — TP1 hit, held for more
- `be_or_trail` — TP1 hit then trailed/stopped
- `sl_full` — stopped out at SL
- `time_exit` — timed out

---

## Kaggle Training Setup

The system uses a **split-dataset** approach on Kaggle:

1. **Code dataset** — GitHub repo clone at `/kaggle/input/datasets/tysonsiwela/multi-bot-system`
2. **Data dataset** — OHLCV + processed data at `/kaggle/input/datasets/tysonsiwela/trading-data`
3. **Fresh clone** at `/kaggle/working/remote/Multi-Bot` for GitHub push (via `step8_push_to_github.py`)

`env_config.py` resolves all paths automatically for both Kaggle and local environments.

Full pipeline: `python kaggle_train.py`

---

## Operations

```bash
# Start stack
docker compose -f docker-compose.dev.yml up -d --build

# Rebuild a single service
docker compose -f docker-compose.dev.yml build trading-engine
docker compose -f docker-compose.dev.yml up -d trading-engine

# Live trade journal
tail -f trading-engine/logs/trade_journal_detailed.jsonl | python -m json.tool

# Engine logs
docker compose -f docker-compose.dev.yml logs trading-engine --tail=50 -f

# Health check
docker exec trading_backend curl -s http://trading-engine:8000/health

# Manual retrain
docker exec trading_model_retrainer python /app/scripts/retrain_incremental.py --model quality
docker exec trading_model_retrainer python /app/scripts/retrain_incremental.py --model rl
docker exec trading_model_retrainer python /app/scripts/retrain_incremental.py --model regime
docker exec trading_model_retrainer python /app/scripts/retrain_incremental.py --model gru
```

---

## Project Structure

```
trading-system/
├── kaggle_train.py                   ← Kaggle entrypoint (full pipeline)
├── env_config.py                     ← Path resolver (local + Kaggle)
├── step8_push_to_github.py           ← Push weights + metrics to GitHub
├── .env                              ← Single source of truth for all services
├── docker-compose.dev.yml
├── docs/                             ← Documentation
│   ├── README.md                     ← This file
│   ├── models.md                     ← ML model details
│   ├── strategies.md                 ← Trading strategy details
│   ├── system_architecture.md        ← Full architecture reference
│   ├── system_assessment.md          ← Current status + bug history
│   └── TRAINING_AND_BACKTEST.md      ← Training runbook
├── instructions/                     ← Design docs
│   ├── project-description.md
│   ├── architecture.md
│   └── research-plan.md
├── pipeline/                         ← 9-step data + training pipeline
│   ├── step0_resample.py  → step7b_train.py
│   └── step6_backtest.py             ← 3-round reinforcement loop
├── backend/                          ← FastAPI (port 3000)
└── trading-engine/                   ← Core engine (port 8000 internal)
    ├── models/
    │   ├── regime_classifier.py      ← PyTorch MLP, dual-TF cascade
    │   ├── gru_lstm_predictor.py     ← GRU(256) + 3 heads
    │   ├── quality_scorer.py         ← EV regressor, tiered Huber loss
    │   ├── sentiment_model.py        ← FinBERT + VADER
    │   └── rl_agent.py               ← PPO, CPU, model.zip save/load
    ├── scripts/
    │   ├── retrain_incremental.py    ← Manual / scheduled retraining
    │   └── retrain_scheduler.py      ← Weekly/monthly trigger daemon
    ├── traders/                      ← trader_1 through trader_5
    ├── services/                     ← SignalPipeline, DataFetcher, TradeJournal
    ├── indicators/market_structure.py← Vectorized ICT (FVG, BOS, CHoCH, OB)
    ├── weights/                      ← Trained model weights
    └── logs/                         ← trade_journal.csv + trade_journal_detailed.jsonl
```

---

## Disclaimer

This software is for educational and research purposes. Automated trading carries substantial risk of financial loss. Past backtest performance does not guarantee future results. Always test thoroughly in paper trading mode before enabling live execution.
