# Multi-Bot ICT/Smart Money Trading System

A production-grade automated trading system implementing Inner Circle Trader (ICT) and Smart Money Concepts (SMC) across forex, commodities, and crypto. Four independent strategy bots run concurrently, each with its own capital allocation and risk controls.

Full documentation: `trading-system/instructions/` and `Instructions/CLAUDE.md`

---

## What It Is

- **4 trading bots**: MACD+ICT trend, SMC confluence, breakout fade, news fade
- **Capital.com broker integration**: REST API, demo + live support
- **Paper trading mode**: realistic slippage, commission, execution delay simulation
- **Optional ML/AI layer**: 6 models including an online RL agent that continuously learns from live trades
- **Two trade journals**: clean CSV and detailed JSONL, both feed back into model retraining
- **Event-driven architecture**: Redis pub/sub, zero shared state between bots
- **Full-stack dashboard**: React 18 + MUI frontend, FastAPI backend, WebSocket real-time updates

---

## Quick Start

```bash
cd trading-system

# 1. Configure
cp .env.example .env
# Fill in CAPITAL_API_KEY, CAPITAL_IDENTIFIER, CAPITAL_PASSWORD

# 2. Start (slim dev stack — recommended, ~1.5GB RAM)
docker compose -f docker-compose.dev.yml up -d

# 3. Check health
docker compose -f docker-compose.dev.yml ps
```

| Interface | URL |
|---|---|
| Frontend | http://localhost:3001 |
| API | http://localhost:3000 |
| API Docs | http://localhost:3000/docs |
| RabbitMQ | http://localhost:15672 |

Login: username `admin` or email `admin@admin.com` (set in `.env`)

> Grafana (port 3002) and Prometheus (port 9090) are only available in the full stack: `docker compose -f docker-compose.yml up -d` (requires ≥8GB RAM)

---

## Trading Strategies

| Bot | Strategy | Timeframe |
|---|---|---|
| Trader 1 | MACD + Price Action + ICT bonus confidence | 4H |
| Trader 2 | SMC/ICT confluence engine (scores 10+10 conditions, fires on ≥2) | 1H |
| Trader 3 | ICT Silver Bullet / Judas Swing FADE-ONLY (CHoCH + kill zones) | 15m |
| Trader 4 | News overreaction fade (USD + Gold) | 5m |

All bots use dynamic R:R (`confidence × 5.0`, min 1.0 to fire) and have hard session trade caps and cooldown periods.

---

## ML/AI Layer

Six models run as an ensemble when `ML_ENABLED=true`:

| Model | Role |
|---|---|
| PricePredictor | Short-term direction (LightGBM) |
| PatternRecognizer | Candlestick/structure patterns (LightGBM) |
| MLSignalFilter | Signal quality gate — trained on live paper trade outcomes |
| AnomalyDetector | Regime anomaly flag (IsolationForest) |
| SentimentAnalyzer | Macro/news sentiment (VADER) |
| RLAgent | Online Q-learning — trains on every closed trade |

Ensemble: `score = ict × 0.6 + ml × 0.4 + sentiment_bonus − anomaly_penalty`

Disabled by default (`ML_ENABLED=false`) — zero startup overhead.

---

## Backtesting

```bash
# Run backtest for traders 1, 2, 3
docker exec trading_engine_main python /app/scripts/run_backtest.py 1 2 3
```

Most recent results:
- Trader 2 (SMC): +43% aggregate, USDJPY +251%
- Trader 3 (T3 Silver Bullet): 28 trades, PF 1.269, +11.57%, DD 12.04% (6yr M1 backtest, 5 pairs)

---

## Documentation

| File | Contents |
|---|---|
| `Instructions/CLAUDE.md` | Full system reference — architecture, bugs, operations, API routes |
| `trading-system/instructions/architecture.md` | System architecture, event flow, services layer, docker stacks |
| `trading-system/instructions/project-description.md` | Bot roles, conflict management, trade journals, ML layer |
| `trading-system/instructions/frontend.md` | Frontend tech stack, pages, API endpoints, Redux slices |
| `trading-system/instructions/research-plan.md` | Strategy implementations, risk management, ML research |
| `trading-system/instructions/News-Strategy.md` | Trader 4 news fade strategy detail |
| `Instructions/bot-strategy-psychology-playbook.md` | Per-bot psychology and discipline rules |

---

## Project Structure

```
trading-system/
├── .env                       ← All credentials and settings
├── docker-compose.dev.yml     ← Slim stack (USE THIS — 6 services)
├── docker-compose.yml         ← Full stack (13 services, needs ≥8GB RAM)
├── instructions/              ← Design and specification documents
├── scripts/                   ← run_backtest.py, retrain_incremental.py
├── backtest_results/          ← JSON backtest outputs
├── backend/                   ← FastAPI (port 3000)
│   └── src/routes/            ← auth, traders, positions, analytics, monitors, system
├── frontend/                  ← React 18 + MUI + Redux Toolkit (port 3001)
│   └── src/pages/             ← 10 pages: Dashboard, Traders, Analytics, Backtest, ML, etc.
└── trading-engine/            ← Core engine
    ├── traders/               ← 4 strategy bots
    ├── models/                ← 6 ML models + weights/
    ├── services/              ← EventBus, PaperTrading, Journal, Execution (24 services)
    ├── indicators/            ← Vectorized ICT indicators
    ├── monitors/              ← 5 monitoring agents
    └── logs/                  ← Trade journals (CSV + JSONL)
```

---

## Disclaimer

This software is for educational and research purposes. Automated trading carries substantial risk of financial loss. Always run in paper trading mode and validate results before enabling live execution.
