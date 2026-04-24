# ML-Native Forex/Gold Trading System

Updated 2026-04-24.

A production-grade, fully containerised automated trading system for Forex and Gold.
Signal generation is driven entirely by ML — ICT/SMC concepts are encoded as numeric
features fed to a GRU-LSTM model. There are no rule-based traders.


---

## What It Does

- **Single unified ML signal generator** — one `ml_trader` across all 11 symbols
- **Dual-cascade RegimeClassifier** — 4H macro bias + 1H intraday structure, both injected into every GRU timestep
- **GRU-LSTM** — 30-bar × 74-feature sequences → direction, magnitude, uncertainty
- **QualityScorer** — EV regressor in R-multiples, runs post-signal with actual rr_ratio
- **PPO RL agent** — selectivity tier selection (currently requires ≥200 journal trades to converge)
- **PortfolioManager** — TP1/TP2 targets, SL→breakeven, trailing stop, correlation cap, streak scaling
- **Tiered retraining** — Quality + RL weekly, Regime + GRU monthly
- **Two trade journals** — clean CSV + detailed JSONL feeding the ML training pipeline
- **Kaggle split-dataset training** on T4 × 2 GPUs

---

## Tech Stack

| Layer | Technology |
|---|---|
| Trading Engine | Python 3.12, asyncio |
| Backend API | FastAPI |
| Frontend | React 18, Vite, Redux Toolkit, MUI |
| Broker | Capital.com REST API (demo + live) |
| Message bus | Redis pub/sub |
| Persistence | PostgreSQL 15, Redis 7 |
| ML — sequences | PyTorch GPU (GRU(64,2L)→LSTM(128,2L)→3 heads, 74 features, temperature scaling) |
| ML — regime | PyTorch GPU (dual MLP cascade: HTF 4H 3-class + LTF 1H 4-class) |
| ML — EV | PyTorch GPU (MLP 17→64→32→1, class-weighted Huber loss) |
| ML — RL | stable-baselines3 PPO (CPU) |
| Training hardware | Kaggle T4 × 2 GPUs (DataParallel) |
| Infrastructure | Docker Compose |

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
NEWS_API_KEY=                     # enriches sentiment features
TELEGRAM_BOT_TOKEN=               # trade alerts
TELEGRAM_CHAT_ID=

# ML gates
MIN_EV_THRESHOLD=0.10
MAX_UNCERTAINTY=2.0                   # GRU expected_variance gate; default 2.0

# Retrainer schedule
RETRAIN_QUALITY_DAY=sunday
RETRAIN_QUALITY_HOUR=2
RETRAIN_MONTHLY_HOUR=3
MIN_JOURNAL_QUALITY=200
MIN_JOURNAL_RL=500
```

> **`ML_ENABLED=true` is the default.** All model weights must exist in `trading-engine/weights/` before starting. Models raise `ModelNotTrainedError` (no silent fallback) if weights are missing.

---

## Services

| Container | Port | Purpose |
|---|---|---|
| `trading_postgres` | 5432 | Trade journal, state |
| `trading_redis` | 6379 | Event bus, state cache |
| `trading_backend` | 3000 | FastAPI — REST + WebSocket |
| `trading_engine_main` | 8000 (internal) | Trading engine (paper + live) |
| `trading_frontend` | 3001 | React dashboard (Nginx) |
| `trading_model_retrainer` | — | `retrain_scheduler.py` |

---

## Architecture (Backtest / Working Path)

```
processed_data/histdata/{SYMBOL}_{TF}.parquet
        │
        ▼
  _precompute_ml_cache(symbol) — GPU-batched, once per symbol
        ├── RegimeClassifier (4H, 34 features) → regime_4h + conf
        ├── RegimeClassifier (1H, 18 features) → regime_1h + conf
        ├── Build 74-feature sequence matrix (regime injected per timestep)
        └── GRU-LSTM (30×74, batched 1024) → p_bull, p_bear, expected_variance
        │
        ▼
  Bar loop (dict lookup + gate evaluation)
        ├── expected_variance > MAX_UNCERTAINTY (env default 2.0) → skip
        ├── max(p_bull, p_bear) < 0.58 → skip
        ├── Dead zone 12:00–13:00 UTC / cooldown / daily cap / drawdown → skip
        ├── ATR-based entry / SL / TP levels
        ├── PortfolioManager: size, TP1/TP2, correlation cap
        ├── QualityScorer (17 features, uses actual rr_ratio) → ev
        ├── ev < 0.10 → skip
        └── Simulate trade → TradeJournal
        │
        ▼
  Trade journal (CSV + JSONL)
        │
        ▼  feedback loop
  retrain_scheduler.py
    ├── Weekly  (Sun 02:00 UTC): Quality + RL  (≥200 / ≥500 journal entries)
    └── Monthly (1st Sun 03:00 UTC): Regime + GRU  (OHLCV-based)
```

---

## Signal Gates

| Gate | Value | Where applied |
|------|-------|--------------|
| GRU uncertainty | `expected_variance ≤ MAX_UNCERTAINTY` (default 2.0) | Before ICT level computation |
| GRU direction | `max(p_bull, p_bear) ≥ 0.58` | Before ICT level computation |
| EV threshold | `ev ≥ 0.10` | After PM enrichment (needs rr_ratio) |
| Dead zone | 12:00–13:00 UTC | Bar loop |
| Cooldown | 10 bars | Bar loop |
| Daily loss cap | 2% | Bar loop |
| Portfolio drawdown halt | 8% | Bar loop |
| Correlation cap | max 2 per group | PortfolioManager |

---

## ML Models

| Model | Architecture | Features | Weights |
|---|---|---|---|
| RegimeClassifier (4H) | MLP 34→128→64→3 (BIAS_UP/DOWN/NEUTRAL) | `REGIME_4H_FEATURES` (34) | `weights/regime_htf.pkl` |
| RegimeClassifier (1H) | MLP 18→128→64→4 (TRENDING/RANGING/CONSOLIDATING/VOLATILE) | `REGIME_1H_FEATURES` (18) | `weights/regime_ltf.pkl` |
| GRU-LSTM | GRU(64,2L)→LSTM(128,2L)→3 heads + temperature.pt | `SEQUENCE_FEATURES` (74) | `weights/gru_lstm/model.pt` |
| QualityScorer | MLP 17→64→32→1, Huber | `QUALITY_FEATURES` (17) | `weights/quality_scorer.pkl` |
| SentimentModel | FinBERT + VADER fallback | news headlines | pre-trained |
| RLAgent | PPO (SB3), CPU, 16 actions | 43-dim state | `weights/rl_ppo/model.zip` |

**HTF regime classes (3):** BIAS_UP, BIAS_DOWN, BIAS_NEUTRAL
**LTF regime classes (4):** TRENDING, RANGING, CONSOLIDATING, VOLATILE

**EV label tiers (QualityScorer):**

| Exit | EV label |
|---|---|
| `tp2` | `+rr_ratio` |
| `tp1` | `+rr × 0.75` |
| `be_or_trail` | `+rr × 0.4` |
| `sl_*` | `-1.0` |
| `time_exit` | `+rr×0.2` (win), `-0.5` (loss), `0.0` (flat) |

All retrains warm-start from existing weights at 5× lower LR. GRU is excluded from per-round reinforcement loop (catastrophic forgetting risk).

---

## ICT Concepts as Features

ICT/SMC is encoded numerically in `SEQUENCE_FEATURES` — the GRU learns which patterns matter.

| Concept | Features in GRU input |
|---------|----------------------|
| BOS | Age (bars ago / 40), strength (move / ATR) |
| FVG | Distance to nearest open FVG / ATR, fill ratio |
| Liquidity sweep | Wick depth / ATR, body recovery ratio |
| EMA pullback | Band position, EMA21 slope, EMA stack |
| Asian range | Range width / ATR, price vs high/low |
| HTF regime (4H) | BIAS_UP/DOWN/NEUTRAL one-hot (3 dims) + conf — indices 26–29 |
| LTF regime (1H) | TRENDING/RANGING/CONSOLIDATING/VOLATILE one-hot (4 dims) + conf — indices 30–34 |

---

## Trade Journals

**`trade_journal.csv`** — clean 12-column log:
```
timestamp, trader, symbol, side, size, entry, stop_loss, take_profit, rr_ratio, confidence, pnl, commission
```

**`trade_journal_detailed.jsonl`** — full decision record:
`ev`, `expected_variance`, `exit_reason`, `ml_model_scores` (p_bull, p_bear, regime), `state_at_entry` (43-dim), `rl_action`

---

## Kaggle Training

Split-dataset approach:
1. **Code** — GitHub repo clone at `/kaggle/input/datasets/tysonsiwela/multi-bot-system`
2. **Data** — OHLCV + processed data at `/kaggle/input/datasets/tysonsiwela/trading-data`
3. **Push** — weights + metrics pushed back to GitHub via fresh clone

`env_config.py` resolves all paths automatically for both environments.

```bash
python kaggle_train.py   # full pipeline: step7a → step6 → step7b
```

---

## Operations

```bash
# Start stack
docker compose -f docker-compose.dev.yml up -d --build

# Rebuild a single service
docker compose -f docker-compose.dev.yml build trading-engine
docker compose -f docker-compose.dev.yml up -d trading-engine

# Engine logs
docker compose -f docker-compose.dev.yml logs trading-engine --tail=50 -f

# Live trade journal
tail -f trading-engine/logs/trade_journal_detailed.jsonl | python -m json.tool

# Manual retrain (from trading-engine/)
cd trading-system/trading-engine
python scripts/retrain_incremental.py --model regime
python scripts/retrain_incremental.py --model gru
python scripts/retrain_incremental.py --model quality
python scripts/retrain_incremental.py --model rl
python scripts/retrain_incremental.py --model all

# Offline pipeline (from trading-system/)
export PYTHONPATH="/home/tybobo/Desktop/Multi-Bot/trading-system:/home/tybobo/Desktop/Multi-Bot/trading-system/trading-engine"
python3 run_pipeline.py
```

---

## Project Structure

```
trading-system/
├── kaggle_train.py                   ← Kaggle entrypoint (full pipeline)
├── env_config.py                     ← Path resolver (local + Kaggle)
├── .env                              ← Single source of truth for all services
├── docker-compose.dev.yml
├── docs/
│   ├── README.md                     ← This file
│   ├── CLAUDE.md                     ← Claude reference (file layout, known issues)
│   ├── models.md                     ← ML model details
│   ├── strategies.md                 ← Signal generation reference
│   ├── system_architecture.md        ← Full pipeline + causal integrity
│   └── TRAINING_AND_BACKTEST.md      ← Training runbook
├── instructions/                     ← Design context docs
├── pipeline/                         ← 9-step data + training pipeline
├── backend/                          ← FastAPI (port 3000)
└── trading-engine/
    ├── main.py                       ← ProductionTradingEngine (functional)
    ├── models/
    │   ├── regime_classifier.py      ← Dual-cascade MLP (HTF 4H 3-class + LTF 1H 4-class)
    │   ├── gru_lstm_predictor.py     ← GRU(64,2L)→LSTM(128,2L)→3 heads + temperature.pt
    │   ├── quality_scorer.py         ← EV regressor, class-weighted Huber
    │   ├── sentiment_model.py        ← FinBERT + VADER
    │   ├── rl_agent.py               ← PPO, CPU, model.zip
    │   └── vector_store.py           ← FAISS index of 64-dim GRU embeddings
    ├── scripts/
    │   ├── run_backtest.py           ← Single ml_trader, GPU-batched
    │   ├── retrain_incremental.py    ← Manual / scheduled retraining
    │   └── retrain_scheduler.py      ← Weekly/monthly trigger daemon
    ├── traders/
    │   └── __init__.py               ← Empty — trader files deleted
    ├── services/
    │   ├── feature_engine.py         ← All feature vectors
    │   ├── signal_pipeline.py        ← _compute_ml_signal mirrors run_backtest exactly
    │   └── ...
    ├── indicators/market_structure.py ← Vectorized ICT (FVG, BOS, sweep, OB)
    ├── weights/                      ← Trained model weights
    └── logs/                         ← trade_journal.csv + trade_journal_detailed.jsonl
```

---

## Latest Backtest Results (2026-04-24)

| Window | WR | PF | Sharpe | MaxDD |
|--------|----|----|--------|-------|
| Val 2021–2023 | 53.8–54.3% | 2.97–3.14 | 5.11–5.29 | ≤2.5% |
| Blind test 2023–2025 | 50.9–51.4% | 2.45–2.52 | 4.83–4.96 | ≤2.9% |
| Post-retrain 3yr | 54.6% WR | — | — | 2.8% (3826% return R1) |

Data: 221,743 rows · 202 features · 11 symbols · 2016–2025

---

## Known Issues

| Issue | File | Impact |
|-------|------|--------|
| HTF BIAS_NEUTRAL recall ~30-38% | `models/regime_classifier.py` | NEUTRAL bars under-predicted; being actively improved |
| VectorStore broken | `models/vector_store.py` | numpy import bug + dim mismatch; being fixed |
| RL policy needs more journal data | `models/rl_agent.py` | Needs ≥200 trades for meaningful action diversity |

---

## Disclaimer

This software is for educational and research purposes. Automated trading carries substantial risk of financial loss. Past backtest performance does not guarantee future results. Always test thoroughly in paper trading mode before enabling live execution.
