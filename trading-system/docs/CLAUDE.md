# Multi-Bot Trading System — Claude Reference

Last updated: 2026-04-19. Reflects ML-native architecture (5 rule-based traders removed).

For full ML pipeline details see `docs/system_architecture.md`.
For training and backtest runbook see `docs/TRAINING_AND_BACKTEST.md`.

---

## Current System State

**The 5 ICT rule-based traders are gone.** `traders/__init__.py` contains only a comment.
Signal generation happens in a single unified ML path in `run_backtest._compute_backtest_signal`.

`main.py` and `signal_pipeline.py` still reference the old traders — they are **broken for live trading**.
The working execution path is the offline pipeline (`pipeline/` + `scripts/run_backtest.py`).

---

## Repository Layout

```
trading-system/
├── .env                              ← single source of truth
├── docker-compose.dev.yml
├── env_config.py                     ← path resolver (_ENV dict)
├── docs/
│   ├── CLAUDE.md                     ← this file
│   ├── system_architecture.md
│   ├── TRAINING_AND_BACKTEST.md
│   ├── strategies.md
│   └── models.md
├── pipeline/                         ← 9-step offline pipeline
│   ├── step0_resample.py
│   ├── step1_inventory.py
│   ├── step2_clean.py
│   ├── step3_align.py
│   ├── step4_features.py
│   ├── step5_split.py
│   ├── step6_backtest.py             ← runs run_backtest.py, builds journal
│   ├── step7a_train.py               ← trains Regime + GRU
│   ├── step7b_train.py               ← trains Quality + RL (needs journal)
│   └── step8_validate.py
├── run_pipeline.py                   ← orchestrates pipeline; skips completed steps
├── kaggle_train.py                   ← Kaggle entry point (step7a → step6 → step7b)
├── processed_data/
│   ├── histdata/                     ← {SYMBOL}_{5M|15M|1H|4H|1D}.parquet (step 0)
│   └── ...
├── ml_training/
│   ├── datasets/                     ← train/val/test.parquet + split_summary.json
│   └── metrics/
├── training_data/
│   ├── forex/                        ← *_m1_histdata.csv (2016–2026)
│   ├── indices/                      ← *_1d.csv (ASX200, DAX, DXY, VIX, etc.)
│   └── fundamental/                  ← treasury_10yr.csv, treasury_2yr.csv
├── backend/
│   └── src/
│       ├── main.py                   ← FastAPI app + CORS
│       └── routes/
│           ├── auth.py
│           ├── traders.py
│           ├── positions.py
│           ├── analytics.py
│           ├── monitors.py
│           └── system.py
└── trading-engine/
    ├── main.py                       ← BROKEN — imports deleted trader classes
    ├── config/settings.py            ← Pydantic BaseSettings
    ├── indicators/market_structure.py ← all vectorized; no .at[i] indexing
    ├── services/
    │   ├── feature_engine.py         ← all feature vectors (SEQUENCE_FEATURES=74,
    │   │                               REGIME_4H_FEATURES=31, REGIME_1H_FEATURES=15,
    │   │                               QUALITY_FEATURES=17, RL_STATE_DIM=43)
    │   ├── signal_pipeline.py        ← BROKEN — calls deleted trader.analyze_market()
    │   ├── data_fetcher.py
    │   ├── broker_connector.py
    │   ├── order_executor.py
    │   ├── risk_engine.py
    │   └── trade_journal.py
    ├── models/
    │   ├── base_model.py
    │   ├── regime_classifier.py      ← dual-cascade: 4H bias (31 feat) + 1H structure (15 feat)
    │   ├── gru_lstm_predictor.py     ← GRU(64)→LSTM(128)→3 heads; 74 SEQUENCE_FEATURES
    │   ├── quality_scorer.py         ← EV regressor; Huber loss; 17 QUALITY_FEATURES
    │   ├── sentiment_model.py        ← FinBERT primary; VADER fallback
    │   ├── rl_agent.py               ← PPO via SB3; 43-dim state; 16 actions
    │   └── weights/
    │       ├── gru_lstm/model.pt
    │       ├── regime_4h.pkl         ← 4H bias classifier
    │       ├── regime_1h.pkl         ← 1H structure classifier
    │       ├── quality_scorer.pkl
    │       └── rl_ppo/
    ├── traders/
    │   └── __init__.py               ← empty; all trader files deleted
    ├── monitors/
    │   └── portfolio_manager.py      ← sizing, TP1/trailing, correlation cap
    └── scripts/
        ├── run_backtest.py           ← single ml_trader; GPU-batched inference
        ├── retrain_incremental.py    ← --model gru|regime|quality|rl|all
        └── retrain_scheduler.py     ← fires Sunday 02:00 UTC
```

---

## ML Architecture

| Model | Role | Output | Weights |
|-------|------|--------|---------|
| RegimeClassifier (4H) | Bias layer — macro regime from 4H+1D data | 5-class + confidence | `weights/regime_4h.pkl` |
| RegimeClassifier (1H) | Structure layer — intraday regime from 1H+4H | 5-class + confidence | `weights/regime_1h.pkl` |
| GRU-LSTM | Direction + magnitude + uncertainty | `p_bull`, `p_bear`, `expected_move`, `expected_variance` | `weights/gru_lstm/model.pt` |
| QualityScorer | EV in R-multiples (runs post-signal with real rr_ratio) | `ev`, `quality_score` | `weights/quality_scorer.pkl` |
| SentimentModel | News headline scoring | `sentiment_score`, `sentiment_label` | pre-trained |
| RLAgent | Selectivity tier selection | action 0–15 | `weights/rl_ppo/` |

**Feature counts — fixed contract. Changing order or length breaks saved weights.**

| List | Length | Model |
|------|--------|-------|
| `SEQUENCE_FEATURES` | 74 | GRU-LSTM |
| `REGIME_4H_FEATURES` | 31 | RegimeClassifier (4H) |
| `REGIME_1H_FEATURES` | 15 | RegimeClassifier (1H) |
| `QUALITY_FEATURES` | 17 | QualityScorer |
| `RL_STATE_DIM` | 43 | RLAgent |

---

## Signal Generation (Working Path)

`scripts/run_backtest.py` — `_compute_backtest_signal()` with `trader_id="ml_trader"`:

```
Per 15M bar:
  1. GRU uncertainty pre-gate: expected_variance > 0.80 → reject
  2. GRU direction gate: max(p_bull, p_bear) ≥ 0.58 → side = buy if p_bull, else sell
  3. ATR-based entry/SL/TP levels computed
  4. PM enrichment (size, TP1/TP2, correlation cap)
  5. QualityScorer: run with actual trader_id + side + rr_ratio → ev
  6. EV gate: ev < 0.10 → reject
  7. Trade simulated
```

Regime is **encoded as features** in the GRU input (indices 26–37 of SEQUENCE_FEATURES), not applied as a hard gate.

---

## Gates

| Gate | Default | Env override |
|------|---------|--------------|
| GRU uncertainty `expected_variance` | `≤ 0.80` | `MAX_UNCERTAINTY` |
| GRU direction | `≥ 0.58` | — |
| EV threshold | `≥ 0.10` | `MIN_EV_THRESHOLD` |
| Daily loss cap | `2%` | — |
| Max drawdown halt | `8%` | — |
| Cooldown | `10 bars` | — |

---

## Running Containers

| Container | Port | Purpose |
|-----------|------|---------|
| trading_backend | 3000 | FastAPI |
| trading_frontend | 3001 | Vite SPA (nginx) |
| trading_postgres | 5432 | |
| trading_redis | 6379 | pub/sub + state |
| trading_mongodb | 27017 | |
| trading_influxdb | 8086 | |
| trading_grafana | 3002 | dashboards |
| trading_prometheus | 9090 | |
| trading_engine_main | 8000 (expose only) | trading engine (BROKEN for live) |
| trading_model_retrainer | — | retrain_scheduler.py |

---

## Key Configuration

### Auth
- `POST /api/auth/login` — `{ username, password }` or `{ email, password }`
- Credentials from `.env`: `ADMIN_USERNAME=admin`, `ADMIN_PASSWORD=AdminPass2026`
- JWT: `JWT_SECRET`, `JWT_ALGORITHM=HS256`, `JWT_EXPIRES_MINUTES=60`

### Broker
- `BROKER_TYPE=capital` — Capital.com REST API
- `CAPITAL_API_KEY`, `CAPITAL_IDENTIFIER`, `CAPITAL_PASSWORD`, `CAPITAL_ENV=demo`

### Trading
- `PAPER_TRADING=true` (default)
- `ML_ENABLED=true` — all 4 models must be trained before first run
- `ACCOUNT_BALANCE=10000.0`; `CAPITAL_PER_TRADER=0.20`; `RISK_PER_TRADE=0.01`
- `MIN_EV_THRESHOLD=0.10`; `MAX_UNCERTAINTY=0.80`

### Active Symbols
All 11: `EURUSD GBPUSD USDJPY AUDUSD NZDUSD USDCAD USDCHF EURGBP EURJPY GBPJPY XAUUSD`

---

## Common Commands

```bash
# All containers
cd trading-system && docker compose up -d

# Backtest only (from trading-engine/)
cd trading-system/trading-engine
python scripts/run_backtest.py

# Retrain (from trading-engine/)
python scripts/retrain_incremental.py --model regime
python scripts/retrain_incremental.py --model gru
python scripts/retrain_incremental.py --model all

# Offline pipeline (from trading-system/)
export PYTHONPATH="/home/tybobo/Desktop/Multi-Bot/trading-system:/home/tybobo/Desktop/Multi-Bot/trading-system/trading-engine"
python3 run_pipeline.py

# Kaggle full training run
python3 kaggle_train.py

# View journal
tail -f trading-engine/logs/trade_journal.csv
tail -f trading-engine/logs/trade_journal_detailed.jsonl | python -m json.tool
```

---

## Known Issues

| Issue | Severity | File |
|-------|----------|------|
| `main.py` broken — imports `Trader1NYEMA` etc. (deleted) | P0 | `trading-engine/main.py` |
| `signal_pipeline.py` broken — calls `trader.analyze_market()` | P0 | `services/signal_pipeline.py` |
| RL always action=1 — policy collapsed | P1 | `models/rl_agent.py` |
| Regime accuracy low (4H ~49%, 1H ~41%) | P2 | `models/regime_classifier.py` |

**Pending work:**
- Fix `main.py` + `signal_pipeline.py` to use unified ML signal path (no trader classes)
- RL entropy tuning after journal reaches ≥ 200 trades
- EV calibration: isotonic regression on validation set
- Regime transition matrix as additional GRU sequence features
- Rewrite `detect_break_of_structure` / `detect_sr_zones` to non-centered rolling (re-enable zeroed BOS/SR features)
