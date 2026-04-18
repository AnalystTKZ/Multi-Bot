# Multi-Bot Trading System — Claude Reference

Last updated: 2026-04-17. Reflects current ML-native architecture.

Last updated: 2026-04-18. All models are PyTorch on GPU — LightGBM/XGBoost removed.

For full ML pipeline details see `docs/system_architecture.md`.
For training and backtest runbook see `docs/TRAINING_AND_BACKTEST.md`.

---

## Repository Layout

```
trading-system/
├── .env                              ← single source of truth (all services read from here)
├── docker-compose.dev.yml
├── env_config.py                     ← path resolver (_ENV dict) used by all scripts
├── docs/
│   ├── CLAUDE.md                     ← this file
│   ├── system_architecture.md        ← full ML pipeline, model details, causal integrity
│   ├── TRAINING_AND_BACKTEST.md      ← training + backtest runbook
│   ├── strategies.md
│   └── models.md
├── pipeline/                         ← 9-step offline data pipeline
│   ├── step0_resample.py             ← M1 histdata → 5 MTF parquets per symbol
│   ├── step1_inventory.py
│   ├── step2_clean.py
│   ├── step3_align.py
│   ├── step4_features.py
│   ├── step5_split.py
│   ├── step6_backtest.py
│   ├── step7_train.py
│   └── step8_validate.py
├── run_pipeline.py                   ← orchestrates pipeline; skips completed steps
├── processed_data/
│   ├── histdata/                     ← {SYMBOL}_{5M|15M|1H|4H|1D}.parquet (step 0)
│   └── ...
├── ml_training/
│   ├── datasets/                     ← train/val/test.parquet + split_summary.json
│   └── metrics/
├── training_data/
│   ├── forex/                        ← *_m1_histdata.csv (2016–2026, ~187 MB each)
│   ├── indices/                      ← *_1d.csv (ASX200, DAX, DXY, VIX, etc.)
│   └── fundamental/                  ← treasury_10yr.csv, treasury_2yr.csv
├── backend/
│   ├── Dockerfile                    ← python:3.11-slim
│   └── src/
│       ├── main.py                   ← FastAPI app + CORS
│       ├── routes/
│       │   ├── auth.py               ← JWT login (username OR email)
│       │   ├── traders.py            ← /api/traders/ (trailing slash required)
│       │   ├── positions.py
│       │   ├── analytics.py
│       │   ├── monitors.py
│       │   └── system.py             ← /api/system/, /api/backtest/, /api/training/, /api/ml/
│       └── services/
│           ├── state_reader.py       ← reads Redis state from trading-engine
│           └── trading_engine_client.py
├── frontend/
│   ├── Dockerfile                    ← node:20-alpine build → nginx:1.27-alpine
│   └── src/
│       ├── services/api.js           ← Axios; auto Bearer; 401 guard
│       └── store/slices/
│           ├── authSlice.js          ← loginUser thunk
│           ├── positionsSlice.js     ← closePosition filters by meta.arg.id
│           └── tradersSlice.js       ← performance keyed by trader_id
└── trading-engine/
    ├── main.py                       ← ProductionTradingEngine; health server port 8000
    ├── config/settings.py            ← Pydantic BaseSettings
    ├── indicators/market_structure.py ← all vectorized; no .at[i] indexing
    ├── services/
    │   ├── feature_engine.py         ← all feature vectors; SEQUENCE_FEATURES, REGIME_FEATURES,
    │   │                               QUALITY_FEATURES, RL_STATE_DIM
    │   ├── signal_pipeline.py        ← ML inference + 5-trader signal generation
    │   ├── data_fetcher.py           ← Capital.com REST; raises on session failure
    │   ├── broker_connector.py
    │   ├── order_executor.py
    │   ├── risk_engine.py
    │   └── trade_journal.py          ← CSV + JSONL; source of QualityScorer labels
    ├── models/
    │   ├── base_model.py             ← BaseModel(ABC); reload_if_updated() mtime check
    │   ├── regime_classifier.py      ← PyTorch MLP 4-class; GMM labels; group-aware
    │   ├── gru_lstm_predictor.py     ← PyTorch GRU 3-head; regime-conditioned
    │   ├── quality_scorer.py         ← PyTorch MLP EV regressor; Huber loss
    │   ├── sentiment_model.py        ← FinBERT primary; VADER fallback
    │   ├── rl_agent.py               ← PPO via SB3; 42-dim state; 16 actions
    │   └── weights/                  ← cwd-relative; run scripts from trading-engine/
    │       ├── gru_lstm/model.pt
    │       ├── regime_classifier.pkl
    │       ├── quality_scorer.pkl
    │       ├── rl_ppo/
    │       └── backups/
    ├── traders/
    │   ├── base_trader.py            ← 8 guards; EV gate (Guard 7); RL gate (Guard 8)
    │   ├── trader_1_ny_ema.py        ← NY EMA Trend Pullback (13–17 UTC)
    │   ├── trader_2_fvg_bos.py       ← Structure Break + FVG (London + NY)
    │   ├── trader_3_london_bo.py     ← London Breakout + Sweep (07–10 UTC)
    │   ├── trader_4_news_momentum.py ← News Momentum (any time)
    │   └── trader_5_asian_mr.py      ← Asian Range MR (02–06:45 UTC)
    └── scripts/
        ├── run_backtest.py           ← batched GPU inference + 5-trader bar loop
        ├── retrain_incremental.py    ← --model gru|regime|quality|rl|all
        └── retrain_scheduler.py     ← fires Sunday 02:00 UTC
```

---

## Current ML Architecture (as of 2026-04-18)

**This is an ML-native system.** ICT/SMC conditions determine entry levels. ML models determine whether to trade. All models are PyTorch on GPU.

| Model | Role | Output | Latest |
|-------|------|--------|--------|
| RegimeClassifier | Labels market state (4 classes) | `regime`, `regime_id`, `proba[4]`, `regime_confidence` | 4H 48.8% / 1H 41.1% acc |
| GRU-LSTM | Predicts direction + magnitude + uncertainty | `p_bull`, `p_bear`, `expected_move`, `expected_variance` | 7.45M samples, 44 combos |
| QualityScorer | Predicts EV in R-multiples | `ev`, `quality_score` | 8,203 journal trades |
| SentimentModel | News sentiment | `sentiment_score`, `sentiment_label` | Pre-trained (FinBERT) |
| RLAgent | Selects trader + selectivity tier | `(trader_id, ev_threshold)` | 16 actions, warm-start |

**Key architectural facts:**
- GRU receives `prev_regime_onehot` (4 dims) + `regime_confidence` at every sequence timestep
- Regime computed BEFORE GRU in backtest — `_precompute_ml_cache` order matters
- QualityScorer: EV regressor (class-weighted Huber loss, `rr_ratio` labels, NOT `pnl/risk_staked`)
- EV gate in Guard 7: `ev ≥ 0.10` AND `expected_variance ≤ 0.80` AND `p_dir ≥ ML_DIRECTION_THRESHOLD`
- GRU excluded from per-round warm-start loop (catastrophic forgetting on ~3k trades vs 7.4M sequences)
- All retrains are warm-start: existing weights + 5× lower LR, not reinitialised from scratch
- No fallback values — all failures raise and propagate

**Feature counts (fixed contract — changing breaks saved weights):**

| List | Length | Model |
|------|--------|-------|
| `SEQUENCE_FEATURES` | 53 | GRU |
| `REGIME_FEATURES` | 59 | RegimeClassifier |
| `QUALITY_FEATURES` | 17 | QualityScorer |
| `RL_STATE_DIM` | 42 | RLAgent |

**Known issues (2026-04-18):**
- Quality score = 0.0 on all backtest trades — scorer trained but output not reaching inference path (P0)
- RL action = 1 for all trades — policy collapsed, needs more journal data + `ent_coef=0.01`

---

## Guard 7 — EV Gate (current)

```python
# Guard 7 in base_trader.py
ev = ml_predictions.get("ev")
if ev is None:
    raise RuntimeError("EV model output missing — ensure QualityScorer is trained")
if float(ev) < self.MIN_EV_THRESHOLD:       # default 0.10, env: MIN_EV_THRESHOLD
    return None
uncertainty = float(ml_predictions.get("expected_variance", 0.0))
if uncertainty > self.MAX_UNCERTAINTY:       # default 0.80, env: MAX_UNCERTAINTY
    return None
# then direction gate: p_bull/p_bear ≥ ML_DIRECTION_THRESHOLD
```

`ML_QUALITY_THRESHOLD` has been removed from all traders — EV is the only gate.

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
| trading_engine_main | 8000 (expose only) | trading engine |
| trading_model_retrainer | — | retrain_scheduler.py |

Engine port 8000 is `expose` only — backend reaches it via `http://trading-engine:8000`, not `localhost:8000`.

---

## Key Configuration

### Auth
- `POST /api/auth/login` — `{ username, password }` or `{ email, password }`
- Credentials from `.env`: `ADMIN_USERNAME=admin`, `ADMIN_PASSWORD=AdminPass2026`
- JWT: `JWT_SECRET`, `JWT_ALGORITHM=HS256`, `JWT_EXPIRES_MINUTES=60`

### Broker
- `BROKER_TYPE=capital` — Capital.com REST API (not MT5; Linux-incompatible)
- `CAPITAL_API_KEY`, `CAPITAL_IDENTIFIER`, `CAPITAL_PASSWORD`, `CAPITAL_ENV=demo`
- Base URL: `https://api-demo.capital.com`
- yfinance is data fallback (not order execution)

### Trading
- `PAPER_TRADING=true` (default) — no live orders
- `ML_ENABLED=true` — default; models must be trained before first run
- `ACCOUNT_BALANCE=10000.0`; `CAPITAL_PER_TRADER=0.20`; `RISK_PER_TRADE=0.01`
- `MIN_EV_THRESHOLD=0.10`; `MAX_UNCERTAINTY=0.80`

### Active Symbols
All 11: `EURUSD GBPUSD USDJPY AUDUSD NZDUSD USDCAD USDCHF EURGBP EURJPY GBPJPY XAUUSD`

---

## Common Commands

```bash
# All containers
cd trading-system && docker compose up -d

# Engine only
docker compose build trading-engine && docker compose up -d trading-engine

# Logs
docker compose logs trading-engine --tail=50 -f

# Backtest (GPU — runs on 2× T4 in Kaggle)
docker exec trading_engine_main python /app/scripts/run_backtest.py

# Retrain
docker exec trading_engine_main python /app/scripts/retrain_incremental.py --model regime
docker exec trading_engine_main python /app/scripts/retrain_incremental.py --model gru
docker exec trading_engine_main python /app/scripts/retrain_incremental.py --model all

# Local (from trading-engine/ directory — weights paths are cwd-relative)
cd trading-system/trading-engine
python scripts/retrain_incremental.py --model regime
python scripts/run_backtest.py

# Offline pipeline (from trading-system/)
export PYTHONPATH="/home/tybobo/Desktop/Multi-Bot/trading-system:/home/tybobo/Desktop/Multi-Bot/trading-system/trading-engine"
python3 run_pipeline.py

# View journal
tail -f trading-engine/logs/trade_journal.csv
tail -f trading-engine/logs/trade_journal_detailed.jsonl | python -m json.tool
```

---

## Redis Key Contracts

**Engine writes (`state_manager.py`):**

| Key | Content |
|-----|---------|
| `engine:status` | `running` / `stopped` / `error` |
| `engine:mode` | `paper` / `live` |
| `engine:heartbeat` | ISO timestamp, every 30s |
| `trader:{id}:state` | `{status, trades_today, pnl_today, win_rate, last_signal}` |
| `strategy_allocations:{id}` | `{is_active, allocated_capital, used_capital, current_risk}` |
| `ml:model:{id}:status` | `{name, status, accuracy}` |
| `positions:open` | JSON list |

**Engine publishes (`event_bus.py`):**

| Channel | When |
|---------|------|
| `SIGNAL_GENERATED` | Signal passes all guards + ensemble gate |
| `TRADE_EXECUTED` | Execution complete |
| `MARKET_DATA` | Every bar |

---

## Trade Journal

**CSV** (`logs/trade_journal.csv`): `timestamp, trader, symbol, side, size, entry, stop_loss, take_profit, rr_ratio, confidence, pnl, commission`

**JSONL** (`logs/trade_journal_detailed.jsonl`): all CSV fields + `strategy, session, exit_reason, correlation_id, state_at_entry[42], rl_action, ml_model_scores, signal_metadata, ev, expected_variance`

The JSONL is the source of truth for QualityScorer retraining. Each record needs `pnl`, `entry`, `stop_loss`, `size`, and `exit_reason` containing "tp" or "sl".

---

## API Routes

All require `Authorization: Bearer <token>` except `/api/auth/login` and `/health`.

| Prefix | Notes |
|--------|-------|
| `/api/auth` | login, logout |
| `/api/traders/` | GET collection (trailing slash required — FastAPI 307 strips auth header) |
| `/api/positions` | GET, POST `/{ticket}/close` |
| `/api/analytics` | |
| `/api/system` | status, mode |
| `/api/backtest` | reads `backtest_results/*.json` |
| `/api/training` | status, start |
| `/api/ml/models` | model list from Redis + filesystem |

---

## Indicators (`indicators/market_structure.py`)

All vectorized — no `.at[i]` integer indexing.

**Causal:** `compute_ema`, `compute_atr`, `compute_rsi`, `compute_adx`, `compute_stochastic`, `compute_bollinger_bands`, `compute_ema_stack_score`, `detect_fair_value_gaps`, `detect_liquidity_sweeps`, `detect_order_blocks`

**Lookahead (do not use for regime features or sequence features):**
- `detect_break_of_structure` — uses `rolling(center=True)`, reads 5 future bars
- `detect_sr_zones` — uses `rolling(center=True)`, reads 5 future bars

These are currently zeroed in all feature arrays. They can be used in `_compute_signal()` for entry/exit logic (live bar has no future) but must not appear in training features.

---

## Known Issues / Next Steps

**Validation required before trusting backtest numbers:**
1. Regime sanity: mean_return + volatility + avg_duration per regime (expect TU > 0, TD < 0, separation)
2. EV distribution: histogram should show most trades 0.0–0.5 EV, tail to 2.0+ (not centered at 0)
3. Trade frequency: trades per day per regime (if too high = overfitting)
4. PnL by regime: if regimes don't differentiate PnL → regime model is useless

**Pending work (in priority order):**
- P0: Fix quality score pipeline — trace `ev`/`quality_score` from `quality_scorer.predict()` through `_run_ml_inference()` into Guard 7
- P0: Fix RL policy collapse — add `ent_coef=0.01`, accumulate ≥500 journal trades before next retrain
- P1: Monitor Round 2→3 PF decay across future runs (currently ~0.5pp; expected to stabilise)
- P2: Raise `ML_DIRECTION_THRESHOLD` for EURUSD (currently 37% WR vs 47–57% for other pairs)
- P3: Test `min_confidence=0.90` filter — 67% WR identified on 291 trades
- Regime transition probabilities: 4×4 matrix flattened → add to GRU sequence features
- `detect_break_of_structure` and `detect_sr_zones`: rewrite to use non-centered rolling
  so BOS/sweep/S/R features can be re-enabled in training

**Not pending (deliberately excluded):**
- More models
- More features beyond what's already planned
- Backwards-compatibility shims for old LightGBM / XGBoost weights (deleted)
