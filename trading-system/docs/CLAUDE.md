# Multi-Bot Trading System — Claude Reference

Last updated: 2026-04-24. Reflects ML-native architecture (5 rule-based traders removed).

For full ML pipeline details see `docs/system_architecture.md`.
For training and backtest runbook see `docs/TRAINING_AND_BACKTEST.md`.

---

## Current System State

**The 5 ICT rule-based traders are gone.** `traders/__init__.py` contains only a comment.
Signal generation uses a single unified ML path: `_compute_backtest_signal` in `run_backtest.py`
(offline/backtest) and `_compute_ml_signal` in `signal_pipeline.py` (live/paper) — both are
kept in exact sync. `run_backtest._compute_backtest_signal` is the source of truth.

`main.py` (trading engine entry point) and `signal_pipeline.py` are both functional — they
no longer import deleted trader classes.

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
│                                       Reinforced loop: backtest → retrain regime/quality/RL × 3 rounds
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
    ├── main.py                       ← ProductionTradingEngine (functional)
    ├── config/settings.py            ← Pydantic BaseSettings
    ├── indicators/market_structure.py ← all vectorized; no .at[i] indexing
    ├── services/
    │   ├── feature_engine.py         ← all feature vectors (SEQUENCE_FEATURES=74,
    │   │                               REGIME_4H_FEATURES=34, REGIME_1H_FEATURES=18,
    │   │                               QUALITY_FEATURES=17, RL_STATE_DIM=43)
    │   ├── signal_pipeline.py        ← live ML signal path (mirrors run_backtest exactly)
    │   ├── data_fetcher.py
    │   ├── broker_connector.py
    │   ├── order_executor.py
    │   ├── risk_engine.py
    │   └── trade_journal.py
    ├── models/
    │   ├── base_model.py
    │   ├── regime_classifier.py      ← dual-cascade: HTF 4H bias (34 feat, 3-class)
    │   │                               + LTF 1H behaviour (18 feat, 4-class)
    │   ├── gru_lstm_predictor.py     ← GRU(64,2L)→LSTM(128,2L)→3 heads; 74 SEQUENCE_FEATURES
    │   │                               temperature.pt sidecar for post-hoc calibration
    │   ├── quality_scorer.py         ← EV regressor; class-weighted Huber; 17 QUALITY_FEATURES
    │   ├── rf_direction.py           ← Random Forest direction ensemble; 30 tabular features
    │   │                               blends with GRU: RF_BLEND_WEIGHT (default 0.30)
    │   ├── win_loss_classifier.py    ← ANN binary win/loss on trade outcomes; 23 features
    │   │                               optional gate: WIN_LOSS_GATE_ENABLED=1, WIN_LOSS_MIN_PROB=0.45
    │   ├── kmeans_regime.py          ← K-Means unsupervised 4H clustering; k=8; KMEANS_N_CLUSTERS
    │   │                               outputs kmeans_regime_id [0,1] in ml_preds
    │   ├── sentiment_model.py        ← FinBERT primary; VADER fallback
    │   ├── rl_agent.py               ← PPO via SB3; CPU; 43-dim state; 16 actions
    │   ├── vector_store.py           ← FAISS index of 64-dim GRU embeddings
    │   └── weights/
    │       ├── gru_lstm/
    │       │   ├── model.pt
    │       │   └── temperature.pt    ← scalar T for sigmoid(logit/T) calibration
    │       ├── regime_htf.pkl        ← HTF bias (3-class: BIAS_UP/DOWN/NEUTRAL)
    │       ├── regime_ltf.pkl        ← LTF behaviour (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)
    │       ├── quality_scorer.pkl
    │       ├── rf_direction/         ← RF ensemble (model.pkl + meta.json)
    │       ├── win_loss_classifier/  ← ANN win/loss (model.pt + meta.json)
    │       ├── kmeans_regime/        ← K-Means (model.pkl + scaler.pkl + meta.json)
    │       └── rl_ppo/model.zip
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
| RegimeClassifier HTF | Directional bias — "what is macro direction?" from 4H+1D | 3-class (BIAS_UP/DOWN/NEUTRAL) + conf | `weights/regime_htf.pkl` |
| RegimeClassifier LTF | Behaviour — "how is price acting?" from 1H+4H | 4-class (TRENDING/RANGING/CONSOLIDATING/VOLATILE) + conf | `weights/regime_ltf.pkl` |
| GRU-LSTM | Direction + magnitude + uncertainty | `p_bull`, `p_bear`, `expected_move`, `expected_variance` | `weights/gru_lstm/model.pt` + `temperature.pt` |
| QualityScorer | EV in R-multiples (runs post-signal with real rr_ratio) | `ev`, `quality_score` | `weights/quality_scorer.pkl` |
| SentimentModel | News headline scoring | `sentiment_score`, `sentiment_label` | pre-trained |
| RLAgent | Selectivity tier selection (CPU) | action 0–15 | `weights/rl_ppo/model.zip` |
| VectorStore | FAISS similarity index of GRU embeddings | nearest trade patterns | `weights/gru_lstm/vector_store/` |
| RFDirectionClassifier | Random Forest tabular direction ensemble | `p_bull_rf` blended with GRU (weight 0.30) | `weights/rf_direction/model.pkl` |
| WinLossClassifier | ANN binary win/loss classifier on trade outcomes | `p_win_ann` ∈ [0,1] (logged; gate with WIN_LOSS_GATE_ENABLED=1) | `weights/win_loss_classifier/model.pt` |
| KMeansRegimeModel | K-Means unsupervised 4H regime clustering (k=8) | `kmeans_regime_id` normalised [0,1] in ml_preds | `weights/kmeans_regime/model.pkl` |

**Feature counts — fixed contract. Changing order or length breaks saved weights.**

| List | Length | Model |
|------|--------|-------|
| `SEQUENCE_FEATURES` | 74 | GRU-LSTM |
| `REGIME_4H_FEATURES` | 34 | RegimeClassifier (4H) |
| `REGIME_1H_FEATURES` | 18 | RegimeClassifier (1H) |
| `QUALITY_FEATURES` | 17 | QualityScorer |
| `RL_STATE_DIM` | 43 | RLAgent |

---

## Signal Generation (Working Path)

`scripts/run_backtest.py` — `_compute_backtest_signal()` with `trader_id="ml_trader"`.
Mirrored exactly in `services/signal_pipeline.py` — `_compute_ml_signal()`.

Gate order (same in both backtest and live):
```
1. GRU uncertainty: expected_variance > MAX_UNCERTAINTY → reject
2. GRU direction:   max(p_bull, p_bear) < 0.58 → reject; side = buy/sell
3. HTF bias:        BIAS_UP + sell → reject; BIAS_DOWN + buy → reject
                    BIAS_NEUTRAL: require conf ≥ NEUTRAL_BIAS_THRESHOLD (0.58)
4. LTF behaviour:
     CONSOLIDATING → reject (if BLOCK_LTF_CONSOLIDATING=1)
     VOLATILE      → require conf ≥ VOLATILE_ENTRY_THRESHOLD
     RANGING       → optional range boundary check (RANGING_REQUIRE_RANGE)
     TRENDING      → optional pullback filter (REQUIRE_TRENDING_PULLBACK)
5. ATR-based entry/SL/TP levels
6. PM enrichment (size, TP1/TP2, correlation cap)
7. QualityScorer: ev with actual rr_ratio → reject if ev < MIN_EV_THRESHOLD (0.10)
8. Dead zone 12:00–13:00 UTC / cooldown / daily cap / drawdown halt
```

Signal pipeline additionally gates on `confidence ≥ 0.55` before publishing.

---

## Gates

| Gate | Default | Env override |
|------|---------|--------------|
| GRU uncertainty `expected_variance` | `≤ 2.0` | `MAX_UNCERTAINTY` |
| GRU direction | `≥ 0.58` | `ML_DIRECTION_THRESHOLD` |
| HTF neutral confidence | `≥ 0.58` | `NEUTRAL_BIAS_THRESHOLD` |
| VOLATILE entry | `≥ ML_DIRECTION_THRESHOLD` | `VOLATILE_ENTRY_THRESHOLD` |
| EV threshold | `≥ 0.10` | `MIN_EV_THRESHOLD` |
| Daily loss cap | `2%` | — |
| Max drawdown halt | `8%` | — |
| Cooldown | `10 bars` | — |
| Signal pipeline confidence | `≥ 0.55` | — |

---

## Running Containers

| Container | Port | Purpose |
|-----------|------|---------|
| trading_backend | 3000 | FastAPI |
| trading_frontend | 3001 | Vite SPA (nginx) |
| trading_postgres | 5432 | trade journal, state |
| trading_redis | 6379 | pub/sub + state |
| trading_engine_main | 8000 (internal) | trading engine |
| trading_model_retrainer | — | retrain_scheduler.py |

---

## Key Configuration

### Auth
- `POST /api/auth/login` — `{ username, password }` or `{ email, password }`
- Credentials from `.env`: `ADMIN_USERNAME=admin`, `ADMIN_PASSWORD=AdminPass2026`
- JWT: `JWT_SECRET`, `JWT_ALGORITHM=HS256`, `JWT_EXPIRES_MINUTES=60`

### Broker
- `BROKER_TYPE=capital` — Capital.com REST API
- Live trading: Capital.com live API (`CAPITAL_ENV=live`)
- Paper trading: Capital.com demo API (`CAPITAL_ENV=demo`, default)
- `CAPITAL_API_KEY`, `CAPITAL_IDENTIFIER`, `CAPITAL_PASSWORD`

### Trading
- `PAPER_TRADING=true` (default)
- `ML_ENABLED=true` — all 4 models must be trained before first run
- `ACCOUNT_BALANCE=10000.0`; `CAPITAL_PER_TRADER=0.20`; `RISK_PER_TRADE=0.01`
- `MIN_EV_THRESHOLD=0.10`; `MAX_UNCERTAINTY=2.0`

### Pydantic (Kaggle compatibility)
- `pydantic==2.7.4`, `pydantic-core==2.18.4`, `pydantic-settings==2.3.4` — pinned in `requirements.txt`

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
python scripts/retrain_incremental.py --model rf        # Random Forest direction
python scripts/retrain_incremental.py --model kmeans    # K-Means regime clustering
python scripts/retrain_incremental.py --model win_loss  # ANN win/loss classifier
python scripts/retrain_incremental.py --model all       # gru+regime+quality+rl+rf+kmeans

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
| RL policy needs more data for action diversity | P2 | `models/rl_agent.py` — needs ≥200 journal trades |
| Regime accuracy improving | P2 | `models/regime_classifier.py` — atr_pctile bug fixed; retrain expected to improve LTF RANGING |

**Pending work:**
- RL entropy tuning after journal reaches ≥ 200 trades
- EV calibration: isotonic regression on validation set
- Regime transition matrix as additional GRU sequence features
- Rewrite `detect_break_of_structure` / `detect_sr_zones` to non-centered rolling (re-enable zeroed BOS/SR features)
