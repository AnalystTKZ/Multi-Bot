# System Architecture — ML-Native Trading System

Updated 2026-04-24. For the full living reference see `docs/system_architecture.md` and `docs/TRAINING_AND_BACKTEST.md`.

> **ML_ENABLED=true is the default.** The engine raises `ModelNotTrainedError` if any model weight is missing. Train all models before starting — see `docs/TRAINING_AND_BACKTEST.md`.

---

## ICT / Smart Money Concepts

ICT concepts are **features**, not rules. They are encoded as numeric distances and fed to the GRU — the model learns which patterns matter in which regime.

**Fair Value Gap (FVG)**
Three-candle imbalance. Encoded as: distance to nearest open FVG / ATR, fill ratio [0–1].

**Break of Structure (BOS)**
Break of prior swing high/low. Encoded as: bars since last BOS / 40, BOS move size / ATR.

**Liquidity Sweep**
Stop-hunt beyond range extremes. Encoded as: wick depth beyond 20-bar range / ATR, body recovery ratio.

**EMA Pullback Zone**
Price inside EMA21-50 band. Encoded as: band-normalised position, EMA21 slope / ATR.

**Asian Range**
02:00–07:00 UTC session range. Encoded as: range width / ATR, price vs Asian high/low.

> **Causality note:** `detect_break_of_structure`, `detect_sr_zones`, and range detection now emit swing levels only after right-side confirmation bars have closed. Regime S/R feature slots remain zeroed to preserve the trained feature distribution until a deliberate retrain enables them.

---

## Full System Architecture (Backtest / Working Path)

```
processed_data/histdata/{SYMBOL}_{TF}.parquet
        │
        ▼
  run_backtest._precompute_ml_cache(symbol, df, htf)
        │
        ├── RegimeClassifier (4H bias)   — 34 features → regime_4h + conf
        ├── RegimeClassifier (1H struct) — 18 features → regime_1h + conf
        ├── Build 74-feature sequence matrix (regime injected at each timestep)
        └── GRU-LSTM batch inference     — (N, 30, 74) → p_bull, p_bear, expected_variance
        │
        ▼
  Bar loop (dict lookups only):
        ├── Gate 1: expected_variance > MAX_UNCERTAINTY (env default 2.0) → skip
        ├── Gate 2: max(p_bull, p_bear) < 0.58 → skip
        ├── Dead zone 12:00–13:00 UTC → skip
        ├── Cooldown 10 bars → skip
        ├── Daily loss cap 2% / drawdown 8% → skip
        ├── ATR-based entry/SL/TP levels
        ├── PortfolioManager enrichment (size, TP1/TP2, correlation cap)
        ├── QualityScorer (post-PM, uses actual rr_ratio) → ev
        ├── Gate 3: ev < 0.10 → skip
        └── Simulate trade → journal
```

---

## Live Trading Architecture

`main.py` and `signal_pipeline.py` are both functional. `signal_pipeline._compute_ml_signal`
mirrors `run_backtest._compute_backtest_signal` exactly and is the live inference path.

Architecture:
```
Capital.com REST API → DataFetcher → MARKET_DATA (Redis)
        │
        ▼
  ProductionTradingEngine.on_market_data()
        │
        ▼
  SignalPipeline.process_bar()
        ├── RegimeClassifier (4H + 1H)
        ├── GRU-LSTM
        ├── QualityScorer (post-signal)
        ├── RLAgent (selectivity tier)
        └── → SIGNAL_GENERATED (Redis)
        │
        ▼
  RiskEngine → ExecutionEngine → PaperTradingService / BrokerConnector
        │
        ▼  TRADE_EXECUTED
  TradeJournal → CSV + JSONL
```

---

## ML Models

All models are PyTorch on GPU except RL. **LightGBM and XGBoost removed.**

| Model | Algorithm | Weight | Role |
|---|---|---|---|
| RegimeClassifier (4H) | PyTorch MLP 34→128→64→3, BatchNorm, GELU, residual | `weights/regime_htf.pkl` | 4H macro bias: BIAS_UP/DOWN/NEUTRAL |
| RegimeClassifier (1H) | PyTorch MLP 18→128→64→4, BatchNorm, GELU, residual | `weights/regime_ltf.pkl` | 1H behaviour: TRENDING/RANGING/CONSOLIDATING/VOLATILE |
| GRU-LSTM | GRU(64,2L)→LSTM(128,2L)→shared FC→3 heads + temperature.pt | `weights/gru_lstm/model.pt` | p_bull, p_bear, magnitude, uncertainty |
| QualityScorer | PyTorch MLP 17→64→32→1, class-weighted Huber loss, identity output | `weights/quality_scorer.pkl` | EV in R-multiples |
| SentimentModel | FinBERT (HuggingFace) + VADER fallback | pre-trained | News/macro directional bias |
| RLAgent | PPO via stable-baselines3, CPU, 43-dim state, 16 actions | `weights/rl_ppo/model.zip` | Selectivity tier |

**No fallback values** — untrained models always raise `ModelNotTrainedError(RuntimeError)`.

**Warm-start incremental retraining:** existing weights loaded and fine-tuned at 5× lower LR.
GRU is excluded from per-round reinforcement loop (catastrophic forgetting risk).

**EV label tiers (QualityScorer):**
| Exit | EV label |
|---|---|
| `tp2` | `+rr_ratio` |
| `tp1` | `+rr × 0.75` |
| `be_or_trail` | `+rr × 0.4` |
| `sl_*` | `-1.0` |

---

## Feature Contracts (fixed — changing breaks saved weights)

| List | Length | Used by |
|------|--------|---------|
| `SEQUENCE_FEATURES` | 74 | GRU training + inference |
| `REGIME_4H_FEATURES` | 34 | RegimeClassifier (4H bias) |
| `REGIME_1H_FEATURES` | 18 | RegimeClassifier (1H behaviour) |
| `QUALITY_FEATURES` | 17 | QualityScorer |
| `RL_STATE_DIM` | 43 | RLAgent |

---

## Offline Data Pipeline (`pipeline/`)

9-step pipeline for datasets, backtests, and training. Runs on Kaggle T4 × 2 GPUs via `kaggle_train.py`.

| Step | Script | Output |
|---|---|---|
| 0 | `step0_resample.py` | `processed_data/histdata/{SYM}_{TF}.parquet` |
| 1 | `step1_inventory.py` | `processed_data/raw_inventory.json` |
| 2 | `step2_clean.py` | Clean parquets with macro columns |
| 3 | `step3_align.py` | `processed_data/aligned_multi_asset.parquet` |
| 4 | `step4_features.py` | Engineered features |
| 5 | `step5_split.py` | `ml_training/datasets/train|val|test.parquet` (70/15/15) |
| 6 | `step6_backtest.py` | 3-round reinforcement backtest + journal |
| 7a | `step7a_train.py` | GRU + Regime weights |
| 7b | `step7b_train.py` | Quality + RL weights (requires journal from step 6) |
| 8 | `step8_validate.py` | Metrics |

---

## Retrain Schedule

| Cadence | Models | Min data |
|---|---|---|
| Weekly (Sun 02:00 UTC) | Quality + RL | 200 trades (Quality), 500 (RL) |
| Monthly (1st Sun 03:00 UTC) | Regime + GRU | OHLCV only |

---

## Docker Services

| Container | Port | Purpose |
|---|---|---|
| trading_backend | 3000 | FastAPI |
| trading_frontend | 3001 | React/Nginx |
| trading_postgres | 5432 | Trade journal, state |
| trading_redis | 6379 | pub/sub + state |
| trading_engine_main | 8000 (internal) | Trading engine (currently broken) |
| trading_model_retrainer | — | `retrain_scheduler.py` |

---

## Causal Integrity

| Feature | Status |
|---------|--------|
| `sr_dist_*`, `sr_in_*`, `sr_*_strength` (REGIME_FEATURES 28–33) | Zeroed for feature-distribution compatibility; detector is causal but enabling requires retrain |
| Macro `bfill()` | Removed — replaced with `fillna(0.0)` only |
| All rolling indicators | Backward-only |
| HTF `reindex(method="ffill")` | Causal — only HTF bars ≤ t contribute |

---

## Known Issues

| Issue | File | Impact |
|-------|------|--------|
| RL policy needs action diversity | `models/rl_agent.py` | Needs ≥200 journal trades + entropy tuning |
| HTF BIAS_NEUTRAL recall ~30-38% | `models/regime_classifier.py` | NEUTRAL bars under-predicted; being improved |
| VectorStore broken | `models/vector_store.py` | numpy import bug + dim mismatch; being fixed |
