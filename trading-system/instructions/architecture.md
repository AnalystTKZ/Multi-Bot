# System Architecture — Multi-Bot ICT/Smart Money Trading System

Updated 2026-04-18. For the full living reference see `docs/CLAUDE.md`.

> **ML_ENABLED=true is the default.** The engine raises `ModelNotTrainedError` if any model weight is missing. Train all models before starting — see `docs/TRAINING_AND_BACKTEST.md`.

---

## ICT / Smart Money Concepts

**Order Block (OB)**
Last opposing candle before a strong impulse. Institutional entry or mitigation zone.

**Fair Value Gap (FVG)**
Three-candle imbalance where price skipped liquidity. Often revisited and filled.
Canonical detection: `high.shift(2) < low` (bullish); `low.shift(2) > high` (bearish). All vectorized.

**Liquidity Sweep**
Stop-hunt beyond prior highs or lows followed by reversal.

**Break of Structure (BOS)**
Break of prior swing high or low in trend direction. Detected via `rolling(window).max/min`.

**Change of Character (CHoCH)**
BOS occurring against the prevailing structural direction — early reversal signal.

> **Lookahead note:** `detect_break_of_structure` and `detect_sr_zones` use `rolling(center=True)` internally — they read future bars. These are safe for live `_compute_signal()` logic (current bar has no future) but are **zeroed** in all training feature arrays.

---

## Full System Architecture

```
Capital.com REST API (demo/live)
        │
        ▼
  DataFetcher — OHLCV per symbol per timeframe (300 bars)
        │
        ▼  MARKET_DATA event (Redis pub/sub)
  ProductionTradingEngine.on_market_data()
        │
        ▼
  SignalPipeline.process_bar(symbol, df, df_htf)
        │
        ├── Step 1: ML inference (RegimeClassifier → GRU-LSTM → QualityScorer → Sentiment)
        │     Order is load-bearing: Regime feeds into GRU sequence features
        │
        ├── Step 2: 5 traders → BaseTrader.analyze_market()
        │     ├── 8 guards (session, dead zone, cooldown, caps, circuit breaker, spread, EV, RL)
        │     └── _compute_signal() → signal dict or None
        │
        ├── Step 3: Ensemble score
        │     score = (ict_score × 0.5 + ml_score × 0.5 + sentiment_bonus) × regime_mult
        │
        └── Step 4: Confidence gate ≥ 0.55 → SIGNAL_GENERATED (Redis pub/sub)
        │
        ▼
  RiskEngine.check_pre_trade()
        │
        ▼
  ExecutionEngine.execute_trade(ExecutionRequest)
        │
        ├── PaperTradingService  (PAPER_TRADING=true)
        └── BrokerConnector      (PAPER_TRADING=false)
        │
        ▼  TRADE_EXECUTED (Redis pub/sub)
  TradeJournal
        ├── logs/trade_journal.csv
        └── logs/trade_journal_detailed.jsonl    (state_at_entry, rl_action, ml_model_scores, ev)
        │
        ▼  (feedback loop — incremental warm-start retraining)
  retrain_scheduler.py
        ├── Weekly  (Sun 02:00 UTC): Quality + RL  (journal-adaptive)
        └── Monthly (1st Sun 03:00 UTC): Regime + GRU  (structural)

Redis pub/sub → backend WebSocket manager → frontend WebSocket → Redux store
```

---

## ML Models (current)

All models are PyTorch on GPU. **LightGBM and XGBoost have been removed.**

| Model | Algorithm | Weight | Role |
|---|---|---|---|
| RegimeClassifier | PyTorch MLP 59→128→64→4, BatchNorm, GELU, residual | `weights/regime_4h.pkl` / `regime_1h.pkl` | 4-class market regime |
| GRU-LSTM | PyTorch GRU(256, 2L) + LayerNorm → 3 heads | `weights/gru_lstm/model.pt` | p_bull, p_bear, magnitude, uncertainty |
| QualityScorer | PyTorch MLP 17→64→32→1, Huber loss | `weights/quality_scorer.pkl` | EV in R-multiples |
| SentimentModel | FinBERT (HuggingFace) + VADER fallback | pre-trained | News/macro directional bias |
| RLAgent | PPO via stable-baselines3, 42-dim state, 16 actions | `weights/rl_ppo/` | Strategy selector + EV threshold |

**No fallback values** — untrained models always raise `ModelNotTrainedError(RuntimeError)`.

**Warm-start incremental retraining:** all models detect existing weights at fit time and continue training at 5× lower LR rather than reinitialising. GRU is excluded from the per-round reinforcement loop (catastrophic forgetting — 7.4M sequence knowledge would be destroyed by fine-tuning on ~3k journal trades).

---

## BaseTrader 8 Guards

| Guard | Check |
|---|---|
| 0 | `EXCLUDED_SYMBOLS` — hard exclusion list per trader |
| 1 | Session window: `SESSION_START_UTC ≤ now.hour < SESSION_END_UTC` |
| 2 | Dead zone: 12:00–13:00 UTC blocks ALL traders every day |
| 3 | Per-symbol cooldown (`COOLDOWN_MINUTES` since last trade) |
| 4 | Session trade cap (`trades_today[symbol] ≥ MAX_TRADES_PER_SESSION`) |
| 5 | Circuit breaker: `daily_loss > MAX_DAILY_LOSS_PCT` or `drawdown > MAX_DRAWDOWN_PCT` |
| 6 | Spread gate: XAUUSD > 50 pips or Forex > 3 pips → blocked |
| 7 | EV gate: `ev ≥ MIN_EV_THRESHOLD (0.10)` AND `expected_variance ≤ 0.80` AND `p_dir ≥ ML_DIRECTION_THRESHOLD` |
| 8 | RL gate: PPO approves trader + sets EV selectivity threshold |

**EV is mandatory when `ML_ENABLED=true`.** Missing `ev` key raises `RuntimeError`.

---

## Feature Contracts (fixed — changing breaks saved weights)

| List | Length | Used by |
|------|--------|---------|
| `SEQUENCE_FEATURES` | 53 | GRU training + inference |
| `REGIME_FEATURES` | 59 | RegimeClassifier |
| `QUALITY_FEATURES` | 17 | QualityScorer |
| `RL_STATE_DIM` | 42 | RLAgent |

---

## Offline Data Pipeline (`pipeline/`)

9-step pipeline for building training datasets, running backtests, and training ML models. Runs on Kaggle T4 × 2 GPUs.

| Step | Script | Output |
|---|---|---|
| 0 | `step0_resample.py` | `processed_data/histdata/{SYM}_{TF}.parquet` (5M/15M/1H/4H/1D) |
| 1 | `step1_inventory.py` | `processed_data/raw_inventory.json` |
| 2 | `step2_clean.py` | Clean parquets with macro columns |
| 3 | `step3_align.py` | `processed_data/aligned_multi_asset.parquet` |
| 4 | `step4_features.py` | 200+ engineered features |
| 5 | `step5_split.py` | `ml_training/datasets/train|val|test.parquet` (70/15/15) |
| 6 | `step6_backtest.py` | 3-round reinforcement backtest + warm-start retraining |
| 7 | `step7_train.py` | GRU + Regime weights |
| 7b | `step7b_train.py` | Quality + RL weights (from 8,203 journal trades) |
| 8 | `step8_push_to_github.py` | Push weights + metrics to GitHub (git init + soft-reset + push) |

**Run on Kaggle:** `python run_pipeline.py`

---

## Retrain Schedule

| Cadence | Models | Trigger | Min data |
|---|---|---|---|
| Weekly (Sun 02:00 UTC) | Quality + RL | Journal volume | 200 trades (Quality), 500 (RL) |
| Monthly (1st Sun 03:00 UTC) | Regime + GRU | Structural change | OHLCV only |

All retrains are warm-start — existing weights loaded and fine-tuned at 5× lower LR.

---

## Docker Services

| Container | Port | Purpose |
|---|---|---|
| trading_backend | 3000 | FastAPI |
| trading_frontend | 3001 | React/Nginx |
| trading_postgres | 5432 | |
| trading_redis | 6379 | pub/sub + state |
| trading_engine_main | 8000 (internal) | Trading engine |
| trading_model_retrainer | — | `retrain_scheduler.py` |

---

## Backtesting Framework

Batched GPU inference: `_precompute_ml_cache` builds all sequences upfront, runs through GPU in batches of 1024, stores results in a dict. The bar loop does dict lookups only — no per-bar inference.

**Execution order inside `_precompute_ml_cache`:**
1. RegimeClassifier batch inference → `regime_preds` + `_regime_series_for_gru`
2. Build sequence feature matrix with regime context injected at each timestep
3. GRU batch inference → `gru_preds`
4. Merge into `cache[bar_idx]`

**Latest results (2026-04-18, Jan 2021 – Aug 2024):**

| Metric | Round 1 | Round 3 |
|---|---|---|
| Trades | 2,712 | 2,769 |
| Win Rate | 45.5% | 44.9% |
| Profit Factor | 1.99 | 1.93 |
| Sharpe | 3.50 | 3.40 |
| Max Drawdown | 2.72% | 2.45% |
| Total Return | +20.1% | +18.0% |

---

## Causal Integrity

| Feature | Why zeroed | Where |
|---------|-----------|-------|
| `sr_dist_resist_atr` .. `sr_support_strength` (indices 28–33) | `detect_sr_zones` uses `rolling(center=True)` | `_build_feature_matrix` |
| `swing_hh_hl_count`, `liquidity_sweep_24h` (indices 6–7) | `detect_break_of_structure` uses `rolling(center=True)` | `_build_feature_matrix` |
| Macro `bfill()` | Pulled future macro data into early bars | Replaced with `fillna(0.0)` |

---

## Output Contracts

| Contract | Description |
|---|---|
| 1 | Redis event schemas (MARKET_DATA, SIGNAL_GENERATED, TRADE_EXECUTED) |
| 2 | Redis state keys (`engine:status`, `trader:{id}:state`, `strategy_allocations:{id}`) |
| 3 | Health endpoint: `{status, timestamp, traders, mode}` |
| 4 | Journal JSONL: all CSV fields + state_at_entry[42], rl_action, ml_model_scores, ev, expected_variance |
| 5 | Backtest JSON: `{run_at, start, end, config, results, trade_log}` |
| 6 | Trader IDs: `trader_1` through `trader_5` |
| 7 | Feature flags: `ML_ENABLED`, `PAPER_TRADING` from `.env` |
