# System Architecture

Last updated: 2026-04-19.

## What This System Is

A multi-trader forex/gold system that trains on Kaggle (2× Tesla T4 GPUs) and runs in Docker locally. Five independent trading strategies share a single ML pipeline. The pipeline generates signals, the ML models evaluate and gate them, and the RL agent decides which strategy to run at what selectivity.

The design principle: **ICT/SMC conditions are features, not rules.** The ML models decide whether to trade; the ICT logic decides where to put the levels.

---

## Pipeline — Bar by Bar

```
Capital.com REST
  └─ DataFetcher.get_ohlcv(symbol, tf=15M, bars=300)
       └─ ProductionTradingEngine.on_market_data()
            └─ SignalPipeline.process_bar(symbol, df, df_htf)

                Step 1: ML Inference (_run_ml_inference)
                  ├─ RegimeClassifier  → regime, regime_id, proba[4], regime_confidence
                  ├─ GRU-LSTM          → p_bull, p_bear, expected_move, expected_variance
                  ├─ QualityScorer     → ev, quality_score
                  └─ SentimentModel    → sentiment_score, sentiment_label
                  Order is load-bearing: Regime feeds into GRU sequence features

                Step 2: 5 Traders run in sequence
                  └─ BaseTrader.analyze_market()
                       ├─ Guard 0: symbol exclusion
                       ├─ Guard 1: session window
                       ├─ Guard 2: dead zone (12:00–13:00 UTC)
                       ├─ Guard 3: cooldown
                       ├─ Guard 4: max trades/session
                       ├─ Guard 5: circuit breaker (daily loss / drawdown)
                       ├─ Guard 6: spread (XAUUSD > 50 pips | Forex > 3 pips)
                       ├─ _compute_signal() ← subclass ICT logic
                       ├─ Guard 7: EV gate (ev ≥ 0.10)
                       │           uncertainty gate (expected_variance ≤ 0.80)
                       │           direction gate (p_dir ≥ ML_DIRECTION_THRESHOLD)
                       └─ Guard 8: RL action gate (PPO approves trader + sets selectivity)

                Step 3: Ensemble scoring
                  score = (ict_score × 0.5 + ml_score × 0.5 + sentiment_bonus) × regime_mult
                  ├─ ml_score = gru_score × 0.5 + quality_score × 0.5
                  ├─ regime_mult: TRENDING ×1.2 (aligned) / ×0.9, RANGING ×0.9, VOLATILE ×0.85
                  ├─ filter: score ≥ 0.5
                  └─ confidence gate: ≥ 0.55

                Step 4: Signal published → Redis → WebSocket → Frontend
                Step 5: RiskEngine.check_pre_trade()
                Step 6: ExecutionEngine → PaperTradingService / BrokerConnector
                Step 7: TradeJournal.log_trade() → CSV + JSONL
```

---

## ML Pipeline in Depth

The four models run in a strict order. Each feeds the next.

### 1. RegimeClassifier

**What it does:** Labels the current market state as one of 4 regimes.

- Architecture: PyTorch MLP — `59 → 128 → 64 → 4` with BatchNorm, GELU, residual skip, Dropout 0.5
- Classes: `0=TRENDING_UP`, `1=TRENDING_DOWN`, `2=RANGING`, `3=VOLATILE`
- Output: `{"regime": str, "regime_id": int, "proba": [4 floats], "regime_confidence": float}`
- 3-bar hysteresis: regime must be predicted 3 consecutive bars before switching
- Latest accuracy: 4H → 47.7%, 1H → 39.5% (4-class balanced; random = 25%)

**Warm-start:** `_warm_start` flag checked before `_build_mlp()` — weights preserved if feature count matches. Trains at `lr=6e-5` (warm) vs `lr=3e-4` (cold).

**Labels:** Per-group GMMs on 4H data (dollar, cross, gold groups). Feature-axis constraints for stable cluster assignment across retrains.

**Minimum persistence filter:** Regime runs shorter than 20 bars (4H) / 48 bars (1H) are zero-weighted during training. This prevents label noise from short-lived regime flips that have no predictive value. Short runs are not relabelled — their confidence score is set to 0.0, causing the weighted trainer to ignore them.

---

### 2. GRU-LSTM Predictor

**What it does:** Given a 30-bar sequence of execution-TF features, predicts direction, magnitude, and uncertainty of the next move.

- Architecture: GRU(256, 2 layers, dropout=0.2) → LayerNorm → shared FC(128) → 3 heads
- Loss: `dir_loss + 0.5×mag_loss + 0.3×vol_loss`
- Latest: 7,081,756 samples, 44 symbol/tf combos, early stop epoch 6 (train=0.5943, val=0.6184)
- DataParallel across both Kaggle T4s

**Regime conditioning:** `prev_regime_0..3` (one-hot) and `regime_confidence` injected at every timestep in the sequence. Regime is computed before GRU — order in `_precompute_ml_cache` is load-bearing.

**Catastrophic forgetting:** GRU is **excluded from `step6_backtest.py`'s per-round retrain loop**. Fine-tuning on ~3k backtest journal trades against 7M training sequences destroys the model. GRU retrains monthly from full OHLCV only.

---

### 3. QualityScorer (EV Regressor)

**What it does:** Predicts expected value of a specific trade setup in R-multiples.

- Architecture: PyTorch MLP — `17 → 64 → 32 → 1` with BatchNorm, GELU, identity output (unbounded)
- Loss: class-weighted Huber — winners upweighted ~4.1× (corrects ~20%/80% positive/negative imbalance)
- Output: `{"ev": float, "quality_score": float}` where `quality_score = sigmoid(ev)`
- Latest: 5,264 journal entries, dir_acc=0.615, MAE=0.933

**EV Label Tiers:**

| Exit reason | EV label | Meaning |
|---|---|---|
| `tp2` | `clip(rr_ratio, 0.1, 10.0)` | Full TP — best outcome |
| `tp1` | `clip(rr_ratio × 0.75, 0.1, 10.0)` | TP1 hit, held further |
| `be_or_trail` | `clip(rr_ratio × 0.4, 0.1, 10.0)` | TP1 hit, trailed out |
| `sl_*` | `-1.0` | Loss |
| `time_exit` | `None` (skipped) | Ambiguous — excluded |

Model learns a gradient of outcomes: tp2 > tp1 > be_or_trail > sl. Exit reasons preserved raw in the journal — never collapsed to `tp/sl`.

**EV Gate:** `ev ≥ 0.10` (Guard 7 in BaseTrader). Missing `ev` key raises `RuntimeError` when `ML_ENABLED=true`.

---

### 4. RLAgent (PPO Strategy Selector)

**What it does:** Decides WHICH trader to approve and at what EV selectivity level.

- Algorithm: PPO via stable-baselines3, **always runs on CPU** (MLP policy faster on CPU per SB3 guidance)
- State: 42-dim float32 vector
- Actions: 16 (0=NoTrade, 1–5 trader approval at EV 0.55, 6–10 at 0.65, 11–15 at 0.75)
- Weights: `weights/rl_ppo/model.zip` (explicit `.zip` extension prevents SB3 directory creation bug)

**Warm-start:** `set_env()` + `learning_rate = 3e-4 / 5.0` + `reset_num_timesteps=False`. Policy diversity requires `ent_coef=0.01` and ≥500 journal trades.

---

## Backtest Architecture

**Batched GPU inference:** `_precompute_ml_cache` builds all sequences upfront and runs them through the GPU in batches of 1024 before the bar loop. The bar loop is pure Python dict lookup + ICT evaluation.

**Execution order inside `_precompute_ml_cache`:**
1. `RegimeClassifier._build_feature_matrix()` → `regime_preds` + `_regime_series_for_gru`
2. `_build_sequence_df()` with regime context injected at each timestep
3. GRU batch inference → `gru_preds`
4. Merge into `cache[bar_idx]`

**Reinforcement rounds:** `step6_backtest.py` runs 3 rounds. After each round, models (`regime`, `quality`, `rl`) warm-start retrain on the trade journal. GRU is excluded.

**Latest results (2026-04-18, Jan 2021 – Aug 2024, $10k capital):**

| Round | Trades | WR | PF | Sharpe | MaxDD | Return |
|---|---|---|---|---|---|---|
| 1 | 2,571 | 45.4% | 2.08 | 3.77 | 2.8% | 1,549% |
| 2 | 2,610 | 44.8% | 2.04 | 3.70 | 3.6% | 1,458% |
| 3 | 2,586 | 45.3% | 2.05 | 3.71 | 2.8% | 1,491% |

All 4 regime classes active. EV distribution: min=0.35, mean=1,444, zero-EV count=0.

---

## Kaggle Training Setup

The system uses a **split-dataset** approach on Kaggle:

| Datasource | Kaggle path | Contents |
|---|---|---|
| Code (GitHub clone) | `/kaggle/input/datasets/tysonsiwela/multi-bot-system` | All Python code |
| Data | `/kaggle/input/datasets/tysonsiwela/trading-data` | `training_data/` + `processed_data/` (read-only) |
| Push clone | `/kaggle/working/remote/Multi-Bot` | Fresh git clone for pushing weights back to GitHub |

`env_config.py` resolves all paths automatically for both Kaggle and local environments.

`step8_push_to_github.py` targets `/kaggle/working/remote/Multi-Bot` (fresh clone made by notebook cell 0), pulls latest with `--ff-only`, copies artifacts, commits, and pushes. No `--force`.

---

## The 5 Trading Strategies

| Trader | Session | Regime Req | Key Signal |
|---|---|---|---|
| T1 NY EMA | 13–17 UTC | TRENDING required | 1H EMA stack + ADX > 20 + pullback |
| T2 FVG BOS | London + NY | Any | 4H BOS + open FVG entry |
| T3 London BO | 07–10 UTC | TRENDING only | Asian sweep + volume > SMA×1.3 |
| T4 News | Any | Any | sentiment ≥ 0.65 + structural BO |
| T5 Asian MR | 02–06:45 UTC | RANGING required | Range extreme + dual oscillator |

**Dead zone:** 12:00–13:00 UTC — all traders blocked.

---

## Feature Contracts

Feature list order is a hard contract — changing order or count breaks saved model weights.

| List | Length | Used by |
|------|--------|---------|
| `SEQUENCE_FEATURES` | 53 | GRU |
| `REGIME_FEATURES` | 59 | RegimeClassifier |
| `QUALITY_FEATURES` | 17 | QualityScorer |
| `RL_STATE_DIM` | 42 | RLAgent |

---

## Causal Integrity

All features are strictly backward-looking. The following were identified as lookahead and zeroed:

| Feature | Why zeroed |
|---------|-----------|
| `sr_dist_resist_atr` .. `sr_support_strength` (indices 28–33) | `detect_sr_zones` uses `rolling(center=True)` |
| `swing_hh_hl_count`, `liquidity_sweep_24h` (indices 6–7) | `detect_break_of_structure` uses `rolling(center=True)` |
| Macro `bfill()` | Pulled future macro data backward into early bars → replaced with `fillna(0.0)` |

---

## Retrain Schedule

| Cadence | Models | Trigger | Min journal entries |
|---|---|---|---|
| Weekly (Sun 02:00 UTC) | Quality + RL | Journal volume | 200 (Quality), 500 (RL) |
| Monthly (1st Sun 03:00 UTC) | Regime + GRU | Structural | OHLCV only |

All retrains warm-start from existing weights at 5× lower LR. GRU excluded from per-round reinforcement loop.

---

## Error Policy

No silent fallbacks anywhere. Every model failure raises and propagates.

- Untrained models: raise `ModelNotTrainedError(RuntimeError)`
- Missing `ev` key when `ML_ENABLED=true`: raise `RuntimeError`
- `_precompute_ml_cache` failures: raise (no fallback to empty cache)
- Feature computation failures: raise (no return of zero-filled arrays)

---

## Training (Kaggle)

```python
# Kaggle notebook — runs full pipeline
python kaggle_train.py

# Individual steps
python pipeline/step7_train.py    # GRU + Regime (GPU intensive)
python pipeline/step7b_train.py   # Quality + RL
python step8_push_to_github.py    # push weights to GitHub via fresh clone
```

**Weight hot-reload:** `BaseModel.reload_if_updated()` checks mtime every 5 min — no engine restart needed.
