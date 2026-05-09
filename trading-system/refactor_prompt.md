The system has become unecessarily complex

# Multi-Bot Trading System — Complexity Reduction & ML Consolidation

## Current State Analysis
The system has evolved into a complex, multi-layered architecture with:
- 6 separate ML models (RegimeClassifier HTF/LTF, GRU-LSTM, QualityScorer, SentimentModel, RLAgent)
- 9-step data pipeline with warm-start retraining
- Distributed infrastructure (FastAPI, PostgreSQL, Redis, RabbitMQ)
- Multiple data sources and feature engineering (200+ features)
- Docker containerization, Kaggle integration
- FAISS vector store, backtesting harness, monitoring stack

**Problem:** Over-engineering creates maintenance burden, slow iteration, and diminishing returns on added complexity.

---

## SIMPLIFICATION OBJECTIVES

Keep the system as a **pure ML-driven trading engine** while:
1. **Reduce model count** from 6 to 2–3 unified models
2. **Consolidate infrastructure** into single cohesive unit (not micro-services)
3. **Streamline data pipeline** from 9 steps to 3–4 core steps
4. **Eliminate non-essential components** (FAISS similarity, QualityScorer if redundant, sentiment if not impactful)
5. **Keep backtest/live parity** — signal generation stays identical between backtest and live
6. **Target: 50% fewer lines of code, same or better accuracy**

---

## PHASE 1: MODEL CONSOLIDATION

### Current: 6 Models in Sequence
- RegimeClassifier HTF (3-class macro bias)
- RegimeClassifier LTF (4-class behaviour)
- GRU-LSTM (p_bull, p_bear, variance)
- QualityScorer (EV in R-multiples)
- SentimentModel (FinBERT + VADER)
- RLAgent (PPO, 43-dim state, 16 actions)

### Simplified: 2 Unified Models

**Model A: Unified Direction + Regime Predictor**
- Single PyTorch model: `78 features → [LSTM(64,2L) + MLP] → p_bull, p_bear, regime_code`
- Combines: GRU-LSTM direction prediction + Regime classification into one forward pass
- Output: probability vector + 3-class regime (BIAS_UP/DOWN/NEUTRAL)
- Benefit: Single inference call, shared hidden representations, easier to optimize

**Model B: Trade Selectivity (RL or Threshold)**
- Option 1: Simplified PPO agent with 43-dim state → 8 actions (selectivity thresholds)
- Option 2: Replace RL with Bayesian threshold optimizer (no RL training complexity)
- Keeps: probability threshold selection, position sizing
- Removes: 16 actions, complex state encoding

**Model C (OPTIONAL - Remove First):**
- QualityScorer: Assess if essential. If backtesting shows it's not improving signal quality significantly, replace with simple p_win-based E[R] gate (formula, no model)
- SentimentModel: Pre-trained FinBERT is overhead. If macro bias isn't critical, remove. If keeping, use off-the-shelf sentiment API (e.g., NewsAPI sentiment endpoint) rather than running your own model.

---

## PHASE 2: DATA PIPELINE STREAMLINE

### Current: 9 Steps


### Simplified: 4 Core Steps

step0_resample → step1_inventory → step2_clean → step3_align →
step4_features → step5_split → step6_backtest → step7_train → step8_push

STEP A: Data Prep
Input: Raw OHLCV (histdata ASCII + macro feeds)
Output: {SYMBOL}_{TF}.parquet (4 timeframes: 5M, 1H, 4H, 1D)
Action: Resample, merge, drop NaNs, compute ATR + session info ONLY

STEP B: Feature Engineering (Selective)
Input: Resampled OHLCV
Output: {SYMBOL}_features.parquet (only 40–50 features, not 200+)
Include:
- Price action: Close, HL range, ATR, EMA short/long
- Structure: BOS_age, FVG_dist (simplified detection, not full ICT)
- Regime: HTF volatility regime (simple percentile-based, not classifier output)
- Volume: VWAP, volume-weighted features
Remove:
- All intermediate ML classifier outputs (régime, sentiment)
- Redundant features
- Institutional structure features if not correlated

STEP C: Train/Backtest
Input: Features + split metadata
Action:
- Train Unified Direction Model (LSTM-based) on GRU-prepared dataset
- Train RL Agent (or threshold optimizer) on backtest results
- Run backtest with warm-start retraining (keep existing logic, but single-pass inference)
Output: model.pt, rl_agent.zip, backtest_stats.json

STEP D: Push + Deploy
Input: Trained weights
Action: Push to GitHub, deploy to live trading engine
Output: Live weights ready for inference


---

## PHASE 3: BACKEND CONSOLIDATION

### Current: Multi-Service
- FastAPI backend (routes, ICT, ml_filter, portfolio, services)
- PostgreSQL (trade logs, metadata)
- Redis (market data cache, session state)
- RabbitMQ (async tasks)

### Simplified: Single Python App
- **SingleThreadedTradingEngine.py** (replace FastAPI complexity)
  - Event-driven loop: fetch bar → preprocess → infer → gate → trade → log
  - All trade logic, inference, gating in one file or simple module structure
  - No REST API needed initially (direct integration with broker)
  - Logging to file (not database) for backtest compatibility
- **Eliminate:** PostgreSQL, Redis, RabbitMQ, unnecessary async (use simple threading if needed)
- **Keep:** SQLite for lightweight journaling (live + backtest use same schema)

---

## PHASE 4: INFRASTRUCTURE SIMPLIFICATION

### Current Overhead
- Docker (backend, frontend, trading-engine containers)
- Kaggle integration (GPU training, dataset management)
- GitHub Actions / CI/CD complexity
- Monitoring stack (Prometheus, AlertManager)

### Simplified
- **Local execution:** Run pipeline + backtest on laptop (don't force Kaggle if data is small enough)
- **Optionally GPU:** If data is large, use Kaggle OR Colab — not both simultaneously
- **No containerization initially:** Run as native Python + optional Flask/FastAPI later if needed for REST API
- **Single GitHub branch:** Keep main clean; feature branches for experiments only

---

## PHASE 5: FEATURE SCOPE REDUCTION

### Current Symbols & Timeframes
- 11 symbols (EURUSD, GBPUSD, USDJPY, XAUUSD, others)
- 5 timeframes (M1 raw, 5M, 15M, 1H, 4H, 1D)

### Simplified Scope (Proof of Concept)
- **Phase 1 (MVP):** 3 symbols (EURUSD, GBPUSD, XAUUSD)
- **Timeframes:** 1H, 4H, 1D (3 only, not 6)
- **Backtest window:** 2020–2024 (no need for 2016 if data is abundant 2020+)
- **Data source:** Single vendor (e.g., histdata ASCII or OHLCV CSV)

Scale horizontally AFTER proving concept.

---

## WHAT TO REMOVE (RANK ORDER)

### MUST REMOVE (Week 1)
1. **FAISS Vector Store** — No evidence it improves trading. Historical trade similarity is nice-to-have, not core.
2. **SentimentModel (FinBERT + VADER)** — High maintenance. Replace with single sentiment API call if macro bias is important.
3. **Distributed queue (RabbitMQ)** — Over-engineered for single trading loop. Use simple file-based job queue or drop entirely.
4. **PostgreSQL schema** — Replace with SQLite or flat JSON trade logs + CSV for analytics.

### SHOULD REMOVE (Week 2)
1. **QualityScorer model** — Test if removing it (use only p_win-based E[R] formula) changes accuracy. If <1% regression, delete.
2. **Separate RegimeClassifier models** — Merge into unified direction model.
3. **Warm-start retraining complexity** — Simplify to monthly full retraining, not incremental per round.
4. **Docker + Kaggle complexity** — Run locally during dev. Deploy to cloud only when mature.

### COULD REMOVE (Week 3+)
1. **RL Agent** — If Bayesian threshold search achieves 95% of RL performance with 10% of code, switch.
2. **Multi-timeframe hierarchy** — Try single 1H features + 4H bias only (not cascading regime logic).
3. **Monitoring stack** — Use simple logging. Add observability after trading is stable.

---

## REFACTORING SEQUENCE

### Week 1: Data & Features
1. Merge step0–step2 into single data prep script
2. Cut features from 200 to 50 (keep only correlated ones)
3. Verify backtest accuracy unchanged

### Week 2: Model Simplification
1. Train unified LSTM-based direction + regime model
2. Remove separate RegimeClassifier files and inference calls
3. Update backtest signal generation to use new model (should be faster, same accuracy)

### Week 3: Backend
1. Replace FastAPI + Redis + RabbitMQ with single event loop (threading-based if needed)
2. Use SQLite for journaling
3. Delete entire distributed infrastructure

### Week 4: Infrastructure
1. Move from Kaggle to local GPU (or Google Colab if needed)
2. Remove Docker from the equation (run native Python)
3. Finalize deployment script for live trading (single Python runner)

### Week 5: Proof & Scale
1. Backtest on full dataset, confirm metrics haven't regressed
2. Run live trading for 1–2 weeks on 3-symbol portfolio
3. Then scale horizontally (more symbols, more timeframes)

---

## SUCCESS CRITERIA

After simplification:
- [ ] **Codebase:** <5K lines of trading logic (vs. current ~15K+)
- [ ] **Models:** 2–3 files (vs. 6+)
- [ ] **Pipeline steps:** 4 (vs. 9)
- [ ] **Data infrastructure:** Single vendor, 3 symbols, 3 timeframes (MVP)
- [ ] **Backtest Sharpe:** ≥4.5 (same or better than current)
- [ ] **Trade frequency:** 50–200 trades/symbol/year (sustainable)
- [ ] **Deployment:** Single Python script (no Docker, no Kubernetes, no microservices)
- [ ] **Development cycle:** Feature → backtest → live in <1 hour (vs. current multi-hour Kaggle cycle)

---

## IMPLEMENTATION NOTES

- **Do NOT** delete current code yet; branch to `simplify-v2` and parallel-build
- **Preserve:** Backtesting methodology (walk-forward, warm-start if impactful), signal parity between backtest/live
- **Use existing:** GRU-LSTM base as foundation for unified model; adapt, don't rewrite from scratch
- **Test incrementally:** After each phase, run backtest and confirm metrics
- **Document:** Decisions to remove code so you can reinstate if needed