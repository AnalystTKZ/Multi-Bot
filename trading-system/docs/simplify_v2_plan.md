# Simplify v2 MVP Plan

## Ownership and Boundaries

This document is the only artifact owned by this planning pass.

- Do not edit production code as part of this report.
- Other agents may be editing code in parallel; do not revert, clean, or normalize their changes.
- Build `simplify-v2` in parallel with the current system until metrics prove it is safe to promote.
- Keep backtest/live signal parity as a hard requirement: the simplified runner and the simplified backtest path must call the same feature and model inference contract.

Source brief: `trading-system/refactor_prompt.md`.

## Current Complexity to Replace

The current repo has these active complexity centers:

- Pipeline orchestration: `trading-system/pipeline/run_pipeline.py` plus step files `step0_resample.py` through `step8_validate.py`.
- Current model stack: `trading-system/trading-engine/models/` with `regime_classifier.py`, `gru_lstm_predictor.py`, `quality_scorer.py`, and `rl_agent.py`.
- Current inference and decision path: `trading-system/trading-engine/services/feature_engine.py`, `signal_pipeline.py`, `market_decision.py`, `risk_engine.py`, and `portfolio_manager.py`.
- Current backtest entrypoint: `trading-system/trading-engine/scripts/run_backtest.py`.
- Current generated data and artifacts: `trading-system/processed_data/`, `trading-system/ml_training/`, and `trading-system/trading-engine/weights/`.
- Infrastructure overhead: `trading-system/backend/`, `trading-system/frontend/`, `trading-system/docker-compose.dev.yml`, and `trading-system/monitoring/`.

The MVP does not delete any of the above. It creates a smaller path that can be compared against the existing path.

## MVP Scope

### Symbols, Timeframes, and Data

- Symbols: `XAUUSD`, `EURUSD`, `USDJPY`, `EURJPY`, `GBPJPY`, `GBPUSD`.
- Timeframes: `1H`, `4H`, `1D`.
- Data source: existing HistData-derived parquet files under `trading-system/processed_data/histdata/`.
- Backtest window: start with `2020-01-01` through `2024-12-31`, then widen only after the MVP passes parity and stability checks.

### Simplified Pipeline Shape

The simplified pipeline should be four stages:

1. `prep`
   - Implemented MVP path: `trading-system/pipeline_simplified/step1_data_prep.py`.
   - Reads selected symbol/timeframe parquet files from `trading-system/processed_data/histdata/`.
   - Produces cleaned MVP inputs under `trading-system/processed_data/simple/ohlcv/`.
   - Keeps only OHLCV, ATR, returns, spread/session fields if available, and minimal timeframe alignment metadata.

2. `features`
   - Implemented MVP path: `trading-system/pipeline_simplified/step2_features.py`.
   - Produces the current compatibility feature set under `trading-system/processed_data/simple/features.parquet`; the 40-50 feature budget is the next reduction target after parity is measurable.
   - Reuses or mirrors canonical math from `trading-system/trading-engine/services/feature_engine.py` and `trading-system/trading-engine/indicators/market_structure.py`.
   - Removes sentiment, FAISS similarity, separate quality score outputs, and intermediate model predictions from the feature set.

3. `train`
   - Implemented MVP path: `trading-system/pipeline_simplified/step4_train_unified.py`.
   - Trains one unified direction/regime model.
   - Optional second stage trains either a small threshold optimizer or a smaller RL policy only if threshold search is materially worse.
   - Writes weights and manifests under `trading-system/trading-engine/weights/unified_direction_regime/`.

4. `backtest`
   - Initial MVP path: existing `trading-system/trading-engine/scripts/run_backtest.py` can opt into the unified model with `SIMPLIFIED_ML_ENABLED=true`.
   - Calls the same simplified inference object used by live mode.
   - Reuses the existing backtest outputs while `SIMPLIFIED_ML_ENABLED=true` is set.

## Simplified Model Contract

### Model A: Unified Direction and Regime

Implemented MVP path: `trading-system/trading-engine/models/unified_direction_regime.py`.

Inputs:

- One aligned feature tensor per symbol.
- Target feature count: 40-50.
- Sequence window: start with the existing GRU/LSTM window convention from `trading-system/trading-engine/models/gru_lstm_predictor.py` unless a smaller value validates better.

Outputs:

- `p_bull`
- `p_bear`
- `p_flat` or `p_no_trade`
- `regime_code`: `BIAS_UP`, `BIAS_DOWN`, or `BIAS_NEUTRAL`
- Optional uncertainty/variance output only if it improves gating.

Replacement target:

- Merge current `RegimeClassifier HTF`, `RegimeClassifier LTF`, and `GRULSTMPredictor` into one forward pass.
- Keep manifest checks strict: the loaded model must expose expected feature names, sequence length, output names, training split, and timestamp.

### Model B: Trade Selectivity

Expected future path if threshold-based: `trading-system/pipeline_simplified/threshold_selectivity.py`.

Expected future path if RL remains necessary: `trading-system/trading-engine/models/selectivity_policy.py`.

Start with threshold search, not PPO:

- Inputs: `p_bull`, `p_bear`, model confidence, ATR, spread/session flags, recent volatility, and simple exposure state.
- Outputs: direction allow/deny, minimum probability threshold, and risk multiplier.
- Promote RL only if threshold search fails to reach at least 95% of the candidate performance with materially simpler code.

## Simplified Runner Contract

Implemented MVP path: `trading-system/pipeline_simplified/run_simple_pipeline.py`.

The runner should expose one command surface for all MVP stages:

```bash
cd /home/tybobo/Desktop/Multi-Bot/trading-system

python3 pipeline_simplified/run_simple_pipeline.py --list
python3 pipeline_simplified/run_simple_pipeline.py --force

SIMPLIFIED_ML_ENABLED=true \
BT_WINDOW=round1 \
python3 pipeline/step6_backtest.py
```

Expected smoke command:

```bash
cd /home/tybobo/Desktop/Multi-Bot/trading-system
SIMPLE_SYMBOLS=XAUUSD,EURUSD,USDJPY,EURJPY,GBPJPY,GBPUSD SIMPLE_START_DATE=2023-01-01 \
UNIFIED_EPOCHS=1 UNIFIED_BATCH_SIZE=128 \
python3 pipeline_simplified/run_simple_pipeline.py --force
```

Expected promotion command after MVP validation:

```bash
cd /home/tybobo/Desktop/Multi-Bot/trading-system
python3 pipeline_simplified/run_simple_pipeline.py --list
```

## Parallel-Build Milestones

### Milestone 1: Skeleton and Data Prep

- Add `trading-system/pipeline_simplified/` without changing the current 9-step runner.
- Implement `prep` and produce deterministic data manifests.
- Verify row counts, date ranges, missing values, and symbol/timeframe coverage.

Acceptance:

- `EURUSD`, `GBPUSD`, and `XAUUSD` produce aligned MVP input files.
- No current files under `trading-system/pipeline/` or `trading-system/trading-engine/` are required to change.

### Milestone 2: Feature Budget

- Implement `features` with a hard feature budget of 50.
- Reuse engine feature math where possible.
- Add a feature manifest with names, source columns, lookback requirements, and null policy.

Acceptance:

- Feature columns are stable across all three symbols.
- No model-output feature leakage.
- Representative feature values match the current engine formulas where the same concepts exist.

### Milestone 3: Unified Model

- Implement `unified_direction_regime.py`.
- Train a single model that replaces GRU direction plus HTF/LTF regime classifiers for MVP scope.
- Save a manifest next to the `.pt` file.

Acceptance:

- Model loads in a fresh Python process.
- Inference returns the full output contract for every candidate row.
- No fallback success based only on weight-file presence.

### Milestone 4: Selectivity Without PPO First

- Implement threshold optimization before any PPO work.
- Compare against current QualityScorer/RL gating behavior using candidate logs and backtest stats.

Acceptance:

- Threshold config is readable JSON.
- Backtest produces trade frequency in the target range before adding RL complexity.

### Milestone 5: Backtest/Live Parity Runner

- Implement `backtest` and a dry-run/live loop against the same simplified inference object.
- Journal to local files or SQLite under a simplified artifact namespace before promotion.

Acceptance:

- Backtest and dry-run live mode call the same inference and selectivity code.
- Candidate logs include model probabilities, regime code, threshold decision, and reason for rejection.

## What Stays Out of MVP

- FAISS/vector similarity.
- FinBERT/VADER sentiment.
- RabbitMQ, Redis, PostgreSQL, and monitoring stack.
- Frontend/dashboard changes.
- Docker and Kaggle orchestration.
- Full symbol universe.
- Separate QualityScorer unless a removal test shows unacceptable regression.
- PPO unless threshold optimization is proven insufficient.

## Validation Criteria

Minimum before considering promotion:

- Backtest/live parity: one shared inference path.
- Data purity: train, validation, and blind test windows documented in manifests.
- Model readiness: load checks prove actual model objects are available, not just files on disk.
- Trade frequency: 50-200 trades per symbol per year on MVP scope.
- Metrics: candidate Sharpe, profit factor, drawdown, and average R do not materially regress versus the current comparable baseline.
- Explainability: every skipped trade has a recorded gate reason.
- Maintainability: MVP runner and model stack remain small enough to understand without the current 9-step orchestration.

## Risks and Controls

- Risk: feature simplification silently changes trading math.
  - Control: compare shared feature values against `trading-system/trading-engine/services/feature_engine.py` and `trading-system/trading-engine/indicators/market_structure.py`.

- Risk: a simpler model looks good because of leakage.
  - Control: write split windows and source files into every manifest; preserve blind evaluation hygiene.

- Risk: threshold search overfits the backtest window.
  - Control: optimize on train/validation only, reserve blind test for final comparison.

- Risk: parallel agents touch current code while this plan is being used.
  - Control: keep all simplify-v2 implementation under `trading-system/pipeline_simplified/` and opt-in model flags until promotion; do not revert unrelated worktree changes.

## First Implementation Checklist

- Create `trading-system/pipeline_simplified/` plus the unified model under `trading-system/trading-engine/models/`.
- Implement `run_simplified.py` with subcommands: `prep`, `features`, `train`, `optimize-thresholds`, `backtest`, `compare`, and `all`.
- Implement deterministic manifests for every stage.
- Add smoke tests for runner argument parsing, manifest writing, model load checks, and backtest/inference path sharing.
- Run the one-symbol smoke command before any full MVP run.
