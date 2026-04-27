# Audit Remediation Log - 2026-04-27

## Scope

This log records the follow-up work after reviewing:

- `trading-system/AUDIT_REPORT_2026-04-27.md`
- `trading-system/output.md`
- `trading-system/trading-engine/logs`
- `trading-system/trading-engine/backtest_results`
- `trading-system/trading-engine/weights`
- `trading-system/trading-engine/monitors/portfolio_manager.py`

## What We Found

The latest run was not the high-trade-count run cited in the audit report. The current run in `output.md` was:

| Round | Trades | Win rate | Profit factor | Sharpe | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Round 1 | 22 | 13.64% | 0.5211 | -3.9434 | Hit production risk halt early. |
| Round 2 | 27 | 22.22% | 1.1590 | 0.8700 | Hit production risk halt early. |
| Round 3 | 66 | 30.30% | 1.2137 | 1.2291 | Hit production risk halt early. |

The old `3,991 / 3,109 / 7,127` trade files were stale artifacts. They still existed locally, had 1970 timestamps, disabled production risk controls, and should not be used for promotion decisions.

Current logs no longer showed the 1970 timestamp issue. Journal rows had valid 2021-2025 timestamps and split provenance fields, but run-level provenance was still incomplete across backtests, journals, calibration, diagnostics, retrain history, and weights.

The early halts were caused by repeated `sl_full` losses that triggered daily halt events and then the 8% portfolio drawdown halt. The deeper issue was clustered entries around the same timestamps, especially London-session batches across correlated symbols.

`portfolio_manager.py` was being used by both live execution and backtests, but the backtest path passed `open_positions_detail: []`, so the PM correlation cap could not see same-timestamp or still-open simulated positions.

QualityScorer was not cleanly restored. Current live weights were missing, and older Quality/RL backups were contaminated by validation/test journal feedback. Current journals contained only validation/test/combined-eval rows, which correctly blocked clean production retraining but left no usable train-only Quality artifact.

## What Was Changed

### Portfolio Manager

Updated `trading-system/trading-engine/monitors/portfolio_manager.py`.

- Added `PortfolioManager.directional_group()` as a public audit helper.
- This exposes the same currency-direction grouping used by the PM correlation cap.

### Backtest PM Exposure

Updated `trading-system/trading-engine/scripts/run_backtest.py`.

- Added a simulated open/pending position ledger.
- Before each bar, closed positions are pruned by simulated `exit_timestamp`.
- Active pending/open positions are now passed into `portfolio_state["open_positions_detail"]`.
- This lets `PortfolioManager.enrich_signal()` apply correlation limits to same-timestamp and still-open simulated trades.
- Added `pm_open_positions_seen` to trade logs and diagnostic breakdowns.

### Artifact Provenance

Updated `run_backtest.py`, `step6_backtest.py`, `trade_journal.py`, `candidate_logger.py`, and `retrain_incremental.py`.

- Added `run_id` propagation.
- Added split hash, code commit, journal hash, and weight artifact hashes where available.
- Added promotion manifest output:
  - `logs/promotion_manifest_<run_id>.json`
  - `logs/promotion_manifest_latest.json`
- Added run/split fields to candidate logs and journal records.
- Updated retrain history records to include:
  - `run_id`
  - `retrain_data_split`
  - `split_summary_hash`
  - `journal_sha256`
  - `artifact_hashes`
  - `code_commit`

### Diagnostic Breakdowns

Updated `run_backtest.py`.

Future backtests now write:

- `logs/backtest_trade_breakdown_<run_id>.csv`
- `logs/backtest_trade_breakdown_latest.csv`
- `logs/backtest_diagnostic_breakdown_<run_id>.csv`
- `logs/backtest_diagnostic_breakdown_latest.csv`
- `logs/backtest_diagnostic_breakdown_<run_id>.json`
- `logs/backtest_diagnostic_breakdown_latest.json`

Breakdowns include:

- symbol
- side
- regime
- confidence bin
- hour/session
- `p_bull` / `p_bear` bins
- ADX bin
- ATR/entry bin
- EMA stack
- directional correlation group
- same-timestamp entry count
- same-timestamp same-group count
- PM open positions seen before acceptance
- exit reason and PnL

### QualityScorer Cleanup

Updated `trading-system/trading-engine/models/quality_scorer.py` and `trading-system/pipeline/step7_train.py`.

- QualityScorer saved payload now records training metadata.
- Synthetic fallback journal rows are explicitly marked `source_split=train`.
- Production QualityScorer training remains restricted to train/live/paper/production rows unless explicitly overridden.
- Existing stale QualityScorer and RL artifacts were deleted.

## Artifact Reset

Generated artifacts were deleted from:

- `trading-system/trading-engine/backtest_results`
- `trading-system/trading-engine/backtest_results/ml_cache`
- `trading-system/trading-engine/logs`
- `trading-system/trading-engine/weights`
- `trading-system/ml_training/metrics`
- `trading-system/backtesting/results`
- `trading-system/backtesting/logs`

The directories were kept so future pipeline runs can recreate outputs cleanly.

## High-Trade Reset Update

After the fresh `output.md` run still halted early, the research defaults were loosened for the next rebuild:

- Daily-loss and portfolio-drawdown halt defaults are now `100%`.
- Backtest step 6 now defaults to high-trade research mode with `BACKTEST_ENFORCE_DAILY_HALT=0`.
- Max concurrent positions and same-direction correlation caps were raised to `25`.
- ML direction, neutral-bias, volatile-entry, and PM confidence floors were lowered to `0.50`.
- Per-symbol backtest cooldown now defaults to `1` bar.

Generated artifacts were cleared again, including:

- `trading-engine/weights`
- `trading-engine/backtest_results`
- `trading-engine/logs`
- `ml_training/metrics`

This leaves the repository ready for a fresh, unconstrained weight rebuild and new high-trade-count backtests.

## Verification

Syntax checks passed with:

```bash
python3 -m py_compile \
  trading-system/trading-engine/monitors/portfolio_manager.py \
  trading-system/trading-engine/scripts/run_backtest.py \
  trading-system/trading-engine/services/candidate_logger.py \
  trading-system/trading-engine/services/trade_journal.py \
  trading-system/pipeline/step6_backtest.py \
  trading-system/pipeline/step7_train.py \
  trading-system/trading-engine/scripts/retrain_incremental.py \
  trading-system/trading-engine/models/quality_scorer.py
```

Artifact directories were checked after reset and contained no files.

## Current State

The repository is intentionally artifact-clean:

- No backtest JSONs.
- No ML cache files.
- No trading-engine logs.
- No live or backup weights.
- No stale ML metric JSONs.

This means the next pipeline run must rebuild weights and generate fresh backtests before any further performance analysis.

## Next Required Run Order

1. Regenerate or verify `split_summary.json`.
2. Train GRU and regime models on the train split.
3. Generate train-only journal labels for QualityScorer.
4. Train QualityScorer from train-only rows.
5. Keep RL disabled unless clean train-only episodes exist.
6. Run Round 1 validation with production risk enabled.
7. Run Round 2 blind test with production risk enabled.
8. Review the new promotion manifest and diagnostic breakdowns before tuning thresholds.

## Remaining Risks

- No fresh model weights exist yet after reset.
- QualityScorer still needs clean train-only labels before it can be active.
- Backtest equity/PnL accounting still computes trade outcomes immediately, while the new ledger only models exposure for PM correlation gating.
- The next full run is required to confirm whether PM correlation caps materially reduce clustered losses.

## Profitability Remediation Update

After reviewing the new run output, the main problem shifted from trade count to profitability quality:

- Candidate logging now happens inside the backtest decision loop, so rejected opportunities are logged with `rejection_reason`, `pm_open_positions_seen`, and a simulated hypothetical outcome.
- Calibration, EV filtering, and statistical binning now ignore rejected hypothetical rows and train only from `executed=1` outcomes.
- Backtest profitability reporting now uses fixed-risk, non-compounded R-multiple metrics as the primary PF/return/drawdown/Sharpe basis.
- PortfolioManager now enforces the total `MAX_CONCURRENT_POSITIONS` cap before correlation checks, which addresses `pm_open_positions_seen` exceeding the configured cap.
- HTF/LTF market-decision logic is shared between backtest and live signal generation.
- `BIAS_NEUTRAL` and `RANGING` labels are now explicit clean market states instead of fallback buckets, with ambiguous bars assigned zero confidence for dropping during retraining.

Verification:

```bash
python3 -m py_compile \
  trading-system/trading-engine/services/candidate_logger.py \
  trading-system/trading-engine/services/quant_analytics.py \
  trading-system/trading-engine/monitors/portfolio_manager.py \
  trading-system/trading-engine/services/market_decision.py \
  trading-system/trading-engine/scripts/run_backtest.py \
  trading-system/trading-engine/services/signal_pipeline.py \
  trading-system/trading-engine/models/regime_classifier.py \
  trading-system/pipeline/step6_backtest.py
```

No full backtest was run after these code changes; the next pipeline run is required to measure the new fixed-risk profitability and candidate rejection distribution.

## Round Journal Quality/RL Training Update

QualityScorer and RLAgent now accept round-produced journal splits by default:

- `validation`
- `test`
- `combined_eval`

The Kaggle pipeline now retrains Quality/RL after Round 1, after Round 2, and after Round 3 using the accumulated journal entries. GRU and Regime remain train-split retrains in the Round 3 incremental phase.

The allowed split list is still overrideable with `JOURNAL_ALLOWED_SPLITS`. Set `ALLOW_ROUND_JOURNAL_TRAINING=0` and provide a stricter split list if a production-only train/live/paper journal retrain is required.

Verification:

```bash
python3 -m py_compile \
  trading-system/kaggle_train.py \
  trading-system/pipeline/step7b_train.py \
  trading-system/trading-engine/models/quality_scorer.py \
  trading-system/trading-engine/models/rl_agent.py \
  trading-system/trading-engine/scripts/retrain_incremental.py
```
