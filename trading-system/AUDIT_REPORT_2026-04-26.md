# Trading System Audit Report - 2026-04-26

Scope:
- `output.md`
- `trading-engine/logs`
- `trading-engine/backtest_results`
- `trading-engine/weights`
- `trading-engine/weights/backups`

## Executive Verdict

The run completed and the core backtest arithmetic reconciles, but the latest output is not valid performance evidence yet.

Critical blockers found:
- Trade timestamps in saved backtest/journal artifacts collapse to `1970-01-20/21`.
- Calibration was cumulative across rounds instead of per backtest.
- RL PPO weights existed but were not loaded because `is_trained` ignored `model.zip`.
- QualityScorer trains successfully now, but final calibration remains unreliable.

## Critical Findings

### 1. Trade Timestamp Corruption

Declared backtest windows:
- Round 1: `2021-08-05` to `2023-08-04`
- Round 2: `2023-08-07` to `2025-08-05`
- Round 3: `2022-08-05` to `2025-08-05`

Actual saved trade timestamps:
- Round 1: `1970-01-20T07:00:00+00:00` to `1970-01-20T13:45:02.700000+00:00`
- Round 2: `1970-01-20T13:49:26.400000+00:00` to `1970-01-21T07:19:06.600000+00:00`
- Round 3: `1970-01-20T07:00:00+00:00` to `1970-01-21T07:19:03+00:00`

Impact:
- Trades/day diagnostics are invalid.
- Daily loss halt diagnostics are invalid.
- Session-hour conclusions are suspect.
- Annualized Sharpe interpretation is not reliable.

Fix applied:
- `run_backtest.py` now detects numeric epoch seconds, milliseconds, microseconds, and nanoseconds before coercing timestamps.

### 2. Calibration Leakage Across Rounds

The latest backtest has `5830` trades, but calibration bins total `11657`, which equals:

`3296 + 2531 + 5830`

That means calibration was accumulated across all three rounds, not computed only for the current backtest.

Impact:
- Per-round calibration reports were contaminated.
- Latest `calibration_report.json` is global/cumulative, not a clean Round 3 audit.

Fix applied:
- `run_backtest.py` now resets `logs/backtest_candidate_log.csv` at the start of each backtest run.
- Candidate log rows now include more signal evidence: `p_bull`, `p_bear`, `ev`, `regime`, `adx`, and `atr`.

### 3. RL PPO Artifact Not Loaded

Artifact exists:
- `trading-engine/weights/rl_ppo/model.zip`

But `RLAgent.is_trained` only checked:
- `policy.pkl`
- `model`

It did not check:
- `model.zip`

Impact:
- PPO retraining always cold-started.
- Existing RL policy would not be used by `select_action()`.

Fix applied:
- `rl_agent.py` now treats `model.zip` as a trained artifact.

## High Findings

### 4. QualityScorer Is Trained But Not Calibrated Enough

Final QualityScorer training:
- Samples: `5827`
- MAE: `1.137`
- Direction accuracy: `0.607`

Final calibration:
- `monotonic=false`
- `reliable=false`
- `2/5` calibration bin pairs violated.

Impact:
- EV filtering is active, but EV ranking is still weak.
- Confidence cannot yet be used as a clean probability estimate.

Fix applied:
- Calibration reporting now separates strict monotonicity from loose reliability.
- It no longer marks a report as `monotonic=true` when there are calibration violations.

### 5. Reported Returns Are Arithmetically Consistent But Aggressive

Latest Round 3:
- Trades: `5830`
- PnL sum: `446350.94`
- Initial capital: `10000`
- Total return: `44.6351`
- PF: `2.9753`
- Max DD: `0.0701`

The math reconciles, but because timestamps are broken in current artifacts, time-normalized metrics should not be trusted until rerun.

## Medium Findings

### 6. NZDUSD Is Anomalously Sparse

Latest Round 3 trades by symbol:
- Most symbols: `485` to `646` trades
- NZDUSD: `20` trades

Likely causes:
- Missing or shorter data coverage.
- Symbol-specific model gating.
- Cache/data alignment issue.

Needs follow-up after timestamp fix rerun.

### 7. Candidate Log Was Not Sufficiently Auditable

Before the patch, `backtest_candidate_log.csv` left most signal fields blank.

Impact:
- Could not reconstruct why trades were accepted/rejected.

Fix applied:
- Backtest candidate rows now include core ML and market fields.

### 8. Dependency Warnings

Repeated warnings:
- FAISS missing for vector indexing.
- Gym is unmaintained and should be migrated to Gymnasium.
- XLA/TensorFlow factory warnings during RL startup.

These are not the main blocker, but they should be cleaned up.

## Artifact Compatibility Summary

Structurally compatible:
- `weights/gru_lstm/model.pt`
- `weights/gru_lstm/weights_manifest.json`
- `weights/regime_htf.pkl`
- `weights/regime_ltf.pkl`
- `weights/quality_scorer.pkl`

Risk:
- `weights/rl_ppo/model.zip` is valid, but runtime versions differ between the artifact environment and local venv.

## Rerun Expectations

After rerun, check:
- Trade timestamps should be in the declared backtest windows, not 1970.
- Active-day counts should span hundreds of days, not 1-2 days.
- Calibration bin counts should approximately match the current round's trade count.
- RL should load/warm-start from `model.zip`.
- Candidate log should contain populated ML/market fields.
- QualityScorer should produce nonzero `quality_block`.

