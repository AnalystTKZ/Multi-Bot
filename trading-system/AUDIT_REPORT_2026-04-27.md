# Trading System Audit Report - 2026-04-27

## Executive Verdict

Status: **not production ready**.

The new run is materially better than the earlier zero-trade failures: Round 1, Round 2, and Round 3 now all generate active `ml_trader` trades, and the train-only retraining guard appears to preserve the intended split boundaries. However, the current results are not production-valid because several artifacts and code paths still break auditability, causality, risk validation, and model provenance.

The most important point: **the splits must remain intact**. The correct remediation is not to redefine or merge the splits. The correct remediation is to repair timestamps, causal feature generation, metadata, journaling, calibration, and production-mode evaluation while preserving the existing train, validation, and blind-test boundaries.

## Scope Reviewed

Primary artifact reviewed:

- `trading-system/output.md`

Supporting paths reviewed by the main audit and read-only agents:

- `trading-system/trading-engine/backtest_results`
- `trading-system/trading-engine/logs`
- `trading-system/trading-engine/weights`
- `trading-system/trading-engine/weights/backups`
- `trading-system/ml_training/datasets`
- `trading-system/ml_training/metrics`
- `trading-system/processed_data`
- `trading-system/pipeline`
- `trading-system/trading-engine/scripts`
- `trading-system/trading-engine/models`
- `trading-system/trading-engine/indicators`
- `trading-system/trading-engine/services`

## Split Assessment

The latest `output.md` reports the intended chronological split:

| Segment | Role | Date range |
| --- | --- | --- |
| Train | Training only | 2016-01-04 to 2021-08-05 |
| Round 1 | Validation / calibration check | 2021-08-05 to 2023-08-04 |
| Round 2 | Blind test | 2023-08-07 to 2025-08-05 |
| Round 3 | Combined post-retrain evaluation window | 2022-08-05 to 2025-08-05 |

The run logs show `Retrain data split: train`, which is the right direction. There is no evidence in `output.md` that the primary GRU/regime retraining intentionally used Round 1 or Round 2 price labels.

Remaining split risks:

- Round 2 and Round 3 result JSON metadata still report `split: validation` even when the date windows are not the validation split.
- Journal records do not carry enough split provenance to safely prevent later Quality/RL training from consuming validation/test outcomes.
- Round 3 is useful as a combined evaluation after Quality/RL retraining, but it should not be treated as a clean blind test because it runs after feedback from prior backtest journals.

Production requirement:

- Keep the split dates as-is.
- Add `source_split`, split date range, split summary hash, and run ID to every trade, journal row, model metric, calibration report, and backtest result.
- Fail any training command that attempts to use non-train rows unless it is explicitly marked as an experimental post-validation loop.

## Headline Backtest Results

The latest local result files show active strategy execution:

| Round | File | Window | Trades | Win rate | Profit factor | Return field | Max DD | Sharpe |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Round 1 | `backtest_20260427_001118.json` | 2021-08-05 to 2023-08-04 | 3,991 | 59.31% | 3.235 | 33.3392 | 7.83% | 8.544 |
| Round 2 | `backtest_20260427_001525.json` | 2023-08-07 to 2025-08-05 | 3,109 | 55.26% | 2.844 | 22.7140 | 3.50% | 7.426 |
| Round 3 | `backtest_20260427_003109.json` | 2022-08-05 to 2025-08-05 | 7,127 | 57.54% | 3.010 | 55.4885 | 6.80% | 7.960 |

These headline numbers are attractive, but they should not be used for production approval until the blockers below are fixed and the backtest is rerun.

## Production Blockers

### 1. Backtest and Journal Timestamps Are Corrupted

Severity: **critical**.

All three saved backtest trade logs and the current journal show event timestamps around `1970-01-20` to `1970-01-21`, while the tested market windows are 2021 to 2025.

Observed examples:

- Round 1 trade timestamps: `1970-01-20T07:00:00+00:00` to `1970-01-20T13:45:03.600000+00:00`
- Round 2 trade timestamps: `1970-01-20T13:49:26.400000+00:00` to `1970-01-21T07:19:08.400000+00:00`
- Round 3 trade timestamps: `1970-01-20T07:00:00+00:00` to `1970-01-21T07:19:10.200000+00:00`
- `trade_journal_detailed.jsonl` starts and ends with 1970 timestamps.
- `backtest_diagnostics.csv` also starts with 1970 timestamps.

Likely source:

- `trading-system/trading-engine/scripts/run_backtest.py` reconstructs loop timestamps with `pd.Timestamp(ts_ns, tz="UTC")`.
- The timestamp scalar is built from index integer views earlier in the backtest loop.

Impact:

- Session attribution is not auditable.
- Trades per day is invalid.
- Daily halt validation is invalid.
- Daily drawdown and Sharpe are suspect.
- Journal chronology is invalid.
- RL replay episodes inherit corrupted time context.
- Any time-based production risk logic cannot be trusted.

Required fix:

- Carry `pd.Timestamp` values directly through the backtest iterator, or convert integer timestamps with an explicit unit using `pd.to_datetime(ts_ns, unit="ns", utc=True)`.
- Add a hard assertion that every entry and exit timestamp lies inside the configured backtest window.
- Rerun Round 1 and Round 2 after the fix without changing the split windows.

### 2. Market-Structure Indicators Still Contain Lookahead

Severity: **critical**.

The backtest feature path uses market-structure functions with centered rolling windows:

- `trading-system/trading-engine/indicators/market_structure.py:458`
- `trading-system/trading-engine/indicators/market_structure.py:622`
- `trading-system/trading-engine/indicators/market_structure.py:773`

These use `rolling(..., center=True)`. Centered windows require future bars. The affected fields feed signal gating and ML feature context through break-of-structure, support/resistance, and range detection.

Impact:

- Current backtest performance may be materially overstated.
- Round 1 and Round 2 cannot be considered causal.
- This is more serious than normal overfitting because the backtest itself can see future bars.

Required fix:

- Replace centered rolling logic with causal confirmation.
- If swing confirmation needs `n` future bars, shift the confirmed signal forward by `n` bars and make it unavailable before that confirmation time.
- Rerun model feature generation and backtests after fixing.

### 3. Result Metadata Mislabels Splits

Severity: **high**.

The result JSONs use `split: validation` even for date windows that are not exactly the validation split. This does not necessarily mean the price windows were wrong, but it does break auditability.

Impact:

- Round 2 blind-test results are not self-describing.
- Any downstream process that trusts the `split` field can train, calibrate, or report incorrectly.

Required fix:

- Result metadata must store both logical split name and explicit `start/end`.
- Manual override windows must be marked as `custom_window` or `combined_eval`, not silently reported as `validation`.
- Backtest output should include the split summary hash.

### 4. Production Risk Controls Were Disabled in the Reported Backtests

Severity: **high**.

`trading-system/pipeline/step6_backtest.py` sets `BACKTEST_ENFORCE_DAILY_HALT=0` for the reported run. Gate diagnostics also show:

- `enforce_daily_halt: false`
- `daily_halt_events: 0`
- `compound_equity: false`

Impact:

- The reported returns are fixed-size accumulated PnL over initial capital, not a production compounding equity curve.
- Daily risk-stop behavior was not validated.
- Max drawdown is not a sufficient substitute for production risk controls.

Required fix:

- Keep a research-mode journal-generation run if needed.
- Add a separate production-mode evaluation that requires daily halt enabled, production slippage/commission, max exposure constraints, and realistic sizing.
- Report research and production-mode results separately.

### 5. Journal Feedback Can Contaminate Quality and RL Training

Severity: **high**.

The journal now contains 14,227 trades:

- 3,991 Round 1 trades
- 3,109 Round 2 trades
- 7,127 Round 3 trades

Quality/RL retraining used 7,100 samples/episodes after Round 1 and Round 2. This means the post-Round-2 Quality/RL update consumed validation and blind-test outcomes unless additional split filtering occurred outside the observed journal path.

Impact:

- Round 3 is no longer a clean blind evaluation.
- Quality/RL can learn from validation/test outcomes.
- The artifact lineage is not acceptable for production promotion.

Required fix:

- Add split provenance to every journal row.
- Train Quality/RL only on train-split journals for production artifacts.
- Treat any validation/test feedback loop as research-only and label it clearly.

### 6. Candidate Outcome Logs Do Not Represent Simulated Exit Time

Severity: **medium-high**.

`backtest_candidate_log.csv` contains one candidate row and one `__outcome` row per latest-round trade. The outcome rows are logged microseconds after candidate rows, even though diagnostics show trades held for 1 to 200 bars.

Impact:

- Candidate logs are useful for batch calibration counts, but not for event-time audit.
- Outcome timing cannot be used to validate latency, holding period, daily risk, or live-like event ordering.

Required fix:

- Store both decision timestamp and simulated exit timestamp.
- Outcome rows should use the simulated exit time, not wall-clock logging time.

### 7. Processed Data Manifests Contain Merge Conflict Markers

Severity: **high**.

The following files contain unresolved conflict markers:

- `trading-system/processed_data/feature_manifest.json`
- `trading-system/processed_data/alignment_summary.json`

Observed markers include `<<<<<<<`, `=======`, and `>>>>>>>`.

Impact:

- These files are invalid JSON.
- Data provenance is broken.
- A deployment or training process that reads these files may fail or silently use stale replacements.

Required fix:

- Resolve the conflict markers.
- Regenerate the manifests from the canonical dataset.
- Add a CI check that rejects conflict markers and invalid JSON in artifact paths.

### 8. Metrics and Provenance Are Stale or Contradictory

Severity: **medium-high**.

Examples:

- `trading-system/ml_training/metrics/critic_report.json` says models are untrained and zero trades, contradicting the latest weights and backtests.
- `trading-system/ml_training/metrics/training_summary.json` references a stale Kaggle path.
- `trading-system/ml_training/metrics/retrain_history.json` is shorter and older than `trading-system/trading-engine/logs/retrain_history.jsonl`.

Impact:

- There is no single production manifest tying together dataset split, feature contract, weights, retrain history, calibration, and backtest output.
- Human review can easily approve the wrong artifact set.

Required fix:

- Generate one promotion manifest per candidate model bundle.
- Include dataset hash, split summary hash, feature manifest hash, model weight hashes, calibration report hash, backtest result hashes, and code commit.

## Model Assessment

### GRU/LSTM Direction Model

Positive:

- Main model artifact exists at `trading-system/trading-engine/weights/gru_lstm/model.pt`.
- `weights_manifest.json` exists.
- Feature contract hashes in the GRU manifest match current feature constants reported by the artifact audit.

Concerns:

- Probability calibration is not proven current. Temperature loading exists, but the retraining path does not clearly require a fresh temperature fit.
- Direction gates depend heavily on `p_bull/p_bear`; uncalibrated probabilities can distort trade selection.
- Lookahead in upstream market-structure features means reported model-assisted performance is not yet causal.

Production decision:

- Not ready until causal features and probability calibration are rerun and documented.

### Regime Models

Positive:

- `regime_htf.pkl` and `regime_ltf.pkl` exist.
- Feature contract hashes match current source constants according to the artifact audit.

Concerns:

- Regime features may consume market-structure fields affected by centered rolling logic.
- The latest performance by regime is profitable, but causality must be repaired first.

Production decision:

- Promising, but blocked by feature causality and provenance.

### QualityScorer

Latest training summary:

- Samples: 7,100
- EV mean: 0.3922
- EV std: 1.2905
- Positives: 4,085
- Negatives: 3,015
- Validation MAE: 1.167
- Direction accuracy: 0.576

Concerns:

- Direction accuracy is only modest.
- MAE is large for an EV model used to filter trades.
- `_compute_ev_label()` appears to map exit reason to heuristic EV constants rather than computing true realized R from PnL divided by initial risk.
- Quality scores are mixed-provenance in the journal: early Round 1 rows have `quality_score: 0.0`, later rows have active quality scores.

Production decision:

- Not ready for autonomous capital allocation. It can remain a research filter after label and provenance fixes.

### RL PPO Model

Positive:

- `trading-system/trading-engine/weights/rl_ppo/model.zip` exists.
- The run logs show the PPO artifact was loaded and warm-started.

Concerns:

- RL replay used 7,100 episodes from journal rows that appear to include Round 1 and Round 2 outcomes.
- Journal timestamps are corrupted.
- `rl_action` is mostly assigned as `1` for `ml_trader`, while the RL action space appears to expect a richer set of trader/action choices.
- The replay environment rewards the next journal result rather than a true counterfactual action choice.

Production decision:

- Disable RL influence in production until the environment is redesigned or recast as supervised policy learning with clean train-only labels.

## Strategy Assessment

Positive:

- The strategy no longer has the zero-trade Round 1/Round 2 failure.
- Round 1 and Round 2 both produce thousands of trades.
- Profit factor remains above 2.8 in both Round 1 and Round 2.
- Round 2 blind-window headline metrics degrade from Round 1 but remain positive, which is directionally expected.

Concerns:

- The performance is too strong to trust while causal lookahead and timestamp corruption remain.
- The return fields are fixed-size accumulated returns over initial capital with compounding disabled.
- Daily halt is disabled in the reported backtest.
- Calibration is non-monotonic.
- Event-time logs are invalid.
- Round 3 is not a clean blind test.

Symbol notes from Round 3:

- Stronger symbols include XAUUSD, GBPUSD, AUDUSD, GBPJPY, and USDCHF by win rate.
- EURGBP is the weakest large-sample symbol at about 50.8% win rate, though still net profitable in the summary.
- NZDUSD has only 29 trades in Round 3, so its high win rate is not meaningful.

Exit behavior:

- `sl_full`: 3,026 exits, all losers by definition.
- `be_or_trail`: 2,269 exits, all winners by recorded outcome.
- `tp2`: 1,826 exits, all winners by recorded outcome.
- `time_exit`: 6 exits, all winners, too small to interpret.

The exit mix is not automatically wrong, but the exit reason is mechanically tied to the win/loss label. It should not be used as an independent validation signal for QualityScorer unless true realized R is computed first.

## Calibration Assessment

Latest calibration report:

| Confidence bin | Count | Actual win rate |
| --- | ---: | ---: |
| 0.55-0.60 | 3,008 | 57.41% |
| 0.60-0.65 | 3,020 | 56.89% |
| 0.65-0.70 | 814 | 59.83% |
| 0.70-0.75 | 206 | 59.71% |
| 0.75-0.80 | 67 | 59.70% |
| 0.80-0.90 | 11 | 54.55% |

The report marks calibration as reliable but not monotonic. Higher confidence does not consistently map to higher realized win rate, and the highest bin has too few samples.

Production requirement:

- Calibrate on a prior closed split only.
- Report ECE, Brier score, bin counts, and monotonicity.
- Do not use same-run executed outcomes to calibrate the same backtest.

## Artifact Readiness

Present artifacts:

- `trading-system/trading-engine/weights/gru_lstm/model.pt`
- `trading-system/trading-engine/weights/gru_lstm/weights_manifest.json`
- `trading-system/trading-engine/weights/regime_htf.pkl`
- `trading-system/trading-engine/weights/regime_ltf.pkl`
- `trading-system/trading-engine/weights/quality_scorer.pkl`
- `trading-system/trading-engine/weights/rl_ppo/model.zip`

Artifact blockers:

- Processed data manifests are invalid due to merge conflict markers.
- Metrics are stale and contradictory.
- No single promotion manifest exists.
- Local audit environment could not deserialize model binaries because `numpy`/`torch` were unavailable in the shell environment used by the read-only agent.
- `weights/backups` at repo root does not exist; backups are under `trading-system/trading-engine/weights/backups`.

## Required Remediation Plan

### P0 - Must Fix Before Trusting Any Backtest

1. Fix timestamp reconstruction in `run_backtest.py`.
2. Add assertions that trade timestamps and exit timestamps are inside the configured backtest window.
3. Remove or causally delay all `rolling(..., center=True)` market-structure features.
4. Resolve conflict markers in processed data manifests.
5. Rerun Round 1 and Round 2 with the same split windows.

### P1 - Must Fix Before Production Promotion

1. Add split provenance to every journal, calibration, and backtest artifact.
2. Prevent Quality/RL training from using validation or blind-test rows for production artifacts.
3. Add production-mode backtesting with daily halt enabled.
4. Separate research-mode journal generation from production-mode evaluation.
5. Generate a single promotion manifest tying together code, data, splits, weights, metrics, calibration, and backtest results.
6. Recompute probability calibration on a closed prior split.

### P2 - Strongly Recommended

1. Replace QualityScorer heuristic labels with true realized-R labels.
2. Disable RL in production until its action/reward semantics are valid.
3. Add JSON/provenance CI checks.
4. Add candidate logs with both decision event time and simulated outcome time.
5. Migrate Gym usage to Gymnasium.
6. Decide whether FAISS and macro inputs are required production dependencies or optional research enhancements.

## Production Readiness Checklist

| Area | Status | Reason |
| --- | --- | --- |
| Split boundaries | Partial pass | Date windows are intact, but metadata and journal provenance are insufficient. |
| Backtest event time | Fail | Trade and journal timestamps are 1970. |
| Causal features | Fail | Centered rolling windows remain in market-structure features. |
| Risk controls | Fail | Daily halt disabled in reported run. |
| Strategy performance | Not valid yet | Good headline metrics, but invalidated by timestamp and lookahead blockers. |
| GRU model | Conditional | Artifact exists, calibration/provenance incomplete. |
| Regime model | Conditional | Artifacts exist, but upstream features need causal audit. |
| Quality model | Fail for production | Weak/modest validation, questionable labels, possible non-train feedback. |
| RL model | Fail for production | Journal contamination, timestamp corruption, weak action semantics. |
| Calibration | Fail for production | Non-monotonic and potentially same-run/in-sample. |
| Artifact provenance | Fail | Conflict markers, stale metrics, no promotion manifest. |
| Deployment readiness | Fail | Required production validation has not passed. |

## Final Assessment

The system is **not production ready** as of 2026-04-27.

The strategy is worth continuing because Round 1 and Round 2 now produce nonzero, directionally strong results while preserving the intended split windows. But the current results cannot be promoted because they are affected by critical timestamp corruption, direct lookahead in market-structure features, disabled production risk controls, and weak artifact provenance.

The next valid milestone is not live deployment. It is a clean rerun of Round 1 and Round 2 after fixing timestamps and causal feature generation, with the existing split windows preserved exactly.
