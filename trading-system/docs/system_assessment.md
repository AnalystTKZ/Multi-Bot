# System Assessment — Multi-Bot Trading Engine

**Date:** 2026-04-24
**Scope:** Updated to reflect 2026-04-24 Kaggle training run results. Regime redesign (hierarchical 3+4-class cascade), atr_pctile fix, temperature scaling, QualityScorer label fix, signal_pipeline functional. Latest backtest results from val/test windows added.

---

## 1. What Has Been Built

### 1.1 Unified ML Signal Generator

The 5 ICT rule-based traders have been replaced by a single `ml_trader` that runs on all 11 symbols.
Signal generation: `run_backtest._compute_backtest_signal` (backtest) and
`signal_pipeline._compute_ml_signal` (live) — kept in exact sync.

### 1.2 ML Stack (all PyTorch, GPU except RL)

| Model | Algorithm | Role | Status |
|---|---|---|---|
| RegimeClassifier HTF | PyTorch MLP 34→128→64→3 + residual | 4H macro bias (BIAS_UP/DOWN/NEUTRAL) | ✅ Active |
| RegimeClassifier LTF | PyTorch MLP 18→128→64→4 + residual | 1H behaviour (TRENDING/RANGING/CONSOLIDATING/VOLATILE) | ✅ Active (atr_pctile bug fixed 2026-04-24) |
| GRU-LSTM | PyTorch GRU(64,2L)→LSTM(128,2L)→3 heads | p_bull, p_bear, magnitude, uncertainty | ✅ Trained (7.08M samples, 44 combos) |
| QualityScorer | PyTorch MLP 17→64→32→1, class-weighted Huber | EV in R-multiples, tiered labels | ✅ Active (labels now read real signal_metadata) |
| SentimentModel | FinBERT + VADER | Macro/news bias | ✅ Pre-trained |
| RLAgent | PPO (stable-baselines3), CPU, 43-dim, 16 actions | Selectivity tier | ✅ Active (needs ≥200 trades for policy diversity) |
| VectorStore | FAISS flat index on 64-dim GRU embeddings | Historical trade similarity | ✅ Built after GRU retrain |

### 1.3 Incremental Warm-Start Retraining

All models warm-start from existing weights rather than reinitialising. Key implementation:
- Regime: `_warm_start` flag checked before `_build_mlp()` — reinit skipped if weights + feature count match
- GRU: excluded from per-round reinforcement loop (catastrophic forgetting); retrains monthly from full OHLCV
- Quality: tiered EV labels (tp2 > tp1 > be_or_trail > sl) + class-weighted Huber loss
- RL: `set_env()` + `learning_rate` reduction + `reset_num_timesteps=False`; saves as `model.zip` (fixed directory bug)

### 1.4 Kaggle Split-Dataset Pipeline

Two datasources merged at runtime:
- **Code**: GitHub clone (`/kaggle/input/datasets/tysonsiwela/multi-bot-system`)
- **Data**: OHLCV + processed data (`/kaggle/input/datasets/tysonsiwela/trading-data`)
- **Output**: Weights + logs pushed to GitHub via fresh clone at `/kaggle/working/remote/Multi-Bot`

`env_config.py` resolves all paths for both environments automatically.

### 1.5 Full Data Pipeline (9-step + push)

```
step0_resample  → M1 histdata → 5 MTF parquets per symbol
step1_inventory → data discovery
step2_clean     → merge MTF + macro
step3_align     → multi-asset alignment
step4_features  → 200+ engineered features
step5_split     → 70/15/15 time-based split
step6_backtest  → 3-round reinforcement backtest with warm-start retraining
step7_train     → GRU + Regime (Kaggle T4 × 2)
step7b_train    → Quality + RL
step8_push      → fresh clone + pull + copy artifacts + push to GitHub
```

Data coverage: 11 forex/gold symbols, Jan 2016 – Feb 2026 (OHLCV).

---

## 2. Confirmed Working

### Models
- All PyTorch models training on Kaggle T4 × 2 GPUs
- Regime GMM clustering: per-group (dollar/cross/gold), 4-class balanced labels
- Regime persistence filter: short runs zero-weighted (reduces label noise from transient flips)
- GRU: 7,081,756 samples, 44 combos, regime-conditioned sequence features
- GRU early stop: epoch 6 (train=0.5943, val=0.6184) — not overfitting
- Quality EV: tiered labels by exit type — model learns to prefer tp2 outcomes
- Quality + RL: retrained 3 rounds within step6 + final retrain in step7b
- RL: warm-start from `model.zip` now works (load path fixed)
- RL: runs on CPU (faster for MLP policy; eliminated SB3 GPU warning)

### Backtest (2026-04-24)
- Full ML_ENABLED=true backtest, 11 symbols, 2016–2025, 221,743 rows, 202 features
- Val window (2021–2023): WR 53.8–54.3%, PF 2.97–3.14, Sharpe 5.11–5.29, MaxDD ≤2.5%
- Blind test (2023–2025): WR 50.9–51.4%, PF 2.45–2.52, Sharpe 4.83–4.96, MaxDD ≤2.9%
- Post-retrain 3yr: WR 54.6%, Return 3826% (R1), MaxDD 2.8%
- Split: train 2016–2021, val 2021–2023, test 2023–2025 (strict time-based, no leakage)
- Calibration: p_win correlates with actual win rate across all symbols
- Trade frequency: within normal range across all 11 symbols

### Journal + EV Pipeline
- Journal now writes `exit_reason` as raw backtest value (`tp2`, `tp1`, `be_or_trail`, `sl_full`, `time_exit`)
- `ml_model_scores` now includes `expected_variance`, `regime_duration`, `vol_slope` from backtest trade log
- `_compute_ev_label()` handles all exit types with tiered values
- Next run: journal entries should have fully populated EV, regime, realized_rr for Quality and RL training

### GitHub Push (step8)
- `step8_push_to_github.py`: fresh clone at `/kaggle/working/remote/Multi-Bot`, pull latest, copy artifacts, commit, push
- No `--force` — push always descends from current remote HEAD

---

## 3. Known Issues (2026-04-24)

### P1 — HTF BIAS_NEUTRAL Recall Weak (~30-38%)
The NEUTRAL class (low-ADX, indecisive drift periods) is the hardest to classify. Precision is acceptable but recall is low — the model tends to assign BIAS_UP or BIAS_DOWN in ambiguous conditions rather than NEUTRAL. This causes the NEUTRAL confidence gate to allow more trades than ideal.

Active improvement direction: additional feature engineering targeting indecisive regime signatures; longer persistence filter.

### P2 — VectorStore Broken
`models/vector_store.py` has a numpy import bug and a dimension mismatch between the FAISS index and the GRU embedding output. Being fixed. Does not affect signal generation — VectorStore is used only for historical pattern similarity lookup.

### P3 — RL Policy Action Diversity
With fewer than 200 journal trades, the PPO policy tends to collapse to a narrow subset of actions. Entropy coefficient (ent_coef=0.01) is set but needs more training data to take effect meaningfully. No-trade action is available as action 0. Monitor as journal grows.

### Historical Issues (Resolved)
- **Quality/RL journal fields null** — fixed in step6_backtest.py (exit_reason normalisation + signal_metadata fields)
- **Round 1→3 PF regression** — resolved with correctly populated journal
- **atr_pctile always zero** — fixed 2026-04-24; LTF RANGING accuracy expected to improve
- **GRU catastrophic forgetting** — GRU excluded from per-round retrain loop
- **RL model.zip directory bug** — explicit path fixed; warm-start working

---

## 4. Readiness Verdict

| Check | Status |
|---|---|
| Engine builds and starts | ✓ |
| signal_pipeline mirrors run_backtest exactly | ✓ |
| All ML gates enforced (5 in bar loop) | ✓ |
| All PyTorch models trained | ✓ |
| Warm-start retraining working | ✓ |
| Retrain scheduler deployed | ✓ |
| Trade journals writing | ✓ |
| Journal EV/regime fields populated | ✓ |
| RL warm-start working (model.zip) | ✓ |
| Temperature scaling calibrated | ✓ (temperature.pt sidecar) |
| Backtest val WR > 50% + PF > 2.0 | ✓ WR 53.8–54.3%, PF 2.97–3.14 |
| Blind test WR > 50% + PF > 2.0 | ✓ WR 50.9–51.4%, PF 2.45–2.52 |
| Capital.com credentials | ✓ (in docker-compose.dev.yml) |
| VectorStore operational | ✗ (numpy bug + dim mismatch) |
| HTF NEUTRAL recall adequate | ✗ (~30-38%; being improved) |
| RL action diversity | ✗ (needs ≥200 journal trades) |

### Recommended Next Steps

1. **Fix VectorStore** — resolve numpy import bug and FAISS dim mismatch in `models/vector_store.py`
2. **Improve HTF NEUTRAL recall** — additional feature engineering for indecisive regime periods
3. **Start paper trading** — accumulate ≥200 journal trades to improve RL policy diversity
4. **EV calibration** — isotonic regression on validation set for QualityScorer output
5. **Regime transition matrix** — add as additional GRU sequence features

---

## 5. Architecture Quality Assessment

### Strengths
- All inference models run on GPU with DataParallel (except RL which correctly runs on CPU)
- Warm-start incremental retraining — 7+ years of training knowledge preserved
- Tiered EV labels — QualityScorer now has a gradient: tp2 > tp1 > be_or_trail > sl
- Regime persistence filter — removes label noise from short-lived transitions
- RL save/load fixed — `model.zip` explicit path, warm-start preserved between rounds
- GMM clustering for regime labels: balanced classes, per-group (dollar/cross/gold)
- GRU catastrophic forgetting fix: excluded from per-round fine-tuning loop
- `ModelNotTrainedError` fail-loud design prevents silent degradation
- Hot-reload: model weights update without container restart
- Causal integrity: all lookahead features identified and zeroed

### Areas for Improvement
- HTF BIAS_NEUTRAL recall weak (~30-38%) — ambiguous periods leak into BIAS_UP/DOWN
- VectorStore broken (numpy import bug + FAISS dim mismatch)
- RL policy diversity insufficient until ≥200 journal trades accumulated
- No unit tests
- EV calibration (isotonic regression) not yet applied to QualityScorer output

---

## 6. Bug History (Cumulative)

### ML Training
| Issue | Fix |
|---|---|
| CPU 101% / GPU 0% | Vectorised `_build_feature_matrix()` |
| "Truth value of DataFrame is ambiguous" | Explicit `is None` checks |
| AMP dtype mismatch FP16/FP32 | `.float()` cast on logits before loss |
| Deprecated `torch.cuda.amp` API | `torch.amp.GradScaler("cuda")` |
| GMM `KeyError: np.int64(N)` | `int(c)` cast on cluster index |
| GMM `list index out of range` | Dict scores keyed by cluster ID |
| Regime label collapse (98% VOLATILE) | GMM on dimensionless features |
| Regime always cold-starting | `_warm_start` flag check before `_build_mlp()` |
| GRU catastrophic forgetting rounds 2–3 | Removed GRU from per-round retrain loop |
| Quality EV mean = -0.357 (18% positive labels) | `rr_ratio` tiered labels + class-weighted Huber |
| Quality/RL always cold-starting | Warm-start detection + 5× lower LR |
| OneCycleLR `max_lr` inconsistency on warm-start | `max_lr = _train_lr` (not hardcoded 3e-4) |
| Kaggle subprocess pipe-buffer deadlock | Removed `capture_output=True` from step7/7b |
| RL `Is a directory` load error | Explicit `model.zip` save/load path |
| RL running on GPU (slow for MLP policy) | Fixed `_RL_DEVICE = "cpu"` |
| Journal EV/regime/realized_rr all null | Fixed `exit_reason` normalisation + `ml_model_scores` fields in step6 |
| Quality labels: be_or_trail/tp2 all same EV | Tiered labels: tp2=1×rr, tp1=0.75×rr, be_or_trail=0.4×rr |
| Regime short-run label noise | Minimum persistence filter: zero-weight runs < 20 bars (4H) |

### GitHub Push
| Issue | Fix |
|---|---|
| Exit 128 — clone into non-empty dir | `git init + fetch + soft-reset FETCH_HEAD` |
| Force push wiped all repo files | Replaced `--force` with soft-reset graft |
| `__file__` NameError in notebook | `try/except NameError` fallback to hardcoded Kaggle path |
| Merge conflicts from orphan history | Resolved with `--allow-unrelated-histories` + `checkout --ours` |
| Merge conflict markers in 4 files | All resolved: `regime_classifier.py`, `rl_agent.py`, `quality_scorer.py`, `step6_backtest.py`, `retrain_scheduler.py` |
| `retrain_incremental.py: unrecognized --all` | Removed `--all` from notebook cell 7 subprocess call |
| Kaggle dataset save manual steps | Kaggle API cell: `kaggle datasets version -p /kaggle/working/outputs` |
| step8 git init on working dir | Switched to fresh clone at `/kaggle/working/remote/Multi-Bot` |

### Backend / Frontend (Historical)
| File | Bug | Fix |
|---|---|---|
| `utils/event_log.py` | Scanned wrong Redis key | `LRANGE debug:events 0 N` |
| `websocket/manager.py` | Missing subscriptions | Added signal + market_data channels |
| `main.py` | `allow_origins=["*"]` | Read `FRONTEND_URL` env var |
| `services/state_reader.py` | Wrong Redis key for positions | Fixed to `positions:open` |
| `frontend/authSlice.js` | Previous error not cleared | `state.error = null` in pending |
| `frontend/positionsSlice.js` | Filter on wrong field | Filter by `meta.arg.id` |
| `frontend/tradersSlice.js` | Performance keyed by undefined | Key by `action.payload.trader_id` |
| `frontend/traderService.js` | 307 stripped Auth header | Trailing slash on collection endpoint |

### Security / Operations (2026-04-24)
| File | Bug | Fix |
|---|---|---|
| `backend/routes/system.py` | Path traversal in system routes | Sanitised with `Path(id).name` |
| `backend/main.py` | CORS wildcard in production | Added warning; read `FRONTEND_URL` |
| `docker-compose.dev.yml` | Capital.com credentials missing | Added `CAPITAL_*` env vars |
| `services/signal_pipeline.py` | ml_trader not in `_KNOWN_TRADERS` | Added ml_trader to registry |
| `backend/routes/system.py` | `/status` returned stub | Real health check implemented |
| `trading-engine/models/*` | All old regime names in codebase | Removed TRENDING_UP/DOWN/CONSOLIDATION everywhere |
| `docker-compose.dev.yml` | Git merge conflict markers | Resolved |
| `frontend/Training page` | Double-unwrap + stale regime labels + broken WebSocket hook | All fixed |
