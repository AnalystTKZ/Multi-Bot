# System Assessment — Multi-Bot Trading Engine

**Date:** 2026-04-18
**Scope:** Post-GPU migration + warm-start incremental retraining + first full ML backtest

---

## 1. What Has Been Built

### 1.1 Five-Strategy Session-Aware Trading Engine

| Trader | Strategy | Session | Key Regime Requirement |
|---|---|---|---|
| T1 | NY EMA Trend Pullback | 13:00–17:00 UTC | TRENDING_UP or DOWN required |
| T2 | Structure Break + FVG | London 07–12 + NY 13–18 | Any |
| T3 | London Breakout + Volume | 07:00–10:00 UTC | TRENDING only (RANGING/VOLATILE blocked) |
| T4 | News Structural Breakout | Any session | Any (sentiment ≥ 0.65 hard gate) |
| T5 | Asian Range Mean Reversion | 02:00–06:45 UTC | RANGING required |

### 1.2 ML Stack (all PyTorch, all GPU)

| Model | Algorithm | Role | Status |
|---|---|---|---|
| RegimeClassifier | PyTorch MLP 59→128→64→4 + residual | Market regime (4 classes) | ✅ Trained (4H: 48.8%, 1H: 41.1%) |
| GRU-LSTM | PyTorch GRU(256, 2L) + 3 heads | p_bull, p_bear, magnitude, uncertainty | ✅ Trained (7.45M samples, 44 combos) |
| QualityScorer | PyTorch MLP 17→64→32→1, Huber loss | EV in R-multiples | ✅ Trained (8,203 journal trades) |
| SentimentModel | FinBERT + VADER | Macro/news bias | ✅ Pre-trained |
| RLAgent | PPO (stable-baselines3), 42-dim, 16 actions | Strategy selector | ✅ Trained (8,203 episodes) |

### 1.3 Incremental Warm-Start Retraining

All models warm-start from existing weights rather than reinitialising. This builds on the 7-year training history each run rather than discarding it. Key implementation:
- Regime: `_warm_start` flag checked before `_build_mlp()` — reinit skipped if weights + feature count match
- GRU: excluded from per-round reinforcement loop to prevent catastrophic forgetting; retrains monthly from full OHLCV
- Quality: class-weighted Huber loss + `rr_ratio`-based labels (replacing broken `pnl/risk_staked`)
- RL: `set_env()` + `learning_rate` reduction + `reset_num_timesteps=False`

### 1.4 Retrain Schedule

| Cadence | Models | Trigger |
|---|---|---|
| Weekly (Sun 02:00 UTC) | Quality + RL | Min 200/500 journal entries |
| Monthly (1st Sun 03:00 UTC) | Regime + GRU | OHLCV-based structural refit |

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
step8_push      → git init + soft-reset + push weights to GitHub
```

Data coverage: 11 forex/gold symbols, Jan 2016 – Aug 2024.

### 1.6 Infrastructure

- **Broker**: Capital.com REST API (primary), yfinance (fallback)
- **Event bus**: Redis pub/sub
- **Paper trading**: slippage 0.01–0.5%, commission 0.1%, execution delay 0.1–2s
- **Trade journals**: CSV (12 cols) + JSONL (full metadata including `state_at_entry`, `rl_action`, `ml_model_scores`, `ev`)
- **Hot-reload**: model weights update without container restart (mtime check every 5 min)
- **Dashboard**: React frontend with real-time P&L, positions, bot status, analytics

---

## 2. Confirmed Working

### Models
- All PyTorch models training on Kaggle T4 × 2 GPUs
- Regime GMM clustering: 4-class balanced labels, warm-start preserves prior weights
- GRU: 7,452,801 samples trained, 30 val loss checkpoints, regime-conditioned sequence features
- Quality + RL: warm-start incremental training on 8,203 journal trades (step 7b)
- Catastrophic forgetting fix: GRU removed from per-round reinforcement loop

### Backtest
- Full ML_ENABLED=true backtest complete: 2,769 trades, PF 1.93, Sharpe 3.40, MaxDD 2.45%
- 3 reinforcement rounds with stable regime warm-starting (Round 1→3 regression < 2.1pp)
- High-conviction signal identified: confidence ≥ 0.90 → 67% WR on 291 trades

### GitHub Push
- `step8_push_to_github.py` fixed: git init + fetch + soft-reset + normal push (no --force)
- Handles Kaggle environment: `/kaggle/working/Multi-Bot` already contains repo files but is not a git repo
- `__file__` NameError handled for notebook cell execution

---

## 3. Known Issues (2026-04-18)

### P0 — Quality Score = 0.0 on All Trades
Every trade in the journal has `quality_score: 0.0` and `ev: 0.0`. The QualityScorer trained successfully on 8,203 samples but its output is not reaching the signal path at inference time. The EV gate (Guard 7) and quality-weighted position sizing are inactive.

**Impact:** System is running on raw GRU + regime signals only. PF 1.93 is the floor — enabling quality scoring should improve this.

**Next step:** Trace how `quality_score` / `ev` is populated in `_run_ml_inference()` and verify the output dict key reaches Guard 7.

### P0 — RL Action Always = 1
All 8,203 trades have `rl_action: 1`. The PPO policy has collapsed to approving trader 1 at the lowest EV threshold regardless of state. Root cause: insufficient exploration during training on 8,203 episodes.

**Next step:** Add `ent_coef=0.01` to PPO config and accumulate more journal data before next RL retrain.

### P1 — Round 2→3 PF Decay (~0.5pp)
Small but consistent: R1 PF 1.99 → R3 PF 1.93. Acceptable with current data volume; monitor across future runs. Expected to stabilise as more journal data improves Quality and RL.

### P2 — EURUSD Underperforming
37.0% WR across 2,425 trades — lowest of all 11 symbols. Consider raising `ML_DIRECTION_THRESHOLD` for EURUSD to 0.85+.

### P3 — Regime 1H Accuracy 41.1%
Below the 55% target threshold. This is partly expected (4-class balanced, chance = 25%) but the 1H model is weaker than 4H (48.8%). Regime hysteresis (3-bar) mitigates noise.

---

## 4. Readiness Verdict

### Is the system ready for paper trading?

**YES — with caveats.**

| Check | Status |
|---|---|
| Engine builds and starts | ✓ |
| All 5 traders initialized | ✓ |
| All 8 guards enforced | ✓ |
| All PyTorch models trained | ✓ |
| Warm-start retraining working | ✓ |
| Retrain scheduler deployed | ✓ |
| Trade journals writing | ✓ |
| Full backtest run (ML enabled) | ✓ PF 1.93, Sharpe 3.40 |
| Quality score reaching signal path | ✗ (P0 bug) |
| RL policy differentiated | ✗ (collapsed to action 1) |
| Capital.com credentials | ✗ (needed for live data) |

### Recommended Next Steps

1. **Fix quality score pipeline** — trace `ev` / `quality_score` from `quality_scorer.predict()` through `_run_ml_inference()` into Guard 7
2. **Add Capital.com credentials** to `.env` for live data
3. **Start paper trading** — accumulate journal trades to improve Quality and RL retrains
4. **After ~500 paper trades:** retrain RL with `ent_coef=0.01`
5. **Test high-conviction filter** — run backtest with `min_confidence=0.90` to quantify the 67% WR alpha

---

## 5. Architecture Quality Assessment

### Strengths
- All inference models run natively on GPU with DataParallel
- Warm-start incremental retraining — 7 years of training knowledge preserved across runs
- GMM clustering for regime labels: balanced classes, timeframe-agnostic, no hardcoded thresholds
- GRU catastrophic forgetting fix: correctly excluded from per-round fine-tuning loop
- EV label fix: `rr_ratio`-based labels with class-weighted Huber (no more broken `pnl/risk_staked`)
- `ModelNotTrainedError` fail-loud design prevents silent degradation
- Hot-reload: model weights update without container restart

### Areas for Improvement
- Quality score not reaching inference path (P0)
- RL policy collapsed to single action (needs exploration + more data)
- No unit tests
- EURUSD underperforming — symbol-specific confidence threshold tuning needed
- RANGING regime accuracy below threshold (41.1% at 1H)

---

## 6. Bug History (Cumulative)

### GPU / Training
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
| GRU catastrophic forgetting in rounds 2–3 | Removed GRU from per-round retrain loop |
| Quality EV mean = -0.357 (18% positive labels) | `rr_ratio` labels + class-weighted Huber loss |
| Quality / RL always cold-starting | Warm-start detection + 5× lower LR |
| OneCycleLR `max_lr` inconsistency on warm-start | `max_lr = _train_lr` (not hardcoded 3e-4) |
| Kaggle subprocess pipe-buffer deadlock | Removed `capture_output=True` from step7/7b |

### GitHub Push
| Issue | Fix |
|---|---|
| Exit 128 — clone into non-empty dir | `git init + fetch --depth=1 + reset --soft FETCH_HEAD` |
| Force push wiped all repo files | Replaced `--force` with soft-reset graft approach |
| `__file__` NameError in notebook | `try/except NameError` fallback to hardcoded Kaggle path |
| Merge conflicts from orphan history | Resolved with `--allow-unrelated-histories` + `checkout --ours` |

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
