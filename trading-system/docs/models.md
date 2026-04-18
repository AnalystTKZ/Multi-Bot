# ML Models Reference

Last updated: 2026-04-18. All models live in `trading-engine/models/`. All are PyTorch — LightGBM and XGBoost have been removed.

**Critical rule:** All models raise `ModelNotTrainedError(RuntimeError)` when used without trained weights. No silent fallbacks. Train all models before setting `ML_ENABLED=true`.

**Incremental warm-start:** All models detect existing weights at fit time and continue training at 5× lower LR instead of reinitialising from scratch. This preserves knowledge accumulated from the full historical dataset.

---

## RegimeClassifier

**File:** `models/regime_classifier.py`  **Weights:** `weights/regime_4h.pkl` / `weights/regime_1h.pkl`

### Purpose
Labels each bar as one of 4 market regimes. Every downstream model conditions on this output.

### Architecture
```
59 → BatchNorm → FC(128) + residual_skip(128) → BN → GELU → Dropout(0.5)
  → FC(64) → BN → GELU → FC(4)
```

### Classes
| ID | Name | Condition |
|----|------|-----------|
| 0 | TRENDING_UP | drift > 5e-5 AND efficiency > 0.25 |
| 1 | TRENDING_DOWN | drift < -5e-5 AND efficiency > 0.25 |
| 2 | RANGING | default (low drift, low efficiency) |
| 3 | VOLATILE | vol > p60 AND vol_slope > 0 |

### Output
```python
{"regime": "TRENDING_UP", "regime_id": 0, "proba": [0.7, 0.1, 0.1, 0.1], "regime_confidence": 0.7}
```
`regime_confidence = max(softmax(logits))` — fed into GRU as a sequence feature.

### Training
- Per-group GMMs on 4H data: `dollar` (7 pairs), `cross` (3 crosses), `gold` (XAUUSD)
- Feature-axis constraints for stable cluster assignment across retrains
- 4H labels forward-filled to training TF
- Latest: 4H — 101,416 train / 25,354 val, accuracy 48.8%; 1H — 394,124 train / 98,531 val, accuracy 41.1%
- Warm-start: `_warm_start` flag detected before `_build_mlp()` — skips reinit if weights present and feature count matches

### 59 Features
| Indices | Description |
|---------|-------------|
| 0–7 | Base: ADX, EMA stack, ATR ratio, BB width, realized vol, session code; indices 6–7 zeroed (BOS/sweep lookahead) |
| 8–27 | 5 TFs × 4 features (ADX, EMA stack, ATR ratio, BB width) for 5M/15M/1H/4H/1D |
| 28–33 | S/R zones — zeroed (detect_sr_zones uses centered rolling window = lookahead) |
| 34–35 | vol_slope (Δ ATR/close over 14 bars), regime_duration |
| 36–39 | prev_regime one-hot (4 dims) |
| 40 | regime_confidence |
| 41–57 | 17 global index returns |
| 58–59 | macro_vix_level, macro_yield_spread |

---

## GRU-LSTM Predictor

**File:** `models/gru_lstm_predictor.py`  **Weights:** `weights/gru_lstm/model.pt`

### Purpose
Predicts direction, magnitude, and uncertainty of the next price move from a 30-bar sequence. Conditions on regime at every timestep.

### Architecture
```
Input: (batch, 30, 53)
  → GRU(256 hidden, 2 layers, dropout=0.2)
  → LayerNorm(256) on last hidden state
  → FC(128) → GELU  [shared representation]
  → direction_head: FC(64) → GELU → FC(1) → sigmoid  → p_bull
  → magnitude_head: FC(64) → GELU → FC(1) → ReLU     → expected_move
  → variance_head:  FC(64) → GELU → FC(1)             → log_variance
```

### Loss
`BCE(dir) + 0.5×SmoothL1(mag) + 0.3×NLL(var)`

### Output
```python
{
    "p_bull": 0.72,
    "p_bear": 0.28,
    "expected_move": 0.0015,
    "expected_variance": 0.3,   # uncertainty gate: < 0.80
}
```

### Training Data
Latest run: **7,452,801 samples** across 44 symbol/timeframe combos, 30 val loss points. Trained on Kaggle T4 × 2 GPUs with DataParallel.

### Regime Conditioning
`prev_regime_0..3` (one-hot) and `regime_confidence` injected at every sequence timestep. Regime inference runs BEFORE GRU sequence building — order is load-bearing.

### GRU Excluded from Reinforcement Loop
GRU is NOT retrained in `step6_backtest.py`'s per-round warm-start loop. Fine-tuning on ~3k backtest journal trades against 7.4M training sequences causes catastrophic forgetting. GRU retrains monthly from full OHLCV history only.

### 53 SEQUENCE_FEATURES per Timestep
| Indices | Count | Description |
|---------|-------|-------------|
| 0–15 | 16 | Base 15M: log_return, hl_range, close_vs_open, ATR_norm, RSI, EMA21/50 dist, BB position, volume ratio, session flags, bos_bull/bear, fvg_bull/bear |
| 16–17 | 2 | 5M: RSI, EMA21 dist |
| 18–20 | 3 | 1H: ADX, EMA21 dist, EMA50 dist |
| 21–23 | 3 | 4H: EMA21-EMA50 diff / close, ADX, RSI |
| 24–25 | 2 | 1D: EMA21 dist, EMA stack score |
| 26–29 | 4 | prev_regime one-hot |
| 30 | 1 | regime_confidence |
| 31 | 1 | vol_slope_seq (Δ ATR/close, 14 bars) |
| 32–33 | 2 | time_sin, time_cos (hour cyclic) |
| 34–52 | 19 | Macro: 17 index returns + VIX level + yield spread |

---

## QualityScorer (EV Regressor)

**File:** `models/quality_scorer.py`  **Weights:** `weights/quality_scorer.pkl`

### Purpose
Predicts expected value (EV) of a trade in R-multiples. Captures both win probability AND payoff magnitude in a single score.

### Architecture
```
17 → BatchNorm → FC(64) → BN → GELU
  → FC(32) → BN → GELU
  → FC(1)  [identity output — unbounded float]
```

### Loss
Class-weighted Huber loss — winners upweighted ~4.6× to correct for 18%/82% positive/negative label imbalance.
```python
_pos_weight = clip(n_neg / n_pos, 1.0, 8.0)
w = where(target > 0, _pos_weight, 1.0)
loss = (w * huber(pred, target, delta=1.0)).mean()
```

### Output
```python
{
    "ev": 0.45,             # R-multiples; gate threshold: ≥ 0.10
    "quality_score": 0.61,  # sigmoid(ev), kept for logging
}
```

### EV Label Computation
Labels use `rr_ratio` directly (not `pnl/risk_staked` which produces broken ±25,000 values):
```python
if is_win:
    ev = clip(rr_planned, 0.1, 10.0)
else:
    ev = -1.0
```
Winners are then normalised by median winner RR so they cluster at ~+1.0: `y[win] = clip(y[win] / median_win, 0.0, 3.0)`.

### Warm-start
Detects `self._model is not None and self._n_features == n_feat` → trains at `lr=2e-4` (vs `lr=1e-3` cold). Latest training: 8,203 journal entries from step 7b.

### 17 QUALITY_FEATURES
| Index | Feature |
|-------|---------|
| 0 | strategy_id |
| 1 | signal_direction (1=buy, -1=sell) |
| 2 | rr_ratio |
| 3 | p_bull_gru |
| 4 | p_bear_gru |
| 5 | regime_class (0–3) |
| 6 | sentiment_score |
| 7 | adx_at_signal |
| 8 | atr_ratio_at_signal |
| 9 | volume_ratio |
| 10 | spread_at_signal |
| 11 | session_at_signal |
| 12 | news_in_30min |
| 13 | strategy_win_rate_20 |
| 14 | gru_uncertainty (expected_variance) |
| 15 | regime_duration |
| 16 | vol_slope_at_signal |

---

## SentimentModel

**File:** `models/sentiment_model.py`  **Weights:** pre-trained (no local file)

### Purpose
Scores news headlines for currency/instrument sentiment direction.

### Backends
1. **FinBERT** (`ProsusAI/finbert`) — primary
2. **VADER** — fallback when FinBERT unavailable

### Output
```python
{"sentiment_score": 0.6, "sentiment_label": "bullish", "sentiment_confidence": 0.8, "sentiment_backend": "finbert"}
```

Gold: USD bullish → XAUUSD bearish score (inverted).

### Usage
- Trader 4 hard gate: `sentiment_score ≥ 0.65` required
- Ensemble: `sentiment_bonus = abs(score) × 0.1` (direction match) or `-0.05` (mismatch)

---

## RLAgent (PPO)

**File:** `models/rl_agent.py`  **Weights:** `weights/rl_ppo/`

### Purpose
Selects which trader to approve and at what EV selectivity tier.

### Algorithm
PPO via stable-baselines3. Trains on completed trade episodes from JSONL journal.

### Warm-start
```python
_warm_start = (self._model is not None and self.is_trained)
if not _warm_start:
    self._model = PPO("MlpPolicy", env, learning_rate=3e-4, ...)
else:
    self._model.set_env(env)
    self._model.learning_rate = 3e-4 / 5.0   # fine-tune at lower LR
self._model.learn(total_timesteps=..., reset_num_timesteps=False)
```

### Action Space (16 actions)
| Actions | Meaning |
|---------|---------|
| 0 | NoTrade |
| 1–5 | Approve trader 1–5 at EV threshold 0.55 |
| 6–10 | Approve trader 1–5 at EV threshold 0.65 |
| 11–15 | Approve trader 1–5 at EV threshold 0.75 |

> **Known issue (2026-04-18):** RL is outputting action=1 for all trades — the policy has collapsed to a single action. Likely insufficient exploration on 8,203 episodes. Increase `ent_coef` and train longer once more journal data accumulates.

### 42-dim State Vector
| Dims | Content |
|------|---------|
| 0–17 | Market features |
| 18–23 | Signal presence per trader (0/1) |
| 24–29 | Signal confidence per trader |
| 30–35 | ML outputs (p_bull, p_bear, regime_id, quality_score, sentiment, ATR lags) |
| 36–41 | Portfolio state |

### Training requirement
≥ 500 completed trades with `rl_action` and `state_at_entry` in JSONL journal.

---

## Shared Properties

| Property | All models |
|----------|-----------|
| Untrained | Raises `ModelNotTrainedError(RuntimeError)` |
| Silent fallback | None — all failures raise |
| Hot-reload | `reload_if_updated()` checks mtime every 5 min |
| GPU | `torch.amp.autocast("cuda")` + DataParallel (Regime, GRU) |
| Warm-start | Existing weights detected → continue training at 5× lower LR |
