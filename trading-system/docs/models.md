# ML Models Reference

Last updated: 2026-04-19. All models live in `trading-engine/models/`. All are PyTorch — LightGBM and XGBoost have been removed.

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
| ID | Name | Rule |
|----|------|------|
| 0 | TRENDING_UP | ADX > 25 AND EMA stack ≥ 1 AND drift > 0 |
| 1 | TRENDING_DOWN | ADX > 25 AND EMA stack ≤ -1 AND drift < 0 |
| 2 | RANGING | Default (low ADX, low drift) |
| 3 | VOLATILE | ATR percentile ≥ 80th percentile |

### Label Quality Features
- **Per-group GMMs** on 4H data: `dollar` (7 pairs), `cross` (3 pairs), `gold` (XAUUSD)
- **Confidence scoring** per bar (0→1): scales with ADX strength, EMA stack alignment, drift magnitude
- **Minimum persistence filter**: regime runs shorter than N bars are zero-weighted (not labelled differently, just ignored by the trainer):
  - 4H: min 20 bars (~80 hours)
  - 1H: min 48 bars (~48 hours)
  - Prevents label noise from short-lived regime flips that have no predictive value
- **Ambiguous bars** (confidence < 0.4): trained with soft targets, not hard labels

### Output
```python
{"regime": "TRENDING_UP", "regime_id": 0, "proba": [0.7, 0.1, 0.1, 0.1], "regime_confidence": 0.7}
```

### Training
- Latest 4H: 120,490 samples — acc=0.477 (4-class balanced; random=0.25)
- Latest 1H: 468,252 samples — acc=0.395
- Val loss diverges quickly (train≈0.61, val≈1.03) → early stop at epoch 12-13
- Root cause of low accuracy: 24-25% of all bars are ambiguous, RANGING is the majority class (~43%) but hardest to separate
- Persistence improvements (min-persist filter) are expected to improve RANGING accuracy on next run
- Warm-start: `_warm_start` flag checked before `_build_mlp()` — skips reinit if weights present and feature count matches; trains at `lr=6e-5` (warm) vs `lr=3e-4` (cold)

### 59 REGIME_FEATURES
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
  → direction_head: FC(64) → GELU → FC(1) → sigmoid  → p_bull ∈ [0,1]
  → magnitude_head: FC(64) → GELU → FC(1) → ReLU     → expected_move ≥ 0
  → variance_head:  FC(64) → GELU → FC(1)             → log_variance (unbounded)
```

### Loss
`BCE(dir) + 0.5×SmoothL1(mag) + 0.3×NLL(var)`

### Output
```python
{
    "p_bull": 0.72,
    "p_bear": 0.28,
    "expected_move": 0.0015,
    "expected_variance": 0.3,   # uncertainty gate: reject if > 0.80
}
```

### Training Data
- Latest: 7,081,756 sequences, 44 symbol/timeframe combos, input_size=72 (includes HTF context)
- DataParallel across both Kaggle T4s
- Early stop at epoch 6 (train=0.5943, val=0.6184)

### Regime Conditioning
`prev_regime_0..3` (one-hot) and `regime_confidence` injected at every sequence timestep. Regime inference runs before GRU sequence building — order is load-bearing.

### GRU Excluded from Reinforcement Loop
GRU is **NOT** retrained in `step6_backtest.py`'s per-round warm-start loop. Fine-tuning on ~3k backtest journal trades against 7M training sequences causes catastrophic forgetting. GRU retrains monthly from full OHLCV history only.

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
| 32–33 | 2 | time_sin, time_cos (hour cyclic encoding) |
| 34–52 | 19 | Macro: 17 index returns + VIX level + yield spread |

---

## QualityScorer (EV Regressor)

**File:** `models/quality_scorer.py`  **Weights:** `weights/quality_scorer.pkl`

### Purpose
Predicts expected value (EV) of a trade in R-multiples. Captures both win probability AND payoff magnitude — the actual decision variable for trade selection.

### Architecture
```
17 → BatchNorm → FC(64) → BN → GELU
  → FC(32) → BN → GELU
  → FC(1)  [identity output — unbounded float]
```

### Loss
Class-weighted Huber loss — winners upweighted ~4.1× to correct for ~20%/80% positive/negative label imbalance.
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

### EV Label Tiers
Labels encode exit quality so the model learns to aim for TP2:

| Exit reason | EV label | Meaning |
|---|---|---|
| `tp2` | `clip(rr_ratio, 0.1, 10.0)` | Full TP — best outcome |
| `tp1` | `clip(rr_ratio × 0.75, 0.1, 10.0)` | TP1 hit, held further |
| `be_or_trail` | `clip(rr_ratio × 0.4, 0.1, 10.0)` | TP1 hit, trailed out |
| `sl_*` | `-1.0` | Loss |
| `time_exit` | `None` (skipped) | Ambiguous — excluded |

Winners are normalised by `median_win` RR so labels cluster around ±1R: `y[win] = clip(y[win] / median_win, 0.0, 3.0)`.

### Warm-start
Detects `self._model is not None and self._n_features == n_feat` → trains at `lr=2e-4` (vs `lr=1e-3` cold).
Latest: 5,264 journal entries, EV stats: mean=-0.356, n_pos=1,003 vs n_neg=4,261, dir_acc=0.615, MAE=0.933.

### 17 QUALITY_FEATURES
| Index | Feature |
|-------|---------|
| 0 | strategy_id |
| 1 | signal_direction (1=buy, 0=sell) |
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

**File:** `models/rl_agent.py`  **Weights:** `weights/rl_ppo/model.zip`

### Purpose
Selects which trader to approve and at what EV selectivity tier.

### Algorithm
PPO via stable-baselines3. MLP policy — **always runs on CPU** (SB3 guidance: MlpPolicy is faster on CPU than GPU).

### Save / Load
Weights saved and loaded as `weights/rl_ppo/model.zip` (explicit `.zip` extension prevents SB3 from creating a directory instead of a file, which broke warm-start in previous versions).

### Warm-start
```python
if self._model is not None and self.is_trained:
    self._model.set_env(env)
    self._model.learning_rate = 3e-4 / 5.0
    self._model.learn(reset_num_timesteps=False)
else:
    self._model = PPO("MlpPolicy", env, device="cpu", ent_coef=0.01, ...)
```

### Action Space (16 actions)
| Actions | Meaning |
|---------|---------|
| 0 | NoTrade |
| 1–5 | Approve trader 1–5 at EV threshold 0.55 |
| 6–10 | Approve trader 1–5 at EV threshold 0.65 |
| 11–15 | Approve trader 1–5 at EV threshold 0.75 |

### 42-dim State Vector
| Dims | Content |
|------|---------|
| 0–5 | ML predictions: p_bull, p_bear, entry_depth, regime_id, sentiment, quality_score |
| 6–13 | Market structure: ADX, EMA stack, ATR ratio, BB width, BOS bull/bear, FVG bull/bear |
| 14–18 | Session context: asian, london, ny, overlap, news_proximity |
| 19–23 | Strategy signal one-hot (traders 1–5) |
| 24–29 | Portfolio state: open_positions, drawdown, daily_pnl, trades_today, last_result, equity_pct |
| 30–33 | Instrument one-hot (EURUSD, GBPUSD, USDJPY, XAUUSD) |
| 34–41 | ATR lag ratios (8 bars) |

### Training requirement
≥ 500 completed trades with `rl_action` and `state_at_entry` in the JSONL journal.

---

## Shared Properties

| Property | All models |
|----------|-----------|
| Untrained | Raises `ModelNotTrainedError(RuntimeError)` |
| Silent fallback | None — all failures raise |
| Hot-reload | `reload_if_updated()` checks mtime every 5 min |
| GPU | RegimeClassifier + GRU: `torch.amp.autocast("cuda")` + DataParallel |
| RL device | CPU always (MLP policy) |
| Warm-start | Existing weights → continue training at 5× lower LR |
| GRU reinforcement loop | Excluded — monthly OHLCV retraining only |
