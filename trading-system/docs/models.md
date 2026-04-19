# ML Models Reference

Last updated: 2026-04-19. All models live in `trading-engine/models/`. All are PyTorch.

**Critical rule:** All models raise `ModelNotTrainedError(RuntimeError)` when used without
trained weights. No silent fallbacks. Train all models before setting `ML_ENABLED=true`.

---

## RegimeClassifier (dual-cascade)

**File:** `models/regime_classifier.py`
**Weights:** `weights/regime_4h.pkl` (bias) + `weights/regime_1h.pkl` (structure)

### Purpose
Two independent classifiers in cascade:
- **4H bias classifier** — labels macro market state from 4H+1D data
- **1H structure classifier** — refines intraday structure from 1H+4H data

Both outputs injected as one-hot context into every GRU sequence timestep.

### Architecture
Both classifiers share the same MLP structure:
```
N → BatchNorm → FC(128) + residual_skip(128) → BN → GELU → Dropout(0.5)
  → FC(64) → BN → GELU → FC(5)
```

### Classes
| ID | Name | Description |
|----|------|-------------|
| 0 | TRENDING_UP | Sustained upward drift, efficiency > 0.25 |
| 1 | TRENDING_DOWN | Sustained downward drift, efficiency > 0.25 |
| 2 | RANGING | Low drift, low efficiency |
| 3 | VOLATILE | Vol > p60 AND vol_slope > 0 |
| 4 | CONSOLIDATION | Tight range, low ATR percentile |

### Output
```python
{"regime": "TRENDING_UP", "regime_id": 0, "proba": [0.7, 0.1, 0.1, 0.05, 0.05], "regime_confidence": 0.7}
```

### Feature Counts
| Classifier | Features | Source |
|-----------|----------|--------|
| 4H bias | 31 (`REGIME_4H_FEATURES`) | 4H base + 1D context + regime dynamics + macro |
| 1H structure | 15 (`REGIME_1H_FEATURES`) | 1H base + session + BOS/sweep + 4H context + dynamics |

### Label Generation
- Per-group GMMs on 4H data only: `dollar` (7 pairs), `cross` (3 crosses), `gold` (XAUUSD)
- Feature-axis constraints for stable cluster assignment:
  - `vol > p60 AND vol_slope > 0` → VOLATILE
  - `drift > 5e-5 AND efficiency > 0.25` → TRENDING_UP
  - `drift < -5e-5 AND efficiency > 0.25` → TRENDING_DOWN
  - low drift AND low ATR percentile → CONSOLIDATION
  - else → RANGING
- 4H labels forward-filled to 1H training TF
- Hysteresis: 3 consecutive bars required to switch regime

---

## GRU-LSTM Predictor

**File:** `models/gru_lstm_predictor.py`  **Weights:** `weights/gru_lstm/model.pt`

### Purpose
Predicts direction, magnitude, and uncertainty of the next price move from a
30-bar sequence of 74 features. Receives 4H and 1H regime context at every timestep.

### Architecture
```
Input: (batch, 30, 74)
  → GRU(64 hidden, 2 layers, dropout=0.2)
  → LSTM(128 hidden, 1 layer)
  → LayerNorm(128) on last hidden state
  → shared FC(128) → GELU
  → direction_head:  FC(64) → GELU → FC(1) → sigmoid  → p_bull
  → magnitude_head:  FC(64) → GELU → FC(1) → ReLU     → expected_move
  → variance_head:   FC(64) → GELU → FC(1)             → log_variance
```

### Loss
`BCE(dir) + 0.5×SmoothL1(mag) + 0.3×NLL(var)`

### Output
```python
{
    "p_bull": 0.72,
    "p_bear": 0.28,
    "expected_move": 0.0015,
    "expected_variance": 0.3,   # uncertainty; gate: ≤ 0.80
    "expected_volatility": 0.55,
    "entry_depth": 0.15,
}
```

### Regime Conditioning
`regime_4h_0..4` (one-hot) + `regime_4h_conf` and `regime_1h_0..4` + `regime_1h_conf`
are injected at every sequence timestep (indices 26–37 of `SEQUENCE_FEATURES`).
Regime inference runs BEFORE GRU sequence building in `_precompute_ml_cache` — order is load-bearing.

### 74 SEQUENCE_FEATURES per Timestep
| Indices | Count | Description |
|---------|-------|-------------|
| 0–15 | 16 | Base 15M: log_return, HL range, close_vs_open, ATR_norm, RSI, EMA21/50 dist, BB pos, vol ratio, session flags, BOS flags, FVG flags |
| 16–17 | 2 | 5M: RSI, EMA21 dist |
| 18–20 | 3 | 1H: ADX, EMA21 dist, EMA50 dist |
| 21–23 | 3 | 4H: EMA21-50 diff/close, ADX, RSI |
| 24–25 | 2 | 1D: EMA21 dist, EMA stack score |
| 26–31 | 6 | 4H regime one-hot (5 dims) + confidence |
| 32–37 | 6 | 1H regime one-hot (5 dims) + confidence |
| 38 | 1 | vol_slope_seq (Δ ATR/close, 14 bars) |
| 39–40 | 2 | time_sin, time_cos (hour cyclic) |
| 41–71 | 31 | ICT structure distances: EMA pullback zone, BOS age/strength, FVG dist/fill, sweep wick depth, Asian range context, candle body/wicks, stochastic, ADX, regime dynamics, session timing |
| 72–73 | 2 | macro_vix_level, macro_yield_spread |

---

## QualityScorer (EV Regressor)

**File:** `models/quality_scorer.py`  **Weights:** `weights/quality_scorer.pkl`

### Purpose
Predicts expected value (EV) of a trade in R-multiples. Runs **post-signal** once the
actual `rr_ratio`, `side`, and `trader_id` are known — not in the GPU pre-compute cache.

### Architecture
```
17 → BatchNorm → FC(64) → BN → GELU
  → FC(32) → BN → GELU
  → FC(1)  [identity output — unbounded float]
```

### Loss
`HuberLoss(delta=1.0)` — robust to large-R outliers.

### Output
```python
{"ev": 0.45, "quality_score": 0.61}  # quality_score = sigmoid(ev)
```

### EV Label Computation
```python
ev = realized_pnl / (size × |entry - sl|)   # preferred
ev = realized_pnl / |entry - sl|             # if size missing
ev = planned_rr                               # TP exit, no pnl recorded
ev = -1.0                                    # SL exit, no pnl recorded
# clipped to [-5.0, 10.0]; trades without "tp"/"sl" in exit_reason excluded
```

### 17 QUALITY_FEATURES
| Index | Feature |
|-------|---------|
| 0 | strategy_id |
| 1 | signal_direction (1=buy, -1=sell) |
| 2 | rr_ratio |
| 3 | p_bull_gru |
| 4 | p_bear_gru |
| 5 | regime_class (0–4) |
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

### Backends
1. **FinBERT** (`ProsusAI/finbert`) — primary
2. **VADER** — fallback when FinBERT unavailable

### Output
```python
{"sentiment_score": 0.6, "sentiment_label": "bullish", "sentiment_confidence": 0.8}
```

Gold: USD bullish → XAUUSD bearish score (inverted).

Used by QualityScorer feature `sentiment_score` (index 6).

---

## RLAgent (PPO)

**File:** `models/rl_agent.py`  **Weights:** `weights/rl_ppo/model.zip`

### Purpose
Selects selectivity tier (EV threshold). Currently outputs action=1 for all bars
(policy collapsed — needs ≥ 200 trades in journal + entropy coefficient tuning).

### Algorithm
PPO via stable-baselines3. Trains on completed trade episodes from JSONL journal.

### Action Space (16 actions)
| Actions | Meaning |
|---------|---------|
| 0 | NoTrade |
| 1–5 | Approve at EV threshold 0.55 |
| 6–10 | Approve at EV threshold 0.65 |
| 11–15 | Approve at EV threshold 0.75 |

### 43-dim State Vector
| Dims | Content |
|------|---------|
| 0–5 | ML predictions (p_bull, p_bear, entry_depth, regime_id, sentiment, quality) |
| 6–13 | Market structure (ADX, EMA stack, ATR ratio, BB width, BOS/FVG flags) |
| 14–18 | Session context (asian, london, ny, dead, news_proximity) |
| 19–23 | Signal presence per slot |
| 24–29 | Portfolio state (open_pos, drawdown, daily_pnl, trades_today, last_result, equity_norm) |
| 30–33 | Instrument one-hot (EURUSD, GBPUSD, USDJPY, XAUUSD) |
| 34–41 | ATR history ratios (8 lags: 1,4,8,24,48,96,168,336 bars) |

### Training Requirement
≥ 50 completed trades with `rl_action` and `state_at_entry` in JSONL journal.

---

## Shared Properties

| Property | All models |
|----------|-----------|
| Untrained | Raises `ModelNotTrainedError(RuntimeError)` |
| Silent fallback | None — all failures raise |
| Hot-reload | `reload_if_updated()` checks mtime every 5 min |
| GPU | `torch.amp.autocast("cuda")` + DataParallel (Regime, GRU) |
| Legacy formats | LightGBM/XGBoost pkl support removed |
