# ML Models Reference

Last updated: 2026-04-24. All models live in `trading-engine/models/`. All are PyTorch.

**Critical rule:** All models raise `ModelNotTrainedError(RuntimeError)` when used without
trained weights. No silent fallbacks. Train all models before setting `ML_ENABLED=true`.

---

## RegimeClassifier (hierarchical dual-cascade)

**File:** `models/regime_classifier.py`
**Weights:** `weights/regime_htf.pkl` (HTF bias) + `weights/regime_ltf.pkl` (LTF behaviour)

### Purpose
Two independent classifiers:
- **HTF classifier (4H)** — "What is the directional bias?" (mode="htf_bias")
- **LTF classifier (1H)** — "How is price behaving right now?" (mode="ltf_behaviour")

Both outputs injected as one-hot context into every GRU sequence timestep.

### Architecture
Both classifiers share the same MLP structure:
```
N → BatchNorm → FC(128) + residual_skip(128) → BN → GELU → Dropout(0.5)
  → FC(64) → BN → GELU → FC(N_CLASSES)
```

### HTF Classes (3-class)
| ID | Name | Description |
|----|------|-------------|
| 0 | BIAS_UP | ADX>25, EMA stack aligned up, positive drift |
| 1 | BIAS_DOWN | ADX>25, EMA stack aligned down, negative drift |
| 2 | BIAS_NEUTRAL | Everything else — low ADX or indecisive drift |

### LTF Classes (4-class)
| ID | Name | Description |
|----|------|-------------|
| 0 | TRENDING | ADX>25, directional EMA stack, clear drift (direction-agnostic) |
| 1 | RANGING | Low ADX, mid-band ATR percentile, near-zero drift |
| 2 | CONSOLIDATING | ATR at multi-period low and falling (pre-breakout compression) |
| 3 | VOLATILE | ATR percentile at or above 80th percentile |

### Output
```python
# HTF
{"regime": "BIAS_UP", "regime_id": 0, "proba": [0.7, 0.2, 0.1], "regime_confidence": 0.7}
# LTF
{"regime": "TRENDING", "regime_id": 0, "proba": [0.6, 0.2, 0.1, 0.1], "regime_confidence": 0.6}
```

### Feature Counts
| Classifier | Features | Source |
|-----------|----------|--------|
| HTF (4H bias) | 34 (`REGIME_4H_FEATURES`) | 4H base + 1D context + regime dynamics + time-series discriminators + 19 macro |
| LTF (1H behaviour) | 18 (`REGIME_1H_FEATURES`) | 1H base + session + BOS/sweep + 4H context + dynamics + time-series discriminators |

### Label Generation
Global GMM (`fit_global_gmm`) fitted on all symbols combined:
- HTF: 3-component GMM, assigned by drift direction (BIAS_UP/DOWN/NEUTRAL)
- LTF: 4-component GMM, assigned by behaviour profile (VOLATILE → TRENDING → CONSOLIDATING → RANGING)
- 8 GMM features per bar: efficiency_ratio, rel_vol, drift, compression, vol_slope, atr_pctile, autocorr_lag1, hurst_proxy
- Rule-based labels (`create_rule_labels`) used as training targets with per-bar confidence weights
- Ambiguous bars (confidence < 0.4) get soft targets — MLP learns uncertainty rather than hard labels
- Hysteresis: 3 consecutive bars required to switch regime at inference time

### Fixed bugs
- `atr_pctile` (col 36 of full REGIME_FEATURES matrix, col 14 of 1H, col 11 of 4H) was always zero before 2026-04-24 fix.

---

## GRU-LSTM Predictor

**File:** `models/gru_lstm_predictor.py`  **Weights:** `weights/gru_lstm/model.pt` + `weights/gru_lstm/temperature.pt`

### Purpose
Predicts direction, magnitude, and uncertainty of the next price move from a
30-bar sequence of 74 features. Receives 4H and 1H regime context at every timestep.

### Architecture
```
Input: (batch, 30, 74)
  → GRU(hidden=64, num_layers=2, dropout=0.3, batch_first=True)
  → Dropout(0.3)
  → LSTM(hidden=128, num_layers=2, dropout=0.3, batch_first=True)
  → Dropout(0.3) on last timestep output
  → shared FC(128→64) → ReLU → Dropout(0.3)
  → direction_head: FC(64→1) → sigmoid(logit / temperature) → p_bull
  → magnitude_head: FC(64→1) → ReLU                         → expected_move
  → variance_head:  FC(64→1) → softplus + 1e-6              → expected_variance
```

### Temperature Scaling
After training, `fit_temperature()` finds a scalar T that minimises BCE NLL on the
calibration set, then saves `temperature.pt` alongside `model.pt`. At inference,
`p_bull = sigmoid(logit / T)`. This is post-hoc calibration — does not change weights.

### Loss
`BCE(dir, pos_weight=n_neg/n_pos) + 0.5×SmoothL1(mag) + 0.3×NLL(var)`.
Dead-zone bars (|log_return| < 0.3×ATR) have NaN direction labels and are masked
out of the BCE term.

### Output
```python
{
    "p_bull": 0.72,
    "p_bear": 0.28,
    "expected_move": 0.0015,
    "expected_variance": 0.3,
    "expected_volatility": 0.55,
    "entry_depth": 0.15,
}
```

### Regime Conditioning
HTF one-hot (BIAS_UP/DOWN/NEUTRAL, 3 dims) + htf_conf, and LTF one-hot
(TRENDING/RANGING/CONSOLIDATING/VOLATILE, 4 dims) + ltf_conf are injected at every
sequence timestep. Regime inference runs BEFORE GRU sequence building — order is load-bearing.

### 74 SEQUENCE_FEATURES per Timestep
| Indices | Count | Description |
|---------|-------|-------------|
| 0–15 | 16 | Base 15M: log_return, HL range, close_vs_open, ATR_norm, RSI, EMA21/50 dist, BB pos, vol ratio, session flags, BOS flags, FVG flags |
| 16–17 | 2 | 5M: RSI, EMA21 dist |
| 18–20 | 3 | 1H: ADX, EMA21 dist, EMA50 dist |
| 21–23 | 3 | 4H: EMA21-50 diff/close, ADX, RSI |
| 24–25 | 2 | 1D: EMA21 dist, EMA stack score |
| 26–28 | 3 | HTF bias one-hot (BIAS_UP, BIAS_DOWN, BIAS_NEUTRAL) |
| 29 | 1 | htf_bias_conf |
| 30–33 | 4 | LTF behaviour one-hot (TRENDING, RANGING, CONSOLIDATING, VOLATILE) |
| 34 | 1 | ltf_conf |
| 35–37 | 3 | htf_ltf_align, htf_regime_dur, ltf_regime_dur |
| 38–40 | 3 | vol_slope_seq, time_sin, time_cos |
| 41–71 | 31 | ICT structure distances: EMA pullback zone, BOS age/strength, FVG dist/fill, sweep wick depth, Asian range context, candle body/wicks, oscillators, ADX, regime dynamics, session timing |
| 72–73 | 2 | macro_vix_level, macro_yield_spread |

---

## QualityScorer (EV Regressor)

**File:** `models/quality_scorer.py`  **Weights:** `weights/quality_scorer.pkl`

### Purpose
Predicts expected value (EV) of a trade in R-multiples. Runs **post-signal** once the
actual `rr_ratio`, `side`, and `trader_id` are known — not in the GPU pre-compute cache.

### Architecture
```
17 → BatchNorm → FC(64) → BN → GELU → Dropout(0.3)
   → FC(32) → BN → GELU → Dropout(0.2)
   → FC(1)  [identity output — unbounded float]
```
Internal `N_FEATURES = 14` is a legacy constant; actual feature count is determined at
training time from `len(QUALITY_FEATURES)` = 17.

### Loss
Class-weighted Huber (δ=1.0) — positive trades boosted by `n_neg/n_pos` ratio (capped at 8×).

### Output
```python
{"ev": 0.45, "quality_score": 0.61}  # quality_score = sigmoid(ev)
```

### EV Label Tiers (from `_compute_ev_label`)
| Exit reason | EV label |
|-------------|----------|
| `tp2` | `+rr_ratio` (full TP) |
| `tp1` | `+rr × 0.75` |
| `be_or_trail` | `+rr × 0.4` |
| `sl_*` | `-1.0` |
| `time_exit` | `+rr×0.2` if pnl>0, `-0.5` if pnl<0, `0.0` if flat |
| Other | excluded |

Labels read real `signal_metadata` fields (adx_at_signal, atr_ratio_at_signal, volume_ratio,
spread_at_signal, news_in_30min) from `trade_journal_detailed.jsonl` — no hardcoded constants.

### 17 QUALITY_FEATURES
| Index | Feature |
|-------|---------|
| 0 | strategy_id |
| 1 | signal_direction (1=buy, 0=sell) |
| 2 | rr_ratio |
| 3 | p_bull_gru |
| 4 | p_bear_gru |
| 5 | regime_class (string, encoded) |
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
Selects selectivity tier (EV threshold). Trains on completed trade episodes from JSONL journal.

### Algorithm
PPO via stable-baselines3, CPU device. Trains from `trade_journal_detailed.jsonl`.

### Action Space (16 actions)
| Actions | Meaning |
|---------|---------|
| 0 | NoTrade |
| 1–5 | Approve at EV threshold 0.55 |
| 6–10 | Approve at EV threshold 0.65 |
| 11–15 | Approve at EV threshold 0.75 |

### 43-dim State Vector (`RL_STATE_DIM=43`)
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

## VectorStore (FAISS)

**File:** `models/vector_store.py`  **Index:** `weights/gru_lstm/vector_store/`

GRU-LSTM produces a 64-dim shared-layer embedding (`get_embedding()`). After training,
`retrain_incremental.py` bulk-indexes trade embeddings into a FAISS flat index for similarity
search. Useful for finding historical bars that look like the current market state.

---

## Shared Properties

| Property | All models |
|----------|-----------|
| Untrained | Raises `ModelNotTrainedError(RuntimeError)` |
| Silent fallback | None — all failures raise |
| Hot-reload | `reload_if_updated()` checks mtime every 5 min |
| GPU | `torch.amp.autocast("cuda")` + DataParallel (Regime, GRU) |
| Pydantic pins | `pydantic==2.7.4`, `pydantic-core==2.18.4`, `pydantic-settings==2.3.4` (Kaggle compat) |
