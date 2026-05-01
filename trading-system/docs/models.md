# ML Models Reference

Last updated: 2026-05-01. All models live in `trading-engine/models/`. All are PyTorch.

**Critical rule:** All models raise `ModelNotTrainedError(RuntimeError)` when used without
trained weights. No silent fallbacks anywhere. Train all models before setting `ML_ENABLED=true`.

---

## RegimeClassifier (hierarchical dual-cascade)

**File:** `models/regime_classifier.py`
**Weights:** `weights/regime_htf.pkl` (HTF bias) + `weights/regime_ltf.pkl` (LTF behaviour)

### Purpose
Two independent classifiers:
- **HTF classifier (4H)** — "What is the directional bias?" (mode="htf_bias")
- **LTF classifier (1H)** — "How is price behaving right now?" (mode="ltf_behaviour")

Both outputs injected as one-hot context + full probability arrays into every GRU sequence
timestep, and into the RL state vector as full probability arrays (not just the winning class).

### Architecture
Both classifiers share the same MLP structure:
```
N → BatchNorm → FC(128) + residual_skip(N→128) → BN → GELU → Dropout(0.5)
  → FC(64) → BN → GELU → FC(N_CLASSES)
```

### HTF Classes (3-class)
| ID | Name | Description |
|----|------|-------------|
| 0 | BIAS_UP | ADX≥28, EMA stack fully bullish (+2), positive drift, trend efficiency ≥0.40 |
| 1 | BIAS_DOWN | ADX≥28, EMA stack fully bearish (−2), negative drift, trend efficiency ≥0.40 |
| 2 | BIAS_NEUTRAL | ADX≤20, EMA stack near zero, low drift and efficiency, mid-band ATR |

### LTF Classes (4-class)
| ID | Name | Description |
|----|------|-------------|
| 0 | TRENDING | ADX≥28, abs(EMA stack)=2, drift above 60th pctile, efficiency≥0.45 — direction-agnostic |
| 1 | RANGING | ADX≤20, EMA stack=0, near-zero drift, autocorr ∈[−0.15, 0.10], mid-band ATR |
| 2 | CONSOLIDATING | ATR percentile ≤20th AND falling slope (pre-breakout compression) |
| 3 | VOLATILE | ATR percentile ≥80th AND expanding slope |

### Output
```python
# HTF — full probability array exposed
{"regime": "BIAS_UP", "regime_id": 0, "proba": [0.72, 0.18, 0.10], "regime_confidence": 0.72}
# LTF
{"regime": "TRENDING", "regime_id": 0, "proba": [0.61, 0.19, 0.09, 0.11], "regime_confidence": 0.61}
```

### Feature Counts
| Classifier | Features | Source |
|-----------|----------|--------|
| HTF (4H bias) | 34 (`REGIME_4H_FEATURES`) | 4H base + 1D context + regime dynamics + time-series discriminators + 19 macro |
| LTF (1H behaviour) | 18 (`REGIME_1H_FEATURES`) | 1H base + session + BOS/sweep + 4H context + dynamics + time-series discriminators |

### Label Generation
1. **Global GMM** (`fit_global_gmm`) fitted on all symbols combined — guarantees consistent regime semantics across symbols and timeframes
2. **8 GMM features** per bar: efficiency_ratio, rel_vol, drift, compression, vol_slope, atr_pctile, autocorr_lag1, hurst_proxy
3. **Rule-based labels** (`create_rule_labels(return_confidence=True)`) used as training targets with per-bar confidence weights
4. **Ambiguous bars** (confidence < 0.4) dropped before MLP training (REGIME_DROP_AMBIGUOUS=1)
5. **Hysteresis:** 3 consecutive identical predictions before switching current regime at inference

### Training Details
- Loss: hybrid per-bar soft-target CE + entropy regularisation (λ=0.10)
- Class weights: inverse frequency × RANGING 3× boost (LTF), BIAS_NEUTRAL 4× boost (HTF)
- Weight decay: 1e-1 (raised to reduce train/val gap ~0.37 seen in prior training)
- Temporal 80/20 split; early stop patience=10 epochs

### Fixed bugs (2026-05-01)
- `create_labels()` used 5 GMM components but mapped one cluster to class 4 (out of range) — rewritten to 4-component GMM matching `fit_global_gmm` exactly
- Silent `try/except: pass` in `_build_feature_matrix` removed from BOS/sweep feature block
- `predict_batch()` returned default dummy arrays when untrained — now raises `ModelNotTrainedError`
- `_last_mtime=0.0` caused spurious hot-reload 5 min after startup — now set after initial load

---

## GRU-LSTM Predictor

**File:** `models/gru_lstm_predictor.py`
**Weights:** `weights/gru_lstm/model.pt` + `weights/gru_lstm/temperature.pt`

### Purpose
Predicts direction, magnitude, and uncertainty of the next price move from a 30-bar sequence
of **78** features (increased from 74 on 2026-05-01). Receives full 4H and 1H regime probability
arrays at every timestep.

### Architecture (updated 2026-05-01)
```
Input: (batch, 30, 78)
  → GRU(hidden=64, num_layers=2, dropout=0.3, batch_first=True)   → (B, T, 64)
  → Dropout(0.3)
  → LSTM(hidden=128, num_layers=2, dropout=0.3, batch_first=True)  → (B, T=30, 128)
  → MultiheadAttention(embed_dim=128, num_heads=4, dropout=0.1)    → (B, T, 128)
  → mean-pool across T                                              → (B, 128)
  → Dropout(0.3)
  → shared FC(128→64) → ReLU → Dropout(0.3)
  → direction_head: FC(64→1) → sigmoid(logit / temperature) → p_bull
  → magnitude_head: FC(64→1) → ReLU                         → expected_move
  → variance_head:  FC(64→1) → softplus + 1e-6              → expected_variance
```

**Why attention:** Previous architecture used only the last LSTM hidden state, discarding 29 earlier bars. The self-attention layer learns which bars in the window matter most (e.g., a BOS at bar −10, pullback completion at bar −3). Mean-pooling after attention retains information from all attended positions.

### Temperature Scaling
`fit_temperature()` finds scalar T minimising NLL on calibration set → saves `temperature.pt`.
At inference: `p_bull = sigmoid(logit / T)`. Produces calibrated probabilities (P(p_bull > 0.7) actually wins ~70% of the time).

### Loss
`BCE(dir, pos_weight=n_neg/n_pos) + 0.5×SmoothL1(mag) + 0.3×NLL(var)`
Dead-zone bars (|log_return| < 0.3×ATR) masked from BCE.

### Output
```python
{
    "p_bull": 0.72, "p_bear": 0.28,
    "expected_move": 0.0015, "expected_variance": 0.18,
    "expected_volatility": 0.42, "entry_depth": 0.12,
}
```

### 78 SEQUENCE_FEATURES
| Indices | Count | Description |
|---------|-------|-------------|
| 0–15 | 16 | Base 15M bars |
| 16–17 | 2 | 5M context |
| 18–20 | 3 | 1H context |
| 21–23 | 3 | 4H context |
| 24–25 | 2 | 1D context |
| 26–29 | 4 | HTF regime one-hot + conf |
| 30–34 | 5 | LTF regime one-hot + conf |
| 35–37 | 3 | htf_ltf_align, htf_regime_dur, ltf_regime_dur |
| 38–40 | 3 | vol_slope_seq, time_sin, time_cos |
| 41–71 | 31 | ICT structure distances |
| 72–73 | 2 | macro_vix_level, macro_yield_spread |
| 74 | 1 | vwap_dist_atr — (close − VWAP) / ATR [−3, +3] |
| 75 | 1 | volume_delta_pct — buy/sell pressure fraction [−1, +1] |
| 76 | 1 | cum_delta_norm — 20-bar cumulative delta normalised |
| 77 | 1 | wick_auction_ratio — lower_wick / total_wick [0, 1] |

**⚠ Old weights (74 features) are incompatible. Full retrain required after the 2026-05-01 feature additions.**

---

## QualityScorer (EV Regressor)

**File:** `models/quality_scorer.py`  **Weights:** `weights/quality_scorer.pkl`

### Purpose
Predicts expected value (EV) in R-multiples. Runs **post-signal** once `rr_ratio`, `side`,
and `trader_id` are known from PM enrichment — not in the GPU pre-compute cache.

### Architecture
```
20 → BatchNorm → FC(128) → BN → GELU → Dropout(0.3)
   → FC(64)  → BN → GELU → Dropout(0.25)
   → FC(32)  → BN → GELU → Dropout(0.2)
   → FC(1)   [identity output — unbounded float]
```

### Loss
Class-weighted Huber (δ=1.0). `pos_weight = n_neg/n_pos` — allowed < 1.0 when WR > 50% so the
model allocates capacity to distinguishing marginal losers from strong winners.
Win labels normalised by median winner RR so +1.0 = "typical win" (range approximately [−1, +3]).

### Output
```python
{"ev": 0.45, "quality_score": 0.61}  # quality_score = sigmoid(ev)
```

### EV Label Tiers
| Exit reason | EV label |
|-------------|----------|
| `tp2` | `+rr_ratio` (full TP) |
| `tp1` | `+rr × 0.75` |
| `be_or_trail` | `+rr × 0.4` |
| `sl_*` | `−1.0` |
| `time_exit` | `+rr×0.2` if pnl>0, `−0.5` if pnl<0, `0.0` if flat |

### 20 QUALITY_FEATURES (updated from 17 on 2026-05-01)
| Index | Feature | Source |
|-------|---------|--------|
| 0 | strategy_id | trade record |
| 1 | signal_direction | trade record |
| 2 | rr_ratio | PM enrichment |
| 3 | p_bull_gru | ml_model_scores |
| 4 | p_bear_gru | ml_model_scores |
| 5 | regime_class | ml_model_scores (string, encoded) |
| 6 | sentiment_score | ml_model_scores |
| 7 | adx_at_signal | signal_metadata |
| 8 | atr_ratio_at_signal | signal_metadata |
| 9 | volume_ratio | signal_metadata |
| 10 | spread_at_signal | signal_metadata |
| 11 | session_at_signal | trade record |
| 12 | news_in_30min | signal_metadata |
| 13 | strategy_win_rate_20 | rolling causal (prior trades only) |
| 14 | gru_uncertainty | ml_model_scores (expected_variance) |
| 15 | regime_duration | ml_model_scores |
| 16 | vol_slope_at_signal | ml_model_scores |
| 17 | strategy_win_rate_5 | rolling causal — short-term momentum |
| 18 | strategy_win_rate_50 | rolling causal — long-term baseline |
| 19 | gru_signal_agreement | 1.0 if GRU direction agrees with trade side |

All win-rate features use **prior trades only** (strict no-lookahead).

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

---

## RLAgent (PPO)

**File:** `models/rl_agent.py`  **Weights:** `weights/rl_ppo/model.zip`

### Purpose
Dynamically selects confidence threshold per bar. **Fully integrated into live signal pipeline
as of 2026-05-01** — `decide()` is called before `_compute_ml_signal()` on every bar.

### Algorithm
PPO via stable-baselines3, CPU device (`MlpPolicy` runs faster on CPU than GPU for small nets).
Trains from `trade_journal_detailed.jsonl` episodes.

### Public API
```python
# Primary entry point — use this in the pipeline
trader_id, threshold = rl_agent.decide(state_43, available_signals, session)
# trader_id=0 → NoTrade; trader_id=1 → proceed with threshold

# PPO-only path — raises ModelNotTrainedError if untrained
trader_id, threshold = rl_agent.select_action(state_43, available_signals)
```

### Action Space (9 actions — v3 threshold-only)
| Action | Threshold | When appropriate |
|--------|-----------|-----------------|
| 0 | NoTrade | Drawdown, CONSOLIDATING, low-IC regime |
| 1 | 0.60 | Strong trend + sustained positive IC |
| 2 | 0.62 | Default London/NY high-confidence |
| 3 | 0.65 | Moderate trending conditions |
| 4 | 0.68 | Mixed HTF/LTF signals |
| 5 | 0.70 | Conservative baseline |
| 6 | 0.72 | Cautious — weakening trend |
| 7 | 0.75 | High selectivity |
| 8 | 0.80 | Volatile or ranging — very conservative |

### 43-dim State Vector (`N_STATE=43`)
Built by `SignalPipeline._build_rl_state(ml_preds, bar, portfolio)`:

| Dims | Feature |
|------|---------|
| 0 | p_bull (GRU) |
| 1 | p_bear (GRU) |
| 2 | expected_variance (GRU uncertainty) |
| 3–5 | HTF regime proba [BIAS_UP, BIAS_DOWN, BIAS_NEUTRAL] |
| 6–9 | LTF regime proba [TRENDING, RANGING, CONSOLIDATING, VOLATILE] |
| 10 | sentiment_score [−1, 1] |
| 11 | sentiment_confidence [0, 1] |
| 12–19 | ATR history ratios — 8 lags (1,4,8,24,48,96,168,336) |
| 20 | spread_pips / 5.0 |
| 21 | sin(2π × hour / 24) |
| 22 | cos(2π × hour / 24) |
| 23 | session_enc (0=INACTIVE, 1/3=ASIAN, 2/3=LONDON, 1=NY) |
| 24 | rolling 10-trade win rate [0, 1] |
| 25 | drawdown_pct / 0.20 |
| 26 | trades_today / 5.0 |
| 27 | open_positions / 5.0 |
| 28 | daily_pnl / equity clipped [−0.10, 0.10] |
| 29 | adx_14 / 50.0 |
| 30 | vol_slope [−1, 1] |
| 31 | macro_vix_level [0, 1] |
| 32 | macro_yield_spread / 0.02 [−1, 1] |
| 33 | vwap_dist_atr / 3.0 [−1, 1] |
| 34 | volume_delta_pct [−1, 1] |
| 35 | wick_auction_ratio [0, 1] |
| 36 | cum_delta_norm [−1, 1] |
| 37 | ema_stack / 2.0 [−1, 1] |
| 38 | HTF regime confidence (max proba) |
| 39 | LTF regime confidence (max proba) |
| 40 | atr_ratio / 3.0 |
| 41 | volume_ratio / 3.0 |
| 42 | max(p_bull, p_bear) × 2 − 1 (direction strength) |

### Training Requirement
≥ 50 completed trades with `rl_action` and `state_at_entry` (length=43) in JSONL journal.
Every approved signal now carries both fields — old journal records with `state_at_entry=[0.0]*42` are excluded by the length check.

### Reward Function
```
total = pnl_reward        + sharpe_bonus  + dd_penalty  + overtrade_pen  + session_bonus  + inaction_pen
      = clip(r_mult,−3,4) + clip(SR×0.3)  + −2×max(0,DD−0.05)  + −0.3×max(0,trades_today−3)
        + 0.1 if(London/NY and pnl>0)    + −0.05 if(action=0 and missed_setup)
clip(total, −5, 6)
```

---

## BaseModel (hot-reload base class)

**File:** `models/base_model.py`

All models inherit from `BaseModel`. Provides:
- `is_trained` property: checks weight file exists and is non-empty
- `reload_if_updated()`: checks mtime every 5 min; reloads on file change; **re-raises on failure** (no silent catch)
- `_last_mtime`: now set by subclasses after initial load — prevents spurious reload on first 5-min check

---

## VectorStore (FAISS)

**File:** `models/vector_store.py`  **Index:** `weights/gru_lstm/vector_store/`

GRU-LSTM produces a 64-dim shared-layer embedding (`encode()`). After training,
`retrain_incremental.py` bulk-indexes trade embeddings into a FAISS flat index.
Useful for finding historical bars similar to current market state.

---

## Shared Properties

| Property | All models |
|----------|-----------|
| Untrained | Raises `ModelNotTrainedError(RuntimeError)` |
| Silent fallback | None — all failures raise |
| Hot-reload | `reload_if_updated()` checks mtime every 5 min, re-raises on failure |
| GPU | `torch.amp.autocast("cuda")` + DataParallel (Regime, GRU only) |
| RL device | CPU always (PPO MlpPolicy) |
| Pydantic pins | `pydantic==2.7.4`, `pydantic-core==2.18.4`, `pydantic-settings==2.3.4` |
