# System Architecture

Last updated: 2026-04-24.

## What This System Is

A ML-native forex/gold trading system. Five ICT rule-based traders have been removed.
All signal generation is driven by a single unified ML pipeline. ICT concepts
(BOS, FVG, sweep, EMA pullback, Asian range) are **features** fed to the GRU — the
models decide whether and where to trade.

Runs on Kaggle (2× Tesla T4 GPUs) for offline training and backtesting.
Live trading engine (`main.py`) is functional — `signal_pipeline.py` mirrors
`run_backtest._compute_backtest_signal` exactly and is the live inference path.

---

## Pipeline — Bar by Bar (Backtest / Working Path)

```
processed_data/histdata/{SYMBOL}_{TF}.parquet
  └─ run_backtest._backtest_trader("ml_trader", all_symbols)
       └─ _precompute_ml_cache(df, symbol, htf, ml_models)
            │
            ├─ Step 1: RegimeClassifier HTF (4H bias, 3-class)
            │    Input:  34 REGIME_4H_FEATURES from 4H + 1D bars
            │    Output: htf_bias (0=BIAS_UP, 1=BIAS_DOWN, 2=BIAS_NEUTRAL) + htf_conf
            │
            ├─ Step 2: RegimeClassifier LTF (1H behaviour, 4-class)
            │    Input:  18 REGIME_1H_FEATURES from 1H + 4H bars
            │    Output: ltf_behaviour (0=TRENDING, 1=RANGING, 2=CONSOLIDATING, 3=VOLATILE) + ltf_conf
            │
            ├─ Step 3: Build sequence features
            │    74 SEQUENCE_FEATURES including:
            │      - htf_bias_up/down/neutral (indices 26–28) + htf_bias_conf (29)
            │      - ltf_trending/ranging/consolidating/volatile (30–33) + ltf_conf (34)
            │      - htf_ltf_align (35), htf_regime_dur (36), ltf_regime_dur (37)
            │      - ICT structure distances (BOS age/strength, FVG dist/fill,
            │        sweep wick depth, Asian range context)
            │    Regime BEFORE sequence build — order is load-bearing.
            │
            └─ Step 4: GRU-LSTM (batched 1024 sequences/batch)
                 Input:  (N, 30, 74) sliding window
                 Output: p_bull, p_bear, expected_move, expected_variance
                 Temperature scaling: p_bull = sigmoid(logit / T) where T loaded from temperature.pt

       └─ Bar loop (pure dict lookup + gate evaluation):
            ├─ Gate 1: expected_variance > MAX_UNCERTAINTY (env default 2.0) → skip
            ├─ Gate 2: max(p_bull, p_bear) < ML_DIRECTION_THRESHOLD (0.58) → skip
            ├─ Gate 3: HTF bias misaligns with side → skip
            │    BIAS_UP + sell → skip; BIAS_DOWN + buy → skip
            │    BIAS_NEUTRAL: require conf ≥ NEUTRAL_BIAS_THRESHOLD (0.58)
            ├─ Gate 4: LTF behaviour filter
            │    CONSOLIDATING → blocked (if BLOCK_LTF_CONSOLIDATING=1)
            │    VOLATILE → require conf ≥ VOLATILE_ENTRY_THRESHOLD
            │    RANGING → optional range boundary check (RANGING_REQUIRE_RANGE)
            │    TRENDING → optional pullback filter (REQUIRE_TRENDING_PULLBACK)
            ├─ Cooldown, dead zone 12:00–13:00 UTC, daily loss, drawdown halt → skip
            ├─ ATR-based entry/SL/TP levels (RANGING uses range walls when valid)
            ├─ PM enrichment: size, TP1/TP2, correlation cap
            ├─ QualityScorer (runs here — needs actual rr_ratio + trader_id)
            │    Input:  17 QUALITY_FEATURES (includes rr_ratio, p_bull/bear, regime, etc.)
            │    Output: ev (R-multiples), quality_score
            ├─ Gate 5: ev < MIN_EV_THRESHOLD (0.10) → skip
            └─ Simulate trade → append to trade_log → write to journal
```

---

## ML Models In Depth

### 1. RegimeClassifier (hierarchical dual-cascade)

**File:** `models/regime_classifier.py`
**Weights:** `weights/regime_htf.pkl` (HTF bias) + `weights/regime_ltf.pkl` (LTF behaviour)

Two independent MLP classifiers, architecture: `N → 128 → 64 → N_CLASSES` with BatchNorm, GELU, residual skip, Dropout 0.5.

**HTF Classifier — "What is the directional bias?" (3 classes)**
- `0=BIAS_UP`, `1=BIAS_DOWN`, `2=BIAS_NEUTRAL`
- Answers macro trend direction only. Simple, clean, high-accuracy target.
- Architecture: `34 → 128 → 64 → 3`
- Accuracy target: ≥65% (random baseline = 33%)

**4H Bias Classifier (34 features):**
- 4H base: ADX, EMA stack, ATR ratio, BB width, realised vol (5)
- 1D context: ADX, EMA stack, ATR ratio, BB width (4)
- Dynamics: vol_slope, regime_duration, ATR percentile (3)
- Time-series discriminators: efficiency_ratio, autocorr_lag1, hurst_proxy (3)
- Macro: 17 index returns + VIX level + yield spread (19)
- Trained on 4H data only; mode="htf_bias"

**LTF Classifier — "How is price behaving right now?" (4 classes)**
- `0=TRENDING`, `1=RANGING`, `2=CONSOLIDATING`, `3=VOLATILE`
- Direction-agnostic intraday price behaviour. Trained on 1H data only.
- Architecture: `18 → 128 → 64 → 4`
- Accuracy target: ≥55% (random baseline = 25%)

**1H Behaviour Classifier (18 features):**
- 1H base: ADX, EMA stack, ATR ratio, BB width, realised vol (5)
- Session: session_code (1)
- Market structure: swing HH/HL count, liquidity sweep count (2)
- 4H context: ADX, EMA stack, ATR ratio, BB width (4)
- Dynamics: vol_slope, regime_duration, ATR percentile (3)
- Time-series discriminators: efficiency_ratio, autocorr_lag1, hurst_proxy (3)
- Trained on 1H-native labels; mode="ltf_behaviour"

**GRU receives both classifiers at every timestep:**
```python
# Slots 26–37 of SEQUENCE_FEATURES:
htf_bias_up, htf_bias_down, htf_bias_neutral, htf_bias_conf,  # HTF 3-class (indices 26–29)
ltf_trending, ltf_ranging, ltf_consolidating, ltf_volatile, ltf_conf,  # LTF 4-class (30–34)
htf_ltf_align, htf_regime_dur, ltf_regime_dur  # alignment + duration (35–37)
# Example: HTF=BIAS_UP + LTF=CONSOLIDATING → expect breakout entry
```

**Hysteresis:** 3 consecutive bars required to switch regime at inference time.

---

### 2. GRU-LSTM Predictor

**File:** `models/gru_lstm_predictor.py`
**Weights:** `weights/gru_lstm/model.pt` + `weights/gru_lstm/temperature.pt`
**Sequence length:** 30 bars × 74 features

```
Input: (batch, 30, 74)
  → GRU(hidden=64, num_layers=2, dropout=0.3, batch_first=True)
  → Dropout(0.3)
  → LSTM(hidden=128, num_layers=2, dropout=0.3, batch_first=True)
  → Dropout(0.3) on last timestep
  → shared FC(128→64) → ReLU → Dropout(0.3)
  → direction_head: FC(64→1) → sigmoid(logit / T)  → p_bull   [T from temperature.pt]
  → magnitude_head: FC(64→1) → ReLU                → expected_move
  → variance_head:  FC(64→1) → softplus + 1e-6      → expected_variance
```

**Loss:** `BCE(dir, pos_weight) + 0.5×SmoothL1(mag) + 0.3×NLL(var)`.
Dead-zone bars (|log_return| < 0.3×ATR) set direction label to NaN and are masked from BCE.

**Temperature scaling:** `fit_temperature()` minimises NLL on calibration set and saves `temperature.pt`. Loaded at inference and applied as `sigmoid(logit / T)`.

**Output:**
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

**74 SEQUENCE_FEATURES per timestep:**
- 0–15: Base 15M (log_return, HL range, close_vs_open, ATR, RSI, EMA21/50 dist, BB pos, vol ratio, session flags, BOS flags, FVG flags)
- 16–17: 5M (RSI, EMA21 dist)
- 18–20: 1H (ADX, EMA21 dist, EMA50 dist)
- 21–23: 4H (EMA21-50 diff, ADX, RSI)
- 24–25: 1D (EMA21 dist, EMA stack)
- 26–28: HTF bias one-hot (BIAS_UP, BIAS_DOWN, BIAS_NEUTRAL)
- 29: htf_bias_conf
- 30–33: LTF behaviour one-hot (TRENDING, RANGING, CONSOLIDATING, VOLATILE)
- 34: ltf_conf
- 35–37: htf_ltf_align, htf_regime_dur, ltf_regime_dur
- 38–40: vol_slope_seq, time_sin, time_cos
- 41–71: ICT structure distances (EMA pullback zone, BOS age/strength, FVG dist/fill, sweep wick, Asian range, candle body/wicks, oscillators, ADX, regime dynamics, session timing)
- 72–73: macro_vix_level, macro_yield_spread

**Batched inference in backtest:** `_precompute_ml_cache` builds full `(N, 30, 74)` sliding windows in batches of 1024. Regime inference runs first so regime context is available at each timestep.

---

### 3. QualityScorer (EV Regressor)

**File:** `models/quality_scorer.py`
**Weights:** `weights/quality_scorer.pkl`

```
17 → BatchNorm → FC(64) → BN → GELU → Dropout(0.3)
   → FC(32) → BN → GELU → Dropout(0.2)
   → FC(1)  [identity output — unbounded float]
```

**Loss:** Class-weighted Huber(δ=1.0). Winners up-weighted by `n_neg/n_pos` (capped 8×).

**Called post-signal** (after PM enrichment gives actual `rr_ratio`). Not in the GPU cache.

**Output:**
```python
{"ev": 0.45, "quality_score": 0.61}  # quality_score = sigmoid(ev)
```

**EV Label tiers (tiered, from `signal_metadata` fields):**
- `tp2` → `+rr_ratio`
- `tp1` → `+rr × 0.75`
- `be_or_trail` → `+rr × 0.4`
- `sl_*` → `-1.0`
- `time_exit` → `+rr×0.2` (win), `-0.5` (loss), `0.0` (flat)

Reads `adx_at_signal`, `atr_ratio_at_signal`, `volume_ratio`, `spread_at_signal`,
`news_in_30min` from real `signal_metadata` fields — not hardcoded constants.

**17 QUALITY_FEATURES:**
strategy_id, signal_direction, rr_ratio, p_bull_gru, p_bear_gru, regime_class, sentiment_score, adx_at_signal, atr_ratio_at_signal, volume_ratio, spread_at_signal, session_at_signal, news_in_30min, strategy_win_rate_20, gru_uncertainty, regime_duration, vol_slope_at_signal

---

### 4. SentimentModel

**File:** `models/sentiment_model.py` **Weights:** pre-trained (no local file)

Backends: **FinBERT** (`ProsusAI/finbert`) primary → **VADER** fallback.

Output: `{"sentiment_score": float, "sentiment_label": str, "sentiment_confidence": float}`
Gold: USD bullish → XAUUSD bearish (inverted).

---

### 5. RLAgent (PPO)

**File:** `models/rl_agent.py` **Weights:** `weights/rl_ppo/model.zip`

PPO via stable-baselines3. **Training requires ≥ 50 completed trades** with `rl_action` and `state_at_entry` in JSONL journal.

**43-dim state vector:**
- 0–5: ML predictions (p_bull, p_bear, entry_depth, regime_id, sentiment, quality)
- 6–13: Market structure (ADX, EMA stack, ATR ratio, BB width, BOS, FVG flags)
- 14–18: Session (asian, london, ny, dead, news_proximity)
- 19–23: Signal presence per slot (5 dims)
- 24–29: Portfolio state (open_pos, drawdown, daily_pnl, trades_today, last_result, equity_norm)
- 30–33: Instrument one-hot (EURUSD, GBPUSD, USDJPY, XAUUSD)
- 34–41: ATR history ratios (8 lags)

**16 actions:** 0=NoTrade, 1–5=Approve at 0.55 EV, 6–10=Approve at 0.65 EV, 11–15=Approve at 0.75 EV

---

## Backtest Architecture

All GPU inference is batched upfront per symbol in `_precompute_ml_cache`. The bar loop does only dict lookups and gate checks.

**Execution order in `_precompute_ml_cache`:**
1. `RegimeClassifier._build_feature_matrix()` → batch 4H + 1H → `regime_preds` + confidence series
2. `_build_sequence_df()` with regime series injected at each timestep → `(N, 74)` matrix
3. GRU sliding-window batch inference (1024/batch) → `gru_preds`
4. Merge into `cache[bar_idx]` dict

QualityScorer runs **outside** the cache, per signal, after PM enrichment.

---

## Causal Integrity

All features are strictly backward-looking.

| Feature | Status |
|---------|--------|
| `sr_dist_*`, `sr_in_*`, `sr_*_strength` (REGIME_FEATURES 28–33) | Zeroed — `detect_sr_zones` uses `rolling(center=True)` |
| `swing_hh_hl_count`, `liquidity_sweep_24h` in REGIME_FEATURES | Used in 1H classifier (live bar OK); zeroed in training if needed |
| `macro bfill()` | Removed — replaced with `fillna(0.0)` only |
| All rolling indicators | Backward-only |
| HTF `reindex(method="ffill")` | Causal — only HTF bars ≤ t contribute |

---

## Feature Contracts

Feature list order is a hard contract — changing order or count breaks saved model weights.

| List | Length | Used by |
|------|--------|---------|
| `SEQUENCE_FEATURES` | 74 | GRU training + inference |
| `REGIME_4H_FEATURES` | 34 | RegimeClassifier (4H bias) |
| `REGIME_1H_FEATURES` | 18 | RegimeClassifier (1H structure) |
| `QUALITY_FEATURES` | 17 | QualityScorer |
| `RL_STATE_DIM` | 43 | RLAgent |

---

## Error Policy

No silent fallbacks anywhere. Every failure raises and propagates.

- Untrained models: raise `ModelNotTrainedError(RuntimeError)`
- Missing `ev` key when QualityScorer loaded: raise `RuntimeError`
- `_precompute_ml_cache` failures: raise (no fallback to empty cache)
- Feature computation failures: raise (no zero-filled arrays)

---

## Known Issues

| Issue | File | Impact |
|-------|------|--------|
| RL policy exploration | `models/rl_agent.py` | Needs ≥200 journal trades + entropy coefficient tuning for meaningful action diversity |
| Regime accuracy | `models/regime_classifier.py` | Target ≥65% (HTF), ≥55% (LTF). atr_pctile bug fixed 2026-04-24 — retrain expected to improve LTF RANGING accuracy |
