# System Architecture

Last updated: 2026-05-01.

## What This System Is

A ML-native forex/gold trading system. Five ICT rule-based traders have been removed.
All signal generation is driven by a single unified ML pipeline. ICT concepts
(BOS, FVG, sweep, EMA pullback, Asian range, VWAP, volume delta, wick auction) are
**features** fed to the GRU — the models decide whether and where to trade.

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
            │    Output: htf_bias (0=BIAS_UP, 1=BIAS_DOWN, 2=BIAS_NEUTRAL) + htf_proba[3]
            │
            ├─ Step 2: RegimeClassifier LTF (1H behaviour, 4-class)
            │    Input:  18 REGIME_1H_FEATURES from 1H + 4H bars
            │    Output: ltf_behaviour (0=TRENDING, 1=RANGING, 2=CONSOLIDATING, 3=VOLATILE) + ltf_proba[4]
            │
            ├─ Step 3: Build sequence features
            │    78 SEQUENCE_FEATURES including:
            │      - htf_bias_up/down/neutral (indices 26–28) + htf_bias_conf (29)
            │      - ltf_trending/ranging/consolidating/volatile (30–33) + ltf_conf (34)
            │      - htf_ltf_align (35), htf_regime_dur (36), ltf_regime_dur (37)
            │      - ICT structure distances (BOS age/strength, FVG dist/fill,
            │        sweep wick depth, Asian range context)
            │      - Institutional structure (74): vwap_dist_atr
            │      - Institutional structure (75): volume_delta_pct
            │      - Institutional structure (76): cum_delta_norm
            │      - Institutional structure (77): wick_auction_ratio
            │    Regime BEFORE sequence build — order is load-bearing.
            │
            └─ Step 4: GRU-LSTM (batched 1024 sequences/batch)
                 Input:  (N, 30, 78) sliding window
                 Output: p_bull, p_bear, expected_move, expected_variance
                 Self-attention: MultiheadAttention(4 heads) over all 30 LSTM outputs → mean-pool
                 Temperature scaling: p_bull = sigmoid(logit / T) where T loaded from temperature.pt

       └─ Bar loop (pure dict lookup + gate evaluation):
            ├─ Gate 1: expected_variance > MAX_UNCERTAINTY (env default 0.25) → skip
            ├─ Gate 2: RLAgent.decide() → (trader_id, threshold)
            │    trader_id=0 → NoTrade immediately
            │    threshold: RL-selected from [0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.80]
            ├─ Gate 3: max(p_bull, p_bear) < rl_threshold → skip
            ├─ Gate 4: HTF bias misaligns with side → skip
            │    BIAS_UP + sell → skip; BIAS_DOWN + buy → skip
            │    BIAS_NEUTRAL: require conf ≥ NEUTRAL_BIAS_THRESHOLD (0.60)
            ├─ Gate 5: LTF behaviour filter
            │    CONSOLIDATING → blocked (if BLOCK_LTF_CONSOLIDATING=1)
            │    VOLATILE → require conf ≥ VOLATILE_ENTRY_THRESHOLD (0.70)
            │    RANGING → optional range boundary check (RANGING_REQUIRE_RANGE)
            │    TRENDING → optional pullback filter (REQUIRE_TRENDING_PULLBACK)
            ├─ Dead zone 12:00–13:00 UTC, daily loss, drawdown halt → skip
            ├─ ATR-based entry/SL/TP levels (RANGING uses range walls when valid)
            ├─ Probability-weighted E[R] gate: p_win × RR − (1−p_win) ≥ MIN_EXPECTED_R (1.30)
            ├─ PM enrichment: size, TP1/TP2, correlation cap
            ├─ QualityScorer (runs here — needs actual rr_ratio + trader_id)
            │    Input:  20 QUALITY_FEATURES (includes rr_ratio, p_bull/bear, regime, etc.)
            │    Output: ev (R-multiples), quality_score
            ├─ Gate 6: ev < MIN_EV_THRESHOLD (0.10) → skip
            └─ Simulate trade → append to trade_log → write to journal
                 Signal carries: rl_action, rl_threshold, state_at_entry[43]
```

---

## Live Trading Architecture

```
Capital.com REST API → DataFetcher → MARKET_DATA (Redis)
        │
        ▼
ProductionTradingEngine.on_market_data(portfolio=current_state)
        │
        ▼
SignalPipeline.process_bar(symbol, df, df_htf, portfolio)
        ├─ _run_ml_inference() → ml_preds
        │    ├─ GRU-LSTM: p_bull, p_bear, expected_variance
        │    ├─ RegimeClassifier HTF: htf_bias + proba[3]
        │    ├─ RegimeClassifier LTF: ltf_behaviour + proba[4]
        │    └─ SentimentModel: sentiment_score, sentiment_label
        │
        ├─ Dead zone 12:00–13:00 UTC → []
        │
        ├─ _build_rl_state(ml_preds, bar, portfolio) → state[43]
        ├─ RLAgent.decide(state, available_signals, session)
        │    → (trader_id=0, 0.0) ── NoTrade: return []
        │    → (trader_id=1, threshold) ── continue with RL threshold
        │
        ├─ _compute_ml_signal(threshold=rl_threshold)
        │    Gate: uncertainty, rl_threshold, HTF align, LTF behaviour, E[R]
        │
        └─ Approved signal:
             {side, entry, sl, tp, confidence, rl_action, rl_threshold, state_at_entry[43]}
        │
        ▼
RiskEngine.check_pre_trade() → allowed
        │
        ▼
PortfolioManager.enrich_signal() → size (fixed-fractional: base × vol_scalar)
  vol_scalar = clip(nominal_ATR / current_ATR, 0.5, 1.0)  ← cap at 1.0, never increases size
        │
        ▼
ExecutionEngine → PaperTradingService / BrokerConnector
        │
        ▼
TradeJournal → CSV + JSONL (records rl_action + state_at_entry for RL training)
```

---

## ML Models In Depth

### 1. RegimeClassifier (hierarchical dual-cascade)

**File:** `models/regime_classifier.py`
**Weights:** `weights/regime_htf.pkl` (HTF bias) + `weights/regime_ltf.pkl` (LTF behaviour)

Architecture: `N → BatchNorm → FC(128) + residual_skip → BN → GELU → Dropout(0.5) → FC(64) → BN → GELU → FC(N_CLASSES)`

**HTF Classifier — "What is the directional bias?" (3 classes)**
- `0=BIAS_UP`, `1=BIAS_DOWN`, `2=BIAS_NEUTRAL`
- Architecture: `34 → 128 → 64 → 3`
- Accuracy target: ≥65% (random baseline = 33%)
- Outputs full `proba[3]` list — all 3 probabilities injected into RL state vector

**LTF Classifier — "How is price behaving right now?" (4 classes)**
- `0=TRENDING`, `1=RANGING`, `2=CONSOLIDATING`, `3=VOLATILE`
- Direction-agnostic. Trained on 1H data only.
- Architecture: `18 → 128 → 64 → 4`
- Accuracy target: ≥55% (random baseline = 25%)
- Outputs full `proba[4]` list — all 4 probabilities injected into RL state vector

**Label generation:**
- Global GMM (`fit_global_gmm`) fitted on all symbols combined for consistent semantics
- 8 GMM features: efficiency_ratio, rel_vol, drift, compression, vol_slope, atr_pctile, autocorr_lag1, hurst_proxy
- `create_rule_labels()` used as training targets with per-bar confidence weights
- Ambiguous bars (confidence < 0.4) get soft targets — prevents memorizing label noise
- 3-bar hysteresis at inference time

**Fixed bugs (2026-05-01):**
- `create_labels()` mapped cluster to class 4 (out of range for 4-class model) — fixed to 4-component GMM with correct mapping
- Silent `try/except: pass` in `_build_feature_matrix` for BOS/sweep features removed
- `predict_batch()` returned silent defaults when untrained — now raises `ModelNotTrainedError`
- `_last_mtime` now set on initial load to prevent spurious hot-reload on first 5-min check

---

### 2. GRU-LSTM Predictor

**File:** `models/gru_lstm_predictor.py`
**Weights:** `weights/gru_lstm/model.pt` + `weights/gru_lstm/temperature.pt`
**Sequence length:** 30 bars × **78** features (increased from 74 on 2026-05-01)

```
Input: (batch, 30, 78)
  → GRU(hidden=64, num_layers=2, dropout=0.3, batch_first=True)
  → Dropout(0.3)
  → LSTM(hidden=128, num_layers=2, dropout=0.3, batch_first=True)  → (B, T=30, 128)
  → MultiheadAttention(embed=128, heads=4, dropout=0.1) over all 30 timesteps  ← NEW
  → mean-pool across T dim  → (B, 128)   [replaces last-timestep extraction]
  → Dropout(0.3)
  → shared FC(128→64) → ReLU → Dropout(0.3)
  → direction_head: FC(64→1) → sigmoid(logit / T)  → p_bull   [T from temperature.pt]
  → magnitude_head: FC(64→1) → ReLU                → expected_move
  → variance_head:  FC(64→1) → softplus + 1e-6      → expected_variance
```

**Why self-attention:** The previous architecture extracted only the last hidden state — discarding information from the 29 earlier bars. The attention layer learns which bars in the 30-bar window are most predictive for the current signal. This is especially important for detecting pullbacks (bar 15) and BOS events (bar 5) that precede the decision bar.

**Loss:** `BCE(dir, pos_weight) + 0.5×SmoothL1(mag) + 0.3×NLL(var)`.
Dead-zone bars (|log_return| < 0.3×ATR) set direction label to NaN and are masked from BCE.

**Temperature scaling:** `fit_temperature()` minimises NLL on calibration set and saves `temperature.pt`.

**78 SEQUENCE_FEATURES per timestep:**
| Indices | Count | Description |
|---------|-------|-------------|
| 0–15 | 16 | Base 15M: log_return, HL range, close_vs_open, ATR, RSI, EMA21/50 dist, BB pos, vol ratio, session flags, BOS flags, FVG flags |
| 16–17 | 2 | 5M: RSI, EMA21 dist |
| 18–20 | 3 | 1H: ADX, EMA21 dist, EMA50 dist |
| 21–23 | 3 | 4H: EMA21-50 diff, ADX, RSI |
| 24–25 | 2 | 1D: EMA21 dist, EMA stack |
| 26–28 | 3 | HTF bias one-hot (BIAS_UP, BIAS_DOWN, BIAS_NEUTRAL) |
| 29 | 1 | htf_bias_conf |
| 30–33 | 4 | LTF behaviour one-hot (TRENDING, RANGING, CONSOLIDATING, VOLATILE) |
| 34 | 1 | ltf_conf |
| 35–37 | 3 | htf_ltf_align, htf_regime_dur, ltf_regime_dur |
| 38–40 | 3 | vol_slope_seq, time_sin, time_cos |
| 41–71 | 31 | ICT structure distances (EMA pullback zone, BOS age/strength, FVG dist/fill, sweep wick depth, Asian range context, candle body/wicks, oscillators, ADX, regime dynamics, session timing) |
| 72–73 | 2 | macro_vix_level, macro_yield_spread |
| 74 | 1 | vwap_dist_atr — (close − VWAP) / ATR, clipped [−3, +3] |
| 75 | 1 | volume_delta_pct — bar-level buy/sell pressure fraction [−1, +1] |
| 76 | 1 | cum_delta_norm — rolling 20-bar cumulative delta normalised |
| 77 | 1 | wick_auction_ratio — lower_wick / (lower + upper wick) [0, 1] |

**Note: old weights (74 features) are incompatible with the new 78-feature model. Full retrain required.**

---

### 3. QualityScorer (EV Regressor)

**File:** `models/quality_scorer.py`
**Weights:** `weights/quality_scorer.pkl`

```
20 → BatchNorm → FC(128) → BN → GELU → Dropout(0.3)
   → FC(64)  → BN → GELU → Dropout(0.25)
   → FC(32)  → BN → GELU → Dropout(0.2)
   → FC(1)   [identity output — unbounded float]
```

**Loss:** Class-weighted Huber(δ=1.0). `pos_weight = n_neg/n_pos` (allowed < 1.0 when WR > 50%).
Win labels normalised by median winner RR → range [-1, +3].

**Called post-signal** (after PM enrichment gives actual `rr_ratio`). Not in the GPU cache.

**20 QUALITY_FEATURES (updated from 17 on 2026-05-01):**
| Index | Feature |
|-------|---------|
| 0 | strategy_id |
| 1 | signal_direction |
| 2 | rr_ratio |
| 3 | p_bull_gru |
| 4 | p_bear_gru |
| 5 | regime_class |
| 6 | sentiment_score |
| 7 | adx_at_signal |
| 8 | atr_ratio_at_signal |
| 9 | volume_ratio |
| 10 | spread_at_signal |
| 11 | session_at_signal |
| 12 | news_in_30min |
| 13 | strategy_win_rate_20 |
| 14 | gru_uncertainty |
| 15 | regime_duration |
| 16 | vol_slope_at_signal |
| 17 | strategy_win_rate_5 |
| 18 | strategy_win_rate_50 |
| 19 | gru_signal_agreement |

**EV Label tiers:**
| Exit | EV label |
|------|----------|
| `tp2` | `+rr_ratio` |
| `tp1` | `+rr × 0.75` |
| `be_or_trail` | `+rr × 0.4` |
| `sl_*` | `-1.0` |
| `time_exit` | `+rr×0.2` (win), `-0.5` (loss), `0.0` (flat) |

---

### 4. SentimentModel

**File:** `models/sentiment_model.py` **Weights:** pre-trained (no local file)

Backends: **FinBERT** (`ProsusAI/finbert`) primary → **VADER** fallback.
Output: `{"sentiment_score": float, "sentiment_label": str, "sentiment_confidence": float}`
Gold: USD bullish → XAUUSD bearish (inverted).

---

### 5. RLAgent (PPO)

**File:** `models/rl_agent.py` **Weights:** `weights/rl_ppo/model.zip`

PPO via stable-baselines3. **Fully wired into live signal pipeline as of 2026-05-01.**

**Public API:**
- `decide(state, available_signals, session)` — primary live-trading entry point. Uses trained PPO when available; session-aware heuristic before training. Always returns `(trader_id, threshold)`.
- `select_action(state, available_signals)` — PPO-only path; raises `ModelNotTrainedError` if untrained.

**9 actions (v3 — threshold-only):**
| Action | Meaning | When to use |
|--------|---------|-------------|
| 0 | NoTrade | Drawdown / unfavorable regime |
| 1 | Trade @ 0.60 | Strong trend + positive IC |
| 2 | Trade @ 0.62 | Default high-confidence |
| 3 | Trade @ 0.65 | Moderate trend |
| 4 | Trade @ 0.68 | Mixed signals |
| 5 | Trade @ 0.70 | Conservative baseline |
| 6 | Trade @ 0.72 | Cautious regime |
| 7 | Trade @ 0.75 | High selectivity |
| 8 | Trade @ 0.80 | Volatile/ranging — very selective |

**43-dim state vector (built by `_build_rl_state()` in `signal_pipeline.py`):**
| Dims | Content |
|------|---------|
| 0–2 | GRU direction: p_bull, p_bear, expected_variance |
| 3–5 | HTF regime full proba: [BIAS_UP, BIAS_DOWN, BIAS_NEUTRAL] |
| 6–9 | LTF regime full proba: [TRENDING, RANGING, CONSOLIDATING, VOLATILE] |
| 10–11 | Sentiment: score [−1,1], confidence [0,1] |
| 12–19 | ATR history ratios — 8 lags (1,4,8,24,48,96,168,336 bars) |
| 20 | Spread normalised (spread_pips / 5) |
| 21–23 | Time: sin(hour), cos(hour), session_enc (0=INACTIVE … 1=NY) |
| 24–28 | Portfolio: win_rate_10, drawdown_norm, trades_today_norm, open_pos_norm, daily_pnl_norm |
| 29–32 | Market structure: adx_norm, vol_slope, macro_vix, macro_yield_spread |
| 33–36 | Auction indicators: vwap_dist_atr_norm, volume_delta_pct, wick_auction_ratio, cum_delta_norm |
| 37–39 | Regime quality: ema_stack_norm, htf_conf, ltf_conf |
| 40–42 | Extra: atr_ratio_norm, volume_ratio_norm, direction_strength |

**Training requirement:** ≥ 50 completed trades with `rl_action` and `state_at_entry[43]` in journal.

---

## Indicator Layer (2026-05-01 additions)

**File:** `indicators/market_structure.py`

Three new vectorised functions added; MACD removed from `compute_all` output (function kept for standalone use):

### `compute_vwap(df)`
Session-anchored VWAP with ±1σ and ±2σ bands.
- Resets at UTC midnight (DatetimeIndex) or rolling `session_length` bars (integer index)
- Online variance formula: `σ² = E[tp²] - E[tp]²` (single pass, numerically stable)
- Output: `vwap`, `vwap_upper1/2`, `vwap_lower1/2`, `vwap_dist_atr`
- Why: Institutional reference price. Price relative to VWAP distinguishes value from extension.

### `compute_volume_delta(df, period=20)`
Directional volume estimated from OHLC bar position.
- `buy_vol = volume × (close−low) / (high−low)` — bars that close near high → buyers dominated
- Cumulative delta divergence signals (`delta_bull_div`, `delta_bear_div`)
- Output: `volume_delta`, `volume_delta_pct`, `cum_delta_20`, `delta_bull_div`, `delta_bear_div`
- Why: Price can make new highs while cumulative delta peaks — leading divergence before reversal.

### `compute_wick_ratio(df)`
Bar auction result — which side dominated each bar's high-low range.
- `wick_auction_ratio = lower_wick / (lower + upper wick + ε)` — near 1.0 = buyers won; near 0.0 = sellers won
- Output: `wick_auction_ratio`, `body_pct`
- Why: Price accepted at a level (large body, small wicks) has very different meaning from price rejected (small body, large wicks).

---

## Position Sizing (2026-05-01 changes)

**File:** `monitors/portfolio_manager.py`

**Removed:** `_streak_scalar` — the 3-loss→0.5×, 4-loss→0.35× adaptive sizing has been deleted.
- Reason: Streak-based scaling introduces non-stationarity. It assumes future performance correlates with recent streaks, which it does not statistically. It also forces the system to size down precisely when a mean-reversion in results is most likely.

**Sizing formula (current):**
```python
size = base_size × vol_scalar
vol_scalar = clip(nominal_ATR / current_ATR, 0.5, 1.0)
```
- `vol_scalar` only REDUCES position size when current ATR exceeds nominal
- Cap is 1.0 — volatile conditions cause size reduction, calm conditions use full `base_size`
- The previous cap of 1.25 could inadvertently increase size in artificially quiet periods (e.g., pre-news compression)

---

## Backtest Architecture

All GPU inference is batched upfront per symbol in `_precompute_ml_cache`. The bar loop does only dict lookups and gate checks.

**Execution order in `_precompute_ml_cache`:**
1. `RegimeClassifier._build_feature_matrix()` → batch 4H + 1H → `regime_preds` + confidence series
2. `_build_sequence_df()` with regime series injected at each timestep → `(N, 78)` matrix
3. GRU sliding-window batch inference (1024/batch) → `gru_preds`
4. Merge into `cache[bar_idx]` dict

QualityScorer runs **outside** the cache, per signal, after PM enrichment.

---

## Causal Integrity

All features are strictly backward-looking.

| Feature | Status |
|---------|--------|
| `sr_dist_*`, `sr_in_*`, `sr_*_strength` (REGIME_FEATURES 28–33) | Zeroed for feature-distribution compatibility; detector is causal but enabling requires retrain |
| VWAP session reset | Causal — `groupby(day_key).cumsum()` only uses bars up to and including bar t |
| Volume delta cumulative | Causal — rolling `.sum()` over prior `period` bars |
| Macro `bfill()` | Removed — replaced with `fillna(0.0)` only |
| All rolling indicators | Backward-only |
| HTF `reindex(method="ffill")` | Causal — only HTF bars ≤ t contribute |

---

## Feature Contracts

Feature list order is a hard contract — changing order or count breaks saved model weights.

| List | Length | Used by |
|------|--------|---------|
| `SEQUENCE_FEATURES` | **78** | GRU training + inference |
| `REGIME_4H_FEATURES` | 34 | RegimeClassifier (4H bias) |
| `REGIME_1H_FEATURES` | 18 | RegimeClassifier (1H behaviour) |
| `QUALITY_FEATURES` | **20** | QualityScorer |
| `RL_STATE_DIM` | 43 | RLAgent |

---

## Error Policy

No silent fallbacks anywhere. Every failure raises and propagates.

- Untrained models: raise `ModelNotTrainedError(RuntimeError)`
- `predict_batch()` when untrained: raises `ModelNotTrainedError` (previously returned dummy arrays)
- Silent `try/except: pass` blocks: removed from `_build_feature_matrix`
- `reload_if_updated()`: re-raises load failures (previously swallowed with `logger.error`)
- `create_labels()` when sklearn missing or insufficient data: raises `ValueError` or `ImportError`
- Feature computation failures: raise (no zero-filled arrays)

---

## Known Issues / Pending Work

| Item | File | Notes |
|------|------|-------|
| GRU retrain required | `models/gru_lstm_predictor.py` | Architecture changed (attention + 78 features); old weights invalid |
| RL cold-start | `models/rl_agent.py` | Uses heuristic until ≥50 journal trades with `state_at_entry[43]` |
| S/R zone features zeroed | `models/regime_classifier.py` | REGIME_FEATURES[28–33] stay zero until retrain with detector enabled |
| Regime BIAS_NEUTRAL recall | `models/regime_classifier.py` | ~30-38%; class_w boost (4×) applied; retrain expected to improve |
