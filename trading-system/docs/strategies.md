# Signal Generation Reference

Last updated: 2026-04-24.

The 5 ICT rule-based traders have been removed. `traders/__init__.py` is empty.
Signal generation is unified in `scripts/run_backtest._compute_backtest_signal()`
with `trader_id="ml_trader"` running on all 11 symbols.
The live path mirrors this exactly in `services/signal_pipeline._compute_ml_signal()`.

---

## How Signals Are Generated

ICT/SMC logic is **encoded as numeric features** in `SEQUENCE_FEATURES`. The GRU
learns from those features; it decides direction and confidence. Regime is applied as
hard gates AFTER the GRU direction step.

```
Per 15M bar:
  ① _precompute_ml_cache (done once per symbol, GPU-batched):
      RegimeClassifier(4H) → BIAS_UP/DOWN/NEUTRAL + conf
      RegimeClassifier(1H) → TRENDING/RANGING/CONSOLIDATING/VOLATILE + conf
      → injected as one-hot into each timestep of 30-bar sequence
      GRU-LSTM(30 bars × 74 features) → p_bull, p_bear, expected_move, expected_variance

  ② Bar loop gate order:
      expected_variance > MAX_UNCERTAINTY (default 2.0)  → reject
      max(p_bull, p_bear) < 0.58                         → reject
      side = "buy" if p_bull ≥ p_bear else "sell"

      HTF bias alignment:
        BIAS_UP   + sell → reject
        BIAS_DOWN + buy  → reject
        BIAS_NEUTRAL: require conf ≥ NEUTRAL_BIAS_THRESHOLD (0.58)

      LTF behaviour filter:
        CONSOLIDATING → reject (if BLOCK_LTF_CONSOLIDATING=1)
        VOLATILE      → require conf ≥ VOLATILE_ENTRY_THRESHOLD
        TRENDING      → optional pullback filter (REQUIRE_TRENDING_PULLBACK=0 by default)
        RANGING       → optional range boundary check (RANGING_REQUIRE_RANGE=0 by default)

  ③ Level computation (ATR-based):
      SL = entry ± (atr × SL_ATR_MULT)
      TP = entry ± (sl_distance × RR_DEFAULT)
      RANGING entries: TP targets the far wall of the range when range_valid

  ④ PM enrichment:
      Size = risk_amount / sl_distance
      TP1 partial close → SL to break-even
      Correlation cap: max 2 concurrent positions per directional group

  ⑤ QualityScorer (post-PM — uses actual rr_ratio):
      ev < MIN_EV_THRESHOLD (0.10)           → reject
```

---

## ICT Features in the GRU Input

These are the numeric encodings of ICT concepts that used to be hard rules.

| Feature | Index | What it encodes |
|---------|-------|-----------------|
| `bos_bull_flag`, `bos_bear_flag` | 12–13 | Binary: BOS on current bar |
| `fvg_bull_open`, `fvg_bear_open` | 14–15 | Binary: open FVG on current bar |
| `ema_pullback_zone` | 41 | Price position in EMA21-50 band, ATR-normalised |
| `ema21_slope_15m`, `ema21_slope_1h` | 42–43 | EMA21 slope normalised by ATR |
| `ema_stack_15m` | 44 | EMA21 > EMA50 score |
| `bos_bull_bars_ago`, `bos_bear_bars_ago` | 45–46 | Age of last BOS / 40 |
| `bos_bull_strength`, `bos_bear_strength` | 47–48 | BOS move size / ATR |
| `fvg_bull_dist_atr`, `fvg_bear_dist_atr` | 49–50 | Distance to nearest open FVG / ATR |
| `fvg_bull_fill_ratio`, `fvg_bear_fill_ratio` | 51–52 | How far price has entered FVG |
| `sweep_wick_depth_atr` | 53 | Wick beyond range extreme / ATR |
| `body_recovery_ratio` | 54 | Candle body recovery after sweep |
| `dist_to_recent_high_atr`, `dist_to_recent_low_atr` | 55–56 | Proximity to 20-bar extremes |
| `asian_range_width_atr` | 57 | Asian session range width / ATR |
| `price_vs_asian_high_atr`, `price_vs_asian_low_atr` | 58–59 | Price vs Asian high/low |

---

## Regime Context

Regime is injected as one-hot + confidence into every GRU timestep AND applied as a
directional gate in the bar loop.

| Features | Indices | Source |
|----------|---------|--------|
| `htf_bias_up/down/neutral` (3 dims) + `htf_bias_conf` | 26–29 | RegimeClassifier (4H, htf_bias mode) |
| `ltf_trending/ranging/consolidating/volatile` (4 dims) + `ltf_conf` | 30–34 | RegimeClassifier (1H, ltf_behaviour mode) |
| `htf_ltf_align`, `htf_regime_dur`, `ltf_regime_dur` | 35–37 | alignment + duration |

**HTF classes (3):** BIAS_UP=0, BIAS_DOWN=1, BIAS_NEUTRAL=2
**LTF classes (4):** TRENDING=0, RANGING=1, CONSOLIDATING=2, VOLATILE=3

The GRU learns `P(direction | sequence, htf_bias, ltf_behaviour)`. Regime is also used
as a post-GRU hard gate: BIAS_UP blocks sell signals; BIAS_DOWN blocks buy signals.

---

## Symbols

All 11 symbols run on `ml_trader`:
`EURUSD GBPUSD USDJPY AUDUSD NZDUSD USDCAD USDCHF EURGBP EURJPY GBPJPY XAUUSD`

No per-symbol exclusions. No session-based symbol filtering.

---

## Session Windows

Sessions are encoded as features (`is_asian`, `is_london`, `is_ny` at indices 9–11) and as
continuous timing features (`mins_since_london_open`, `mins_since_ny_open` at indices 68–69).
The GRU learns session patterns from data — no hard session gates remain.

**Dead zone 12:00–13:00 UTC** is still applied as a hard skip in the bar loop.
**Cooldown** of 10 bars since last trade on the same symbol is still enforced.

---

## Capital and Risk

```
ACCOUNT_BALANCE    = 10000.0
CAPITAL_PER_TRADER = 0.20     (single ml_trader uses 100%)
RISK_PER_TRADE     = 0.01     (1% per trade)
MAX_DAILY_LOSS_PCT = 0.02     (2% daily loss cap)
MAX_DRAWDOWN_PCT   = 0.08     (8% portfolio halt)
MAX_CONCURRENT_POSITIONS = 2
```

Position size: `risk_amount / (|entry - sl| in price)`
ATR-scaled stop loss. TP1 at 2× risk, TP2 at 4× risk (PM-adjusted by confidence).

---

## Gate Summary

| Gate | Default value | Env override | Applied in |
|------|--------------|--------------|-----------|
| GRU uncertainty | `expected_variance ≤ 2.0` | `MAX_UNCERTAINTY` | `_compute_backtest_signal` |
| GRU direction | `max(p_bull, p_bear) ≥ 0.58` | `ML_DIRECTION_THRESHOLD` | `_compute_backtest_signal` |
| HTF bias alignment | BIAS_UP blocks sell; BIAS_DOWN blocks buy | — | `_compute_backtest_signal` |
| HTF neutral confidence | `≥ 0.58` | `NEUTRAL_BIAS_THRESHOLD` | `_compute_backtest_signal` |
| LTF VOLATILE entry | `≥ ML_DIRECTION_THRESHOLD` | `VOLATILE_ENTRY_THRESHOLD` | `_compute_backtest_signal` |
| LTF CONSOLIDATING | blocked if env set | `BLOCK_LTF_CONSOLIDATING` | `_compute_backtest_signal` |
| LTF RANGING boundary | optional | `RANGING_REQUIRE_RANGE` | `_compute_backtest_signal` |
| EV threshold | `ev ≥ 0.10` | `MIN_EV_THRESHOLD` | `_backtest_trader` (post-PM) |
| Dead zone | 12:00–13:00 UTC | — | `_backtest_trader` |
| Cooldown | 10 bars | — | `_backtest_trader` |
| Daily loss cap | 2% | — | `_backtest_trader` |
| Portfolio drawdown | 8% | — | `_backtest_trader` |
| Correlation cap | max 2 per group | — | `PortfolioManager` |
| Signal pipeline confidence | `≥ 0.55` | — | `signal_pipeline.process_bar` |
