# Signal Generation Reference

Last updated: 2026-04-19.

The 5 ICT rule-based traders have been removed. `traders/__init__.py` is empty.
Signal generation is unified in `scripts/run_backtest._compute_backtest_signal()`
with `trader_id="ml_trader"` running on all 11 symbols.

---

## How Signals Are Generated

ICT/SMC logic is **encoded as numeric features** in `SEQUENCE_FEATURES`. The GRU
learns from those features; it decides direction and confidence. There are no rule
branches, no regime hard-gates, no per-symbol exclusions.

```
Per 15M bar:
  ① _precompute_ml_cache (done once per symbol, GPU-batched):
      RegimeClassifier(4H) → regime_4h + conf
      RegimeClassifier(1H) → regime_1h + conf
      → injected into each timestep of 30-bar sequence
      GRU-LSTM(30 bars × 74 features) → p_bull, p_bear, expected_move, expected_variance

  ② Bar loop:
      expected_variance > 0.80               → reject
      max(p_bull, p_bear) < 0.58             → reject
      side = "buy" if p_bull > p_bear else "sell"

  ③ Level computation (ATR-based):
      SL  = entry ± (atr × SL_ATR_MULT)
      TP1 = entry ± (sl_distance × TP1_MULT)
      TP2 = entry ± (sl_distance × TP2_MULT)

  ④ PM enrichment:
      Size = risk_amount / sl_distance
      TP1 partial close (50%) → SL to break-even
      Correlation cap: max 2 concurrent positions per directional group

  ⑤ QualityScorer (post-PM — uses actual rr_ratio):
      ev < 0.10                              → reject
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

Regime is not a hard gate. It is injected as one-hot + confidence into every GRU timestep:

| Features | Indices | Source |
|----------|---------|--------|
| `regime_4h_0..4` + `regime_4h_conf` | 26–31 | RegimeClassifier (4H bias) |
| `regime_1h_0..4` + `regime_1h_conf` | 32–37 | RegimeClassifier (1H structure) |

**5 classes per level:** TRENDING_UP=0, TRENDING_DOWN=1, RANGING=2, VOLATILE=3, CONSOLIDATION=4

The GRU learns `P(direction | sequence, regime_4h, regime_1h)` — regime context
gives it the ability to weight the same price structure differently in trending vs ranging markets.

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

| Gate | Value | Applied in |
|------|-------|-----------|
| GRU uncertainty | `expected_variance ≤ 0.80` | `_compute_backtest_signal` |
| GRU direction | `max(p_bull, p_bear) ≥ 0.58` | `_compute_backtest_signal` |
| EV threshold | `ev ≥ 0.10` | `_backtest_trader` (post-PM) |
| Dead zone | 12:00–13:00 UTC | `_backtest_trader` |
| Cooldown | 10 bars | `_backtest_trader` |
| Daily loss cap | 2% | `_backtest_trader` |
| Portfolio drawdown | 8% | `_backtest_trader` |
| Correlation cap | max 2 per group | `PortfolioManager` |
