# Trading Strategies Reference

Last updated: 2026-04-17.

All traders inherit from `BaseTrader` (`traders/base_trader.py`). Each runs independently
with its own capital allocation, session window, and risk limits. ICT/SMC conditions
determine WHERE to trade. ML models (EV gate) determine WHETHER to trade.

---

## BaseTrader — 8 Guards

Every signal passes through these in order. Any returning `None` rejects the trade.

| Guard | Condition |
|-------|-----------|
| 0 | Symbol in `EXCLUDED_SYMBOLS` → reject |
| 1 | Outside `SESSION_START_UTC` to `SESSION_END_UTC` → reject |
| 2 | Dead zone 12:00–13:00 UTC → reject (all traders) |
| 3 | Symbol in cooldown (`COOLDOWN_MINUTES`) → reject |
| 4 | `trades_today[symbol] ≥ MAX_TRADES_PER_SESSION` → reject |
| 5 | Circuit breaker: daily loss > limit OR drawdown > limit → reject |
| 6 | Spread: XAUUSD > 50 pips OR Forex > 3 pips → reject |
| 7 | EV gate: `ev < MIN_EV_THRESHOLD (0.10)` OR `expected_variance > 0.80` OR `p_dir < ML_DIRECTION_THRESHOLD` → reject |
| 8 | RL gate: PPO does not approve this trader OR EV < RL selectivity threshold → reject |

**EV is mandatory when `ML_ENABLED=true`.** Missing `ev` key raises `RuntimeError`.

**Position sizing:** ATR-based, 1% of allocated capital per trade. `CAPITAL_PER_TRADER=0.20`
means each trader controls 20% of the account.

---

## Trader 1 — NY EMA Trend Pullback

**File:** `traders/trader_1_ny_ema.py`  **ID:** `trader_1`

### Session
13:00–17:00 UTC (hard close 17:00)

### Symbols
All except EURGBP (EURGBP excluded — consistently mean-reverts against EMA entries)

### Signal Logic
1. 1H EMA stack bullish (for longs) or bearish (for shorts)
2. ADX > 20 (trending, not oscillating)
3. Price has pulled back to EMA zone
4. Candle body ratio > 0.25 (genuine momentum candle, not a doji)
5. **Regime: TRENDING_UP (for longs) or TRENDING_DOWN (for shorts) required**
   — raises `RuntimeError` if regime missing when `ML_ENABLED=true`

### Additional Filters
- XAUUSD: ATR ratio > 0.7 (enough volatility to cover spread)
- Forex: round number filter ±15 pips (avoid stop clusters)

### Parameters
- `MAX_TRADES_PER_SESSION = 2`
- `COOLDOWN_MINUTES = 10`
- `ML_DIRECTION_THRESHOLD = 0.55`

---

## Trader 2 — Structure Break + FVG Continuation

**File:** `traders/trader_2_fvg_bos.py`  **ID:** `trader_2`

### Session
London 07:00–12:00 + NY 13:00–18:00 UTC (dead zone 12–13 blocked by Guard 2)

### Symbols
All 11

### Signal Logic
1. 4H BOS establishes directional bias (bullish or bearish structure)
2. Open FVG exists in registry (max 20 bars old)
3. Price returns to fill the FVG
4. R:R pre-check: TP/SL ratio ≥ 2.0 before computing entry
5. News block: no high-impact event within 15 min

### Parameters
- `MAX_TRADES_PER_SESSION = 2`
- `COOLDOWN_MINUTES = 15`
- `ML_DIRECTION_THRESHOLD = 0.62`

---

## Trader 3 — London Breakout + Liquidity Sweep

**File:** `traders/trader_3_london_bo.py`  **ID:** `trader_3`

### Session
07:00–10:00 UTC (hard close 12:00)

### Symbols
All except USDJPY

### Signal Logic
1. Asian range identified (02:00–07:00 UTC high/low)
2. Liquidity sweep of Asian range extreme: wick-to-body ratio > 1.5 (strong rejection)
3. Volume on breakout bar > 20-bar SMA × 1.3
4. ADX > 20
5. **Regime: TRENDING_UP or TRENDING_DOWN only** — RANGING/VOLATILE blocked at signal level
   (not a runtime error, just returns None)

### Additional Filters
- `breakout_strength`: ATR-normalized range of breakout bar
- `fakeout_prob`: >55% → reject (historical fakeout rate for this setup type)
- `range_compression`: tight Asian range increases breakout reliability

### Parameters
- `MAX_TRADES_PER_SESSION = 1`
- `COOLDOWN_MINUTES = 5`
- `ML_DIRECTION_THRESHOLD = 0.55`

---

## Trader 4 — News Momentum

**File:** `traders/trader_4_news_momentum.py`  **ID:** `trader_4`

### Session
Any time (no session restriction override)

### Symbols
EURUSD, GBPUSD, XAUUSD

### Signal Logic
1. `news_in_15min = True` required (high-impact event imminent)
2. `sentiment_score ≥ 0.65` **hard gate** (not EV-adjustable)
3. Structural breakout in direction of sentiment (NOT a fade)
4. Hard close 60 min post-news

### Parameters
- `MAX_TRADES_PER_SESSION = 1`
- `COOLDOWN_MINUTES = 30`
- `ML_DIRECTION_THRESHOLD = 0.55`
- `SENTIMENT_THRESHOLD = 0.65`

---

## Trader 5 — Asian Range Mean Reversion

**File:** `traders/trader_5_asian_mr.py`  **ID:** `trader_5`

### Session
02:00–06:45 UTC (hard close 06:45)

### Symbols
USDJPY, EURJPY, AUDJPY, EURUSD, AUDUSD (XAUUSD and GBPJPY excluded)

### Signal Logic
1. **Regime: RANGING required** — raises `RuntimeError` if missing when `ML_ENABLED=true`
2. `volatility_stable`: ATR expansion ratio < 1.3× (not a breakout setup)
3. Price at Asian range extreme (near high or low)
4. Dual oscillator confirmation (RSI + stochastic oversold/overbought)
   — one oscillator OK if `p_win ≥ 0.65` from QualityScorer
5. `distance_from_mean_atr`: how far price is stretched from range midpoint

### Parameters
- `MAX_TRADES_PER_SESSION = 2`
- `COOLDOWN_MINUTES = 30`
- `ML_DIRECTION_THRESHOLD = 0.52`

---

## Symbol Coverage

| Symbol | T1 | T2 | T3 | T4 | T5 |
|--------|----|----|----|----|-----|
| EURUSD | ✓ | ✓ | ✓ | ✓ | ✓ |
| GBPUSD | ✓ | ✓ | ✓ | ✓ | — |
| USDJPY | ✓ | ✓ | — | — | ✓ |
| AUDUSD | ✓ | ✓ | ✓ | — | ✓ |
| NZDUSD | ✓ | ✓ | ✓ | — | — |
| USDCAD | ✓ | ✓ | ✓ | — | — |
| USDCHF | ✓ | ✓ | ✓ | — | — |
| EURGBP | — | ✓ | ✓ | — | — |
| EURJPY | ✓ | ✓ | ✓ | — | ✓ |
| GBPJPY | ✓ | ✓ | ✓ | — | — |
| XAUUSD | ✓ | ✓ | ✓ | ✓ | — |

---

## Capital and Risk

```
ACCOUNT_BALANCE    = 10000.0
CAPITAL_PER_TRADER = 0.20     (20% each × 5 = 100%)
RISK_PER_TRADE     = 0.01     (1% of allocated capital per trade)
```

Position size: `risk_amount / (|entry - sl| in price)`  
ATR-scaled stop loss. Take profit at `entry ± (sl_distance × rr_ratio)`.

---

## How Strategies Interact with ML

ICT logic runs inside `_compute_signal()` — it produces entry/stop/target levels.
ML gates run in `analyze_market()` after the signal is produced.

```
_compute_signal()          ← ICT: where to trade, what levels
Guard 7 (EV gate)          ← QualityScorer: is this setup worth trading?
Guard 8 (RL gate)          ← RLAgent: should this trader trade right now?
Ensemble scoring           ← final confidence: (ict × 0.5 + ml × 0.5 + sentiment) × regime_mult
```

Regime affects both the ICT logic (some traders require specific regimes) and the ensemble
multiplier (TRENDING ×1.2, RANGING ×0.9, VOLATILE ×0.85).
