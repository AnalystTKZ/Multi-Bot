# Project Description — ML-Native Forex/Gold Trading System

Updated 2026-04-24.

---

## Executive Summary

A production-grade, fully containerised automated trading system for Forex and Gold.
The system has evolved from 5 ICT rule-based traders to a **single unified ML-native signal generator**.

ICT/SMC concepts (BOS, FVG, liquidity sweep, EMA pullback, Asian range) are encoded as
numeric features fed to a GRU-LSTM model. The models decide direction, confidence, and
expected value — there are no rule branches.

Training runs on Kaggle T4 × 2 GPUs. Models warm-start from existing weights across
each retraining run — historical training is preserved, not discarded.

Broker: Capital.com REST API (`CAPITAL_ENV=demo`, `PAPER_TRADING=true` default).

> **All 4 trainable models must exist before the engine runs.** See `docs/TRAINING_AND_BACKTEST.md`.

---

## Signal Generation

One unified path — `ml_trader` across all 11 symbols.

```
For each 15M bar:
  1. GRU-LSTM predicts p_bull, p_bear, expected_variance
  2. Uncertainty gate: expected_variance > MAX_UNCERTAINTY (env default 2.0) → skip
  3. Direction gate: max(p_bull, p_bear) < 0.58 → skip
  4. ATR-based entry/SL/TP levels
  5. PortfolioManager: size, TP1/TP2, correlation cap
  6. QualityScorer: ev in R-multiples (uses actual rr_ratio)
  7. EV gate: ev < 0.10 → skip
  8. Trade
```

Regime is encoded as one-hot features at indices 26–37 of the GRU input AND applied as a hard directional gate after the GRU direction step (BIAS_UP blocks sell; BIAS_DOWN blocks buy).

---

## ICT Features Encoded in GRU

| Concept | Features |
|---------|----------|
| BOS | age (bars ago / 40), strength (move / ATR) |
| FVG | distance to nearest open FVG / ATR, fill ratio |
| Liquidity sweep | wick depth / ATR, body recovery ratio |
| EMA pullback | band position, EMA21 slope, EMA stack |
| Asian range | range width / ATR, price vs high/low |
| Market structure | proximity to 20-bar highs/lows |
| HTF regime (4H) | BIAS_UP/DOWN/NEUTRAL one-hot (3 dims) + conf — indices 26–29 |
| LTF regime (1H) | TRENDING/RANGING/CONSOLIDATING/VOLATILE one-hot (4 dims) + conf — indices 30–34 |

---

## Operating Flow (Working Path — Backtest)

```
1. processed_data/histdata/{SYMBOL}_{TF}.parquet loaded
2. GPU-batched ML inference per symbol:
   - RegimeClassifier (4H) → regime_4h one-hot
   - RegimeClassifier (1H) → regime_1h one-hot
   - GRU-LSTM (30-bar sequences) → p_bull, p_bear, variance
3. Bar loop: gates → levels → PM → QualityScorer → EV gate → simulate
4. Trade journal: CSV + JSONL
5. Weekly/monthly retraining from journal
```

## Operating Flow (Live Trading)

`main.py` and `signal_pipeline.py` are both functional. `signal_pipeline._compute_ml_signal`
mirrors `run_backtest._compute_backtest_signal` exactly.

---

## Risk Management

- **Daily loss cap**: 2% per day (halts trading for remainder of day)
- **Portfolio drawdown halt**: 8% (halts all trading until reset)
- **Position sizing**: 1% risk per trade, ATR-scaled SL
- **TP1/TP2**: 50% close at TP1 → SL to break-even; trail to TP2
- **Correlation cap**: max 2 concurrent positions per directional group
- **Cooldown**: 10 bars (150 min) between trades on same symbol
- **Dead zone**: 12:00–13:00 UTC blocked every day

---

## ML/AI Layer

| Model | Framework | Role |
|---|---|---|
| RegimeClassifier (4H) | PyTorch GPU | Macro bias: BIAS_UP/BIAS_DOWN/BIAS_NEUTRAL (3-class, weights: regime_htf.pkl) |
| RegimeClassifier (1H) | PyTorch GPU | Intraday behaviour: TRENDING/RANGING/CONSOLIDATING/VOLATILE (4-class, weights: regime_ltf.pkl) |
| GRU-LSTM | PyTorch GPU | Direction + magnitude + uncertainty from 30-bar sequences |
| QualityScorer | PyTorch GPU | Expected value in R-multiples (post-signal) |
| SentimentModel | FinBERT + VADER | News/macro directional bias |
| RLAgent | PPO (SB3 CPU) | EV selectivity tier (currently collapsed — needs more data) |

---

## Trade Journals

**Clean CSV** (`logs/trade_journal.csv`):
```
timestamp, trader, symbol, side, size, entry, stop_loss, take_profit, rr_ratio, confidence, pnl, commission
```

**Detailed JSONL** (`logs/trade_journal_detailed.jsonl`):
Full record including `ev`, `expected_variance`, `exit_reason`, `ml_model_scores` (p_bull, p_bear, regime),
`state_at_entry` (43-dim RL state), `rl_action`.

Exit reason values used by QualityScorer labels:
- `tp2` — full TP hit → EV label = rr_ratio
- `tp1` — partial close → EV label = rr × 0.75
- `be_or_trail` — trailed out → EV label = rr × 0.4
- `sl_full` — stopped → EV label = -1.0
- `time_exit` — excluded from EV labels

---

## Trading Pairs

All 11: `EURUSD GBPUSD USDJPY AUDUSD NZDUSD USDCAD USDCHF EURGBP EURJPY GBPJPY XAUUSD`

---

## Data Pipeline

9-step offline pipeline in `pipeline/`. Runs on Kaggle T4 × 2 GPUs via `kaggle_train.py`.

| Step | Output |
|---|---|
| 0 — Resample | 5 MTF parquets per symbol (5M/15M/1H/4H/1D) |
| 1–3 — Clean/Align | Aligned multi-asset parquet with macro |
| 4 — Features | Engineered feature columns |
| 5 — Split | Train/val/test (70/15/15, time-based) |
| 6 — Backtest | 3-round reinforcement backtest + trade journal |
| 7a — Train | GRU + Regime weights |
| 7b — Train | Quality + RL weights (requires step 6 journal) |
| 8 — Validate | Metrics |

---

## Current Limitations

| Issue | Impact |
|-------|--------|
| RL policy needs action diversity | Needs ≥200 journal trades + entropy tuning |
| HTF BIAS_NEUTRAL recall ~30-38% | NEUTRAL regime under-predicted; being improved |
| VectorStore broken | numpy import bug + dim mismatch; being fixed |
