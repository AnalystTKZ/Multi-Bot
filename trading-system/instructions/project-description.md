# Project Description — Multi-Bot ICT/Smart Money Trading System

Updated 2026-04-19.

---

## Executive Summary

A production-grade, fully containerised 5-bot automated trading system for Forex and Gold. Uses Inner Circle Trader (ICT) and Smart Money Concepts (SMC) as the primary signal framework. Five session-aware strategy bots each target a specific market window and instrument set. A Hybrid ML + RL layer (GRU-LSTM, RegimeClassifier, QualityScorer EV regressor, PPO) is **always active** — `ML_ENABLED=true` is the default.

Training runs on Kaggle T4 × 2 GPUs. Models warm-start incrementally from existing weights — 7+ years of training history is preserved across each retraining run rather than discarded.

Broker: Capital.com REST API (`CAPITAL_ENV=demo`, `PAPER_TRADING=true` default).

> **ML is required.** All 5 models must be trained before starting the engine. Run the Kaggle pipeline (`kaggle_train.py`) or `retrain_incremental.py` — see `docs/TRAINING_AND_BACKTEST.md`.

---

## Bot Roles

### Trader 1 — NY Session EMA Trend Pullback
- Session: 13:00–17:00 UTC (hard close 17:00)
- Symbols: All except EURGBP
- Entry: 1H EMA stack aligned + ADX > 20 + pullback to EMA zone + candle body ratio > 0.25 + **regime TRENDING_UP/DOWN required**
- Gate: EV ≥ 0.10 + uncertainty ≤ 0.80 + RL approval
- Max 2 trades/session, 10-min cooldown

### Trader 2 — Structure Break + FVG Continuation
- Session: London 07:00–12:00 + NY 13:00–18:00 UTC (dead zone 12–13 blocked)
- Symbols: All 11
- Entry: 4H BOS establishes directional bias, price retraces into open FVG (max age 20 bars), R:R ≥ 2.0 pre-checked, news block 15 min
- Max 2 trades/session, 15-min cooldown

### Trader 3 — London Breakout + Liquidity Sweep
- Session: 07:00–10:00 UTC (hard close 12:00)
- Symbols: All except USDJPY
- Entry: Asian range sweep (wick/body ratio > 1.5) + volume > SMA × 1.3 + ADX > 20 + **regime TRENDING_UP or DOWN only** (RANGING/VOLATILE blocked)
- Max 1 trade/session per symbol

### Trader 4 — News Structural Breakout
- Session: Any (news-driven)
- Symbols: EURUSD, GBPUSD, XAUUSD
- Entry: FinBERT `sentiment_score ≥ 0.65` (hard gate) + structural breakout in sentiment direction
- Hard close 60 min post-news; max 1 trade/session

### Trader 5 — Asian Range Mean Reversion
- Session: 02:00–06:45 UTC (hard close 06:45)
- Symbols: USDJPY, EURJPY, AUDJPY, EURUSD, AUDUSD (XAUUSD and GBPJPY excluded)
- Entry: **regime RANGING mandatory** + ATR expansion < 1.3× + price at range extreme + dual oscillator (RSI + Stoch)
- Max 2 trades/session, 30-min cooldown

---

## Dead Zone

**12:00–13:00 UTC** — all 5 traders blocked every day.

---

## Operating Flow

```
1. DataFetcher polls Capital.com REST API → MARKET_DATA events (Redis pub/sub)
2. SignalPipeline.process_bar() runs ML inference (RegimeClassifier → GRU → QualityScorer → Sentiment)
3. Each trader enforces 8 guards → _compute_signal() → EV gate → RL gate → ensemble score
4. Approved signal → RiskEngine.check_pre_trade() → ExecutionEngine
5. PaperTradingService simulates fill (slippage + commission) or BrokerConnector places live order
6. TRADE_EXECUTED → TradeJournal writes CSV + JSONL
7. Journals feed back into scheduled ML retraining (Quality + RL weekly, Regime + GRU monthly)
```

---

## Conflict Management

- **Capital isolation**: each trader controls 20% of account balance (`CAPITAL_PER_TRADER=0.20`)
- **Dead zone**: 12:00–13:00 UTC blocks all traders
- **Session hard closes**: T1@17:00, T2@18:00, T3 window ends 10:00, T4@60min post-news, T5@06:45
- **Per-symbol cooldown**: prevents re-entry on same symbol within cooldown window
- **Session trade caps**: 1–2 trades max per trader per session
- **Circuit breaker**: daily loss > 2% or drawdown > 8% halts all trading

---

## Trade Journals

**Clean CSV** (`logs/trade_journal.csv`):
```
timestamp, trader, symbol, side, size, entry, stop_loss, take_profit, rr_ratio, confidence, pnl, commission
```

**Detailed JSONL** (`logs/trade_journal_detailed.jsonl`):
Full record including `ev`, `expected_variance`, `exit_reason`, `session`, `ml_model_scores` (p_bull, p_bear, regime, sentiment_backend), `state_at_entry` (42-dim RL state), `rl_action`.

Exit reason values:
- `tp2` — full TP hit (best outcome, highest EV label)
- `tp1` — TP1 hit, held further
- `be_or_trail` — TP1 hit then trailed/stopped
- `sl_full` — stopped at SL
- `time_exit` — timed out (excluded from EV labels)

---

## ML/AI Layer (always active)

| Model | Framework | Algorithm | Role |
|---|---|---|---|
| RegimeClassifier | PyTorch GPU | MLP 59→128→64→4, BatchNorm, GELU, residual | TRENDING_UP/DOWN/RANGING/VOLATILE |
| GRU-LSTM | PyTorch GPU | GRU(256, 2L) + LayerNorm → 3 heads | p_bull, p_bear, magnitude, uncertainty |
| QualityScorer | PyTorch GPU | MLP 17→64→32→1, tiered Huber EV loss | Expected value in R-multiples |
| SentimentModel | HuggingFace | FinBERT primary + VADER fallback | News/macro directional bias |
| RLAgent | stable-baselines3 | PPO, CPU, 42-dim state, 16 actions | Strategy selector + EV threshold |

**EV label tiers** — QualityScorer learns a preference for tp2:

| Exit | EV label |
|---|---|
| `tp2` | `+rr_ratio` |
| `tp1` | `+rr × 0.75` |
| `be_or_trail` | `+rr × 0.4` |
| `sl_*` | `-1.0` |

**Regime persistence filter:** runs shorter than 20 bars (4H) / 48 bars (1H) are zero-weighted to prevent label noise.

**Warm-start retraining:** all models detect existing weights and continue at 5× lower LR. GRU is excluded from per-round reinforcement loop (catastrophic forgetting risk with 7M training sequences).

Ensemble: `(ict_score × 0.5 + ml_score × 0.5 + sentiment_bonus) × regime_mult`

---

## Latest Backtest Results (2026-04-18, ML_ENABLED=true)

Period: Jan 2021 – Aug 2024 · Capital: $10,000 · 3 reinforcement rounds

| Round | Trades | Win Rate | Profit Factor | Sharpe | Max DD | Return |
|---|---|---|---|---|---|---|
| Round 1 | 2,571 | 45.4% | 2.08 | 3.77 | 2.8% | 1,549% |
| Round 2 | 2,610 | 44.8% | 2.04 | 3.70 | 3.6% | 1,458% |
| Round 3 | 2,586 | 45.3% | 2.05 | 3.71 | 2.8% | 1,491% |

All 4 regime classes active. Calibration OK. Trade frequency: 1.0–1.3 trades/day/symbol.

---

## Trading Pairs

`EURUSD GBPUSD USDJPY AUDUSD NZDUSD USDCAD USDCHF EURGBP EURJPY GBPJPY XAUUSD`

---

## Data Pipeline

9-step offline pipeline in `pipeline/`. Runs on Kaggle T4 × 2 GPUs via `kaggle_train.py`.

| Step | Script | Output |
|---|---|---|
| 0 | `step0_resample.py` | 5 MTF parquets per symbol (5M/15M/1H/4H/1D) |
| 1 | `step1_inventory.py` | `processed_data/raw_inventory.json` |
| 2 | `step2_clean.py` | Clean parquets with macro columns merged |
| 3 | `step3_align.py` | `processed_data/aligned_multi_asset.parquet` |
| 4 | `step4_features.py` | 200+ engineered features |
| 5 | `step5_split.py` | Train/val/test parquets (70/15/15, time-based) |
| 6 | `step6_backtest.py` | 3-round reinforcement backtest + warm-start retraining |
| 7 | `step7_train.py` | GRU + Regime weights |
| 7b | `step7b_train.py` | Quality + RL weights |
| 8 | `step8_push_to_github.py` | Push weights + metrics to GitHub |

Training data: 11 symbols, Jan 2016–Feb 2026. GRU trained on 7,081,756 sequences across 44 symbol/timeframe combos.
