# Project Description — Multi-Bot ICT/Smart Money Trading System

Updated 2026-04-18.

---

## Executive Summary

A production-grade, fully containerised 5-bot automated trading system for Forex and Gold. Uses Inner Circle Trader (ICT) and Smart Money Concepts (SMC) as the primary signal filter. Five session-aware strategy bots each target a specific market window and instrument set. A Hybrid ML + RL layer (GRU-LSTM, RegimeClassifier, QualityScorer, FinBERT, PPO) is **always active** — `ML_ENABLED=true` is the default for both live engine and backtesting.

Broker: Capital.com REST API (`CAPITAL_ENV=demo`, `PAPER_TRADING=true` default).

> **ML is required.** All 5 models must be trained before starting the engine. Run the Kaggle training pipeline or `retrain_incremental.py` from the trading-engine directory — see `docs/TRAINING_AND_BACKTEST.md`.

---

## Bot Roles

### Trader 1 — NY Session EMA Trend Pullback
- Session: 13:00–17:00 UTC (hard close 17:00)
- Symbols: All except EURGBP
- Entry: H1 EMA stack aligned + ADX > 20 + pullback to EMA zone + candle body ratio > 0.25 + **regime TRENDING_UP/DOWN** required
- Gate: EV ≥ 0.10 from QualityScorer + RL approval
- Max 2 trades/session, 10-min cooldown
- File: `trading-engine/traders/trader_1_ny_ema.py`

### Trader 2 — Structure Break + FVG Continuation
- Session: London 07:00–12:00 + NY 13:00–18:00 (dead zone 12–13 blocked)
- Symbols: All 11
- Entry: H4 BOS establishes directional bias, price retraces into open FVG (max age 20 bars), R:R ≥ 2.0 pre-checked, news block 15 min
- Max 2 trades/session, 15-min cooldown
- File: `trading-engine/traders/trader_2_fvg_bos.py`

### Trader 3 — London Breakout + Liquidity Sweep
- Session: 07:00–10:00 UTC (hard close 12:00)
- Symbols: All except USDJPY
- Entry: Asian range sweep (wick/body ratio > 1.5) + volume > SMA × 1.3 + ADX > 20 + **regime TRENDING_UP or TRENDING_DOWN only** (RANGING/VOLATILE blocked)
- Max 1 trade/session per symbol
- File: `trading-engine/traders/trader_3_london_bo.py`

### Trader 4 — News Structural Breakout
- Session: Any (news happens at any hour)
- Symbols: EURUSD, GBPUSD, XAUUSD
- Entry: `news_in_15min` = True + FinBERT `sentiment_score ≥ 0.65` (hard gate) + structural breakout in sentiment direction
- Hard close 60 min post-news; max 1 trade/session
- File: `trading-engine/traders/trader_4_news_momentum.py`

### Trader 5 — Asian Range Mean Reversion
- Session: 02:00–06:45 UTC (hard close 06:45)
- Symbols: USDJPY, EURJPY, AUDJPY, EURUSD, AUDUSD (XAUUSD and GBPJPY excluded)
- Entry: **regime RANGING mandatory** + ATR expansion < 1.3× + price at range extreme + dual oscillator (RSI + Stoch)
- Max 2 trades/session, 30-min cooldown
- File: `trading-engine/traders/trader_5_asian_mr.py`

---

## Dead Zone

**12:00–13:00 UTC** — all 5 traders blocked. No trades during London/NY overlap transition regardless of signal quality.

---

## Operating Flow

```
1. DataFetcher polls Capital.com REST API → MARKET_DATA events (Redis pub/sub)
2. SignalPipeline.process_bar() runs ML inference (RegimeClassifier → GRU → QualityScorer) and all 5 traders
3. Each trader enforces 8 guards → _compute_signal() → EV gate (Guard 7) → RL gate (Guard 8) → ensemble score
4. Approved signal → RiskEngine.check_pre_trade() → ExecutionEngine
5. PaperTradingService simulates fill (slippage + commission) or BrokerConnector places live order
6. TRADE_EXECUTED → TradeJournal writes CSV + JSONL
7. Journals feed back into scheduled ML retraining (Quality + RL weekly, Regime + GRU monthly)
```

---

## Conflict Management

- **Capital isolation**: each trader has `20%` of account balance (`CAPITAL_PER_TRADER=0.20`)
- **Dead zone**: 12:00–13:00 UTC blocks all traders every day
- **Session hard closes**: T1@17:00, T2@18:00, T3 window ends 10:00, T4@60min post-news, T5@06:45
- **Per-symbol cooldown**: prevents re-entry on the same symbol within cooldown window
- **Session trade caps**: 1–2 trades max per trader per session
- **Circuit breaker**: daily loss > 2% or drawdown > 8% halts trading

---

## Trade Journals

Every executed trade is recorded in two formats:

**Clean CSV** (`logs/trade_journal.csv`):
```
timestamp, trader, symbol, side, size, entry, stop_loss, take_profit, rr_ratio, confidence, pnl, commission
```

**Detailed JSONL** (`logs/trade_journal_detailed.jsonl`):
Full record including session, exit_reason, correlation_id, `state_at_entry` (42-dim RL vector), `rl_action`, `ml_model_scores`, signal_metadata, `ev`, `expected_variance`.

Both journals feed into the scheduled ML retraining pipeline and PPO reward buffering.

---

## ML/AI Layer (always active)

`ML_ENABLED=true` is the **default**. All 5 models must be trained before starting the engine. Models raise `ModelNotTrainedError` (not a silent fallback) if weights are missing.

| Model | Framework | Algorithm | Role |
|---|---|---|---|
| RegimeClassifier | PyTorch GPU | MLP 59→128→64→4 + residual, BatchNorm, GELU | TRENDING_UP/DOWN/RANGING/VOLATILE |
| GRU-LSTM | PyTorch GPU | GRU(256, 2L) + LayerNorm → 3 heads | p_bull, p_bear, magnitude, uncertainty |
| QualityScorer | PyTorch GPU | MLP 17→64→32→1, Huber loss | Expected value in R-multiples |
| SentimentModel | HuggingFace | FinBERT primary + VADER fallback | News/macro directional bias |
| RLAgent | stable-baselines3 | PPO, 42-dim state, 16 actions | Strategy selector + EV threshold |

**Incremental retraining (warm-start):** all models detect existing weights and continue training at 5× lower LR rather than reinitialising. This preserves knowledge from 7.4M GRU training sequences and 619k regime samples.

Ensemble: `(ict_score × 0.5 + ml_score × 0.5 + sentiment_bonus) × regime_mult`

See `docs/models.md` for full model documentation.

---

## Latest Backtest Results (2026-04-18, ML_ENABLED=true)

Period: Jan 2021 – Aug 2024 · Capital: $10,000 · 3 reinforcement rounds

| Round | Trades | Win Rate | Profit Factor | Sharpe | Max DD | Return |
|---|---|---|---|---|---|---|
| Round 1 | 2,712 | 45.5% | 1.99 | 3.50 | 2.72% | +20.1% |
| Round 2 | 2,722 | 44.9% | ~1.87 | ~3.3 | ~2.5% | ~18.4% |
| Round 3 | 2,769 | 44.9% | 1.93 | 3.40 | 2.45% | +18.0% |

Best symbol: GBPUSD (1,312 trades, 47.9% WR). High-conviction trades (confidence ≥ 0.90): 67% WR on 291 trades.

> **Note:** Quality score is currently 0.0 on all trades — the EV pipeline is wired but the scorer output is not reaching the signal path. Once fixed, PF should improve further.

---

## Trading Pairs

Active (all traders, subject to per-trader exclusions):
`EURUSD GBPUSD USDJPY AUDUSD NZDUSD USDCAD USDCHF EURGBP EURJPY GBPJPY XAUUSD`

---

## Data Pipeline

A 9-step offline pipeline in `pipeline/` produces training datasets, backtest inputs, and trained model weights. Runs on Kaggle T4 × 2 GPUs:

| Step | Script | Output |
|---|---|---|
| 0 | `step0_resample.py` | 5 MTF parquets per symbol (5M/15M/1H/4H/1D) |
| 1 | `step1_inventory.py` | `processed_data/raw_inventory.json` |
| 2 | `step2_clean.py` | Clean parquets with macro columns merged |
| 3 | `step3_align.py` | `processed_data/aligned_multi_asset.parquet` |
| 4 | `step4_features.py` | 200+ engineered features |
| 5 | `step5_split.py` | Train/val/test parquets (70/15/15, time-based) |
| 6 | `step6_backtest.py` | 3-round reinforcement backtest with warm-start retraining |
| 7 | `step7_train.py` | GRU + Regime weights (Kaggle GPU) |
| 7b | `step7b_train.py` | Quality + RL weights (Kaggle GPU) |
| 8 | `step8_push_to_github.py` | Push weights + metrics to GitHub |

**Training data:** 11 symbols, Jan 2016–Aug 2024. GRU trained on 7,452,801 sequences across 44 symbol/timeframe combos.
