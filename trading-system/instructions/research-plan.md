# Research & Implementation Plan

Updated 2026-04-06. **SUPERSEDED 2026-04-24.**

> The 5 ICT rule-based traders described below have been removed. ICT concepts are now
> encoded as numeric features in `SEQUENCE_FEATURES` (74 features) and learned by the GRU-LSTM.
> Regime class names used below (TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE, CONSOLIDATION)
> are old and no longer exist. Current HTF classes: BIAS_UP/BIAS_DOWN/BIAS_NEUTRAL. Current LTF
> classes: TRENDING/RANGING/CONSOLIDATING/VOLATILE.
> This file is retained as background context. For current architecture see `docs/system_architecture.md`.

---

## ICT / Smart Money Foundations (Implemented)

- Order Blocks: last opposing candle before impulse → institutional footprint
- Fair Value Gaps: 3-candle imbalance → price revisits for equilibrium
  - Canonical detection: `high.shift(2) < low` (bull); `low.shift(2) > high` (bear) — vectorized
- Liquidity Sweeps: stop-hunt beyond key levels → institutional positioning
- Break of Structure (BOS): directional confirmation via `rolling(window).max/min(center=True)`
- Change of Character (CHoCH): early reversal signal (BOS against prevailing structure)
- All indicators vectorized in `indicators/market_structure.py` — no `.at[i]` integer indexing anywhere

---

## Strategy Implementations (Current Status — 5 traders)

### NY EMA Trend Pullback (Trader 1) — COMPLETE (Quant v2)
- H1 EMA stack (20/50/200) alignment + ADX > 20 + London bias
- **Regime gate**: TRENDING_UP (buy) or TRENDING_DOWN (sell) — mandatory when ML_ENABLED
- Hard RSI 40–55 band removed → replaced by `p_win = QualityScorer.predict()` + EV filter
- New features: `pullback_depth`, `candle_body_ratio`, `dist_ema50_atr`
- All candidates logged to `logs/candidate_log.csv` before gates fire (for stat analysis)

### Structure Break + FVG Continuation (Trader 2) — COMPLETE (unchanged)
- H4 BOS directional bias + FVG registry (max age 20 bars) + R:R ≥ 2.0 pre-check
- News block 15 min; hard close 18:00 UTC
- Max 2 trades/session, 15-min cooldown

### London Breakout + Liquidity Sweep (Trader 3) — COMPLETE (Quant v2)
- Asian range cached per symbol (00–07 UTC); sweep detection (wick/body > 1.5) + volume > SMA × 1.3
- **Regime gate**: TRENDING_UP or TRENDING_DOWN only — RANGING and VOLATILE blocked
- New features: `breakout_strength`, `fakeout_prob` (>55% blocks trade), `range_compression`
- `p_win` gate + EV filter replaces simple ML_QUALITY_THRESHOLD binary check
- Max 1 trade/session per symbol

### News Structural Breakout (Trader 4) — COMPLETE (unchanged); live news feed required for backtest
- FinBERT sentiment ≥ 0.65 HARD gate + structural BOS in sentiment direction
- NOT a fade — trades continuation of news impulse
- Hard close 60 min post-news; max 1 trade/session

### Asian Range Mean Reversion (Trader 5) — COMPLETE (Quant v2)
- Hard ADX < 25 gate removed → replaced by **regime RANGING** (mandatory, ML_ENABLED) + `volatility_stable` flag (ATR expansion < 1.3×)
- New features: `distance_from_mean_atr`, `range_age_bars`, `volatility_expansion`
- Dual oscillator softened: one signal OK if `p_win ≥ 0.65`; both required otherwise
- `p_win` gate + EV filter; XAUUSD excluded; hard close 06:45 UTC

---

## ML / AI Layer

**ML_ENABLED=true is the default.** The engine will not start correctly without trained weights. See `docs/TRAINING_AND_BACKTEST.md` for the full training procedure.

### 5 Models — Implementation and Training Status

| Model | Algorithm | Status |
|---|---|---|
| GRULSTMPredictor | PyTorch GRU(64,2L)→LSTM(128,2L)→3 heads, 74 features, temperature.pt | **TRAINED** — `weights/gru_lstm/model.pt` |
| RegimeClassifier (HTF) | PyTorch MLP 34→128→64→3, BIAS_UP/DOWN/NEUTRAL | **TRAINED** — `weights/regime_htf.pkl` |
| RegimeClassifier (LTF) | PyTorch MLP 18→128→64→4, TRENDING/RANGING/CONSOLIDATING/VOLATILE | **TRAINED** — `weights/regime_ltf.pkl` |
| QualityScorer | PyTorch MLP 17→64→32→1, class-weighted Huber, EV regressor | **TRAINED** — `weights/quality_scorer.pkl` |
| SentimentModel | FinBERT (ProsusAI/finbert) + VADER | Pre-trained — available immediately |
| RLAgent | PPO via stable-baselines3, CPU, 43-dim state, **16 actions** | **TRAINED** — `weights/rl_ppo/model.zip` |

> **Note:** LightGBM and XGBoost have been removed. All models are now PyTorch.

### Training Infrastructure (Pipeline)
- All training runs through `pipeline/step7_train.py` → `trading-engine/scripts/retrain_incremental.py`
- Models save weights to `trading-engine/weights/` (cwd-relative when run from engine dir)
- Weights copied to `trading-system/weights/` by step 7 for step 8 validation
- GRU training timeout: 2 hours (increased from 30 min after initial timeout)
- TensorFlow: `tensorflow-cpu 2.21.0` installed in `.venv`

### Constraint (carried forward)
- All model training uses real market data (OHLCV, news, ForexFactory calendar)
- No synthetic or AI-generated outputs used as training labels
- Journal trade outcomes (paper or live) are the primary quality scorer labels

### Continuous Learning Pipeline
- Weekly cron retraining (`model-retrainer` Docker service, Sunday 02:00 UTC)
- `retrain_incremental.py --model [gru|regime|quality|rl|sentiment|all]`
- Hot-reload: `BaseModel.reload_if_updated()` checks mtime every 5 min — no container restart
- Weights backed up before overwrite (last 5 kept in `weights/backups/`)

### Fail-Loud Design (enforced in all traders)
- All models raise `ModelNotTrainedError(RuntimeError)` when weights absent — no fallback values
- When `ML_ENABLED=true`, missing `regime` or `quality_score` in `ml_predictions` raises `RuntimeError` immediately inside the trader — the signal is not computed
- `ModelNotTrainedError` propagates up through `signal_pipeline.py` to be logged at engine level, not swallowed

### Quant Analytics Layer (new — 2026-04-06)
- `services/candidate_logger.py` — logs all rule-based candidates (pre-gate) to `logs/candidate_log.csv`
- `services/quant_analytics.py` — three components:
  - `StatisticalBinningEngine`: feature → bin → win rate / avg-R lookup
  - `EVFilter`: EV = (p_win × avg_win) − ((1 − p_win) × avg_loss) gate
  - `ConfidenceCalibrator`: validates p_win → actual win rate is monotonic; flags unreliable models
- `services/signal_pipeline.py` injects both logger and EV filter into all traders at startup
- `scripts/run_backtest.py` logs backtest trades as candidates and auto-generates `logs/calibration_report.json`

---

## Data Pipeline (Implemented — 2026-04-03)

### 9-Step Pipeline (`pipeline/` directory)

| Step | Script | Status | Output |
|---|---|---|---|
| 0 | `step0_resample.py` | **DONE** | 11 symbols × 5 TFs (5M/15M/1H/4H/1D) parquets |
| 1 | `step1_inventory.py` | **DONE** | `raw_inventory.json` (215 files, 45 symbols) |
| 2 | `step2_clean.py` | **DONE** | 6 clean 15M parquets (1.57M total bars) |
| 3 | `step3_align.py` | **DONE** | `aligned_multi_asset.parquet` (235k bars, 102 cols) |
| 4 | `step4_features.py` | **DONE** | `feature_engineered.parquet` (235k bars, 180 features) |
| 5 | `step5_split.py` | **DONE** | Train/val/test split (70/15/15 time-based, no leakage) |
| 6 | `step6_backtest.py` | **DONE** | Backtest results for all 5 traders, 10yr range |
| 7 | `step7_train.py` | **DONE** (GRU pending) | regime ✓, quality ✓, rl ✓, gru in-progress |
| 8 | `step8_validate.py` | **DONE** | `critic_report.json` |

### Key Pipeline Fixes Applied
- `pyarrow.Table.from_pandas(nthreads=1)` everywhere — prevents `can't start new thread` on constrained system
- `df.replace([np.inf,-np.inf])` done column-by-column — prevents 300MB contiguous allocation failure
- `ulimit -v 4000000` (4 GB) — allows XGBoost/TF to mmap their shared libraries
- Step 6 exports 15M + 1H + 4H + 1D CSVs to `trading-engine/training_data/` so retrain scripts find them

### Data Coverage
| Source | Symbols | Timeframe | Bars |
|---|---|---|---|
| Forex M1 histdata | AUDUSD, EURGBP, EURJPY, EURUSD, GBPJPY, GBPUSD, NZDUSD, USDCAD, USDCHF, USDJPY | 2016-2026 | ~3.5M/pair |
| XAUUSD M1 histdata | XAUUSD | 2009-2026 | 5.97M |
| Fundamental | FEDFUNDS, CPI, yield curve | Daily | — |
| Indices | SPX, DAX, NIKKEI, VIX | Daily | — |

---

## Decision Hierarchy (Implemented in All Traders)

1. **Guard 0**: Symbol exclusion (`EXCLUDED_SYMBOLS`)
2. **Guard 1**: Session window (start/end UTC hours)
3. **Guard 2**: Dead zone (12:00–13:00 UTC, all traders)
4. **Guard 3**: Per-symbol cooldown
5. **Guard 4**: Session trade cap
6. **Guard 5**: Circuit breaker (daily loss + drawdown)
7. **Guard 6**: Spread gate
8. **Guard 7**: ML quality gate (quality_score + direction probability thresholds)
9. **Guard 8**: RL action gate (PPO must approve trader's action)
10. **_compute_signal()**: ICT/SMC rule-based signal logic
11. **Ensemble score**: `(ict_score × 0.5 + ml_score × 0.5 + sentiment_bonus) × regime_mult ≥ 0.5`
12. **Confidence gate**: ≥ 0.55 before SIGNAL_GENERATED

---

## Risk Management (Implemented)

- Capital isolation: `CAPITAL_PER_TRADER=0.20` (20% each × 5 = 100%)
- Fixed fractional sizing: `risk_amount = allocated_capital × RISK_PER_TRADE (1%)`
- Volatility-based SL: ATR-adjusted per signal per trader
- Max daily loss: `MAX_DAILY_LOSS_PCT=0.02` (2% account)
- Max drawdown: `MAX_DRAWDOWN_PCT=0.08` (8% account)
- Max concurrent positions: `MAX_CONCURRENT_POSITIONS=2`
- Spread gates: Forex > 3 pips, Gold > 50 pips → blocked

---

## Tech Stack (Current)

| Layer | Technology | Notes |
|---|---|---|
| Market data | Capital.com REST API | Primary; session auto-refreshes every ~9m50s |
| Data fallback | yfinance | Auto symbol mapping (XAUUSD→GC=F etc.) |
| Execution | Capital.com REST | `CapitalComSession` + `BrokerConnector` |
| Indicators | pandas, numpy | All vectorized in `market_structure.py` |
| ML — time series | PyTorch GPU | GRU-LSTM (GRU(64,2L)→LSTM(128,2L)→3 heads) |
| ML — regime | PyTorch GPU | Dual-cascade MLP (HTF 34→3, LTF 18→4) |
| ML — EV | PyTorch GPU | QualityScorer MLP (17→64→32→1) |
| ML — NLP | HuggingFace (FinBERT) | Sentiment |
| ML — RL | stable-baselines3 PPO (CPU) | RLAgent 43-dim state, 16 actions |
| Infrastructure | Docker Compose (6 services) | `docker-compose.dev.yml` |
| Persistence | PostgreSQL 15, Redis 7 | State, events, journals |
| Frontend | React 18 + MUI + Redux Toolkit | Port 3001 |
| Backend API | FastAPI + Python 3.11 | Port 3000 |
| Logging | Python logging + structured JSON | `utils/observability.py` |
| Monitoring | Prometheus + Grafana | Ports 9090/3002 |
| Pipeline Python | `.venv` (Python 3.12) at `/home/tybobo/Desktop/Multi-Bot/.venv` | pandas 3.0.1, pyarrow 23.0.1 |

---

## KPI Targets

| Timeframe | Target |
|---|---|
| 30 days | Stable paper trading, 50+ trades logged per active trader |
| 60 days | Positive net PnL across portfolio, drawdown < 8% |
| 90 days | ML models trained from journal data; first ML-gated paper run evaluated |
| 120 days | PPO RL agent with 500+ episodes; evaluate vs rule-only baseline |

---

## Backtest Baselines

### Pre-ML Baseline (2026-04-03, rule-only, 10yr 2016-2026)
Raw rule-based signals only — no ML filtering, no regime gate, no EV filter.

| Trader | Trades | Win Rate | PF | Notes |
|---|---|---|---|---|
| T1 (NY EMA) | 5,655 | 40.4% | 0.62 | Hard RSI band allowed noise entries |
| T2 (FVG BOS) | 3,702 | 27.4% | 0.63 | |
| T3 (London BO) | 5,370 | 41.8% | 0.57 | No regime filter; traded all conditions |
| T4 (News Mom.) | 0 | — | — | Requires live news feed |
| T5 (Asian MR) | 12,211 | 14.9% | 0.66 | Over-trading without regime RANGING gate |

All 100% drawdown without ML — expected. **Do not use rule-only mode in production.**

### Target (post quant upgrade + ML_ENABLED=true)
Re-run backtest after GRU-LSTM training completes. Expected improvements:
- T1: fewer trades, higher WR via regime + p_win gate
- T3: RANGING/VOLATILE regime blocks removed → higher quality breakouts only
- T5: volatility_stable + regime RANGING cuts 12k noise trades to ~1-2k quality setups

---

## Remaining Research Directions

### Immediate (next steps)
1. **Train GRU-LSTM** — last model still pending; see `docs/TRAINING_AND_BACKTEST.md` for exact commands
2. **Run backtest with ML_ENABLED=true** to validate quant upgrade performance
3. **Add Capital.com credentials** to `.env` for live OHLCV data (currently falls back to yfinance)
4. **Paper trading baseline**: run 30–60 days with `ML_ENABLED=true` to accumulate journal data for QualityScorer retraining

### Strategy tuning (post-backtest with ML enabled)
- **T1**: Verify regime TRENDING gate reduces false entries; calibration report should show p_win monotonicity
- **T2**: FVG max_age=20 bars — consider shortening for faster markets; WR 27.4% suggests stale FVGs
- **T3**: Verify fakeout_prob correctly identifies unreliable sweep setups; tune 0.55 threshold
- **T4**: Validate news momentum on historical data; needs real ForexFactory calendar feed
- **T5**: volatility_stable gate should fix the 14.9% WR baseline; check EVFilter defaults via calibration
- **Portfolio**: combined drawdown analysis — sessions don't overlap but capital does

### Architecture improvements
- **Category lock**: prevent T1 and T2 opening simultaneously on same currency
- **Monitors directory**: implement chart_monitor, risk_monitor, drawdown_monitor, system_monitor
- **Unit tests**: add pytest coverage for `market_structure.py`, `feature_engine.py`, `base_trader.py`
- **Walk-forward validation**: implement OOS validation in `run_backtest.py`
- **Telegram alerts**: `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` for real-time trade notifications
- **Live trading**: verify Capital.com order flow end-to-end in demo before setting `PAPER_TRADING=false`
- **Tick volume**: step 0 has tick extraction commented out — future enhancement for volume profiling
