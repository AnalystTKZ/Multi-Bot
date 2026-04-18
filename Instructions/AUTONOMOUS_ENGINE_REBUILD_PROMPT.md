# 🔁 Autonomous Ground-Up Trading Engine Replacement
## Claude Code — Self-Healing Multi-Agent Loop

`trading-system/` directory root.**
> The three strategy/system documents @trading-system/docs/forex_gold_trading_system.md  and @trading-system/docs/hybrid_ml_rl_trading_system.md  and @trading-system/docs/CLAUDE.md are also provided alongside this prompt as context.
> The loop does not stop until the new `trading-engine/` is fully built, integrated end-to-end, and all verification gates pass.

---

## WHAT YOU ARE DOING

You are **completely demolishing and rebuilding** `trading-engine/` from scratch.

The new engine implements two systems in full:
1. **Forex + Gold Automated Trading System** — 5 session-aware rule-based strategies
2. **Hybrid ML + RL + Rule-Based Trading System** — GRU-LSTM, LightGBM, XGBoost, FinBERT, PPO layered on top

**What is demolished (delete everything inside `trading-engine/` before writing a single new file):**
- All traders (`trader_1_ema_ict.py`, `trader_2_mean_reversion.py`, `trader_3_breakout.py`, `trader_4_news.py`)
- All models (`price_predictor.py`, `pattern_recognizer.py`, `ml_signal_filter.py`, `sentiment_analyzer.py`, `anomaly_detector.py`, `rl_agent.py`, `base_model.py`)
- All indicators (`ict_indicators.py`)
- All services (`signal_engine.py`, `strategy_manager.py`) — replaced entirely
- `main.py` — replaced entirely
- `requirements.txt` — replaced entirely

**What is preserved absolutely (do not touch, do not read-modify-write, do not drift):**

| Layer | What | Why |
|---|---|---|
| `backend/` | All files | FastAPI API, routes, Redis client, WebSocket manager — unchanged |
| `frontend/` | All files | React/Redux UI — unchanged |
| `docker-compose.yml` | Entire file | Container names, ports, volumes — unchanged |
| `.env` | Entire file | All credentials and flags — unchanged |
| `scripts/` | `run_backtest.py`, `retrain_incremental.py`, `retrain_scheduler.py` — **will be fully rewritten** to match new engine |
| `training_data/` | CSVs | Data files untouched |
| `backtest_results/` | JSON files | Output directory untouched |
| `docs/` | Updated at end | CLAUDE.md updated to reflect new system |

---

## THE CONTRACTS YOU MUST NEVER BREAK

These interfaces connect the new `trading-engine` to the rest of the system. Every single one is non-negotiable.

### Contract 1 — Redis Event Schema
The backend `WebSocket manager` and `state_reader.py` listen for these exact event shapes. Do not rename events. Do not add required fields without backward-compatible defaults.

```python
# MARKET_DATA — published by DataFetcher, consumed by engine
{
    "symbol": "EURUSD",
    "timeframe": "15M",
    "open": float, "high": float, "low": float, "close": float, "volume": float,
    "timestamp": "ISO8601 string"
}

# SIGNAL_GENERATED — published by traders, consumed by ExecutionEngine + backend
{
    "trader_id": "trader_1",          # string, matches backend registration
    "symbol": "EURUSD",
    "side": "buy" | "sell",
    "confidence": float,              # 0.0–1.0
    "stop_loss": float,
    "take_profit": float,
    "correlation_id": "uuid4 string",
    "signal_metadata": {              # arbitrary dict — backend stores as-is
        "strategy": str,
        "session": str,
        "rl_action": int,
        "quality_score": float,
        "p_bull": float,
        "p_bear": float,
        "regime": str,
        "sentiment_score": float,
        "rr_ratio": float
    }
}

# TRADE_EXECUTED — published by PaperTradingService/BrokerConnector
{
    "ticket": "uuid4 string",
    "trader_id": str,
    "symbol": str,
    "side": str,
    "size": float,
    "entry": float,
    "stop_loss": float,
    "take_profit": float,
    "rr_ratio": float,
    "confidence": float,
    "strategy_id": str,
    "timestamp": str,
    "pnl": float | null,              # null on open, filled on close
    "signal_metadata": dict
}

# TRADE_CLOSED — published when position closes
{
    "ticket": str,
    "pnl": float,
    "rr_ratio": float,
    "exit_reason": "tp1" | "tp2" | "sl" | "time_exit" | "manual",
    "trader_id": str,
    "symbol": str,
    "signal_metadata": dict
}
```

### Contract 2 — Redis State Keys
`backend/src/services/state_reader.py` reads these exact keys. Write them identically.

```python
# Engine health
redis.set("engine:status", "running" | "stopped" | "error")
redis.set("engine:last_heartbeat", ISO8601_timestamp)
redis.set("engine:mode", "paper" | "live")

# Per-trader state (used by GET /api/traders/{id}/performance)
redis.hset(f"trader:{trader_id}:state", mapping={
    "status": "active" | "paused" | "cooldown",
    "trades_today": int,
    "pnl_today": float,
    "win_rate": float,
    "last_signal": ISO8601_timestamp
})

# Per-trader performance (used by GET /api/traders/{id}/performance)
redis.hset(f"trader:{trader_id}:performance", mapping={
    "trader_id": trader_id,
    "monthly_pnl": float,
    "profit_factor": float,
    "total_trades": int,
    "win_rate": float,
    "avg_rr": float,
    "max_drawdown": float
})

# Open positions
redis.hset("positions:open", ticket, json.dumps(position_dict))

# ML model status (used by GET /api/ml/models)
redis.hset(f"ml:model:{model_id}", mapping={
    "model_id": model_id,
    "name": str,
    "status": "active" | "training" | "error",
    "accuracy": float,
    "last_trained": ISO8601_timestamp
})

# RL agent state (used by GET /api/ml/rl-agent)
redis.hset("ml:rl_agent", mapping={
    "algorithm": "PPO",
    "state_dim": 42,
    "action_dim": 6,
    "episodes": int,
    "avg_reward": float,
    "last_updated": ISO8601_timestamp
})

# Context/regime per symbol
redis.set(f"context:{symbol}", json.dumps({"regime": str, "bias": str, "confidence": float}))
```

### Contract 3 — Health Server
The backend polls `http://trading-engine:8000/health`. Must return:
```json
{"status": "ok", "timestamp": "ISO8601", "traders": 5, "mode": "paper"}
```

### Contract 4 — Trade Journal Files
`scripts/retrain_incremental.py` reads these. Format must be preserved exactly.

**`logs/trade_journal.csv`** columns:
```
timestamp,trader,symbol,side,size,entry,stop_loss,take_profit,rr_ratio,confidence,pnl,commission
```

**`logs/trade_journal_detailed.jsonl`** fields:
```json
{
    "timestamp": str, "trader": str, "symbol": str, "side": str,
    "size": float, "entry": float, "stop_loss": float, "take_profit": float,
    "rr_ratio": float, "confidence": float, "pnl": float, "commission": float,
    "strategy": str, "timeframe": str, "session": str,
    "smc_score": float, "ict_conditions": dict,
    "ml_model_scores": {
        "p_bull": float, "p_bear": float, "quality_score": float,
        "regime": str, "sentiment_score": float, "rl_action": int
    },
    "entry_reason": str, "exit_reason": str,
    "correlation_id": str, "signal_metadata": dict,
    "state_at_entry": list,    ← 42-element RL state vector
    "rl_action": int
}
```

### Contract 5 — Backtest Output Format
`backend/src/routes/system.py` reads `backtest_results/*.json` with this schema:
```json
{
    "run_at": "ISO8601",
    "start": "YYYY-MM-DD",
    "end": "YYYY-MM-DD",
    "config": {"traders": [...], "symbols": [...], "initial_capital": float},
    "results": {
        "trader_id": {
            "trades": int, "win_rate": float, "total_return": float,
            "profit_factor": float, "max_drawdown": float, "sharpe": float
        }
    },
    "trade_log": [...]
}
```

### Contract 6 — Trader IDs (registered in backend)
The backend has these trader IDs hardcoded in `routes/traders.py`. The new engine MUST use exactly these IDs:
```
trader_1  →  NY Session EMA Trend Pullback
trader_2  →  Structure Break + FVG Continuation
trader_3  →  London Breakout + Liquidity Sweep
trader_4  →  High-Impact News Momentum
trader_5  →  Asian Range Mean Reversion  (NEW — backend needs one-line addition)
```
Add the strategy each trader trades as well
### Contract 7 — ML Flags
Honour `ML_ENABLED` environment variable exactly as before:
- `ML_ENABLED=false` → all ML models skipped at startup; zero imports; engine runs on rules only
- `ML_ENABLED=true` → full ML + RL pipeline active
- `PAPER_TRADING=true/false` → routes to `PaperTradingService` vs `BrokerConnector`

---

## AGENT ROSTER

Label every reasoning block with its agent. Do not skip. Do not collapse two agents into one.

```
ARCHITECT_AGENT    → designs the complete new directory tree before writing files
READER_AGENT       → reads all contracts and existing keep-files before coding
CODER_AGENT        → writes one complete file at a time, dependency order
TESTER_AGENT       → runs verification after every file or logical group
INTEGRATOR_AGENT   → connects the engine to backend/broker/Redis/journals
CRITIC_AGENT       → challenges all implementations for bugs, leakage, missing edge cases
FIXER_AGENT        → applies all CRITIC fixes; re-runs relevant tests
EVALUATOR_AGENT    → scores each subsystem 0–10; reports composite
LOOP_CONTROLLER    → decides CONTINUE | COMPLETE based on scores and gate pass/fail
```

---

## AUTONOMOUS LOOP

```
SET iteration = 1
SET MAX_ITERATIONS = 25
SET PASS_THRESHOLD = 8.0       ← all EVALUATOR component scores must reach this

WHILE True:

    ARCHITECT_AGENT:  confirm or refine this iteration's build scope
    READER_AGENT:     read any keep-files relevant to this iteration's work
    CODER_AGENT:      build all files planned for this iteration
    TESTER_AGENT:     run syntax + import + logic + contract tests
    INTEGRATOR_AGENT: verify engine ↔ backend ↔ broker ↔ journal connections
    CRITIC_AGENT:     produce issue list (zero tolerance for contract violations)
    FIXER_AGENT:      fix every issue; re-run affected tests
    EVALUATOR_AGENT:  score all subsystems built so far

    IF all scores >= PASS_THRESHOLD AND all gate checks below pass:
        LOOP_CONTROLLER: emit "✅ ENGINE REPLACEMENT COMPLETE"
        update docs/CLAUDE.md
        STOP

    ELSE IF iteration >= MAX_ITERATIONS:
        LOOP_CONTROLLER: emit "⛔ CEILING HIT"
        list every unresolved issue with file + line reference
        STOP

    ELSE:
        LOOP_CONTROLLER: set next iteration scope = lowest-scoring subsystem
        iteration += 1
```

---

## PHASE 0 — DEMOLITION + ARCHITECT

### Step 0A — Demolition
```bash
# Back up old engine (safety net)
cp -r trading-engine/ trading-engine-backup-$(date +%Y%m%d)/

# Delete everything inside trading-engine/ (not the directory itself — docker-compose volume mount)
find trading-engine/ -mindepth 1 -delete
echo "trading-engine/ cleared"
```

### Step 0B — ARCHITECT_AGENT: New Directory Tree

Build this exact structure from scratch:

```
trading-engine/
├── main.py                          ← NEW ProductionTradingEngine (5 traders + ML + RL)
├── requirements.txt                 ← COMPLETE replacement
├── config/
│   └── settings.py                  ← All env vars; session windows; thresholds
├── services/
│   ├── event_bus.py                 ← KEEP CONTRACT: Redis pub/sub, EventType enum
│   ├── state_manager.py             ← KEEP CONTRACT: Redis state writer
│   ├── data_fetcher.py              ← KEEP CONTRACT: Capital.com + yfinance fallback
│   ├── broker_connector.py          ← KEEP CONTRACT: Capital.com REST
│   ├── order_executor.py            ← KEEP CONTRACT: ExecutionRequest dataclass
│   ├── paper_trading_service.py     ← KEEP CONTRACT: slippage + commission sim
│   ├── risk_engine.py               ← REPLACED: new position sizing + circuit breakers
│   ├── trade_journal.py             ← REPLACED: new JSONL fields (state_at_entry etc.)
│   ├── feature_engine.py            ← NEW: all feature computation for ML + RL
│   ├── session_manager.py           ← NEW: UTC session detection + scheduling
│   ├── signal_pipeline.py           ← NEW: replaces signal_engine.py + strategy_manager.py
│   └── news_service.py              ← NEW: unified news feed + pre-news zone tracking
├── indicators/
│   └── market_structure.py          ← NEW: complete replacement for ict_indicators.py
│                                       Vectorized: FVG, BOS, sweeps, OB, ADX, EMA stack
├── models/
│   ├── base_model.py                ← NEW: hot-reload pattern preserved; new base class
│   ├── gru_lstm_predictor.py        ← NEW: GRU-LSTM hybrid (direction + entry depth)
│   ├── regime_classifier.py         ← NEW: LightGBM 4-class regime
│   ├── quality_scorer.py            ← NEW: XGBoost trade quality scorer
│   ├── sentiment_model.py           ← NEW: FinBERT + VADER fallback
│   ├── rl_agent.py                  ← NEW: PPO (Stable-Baselines3) 42-dim state
│   └── weights/
│       ├── gru_lstm/                ← SavedModel or .pt weights dir
│       ├── regime_classifier.pkl
│       ├── quality_scorer.pkl
│       ├── sentiment_model.pkl      ← FinBERT cache / VADER lexicon
│       ├── rl_ppo/                  ← PPO model dir
│       └── backups/
├── traders/
│   ├── base_trader.py               ← NEW: capital isolation, cooldown, session-aware
│   ├── trader_1_ny_ema.py           ← NEW: NY Session EMA Trend Pullback
│   ├── trader_2_fvg_bos.py         ← NEW: Structure Break + FVG Continuation
│   ├── trader_3_london_bo.py        ← NEW: London Breakout + Liquidity Sweep
│   ├── trader_4_news_momentum.py    ← NEW: High-Impact News Momentum
│   ├── trader_5_asian_mr.py         ← NEW: Asian Range Mean Reversion
│   └── __init__.py                  ← Exports all 5 traders
├── monitors/
│   ├── chart_monitor.py             ← REBUILT: publishes to same Redis keys
│   ├── news_monitor.py              ← REBUILT: feeds news_service.py
│   ├── risk_monitor.py              ← REBUILT: circuit breaker checks
│   ├── drawdown_monitor.py          ← REBUILT: per-trader + system DD tracking
│   └── system_monitor.py            ← REBUILT: heartbeat, health endpoint
└── logs/
    ├── trade_journal.csv            ← Same columns as Contract 4
    └── trade_journal_detailed.jsonl ← Same fields + new ML fields (Contract 4)
```

---

## PHASE 1 — REQUIREMENTS + CONFIG

### `requirements.txt` (complete, no omissions)

```
# Core
pandas>=2.1.0
numpy>=1.26.0
python-dotenv>=1.0.0
pydantic>=2.5.0
pydantic-settings>=2.1.0

# Async + HTTP
aiohttp>=3.9.0
httpx>=0.26.0
fastapi>=0.109.0
uvicorn>=0.27.0

# Redis
redis>=5.0.0

# Data
yfinance>=0.2.36
pandas-ta>=0.3.14b

# ML — Gradient Boosting
lightgbm>=4.3.0
xgboost>=2.0.3
scikit-learn>=1.4.0

# ML — Deep Learning (GRU-LSTM)
tensorflow>=2.15.0
# NOTE: use tensorflow-cpu if GPU not available; same API

# ML — NLP Sentiment
transformers>=4.40.0
torch>=2.2.0
sentencepiece>=0.2.0

# RL
stable-baselines3>=2.3.0
gymnasium>=0.29.1

# Explainability
shap>=0.44.0

# Scheduling + utilities
apscheduler>=3.10.4
pytz>=2024.1

# Monitoring
prometheus-client>=0.20.0

# Broker
requests>=2.31.0

# Logging
structlog>=24.1.0
```

### `config/settings.py`

```python
from pydantic_settings import BaseSettings
from typing import List
import pytz

class Settings(BaseSettings):
    # Broker
    BROKER_TYPE: str = "capital"
    CAPITAL_API_KEY: str = ""
    CAPITAL_IDENTIFIER: str = ""
    CAPITAL_PASSWORD: str = ""
    CAPITAL_ENV: str = "demo"

    # Trading mode
    PAPER_TRADING: bool = True
    ML_ENABLED: bool = False

    # Symbols (matches existing system exactly)
    TRADING_PAIRS: List[str] = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","XAUUSD"]

    # Capital allocation per trader (% of total)
    CAPITAL_PER_TRADER: float = 0.20   # 20% each × 5 traders = 100%
    RISK_PER_TRADE: float = 0.01       # 1% per trade

    # Risk limits
    MAX_DAILY_LOSS_PCT: float = 0.02
    MAX_DRAWDOWN_PCT: float = 0.08
    MAX_CONCURRENT_POSITIONS: int = 2

    # ML thresholds
    ML_QUALITY_THRESHOLD_DEFAULT: float = 0.55
    ML_DIRECTION_THRESHOLD: float = 0.58
    RL_ACTION_CONFIDENCE_MIN: float = 0.45

    # Session windows (UTC hours)
    ASIAN_SESSION_START: int = 0
    ASIAN_SESSION_END: int = 7
    ASIAN_TRADE_START: int = 2
    ASIAN_TRADE_END_HARD: int = 6
    ASIAN_HARD_CLOSE_MINUTE: int = 45
    LONDON_SESSION_START: int = 7
    LONDON_SESSION_END: int = 12
    NY_SESSION_START: int = 13
    NY_SESSION_END: int = 18
    DEAD_ZONE_START: int = 12
    DEAD_ZONE_END: int = 13

    # Redis
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""

    # Logging
    LOG_LEVEL: str = "INFO"

    # Retrain schedule
    RETRAIN_DAY: str = "sunday"
    RETRAIN_HOUR: int = 2

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
UTC = pytz.utc
```

---

## PHASE 2 — INDICATORS (`indicators/market_structure.py`)

This is a **complete replacement** for `ict_indicators.py`. All functions are vectorized — no `.at[i]` integer indexing on datetime-indexed DataFrames (this was a known bug in the old system).

**CODER_AGENT must implement all of these functions:**

```python
"""
market_structure.py — Vectorized market structure indicators for Forex + Gold.
All functions accept a pandas DataFrame with columns: open, high, low, close, volume.
All functions are pure (no side effects). No .at[i] integer indexing.
"""

import pandas as pd
import numpy as np
from typing import Tuple

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing (ewm span=period)."""

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI via Wilder's smoothing."""

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ADX — +DI, -DI, and ADX line. Returns ADX series only."""

def compute_stochastic(df: pd.DataFrame, k: int = 14,
                        smooth_k: int = 3, d: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Stochastic %K and %D. Returns (stoch_k, stoch_d)."""

def compute_bollinger_bands(series: pd.Series, period: int = 20,
                             std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, middle, lower)."""

def compute_ema_stack_score(df: pd.DataFrame) -> pd.Series:
    """
    EMA alignment score.
    +2: ema21 > ema50 > ema200 (bull trend)
    -2: ema21 < ema50 < ema200 (bear trend)
     0: mixed / transitioning
    """

def detect_fair_value_gaps(df: pd.DataFrame,
                            min_atr_ratio: float = 0.25) -> pd.DataFrame:
    """
    Detect 3-candle Fair Value Gaps.
    Returns DataFrame with columns:
      fvg_bull (bool), fvg_bear (bool),
      fvg_bull_top (float), fvg_bull_bottom (float),
      fvg_bear_top (float), fvg_bear_bottom (float)
    All vectorized — no loops.
    """

def detect_break_of_structure(df: pd.DataFrame,
                               swing_n: int = 5) -> pd.DataFrame:
    """
    Vectorized BOS detection.
    Returns DataFrame with:
      swing_high (float, NaN where not swing), swing_low (float, NaN),
      bos_bull (bool), bos_bear (bool),
      last_swing_high (float, forward-filled), last_swing_low (float, forward-filled)
    """

def detect_liquidity_sweeps(df: pd.DataFrame,
                             lookback: int = 20) -> pd.DataFrame:
    """
    Detect stop hunts: price briefly exceeds recent high/low then reverses.
    Returns DataFrame with:
      sweep_bull (bool): swept below recent low then closed above
      sweep_bear (bool): swept above recent high then closed below
      sweep_low_level (float): level swept
      sweep_high_level (float): level swept
    """

def detect_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Last opposing candle before an impulse move.
    Returns: ob_bull (bool), ob_bear (bool), ob_high (float), ob_low (float)
    """

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function. Adds all indicator columns to df and returns it.
    Column naming convention:
      ema_21, ema_50, ema_200, atr_14, rsi_14, adx_14,
      stoch_k, stoch_d, bb_upper, bb_mid, bb_lower, bb_width,
      ema_stack, fvg_bull, fvg_bear, bos_bull, bos_bear,
      sweep_bull, sweep_bear, ob_bull, ob_bear
    """
```

**CODER_AGENT critical requirements:**
- `detect_fair_value_gaps`: `candle[i-2].high < candle[i].low` — use `df['high'].shift(2) < df['low']` vectorized
- `detect_break_of_structure`: swing high = `df['high'] == df['high'].rolling(2*swing_n+1, center=True).max()` — no loops
- `compute_atr`: `tr = max(H-L, |H-prev_C|, |L-prev_C|)` vectorized with `pd.concat().max(axis=1)`
- Never divide by zero — all ATR ratios use `+ 1e-9` denominator guard
- All functions handle DataFrames shorter than required window — return NaN gracefully

**TESTER_AGENT:**
```python
import pandas as pd, numpy as np
from indicators.market_structure import compute_all

n = 200
df = pd.DataFrame({
    'open':   np.random.randn(n).cumsum() + 1.1000,
    'high':   np.random.randn(n).cumsum() + 1.1010,
    'low':    np.random.randn(n).cumsum() + 1.0990,
    'close':  np.random.randn(n).cumsum() + 1.1005,
    'volume': np.random.randint(1000, 50000, n).astype(float)
}, index=pd.date_range('2024-01-01', periods=n, freq='15min'))

result = compute_all(df)
required = ['ema_21','ema_50','ema_200','atr_14','rsi_14','adx_14',
            'stoch_k','stoch_d','bb_upper','bb_mid','bb_lower',
            'ema_stack','fvg_bull','fvg_bear','bos_bull','bos_bear',
            'sweep_bull','sweep_bear','ob_bull','ob_bear']
for col in required:
    assert col in result.columns, f"Missing column: {col}"
assert not result['atr_14'].iloc[20:].isna().any(), "ATR has NaN after warmup"
assert result['ema_stack'].isin([-2, 0, 2]).all(), "Invalid ema_stack values"
print("market_structure.py: ALL TESTS PASS")
```

---

## PHASE 3 — FEATURE ENGINE (`services/feature_engine.py`)

Single class that computes all feature vectors consumed by every ML model and the RL state builder. No ML logic here — pure feature math.

```python
class FeatureEngine:
    """
    Produces feature arrays for:
      - GRU-LSTM: get_sequence(df, length=30) → shape (30, 18)
      - LightGBM regime: get_regime_features(df) → shape (11,)
      - XGBoost quality: get_quality_features(signal, context) → shape (14,)
      - RL state: get_rl_state(bar, portfolio, signals, ml_preds) → shape (42,)
    All outputs: numpy float32. No NaN. No lookahead.
    """

    SEQUENCE_FEATURES = [
        'log_return', 'high_low_range', 'close_vs_open', 'atr_normalized',
        'rsi_14', 'ema21_dist', 'ema50_dist', 'bb_position', 'volume_ratio',
        'is_asian', 'is_london', 'is_ny',
        'bos_bull_flag', 'bos_bear_flag', 'fvg_bull_open', 'fvg_bear_open',
        'h4_ema21_ema50_diff', 'adx_h1'
    ]  # 18 features

    REGIME_FEATURES = [
        'adx_14_h1', 'adx_14_h4', 'ema_stack_score',
        'atr_ratio', 'bb_width_pct', 'realized_vol_20', 'session_code',
        'swing_hh_hl_count', 'liquidity_sweep_24h', 'dxy_1h_return', 'gold_atr_ratio'
    ]  # 11 features

    QUALITY_FEATURES = [
        'strategy_id', 'signal_direction', 'rr_ratio',
        'p_bull_gru', 'p_bear_gru', 'regime_class', 'sentiment_score',
        'adx_at_signal', 'atr_ratio_at_signal', 'volume_ratio',
        'spread_at_signal', 'session_at_signal', 'news_in_30min',
        'strategy_win_rate_20'
    ]  # 14 features

    # RL STATE VECTOR DIMENSIONS:
    # [0-5]   ML predictions (p_bull, p_bear, entry_depth, regime_id, sentiment, quality)
    # [6-13]  Market structure (adx, ema_stack, atr_ratio, bb_width, bos_bull, bos_bear, fvg_bull, fvg_bear)
    # [14-18] Session context (asian, london, ny, dead, news_proximity)
    # [19-23] Strategy signals present (S1–S5 booleans)
    # [24-29] Portfolio state (open_pos, drawdown, daily_pnl, trades_today, last_result, equity_norm)
    # [30-33] Instrument one-hot (EURUSD, GBPUSD, USDJPY, XAUUSD)
    # [34-41] ATR history ratios (8 lags: 1,4,8,24,48,96,168,336 bars)
    # TOTAL: 42 dimensions
```

**CODER_AGENT must ensure:**
- `get_sequence()` pads with zeros at start if `len(df) < length` — never raises IndexError
- `get_rl_state()` clamps all values to `[-10, 10]` before returning — prevents RL divergence
- Missing columns in `df` return 0.0 with a structured log warning — never raise KeyError
- HTF features (`adx_h1`, `h4_ema21_ema50_diff`) accept an optional `df_htf` param; if None, compute from `df`
- `strategy_win_rate_20` reads from `TradeJournal.get_rolling_stats(trader_id, n=20)` — injected via constructor

**TESTER_AGENT:**
```python
from services.feature_engine import FeatureEngine
import numpy as np, pandas as pd

fe = FeatureEngine()
n = 100
df = make_dummy_ohlcv(n)  # helper: random valid OHLCV with datetime index

seq = fe.get_sequence(df, length=30)
assert seq.shape == (30, 18), f"Sequence shape: {seq.shape}"
assert seq.dtype == np.float32
assert not np.any(np.isnan(seq)), "NaN in sequence"
assert not np.any(np.isinf(seq)), "Inf in sequence"

regime_feat = fe.get_regime_features(df)
assert regime_feat.shape == (11,)
assert not np.any(np.isnan(regime_feat))

state = fe.get_rl_state(df.iloc[-1], portfolio={}, signals={}, ml_preds={})
assert state.shape == (42,)
assert state.dtype == np.float32
assert np.all(state >= -10) and np.all(state <= 10), "RL state not clamped"

print("FeatureEngine: ALL TESTS PASS")
```

---

## PHASE 4 — ML MODELS (one file at a time, in dependency order)

### 4A — `models/base_model.py`
Hot-reload base class. Must implement:
- `reload_if_updated()`: checks `os.path.getmtime(weight_path)` every 5 min; reloads on change
- `save(path)`, `load(path)` abstract methods
- `is_trained` property: `True` if weight file exists and is loadable
- All subclasses call `super().__init__()` which sets up the reload timer

### 4B — `models/gru_lstm_predictor.py`

**Complete implementation — no stubs:**

```python
class GRULSTMPredictor(BaseModel):
    """
    GRU(20) → Dropout(0.25) → LSTM(128) → Dropout(0.25) → Dense(64, relu, L2) → Dense(3, sigmoid)
    Outputs: p_bull, p_bear, entry_depth (all sigmoid 0-1)
    Fallback if untrained: {'p_bull': 0.5, 'p_bear': 0.5, 'entry_depth': 0.3}
    """
    SEQUENCE_LENGTH = 30
    N_FEATURES = 18
    WEIGHT_DIR = "weights/gru_lstm/"

    def build_model(self) -> None:
        """Build architecture. Called once. Saved as TF SavedModel."""

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Returns dict with p_bull, p_bear, entry_depth.
        If model not trained: return neutral fallback dict.
        Never raises — always returns valid dict.
        """

    def train(self, df: pd.DataFrame, labels: pd.DataFrame,
              epochs: int = 50, batch_size: int = 64,
              validation_split: float = 0.2) -> dict:
        """
        labels columns: direction_up (binary), entry_depth (float 0-1)
        Returns training history dict.
        Enforces strict temporal split: last 20% of data is validation (no shuffle).
        Early stopping: patience=10, monitor=val_loss.
        """

    def create_labels(self, df: pd.DataFrame,
                      horizon_bars: int = 4,
                      atr_threshold: float = 0.5) -> pd.DataFrame:
        """
        direction_up = 1 if close[t+horizon] > close[t] + atr[t]*atr_threshold, else 0
        entry_depth = ATR-normalized distance to ema21 at signal bar
        CRITICAL: labels are forward-looking — must be excluded from training features.
        """
```

### 4C — `models/regime_classifier.py`

```python
class RegimeClassifier(BaseModel):
    """
    LightGBM 4-class regime classifier.
    Classes: 0=TRENDING_UP, 1=TRENDING_DOWN, 2=RANGING, 3=VOLATILE
    3-bar hysteresis: regime must persist 3 bars before switching.
    Fallback: {'regime': 'RANGING', 'regime_id': 2, 'proba': [0.25]*4}
    """
    CLASSES = ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'VOLATILE']
    N_FEATURES = 11
    WEIGHT_PATH = "weights/regime_classifier.pkl"

    LGB_PARAMS = {
        'n_estimators': 500, 'learning_rate': 0.05,
        'num_leaves': 31, 'max_depth': 6,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'class_weight': 'balanced', 'random_state': 42
    }

    def predict(self, df: pd.DataFrame) -> dict:
        """Returns {'regime': str, 'regime_id': int, 'proba': list[float]}"""

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Rule-based labeling for training:
        ADX > 25 AND ema_stack == 2  → TRENDING_UP
        ADX > 25 AND ema_stack == -2 → TRENDING_DOWN
        ATR_ratio > 1.5              → VOLATILE
        else                         → RANGING
        """
```

### 4D — `models/quality_scorer.py`

```python
class QualityScorer(BaseModel):
    """
    XGBoost binary classifier.
    Output: float quality_score in [0, 1]
    Threshold: > 0.55 default (per-strategy overrides in traders)
    Fallback: 0.5
    """
    N_FEATURES = 14
    WEIGHT_PATH = "weights/quality_scorer.pkl"

    XGB_PARAMS = {
        'n_estimators': 1000, 'learning_rate': 0.05,
        'max_depth': 4, 'subsample': 0.7,
        'colsample_bytree': 0.7, 'gamma': 0.1,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'random_state': 42, 'eval_metric': 'logloss'
    }

    def predict(self, features: dict) -> float:
        """features keys = FeatureEngine.QUALITY_FEATURES (exact order enforced internally)"""

    def create_labels(self, journal_path: str) -> pd.DataFrame:
        """
        Read trade_journal_detailed.jsonl.
        Label 1 = trade reached TP1 (exit_reason contains 'tp')
        Label 0 = trade hit SL (exit_reason == 'sl')
        """
```

### 4E — `models/sentiment_model.py`

```python
class SentimentModel(BaseModel):
    """
    Primary: FinBERT (ProsusAI/finbert) via HuggingFace transformers
    Fallback: VADER + finance keyword lexicon (if torch/transformers unavailable)
    Gold: sentiment score is INVERTED (USD bullish = Gold bearish)
    Cache: LRU 50 headlines, TTL 30 minutes
    """

    def analyze(self, text: str, instrument: str = 'USD') -> dict:
        """
        Returns: {'score': float [-1,1], 'label': str, 'confidence': float}
        """

    def reload_if_updated(self):
        """FinBERT is pre-trained. Override to no-op."""
        pass
```

### 4F — `models/rl_agent.py`

```python
class RLAgent(BaseModel):
    """
    PPO strategy selector via Stable-Baselines3.
    State: 42-dim vector. Actions: 6 (0=NoTrade, 1-5=Strategies).
    Fallback: session-heuristic when model not trained.
    """

    N_STATE = 42
    N_ACTIONS = 6
    MODEL_DIR = "weights/rl_ppo/"

    # Interface preserved from old system
    def record_outcome(self, trade_result: dict) -> None:
        """
        Called by TradeJournal after every trade close.
        Adds to experience buffer. Triggers mini-update when buffer >= 64.
        trade_result must contain: pnl, rr_ratio, confidence, rl_action,
                                   state_at_entry (list[float], len=42)
        """

    def select_action(self, state: np.ndarray,
                      available_signals: dict) -> tuple[int, float]:
        """
        Returns (action_id, confidence).
        HARD RULE: if selected action's strategy not in available_signals → return (0, 0.0)
        """

    def _compute_reward(self, trade_result: dict) -> float:
        """
        Multi-component reward:
          pnl_reward      = clip(r_multiple, -3, 4) × 1.0
          sharpe_bonus    = clip(rolling_sharpe_20 × 0.3, -0.5, 0.5)
          dd_penalty      = -2.0 × max(0, drawdown - 0.05)
          overtrade_pen   = -0.3 × max(0, trades_today - 4)
          session_bonus   = +0.15 if correct session strategy used AND profitable
          inaction_pen    = -0.05 if valid London setup skipped (action==0)
        """

    def _heuristic_fallback(self, available_signals: dict,
                             session: str) -> tuple[int, float]:
        """
        When PPO not trained:
        London session + S3 available → (3, 0.6)
        NY session + S1 available     → (1, 0.6)
        NY session + S2 available     → (2, 0.55)
        Asian session + S4 available  → (4, 0.55)
        News event + S5 available     → (5, 0.65)
        Else                          → (0, 0.0)
        """

    def retrain_from_journal(self, journal_path: str,
                              n_epochs: int = 10) -> None:
        """For retrain_incremental.py: reconstruct episodes from journal, run PPO update."""
```

---

## PHASE 5 — BASE TRADER (`traders/base_trader.py`)

Complete new base class. Capital isolation + cooldown + session awareness + ML gate + RL gate.

```python
class BaseTrader:
    """
    New BaseTrader for the Forex + Gold system.
    All 5 traders inherit this. Do not bypass any guard.
    """

    # Subclass must define these
    TRADER_ID: str = ""
    NAME: str = ""
    SESSION_START_UTC: int = 0     # inclusive hour
    SESSION_END_UTC: int = 24      # exclusive hour
    HARD_CLOSE_HOUR: int = 23
    HARD_CLOSE_MINUTE: int = 59
    MAX_TRADES_PER_SESSION: int = 2
    COOLDOWN_MINUTES: int = 10
    ML_QUALITY_THRESHOLD: float = 0.55
    ML_DIRECTION_THRESHOLD: float = 0.58
    EXCLUDED_SYMBOLS: list = []

    def __init__(self, settings, redis_client, state_manager,
                 feature_engine, ml_models: dict):
        self.settings = settings
        self.redis = redis_client
        self.state = state_manager
        self.fe = feature_engine
        self.ml = ml_models   # keys: 'gru_lstm', 'regime', 'quality', 'sentiment', 'rl'
        self._cooldown_until: dict = {}        # {symbol: datetime}
        self._trades_today: dict = {}          # {symbol: int}
        self._session_traded: set = set()      # {symbol} — reset at session end
        self._fvg_registry: dict = {}          # For trader_2 persistence
        self._asian_range_cache: dict = {}     # For trader_3 persistence

    def analyze_market(self, symbol: str, df: pd.DataFrame,
                       df_htf: pd.DataFrame,
                       ml_predictions: dict) -> dict | None:
        """
        Called by SignalPipeline on every bar.
        Subclasses implement _compute_signal(); do not override this method.
        Returns signal dict or None.
        """
        # Guard 0: Symbol exclusion
        if symbol in self.EXCLUDED_SYMBOLS:
            return None

        # Guard 1: Session window
        if not self._in_session():
            return None

        # Guard 2: Dead zone
        if self._in_dead_zone():
            return None

        # Guard 3: Cooldown
        if self._in_cooldown(symbol):
            return None

        # Guard 4: Max trades per session
        if self._trades_today.get(symbol, 0) >= self.MAX_TRADES_PER_SESSION:
            return None

        # Guard 5: Daily loss and drawdown circuit breakers
        if self._circuit_breaker_active():
            return None

        # Guard 6: Spread
        spread = ml_predictions.get('spread_pips', 0)
        if symbol == 'XAUUSD' and spread > 50:
            return None
        elif spread > 3 and symbol != 'XAUUSD':
            return None

        # Core: delegate to subclass
        signal = self._compute_signal(symbol, df, df_htf, ml_predictions)

        if signal is None:
            return None

        # Guard 7: ML quality gate
        if self.settings.ML_ENABLED:
            quality = ml_predictions.get('quality_score', 0.5)
            if quality < self.ML_QUALITY_THRESHOLD:
                return None

            p = ml_predictions.get('p_bull') if signal['side'] == 'buy' \
                else ml_predictions.get('p_bear')
            if p is not None and p < self.ML_DIRECTION_THRESHOLD:
                return None

        # Guard 8: RL action gate
        if self.settings.ML_ENABLED:
            rl_agent = self.ml.get('rl')
            if rl_agent:
                state_vec = self.fe.get_rl_state(
                    df.iloc[-1], self._get_portfolio_state(),
                    {self.TRADER_ID: signal}, ml_predictions
                )
                action, rl_conf = rl_agent.select_action(
                    state_vec, {self.TRADER_ID: True}
                )
                expected_action = int(self.TRADER_ID.split('_')[1])
                if action != expected_action:
                    return None  # RL chose a different strategy
                signal['signal_metadata']['rl_action'] = action
                signal['signal_metadata']['rl_confidence'] = rl_conf
                signal['state_at_entry'] = state_vec.tolist()

        # Compute R:R and confidence-driven R:R scaling
        rr = (signal['take_profit'] - signal['entry']) / (signal['entry'] - signal['stop_loss']) \
             if signal['side'] == 'buy' else \
             (signal['entry'] - signal['take_profit']) / (signal['stop_loss'] - signal['entry'])
        if rr < 1.0:
            return None  # Hard R:R minimum

        signal['rr_ratio'] = round(rr, 2)
        signal['signal_metadata']['rr_ratio'] = signal['rr_ratio']

        # Compute position size
        signal['size'] = self._compute_position_size(
            symbol, signal['entry'], signal['stop_loss']
        )

        # Update state
        self._set_cooldown(symbol)
        self._trades_today[symbol] = self._trades_today.get(symbol, 0) + 1

        # Publish Redis state
        self._update_redis_state(symbol)

        return signal

    def _compute_signal(self, symbol: str, df: pd.DataFrame,
                         df_htf: pd.DataFrame, ml_predictions: dict) -> dict | None:
        """Subclasses implement this. Return raw signal dict or None."""
        raise NotImplementedError

    def _compute_position_size(self, symbol: str,
                                entry: float, stop_loss: float) -> float:
        """1% risk, scaled by ML confidence if enabled."""
        risk_amount = self.settings.CAPITAL_PER_TRADER * self.settings.RISK_PER_TRADE
        pip_distance = abs(entry - stop_loss)
        if pip_distance < 1e-9:
            return 0.0
        size = risk_amount / pip_distance
        # ML confidence scaling: 0.5× to 1.5× (only if ML_ENABLED)
        # Applied in subclasses that have quality_score
        return round(size, 2)
```

---

## PHASE 6 — FIVE TRADERS (complete implementations)

Each trader file is standalone and complete. No stubs. Every rule from the strategy documents is implemented.

### 6A — `traders/trader_1_ny_ema.py` — NY Session EMA Trend Pullback

```
Inherits: BaseTrader
TRADER_ID = "trader_1"
SESSION: 13:00–17:00 UTC (hard close 17:00)
INSTRUMENTS: EURUSD, GBPUSD, USDJPY, XAUUSD
MAX_TRADES_PER_SESSION = 2
COOLDOWN_MINUTES = 10
ML_QUALITY_THRESHOLD = 0.58

SIGNAL LOGIC:
  Step 1: EMA stack on H1: ema_21 > ema_50 > ema_200 (long) or reverse (short)
  Step 2: ADX_H1 > 22
  Step 3: London direction alignment:
          london_open = close at 07:00 UTC bar
          london_close = close at 12:00 UTC bar
          london_direction = "up" if london_close > london_open else "down"
          NY entry must match london_direction
  Step 4: Pullback on 15M:
          abs(close - ema_21) < atr_14 * 0.5 AND
          close > ema_21 (long) AND
          rsi_14 between 40 and 55
  Step 5: Signal candle: bullish close (close > open, body > 0.3×range)
  Step 6: Forex only: skip if close within 15 pips of round number
  Step 7: Gold only: atr_ratio > 0.7 (not dead session)
  
  SL: ema_50_H1 - atr_H1 * 0.3 (long) / + (short)
  TP1: entry + atr_H1 * 1.5 (50% close)
  TP2: entry + atr_H1 * 3.0 (50% close)
  
  confidence base: 0.60
  + 0.05 if fvg_bull near entry (long)
  + 0.05 if bos_bull on H4
  + 0.05 if sweep_bull in last 5 bars
  max confidence: 0.85
  
  signal_metadata fields:
    strategy: "ny_ema_pullback"
    session: "ny"
    ema_stack: int
    adx_h1: float
    london_direction: str
    pullback_pct: float
```

### 6B — `traders/trader_2_fvg_bos.py` — Structure Break + FVG Continuation

```
Inherits: BaseTrader
TRADER_ID = "trader_2"
SESSION: London (07:00–12:00) + NY (13:00–18:00) — BLOCKED 12:00–13:00 dead zone
INSTRUMENTS: All 4 (EURUSD, GBPUSD, USDJPY, XAUUSD)
MAX_TRADES_PER_SESSION = 2
COOLDOWN_MINUTES = 15
ML_QUALITY_THRESHOLD = 0.60
ML_DIRECTION_THRESHOLD = 0.62

INSTANCE STATE (persists across bars):
  self._bias: dict = {}            # {symbol: 'bullish'|'bearish'}
  self._fvg_registry: dict = {}   # {symbol: list of active FVG dicts}

SIGNAL LOGIC:
  Step 1: HTF BOS on H4:
          Use detect_break_of_structure(df_h4, swing_n=5)
          bos_bull → set self._bias[symbol] = 'bullish'
          bos_bear → set self._bias[symbol] = 'bearish'
          If no bias set yet → return None

  Step 2: FVG detection on 15M (after bias bar only):
          Use detect_fair_value_gaps(df_15m, min_atr_ratio=0.3)
          For XAUUSD: min_atr_ratio = 0.5
          Append to self._fvg_registry[symbol]
          Age FVGs: expire after 20 bars

  Step 3: Entry check — scan active FVGs:
          For each active FVG aligned with bias:
            in_zone = (fvg_bottom <= current_price <= fvg_mid)
            entry_candle_confirms = (close > open for bull, close < open for bear)
          
  Step 4: R:R pre-check:
          sl = fvg_bottom - atr_14 * 0.1 (bull)
          tp1 = nearest swing high (from detect_break_of_structure last_swing_high)
          rr = (tp1 - entry) / (entry - sl)
          if rr < 2.0 → skip this FVG

  Step 5: Forex: skip if within 20 pips of daily pivot
          Gold: spread must be < $0.50 equivalent; min FVG = atr * 0.5

  Step 6: News check: no entry if high-impact news within 15 min

  SL: fvg_bottom - atr_14 * 0.1 (bull) / fvg_top + (bear)
  TP1: last_swing_high/low (50%)
  TP2: BOS origin point (50%)

  Invalidation: if price passes through full FVG against bias → remove FVG from registry

  confidence base: 0.65 (higher than other strategies — R:R driven selection)
  + 0.05 if ob_bull near FVG entry
  max confidence: 0.90
```

### 6C — `traders/trader_3_london_bo.py` — London Breakout + Liquidity Sweep

```
Inherits: BaseTrader
TRADER_ID = "trader_3"
SESSION: 07:00–10:00 UTC (hard close 12:00 for all London positions)
INSTRUMENTS: EURUSD, GBPUSD, XAUUSD (NOT USDJPY — JPY has own Asian range dynamics)
MAX_TRADES_PER_SESSION = 1 per symbol
COOLDOWN_MINUTES = 5
ML_QUALITY_THRESHOLD = 0.60
ML_DIRECTION_THRESHOLD = 0.60

INSTANCE STATE:
  self._asian_range_cache: dict = {}  # {symbol: {date: {high, low, range, valid}}}

ASIAN RANGE COMPUTATION (called at first bar >= 07:00 each day):
  asia_bars = df bars where time is 00:00–07:00 UTC on current date
  asian_high = max(high) of asia_bars
  asian_low = min(low) of asia_bars
  asian_range = asian_high - asian_low
  Valid (EURUSD/GBPUSD): 0.0015 <= range <= 0.0080 (15–80 pips)
  Valid (XAUUSD):        1.00 <= range <= 5.00 (100–500 "pips")
  Cache result: self._asian_range_cache[symbol][date]

SIGNAL LOGIC (07:00–10:00 UTC bars):
  Step 1: Load cached asian range — if missing or invalid → return None
  Step 2: Sweep detection on current 15M bar:
    BULLISH SWEEP (real move UP):
      bar.low < asian_low - buffer AND
      bar.close > asian_low AND
      wick_below = asian_low - bar.low > 0 AND
      body = abs(bar.close - bar.open) > 0 AND
      wick_below / body > 1.5
      buffer: 0.0005 (EURUSD/GBPUSD) / 0.50 (XAUUSD)

    BEARISH SWEEP (real move DOWN):
      bar.high > asian_high + buffer AND
      bar.close < asian_high AND
      wick_above / body > 1.5

  Step 3: Volume: bar.volume > vol_sma_20 * 1.3
  Step 4: ML regime: NOT 'RANGING'
  Step 5: News: no high-impact news within 30 min (from news_service)
  Step 6: One trade per session per symbol gate

  SL (BULL): bar.low - buffer * 5
  SL (BEAR): bar.high + buffer * 5
  TP1: entry +/- asian_range (50%)
  TP2: entry +/- asian_range * 2 (50%) [Gold: 1.5× / 3×]

  Hard close: generate exit signal for all open London positions at 12:00 UTC bar

  confidence base: 0.68 (clean sweep is high-confidence)
  + 0.07 if bos_bull confirms direction on H1 after sweep
  max confidence: 0.90
```

### 6D — `traders/trader_4_news_momentum.py` — High-Impact News Momentum

```
Inherits: BaseTrader
TRADER_ID = "trader_4"
SESSION: Any — activated by news calendar events
INSTRUMENTS: EURUSD, GBPUSD, XAUUSD (Gold = priority, highest moves)
MAX_TRADES_PER_SESSION = 1
COOLDOWN_MINUTES = 30
ML_QUALITY_THRESHOLD = 0.60
SENTIMENT_THRESHOLD = 0.65  # FinBERT hard gate (not optional)

INSTANCE STATE:
  self._pre_news_zones: dict = {}   # {symbol: {news_time, high, low, mid, range}}
  self._news_active_until: dict = {}  # {symbol: datetime — hard close time}

PRE-NEWS ZONE (computed 30 min before known news event):
  window = bars from (news_time - 30min) to news_time
  news_high = max(high) of window
  news_low = min(low) of window
  news_range = news_high - news_low
  Valid: news_range < atr_14 * 2.0
  Cache in self._pre_news_zones[symbol]

SIGNAL LOGIC (2–15 min after news release):
  Step 1: news_service.get_active_events() — get events within last 15 min
  Step 2: Load pre-news zone for symbol
  Step 3: Spread gate: spread < 3 pips (Forex) / < $0.50 (Gold)
  Step 4: Sentiment hard gate: |sentiment_score| > 0.65 in signal direction
          (FinBERT required — if sentiment model unavailable → SKIP THIS STRATEGY)
  Step 5: Structural breakout:
          Long: bar.close > news_high + spread * 1.5
          Short: bar.close < news_low - spread * 1.5
  Step 6: Volume: bar.volume > vol_sma_20 * 2.0
  Step 7: Trend alignment:
          H4 trend aligns → size_multiplier = 1.25
          H4 trend opposes → size_multiplier = 0.75

  SL: news_mid (midpoint of pre-news zone)
  TP1: entry +/- news_range * 1.5 (50%)
  TP2: entry +/- news_range * 3.0 (50%) [Gold: 4×]

  Hard close: 60 minutes after news event time
  Schedule self._news_active_until[symbol] = news_time + 60min
  At each subsequent bar: check if past hard close → emit exit signal

  confidence base: 0.70 (news momentum = directional conviction)
  max confidence: 0.90

  signal_metadata extra fields:
    event_name: str
    news_surprise_direction: "beat" | "miss" | "neutral"
    sentiment_score: float
    pre_news_range: float
```

### 6E — `traders/trader_5_asian_mr.py` — Asian Range Mean Reversion

```
Inherits: BaseTrader
TRADER_ID = "trader_5"
SESSION: 02:00–06:30 UTC (HARD CLOSE ALL POSITIONS at 06:45 UTC)
INSTRUMENTS: USDJPY, EURJPY, AUDJPY, EURUSD, AUDUSD (XAUUSD EXCLUDED)
EXCLUDED_SYMBOLS = ['XAUUSD']
MAX_TRADES_PER_SESSION = 2
COOLDOWN_MINUTES = 30
ML_QUALITY_THRESHOLD = 0.55  # lower threshold — counter-directional strategy
ML_DIRECTION_THRESHOLD = 0.55

INSTANCE STATE:
  self._range_cache: dict = {}   # {symbol: {date: {high, low, mid, range, valid}}}

RANGE COMPUTATION (first 2 hours: 00:00–02:00 UTC):
  build_bars = bars where time is 00:00–02:00 UTC on current date
  asian_high = max(high), asian_low = min(low)
  Valid (USDJPY): 0.15 <= range <= 0.70 (15–70 pips in yen)
  Valid (others): 0.0010 <= range <= 0.0060 (10–60 pips)

SIGNAL LOGIC (02:00–06:30 UTC):
  Step 1: Load range cache — if invalid → return None
  Step 2: ADX gate: adx_14 (H1) < 25 (MANDATORY — ranging regime)
  Step 3: ML regime gate: regime == 'RANGING' (MANDATORY hard gate)
  Step 4: News block: no trade if high-impact news within 90 min
  Step 5: Sentiment gate: abs(sentiment_score) < 0.5
          (active news sentiment invalidates range assumptions)
  Step 6: Range integrity: price NOT beyond range by > 8 pips already
  Step 7: Dual oscillator (both required within 3 bars):
    LONG (at asian_low zone):
      price within 5 pips of asian_low AND
      rsi_14 < 35 AND rsi just crossed above 35 AND
      stoch_k < 30 AND stoch_k crossed above stoch_d
    SHORT (at asian_high zone):
      price within 5 pips of asian_high AND
      rsi_14 > 65 AND rsi just crossed below 65 AND
      stoch_k > 70 AND stoch_k crossed below stoch_d

  SL (LONG):  asian_low - 8 pips (or equivalent in symbol units)
  SL (SHORT): asian_high + 8 pips
  TP1: asian_mid (50% close — high probability target)
  TP2: opposite extreme (50%)
  Time stop: if TP1 not hit within 3 hours → close at market

  HARD CLOSE at 06:45 UTC: generate forced exit signals for all open positions

  confidence base: 0.60
  + 0.05 if ob_bull/bear near entry
  max confidence: 0.80
```

---

## PHASE 7 — SERVICES

### `services/session_manager.py`
```python
class SessionManager:
    """
    UTC-based session detection. Single source of truth for all session logic.
    No trader implements its own time checks — all delegate here.
    """
    def get_current_session(self, dt: datetime = None) -> str:
        """Returns: 'ASIAN' | 'LONDON' | 'NY' | 'DEAD' | 'INACTIVE'"""

    def is_hard_close_time(self, trader_id: str, dt: datetime = None) -> bool:
        """Returns True if trader's hard close time has passed this session."""

    def get_session_open_price(self, df: pd.DataFrame, session: str) -> float | None:
        """Returns first close of given session on same date as df.index[-1]."""

    def should_trade(self, trader_id: str, dt: datetime = None) -> bool:
        """Combined session + dead zone check for a specific trader."""
```

### `services/news_service.py`
```python
class NewsService:
    """
    Unified news feed for all traders.
    Sources: news_monitor (Redis pub/sub) + ForexFactory calendar polling.
    """
    def get_upcoming_events(self, within_minutes: int = 30,
                             currencies: list = None) -> list[dict]:
        """Returns high-impact events within window."""

    def get_recent_events(self, within_minutes: int = 15,
                           currencies: list = None) -> list[dict]:
        """Returns recently released high-impact events."""

    def get_pre_news_block(self, symbol: str,
                            df: pd.DataFrame,
                            news_time: datetime) -> dict | None:
        """Compute pre-news consolidation zone for a symbol."""

    def is_blocked(self, symbol: str, block_minutes: int = 30) -> bool:
        """True if high-impact news for symbol's currency within block_minutes."""
```

### `services/signal_pipeline.py` (replaces `signal_engine.py` + `strategy_manager.py`)

```python
class SignalPipeline:
    """
    Orchestrates: ML inference → strategy scan → RL selection → execution gate.
    Called by main.py on every MARKET_DATA event.
    """

    def __init__(self, traders, ml_models, feature_engine,
                 session_manager, news_service, settings):
        ...

    async def process_bar(self, symbol: str, df: pd.DataFrame,
                           df_htf: dict) -> list[dict]:
        """
        Returns list of approved signals (usually 0 or 1).
        Full pipeline per bar:
          1. ML inference (if ML_ENABLED)
          2. Run all session-appropriate traders
          3. Collect valid rule-based signals
          4. RL action selection (if ML_ENABLED)
          5. Confidence gate
          6. Return approved signals
        """

    def _run_ml_inference(self, symbol: str, df: pd.DataFrame,
                           df_htf: pd.DataFrame) -> dict:
        """
        Returns ml_predictions dict:
          p_bull, p_bear, entry_depth (GRU-LSTM)
          regime, regime_id (LightGBM)
          quality_score (XGBoost — updated after strategy signals collected)
          sentiment_score (FinBERT)
          spread_pips (from data_fetcher)
        """

    def _get_ml_score_for_signal(self, signal: dict,
                                  ml_base: dict, bar_data) -> float:
        """Compute per-signal quality score using XGBoost."""

    def _compute_ensemble_score(self, ict_score: float,
                                  ml_preds: dict,
                                  signal_direction: str) -> float:
        """
        New ensemble formula:
          gru_score = (p_bull - 0.5) * 2.0  if long  (maps [0,1]→[-1,1])
                    = (p_bear - 0.5) * 2.0  if short
          ml_score = gru_score * 0.5 + quality_score * 0.5
          sentiment_bonus = sign_match * abs(sentiment) * 0.1 else -0.05
          regime_mult = {TRENDING_UP: 1.2 long, TRENDING_DOWN: 1.2 short,
                         RANGING: 0.9, VOLATILE: 0.85}.get(regime, 1.0)
          score = (ict_score * 0.5 + ml_score * 0.5 + sentiment_bonus) * regime_mult
        """
```

### `services/risk_engine.py` (complete replacement)

```python
class RiskEngine:
    """
    System-wide risk controls. All checks run before order submission.
    """
    def check_pre_trade(self, signal: dict, portfolio_state: dict) -> tuple[bool, str]:
        """
        Returns (allowed, reason_if_blocked).
        Checks:
          - daily_loss_pct < 0.02 (2% daily circuit breaker)
          - drawdown_pct < 0.08 (8% max drawdown)
          - open_positions < 2 (max concurrent)
          - one position per symbol (no stacking)
          - spread within tolerance
        """

    def compute_position_size(self, symbol: str, entry: float,
                               stop_loss: float, equity: float,
                               quality_score: float = 0.5,
                               ml_enabled: bool = False) -> float:
        """
        Base: equity * 0.01 / pip_distance
        ML scaling: base * (0.75 + quality_score * 0.5) if ml_enabled
        Clamped: [0.5× base, 1.5× base]
        """
```

### `services/trade_journal.py` (complete replacement)

All Contract 4 fields. Additional fields for ML/RL:

```python
class TradeJournal:
    def log_trade(self, trade: dict) -> None:
        """Write to both CSV (clean) and JSONL (detailed)."""

    def get_rolling_stats(self, trader_id: str, n: int = 20) -> dict:
        """Returns {'win_rate': float, 'profit_factor': float, 'avg_rr': float}
           from last n closed trades for this trader. Used by FeatureEngine."""

    def get_rl_episodes(self) -> list[dict]:
        """Returns all closed trades with state_at_entry for RL retraining."""
```

---

## PHASE 8 — MAIN ENGINE (`main.py`)

Complete replacement. Orchestrates all components.

```python
class ProductionTradingEngine:
    """
    New engine. Same external interface as old engine:
    - Health server on port 8000
    - Redis event publisher/subscriber
    - Honour ML_ENABLED and PAPER_TRADING flags
    """

    def __init__(self):
        self.settings = settings
        self._init_redis()
        self._init_ml_models()        # skip if ML_ENABLED=false
        self._init_feature_engine()
        self._init_session_manager()
        self._init_news_service()
        self._init_traders()          # 5 traders
        self._init_signal_pipeline()
        self._init_risk_engine()
        self._init_trade_journal()
        self._init_monitors()
        self._init_health_server()

    def _init_traders(self):
        """Register all 5 traders with shared services."""
        trader_kwargs = dict(
            settings=self.settings,
            redis_client=self.redis,
            state_manager=self.state_mgr,
            feature_engine=self.feature_engine,
            ml_models=self.ml_models
        )
        self.traders = [
            Trader1NYEMA(**trader_kwargs),          # trader_id = "trader_1"
            Trader2FVGBos(**trader_kwargs),          # trader_id = "trader_2"
            Trader3LondonBO(**trader_kwargs),        # trader_id = "trader_3"
            Trader4NewsMomentum(**trader_kwargs),    # trader_id = "trader_4"
            Trader5AsianMR(**trader_kwargs),         # trader_id = "trader_5"
        ]

    async def on_market_data(self, event: dict):
        """
        Handler for MARKET_DATA Redis event.
        1. Update OHLCV store
        2. Run signal pipeline
        3. For each approved signal: risk check → size → execute → journal
        4. Update Redis state keys (Contract 2)
        """

    def run(self):
        """Main loop. Subscribes to MARKET_DATA. Runs health server thread."""
```

---

## PHASE 9 — SCRIPTS (complete rewrites)

### `scripts/run_backtest.py`

Rewrite to support all 5 new traders. Must:
- Accept trader IDs as args: `python run_backtest.py 1 2 3 4 5`
- Use `training_data/` CSVs as data source
- Output to `backtest_results/backtest_YYYYMMDD_HHMMSS.json` (Contract 5 format)
- Run each trader through its session window only (session-aware backtest)
- Report: trades, win_rate, profit_factor, max_drawdown, sharpe, total_return

### `scripts/retrain_incremental.py`

Rewrite to support all new models:

```bash
python retrain_incremental.py              # retrain all models
python retrain_incremental.py --model gru  # GRU-LSTM only
python retrain_incremental.py --model regime  # regime classifier
python retrain_incremental.py --model quality # XGBoost quality scorer
python retrain_incremental.py --model rl      # PPO retrain from journal
python retrain_incremental.py --dry-run   # validate without saving
```

Model-specific steps:
- GRU-LSTM: load 24 months of 15M OHLCV; create labels; train with temporal split; save SavedModel
- LightGBM regime: load H1 data; rule-label; retrain; validate accuracy > 65%
- XGBoost quality: load `trade_journal_detailed.jsonl`; create TP1/SL labels; retrain
- PPO RL: call `rl_agent.retrain_from_journal(journal_path)`
- FinBERT: no retrain (log skip message)

### `scripts/retrain_scheduler.py`

Minimal change: update model list from old 6 models to new 5 models. Same Sunday 02:00 UTC schedule.

---

## PHASE 10 — DOCKER UPDATE

The `docker-compose.yml` is **not touched**. But the engine's `Dockerfile` may need updating for new dependencies (TensorFlow, torch, transformers).

CODER_AGENT checks existing `trading-engine/Dockerfile` if present and updates only:
```dockerfile
# Add if not present — TensorFlow needs more space
FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=120 --retries=5 -r requirements.txt
```

For CPU-only (default Docker): use `tensorflow-cpu` instead of `tensorflow` in requirements.txt to reduce image size from ~2GB to ~600MB.

---

## PHASE 11 — BACKEND PATCH (one-line addition only)

The backend `routes/traders.py` has a hardcoded list of traders. Add trader_5:

```python
# In backend/src/routes/traders.py
# Find the static TRADERS list and add:
{
    "id": "trader_5",
    "trader_id": "trader_5",
    "name": "Asian Range Mean Reversion",
    "status": "active",
    "timeframe": "15M",
    "description": "Asian session mean reversion on JPY pairs"
}
```

This is the ONLY change to any backend file.

---

## PHASE 12 — END-TO-END VERIFICATION GATES

TESTER_AGENT runs all gates. Every gate must pass before EVALUATOR scores.

```bash
# ─── GATE 1: Syntax ───────────────────────────────────────────
python -m py_compile trading-engine/main.py
python -m py_compile trading-engine/indicators/market_structure.py
python -m py_compile trading-engine/services/feature_engine.py
python -m py_compile trading-engine/services/signal_pipeline.py
python -m py_compile trading-engine/services/session_manager.py
python -m py_compile trading-engine/services/news_service.py
python -m py_compile trading-engine/services/risk_engine.py
python -m py_compile trading-engine/services/trade_journal.py
python -m py_compile trading-engine/models/base_model.py
python -m py_compile trading-engine/models/gru_lstm_predictor.py
python -m py_compile trading-engine/models/regime_classifier.py
python -m py_compile trading-engine/models/quality_scorer.py
python -m py_compile trading-engine/models/sentiment_model.py
python -m py_compile trading-engine/models/rl_agent.py
python -m py_compile trading-engine/traders/base_trader.py
python -m py_compile trading-engine/traders/trader_1_ny_ema.py
python -m py_compile trading-engine/traders/trader_2_fvg_bos.py
python -m py_compile trading-engine/traders/trader_3_london_bo.py
python -m py_compile trading-engine/traders/trader_4_news_momentum.py
python -m py_compile trading-engine/traders/trader_5_asian_mr.py
echo "GATE 1 SYNTAX: PASS"

# ─── GATE 2: Import chain ─────────────────────────────────────
cd trading-engine && python -c "
from indicators.market_structure import compute_all
from services.feature_engine import FeatureEngine
from services.session_manager import SessionManager
from services.news_service import NewsService
from services.signal_pipeline import SignalPipeline
from services.risk_engine import RiskEngine
from services.trade_journal import TradeJournal
from models.base_model import BaseModel
from models.gru_lstm_predictor import GRULSTMPredictor
from models.regime_classifier import RegimeClassifier
from models.quality_scorer import QualityScorer
from models.sentiment_model import SentimentModel
from models.rl_agent import RLAgent
from traders.base_trader import BaseTrader
from traders.trader_1_ny_ema import Trader1NYEMA
from traders.trader_2_fvg_bos import Trader2FVGBos
from traders.trader_3_london_bo import Trader3LondonBO
from traders.trader_4_news_momentum import Trader4NewsMomentum
from traders.trader_5_asian_mr import Trader5AsianMR
print('GATE 2 IMPORTS: PASS')
"

# ─── GATE 3: Trader IDs match contracts ───────────────────────
cd trading-engine && python -c "
from traders.trader_1_ny_ema import Trader1NYEMA
from traders.trader_2_fvg_bos import Trader2FVGBos
from traders.trader_3_london_bo import Trader3LondonBO
from traders.trader_4_news_momentum import Trader4NewsMomentum
from traders.trader_5_asian_mr import Trader5AsianMR
assert Trader1NYEMA.TRADER_ID == 'trader_1'
assert Trader2FVGBos.TRADER_ID == 'trader_2'
assert Trader3LondonBO.TRADER_ID == 'trader_3'
assert Trader4NewsMomentum.TRADER_ID == 'trader_4'
assert Trader5AsianMR.TRADER_ID == 'trader_5'
assert 'XAUUSD' in Trader5AsianMR.EXCLUDED_SYMBOLS
print('GATE 3 TRADER IDs: PASS')
"

# ─── GATE 4: Indicator shapes ─────────────────────────────────
cd trading-engine && python -c "
import pandas as pd, numpy as np
from indicators.market_structure import compute_all

n = 200
df = pd.DataFrame({
    'open': 1.1000 + np.random.randn(n).cumsum()*0.001,
    'high': 1.1010 + np.random.randn(n).cumsum()*0.001,
    'low':  1.0990 + np.random.randn(n).cumsum()*0.001,
    'close':1.1005 + np.random.randn(n).cumsum()*0.001,
    'volume': np.random.randint(1000,50000,n).astype(float)
}, index=pd.date_range('2024-01-01', periods=n, freq='15min'))
df['high'] = df[['open','high','close']].max(axis=1) + 0.0001
df['low']  = df[['open','low','close']].min(axis=1)  - 0.0001

result = compute_all(df)
for col in ['ema_21','ema_50','ema_200','atr_14','rsi_14','adx_14',
            'stoch_k','stoch_d','bb_upper','bb_mid','bb_lower',
            'ema_stack','fvg_bull','fvg_bear','bos_bull','bos_bear',
            'sweep_bull','sweep_bear']:
    assert col in result.columns, f'Missing: {col}'
assert not result['atr_14'].iloc[20:].isna().any(), 'ATR has NaN after warmup'
print('GATE 4 INDICATORS: PASS')
"

# ─── GATE 5: Feature engine shapes and types ──────────────────
cd trading-engine && python -c "
import pandas as pd, numpy as np
from services.feature_engine import FeatureEngine
from indicators.market_structure import compute_all

n = 100
df = pd.DataFrame({
    'open': 1.1000 + np.random.randn(n).cumsum()*0.001,
    'high': 1.1010 + np.random.randn(n).cumsum()*0.001,
    'low':  1.0990 + np.random.randn(n).cumsum()*0.001,
    'close':1.1005 + np.random.randn(n).cumsum()*0.001,
    'volume': np.random.randint(1000,50000,n).astype(float)
}, index=pd.date_range('2024-01-01', periods=n, freq='15min'))
df = compute_all(df)

fe = FeatureEngine()
seq = fe.get_sequence(df)
assert seq.shape == (30, 18), f'Sequence shape: {seq.shape}'
assert seq.dtype == np.float32
assert not np.any(np.isnan(seq))
assert not np.any(np.isinf(seq))

reg = fe.get_regime_features(df)
assert reg.shape == (11,)
assert not np.any(np.isnan(reg))

state = fe.get_rl_state(df.iloc[-1], {}, {}, {})
assert state.shape == (42,)
assert state.dtype == np.float32
assert np.all(state >= -10) and np.all(state <= 10)
print('GATE 5 FEATURE ENGINE: PASS')
"

# ─── GATE 6: ML model fallbacks (no weights available) ────────
cd trading-engine && python -c "
from models.gru_lstm_predictor import GRULSTMPredictor
from models.regime_classifier import RegimeClassifier
from models.quality_scorer import QualityScorer
from models.sentiment_model import SentimentModel
from models.rl_agent import RLAgent
import numpy as np

gru = GRULSTMPredictor()
r = gru.predict(None)
assert r == {'p_bull':0.5,'p_bear':0.5,'entry_depth':0.3}, f'GRU fallback: {r}'

reg = RegimeClassifier()
r = reg.predict(None)
assert r['regime'] == 'RANGING'

qs = QualityScorer()
r = qs.predict({})
assert r == 0.5, f'QS fallback: {r}'

sa = SentimentModel()
r = sa.analyze('test headline', instrument='EURUSD')
assert -1.0 <= r['score'] <= 1.0
r_gold = sa.analyze('dollar surges', instrument='XAUUSD')
# Gold inverts USD sentiment

rl = RLAgent()
state = np.zeros(42, dtype=np.float32)
action, conf = rl.select_action(state, {'trader_1': True})
assert 0 <= action <= 5
rl.record_outcome({'pnl':50.0,'rr_ratio':2.0,'confidence':0.7,
                   'rl_action':1,'state_at_entry':[0.0]*42})
print('GATE 6 ML FALLBACKS: PASS')
"

# ─── GATE 7: RL record_outcome interface (TradeJournal contract) ─
cd trading-engine && python -c "
from models.rl_agent import RLAgent
import inspect
sig = inspect.signature(RLAgent.record_outcome)
params = list(sig.parameters.keys())
assert 'trade_result' in params, f'record_outcome missing trade_result: {params}'
print('GATE 7 RL INTERFACE: PASS')
"

# ─── GATE 8: Session exclusions ──────────────────────────────
cd trading-engine && python -c "
from traders.trader_5_asian_mr import Trader5AsianMR
assert 'XAUUSD' in Trader5AsianMR.EXCLUDED_SYMBOLS, 'Trader5 must exclude XAUUSD'
from traders.trader_3_london_bo import Trader3LondonBO
# USDJPY excluded from London BO (per strategy spec)
print('GATE 8 SESSION EXCLUSIONS: PASS')
"

# ─── GATE 9: Redis state key contract ─────────────────────────
cd trading-engine && python -c "
# Verify state_manager writes correct keys
from services.state_manager import StateManager
import inspect, ast, textwrap

src = inspect.getsource(StateManager)
for key in ['engine:status', 'engine:last_heartbeat', 'trader:', ':performance', ':state',
            'positions:open', 'ml:model:', 'ml:rl_agent', 'context:']:
    assert key in src, f'Missing Redis key pattern: {key}'
print('GATE 9 REDIS KEYS: PASS')
"

# ─── GATE 10: Trade journal contract ─────────────────────────
cd trading-engine && python -c "
from services.trade_journal import TradeJournal
import inspect
src = inspect.getsource(TradeJournal.log_trade)
required_fields = ['timestamp','trader','symbol','side','size','entry',
                   'stop_loss','take_profit','rr_ratio','confidence','pnl',
                   'commission','strategy','session','ml_model_scores',
                   'state_at_entry','rl_action']
for f in required_fields:
    assert f in src, f'Missing journal field: {f}'
print('GATE 10 JOURNAL CONTRACT: PASS')
"

# ─── GATE 11: Docker build ────────────────────────────────────
docker compose build trading-engine 2>&1 | tail -5
echo "GATE 11 DOCKER BUILD: CHECK ABOVE"

# ─── GATE 12: Container health ────────────────────────────────
docker compose up -d trading-engine
sleep 15
curl -sf http://localhost:8000/health | python -c "
import sys, json
d = json.load(sys.stdin)
assert d.get('status') == 'ok', f'Health failed: {d}'
assert d.get('traders') == 5, f'Expected 5 traders, got {d.get(\"traders\")}'
print('GATE 12 HEALTH CHECK: PASS')
"

# ─── GATE 13: Event flow (MARKET_DATA → SIGNAL_GENERATED) ────
cd trading-engine && python -c "
import redis, json, time, threading

r = redis.Redis(host='localhost', port=6379, decode_responses=True)
received = []

def listen():
    p = r.pubsub()
    p.subscribe('SIGNAL_GENERATED', 'TRADE_EXECUTED')
    for msg in p.listen():
        if msg['type'] == 'message':
            received.append(json.loads(msg['data']))

t = threading.Thread(target=listen, daemon=True)
t.start()
time.sleep(1)

# Emit fake market data bar (London session time, EURUSD)
r.publish('MARKET_DATA', json.dumps({
    'symbol': 'EURUSD', 'timeframe': '15M',
    'open': 1.0850, 'high': 1.0865, 'low': 1.0838,
    'close': 1.0860, 'volume': 12500,
    'timestamp': '2024-01-15T08:15:00Z'   # London session bar
}))
time.sleep(5)

print(f'GATE 13 EVENT FLOW: received {len(received)} events (0 is OK if no setup active)')
if received:
    e = received[0]
    assert 'trader_id' in e or 'ticket' in e, f'Malformed event: {e}'
print('GATE 13: PASS')
"

# ─── GATE 14: ML disabled mode ───────────────────────────────
cd trading-engine && ML_ENABLED=false python -c "
from main import ProductionTradingEngine
import os
os.environ['ML_ENABLED'] = 'false'
# Should not import tensorflow, torch, transformers
import sys
tfe_loaded = any('tensorflow' in m for m in sys.modules)
torch_loaded = any('torch' in m for m in sys.modules)
print(f'TF loaded when ML_ENABLED=false: {tfe_loaded} (expected: False)')
print(f'Torch loaded when ML_ENABLED=false: {torch_loaded} (expected: False)')
print('GATE 14 ML DISABLED MODE: CHECK ABOVE')
"

# ─── GATE 15: Backtest runs without crash ────────────────────
python scripts/run_backtest.py 1 2 3 4 5 2>&1 | tail -10
ls backtest_results/ | tail -1
echo "GATE 15 BACKTEST: CHECK ABOVE"
```

---

## PHASE 13 — EVALUATOR SCORECARD

Score every component after all gates pass. Loop continues until all ≥ 8.

```
EVALUATOR SCORECARD — Iteration N

INDICATORS
  market_structure.py            [ /10]  vectorized, no loops, no NaN, all columns

FEATURE ENGINEERING
  feature_engine.py              [ /10]  shapes correct, float32, clamped, no lookahead

ML MODELS
  gru_lstm_predictor.py          [ /10]  fallback works, temporal split, saves/loads
  regime_classifier.py           [ /10]  4 classes, hysteresis, fallback, rule-based labels
  quality_scorer.py              [ /10]  journal-sourced labels, dict→array ordering
  sentiment_model.py             [ /10]  Gold inversion, cache, VADER fallback
  rl_agent.py                    [ /10]  record_outcome interface, heuristic fallback, reward

TRADERS
  base_trader.py                 [ /10]  all 8 guards, position sizing, session awareness
  trader_1_ny_ema.py             [ /10]  EMA stack, London align, pullback, ML gates
  trader_2_fvg_bos.py            [ /10]  FVG registry, BOS bias, R:R pre-check, dead zone
  trader_3_london_bo.py          [ /10]  Asian range cache, sweep detection, hard close
  trader_4_news_momentum.py      [ /10]  structural (NOT fade), sentiment hard gate
  trader_5_asian_mr.py           [ /10]  XAUUSD excluded, RANGING gate, dual oscillator

SERVICES
  session_manager.py             [ /10]  UTC-correct, hard close logic
  news_service.py                [ /10]  pre-news zone, blocking, Redis feed
  signal_pipeline.py             [ /10]  ML→rules→RL→gate flow, ensemble formula
  risk_engine.py                 [ /10]  all circuit breakers, ML position scaling
  trade_journal.py               [ /10]  all Contract 4 fields, rolling stats, RL episodes

MAIN + INTEGRATION
  main.py                        [ /10]  5 traders, health server, ML_ENABLED/PAPER flags
  docker_build                   [ /10]  clean build, no errors
  health_check                   [ /10]  {status:ok, traders:5}
  event_flow                     [ /10]  MARKET_DATA → engine → contracts
  ml_disabled_mode               [ /10]  no TF/torch imports when ML_ENABLED=false

SCRIPTS
  run_backtest.py                [ /10]  5 traders, session-aware, correct JSON output
  retrain_incremental.py         [ /10]  all 5 model types handled
  retrain_scheduler.py           [ /10]  Sunday 02:00 UTC, updated model list

DOCUMENTATION
  docs/CLAUDE.md updated         [ /10]  reflects new system completely

──────────────────────────────────────────────────────────────
COMPOSITE AVERAGE                [ /10]
```

---

## PHASE 14 — CRITIC_AGENT CHECKLIST (every iteration)

Zero tolerance on these. One failure = FIXER_AGENT activates before EVALUATOR scores.

**No lookahead anywhere:**
- [ ] No `shift(-1)` or negative shift in any feature
- [ ] GRU-LSTM labels computed from `close[t + horizon]` — these rows are excluded from training features
- [ ] XGBoost labels from journal (already closed trades only)
- [ ] Backtest uses `df.iloc[:i+1]` at each bar — not full df

**Contract fidelity:**
- [ ] All 7 Redis event schema fields present (Contract 1)
- [ ] All Redis state key patterns present (Contract 2)
- [ ] Health endpoint returns `{status, timestamp, traders, mode}` (Contract 3)
- [ ] Both journal files have exact columns from Contract 4
- [ ] Backtest JSON has all Contract 5 keys
- [ ] All 5 trader IDs match Contract 6 exactly
- [ ] ML_ENABLED and PAPER_TRADING flags honoured (Contract 7)

**No integer indexing on datetime DataFrames:**
- [ ] Zero occurrences of `.at[integer,` anywhere
- [ ] Zero occurrences of `.iloc[i]` inside a loop where `i` is a row counter on a datetime-indexed df

**Gold-specific:**
- [ ] `Trader5AsianMR.EXCLUDED_SYMBOLS` contains 'XAUUSD'
- [ ] `SentimentModel.analyze()` inverts score for XAUUSD instrument
- [ ] London Breakout uses 10× pip scaling for XAUUSD

**RL correctness:**
- [ ] `record_outcome()` method name unchanged
- [ ] RL cannot generate a trade without a corresponding rule-based signal
- [ ] `select_action()` returns `(0, 0.0)` if selected strategy not in available_signals
- [ ] `state_at_entry` (42 floats) logged to journal in every trade

**ML disabled path:**
- [ ] When `ML_ENABLED=false`: traders run with `quality_score=0.5` (neutral), no ML gates applied
- [ ] TensorFlow, torch, transformers NOT imported at module level — inside `if ML_ENABLED` blocks
- [ ] Engine starts in < 5 seconds when `ML_ENABLED=false`

**Session correctness:**
- [ ] Dead zone 12:00–13:00 UTC blocks ALL traders (verified in `BaseTrader._in_dead_zone()`)
- [ ] `Trader5AsianMR` hard closes ALL positions at `06:45 UTC`
- [ ] `Trader3LondonBO` hard closes ALL positions at `12:00 UTC`
- [ ] `Trader1NYEMA` hard closes ALL positions at `17:00 UTC`
- [ ] `Trader4NewsMomentum` hard closes event positions `60 min` after news time

**Docker:**
- [ ] `tensorflow-cpu` in requirements.txt (not full TF — image size)
- [ ] `pip install --no-cache-dir --timeout=120 --retries=5` in Dockerfile
- [ ] Health server on port 8000 (docker-compose expects this)

---

## COMPLETION CRITERIA

`LOOP_CONTROLLER` emits **✅ ENGINE REPLACEMENT COMPLETE** only when:

```
□ All 33 EVALUATOR components score ≥ 8/10
□ All 15 verification gates pass (no skipped gates)
□ CRITIC checklist: every item checked green
□ docker compose up -d trading-engine succeeds
□ curl http://localhost:8000/health → {status:ok, traders:5}
□ Both journal files (CSV + JSONL) created in logs/
□ backtest_results/ produces valid JSON with 5 trader entries
□ ML_ENABLED=false: engine starts without importing TF/torch
□ ML_ENABLED=true: all 5 models instantiate without crashing (fallback ok)
□ docs/CLAUDE.md updated to reflect new engine fully
□ trading-engine-backup-*/ exists (safety net confirmed before demolition)
□ backend/, frontend/, docker-compose.yml, .env — git diff shows zero changes
```

If any item is unchecked: document file + line + issue → FIXER_AGENT → re-run affected gates → re-score → loop continues.

---

## APPENDIX — SESSION SCHEDULE

| UTC Window | Session | Active Traders |
|---|---|---|
| 00:00–02:00 | Asian build | None — range accumulates |
| 02:00–06:30 | Asian trade | **Trader5** (USDJPY, EURJPY, AUDJPY, EURUSD, AUDUSD) |
| 06:45 | HARD CLOSE | Trader5 exits all positions |
| 07:00–10:00 | London open | **Trader3** (EURUSD, GBPUSD, XAUUSD) |
| 07:00–12:00 | London full | **Trader2** (all instruments) |
| 12:00 | HARD CLOSE | Trader3 exits all London positions |
| 12:00–13:00 | DEAD ZONE | Zero entries — all traders blocked |
| 13:00–17:00 | NY session | **Trader1** (EURUSD, GBPUSD, USDJPY, XAUUSD) |
| 13:00–18:00 | NY extended | **Trader2** continues (all instruments) |
| 17:00 | HARD CLOSE | Trader1 exits all NY positions |
| News events | Any session | **Trader4** (EURUSD, GBPUSD, XAUUSD) |
| 60 min post-news | | Trader4 exits event positions |

---

*Begin with ARCHITECT_AGENT confirming the directory tree. Then READER_AGENT reads backend/src/services/state_reader.py, backend/src/routes/traders.py, and docker-compose.yml to extract every contract detail. Only then does CODER_AGENT write the first file.*
