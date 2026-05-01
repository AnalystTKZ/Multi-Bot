Before editing files, first scan the repository and identify where the old individual trader modules are used. Then propose a migration plan to disconnect them from the active execution path and replace them with one unified ML trading engine. Do not delete files immediately unless necessary. First make the pure ML path work, then mark old trader modules as deprecated.

You are an expert quantitative trading systems engineer, ML engineer, and risk architect.

I have an existing forex/gold trading system. Your task is to refactor it into a mathematically disciplined, statistically validated, risk-controlled ML trading system.

The goal is NOT to add random indicators. The goal is to redesign the system so every trade is justified by measurable expectancy, volatility-adjusted risk, regime filtering, uncertainty control, and robust backtesting.

Do not remove existing working functionality unless it conflicts with the new architecture. Preserve current trader modules where possible, but refactor them into cleaner layers.

==================================================
PRIMARY OBJECTIVE
==================================================

Refactor the trading system into the following architecture:

1. Data Layer
   - Load OHLCV data per symbol and timeframe.
   - Include spread, commission, slippage, session, and symbol metadata where available.
   - Prevent future leakage.
   - Ensure all features are calculated using only past and current candle data.

2. Feature Engineering Layer
   - Create mathematical and statistical features:
     - returns
     - log returns
     - rolling volatility
     - ATR
     - ATR ratio
     - EMA slope
     - moving-average distance
     - RSI
     - ADX
     - Bollinger z-score
     - candle body ratio
     - wick ratio
     - range expansion
     - session flags
     - multi-timeframe alignment
     - rolling correlation where applicable
   - Features must be reusable across traders and models.
   - No duplicated feature logic inside individual trader files.

3. Regime Detection Layer
   - Add a regime classifier or rule-based regime detector.
   - Classify market states into:
     - TRENDING
     - RANGING
     - CHOP
     - HIGH_VOLATILITY
     - LOW_VOLATILITY
     - NEWS_SPIKE if news/calendar data exists
   - Use ADX, ATR ratio, EMA slope, rolling volatility, candle structure, and range compression/expansion.
   - Traders must avoid CHOP unless they are explicitly mean-reversion strategies.
   - Gold/XAUUSD must have stricter volatility controls.

4. ML Prediction Layer
   - Treat ML models as signal generators only.
   - Do not allow models to directly place trades or size positions.
   - Support three outputs:
     - direction classification
     - magnitude or expected-R regression
     - uncertainty or confidence score
   - Prefer interpretable models such as LightGBM, XGBoost, RandomForest, or logistic regression before deep learning.
   - If GRU/LSTM models exist, keep them, but wrap their outputs into the same prediction interface.

5. Signal Decision Layer
   - Convert model outputs into trade decisions using strict statistical rules.
   - A trade is allowed only when:
     - model direction probability exceeds threshold
     - expected R exceeds threshold
     - uncertainty is below threshold
     - regime is allowed
     - spread is below max allowed spread
     - reward-to-risk ratio is acceptable
     - no daily/weekly risk limit is breached
   - Add a NO_TRADE decision as a first-class output.
   - The system should prefer missing trades over taking weak trades.

6. Risk Management Layer
   - Implement fixed fractional risk sizing.
   - Position size must be calculated from:
     - account balance
     - risk percentage
     - stop-loss distance
     - pip/point value
     - symbol-specific contract size
   - Add hard limits:
     - max risk per trade
     - max daily loss
     - max weekly loss
     - max open trades
     - max correlated exposure
     - max symbol exposure
     - max consecutive losses before cooldown
   - No martingale.
   - No grid averaging.
   - No increasing lot size after losses.
   - Risk engine must be able to reject trades even when ML predicts a good setup.

7. Stop Loss and Take Profit Layer
   - Replace fixed arbitrary stops with volatility-aware stops.
   - Support ATR-based stops:
     - stop_loss = ATR × stop_multiplier
     - take_profit = ATR × target_multiplier
   - Support structure-based stops where existing traders require them.
   - Enforce minimum reward-to-risk ratio.
   - Enforce maximum stop distance so trades do not become oversized.
   - Add timeout exits where appropriate.

8. Backtesting and Validation Layer
   - Backtest the full trading system, not only model accuracy.
   - Include realistic:
     - spread
     - commission
     - slippage
     - execution delay
     - rollover/session spread widening if available
   - Track metrics:
     - expectancy per trade
     - profit factor
     - Sharpe ratio
     - Sortino ratio
     - max drawdown
     - win rate
     - average win
     - average loss
     - average R
     - total trades
     - trades per symbol
     - trades per regime
     - monthly returns
     - worst month
     - best month
     - longest losing streak
     - exposure time
   - Add acceptance/rejection rules:
     - expectancy must be positive after costs
     - profit factor should be above 1.25
     - max drawdown must remain below configured threshold
     - strategy must remain profitable under increased spread/slippage stress tests
     - strategy must not depend on one lucky month or one lucky trade

9. Walk-Forward Validation Layer
   - Do not use random train/test splitting.
   - Implement chronological splits only.
   - Use walk-forward validation:
     - train period
     - validation period
     - out-of-sample test period
   - Never tune thresholds on the final test set.
   - Ensure no future leakage in feature engineering, target creation, or scaling.

10. Reporting Layer
   - Generate a detailed report after backtesting.
   - Include:
     - system configuration
     - model thresholds
     - risk settings
     - performance summary
     - rejected trades summary
     - trades by regime
     - trades by symbol
     - drawdown curve data
     - monthly returns
     - stress test results
     - final PASS/FAIL decision
   - A strategy should be marked FAIL if it is profitable only before costs or only under unrealistic execution assumptions.

==================================================
MATHEMATICAL AND STATISTICAL RULES TO IMPLEMENT
==================================================

Implement or centralise the following calculations:

1. Expectancy

expectancy = (win_rate * average_win_R) - (loss_rate * average_loss_R)

2. Break-even win rate

break_even_win_rate = 1 / (1 + reward_to_risk)

3. Position sizing

risk_amount = account_balance * risk_percent

position_size = risk_amount / stop_loss_value

4. ATR-based stop loss and take profit

stop_distance = ATR * stop_multiplier

target_distance = ATR * target_multiplier

5. Z-score

z_score = (price - rolling_mean) / rolling_std

6. Rolling volatility

rolling_volatility = std(returns over rolling window)

7. Sharpe ratio

sharpe = mean(strategy_returns) / std(strategy_returns)

Use annualisation only if timeframe assumptions are explicitly defined.

8. Sortino ratio

sortino = mean(strategy_returns) / downside_deviation

9. Profit factor

profit_factor = gross_profit / gross_loss

10. Maximum drawdown

drawdown = (equity_peak - equity_current) / equity_peak

11. Correlation filter

Do not allow too many open trades with highly correlated exposure.

Example:
- EURUSD long and GBPUSD long may both represent USD short exposure.
- Gold long and USD short exposure should be treated carefully depending on correlation regime.

==================================================
RECOMMENDED TRADE DECISION LOGIC
==================================================

Create a central decision function similar to this:

def should_trade(prediction, regime, market_state, risk_state, config):
    if risk_state.daily_loss_exceeded:
        return NO_TRADE

    if risk_state.weekly_loss_exceeded:
        return NO_TRADE

    if risk_state.max_open_trades_reached:
        return NO_TRADE

    if market_state.spread > config.max_spread:
        return NO_TRADE

    if market_state.news_window_active:
        return NO_TRADE

    if regime in ["CHOP", "NEWS_SPIKE"]:
        return NO_TRADE

    if prediction.direction_probability < config.min_direction_probability:
        return NO_TRADE

    if prediction.expected_R < config.min_expected_R:
        return NO_TRADE

    if prediction.uncertainty > config.max_uncertainty:
        return NO_TRADE

    if market_state.reward_to_risk < config.min_reward_to_risk:
        return NO_TRADE

    return TRADE_ALLOWED

The exact implementation can differ depending on the existing codebase, but the logic must be centralised and testable.

==================================================
TARGET DESIGN FOR ML MODELS
==================================================

Refactor target creation away from naive next-candle direction.

Avoid weak targets like:

target = next_close > current_close

Instead implement trade-aware targets such as:

1. TP-before-SL classification

For each candle:
- define ATR-based stop distance
- define target distance
- look forward N candles
- label BUY if price hits TP before SL
- label SELL if price hits downside TP before upside SL
- label NO_TRADE if neither setup is clean

2. Expected-R regression

future_R = future_return / stop_distance

3. Volatility or uncertainty target

future_volatility = std(future_returns over horizon)

The system should support configurable horizons and ATR multipliers.

==================================================
CONFIGURATION REQUIREMENTS
==================================================

Add or update a central config file with values such as:

min_direction_probability: 0.62
min_expected_R: 1.30
max_uncertainty: 0.25
min_reward_to_risk: 1.50
risk_per_trade: 0.005
max_daily_loss: 0.02
max_weekly_loss: 0.05
max_open_trades: 2
max_correlated_trades: 1
atr_stop_multiplier: 1.5
atr_target_multiplier: 2.5
max_spread_by_symbol:
  EURUSD: configurable
  GBPUSD: configurable
  USDJPY: configurable
  XAUUSD: configurable
gold_volatility_multiplier: stricter than forex pairs

Use sensible defaults, but make everything configurable.

==================================================
CODE QUALITY REQUIREMENTS
==================================================

1. Inspect the current repository before making changes.
2. Identify the existing architecture, trader modules, model files, backtest engine, config files, and data paths.
3. Produce a refactor plan before editing.
4. Keep changes incremental and testable.
5. Do not create duplicate competing systems.
6. Do not hardcode account balance, symbols, or thresholds inside strategy files.
7. Add type hints where practical.
8. Add docstrings for important mathematical functions.
9. Add unit tests for:
   - expectancy
   - drawdown
   - profit factor
   - ATR stop calculation
   - position sizing
   - target creation
   - signal rejection logic
10. Add integration tests for:
   - one symbol backtest
   - no future leakage
   - risk engine rejecting trades
   - spread/slippage stress test
11. Log rejected trades with reasons.
12. Ensure all reports are saved to an output directory.

==================================================
IMPORTANT SAFETY RULES
==================================================

Do not implement:
- martingale
- grid averaging
- doubling down after losses
- unlimited averaging into losing positions
- position sizing based directly on ML confidence
- random train/test splitting
- future data leakage
- strategy rules that only work with zero spread
- unrealistic candle-perfect execution

==================================================
EXPECTED DELIVERABLES
==================================================

After refactoring, provide:

1. A summary of the existing system structure.
2. A clear list of files changed.
3. A clear list of new files added.
4. Explanation of the new architecture.
5. Explanation of the mathematical/statistical rules implemented.
6. Explanation of the ML target design.
7. Explanation of the risk engine.
8. Explanation of the backtest validation process.
9. Commands to run:
   - unit tests
   - backtest
   - walk-forward validation
   - report generation
10. A short note on remaining limitations or assumptions.

==================================================
FINAL GOAL
==================================================

The final system should not simply generate more trades.

The final system should be able to say:

NO_TRADE

most of the time, and only trade when the model prediction, regime, volatility, risk, and expected reward justify the trade.

Profitability must be evaluated through positive expectancy, profit factor, drawdown control, robustness under costs, and walk-forward out-of-sample performance.
