# Multi-Bot ICT Smart Money Trading System

A production-grade automated trading system implementing Inner Circle Trader (ICT) and Smart Money Concepts (SMC) across multiple asset classes.

## Overview

This system deploys 6 independent trading bots:

- **4 Trader Bots**: EMA Crossover, Mean Reversion, Breakout, News-Driven
- **2 Monitor Bots**: Chart Surveillance, News Sentiment Tracking

## Architecture

- **Frontend**: React.js dashboard for monitoring and control
- **Backend**: FastAPI service for REST + monitoring endpoints
- **Trading Engine**: Python microservices for bot logic and execution
- **Data Layer**: PostgreSQL, MongoDB, InfluxDB for persistence
- **Infrastructure**: Docker containers orchestrated with Docker Compose

## Key Features

- ICT/SMC strategy implementation (Order Blocks, FVGs, Liquidity Sweeps)
- Multi-asset support (Forex, Commodities, Crypto)
- Risk management with position locking and drawdown controls
- Real-time monitoring and alerting
- Backtesting framework for strategy validation
- Production-ready with logging, error handling, and health checks

## Quick Start

1. Clone the repository
2. Configure environment variables (copy `trading-system/.env.example` to `trading-system/.env`)
3. Run `docker compose -f docker-compose.dev.yml up -d --build`
4. Access the dashboard at http://localhost:3001
# Inside the trading-engine container:
python -m pytest tests/test_risk_math.py -v --tb=short

# Run backtest with new metrics:
python scripts/run_backtest.py --split validation

# Run walk-forward validation:
python pipeline/step6_backtest.py

## Documentation

See [docs/](docs/) for detailed documentation.

## Disclaimer

This is a trading system for educational and research purposes. Use at your own risk. Past performance does not guarantee future results.

Here is the full analysis of where each component lacks edge and what to change:

GRU-LSTM Predictor
Problem 1 — Last-timestep pooling discards temporal attention

Line 113 in gru_lstm_predictor.py:


out = self.drop2(out[:, -1, :])   # last timestep
The model sees 30 bars but only reads the final hidden state. The intermediate LSTM outputs encode where in the 30-bar window the relevant signal appears — a swing, a BOS, an FVG forming — and that positional information is thrown away. A multi-head attention layer over all 30 timesteps before pooling would let the model learn to attend to the most informative bars in the window.

Problem 2 — Training labels are almost certainly binary next-candle direction

This is the most damaging structural issue. If the label is 1 = close[t+1] > close[t], the model is being trained to solve a task that doesn't correspond to profitability. A signal that wins on "was the next candle up" with 54% accuracy at 1:1 R:R breaks even but loses money to spread and commission.

The correct label is TP-before-SL: given entry at close[t], long SL at close[t] - 1.5×ATR, long TP at close[t] + 2.5×ATR — does price reach TP before SL within the next 48 bars? This is a forward-simulation label that directly aligns the training objective with the profitability objective. The model would learn the patterns that precede a specific R:R outcome rather than the patterns that precede a directional close.

Problem 3 — No regime conditioning

The same weights handle trending, ranging, and volatile regimes. A bar in a trending regime with a pullback to EMA21 is structurally different from the same bar in a ranging regime. The model has no regime input, so it mixes all regime data into one distribution, producing mediocre weights for each.

Options: (a) separate models per HTF regime label, or (b) add the 3-class HTF regime logit as an additional input feature to the sequence. Option (b) is simpler.

Problem 4 — entry_depth mapping is arbitrary

Line 208:


entry_depth = float(np.clip(expected_move * 100.0, 0.0, 1.0))
expected_move is a raw regression output in units of price return. Multiplying by 100 and clipping to [0, 1] is an ad-hoc transformation that has no semantic meaning. entry_depth should represent how far price has already moved toward the target, estimated from the pullback depth. This is useful as a timing signal — entering at 30% depth into an order block is better than at 80% depth. This could come from (close - ob_low) / (ob_high - ob_low) for a bullish OB entry.

QualityScorer
Problem 1 — Selection bias in training data

The scorer is trained on trades that already passed the signal pipeline (direction gate, uncertainty gate, expected-R gate). It never sees trades that were rejected. This means it learns to score "relatively good trade vs. relatively bad trade among already-gated trades" — not "trade vs. no-trade." The distribution it trains on is fundamentally truncated. The IC (information coefficient) will be lower than it appears because the hard work was already done upstream.

Problem 2 — Circular label definition

If the EV label is realized_rr for wins and -1.0 for losses, the scorer is being trained on an outcome-labelled subset. But realized_rr for wins depends on where TP was placed, which depends on the signal confidence score — creating a feedback loop if the QualityScorer output feeds back into signal confidence.

The cleanest fix: train the QualityScorer on a feature-only dataset with a pure forward-simulation outcome label (TP-before-SL, same as the GRU-LSTM fix above), without any model output features as inputs. This eliminates the circularity.

Problem 3 — No regime conditioning

Same issue as the GRU-LSTM. The QualityScorer has no regime input, so a "quality score" of 0.7 means different things in a trending vs ranging market.

RLAgent
Problem — Action space references non-existent traders

Lines 6–14 in rl_agent.py:


Actions: 16
  0       = NoTrade
  1–5     = Trader 1–5 @ default threshold (0.55)
  6–10    = Trader 1–5 @ medium threshold (0.65)
  11–15   = Trader 1–5 @ high threshold (0.75)
The individual Traders 1–5 were unified into ml_trader. Actions 1–15 all route to the same trader now, so the "which trader" dimension of the action is meaningless. The agent trained with this space has been learning to distinguish meaningless distinctions.

Redesign

The RL agent's real value is regime-adaptive threshold selection — it can observe drawdown, recent IC, and market regime, then decide how aggressive the confidence filter should be. The new action space should be:


Action 0: NoTrade
Actions 1–8: trade with threshold in [0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.80]
The agent then has a genuine decision: in a strongly trending regime with positive recent IC, lower the threshold (more trades); in a volatile ranging regime with degrading IC, raise the threshold (fewer, higher-quality trades). This is a real RL problem with meaningful state-action coupling.

The N_STATE=43 input vector should include: HTF regime logits (3), LTF regime logits (4), recent win rate (1), rolling IC (1), current drawdown (1), weeks in drawdown (1), session_code (1), hour_sin/cos (2), recent daily PnL (1), plus the market features currently in there. These should be explicitly documented and stable, not implicitly assembled.

market_structure.py Indicators
Missing: VWAP and VWAP bands

VWAP (Σ(typical_price × volume) / Σvolume) is the institutional reference price. Banks and market makers execute around VWAP. It is calculable from OHLCV data with no lookahead. Distance from VWAP in ATR units is a more meaningful entry filter than distance from a simple EMA, because VWAP incorporates actual transaction volume — a 100-pip move on thin volume has less VWAP impact than on heavy volume.

Key derived features:

vwap_dist_atr = (close - vwap) / atr — how far price is from institutional fair value
vwap_upper_band = vwap + 2×vwap_std, vwap_lower_band = vwap - 2×vwap_std — VWAP standard deviation bands function like Bollinger Bands but volume-weighted
Missing: Volume delta (directional volume)

The current volume_ratio is symmetric — a high-volume candle closing at its low is treated identically to one closing at its high. Volume delta estimates buy vs sell pressure from close position within the bar:


buy_vol  = volume × (close - low)  / (high - low + ε)
sell_vol = volume × (high - close) / (high - low + ε)
delta    = buy_vol - sell_vol
Cumulative delta (rolling 20-bar sum) is a leading divergence signal: when price makes a new high but cumulative delta is falling, sellers are absorbing the breakout — a reversal setup. When price is flat but cumulative delta is rising, buyers are quietly accumulating.

Missing: Wick ratio (bar auction result)


upper_wick = high - max(open, close)
lower_wick = min(open, close) - low
wick_ratio = lower_wick / (upper_wick + lower_wick + ε)
A value near 1.0 means buyers dominated the bar's auction (rejected lows). A value near 0.0 means sellers dominated (rejected highs). This is directly relevant for entry quality at order blocks and FVGs — you want to enter from a zone where the wick ratio confirms buyers are defending.

What's already correct

The BOS, FVG, liquidity sweep, and order block detection are all causally correct — _confirmed_swing_arrays emits swing levels only after confirmation bars, not in advance. This is a genuine edge over simpler implementations that backtest-peek. The SR zone clustering via Numba is also solid. These should be preserved unchanged.

PortfolioManager
Problem 1 — _streak_scalar is adaptive sizing, contradicts fixed-fractional

Lines 395–422 in portfolio_manager.py. The function reduces size to 0.35× after 4 consecutive losses. This is the mirror of a martingale — it's anti-martingale sizing, which is psychologically intuitive but statistically incorrect for a system with a positive edge.

Fixed-fractional sizing guarantees that each trade risks exactly equity × RISK_PER_TRADE regardless of recent outcomes. The streak information is already captured by the RiskEngine's consecutive-loss cooldown (which stops trading entirely after N losses rather than reducing size). Two overlapping loss-response mechanisms conflict: the RiskEngine pauses trading after 3 consecutive losses, and the PortfolioManager reduces size after 3 consecutive losses. In practice whichever runs first wins, but the interaction is untested.

The _streak_scalar should be removed. The single authoritative loss-response is in RiskEngine (CONSECUTIVE_LOSS_COOLDOWN_BARS).

Problem 2 — _volatility_scalar implicitly increases size in calm markets

When current_atr < nominal_atr, nominal / current > 1.0, so the scalar exceeds 1.0 (capped at 1.25). This means during calm/ranging periods you take larger positions. Calm periods in forex often precede high-impact news — sizing up into low-volatility is the opposite of what a disciplined system should do. The scalar should only reduce size (cap at 1.0), never increase it: min(nominal / atr, 1.0).

Problem 3 — Dynamic R:R assumes higher confidence → justified higher target

The _dynamic_rr_multipliers function scales TP targets from 2×SL at low confidence to 4×SL at confidence=0.90. This assumes the model is calibrated such that 0.90 confidence corresponds to a 4:1 R:R trade actually reaching that target. There's no empirical validation of this relationship in the codebase. Unless the backtest explicitly confirms that high-confidence signals hit higher R-multiples, this inflates paper P&L during backtesting versus live performance. A fixed TP (from settings ATR multipliers) with no confidence scaling would be more honest about what the system actually captures.
