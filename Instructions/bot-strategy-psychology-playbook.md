# Bot Strategy and Psychology Playbook

This playbook maps each bot to concrete strategy and execution-discipline rules inspired by Mastery Trader Academy articles:

- [Risk Management: 7 Smart Rules to Avoid Big Losses](https://masterytraderacademy.com/risk-management-101-trading-tips-you-need/)
- [Mastering Trading Psychology](https://masterytraderacademy.com/mastering-trading-psychology-conquer-fear-greed-emotion-for-consistent-profits/)
- [Trading Psychology Secrets: 7 Powerful Ways to Stay Calm](https://masterytraderacademy.com/trading-psychology-mistakes-to-avoid-2025/)
- [Trailing Drawdown Explained](https://masterytraderacademy.com/trailing-drawdown-explained-2026/)
- [Just One More Trade](https://masterytraderacademy.com/just-one-more-trade-discipline-mistake/)

## Global Execution Psychology Layer (applies to all trader bots)

- Hard session trade cap (`max_trades_per_session`) to block overtrading.
- Cooldown between trades (`cooldown_after_trade_seconds`) to avoid impulse re-entry.
- Loss streak pause (`max_consecutive_losses`, `pause_after_loss_streak_seconds`) to stop revenge trading.
- Daily session roll reset for clean risk framing.

These controls are implemented in `trading-engine/traders/base_trader.py`.

## Trader 1 - EMA + ICT Trend Continuation

**Market role**
- 4H trend continuation after EMA crossover with ICT confluence.

**Entry framework**
- EMA crossover is required.
- ICT confirmation already enforced (minimum 2 confluences).
- Added trend spread filter to avoid weak/noise crossovers.

**Psychology/risk constraints**
- Max 2 trades per session.
- 10-minute cooldown between trades.

**File**
- `trading-engine/traders/trader_1_ema_ict.py`

## Trader 2 - Mean Reversion + ICT

**Market role**
- 1H reversal from RSI/Bollinger extremes with structure support.

**Entry framework**
- RSI oversold/overbought and Bollinger band excursion.
- Increased ICT confirmation requirement from any signal to at least 2 confluences.
- Added EMA trend-bias guard to avoid fading strong one-direction trend legs.

**Psychology/risk constraints**
- Max 2 trades per session.
- 15-minute cooldown between trades.

**File**
- `trading-engine/traders/trader_2_mean_reversion.py`

## Trader 3 - Breakout Failure / Liquidity Grab

**Market role**
- 15m failed breakout fades after liquidity sweeps.

**Entry framework**
- Liquidity sweep required.
- Added minimum sweep size threshold (relative to ATR) to avoid micro-fakes.
- Added close-direction confirmation versus previous candle for stronger rejection proof.

**Psychology/risk constraints**
- Max 3 trades per session.
- 5-minute cooldown between trades.

**File**
- `trading-engine/traders/trader_3_breakout.py`

## Trader 4 - News Fade

**Market role**
- 5m post-news overreaction fade for USD and XAU-linked symbols.

**Entry framework**
- High-impact event relevance required.
- Active window tightened to avoid first-minute chaos and late low-edge entries.

**Psychology/risk constraints**
- Max 1 trade per session (highest emotional/volatility risk).
- 30-minute cooldown.

**File**
- `trading-engine/traders/trader_4_news.py`

## Why This Improves System Reliability

- Converts human psychology failures into deterministic bot constraints.
- Reduces blow-up patterns: overtrading, revenge trades, and "one more trade" behavior.
- Keeps strategy behavior aligned with capital preservation and drawdown control.
- Makes behavior auditable and easier to tune per bot.

## Next Tuning Suggestions

- Move per-bot psychology/risk limits into environment-backed settings.
- Add per-bot daily loss caps (dollar or R-based) in execution/state layer.
- Add weekly review metrics for "rule-violation prevented" counters.
