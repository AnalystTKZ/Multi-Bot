# Day-Trading Feature Context

This document records the indicator and feature policy for the active ML trading path. The system should treat indicators as model context, not as standalone entry triggers.

## Current Feature Contracts

The active contracts live in `trading-engine/services/feature_engine.py`.

| Contract | Current width | Purpose |
|---|---:|---|
| `DAY_TRADING_FEATURES` | 22 | Shared causal MACD, RSI, VWAP, volume-pressure, and pivot context |
| `SEQUENCE_FEATURES` | 94 | 15M GRU execution sequence |
| `REGIME_4H_FEATURES` | 51 | HTF directional-bias classifier |
| `REGIME_1H_FEATURES` | 53 | LTF intraday-behaviour classifier |

Any change to these lists changes model input dimensions. Retrain affected models after changing them.

## Model Responsibilities

| Layer | Job | Feature usage |
|---|---|---|
| HTF regime | Directional bias and macro/4H context | Uses MACD/RSI momentum lightly, plus structure, ADX, EMA, volatility, candle pressure, macro |
| LTF regime | Is the current intraday environment tradeable? | Uses full day-trading context: VWAP, pivots, session-normalised volume proxies, RSI/MACD |
| GRU | Is this exact 15M setup worth entering? | Uses execution-level candle, structure, session, VWAP, pivot, momentum, and auction features |
| QualityScorer | Is the candidate worth taking after costs/location/RR? | Should treat location, spread, expected R, session, and realized setup quality as final selectivity context |

HTF should bias. It should not be the only reason a trade exists. LTF and GRU handle the day-trading decision.

## Indicator Policy

| Indicator group | Active features | Correct interpretation | Caveat |
|---|---|---|---|
| MACD | `macd_line_atr`, `macd_hist_atr`, `macd_hist_slope_3`, `macd_cross_age` | Momentum expansion, fading momentum, old-vs-fresh cross context | Do not use as "MACD cross = trade" |
| RSI | `rsi_14`, `rsi_slope_5`, MTF RSI | Momentum state and change of pressure | Overbought/oversold is not enough in a trend |
| VWAP | `vwap_dist_atr`, `vwap_band_position`, `vwap_slope_20`, `bars_since_vwap_cross` | Intraday fair value, acceptance above/below VWAP, extension risk | Useful mainly for intraday timing, not multi-week bias |
| Pivots / prior day | daily open, previous-day high/low, floor pivot, nearest pivot, pivot band, CPR width | Day structure, rejection zones, breakout/target context | Must use completed prior-day levels only |
| Volume / delta | `relative_volume_20`, `session_relative_volume`, `volume_delta_pct`, `cum_delta_20_z`, `delta_divergence` | Participation proxy and candle-auction pressure | Current processed data has all-zero volume, so see volume note below |
| Candle/auction | body, wick, close-location, `wick_auction_ratio`, `body_recovery_ratio` | Rejection, absorption, sweep recovery, impulse quality | Strong substitute when real volume is absent |
| Structure | BOS, MSS/CHoCH, FVG, sweeps, swing sequence, external range position | Actual setup context and trend/body location | Use with time-in-trend features to avoid labelling only breakout bars |
| Volatility | ATR percentile, volatility expansion, range expansion, BB width | Compression, expansion, exhaustion, high-risk volatility | Extreme volatility can reduce trade quality even when direction is right |
| Session/time | Asian/London/NY flags, cyclic hour, minutes since opens | Time-of-day behaviour and symbol/session fit | Session rules should be features/gates, not hidden assumptions |

## Volume Note

The current `processed_data/simple/ohlcv` parquet files have `volume == 0` for every checked 15M, 1H, and 4H row across EURJPY, EURUSD, GBPJPY, GBPUSD, USDJPY, and XAUUSD. That means raw volume is not currently usable as true market participation.

The feature builder handles this defensively:

- If all source volume is zero, it substitutes unit volume only to avoid divide-by-zero failures.
- `relative_volume_20` and `session_relative_volume` become neutral and should not be interpreted as real volume confirmation.
- `volume_delta_pct` becomes mostly a candle-close-location auction proxy, not true buy/sell volume.
- `cum_delta_20_z` and `delta_divergence` are also proxy signals until real tick volume exists.

For the current dataset, the better substitutes for volume confirmation are:

- `candle_body_ratio`
- `upper_wick_ratio` / `lower_wick_ratio`
- `wick_auction_ratio`
- `body_recovery_ratio`
- `candle_close_location`
- `trend_body_pressure_20`
- `range_expansion_zscore`
- `vol_expansion`
- `atr_pctile`
- `vwap_dist_atr`
- `vwap_band_position`
- `dist_to_nearest_pivot_atr`
- `breakout_close_strength`

If real tick volume is added later, keep the same normalized features but retrain GRU, HTF regime, and LTF regime from scratch or from compatible manifests only.

## Causal Rules

All active day-trading features must be known at the current bar close:

- MACD, RSI, VWAP, ADX, ATR, candle, and structure features use current/past bars only.
- Previous-day pivots use the completed prior day, not the current day's final high/low/close.
- Session-relative volume uses past samples from the same minute bucket where available.
- No feature may use future returns, future extremes, or labels during inference.

## Expected Benefit

These features should help the models distinguish:

- trend body vs stale breakout,
- active momentum vs exhausted extension,
- fair-value acceptance vs one-bar spike,
- clean intraday breakout vs pivot/VWAP rejection,
- real setup context vs random 15M bar.

They do not guarantee higher backtest performance by themselves. The validation target is better out-of-sample regime precision and better GRU/Quality selectivity after retraining.
