You are an expert quantitative ML engineer specialising in financial regime detection.

My current regime classifier is failing. The logs show that the HTF 4H bias classifier collapses toward BIAS_NEUTRAL and barely detects BIAS_DOWN, while the LTF 1H behaviour classifier almost never detects TRENDING. The current flat regime labels are not reliable enough for live trading.

Your task is to refactor the regime detection system.

IMPORTANT:
This is a pure ML trading system, but regime detection may use mathematical/statistical rule-based labels as training targets. The final trading engine must use regime outputs as filters, not as standalone traders.

Current problem:
- HTF bias model:
  - BIAS_UP recall is weak
  - BIAS_DOWN recall is extremely weak
  - BIAS_NEUTRAL dominates
- LTF behaviour model:
  - TRENDING recall is almost zero
  - CONSOLIDATING and VOLATILE dominate predictions
- This suggests the labels are overlapping, noisy, or poorly separated.

Refactor goals:

1. Replace the current flat regime classification design with a two-axis regime system.

Axis A: Directional bias
- BIAS_UP
- BIAS_DOWN
- BIAS_NEUTRAL

Axis B: Behaviour state
- TREND_SCORE
- RANGE_SCORE
- CHOP_SCORE
- VOLATILITY_PERCENTILE
- CONSOLIDATION_SCORE

Do not force all behaviour states into one mutually exclusive class unless absolutely necessary. Prefer multi-label or score-based outputs.

2. Add robust mathematical regime features:
- ADX
- +DI and -DI
- EMA 20/50/200 slope
- price distance from EMA 50 and EMA 200
- ATR / close
- ATR percentile per symbol
- rolling volatility percentile per symbol
- Bollinger bandwidth percentile
- efficiency ratio
- rolling range percentile
- candle body ratio
- wick ratio
- range expansion z-score
- higher-high / lower-low structure
- symbol group feature: dollar, cross, yen, gold
- session feature if available

3. Implement efficiency ratio:

efficiency_ratio = abs(close - close.shift(window)) / sum(abs(close.diff()), window)

Use this to separate true trends from noisy chop.

4. Use per-symbol normalisation.
Do not compare raw ATR or raw volatility across EURUSD, GBPJPY, and XAUUSD.
Use:
- ATR / close
- rolling percentiles per symbol
- z-scores per symbol
- symbol group encoding

5. Redesign the HTF bias labels.

BIAS_UP should require:
- +DI > -DI
- ADX above threshold
- EMA slope positive
- close above EMA 50 or EMA 200
- positive forward directional movement if using forward structural labels

BIAS_DOWN should require:
- -DI > +DI
- ADX above threshold
- EMA slope negative
- close below EMA 50 or EMA 200
- negative forward directional movement if using forward structural labels

BIAS_NEUTRAL should only be used when neither bullish nor bearish structure is clear.
Do not allow BIAS_NEUTRAL to become a dumping ground for everything.

6. Redesign LTF behaviour labels.

TRENDING should require:
- ADX above threshold
- efficiency ratio above threshold
- EMA slope meaningful
- directional persistence

RANGING should require:
- ADX below threshold
- efficiency ratio low
- price oscillating around mean
- moderate volatility

CONSOLIDATING should require:
- low ATR percentile
- low Bollinger bandwidth percentile
- compressed rolling range

VOLATILE should require:
- high ATR percentile
- high rolling volatility percentile
- large range expansion z-score

Allow overlap by producing scores where possible:
- trend_score
- range_score
- consolidation_score
- volatility_score
- chop_score

7. Add a final regime decision function.

Example output:
- TRADEABLE_TREND
- TRADEABLE_TREND_HIGH_VOL
- RANGE
- CONSOLIDATION
- NO_TRADE_CHOP
- NO_TRADE_EXTREME_VOL
- UNCERTAIN

The trading engine should block:
- NO_TRADE_CHOP
- NO_TRADE_EXTREME_VOL
- UNCERTAIN

8. Improve model training.

If using classifiers:
- use chronological validation only
- report per-class precision, recall, F1, balanced accuracy
- report confusion matrix
- reject weights if any critical class recall is below threshold
- tune thresholds using validation set only
- do not tune on test set

If using multi-label outputs:
- train separate binary classifiers or a multi-output model for:
  - is_trend
  - is_range
  - is_consolidation
  - is_volatile
  - is_chop

9. Add diagnostics.

After training, print:
- label distribution by symbol
- label distribution by year
- label distribution by symbol group
- per-class recall
- per-class precision
- confusion matrix
- examples of misclassified TRENDING as CONSOLIDATING
- examples of BIAS_DOWN misclassified as NEUTRAL
- feature importance if using tree models

10. Add tests.

Create unit tests for:
- efficiency ratio
- ATR percentile
- Bollinger bandwidth percentile
- HTF bias label generation
- LTF behaviour score generation
- no future leakage
- final regime decision logic

11. Important restrictions.

Do not create individual rule-based traders.
Do not make regime detection a trading strategy by itself.
Do not use future candles in live features.
Do not random-shuffle time-series data.
Do not compare raw volatility across symbols without normalisation.
Do not save regime weights if validation per-class recall is misleading.

Expected deliverables:
- list of files changed
- explanation of new regime architecture
- explanation of new labels/scores
- explanation of how future leakage is prevented
- commands to train regime detector
- commands to evaluate regime detector
- sample output report

My recommendation
Do not try to fix this by simply lowering the acceptance threshold.
Your current model is failing in the exact classes you need most:
BIAS_DOWNTRENDING
Lowering the threshold would just save bad weights.
The correct fix is:
Redesign regime labels → add better trend/chop features → normalise per symbol → move from flat class labels to score-based/multi-label regime detection.
That should make the regime module much more useful for the pure ML trading system.
