First read the trading engine spun an agent to visit the following webpages and Read and report: goal is to find useful ways to make my strategy work.

https://arxiv.org/html/2407.04500v1 https://medium.com/@jsgastoniriartecabrera/from-academic-research-to-profitable-trading-building-a-market-regime-detection-algorithm-46a4791ee014 https://www.tradingview.com/script/IUR1cEYW-Market-Structure-Algo/



Refactor the trading system to remove RL from the active pipeline for now.

The system should no longer require, load, train, validate, cache, or execute the RL/PPO agent in the main trading path.

New active architecture:

1. Regime model
   - Determines tradeability context:
     TRADEABLE_UP, TRADEABLE_DOWN, NO_TRADE_CHOP, NO_TRADE_EXTREME_VOL, NO_TRADE_UNCERTAIN.

2. GRU model
   - Technical execution model only.
   - Outputs direction probability, expected move/R, and predicted volatility.

3. Quality scorer
   - EV judge.
   - Decides whether the proposed trade has positive expected value after costs.

4. Decision engine
   - Combines regime + GRU + quality + risk rules.
   - Final output is BUY, SELL, or NO_TRADE.

Remove RL from:
- live signal pipeline
- backtest decision path
- model loading requirements
- training pipeline requirements
- cache schema
- trade journal feature requirements
- quality/RL circular dependencies
- documentation references to required active models

Do not delete RL source files permanently. Mark RL as dormant/experimental.

Expected changes:
- main.py should not hard-fail if RL weights are missing.
- signal_pipeline.py should not call RL.
- run_backtest.py should not require RL.
- step6/step7 pipeline should not train or expect RL.
- Quality should not depend on RL outputs.
- Reports should not include RL metrics as required.
- Any env vars such as RL_ENABLED should default to false.
- If RL code remains, it must be isolated behind an explicit optional flag.

Final decision rule:

if risk_limits_breached:
    return NO_TRADE

if regime blocks trade:
    return NO_TRADE

if GRU confidence is too low:
    return NO_TRADE

if GRU expected_R is too low:
    return NO_TRADE

if Quality EV <= threshold:
    return NO_TRADE

return BUY or SELL

After implementation, provide:
1. files changed
2. files where RL was disconnected
3. confirmation that system can run without RL weights
4. commands to train Regime, GRU, and Quality
5. commands to backtest without RL
6. any dormant RL files left untouched

Regime → GRU → Quality → Decision Engine → Risk/Execution
