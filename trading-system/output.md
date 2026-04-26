Cleared done-check: training_summary.json
  Cleared done-check: training_7b_r1_summary.json
  Cleared done-check: training_7b_r2_summary.json
Environment : KAGGLE
  base      -> /kaggle/working/Multi-Bot/trading-system
  data      -> /kaggle/working/Multi-Bot/trading-system/training_data
  processed -> /kaggle/working/Multi-Bot/trading-system/processed_data
  ml_train  -> /kaggle/working/Multi-Bot/trading-system/ml_training
  weights   -> /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
  output    -> /kaggle/working
  kaggle/input -> /kaggle/input
    dataset: datasets  (has training_data=False, processed_data=False)
  WARNING: optional file missing (macro features reduced): /kaggle/working/Multi-Bot/trading-system/training_data/indices/VIX_1d.csv
  WARNING: optional file missing (macro features reduced): /kaggle/working/Multi-Bot/trading-system/training_data/fundamental/macro_releases.csv

All scripts and inputs verified.

=== Phase 0-5: Data preparation ===
  SKIP  Step 0 - Resample
  SKIP  Step 1 - Inventory
  SKIP  Step 2 - Cleaning
  SKIP  Step 3 - Alignment
  SKIP  Step 4 - Features
  START Step 5 - Split
2026-04-26 21:55:44,236 INFO Loading feature-engineered data...
2026-04-26 21:55:44,838 INFO Loaded 221743 rows, 202 features
2026-04-26 21:55:44,839 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-26 21:55:44,842 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-26 21:55:44,842 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-26 21:55:44,842 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-26 21:55:44,842 INFO No leakage confirmed: train < val < test timestamps

=== SPLIT COMPLETE (CALENDAR, no shuffling) ===
  Train:      130,951 bars  2016-01-04 → 2021-08-05
  Validation:  44,000 bars  2021-08-05 → 2023-08-04  ← Round 1 backtest
  Test:        46,792 bars  2023-08-07 → 2025-08-05  ← Blind / Round 2 backtest
  Features:   202
  Leakage check: PASS
  DONE  Step 5 - Split

  Data split (calendar):
    train         130951 bars  2016-01-04 → 2021-08-05
    validation     44000 bars  2021-08-05 → 2023-08-04
    test           46792 bars  2023-08-07 → 2025-08-05

=== Phase 7a: Train GRU + Regime (train set only) ===
  START Step 7a - GRU+Regime
2026-04-26 21:55:47,224 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-26 21:55:47,224 INFO --- Training regime ---
2026-04-26 21:55:47,225 INFO Running retrain --model regime
2026-04-26 21:55:47,395 INFO retrain environment: KAGGLE
2026-04-26 21:55:48,976 INFO Device: CUDA (2 GPU(s))
2026-04-26 21:55:48,987 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 21:55:48,987 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 21:55:48,987 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 21:55:48,989 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 21:55:48,990 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-26 21:55:49,141 INFO NumExpr defaulting to 4 threads.
2026-04-26 21:55:49,345 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 21:55:49,345 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 21:55:49,345 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 21:55:49,346 INFO Regime phase macro_correlations: 0.0s
2026-04-26 21:55:49,346 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-26 21:55:49,381 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 21:55:49,381 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:55:49,407 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:55:49,419 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:55:49,441 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:55:49,454 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:55:49,476 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:55:49,491 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:55:49,517 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:55:49,530 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:55:49,552 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:55:49,566 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:55:49,586 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:55:49,598 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:55:49,615 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:55:49,628 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:55:49,649 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:55:49,662 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:55:49,682 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:55:49,696 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:55:49,717 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:55:49,732 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 21:55:49,769 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 21:55:50,840 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 21:56:13,134 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 21:56:13,138 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 23.4s
2026-04-26 21:56:13,139 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 21:56:23,361 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 21:56:23,362 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 10.2s
2026-04-26 21:56:23,365 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 21:56:30,839 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 21:56:30,843 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 7.5s
2026-04-26 21:56:30,843 INFO Regime phase GMM HTF total: 41.1s
2026-04-26 21:56:30,843 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 21:57:38,227 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 21:57:38,230 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 67.4s
2026-04-26 21:57:38,230 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 21:58:07,248 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 21:58:07,251 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 29.0s
2026-04-26 21:58:07,251 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 21:58:27,975 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 21:58:27,976 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 20.7s
2026-04-26 21:58:27,977 INFO Regime phase GMM LTF total: 117.1s
2026-04-26 21:58:28,079 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-26 21:58:28,081 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:28,082 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:28,083 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:28,085 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:28,086 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:28,087 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:28,088 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:28,089 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:28,090 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:28,091 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:28,092 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 21:58:28,214 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:28,254 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:28,255 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:28,256 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:28,263 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:28,264 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:28,667 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-26 21:58:28,668 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-26 21:58:28,842 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:28,873 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:28,874 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:28,874 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:28,882 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:28,883 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:29,240 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-26 21:58:29,241 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-26 21:58:29,426 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:29,462 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:29,462 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:29,463 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:29,471 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:29,472 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:29,826 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-26 21:58:29,828 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-26 21:58:30,000 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:30,036 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:30,036 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:30,037 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:30,045 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:30,046 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:30,400 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-26 21:58:30,401 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-26 21:58:30,562 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:30,596 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:30,597 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:30,597 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:30,605 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:30,606 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:30,964 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-26 21:58:30,965 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-26 21:58:31,129 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:31,161 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:31,162 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:31,162 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:31,170 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:31,171 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:31,540 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-26 21:58:31,542 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-26 21:58:31,706 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 21:58:31,736 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 21:58:31,737 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 21:58:31,737 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 21:58:31,745 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 21:58:31,746 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:32,116 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-26 21:58:32,117 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-26 21:58:32,280 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:32,312 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:32,313 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:32,313 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:32,321 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:32,322 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:32,673 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-26 21:58:32,674 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-26 21:58:32,837 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:32,868 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:32,869 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:32,869 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:32,878 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:32,879 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:33,244 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-26 21:58:33,245 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-26 21:58:33,413 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:33,449 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:33,450 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:33,450 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:33,458 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:33,458 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:33,818 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-26 21:58:33,819 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-26 21:58:34,080 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 21:58:34,139 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 21:58:34,140 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 21:58:34,140 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 21:58:34,150 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 21:58:34,152 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 21:58:34,922 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 21:58:34,924 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-26 21:58:35,082 INFO Regime phase HTF dataset build: 7.0s (103290 samples)
2026-04-26 21:58:35,083 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=38114 dropped=65176 classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728})
2026-04-26 21:58:35,092 INFO RegimeClassifier[mode=htf_bias]: 38114 samples, classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728}, device=cuda
2026-04-26 21:58:35,093 INFO RegimeClassifier: sample weights — mean=0.708  ambiguous(<0.4)=0.0%
2026-04-26 21:58:35,361 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-04-26 21:58:35,362 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 21:58:39,827 INFO Regime epoch  1/50 — tr=0.7499 va=1.9797 acc=0.289 per_class={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.409, 'BIAS_NEUTRAL': 0.828}
2026-04-26 21:58:39,894 INFO Regime epoch  2/50 — tr=0.7449 va=1.9789 acc=0.338
2026-04-26 21:58:39,958 INFO Regime epoch  3/50 — tr=0.7375 va=1.9629 acc=0.374
2026-04-26 21:58:40,025 INFO Regime epoch  4/50 — tr=0.7242 va=1.9204 acc=0.453
2026-04-26 21:58:40,099 INFO Regime epoch  5/50 — tr=0.7008 va=1.8469 acc=0.547 per_class={'BIAS_UP': 0.312, 'BIAS_DOWN': 0.824, 'BIAS_NEUTRAL': 0.645}
2026-04-26 21:58:40,170 INFO Regime epoch  6/50 — tr=0.6774 va=1.7544 acc=0.649
2026-04-26 21:58:40,238 INFO Regime epoch  7/50 — tr=0.6492 va=1.6472 acc=0.742
2026-04-26 21:58:40,303 INFO Regime epoch  8/50 — tr=0.6207 va=1.5378 acc=0.824
2026-04-26 21:58:40,373 INFO Regime epoch  9/50 — tr=0.5934 va=1.4505 acc=0.879
2026-04-26 21:58:40,447 INFO Regime epoch 10/50 — tr=0.5733 va=1.3906 acc=0.915 per_class={'BIAS_UP': 0.929, 'BIAS_DOWN': 0.967, 'BIAS_NEUTRAL': 0.781}
2026-04-26 21:58:40,518 INFO Regime epoch 11/50 — tr=0.5551 va=1.3444 acc=0.927
2026-04-26 21:58:40,581 INFO Regime epoch 12/50 — tr=0.5426 va=1.3045 acc=0.936
2026-04-26 21:58:40,649 INFO Regime epoch 13/50 — tr=0.5312 va=1.2695 acc=0.944
2026-04-26 21:58:40,715 INFO Regime epoch 14/50 — tr=0.5237 va=1.2500 acc=0.946
2026-04-26 21:58:40,785 INFO Regime epoch 15/50 — tr=0.5166 va=1.2312 acc=0.948 per_class={'BIAS_UP': 0.98, 'BIAS_DOWN': 0.998, 'BIAS_NEUTRAL': 0.77}
2026-04-26 21:58:40,849 INFO Regime epoch 16/50 — tr=0.5122 va=1.2106 acc=0.953
2026-04-26 21:58:40,913 INFO Regime epoch 17/50 — tr=0.5082 va=1.1957 acc=0.956
2026-04-26 21:58:40,978 INFO Regime epoch 18/50 — tr=0.5052 va=1.1842 acc=0.957
2026-04-26 21:58:41,042 INFO Regime epoch 19/50 — tr=0.5025 va=1.1740 acc=0.959
2026-04-26 21:58:41,115 INFO Regime epoch 20/50 — tr=0.5002 va=1.1627 acc=0.961 per_class={'BIAS_UP': 0.992, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.804}
2026-04-26 21:58:41,182 INFO Regime epoch 21/50 — tr=0.4977 va=1.1556 acc=0.962
2026-04-26 21:58:41,247 INFO Regime epoch 22/50 — tr=0.4961 va=1.1471 acc=0.966
2026-04-26 21:58:41,311 INFO Regime epoch 23/50 — tr=0.4947 va=1.1396 acc=0.967
2026-04-26 21:58:41,375 INFO Regime epoch 24/50 — tr=0.4929 va=1.1311 acc=0.969
2026-04-26 21:58:41,457 INFO Regime epoch 25/50 — tr=0.4916 va=1.1233 acc=0.970 per_class={'BIAS_UP': 0.998, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.84}
2026-04-26 21:58:41,529 INFO Regime epoch 26/50 — tr=0.4905 va=1.1197 acc=0.970
2026-04-26 21:58:41,600 INFO Regime epoch 27/50 — tr=0.4896 va=1.1171 acc=0.970
2026-04-26 21:58:41,672 INFO Regime epoch 28/50 — tr=0.4886 va=1.1133 acc=0.971
2026-04-26 21:58:41,741 INFO Regime epoch 29/50 — tr=0.4880 va=1.1130 acc=0.973
2026-04-26 21:58:41,810 INFO Regime epoch 30/50 — tr=0.4866 va=1.1076 acc=0.974 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.853}
2026-04-26 21:58:41,875 INFO Regime epoch 31/50 — tr=0.4860 va=1.1022 acc=0.976
2026-04-26 21:58:41,940 INFO Regime epoch 32/50 — tr=0.4851 va=1.0987 acc=0.975
2026-04-26 21:58:42,004 INFO Regime epoch 33/50 — tr=0.4849 va=1.0965 acc=0.976
2026-04-26 21:58:42,069 INFO Regime epoch 34/50 — tr=0.4846 va=1.0944 acc=0.977
2026-04-26 21:58:42,144 INFO Regime epoch 35/50 — tr=0.4837 va=1.0938 acc=0.976 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.869}
2026-04-26 21:58:42,210 INFO Regime epoch 36/50 — tr=0.4835 va=1.0906 acc=0.977
2026-04-26 21:58:42,276 INFO Regime epoch 37/50 — tr=0.4832 va=1.0888 acc=0.977
2026-04-26 21:58:42,342 INFO Regime epoch 38/50 — tr=0.4829 va=1.0879 acc=0.977
2026-04-26 21:58:42,408 INFO Regime epoch 39/50 — tr=0.4831 va=1.0883 acc=0.977
2026-04-26 21:58:42,476 INFO Regime epoch 40/50 — tr=0.4826 va=1.0876 acc=0.977 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.874}
2026-04-26 21:58:42,541 INFO Regime epoch 41/50 — tr=0.4828 va=1.0861 acc=0.978
2026-04-26 21:58:42,605 INFO Regime epoch 42/50 — tr=0.4824 va=1.0868 acc=0.978
2026-04-26 21:58:42,671 INFO Regime epoch 43/50 — tr=0.4824 va=1.0856 acc=0.978
2026-04-26 21:58:42,739 INFO Regime epoch 44/50 — tr=0.4819 va=1.0844 acc=0.978
2026-04-26 21:58:42,814 INFO Regime epoch 45/50 — tr=0.4817 va=1.0846 acc=0.978 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.877}
2026-04-26 21:58:42,884 INFO Regime epoch 46/50 — tr=0.4820 va=1.0863 acc=0.978
2026-04-26 21:58:42,953 INFO Regime epoch 47/50 — tr=0.4821 va=1.0851 acc=0.978
2026-04-26 21:58:43,018 INFO Regime epoch 48/50 — tr=0.4819 va=1.0815 acc=0.978
2026-04-26 21:58:43,082 INFO Regime epoch 49/50 — tr=0.4823 va=1.0829 acc=0.978
2026-04-26 21:58:43,157 INFO Regime epoch 50/50 — tr=0.4822 va=1.0850 acc=0.978 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.877}
2026-04-26 21:58:43,167 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 21:58:43,167 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 21:58:43,168 INFO Regime phase HTF train: 8.1s
2026-04-26 21:58:43,294 INFO Regime HTF complete: acc=0.978, n=103290 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.878}
2026-04-26 21:58:43,295 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 21:58:43,438 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 21:58:43,447 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 11.41578947368421, 'BIAS_DOWN': 10.635761589403973, 'BIAS_NEUTRAL': 17.960468521229867}
2026-04-26 21:58:43,451 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 4338, 'mean': 0.00011110466582543088, 'mean_over_std': 0.027632581210502927}, 'BIAS_DOWN': {'n': 3212, 'mean': 6.063430222660252e-05, 'mean_over_std': 0.013898489590377157}, 'BIAS_NEUTRAL': {'n': 12266, 'mean': 1.2151554902814046e-05, 'mean_over_std': 0.0031957000513162357}}
2026-04-26 21:58:43,451 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 3651, 'mean': 0.0002852242108836959, 'mean_over_std': 0.07015017232491755}, 'BIAS_DOWN': {'n': 2644, 'mean': -0.00014900749618981248, 'mean_over_std': -0.03333037041298237}, 'BIAS_NEUTRAL': {'n': 1376, 'mean': 9.595797941249082e-05, 'mean_over_std': 0.02492411099521803}}
2026-04-26 21:58:43,452 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-26 21:58:43,454 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:43,456 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:43,457 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:43,459 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:43,460 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:43,462 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:43,463 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:43,465 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:43,466 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:43,468 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:43,471 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 21:58:43,481 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:43,486 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:43,486 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:43,487 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:43,487 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:43,489 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:44,079 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-26 21:58:44,082 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-26 21:58:44,215 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:44,218 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:44,219 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:44,219 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:44,219 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:44,221 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:44,788 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-26 21:58:44,791 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-26 21:58:44,927 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:44,929 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:44,930 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:44,930 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:44,931 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:44,933 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:45,484 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-26 21:58:45,487 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-26 21:58:45,621 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:45,623 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:45,624 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:45,624 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:45,625 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:45,627 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:46,178 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-26 21:58:46,181 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-26 21:58:46,314 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:46,316 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:46,317 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:46,318 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:46,318 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:46,320 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:46,914 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-26 21:58:46,917 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-26 21:58:47,050 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:47,053 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:47,053 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:47,054 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:47,054 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:47,056 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:47,633 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-26 21:58:47,636 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-26 21:58:47,765 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 21:58:47,767 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 21:58:47,768 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 21:58:47,768 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 21:58:47,768 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 21:58:47,770 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:48,315 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-26 21:58:48,318 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-26 21:58:48,453 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:48,455 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:48,456 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:48,456 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:48,457 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:48,459 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:49,019 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-26 21:58:49,021 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-26 21:58:49,160 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:49,162 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:49,163 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:49,164 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:49,164 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:49,166 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:49,741 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-26 21:58:49,744 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-26 21:58:49,878 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:49,880 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:49,881 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:49,881 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:49,882 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:58:49,884 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:58:50,437 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-26 21:58:50,439 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-26 21:58:50,580 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 21:58:50,584 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 21:58:50,585 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 21:58:50,585 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 21:58:50,586 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 21:58:50,589 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 21:58:51,816 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 21:58:51,821 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-26 21:58:52,107 INFO Regime phase LTF dataset build: 8.7s (401471 samples)
2026-04-26 21:58:52,110 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=299185 dropped=102286 classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127})
2026-04-26 21:58:52,168 INFO RegimeClassifier[mode=ltf_behaviour]: 299185 samples, classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127}, device=cuda
2026-04-26 21:58:52,169 INFO RegimeClassifier: sample weights — mean=0.693  ambiguous(<0.4)=0.0%
2026-04-26 21:58:52,171 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-04-26 21:58:52,171 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 21:58:52,747 INFO Regime epoch  1/50 — tr=0.9285 va=2.1235 acc=0.252 per_class={'TRENDING': 0.328, 'RANGING': 0.579, 'CONSOLIDATING': 0.115, 'VOLATILE': 0.002}
2026-04-26 21:58:53,226 INFO Regime epoch  2/50 — tr=0.8945 va=1.9653 acc=0.359
2026-04-26 21:58:53,717 INFO Regime epoch  3/50 — tr=0.8412 va=1.8135 acc=0.560
2026-04-26 21:58:54,214 INFO Regime epoch  4/50 — tr=0.7838 va=1.6800 acc=0.620
2026-04-26 21:58:54,744 INFO Regime epoch  5/50 — tr=0.7408 va=1.5809 acc=0.643 per_class={'TRENDING': 0.508, 'RANGING': 0.549, 'CONSOLIDATING': 0.69, 'VOLATILE': 0.925}
2026-04-26 21:58:55,227 INFO Regime epoch  6/50 — tr=0.7143 va=1.5109 acc=0.660
2026-04-26 21:58:55,700 INFO Regime epoch  7/50 — tr=0.6968 va=1.4689 acc=0.676
2026-04-26 21:58:56,170 INFO Regime epoch  8/50 — tr=0.6845 va=1.4440 acc=0.689
2026-04-26 21:58:56,677 INFO Regime epoch  9/50 — tr=0.6756 va=1.4301 acc=0.702
2026-04-26 21:58:57,225 INFO Regime epoch 10/50 — tr=0.6683 va=1.4084 acc=0.715 per_class={'TRENDING': 0.595, 'RANGING': 0.717, 'CONSOLIDATING': 0.719, 'VOLATILE': 0.94}
2026-04-26 21:58:57,706 INFO Regime epoch 11/50 — tr=0.6639 va=1.3956 acc=0.726
2026-04-26 21:58:58,191 INFO Regime epoch 12/50 — tr=0.6590 va=1.3813 acc=0.739
2026-04-26 21:58:58,669 INFO Regime epoch 13/50 — tr=0.6556 va=1.3641 acc=0.742
2026-04-26 21:58:59,144 INFO Regime epoch 14/50 — tr=0.6528 va=1.3561 acc=0.755
2026-04-26 21:58:59,672 INFO Regime epoch 15/50 — tr=0.6500 va=1.3424 acc=0.761 per_class={'TRENDING': 0.691, 'RANGING': 0.736, 'CONSOLIDATING': 0.739, 'VOLATILE': 0.922}
2026-04-26 21:59:00,188 INFO Regime epoch 16/50 — tr=0.6475 va=1.3369 acc=0.767
2026-04-26 21:59:00,673 INFO Regime epoch 17/50 — tr=0.6453 va=1.3298 acc=0.774
2026-04-26 21:59:01,179 INFO Regime epoch 18/50 — tr=0.6437 va=1.3183 acc=0.778
2026-04-26 21:59:01,665 INFO Regime epoch 19/50 — tr=0.6420 va=1.3127 acc=0.782
2026-04-26 21:59:02,194 INFO Regime epoch 20/50 — tr=0.6407 va=1.3039 acc=0.784 per_class={'TRENDING': 0.735, 'RANGING': 0.754, 'CONSOLIDATING': 0.744, 'VOLATILE': 0.919}
2026-04-26 21:59:02,678 INFO Regime epoch 21/50 — tr=0.6394 va=1.3009 acc=0.790
2026-04-26 21:59:03,171 INFO Regime epoch 22/50 — tr=0.6381 va=1.2916 acc=0.790
2026-04-26 21:59:03,646 INFO Regime epoch 23/50 — tr=0.6371 va=1.2887 acc=0.793
2026-04-26 21:59:04,136 INFO Regime epoch 24/50 — tr=0.6361 va=1.2877 acc=0.799
2026-04-26 21:59:04,672 INFO Regime epoch 25/50 — tr=0.6351 va=1.2839 acc=0.800 per_class={'TRENDING': 0.765, 'RANGING': 0.761, 'CONSOLIDATING': 0.764, 'VOLATILE': 0.911}
2026-04-26 21:59:05,199 INFO Regime epoch 26/50 — tr=0.6342 va=1.2771 acc=0.802
2026-04-26 21:59:05,692 INFO Regime epoch 27/50 — tr=0.6333 va=1.2687 acc=0.802
2026-04-26 21:59:06,167 INFO Regime epoch 28/50 — tr=0.6324 va=1.2703 acc=0.805
2026-04-26 21:59:06,654 INFO Regime epoch 29/50 — tr=0.6321 va=1.2704 acc=0.808
2026-04-26 21:59:07,221 INFO Regime epoch 30/50 — tr=0.6317 va=1.2693 acc=0.809 per_class={'TRENDING': 0.776, 'RANGING': 0.77, 'CONSOLIDATING': 0.782, 'VOLATILE': 0.911}
2026-04-26 21:59:07,714 INFO Regime epoch 31/50 — tr=0.6306 va=1.2642 acc=0.811
2026-04-26 21:59:08,211 INFO Regime epoch 32/50 — tr=0.6302 va=1.2625 acc=0.813
2026-04-26 21:59:08,695 INFO Regime epoch 33/50 — tr=0.6297 va=1.2627 acc=0.814
2026-04-26 21:59:09,182 INFO Regime epoch 34/50 — tr=0.6295 va=1.2586 acc=0.813
2026-04-26 21:59:09,699 INFO Regime epoch 35/50 — tr=0.6290 va=1.2551 acc=0.814 per_class={'TRENDING': 0.783, 'RANGING': 0.771, 'CONSOLIDATING': 0.807, 'VOLATILE': 0.903}
2026-04-26 21:59:10,208 INFO Regime epoch 36/50 — tr=0.6288 va=1.2586 acc=0.816
2026-04-26 21:59:10,685 INFO Regime epoch 37/50 — tr=0.6280 va=1.2559 acc=0.817
2026-04-26 21:59:11,167 INFO Regime epoch 38/50 — tr=0.6281 va=1.2539 acc=0.817
2026-04-26 21:59:11,643 INFO Regime epoch 39/50 — tr=0.6278 va=1.2525 acc=0.817
2026-04-26 21:59:12,169 INFO Regime epoch 40/50 — tr=0.6278 va=1.2569 acc=0.820 per_class={'TRENDING': 0.795, 'RANGING': 0.767, 'CONSOLIDATING': 0.818, 'VOLATILE': 0.901}
2026-04-26 21:59:12,641 INFO Regime epoch 41/50 — tr=0.6275 va=1.2536 acc=0.820
2026-04-26 21:59:13,118 INFO Regime epoch 42/50 — tr=0.6271 va=1.2499 acc=0.819
2026-04-26 21:59:13,593 INFO Regime epoch 43/50 — tr=0.6276 va=1.2528 acc=0.819
2026-04-26 21:59:14,066 INFO Regime epoch 44/50 — tr=0.6272 va=1.2519 acc=0.818
2026-04-26 21:59:14,577 INFO Regime epoch 45/50 — tr=0.6271 va=1.2511 acc=0.820 per_class={'TRENDING': 0.793, 'RANGING': 0.774, 'CONSOLIDATING': 0.81, 'VOLATILE': 0.905}
2026-04-26 21:59:15,051 INFO Regime epoch 46/50 — tr=0.6269 va=1.2508 acc=0.820
2026-04-26 21:59:15,536 INFO Regime epoch 47/50 — tr=0.6271 va=1.2548 acc=0.822
2026-04-26 21:59:16,028 INFO Regime epoch 48/50 — tr=0.6272 va=1.2511 acc=0.820
2026-04-26 21:59:16,541 INFO Regime epoch 49/50 — tr=0.6271 va=1.2532 acc=0.819
2026-04-26 21:59:17,090 INFO Regime epoch 50/50 — tr=0.6270 va=1.2484 acc=0.819 per_class={'TRENDING': 0.789, 'RANGING': 0.777, 'CONSOLIDATING': 0.809, 'VOLATILE': 0.907}
2026-04-26 21:59:17,129 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 21:59:17,129 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 21:59:17,131 INFO Regime phase LTF train: 25.0s
2026-04-26 21:59:17,259 INFO Regime LTF complete: acc=0.819, n=401471 per_class={'TRENDING': 0.789, 'RANGING': 0.777, 'CONSOLIDATING': 0.809, 'VOLATILE': 0.907}
2026-04-26 21:59:17,262 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 21:59:17,715 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 21:59:17,718 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 9.355291913830783, 'RANGING': 6.096303199751476, 'CONSOLIDATING': 5.598885793871866, 'VOLATILE': 6.771351107094442}
2026-04-26 21:59:17,726 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 29965, 'mean': 2.149062259083323e-05, 'mean_over_std': 0.011155601434349563}, 'RANGING': {'n': 19623, 'mean': -3.254493608872846e-06, 'mean_over_std': -0.001761415548292494}, 'CONSOLIDATING': {'n': 10050, 'mean': -3.039756158146958e-06, 'mean_over_std': -0.001792488091618151}, 'VOLATILE': {'n': 14985, 'mean': 1.8102764458593193e-05, 'mean_over_std': 0.006974696291378742}}
2026-04-26 21:59:17,726 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 26145, 'mean': 2.874435605138053e-05, 'mean_over_std': 0.01595450632691917}, 'RANGING': {'n': 7858, 'mean': 1.7829776785521727e-05, 'mean_over_std': 0.011851561804467463}, 'CONSOLIDATING': {'n': 8543, 'mean': 7.845231961250924e-06, 'mean_over_std': 0.004915839521549481}, 'VOLATILE': {'n': 13636, 'mean': 5.480114267306668e-06, 'mean_over_std': 0.0020378119917389375}}
2026-04-26 21:59:17,726 INFO Regime retrain total: 208.7s (504761 samples)
2026-04-26 21:59:17,739 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 21:59:17,739 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 21:59:17,739 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 21:59:17,739 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-26 21:59:17,740 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-26 21:59:17,740 INFO Retrain complete. Total wall-clock: 208.7s
2026-04-26 21:59:19,953 INFO Model regime: SUCCESS
2026-04-26 21:59:19,953 INFO --- Training gru ---
2026-04-26 21:59:19,953 INFO Running retrain --model gru
2026-04-26 21:59:20,201 INFO retrain environment: KAGGLE
2026-04-26 21:59:21,740 INFO Device: CUDA (2 GPU(s))
2026-04-26 21:59:21,749 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 21:59:21,749 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 21:59:21,749 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 21:59:21,749 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 21:59:21,750 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-26 21:59:21,894 INFO NumExpr defaulting to 4 threads.
2026-04-26 21:59:22,084 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 21:59:22,084 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 21:59:22,084 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 21:59:22,085 INFO GRU phase macro_correlations: 0.0s
2026-04-26 21:59:22,085 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-26 21:59:22,085 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260426_215922
2026-04-26 21:59:22,088 WARNING WeightsManifest: no manifest at /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json — assuming compatible (run a full retrain to generate one)
2026-04-26 21:59:22,228 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:22,247 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:22,260 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:22,266 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:22,268 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 21:59:22,268 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 21:59:22,268 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 21:59:22,268 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 21:59:22,269 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:22,344 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-26 21:59:22,346 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:22,561 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-26 21:59:22,588 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:22,852 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:22,975 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:23,066 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:23,257 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:23,276 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:23,288 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:23,294 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:23,295 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:23,370 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-26 21:59:23,372 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:23,603 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-26 21:59:23,620 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:23,873 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:23,996 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:24,093 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:24,281 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:24,301 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:24,315 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:24,322 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:24,323 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:24,402 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-26 21:59:24,404 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:24,633 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-26 21:59:24,648 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:24,912 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:25,040 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:25,133 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:25,311 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:25,329 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:25,343 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:25,349 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:25,350 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:25,421 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-26 21:59:25,422 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:25,634 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-26 21:59:25,657 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:25,920 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:26,046 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:26,139 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:26,317 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:26,337 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:26,352 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:26,359 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:26,360 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:26,455 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-26 21:59:26,457 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:26,680 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-26 21:59:26,696 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:26,974 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:27,097 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:27,186 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:27,362 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:27,380 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:27,394 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:27,400 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:27,401 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:27,472 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-26 21:59:27,474 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:27,698 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-26 21:59:27,712 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:27,970 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:28,095 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:28,187 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:28,347 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 21:59:28,364 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 21:59:28,377 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 21:59:28,383 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 21:59:28,384 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:28,457 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-26 21:59:28,458 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:28,672 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-26 21:59:28,684 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:28,935 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:29,057 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:29,147 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:29,317 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:29,334 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:29,347 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:29,354 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:29,355 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:29,426 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-26 21:59:29,428 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:29,642 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-26 21:59:29,660 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:29,913 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:30,036 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:30,129 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:30,299 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:30,317 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:30,330 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:30,336 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:30,337 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:30,407 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-26 21:59:30,409 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:30,623 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-26 21:59:30,638 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:30,891 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:31,019 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:31,116 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:31,295 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:31,314 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:31,328 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:31,334 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 21:59:31,335 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:31,409 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-26 21:59:31,410 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:31,633 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-26 21:59:31,647 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:31,905 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:32,030 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:32,123 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 21:59:32,399 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 21:59:32,424 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 21:59:32,439 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 21:59:32,448 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 21:59:32,449 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 21:59:32,600 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 21:59:32,603 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 21:59:33,070 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 21:59:33,114 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-26 21:59:33,630 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-26 21:59:33,823 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 21:59:33,949 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 21:59:34,064 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-26 21:59:34,319 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-04-26 21:59:34,319 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-04-26 21:59:34,319 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-26 21:59:34,319 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-26 22:00:20,154 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-26 22:00:20,154 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-26 22:00:21,470 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-26 22:00:25,551 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=5)
2026-04-26 22:00:38,971 INFO train_multi TF=ALL epoch 1/50 train=0.8891 val=0.8820
2026-04-26 22:00:38,979 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:00:38,979 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:00:38,979 INFO train_multi TF=ALL: new best val=0.8820 — saved
2026-04-26 22:00:50,437 INFO train_multi TF=ALL epoch 2/50 train=0.8661 val=0.8297
2026-04-26 22:00:50,441 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:00:50,441 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:00:50,442 INFO train_multi TF=ALL: new best val=0.8297 — saved
2026-04-26 22:01:01,886 INFO train_multi TF=ALL epoch 3/50 train=0.7342 val=0.6882
2026-04-26 22:01:01,890 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:01:01,890 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:01:01,890 INFO train_multi TF=ALL: new best val=0.6882 — saved
2026-04-26 22:01:13,306 INFO train_multi TF=ALL epoch 4/50 train=0.6909 val=0.6878
2026-04-26 22:01:13,310 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:01:13,310 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:01:13,310 INFO train_multi TF=ALL: new best val=0.6878 — saved
2026-04-26 22:01:24,791 INFO train_multi TF=ALL epoch 5/50 train=0.6899 val=0.6880
2026-04-26 22:01:36,265 INFO train_multi TF=ALL epoch 6/50 train=0.6895 val=0.6886
2026-04-26 22:01:47,790 INFO train_multi TF=ALL epoch 7/50 train=0.6893 val=0.6885
2026-04-26 22:01:59,412 INFO train_multi TF=ALL epoch 8/50 train=0.6891 val=0.6879
2026-04-26 22:02:11,118 INFO train_multi TF=ALL epoch 9/50 train=0.6890 val=0.6880
2026-04-26 22:02:11,118 INFO train_multi TF=ALL early stop at epoch 9
2026-04-26 22:02:11,263 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-26 22:02:11,264 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-26 22:02:11,264 INFO Retrain complete. Total wall-clock: 169.5s
2026-04-26 22:02:13,206 INFO Model gru: SUCCESS
2026-04-26 22:02:13,206 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:02:13,206 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 22:02:13,206 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 22:02:13,206 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-04-26 22:02:13,206 INFO   [DEFERRED] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip (expected after Round 1)
2026-04-26 22:02:13,206 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-04-26 22:02:13,207 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer', 'rl_ppo']
2026-04-26 22:02:13,207 INFO Saved 5 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  START Round 1 - Backtest (val)
2026-04-26 22:02:13,817 INFO === STEP 6: BACKTEST (round1) ===
2026-04-26 22:02:13,818 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-04-26 22:02:13,818 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-26 22:02:13,818 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
2026-04-26 22:02:16,189 WARNING QualityScorer unavailable (weights missing or load failed)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260426_220215.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)                0   0.0%   0.00    0.0%  0.0%  0.0%   0.0%    0.00
  gate_diagnostics: bars=468696 no_signal=150801 quality_block=0 session_skip=317895 density=0 pm_reject=0 daily_skip=0 cooldown=0 daily_halt_events=0 enforce_daily_halt=False
  no_signal_reasons: weak_gru_direction=150801

Calibration Summary:
  all          [N/A] No outcome data yet
2026-04-26 22:04:28,244 INFO Round 1 backtest — 0 trades | avg WR=0.0% | avg PF=0.00 | avg Sharpe=0.00
2026-04-26 22:04:28,244 INFO   ml_trader: 0 trades | WR=0.0% | PF=0.00 | Return=0.0% | DD=0.0% | Sharpe=0.00
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 0
ERROR  trade_log is empty.
2026-04-26 22:04:28,461 WARNING Round 1: trade_log is empty — nothing to journal
2026-04-26 22:04:28,461 WARNING Round 1: no trades to journal

======================================================================
  BACKTEST COMPLETE  (round 1 / window=round1)
======================================================================
  Round     Trades       WR      PF   Sharpe
  ------------------------------------------
  Round 1          0      0.0%    0.000     0.000

  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 0 entries

=== Round 1 → Retrain Quality + RL ===
  START Round 1 - Quality+RL retrain
2026-04-26 22:04:28,694 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-26 22:04:28,694 WARNING Journal missing or empty at /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/trade_journal_detailed.jsonl — backtest produced no trades yet. Skipping Quality+RL training (will train after first successful backtest).
  DONE  Round 1 - Quality+RL retrain

=== Round 2: BLIND backtest on test window (unseen 2yr) ===
  START Round 2 - Blind backtest (test)
2026-04-26 22:04:29,190 INFO === STEP 6: BACKTEST (round2) ===
2026-04-26 22:04:29,190 INFO BT_WINDOW=round2 — BLIND backtest: 2023-08-07 → 2025-08-05 (test set)
2026-04-26 22:04:29,190 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-26 22:04:29,191 INFO Round 2 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-26 22:04:31,577 WARNING QualityScorer unavailable (weights missing or load failed)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260426_220431.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)                0   0.0%   0.00    0.0%  0.0%  0.0%   0.0%    0.00
  gate_diagnostics: bars=482221 no_signal=120986 quality_block=0 session_skip=361235 density=0 pm_reject=0 daily_skip=0 cooldown=0 daily_halt_events=0 enforce_daily_halt=False
  no_signal_reasons: weak_gru_direction=120986

Calibration Summary:
  all          [N/A] No outcome data yet
2026-04-26 22:06:43,854 INFO Round 2 backtest — 0 trades | avg WR=0.0% | avg PF=0.00 | avg Sharpe=0.00
2026-04-26 22:06:43,854 INFO   ml_trader: 0 trades | WR=0.0% | PF=0.00 | Return=0.0% | DD=0.0% | Sharpe=0.00
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 0
ERROR  trade_log is empty.
2026-04-26 22:06:44,071 WARNING Round 2: trade_log is empty — nothing to journal
2026-04-26 22:06:44,071 WARNING Round 2: no trades to journal

======================================================================
  BACKTEST COMPLETE  (round 2 / window=round2)
======================================================================
  Round     Trades       WR      PF   Sharpe
  ------------------------------------------
  Round 2          0      0.0%    0.000     0.000

  DONE  Round 2 - Blind backtest (test)
  Saved Round 2 result → round2_summary.json
  Journal after Round 2: 0 entries

=== Round 2 → Retrain Quality + RL (Round 1+2 journal) ===
  START Round 2 - Quality+RL retrain
2026-04-26 22:06:44,295 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-26 22:06:44,295 WARNING Journal missing or empty at /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/trade_journal_detailed.jsonl — backtest produced no trades yet. Skipping Quality+RL training (will train after first successful backtest).
2026-04-26 22:06:44,483 INFO retrain environment: KAGGLE
  DONE  Round 2 - Quality+RL retrain

=== Round 3: Incremental retrain ALL models on full data ===
  START Retrain gru [full-data retrain]
2026-04-26 22:06:46,088 INFO Device: CUDA (2 GPU(s))
2026-04-26 22:06:46,100 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 22:06:46,100 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 22:06:46,100 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 22:06:46,100 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 22:06:46,101 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-26 22:06:46,249 INFO NumExpr defaulting to 4 threads.
2026-04-26 22:06:46,462 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 22:06:46,462 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 22:06:46,462 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 22:06:46,713 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-26 22:06:46,713 INFO GRU phase macro_correlations: 0.0s
2026-04-26 22:06:46,713 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-26 22:06:46,715 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260426_220646
2026-04-26 22:06:46,721 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-04-26 22:06:46,872 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:46,891 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:46,905 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:46,912 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:46,913 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 22:06:46,913 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 22:06:46,913 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 22:06:46,914 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 22:06:46,915 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:46,994 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-26 22:06:46,996 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:47,227 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-26 22:06:47,256 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:47,519 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:47,649 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:47,750 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:47,950 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:47,968 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:47,982 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:47,988 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:47,989 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:48,065 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-26 22:06:48,066 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:48,295 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-26 22:06:48,311 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:48,578 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:48,703 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:48,795 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:48,977 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:48,996 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:49,010 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:49,017 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:49,018 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:49,091 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-26 22:06:49,093 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:49,318 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-26 22:06:49,336 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:49,603 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:49,725 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:49,822 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:50,009 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:50,028 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:50,042 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:50,049 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:50,050 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:50,128 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-26 22:06:50,130 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:50,347 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-26 22:06:50,370 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:50,628 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:50,760 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:50,861 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:51,055 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:51,075 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:51,088 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:51,097 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:51,097 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:51,171 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-26 22:06:51,173 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:51,396 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-26 22:06:51,411 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:51,682 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:51,812 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:51,909 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:52,088 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:52,106 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:52,120 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:52,126 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:52,127 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:52,200 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-26 22:06:52,201 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:52,425 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-26 22:06:52,440 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:52,702 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:52,828 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:52,923 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:53,088 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 22:06:53,104 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 22:06:53,117 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 22:06:53,123 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 22:06:53,124 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:53,198 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-26 22:06:53,199 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:53,420 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-26 22:06:53,432 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:53,697 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:53,825 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:53,929 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:54,100 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:54,117 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:54,131 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:54,137 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:54,138 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:54,211 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-26 22:06:54,212 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:54,430 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-26 22:06:54,450 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:54,708 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:54,837 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:54,937 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:55,114 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:55,131 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:55,145 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:55,152 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:55,153 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:55,224 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-26 22:06:55,225 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:55,451 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-26 22:06:55,466 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:55,736 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:55,859 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:55,950 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:56,131 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:56,150 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:56,164 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:56,171 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:06:56,172 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:56,242 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-26 22:06:56,244 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:56,487 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-26 22:06:56,502 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:56,785 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:56,907 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:57,010 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:06:57,294 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 22:06:57,317 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 22:06:57,332 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 22:06:57,342 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 22:06:57,343 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 22:06:57,489 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 22:06:57,492 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 22:06:57,970 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 22:06:58,015 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-26 22:06:58,515 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-26 22:06:58,710 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 22:06:58,840 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 22:06:58,950 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-26 22:06:58,950 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-26 22:06:58,950 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-26 22:07:45,999 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-26 22:07:45,999 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-26 22:07:47,334 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-26 22:07:51,400 INFO train_multi TF=ALL: warm-start detected — using CosineAnnealingLR (lr=3e-05, patience=12)
2026-04-26 22:08:04,674 INFO train_multi TF=ALL epoch 1/50 train=0.6896 val=0.6885
2026-04-26 22:08:04,679 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:08:04,679 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:08:04,679 INFO train_multi TF=ALL: new best val=0.6885 — saved
2026-04-26 22:08:16,322 INFO train_multi TF=ALL epoch 2/50 train=0.6893 val=0.6888
2026-04-26 22:08:28,061 INFO train_multi TF=ALL epoch 3/50 train=0.6891 val=0.6885
2026-04-26 22:08:28,066 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:08:28,066 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:08:28,066 INFO train_multi TF=ALL: new best val=0.6885 — saved
2026-04-26 22:08:39,788 INFO train_multi TF=ALL epoch 4/50 train=0.6888 val=0.6885
2026-04-26 22:08:39,792 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:08:39,792 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:08:39,792 INFO train_multi TF=ALL: new best val=0.6885 — saved
2026-04-26 22:08:51,432 INFO train_multi TF=ALL epoch 5/50 train=0.6884 val=0.6868
2026-04-26 22:08:51,436 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:08:51,436 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:08:51,436 INFO train_multi TF=ALL: new best val=0.6868 — saved
2026-04-26 22:09:03,153 INFO train_multi TF=ALL epoch 6/50 train=0.6874 val=0.6857
2026-04-26 22:09:03,158 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:09:03,158 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:09:03,158 INFO train_multi TF=ALL: new best val=0.6857 — saved
2026-04-26 22:09:14,767 INFO train_multi TF=ALL epoch 7/50 train=0.6858 val=0.6842
2026-04-26 22:09:14,772 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:09:14,772 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:09:14,772 INFO train_multi TF=ALL: new best val=0.6842 — saved
2026-04-26 22:09:26,384 INFO train_multi TF=ALL epoch 8/50 train=0.6834 val=0.6805
2026-04-26 22:09:26,390 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:09:26,390 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:09:26,390 INFO train_multi TF=ALL: new best val=0.6805 — saved
2026-04-26 22:09:38,097 INFO train_multi TF=ALL epoch 9/50 train=0.6802 val=0.6774
2026-04-26 22:09:38,101 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:09:38,102 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:09:38,102 INFO train_multi TF=ALL: new best val=0.6774 — saved
2026-04-26 22:09:49,696 INFO train_multi TF=ALL epoch 10/50 train=0.6772 val=0.6742
2026-04-26 22:09:49,700 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:09:49,700 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:09:49,700 INFO train_multi TF=ALL: new best val=0.6742 — saved
2026-04-26 22:10:01,293 INFO train_multi TF=ALL epoch 11/50 train=0.6742 val=0.6717
2026-04-26 22:10:01,297 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:10:01,297 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:10:01,297 INFO train_multi TF=ALL: new best val=0.6717 — saved
2026-04-26 22:10:12,996 INFO train_multi TF=ALL epoch 12/50 train=0.6714 val=0.6679
2026-04-26 22:10:13,001 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:10:13,001 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:10:13,001 INFO train_multi TF=ALL: new best val=0.6679 — saved
2026-04-26 22:10:24,615 INFO train_multi TF=ALL epoch 13/50 train=0.6683 val=0.6653
2026-04-26 22:10:24,620 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:10:24,620 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:10:24,620 INFO train_multi TF=ALL: new best val=0.6653 — saved
2026-04-26 22:10:36,444 INFO train_multi TF=ALL epoch 14/50 train=0.6654 val=0.6620
2026-04-26 22:10:36,448 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:10:36,448 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:10:36,449 INFO train_multi TF=ALL: new best val=0.6620 — saved
2026-04-26 22:10:48,166 INFO train_multi TF=ALL epoch 15/50 train=0.6626 val=0.6594
2026-04-26 22:10:48,170 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:10:48,170 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:10:48,170 INFO train_multi TF=ALL: new best val=0.6594 — saved
2026-04-26 22:10:59,880 INFO train_multi TF=ALL epoch 16/50 train=0.6600 val=0.6557
2026-04-26 22:10:59,885 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:10:59,885 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:10:59,885 INFO train_multi TF=ALL: new best val=0.6557 — saved
2026-04-26 22:11:11,552 INFO train_multi TF=ALL epoch 17/50 train=0.6573 val=0.6525
2026-04-26 22:11:11,557 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:11:11,557 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:11:11,557 INFO train_multi TF=ALL: new best val=0.6525 — saved
2026-04-26 22:11:23,246 INFO train_multi TF=ALL epoch 18/50 train=0.6550 val=0.6502
2026-04-26 22:11:23,250 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:11:23,250 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:11:23,250 INFO train_multi TF=ALL: new best val=0.6502 — saved
2026-04-26 22:11:34,854 INFO train_multi TF=ALL epoch 19/50 train=0.6529 val=0.6473
2026-04-26 22:11:34,858 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:11:34,858 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:11:34,858 INFO train_multi TF=ALL: new best val=0.6473 — saved
2026-04-26 22:11:46,557 INFO train_multi TF=ALL epoch 20/50 train=0.6506 val=0.6447
2026-04-26 22:11:46,562 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:11:46,562 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:11:46,562 INFO train_multi TF=ALL: new best val=0.6447 — saved
2026-04-26 22:11:58,157 INFO train_multi TF=ALL epoch 21/50 train=0.6487 val=0.6434
2026-04-26 22:11:58,161 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:11:58,161 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:11:58,162 INFO train_multi TF=ALL: new best val=0.6434 — saved
2026-04-26 22:12:09,809 INFO train_multi TF=ALL epoch 22/50 train=0.6470 val=0.6413
2026-04-26 22:12:09,814 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:12:09,814 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:12:09,814 INFO train_multi TF=ALL: new best val=0.6413 — saved
2026-04-26 22:12:21,470 INFO train_multi TF=ALL epoch 23/50 train=0.6458 val=0.6396
2026-04-26 22:12:21,475 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:12:21,475 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:12:21,475 INFO train_multi TF=ALL: new best val=0.6396 — saved
2026-04-26 22:12:33,143 INFO train_multi TF=ALL epoch 24/50 train=0.6448 val=0.6384
2026-04-26 22:12:33,148 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:12:33,148 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:12:33,148 INFO train_multi TF=ALL: new best val=0.6384 — saved
2026-04-26 22:12:44,812 INFO train_multi TF=ALL epoch 25/50 train=0.6434 val=0.6381
2026-04-26 22:12:44,816 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:12:44,816 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:12:44,816 INFO train_multi TF=ALL: new best val=0.6381 — saved
2026-04-26 22:12:56,459 INFO train_multi TF=ALL epoch 26/50 train=0.6427 val=0.6376
2026-04-26 22:12:56,463 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:12:56,463 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:12:56,463 INFO train_multi TF=ALL: new best val=0.6376 — saved
2026-04-26 22:13:08,170 INFO train_multi TF=ALL epoch 27/50 train=0.6414 val=0.6365
2026-04-26 22:13:08,175 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:13:08,175 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:13:08,175 INFO train_multi TF=ALL: new best val=0.6365 — saved
2026-04-26 22:13:19,812 INFO train_multi TF=ALL epoch 28/50 train=0.6407 val=0.6364
2026-04-26 22:13:19,816 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:13:19,816 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:13:19,816 INFO train_multi TF=ALL: new best val=0.6364 — saved
2026-04-26 22:13:31,515 INFO train_multi TF=ALL epoch 29/50 train=0.6402 val=0.6351
2026-04-26 22:13:31,520 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:13:31,520 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:13:31,520 INFO train_multi TF=ALL: new best val=0.6351 — saved
2026-04-26 22:13:43,095 INFO train_multi TF=ALL epoch 30/50 train=0.6395 val=0.6348
2026-04-26 22:13:43,100 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:13:43,100 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:13:43,100 INFO train_multi TF=ALL: new best val=0.6348 — saved
2026-04-26 22:13:54,741 INFO train_multi TF=ALL epoch 31/50 train=0.6391 val=0.6345
2026-04-26 22:13:54,745 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:13:54,746 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:13:54,746 INFO train_multi TF=ALL: new best val=0.6345 — saved
2026-04-26 22:14:06,392 INFO train_multi TF=ALL epoch 32/50 train=0.6386 val=0.6341
2026-04-26 22:14:06,397 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:14:06,397 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:14:06,397 INFO train_multi TF=ALL: new best val=0.6341 — saved
2026-04-26 22:14:18,021 INFO train_multi TF=ALL epoch 33/50 train=0.6379 val=0.6337
2026-04-26 22:14:18,025 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:14:18,025 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:14:18,025 INFO train_multi TF=ALL: new best val=0.6337 — saved
2026-04-26 22:14:29,676 INFO train_multi TF=ALL epoch 34/50 train=0.6376 val=0.6328
2026-04-26 22:14:29,681 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:14:29,681 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:14:29,681 INFO train_multi TF=ALL: new best val=0.6328 — saved
2026-04-26 22:14:41,346 INFO train_multi TF=ALL epoch 35/50 train=0.6372 val=0.6331
2026-04-26 22:14:52,988 INFO train_multi TF=ALL epoch 36/50 train=0.6368 val=0.6335
2026-04-26 22:15:04,674 INFO train_multi TF=ALL epoch 37/50 train=0.6363 val=0.6327
2026-04-26 22:15:04,678 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:15:04,679 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:15:04,679 INFO train_multi TF=ALL: new best val=0.6327 — saved
2026-04-26 22:15:16,303 INFO train_multi TF=ALL epoch 38/50 train=0.6361 val=0.6331
2026-04-26 22:15:28,016 INFO train_multi TF=ALL epoch 39/50 train=0.6359 val=0.6329
2026-04-26 22:15:39,638 INFO train_multi TF=ALL epoch 40/50 train=0.6357 val=0.6324
2026-04-26 22:15:39,643 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:15:39,643 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:15:39,643 INFO train_multi TF=ALL: new best val=0.6324 — saved
2026-04-26 22:15:51,251 INFO train_multi TF=ALL epoch 41/50 train=0.6358 val=0.6324
2026-04-26 22:16:02,843 INFO train_multi TF=ALL epoch 42/50 train=0.6356 val=0.6323
2026-04-26 22:16:02,847 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:16:02,847 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:16:02,847 INFO train_multi TF=ALL: new best val=0.6323 — saved
2026-04-26 22:16:14,411 INFO train_multi TF=ALL epoch 43/50 train=0.6354 val=0.6324
2026-04-26 22:16:25,941 INFO train_multi TF=ALL epoch 44/50 train=0.6352 val=0.6322
2026-04-26 22:16:25,946 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:16:25,947 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:16:25,947 INFO train_multi TF=ALL: new best val=0.6322 — saved
2026-04-26 22:16:37,610 INFO train_multi TF=ALL epoch 45/50 train=0.6355 val=0.6325
2026-04-26 22:16:49,224 INFO train_multi TF=ALL epoch 46/50 train=0.6352 val=0.6322
2026-04-26 22:16:49,229 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:16:49,229 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:16:49,229 INFO train_multi TF=ALL: new best val=0.6322 — saved
2026-04-26 22:17:00,814 INFO train_multi TF=ALL epoch 47/50 train=0.6352 val=0.6321
2026-04-26 22:17:00,818 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:17:00,818 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:17:00,818 INFO train_multi TF=ALL: new best val=0.6321 — saved
2026-04-26 22:17:12,421 INFO train_multi TF=ALL epoch 48/50 train=0.6349 val=0.6321
2026-04-26 22:17:24,086 INFO train_multi TF=ALL epoch 49/50 train=0.6352 val=0.6320
2026-04-26 22:17:24,091 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 22:17:24,091 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 22:17:24,091 INFO train_multi TF=ALL: new best val=0.6320 — saved
2026-04-26 22:17:35,701 INFO train_multi TF=ALL epoch 50/50 train=0.6350 val=0.6320
2026-04-26 22:17:35,847 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-26 22:17:35,847 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-26 22:17:35,847 INFO Retrain complete. Total wall-clock: 649.7s
  DONE  Retrain gru [full-data retrain]
  START Retrain regime [full-data retrain]
2026-04-26 22:17:38,009 INFO retrain environment: KAGGLE
2026-04-26 22:17:39,576 INFO Device: CUDA (2 GPU(s))
2026-04-26 22:17:39,584 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 22:17:39,584 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 22:17:39,584 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 22:17:39,584 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 22:17:39,585 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-26 22:17:39,727 INFO NumExpr defaulting to 4 threads.
2026-04-26 22:17:39,915 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 22:17:39,915 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 22:17:39,915 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 22:17:39,916 INFO Regime phase macro_correlations: 0.0s
2026-04-26 22:17:39,916 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-26 22:17:39,952 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 22:17:39,952 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:17:39,978 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:17:39,992 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:17:40,015 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:17:40,029 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:17:40,052 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:17:40,066 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:17:40,089 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:17:40,104 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:17:40,126 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:17:40,140 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:17:40,160 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:17:40,173 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:17:40,193 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:17:40,207 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:17:40,227 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:17:40,241 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:17:40,261 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:17:40,274 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:17:40,295 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:17:40,310 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 22:17:40,346 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 22:17:41,083 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 22:18:02,405 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 22:18:02,407 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 22.1s
2026-04-26 22:18:02,408 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 22:18:11,902 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 22:18:11,903 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 9.5s
2026-04-26 22:18:11,904 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 22:18:19,297 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 22:18:19,297 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 7.4s
2026-04-26 22:18:19,297 INFO Regime phase GMM HTF total: 39.0s
2026-04-26 22:18:19,298 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 22:19:27,886 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 22:19:27,892 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 68.6s
2026-04-26 22:19:27,892 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 22:19:58,400 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 22:19:58,404 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 30.5s
2026-04-26 22:19:58,405 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 22:20:19,668 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 22:20:19,669 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 21.3s
2026-04-26 22:20:19,669 INFO Regime phase GMM LTF total: 120.4s
2026-04-26 22:20:19,772 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-26 22:20:19,774 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:19,775 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:19,776 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:19,777 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:19,778 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:19,779 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:19,780 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:19,781 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:19,782 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:19,783 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:19,785 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 22:20:19,907 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:19,947 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:19,948 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:19,948 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:19,956 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:19,957 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:20,353 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-26 22:20:20,354 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-26 22:20:20,535 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:20,568 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:20,569 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:20,569 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:20,577 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:20,578 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:20,941 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-26 22:20:20,942 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-26 22:20:21,139 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:21,175 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:21,176 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:21,177 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:21,185 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:21,186 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:21,565 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-26 22:20:21,566 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-26 22:20:21,733 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:21,768 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:21,769 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:21,770 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:21,777 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:21,778 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:22,143 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-26 22:20:22,144 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-26 22:20:22,325 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:22,360 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:22,361 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:22,362 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:22,369 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:22,370 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:22,740 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-26 22:20:22,741 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-26 22:20:22,911 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:22,943 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:22,944 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:22,944 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:22,951 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:22,952 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:23,314 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-26 22:20:23,315 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-26 22:20:23,471 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 22:20:23,499 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 22:20:23,500 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 22:20:23,500 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 22:20:23,507 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 22:20:23,508 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:23,886 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-26 22:20:23,887 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-26 22:20:24,051 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:24,085 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:24,086 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:24,086 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:24,095 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:24,096 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:24,460 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-26 22:20:24,461 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-26 22:20:24,629 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:24,664 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:24,665 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:24,665 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:24,673 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:24,674 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:25,044 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-26 22:20:25,045 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-26 22:20:25,218 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:25,251 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:25,252 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:25,253 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:25,260 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:25,261 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:25,635 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-26 22:20:25,636 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-26 22:20:25,904 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 22:20:25,958 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 22:20:25,959 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 22:20:25,960 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 22:20:25,969 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 22:20:25,970 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 22:20:26,784 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 22:20:26,786 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-26 22:20:26,949 INFO Regime phase HTF dataset build: 7.2s (103290 samples)
2026-04-26 22:20:26,950 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260426_222026
2026-04-26 22:20:27,146 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=34, n_classes=3)
2026-04-26 22:20:27,147 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=38114 dropped=65176 classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728})
2026-04-26 22:20:27,156 INFO RegimeClassifier[mode=htf_bias]: 38114 samples, classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728}, device=cuda
2026-04-26 22:20:27,157 INFO RegimeClassifier: sample weights — mean=0.708  ambiguous(<0.4)=0.0%
2026-04-26 22:20:27,157 INFO RegimeClassifier[mode=htf_bias]: warm start from existing weights
2026-04-26 22:20:27,157 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 22:20:29,365 INFO Regime epoch  1/50 — tr=0.4819 va=1.0828 acc=0.978 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.878}
2026-04-26 22:20:29,441 INFO Regime epoch  2/50 — tr=0.4822 va=1.0850 acc=0.978
2026-04-26 22:20:29,515 INFO Regime epoch  3/50 — tr=0.4819 va=1.0834 acc=0.978
2026-04-26 22:20:29,584 INFO Regime epoch  4/50 — tr=0.4819 va=1.0844 acc=0.977
2026-04-26 22:20:29,655 INFO Regime epoch  5/50 — tr=0.4812 va=1.0860 acc=0.977 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.875}
2026-04-26 22:20:29,721 INFO Regime epoch  6/50 — tr=0.4813 va=1.0851 acc=0.978
2026-04-26 22:20:29,787 INFO Regime epoch  7/50 — tr=0.4814 va=1.0814 acc=0.977
2026-04-26 22:20:29,852 INFO Regime epoch  8/50 — tr=0.4810 va=1.0812 acc=0.978
2026-04-26 22:20:29,921 INFO Regime epoch  9/50 — tr=0.4798 va=1.0810 acc=0.977
2026-04-26 22:20:29,995 INFO Regime epoch 10/50 — tr=0.4798 va=1.0775 acc=0.978 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.877}
2026-04-26 22:20:30,064 INFO Regime epoch 11/50 — tr=0.4788 va=1.0743 acc=0.979
2026-04-26 22:20:30,134 INFO Regime epoch 12/50 — tr=0.4786 va=1.0710 acc=0.979
2026-04-26 22:20:30,204 INFO Regime epoch 13/50 — tr=0.4776 va=1.0710 acc=0.980
2026-04-26 22:20:30,271 INFO Regime epoch 14/50 — tr=0.4778 va=1.0693 acc=0.980
2026-04-26 22:20:30,351 INFO Regime epoch 15/50 — tr=0.4769 va=1.0678 acc=0.980 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.889}
2026-04-26 22:20:30,425 INFO Regime epoch 16/50 — tr=0.4767 va=1.0652 acc=0.981
2026-04-26 22:20:30,499 INFO Regime epoch 17/50 — tr=0.4760 va=1.0633 acc=0.981
2026-04-26 22:20:30,573 INFO Regime epoch 18/50 — tr=0.4758 va=1.0611 acc=0.981
2026-04-26 22:20:30,646 INFO Regime epoch 19/50 — tr=0.4753 va=1.0595 acc=0.981
2026-04-26 22:20:30,724 INFO Regime epoch 20/50 — tr=0.4752 va=1.0579 acc=0.981 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.896}
2026-04-26 22:20:30,793 INFO Regime epoch 21/50 — tr=0.4748 va=1.0576 acc=0.982
2026-04-26 22:20:30,861 INFO Regime epoch 22/50 — tr=0.4748 va=1.0566 acc=0.982
2026-04-26 22:20:30,927 INFO Regime epoch 23/50 — tr=0.4738 va=1.0520 acc=0.982
2026-04-26 22:20:30,992 INFO Regime epoch 24/50 — tr=0.4741 va=1.0513 acc=0.982
2026-04-26 22:20:31,070 INFO Regime epoch 25/50 — tr=0.4735 va=1.0485 acc=0.983 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.904}
2026-04-26 22:20:31,147 INFO Regime epoch 26/50 — tr=0.4733 va=1.0464 acc=0.982
2026-04-26 22:20:31,224 INFO Regime epoch 27/50 — tr=0.4731 va=1.0454 acc=0.983
2026-04-26 22:20:31,302 INFO Regime epoch 28/50 — tr=0.4734 va=1.0458 acc=0.983
2026-04-26 22:20:31,378 INFO Regime epoch 29/50 — tr=0.4733 va=1.0472 acc=0.983
2026-04-26 22:20:31,460 INFO Regime epoch 30/50 — tr=0.4730 va=1.0450 acc=0.983 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.908}
2026-04-26 22:20:31,534 INFO Regime epoch 31/50 — tr=0.4722 va=1.0425 acc=0.983
2026-04-26 22:20:31,607 INFO Regime epoch 32/50 — tr=0.4724 va=1.0426 acc=0.983
2026-04-26 22:20:31,683 INFO Regime epoch 33/50 — tr=0.4727 va=1.0429 acc=0.984
2026-04-26 22:20:31,752 INFO Regime epoch 34/50 — tr=0.4722 va=1.0428 acc=0.984
2026-04-26 22:20:31,829 INFO Regime epoch 35/50 — tr=0.4723 va=1.0420 acc=0.984 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.909}
2026-04-26 22:20:31,903 INFO Regime epoch 36/50 — tr=0.4722 va=1.0402 acc=0.984
2026-04-26 22:20:31,973 INFO Regime epoch 37/50 — tr=0.4719 va=1.0401 acc=0.984
2026-04-26 22:20:32,041 INFO Regime epoch 38/50 — tr=0.4721 va=1.0406 acc=0.984
2026-04-26 22:20:32,113 INFO Regime epoch 39/50 — tr=0.4719 va=1.0420 acc=0.984
2026-04-26 22:20:32,190 INFO Regime epoch 40/50 — tr=0.4719 va=1.0412 acc=0.985 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.914}
2026-04-26 22:20:32,262 INFO Regime epoch 41/50 — tr=0.4718 va=1.0429 acc=0.984
2026-04-26 22:20:32,335 INFO Regime epoch 42/50 — tr=0.4714 va=1.0393 acc=0.985
2026-04-26 22:20:32,402 INFO Regime epoch 43/50 — tr=0.4718 va=1.0421 acc=0.984
2026-04-26 22:20:32,473 INFO Regime epoch 44/50 — tr=0.4718 va=1.0406 acc=0.985
2026-04-26 22:20:32,549 INFO Regime epoch 45/50 — tr=0.4720 va=1.0407 acc=0.984 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.912}
2026-04-26 22:20:32,622 INFO Regime epoch 46/50 — tr=0.4714 va=1.0394 acc=0.984
2026-04-26 22:20:32,689 INFO Regime epoch 47/50 — tr=0.4717 va=1.0404 acc=0.985
2026-04-26 22:20:32,757 INFO Regime epoch 48/50 — tr=0.4716 va=1.0407 acc=0.984
2026-04-26 22:20:32,828 INFO Regime epoch 49/50 — tr=0.4713 va=1.0396 acc=0.984
2026-04-26 22:20:32,906 INFO Regime epoch 50/50 — tr=0.4714 va=1.0384 acc=0.984 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.913}
2026-04-26 22:20:32,915 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 22:20:32,916 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 22:20:32,916 INFO Regime phase HTF train: 5.8s
2026-04-26 22:20:33,042 INFO Regime HTF complete: acc=0.984, n=103290 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.913}
2026-04-26 22:20:33,044 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 22:20:33,200 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 22:20:33,203 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 11.41578947368421, 'BIAS_DOWN': 10.635761589403973, 'BIAS_NEUTRAL': 17.960468521229867}
2026-04-26 22:20:33,207 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 4338, 'mean': 0.00011110466582543088, 'mean_over_std': 0.027632581210502927}, 'BIAS_DOWN': {'n': 3212, 'mean': 6.063430222660252e-05, 'mean_over_std': 0.013898489590377157}, 'BIAS_NEUTRAL': {'n': 12266, 'mean': 1.2151554902814046e-05, 'mean_over_std': 0.0031957000513162357}}
2026-04-26 22:20:33,207 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 3651, 'mean': 0.0002852242108836959, 'mean_over_std': 0.07015017232491755}, 'BIAS_DOWN': {'n': 2644, 'mean': -0.00014900749618981248, 'mean_over_std': -0.03333037041298237}, 'BIAS_NEUTRAL': {'n': 1376, 'mean': 9.595797941249082e-05, 'mean_over_std': 0.02492411099521803}}
2026-04-26 22:20:33,207 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-26 22:20:33,209 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:33,211 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:33,212 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:33,214 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:33,215 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:33,217 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:33,218 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:33,220 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:33,221 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:33,223 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:33,226 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 22:20:33,236 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:33,240 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:33,240 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:33,241 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:33,241 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:33,243 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:33,868 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-26 22:20:33,871 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-26 22:20:34,007 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:34,009 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:34,010 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:34,010 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:34,011 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:34,013 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:34,598 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-26 22:20:34,601 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-26 22:20:34,745 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:34,748 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:34,748 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:34,749 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:34,749 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:34,751 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:35,336 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-26 22:20:35,339 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-26 22:20:35,479 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:35,481 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:35,482 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:35,483 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:35,483 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:35,485 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:36,058 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-26 22:20:36,061 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-26 22:20:36,195 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:36,197 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:36,198 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:36,199 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:36,199 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:36,201 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:36,816 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-26 22:20:36,819 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-26 22:20:36,957 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:36,959 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:36,960 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:36,961 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:36,961 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:36,963 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:37,546 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-26 22:20:37,548 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-26 22:20:37,679 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 22:20:37,681 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 22:20:37,682 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 22:20:37,682 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 22:20:37,682 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 22:20:37,684 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:38,244 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-26 22:20:38,246 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-26 22:20:38,388 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:38,391 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:38,392 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:38,392 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:38,392 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:38,394 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:38,963 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-26 22:20:38,966 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-26 22:20:39,101 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:39,103 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:39,104 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:39,104 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:39,105 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:39,107 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:39,690 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-26 22:20:39,693 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-26 22:20:39,832 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:39,834 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:39,835 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:39,836 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:39,836 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 22:20:39,838 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 22:20:40,409 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-26 22:20:40,413 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-26 22:20:40,553 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 22:20:40,560 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 22:20:40,561 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 22:20:40,561 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 22:20:40,562 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 22:20:40,565 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 22:20:41,788 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 22:20:41,794 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-26 22:20:42,089 INFO Regime phase LTF dataset build: 8.9s (401471 samples)
2026-04-26 22:20:42,090 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260426_222042
2026-04-26 22:20:42,094 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=18, n_classes=4)
2026-04-26 22:20:42,097 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=299185 dropped=102286 classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127})
2026-04-26 22:20:42,153 INFO RegimeClassifier[mode=ltf_behaviour]: 299185 samples, classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127}, device=cuda
2026-04-26 22:20:42,154 INFO RegimeClassifier: sample weights — mean=0.693  ambiguous(<0.4)=0.0%
2026-04-26 22:20:42,154 INFO RegimeClassifier[mode=ltf_behaviour]: warm start from existing weights
2026-04-26 22:20:42,154 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 22:20:42,713 INFO Regime epoch  1/50 — tr=0.6272 va=1.2544 acc=0.822 per_class={'TRENDING': 0.799, 'RANGING': 0.768, 'CONSOLIDATING': 0.825, 'VOLATILE': 0.896}
2026-04-26 22:20:43,205 INFO Regime epoch  2/50 — tr=0.6272 va=1.2512 acc=0.821
2026-04-26 22:20:43,713 INFO Regime epoch  3/50 — tr=0.6268 va=1.2523 acc=0.821
2026-04-26 22:20:44,248 INFO Regime epoch  4/50 — tr=0.6270 va=1.2496 acc=0.821
2026-04-26 22:20:44,794 INFO Regime epoch  5/50 — tr=0.6269 va=1.2503 acc=0.820 per_class={'TRENDING': 0.79, 'RANGING': 0.775, 'CONSOLIDATING': 0.813, 'VOLATILE': 0.907}
2026-04-26 22:20:45,305 INFO Regime epoch  6/50 — tr=0.6265 va=1.2511 acc=0.822
2026-04-26 22:20:45,803 INFO Regime epoch  7/50 — tr=0.6263 va=1.2488 acc=0.824
2026-04-26 22:20:46,320 INFO Regime epoch  8/50 — tr=0.6264 va=1.2457 acc=0.822
2026-04-26 22:20:46,853 INFO Regime epoch  9/50 — tr=0.6257 va=1.2446 acc=0.825
2026-04-26 22:20:47,419 INFO Regime epoch 10/50 — tr=0.6258 va=1.2443 acc=0.824 per_class={'TRENDING': 0.798, 'RANGING': 0.772, 'CONSOLIDATING': 0.83, 'VOLATILE': 0.901}
2026-04-26 22:20:47,920 INFO Regime epoch 11/50 — tr=0.6254 va=1.2422 acc=0.824
2026-04-26 22:20:48,415 INFO Regime epoch 12/50 — tr=0.6248 va=1.2399 acc=0.827
2026-04-26 22:20:48,906 INFO Regime epoch 13/50 — tr=0.6249 va=1.2408 acc=0.827
2026-04-26 22:20:49,399 INFO Regime epoch 14/50 — tr=0.6247 va=1.2389 acc=0.826
2026-04-26 22:20:49,930 INFO Regime epoch 15/50 — tr=0.6242 va=1.2382 acc=0.827 per_class={'TRENDING': 0.802, 'RANGING': 0.776, 'CONSOLIDATING': 0.837, 'VOLATILE': 0.899}
2026-04-26 22:20:50,427 INFO Regime epoch 16/50 — tr=0.6241 va=1.2353 acc=0.828
2026-04-26 22:20:50,964 INFO Regime epoch 17/50 — tr=0.6238 va=1.2326 acc=0.827
2026-04-26 22:20:51,480 INFO Regime epoch 18/50 — tr=0.6234 va=1.2381 acc=0.829
2026-04-26 22:20:51,976 INFO Regime epoch 19/50 — tr=0.6234 va=1.2339 acc=0.829
2026-04-26 22:20:52,534 INFO Regime epoch 20/50 — tr=0.6232 va=1.2363 acc=0.830 per_class={'TRENDING': 0.806, 'RANGING': 0.782, 'CONSOLIDATING': 0.836, 'VOLATILE': 0.901}
2026-04-26 22:20:53,044 INFO Regime epoch 21/50 — tr=0.6228 va=1.2355 acc=0.831
2026-04-26 22:20:53,570 INFO Regime epoch 22/50 — tr=0.6228 va=1.2339 acc=0.831
2026-04-26 22:20:54,087 INFO Regime epoch 23/50 — tr=0.6227 va=1.2349 acc=0.833
2026-04-26 22:20:54,591 INFO Regime epoch 24/50 — tr=0.6225 va=1.2331 acc=0.832
2026-04-26 22:20:55,147 INFO Regime epoch 25/50 — tr=0.6225 va=1.2317 acc=0.833 per_class={'TRENDING': 0.811, 'RANGING': 0.777, 'CONSOLIDATING': 0.857, 'VOLATILE': 0.893}
2026-04-26 22:20:55,678 INFO Regime epoch 26/50 — tr=0.6221 va=1.2358 acc=0.834
2026-04-26 22:20:56,204 INFO Regime epoch 27/50 — tr=0.6222 va=1.2320 acc=0.834
2026-04-26 22:20:56,783 INFO Regime epoch 28/50 — tr=0.6220 va=1.2283 acc=0.834
2026-04-26 22:20:57,307 INFO Regime epoch 29/50 — tr=0.6217 va=1.2318 acc=0.836
2026-04-26 22:20:57,879 INFO Regime epoch 30/50 — tr=0.6214 va=1.2300 acc=0.835 per_class={'TRENDING': 0.815, 'RANGING': 0.778, 'CONSOLIDATING': 0.854, 'VOLATILE': 0.894}
2026-04-26 22:20:58,391 INFO Regime epoch 31/50 — tr=0.6214 va=1.2294 acc=0.834
2026-04-26 22:20:58,917 INFO Regime epoch 32/50 — tr=0.6215 va=1.2276 acc=0.833
2026-04-26 22:20:59,436 INFO Regime epoch 33/50 — tr=0.6215 va=1.2282 acc=0.834
2026-04-26 22:20:59,947 INFO Regime epoch 34/50 — tr=0.6213 va=1.2281 acc=0.835
2026-04-26 22:21:00,487 INFO Regime epoch 35/50 — tr=0.6213 va=1.2300 acc=0.834 per_class={'TRENDING': 0.811, 'RANGING': 0.779, 'CONSOLIDATING': 0.852, 'VOLATILE': 0.9}
2026-04-26 22:21:01,032 INFO Regime epoch 36/50 — tr=0.6213 va=1.2307 acc=0.836
2026-04-26 22:21:01,535 INFO Regime epoch 37/50 — tr=0.6212 va=1.2311 acc=0.837
2026-04-26 22:21:02,054 INFO Regime epoch 38/50 — tr=0.6211 va=1.2288 acc=0.836
2026-04-26 22:21:02,573 INFO Regime epoch 39/50 — tr=0.6209 va=1.2290 acc=0.835
2026-04-26 22:21:03,140 INFO Regime epoch 40/50 — tr=0.6208 va=1.2259 acc=0.836 per_class={'TRENDING': 0.814, 'RANGING': 0.778, 'CONSOLIDATING': 0.858, 'VOLATILE': 0.897}
2026-04-26 22:21:03,655 INFO Regime epoch 41/50 — tr=0.6211 va=1.2265 acc=0.835
2026-04-26 22:21:04,155 INFO Regime epoch 42/50 — tr=0.6208 va=1.2259 acc=0.836
2026-04-26 22:21:04,712 INFO Regime epoch 43/50 — tr=0.6212 va=1.2262 acc=0.836
2026-04-26 22:21:05,228 INFO Regime epoch 44/50 — tr=0.6208 va=1.2261 acc=0.835
2026-04-26 22:21:05,762 INFO Regime epoch 45/50 — tr=0.6213 va=1.2280 acc=0.837 per_class={'TRENDING': 0.817, 'RANGING': 0.779, 'CONSOLIDATING': 0.856, 'VOLATILE': 0.897}
2026-04-26 22:21:06,268 INFO Regime epoch 46/50 — tr=0.6208 va=1.2278 acc=0.836
2026-04-26 22:21:06,828 INFO Regime epoch 47/50 — tr=0.6209 va=1.2296 acc=0.835
2026-04-26 22:21:07,315 INFO Regime epoch 48/50 — tr=0.6207 va=1.2287 acc=0.835
2026-04-26 22:21:07,850 INFO Regime epoch 49/50 — tr=0.6210 va=1.2254 acc=0.836
2026-04-26 22:21:08,397 INFO Regime epoch 50/50 — tr=0.6209 va=1.2250 acc=0.835 per_class={'TRENDING': 0.813, 'RANGING': 0.782, 'CONSOLIDATING': 0.858, 'VOLATILE': 0.894}
2026-04-26 22:21:08,437 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 22:21:08,437 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 22:21:08,439 INFO Regime phase LTF train: 26.3s
2026-04-26 22:21:08,568 INFO Regime LTF complete: acc=0.835, n=401471 per_class={'TRENDING': 0.813, 'RANGING': 0.782, 'CONSOLIDATING': 0.858, 'VOLATILE': 0.894}
2026-04-26 22:21:08,572 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 22:21:09,035 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 22:21:09,039 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 9.355291913830783, 'RANGING': 6.096303199751476, 'CONSOLIDATING': 5.598885793871866, 'VOLATILE': 6.771351107094442}
2026-04-26 22:21:09,047 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 29965, 'mean': 2.149062259083323e-05, 'mean_over_std': 0.011155601434349563}, 'RANGING': {'n': 19623, 'mean': -3.254493608872846e-06, 'mean_over_std': -0.001761415548292494}, 'CONSOLIDATING': {'n': 10050, 'mean': -3.039756158146958e-06, 'mean_over_std': -0.001792488091618151}, 'VOLATILE': {'n': 14985, 'mean': 1.8102764458593193e-05, 'mean_over_std': 0.006974696291378742}}
2026-04-26 22:21:09,048 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 26145, 'mean': 2.874435605138053e-05, 'mean_over_std': 0.01595450632691917}, 'RANGING': {'n': 7858, 'mean': 1.7829776785521727e-05, 'mean_over_std': 0.011851561804467463}, 'CONSOLIDATING': {'n': 8543, 'mean': 7.845231961250924e-06, 'mean_over_std': 0.004915839521549481}, 'VOLATILE': {'n': 13636, 'mean': 5.480114267306668e-06, 'mean_over_std': 0.0020378119917389375}}
2026-04-26 22:21:09,048 INFO Regime retrain total: 209.5s (504761 samples)
2026-04-26 22:21:09,050 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 22:21:09,050 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 22:21:09,050 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 22:21:09,051 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-26 22:21:09,051 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-26 22:21:09,051 INFO Retrain complete. Total wall-clock: 209.5s
  DONE  Retrain regime [full-data retrain]
  START Retrain quality [full-data retrain]
2026-04-26 22:21:10,368 INFO retrain environment: KAGGLE
2026-04-26 22:21:11,968 INFO Device: CUDA (2 GPU(s))
2026-04-26 22:21:11,979 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 22:21:11,979 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 22:21:11,980 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 22:21:11,980 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 22:21:11,981 INFO === QualityScorer retrain ===
2026-04-26 22:21:12,133 INFO NumExpr defaulting to 4 threads.
2026-04-26 22:21:12,325 INFO QualityScorer: CUDA available — using GPU
2026-04-26 22:21:12,325 INFO Retrain complete. Total wall-clock: 0.3s
  WARN  Retrain quality failed (exit 1) — continuing
  START Retrain rl [full-data retrain]
2026-04-26 22:21:12,895 INFO retrain environment: KAGGLE
2026-04-26 22:21:14,503 INFO Device: CUDA (2 GPU(s))
2026-04-26 22:21:14,514 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 22:21:14,515 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 22:21:14,515 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 22:21:14,515 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 22:21:14,516 INFO === RLAgent (PPO) retrain ===
2026-04-26 22:21:14,522 INFO Retrain complete. Total wall-clock: 0.0s
  WARN  Retrain rl failed (exit 1) — continuing

=== Round 3: Backtest on last 3yr (post-retrain evaluation) ===
  START Round 3 - Post-retrain backtest (last 3yr)
2026-04-26 22:21:15,352 INFO === STEP 6: BACKTEST (round3) ===
2026-04-26 22:21:15,353 INFO BT_WINDOW=round3 — post-retrain eval: 2022-08-05 → 2025-08-05 (last 3yr)
2026-04-26 22:21:15,353 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-04-26 22:21:15,353 INFO Round 3 — running backtest: 2022-08-05 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-26 22:21:17,748 WARNING QualityScorer unavailable (weights missing or load failed)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260426_222117.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)             4820  58.6%   3.12 3871.2% 58.6% 28.7%   6.9%    8.26
  gate_diagnostics: bars=692253 no_signal=159302 quality_block=0 session_skip=419772 density=64979 pm_reject=0 daily_skip=0 cooldown=43380 daily_halt_events=0 enforce_daily_halt=False
  no_signal_reasons: weak_gru_direction=87288, htf_bias_conflict=70375, trend_pullback_conflict=1607, range_side_conflict=32

Calibration Summary:
  all          [WARN] Non-monotonic calibration: 2/5 pairs violated. Consider retraining QualityScorer
  ml_trader    [WARN] Non-monotonic calibration: 2/5 pairs violated. Consider retraining QualityScorer
2026-04-26 22:24:26,586 INFO Round 3 backtest — 4820 trades | avg WR=58.6% | avg PF=3.12 | avg Sharpe=8.26
2026-04-26 22:24:26,586 INFO   ml_trader: 4820 trades | WR=58.6% | PF=3.12 | Return=3871.2% | DD=6.9% | Sharpe=8.26
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 4820
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (4820 rows)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  AUDUSD        466 trades     2 days  233.00/day  [OVERTRADE]
  EURGBP        500 trades     2 days  250.00/day  [OVERTRADE]
  EURJPY        458 trades     2 days  229.00/day  [OVERTRADE]
  EURUSD        495 trades     2 days  247.50/day  [OVERTRADE]
  GBPJPY        478 trades     2 days  239.00/day  [OVERTRADE]
  GBPUSD        495 trades     2 days  247.50/day  [OVERTRADE]
  NZDUSD         19 trades     2 days   9.50/day  [OVERTRADE]
  USDCAD        466 trades     2 days  233.00/day  [OVERTRADE]
  USDCHF        508 trades     2 days  254.00/day  [OVERTRADE]
  USDJPY        467 trades     2 days  233.50/day  [OVERTRADE]
  XAUUSD        468 trades     2 days  234.00/day  [OVERTRADE]
  ⚠  AUDUSD: 233.00/day (>1.5)
  ⚠  EURGBP: 250.00/day (>1.5)
  ⚠  EURJPY: 229.00/day (>1.5)
  ⚠  EURUSD: 247.50/day (>1.5)
  ⚠  GBPJPY: 239.00/day (>1.5)
  ⚠  GBPUSD: 247.50/day (>1.5)
  ⚠  NZDUSD: 9.50/day (>1.5)
  ⚠  USDCAD: 233.00/day (>1.5)
  ⚠  USDCHF: 254.00/day (>1.5)
  ⚠  USDJPY: 233.50/day (>1.5)
  ⚠  XAUUSD: 234.00/day (>1.5)

──────────────────────────────────────────────────────────────
CHECK 2: REGIME DISTRIBUTION  (% of trades, WR, avgEV)
──────────────────────────────────────────────────────────────
  BIAS_DOWN          1252 trades   26.0%  WR=58.5%  avgEV=0.000
  BIAS_NEUTRAL       1912 trades  DONE  Round 3 - Post-retrain backtest (last 3yr)
  Saved Round 3 result → round3_summary.json

======================================================================
  BLIND BACKTEST PIPELINE COMPLETE
======================================================================
  Round 1 (val window)          trades=0  WR=0.0%  PF=0.000  Sharpe=0.000
  Round 2 (blind test)          trades=0  WR=0.0%  PF=0.000  Sharpe=0.000
  Round 3 (last 3yr)            trades=4820  WR=58.6%  PF=3.117  Sharpe=8.263


WARNING: GITHUB_TOKEN not set — skipping GitHub push
2026-04-26 22:24:28,426 INFO Round 3: wrote 4820 journal entries (total in file: 4820)
