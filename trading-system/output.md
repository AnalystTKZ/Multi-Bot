Cleared done-check: training_summary.json
  Cleared done-check: latest_summary.json
Environment : KAGGLE
  base      -> /kaggle/working/Multi-Bot/trading-system
  data      -> /kaggle/input/datasets/tysonsiwela/ml-dataset/training_data
  processed -> /kaggle/input/datasets/tysonsiwela/ml-dataset/processed_data
  ml_train  -> /kaggle/working/Multi-Bot/trading-system/ml_training
  weights   -> /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
  output    -> /kaggle/working
  kaggle/input -> /kaggle/input
    dataset: datasets  (has training_data=False, processed_data=False)

All scripts and inputs verified.

=== Phase 0-5: Data preparation ===
  SKIP  Step 0 - Resample
  SKIP  Step 1 - Inventory
  SKIP  Step 2 - Cleaning
  SKIP  Step 3 - Alignment
  SKIP  Step 4 - Features
  START Step 5 - Split
2026-04-23 22:56:33,661 INFO Loading feature-engineered data...
2026-04-23 22:56:34,319 INFO Loaded 221743 rows, 202 features
2026-04-23 22:56:34,320 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-23 22:56:34,323 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-23 22:56:34,323 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-23 22:56:34,323 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-23 22:56:34,323 INFO No leakage confirmed: train < val < test timestamps

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
2026-04-23 22:56:36,664 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-23 22:56:36,664 INFO --- Training regime ---
2026-04-23 22:56:36,664 INFO Running retrain --model regime
2026-04-23 22:56:36,849 INFO retrain environment: KAGGLE
2026-04-23 22:56:38,513 INFO Device: CUDA (2 GPU(s))
2026-04-23 22:56:38,524 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-23 22:56:38,524 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-23 22:56:38,525 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-23 22:56:38,671 INFO NumExpr defaulting to 4 threads.
2026-04-23 22:56:38,877 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-23 22:56:38,877 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-23 22:56:38,877 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-23 22:56:39,152 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-23 22:56:39,154 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:39,254 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:39,327 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:39,398 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:39,471 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:39,544 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:39,615 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:39,686 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:39,755 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:39,824 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:39,910 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-23 22:56:39,969 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-23 22:56:39,986 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:39,987 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:40,003 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:40,004 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:40,020 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:40,022 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:40,040 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:40,042 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:40,059 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:40,062 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:40,079 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:40,083 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:40,098 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:40,101 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:40,116 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:40,120 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:40,136 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:40,139 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:40,155 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:40,158 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:56:40,176 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-23 22:56:40,183 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-23 22:56:41,356 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-23 22:57:04,594 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-23 22:57:04,596 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias)
2026-04-23 22:57:04,596 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-23 22:57:14,009 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-23 22:57:14,013 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias)
2026-04-23 22:57:14,014 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-23 22:57:21,357 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-23 22:57:21,361 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias)
2026-04-23 22:57:21,361 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-23 22:58:30,829 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-23 22:58:30,832 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour)
2026-04-23 22:58:30,832 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-23 22:59:01,453 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-23 22:59:01,453 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour)
2026-04-23 22:59:01,453 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-23 22:59:22,932 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-23 22:59:22,934 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour)
2026-04-23 22:59:23,033 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-23 22:59:23,035 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:59:23,036 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:59:23,037 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:59:23,039 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:59:23,040 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:59:23,041 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:59:23,042 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:59:23,043 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:59:23,044 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:59:23,045 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:59:23,047 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-23 22:59:23,186 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:23,230 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:23,231 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:23,232 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:23,240 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:23,241 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:59:26,020 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-23 22:59:26,021 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-23 22:59:26,201 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:26,235 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:26,235 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:26,236 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:26,244 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:26,245 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:59:28,918 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-23 22:59:28,919 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-23 22:59:29,093 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:29,131 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:29,132 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:29,132 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:29,142 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:29,143 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:59:31,869 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-23 22:59:31,870 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-23 22:59:32,041 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:32,079 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:32,080 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:32,081 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:32,090 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:32,091 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:59:34,756 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-23 22:59:34,757 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-23 22:59:34,929 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:34,966 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:34,967 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:34,967 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:35,004 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:35,005 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:59:37,710 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-23 22:59:37,711 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-23 22:59:37,883 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:37,919 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:37,920 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:37,920 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:37,930 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:37,931 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:59:40,561 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-23 22:59:40,562 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-23 22:59:40,710 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-23 22:59:40,738 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-23 22:59:40,739 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-23 22:59:40,739 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-23 22:59:40,747 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-23 22:59:40,748 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:59:43,384 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-23 22:59:43,385 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-23 22:59:43,550 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:43,583 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:43,584 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:43,584 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:43,594 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:43,594 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:59:46,376 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-23 22:59:46,377 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-23 22:59:46,543 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:46,578 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:46,579 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:46,579 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:46,588 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:46,589 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:59:49,336 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-23 22:59:49,337 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-23 22:59:49,512 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:49,547 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:49,547 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:49,548 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:49,557 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 22:59:49,558 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-23 22:59:52,264 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-23 22:59:52,265 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-23 22:59:52,537 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-23 22:59:52,606 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-23 22:59:52,607 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-23 22:59:52,607 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-23 22:59:52,619 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-23 22:59:52,621 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-23 22:59:58,922 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-23 22:59:58,924 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-23 22:59:59,105 INFO RegimeClassifier[mode=htf_bias]: 103290 samples, classes={'BIAS_UP': 18622, 'BIAS_DOWN': 18286, 'BIAS_NEUTRAL': 66382}, device=cuda
2026-04-23 22:59:59,106 INFO RegimeClassifier: sample weights — mean=0.360  ambiguous(<0.4)=69.5%
2026-04-23 22:59:59,399 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-04-23 22:59:59,400 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-23 23:00:04,346 INFO Regime epoch  1/50 — tr=0.6879 va=1.3057 acc=0.480 per_class={'BIAS_UP': 0.091, 'BIAS_DOWN': 0.517, 'BIAS_NEUTRAL': 0.61}
2026-04-23 23:00:04,522 INFO Regime epoch  2/50 — tr=0.6753 va=1.2707 acc=0.488
2026-04-23 23:00:04,691 INFO Regime epoch  3/50 — tr=0.6564 va=1.2266 acc=0.511
2026-04-23 23:00:04,865 INFO Regime epoch  4/50 — tr=0.6284 va=1.1684 acc=0.525
2026-04-23 23:00:05,061 INFO Regime epoch  5/50 — tr=0.5944 va=1.1059 acc=0.522 per_class={'BIAS_UP': 0.908, 'BIAS_DOWN': 0.927, 'BIAS_NEUTRAL': 0.282}
2026-04-23 23:00:05,234 INFO Regime epoch  6/50 — tr=0.5663 va=1.0404 acc=0.536
2026-04-23 23:00:05,413 INFO Regime epoch  7/50 — tr=0.5416 va=0.9827 acc=0.540
2026-04-23 23:00:05,595 INFO Regime epoch  8/50 — tr=0.5256 va=0.9386 acc=0.549
2026-04-23 23:00:05,766 INFO Regime epoch  9/50 — tr=0.5143 va=0.9116 acc=0.556
2026-04-23 23:00:05,952 INFO Regime epoch 10/50 — tr=0.5060 va=0.8984 acc=0.559 per_class={'BIAS_UP': 0.978, 'BIAS_DOWN': 0.993, 'BIAS_NEUTRAL': 0.299}
2026-04-23 23:00:06,133 INFO Regime epoch 11/50 — tr=0.4991 va=0.8900 acc=0.566
2026-04-23 23:00:06,302 INFO Regime epoch 12/50 — tr=0.4937 va=0.8807 acc=0.573
2026-04-23 23:00:06,473 INFO Regime epoch 13/50 — tr=0.4904 va=0.8773 acc=0.573
2026-04-23 23:00:06,649 INFO Regime epoch 14/50 — tr=0.4872 va=0.8757 acc=0.576
2026-04-23 23:00:06,831 INFO Regime epoch 15/50 — tr=0.4835 va=0.8742 acc=0.579 per_class={'BIAS_UP': 0.982, 'BIAS_DOWN': 0.991, 'BIAS_NEUTRAL': 0.331}
2026-04-23 23:00:07,005 INFO Regime epoch 16/50 — tr=0.4818 va=0.8734 acc=0.579
2026-04-23 23:00:07,179 INFO Regime epoch 17/50 — tr=0.4793 va=0.8727 acc=0.584
2026-04-23 23:00:07,353 INFO Regime epoch 18/50 — tr=0.4774 va=0.8729 acc=0.583
2026-04-23 23:00:07,527 INFO Regime epoch 19/50 — tr=0.4758 va=0.8735 acc=0.586
2026-04-23 23:00:07,713 INFO Regime epoch 20/50 — tr=0.4743 va=0.8727 acc=0.583 per_class={'BIAS_UP': 0.986, 'BIAS_DOWN': 0.993, 'BIAS_NEUTRAL': 0.335}
2026-04-23 23:00:07,886 INFO Regime epoch 21/50 — tr=0.4734 va=0.8722 acc=0.590
2026-04-23 23:00:08,060 INFO Regime epoch 22/50 — tr=0.4719 va=0.8713 acc=0.591
2026-04-23 23:00:08,243 INFO Regime epoch 23/50 — tr=0.4714 va=0.8702 acc=0.592
2026-04-23 23:00:08,415 INFO Regime epoch 24/50 — tr=0.4708 va=0.8704 acc=0.595
2026-04-23 23:00:08,599 INFO Regime epoch 25/50 — tr=0.4695 va=0.8704 acc=0.595 per_class={'BIAS_UP': 0.988, 'BIAS_DOWN': 0.993, 'BIAS_NEUTRAL': 0.353}
2026-04-23 23:00:08,771 INFO Regime epoch 26/50 — tr=0.4693 va=0.8704 acc=0.596
2026-04-23 23:00:08,952 INFO Regime epoch 27/50 — tr=0.4682 va=0.8700 acc=0.596
2026-04-23 23:00:09,133 INFO Regime epoch 28/50 — tr=0.4683 va=0.8695 acc=0.598
2026-04-23 23:00:09,325 INFO Regime epoch 29/50 — tr=0.4676 va=0.8696 acc=0.599
2026-04-23 23:00:09,510 INFO Regime epoch 30/50 — tr=0.4673 va=0.8697 acc=0.600 per_class={'BIAS_UP': 0.99, 'BIAS_DOWN': 0.994, 'BIAS_NEUTRAL': 0.361}
2026-04-23 23:00:09,684 INFO Regime epoch 31/50 — tr=0.4666 va=0.8692 acc=0.601
2026-04-23 23:00:09,874 INFO Regime epoch 32/50 — tr=0.4668 va=0.8708 acc=0.604
2026-04-23 23:00:10,066 INFO Regime epoch 33/50 — tr=0.4668 va=0.8703 acc=0.600
2026-04-23 23:00:10,235 INFO Regime epoch 34/50 — tr=0.4661 va=0.8709 acc=0.605
2026-04-23 23:00:10,424 INFO Regime epoch 35/50 — tr=0.4658 va=0.8704 acc=0.599 per_class={'BIAS_UP': 0.989, 'BIAS_DOWN': 0.995, 'BIAS_NEUTRAL': 0.359}
2026-04-23 23:00:10,593 INFO Regime epoch 36/50 — tr=0.4659 va=0.8694 acc=0.602
2026-04-23 23:00:10,761 INFO Regime epoch 37/50 — tr=0.4654 va=0.8693 acc=0.603
2026-04-23 23:00:10,931 INFO Regime epoch 38/50 — tr=0.4654 va=0.8696 acc=0.604
2026-04-23 23:00:11,104 INFO Regime epoch 39/50 — tr=0.4655 va=0.8702 acc=0.602
2026-04-23 23:00:11,287 INFO Regime epoch 40/50 — tr=0.4651 va=0.8704 acc=0.604 per_class={'BIAS_UP': 0.989, 'BIAS_DOWN': 0.995, 'BIAS_NEUTRAL': 0.367}
2026-04-23 23:00:11,466 INFO Regime epoch 41/50 — tr=0.4652 va=0.8703 acc=0.602
2026-04-23 23:00:11,466 INFO Regime early stop at epoch 41 (no_improve=10)
2026-04-23 23:00:11,480 WARNING RegimeClassifier accuracy 0.60 < 0.65 threshold
2026-04-23 23:00:11,484 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-23 23:00:11,484 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-23 23:00:11,606 INFO Regime HTF complete: acc=0.601, n=103290
2026-04-23 23:00:11,608 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-23 23:00:11,759 INFO Rule labels LTF_BEHAVIOUR [4H]: {'TRENDING': 8970, 'RANGING': 4592, 'CONSOLIDATING': 2247, 'VOLATILE': 4008}  ambiguous=4539 (total=19817)  short_runs_zeroed=570
2026-04-23 23:00:11,765 INFO Regime[4H] persistence (avg bars/run) on XAUUSD 4H:
{0: 12.121621621621621, 1: 6.4858757062146895, 2: 5.575682382133995, 3: 10.329896907216495}
2026-04-23 23:00:11,767 INFO Regime[4H] return separation on XAUUSD 4H:
{0: 5.9502411776840314e-05, 1: 5.664222849552333e-05, 2: -6.138442068010657e-05, 3: 4.239843820627722e-05}
2026-04-23 23:00:11,767 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-23 23:00:11,769 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:00:11,770 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:00:11,772 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:00:11,774 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:00:11,775 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:00:11,777 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:00:11,778 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:00:11,779 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:00:11,781 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:00:11,782 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:00:11,785 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-23 23:00:11,795 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:11,797 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:11,797 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:11,798 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:11,798 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:11,800 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:00:21,506 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-23 23:00:21,508 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-23 23:00:21,641 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:21,643 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:21,644 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:21,644 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:21,645 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:21,647 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:00:31,388 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-23 23:00:31,391 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-23 23:00:31,524 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:31,527 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:31,527 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:31,528 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:31,528 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:31,530 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:00:41,235 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-23 23:00:41,238 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-23 23:00:41,375 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:41,377 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:41,378 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:41,378 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:41,379 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:41,381 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:00:51,142 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-23 23:00:51,145 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-23 23:00:51,285 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:51,287 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:51,288 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:51,288 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:51,289 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:00:51,291 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:01:00,934 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-23 23:01:00,936 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-23 23:01:01,070 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:01:01,072 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:01:01,073 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:01:01,073 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:01:01,074 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:01:01,076 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:01:10,737 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-23 23:01:10,740 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-23 23:01:10,873 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-23 23:01:10,874 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-23 23:01:10,875 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-23 23:01:10,875 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-23 23:01:10,876 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-23 23:01:10,877 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:01:20,519 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-23 23:01:20,522 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-23 23:01:20,653 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:01:20,655 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:01:20,656 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:01:20,656 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:01:20,657 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:01:20,659 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:01:30,329 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-23 23:01:30,332 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-23 23:01:30,469 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:01:30,472 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:01:30,472 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:01:30,473 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:01:30,473 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:01:30,475 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:01:40,151 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-23 23:01:40,154 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-23 23:01:40,291 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:01:40,293 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:01:40,294 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:01:40,294 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:01:40,295 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:01:40,297 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:01:49,960 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-23 23:01:49,963 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-23 23:01:50,106 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-23 23:01:50,110 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-23 23:01:50,111 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-23 23:01:50,111 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-23 23:01:50,112 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-23 23:01:50,115 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-23 23:02:11,998 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-23 23:02:12,004 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-23 23:02:12,351 INFO RegimeClassifier[mode=ltf_behaviour]: 401471 samples, classes={'TRENDING': 160094, 'RANGING': 105286, 'CONSOLIDATING': 53524, 'VOLATILE': 82567}, device=cuda
2026-04-23 23:02:12,353 INFO RegimeClassifier: sample weights — mean=0.505  ambiguous(<0.4)=33.0%
2026-04-23 23:02:12,355 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-04-23 23:02:12,355 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-23 23:02:13,069 INFO Regime epoch  1/50 — tr=0.9758 va=1.6937 acc=0.306 per_class={'TRENDING': 0.326, 'RANGING': 0.202, 'CONSOLIDATING': 0.602, 'VOLATILE': 0.201}
2026-04-23 23:02:13,699 INFO Regime epoch  2/50 — tr=0.9357 va=1.6120 acc=0.411
2026-04-23 23:02:14,320 INFO Regime epoch  3/50 — tr=0.8820 va=1.5391 acc=0.430
2026-04-23 23:02:14,976 INFO Regime epoch  4/50 — tr=0.8346 va=1.4958 acc=0.431
2026-04-23 23:02:15,667 INFO Regime epoch  5/50 — tr=0.8041 va=1.4765 acc=0.438 per_class={'TRENDING': 0.34, 'RANGING': 0.013, 'CONSOLIDATING': 0.858, 'VOLATILE': 0.907}
2026-04-23 23:02:16,327 INFO Regime epoch  6/50 — tr=0.7835 va=1.4728 acc=0.447
2026-04-23 23:02:17,024 INFO Regime epoch  7/50 — tr=0.7693 va=1.4758 acc=0.459
2026-04-23 23:02:17,752 INFO Regime epoch  8/50 — tr=0.7596 va=1.4741 acc=0.473
2026-04-23 23:02:18,408 INFO Regime epoch  9/50 — tr=0.7523 va=1.4712 acc=0.490
2026-04-23 23:02:19,100 INFO Regime epoch 10/50 — tr=0.7464 va=1.4607 acc=0.501 per_class={'TRENDING': 0.494, 'RANGING': 0.012, 'CONSOLIDATING': 0.855, 'VOLATILE': 0.913}
2026-04-23 23:02:19,781 INFO Regime epoch 11/50 — tr=0.7412 va=1.4512 acc=0.517
2026-04-23 23:02:20,443 INFO Regime epoch 12/50 — tr=0.7370 va=1.4393 acc=0.527
2026-04-23 23:02:21,087 INFO Regime epoch 13/50 — tr=0.7329 va=1.4282 acc=0.539
2026-04-23 23:02:21,750 INFO Regime epoch 14/50 — tr=0.7296 va=1.4164 acc=0.549
2026-04-23 23:02:22,454 INFO Regime epoch 15/50 — tr=0.7267 va=1.4066 acc=0.559 per_class={'TRENDING': 0.627, 'RANGING': 0.026, 'CONSOLIDATING': 0.882, 'VOLATILE': 0.905}
2026-04-23 23:02:23,079 INFO Regime epoch 16/50 — tr=0.7235 va=1.4004 acc=0.561
2026-04-23 23:02:23,717 INFO Regime epoch 17/50 — tr=0.7215 va=1.3944 acc=0.568
2026-04-23 23:02:24,366 INFO Regime epoch 18/50 — tr=0.7196 va=1.3863 acc=0.576
2026-04-23 23:02:25,017 INFO Regime epoch 19/50 — tr=0.7170 va=1.3822 acc=0.576
2026-04-23 23:02:25,717 INFO Regime epoch 20/50 — tr=0.7153 va=1.3765 acc=0.582 per_class={'TRENDING': 0.663, 'RANGING': 0.036, 'CONSOLIDATING': 0.916, 'VOLATILE': 0.913}
2026-04-23 23:02:26,367 INFO Regime epoch 21/50 — tr=0.7141 va=1.3734 acc=0.582
2026-04-23 23:02:27,004 INFO Regime epoch 22/50 — tr=0.7123 va=1.3672 acc=0.590
2026-04-23 23:02:27,658 INFO Regime epoch 23/50 — tr=0.7114 va=1.3638 acc=0.591
2026-04-23 23:02:28,295 INFO Regime epoch 24/50 — tr=0.7104 va=1.3596 acc=0.590
2026-04-23 23:02:28,985 INFO Regime epoch 25/50 — tr=0.7090 va=1.3610 acc=0.591 per_class={'TRENDING': 0.662, 'RANGING': 0.052, 'CONSOLIDATING': 0.941, 'VOLATILE': 0.921}
2026-04-23 23:02:29,651 INFO Regime epoch 26/50 — tr=0.7082 va=1.3541 acc=0.595
2026-04-23 23:02:30,290 INFO Regime epoch 27/50 — tr=0.7077 va=1.3511 acc=0.597
2026-04-23 23:02:30,938 INFO Regime epoch 28/50 — tr=0.7070 va=1.3533 acc=0.595
2026-04-23 23:02:31,604 INFO Regime epoch 29/50 — tr=0.7061 va=1.3490 acc=0.600
2026-04-23 23:02:32,303 INFO Regime epoch 30/50 — tr=0.7057 va=1.3524 acc=0.598 per_class={'TRENDING': 0.672, 'RANGING': 0.059, 'CONSOLIDATING': 0.946, 'VOLATILE': 0.922}
2026-04-23 23:02:32,959 INFO Regime epoch 31/50 — tr=0.7054 va=1.3434 acc=0.607
2026-04-23 23:02:33,601 INFO Regime epoch 32/50 — tr=0.7049 va=1.3441 acc=0.603
2026-04-23 23:02:34,260 INFO Regime epoch 33/50 — tr=0.7045 va=1.3451 acc=0.603
2026-04-23 23:02:34,901 INFO Regime epoch 34/50 — tr=0.7043 va=1.3445 acc=0.603
2026-04-23 23:02:35,600 INFO Regime epoch 35/50 — tr=0.7043 va=1.3434 acc=0.606 per_class={'TRENDING': 0.691, 'RANGING': 0.062, 'CONSOLIDATING': 0.951, 'VOLATILE': 0.915}
2026-04-23 23:02:36,223 INFO Regime epoch 36/50 — tr=0.7039 va=1.3440 acc=0.605
2026-04-23 23:02:36,869 INFO Regime epoch 37/50 — tr=0.7037 va=1.3392 acc=0.605
2026-04-23 23:02:37,525 INFO Regime epoch 38/50 — tr=0.7033 va=1.3433 acc=0.604
2026-04-23 23:02:38,173 INFO Regime epoch 39/50 — tr=0.7033 va=1.3376 acc=0.608
2026-04-23 23:02:38,857 INFO Regime epoch 40/50 — tr=0.7032 va=1.3454 acc=0.603 per_class={'TRENDING': 0.681, 'RANGING': 0.063, 'CONSOLIDATING': 0.948, 'VOLATILE': 0.922}
2026-04-23 23:02:39,537 INFO Regime epoch 41/50 — tr=0.7029 va=1.3422 acc=0.604
2026-04-23 23:02:40,172 INFO Regime epoch 42/50 — tr=0.7027 va=1.3398 acc=0.606
2026-04-23 23:02:40,824 INFO Regime epoch 43/50 — tr=0.7029 va=1.3364 acc=0.609
2026-04-23 23:02:41,466 INFO Regime epoch 44/50 — tr=0.7029 va=1.3394 acc=0.606
2026-04-23 23:02:42,185 INFO Regime epoch 45/50 — tr=0.7028 va=1.3347 acc=0.610 per_class={'TRENDING': 0.698, 'RANGING': 0.067, 'CONSOLIDATING': 0.956, 'VOLATILE': 0.912}
2026-04-23 23:02:42,812 INFO Regime epoch 46/50 — tr=0.7027 va=1.3352 acc=0.610
2026-04-23 23:02:43,431 INFO Regime epoch 47/50 — tr=0.7029 va=1.3397 acc=0.606
2026-04-23 23:02:44,052 INFO Regime epoch 48/50 — tr=0.7029 va=1.3407 acc=0.604
2026-04-23 23:02:44,676 INFO Regime epoch 49/50 — tr=0.7027 va=1.3380 acc=0.609
2026-04-23 23:02:45,363 INFO Regime epoch 50/50 — tr=0.7026 va=1.3402 acc=0.606 per_class={'TRENDING': 0.687, 'RANGING': 0.067, 'CONSOLIDATING': 0.952, 'VOLATILE': 0.917}
2026-04-23 23:02:45,409 WARNING RegimeClassifier accuracy 0.61 < 0.65 threshold
2026-04-23 23:02:45,412 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-23 23:02:45,412 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-23 23:02:45,540 INFO Regime LTF complete: acc=0.610, n=401471
2026-04-23 23:02:45,543 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-23 23:02:46,039 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-23 23:02:46,043 INFO Regime[1H] persistence (avg bars/run) on XAUUSD 1H:
{0: 9.355291913830783, 1: 6.096303199751476, 2: 5.598885793871866, 3: 6.771351107094442}
2026-04-23 23:02:46,046 INFO Regime[1H] return separation on XAUUSD 1H:
{0: 2.149062259083323e-05, 1: -3.254493608872846e-06, 2: -3.039756158146958e-06, 3: 1.8102764458593193e-05}
2026-04-23 23:02:46,060 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-23 23:02:46,060 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-23 23:02:46,060 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-23 23:02:46,060 INFO === VectorStore: building similarity indices ===
2026-04-23 23:02:46,061 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-23 23:02:46,061 INFO Retrain complete.
2026-04-23 23:02:48,576 INFO Model regime: SUCCESS
2026-04-23 23:02:48,576 INFO --- Training gru ---
2026-04-23 23:02:48,577 INFO Running retrain --model gru
2026-04-23 23:02:48,923 INFO retrain environment: KAGGLE
2026-04-23 23:02:50,637 INFO Device: CUDA (2 GPU(s))
2026-04-23 23:02:50,648 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-23 23:02:50,649 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-23 23:02:50,650 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-23 23:02:50,785 INFO NumExpr defaulting to 4 threads.
2026-04-23 23:02:50,973 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-23 23:02:50,973 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-23 23:02:50,973 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-23 23:02:51,259 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-23 23:02:51,481 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-23 23:02:51,482 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:51,558 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:51,632 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:51,703 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:51,773 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:51,844 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:51,911 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:51,979 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:52,049 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:52,117 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:52,202 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-23 23:02:52,261 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-23 23:02:52,262 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260423_230252
2026-04-23 23:02:52,266 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-04-23 23:02:52,382 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:52,383 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:52,398 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:52,406 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:52,407 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-23 23:02:52,407 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-23 23:02:52,407 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-23 23:02:52,408 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:52,481 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-23 23:02:52,483 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:52,700 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-23 23:02:52,727 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:52,993 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:53,119 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:53,213 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:53,407 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:53,408 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:53,424 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:53,431 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:53,432 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:53,504 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-23 23:02:53,506 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:53,727 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-23 23:02:53,742 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:54,001 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:54,122 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:54,213 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:54,393 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:54,394 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:54,411 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:54,419 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:54,420 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:54,496 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-23 23:02:54,498 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:54,719 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-23 23:02:54,734 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:54,999 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:55,121 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:55,211 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:55,391 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:55,392 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:55,408 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:55,415 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:55,416 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:55,489 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-23 23:02:55,491 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:55,719 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-23 23:02:55,740 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:55,997 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:56,119 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:56,210 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:56,387 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:56,388 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:56,405 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:56,413 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:56,414 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:56,486 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-23 23:02:56,488 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:56,709 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-23 23:02:56,724 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:56,974 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:57,095 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:57,184 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:57,360 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:57,361 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:57,378 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:57,386 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:57,387 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:57,464 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-23 23:02:57,466 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:57,694 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-23 23:02:57,709 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:57,963 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:58,088 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:58,177 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:58,334 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-23 23:02:58,335 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-23 23:02:58,349 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-23 23:02:58,356 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-23 23:02:58,357 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:58,426 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-23 23:02:58,428 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:58,640 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-23 23:02:58,652 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:58,914 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:59,034 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:59,125 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:59,324 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:59,325 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:59,340 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:59,348 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:02:59,348 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:59,418 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-23 23:02:59,420 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:59,641 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-23 23:02:59,658 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:02:59,918 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:03:00,042 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:03:00,135 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:03:00,309 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:03:00,310 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:03:00,325 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:03:00,333 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:03:00,334 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:03:00,403 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-23 23:03:00,405 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:03:00,629 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-23 23:03:00,644 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:03:00,901 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:03:01,023 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:03:01,113 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:03:01,294 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:03:01,295 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:03:01,312 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:03:01,319 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-23 23:03:01,320 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:03:01,392 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-23 23:03:01,393 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:03:01,612 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-23 23:03:01,627 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:03:01,887 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:03:02,014 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:03:02,108 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-23 23:03:02,389 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-23 23:03:02,391 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-23 23:03:02,408 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-23 23:03:02,417 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-23 23:03:02,418 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-23 23:03:02,563 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-23 23:03:02,566 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-23 23:03:03,037 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-23 23:03:03,082 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-23 23:03:03,591 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-23 23:03:03,787 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-23 23:03:03,914 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-23 23:03:04,025 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-23 23:03:04,025 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-23 23:03:04,025 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-23 23:07:28,601 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-23 23:07:28,602 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-23 23:07:29,939 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-23 23:07:53,858 INFO train_multi TF=ALL epoch 1/50 train=0.6052 val=0.6125
2026-04-23 23:07:53,863 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-23 23:07:53,864 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-23 23:07:53,864 INFO train_multi TF=ALL: new best val=0.6125 — saved
2026-04-23 23:08:11,726 INFO train_multi TF=ALL epoch 2/50 train=0.6052 val=0.6129
2026-04-23 23:08:29,680 INFO train_multi TF=ALL epoch 3/50 train=0.6051 val=0.6126
2026-04-23 23:08:47,398 INFO train_multi TF=ALL epoch 4/50 train=0.6048 val=0.6131
2026-04-23 23:09:05,093 INFO train_multi TF=ALL epoch 5/50 train=0.6044 val=0.6134
2026-04-23 23:09:22,782 INFO train_multi TF=ALL epoch 6/50 train=0.6044 val=0.6127
2026-04-23 23:09:22,782 INFO train_multi TF=ALL early stop at epoch 6
2026-04-23 23:09:22,917 INFO === VectorStore: building similarity indices ===
2026-04-23 23:09:22,917 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-23 23:09:22,917 INFO Retrain complete.
2026-04-23 23:09:24,834 INFO Model gru: SUCCESS
2026-04-23 23:09:24,834 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-23 23:09:24,834 WARNING   [MISSING] regime_classifier → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_classifier.pkl
2026-04-23 23:09:24,834 INFO   [OK] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-23 23:09:24,834 WARNING   [MISSING] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-23 23:09:24,834 WARNING Missing weights: ['regime_classifier', 'rl_ppo'] — run retrain_incremental.py for each
2026-04-23 23:09:24,835 INFO Saved 18 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  Cleared journal for fresh Round 1 run
  START Round 1 - Backtest (val)
2026-04-23 23:09:25,368 INFO === STEP 6: BACKTEST + REINFORCED TRAINING (3 rounds, window=round1) ===
2026-04-23 23:09:25,369 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-04-23 23:09:25,369 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-23 23:09:25,369 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)

======================================================================
  BACKTEST + REINFORCED TRAINING COMPLETE  (0 rounds)
======================================================================
  Round     Trades       WR      PF   Sharpe
  ------------------------------------------

  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 0 entries