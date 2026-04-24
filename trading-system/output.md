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
2026-04-24 00:06:25,646 INFO Loading feature-engineered data...
2026-04-24 00:06:26,247 INFO Loaded 221743 rows, 202 features
2026-04-24 00:06:26,248 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-24 00:06:26,250 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-24 00:06:26,250 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-24 00:06:26,250 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-24 00:06:26,250 INFO No leakage confirmed: train < val < test timestamps

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
2026-04-24 00:06:28,573 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-24 00:06:28,573 INFO --- Training regime ---
2026-04-24 00:06:28,574 INFO Running retrain --model regime
2026-04-24 00:06:28,758 INFO retrain environment: KAGGLE
2026-04-24 00:06:30,423 INFO Device: CUDA (2 GPU(s))
2026-04-24 00:06:30,434 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 00:06:30,434 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 00:06:30,435 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-24 00:06:30,582 INFO NumExpr defaulting to 4 threads.
2026-04-24 00:06:30,790 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-24 00:06:30,790 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 00:06:30,790 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 00:06:30,991 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-24 00:06:30,993 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,081 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,155 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,229 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,300 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,375 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,443 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,513 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,588 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,665 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,753 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:06:31,811 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-24 00:06:31,826 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,827 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,843 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,844 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,865 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,867 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,882 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,884 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,900 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,903 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,917 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,920 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,934 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,936 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,951 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,954 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,968 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,972 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,987 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:31,990 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:06:32,007 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:06:32,014 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:06:33,152 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-24 00:06:55,436 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-24 00:06:55,441 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias)
2026-04-24 00:06:55,441 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-24 00:07:06,040 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-24 00:07:06,041 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias)
2026-04-24 00:07:06,041 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-24 00:07:13,635 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-24 00:07:13,639 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias)
2026-04-24 00:07:13,639 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-24 00:08:24,092 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-24 00:08:24,095 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour)
2026-04-24 00:08:24,095 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-24 00:08:55,095 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-24 00:08:55,097 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour)
2026-04-24 00:08:55,097 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-24 00:09:16,827 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-24 00:09:16,829 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour)
2026-04-24 00:09:16,937 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-24 00:09:16,939 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:09:16,940 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:09:16,941 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:09:16,942 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:09:16,943 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:09:16,944 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:09:16,945 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:09:16,946 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:09:16,947 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:09:16,948 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:09:16,949 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:09:17,084 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:17,128 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:17,129 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:17,129 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:17,138 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:17,139 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:09:19,847 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-24 00:09:19,848 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-24 00:09:20,029 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:20,063 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:20,064 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:20,064 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:20,072 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:20,073 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:09:22,781 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-24 00:09:22,782 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-24 00:09:22,961 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:23,000 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:23,000 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:23,001 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:23,009 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:23,010 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:09:25,713 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-24 00:09:25,714 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-24 00:09:25,877 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:25,914 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:25,915 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:25,915 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:25,924 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:25,925 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:09:28,673 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-24 00:09:28,674 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-24 00:09:28,855 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:28,892 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:28,893 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:28,893 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:28,903 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:28,904 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:09:31,679 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-24 00:09:31,680 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-24 00:09:31,851 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:31,886 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:31,887 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:31,887 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:31,896 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:31,897 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:09:34,658 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-24 00:09:34,660 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-24 00:09:34,815 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:09:34,842 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:09:34,842 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:09:34,843 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:09:34,853 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:09:34,854 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:09:37,640 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-24 00:09:37,641 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-24 00:09:37,809 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:37,842 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:37,843 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:37,843 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:37,852 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:37,853 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:09:40,647 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-24 00:09:40,649 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-24 00:09:40,823 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:40,857 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:40,858 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:40,859 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:40,867 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:40,869 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:09:43,597 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-24 00:09:43,599 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-24 00:09:43,768 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:43,801 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:43,802 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:43,802 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:43,811 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:09:43,812 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:09:46,531 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-24 00:09:46,532 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-24 00:09:46,801 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:09:46,863 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:09:46,864 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:09:46,864 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:09:46,875 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:09:46,876 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:09:53,287 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-24 00:09:53,289 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-24 00:09:53,472 INFO RegimeClassifier[mode=htf_bias]: 103290 samples, classes={'BIAS_UP': 18622, 'BIAS_DOWN': 18286, 'BIAS_NEUTRAL': 66382}, device=cuda
2026-04-24 00:09:53,472 INFO RegimeClassifier: sample weights — mean=0.360  ambiguous(<0.4)=69.5%
2026-04-24 00:09:53,750 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-04-24 00:09:53,750 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-24 00:09:58,392 INFO Regime epoch  1/50 — tr=0.7157 va=1.3310 acc=0.266 per_class={'BIAS_UP': 0.837, 'BIAS_DOWN': 0.013, 'BIAS_NEUTRAL': 0.126}
2026-04-24 00:09:58,562 INFO Regime epoch  2/50 — tr=0.7010 va=1.3013 acc=0.336
2026-04-24 00:09:58,735 INFO Regime epoch  3/50 — tr=0.6807 va=1.2506 acc=0.403
2026-04-24 00:09:58,903 INFO Regime epoch  4/50 — tr=0.6465 va=1.1827 acc=0.479
2026-04-24 00:09:59,093 INFO Regime epoch  5/50 — tr=0.6065 va=1.1121 acc=0.520 per_class={'BIAS_UP': 0.879, 'BIAS_DOWN': 0.893, 'BIAS_NEUTRAL': 0.296}
2026-04-24 00:09:59,273 INFO Regime epoch  6/50 — tr=0.5722 va=1.0331 acc=0.535
2026-04-24 00:09:59,461 INFO Regime epoch  7/50 — tr=0.5439 va=0.9622 acc=0.538
2026-04-24 00:09:59,646 INFO Regime epoch  8/50 — tr=0.5236 va=0.9187 acc=0.536
2026-04-24 00:09:59,822 INFO Regime epoch  9/50 — tr=0.5113 va=0.8921 acc=0.539
2026-04-24 00:10:00,016 INFO Regime epoch 10/50 — tr=0.5040 va=0.8882 acc=0.539 per_class={'BIAS_UP': 0.981, 'BIAS_DOWN': 0.994, 'BIAS_NEUTRAL': 0.266}
2026-04-24 00:10:00,196 INFO Regime epoch 11/50 — tr=0.4969 va=0.8789 acc=0.547
2026-04-24 00:10:00,376 INFO Regime epoch 12/50 — tr=0.4929 va=0.8815 acc=0.546
2026-04-24 00:10:00,560 INFO Regime epoch 13/50 — tr=0.4889 va=0.8802 acc=0.550
2026-04-24 00:10:00,738 INFO Regime epoch 14/50 — tr=0.4862 va=0.8779 acc=0.551
2026-04-24 00:10:00,925 INFO Regime epoch 15/50 — tr=0.4833 va=0.8799 acc=0.553 per_class={'BIAS_UP': 0.989, 'BIAS_DOWN': 0.997, 'BIAS_NEUTRAL': 0.285}
2026-04-24 00:10:01,090 INFO Regime epoch 16/50 — tr=0.4810 va=0.8771 acc=0.556
2026-04-24 00:10:01,270 INFO Regime epoch 17/50 — tr=0.4790 va=0.8784 acc=0.558
2026-04-24 00:10:01,441 INFO Regime epoch 18/50 — tr=0.4782 va=0.8765 acc=0.557
2026-04-24 00:10:01,634 INFO Regime epoch 19/50 — tr=0.4764 va=0.8757 acc=0.556
2026-04-24 00:10:01,844 INFO Regime epoch 20/50 — tr=0.4750 va=0.8759 acc=0.563 per_class={'BIAS_UP': 0.992, 'BIAS_DOWN': 0.996, 'BIAS_NEUTRAL': 0.3}
2026-04-24 00:10:02,022 INFO Regime epoch 21/50 — tr=0.4740 va=0.8734 acc=0.564
2026-04-24 00:10:02,214 INFO Regime epoch 22/50 — tr=0.4733 va=0.8768 acc=0.565
2026-04-24 00:10:02,435 INFO Regime epoch 23/50 — tr=0.4720 va=0.8748 acc=0.569
2026-04-24 00:10:02,626 INFO Regime epoch 24/50 — tr=0.4710 va=0.8757 acc=0.571
2026-04-24 00:10:02,813 INFO Regime epoch 25/50 — tr=0.4709 va=0.8741 acc=0.570 per_class={'BIAS_UP': 0.994, 'BIAS_DOWN': 0.996, 'BIAS_NEUTRAL': 0.311}
2026-04-24 00:10:02,987 INFO Regime epoch 26/50 — tr=0.4696 va=0.8744 acc=0.571
2026-04-24 00:10:03,173 INFO Regime epoch 27/50 — tr=0.4695 va=0.8732 acc=0.570
2026-04-24 00:10:03,343 INFO Regime epoch 28/50 — tr=0.4693 va=0.8731 acc=0.577
2026-04-24 00:10:03,512 INFO Regime epoch 29/50 — tr=0.4684 va=0.8740 acc=0.573
2026-04-24 00:10:03,709 INFO Regime epoch 30/50 — tr=0.4679 va=0.8729 acc=0.573 per_class={'BIAS_UP': 0.995, 'BIAS_DOWN': 0.997, 'BIAS_NEUTRAL': 0.315}
2026-04-24 00:10:03,887 INFO Regime epoch 31/50 — tr=0.4679 va=0.8729 acc=0.574
2026-04-24 00:10:04,053 INFO Regime epoch 32/50 — tr=0.4678 va=0.8721 acc=0.574
2026-04-24 00:10:04,227 INFO Regime epoch 33/50 — tr=0.4670 va=0.8735 acc=0.580
2026-04-24 00:10:04,397 INFO Regime epoch 34/50 — tr=0.4674 va=0.8729 acc=0.578
2026-04-24 00:10:04,581 INFO Regime epoch 35/50 — tr=0.4669 va=0.8730 acc=0.580 per_class={'BIAS_UP': 0.995, 'BIAS_DOWN': 0.998, 'BIAS_NEUTRAL': 0.325}
2026-04-24 00:10:04,749 INFO Regime epoch 36/50 — tr=0.4664 va=0.8730 acc=0.576
2026-04-24 00:10:04,916 INFO Regime epoch 37/50 — tr=0.4664 va=0.8719 acc=0.575
2026-04-24 00:10:05,088 INFO Regime epoch 38/50 — tr=0.4666 va=0.8710 acc=0.581
2026-04-24 00:10:05,261 INFO Regime epoch 39/50 — tr=0.4662 va=0.8721 acc=0.577
2026-04-24 00:10:05,451 INFO Regime epoch 40/50 — tr=0.4661 va=0.8724 acc=0.577 per_class={'BIAS_UP': 0.995, 'BIAS_DOWN': 0.998, 'BIAS_NEUTRAL': 0.321}
2026-04-24 00:10:05,629 INFO Regime epoch 41/50 — tr=0.4657 va=0.8728 acc=0.578
2026-04-24 00:10:05,795 INFO Regime epoch 42/50 — tr=0.4659 va=0.8719 acc=0.581
2026-04-24 00:10:05,966 INFO Regime epoch 43/50 — tr=0.4660 va=0.8712 acc=0.579
2026-04-24 00:10:06,136 INFO Regime epoch 44/50 — tr=0.4659 va=0.8717 acc=0.579
2026-04-24 00:10:06,318 INFO Regime epoch 45/50 — tr=0.4660 va=0.8722 acc=0.582 per_class={'BIAS_UP': 0.995, 'BIAS_DOWN': 0.998, 'BIAS_NEUTRAL': 0.328}
2026-04-24 00:10:06,483 INFO Regime epoch 46/50 — tr=0.4653 va=0.8718 acc=0.581
2026-04-24 00:10:06,652 INFO Regime epoch 47/50 — tr=0.4658 va=0.8728 acc=0.579
2026-04-24 00:10:06,820 INFO Regime epoch 48/50 — tr=0.4655 va=0.8724 acc=0.580
2026-04-24 00:10:06,820 INFO Regime early stop at epoch 48 (no_improve=10)
2026-04-24 00:10:06,834 WARNING RegimeClassifier accuracy 0.58 < 0.65 threshold
2026-04-24 00:10:06,838 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-24 00:10:06,838 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-24 00:10:06,969 INFO Regime HTF complete: acc=0.581, n=103290
2026-04-24 00:10:06,971 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:10:07,138 INFO Rule labels LTF_BEHAVIOUR [4H]: {'TRENDING': 8970, 'RANGING': 4592, 'CONSOLIDATING': 2247, 'VOLATILE': 4008}  ambiguous=4539 (total=19817)  short_runs_zeroed=570
2026-04-24 00:10:07,144 INFO Regime[4H] persistence (avg bars/run) on XAUUSD 4H:
{0: 12.121621621621621, 1: 6.4858757062146895, 2: 5.575682382133995, 3: 10.329896907216495}
2026-04-24 00:10:07,146 INFO Regime[4H] return separation on XAUUSD 4H:
{0: 5.9502411776840314e-05, 1: 5.664222849552333e-05, 2: -6.138442068010657e-05, 3: 4.239843820627722e-05}
2026-04-24 00:10:07,146 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-24 00:10:07,148 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:10:07,149 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:10:07,151 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:10:07,152 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:10:07,154 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:10:07,155 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:10:07,157 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:10:07,158 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:10:07,160 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:10:07,161 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:10:07,164 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:10:07,175 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:07,177 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:07,178 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:07,178 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:07,179 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:07,180 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:10:17,378 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-24 00:10:17,381 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-24 00:10:17,519 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:17,521 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:17,522 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:17,522 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:17,523 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:17,525 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:10:27,408 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-24 00:10:27,411 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-24 00:10:27,542 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:27,546 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:27,547 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:27,547 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:27,548 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:27,549 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:10:37,553 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-24 00:10:37,556 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-24 00:10:37,694 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:37,697 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:37,697 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:37,698 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:37,698 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:37,700 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:10:47,883 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-24 00:10:47,885 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-24 00:10:48,025 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:48,027 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:48,028 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:48,028 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:48,029 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:48,031 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:10:58,097 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-24 00:10:58,100 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-24 00:10:58,244 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:58,247 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:58,247 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:58,248 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:58,248 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:10:58,250 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:11:08,361 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-24 00:11:08,364 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-24 00:11:08,506 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:11:08,507 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:11:08,508 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:11:08,509 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:11:08,509 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:11:08,511 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:11:18,642 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-24 00:11:18,645 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-24 00:11:18,791 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:11:18,793 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:11:18,794 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:11:18,794 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:11:18,795 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:11:18,796 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:11:28,725 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-24 00:11:28,728 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-24 00:11:28,873 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:11:28,875 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:11:28,876 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:11:28,876 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:11:28,877 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:11:28,879 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:11:38,837 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-24 00:11:38,840 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-24 00:11:38,978 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:11:38,981 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:11:38,981 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:11:38,982 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:11:38,982 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:11:38,984 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:11:48,934 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-24 00:11:48,937 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-24 00:11:49,085 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:11:49,088 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:11:49,090 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:11:49,090 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:11:49,091 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:11:49,094 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:12:11,544 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-24 00:12:11,550 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-24 00:12:11,909 INFO RegimeClassifier[mode=ltf_behaviour]: 401471 samples, classes={'TRENDING': 160094, 'RANGING': 105286, 'CONSOLIDATING': 53524, 'VOLATILE': 82567}, device=cuda
2026-04-24 00:12:11,910 INFO RegimeClassifier: sample weights — mean=0.505  ambiguous(<0.4)=33.0%
2026-04-24 00:12:11,912 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-04-24 00:12:11,913 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-24 00:12:12,656 INFO Regime epoch  1/50 — tr=0.9642 va=1.7937 acc=0.196 per_class={'TRENDING': 0.145, 'RANGING': 0.093, 'CONSOLIDATING': 0.33, 'VOLATILE': 0.344}
2026-04-24 00:12:13,294 INFO Regime epoch  2/50 — tr=0.9283 va=1.6912 acc=0.334
2026-04-24 00:12:13,937 INFO Regime epoch  3/50 — tr=0.8759 va=1.5986 acc=0.395
2026-04-24 00:12:14,572 INFO Regime epoch  4/50 — tr=0.8285 va=1.5438 acc=0.422
2026-04-24 00:12:15,259 INFO Regime epoch  5/50 — tr=0.7996 va=1.5137 acc=0.432 per_class={'TRENDING': 0.323, 'RANGING': 0.008, 'CONSOLIDATING': 0.841, 'VOLATILE': 0.929}
2026-04-24 00:12:15,894 INFO Regime epoch  6/50 — tr=0.7808 va=1.4941 acc=0.442
2026-04-24 00:12:16,536 INFO Regime epoch  7/50 — tr=0.7678 va=1.4911 acc=0.450
2026-04-24 00:12:17,210 INFO Regime epoch  8/50 — tr=0.7580 va=1.4854 acc=0.462
2026-04-24 00:12:17,860 INFO Regime epoch  9/50 — tr=0.7505 va=1.4760 acc=0.480
2026-04-24 00:12:18,580 INFO Regime epoch 10/50 — tr=0.7440 va=1.4651 acc=0.496 per_class={'TRENDING': 0.473, 'RANGING': 0.012, 'CONSOLIDATING': 0.864, 'VOLATILE': 0.93}
2026-04-24 00:12:19,224 INFO Regime epoch 11/50 — tr=0.7395 va=1.4516 acc=0.510
2026-04-24 00:12:19,862 INFO Regime epoch 12/50 — tr=0.7352 va=1.4388 acc=0.526
2026-04-24 00:12:20,564 INFO Regime epoch 13/50 — tr=0.7313 va=1.4285 acc=0.537
2026-04-24 00:12:21,231 INFO Regime epoch 14/50 — tr=0.7283 va=1.4192 acc=0.545
2026-04-24 00:12:21,923 INFO Regime epoch 15/50 — tr=0.7254 va=1.4110 acc=0.552 per_class={'TRENDING': 0.595, 'RANGING': 0.026, 'CONSOLIDATING': 0.89, 'VOLATILE': 0.927}
2026-04-24 00:12:22,591 INFO Regime epoch 16/50 — tr=0.7230 va=1.4014 acc=0.559
2026-04-24 00:12:23,228 INFO Regime epoch 17/50 — tr=0.7210 va=1.3975 acc=0.562
2026-04-24 00:12:23,877 INFO Regime epoch 18/50 — tr=0.7191 va=1.3895 acc=0.567
2026-04-24 00:12:24,509 INFO Regime epoch 19/50 — tr=0.7174 va=1.3883 acc=0.568
2026-04-24 00:12:25,186 INFO Regime epoch 20/50 — tr=0.7158 va=1.3842 acc=0.572 per_class={'TRENDING': 0.63, 'RANGING': 0.037, 'CONSOLIDATING': 0.924, 'VOLATILE': 0.922}
2026-04-24 00:12:25,855 INFO Regime epoch 21/50 — tr=0.7146 va=1.3801 acc=0.576
2026-04-24 00:12:26,504 INFO Regime epoch 22/50 — tr=0.7134 va=1.3747 acc=0.580
2026-04-24 00:12:27,139 INFO Regime epoch 23/50 — tr=0.7121 va=1.3677 acc=0.586
2026-04-24 00:12:27,805 INFO Regime epoch 24/50 — tr=0.7114 va=1.3662 acc=0.586
2026-04-24 00:12:28,500 INFO Regime epoch 25/50 — tr=0.7102 va=1.3671 acc=0.588 per_class={'TRENDING': 0.658, 'RANGING': 0.043, 'CONSOLIDATING': 0.951, 'VOLATILE': 0.916}
2026-04-24 00:12:29,168 INFO Regime epoch 26/50 — tr=0.7095 va=1.3638 acc=0.589
2026-04-24 00:12:29,827 INFO Regime epoch 27/50 — tr=0.7089 va=1.3581 acc=0.593
2026-04-24 00:12:30,478 INFO Regime epoch 28/50 — tr=0.7082 va=1.3615 acc=0.592
2026-04-24 00:12:31,138 INFO Regime epoch 29/50 — tr=0.7077 va=1.3550 acc=0.593
2026-04-24 00:12:31,829 INFO Regime epoch 30/50 — tr=0.7071 va=1.3530 acc=0.597 per_class={'TRENDING': 0.673, 'RANGING': 0.056, 'CONSOLIDATING': 0.953, 'VOLATILE': 0.916}
2026-04-24 00:12:32,530 INFO Regime epoch 31/50 — tr=0.7067 va=1.3537 acc=0.597
2026-04-24 00:12:33,169 INFO Regime epoch 32/50 — tr=0.7065 va=1.3538 acc=0.596
2026-04-24 00:12:33,809 INFO Regime epoch 33/50 — tr=0.7058 va=1.3502 acc=0.598
2026-04-24 00:12:34,457 INFO Regime epoch 34/50 — tr=0.7058 va=1.3486 acc=0.600
2026-04-24 00:12:35,141 INFO Regime epoch 35/50 — tr=0.7051 va=1.3498 acc=0.600 per_class={'TRENDING': 0.674, 'RANGING': 0.059, 'CONSOLIDATING': 0.954, 'VOLATILE': 0.919}
2026-04-24 00:12:35,785 INFO Regime epoch 36/50 — tr=0.7052 va=1.3489 acc=0.600
2026-04-24 00:12:36,445 INFO Regime epoch 37/50 — tr=0.7052 va=1.3490 acc=0.602
2026-04-24 00:12:37,107 INFO Regime epoch 38/50 — tr=0.7048 va=1.3450 acc=0.602
2026-04-24 00:12:37,768 INFO Regime epoch 39/50 — tr=0.7047 va=1.3455 acc=0.602
2026-04-24 00:12:38,491 INFO Regime epoch 40/50 — tr=0.7046 va=1.3474 acc=0.603 per_class={'TRENDING': 0.684, 'RANGING': 0.058, 'CONSOLIDATING': 0.956, 'VOLATILE': 0.915}
2026-04-24 00:12:39,129 INFO Regime epoch 41/50 — tr=0.7043 va=1.3489 acc=0.602
2026-04-24 00:12:39,787 INFO Regime epoch 42/50 — tr=0.7046 va=1.3460 acc=0.601
2026-04-24 00:12:40,435 INFO Regime epoch 43/50 — tr=0.7044 va=1.3451 acc=0.604
2026-04-24 00:12:41,096 INFO Regime epoch 44/50 — tr=0.7041 va=1.3506 acc=0.599
2026-04-24 00:12:41,803 INFO Regime epoch 45/50 — tr=0.7039 va=1.3474 acc=0.602 per_class={'TRENDING': 0.681, 'RANGING': 0.061, 'CONSOLIDATING': 0.957, 'VOLATILE': 0.916}
2026-04-24 00:12:42,498 INFO Regime epoch 46/50 — tr=0.7040 va=1.3436 acc=0.603
2026-04-24 00:12:43,154 INFO Regime epoch 47/50 — tr=0.7042 va=1.3482 acc=0.601
2026-04-24 00:12:43,792 INFO Regime epoch 48/50 — tr=0.7040 va=1.3479 acc=0.602
2026-04-24 00:12:44,448 INFO Regime epoch 49/50 — tr=0.7040 va=1.3456 acc=0.601
2026-04-24 00:12:45,190 INFO Regime epoch 50/50 — tr=0.7042 va=1.3419 acc=0.605 per_class={'TRENDING': 0.69, 'RANGING': 0.06, 'CONSOLIDATING': 0.959, 'VOLATILE': 0.91}
2026-04-24 00:12:45,235 WARNING RegimeClassifier accuracy 0.60 < 0.65 threshold
2026-04-24 00:12:45,237 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-24 00:12:45,237 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-24 00:12:45,371 INFO Regime LTF complete: acc=0.605, n=401471
2026-04-24 00:12:45,374 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:12:45,871 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-24 00:12:45,875 INFO Regime[1H] persistence (avg bars/run) on XAUUSD 1H:
{0: 9.355291913830783, 1: 6.096303199751476, 2: 5.598885793871866, 3: 6.771351107094442}
2026-04-24 00:12:45,878 INFO Regime[1H] return separation on XAUUSD 1H:
{0: 2.149062259083323e-05, 1: -3.254493608872846e-06, 2: -3.039756158146958e-06, 3: 1.8102764458593193e-05}
2026-04-24 00:12:45,891 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-24 00:12:45,891 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 00:12:45,891 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 00:12:45,891 INFO === VectorStore: building similarity indices ===
2026-04-24 00:12:45,891 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-24 00:12:45,892 INFO Retrain complete.
2026-04-24 00:12:48,213 INFO Model regime: SUCCESS
2026-04-24 00:12:48,213 INFO --- Training gru ---
2026-04-24 00:12:48,214 INFO Running retrain --model gru
2026-04-24 00:12:48,545 INFO retrain environment: KAGGLE
2026-04-24 00:12:50,272 INFO Device: CUDA (2 GPU(s))
2026-04-24 00:12:50,283 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 00:12:50,283 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 00:12:50,284 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-24 00:12:50,420 INFO NumExpr defaulting to 4 threads.
2026-04-24 00:12:50,611 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-24 00:12:50,612 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 00:12:50,612 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 00:12:50,889 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-24 00:12:51,131 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-24 00:12:51,133 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:51,214 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:51,289 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:51,365 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:51,439 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:51,512 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:51,582 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:51,656 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:51,729 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:51,808 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:51,899 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:12:51,962 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-24 00:12:51,964 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260424_001251
2026-04-24 00:12:51,968 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-04-24 00:12:52,088 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:52,088 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:52,104 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:52,118 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:52,119 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-24 00:12:52,120 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 00:12:52,120 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 00:12:52,121 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:52,219 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-24 00:12:52,221 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:52,482 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-24 00:12:52,510 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:52,781 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:52,908 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:53,004 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:53,201 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:53,202 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:53,218 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:53,227 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:53,228 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:53,312 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-24 00:12:53,314 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:53,586 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-24 00:12:53,602 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:53,874 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:54,001 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:54,098 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:54,283 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:54,284 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:54,301 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:54,310 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:54,310 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:54,386 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-24 00:12:54,387 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:54,611 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-24 00:12:54,625 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:54,888 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:55,020 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:55,117 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:55,311 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:55,312 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:55,328 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:55,336 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:55,337 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:55,408 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-24 00:12:55,410 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:55,625 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-24 00:12:55,646 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:55,914 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:56,038 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:56,132 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:56,318 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:56,319 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:56,336 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:56,344 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:56,345 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:56,419 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-24 00:12:56,421 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:56,656 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-24 00:12:56,672 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:56,950 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:57,076 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:57,173 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:57,354 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:57,355 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:57,371 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:57,379 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:57,380 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:57,460 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-24 00:12:57,462 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:57,703 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-24 00:12:57,719 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:57,999 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:58,128 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:58,225 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:58,395 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:12:58,396 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:12:58,411 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:12:58,418 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:12:58,419 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:58,494 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-24 00:12:58,495 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:58,737 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-24 00:12:58,749 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:59,016 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:59,142 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:59,240 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:59,421 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:59,421 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:59,437 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:59,445 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:12:59,446 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:59,517 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-24 00:12:59,519 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:12:59,758 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-24 00:12:59,775 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:13:00,049 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:13:00,176 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:13:00,275 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:13:00,454 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:13:00,454 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:13:00,470 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:13:00,477 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:13:00,478 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:13:00,549 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-24 00:13:00,551 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:13:00,779 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-24 00:13:00,796 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:13:01,062 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:13:01,192 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:13:01,291 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:13:01,481 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:13:01,482 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:13:01,498 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:13:01,505 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:13:01,506 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:13:01,576 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-24 00:13:01,577 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:13:01,802 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-24 00:13:01,817 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:13:02,092 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:13:02,232 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:13:02,331 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:13:02,642 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:13:02,643 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:13:02,661 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:13:02,671 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:13:02,672 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:13:02,827 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-24 00:13:02,830 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:13:03,319 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-24 00:13:03,363 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:13:03,874 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:13:04,065 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:13:04,194 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:13:04,309 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-24 00:13:04,309 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-24 00:13:04,309 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-24 00:17:33,387 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-24 00:17:33,388 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-24 00:17:34,715 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-24 00:17:58,565 INFO train_multi TF=ALL epoch 1/50 train=0.6048 val=0.6128
2026-04-24 00:17:58,570 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-24 00:17:58,570 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-24 00:17:58,570 INFO train_multi TF=ALL: new best val=0.6128 — saved
2026-04-24 00:18:16,415 INFO train_multi TF=ALL epoch 2/50 train=0.6046 val=0.6128
2026-04-24 00:18:16,420 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-24 00:18:16,420 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-24 00:18:16,420 INFO train_multi TF=ALL: new best val=0.6128 — saved
2026-04-24 00:18:34,088 INFO train_multi TF=ALL epoch 3/50 train=0.6048 val=0.6127
2026-04-24 00:18:34,093 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-24 00:18:34,093 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-24 00:18:34,093 INFO train_multi TF=ALL: new best val=0.6127 — saved
2026-04-24 00:18:51,901 INFO train_multi TF=ALL epoch 4/50 train=0.6048 val=0.6141
2026-04-24 00:19:09,913 INFO train_multi TF=ALL epoch 5/50 train=0.6044 val=0.6132
2026-04-24 00:19:27,652 INFO train_multi TF=ALL epoch 6/50 train=0.6042 val=0.6130
2026-04-24 00:19:45,837 INFO train_multi TF=ALL epoch 7/50 train=0.6039 val=0.6137
2026-04-24 00:20:03,808 INFO train_multi TF=ALL epoch 8/50 train=0.6040 val=0.6137
2026-04-24 00:20:03,808 INFO train_multi TF=ALL early stop at epoch 8
2026-04-24 00:20:03,949 INFO === VectorStore: building similarity indices ===
2026-04-24 00:20:03,950 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-24 00:20:03,950 INFO Retrain complete.
2026-04-24 00:20:05,896 INFO Model gru: SUCCESS
2026-04-24 00:20:05,897 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-24 00:20:05,897 WARNING   [MISSING] regime_classifier → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_classifier.pkl
2026-04-24 00:20:05,897 INFO   [OK] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-24 00:20:05,897 WARNING   [MISSING] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-24 00:20:05,897 WARNING Missing weights: ['regime_classifier', 'rl_ppo'] — run retrain_incremental.py for each
2026-04-24 00:20:05,898 INFO Saved 18 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  Cleared journal for fresh Round 1 run
  START Round 1 - Backtest (val)
2026-04-24 00:20:06,431 INFO === STEP 6: BACKTEST + REINFORCED TRAINING (3 rounds, window=round1) ===
2026-04-24 00:20:06,432 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-04-24 00:20:06,432 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-24 00:20:06,432 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)

======================================================================
  BACKTEST + REINFORCED TRAINING COMPLETE  (0 rounds)
======================================================================
  Round     Trades       WR      PF   Sharpe
  ------------------------------------------

  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 0 entries

=== Round 1 → Retrain Quality + RL ===
  START Round 1 - Quality+RL retrain
2026-04-24 00:20:07,001 ERROR Backtest failed (rc=1):
Traceback (most recent call last):
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 1741, in <module>
    main()
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 1543, in main
    from config.settings import Settings
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../config/settings.py", line 6, in <module>
    from pydantic_settings import BaseSettings
  File "/usr/local/lib/python3.12/dist-packages/pydantic_settings/__init__.py", line 1, in <module>
    from .main import BaseSettings, SettingsConfigDict
  File "/usr/local/lib/python3.12/dist-packages/pydantic_settings/main.py", line 9, in <module>
    from pydantic.main import BaseModel
  File "/usr/local/lib/python3.12/dist-packages/pydantic/main.py", line 15, in <module>
    from ._internal import (
  File "/usr/local/lib/python3.12/dist-packages/pydantic/_internal/_decorators.py", line 15, in <module>
    from ._core_utils import get_type_ref
  File "/usr/local/lib/python3.12/dist-packages/pydantic/_internal/_core_utils.py", line 15, in <module>
    from pydantic_core import validate_core_schema as _validate_core_schema
ImportError: cannot import name 'validate_core_schema' from 'pydantic_core' (/usr/local/lib/python3.12/dist-packages/pydantic_core/__init__.py)

2026-04-24 00:20:07,001 ERROR Round 1 backtest failed — stopping reinforcement loop: Traceback (most recent call last):
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 1741, in <module>
    main()
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 1543, in main
    from config.settings import Settings
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../config/settings.py", line 6, in <module>
    from pydantic_settings import BaseSettings
  File "/usr/local/lib/python3.12/dist-packages/pydantic_settings/__init__.py", line 1, in <module>
    from .main import BaseSettings, SettingsConfigDict
  File "/usr/local/lib/python3.12/dist-packages/pydantic_settings/main.py", line 9, in <module>
    from pydantic.main import BaseModel
  File "/usr/local/lib/python3.12/dist-packages/pydantic/main.py", line 15, in <module>
    from ._internal import (
  File "/usr/local/lib/python3.12/dist-packages/pydantic/_internal/_decorators.py", line 15, in <module>
    from ._core_utils import get_type_ref
  File "/usr/local/lib/python3.12/dist-packages/pydantic/_internal/_core_utils.py", line 15, in <module>
    from pydantic_core import validate_core_schema as _validate_core_schema
ImportError: cannot import name 'validate_core_schema' from 'pydantic_core' (/usr/local/lib/python3.12/dist-packages/pydantic_core/__init__.py)

2026-04-24 00:20:07,151 INFO === STEP 7b: QUALITY + RL TRAINING ===
Traceback (most recent call last):
  File "/kaggle/working/Multi-Bot/trading-system/pipeline/step7b_train.py", line 105, in <module>
    main()
  File "/kaggle/working/Multi-Bot/trading-system/pipeline/step7b_train.py", line 65, in main
    raise RuntimeError(
RuntimeError: Journal missing at /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/trade_journal_detailed.jsonl — run step6_backtest.py first
