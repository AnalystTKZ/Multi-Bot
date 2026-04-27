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
2026-04-27 08:39:02,600 INFO Loading feature-engineered data...
2026-04-27 08:39:03,221 INFO Loaded 221743 rows, 202 features
2026-04-27 08:39:03,222 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-27 08:39:03,224 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-27 08:39:03,224 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-27 08:39:03,224 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-27 08:39:03,224 INFO No leakage confirmed: train < val < test timestamps

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
2026-04-27 08:39:05,661 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-27 08:39:05,661 INFO --- Training regime ---
2026-04-27 08:39:05,661 INFO Running retrain --model regime
2026-04-27 08:39:05,841 INFO retrain environment: KAGGLE
2026-04-27 08:39:07,485 INFO Device: CUDA (2 GPU(s))
2026-04-27 08:39:07,496 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 08:39:07,496 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 08:39:07,496 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 08:39:07,498 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 08:39:07,498 INFO Retrain data split: train
2026-04-27 08:39:07,499 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-27 08:39:07,657 INFO NumExpr defaulting to 4 threads.
2026-04-27 08:39:07,873 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-27 08:39:07,873 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 08:39:07,873 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 08:39:07,874 INFO Regime phase macro_correlations: 0.0s
2026-04-27 08:39:07,874 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-27 08:39:07,916 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-27 08:39:07,917 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:39:07,946 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:39:07,961 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:39:07,984 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:39:08,001 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:39:08,027 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:39:08,043 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:39:08,068 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:39:08,084 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:39:08,108 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:39:08,123 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:39:08,146 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:39:08,160 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:39:08,181 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:39:08,196 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:39:08,218 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:39:08,234 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:39:08,256 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:39:08,270 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:39:08,291 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:39:08,311 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 08:39:08,350 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 08:39:09,479 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 08:39:32,344 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 08:39:32,346 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 24.0s
2026-04-27 08:39:32,346 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 08:39:42,443 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 08:39:42,445 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 10.1s
2026-04-27 08:39:42,445 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 08:39:50,116 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 08:39:50,117 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 7.7s
2026-04-27 08:39:50,117 INFO Regime phase GMM HTF total: 41.8s
2026-04-27 08:39:50,117 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 08:41:01,955 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 08:41:01,958 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 71.8s
2026-04-27 08:41:01,958 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 08:41:33,435 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 08:41:33,436 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 31.5s
2026-04-27 08:41:33,436 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 08:41:55,769 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 08:41:55,770 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 22.3s
2026-04-27 08:41:55,770 INFO Regime phase GMM LTF total: 125.7s
2026-04-27 08:41:55,888 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-27 08:41:55,893 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:41:55,896 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:41:55,897 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:41:55,898 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:41:55,899 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:41:55,900 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:41:55,901 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:41:55,902 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:41:55,904 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:41:55,905 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:41:55,907 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 08:41:56,041 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:56,083 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:56,084 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:56,084 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:56,092 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:56,093 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:41:56,507 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-27 08:41:56,508 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-27 08:41:56,687 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:56,720 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:56,721 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:56,721 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:56,729 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:56,730 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:41:57,106 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-27 08:41:57,107 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-27 08:41:57,295 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:57,332 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:57,332 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:57,333 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:57,341 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:57,342 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:41:57,719 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-27 08:41:57,720 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-27 08:41:57,895 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:57,931 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:57,932 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:57,932 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:57,939 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:57,940 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:41:58,343 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-27 08:41:58,345 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-27 08:41:58,523 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:58,560 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:58,560 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:58,561 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:58,569 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:58,570 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:41:58,949 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-27 08:41:58,950 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-27 08:41:59,123 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:59,155 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:59,156 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:59,156 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:59,164 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:41:59,165 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:41:59,548 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-27 08:41:59,550 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-27 08:41:59,706 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 08:41:59,734 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-27 08:41:59,735 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 08:41:59,735 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 08:41:59,742 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 08:41:59,743 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:00,125 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-27 08:42:00,126 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-27 08:42:00,296 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:00,330 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:00,330 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:00,331 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:00,339 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:00,340 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:00,727 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-27 08:42:00,728 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-27 08:42:00,907 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:00,942 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:00,943 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:00,943 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:00,951 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:00,952 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:01,334 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-27 08:42:01,335 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-27 08:42:01,505 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:01,542 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:01,542 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:01,543 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:01,551 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:01,552 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:01,938 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-27 08:42:01,939 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-27 08:42:02,209 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 08:42:02,273 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-27 08:42:02,274 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 08:42:02,275 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 08:42:02,286 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 08:42:02,288 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 08:42:03,138 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 08:42:03,140 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-27 08:42:03,321 INFO Regime phase HTF dataset build: 7.4s (103290 samples)
2026-04-27 08:42:03,322 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=38114 dropped=65176 classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728})
2026-04-27 08:42:03,332 INFO RegimeClassifier[mode=htf_bias]: 38114 samples, classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728}, device=cuda
2026-04-27 08:42:03,332 INFO RegimeClassifier: sample weights — mean=0.708  ambiguous(<0.4)=0.0%
2026-04-27 08:42:03,621 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-04-27 08:42:03,621 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-27 08:42:08,363 INFO Regime epoch  1/50 — tr=0.7749 va=2.0170 acc=0.135 per_class={'BIAS_UP': 0.054, 'BIAS_DOWN': 0.058, 'BIAS_NEUTRAL': 0.499}
2026-04-27 08:42:08,433 INFO Regime epoch  2/50 — tr=0.7681 va=2.0062 acc=0.198
2026-04-27 08:42:08,504 INFO Regime epoch  3/50 — tr=0.7601 va=1.9546 acc=0.273
2026-04-27 08:42:08,576 INFO Regime epoch  4/50 — tr=0.7413 va=1.8749 acc=0.376
2026-04-27 08:42:08,650 INFO Regime epoch  5/50 — tr=0.7147 va=1.7735 acc=0.515 per_class={'BIAS_UP': 0.565, 'BIAS_DOWN': 0.304, 'BIAS_NEUTRAL': 0.783}
2026-04-27 08:42:08,717 INFO Regime epoch  6/50 — tr=0.6820 va=1.6695 acc=0.669
2026-04-27 08:42:08,789 INFO Regime epoch  7/50 — tr=0.6514 va=1.5732 acc=0.786
2026-04-27 08:42:08,865 INFO Regime epoch  8/50 — tr=0.6220 va=1.5008 acc=0.877
2026-04-27 08:42:08,939 INFO Regime epoch  9/50 — tr=0.5950 va=1.4415 acc=0.920
2026-04-27 08:42:09,019 INFO Regime epoch 10/50 — tr=0.5729 va=1.3881 acc=0.935 per_class={'BIAS_UP': 0.989, 'BIAS_DOWN': 0.962, 'BIAS_NEUTRAL': 0.741}
2026-04-27 08:42:09,096 INFO Regime epoch 11/50 — tr=0.5573 va=1.3516 acc=0.943
2026-04-27 08:42:09,172 INFO Regime epoch 12/50 — tr=0.5446 va=1.3223 acc=0.944
2026-04-27 08:42:09,243 INFO Regime epoch 13/50 — tr=0.5334 va=1.2940 acc=0.945
2026-04-27 08:42:09,313 INFO Regime epoch 14/50 — tr=0.5273 va=1.2663 acc=0.947
2026-04-27 08:42:09,392 INFO Regime epoch 15/50 — tr=0.5194 va=1.2440 acc=0.949 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.994, 'BIAS_NEUTRAL': 0.729}
2026-04-27 08:42:09,464 INFO Regime epoch 16/50 — tr=0.5159 va=1.2274 acc=0.951
2026-04-27 08:42:09,534 INFO Regime epoch 17/50 — tr=0.5113 va=1.2152 acc=0.952
2026-04-27 08:42:09,603 INFO Regime epoch 18/50 — tr=0.5087 va=1.2058 acc=0.953
2026-04-27 08:42:09,672 INFO Regime epoch 19/50 — tr=0.5062 va=1.1942 acc=0.955
2026-04-27 08:42:09,743 INFO Regime epoch 20/50 — tr=0.5038 va=1.1844 acc=0.957 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.762}
2026-04-27 08:42:09,811 INFO Regime epoch 21/50 — tr=0.5020 va=1.1757 acc=0.958
2026-04-27 08:42:09,879 INFO Regime epoch 22/50 — tr=0.5001 va=1.1743 acc=0.958
2026-04-27 08:42:09,947 INFO Regime epoch 23/50 — tr=0.4977 va=1.1675 acc=0.959
2026-04-27 08:42:10,013 INFO Regime epoch 24/50 — tr=0.4975 va=1.1646 acc=0.960
2026-04-27 08:42:10,087 INFO Regime epoch 25/50 — tr=0.4953 va=1.1558 acc=0.961 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.781}
2026-04-27 08:42:10,156 INFO Regime epoch 26/50 — tr=0.4941 va=1.1517 acc=0.961
2026-04-27 08:42:10,224 INFO Regime epoch 27/50 — tr=0.4935 va=1.1454 acc=0.963
2026-04-27 08:42:10,291 INFO Regime epoch 28/50 — tr=0.4922 va=1.1414 acc=0.963
2026-04-27 08:42:10,359 INFO Regime epoch 29/50 — tr=0.4919 va=1.1378 acc=0.964
2026-04-27 08:42:10,433 INFO Regime epoch 30/50 — tr=0.4907 va=1.1376 acc=0.964 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.799}
2026-04-27 08:42:10,504 INFO Regime epoch 31/50 — tr=0.4894 va=1.1342 acc=0.965
2026-04-27 08:42:10,573 INFO Regime epoch 32/50 — tr=0.4885 va=1.1307 acc=0.966
2026-04-27 08:42:10,644 INFO Regime epoch 33/50 — tr=0.4883 va=1.1314 acc=0.966
2026-04-27 08:42:10,714 INFO Regime epoch 34/50 — tr=0.4880 va=1.1257 acc=0.967
2026-04-27 08:42:10,788 INFO Regime epoch 35/50 — tr=0.4870 va=1.1272 acc=0.967 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.817}
2026-04-27 08:42:10,863 INFO Regime epoch 36/50 — tr=0.4871 va=1.1232 acc=0.967
2026-04-27 08:42:10,933 INFO Regime epoch 37/50 — tr=0.4870 va=1.1247 acc=0.968
2026-04-27 08:42:11,003 INFO Regime epoch 38/50 — tr=0.4862 va=1.1240 acc=0.968
2026-04-27 08:42:11,071 INFO Regime epoch 39/50 — tr=0.4862 va=1.1253 acc=0.967
2026-04-27 08:42:11,143 INFO Regime epoch 40/50 — tr=0.4858 va=1.1255 acc=0.968 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.821}
2026-04-27 08:42:11,212 INFO Regime epoch 41/50 — tr=0.4854 va=1.1247 acc=0.968
2026-04-27 08:42:11,287 INFO Regime epoch 42/50 — tr=0.4849 va=1.1238 acc=0.968
2026-04-27 08:42:11,356 INFO Regime epoch 43/50 — tr=0.4856 va=1.1186 acc=0.968
2026-04-27 08:42:11,425 INFO Regime epoch 44/50 — tr=0.4854 va=1.1171 acc=0.969
2026-04-27 08:42:11,502 INFO Regime epoch 45/50 — tr=0.4855 va=1.1182 acc=0.969 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.828}
2026-04-27 08:42:11,572 INFO Regime epoch 46/50 — tr=0.4852 va=1.1179 acc=0.969
2026-04-27 08:42:11,639 INFO Regime epoch 47/50 — tr=0.4855 va=1.1187 acc=0.969
2026-04-27 08:42:11,706 INFO Regime epoch 48/50 — tr=0.4849 va=1.1188 acc=0.968
2026-04-27 08:42:11,785 INFO Regime epoch 49/50 — tr=0.4854 va=1.1218 acc=0.968
2026-04-27 08:42:11,869 INFO Regime epoch 50/50 — tr=0.4847 va=1.1180 acc=0.969 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.827}
2026-04-27 08:42:11,880 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 08:42:11,880 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 08:42:11,881 INFO Regime phase HTF train: 8.6s
2026-04-27 08:42:12,021 INFO Regime HTF complete: acc=0.969, n=103290 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.826}
2026-04-27 08:42:12,023 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 08:42:12,180 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 08:42:12,188 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 11.41578947368421, 'BIAS_DOWN': 10.635761589403973, 'BIAS_NEUTRAL': 17.960468521229867}
2026-04-27 08:42:12,192 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 4338, 'mean': 0.00011110466582543088, 'mean_over_std': 0.027632581210502927}, 'BIAS_DOWN': {'n': 3212, 'mean': 6.063430222660252e-05, 'mean_over_std': 0.013898489590377157}, 'BIAS_NEUTRAL': {'n': 12266, 'mean': 1.2151554902814046e-05, 'mean_over_std': 0.0031957000513162357}}
2026-04-27 08:42:12,192 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 3651, 'mean': 0.0002852242108836959, 'mean_over_std': 0.07015017232491755}, 'BIAS_DOWN': {'n': 2644, 'mean': -0.00014900749618981248, 'mean_over_std': -0.03333037041298237}, 'BIAS_NEUTRAL': {'n': 1376, 'mean': 9.595797941249082e-05, 'mean_over_std': 0.02492411099521803}}
2026-04-27 08:42:12,196 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-27 08:42:12,197 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:12,199 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:12,201 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:12,202 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:12,204 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:12,206 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:12,207 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:12,209 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:12,210 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:12,212 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:12,215 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 08:42:12,228 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:12,231 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:12,232 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:12,232 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:12,233 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:12,235 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:12,871 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-27 08:42:12,874 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-27 08:42:13,012 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:13,015 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:13,016 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:13,016 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:13,016 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:13,018 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:13,644 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-27 08:42:13,647 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-27 08:42:13,785 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:13,787 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:13,788 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:13,788 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:13,789 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:13,791 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:14,392 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-27 08:42:14,396 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-27 08:42:14,537 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:14,540 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:14,541 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:14,541 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:14,541 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:14,543 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:15,134 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-27 08:42:15,137 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-27 08:42:15,280 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:15,282 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:15,283 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:15,283 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:15,283 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:15,286 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:15,922 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-27 08:42:15,926 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-27 08:42:16,068 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:16,070 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:16,071 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:16,071 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:16,072 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:16,074 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:16,670 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-27 08:42:16,673 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-27 08:42:16,807 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 08:42:16,809 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-27 08:42:16,810 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 08:42:16,810 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 08:42:16,810 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 08:42:16,812 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:17,421 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-27 08:42:17,424 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-27 08:42:17,560 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:17,564 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:17,565 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:17,566 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:17,566 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:17,568 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:18,188 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-27 08:42:18,191 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-27 08:42:18,334 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:18,337 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:18,337 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:18,338 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:18,338 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:18,340 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:18,952 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-27 08:42:18,955 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-27 08:42:19,092 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:19,094 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:19,095 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:19,095 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:19,095 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:19,097 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:19,715 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-27 08:42:19,718 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-27 08:42:19,864 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 08:42:19,868 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-27 08:42:19,869 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 08:42:19,870 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 08:42:19,870 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 08:42:19,873 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 08:42:21,197 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 08:42:21,204 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-27 08:42:21,518 INFO Regime phase LTF dataset build: 9.3s (401471 samples)
2026-04-27 08:42:21,522 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=299185 dropped=102286 classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127})
2026-04-27 08:42:21,579 INFO RegimeClassifier[mode=ltf_behaviour]: 299185 samples, classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127}, device=cuda
2026-04-27 08:42:21,580 INFO RegimeClassifier: sample weights — mean=0.693  ambiguous(<0.4)=0.0%
2026-04-27 08:42:21,582 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-04-27 08:42:21,582 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-27 08:42:22,164 INFO Regime epoch  1/50 — tr=0.8529 va=2.0025 acc=0.482 per_class={'TRENDING': 0.762, 'RANGING': 0.497, 'CONSOLIDATING': 0.024, 'VOLATILE': 0.225}
2026-04-27 08:42:22,678 INFO Regime epoch  2/50 — tr=0.8296 va=1.9285 acc=0.542
2026-04-27 08:42:23,166 INFO Regime epoch  3/50 — tr=0.7931 va=1.7863 acc=0.618
2026-04-27 08:42:23,684 INFO Regime epoch  4/50 — tr=0.7530 va=1.6500 acc=0.645
2026-04-27 08:42:24,214 INFO Regime epoch  5/50 — tr=0.7209 va=1.5438 acc=0.665 per_class={'TRENDING': 0.549, 'RANGING': 0.57, 'CONSOLIDATING': 0.716, 'VOLATILE': 0.912}
2026-04-27 08:42:24,726 INFO Regime epoch  6/50 — tr=0.7003 va=1.4827 acc=0.683
2026-04-27 08:42:25,220 INFO Regime epoch  7/50 — tr=0.6863 va=1.4523 acc=0.702
2026-04-27 08:42:25,742 INFO Regime epoch  8/50 — tr=0.6764 va=1.4337 acc=0.717
2026-04-27 08:42:26,267 INFO Regime epoch  9/50 — tr=0.6691 va=1.4119 acc=0.728
2026-04-27 08:42:26,813 INFO Regime epoch 10/50 — tr=0.6627 va=1.3987 acc=0.745 per_class={'TRENDING': 0.684, 'RANGING': 0.696, 'CONSOLIDATING': 0.712, 'VOLATILE': 0.913}
2026-04-27 08:42:27,300 INFO Regime epoch 11/50 — tr=0.6574 va=1.3736 acc=0.751
2026-04-27 08:42:27,816 INFO Regime epoch 12/50 — tr=0.6538 va=1.3595 acc=0.762
2026-04-27 08:42:28,351 INFO Regime epoch 13/50 — tr=0.6502 va=1.3494 acc=0.770
2026-04-27 08:42:28,866 INFO Regime epoch 14/50 — tr=0.6470 va=1.3340 acc=0.779
2026-04-27 08:42:29,432 INFO Regime epoch 15/50 — tr=0.6447 va=1.3284 acc=0.784 per_class={'TRENDING': 0.749, 'RANGING': 0.723, 'CONSOLIDATING': 0.744, 'VOLATILE': 0.912}
2026-04-27 08:42:29,923 INFO Regime epoch 16/50 — tr=0.6423 va=1.3177 acc=0.791
2026-04-27 08:42:30,407 INFO Regime epoch 17/50 — tr=0.6403 va=1.3011 acc=0.793
2026-04-27 08:42:30,921 INFO Regime epoch 18/50 — tr=0.6386 va=1.2959 acc=0.799
2026-04-27 08:42:31,415 INFO Regime epoch 19/50 — tr=0.6367 va=1.2949 acc=0.802
2026-04-27 08:42:31,969 INFO Regime epoch 20/50 — tr=0.6355 va=1.2875 acc=0.806 per_class={'TRENDING': 0.785, 'RANGING': 0.749, 'CONSOLIDATING': 0.766, 'VOLATILE': 0.905}
2026-04-27 08:42:32,499 INFO Regime epoch 21/50 — tr=0.6339 va=1.2812 acc=0.810
2026-04-27 08:42:33,001 INFO Regime epoch 22/50 — tr=0.6328 va=1.2796 acc=0.813
2026-04-27 08:42:33,535 INFO Regime epoch 23/50 — tr=0.6317 va=1.2737 acc=0.816
2026-04-27 08:42:34,058 INFO Regime epoch 24/50 — tr=0.6305 va=1.2642 acc=0.816
2026-04-27 08:42:34,635 INFO Regime epoch 25/50 — tr=0.6294 va=1.2637 acc=0.821 per_class={'TRENDING': 0.808, 'RANGING': 0.758, 'CONSOLIDATING': 0.799, 'VOLATILE': 0.895}
2026-04-27 08:42:35,133 INFO Regime epoch 26/50 — tr=0.6290 va=1.2593 acc=0.821
2026-04-27 08:42:35,639 INFO Regime epoch 27/50 — tr=0.6280 va=1.2561 acc=0.825
2026-04-27 08:42:36,170 INFO Regime epoch 28/50 — tr=0.6273 va=1.2540 acc=0.824
2026-04-27 08:42:36,708 INFO Regime epoch 29/50 — tr=0.6267 va=1.2513 acc=0.825
2026-04-27 08:42:37,243 INFO Regime epoch 30/50 — tr=0.6260 va=1.2473 acc=0.827 per_class={'TRENDING': 0.81, 'RANGING': 0.763, 'CONSOLIDATING': 0.833, 'VOLATILE': 0.894}
2026-04-27 08:42:37,755 INFO Regime epoch 31/50 — tr=0.6254 va=1.2456 acc=0.827
2026-04-27 08:42:38,290 INFO Regime epoch 32/50 — tr=0.6251 va=1.2447 acc=0.830
2026-04-27 08:42:38,779 INFO Regime epoch 33/50 — tr=0.6247 va=1.2444 acc=0.830
2026-04-27 08:42:39,276 INFO Regime epoch 34/50 — tr=0.6246 va=1.2403 acc=0.830
2026-04-27 08:42:39,809 INFO Regime epoch 35/50 — tr=0.6240 va=1.2425 acc=0.832 per_class={'TRENDING': 0.814, 'RANGING': 0.758, 'CONSOLIDATING': 0.855, 'VOLATILE': 0.895}
2026-04-27 08:42:40,312 INFO Regime epoch 36/50 — tr=0.6235 va=1.2410 acc=0.832
2026-04-27 08:42:40,835 INFO Regime epoch 37/50 — tr=0.6234 va=1.2379 acc=0.832
2026-04-27 08:42:41,342 INFO Regime epoch 38/50 — tr=0.6231 va=1.2372 acc=0.833
2026-04-27 08:42:41,838 INFO Regime epoch 39/50 — tr=0.6230 va=1.2333 acc=0.832
2026-04-27 08:42:42,385 INFO Regime epoch 40/50 — tr=0.6228 va=1.2339 acc=0.832 per_class={'TRENDING': 0.811, 'RANGING': 0.763, 'CONSOLIDATING': 0.857, 'VOLATILE': 0.895}
2026-04-27 08:42:42,875 INFO Regime epoch 41/50 — tr=0.6228 va=1.2395 acc=0.835
2026-04-27 08:42:43,389 INFO Regime epoch 42/50 — tr=0.6227 va=1.2307 acc=0.835
2026-04-27 08:42:43,891 INFO Regime epoch 43/50 — tr=0.6223 va=1.2343 acc=0.833
2026-04-27 08:42:44,392 INFO Regime epoch 44/50 — tr=0.6222 va=1.2327 acc=0.834
2026-04-27 08:42:44,923 INFO Regime epoch 45/50 — tr=0.6224 va=1.2371 acc=0.835 per_class={'TRENDING': 0.816, 'RANGING': 0.76, 'CONSOLIDATING': 0.861, 'VOLATILE': 0.898}
2026-04-27 08:42:45,412 INFO Regime epoch 46/50 — tr=0.6225 va=1.2328 acc=0.834
2026-04-27 08:42:45,953 INFO Regime epoch 47/50 — tr=0.6225 va=1.2324 acc=0.834
2026-04-27 08:42:46,460 INFO Regime epoch 48/50 — tr=0.6222 va=1.2358 acc=0.834
2026-04-27 08:42:46,960 INFO Regime epoch 49/50 — tr=0.6223 va=1.2335 acc=0.834
2026-04-27 08:42:47,485 INFO Regime epoch 50/50 — tr=0.6225 va=1.2362 acc=0.835 per_class={'TRENDING': 0.819, 'RANGING': 0.761, 'CONSOLIDATING': 0.866, 'VOLATILE': 0.89}
2026-04-27 08:42:47,525 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 08:42:47,525 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 08:42:47,527 INFO Regime phase LTF train: 26.0s
2026-04-27 08:42:47,659 INFO Regime LTF complete: acc=0.835, n=401471 per_class={'TRENDING': 0.821, 'RANGING': 0.761, 'CONSOLIDATING': 0.868, 'VOLATILE': 0.883}
2026-04-27 08:42:47,664 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 08:42:48,194 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 08:42:48,199 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 9.355291913830783, 'RANGING': 6.096303199751476, 'CONSOLIDATING': 5.598885793871866, 'VOLATILE': 6.771351107094442}
2026-04-27 08:42:48,207 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 29965, 'mean': 2.149062259083323e-05, 'mean_over_std': 0.011155601434349563}, 'RANGING': {'n': 19623, 'mean': -3.254493608872846e-06, 'mean_over_std': -0.001761415548292494}, 'CONSOLIDATING': {'n': 10050, 'mean': -3.039756158146958e-06, 'mean_over_std': -0.001792488091618151}, 'VOLATILE': {'n': 14985, 'mean': 1.8102764458593193e-05, 'mean_over_std': 0.006974696291378742}}
2026-04-27 08:42:48,208 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 26145, 'mean': 2.874435605138053e-05, 'mean_over_std': 0.01595450632691917}, 'RANGING': {'n': 7858, 'mean': 1.7829776785521727e-05, 'mean_over_std': 0.011851561804467463}, 'CONSOLIDATING': {'n': 8543, 'mean': 7.845231961250924e-06, 'mean_over_std': 0.004915839521549481}, 'VOLATILE': {'n': 13636, 'mean': 5.480114267306668e-06, 'mean_over_std': 0.0020378119917389375}}
2026-04-27 08:42:48,212 INFO Regime retrain total: 220.7s (504761 samples)
2026-04-27 08:42:48,228 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-27 08:42:48,228 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 08:42:48,229 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 08:42:48,229 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-27 08:42:48,229 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-27 08:42:48,229 INFO Retrain complete. Total wall-clock: 220.7s
2026-04-27 08:42:50,445 INFO Model regime: SUCCESS
2026-04-27 08:42:50,446 INFO --- Training gru ---
2026-04-27 08:42:50,446 INFO Running retrain --model gru
2026-04-27 08:42:50,715 INFO retrain environment: KAGGLE
2026-04-27 08:42:52,359 INFO Device: CUDA (2 GPU(s))
2026-04-27 08:42:52,370 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 08:42:52,370 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 08:42:52,371 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 08:42:52,371 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 08:42:52,371 INFO Retrain data split: train
2026-04-27 08:42:52,372 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-27 08:42:52,517 INFO NumExpr defaulting to 4 threads.
2026-04-27 08:42:52,713 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-27 08:42:52,713 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 08:42:52,713 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 08:42:52,714 INFO GRU phase macro_correlations: 0.0s
2026-04-27 08:42:52,714 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-27 08:42:52,714 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260427_084252
2026-04-27 08:42:52,718 WARNING WeightsManifest: no manifest at /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json — assuming compatible (run a full retrain to generate one)
2026-04-27 08:42:52,864 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:52,884 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:52,898 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:52,905 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:52,906 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-27 08:42:52,906 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 08:42:52,906 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 08:42:52,907 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-27 08:42:52,907 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:52,988 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-27 08:42:52,990 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:53,234 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-27 08:42:53,263 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:53,539 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:53,670 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:53,768 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:53,969 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:53,988 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:54,003 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:54,011 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:54,011 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:54,087 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-27 08:42:54,089 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:54,324 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-27 08:42:54,341 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:54,609 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:54,737 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:54,835 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:55,027 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:55,046 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:55,061 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:55,068 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:55,069 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:55,144 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-27 08:42:55,146 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:55,371 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-27 08:42:55,387 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:55,674 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:55,806 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:55,925 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:56,125 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:56,144 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:56,160 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:56,167 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:56,168 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:56,243 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-27 08:42:56,244 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:56,475 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-27 08:42:56,497 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:56,770 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:56,897 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:56,993 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:57,178 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:57,198 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:57,212 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:57,219 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:57,220 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:57,294 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-27 08:42:57,296 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:57,522 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-27 08:42:57,537 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:57,804 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:57,928 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:58,028 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:58,223 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:58,244 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:58,257 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:58,264 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:42:58,265 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:58,341 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-27 08:42:58,342 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:58,576 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-27 08:42:58,592 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:58,856 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:58,985 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:59,084 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:59,247 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 08:42:59,264 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 08:42:59,277 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 08:42:59,283 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 08:42:59,283 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:59,360 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-27 08:42:59,361 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:59,613 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-27 08:42:59,626 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:42:59,886 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:43:00,011 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:43:00,105 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:43:00,284 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:43:00,301 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:43:00,316 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:43:00,322 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:43:00,323 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:43:00,401 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-27 08:43:00,403 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:43:00,645 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-27 08:43:00,663 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:43:00,932 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:43:01,062 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:43:01,156 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:43:01,336 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:43:01,354 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:43:01,369 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:43:01,376 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:43:01,376 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:43:01,451 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-27 08:43:01,453 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:43:01,702 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-27 08:43:01,718 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:43:01,991 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:43:02,127 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:43:02,222 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:43:02,410 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:43:02,431 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:43:02,444 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:43:02,451 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 08:43:02,452 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:43:02,531 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-27 08:43:02,533 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:43:02,774 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-27 08:43:02,790 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:43:03,073 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:43:03,207 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:43:03,309 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 08:43:03,618 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 08:43:03,644 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 08:43:03,661 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 08:43:03,671 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 08:43:03,673 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 08:43:03,830 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 08:43:03,833 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 08:43:04,342 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 08:43:04,391 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-27 08:43:04,931 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-27 08:43:05,132 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 08:43:05,264 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 08:43:05,385 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-27 08:43:05,658 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-04-27 08:43:05,658 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-04-27 08:43:05,658 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-27 08:43:05,659 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-27 08:43:54,548 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-27 08:43:54,548 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-27 08:43:55,919 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-27 08:44:00,060 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=18, min_epochs=22)
2026-04-27 08:44:13,897 INFO train_multi TF=ALL epoch 1/50 train=0.8869 val=0.8797
2026-04-27 08:44:13,905 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:44:13,905 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:44:13,905 INFO train_multi TF=ALL: new best val=0.8797 — saved
2026-04-27 08:44:25,871 INFO train_multi TF=ALL epoch 2/50 train=0.8644 val=0.8254
2026-04-27 08:44:25,878 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:44:25,878 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:44:25,879 INFO train_multi TF=ALL: new best val=0.8254 — saved
2026-04-27 08:44:37,858 INFO train_multi TF=ALL epoch 3/50 train=0.7342 val=0.6884
2026-04-27 08:44:37,862 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:44:37,863 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:44:37,863 INFO train_multi TF=ALL: new best val=0.6884 — saved
2026-04-27 08:44:49,758 INFO train_multi TF=ALL epoch 4/50 train=0.6907 val=0.6879
2026-04-27 08:44:49,762 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:44:49,762 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:44:49,763 INFO train_multi TF=ALL: new best val=0.6879 — saved
2026-04-27 08:45:01,636 INFO train_multi TF=ALL epoch 5/50 train=0.6894 val=0.6888
2026-04-27 08:45:13,481 INFO train_multi TF=ALL epoch 6/50 train=0.6892 val=0.6885
2026-04-27 08:45:25,444 INFO train_multi TF=ALL epoch 7/50 train=0.6891 val=0.6884
2026-04-27 08:45:37,356 INFO train_multi TF=ALL epoch 8/50 train=0.6888 val=0.6887
2026-04-27 08:45:49,238 INFO train_multi TF=ALL epoch 9/50 train=0.6886 val=0.6884
2026-04-27 08:46:01,131 INFO train_multi TF=ALL epoch 10/50 train=0.6881 val=0.6874
2026-04-27 08:46:01,136 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:46:01,136 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:46:01,136 INFO train_multi TF=ALL: new best val=0.6874 — saved
2026-04-27 08:46:12,979 INFO train_multi TF=ALL epoch 11/50 train=0.6872 val=0.6858
2026-04-27 08:46:12,983 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:46:12,983 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:46:12,983 INFO train_multi TF=ALL: new best val=0.6858 — saved
2026-04-27 08:46:24,830 INFO train_multi TF=ALL epoch 12/50 train=0.6839 val=0.6823
2026-04-27 08:46:24,834 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:46:24,835 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:46:24,835 INFO train_multi TF=ALL: new best val=0.6823 — saved
2026-04-27 08:46:36,874 INFO train_multi TF=ALL epoch 13/50 train=0.6749 val=0.6659
2026-04-27 08:46:36,879 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:46:36,879 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:46:36,879 INFO train_multi TF=ALL: new best val=0.6659 — saved
2026-04-27 08:46:48,784 INFO train_multi TF=ALL epoch 14/50 train=0.6602 val=0.6485
2026-04-27 08:46:48,788 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:46:48,788 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:46:48,789 INFO train_multi TF=ALL: new best val=0.6485 — saved
2026-04-27 08:47:00,656 INFO train_multi TF=ALL epoch 15/50 train=0.6477 val=0.6347
2026-04-27 08:47:00,661 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:47:00,661 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:47:00,661 INFO train_multi TF=ALL: new best val=0.6347 — saved
2026-04-27 08:47:12,493 INFO train_multi TF=ALL epoch 16/50 train=0.6387 val=0.6292
2026-04-27 08:47:12,497 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:47:12,497 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:47:12,497 INFO train_multi TF=ALL: new best val=0.6292 — saved
2026-04-27 08:47:24,487 INFO train_multi TF=ALL epoch 17/50 train=0.6332 val=0.6267
2026-04-27 08:47:24,492 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:47:24,492 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:47:24,492 INFO train_multi TF=ALL: new best val=0.6267 — saved
2026-04-27 08:47:36,531 INFO train_multi TF=ALL epoch 18/50 train=0.6284 val=0.6247
2026-04-27 08:47:36,535 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:47:36,535 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:47:36,535 INFO train_multi TF=ALL: new best val=0.6247 — saved
2026-04-27 08:47:48,386 INFO train_multi TF=ALL epoch 19/50 train=0.6249 val=0.6240
2026-04-27 08:47:48,391 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:47:48,391 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:47:48,391 INFO train_multi TF=ALL: new best val=0.6240 — saved
2026-04-27 08:48:00,242 INFO train_multi TF=ALL epoch 20/50 train=0.6219 val=0.6229
2026-04-27 08:48:00,246 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:48:00,246 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:48:00,246 INFO train_multi TF=ALL: new best val=0.6229 — saved
2026-04-27 08:48:12,157 INFO train_multi TF=ALL epoch 21/50 train=0.6188 val=0.6186
2026-04-27 08:48:12,161 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:48:12,161 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:48:12,162 INFO train_multi TF=ALL: new best val=0.6186 — saved
2026-04-27 08:48:24,040 INFO train_multi TF=ALL epoch 22/50 train=0.6168 val=0.6202
2026-04-27 08:48:36,042 INFO train_multi TF=ALL epoch 23/50 train=0.6145 val=0.6186
2026-04-27 08:48:36,047 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:48:36,047 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:48:36,047 INFO train_multi TF=ALL: new best val=0.6186 — saved
2026-04-27 08:48:47,978 INFO train_multi TF=ALL epoch 24/50 train=0.6126 val=0.6163
2026-04-27 08:48:47,983 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:48:47,983 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:48:47,983 INFO train_multi TF=ALL: new best val=0.6163 — saved
2026-04-27 08:48:59,803 INFO train_multi TF=ALL epoch 25/50 train=0.6104 val=0.6154
2026-04-27 08:48:59,807 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:48:59,808 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:48:59,808 INFO train_multi TF=ALL: new best val=0.6154 — saved
2026-04-27 08:49:11,687 INFO train_multi TF=ALL epoch 26/50 train=0.6083 val=0.6164
2026-04-27 08:49:23,518 INFO train_multi TF=ALL epoch 27/50 train=0.6071 val=0.6130
2026-04-27 08:49:23,523 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:49:23,523 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:49:23,524 INFO train_multi TF=ALL: new best val=0.6130 — saved
2026-04-27 08:49:35,468 INFO train_multi TF=ALL epoch 28/50 train=0.6049 val=0.6157
2026-04-27 08:49:47,422 INFO train_multi TF=ALL epoch 29/50 train=0.6034 val=0.6139
2026-04-27 08:49:59,316 INFO train_multi TF=ALL epoch 30/50 train=0.6026 val=0.6126
2026-04-27 08:49:59,321 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:49:59,321 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:49:59,321 INFO train_multi TF=ALL: new best val=0.6126 — saved
2026-04-27 08:50:11,158 INFO train_multi TF=ALL epoch 31/50 train=0.6004 val=0.6129
2026-04-27 08:50:23,014 INFO train_multi TF=ALL epoch 32/50 train=0.5984 val=0.6121
2026-04-27 08:50:23,019 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:50:23,019 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:50:23,019 INFO train_multi TF=ALL: new best val=0.6121 — saved
2026-04-27 08:50:35,036 INFO train_multi TF=ALL epoch 33/50 train=0.5975 val=0.6114
2026-04-27 08:50:35,040 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:50:35,040 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:50:35,040 INFO train_multi TF=ALL: new best val=0.6114 — saved
2026-04-27 08:50:47,006 INFO train_multi TF=ALL epoch 34/50 train=0.5960 val=0.6118
2026-04-27 08:50:58,865 INFO train_multi TF=ALL epoch 35/50 train=0.5945 val=0.6101
2026-04-27 08:50:58,870 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:50:58,870 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:50:58,870 INFO train_multi TF=ALL: new best val=0.6101 — saved
2026-04-27 08:51:10,764 INFO train_multi TF=ALL epoch 36/50 train=0.5929 val=0.6103
2026-04-27 08:51:22,761 INFO train_multi TF=ALL epoch 37/50 train=0.5919 val=0.6128
2026-04-27 08:51:34,724 INFO train_multi TF=ALL epoch 38/50 train=0.5907 val=0.6115
2026-04-27 08:51:46,643 INFO train_multi TF=ALL epoch 39/50 train=0.5894 val=0.6150
2026-04-27 08:51:58,536 INFO train_multi TF=ALL epoch 40/50 train=0.5879 val=0.6137
2026-04-27 08:52:10,403 INFO train_multi TF=ALL epoch 41/50 train=0.5866 val=0.6172
2026-04-27 08:52:22,381 INFO train_multi TF=ALL epoch 42/50 train=0.5853 val=0.6160
2026-04-27 08:52:34,471 INFO train_multi TF=ALL epoch 43/50 train=0.5843 val=0.6179
2026-04-27 08:52:46,447 INFO train_multi TF=ALL epoch 44/50 train=0.5836 val=0.6126
2026-04-27 08:52:58,348 INFO train_multi TF=ALL epoch 45/50 train=0.5822 val=0.6117
2026-04-27 08:53:10,289 INFO train_multi TF=ALL epoch 46/50 train=0.5812 val=0.6116
2026-04-27 08:53:22,248 INFO train_multi TF=ALL epoch 47/50 train=0.5792 val=0.6139
2026-04-27 08:53:34,252 INFO train_multi TF=ALL epoch 48/50 train=0.5782 val=0.6099
2026-04-27 08:53:34,256 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 08:53:34,256 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:53:34,256 INFO train_multi TF=ALL: new best val=0.6099 — saved
2026-04-27 08:53:46,190 INFO train_multi TF=ALL epoch 49/50 train=0.5769 val=0.6164
2026-04-27 08:53:58,065 INFO train_multi TF=ALL epoch 50/50 train=0.5762 val=0.6122
2026-04-27 08:53:58,224 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-27 08:53:58,224 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-27 08:53:58,224 INFO Retrain complete. Total wall-clock: 665.9s
2026-04-27 08:54:00,202 INFO Model gru: SUCCESS
2026-04-27 08:54:00,202 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 08:54:00,202 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 08:54:00,202 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 08:54:00,202 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-04-27 08:54:00,202 INFO   [DEFERRED] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip (expected after Round 1)
2026-04-27 08:54:00,202 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-04-27 08:54:00,203 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer', 'rl_ppo']
2026-04-27 08:54:00,203 INFO Saved 5 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  START Round 1 - Backtest (val)
2026-04-27 08:54:00,956 INFO === STEP 6: BACKTEST (round1) ===
2026-04-27 08:54:00,957 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-04-27 08:54:00,957 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-27 08:54:00,957 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
2026-04-27 08:54:03,399 WARNING QualityScorer unavailable (weights missing or load failed)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260427_085403.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)            16100  29.4%   1.28 27755.2% 29.4% 12.9%  97.6%    0.46
  gate_diagnostics: bars=468696 no_signal=82018 quality_block=0 session_skip=256819 density=99959 pm_reject=13800 daily_skip=0 cooldown=0 daily_halt_events=0 enforce_daily_halt=False
  no_signal_reasons: htf_bias_conflict=77893, trend_pullback_conflict=4077, range_side_conflict=48

Calibration Summary:
  all          [OK] Calibration OK — p_win correlates with actual win rate.
  ml_trader    [OK] Calibration OK — p_win correlates with actual win rate.
2026-04-27 08:57:23,676 INFO Round 1 backtest — 16100 trades | avg WR=29.4% | avg PF=1.28 | avg Sharpe=0.46
2026-04-27 08:57:23,676 INFO   ml_trader: 16100 trades | WR=29.4% | PF=1.28 | Return=27755.2% | DD=97.6% | Sharpe=0.46
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 16100
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (16100 rows)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  AUDUSD       1498 trades   434 days   3.45/day  [OVERTRADE]
  EURGBP       1522 trades   432 days   3.52/day  [OVERTRADE]
  EURJPY       1475 trades   425 days   3.47/day  [OVERTRADE]
  EURUSD       1532 trades   433 days   3.54/day  [OVERTRADE]
  GBPJPY       1464 trades   413 days   3.54/day  [OVERTRADE]
  GBPUSD       1460 trades   415 days   3.52/day  [OVERTRADE]
  NZDUSD       1132 trades   299 days   3.79/day  [OVERTRADE]
  USDCAD       1543 trades   418 days   3.69/day  [OVERTRADE]
  USDCHF       1526 trades   417 days   3.66/day  [OVERTRADE]
  USDJPY       1499 trades   421 days   3.56/day  [OVERTRADE]
  XAUUSD       1449 trades   414 days   3.50/day  [OVERTRADE]
  ⚠  AUDUSD: 3.45/day (>1.5)
  ⚠  EURGBP: 3.52/day (>1.5)
  ⚠  EURJPY: 3.47/day (>1.5)
  ⚠  EURUSD: 3.54/day (>1.5)
  ⚠  GBPJPY: 3.54/day (>1.5)
  ⚠  GBPUSD: 3.52/day (>1.5)
  ⚠  NZDUSD: 3.79/day (>1.5)
  ⚠  USDCAD: 3.69/day (>1.5)
  ⚠  USDCHF: 3.66/day (>1.5)
  ⚠  USDJPY: 3.56/day (>1.5)
  ⚠  XAUUSD: 3.50/day (>1.5)

──────────────────────────────────────────────────────────────
CHECK 2: REGIME DISTRIBUTION  (% of trades, WR, avgEV)
──────────────────────────────────────────────────────────────
  BIAS_DOWN          4970 trades   30.9%  WR=29.2%  avgEV=0.000
  BIAS_NEUTRAL       5091 trades   31.6%  WR=28.3%  avgEV=0.000
  BIAS_UP            6039 trades   37.5%  WR=30.4%  avgEV=0.000
  ⚠  Regimes never traded: ['CONSOLIDATING', 'RANGING', 'TRENDING', 'VOLATILE']

──────────────────────────────────────────────────────────────
CHECK 3: EV PREDICTED vs REALIZED RR
──────────────────────────────────────────────────────────────
  Pearson  = +nan   Spearman = -0.3323

  Bucket                  N     AvgEV     AvgRR   WinRate
  Q1 (low EV)             0       n/a       n/a       n/a
  Q2                      0       n/a       n/a       n/a
  Q3                      0       n/a       n/a       n/a
  Q4 (high EV)        16100     0.000    -0.097     29.4%

  Top-20% EV trades: n=16100  avgEV=0.0  avgRR=-0.097  WR=29.4%

  Per-regime EV↔RR correlation:
  Regime                N    Pearson   Spearman       WR     AvgEV
  BIAS_DOWN          4970       +nan    -0.3530   29.2%     0.000
  BIAS_NEUTRAL       5091       +nan    -0.3151   28.3%     0.000
  BIAS_UP            6039       +nan    -0.3346   30.4%     0.000
  ⚠  EV↔RR Spearman=-0.332 < 0.15 — EV rankings don't predict outcomes
  ⚠  Top-20% EV trades win_rate=29.4% — high-EV selection not working
  ⚠  EV↔RR Spearman in BIAS_DOWN = -0.353 — EV useless in this regime
  ⚠  EV↔RR Spearman in BIAS_NEUTRAL = -0.315 — EV useless in this regime
  ⚠  EV↔RR Spearman in BIAS_UP = -0.335 — EV useless in this regime

──────────────────────────────────────────────────────────────
CHECK 4: GRU CONFIDENCE CALIBRATION  (p_bull/bear vs win rate)
──────────────────────────────────────────────────────────────
  ECE = 0.4684  (target < 0.10)
  Bin                   N   Midpoint    WinRate    Error
  [0.50-0.59]         736      0.544      0.311    0.233
  [0.59-0.68]        2129      0.631      0.292    0.339
  [0.68-0.76]        4257      0.719      0.292    0.427
  [0.76-0.85]        6451      0.807      0.294    0.513
  [0.85-0.94]        2527      0.894      0.292    0.602
  ⚠  Bin [0.50-0.59]: midpoint=0.54 win_rate=0.31 (err=0.23 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.59-0.68]: midpoint=0.63 win_rate=0.29 (err=0.34 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.68-0.76]: midpoint=0.72 win_rate=0.29 (err=0.43 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.76-0.85]: midpoint=0.81 win_rate=0.29 (err=0.51 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.85-0.94]: midpoint=0.89 win_rate=0.29 (err=0.60 > 0.15) — GRU miscalibrated
  ⚠  Win rate non-monotonic across confidence bins — GRU confidence unreliable

──────────────────────────────────────────────────────────────
CHECK 5: GRU ↔ EV MODEL CONSISTENCY  (direction × value agreement)
──────────────────────────────────────────────────────────────
  Pearson=+nan  Spearman=-0.0649  Agree=50%

  Quadrants  (conf_threshold=median, ev_threshold=median):
  high_conf + high_ev:  8063  ← ideal
  high_conf + low_ev:      0  ← GRU overconfident
  low_conf  + high_ev:  8037  ← EV optimistic
  low_conf  + low_ev:      0  ← correct abstention
  ⚠  GRU and EV agree on only 50.1% of trades — models pulling in opposite directions

───────────────────────────────────────────────────────────
2026-04-27 08:57:30,303 INFO Round 1: wrote 16100 journal entries (total in file: 16100)
  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 16100 entries

=== Round 1 → Quality + RL retrain skipped (validation journal is not train data) ===

=== Round 2: BLIND backtest on test window (unseen 2yr) ===
  START Round 2 - Blind backtest (test)
2026-04-27 08:57:31,001 INFO === STEP 6: BACKTEST (round2) ===
2026-04-27 08:57:31,002 INFO BT_WINDOW=round2 — BLIND backtest: 2023-08-07 → 2025-08-05 (test set)
2026-04-27 08:57:31,002 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-27 08:57:31,002 INFO Round 2 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-27 08:57:33,430 WARNING QualityScorer unavailable (weights missing or load failed)
2026-04-27 09:00:59,076 INFO Round 2 backtest — 12963 trades | avg WR=29.7% | avg PF=1.27 | avg Sharpe=0.49
2026-04-27 09:00:59,076 INFO   ml_trader: 12963 trades | WR=29.7% | PF=1.27 | Return=37360.0% | DD=86.4% | Sharpe=0.49
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 12963
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (12963 rows)
2026-04-27 09:01:04,487 INFO Round 2: wrote 12963 journal entries (total in file: 29063)
  DONE  Round 2 - Blind backtest (test)
  Saved Round 2 result → round2_summary.json
  Journal after Round 2: 29063 entries

=== Round 2 → Quality + RL retrain skipped (blind-test journal is not train data) ===

=== Round 3: Incremental retrain on train split only ===
  START Retrain gru [train-split retrain]
2026-04-27 09:01:04,883 INFO retrain environment: KAGGLE
2026-04-27 09:01:06,559 INFO Device: CUDA (2 GPU(s))
2026-04-27 09:01:06,571 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 09:01:06,571 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 09:01:06,571 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 09:01:06,572 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 09:01:06,572 INFO Retrain data split: train
2026-04-27 09:01:06,573 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-27 09:01:06,719 INFO NumExpr defaulting to 4 threads.
2026-04-27 09:01:06,918 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-27 09:01:06,918 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 09:01:06,918 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 09:01:07,162 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-27 09:01:07,162 INFO GRU phase macro_correlations: 0.0s
2026-04-27 09:01:07,162 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-27 09:01:07,164 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260427_090107
2026-04-27 09:01:07,167 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-04-27 09:01:07,316 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:07,336 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:07,349 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:07,356 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:07,357 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-27 09:01:07,357 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 09:01:07,357 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 09:01:07,357 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-27 09:01:07,358 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:07,433 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-27 09:01:07,434 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:07,666 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-27 09:01:07,696 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:07,973 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:08,110 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:08,221 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:08,427 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:08,446 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:08,460 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:08,467 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:08,468 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:08,544 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-27 09:01:08,546 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:08,761 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-27 09:01:08,776 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:09,043 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:09,171 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:09,269 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:09,472 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:09,493 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:09,508 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:09,516 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:09,516 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:09,594 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-27 09:01:09,596 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:09,812 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-27 09:01:09,828 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:10,107 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:10,238 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:10,338 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:10,528 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:10,547 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:10,562 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:10,569 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:10,570 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:10,646 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-27 09:01:10,648 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:10,887 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-27 09:01:10,912 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:11,185 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:11,319 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:11,418 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:11,613 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:11,635 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:11,650 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:11,656 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:11,657 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:11,731 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-27 09:01:11,733 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:11,962 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-27 09:01:11,979 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:12,253 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:12,386 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:12,487 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:12,678 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:12,698 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:12,712 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:12,719 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:12,720 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:12,795 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-27 09:01:12,797 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:13,030 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-27 09:01:13,046 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:13,335 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:13,472 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:13,571 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:13,741 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 09:01:13,758 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 09:01:13,771 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 09:01:13,778 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 09:01:13,778 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:13,853 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-27 09:01:13,855 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:14,082 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-27 09:01:14,095 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:14,363 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:14,495 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:14,597 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:14,775 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:14,793 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:14,807 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:14,814 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:14,815 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:14,890 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-27 09:01:14,892 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:15,113 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-27 09:01:15,131 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:15,402 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:15,533 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:15,637 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:15,818 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:15,837 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:15,856 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:15,869 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:15,870 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:15,973 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-27 09:01:15,975 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:16,196 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-27 09:01:16,213 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:16,482 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:16,617 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:16,717 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:16,916 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:16,936 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:16,951 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:16,958 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:01:16,959 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:17,035 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-27 09:01:17,037 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:17,260 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-27 09:01:17,276 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:17,549 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:17,682 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:17,778 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:01:18,070 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 09:01:18,097 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 09:01:18,115 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 09:01:18,126 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 09:01:18,127 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 09:01:18,295 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 09:01:18,298 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 09:01:18,784 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 09:01:18,833 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-27 09:01:19,371 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-27 09:01:19,575 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 09:01:19,710 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 09:01:19,829 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-27 09:01:19,829 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-27 09:01:19,829 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-27 09:02:09,499 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-27 09:02:09,499 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-27 09:02:10,850 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-27 09:02:15,026 INFO train_multi TF=ALL: warm-start detected — using CosineAnnealingLR (lr=3e-05, patience=12)
2026-04-27 09:02:28,787 INFO train_multi TF=ALL epoch 1/50 train=0.5751 val=0.6122
2026-04-27 09:02:28,792 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 09:02:28,792 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 09:02:28,793 INFO train_multi TF=ALL: new best val=0.6122 — saved
2026-04-27 09:02:40,817 INFO train_multi TF=ALL epoch 2/50 train=0.5745 val=0.6117
2026-04-27 09:02:40,822 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 09:02:40,822 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 09:02:40,822 INFO train_multi TF=ALL: new best val=0.6117 — saved
2026-04-27 09:02:52,784 INFO train_multi TF=ALL epoch 3/50 train=0.5745 val=0.6111
2026-04-27 09:02:52,789 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 09:02:52,789 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 09:02:52,789 INFO train_multi TF=ALL: new best val=0.6111 — saved
2026-04-27 09:03:04,846 INFO train_multi TF=ALL epoch 4/50 train=0.5742 val=0.6122
2026-04-27 09:03:16,969 INFO train_multi TF=ALL epoch 5/50 train=0.5743 val=0.6120
2026-04-27 09:03:29,051 INFO train_multi TF=ALL epoch 6/50 train=0.5736 val=0.6127
2026-04-27 09:03:41,091 INFO train_multi TF=ALL epoch 7/50 train=0.5736 val=0.6123
2026-04-27 09:03:53,092 INFO train_multi TF=ALL epoch 8/50 train=0.5735 val=0.6121
2026-04-27 09:04:05,137 INFO train_multi TF=ALL epoch 9/50 train=0.5736 val=0.6119
2026-04-27 09:04:17,177 INFO train_multi TF=ALL epoch 10/50 train=0.5733 val=0.6124
2026-04-27 09:04:29,151 INFO train_multi TF=ALL epoch 11/50 train=0.5731 val=0.6121
2026-04-27 09:04:41,200 INFO train_multi TF=ALL epoch 12/50 train=0.5733 val=0.6120
2026-04-27 09:04:53,157 INFO train_multi TF=ALL epoch 13/50 train=0.5727 val=0.6116
2026-04-27 09:05:05,158 INFO train_multi TF=ALL epoch 14/50 train=0.5726 val=0.6132
2026-04-27 09:05:17,236 INFO train_multi TF=ALL epoch 15/50 train=0.5724 val=0.6129
2026-04-27 09:05:17,236 INFO train_multi TF=ALL early stop at epoch 15
2026-04-27 09:05:17,646 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-27 09:05:17,646 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-27 09:05:17,646 INFO Retrain complete. Total wall-clock: 251.1s
  DONE  Retrain gru [train-split retrain]
  START Retrain regime [train-split retrain]
2026-04-27 09:05:19,857 INFO retrain environment: KAGGLE
2026-04-27 09:05:21,489 INFO Device: CUDA (2 GPU(s))
2026-04-27 09:05:21,498 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 09:05:21,498 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 09:05:21,498 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 09:05:21,499 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 09:05:21,499 INFO Retrain data split: train
2026-04-27 09:05:21,500 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-27 09:05:21,647 INFO NumExpr defaulting to 4 threads.
2026-04-27 09:05:21,854 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-27 09:05:21,854 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 09:05:21,854 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 09:05:21,854 INFO Regime phase macro_correlations: 0.0s
2026-04-27 09:05:21,855 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-27 09:05:21,893 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-27 09:05:21,894 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:05:21,921 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:05:21,936 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:05:21,963 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:05:21,978 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:05:22,002 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:05:22,016 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:05:22,039 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:05:22,055 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:05:22,079 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:05:22,093 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:05:22,115 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:05:22,130 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:05:22,149 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:05:22,163 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:05:22,185 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:05:22,200 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:05:22,221 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:05:22,237 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:05:22,260 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:05:22,277 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 09:05:22,315 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 09:05:23,115 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 09:05:45,683 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 09:05:45,688 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 23.4s
2026-04-27 09:05:45,691 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 09:05:56,347 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 09:05:56,349 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 10.7s
2026-04-27 09:05:56,349 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 09:06:03,940 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 09:06:03,941 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 7.6s
2026-04-27 09:06:03,941 INFO Regime phase GMM HTF total: 41.6s
2026-04-27 09:06:03,941 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 09:07:16,417 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 09:07:16,420 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 72.5s
2026-04-27 09:07:16,420 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 09:07:48,387 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 09:07:48,389 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 32.0s
2026-04-27 09:07:48,390 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 09:08:11,234 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 09:08:11,235 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 22.8s
2026-04-27 09:08:11,235 INFO Regime phase GMM LTF total: 127.3s
2026-04-27 09:08:11,341 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-27 09:08:11,342 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:11,343 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:11,344 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:11,345 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:11,346 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:11,347 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:11,348 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:11,350 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:11,351 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:11,352 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:11,353 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 09:08:11,475 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:11,517 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:11,517 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:11,518 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:11,526 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:11,526 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:11,927 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-27 09:08:11,928 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-27 09:08:12,107 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:12,143 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:12,144 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:12,145 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:12,152 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:12,153 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:12,521 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-27 09:08:12,522 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-27 09:08:12,712 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:12,748 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:12,749 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:12,749 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:12,757 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:12,758 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:13,136 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-27 09:08:13,138 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-27 09:08:13,311 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:13,350 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:13,351 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:13,351 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:13,360 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:13,360 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:13,750 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-27 09:08:13,751 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-27 09:08:13,932 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:13,967 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:13,968 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:13,968 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:13,976 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:13,977 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:14,347 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-27 09:08:14,348 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-27 09:08:14,525 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:14,558 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:14,559 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:14,560 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:14,567 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:14,569 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:14,944 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-27 09:08:14,945 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-27 09:08:15,103 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 09:08:15,131 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-27 09:08:15,131 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 09:08:15,132 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 09:08:15,139 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 09:08:15,140 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:15,512 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-27 09:08:15,514 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-27 09:08:15,691 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:15,726 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:15,727 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:15,728 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:15,735 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:15,736 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:16,127 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-27 09:08:16,129 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-27 09:08:16,303 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:16,337 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:16,337 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:16,338 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:16,346 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:16,347 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:16,725 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-27 09:08:16,726 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-27 09:08:16,901 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:16,935 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:16,935 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:16,936 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:16,944 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:16,944 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:17,324 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-27 09:08:17,325 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-27 09:08:17,595 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 09:08:17,650 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-27 09:08:17,651 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 09:08:17,651 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 09:08:17,661 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 09:08:17,663 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 09:08:18,464 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 09:08:18,466 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-27 09:08:18,630 INFO Regime phase HTF dataset build: 7.3s (103290 samples)
2026-04-27 09:08:18,631 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260427_090818
2026-04-27 09:08:18,832 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=34, n_classes=3)
2026-04-27 09:08:18,833 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=38114 dropped=65176 classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728})
2026-04-27 09:08:18,842 INFO RegimeClassifier[mode=htf_bias]: 38114 samples, classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728}, device=cuda
2026-04-27 09:08:18,843 INFO RegimeClassifier: sample weights — mean=0.708  ambiguous(<0.4)=0.0%
2026-04-27 09:08:18,843 INFO RegimeClassifier[mode=htf_bias]: warm start from existing weights
2026-04-27 09:08:18,843 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-27 09:08:21,134 INFO Regime epoch  1/50 — tr=0.4851 va=1.1194 acc=0.969 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.825}
2026-04-27 09:08:21,205 INFO Regime epoch  2/50 — tr=0.4853 va=1.1192 acc=0.969
2026-04-27 09:08:21,276 INFO Regime epoch  3/50 — tr=0.4857 va=1.1205 acc=0.968
2026-04-27 09:08:21,342 INFO Regime epoch  4/50 — tr=0.4847 va=1.1202 acc=0.968
2026-04-27 09:08:21,416 INFO Regime epoch  5/50 — tr=0.4852 va=1.1195 acc=0.968 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.824}
2026-04-27 09:08:21,489 INFO Regime epoch  6/50 — tr=0.4846 va=1.1174 acc=0.968
2026-04-27 09:08:21,561 INFO Regime epoch  7/50 — tr=0.4832 va=1.1171 acc=0.969
2026-04-27 09:08:21,631 INFO Regime epoch  8/50 — tr=0.4839 va=1.1113 acc=0.969
2026-04-27 09:08:21,702 INFO Regime epoch  9/50 — tr=0.4825 va=1.1108 acc=0.969
2026-04-27 09:08:21,777 INFO Regime epoch 10/50 — tr=0.4824 va=1.1118 acc=0.969 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.826}
2026-04-27 09:08:21,846 INFO Regime epoch 11/50 — tr=0.4823 va=1.1086 acc=0.970
2026-04-27 09:08:21,915 INFO Regime epoch 12/50 — tr=0.4814 va=1.1052 acc=0.970
2026-04-27 09:08:21,985 INFO Regime epoch 13/50 — tr=0.4804 va=1.1026 acc=0.970
2026-04-27 09:08:22,053 INFO Regime epoch 14/50 — tr=0.4804 va=1.1020 acc=0.970
2026-04-27 09:08:22,125 INFO Regime epoch 15/50 — tr=0.4803 va=1.0990 acc=0.971 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.84}
2026-04-27 09:08:22,194 INFO Regime epoch 16/50 — tr=0.4800 va=1.1012 acc=0.971
2026-04-27 09:08:22,260 INFO Regime epoch 17/50 — tr=0.4791 va=1.0979 acc=0.972
2026-04-27 09:08:22,327 INFO Regime epoch 18/50 — tr=0.4785 va=1.0928 acc=0.973
2026-04-27 09:08:22,393 INFO Regime epoch 19/50 — tr=0.4784 va=1.0944 acc=0.973
2026-04-27 09:08:22,466 INFO Regime epoch 20/50 — tr=0.4784 va=1.0915 acc=0.973 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.851}
2026-04-27 09:08:22,534 INFO Regime epoch 21/50 — tr=0.4782 va=1.0919 acc=0.974
2026-04-27 09:08:22,605 INFO Regime epoch 22/50 — tr=0.4773 va=1.0906 acc=0.973
2026-04-27 09:08:22,674 INFO Regime epoch 23/50 — tr=0.4771 va=1.0890 acc=0.974
2026-04-27 09:08:22,740 INFO Regime epoch 24/50 — tr=0.4768 va=1.0869 acc=0.974
2026-04-27 09:08:22,811 INFO Regime epoch 25/50 — tr=0.4768 va=1.0852 acc=0.974 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.854}
2026-04-27 09:08:22,880 INFO Regime epoch 26/50 — tr=0.4764 va=1.0855 acc=0.974
2026-04-27 09:08:22,946 INFO Regime epoch 27/50 — tr=0.4762 va=1.0835 acc=0.974
2026-04-27 09:08:23,016 INFO Regime epoch 28/50 — tr=0.4753 va=1.0837 acc=0.975
2026-04-27 09:08:23,085 INFO Regime epoch 29/50 — tr=0.4755 va=1.0818 acc=0.975
2026-04-27 09:08:23,157 INFO Regime epoch 30/50 — tr=0.4753 va=1.0817 acc=0.975 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.863}
2026-04-27 09:08:23,228 INFO Regime epoch 31/50 — tr=0.4746 va=1.0782 acc=0.976
2026-04-27 09:08:23,296 INFO Regime epoch 32/50 — tr=0.4746 va=1.0779 acc=0.976
2026-04-27 09:08:23,364 INFO Regime epoch 33/50 — tr=0.4743 va=1.0787 acc=0.975
2026-04-27 09:08:23,434 INFO Regime epoch 34/50 — tr=0.4749 va=1.0778 acc=0.976
2026-04-27 09:08:23,508 INFO Regime epoch 35/50 — tr=0.4746 va=1.0782 acc=0.975 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.863}
2026-04-27 09:08:23,575 INFO Regime epoch 36/50 — tr=0.4751 va=1.0762 acc=0.975
2026-04-27 09:08:23,644 INFO Regime epoch 37/50 — tr=0.4746 va=1.0729 acc=0.976
2026-04-27 09:08:23,712 INFO Regime epoch 38/50 — tr=0.4745 va=1.0751 acc=0.976
2026-04-27 09:08:23,778 INFO Regime epoch 39/50 — tr=0.4746 va=1.0750 acc=0.976
2026-04-27 09:08:23,848 INFO Regime epoch 40/50 — tr=0.4737 va=1.0720 acc=0.976 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.867}
2026-04-27 09:08:23,917 INFO Regime epoch 41/50 — tr=0.4743 va=1.0718 acc=0.976
2026-04-27 09:08:23,984 INFO Regime epoch 42/50 — tr=0.4746 va=1.0742 acc=0.976
2026-04-27 09:08:24,052 INFO Regime epoch 43/50 — tr=0.4741 va=1.0734 acc=0.976
2026-04-27 09:08:24,124 INFO Regime epoch 44/50 — tr=0.4740 va=1.0720 acc=0.977
2026-04-27 09:08:24,196 INFO Regime epoch 45/50 — tr=0.4741 va=1.0708 acc=0.976 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.867}
2026-04-27 09:08:24,262 INFO Regime epoch 46/50 — tr=0.4743 va=1.0709 acc=0.976
2026-04-27 09:08:24,329 INFO Regime epoch 47/50 — tr=0.4738 va=1.0728 acc=0.976
2026-04-27 09:08:24,398 INFO Regime epoch 48/50 — tr=0.4744 va=1.0716 acc=0.976
2026-04-27 09:08:24,473 INFO Regime epoch 49/50 — tr=0.4741 va=1.0715 acc=0.976
2026-04-27 09:08:24,550 INFO Regime epoch 50/50 — tr=0.4740 va=1.0725 acc=0.976 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.865}
2026-04-27 09:08:24,559 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 09:08:24,559 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 09:08:24,560 INFO Regime phase HTF train: 5.7s
2026-04-27 09:08:24,689 INFO Regime HTF complete: acc=0.976, n=103290 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.867}
2026-04-27 09:08:24,691 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 09:08:24,846 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 09:08:24,849 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 11.41578947368421, 'BIAS_DOWN': 10.635761589403973, 'BIAS_NEUTRAL': 17.960468521229867}
2026-04-27 09:08:24,853 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 4338, 'mean': 0.00011110466582543088, 'mean_over_std': 0.027632581210502927}, 'BIAS_DOWN': {'n': 3212, 'mean': 6.063430222660252e-05, 'mean_over_std': 0.013898489590377157}, 'BIAS_NEUTRAL': {'n': 12266, 'mean': 1.2151554902814046e-05, 'mean_over_std': 0.0031957000513162357}}
2026-04-27 09:08:24,854 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 3651, 'mean': 0.0002852242108836959, 'mean_over_std': 0.07015017232491755}, 'BIAS_DOWN': {'n': 2644, 'mean': -0.00014900749618981248, 'mean_over_std': -0.03333037041298237}, 'BIAS_NEUTRAL': {'n': 1376, 'mean': 9.595797941249082e-05, 'mean_over_std': 0.02492411099521803}}
2026-04-27 09:08:24,997 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-27 09:08:24,998 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:25,000 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:25,002 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:25,004 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:25,006 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:25,007 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:25,009 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:25,011 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:25,012 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:25,014 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:25,017 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 09:08:25,027 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:25,032 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:25,032 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:25,033 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:25,033 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:25,035 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:25,732 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-27 09:08:25,735 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-27 09:08:25,883 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:25,886 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:25,887 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:25,887 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:25,887 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:25,891 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:26,499 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-27 09:08:26,502 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-27 09:08:26,644 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:26,646 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:26,647 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:26,647 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:26,647 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:26,650 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:27,236 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-27 09:08:27,239 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-27 09:08:27,370 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:27,372 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:27,373 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:27,373 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:27,373 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:27,375 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:27,971 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-27 09:08:27,974 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-27 09:08:28,119 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:28,122 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:28,123 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:28,123 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:28,124 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:28,126 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:28,742 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-27 09:08:28,745 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-27 09:08:28,885 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:28,887 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:28,888 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:28,889 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:28,889 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:28,891 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:29,489 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-27 09:08:29,492 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-27 09:08:29,627 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 09:08:29,629 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-27 09:08:29,629 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 09:08:29,630 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 09:08:29,630 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 09:08:29,632 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:30,219 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-27 09:08:30,222 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-27 09:08:30,358 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:30,361 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:30,362 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:30,362 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:30,362 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:30,364 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:30,988 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-27 09:08:30,991 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-27 09:08:31,130 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:31,132 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:31,133 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:31,133 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:31,134 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:31,135 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:31,735 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-27 09:08:31,738 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-27 09:08:31,879 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:31,881 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:31,882 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:31,882 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:31,883 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 09:08:31,885 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 09:08:32,500 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-27 09:08:32,503 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-27 09:08:32,647 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 09:08:32,650 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-27 09:08:32,651 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 09:08:32,652 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 09:08:32,652 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 09:08:32,656 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 09:08:33,972 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 09:08:33,978 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-27 09:08:34,278 INFO Regime phase LTF dataset build: 9.3s (401471 samples)
2026-04-27 09:08:34,279 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260427_090834
2026-04-27 09:08:34,284 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=18, n_classes=4)
2026-04-27 09:08:34,287 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=299185 dropped=102286 classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127})
2026-04-27 09:08:34,347 INFO RegimeClassifier[mode=ltf_behaviour]: 299185 samples, classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127}, device=cuda
2026-04-27 09:08:34,348 INFO RegimeClassifier: sample weights — mean=0.693  ambiguous(<0.4)=0.0%
2026-04-27 09:08:34,348 INFO RegimeClassifier[mode=ltf_behaviour]: warm start from existing weights
2026-04-27 09:08:34,349 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-27 09:08:34,939 INFO Regime epoch  1/50 — tr=0.6226 va=1.2304 acc=0.832 per_class={'TRENDING': 0.811, 'RANGING': 0.766, 'CONSOLIDATING': 0.862, 'VOLATILE': 0.892}
2026-04-27 09:08:35,454 INFO Regime epoch  2/50 — tr=0.6225 va=1.2341 acc=0.834
2026-04-27 09:08:35,994 INFO Regime epoch  3/50 — tr=0.6222 va=1.2312 acc=0.833
2026-04-27 09:08:36,529 INFO Regime epoch  4/50 — tr=0.6221 va=1.2356 acc=0.835
2026-04-27 09:08:37,078 INFO Regime epoch  5/50 — tr=0.6224 va=1.2334 acc=0.833 per_class={'TRENDING': 0.814, 'RANGING': 0.763, 'CONSOLIDATING': 0.864, 'VOLATILE': 0.891}
2026-04-27 09:08:37,599 INFO Regime epoch  6/50 — tr=0.6221 va=1.2340 acc=0.835
2026-04-27 09:08:38,107 INFO Regime epoch  7/50 — tr=0.6222 va=1.2308 acc=0.832
2026-04-27 09:08:38,610 INFO Regime epoch  8/50 — tr=0.6218 va=1.2319 acc=0.837
2026-04-27 09:08:39,136 INFO Regime epoch  9/50 — tr=0.6217 va=1.2346 acc=0.836
2026-04-27 09:08:39,694 INFO Regime epoch 10/50 — tr=0.6215 va=1.2266 acc=0.834 per_class={'TRENDING': 0.817, 'RANGING': 0.762, 'CONSOLIDATING': 0.876, 'VOLATILE': 0.885}
2026-04-27 09:08:40,204 INFO Regime epoch 11/50 — tr=0.6213 va=1.2288 acc=0.837
2026-04-27 09:08:40,703 INFO Regime epoch 12/50 — tr=0.6209 va=1.2272 acc=0.835
2026-04-27 09:08:41,211 INFO Regime epoch 13/50 — tr=0.6208 va=1.2250 acc=0.836
2026-04-27 09:08:41,711 INFO Regime epoch 14/50 — tr=0.6207 va=1.2287 acc=0.838
2026-04-27 09:08:42,252 INFO Regime epoch 15/50 — tr=0.6203 va=1.2284 acc=0.838 per_class={'TRENDING': 0.824, 'RANGING': 0.762, 'CONSOLIDATING': 0.869, 'VOLATILE': 0.89}
2026-04-27 09:08:42,746 INFO Regime epoch 16/50 — tr=0.6203 va=1.2261 acc=0.838
2026-04-27 09:08:43,248 INFO Regime epoch 17/50 — tr=0.6198 va=1.2275 acc=0.836
2026-04-27 09:08:43,756 INFO Regime epoch 18/50 — tr=0.6199 va=1.2262 acc=0.838
2026-04-27 09:08:44,260 INFO Regime epoch 19/50 — tr=0.6198 va=1.2244 acc=0.837
2026-04-27 09:08:44,825 INFO Regime epoch 20/50 — tr=0.6197 va=1.2182 acc=0.837 per_class={'TRENDING': 0.82, 'RANGING': 0.768, 'CONSOLIDATING': 0.869, 'VOLATILE': 0.888}
2026-04-27 09:08:45,323 INFO Regime epoch 21/50 — tr=0.6195 va=1.2227 acc=0.838
2026-04-27 09:08:45,834 INFO Regime epoch 22/50 — tr=0.6194 va=1.2204 acc=0.837
2026-04-27 09:08:46,385 INFO Regime epoch 23/50 — tr=0.6192 va=1.2208 acc=0.837
2026-04-27 09:08:46,901 INFO Regime epoch 24/50 — tr=0.6190 va=1.2209 acc=0.839
2026-04-27 09:08:47,457 INFO Regime epoch 25/50 — tr=0.6190 va=1.2261 acc=0.840 per_class={'TRENDING': 0.826, 'RANGING': 0.765, 'CONSOLIDATING': 0.866, 'VOLATILE': 0.894}
2026-04-27 09:08:47,970 INFO Regime epoch 26/50 — tr=0.6191 va=1.2184 acc=0.838
2026-04-27 09:08:48,519 INFO Regime epoch 27/50 — tr=0.6187 va=1.2210 acc=0.839
2026-04-27 09:08:49,052 INFO Regime epoch 28/50 — tr=0.6187 va=1.2231 acc=0.840
2026-04-27 09:08:49,587 INFO Regime epoch 29/50 — tr=0.6187 va=1.2188 acc=0.839
2026-04-27 09:08:50,171 INFO Regime epoch 30/50 — tr=0.6186 va=1.2213 acc=0.840 per_class={'TRENDING': 0.827, 'RANGING': 0.765, 'CONSOLIDATING': 0.868, 'VOLATILE': 0.892}
2026-04-27 09:08:50,171 INFO Regime early stop at epoch 30 (no_improve=10)
2026-04-27 09:08:50,211 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 09:08:50,211 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 09:08:50,213 INFO Regime phase LTF train: 15.9s
2026-04-27 09:08:50,345 INFO Regime LTF complete: acc=0.837, n=401471 per_class={'TRENDING': 0.82, 'RANGING': 0.768, 'CONSOLIDATING': 0.869, 'VOLATILE': 0.888}
2026-04-27 09:08:50,349 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 09:08:50,860 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 09:08:50,864 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 9.355291913830783, 'RANGING': 6.096303199751476, 'CONSOLIDATING': 5.598885793871866, 'VOLATILE': 6.771351107094442}
2026-04-27 09:08:50,873 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 29965, 'mean': 2.149062259083323e-05, 'mean_over_std': 0.011155601434349563}, 'RANGING': {'n': 19623, 'mean': -3.254493608872846e-06, 'mean_over_std': -0.001761415548292494}, 'CONSOLIDATING': {'n': 10050, 'mean': -3.039756158146958e-06, 'mean_over_std': -0.001792488091618151}, 'VOLATILE': {'n': 14985, 'mean': 1.8102764458593193e-05, 'mean_over_std': 0.006974696291378742}}
2026-04-27 09:08:50,874 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 26145, 'mean': 2.874435605138053e-05, 'mean_over_std': 0.01595450632691917}, 'RANGING': {'n': 7858, 'mean': 1.7829776785521727e-05, 'mean_over_std': 0.011851561804467463}, 'CONSOLIDATING': {'n': 8543, 'mean': 7.845231961250924e-06, 'mean_over_std': 0.004915839521549481}, 'VOLATILE': {'n': 13636, 'mean': 5.480114267306668e-06, 'mean_over_std': 0.0020378119917389375}}
2026-04-27 09:08:51,008 INFO Regime retrain total: 209.5s (504761 samples)
2026-04-27 09:08:51,138 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-27 09:08:51,138 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 09:08:51,138 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 09:08:51,138 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-27 09:08:51,139 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-27 09:08:51,139 INFO Retrain complete. Total wall-clock: 209.6s
  DONE  Retrain regime [train-split retrain]
  START Retrain quality [train-split retrain]
2026-04-27 09:08:52,464 INFO retrain environment: KAGGLE
2026-04-27 09:08:54,087 INFO Device: CUDA (2 GPU(s))
2026-04-27 09:08:54,097 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 09:08:54,097 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 09:08:54,097 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 09:08:54,097 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 09:08:54,097 INFO Retrain data split: train
2026-04-27 09:08:54,098 INFO === QualityScorer retrain ===
2026-04-27 09:08:54,247 INFO NumExpr defaulting to 4 threads.
2026-04-27 09:08:54,446 INFO QualityScorer: CUDA available — using GPU
2026-04-27 09:08:55,105 INFO QualityScorer: skipped 29063 journal records outside allowed splits ['live', 'paper', 'production', 'train']
2026-04-27 09:08:55,106 INFO Quality phase label creation: 0.7s (0 trades)
2026-04-27 09:08:55,235 INFO Retrain complete. Total wall-clock: 1.1s
  WARN  Retrain quality failed (exit 1) — continuing
  START Retrain rl [train-split retrain]
2026-04-27 09:08:55,852 INFO retrain environment: KAGGLE
2026-04-27 09:08:57,501 INFO Device: CUDA (2 GPU(s))
2026-04-27 09:08:57,512 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 09:08:57,512 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 09:08:57,512 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 09:08:57,513 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 09:08:57,513 INFO Retrain data split: train
2026-04-27 09:08:57,514 INFO === RLAgent (PPO) retrain ===
2026-04-27 09:08:57,520 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260427_090857
2026-04-27 09:08:58,215 INFO RLAgent: skipped 29063 journal records outside allowed splits ['live', 'paper', 'production', 'train']
2026-04-27 09:08:58,215 INFO RL phase episode loading: 0.7s (0 episodes)
2026-04-27 09:09:01.237210: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777280941.423240   81982 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777280941.480200   81982 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777280941.905040   81982 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777280941.905079   81982 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777280941.905082   81982 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777280941.905084   81982 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-27 09:09:15,097 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-27 09:09:17,747 INFO RLAgent: skipped 29063 journal records outside allowed splits ['live', 'paper', 'production', 'train']
2026-04-27 09:09:17,747 WARNING RLAgent.retrain: only 0 episodes — skipping
2026-04-27 09:09:17,747 INFO RL phase PPO train: 19.5s | total: 20.2s
2026-04-27 09:09:17,880 INFO Retrain complete. Total wall-clock: 20.4s
  WARN  Retrain rl failed (exit 1) — continuing

=== Round 3: Backtest on last 3yr (post-retrain evaluation) ===
  START Round 3 - Post-retrain backtest (last 3yr)
2026-04-27 09:09:19,807 INFO === STEP 6: BACKTEST (round3) ===
2026-04-27 09:09:19,808 INFO BT_WINDOW=round3 — post-retrain eval: 2022-08-05 → 2025-08-05 (last 3yr)
2026-04-27 09:09:19,808 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-04-27 09:09:19,808 INFO Round 3 — running backtest: 2022-08-05 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-27 09:09:22,260 WARNING QualityScorer unavailable (weights missing or load failed)
2026-04-27 09:14:08,492 INFO Round 3 backtest — 7649 trades | avg WR=32.3% | avg PF=1.40 | avg Sharpe=0.75
2026-04-27 09:14:08,492 INFO   ml_trader: 7649 trades | WR=32.3% | PF=1.40 | Return=2968903.6% | DD=55.9% | Sharpe=0.75
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 7649
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (7649 rows)
  DONE  Round 3 - Post-retrain backtest (last 3yr)
  Saved Round 3 result → round3_summary.json

======================================================================
  BLIND BACKTEST PIPELINE COMPLETE
======================================================================
  Round 1 (val window)          trades=16100  WR=29.4%  PF=1.279  Sharpe=0.455
  Round 2 (blind test)          trades=12963  WR=29.7%  PF=1.267  Sharpe=0.494
  Round 3 (last 3yr)            trades=7649  WR=32.3%  PF=1.404  Sharpe=0.754


WARNING: GITHUB_TOKEN not set — skipping GitHub push