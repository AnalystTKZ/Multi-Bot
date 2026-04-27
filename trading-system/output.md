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
2026-04-27 14:58:15,162 INFO Loading feature-engineered data...
2026-04-27 14:58:15,930 INFO Loaded 221743 rows, 202 features
2026-04-27 14:58:15,932 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-27 14:58:15,936 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-27 14:58:15,936 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-27 14:58:15,936 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-27 14:58:15,936 INFO No leakage confirmed: train < val < test timestamps

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
2026-04-27 14:58:18,462 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-27 14:58:18,462 INFO --- Training regime ---
2026-04-27 14:58:18,463 INFO Running retrain --model regime
2026-04-27 14:58:18,657 INFO retrain environment: KAGGLE
2026-04-27 14:58:20,431 INFO Device: CUDA (2 GPU(s))
2026-04-27 14:58:20,442 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 14:58:20,442 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 14:58:20,442 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 14:58:20,445 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 14:58:20,446 INFO Retrain data split: train
2026-04-27 14:58:20,447 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-27 14:58:20,610 INFO NumExpr defaulting to 4 threads.
2026-04-27 14:58:20,875 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-27 14:58:20,875 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 14:58:20,875 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 14:58:20,876 INFO Regime phase macro_correlations: 0.0s
2026-04-27 14:58:20,876 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-27 14:58:20,920 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-27 14:58:20,921 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 14:58:20,950 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 14:58:20,965 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 14:58:20,989 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 14:58:21,004 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 14:58:21,028 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 14:58:21,042 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 14:58:21,069 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 14:58:21,091 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 14:58:21,124 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 14:58:21,149 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 14:58:21,172 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 14:58:21,186 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 14:58:21,207 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 14:58:21,223 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 14:58:21,244 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 14:58:21,259 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 14:58:21,282 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 14:58:21,296 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 14:58:21,318 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 14:58:21,334 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 14:58:21,373 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 14:58:22,800 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 14:58:45,672 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 14:58:45,677 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 24.3s
2026-04-27 14:58:45,677 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 14:58:56,579 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 14:58:56,583 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 10.9s
2026-04-27 14:58:56,583 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 14:59:04,371 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 14:59:04,374 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 7.8s
2026-04-27 14:59:04,375 INFO Regime phase GMM HTF total: 43.0s
2026-04-27 14:59:04,375 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 15:00:15,777 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 15:00:15,781 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 71.4s
2026-04-27 15:00:15,781 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 15:00:47,162 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 15:00:47,166 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 31.4s
2026-04-27 15:00:47,166 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 15:01:09,604 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 15:01:09,605 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 22.4s
2026-04-27 15:01:09,605 INFO Regime phase GMM LTF total: 125.2s
2026-04-27 15:01:09,726 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-27 15:01:09,728 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:09,729 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:09,730 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:09,731 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:09,731 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:09,732 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:09,733 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:09,734 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:09,735 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:09,735 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:09,736 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:01:09,866 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:09,913 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:09,914 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:09,914 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:09,923 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:09,924 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:10,367 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 17, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 8360}  ambiguous=8312 (total=8402)  short_runs_zeroed=15
2026-04-27 15:01:10,369 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-27 15:01:10,562 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:10,599 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:10,600 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:10,600 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:10,608 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:10,609 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:11,014 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 100, 'BIAS_DOWN': 31, 'BIAS_NEUTRAL': 8271}  ambiguous=8272 (total=8402)  short_runs_zeroed=52
2026-04-27 15:01:11,015 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-27 15:01:11,248 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:11,283 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:11,284 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:11,285 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:11,293 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:11,294 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:11,694 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 46, 'BIAS_DOWN': 4, 'BIAS_NEUTRAL': 8352}  ambiguous=8316 (total=8402)  short_runs_zeroed=28
2026-04-27 15:01:11,695 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-27 15:01:11,893 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:11,932 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:11,933 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:11,934 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:11,942 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:11,943 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:12,344 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 15, 'BIAS_DOWN': 19, 'BIAS_NEUTRAL': 8368}  ambiguous=8331 (total=8402)  short_runs_zeroed=27
2026-04-27 15:01:12,345 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-27 15:01:12,560 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:12,596 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:12,597 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:12,597 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:12,605 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:12,606 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:13,009 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 93, 'BIAS_DOWN': 58, 'BIAS_NEUTRAL': 8252}  ambiguous=8287 (total=8403)  short_runs_zeroed=75
2026-04-27 15:01:13,010 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-27 15:01:13,211 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:13,246 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:13,247 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:13,247 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:13,256 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:13,257 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:13,649 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 87, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 8261}  ambiguous=8278 (total=8403)  short_runs_zeroed=63
2026-04-27 15:01:13,651 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-27 15:01:13,822 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:01:13,851 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:01:13,852 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:01:13,852 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:01:13,859 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:01:13,860 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:14,251 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 21, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 8314}  ambiguous=8303 (total=8402)  short_runs_zeroed=39
2026-04-27 15:01:14,252 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-27 15:01:14,457 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:14,491 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:14,492 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:14,493 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:14,500 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:14,501 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:14,899 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 37, 'BIAS_DOWN': 38, 'BIAS_NEUTRAL': 8327}  ambiguous=8321 (total=8402)  short_runs_zeroed=44
2026-04-27 15:01:14,900 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-27 15:01:15,096 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:15,131 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:15,132 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:15,132 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:15,140 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:15,141 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:15,546 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 79, 'BIAS_DOWN': 33, 'BIAS_NEUTRAL': 8290}  ambiguous=8328 (total=8402)  short_runs_zeroed=61
2026-04-27 15:01:15,547 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-27 15:01:15,744 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:15,781 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:15,782 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:15,782 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:15,790 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:15,792 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:16,191 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 42, 'BIAS_DOWN': 44, 'BIAS_NEUTRAL': 8317}  ambiguous=8318 (total=8403)  short_runs_zeroed=44
2026-04-27 15:01:16,192 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-27 15:01:16,481 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:01:16,544 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:01:16,545 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:01:16,545 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:01:16,556 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:01:16,557 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:01:17,401 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 107, 'BIAS_DOWN': 115, 'BIAS_NEUTRAL': 19595}  ambiguous=19636 (total=19817)  short_runs_zeroed=112
2026-04-27 15:01:17,403 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-27 15:01:17,615 INFO Regime phase HTF dataset build: 7.9s (103290 samples)
2026-04-27 15:01:17,616 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=1138 dropped=102152 classes={'BIAS_UP': 399, 'BIAS_DOWN': 243, 'BIAS_NEUTRAL': 496})
2026-04-27 15:01:17,617 INFO RegimeClassifier[mode=htf_bias]: 1138 samples, classes={'BIAS_UP': 399, 'BIAS_DOWN': 243, 'BIAS_NEUTRAL': 496}, device=cuda
2026-04-27 15:01:17,617 INFO RegimeClassifier: sample weights — mean=0.713  ambiguous(<0.4)=0.0%
2026-04-27 15:01:17,998 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-04-27 15:01:17,998 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-27 15:01:24,314 INFO Regime epoch  1/50 — tr=0.7471 va=2.1752 acc=0.338 per_class={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 1.0}
2026-04-27 15:01:24,331 INFO Regime epoch  2/50 — tr=0.7525 va=2.1023 acc=0.338
2026-04-27 15:01:24,345 INFO Regime epoch  3/50 — tr=0.7565 va=2.0749 acc=0.338
2026-04-27 15:01:24,359 INFO Regime epoch  4/50 — tr=0.7547 va=2.0568 acc=0.338
2026-04-27 15:01:24,376 INFO Regime epoch  5/50 — tr=0.7371 va=2.0410 acc=0.338 per_class={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 1.0}
2026-04-27 15:01:24,390 INFO Regime epoch  6/50 — tr=0.7452 va=2.0251 acc=0.338
2026-04-27 15:01:24,404 INFO Regime epoch  7/50 — tr=0.7313 va=2.0081 acc=0.338
2026-04-27 15:01:24,418 INFO Regime epoch  8/50 — tr=0.7208 va=1.9890 acc=0.338
2026-04-27 15:01:24,431 INFO Regime epoch  9/50 — tr=0.7152 va=1.9660 acc=0.338
2026-04-27 15:01:24,447 INFO Regime epoch 10/50 — tr=0.7118 va=1.9409 acc=0.338 per_class={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 1.0}
2026-04-27 15:01:24,460 INFO Regime epoch 11/50 — tr=0.6883 va=1.9139 acc=0.351
2026-04-27 15:01:24,474 INFO Regime epoch 12/50 — tr=0.6783 va=1.8859 acc=0.373
2026-04-27 15:01:24,487 INFO Regime epoch 13/50 — tr=0.6687 va=1.8566 acc=0.395
2026-04-27 15:01:24,502 INFO Regime epoch 14/50 — tr=0.6639 va=1.8270 acc=0.575
2026-04-27 15:01:24,519 INFO Regime epoch 15/50 — tr=0.6505 va=1.7963 acc=0.627 per_class={'BIAS_UP': 0.203, 'BIAS_DOWN': 0.662, 'BIAS_NEUTRAL': 1.0}
2026-04-27 15:01:24,531 INFO Regime epoch 16/50 — tr=0.6476 va=1.7650 acc=0.671
2026-04-27 15:01:24,543 INFO Regime epoch 17/50 — tr=0.6332 va=1.7323 acc=0.715
2026-04-27 15:01:24,555 INFO Regime epoch 18/50 — tr=0.6258 va=1.6991 acc=0.741
2026-04-27 15:01:24,569 INFO Regime epoch 19/50 — tr=0.6168 va=1.6659 acc=0.776
2026-04-27 15:01:24,584 INFO Regime epoch 20/50 — tr=0.6071 va=1.6328 acc=0.803 per_class={'BIAS_UP': 0.5, 'BIAS_DOWN': 0.896, 'BIAS_NEUTRAL': 1.0}
2026-04-27 15:01:24,597 INFO Regime epoch 21/50 — tr=0.5974 va=1.6004 acc=0.829
2026-04-27 15:01:24,609 INFO Regime epoch 22/50 — tr=0.5863 va=1.5682 acc=0.838
2026-04-27 15:01:24,621 INFO Regime epoch 23/50 — tr=0.5842 va=1.5374 acc=0.846
2026-04-27 15:01:24,634 INFO Regime epoch 24/50 — tr=0.5802 va=1.5079 acc=0.855
2026-04-27 15:01:24,652 INFO Regime epoch 25/50 — tr=0.5761 va=1.4795 acc=0.877 per_class={'BIAS_UP': 0.703, 'BIAS_DOWN': 0.922, 'BIAS_NEUTRAL': 1.0}
2026-04-27 15:01:24,665 INFO Regime epoch 26/50 — tr=0.5655 va=1.4540 acc=0.886
2026-04-27 15:01:24,679 INFO Regime epoch 27/50 — tr=0.5673 va=1.4294 acc=0.886
2026-04-27 15:01:24,692 INFO Regime epoch 28/50 — tr=0.5561 va=1.4056 acc=0.895
2026-04-27 15:01:24,705 INFO Regime epoch 29/50 — tr=0.5569 va=1.3831 acc=0.908
2026-04-27 15:01:24,722 INFO Regime epoch 30/50 — tr=0.5561 va=1.3608 acc=0.921 per_class={'BIAS_UP': 0.797, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 1.0}
2026-04-27 15:01:24,734 INFO Regime epoch 31/50 — tr=0.5529 va=1.3391 acc=0.921
2026-04-27 15:01:24,746 INFO Regime epoch 32/50 — tr=0.5469 va=1.3192 acc=0.921
2026-04-27 15:01:24,759 INFO Regime epoch 33/50 — tr=0.5403 va=1.3022 acc=0.921
2026-04-27 15:01:24,773 INFO Regime epoch 34/50 — tr=0.5479 va=1.2887 acc=0.921
2026-04-27 15:01:24,792 INFO Regime epoch 35/50 — tr=0.5398 va=1.2747 acc=0.917 per_class={'BIAS_UP': 0.784, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 1.0}
2026-04-27 15:01:24,805 INFO Regime epoch 36/50 — tr=0.5372 va=1.2612 acc=0.921
2026-04-27 15:01:24,818 INFO Regime epoch 37/50 — tr=0.5413 va=1.2494 acc=0.921
2026-04-27 15:01:24,831 INFO Regime epoch 38/50 — tr=0.5383 va=1.2395 acc=0.921
2026-04-27 15:01:24,844 INFO Regime epoch 39/50 — tr=0.5404 va=1.2306 acc=0.921
2026-04-27 15:01:24,860 INFO Regime epoch 40/50 — tr=0.5401 va=1.2213 acc=0.921 per_class={'BIAS_UP': 0.797, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 1.0}
2026-04-27 15:01:24,873 INFO Regime epoch 41/50 — tr=0.5397 va=1.2137 acc=0.921
2026-04-27 15:01:24,886 INFO Regime epoch 42/50 — tr=0.5393 va=1.2086 acc=0.921
2026-04-27 15:01:24,899 INFO Regime epoch 43/50 — tr=0.5386 va=1.2051 acc=0.925
2026-04-27 15:01:24,912 INFO Regime epoch 44/50 — tr=0.5299 va=1.2013 acc=0.925
2026-04-27 15:01:24,930 INFO Regime epoch 45/50 — tr=0.5357 va=1.1994 acc=0.925 per_class={'BIAS_UP': 0.811, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 1.0}
2026-04-27 15:01:24,944 INFO Regime epoch 46/50 — tr=0.5349 va=1.1958 acc=0.925
2026-04-27 15:01:24,958 INFO Regime epoch 47/50 — tr=0.5309 va=1.1929 acc=0.925
2026-04-27 15:01:24,970 INFO Regime epoch 48/50 — tr=0.5344 va=1.1908 acc=0.925
2026-04-27 15:01:24,983 INFO Regime epoch 49/50 — tr=0.5398 va=1.1877 acc=0.921
2026-04-27 15:01:25,000 INFO Regime epoch 50/50 — tr=0.5363 va=1.1851 acc=0.921 per_class={'BIAS_UP': 0.797, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 1.0}
2026-04-27 15:01:25,010 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 15:01:25,010 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 15:01:25,010 INFO Regime phase HTF train: 7.4s
2026-04-27 15:01:25,165 INFO Regime HTF complete: acc=0.921, n=103290 per_class={'BIAS_UP': 0.797, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 1.0}
2026-04-27 15:01:25,167 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:01:25,339 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 107, 'BIAS_DOWN': 115, 'BIAS_NEUTRAL': 19595}  ambiguous=19636 (total=19817)  short_runs_zeroed=112
2026-04-27 15:01:25,355 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 4.28, 'BIAS_DOWN': 4.791666666666667, 'BIAS_NEUTRAL': 391.9}
2026-04-27 15:01:25,359 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 107, 'mean': -0.00021706688424743013, 'mean_over_std': -0.04806767606653151}, 'BIAS_DOWN': {'n': 115, 'mean': -0.00020797041876048362, 'mean_over_std': -0.029260022054973262}, 'BIAS_NEUTRAL': {'n': 19594, 'mean': 4.4550533138328785e-05, 'mean_over_std': 0.011372683463534268}}
2026-04-27 15:01:25,359 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 66, 'mean': 0.0003288215669218909, 'mean_over_std': 0.07277335807221066}, 'BIAS_DOWN': {'n': 59, 'mean': -0.0010984382215802496, 'mean_over_std': -0.13394112338746375}, 'BIAS_NEUTRAL': {'n': 56, 'mean': -0.00020920056804862467, 'mean_over_std': -0.06887192756862072}}
2026-04-27 15:01:25,363 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-27 15:01:25,365 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:25,367 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:25,368 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:25,370 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:25,372 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:25,373 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:25,375 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:25,376 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:25,378 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:25,379 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:25,382 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:01:25,395 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:25,398 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:25,399 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:25,400 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:25,400 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:25,402 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:26,074 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 619, 'RANGING': 24879, 'CONSOLIDATING': 1882, 'VOLATILE': 5358}  ambiguous=26508 (total=32738)  short_runs_zeroed=2571
2026-04-27 15:01:26,077 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-27 15:01:26,240 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:26,242 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:26,243 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:26,244 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:26,244 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:26,246 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:26,877 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 752, 'RANGING': 24932, 'CONSOLIDATING': 2099, 'VOLATILE': 4955}  ambiguous=26224 (total=32738)  short_runs_zeroed=2112
2026-04-27 15:01:26,880 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-27 15:01:27,038 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:27,040 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:27,041 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:27,042 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:27,042 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:27,044 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:27,685 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 617, 'RANGING': 24738, 'CONSOLIDATING': 2103, 'VOLATILE': 5282}  ambiguous=26101 (total=32740)  short_runs_zeroed=2218
2026-04-27 15:01:27,688 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-27 15:01:27,855 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:27,857 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:27,858 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:27,859 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:27,859 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:27,861 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:28,491 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 842, 'RANGING': 25174, 'CONSOLIDATING': 1996, 'VOLATILE': 4727}  ambiguous=26490 (total=32739)  short_runs_zeroed=2045
2026-04-27 15:01:28,495 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-27 15:01:28,651 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:28,653 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:28,654 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:28,655 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:28,655 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:28,657 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:29,283 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 761, 'RANGING': 24579, 'CONSOLIDATING': 2089, 'VOLATILE': 5311}  ambiguous=25954 (total=32740)  short_runs_zeroed=2172
2026-04-27 15:01:29,286 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-27 15:01:29,444 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:29,447 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:29,447 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:29,448 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:29,448 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:29,450 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:30,064 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 763, 'RANGING': 25215, 'CONSOLIDATING': 1968, 'VOLATILE': 4793}  ambiguous=26501 (total=32739)  short_runs_zeroed=1959
2026-04-27 15:01:30,067 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-27 15:01:30,221 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:01:30,223 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:01:30,223 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:01:30,224 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:01:30,224 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:01:30,226 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:30,857 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 474, 'RANGING': 25095, 'CONSOLIDATING': 1967, 'VOLATILE': 5203}  ambiguous=26572 (total=32739)  short_runs_zeroed=2307
2026-04-27 15:01:30,860 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-27 15:01:31,029 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:31,031 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:31,032 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:31,032 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:31,033 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:31,035 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:31,702 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 881, 'RANGING': 25133, 'CONSOLIDATING': 1987, 'VOLATILE': 4739}  ambiguous=26356 (total=32740)  short_runs_zeroed=1985
2026-04-27 15:01:31,705 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-27 15:01:31,874 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:31,877 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:31,878 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:31,878 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:31,879 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:31,881 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:32,535 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 699, 'RANGING': 24989, 'CONSOLIDATING': 2130, 'VOLATILE': 4923}  ambiguous=26228 (total=32741)  short_runs_zeroed=2004
2026-04-27 15:01:32,538 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-27 15:01:32,702 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:32,705 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:32,706 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:32,706 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:32,706 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:32,708 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:33,343 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 808, 'RANGING': 24290, 'CONSOLIDATING': 2180, 'VOLATILE': 5465}  ambiguous=25645 (total=32743)  short_runs_zeroed=2211
2026-04-27 15:01:33,346 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-27 15:01:33,520 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:01:33,524 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:01:33,525 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:01:33,525 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:01:33,526 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:01:33,529 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:01:34,892 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 1778, 'RANGING': 57285, 'CONSOLIDATING': 4542, 'VOLATILE': 11019}  ambiguous=60314 (total=74624)  short_runs_zeroed=4774
2026-04-27 15:01:34,899 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-27 15:01:35,249 INFO Regime phase LTF dataset build: 9.9s (401471 samples)
2026-04-27 15:01:35,251 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=79106 dropped=322365 classes={'TRENDING': 4775, 'RANGING': 1876, 'CONSOLIDATING': 18867, 'VOLATILE': 53588})
2026-04-27 15:01:35,267 INFO RegimeClassifier[mode=ltf_behaviour]: 79106 samples, classes={'TRENDING': 4775, 'RANGING': 1876, 'CONSOLIDATING': 18867, 'VOLATILE': 53588}, device=cuda
2026-04-27 15:01:35,267 INFO RegimeClassifier: sample weights — mean=0.811  ambiguous(<0.4)=0.0%
2026-04-27 15:01:35,269 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-04-27 15:01:35,270 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-27 15:01:35,461 INFO Regime epoch  1/50 — tr=1.0801 va=2.2137 acc=0.184 per_class={'TRENDING': 0.384, 'RANGING': 0.18, 'CONSOLIDATING': 0.635, 'VOLATILE': 0.003}
2026-04-27 15:01:35,601 INFO Regime epoch  2/50 — tr=1.0727 va=2.1637 acc=0.175
2026-04-27 15:01:35,742 INFO Regime epoch  3/50 — tr=1.0512 va=2.0993 acc=0.215
2026-04-27 15:01:35,890 INFO Regime epoch  4/50 — tr=1.0179 va=2.0075 acc=0.429
2026-04-27 15:01:36,046 INFO Regime epoch  5/50 — tr=0.9736 va=1.8911 acc=0.714 per_class={'TRENDING': 0.609, 'RANGING': 0.429, 'CONSOLIDATING': 0.754, 'VOLATILE': 0.72}
2026-04-27 15:01:36,192 INFO Regime epoch  6/50 — tr=0.9196 va=1.7604 acc=0.799
2026-04-27 15:01:36,335 INFO Regime epoch  7/50 — tr=0.8678 va=1.6449 acc=0.837
2026-04-27 15:01:36,479 INFO Regime epoch  8/50 — tr=0.8193 va=1.5629 acc=0.855
2026-04-27 15:01:36,620 INFO Regime epoch  9/50 — tr=0.7826 va=1.4957 acc=0.871
2026-04-27 15:01:36,767 INFO Regime epoch 10/50 — tr=0.7520 va=1.4380 acc=0.889 per_class={'TRENDING': 0.833, 'RANGING': 0.541, 'CONSOLIDATING': 0.948, 'VOLATILE': 0.887}
2026-04-27 15:01:36,911 INFO Regime epoch 11/50 — tr=0.7330 va=1.3783 acc=0.895
2026-04-27 15:01:37,049 INFO Regime epoch 12/50 — tr=0.7175 va=1.3401 acc=0.908
2026-04-27 15:01:37,192 INFO Regime epoch 13/50 — tr=0.7056 va=1.3037 acc=0.916
2026-04-27 15:01:37,335 INFO Regime epoch 14/50 — tr=0.6962 va=1.2686 acc=0.918
2026-04-27 15:01:37,488 INFO Regime epoch 15/50 — tr=0.6887 va=1.2456 acc=0.923 per_class={'TRENDING': 0.843, 'RANGING': 0.72, 'CONSOLIDATING': 0.933, 'VOLATILE': 0.935}
2026-04-27 15:01:37,628 INFO Regime epoch 16/50 — tr=0.6823 va=1.2203 acc=0.926
2026-04-27 15:01:37,765 INFO Regime epoch 17/50 — tr=0.6773 va=1.1967 acc=0.932
2026-04-27 15:01:37,903 INFO Regime epoch 18/50 — tr=0.6744 va=1.1764 acc=0.933
2026-04-27 15:01:38,036 INFO Regime epoch 19/50 — tr=0.6692 va=1.1576 acc=0.932
2026-04-27 15:01:38,188 INFO Regime epoch 20/50 — tr=0.6664 va=1.1479 acc=0.935 per_class={'TRENDING': 0.827, 'RANGING': 0.78, 'CONSOLIDATING': 0.924, 'VOLATILE': 0.956}
2026-04-27 15:01:38,333 INFO Regime epoch 21/50 — tr=0.6642 va=1.1333 acc=0.935
2026-04-27 15:01:38,483 INFO Regime epoch 22/50 — tr=0.6623 va=1.1225 acc=0.936
2026-04-27 15:01:38,632 INFO Regime epoch 23/50 — tr=0.6594 va=1.1105 acc=0.937
2026-04-27 15:01:38,780 INFO Regime epoch 24/50 — tr=0.6543 va=1.1035 acc=0.937
2026-04-27 15:01:38,939 INFO Regime epoch 25/50 — tr=0.6530 va=1.0906 acc=0.937 per_class={'TRENDING': 0.81, 'RANGING': 0.798, 'CONSOLIDATING': 0.91, 'VOLATILE': 0.964}
2026-04-27 15:01:39,096 INFO Regime epoch 26/50 — tr=0.6497 va=1.0817 acc=0.938
2026-04-27 15:01:39,243 INFO Regime epoch 27/50 — tr=0.6510 va=1.0763 acc=0.938
2026-04-27 15:01:39,392 INFO Regime epoch 28/50 — tr=0.6487 va=1.0663 acc=0.937
2026-04-27 15:01:39,529 INFO Regime epoch 29/50 — tr=0.6460 va=1.0650 acc=0.937
2026-04-27 15:01:39,673 INFO Regime epoch 30/50 — tr=0.6467 va=1.0562 acc=0.937 per_class={'TRENDING': 0.816, 'RANGING': 0.81, 'CONSOLIDATING': 0.903, 'VOLATILE': 0.965}
2026-04-27 15:01:39,816 INFO Regime epoch 31/50 — tr=0.6460 va=1.0485 acc=0.937
2026-04-27 15:01:39,964 INFO Regime epoch 32/50 — tr=0.6433 va=1.0455 acc=0.938
2026-04-27 15:01:40,111 INFO Regime epoch 33/50 — tr=0.6430 va=1.0411 acc=0.938
2026-04-27 15:01:40,258 INFO Regime epoch 34/50 — tr=0.6437 va=1.0356 acc=0.937
2026-04-27 15:01:40,414 INFO Regime epoch 35/50 — tr=0.6438 va=1.0322 acc=0.937 per_class={'TRENDING': 0.811, 'RANGING': 0.822, 'CONSOLIDATING': 0.9, 'VOLATILE': 0.966}
2026-04-27 15:01:40,557 INFO Regime epoch 36/50 — tr=0.6400 va=1.0277 acc=0.938
2026-04-27 15:01:40,701 INFO Regime epoch 37/50 — tr=0.6398 va=1.0248 acc=0.937
2026-04-27 15:01:40,848 INFO Regime epoch 38/50 — tr=0.6402 va=1.0264 acc=0.938
2026-04-27 15:01:40,985 INFO Regime epoch 39/50 — tr=0.6391 va=1.0249 acc=0.938
2026-04-27 15:01:41,159 INFO Regime epoch 40/50 — tr=0.6385 va=1.0226 acc=0.938 per_class={'TRENDING': 0.814, 'RANGING': 0.824, 'CONSOLIDATING': 0.907, 'VOLATILE': 0.966}
2026-04-27 15:01:41,297 INFO Regime epoch 41/50 — tr=0.6388 va=1.0203 acc=0.938
2026-04-27 15:01:41,439 INFO Regime epoch 42/50 — tr=0.6390 va=1.0224 acc=0.939
2026-04-27 15:01:41,580 INFO Regime epoch 43/50 — tr=0.6381 va=1.0169 acc=0.938
2026-04-27 15:01:41,718 INFO Regime epoch 44/50 — tr=0.6372 va=1.0172 acc=0.938
2026-04-27 15:01:41,862 INFO Regime epoch 45/50 — tr=0.6388 va=1.0144 acc=0.937 per_class={'TRENDING': 0.81, 'RANGING': 0.832, 'CONSOLIDATING': 0.898, 'VOLATILE': 0.968}
2026-04-27 15:01:42,006 INFO Regime epoch 46/50 — tr=0.6378 va=1.0139 acc=0.938
2026-04-27 15:01:42,156 INFO Regime epoch 47/50 — tr=0.6380 va=1.0143 acc=0.938
2026-04-27 15:01:42,292 INFO Regime epoch 48/50 — tr=0.6380 va=1.0140 acc=0.938
2026-04-27 15:01:42,429 INFO Regime epoch 49/50 — tr=0.6399 va=1.0174 acc=0.938
2026-04-27 15:01:42,581 INFO Regime epoch 50/50 — tr=0.6374 va=1.0155 acc=0.938 per_class={'TRENDING': 0.804, 'RANGING': 0.832, 'CONSOLIDATING': 0.896, 'VOLATILE': 0.97}
2026-04-27 15:01:42,595 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 15:01:42,595 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 15:01:42,596 INFO Regime phase LTF train: 7.3s
2026-04-27 15:01:42,745 INFO Regime LTF complete: acc=0.938, n=401471 per_class={'TRENDING': 0.813, 'RANGING': 0.827, 'CONSOLIDATING': 0.901, 'VOLATILE': 0.967}
2026-04-27 15:01:42,748 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:01:43,272 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 1778, 'RANGING': 57285, 'CONSOLIDATING': 4542, 'VOLATILE': 11019}  ambiguous=60314 (total=74624)  short_runs_zeroed=4774
2026-04-27 15:01:43,276 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 3.4794520547945207, 'RANGING': 17.033898305084747, 'CONSOLIDATING': 3.9358752166377817, 'VOLATILE': 5.842523860021209}
2026-04-27 15:01:43,283 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 1778, 'mean': 2.6590471099160558e-05, 'mean_over_std': 0.012770375199031195}, 'RANGING': {'n': 57284, 'mean': 8.05210952910413e-06, 'mean_over_std': 0.004238152373085909}, 'CONSOLIDATING': {'n': 4542, 'mean': 2.6402283987281153e-07, 'mean_over_std': 0.00015354245991368998}, 'VOLATILE': {'n': 11019, 'mean': 2.823213197675263e-05, 'mean_over_std': 0.010468794666397715}}
2026-04-27 15:01:43,284 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 915, 'mean': 7.666310957571063e-05, 'mean_over_std': 0.04299550456929985}, 'RANGING': {'n': 382, 'mean': 7.196437117461198e-05, 'mean_over_std': 0.05537815835040595}, 'CONSOLIDATING': {'n': 3375, 'mean': -3.0062004816363933e-06, 'mean_over_std': -0.0019542830050176476}, 'VOLATILE': {'n': 9638, 'mean': 1.2389816251671513e-05, 'mean_over_std': 0.004377959254493111}}
2026-04-27 15:01:43,287 INFO Regime retrain total: 202.8s (504761 samples)
2026-04-27 15:01:43,304 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-27 15:01:43,304 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 15:01:43,304 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 15:01:43,304 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-27 15:01:43,304 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-27 15:01:43,305 INFO Retrain complete. Total wall-clock: 202.9s
2026-04-27 15:01:47,718 INFO Model regime: SUCCESS
2026-04-27 15:01:47,718 INFO --- Training gru ---
2026-04-27 15:01:47,719 INFO Running retrain --model gru
2026-04-27 15:01:48,098 INFO retrain environment: KAGGLE
2026-04-27 15:01:49,796 INFO Device: CUDA (2 GPU(s))
2026-04-27 15:01:49,807 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 15:01:49,808 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 15:01:49,808 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 15:01:49,808 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 15:01:49,808 INFO Retrain data split: train
2026-04-27 15:01:49,810 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-27 15:01:49,972 INFO NumExpr defaulting to 4 threads.
2026-04-27 15:01:50,193 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-27 15:01:50,193 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 15:01:50,193 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 15:01:50,194 INFO GRU phase macro_correlations: 0.0s
2026-04-27 15:01:50,194 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-27 15:01:50,194 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260427_150150
2026-04-27 15:01:50,197 WARNING WeightsManifest: no manifest at /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json — assuming compatible (run a full retrain to generate one)
2026-04-27 15:01:50,346 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:50,366 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:50,381 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:50,388 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:50,389 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-27 15:01:50,390 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 15:01:50,390 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 15:01:50,390 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-27 15:01:50,391 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:50,485 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 17, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 8360}  ambiguous=8312 (total=8402)  short_runs_zeroed=15
2026-04-27 15:01:50,487 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:50,753 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 619, 'RANGING': 24879, 'CONSOLIDATING': 1882, 'VOLATILE': 5358}  ambiguous=26508 (total=32738)  short_runs_zeroed=2571
2026-04-27 15:01:50,790 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:51,126 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:51,278 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:51,394 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:51,623 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:51,641 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:51,656 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:51,663 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:51,664 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:51,753 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 100, 'BIAS_DOWN': 31, 'BIAS_NEUTRAL': 8271}  ambiguous=8272 (total=8402)  short_runs_zeroed=52
2026-04-27 15:01:51,755 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:52,019 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 752, 'RANGING': 24932, 'CONSOLIDATING': 2099, 'VOLATILE': 4955}  ambiguous=26224 (total=32738)  short_runs_zeroed=2112
2026-04-27 15:01:52,038 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:52,332 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:52,470 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:52,588 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:52,803 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:52,823 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:52,838 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:52,846 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:52,847 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:52,937 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 46, 'BIAS_DOWN': 4, 'BIAS_NEUTRAL': 8352}  ambiguous=8316 (total=8402)  short_runs_zeroed=28
2026-04-27 15:01:52,939 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:53,202 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 617, 'RANGING': 24738, 'CONSOLIDATING': 2103, 'VOLATILE': 5282}  ambiguous=26101 (total=32740)  short_runs_zeroed=2218
2026-04-27 15:01:53,220 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:53,522 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:53,667 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:53,781 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:54,001 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:54,020 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:54,035 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:54,042 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:54,043 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:54,135 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 15, 'BIAS_DOWN': 19, 'BIAS_NEUTRAL': 8368}  ambiguous=8331 (total=8402)  short_runs_zeroed=27
2026-04-27 15:01:54,137 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:54,411 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 842, 'RANGING': 25174, 'CONSOLIDATING': 1996, 'VOLATILE': 4727}  ambiguous=26490 (total=32739)  short_runs_zeroed=2045
2026-04-27 15:01:54,435 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:54,736 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:54,886 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:54,991 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:55,199 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:55,221 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:55,237 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:55,244 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:55,245 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:55,340 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 93, 'BIAS_DOWN': 58, 'BIAS_NEUTRAL': 8252}  ambiguous=8287 (total=8403)  short_runs_zeroed=75
2026-04-27 15:01:55,341 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:55,608 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 761, 'RANGING': 24579, 'CONSOLIDATING': 2089, 'VOLATILE': 5311}  ambiguous=25954 (total=32740)  short_runs_zeroed=2172
2026-04-27 15:01:55,626 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:55,925 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:56,073 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:56,184 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:56,386 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:56,405 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:56,421 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:56,428 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:56,429 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:56,519 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 87, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 8261}  ambiguous=8278 (total=8403)  short_runs_zeroed=63
2026-04-27 15:01:56,521 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:56,785 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 763, 'RANGING': 25215, 'CONSOLIDATING': 1968, 'VOLATILE': 4793}  ambiguous=26501 (total=32739)  short_runs_zeroed=1959
2026-04-27 15:01:56,801 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:57,108 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:57,258 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:57,374 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:57,567 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:01:57,585 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:01:57,599 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:01:57,606 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:01:57,607 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:57,700 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 21, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 8314}  ambiguous=8303 (total=8402)  short_runs_zeroed=39
2026-04-27 15:01:57,701 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:57,971 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 474, 'RANGING': 25095, 'CONSOLIDATING': 1967, 'VOLATILE': 5203}  ambiguous=26572 (total=32739)  short_runs_zeroed=2307
2026-04-27 15:01:57,984 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:58,277 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:58,428 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:58,548 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:58,765 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:58,784 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:58,800 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:58,808 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:01:58,809 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:58,903 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 37, 'BIAS_DOWN': 38, 'BIAS_NEUTRAL': 8327}  ambiguous=8321 (total=8402)  short_runs_zeroed=44
2026-04-27 15:01:58,905 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:59,172 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 881, 'RANGING': 25133, 'CONSOLIDATING': 1987, 'VOLATILE': 4739}  ambiguous=26356 (total=32740)  short_runs_zeroed=1985
2026-04-27 15:01:59,190 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:59,482 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:59,638 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:59,764 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:01:59,985 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:02:00,003 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:02:00,018 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:02:00,026 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:02:00,027 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:02:00,118 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 79, 'BIAS_DOWN': 33, 'BIAS_NEUTRAL': 8290}  ambiguous=8328 (total=8402)  short_runs_zeroed=61
2026-04-27 15:02:00,120 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:02:00,388 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 699, 'RANGING': 24989, 'CONSOLIDATING': 2130, 'VOLATILE': 4923}  ambiguous=26228 (total=32741)  short_runs_zeroed=2004
2026-04-27 15:02:00,404 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:02:00,706 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:02:00,859 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:02:00,985 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:02:01,224 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:02:01,245 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:02:01,261 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:02:01,268 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:02:01,269 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:02:01,360 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 42, 'BIAS_DOWN': 44, 'BIAS_NEUTRAL': 8317}  ambiguous=8318 (total=8403)  short_runs_zeroed=44
2026-04-27 15:02:01,362 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:02:01,629 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 808, 'RANGING': 24290, 'CONSOLIDATING': 2180, 'VOLATILE': 5465}  ambiguous=25645 (total=32743)  short_runs_zeroed=2211
2026-04-27 15:02:01,648 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:02:01,951 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:02:02,107 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:02:02,226 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:02:02,543 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:02:02,570 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:02:02,588 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:02:02,598 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:02:02,599 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:02:02,774 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 107, 'BIAS_DOWN': 115, 'BIAS_NEUTRAL': 19595}  ambiguous=19636 (total=19817)  short_runs_zeroed=112
2026-04-27 15:02:02,777 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:02:03,331 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 1778, 'RANGING': 57285, 'CONSOLIDATING': 4542, 'VOLATILE': 11019}  ambiguous=60314 (total=74624)  short_runs_zeroed=4774
2026-04-27 15:02:03,375 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:02:04,019 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:02:04,250 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:02:04,399 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:02:04,531 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-27 15:02:04,837 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-04-27 15:02:04,838 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-04-27 15:02:04,838 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-27 15:02:04,838 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-27 15:02:55,667 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-27 15:02:55,667 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-27 15:02:57,040 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-27 15:03:01,226 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=18, min_epochs=22)
2026-04-27 15:03:16,142 INFO train_multi TF=ALL epoch 1/50 train=0.8898 val=0.8809
2026-04-27 15:03:16,153 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 15:03:16,153 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 15:03:16,153 INFO train_multi TF=ALL: new best val=0.8809 — saved
2026-04-27 15:03:28,933 INFO train_multi TF=ALL epoch 2/50 train=0.8629 val=0.8201
2026-04-27 15:03:28,937 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 15:03:28,937 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 15:03:28,938 INFO train_multi TF=ALL: new best val=0.8201 — saved
2026-04-27 15:03:41,605 INFO train_multi TF=ALL epoch 3/50 train=0.7275 val=0.6878
2026-04-27 15:03:41,610 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 15:03:41,610 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 15:03:41,610 INFO train_multi TF=ALL: new best val=0.6878 — saved
2026-04-27 15:03:54,200 INFO train_multi TF=ALL epoch 4/50 train=0.6908 val=0.6879
2026-04-27 15:04:06,943 INFO train_multi TF=ALL epoch 5/50 train=0.6897 val=0.6892
2026-04-27 15:04:19,683 INFO train_multi TF=ALL epoch 6/50 train=0.6894 val=0.6898
2026-04-27 15:04:32,277 INFO train_multi TF=ALL epoch 7/50 train=0.6894 val=0.6896
2026-04-27 15:04:44,920 INFO train_multi TF=ALL epoch 8/50 train=0.6892 val=0.6897
2026-04-27 15:04:57,771 INFO train_multi TF=ALL epoch 9/50 train=0.6889 val=0.6891
2026-04-27 15:05:10,519 INFO train_multi TF=ALL epoch 10/50 train=0.6886 val=0.6877
2026-04-27 15:05:10,523 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 15:05:10,523 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 15:05:10,523 INFO train_multi TF=ALL: new best val=0.6877 — saved
2026-04-27 15:05:23,226 INFO train_multi TF=ALL epoch 11/50 train=0.6874 val=0.6871
2026-04-27 15:05:23,231 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 15:05:23,231 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 15:05:23,231 INFO train_multi TF=ALL: new best val=0.6871 — saved
2026-04-27 15:05:36,180 INFO train_multi TF=ALL epoch 12/50 train=0.6845 val=0.6820
2026-04-27 15:05:36,185 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 15:05:36,185 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 15:05:36,185 INFO train_multi TF=ALL: new best val=0.6820 — saved
2026-04-27 15:05:48,850 INFO train_multi TF=ALL epoch 13/50 train=0.6781 val=0.6706
2026-04-27 15:05:48,854 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 15:05:48,855 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 15:05:48,855 INFO train_multi TF=ALL: new best val=0.6706 — saved
2026-04-27 15:06:01,599 INFO train_multi TF=ALL epoch 14/50 train=0.6648 val=0.6553
2026-04-27 15:06:01,603 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 15:06:01,603 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 15:06:01,603 INFO train_multi TF=ALL: new best val=0.6553 — saved
2026-04-27 15:06:14,501 INFO train_multi TF=ALL epoch 15/50 train=0.6504 val=0.6390
2026-04-27 15:06:14,505 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 15:06:14,505 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 15:06:14,505 INFO train_multi TF=ALL: new best val=0.6390 — saved
2026-04-27 15:06:27,202 INFO train_multi TF=ALL epoch 16/50 train=0.6404 val=0.6348
2026-04-27 15:06:27,207 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 15:06:27,207 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 15:06:27,207 INFO train_multi TF=ALL: new best val=0.6348 — saved
2026-04-27 15:06:39,901 INFO train_multi TF=ALL epoch 17/50 train=0.6344 val=0.6279
2026-04-27 15:06:39,906 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 15:06:39,906 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 15:06:39,906 INFO train_multi TF=ALL: new best val=0.6279 — saved
2026-04-27 15:06:52,592 INFO train_multi TF=ALL epoch 18/50 train=0.6297 val=0.6282
2026-04-27 15:07:05,258 INFO train_multi TF=ALL epoch 19/50 train=0.6260 val=0.6276
2026-04-27 15:07:05,263 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 15:07:05,263 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 15:07:05,263 INFO train_multi TF=ALL: new best val=0.6276 — saved
2026-04-27 15:07:18,101 INFO train_multi TF=ALL epoch 20/50 train=0.6231 val=0.6215
2026-04-27 15:07:18,106 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 15:07:18,106 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 15:07:18,106 INFO train_multi TF=ALL: new best val=0.6215 — saved
2026-04-27 15:07:30,710 INFO train_multi TF=ALL epoch 21/50 train=0.6202 val=0.6203
2026-04-27 15:07:30,714 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 15:07:30,714 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 15:07:30,714 INFO train_multi TF=ALL: new best val=0.6203 — saved
2026-04-27 15:07:43,182 INFO train_multi TF=ALL epoch 22/50 train=0.6176 val=0.6211
2026-04-27 15:07:55,925 INFO train_multi TF=ALL epoch 23/50 train=0.6155 val=0.6171
2026-04-27 15:07:55,929 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 15:07:55,929 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 15:07:55,929 INFO train_multi TF=ALL: new best val=0.6171 — saved
2026-04-27 15:08:08,514 INFO train_multi TF=ALL epoch 24/50 train=0.6137 val=0.6200
2026-04-27 15:08:21,280 INFO train_multi TF=ALL epoch 25/50 train=0.6114 val=0.6185
2026-04-27 15:08:33,900 INFO train_multi TF=ALL epoch 26/50 train=0.6099 val=0.6174
2026-04-27 15:08:46,318 INFO train_multi TF=ALL epoch 27/50 train=0.6077 val=0.6191
2026-04-27 15:08:59,065 INFO train_multi TF=ALL epoch 28/50 train=0.6062 val=0.6190
2026-04-27 15:09:11,780 INFO train_multi TF=ALL epoch 29/50 train=0.6047 val=0.6170
2026-04-27 15:09:11,785 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 15:09:11,785 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 15:09:11,785 INFO train_multi TF=ALL: new best val=0.6170 — saved
2026-04-27 15:09:24,227 INFO train_multi TF=ALL epoch 30/50 train=0.6029 val=0.6180
2026-04-27 15:09:37,036 INFO train_multi TF=ALL epoch 31/50 train=0.6016 val=0.6149
2026-04-27 15:09:37,040 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 15:09:37,040 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 15:09:37,040 INFO train_multi TF=ALL: new best val=0.6149 — saved
2026-04-27 15:09:49,858 INFO train_multi TF=ALL epoch 32/50 train=0.6003 val=0.6232
2026-04-27 15:10:02,438 INFO train_multi TF=ALL epoch 33/50 train=0.5993 val=0.6221
2026-04-27 15:10:15,221 INFO train_multi TF=ALL epoch 34/50 train=0.5980 val=0.6150
2026-04-27 15:10:28,199 INFO train_multi TF=ALL epoch 35/50 train=0.5967 val=0.6156
2026-04-27 15:10:40,775 INFO train_multi TF=ALL epoch 36/50 train=0.5952 val=0.6181
2026-04-27 15:10:53,399 INFO train_multi TF=ALL epoch 37/50 train=0.5939 val=0.6179
2026-04-27 15:11:06,186 INFO train_multi TF=ALL epoch 38/50 train=0.5928 val=0.6152
2026-04-27 15:11:18,837 INFO train_multi TF=ALL epoch 39/50 train=0.5914 val=0.6131
2026-04-27 15:11:18,841 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 15:11:18,842 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 15:11:18,842 INFO train_multi TF=ALL: new best val=0.6131 — saved
2026-04-27 15:11:31,288 INFO train_multi TF=ALL epoch 40/50 train=0.5907 val=0.6154
2026-04-27 15:11:44,088 INFO train_multi TF=ALL epoch 41/50 train=0.5892 val=0.6173
2026-04-27 15:11:56,994 INFO train_multi TF=ALL epoch 42/50 train=0.5885 val=0.6169
2026-04-27 15:12:09,310 INFO train_multi TF=ALL epoch 43/50 train=0.5877 val=0.6216
2026-04-27 15:12:22,166 INFO train_multi TF=ALL epoch 44/50 train=0.5859 val=0.6168
2026-04-27 15:12:34,929 INFO train_multi TF=ALL epoch 45/50 train=0.5850 val=0.6162
2026-04-27 15:12:47,287 INFO train_multi TF=ALL epoch 46/50 train=0.5837 val=0.6158
2026-04-27 15:12:59,939 INFO train_multi TF=ALL epoch 47/50 train=0.5832 val=0.6198
2026-04-27 15:13:12,733 INFO train_multi TF=ALL epoch 48/50 train=0.5821 val=0.6169
2026-04-27 15:13:25,203 INFO train_multi TF=ALL epoch 49/50 train=0.5816 val=0.6159
2026-04-27 15:13:37,915 INFO train_multi TF=ALL epoch 50/50 train=0.5795 val=0.6186
2026-04-27 15:13:38,090 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-27 15:13:38,091 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-27 15:13:38,091 INFO Retrain complete. Total wall-clock: 708.3s
2026-04-27 15:13:40,307 INFO Model gru: SUCCESS
2026-04-27 15:13:40,308 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 15:13:40,308 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 15:13:40,308 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 15:13:40,308 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-04-27 15:13:40,308 INFO   [DEFERRED] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip (expected after Round 1)
2026-04-27 15:13:40,308 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-04-27 15:13:40,308 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer', 'rl_ppo']
2026-04-27 15:13:40,309 INFO Saved 5 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  START Round 1 - Backtest (val)
2026-04-27 15:13:41,270 INFO === STEP 6: BACKTEST (round1) ===
2026-04-27 15:13:41,271 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-04-27 15:13:41,271 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-27 15:13:41,271 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
2026-04-27 15:13:43,819 WARNING QualityScorer unavailable (weights missing or load failed)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260427_151343.json
Trader                                   Trades      WR    PF*  Return*    ExpR   TP1%   TP2%     DD*  Sharpe*
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)              302  19.2%   0.60  -96.8%  -0.321 19.2%  6.3% 115.7%    -3.55
  gate_diagnostics: bars=468696 no_signal=211522 quality_block=0 session_skip=256819 density=53 pm_reject=0 daily_skip=0 cooldown=0 daily_halt_events=0 enforce_daily_halt=False
  no_signal_reasons: blocked_consolidating=146544, weak_gru_direction=28621, neutral_requires_ltf_ranging=20123, htf_bias_conflict=6110, neutral_bias_weak_conf=3698, volatile_weak_conf=2883

Calibration Summary:
  all          [OK] Too few populated bins for calibration check
  ml_trader    [OK] Too few populated bins for calibration check
2026-04-27 15:16:46,177 INFO Round 1 backtest — 302 trades | avg WR=19.2% | avg PF=0.60 | avg Sharpe=-3.55
2026-04-27 15:16:46,177 INFO   ml_trader: 302 trades | WR=19.2% | fixed PF=0.60 | Return=-96.8% | ExpR=-0.321 | DD=115.7% | Sharpe=-3.55
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 302
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (302 rows)
2026-04-27 15:16:46,550 INFO Round 1: wrote 302 journal entries (total in file: 302)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  AUDUSD         45 trades    26 days   1.73/day  [OVERTRADE]
  EURGBP         23 trades    13 days   1.77/day  [OVERTRADE]
  EURJPY         16 trades    13 days   1.23/day
  EURUSD         34 trades    17 days   2.00/day  [OVERTRADE]
  GBPJPY         17 trades     8 days   2.12/day  [OVERTRADE]
  GBPUSD         37 trades    15 days   2.47/day  [OVERTRADE]
  NZDUSD         28 trades    15 days   1.87/day  [OVERTRADE]
  USDCAD         10 trades     5 days   2.00/day  [OVERTRADE]
  USDCHF         25 trades    10 days   2.50/day  [OVERTRADE]
  USDJPY         32 trades    17 days   1.88/day  [OVERTRADE]
  XAUUSD         35 trades    15 days   2.33/day  [OVERTRADE]
  ⚠  AUDUSD: 1.73/day (>1.5)
  ⚠  EURGBP: 1.77/day (>1.5)
  ⚠  EURUSD: 2.00/day (>1.5)
  ⚠  GBPJPY: 2.12/day (>1.5)
  ⚠  GBPUSD: 2.47/day (>1.5)
  ⚠  NZDUSD: 1.87/day (>1.5)
  ⚠  USDCAD: 2.00/day (>1.5)
  ⚠  USDCHF: 2.50/day (>1.5)
  ⚠  USDJPY: 1.88/day (>1.5)
  ⚠  XAUUSD: 2.33/day (>1.5)

──────────────────────────────────────────────────────────────
CHECK 2: REGIME DISTRIBUTION  (% of trades, WR, avgEV)
──────────────────────────────────────────────────────────────
  BIAS_DOWN           213 trades   70.5%  WR=21.1%  avgEV=0.000
  BIAS_NEUTRAL         10 trades    3.3%  WR=10.0%  avgEV=0.000
  BIAS_UP              79 trades   26.2%  WR=15.2%  avgEV=0.000
  ⚠  BIAS_DOWN = 71% of trades — regime collapse?
  ⚠  Regimes never traded: ['CONSOLIDATING', 'RANGING', 'TRENDING', 'VOLATILE']

──────────────────────────────────────────────────────────────
CHECK 3: EV PREDICTED vs REALIZED RR
──────────────────────────────────────────────────────────────
  Pearson  = +nan   Spearman = -0.0032

  Bucket                  N     AvgEV     AvgRR   WinRate
  Q1 (low EV)             0       n/a       n/a       n/a
  Q2                      0       n/a       n/a       n/a
  Q3                      0       n/a       n/a       n/a
  Q4 (high EV)          302     0.000    -0.321     19.2%

  Top-20% EV trades: n=302  avgEV=0.0  avgRR=-0.321  WR=19.2%

  Per-regime EV↔RR correlation:
  Regime                N    Pearson   Spearman       WR     AvgEV
  BIAS_DOWN           213       +nan    +0.1011   21.1%     0.000
  BIAS_NEUTRAL         10       +nan    +0.9030   10.0%     0.000
  BIAS_UP              79       +nan    -0.1899   15.2%     0.000
  ⚠  EV↔RR Spearman=-0.003 < 0.15 — EV rankings don't predict outcomes
  ⚠  Top-20% EV trades win_rate=19.2% — high-EV selection not working
  ⚠  EV↔RR Spearman in BIAS_UP = -0.190 — EV useless in this regime

──────────────────────────────────────────────────────────────
CHECK 4: GRU CONFIDENCE CALIBRATION  (p_bull/bear vs win rate)
──────────────────────────────────────────────────────────────
  ECE = 0.5816  (target < 0.10)
  Bin                   N   Midpoint    WinRate    Error
  [0.58-0.65]           6      0.613      0.500    0.113
  [0.65-0.71]          33      0.678      0.061    0.617
  [0.71-0.78]         121      0.744      0.223    0.521
  [0.78-0.84]         101      0.809      0.208    0.601
  [0.84-0.91]          41      0.875      0.122    0.753
  ⚠  Bin [0.65-0.71]: midpoint=0.68 win_rate=0.06 (err=0.62 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.71-0.78]: midpoint=0.74 win_rate=0.22 (err=0.52 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.78-0.84]: midpoint=0.81 win_rate=0.21 (err=0.60 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.84-0.91]: midpoint=0.88 win_rate=0.12 (err=0.75 > 0.15) — GRU miscalibrated
  ⚠  Win rate non-monotonic across confidence bins — GRU confidence unreliable

──────────────────────────────────────────────────────────────
CHECK 5: GRU ↔ EV MODEL CONSISTENCY  (direction × value agreement)
──────────────────────────────────────────────────────────────
  Pearson=+nan  Spearman=+0.0228  Agree=50%

  Quadrants  (conf_threshold=median, ev_threshold=median):
  high_conf + high_ev:   151  ← ideal
  high_conf + low_ev:      0  ← GRU overconfident
  low_conf  + high_ev:   151  ← EV optimistic
  low_conf  + low_ev:      0  ← correct abstention
  ⚠  GRU and EV agree on only 50.0% of trades — models pulling in opposite directions

──────────────────────────────────────────────────────────────
SUMMARY — 21 flag(s):
  ⚠  AUDUSD: 1.73/day (>1.5)
  ⚠  EURGBP: 1.77/day (>1.5)
  ⚠  EURUSD: 2.00/day (>1.5)
  ⚠  GBPJPY: 2.12/day (>1.5)
  ⚠  GBPUSD: 2.47/day (>1.5)
  ⚠  NZDUSD: 1.87/day (>1.5)
  ⚠  USDCAD: 2.00/day (>1.5)
  ⚠  USDCHF: 2.50/day (>1.5)
  ⚠  USDJPY: 1.88/day (>1.5)
  ⚠  XAUUSD: 2.33/day (>1.5)
  ⚠  BIAS_DOWN = 71% of trades — regime collapse?
  ⚠  Regimes never traded: ['CONSOLIDATING', 'RANGING', 'TRENDING', 'VOLATILE']
  ⚠  EV↔RR Spearman=-0.003 < 0.15 — EV rankings don't predict outcomes
  ⚠  Top-20% EV trades win_rate=19.2% — high-EV selection not working
  ⚠  EV↔RR Spearman in BIAS_UP = -0.190 — EV useless in this regime
  ⚠  Bin [0.65-0.71]: midpoint=0.68 win_rate=0.06 (err=0.62 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.71-0.78]: midpoint=0.74 win_rate=0.22 (err=0.52 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.78-0.84]: midpoint=0.81 win_rate=0.21 (err=0.60 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.84-0.91]: midpoint=0.88 win_rate=0.12 (err=0.75 > 0.15) — GRU miscalibrated
  ⚠  Win rate non-monotonic across confidence bins — GRU confidence unreliable
  ⚠  GRU and EV agree on only 50.0% of trades — models pulling in opposite directions
──────────────────────────────────────────────────────────────

======================================================================
  BACKTEST COMPLETE  (round 1 / window=round1)
======================================================================
  Round     Trades       WR     PF*  Sharpe*
  ------------------------------------------
  Round 1        302     19.2%    0.604    -3.554

  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 302 entries

=== Round 1 → Retrain Quality + RL on Round 1 journal ===
  START Round 1 - Quality+RL retrain
2026-04-27 15:16:46,986 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-27 15:16:46,987 INFO Journal entries: 302
2026-04-27 15:16:46,987 INFO --- Training quality ---
2026-04-27 15:16:46,988 INFO Running retrain --model quality with JOURNAL_ALLOWED_SPLITS=train,validation,test,combined_eval,live,paper,production
2026-04-27 15:16:47,188 INFO retrain environment: KAGGLE
2026-04-27 15:16:48,929 INFO Device: CUDA (2 GPU(s))
2026-04-27 15:16:48,941 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 15:16:48,941 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 15:16:48,941 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 15:16:48,941 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 15:16:48,942 INFO Retrain data split: train
2026-04-27 15:16:48,943 INFO === QualityScorer retrain ===
2026-04-27 15:16:49,092 INFO NumExpr defaulting to 4 threads.
2026-04-27 15:16:49,309 INFO QualityScorer: CUDA available — using GPU
2026-04-27 15:16:49,331 INFO Quality phase label creation: 0.0s (302 trades)
2026-04-27 15:16:49,356 INFO QualityScorer: 302 samples, EV stats={'mean': -0.5219867825508118, 'std': 1.0232081413269043, 'n_pos': 58, 'n_neg': 244}, device=cuda
2026-04-27 15:16:49,357 INFO QualityScorer: normalised win labels by median_win=1.156 — EV range now [-1, +3]
2026-04-27 15:16:49,561 INFO QualityScorer: DataParallel across 2 GPUs
2026-04-27 15:16:49,561 INFO QualityScorer: cold start
2026-04-27 15:16:49,561 INFO QualityScorer: pos_weight=5.34 (n_pos=38 n_neg=203)
2026-04-27 15:16:51,993 INFO Quality epoch   1/100 — va_huber=1.8162
2026-04-27 15:16:52,039 INFO Quality epoch   2/100 — va_huber=1.8003
2026-04-27 15:16:52,062 INFO Quality epoch   3/100 — va_huber=1.7900
2026-04-27 15:16:52,083 INFO Quality epoch   4/100 — va_huber=1.7795
2026-04-27 15:16:52,104 INFO Quality epoch   5/100 — va_huber=1.7689
2026-04-27 15:16:52,234 INFO Quality epoch  11/100 — va_huber=1.7069
2026-04-27 15:16:52,456 INFO Quality epoch  21/100 — va_huber=1.6199
2026-04-27 15:16:52,672 INFO Quality epoch  31/100 — va_huber=1.5306
2026-04-27 15:16:52,890 INFO Quality epoch  41/100 — va_huber=1.4955
2026-04-27 15:16:53,096 INFO Quality epoch  51/100 — va_huber=1.4907
2026-04-27 15:16:53,206 INFO Quality early stop at epoch 56
2026-04-27 15:16:53,214 INFO QualityScorer EV model: MAE=1.036 dir_acc=0.541 n_val=61
2026-04-27 15:16:53,219 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-27 15:16:53,271 INFO Quality phase train: 3.9s | total: 4.3s
2026-04-27 15:16:53,277 INFO Retrain complete. Total wall-clock: 4.3s
2026-04-27 15:16:54,483 INFO Model quality: SUCCESS
2026-04-27 15:16:54,483 INFO --- Training rl ---
2026-04-27 15:16:54,483 INFO Running retrain --model rl with JOURNAL_ALLOWED_SPLITS=train,validation,test,combined_eval,live,paper,production
2026-04-27 15:16:54,681 INFO retrain environment: KAGGLE
2026-04-27 15:16:56,398 INFO Device: CUDA (2 GPU(s))
2026-04-27 15:16:56,409 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 15:16:56,409 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 15:16:56,409 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 15:16:56,410 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 15:16:56,410 INFO Retrain data split: train
2026-04-27 15:16:56,411 INFO === RLAgent (PPO) retrain ===
2026-04-27 15:16:56,418 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260427_151656
2026-04-27 15:16:56,428 INFO RL phase episode loading: 0.0s (302 episodes)
2026-04-27 15:17:02.043658: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777303022.482932   55488 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777303022.600702   55488 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777303023.633075   55488 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777303023.633151   55488 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777303023.633156   55488 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777303023.633159   55488 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-27 15:17:22,336 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-27 15:17:25,976 INFO RLAgent: cold start — building new PPO policy
2026-04-27 15:17:32,814 INFO RLAgent: retrain complete, 302 episodes
2026-04-27 15:17:32,815 INFO RL phase PPO train: 36.4s | total: 36.4s
2026-04-27 15:17:32,822 INFO Retrain complete. Total wall-clock: 36.4s
2026-04-27 15:17:34,895 INFO Model rl: SUCCESS
2026-04-27 15:17:34,896 INFO Step 7b complete — summary: /kaggle/working/Multi-Bot/trading-system/ml_training/metrics/training_7b_summary.json
  DONE  Round 1 - Quality+RL retrain

=== Round 2: BLIND backtest on test window (unseen 2yr) ===
  START Round 2 - Blind backtest (test)
2026-04-27 15:17:35,440 INFO === STEP 6: BACKTEST (round2) ===
2026-04-27 15:17:35,441 INFO BT_WINDOW=round2 — BLIND backtest: 2023-08-07 → 2025-08-05 (test set)
2026-04-27 15:17:35,441 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-27 15:17:35,441 INFO Round 2 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260427_151737.json
Trader                                   Trades      WR    PF*  Return*    ExpR   TP1%   TP2%     DD*  Sharpe*
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)              159  23.9%   0.80  -24.5%  -0.154 23.9% 12.6%  49.5%    -1.58
  gate_diagnostics: bars=482221 no_signal=226997 quality_block=175 session_skip=254850 density=40 pm_reject=0 daily_skip=0 cooldown=0 daily_halt_events=0 enforce_daily_halt=False
  no_signal_reasons: blocked_consolidating=151543, weak_gru_direction=31333, neutral_requires_ltf_ranging=25632, htf_bias_conflict=7276, neutral_bias_weak_conf=4565, volatile_weak_conf=3122

Calibration Summary:
  all          [OK] Calibration usable but not strictly monotonic: 1/4 pairs violated.
  ml_trader    [OK] Calibration usable but not strictly monotonic: 1/4 pairs violated.
2026-04-27 15:20:43,483 INFO Round 2 backtest — 159 trades | avg WR=23.9% | avg PF=0.80 | avg Sharpe=-1.58
2026-04-27 15:20:43,484 INFO   ml_trader: 159 trades | WR=23.9% | fixed PF=0.80 | Return=-24.5% | ExpR=-0.154 | DD=49.5% | Sharpe=-1.58
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 159
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (159 rows)
2026-04-27 15:20:43,808 INFO Round 2: wrote 159 journal entries (total in file: 461)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  AUDUSD         13 trades     9 days   1.44/day
  EURGBP          5 trades     4 days   1.25/day
  EURJPY         18 trades    10 days   1.80/day  [OVERTRADE]
  EURUSD          5 trades     4 days   1.25/day
  GBPJPY         11 trades     6 days   1.83/day  [OVERTRADE]
  GBPUSD         12 trades    10 days   1.20/day
  NZDUSD          7 trades     2 days   3.50/day  [OVERTRADE]
  USDCAD          4 trades     4 days   1.00/day
  USDCHF         17 trades    10 days   1.70/day  [OVERTRADE]
  USDJPY         30 trades    14 days   2.14/day  [OVERTRADE]
  XAUUSD         37 trad  DONE  Round 2 - Blind backtest (test)
  Saved Round 2 result → round2_summary.json
  Journal after Round 2: 461 entries

=== Round 2 → Retrain Quality + RL on Round 1+2 journal ===
  START Round 2 - Quality+RL retrain
2026-04-27 15:20:44,270 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-27 15:20:44,271 INFO Journal entries: 461
2026-04-27 15:20:44,272 INFO --- Training quality ---
2026-04-27 15:20:44,272 INFO Running retrain --model quality with JOURNAL_ALLOWED_SPLITS=train,validation,test,combined_eval,live,paper,production
2026-04-27 15:20:44,469 INFO retrain environment: KAGGLE
2026-04-27 15:20:46,201 INFO Device: CUDA (2 GPU(s))
2026-04-27 15:20:46,212 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 15:20:46,213 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 15:20:46,213 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 15:20:46,213 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 15:20:46,213 INFO Retrain data split: train
2026-04-27 15:20:46,214 INFO === QualityScorer retrain ===
2026-04-27 15:20:46,365 INFO NumExpr defaulting to 4 threads.
2026-04-27 15:20:46,583 INFO QualityScorer: CUDA available — using GPU
2026-04-27 15:20:46,799 INFO QualityScorer loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (device=cuda)
2026-04-27 15:20:46,830 INFO Quality phase label creation: 0.0s (461 trades)
2026-04-27 15:20:46,830 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/quality_scorer.pkl_20260427_152046
2026-04-27 15:20:46,865 INFO QualityScorer: 461 samples, EV stats={'mean': -0.4635184109210968, 'std': 1.0923813581466675, 'n_pos': 96, 'n_neg': 365}, device=cuda
2026-04-27 15:20:46,866 INFO QualityScorer: normalised win labels by median_win=1.164 — EV range now [-1, +3]
2026-04-27 15:20:46,866 INFO QualityScorer: warm start from existing weights
2026-04-27 15:20:46,866 INFO QualityScorer: pos_weight=3.78 (n_pos=77 n_neg=291)
2026-04-27 15:20:49,274 INFO Quality epoch   1/100 — va_huber=0.8442
2026-04-27 15:20:49,316 INFO Quality epoch   2/100 — va_huber=0.8359
2026-04-27 15:20:49,341 INFO Quality epoch   3/100 — va_huber=0.8287
2026-04-27 15:20:49,363 INFO Quality epoch   4/100 — va_huber=0.8211
2026-04-27 15:20:49,385 INFO Quality epoch   5/100 — va_huber=0.8173
2026-04-27 15:20:49,521 INFO Quality epoch  11/100 — va_huber=0.7972
2026-04-27 15:20:49,746 INFO Quality epoch  21/100 — va_huber=0.7848
2026-04-27 15:20:50,174 INFO Quality epoch  31/100 — va_huber=0.7831
2026-04-27 15:20:50,395 INFO Quality epoch  41/100 — va_huber=0.7797
2026-04-27 15:20:50,466 INFO Quality early stop at epoch 44
2026-04-27 15:20:50,475 INFO QualityScorer EV model: MAE=0.880 dir_acc=0.667 n_val=93
2026-04-27 15:20:50,480 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-27 15:20:50,533 INFO Quality phase train: 3.7s | total: 4.3s
2026-04-27 15:20:50,540 INFO Retrain complete. Total wall-clock: 4.3s
2026-04-27 15:20:51,819 INFO Model quality: SUCCESS
2026-04-27 15:20:51,819 INFO --- Training rl ---
2026-04-27 15:20:51,819 INFO Running retrain --model rl with JOURNAL_ALLOWED_SPLITS=train,validation,test,combined_eval,live,paper,production
2026-04-27 15:20:52,016 INFO retrain environment: KAGGLE
2026-04-27 15:20:53,728 INFO Device: CUDA (2 GPU(s))
2026-04-27 15:20:53,739 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 15:20:53,739 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 15:20:53,740 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 15:20:53,740 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 15:20:53,740 INFO Retrain data split: train
2026-04-27 15:20:53,741 INFO === RLAgent (PPO) retrain ===
2026-04-27 15:20:53,743 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260427_152053
2026-04-27 15:20:54.667083: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777303254.690783   55783 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777303254.698605   55783 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777303254.719070   55783 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777303254.719123   55783 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777303254.719127   55783 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777303254.719129   55783 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-27 15:20:59,389 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-27 15:21:02,278 INFO RLAgent: PPO model loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-27 15:21:02,293 INFO RL phase episode loading: 0.0s (461 episodes)
2026-04-27 15:21:02,307 INFO RLAgent: warm start — fine-tuning existing PPO policy (lr=6.00e-05)
2026-04-27 15:21:11,919 INFO RLAgent: retrain complete, 461 episodes
2026-04-27 15:21:11,919 INFO RL phase PPO train: 9.6s | total: 18.2s
2026-04-27 15:21:11,928 INFO Retrain complete. Total wall-clock: 18.2s
2026-04-27 15:21:13,851 INFO Model rl: SUCCESS
2026-04-27 15:21:13,852 INFO Step 7b complete — summary: /kaggle/working/Multi-Bot/trading-system/ml_training/metrics/training_7b_summary.json
  DONE  Round 2 - Quality+RL retrain

=== Round 3: Incremental retrain ===
  START Retrain gru [train-split retrain]
2026-04-27 15:21:14,067 INFO retrain environment: KAGGLE
2026-04-27 15:21:15,776 INFO Device: CUDA (2 GPU(s))
2026-04-27 15:21:15,787 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 15:21:15,788 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 15:21:15,788 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 15:21:15,788 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 15:21:15,788 INFO Retrain data split: train
2026-04-27 15:21:15,789 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-27 15:21:15,944 INFO NumExpr defaulting to 4 threads.
2026-04-27 15:21:16,180 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-27 15:21:16,180 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 15:21:16,181 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 15:21:16,428 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-27 15:21:16,428 INFO GRU phase macro_correlations: 0.0s
2026-04-27 15:21:16,428 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-27 15:21:16,430 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260427_152116
2026-04-27 15:21:16,433 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-04-27 15:21:16,586 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:16,607 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:16,621 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:16,629 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:16,631 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-27 15:21:16,631 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 15:21:16,631 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 15:21:16,632 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-27 15:21:16,633 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:16,723 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 17, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 8360}  ambiguous=8312 (total=8402)  short_runs_zeroed=15
2026-04-27 15:21:16,725 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:16,974 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 619, 'RANGING': 24879, 'CONSOLIDATING': 1882, 'VOLATILE': 5358}  ambiguous=26508 (total=32738)  short_runs_zeroed=2571
2026-04-27 15:21:17,002 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:17,316 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:17,473 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:17,590 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:17,817 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:17,836 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:17,851 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:17,858 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:17,859 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:17,941 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 100, 'BIAS_DOWN': 31, 'BIAS_NEUTRAL': 8271}  ambiguous=8272 (total=8402)  short_runs_zeroed=52
2026-04-27 15:21:17,944 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:18,187 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 752, 'RANGING': 24932, 'CONSOLIDATING': 2099, 'VOLATILE': 4955}  ambiguous=26224 (total=32738)  short_runs_zeroed=2112
2026-04-27 15:21:18,203 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:18,511 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:18,665 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:18,789 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:19,011 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:19,031 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:19,046 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:19,054 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:19,055 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:19,139 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 46, 'BIAS_DOWN': 4, 'BIAS_NEUTRAL': 8352}  ambiguous=8316 (total=8402)  short_runs_zeroed=28
2026-04-27 15:21:19,141 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:19,381 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 617, 'RANGING': 24738, 'CONSOLIDATING': 2103, 'VOLATILE': 5282}  ambiguous=26101 (total=32740)  short_runs_zeroed=2218
2026-04-27 15:21:19,397 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:19,710 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:19,855 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:19,976 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:20,195 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:20,215 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:20,230 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:20,238 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:20,239 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:20,323 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 15, 'BIAS_DOWN': 19, 'BIAS_NEUTRAL': 8368}  ambiguous=8331 (total=8402)  short_runs_zeroed=27
2026-04-27 15:21:20,325 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:20,571 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 842, 'RANGING': 25174, 'CONSOLIDATING': 1996, 'VOLATILE': 4727}  ambiguous=26490 (total=32739)  short_runs_zeroed=2045
2026-04-27 15:21:20,595 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:20,912 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:21,074 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:21,216 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:21,440 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:21,461 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:21,477 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:21,484 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:21,485 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:21,572 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 93, 'BIAS_DOWN': 58, 'BIAS_NEUTRAL': 8252}  ambiguous=8287 (total=8403)  short_runs_zeroed=75
2026-04-27 15:21:21,574 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:21,829 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 761, 'RANGING': 24579, 'CONSOLIDATING': 2089, 'VOLATILE': 5311}  ambiguous=25954 (total=32740)  short_runs_zeroed=2172
2026-04-27 15:21:21,846 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:22,164 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:22,321 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:22,446 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:22,663 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:22,683 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:22,698 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:22,705 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:22,706 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:22,792 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 87, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 8261}  ambiguous=8278 (total=8403)  short_runs_zeroed=63
2026-04-27 15:21:22,794 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:23,045 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 763, 'RANGING': 25215, 'CONSOLIDATING': 1968, 'VOLATILE': 4793}  ambiguous=26501 (total=32739)  short_runs_zeroed=1959
2026-04-27 15:21:23,061 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:23,375 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:23,532 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:23,661 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:23,858 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:21:23,876 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:21:23,891 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:21:23,898 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:21:23,899 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:23,987 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 21, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 8314}  ambiguous=8303 (total=8402)  short_runs_zeroed=39
2026-04-27 15:21:23,989 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:24,241 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 474, 'RANGING': 25095, 'CONSOLIDATING': 1967, 'VOLATILE': 5203}  ambiguous=26572 (total=32739)  short_runs_zeroed=2307
2026-04-27 15:21:24,255 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:24,557 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:24,716 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:24,836 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:25,052 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:25,074 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:25,089 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:25,096 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:25,097 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:25,184 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 37, 'BIAS_DOWN': 38, 'BIAS_NEUTRAL': 8327}  ambiguous=8321 (total=8402)  short_runs_zeroed=44
2026-04-27 15:21:25,186 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:25,435 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 881, 'RANGING': 25133, 'CONSOLIDATING': 1987, 'VOLATILE': 4739}  ambiguous=26356 (total=32740)  short_runs_zeroed=1985
2026-04-27 15:21:25,453 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:25,775 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:25,933 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:26,061 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:26,279 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:26,300 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:26,314 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:26,322 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:26,323 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:26,416 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 79, 'BIAS_DOWN': 33, 'BIAS_NEUTRAL': 8290}  ambiguous=8328 (total=8402)  short_runs_zeroed=61
2026-04-27 15:21:26,418 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:26,690 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 699, 'RANGING': 24989, 'CONSOLIDATING': 2130, 'VOLATILE': 4923}  ambiguous=26228 (total=32741)  short_runs_zeroed=2004
2026-04-27 15:21:26,709 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:27,029 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:27,183 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:27,305 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:27,530 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:27,550 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:27,566 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:27,575 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:21:27,576 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:27,662 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 42, 'BIAS_DOWN': 44, 'BIAS_NEUTRAL': 8317}  ambiguous=8318 (total=8403)  short_runs_zeroed=44
2026-04-27 15:21:27,664 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:27,910 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 808, 'RANGING': 24290, 'CONSOLIDATING': 2180, 'VOLATILE': 5465}  ambiguous=25645 (total=32743)  short_runs_zeroed=2211
2026-04-27 15:21:27,926 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:28,245 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:28,402 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:28,521 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:21:28,843 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:21:28,869 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:21:28,886 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:21:28,895 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:21:28,896 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:21:29,061 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 107, 'BIAS_DOWN': 115, 'BIAS_NEUTRAL': 19595}  ambiguous=19636 (total=19817)  short_runs_zeroed=112
2026-04-27 15:21:29,065 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:21:29,590 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 1778, 'RANGING': 57285, 'CONSOLIDATING': 4542, 'VOLATILE': 11019}  ambiguous=60314 (total=74624)  short_runs_zeroed=4774
2026-04-27 15:21:29,639 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:21:30,304 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:21:30,536 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:21:30,703 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:21:30,849 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-27 15:21:30,849 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-27 15:21:30,849 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-27 15:22:21,975 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-27 15:22:21,975 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-27 15:22:23,363 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-27 15:22:27,536 INFO train_multi TF=ALL: warm-start detected — using CosineAnnealingLR (lr=3e-05, patience=12)
2026-04-27 15:22:41,958 INFO train_multi TF=ALL epoch 1/50 train=0.5890 val=0.6146
2026-04-27 15:22:41,964 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 15:22:41,964 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 15:22:41,964 INFO train_multi TF=ALL: new best val=0.6146 — saved
2026-04-27 15:22:54,803 INFO train_multi TF=ALL epoch 2/50 train=0.5885 val=0.6152
2026-04-27 15:23:07,429 INFO train_multi TF=ALL epoch 3/50 train=0.5885 val=0.6154
2026-04-27 15:23:20,229 INFO train_multi TF=ALL epoch 4/50 train=0.5882 val=0.6158
2026-04-27 15:23:33,087 INFO train_multi TF=ALL epoch 5/50 train=0.5883 val=0.6153
2026-04-27 15:23:45,637 INFO train_multi TF=ALL epoch 6/50 train=0.5882 val=0.6156
2026-04-27 15:23:58,318 INFO train_multi TF=ALL epoch 7/50 train=0.5878 val=0.6156
2026-04-27 15:24:11,235 INFO train_multi TF=ALL epoch 8/50 train=0.5876 val=0.6155
2026-04-27 15:24:24,004 INFO train_multi TF=ALL epoch 9/50 train=0.5876 val=0.6156
2026-04-27 15:24:36,615 INFO train_multi TF=ALL epoch 10/50 train=0.5874 val=0.6161
2026-04-27 15:24:49,523 INFO train_multi TF=ALL epoch 11/50 train=0.5873 val=0.6152
2026-04-27 15:25:02,227 INFO train_multi TF=ALL epoch 12/50 train=0.5872 val=0.6159
2026-04-27 15:25:14,978 INFO train_multi TF=ALL epoch 13/50 train=0.5871 val=0.6163
2026-04-27 15:25:14,978 INFO train_multi TF=ALL early stop at epoch 13
2026-04-27 15:25:15,165 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-27 15:25:15,166 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-27 15:25:15,166 INFO Retrain complete. Total wall-clock: 239.4s
  DONE  Retrain gru [train-split retrain]
  START Retrain regime [train-split retrain]
2026-04-27 15:25:17,659 INFO retrain environment: KAGGLE
2026-04-27 15:25:19,340 INFO Device: CUDA (2 GPU(s))
2026-04-27 15:25:19,349 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 15:25:19,350 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 15:25:19,350 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 15:25:19,350 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 15:25:19,350 INFO Retrain data split: train
2026-04-27 15:25:19,351 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-27 15:25:19,512 INFO NumExpr defaulting to 4 threads.
2026-04-27 15:25:19,744 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-27 15:25:19,744 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 15:25:19,744 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 15:25:19,744 INFO Regime phase macro_correlations: 0.0s
2026-04-27 15:25:19,744 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-27 15:25:19,786 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-27 15:25:19,787 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:25:19,816 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:25:19,832 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:25:19,857 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:25:19,874 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:25:19,900 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:25:19,916 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:25:19,941 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:25:19,957 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:25:19,983 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:25:19,999 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:25:20,021 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:25:20,035 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:25:20,056 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:25:20,072 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:25:20,095 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:25:20,111 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:25:20,134 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:25:20,150 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:25:20,173 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:25:20,192 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:25:20,236 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:25:21,110 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 15:25:44,822 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 15:25:44,827 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 24.6s
2026-04-27 15:25:44,827 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 15:25:55,708 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 15:25:55,712 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 10.9s
2026-04-27 15:25:55,715 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 15:26:03,994 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 15:26:03,995 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 8.3s
2026-04-27 15:26:03,995 INFO Regime phase GMM HTF total: 43.8s
2026-04-27 15:26:03,995 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 15:27:15,534 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 15:27:15,538 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 71.5s
2026-04-27 15:27:15,538 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 15:27:46,673 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 15:27:46,676 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 31.1s
2026-04-27 15:27:46,676 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 15:28:09,117 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 15:28:09,118 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 22.4s
2026-04-27 15:28:09,118 INFO Regime phase GMM LTF total: 125.1s
2026-04-27 15:28:09,255 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-27 15:28:09,256 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:09,257 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:09,259 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:09,259 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:09,260 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:09,261 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:09,262 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:09,263 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:09,264 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:09,265 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:09,266 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:28:09,397 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:09,444 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:09,445 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:09,445 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:09,454 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:09,455 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:09,916 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 17, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 8360}  ambiguous=8312 (total=8402)  short_runs_zeroed=15
2026-04-27 15:28:09,917 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-27 15:28:10,131 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:10,166 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:10,167 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:10,167 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:10,176 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:10,177 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:10,593 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 100, 'BIAS_DOWN': 31, 'BIAS_NEUTRAL': 8271}  ambiguous=8272 (total=8402)  short_runs_zeroed=52
2026-04-27 15:28:10,595 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-27 15:28:10,822 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:10,858 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:10,859 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:10,860 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:10,868 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:10,869 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:11,289 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 46, 'BIAS_DOWN': 4, 'BIAS_NEUTRAL': 8352}  ambiguous=8316 (total=8402)  short_runs_zeroed=28
2026-04-27 15:28:11,291 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-27 15:28:11,505 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:11,545 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:11,546 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:11,546 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:11,554 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:11,555 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:11,967 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 15, 'BIAS_DOWN': 19, 'BIAS_NEUTRAL': 8368}  ambiguous=8331 (total=8402)  short_runs_zeroed=27
2026-04-27 15:28:11,969 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-27 15:28:12,184 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:12,222 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:12,223 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:12,223 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:12,232 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:12,233 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:12,648 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 93, 'BIAS_DOWN': 58, 'BIAS_NEUTRAL': 8252}  ambiguous=8287 (total=8403)  short_runs_zeroed=75
2026-04-27 15:28:12,649 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-27 15:28:12,852 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:12,889 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:12,890 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:12,891 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:12,899 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:12,900 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:13,312 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 87, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 8261}  ambiguous=8278 (total=8403)  short_runs_zeroed=63
2026-04-27 15:28:13,313 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-27 15:28:13,503 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:28:13,532 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:28:13,533 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:28:13,534 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:28:13,541 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:28:13,542 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:13,946 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 21, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 8314}  ambiguous=8303 (total=8402)  short_runs_zeroed=39
2026-04-27 15:28:13,947 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-27 15:28:14,144 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:14,179 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:14,180 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:14,181 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:14,189 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:14,190 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:14,601 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 37, 'BIAS_DOWN': 38, 'BIAS_NEUTRAL': 8327}  ambiguous=8321 (total=8402)  short_runs_zeroed=44
2026-04-27 15:28:14,603 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-27 15:28:14,807 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:14,843 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:14,844 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:14,844 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:14,853 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:14,854 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:15,267 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 79, 'BIAS_DOWN': 33, 'BIAS_NEUTRAL': 8290}  ambiguous=8328 (total=8402)  short_runs_zeroed=61
2026-04-27 15:28:15,269 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-27 15:28:15,481 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:15,519 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:15,520 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:15,520 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:15,529 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:15,530 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:15,933 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 42, 'BIAS_DOWN': 44, 'BIAS_NEUTRAL': 8317}  ambiguous=8318 (total=8403)  short_runs_zeroed=44
2026-04-27 15:28:15,934 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-27 15:28:16,235 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:28:16,299 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:28:16,300 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:28:16,301 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:28:16,312 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:28:16,313 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:28:17,172 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 107, 'BIAS_DOWN': 115, 'BIAS_NEUTRAL': 19595}  ambiguous=19636 (total=19817)  short_runs_zeroed=112
2026-04-27 15:28:17,175 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-27 15:28:17,404 INFO Regime phase HTF dataset build: 8.1s (103290 samples)
2026-04-27 15:28:17,405 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260427_152817
2026-04-27 15:28:17,614 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=34, n_classes=3)
2026-04-27 15:28:17,615 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=1138 dropped=102152 classes={'BIAS_UP': 399, 'BIAS_DOWN': 243, 'BIAS_NEUTRAL': 496})
2026-04-27 15:28:17,616 INFO RegimeClassifier[mode=htf_bias]: 1138 samples, classes={'BIAS_UP': 399, 'BIAS_DOWN': 243, 'BIAS_NEUTRAL': 496}, device=cuda
2026-04-27 15:28:17,616 INFO RegimeClassifier: sample weights — mean=0.713  ambiguous(<0.4)=0.0%
2026-04-27 15:28:17,616 INFO RegimeClassifier[mode=htf_bias]: warm start from existing weights
2026-04-27 15:28:17,616 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-27 15:28:19,855 INFO Regime epoch  1/50 — tr=0.5423 va=1.1827 acc=0.921 per_class={'BIAS_UP': 0.797, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 1.0}
2026-04-27 15:28:19,869 INFO Regime epoch  2/50 — tr=0.5376 va=1.1806 acc=0.921
2026-04-27 15:28:19,882 INFO Regime epoch  3/50 — tr=0.5359 va=1.1772 acc=0.921
2026-04-27 15:28:19,895 INFO Regime epoch  4/50 — tr=0.5312 va=1.1736 acc=0.925
2026-04-27 15:28:19,912 INFO Regime epoch  5/50 — tr=0.5340 va=1.1714 acc=0.930 per_class={'BIAS_UP': 0.824, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 1.0}
2026-04-27 15:28:19,925 INFO Regime epoch  6/50 — tr=0.5375 va=1.1670 acc=0.930
2026-04-27 15:28:19,938 INFO Regime epoch  7/50 — tr=0.5331 va=1.1628 acc=0.930
2026-04-27 15:28:19,951 INFO Regime epoch  8/50 — tr=0.5364 va=1.1576 acc=0.930
2026-04-27 15:28:19,963 INFO Regime epoch  9/50 — tr=0.5374 va=1.1504 acc=0.930
2026-04-27 15:28:19,979 INFO Regime epoch 10/50 — tr=0.5339 va=1.1452 acc=0.934 per_class={'BIAS_UP': 0.838, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 1.0}
2026-04-27 15:28:19,993 INFO Regime epoch 11/50 — tr=0.5328 va=1.1390 acc=0.947
2026-04-27 15:28:20,006 INFO Regime epoch 12/50 — tr=0.5318 va=1.1328 acc=0.952
2026-04-27 15:28:20,021 INFO Regime epoch 13/50 — tr=0.5290 va=1.1256 acc=0.952
2026-04-27 15:28:20,034 INFO Regime epoch 14/50 — tr=0.5352 va=1.1207 acc=0.952
2026-04-27 15:28:20,051 INFO Regime epoch 15/50 — tr=0.5338 va=1.1143 acc=0.956 per_class={'BIAS_UP': 0.905, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 1.0}
2026-04-27 15:28:20,066 INFO Regime epoch 16/50 — tr=0.5296 va=1.1084 acc=0.956
2026-04-27 15:28:20,080 INFO Regime epoch 17/50 — tr=0.5260 va=1.1020 acc=0.961
2026-04-27 15:28:20,093 INFO Regime epoch 18/50 — tr=0.5260 va=1.0962 acc=0.961
2026-04-27 15:28:20,106 INFO Regime epoch 19/50 — tr=0.5331 va=1.0896 acc=0.965
2026-04-27 15:28:20,121 INFO Regime epoch 20/50 — tr=0.5265 va=1.0848 acc=0.965 per_class={'BIAS_UP': 0.932, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 1.0}
2026-04-27 15:28:20,134 INFO Regime epoch 21/50 — tr=0.5234 va=1.0807 acc=0.965
2026-04-27 15:28:20,147 INFO Regime epoch 22/50 — tr=0.5251 va=1.0766 acc=0.965
2026-04-27 15:28:20,160 INFO Regime epoch 23/50 — tr=0.5262 va=1.0728 acc=0.965
2026-04-27 15:28:20,174 INFO Regime epoch 24/50 — tr=0.5179 va=1.0688 acc=0.965
2026-04-27 15:28:20,191 INFO Regime epoch 25/50 — tr=0.5226 va=1.0657 acc=0.978 per_class={'BIAS_UP': 0.973, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 1.0}
2026-04-27 15:28:20,204 INFO Regime epoch 26/50 — tr=0.5177 va=1.0617 acc=0.978
2026-04-27 15:28:20,219 INFO Regime epoch 27/50 — tr=0.5242 va=1.0585 acc=0.987
2026-04-27 15:28:20,232 INFO Regime epoch 28/50 — tr=0.5241 va=1.0546 acc=0.987
2026-04-27 15:28:20,245 INFO Regime epoch 29/50 — tr=0.5181 va=1.0501 acc=0.982
2026-04-27 15:28:20,262 INFO Regime epoch 30/50 — tr=0.5199 va=1.0462 acc=0.982 per_class={'BIAS_UP': 0.986, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 1.0}
2026-04-27 15:28:20,275 INFO Regime epoch 31/50 — tr=0.5215 va=1.0425 acc=0.982
2026-04-27 15:28:20,288 INFO Regime epoch 32/50 — tr=0.5153 va=1.0415 acc=0.982
2026-04-27 15:28:20,300 INFO Regime epoch 33/50 — tr=0.5221 va=1.0400 acc=0.982
2026-04-27 15:28:20,313 INFO Regime epoch 34/50 — tr=0.5179 va=1.0400 acc=0.982
2026-04-27 15:28:20,330 INFO Regime epoch 35/50 — tr=0.5207 va=1.0379 acc=0.982 per_class={'BIAS_UP': 0.986, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 1.0}
2026-04-27 15:28:20,342 INFO Regime epoch 36/50 — tr=0.5168 va=1.0345 acc=0.978
2026-04-27 15:28:20,355 INFO Regime epoch 37/50 — tr=0.5198 va=1.0339 acc=0.978
2026-04-27 15:28:20,368 INFO Regime epoch 38/50 — tr=0.5160 va=1.0323 acc=0.978
2026-04-27 15:28:20,381 INFO Regime epoch 39/50 — tr=0.5202 va=1.0306 acc=0.982
2026-04-27 15:28:20,398 INFO Regime epoch 40/50 — tr=0.5139 va=1.0309 acc=0.982 per_class={'BIAS_UP': 0.986, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 1.0}
2026-04-27 15:28:20,411 INFO Regime epoch 41/50 — tr=0.5139 va=1.0305 acc=0.982
2026-04-27 15:28:20,425 INFO Regime epoch 42/50 — tr=0.5214 va=1.0314 acc=0.982
2026-04-27 15:28:20,438 INFO Regime epoch 43/50 — tr=0.5138 va=1.0307 acc=0.987
2026-04-27 15:28:20,451 INFO Regime epoch 44/50 — tr=0.5169 va=1.0306 acc=0.987
2026-04-27 15:28:20,467 INFO Regime epoch 45/50 — tr=0.5186 va=1.0305 acc=0.987 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 1.0}
2026-04-27 15:28:20,480 INFO Regime epoch 46/50 — tr=0.5161 va=1.0308 acc=0.987
2026-04-27 15:28:20,493 INFO Regime epoch 47/50 — tr=0.5191 va=1.0311 acc=0.982
2026-04-27 15:28:20,506 INFO Regime epoch 48/50 — tr=0.5167 va=1.0302 acc=0.982
2026-04-27 15:28:20,519 INFO Regime epoch 49/50 — tr=0.5165 va=1.0301 acc=0.982
2026-04-27 15:28:20,535 INFO Regime epoch 50/50 — tr=0.5155 va=1.0295 acc=0.982 per_class={'BIAS_UP': 0.986, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 1.0}
2026-04-27 15:28:20,543 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 15:28:20,543 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 15:28:20,544 INFO Regime phase HTF train: 2.9s
2026-04-27 15:28:20,702 INFO Regime HTF complete: acc=0.982, n=103290 per_class={'BIAS_UP': 0.986, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 1.0}
2026-04-27 15:28:20,704 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:28:20,885 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 107, 'BIAS_DOWN': 115, 'BIAS_NEUTRAL': 19595}  ambiguous=19636 (total=19817)  short_runs_zeroed=112
2026-04-27 15:28:20,888 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 4.28, 'BIAS_DOWN': 4.791666666666667, 'BIAS_NEUTRAL': 391.9}
2026-04-27 15:28:20,892 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 107, 'mean': -0.00021706688424743013, 'mean_over_std': -0.04806767606653151}, 'BIAS_DOWN': {'n': 115, 'mean': -0.00020797041876048362, 'mean_over_std': -0.029260022054973262}, 'BIAS_NEUTRAL': {'n': 19594, 'mean': 4.4550533138328785e-05, 'mean_over_std': 0.011372683463534268}}
2026-04-27 15:28:20,892 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 66, 'mean': 0.0003288215669218909, 'mean_over_std': 0.07277335807221066}, 'BIAS_DOWN': {'n': 59, 'mean': -0.0010984382215802496, 'mean_over_std': -0.13394112338746375}, 'BIAS_NEUTRAL': {'n': 56, 'mean': -0.00020920056804862467, 'mean_over_std': -0.06887192756862072}}
2026-04-27 15:28:20,898 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-27 15:28:20,900 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:20,902 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:20,904 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:20,906 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:20,907 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:20,909 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:20,910 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:20,912 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:20,914 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:20,915 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:20,918 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:28:20,930 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:20,935 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:20,935 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:20,936 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:20,936 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:20,938 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:21,648 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 619, 'RANGING': 24879, 'CONSOLIDATING': 1882, 'VOLATILE': 5358}  ambiguous=26508 (total=32738)  short_runs_zeroed=2571
2026-04-27 15:28:21,651 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-27 15:28:21,813 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:21,816 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:21,817 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:21,817 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:21,817 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:21,819 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:22,451 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 752, 'RANGING': 24932, 'CONSOLIDATING': 2099, 'VOLATILE': 4955}  ambiguous=26224 (total=32738)  short_runs_zeroed=2112
2026-04-27 15:28:22,454 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-27 15:28:22,621 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:22,624 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:22,624 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:22,625 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:22,625 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:22,627 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:23,280 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 617, 'RANGING': 24738, 'CONSOLIDATING': 2103, 'VOLATILE': 5282}  ambiguous=26101 (total=32740)  short_runs_zeroed=2218
2026-04-27 15:28:23,283 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-27 15:28:23,454 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:23,457 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:23,458 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:23,458 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:23,458 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:23,460 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:24,117 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 842, 'RANGING': 25174, 'CONSOLIDATING': 1996, 'VOLATILE': 4727}  ambiguous=26490 (total=32739)  short_runs_zeroed=2045
2026-04-27 15:28:24,121 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-27 15:28:24,285 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:24,289 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:24,290 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:24,291 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:24,291 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:24,293 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:24,938 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 761, 'RANGING': 24579, 'CONSOLIDATING': 2089, 'VOLATILE': 5311}  ambiguous=25954 (total=32740)  short_runs_zeroed=2172
2026-04-27 15:28:24,941 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-27 15:28:25,108 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:25,111 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:25,112 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:25,112 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:25,112 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:25,114 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:25,763 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 763, 'RANGING': 25215, 'CONSOLIDATING': 1968, 'VOLATILE': 4793}  ambiguous=26501 (total=32739)  short_runs_zeroed=1959
2026-04-27 15:28:25,767 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-27 15:28:25,925 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:28:25,927 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:28:25,928 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:28:25,929 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:28:25,929 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 15:28:25,931 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:26,577 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 474, 'RANGING': 25095, 'CONSOLIDATING': 1967, 'VOLATILE': 5203}  ambiguous=26572 (total=32739)  short_runs_zeroed=2307
2026-04-27 15:28:26,581 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-27 15:28:26,746 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:26,748 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:26,749 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:26,749 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:26,750 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:26,752 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:27,400 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 881, 'RANGING': 25133, 'CONSOLIDATING': 1987, 'VOLATILE': 4739}  ambiguous=26356 (total=32740)  short_runs_zeroed=1985
2026-04-27 15:28:27,403 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-27 15:28:27,564 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:27,567 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:27,568 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:27,568 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:27,568 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:27,570 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:28,217 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 699, 'RANGING': 24989, 'CONSOLIDATING': 2130, 'VOLATILE': 4923}  ambiguous=26228 (total=32741)  short_runs_zeroed=2004
2026-04-27 15:28:28,220 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-27 15:28:28,386 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:28,388 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:28,389 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:28,390 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:28,390 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 15:28:28,392 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 15:28:29,034 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 808, 'RANGING': 24290, 'CONSOLIDATING': 2180, 'VOLATILE': 5465}  ambiguous=25645 (total=32743)  short_runs_zeroed=2211
2026-04-27 15:28:29,037 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-27 15:28:29,214 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:28:29,218 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:28:29,219 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:28:29,220 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:28:29,220 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 15:28:29,223 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:28:30,605 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 1778, 'RANGING': 57285, 'CONSOLIDATING': 4542, 'VOLATILE': 11019}  ambiguous=60314 (total=74624)  short_runs_zeroed=4774
2026-04-27 15:28:30,611 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-27 15:28:30,977 INFO Regime phase LTF dataset build: 10.1s (401471 samples)
2026-04-27 15:28:30,978 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260427_152830
2026-04-27 15:28:30,982 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=18, n_classes=4)
2026-04-27 15:28:30,985 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=79106 dropped=322365 classes={'TRENDING': 4775, 'RANGING': 1876, 'CONSOLIDATING': 18867, 'VOLATILE': 53588})
2026-04-27 15:28:31,002 INFO RegimeClassifier[mode=ltf_behaviour]: 79106 samples, classes={'TRENDING': 4775, 'RANGING': 1876, 'CONSOLIDATING': 18867, 'VOLATILE': 53588}, device=cuda
2026-04-27 15:28:31,003 INFO RegimeClassifier: sample weights — mean=0.811  ambiguous(<0.4)=0.0%
2026-04-27 15:28:31,003 INFO RegimeClassifier[mode=ltf_behaviour]: warm start from existing weights
2026-04-27 15:28:31,003 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-27 15:28:31,230 INFO Regime epoch  1/50 — tr=0.6394 va=1.0141 acc=0.937 per_class={'TRENDING': 0.812, 'RANGING': 0.827, 'CONSOLIDATING': 0.901, 'VOLATILE': 0.967}
2026-04-27 15:28:31,392 INFO Regime epoch  2/50 — tr=0.6387 va=1.0134 acc=0.938
2026-04-27 15:28:31,567 INFO Regime epoch  3/50 — tr=0.6379 va=1.0168 acc=0.938
2026-04-27 15:28:31,723 INFO Regime epoch  4/50 — tr=0.6390 va=1.0151 acc=0.938
2026-04-27 15:28:31,881 INFO Regime epoch  5/50 — tr=0.6387 va=1.0100 acc=0.938 per_class={'TRENDING': 0.81, 'RANGING': 0.827, 'CONSOLIDATING': 0.902, 'VOLATILE': 0.968}
2026-04-27 15:28:32,023 INFO Regime epoch  6/50 — tr=0.6369 va=1.0102 acc=0.939
2026-04-27 15:28:32,180 INFO Regime epoch  7/50 — tr=0.6384 va=1.0090 acc=0.939
2026-04-27 15:28:32,320 INFO Regime epoch  8/50 — tr=0.6366 va=1.0091 acc=0.939
2026-04-27 15:28:32,462 INFO Regime epoch  9/50 — tr=0.6357 va=1.0054 acc=0.938
2026-04-27 15:28:32,624 INFO Regime epoch 10/50 — tr=0.6342 va=1.0039 acc=0.939 per_class={'TRENDING': 0.806, 'RANGING': 0.834, 'CONSOLIDATING': 0.901, 'VOLATILE': 0.969}
2026-04-27 15:28:32,776 INFO Regime epoch 11/50 — tr=0.6348 va=0.9977 acc=0.939
2026-04-27 15:28:32,916 INFO Regime epoch 12/50 — tr=0.6329 va=0.9966 acc=0.940
2026-04-27 15:28:33,062 INFO Regime epoch 13/50 — tr=0.6339 va=0.9942 acc=0.939
2026-04-27 15:28:33,207 INFO Regime epoch 14/50 — tr=0.6314 va=0.9918 acc=0.938
2026-04-27 15:28:33,357 INFO Regime epoch 15/50 — tr=0.6331 va=0.9883 acc=0.938 per_class={'TRENDING': 0.81, 'RANGING': 0.839, 'CONSOLIDATING': 0.897, 'VOLATILE': 0.969}
2026-04-27 15:28:33,500 INFO Regime epoch 16/50 — tr=0.6325 va=0.9897 acc=0.939
2026-04-27 15:28:33,643 INFO Regime epoch 17/50 — tr=0.6307 va=0.9864 acc=0.940
2026-04-27 15:28:33,783 INFO Regime epoch 18/50 — tr=0.6297 va=0.9845 acc=0.939
2026-04-27 15:28:33,923 INFO Regime epoch 19/50 — tr=0.6300 va=0.9806 acc=0.939
2026-04-27 15:28:34,072 INFO Regime epoch 20/50 — tr=0.6285 va=0.9812 acc=0.940 per_class={'TRENDING': 0.804, 'RANGING': 0.839, 'CONSOLIDATING': 0.9, 'VOLATILE': 0.971}
2026-04-27 15:28:34,210 INFO Regime epoch 21/50 — tr=0.6285 va=0.9808 acc=0.940
2026-04-27 15:28:34,357 INFO Regime epoch 22/50 — tr=0.6307 va=0.9778 acc=0.941
2026-04-27 15:28:34,508 INFO Regime epoch 23/50 — tr=0.6281 va=0.9713 acc=0.940
2026-04-27 15:28:34,651 INFO Regime epoch 24/50 — tr=0.6275 va=0.9718 acc=0.940
2026-04-27 15:28:34,815 INFO Regime epoch 25/50 — tr=0.6280 va=0.9744 acc=0.941 per_class={'TRENDING': 0.801, 'RANGING': 0.841, 'CONSOLIDATING': 0.897, 'VOLATILE': 0.974}
2026-04-27 15:28:34,962 INFO Regime epoch 26/50 — tr=0.6273 va=0.9692 acc=0.940
2026-04-27 15:28:35,100 INFO Regime epoch 27/50 — tr=0.6266 va=0.9664 acc=0.940
2026-04-27 15:28:35,239 INFO Regime epoch 28/50 — tr=0.6271 va=0.9627 acc=0.940
2026-04-27 15:28:35,381 INFO Regime epoch 29/50 — tr=0.6264 va=0.9622 acc=0.939
2026-04-27 15:28:35,530 INFO Regime epoch 30/50 — tr=0.6259 va=0.9616 acc=0.939 per_class={'TRENDING': 0.806, 'RANGING': 0.844, 'CONSOLIDATING': 0.895, 'VOLATILE': 0.972}
2026-04-27 15:28:35,669 INFO Regime epoch 31/50 — tr=0.6270 va=0.9619 acc=0.939
2026-04-27 15:28:35,809 INFO Regime epoch 32/50 — tr=0.6265 va=0.9648 acc=0.940
2026-04-27 15:28:35,952 INFO Regime epoch 33/50 — tr=0.6255 va=0.9584 acc=0.940
2026-04-27 15:28:36,093 INFO Regime epoch 34/50 — tr=0.6258 va=0.9595 acc=0.940
2026-04-27 15:28:36,244 INFO Regime epoch 35/50 — tr=0.6263 va=0.9612 acc=0.940 per_class={'TRENDING': 0.8, 'RANGING': 0.844, 'CONSOLIDATING': 0.893, 'VOLATILE': 0.974}
2026-04-27 15:28:36,389 INFO Regime epoch 36/50 — tr=0.6263 va=0.9603 acc=0.940
2026-04-27 15:28:36,525 INFO Regime epoch 37/50 — tr=0.6261 va=0.9599 acc=0.940
2026-04-27 15:28:36,665 INFO Regime epoch 38/50 — tr=0.6259 va=0.9562 acc=0.939
2026-04-27 15:28:36,810 INFO Regime epoch 39/50 — tr=0.6258 va=0.9573 acc=0.939
2026-04-27 15:28:36,958 INFO Regime epoch 40/50 — tr=0.6247 va=0.9564 acc=0.940 per_class={'TRENDING': 0.803, 'RANGING': 0.844, 'CONSOLIDATING': 0.896, 'VOLATILE': 0.973}
2026-04-27 15:28:37,092 INFO Regime epoch 41/50 — tr=0.6250 va=0.9581 acc=0.940
2026-04-27 15:28:37,225 INFO Regime epoch 42/50 — tr=0.6249 va=0.9575 acc=0.940
2026-04-27 15:28:37,355 INFO Regime epoch 43/50 — tr=0.6248 va=0.9582 acc=0.940
2026-04-27 15:28:37,487 INFO Regime epoch 44/50 — tr=0.6249 va=0.9572 acc=0.940
2026-04-27 15:28:37,630 INFO Regime epoch 45/50 — tr=0.6245 va=0.9565 acc=0.940 per_class={'TRENDING': 0.801, 'RANGING': 0.846, 'CONSOLIDATING': 0.898, 'VOLATILE': 0.972}
2026-04-27 15:28:37,762 INFO Regime epoch 46/50 — tr=0.6246 va=0.9545 acc=0.940
2026-04-27 15:28:37,895 INFO Regime epoch 47/50 — tr=0.6262 va=0.9535 acc=0.939
2026-04-27 15:28:38,026 INFO Regime epoch 48/50 — tr=0.6247 va=0.9553 acc=0.939
2026-04-27 15:28:38,160 INFO Regime epoch 49/50 — tr=0.6257 va=0.9533 acc=0.939
2026-04-27 15:28:38,302 INFO Regime epoch 50/50 — tr=0.6242 va=0.9545 acc=0.940 per_class={'TRENDING': 0.809, 'RANGING': 0.844, 'CONSOLIDATING': 0.9, 'VOLATILE': 0.97}
2026-04-27 15:28:38,315 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 15:28:38,315 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 15:28:38,316 INFO Regime phase LTF train: 7.3s
2026-04-27 15:28:38,465 INFO Regime LTF complete: acc=0.939, n=401471 per_class={'TRENDING': 0.806, 'RANGING': 0.846, 'CONSOLIDATING': 0.897, 'VOLATILE': 0.971}
2026-04-27 15:28:38,468 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 15:28:38,994 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 1778, 'RANGING': 57285, 'CONSOLIDATING': 4542, 'VOLATILE': 11019}  ambiguous=60314 (total=74624)  short_runs_zeroed=4774
2026-04-27 15:28:38,999 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 3.4794520547945207, 'RANGING': 17.033898305084747, 'CONSOLIDATING': 3.9358752166377817, 'VOLATILE': 5.842523860021209}
2026-04-27 15:28:39,005 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 1778, 'mean': 2.6590471099160558e-05, 'mean_over_std': 0.012770375199031195}, 'RANGING': {'n': 57284, 'mean': 8.05210952910413e-06, 'mean_over_std': 0.004238152373085909}, 'CONSOLIDATING': {'n': 4542, 'mean': 2.6402283987281153e-07, 'mean_over_std': 0.00015354245991368998}, 'VOLATILE': {'n': 11019, 'mean': 2.823213197675263e-05, 'mean_over_std': 0.010468794666397715}}
2026-04-27 15:28:39,006 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 915, 'mean': 7.666310957571063e-05, 'mean_over_std': 0.04299550456929985}, 'RANGING': {'n': 382, 'mean': 7.196437117461198e-05, 'mean_over_std': 0.05537815835040595}, 'CONSOLIDATING': {'n': 3375, 'mean': -3.0062004816363933e-06, 'mean_over_std': -0.0019542830050176476}, 'VOLATILE': {'n': 9638, 'mean': 1.2389816251671513e-05, 'mean_over_std': 0.004377959254493111}}
2026-04-27 15:28:39,012 INFO Regime retrain total: 199.7s (504761 samples)
2026-04-27 15:28:39,018 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-27 15:28:39,019 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 15:28:39,019 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 15:28:39,019 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-27 15:28:39,019 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-27 15:28:39,019 INFO Retrain complete. Total wall-clock: 199.7s
  DONE  Retrain regime [train-split retrain]
  START Retrain quality [round-journal retrain]
2026-04-27 15:28:40,614 INFO retrain environment: KAGGLE
2026-04-27 15:28:42,338 INFO Device: CUDA (2 GPU(s))
2026-04-27 15:28:42,349 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 15:28:42,350 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 15:28:42,350 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 15:28:42,350 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 15:28:42,350 INFO Retrain data split: train
2026-04-27 15:28:42,351 INFO === QualityScorer retrain ===
2026-04-27 15:28:42,501 INFO NumExpr defaulting to 4 threads.
2026-04-27 15:28:42,720 INFO QualityScorer: CUDA available — using GPU
2026-04-27 15:28:42,931 INFO QualityScorer loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (device=cuda)
2026-04-27 15:28:42,962 INFO Quality phase label creation: 0.0s (461 trades)
2026-04-27 15:28:42,962 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/quality_scorer.pkl_20260427_152842
2026-04-27 15:28:42,996 INFO QualityScorer: 461 samples, EV stats={'mean': -0.4635184109210968, 'std': 1.0923813581466675, 'n_pos': 96, 'n_neg': 365}, device=cuda
2026-04-27 15:28:42,996 INFO QualityScorer: normalised win labels by median_win=1.164 — EV range now [-1, +3]
2026-04-27 15:28:42,996 INFO QualityScorer: warm start from existing weights
2026-04-27 15:28:42,997 INFO QualityScorer: pos_weight=3.78 (n_pos=77 n_neg=291)
2026-04-27 15:28:45,275 INFO Quality epoch   1/100 — va_huber=0.7785
2026-04-27 15:28:45,314 INFO Quality epoch   2/100 — va_huber=0.7797
2026-04-27 15:28:45,335 INFO Quality epoch   3/100 — va_huber=0.7804
2026-04-27 15:28:45,357 INFO Quality epoch   4/100 — va_huber=0.7822
2026-04-27 15:28:45,379 INFO Quality epoch   5/100 — va_huber=0.7840
2026-04-27 15:28:45,505 INFO Quality epoch  11/100 — va_huber=0.7836
2026-04-27 15:28:45,506 INFO Quality early stop at epoch 11
2026-04-27 15:28:45,514 INFO QualityScorer EV model: MAE=0.878 dir_acc=0.667 n_val=93
2026-04-27 15:28:45,521 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-27 15:28:45,578 INFO Quality phase train: 2.6s | total: 3.2s
2026-04-27 15:28:45,585 INFO Retrain complete. Total wall-clock: 3.2s
  DONE  Retrain quality [round-journal retrain]
  START Retrain rl [round-journal retrain]
2026-04-27 15:28:46,985 INFO retrain environment: KAGGLE
2026-04-27 15:28:48,731 INFO Device: CUDA (2 GPU(s))
2026-04-27 15:28:48,742 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 15:28:48,742 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 15:28:48,742 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 15:28:48,743 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 15:28:48,743 INFO Retrain data split: train
2026-04-27 15:28:48,744 INFO === RLAgent (PPO) retrain ===
2026-04-27 15:28:48,746 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260427_152848
2026-04-27 15:28:49.678657: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777303729.702445   71794 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777303729.710436   71794 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777303729.731711   71794 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777303729.731753   71794 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777303729.731756   71794 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777303729.731759   71794 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-27 15:28:54,534 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-27 15:28:57,523 INFO RLAgent: PPO model loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-27 15:28:57,539 INFO RL phase episode loading: 0.0s (461 episodes)
2026-04-27 15:28:57,553 INFO RLAgent: warm start — fine-tuning existing PPO policy (lr=6.00e-05)
2026-04-27 15:29:07,310 INFO RLAgent: retrain complete, 461 episodes
2026-04-27 15:29:07,310 INFO RL phase PPO train: 9.8s | total: 18.6s
2026-04-27 15:29:07,319 INFO Retrain complete. Total wall-clock: 18.6s
  DONE  Retrain rl [round-journal retrain]

=== Round 3: Backtest on last 3yr (post-retrain evaluation) ===
  START Round 3 - Post-retrain backtest (last 3yr)
2026-04-27 15:29:09,693 INFO === STEP 6: BACKTEST (round3) ===
2026-04-27 15:29:09,694 INFO BT_WINDOW=round3 — post-retrain eval: 2022-08-05 → 2025-08-05 (last 3yr)
2026-04-27 15:29:09,695 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-04-27 15:29:09,695 INFO Round 3 — running backtest: 2022-08-05 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-27 15:33:35,655 INFO Round 3 backtest — 121 trades | avg WR=33.1% | avg PF=1.18 | avg Sharpe=1.17
2026-04-27 15:33:35,655 INFO   ml_trader: 121 trades | WR=33.1% | fixed PF=1.18 | Return=14.8% | ExpR=0.122 | DD=16.8% | Sharpe=1.17
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 121
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (121 rows)
2026-04-27 15:33:35,969 INFO Round 3: wrote 121 journal entries (total in file: 582)
  DONE  Round 3 - Post-retrain backtest (last 3yr)
  Saved Round 3 result → round3_summary.json
  Journal after Round 3: 582 entries

=== Round 3 → Retrain Quality + RL on Round 1+2+3 journal ===
  START Round 3 - Quality+RL retrain
2026-04-27 15:33:36,367 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-27 15:33:36,368 INFO Journal entries: 582
2026-04-27 15:33:36,368 INFO --- Training quality ---
2026-04-27 15:33:36,369 INFO Running retrain --model quality with JOURNAL_ALLOWED_SPLITS=train,validation,test,combined_eval,live,paper,production
2026-04-27 15:33:36,571 INFO retrain environment: KAGGLE
2026-04-27 15:33:38,345 INFO Device: CUDA (2 GPU(s))
2026-04-27 15:33:38,357 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 15:33:38,357 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 15:33:38,357 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 15:33:38,358 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 15:33:38,358 INFO Retrain data split: train
2026-04-27 15:33:38,359 INFO === QualityScorer retrain ===
2026-04-27 15:33:38,513 INFO NumExpr defaulting to 4 threads.
2026-04-27 15:33:38,741 INFO QualityScorer: CUDA available — using GPU
2026-04-27 15:33:38,957 INFO QualityScorer loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (device=cuda)
2026-04-27 15:33:38,999 INFO Quality phase label creation: 0.0s (582 trades)
2026-04-27 15:33:38,999 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/quality_scorer.pkl_20260427_153338
2026-04-27 15:33:39,042 INFO QualityScorer: 582 samples, EV stats={'mean': -0.4106804132461548, 'std': 1.117168664932251, 'n_pos': 136, 'n_neg': 446}, device=cuda
2026-04-27 15:33:39,042 INFO QualityScorer: normalised win labels by median_win=1.124 — EV range now [-1, +3]
2026-04-27 15:33:39,043 INFO QualityScorer: warm start from existing weights
2026-04-27 15:33:39,043 INFO QualityScorer: pos_weight=3.70 (n_pos=99 n_neg=366)
2026-04-27 15:33:41,523 INFO Quality epoch   1/100 — va_huber=1.0543
2026-04-27 15:33:41,565 INFO Quality epoch   2/100 — va_huber=1.0506
2026-04-27 15:33:41,590 INFO Quality epoch   3/100 — va_huber=1.0556
2026-04-27 15:33:41,614 INFO Quality epoch   4/100 — va_huber=1.0543
2026-04-27 15:33:41,637 INFO Quality epoch   5/100 — va_huber=1.0538
2026-04-27 15:33:41,983 INFO Quality epoch  11/100 — va_huber=1.0507
2026-04-27 15:33:42,213 INFO Quality epoch  21/100 — va_huber=1.0486
2026-04-27 15:33:42,446 INFO Quality epoch  31/100 — va_huber=1.0369
2026-04-27 15:33:42,677 INFO Quality epoch  41/100 — va_huber=1.0446
2026-04-27 15:33:42,699 INFO Quality early stop at epoch 42
2026-04-27 15:33:42,708 INFO QualityScorer EV model: MAE=1.199 dir_acc=0.427 n_val=117
2026-04-27 15:33:42,713 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-27 15:33:42,772 INFO Quality phase train: 3.8s | total: 4.4s
2026-04-27 15:33:42,779 INFO Retrain complete. Total wall-clock: 4.4s
2026-04-27 15:33:44,067 INFO Model quality: SUCCESS
2026-04-27 15:33:44,067 INFO --- Training rl ---
2026-04-27 15:33:44,067 INFO Running retrain --model rl with JOURNAL_ALLOWED_SPLITS=train,validation,test,combined_eval,live,paper,production
2026-04-27 15:33:44,266 INFO retrain environment: KAGGLE
2026-04-27 15:33:45,992 INFO Device: CUDA (2 GPU(s))
2026-04-27 15:33:46,004 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 15:33:46,004 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 15:33:46,004 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 15:33:46,004 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 15:33:46,005 INFO Retrain data split: train
2026-04-27 15:33:46,006 INFO === RLAgent (PPO) retrain ===
2026-04-27 15:33:46,008 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260427_153346
2026-04-27 15:33:46.941639: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777304026.965921   72081 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777304026.974008   72081 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777304026.996658   72081 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777304026.996694   72081 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777304026.996697   72081 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777304026.996699   72081 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-27 15:33:51,767 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-27 15:33:54,742 INFO RLAgent: PPO model loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-27 15:33:54,762 INFO RL phase episode loading: 0.0s (582 episodes)
2026-04-27 15:33:54,779 INFO RLAgent: warm start — fine-tuning existing PPO policy (lr=6.00e-05)
2026-04-27 15:34:06,940 INFO RLAgent: retrain complete, 582 episodes
2026-04-27 15:34:06,940 INFO RL phase PPO train: 12.2s | total: 20.9s
2026-04-27 15:34:06,950 INFO Retrain complete. Total wall-clock: 20.9s
  DONE  Round 3 - Quality+RL retrain

======================================================================
  BLIND BACKTEST PIPELINE COMPLETE
======================================================================
  Round 1 (val window)          trades=302  WR=19.2%  PF=0.604  Sharpe=-3.554
  Round 2 (blind test)          trades=159  WR=23.9%  PF=0.797  Sharpe=-1.583
  Round 3 (last 3yr)            trades=121  WR=33.1%  PF=1.183  Sharpe=1.170


WARNING: GITHUB_TOKEN not set — skipping GitHub push
2026-04-27 15:34:08,724 INFO Model rl: SUCCESS
2026-04-27 15:34:08,725 INFO Step 7b complete — summary: /kaggle/working/Multi-Bot/trading-system/ml_training/metrics/training_7b_summary.json