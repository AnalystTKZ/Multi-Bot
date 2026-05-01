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
2026-05-01 07:01:59,396 INFO Loading feature-engineered data...
2026-05-01 07:02:00,059 INFO Loaded 221743 rows, 202 features
2026-05-01 07:02:00,060 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-05-01 07:02:00,063 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-05-01 07:02:00,063 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-05-01 07:02:00,063 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-05-01 07:02:00,063 INFO No leakage confirmed: train < val < test timestamps

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
2026-05-01 07:02:02,529 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-05-01 07:02:02,529 INFO --- Training regime ---
2026-05-01 07:02:02,529 INFO Running retrain --model regime
2026-05-01 07:02:02,718 INFO retrain environment: KAGGLE
2026-05-01 07:02:04,413 INFO Device: CUDA (2 GPU(s))
2026-05-01 07:02:04,424 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 07:02:04,424 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 07:02:04,424 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-01 07:02:04,428 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-01 07:02:04,428 INFO Retrain data split: train
2026-05-01 07:02:04,429 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-05-01 07:02:04,590 INFO NumExpr defaulting to 4 threads.
2026-05-01 07:02:04,810 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-01 07:02:04,810 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 07:02:04,810 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 07:02:04,810 INFO Regime phase macro_correlations: 0.0s
2026-05-01 07:02:04,810 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-05-01 07:02:04,849 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-05-01 07:02:04,850 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:02:04,876 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:02:04,891 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:02:04,914 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:02:04,929 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:02:04,953 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:02:04,968 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:02:04,991 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:02:05,006 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:02:05,029 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:02:05,043 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:02:05,066 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:02:05,080 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:02:05,101 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:02:05,115 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:02:05,137 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:02:05,151 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:02:05,173 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:02:05,188 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:02:05,212 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:02:05,230 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:02:05,271 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:02:06,497 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-05-01 07:02:30,324 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-05-01 07:02:30,326 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 25.1s
2026-05-01 07:02:30,326 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-05-01 07:02:41,267 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-05-01 07:02:41,268 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 10.9s
2026-05-01 07:02:41,271 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-05-01 07:02:49,364 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-05-01 07:02:49,364 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 8.1s
2026-05-01 07:02:49,364 INFO Regime phase GMM HTF total: 44.1s
2026-05-01 07:02:49,365 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-05-01 07:04:02,542 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-05-01 07:04:02,545 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 73.2s
2026-05-01 07:04:02,546 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-05-01 07:04:35,042 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-05-01 07:04:35,046 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 32.5s
2026-05-01 07:04:35,047 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-05-01 07:04:57,797 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-05-01 07:04:57,798 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 22.8s
2026-05-01 07:04:57,798 INFO Regime phase GMM LTF total: 128.4s
2026-05-01 07:04:57,904 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-05-01 07:04:57,905 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:04:57,906 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:04:57,908 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:04:57,909 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:04:57,910 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:04:57,911 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:04:57,912 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:04:57,913 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:04:57,914 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:04:57,915 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:04:57,917 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:04:58,049 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:04:58,095 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:04:58,096 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:04:58,096 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:04:58,105 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:04:58,106 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:04:58,538 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 17, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 8360}  ambiguous=8312 (total=8402)  short_runs_zeroed=15
2026-05-01 07:04:58,539 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-05-01 07:04:58,721 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:04:58,754 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:04:58,755 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:04:58,756 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:04:58,764 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:04:58,765 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:04:59,157 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 100, 'BIAS_DOWN': 31, 'BIAS_NEUTRAL': 8271}  ambiguous=8272 (total=8402)  short_runs_zeroed=52
2026-05-01 07:04:59,158 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-05-01 07:04:59,348 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:04:59,385 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:04:59,385 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:04:59,386 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:04:59,394 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:04:59,395 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:04:59,788 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 46, 'BIAS_DOWN': 4, 'BIAS_NEUTRAL': 8352}  ambiguous=8316 (total=8402)  short_runs_zeroed=28
2026-05-01 07:04:59,789 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-05-01 07:04:59,963 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:00,000 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:00,001 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:00,002 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:00,011 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:00,012 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:00,424 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 15, 'BIAS_DOWN': 19, 'BIAS_NEUTRAL': 8368}  ambiguous=8331 (total=8402)  short_runs_zeroed=27
2026-05-01 07:05:00,425 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-05-01 07:05:00,609 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:00,644 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:00,645 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:00,645 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:00,653 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:00,654 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:01,045 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 93, 'BIAS_DOWN': 58, 'BIAS_NEUTRAL': 8252}  ambiguous=8287 (total=8403)  short_runs_zeroed=75
2026-05-01 07:05:01,046 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-05-01 07:05:01,221 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:01,255 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:01,256 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:01,256 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:01,264 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:01,265 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:01,648 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 87, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 8261}  ambiguous=8278 (total=8403)  short_runs_zeroed=63
2026-05-01 07:05:01,649 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-05-01 07:05:01,803 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:05:01,833 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:05:01,834 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:05:01,834 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:05:01,841 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:05:01,842 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:02,241 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 21, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 8314}  ambiguous=8303 (total=8402)  short_runs_zeroed=39
2026-05-01 07:05:02,242 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-05-01 07:05:02,404 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:02,436 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:02,437 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:02,437 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:02,446 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:02,447 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:02,859 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 37, 'BIAS_DOWN': 38, 'BIAS_NEUTRAL': 8327}  ambiguous=8321 (total=8402)  short_runs_zeroed=44
2026-05-01 07:05:02,861 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-05-01 07:05:03,044 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:03,080 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:03,081 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:03,081 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:03,090 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:03,091 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:03,506 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 79, 'BIAS_DOWN': 33, 'BIAS_NEUTRAL': 8290}  ambiguous=8328 (total=8402)  short_runs_zeroed=61
2026-05-01 07:05:03,507 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-05-01 07:05:03,676 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:03,711 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:03,712 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:03,712 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:03,721 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:03,722 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:04,128 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 42, 'BIAS_DOWN': 44, 'BIAS_NEUTRAL': 8317}  ambiguous=8318 (total=8403)  short_runs_zeroed=44
2026-05-01 07:05:04,130 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-05-01 07:05:04,405 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:05:04,464 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:05:04,465 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:05:04,466 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:05:04,476 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:05:04,478 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:05:05,302 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 107, 'BIAS_DOWN': 115, 'BIAS_NEUTRAL': 19595}  ambiguous=19636 (total=19817)  short_runs_zeroed=112
2026-05-01 07:05:05,304 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-05-01 07:05:05,481 INFO Regime phase HTF dataset build: 7.6s (103290 samples)
2026-05-01 07:05:05,482 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=1138 dropped=102152 classes={'BIAS_UP': 399, 'BIAS_DOWN': 243, 'BIAS_NEUTRAL': 496})
2026-05-01 07:05:05,483 INFO RegimeClassifier[mode=htf_bias]: 1138 samples, classes={'BIAS_UP': 399, 'BIAS_DOWN': 243, 'BIAS_NEUTRAL': 496}, device=cuda
2026-05-01 07:05:05,483 INFO RegimeClassifier: sample weights — mean=0.713  ambiguous(<0.4)=0.0%
2026-05-01 07:05:05,771 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-05-01 07:05:05,771 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-05-01 07:05:10,653 INFO Regime epoch  1/50 — tr=0.7438 va=2.6332 acc=0.338 per_class={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 1.0}
2026-05-01 07:05:10,668 INFO Regime epoch  2/50 — tr=0.7411 va=2.4233 acc=0.338
2026-05-01 07:05:10,680 INFO Regime epoch  3/50 — tr=0.7395 va=2.3352 acc=0.338
2026-05-01 07:05:10,693 INFO Regime epoch  4/50 — tr=0.7337 va=2.2846 acc=0.338
2026-05-01 07:05:10,709 INFO Regime epoch  5/50 — tr=0.7365 va=2.2452 acc=0.338 per_class={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 1.0}
2026-05-01 07:05:10,721 INFO Regime epoch  6/50 — tr=0.7287 va=2.2137 acc=0.338
2026-05-01 07:05:10,734 INFO Regime epoch  7/50 — tr=0.7282 va=2.1809 acc=0.338
2026-05-01 07:05:10,747 INFO Regime epoch  8/50 — tr=0.7269 va=2.1494 acc=0.338
2026-05-01 07:05:10,759 INFO Regime epoch  9/50 — tr=0.6919 va=2.1161 acc=0.338
2026-05-01 07:05:10,775 INFO Regime epoch 10/50 — tr=0.6848 va=2.0819 acc=0.338 per_class={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 1.0}
2026-05-01 07:05:10,788 INFO Regime epoch 11/50 — tr=0.6675 va=2.0480 acc=0.338
2026-05-01 07:05:10,801 INFO Regime epoch 12/50 — tr=0.6694 va=2.0148 acc=0.338
2026-05-01 07:05:10,814 INFO Regime epoch 13/50 — tr=0.6534 va=1.9795 acc=0.338
2026-05-01 07:05:10,827 INFO Regime epoch 14/50 — tr=0.6384 va=1.9443 acc=0.346
2026-05-01 07:05:10,844 INFO Regime epoch 15/50 — tr=0.6382 va=1.9070 acc=0.377 per_class={'BIAS_UP': 0.122, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 1.0}
2026-05-01 07:05:10,857 INFO Regime epoch 16/50 — tr=0.6236 va=1.8711 acc=0.469
2026-05-01 07:05:10,870 INFO Regime epoch 17/50 — tr=0.6214 va=1.8340 acc=0.575
2026-05-01 07:05:10,883 INFO Regime epoch 18/50 — tr=0.6019 va=1.7994 acc=0.715
2026-05-01 07:05:10,895 INFO Regime epoch 19/50 — tr=0.6078 va=1.7617 acc=0.737
2026-05-01 07:05:10,911 INFO Regime epoch 20/50 — tr=0.5965 va=1.7262 acc=0.759 per_class={'BIAS_UP': 0.635, 'BIAS_DOWN': 0.636, 'BIAS_NEUTRAL': 1.0}
2026-05-01 07:05:10,924 INFO Regime epoch 21/50 — tr=0.5786 va=1.6916 acc=0.781
2026-05-01 07:05:10,937 INFO Regime epoch 22/50 — tr=0.5834 va=1.6580 acc=0.825
2026-05-01 07:05:10,950 INFO Regime epoch 23/50 — tr=0.5758 va=1.6246 acc=0.873
2026-05-01 07:05:10,963 INFO Regime epoch 24/50 — tr=0.5709 va=1.5925 acc=0.886
2026-05-01 07:05:10,978 INFO Regime epoch 25/50 — tr=0.5710 va=1.5630 acc=0.895 per_class={'BIAS_UP': 0.77, 'BIAS_DOWN': 0.909, 'BIAS_NEUTRAL': 1.0}
2026-05-01 07:05:10,991 INFO Regime epoch 26/50 — tr=0.5635 va=1.5366 acc=0.904
2026-05-01 07:05:11,004 INFO Regime epoch 27/50 — tr=0.5584 va=1.5115 acc=0.908
2026-05-01 07:05:11,017 INFO Regime epoch 28/50 — tr=0.5577 va=1.4873 acc=0.917
2026-05-01 07:05:11,029 INFO Regime epoch 29/50 — tr=0.5547 va=1.4642 acc=0.925
2026-05-01 07:05:11,044 INFO Regime epoch 30/50 — tr=0.5502 va=1.4438 acc=0.925 per_class={'BIAS_UP': 0.811, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 1.0}
2026-05-01 07:05:11,056 INFO Regime epoch 31/50 — tr=0.5488 va=1.4252 acc=0.925
2026-05-01 07:05:11,070 INFO Regime epoch 32/50 — tr=0.5525 va=1.4085 acc=0.925
2026-05-01 07:05:11,084 INFO Regime epoch 33/50 — tr=0.5489 va=1.3911 acc=0.925
2026-05-01 07:05:11,097 INFO Regime epoch 34/50 — tr=0.5449 va=1.3774 acc=0.930
2026-05-01 07:05:11,114 INFO Regime epoch 35/50 — tr=0.5483 va=1.3638 acc=0.934 per_class={'BIAS_UP': 0.838, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 1.0}
2026-05-01 07:05:11,128 INFO Regime epoch 36/50 — tr=0.5359 va=1.3524 acc=0.934
2026-05-01 07:05:11,140 INFO Regime epoch 37/50 — tr=0.5425 va=1.3407 acc=0.930
2026-05-01 07:05:11,152 INFO Regime epoch 38/50 — tr=0.5369 va=1.3326 acc=0.930
2026-05-01 07:05:11,164 INFO Regime epoch 39/50 — tr=0.5351 va=1.3239 acc=0.934
2026-05-01 07:05:11,180 INFO Regime epoch 40/50 — tr=0.5317 va=1.3164 acc=0.934 per_class={'BIAS_UP': 0.851, 'BIAS_DOWN': 0.948, 'BIAS_NEUTRAL': 1.0}
2026-05-01 07:05:11,193 INFO Regime epoch 41/50 — tr=0.5315 va=1.3075 acc=0.934
2026-05-01 07:05:11,204 INFO Regime epoch 42/50 — tr=0.5334 va=1.3020 acc=0.934
2026-05-01 07:05:11,216 INFO Regime epoch 43/50 — tr=0.5401 va=1.2989 acc=0.934
2026-05-01 07:05:11,228 INFO Regime epoch 44/50 — tr=0.5318 va=1.2944 acc=0.939
2026-05-01 07:05:11,243 INFO Regime epoch 45/50 — tr=0.5348 va=1.2900 acc=0.939 per_class={'BIAS_UP': 0.865, 'BIAS_DOWN': 0.948, 'BIAS_NEUTRAL': 1.0}
2026-05-01 07:05:11,255 INFO Regime epoch 46/50 — tr=0.5402 va=1.2850 acc=0.939
2026-05-01 07:05:11,267 INFO Regime epoch 47/50 — tr=0.5311 va=1.2833 acc=0.934
2026-05-01 07:05:11,279 INFO Regime epoch 48/50 — tr=0.5379 va=1.2815 acc=0.934
2026-05-01 07:05:11,291 INFO Regime epoch 49/50 — tr=0.5336 va=1.2788 acc=0.934
2026-05-01 07:05:11,306 INFO Regime epoch 50/50 — tr=0.5334 va=1.2788 acc=0.939 per_class={'BIAS_UP': 0.878, 'BIAS_DOWN': 0.948, 'BIAS_NEUTRAL': 0.987}
2026-05-01 07:05:11,315 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-01 07:05:11,315 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-01 07:05:11,315 INFO Regime phase HTF train: 5.8s
2026-05-01 07:05:11,444 INFO Regime HTF complete: acc=0.934, n=103290 per_class={'BIAS_UP': 0.865, 'BIAS_DOWN': 0.948, 'BIAS_NEUTRAL': 0.987}
2026-05-01 07:05:11,446 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:05:11,621 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 107, 'BIAS_DOWN': 115, 'BIAS_NEUTRAL': 19595}  ambiguous=19636 (total=19817)  short_runs_zeroed=112
2026-05-01 07:05:11,629 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 4.28, 'BIAS_DOWN': 4.791666666666667, 'BIAS_NEUTRAL': 391.9}
2026-05-01 07:05:11,632 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 107, 'mean': -0.00021706688424743013, 'mean_over_std': -0.04806767606653151}, 'BIAS_DOWN': {'n': 115, 'mean': -0.00020797041876048362, 'mean_over_std': -0.029260022054973262}, 'BIAS_NEUTRAL': {'n': 19594, 'mean': 4.4550533138328785e-05, 'mean_over_std': 0.011372683463534268}}
2026-05-01 07:05:11,633 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 66, 'mean': 0.0003288215669218909, 'mean_over_std': 0.07277335807221066}, 'BIAS_DOWN': {'n': 59, 'mean': -0.0010984382215802496, 'mean_over_std': -0.13394112338746375}, 'BIAS_NEUTRAL': {'n': 56, 'mean': -0.00020920056804862467, 'mean_over_std': -0.06887192756862072}}
2026-05-01 07:05:11,636 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-05-01 07:05:11,638 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:11,639 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:11,641 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:11,643 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:11,644 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:11,646 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:11,647 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:11,649 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:11,651 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:11,652 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:11,655 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:05:11,668 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:11,672 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:11,673 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:11,673 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:11,673 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:11,676 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:12,367 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 619, 'RANGING': 24879, 'CONSOLIDATING': 1882, 'VOLATILE': 5358}  ambiguous=26508 (total=32738)  short_runs_zeroed=2571
2026-05-01 07:05:12,370 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-05-01 07:05:12,508 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:12,510 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:12,511 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:12,512 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:12,512 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:12,514 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:13,135 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 752, 'RANGING': 24932, 'CONSOLIDATING': 2099, 'VOLATILE': 4955}  ambiguous=26224 (total=32738)  short_runs_zeroed=2112
2026-05-01 07:05:13,139 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-05-01 07:05:13,280 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:13,283 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:13,284 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:13,284 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:13,284 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:13,286 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:13,905 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 617, 'RANGING': 24738, 'CONSOLIDATING': 2103, 'VOLATILE': 5282}  ambiguous=26101 (total=32740)  short_runs_zeroed=2218
2026-05-01 07:05:13,908 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-05-01 07:05:14,049 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:14,052 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:14,053 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:14,053 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:14,053 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:14,055 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:14,668 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 842, 'RANGING': 25174, 'CONSOLIDATING': 1996, 'VOLATILE': 4727}  ambiguous=26490 (total=32739)  short_runs_zeroed=2045
2026-05-01 07:05:14,671 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-05-01 07:05:14,804 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:14,807 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:14,807 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:14,808 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:14,808 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:14,810 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:15,470 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 761, 'RANGING': 24579, 'CONSOLIDATING': 2089, 'VOLATILE': 5311}  ambiguous=25954 (total=32740)  short_runs_zeroed=2172
2026-05-01 07:05:15,473 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-05-01 07:05:15,608 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:15,610 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:15,611 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:15,612 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:15,612 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:15,614 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:16,234 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 763, 'RANGING': 25215, 'CONSOLIDATING': 1968, 'VOLATILE': 4793}  ambiguous=26501 (total=32739)  short_runs_zeroed=1959
2026-05-01 07:05:16,237 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-05-01 07:05:16,373 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:05:16,374 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:05:16,375 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:05:16,375 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:05:16,375 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:05:16,377 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:16,989 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 474, 'RANGING': 25095, 'CONSOLIDATING': 1967, 'VOLATILE': 5203}  ambiguous=26572 (total=32739)  short_runs_zeroed=2307
2026-05-01 07:05:16,992 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-05-01 07:05:17,129 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:17,131 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:17,132 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:17,133 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:17,133 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:17,135 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:17,804 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 881, 'RANGING': 25133, 'CONSOLIDATING': 1987, 'VOLATILE': 4739}  ambiguous=26356 (total=32740)  short_runs_zeroed=1985
2026-05-01 07:05:17,807 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-05-01 07:05:17,941 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:17,943 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:17,944 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:17,945 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:17,945 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:17,947 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:18,576 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 699, 'RANGING': 24989, 'CONSOLIDATING': 2130, 'VOLATILE': 4923}  ambiguous=26228 (total=32741)  short_runs_zeroed=2004
2026-05-01 07:05:18,579 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-05-01 07:05:18,719 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:18,722 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:18,722 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:18,723 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:18,723 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:18,725 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:19,345 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 808, 'RANGING': 24290, 'CONSOLIDATING': 2180, 'VOLATILE': 5465}  ambiguous=25645 (total=32743)  short_runs_zeroed=2211
2026-05-01 07:05:19,348 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-05-01 07:05:19,499 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:05:19,503 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:05:19,504 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:05:19,504 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:05:19,505 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:05:19,508 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:05:20,827 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 1778, 'RANGING': 57285, 'CONSOLIDATING': 4542, 'VOLATILE': 11019}  ambiguous=60314 (total=74624)  short_runs_zeroed=4774
2026-05-01 07:05:20,833 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-05-01 07:05:21,131 INFO Regime phase LTF dataset build: 9.5s (401471 samples)
2026-05-01 07:05:21,133 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=79106 dropped=322365 classes={'TRENDING': 4775, 'RANGING': 1876, 'CONSOLIDATING': 18867, 'VOLATILE': 53588})
2026-05-01 07:05:21,149 INFO RegimeClassifier[mode=ltf_behaviour]: 79106 samples, classes={'TRENDING': 4775, 'RANGING': 1876, 'CONSOLIDATING': 18867, 'VOLATILE': 53588}, device=cuda
2026-05-01 07:05:21,149 INFO RegimeClassifier: sample weights — mean=0.811  ambiguous(<0.4)=0.0%
2026-05-01 07:05:21,151 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-05-01 07:05:21,152 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-05-01 07:05:21,331 INFO Regime epoch  1/50 — tr=1.0813 va=2.2002 acc=0.160 per_class={'TRENDING': 0.127, 'RANGING': 0.6, 'CONSOLIDATING': 0.077, 'VOLATILE': 0.176}
2026-05-01 07:05:21,469 INFO Regime epoch  2/50 — tr=1.0704 va=2.1742 acc=0.172
2026-05-01 07:05:21,611 INFO Regime epoch  3/50 — tr=1.0469 va=2.1117 acc=0.294
2026-05-01 07:05:21,749 INFO Regime epoch  4/50 — tr=1.0103 va=2.0146 acc=0.530
2026-05-01 07:05:21,899 INFO Regime epoch  5/50 — tr=0.9601 va=1.8855 acc=0.683 per_class={'TRENDING': 0.647, 'RANGING': 0.573, 'CONSOLIDATING': 0.3, 'VOLATILE': 0.828}
2026-05-01 07:05:22,044 INFO Regime epoch  6/50 — tr=0.9054 va=1.7343 acc=0.751
2026-05-01 07:05:22,192 INFO Regime epoch  7/50 — tr=0.8546 va=1.6102 acc=0.824
2026-05-01 07:05:22,333 INFO Regime epoch  8/50 — tr=0.8113 va=1.5148 acc=0.865
2026-05-01 07:05:22,471 INFO Regime epoch  9/50 — tr=0.7788 va=1.4489 acc=0.883
2026-05-01 07:05:22,623 INFO Regime epoch 10/50 — tr=0.7528 va=1.3944 acc=0.898 per_class={'TRENDING': 0.815, 'RANGING': 0.585, 'CONSOLIDATING': 0.891, 'VOLATILE': 0.921}
2026-05-01 07:05:22,767 INFO Regime epoch 11/50 — tr=0.7336 va=1.3493 acc=0.906
2026-05-01 07:05:22,907 INFO Regime epoch 12/50 — tr=0.7199 va=1.3182 acc=0.914
2026-05-01 07:05:23,058 INFO Regime epoch 13/50 — tr=0.7079 va=1.2849 acc=0.917
2026-05-01 07:05:23,208 INFO Regime epoch 14/50 — tr=0.6967 va=1.2623 acc=0.922
2026-05-01 07:05:23,365 INFO Regime epoch 15/50 — tr=0.6910 va=1.2395 acc=0.923 per_class={'TRENDING': 0.813, 'RANGING': 0.688, 'CONSOLIDATING': 0.904, 'VOLATILE': 0.949}
2026-05-01 07:05:23,510 INFO Regime epoch 16/50 — tr=0.6862 va=1.2179 acc=0.924
2026-05-01 07:05:23,653 INFO Regime epoch 17/50 — tr=0.6789 va=1.2018 acc=0.926
2026-05-01 07:05:23,797 INFO Regime epoch 18/50 — tr=0.6758 va=1.1864 acc=0.926
2026-05-01 07:05:23,941 INFO Regime epoch 19/50 — tr=0.6712 va=1.1751 acc=0.928
2026-05-01 07:05:24,091 INFO Regime epoch 20/50 — tr=0.6697 va=1.1575 acc=0.928 per_class={'TRENDING': 0.799, 'RANGING': 0.771, 'CONSOLIDATING': 0.879, 'VOLATILE': 0.964}
2026-05-01 07:05:24,232 INFO Regime epoch 21/50 — tr=0.6664 va=1.1415 acc=0.928
2026-05-01 07:05:24,369 INFO Regime epoch 22/50 — tr=0.6617 va=1.1356 acc=0.930
2026-05-01 07:05:24,505 INFO Regime epoch 23/50 — tr=0.6595 va=1.1251 acc=0.930
2026-05-01 07:05:24,644 INFO Regime epoch 24/50 — tr=0.6569 va=1.1142 acc=0.930
2026-05-01 07:05:24,795 INFO Regime epoch 25/50 — tr=0.6570 va=1.1039 acc=0.930 per_class={'TRENDING': 0.8, 'RANGING': 0.807, 'CONSOLIDATING': 0.873, 'VOLATILE': 0.968}
2026-05-01 07:05:24,944 INFO Regime epoch 26/50 — tr=0.6542 va=1.0911 acc=0.928
2026-05-01 07:05:25,091 INFO Regime epoch 27/50 — tr=0.6538 va=1.0829 acc=0.928
2026-05-01 07:05:25,237 INFO Regime epoch 28/50 — tr=0.6503 va=1.0810 acc=0.930
2026-05-01 07:05:25,385 INFO Regime epoch 29/50 — tr=0.6490 va=1.0709 acc=0.930
2026-05-01 07:05:25,543 INFO Regime epoch 30/50 — tr=0.6468 va=1.0704 acc=0.932 per_class={'TRENDING': 0.798, 'RANGING': 0.824, 'CONSOLIDATING': 0.87, 'VOLATILE': 0.971}
2026-05-01 07:05:25,689 INFO Regime epoch 31/50 — tr=0.6447 va=1.0599 acc=0.931
2026-05-01 07:05:25,841 INFO Regime epoch 32/50 — tr=0.6449 va=1.0541 acc=0.931
2026-05-01 07:05:25,994 INFO Regime epoch 33/50 — tr=0.6463 va=1.0547 acc=0.931
2026-05-01 07:05:26,146 INFO Regime epoch 34/50 — tr=0.6433 va=1.0488 acc=0.931
2026-05-01 07:05:26,299 INFO Regime epoch 35/50 — tr=0.6432 va=1.0444 acc=0.931 per_class={'TRENDING': 0.806, 'RANGING': 0.837, 'CONSOLIDATING': 0.867, 'VOLATILE': 0.97}
2026-05-01 07:05:26,440 INFO Regime epoch 36/50 — tr=0.6426 va=1.0377 acc=0.931
2026-05-01 07:05:26,579 INFO Regime epoch 37/50 — tr=0.6405 va=1.0342 acc=0.931
2026-05-01 07:05:26,716 INFO Regime epoch 38/50 — tr=0.6403 va=1.0342 acc=0.931
2026-05-01 07:05:26,858 INFO Regime epoch 39/50 — tr=0.6408 va=1.0312 acc=0.929
2026-05-01 07:05:27,005 INFO Regime epoch 40/50 — tr=0.6413 va=1.0281 acc=0.931 per_class={'TRENDING': 0.806, 'RANGING': 0.841, 'CONSOLIDATING': 0.864, 'VOLATILE': 0.97}
2026-05-01 07:05:27,146 INFO Regime epoch 41/50 — tr=0.6392 va=1.0247 acc=0.931
2026-05-01 07:05:27,297 INFO Regime epoch 42/50 — tr=0.6399 va=1.0257 acc=0.931
2026-05-01 07:05:27,439 INFO Regime epoch 43/50 — tr=0.6387 va=1.0223 acc=0.931
2026-05-01 07:05:27,591 INFO Regime epoch 44/50 — tr=0.6396 va=1.0236 acc=0.931
2026-05-01 07:05:27,766 INFO Regime epoch 45/50 — tr=0.6379 va=1.0234 acc=0.931 per_class={'TRENDING': 0.803, 'RANGING': 0.841, 'CONSOLIDATING': 0.868, 'VOLATILE': 0.97}
2026-05-01 07:05:27,908 INFO Regime epoch 46/50 — tr=0.6390 va=1.0242 acc=0.931
2026-05-01 07:05:28,045 INFO Regime epoch 47/50 — tr=0.6389 va=1.0247 acc=0.931
2026-05-01 07:05:28,195 INFO Regime epoch 48/50 — tr=0.6404 va=1.0275 acc=0.930
2026-05-01 07:05:28,337 INFO Regime epoch 49/50 — tr=0.6399 va=1.0233 acc=0.931
2026-05-01 07:05:28,488 INFO Regime epoch 50/50 — tr=0.6381 va=1.0216 acc=0.931 per_class={'TRENDING': 0.803, 'RANGING': 0.844, 'CONSOLIDATING': 0.866, 'VOLATILE': 0.97}
2026-05-01 07:05:28,503 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-01 07:05:28,503 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-01 07:05:28,504 INFO Regime phase LTF train: 7.4s
2026-05-01 07:05:28,636 INFO Regime LTF complete: acc=0.931, n=401471 per_class={'TRENDING': 0.803, 'RANGING': 0.844, 'CONSOLIDATING': 0.866, 'VOLATILE': 0.97}
2026-05-01 07:05:28,640 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:05:29,185 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 1778, 'RANGING': 57285, 'CONSOLIDATING': 4542, 'VOLATILE': 11019}  ambiguous=60314 (total=74624)  short_runs_zeroed=4774
2026-05-01 07:05:29,190 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 3.4794520547945207, 'RANGING': 17.033898305084747, 'CONSOLIDATING': 3.9358752166377817, 'VOLATILE': 5.842523860021209}
2026-05-01 07:05:29,197 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 1778, 'mean': 2.6590471099160558e-05, 'mean_over_std': 0.012770375199031195}, 'RANGING': {'n': 57284, 'mean': 8.05210952910413e-06, 'mean_over_std': 0.004238152373085909}, 'CONSOLIDATING': {'n': 4542, 'mean': 2.6402283987281153e-07, 'mean_over_std': 0.00015354245991368998}, 'VOLATILE': {'n': 11019, 'mean': 2.823213197675263e-05, 'mean_over_std': 0.010468794666397715}}
2026-05-01 07:05:29,198 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 915, 'mean': 7.666310957571063e-05, 'mean_over_std': 0.04299550456929985}, 'RANGING': {'n': 382, 'mean': 7.196437117461198e-05, 'mean_over_std': 0.05537815835040595}, 'CONSOLIDATING': {'n': 3375, 'mean': -3.0062004816363933e-06, 'mean_over_std': -0.0019542830050176476}, 'VOLATILE': {'n': 9638, 'mean': 1.2389816251671513e-05, 'mean_over_std': 0.004377959254493111}}
2026-05-01 07:05:29,202 INFO Regime retrain total: 204.8s (504761 samples)
2026-05-01 07:05:29,218 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-01 07:05:29,218 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 07:05:29,218 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 07:05:29,218 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-05-01 07:05:29,219 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-05-01 07:05:29,219 INFO Retrain complete. Total wall-clock: 204.8s
2026-05-01 07:05:31,789 INFO Model regime: SUCCESS
2026-05-01 07:05:31,789 INFO --- Training gru ---
2026-05-01 07:05:31,789 INFO Running retrain --model gru
2026-05-01 07:05:32,026 INFO retrain environment: KAGGLE
2026-05-01 07:05:33,683 INFO Device: CUDA (2 GPU(s))
2026-05-01 07:05:33,692 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 07:05:33,692 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 07:05:33,692 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-01 07:05:33,693 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-01 07:05:33,693 INFO Retrain data split: train
2026-05-01 07:05:33,694 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-05-01 07:05:33,845 INFO NumExpr defaulting to 4 threads.
2026-05-01 07:05:34,048 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-01 07:05:34,048 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 07:05:34,049 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 07:05:34,049 INFO GRU phase macro_correlations: 0.0s
2026-05-01 07:05:34,049 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-05-01 07:05:34,050 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260501_070534
2026-05-01 07:05:34,053 WARNING WeightsManifest: no manifest at /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json — assuming compatible (run a full retrain to generate one)
2026-05-01 07:05:34,201 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:34,223 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:34,238 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:34,246 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:34,248 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-01 07:05:34,248 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 07:05:34,248 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 07:05:34,249 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-05-01 07:05:34,250 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:34,344 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 17, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 8360}  ambiguous=8312 (total=8402)  short_runs_zeroed=15
2026-05-01 07:05:34,345 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:34,607 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 619, 'RANGING': 24879, 'CONSOLIDATING': 1882, 'VOLATILE': 5358}  ambiguous=26508 (total=32738)  short_runs_zeroed=2571
2026-05-01 07:05:34,634 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:34,926 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:35,067 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:35,172 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:35,389 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:35,409 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:35,425 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:35,434 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:35,435 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:35,526 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 100, 'BIAS_DOWN': 31, 'BIAS_NEUTRAL': 8271}  ambiguous=8272 (total=8402)  short_runs_zeroed=52
2026-05-01 07:05:35,528 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:35,791 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 752, 'RANGING': 24932, 'CONSOLIDATING': 2099, 'VOLATILE': 4955}  ambiguous=26224 (total=32738)  short_runs_zeroed=2112
2026-05-01 07:05:35,807 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:36,092 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:36,218 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:36,315 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:36,503 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:36,523 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:36,538 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:36,546 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:36,546 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:36,642 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 46, 'BIAS_DOWN': 4, 'BIAS_NEUTRAL': 8352}  ambiguous=8316 (total=8402)  short_runs_zeroed=28
2026-05-01 07:05:36,644 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:36,914 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 617, 'RANGING': 24738, 'CONSOLIDATING': 2103, 'VOLATILE': 5282}  ambiguous=26101 (total=32740)  short_runs_zeroed=2218
2026-05-01 07:05:36,929 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:37,197 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:37,340 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:37,441 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:37,658 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:37,678 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:37,691 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:37,698 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:37,699 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:37,789 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 15, 'BIAS_DOWN': 19, 'BIAS_NEUTRAL': 8368}  ambiguous=8331 (total=8402)  short_runs_zeroed=27
2026-05-01 07:05:37,791 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:38,042 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 842, 'RANGING': 25174, 'CONSOLIDATING': 1996, 'VOLATILE': 4727}  ambiguous=26490 (total=32739)  short_runs_zeroed=2045
2026-05-01 07:05:38,066 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:38,342 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:38,472 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:38,568 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:38,758 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:38,780 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:38,795 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:38,803 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:38,804 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:38,901 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 93, 'BIAS_DOWN': 58, 'BIAS_NEUTRAL': 8252}  ambiguous=8287 (total=8403)  short_runs_zeroed=75
2026-05-01 07:05:38,903 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:39,167 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 761, 'RANGING': 24579, 'CONSOLIDATING': 2089, 'VOLATILE': 5311}  ambiguous=25954 (total=32740)  short_runs_zeroed=2172
2026-05-01 07:05:39,185 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:39,459 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:39,595 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:39,693 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:39,888 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:39,908 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:39,923 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:39,930 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:39,931 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:40,020 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 87, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 8261}  ambiguous=8278 (total=8403)  short_runs_zeroed=63
2026-05-01 07:05:40,021 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:40,283 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 763, 'RANGING': 25215, 'CONSOLIDATING': 1968, 'VOLATILE': 4793}  ambiguous=26501 (total=32739)  short_runs_zeroed=1959
2026-05-01 07:05:40,298 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:40,565 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:40,693 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:40,791 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:40,958 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:05:40,976 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:05:40,991 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:05:40,998 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:05:40,998 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:41,090 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 21, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 8314}  ambiguous=8303 (total=8402)  short_runs_zeroed=39
2026-05-01 07:05:41,092 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:41,351 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 474, 'RANGING': 25095, 'CONSOLIDATING': 1967, 'VOLATILE': 5203}  ambiguous=26572 (total=32739)  short_runs_zeroed=2307
2026-05-01 07:05:41,363 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:41,625 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:41,754 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:41,856 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:42,036 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:42,055 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:42,069 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:42,076 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:42,077 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:42,166 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 37, 'BIAS_DOWN': 38, 'BIAS_NEUTRAL': 8327}  ambiguous=8321 (total=8402)  short_runs_zeroed=44
2026-05-01 07:05:42,168 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:42,432 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 881, 'RANGING': 25133, 'CONSOLIDATING': 1987, 'VOLATILE': 4739}  ambiguous=26356 (total=32740)  short_runs_zeroed=1985
2026-05-01 07:05:42,450 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:42,708 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:42,838 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:42,936 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:43,119 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:43,138 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:43,152 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:43,160 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:43,161 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:43,249 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 79, 'BIAS_DOWN': 33, 'BIAS_NEUTRAL': 8290}  ambiguous=8328 (total=8402)  short_runs_zeroed=61
2026-05-01 07:05:43,251 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:43,510 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 699, 'RANGING': 24989, 'CONSOLIDATING': 2130, 'VOLATILE': 4923}  ambiguous=26228 (total=32741)  short_runs_zeroed=2004
2026-05-01 07:05:43,525 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:43,792 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:43,924 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:44,024 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:44,214 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:44,235 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:44,249 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:44,256 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:05:44,257 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:44,349 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 42, 'BIAS_DOWN': 44, 'BIAS_NEUTRAL': 8317}  ambiguous=8318 (total=8403)  short_runs_zeroed=44
2026-05-01 07:05:44,351 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:44,612 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 808, 'RANGING': 24290, 'CONSOLIDATING': 2180, 'VOLATILE': 5465}  ambiguous=25645 (total=32743)  short_runs_zeroed=2211
2026-05-01 07:05:44,631 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:44,903 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:45,030 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:45,131 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:05:45,440 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:05:45,465 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:05:45,481 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:05:45,491 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:05:45,492 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:05:45,663 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 107, 'BIAS_DOWN': 115, 'BIAS_NEUTRAL': 19595}  ambiguous=19636 (total=19817)  short_runs_zeroed=112
2026-05-01 07:05:45,666 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:05:46,209 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 1778, 'RANGING': 57285, 'CONSOLIDATING': 4542, 'VOLATILE': 11019}  ambiguous=60314 (total=74624)  short_runs_zeroed=4774
2026-05-01 07:05:46,255 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:05:46,781 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:05:46,984 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:05:47,115 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:05:47,234 INFO train_multi: 44 segments, ~6069276 total bars
2026-05-01 07:05:47,503 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-05-01 07:05:47,503 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-05-01 07:05:47,503 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-05-01 07:05:47,503 INFO train_multi: building combined dataset for TF=ALL (44 segments)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
2026-05-01 07:06:41,594 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-05-01 07:06:41,594 INFO train_multi TF=ALL: estimated peak RAM = 11232 MB (train=479966 val=120011 n_feat=78 seq_len=30)
2026-05-01 07:06:43,007 INFO train_multi TF=ALL: train=479966 val=120011 (5623 MB tensors)
2026-05-01 07:06:49,858 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=18, min_epochs=22)
2026-05-01 07:07:06,412 INFO train_multi TF=ALL epoch 1/50 train=0.8945 val=0.8880
2026-05-01 07:07:06,421 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 07:07:06,422 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:07:06,422 INFO train_multi TF=ALL: new best val=0.8880 — saved
2026-05-01 07:07:20,335 INFO train_multi TF=ALL epoch 2/50 train=0.8688 val=0.8212
2026-05-01 07:07:20,340 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 07:07:20,340 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:07:20,341 INFO train_multi TF=ALL: new best val=0.8212 — saved
2026-05-01 07:07:34,535 INFO train_multi TF=ALL epoch 3/50 train=0.7227 val=0.6879
2026-05-01 07:07:34,540 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 07:07:34,540 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:07:34,540 INFO train_multi TF=ALL: new best val=0.6879 — saved
2026-05-01 07:07:48,469 INFO train_multi TF=ALL epoch 4/50 train=0.6924 val=0.6885
2026-05-01 07:08:02,494 INFO train_multi TF=ALL epoch 5/50 train=0.6910 val=0.6885
2026-05-01 07:08:16,561 INFO train_multi TF=ALL epoch 6/50 train=0.6909 val=0.6885
2026-05-01 07:08:30,863 INFO train_multi TF=ALL epoch 7/50 train=0.6904 val=0.6885
2026-05-01 07:08:44,806 INFO train_multi TF=ALL epoch 8/50 train=0.6903 val=0.6885
2026-05-01 07:08:58,659 INFO train_multi TF=ALL epoch 9/50 train=0.6897 val=0.6879
2026-05-01 07:08:58,665 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 07:08:58,665 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:08:58,665 INFO train_multi TF=ALL: new best val=0.6879 — saved
2026-05-01 07:09:12,479 INFO train_multi TF=ALL epoch 10/50 train=0.6896 val=0.6879
2026-05-01 07:09:12,484 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 07:09:12,484 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:09:12,484 INFO train_multi TF=ALL: new best val=0.6879 — saved
2026-05-01 07:09:26,487 INFO train_multi TF=ALL epoch 11/50 train=0.6891 val=0.6879
2026-05-01 07:09:40,715 INFO train_multi TF=ALL epoch 12/50 train=0.6889 val=0.6875
2026-05-01 07:09:40,720 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 07:09:40,720 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:09:40,720 INFO train_multi TF=ALL: new best val=0.6875 — saved
2026-05-01 07:09:54,700 INFO train_multi TF=ALL epoch 13/50 train=0.6884 val=0.6870
2026-05-01 07:09:54,706 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 07:09:54,706 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:09:54,706 INFO train_multi TF=ALL: new best val=0.6870 — saved
2026-05-01 07:10:08,787 INFO train_multi TF=ALL epoch 14/50 train=0.6870 val=0.6867
2026-05-01 07:10:08,793 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 07:10:08,793 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:10:08,793 INFO train_multi TF=ALL: new best val=0.6867 — saved
2026-05-01 07:10:22,887 INFO train_multi TF=ALL epoch 15/50 train=0.6852 val=0.6856
2026-05-01 07:10:22,892 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 07:10:22,892 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:10:22,892 INFO train_multi TF=ALL: new best val=0.6856 — saved
2026-05-01 07:10:37,057 INFO train_multi TF=ALL epoch 16/50 train=0.6820 val=0.6832
2026-05-01 07:10:37,062 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 07:10:37,063 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:10:37,063 INFO train_multi TF=ALL: new best val=0.6832 — saved
2026-05-01 07:10:51,518 INFO train_multi TF=ALL epoch 17/50 train=0.6745 val=0.6706
2026-05-01 07:10:51,524 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 07:10:51,524 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:10:51,524 INFO train_multi TF=ALL: new best val=0.6706 — saved
2026-05-01 07:11:05,587 INFO train_multi TF=ALL epoch 18/50 train=0.6641 val=0.6690
2026-05-01 07:11:05,593 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 07:11:05,594 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:11:05,594 INFO train_multi TF=ALL: new best val=0.6690 — saved
2026-05-01 07:11:19,784 INFO train_multi TF=ALL epoch 19/50 train=0.6496 val=0.6393
2026-05-01 07:11:19,789 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 07:11:19,789 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:11:19,789 INFO train_multi TF=ALL: new best val=0.6393 — saved
2026-05-01 07:11:33,941 INFO train_multi TF=ALL epoch 20/50 train=0.6384 val=0.6363
2026-05-01 07:11:33,946 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 07:11:33,946 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:11:33,946 INFO train_multi TF=ALL: new best val=0.6363 — saved
2026-05-01 07:11:47,939 INFO train_multi TF=ALL epoch 21/50 train=0.6313 val=0.6248
2026-05-01 07:11:47,945 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 07:11:47,945 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:11:47,945 INFO train_multi TF=ALL: new best val=0.6248 — saved
2026-05-01 07:12:01,925 INFO train_multi TF=ALL epoch 22/50 train=0.6246 val=0.6221
2026-05-01 07:12:01,930 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 07:12:01,930 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:12:01,930 INFO train_multi TF=ALL: new best val=0.6221 — saved
2026-05-01 07:12:16,043 INFO train_multi TF=ALL epoch 23/50 train=0.6206 val=0.6211
2026-05-01 07:12:16,048 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 07:12:16,048 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:12:16,048 INFO train_multi TF=ALL: new best val=0.6211 — saved
2026-05-01 07:12:30,034 INFO train_multi TF=ALL epoch 24/50 train=0.6172 val=0.6185
2026-05-01 07:12:30,039 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 07:12:30,039 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:12:30,039 INFO train_multi TF=ALL: new best val=0.6185 — saved
2026-05-01 07:12:44,153 INFO train_multi TF=ALL epoch 25/50 train=0.6140 val=0.6172
2026-05-01 07:12:44,158 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 07:12:44,158 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:12:44,158 INFO train_multi TF=ALL: new best val=0.6172 — saved
2026-05-01 07:12:57,961 INFO train_multi TF=ALL epoch 26/50 train=0.6118 val=0.6186
2026-05-01 07:13:12,137 INFO train_multi TF=ALL epoch 27/50 train=0.6095 val=0.6182
2026-05-01 07:13:26,348 INFO train_multi TF=ALL epoch 28/50 train=0.6076 val=0.6127
2026-05-01 07:13:26,353 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 07:13:26,353 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:13:26,353 INFO train_multi TF=ALL: new best val=0.6127 — saved
2026-05-01 07:13:40,478 INFO train_multi TF=ALL epoch 29/50 train=0.6059 val=0.6172
2026-05-01 07:13:54,565 INFO train_multi TF=ALL epoch 30/50 train=0.6036 val=0.6180
2026-05-01 07:14:08,605 INFO train_multi TF=ALL epoch 31/50 train=0.6021 val=0.6207
2026-05-01 07:14:22,511 INFO train_multi TF=ALL epoch 32/50 train=0.6003 val=0.6136
2026-05-01 07:14:36,683 INFO train_multi TF=ALL epoch 33/50 train=0.5992 val=0.6131
2026-05-01 07:14:50,938 INFO train_multi TF=ALL epoch 34/50 train=0.5979 val=0.6128
2026-05-01 07:15:05,339 INFO train_multi TF=ALL epoch 35/50 train=0.5961 val=0.6159
2026-05-01 07:15:19,266 INFO train_multi TF=ALL epoch 36/50 train=0.5948 val=0.6113
2026-05-01 07:15:19,272 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 07:15:19,272 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:15:19,272 INFO train_multi TF=ALL: new best val=0.6113 — saved
2026-05-01 07:15:33,497 INFO train_multi TF=ALL epoch 37/50 train=0.5933 val=0.6141
2026-05-01 07:15:47,498 INFO train_multi TF=ALL epoch 38/50 train=0.5920 val=0.6128
2026-05-01 07:16:01,528 INFO train_multi TF=ALL epoch 39/50 train=0.5908 val=0.6229
2026-05-01 07:16:15,524 INFO train_multi TF=ALL epoch 40/50 train=0.5900 val=0.6181
2026-05-01 07:16:29,675 INFO train_multi TF=ALL epoch 41/50 train=0.5887 val=0.6125
2026-05-01 07:16:43,778 INFO train_multi TF=ALL epoch 42/50 train=0.5871 val=0.6246
2026-05-01 07:16:57,901 INFO train_multi TF=ALL epoch 43/50 train=0.5860 val=0.6163
2026-05-01 07:17:11,942 INFO train_multi TF=ALL epoch 44/50 train=0.5849 val=0.6232
2026-05-01 07:17:25,934 INFO train_multi TF=ALL epoch 45/50 train=0.5843 val=0.6204
2026-05-01 07:17:40,114 INFO train_multi TF=ALL epoch 46/50 train=0.5834 val=0.6179
2026-05-01 07:17:54,048 INFO train_multi TF=ALL epoch 47/50 train=0.5819 val=0.6174
2026-05-01 07:18:08,119 INFO train_multi TF=ALL epoch 48/50 train=0.5802 val=0.6183
2026-05-01 07:18:22,276 INFO train_multi TF=ALL epoch 49/50 train=0.5794 val=0.6228
2026-05-01 07:18:36,366 INFO train_multi TF=ALL epoch 50/50 train=0.5781 val=0.6176
2026-05-01 07:18:36,524 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-05-01 07:18:36,524 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-05-01 07:18:36,524 INFO Retrain complete. Total wall-clock: 782.8s
2026-05-01 07:18:39,391 INFO Model gru: SUCCESS
2026-05-01 07:18:39,391 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:18:39,391 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-01 07:18:39,391 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-01 07:18:39,391 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-05-01 07:18:39,391 INFO   [DEFERRED] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip (expected after Round 1)
2026-05-01 07:18:39,391 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-05-01 07:18:39,392 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer', 'rl_ppo']
2026-05-01 07:18:39,392 INFO Saved 5 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Clean Quality/RL source: Backtest on train window ===
  START Train-window backtest for Quality/RL labels
2026-05-01 07:18:40,130 INFO === STEP 6: BACKTEST (train) ===
2026-05-01 07:18:40,131 INFO BT_WINDOW=train — train-window backtest: 2016-01-04 → 2021-08-05 (clean Quality/RL labels)
2026-05-01 07:18:40,131 INFO ================================================================
  ROUND 0 / 3
================================================================
2026-05-01 07:18:40,132 INFO Round 0 — running backtest: 2016-01-04 → 2021-08-05 (ml_trader, shared ML cache)
2026-05-01 07:18:42,581 WARNING QualityScorer unavailable (weights missing or load failed)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260501_071842.json
Trader                                   Trades      WR    PF*  Return*    ExpR   TP1%   TP2%     DD*  Sharpe*  Sortino     IC  Verdict
-----------------------------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)              167  18.6%   0.52  -65.1%  -0.390 18.6%  5.4%  68.7%    -4.61    -0.39 -0.108     FAIL
  FAILED rules: positive_expectancy, profit_factor_min_1_25, drawdown_below_20pct, sharpe_positive, sortino_positive, win_rate_above_breakeven, mc_p10_not_ruin, sharpe_ci_positive
  monthly R: 2020-12=-2.31  2021-02=-9.01  2021-04=-2.00  2021-05=-1.00  2021-06=-2.00  2021-07=-0.77
  MonteCarlo P95 DD=71.7%  P10 equity=3,489  t=-3.75 (p=0.000)  Sharpe CI=[-7.91, -1.95]  streak=22
  gate_diagnostics: bars=1439381 no_signal=680893 quality_block=0 session_skip=758293 density=28 pm_reject=0 daily_skip=0 cooldown=0 daily_halt_events=0 enforce_daily_halt=False
  no_signal_reasons: blocked_consolidating=586316, weak_gru_direction=84321, neutral_range_missing=3248, neutral_requires_ltf_ranging=2553, htf_bias_conflict=1386, neutral_bias_weak_conf=1274

Calibration Summary:
  all          [OK] Too few populated bins for calibration check
  ml_trader    [OK] Too few populated bins for calibration check
2026-05-01 07:27:42,344 INFO Round 0 backtest — 167 trades | avg WR=18.6% | avg PF=0.52 | avg Sharpe=-4.61
2026-05-01 07:27:42,344 INFO   ml_trader: 167 trades | WR=18.6% | fixed PF=0.52 | Return=-65.1% | ExpR=-0.390 | DD=68.7% | Sharpe=-4.61
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_0.json
INFO  Total trades: 167
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (167 rows)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  AUDUSD         12 trades     8 days   1.50/day
  EURGBP         10 trades     6 days   1.67/day  [OVERTRADE]
  EURJPY         18 trades    13 days   1.39/day
  EURUSD         11 trades     6 days   1.83/day  [OVERTRADE]
  GBPJPY         18 trades     9 days   2.00/day  [OVERTRADE]
  GBPUSD         28 trades    13 days   2.15/day  [OVERTRADE]
  NZDUSD          8 trades     5 days   1.60/day  [OVERTRADE]
  USDCAD         12 trades     8 days   1.50/day
  USDCHF         16 trades    11 days   1.46/day
  USDJPY         11 trades     8 days   1.38/day
  XAUUSD         23 trad  DONE  Train-window backtest for Quality/RL labels
  Saved Train Quality/RL source result → train_quality_rl_source_summary.json
  Train-label journal entries: 167

=== Train Quality + RL on train-only journal ===
  START Train-only Quality+RL retrain
2026-05-01 07:27:42,670 INFO Round 0: wrote 167 journal entries (total in file: 167)
2026-05-01 07:27:42,979 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-05-01 07:27:42,984 INFO Journal entries: 167 total, 167 allowed for training (['live', 'paper', 'production', 'train'])
2026-05-01 07:27:42,984 INFO --- Training quality ---
2026-05-01 07:27:42,984 INFO Running retrain --model quality with JOURNAL_ALLOWED_SPLITS=train,live,paper,production
2026-05-01 07:27:43,178 INFO retrain environment: KAGGLE
2026-05-01 07:27:44,859 INFO Device: CUDA (2 GPU(s))
2026-05-01 07:27:44,871 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 07:27:44,871 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 07:27:44,871 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-01 07:27:44,872 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-01 07:27:44,872 INFO Retrain data split: train
2026-05-01 07:27:44,873 INFO === QualityScorer retrain ===
2026-05-01 07:27:45,024 INFO NumExpr defaulting to 4 threads.
2026-05-01 07:27:45,227 INFO QualityScorer: CUDA available — using GPU
2026-05-01 07:27:45,239 INFO Quality phase label creation: 0.0s (167 trades)
2026-05-01 07:27:45,255 INFO QualityScorer: 167 samples, EV stats={'mean': -0.5487784147262573, 'std': 0.9995700120925903, 'n_pos': 31, 'n_neg': 136}, device=cuda
2026-05-01 07:27:45,256 INFO QualityScorer: normalised win labels by median_win=1.000 — EV range now [-1, +3]
2026-05-01 07:27:45,463 INFO QualityScorer: DataParallel across 2 GPUs
2026-05-01 07:27:45,464 INFO QualityScorer: cold start
2026-05-01 07:27:45,464 INFO QualityScorer: pos_weight=4.12 (n_pos=26 n_neg=107)
2026-05-01 07:27:47,865 INFO Quality epoch   1/100 — va_huber=1.0829
2026-05-01 07:27:47,906 INFO Quality epoch   2/100 — va_huber=1.0707
2026-05-01 07:27:47,927 INFO Quality epoch   3/100 — va_huber=1.0669
2026-05-01 07:27:47,948 INFO Quality epoch   4/100 — va_huber=1.0633
2026-05-01 07:27:47,969 INFO Quality epoch   5/100 — va_huber=1.0616
2026-05-01 07:27:48,094 INFO Quality epoch  11/100 — va_huber=1.0607
2026-05-01 07:27:48,252 INFO Quality early stop at epoch 19
2026-05-01 07:27:48,260 INFO QualityScorer EV model: MAE=0.999 dir_acc=0.853 n_val=34
2026-05-01 07:27:48,265 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-05-01 07:27:48,309 INFO Quality phase train: 3.1s | total: 3.4s
2026-05-01 07:27:48,315 INFO Retrain complete. Total wall-clock: 3.4s
2026-05-01 07:27:49,373 INFO Model quality: SUCCESS
2026-05-01 07:27:49,373 INFO --- Training rl ---
2026-05-01 07:27:49,373 INFO Running retrain --model rl with JOURNAL_ALLOWED_SPLITS=train,live,paper,production
2026-05-01 07:27:49,562 INFO retrain environment: KAGGLE
2026-05-01 07:27:51,275 INFO Device: CUDA (2 GPU(s))
2026-05-01 07:27:51,287 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 07:27:51,287 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 07:27:51,287 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-01 07:27:51,287 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-01 07:27:51,287 INFO Retrain data split: train
2026-05-01 07:27:51,288 INFO === RLAgent (PPO) retrain ===
2026-05-01 07:27:51,295 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260501_072751
2026-05-01 07:27:51,301 INFO RL phase episode loading: 0.0s (167 episodes)
2026-05-01 07:27:54.977869: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777620475.226869   55340 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777620475.302929   55340 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777620475.876154   55340 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777620475.876200   55340 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777620475.876203   55340 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777620475.876206   55340 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-05-01 07:28:10,941 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-05-01 07:28:14,411 INFO RLAgent: cold start — building new PPO policy
2026-05-01 07:28:18,305 INFO RLAgent: retrain complete, 167 episodes
2026-05-01 07:28:18,306 INFO RL phase PPO train: 27.0s | total: 27.0s
2026-05-01 07:28:18,313 INFO Retrain complete. Total wall-clock: 27.0s
2026-05-01 07:28:20,111 INFO Model rl: SUCCESS
2026-05-01 07:28:20,112 INFO Step 7b complete — summary: /kaggle/working/Multi-Bot/trading-system/ml_training/metrics/training_7b_summary.json
  DONE  Train-only Quality+RL retrain
  Archived journal → trade_journal_train_only.jsonl
  Archived journal CSV → trade_journal_train_only.csv

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  Cleared journal for fresh Round 1 run
  START Round 1 - Backtest (val)
2026-05-01 07:28:20,653 INFO === STEP 6: BACKTEST (round1) ===
2026-05-01 07:28:20,654 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-05-01 07:28:20,654 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-05-01 07:28:20,654 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
2026-05-01 07:31:30,596 INFO Round 1 backtest — 0 trades | avg WR=0.0% | avg PF=0.00 | avg Sharpe=0.00
2026-05-01 07:31:30,596 INFO   ml_trader: 0 trades | WR=0.0% | fixed PF=0.00 | Return=0.0% | ExpR=0.000 | DD=0.0% | Sharpe=0.00
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 0
ERROR  trade_log is empty.
2026-05-01 07:31:30,813 WARNING Round 1: trade_log is empty — nothing to journal
2026-05-01 07:31:30,813 WARNING Round 1: no trades to journal
  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 0 entries

  SKIP  Round 1 Quality+RL retrain — validation journal kept evaluation-only

=== Round 2: BLIND backtest on test window (unseen 2yr) ===
  START Round 2 - Blind backtest (test)
2026-05-01 07:31:31,538 INFO === STEP 6: BACKTEST (round2) ===
2026-05-01 07:31:31,539 INFO BT_WINDOW=round2 — BLIND backtest: 2023-08-07 → 2025-08-05 (test set)
2026-05-01 07:31:31,539 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-05-01 07:31:31,539 INFO Round 2 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
2026-05-01 07:34:45,239 INFO Round 2 backtest — 1 trades | avg WR=0.0% | avg PF=0.00 | avg Sharpe=0.00
2026-05-01 07:34:45,239 INFO   ml_trader: 1 trades | WR=0.0% | fixed PF=0.00 | Return=-1.0% | ExpR=-1.001 | DD=1.0% | Sharpe=0.00
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 1
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (1 rows)
2026-05-01 07:34:45,457 INFO Round 2: wrote 1 journal entries (total in file: 1)
  DONE  Round 2 - Blind backtest (test)
  Saved Round 2 result → round2_summary.json
  Journal after Round 2: 1 entries

  SKIP  Round 2 Quality+RL retrain — blind test journal kept untouched

=== Round 3: Incremental retrain ===
  START Retrain gru [train-split retrain]
2026-05-01 07:34:45,776 INFO retrain environment: KAGGLE
2026-05-01 07:34:47,496 INFO Device: CUDA (2 GPU(s))
2026-05-01 07:34:47,507 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 07:34:47,507 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 07:34:47,507 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-01 07:34:47,508 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-01 07:34:47,508 INFO Retrain data split: train
2026-05-01 07:34:47,509 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-05-01 07:34:47,686 INFO NumExpr defaulting to 4 threads.
2026-05-01 07:34:47,895 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-01 07:34:47,895 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 07:34:47,895 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 07:34:48,149 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-05-01 07:34:48,149 INFO GRU phase macro_correlations: 0.0s
2026-05-01 07:34:48,149 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-05-01 07:34:48,151 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260501_073448
2026-05-01 07:34:48,155 INFO GRU feature contract unchanged (input_size=78) — incremental retrain
2026-05-01 07:34:48,309 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:48,330 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:48,344 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:48,352 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:48,353 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-01 07:34:48,353 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 07:34:48,353 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 07:34:48,354 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-05-01 07:34:48,355 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:48,444 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 17, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 8360}  ambiguous=8312 (total=8402)  short_runs_zeroed=15
2026-05-01 07:34:48,446 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:48,697 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 619, 'RANGING': 24879, 'CONSOLIDATING': 1882, 'VOLATILE': 5358}  ambiguous=26508 (total=32738)  short_runs_zeroed=2571
2026-05-01 07:34:48,729 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:49,008 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:49,145 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:49,254 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:49,462 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:49,480 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:49,494 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:49,501 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:49,502 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:49,586 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 100, 'BIAS_DOWN': 31, 'BIAS_NEUTRAL': 8271}  ambiguous=8272 (total=8402)  short_runs_zeroed=52
2026-05-01 07:34:49,588 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:49,848 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 752, 'RANGING': 24932, 'CONSOLIDATING': 2099, 'VOLATILE': 4955}  ambiguous=26224 (total=32738)  short_runs_zeroed=2112
2026-05-01 07:34:49,864 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:50,149 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:50,285 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:50,385 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:50,583 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:50,603 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:50,617 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:50,624 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:50,625 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:50,711 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 46, 'BIAS_DOWN': 4, 'BIAS_NEUTRAL': 8352}  ambiguous=8316 (total=8402)  short_runs_zeroed=28
2026-05-01 07:34:50,713 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:50,966 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 617, 'RANGING': 24738, 'CONSOLIDATING': 2103, 'VOLATILE': 5282}  ambiguous=26101 (total=32740)  short_runs_zeroed=2218
2026-05-01 07:34:50,982 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:51,258 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:51,390 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:51,491 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:51,691 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:51,711 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:51,725 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:51,732 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:51,733 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:51,824 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 15, 'BIAS_DOWN': 19, 'BIAS_NEUTRAL': 8368}  ambiguous=8331 (total=8402)  short_runs_zeroed=27
2026-05-01 07:34:51,825 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:52,084 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 842, 'RANGING': 25174, 'CONSOLIDATING': 1996, 'VOLATILE': 4727}  ambiguous=26490 (total=32739)  short_runs_zeroed=2045
2026-05-01 07:34:52,108 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:52,385 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:52,520 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:52,623 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:52,815 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:52,837 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:52,853 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:52,861 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:52,862 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:52,948 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 93, 'BIAS_DOWN': 58, 'BIAS_NEUTRAL': 8252}  ambiguous=8287 (total=8403)  short_runs_zeroed=75
2026-05-01 07:34:52,950 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:53,202 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 761, 'RANGING': 24579, 'CONSOLIDATING': 2089, 'VOLATILE': 5311}  ambiguous=25954 (total=32740)  short_runs_zeroed=2172
2026-05-01 07:34:53,218 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:53,500 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:53,635 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:53,738 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:53,935 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:53,956 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:53,971 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:53,978 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:53,979 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:54,071 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 87, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 8261}  ambiguous=8278 (total=8403)  short_runs_zeroed=63
2026-05-01 07:34:54,072 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:54,324 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 763, 'RANGING': 25215, 'CONSOLIDATING': 1968, 'VOLATILE': 4793}  ambiguous=26501 (total=32739)  short_runs_zeroed=1959
2026-05-01 07:34:54,341 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:54,611 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:54,742 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:54,840 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:55,018 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:34:55,036 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:34:55,051 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:34:55,057 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:34:55,058 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:55,154 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 21, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 8314}  ambiguous=8303 (total=8402)  short_runs_zeroed=39
2026-05-01 07:34:55,156 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:55,422 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 474, 'RANGING': 25095, 'CONSOLIDATING': 1967, 'VOLATILE': 5203}  ambiguous=26572 (total=32739)  short_runs_zeroed=2307
2026-05-01 07:34:55,436 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:55,706 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:55,834 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:55,940 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:56,129 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:56,149 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:56,163 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:56,170 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:56,171 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:56,254 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 37, 'BIAS_DOWN': 38, 'BIAS_NEUTRAL': 8327}  ambiguous=8321 (total=8402)  short_runs_zeroed=44
2026-05-01 07:34:56,256 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:56,496 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 881, 'RANGING': 25133, 'CONSOLIDATING': 1987, 'VOLATILE': 4739}  ambiguous=26356 (total=32740)  short_runs_zeroed=1985
2026-05-01 07:34:56,514 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:56,790 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:56,929 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:57,032 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:57,232 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:57,252 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:57,269 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:57,279 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:57,280 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:57,367 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 79, 'BIAS_DOWN': 33, 'BIAS_NEUTRAL': 8290}  ambiguous=8328 (total=8402)  short_runs_zeroed=61
2026-05-01 07:34:57,368 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:57,636 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 699, 'RANGING': 24989, 'CONSOLIDATING': 2130, 'VOLATILE': 4923}  ambiguous=26228 (total=32741)  short_runs_zeroed=2004
2026-05-01 07:34:57,656 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:57,943 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:58,083 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:58,188 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:58,380 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:58,401 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:58,416 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:58,423 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:34:58,424 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:58,511 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 42, 'BIAS_DOWN': 44, 'BIAS_NEUTRAL': 8317}  ambiguous=8318 (total=8403)  short_runs_zeroed=44
2026-05-01 07:34:58,513 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:58,763 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 808, 'RANGING': 24290, 'CONSOLIDATING': 2180, 'VOLATILE': 5465}  ambiguous=25645 (total=32743)  short_runs_zeroed=2211
2026-05-01 07:34:58,780 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:59,056 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:59,194 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:59,295 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:34:59,585 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:34:59,611 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:34:59,627 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:34:59,637 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:34:59,639 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:34:59,805 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 107, 'BIAS_DOWN': 115, 'BIAS_NEUTRAL': 19595}  ambiguous=19636 (total=19817)  short_runs_zeroed=112
2026-05-01 07:34:59,808 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:35:00,361 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 1778, 'RANGING': 57285, 'CONSOLIDATING': 4542, 'VOLATILE': 11019}  ambiguous=60314 (total=74624)  short_runs_zeroed=4774
2026-05-01 07:35:00,407 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:35:00,952 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:35:01,157 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:35:01,293 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:35:01,412 INFO train_multi: 44 segments, ~6069276 total bars
2026-05-01 07:35:01,412 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-05-01 07:35:01,412 INFO train_multi: building combined dataset for TF=ALL (44 segments)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
2026-05-01 07:35:56,135 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-05-01 07:35:56,136 INFO train_multi TF=ALL: estimated peak RAM = 11232 MB (train=479966 val=120011 n_feat=78 seq_len=30)
2026-05-01 07:35:57,537 INFO train_multi TF=ALL: train=479966 val=120011 (5623 MB tensors)
2026-05-01 07:36:04,273 INFO train_multi TF=ALL: warm-start detected — using CosineAnnealingLR (lr=3e-05, patience=12)
2026-05-01 07:36:20,319 INFO train_multi TF=ALL epoch 1/50 train=0.5922 val=0.6140
2026-05-01 07:36:20,325 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 07:36:20,325 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:36:20,325 INFO train_multi TF=ALL: new best val=0.6140 — saved
2026-05-01 07:36:34,401 INFO train_multi TF=ALL epoch 2/50 train=0.5918 val=0.6137
2026-05-01 07:36:34,406 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 07:36:34,406 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:36:34,406 INFO train_multi TF=ALL: new best val=0.6137 — saved
2026-05-01 07:36:48,605 INFO train_multi TF=ALL epoch 3/50 train=0.5915 val=0.6136
2026-05-01 07:36:48,610 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 07:36:48,610 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 07:36:48,610 INFO train_multi TF=ALL: new best val=0.6136 — saved
2026-05-01 07:37:02,457 INFO train_multi TF=ALL epoch 4/50 train=0.5914 val=0.6140
2026-05-01 07:37:16,430 INFO train_multi TF=ALL epoch 5/50 train=0.5912 val=0.6140
2026-05-01 07:37:30,462 INFO train_multi TF=ALL epoch 6/50 train=0.5909 val=0.6148
2026-05-01 07:37:44,349 INFO train_multi TF=ALL epoch 7/50 train=0.5908 val=0.6147
2026-05-01 07:37:58,313 INFO train_multi TF=ALL epoch 8/50 train=0.5907 val=0.6151
2026-05-01 07:38:12,413 INFO train_multi TF=ALL epoch 9/50 train=0.5906 val=0.6151
2026-05-01 07:38:26,464 INFO train_multi TF=ALL epoch 10/50 train=0.5904 val=0.6143
2026-05-01 07:38:40,565 INFO train_multi TF=ALL epoch 11/50 train=0.5902 val=0.6144
2026-05-01 07:38:54,434 INFO train_multi TF=ALL epoch 12/50 train=0.5898 val=0.6161
2026-05-01 07:39:08,450 INFO train_multi TF=ALL epoch 13/50 train=0.5897 val=0.6154
2026-05-01 07:39:22,414 INFO train_multi TF=ALL epoch 14/50 train=0.5894 val=0.6149
2026-05-01 07:39:36,704 INFO train_multi TF=ALL epoch 15/50 train=0.5894 val=0.6145
2026-05-01 07:39:36,704 INFO train_multi TF=ALL early stop at epoch 15
2026-05-01 07:39:36,865 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-05-01 07:39:36,865 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-05-01 07:39:36,865 INFO Retrain complete. Total wall-clock: 289.4s
  DONE  Retrain gru [train-split retrain]
  START Retrain regime [train-split retrain]
2026-05-01 07:39:40,030 INFO retrain environment: KAGGLE
2026-05-01 07:39:41,688 INFO Device: CUDA (2 GPU(s))
2026-05-01 07:39:41,697 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 07:39:41,697 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 07:39:41,697 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-01 07:39:41,698 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-01 07:39:41,698 INFO Retrain data split: train
2026-05-01 07:39:41,699 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-05-01 07:39:41,857 INFO NumExpr defaulting to 4 threads.
2026-05-01 07:39:42,057 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-01 07:39:42,057 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 07:39:42,057 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 07:39:42,058 INFO Regime phase macro_correlations: 0.0s
2026-05-01 07:39:42,058 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-05-01 07:39:42,096 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-05-01 07:39:42,097 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:39:42,127 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:39:42,142 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:39:42,167 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:39:42,185 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:39:42,211 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:39:42,225 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:39:42,249 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:39:42,265 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:39:42,288 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:39:42,303 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:39:42,325 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:39:42,339 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:39:42,358 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:39:42,373 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:39:42,396 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:39:42,411 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:39:42,434 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:39:42,449 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:39:42,472 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:39:42,490 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:39:42,531 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:39:43,301 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-05-01 07:40:08,067 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-05-01 07:40:08,069 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 25.5s
2026-05-01 07:40:08,072 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-05-01 07:40:18,928 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-05-01 07:40:18,932 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 10.9s
2026-05-01 07:40:18,932 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-05-01 07:40:26,784 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-05-01 07:40:26,785 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 7.9s
2026-05-01 07:40:26,785 INFO Regime phase GMM HTF total: 44.3s
2026-05-01 07:40:26,786 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-05-01 07:41:42,646 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-05-01 07:41:42,650 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 75.9s
2026-05-01 07:41:42,650 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-05-01 07:42:16,792 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-05-01 07:42:16,796 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 34.1s
2026-05-01 07:42:16,797 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-05-01 07:42:40,647 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-05-01 07:42:40,648 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 23.9s
2026-05-01 07:42:40,648 INFO Regime phase GMM LTF total: 133.9s
2026-05-01 07:42:40,759 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-05-01 07:42:40,760 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:40,762 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:40,763 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:40,764 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:40,765 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:40,766 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:40,767 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:40,768 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:40,770 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:40,771 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:40,772 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:42:40,901 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:40,945 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:40,946 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:40,947 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:40,955 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:40,956 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:41,393 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 17, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 8360}  ambiguous=8312 (total=8402)  short_runs_zeroed=15
2026-05-01 07:42:41,394 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-05-01 07:42:41,579 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:41,615 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:41,616 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:41,616 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:41,625 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:41,626 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:42,032 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 100, 'BIAS_DOWN': 31, 'BIAS_NEUTRAL': 8271}  ambiguous=8272 (total=8402)  short_runs_zeroed=52
2026-05-01 07:42:42,033 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-05-01 07:42:42,236 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:42,273 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:42,274 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:42,274 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:42,283 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:42,284 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:42,679 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 46, 'BIAS_DOWN': 4, 'BIAS_NEUTRAL': 8352}  ambiguous=8316 (total=8402)  short_runs_zeroed=28
2026-05-01 07:42:42,680 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-05-01 07:42:42,859 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:42,898 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:42,899 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:42,899 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:42,908 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:42,909 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:43,313 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 15, 'BIAS_DOWN': 19, 'BIAS_NEUTRAL': 8368}  ambiguous=8331 (total=8402)  short_runs_zeroed=27
2026-05-01 07:42:43,314 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-05-01 07:42:43,510 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:43,547 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:43,548 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:43,548 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:43,557 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:43,558 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:43,960 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 93, 'BIAS_DOWN': 58, 'BIAS_NEUTRAL': 8252}  ambiguous=8287 (total=8403)  short_runs_zeroed=75
2026-05-01 07:42:43,961 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-05-01 07:42:44,139 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:44,175 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:44,176 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:44,177 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:44,185 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:44,186 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:44,581 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 87, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 8261}  ambiguous=8278 (total=8403)  short_runs_zeroed=63
2026-05-01 07:42:44,582 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-05-01 07:42:44,743 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:42:44,771 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:42:44,772 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:42:44,772 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:42:44,781 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:42:44,782 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:45,188 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 21, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 8314}  ambiguous=8303 (total=8402)  short_runs_zeroed=39
2026-05-01 07:42:45,189 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-05-01 07:42:45,362 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:45,399 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:45,400 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:45,400 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:45,409 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:45,410 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:45,810 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 37, 'BIAS_DOWN': 38, 'BIAS_NEUTRAL': 8327}  ambiguous=8321 (total=8402)  short_runs_zeroed=44
2026-05-01 07:42:45,812 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-05-01 07:42:45,993 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:46,028 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:46,029 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:46,030 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:46,039 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:46,040 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:46,442 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 79, 'BIAS_DOWN': 33, 'BIAS_NEUTRAL': 8290}  ambiguous=8328 (total=8402)  short_runs_zeroed=61
2026-05-01 07:42:46,443 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-05-01 07:42:46,623 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:46,660 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:46,661 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:46,661 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:46,670 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:46,671 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:47,080 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 42, 'BIAS_DOWN': 44, 'BIAS_NEUTRAL': 8317}  ambiguous=8318 (total=8403)  short_runs_zeroed=44
2026-05-01 07:42:47,082 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-05-01 07:42:47,368 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:42:47,431 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:42:47,432 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:42:47,433 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:42:47,444 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:42:47,445 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:42:48,312 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 107, 'BIAS_DOWN': 115, 'BIAS_NEUTRAL': 19595}  ambiguous=19636 (total=19817)  short_runs_zeroed=112
2026-05-01 07:42:48,314 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-05-01 07:42:48,481 INFO Regime phase HTF dataset build: 7.7s (103290 samples)
2026-05-01 07:42:48,482 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260501_074248
2026-05-01 07:42:48,685 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=34, n_classes=3)
2026-05-01 07:42:48,686 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=1138 dropped=102152 classes={'BIAS_UP': 399, 'BIAS_DOWN': 243, 'BIAS_NEUTRAL': 496})
2026-05-01 07:42:48,687 INFO RegimeClassifier[mode=htf_bias]: 1138 samples, classes={'BIAS_UP': 399, 'BIAS_DOWN': 243, 'BIAS_NEUTRAL': 496}, device=cuda
2026-05-01 07:42:48,687 INFO RegimeClassifier: sample weights — mean=0.713  ambiguous(<0.4)=0.0%
2026-05-01 07:42:48,687 INFO RegimeClassifier[mode=htf_bias]: warm start from existing weights
2026-05-01 07:42:48,687 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-05-01 07:42:50,939 INFO Regime epoch  1/50 — tr=0.5347 va=1.2739 acc=0.934 per_class={'BIAS_UP': 0.865, 'BIAS_DOWN': 0.948, 'BIAS_NEUTRAL': 0.987}
2026-05-01 07:42:50,955 INFO Regime epoch  2/50 — tr=0.5359 va=1.2691 acc=0.939
2026-05-01 07:42:50,969 INFO Regime epoch  3/50 — tr=0.5358 va=1.2641 acc=0.939
2026-05-01 07:42:50,983 INFO Regime epoch  4/50 — tr=0.5285 va=1.2604 acc=0.939
2026-05-01 07:42:50,999 INFO Regime epoch  5/50 — tr=0.5364 va=1.2553 acc=0.939 per_class={'BIAS_UP': 0.878, 'BIAS_DOWN': 0.948, 'BIAS_NEUTRAL': 0.987}
2026-05-01 07:42:51,013 INFO Regime epoch  6/50 — tr=0.5357 va=1.2489 acc=0.939
2026-05-01 07:42:51,026 INFO Regime epoch  7/50 — tr=0.5336 va=1.2436 acc=0.943
2026-05-01 07:42:51,041 INFO Regime epoch  8/50 — tr=0.5362 va=1.2349 acc=0.943
2026-05-01 07:42:51,053 INFO Regime epoch  9/50 — tr=0.5329 va=1.2255 acc=0.947
2026-05-01 07:42:51,070 INFO Regime epoch 10/50 — tr=0.5340 va=1.2161 acc=0.943 per_class={'BIAS_UP': 0.878, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 0.987}
2026-05-01 07:42:51,083 INFO Regime epoch 11/50 — tr=0.5364 va=1.2053 acc=0.939
2026-05-01 07:42:51,096 INFO Regime epoch 12/50 — tr=0.5322 va=1.1956 acc=0.939
2026-05-01 07:42:51,110 INFO Regime epoch 13/50 — tr=0.5264 va=1.1857 acc=0.947
2026-05-01 07:42:51,123 INFO Regime epoch 14/50 — tr=0.5311 va=1.1779 acc=0.947
2026-05-01 07:42:51,140 INFO Regime epoch 15/50 — tr=0.5312 va=1.1689 acc=0.947 per_class={'BIAS_UP': 0.892, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 0.987}
2026-05-01 07:42:51,154 INFO Regime epoch 16/50 — tr=0.5323 va=1.1599 acc=0.947
2026-05-01 07:42:51,168 INFO Regime epoch 17/50 — tr=0.5287 va=1.1496 acc=0.956
2026-05-01 07:42:51,182 INFO Regime epoch 18/50 — tr=0.5229 va=1.1409 acc=0.952
2026-05-01 07:42:51,195 INFO Regime epoch 19/50 — tr=0.5242 va=1.1317 acc=0.956
2026-05-01 07:42:51,212 INFO Regime epoch 20/50 — tr=0.5286 va=1.1237 acc=0.956 per_class={'BIAS_UP': 0.905, 'BIAS_DOWN': 0.974, 'BIAS_NEUTRAL': 0.987}
2026-05-01 07:42:51,225 INFO Regime epoch 21/50 — tr=0.5249 va=1.1142 acc=0.961
2026-05-01 07:42:51,239 INFO Regime epoch 22/50 — tr=0.5263 va=1.1040 acc=0.956
2026-05-01 07:42:51,251 INFO Regime epoch 23/50 — tr=0.5237 va=1.0957 acc=0.956
2026-05-01 07:42:51,264 INFO Regime epoch 24/50 — tr=0.5237 va=1.0877 acc=0.956
2026-05-01 07:42:51,279 INFO Regime epoch 25/50 — tr=0.5228 va=1.0816 acc=0.956 per_class={'BIAS_UP': 0.905, 'BIAS_DOWN': 0.974, 'BIAS_NEUTRAL': 0.987}
2026-05-01 07:42:51,291 INFO Regime epoch 26/50 — tr=0.5169 va=1.0750 acc=0.956
2026-05-01 07:42:51,305 INFO Regime epoch 27/50 — tr=0.5180 va=1.0684 acc=0.956
2026-05-01 07:42:51,318 INFO Regime epoch 28/50 — tr=0.5178 va=1.0616 acc=0.956
2026-05-01 07:42:51,331 INFO Regime epoch 29/50 — tr=0.5215 va=1.0563 acc=0.956
2026-05-01 07:42:51,348 INFO Regime epoch 30/50 — tr=0.5182 va=1.0504 acc=0.956 per_class={'BIAS_UP': 0.905, 'BIAS_DOWN': 0.974, 'BIAS_NEUTRAL': 0.987}
2026-05-01 07:42:51,362 INFO Regime epoch 31/50 — tr=0.5213 va=1.0469 acc=0.961
2026-05-01 07:42:51,376 INFO Regime epoch 32/50 — tr=0.5199 va=1.0422 acc=0.961
2026-05-01 07:42:51,389 INFO Regime epoch 33/50 — tr=0.5237 va=1.0357 acc=0.961
2026-05-01 07:42:51,402 INFO Regime epoch 34/50 — tr=0.5195 va=1.0319 acc=0.961
2026-05-01 07:42:51,419 INFO Regime epoch 35/50 — tr=0.5216 va=1.0274 acc=0.961 per_class={'BIAS_UP': 0.905, 'BIAS_DOWN': 0.987, 'BIAS_NEUTRAL': 0.987}
2026-05-01 07:42:51,433 INFO Regime epoch 36/50 — tr=0.5196 va=1.0224 acc=0.961
2026-05-01 07:42:51,446 INFO Regime epoch 37/50 — tr=0.5188 va=1.0192 acc=0.961
2026-05-01 07:42:51,460 INFO Regime epoch 38/50 — tr=0.5200 va=1.0159 acc=0.961
2026-05-01 07:42:51,473 INFO Regime epoch 39/50 — tr=0.5165 va=1.0113 acc=0.961
2026-05-01 07:42:51,490 INFO Regime epoch 40/50 — tr=0.5139 va=1.0106 acc=0.961 per_class={'BIAS_UP': 0.905, 'BIAS_DOWN': 0.987, 'BIAS_NEUTRAL': 0.987}
2026-05-01 07:42:51,502 INFO Regime epoch 41/50 — tr=0.5214 va=1.0112 acc=0.961
2026-05-01 07:42:51,515 INFO Regime epoch 42/50 — tr=0.5190 va=1.0116 acc=0.961
2026-05-01 07:42:51,527 INFO Regime epoch 43/50 — tr=0.5201 va=1.0101 acc=0.961
2026-05-01 07:42:51,540 INFO Regime epoch 44/50 — tr=0.5110 va=1.0086 acc=0.965
2026-05-01 07:42:51,557 INFO Regime epoch 45/50 — tr=0.5178 va=1.0044 acc=0.961 per_class={'BIAS_UP': 0.905, 'BIAS_DOWN': 0.987, 'BIAS_NEUTRAL': 0.987}
2026-05-01 07:42:51,571 INFO Regime epoch 46/50 — tr=0.5185 va=1.0053 acc=0.961
2026-05-01 07:42:51,583 INFO Regime epoch 47/50 — tr=0.5177 va=1.0052 acc=0.965
2026-05-01 07:42:51,594 INFO Regime epoch 48/50 — tr=0.5163 va=1.0034 acc=0.965
2026-05-01 07:42:51,607 INFO Regime epoch 49/50 — tr=0.5142 va=1.0037 acc=0.969
2026-05-01 07:42:51,623 INFO Regime epoch 50/50 — tr=0.5205 va=1.0052 acc=0.969 per_class={'BIAS_UP': 0.932, 'BIAS_DOWN': 0.987, 'BIAS_NEUTRAL': 0.987}
2026-05-01 07:42:51,630 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-01 07:42:51,630 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-01 07:42:51,631 INFO Regime phase HTF train: 2.9s
2026-05-01 07:42:51,768 INFO Regime HTF complete: acc=0.965, n=103290 per_class={'BIAS_UP': 0.919, 'BIAS_DOWN': 0.987, 'BIAS_NEUTRAL': 0.987}
2026-05-01 07:42:51,770 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:42:51,943 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 107, 'BIAS_DOWN': 115, 'BIAS_NEUTRAL': 19595}  ambiguous=19636 (total=19817)  short_runs_zeroed=112
2026-05-01 07:42:51,946 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 4.28, 'BIAS_DOWN': 4.791666666666667, 'BIAS_NEUTRAL': 391.9}
2026-05-01 07:42:51,949 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 107, 'mean': -0.00021706688424743013, 'mean_over_std': -0.04806767606653151}, 'BIAS_DOWN': {'n': 115, 'mean': -0.00020797041876048362, 'mean_over_std': -0.029260022054973262}, 'BIAS_NEUTRAL': {'n': 19594, 'mean': 4.4550533138328785e-05, 'mean_over_std': 0.011372683463534268}}
2026-05-01 07:42:51,950 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 66, 'mean': 0.0003288215669218909, 'mean_over_std': 0.07277335807221066}, 'BIAS_DOWN': {'n': 59, 'mean': -0.0010984382215802496, 'mean_over_std': -0.13394112338746375}, 'BIAS_NEUTRAL': {'n': 56, 'mean': -0.00020920056804862467, 'mean_over_std': -0.06887192756862072}}
2026-05-01 07:42:51,953 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-05-01 07:42:51,955 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:51,957 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:51,958 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:51,960 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:51,962 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:51,964 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:51,965 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:51,967 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:51,969 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:51,971 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:51,974 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:42:51,987 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:51,990 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:51,991 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:51,991 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:51,991 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:51,993 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:52,672 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 619, 'RANGING': 24879, 'CONSOLIDATING': 1882, 'VOLATILE': 5358}  ambiguous=26508 (total=32738)  short_runs_zeroed=2571
2026-05-01 07:42:52,675 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-05-01 07:42:52,816 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:52,818 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:52,819 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:52,820 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:52,820 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:52,822 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:53,483 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 752, 'RANGING': 24932, 'CONSOLIDATING': 2099, 'VOLATILE': 4955}  ambiguous=26224 (total=32738)  short_runs_zeroed=2112
2026-05-01 07:42:53,486 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-05-01 07:42:53,629 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:53,631 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:53,632 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:53,633 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:53,633 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:53,635 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:54,278 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 617, 'RANGING': 24738, 'CONSOLIDATING': 2103, 'VOLATILE': 5282}  ambiguous=26101 (total=32740)  short_runs_zeroed=2218
2026-05-01 07:42:54,281 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-05-01 07:42:54,425 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:54,428 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:54,428 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:54,429 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:54,429 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:54,431 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:55,083 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 842, 'RANGING': 25174, 'CONSOLIDATING': 1996, 'VOLATILE': 4727}  ambiguous=26490 (total=32739)  short_runs_zeroed=2045
2026-05-01 07:42:55,086 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-05-01 07:42:55,237 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:55,239 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:55,240 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:55,241 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:55,241 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:55,243 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:55,881 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 761, 'RANGING': 24579, 'CONSOLIDATING': 2089, 'VOLATILE': 5311}  ambiguous=25954 (total=32740)  short_runs_zeroed=2172
2026-05-01 07:42:55,884 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-05-01 07:42:56,026 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:56,028 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:56,029 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:56,030 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:56,030 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:56,032 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:56,663 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 763, 'RANGING': 25215, 'CONSOLIDATING': 1968, 'VOLATILE': 4793}  ambiguous=26501 (total=32739)  short_runs_zeroed=1959
2026-05-01 07:42:56,667 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-05-01 07:42:56,810 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:42:56,812 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:42:56,813 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:42:56,813 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:42:56,813 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-01 07:42:56,815 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:57,450 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 474, 'RANGING': 25095, 'CONSOLIDATING': 1967, 'VOLATILE': 5203}  ambiguous=26572 (total=32739)  short_runs_zeroed=2307
2026-05-01 07:42:57,453 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-05-01 07:42:57,596 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:57,599 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:57,599 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:57,600 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:57,600 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:57,603 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:58,250 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 881, 'RANGING': 25133, 'CONSOLIDATING': 1987, 'VOLATILE': 4739}  ambiguous=26356 (total=32740)  short_runs_zeroed=1985
2026-05-01 07:42:58,253 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-05-01 07:42:58,398 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:58,400 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:58,401 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:58,402 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:58,402 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:58,404 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:59,028 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 699, 'RANGING': 24989, 'CONSOLIDATING': 2130, 'VOLATILE': 4923}  ambiguous=26228 (total=32741)  short_runs_zeroed=2004
2026-05-01 07:42:59,031 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-05-01 07:42:59,174 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:59,176 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:59,177 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:59,178 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:59,178 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 07:42:59,180 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-05-01 07:42:59,814 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 808, 'RANGING': 24290, 'CONSOLIDATING': 2180, 'VOLATILE': 5465}  ambiguous=25645 (total=32743)  short_runs_zeroed=2211
2026-05-01 07:42:59,817 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-05-01 07:42:59,967 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:42:59,971 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:42:59,972 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:42:59,973 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:42:59,973 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-01 07:42:59,976 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:43:01,335 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 1778, 'RANGING': 57285, 'CONSOLIDATING': 4542, 'VOLATILE': 11019}  ambiguous=60314 (total=74624)  short_runs_zeroed=4774
2026-05-01 07:43:01,341 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-05-01 07:43:01,649 INFO Regime phase LTF dataset build: 9.7s (401471 samples)
2026-05-01 07:43:01,650 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260501_074301
2026-05-01 07:43:01,654 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=18, n_classes=4)
2026-05-01 07:43:01,657 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=79106 dropped=322365 classes={'TRENDING': 4775, 'RANGING': 1876, 'CONSOLIDATING': 18867, 'VOLATILE': 53588})
2026-05-01 07:43:01,671 INFO RegimeClassifier[mode=ltf_behaviour]: 79106 samples, classes={'TRENDING': 4775, 'RANGING': 1876, 'CONSOLIDATING': 18867, 'VOLATILE': 53588}, device=cuda
2026-05-01 07:43:01,672 INFO RegimeClassifier: sample weights — mean=0.811  ambiguous(<0.4)=0.0%
2026-05-01 07:43:01,672 INFO RegimeClassifier[mode=ltf_behaviour]: warm start from existing weights
2026-05-01 07:43:01,672 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-05-01 07:43:01,863 INFO Regime epoch  1/50 — tr=0.6402 va=1.0239 acc=0.931 per_class={'TRENDING': 0.805, 'RANGING': 0.844, 'CONSOLIDATING': 0.865, 'VOLATILE': 0.97}
2026-05-01 07:43:02,013 INFO Regime epoch  2/50 — tr=0.6400 va=1.0242 acc=0.931
2026-05-01 07:43:02,152 INFO Regime epoch  3/50 — tr=0.6384 va=1.0204 acc=0.931
2026-05-01 07:43:02,292 INFO Regime epoch  4/50 — tr=0.6392 va=1.0214 acc=0.931
2026-05-01 07:43:02,453 INFO Regime epoch  5/50 — tr=0.6379 va=1.0220 acc=0.930 per_class={'TRENDING': 0.798, 'RANGING': 0.844, 'CONSOLIDATING': 0.862, 'VOLATILE': 0.971}
2026-05-01 07:43:02,603 INFO Regime epoch  6/50 — tr=0.6375 va=1.0226 acc=0.932
2026-05-01 07:43:02,741 INFO Regime epoch  7/50 — tr=0.6379 va=1.0195 acc=0.931
2026-05-01 07:43:02,883 INFO Regime epoch  8/50 — tr=0.6372 va=1.0133 acc=0.931
2026-05-01 07:43:03,034 INFO Regime epoch  9/50 — tr=0.6347 va=1.0091 acc=0.930
2026-05-01 07:43:03,195 INFO Regime epoch 10/50 — tr=0.6363 va=1.0077 acc=0.931 per_class={'TRENDING': 0.797, 'RANGING': 0.844, 'CONSOLIDATING': 0.862, 'VOLATILE': 0.973}
2026-05-01 07:43:03,337 INFO Regime epoch 11/50 — tr=0.6353 va=1.0026 acc=0.933
2026-05-01 07:43:03,490 INFO Regime epoch 12/50 — tr=0.6339 va=1.0013 acc=0.933
2026-05-01 07:43:03,645 INFO Regime epoch 13/50 — tr=0.6338 va=0.9965 acc=0.932
2026-05-01 07:43:03,790 INFO Regime epoch 14/50 — tr=0.6321 va=0.9963 acc=0.933
2026-05-01 07:43:03,950 INFO Regime epoch 15/50 — tr=0.6331 va=0.9913 acc=0.933 per_class={'TRENDING': 0.797, 'RANGING': 0.846, 'CONSOLIDATING': 0.872, 'VOLATILE': 0.972}
2026-05-01 07:43:04,101 INFO Regime epoch 16/50 — tr=0.6334 va=0.9877 acc=0.932
2026-05-01 07:43:04,244 INFO Regime epoch 17/50 — tr=0.6323 va=0.9842 acc=0.932
2026-05-01 07:43:04,386 INFO Regime epoch 18/50 — tr=0.6317 va=0.9788 acc=0.930
2026-05-01 07:43:04,536 INFO Regime epoch 19/50 — tr=0.6300 va=0.9785 acc=0.932
2026-05-01 07:43:04,684 INFO Regime epoch 20/50 — tr=0.6309 va=0.9772 acc=0.931 per_class={'TRENDING': 0.79, 'RANGING': 0.856, 'CONSOLIDATING': 0.857, 'VOLATILE': 0.974}
2026-05-01 07:43:04,823 INFO Regime epoch 21/50 — tr=0.6295 va=0.9710 acc=0.933
2026-05-01 07:43:04,958 INFO Regime epoch 22/50 — tr=0.6301 va=0.9748 acc=0.934
2026-05-01 07:43:05,097 INFO Regime epoch 23/50 — tr=0.6274 va=0.9705 acc=0.933
2026-05-01 07:43:05,236 INFO Regime epoch 24/50 — tr=0.6285 va=0.9718 acc=0.934
2026-05-01 07:43:05,384 INFO Regime epoch 25/50 — tr=0.6270 va=0.9688 acc=0.935 per_class={'TRENDING': 0.79, 'RANGING': 0.859, 'CONSOLIDATING': 0.873, 'VOLATILE': 0.973}
2026-05-01 07:43:05,538 INFO Regime epoch 26/50 — tr=0.6279 va=0.9669 acc=0.934
2026-05-01 07:43:05,678 INFO Regime epoch 27/50 — tr=0.6273 va=0.9664 acc=0.932
2026-05-01 07:43:05,820 INFO Regime epoch 28/50 — tr=0.6272 va=0.9649 acc=0.935
2026-05-01 07:43:05,971 INFO Regime epoch 29/50 — tr=0.6255 va=0.9637 acc=0.934
2026-05-01 07:43:06,127 INFO Regime epoch 30/50 — tr=0.6269 va=0.9604 acc=0.934 per_class={'TRENDING': 0.791, 'RANGING': 0.859, 'CONSOLIDATING': 0.87, 'VOLATILE': 0.973}
2026-05-01 07:43:06,274 INFO Regime epoch 31/50 — tr=0.6272 va=0.9602 acc=0.934
2026-05-01 07:43:06,426 INFO Regime epoch 32/50 — tr=0.6261 va=0.9609 acc=0.935
2026-05-01 07:43:06,567 INFO Regime epoch 33/50 — tr=0.6267 va=0.9579 acc=0.935
2026-05-01 07:43:06,714 INFO Regime epoch 34/50 — tr=0.6255 va=0.9597 acc=0.935
2026-05-01 07:43:06,873 INFO Regime epoch 35/50 — tr=0.6253 va=0.9568 acc=0.935 per_class={'TRENDING': 0.791, 'RANGING': 0.859, 'CONSOLIDATING': 0.872, 'VOLATILE': 0.974}
2026-05-01 07:43:07,015 INFO Regime epoch 36/50 — tr=0.6250 va=0.9584 acc=0.935
2026-05-01 07:43:07,154 INFO Regime epoch 37/50 — tr=0.6242 va=0.9546 acc=0.934
2026-05-01 07:43:07,301 INFO Regime epoch 38/50 — tr=0.6239 va=0.9504 acc=0.933
2026-05-01 07:43:07,441 INFO Regime epoch 39/50 — tr=0.6244 va=0.9534 acc=0.934
2026-05-01 07:43:07,599 INFO Regime epoch 40/50 — tr=0.6260 va=0.9548 acc=0.934 per_class={'TRENDING': 0.787, 'RANGING': 0.863, 'CONSOLIDATING': 0.865, 'VOLATILE': 0.975}
2026-05-01 07:43:07,755 INFO Regime epoch 41/50 — tr=0.6250 va=0.9512 acc=0.934
2026-05-01 07:43:07,894 INFO Regime epoch 42/50 — tr=0.6247 va=0.9496 acc=0.935
2026-05-01 07:43:08,039 INFO Regime epoch 43/50 — tr=0.6246 va=0.9504 acc=0.934
2026-05-01 07:43:08,184 INFO Regime epoch 44/50 — tr=0.6252 va=0.9533 acc=0.935
2026-05-01 07:43:08,346 INFO Regime epoch 45/50 — tr=0.6247 va=0.9506 acc=0.934 per_class={'TRENDING': 0.793, 'RANGING': 0.859, 'CONSOLIDATING': 0.869, 'VOLATILE': 0.974}
2026-05-01 07:43:08,496 INFO Regime epoch 46/50 — tr=0.6253 va=0.9532 acc=0.934
2026-05-01 07:43:08,641 INFO Regime epoch 47/50 — tr=0.6247 va=0.9500 acc=0.934
2026-05-01 07:43:08,789 INFO Regime epoch 48/50 — tr=0.6236 va=0.9524 acc=0.935
2026-05-01 07:43:08,928 INFO Regime epoch 49/50 — tr=0.6250 va=0.9551 acc=0.934
2026-05-01 07:43:09,080 INFO Regime epoch 50/50 — tr=0.6253 va=0.9552 acc=0.935 per_class={'TRENDING': 0.791, 'RANGING': 0.859, 'CONSOLIDATING': 0.875, 'VOLATILE': 0.974}
2026-05-01 07:43:09,094 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-01 07:43:09,094 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-01 07:43:09,095 INFO Regime phase LTF train: 7.4s
2026-05-01 07:43:09,235 INFO Regime LTF complete: acc=0.935, n=401471 per_class={'TRENDING': 0.795, 'RANGING': 0.859, 'CONSOLIDATING': 0.873, 'VOLATILE': 0.973}
2026-05-01 07:43:09,239 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 07:43:09,797 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 1778, 'RANGING': 57285, 'CONSOLIDATING': 4542, 'VOLATILE': 11019}  ambiguous=60314 (total=74624)  short_runs_zeroed=4774
2026-05-01 07:43:09,801 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 3.4794520547945207, 'RANGING': 17.033898305084747, 'CONSOLIDATING': 3.9358752166377817, 'VOLATILE': 5.842523860021209}
2026-05-01 07:43:09,809 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 1778, 'mean': 2.6590471099160558e-05, 'mean_over_std': 0.012770375199031195}, 'RANGING': {'n': 57284, 'mean': 8.05210952910413e-06, 'mean_over_std': 0.004238152373085909}, 'CONSOLIDATING': {'n': 4542, 'mean': 2.6402283987281153e-07, 'mean_over_std': 0.00015354245991368998}, 'VOLATILE': {'n': 11019, 'mean': 2.823213197675263e-05, 'mean_over_std': 0.010468794666397715}}
2026-05-01 07:43:09,809 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 915, 'mean': 7.666310957571063e-05, 'mean_over_std': 0.04299550456929985}, 'RANGING': {'n': 382, 'mean': 7.196437117461198e-05, 'mean_over_std': 0.05537815835040595}, 'CONSOLIDATING': {'n': 3375, 'mean': -3.0062004816363933e-06, 'mean_over_std': -0.0019542830050176476}, 'VOLATILE': {'n': 9638, 'mean': 1.2389816251671513e-05, 'mean_over_std': 0.004377959254493111}}
2026-05-01 07:43:09,813 INFO Regime retrain total: 208.1s (504761 samples)
2026-05-01 07:43:09,819 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-01 07:43:09,820 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 07:43:09,820 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 07:43:09,820 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-05-01 07:43:09,820 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-05-01 07:43:09,821 INFO Retrain complete. Total wall-clock: 208.1s
  DONE  Retrain regime [train-split retrain]
  SKIP  Quality/RL incremental retrain — clean train-only weights retained

=== Round 3: Backtest on last 3yr (post-retrain evaluation) ===
  START Round 3 - Post-retrain backtest (last 3yr)
2026-05-01 07:43:11,548 INFO === STEP 6: BACKTEST (round3) ===
2026-05-01 07:43:11,549 INFO BT_WINDOW=round3 — post-retrain eval: 2022-08-05 → 2025-08-05 (last 3yr)
2026-05-01 07:43:11,549 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-05-01 07:43:11,549 INFO Round 3 — running backtest: 2022-08-05 → 2025-08-05 (ml_trader, shared ML cache)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
2026-05-01 07:47:42,316 INFO Round 3 backtest — 1 trades | avg WR=0.0% | avg PF=0.00 | avg Sharpe=0.00
2026-05-01 07:47:42,316 INFO   ml_trader: 1 trades | WR=0.0% | fixed PF=0.00 | Return=-1.0% | ExpR=-1.001 | DD=1.0% | Sharpe=0.00
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 1
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (1 rows)
  DONE  Round 3 - Post-retrain backtest (last 3yr)
  Saved Round 3 result → round3_summary.json
  Journal after Round 3: 2 entries

  SKIP  Round 3 Quality+RL retrain — evaluation journals not used for fitting

======================================================================
  BLIND BACKTEST PIPELINE COMPLETE
======================================================================
  Round 1 (val window)          trades=0  WR=0.0%  PF=0.000  Sharpe=0.000
  Round 2 (blind test)          trades=1  WR=0.0%  PF=0.000  Sharpe=0.000
  Round 3 (last 3yr)            trades=1  WR=0.0%  PF=0.000  Sharpe=0.000