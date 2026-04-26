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
2026-04-26 01:06:47,464 INFO Loading feature-engineered data...
2026-04-26 01:06:47,996 INFO Loaded 221743 rows, 202 features
2026-04-26 01:06:47,997 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-26 01:06:47,997 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-26 01:06:47,997 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-26 01:06:47,997 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-26 01:06:47,997 INFO No leakage confirmed: train < val < test timestamps

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
2026-04-26 01:06:50,396 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-26 01:06:50,396 INFO --- Training regime ---
2026-04-26 01:06:50,396 INFO Running retrain --model regime
2026-04-26 01:06:50,582 INFO retrain environment: KAGGLE
2026-04-26 01:06:52,229 INFO Device: CUDA (2 GPU(s))
2026-04-26 01:06:52,240 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 01:06:52,241 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 01:06:52,241 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 01:06:52,241 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 01:06:52,242 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-26 01:06:52,393 INFO NumExpr defaulting to 4 threads.
2026-04-26 01:06:52,605 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 01:06:52,606 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 01:06:52,606 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 01:06:52,798 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 01:06:52,800 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:52,876 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:52,949 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,019 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,093 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,166 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,237 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,312 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,385 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,468 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,564 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 01:06:53,625 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-26 01:06:53,640 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,641 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,662 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,664 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,682 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,684 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,703 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,706 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,729 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,734 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,752 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,756 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,771 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,775 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,794 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,797 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,812 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,816 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,834 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,838 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:06:53,855 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 01:06:53,862 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 01:06:54,607 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 01:07:17,269 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 01:07:17,271 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias)
2026-04-26 01:07:17,271 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 01:07:27,124 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 01:07:27,126 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias)
2026-04-26 01:07:27,126 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 01:07:34,784 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 01:07:34,785 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias)
2026-04-26 01:07:34,786 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 01:08:44,215 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 01:08:44,218 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour)
2026-04-26 01:08:44,218 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 01:09:15,295 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 01:09:15,299 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour)
2026-04-26 01:09:15,299 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 01:09:36,829 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 01:09:36,829 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour)
2026-04-26 01:09:36,931 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-26 01:09:36,933 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:09:36,934 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:09:36,935 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:09:36,937 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:09:36,938 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:09:36,939 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:09:36,940 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:09:36,941 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:09:36,942 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:09:36,943 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:09:36,945 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 01:09:37,069 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:37,117 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:37,118 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:37,118 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:37,127 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:37,128 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:09:39,897 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-26 01:09:39,898 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-26 01:09:40,076 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:40,110 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:40,111 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:40,111 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:40,119 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:40,120 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:09:42,870 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-26 01:09:42,871 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-26 01:09:43,050 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:43,087 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:43,088 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:43,088 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:43,097 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:43,098 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:09:45,933 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-26 01:09:45,935 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-26 01:09:46,100 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:46,138 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:46,139 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:46,139 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:46,148 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:46,149 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:09:48,935 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-26 01:09:48,936 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-26 01:09:49,110 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:49,146 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:49,147 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:49,147 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:49,156 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:49,157 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:09:51,977 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-26 01:09:51,978 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-26 01:09:52,150 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:52,184 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:52,185 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:52,186 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:52,194 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:52,195 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:09:55,072 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-26 01:09:55,073 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-26 01:09:55,228 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 01:09:55,257 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 01:09:55,258 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 01:09:55,258 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 01:09:55,266 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 01:09:55,267 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:09:58,066 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-26 01:09:58,067 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-26 01:09:58,233 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:58,267 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:58,268 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:58,268 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:58,277 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:09:58,278 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:10:01,072 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-26 01:10:01,073 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-26 01:10:01,245 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:01,279 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:01,280 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:01,280 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:01,289 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:01,290 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:10:04,166 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-26 01:10:04,167 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-26 01:10:04,343 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:04,377 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:04,378 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:04,379 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:04,388 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:04,389 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:10:07,196 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-26 01:10:07,197 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-26 01:10:07,468 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 01:10:07,525 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 01:10:07,526 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 01:10:07,527 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 01:10:07,538 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 01:10:07,539 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 01:10:14,045 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-26 01:10:14,047 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-26 01:10:14,232 INFO RegimeClassifier[mode=htf_bias]: 103290 samples, classes={'BIAS_UP': 18622, 'BIAS_DOWN': 18286, 'BIAS_NEUTRAL': 66382}, device=cuda
2026-04-26 01:10:14,233 INFO RegimeClassifier: sample weights — mean=0.360  ambiguous(<0.4)=69.5%
2026-04-26 01:10:14,426 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-04-26 01:10:14,426 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 01:10:16,781 INFO Regime epoch  1/50 — tr=0.3387 va=2.3171 acc=0.246 per_class={'BIAS_UP': 0.038, 'BIAS_DOWN': 0.859, 'BIAS_NEUTRAL': 0.165}
2026-04-26 01:10:16,978 INFO Regime epoch  2/50 — tr=0.3345 va=2.3788 acc=0.237
2026-04-26 01:10:17,158 INFO Regime epoch  3/50 — tr=0.3278 va=2.3417 acc=0.300
2026-04-26 01:10:17,348 INFO Regime epoch  4/50 — tr=0.3179 va=2.2503 acc=0.412
2026-04-26 01:10:17,558 INFO Regime epoch  5/50 — tr=0.3043 va=2.1473 acc=0.493 per_class={'BIAS_UP': 0.801, 'BIAS_DOWN': 0.966, 'BIAS_NEUTRAL': 0.263}
2026-04-26 01:10:17,749 INFO Regime epoch  6/50 — tr=0.2927 va=2.0364 acc=0.541
2026-04-26 01:10:17,922 INFO Regime epoch  7/50 — tr=0.2823 va=1.9570 acc=0.564
2026-04-26 01:10:18,101 INFO Regime epoch  8/50 — tr=0.2749 va=1.9140 acc=0.574
2026-04-26 01:10:18,289 INFO Regime epoch  9/50 — tr=0.2699 va=1.8870 acc=0.579
2026-04-26 01:10:18,494 INFO Regime epoch 10/50 — tr=0.2651 va=1.8647 acc=0.584 per_class={'BIAS_UP': 0.931, 'BIAS_DOWN': 0.992, 'BIAS_NEUTRAL': 0.356}
2026-04-26 01:10:18,686 INFO Regime epoch 11/50 — tr=0.2618 va=1.8532 acc=0.588
2026-04-26 01:10:18,879 INFO Regime epoch 12/50 — tr=0.2589 va=1.8422 acc=0.595
2026-04-26 01:10:19,057 INFO Regime epoch 13/50 — tr=0.2565 va=1.8345 acc=0.601
2026-04-26 01:10:19,237 INFO Regime epoch 14/50 — tr=0.2548 va=1.8343 acc=0.599
2026-04-26 01:10:19,429 INFO Regime epoch 15/50 — tr=0.2536 va=1.8250 acc=0.607 per_class={'BIAS_UP': 0.954, 'BIAS_DOWN': 0.989, 'BIAS_NEUTRAL': 0.387}
2026-04-26 01:10:19,606 INFO Regime epoch 16/50 — tr=0.2522 va=1.8227 acc=0.608
2026-04-26 01:10:19,784 INFO Regime epoch 17/50 — tr=0.2510 va=1.8186 acc=0.612
2026-04-26 01:10:19,975 INFO Regime epoch 18/50 — tr=0.2503 va=1.8151 acc=0.617
2026-04-26 01:10:20,152 INFO Regime epoch 19/50 — tr=0.2494 va=1.8130 acc=0.619
2026-04-26 01:10:20,345 INFO Regime epoch 20/50 — tr=0.2487 va=1.8115 acc=0.622 per_class={'BIAS_UP': 0.962, 'BIAS_DOWN': 0.989, 'BIAS_NEUTRAL': 0.408}
2026-04-26 01:10:20,529 INFO Regime epoch 21/50 — tr=0.2482 va=1.8067 acc=0.624
2026-04-26 01:10:20,704 INFO Regime epoch 22/50 — tr=0.2477 va=1.8059 acc=0.627
2026-04-26 01:10:20,889 INFO Regime epoch 23/50 — tr=0.2475 va=1.8031 acc=0.630
2026-04-26 01:10:21,083 INFO Regime epoch 24/50 — tr=0.2469 va=1.8033 acc=0.629
2026-04-26 01:10:21,281 INFO Regime epoch 25/50 — tr=0.2465 va=1.7974 acc=0.636 per_class={'BIAS_UP': 0.967, 'BIAS_DOWN': 0.987, 'BIAS_NEUTRAL': 0.429}
2026-04-26 01:10:21,473 INFO Regime epoch 26/50 — tr=0.2460 va=1.7992 acc=0.636
2026-04-26 01:10:21,653 INFO Regime epoch 27/50 — tr=0.2459 va=1.7972 acc=0.639
2026-04-26 01:10:21,828 INFO Regime epoch 28/50 — tr=0.2458 va=1.7984 acc=0.636
2026-04-26 01:10:22,011 INFO Regime epoch 29/50 — tr=0.2454 va=1.7972 acc=0.640
2026-04-26 01:10:22,211 INFO Regime epoch 30/50 — tr=0.2453 va=1.7951 acc=0.642 per_class={'BIAS_UP': 0.968, 'BIAS_DOWN': 0.988, 'BIAS_NEUTRAL': 0.438}
2026-04-26 01:10:22,401 INFO Regime epoch 31/50 — tr=0.2451 va=1.7966 acc=0.641
2026-04-26 01:10:22,579 INFO Regime epoch 32/50 — tr=0.2451 va=1.7927 acc=0.644
2026-04-26 01:10:22,752 INFO Regime epoch 33/50 — tr=0.2448 va=1.7937 acc=0.645
2026-04-26 01:10:22,936 INFO Regime epoch 34/50 — tr=0.2448 va=1.7904 acc=0.645
2026-04-26 01:10:23,132 INFO Regime epoch 35/50 — tr=0.2447 va=1.7908 acc=0.646 per_class={'BIAS_UP': 0.972, 'BIAS_DOWN': 0.988, 'BIAS_NEUTRAL': 0.442}
2026-04-26 01:10:23,319 INFO Regime epoch 36/50 — tr=0.2445 va=1.7933 acc=0.644
2026-04-26 01:10:23,499 INFO Regime epoch 37/50 — tr=0.2445 va=1.7912 acc=0.645
2026-04-26 01:10:23,698 INFO Regime epoch 38/50 — tr=0.2444 va=1.7875 acc=0.649
2026-04-26 01:10:23,886 INFO Regime epoch 39/50 — tr=0.2445 va=1.7898 acc=0.647
2026-04-26 01:10:24,084 INFO Regime epoch 40/50 — tr=0.2445 va=1.7892 acc=0.648 per_class={'BIAS_UP': 0.974, 'BIAS_DOWN': 0.989, 'BIAS_NEUTRAL': 0.445}
2026-04-26 01:10:24,257 INFO Regime epoch 41/50 — tr=0.2443 va=1.7918 acc=0.645
2026-04-26 01:10:24,436 INFO Regime epoch 42/50 — tr=0.2441 va=1.7894 acc=0.648
2026-04-26 01:10:24,609 INFO Regime epoch 43/50 — tr=0.2441 va=1.7881 acc=0.651
2026-04-26 01:10:24,785 INFO Regime epoch 44/50 — tr=0.2442 va=1.7896 acc=0.648
2026-04-26 01:10:24,973 INFO Regime epoch 45/50 — tr=0.2441 va=1.7886 acc=0.652 per_class={'BIAS_UP': 0.971, 'BIAS_DOWN': 0.988, 'BIAS_NEUTRAL': 0.452}
2026-04-26 01:10:25,157 INFO Regime epoch 46/50 — tr=0.2443 va=1.7894 acc=0.648
2026-04-26 01:10:25,335 INFO Regime epoch 47/50 — tr=0.2441 va=1.7903 acc=0.649
2026-04-26 01:10:25,519 INFO Regime epoch 48/50 — tr=0.2442 va=1.7898 acc=0.649
2026-04-26 01:10:25,519 INFO Regime early stop at epoch 48 (no_improve=10)
2026-04-26 01:10:25,532 WARNING RegimeClassifier accuracy 0.65 < 0.65 threshold
2026-04-26 01:10:25,535 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 01:10:25,535 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 01:10:25,661 INFO Regime HTF complete: acc=0.649, n=103290
2026-04-26 01:10:25,663 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 01:10:25,838 INFO Rule labels LTF_BEHAVIOUR [4H]: {'TRENDING': 8970, 'RANGING': 4592, 'CONSOLIDATING': 2247, 'VOLATILE': 4008}  ambiguous=4539 (total=19817)  short_runs_zeroed=570
2026-04-26 01:10:25,841 INFO Regime[4H] persistence (avg bars/run) on XAUUSD 4H:
{0: 12.121621621621621, 1: 6.4858757062146895, 2: 5.575682382133995, 3: 10.329896907216495}
2026-04-26 01:10:25,842 INFO Regime[4H] return separation on XAUUSD 4H:
{0: 5.9502411776840314e-05, 1: 5.664222849552333e-05, 2: -6.138442068010657e-05, 3: 4.239843820627722e-05}
2026-04-26 01:10:25,843 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-26 01:10:25,844 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:10:25,846 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:10:25,847 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:10:25,849 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:10:25,851 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:10:25,852 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:10:25,854 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:10:25,855 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:10:25,857 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:10:25,858 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:10:25,861 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 01:10:25,871 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:25,873 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:25,874 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:25,874 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:25,875 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:25,876 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:10:36,051 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-26 01:10:36,054 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-26 01:10:36,191 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:36,194 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:36,194 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:36,195 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:36,195 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:36,197 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:10:46,200 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-26 01:10:46,203 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-26 01:10:46,339 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:46,341 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:46,342 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:46,342 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:46,342 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:46,344 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:10:56,495 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-26 01:10:56,498 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-26 01:10:56,635 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:56,637 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:56,638 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:56,639 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:56,639 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:10:56,641 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:11:06,761 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-26 01:11:06,764 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-26 01:11:06,895 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:06,898 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:06,898 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:06,899 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:06,899 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:06,901 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:11:16,851 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-26 01:11:16,854 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-26 01:11:16,998 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:17,000 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:17,001 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:17,001 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:17,002 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:17,004 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:11:26,918 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-26 01:11:26,920 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-26 01:11:27,058 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 01:11:27,060 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 01:11:27,060 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 01:11:27,061 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 01:11:27,061 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 01:11:27,063 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:11:36,907 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-26 01:11:36,910 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-26 01:11:37,044 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:37,047 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:37,048 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:37,048 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:37,048 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:37,050 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:11:46,930 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-26 01:11:46,933 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-26 01:11:47,070 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:47,072 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:47,073 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:47,074 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:47,074 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:47,076 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:11:57,054 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-26 01:11:57,057 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-26 01:11:57,195 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:57,198 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:57,198 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:57,199 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:57,199 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:11:57,201 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:12:07,052 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-26 01:12:07,055 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-26 01:12:07,196 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 01:12:07,199 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 01:12:07,201 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 01:12:07,201 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 01:12:07,201 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 01:12:07,205 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 01:12:29,898 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-26 01:12:29,903 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-26 01:12:30,249 INFO RegimeClassifier[mode=ltf_behaviour]: 401471 samples, classes={'TRENDING': 160094, 'RANGING': 105286, 'CONSOLIDATING': 53524, 'VOLATILE': 82567}, device=cuda
2026-04-26 01:12:30,250 INFO RegimeClassifier: sample weights — mean=0.505  ambiguous(<0.4)=33.0%
2026-04-26 01:12:30,253 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-04-26 01:12:30,253 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 01:12:30,987 INFO Regime epoch  1/50 — tr=0.6397 va=2.0585 acc=0.359 per_class={'TRENDING': 0.429, 'RANGING': 0.33, 'CONSOLIDATING': 0.309, 'VOLATILE': 0.29}
2026-04-26 01:12:31,642 INFO Regime epoch  2/50 — tr=0.6159 va=2.0032 acc=0.493
2026-04-26 01:12:32,323 INFO Regime epoch  3/50 — tr=0.5832 va=1.9220 acc=0.511
2026-04-26 01:12:32,982 INFO Regime epoch  4/50 — tr=0.5548 va=1.8607 acc=0.524
2026-04-26 01:12:33,753 INFO Regime epoch  5/50 — tr=0.5374 va=1.8141 acc=0.541 per_class={'TRENDING': 0.524, 'RANGING': 0.145, 'CONSOLIDATING': 0.759, 'VOLATILE': 0.945}
2026-04-26 01:12:34,444 INFO Regime epoch  6/50 — tr=0.5258 va=1.7823 acc=0.569
2026-04-26 01:12:35,114 INFO Regime epoch  7/50 — tr=0.5175 va=1.7577 acc=0.596
2026-04-26 01:12:35,771 INFO Regime epoch  8/50 — tr=0.5101 va=1.7327 acc=0.624
2026-04-26 01:12:36,448 INFO Regime epoch  9/50 — tr=0.5043 va=1.7123 acc=0.640
2026-04-26 01:12:37,185 INFO Regime epoch 10/50 — tr=0.4996 va=1.6899 acc=0.654 per_class={'TRENDING': 0.748, 'RANGING': 0.22, 'CONSOLIDATING': 0.829, 'VOLATILE': 0.918}
2026-04-26 01:12:37,875 INFO Regime epoch 11/50 — tr=0.4960 va=1.6739 acc=0.662
2026-04-26 01:12:38,552 INFO Regime epoch 12/50 — tr=0.4932 va=1.6639 acc=0.668
2026-04-26 01:12:39,236 INFO Regime epoch 13/50 — tr=0.4907 va=1.6534 acc=0.676
2026-04-26 01:12:39,911 INFO Regime epoch 14/50 — tr=0.4890 va=1.6447 acc=0.680
2026-04-26 01:12:40,621 INFO Regime epoch 15/50 — tr=0.4876 va=1.6372 acc=0.684 per_class={'TRENDING': 0.788, 'RANGING': 0.256, 'CONSOLIDATING': 0.861, 'VOLATILE': 0.919}
2026-04-26 01:12:41,294 INFO Regime epoch 16/50 — tr=0.4863 va=1.6302 acc=0.687
2026-04-26 01:12:41,977 INFO Regime epoch 17/50 — tr=0.4850 va=1.6228 acc=0.694
2026-04-26 01:12:42,649 INFO Regime epoch 18/50 — tr=0.4839 va=1.6177 acc=0.696
2026-04-26 01:12:43,328 INFO Regime epoch 19/50 — tr=0.4830 va=1.6166 acc=0.697
2026-04-26 01:12:44,078 INFO Regime epoch 20/50 — tr=0.4820 va=1.6132 acc=0.700 per_class={'TRENDING': 0.802, 'RANGING': 0.285, 'CONSOLIDATING': 0.893, 'VOLATILE': 0.911}
2026-04-26 01:12:44,742 INFO Regime epoch 21/50 — tr=0.4813 va=1.6082 acc=0.704
2026-04-26 01:12:45,406 INFO Regime epoch 22/50 — tr=0.4806 va=1.6071 acc=0.704
2026-04-26 01:12:46,095 INFO Regime epoch 23/50 — tr=0.4799 va=1.6045 acc=0.706
2026-04-26 01:12:46,765 INFO Regime epoch 24/50 — tr=0.4796 va=1.6012 acc=0.708
2026-04-26 01:12:47,478 INFO Regime epoch 25/50 — tr=0.4791 va=1.5981 acc=0.710 per_class={'TRENDING': 0.806, 'RANGING': 0.306, 'CONSOLIDATING': 0.911, 'VOLATILE': 0.911}
2026-04-26 01:12:48,171 INFO Regime epoch 26/50 — tr=0.4783 va=1.5966 acc=0.710
2026-04-26 01:12:48,851 INFO Regime epoch 27/50 — tr=0.4780 va=1.5937 acc=0.713
2026-04-26 01:12:49,530 INFO Regime epoch 28/50 — tr=0.4778 va=1.5944 acc=0.715
2026-04-26 01:12:50,225 INFO Regime epoch 29/50 — tr=0.4774 va=1.5965 acc=0.712
2026-04-26 01:12:50,939 INFO Regime epoch 30/50 — tr=0.4769 va=1.5930 acc=0.714 per_class={'TRENDING': 0.815, 'RANGING': 0.31, 'CONSOLIDATING': 0.919, 'VOLATILE': 0.905}
2026-04-26 01:12:51,618 INFO Regime epoch 31/50 — tr=0.4766 va=1.5906 acc=0.716
2026-04-26 01:12:52,279 INFO Regime epoch 32/50 — tr=0.4764 va=1.5896 acc=0.717
2026-04-26 01:12:52,955 INFO Regime epoch 33/50 — tr=0.4763 va=1.5874 acc=0.719
2026-04-26 01:12:53,631 INFO Regime epoch 34/50 — tr=0.4763 va=1.5884 acc=0.719
2026-04-26 01:12:54,373 INFO Regime epoch 35/50 — tr=0.4761 va=1.5882 acc=0.719 per_class={'TRENDING': 0.818, 'RANGING': 0.322, 'CONSOLIDATING': 0.924, 'VOLATILE': 0.905}
2026-04-26 01:12:55,057 INFO Regime epoch 36/50 — tr=0.4759 va=1.5871 acc=0.719
2026-04-26 01:12:55,723 INFO Regime epoch 37/50 — tr=0.4758 va=1.5844 acc=0.720
2026-04-26 01:12:56,374 INFO Regime epoch 38/50 — tr=0.4757 va=1.5873 acc=0.720
2026-04-26 01:12:57,042 INFO Regime epoch 39/50 — tr=0.4754 va=1.5862 acc=0.719
2026-04-26 01:12:57,768 INFO Regime epoch 40/50 — tr=0.4754 va=1.5849 acc=0.720 per_class={'TRENDING': 0.824, 'RANGING': 0.318, 'CONSOLIDATING': 0.931, 'VOLATILE': 0.899}
2026-04-26 01:12:58,455 INFO Regime epoch 41/50 — tr=0.4755 va=1.5849 acc=0.721
2026-04-26 01:12:59,135 INFO Regime epoch 42/50 — tr=0.4754 va=1.5878 acc=0.719
2026-04-26 01:12:59,800 INFO Regime epoch 43/50 — tr=0.4753 va=1.5823 acc=0.722
2026-04-26 01:13:00,454 INFO Regime epoch 44/50 — tr=0.4752 va=1.5858 acc=0.720
2026-04-26 01:13:01,189 INFO Regime epoch 45/50 — tr=0.4751 va=1.5858 acc=0.720 per_class={'TRENDING': 0.825, 'RANGING': 0.315, 'CONSOLIDATING': 0.928, 'VOLATILE': 0.901}
2026-04-26 01:13:01,884 INFO Regime epoch 46/50 — tr=0.4752 va=1.5879 acc=0.719
2026-04-26 01:13:02,560 INFO Regime epoch 47/50 — tr=0.4752 va=1.5856 acc=0.720
2026-04-26 01:13:03,270 INFO Regime epoch 48/50 — tr=0.4752 va=1.5866 acc=0.720
2026-04-26 01:13:03,988 INFO Regime epoch 49/50 — tr=0.4752 va=1.5873 acc=0.720
2026-04-26 01:13:04,732 INFO Regime epoch 50/50 — tr=0.4753 va=1.5877 acc=0.719 per_class={'TRENDING': 0.819, 'RANGING': 0.317, 'CONSOLIDATING': 0.925, 'VOLATILE': 0.909}
2026-04-26 01:13:04,780 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 01:13:04,780 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 01:13:04,915 INFO Regime LTF complete: acc=0.722, n=401471
2026-04-26 01:13:04,919 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 01:13:05,436 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-26 01:13:05,441 INFO Regime[1H] persistence (avg bars/run) on XAUUSD 1H:
{0: 9.355291913830783, 1: 6.096303199751476, 2: 5.598885793871866, 3: 6.771351107094442}
2026-04-26 01:13:05,444 INFO Regime[1H] return separation on XAUUSD 1H:
{0: 2.149062259083323e-05, 1: -3.254493608872846e-06, 2: -3.039756158146958e-06, 3: 1.8102764458593193e-05}
2026-04-26 01:13:05,459 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 01:13:05,460 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 01:13:05,460 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 01:13:05,460 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-26 01:13:05,465 INFO Loading faiss with AVX512 support.
2026-04-26 01:13:05,489 INFO Successfully loaded faiss with AVX512 support.
2026-04-26 01:13:05,494 INFO VectorStore: CPU FAISS index built (dim=74)
2026-04-26 01:13:05,494 INFO VectorStore: CPU FAISS index built (dim=34)
2026-04-26 01:13:05,494 INFO VectorStore: CPU FAISS index built (dim=64)
2026-04-26 01:13:05,497 WARNING GRULSTMPredictor: stale weights detected (quality feature contract changed: added=['gru_signal_agreement', 'strategy_win_rate_5', 'strategy_win_rate_50']; count 17→20) — deleting /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt so retrain starts fresh
2026-04-26 01:13:05,497 INFO Deleted stale weights (quality feature contract changed: added=['gru_signal_agreement', 'strategy_win_rate_5', 'strategy_win_rate_50']; count 17→20): /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:13:05,518 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:13:05,519 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:13:05,519 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:13:05,521 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:13:40,857 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:13:40,864 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:13:40,866 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:13:40,869 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:13:40,869 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:13:41,972 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:13:41,978 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:13:41,979 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:13:41,979 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:13:41,980 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:13:42,560 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:13:42,562 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:13:42,564 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:13:42,567 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:13:42,568 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:13:43,441 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:13:43,444 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:13:43,446 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:13:43,447 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:13:43,447 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:14:42,624 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:15:34,670 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:16:20,433 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:17:13,886 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:17:33,853 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:17:33,856 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:17:33,857 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:17:33,860 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:17:33,862 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:17:35,612 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:17:35,615 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:17:35,618 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:17:35,622 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:17:35,622 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:17:37,847 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 01:17:37,851 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 01:17:37,852 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 01:17:37,853 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 01:17:37,855 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 01:17:39,447 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:17:39,452 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:17:39,454 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:17:39,456 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:17:39,458 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:18:35,948 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:19:23,575 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:20:18,860 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-26 01:21:22,745 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:21:22,748 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:21:22,750 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:21:22,750 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:21:22,751 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:21:23,555 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:21:23,558 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:21:23,558 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:21:23,559 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:21:23,560 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:23:04,425 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 01:23:04,429 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 01:23:04,430 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 01:23:04,431 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 01:23:04,431 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 01:24:28,378 INFO VectorStore trade_patterns: +50000 vectors for AUDUSD
2026-04-26 01:24:28,391 INFO VectorStore market_structures: +65447 vectors (34-dim 4H) for AUDUSD
2026-04-26 01:24:28,577 INFO VectorStore trade_patterns: +50000 vectors for EURGBP
2026-04-26 01:24:28,593 INFO VectorStore market_structures: +65448 vectors (34-dim 4H) for EURGBP
2026-04-26 01:24:28,791 INFO VectorStore trade_patterns: +50000 vectors for EURJPY
2026-04-26 01:24:28,808 INFO VectorStore market_structures: +65453 vectors (34-dim 4H) for EURJPY
2026-04-26 01:24:28,993 INFO VectorStore trade_patterns: +50000 vectors for EURUSD
2026-04-26 01:24:29,033 INFO VectorStore market_structures: +65451 vectors (34-dim 4H) for EURUSD
2026-04-26 01:24:29,273 INFO VectorStore trade_patterns: +50000 vectors for GBPJPY
2026-04-26 01:24:29,291 INFO VectorStore market_structures: +65452 vectors (34-dim 4H) for GBPJPY
2026-04-26 01:24:29,485 INFO VectorStore trade_patterns: +50000 vectors for GBPUSD
2026-04-26 01:24:29,541 INFO VectorStore market_structures: +65451 vectors (34-dim 4H) for GBPUSD
2026-04-26 01:24:29,740 INFO VectorStore trade_patterns: +50000 vectors for NZDUSD
2026-04-26 01:24:29,759 INFO VectorStore market_structures: +65451 vectors (34-dim 4H) for NZDUSD
2026-04-26 01:24:29,959 INFO VectorStore trade_patterns: +50000 vectors for USDCAD
2026-04-26 01:24:29,977 INFO VectorStore market_structures: +65452 vectors (34-dim 4H) for USDCAD
2026-04-26 01:24:30,276 INFO VectorStore trade_patterns: +50000 vectors for USDCHF
2026-04-26 01:24:30,293 INFO VectorStore market_structures: +65454 vectors (34-dim 4H) for USDCHF
2026-04-26 01:24:30,503 INFO VectorStore trade_patterns: +50000 vectors for USDJPY
2026-04-26 01:24:30,587 INFO VectorStore market_structures: +65461 vectors (34-dim 4H) for USDJPY
2026-04-26 01:24:30,789 INFO VectorStore trade_patterns: +50000 vectors for XAUUSD
2026-04-26 01:24:30,805 INFO VectorStore market_structures: +59006 vectors (34-dim 4H) for XAUUSD
2026-04-26 01:24:31,836 INFO VectorStore: saved 1263526 total vectors to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/vector_store
2026-04-26 01:24:31,836 INFO VectorStore saved: {'trade_patterns': 550000, 'market_structures': 713526, 'regime_embeddings': 0}
2026-04-26 01:24:31,962 INFO Retrain complete.
2026-04-26 01:24:33,084 INFO Model regime: SUCCESS
2026-04-26 01:24:33,085 INFO --- Training gru ---
2026-04-26 01:24:33,085 INFO Running retrain --model gru
2026-04-26 01:24:33,443 INFO retrain environment: KAGGLE
2026-04-26 01:24:35,074 INFO Device: CUDA (2 GPU(s))
2026-04-26 01:24:35,085 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 01:24:35,085 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 01:24:35,086 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 01:24:35,086 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 01:24:35,087 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-26 01:24:35,230 INFO NumExpr defaulting to 4 threads.
2026-04-26 01:24:35,420 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 01:24:35,421 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 01:24:35,421 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 01:24:35,638 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 01:24:35,640 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:35,717 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:35,791 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:35,868 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:35,945 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:36,019 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:36,111 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:36,191 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:36,260 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:36,330 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:36,414 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 01:24:36,474 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-26 01:24:36,474 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260426_012436
2026-04-26 01:24:36,476 INFO GRU weights stale (quality feature contract changed: added=['gru_signal_agreement', 'strategy_win_rate_5', 'strategy_win_rate_50']; count 17→20) — deleting for full retrain
2026-04-26 01:24:36,594 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:36,595 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:36,610 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:36,617 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:36,618 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 01:24:36,618 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 01:24:36,618 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 01:24:36,619 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:36,692 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-26 01:24:36,693 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:36,921 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-26 01:24:36,950 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:37,222 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:37,343 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:37,434 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:37,627 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:37,628 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:37,644 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:37,651 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:37,652 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:37,720 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-26 01:24:37,722 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:37,945 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-26 01:24:37,960 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:38,216 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:38,339 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:38,430 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:38,618 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:38,618 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:38,635 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:38,642 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:38,642 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:38,712 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-26 01:24:38,714 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:38,931 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-26 01:24:38,946 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:39,208 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:39,330 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:39,423 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:39,603 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:39,604 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:39,619 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:39,626 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:39,627 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:39,696 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-26 01:24:39,698 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:39,908 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-26 01:24:39,930 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:40,194 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:40,319 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:40,421 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:40,607 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:40,608 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:40,625 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:40,634 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:40,634 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:40,703 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-26 01:24:40,705 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:40,925 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-26 01:24:40,940 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:41,220 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:41,351 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:41,448 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:41,629 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:41,630 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:41,656 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:41,667 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:41,668 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:41,762 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-26 01:24:41,764 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:42,004 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-26 01:24:42,020 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:42,291 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:42,420 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:42,526 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:42,707 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 01:24:42,708 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 01:24:42,722 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 01:24:42,729 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 01:24:42,730 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:42,799 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-26 01:24:42,801 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:43,028 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-26 01:24:43,041 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:43,301 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:43,428 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:43,522 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:43,711 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:43,712 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:43,737 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:43,744 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:43,745 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:43,819 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-26 01:24:43,820 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:44,045 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-26 01:24:44,062 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:44,322 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:44,443 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:44,540 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:44,724 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:44,724 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:44,739 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:44,746 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:44,747 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:44,816 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-26 01:24:44,818 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:45,035 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-26 01:24:45,051 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:45,308 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:45,433 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:45,528 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:45,709 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:45,710 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:45,726 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:45,734 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:24:45,734 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:45,805 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-26 01:24:45,807 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:46,030 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-26 01:24:46,045 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:46,306 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:46,435 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:46,529 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:24:46,810 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 01:24:46,811 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 01:24:46,829 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 01:24:46,839 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 01:24:46,840 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 01:24:46,985 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-26 01:24:46,988 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 01:24:47,443 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-26 01:24:47,489 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-26 01:24:48,014 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-26 01:24:48,206 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 01:24:48,329 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 01:24:48,444 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-26 01:24:48,673 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-04-26 01:24:48,673 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-04-26 01:24:48,673 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-26 01:24:48,673 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-26 01:29:06,667 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-26 01:29:06,667 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-26 01:29:07,953 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-26 01:29:11,993 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=5)
2026-04-26 01:29:25,222 INFO train_multi TF=ALL epoch 1/50 train=0.8939 val=0.8854
2026-04-26 01:29:25,227 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:29:25,227 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:29:25,227 INFO train_multi TF=ALL: new best val=0.8854 — saved
2026-04-26 01:29:36,834 INFO train_multi TF=ALL epoch 2/50 train=0.8726 val=0.8442
2026-04-26 01:29:36,839 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:29:36,839 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:29:36,839 INFO train_multi TF=ALL: new best val=0.8442 — saved
2026-04-26 01:29:48,451 INFO train_multi TF=ALL epoch 3/50 train=0.7612 val=0.6903
2026-04-26 01:29:48,455 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:29:48,456 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:29:48,456 INFO train_multi TF=ALL: new best val=0.6903 — saved
2026-04-26 01:29:59,977 INFO train_multi TF=ALL epoch 4/50 train=0.6924 val=0.6882
2026-04-26 01:29:59,982 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:29:59,982 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:29:59,982 INFO train_multi TF=ALL: new best val=0.6882 — saved
2026-04-26 01:30:11,539 INFO train_multi TF=ALL epoch 5/50 train=0.6909 val=0.6880
2026-04-26 01:30:11,544 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:30:11,544 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:30:11,544 INFO train_multi TF=ALL: new best val=0.6880 — saved
2026-04-26 01:30:23,133 INFO train_multi TF=ALL epoch 6/50 train=0.6903 val=0.6881
2026-04-26 01:30:34,768 INFO train_multi TF=ALL epoch 7/50 train=0.6897 val=0.6879
2026-04-26 01:30:34,773 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:30:34,773 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:30:34,773 INFO train_multi TF=ALL: new best val=0.6879 — saved
2026-04-26 01:30:46,349 INFO train_multi TF=ALL epoch 8/50 train=0.6895 val=0.6878
2026-04-26 01:30:46,354 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:30:46,354 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:30:46,354 INFO train_multi TF=ALL: new best val=0.6878 — saved
2026-04-26 01:30:57,882 INFO train_multi TF=ALL epoch 9/50 train=0.6891 val=0.6880
2026-04-26 01:31:09,468 INFO train_multi TF=ALL epoch 10/50 train=0.6888 val=0.6874
2026-04-26 01:31:09,472 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:31:09,472 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:31:09,472 INFO train_multi TF=ALL: new best val=0.6874 — saved
2026-04-26 01:31:20,998 INFO train_multi TF=ALL epoch 11/50 train=0.6880 val=0.6868
2026-04-26 01:31:21,003 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:31:21,003 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:31:21,003 INFO train_multi TF=ALL: new best val=0.6868 — saved
2026-04-26 01:31:32,566 INFO train_multi TF=ALL epoch 12/50 train=0.6856 val=0.6850
2026-04-26 01:31:32,571 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:31:32,571 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:31:32,571 INFO train_multi TF=ALL: new best val=0.6850 — saved
2026-04-26 01:31:44,151 INFO train_multi TF=ALL epoch 13/50 train=0.6793 val=0.6758
2026-04-26 01:31:44,155 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:31:44,155 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:31:44,156 INFO train_multi TF=ALL: new best val=0.6758 — saved
2026-04-26 01:31:55,744 INFO train_multi TF=ALL epoch 14/50 train=0.6698 val=0.6616
2026-04-26 01:31:55,748 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:31:55,748 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:31:55,748 INFO train_multi TF=ALL: new best val=0.6616 — saved
2026-04-26 01:32:07,368 INFO train_multi TF=ALL epoch 15/50 train=0.6581 val=0.6489
2026-04-26 01:32:07,372 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:32:07,372 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:32:07,372 INFO train_multi TF=ALL: new best val=0.6489 — saved
2026-04-26 01:32:18,854 INFO train_multi TF=ALL epoch 16/50 train=0.6468 val=0.6353
2026-04-26 01:32:18,858 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:32:18,858 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:32:18,858 INFO train_multi TF=ALL: new best val=0.6353 — saved
2026-04-26 01:32:30,393 INFO train_multi TF=ALL epoch 17/50 train=0.6380 val=0.6318
2026-04-26 01:32:30,397 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:32:30,397 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:32:30,397 INFO train_multi TF=ALL: new best val=0.6318 — saved
2026-04-26 01:32:41,965 INFO train_multi TF=ALL epoch 18/50 train=0.6326 val=0.6287
2026-04-26 01:32:41,969 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:32:41,970 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:32:41,970 INFO train_multi TF=ALL: new best val=0.6287 — saved
2026-04-26 01:32:53,601 INFO train_multi TF=ALL epoch 19/50 train=0.6280 val=0.6263
2026-04-26 01:32:53,605 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:32:53,605 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:32:53,605 INFO train_multi TF=ALL: new best val=0.6263 — saved
2026-04-26 01:33:05,225 INFO train_multi TF=ALL epoch 20/50 train=0.6247 val=0.6234
2026-04-26 01:33:05,229 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:33:05,230 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:33:05,230 INFO train_multi TF=ALL: new best val=0.6234 — saved
2026-04-26 01:33:16,843 INFO train_multi TF=ALL epoch 21/50 train=0.6214 val=0.6224
2026-04-26 01:33:16,847 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:33:16,848 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:33:16,848 INFO train_multi TF=ALL: new best val=0.6224 — saved
2026-04-26 01:33:28,383 INFO train_multi TF=ALL epoch 22/50 train=0.6190 val=0.6274
2026-04-26 01:33:39,935 INFO train_multi TF=ALL epoch 23/50 train=0.6167 val=0.6224
2026-04-26 01:33:39,939 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:33:39,939 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:33:39,939 INFO train_multi TF=ALL: new best val=0.6224 — saved
2026-04-26 01:33:51,452 INFO train_multi TF=ALL epoch 24/50 train=0.6150 val=0.6197
2026-04-26 01:33:51,456 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:33:51,456 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:33:51,457 INFO train_multi TF=ALL: new best val=0.6197 — saved
2026-04-26 01:34:03,009 INFO train_multi TF=ALL epoch 25/50 train=0.6124 val=0.6203
2026-04-26 01:34:14,530 INFO train_multi TF=ALL epoch 26/50 train=0.6105 val=0.6201
2026-04-26 01:34:26,072 INFO train_multi TF=ALL epoch 27/50 train=0.6087 val=0.6189
2026-04-26 01:34:26,077 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:34:26,077 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:34:26,077 INFO train_multi TF=ALL: new best val=0.6189 — saved
2026-04-26 01:34:37,633 INFO train_multi TF=ALL epoch 28/50 train=0.6069 val=0.6173
2026-04-26 01:34:37,637 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:34:37,637 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:34:37,637 INFO train_multi TF=ALL: new best val=0.6173 — saved
2026-04-26 01:34:49,264 INFO train_multi TF=ALL epoch 29/50 train=0.6052 val=0.6185
2026-04-26 01:35:00,821 INFO train_multi TF=ALL epoch 30/50 train=0.6034 val=0.6162
2026-04-26 01:35:00,825 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:35:00,826 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:35:00,826 INFO train_multi TF=ALL: new best val=0.6162 — saved
2026-04-26 01:35:12,331 INFO train_multi TF=ALL epoch 31/50 train=0.6023 val=0.6175
2026-04-26 01:35:23,977 INFO train_multi TF=ALL epoch 32/50 train=0.6004 val=0.6141
2026-04-26 01:35:23,981 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:35:23,981 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:35:23,981 INFO train_multi TF=ALL: new best val=0.6141 — saved
2026-04-26 01:35:35,558 INFO train_multi TF=ALL epoch 33/50 train=0.5994 val=0.6237
2026-04-26 01:35:47,072 INFO train_multi TF=ALL epoch 34/50 train=0.5979 val=0.6141
2026-04-26 01:35:47,077 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:35:47,077 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:35:47,077 INFO train_multi TF=ALL: new best val=0.6141 — saved
2026-04-26 01:35:58,563 INFO train_multi TF=ALL epoch 35/50 train=0.5965 val=0.6137
2026-04-26 01:35:58,568 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 01:35:58,568 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:35:58,568 INFO train_multi TF=ALL: new best val=0.6137 — saved
2026-04-26 01:36:10,162 INFO train_multi TF=ALL epoch 36/50 train=0.5954 val=0.6211
2026-04-26 01:36:21,708 INFO train_multi TF=ALL epoch 37/50 train=0.5941 val=0.6152
2026-04-26 01:36:33,202 INFO train_multi TF=ALL epoch 38/50 train=0.5927 val=0.6165
2026-04-26 01:36:44,880 INFO train_multi TF=ALL epoch 39/50 train=0.5915 val=0.6190
2026-04-26 01:36:56,432 INFO train_multi TF=ALL epoch 40/50 train=0.5899 val=0.6202
2026-04-26 01:36:56,432 INFO train_multi TF=ALL early stop at epoch 40
2026-04-26 01:36:56,570 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-26 01:36:56,575 INFO Loading faiss with AVX512 support.
2026-04-26 01:36:56,597 INFO Successfully loaded faiss with AVX512 support.
2026-04-26 01:36:56,602 INFO VectorStore: CPU FAISS index built (dim=74)
2026-04-26 01:36:56,603 INFO VectorStore: CPU FAISS index built (dim=34)
2026-04-26 01:36:56,603 INFO VectorStore: CPU FAISS index built (dim=64)
2026-04-26 01:36:57,384 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-26 01:36:57,394 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:36:57,408 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:36:57,409 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:36:57,411 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:37:31,464 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:37:31,467 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:37:31,469 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:37:31,473 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:37:31,477 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:37:33,390 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:37:33,403 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:37:33,409 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:37:33,412 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:37:33,413 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:37:33,420 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:37:33,425 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:37:33,429 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:37:33,431 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:37:33,432 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:37:34,543 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:37:34,549 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:37:34,550 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:37:34,550 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:37:34,555 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:41:22,079 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:41:22,130 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:41:22,152 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:41:33,149 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:41:55,621 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:41:55,625 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:41:55,626 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:41:55,627 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:41:55,628 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:41:56,298 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:41:56,302 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:41:56,304 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:41:56,306 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:41:56,307 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:41:58,989 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 01:41:58,994 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 01:41:58,996 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 01:41:58,997 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 01:41:59,003 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 01:42:00,610 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:42:00,615 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:42:00,618 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:42:00,619 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:42:00,621 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:45:45,244 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:45:45,251 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-26 01:45:45,336 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-26 01:46:15,149 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:46:15,177 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:46:15,189 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:46:15,214 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:46:15,220 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:46:15,258 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:46:15,261 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:46:15,262 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:46:15,263 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:46:15,265 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 01:47:55,614 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 01:47:55,629 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 01:47:55,630 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 01:47:55,631 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 01:47:55,631 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 01:49:55,411 INFO VectorStore trade_patterns: +50000 vectors for AUDUSD
2026-04-26 01:49:55,509 INFO VectorStore market_structures: +65447 vectors (34-dim 4H) for AUDUSD
2026-04-26 01:49:56,033 INFO VectorStore regime_embeddings: +13090 vectors for AUDUSD
2026-04-26 01:49:56,209 INFO VectorStore trade_patterns: +50000 vectors for EURGBP
2026-04-26 01:49:56,227 INFO VectorStore market_structures: +65448 vectors (34-dim 4H) for EURGBP
2026-04-26 01:49:56,646 INFO VectorStore regime_embeddings: +13090 vectors for EURGBP
2026-04-26 01:49:56,826 INFO VectorStore trade_patterns: +50000 vectors for EURJPY
2026-04-26 01:49:56,845 INFO VectorStore market_structures: +65453 vectors (34-dim 4H) for EURJPY
2026-04-26 01:49:57,266 INFO VectorStore regime_embeddings: +13091 vectors for EURJPY
2026-04-26 01:49:57,449 INFO VectorStore trade_patterns: +50000 vectors for EURUSD
2026-04-26 01:49:57,468 INFO VectorStore market_structures: +65451 vectors (34-dim 4H) for EURUSD
2026-04-26 01:49:57,892 INFO VectorStore regime_embeddings: +13091 vectors for EURUSD
2026-04-26 01:49:58,078 INFO VectorStore trade_patterns: +50000 vectors for GBPJPY
2026-04-26 01:49:58,098 INFO VectorStore market_structures: +65452 vectors (34-dim 4H) for GBPJPY
2026-04-26 01:49:58,517 INFO VectorStore regime_embeddings: +13091 vectors for GBPJPY
2026-04-26 01:49:58,705 INFO VectorStore trade_patterns: +50000 vectors for GBPUSD
2026-04-26 01:49:58,724 INFO VectorStore market_structures: +65451 vectors (34-dim 4H) for GBPUSD
2026-04-26 01:49:59,145 INFO VectorStore regime_embeddings: +13091 vectors for GBPUSD
2026-04-26 01:49:59,337 INFO VectorStore trade_patterns: +50000 vectors for NZDUSD
2026-04-26 01:49:59,357 INFO VectorStore market_structures: +65451 vectors (34-dim 4H) for NZDUSD
2026-04-26 01:49:59,783 INFO VectorStore regime_embeddings: +13091 vectors for NZDUSD
2026-04-26 01:49:59,974 INFO VectorStore trade_patterns: +50000 vectors for USDCAD
2026-04-26 01:49:59,992 INFO VectorStore market_structures: +65452 vectors (34-dim 4H) for USDCAD
2026-04-26 01:50:00,415 INFO VectorStore regime_embeddings: +13091 vectors for USDCAD
2026-04-26 01:50:00,610 INFO VectorStore trade_patterns: +50000 vectors for USDCHF
2026-04-26 01:50:00,629 INFO VectorStore market_structures: +65454 vectors (34-dim 4H) for USDCHF
2026-04-26 01:50:01,044 INFO VectorStore regime_embeddings: +13091 vectors for USDCHF
2026-04-26 01:50:01,239 INFO VectorStore trade_patterns: +50000 vectors for USDJPY
2026-04-26 01:50:01,260 INFO VectorStore market_structures: +65461 vectors (34-dim 4H) for USDJPY
2026-04-26 01:50:01,691 INFO VectorStore regime_embeddings: +13093 vectors for USDJPY
2026-04-26 01:50:01,889 INFO VectorStore trade_patterns: +50000 vectors for XAUUSD
2026-04-26 01:50:01,905 INFO VectorStore market_structures: +59006 vectors (34-dim 4H) for XAUUSD
2026-04-26 01:50:02,318 INFO VectorStore regime_embeddings: +12828 vectors for XAUUSD
2026-04-26 01:50:05,007 INFO VectorStore: saved 2670790 total vectors to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/vector_store
2026-04-26 01:50:05,007 INFO VectorStore saved: {'trade_patterns': 1100000, 'market_structures': 1427052, 'regime_embeddings': 143738}
2026-04-26 01:50:05,273 INFO Retrain complete.
2026-04-26 01:50:07,227 INFO Model gru: SUCCESS
2026-04-26 01:50:07,227 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 01:50:07,227 WARNING   [MISSING] regime_classifier → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_classifier.pkl
2026-04-26 01:50:07,227 WARNING   [MISSING] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-26 01:50:07,227 INFO   [OK] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-26 01:50:07,227 WARNING Missing weights: ['regime_classifier', 'quality_scorer'] — run retrain_incremental.py for each
2026-04-26 01:50:07,228 INFO Saved 18 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  Cleared journal for fresh Round 1 run
  START Round 1 - Backtest (val)
2026-04-26 01:50:07,979 INFO === STEP 6: BACKTEST + REINFORCED TRAINING (3 rounds, window=round1) ===
2026-04-26 01:50:07,981 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-04-26 01:50:07,981 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-26 01:50:07,981 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)