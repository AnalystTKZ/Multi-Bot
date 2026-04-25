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
2026-04-25 00:04:53,173 INFO Loading feature-engineered data...
2026-04-25 00:04:53,838 INFO Loaded 221743 rows, 202 features
2026-04-25 00:04:53,840 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-25 00:04:53,842 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-25 00:04:53,842 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-25 00:04:53,842 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-25 00:04:53,842 INFO No leakage confirmed: train < val < test timestamps

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
2026-04-25 00:04:56,237 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-25 00:04:56,238 INFO --- Training regime ---
2026-04-25 00:04:56,238 INFO Running retrain --model regime
2026-04-25 00:04:56,491 INFO retrain environment: KAGGLE
2026-04-25 00:04:58,150 INFO Device: CUDA (2 GPU(s))
2026-04-25 00:04:58,161 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-25 00:04:58,161 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-25 00:04:58,162 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-25 00:04:58,309 INFO NumExpr defaulting to 4 threads.
2026-04-25 00:04:58,519 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-25 00:04:58,519 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-25 00:04:58,519 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-25 00:04:58,731 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-25 00:04:58,732 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:58,825 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:58,900 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:58,976 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,053 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,126 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,198 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,271 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,345 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,420 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,508 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-25 00:04:59,571 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-25 00:04:59,588 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,590 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,605 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,607 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,622 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,624 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,639 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,642 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,658 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,661 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,677 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,680 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,695 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,697 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,712 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,715 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,732 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,735 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,751 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,754 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:04:59,772 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-25 00:04:59,780 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-25 00:05:01,005 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-25 00:05:23,664 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-25 00:05:23,668 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias)
2026-04-25 00:05:23,668 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-25 00:05:33,186 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-25 00:05:33,190 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias)
2026-04-25 00:05:33,194 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-25 00:05:40,688 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-25 00:05:40,689 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias)
2026-04-25 00:05:40,689 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-25 00:06:50,381 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-25 00:06:50,388 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour)
2026-04-25 00:06:50,388 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-25 00:07:21,023 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-25 00:07:21,024 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour)
2026-04-25 00:07:21,025 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-25 00:07:42,260 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-25 00:07:42,261 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour)
2026-04-25 00:07:42,374 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-25 00:07:42,376 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:07:42,377 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:07:42,378 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:07:42,379 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:07:42,380 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:07:42,381 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:07:42,382 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:07:42,384 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:07:42,385 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:07:42,386 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:07:42,387 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-25 00:07:42,514 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:42,562 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:42,563 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:42,563 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:42,573 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:42,574 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:07:45,320 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-25 00:07:45,321 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-25 00:07:45,499 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:45,533 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:45,534 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:45,534 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:45,543 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:45,544 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:07:48,215 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-25 00:07:48,216 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-25 00:07:48,390 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:48,427 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:48,428 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:48,429 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:48,437 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:48,438 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:07:51,078 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-25 00:07:51,079 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-25 00:07:51,252 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:51,287 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:51,288 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:51,289 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:51,297 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:51,298 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:07:54,035 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-25 00:07:54,036 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-25 00:07:54,205 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:54,241 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:54,242 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:54,242 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:54,251 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:54,252 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:07:56,909 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-25 00:07:56,910 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-25 00:07:57,082 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:57,118 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:57,119 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:57,120 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:57,129 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:07:57,130 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:07:59,780 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-25 00:07:59,782 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-25 00:07:59,937 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-25 00:07:59,966 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-25 00:07:59,967 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-25 00:07:59,967 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-25 00:07:59,975 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-25 00:07:59,976 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:08:02,662 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-25 00:08:02,663 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-25 00:08:02,828 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:02,861 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:02,861 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:02,862 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:02,871 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:02,872 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:08:05,601 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-25 00:08:05,601 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-25 00:08:05,770 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:05,805 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:05,806 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:05,807 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:05,816 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:05,817 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:08:08,509 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-25 00:08:08,510 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-25 00:08:08,690 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:08,726 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:08,727 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:08,727 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:08,737 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:08,738 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:08:11,531 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-25 00:08:11,532 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-25 00:08:11,808 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-25 00:08:11,873 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-25 00:08:11,874 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-25 00:08:11,874 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-25 00:08:11,886 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-25 00:08:11,888 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-25 00:08:18,243 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-25 00:08:18,245 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-25 00:08:18,450 INFO RegimeClassifier[mode=htf_bias]: 103290 samples, classes={'BIAS_UP': 18622, 'BIAS_DOWN': 18286, 'BIAS_NEUTRAL': 66382}, device=cuda
2026-04-25 00:08:18,451 INFO RegimeClassifier: sample weights — mean=0.360  ambiguous(<0.4)=69.5%
2026-04-25 00:08:18,735 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-04-25 00:08:18,736 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-25 00:08:23,672 INFO Regime epoch  1/50 — tr=0.3391 va=2.3634 acc=0.138 per_class={'BIAS_UP': 0.366, 'BIAS_DOWN': 0.1, 'BIAS_NEUTRAL': 0.067}
2026-04-25 00:08:23,841 INFO Regime epoch  2/50 — tr=0.3344 va=2.3983 acc=0.190
2026-04-25 00:08:24,014 INFO Regime epoch  3/50 — tr=0.3258 va=2.3731 acc=0.273
2026-04-25 00:08:24,188 INFO Regime epoch  4/50 — tr=0.3135 va=2.3121 acc=0.373
2026-04-25 00:08:24,372 INFO Regime epoch  5/50 — tr=0.3007 va=2.2204 acc=0.451 per_class={'BIAS_UP': 0.919, 'BIAS_DOWN': 0.912, 'BIAS_NEUTRAL': 0.167}
2026-04-25 00:08:24,545 INFO Regime epoch  6/50 — tr=0.2902 va=2.1050 acc=0.501
2026-04-25 00:08:24,722 INFO Regime epoch  7/50 — tr=0.2811 va=2.0422 acc=0.518
2026-04-25 00:08:24,893 INFO Regime epoch  8/50 — tr=0.2748 va=1.9732 acc=0.539
2026-04-25 00:08:25,072 INFO Regime epoch  9/50 — tr=0.2692 va=1.9284 acc=0.558
2026-04-25 00:08:25,271 INFO Regime epoch 10/50 — tr=0.2647 va=1.9013 acc=0.571 per_class={'BIAS_UP': 0.988, 'BIAS_DOWN': 0.991, 'BIAS_NEUTRAL': 0.316}
2026-04-25 00:08:25,453 INFO Regime epoch 11/50 — tr=0.2614 va=1.8777 acc=0.582
2026-04-25 00:08:25,628 INFO Regime epoch 12/50 — tr=0.2585 va=1.8650 acc=0.594
2026-04-25 00:08:25,799 INFO Regime epoch 13/50 — tr=0.2568 va=1.8602 acc=0.597
2026-04-25 00:08:25,975 INFO Regime epoch 14/50 — tr=0.2546 va=1.8413 acc=0.615
2026-04-25 00:08:26,163 INFO Regime epoch 15/50 — tr=0.2532 va=1.8375 acc=0.617 per_class={'BIAS_UP': 0.984, 'BIAS_DOWN': 0.99, 'BIAS_NEUTRAL': 0.391}
2026-04-25 00:08:26,351 INFO Regime epoch 16/50 — tr=0.2518 va=1.8311 acc=0.624
2026-04-25 00:08:26,521 INFO Regime epoch 17/50 — tr=0.2509 va=1.8274 acc=0.626
2026-04-25 00:08:26,707 INFO Regime epoch 18/50 — tr=0.2501 va=1.8200 acc=0.632
2026-04-25 00:08:26,884 INFO Regime epoch 19/50 — tr=0.2492 va=1.8157 acc=0.638
2026-04-25 00:08:27,082 INFO Regime epoch 20/50 — tr=0.2485 va=1.8114 acc=0.644 per_class={'BIAS_UP': 0.969, 'BIAS_DOWN': 0.989, 'BIAS_NEUTRAL': 0.44}
2026-04-25 00:08:27,261 INFO Regime epoch 21/50 — tr=0.2479 va=1.8085 acc=0.643
2026-04-25 00:08:27,433 INFO Regime epoch 22/50 — tr=0.2474 va=1.8029 acc=0.647
2026-04-25 00:08:27,611 INFO Regime epoch 23/50 — tr=0.2470 va=1.8012 acc=0.649
2026-04-25 00:08:27,783 INFO Regime epoch 24/50 — tr=0.2466 va=1.7953 acc=0.653
2026-04-25 00:08:27,970 INFO Regime epoch 25/50 — tr=0.2462 va=1.7958 acc=0.655 per_class={'BIAS_UP': 0.971, 'BIAS_DOWN': 0.99, 'BIAS_NEUTRAL': 0.457}
2026-04-25 00:08:28,139 INFO Regime epoch 26/50 — tr=0.2460 va=1.7926 acc=0.659
2026-04-25 00:08:28,315 INFO Regime epoch 27/50 — tr=0.2457 va=1.7892 acc=0.662
2026-04-25 00:08:28,493 INFO Regime epoch 28/50 — tr=0.2454 va=1.7897 acc=0.660
2026-04-25 00:08:28,666 INFO Regime epoch 29/50 — tr=0.2452 va=1.7853 acc=0.665
2026-04-25 00:08:28,851 INFO Regime epoch 30/50 — tr=0.2450 va=1.7865 acc=0.664 per_class={'BIAS_UP': 0.969, 'BIAS_DOWN': 0.99, 'BIAS_NEUTRAL': 0.472}
2026-04-25 00:08:29,024 INFO Regime epoch 31/50 — tr=0.2448 va=1.7838 acc=0.668
2026-04-25 00:08:29,193 INFO Regime epoch 32/50 — tr=0.2448 va=1.7814 acc=0.670
2026-04-25 00:08:29,364 INFO Regime epoch 33/50 — tr=0.2447 va=1.7808 acc=0.669
2026-04-25 00:08:29,548 INFO Regime epoch 34/50 — tr=0.2443 va=1.7830 acc=0.667
2026-04-25 00:08:29,770 INFO Regime epoch 35/50 — tr=0.2443 va=1.7805 acc=0.669 per_class={'BIAS_UP': 0.967, 'BIAS_DOWN': 0.99, 'BIAS_NEUTRAL': 0.482}
2026-04-25 00:08:29,969 INFO Regime epoch 36/50 — tr=0.2442 va=1.7808 acc=0.670
2026-04-25 00:08:30,160 INFO Regime epoch 37/50 — tr=0.2442 va=1.7790 acc=0.671
2026-04-25 00:08:30,352 INFO Regime epoch 38/50 — tr=0.2441 va=1.7766 acc=0.674
2026-04-25 00:08:30,534 INFO Regime epoch 39/50 — tr=0.2441 va=1.7757 acc=0.675
2026-04-25 00:08:30,725 INFO Regime epoch 40/50 — tr=0.2439 va=1.7793 acc=0.672 per_class={'BIAS_UP': 0.965, 'BIAS_DOWN': 0.99, 'BIAS_NEUTRAL': 0.487}
2026-04-25 00:08:30,903 INFO Regime epoch 41/50 — tr=0.2440 va=1.7756 acc=0.674
2026-04-25 00:08:31,078 INFO Regime epoch 42/50 — tr=0.2439 va=1.7753 acc=0.676
2026-04-25 00:08:31,254 INFO Regime epoch 43/50 — tr=0.2439 va=1.7780 acc=0.671
2026-04-25 00:08:31,433 INFO Regime epoch 44/50 — tr=0.2437 va=1.7769 acc=0.673
2026-04-25 00:08:31,627 INFO Regime epoch 45/50 — tr=0.2438 va=1.7742 acc=0.676 per_class={'BIAS_UP': 0.965, 'BIAS_DOWN': 0.988, 'BIAS_NEUTRAL': 0.493}
2026-04-25 00:08:31,808 INFO Regime epoch 46/50 — tr=0.2438 va=1.7794 acc=0.671
2026-04-25 00:08:31,997 INFO Regime epoch 47/50 — tr=0.2438 va=1.7746 acc=0.677
2026-04-25 00:08:32,179 INFO Regime epoch 48/50 — tr=0.2438 va=1.7764 acc=0.674
2026-04-25 00:08:32,359 INFO Regime epoch 49/50 — tr=0.2438 va=1.7762 acc=0.674
2026-04-25 00:08:32,558 INFO Regime epoch 50/50 — tr=0.2438 va=1.7767 acc=0.673 per_class={'BIAS_UP': 0.966, 'BIAS_DOWN': 0.99, 'BIAS_NEUTRAL': 0.489}
2026-04-25 00:08:32,576 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-25 00:08:32,576 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-25 00:08:32,707 INFO Regime HTF complete: acc=0.676, n=103290
2026-04-25 00:08:32,709 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-25 00:08:32,863 INFO Rule labels LTF_BEHAVIOUR [4H]: {'TRENDING': 8970, 'RANGING': 4592, 'CONSOLIDATING': 2247, 'VOLATILE': 4008}  ambiguous=4539 (total=19817)  short_runs_zeroed=570
2026-04-25 00:08:32,870 INFO Regime[4H] persistence (avg bars/run) on XAUUSD 4H:
{0: 12.121621621621621, 1: 6.4858757062146895, 2: 5.575682382133995, 3: 10.329896907216495}
2026-04-25 00:08:32,871 INFO Regime[4H] return separation on XAUUSD 4H:
{0: 5.9502411776840314e-05, 1: 5.664222849552333e-05, 2: -6.138442068010657e-05, 3: 4.239843820627722e-05}
2026-04-25 00:08:32,871 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-25 00:08:32,873 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:08:32,875 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:08:32,876 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:08:32,878 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:08:32,879 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:08:32,881 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:08:32,882 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:08:32,884 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:08:32,885 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:08:32,887 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:08:32,890 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-25 00:08:32,900 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:32,903 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:32,903 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:32,903 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:32,904 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:32,905 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:08:42,806 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-25 00:08:42,809 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-25 00:08:42,957 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:42,960 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:42,960 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:42,961 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:42,961 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:42,963 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:08:52,820 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-25 00:08:52,823 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-25 00:08:52,970 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:52,972 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:52,973 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:52,973 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:52,974 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:08:52,975 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:09:02,917 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-25 00:09:02,920 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-25 00:09:03,055 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:03,057 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:03,058 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:03,058 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:03,059 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:03,060 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:09:12,750 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-25 00:09:12,753 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-25 00:09:12,893 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:12,895 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:12,896 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:12,896 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:12,897 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:12,899 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:09:22,572 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-25 00:09:22,575 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-25 00:09:22,723 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:22,726 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:22,726 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:22,727 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:22,727 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:22,729 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:09:32,480 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-25 00:09:32,483 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-25 00:09:32,623 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-25 00:09:32,625 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-25 00:09:32,626 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-25 00:09:32,626 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-25 00:09:32,626 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-25 00:09:32,628 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:09:42,255 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-25 00:09:42,257 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-25 00:09:42,402 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:42,404 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:42,405 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:42,406 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:42,406 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:42,408 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:09:52,172 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-25 00:09:52,175 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-25 00:09:52,327 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:52,330 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:52,331 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:52,331 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:52,331 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:09:52,333 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:10:02,109 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-25 00:10:02,112 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-25 00:10:02,260 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:10:02,262 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:10:02,263 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:10:02,264 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:10:02,264 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:10:02,266 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:10:11,905 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-25 00:10:11,908 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-25 00:10:12,061 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-25 00:10:12,065 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-25 00:10:12,066 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-25 00:10:12,067 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-25 00:10:12,067 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-25 00:10:12,070 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-25 00:10:34,345 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-25 00:10:34,351 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-25 00:10:34,716 INFO RegimeClassifier[mode=ltf_behaviour]: 401471 samples, classes={'TRENDING': 160094, 'RANGING': 105286, 'CONSOLIDATING': 53524, 'VOLATILE': 82567}, device=cuda
2026-04-25 00:10:34,717 INFO RegimeClassifier: sample weights — mean=0.505  ambiguous(<0.4)=33.0%
2026-04-25 00:10:34,719 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-04-25 00:10:34,719 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-25 00:10:35,416 INFO Regime epoch  1/50 — tr=0.6510 va=1.9967 acc=0.367 per_class={'TRENDING': 0.413, 'RANGING': 0.254, 'CONSOLIDATING': 0.119, 'VOLATILE': 0.593}
2026-04-25 00:10:36,076 INFO Regime epoch  2/50 — tr=0.6258 va=1.9452 acc=0.440
2026-04-25 00:10:36,719 INFO Regime epoch  3/50 — tr=0.5927 va=1.8953 acc=0.491
2026-04-25 00:10:37,385 INFO Regime epoch  4/50 — tr=0.5628 va=1.8474 acc=0.521
2026-04-25 00:10:38,110 INFO Regime epoch  5/50 — tr=0.5428 va=1.8169 acc=0.541 per_class={'TRENDING': 0.557, 'RANGING': 0.1, 'CONSOLIDATING': 0.756, 'VOLATILE': 0.939}
2026-04-25 00:10:38,760 INFO Regime epoch  6/50 — tr=0.5305 va=1.8086 acc=0.555
2026-04-25 00:10:39,408 INFO Regime epoch  7/50 — tr=0.5216 va=1.8016 acc=0.567
2026-04-25 00:10:40,082 INFO Regime epoch  8/50 — tr=0.5147 va=1.7969 acc=0.578
2026-04-25 00:10:40,750 INFO Regime epoch  9/50 — tr=0.5098 va=1.7874 acc=0.591
2026-04-25 00:10:41,458 INFO Regime epoch 10/50 — tr=0.5055 va=1.7750 acc=0.605 per_class={'TRENDING': 0.67, 'RANGING': 0.148, 'CONSOLIDATING': 0.806, 'VOLATILE': 0.938}
2026-04-25 00:10:42,114 INFO Regime epoch 11/50 — tr=0.5016 va=1.7599 acc=0.618
2026-04-25 00:10:42,747 INFO Regime epoch 12/50 — tr=0.4986 va=1.7456 acc=0.629
2026-04-25 00:10:43,407 INFO Regime epoch 13/50 — tr=0.4955 va=1.7295 acc=0.641
2026-04-25 00:10:44,077 INFO Regime epoch 14/50 — tr=0.4931 va=1.7130 acc=0.650
2026-04-25 00:10:44,781 INFO Regime epoch 15/50 — tr=0.4907 va=1.6998 acc=0.656 per_class={'TRENDING': 0.767, 'RANGING': 0.185, 'CONSOLIDATING': 0.84, 'VOLATILE': 0.928}
2026-04-25 00:10:45,452 INFO Regime epoch 16/50 — tr=0.4891 va=1.6866 acc=0.664
2026-04-25 00:10:46,153 INFO Regime epoch 17/50 — tr=0.4873 va=1.6763 acc=0.670
2026-04-25 00:10:46,786 INFO Regime epoch 18/50 — tr=0.4859 va=1.6658 acc=0.674
2026-04-25 00:10:47,427 INFO Regime epoch 19/50 — tr=0.4847 va=1.6569 acc=0.680
2026-04-25 00:10:48,115 INFO Regime epoch 20/50 — tr=0.4833 va=1.6463 acc=0.684 per_class={'TRENDING': 0.793, 'RANGING': 0.232, 'CONSOLIDATING': 0.905, 'VOLATILE': 0.911}
2026-04-25 00:10:48,759 INFO Regime epoch 21/50 — tr=0.4823 va=1.6372 acc=0.690
2026-04-25 00:10:49,402 INFO Regime epoch 22/50 — tr=0.4812 va=1.6331 acc=0.693
2026-04-25 00:10:50,070 INFO Regime epoch 23/50 — tr=0.4807 va=1.6272 acc=0.698
2026-04-25 00:10:50,732 INFO Regime epoch 24/50 — tr=0.4799 va=1.6236 acc=0.699
2026-04-25 00:10:51,448 INFO Regime epoch 25/50 — tr=0.4792 va=1.6164 acc=0.703 per_class={'TRENDING': 0.803, 'RANGING': 0.277, 'CONSOLIDATING': 0.922, 'VOLATILE': 0.917}
2026-04-25 00:10:52,117 INFO Regime epoch 26/50 — tr=0.4788 va=1.6141 acc=0.703
2026-04-25 00:10:52,750 INFO Regime epoch 27/50 — tr=0.4783 va=1.6085 acc=0.706
2026-04-25 00:10:53,410 INFO Regime epoch 28/50 — tr=0.4778 va=1.6101 acc=0.708
2026-04-25 00:10:54,105 INFO Regime epoch 29/50 — tr=0.4774 va=1.6051 acc=0.708
2026-04-25 00:10:54,821 INFO Regime epoch 30/50 — tr=0.4770 va=1.6049 acc=0.710 per_class={'TRENDING': 0.814, 'RANGING': 0.284, 'CONSOLIDATING': 0.931, 'VOLATILE': 0.912}
2026-04-25 00:10:55,461 INFO Regime epoch 31/50 — tr=0.4768 va=1.6024 acc=0.711
2026-04-25 00:10:56,117 INFO Regime epoch 32/50 — tr=0.4765 va=1.5981 acc=0.713
2026-04-25 00:10:56,737 INFO Regime epoch 33/50 — tr=0.4764 va=1.5986 acc=0.713
2026-04-25 00:10:57,374 INFO Regime epoch 34/50 — tr=0.4761 va=1.5959 acc=0.714
2026-04-25 00:10:58,102 INFO Regime epoch 35/50 — tr=0.4757 va=1.5963 acc=0.713 per_class={'TRENDING': 0.81, 'RANGING': 0.3, 'CONSOLIDATING': 0.936, 'VOLATILE': 0.912}
2026-04-25 00:10:58,728 INFO Regime epoch 36/50 — tr=0.4756 va=1.5965 acc=0.715
2026-04-25 00:10:59,359 INFO Regime epoch 37/50 — tr=0.4757 va=1.5928 acc=0.716
2026-04-25 00:10:59,992 INFO Regime epoch 38/50 — tr=0.4754 va=1.5940 acc=0.715
2026-04-25 00:11:00,618 INFO Regime epoch 39/50 — tr=0.4753 va=1.5943 acc=0.716
2026-04-25 00:11:01,296 INFO Regime epoch 40/50 — tr=0.4752 va=1.5929 acc=0.715 per_class={'TRENDING': 0.823, 'RANGING': 0.294, 'CONSOLIDATING': 0.941, 'VOLATILE': 0.901}
2026-04-25 00:11:01,926 INFO Regime epoch 41/50 — tr=0.4752 va=1.5917 acc=0.717
2026-04-25 00:11:02,569 INFO Regime epoch 42/50 — tr=0.4751 va=1.5930 acc=0.717
2026-04-25 00:11:03,217 INFO Regime epoch 43/50 — tr=0.4752 va=1.5932 acc=0.716
2026-04-25 00:11:03,888 INFO Regime epoch 44/50 — tr=0.4751 va=1.5937 acc=0.716
2026-04-25 00:11:04,585 INFO Regime epoch 45/50 — tr=0.4751 va=1.5925 acc=0.717 per_class={'TRENDING': 0.826, 'RANGING': 0.296, 'CONSOLIDATING': 0.938, 'VOLATILE': 0.9}
2026-04-25 00:11:05,238 INFO Regime epoch 46/50 — tr=0.4750 va=1.5944 acc=0.715
2026-04-25 00:11:05,890 INFO Regime epoch 47/50 — tr=0.4750 va=1.5939 acc=0.716
2026-04-25 00:11:06,550 INFO Regime epoch 48/50 — tr=0.4750 va=1.5920 acc=0.717
2026-04-25 00:11:07,215 INFO Regime epoch 49/50 — tr=0.4750 va=1.5924 acc=0.717
2026-04-25 00:11:07,936 INFO Regime epoch 50/50 — tr=0.4751 va=1.5925 acc=0.717 per_class={'TRENDING': 0.822, 'RANGING': 0.301, 'CONSOLIDATING': 0.938, 'VOLATILE': 0.905}
2026-04-25 00:11:07,984 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-25 00:11:07,984 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-25 00:11:08,119 INFO Regime LTF complete: acc=0.717, n=401471
2026-04-25 00:11:08,122 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-25 00:11:08,621 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-25 00:11:08,625 INFO Regime[1H] persistence (avg bars/run) on XAUUSD 1H:
{0: 9.355291913830783, 1: 6.096303199751476, 2: 5.598885793871866, 3: 6.771351107094442}
2026-04-25 00:11:08,628 INFO Regime[1H] return separation on XAUUSD 1H:
{0: 2.149062259083323e-05, 1: -3.254493608872846e-06, 2: -3.039756158146958e-06, 3: 1.8102764458593193e-05}
2026-04-25 00:11:08,641 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-25 00:11:08,642 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-25 00:11:08,642 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-25 00:11:08,642 INFO === VectorStore: building similarity indices ===
2026-04-25 00:11:08,646 INFO Loading faiss with AVX512 support.
2026-04-25 00:11:08,670 INFO Successfully loaded faiss with AVX512 support.
2026-04-25 00:11:08,677 INFO VectorStore: CPU FAISS index built (dim=74)
2026-04-25 00:11:08,677 INFO VectorStore: CPU FAISS index built (dim=53)
2026-04-25 00:11:08,677 INFO VectorStore: CPU FAISS index built (dim=64)
2026-04-25 00:11:08,752 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-25 00:11:08,759 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:11:14,847 INFO VectorStore trade_patterns: +50000 vectors for AUDUSD
2026-04-25 00:11:14,996 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:11:14,999 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:11:15,000 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:11:15,000 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:11:15,000 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:11:52,476 WARNING VectorStore market_structures failed for AUDUSD: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 00:11:58,673 INFO VectorStore regime_embeddings: +13090 vectors for AUDUSD
2026-04-25 00:11:58,962 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:12:05,026 INFO VectorStore trade_patterns: +50000 vectors for EURGBP
2026-04-25 00:12:05,168 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:12:05,171 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:12:05,172 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:12:05,172 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:12:05,172 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:12:42,892 WARNING VectorStore market_structures failed for EURGBP: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 00:12:49,058 INFO VectorStore regime_embeddings: +13090 vectors for EURGBP
2026-04-25 00:12:49,338 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:12:55,521 INFO VectorStore trade_patterns: +50000 vectors for EURJPY
2026-04-25 00:12:55,666 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:12:55,669 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:12:55,670 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:12:55,670 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:12:55,670 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:13:33,492 WARNING VectorStore market_structures failed for EURJPY: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 00:13:39,620 INFO VectorStore regime_embeddings: +13091 vectors for EURJPY
2026-04-25 00:13:39,928 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:13:46,039 INFO VectorStore trade_patterns: +50000 vectors for EURUSD
2026-04-25 00:13:46,197 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:13:46,199 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:13:46,200 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:13:46,200 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:13:46,200 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:14:24,415 WARNING VectorStore market_structures failed for EURUSD: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 00:14:30,535 INFO VectorStore regime_embeddings: +13091 vectors for EURUSD
2026-04-25 00:14:30,810 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:14:37,103 INFO VectorStore trade_patterns: +50000 vectors for GBPJPY
2026-04-25 00:14:37,263 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:14:37,265 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:14:37,266 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:14:37,266 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:14:37,267 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:15:15,548 WARNING VectorStore market_structures failed for GBPJPY: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 00:15:21,961 INFO VectorStore regime_embeddings: +13091 vectors for GBPJPY
2026-04-25 00:15:22,261 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:15:28,277 INFO VectorStore trade_patterns: +50000 vectors for GBPUSD
2026-04-25 00:15:28,421 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:15:28,423 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:15:28,424 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:15:28,424 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:15:28,424 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:16:06,292 WARNING VectorStore market_structures failed for GBPUSD: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 00:16:12,416 INFO VectorStore regime_embeddings: +13091 vectors for GBPUSD
2026-04-25 00:16:12,731 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:16:18,856 INFO VectorStore trade_patterns: +50000 vectors for NZDUSD
2026-04-25 00:16:19,002 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-25 00:16:19,004 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-25 00:16:19,005 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-25 00:16:19,005 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-25 00:16:19,006 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-25 00:16:56,921 WARNING VectorStore market_structures failed for NZDUSD: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 00:17:03,085 INFO VectorStore regime_embeddings: +13091 vectors for NZDUSD
2026-04-25 00:17:03,369 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:17:09,455 INFO VectorStore trade_patterns: +50000 vectors for USDCAD
2026-04-25 00:17:09,616 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:17:09,618 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:17:09,619 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:17:09,620 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:17:09,620 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:17:48,111 WARNING VectorStore market_structures failed for USDCAD: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 00:17:54,234 INFO VectorStore regime_embeddings: +13091 vectors for USDCAD
2026-04-25 00:17:54,543 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:18:00,785 INFO VectorStore trade_patterns: +50000 vectors for USDCHF
2026-04-25 00:18:00,952 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:18:00,955 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:18:00,955 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:18:00,956 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:18:00,956 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:18:39,270 WARNING VectorStore market_structures failed for USDCHF: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 00:18:45,788 INFO VectorStore regime_embeddings: +13091 vectors for USDCHF
2026-04-25 00:18:46,103 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:18:52,242 INFO VectorStore trade_patterns: +50000 vectors for USDJPY
2026-04-25 00:18:52,394 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:18:52,397 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:18:52,397 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:18:52,398 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:18:52,398 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:19:30,963 WARNING VectorStore market_structures failed for USDJPY: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 00:19:37,406 INFO VectorStore regime_embeddings: +13093 vectors for USDJPY
2026-04-25 00:19:37,722 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-25 00:19:50,780 INFO VectorStore trade_patterns: +50000 vectors for XAUUSD
2026-04-25 00:19:50,957 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-25 00:19:50,961 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-25 00:19:50,962 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-25 00:19:50,962 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-25 00:19:50,963 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-25 00:21:14,964 WARNING VectorStore market_structures failed for XAUUSD: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 00:21:28,242 INFO VectorStore regime_embeddings: +12828 vectors for XAUUSD
2026-04-25 00:21:29,010 INFO VectorStore: saved 693738 total vectors to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/vector_store
2026-04-25 00:21:29,011 INFO VectorStore saved: {'trade_patterns': 550000, 'market_structures': 0, 'regime_embeddings': 143738}
2026-04-25 00:21:29,080 INFO Retrain complete.
2026-04-25 00:21:31,812 INFO Model regime: SUCCESS
2026-04-25 00:21:31,812 INFO --- Training gru ---
2026-04-25 00:21:31,813 INFO Running retrain --model gru
2026-04-25 00:21:32,408 INFO retrain environment: KAGGLE
2026-04-25 00:21:34,092 INFO Device: CUDA (2 GPU(s))
2026-04-25 00:21:34,104 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-25 00:21:34,104 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-25 00:21:34,106 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-25 00:21:34,253 INFO NumExpr defaulting to 4 threads.
2026-04-25 00:21:34,453 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-25 00:21:34,453 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-25 00:21:34,453 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-25 00:21:34,698 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-25 00:21:34,941 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-25 00:21:34,943 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:35,023 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:35,101 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:35,176 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:35,248 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:35,320 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:35,389 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:35,461 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:35,540 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:35,615 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:35,705 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-25 00:21:35,768 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-25 00:21:35,769 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260425_002135
2026-04-25 00:21:35,773 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-04-25 00:21:35,893 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:35,893 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:35,908 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:35,917 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:35,918 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-25 00:21:35,918 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-25 00:21:35,919 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-25 00:21:35,919 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:35,998 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-25 00:21:36,000 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:36,231 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-25 00:21:36,260 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:36,544 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:36,682 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:36,785 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:36,977 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:36,978 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:36,994 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:37,001 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:37,002 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:37,078 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-25 00:21:37,080 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:37,311 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-25 00:21:37,326 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:37,597 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:37,736 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:37,841 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:38,040 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:38,041 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:38,058 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:38,065 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:38,066 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:38,139 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-25 00:21:38,141 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:38,375 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-25 00:21:38,391 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:38,658 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:38,802 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:38,915 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:39,104 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:39,105 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:39,121 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:39,128 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:39,129 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:39,221 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-25 00:21:39,223 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:39,504 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-25 00:21:39,529 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:39,826 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:39,969 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:40,078 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:40,268 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:40,269 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:40,286 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:40,295 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:40,297 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:40,375 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-25 00:21:40,377 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:40,617 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-25 00:21:40,633 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:40,904 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:41,041 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:41,150 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:41,334 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:41,334 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:41,350 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:41,358 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:41,359 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:41,435 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-25 00:21:41,437 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:41,683 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-25 00:21:41,699 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:41,972 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:42,116 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:42,223 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:42,391 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-25 00:21:42,392 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-25 00:21:42,406 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-25 00:21:42,413 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-25 00:21:42,414 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:42,490 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-25 00:21:42,492 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:42,729 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-25 00:21:42,742 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:43,013 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:43,150 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:43,259 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:43,450 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:43,451 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:43,472 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:43,482 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:43,483 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:43,577 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-25 00:21:43,579 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:43,817 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-25 00:21:43,835 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:44,101 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:44,235 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:44,342 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:44,537 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:44,538 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:44,554 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:44,562 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:44,563 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:44,636 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-25 00:21:44,638 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:44,875 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-25 00:21:44,891 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:45,176 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:45,319 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:45,430 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:45,637 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:45,638 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:45,654 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:45,661 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:21:45,662 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:45,737 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-25 00:21:45,739 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:45,984 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-25 00:21:46,000 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:46,287 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:46,430 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:46,544 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:21:46,845 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-25 00:21:46,846 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-25 00:21:46,864 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-25 00:21:46,877 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-25 00:21:46,878 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-25 00:21:47,036 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-25 00:21:47,039 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-25 00:21:47,533 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-25 00:21:47,579 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-25 00:21:48,123 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-25 00:21:48,333 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-25 00:21:48,465 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-25 00:21:48,581 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-25 00:21:48,582 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-25 00:21:48,582 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-25 00:26:17,006 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-25 00:26:17,006 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-25 00:26:18,356 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-25 00:26:42,768 INFO train_multi TF=ALL epoch 1/50 train=0.6009 val=0.6138
2026-04-25 00:26:42,774 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 00:26:42,774 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 00:26:42,774 INFO train_multi TF=ALL: new best val=0.6138 — saved
2026-04-25 00:27:00,836 INFO train_multi TF=ALL epoch 2/50 train=0.6009 val=0.6142
2026-04-25 00:27:18,993 INFO train_multi TF=ALL epoch 3/50 train=0.6007 val=0.6142
2026-04-25 00:27:37,077 INFO train_multi TF=ALL epoch 4/50 train=0.6003 val=0.6142
2026-04-25 00:27:55,292 INFO train_multi TF=ALL epoch 5/50 train=0.6004 val=0.6147
2026-04-25 00:28:13,399 INFO train_multi TF=ALL epoch 6/50 train=0.6000 val=0.6147
2026-04-25 00:28:13,400 INFO train_multi TF=ALL early stop at epoch 6
2026-04-25 00:28:13,566 INFO === VectorStore: building similarity indices ===
2026-04-25 00:28:13,570 INFO Loading faiss with AVX512 support.
2026-04-25 00:28:13,593 INFO Successfully loaded faiss with AVX512 support.
2026-04-25 00:28:13,598 INFO VectorStore: CPU FAISS index built (dim=74)
2026-04-25 00:28:13,598 INFO VectorStore: CPU FAISS index built (dim=53)
2026-04-25 00:28:13,598 INFO VectorStore: CPU FAISS index built (dim=64)
2026-04-25 00:28:13,989 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-25 00:28:13,995 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:28:20,082 INFO VectorStore trade_patterns: +50000 vectors for AUDUSD
2026-04-25 00:28:20,206 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:28:20,209 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:28:20,210 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:28:20,210 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:28:20,210 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:28:59,611 WARNING VectorStore market_structures failed for AUDUSD: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 00:29:05,649 INFO VectorStore regime_embeddings: +13090 vectors for AUDUSD
2026-04-25 00:29:05,891 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:29:11,874 INFO VectorStore trade_patterns: +50000 vectors for EURGBP
2026-04-25 00:29:12,001 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:29:12,003 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:29:12,004 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:29:12,005 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:29:12,005 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:29:51,603 WARNING VectorStore market_structures failed for EURGBP: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 00:29:57,587 INFO VectorStore regime_embeddings: +13090 vectors for EURGBP
2026-04-25 00:29:57,820 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:30:03,673 INFO VectorStore trade_patterns: +50000 vectors for EURJPY
2026-04-25 00:30:03,803 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:30:03,805 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:30:03,806 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:30:03,807 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:30:03,807 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:30:43,017 WARNING VectorStore market_structures failed for EURJPY: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 00:30:49,321 INFO VectorStore regime_embeddings: +13091 vectors for EURJPY
2026-04-25 00:30:49,569 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:30:55,555 INFO VectorStore trade_patterns: +50000 vectors for EURUSD
2026-04-25 00:30:55,691 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:30:55,693 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:30:55,694 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:30:55,694 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:30:55,695 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:31:35,290 WARNING VectorStore market_structures failed for EURUSD: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 00:31:41,304 INFO VectorStore regime_embeddings: +13091 vectors for EURUSD
2026-04-25 00:31:41,546 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:31:47,551 INFO VectorStore trade_patterns: +50000 vectors for GBPJPY
2026-04-25 00:31:47,691 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:31:47,694 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:31:47,695 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:31:47,695 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:31:47,696 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:32:26,844 WARNING VectorStore market_structures failed for GBPJPY: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 00:32:33,055 INFO VectorStore regime_embeddings: +13091 vectors for GBPJPY
2026-04-25 00:32:33,299 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:32:39,514 INFO VectorStore trade_patterns: +50000 vectors for GBPUSD
2026-04-25 00:32:39,657 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:32:39,659 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:32:39,660 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:32:39,660 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:32:39,660 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:33:19,408 WARNING VectorStore market_structures failed for GBPUSD: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 00:33:25,459 INFO VectorStore regime_embeddings: +13091 vectors for GBPUSD
2026-04-25 00:33:25,735 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:33:31,628 INFO VectorStore trade_patterns: +50000 vectors for NZDUSD
2026-04-25 00:33:31,768 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-25 00:33:31,770 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-25 00:33:31,771 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-25 00:33:31,771 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-25 00:33:31,772 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-25 00:34:11,045 WARNING VectorStore market_structures failed for NZDUSD: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 00:34:17,296 INFO VectorStore regime_embeddings: +13091 vectors for NZDUSD
2026-04-25 00:34:17,575 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:34:23,555 INFO VectorStore trade_patterns: +50000 vectors for USDCAD
2026-04-25 00:34:23,688 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:34:23,690 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:34:23,691 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:34:23,692 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:34:23,692 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:35:03,641 WARNING VectorStore market_structures failed for USDCAD: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 00:35:09,829 INFO VectorStore regime_embeddings: +13091 vectors for USDCAD
2026-04-25 00:35:10,066 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:35:16,102 INFO VectorStore trade_patterns: +50000 vectors for USDCHF
2026-04-25 00:35:16,247 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:35:16,250 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:35:16,251 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:35:16,251 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:35:16,251 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:35:56,013 WARNING VectorStore market_structures failed for USDCHF: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 00:36:02,065 INFO VectorStore regime_embeddings: +13091 vectors for USDCHF
2026-04-25 00:36:02,313 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-25 00:36:08,205 INFO VectorStore trade_patterns: +50000 vectors for USDJPY
2026-04-25 00:36:08,352 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:36:08,354 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:36:08,355 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:36:08,356 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:36:08,356 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 00:36:47,968 WARNING VectorStore market_structures failed for USDJPY: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 00:36:53,942 INFO VectorStore regime_embeddings: +13093 vectors for USDJPY
2026-04-25 00:36:54,190 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-25 00:37:06,645 INFO VectorStore trade_patterns: +50000 vectors for XAUUSD
2026-04-25 00:37:06,779 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-25 00:37:06,783 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-25 00:37:06,784 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-25 00:37:06,785 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-25 00:37:06,785 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-25 00:38:33,215 WARNING VectorStore market_structures failed for XAUUSD: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 00:38:45,964 INFO VectorStore regime_embeddings: +12828 vectors for XAUUSD
2026-04-25 00:38:47,525 INFO VectorStore: saved 1387476 total vectors to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/vector_store
2026-04-25 00:38:47,526 INFO VectorStore saved: {'trade_patterns': 1100000, 'market_structures': 0, 'regime_embeddings': 287476}
2026-04-25 00:38:47,654 INFO Retrain complete.
2026-04-25 00:38:49,703 INFO Model gru: SUCCESS
2026-04-25 00:38:49,703 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 00:38:49,704 WARNING   [MISSING] regime_classifier → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_classifier.pkl
2026-04-25 00:38:49,704 INFO   [OK] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-25 00:38:49,704 WARNING   [MISSING] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-25 00:38:49,704 WARNING Missing weights: ['regime_classifier', 'rl_ppo'] — run retrain_incremental.py for each
2026-04-25 00:38:49,705 INFO Saved 18 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  Cleared journal for fresh Round 1 run
  START Round 1 - Backtest (val)
2026-04-25 00:38:50,261 INFO === STEP 6: BACKTEST + REINFORCED TRAINING (3 rounds, window=round1) ===
2026-04-25 00:38:50,261 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-04-25 00:38:50,262 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-25 00:38:50,262 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260425_003850.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)             2129  53.8%   2.77 1970.9% 53.8% 16.8%   3.2%    5.15
  gate_diagnostics: bars=466496 no_signal=50515 quality_block=0 session_skip=192823 density=4977 pm_reject=34740 daily_skip=171259 cooldown=10053

Calibration Summary:
  all          [OK] Calibration OK — p_win correlates with actual win rate.
  ml_trader    [OK] Calibration OK — p_win correlates with actual win rate.
2026-04-25 00:44:49,734 INFO Round 1 backtest — 2129 trades | avg WR=53.8% | avg PF=2.77 | avg Sharpe=5.15
2026-04-25 00:44:49,734 INFO   ml_trader: 2129 trades | WR=53.8% | PF=2.77 | Return=1970.9% | DD=3.2% | Sharpe=5.15
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 2129
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2129 rows)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  AUDUSD        190 trades   160 days   1.19/day
  EURGBP        182 trades   146 days   1.25/day
  EURJPY        145 trades   114 days   1.27/day
  EURUSD        335 trades   299 days   1.12/day
  GBPJPY        153 trades   120 days   1.27/day
  GBPUSD        244 trades   202 days   1.21/day
  NZDUSD        149 trades   120 days   1.24/day
  USDCAD        192 trades   155 days   1.24/day
  USDCHF        192 trades   155 days   1.24/day
  USDJPY        223 trades   189 days   1.18/day
  XAUUSD        124 trades   100 days   1.24/day
  ✓  All symbols within normal range.
2026-04-25 00:44:50,690 INFO Round 1: wrote 2129 journal entries (total in file: 2129)
2026-04-25 00:44:50,692 INFO Round 1 — retraining regime...
2026-04-25 01:01:29,095 INFO Retrain regime: OK
2026-04-25 01:01:29,114 INFO Round 1 — retraining quality...
2026-04-25 01:01:36,400 INFO Retrain quality: OK
2026-04-25 01:01:36,416 INFO Round 1 — retraining rl...
2026-04-25 01:02:38,143 INFO Retrain rl: OK
2026-04-25 01:02:38,162 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-25 01:02:38,162 INFO Round 2 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
2026-04-25 01:08:53,215 INFO Round 2 backtest — 2178 trades | avg WR=53.8% | avg PF=2.80 | avg Sharpe=5.19
2026-04-25 01:08:53,215 INFO   ml_trader: 2178 trades | WR=53.8% | PF=2.80 | Return=1994.0% | DD=2.4% | Sharpe=5.19
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 2178
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2178 rows)
2026-04-25 01:08:54,237 INFO Round 2: wrote 2178 journal entries (total in file: 4307)
2026-04-25 01:08:54,239 INFO Round 2 — retraining regime...
2026-04-25 01:25:43,469 INFO Retrain regime: OK
2026-04-25 01:25:43,494 INFO Round 2 — retraining quality...
2026-04-25 01:25:54,084 INFO Retrain quality: OK
2026-04-25 01:25:54,103 INFO Round 2 — retraining rl...
2026-04-25 01:27:17,974 INFO Retrain rl: OK
2026-04-25 01:27:17,992 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-04-25 01:27:17,993 INFO Round 3 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
2026-04-25 01:33:29,392 INFO Round 3 backtest — 2145 trades | avg WR=54.2% | avg PF=2.76 | avg Sharpe=5.15
2026-04-25 01:33:29,392 INFO   ml_trader: 2145 trades | WR=54.2% | PF=2.76 | Return=2017.4% | DD=2.3% | Sharpe=5.15
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 2145
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2145 rows)
2026-04-25 01:33:30,359 INFO Round 3: wrote 2145 journal entries (total in file: 6452)
2026-04-25 01:33:30,361 INFO Round 3 (final): retraining after last backtest...
2026-04-25 01:33:30,361 INFO Round 3 — retraining regime...
2026-04-25 01:50:31,136 INFO Retrain regime: OK
2026-04-25 01:50:31,154 INFO Round 3 — retraining quality...
2026-04-25 01:50:40,588 INFO Retrain quality: OK
2026-04-25 01:50:40,605 INFO Round 3 — retraining rl...
2026-04-25 01:52:38,601 INFO Retrain rl: OK
2026-04-25 01:52:38,620 INFO Improvement round 1 → 3: WR +0.4% | PF -0.005 | Sharpe +0.006
2026-04-25 01:52:38,777 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-25 01:52:38,784 INFO Journal entries: 6452
2026-04-25 01:52:38,784 INFO --- Training quality ---
2026-04-25 01:52:38,784 INFO Running retrain --model quality
  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 6452 entries

=== Round 1 → Retrain Quality + RL ===
  START Round 1 - Quality+RL retrain
2026-04-25 01:52:39,171 INFO retrain environment: KAGGLE
2026-04-25 01:52:40,822 INFO Device: CUDA (2 GPU(s))
2026-04-25 01:52:40,834 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-25 01:52:40,834 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-25 01:52:40,835 INFO === QualityScorer retrain ===
2026-04-25 01:52:40,977 INFO NumExpr defaulting to 4 threads.
2026-04-25 01:52:41,185 INFO QualityScorer: CUDA available — using GPU
2026-04-25 01:52:41,394 INFO QualityScorer loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (device=cuda)
2026-04-25 01:52:41,631 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/quality_scorer.pkl_20260425_015241
2026-04-25 01:52:41,850 INFO QualityScorer: 6452 samples, EV stats={'mean': 0.2351999431848526, 'std': 1.2371082305908203, 'n_pos': 3479, 'n_neg': 2973}, device=cuda
2026-04-25 01:52:41,851 INFO QualityScorer: normalised win labels by median_win=0.836 — EV range now [-1, +3]
2026-04-25 01:52:41,851 INFO QualityScorer: warm start from existing weights
2026-04-25 01:52:41,851 INFO QualityScorer: pos_weight=1.00 (n_pos=2784 n_neg=2377)
2026-04-25 01:52:43,617 INFO Quality epoch   1/100 — va_huber=0.7563
2026-04-25 01:52:43,758 INFO Quality epoch   2/100 — va_huber=0.7580
2026-04-25 01:52:43,881 INFO Quality epoch   3/100 — va_huber=0.7582
2026-04-25 01:52:44,006 INFO Quality epoch   4/100 — va_huber=0.7572
2026-04-25 01:52:44,131 INFO Quality epoch   5/100 — va_huber=0.7559
2026-04-25 01:52:44,883 INFO Quality epoch  11/100 — va_huber=0.7551
2026-04-25 01:52:46,124 INFO Quality epoch  21/100 — va_huber=0.7561
2026-04-25 01:52:46,249 INFO Quality early stop at epoch 22
2026-04-25 01:52:46,270 INFO QualityScorer EV model: MAE=1.232 dir_acc=0.569 n_val=1291
2026-04-25 01:52:46,274 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-25 01:52:46,340 INFO Retrain complete.
2026-04-25 01:52:47,221 INFO Model quality: SUCCESS
2026-04-25 01:52:47,222 INFO --- Training rl ---
2026-04-25 01:52:47,222 INFO Running retrain --model rl
2026-04-25 01:52:47,484 INFO retrain environment: KAGGLE
2026-04-25 01:52:49,112 INFO Device: CUDA (2 GPU(s))
2026-04-25 01:52:49,124 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-25 01:52:49,124 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-25 01:52:49,125 INFO === RLAgent (PPO) retrain ===
2026-04-25 01:52:49,128 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260425_015249
2026-04-25 01:52:49.976534: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777081969.999864   53870 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777081970.007483   53870 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777081970.027160   53870 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777081970.027190   53870 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777081970.027196   53870 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777081970.027198   53870 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-25 01:52:54,478 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-25 01:52:57,302 INFO RLAgent: PPO model loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-25 01:52:57,474 INFO RLAgent: warm start — fine-tuning existing PPO policy (lr=6.00e-05)
2026-04-25 01:54:55,902 INFO RLAgent: retrain complete, 6452 episodes
2026-04-25 01:54:55,903 INFO Retrain complete.
2026-04-25 01:54:57,537 INFO Model rl: SUCCESS
2026-04-25 01:54:57,537 INFO Step 7b complete — summary: /kaggle/working/Multi-Bot/trading-system/ml_training/metrics/training_7b_summary.json
  DONE  Round 1 - Quality+RL retrain

=== Round 2: BLIND backtest on test window (unseen 2yr) ===
  START Round 2 - Blind backtest (test)
2026-04-25 01:54:58,035 INFO === STEP 6: BACKTEST + REINFORCED TRAINING (3 rounds, window=round2) ===
2026-04-25 01:54:58,035 INFO BT_WINDOW=round2 — BLIND backtest: 2023-08-07 → 2025-08-05 (test set)
2026-04-25 01:54:58,037 INFO Cleared existing journal for fresh reinforced training run
2026-04-25 01:54:58,037 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-25 01:54:58,037 INFO Round 1 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-25 02:01:24,368 INFO Round 1 backtest — 2092 trades | avg WR=52.0% | avg PF=2.49 | avg Sharpe=4.87
2026-04-25 02:01:24,368 INFO   ml_trader: 2092 trades | WR=52.0% | PF=2.49 | Return=1373.1% | DD=2.2% | Sharpe=4.87
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 2092
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2092 rows)
2026-04-25 02:01:25,328 INFO Round 1: wrote 2092 journal entries (total in file: 2092)
2026-04-25 02:01:25,329 INFO Round 1 — retraining regime...
2026-04-25 02:17:50,161 INFO Retrain regime: OK
2026-04-25 02:17:50,180 INFO Round 1 — retraining quality...
2026-04-25 02:17:56,561 INFO Retrain quality: OK
2026-04-25 02:17:56,576 INFO Round 1 — retraining rl...
2026-04-25 02:18:42,771 INFO Retrain rl: OK
2026-04-25 02:18:42,789 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-25 02:18:42,789 INFO Round 2 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-25 02:24:57,021 INFO Round 2 backtest — 2106 trades | avg WR=52.0% | avg PF=2.46 | avg Sharpe=4.83
2026-04-25 02:24:57,021 INFO   ml_trader: 2106 trades | WR=52.0% | PF=2.46 | Return=1327.6% | DD=2.2% | Sharpe=4.83
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 2106
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2106 rows)
2026-04-25 02:24:57,983 INFO Round 2: wrote 2106 journal entries (total in file: 4198)
2026-04-25 02:24:57,984 INFO Round 2 — retraining regime...
2026-04-25 02:41:14,388 INFO Retrain regime: OK
2026-04-25 02:41:14,409 INFO Round 2 — retraining quality...
2026-04-25 02:41:22,989 INFO Retrain quality: OK
2026-04-25 02:41:23,006 INFO Round 2 — retraining rl...
2026-04-25 02:42:46,284 INFO Retrain rl: OK
2026-04-25 02:42:46,301 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-04-25 02:42:46,301 INFO Round 3 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-25 02:48:56,889 INFO Round 3 backtest — 2110 trades | avg WR=51.6% | avg PF=2.46 | avg Sharpe=4.82
2026-04-25 02:48:56,889 INFO   ml_trader: 2110 trades | WR=51.6% | PF=2.46 | Return=1294.9% | DD=2.2% | Sharpe=4.82
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 2110
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2110 rows)
2026-04-25 02:48:57,847 INFO Round 3: wrote 2110 journal entries (total in file: 6308)
2026-04-25 02:48:57,849 INFO Round 3 (final): retraining after last backtest...
2026-04-25 02:48:57,849 INFO Round 3 — retraining regime...
2026-04-25 03:05:16,955 INFO Retrain regime: OK
2026-04-25 03:05:16,972 INFO Round 3 — retraining quality...
2026-04-25 03:05:30,965 INFO Retrain quality: OK
2026-04-25 03:05:30,982 INFO Round 3 — retraining rl...
2026-04-25 03:07:34,119 INFO Retrain rl: OK
2026-04-25 03:07:34,139 INFO Improvement round 1 → 3: WR -0.4% | PF -0.030 | Sharpe -0.046
2026-04-25 03:07:34,307 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-25 03:07:34,314 INFO Journal entries: 6308
2026-04-25 03:07:34,314 INFO --- Training quality ---
2026-04-25 03:07:34,315 INFO Running retrain --model quality
  DONE  Round 2 - Blind backtest (test)
  Saved Round 2 result → round2_summary.json
  Journal after Round 2: 6308 entries

=== Round 2 → Retrain Quality + RL (Round 1+2 journal) ===
  START Round 2 - Quality+RL retrain
2026-04-25 03:07:34,845 INFO retrain environment: KAGGLE
2026-04-25 03:07:36,495 INFO Device: CUDA (2 GPU(s))
2026-04-25 03:07:36,507 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-25 03:07:36,507 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-25 03:07:36,508 INFO === QualityScorer retrain ===
2026-04-25 03:07:36,647 INFO NumExpr defaulting to 4 threads.
2026-04-25 03:07:36,853 INFO QualityScorer: CUDA available — using GPU
2026-04-25 03:07:37,061 INFO QualityScorer loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (device=cuda)
2026-04-25 03:07:37,291 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/quality_scorer.pkl_20260425_030737
2026-04-25 03:07:37,514 INFO QualityScorer: 6308 samples, EV stats={'mean': 0.15640868246555328, 'std': 1.1945221424102783, 'n_pos': 3272, 'n_neg': 3036}, device=cuda
2026-04-25 03:07:37,515 INFO QualityScorer: normalised win labels by median_win=0.828 — EV range now [-1, +3]
2026-04-25 03:07:37,515 INFO QualityScorer: warm start from existing weights
2026-04-25 03:07:37,516 INFO QualityScorer: pos_weight=1.00 (n_pos=2611 n_neg=2435)
2026-04-25 03:07:39,224 INFO Quality epoch   1/100 — va_huber=0.7043
2026-04-25 03:07:39,350 INFO Quality epoch   2/100 — va_huber=0.7041
2026-04-25 03:07:39,465 INFO Quality epoch   3/100 — va_huber=0.7037
2026-04-25 03:07:39,583 INFO Quality epoch   4/100 — va_huber=0.7042
2026-04-25 03:07:39,702 INFO Quality epoch   5/100 — va_huber=0.7042
2026-04-25 03:07:40,426 INFO Quality epoch  11/100 — va_huber=0.7032
2026-04-25 03:07:41,617 INFO Quality epoch  21/100 — va_huber=0.7031
2026-04-25 03:07:42,778 INFO Quality epoch  31/100 — va_huber=0.7023
2026-04-25 03:07:44,075 INFO Quality epoch  41/100 — va_huber=0.7019
2026-04-25 03:07:45,257 INFO Quality epoch  51/100 — va_huber=0.7014
2026-04-25 03:07:46,431 INFO Quality epoch  61/100 — va_huber=0.7011
2026-04-25 03:07:46,782 INFO Quality early stop at epoch 64
2026-04-25 03:07:46,799 INFO QualityScorer EV model: MAE=1.167 dir_acc=0.577 n_val=1262
2026-04-25 03:07:46,802 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-25 03:07:46,874 INFO Retrain complete.
2026-04-25 03:07:47,822 INFO Model quality: SUCCESS
2026-04-25 03:07:47,822 INFO --- Training rl ---
2026-04-25 03:07:47,822 INFO Running retrain --model rl
2026-04-25 03:07:48,094 INFO retrain environment: KAGGLE
2026-04-25 03:07:49,764 INFO Device: CUDA (2 GPU(s))
2026-04-25 03:07:49,775 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-25 03:07:49,775 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-25 03:07:49,777 INFO === RLAgent (PPO) retrain ===
2026-04-25 03:07:49,779 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260425_030749
2026-04-25 03:07:50.640655: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777086470.668207   72048 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777086470.677126   72048 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777086470.702081   72048 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777086470.702154   72048 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777086470.702162   72048 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777086470.702168   72048 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-25 03:07:55,221 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-25 03:07:58,057 INFO RLAgent: PPO model loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-25 03:07:58,219 INFO RLAgent: warm start — fine-tuning existing PPO policy (lr=6.00e-05)
2026-04-25 03:09:54,766 INFO RLAgent: retrain complete, 6308 episodes
2026-04-25 03:09:54,766 INFO Retrain complete.
2026-04-25 03:09:56,480 INFO Model rl: SUCCESS
2026-04-25 03:09:56,481 INFO Step 7b complete — summary: /kaggle/working/Multi-Bot/trading-system/ml_training/metrics/training_7b_summary.json
  DONE  Round 2 - Quality+RL retrain

=== Round 3: Incremental retrain ALL models on full data ===
  START Retrain gru [full-data retrain]
2026-04-25 03:09:56,963 INFO retrain environment: KAGGLE
2026-04-25 03:09:58,607 INFO Device: CUDA (2 GPU(s))
2026-04-25 03:09:58,619 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-25 03:09:58,619 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-25 03:09:58,620 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-25 03:09:58,756 INFO NumExpr defaulting to 4 threads.
2026-04-25 03:09:58,959 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-25 03:09:58,960 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-25 03:09:58,960 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-25 03:09:59,207 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-25 03:09:59,455 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-25 03:09:59,456 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:09:59,542 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:09:59,622 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:09:59,697 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:09:59,769 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:09:59,842 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:09:59,911 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:09:59,985 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:00,060 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:00,137 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:00,231 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-25 03:10:00,295 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-25 03:10:00,297 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260425_031000
2026-04-25 03:10:00,301 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-04-25 03:10:00,418 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:00,418 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:00,434 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:00,449 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:00,450 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-25 03:10:00,450 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-25 03:10:00,451 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-25 03:10:00,451 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:00,527 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-25 03:10:00,529 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:00,762 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-25 03:10:00,793 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:01,087 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:01,226 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:01,339 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:01,549 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:01,550 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:01,568 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:01,576 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:01,577 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:01,659 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-25 03:10:01,661 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:01,889 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-25 03:10:01,904 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:02,186 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:02,334 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:02,446 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:02,636 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:02,637 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:02,654 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:02,662 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:02,663 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:02,737 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-25 03:10:02,739 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:02,984 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-25 03:10:02,999 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:03,281 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:03,433 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:03,554 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:03,740 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:03,741 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:03,758 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:03,767 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:03,768 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:03,840 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-25 03:10:03,842 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:04,077 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-25 03:10:04,100 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:04,394 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:04,536 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:04,637 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:04,827 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:04,828 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:04,847 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:04,858 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:04,859 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:04,934 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-25 03:10:04,936 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:05,176 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-25 03:10:05,192 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:05,476 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:05,618 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:05,722 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:05,908 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:05,909 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:05,925 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:05,933 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:05,934 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:06,008 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-25 03:10:06,010 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:06,250 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-25 03:10:06,266 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:06,542 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:06,684 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:06,788 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:06,954 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-25 03:10:06,955 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-25 03:10:06,970 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-25 03:10:06,977 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-25 03:10:06,978 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:07,056 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-25 03:10:07,057 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:07,299 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-25 03:10:07,311 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:07,589 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:07,723 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:07,832 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:08,015 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:08,016 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:08,032 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:08,040 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:08,041 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:08,119 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-25 03:10:08,120 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:08,367 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-25 03:10:08,384 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:08,691 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:08,835 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:08,945 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:09,127 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:09,128 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:09,143 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:09,152 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:09,153 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:09,229 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-25 03:10:09,231 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:09,476 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-25 03:10:09,491 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:09,769 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:09,913 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:10,021 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:10,206 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:10,207 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:10,224 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:10,232 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:10:10,233 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:10,310 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-25 03:10:10,312 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:10,552 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-25 03:10:10,567 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:10,835 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:10,978 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:11,092 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:10:11,385 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-25 03:10:11,387 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-25 03:10:11,405 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-25 03:10:11,417 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-25 03:10:11,418 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-25 03:10:11,576 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-25 03:10:11,580 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-25 03:10:12,078 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-25 03:10:12,127 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-25 03:10:12,666 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-25 03:10:12,881 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-25 03:10:13,033 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-25 03:10:13,168 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-25 03:10:13,169 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-25 03:10:13,169 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-25 03:14:38,183 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-25 03:14:38,183 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-25 03:14:39,706 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-25 03:15:04,498 INFO train_multi TF=ALL epoch 1/50 train=0.6007 val=0.6141
2026-04-25 03:15:04,504 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 03:15:04,504 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 03:15:04,504 INFO train_multi TF=ALL: new best val=0.6141 — saved
2026-04-25 03:15:22,743 INFO train_multi TF=ALL epoch 2/50 train=0.6005 val=0.6141
2026-04-25 03:15:22,747 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 03:15:22,747 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 03:15:22,747 INFO train_multi TF=ALL: new best val=0.6141 — saved
2026-04-25 03:15:40,854 INFO train_multi TF=ALL epoch 3/50 train=0.6007 val=0.6144
2026-04-25 03:15:58,698 INFO train_multi TF=ALL epoch 4/50 train=0.6003 val=0.6142
2026-04-25 03:16:16,659 INFO train_multi TF=ALL epoch 5/50 train=0.6002 val=0.6143
2026-04-25 03:16:34,418 INFO train_multi TF=ALL epoch 6/50 train=0.6001 val=0.6147
2026-04-25 03:16:52,453 INFO train_multi TF=ALL epoch 7/50 train=0.6000 val=0.6157
2026-04-25 03:16:52,453 INFO train_multi TF=ALL early stop at epoch 7
2026-04-25 03:16:52,608 INFO === VectorStore: building similarity indices ===
2026-04-25 03:16:52,613 INFO Loading faiss with AVX512 support.
2026-04-25 03:16:52,637 INFO Successfully loaded faiss with AVX512 support.
2026-04-25 03:16:52,642 INFO VectorStore: CPU FAISS index built (dim=74)
2026-04-25 03:16:52,642 INFO VectorStore: CPU FAISS index built (dim=53)
2026-04-25 03:16:52,642 INFO VectorStore: CPU FAISS index built (dim=64)
2026-04-25 03:16:56,607 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-25 03:16:56,613 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:17:03,677 INFO VectorStore trade_patterns: +50000 vectors for AUDUSD
2026-04-25 03:17:03,868 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:17:03,870 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:17:03,871 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:17:03,872 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:17:03,872 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:17:42,827 WARNING VectorStore market_structures failed for AUDUSD: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 03:17:49,069 INFO VectorStore regime_embeddings: +13090 vectors for AUDUSD
2026-04-25 03:17:49,464 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:17:55,339 INFO VectorStore trade_patterns: +50000 vectors for EURGBP
2026-04-25 03:17:55,534 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:17:55,536 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:17:55,537 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:17:55,537 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:17:55,538 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:18:34,729 WARNING VectorStore market_structures failed for EURGBP: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 03:18:40,817 INFO VectorStore regime_embeddings: +13090 vectors for EURGBP
2026-04-25 03:18:41,174 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:18:47,067 INFO VectorStore trade_patterns: +50000 vectors for EURJPY
2026-04-25 03:18:47,267 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:18:47,269 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:18:47,270 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:18:47,271 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:18:47,271 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:19:26,090 WARNING VectorStore market_structures failed for EURJPY: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 03:19:32,121 INFO VectorStore regime_embeddings: +13091 vectors for EURJPY
2026-04-25 03:19:32,526 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:19:38,479 INFO VectorStore trade_patterns: +50000 vectors for EURUSD
2026-04-25 03:19:38,693 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:19:38,695 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:19:38,696 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:19:38,697 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:19:38,697 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:20:17,550 WARNING VectorStore market_structures failed for EURUSD: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 03:20:23,599 INFO VectorStore regime_embeddings: +13091 vectors for EURUSD
2026-04-25 03:20:23,990 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:20:29,862 INFO VectorStore trade_patterns: +50000 vectors for GBPJPY
2026-04-25 03:20:30,076 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:20:30,079 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:20:30,080 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:20:30,080 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:20:30,081 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:21:09,182 WARNING VectorStore market_structures failed for GBPJPY: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 03:21:15,500 INFO VectorStore regime_embeddings: +13091 vectors for GBPJPY
2026-04-25 03:21:15,879 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:21:21,802 INFO VectorStore trade_patterns: +50000 vectors for GBPUSD
2026-04-25 03:21:22,004 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:21:22,006 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:21:22,007 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:21:22,007 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:21:22,008 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:22:00,902 WARNING VectorStore market_structures failed for GBPUSD: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 03:22:06,947 INFO VectorStore regime_embeddings: +13091 vectors for GBPUSD
2026-04-25 03:22:07,318 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:22:13,256 INFO VectorStore trade_patterns: +50000 vectors for NZDUSD
2026-04-25 03:22:13,464 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-25 03:22:13,466 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-25 03:22:13,467 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-25 03:22:13,467 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-25 03:22:13,468 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-25 03:22:52,528 WARNING VectorStore market_structures failed for NZDUSD: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 03:22:58,553 INFO VectorStore regime_embeddings: +13091 vectors for NZDUSD
2026-04-25 03:22:58,931 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:23:04,832 INFO VectorStore trade_patterns: +50000 vectors for USDCAD
2026-04-25 03:23:05,040 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:23:05,042 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:23:05,043 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:23:05,044 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:23:05,044 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:23:44,274 WARNING VectorStore market_structures failed for USDCAD: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 03:23:50,248 INFO VectorStore regime_embeddings: +13091 vectors for USDCAD
2026-04-25 03:23:50,647 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:23:56,617 INFO VectorStore trade_patterns: +50000 vectors for USDCHF
2026-04-25 03:23:56,827 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:23:56,829 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:23:56,830 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:23:56,831 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:23:56,831 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:24:35,999 WARNING VectorStore market_structures failed for USDCHF: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 03:24:42,036 INFO VectorStore regime_embeddings: +13091 vectors for USDCHF
2026-04-25 03:24:42,452 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:24:48,324 INFO VectorStore trade_patterns: +50000 vectors for USDJPY
2026-04-25 03:24:48,525 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:24:48,527 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:24:48,528 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:24:48,529 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:24:48,529 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:25:27,944 WARNING VectorStore market_structures failed for USDJPY: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 03:25:34,153 INFO VectorStore regime_embeddings: +13093 vectors for USDJPY
2026-04-25 03:25:34,577 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-25 03:25:47,499 INFO VectorStore trade_patterns: +50000 vectors for XAUUSD
2026-04-25 03:25:47,705 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-25 03:25:47,709 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-25 03:25:47,710 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-25 03:25:47,711 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-25 03:25:47,711 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-25 03:27:14,433 WARNING VectorStore market_structures failed for XAUUSD: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 03:27:27,181 INFO VectorStore regime_embeddings: +12828 vectors for XAUUSD
2026-04-25 03:27:34,654 INFO VectorStore: saved 6243642 total vectors to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/vector_store
2026-04-25 03:27:34,654 INFO VectorStore saved: {'trade_patterns': 4950000, 'market_structures': 0, 'regime_embeddings': 1293642}
2026-04-25 03:27:35,280 INFO Retrain complete.
  DONE  Retrain gru [full-data retrain]
  START Retrain regime [full-data retrain]
2026-04-25 03:27:38,409 INFO retrain environment: KAGGLE
2026-04-25 03:27:40,659 INFO Device: CUDA (2 GPU(s))
2026-04-25 03:27:40,671 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-25 03:27:40,671 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-25 03:27:40,672 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-25 03:27:40,973 INFO NumExpr defaulting to 4 threads.
2026-04-25 03:27:41,209 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-25 03:27:41,210 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-25 03:27:41,210 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-25 03:27:41,458 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-25 03:27:41,460 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:41,542 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:41,621 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:41,699 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:41,775 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:41,847 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:41,919 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:41,993 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:42,068 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:42,142 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:42,236 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-25 03:27:42,296 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-25 03:27:42,312 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:42,314 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:42,328 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:42,329 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:42,345 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:42,347 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:42,362 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:42,364 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:42,381 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:42,384 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:42,399 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:42,402 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:42,416 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:42,419 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:42,433 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:42,437 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:42,452 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:42,455 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:42,470 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:42,474 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:27:42,491 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-25 03:27:42,499 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-25 03:27:43,470 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-25 03:28:05,294 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-25 03:28:05,299 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias)
2026-04-25 03:28:05,299 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-25 03:28:14,852 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-25 03:28:14,856 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias)
2026-04-25 03:28:14,856 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-25 03:28:22,551 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-25 03:28:22,554 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias)
2026-04-25 03:28:22,554 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-25 03:29:31,920 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-25 03:29:31,924 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour)
2026-04-25 03:29:31,924 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-25 03:30:02,523 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-25 03:30:02,525 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour)
2026-04-25 03:30:02,526 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-25 03:30:24,384 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-25 03:30:24,385 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour)
2026-04-25 03:30:24,502 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-25 03:30:24,504 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:30:24,505 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:30:24,506 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:30:24,507 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:30:24,508 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:30:24,510 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:30:24,510 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:30:24,511 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:30:24,512 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:30:24,513 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:30:24,515 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-25 03:30:24,648 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:24,694 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:24,695 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:24,695 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:24,705 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:24,706 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:30:27,501 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-25 03:30:27,502 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-25 03:30:27,682 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:27,716 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:27,717 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:27,717 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:27,726 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:27,727 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:30:30,461 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-25 03:30:30,462 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-25 03:30:30,651 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:30,688 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:30,689 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:30,689 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:30,698 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:30,699 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:30:33,447 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-25 03:30:33,448 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-25 03:30:33,642 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:33,677 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:33,678 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:33,679 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:33,687 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:33,688 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:30:36,498 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-25 03:30:36,499 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-25 03:30:36,689 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:36,725 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:36,726 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:36,727 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:36,736 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:36,737 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:30:39,524 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-25 03:30:39,526 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-25 03:30:39,696 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:39,730 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:39,731 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:39,731 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:39,740 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:39,741 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:30:42,543 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-25 03:30:42,545 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-25 03:30:42,720 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-25 03:30:42,753 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-25 03:30:42,754 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-25 03:30:42,754 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-25 03:30:42,763 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-25 03:30:42,764 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:30:45,606 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-25 03:30:45,607 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-25 03:30:45,779 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:45,815 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:45,816 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:45,816 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:45,828 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:45,829 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:30:48,623 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-25 03:30:48,624 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-25 03:30:48,813 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:48,848 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:48,849 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:48,849 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:48,858 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:48,859 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:30:51,602 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-25 03:30:51,603 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-25 03:30:51,776 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:51,811 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:51,812 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:51,812 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:51,821 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:30:51,822 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:30:54,637 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-25 03:30:54,638 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-25 03:30:54,925 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-25 03:30:54,981 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-25 03:30:54,983 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-25 03:30:54,983 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-25 03:30:54,994 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-25 03:30:54,995 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-25 03:31:01,427 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-25 03:31:01,429 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-25 03:31:01,602 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260425_033101
2026-04-25 03:31:01,810 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=34, n_classes=3)
2026-04-25 03:31:01,827 INFO RegimeClassifier[mode=htf_bias]: 103290 samples, classes={'BIAS_UP': 18622, 'BIAS_DOWN': 18286, 'BIAS_NEUTRAL': 66382}, device=cuda
2026-04-25 03:31:01,827 INFO RegimeClassifier: sample weights — mean=0.360  ambiguous(<0.4)=69.5%
2026-04-25 03:31:01,827 INFO RegimeClassifier[mode=htf_bias]: warm start from existing weights
2026-04-25 03:31:01,828 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-25 03:31:04,275 INFO Regime epoch  1/50 — tr=0.2392 va=1.7091 acc=0.758 per_class={'BIAS_UP': 0.967, 'BIAS_DOWN': 0.989, 'BIAS_NEUTRAL': 0.624}
2026-04-25 03:31:04,460 INFO Regime epoch  2/50 — tr=0.2393 va=1.7082 acc=0.758
2026-04-25 03:31:04,644 INFO Regime epoch  3/50 — tr=0.2392 va=1.7103 acc=0.756
2026-04-25 03:31:04,823 INFO Regime epoch  4/50 — tr=0.2392 va=1.7089 acc=0.757
2026-04-25 03:31:05,027 INFO Regime epoch  5/50 — tr=0.2393 va=1.7101 acc=0.757 per_class={'BIAS_UP': 0.968, 'BIAS_DOWN': 0.991, 'BIAS_NEUTRAL': 0.623}
2026-04-25 03:31:05,222 INFO Regime epoch  6/50 — tr=0.2393 va=1.7085 acc=0.759
2026-04-25 03:31:05,413 INFO Regime epoch  7/50 — tr=0.2393 va=1.7097 acc=0.756
2026-04-25 03:31:05,600 INFO Regime epoch  8/50 — tr=0.2393 va=1.7083 acc=0.759
2026-04-25 03:31:05,784 INFO Regime epoch  9/50 — tr=0.2392 va=1.7088 acc=0.760
2026-04-25 03:31:05,994 INFO Regime epoch 10/50 — tr=0.2393 va=1.7074 acc=0.759 per_class={'BIAS_UP': 0.969, 'BIAS_DOWN': 0.99, 'BIAS_NEUTRAL': 0.625}
2026-04-25 03:31:06,188 INFO Regime epoch 11/50 — tr=0.2392 va=1.7084 acc=0.758
2026-04-25 03:31:06,374 INFO Regime epoch 12/50 — tr=0.2393 va=1.7081 acc=0.758
2026-04-25 03:31:06,563 INFO Regime epoch 13/50 — tr=0.2391 va=1.7087 acc=0.758
2026-04-25 03:31:06,750 INFO Regime epoch 14/50 — tr=0.2392 va=1.7082 acc=0.759
2026-04-25 03:31:06,963 INFO Regime epoch 15/50 — tr=0.2392 va=1.7108 acc=0.754 per_class={'BIAS_UP': 0.972, 'BIAS_DOWN': 0.991, 'BIAS_NEUTRAL': 0.616}
2026-04-25 03:31:07,151 INFO Regime epoch 16/50 — tr=0.2391 va=1.7097 acc=0.758
2026-04-25 03:31:07,339 INFO Regime epoch 17/50 — tr=0.2391 va=1.7073 acc=0.760
2026-04-25 03:31:07,544 INFO Regime epoch 18/50 — tr=0.2392 va=1.7084 acc=0.759
2026-04-25 03:31:07,735 INFO Regime epoch 19/50 — tr=0.2391 va=1.7065 acc=0.762
2026-04-25 03:31:07,931 INFO Regime epoch 20/50 — tr=0.2391 va=1.7056 acc=0.762 per_class={'BIAS_UP': 0.968, 'BIAS_DOWN': 0.989, 'BIAS_NEUTRAL': 0.631}
2026-04-25 03:31:08,110 INFO Regime epoch 21/50 — tr=0.2390 va=1.7067 acc=0.759
2026-04-25 03:31:08,291 INFO Regime epoch 22/50 — tr=0.2391 va=1.7069 acc=0.760
2026-04-25 03:31:08,476 INFO Regime epoch 23/50 — tr=0.2392 va=1.7069 acc=0.758
2026-04-25 03:31:08,665 INFO Regime epoch 24/50 — tr=0.2390 va=1.7078 acc=0.758
2026-04-25 03:31:08,865 INFO Regime epoch 25/50 — tr=0.2390 va=1.7078 acc=0.759 per_class={'BIAS_UP': 0.969, 'BIAS_DOWN': 0.991, 'BIAS_NEUTRAL': 0.625}
2026-04-25 03:31:09,064 INFO Regime epoch 26/50 — tr=0.2390 va=1.7072 acc=0.759
2026-04-25 03:31:09,245 INFO Regime epoch 27/50 — tr=0.2391 va=1.7051 acc=0.762
2026-04-25 03:31:09,421 INFO Regime epoch 28/50 — tr=0.2391 va=1.7066 acc=0.760
2026-04-25 03:31:09,608 INFO Regime epoch 29/50 — tr=0.2390 va=1.7046 acc=0.761
2026-04-25 03:31:09,816 INFO Regime epoch 30/50 — tr=0.2390 va=1.7088 acc=0.759 per_class={'BIAS_UP': 0.97, 'BIAS_DOWN': 0.991, 'BIAS_NEUTRAL': 0.625}
2026-04-25 03:31:10,009 INFO Regime epoch 31/50 — tr=0.2391 va=1.7039 acc=0.761
2026-04-25 03:31:10,204 INFO Regime epoch 32/50 — tr=0.2391 va=1.7064 acc=0.759
2026-04-25 03:31:10,401 INFO Regime epoch 33/50 — tr=0.2392 va=1.7056 acc=0.760
2026-04-25 03:31:10,597 INFO Regime epoch 34/50 — tr=0.2390 va=1.7059 acc=0.761
2026-04-25 03:31:10,810 INFO Regime epoch 35/50 — tr=0.2389 va=1.7057 acc=0.760 per_class={'BIAS_UP': 0.971, 'BIAS_DOWN': 0.99, 'BIAS_NEUTRAL': 0.626}
2026-04-25 03:31:10,997 INFO Regime epoch 36/50 — tr=0.2391 va=1.7048 acc=0.760
2026-04-25 03:31:11,185 INFO Regime epoch 37/50 — tr=0.2391 va=1.7068 acc=0.760
2026-04-25 03:31:11,381 INFO Regime epoch 38/50 — tr=0.2391 va=1.7064 acc=0.760
2026-04-25 03:31:11,571 INFO Regime epoch 39/50 — tr=0.2391 va=1.7059 acc=0.762
2026-04-25 03:31:11,780 INFO Regime epoch 40/50 — tr=0.2390 va=1.7048 acc=0.760 per_class={'BIAS_UP': 0.972, 'BIAS_DOWN': 0.99, 'BIAS_NEUTRAL': 0.627}
2026-04-25 03:31:11,969 INFO Regime epoch 41/50 — tr=0.2390 va=1.7069 acc=0.760
2026-04-25 03:31:11,969 INFO Regime early stop at epoch 41 (no_improve=10)
2026-04-25 03:31:11,986 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-25 03:31:11,986 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-25 03:31:12,116 INFO Regime HTF complete: acc=0.761, n=103290
2026-04-25 03:31:12,118 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-25 03:31:12,276 INFO Rule labels LTF_BEHAVIOUR [4H]: {'TRENDING': 8970, 'RANGING': 4592, 'CONSOLIDATING': 2247, 'VOLATILE': 4008}  ambiguous=4539 (total=19817)  short_runs_zeroed=570
2026-04-25 03:31:12,279 INFO Regime[4H] persistence (avg bars/run) on XAUUSD 4H:
{0: 12.121621621621621, 1: 6.4858757062146895, 2: 5.575682382133995, 3: 10.329896907216495}
2026-04-25 03:31:12,280 INFO Regime[4H] return separation on XAUUSD 4H:
{0: 5.9502411776840314e-05, 1: 5.664222849552333e-05, 2: -6.138442068010657e-05, 3: 4.239843820627722e-05}
2026-04-25 03:31:12,281 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-25 03:31:12,282 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:31:12,284 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:31:12,286 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:31:12,288 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:31:12,289 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:31:12,291 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:31:12,292 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:31:12,294 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:31:12,296 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:31:12,297 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:31:12,300 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-25 03:31:12,311 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:12,314 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:12,314 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:12,315 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:12,315 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:12,317 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:31:22,300 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-25 03:31:22,303 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-25 03:31:22,440 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:22,442 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:22,443 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:22,444 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:22,444 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:22,446 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:31:32,296 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-25 03:31:32,299 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-25 03:31:32,453 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:32,455 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:32,456 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:32,456 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:32,457 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:32,459 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:31:42,386 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-25 03:31:42,389 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-25 03:31:42,541 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:42,543 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:42,544 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:42,544 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:42,545 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:42,547 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:31:52,554 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-25 03:31:52,557 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-25 03:31:52,704 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:52,706 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:52,707 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:52,707 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:52,708 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:31:52,710 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:32:02,512 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-25 03:32:02,515 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-25 03:32:02,660 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:32:02,663 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:32:02,664 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:32:02,664 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:32:02,665 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:32:02,667 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:32:12,474 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-25 03:32:12,477 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-25 03:32:12,629 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-25 03:32:12,630 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-25 03:32:12,631 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-25 03:32:12,632 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-25 03:32:12,632 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-25 03:32:12,634 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:32:22,595 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-25 03:32:22,598 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-25 03:32:22,743 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:32:22,745 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:32:22,746 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:32:22,747 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:32:22,747 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:32:22,749 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:32:32,579 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-25 03:32:32,581 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-25 03:32:32,735 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:32:32,737 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:32:32,738 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:32:32,738 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:32:32,739 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:32:32,741 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:32:42,496 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-25 03:32:42,499 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-25 03:32:42,649 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:32:42,652 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:32:42,653 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:32:42,653 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:32:42,654 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:32:42,656 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:32:52,459 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-25 03:32:52,462 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-25 03:32:52,624 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-25 03:32:52,627 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-25 03:32:52,629 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-25 03:32:52,629 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-25 03:32:52,630 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-25 03:32:52,634 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-25 03:33:14,970 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-25 03:33:14,976 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-25 03:33:15,292 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260425_033315
2026-04-25 03:33:15,297 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=18, n_classes=4)
2026-04-25 03:33:15,354 INFO RegimeClassifier[mode=ltf_behaviour]: 401471 samples, classes={'TRENDING': 160094, 'RANGING': 105286, 'CONSOLIDATING': 53524, 'VOLATILE': 82567}, device=cuda
2026-04-25 03:33:15,355 INFO RegimeClassifier: sample weights — mean=0.505  ambiguous(<0.4)=33.0%
2026-04-25 03:33:15,355 INFO RegimeClassifier[mode=ltf_behaviour]: warm start from existing weights
2026-04-25 03:33:15,356 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-25 03:33:16,107 INFO Regime epoch  1/50 — tr=0.4704 va=1.5781 acc=0.724 per_class={'TRENDING': 0.836, 'RANGING': 0.307, 'CONSOLIDATING': 0.932, 'VOLATILE': 0.905}
2026-04-25 03:33:16,787 INFO Regime epoch  2/50 — tr=0.4703 va=1.5779 acc=0.723
2026-04-25 03:33:17,472 INFO Regime epoch  3/50 — tr=0.4704 va=1.5796 acc=0.723
2026-04-25 03:33:18,197 INFO Regime epoch  4/50 — tr=0.4702 va=1.5751 acc=0.724
2026-04-25 03:33:19,000 INFO Regime epoch  5/50 — tr=0.4701 va=1.5804 acc=0.722 per_class={'TRENDING': 0.831, 'RANGING': 0.306, 'CONSOLIDATING': 0.929, 'VOLATILE': 0.911}
2026-04-25 03:33:19,670 INFO Regime epoch  6/50 — tr=0.4702 va=1.5805 acc=0.722
2026-04-25 03:33:20,365 INFO Regime epoch  7/50 — tr=0.4703 va=1.5760 acc=0.723
2026-04-25 03:33:21,033 INFO Regime epoch  8/50 — tr=0.4702 va=1.5780 acc=0.723
2026-04-25 03:33:21,713 INFO Regime epoch  9/50 — tr=0.4702 va=1.5792 acc=0.723
2026-04-25 03:33:22,466 INFO Regime epoch 10/50 — tr=0.4701 va=1.5786 acc=0.723 per_class={'TRENDING': 0.832, 'RANGING': 0.309, 'CONSOLIDATING': 0.93, 'VOLATILE': 0.909}
2026-04-25 03:33:23,141 INFO Regime epoch 11/50 — tr=0.4703 va=1.5791 acc=0.723
2026-04-25 03:33:23,855 INFO Regime epoch 12/50 — tr=0.4701 va=1.5763 acc=0.725
2026-04-25 03:33:24,558 INFO Regime epoch 13/50 — tr=0.4700 va=1.5782 acc=0.723
2026-04-25 03:33:25,237 INFO Regime epoch 14/50 — tr=0.4700 va=1.5769 acc=0.724
2026-04-25 03:33:25,237 INFO Regime early stop at epoch 14 (no_improve=10)
2026-04-25 03:33:25,285 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-25 03:33:25,285 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-25 03:33:25,434 INFO Regime LTF complete: acc=0.724, n=401471
2026-04-25 03:33:25,438 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-25 03:33:25,924 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-25 03:33:25,929 INFO Regime[1H] persistence (avg bars/run) on XAUUSD 1H:
{0: 9.355291913830783, 1: 6.096303199751476, 2: 5.598885793871866, 3: 6.771351107094442}
2026-04-25 03:33:25,932 INFO Regime[1H] return separation on XAUUSD 1H:
{0: 2.149062259083323e-05, 1: -3.254493608872846e-06, 2: -3.039756158146958e-06, 3: 1.8102764458593193e-05}
2026-04-25 03:33:25,934 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-25 03:33:25,934 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-25 03:33:25,935 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-25 03:33:25,935 INFO === VectorStore: building similarity indices ===
2026-04-25 03:33:25,940 INFO Loading faiss with AVX512 support.
2026-04-25 03:33:25,964 INFO Successfully loaded faiss with AVX512 support.
2026-04-25 03:33:25,970 INFO VectorStore: CPU FAISS index built (dim=74)
2026-04-25 03:33:25,970 INFO VectorStore: CPU FAISS index built (dim=53)
2026-04-25 03:33:25,970 INFO VectorStore: CPU FAISS index built (dim=64)
2026-04-25 03:33:30,394 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-25 03:33:30,400 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:33:37,562 INFO VectorStore trade_patterns: +50000 vectors for AUDUSD
2026-04-25 03:33:37,792 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:33:37,794 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:33:37,795 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:33:37,796 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:33:37,796 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:34:16,414 WARNING VectorStore market_structures failed for AUDUSD: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 03:34:22,727 INFO VectorStore regime_embeddings: +13090 vectors for AUDUSD
2026-04-25 03:34:23,198 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:34:29,260 INFO VectorStore trade_patterns: +50000 vectors for EURGBP
2026-04-25 03:34:29,508 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:34:29,511 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:34:29,512 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:34:29,512 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:34:29,512 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:35:08,497 WARNING VectorStore market_structures failed for EURGBP: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 03:35:14,625 INFO VectorStore regime_embeddings: +13090 vectors for EURGBP
2026-04-25 03:35:15,095 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:35:21,161 INFO VectorStore trade_patterns: +50000 vectors for EURJPY
2026-04-25 03:35:21,415 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:35:21,417 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:35:21,418 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:35:21,418 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:35:21,419 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:36:00,284 WARNING VectorStore market_structures failed for EURJPY: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 03:36:06,317 INFO VectorStore regime_embeddings: +13091 vectors for EURJPY
2026-04-25 03:36:06,778 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:36:12,759 INFO VectorStore trade_patterns: +50000 vectors for EURUSD
2026-04-25 03:36:13,021 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:36:13,023 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:36:13,024 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:36:13,025 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:36:13,025 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:36:52,108 WARNING VectorStore market_structures failed for EURUSD: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 03:36:58,380 INFO VectorStore regime_embeddings: +13091 vectors for EURUSD
2026-04-25 03:36:58,855 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:37:04,834 INFO VectorStore trade_patterns: +50000 vectors for GBPJPY
2026-04-25 03:37:05,079 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:37:05,081 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:37:05,082 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:37:05,083 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:37:05,083 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:37:44,052 WARNING VectorStore market_structures failed for GBPJPY: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 03:37:49,996 INFO VectorStore regime_embeddings: +13091 vectors for GBPJPY
2026-04-25 03:37:50,479 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:37:56,647 INFO VectorStore trade_patterns: +50000 vectors for GBPUSD
2026-04-25 03:37:56,908 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:37:56,910 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:37:56,911 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:37:56,912 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:37:56,912 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:38:35,714 WARNING VectorStore market_structures failed for GBPUSD: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 03:38:41,740 INFO VectorStore regime_embeddings: +13091 vectors for GBPUSD
2026-04-25 03:38:42,222 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:38:48,133 INFO VectorStore trade_patterns: +50000 vectors for NZDUSD
2026-04-25 03:38:48,386 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-25 03:38:48,388 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-25 03:38:48,389 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-25 03:38:48,389 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-25 03:38:48,389 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-25 03:39:26,813 WARNING VectorStore market_structures failed for NZDUSD: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 03:39:32,742 INFO VectorStore regime_embeddings: +13091 vectors for NZDUSD
2026-04-25 03:39:33,218 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:39:39,139 INFO VectorStore trade_patterns: +50000 vectors for USDCAD
2026-04-25 03:39:39,384 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:39:39,386 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:39:39,387 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:39:39,387 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:39:39,388 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:40:16,809 WARNING VectorStore market_structures failed for USDCAD: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 03:40:22,803 INFO VectorStore regime_embeddings: +13091 vectors for USDCAD
2026-04-25 03:40:23,270 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:40:29,180 INFO VectorStore trade_patterns: +50000 vectors for USDCHF
2026-04-25 03:40:29,438 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:40:29,440 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:40:29,441 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:40:29,441 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:40:29,442 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:41:08,119 WARNING VectorStore market_structures failed for USDCHF: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 03:41:14,334 INFO VectorStore regime_embeddings: +13091 vectors for USDCHF
2026-04-25 03:41:14,833 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-25 03:41:20,870 INFO VectorStore trade_patterns: +50000 vectors for USDJPY
2026-04-25 03:41:21,120 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:41:21,123 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:41:21,124 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:41:21,124 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:41:21,124 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 03:41:59,810 WARNING VectorStore market_structures failed for USDJPY: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 03:42:05,908 INFO VectorStore regime_embeddings: +13093 vectors for USDJPY
2026-04-25 03:42:06,413 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-25 03:42:19,209 INFO VectorStore trade_patterns: +50000 vectors for XAUUSD
2026-04-25 03:42:19,470 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-25 03:42:19,473 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-25 03:42:19,474 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-25 03:42:19,475 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-25 03:42:19,475 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-25 03:43:45,118 WARNING VectorStore market_structures failed for XAUUSD: VectorStore.add_batch: 'market_structures' expects dim=53, got 67
2026-04-25 03:43:57,923 INFO VectorStore regime_embeddings: +12828 vectors for XAUUSD
2026-04-25 03:44:06,802 INFO VectorStore: saved 6937380 total vectors to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/vector_store
2026-04-25 03:44:06,802 INFO VectorStore saved: {'trade_patterns': 5500000, 'market_structures': 0, 'regime_embeddings': 1437380}
2026-04-25 03:44:07,476 INFO Retrain complete.
  DONE  Retrain regime [full-data retrain]
  START Retrain quality [full-data retrain]
2026-04-25 03:44:09,472 INFO retrain environment: KAGGLE
2026-04-25 03:44:11,290 INFO Device: CUDA (2 GPU(s))
2026-04-25 03:44:11,303 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-25 03:44:11,303 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-25 03:44:11,304 INFO === QualityScorer retrain ===
2026-04-25 03:44:11,460 INFO NumExpr defaulting to 4 threads.
2026-04-25 03:44:11,674 INFO QualityScorer: CUDA available — using GPU
2026-04-25 03:44:11,896 INFO QualityScorer loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (device=cuda)
2026-04-25 03:44:12,154 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/quality_scorer.pkl_20260425_034412
2026-04-25 03:44:12,392 INFO QualityScorer: 6308 samples, EV stats={'mean': 0.15640868246555328, 'std': 1.1945221424102783, 'n_pos': 3272, 'n_neg': 3036}, device=cuda
2026-04-25 03:44:12,393 INFO QualityScorer: normalised win labels by median_win=0.828 — EV range now [-1, +3]
2026-04-25 03:44:12,393 INFO QualityScorer: warm start from existing weights
2026-04-25 03:44:12,394 INFO QualityScorer: pos_weight=1.00 (n_pos=2611 n_neg=2435)
2026-04-25 03:44:14,185 INFO Quality epoch   1/100 — va_huber=0.7017
2026-04-25 03:44:14,310 INFO Quality epoch   2/100 — va_huber=0.7010
2026-04-25 03:44:14,428 INFO Quality epoch   3/100 — va_huber=0.7011
2026-04-25 03:44:14,540 INFO Quality epoch   4/100 — va_huber=0.7009
2026-04-25 03:44:14,659 INFO Quality epoch   5/100 — va_huber=0.7011
2026-04-25 03:44:15,362 INFO Quality epoch  11/100 — va_huber=0.7005
2026-04-25 03:44:16,546 INFO Quality epoch  21/100 — va_huber=0.7002
2026-04-25 03:44:17,710 INFO Quality epoch  31/100 — va_huber=0.7002
2026-04-25 03:44:18,913 INFO Quality epoch  41/100 — va_huber=0.6993
2026-04-25 03:44:20,089 INFO Quality epoch  51/100 — va_huber=0.6993
2026-04-25 03:44:21,284 INFO Quality epoch  61/100 — va_huber=0.6990
2026-04-25 03:44:22,684 INFO Quality epoch  71/100 — va_huber=0.6984
2026-04-25 03:44:23,322 INFO Quality early stop at epoch 76
2026-04-25 03:44:23,340 INFO QualityScorer EV model: MAE=1.163 dir_acc=0.571 n_val=1262
2026-04-25 03:44:23,343 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-25 03:44:23,437 INFO Retrain complete.
  DONE  Retrain quality [full-data retrain]
  START Retrain rl [full-data retrain]
2026-04-25 03:44:24,610 INFO retrain environment: KAGGLE
2026-04-25 03:44:26,285 INFO Device: CUDA (2 GPU(s))
2026-04-25 03:44:26,297 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-25 03:44:26,297 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-25 03:44:26,298 INFO === RLAgent (PPO) retrain ===
2026-04-25 03:44:26,300 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260425_034426
2026-04-25 03:44:27.303011: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777088667.328459   91551 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777088667.336506   91551 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777088667.357736   91551 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777088667.357771   91551 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777088667.357774   91551 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777088667.357776   91551 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-25 03:44:31,954 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-25 03:44:34,804 INFO RLAgent: PPO model loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-25 03:44:34,961 INFO RLAgent: warm start — fine-tuning existing PPO policy (lr=6.00e-05)
2026-04-25 03:46:27,823 INFO RLAgent: retrain complete, 6308 episodes
2026-04-25 03:46:27,824 INFO Retrain complete.
  DONE  Retrain rl [full-data retrain]

=== Round 3: Backtest on last 3yr (post-retrain evaluation) ===
  START Round 3 - Post-retrain backtest (last 3yr)
2026-04-25 03:46:29,975 INFO === STEP 6: BACKTEST + REINFORCED TRAINING (3 rounds, window=round3) ===
2026-04-25 03:46:29,976 INFO BT_WINDOW=round3 — post-retrain eval: 2022-08-05 → 2025-08-05 (last 3yr)
2026-04-25 03:46:29,977 INFO Cleared existing journal for fresh reinforced training run
2026-04-25 03:46:29,978 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-25 03:46:29,978 INFO Round 1 — running backtest: 2022-08-05 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-25 03:55:12,006 INFO Round 1 backtest — 2455 trades | avg WR=54.1% | avg PF=2.41 | avg Sharpe=4.53
2026-04-25 03:55:12,006 INFO   ml_trader: 2455 trades | WR=54.1% | PF=2.41 | Return=3093.6% | DD=2.4% | Sharpe=4.53
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 2455
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2455 rows)
2026-04-25 03:55:13,075 INFO Round 1: wrote 2455 journal entries (total in file: 2455)
2026-04-25 03:55:13,077 INFO Round 1 — retraining regime...