 Cleared done-check: training_summary.json
  Cleared done-check: latest_summary.json
Environment : KAGGLE
  base      -> /kaggle/working/Multi-Bot/trading-system
  data      -> /kaggle/input/datasets/tysonsiwela/ml-dataset/training_data
  processed -> /kaggle/input/datasets/tysonsiwela/ml-dataset/processed_data
  ml_train  -> /kaggle/working/remote/Multi-Bot/trading-system/ml_training
  weights   -> /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights
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
2026-04-25 05:10:20,678 INFO Loading feature-engineered data...
2026-04-25 05:10:21,155 INFO Loaded 221743 rows, 202 features
2026-04-25 05:10:21,155 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-25 05:10:21,156 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-25 05:10:21,156 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-25 05:10:21,156 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-25 05:10:21,156 INFO No leakage confirmed: train < val < test timestamps

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
2026-04-25 05:10:23,687 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-25 05:10:23,688 INFO --- Training regime ---
2026-04-25 05:10:23,688 INFO Running retrain --model regime
2026-04-25 05:10:23,964 INFO retrain environment: KAGGLE
2026-04-25 05:10:25,550 INFO Device: CUDA (2 GPU(s))
2026-04-25 05:10:25,561 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-25 05:10:25,562 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-25 05:10:25,563 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-25 05:10:25,709 INFO NumExpr defaulting to 4 threads.
2026-04-25 05:10:25,920 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-25 05:10:25,920 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-25 05:10:25,920 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-25 05:10:26,114 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-25 05:10:26,116 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:26,191 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:26,263 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:26,336 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:26,408 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:26,484 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:26,551 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:26,621 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:26,689 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:26,759 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:26,844 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-25 05:10:26,902 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-25 05:10:26,918 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:26,920 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:26,935 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:26,936 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:26,952 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:26,953 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:26,969 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:26,971 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:26,987 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:26,991 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:27,005 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:27,007 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:27,022 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:27,025 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:27,040 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:27,043 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:27,058 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:27,061 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:27,075 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:27,078 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:10:27,095 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-25 05:10:27,101 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-25 05:10:27,828 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-25 05:10:49,104 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-25 05:10:49,106 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias)
2026-04-25 05:10:49,106 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-25 05:10:58,136 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-25 05:10:58,137 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias)
2026-04-25 05:10:58,138 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-25 05:11:05,512 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-25 05:11:05,512 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias)
2026-04-25 05:11:05,512 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-25 05:12:12,861 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-25 05:12:12,866 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour)
2026-04-25 05:12:12,867 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-25 05:12:42,704 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-25 05:12:42,705 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour)
2026-04-25 05:12:42,705 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-25 05:13:03,733 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-25 05:13:03,734 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour)
2026-04-25 05:13:03,834 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-25 05:13:03,835 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:03,837 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:03,838 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:03,839 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:03,840 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:03,841 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:03,842 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:03,843 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:03,844 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:03,845 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:03,846 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-25 05:13:03,973 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:04,016 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:04,017 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:04,018 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:04,027 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:04,028 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:06,777 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-25 05:13:06,779 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-25 05:13:06,968 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:07,001 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:07,002 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:07,003 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:07,011 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:07,012 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:09,703 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-25 05:13:09,704 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-25 05:13:09,877 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:09,911 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:09,912 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:09,912 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:09,921 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:09,922 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:12,633 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-25 05:13:12,634 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-25 05:13:12,804 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:12,843 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:12,843 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:12,844 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:12,853 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:12,854 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:15,638 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-25 05:13:15,639 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-25 05:13:15,815 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:15,850 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:15,851 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:15,851 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:15,860 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:15,861 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:18,522 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-25 05:13:18,523 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-25 05:13:18,695 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:18,730 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:18,731 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:18,731 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:18,740 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:18,741 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:21,397 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-25 05:13:21,398 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-25 05:13:21,555 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-25 05:13:21,584 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-25 05:13:21,584 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-25 05:13:21,585 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-25 05:13:21,593 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-25 05:13:21,594 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:24,490 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-25 05:13:24,491 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-25 05:13:24,670 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:24,703 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:24,703 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:24,704 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:24,717 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:24,718 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:27,455 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-25 05:13:27,456 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-25 05:13:27,633 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:27,671 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:27,671 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:27,672 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:27,680 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:27,681 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:30,385 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-25 05:13:30,387 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-25 05:13:30,565 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:30,600 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:30,601 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:30,601 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:30,610 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:30,610 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:33,334 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-25 05:13:33,335 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-25 05:13:33,634 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-25 05:13:33,694 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-25 05:13:33,696 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-25 05:13:33,696 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-25 05:13:33,708 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-25 05:13:33,709 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-25 05:13:40,006 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-25 05:13:40,007 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-25 05:13:40,214 INFO RegimeClassifier[mode=htf_bias]: 103290 samples, classes={'BIAS_UP': 18622, 'BIAS_DOWN': 18286, 'BIAS_NEUTRAL': 66382}, device=cuda
2026-04-25 05:13:40,214 INFO RegimeClassifier: sample weights — mean=0.360  ambiguous(<0.4)=69.5%
2026-04-25 05:13:40,405 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-04-25 05:13:40,406 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-25 05:13:42,775 INFO Regime epoch  1/50 — tr=0.3295 va=2.2163 acc=0.242 per_class={'BIAS_UP': 0.533, 'BIAS_DOWN': 0.183, 'BIAS_NEUTRAL': 0.153}
2026-04-25 05:13:42,963 INFO Regime epoch  2/50 — tr=0.3245 va=2.1874 acc=0.340
2026-04-25 05:13:43,148 INFO Regime epoch  3/50 — tr=0.3166 va=2.1579 acc=0.414
2026-04-25 05:13:43,334 INFO Regime epoch  4/50 — tr=0.3061 va=2.1283 acc=0.483
2026-04-25 05:13:43,557 INFO Regime epoch  5/50 — tr=0.2946 va=2.0699 acc=0.522 per_class={'BIAS_UP': 0.925, 'BIAS_DOWN': 0.892, 'BIAS_NEUTRAL': 0.283}
2026-04-25 05:13:43,736 INFO Regime epoch  6/50 — tr=0.2844 va=1.9938 acc=0.548
2026-04-25 05:13:43,929 INFO Regime epoch  7/50 — tr=0.2764 va=1.9308 acc=0.566
2026-04-25 05:13:44,140 INFO Regime epoch  8/50 — tr=0.2696 va=1.8776 acc=0.584
2026-04-25 05:13:44,330 INFO Regime epoch  9/50 — tr=0.2651 va=1.8544 acc=0.594
2026-04-25 05:13:44,523 INFO Regime epoch 10/50 — tr=0.2615 va=1.8393 acc=0.604 per_class={'BIAS_UP': 0.968, 'BIAS_DOWN': 0.985, 'BIAS_NEUTRAL': 0.378}
2026-04-25 05:13:44,708 INFO Regime epoch 11/50 — tr=0.2588 va=1.8245 acc=0.611
2026-04-25 05:13:44,896 INFO Regime epoch 12/50 — tr=0.2563 va=1.8173 acc=0.617
2026-04-25 05:13:45,070 INFO Regime epoch 13/50 — tr=0.2544 va=1.8112 acc=0.622
2026-04-25 05:13:45,251 INFO Regime epoch 14/50 — tr=0.2530 va=1.8031 acc=0.629
2026-04-25 05:13:45,449 INFO Regime epoch 15/50 — tr=0.2518 va=1.7996 acc=0.634 per_class={'BIAS_UP': 0.967, 'BIAS_DOWN': 0.981, 'BIAS_NEUTRAL': 0.427}
2026-04-25 05:13:45,629 INFO Regime epoch 16/50 — tr=0.2508 va=1.8014 acc=0.634
2026-04-25 05:13:45,810 INFO Regime epoch 17/50 — tr=0.2496 va=1.8024 acc=0.636
2026-04-25 05:13:46,005 INFO Regime epoch 18/50 — tr=0.2488 va=1.7980 acc=0.643
2026-04-25 05:13:46,187 INFO Regime epoch 19/50 — tr=0.2482 va=1.7975 acc=0.642
2026-04-25 05:13:46,384 INFO Regime epoch 20/50 — tr=0.2475 va=1.7986 acc=0.641 per_class={'BIAS_UP': 0.974, 'BIAS_DOWN': 0.984, 'BIAS_NEUTRAL': 0.434}
2026-04-25 05:13:46,563 INFO Regime epoch 21/50 — tr=0.2471 va=1.8011 acc=0.640
2026-04-25 05:13:46,741 INFO Regime epoch 22/50 — tr=0.2468 va=1.7957 acc=0.646
2026-04-25 05:13:46,922 INFO Regime epoch 23/50 — tr=0.2462 va=1.7976 acc=0.645
2026-04-25 05:13:47,099 INFO Regime epoch 24/50 — tr=0.2459 va=1.7959 acc=0.648
2026-04-25 05:13:47,290 INFO Regime epoch 25/50 — tr=0.2456 va=1.7902 acc=0.654 per_class={'BIAS_UP': 0.973, 'BIAS_DOWN': 0.982, 'BIAS_NEUTRAL': 0.457}
2026-04-25 05:13:47,467 INFO Regime epoch 26/50 — tr=0.2454 va=1.7923 acc=0.655
2026-04-25 05:13:47,641 INFO Regime epoch 27/50 — tr=0.2451 va=1.7940 acc=0.651
2026-04-25 05:13:47,818 INFO Regime epoch 28/50 — tr=0.2449 va=1.7933 acc=0.652
2026-04-25 05:13:47,995 INFO Regime epoch 29/50 — tr=0.2449 va=1.7938 acc=0.653
2026-04-25 05:13:48,184 INFO Regime epoch 30/50 — tr=0.2448 va=1.7905 acc=0.654 per_class={'BIAS_UP': 0.976, 'BIAS_DOWN': 0.986, 'BIAS_NEUTRAL': 0.455}
2026-04-25 05:13:48,361 INFO Regime epoch 31/50 — tr=0.2444 va=1.7906 acc=0.655
2026-04-25 05:13:48,535 INFO Regime epoch 32/50 — tr=0.2444 va=1.7900 acc=0.657
2026-04-25 05:13:48,725 INFO Regime epoch 33/50 — tr=0.2442 va=1.7911 acc=0.656
2026-04-25 05:13:48,918 INFO Regime epoch 34/50 — tr=0.2442 va=1.7909 acc=0.656
2026-04-25 05:13:49,125 INFO Regime epoch 35/50 — tr=0.2440 va=1.7904 acc=0.657 per_class={'BIAS_UP': 0.976, 'BIAS_DOWN': 0.986, 'BIAS_NEUTRAL': 0.46}
2026-04-25 05:13:49,307 INFO Regime epoch 36/50 — tr=0.2440 va=1.7906 acc=0.657
2026-04-25 05:13:49,488 INFO Regime epoch 37/50 — tr=0.2439 va=1.7908 acc=0.656
2026-04-25 05:13:49,680 INFO Regime epoch 38/50 — tr=0.2437 va=1.7916 acc=0.655
2026-04-25 05:13:49,872 INFO Regime epoch 39/50 — tr=0.2439 va=1.7896 acc=0.659
2026-04-25 05:13:50,078 INFO Regime epoch 40/50 — tr=0.2437 va=1.7894 acc=0.660 per_class={'BIAS_UP': 0.977, 'BIAS_DOWN': 0.987, 'BIAS_NEUTRAL': 0.464}
2026-04-25 05:13:50,262 INFO Regime epoch 41/50 — tr=0.2436 va=1.7890 acc=0.661
2026-04-25 05:13:50,443 INFO Regime epoch 42/50 — tr=0.2437 va=1.7914 acc=0.657
2026-04-25 05:13:50,627 INFO Regime epoch 43/50 — tr=0.2436 va=1.7881 acc=0.660
2026-04-25 05:13:50,815 INFO Regime epoch 44/50 — tr=0.2437 va=1.7908 acc=0.658
2026-04-25 05:13:51,008 INFO Regime epoch 45/50 — tr=0.2435 va=1.7893 acc=0.660 per_class={'BIAS_UP': 0.977, 'BIAS_DOWN': 0.988, 'BIAS_NEUTRAL': 0.464}
2026-04-25 05:13:51,202 INFO Regime epoch 46/50 — tr=0.2436 va=1.7881 acc=0.660
2026-04-25 05:13:51,401 INFO Regime epoch 47/50 — tr=0.2436 va=1.7895 acc=0.659
2026-04-25 05:13:51,594 INFO Regime epoch 48/50 — tr=0.2436 va=1.7873 acc=0.661
2026-04-25 05:13:51,788 INFO Regime epoch 49/50 — tr=0.2435 va=1.7877 acc=0.660
2026-04-25 05:13:51,989 INFO Regime epoch 50/50 — tr=0.2436 va=1.7882 acc=0.661 per_class={'BIAS_UP': 0.977, 'BIAS_DOWN': 0.989, 'BIAS_NEUTRAL': 0.465}
2026-04-25 05:13:52,006 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-25 05:13:52,006 INFO RegimeClassifier[4H] saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-25 05:13:52,132 INFO Regime HTF complete: acc=0.661, n=103290
2026-04-25 05:13:52,134 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-25 05:13:52,289 INFO Rule labels LTF_BEHAVIOUR [4H]: {'TRENDING': 8970, 'RANGING': 4592, 'CONSOLIDATING': 2247, 'VOLATILE': 4008}  ambiguous=4539 (total=19817)  short_runs_zeroed=570
2026-04-25 05:13:52,291 INFO Regime[4H] persistence (avg bars/run) on XAUUSD 4H:
{0: 12.121621621621621, 1: 6.4858757062146895, 2: 5.575682382133995, 3: 10.329896907216495}
2026-04-25 05:13:52,293 INFO Regime[4H] return separation on XAUUSD 4H:
{0: 5.9502411776840314e-05, 1: 5.664222849552333e-05, 2: -6.138442068010657e-05, 3: 4.239843820627722e-05}
2026-04-25 05:13:52,293 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-25 05:13:52,295 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:52,297 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:52,299 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:52,300 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:52,302 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:52,304 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:52,305 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:52,307 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:52,308 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:52,310 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:13:52,313 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-25 05:13:52,323 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:52,325 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:52,326 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:52,326 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:52,327 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:13:52,328 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:14:02,201 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-25 05:14:02,204 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-25 05:14:02,343 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:02,345 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:02,346 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:02,347 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:02,347 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:02,349 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:14:12,016 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-25 05:14:12,019 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-25 05:14:12,167 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:12,170 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:12,170 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:12,171 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:12,171 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:12,173 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:14:22,057 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-25 05:14:22,060 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-25 05:14:22,192 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:22,195 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:22,196 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:22,196 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:22,196 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:22,198 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:14:32,221 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-25 05:14:32,224 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-25 05:14:32,376 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:32,379 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:32,380 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:32,380 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:32,381 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:32,383 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:14:42,135 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-25 05:14:42,138 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-25 05:14:42,278 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:42,281 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:42,281 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:42,282 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:42,282 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:14:42,284 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:14:52,053 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-25 05:14:52,056 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-25 05:14:52,187 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-25 05:14:52,188 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-25 05:14:52,189 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-25 05:14:52,189 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-25 05:14:52,190 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-25 05:14:52,191 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:15:02,191 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-25 05:15:02,194 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-25 05:15:02,337 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:15:02,339 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:15:02,340 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:15:02,341 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:15:02,341 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:15:02,343 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:15:12,170 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-25 05:15:12,173 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-25 05:15:12,309 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:15:12,313 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:15:12,314 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:15:12,314 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:15:12,314 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:15:12,316 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:15:22,135 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-25 05:15:22,138 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-25 05:15:22,267 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:15:22,271 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:15:22,271 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:15:22,272 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:15:22,272 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:15:22,274 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:15:31,964 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-25 05:15:31,967 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-25 05:15:32,103 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-25 05:15:32,109 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-25 05:15:32,110 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-25 05:15:32,111 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-25 05:15:32,111 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-25 05:15:32,114 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-25 05:15:54,448 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-25 05:15:54,454 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-25 05:15:54,811 INFO RegimeClassifier[mode=ltf_behaviour]: 401471 samples, classes={'TRENDING': 160094, 'RANGING': 105286, 'CONSOLIDATING': 53524, 'VOLATILE': 82567}, device=cuda
2026-04-25 05:15:54,812 INFO RegimeClassifier: sample weights — mean=0.505  ambiguous(<0.4)=33.0%
2026-04-25 05:15:54,814 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-04-25 05:15:54,814 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-25 05:15:55,550 INFO Regime epoch  1/50 — tr=0.6541 va=2.0069 acc=0.312 per_class={'TRENDING': 0.174, 'RANGING': 0.385, 'CONSOLIDATING': 0.382, 'VOLATILE': 0.444}
2026-04-25 05:15:56,208 INFO Regime epoch  2/50 — tr=0.6265 va=1.9335 acc=0.422
2026-04-25 05:15:56,872 INFO Regime epoch  3/50 — tr=0.5914 va=1.8665 acc=0.457
2026-04-25 05:15:57,571 INFO Regime epoch  4/50 — tr=0.5601 va=1.8214 acc=0.481
2026-04-25 05:15:58,291 INFO Regime epoch  5/50 — tr=0.5407 va=1.8041 acc=0.506 per_class={'TRENDING': 0.42, 'RANGING': 0.113, 'CONSOLIDATING': 0.826, 'VOLATILE': 0.976}
2026-04-25 05:15:58,986 INFO Regime epoch  6/50 — tr=0.5289 va=1.8049 acc=0.526
2026-04-25 05:15:59,682 INFO Regime epoch  7/50 — tr=0.5201 va=1.7961 acc=0.547
2026-04-25 05:16:00,357 INFO Regime epoch  8/50 — tr=0.5131 va=1.7796 acc=0.577
2026-04-25 05:16:01,042 INFO Regime epoch  9/50 — tr=0.5076 va=1.7651 acc=0.601
2026-04-25 05:16:01,767 INFO Regime epoch 10/50 — tr=0.5026 va=1.7485 acc=0.617 per_class={'TRENDING': 0.677, 'RANGING': 0.163, 'CONSOLIDATING': 0.824, 'VOLATILE': 0.951}
2026-04-25 05:16:02,422 INFO Regime epoch 11/50 — tr=0.4987 va=1.7242 acc=0.632
2026-04-25 05:16:03,119 INFO Regime epoch 12/50 — tr=0.4954 va=1.7078 acc=0.643
2026-04-25 05:16:03,827 INFO Regime epoch 13/50 — tr=0.4929 va=1.6948 acc=0.655
2026-04-25 05:16:04,521 INFO Regime epoch 14/50 — tr=0.4907 va=1.6787 acc=0.665
2026-04-25 05:16:05,266 INFO Regime epoch 15/50 — tr=0.4887 va=1.6693 acc=0.667 per_class={'TRENDING': 0.759, 'RANGING': 0.217, 'CONSOLIDATING': 0.871, 'VOLATILE': 0.935}
2026-04-25 05:16:05,933 INFO Regime epoch 16/50 — tr=0.4871 va=1.6581 acc=0.675
2026-04-25 05:16:06,594 INFO Regime epoch 17/50 — tr=0.4857 va=1.6511 acc=0.680
2026-04-25 05:16:07,280 INFO Regime epoch 18/50 — tr=0.4845 va=1.6420 acc=0.685
2026-04-25 05:16:07,943 INFO Regime epoch 19/50 — tr=0.4834 va=1.6317 acc=0.689
2026-04-25 05:16:08,688 INFO Regime epoch 20/50 — tr=0.4822 va=1.6284 acc=0.692 per_class={'TRENDING': 0.785, 'RANGING': 0.264, 'CONSOLIDATING': 0.903, 'VOLATILE': 0.925}
2026-04-25 05:16:09,366 INFO Regime epoch 21/50 — tr=0.4815 va=1.6235 acc=0.697
2026-04-25 05:16:10,038 INFO Regime epoch 22/50 — tr=0.4806 va=1.6175 acc=0.700
2026-04-25 05:16:10,723 INFO Regime epoch 23/50 — tr=0.4799 va=1.6135 acc=0.702
2026-04-25 05:16:11,403 INFO Regime epoch 24/50 — tr=0.4794 va=1.6106 acc=0.706
2026-04-25 05:16:12,138 INFO Regime epoch 25/50 — tr=0.4786 va=1.6059 acc=0.707 per_class={'TRENDING': 0.795, 'RANGING': 0.296, 'CONSOLIDATING': 0.926, 'VOLATILE': 0.922}
2026-04-25 05:16:12,814 INFO Regime epoch 26/50 — tr=0.4782 va=1.6034 acc=0.709
2026-04-25 05:16:13,505 INFO Regime epoch 27/50 — tr=0.4778 va=1.6051 acc=0.708
2026-04-25 05:16:14,190 INFO Regime epoch 28/50 — tr=0.4773 va=1.5998 acc=0.712
2026-04-25 05:16:14,893 INFO Regime epoch 29/50 — tr=0.4772 va=1.5970 acc=0.712
2026-04-25 05:16:15,667 INFO Regime epoch 30/50 — tr=0.4766 va=1.5962 acc=0.713 per_class={'TRENDING': 0.805, 'RANGING': 0.304, 'CONSOLIDATING': 0.934, 'VOLATILE': 0.914}
2026-04-25 05:16:16,343 INFO Regime epoch 31/50 — tr=0.4765 va=1.5949 acc=0.713
2026-04-25 05:16:17,018 INFO Regime epoch 32/50 — tr=0.4761 va=1.5924 acc=0.716
2026-04-25 05:16:17,688 INFO Regime epoch 33/50 — tr=0.4760 va=1.5921 acc=0.716
2026-04-25 05:16:18,364 INFO Regime epoch 34/50 — tr=0.4758 va=1.5944 acc=0.714
2026-04-25 05:16:19,073 INFO Regime epoch 35/50 — tr=0.4755 va=1.5919 acc=0.716 per_class={'TRENDING': 0.81, 'RANGING': 0.309, 'CONSOLIDATING': 0.936, 'VOLATILE': 0.912}
2026-04-25 05:16:19,722 INFO Regime epoch 36/50 — tr=0.4754 va=1.5892 acc=0.717
2026-04-25 05:16:20,399 INFO Regime epoch 37/50 — tr=0.4753 va=1.5896 acc=0.718
2026-04-25 05:16:21,089 INFO Regime epoch 38/50 — tr=0.4752 va=1.5899 acc=0.717
2026-04-25 05:16:21,766 INFO Regime epoch 39/50 — tr=0.4751 va=1.5884 acc=0.718
2026-04-25 05:16:22,500 INFO Regime epoch 40/50 — tr=0.4751 va=1.5858 acc=0.719 per_class={'TRENDING': 0.818, 'RANGING': 0.314, 'CONSOLIDATING': 0.939, 'VOLATILE': 0.905}
2026-04-25 05:16:23,159 INFO Regime epoch 41/50 — tr=0.4749 va=1.5888 acc=0.718
2026-04-25 05:16:23,899 INFO Regime epoch 42/50 — tr=0.4750 va=1.5894 acc=0.719
2026-04-25 05:16:24,583 INFO Regime epoch 43/50 — tr=0.4748 va=1.5863 acc=0.719
2026-04-25 05:16:25,262 INFO Regime epoch 44/50 — tr=0.4748 va=1.5872 acc=0.718
2026-04-25 05:16:25,994 INFO Regime epoch 45/50 — tr=0.4748 va=1.5878 acc=0.718 per_class={'TRENDING': 0.812, 'RANGING': 0.315, 'CONSOLIDATING': 0.939, 'VOLATILE': 0.911}
2026-04-25 05:16:26,693 INFO Regime epoch 46/50 — tr=0.4749 va=1.5875 acc=0.718
2026-04-25 05:16:27,352 INFO Regime epoch 47/50 — tr=0.4747 va=1.5873 acc=0.719
2026-04-25 05:16:28,013 INFO Regime epoch 48/50 — tr=0.4747 va=1.5864 acc=0.718
2026-04-25 05:16:28,675 INFO Regime epoch 49/50 — tr=0.4746 va=1.5875 acc=0.719
2026-04-25 05:16:29,429 INFO Regime epoch 50/50 — tr=0.4749 va=1.5876 acc=0.719 per_class={'TRENDING': 0.819, 'RANGING': 0.311, 'CONSOLIDATING': 0.936, 'VOLATILE': 0.906}
2026-04-25 05:16:29,429 INFO Regime early stop at epoch 50 (no_improve=10)
2026-04-25 05:16:29,477 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-25 05:16:29,477 INFO RegimeClassifier[1H] saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-25 05:16:29,604 INFO Regime LTF complete: acc=0.719, n=401471
2026-04-25 05:16:29,608 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-25 05:16:30,069 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-25 05:16:30,073 INFO Regime[1H] persistence (avg bars/run) on XAUUSD 1H:
{0: 9.355291913830783, 1: 6.096303199751476, 2: 5.598885793871866, 3: 6.771351107094442}
2026-04-25 05:16:30,076 INFO Regime[1H] return separation on XAUUSD 1H:
{0: 2.149062259083323e-05, 1: -3.254493608872846e-06, 2: -3.039756158146958e-06, 3: 1.8102764458593193e-05}
2026-04-25 05:16:30,089 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-25 05:16:30,089 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-25 05:16:30,089 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-25 05:16:30,089 INFO === VectorStore: building similarity indices ===
2026-04-25 05:16:30,093 INFO Loading faiss with AVX512 support.
2026-04-25 05:16:30,115 INFO Successfully loaded faiss with AVX512 support.
2026-04-25 05:16:30,121 INFO VectorStore: CPU FAISS index built (dim=74)
2026-04-25 05:16:30,121 INFO VectorStore: CPU FAISS index built (dim=34)
2026-04-25 05:16:30,121 INFO VectorStore: CPU FAISS index built (dim=64)
2026-04-25 05:16:30,124 WARNING GRULSTMPredictor: stale weights detected (quality feature contract changed: added=['gru_signal_agreement', 'strategy_win_rate_5', 'strategy_win_rate_50']; count 17→20) — deleting /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt so retrain starts fresh
2026-04-25 05:16:30,124 INFO Deleted stale weights (quality feature contract changed: added=['gru_signal_agreement', 'strategy_win_rate_5', 'strategy_win_rate_50']; count 17→20): /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:16:30,129 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:16:35,949 INFO VectorStore trade_patterns: +50000 vectors for AUDUSD
2026-04-25 05:16:36,091 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:16:36,093 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:16:36,094 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:16:36,095 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:16:36,095 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:17:13,745 INFO VectorStore market_structures: +65447 vectors (34-dim 4H) for AUDUSD
2026-04-25 05:17:13,982 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:17:19,736 INFO VectorStore trade_patterns: +50000 vectors for EURGBP
2026-04-25 05:17:19,874 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:17:19,876 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:17:19,877 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:17:19,877 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:17:19,878 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:17:57,062 INFO VectorStore market_structures: +65448 vectors (34-dim 4H) for EURGBP
2026-04-25 05:17:57,360 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:18:03,183 INFO VectorStore trade_patterns: +50000 vectors for EURJPY
2026-04-25 05:18:03,337 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:18:03,339 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:18:03,340 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:18:03,341 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:18:03,341 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:18:40,516 INFO VectorStore market_structures: +65453 vectors (34-dim 4H) for EURJPY
2026-04-25 05:18:40,808 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:18:46,724 INFO VectorStore trade_patterns: +50000 vectors for EURUSD
2026-04-25 05:18:46,871 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:18:46,873 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:18:46,874 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:18:46,874 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:18:46,874 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:19:24,177 INFO VectorStore market_structures: +65451 vectors (34-dim 4H) for EURUSD
2026-04-25 05:19:24,436 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:19:30,374 INFO VectorStore trade_patterns: +50000 vectors for GBPJPY
2026-04-25 05:19:30,527 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:19:30,530 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:19:30,531 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:19:30,531 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:19:30,531 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:20:08,988 INFO VectorStore market_structures: +65452 vectors (34-dim 4H) for GBPJPY
2026-04-25 05:20:09,262 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:20:15,348 INFO VectorStore trade_patterns: +50000 vectors for GBPUSD
2026-04-25 05:20:15,514 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:20:15,516 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:20:15,517 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:20:15,518 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:20:15,518 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:20:53,888 INFO VectorStore market_structures: +65451 vectors (34-dim 4H) for GBPUSD
2026-04-25 05:20:54,191 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:21:00,156 INFO VectorStore trade_patterns: +50000 vectors for NZDUSD
2026-04-25 05:21:00,301 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-25 05:21:00,303 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-25 05:21:00,303 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-25 05:21:00,304 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-25 05:21:00,304 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-25 05:21:38,858 INFO VectorStore market_structures: +65451 vectors (34-dim 4H) for NZDUSD
2026-04-25 05:21:39,169 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:21:45,151 INFO VectorStore trade_patterns: +50000 vectors for USDCAD
2026-04-25 05:21:45,303 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:21:45,305 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:21:45,306 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:21:45,306 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:21:45,306 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:22:24,048 INFO VectorStore market_structures: +65452 vectors (34-dim 4H) for USDCAD
2026-04-25 05:22:24,322 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:22:30,516 INFO VectorStore trade_patterns: +50000 vectors for USDCHF
2026-04-25 05:22:30,684 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:22:30,686 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:22:30,687 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:22:30,688 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:22:30,688 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:23:09,338 INFO VectorStore market_structures: +65454 vectors (34-dim 4H) for USDCHF
2026-04-25 05:23:09,663 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:23:15,891 INFO VectorStore trade_patterns: +50000 vectors for USDJPY
2026-04-25 05:23:16,061 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:23:16,063 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:23:16,064 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:23:16,065 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:23:16,065 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:23:54,611 INFO VectorStore market_structures: +65461 vectors (34-dim 4H) for USDJPY
2026-04-25 05:23:54,907 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-25 05:24:07,694 INFO VectorStore trade_patterns: +50000 vectors for XAUUSD
2026-04-25 05:24:07,873 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-25 05:24:07,876 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-25 05:24:07,877 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-25 05:24:07,878 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-25 05:24:07,878 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-25 05:25:34,711 INFO VectorStore market_structures: +59006 vectors (34-dim 4H) for XAUUSD
2026-04-25 05:25:35,917 INFO VectorStore: saved 1263526 total vectors to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/vector_store
2026-04-25 05:25:35,917 INFO VectorStore saved: {'trade_patterns': 550000, 'market_structures': 713526, 'regime_embeddings': 0}
2026-04-25 05:25:36,028 INFO Retrain complete.
2026-04-25 05:25:37,157 INFO Model regime: SUCCESS
2026-04-25 05:25:37,158 INFO --- Training gru ---
2026-04-25 05:25:37,158 INFO Running retrain --model gru
2026-04-25 05:25:37,832 INFO retrain environment: KAGGLE
2026-04-25 05:25:39,457 INFO Device: CUDA (2 GPU(s))
2026-04-25 05:25:39,468 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-25 05:25:39,468 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-25 05:25:39,469 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-25 05:25:39,618 INFO NumExpr defaulting to 4 threads.
2026-04-25 05:25:39,824 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-25 05:25:39,824 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-25 05:25:39,824 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-25 05:25:40,058 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-25 05:25:40,060 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:40,139 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:40,228 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:40,316 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:40,394 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:40,467 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:40,537 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:40,612 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:40,685 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:40,757 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:40,848 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-25 05:25:40,908 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-25 05:25:40,908 INFO Backed up /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260425_052540
2026-04-25 05:25:40,910 INFO GRU weights stale (quality feature contract changed: added=['gru_signal_agreement', 'strategy_win_rate_5', 'strategy_win_rate_50']; count 17→20) — deleting for full retrain
2026-04-25 05:25:41,032 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:41,033 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:41,056 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:41,064 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:41,065 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-25 05:25:41,065 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-25 05:25:41,065 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-25 05:25:41,066 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:41,141 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-25 05:25:41,143 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:41,380 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-25 05:25:41,412 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:41,702 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:41,831 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:41,932 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:42,132 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:42,133 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:42,149 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:42,156 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:42,157 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:42,229 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-25 05:25:42,231 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:42,462 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-25 05:25:42,477 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:42,748 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:42,877 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:42,968 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:43,148 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:43,148 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:43,164 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:43,173 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:43,174 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:43,247 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-25 05:25:43,250 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:43,526 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-25 05:25:43,542 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:43,808 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:43,947 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:44,042 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:44,217 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:44,218 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:44,233 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:44,241 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:44,241 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:44,315 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-25 05:25:44,317 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:44,534 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-25 05:25:44,554 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:44,824 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:44,958 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:45,063 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:45,250 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:45,250 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:45,268 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:45,277 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:45,278 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:45,353 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-25 05:25:45,355 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:45,578 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-25 05:25:45,592 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:45,870 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:46,018 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:46,123 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:46,318 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:46,319 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:46,336 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:46,344 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:46,345 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:46,427 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-25 05:25:46,429 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:46,672 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-25 05:25:46,688 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:46,961 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:47,094 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:47,201 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:47,369 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-25 05:25:47,369 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-25 05:25:47,384 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-25 05:25:47,392 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-25 05:25:47,393 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:47,469 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-25 05:25:47,470 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:47,686 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-25 05:25:47,698 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:47,954 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:48,081 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:48,183 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:48,376 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:48,376 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:48,394 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:48,402 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:48,403 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:48,473 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-25 05:25:48,475 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:48,687 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-25 05:25:48,702 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:48,977 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:49,099 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:49,200 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:49,395 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:49,396 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:49,414 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:49,422 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:49,423 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:49,499 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-25 05:25:49,501 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:49,722 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-25 05:25:49,737 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:50,016 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:50,139 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:50,231 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:50,428 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:50,429 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:50,445 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:50,452 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:25:50,453 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:50,525 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-25 05:25:50,527 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:50,743 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-25 05:25:50,757 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:51,038 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:51,170 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:51,262 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:25:51,545 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-25 05:25:51,546 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-25 05:25:51,563 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-25 05:25:51,573 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-25 05:25:51,574 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-25 05:25:51,716 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-25 05:25:51,719 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-25 05:25:52,250 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-25 05:25:52,297 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-25 05:25:52,843 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-25 05:25:53,043 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-25 05:25:53,178 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-25 05:25:53,300 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-25 05:25:53,542 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-04-25 05:25:53,543 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-04-25 05:25:53,543 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-25 05:25:53,543 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-25 05:30:21,546 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-25 05:30:21,547 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-25 05:30:24,358 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-25 05:30:29,003 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=5)
2026-04-25 05:30:48,986 INFO train_multi TF=ALL epoch 1/50 train=0.8819 val=0.8556
2026-04-25 05:30:48,992 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:30:48,992 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:30:48,992 INFO train_multi TF=ALL: new best val=0.8556 — saved
2026-04-25 05:31:07,312 INFO train_multi TF=ALL epoch 2/50 train=0.7411 val=0.6880
2026-04-25 05:31:07,316 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:31:07,316 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:31:07,316 INFO train_multi TF=ALL: new best val=0.6880 — saved
2026-04-25 05:31:25,672 INFO train_multi TF=ALL epoch 3/50 train=0.6904 val=0.6881
2026-04-25 05:31:43,979 INFO train_multi TF=ALL epoch 4/50 train=0.6896 val=0.6884
2026-04-25 05:32:02,097 INFO train_multi TF=ALL epoch 5/50 train=0.6894 val=0.6883
2026-04-25 05:32:20,635 INFO train_multi TF=ALL epoch 6/50 train=0.6892 val=0.6880
2026-04-25 05:32:20,640 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:32:20,640 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:32:20,640 INFO train_multi TF=ALL: new best val=0.6880 — saved
2026-04-25 05:32:38,672 INFO train_multi TF=ALL epoch 7/50 train=0.6889 val=0.6882
2026-04-25 05:32:56,769 INFO train_multi TF=ALL epoch 8/50 train=0.6883 val=0.6870
2026-04-25 05:32:56,774 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:32:56,775 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:32:56,775 INFO train_multi TF=ALL: new best val=0.6870 — saved
2026-04-25 05:33:15,245 INFO train_multi TF=ALL epoch 9/50 train=0.6854 val=0.6842
2026-04-25 05:33:15,249 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:33:15,249 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:33:15,249 INFO train_multi TF=ALL: new best val=0.6842 — saved
2026-04-25 05:33:33,416 INFO train_multi TF=ALL epoch 10/50 train=0.6779 val=0.6716
2026-04-25 05:33:33,420 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:33:33,420 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:33:33,420 INFO train_multi TF=ALL: new best val=0.6716 — saved
2026-04-25 05:33:51,782 INFO train_multi TF=ALL epoch 11/50 train=0.6663 val=0.6595
2026-04-25 05:33:51,786 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:33:51,786 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:33:51,786 INFO train_multi TF=ALL: new best val=0.6595 — saved
2026-04-25 05:34:09,760 INFO train_multi TF=ALL epoch 12/50 train=0.6524 val=0.6418
2026-04-25 05:34:09,765 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:34:09,765 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:34:09,765 INFO train_multi TF=ALL: new best val=0.6418 — saved
2026-04-25 05:34:27,985 INFO train_multi TF=ALL epoch 13/50 train=0.6420 val=0.6356
2026-04-25 05:34:27,990 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:34:27,990 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:34:27,990 INFO train_multi TF=ALL: new best val=0.6356 — saved
2026-04-25 05:34:46,020 INFO train_multi TF=ALL epoch 14/50 train=0.6351 val=0.6333
2026-04-25 05:34:46,025 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:34:46,025 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:34:46,025 INFO train_multi TF=ALL: new best val=0.6333 — saved
2026-04-25 05:35:04,475 INFO train_multi TF=ALL epoch 15/50 train=0.6300 val=0.6284
2026-04-25 05:35:04,480 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:35:04,480 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:35:04,480 INFO train_multi TF=ALL: new best val=0.6284 — saved
2026-04-25 05:35:22,915 INFO train_multi TF=ALL epoch 16/50 train=0.6261 val=0.6266
2026-04-25 05:35:22,920 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:35:22,920 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:35:22,920 INFO train_multi TF=ALL: new best val=0.6266 — saved
2026-04-25 05:35:41,017 INFO train_multi TF=ALL epoch 17/50 train=0.6223 val=0.6283
2026-04-25 05:35:59,315 INFO train_multi TF=ALL epoch 18/50 train=0.6197 val=0.6256
2026-04-25 05:35:59,320 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:35:59,320 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:35:59,320 INFO train_multi TF=ALL: new best val=0.6256 — saved
2026-04-25 05:36:17,430 INFO train_multi TF=ALL epoch 19/50 train=0.6169 val=0.6275
2026-04-25 05:36:35,559 INFO train_multi TF=ALL epoch 20/50 train=0.6144 val=0.6208
2026-04-25 05:36:35,564 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:36:35,564 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:36:35,564 INFO train_multi TF=ALL: new best val=0.6208 — saved
2026-04-25 05:36:53,981 INFO train_multi TF=ALL epoch 21/50 train=0.6126 val=0.6195
2026-04-25 05:36:53,985 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:36:53,985 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:36:53,985 INFO train_multi TF=ALL: new best val=0.6195 — saved
2026-04-25 05:37:12,001 INFO train_multi TF=ALL epoch 22/50 train=0.6105 val=0.6193
2026-04-25 05:37:12,006 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:37:12,006 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:37:12,006 INFO train_multi TF=ALL: new best val=0.6193 — saved
2026-04-25 05:37:30,229 INFO train_multi TF=ALL epoch 23/50 train=0.6086 val=0.6199
2026-04-25 05:37:48,510 INFO train_multi TF=ALL epoch 24/50 train=0.6069 val=0.6204
2026-04-25 05:38:06,461 INFO train_multi TF=ALL epoch 25/50 train=0.6050 val=0.6203
2026-04-25 05:38:24,252 INFO train_multi TF=ALL epoch 26/50 train=0.6035 val=0.6157
2026-04-25 05:38:24,257 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:38:24,257 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:38:24,257 INFO train_multi TF=ALL: new best val=0.6157 — saved
2026-04-25 05:38:42,452 INFO train_multi TF=ALL epoch 27/50 train=0.6017 val=0.6150
2026-04-25 05:38:42,456 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:38:42,457 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:38:42,457 INFO train_multi TF=ALL: new best val=0.6150 — saved
2026-04-25 05:39:00,798 INFO train_multi TF=ALL epoch 28/50 train=0.6002 val=0.6180
2026-04-25 05:39:19,064 INFO train_multi TF=ALL epoch 29/50 train=0.5984 val=0.6131
2026-04-25 05:39:19,069 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:39:19,069 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:39:19,069 INFO train_multi TF=ALL: new best val=0.6131 — saved
2026-04-25 05:39:37,314 INFO train_multi TF=ALL epoch 30/50 train=0.5966 val=0.6157
2026-04-25 05:39:55,413 INFO train_multi TF=ALL epoch 31/50 train=0.5952 val=0.6130
2026-04-25 05:39:55,418 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:39:55,418 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:39:55,418 INFO train_multi TF=ALL: new best val=0.6130 — saved
2026-04-25 05:40:13,752 INFO train_multi TF=ALL epoch 32/50 train=0.5934 val=0.6125
2026-04-25 05:40:13,757 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:40:13,757 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:40:13,757 INFO train_multi TF=ALL: new best val=0.6125 — saved
2026-04-25 05:40:32,144 INFO train_multi TF=ALL epoch 33/50 train=0.5918 val=0.6152
2026-04-25 05:40:50,595 INFO train_multi TF=ALL epoch 34/50 train=0.5899 val=0.6133
2026-04-25 05:41:09,005 INFO train_multi TF=ALL epoch 35/50 train=0.5880 val=0.6125
2026-04-25 05:41:09,009 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:41:09,009 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:41:09,009 INFO train_multi TF=ALL: new best val=0.6125 — saved
2026-04-25 05:41:27,154 INFO train_multi TF=ALL epoch 36/50 train=0.5869 val=0.6177
2026-04-25 05:41:45,583 INFO train_multi TF=ALL epoch 37/50 train=0.5853 val=0.6116
2026-04-25 05:41:45,588 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:41:45,588 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:41:45,588 INFO train_multi TF=ALL: new best val=0.6116 — saved
2026-04-25 05:42:03,759 INFO train_multi TF=ALL epoch 38/50 train=0.5836 val=0.6108
2026-04-25 05:42:03,764 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:42:03,764 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:42:03,764 INFO train_multi TF=ALL: new best val=0.6108 — saved
2026-04-25 05:42:22,213 INFO train_multi TF=ALL epoch 39/50 train=0.5822 val=0.6081
2026-04-25 05:42:22,218 INFO WeightsManifest written → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-25 05:42:22,218 INFO GRULSTMPredictor saved to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:42:22,218 INFO train_multi TF=ALL: new best val=0.6081 — saved
2026-04-25 05:42:40,474 INFO train_multi TF=ALL epoch 40/50 train=0.5805 val=0.6154
2026-04-25 05:42:58,841 INFO train_multi TF=ALL epoch 41/50 train=0.5789 val=0.6102
2026-04-25 05:43:17,266 INFO train_multi TF=ALL epoch 42/50 train=0.5770 val=0.6155
2026-04-25 05:43:35,626 INFO train_multi TF=ALL epoch 43/50 train=0.5754 val=0.6165
2026-04-25 05:43:54,050 INFO train_multi TF=ALL epoch 44/50 train=0.5735 val=0.6206
2026-04-25 05:43:54,050 INFO train_multi TF=ALL early stop at epoch 44
2026-04-25 05:43:54,189 INFO === VectorStore: building similarity indices ===
2026-04-25 05:43:54,193 INFO Loading faiss with AVX512 support.
2026-04-25 05:43:54,217 INFO Successfully loaded faiss with AVX512 support.
2026-04-25 05:43:54,222 INFO VectorStore: CPU FAISS index built (dim=74)
2026-04-25 05:43:54,223 INFO VectorStore: CPU FAISS index built (dim=34)
2026-04-25 05:43:54,223 INFO VectorStore: CPU FAISS index built (dim=64)
2026-04-25 05:43:54,967 INFO GRULSTMPredictor loaded from /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-25 05:43:54,973 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:44:01,023 INFO VectorStore trade_patterns: +50000 vectors for AUDUSD
2026-04-25 05:44:01,145 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:44:01,148 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:44:01,148 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:44:01,149 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:44:01,149 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:44:40,151 INFO VectorStore market_structures: +65447 vectors (34-dim 4H) for AUDUSD
2026-04-25 05:44:46,378 INFO VectorStore regime_embeddings: +13090 vectors for AUDUSD
2026-04-25 05:44:46,650 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:44:52,621 INFO VectorStore trade_patterns: +50000 vectors for EURGBP
2026-04-25 05:44:52,767 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:44:52,770 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:44:52,771 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:44:52,771 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:44:52,772 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:45:32,113 INFO VectorStore market_structures: +65448 vectors (34-dim 4H) for EURGBP
2026-04-25 05:45:38,432 INFO VectorStore regime_embeddings: +13090 vectors for EURGBP
2026-04-25 05:45:38,716 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:45:44,683 INFO VectorStore trade_patterns: +50000 vectors for EURJPY
2026-04-25 05:45:44,828 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:45:44,830 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:45:44,831 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:45:44,831 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:45:44,832 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:46:23,525 INFO VectorStore market_structures: +65453 vectors (34-dim 4H) for EURJPY
2026-04-25 05:46:29,751 INFO VectorStore regime_embeddings: +13091 vectors for EURJPY
2026-04-25 05:46:30,003 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:46:36,037 INFO VectorStore trade_patterns: +50000 vectors for EURUSD
2026-04-25 05:46:36,178 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:46:36,180 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:46:36,181 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:46:36,181 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:46:36,182 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:47:15,436 INFO VectorStore market_structures: +65451 vectors (34-dim 4H) for EURUSD
2026-04-25 05:47:21,574 INFO VectorStore regime_embeddings: +13091 vectors for EURUSD
2026-04-25 05:47:21,851 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:47:27,906 INFO VectorStore trade_patterns: +50000 vectors for GBPJPY
2026-04-25 05:47:28,039 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:47:28,041 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:47:28,042 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:47:28,042 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:47:28,043 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:48:06,999 INFO VectorStore market_structures: +65452 vectors (34-dim 4H) for GBPJPY
2026-04-25 05:48:13,184 INFO VectorStore regime_embeddings: +13091 vectors for GBPJPY
2026-04-25 05:48:13,484 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:48:19,464 INFO VectorStore trade_patterns: +50000 vectors for GBPUSD
2026-04-25 05:48:19,616 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:48:19,619 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:48:19,619 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:48:19,620 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:48:19,620 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:48:58,589 INFO VectorStore market_structures: +65451 vectors (34-dim 4H) for GBPUSD
2026-04-25 05:49:04,768 INFO VectorStore regime_embeddings: +13091 vectors for GBPUSD
2026-04-25 05:49:05,046 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:49:10,989 INFO VectorStore trade_patterns: +50000 vectors for NZDUSD
2026-04-25 05:49:11,123 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-25 05:49:11,125 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-25 05:49:11,125 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-25 05:49:11,126 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-25 05:49:11,126 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-25 05:49:49,631 INFO VectorStore market_structures: +65451 vectors (34-dim 4H) for NZDUSD
2026-04-25 05:49:56,086 INFO VectorStore regime_embeddings: +13091 vectors for NZDUSD
2026-04-25 05:49:56,371 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:50:02,236 INFO VectorStore trade_patterns: +50000 vectors for USDCAD
2026-04-25 05:50:02,388 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:50:02,390 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:50:02,392 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:50:02,392 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:50:02,392 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:50:41,421 INFO VectorStore market_structures: +65452 vectors (34-dim 4H) for USDCAD
2026-04-25 05:50:47,650 INFO VectorStore regime_embeddings: +13091 vectors for USDCAD
2026-04-25 05:50:47,909 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:50:53,815 INFO VectorStore trade_patterns: +50000 vectors for USDCHF
2026-04-25 05:50:53,957 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:50:53,960 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:50:53,960 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:50:53,961 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:50:53,961 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:51:32,612 INFO VectorStore market_structures: +65454 vectors (34-dim 4H) for USDCHF
2026-04-25 05:51:38,799 INFO VectorStore regime_embeddings: +13091 vectors for USDCHF
2026-04-25 05:51:39,067 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-25 05:51:45,003 INFO VectorStore trade_patterns: +50000 vectors for USDJPY
2026-04-25 05:51:45,142 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:51:45,145 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:51:45,146 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:51:45,146 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:51:45,146 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-25 05:52:23,937 INFO VectorStore market_structures: +65461 vectors (34-dim 4H) for USDJPY
2026-04-25 05:52:30,071 INFO VectorStore regime_embeddings: +13093 vectors for USDJPY
2026-04-25 05:52:30,384 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-25 05:52:42,850 INFO VectorStore trade_patterns: +50000 vectors for XAUUSD
2026-04-25 05:52:43,005 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-25 05:52:43,008 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-25 05:52:43,010 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-25 05:52:43,010 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-25 05:52:43,011 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-25 05:54:09,318 INFO VectorStore market_structures: +59006 vectors (34-dim 4H) for XAUUSD
2026-04-25 05:54:22,190 INFO VectorStore regime_embeddings: +12828 vectors for XAUUSD
2026-04-25 05:54:24,968 INFO VectorStore: saved 2670790 total vectors to /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/vector_store
2026-04-25 05:54:24,968 INFO VectorStore saved: {'trade_patterns': 1100000, 'market_structures': 1427052, 'regime_embeddings': 143738}
2026-04-25 05:54:25,202 INFO Retrain complete.
2026-04-25 05:54:27,157 INFO Model gru: SUCCESS
2026-04-25 05:54:27,157 INFO   [OK] gru_lstm → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-25 05:54:27,157 WARNING   [MISSING] regime_classifier → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/regime_classifier.pkl
2026-04-25 05:54:27,157 WARNING   [MISSING] quality_scorer → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-25 05:54:27,158 INFO   [OK] rl_ppo → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-25 05:54:27,158 WARNING Missing weights: ['regime_classifier', 'quality_scorer'] — run retrain_incremental.py for each
2026-04-25 05:54:27,159 INFO Saved 18 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  Cleared journal for fresh Round 1 run
  START Round 1 - Backtest (val)
2026-04-25 05:54:28,383 INFO === STEP 6: BACKTEST + REINFORCED TRAINING (3 rounds, window=round1) ===
2026-04-25 05:54:28,384 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-04-25 05:54:28,385 INFO Cleared existing journal for fresh reinforced training run
2026-04-25 05:54:28,386 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-25 05:54:28,386 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)

Backtest results → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260425_055428.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)                0   0.0%   0.00    0.0%  0.0%  0.0%   0.0%    0.00

Calibration Summary:
  all          [OK] Calibration OK — p_win correlates with actual win rate.
  ml_trader    [OK] Calibration OK — p_win correlates with actual win rate.
2026-04-25 05:54:32,008 INFO Round 1 backtest — 0 trades | avg WR=0.0% | avg PF=0.00 | avg Sharpe=0.00
2026-04-25 05:54:32,008 INFO   ml_trader: 0 trades | WR=0.0% | PF=0.00 | Return=0.0% | DD=0.0% | Sharpe=0.00
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 0
ERROR  trade_log is empty.
2026-04-25 05:54:32,225 WARNING Round 1: trade_log is empty — nothing to journal
2026-04-25 05:54:32,225 WARNING Round 1: no trades to journal — skipping retrain
2026-04-25 05:54:32,225 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-25 05:54:32,225 INFO Round 2 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)

Backtest results → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260425_055432.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)                0   0.0%   0.00    0.0%  0.0%  0.0%   0.0%    0.00

Calibration Summary:
  all          [OK] Calibration OK — p_win correlates with actual win rate.
  ml_trader    [OK] Calibration OK — p_win correlates with actual win rate.
2026-04-25 05:54:35,530 INFO Round 2 backtest — 0 trades | avg WR=0.0% | avg PF=0.00 | avg Sharpe=0.00
2026-04-25 05:54:35,530 INFO   ml_trader: 0 trades | WR=0.0% | PF=0.00 | Return=0.0% | DD=0.0% | Sharpe=0.00
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 0
ERROR  trade_log is empty.
2026-04-25 05:54:35,748 WARNING Round 2: trade_log is empty — nothing to journal
2026-04-25 05:54:35,748 WARNING Round 2: no trades to journal — skipping retrain
2026-04-25 05:54:35,748 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-04-25 05:54:35,748 INFO Round 3 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)

Backtest results → /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260425_055436.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)                0   0.0%   0.00    0.0%  0.0%  0.0%   0.0%    0.00

Calibration Summary:
  all          [OK] Calibration OK — p_win correlates with actual win rate.
  ml_trader    [OK] Calibration OK — p_win correlates with actual win rate.
2026-04-25 05:54:38,985 INFO Round 3 backtest — 0 trades | avg WR=0.0% | avg PF=0.00 | avg Sharpe=0.00
2026-04-25 05:54:38,985 INFO   ml_trader: 0 trades | WR=0.0% | PF=0.00 | Return=0.0% | DD=0.0% | Sharpe=0.00
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 0
ERROR  trade_log is empty.
2026-04-25 05:54:39,201 WARNING Round 3: trade_log is empty — nothing to journal
2026-04-25 05:54:39,201 WARNING Round 3: no trades to journal — skipping retrain
2026-04-25 05:54:39,201 INFO Improvement round 1 → 3: WR +0.0% | PF +0.000 | Sharpe +0.000

======================================================================
  BACKTEST + REINFORCED TRAINING COMPLETE  (3 rounds)
======================================================================
  Round     Trades       WR      PF   Sharpe
  ------------------------------------------
  Round 1          0      0.0%    0.000     0.000
  Round 2          0      0.0%    0.000     0.000
  Round 3          0      0.0%    0.000     0.000

  Net improvement (round 1 → 3):
    Win rate:      +0.0%
    Profit factor: +0.000
    Sharpe:        +0.000

  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 0 entries

=== Round 1 → Retrain Quality + RL ===
  START Round 1 - Quality+RL retrain
2026-04-25 05:54:39,446 INFO === STEP 7b: QUALITY + RL TRAINING ===
Traceback (most recent call last):
  File "/kaggle/working/Multi-Bot/trading-system/pipeline/step7b_train.py", line 110, in <module>
    main()
  File "/kaggle/working/Multi-Bot/trading-system/pipeline/step7b_train.py", line 70, in main
    raise RuntimeError(
RuntimeError: Journal missing at /kaggle/working/remote/Multi-Bot/trading-system/trading-engine/logs/trade_journal_detailed.jsonl — run step6_backtest.py first
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/kaggle/working/Multi-Bot/trading-system/kaggle_train.py in <module>
    266 # ─── Train Quality + RL on Round 1 journal ───────────────────────────────────
    267 print("\n=== Round 1 → Retrain Quality + RL ===")
--> 268 run_step(
    269     "Round 1 - Quality+RL retrain",
    270     "step7b_train.py",

/kaggle/working/Multi-Bot/trading-system/kaggle_train.py in run_step(name, script, done_check, extra_env)
    152     )
    153     if result.returncode != 0:
--> 154         raise RuntimeError(f"{name} FAILED (exit {result.returncode})")
    155     print(f"  DONE  {name}")
    156 

RuntimeError: Round 1 - Quality+RL retrain FAILED (exit 1)