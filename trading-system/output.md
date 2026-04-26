  Cleared done-check: training_summary.json
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
2026-04-26 13:58:30,037 INFO Loading feature-engineered data...
2026-04-26 13:58:30,546 INFO Loaded 221743 rows, 202 features
2026-04-26 13:58:30,546 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-26 13:58:30,547 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-26 13:58:30,547 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-26 13:58:30,547 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-26 13:58:30,547 INFO No leakage confirmed: train < val < test timestamps

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
2026-04-26 13:58:32,974 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-26 13:58:32,974 INFO --- Training regime ---
2026-04-26 13:58:32,974 INFO Running retrain --model regime
2026-04-26 13:58:33,160 INFO retrain environment: KAGGLE
2026-04-26 13:58:34,968 INFO Device: CUDA (2 GPU(s))
2026-04-26 13:58:34,979 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 13:58:34,979 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 13:58:34,979 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 13:58:34,980 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 13:58:34,981 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-26 13:58:35,154 INFO NumExpr defaulting to 4 threads.
2026-04-26 13:58:35,400 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 13:58:35,401 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 13:58:35,401 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 13:58:35,401 INFO Regime phase macro_correlations: 0.0s
2026-04-26 13:58:35,401 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-26 13:58:35,439 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 13:58:35,440 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 13:58:35,468 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 13:58:35,484 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 13:58:35,511 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 13:58:35,530 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 13:58:35,558 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 13:58:35,574 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 13:58:35,600 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 13:58:35,617 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 13:58:35,644 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 13:58:35,662 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 13:58:35,686 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 13:58:35,701 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 13:58:35,725 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 13:58:35,740 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 13:58:35,764 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 13:58:35,779 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 13:58:35,801 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 13:58:35,816 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 13:58:35,840 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 13:58:35,859 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 13:58:35,899 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 13:58:36,685 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 13:59:00,783 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 13:59:00,785 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 24.9s
2026-04-26 13:59:00,785 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 13:59:12,085 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 13:59:12,088 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 11.3s
2026-04-26 13:59:12,089 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 13:59:20,512 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 13:59:20,513 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 8.4s
2026-04-26 13:59:20,513 INFO Regime phase GMM HTF total: 44.6s
2026-04-26 13:59:20,515 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 14:00:35,127 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 14:00:35,134 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 74.6s
2026-04-26 14:00:35,137 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 14:01:08,456 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 14:01:08,461 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 33.3s
2026-04-26 14:01:08,462 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 14:01:31,585 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 14:01:31,586 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 23.1s
2026-04-26 14:01:31,586 INFO Regime phase GMM LTF total: 131.1s
2026-04-26 14:01:31,710 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-26 14:01:31,712 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:31,713 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:31,714 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:31,715 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:31,716 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:31,717 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:31,718 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:31,719 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:31,719 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:31,720 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:31,722 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:01:31,852 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:31,900 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:31,901 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:31,902 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:31,910 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:31,911 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:32,347 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-26 14:01:32,349 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-26 14:01:32,551 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:32,587 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:32,588 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:32,589 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:32,597 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:32,598 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:33,004 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-26 14:01:33,005 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-26 14:01:33,218 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:33,254 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:33,255 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:33,256 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:33,264 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:33,265 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:33,661 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-26 14:01:33,662 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-26 14:01:33,858 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:33,897 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:33,898 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:33,899 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:33,907 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:33,908 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:34,316 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-26 14:01:34,317 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-26 14:01:34,531 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:34,583 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:34,584 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:34,584 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:34,596 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:34,597 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:35,002 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-26 14:01:35,004 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-26 14:01:35,202 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:35,237 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:35,238 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:35,239 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:35,246 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:35,247 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:35,635 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-26 14:01:35,637 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-26 14:01:35,814 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:01:35,843 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:01:35,844 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:01:35,845 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:01:35,852 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:01:35,853 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:36,246 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-26 14:01:36,248 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-26 14:01:36,449 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:36,484 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:36,485 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:36,486 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:36,495 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:36,496 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:36,889 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-26 14:01:36,890 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-26 14:01:37,085 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:37,120 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:37,121 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:37,122 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:37,130 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:37,131 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:37,522 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-26 14:01:37,523 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-26 14:01:37,727 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:37,764 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:37,765 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:37,765 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:37,773 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:37,774 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:38,171 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-26 14:01:38,173 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-26 14:01:38,464 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:01:38,531 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:01:38,532 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:01:38,532 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:01:38,544 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:01:38,545 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:01:39,392 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-26 14:01:39,394 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-26 14:01:39,600 INFO Regime phase HTF dataset build: 7.9s (103290 samples)
2026-04-26 14:01:39,617 INFO RegimeClassifier[mode=htf_bias]: 103290 samples, classes={'BIAS_UP': 18622, 'BIAS_DOWN': 18286, 'BIAS_NEUTRAL': 66382}, device=cuda
2026-04-26 14:01:39,618 INFO RegimeClassifier: sample weights — mean=0.360  ambiguous(<0.4)=69.5%
2026-04-26 14:01:39,815 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-04-26 14:01:39,816 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 14:01:42,277 INFO Regime epoch  1/50 — tr=0.3314 va=2.1237 acc=0.422 per_class={'BIAS_UP': 0.556, 'BIAS_DOWN': 0.275, 'BIAS_NEUTRAL': 0.411}
2026-04-26 14:01:42,467 INFO Regime epoch  2/50 — tr=0.3273 va=2.0519 acc=0.515
2026-04-26 14:01:42,652 INFO Regime epoch  3/50 — tr=0.3192 va=2.0057 acc=0.552
2026-04-26 14:01:42,836 INFO Regime epoch  4/50 — tr=0.3089 va=1.9789 acc=0.580
2026-04-26 14:01:43,042 INFO Regime epoch  5/50 — tr=0.2968 va=1.9471 acc=0.589 per_class={'BIAS_UP': 0.858, 'BIAS_DOWN': 0.892, 'BIAS_NEUTRAL': 0.416}
2026-04-26 14:01:43,242 INFO Regime epoch  6/50 — tr=0.2862 va=1.9102 acc=0.581
2026-04-26 14:01:43,444 INFO Regime epoch  7/50 — tr=0.2788 va=1.8853 acc=0.575
2026-04-26 14:01:43,634 INFO Regime epoch  8/50 — tr=0.2723 va=1.8642 acc=0.579
2026-04-26 14:01:43,839 INFO Regime epoch  9/50 — tr=0.2674 va=1.8573 acc=0.577
2026-04-26 14:01:44,056 INFO Regime epoch 10/50 — tr=0.2633 va=1.8369 acc=0.589 per_class={'BIAS_UP': 0.949, 'BIAS_DOWN': 0.981, 'BIAS_NEUTRAL': 0.361}
2026-04-26 14:01:44,263 INFO Regime epoch 11/50 — tr=0.2602 va=1.8354 acc=0.592
2026-04-26 14:01:44,450 INFO Regime epoch 12/50 — tr=0.2577 va=1.8272 acc=0.598
2026-04-26 14:01:44,672 INFO Regime epoch 13/50 — tr=0.2559 va=1.8217 acc=0.602
2026-04-26 14:01:44,864 INFO Regime epoch 14/50 — tr=0.2543 va=1.8167 acc=0.609
2026-04-26 14:01:45,066 INFO Regime epoch 15/50 — tr=0.2529 va=1.8138 acc=0.612 per_class={'BIAS_UP': 0.957, 'BIAS_DOWN': 0.984, 'BIAS_NEUTRAL': 0.395}
2026-04-26 14:01:45,252 INFO Regime epoch 16/50 — tr=0.2516 va=1.8138 acc=0.614
2026-04-26 14:01:45,443 INFO Regime epoch 17/50 — tr=0.2508 va=1.8112 acc=0.620
2026-04-26 14:01:45,638 INFO Regime epoch 18/50 — tr=0.2496 va=1.8092 acc=0.625
2026-04-26 14:01:45,832 INFO Regime epoch 19/50 — tr=0.2493 va=1.8080 acc=0.628
2026-04-26 14:01:46,046 INFO Regime epoch 20/50 — tr=0.2487 va=1.8091 acc=0.627 per_class={'BIAS_UP': 0.967, 'BIAS_DOWN': 0.985, 'BIAS_NEUTRAL': 0.414}
2026-04-26 14:01:46,236 INFO Regime epoch 21/50 — tr=0.2478 va=1.8078 acc=0.629
2026-04-26 14:01:46,431 INFO Regime epoch 22/50 — tr=0.2473 va=1.8084 acc=0.630
2026-04-26 14:01:46,621 INFO Regime epoch 23/50 — tr=0.2469 va=1.8087 acc=0.630
2026-04-26 14:01:46,810 INFO Regime epoch 24/50 — tr=0.2466 va=1.8055 acc=0.633
2026-04-26 14:01:47,011 INFO Regime epoch 25/50 — tr=0.2464 va=1.8055 acc=0.636 per_class={'BIAS_UP': 0.974, 'BIAS_DOWN': 0.986, 'BIAS_NEUTRAL': 0.427}
2026-04-26 14:01:47,206 INFO Regime epoch 26/50 — tr=0.2460 va=1.8048 acc=0.636
2026-04-26 14:01:47,404 INFO Regime epoch 27/50 — tr=0.2458 va=1.8063 acc=0.635
2026-04-26 14:01:47,587 INFO Regime epoch 28/50 — tr=0.2454 va=1.8082 acc=0.634
2026-04-26 14:01:47,779 INFO Regime epoch 29/50 — tr=0.2455 va=1.8039 acc=0.640
2026-04-26 14:01:47,991 INFO Regime epoch 30/50 — tr=0.2450 va=1.8036 acc=0.640 per_class={'BIAS_UP': 0.977, 'BIAS_DOWN': 0.986, 'BIAS_NEUTRAL': 0.431}
2026-04-26 14:01:48,193 INFO Regime epoch 31/50 — tr=0.2449 va=1.8029 acc=0.639
2026-04-26 14:01:48,384 INFO Regime epoch 32/50 — tr=0.2449 va=1.8034 acc=0.641
2026-04-26 14:01:48,592 INFO Regime epoch 33/50 — tr=0.2447 va=1.8046 acc=0.640
2026-04-26 14:01:48,794 INFO Regime epoch 34/50 — tr=0.2445 va=1.8042 acc=0.640
2026-04-26 14:01:49,009 INFO Regime epoch 35/50 — tr=0.2445 va=1.8053 acc=0.641 per_class={'BIAS_UP': 0.977, 'BIAS_DOWN': 0.988, 'BIAS_NEUTRAL': 0.433}
2026-04-26 14:01:49,216 INFO Regime epoch 36/50 — tr=0.2443 va=1.8052 acc=0.642
2026-04-26 14:01:49,416 INFO Regime epoch 37/50 — tr=0.2444 va=1.8040 acc=0.640
2026-04-26 14:01:49,617 INFO Regime epoch 38/50 — tr=0.2443 va=1.8037 acc=0.642
2026-04-26 14:01:49,804 INFO Regime epoch 39/50 — tr=0.2445 va=1.8036 acc=0.641
2026-04-26 14:01:50,018 INFO Regime epoch 40/50 — tr=0.2441 va=1.8055 acc=0.641 per_class={'BIAS_UP': 0.978, 'BIAS_DOWN': 0.99, 'BIAS_NEUTRAL': 0.432}
2026-04-26 14:01:50,222 INFO Regime epoch 41/50 — tr=0.2442 va=1.8022 acc=0.645
2026-04-26 14:01:50,421 INFO Regime epoch 42/50 — tr=0.2442 va=1.8011 acc=0.647
2026-04-26 14:01:50,625 INFO Regime epoch 43/50 — tr=0.2442 va=1.8031 acc=0.643
2026-04-26 14:01:50,821 INFO Regime epoch 44/50 — tr=0.2441 va=1.7999 acc=0.647
2026-04-26 14:01:51,026 INFO Regime epoch 45/50 — tr=0.2442 va=1.8018 acc=0.645 per_class={'BIAS_UP': 0.979, 'BIAS_DOWN': 0.988, 'BIAS_NEUTRAL': 0.439}
2026-04-26 14:01:51,221 INFO Regime epoch 46/50 — tr=0.2441 va=1.8033 acc=0.644
2026-04-26 14:01:51,410 INFO Regime epoch 47/50 — tr=0.2440 va=1.7991 acc=0.647
2026-04-26 14:01:51,602 INFO Regime epoch 48/50 — tr=0.2441 va=1.8024 acc=0.643
2026-04-26 14:01:51,785 INFO Regime epoch 49/50 — tr=0.2441 va=1.8029 acc=0.644
2026-04-26 14:01:51,989 INFO Regime epoch 50/50 — tr=0.2442 va=1.8042 acc=0.642 per_class={'BIAS_UP': 0.978, 'BIAS_DOWN': 0.99, 'BIAS_NEUTRAL': 0.434}
2026-04-26 14:01:52,003 WARNING RegimeClassifier accuracy 0.65 < 0.65 threshold
2026-04-26 14:01:52,006 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 14:01:52,007 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 14:01:52,008 INFO Regime phase HTF train: 12.4s
2026-04-26 14:01:52,141 INFO Regime HTF complete: acc=0.647, n=103290
2026-04-26 14:01:52,142 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:01:52,310 INFO Rule labels LTF_BEHAVIOUR [4H]: {'TRENDING': 8970, 'RANGING': 4592, 'CONSOLIDATING': 2247, 'VOLATILE': 4008}  ambiguous=4539 (total=19817)  short_runs_zeroed=570
2026-04-26 14:01:52,313 INFO Regime[4H] persistence (avg bars/run) on XAUUSD 4H:
{0: 12.121621621621621, 1: 6.4858757062146895, 2: 5.575682382133995, 3: 10.329896907216495}
2026-04-26 14:01:52,314 INFO Regime[4H] return separation on XAUUSD 4H:
{0: 5.9502411776840314e-05, 1: 5.664222849552333e-05, 2: -6.138442068010657e-05, 3: 4.239843820627722e-05}
2026-04-26 14:01:52,315 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-26 14:01:52,316 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:52,318 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:52,320 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:52,321 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:52,323 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:52,324 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:52,326 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:52,327 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:52,329 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:52,331 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:52,334 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:01:52,345 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:52,347 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:52,348 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:52,348 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:52,349 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:52,350 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:53,015 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-26 14:01:53,018 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-26 14:01:53,174 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:53,176 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:53,177 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:53,178 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:53,178 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:53,180 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:53,807 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-26 14:01:53,811 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-26 14:01:53,967 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:53,969 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:53,970 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:53,970 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:53,971 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:53,973 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:54,647 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-26 14:01:54,650 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-26 14:01:54,799 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:54,802 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:54,803 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:54,803 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:54,803 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:54,806 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:55,430 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-26 14:01:55,433 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-26 14:01:55,589 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:55,591 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:55,592 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:55,593 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:55,593 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:55,595 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:56,219 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-26 14:01:56,222 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-26 14:01:56,360 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:56,362 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:56,363 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:56,364 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:56,364 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:56,366 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:56,975 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-26 14:01:56,978 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-26 14:01:57,113 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:01:57,115 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:01:57,116 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:01:57,116 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:01:57,117 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:01:57,118 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:57,721 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-26 14:01:57,724 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-26 14:01:57,858 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:57,860 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:57,861 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:57,862 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:57,862 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:57,864 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:58,465 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-26 14:01:58,468 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-26 14:01:58,606 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:58,610 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:58,610 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:58,611 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:58,611 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:58,613 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:59,228 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-26 14:01:59,230 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-26 14:01:59,379 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:59,382 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:59,383 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:59,383 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:59,383 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:01:59,385 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:01:59,995 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-26 14:01:59,998 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-26 14:02:00,151 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:02:00,154 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:02:00,156 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:02:00,156 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:02:00,157 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:02:00,160 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:02:01,476 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-26 14:02:01,482 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-26 14:02:01,804 INFO Regime phase LTF dataset build: 9.5s (401471 samples)
2026-04-26 14:02:01,862 INFO RegimeClassifier[mode=ltf_behaviour]: 401471 samples, classes={'TRENDING': 160094, 'RANGING': 105286, 'CONSOLIDATING': 53524, 'VOLATILE': 82567}, device=cuda
2026-04-26 14:02:01,863 INFO RegimeClassifier: sample weights — mean=0.505  ambiguous(<0.4)=33.0%
2026-04-26 14:02:01,865 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-04-26 14:02:01,866 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 14:02:02,628 INFO Regime epoch  1/50 — tr=0.6437 va=2.0848 acc=0.271 per_class={'TRENDING': 0.165, 'RANGING': 0.272, 'CONSOLIDATING': 0.109, 'VOLATILE': 0.588}
2026-04-26 14:02:03,325 INFO Regime epoch  2/50 — tr=0.6162 va=2.0209 acc=0.352
2026-04-26 14:02:04,033 INFO Regime epoch  3/50 — tr=0.5835 va=1.9420 acc=0.448
2026-04-26 14:02:04,761 INFO Regime epoch  4/50 — tr=0.5570 va=1.8724 acc=0.501
2026-04-26 14:02:05,497 INFO Regime epoch  5/50 — tr=0.5395 va=1.8304 acc=0.530 per_class={'TRENDING': 0.483, 'RANGING': 0.185, 'CONSOLIDATING': 0.702, 'VOLATILE': 0.957}
2026-04-26 14:02:06,222 INFO Regime epoch  6/50 — tr=0.5281 va=1.8147 acc=0.550
2026-04-26 14:02:06,941 INFO Regime epoch  7/50 — tr=0.5196 va=1.7999 acc=0.565
2026-04-26 14:02:07,642 INFO Regime epoch  8/50 — tr=0.5132 va=1.7907 acc=0.581
2026-04-26 14:02:08,384 INFO Regime epoch  9/50 — tr=0.5080 va=1.7752 acc=0.601
2026-04-26 14:02:09,165 INFO Regime epoch 10/50 — tr=0.5037 va=1.7582 acc=0.615 per_class={'TRENDING': 0.673, 'RANGING': 0.174, 'CONSOLIDATING': 0.814, 'VOLATILE': 0.944}
2026-04-26 14:02:09,875 INFO Regime epoch 11/50 — tr=0.4997 va=1.7405 acc=0.629
2026-04-26 14:02:10,572 INFO Regime epoch 12/50 — tr=0.4964 va=1.7206 acc=0.642
2026-04-26 14:02:11,313 INFO Regime epoch 13/50 — tr=0.4936 va=1.7032 acc=0.652
2026-04-26 14:02:12,009 INFO Regime epoch 14/50 — tr=0.4911 va=1.6869 acc=0.661
2026-04-26 14:02:12,747 INFO Regime epoch 15/50 — tr=0.4891 va=1.6730 acc=0.669 per_class={'TRENDING': 0.764, 'RANGING': 0.223, 'CONSOLIDATING': 0.868, 'VOLATILE': 0.93}
2026-04-26 14:02:13,464 INFO Regime epoch 16/50 — tr=0.4872 va=1.6608 acc=0.677
2026-04-26 14:02:14,175 INFO Regime epoch 17/50 — tr=0.4852 va=1.6530 acc=0.680
2026-04-26 14:02:14,918 INFO Regime epoch 18/50 — tr=0.4838 va=1.6441 acc=0.685
2026-04-26 14:02:15,643 INFO Regime epoch 19/50 — tr=0.4827 va=1.6360 acc=0.691
2026-04-26 14:02:16,407 INFO Regime epoch 20/50 — tr=0.4815 va=1.6274 acc=0.695 per_class={'TRENDING': 0.792, 'RANGING': 0.263, 'CONSOLIDATING': 0.912, 'VOLATILE': 0.922}
2026-04-26 14:02:17,154 INFO Regime epoch 21/50 — tr=0.4806 va=1.6211 acc=0.699
2026-04-26 14:02:17,894 INFO Regime epoch 22/50 — tr=0.4799 va=1.6172 acc=0.701
2026-04-26 14:02:18,608 INFO Regime epoch 23/50 — tr=0.4790 va=1.6142 acc=0.704
2026-04-26 14:02:19,305 INFO Regime epoch 24/50 — tr=0.4785 va=1.6081 acc=0.708
2026-04-26 14:02:20,075 INFO Regime epoch 25/50 — tr=0.4779 va=1.6048 acc=0.708 per_class={'TRENDING': 0.808, 'RANGING': 0.284, 'CONSOLIDATING': 0.929, 'VOLATILE': 0.915}
2026-04-26 14:02:20,772 INFO Regime epoch 26/50 — tr=0.4774 va=1.6032 acc=0.709
2026-04-26 14:02:21,490 INFO Regime epoch 27/50 — tr=0.4770 va=1.6036 acc=0.711
2026-04-26 14:02:22,180 INFO Regime epoch 28/50 — tr=0.4768 va=1.5986 acc=0.713
2026-04-26 14:02:22,900 INFO Regime epoch 29/50 — tr=0.4764 va=1.5976 acc=0.714
2026-04-26 14:02:23,656 INFO Regime epoch 30/50 — tr=0.4760 va=1.5983 acc=0.714 per_class={'TRENDING': 0.817, 'RANGING': 0.294, 'CONSOLIDATING': 0.936, 'VOLATILE': 0.91}
2026-04-26 14:02:24,368 INFO Regime epoch 31/50 — tr=0.4757 va=1.5951 acc=0.715
2026-04-26 14:02:25,095 INFO Regime epoch 32/50 — tr=0.4754 va=1.5929 acc=0.716
2026-04-26 14:02:25,809 INFO Regime epoch 33/50 — tr=0.4754 va=1.5948 acc=0.716
2026-04-26 14:02:26,521 INFO Regime epoch 34/50 — tr=0.4753 va=1.5953 acc=0.715
2026-04-26 14:02:27,305 INFO Regime epoch 35/50 — tr=0.4751 va=1.5940 acc=0.715 per_class={'TRENDING': 0.82, 'RANGING': 0.296, 'CONSOLIDATING': 0.934, 'VOLATILE': 0.909}
2026-04-26 14:02:28,012 INFO Regime epoch 36/50 — tr=0.4749 va=1.5962 acc=0.715
2026-04-26 14:02:28,727 INFO Regime epoch 37/50 — tr=0.4748 va=1.5929 acc=0.716
2026-04-26 14:02:29,458 INFO Regime epoch 38/50 — tr=0.4746 va=1.5910 acc=0.717
2026-04-26 14:02:30,165 INFO Regime epoch 39/50 — tr=0.4744 va=1.5935 acc=0.716
2026-04-26 14:02:30,929 INFO Regime epoch 40/50 — tr=0.4745 va=1.5910 acc=0.717 per_class={'TRENDING': 0.826, 'RANGING': 0.296, 'CONSOLIDATING': 0.936, 'VOLATILE': 0.905}
2026-04-26 14:02:31,665 INFO Regime epoch 41/50 — tr=0.4744 va=1.5931 acc=0.716
2026-04-26 14:02:32,421 INFO Regime epoch 42/50 — tr=0.4744 va=1.5916 acc=0.716
2026-04-26 14:02:33,122 INFO Regime epoch 43/50 — tr=0.4744 va=1.5890 acc=0.718
2026-04-26 14:02:33,856 INFO Regime epoch 44/50 — tr=0.4742 va=1.5924 acc=0.716
2026-04-26 14:02:34,673 INFO Regime epoch 45/50 — tr=0.4744 va=1.5902 acc=0.718 per_class={'TRENDING': 0.825, 'RANGING': 0.298, 'CONSOLIDATING': 0.934, 'VOLATILE': 0.907}
2026-04-26 14:02:35,387 INFO Regime epoch 46/50 — tr=0.4744 va=1.5900 acc=0.718
2026-04-26 14:02:36,094 INFO Regime epoch 47/50 — tr=0.4742 va=1.5913 acc=0.717
2026-04-26 14:02:36,809 INFO Regime epoch 48/50 — tr=0.4743 va=1.5914 acc=0.717
2026-04-26 14:02:37,533 INFO Regime epoch 49/50 — tr=0.4744 va=1.5892 acc=0.717
2026-04-26 14:02:38,310 INFO Regime epoch 50/50 — tr=0.4742 va=1.5901 acc=0.717 per_class={'TRENDING': 0.824, 'RANGING': 0.297, 'CONSOLIDATING': 0.939, 'VOLATILE': 0.904}
2026-04-26 14:02:38,363 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 14:02:38,363 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 14:02:38,366 INFO Regime phase LTF train: 36.6s
2026-04-26 14:02:38,520 INFO Regime LTF complete: acc=0.718, n=401471
2026-04-26 14:02:38,524 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:02:39,041 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-26 14:02:39,046 INFO Regime[1H] persistence (avg bars/run) on XAUUSD 1H:
{0: 9.355291913830783, 1: 6.096303199751476, 2: 5.598885793871866, 3: 6.771351107094442}
2026-04-26 14:02:39,049 INFO Regime[1H] return separation on XAUUSD 1H:
{0: 2.149062259083323e-05, 1: -3.254493608872846e-06, 2: -3.039756158146958e-06, 3: 1.8102764458593193e-05}
2026-04-26 14:02:39,050 INFO Regime retrain total: 244.1s (504761 samples)
2026-04-26 14:02:39,064 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 14:02:39,064 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 14:02:39,064 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 14:02:39,064 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-26 14:02:39,065 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-26 14:02:39,065 INFO Retrain complete. Total wall-clock: 244.1s
2026-04-26 14:02:40,275 INFO Model regime: SUCCESS
2026-04-26 14:02:40,275 INFO --- Training gru ---
2026-04-26 14:02:40,276 INFO Running retrain --model gru
2026-04-26 14:02:40,618 INFO retrain environment: KAGGLE
2026-04-26 14:02:42,250 INFO Device: CUDA (2 GPU(s))
2026-04-26 14:02:42,261 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 14:02:42,261 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 14:02:42,261 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 14:02:42,262 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 14:02:42,263 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-26 14:02:42,406 INFO NumExpr defaulting to 4 threads.
2026-04-26 14:02:42,602 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 14:02:42,603 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 14:02:42,603 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 14:02:42,603 INFO GRU phase macro_correlations: 0.0s
2026-04-26 14:02:42,603 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-26 14:02:42,604 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260426_140242
2026-04-26 14:02:42,607 WARNING WeightsManifest: no manifest at /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json — assuming compatible (run a full retrain to generate one)
2026-04-26 14:02:42,751 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:42,771 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:42,786 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:42,794 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:42,795 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 14:02:42,795 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 14:02:42,795 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 14:02:42,796 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 14:02:42,797 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:42,877 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-26 14:02:42,879 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:43,123 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-26 14:02:43,154 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:43,440 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:43,578 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:43,684 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:43,899 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:43,918 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:43,933 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:43,940 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:43,941 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:44,022 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-26 14:02:44,025 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:44,306 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-26 14:02:44,322 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:44,628 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:44,762 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:44,866 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:45,066 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:45,086 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:45,101 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:45,109 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:45,110 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:45,189 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-26 14:02:45,191 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:45,439 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-26 14:02:45,455 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:45,734 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:45,876 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:45,989 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:46,192 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:46,212 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:46,227 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:46,234 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:46,235 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:46,317 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-26 14:02:46,320 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:46,585 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-26 14:02:46,609 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:46,894 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:47,029 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:47,133 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:47,325 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:47,345 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:47,360 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:47,368 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:47,369 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:47,450 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-26 14:02:47,451 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:47,705 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-26 14:02:47,720 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:48,014 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:48,146 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:48,252 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:48,446 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:48,466 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:48,481 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:48,489 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:48,490 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:48,577 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-26 14:02:48,579 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:48,832 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-26 14:02:48,848 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:49,127 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:49,271 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:49,380 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:49,559 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:02:49,577 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:02:49,591 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:02:49,598 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:02:49,599 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:49,676 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-26 14:02:49,678 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:49,931 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-26 14:02:49,944 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:50,237 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:50,373 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:50,484 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:50,697 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:50,715 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:50,729 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:50,737 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:50,738 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:50,819 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-26 14:02:50,821 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:51,062 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-26 14:02:51,080 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:51,365 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:51,510 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:51,623 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:51,830 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:51,849 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:51,864 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:51,871 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:51,872 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:51,956 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-26 14:02:51,958 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:52,214 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-26 14:02:52,229 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:52,521 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:52,662 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:52,781 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:52,997 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:53,017 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:53,034 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:53,041 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:02:53,042 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:53,123 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-26 14:02:53,125 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:53,373 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-26 14:02:53,388 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:53,677 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:53,825 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:53,939 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:02:54,242 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:02:54,268 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:02:54,286 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:02:54,296 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:02:54,297 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:02:54,459 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-26 14:02:54,462 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:02:55,035 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-26 14:02:55,083 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:02:55,681 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:02:55,907 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:02:56,056 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:02:56,192 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-26 14:02:56,421 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-04-26 14:02:56,421 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-04-26 14:02:56,421 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-26 14:02:56,421 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-26 14:03:46,760 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-26 14:03:46,760 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-26 14:03:48,115 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-26 14:03:52,385 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=5)
2026-04-26 14:04:06,886 INFO train_multi TF=ALL epoch 1/50 train=0.8502 val=0.8413
2026-04-26 14:04:06,892 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:04:06,892 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:04:06,893 INFO train_multi TF=ALL: new best val=0.8413 — saved
2026-04-26 14:04:19,318 INFO train_multi TF=ALL epoch 2/50 train=0.8228 val=0.7772
2026-04-26 14:04:19,322 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:04:19,322 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:04:19,322 INFO train_multi TF=ALL: new best val=0.7772 — saved
2026-04-26 14:04:31,967 INFO train_multi TF=ALL epoch 3/50 train=0.7095 val=0.6880
2026-04-26 14:04:31,972 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:04:31,972 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:04:31,972 INFO train_multi TF=ALL: new best val=0.6880 — saved
2026-04-26 14:04:44,790 INFO train_multi TF=ALL epoch 4/50 train=0.6897 val=0.6880
2026-04-26 14:04:44,795 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:04:44,795 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:04:44,795 INFO train_multi TF=ALL: new best val=0.6880 — saved
2026-04-26 14:04:57,405 INFO train_multi TF=ALL epoch 5/50 train=0.6890 val=0.6879
2026-04-26 14:04:57,409 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:04:57,409 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:04:57,409 INFO train_multi TF=ALL: new best val=0.6879 — saved
2026-04-26 14:05:10,199 INFO train_multi TF=ALL epoch 6/50 train=0.6888 val=0.6881
2026-04-26 14:05:23,264 INFO train_multi TF=ALL epoch 7/50 train=0.6885 val=0.6884
2026-04-26 14:05:36,136 INFO train_multi TF=ALL epoch 8/50 train=0.6884 val=0.6882
2026-04-26 14:05:49,050 INFO train_multi TF=ALL epoch 9/50 train=0.6881 val=0.6876
2026-04-26 14:05:49,055 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:05:49,055 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:05:49,055 INFO train_multi TF=ALL: new best val=0.6876 — saved
2026-04-26 14:06:01,911 INFO train_multi TF=ALL epoch 10/50 train=0.6876 val=0.6867
2026-04-26 14:06:01,916 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:06:01,916 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:06:01,916 INFO train_multi TF=ALL: new best val=0.6867 — saved
2026-04-26 14:06:14,569 INFO train_multi TF=ALL epoch 11/50 train=0.6859 val=0.6850
2026-04-26 14:06:14,575 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:06:14,575 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:06:14,576 INFO train_multi TF=ALL: new best val=0.6850 — saved
2026-04-26 14:06:27,400 INFO train_multi TF=ALL epoch 12/50 train=0.6817 val=0.6767
2026-04-26 14:06:27,404 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:06:27,404 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:06:27,404 INFO train_multi TF=ALL: new best val=0.6767 — saved
2026-04-26 14:06:40,124 INFO train_multi TF=ALL epoch 13/50 train=0.6720 val=0.6656
2026-04-26 14:06:40,128 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:06:40,128 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:06:40,128 INFO train_multi TF=ALL: new best val=0.6656 — saved
2026-04-26 14:06:52,739 INFO train_multi TF=ALL epoch 14/50 train=0.6607 val=0.6510
2026-04-26 14:06:52,743 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:06:52,744 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:06:52,744 INFO train_multi TF=ALL: new best val=0.6510 — saved
2026-04-26 14:07:05,389 INFO train_multi TF=ALL epoch 15/50 train=0.6491 val=0.6420
2026-04-26 14:07:05,394 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:07:05,394 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:07:05,394 INFO train_multi TF=ALL: new best val=0.6420 — saved
2026-04-26 14:07:18,254 INFO train_multi TF=ALL epoch 16/50 train=0.6405 val=0.6307
2026-04-26 14:07:18,258 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:07:18,258 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:07:18,258 INFO train_multi TF=ALL: new best val=0.6307 — saved
2026-04-26 14:07:31,063 INFO train_multi TF=ALL epoch 17/50 train=0.6342 val=0.6290
2026-04-26 14:07:31,067 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:07:31,067 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:07:31,067 INFO train_multi TF=ALL: new best val=0.6290 — saved
2026-04-26 14:07:43,626 INFO train_multi TF=ALL epoch 18/50 train=0.6294 val=0.6261
2026-04-26 14:07:43,630 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:07:43,630 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:07:43,631 INFO train_multi TF=ALL: new best val=0.6261 — saved
2026-04-26 14:07:56,253 INFO train_multi TF=ALL epoch 19/50 train=0.6255 val=0.6228
2026-04-26 14:07:56,257 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:07:56,257 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:07:56,257 INFO train_multi TF=ALL: new best val=0.6228 — saved
2026-04-26 14:08:08,819 INFO train_multi TF=ALL epoch 20/50 train=0.6222 val=0.6204
2026-04-26 14:08:08,824 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:08:08,824 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:08:08,824 INFO train_multi TF=ALL: new best val=0.6204 — saved
2026-04-26 14:08:21,434 INFO train_multi TF=ALL epoch 21/50 train=0.6192 val=0.6198
2026-04-26 14:08:21,439 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:08:21,439 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:08:21,439 INFO train_multi TF=ALL: new best val=0.6198 — saved
2026-04-26 14:08:34,066 INFO train_multi TF=ALL epoch 22/50 train=0.6169 val=0.6210
2026-04-26 14:08:46,454 INFO train_multi TF=ALL epoch 23/50 train=0.6144 val=0.6206
2026-04-26 14:08:59,217 INFO train_multi TF=ALL epoch 24/50 train=0.6126 val=0.6170
2026-04-26 14:08:59,221 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:08:59,221 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:08:59,221 INFO train_multi TF=ALL: new best val=0.6170 — saved
2026-04-26 14:09:11,718 INFO train_multi TF=ALL epoch 25/50 train=0.6108 val=0.6149
2026-04-26 14:09:11,722 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:09:11,722 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:09:11,722 INFO train_multi TF=ALL: new best val=0.6149 — saved
2026-04-26 14:09:24,066 INFO train_multi TF=ALL epoch 26/50 train=0.6087 val=0.6121
2026-04-26 14:09:24,070 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:09:24,071 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:09:24,071 INFO train_multi TF=ALL: new best val=0.6121 — saved
2026-04-26 14:09:36,824 INFO train_multi TF=ALL epoch 27/50 train=0.6075 val=0.6123
2026-04-26 14:09:49,508 INFO train_multi TF=ALL epoch 28/50 train=0.6055 val=0.6125
2026-04-26 14:10:01,902 INFO train_multi TF=ALL epoch 29/50 train=0.6039 val=0.6101
2026-04-26 14:10:01,907 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:10:01,907 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:10:01,907 INFO train_multi TF=ALL: new best val=0.6101 — saved
2026-04-26 14:10:14,457 INFO train_multi TF=ALL epoch 30/50 train=0.6024 val=0.6113
2026-04-26 14:10:27,166 INFO train_multi TF=ALL epoch 31/50 train=0.6007 val=0.6100
2026-04-26 14:10:27,171 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:10:27,171 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:10:27,171 INFO train_multi TF=ALL: new best val=0.6100 — saved
2026-04-26 14:10:39,840 INFO train_multi TF=ALL epoch 32/50 train=0.5993 val=0.6100
2026-04-26 14:10:39,845 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:10:39,845 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:10:39,845 INFO train_multi TF=ALL: new best val=0.6100 — saved
2026-04-26 14:10:52,431 INFO train_multi TF=ALL epoch 33/50 train=0.5981 val=0.6136
2026-04-26 14:11:05,034 INFO train_multi TF=ALL epoch 34/50 train=0.5966 val=0.6113
2026-04-26 14:11:17,724 INFO train_multi TF=ALL epoch 35/50 train=0.5952 val=0.6134
2026-04-26 14:11:30,242 INFO train_multi TF=ALL epoch 36/50 train=0.5940 val=0.6088
2026-04-26 14:11:30,246 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:11:30,246 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:11:30,247 INFO train_multi TF=ALL: new best val=0.6088 — saved
2026-04-26 14:11:42,876 INFO train_multi TF=ALL epoch 37/50 train=0.5927 val=0.6092
2026-04-26 14:11:55,231 INFO train_multi TF=ALL epoch 38/50 train=0.5913 val=0.6101
2026-04-26 14:12:07,349 INFO train_multi TF=ALL epoch 39/50 train=0.5898 val=0.6134
2026-04-26 14:12:19,480 INFO train_multi TF=ALL epoch 40/50 train=0.5890 val=0.6122
2026-04-26 14:12:31,767 INFO train_multi TF=ALL epoch 41/50 train=0.5875 val=0.6097
2026-04-26 14:12:31,767 INFO train_multi TF=ALL early stop at epoch 41
2026-04-26 14:12:31,927 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-26 14:12:31,927 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-26 14:12:31,927 INFO Retrain complete. Total wall-clock: 589.7s
2026-04-26 14:12:33,925 INFO Model gru: SUCCESS
2026-04-26 14:12:33,925 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:12:33,925 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 14:12:33,925 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 14:12:33,925 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-04-26 14:12:33,926 INFO   [DEFERRED] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip (expected after Round 1)
2026-04-26 14:12:33,926 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-04-26 14:12:33,926 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer', 'rl_ppo']
2026-04-26 14:12:33,927 INFO Saved 5 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  START Round 1 - Backtest (val)
2026-04-26 14:12:34,700 INFO === STEP 6: BACKTEST (round1) ===
2026-04-26 14:12:34,701 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-04-26 14:12:34,701 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-26 14:12:34,701 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
2026-04-26 14:12:37,239 WARNING QualityScorer unavailable (weights missing or load failed)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260426_141236.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)                3   0.0%   0.00   -3.0%  0.0%  0.0%   3.0% -1932.54
  gate_diagnostics: bars=468696 no_signal=3 quality_block=0 session_skip=317895 density=0 pm_reject=0 daily_skip=150795 cooldown=0

Calibration Summary:
  all          [N/A] Insufficient data: 3 samples
  ml_trader    [N/A] Insufficient data: 3 samples
2026-04-26 14:14:50,359 INFO Round 1 backtest — 3 trades | avg WR=0.0% | avg PF=0.00 | avg Sharpe=-1932.54
2026-04-26 14:14:50,359 INFO   ml_trader: 3 trades | WR=0.0% | PF=0.00 | Return=-3.0% | DD=3.0% | Sharpe=-1932.54
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 3
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (3 rows)
2026-04-26 14:14:50,579 INFO Round 1: wrote 3 journal entries (total in file: 3)
  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 3 entries

=== Round 1 → Retrain Quality + RL ===
  START Round 1 - Quality+RL retrain
2026-04-26 14:14:50,903 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-26 14:14:50,904 INFO Journal entries: 3
2026-04-26 14:14:50,904 WARNING Journal has only 3 entries (need 50) — backtest produced too few trades. Skipping Quality+RL training. Check step6 logs.
  DONE  Round 1 - Quality+RL retrain

=== Round 2: BLIND backtest on test window (unseen 2yr) ===
  START Round 2 - Blind backtest (test)
2026-04-26 14:14:51,434 INFO === STEP 6: BACKTEST (round2) ===
2026-04-26 14:14:51,435 INFO BT_WINDOW=round2 — BLIND backtest: 2023-08-07 → 2025-08-05 (test set)
2026-04-26 14:14:51,435 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-26 14:14:51,435 INFO Round 2 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-26 14:14:53,944 WARNING QualityScorer unavailable (weights missing or load failed)
2026-04-26 14:17:08,259 INFO Round 2 backtest — 8 trades | avg WR=50.0% | avg PF=0.96 | avg Sharpe=-0.30
2026-04-26 14:17:08,260 INFO   ml_trader: 8 trades | WR=50.0% | PF=0.96 | Return=-0.1% | DD=2.0% | Sharpe=-0.30
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 8
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (8 rows)
2026-04-26 14:17:08,480 INFO Round 2: wrote 8 journal entries (total in file: 11)
  DONE  Round 2 - Blind backtest (test)
  Saved Round 2 result → round2_summary.json
  Journal after Round 2: 11 entries

=== Round 2 → Retrain Quality + RL (Round 1+2 journal) ===
  START Round 2 - Quality+RL retrain
2026-04-26 14:17:08,824 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-26 14:17:08,825 INFO Journal entries: 11
2026-04-26 14:17:08,825 WARNING Journal has only 11 entries (need 50) — backtest produced too few trades. Skipping Quality+RL training. Check step6 logs.
  DONE  Round 2 - Quality+RL retrain

=== Round 3: Incremental retrain ALL models on full data ===
  START Retrain gru [full-data retrain]
2026-04-26 14:17:09,033 INFO retrain environment: KAGGLE
2026-04-26 14:17:10,765 INFO Device: CUDA (2 GPU(s))
2026-04-26 14:17:10,776 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 14:17:10,776 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 14:17:10,776 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 14:17:10,777 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 14:17:10,778 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-26 14:17:10,929 INFO NumExpr defaulting to 4 threads.
2026-04-26 14:17:11,137 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 14:17:11,138 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 14:17:11,138 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 14:17:11,391 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-26 14:17:11,391 INFO GRU phase macro_correlations: 0.0s
2026-04-26 14:17:11,391 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-26 14:17:11,393 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260426_141711
2026-04-26 14:17:11,397 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-04-26 14:17:11,549 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:11,568 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:11,583 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:11,591 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:11,592 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 14:17:11,592 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 14:17:11,592 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 14:17:11,593 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 14:17:11,594 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:11,675 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-26 14:17:11,677 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:11,916 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-26 14:17:11,946 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:12,229 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:12,381 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:12,503 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:12,731 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:12,750 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:12,765 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:12,772 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:12,773 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:12,857 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-26 14:17:12,859 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:13,104 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-26 14:17:13,120 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:13,409 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:13,550 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:13,656 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:13,868 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:13,889 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:13,905 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:13,914 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:13,915 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:14,000 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-26 14:17:14,002 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:14,258 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-26 14:17:14,281 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:14,600 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:14,749 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:14,863 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:15,084 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:15,104 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:15,120 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:15,128 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:15,129 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:15,211 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-26 14:17:15,213 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:15,454 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-26 14:17:15,480 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:15,793 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:15,938 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:16,048 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:16,256 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:16,278 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:16,296 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:16,304 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:16,305 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:16,391 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-26 14:17:16,393 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:16,633 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-26 14:17:16,650 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:16,949 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:17,093 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:17,214 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:17,414 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:17,432 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:17,447 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:17,454 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:17,455 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:17,536 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-26 14:17:17,538 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:17,779 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-26 14:17:17,795 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:18,096 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:18,248 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:18,366 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:18,554 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:17:18,573 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:17:18,587 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:17:18,594 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:17:18,595 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:18,675 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-26 14:17:18,677 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:18,925 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-26 14:17:18,938 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:19,235 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:19,386 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:19,506 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:19,706 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:19,726 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:19,741 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:19,749 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:19,750 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:19,831 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-26 14:17:19,833 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:20,066 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-26 14:17:20,083 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:20,365 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:20,512 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:20,626 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:20,831 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:20,850 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:20,864 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:20,872 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:20,872 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:20,952 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-26 14:17:20,954 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:21,190 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-26 14:17:21,206 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:21,500 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:21,640 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:21,752 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:21,965 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:21,986 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:22,002 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:22,010 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:17:22,011 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:22,091 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-26 14:17:22,093 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:22,323 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-26 14:17:22,339 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:22,634 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:22,781 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:22,894 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:17:23,200 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:17:23,227 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:17:23,245 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:17:23,256 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:17:23,257 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:17:23,421 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-26 14:17:23,425 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:17:23,946 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-26 14:17:23,996 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:17:24,645 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:17:24,867 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:17:25,017 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:17:25,148 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-26 14:17:25,148 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-26 14:17:25,148 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-26 14:18:16,172 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-26 14:18:16,173 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-26 14:18:17,534 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-26 14:18:21,755 INFO train_multi TF=ALL: warm-start detected — using CosineAnnealingLR (lr=3e-05, patience=12)
2026-04-26 14:18:36,111 INFO train_multi TF=ALL epoch 1/50 train=0.5913 val=0.6090
2026-04-26 14:18:36,116 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:18:36,116 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:18:36,117 INFO train_multi TF=ALL: new best val=0.6090 — saved
2026-04-26 14:18:48,934 INFO train_multi TF=ALL epoch 2/50 train=0.5911 val=0.6095
2026-04-26 14:19:01,770 INFO train_multi TF=ALL epoch 3/50 train=0.5907 val=0.6098
2026-04-26 14:19:14,632 INFO train_multi TF=ALL epoch 4/50 train=0.5907 val=0.6099
2026-04-26 14:19:27,516 INFO train_multi TF=ALL epoch 5/50 train=0.5906 val=0.6098
2026-04-26 14:19:40,264 INFO train_multi TF=ALL epoch 6/50 train=0.5903 val=0.6098
2026-04-26 14:19:53,181 INFO train_multi TF=ALL epoch 7/50 train=0.5899 val=0.6089
2026-04-26 14:19:53,186 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 14:19:53,186 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 14:19:53,186 INFO train_multi TF=ALL: new best val=0.6089 — saved
2026-04-26 14:20:06,064 INFO train_multi TF=ALL epoch 8/50 train=0.5900 val=0.6095
2026-04-26 14:20:18,827 INFO train_multi TF=ALL epoch 9/50 train=0.5898 val=0.6096
2026-04-26 14:20:31,659 INFO train_multi TF=ALL epoch 10/50 train=0.5895 val=0.6095
2026-04-26 14:20:44,433 INFO train_multi TF=ALL epoch 11/50 train=0.5895 val=0.6097
2026-04-26 14:20:57,113 INFO train_multi TF=ALL epoch 12/50 train=0.5891 val=0.6100
2026-04-26 14:21:10,003 INFO train_multi TF=ALL epoch 13/50 train=0.5890 val=0.6092
2026-04-26 14:21:23,008 INFO train_multi TF=ALL epoch 14/50 train=0.5894 val=0.6096
2026-04-26 14:21:35,934 INFO train_multi TF=ALL epoch 15/50 train=0.5889 val=0.6101
2026-04-26 14:21:48,607 INFO train_multi TF=ALL epoch 16/50 train=0.5887 val=0.6100
2026-04-26 14:22:01,381 INFO train_multi TF=ALL epoch 17/50 train=0.5887 val=0.6101
2026-04-26 14:22:14,106 INFO train_multi TF=ALL epoch 18/50 train=0.5885 val=0.6101
2026-04-26 14:22:26,679 INFO train_multi TF=ALL epoch 19/50 train=0.5883 val=0.6095
2026-04-26 14:22:26,679 INFO train_multi TF=ALL early stop at epoch 19
2026-04-26 14:22:26,840 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-26 14:22:26,840 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-26 14:22:26,840 INFO Retrain complete. Total wall-clock: 316.1s
  DONE  Retrain gru [full-data retrain]
  START Retrain regime [full-data retrain]
2026-04-26 14:22:29,449 INFO retrain environment: KAGGLE
2026-04-26 14:22:31,158 INFO Device: CUDA (2 GPU(s))
2026-04-26 14:22:31,167 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 14:22:31,167 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 14:22:31,167 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 14:22:31,168 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 14:22:31,169 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-26 14:22:31,333 INFO NumExpr defaulting to 4 threads.
2026-04-26 14:22:31,557 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 14:22:31,557 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 14:22:31,557 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 14:22:31,558 INFO Regime phase macro_correlations: 0.0s
2026-04-26 14:22:31,558 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-26 14:22:31,599 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 14:22:31,600 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:22:31,630 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:22:31,646 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:22:31,670 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:22:31,686 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:22:31,713 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:22:31,729 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:22:31,754 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:22:31,772 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:22:31,797 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:22:31,812 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:22:31,834 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:22:31,848 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:22:31,869 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:22:31,885 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:22:31,910 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:22:31,926 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:22:31,950 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:22:31,966 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:22:31,991 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:22:32,009 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:22:32,053 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:22:32,876 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 14:22:57,895 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 14:22:57,900 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 25.8s
2026-04-26 14:22:57,900 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 14:23:08,829 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 14:23:08,832 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 10.9s
2026-04-26 14:23:08,833 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 14:23:17,226 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 14:23:17,229 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 8.4s
2026-04-26 14:23:17,229 INFO Regime phase GMM HTF total: 45.2s
2026-04-26 14:23:17,229 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 14:24:31,720 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 14:24:31,726 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 74.5s
2026-04-26 14:24:31,726 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 14:25:05,138 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 14:25:05,142 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 33.4s
2026-04-26 14:25:05,143 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 14:25:28,266 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 14:25:28,268 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 23.1s
2026-04-26 14:25:28,268 INFO Regime phase GMM LTF total: 131.0s
2026-04-26 14:25:28,383 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-26 14:25:28,384 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:28,385 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:28,386 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:28,387 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:28,388 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:28,389 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:28,390 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:28,391 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:28,392 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:28,393 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:28,395 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:25:28,524 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:28,565 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:28,566 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:28,566 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:28,575 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:28,576 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:28,993 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-26 14:25:28,994 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-26 14:25:29,188 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:29,224 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:29,225 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:29,226 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:29,234 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:29,235 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:29,626 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-26 14:25:29,628 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-26 14:25:29,818 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:29,854 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:29,855 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:29,856 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:29,864 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:29,865 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:30,256 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-26 14:25:30,258 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-26 14:25:30,449 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:30,489 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:30,490 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:30,491 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:30,499 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:30,500 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:30,896 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-26 14:25:30,897 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-26 14:25:31,092 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:31,130 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:31,131 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:31,131 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:31,140 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:31,141 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:31,540 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-26 14:25:31,542 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-26 14:25:31,733 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:31,769 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:31,770 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:31,770 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:31,778 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:31,779 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:32,169 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-26 14:25:32,171 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-26 14:25:32,339 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:25:32,368 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:25:32,369 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:25:32,369 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:25:32,377 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:25:32,378 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:32,762 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-26 14:25:32,763 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-26 14:25:32,957 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:32,993 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:32,994 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:32,994 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:33,002 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:33,003 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:33,406 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-26 14:25:33,408 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-26 14:25:33,595 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:33,631 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:33,631 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:33,632 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:33,641 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:33,642 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:34,036 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-26 14:25:34,037 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-26 14:25:34,228 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:34,267 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:34,268 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:34,268 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:34,277 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:34,278 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:34,700 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-26 14:25:34,702 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-26 14:25:35,008 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:25:35,075 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:25:35,076 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:25:35,077 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:25:35,088 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:25:35,089 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:25:36,005 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-26 14:25:36,008 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-26 14:25:36,234 INFO Regime phase HTF dataset build: 7.9s (103290 samples)
2026-04-26 14:25:36,235 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260426_142536
2026-04-26 14:25:36,438 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=34, n_classes=3)
2026-04-26 14:25:36,457 INFO RegimeClassifier[mode=htf_bias]: 103290 samples, classes={'BIAS_UP': 18622, 'BIAS_DOWN': 18286, 'BIAS_NEUTRAL': 66382}, device=cuda
2026-04-26 14:25:36,457 INFO RegimeClassifier: sample weights — mean=0.360  ambiguous(<0.4)=69.5%
2026-04-26 14:25:36,457 INFO RegimeClassifier[mode=htf_bias]: warm start from existing weights
2026-04-26 14:25:36,458 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 14:25:38,931 INFO Regime epoch  1/50 — tr=0.2441 va=1.8013 acc=0.644 per_class={'BIAS_UP': 0.979, 'BIAS_DOWN': 0.988, 'BIAS_NEUTRAL': 0.437}
2026-04-26 14:25:39,127 INFO Regime epoch  2/50 — tr=0.2441 va=1.7999 acc=0.646
2026-04-26 14:25:39,318 INFO Regime epoch  3/50 — tr=0.2439 va=1.8023 acc=0.645
2026-04-26 14:25:39,512 INFO Regime epoch  4/50 — tr=0.2441 va=1.8034 acc=0.644
2026-04-26 14:25:39,722 INFO Regime epoch  5/50 — tr=0.2440 va=1.8032 acc=0.642 per_class={'BIAS_UP': 0.979, 'BIAS_DOWN': 0.99, 'BIAS_NEUTRAL': 0.433}
2026-04-26 14:25:39,910 INFO Regime epoch  6/50 — tr=0.2440 va=1.7995 acc=0.649
2026-04-26 14:25:40,102 INFO Regime epoch  7/50 — tr=0.2437 va=1.7994 acc=0.649
2026-04-26 14:25:40,298 INFO Regime epoch  8/50 — tr=0.2438 va=1.8002 acc=0.648
2026-04-26 14:25:40,483 INFO Regime epoch  9/50 — tr=0.2435 va=1.7985 acc=0.650
2026-04-26 14:25:40,685 INFO Regime epoch 10/50 — tr=0.2436 va=1.7959 acc=0.654 per_class={'BIAS_UP': 0.979, 'BIAS_DOWN': 0.988, 'BIAS_NEUTRAL': 0.453}
2026-04-26 14:25:40,880 INFO Regime epoch 11/50 — tr=0.2434 va=1.7972 acc=0.653
2026-04-26 14:25:41,064 INFO Regime epoch 12/50 — tr=0.2433 va=1.7955 acc=0.654
2026-04-26 14:25:41,260 INFO Regime epoch 13/50 — tr=0.2432 va=1.7960 acc=0.655
2026-04-26 14:25:41,452 INFO Regime epoch 14/50 — tr=0.2431 va=1.7919 acc=0.659
2026-04-26 14:25:41,662 INFO Regime epoch 15/50 — tr=0.2428 va=1.7919 acc=0.659 per_class={'BIAS_UP': 0.978, 'BIAS_DOWN': 0.989, 'BIAS_NEUTRAL': 0.462}
2026-04-26 14:25:41,849 INFO Regime epoch 16/50 — tr=0.2429 va=1.7905 acc=0.660
2026-04-26 14:25:42,047 INFO Regime epoch 17/50 — tr=0.2429 va=1.7895 acc=0.660
2026-04-26 14:25:42,231 INFO Regime epoch 18/50 — tr=0.2428 va=1.7900 acc=0.660
2026-04-26 14:25:42,419 INFO Regime epoch 19/50 — tr=0.2426 va=1.7890 acc=0.661
2026-04-26 14:25:42,612 INFO Regime epoch 20/50 — tr=0.2427 va=1.7883 acc=0.661 per_class={'BIAS_UP': 0.979, 'BIAS_DOWN': 0.99, 'BIAS_NEUTRAL': 0.464}
2026-04-26 14:25:42,798 INFO Regime epoch 21/50 — tr=0.2425 va=1.7865 acc=0.663
2026-04-26 14:25:42,988 INFO Regime epoch 22/50 — tr=0.2425 va=1.7845 acc=0.665
2026-04-26 14:25:43,169 INFO Regime epoch 23/50 — tr=0.2425 va=1.7858 acc=0.664
2026-04-26 14:25:43,354 INFO Regime epoch 24/50 — tr=0.2424 va=1.7824 acc=0.668
2026-04-26 14:25:43,549 INFO Regime epoch 25/50 — tr=0.2424 va=1.7837 acc=0.667 per_class={'BIAS_UP': 0.98, 'BIAS_DOWN': 0.99, 'BIAS_NEUTRAL': 0.474}
2026-04-26 14:25:43,728 INFO Regime epoch 26/50 — tr=0.2422 va=1.7829 acc=0.668
2026-04-26 14:25:43,904 INFO Regime epoch 27/50 — tr=0.2423 va=1.7823 acc=0.668
2026-04-26 14:25:44,088 INFO Regime epoch 28/50 — tr=0.2421 va=1.7820 acc=0.667
2026-04-26 14:25:44,299 INFO Regime epoch 29/50 — tr=0.2422 va=1.7796 acc=0.673
2026-04-26 14:25:44,511 INFO Regime epoch 30/50 — tr=0.2420 va=1.7801 acc=0.672 per_class={'BIAS_UP': 0.978, 'BIAS_DOWN': 0.99, 'BIAS_NEUTRAL': 0.482}
2026-04-26 14:25:44,730 INFO Regime epoch 31/50 — tr=0.2421 va=1.7802 acc=0.671
2026-04-26 14:25:44,908 INFO Regime epoch 32/50 — tr=0.2419 va=1.7784 acc=0.673
2026-04-26 14:25:45,099 INFO Regime epoch 33/50 — tr=0.2419 va=1.7775 acc=0.677
2026-04-26 14:25:45,282 INFO Regime epoch 34/50 — tr=0.2420 va=1.7757 acc=0.679
2026-04-26 14:25:45,484 INFO Regime epoch 35/50 — tr=0.2419 va=1.7770 acc=0.676 per_class={'BIAS_UP': 0.976, 'BIAS_DOWN': 0.99, 'BIAS_NEUTRAL': 0.489}
2026-04-26 14:25:45,675 INFO Regime epoch 36/50 — tr=0.2418 va=1.7810 acc=0.671
2026-04-26 14:25:45,880 INFO Regime epoch 37/50 — tr=0.2420 va=1.7808 acc=0.671
2026-04-26 14:25:46,079 INFO Regime epoch 38/50 — tr=0.2420 va=1.7781 acc=0.675
2026-04-26 14:25:46,266 INFO Regime epoch 39/50 — tr=0.2419 va=1.7773 acc=0.673
2026-04-26 14:25:46,480 INFO Regime epoch 40/50 — tr=0.2418 va=1.7781 acc=0.672 per_class={'BIAS_UP': 0.98, 'BIAS_DOWN': 0.99, 'BIAS_NEUTRAL': 0.482}
2026-04-26 14:25:46,665 INFO Regime epoch 41/50 — tr=0.2419 va=1.7740 acc=0.679
2026-04-26 14:25:46,854 INFO Regime epoch 42/50 — tr=0.2419 va=1.7759 acc=0.677
2026-04-26 14:25:47,045 INFO Regime epoch 43/50 — tr=0.2419 va=1.7772 acc=0.675
2026-04-26 14:25:47,242 INFO Regime epoch 44/50 — tr=0.2418 va=1.7755 acc=0.677
2026-04-26 14:25:47,453 INFO Regime epoch 45/50 — tr=0.2418 va=1.7769 acc=0.676 per_class={'BIAS_UP': 0.979, 'BIAS_DOWN': 0.99, 'BIAS_NEUTRAL': 0.487}
2026-04-26 14:25:47,643 INFO Regime epoch 46/50 — tr=0.2421 va=1.7770 acc=0.676
2026-04-26 14:25:47,825 INFO Regime epoch 47/50 — tr=0.2419 va=1.7764 acc=0.678
2026-04-26 14:25:48,017 INFO Regime epoch 48/50 — tr=0.2418 va=1.7769 acc=0.676
2026-04-26 14:25:48,199 INFO Regime epoch 49/50 — tr=0.2419 va=1.7794 acc=0.672
2026-04-26 14:25:48,402 INFO Regime epoch 50/50 — tr=0.2417 va=1.7767 acc=0.676 per_class={'BIAS_UP': 0.979, 'BIAS_DOWN': 0.99, 'BIAS_NEUTRAL': 0.488}
2026-04-26 14:25:48,419 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 14:25:48,419 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 14:25:48,420 INFO Regime phase HTF train: 12.0s
2026-04-26 14:25:48,563 INFO Regime HTF complete: acc=0.679, n=103290
2026-04-26 14:25:48,565 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:25:48,729 INFO Rule labels LTF_BEHAVIOUR [4H]: {'TRENDING': 8970, 'RANGING': 4592, 'CONSOLIDATING': 2247, 'VOLATILE': 4008}  ambiguous=4539 (total=19817)  short_runs_zeroed=570
2026-04-26 14:25:48,732 INFO Regime[4H] persistence (avg bars/run) on XAUUSD 4H:
{0: 12.121621621621621, 1: 6.4858757062146895, 2: 5.575682382133995, 3: 10.329896907216495}
2026-04-26 14:25:48,734 INFO Regime[4H] return separation on XAUUSD 4H:
{0: 5.9502411776840314e-05, 1: 5.664222849552333e-05, 2: -6.138442068010657e-05, 3: 4.239843820627722e-05}
2026-04-26 14:25:48,735 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-26 14:25:48,736 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:48,738 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:48,740 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:48,742 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:48,744 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:48,746 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:48,747 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:48,749 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:48,751 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:48,752 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:48,756 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:25:48,766 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:48,769 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:48,769 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:48,770 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:48,770 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:48,772 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:49,428 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-26 14:25:49,431 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-26 14:25:49,588 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:49,590 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:49,591 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:49,591 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:49,592 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:49,594 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:50,215 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-26 14:25:50,218 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-26 14:25:50,379 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:50,382 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:50,383 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:50,383 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:50,383 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:50,386 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:50,993 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-26 14:25:50,996 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-26 14:25:51,151 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:51,153 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:51,154 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:51,155 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:51,155 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:51,157 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:51,774 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-26 14:25:51,777 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-26 14:25:51,932 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:51,935 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:51,936 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:51,936 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:51,937 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:51,939 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:52,569 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-26 14:25:52,572 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-26 14:25:52,730 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:52,733 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:52,734 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:52,735 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:52,735 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:52,737 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:53,361 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-26 14:25:53,364 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-26 14:25:53,524 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:25:53,526 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:25:53,526 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:25:53,527 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:25:53,527 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 14:25:53,529 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:54,137 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-26 14:25:54,140 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-26 14:25:54,295 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:54,298 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:54,298 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:54,299 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:54,299 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:54,301 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:54,931 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-26 14:25:54,934 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-26 14:25:55,096 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:55,099 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:55,100 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:55,100 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:55,100 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:55,102 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:55,717 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-26 14:25:55,721 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-26 14:25:55,881 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:55,883 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:55,884 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:55,885 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:55,885 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 14:25:55,887 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 14:25:56,517 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-26 14:25:56,520 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-26 14:25:56,685 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:25:56,692 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:25:56,693 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:25:56,694 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:25:56,694 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 14:25:56,698 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:25:58,048 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-26 14:25:58,054 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-26 14:25:58,403 INFO Regime phase LTF dataset build: 9.7s (401471 samples)
2026-04-26 14:25:58,404 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260426_142558
2026-04-26 14:25:58,409 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=18, n_classes=4)
2026-04-26 14:25:58,468 INFO RegimeClassifier[mode=ltf_behaviour]: 401471 samples, classes={'TRENDING': 160094, 'RANGING': 105286, 'CONSOLIDATING': 53524, 'VOLATILE': 82567}, device=cuda
2026-04-26 14:25:58,469 INFO RegimeClassifier: sample weights — mean=0.505  ambiguous(<0.4)=33.0%
2026-04-26 14:25:58,469 INFO RegimeClassifier[mode=ltf_behaviour]: warm start from existing weights
2026-04-26 14:25:58,469 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 14:25:59,283 INFO Regime epoch  1/50 — tr=0.4744 va=1.5886 acc=0.717 per_class={'TRENDING': 0.823, 'RANGING': 0.301, 'CONSOLIDATING': 0.938, 'VOLATILE': 0.904}
2026-04-26 14:25:59,996 INFO Regime epoch  2/50 — tr=0.4744 va=1.5923 acc=0.717
2026-04-26 14:26:00,695 INFO Regime epoch  3/50 — tr=0.4744 va=1.5934 acc=0.717
2026-04-26 14:26:01,387 INFO Regime epoch  4/50 — tr=0.4743 va=1.5925 acc=0.717
2026-04-26 14:26:02,119 INFO Regime epoch  5/50 — tr=0.4743 va=1.5880 acc=0.719 per_class={'TRENDING': 0.825, 'RANGING': 0.303, 'CONSOLIDATING': 0.934, 'VOLATILE': 0.905}
2026-04-26 14:26:02,832 INFO Regime epoch  6/50 — tr=0.4741 va=1.5896 acc=0.718
2026-04-26 14:26:03,532 INFO Regime epoch  7/50 — tr=0.4741 va=1.5896 acc=0.718
2026-04-26 14:26:04,225 INFO Regime epoch  8/50 — tr=0.4740 va=1.5902 acc=0.718
2026-04-26 14:26:04,937 INFO Regime epoch  9/50 — tr=0.4740 va=1.5905 acc=0.718
2026-04-26 14:26:05,659 INFO Regime epoch 10/50 — tr=0.4740 va=1.5894 acc=0.717 per_class={'TRENDING': 0.826, 'RANGING': 0.295, 'CONSOLIDATING': 0.937, 'VOLATILE': 0.904}
2026-04-26 14:26:06,367 INFO Regime epoch 11/50 — tr=0.4738 va=1.5872 acc=0.719
2026-04-26 14:26:07,066 INFO Regime epoch 12/50 — tr=0.4735 va=1.5883 acc=0.719
2026-04-26 14:26:07,736 INFO Regime epoch 13/50 — tr=0.4734 va=1.5880 acc=0.718
2026-04-26 14:26:08,411 INFO Regime epoch 14/50 — tr=0.4735 va=1.5876 acc=0.718
2026-04-26 14:26:09,148 INFO Regime epoch 15/50 — tr=0.4733 va=1.5867 acc=0.719 per_class={'TRENDING': 0.828, 'RANGING': 0.301, 'CONSOLIDATING': 0.936, 'VOLATILE': 0.903}
2026-04-26 14:26:09,844 INFO Regime epoch 16/50 — tr=0.4732 va=1.5894 acc=0.718
2026-04-26 14:26:10,547 INFO Regime epoch 17/50 — tr=0.4732 va=1.5875 acc=0.720
2026-04-26 14:26:11,251 INFO Regime epoch 18/50 — tr=0.4732 va=1.5852 acc=0.720
2026-04-26 14:26:11,937 INFO Regime epoch 19/50 — tr=0.4731 va=1.5875 acc=0.719
2026-04-26 14:26:12,666 INFO Regime epoch 20/50 — tr=0.4731 va=1.5848 acc=0.720 per_class={'TRENDING': 0.836, 'RANGING': 0.297, 'CONSOLIDATING': 0.937, 'VOLATILE': 0.895}
2026-04-26 14:26:13,345 INFO Regime epoch 21/50 — tr=0.4730 va=1.5869 acc=0.719
2026-04-26 14:26:14,022 INFO Regime epoch 22/50 — tr=0.4729 va=1.5870 acc=0.719
2026-04-26 14:26:14,780 INFO Regime epoch 23/50 — tr=0.4727 va=1.5851 acc=0.720
2026-04-26 14:26:15,526 INFO Regime epoch 24/50 — tr=0.4727 va=1.5863 acc=0.720
2026-04-26 14:26:16,282 INFO Regime epoch 25/50 — tr=0.4727 va=1.5866 acc=0.719 per_class={'TRENDING': 0.83, 'RANGING': 0.298, 'CONSOLIDATING': 0.932, 'VOLATILE': 0.903}
2026-04-26 14:26:16,979 INFO Regime epoch 26/50 — tr=0.4726 va=1.5836 acc=0.720
2026-04-26 14:26:17,679 INFO Regime epoch 27/50 — tr=0.4725 va=1.5857 acc=0.720
2026-04-26 14:26:18,408 INFO Regime epoch 28/50 — tr=0.4727 va=1.5858 acc=0.720
2026-04-26 14:26:19,098 INFO Regime epoch 29/50 — tr=0.4725 va=1.5822 acc=0.720
2026-04-26 14:26:19,870 INFO Regime epoch 30/50 — tr=0.4725 va=1.5839 acc=0.719 per_class={'TRENDING': 0.833, 'RANGING': 0.298, 'CONSOLIDATING': 0.937, 'VOLATILE': 0.898}
2026-04-26 14:26:20,607 INFO Regime epoch 31/50 — tr=0.4724 va=1.5853 acc=0.719
2026-04-26 14:26:21,327 INFO Regime epoch 32/50 — tr=0.4727 va=1.5854 acc=0.720
2026-04-26 14:26:22,024 INFO Regime epoch 33/50 — tr=0.4724 va=1.5837 acc=0.720
2026-04-26 14:26:22,721 INFO Regime epoch 34/50 — tr=0.4724 va=1.5821 acc=0.721
2026-04-26 14:26:23,463 INFO Regime epoch 35/50 — tr=0.4723 va=1.5843 acc=0.719 per_class={'TRENDING': 0.831, 'RANGING': 0.301, 'CONSOLIDATING': 0.933, 'VOLATILE': 0.901}
2026-04-26 14:26:24,149 INFO Regime epoch 36/50 — tr=0.4724 va=1.5825 acc=0.719
2026-04-26 14:26:24,875 INFO Regime epoch 37/50 — tr=0.4722 va=1.5850 acc=0.720
2026-04-26 14:26:25,580 INFO Regime epoch 38/50 — tr=0.4723 va=1.5822 acc=0.721
2026-04-26 14:26:26,328 INFO Regime epoch 39/50 — tr=0.4724 va=1.5837 acc=0.720
2026-04-26 14:26:27,108 INFO Regime epoch 40/50 — tr=0.4722 va=1.5869 acc=0.719 per_class={'TRENDING': 0.829, 'RANGING': 0.299, 'CONSOLIDATING': 0.933, 'VOLATILE': 0.905}
2026-04-26 14:26:27,814 INFO Regime epoch 41/50 — tr=0.4725 va=1.5852 acc=0.719
2026-04-26 14:26:28,505 INFO Regime epoch 42/50 — tr=0.4721 va=1.5839 acc=0.721
2026-04-26 14:26:29,201 INFO Regime epoch 43/50 — tr=0.4723 va=1.5860 acc=0.719
2026-04-26 14:26:29,876 INFO Regime epoch 44/50 — tr=0.4723 va=1.5865 acc=0.719
2026-04-26 14:26:29,876 INFO Regime early stop at epoch 44 (no_improve=10)
2026-04-26 14:26:29,927 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 14:26:29,927 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 14:26:29,931 INFO Regime phase LTF train: 31.5s
2026-04-26 14:26:30,080 INFO Regime LTF complete: acc=0.721, n=401471
2026-04-26 14:26:30,084 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 14:26:30,593 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-26 14:26:30,598 INFO Regime[1H] persistence (avg bars/run) on XAUUSD 1H:
{0: 9.355291913830783, 1: 6.096303199751476, 2: 5.598885793871866, 3: 6.771351107094442}
2026-04-26 14:26:30,601 INFO Regime[1H] return separation on XAUUSD 1H:
{0: 2.149062259083323e-05, 1: -3.254493608872846e-06, 2: -3.039756158146958e-06, 3: 1.8102764458593193e-05}
2026-04-26 14:26:30,602 INFO Regime retrain total: 239.4s (504761 samples)
2026-04-26 14:26:30,604 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 14:26:30,604 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 14:26:30,604 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 14:26:30,604 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-26 14:26:30,605 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-26 14:26:30,605 INFO Retrain complete. Total wall-clock: 239.4s
  DONE  Retrain regime [full-data retrain]
  START Retrain quality [full-data retrain]
2026-04-26 14:26:32,347 INFO retrain environment: KAGGLE
2026-04-26 14:26:34,101 INFO Device: CUDA (2 GPU(s))
2026-04-26 14:26:34,112 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 14:26:34,112 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 14:26:34,112 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 14:26:34,113 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 14:26:34,114 INFO === QualityScorer retrain ===
2026-04-26 14:26:34,295 INFO NumExpr defaulting to 4 threads.
2026-04-26 14:26:34,536 INFO QualityScorer: CUDA available — using GPU
2026-04-26 14:26:34,540 INFO Quality phase label creation: 0.0s (11 trades)
2026-04-26 14:26:34,540 INFO Retrain complete. Total wall-clock: 0.4s
  WARN  Retrain quality failed (exit 1) — continuing
  START Retrain rl [full-data retrain]
2026-04-26 14:26:35,261 INFO retrain environment: KAGGLE
2026-04-26 14:26:36,990 INFO Device: CUDA (2 GPU(s))
2026-04-26 14:26:37,001 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 14:26:37,001 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 14:26:37,001 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 14:26:37,002 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 14:26:37,003 INFO === RLAgent (PPO) retrain ===
2026-04-26 14:26:37,008 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260426_142637
2026-04-26 14:26:37,009 INFO RL phase episode loading: 0.0s (11 episodes)
2026-04-26 14:26:37.875694: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777213597.899158  170422 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777213597.907232  170422 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777213597.927805  170422 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777213597.927857  170422 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777213597.927863  170422 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777213597.927867  170422 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-26 14:26:42,454 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-26 14:26:44,119 WARNING RLAgent.retrain: only 11 episodes — skipping
2026-04-26 14:26:44,119 INFO RL phase PPO train: 7.1s | total: 7.1s
2026-04-26 14:26:44,120 INFO Retrain complete. Total wall-clock: 7.1s
  WARN  Retrain rl failed (exit 1) — continuing

=== Round 3: Backtest on last 3yr (post-retrain evaluation) ===
  START Round 3 - Post-retrain backtest (last 3yr)
2026-04-26 14:26:46,199 INFO === STEP 6: BACKTEST (round3) ===
2026-04-26 14:26:46,200 INFO BT_WINDOW=round3 — post-retrain eval: 2022-08-05 → 2025-08-05 (last 3yr)
2026-04-26 14:26:46,201 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-04-26 14:26:46,201 INFO Round 3 — running backtest: 2022-08-05 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-26 14:26:48,764 WARNING QualityScorer unavailable (weights missing or load failed)
2026-04-26 14:29:59,268 INFO Round 3 backtest — 9 trades | avg WR=33.3% | avg PF=0.49 | avg Sharpe=-5.62
2026-04-26 14:29:59,268 INFO   ml_trader: 9 trades | WR=33.3% | PF=0.49 | Return=-3.0% | DD=3.0% | Sharpe=-5.62
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 9
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (9 rows)
  DONE  Round 3 - Post-retrain backtest (last 3yr)
  Saved Round 3 result → round3_summary.json

======================================================================
  BLIND BACKTEST PIPELINE COMPLETE
======================================================================
  Round 1 (val window)          trades=?  WR=0.0%  PF=0.000  Sharpe=0.000
  Round 2 (blind test)          trades=?  WR=0.0%  PF=0.000  Sharpe=0.000
  Round 3 (last 3yr)            trades=?  WR=0.0%  PF=0.000  Sharpe=0.000


WARNING: GITHUB_TOKEN not set — skipping GitHub push
2026-04-26 14:29:59,488 INFO Round 3: wrote 9 journal entries (total in file: 20)
