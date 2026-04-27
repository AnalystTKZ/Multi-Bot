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
2026-04-27 01:26:13,624 INFO Loading feature-engineered data...
2026-04-27 01:26:14,262 INFO Loaded 221743 rows, 202 features
2026-04-27 01:26:14,264 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-27 01:26:14,266 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-27 01:26:14,266 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-27 01:26:14,266 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-27 01:26:14,266 INFO No leakage confirmed: train < val < test timestamps

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
2026-04-27 01:26:16,761 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-27 01:26:16,761 INFO --- Training regime ---
2026-04-27 01:26:16,762 INFO Running retrain --model regime
2026-04-27 01:26:16,942 INFO retrain environment: KAGGLE
2026-04-27 01:26:18,640 INFO Device: CUDA (2 GPU(s))
2026-04-27 01:26:18,651 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 01:26:18,651 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 01:26:18,651 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 01:26:18,654 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 01:26:18,655 INFO Retrain data split: train
2026-04-27 01:26:18,656 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-27 01:26:18,824 INFO NumExpr defaulting to 4 threads.
2026-04-27 01:26:19,074 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-27 01:26:19,074 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 01:26:19,074 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 01:26:19,075 INFO Regime phase macro_correlations: 0.0s
2026-04-27 01:26:19,075 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-27 01:26:19,115 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-27 01:26:19,116 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:26:19,152 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:26:19,174 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:26:19,205 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:26:19,230 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:26:19,256 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:26:19,271 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:26:19,298 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:26:19,314 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:26:19,337 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:26:19,353 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:26:19,376 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:26:19,391 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:26:19,411 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:26:19,426 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:26:19,447 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:26:19,462 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:26:19,485 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:26:19,500 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:26:19,522 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:26:19,539 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:26:19,580 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:26:20,813 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 01:26:44,246 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 01:26:44,251 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 24.7s
2026-04-27 01:26:44,251 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 01:26:54,416 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 01:26:54,418 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 10.2s
2026-04-27 01:26:54,418 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 01:27:02,321 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 01:27:02,322 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 7.9s
2026-04-27 01:27:02,322 INFO Regime phase GMM HTF total: 42.7s
2026-04-27 01:27:02,323 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 01:28:14,917 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 01:28:14,930 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 72.6s
2026-04-27 01:28:14,930 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 01:28:47,486 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 01:28:47,488 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 32.6s
2026-04-27 01:28:47,488 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 01:29:10,671 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 01:29:10,672 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 23.2s
2026-04-27 01:29:10,673 INFO Regime phase GMM LTF total: 128.3s
2026-04-27 01:29:10,777 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-27 01:29:10,778 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:10,779 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:10,780 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:10,782 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:10,783 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:10,784 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:10,785 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:10,786 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:10,787 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:10,788 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:10,790 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:29:10,916 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:10,960 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:10,961 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:10,961 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:10,969 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:10,970 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:11,404 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-27 01:29:11,405 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-27 01:29:11,587 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:11,621 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:11,622 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:11,623 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:11,631 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:11,632 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:12,007 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-27 01:29:12,009 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-27 01:29:12,208 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:12,245 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:12,245 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:12,246 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:12,254 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:12,255 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:12,649 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-27 01:29:12,650 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-27 01:29:12,824 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:12,861 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:12,861 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:12,862 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:12,870 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:12,871 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:13,251 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-27 01:29:13,252 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-27 01:29:13,428 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:13,464 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:13,465 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:13,466 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:13,474 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:13,475 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:13,857 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-27 01:29:13,859 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-27 01:29:14,037 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:14,073 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:14,073 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:14,074 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:14,082 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:14,083 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:14,465 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-27 01:29:14,466 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-27 01:29:14,627 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:29:14,655 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:29:14,655 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:29:14,656 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:29:14,663 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:29:14,664 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:15,038 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-27 01:29:15,039 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-27 01:29:15,205 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:15,240 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:15,241 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:15,241 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:15,249 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:15,250 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:15,633 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-27 01:29:15,634 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-27 01:29:15,811 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:15,846 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:15,847 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:15,847 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:15,855 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:15,856 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:16,250 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-27 01:29:16,252 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-27 01:29:16,433 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:16,470 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:16,470 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:16,471 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:16,479 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:16,480 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:16,872 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-27 01:29:16,874 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-27 01:29:17,151 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:29:17,211 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:29:17,212 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:29:17,212 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:29:17,223 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:29:17,224 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:29:18,047 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 01:29:18,050 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-27 01:29:18,228 INFO Regime phase HTF dataset build: 7.5s (103290 samples)
2026-04-27 01:29:18,229 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260427_012918
2026-04-27 01:29:18,522 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=34, n_classes=3)
2026-04-27 01:29:18,523 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=38114 dropped=65176 classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728})
2026-04-27 01:29:18,533 INFO RegimeClassifier[mode=htf_bias]: 38114 samples, classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728}, device=cuda
2026-04-27 01:29:18,533 INFO RegimeClassifier: sample weights — mean=0.708  ambiguous(<0.4)=0.0%
2026-04-27 01:29:18,533 INFO RegimeClassifier[mode=htf_bias]: warm start from existing weights
2026-04-27 01:29:18,533 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-27 01:29:23,297 INFO Regime epoch  1/50 — tr=0.4714 va=1.0544 acc=0.984 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.912}
2026-04-27 01:29:23,365 INFO Regime epoch  2/50 — tr=0.4710 va=1.0563 acc=0.983
2026-04-27 01:29:23,433 INFO Regime epoch  3/50 — tr=0.4708 va=1.0545 acc=0.984
2026-04-27 01:29:23,505 INFO Regime epoch  4/50 — tr=0.4707 va=1.0536 acc=0.984
2026-04-27 01:29:23,579 INFO Regime epoch  5/50 — tr=0.4706 va=1.0537 acc=0.983 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.907}
2026-04-27 01:29:23,649 INFO Regime epoch  6/50 — tr=0.4704 va=1.0524 acc=0.983
2026-04-27 01:29:23,722 INFO Regime epoch  7/50 — tr=0.4701 va=1.0512 acc=0.983
2026-04-27 01:29:23,798 INFO Regime epoch  8/50 — tr=0.4700 va=1.0491 acc=0.984
2026-04-27 01:29:23,867 INFO Regime epoch  9/50 — tr=0.4699 va=1.0462 acc=0.984
2026-04-27 01:29:23,941 INFO Regime epoch 10/50 — tr=0.4695 va=1.0451 acc=0.984 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.912}
2026-04-27 01:29:24,013 INFO Regime epoch 11/50 — tr=0.4697 va=1.0395 acc=0.985
2026-04-27 01:29:24,087 INFO Regime epoch 12/50 — tr=0.4692 va=1.0395 acc=0.985
2026-04-27 01:29:24,158 INFO Regime epoch 13/50 — tr=0.4689 va=1.0397 acc=0.985
2026-04-27 01:29:24,228 INFO Regime epoch 14/50 — tr=0.4685 va=1.0354 acc=0.985
2026-04-27 01:29:24,304 INFO Regime epoch 15/50 — tr=0.4685 va=1.0333 acc=0.985 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.918}
2026-04-27 01:29:24,374 INFO Regime epoch 16/50 — tr=0.4683 va=1.0319 acc=0.986
2026-04-27 01:29:24,445 INFO Regime epoch 17/50 — tr=0.4678 va=1.0324 acc=0.985
2026-04-27 01:29:24,514 INFO Regime epoch 18/50 — tr=0.4678 va=1.0315 acc=0.985
2026-04-27 01:29:24,588 INFO Regime epoch 19/50 — tr=0.4675 va=1.0301 acc=0.985
2026-04-27 01:29:24,669 INFO Regime epoch 20/50 — tr=0.4673 va=1.0267 acc=0.986 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.92}
2026-04-27 01:29:24,746 INFO Regime epoch 21/50 — tr=0.4674 va=1.0265 acc=0.986
2026-04-27 01:29:24,820 INFO Regime epoch 22/50 — tr=0.4673 va=1.0252 acc=0.986
2026-04-27 01:29:24,889 INFO Regime epoch 23/50 — tr=0.4671 va=1.0240 acc=0.986
2026-04-27 01:29:24,961 INFO Regime epoch 24/50 — tr=0.4667 va=1.0231 acc=0.986
2026-04-27 01:29:25,042 INFO Regime epoch 25/50 — tr=0.4663 va=1.0215 acc=0.986 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.925}
2026-04-27 01:29:25,113 INFO Regime epoch 26/50 — tr=0.4661 va=1.0209 acc=0.986
2026-04-27 01:29:25,182 INFO Regime epoch 27/50 — tr=0.4664 va=1.0210 acc=0.986
2026-04-27 01:29:25,251 INFO Regime epoch 28/50 — tr=0.4660 va=1.0218 acc=0.987
2026-04-27 01:29:25,322 INFO Regime epoch 29/50 — tr=0.4658 va=1.0179 acc=0.987
2026-04-27 01:29:25,398 INFO Regime epoch 30/50 — tr=0.4659 va=1.0178 acc=0.987 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.926}
2026-04-27 01:29:25,473 INFO Regime epoch 31/50 — tr=0.4660 va=1.0180 acc=0.987
2026-04-27 01:29:25,546 INFO Regime epoch 32/50 — tr=0.4659 va=1.0156 acc=0.987
2026-04-27 01:29:25,620 INFO Regime epoch 33/50 — tr=0.4657 va=1.0161 acc=0.987
2026-04-27 01:29:25,694 INFO Regime epoch 34/50 — tr=0.4654 va=1.0157 acc=0.987
2026-04-27 01:29:25,769 INFO Regime epoch 35/50 — tr=0.4655 va=1.0125 acc=0.987 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.929}
2026-04-27 01:29:25,840 INFO Regime epoch 36/50 — tr=0.4655 va=1.0129 acc=0.988
2026-04-27 01:29:25,913 INFO Regime epoch 37/50 — tr=0.4655 va=1.0111 acc=0.987
2026-04-27 01:29:25,986 INFO Regime epoch 38/50 — tr=0.4655 va=1.0127 acc=0.987
2026-04-27 01:29:26,054 INFO Regime epoch 39/50 — tr=0.4655 va=1.0106 acc=0.988
2026-04-27 01:29:26,127 INFO Regime epoch 40/50 — tr=0.4652 va=1.0119 acc=0.987 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.929}
2026-04-27 01:29:26,194 INFO Regime epoch 41/50 — tr=0.4654 va=1.0148 acc=0.987
2026-04-27 01:29:26,268 INFO Regime epoch 42/50 — tr=0.4655 va=1.0122 acc=0.988
2026-04-27 01:29:26,334 INFO Regime epoch 43/50 — tr=0.4654 va=1.0134 acc=0.987
2026-04-27 01:29:26,400 INFO Regime epoch 44/50 — tr=0.4654 va=1.0137 acc=0.987
2026-04-27 01:29:26,472 INFO Regime epoch 45/50 — tr=0.4653 va=1.0138 acc=0.988 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.931}
2026-04-27 01:29:26,540 INFO Regime epoch 46/50 — tr=0.4655 va=1.0129 acc=0.987
2026-04-27 01:29:26,605 INFO Regime epoch 47/50 — tr=0.4652 va=1.0118 acc=0.987
2026-04-27 01:29:26,671 INFO Regime epoch 48/50 — tr=0.4650 va=1.0128 acc=0.987
2026-04-27 01:29:26,740 INFO Regime epoch 49/50 — tr=0.4652 va=1.0120 acc=0.988
2026-04-27 01:29:26,740 INFO Regime early stop at epoch 49 (no_improve=10)
2026-04-27 01:29:26,751 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 01:29:26,751 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 01:29:26,752 INFO Regime phase HTF train: 8.2s
2026-04-27 01:29:26,882 INFO Regime HTF complete: acc=0.988, n=103290 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.931}
2026-04-27 01:29:26,884 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:29:27,049 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 01:29:27,056 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 11.41578947368421, 'BIAS_DOWN': 10.635761589403973, 'BIAS_NEUTRAL': 17.960468521229867}
2026-04-27 01:29:27,060 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 4338, 'mean': 0.00011110466582543088, 'mean_over_std': 0.027632581210502927}, 'BIAS_DOWN': {'n': 3212, 'mean': 6.063430222660252e-05, 'mean_over_std': 0.013898489590377157}, 'BIAS_NEUTRAL': {'n': 12266, 'mean': 1.2151554902814046e-05, 'mean_over_std': 0.0031957000513162357}}
2026-04-27 01:29:27,061 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 3651, 'mean': 0.0002852242108836959, 'mean_over_std': 0.07015017232491755}, 'BIAS_DOWN': {'n': 2644, 'mean': -0.00014900749618981248, 'mean_over_std': -0.03333037041298237}, 'BIAS_NEUTRAL': {'n': 1376, 'mean': 9.595797941249082e-05, 'mean_over_std': 0.02492411099521803}}
2026-04-27 01:29:27,061 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-27 01:29:27,063 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:27,065 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:27,066 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:27,068 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:27,069 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:27,071 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:27,072 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:27,074 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:27,076 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:27,077 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:27,080 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:29:27,092 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:27,096 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:27,097 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:27,097 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:27,097 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:27,100 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:27,767 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-27 01:29:27,770 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-27 01:29:27,910 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:27,912 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:27,913 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:27,913 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:27,914 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:27,916 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:28,523 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-27 01:29:28,526 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-27 01:29:28,667 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:28,669 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:28,670 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:28,671 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:28,671 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:28,673 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:29,344 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-27 01:29:29,347 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-27 01:29:29,493 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:29,495 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:29,496 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:29,497 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:29,497 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:29,499 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:30,123 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-27 01:29:30,127 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-27 01:29:30,261 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:30,264 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:30,264 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:30,265 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:30,265 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:30,267 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:30,874 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-27 01:29:30,877 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-27 01:29:31,019 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:31,022 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:31,023 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:31,023 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:31,023 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:31,026 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:31,647 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-27 01:29:31,650 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-27 01:29:31,795 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:29:31,797 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:29:31,798 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:29:31,798 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:29:31,798 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:29:31,800 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:32,425 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-27 01:29:32,428 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-27 01:29:32,569 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:32,571 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:32,572 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:32,572 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:32,573 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:32,574 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:33,186 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-27 01:29:33,189 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-27 01:29:33,332 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:33,335 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:33,335 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:33,336 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:33,336 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:33,338 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:33,952 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-27 01:29:33,955 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-27 01:29:34,103 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:34,105 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:34,106 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:34,106 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:34,107 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:29:34,109 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:29:34,725 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-27 01:29:34,728 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-27 01:29:34,879 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:29:34,883 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:29:34,884 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:29:34,885 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:29:34,885 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:29:34,889 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:29:36,216 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 01:29:36,222 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-27 01:29:36,534 INFO Regime phase LTF dataset build: 9.5s (401471 samples)
2026-04-27 01:29:36,535 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260427_012936
2026-04-27 01:29:36,540 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=18, n_classes=4)
2026-04-27 01:29:36,543 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=299185 dropped=102286 classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127})
2026-04-27 01:29:36,610 INFO RegimeClassifier[mode=ltf_behaviour]: 299185 samples, classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127}, device=cuda
2026-04-27 01:29:36,611 INFO RegimeClassifier: sample weights — mean=0.693  ambiguous(<0.4)=0.0%
2026-04-27 01:29:36,611 INFO RegimeClassifier[mode=ltf_behaviour]: warm start from existing weights
2026-04-27 01:29:36,611 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-27 01:29:37,201 INFO Regime epoch  1/50 — tr=0.6198 va=1.2245 acc=0.835 per_class={'TRENDING': 0.811, 'RANGING': 0.77, 'CONSOLIDATING': 0.867, 'VOLATILE': 0.897}
2026-04-27 01:29:37,709 INFO Regime epoch  2/50 — tr=0.6199 va=1.2264 acc=0.836
2026-04-27 01:29:38,229 INFO Regime epoch  3/50 — tr=0.6197 va=1.2269 acc=0.837
2026-04-27 01:29:38,750 INFO Regime epoch  4/50 — tr=0.6198 va=1.2245 acc=0.837
2026-04-27 01:29:39,329 INFO Regime epoch  5/50 — tr=0.6198 va=1.2280 acc=0.836 per_class={'TRENDING': 0.815, 'RANGING': 0.77, 'CONSOLIDATING': 0.867, 'VOLATILE': 0.896}
2026-04-27 01:29:39,862 INFO Regime epoch  6/50 — tr=0.6194 va=1.2235 acc=0.835
2026-04-27 01:29:40,378 INFO Regime epoch  7/50 — tr=0.6194 va=1.2271 acc=0.838
2026-04-27 01:29:40,891 INFO Regime epoch  8/50 — tr=0.6195 va=1.2237 acc=0.839
2026-04-27 01:29:41,398 INFO Regime epoch  9/50 — tr=0.6195 va=1.2246 acc=0.839
2026-04-27 01:29:41,976 INFO Regime epoch 10/50 — tr=0.6189 va=1.2248 acc=0.838 per_class={'TRENDING': 0.819, 'RANGING': 0.77, 'CONSOLIDATING': 0.864, 'VOLATILE': 0.898}
2026-04-27 01:29:42,480 INFO Regime epoch 11/50 — tr=0.6190 va=1.2243 acc=0.839
2026-04-27 01:29:42,996 INFO Regime epoch 12/50 — tr=0.6186 va=1.2213 acc=0.839
2026-04-27 01:29:43,516 INFO Regime epoch 13/50 — tr=0.6185 va=1.2247 acc=0.837
2026-04-27 01:29:44,036 INFO Regime epoch 14/50 — tr=0.6184 va=1.2230 acc=0.839
2026-04-27 01:29:44,591 INFO Regime epoch 15/50 — tr=0.6184 va=1.2202 acc=0.840 per_class={'TRENDING': 0.824, 'RANGING': 0.772, 'CONSOLIDATING': 0.869, 'VOLATILE': 0.893}
2026-04-27 01:29:45,123 INFO Regime epoch 16/50 — tr=0.6182 va=1.2160 acc=0.837
2026-04-27 01:29:45,654 INFO Regime epoch 17/50 — tr=0.6183 va=1.2220 acc=0.840
2026-04-27 01:29:46,194 INFO Regime epoch 18/50 — tr=0.6179 va=1.2205 acc=0.840
2026-04-27 01:29:46,740 INFO Regime epoch 19/50 — tr=0.6182 va=1.2183 acc=0.839
2026-04-27 01:29:47,284 INFO Regime epoch 20/50 — tr=0.6180 va=1.2186 acc=0.841 per_class={'TRENDING': 0.826, 'RANGING': 0.767, 'CONSOLIDATING': 0.875, 'VOLATILE': 0.89}
2026-04-27 01:29:47,818 INFO Regime epoch 21/50 — tr=0.6180 va=1.2191 acc=0.837
2026-04-27 01:29:48,352 INFO Regime epoch 22/50 — tr=0.6176 va=1.2155 acc=0.840
2026-04-27 01:29:48,873 INFO Regime epoch 23/50 — tr=0.6177 va=1.2157 acc=0.839
2026-04-27 01:29:49,427 INFO Regime epoch 24/50 — tr=0.6173 va=1.2165 acc=0.839
2026-04-27 01:29:49,997 INFO Regime epoch 25/50 — tr=0.6175 va=1.2193 acc=0.840 per_class={'TRENDING': 0.825, 'RANGING': 0.769, 'CONSOLIDATING': 0.871, 'VOLATILE': 0.892}
2026-04-27 01:29:50,541 INFO Regime epoch 26/50 — tr=0.6177 va=1.2164 acc=0.840
2026-04-27 01:29:51,054 INFO Regime epoch 27/50 — tr=0.6171 va=1.2184 acc=0.840
2026-04-27 01:29:51,568 INFO Regime epoch 28/50 — tr=0.6172 va=1.2171 acc=0.840
2026-04-27 01:29:52,082 INFO Regime epoch 29/50 — tr=0.6171 va=1.2165 acc=0.840
2026-04-27 01:29:52,650 INFO Regime epoch 30/50 — tr=0.6170 va=1.2158 acc=0.840 per_class={'TRENDING': 0.824, 'RANGING': 0.772, 'CONSOLIDATING': 0.868, 'VOLATILE': 0.894}
2026-04-27 01:29:53,147 INFO Regime epoch 31/50 — tr=0.6170 va=1.2167 acc=0.842
2026-04-27 01:29:53,672 INFO Regime epoch 32/50 — tr=0.6168 va=1.2151 acc=0.841
2026-04-27 01:29:54,178 INFO Regime epoch 33/50 — tr=0.6170 va=1.2144 acc=0.841
2026-04-27 01:29:54,702 INFO Regime epoch 34/50 — tr=0.6165 va=1.2156 acc=0.840
2026-04-27 01:29:55,287 INFO Regime epoch 35/50 — tr=0.6168 va=1.2174 acc=0.840 per_class={'TRENDING': 0.823, 'RANGING': 0.775, 'CONSOLIDATING': 0.862, 'VOLATILE': 0.897}
2026-04-27 01:29:55,812 INFO Regime epoch 36/50 — tr=0.6170 va=1.2103 acc=0.839
2026-04-27 01:29:56,328 INFO Regime epoch 37/50 — tr=0.6166 va=1.2155 acc=0.841
2026-04-27 01:29:56,845 INFO Regime epoch 38/50 — tr=0.6168 va=1.2124 acc=0.841
2026-04-27 01:29:57,349 INFO Regime epoch 39/50 — tr=0.6168 va=1.2166 acc=0.842
2026-04-27 01:29:57,890 INFO Regime epoch 40/50 — tr=0.6165 va=1.2139 acc=0.840 per_class={'TRENDING': 0.825, 'RANGING': 0.774, 'CONSOLIDATING': 0.869, 'VOLATILE': 0.889}
2026-04-27 01:29:58,391 INFO Regime epoch 41/50 — tr=0.6167 va=1.2155 acc=0.840
2026-04-27 01:29:58,929 INFO Regime epoch 42/50 — tr=0.6166 va=1.2155 acc=0.842
2026-04-27 01:29:59,521 INFO Regime epoch 43/50 — tr=0.6167 va=1.2131 acc=0.839
2026-04-27 01:30:00,025 INFO Regime epoch 44/50 — tr=0.6166 va=1.2144 acc=0.840
2026-04-27 01:30:00,598 INFO Regime epoch 45/50 — tr=0.6166 va=1.2186 acc=0.838 per_class={'TRENDING': 0.817, 'RANGING': 0.777, 'CONSOLIDATING': 0.851, 'VOLATILE': 0.906}
2026-04-27 01:30:01,133 INFO Regime epoch 46/50 — tr=0.6166 va=1.2168 acc=0.841
2026-04-27 01:30:01,133 INFO Regime early stop at epoch 46 (no_improve=10)
2026-04-27 01:30:01,175 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 01:30:01,175 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 01:30:01,178 INFO Regime phase LTF train: 24.6s
2026-04-27 01:30:01,318 INFO Regime LTF complete: acc=0.839, n=401471 per_class={'TRENDING': 0.82, 'RANGING': 0.778, 'CONSOLIDATING': 0.863, 'VOLATILE': 0.895}
2026-04-27 01:30:01,321 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:30:01,852 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 01:30:01,858 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 9.355291913830783, 'RANGING': 6.096303199751476, 'CONSOLIDATING': 5.598885793871866, 'VOLATILE': 6.771351107094442}
2026-04-27 01:30:01,868 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 29965, 'mean': 2.149062259083323e-05, 'mean_over_std': 0.011155601434349563}, 'RANGING': {'n': 19623, 'mean': -3.254493608872846e-06, 'mean_over_std': -0.001761415548292494}, 'CONSOLIDATING': {'n': 10050, 'mean': -3.039756158146958e-06, 'mean_over_std': -0.001792488091618151}, 'VOLATILE': {'n': 14985, 'mean': 1.8102764458593193e-05, 'mean_over_std': 0.006974696291378742}}
2026-04-27 01:30:01,868 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 26145, 'mean': 2.874435605138053e-05, 'mean_over_std': 0.01595450632691917}, 'RANGING': {'n': 7858, 'mean': 1.7829776785521727e-05, 'mean_over_std': 0.011851561804467463}, 'CONSOLIDATING': {'n': 8543, 'mean': 7.845231961250924e-06, 'mean_over_std': 0.004915839521549481}, 'VOLATILE': {'n': 13636, 'mean': 5.480114267306668e-06, 'mean_over_std': 0.0020378119917389375}}
2026-04-27 01:30:01,869 INFO Regime retrain total: 223.2s (504761 samples)
2026-04-27 01:30:01,886 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-27 01:30:01,886 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 01:30:01,886 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 01:30:01,887 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-27 01:30:01,887 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-27 01:30:01,887 INFO Retrain complete. Total wall-clock: 223.2s
2026-04-27 01:30:04,412 INFO Model regime: SUCCESS
2026-04-27 01:30:04,412 INFO --- Training gru ---
2026-04-27 01:30:04,413 INFO Running retrain --model gru
2026-04-27 01:30:04,702 INFO retrain environment: KAGGLE
2026-04-27 01:30:06,373 INFO Device: CUDA (2 GPU(s))
2026-04-27 01:30:06,384 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 01:30:06,384 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 01:30:06,384 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 01:30:06,384 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 01:30:06,385 INFO Retrain data split: train
2026-04-27 01:30:06,386 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-27 01:30:06,536 INFO NumExpr defaulting to 4 threads.
2026-04-27 01:30:06,738 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-27 01:30:06,738 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 01:30:06,738 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 01:30:07,024 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-27 01:30:07,024 INFO GRU phase macro_correlations: 0.0s
2026-04-27 01:30:07,024 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-27 01:30:07,026 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260427_013007
2026-04-27 01:30:07,030 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-04-27 01:30:07,182 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:07,205 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:07,220 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:07,228 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:07,230 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-27 01:30:07,230 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 01:30:07,230 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 01:30:07,230 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-27 01:30:07,231 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:07,322 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-27 01:30:07,324 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:07,574 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-27 01:30:07,604 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:07,893 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:08,025 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:08,131 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:08,347 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:08,367 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:08,382 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:08,389 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:08,390 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:08,478 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-27 01:30:08,480 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:08,733 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-27 01:30:08,749 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:09,038 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:09,186 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:09,296 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:09,492 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:09,514 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:09,529 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:09,536 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:09,537 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:09,619 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-27 01:30:09,621 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:09,868 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-27 01:30:09,886 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:10,163 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:10,296 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:10,396 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:10,589 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:10,610 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:10,624 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:10,632 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:10,632 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:10,717 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-27 01:30:10,719 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:10,970 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-27 01:30:10,994 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:11,281 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:11,412 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:11,517 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:11,708 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:11,731 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:11,747 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:11,755 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:11,756 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:11,841 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-27 01:30:11,843 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:12,097 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-27 01:30:12,112 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:12,404 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:12,539 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:12,645 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:12,856 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:12,878 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:12,894 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:12,902 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:12,903 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:12,997 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-27 01:30:12,999 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:13,265 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-27 01:30:13,281 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:13,565 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:13,701 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:13,805 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:13,986 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:30:14,004 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:30:14,019 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:30:14,026 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:30:14,026 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:14,112 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-27 01:30:14,114 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:14,375 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-27 01:30:14,388 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:14,658 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:14,788 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:14,892 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:15,080 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:15,099 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:15,114 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:15,121 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:15,122 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:15,209 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-27 01:30:15,211 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:15,467 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-27 01:30:15,486 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:15,760 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:15,902 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:16,014 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:16,215 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:16,235 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:16,249 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:16,257 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:16,258 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:16,345 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-27 01:30:16,347 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:16,604 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-27 01:30:16,620 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:16,910 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:17,041 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:17,141 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:17,337 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:17,358 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:17,373 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:17,380 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:30:17,381 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:17,468 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-27 01:30:17,470 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:17,720 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-27 01:30:17,736 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:18,014 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:18,150 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:18,251 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:30:18,570 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:30:18,595 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:30:18,612 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:30:18,622 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:30:18,623 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:30:18,792 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 01:30:18,795 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:30:19,406 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 01:30:19,455 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:30:20,001 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:30:20,219 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:30:20,363 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:30:20,490 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-27 01:30:20,490 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-27 01:30:20,490 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-27 01:31:11,072 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-27 01:31:11,073 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-27 01:31:12,414 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-27 01:31:16,570 INFO train_multi TF=ALL: warm-start detected — using CosineAnnealingLR (lr=3e-05, patience=12)
2026-04-27 01:31:30,764 INFO train_multi TF=ALL epoch 1/50 train=0.5884 val=0.6113
2026-04-27 01:31:30,768 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 01:31:30,768 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 01:31:30,768 INFO train_multi TF=ALL: new best val=0.6113 — saved
2026-04-27 01:31:42,891 INFO train_multi TF=ALL epoch 2/50 train=0.5883 val=0.6109
2026-04-27 01:31:42,895 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 01:31:42,895 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 01:31:42,896 INFO train_multi TF=ALL: new best val=0.6109 — saved
2026-04-27 01:31:55,011 INFO train_multi TF=ALL epoch 3/50 train=0.5885 val=0.6111
2026-04-27 01:32:07,077 INFO train_multi TF=ALL epoch 4/50 train=0.5882 val=0.6113
2026-04-27 01:32:19,111 INFO train_multi TF=ALL epoch 5/50 train=0.5876 val=0.6110
2026-04-27 01:32:31,379 INFO train_multi TF=ALL epoch 6/50 train=0.5877 val=0.6110
2026-04-27 01:32:43,590 INFO train_multi TF=ALL epoch 7/50 train=0.5874 val=0.6114
2026-04-27 01:32:55,745 INFO train_multi TF=ALL epoch 8/50 train=0.5872 val=0.6112
2026-04-27 01:33:07,978 INFO train_multi TF=ALL epoch 9/50 train=0.5871 val=0.6116
2026-04-27 01:33:20,132 INFO train_multi TF=ALL epoch 10/50 train=0.5868 val=0.6113
2026-04-27 01:33:32,357 INFO train_multi TF=ALL epoch 11/50 train=0.5868 val=0.6114
2026-04-27 01:33:44,467 INFO train_multi TF=ALL epoch 12/50 train=0.5869 val=0.6114
2026-04-27 01:33:56,653 INFO train_multi TF=ALL epoch 13/50 train=0.5866 val=0.6108
2026-04-27 01:33:56,658 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 01:33:56,658 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 01:33:56,658 INFO train_multi TF=ALL: new best val=0.6108 — saved
2026-04-27 01:34:08,737 INFO train_multi TF=ALL epoch 14/50 train=0.5864 val=0.6108
2026-04-27 01:34:20,893 INFO train_multi TF=ALL epoch 15/50 train=0.5863 val=0.6127
2026-04-27 01:34:32,997 INFO train_multi TF=ALL epoch 16/50 train=0.5861 val=0.6110
2026-04-27 01:34:45,239 INFO train_multi TF=ALL epoch 17/50 train=0.5858 val=0.6117
2026-04-27 01:34:57,392 INFO train_multi TF=ALL epoch 18/50 train=0.5861 val=0.6113
2026-04-27 01:35:09,561 INFO train_multi TF=ALL epoch 19/50 train=0.5859 val=0.6117
2026-04-27 01:35:21,596 INFO train_multi TF=ALL epoch 20/50 train=0.5858 val=0.6117
2026-04-27 01:35:33,682 INFO train_multi TF=ALL epoch 21/50 train=0.5857 val=0.6119
2026-04-27 01:35:45,725 INFO train_multi TF=ALL epoch 22/50 train=0.5853 val=0.6120
2026-04-27 01:35:57,783 INFO train_multi TF=ALL epoch 23/50 train=0.5853 val=0.6120
2026-04-27 01:36:09,826 INFO train_multi TF=ALL epoch 24/50 train=0.5852 val=0.6118
2026-04-27 01:36:21,788 INFO train_multi TF=ALL epoch 25/50 train=0.5849 val=0.6122
2026-04-27 01:36:21,789 INFO train_multi TF=ALL early stop at epoch 25
2026-04-27 01:36:21,927 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-27 01:36:21,927 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-27 01:36:21,927 INFO Retrain complete. Total wall-clock: 375.5s
2026-04-27 01:36:23,973 INFO Model gru: SUCCESS
2026-04-27 01:36:23,974 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 01:36:23,974 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 01:36:23,974 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 01:36:23,974 INFO   [OK] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-27 01:36:23,974 INFO   [OK] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-27 01:36:23,974 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-04-27 01:36:23,975 INFO Saved 21 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  Cleared journal for fresh Round 1 run
  START Round 1 - Backtest (val)
2026-04-27 01:36:24,746 INFO === STEP 6: BACKTEST (round1) ===
2026-04-27 01:36:24,747 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-04-27 01:36:24,748 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-27 01:36:24,748 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
2026-04-27 01:38:38,466 WARNING ml_trader: portfolio drawdown 8.6% — halting all trading

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260427_013626.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)               60  28.3%   1.00   -0.1% 28.3% 10.0%   8.6%   -0.01
  gate_diagnostics: bars=15249 no_signal=227 quality_block=0 session_skip=5000 density=3 pm_reject=0 daily_skip=9878 cooldown=80 daily_halt_events=15 enforce_daily_halt=True
  no_signal_reasons: htf_bias_conflict=186, weak_gru_direction=35, trend_pullback_conflict=6

Calibration Summary:
  all          [WARN] Non-monotonic calibration: 3/5 pairs violated. Consider retraining QualityScorer
  ml_trader    [WARN] Non-monotonic calibration: 3/5 pairs violated. Consider retraining QualityScorer
2026-04-27 01:38:39,164 INFO Round 1 backtest — 60 trades | avg WR=28.3% | avg PF=1.00 | avg Sharpe=-0.01
2026-04-27 01:38:39,164 INFO   ml_trader: 60 trades | WR=28.3% | PF=1.00 | Return=-0.1% | DD=8.6% | Sharpe=-0.01
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 60
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (60 rows)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  AUDUSD          8 trades     8 days   1.00/day
  EURGBP         10 trades    10 days   1.00/day
  EURJPY          8 trades     8 days   1.00/day
  EURUSD          6 trades     6 days   1.00/day
  GBPJPY          5 trades     5 days   1.00/day
  GBPUSD          2 trades     2 days   1.00/day
  NZDUSD          4 trades     3 days   1.33/day
  USDCAD          6 trades     5 days   1.20/day
  USDCHF          4 trades     4 days   1.00/day
  USDJPY          3 trades     3 days   1.00/day
  XAUUSD          4 trades     4 days   1.00/day
  ✓  All symbols within normal range.
  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 60 entries

=== Round 1 → Quality + RL retrain skipped (validation journal is not train data) ===

=== Round 2: BLIND backtest on test window (unseen 2yr) ===
  START Round 2 - Blind backtest (test)
2026-04-27 01:38:39,453 INFO Round 1: wrote 60 journal entries (total in file: 60)
2026-04-27 01:38:40,078 INFO === STEP 6: BACKTEST (round2) ===
2026-04-27 01:38:40,079 INFO BT_WINDOW=round2 — BLIND backtest: 2023-08-07 → 2025-08-05 (test set)
2026-04-27 01:38:40,079 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-27 01:38:40,079 INFO Round 2 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-27 01:40:55,168 WARNING ml_trader: portfolio drawdown 8.9% — halting all trading
2026-04-27 01:40:55,854 INFO Round 2 backtest — 22 trades | avg WR=27.3% | avg PF=1.48 | avg Sharpe=2.26
2026-04-27 01:40:55,854 INFO   ml_trader: 22 trades | WR=27.3% | PF=1.48 | Return=5.1% | DD=8.9% | Sharpe=2.26
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 22
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (22 rows)
2026-04-27 01:40:56,128 INFO Round 2: wrote 22 journal entries (total in file: 82)
  DONE  Round 2 - Blind backtest (test)
  Saved Round 2 result → round2_summary.json
  Journal after Round 2: 82 entries

=== Round 2 → Quality + RL retrain skipped (blind-test journal is not train data) ===

=== Round 3: Incremental retrain on train split only ===
  START Retrain gru [train-split retrain]
2026-04-27 01:40:56,431 INFO retrain environment: KAGGLE
2026-04-27 01:40:58,116 INFO Device: CUDA (2 GPU(s))
2026-04-27 01:40:58,126 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 01:40:58,127 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 01:40:58,127 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 01:40:58,127 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 01:40:58,127 INFO Retrain data split: train
2026-04-27 01:40:58,128 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-27 01:40:58,282 INFO NumExpr defaulting to 4 threads.
2026-04-27 01:40:58,489 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-27 01:40:58,489 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 01:40:58,489 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 01:40:58,735 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-27 01:40:58,735 INFO GRU phase macro_correlations: 0.0s
2026-04-27 01:40:58,735 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-27 01:40:58,737 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260427_014058
2026-04-27 01:40:58,741 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-04-27 01:40:58,907 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:40:58,928 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:40:58,943 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:40:58,951 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:40:58,952 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-27 01:40:58,952 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 01:40:58,952 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 01:40:58,953 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-27 01:40:58,954 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:40:59,041 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-27 01:40:59,043 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:40:59,334 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-27 01:40:59,365 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:40:59,651 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:40:59,784 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:40:59,887 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:00,103 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:00,121 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:00,136 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:00,143 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:00,144 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:00,233 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-27 01:41:00,235 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:00,480 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-27 01:41:00,495 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:00,773 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:00,907 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:01,008 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:01,205 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:01,227 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:01,243 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:01,251 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:01,252 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:01,341 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-27 01:41:01,343 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:01,600 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-27 01:41:01,618 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:01,900 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:02,038 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:02,144 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:02,344 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:02,365 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:02,380 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:02,388 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:02,389 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:02,473 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-27 01:41:02,475 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:02,724 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-27 01:41:02,748 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:03,028 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:03,161 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:03,269 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:03,461 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:03,481 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:03,495 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:03,503 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:03,504 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:03,587 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-27 01:41:03,590 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:03,835 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-27 01:41:03,852 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:04,131 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:04,273 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:04,378 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:04,570 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:04,591 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:04,605 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:04,613 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:04,614 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:04,700 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-27 01:41:04,701 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:04,954 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-27 01:41:04,970 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:05,254 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:05,387 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:05,487 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:05,663 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:41:05,680 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:41:05,695 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:41:05,701 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:41:05,702 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:05,786 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-27 01:41:05,787 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:06,049 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-27 01:41:06,062 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:06,329 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:06,453 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:06,552 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:06,753 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:06,773 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:06,788 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:06,796 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:06,797 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:06,880 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-27 01:41:06,882 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:07,128 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-27 01:41:07,146 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:07,418 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:07,554 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:07,657 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:07,850 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:07,870 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:07,884 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:07,891 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:07,892 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:07,973 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-27 01:41:07,974 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:08,224 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-27 01:41:08,240 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:08,511 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:08,644 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:08,746 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:08,941 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:08,963 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:08,978 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:08,985 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:41:08,986 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:09,073 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-27 01:41:09,075 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:09,363 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-27 01:41:09,378 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:09,658 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:09,794 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:09,897 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:41:10,202 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:41:10,226 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:41:10,244 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:41:10,255 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:41:10,256 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:41:10,431 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 01:41:10,434 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:41:10,956 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 01:41:11,004 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:41:11,558 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:41:11,763 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:41:11,899 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:41:12,019 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-27 01:41:12,019 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-27 01:41:12,019 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-27 01:42:02,099 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-27 01:42:02,100 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-27 01:42:03,424 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-27 01:42:07,658 INFO train_multi TF=ALL: warm-start detected — using CosineAnnealingLR (lr=3e-05, patience=12)
2026-04-27 01:42:21,501 INFO train_multi TF=ALL epoch 1/50 train=0.5866 val=0.6122
2026-04-27 01:42:21,505 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 01:42:21,505 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 01:42:21,506 INFO train_multi TF=ALL: new best val=0.6122 — saved
2026-04-27 01:42:33,630 INFO train_multi TF=ALL epoch 2/50 train=0.5861 val=0.6117
2026-04-27 01:42:33,635 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 01:42:33,635 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 01:42:33,636 INFO train_multi TF=ALL: new best val=0.6117 — saved
2026-04-27 01:42:45,739 INFO train_multi TF=ALL epoch 3/50 train=0.5859 val=0.6118
2026-04-27 01:42:57,987 INFO train_multi TF=ALL epoch 4/50 train=0.5858 val=0.6115
2026-04-27 01:42:57,991 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 01:42:57,992 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 01:42:57,992 INFO train_multi TF=ALL: new best val=0.6115 — saved
2026-04-27 01:43:10,042 INFO train_multi TF=ALL epoch 5/50 train=0.5860 val=0.6114
2026-04-27 01:43:10,047 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 01:43:10,047 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 01:43:10,047 INFO train_multi TF=ALL: new best val=0.6114 — saved
2026-04-27 01:43:22,025 INFO train_multi TF=ALL epoch 6/50 train=0.5854 val=0.6121
2026-04-27 01:43:34,055 INFO train_multi TF=ALL epoch 7/50 train=0.5854 val=0.6118
2026-04-27 01:43:46,049 INFO train_multi TF=ALL epoch 8/50 train=0.5853 val=0.6116
2026-04-27 01:43:58,137 INFO train_multi TF=ALL epoch 9/50 train=0.5854 val=0.6124
2026-04-27 01:44:10,403 INFO train_multi TF=ALL epoch 10/50 train=0.5849 val=0.6118
2026-04-27 01:44:22,521 INFO train_multi TF=ALL epoch 11/50 train=0.5848 val=0.6124
2026-04-27 01:44:34,650 INFO train_multi TF=ALL epoch 12/50 train=0.5849 val=0.6122
2026-04-27 01:44:46,817 INFO train_multi TF=ALL epoch 13/50 train=0.5846 val=0.6116
2026-04-27 01:44:58,877 INFO train_multi TF=ALL epoch 14/50 train=0.5845 val=0.6120
2026-04-27 01:45:11,053 INFO train_multi TF=ALL epoch 15/50 train=0.5845 val=0.6125
2026-04-27 01:45:23,064 INFO train_multi TF=ALL epoch 16/50 train=0.5844 val=0.6126
2026-04-27 01:45:35,184 INFO train_multi TF=ALL epoch 17/50 train=0.5841 val=0.6119
2026-04-27 01:45:35,184 INFO train_multi TF=ALL early stop at epoch 17
2026-04-27 01:45:35,327 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-27 01:45:35,328 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-27 01:45:35,328 INFO Retrain complete. Total wall-clock: 277.2s
  DONE  Retrain gru [train-split retrain]
  START Retrain regime [train-split retrain]
2026-04-27 01:45:37,578 INFO retrain environment: KAGGLE
2026-04-27 01:45:39,246 INFO Device: CUDA (2 GPU(s))
2026-04-27 01:45:39,255 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 01:45:39,256 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 01:45:39,256 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 01:45:39,256 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 01:45:39,256 INFO Retrain data split: train
2026-04-27 01:45:39,257 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-27 01:45:39,412 INFO NumExpr defaulting to 4 threads.
2026-04-27 01:45:39,611 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-27 01:45:39,612 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 01:45:39,612 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 01:45:39,612 INFO Regime phase macro_correlations: 0.0s
2026-04-27 01:45:39,612 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-27 01:45:39,651 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-27 01:45:39,652 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:45:39,681 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:45:39,696 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:45:39,720 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:45:39,736 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:45:39,761 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:45:39,776 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:45:39,799 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:45:39,814 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:45:39,837 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:45:39,851 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:45:39,874 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:45:39,888 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:45:39,906 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:45:39,921 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:45:39,945 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:45:39,960 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:45:39,981 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:45:39,996 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:45:40,020 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:45:40,037 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:45:40,076 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:45:40,849 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 01:46:05,394 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 01:46:05,396 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 25.3s
2026-04-27 01:46:05,396 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 01:46:16,666 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 01:46:16,667 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 11.3s
2026-04-27 01:46:16,670 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 01:46:25,111 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 01:46:25,111 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 8.4s
2026-04-27 01:46:25,111 INFO Regime phase GMM HTF total: 45.0s
2026-04-27 01:46:25,112 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 01:47:41,250 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 01:47:41,254 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 76.1s
2026-04-27 01:47:41,255 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 01:48:14,475 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 01:48:14,477 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 33.2s
2026-04-27 01:48:14,477 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 01:48:38,169 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 01:48:38,171 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 23.7s
2026-04-27 01:48:38,171 INFO Regime phase GMM LTF total: 133.1s
2026-04-27 01:48:38,289 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-27 01:48:38,291 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:38,292 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:38,293 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:38,295 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:38,295 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:38,296 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:38,297 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:38,298 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:38,299 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:38,300 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:38,302 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:48:38,430 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:38,474 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:38,475 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:38,475 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:38,484 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:38,485 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:38,931 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-27 01:48:38,932 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-27 01:48:39,131 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:39,178 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:39,179 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:39,180 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:39,189 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:39,190 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:39,600 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-27 01:48:39,601 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-27 01:48:39,813 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:39,851 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:39,852 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:39,853 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:39,861 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:39,862 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:40,274 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-27 01:48:40,275 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-27 01:48:40,467 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:40,503 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:40,504 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:40,504 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:40,512 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:40,513 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:40,913 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-27 01:48:40,914 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-27 01:48:41,111 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:41,148 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:41,149 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:41,150 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:41,158 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:41,159 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:41,577 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-27 01:48:41,578 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-27 01:48:41,767 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:41,801 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:41,802 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:41,803 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:41,811 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:41,812 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:42,225 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-27 01:48:42,227 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-27 01:48:42,398 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:48:42,427 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:48:42,428 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:48:42,428 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:48:42,435 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:48:42,436 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:42,839 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-27 01:48:42,840 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-27 01:48:43,023 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:43,059 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:43,060 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:43,060 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:43,069 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:43,070 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:43,495 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-27 01:48:43,496 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-27 01:48:43,685 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:43,720 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:43,721 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:43,721 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:43,730 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:43,731 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:44,147 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-27 01:48:44,149 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-27 01:48:44,342 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:44,381 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:44,382 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:44,383 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:44,392 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:44,393 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:44,801 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-27 01:48:44,803 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-27 01:48:45,094 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:48:45,155 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:48:45,156 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:48:45,157 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:48:45,168 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:48:45,169 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:48:46,062 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 01:48:46,064 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-27 01:48:46,261 INFO Regime phase HTF dataset build: 8.0s (103290 samples)
2026-04-27 01:48:46,261 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260427_014846
2026-04-27 01:48:46,465 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=34, n_classes=3)
2026-04-27 01:48:46,466 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=38114 dropped=65176 classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728})
2026-04-27 01:48:46,476 INFO RegimeClassifier[mode=htf_bias]: 38114 samples, classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728}, device=cuda
2026-04-27 01:48:46,476 INFO RegimeClassifier: sample weights — mean=0.708  ambiguous(<0.4)=0.0%
2026-04-27 01:48:46,476 INFO RegimeClassifier[mode=htf_bias]: warm start from existing weights
2026-04-27 01:48:46,476 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-27 01:48:48,887 INFO Regime epoch  1/50 — tr=0.4651 va=1.0124 acc=0.987 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.93}
2026-04-27 01:48:48,962 INFO Regime epoch  2/50 — tr=0.4652 va=1.0127 acc=0.987
2026-04-27 01:48:49,034 INFO Regime epoch  3/50 — tr=0.4654 va=1.0136 acc=0.987
2026-04-27 01:48:49,107 INFO Regime epoch  4/50 — tr=0.4651 va=1.0129 acc=0.987
2026-04-27 01:48:49,220 INFO Regime epoch  5/50 — tr=0.4654 va=1.0117 acc=0.987 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.929}
2026-04-27 01:48:49,297 INFO Regime epoch  6/50 — tr=0.4652 va=1.0118 acc=0.987
2026-04-27 01:48:49,372 INFO Regime epoch  7/50 — tr=0.4654 va=1.0088 acc=0.988
2026-04-27 01:48:49,446 INFO Regime epoch  8/50 — tr=0.4647 va=1.0076 acc=0.987
2026-04-27 01:48:49,518 INFO Regime epoch  9/50 — tr=0.4646 va=1.0054 acc=0.988
2026-04-27 01:48:49,597 INFO Regime epoch 10/50 — tr=0.4645 va=1.0039 acc=0.988 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.932}
2026-04-27 01:48:49,669 INFO Regime epoch 11/50 — tr=0.4646 va=1.0044 acc=0.988
2026-04-27 01:48:49,741 INFO Regime epoch 12/50 — tr=0.4642 va=1.0038 acc=0.988
2026-04-27 01:48:49,812 INFO Regime epoch 13/50 — tr=0.4643 va=0.9992 acc=0.988
2026-04-27 01:48:49,885 INFO Regime epoch 14/50 — tr=0.4639 va=0.9986 acc=0.989
2026-04-27 01:48:49,966 INFO Regime epoch 15/50 — tr=0.4639 va=0.9967 acc=0.988 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.933}
2026-04-27 01:48:50,040 INFO Regime epoch 16/50 — tr=0.4635 va=0.9954 acc=0.989
2026-04-27 01:48:50,116 INFO Regime epoch 17/50 — tr=0.4635 va=0.9929 acc=0.989
2026-04-27 01:48:50,188 INFO Regime epoch 18/50 — tr=0.4635 va=0.9913 acc=0.989
2026-04-27 01:48:50,262 INFO Regime epoch 19/50 — tr=0.4632 va=0.9901 acc=0.989
2026-04-27 01:48:50,347 INFO Regime epoch 20/50 — tr=0.4632 va=0.9909 acc=0.989 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.941}
2026-04-27 01:48:50,422 INFO Regime epoch 21/50 — tr=0.4627 va=0.9904 acc=0.990
2026-04-27 01:48:50,497 INFO Regime epoch 22/50 — tr=0.4630 va=0.9914 acc=0.990
2026-04-27 01:48:50,568 INFO Regime epoch 23/50 — tr=0.4630 va=0.9885 acc=0.989
2026-04-27 01:48:50,642 INFO Regime epoch 24/50 — tr=0.4626 va=0.9867 acc=0.989
2026-04-27 01:48:50,727 INFO Regime epoch 25/50 — tr=0.4624 va=0.9880 acc=0.989 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.941}
2026-04-27 01:48:50,805 INFO Regime epoch 26/50 — tr=0.4623 va=0.9867 acc=0.990
2026-04-27 01:48:50,882 INFO Regime epoch 27/50 — tr=0.4626 va=0.9844 acc=0.990
2026-04-27 01:48:50,958 INFO Regime epoch 28/50 — tr=0.4621 va=0.9860 acc=0.990
2026-04-27 01:48:51,031 INFO Regime epoch 29/50 — tr=0.4624 va=0.9845 acc=0.990
2026-04-27 01:48:51,107 INFO Regime epoch 30/50 — tr=0.4623 va=0.9827 acc=0.990 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.947}
2026-04-27 01:48:51,178 INFO Regime epoch 31/50 — tr=0.4624 va=0.9810 acc=0.990
2026-04-27 01:48:51,253 INFO Regime epoch 32/50 — tr=0.4621 va=0.9804 acc=0.990
2026-04-27 01:48:51,324 INFO Regime epoch 33/50 — tr=0.4622 va=0.9811 acc=0.990
2026-04-27 01:48:51,398 INFO Regime epoch 34/50 — tr=0.4622 va=0.9817 acc=0.990
2026-04-27 01:48:51,477 INFO Regime epoch 35/50 — tr=0.4618 va=0.9810 acc=0.990 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.945}
2026-04-27 01:48:51,547 INFO Regime epoch 36/50 — tr=0.4619 va=0.9820 acc=0.990
2026-04-27 01:48:51,623 INFO Regime epoch 37/50 — tr=0.4618 va=0.9824 acc=0.991
2026-04-27 01:48:51,701 INFO Regime epoch 38/50 — tr=0.4617 va=0.9794 acc=0.991
2026-04-27 01:48:51,779 INFO Regime epoch 39/50 — tr=0.4618 va=0.9811 acc=0.991
2026-04-27 01:48:51,862 INFO Regime epoch 40/50 — tr=0.4621 va=0.9812 acc=0.991 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.95}
2026-04-27 01:48:51,942 INFO Regime epoch 41/50 — tr=0.4617 va=0.9800 acc=0.991
2026-04-27 01:48:52,018 INFO Regime epoch 42/50 — tr=0.4618 va=0.9796 acc=0.991
2026-04-27 01:48:52,093 INFO Regime epoch 43/50 — tr=0.4615 va=0.9797 acc=0.991
2026-04-27 01:48:52,173 INFO Regime epoch 44/50 — tr=0.4616 va=0.9797 acc=0.991
2026-04-27 01:48:52,258 INFO Regime epoch 45/50 — tr=0.4615 va=0.9781 acc=0.991 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.95}
2026-04-27 01:48:52,338 INFO Regime epoch 46/50 — tr=0.4617 va=0.9784 acc=0.990
2026-04-27 01:48:52,414 INFO Regime epoch 47/50 — tr=0.4618 va=0.9789 acc=0.991
2026-04-27 01:48:52,485 INFO Regime epoch 48/50 — tr=0.4618 va=0.9785 acc=0.991
2026-04-27 01:48:52,558 INFO Regime epoch 49/50 — tr=0.4619 va=0.9798 acc=0.990
2026-04-27 01:48:52,636 INFO Regime epoch 50/50 — tr=0.4618 va=0.9790 acc=0.991 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.948}
2026-04-27 01:48:52,645 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 01:48:52,646 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 01:48:52,646 INFO Regime phase HTF train: 6.2s
2026-04-27 01:48:52,786 INFO Regime HTF complete: acc=0.991, n=103290 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.95}
2026-04-27 01:48:52,788 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:48:52,967 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 01:48:52,970 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 11.41578947368421, 'BIAS_DOWN': 10.635761589403973, 'BIAS_NEUTRAL': 17.960468521229867}
2026-04-27 01:48:52,974 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 4338, 'mean': 0.00011110466582543088, 'mean_over_std': 0.027632581210502927}, 'BIAS_DOWN': {'n': 3212, 'mean': 6.063430222660252e-05, 'mean_over_std': 0.013898489590377157}, 'BIAS_NEUTRAL': {'n': 12266, 'mean': 1.2151554902814046e-05, 'mean_over_std': 0.0031957000513162357}}
2026-04-27 01:48:52,975 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 3651, 'mean': 0.0002852242108836959, 'mean_over_std': 0.07015017232491755}, 'BIAS_DOWN': {'n': 2644, 'mean': -0.00014900749618981248, 'mean_over_std': -0.03333037041298237}, 'BIAS_NEUTRAL': {'n': 1376, 'mean': 9.595797941249082e-05, 'mean_over_std': 0.02492411099521803}}
2026-04-27 01:48:52,975 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-27 01:48:52,977 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:52,979 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:52,980 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:52,982 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:52,984 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:52,986 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:52,987 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:52,989 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:52,991 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:52,992 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:52,995 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:48:53,007 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:53,010 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:53,011 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:53,011 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:53,012 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:53,014 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:53,712 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-27 01:48:53,715 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-27 01:48:53,865 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:53,868 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:53,869 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:53,869 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:53,869 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:53,872 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:54,545 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-27 01:48:54,548 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-27 01:48:54,705 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:54,707 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:54,708 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:54,708 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:54,709 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:54,711 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:55,372 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-27 01:48:55,375 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-27 01:48:55,524 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:55,528 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:55,528 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:55,529 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:55,529 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:55,531 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:56,184 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-27 01:48:56,187 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-27 01:48:56,340 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:56,342 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:56,343 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:56,343 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:56,344 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:56,346 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:56,997 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-27 01:48:57,000 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-27 01:48:57,147 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:57,151 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:57,152 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:57,153 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:57,153 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:57,155 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:57,816 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-27 01:48:57,819 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-27 01:48:57,967 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:48:57,969 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:48:57,970 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:48:57,970 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:48:57,970 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 01:48:57,972 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:58,617 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-27 01:48:58,620 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-27 01:48:58,767 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:58,769 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:58,770 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:58,771 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:58,771 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:58,773 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:48:59,468 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-27 01:48:59,471 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-27 01:48:59,621 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:59,624 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:59,625 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:59,625 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:59,626 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:48:59,628 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:49:00,270 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-27 01:49:00,273 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-27 01:49:00,423 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:49:00,425 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:49:00,426 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:49:00,427 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:49:00,427 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 01:49:00,429 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 01:49:01,079 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-27 01:49:01,082 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-27 01:49:01,238 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:49:01,242 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:49:01,243 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:49:01,243 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:49:01,244 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 01:49:01,247 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:49:02,643 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 01:49:02,650 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-27 01:49:02,983 INFO Regime phase LTF dataset build: 10.0s (401471 samples)
2026-04-27 01:49:02,984 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260427_014902
2026-04-27 01:49:02,989 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=18, n_classes=4)
2026-04-27 01:49:02,992 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=299185 dropped=102286 classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127})
2026-04-27 01:49:03,053 INFO RegimeClassifier[mode=ltf_behaviour]: 299185 samples, classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127}, device=cuda
2026-04-27 01:49:03,054 INFO RegimeClassifier: sample weights — mean=0.693  ambiguous(<0.4)=0.0%
2026-04-27 01:49:03,054 INFO RegimeClassifier[mode=ltf_behaviour]: warm start from existing weights
2026-04-27 01:49:03,054 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-27 01:49:03,657 INFO Regime epoch  1/50 — tr=0.6170 va=1.2146 acc=0.841 per_class={'TRENDING': 0.827, 'RANGING': 0.772, 'CONSOLIDATING': 0.869, 'VOLATILE': 0.891}
2026-04-27 01:49:04,190 INFO Regime epoch  2/50 — tr=0.6168 va=1.2135 acc=0.840
2026-04-27 01:49:04,744 INFO Regime epoch  3/50 — tr=0.6168 va=1.2130 acc=0.842
2026-04-27 01:49:05,290 INFO Regime epoch  4/50 — tr=0.6166 va=1.2161 acc=0.842
2026-04-27 01:49:05,896 INFO Regime epoch  5/50 — tr=0.6166 va=1.2150 acc=0.841 per_class={'TRENDING': 0.824, 'RANGING': 0.778, 'CONSOLIDATING': 0.862, 'VOLATILE': 0.895}
2026-04-27 01:49:06,423 INFO Regime epoch  6/50 — tr=0.6167 va=1.2123 acc=0.840
2026-04-27 01:49:06,958 INFO Regime epoch  7/50 — tr=0.6168 va=1.2146 acc=0.842
2026-04-27 01:49:07,511 INFO Regime epoch  8/50 — tr=0.6165 va=1.2157 acc=0.844
2026-04-27 01:49:08,050 INFO Regime epoch  9/50 — tr=0.6165 va=1.2143 acc=0.841
2026-04-27 01:49:08,653 INFO Regime epoch 10/50 — tr=0.6164 va=1.2128 acc=0.841 per_class={'TRENDING': 0.825, 'RANGING': 0.775, 'CONSOLIDATING': 0.868, 'VOLATILE': 0.892}
2026-04-27 01:49:09,214 INFO Regime epoch 11/50 — tr=0.6163 va=1.2132 acc=0.840
2026-04-27 01:49:09,745 INFO Regime epoch 12/50 — tr=0.6166 va=1.2120 acc=0.840
2026-04-27 01:49:10,271 INFO Regime epoch 13/50 — tr=0.6161 va=1.2119 acc=0.841
2026-04-27 01:49:10,807 INFO Regime epoch 14/50 — tr=0.6159 va=1.2108 acc=0.842
2026-04-27 01:49:11,388 INFO Regime epoch 15/50 — tr=0.6161 va=1.2132 acc=0.841 per_class={'TRENDING': 0.823, 'RANGING': 0.778, 'CONSOLIDATING': 0.864, 'VOLATILE': 0.897}
2026-04-27 01:49:11,934 INFO Regime epoch 16/50 — tr=0.6158 va=1.2121 acc=0.841
2026-04-27 01:49:12,493 INFO Regime epoch 17/50 — tr=0.6158 va=1.2096 acc=0.841
2026-04-27 01:49:13,074 INFO Regime epoch 18/50 — tr=0.6157 va=1.2109 acc=0.841
2026-04-27 01:49:13,622 INFO Regime epoch 19/50 — tr=0.6155 va=1.2118 acc=0.842
2026-04-27 01:49:14,230 INFO Regime epoch 20/50 — tr=0.6156 va=1.2106 acc=0.841 per_class={'TRENDING': 0.826, 'RANGING': 0.78, 'CONSOLIDATING': 0.861, 'VOLATILE': 0.895}
2026-04-27 01:49:14,775 INFO Regime epoch 21/50 — tr=0.6154 va=1.2080 acc=0.842
2026-04-27 01:49:15,317 INFO Regime epoch 22/50 — tr=0.6151 va=1.2096 acc=0.842
2026-04-27 01:49:15,886 INFO Regime epoch 23/50 — tr=0.6152 va=1.2098 acc=0.843
2026-04-27 01:49:16,432 INFO Regime epoch 24/50 — tr=0.6150 va=1.2135 acc=0.843
2026-04-27 01:49:17,039 INFO Regime epoch 25/50 — tr=0.6150 va=1.2107 acc=0.843 per_class={'TRENDING': 0.829, 'RANGING': 0.779, 'CONSOLIDATING': 0.861, 'VOLATILE': 0.895}
2026-04-27 01:49:17,606 INFO Regime epoch 26/50 — tr=0.6153 va=1.2104 acc=0.841
2026-04-27 01:49:18,172 INFO Regime epoch 27/50 — tr=0.6151 va=1.2085 acc=0.842
2026-04-27 01:49:18,721 INFO Regime epoch 28/50 — tr=0.6148 va=1.2097 acc=0.844
2026-04-27 01:49:19,298 INFO Regime epoch 29/50 — tr=0.6151 va=1.2048 acc=0.843
2026-04-27 01:49:19,910 INFO Regime epoch 30/50 — tr=0.6151 va=1.2052 acc=0.842 per_class={'TRENDING': 0.827, 'RANGING': 0.78, 'CONSOLIDATING': 0.863, 'VOLATILE': 0.891}
2026-04-27 01:49:20,473 INFO Regime epoch 31/50 — tr=0.6146 va=1.2051 acc=0.841
2026-04-27 01:49:21,034 INFO Regime epoch 32/50 — tr=0.6147 va=1.2101 acc=0.843
2026-04-27 01:49:21,612 INFO Regime epoch 33/50 — tr=0.6148 va=1.2111 acc=0.842
2026-04-27 01:49:22,175 INFO Regime epoch 34/50 — tr=0.6149 va=1.2067 acc=0.842
2026-04-27 01:49:22,762 INFO Regime epoch 35/50 — tr=0.6150 va=1.2078 acc=0.842 per_class={'TRENDING': 0.829, 'RANGING': 0.783, 'CONSOLIDATING': 0.853, 'VOLATILE': 0.896}
2026-04-27 01:49:23,300 INFO Regime epoch 36/50 — tr=0.6147 va=1.2063 acc=0.843
2026-04-27 01:49:23,868 INFO Regime epoch 37/50 — tr=0.6145 va=1.2073 acc=0.843
2026-04-27 01:49:24,405 INFO Regime epoch 38/50 — tr=0.6148 va=1.2045 acc=0.842
2026-04-27 01:49:24,964 INFO Regime epoch 39/50 — tr=0.6147 va=1.2080 acc=0.844
2026-04-27 01:49:25,566 INFO Regime epoch 40/50 — tr=0.6147 va=1.2028 acc=0.840 per_class={'TRENDING': 0.822, 'RANGING': 0.777, 'CONSOLIDATING': 0.869, 'VOLATILE': 0.89}
2026-04-27 01:49:26,118 INFO Regime epoch 41/50 — tr=0.6148 va=1.2067 acc=0.841
2026-04-27 01:49:26,681 INFO Regime epoch 42/50 — tr=0.6148 va=1.2045 acc=0.842
2026-04-27 01:49:27,217 INFO Regime epoch 43/50 — tr=0.6145 va=1.2071 acc=0.841
2026-04-27 01:49:27,788 INFO Regime epoch 44/50 — tr=0.6147 va=1.2087 acc=0.844
2026-04-27 01:49:28,394 INFO Regime epoch 45/50 — tr=0.6146 va=1.2085 acc=0.842 per_class={'TRENDING': 0.826, 'RANGING': 0.781, 'CONSOLIDATING': 0.861, 'VOLATILE': 0.894}
2026-04-27 01:49:28,974 INFO Regime epoch 46/50 — tr=0.6148 va=1.2067 acc=0.842
2026-04-27 01:49:29,548 INFO Regime epoch 47/50 — tr=0.6146 va=1.2058 acc=0.842
2026-04-27 01:49:30,075 INFO Regime epoch 48/50 — tr=0.6147 va=1.2074 acc=0.841
2026-04-27 01:49:30,602 INFO Regime epoch 49/50 — tr=0.6147 va=1.2094 acc=0.842
2026-04-27 01:49:31,200 INFO Regime epoch 50/50 — tr=0.6146 va=1.2057 acc=0.844 per_class={'TRENDING': 0.833, 'RANGING': 0.778, 'CONSOLIDATING': 0.868, 'VOLATILE': 0.887}
2026-04-27 01:49:31,200 INFO Regime early stop at epoch 50 (no_improve=10)
2026-04-27 01:49:31,243 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 01:49:31,243 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 01:49:31,246 INFO Regime phase LTF train: 28.3s
2026-04-27 01:49:31,390 INFO Regime LTF complete: acc=0.840, n=401471 per_class={'TRENDING': 0.822, 'RANGING': 0.777, 'CONSOLIDATING': 0.869, 'VOLATILE': 0.89}
2026-04-27 01:49:31,394 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 01:49:31,955 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 01:49:31,960 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 9.355291913830783, 'RANGING': 6.096303199751476, 'CONSOLIDATING': 5.598885793871866, 'VOLATILE': 6.771351107094442}
2026-04-27 01:49:31,968 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 29965, 'mean': 2.149062259083323e-05, 'mean_over_std': 0.011155601434349563}, 'RANGING': {'n': 19623, 'mean': -3.254493608872846e-06, 'mean_over_std': -0.001761415548292494}, 'CONSOLIDATING': {'n': 10050, 'mean': -3.039756158146958e-06, 'mean_over_std': -0.001792488091618151}, 'VOLATILE': {'n': 14985, 'mean': 1.8102764458593193e-05, 'mean_over_std': 0.006974696291378742}}
2026-04-27 01:49:31,969 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 26145, 'mean': 2.874435605138053e-05, 'mean_over_std': 0.01595450632691917}, 'RANGING': {'n': 7858, 'mean': 1.7829776785521727e-05, 'mean_over_std': 0.011851561804467463}, 'CONSOLIDATING': {'n': 8543, 'mean': 7.845231961250924e-06, 'mean_over_std': 0.004915839521549481}, 'VOLATILE': {'n': 13636, 'mean': 5.480114267306668e-06, 'mean_over_std': 0.0020378119917389375}}
2026-04-27 01:49:31,969 INFO Regime retrain total: 232.7s (504761 samples)
2026-04-27 01:49:31,971 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-27 01:49:31,971 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 01:49:31,972 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 01:49:31,972 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-27 01:49:31,972 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-27 01:49:31,972 INFO Retrain complete. Total wall-clock: 232.7s
  DONE  Retrain regime [train-split retrain]
  START Retrain quality [train-split retrain]
2026-04-27 01:49:33,485 INFO retrain environment: KAGGLE
2026-04-27 01:49:35,214 INFO Device: CUDA (2 GPU(s))
2026-04-27 01:49:35,225 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 01:49:35,225 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 01:49:35,225 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 01:49:35,226 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 01:49:35,226 INFO Retrain data split: train
2026-04-27 01:49:35,227 INFO === QualityScorer retrain ===
2026-04-27 01:49:35,379 INFO NumExpr defaulting to 4 threads.
2026-04-27 01:49:35,584 INFO QualityScorer: CUDA available — using GPU
2026-04-27 01:49:35,797 INFO QualityScorer loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (device=cuda)
2026-04-27 01:49:35,800 INFO QualityScorer: skipped 82 journal records outside allowed splits ['live', 'paper', 'production', 'train']
2026-04-27 01:49:35,802 INFO Quality phase label creation: 0.0s (0 trades)
2026-04-27 01:49:35,802 INFO Retrain complete. Total wall-clock: 0.6s
  WARN  Retrain quality failed (exit 1) — continuing
  START Retrain rl [train-split retrain]
2026-04-27 01:49:36,552 INFO retrain environment: KAGGLE
2026-04-27 01:49:38,265 INFO Device: CUDA (2 GPU(s))
2026-04-27 01:49:38,277 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 01:49:38,277 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 01:49:38,277 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 01:49:38,277 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 01:49:38,278 INFO Retrain data split: train
2026-04-27 01:49:38,279 INFO === RLAgent (PPO) retrain ===
2026-04-27 01:49:38,286 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260427_014938
2026-04-27 01:49:42.063366: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777254582.331935   59852 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777254582.399503   59852 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777254582.953685   59852 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777254582.953727   59852 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777254582.953730   59852 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777254582.953733   59852 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-27 01:49:57,900 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-27 01:50:01,631 INFO RLAgent: PPO model loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-27 01:50:01,634 INFO RLAgent: skipped 82 journal records outside allowed splits ['live', 'paper', 'production', 'train']
2026-04-27 01:50:01,634 INFO RL phase episode loading: 0.0s (0 episodes)
2026-04-27 01:50:01,636 INFO RLAgent: skipped 82 journal records outside allowed splits ['live', 'paper', 'production', 'train']
2026-04-27 01:50:01,636 WARNING RLAgent.retrain: only 0 episodes — skipping
2026-04-27 01:50:01,637 INFO RL phase PPO train: 0.0s | total: 23.4s
2026-04-27 01:50:01,638 INFO Retrain complete. Total wall-clock: 23.4s
  WARN  Retrain rl failed (exit 1) — continuing

=== Round 3: Backtest on last 3yr (post-retrain evaluation) ===
  START Round 3 - Post-retrain backtest (last 3yr)
2026-04-27 01:50:03,983 INFO === STEP 6: BACKTEST (round3) ===
2026-04-27 01:50:03,984 INFO BT_WINDOW=round3 — post-retrain eval: 2022-08-05 → 2025-08-05 (last 3yr)
2026-04-27 01:50:03,984 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-04-27 01:50:03,985 INFO Round 3 — running backtest: 2022-08-05 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-27 01:53:12,483 WARNING ml_trader: portfolio drawdown 8.2% — halting all trading
2026-04-27 01:53:13,175 INFO Round 3 backtest — 67 trades | avg WR=40.3% | avg PF=2.19 | avg Sharpe=5.01
2026-04-27 01:53:13,175 INFO   ml_trader: 67 trades | WR=40.3% | PF=2.19 | Return=48.5% | DD=8.2% | Sharpe=5.01
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 67
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (67 rows)
  DONE  Round 3 - Post-retrain backtest (last 3yr)
  Saved Round 3 result → round3_summary.json

======================================================================
  BLIND BACKTEST PIPELINE COMPLETE
======================================================================
  Round 1 (val window)          trades=60  WR=28.3%  PF=0.998  Sharpe=-0.014
  Round 2 (blind test)          trades=22  WR=27.3%  PF=1.483  Sharpe=2.258
  Round 3 (last 3yr)            trades=67  WR=40.3%  PF=2.189  Sharpe=5.012


WARNING: GITHUB_TOKEN not set — skipping GitHub push
2026-04-27 01:53:13,466 INFO Round 3: wrote 67 journal entries (total in file: 149)