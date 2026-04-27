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
2026-04-26 23:56:49,280 INFO Loading feature-engineered data...
2026-04-26 23:56:49,902 INFO Loaded 221743 rows, 202 features
2026-04-26 23:56:49,904 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-26 23:56:49,906 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-26 23:56:49,906 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-26 23:56:49,906 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-26 23:56:49,906 INFO No leakage confirmed: train < val < test timestamps

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
2026-04-26 23:56:52,275 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-26 23:56:52,275 INFO --- Training regime ---
2026-04-26 23:56:52,275 INFO Running retrain --model regime
2026-04-26 23:56:52,454 INFO retrain environment: KAGGLE
2026-04-26 23:56:54,039 INFO Device: CUDA (2 GPU(s))
2026-04-26 23:56:54,050 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 23:56:54,050 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 23:56:54,050 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 23:56:54,053 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 23:56:54,053 INFO Retrain data split: train
2026-04-26 23:56:54,054 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-26 23:56:54,209 INFO NumExpr defaulting to 4 threads.
2026-04-26 23:56:54,419 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 23:56:54,420 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 23:56:54,420 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 23:56:54,420 INFO Regime phase macro_correlations: 0.0s
2026-04-26 23:56:54,420 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-26 23:56:54,461 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 23:56:54,462 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:56:54,503 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:56:54,531 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:56:54,564 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:56:54,578 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:56:54,603 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:56:54,617 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:56:54,640 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:56:54,655 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:56:54,676 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:56:54,690 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:56:54,709 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:56:54,721 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:56:54,739 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:56:54,753 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:56:54,773 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:56:54,787 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:56:54,810 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:56:54,825 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:56:54,848 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:56:54,867 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 23:56:54,905 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 23:56:56,022 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 23:57:16,905 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 23:57:16,907 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 22.0s
2026-04-26 23:57:16,908 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 23:57:26,843 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 23:57:26,848 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 9.9s
2026-04-26 23:57:26,848 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 23:57:34,693 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 23:57:34,694 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 7.8s
2026-04-26 23:57:34,694 INFO Regime phase GMM HTF total: 39.8s
2026-04-26 23:57:34,698 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 23:58:42,743 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 23:58:42,749 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 68.1s
2026-04-26 23:58:42,749 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 23:59:12,632 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 23:59:12,634 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 29.9s
2026-04-26 23:59:12,634 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 23:59:33,496 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 23:59:33,497 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 20.9s
2026-04-26 23:59:33,497 INFO Regime phase GMM LTF total: 118.8s
2026-04-26 23:59:33,594 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-26 23:59:33,595 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:33,597 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:33,598 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:33,598 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:33,599 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:33,601 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:33,601 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:33,602 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:33,603 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:33,604 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:33,606 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 23:59:33,725 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:33,766 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:33,767 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:33,767 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:33,775 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:33,776 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:34,179 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-26 23:59:34,180 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-26 23:59:34,351 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:34,382 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:34,383 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:34,384 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:34,391 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:34,392 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:34,758 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-26 23:59:34,760 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-26 23:59:34,945 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:34,981 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:34,981 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:34,982 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:34,990 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:34,991 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:35,359 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-26 23:59:35,360 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-26 23:59:35,524 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:35,559 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:35,560 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:35,560 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:35,568 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:35,569 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:35,929 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-26 23:59:35,931 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-26 23:59:36,108 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:36,143 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:36,143 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:36,144 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:36,151 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:36,152 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:36,507 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-26 23:59:36,508 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-26 23:59:36,674 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:36,707 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:36,708 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:36,708 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:36,716 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:36,717 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:37,081 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-26 23:59:37,082 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-26 23:59:37,235 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 23:59:37,262 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 23:59:37,263 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 23:59:37,263 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 23:59:37,270 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 23:59:37,271 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:37,626 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-26 23:59:37,627 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-26 23:59:37,798 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:37,830 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:37,830 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:37,831 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:37,838 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:37,839 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:38,198 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-26 23:59:38,199 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-26 23:59:38,364 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:38,395 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:38,396 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:38,396 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:38,404 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:38,405 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:38,769 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-26 23:59:38,770 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-26 23:59:38,942 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:38,989 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:38,989 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:38,990 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:39,001 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:39,002 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:39,358 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-26 23:59:39,359 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-26 23:59:39,618 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 23:59:39,677 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 23:59:39,678 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 23:59:39,678 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 23:59:39,688 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 23:59:39,689 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 23:59:40,444 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 23:59:40,446 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-26 23:59:40,598 INFO Regime phase HTF dataset build: 7.0s (103290 samples)
2026-04-26 23:59:40,600 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=38114 dropped=65176 classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728})
2026-04-26 23:59:40,609 INFO RegimeClassifier[mode=htf_bias]: 38114 samples, classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728}, device=cuda
2026-04-26 23:59:40,609 INFO RegimeClassifier: sample weights — mean=0.708  ambiguous(<0.4)=0.0%
2026-04-26 23:59:40,895 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-04-26 23:59:40,896 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 23:59:45,449 INFO Regime epoch  1/50 — tr=0.6960 va=2.1613 acc=0.351 per_class={'BIAS_UP': 0.697, 'BIAS_DOWN': 0.05, 'BIAS_NEUTRAL': 0.0}
2026-04-26 23:59:45,516 INFO Regime epoch  2/50 — tr=0.6917 va=2.1178 acc=0.449
2026-04-26 23:59:45,581 INFO Regime epoch  3/50 — tr=0.6838 va=2.0590 acc=0.574
2026-04-26 23:59:45,644 INFO Regime epoch  4/50 — tr=0.6707 va=1.9822 acc=0.735
2026-04-26 23:59:45,718 INFO Regime epoch  5/50 — tr=0.6516 va=1.8952 acc=0.820 per_class={'BIAS_UP': 0.892, 'BIAS_DOWN': 0.946, 'BIAS_NEUTRAL': 0.39}
2026-04-26 23:59:45,791 INFO Regime epoch  6/50 — tr=0.6275 va=1.8024 acc=0.863
2026-04-26 23:59:45,860 INFO Regime epoch  7/50 — tr=0.6040 va=1.6944 acc=0.888
2026-04-26 23:59:45,925 INFO Regime epoch  8/50 — tr=0.5809 va=1.5941 acc=0.906
2026-04-26 23:59:45,992 INFO Regime epoch  9/50 — tr=0.5588 va=1.5023 acc=0.916
2026-04-26 23:59:46,063 INFO Regime epoch 10/50 — tr=0.5449 va=1.4316 acc=0.924 per_class={'BIAS_UP': 0.988, 'BIAS_DOWN': 0.983, 'BIAS_NEUTRAL': 0.641}
2026-04-26 23:59:46,129 INFO Regime epoch 11/50 — tr=0.5338 va=1.3705 acc=0.931
2026-04-26 23:59:46,195 INFO Regime epoch 12/50 — tr=0.5252 va=1.3237 acc=0.936
2026-04-26 23:59:46,261 INFO Regime epoch 13/50 — tr=0.5181 va=1.2872 acc=0.940
2026-04-26 23:59:46,326 INFO Regime epoch 14/50 — tr=0.5136 va=1.2683 acc=0.942
2026-04-26 23:59:46,399 INFO Regime epoch 15/50 — tr=0.5101 va=1.2490 acc=0.945 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.995, 'BIAS_NEUTRAL': 0.706}
2026-04-26 23:59:46,469 INFO Regime epoch 16/50 — tr=0.5078 va=1.2284 acc=0.948
2026-04-26 23:59:46,537 INFO Regime epoch 17/50 — tr=0.5042 va=1.2159 acc=0.950
2026-04-26 23:59:46,603 INFO Regime epoch 18/50 — tr=0.5023 va=1.2049 acc=0.952
2026-04-26 23:59:46,673 INFO Regime epoch 19/50 — tr=0.4994 va=1.1959 acc=0.954
2026-04-26 23:59:46,747 INFO Regime epoch 20/50 — tr=0.4985 va=1.1837 acc=0.955 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.996, 'BIAS_NEUTRAL': 0.756}
2026-04-26 23:59:46,816 INFO Regime epoch 21/50 — tr=0.4957 va=1.1774 acc=0.955
2026-04-26 23:59:46,886 INFO Regime epoch 22/50 — tr=0.4945 va=1.1680 acc=0.959
2026-04-26 23:59:46,957 INFO Regime epoch 23/50 — tr=0.4932 va=1.1605 acc=0.960
2026-04-26 23:59:47,031 INFO Regime epoch 24/50 — tr=0.4920 va=1.1560 acc=0.962
2026-04-26 23:59:47,103 INFO Regime epoch 25/50 — tr=0.4905 va=1.1516 acc=0.964 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.997, 'BIAS_NEUTRAL': 0.805}
2026-04-26 23:59:47,169 INFO Regime epoch 26/50 — tr=0.4898 va=1.1481 acc=0.964
2026-04-26 23:59:47,238 INFO Regime epoch 27/50 — tr=0.4881 va=1.1451 acc=0.965
2026-04-26 23:59:47,310 INFO Regime epoch 28/50 — tr=0.4878 va=1.1382 acc=0.966
2026-04-26 23:59:47,385 INFO Regime epoch 29/50 — tr=0.4867 va=1.1354 acc=0.967
2026-04-26 23:59:47,459 INFO Regime epoch 30/50 — tr=0.4861 va=1.1327 acc=0.968 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 0.998, 'BIAS_NEUTRAL': 0.825}
2026-04-26 23:59:47,529 INFO Regime epoch 31/50 — tr=0.4854 va=1.1290 acc=0.968
2026-04-26 23:59:47,594 INFO Regime epoch 32/50 — tr=0.4848 va=1.1279 acc=0.969
2026-04-26 23:59:47,660 INFO Regime epoch 33/50 — tr=0.4843 va=1.1258 acc=0.969
2026-04-26 23:59:47,730 INFO Regime epoch 34/50 — tr=0.4834 va=1.1218 acc=0.969
2026-04-26 23:59:47,799 INFO Regime epoch 35/50 — tr=0.4829 va=1.1197 acc=0.970 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 0.998, 'BIAS_NEUTRAL': 0.837}
2026-04-26 23:59:47,864 INFO Regime epoch 36/50 — tr=0.4829 va=1.1188 acc=0.971
2026-04-26 23:59:47,929 INFO Regime epoch 37/50 — tr=0.4825 va=1.1129 acc=0.972
2026-04-26 23:59:48,000 INFO Regime epoch 38/50 — tr=0.4823 va=1.1132 acc=0.972
2026-04-26 23:59:48,068 INFO Regime epoch 39/50 — tr=0.4817 va=1.1124 acc=0.972
2026-04-26 23:59:48,137 INFO Regime epoch 40/50 — tr=0.4816 va=1.1075 acc=0.973 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 0.998, 'BIAS_NEUTRAL': 0.853}
2026-04-26 23:59:48,206 INFO Regime epoch 41/50 — tr=0.4813 va=1.1082 acc=0.972
2026-04-26 23:59:48,272 INFO Regime epoch 42/50 — tr=0.4810 va=1.1074 acc=0.972
2026-04-26 23:59:48,338 INFO Regime epoch 43/50 — tr=0.4817 va=1.1089 acc=0.973
2026-04-26 23:59:48,410 INFO Regime epoch 44/50 — tr=0.4815 va=1.1076 acc=0.973
2026-04-26 23:59:48,484 INFO Regime epoch 45/50 — tr=0.4813 va=1.1092 acc=0.972 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 0.998, 'BIAS_NEUTRAL': 0.85}
2026-04-26 23:59:48,557 INFO Regime epoch 46/50 — tr=0.4814 va=1.1111 acc=0.972
2026-04-26 23:59:48,633 INFO Regime epoch 47/50 — tr=0.4813 va=1.1092 acc=0.973
2026-04-26 23:59:48,701 INFO Regime epoch 48/50 — tr=0.4809 va=1.1064 acc=0.973
2026-04-26 23:59:48,767 INFO Regime epoch 49/50 — tr=0.4808 va=1.1070 acc=0.973
2026-04-26 23:59:48,842 INFO Regime epoch 50/50 — tr=0.4815 va=1.1089 acc=0.973 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 0.998, 'BIAS_NEUTRAL': 0.852}
2026-04-26 23:59:48,852 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 23:59:48,852 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 23:59:48,853 INFO Regime phase HTF train: 8.3s
2026-04-26 23:59:48,996 INFO Regime HTF complete: acc=0.973, n=103290 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 0.998, 'BIAS_NEUTRAL': 0.854}
2026-04-26 23:59:48,998 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 23:59:49,159 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 23:59:49,168 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 11.41578947368421, 'BIAS_DOWN': 10.635761589403973, 'BIAS_NEUTRAL': 17.960468521229867}
2026-04-26 23:59:49,172 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 4338, 'mean': 0.00011110466582543088, 'mean_over_std': 0.027632581210502927}, 'BIAS_DOWN': {'n': 3212, 'mean': 6.063430222660252e-05, 'mean_over_std': 0.013898489590377157}, 'BIAS_NEUTRAL': {'n': 12266, 'mean': 1.2151554902814046e-05, 'mean_over_std': 0.0031957000513162357}}
2026-04-26 23:59:49,172 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 3651, 'mean': 0.0002852242108836959, 'mean_over_std': 0.07015017232491755}, 'BIAS_DOWN': {'n': 2644, 'mean': -0.00014900749618981248, 'mean_over_std': -0.03333037041298237}, 'BIAS_NEUTRAL': {'n': 1376, 'mean': 9.595797941249082e-05, 'mean_over_std': 0.02492411099521803}}
2026-04-26 23:59:49,173 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-26 23:59:49,174 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:49,176 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:49,178 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:49,180 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:49,181 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:49,183 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:49,184 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:49,186 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:49,188 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:49,189 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:49,192 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 23:59:49,204 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:49,207 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:49,208 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:49,208 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:49,209 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:49,211 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:49,812 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-26 23:59:49,815 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-26 23:59:49,946 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:49,948 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:49,949 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:49,949 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:49,949 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:49,951 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:50,518 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-26 23:59:50,520 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-26 23:59:50,650 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:50,652 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:50,653 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:50,653 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:50,654 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:50,656 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:51,231 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-26 23:59:51,233 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-26 23:59:51,365 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:51,368 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:51,368 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:51,369 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:51,369 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:51,371 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:51,941 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-26 23:59:51,944 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-26 23:59:52,072 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:52,074 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:52,075 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:52,075 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:52,075 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:52,077 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:52,648 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-26 23:59:52,651 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-26 23:59:52,782 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:52,785 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:52,785 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:52,786 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:52,786 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:52,788 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:53,375 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-26 23:59:53,378 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-26 23:59:53,510 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 23:59:53,512 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 23:59:53,512 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 23:59:53,513 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 23:59:53,513 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 23:59:53,515 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:54,101 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-26 23:59:54,104 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-26 23:59:54,231 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:54,233 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:54,234 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:54,234 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:54,235 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:54,237 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:54,802 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-26 23:59:54,805 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-26 23:59:54,936 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:54,938 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:54,939 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:54,939 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:54,939 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:54,941 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:55,512 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-26 23:59:55,515 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-26 23:59:55,649 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:55,651 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:55,652 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:55,652 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:55,653 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 23:59:55,654 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 23:59:56,230 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-26 23:59:56,233 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-26 23:59:56,376 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 23:59:56,380 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 23:59:56,381 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 23:59:56,382 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 23:59:56,382 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 23:59:56,386 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 23:59:57,594 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 23:59:57,599 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-26 23:59:57,882 INFO Regime phase LTF dataset build: 8.7s (401471 samples)
2026-04-26 23:59:57,885 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=299185 dropped=102286 classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127})
2026-04-26 23:59:57,949 INFO RegimeClassifier[mode=ltf_behaviour]: 299185 samples, classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127}, device=cuda
2026-04-26 23:59:57,949 INFO RegimeClassifier: sample weights — mean=0.693  ambiguous(<0.4)=0.0%
2026-04-26 23:59:57,951 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-04-26 23:59:57,952 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 23:59:58,516 INFO Regime epoch  1/50 — tr=0.8687 va=2.0045 acc=0.383 per_class={'TRENDING': 0.204, 'RANGING': 0.361, 'CONSOLIDATING': 0.282, 'VOLATILE': 0.803}
2026-04-26 23:59:59,053 INFO Regime epoch  2/50 — tr=0.8407 va=1.9087 acc=0.451
2026-04-26 23:59:59,550 INFO Regime epoch  3/50 — tr=0.8021 va=1.8044 acc=0.524
2026-04-27 00:00:00,029 INFO Regime epoch  4/50 — tr=0.7628 va=1.6974 acc=0.574
2026-04-27 00:00:00,548 INFO Regime epoch  5/50 — tr=0.7324 va=1.6022 acc=0.611 per_class={'TRENDING': 0.435, 'RANGING': 0.469, 'CONSOLIDATING': 0.745, 'VOLATILE': 0.945}
2026-04-27 00:00:01,065 INFO Regime epoch  6/50 — tr=0.7110 va=1.5313 acc=0.644
2026-04-27 00:00:01,562 INFO Regime epoch  7/50 — tr=0.6957 va=1.4808 acc=0.668
2026-04-27 00:00:02,053 INFO Regime epoch  8/50 — tr=0.6841 va=1.4535 acc=0.688
2026-04-27 00:00:02,550 INFO Regime epoch  9/50 — tr=0.6752 va=1.4284 acc=0.705
2026-04-27 00:00:03,091 INFO Regime epoch 10/50 — tr=0.6682 va=1.4097 acc=0.722 per_class={'TRENDING': 0.613, 'RANGING': 0.709, 'CONSOLIDATING': 0.719, 'VOLATILE': 0.94}
2026-04-27 00:00:03,605 INFO Regime epoch 11/50 — tr=0.6625 va=1.3875 acc=0.738
2026-04-27 00:00:04,098 INFO Regime epoch 12/50 — tr=0.6582 va=1.3665 acc=0.748
2026-04-27 00:00:04,606 INFO Regime epoch 13/50 — tr=0.6543 va=1.3550 acc=0.759
2026-04-27 00:00:05,084 INFO Regime epoch 14/50 — tr=0.6510 va=1.3421 acc=0.766
2026-04-27 00:00:05,619 INFO Regime epoch 15/50 — tr=0.6485 va=1.3343 acc=0.774 per_class={'TRENDING': 0.72, 'RANGING': 0.736, 'CONSOLIDATING': 0.734, 'VOLATILE': 0.922}
2026-04-27 00:00:06,104 INFO Regime epoch 16/50 — tr=0.6463 va=1.3281 acc=0.778
2026-04-27 00:00:06,596 INFO Regime epoch 17/50 — tr=0.6441 va=1.3172 acc=0.784
2026-04-27 00:00:07,069 INFO Regime epoch 18/50 — tr=0.6426 va=1.3114 acc=0.787
2026-04-27 00:00:07,552 INFO Regime epoch 19/50 — tr=0.6408 va=1.3069 acc=0.791
2026-04-27 00:00:08,100 INFO Regime epoch 20/50 — tr=0.6393 va=1.2945 acc=0.794 per_class={'TRENDING': 0.76, 'RANGING': 0.759, 'CONSOLIDATING': 0.748, 'VOLATILE': 0.909}
2026-04-27 00:00:08,583 INFO Regime epoch 21/50 — tr=0.6381 va=1.2916 acc=0.796
2026-04-27 00:00:09,103 INFO Regime epoch 22/50 — tr=0.6367 va=1.2865 acc=0.801
2026-04-27 00:00:09,603 INFO Regime epoch 23/50 — tr=0.6357 va=1.2819 acc=0.802
2026-04-27 00:00:10,121 INFO Regime epoch 24/50 — tr=0.6346 va=1.2780 acc=0.802
2026-04-27 00:00:10,651 INFO Regime epoch 25/50 — tr=0.6336 va=1.2749 acc=0.804 per_class={'TRENDING': 0.773, 'RANGING': 0.762, 'CONSOLIDATING': 0.769, 'VOLATILE': 0.911}
2026-04-27 00:00:11,134 INFO Regime epoch 26/50 — tr=0.6327 va=1.2726 acc=0.808
2026-04-27 00:00:11,623 INFO Regime epoch 27/50 — tr=0.6318 va=1.2699 acc=0.809
2026-04-27 00:00:12,122 INFO Regime epoch 28/50 — tr=0.6310 va=1.2658 acc=0.812
2026-04-27 00:00:12,620 INFO Regime epoch 29/50 — tr=0.6303 va=1.2661 acc=0.815
2026-04-27 00:00:13,162 INFO Regime epoch 30/50 — tr=0.6297 va=1.2606 acc=0.814 per_class={'TRENDING': 0.786, 'RANGING': 0.764, 'CONSOLIDATING': 0.796, 'VOLATILE': 0.908}
2026-04-27 00:00:13,656 INFO Regime epoch 31/50 — tr=0.6289 va=1.2631 acc=0.818
2026-04-27 00:00:14,159 INFO Regime epoch 32/50 — tr=0.6283 va=1.2582 acc=0.818
2026-04-27 00:00:14,658 INFO Regime epoch 33/50 — tr=0.6280 va=1.2581 acc=0.820
2026-04-27 00:00:15,142 INFO Regime epoch 34/50 — tr=0.6276 va=1.2560 acc=0.820
2026-04-27 00:00:15,679 INFO Regime epoch 35/50 — tr=0.6269 va=1.2521 acc=0.820 per_class={'TRENDING': 0.794, 'RANGING': 0.765, 'CONSOLIDATING': 0.818, 'VOLATILE': 0.904}
2026-04-27 00:00:16,183 INFO Regime epoch 36/50 — tr=0.6267 va=1.2527 acc=0.824
2026-04-27 00:00:16,674 INFO Regime epoch 37/50 — tr=0.6260 va=1.2543 acc=0.823
2026-04-27 00:00:17,177 INFO Regime epoch 38/50 — tr=0.6263 va=1.2482 acc=0.821
2026-04-27 00:00:17,692 INFO Regime epoch 39/50 — tr=0.6261 va=1.2494 acc=0.823
2026-04-27 00:00:18,222 INFO Regime epoch 40/50 — tr=0.6257 va=1.2489 acc=0.825 per_class={'TRENDING': 0.799, 'RANGING': 0.765, 'CONSOLIDATING': 0.829, 'VOLATILE': 0.904}
2026-04-27 00:00:18,717 INFO Regime epoch 41/50 — tr=0.6255 va=1.2487 acc=0.826
2026-04-27 00:00:19,238 INFO Regime epoch 42/50 — tr=0.6252 va=1.2463 acc=0.826
2026-04-27 00:00:19,717 INFO Regime epoch 43/50 — tr=0.6251 va=1.2447 acc=0.825
2026-04-27 00:00:20,193 INFO Regime epoch 44/50 — tr=0.6254 va=1.2490 acc=0.826
2026-04-27 00:00:20,720 INFO Regime epoch 45/50 — tr=0.6253 va=1.2451 acc=0.826 per_class={'TRENDING': 0.802, 'RANGING': 0.766, 'CONSOLIDATING': 0.831, 'VOLATILE': 0.904}
2026-04-27 00:00:21,217 INFO Regime epoch 46/50 — tr=0.6253 va=1.2444 acc=0.828
2026-04-27 00:00:21,690 INFO Regime epoch 47/50 — tr=0.6254 va=1.2457 acc=0.825
2026-04-27 00:00:22,167 INFO Regime epoch 48/50 — tr=0.6250 va=1.2439 acc=0.827
2026-04-27 00:00:22,650 INFO Regime epoch 49/50 — tr=0.6249 va=1.2448 acc=0.825
2026-04-27 00:00:23,205 INFO Regime epoch 50/50 — tr=0.6248 va=1.2456 acc=0.826 per_class={'TRENDING': 0.802, 'RANGING': 0.766, 'CONSOLIDATING': 0.832, 'VOLATILE': 0.903}
2026-04-27 00:00:23,245 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 00:00:23,245 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 00:00:23,247 INFO Regime phase LTF train: 25.4s
2026-04-27 00:00:23,373 INFO Regime LTF complete: acc=0.827, n=401471 per_class={'TRENDING': 0.803, 'RANGING': 0.767, 'CONSOLIDATING': 0.84, 'VOLATILE': 0.899}
2026-04-27 00:00:23,377 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 00:00:23,863 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 00:00:23,867 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 9.355291913830783, 'RANGING': 6.096303199751476, 'CONSOLIDATING': 5.598885793871866, 'VOLATILE': 6.771351107094442}
2026-04-27 00:00:23,875 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 29965, 'mean': 2.149062259083323e-05, 'mean_over_std': 0.011155601434349563}, 'RANGING': {'n': 19623, 'mean': -3.254493608872846e-06, 'mean_over_std': -0.001761415548292494}, 'CONSOLIDATING': {'n': 10050, 'mean': -3.039756158146958e-06, 'mean_over_std': -0.001792488091618151}, 'VOLATILE': {'n': 14985, 'mean': 1.8102764458593193e-05, 'mean_over_std': 0.006974696291378742}}
2026-04-27 00:00:23,875 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 26145, 'mean': 2.874435605138053e-05, 'mean_over_std': 0.01595450632691917}, 'RANGING': {'n': 7858, 'mean': 1.7829776785521727e-05, 'mean_over_std': 0.011851561804467463}, 'CONSOLIDATING': {'n': 8543, 'mean': 7.845231961250924e-06, 'mean_over_std': 0.004915839521549481}, 'VOLATILE': {'n': 13636, 'mean': 5.480114267306668e-06, 'mean_over_std': 0.0020378119917389375}}
2026-04-27 00:00:23,876 INFO Regime retrain total: 209.8s (504761 samples)
2026-04-27 00:00:23,890 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-27 00:00:23,891 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 00:00:23,891 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 00:00:23,891 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-27 00:00:23,891 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-27 00:00:23,891 INFO Retrain complete. Total wall-clock: 209.8s
2026-04-27 00:00:26,303 INFO Model regime: SUCCESS
2026-04-27 00:00:26,303 INFO --- Training gru ---
2026-04-27 00:00:26,304 INFO Running retrain --model gru
2026-04-27 00:00:26,570 INFO retrain environment: KAGGLE
2026-04-27 00:00:28,176 INFO Device: CUDA (2 GPU(s))
2026-04-27 00:00:28,187 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 00:00:28,187 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 00:00:28,188 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 00:00:28,188 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 00:00:28,188 INFO Retrain data split: train
2026-04-27 00:00:28,189 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-27 00:00:28,335 INFO NumExpr defaulting to 4 threads.
2026-04-27 00:00:28,529 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-27 00:00:28,529 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 00:00:28,529 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 00:00:28,530 INFO GRU phase macro_correlations: 0.0s
2026-04-27 00:00:28,530 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-27 00:00:28,530 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260427_000028
2026-04-27 00:00:28,533 WARNING WeightsManifest: no manifest at /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json — assuming compatible (run a full retrain to generate one)
2026-04-27 00:00:28,693 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:28,713 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:28,726 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:28,734 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:28,735 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-27 00:00:28,735 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 00:00:28,736 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 00:00:28,736 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-27 00:00:28,737 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:28,812 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-27 00:00:28,813 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:29,077 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-27 00:00:29,105 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:29,370 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:29,491 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:29,582 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:29,774 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:29,792 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:29,805 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:29,812 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:29,813 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:29,887 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-27 00:00:29,889 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:30,124 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-27 00:00:30,141 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:30,415 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:30,538 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:30,629 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:30,812 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:30,831 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:30,846 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:30,853 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:30,854 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:30,929 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-27 00:00:30,931 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:31,159 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-27 00:00:31,174 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:31,433 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:31,560 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:31,653 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:31,833 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:31,851 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:31,866 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:31,873 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:31,874 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:31,951 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-27 00:00:31,952 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:32,180 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-27 00:00:32,203 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:32,465 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:32,590 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:32,682 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:32,856 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:32,874 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:32,889 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:32,896 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:32,897 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:32,974 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-27 00:00:32,976 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:33,255 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-27 00:00:33,270 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:33,530 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:33,652 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:33,743 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:33,921 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:33,939 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:33,952 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:33,959 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:33,959 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:34,030 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-27 00:00:34,031 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:34,250 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-27 00:00:34,264 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:34,527 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:34,650 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:34,747 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:34,904 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 00:00:34,921 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 00:00:34,934 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 00:00:34,940 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 00:00:34,941 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:35,015 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-27 00:00:35,016 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:35,240 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-27 00:00:35,252 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:35,508 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:35,626 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:35,719 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:35,894 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:35,911 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:35,925 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:35,932 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:35,933 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:36,009 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-27 00:00:36,011 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:36,237 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-27 00:00:36,255 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:36,512 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:36,634 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:36,725 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:36,898 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:36,915 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:36,928 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:36,935 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:36,936 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:37,009 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-27 00:00:37,010 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:37,237 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-27 00:00:37,251 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:37,511 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:37,634 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:37,725 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:37,899 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:37,918 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:37,933 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:37,940 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:00:37,941 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:38,015 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-27 00:00:38,017 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:38,243 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-27 00:00:38,262 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:38,545 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:38,674 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:38,767 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:00:39,056 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 00:00:39,081 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 00:00:39,097 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 00:00:39,107 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 00:00:39,108 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 00:00:39,266 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 00:00:39,269 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 00:00:39,768 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 00:00:39,815 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-27 00:00:40,311 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-27 00:00:40,506 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 00:00:40,634 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 00:00:40,745 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-27 00:00:41,003 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-04-27 00:00:41,003 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-04-27 00:00:41,004 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-27 00:00:41,004 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-27 00:01:26,743 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-27 00:01:26,743 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-27 00:01:28,062 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-27 00:01:32,145 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=18, min_epochs=22)
2026-04-27 00:01:45,734 INFO train_multi TF=ALL epoch 1/50 train=0.8518 val=0.8449
2026-04-27 00:01:45,743 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 00:01:45,743 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 00:01:45,743 INFO train_multi TF=ALL: new best val=0.8449 — saved
2026-04-27 00:01:57,247 INFO train_multi TF=ALL epoch 2/50 train=0.8277 val=0.7844
2026-04-27 00:01:57,252 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 00:01:57,252 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 00:01:57,252 INFO train_multi TF=ALL: new best val=0.7844 — saved
2026-04-27 00:02:08,807 INFO train_multi TF=ALL epoch 3/50 train=0.7122 val=0.6879
2026-04-27 00:02:08,812 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 00:02:08,812 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 00:02:08,812 INFO train_multi TF=ALL: new best val=0.6879 — saved
2026-04-27 00:02:20,420 INFO train_multi TF=ALL epoch 4/50 train=0.6904 val=0.6879
2026-04-27 00:02:31,988 INFO train_multi TF=ALL epoch 5/50 train=0.6897 val=0.6879
2026-04-27 00:02:43,512 INFO train_multi TF=ALL epoch 6/50 train=0.6893 val=0.6885
2026-04-27 00:02:55,184 INFO train_multi TF=ALL epoch 7/50 train=0.6890 val=0.6887
2026-04-27 00:03:06,808 INFO train_multi TF=ALL epoch 8/50 train=0.6887 val=0.6882
2026-04-27 00:03:18,497 INFO train_multi TF=ALL epoch 9/50 train=0.6882 val=0.6873
2026-04-27 00:03:18,502 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 00:03:18,502 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 00:03:18,502 INFO train_multi TF=ALL: new best val=0.6873 — saved
2026-04-27 00:03:30,159 INFO train_multi TF=ALL epoch 10/50 train=0.6869 val=0.6863
2026-04-27 00:03:30,163 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 00:03:30,163 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 00:03:30,164 INFO train_multi TF=ALL: new best val=0.6863 — saved
2026-04-27 00:03:41,775 INFO train_multi TF=ALL epoch 11/50 train=0.6835 val=0.6810
2026-04-27 00:03:41,779 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 00:03:41,779 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 00:03:41,779 INFO train_multi TF=ALL: new best val=0.6810 — saved
2026-04-27 00:03:53,433 INFO train_multi TF=ALL epoch 12/50 train=0.6762 val=0.6724
2026-04-27 00:03:53,437 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 00:03:53,437 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 00:03:53,437 INFO train_multi TF=ALL: new best val=0.6724 — saved
2026-04-27 00:04:05,007 INFO train_multi TF=ALL epoch 13/50 train=0.6672 val=0.6602
2026-04-27 00:04:05,011 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 00:04:05,011 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 00:04:05,011 INFO train_multi TF=ALL: new best val=0.6602 — saved
2026-04-27 00:04:16,611 INFO train_multi TF=ALL epoch 14/50 train=0.6561 val=0.6487
2026-04-27 00:04:16,615 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 00:04:16,615 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 00:04:16,615 INFO train_multi TF=ALL: new best val=0.6487 — saved
2026-04-27 00:04:28,211 INFO train_multi TF=ALL epoch 15/50 train=0.6461 val=0.6368
2026-04-27 00:04:28,215 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 00:04:28,215 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 00:04:28,216 INFO train_multi TF=ALL: new best val=0.6368 — saved
2026-04-27 00:04:39,901 INFO train_multi TF=ALL epoch 16/50 train=0.6384 val=0.6311
2026-04-27 00:04:39,906 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 00:04:39,906 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 00:04:39,906 INFO train_multi TF=ALL: new best val=0.6311 — saved
2026-04-27 00:04:51,530 INFO train_multi TF=ALL epoch 17/50 train=0.6329 val=0.6259
2026-04-27 00:04:51,535 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 00:04:51,535 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 00:04:51,535 INFO train_multi TF=ALL: new best val=0.6259 — saved
2026-04-27 00:05:03,091 INFO train_multi TF=ALL epoch 18/50 train=0.6290 val=0.6255
2026-04-27 00:05:03,095 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 00:05:03,095 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 00:05:03,095 INFO train_multi TF=ALL: new best val=0.6255 — saved
2026-04-27 00:05:14,666 INFO train_multi TF=ALL epoch 19/50 train=0.6246 val=0.6205
2026-04-27 00:05:14,671 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 00:05:14,671 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 00:05:14,671 INFO train_multi TF=ALL: new best val=0.6205 — saved
2026-04-27 00:05:26,218 INFO train_multi TF=ALL epoch 20/50 train=0.6214 val=0.6190
2026-04-27 00:05:26,222 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 00:05:26,222 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 00:05:26,222 INFO train_multi TF=ALL: new best val=0.6190 — saved
2026-04-27 00:05:37,798 INFO train_multi TF=ALL epoch 21/50 train=0.6190 val=0.6164
2026-04-27 00:05:37,802 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 00:05:37,802 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 00:05:37,803 INFO train_multi TF=ALL: new best val=0.6164 — saved
2026-04-27 00:05:49,572 INFO train_multi TF=ALL epoch 22/50 train=0.6171 val=0.6167
2026-04-27 00:06:01,250 INFO train_multi TF=ALL epoch 23/50 train=0.6150 val=0.6139
2026-04-27 00:06:01,254 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 00:06:01,254 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 00:06:01,254 INFO train_multi TF=ALL: new best val=0.6139 — saved
2026-04-27 00:06:12,821 INFO train_multi TF=ALL epoch 24/50 train=0.6131 val=0.6142
2026-04-27 00:06:24,410 INFO train_multi TF=ALL epoch 25/50 train=0.6113 val=0.6164
2026-04-27 00:06:36,037 INFO train_multi TF=ALL epoch 26/50 train=0.6097 val=0.6128
2026-04-27 00:06:36,041 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 00:06:36,042 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 00:06:36,042 INFO train_multi TF=ALL: new best val=0.6128 — saved
2026-04-27 00:06:47,610 INFO train_multi TF=ALL epoch 27/50 train=0.6074 val=0.6132
2026-04-27 00:06:59,174 INFO train_multi TF=ALL epoch 28/50 train=0.6065 val=0.6149
2026-04-27 00:07:10,783 INFO train_multi TF=ALL epoch 29/50 train=0.6048 val=0.6162
2026-04-27 00:07:22,336 INFO train_multi TF=ALL epoch 30/50 train=0.6038 val=0.6133
2026-04-27 00:07:33,895 INFO train_multi TF=ALL epoch 31/50 train=0.6020 val=0.6103
2026-04-27 00:07:33,899 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 00:07:33,899 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 00:07:33,899 INFO train_multi TF=ALL: new best val=0.6103 — saved
2026-04-27 00:07:45,560 INFO train_multi TF=ALL epoch 32/50 train=0.6006 val=0.6146
2026-04-27 00:07:57,145 INFO train_multi TF=ALL epoch 33/50 train=0.5988 val=0.6105
2026-04-27 00:08:08,710 INFO train_multi TF=ALL epoch 34/50 train=0.5981 val=0.6126
2026-04-27 00:08:20,368 INFO train_multi TF=ALL epoch 35/50 train=0.5968 val=0.6155
2026-04-27 00:08:31,916 INFO train_multi TF=ALL epoch 36/50 train=0.5949 val=0.6131
2026-04-27 00:08:43,511 INFO train_multi TF=ALL epoch 37/50 train=0.5939 val=0.6110
2026-04-27 00:08:55,010 INFO train_multi TF=ALL epoch 38/50 train=0.5930 val=0.6102
2026-04-27 00:08:55,015 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 00:08:55,015 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 00:08:55,015 INFO train_multi TF=ALL: new best val=0.6102 — saved
2026-04-27 00:09:06,575 INFO train_multi TF=ALL epoch 39/50 train=0.5913 val=0.6095
2026-04-27 00:09:06,580 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 00:09:06,580 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 00:09:06,580 INFO train_multi TF=ALL: new best val=0.6095 — saved
2026-04-27 00:09:18,193 INFO train_multi TF=ALL epoch 40/50 train=0.5898 val=0.6108
2026-04-27 00:09:29,807 INFO train_multi TF=ALL epoch 41/50 train=0.5895 val=0.6115
2026-04-27 00:09:41,395 INFO train_multi TF=ALL epoch 42/50 train=0.5880 val=0.6144
2026-04-27 00:09:53,012 INFO train_multi TF=ALL epoch 43/50 train=0.5864 val=0.6128
2026-04-27 00:10:04,469 INFO train_multi TF=ALL epoch 44/50 train=0.5849 val=0.6143
2026-04-27 00:10:16,024 INFO train_multi TF=ALL epoch 45/50 train=0.5838 val=0.6125
2026-04-27 00:10:27,550 INFO train_multi TF=ALL epoch 46/50 train=0.5828 val=0.6139
2026-04-27 00:10:39,145 INFO train_multi TF=ALL epoch 47/50 train=0.5818 val=0.6114
2026-04-27 00:10:50,755 INFO train_multi TF=ALL epoch 48/50 train=0.5805 val=0.6139
2026-04-27 00:11:02,419 INFO train_multi TF=ALL epoch 49/50 train=0.5793 val=0.6120
2026-04-27 00:11:13,935 INFO train_multi TF=ALL epoch 50/50 train=0.5779 val=0.6180
2026-04-27 00:11:14,068 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-27 00:11:14,068 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-27 00:11:14,068 INFO Retrain complete. Total wall-clock: 645.9s
2026-04-27 00:11:15,938 INFO Model gru: SUCCESS
2026-04-27 00:11:15,939 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 00:11:15,939 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 00:11:15,939 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 00:11:15,939 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-04-27 00:11:15,939 INFO   [DEFERRED] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip (expected after Round 1)
2026-04-27 00:11:15,939 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-04-27 00:11:15,939 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer', 'rl_ppo']
2026-04-27 00:11:15,940 INFO Saved 5 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  START Round 1 - Backtest (val)
2026-04-27 00:11:16,630 INFO === STEP 6: BACKTEST (round1) ===
2026-04-27 00:11:16,631 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-04-27 00:11:16,631 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-27 00:11:16,631 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
2026-04-27 00:11:18,977 WARNING QualityScorer unavailable (weights missing or load failed)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260427_001118.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)             3991  59.3%   3.24 3333.9% 59.3% 28.2%   7.8%    8.54
  gate_diagnostics: bars=468696 no_signal=71297 quality_block=0 session_skip=317895 density=39604 pm_reject=0 daily_skip=0 cooldown=35909 daily_halt_events=0 enforce_daily_halt=False
  no_signal_reasons: htf_bias_conflict=43338, weak_gru_direction=27521, trend_pullback_conflict=412, range_side_conflict=26

Calibration Summary:
  all          [OK] Calibration usable but not strictly monotonic: 1/5 pairs violated.
  ml_trader    [OK] Calibration usable but not strictly monotonic: 1/5 pairs violated.
2026-04-27 00:13:31,216 INFO Round 1 backtest — 3991 trades | avg WR=59.3% | avg PF=3.23 | avg Sharpe=8.54
2026-04-27 00:13:31,216 INFO   ml_trader: 3991 trades | WR=59.3% | PF=3.23 | Return=3333.9% | DD=7.8% | Sharpe=8.54
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 3991
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (3991 rows)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  AUDUSD        401 trades     1 days  401.00/day  [OVERTRADE]
  EURGBP        404 trades     1 days  404.00/day  [OVERTRADE]
  EURJPY        393 trades     1 days  393.00/day  [OVERTRADE]
  EURUSD        402 trades     1 days  402.00/day  [OVERTRADE]
  GBPJPY        397 trades     1 days  397.00/day  [OVERTRADE]
  GBPUSD        410 trades     1 days  410.00/day  [OVERTRADE]
  NZDUSD          6 trades     1 days   6.00/day  [OVERTRADE]
  USDCAD        393 trades     1 days  393.00/day  [OVERTRADE]
  USDCHF        413 trades     1 days  413.00/day  [OVERTRADE]
  USDJPY        379 trades     1 days  379.00/day  [OVERTRADE]
  XAUUSD        393 trades     1 days  393.00/day  [OVERTRADE]
  ⚠  AUDUSD: 401.00/day (>1.5)
  ⚠  EURGBP: 404.00/day (>1.5)
  ⚠  EURJPY: 393.00/day (>1.5)
  ⚠  EURUSD: 402.00/day (>1.5)
  ⚠  GBPJPY: 397.00/day (>1.5)
  ⚠  GBPUSD: 410.00/day (>1.5)
  ⚠  NZDUSD: 6.00/day (>1.5)
  ⚠  USDCAD: 393.00/day (>1.5)
  ⚠  USDCHF: 413.00/day (>1.5)
  ⚠  USDJPY: 379.00/day (>1.5)
  ⚠  XAUUSD: 393.00/day (>1.5)

──────────────────────────────────────────────────────────────
CHECK 2: REGIME DISTRIBUTION  (% of trades, WR, avgEV)
──────────────────────────────────────────────────────────────
  BIAS_DOWN           861 trades   21.6%  WR=58.7%  avgEV=0.000
  BIAS_NEUTRAL       1630 trades
2026-04-27 00:13:32,817 INFO Round 1: wrote 3991 journal entries (total in file: 3991)
  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 3991 entries

=== Round 1 → Retrain Quality + RL ===
  START Round 1 - Quality+RL retrain
2026-04-27 00:13:33,074 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-27 00:13:33,078 INFO Journal entries: 3991
2026-04-27 00:13:33,078 INFO --- Training quality ---
2026-04-27 00:13:33,078 INFO Running retrain --model quality
2026-04-27 00:13:33,253 INFO retrain environment: KAGGLE
2026-04-27 00:13:34,848 INFO Device: CUDA (2 GPU(s))
2026-04-27 00:13:34,859 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 00:13:34,860 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 00:13:34,860 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 00:13:34,860 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 00:13:34,860 INFO Retrain data split: train
2026-04-27 00:13:34,861 INFO === QualityScorer retrain ===
2026-04-27 00:13:35,004 INFO NumExpr defaulting to 4 threads.
2026-04-27 00:13:35,196 INFO QualityScorer: CUDA available — using GPU
2026-04-27 00:13:35,440 INFO Quality phase label creation: 0.2s (3991 trades)
2026-04-27 00:13:35,675 INFO QualityScorer: 3991 samples, EV stats={'mean': 0.4533344805240631, 'std': 1.299851655960083, 'n_pos': 2367, 'n_neg': 1624}, device=cuda
2026-04-27 00:13:35,675 INFO QualityScorer: normalised win labels by median_win=0.960 — EV range now [-1, +3]
2026-04-27 00:13:35,877 INFO QualityScorer: DataParallel across 2 GPUs
2026-04-27 00:13:35,878 INFO QualityScorer: cold start
2026-04-27 00:13:35,878 INFO QualityScorer: pos_weight=0.68 (n_pos=1898 n_neg=1294)
2026-04-27 00:13:38,251 INFO Quality epoch   1/100 — va_huber=0.6224
2026-04-27 00:13:38,346 INFO Quality epoch   2/100 — va_huber=0.6211
2026-04-27 00:13:38,442 INFO Quality epoch   3/100 — va_huber=0.6172
2026-04-27 00:13:38,534 INFO Quality epoch   4/100 — va_huber=0.6107
2026-04-27 00:13:38,633 INFO Quality epoch   5/100 — va_huber=0.6055
2026-04-27 00:13:39,256 INFO Quality epoch  11/100 — va_huber=1.5491
2026-04-27 00:13:39,622 INFO Quality early stop at epoch 15
2026-04-27 00:13:39,639 INFO QualityScorer EV model: MAE=1.255 dir_acc=0.564 n_val=799
2026-04-27 00:13:39,644 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-27 00:13:39,708 INFO Quality phase train: 4.3s | total: 4.8s
2026-04-27 00:13:39,709 INFO Retrain complete. Total wall-clock: 4.8s
2026-04-27 00:13:40,677 INFO Model quality: SUCCESS
2026-04-27 00:13:40,677 INFO --- Training rl ---
2026-04-27 00:13:40,678 INFO Running retrain --model rl
2026-04-27 00:13:40,859 INFO retrain environment: KAGGLE
2026-04-27 00:13:42,427 INFO Device: CUDA (2 GPU(s))
2026-04-27 00:13:42,436 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 00:13:42,436 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 00:13:42,436 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 00:13:42,436 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 00:13:42,437 INFO Retrain data split: train
2026-04-27 00:13:42,438 INFO === RLAgent (PPO) retrain ===
2026-04-27 00:13:42,443 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260427_001342
2026-04-27 00:13:42,547 INFO RL phase episode loading: 0.1s (3991 episodes)
2026-04-27 00:13:45.921254: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777248826.119441   61270 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777248826.177290   61270 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777248826.670100   61270 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777248826.670132   61270 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777248826.670135   61270 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777248826.670138   61270 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-27 00:14:00,314 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-27 00:14:03,552 INFO RLAgent: cold start — building new PPO policy
2026-04-27 00:15:21,576 INFO RLAgent: retrain complete, 3991 episodes
2026-04-27 00:15:21,576 INFO RL phase PPO train: 99.0s | total: 99.1s
2026-04-27 00:15:21,589 INFO Retrain complete. Total wall-clock: 99.2s
2026-04-27 00:15:23,226 INFO Model rl: SUCCESS
2026-04-27 00:15:23,227 INFO Step 7b complete — summary: /kaggle/working/Multi-Bot/trading-system/ml_training/metrics/training_7b_summary.json
  DONE  Round 1 - Quality+RL retrain

=== Round 2: BLIND backtest on test window (unseen 2yr) ===
  START Round 2 - Blind backtest (test)
2026-04-27 00:15:23,818 INFO === STEP 6: BACKTEST (round2) ===
2026-04-27 00:15:23,819 INFO BT_WINDOW=round2 — BLIND backtest: 2023-08-07 → 2025-08-05 (test set)
2026-04-27 00:15:23,819 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-27 00:15:23,819 INFO Round 2 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-27 00:17:46,451 INFO Round 2 backtest — 3109 trades | avg WR=55.3% | avg PF=2.84 | avg Sharpe=7.43
2026-04-27 00:17:46,451 INFO   ml_trader: 3109 trades | WR=55.3% | PF=2.84 | Return=2271.4% | DD=3.5% | Sharpe=7.43
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 3109
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (3109 rows)
2026-04-27 00:17:47,737 INFO Round 2: wrote 3109 journal entries (total in file: 7100)
  DONE  Round 2 - Blind backtest (test)
  Saved Round 2 result → round2_summary.json
  Journal after Round 2: 7100 entries

=== Round 2 → Retrain Quality + RL (Round 1+2 journal) ===
  START Round 2 - Quality+RL retrain
2026-04-27 00:17:47,991 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-27 00:17:47,998 INFO Journal entries: 7100
2026-04-27 00:17:47,998 INFO --- Training quality ---
2026-04-27 00:17:47,999 INFO Running retrain --model quality
2026-04-27 00:17:48,174 INFO retrain environment: KAGGLE
2026-04-27 00:17:49,823 INFO Device: CUDA (2 GPU(s))
2026-04-27 00:17:49,834 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 00:17:49,834 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 00:17:49,834 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 00:17:49,834 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 00:17:49,835 INFO Retrain data split: train
2026-04-27 00:17:49,836 INFO === QualityScorer retrain ===
2026-04-27 00:17:49,981 INFO NumExpr defaulting to 4 threads.
2026-04-27 00:17:50,168 INFO QualityScorer: CUDA available — using GPU
2026-04-27 00:17:50,379 INFO QualityScorer loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (device=cuda)
2026-04-27 00:17:50,817 INFO Quality phase label creation: 0.4s (7100 trades)
2026-04-27 00:17:50,817 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/quality_scorer.pkl_20260427_001750
2026-04-27 00:17:51,237 INFO QualityScorer: 7100 samples, EV stats={'mean': 0.3922163248062134, 'std': 1.2904682159423828, 'n_pos': 4085, 'n_neg': 3015}, device=cuda
2026-04-27 00:17:51,238 INFO QualityScorer: normalised win labels by median_win=0.940 — EV range now [-1, +3]
2026-04-27 00:17:51,238 INFO QualityScorer: warm start from existing weights
2026-04-27 00:17:51,239 INFO QualityScorer: pos_weight=0.72 (n_pos=3299 n_neg=2381)
2026-04-27 00:17:53,624 INFO Quality epoch   1/100 — va_huber=0.5752
2026-04-27 00:17:53,775 INFO Quality epoch   2/100 — va_huber=0.5724
2026-04-27 00:17:53,923 INFO Quality epoch   3/100 — va_huber=0.5702
2026-04-27 00:17:54,065 INFO Quality epoch   4/100 — va_huber=0.5698
2026-04-27 00:17:54,213 INFO Quality epoch   5/100 — va_huber=0.5704
2026-04-27 00:17:55,080 INFO Quality epoch  11/100 — va_huber=0.5714
2026-04-27 00:17:55,965 INFO Quality early stop at epoch 17
2026-04-27 00:17:55,988 INFO QualityScorer EV model: MAE=1.163 dir_acc=0.587 n_val=1420
2026-04-27 00:17:55,992 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-27 00:17:56,064 INFO Quality phase train: 5.2s | total: 6.2s
2026-04-27 00:17:56,066 INFO Retrain complete. Total wall-clock: 6.2s
2026-04-27 00:17:57,050 INFO Model quality: SUCCESS
2026-04-27 00:17:57,051 INFO --- Training rl ---
2026-04-27 00:17:57,051 INFO Running retrain --model rl
2026-04-27 00:17:57,229 INFO retrain environment: KAGGLE
2026-04-27 00:17:58,828 INFO Device: CUDA (2 GPU(s))
2026-04-27 00:17:58,840 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 00:17:58,840 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 00:17:58,840 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 00:17:58,840 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 00:17:58,840 INFO Retrain data split: train
2026-04-27 00:17:58,841 INFO === RLAgent (PPO) retrain ===
2026-04-27 00:17:58,843 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260427_001758
2026-04-27 00:17:59.700638: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777249079.724445   61902 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777249079.731966   61902 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777249079.751316   61902 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777249079.751341   61902 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777249079.751343   61902 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777249079.751346   61902 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-27 00:18:04,036 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-27 00:18:06,791 INFO RLAgent: PPO model loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-27 00:18:06,968 INFO RL phase episode loading: 0.2s (7100 episodes)
2026-04-27 00:18:07,144 INFO RLAgent: warm start — fine-tuning existing PPO policy (lr=6.00e-05)
2026-04-27 00:20:20,593 INFO RLAgent: retrain complete, 7100 episodes
2026-04-27 00:20:20,594 INFO RL phase PPO train: 133.6s | total: 141.8s
2026-04-27 00:20:20,614 INFO Retrain complete. Total wall-clock: 141.8s
2026-04-27 00:20:22,124 INFO Model rl: SUCCESS
2026-04-27 00:20:22,125 INFO Step 7b complete — summary: /kaggle/working/Multi-Bot/trading-system/ml_training/metrics/training_7b_summary.json
  DONE  Round 2 - Quality+RL retrain

=== Round 3: Incremental retrain on train split only ===
  START Retrain gru [train-split retrain]
2026-04-27 00:20:22,424 INFO retrain environment: KAGGLE
2026-04-27 00:20:24,010 INFO Device: CUDA (2 GPU(s))
2026-04-27 00:20:24,021 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 00:20:24,021 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 00:20:24,021 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 00:20:24,022 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 00:20:24,022 INFO Retrain data split: train
2026-04-27 00:20:24,023 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-27 00:20:24,163 INFO NumExpr defaulting to 4 threads.
2026-04-27 00:20:24,352 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-27 00:20:24,353 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 00:20:24,353 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 00:20:24,595 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-27 00:20:24,595 INFO GRU phase macro_correlations: 0.0s
2026-04-27 00:20:24,595 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-27 00:20:24,596 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260427_002024
2026-04-27 00:20:24,600 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-04-27 00:20:24,744 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:24,763 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:24,776 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:24,782 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:24,783 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-27 00:20:24,783 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 00:20:24,783 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 00:20:24,784 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-27 00:20:24,785 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:24,862 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-27 00:20:24,863 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:25,089 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-27 00:20:25,119 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:25,377 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:25,498 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:25,588 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:25,778 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:25,796 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:25,809 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:25,816 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:25,817 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:25,891 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-27 00:20:25,893 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:26,117 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-27 00:20:26,132 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:26,389 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:26,513 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:26,604 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:26,779 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:26,798 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:26,812 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:26,819 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:26,820 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:26,896 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-27 00:20:26,897 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:27,124 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-27 00:20:27,141 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:27,398 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:27,519 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:27,614 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:27,789 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:27,806 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:27,820 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:27,828 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:27,828 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:27,905 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-27 00:20:27,906 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:28,132 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-27 00:20:28,154 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:28,410 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:28,531 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:28,626 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:28,822 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:28,842 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:28,856 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:28,863 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:28,864 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:28,947 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-27 00:20:28,950 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:29,193 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-27 00:20:29,208 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:29,465 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:29,587 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:29,677 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:29,847 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:29,866 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:29,879 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:29,886 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:29,887 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:29,960 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-27 00:20:29,961 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:30,184 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-27 00:20:30,200 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:30,455 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:30,578 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:30,670 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:30,827 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 00:20:30,843 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 00:20:30,857 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 00:20:30,863 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 00:20:30,864 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:30,937 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-27 00:20:30,939 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:31,158 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-27 00:20:31,170 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:31,421 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:31,544 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:31,634 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:31,804 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:31,821 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:31,833 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:31,840 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:31,841 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:31,915 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-27 00:20:31,917 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:32,129 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-27 00:20:32,145 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:32,404 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:32,528 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:32,616 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:32,789 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:32,806 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:32,820 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:32,827 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:32,828 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:32,901 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-27 00:20:32,902 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:33,118 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-27 00:20:33,133 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:33,389 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:33,514 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:33,604 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:33,781 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:33,799 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:33,813 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:33,820 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:20:33,821 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:33,897 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-27 00:20:33,899 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:34,119 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-27 00:20:34,134 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:34,393 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:34,517 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:34,607 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:20:34,906 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 00:20:34,931 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 00:20:34,947 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 00:20:34,956 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 00:20:34,958 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 00:20:35,112 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 00:20:35,115 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 00:20:35,584 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 00:20:35,628 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-27 00:20:36,123 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-27 00:20:36,311 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 00:20:36,437 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 00:20:36,552 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-27 00:20:36,552 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-27 00:20:36,552 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-27 00:21:22,529 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-27 00:21:22,529 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-27 00:21:23,834 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-27 00:21:27,815 INFO train_multi TF=ALL: warm-start detected — using CosineAnnealingLR (lr=3e-05, patience=12)
2026-04-27 00:21:41,045 INFO train_multi TF=ALL epoch 1/50 train=0.5887 val=0.6107
2026-04-27 00:21:41,050 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 00:21:41,050 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 00:21:41,050 INFO train_multi TF=ALL: new best val=0.6107 — saved
2026-04-27 00:21:52,600 INFO train_multi TF=ALL epoch 2/50 train=0.5884 val=0.6115
2026-04-27 00:22:04,074 INFO train_multi TF=ALL epoch 3/50 train=0.5885 val=0.6109
2026-04-27 00:22:15,468 INFO train_multi TF=ALL epoch 4/50 train=0.5881 val=0.6114
2026-04-27 00:22:26,955 INFO train_multi TF=ALL epoch 5/50 train=0.5879 val=0.6106
2026-04-27 00:22:26,960 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 00:22:26,960 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 00:22:26,960 INFO train_multi TF=ALL: new best val=0.6106 — saved
2026-04-27 00:22:38,417 INFO train_multi TF=ALL epoch 6/50 train=0.5878 val=0.6122
2026-04-27 00:22:49,930 INFO train_multi TF=ALL epoch 7/50 train=0.5875 val=0.6118
2026-04-27 00:23:01,500 INFO train_multi TF=ALL epoch 8/50 train=0.5876 val=0.6119
2026-04-27 00:23:13,039 INFO train_multi TF=ALL epoch 9/50 train=0.5873 val=0.6115
2026-04-27 00:23:25,072 INFO train_multi TF=ALL epoch 10/50 train=0.5873 val=0.6119
2026-04-27 00:23:37,481 INFO train_multi TF=ALL epoch 11/50 train=0.5867 val=0.6114
2026-04-27 00:23:49,439 INFO train_multi TF=ALL epoch 12/50 train=0.5867 val=0.6113
2026-04-27 00:24:01,309 INFO train_multi TF=ALL epoch 13/50 train=0.5866 val=0.6122
2026-04-27 00:24:13,033 INFO train_multi TF=ALL epoch 14/50 train=0.5866 val=0.6110
2026-04-27 00:24:24,867 INFO train_multi TF=ALL epoch 15/50 train=0.5863 val=0.6113
2026-04-27 00:24:36,761 INFO train_multi TF=ALL epoch 16/50 train=0.5861 val=0.6123
2026-04-27 00:24:48,646 INFO train_multi TF=ALL epoch 17/50 train=0.5858 val=0.6119
2026-04-27 00:24:48,646 INFO train_multi TF=ALL early stop at epoch 17
2026-04-27 00:24:48,786 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-27 00:24:48,787 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-27 00:24:48,787 INFO Retrain complete. Total wall-clock: 264.8s
  DONE  Retrain gru [train-split retrain]
  START Retrain regime [train-split retrain]
2026-04-27 00:24:50,981 INFO retrain environment: KAGGLE
2026-04-27 00:24:52,586 INFO Device: CUDA (2 GPU(s))
2026-04-27 00:24:52,595 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 00:24:52,595 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 00:24:52,595 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 00:24:52,595 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 00:24:52,595 INFO Retrain data split: train
2026-04-27 00:24:52,596 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-27 00:24:52,745 INFO NumExpr defaulting to 4 threads.
2026-04-27 00:24:52,939 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-27 00:24:52,939 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 00:24:52,939 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 00:24:52,939 INFO Regime phase macro_correlations: 0.0s
2026-04-27 00:24:52,939 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-27 00:24:52,977 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-27 00:24:52,978 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:24:53,004 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:24:53,019 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:24:53,041 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:24:53,055 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:24:53,083 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:24:53,098 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:24:53,120 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:24:53,134 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:24:53,157 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:24:53,172 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:24:53,193 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:24:53,207 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:24:53,227 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:24:53,243 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:24:53,264 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:24:53,279 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:24:53,301 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:24:53,316 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:24:53,339 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:24:53,356 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 00:24:53,395 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 00:24:54,169 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 00:25:17,730 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 00:25:17,735 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 24.3s
2026-04-27 00:25:17,735 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 00:25:27,848 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 00:25:27,849 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 10.1s
2026-04-27 00:25:27,852 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 00:25:35,677 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 00:25:35,681 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 7.8s
2026-04-27 00:25:35,681 INFO Regime phase GMM HTF total: 42.3s
2026-04-27 00:25:35,681 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 00:26:44,186 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 00:26:44,189 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 68.5s
2026-04-27 00:26:44,190 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 00:27:13,971 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 00:27:13,976 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 29.8s
2026-04-27 00:27:13,976 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 00:27:35,488 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 00:27:35,490 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 21.5s
2026-04-27 00:27:35,490 INFO Regime phase GMM LTF total: 119.8s
2026-04-27 00:27:35,592 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-27 00:27:35,593 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:35,594 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:35,595 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:35,596 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:35,597 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:35,598 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:35,599 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:35,600 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:35,601 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:35,602 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:35,604 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 00:27:35,726 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:35,769 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:35,770 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:35,770 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:35,778 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:35,779 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:36,163 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-27 00:27:36,164 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-27 00:27:36,335 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:36,368 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:36,369 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:36,369 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:36,377 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:36,378 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:36,737 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-27 00:27:36,739 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-27 00:27:36,923 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:36,957 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:36,958 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:36,958 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:36,966 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:36,967 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:37,335 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-27 00:27:37,336 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-27 00:27:37,506 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:37,539 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:37,540 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:37,540 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:37,547 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:37,548 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:37,919 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-27 00:27:37,920 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-27 00:27:38,096 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:38,130 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:38,130 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:38,131 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:38,138 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:38,139 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:38,501 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-27 00:27:38,502 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-27 00:27:38,675 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:38,707 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:38,708 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:38,709 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:38,718 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:38,719 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:39,111 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-27 00:27:39,112 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-27 00:27:39,259 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 00:27:39,286 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-27 00:27:39,286 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 00:27:39,287 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 00:27:39,294 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 00:27:39,294 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:39,647 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-27 00:27:39,648 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-27 00:27:39,813 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:39,845 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:39,846 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:39,847 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:39,855 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:39,856 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:40,219 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-27 00:27:40,220 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-27 00:27:40,384 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:40,414 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:40,415 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:40,416 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:40,423 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:40,424 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:40,781 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-27 00:27:40,782 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-27 00:27:40,951 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:40,985 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:40,986 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:40,986 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:40,993 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:40,994 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:41,361 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-27 00:27:41,363 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-27 00:27:41,629 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 00:27:41,684 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-27 00:27:41,685 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 00:27:41,686 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 00:27:41,696 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 00:27:41,698 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 00:27:42,455 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 00:27:42,456 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-27 00:27:42,615 INFO Regime phase HTF dataset build: 7.0s (103290 samples)
2026-04-27 00:27:42,615 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260427_002742
2026-04-27 00:27:42,814 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=34, n_classes=3)
2026-04-27 00:27:42,815 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=38114 dropped=65176 classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728})
2026-04-27 00:27:42,824 INFO RegimeClassifier[mode=htf_bias]: 38114 samples, classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728}, device=cuda
2026-04-27 00:27:42,825 INFO RegimeClassifier: sample weights — mean=0.708  ambiguous(<0.4)=0.0%
2026-04-27 00:27:42,825 INFO RegimeClassifier[mode=htf_bias]: warm start from existing weights
2026-04-27 00:27:42,825 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-27 00:27:44,984 INFO Regime epoch  1/50 — tr=0.4811 va=1.1086 acc=0.973 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 0.998, 'BIAS_NEUTRAL': 0.853}
2026-04-27 00:27:45,057 INFO Regime epoch  2/50 — tr=0.4809 va=1.1055 acc=0.973
2026-04-27 00:27:45,130 INFO Regime epoch  3/50 — tr=0.4809 va=1.1077 acc=0.973
2026-04-27 00:27:45,201 INFO Regime epoch  4/50 — tr=0.4807 va=1.1080 acc=0.973
2026-04-27 00:27:45,274 INFO Regime epoch  5/50 — tr=0.4802 va=1.1049 acc=0.973 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 0.998, 'BIAS_NEUTRAL': 0.853}
2026-04-27 00:27:45,339 INFO Regime epoch  6/50 — tr=0.4802 va=1.1044 acc=0.973
2026-04-27 00:27:45,409 INFO Regime epoch  7/50 — tr=0.4795 va=1.1015 acc=0.973
2026-04-27 00:27:45,481 INFO Regime epoch  8/50 — tr=0.4794 va=1.0967 acc=0.974
2026-04-27 00:27:45,554 INFO Regime epoch  9/50 — tr=0.4788 va=1.0945 acc=0.974
2026-04-27 00:27:45,630 INFO Regime epoch 10/50 — tr=0.4782 va=1.0935 acc=0.975 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.862}
2026-04-27 00:27:45,700 INFO Regime epoch 11/50 — tr=0.4782 va=1.0944 acc=0.975
2026-04-27 00:27:45,767 INFO Regime epoch 12/50 — tr=0.4775 va=1.0899 acc=0.976
2026-04-27 00:27:45,833 INFO Regime epoch 13/50 — tr=0.4772 va=1.0860 acc=0.977
2026-04-27 00:27:45,905 INFO Regime epoch 14/50 — tr=0.4769 va=1.0835 acc=0.977
2026-04-27 00:27:45,980 INFO Regime epoch 15/50 — tr=0.4761 va=1.0813 acc=0.977 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.875}
2026-04-27 00:27:46,052 INFO Regime epoch 16/50 — tr=0.4759 va=1.0819 acc=0.977
2026-04-27 00:27:46,123 INFO Regime epoch 17/50 — tr=0.4756 va=1.0803 acc=0.978
2026-04-27 00:27:46,196 INFO Regime epoch 18/50 — tr=0.4750 va=1.0795 acc=0.978
2026-04-27 00:27:46,264 INFO Regime epoch 19/50 — tr=0.4750 va=1.0762 acc=0.979
2026-04-27 00:27:46,339 INFO Regime epoch 20/50 — tr=0.4751 va=1.0741 acc=0.980 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.888}
2026-04-27 00:27:46,410 INFO Regime epoch 21/50 — tr=0.4742 va=1.0723 acc=0.980
2026-04-27 00:27:46,482 INFO Regime epoch 22/50 — tr=0.4741 va=1.0728 acc=0.980
2026-04-27 00:27:46,549 INFO Regime epoch 23/50 — tr=0.4737 va=1.0715 acc=0.981
2026-04-27 00:27:46,614 INFO Regime epoch 24/50 — tr=0.4733 va=1.0683 acc=0.981
2026-04-27 00:27:46,690 INFO Regime epoch 25/50 — tr=0.4733 va=1.0651 acc=0.982 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.897}
2026-04-27 00:27:46,762 INFO Regime epoch 26/50 — tr=0.4734 va=1.0634 acc=0.981
2026-04-27 00:27:46,827 INFO Regime epoch 27/50 — tr=0.4726 va=1.0615 acc=0.982
2026-04-27 00:27:46,894 INFO Regime epoch 28/50 — tr=0.4727 va=1.0595 acc=0.982
2026-04-27 00:27:46,959 INFO Regime epoch 29/50 — tr=0.4722 va=1.0579 acc=0.982
2026-04-27 00:27:47,031 INFO Regime epoch 30/50 — tr=0.4723 va=1.0576 acc=0.982 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.902}
2026-04-27 00:27:47,100 INFO Regime epoch 31/50 — tr=0.4720 va=1.0596 acc=0.983
2026-04-27 00:27:47,169 INFO Regime epoch 32/50 — tr=0.4715 va=1.0614 acc=0.982
2026-04-27 00:27:47,239 INFO Regime epoch 33/50 — tr=0.4717 va=1.0594 acc=0.983
2026-04-27 00:27:47,309 INFO Regime epoch 34/50 — tr=0.4719 va=1.0601 acc=0.983
2026-04-27 00:27:47,382 INFO Regime epoch 35/50 — tr=0.4716 va=1.0574 acc=0.983 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.907}
2026-04-27 00:27:47,446 INFO Regime epoch 36/50 — tr=0.4718 va=1.0579 acc=0.983
2026-04-27 00:27:47,512 INFO Regime epoch 37/50 — tr=0.4715 va=1.0580 acc=0.982
2026-04-27 00:27:47,578 INFO Regime epoch 38/50 — tr=0.4717 va=1.0577 acc=0.983
2026-04-27 00:27:47,642 INFO Regime epoch 39/50 — tr=0.4715 va=1.0564 acc=0.982
2026-04-27 00:27:47,718 INFO Regime epoch 40/50 — tr=0.4711 va=1.0587 acc=0.982 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.901}
2026-04-27 00:27:47,786 INFO Regime epoch 41/50 — tr=0.4713 va=1.0561 acc=0.982
2026-04-27 00:27:47,853 INFO Regime epoch 42/50 — tr=0.4714 va=1.0553 acc=0.984
2026-04-27 00:27:47,922 INFO Regime epoch 43/50 — tr=0.4712 va=1.0559 acc=0.984
2026-04-27 00:27:47,990 INFO Regime epoch 44/50 — tr=0.4710 va=1.0557 acc=0.984
2026-04-27 00:27:48,060 INFO Regime epoch 45/50 — tr=0.4714 va=1.0552 acc=0.984 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.911}
2026-04-27 00:27:48,132 INFO Regime epoch 46/50 — tr=0.4707 va=1.0556 acc=0.984
2026-04-27 00:27:48,215 INFO Regime epoch 47/50 — tr=0.4713 va=1.0554 acc=0.983
2026-04-27 00:27:48,296 INFO Regime epoch 48/50 — tr=0.4710 va=1.0545 acc=0.984
2026-04-27 00:27:48,371 INFO Regime epoch 49/50 — tr=0.4709 va=1.0549 acc=0.983
2026-04-27 00:27:48,442 INFO Regime epoch 50/50 — tr=0.4710 va=1.0558 acc=0.982 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.902}
2026-04-27 00:27:48,451 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 00:27:48,451 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 00:27:48,452 INFO Regime phase HTF train: 5.6s
2026-04-27 00:27:48,575 INFO Regime HTF complete: acc=0.984, n=103290 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.911}
2026-04-27 00:27:48,576 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 00:27:48,723 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 00:27:48,726 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 11.41578947368421, 'BIAS_DOWN': 10.635761589403973, 'BIAS_NEUTRAL': 17.960468521229867}
2026-04-27 00:27:48,729 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 4338, 'mean': 0.00011110466582543088, 'mean_over_std': 0.027632581210502927}, 'BIAS_DOWN': {'n': 3212, 'mean': 6.063430222660252e-05, 'mean_over_std': 0.013898489590377157}, 'BIAS_NEUTRAL': {'n': 12266, 'mean': 1.2151554902814046e-05, 'mean_over_std': 0.0031957000513162357}}
2026-04-27 00:27:48,730 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 3651, 'mean': 0.0002852242108836959, 'mean_over_std': 0.07015017232491755}, 'BIAS_DOWN': {'n': 2644, 'mean': -0.00014900749618981248, 'mean_over_std': -0.03333037041298237}, 'BIAS_NEUTRAL': {'n': 1376, 'mean': 9.595797941249082e-05, 'mean_over_std': 0.02492411099521803}}
2026-04-27 00:27:48,730 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-27 00:27:48,732 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:48,733 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:48,735 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:48,737 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:48,738 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:48,740 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:48,741 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:48,743 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:48,745 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:48,746 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:48,749 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 00:27:48,761 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:48,764 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:48,765 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:48,765 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:48,765 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:48,767 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:49,377 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-27 00:27:49,380 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-27 00:27:49,511 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:49,514 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:49,515 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:49,515 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:49,515 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:49,517 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:50,085 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-27 00:27:50,088 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-27 00:27:50,219 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:50,221 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:50,222 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:50,222 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:50,223 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:50,225 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:50,784 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-27 00:27:50,787 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-27 00:27:50,919 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:50,921 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:50,922 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:50,923 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:50,923 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:50,925 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:51,486 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-27 00:27:51,489 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-27 00:27:51,622 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:51,625 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:51,625 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:51,626 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:51,626 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:51,628 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:52,211 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-27 00:27:52,213 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-27 00:27:52,346 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:52,348 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:52,349 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:52,349 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:52,350 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:52,351 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:52,930 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-27 00:27:52,932 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-27 00:27:53,060 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 00:27:53,062 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-27 00:27:53,062 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 00:27:53,063 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 00:27:53,063 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 00:27:53,065 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:53,634 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-27 00:27:53,636 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-27 00:27:53,772 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:53,774 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:53,775 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:53,776 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:53,776 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:53,778 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:54,350 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-27 00:27:54,352 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-27 00:27:54,486 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:54,488 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:54,489 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:54,490 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:54,490 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:54,492 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:55,079 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-27 00:27:55,081 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-27 00:27:55,217 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:55,219 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:55,220 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:55,221 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:55,221 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 00:27:55,223 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 00:27:55,800 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-27 00:27:55,803 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-27 00:27:55,942 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 00:27:55,946 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-27 00:27:55,947 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 00:27:55,948 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 00:27:55,948 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 00:27:55,951 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 00:27:57,174 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 00:27:57,179 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-27 00:27:57,479 INFO Regime phase LTF dataset build: 8.7s (401471 samples)
2026-04-27 00:27:57,480 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260427_002757
2026-04-27 00:27:57,484 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=18, n_classes=4)
2026-04-27 00:27:57,487 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=299185 dropped=102286 classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127})
2026-04-27 00:27:57,542 INFO RegimeClassifier[mode=ltf_behaviour]: 299185 samples, classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127}, device=cuda
2026-04-27 00:27:57,543 INFO RegimeClassifier: sample weights — mean=0.693  ambiguous(<0.4)=0.0%
2026-04-27 00:27:57,544 INFO RegimeClassifier[mode=ltf_behaviour]: warm start from existing weights
2026-04-27 00:27:57,544 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-27 00:27:58,123 INFO Regime epoch  1/50 — tr=0.6250 va=1.2474 acc=0.827 per_class={'TRENDING': 0.804, 'RANGING': 0.763, 'CONSOLIDATING': 0.839, 'VOLATILE': 0.901}
2026-04-27 00:27:58,637 INFO Regime epoch  2/50 — tr=0.6251 va=1.2465 acc=0.827
2026-04-27 00:27:59,154 INFO Regime epoch  3/50 — tr=0.6248 va=1.2470 acc=0.827
2026-04-27 00:27:59,648 INFO Regime epoch  4/50 — tr=0.6251 va=1.2492 acc=0.827
2026-04-27 00:28:00,189 INFO Regime epoch  5/50 — tr=0.6248 va=1.2396 acc=0.825 per_class={'TRENDING': 0.799, 'RANGING': 0.771, 'CONSOLIDATING': 0.827, 'VOLATILE': 0.906}
2026-04-27 00:28:00,710 INFO Regime epoch  6/50 — tr=0.6248 va=1.2425 acc=0.827
2026-04-27 00:28:01,213 INFO Regime epoch  7/50 — tr=0.6243 va=1.2450 acc=0.828
2026-04-27 00:28:01,715 INFO Regime epoch  8/50 — tr=0.6245 va=1.2429 acc=0.825
2026-04-27 00:28:02,231 INFO Regime epoch  9/50 — tr=0.6240 va=1.2427 acc=0.829
2026-04-27 00:28:02,756 INFO Regime epoch 10/50 — tr=0.6239 va=1.2440 acc=0.830 per_class={'TRENDING': 0.807, 'RANGING': 0.767, 'CONSOLIDATING': 0.846, 'VOLATILE': 0.899}
2026-04-27 00:28:03,263 INFO Regime epoch 11/50 — tr=0.6239 va=1.2394 acc=0.831
2026-04-27 00:28:03,774 INFO Regime epoch 12/50 — tr=0.6235 va=1.2419 acc=0.831
2026-04-27 00:28:04,299 INFO Regime epoch 13/50 — tr=0.6231 va=1.2429 acc=0.832
2026-04-27 00:28:04,785 INFO Regime epoch 14/50 — tr=0.6230 va=1.2338 acc=0.829
2026-04-27 00:28:05,331 INFO Regime epoch 15/50 — tr=0.6227 va=1.2403 acc=0.833 per_class={'TRENDING': 0.81, 'RANGING': 0.761, 'CONSOLIDATING': 0.86, 'VOLATILE': 0.901}
2026-04-27 00:28:05,846 INFO Regime epoch 16/50 — tr=0.6222 va=1.2347 acc=0.831
2026-04-27 00:28:06,363 INFO Regime epoch 17/50 — tr=0.6225 va=1.2368 acc=0.832
2026-04-27 00:28:06,859 INFO Regime epoch 18/50 — tr=0.6222 va=1.2328 acc=0.833
2026-04-27 00:28:07,366 INFO Regime epoch 19/50 — tr=0.6220 va=1.2338 acc=0.833
2026-04-27 00:28:07,916 INFO Regime epoch 20/50 — tr=0.6216 va=1.2388 acc=0.834 per_class={'TRENDING': 0.811, 'RANGING': 0.769, 'CONSOLIDATING': 0.852, 'VOLATILE': 0.905}
2026-04-27 00:28:08,418 INFO Regime epoch 21/50 — tr=0.6215 va=1.2323 acc=0.835
2026-04-27 00:28:08,925 INFO Regime epoch 22/50 — tr=0.6215 va=1.2331 acc=0.836
2026-04-27 00:28:09,433 INFO Regime epoch 23/50 — tr=0.6212 va=1.2308 acc=0.836
2026-04-27 00:28:09,911 INFO Regime epoch 24/50 — tr=0.6211 va=1.2300 acc=0.835
2026-04-27 00:28:10,433 INFO Regime epoch 25/50 — tr=0.6208 va=1.2300 acc=0.834 per_class={'TRENDING': 0.811, 'RANGING': 0.771, 'CONSOLIDATING': 0.861, 'VOLATILE': 0.9}
2026-04-27 00:28:10,910 INFO Regime epoch 26/50 — tr=0.6209 va=1.2318 acc=0.838
2026-04-27 00:28:11,389 INFO Regime epoch 27/50 — tr=0.6208 va=1.2305 acc=0.835
2026-04-27 00:28:11,878 INFO Regime epoch 28/50 — tr=0.6207 va=1.2319 acc=0.836
2026-04-27 00:28:12,367 INFO Regime epoch 29/50 — tr=0.6205 va=1.2288 acc=0.836
2026-04-27 00:28:12,886 INFO Regime epoch 30/50 — tr=0.6201 va=1.2300 acc=0.837 per_class={'TRENDING': 0.817, 'RANGING': 0.767, 'CONSOLIDATING': 0.867, 'VOLATILE': 0.897}
2026-04-27 00:28:13,386 INFO Regime epoch 31/50 — tr=0.6203 va=1.2261 acc=0.835
2026-04-27 00:28:13,877 INFO Regime epoch 32/50 — tr=0.6202 va=1.2283 acc=0.837
2026-04-27 00:28:14,361 INFO Regime epoch 33/50 — tr=0.6201 va=1.2301 acc=0.836
2026-04-27 00:28:14,872 INFO Regime epoch 34/50 — tr=0.6202 va=1.2270 acc=0.837
2026-04-27 00:28:15,426 INFO Regime epoch 35/50 — tr=0.6201 va=1.2279 acc=0.837 per_class={'TRENDING': 0.817, 'RANGING': 0.769, 'CONSOLIDATING': 0.867, 'VOLATILE': 0.896}
2026-04-27 00:28:15,949 INFO Regime epoch 36/50 — tr=0.6198 va=1.2258 acc=0.836
2026-04-27 00:28:16,449 INFO Regime epoch 37/50 — tr=0.6200 va=1.2269 acc=0.838
2026-04-27 00:28:16,955 INFO Regime epoch 38/50 — tr=0.6200 va=1.2283 acc=0.836
2026-04-27 00:28:17,450 INFO Regime epoch 39/50 — tr=0.6198 va=1.2276 acc=0.837
2026-04-27 00:28:17,986 INFO Regime epoch 40/50 — tr=0.6200 va=1.2269 acc=0.837 per_class={'TRENDING': 0.819, 'RANGING': 0.77, 'CONSOLIDATING': 0.868, 'VOLATILE': 0.893}
2026-04-27 00:28:18,490 INFO Regime epoch 41/50 — tr=0.6197 va=1.2295 acc=0.836
2026-04-27 00:28:19,026 INFO Regime epoch 42/50 — tr=0.6198 va=1.2243 acc=0.835
2026-04-27 00:28:19,524 INFO Regime epoch 43/50 — tr=0.6199 va=1.2280 acc=0.836
2026-04-27 00:28:20,023 INFO Regime epoch 44/50 — tr=0.6197 va=1.2302 acc=0.838
2026-04-27 00:28:20,569 INFO Regime epoch 45/50 — tr=0.6199 va=1.2289 acc=0.837 per_class={'TRENDING': 0.816, 'RANGING': 0.767, 'CONSOLIDATING': 0.867, 'VOLATILE': 0.898}
2026-04-27 00:28:21,079 INFO Regime epoch 46/50 — tr=0.6197 va=1.2284 acc=0.838
2026-04-27 00:28:21,588 INFO Regime epoch 47/50 — tr=0.6197 va=1.2260 acc=0.839
2026-04-27 00:28:22,083 INFO Regime epoch 48/50 — tr=0.6199 va=1.2294 acc=0.839
2026-04-27 00:28:22,591 INFO Regime epoch 49/50 — tr=0.6199 va=1.2266 acc=0.837
2026-04-27 00:28:23,135 INFO Regime epoch 50/50 — tr=0.6198 va=1.2290 acc=0.838 per_class={'TRENDING': 0.819, 'RANGING': 0.768, 'CONSOLIDATING': 0.868, 'VOLATILE': 0.894}
2026-04-27 00:28:23,173 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 00:28:23,174 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 00:28:23,176 INFO Regime phase LTF train: 25.7s
2026-04-27 00:28:23,298 INFO Regime LTF complete: acc=0.835, n=401471 per_class={'TRENDING': 0.812, 'RANGING': 0.773, 'CONSOLIDATING': 0.859, 'VOLATILE': 0.9}
2026-04-27 00:28:23,301 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 00:28:23,801 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 00:28:23,806 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 9.355291913830783, 'RANGING': 6.096303199751476, 'CONSOLIDATING': 5.598885793871866, 'VOLATILE': 6.771351107094442}
2026-04-27 00:28:23,813 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 29965, 'mean': 2.149062259083323e-05, 'mean_over_std': 0.011155601434349563}, 'RANGING': {'n': 19623, 'mean': -3.254493608872846e-06, 'mean_over_std': -0.001761415548292494}, 'CONSOLIDATING': {'n': 10050, 'mean': -3.039756158146958e-06, 'mean_over_std': -0.001792488091618151}, 'VOLATILE': {'n': 14985, 'mean': 1.8102764458593193e-05, 'mean_over_std': 0.006974696291378742}}
2026-04-27 00:28:23,814 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 26145, 'mean': 2.874435605138053e-05, 'mean_over_std': 0.01595450632691917}, 'RANGING': {'n': 7858, 'mean': 1.7829776785521727e-05, 'mean_over_std': 0.011851561804467463}, 'CONSOLIDATING': {'n': 8543, 'mean': 7.845231961250924e-06, 'mean_over_std': 0.004915839521549481}, 'VOLATILE': {'n': 13636, 'mean': 5.480114267306668e-06, 'mean_over_std': 0.0020378119917389375}}
2026-04-27 00:28:23,814 INFO Regime retrain total: 211.2s (504761 samples)
2026-04-27 00:28:23,816 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-27 00:28:23,816 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 00:28:23,816 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 00:28:23,817 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-27 00:28:23,817 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-27 00:28:23,817 INFO Retrain complete. Total wall-clock: 211.2s
  DONE  Retrain regime [train-split retrain]
  START Retrain quality [train-split retrain]
2026-04-27 00:28:25,123 INFO retrain environment: KAGGLE
2026-04-27 00:28:26,703 INFO Device: CUDA (2 GPU(s))
2026-04-27 00:28:26,714 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 00:28:26,714 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 00:28:26,715 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 00:28:26,715 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 00:28:26,715 INFO Retrain data split: train
2026-04-27 00:28:26,716 INFO === QualityScorer retrain ===
2026-04-27 00:28:26,866 INFO NumExpr defaulting to 4 threads.
2026-04-27 00:28:27,054 INFO QualityScorer: CUDA available — using GPU
2026-04-27 00:28:27,261 INFO QualityScorer loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (device=cuda)
2026-04-27 00:28:27,701 INFO Quality phase label creation: 0.4s (7100 trades)
2026-04-27 00:28:27,702 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/quality_scorer.pkl_20260427_002827
2026-04-27 00:28:28,129 INFO QualityScorer: 7100 samples, EV stats={'mean': 0.3922163248062134, 'std': 1.2904682159423828, 'n_pos': 4085, 'n_neg': 3015}, device=cuda
2026-04-27 00:28:28,130 INFO QualityScorer: normalised win labels by median_win=0.940 — EV range now [-1, +3]
2026-04-27 00:28:28,130 INFO QualityScorer: warm start from existing weights
2026-04-27 00:28:28,131 INFO QualityScorer: pos_weight=0.72 (n_pos=3299 n_neg=2381)
2026-04-27 00:28:30,543 INFO Quality epoch   1/100 — va_huber=0.5691
2026-04-27 00:28:30,698 INFO Quality epoch   2/100 — va_huber=0.5691
2026-04-27 00:28:30,843 INFO Quality epoch   3/100 — va_huber=0.5686
2026-04-27 00:28:30,987 INFO Quality epoch   4/100 — va_huber=0.5676
2026-04-27 00:28:31,143 INFO Quality epoch   5/100 — va_huber=0.5676
2026-04-27 00:28:32,026 INFO Quality epoch  11/100 — va_huber=0.5701
2026-04-27 00:28:32,461 INFO Quality early stop at epoch 14
2026-04-27 00:28:32,482 INFO QualityScorer EV model: MAE=1.167 dir_acc=0.576 n_val=1420
2026-04-27 00:28:32,486 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-27 00:28:32,553 INFO Quality phase train: 4.9s | total: 5.8s
2026-04-27 00:28:32,555 INFO Retrain complete. Total wall-clock: 5.8s
  DONE  Retrain quality [train-split retrain]
  START Retrain rl [train-split retrain]
2026-04-27 00:28:33,705 INFO retrain environment: KAGGLE
2026-04-27 00:28:35,300 INFO Device: CUDA (2 GPU(s))
2026-04-27 00:28:35,309 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 00:28:35,309 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 00:28:35,309 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 00:28:35,309 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 00:28:35,310 INFO Retrain data split: train
2026-04-27 00:28:35,311 INFO === RLAgent (PPO) retrain ===
2026-04-27 00:28:35,312 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260427_002835
2026-04-27 00:28:36.141948: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777249716.165213   88256 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777249716.172905   88256 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777249716.192489   88256 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777249716.192541   88256 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777249716.192547   88256 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777249716.192551   88256 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-27 00:28:40,523 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-27 00:28:43,244 INFO RLAgent: PPO model loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-27 00:28:43,421 INFO RL phase episode loading: 0.2s (7100 episodes)
2026-04-27 00:28:43,598 INFO RLAgent: warm start — fine-tuning existing PPO policy (lr=6.00e-05)
2026-04-27 00:31:05,156 INFO RLAgent: retrain complete, 7100 episodes
2026-04-27 00:31:05,157 INFO RL phase PPO train: 141.7s | total: 149.8s
2026-04-27 00:31:05,178 INFO Retrain complete. Total wall-clock: 149.9s
  DONE  Retrain rl [train-split retrain]

=== Round 3: Backtest on last 3yr (post-retrain evaluation) ===
  START Round 3 - Post-retrain backtest (last 3yr)
2026-04-27 00:31:07,202 INFO === STEP 6: BACKTEST (round3) ===
2026-04-27 00:31:07,203 INFO BT_WINDOW=round3 — post-retrain eval: 2022-08-05 → 2025-08-05 (last 3yr)
2026-04-27 00:31:07,203 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-04-27 00:31:07,204 INFO Round 3 — running backtest: 2022-08-05 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-27 00:34:24,485 INFO Round 3 backtest — 7127 trades | avg WR=57.5% | avg PF=3.01 | avg Sharpe=7.96
2026-04-27 00:34:24,485 INFO   ml_trader: 7127 trades | WR=57.5% | PF=3.01 | Return=5548.9% | DD=6.8% | Sharpe=7.96
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 7127
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (7127 rows)
  DONE  Round 3 - Post-retrain backtest (last 3yr)
  Saved Round 3 result → round3_summary.json

======================================================================
  BLIND BACKTEST PIPELINE COMPLETE
======================================================================
  Round 1 (val window)          trades=3991  WR=59.3%  PF=3.235  Sharpe=8.544
  Round 2 (blind test)          trades=3109  WR=55.3%  PF=2.844  Sharpe=7.426
  Round 3 (last 3yr)            trades=7127  WR=57.5%  PF=3.010  Sharpe=7.960


WARNING: GITHUB_TOKEN not set — skipping GitHub push
2026-04-27 00:34:27,096 INFO Round 3: wrote 7127 journal entries (total in file: 14227)