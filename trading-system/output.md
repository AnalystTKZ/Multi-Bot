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
2026-04-26 20:19:49,235 INFO Loading feature-engineered data...
2026-04-26 20:19:50,075 INFO Loaded 221743 rows, 202 features
2026-04-26 20:19:50,079 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-26 20:19:50,084 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-26 20:19:50,084 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-26 20:19:50,084 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-26 20:19:50,084 INFO No leakage confirmed: train < val < test timestamps

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
2026-04-26 20:19:52,775 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-26 20:19:52,775 INFO --- Training regime ---
2026-04-26 20:19:52,776 INFO Running retrain --model regime
2026-04-26 20:19:52,966 INFO retrain environment: KAGGLE
2026-04-26 20:19:54,777 INFO Device: CUDA (2 GPU(s))
2026-04-26 20:19:54,788 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 20:19:54,788 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 20:19:54,788 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 20:19:54,791 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 20:19:54,792 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-26 20:19:54,962 INFO NumExpr defaulting to 4 threads.
2026-04-26 20:19:55,195 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 20:19:55,195 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 20:19:55,195 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 20:19:55,195 INFO Regime phase macro_correlations: 0.0s
2026-04-26 20:19:55,195 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-26 20:19:55,236 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 20:19:55,237 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:19:55,266 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:19:55,281 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:19:55,306 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:19:55,320 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:19:55,345 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:19:55,361 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:19:55,385 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:19:55,402 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:19:55,425 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:19:55,440 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:19:55,463 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:19:55,477 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:19:55,497 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:19:55,512 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:19:55,535 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:19:55,552 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:19:55,576 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:19:55,592 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:19:55,619 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:19:55,640 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:19:55,689 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:19:57,259 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 20:20:23,542 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 20:20:23,544 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 27.9s
2026-04-26 20:20:23,544 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 20:20:34,720 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 20:20:34,724 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 11.2s
2026-04-26 20:20:34,724 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 20:20:42,740 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 20:20:42,741 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 8.0s
2026-04-26 20:20:42,741 INFO Regime phase GMM HTF total: 47.1s
2026-04-26 20:20:42,741 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 20:22:01,732 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 20:22:01,736 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 79.0s
2026-04-26 20:22:01,736 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 20:22:37,067 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 20:22:37,071 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 35.3s
2026-04-26 20:22:37,074 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 20:23:02,268 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 20:23:02,269 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 25.2s
2026-04-26 20:23:02,269 INFO Regime phase GMM LTF total: 139.5s
2026-04-26 20:23:02,387 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-26 20:23:02,389 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:02,390 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:02,391 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:02,392 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:02,393 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:02,394 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:02,395 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:02,396 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:02,397 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:02,398 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:02,399 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:23:02,529 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:02,573 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:02,574 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:02,574 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:02,583 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:02,584 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:03,044 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-26 20:23:03,046 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-26 20:23:03,240 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:03,276 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:03,277 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:03,278 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:03,286 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:03,287 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:03,708 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-26 20:23:03,709 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-26 20:23:03,917 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:03,956 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:03,956 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:03,957 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:03,966 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:03,967 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:04,367 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-26 20:23:04,368 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-26 20:23:04,544 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:04,581 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:04,582 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:04,582 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:04,590 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:04,591 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:04,997 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-26 20:23:04,998 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-26 20:23:05,201 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:05,238 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:05,239 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:05,240 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:05,248 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:05,249 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:05,644 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-26 20:23:05,646 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-26 20:23:05,842 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:05,877 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:05,878 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:05,878 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:05,886 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:05,887 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:06,280 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-26 20:23:06,282 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-26 20:23:06,455 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:23:06,485 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:23:06,485 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:23:06,486 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:23:06,493 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:23:06,494 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:06,899 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-26 20:23:06,901 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-26 20:23:07,086 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:07,121 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:07,122 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:07,122 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:07,131 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:07,132 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:07,530 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-26 20:23:07,531 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-26 20:23:07,714 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:07,750 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:07,750 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:07,751 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:07,759 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:07,760 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:08,159 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-26 20:23:08,160 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-26 20:23:08,356 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:08,395 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:08,396 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:08,396 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:08,405 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:08,406 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:08,836 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-26 20:23:08,837 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-26 20:23:09,121 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:23:09,182 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:23:09,183 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:23:09,184 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:23:09,196 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:23:09,197 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:23:10,020 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 20:23:10,022 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-26 20:23:10,202 INFO Regime phase HTF dataset build: 7.8s (103290 samples)
2026-04-26 20:23:10,203 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=38114 dropped=65176 classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728})
2026-04-26 20:23:10,212 INFO RegimeClassifier[mode=htf_bias]: 38114 samples, classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728}, device=cuda
2026-04-26 20:23:10,213 INFO RegimeClassifier: sample weights — mean=0.708  ambiguous(<0.4)=0.0%
2026-04-26 20:23:10,615 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-04-26 20:23:10,616 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 20:23:17,119 INFO Regime epoch  1/50 — tr=0.7407 va=2.0411 acc=0.502 per_class={'BIAS_UP': 0.961, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 0.234}
2026-04-26 20:23:17,200 INFO Regime epoch  2/50 — tr=0.7354 va=2.0443 acc=0.439
2026-04-26 20:23:17,277 INFO Regime epoch  3/50 — tr=0.7269 va=2.0319 acc=0.424
2026-04-26 20:23:17,350 INFO Regime epoch  4/50 — tr=0.7149 va=1.9971 acc=0.472
2026-04-26 20:23:17,434 INFO Regime epoch  5/50 — tr=0.6943 va=1.9350 acc=0.548 per_class={'BIAS_UP': 0.738, 'BIAS_DOWN': 0.257, 'BIAS_NEUTRAL': 0.595}
2026-04-26 20:23:17,511 INFO Regime epoch  6/50 — tr=0.6727 va=1.8535 acc=0.675
2026-04-26 20:23:17,585 INFO Regime epoch  7/50 — tr=0.6473 va=1.7560 acc=0.778
2026-04-26 20:23:17,656 INFO Regime epoch  8/50 — tr=0.6194 va=1.6597 acc=0.845
2026-04-26 20:23:17,728 INFO Regime epoch  9/50 — tr=0.5962 va=1.5695 acc=0.885
2026-04-26 20:23:17,812 INFO Regime epoch 10/50 — tr=0.5757 va=1.4847 acc=0.914 per_class={'BIAS_UP': 0.942, 'BIAS_DOWN': 0.965, 'BIAS_NEUTRAL': 0.744}
2026-04-26 20:23:17,891 INFO Regime epoch 11/50 — tr=0.5558 va=1.4182 acc=0.926
2026-04-26 20:23:17,965 INFO Regime epoch 12/50 — tr=0.5428 va=1.3621 acc=0.935
2026-04-26 20:23:18,038 INFO Regime epoch 13/50 — tr=0.5316 va=1.3145 acc=0.942
2026-04-26 20:23:18,109 INFO Regime epoch 14/50 — tr=0.5226 va=1.2810 acc=0.944
2026-04-26 20:23:18,184 INFO Regime epoch 15/50 — tr=0.5147 va=1.2538 acc=0.949 per_class={'BIAS_UP': 0.985, 'BIAS_DOWN': 0.991, 'BIAS_NEUTRAL': 0.775}
2026-04-26 20:23:18,263 INFO Regime epoch 16/50 — tr=0.5103 va=1.2334 acc=0.952
2026-04-26 20:23:18,341 INFO Regime epoch 17/50 — tr=0.5061 va=1.2115 acc=0.955
2026-04-26 20:23:18,412 INFO Regime epoch 18/50 — tr=0.5026 va=1.1953 acc=0.958
2026-04-26 20:23:18,485 INFO Regime epoch 19/50 — tr=0.5003 va=1.1858 acc=0.958
2026-04-26 20:23:18,560 INFO Regime epoch 20/50 — tr=0.4981 va=1.1752 acc=0.959 per_class={'BIAS_UP': 0.987, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.81}
2026-04-26 20:23:18,632 INFO Regime epoch 21/50 — tr=0.4961 va=1.1677 acc=0.961
2026-04-26 20:23:18,733 INFO Regime epoch 22/50 — tr=0.4935 va=1.1580 acc=0.963
2026-04-26 20:23:18,806 INFO Regime epoch 23/50 — tr=0.4924 va=1.1519 acc=0.965
2026-04-26 20:23:18,878 INFO Regime epoch 24/50 — tr=0.4914 va=1.1489 acc=0.966
2026-04-26 20:23:18,954 INFO Regime epoch 25/50 — tr=0.4900 va=1.1410 acc=0.967 per_class={'BIAS_UP': 0.996, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.829}
2026-04-26 20:23:19,026 INFO Regime epoch 26/50 — tr=0.4890 va=1.1355 acc=0.969
2026-04-26 20:23:19,097 INFO Regime epoch 27/50 — tr=0.4877 va=1.1331 acc=0.969
2026-04-26 20:23:19,169 INFO Regime epoch 28/50 — tr=0.4871 va=1.1302 acc=0.970
2026-04-26 20:23:19,239 INFO Regime epoch 29/50 — tr=0.4856 va=1.1297 acc=0.970
2026-04-26 20:23:19,316 INFO Regime epoch 30/50 — tr=0.4851 va=1.1247 acc=0.971 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.841}
2026-04-26 20:23:19,390 INFO Regime epoch 31/50 — tr=0.4851 va=1.1193 acc=0.972
2026-04-26 20:23:19,464 INFO Regime epoch 32/50 — tr=0.4841 va=1.1206 acc=0.971
2026-04-26 20:23:19,535 INFO Regime epoch 33/50 — tr=0.4833 va=1.1162 acc=0.972
2026-04-26 20:23:19,608 INFO Regime epoch 34/50 — tr=0.4827 va=1.1131 acc=0.972
2026-04-26 20:23:19,686 INFO Regime epoch 35/50 — tr=0.4823 va=1.1121 acc=0.972 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.848}
2026-04-26 20:23:19,760 INFO Regime epoch 36/50 — tr=0.4820 va=1.1109 acc=0.973
2026-04-26 20:23:19,829 INFO Regime epoch 37/50 — tr=0.4813 va=1.1090 acc=0.973
2026-04-26 20:23:19,900 INFO Regime epoch 38/50 — tr=0.4808 va=1.1106 acc=0.973
2026-04-26 20:23:19,970 INFO Regime epoch 39/50 — tr=0.4818 va=1.1092 acc=0.973
2026-04-26 20:23:20,049 INFO Regime epoch 40/50 — tr=0.4812 va=1.1085 acc=0.974 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.856}
2026-04-26 20:23:20,121 INFO Regime epoch 41/50 — tr=0.4805 va=1.1082 acc=0.974
2026-04-26 20:23:20,195 INFO Regime epoch 42/50 — tr=0.4806 va=1.1065 acc=0.974
2026-04-26 20:23:20,268 INFO Regime epoch 43/50 — tr=0.4803 va=1.1084 acc=0.974
2026-04-26 20:23:20,338 INFO Regime epoch 44/50 — tr=0.4807 va=1.1066 acc=0.974
2026-04-26 20:23:20,417 INFO Regime epoch 45/50 — tr=0.4806 va=1.1014 acc=0.975 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.863}
2026-04-26 20:23:20,491 INFO Regime epoch 46/50 — tr=0.4805 va=1.1036 acc=0.974
2026-04-26 20:23:20,561 INFO Regime epoch 47/50 — tr=0.4804 va=1.1048 acc=0.974
2026-04-26 20:23:20,633 INFO Regime epoch 48/50 — tr=0.4804 va=1.1061 acc=0.974
2026-04-26 20:23:20,705 INFO Regime epoch 49/50 — tr=0.4805 va=1.1026 acc=0.974
2026-04-26 20:23:20,783 INFO Regime epoch 50/50 — tr=0.4805 va=1.1030 acc=0.974 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.859}
2026-04-26 20:23:20,795 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 20:23:20,795 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 20:23:20,795 INFO Regime phase HTF train: 10.6s
2026-04-26 20:23:20,932 INFO Regime HTF complete: acc=0.975, n=103290 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.863}
2026-04-26 20:23:20,933 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:23:21,100 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 20:23:21,112 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 11.41578947368421, 'BIAS_DOWN': 10.635761589403973, 'BIAS_NEUTRAL': 17.960468521229867}
2026-04-26 20:23:21,117 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 4338, 'mean': 0.00011110466582543088, 'mean_over_std': 0.027632581210502927}, 'BIAS_DOWN': {'n': 3212, 'mean': 6.063430222660252e-05, 'mean_over_std': 0.013898489590377157}, 'BIAS_NEUTRAL': {'n': 12266, 'mean': 1.2151554902814046e-05, 'mean_over_std': 0.0031957000513162357}}
2026-04-26 20:23:21,118 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 3651, 'mean': 0.0002852242108836959, 'mean_over_std': 0.07015017232491755}, 'BIAS_DOWN': {'n': 2644, 'mean': -0.00014900749618981248, 'mean_over_std': -0.03333037041298237}, 'BIAS_NEUTRAL': {'n': 1376, 'mean': 9.595797941249082e-05, 'mean_over_std': 0.02492411099521803}}
2026-04-26 20:23:21,118 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-26 20:23:21,120 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:21,122 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:21,123 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:21,125 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:21,127 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:21,129 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:21,130 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:21,132 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:21,133 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:21,135 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:21,138 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:23:21,150 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:21,153 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:21,154 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:21,155 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:21,155 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:21,158 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:21,845 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-26 20:23:21,848 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-26 20:23:22,003 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:22,005 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:22,006 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:22,007 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:22,007 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:22,009 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:22,660 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-26 20:23:22,663 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-26 20:23:22,813 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:22,815 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:22,816 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:22,816 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:22,817 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:22,819 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:23,456 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-26 20:23:23,459 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-26 20:23:23,598 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:23,601 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:23,602 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:23,602 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:23,603 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:23,605 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:24,243 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-26 20:23:24,246 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-26 20:23:24,389 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:24,391 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:24,392 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:24,392 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:24,393 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:24,395 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:25,022 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-26 20:23:25,025 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-26 20:23:25,170 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:25,172 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:25,173 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:25,173 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:25,174 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:25,176 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:25,804 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-26 20:23:25,807 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-26 20:23:25,945 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:23:25,947 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:23:25,948 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:23:25,948 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:23:25,949 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:23:25,950 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:26,587 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-26 20:23:26,590 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-26 20:23:26,735 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:26,737 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:26,738 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:26,739 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:26,739 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:26,741 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:27,369 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-26 20:23:27,372 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-26 20:23:27,518 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:27,520 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:27,521 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:27,522 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:27,522 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:27,524 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:28,163 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-26 20:23:28,166 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-26 20:23:28,319 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:28,321 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:28,322 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:28,323 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:28,323 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:23:28,325 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:23:28,990 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-26 20:23:28,993 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-26 20:23:29,145 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:23:29,148 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:23:29,149 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:23:29,150 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:23:29,150 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:23:29,154 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:23:30,513 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 20:23:30,519 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-26 20:23:30,845 INFO Regime phase LTF dataset build: 9.7s (401471 samples)
2026-04-26 20:23:30,849 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=299185 dropped=102286 classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127})
2026-04-26 20:23:30,917 INFO RegimeClassifier[mode=ltf_behaviour]: 299185 samples, classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127}, device=cuda
2026-04-26 20:23:30,918 INFO RegimeClassifier: sample weights — mean=0.693  ambiguous(<0.4)=0.0%
2026-04-26 20:23:30,921 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-04-26 20:23:30,921 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 20:23:31,514 INFO Regime epoch  1/50 — tr=0.8883 va=2.1662 acc=0.373 per_class={'TRENDING': 0.34, 'RANGING': 0.05, 'CONSOLIDATING': 0.28, 'VOLATILE': 0.682}
2026-04-26 20:23:32,060 INFO Regime epoch  2/50 — tr=0.8580 va=2.0332 acc=0.487
2026-04-26 20:23:32,597 INFO Regime epoch  3/50 — tr=0.8127 va=1.8820 acc=0.553
2026-04-26 20:23:33,114 INFO Regime epoch  4/50 — tr=0.7651 va=1.7239 acc=0.603
2026-04-26 20:23:33,691 INFO Regime epoch  5/50 — tr=0.7284 va=1.5999 acc=0.641 per_class={'TRENDING': 0.507, 'RANGING': 0.426, 'CONSOLIDATING': 0.759, 'VOLATILE': 0.947}
2026-04-26 20:23:34,216 INFO Regime epoch  6/50 — tr=0.7051 va=1.5298 acc=0.667
2026-04-26 20:23:34,736 INFO Regime epoch  7/50 — tr=0.6904 va=1.4895 acc=0.682
2026-04-26 20:23:35,266 INFO Regime epoch  8/50 — tr=0.6807 va=1.4633 acc=0.701
2026-04-26 20:23:35,786 INFO Regime epoch  9/50 — tr=0.6728 va=1.4353 acc=0.718
2026-04-26 20:23:36,359 INFO Regime epoch 10/50 — tr=0.6667 va=1.4125 acc=0.729 per_class={'TRENDING': 0.632, 'RANGING': 0.69, 'CONSOLIDATING': 0.735, 'VOLATILE': 0.933}
2026-04-26 20:23:36,903 INFO Regime epoch 11/50 — tr=0.6612 va=1.3968 acc=0.743
2026-04-26 20:23:37,436 INFO Regime epoch 12/50 — tr=0.6572 va=1.3774 acc=0.751
2026-04-26 20:23:37,979 INFO Regime epoch 13/50 — tr=0.6536 va=1.3610 acc=0.761
2026-04-26 20:23:38,547 INFO Regime epoch 14/50 — tr=0.6506 va=1.3483 acc=0.767
2026-04-26 20:23:39,165 INFO Regime epoch 15/50 — tr=0.6482 va=1.3376 acc=0.773 per_class={'TRENDING': 0.718, 'RANGING': 0.736, 'CONSOLIDATING': 0.739, 'VOLATILE': 0.919}
2026-04-26 20:23:39,719 INFO Regime epoch 16/50 — tr=0.6457 va=1.3338 acc=0.780
2026-04-26 20:23:40,274 INFO Regime epoch 17/50 — tr=0.6438 va=1.3224 acc=0.785
2026-04-26 20:23:40,816 INFO Regime epoch 18/50 — tr=0.6424 va=1.3216 acc=0.789
2026-04-26 20:23:41,367 INFO Regime epoch 19/50 — tr=0.6402 va=1.3081 acc=0.792
2026-04-26 20:23:41,936 INFO Regime epoch 20/50 — tr=0.6392 va=1.3038 acc=0.795 per_class={'TRENDING': 0.759, 'RANGING': 0.752, 'CONSOLIDATING': 0.752, 'VOLATILE': 0.915}
2026-04-26 20:23:42,462 INFO Regime epoch 21/50 — tr=0.6375 va=1.2979 acc=0.798
2026-04-26 20:23:43,015 INFO Regime epoch 22/50 — tr=0.6365 va=1.2860 acc=0.799
2026-04-26 20:23:43,541 INFO Regime epoch 23/50 — tr=0.6354 va=1.2826 acc=0.802
2026-04-26 20:23:44,081 INFO Regime epoch 24/50 — tr=0.6340 va=1.2821 acc=0.805
2026-04-26 20:23:44,650 INFO Regime epoch 25/50 — tr=0.6334 va=1.2782 acc=0.808 per_class={'TRENDING': 0.782, 'RANGING': 0.76, 'CONSOLIDATING': 0.778, 'VOLATILE': 0.906}
2026-04-26 20:23:45,210 INFO Regime epoch 26/50 — tr=0.6327 va=1.2753 acc=0.810
2026-04-26 20:23:45,750 INFO Regime epoch 27/50 — tr=0.6319 va=1.2707 acc=0.812
2026-04-26 20:23:46,283 INFO Regime epoch 28/50 — tr=0.6310 va=1.2685 acc=0.813
2026-04-26 20:23:46,831 INFO Regime epoch 29/50 — tr=0.6305 va=1.2634 acc=0.813
2026-04-26 20:23:47,407 INFO Regime epoch 30/50 — tr=0.6296 va=1.2690 acc=0.817 per_class={'TRENDING': 0.793, 'RANGING': 0.759, 'CONSOLIDATING': 0.79, 'VOLATILE': 0.912}
2026-04-26 20:23:47,948 INFO Regime epoch 31/50 — tr=0.6295 va=1.2601 acc=0.818
2026-04-26 20:23:48,497 INFO Regime epoch 32/50 — tr=0.6289 va=1.2575 acc=0.817
2026-04-26 20:23:49,075 INFO Regime epoch 33/50 — tr=0.6285 va=1.2537 acc=0.817
2026-04-26 20:23:49,616 INFO Regime epoch 34/50 — tr=0.6278 va=1.2516 acc=0.819
2026-04-26 20:23:50,202 INFO Regime epoch 35/50 — tr=0.6277 va=1.2513 acc=0.820 per_class={'TRENDING': 0.796, 'RANGING': 0.77, 'CONSOLIDATING': 0.806, 'VOLATILE': 0.906}
2026-04-26 20:23:50,770 INFO Regime epoch 36/50 — tr=0.6275 va=1.2550 acc=0.822
2026-04-26 20:23:51,313 INFO Regime epoch 37/50 — tr=0.6274 va=1.2503 acc=0.820
2026-04-26 20:23:51,856 INFO Regime epoch 38/50 — tr=0.6268 va=1.2508 acc=0.823
2026-04-26 20:23:52,395 INFO Regime epoch 39/50 — tr=0.6265 va=1.2503 acc=0.824
2026-04-26 20:23:52,977 INFO Regime epoch 40/50 — tr=0.6265 va=1.2490 acc=0.823 per_class={'TRENDING': 0.801, 'RANGING': 0.767, 'CONSOLIDATING': 0.823, 'VOLATILE': 0.899}
2026-04-26 20:23:53,510 INFO Regime epoch 41/50 — tr=0.6261 va=1.2515 acc=0.824
2026-04-26 20:23:54,050 INFO Regime epoch 42/50 — tr=0.6264 va=1.2492 acc=0.826
2026-04-26 20:23:54,602 INFO Regime epoch 43/50 — tr=0.6265 va=1.2449 acc=0.824
2026-04-26 20:23:55,160 INFO Regime epoch 44/50 — tr=0.6262 va=1.2431 acc=0.823
2026-04-26 20:23:55,749 INFO Regime epoch 45/50 — tr=0.6260 va=1.2489 acc=0.826 per_class={'TRENDING': 0.803, 'RANGING': 0.77, 'CONSOLIDATING': 0.817, 'VOLATILE': 0.907}
2026-04-26 20:23:56,302 INFO Regime epoch 46/50 — tr=0.6263 va=1.2511 acc=0.826
2026-04-26 20:23:56,849 INFO Regime epoch 47/50 — tr=0.6260 va=1.2459 acc=0.825
2026-04-26 20:23:57,400 INFO Regime epoch 48/50 — tr=0.6259 va=1.2491 acc=0.825
2026-04-26 20:23:57,937 INFO Regime epoch 49/50 — tr=0.6256 va=1.2490 acc=0.826
2026-04-26 20:23:58,516 INFO Regime epoch 50/50 — tr=0.6259 va=1.2466 acc=0.825 per_class={'TRENDING': 0.802, 'RANGING': 0.774, 'CONSOLIDATING': 0.817, 'VOLATILE': 0.904}
2026-04-26 20:23:58,558 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 20:23:58,559 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 20:23:58,561 INFO Regime phase LTF train: 27.7s
2026-04-26 20:23:58,717 INFO Regime LTF complete: acc=0.823, n=401471 per_class={'TRENDING': 0.801, 'RANGING': 0.77, 'CONSOLIDATING': 0.832, 'VOLATILE': 0.892}
2026-04-26 20:23:58,721 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:23:59,290 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 20:23:59,294 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 9.355291913830783, 'RANGING': 6.096303199751476, 'CONSOLIDATING': 5.598885793871866, 'VOLATILE': 6.771351107094442}
2026-04-26 20:23:59,303 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 29965, 'mean': 2.149062259083323e-05, 'mean_over_std': 0.011155601434349563}, 'RANGING': {'n': 19623, 'mean': -3.254493608872846e-06, 'mean_over_std': -0.001761415548292494}, 'CONSOLIDATING': {'n': 10050, 'mean': -3.039756158146958e-06, 'mean_over_std': -0.001792488091618151}, 'VOLATILE': {'n': 14985, 'mean': 1.8102764458593193e-05, 'mean_over_std': 0.006974696291378742}}
2026-04-26 20:23:59,304 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 26145, 'mean': 2.874435605138053e-05, 'mean_over_std': 0.01595450632691917}, 'RANGING': {'n': 7858, 'mean': 1.7829776785521727e-05, 'mean_over_std': 0.011851561804467463}, 'CONSOLIDATING': {'n': 8543, 'mean': 7.845231961250924e-06, 'mean_over_std': 0.004915839521549481}, 'VOLATILE': {'n': 13636, 'mean': 5.480114267306668e-06, 'mean_over_std': 0.0020378119917389375}}
2026-04-26 20:23:59,304 INFO Regime retrain total: 244.5s (504761 samples)
2026-04-26 20:23:59,318 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 20:23:59,318 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 20:23:59,318 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 20:23:59,318 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-26 20:23:59,319 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-26 20:23:59,319 INFO Retrain complete. Total wall-clock: 244.5s
2026-04-26 20:24:02,187 INFO Model regime: SUCCESS
2026-04-26 20:24:02,187 INFO --- Training gru ---
2026-04-26 20:24:02,187 INFO Running retrain --model gru
2026-04-26 20:24:02,580 INFO retrain environment: KAGGLE
2026-04-26 20:24:04,301 INFO Device: CUDA (2 GPU(s))
2026-04-26 20:24:04,312 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 20:24:04,312 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 20:24:04,313 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 20:24:04,313 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 20:24:04,314 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-26 20:24:04,470 INFO NumExpr defaulting to 4 threads.
2026-04-26 20:24:04,680 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 20:24:04,681 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 20:24:04,681 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 20:24:04,681 INFO GRU phase macro_correlations: 0.0s
2026-04-26 20:24:04,681 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-26 20:24:04,682 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260426_202404
2026-04-26 20:24:04,685 WARNING WeightsManifest: no manifest at /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json — assuming compatible (run a full retrain to generate one)
2026-04-26 20:24:04,842 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:04,863 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:04,879 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:04,887 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:04,888 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 20:24:04,888 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 20:24:04,888 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 20:24:04,889 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 20:24:04,890 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:04,982 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-26 20:24:04,984 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:05,243 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-26 20:24:05,275 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:05,566 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:05,708 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:05,822 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:06,045 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:06,064 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:06,079 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:06,087 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:06,088 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:06,170 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-26 20:24:06,172 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:06,426 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-26 20:24:06,444 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:06,726 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:06,867 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:06,980 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:07,205 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:07,228 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:07,246 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:07,255 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:07,256 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:07,341 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-26 20:24:07,343 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:07,600 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-26 20:24:07,617 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:07,912 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:08,059 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:08,168 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:08,407 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:08,428 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:08,444 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:08,452 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:08,453 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:08,539 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-26 20:24:08,541 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:08,831 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-26 20:24:08,853 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:09,158 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:09,295 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:09,405 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:09,615 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:09,637 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:09,653 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:09,661 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:09,662 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:09,747 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-26 20:24:09,750 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:10,004 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-26 20:24:10,021 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:10,315 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:10,458 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:10,569 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:10,779 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:10,798 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:10,814 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:10,821 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:10,822 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:10,911 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-26 20:24:10,913 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:11,169 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-26 20:24:11,187 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:11,485 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:11,628 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:11,743 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:11,932 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:24:11,951 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:24:11,967 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:24:11,974 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:24:11,975 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:12,061 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-26 20:24:12,063 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:12,318 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-26 20:24:12,332 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:12,621 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:12,758 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:12,867 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:13,069 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:13,088 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:13,103 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:13,110 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:13,111 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:13,194 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-26 20:24:13,196 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:13,457 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-26 20:24:13,477 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:13,770 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:13,914 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:14,028 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:14,235 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:14,254 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:14,270 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:14,277 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:14,278 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:14,362 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-26 20:24:14,364 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:14,617 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-26 20:24:14,634 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:14,939 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:15,092 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:15,208 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:15,416 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:15,437 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:15,453 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:15,461 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:24:15,462 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:15,546 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-26 20:24:15,548 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:15,801 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-26 20:24:15,817 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:16,123 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:16,263 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:16,380 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:24:16,705 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:24:16,732 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:24:16,751 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:24:16,761 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:24:16,762 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:24:16,933 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 20:24:16,937 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:24:17,495 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 20:24:17,543 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:24:18,151 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:24:18,374 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:24:18,518 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:24:18,646 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-26 20:24:18,963 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-04-26 20:24:18,964 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-04-26 20:24:18,964 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-26 20:24:18,964 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-26 20:25:12,746 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-26 20:25:12,746 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-26 20:25:14,146 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-26 20:25:18,458 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=5)
2026-04-26 20:25:33,790 INFO train_multi TF=ALL epoch 1/50 train=0.8572 val=0.8503
2026-04-26 20:25:33,800 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:25:33,800 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:25:33,800 INFO train_multi TF=ALL: new best val=0.8503 — saved
2026-04-26 20:25:46,804 INFO train_multi TF=ALL epoch 2/50 train=0.8403 val=0.8173
2026-04-26 20:25:46,809 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:25:46,809 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:25:46,809 INFO train_multi TF=ALL: new best val=0.8173 — saved
2026-04-26 20:25:59,890 INFO train_multi TF=ALL epoch 3/50 train=0.7435 val=0.6881
2026-04-26 20:25:59,895 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:25:59,896 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:25:59,896 INFO train_multi TF=ALL: new best val=0.6881 — saved
2026-04-26 20:26:13,007 INFO train_multi TF=ALL epoch 4/50 train=0.6898 val=0.6879
2026-04-26 20:26:13,012 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:26:13,012 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:26:13,012 INFO train_multi TF=ALL: new best val=0.6879 — saved
2026-04-26 20:26:26,045 INFO train_multi TF=ALL epoch 5/50 train=0.6894 val=0.6879
2026-04-26 20:26:26,050 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:26:26,050 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:26:26,050 INFO train_multi TF=ALL: new best val=0.6879 — saved
2026-04-26 20:26:39,242 INFO train_multi TF=ALL epoch 6/50 train=0.6891 val=0.6879
2026-04-26 20:26:52,392 INFO train_multi TF=ALL epoch 7/50 train=0.6887 val=0.6879
2026-04-26 20:27:05,527 INFO train_multi TF=ALL epoch 8/50 train=0.6886 val=0.6881
2026-04-26 20:27:18,703 INFO train_multi TF=ALL epoch 9/50 train=0.6884 val=0.6879
2026-04-26 20:27:31,896 INFO train_multi TF=ALL epoch 10/50 train=0.6884 val=0.6878
2026-04-26 20:27:31,901 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:27:31,901 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:27:31,901 INFO train_multi TF=ALL: new best val=0.6878 — saved
2026-04-26 20:27:45,042 INFO train_multi TF=ALL epoch 11/50 train=0.6883 val=0.6880
2026-04-26 20:27:58,023 INFO train_multi TF=ALL epoch 12/50 train=0.6879 val=0.6870
2026-04-26 20:27:58,028 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:27:58,028 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:27:58,028 INFO train_multi TF=ALL: new best val=0.6870 — saved
2026-04-26 20:28:11,310 INFO train_multi TF=ALL epoch 13/50 train=0.6863 val=0.6841
2026-04-26 20:28:11,315 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:28:11,315 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:28:11,315 INFO train_multi TF=ALL: new best val=0.6841 — saved
2026-04-26 20:28:24,548 INFO train_multi TF=ALL epoch 14/50 train=0.6776 val=0.6693
2026-04-26 20:28:24,552 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:28:24,552 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:28:24,552 INFO train_multi TF=ALL: new best val=0.6693 — saved
2026-04-26 20:28:37,712 INFO train_multi TF=ALL epoch 15/50 train=0.6621 val=0.6545
2026-04-26 20:28:37,717 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:28:37,717 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:28:37,717 INFO train_multi TF=ALL: new best val=0.6545 — saved
2026-04-26 20:28:50,812 INFO train_multi TF=ALL epoch 16/50 train=0.6480 val=0.6359
2026-04-26 20:28:50,816 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:28:50,817 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:28:50,817 INFO train_multi TF=ALL: new best val=0.6359 — saved
2026-04-26 20:29:03,887 INFO train_multi TF=ALL epoch 17/50 train=0.6389 val=0.6308
2026-04-26 20:29:03,892 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:29:03,892 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:29:03,892 INFO train_multi TF=ALL: new best val=0.6308 — saved
2026-04-26 20:29:16,824 INFO train_multi TF=ALL epoch 18/50 train=0.6328 val=0.6261
2026-04-26 20:29:16,829 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:29:16,829 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:29:16,829 INFO train_multi TF=ALL: new best val=0.6261 — saved
2026-04-26 20:29:29,792 INFO train_multi TF=ALL epoch 19/50 train=0.6287 val=0.6257
2026-04-26 20:29:29,798 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:29:29,798 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:29:29,798 INFO train_multi TF=ALL: new best val=0.6257 — saved
2026-04-26 20:29:42,926 INFO train_multi TF=ALL epoch 20/50 train=0.6254 val=0.6218
2026-04-26 20:29:42,931 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:29:42,931 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:29:42,931 INFO train_multi TF=ALL: new best val=0.6218 — saved
2026-04-26 20:29:56,035 INFO train_multi TF=ALL epoch 21/50 train=0.6218 val=0.6211
2026-04-26 20:29:56,040 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:29:56,040 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:29:56,040 INFO train_multi TF=ALL: new best val=0.6211 — saved
2026-04-26 20:30:09,044 INFO train_multi TF=ALL epoch 22/50 train=0.6199 val=0.6208
2026-04-26 20:30:09,049 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:30:09,049 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:30:09,049 INFO train_multi TF=ALL: new best val=0.6208 — saved
2026-04-26 20:30:22,050 INFO train_multi TF=ALL epoch 23/50 train=0.6173 val=0.6185
2026-04-26 20:30:22,055 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:30:22,055 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:30:22,055 INFO train_multi TF=ALL: new best val=0.6185 — saved
2026-04-26 20:30:35,062 INFO train_multi TF=ALL epoch 24/50 train=0.6155 val=0.6159
2026-04-26 20:30:35,067 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:30:35,068 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:30:35,068 INFO train_multi TF=ALL: new best val=0.6159 — saved
2026-04-26 20:30:48,182 INFO train_multi TF=ALL epoch 25/50 train=0.6132 val=0.6171
2026-04-26 20:31:01,244 INFO train_multi TF=ALL epoch 26/50 train=0.6112 val=0.6172
2026-04-26 20:31:14,431 INFO train_multi TF=ALL epoch 27/50 train=0.6091 val=0.6189
2026-04-26 20:31:27,477 INFO train_multi TF=ALL epoch 28/50 train=0.6077 val=0.6162
2026-04-26 20:31:40,604 INFO train_multi TF=ALL epoch 29/50 train=0.6055 val=0.6162
2026-04-26 20:31:40,604 INFO train_multi TF=ALL early stop at epoch 29
2026-04-26 20:31:40,759 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-26 20:31:40,759 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-26 20:31:40,759 INFO Retrain complete. Total wall-clock: 456.4s
2026-04-26 20:31:42,944 INFO Model gru: SUCCESS
2026-04-26 20:31:42,945 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:31:42,945 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 20:31:42,945 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 20:31:42,945 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-04-26 20:31:42,945 INFO   [DEFERRED] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip (expected after Round 1)
2026-04-26 20:31:42,945 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-04-26 20:31:42,945 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer', 'rl_ppo']
2026-04-26 20:31:42,946 INFO Saved 5 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  START Round 1 - Backtest (val)
2026-04-26 20:31:43,833 INFO === STEP 6: BACKTEST (round1) ===
2026-04-26 20:31:43,834 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-04-26 20:31:43,834 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-26 20:31:43,834 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
2026-04-26 20:31:46,445 WARNING QualityScorer unavailable (weights missing or load failed)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260426_203146.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)             3296  58.0%   3.03 2581.3% 58.0% 29.4%   9.0%    8.03
  gate_diagnostics: bars=468696 no_signal=74005 quality_block=0 session_skip=317895 density=43840 pm_reject=0 daily_skip=0 cooldown=29660 daily_halt_events=0 enforce_daily_halt=False
  no_signal_reasons: weak_gru_direction=37913, htf_bias_conflict=36053, range_side_conflict=27, trend_pullback_conflict=12

Calibration Summary:
  all          [OK] Calibration OK — p_win correlates with actual win rate.
  ml_trader    [OK] Calibration OK — p_win correlates with actual win rate.
2026-04-26 20:34:10,535 INFO Round 1 backtest — 3296 trades | avg WR=58.0% | avg PF=3.03 | avg Sharpe=8.04
2026-04-26 20:34:10,535 INFO   ml_trader: 3296 trades | WR=58.0% | PF=3.03 | Return=2581.3% | DD=9.0% | Sharpe=8.04
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 3296
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (3296 rows)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  AUDUSD        324 trades     1 days  324.00/day  [OVERTRADE]
  EURGBP        336 trades     1 days  336.00/day  [OVERTRADE]
  EURJPY        343 trades     1 days  343.00/day  [OVERTRADE]
  EURUSD        358 trades     1 days  358.00/day  [OVERTRADE]
  GBPJPY        346 trades     1 days  346.00/day  [OVERTRADE]
  GBPUSD        356 trades     1 days  356.00/day  [OVERTRADE]
  NZDUSD          1 trades     1 days   1.00/day
  USDCAD        302 trades     1 days  302.00/day  [OVERTRADE]
  USDCHF        306 trades     1 days  306.00/day  [OVERTRADE]
  USDJPY        288 trades     1 days  288.00/day  [OVERTRADE]
  XAUUSD        336 trades     1 days  336.00/day  [OVERTRADE]
  ⚠  AUDUSD: 324.00/day (>1.5)
  ⚠  EURGBP: 336.00/day (>1.5)
  ⚠  EURJPY: 343.00/day (>1.5)
  ⚠  EURUSD: 358.00/day (>1.5)
  ⚠  GBPJPY: 346.00/day (>1.5)
  ⚠  GBPUSD: 356.00/day (>1.5)
  ⚠  USDCAD: 302.00/day (>1.5)
  ⚠  USDCHF: 306.00/day (>1.5)
  ⚠  USDJPY: 288.00/day (>1.5)
  ⚠  XAUUSD: 336.00/day (>1.5)

──────────────────────────────────────────────────────────────
CHECK 2: REGIME DISTRIBUTION  (% of trades, WR, avgEV)
──────────────────────────────────────────────────────────────
  BIAS_DOWN           524 trades   15.9%  WR=68.5%  avgEV=0.000
  BIAS_NEUTRAL       1332 trades   40.4%  WR=57.5%  avgEV=0.000
  BIAS_UP            1440 trades   43.7%  WR=54.5%  avgEV=0.000
  ⚠  Regimes never traded: ['CONSOLIDATING', 'RANGING', 'TRENDING', 'VOLATILE']

──────────────────────────────────────────────────────────────
CHECK 3: EV PREDICTED vs REALIZED RR
──────────────────────────────────────────────────────────────
  Pearson  = +nan   Spearman = +0.0223

  Bucket                  N     AvgEV     AvgRR   WinRate
  Q1 (low EV)             0       n/a       n/a       n/a
  Q2                      0       n/a       n/a       n/a
  Q3                      0       n/a       n/a       n/a
  Q4 (high EV)         3296     0.000     0.818     57.9%

  Top-20% EV trades: n=3296  avgEV=0.0  avgRR=0.818  WR=57.9%

  Per-regime EV↔RR correlation:
  Regime                N    Pearson   Spearman       WR     AvgEV
  BIAS_DOWN           524       +nan    +0.0570   68.5%     0.000
  BIAS_NEUTRAL       1332       +nan    -0.0145   57.5%     0.000
  BIAS_UP            1440       +nan    +0.0182   54.5%     0.000
  ⚠  EV↔RR Spearman=0.022 < 0.15 — EV rankings don't predict outcomes
  ⚠  EV↔RR Spearman in BIAS_NEUTRAL = -0.015 — EV useless in this regime
  ⚠  EV↔RR Spearman in BIAS_UP = 0.018 — EV useless in this regime

──────────────────────────────────────────────────────────────
CHECK 4: GRU CONFIDENCE CALIBRATION  (p_bull/bear vs win rate)
──────────────────────────────────────────────────────────────
  ECE = 0.1550  (target < 0.10)
  Bin                   N   Midpoint    WinRate    Error
  [0.58-0.64]         237      0.612      0.489    0.123
  [0.64-0.71]         930      0.675      0.558    0.117
  [0.71-0.77]        1249      0.738      0.552    0.186
  [0.77-0.83]         548      0.802      0.628    0.174
  [0.83-0.90]         332      0.865      0.729    0.136
  ⚠  Bin [0.71-0.77]: midpoint=0.74 win_rate=0.55 (err=0.19 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.77-0.83]: midpoint=0.80 win_rate=0.63 (err=0.17 > 0.15) — GRU miscalibrated
  ⚠  Win rate non-monotonic across confidence bins — GRU confidence unreliable

──────────────────────────────────────────────────────────────
CHECK 5: GRU ↔ EV MODEL CONSISTENCY  (direction × value agreement)
──────────────────────────────────────────────────────────────
  Pearson=+nan  Spearman=-0.0714  Agree=51%

  Quadrants  (conf_threshold=median, ev_threshold=median):
  high_conf + high_ev:  1670  ← ideal
  high_conf + low_ev:      0  ← GRU overconfident
  low_conf  + high_ev:  1626  ← EV optimistic
  low_conf  + low_ev:      0  ← correct abstention
  ⚠  GRU and EV agree on only 50.7% of trades — models pulling in opposite directions

──────────────────────────────────────────────────────────────
SUMMARY — 18 flag(s):
  ⚠  AUDUSD: 324.00/day (>1.5)
  ⚠  EURGBP: 336.00/day (>1.5)
  ⚠  EURJPY: 343.00/day (>1.5)
  ⚠  EURUSD: 358.00/day (>1.5)
  ⚠  GBPJPY: 346.00/day (>1.5)
  ⚠  GBPUSD: 356.00/day (>1.5)
  ⚠  USDCAD: 302.00/day (>1.5)
  ⚠  USDCHF: 306.00/day (>1.5)
  ⚠  USDJPY: 288.00/day (>1.5)
  ⚠  XAUUSD: 336.00/day (>1.5)
  ⚠  Regimes never traded: ['CONSOLIDATING', 'RANGING', 'TRENDING', 'VOLATILE']
  ⚠  EV↔RR Spearman=0.022 < 0.15 — EV rankings don't predict outcomes
  ⚠  EV↔RR Spearman in BIAS_NEUTRAL = -0.015 — EV useless in this regime
  ⚠  EV↔RR Spearman in BIAS_UP = 0.018 — EV useless in this regime
  ⚠  Bin [0.71-0.77]: midpoint=0.74 win_rate=0.55 (err=0.19 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.77-0.83]: midpoint=0.80 win_rate=0.63 (err=0.17 > 0.15) — GRU miscalibrated
  ⚠  Win rate non-monotonic across confidence bins — GRU confidence unreliable
  ⚠  GRU and EV agree on only 50.7% of trades — models pulling in opposite directions
──────────────────────────────────────────────────────────────
2026-04-26 20:34:11,993 INFO Round 1: wrote 3296 journal entries (total in file: 3296)

======================================================================
  BACKTEST COMPLETE  (round 1 / window=round1)
======================================================================
  Round     Trades       WR      PF   Sharpe
  ------------------------------------------
  Round 1       3296     58.0%    3.029     8.035

  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 3296 entries

=== Round 1 → Retrain Quality + RL ===
  START Round 1 - Quality+RL retrain
2026-04-26 20:34:12,312 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-26 20:34:12,316 INFO Journal entries: 3296
2026-04-26 20:34:12,316 INFO --- Training quality ---
2026-04-26 20:34:12,316 INFO Running retrain --model quality
2026-04-26 20:34:12,503 INFO retrain environment: KAGGLE
2026-04-26 20:34:14,264 INFO Device: CUDA (2 GPU(s))
2026-04-26 20:34:14,276 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 20:34:14,276 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 20:34:14,276 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 20:34:14,277 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 20:34:14,278 INFO === QualityScorer retrain ===
2026-04-26 20:34:14,431 INFO NumExpr defaulting to 4 threads.
2026-04-26 20:34:14,633 INFO QualityScorer: CUDA available — using GPU
2026-04-26 20:34:14,849 INFO Quality phase label creation: 0.2s (3296 trades)
2026-04-26 20:34:15,068 INFO QualityScorer: 3296 samples, EV stats={'mean': 0.4344211220741272, 'std': 1.313375473022461, 'n_pos': 1910, 'n_neg': 1386}, device=cuda
2026-04-26 20:34:15,069 INFO QualityScorer: normalised win labels by median_win=2.000 — EV range now [-1, +3]
2026-04-26 20:34:15,293 INFO QualityScorer: DataParallel across 2 GPUs
2026-04-26 20:34:15,293 INFO QualityScorer: cold start
2026-04-26 20:34:15,294 INFO QualityScorer: pos_weight=0.73 (n_pos=1522 n_neg=1114)
2026-04-26 20:34:17,936 INFO Quality epoch   1/100 — va_huber=0.3667
2026-04-26 20:34:18,031 INFO Quality epoch   2/100 — va_huber=0.3625
2026-04-26 20:34:18,120 INFO Quality epoch   3/100 — va_huber=0.3590
2026-04-26 20:34:18,205 INFO Quality epoch   4/100 — va_huber=0.3615
2026-04-26 20:34:18,283 INFO Quality epoch   5/100 — va_huber=0.3691
2026-04-26 20:34:18,908 INFO Quality epoch  11/100 — va_huber=0.5305
2026-04-26 20:34:19,112 INFO Quality early stop at epoch 13
2026-04-26 20:34:19,130 INFO QualityScorer EV model: MAE=0.824 dir_acc=0.529 n_val=660
2026-04-26 20:34:19,135 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-26 20:34:19,200 INFO Quality phase train: 4.4s | total: 4.9s
2026-04-26 20:34:19,202 INFO Retrain complete. Total wall-clock: 4.9s
2026-04-26 20:34:20,367 INFO Model quality: SUCCESS
2026-04-26 20:34:20,367 INFO --- Training rl ---
2026-04-26 20:34:20,367 INFO Running retrain --model rl
2026-04-26 20:34:20,556 INFO retrain environment: KAGGLE
2026-04-26 20:34:22,321 INFO Device: CUDA (2 GPU(s))
2026-04-26 20:34:22,332 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 20:34:22,332 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 20:34:22,333 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 20:34:22,333 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 20:34:22,334 INFO === RLAgent (PPO) retrain ===
2026-04-26 20:34:22,340 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260426_203422
2026-04-26 20:34:22,432 INFO RL phase episode loading: 0.1s (3296 episodes)
2026-04-26 20:34:28.133983: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777235668.580902   39143 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777235668.705542   39143 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777235669.704914   39143 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777235669.704963   39143 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777235669.704967   39143 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777235669.704970   39143 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-26 20:34:48,770 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-26 20:34:52,581 INFO RLAgent: cold start — building new PPO policy
2026-04-26 20:36:05,111 INFO RLAgent: retrain complete, 3296 episodes
2026-04-26 20:36:05,111 INFO RL phase PPO train: 102.7s | total: 102.8s
2026-04-26 20:36:05,124 INFO Retrain complete. Total wall-clock: 102.8s
2026-04-26 20:36:07,049 INFO Model rl: SUCCESS
2026-04-26 20:36:07,050 INFO Step 7b complete — summary: /kaggle/working/Multi-Bot/trading-system/ml_training/metrics/training_7b_summary.json
  DONE  Round 1 - Quality+RL retrain

=== Round 2: BLIND backtest on test window (unseen 2yr) ===
  START Round 2 - Blind backtest (test)
2026-04-26 20:36:07,718 INFO === STEP 6: BACKTEST (round2) ===
2026-04-26 20:36:07,719 INFO BT_WINDOW=round2 — BLIND backtest: 2023-08-07 → 2025-08-05 (test set)
2026-04-26 20:36:07,719 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-26 20:36:07,720 INFO Round 2 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260426_203610.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)             2531  55.6%   2.85 1855.5% 55.6% 23.0%   3.5%    7.51
  gate_diagnostics: bars=482221 no_signal=54716 quality_block=5919 session_skip=361235 density=35048 pm_reject=0 daily_skip=0 cooldown=22772 daily_halt_events=0 enforce_daily_halt=False
  no_signal_reasons: weak_gru_direction=28898, htf_bias_conflict=25781, trend_pullback_conflict=27, range_side_conflict=10

Calibration Summary:
  all          [WARN] Non-monotonic calibration: 2/5 pairs violated. Consider retraining QualityScorer
  ml_trader    [WARN] Non-monotonic calibration: 2/5 pairs violated. Consider retraining QualityScorer
2026-04-26 20:38:41,847 INFO Round 2 backtest — 2531 trades | avg WR=55.6% | avg PF=2.85 | avg Sharpe=7.51
2026-04-26 20:38:41,848 INFO   ml_trader: 2531 trades | WR=55.6% | PF=2.85 | Return=1855.5% | DD=3.5% | Sharpe=7.51
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 2531
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2531 rows)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  AUDUSD        237 trades     2 days  118.50/day  [OVERTRADE]
  EURGBP        253 trades     2 days  126.50/day  [OVERTRADE]
  EURJPY        262 trades     2 days  131.00/day  [OVERTRADE]
  EURUSD        243 trades     2 days  121.50/day  [OVERTRADE]
  GBPJPY        258 trades     2 days  129.00/day  [OVERTRADE]
  GBPUSD        244 trades     2 days  122.00/day  [OVERTRADE]
  NZDUSD         14 trades     1 days  14.00/day  [OVERTRADE]
  USDCAD        256 trades     2 days  128.00/day  [OVERTRADE]
  USDCHF        239 trades     2 days  119.50/day  [OVERTRADE]
  USDJPY        273 trades     2 days  136.50/day  [OVERTRADE]
  XAUUSD        252 trades     2 days  126.00/day  [OVERTRADE]
  ⚠  AUDUSD: 118.50/day (>1.5)
  ⚠  EURGBP: 126.50/day (>1.5)
  ⚠  EURJPY: 131.00/day (>1.5)
  ⚠  EURUSD: 121.50/day (>1.5)
  ⚠  GBPJPY: 129.00/day (>1.5)
  ⚠  GBPUSD: 122.00/day (>1.5)
  ⚠  NZDUSD: 14.00/day (>1.5)
  ⚠  USDCAD: 128.00/day (>1.5)
  ⚠  USDCHF: 119.50/day (>1.5)
  ⚠  USDJPY: 136.50/day (>1.5)
  ⚠  XAUUSD: 126.00/day (>1.5)

──────────────────────────────────────────────────────────────
CHECK 2: REGIME DISTRIBUTION  (% of trades, WR, avgEV)
──────────────────────────────────────────────────────────────
  BIAS_DOWN           570 trades   22.5%  WR=65.8%  avgEV=0.150
  BIAS_NEUTRAL        761 trades   30.1%  WR=59.8%  avgEV=0.068
  BIAS_UP            1200 trades   47.4%  WR=48.1%  avgEV=0.018
  ⚠  Regimes never traded: ['CONSOLIDATING', 'RANGING', 'TRENDING', 'VOLATILE']

──────────────────────────────────────────────────────────────
CHECK 3: EV PREDICTED vs REALIZED RR
──────────────────────────────────────────────────────────────
  Pearson  = +0.1720   Spearman = +0.1428

  Bucket                  N     AvgEV     AvgRR   WinRate
  Q1 (low EV)           633     0.004     0.481     47.4%
  Q2                    632     0.016     0.577     50.3%
  Q3                    633     0.067     0.753     55.8%
  Q4 (high EV)          633     0.165     1.179     68.9%

  Top-20% EV trades: n=507  avgEV=0.174  avgRR=1.223  WR=69.2%

  Per-regime EV↔RR correlation:
  Regime                N    Pearson   Spearman       WR     AvgEV
  BIAS_DOWN           570    +0.1224    +0.1440   65.8%     0.150
  BIAS_NEUTRAL        761    +0.1509    +0.1362   59.8%     0.068
  BIAS_UP            1200    +0.0554    +0.0750   48.1%     0.018
  ⚠  EV↔RR Spearman=0.143 < 0.15 — EV rankings don't predict outcomes

──────────────────────────────────────────────────────────────
CHECK 4: GRU CONFIDENCE CALIBRATION  (p_bull/bear vs win rate)
──────────────────────────────────────────────────────────────
  ECE = 0.1891  (target < 0.10)
  Bin                   N   Midpoint    WinRate    Error
  [0.58-0.64]         176      0.612      0.449    0.163
  [0.64-0.71]         606      0.676      0.561    0
2026-04-26 20:38:43,061 INFO Round 2: wrote 2531 journal entries (total in file: 5827)
  DONE  Round 2 - Blind backtest (test)
  Saved Round 2 result → round2_summary.json
  Journal after Round 2: 5827 entries

=== Round 2 → Retrain Quality + RL (Round 1+2 journal) ===
  START Round 2 - Quality+RL retrain
2026-04-26 20:38:43,338 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-26 20:38:43,344 INFO Journal entries: 5827
2026-04-26 20:38:43,345 INFO --- Training quality ---
2026-04-26 20:38:43,345 INFO Running retrain --model quality
2026-04-26 20:38:43,534 INFO retrain environment: KAGGLE
2026-04-26 20:38:45,400 INFO Device: CUDA (2 GPU(s))
2026-04-26 20:38:45,412 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 20:38:45,413 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 20:38:45,413 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 20:38:45,413 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 20:38:45,414 INFO === QualityScorer retrain ===
2026-04-26 20:38:45,576 INFO NumExpr defaulting to 4 threads.
2026-04-26 20:38:45,788 INFO QualityScorer: CUDA available — using GPU
2026-04-26 20:38:46,001 INFO QualityScorer loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (device=cuda)
2026-04-26 20:38:46,413 INFO Quality phase label creation: 0.4s (5827 trades)
2026-04-26 20:38:46,414 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/quality_scorer.pkl_20260426_203846
2026-04-26 20:38:46,814 INFO QualityScorer: 5827 samples, EV stats={'mean': 0.38798898458480835, 'std': 1.2986980676651, 'n_pos': 3317, 'n_neg': 2510}, device=cuda
2026-04-26 20:38:46,815 INFO QualityScorer: normalised win labels by median_win=0.960 — EV range now [-1, +3]
2026-04-26 20:38:46,815 INFO QualityScorer: warm start from existing weights
2026-04-26 20:38:46,815 INFO QualityScorer: pos_weight=0.75 (n_pos=2671 n_neg=1990)
2026-04-26 20:38:49,484 INFO Quality epoch   1/100 — va_huber=0.5833
2026-04-26 20:38:49,637 INFO Quality epoch   2/100 — va_huber=0.5732
2026-04-26 20:38:49,771 INFO Quality epoch   3/100 — va_huber=0.5713
2026-04-26 20:38:49,902 INFO Quality epoch   4/100 — va_huber=0.5706
2026-04-26 20:38:50,056 INFO Quality epoch   5/100 — va_huber=0.5687
2026-04-26 20:38:50,928 INFO Quality epoch  11/100 — va_huber=0.5663
2026-04-26 20:38:52,324 INFO Quality epoch  21/100 — va_huber=0.5629
2026-04-26 20:38:53,661 INFO Quality epoch  31/100 — va_huber=0.5623
2026-04-26 20:38:55,239 INFO Quality epoch  41/100 — va_huber=0.5634
2026-04-26 20:38:55,508 INFO Quality early stop at epoch 43
2026-04-26 20:38:55,529 INFO QualityScorer EV model: MAE=1.139 dir_acc=0.610 n_val=1166
2026-04-26 20:38:55,534 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-26 20:38:55,613 INFO Quality phase train: 9.2s | total: 10.2s
2026-04-26 20:38:55,615 INFO Retrain complete. Total wall-clock: 10.2s
2026-04-26 20:38:56,861 INFO Model quality: SUCCESS
2026-04-26 20:38:56,861 INFO --- Training rl ---
2026-04-26 20:38:56,861 INFO Running retrain --model rl
2026-04-26 20:38:57,053 INFO retrain environment: KAGGLE
2026-04-26 20:38:58,916 INFO Device: CUDA (2 GPU(s))
2026-04-26 20:38:58,928 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 20:38:58,928 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 20:38:58,928 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 20:38:58,928 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 20:38:58,930 INFO === RLAgent (PPO) retrain ===
2026-04-26 20:38:58,931 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260426_203858
2026-04-26 20:38:59,095 INFO RL phase episode loading: 0.2s (5827 episodes)
2026-04-26 20:39:00.076982: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777235940.102679   40383 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777235940.110764   40383 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777235940.131637   40383 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777235940.131677   40383 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777235940.131680   40383 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777235940.131682   40383 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-26 20:39:04,989 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-26 20:39:08,207 INFO RLAgent: cold start — building new PPO policy
2026-04-26 20:41:14,174 INFO RLAgent: retrain complete, 5827 episodes
2026-04-26 20:41:14,174 INFO RL phase PPO train: 135.1s | total: 135.2s
2026-04-26 20:41:14,195 INFO Retrain complete. Total wall-clock: 135.3s
2026-04-26 20:41:16,096 INFO Model rl: SUCCESS
2026-04-26 20:41:16,097 INFO Step 7b complete — summary: /kaggle/working/Multi-Bot/trading-system/ml_training/metrics/training_7b_summary.json
  DONE  Round 2 - Quality+RL retrain

=== Round 3: Incremental retrain ALL models on full data ===
  START Retrain gru [full-data retrain]
2026-04-26 20:41:16,494 INFO retrain environment: KAGGLE
2026-04-26 20:41:18,272 INFO Device: CUDA (2 GPU(s))
2026-04-26 20:41:18,283 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 20:41:18,283 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 20:41:18,283 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 20:41:18,283 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 20:41:18,285 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-26 20:41:18,457 INFO NumExpr defaulting to 4 threads.
2026-04-26 20:41:18,711 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 20:41:18,711 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 20:41:18,711 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 20:41:18,976 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-26 20:41:18,976 INFO GRU phase macro_correlations: 0.0s
2026-04-26 20:41:18,977 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-26 20:41:18,979 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260426_204118
2026-04-26 20:41:18,983 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-04-26 20:41:19,139 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:19,159 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:19,175 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:19,183 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:19,184 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 20:41:19,184 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 20:41:19,184 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 20:41:19,185 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 20:41:19,186 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:19,275 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-26 20:41:19,277 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:19,550 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-26 20:41:19,582 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:19,880 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:20,018 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:20,121 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:20,325 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:20,342 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:20,356 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:20,364 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:20,366 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:20,456 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-26 20:41:20,458 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:20,733 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-26 20:41:20,750 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:21,063 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:21,219 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:21,329 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:21,527 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:21,549 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:21,564 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:21,575 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:21,576 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:21,664 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-26 20:41:21,666 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:21,928 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-26 20:41:21,945 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:22,234 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:22,394 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:22,532 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:22,772 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:22,794 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:22,811 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:22,819 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:22,820 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:22,907 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-26 20:41:22,909 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:23,173 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-26 20:41:23,199 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:23,499 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:23,644 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:23,756 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:23,957 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:23,979 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:23,995 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:24,003 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:24,004 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:24,090 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-26 20:41:24,092 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:24,367 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-26 20:41:24,384 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:24,724 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:24,890 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:25,012 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:25,224 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:25,245 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:25,261 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:25,268 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:25,269 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:25,355 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-26 20:41:25,357 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:25,619 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-26 20:41:25,634 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:25,915 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:26,068 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:26,207 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:26,414 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:41:26,433 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:41:26,449 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:41:26,456 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:41:26,457 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:26,546 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-26 20:41:26,548 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:26,822 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-26 20:41:26,836 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:27,139 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:27,285 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:27,388 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:27,578 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:27,600 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:27,614 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:27,621 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:27,622 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:27,707 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-26 20:41:27,709 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:27,968 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-26 20:41:27,988 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:28,336 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:28,504 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:28,624 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:28,870 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:28,890 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:28,907 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:28,915 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:28,916 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:29,004 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-26 20:41:29,006 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:29,268 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-26 20:41:29,284 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:29,576 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:29,727 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:29,850 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:30,093 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:30,115 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:30,133 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:30,142 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:41:30,143 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:30,235 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-26 20:41:30,237 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:30,509 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-26 20:41:30,525 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:30,848 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:30,991 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:31,106 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:41:31,413 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:41:31,438 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:41:31,453 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:41:31,464 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:41:31,465 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:41:31,629 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 20:41:31,632 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:41:32,198 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 20:41:32,249 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:41:32,845 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:41:33,055 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:41:33,192 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:41:33,311 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-26 20:41:33,312 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-26 20:41:33,312 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-26 20:42:28,064 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-26 20:42:28,065 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-26 20:42:29,465 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-26 20:42:33,951 INFO train_multi TF=ALL: warm-start detected — using CosineAnnealingLR (lr=3e-05, patience=12)
2026-04-26 20:42:48,730 INFO train_multi TF=ALL epoch 1/50 train=0.6133 val=0.6168
2026-04-26 20:42:48,736 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:42:48,736 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:42:48,737 INFO train_multi TF=ALL: new best val=0.6168 — saved
2026-04-26 20:43:01,551 INFO train_multi TF=ALL epoch 2/50 train=0.6127 val=0.6174
2026-04-26 20:43:14,321 INFO train_multi TF=ALL epoch 3/50 train=0.6123 val=0.6176
2026-04-26 20:43:27,084 INFO train_multi TF=ALL epoch 4/50 train=0.6118 val=0.6175
2026-04-26 20:43:39,762 INFO train_multi TF=ALL epoch 5/50 train=0.6114 val=0.6182
2026-04-26 20:43:52,580 INFO train_multi TF=ALL epoch 6/50 train=0.6110 val=0.6184
2026-04-26 20:44:05,507 INFO train_multi TF=ALL epoch 7/50 train=0.6110 val=0.6173
2026-04-26 20:44:18,329 INFO train_multi TF=ALL epoch 8/50 train=0.6105 val=0.6176
2026-04-26 20:44:31,213 INFO train_multi TF=ALL epoch 9/50 train=0.6101 val=0.6183
2026-04-26 20:44:44,271 INFO train_multi TF=ALL epoch 10/50 train=0.6099 val=0.6177
2026-04-26 20:44:57,403 INFO train_multi TF=ALL epoch 11/50 train=0.6098 val=0.6183
2026-04-26 20:45:10,376 INFO train_multi TF=ALL epoch 12/50 train=0.6095 val=0.6166
2026-04-26 20:45:10,381 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:45:10,381 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:45:10,381 INFO train_multi TF=ALL: new best val=0.6166 — saved
2026-04-26 20:45:23,288 INFO train_multi TF=ALL epoch 13/50 train=0.6092 val=0.6173
2026-04-26 20:45:36,103 INFO train_multi TF=ALL epoch 14/50 train=0.6088 val=0.6167
2026-04-26 20:45:48,746 INFO train_multi TF=ALL epoch 15/50 train=0.6088 val=0.6166
2026-04-26 20:45:48,751 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:45:48,751 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:45:48,751 INFO train_multi TF=ALL: new best val=0.6166 — saved
2026-04-26 20:46:01,280 INFO train_multi TF=ALL epoch 16/50 train=0.6085 val=0.6175
2026-04-26 20:46:13,916 INFO train_multi TF=ALL epoch 17/50 train=0.6082 val=0.6170
2026-04-26 20:46:26,477 INFO train_multi TF=ALL epoch 18/50 train=0.6078 val=0.6168
2026-04-26 20:46:39,167 INFO train_multi TF=ALL epoch 19/50 train=0.6075 val=0.6175
2026-04-26 20:46:51,901 INFO train_multi TF=ALL epoch 20/50 train=0.6075 val=0.6158
2026-04-26 20:46:51,905 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:46:51,905 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:46:51,905 INFO train_multi TF=ALL: new best val=0.6158 — saved
2026-04-26 20:47:04,493 INFO train_multi TF=ALL epoch 21/50 train=0.6074 val=0.6162
2026-04-26 20:47:17,264 INFO train_multi TF=ALL epoch 22/50 train=0.6073 val=0.6165
2026-04-26 20:47:30,238 INFO train_multi TF=ALL epoch 23/50 train=0.6070 val=0.6161
2026-04-26 20:47:42,930 INFO train_multi TF=ALL epoch 24/50 train=0.6068 val=0.6158
2026-04-26 20:47:55,890 INFO train_multi TF=ALL epoch 25/50 train=0.6062 val=0.6160
2026-04-26 20:48:08,582 INFO train_multi TF=ALL epoch 26/50 train=0.6063 val=0.6155
2026-04-26 20:48:08,587 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:48:08,587 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:48:08,587 INFO train_multi TF=ALL: new best val=0.6155 — saved
2026-04-26 20:48:21,383 INFO train_multi TF=ALL epoch 27/50 train=0.6060 val=0.6157
2026-04-26 20:48:34,045 INFO train_multi TF=ALL epoch 28/50 train=0.6062 val=0.6158
2026-04-26 20:48:48,025 INFO train_multi TF=ALL epoch 29/50 train=0.6059 val=0.6157
2026-04-26 20:49:01,015 INFO train_multi TF=ALL epoch 30/50 train=0.6056 val=0.6154
2026-04-26 20:49:01,019 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:49:01,019 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:49:01,019 INFO train_multi TF=ALL: new best val=0.6154 — saved
2026-04-26 20:49:12,840 INFO train_multi TF=ALL epoch 31/50 train=0.6057 val=0.6157
2026-04-26 20:49:24,644 INFO train_multi TF=ALL epoch 32/50 train=0.6056 val=0.6156
2026-04-26 20:49:36,481 INFO train_multi TF=ALL epoch 33/50 train=0.6056 val=0.6158
2026-04-26 20:49:48,343 INFO train_multi TF=ALL epoch 34/50 train=0.6057 val=0.6158
2026-04-26 20:50:00,412 INFO train_multi TF=ALL epoch 35/50 train=0.6054 val=0.6155
2026-04-26 20:50:12,319 INFO train_multi TF=ALL epoch 36/50 train=0.6050 val=0.6155
2026-04-26 20:50:24,154 INFO train_multi TF=ALL epoch 37/50 train=0.6052 val=0.6155
2026-04-26 20:50:35,938 INFO train_multi TF=ALL epoch 38/50 train=0.6053 val=0.6154
2026-04-26 20:50:35,942 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:50:35,942 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:50:35,942 INFO train_multi TF=ALL: new best val=0.6154 — saved
2026-04-26 20:50:47,718 INFO train_multi TF=ALL epoch 39/50 train=0.6051 val=0.6154
2026-04-26 20:50:59,638 INFO train_multi TF=ALL epoch 40/50 train=0.6054 val=0.6155
2026-04-26 20:51:11,439 INFO train_multi TF=ALL epoch 41/50 train=0.6049 val=0.6155
2026-04-26 20:51:23,179 INFO train_multi TF=ALL epoch 42/50 train=0.6050 val=0.6153
2026-04-26 20:51:23,184 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:51:23,184 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:51:23,184 INFO train_multi TF=ALL: new best val=0.6153 — saved
2026-04-26 20:51:34,952 INFO train_multi TF=ALL epoch 43/50 train=0.6052 val=0.6154
2026-04-26 20:51:46,750 INFO train_multi TF=ALL epoch 44/50 train=0.6051 val=0.6154
2026-04-26 20:51:58,545 INFO train_multi TF=ALL epoch 45/50 train=0.6049 val=0.6154
2026-04-26 20:52:10,370 INFO train_multi TF=ALL epoch 46/50 train=0.6049 val=0.6153
2026-04-26 20:52:22,104 INFO train_multi TF=ALL epoch 47/50 train=0.6047 val=0.6154
2026-04-26 20:52:33,970 INFO train_multi TF=ALL epoch 48/50 train=0.6047 val=0.6154
2026-04-26 20:52:45,846 INFO train_multi TF=ALL epoch 49/50 train=0.6046 val=0.6153
2026-04-26 20:52:57,634 INFO train_multi TF=ALL epoch 50/50 train=0.6049 val=0.6153
2026-04-26 20:52:57,639 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 20:52:57,639 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 20:52:57,639 INFO train_multi TF=ALL: new best val=0.6153 — saved
2026-04-26 20:52:57,781 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-26 20:52:57,781 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-26 20:52:57,782 INFO Retrain complete. Total wall-clock: 699.5s
  DONE  Retrain gru [full-data retrain]
  START Retrain regime [full-data retrain]
2026-04-26 20:53:00,111 INFO retrain environment: KAGGLE
2026-04-26 20:53:01,703 INFO Device: CUDA (2 GPU(s))
2026-04-26 20:53:01,711 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 20:53:01,712 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 20:53:01,712 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 20:53:01,712 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 20:53:01,713 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-26 20:53:01,856 INFO NumExpr defaulting to 4 threads.
2026-04-26 20:53:02,046 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 20:53:02,047 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 20:53:02,047 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 20:53:02,047 INFO Regime phase macro_correlations: 0.0s
2026-04-26 20:53:02,047 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-26 20:53:02,086 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 20:53:02,086 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:53:02,115 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:53:02,129 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:53:02,154 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:53:02,168 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:53:02,192 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:53:02,207 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:53:02,230 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:53:02,245 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:53:02,269 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:53:02,283 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:53:02,303 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:53:02,317 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:53:02,336 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:53:02,350 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:53:02,371 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:53:02,385 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:53:02,406 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:53:02,419 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:53:02,441 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:53:02,457 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:53:02,495 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:53:03,245 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 20:53:27,028 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 20:53:27,033 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 24.5s
2026-04-26 20:53:27,033 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 20:53:36,953 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 20:53:36,955 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 9.9s
2026-04-26 20:53:36,955 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 20:53:45,055 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 20:53:45,058 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 8.1s
2026-04-26 20:53:45,058 INFO Regime phase GMM HTF total: 42.6s
2026-04-26 20:53:45,058 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 20:54:56,917 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 20:54:56,921 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 71.9s
2026-04-26 20:54:56,922 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 20:55:28,457 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 20:55:28,458 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 31.5s
2026-04-26 20:55:28,458 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 20:55:50,534 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 20:55:50,536 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 22.1s
2026-04-26 20:55:50,536 INFO Regime phase GMM LTF total: 125.5s
2026-04-26 20:55:50,641 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-26 20:55:50,643 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:55:50,644 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:55:50,645 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:55:50,646 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:55:50,647 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:55:50,648 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:55:50,649 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:55:50,650 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:55:50,652 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:55:50,653 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:55:50,654 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:55:50,780 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:50,826 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:50,827 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:50,827 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:50,835 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:50,836 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:55:51,246 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-26 20:55:51,247 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-26 20:55:51,423 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:51,455 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:51,456 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:51,457 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:51,465 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:51,465 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:55:51,837 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-26 20:55:51,838 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-26 20:55:52,026 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:52,061 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:52,062 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:52,063 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:52,071 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:52,072 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:55:52,438 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-26 20:55:52,439 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-26 20:55:52,617 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:52,650 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:52,651 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:52,651 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:52,659 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:52,660 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:55:53,026 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-26 20:55:53,027 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-26 20:55:53,200 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:53,237 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:53,238 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:53,239 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:53,247 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:53,248 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:55:53,627 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-26 20:55:53,628 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-26 20:55:53,807 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:53,839 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:53,840 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:53,841 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:53,848 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:53,849 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:55:54,222 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-26 20:55:54,223 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-26 20:55:54,379 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:55:54,407 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:55:54,408 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:55:54,408 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:55:54,415 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:55:54,416 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:55:54,784 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-26 20:55:54,785 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-26 20:55:54,950 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:54,983 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:54,984 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:54,984 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:54,992 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:54,993 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:55:55,363 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-26 20:55:55,364 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-26 20:55:55,536 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:55,568 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:55,569 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:55,569 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:55,577 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:55,578 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:55:55,950 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-26 20:55:55,951 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-26 20:55:56,121 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:56,156 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:56,156 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:56,157 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:56,165 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:55:56,166 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:55:56,536 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-26 20:55:56,537 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-26 20:55:56,793 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:55:56,851 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:55:56,852 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:55:56,853 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:55:56,862 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:55:56,864 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:55:57,657 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 20:55:57,658 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-26 20:55:57,822 INFO Regime phase HTF dataset build: 7.2s (103290 samples)
2026-04-26 20:55:57,823 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260426_205557
2026-04-26 20:55:58,026 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=34, n_classes=3)
2026-04-26 20:55:58,027 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=38114 dropped=65176 classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728})
2026-04-26 20:55:58,036 INFO RegimeClassifier[mode=htf_bias]: 38114 samples, classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728}, device=cuda
2026-04-26 20:55:58,037 INFO RegimeClassifier: sample weights — mean=0.708  ambiguous(<0.4)=0.0%
2026-04-26 20:55:58,037 INFO RegimeClassifier[mode=htf_bias]: warm start from existing weights
2026-04-26 20:55:58,037 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 20:56:00,339 INFO Regime epoch  1/50 — tr=0.4804 va=1.1020 acc=0.974 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.859}
2026-04-26 20:56:00,413 INFO Regime epoch  2/50 — tr=0.4800 va=1.0997 acc=0.975
2026-04-26 20:56:00,481 INFO Regime epoch  3/50 — tr=0.4800 va=1.1017 acc=0.975
2026-04-26 20:56:00,552 INFO Regime epoch  4/50 — tr=0.4793 va=1.1025 acc=0.975
2026-04-26 20:56:00,628 INFO Regime epoch  5/50 — tr=0.4802 va=1.1012 acc=0.974 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.859}
2026-04-26 20:56:00,704 INFO Regime epoch  6/50 — tr=0.4795 va=1.0993 acc=0.975
2026-04-26 20:56:00,772 INFO Regime epoch  7/50 — tr=0.4789 va=1.0971 acc=0.976
2026-04-26 20:56:00,838 INFO Regime epoch  8/50 — tr=0.4792 va=1.0934 acc=0.977
2026-04-26 20:56:00,905 INFO Regime epoch  9/50 — tr=0.4784 va=1.0917 acc=0.977
2026-04-26 20:56:00,981 INFO Regime epoch 10/50 — tr=0.4784 va=1.0895 acc=0.977 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.873}
2026-04-26 20:56:01,049 INFO Regime epoch 11/50 — tr=0.4774 va=1.0856 acc=0.977
2026-04-26 20:56:01,120 INFO Regime epoch 12/50 — tr=0.4770 va=1.0846 acc=0.978
2026-04-26 20:56:01,194 INFO Regime epoch 13/50 — tr=0.4763 va=1.0818 acc=0.978
2026-04-26 20:56:01,264 INFO Regime epoch 14/50 — tr=0.4758 va=1.0810 acc=0.978
2026-04-26 20:56:01,340 INFO Regime epoch 15/50 — tr=0.4754 va=1.0800 acc=0.978 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.88}
2026-04-26 20:56:01,409 INFO Regime epoch 16/50 — tr=0.4754 va=1.0741 acc=0.979
2026-04-26 20:56:01,477 INFO Regime epoch 17/50 — tr=0.4748 va=1.0702 acc=0.980
2026-04-26 20:56:01,549 INFO Regime epoch 18/50 — tr=0.4744 va=1.0686 acc=0.980
2026-04-26 20:56:01,616 INFO Regime epoch 19/50 — tr=0.4738 va=1.0683 acc=0.980
2026-04-26 20:56:01,691 INFO Regime epoch 20/50 — tr=0.4739 va=1.0646 acc=0.981 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.896}
2026-04-26 20:56:01,759 INFO Regime epoch 21/50 — tr=0.4733 va=1.0646 acc=0.981
2026-04-26 20:56:01,827 INFO Regime epoch 22/50 — tr=0.4731 va=1.0630 acc=0.982
2026-04-26 20:56:01,893 INFO Regime epoch 23/50 — tr=0.4725 va=1.0615 acc=0.982
2026-04-26 20:56:01,959 INFO Regime epoch 24/50 — tr=0.4726 va=1.0600 acc=0.983
2026-04-26 20:56:02,032 INFO Regime epoch 25/50 — tr=0.4720 va=1.0578 acc=0.983 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.905}
2026-04-26 20:56:02,099 INFO Regime epoch 26/50 — tr=0.4719 va=1.0562 acc=0.983
2026-04-26 20:56:02,165 INFO Regime epoch 27/50 — tr=0.4719 va=1.0522 acc=0.984
2026-04-26 20:56:02,232 INFO Regime epoch 28/50 — tr=0.4715 va=1.0546 acc=0.983
2026-04-26 20:56:02,298 INFO Regime epoch 29/50 — tr=0.4713 va=1.0523 acc=0.983
2026-04-26 20:56:02,369 INFO Regime epoch 30/50 — tr=0.4715 va=1.0512 acc=0.984 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.91}
2026-04-26 20:56:02,435 INFO Regime epoch 31/50 — tr=0.4713 va=1.0494 acc=0.984
2026-04-26 20:56:02,502 INFO Regime epoch 32/50 — tr=0.4711 va=1.0501 acc=0.984
2026-04-26 20:56:02,567 INFO Regime epoch 33/50 — tr=0.4713 va=1.0513 acc=0.984
2026-04-26 20:56:02,632 INFO Regime epoch 34/50 — tr=0.4712 va=1.0504 acc=0.984
2026-04-26 20:56:02,705 INFO Regime epoch 35/50 — tr=0.4705 va=1.0489 acc=0.984 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.912}
2026-04-26 20:56:02,772 INFO Regime epoch 36/50 — tr=0.4706 va=1.0491 acc=0.984
2026-04-26 20:56:02,838 INFO Regime epoch 37/50 — tr=0.4707 va=1.0483 acc=0.984
2026-04-26 20:56:02,904 INFO Regime epoch 38/50 — tr=0.4706 va=1.0479 acc=0.984
2026-04-26 20:56:02,969 INFO Regime epoch 39/50 — tr=0.4703 va=1.0480 acc=0.984
2026-04-26 20:56:03,040 INFO Regime epoch 40/50 — tr=0.4706 va=1.0462 acc=0.985 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.915}
2026-04-26 20:56:03,106 INFO Regime epoch 41/50 — tr=0.4705 va=1.0489 acc=0.984
2026-04-26 20:56:03,171 INFO Regime epoch 42/50 — tr=0.4702 va=1.0497 acc=0.984
2026-04-26 20:56:03,237 INFO Regime epoch 43/50 — tr=0.4707 va=1.0479 acc=0.985
2026-04-26 20:56:03,302 INFO Regime epoch 44/50 — tr=0.4703 va=1.0465 acc=0.985
2026-04-26 20:56:03,372 INFO Regime epoch 45/50 — tr=0.4704 va=1.0456 acc=0.984 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.912}
2026-04-26 20:56:03,439 INFO Regime epoch 46/50 — tr=0.4708 va=1.0464 acc=0.984
2026-04-26 20:56:03,506 INFO Regime epoch 47/50 — tr=0.4700 va=1.0481 acc=0.984
2026-04-26 20:56:03,572 INFO Regime epoch 48/50 — tr=0.4700 va=1.0465 acc=0.985
2026-04-26 20:56:03,639 INFO Regime epoch 49/50 — tr=0.4703 va=1.0460 acc=0.984
2026-04-26 20:56:03,711 INFO Regime epoch 50/50 — tr=0.4706 va=1.0458 acc=0.985 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.915}
2026-04-26 20:56:03,721 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 20:56:03,721 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 20:56:03,721 INFO Regime phase HTF train: 5.7s
2026-04-26 20:56:03,848 INFO Regime HTF complete: acc=0.984, n=103290 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.912}
2026-04-26 20:56:03,850 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:56:04,003 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 20:56:04,006 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 11.41578947368421, 'BIAS_DOWN': 10.635761589403973, 'BIAS_NEUTRAL': 17.960468521229867}
2026-04-26 20:56:04,010 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 4338, 'mean': 0.00011110466582543088, 'mean_over_std': 0.027632581210502927}, 'BIAS_DOWN': {'n': 3212, 'mean': 6.063430222660252e-05, 'mean_over_std': 0.013898489590377157}, 'BIAS_NEUTRAL': {'n': 12266, 'mean': 1.2151554902814046e-05, 'mean_over_std': 0.0031957000513162357}}
2026-04-26 20:56:04,011 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 3651, 'mean': 0.0002852242108836959, 'mean_over_std': 0.07015017232491755}, 'BIAS_DOWN': {'n': 2644, 'mean': -0.00014900749618981248, 'mean_over_std': -0.03333037041298237}, 'BIAS_NEUTRAL': {'n': 1376, 'mean': 9.595797941249082e-05, 'mean_over_std': 0.02492411099521803}}
2026-04-26 20:56:04,011 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-26 20:56:04,013 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:56:04,015 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:56:04,017 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:56:04,018 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:56:04,020 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:56:04,021 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:56:04,023 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:56:04,024 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:56:04,026 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:56:04,027 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:56:04,030 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:56:04,040 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:04,044 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:04,045 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:04,045 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:04,045 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:04,047 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:56:04,670 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-26 20:56:04,673 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-26 20:56:04,808 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:04,810 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:04,811 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:04,811 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:04,812 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:04,814 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:56:05,408 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-26 20:56:05,411 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-26 20:56:05,543 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:05,545 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:05,546 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:05,547 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:05,547 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:05,549 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:56:06,140 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-26 20:56:06,144 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-26 20:56:06,273 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:06,278 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:06,278 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:06,279 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:06,279 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:06,281 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:56:06,875 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-26 20:56:06,878 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-26 20:56:07,012 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:07,015 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:07,015 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:07,016 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:07,016 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:07,018 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:56:07,599 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-26 20:56:07,602 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-26 20:56:07,742 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:07,744 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:07,745 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:07,745 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:07,746 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:07,748 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:56:08,325 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-26 20:56:08,328 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-26 20:56:08,463 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:56:08,465 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:56:08,465 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:56:08,466 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:56:08,466 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 20:56:08,468 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:56:09,072 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-26 20:56:09,074 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-26 20:56:09,205 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:09,208 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:09,208 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:09,209 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:09,209 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:09,211 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:56:09,793 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-26 20:56:09,795 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-26 20:56:09,928 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:09,931 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:09,931 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:09,932 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:09,932 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:09,934 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:56:10,512 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-26 20:56:10,514 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-26 20:56:10,649 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:10,651 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:10,652 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:10,653 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:10,653 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 20:56:10,655 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 20:56:11,239 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-26 20:56:11,242 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-26 20:56:11,382 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:56:11,385 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:56:11,387 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:56:11,387 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:56:11,387 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 20:56:11,391 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:56:12,621 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 20:56:12,627 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-26 20:56:12,917 INFO Regime phase LTF dataset build: 8.9s (401471 samples)
2026-04-26 20:56:12,918 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260426_205612
2026-04-26 20:56:12,922 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=18, n_classes=4)
2026-04-26 20:56:12,925 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=299185 dropped=102286 classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127})
2026-04-26 20:56:12,990 INFO RegimeClassifier[mode=ltf_behaviour]: 299185 samples, classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127}, device=cuda
2026-04-26 20:56:12,991 INFO RegimeClassifier: sample weights — mean=0.693  ambiguous(<0.4)=0.0%
2026-04-26 20:56:12,991 INFO RegimeClassifier[mode=ltf_behaviour]: warm start from existing weights
2026-04-26 20:56:12,992 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 20:56:13,543 INFO Regime epoch  1/50 — tr=0.6261 va=1.2468 acc=0.825 per_class={'TRENDING': 0.803, 'RANGING': 0.775, 'CONSOLIDATING': 0.817, 'VOLATILE': 0.901}
2026-04-26 20:56:14,050 INFO Regime epoch  2/50 — tr=0.6260 va=1.2458 acc=0.824
2026-04-26 20:56:14,537 INFO Regime epoch  3/50 — tr=0.6263 va=1.2425 acc=0.822
2026-04-26 20:56:15,030 INFO Regime epoch  4/50 — tr=0.6260 va=1.2445 acc=0.822
2026-04-26 20:56:15,566 INFO Regime epoch  5/50 — tr=0.6259 va=1.2460 acc=0.824 per_class={'TRENDING': 0.799, 'RANGING': 0.775, 'CONSOLIDATING': 0.812, 'VOLATILE': 0.907}
2026-04-26 20:56:16,057 INFO Regime epoch  6/50 — tr=0.6255 va=1.2465 acc=0.826
2026-04-26 20:56:16,542 INFO Regime epoch  7/50 — tr=0.6258 va=1.2444 acc=0.825
2026-04-26 20:56:17,025 INFO Regime epoch  8/50 — tr=0.6255 va=1.2434 acc=0.827
2026-04-26 20:56:17,508 INFO Regime epoch  9/50 — tr=0.6250 va=1.2452 acc=0.826
2026-04-26 20:56:18,033 INFO Regime epoch 10/50 — tr=0.6250 va=1.2396 acc=0.827 per_class={'TRENDING': 0.806, 'RANGING': 0.774, 'CONSOLIDATING': 0.827, 'VOLATILE': 0.898}
2026-04-26 20:56:18,521 INFO Regime epoch 11/50 — tr=0.6247 va=1.2401 acc=0.826
2026-04-26 20:56:19,038 INFO Regime epoch 12/50 — tr=0.6244 va=1.2408 acc=0.828
2026-04-26 20:56:19,536 INFO Regime epoch 13/50 — tr=0.6240 va=1.2378 acc=0.828
2026-04-26 20:56:20,039 INFO Regime epoch 14/50 — tr=0.6241 va=1.2391 acc=0.828
2026-04-26 20:56:20,571 INFO Regime epoch 15/50 — tr=0.6238 va=1.2344 acc=0.829 per_class={'TRENDING': 0.809, 'RANGING': 0.773, 'CONSOLIDATING': 0.836, 'VOLATILE': 0.896}
2026-04-26 20:56:21,074 INFO Regime epoch 16/50 — tr=0.6231 va=1.2392 acc=0.832
2026-04-26 20:56:21,578 INFO Regime epoch 17/50 — tr=0.6233 va=1.2375 acc=0.834
2026-04-26 20:56:22,078 INFO Regime epoch 18/50 — tr=0.6231 va=1.2315 acc=0.830
2026-04-26 20:56:22,567 INFO Regime epoch 19/50 — tr=0.6231 va=1.2353 acc=0.832
2026-04-26 20:56:23,120 INFO Regime epoch 20/50 — tr=0.6228 va=1.2330 acc=0.833 per_class={'TRENDING': 0.815, 'RANGING': 0.777, 'CONSOLIDATING': 0.839, 'VOLATILE': 0.895}
2026-04-26 20:56:23,632 INFO Regime epoch 21/50 — tr=0.6229 va=1.2302 acc=0.831
2026-04-26 20:56:24,152 INFO Regime epoch 22/50 — tr=0.6226 va=1.2359 acc=0.834
2026-04-26 20:56:24,635 INFO Regime epoch 23/50 — tr=0.6224 va=1.2330 acc=0.834
2026-04-26 20:56:25,142 INFO Regime epoch 24/50 — tr=0.6222 va=1.2337 acc=0.834
2026-04-26 20:56:25,686 INFO Regime epoch 25/50 — tr=0.6222 va=1.2317 acc=0.833 per_class={'TRENDING': 0.813, 'RANGING': 0.776, 'CONSOLIDATING': 0.843, 'VOLATILE': 0.899}
2026-04-26 20:56:26,214 INFO Regime epoch 26/50 — tr=0.6222 va=1.2310 acc=0.834
2026-04-26 20:56:26,737 INFO Regime epoch 27/50 — tr=0.6218 va=1.2295 acc=0.834
2026-04-26 20:56:27,255 INFO Regime epoch 28/50 — tr=0.6218 va=1.2294 acc=0.834
2026-04-26 20:56:27,770 INFO Regime epoch 29/50 — tr=0.6218 va=1.2270 acc=0.833
2026-04-26 20:56:28,310 INFO Regime epoch 30/50 — tr=0.6216 va=1.2319 acc=0.835 per_class={'TRENDING': 0.816, 'RANGING': 0.778, 'CONSOLIDATING': 0.836, 'VOLATILE': 0.903}
2026-04-26 20:56:28,836 INFO Regime epoch 31/50 — tr=0.6215 va=1.2303 acc=0.836
2026-04-26 20:56:29,334 INFO Regime epoch 32/50 — tr=0.6218 va=1.2260 acc=0.834
2026-04-26 20:56:29,834 INFO Regime epoch 33/50 — tr=0.6214 va=1.2298 acc=0.837
2026-04-26 20:56:30,340 INFO Regime epoch 34/50 — tr=0.6213 va=1.2251 acc=0.833
2026-04-26 20:56:30,883 INFO Regime epoch 35/50 — tr=0.6213 va=1.2297 acc=0.835 per_class={'TRENDING': 0.818, 'RANGING': 0.775, 'CONSOLIDATING': 0.85, 'VOLATILE': 0.893}
2026-04-26 20:56:31,383 INFO Regime epoch 36/50 — tr=0.6213 va=1.2269 acc=0.834
2026-04-26 20:56:31,870 INFO Regime epoch 37/50 — tr=0.6212 va=1.2275 acc=0.835
2026-04-26 20:56:32,380 INFO Regime epoch 38/50 — tr=0.6211 va=1.2280 acc=0.834
2026-04-26 20:56:32,865 INFO Regime epoch 39/50 — tr=0.6210 va=1.2267 acc=0.833
2026-04-26 20:56:33,390 INFO Regime epoch 40/50 — tr=0.6212 va=1.2262 acc=0.836 per_class={'TRENDING': 0.82, 'RANGING': 0.771, 'CONSOLIDATING': 0.859, 'VOLATILE': 0.888}
2026-04-26 20:56:33,883 INFO Regime epoch 41/50 — tr=0.6209 va=1.2241 acc=0.835
2026-04-26 20:56:34,369 INFO Regime epoch 42/50 — tr=0.6212 va=1.2267 acc=0.835
2026-04-26 20:56:34,865 INFO Regime epoch 43/50 — tr=0.6208 va=1.2254 acc=0.834
2026-04-26 20:56:35,372 INFO Regime epoch 44/50 — tr=0.6212 va=1.2242 acc=0.833
2026-04-26 20:56:35,911 INFO Regime epoch 45/50 — tr=0.6210 va=1.2283 acc=0.835 per_class={'TRENDING': 0.816, 'RANGING': 0.776, 'CONSOLIDATING': 0.85, 'VOLATILE': 0.896}
2026-04-26 20:56:36,421 INFO Regime epoch 46/50 — tr=0.6211 va=1.2282 acc=0.836
2026-04-26 20:56:36,911 INFO Regime epoch 47/50 — tr=0.6209 va=1.2266 acc=0.833
2026-04-26 20:56:37,397 INFO Regime epoch 48/50 — tr=0.6211 va=1.2295 acc=0.837
2026-04-26 20:56:37,904 INFO Regime epoch 49/50 — tr=0.6214 va=1.2274 acc=0.834
2026-04-26 20:56:38,453 INFO Regime epoch 50/50 — tr=0.6213 va=1.2258 acc=0.833 per_class={'TRENDING': 0.811, 'RANGING': 0.777, 'CONSOLIDATING': 0.851, 'VOLATILE': 0.897}
2026-04-26 20:56:38,493 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 20:56:38,493 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 20:56:38,495 INFO Regime phase LTF train: 25.6s
2026-04-26 20:56:38,625 INFO Regime LTF complete: acc=0.835, n=401471 per_class={'TRENDING': 0.816, 'RANGING': 0.777, 'CONSOLIDATING': 0.85, 'VOLATILE': 0.895}
2026-04-26 20:56:38,629 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 20:56:39,159 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 20:56:39,163 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 9.355291913830783, 'RANGING': 6.096303199751476, 'CONSOLIDATING': 5.598885793871866, 'VOLATILE': 6.771351107094442}
2026-04-26 20:56:39,171 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 29965, 'mean': 2.149062259083323e-05, 'mean_over_std': 0.011155601434349563}, 'RANGING': {'n': 19623, 'mean': -3.254493608872846e-06, 'mean_over_std': -0.001761415548292494}, 'CONSOLIDATING': {'n': 10050, 'mean': -3.039756158146958e-06, 'mean_over_std': -0.001792488091618151}, 'VOLATILE': {'n': 14985, 'mean': 1.8102764458593193e-05, 'mean_over_std': 0.006974696291378742}}
2026-04-26 20:56:39,171 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 26145, 'mean': 2.874435605138053e-05, 'mean_over_std': 0.01595450632691917}, 'RANGING': {'n': 7858, 'mean': 1.7829776785521727e-05, 'mean_over_std': 0.011851561804467463}, 'CONSOLIDATING': {'n': 8543, 'mean': 7.845231961250924e-06, 'mean_over_std': 0.004915839521549481}, 'VOLATILE': {'n': 13636, 'mean': 5.480114267306668e-06, 'mean_over_std': 0.0020378119917389375}}
2026-04-26 20:56:39,172 INFO Regime retrain total: 217.5s (504761 samples)
2026-04-26 20:56:39,173 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 20:56:39,174 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 20:56:39,174 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 20:56:39,174 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-26 20:56:39,174 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-26 20:56:39,175 INFO Retrain complete. Total wall-clock: 217.5s
  DONE  Retrain regime [full-data retrain]
  START Retrain quality [full-data retrain]
2026-04-26 20:56:40,616 INFO retrain environment: KAGGLE
2026-04-26 20:56:42,228 INFO Device: CUDA (2 GPU(s))
2026-04-26 20:56:42,239 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 20:56:42,239 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 20:56:42,239 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 20:56:42,239 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 20:56:42,241 INFO === QualityScorer retrain ===
2026-04-26 20:56:42,384 INFO NumExpr defaulting to 4 threads.
2026-04-26 20:56:42,577 INFO QualityScorer: CUDA available — using GPU
2026-04-26 20:56:42,789 INFO QualityScorer loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (device=cuda)
2026-04-26 20:56:43,159 INFO Quality phase label creation: 0.4s (5827 trades)
2026-04-26 20:56:43,160 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/quality_scorer.pkl_20260426_205643
2026-04-26 20:56:43,514 INFO QualityScorer: 5827 samples, EV stats={'mean': 0.38798898458480835, 'std': 1.2986980676651, 'n_pos': 3317, 'n_neg': 2510}, device=cuda
2026-04-26 20:56:43,515 INFO QualityScorer: normalised win labels by median_win=0.960 — EV range now [-1, +3]
2026-04-26 20:56:43,515 INFO QualityScorer: warm start from existing weights
2026-04-26 20:56:43,516 INFO QualityScorer: pos_weight=0.75 (n_pos=2671 n_neg=1990)
2026-04-26 20:56:45,941 INFO Quality epoch   1/100 — va_huber=0.5617
2026-04-26 20:56:46,082 INFO Quality epoch   2/100 — va_huber=0.5626
2026-04-26 20:56:46,215 INFO Quality epoch   3/100 — va_huber=0.5629
2026-04-26 20:56:46,345 INFO Quality epoch   4/100 — va_huber=0.5639
2026-04-26 20:56:46,475 INFO Quality epoch   5/100 — va_huber=0.5638
2026-04-26 20:56:47,235 INFO Quality epoch  11/100 — va_huber=0.5617
2026-04-26 20:56:47,235 INFO Quality early stop at epoch 11
2026-04-26 20:56:47,256 INFO QualityScorer EV model: MAE=1.137 dir_acc=0.607 n_val=1166
2026-04-26 20:56:47,260 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-26 20:56:47,334 INFO Quality phase train: 4.2s | total: 5.1s
2026-04-26 20:56:47,336 INFO Retrain complete. Total wall-clock: 5.1s
  DONE  Retrain quality [full-data retrain]
  START Retrain rl [full-data retrain]
2026-04-26 20:56:48,515 INFO retrain environment: KAGGLE
2026-04-26 20:56:50,146 INFO Device: CUDA (2 GPU(s))
2026-04-26 20:56:50,157 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 20:56:50,157 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 20:56:50,157 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 20:56:50,157 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 20:56:50,158 INFO === RLAgent (PPO) retrain ===
2026-04-26 20:56:50,160 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260426_205650
2026-04-26 20:56:50,316 INFO RL phase episode loading: 0.2s (5827 episodes)
2026-04-26 20:56:51.169315: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777237011.192855  101451 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777237011.200842  101451 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777237011.220707  101451 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777237011.220732  101451 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777237011.220734  101451 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777237011.220737  101451 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-26 20:56:55,638 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-26 20:56:58,625 INFO RLAgent: cold start — building new PPO policy
2026-04-26 20:58:52,954 INFO RLAgent: retrain complete, 5827 episodes
2026-04-26 20:58:52,955 INFO RL phase PPO train: 122.6s | total: 122.8s
2026-04-26 20:58:52,974 INFO Retrain complete. Total wall-clock: 122.8s
  DONE  Retrain rl [full-data retrain]

=== Round 3: Backtest on last 3yr (post-retrain evaluation) ===
  START Round 3 - Post-retrain backtest (last 3yr)
2026-04-26 20:58:55,093 INFO === STEP 6: BACKTEST (round3) ===
2026-04-26 20:58:55,094 INFO BT_WINDOW=round3 — post-retrain eval: 2022-08-05 → 2025-08-05 (last 3yr)
2026-04-26 20:58:55,094 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-04-26 20:58:55,094 INFO Round 3 — running backtest: 2022-08-05 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-26 21:02:21,189 INFO Round 3 backtest — 5830 trades | avg WR=57.5% | avg PF=2.98 | avg Sharpe=7.89
2026-04-26 21:02:21,189 INFO   ml_trader: 5830 trades | WR=57.5% | PF=2.98 | Return=4463.5% | DD=7.0% | Sharpe=7.89
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 5830
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (5830 rows)
  DONE  Round 3 - Post-retrain backtest (last 3yr)
  Saved Round 3 result → round3_summary.json

======================================================================
  BLIND BACKTEST PIPELINE COMPLETE
======================================================================
  Round 1 (val window)          trades=3296  WR=58.0%  PF=3.029  Sharpe=8.035
  Round 2 (blind test)          trades=2531  WR=55.6%  PF=2.848  Sharpe=7.506
  Round 3 (last 3yr)            trades=5830  WR=57.5%  PF=2.975  Sharpe=7.894


WARNING: GITHUB_TOKEN not set — skipping GitHub push
2026-04-26 21:02:23,453 INFO Round 3: wrote 5830 journal entries (total in file: 11657)