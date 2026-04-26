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
2026-04-26 18:54:34,277 INFO Loading feature-engineered data...
2026-04-26 18:54:34,912 INFO Loaded 221743 rows, 202 features
2026-04-26 18:54:34,913 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-26 18:54:34,915 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-26 18:54:34,915 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-26 18:54:34,916 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-26 18:54:34,916 INFO No leakage confirmed: train < val < test timestamps

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
2026-04-26 18:54:37,281 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-26 18:54:37,281 INFO --- Training regime ---
2026-04-26 18:54:37,282 INFO Running retrain --model regime
2026-04-26 18:54:37,478 INFO retrain environment: KAGGLE
2026-04-26 18:54:39,136 INFO Device: CUDA (2 GPU(s))
2026-04-26 18:54:39,147 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 18:54:39,147 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 18:54:39,147 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 18:54:39,150 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 18:54:39,151 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-26 18:54:39,306 INFO NumExpr defaulting to 4 threads.
2026-04-26 18:54:39,517 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 18:54:39,517 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 18:54:39,517 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 18:54:39,517 INFO Regime phase macro_correlations: 0.0s
2026-04-26 18:54:39,517 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-26 18:54:39,554 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 18:54:39,555 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:54:39,581 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:54:39,594 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:54:39,616 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:54:39,631 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:54:39,653 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:54:39,667 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:54:39,689 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:54:39,704 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:54:39,727 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:54:39,741 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:54:39,761 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:54:39,774 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:54:39,793 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:54:39,807 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:54:39,829 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:54:39,843 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:54:39,863 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:54:39,878 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:54:39,899 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:54:39,916 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:54:39,955 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:54:41,085 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 18:55:02,756 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 18:55:02,758 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 22.8s
2026-04-26 18:55:02,758 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 18:55:13,306 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 18:55:13,310 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 10.6s
2026-04-26 18:55:13,311 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 18:55:21,188 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 18:55:21,191 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 7.9s
2026-04-26 18:55:21,193 INFO Regime phase GMM HTF total: 41.2s
2026-04-26 18:55:21,194 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 18:56:31,990 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 18:56:31,994 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 70.8s
2026-04-26 18:56:31,994 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 18:57:02,167 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 18:57:02,168 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 30.2s
2026-04-26 18:57:02,169 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 18:57:23,376 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 18:57:23,377 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 21.2s
2026-04-26 18:57:23,377 INFO Regime phase GMM LTF total: 122.2s
2026-04-26 18:57:23,477 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-26 18:57:23,478 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:23,479 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:23,480 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:23,481 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:23,482 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:23,483 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:23,484 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:23,485 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:23,486 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:23,487 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:23,488 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:57:23,608 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:23,648 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:23,649 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:23,649 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:23,657 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:23,658 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:24,054 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-26 18:57:24,055 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-26 18:57:24,229 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:24,262 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:24,263 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:24,264 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:24,271 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:24,272 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:24,641 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-26 18:57:24,642 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-26 18:57:24,829 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:24,865 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:24,866 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:24,866 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:24,874 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:24,875 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:25,243 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-26 18:57:25,244 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-26 18:57:25,411 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:25,447 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:25,448 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:25,448 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:25,456 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:25,457 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:25,839 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-26 18:57:25,840 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-26 18:57:26,028 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:26,062 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:26,063 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:26,063 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:26,071 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:26,072 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:26,440 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-26 18:57:26,441 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-26 18:57:26,612 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:26,645 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:26,646 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:26,647 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:26,654 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:26,655 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:27,040 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-26 18:57:27,042 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-26 18:57:27,201 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:57:27,228 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:57:27,229 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:57:27,229 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:57:27,236 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:57:27,237 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:27,605 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-26 18:57:27,606 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-26 18:57:27,783 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:27,815 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:27,816 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:27,816 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:27,824 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:27,825 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:28,192 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-26 18:57:28,193 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-26 18:57:28,362 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:28,395 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:28,395 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:28,396 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:28,406 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:28,407 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:28,800 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-26 18:57:28,801 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-26 18:57:28,968 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:29,001 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:29,002 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:29,002 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:29,010 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:29,011 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:29,380 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-26 18:57:29,381 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-26 18:57:29,641 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:57:29,697 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:57:29,698 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:57:29,699 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:57:29,708 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:57:29,709 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:57:30,487 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 18:57:30,489 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-26 18:57:30,646 INFO Regime phase HTF dataset build: 7.2s (103290 samples)
2026-04-26 18:57:30,647 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=38114 dropped=65176 classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728})
2026-04-26 18:57:30,656 INFO RegimeClassifier[mode=htf_bias]: 38114 samples, classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728}, device=cuda
2026-04-26 18:57:30,657 INFO RegimeClassifier: sample weights — mean=0.708  ambiguous(<0.4)=0.0%
2026-04-26 18:57:30,927 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-04-26 18:57:30,927 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 18:57:35,428 INFO Regime epoch  1/50 — tr=0.7536 va=2.0259 acc=0.299 per_class={'BIAS_UP': 0.532, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 0.244}
2026-04-26 18:57:35,494 INFO Regime epoch  2/50 — tr=0.7474 va=2.0182 acc=0.262
2026-04-26 18:57:35,561 INFO Regime epoch  3/50 — tr=0.7408 va=1.9847 acc=0.300
2026-04-26 18:57:35,632 INFO Regime epoch  4/50 — tr=0.7254 va=1.9315 acc=0.397
2026-04-26 18:57:35,708 INFO Regime epoch  5/50 — tr=0.7061 va=1.8601 acc=0.576 per_class={'BIAS_UP': 0.599, 'BIAS_DOWN': 0.505, 'BIAS_NEUTRAL': 0.651}
2026-04-26 18:57:35,774 INFO Regime epoch  6/50 — tr=0.6784 va=1.7798 acc=0.714
2026-04-26 18:57:35,843 INFO Regime epoch  7/50 — tr=0.6539 va=1.6950 acc=0.790
2026-04-26 18:57:35,910 INFO Regime epoch  8/50 — tr=0.6244 va=1.6067 acc=0.849
2026-04-26 18:57:35,974 INFO Regime epoch  9/50 — tr=0.6002 va=1.5296 acc=0.890
2026-04-26 18:57:36,050 INFO Regime epoch 10/50 — tr=0.5789 va=1.4605 acc=0.914 per_class={'BIAS_UP': 0.95, 'BIAS_DOWN': 0.956, 'BIAS_NEUTRAL': 0.742}
2026-04-26 18:57:36,121 INFO Regime epoch 11/50 — tr=0.5596 va=1.4070 acc=0.924
2026-04-26 18:57:36,191 INFO Regime epoch 12/50 — tr=0.5479 va=1.3620 acc=0.933
2026-04-26 18:57:36,270 INFO Regime epoch 13/50 — tr=0.5368 va=1.3257 acc=0.941
2026-04-26 18:57:36,337 INFO Regime epoch 14/50 — tr=0.5278 va=1.2970 acc=0.946
2026-04-26 18:57:36,412 INFO Regime epoch 15/50 — tr=0.5218 va=1.2751 acc=0.949 per_class={'BIAS_UP': 0.99, 'BIAS_DOWN': 0.987, 'BIAS_NEUTRAL': 0.769}
2026-04-26 18:57:36,478 INFO Regime epoch 16/50 — tr=0.5167 va=1.2589 acc=0.953
2026-04-26 18:57:36,544 INFO Regime epoch 17/50 — tr=0.5114 va=1.2420 acc=0.956
2026-04-26 18:57:36,610 INFO Regime epoch 18/50 — tr=0.5075 va=1.2283 acc=0.958
2026-04-26 18:57:36,676 INFO Regime epoch 19/50 — tr=0.5058 va=1.2231 acc=0.960
2026-04-26 18:57:36,749 INFO Regime epoch 20/50 — tr=0.5035 va=1.2141 acc=0.961 per_class={'BIAS_UP': 0.997, 'BIAS_DOWN': 0.995, 'BIAS_NEUTRAL': 0.802}
2026-04-26 18:57:36,818 INFO Regime epoch 21/50 — tr=0.5011 va=1.2050 acc=0.962
2026-04-26 18:57:36,885 INFO Regime epoch 22/50 — tr=0.4991 va=1.1979 acc=0.963
2026-04-26 18:57:36,950 INFO Regime epoch 23/50 — tr=0.4976 va=1.1912 acc=0.965
2026-04-26 18:57:37,014 INFO Regime epoch 24/50 — tr=0.4961 va=1.1840 acc=0.966
2026-04-26 18:57:37,085 INFO Regime epoch 25/50 — tr=0.4956 va=1.1810 acc=0.967 per_class={'BIAS_UP': 0.998, 'BIAS_DOWN': 0.997, 'BIAS_NEUTRAL': 0.826}
2026-04-26 18:57:37,150 INFO Regime epoch 26/50 — tr=0.4943 va=1.1773 acc=0.967
2026-04-26 18:57:37,223 INFO Regime epoch 27/50 — tr=0.4924 va=1.1716 acc=0.968
2026-04-26 18:57:37,290 INFO Regime epoch 28/50 — tr=0.4923 va=1.1666 acc=0.969
2026-04-26 18:57:37,356 INFO Regime epoch 29/50 — tr=0.4906 va=1.1643 acc=0.969
2026-04-26 18:57:37,427 INFO Regime epoch 30/50 — tr=0.4901 va=1.1638 acc=0.969 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.834}
2026-04-26 18:57:37,493 INFO Regime epoch 31/50 — tr=0.4901 va=1.1589 acc=0.970
2026-04-26 18:57:37,558 INFO Regime epoch 32/50 — tr=0.4889 va=1.1558 acc=0.970
2026-04-26 18:57:37,625 INFO Regime epoch 33/50 — tr=0.4886 va=1.1537 acc=0.970
2026-04-26 18:57:37,691 INFO Regime epoch 34/50 — tr=0.4876 va=1.1510 acc=0.971
2026-04-26 18:57:37,761 INFO Regime epoch 35/50 — tr=0.4869 va=1.1464 acc=0.972 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.845}
2026-04-26 18:57:37,826 INFO Regime epoch 36/50 — tr=0.4869 va=1.1454 acc=0.971
2026-04-26 18:57:37,894 INFO Regime epoch 37/50 — tr=0.4863 va=1.1414 acc=0.972
2026-04-26 18:57:37,960 INFO Regime epoch 38/50 — tr=0.4859 va=1.1403 acc=0.972
2026-04-26 18:57:38,026 INFO Regime epoch 39/50 — tr=0.4862 va=1.1421 acc=0.972
2026-04-26 18:57:38,094 INFO Regime epoch 40/50 — tr=0.4857 va=1.1406 acc=0.972 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.848}
2026-04-26 18:57:38,157 INFO Regime epoch 41/50 — tr=0.4853 va=1.1366 acc=0.973
2026-04-26 18:57:38,231 INFO Regime epoch 42/50 — tr=0.4853 va=1.1391 acc=0.973
2026-04-26 18:57:38,295 INFO Regime epoch 43/50 — tr=0.4853 va=1.1373 acc=0.972
2026-04-26 18:57:38,363 INFO Regime epoch 44/50 — tr=0.4850 va=1.1381 acc=0.972
2026-04-26 18:57:38,450 INFO Regime epoch 45/50 — tr=0.4854 va=1.1381 acc=0.972 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.85}
2026-04-26 18:57:38,518 INFO Regime epoch 46/50 — tr=0.4850 va=1.1395 acc=0.972
2026-04-26 18:57:38,586 INFO Regime epoch 47/50 — tr=0.4851 va=1.1405 acc=0.973
2026-04-26 18:57:38,655 INFO Regime epoch 48/50 — tr=0.4856 va=1.1397 acc=0.972
2026-04-26 18:57:38,755 INFO Regime epoch 49/50 — tr=0.4848 va=1.1370 acc=0.972
2026-04-26 18:57:38,826 INFO Regime epoch 50/50 — tr=0.4852 va=1.1372 acc=0.972 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.849}
2026-04-26 18:57:38,836 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 18:57:38,836 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 18:57:38,836 INFO Regime phase HTF train: 8.2s
2026-04-26 18:57:38,962 INFO Regime HTF complete: acc=0.973, n=103290 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.851}
2026-04-26 18:57:38,964 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:57:39,109 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 18:57:39,116 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 11.41578947368421, 'BIAS_DOWN': 10.635761589403973, 'BIAS_NEUTRAL': 17.960468521229867}
2026-04-26 18:57:39,120 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 4338, 'mean': 0.00011110466582543088, 'mean_over_std': 0.027632581210502927}, 'BIAS_DOWN': {'n': 3212, 'mean': 6.063430222660252e-05, 'mean_over_std': 0.013898489590377157}, 'BIAS_NEUTRAL': {'n': 12266, 'mean': 1.2151554902814046e-05, 'mean_over_std': 0.0031957000513162357}}
2026-04-26 18:57:39,120 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 3651, 'mean': 0.0002852242108836959, 'mean_over_std': 0.07015017232491755}, 'BIAS_DOWN': {'n': 2644, 'mean': -0.00014900749618981248, 'mean_over_std': -0.03333037041298237}, 'BIAS_NEUTRAL': {'n': 1376, 'mean': 9.595797941249082e-05, 'mean_over_std': 0.02492411099521803}}
2026-04-26 18:57:39,120 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-26 18:57:39,122 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:39,124 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:39,125 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:39,127 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:39,128 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:39,130 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:39,131 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:39,132 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:39,134 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:39,135 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:39,138 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:57:39,148 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:39,151 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:39,152 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:39,152 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:39,153 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:39,155 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:39,753 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-26 18:57:39,756 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-26 18:57:39,887 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:39,889 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:39,890 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:39,890 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:39,890 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:39,892 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:40,452 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-26 18:57:40,455 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-26 18:57:40,588 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:40,590 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:40,591 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:40,591 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:40,591 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:40,593 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:41,160 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-26 18:57:41,162 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-26 18:57:41,294 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:41,296 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:41,297 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:41,297 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:41,298 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:41,300 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:41,875 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-26 18:57:41,878 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-26 18:57:42,007 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:42,011 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:42,012 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:42,013 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:42,013 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:42,015 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:42,583 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-26 18:57:42,586 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-26 18:57:42,718 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:42,720 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:42,721 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:42,722 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:42,722 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:42,724 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:43,293 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-26 18:57:43,296 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-26 18:57:43,426 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:57:43,427 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:57:43,428 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:57:43,428 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:57:43,429 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:57:43,430 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:44,016 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-26 18:57:44,018 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-26 18:57:44,149 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:44,152 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:44,153 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:44,153 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:44,153 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:44,155 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:44,742 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-26 18:57:44,744 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-26 18:57:44,880 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:44,882 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:44,883 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:44,883 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:44,884 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:44,885 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:45,472 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-26 18:57:45,474 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-26 18:57:45,605 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:45,609 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:45,609 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:45,610 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:45,610 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:57:45,612 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:57:46,193 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-26 18:57:46,195 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-26 18:57:46,334 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:57:46,340 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:57:46,341 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:57:46,341 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:57:46,342 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:57:46,345 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:57:47,610 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 18:57:47,617 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-26 18:57:47,912 INFO Regime phase LTF dataset build: 8.8s (401471 samples)
2026-04-26 18:57:47,915 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=299185 dropped=102286 classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127})
2026-04-26 18:57:47,980 INFO RegimeClassifier[mode=ltf_behaviour]: 299185 samples, classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127}, device=cuda
2026-04-26 18:57:47,981 INFO RegimeClassifier: sample weights — mean=0.693  ambiguous(<0.4)=0.0%
2026-04-26 18:57:47,983 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-04-26 18:57:47,983 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 18:57:48,543 INFO Regime epoch  1/50 — tr=0.8706 va=2.2113 acc=0.302 per_class={'TRENDING': 0.25, 'RANGING': 0.055, 'CONSOLIDATING': 0.237, 'VOLATILE': 0.582}
2026-04-26 18:57:49,047 INFO Regime epoch  2/50 — tr=0.8444 va=2.1053 acc=0.420
2026-04-26 18:57:49,532 INFO Regime epoch  3/50 — tr=0.8044 va=1.9474 acc=0.523
2026-04-26 18:57:50,026 INFO Regime epoch  4/50 — tr=0.7620 va=1.7767 acc=0.580
2026-04-26 18:57:50,548 INFO Regime epoch  5/50 — tr=0.7292 va=1.6413 acc=0.622 per_class={'TRENDING': 0.507, 'RANGING': 0.481, 'CONSOLIDATING': 0.622, 'VOLATILE': 0.924}
2026-04-26 18:57:51,033 INFO Regime epoch  6/50 — tr=0.7070 va=1.5570 acc=0.655
2026-04-26 18:57:51,521 INFO Regime epoch  7/50 — tr=0.6920 va=1.5009 acc=0.683
2026-04-26 18:57:52,054 INFO Regime epoch  8/50 — tr=0.6810 va=1.4692 acc=0.705
2026-04-26 18:57:52,545 INFO Regime epoch  9/50 — tr=0.6724 va=1.4367 acc=0.725
2026-04-26 18:57:53,074 INFO Regime epoch 10/50 — tr=0.6661 va=1.4131 acc=0.741 per_class={'TRENDING': 0.659, 'RANGING': 0.704, 'CONSOLIDATING': 0.722, 'VOLATILE': 0.931}
2026-04-26 18:57:53,580 INFO Regime epoch 11/50 — tr=0.6604 va=1.3852 acc=0.754
2026-04-26 18:57:54,061 INFO Regime epoch 12/50 — tr=0.6560 va=1.3696 acc=0.766
2026-04-26 18:57:54,543 INFO Regime epoch 13/50 — tr=0.6523 va=1.3493 acc=0.772
2026-04-26 18:57:55,033 INFO Regime epoch 14/50 — tr=0.6489 va=1.3435 acc=0.780
2026-04-26 18:57:55,556 INFO Regime epoch 15/50 — tr=0.6464 va=1.3299 acc=0.786 per_class={'TRENDING': 0.74, 'RANGING': 0.743, 'CONSOLIDATING': 0.745, 'VOLATILE': 0.923}
2026-04-26 18:57:56,041 INFO Regime epoch 16/50 — tr=0.6444 va=1.3220 acc=0.790
2026-04-26 18:57:56,520 INFO Regime epoch 17/50 — tr=0.6422 va=1.3100 acc=0.795
2026-04-26 18:57:57,012 INFO Regime epoch 18/50 — tr=0.6405 va=1.3007 acc=0.798
2026-04-26 18:57:57,526 INFO Regime epoch 19/50 — tr=0.6393 va=1.2974 acc=0.801
2026-04-26 18:57:58,052 INFO Regime epoch 20/50 — tr=0.6377 va=1.2890 acc=0.803 per_class={'TRENDING': 0.771, 'RANGING': 0.76, 'CONSOLIDATING': 0.765, 'VOLATILE': 0.913}
2026-04-26 18:57:58,546 INFO Regime epoch 21/50 — tr=0.6365 va=1.2875 acc=0.806
2026-04-26 18:57:59,076 INFO Regime epoch 22/50 — tr=0.6354 va=1.2762 acc=0.806
2026-04-26 18:57:59,577 INFO Regime epoch 23/50 — tr=0.6346 va=1.2729 acc=0.808
2026-04-26 18:58:00,069 INFO Regime epoch 24/50 — tr=0.6334 va=1.2740 acc=0.810
2026-04-26 18:58:00,600 INFO Regime epoch 25/50 — tr=0.6325 va=1.2711 acc=0.813 per_class={'TRENDING': 0.79, 'RANGING': 0.762, 'CONSOLIDATING': 0.785, 'VOLATILE': 0.904}
2026-04-26 18:58:01,095 INFO Regime epoch 26/50 — tr=0.6317 va=1.2704 acc=0.815
2026-04-26 18:58:01,580 INFO Regime epoch 27/50 — tr=0.6308 va=1.2667 acc=0.817
2026-04-26 18:58:02,054 INFO Regime epoch 28/50 — tr=0.6305 va=1.2642 acc=0.817
2026-04-26 18:58:02,557 INFO Regime epoch 29/50 — tr=0.6296 va=1.2611 acc=0.818
2026-04-26 18:58:03,078 INFO Regime epoch 30/50 — tr=0.6292 va=1.2595 acc=0.819 per_class={'TRENDING': 0.796, 'RANGING': 0.766, 'CONSOLIDATING': 0.806, 'VOLATILE': 0.899}
2026-04-26 18:58:03,553 INFO Regime epoch 31/50 — tr=0.6287 va=1.2578 acc=0.821
2026-04-26 18:58:04,039 INFO Regime epoch 32/50 — tr=0.6280 va=1.2553 acc=0.820
2026-04-26 18:58:04,522 INFO Regime epoch 33/50 — tr=0.6280 va=1.2538 acc=0.822
2026-04-26 18:58:05,010 INFO Regime epoch 34/50 — tr=0.6273 va=1.2517 acc=0.820
2026-04-26 18:58:05,542 INFO Regime epoch 35/50 — tr=0.6270 va=1.2497 acc=0.822 per_class={'TRENDING': 0.797, 'RANGING': 0.767, 'CONSOLIDATING': 0.821, 'VOLATILE': 0.901}
2026-04-26 18:58:06,042 INFO Regime epoch 36/50 — tr=0.6267 va=1.2530 acc=0.824
2026-04-26 18:58:06,523 INFO Regime epoch 37/50 — tr=0.6268 va=1.2502 acc=0.825
2026-04-26 18:58:07,019 INFO Regime epoch 38/50 — tr=0.6261 va=1.2497 acc=0.824
2026-04-26 18:58:07,522 INFO Regime epoch 39/50 — tr=0.6260 va=1.2509 acc=0.823
2026-04-26 18:58:08,033 INFO Regime epoch 40/50 — tr=0.6259 va=1.2473 acc=0.825 per_class={'TRENDING': 0.804, 'RANGING': 0.766, 'CONSOLIDATING': 0.829, 'VOLATILE': 0.896}
2026-04-26 18:58:08,510 INFO Regime epoch 41/50 — tr=0.6258 va=1.2440 acc=0.825
2026-04-26 18:58:09,016 INFO Regime epoch 42/50 — tr=0.6258 va=1.2437 acc=0.825
2026-04-26 18:58:09,486 INFO Regime epoch 43/50 — tr=0.6254 va=1.2436 acc=0.825
2026-04-26 18:58:09,987 INFO Regime epoch 44/50 — tr=0.6256 va=1.2447 acc=0.825
2026-04-26 18:58:10,506 INFO Regime epoch 45/50 — tr=0.6252 va=1.2434 acc=0.827 per_class={'TRENDING': 0.808, 'RANGING': 0.764, 'CONSOLIDATING': 0.84, 'VOLATILE': 0.891}
2026-04-26 18:58:10,980 INFO Regime epoch 46/50 — tr=0.6257 va=1.2431 acc=0.825
2026-04-26 18:58:11,455 INFO Regime epoch 47/50 — tr=0.6254 va=1.2452 acc=0.827
2026-04-26 18:58:11,950 INFO Regime epoch 48/50 — tr=0.6253 va=1.2429 acc=0.824
2026-04-26 18:58:12,432 INFO Regime epoch 49/50 — tr=0.6255 va=1.2438 acc=0.824
2026-04-26 18:58:12,951 INFO Regime epoch 50/50 — tr=0.6255 va=1.2433 acc=0.826 per_class={'TRENDING': 0.807, 'RANGING': 0.771, 'CONSOLIDATING': 0.827, 'VOLATILE': 0.895}
2026-04-26 18:58:12,991 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 18:58:12,991 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 18:58:12,993 INFO Regime phase LTF train: 25.1s
2026-04-26 18:58:13,120 INFO Regime LTF complete: acc=0.824, n=401471 per_class={'TRENDING': 0.8, 'RANGING': 0.769, 'CONSOLIDATING': 0.83, 'VOLATILE': 0.899}
2026-04-26 18:58:13,123 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:58:13,609 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 18:58:13,613 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 9.355291913830783, 'RANGING': 6.096303199751476, 'CONSOLIDATING': 5.598885793871866, 'VOLATILE': 6.771351107094442}
2026-04-26 18:58:13,621 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 29965, 'mean': 2.149062259083323e-05, 'mean_over_std': 0.011155601434349563}, 'RANGING': {'n': 19623, 'mean': -3.254493608872846e-06, 'mean_over_std': -0.001761415548292494}, 'CONSOLIDATING': {'n': 10050, 'mean': -3.039756158146958e-06, 'mean_over_std': -0.001792488091618151}, 'VOLATILE': {'n': 14985, 'mean': 1.8102764458593193e-05, 'mean_over_std': 0.006974696291378742}}
2026-04-26 18:58:13,621 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 26145, 'mean': 2.874435605138053e-05, 'mean_over_std': 0.01595450632691917}, 'RANGING': {'n': 7858, 'mean': 1.7829776785521727e-05, 'mean_over_std': 0.011851561804467463}, 'CONSOLIDATING': {'n': 8543, 'mean': 7.845231961250924e-06, 'mean_over_std': 0.004915839521549481}, 'VOLATILE': {'n': 13636, 'mean': 5.480114267306668e-06, 'mean_over_std': 0.0020378119917389375}}
2026-04-26 18:58:13,622 INFO Regime retrain total: 214.5s (504761 samples)
2026-04-26 18:58:13,635 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 18:58:13,635 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 18:58:13,635 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 18:58:13,635 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-26 18:58:13,636 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-26 18:58:13,636 INFO Retrain complete. Total wall-clock: 214.5s
2026-04-26 18:58:15,889 INFO Model regime: SUCCESS
2026-04-26 18:58:15,889 INFO --- Training gru ---
2026-04-26 18:58:15,889 INFO Running retrain --model gru
2026-04-26 18:58:16,200 INFO retrain environment: KAGGLE
2026-04-26 18:58:17,783 INFO Device: CUDA (2 GPU(s))
2026-04-26 18:58:17,794 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 18:58:17,794 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 18:58:17,794 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 18:58:17,795 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 18:58:17,796 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-26 18:58:17,938 INFO NumExpr defaulting to 4 threads.
2026-04-26 18:58:18,132 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 18:58:18,132 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 18:58:18,132 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 18:58:18,133 INFO GRU phase macro_correlations: 0.0s
2026-04-26 18:58:18,133 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-26 18:58:18,133 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260426_185818
2026-04-26 18:58:18,136 WARNING WeightsManifest: no manifest at /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json — assuming compatible (run a full retrain to generate one)
2026-04-26 18:58:18,278 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:18,300 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:18,313 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:18,320 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:18,321 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 18:58:18,321 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 18:58:18,321 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 18:58:18,322 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 18:58:18,323 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:18,407 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-26 18:58:18,409 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:18,646 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-26 18:58:18,680 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:18,958 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:19,087 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:19,181 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:19,376 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:19,394 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:19,408 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:19,414 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:19,415 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:19,491 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-26 18:58:19,492 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:19,717 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-26 18:58:19,731 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:19,996 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:20,121 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:20,211 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:20,386 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:20,407 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:20,421 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:20,428 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:20,429 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:20,503 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-26 18:58:20,504 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:20,721 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-26 18:58:20,736 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:20,996 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:21,122 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:21,213 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:21,396 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:21,416 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:21,430 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:21,436 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:21,437 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:21,512 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-26 18:58:21,514 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:21,741 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-26 18:58:21,764 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:22,024 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:22,146 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:22,241 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:22,420 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:22,439 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:22,454 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:22,461 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:22,462 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:22,540 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-26 18:58:22,541 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:22,763 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-26 18:58:22,777 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:23,044 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:23,166 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:23,259 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:23,435 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:23,453 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:23,468 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:23,474 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:23,475 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:23,547 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-26 18:58:23,549 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:23,774 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-26 18:58:23,789 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:24,050 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:24,171 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:24,265 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:24,428 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:58:24,445 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:58:24,459 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:58:24,465 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:58:24,465 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:24,543 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-26 18:58:24,544 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:24,776 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-26 18:58:24,789 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:25,060 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:25,185 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:25,281 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:25,459 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:25,478 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:25,492 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:25,499 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:25,500 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:25,577 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-26 18:58:25,579 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:25,813 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-26 18:58:25,831 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:26,100 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:26,224 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:26,322 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:26,502 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:26,520 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:26,535 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:26,542 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:26,543 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:26,620 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-26 18:58:26,621 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:26,861 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-26 18:58:26,876 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:27,155 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:27,285 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:27,384 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:27,572 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:27,593 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:27,608 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:27,616 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:58:27,617 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:27,699 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-26 18:58:27,701 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:27,949 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-26 18:58:27,965 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:28,233 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:28,357 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:28,457 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:58:28,759 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:58:28,783 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:58:28,799 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:58:28,808 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:58:28,810 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:58:28,972 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 18:58:28,975 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:58:29,476 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 18:58:29,520 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:58:30,042 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:58:30,232 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:58:30,360 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:58:30,471 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-26 18:58:30,721 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-04-26 18:58:30,721 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-04-26 18:58:30,721 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-26 18:58:30,721 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-26 18:59:17,571 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-26 18:59:17,571 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-26 18:59:18,922 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-26 18:59:22,957 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=5)
2026-04-26 18:59:36,725 INFO train_multi TF=ALL epoch 1/50 train=0.8713 val=0.8648
2026-04-26 18:59:36,733 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:59:36,734 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:59:36,734 INFO train_multi TF=ALL: new best val=0.8648 — saved
2026-04-26 18:59:48,524 INFO train_multi TF=ALL epoch 2/50 train=0.8558 val=0.8369
2026-04-26 18:59:48,528 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:59:48,528 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:59:48,528 INFO train_multi TF=ALL: new best val=0.8369 — saved
2026-04-26 19:00:00,285 INFO train_multi TF=ALL epoch 3/50 train=0.7661 val=0.6885
2026-04-26 19:00:00,289 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 19:00:00,289 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 19:00:00,290 INFO train_multi TF=ALL: new best val=0.6885 — saved
2026-04-26 19:00:12,076 INFO train_multi TF=ALL epoch 4/50 train=0.6916 val=0.6879
2026-04-26 19:00:12,081 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 19:00:12,081 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 19:00:12,081 INFO train_multi TF=ALL: new best val=0.6879 — saved
2026-04-26 19:00:23,775 INFO train_multi TF=ALL epoch 5/50 train=0.6905 val=0.6879
2026-04-26 19:00:35,582 INFO train_multi TF=ALL epoch 6/50 train=0.6899 val=0.6879
2026-04-26 19:00:47,350 INFO train_multi TF=ALL epoch 7/50 train=0.6894 val=0.6879
2026-04-26 19:00:47,355 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 19:00:47,355 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 19:00:47,355 INFO train_multi TF=ALL: new best val=0.6879 — saved
2026-04-26 19:00:59,252 INFO train_multi TF=ALL epoch 8/50 train=0.6891 val=0.6878
2026-04-26 19:00:59,257 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 19:00:59,257 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 19:00:59,257 INFO train_multi TF=ALL: new best val=0.6878 — saved
2026-04-26 19:01:11,149 INFO train_multi TF=ALL epoch 9/50 train=0.6888 val=0.6877
2026-04-26 19:01:11,154 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 19:01:11,154 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 19:01:11,154 INFO train_multi TF=ALL: new best val=0.6877 — saved
2026-04-26 19:01:22,880 INFO train_multi TF=ALL epoch 10/50 train=0.6887 val=0.6877
2026-04-26 19:01:22,885 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 19:01:22,885 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 19:01:22,885 INFO train_multi TF=ALL: new best val=0.6877 — saved
2026-04-26 19:01:34,680 INFO train_multi TF=ALL epoch 11/50 train=0.6884 val=0.6873
2026-04-26 19:01:34,684 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 19:01:34,684 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 19:01:34,684 INFO train_multi TF=ALL: new best val=0.6873 — saved
2026-04-26 19:01:46,400 INFO train_multi TF=ALL epoch 12/50 train=0.6878 val=0.6866
2026-04-26 19:01:46,404 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 19:01:46,404 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 19:01:46,404 INFO train_multi TF=ALL: new best val=0.6866 — saved
2026-04-26 19:01:58,160 INFO train_multi TF=ALL epoch 13/50 train=0.6861 val=0.6850
2026-04-26 19:01:58,164 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 19:01:58,164 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 19:01:58,164 INFO train_multi TF=ALL: new best val=0.6850 — saved
2026-04-26 19:02:09,907 INFO train_multi TF=ALL epoch 14/50 train=0.6778 val=0.6703
2026-04-26 19:02:09,912 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 19:02:09,912 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 19:02:09,912 INFO train_multi TF=ALL: new best val=0.6703 — saved
2026-04-26 19:02:21,626 INFO train_multi TF=ALL epoch 15/50 train=0.6624 val=0.6515
2026-04-26 19:02:21,630 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 19:02:21,631 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 19:02:21,631 INFO train_multi TF=ALL: new best val=0.6515 — saved
2026-04-26 19:02:33,362 INFO train_multi TF=ALL epoch 16/50 train=0.6489 val=0.6382
2026-04-26 19:02:33,366 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 19:02:33,366 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 19:02:33,366 INFO train_multi TF=ALL: new best val=0.6382 — saved
2026-04-26 19:02:45,062 INFO train_multi TF=ALL epoch 17/50 train=0.6397 val=0.6318
2026-04-26 19:02:45,066 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 19:02:45,066 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 19:02:45,067 INFO train_multi TF=ALL: new best val=0.6318 — saved
2026-04-26 19:02:56,651 INFO train_multi TF=ALL epoch 18/50 train=0.6333 val=0.6263
2026-04-26 19:02:56,656 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 19:02:56,656 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 19:02:56,656 INFO train_multi TF=ALL: new best val=0.6263 — saved
2026-04-26 19:03:08,454 INFO train_multi TF=ALL epoch 19/50 train=0.6283 val=0.6218
2026-04-26 19:03:08,458 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 19:03:08,458 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 19:03:08,458 INFO train_multi TF=ALL: new best val=0.6218 — saved
2026-04-26 19:03:20,238 INFO train_multi TF=ALL epoch 20/50 train=0.6244 val=0.6220
2026-04-26 19:03:31,948 INFO train_multi TF=ALL epoch 21/50 train=0.6216 val=0.6204
2026-04-26 19:03:31,952 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 19:03:31,952 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 19:03:31,952 INFO train_multi TF=ALL: new best val=0.6204 — saved
2026-04-26 19:03:43,782 INFO train_multi TF=ALL epoch 22/50 train=0.6185 val=0.6214
2026-04-26 19:03:55,530 INFO train_multi TF=ALL epoch 23/50 train=0.6164 val=0.6195
2026-04-26 19:03:55,534 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 19:03:55,534 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 19:03:55,534 INFO train_multi TF=ALL: new best val=0.6195 — saved
2026-04-26 19:04:07,358 INFO train_multi TF=ALL epoch 24/50 train=0.6137 val=0.6199
2026-04-26 19:04:19,147 INFO train_multi TF=ALL epoch 25/50 train=0.6118 val=0.6179
2026-04-26 19:04:19,152 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 19:04:19,152 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 19:04:19,152 INFO train_multi TF=ALL: new best val=0.6179 — saved
2026-04-26 19:04:30,961 INFO train_multi TF=ALL epoch 26/50 train=0.6103 val=0.6140
2026-04-26 19:04:30,966 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 19:04:30,966 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 19:04:30,966 INFO train_multi TF=ALL: new best val=0.6140 — saved
2026-04-26 19:04:42,650 INFO train_multi TF=ALL epoch 27/50 train=0.6082 val=0.6149
2026-04-26 19:04:54,388 INFO train_multi TF=ALL epoch 28/50 train=0.6068 val=0.6177
2026-04-26 19:05:06,164 INFO train_multi TF=ALL epoch 29/50 train=0.6044 val=0.6124
2026-04-26 19:05:06,169 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 19:05:06,169 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 19:05:06,169 INFO train_multi TF=ALL: new best val=0.6124 — saved
2026-04-26 19:05:17,967 INFO train_multi TF=ALL epoch 30/50 train=0.6034 val=0.6175
2026-04-26 19:05:29,859 INFO train_multi TF=ALL epoch 31/50 train=0.6023 val=0.6190
2026-04-26 19:05:41,648 INFO train_multi TF=ALL epoch 32/50 train=0.6007 val=0.6079
2026-04-26 19:05:41,653 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 19:05:41,653 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 19:05:41,653 INFO train_multi TF=ALL: new best val=0.6079 — saved
2026-04-26 19:05:53,373 INFO train_multi TF=ALL epoch 33/50 train=0.5993 val=0.6097
2026-04-26 19:06:05,156 INFO train_multi TF=ALL epoch 34/50 train=0.5979 val=0.6115
2026-04-26 19:06:16,947 INFO train_multi TF=ALL epoch 35/50 train=0.5966 val=0.6136
2026-04-26 19:06:28,700 INFO train_multi TF=ALL epoch 36/50 train=0.5953 val=0.6125
2026-04-26 19:06:40,449 INFO train_multi TF=ALL epoch 37/50 train=0.5940 val=0.6111
2026-04-26 19:06:40,449 INFO train_multi TF=ALL early stop at epoch 37
2026-04-26 19:06:40,583 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-26 19:06:40,583 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-26 19:06:40,583 INFO Retrain complete. Total wall-clock: 502.8s
2026-04-26 19:06:42,513 INFO Model gru: SUCCESS
2026-04-26 19:06:42,513 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 19:06:42,513 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 19:06:42,513 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 19:06:42,514 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-04-26 19:06:42,514 INFO   [DEFERRED] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip (expected after Round 1)
2026-04-26 19:06:42,514 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-04-26 19:06:42,514 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer', 'rl_ppo']
2026-04-26 19:06:42,515 INFO Saved 5 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  START Round 1 - Backtest (val)
2026-04-26 19:06:43,191 INFO === STEP 6: BACKTEST (round1) ===
2026-04-26 19:06:43,192 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-04-26 19:06:43,192 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-26 19:06:43,192 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
2026-04-26 19:06:45,491 WARNING QualityScorer unavailable (weights missing or load failed)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260426_190645.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)             3341  61.3%   2.74 992805447240838.1% 61.3% 30.6%  11.4%    1.91
  gate_diagnostics: bars=468696 no_signal=84360 quality_block=0 session_skip=317895 density=33042 pm_reject=0 daily_skip=0 cooldown=30058 daily_halt_events=0 enforce_daily_halt=False
  no_signal_reasons: weak_gru_direction=45472, htf_bias_conflict=37297, trend_pullback_conflict=1579, range_side_conflict=12

Calibration Summary:
  all          [WARN] Non-monotonic calibration: 2/4 pairs violated. Consider retraining QualityScorer
  ml_trader    [WARN] Non-monotonic calibration: 2/4 pairs violated. Consider retraining QualityScorer
2026-04-26 19:09:01,642 INFO Round 1 backtest — 3341 trades | avg WR=61.3% | avg PF=2.74 | avg Sharpe=1.91
2026-04-26 19:09:01,642 INFO   ml_trader: 3341 trades | WR=61.3% | PF=2.74 | Return=992805447240838.1% | DD=11.4% | Sharpe=1.91
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 3341
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (3341 rows)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  AUDUSD        328 trades     1 days  328.00/day  [OVERTRADE]
  EURGBP        350 trades     1 days  350.00/day  [OVERTRADE]
  EURJPY        332 trades     1 days  332.00/day  [OVERTRADE]
  EURUSD        330 trades     1 days  330.00/day  [OVERTRADE]
  GBPJPY        336 trades     1 days  336.00/day  [OVERTRADE]
  GBPUSD        334 trades     1 days  334.00/day  [OVERTRADE]
  NZDUSD          5 trades     1 days   5.00/day  [OVERTRADE]
  USDCAD        325 trades     1 days  325.00/day  [OVERTRADE]
  USDCHF        363 trades     1 days  363.00/day  [OVERTRADE]
  USDJPY        320 trades     1 days  320.00/day  [OVERTRADE]
  XAUUSD        318 trades     1 days  318.00/day  [OVERTRADE]
  ⚠  AUDUSD: 328.00/day (>1.5)
  ⚠  EURGBP: 350.00/day (>1.5)
  ⚠  EURJPY: 332.00/day (>1.5)
  ⚠  EURUSD: 330.00/day (>1.5)
  ⚠  GBPJPY: 336.00/day (>1.5)
  ⚠  GBPUSD: 334.00/day (>1.5)
  ⚠  NZDUSD: 5.00/day (>1.5)
  ⚠  USDCAD: 325.00/day (>1.5)
  ⚠  USDCHF: 363.00/day (>1.5)
  ⚠  USDJPY: 320.00/day (>1.5)
  ⚠  XAUUSD: 318.00/day (>1.5)

──────────────────────────────────────────────────────────────
CHECK 2: REGIME DISTRIBUTION  (% of trades, WR, avgEV)
──────────────────────────────────────────────────────────────
  BIAS_DOWN           805 trades   24.1%  WR=57.9%  avgEV=0.000
  BIAS_NEUTRAL       1420 trades
2026-04-26 19:09:03,039 INFO Round 1: wrote 3341 journal entries (total in file: 3341)
  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 3341 entries

=== Round 1 → Retrain Quality + RL ===
  START Round 1 - Quality+RL retrain
2026-04-26 19:09:03,335 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-26 19:09:03,339 INFO Journal entries: 3341
2026-04-26 19:09:03,339 INFO --- Training quality ---
2026-04-26 19:09:03,339 INFO Running retrain --model quality
2026-04-26 19:09:03,513 INFO retrain environment: KAGGLE
2026-04-26 19:09:05,111 INFO Device: CUDA (2 GPU(s))
2026-04-26 19:09:05,123 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 19:09:05,123 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 19:09:05,123 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 19:09:05,124 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 19:09:05,125 INFO === QualityScorer retrain ===
2026-04-26 19:09:05,271 INFO NumExpr defaulting to 4 threads.
2026-04-26 19:09:05,466 INFO QualityScorer: CUDA available — using GPU
2026-04-26 19:09:05,675 INFO Quality phase label creation: 0.2s (3341 trades)
2026-04-26 19:09:05,877 INFO QualityScorer: 3341 samples, EV stats={'mean': 0.517761766910553, 'std': 1.3029541969299316, 'n_pos': 2049, 'n_neg': 1292}, device=cuda
2026-04-26 19:09:05,878 INFO QualityScorer: normalised win labels by median_win=1.104 — EV range now [-1, +3]
2026-04-26 19:09:06,080 INFO QualityScorer: DataParallel across 2 GPUs
2026-04-26 19:09:06,080 INFO QualityScorer: cold start
2026-04-26 19:09:06,080 ERROR QualityScorer.train failed: cannot access local variable 'tr_ds' where it is not associated with a value
Traceback (most recent call last):
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1343, in <module>
    main()
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1316, in main
    result = retrain_quality(dry)
             ^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1012, in retrain_quality
    result = model.train(JOURNAL_PATH)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../models/quality_scorer.py", line 437, in train
    _bs      = min(512, len(tr_ds)) if DEVICE.type == "cuda" else min(256, len(tr_ds))
                            ^^^^^
UnboundLocalError: cannot access local variable 'tr_ds' where it is not associated with a value
2026-04-26 19:09:06,617 ERROR retrain quality failed (exit 1)
2026-04-26 19:09:06,617 ERROR Model quality failed: exit 1
2026-04-26 19:09:06,618 INFO --- Training rl ---
2026-04-26 19:09:06,618 INFO Running retrain --model rl
2026-04-26 19:09:06,796 INFO retrain environment: KAGGLE
2026-04-26 19:09:08,421 INFO Device: CUDA (2 GPU(s))
2026-04-26 19:09:08,432 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 19:09:08,433 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 19:09:08,433 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 19:09:08,433 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 19:09:08,434 INFO === RLAgent (PPO) retrain ===
2026-04-26 19:09:08,440 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260426_190908
2026-04-26 19:09:08,537 INFO RL phase episode loading: 0.1s (3341 episodes)
2026-04-26 19:09:12.043580: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777230552.290823   47284 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777230552.354141   47284 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777230552.876659   47284 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777230552.876715   47284 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777230552.876721   47284 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777230552.876728   47284 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-26 19:09:26,956 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-26 19:09:30,374 INFO RLAgent: cold start — building new PPO policy
2026-04-26 19:10:40,004 INFO RLAgent: retrain complete, 3341 episodes
2026-04-26 19:10:40,005 INFO RL phase PPO train: 91.5s | total: 91.6s
2026-04-26 19:10:40,016 INFO Retrain complete. Total wall-clock: 91.6s
2026-04-26 19:10:41,647 INFO Model rl: SUCCESS
2026-04-26 19:10:41,648 INFO Step 7b complete — summary: /kaggle/working/Multi-Bot/trading-system/ml_training/metrics/training_7b_summary.json
  DONE  Round 1 - Quality+RL retrain

=== Round 2: BLIND backtest on test window (unseen 2yr) ===
  START Round 2 - Blind backtest (test)
2026-04-26 19:10:42,207 INFO === STEP 6: BACKTEST (round2) ===
2026-04-26 19:10:42,208 INFO BT_WINDOW=round2 — BLIND backtest: 2023-08-07 → 2025-08-05 (test set)
2026-04-26 19:10:42,208 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-26 19:10:42,208 INFO Round 2 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-26 19:10:44,578 WARNING QualityScorer unavailable (weights missing or load failed)
2026-04-26 19:13:01,399 INFO Round 2 backtest — 2641 trades | avg WR=57.7% | avg PF=2.40 | avg Sharpe=2.00
2026-04-26 19:13:01,400 INFO   ml_trader: 2641 trades | WR=57.7% | PF=2.40 | Return=38403987922.8% | DD=11.2% | Sharpe=2.00
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 2641
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2641 rows)
2026-04-26 19:13:02,535 INFO Round 2: wrote 2641 journal entries (total in file: 5982)
  DONE  Round 2 - Blind backtest (test)
  Saved Round 2 result → round2_summary.json
  Journal after Round 2: 5982 entries

=== Round 2 → Retrain Quality + RL (Round 1+2 journal) ===
  START Round 2 - Quality+RL retrain
2026-04-26 19:13:02,830 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-26 19:13:02,836 INFO Journal entries: 5982
2026-04-26 19:13:02,836 INFO --- Training quality ---
2026-04-26 19:13:02,837 INFO Running retrain --model quality
2026-04-26 19:13:03,009 INFO retrain environment: KAGGLE
2026-04-26 19:13:04,654 INFO Device: CUDA (2 GPU(s))
2026-04-26 19:13:04,666 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 19:13:04,666 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 19:13:04,666 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 19:13:04,666 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 19:13:04,668 INFO === QualityScorer retrain ===
2026-04-26 19:13:04,830 INFO NumExpr defaulting to 4 threads.
2026-04-26 19:13:05,032 INFO QualityScorer: CUDA available — using GPU
2026-04-26 19:13:05,406 INFO Quality phase label creation: 0.4s (5982 trades)
2026-04-26 19:13:05,779 INFO QualityScorer: 5982 samples, EV stats={'mean': 0.455624520778656, 'std': 1.2912683486938477, 'n_pos': 3573, 'n_neg': 2409}, device=cuda
2026-04-26 19:13:05,779 INFO QualityScorer: normalised win labels by median_win=0.940 — EV range now [-1, +3]
2026-04-26 19:13:05,979 INFO QualityScorer: DataParallel across 2 GPUs
2026-04-26 19:13:05,979 INFO QualityScorer: cold start
2026-04-26 19:13:05,979 ERROR QualityScorer.train failed: cannot access local variable 'tr_ds' where it is not associated with a value
Traceback (most recent call last):
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1343, in <module>
    main()
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1316, in main
    result = retrain_quality(dry)
             ^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1012, in retrain_quality
    result = model.train(JOURNAL_PATH)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../models/quality_scorer.py", line 437, in train
    _bs      = min(512, len(tr_ds)) if DEVICE.type == "cuda" else min(256, len(tr_ds))
                            ^^^^^
UnboundLocalError: cannot access local variable 'tr_ds' where it is not associated with a value
2026-04-26 19:13:06,516 ERROR retrain quality failed (exit 1)
2026-04-26 19:13:06,516 ERROR Model quality failed: exit 1
2026-04-26 19:13:06,517 INFO --- Training rl ---
2026-04-26 19:13:06,517 INFO Running retrain --model rl
2026-04-26 19:13:06,698 INFO retrain environment: KAGGLE
2026-04-26 19:13:08,313 INFO Device: CUDA (2 GPU(s))
2026-04-26 19:13:08,324 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 19:13:08,324 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 19:13:08,325 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 19:13:08,325 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 19:13:08,326 INFO === RLAgent (PPO) retrain ===
2026-04-26 19:13:08,328 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260426_191308
2026-04-26 19:13:08,510 INFO RL phase episode loading: 0.2s (5982 episodes)
2026-04-26 19:13:09.390768: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777230789.414051   47355 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777230789.421640   47355 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777230789.441639   47355 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777230789.441665   47355 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777230789.441668   47355 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777230789.441671   47355 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-26 19:13:13,896 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-26 19:13:16,872 INFO RLAgent: cold start — building new PPO policy
2026-04-26 19:15:18,350 INFO RLAgent: retrain complete, 5982 episodes
2026-04-26 19:15:18,350 INFO RL phase PPO train: 129.8s | total: 130.0s
2026-04-26 19:15:18,369 INFO Retrain complete. Total wall-clock: 130.0s
2026-04-26 19:15:19,935 INFO Model rl: SUCCESS
2026-04-26 19:15:19,936 INFO Step 7b complete — summary: /kaggle/working/Multi-Bot/trading-system/ml_training/metrics/training_7b_summary.json
  DONE  Round 2 - Quality+RL retrain

=== Round 3: Incremental retrain ALL models on full data ===
  START Retrain gru [full-data retrain]
2026-04-26 19:15:20,269 INFO retrain environment: KAGGLE
2026-04-26 19:15:21,899 INFO Device: CUDA (2 GPU(s))
2026-04-26 19:15:21,911 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 19:15:21,911 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 19:15:21,911 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 19:15:21,911 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 19:15:21,912 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-26 19:15:22,061 INFO NumExpr defaulting to 4 threads.
2026-04-26 19:15:22,260 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 19:15:22,260 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 19:15:22,260 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 19:15:22,502 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-26 19:15:22,502 INFO GRU phase macro_correlations: 0.0s
2026-04-26 19:15:22,502 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-26 19:15:22,504 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260426_191522
2026-04-26 19:15:22,507 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-04-26 19:15:22,659 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:22,679 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:22,693 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:22,700 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:22,701 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 19:15:22,701 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 19:15:22,701 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 19:15:22,702 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 19:15:22,703 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:22,783 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-26 19:15:22,785 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:23,015 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-26 19:15:23,046 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:23,315 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:23,444 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:23,541 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:23,738 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:23,757 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:23,771 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:23,778 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:23,779 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:23,857 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-26 19:15:23,859 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:24,098 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-26 19:15:24,113 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:24,376 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:24,502 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:24,597 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:24,787 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:24,807 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:24,822 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:24,829 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:24,829 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:24,909 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-26 19:15:24,910 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:25,140 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-26 19:15:25,157 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:25,435 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:25,558 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:25,659 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:25,851 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:25,870 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:25,884 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:25,891 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:25,892 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:25,969 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-26 19:15:25,971 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:26,201 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-26 19:15:26,224 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:26,488 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:26,616 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:26,710 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:26,897 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:26,919 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:26,933 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:26,940 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:26,941 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:27,025 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-26 19:15:27,027 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:27,262 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-26 19:15:27,278 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:27,548 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:27,676 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:27,786 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:27,979 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:27,998 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:28,012 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:28,020 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:28,021 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:28,103 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-26 19:15:28,105 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:28,333 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-26 19:15:28,348 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:28,629 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:28,783 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:28,882 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:29,053 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 19:15:29,072 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 19:15:29,086 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 19:15:29,093 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 19:15:29,094 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:29,175 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-26 19:15:29,176 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:29,420 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-26 19:15:29,432 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:29,694 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:29,822 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:29,922 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:30,105 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:30,123 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:30,137 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:30,144 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:30,145 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:30,222 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-26 19:15:30,224 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:30,454 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-26 19:15:30,473 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:30,728 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:30,855 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:30,957 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:31,138 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:31,157 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:31,171 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:31,178 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:31,179 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:31,259 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-26 19:15:31,261 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:31,497 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-26 19:15:31,514 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:31,785 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:31,915 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:32,012 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:32,199 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:32,219 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:32,234 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:32,241 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:15:32,242 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:32,325 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-26 19:15:32,327 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:32,570 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-26 19:15:32,585 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:32,853 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:32,986 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:33,087 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:15:33,379 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 19:15:33,404 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 19:15:33,421 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 19:15:33,430 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 19:15:33,432 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 19:15:33,595 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 19:15:33,599 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 19:15:34,110 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 19:15:34,158 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-26 19:15:34,689 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-26 19:15:34,884 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 19:15:35,019 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 19:15:35,137 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-26 19:15:35,137 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-26 19:15:35,137 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-26 19:16:23,176 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-26 19:16:23,176 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-26 19:16:24,521 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-26 19:16:28,749 INFO train_multi TF=ALL: warm-start detected — using CosineAnnealingLR (lr=3e-05, patience=12)
2026-04-26 19:16:42,442 INFO train_multi TF=ALL epoch 1/50 train=0.5985 val=0.6097
2026-04-26 19:16:42,447 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 19:16:42,447 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 19:16:42,448 INFO train_multi TF=ALL: new best val=0.6097 — saved
2026-04-26 19:16:54,278 INFO train_multi TF=ALL epoch 2/50 train=0.5981 val=0.6109
2026-04-26 19:17:06,167 INFO train_multi TF=ALL epoch 3/50 train=0.5976 val=0.6103
2026-04-26 19:17:17,988 INFO train_multi TF=ALL epoch 4/50 train=0.5976 val=0.6109
2026-04-26 19:17:29,920 INFO train_multi TF=ALL epoch 5/50 train=0.5975 val=0.6110
2026-04-26 19:17:41,861 INFO train_multi TF=ALL epoch 6/50 train=0.5974 val=0.6111
2026-04-26 19:17:53,760 INFO train_multi TF=ALL epoch 7/50 train=0.5975 val=0.6112
2026-04-26 19:18:05,794 INFO train_multi TF=ALL epoch 8/50 train=0.5970 val=0.6106
2026-04-26 19:18:17,685 INFO train_multi TF=ALL epoch 9/50 train=0.5965 val=0.6114
2026-04-26 19:18:29,582 INFO train_multi TF=ALL epoch 10/50 train=0.5966 val=0.6116
2026-04-26 19:18:41,452 INFO train_multi TF=ALL epoch 11/50 train=0.5962 val=0.6118
2026-04-26 19:18:53,201 INFO train_multi TF=ALL epoch 12/50 train=0.5963 val=0.6114
2026-04-26 19:19:04,941 INFO train_multi TF=ALL epoch 13/50 train=0.5962 val=0.6115
2026-04-26 19:19:04,941 INFO train_multi TF=ALL early stop at epoch 13
2026-04-26 19:19:05,085 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-26 19:19:05,086 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-26 19:19:05,086 INFO Retrain complete. Total wall-clock: 223.2s
  DONE  Retrain gru [full-data retrain]
  START Retrain regime [full-data retrain]
2026-04-26 19:19:07,255 INFO retrain environment: KAGGLE
2026-04-26 19:19:08,948 INFO Device: CUDA (2 GPU(s))
2026-04-26 19:19:08,957 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 19:19:08,957 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 19:19:08,957 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 19:19:08,957 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 19:19:08,958 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-26 19:19:09,122 INFO NumExpr defaulting to 4 threads.
2026-04-26 19:19:09,328 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 19:19:09,328 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 19:19:09,328 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 19:19:09,329 INFO Regime phase macro_correlations: 0.0s
2026-04-26 19:19:09,329 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-26 19:19:09,366 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 19:19:09,367 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:19:09,395 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:19:09,409 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:19:09,432 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:19:09,445 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:19:09,469 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:19:09,483 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:19:09,504 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:19:09,519 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:19:09,542 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:19:09,556 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:19:09,577 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:19:09,591 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:19:09,611 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:19:09,625 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:19:09,646 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:19:09,661 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:19:09,682 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:19:09,697 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:19:09,720 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:19:09,737 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 19:19:09,778 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 19:19:10,550 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 19:19:32,792 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 19:19:32,794 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 23.0s
2026-04-26 19:19:32,794 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 19:19:42,862 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 19:19:42,864 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 10.1s
2026-04-26 19:19:42,864 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 19:19:50,644 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 19:19:50,645 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 7.8s
2026-04-26 19:19:50,646 INFO Regime phase GMM HTF total: 40.9s
2026-04-26 19:19:50,646 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 19:21:01,114 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 19:21:01,119 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 70.5s
2026-04-26 19:21:01,121 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 19:21:32,773 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 19:21:32,778 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 31.7s
2026-04-26 19:21:32,778 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 19:21:55,145 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 19:21:55,146 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 22.4s
2026-04-26 19:21:55,146 INFO Regime phase GMM LTF total: 124.5s
2026-04-26 19:21:55,252 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-26 19:21:55,254 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:21:55,255 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:21:55,256 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:21:55,257 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:21:55,258 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:21:55,259 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:21:55,260 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:21:55,261 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:21:55,262 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:21:55,263 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:21:55,265 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 19:21:55,388 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:55,429 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:55,429 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:55,430 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:55,438 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:55,439 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:21:55,841 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-26 19:21:55,842 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-26 19:21:56,018 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:56,050 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:56,051 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:56,051 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:56,059 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:56,060 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:21:56,424 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-26 19:21:56,425 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-26 19:21:56,617 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:56,653 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:56,654 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:56,654 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:56,662 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:56,663 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:21:57,026 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-26 19:21:57,027 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-26 19:21:57,197 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:57,233 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:57,234 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:57,234 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:57,242 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:57,243 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:21:57,644 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-26 19:21:57,646 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-26 19:21:57,826 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:57,861 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:57,862 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:57,863 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:57,871 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:57,872 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:21:58,240 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-26 19:21:58,241 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-26 19:21:58,413 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:58,448 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:58,448 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:58,449 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:58,457 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:58,458 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:21:58,859 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-26 19:21:58,860 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-26 19:21:59,010 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 19:21:59,038 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 19:21:59,038 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 19:21:59,039 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 19:21:59,046 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 19:21:59,047 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:21:59,421 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-26 19:21:59,422 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-26 19:21:59,591 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:59,626 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:59,627 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:59,628 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:59,635 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:21:59,636 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:22:00,002 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-26 19:22:00,003 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-26 19:22:00,172 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:00,206 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:00,207 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:00,207 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:00,215 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:00,216 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:22:00,594 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-26 19:22:00,596 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-26 19:22:00,776 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:00,812 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:00,812 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:00,813 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:00,821 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:00,822 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:22:01,190 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-26 19:22:01,191 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-26 19:22:01,462 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 19:22:01,522 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 19:22:01,524 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 19:22:01,524 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 19:22:01,535 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 19:22:01,536 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 19:22:02,302 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 19:22:02,304 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-26 19:22:02,479 INFO Regime phase HTF dataset build: 7.2s (103290 samples)
2026-04-26 19:22:02,480 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260426_192202
2026-04-26 19:22:02,680 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=34, n_classes=3)
2026-04-26 19:22:02,681 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=38114 dropped=65176 classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728})
2026-04-26 19:22:02,690 INFO RegimeClassifier[mode=htf_bias]: 38114 samples, classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728}, device=cuda
2026-04-26 19:22:02,690 INFO RegimeClassifier: sample weights — mean=0.708  ambiguous(<0.4)=0.0%
2026-04-26 19:22:02,690 INFO RegimeClassifier[mode=htf_bias]: warm start from existing weights
2026-04-26 19:22:02,690 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 19:22:04,897 INFO Regime epoch  1/50 — tr=0.4853 va=1.1361 acc=0.973 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.851}
2026-04-26 19:22:04,969 INFO Regime epoch  2/50 — tr=0.4856 va=1.1376 acc=0.973
2026-04-26 19:22:05,043 INFO Regime epoch  3/50 — tr=0.4857 va=1.1386 acc=0.972
2026-04-26 19:22:05,112 INFO Regime epoch  4/50 — tr=0.4852 va=1.1392 acc=0.972
2026-04-26 19:22:05,183 INFO Regime epoch  5/50 — tr=0.4851 va=1.1381 acc=0.972 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.846}
2026-04-26 19:22:05,251 INFO Regime epoch  6/50 — tr=0.4846 va=1.1357 acc=0.972
2026-04-26 19:22:05,320 INFO Regime epoch  7/50 — tr=0.4846 va=1.1331 acc=0.972
2026-04-26 19:22:05,390 INFO Regime epoch  8/50 — tr=0.4833 va=1.1336 acc=0.973
2026-04-26 19:22:05,460 INFO Regime epoch  9/50 — tr=0.4835 va=1.1338 acc=0.974
2026-04-26 19:22:05,537 INFO Regime epoch 10/50 — tr=0.4828 va=1.1312 acc=0.974 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.857}
2026-04-26 19:22:05,603 INFO Regime epoch 11/50 — tr=0.4825 va=1.1257 acc=0.975
2026-04-26 19:22:05,669 INFO Regime epoch 12/50 — tr=0.4814 va=1.1258 acc=0.975
2026-04-26 19:22:05,741 INFO Regime epoch 13/50 — tr=0.4809 va=1.1250 acc=0.976
2026-04-26 19:22:05,808 INFO Regime epoch 14/50 — tr=0.4802 va=1.1201 acc=0.976
2026-04-26 19:22:05,880 INFO Regime epoch 15/50 — tr=0.4807 va=1.1184 acc=0.977 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.87}
2026-04-26 19:22:05,947 INFO Regime epoch 16/50 — tr=0.4799 va=1.1108 acc=0.978
2026-04-26 19:22:06,018 INFO Regime epoch 17/50 — tr=0.4793 va=1.1124 acc=0.978
2026-04-26 19:22:06,091 INFO Regime epoch 18/50 — tr=0.4792 va=1.1097 acc=0.978
2026-04-26 19:22:06,163 INFO Regime epoch 19/50 — tr=0.4783 va=1.1079 acc=0.979
2026-04-26 19:22:06,238 INFO Regime epoch 20/50 — tr=0.4785 va=1.1056 acc=0.979 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.885}
2026-04-26 19:22:06,310 INFO Regime epoch 21/50 — tr=0.4778 va=1.1044 acc=0.978
2026-04-26 19:22:06,383 INFO Regime epoch 22/50 — tr=0.4776 va=1.1038 acc=0.979
2026-04-26 19:22:06,452 INFO Regime epoch 23/50 — tr=0.4774 va=1.1017 acc=0.980
2026-04-26 19:22:06,521 INFO Regime epoch 24/50 — tr=0.4770 va=1.1021 acc=0.980
2026-04-26 19:22:06,597 INFO Regime epoch 25/50 — tr=0.4762 va=1.0998 acc=0.981 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.894}
2026-04-26 19:22:06,667 INFO Regime epoch 26/50 — tr=0.4762 va=1.0967 acc=0.982
2026-04-26 19:22:06,736 INFO Regime epoch 27/50 — tr=0.4762 va=1.0954 acc=0.982
2026-04-26 19:22:06,804 INFO Regime epoch 28/50 — tr=0.4759 va=1.0958 acc=0.981
2026-04-26 19:22:06,871 INFO Regime epoch 29/50 — tr=0.4759 va=1.0957 acc=0.982
2026-04-26 19:22:06,942 INFO Regime epoch 30/50 — tr=0.4754 va=1.0950 acc=0.982 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.898}
2026-04-26 19:22:07,008 INFO Regime epoch 31/50 — tr=0.4759 va=1.0926 acc=0.982
2026-04-26 19:22:07,074 INFO Regime epoch 32/50 — tr=0.4752 va=1.0902 acc=0.982
2026-04-26 19:22:07,144 INFO Regime epoch 33/50 — tr=0.4752 va=1.0906 acc=0.982
2026-04-26 19:22:07,213 INFO Regime epoch 34/50 — tr=0.4756 va=1.0893 acc=0.982
2026-04-26 19:22:07,284 INFO Regime epoch 35/50 — tr=0.4751 va=1.0878 acc=0.982 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.902}
2026-04-26 19:22:07,353 INFO Regime epoch 36/50 — tr=0.4747 va=1.0877 acc=0.982
2026-04-26 19:22:07,422 INFO Regime epoch 37/50 — tr=0.4749 va=1.0872 acc=0.982
2026-04-26 19:22:07,489 INFO Regime epoch 38/50 — tr=0.4749 va=1.0879 acc=0.982
2026-04-26 19:22:07,561 INFO Regime epoch 39/50 — tr=0.4746 va=1.0870 acc=0.982
2026-04-26 19:22:07,637 INFO Regime epoch 40/50 — tr=0.4744 va=1.0869 acc=0.982 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.902}
2026-04-26 19:22:07,705 INFO Regime epoch 41/50 — tr=0.4742 va=1.0844 acc=0.983
2026-04-26 19:22:07,776 INFO Regime epoch 42/50 — tr=0.4742 va=1.0857 acc=0.982
2026-04-26 19:22:07,847 INFO Regime epoch 43/50 — tr=0.4744 va=1.0858 acc=0.982
2026-04-26 19:22:07,914 INFO Regime epoch 44/50 — tr=0.4744 va=1.0845 acc=0.983
2026-04-26 19:22:07,986 INFO Regime epoch 45/50 — tr=0.4742 va=1.0833 acc=0.983 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.905}
2026-04-26 19:22:08,053 INFO Regime epoch 46/50 — tr=0.4743 va=1.0845 acc=0.983
2026-04-26 19:22:08,120 INFO Regime epoch 47/50 — tr=0.4742 va=1.0865 acc=0.982
2026-04-26 19:22:08,190 INFO Regime epoch 48/50 — tr=0.4744 va=1.0839 acc=0.983
2026-04-26 19:22:08,256 INFO Regime epoch 49/50 — tr=0.4746 va=1.0863 acc=0.983
2026-04-26 19:22:08,327 INFO Regime epoch 50/50 — tr=0.4745 va=1.0867 acc=0.982 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.902}
2026-04-26 19:22:08,335 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 19:22:08,335 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 19:22:08,336 INFO Regime phase HTF train: 5.7s
2026-04-26 19:22:08,470 INFO Regime HTF complete: acc=0.983, n=103290 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.905}
2026-04-26 19:22:08,472 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 19:22:08,623 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 19:22:08,625 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 11.41578947368421, 'BIAS_DOWN': 10.635761589403973, 'BIAS_NEUTRAL': 17.960468521229867}
2026-04-26 19:22:08,629 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 4338, 'mean': 0.00011110466582543088, 'mean_over_std': 0.027632581210502927}, 'BIAS_DOWN': {'n': 3212, 'mean': 6.063430222660252e-05, 'mean_over_std': 0.013898489590377157}, 'BIAS_NEUTRAL': {'n': 12266, 'mean': 1.2151554902814046e-05, 'mean_over_std': 0.0031957000513162357}}
2026-04-26 19:22:08,630 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 3651, 'mean': 0.0002852242108836959, 'mean_over_std': 0.07015017232491755}, 'BIAS_DOWN': {'n': 2644, 'mean': -0.00014900749618981248, 'mean_over_std': -0.03333037041298237}, 'BIAS_NEUTRAL': {'n': 1376, 'mean': 9.595797941249082e-05, 'mean_over_std': 0.02492411099521803}}
2026-04-26 19:22:08,630 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-26 19:22:08,632 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:22:08,634 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:22:08,635 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:22:08,638 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:22:08,640 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:22:08,642 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:22:08,644 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:22:08,648 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:22:08,650 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:22:08,653 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:22:08,660 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 19:22:08,673 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:08,678 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:08,679 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:08,679 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:08,679 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:08,683 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:22:09,312 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-26 19:22:09,315 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-26 19:22:09,449 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:09,452 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:09,453 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:09,453 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:09,453 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:09,455 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:22:10,041 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-26 19:22:10,044 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-26 19:22:10,184 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:10,187 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:10,188 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:10,188 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:10,188 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:10,191 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:22:10,870 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-26 19:22:10,874 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-26 19:22:11,010 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:11,013 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:11,014 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:11,014 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:11,014 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:11,016 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:22:11,596 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-26 19:22:11,599 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-26 19:22:11,731 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:11,735 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:11,736 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:11,736 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:11,736 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:11,738 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:22:12,307 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-26 19:22:12,310 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-26 19:22:12,441 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:12,444 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:12,445 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:12,446 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:12,446 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:12,448 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:22:13,035 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-26 19:22:13,038 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-26 19:22:13,168 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 19:22:13,169 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 19:22:13,170 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 19:22:13,170 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 19:22:13,171 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 19:22:13,172 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:22:13,743 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-26 19:22:13,746 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-26 19:22:13,882 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:13,884 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:13,885 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:13,886 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:13,886 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:13,888 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:22:14,457 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-26 19:22:14,460 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-26 19:22:14,595 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:14,597 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:14,598 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:14,598 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:14,598 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:14,600 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:22:15,170 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-26 19:22:15,173 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-26 19:22:15,304 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:15,306 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:15,307 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:15,307 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:15,308 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 19:22:15,310 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 19:22:15,894 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-26 19:22:15,897 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-26 19:22:16,041 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 19:22:16,048 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 19:22:16,049 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 19:22:16,050 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 19:22:16,050 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 19:22:16,053 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 19:22:17,291 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 19:22:17,296 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-26 19:22:17,592 INFO Regime phase LTF dataset build: 9.0s (401471 samples)
2026-04-26 19:22:17,592 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260426_192217
2026-04-26 19:22:17,597 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=18, n_classes=4)
2026-04-26 19:22:17,600 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=299185 dropped=102286 classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127})
2026-04-26 19:22:17,666 INFO RegimeClassifier[mode=ltf_behaviour]: 299185 samples, classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127}, device=cuda
2026-04-26 19:22:17,667 INFO RegimeClassifier: sample weights — mean=0.693  ambiguous(<0.4)=0.0%
2026-04-26 19:22:17,667 INFO RegimeClassifier[mode=ltf_behaviour]: warm start from existing weights
2026-04-26 19:22:17,668 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 19:22:18,225 INFO Regime epoch  1/50 — tr=0.6255 va=1.2407 acc=0.825 per_class={'TRENDING': 0.803, 'RANGING': 0.774, 'CONSOLIDATING': 0.823, 'VOLATILE': 0.897}
2026-04-26 19:22:18,763 INFO Regime epoch  2/50 — tr=0.6253 va=1.2424 acc=0.825
2026-04-26 19:22:19,266 INFO Regime epoch  3/50 — tr=0.6254 va=1.2416 acc=0.826
2026-04-26 19:22:19,757 INFO Regime epoch  4/50 — tr=0.6254 va=1.2455 acc=0.826
2026-04-26 19:22:20,298 INFO Regime epoch  5/50 — tr=0.6250 va=1.2435 acc=0.828 per_class={'TRENDING': 0.808, 'RANGING': 0.768, 'CONSOLIDATING': 0.829, 'VOLATILE': 0.898}
2026-04-26 19:22:20,792 INFO Regime epoch  6/50 — tr=0.6250 va=1.2397 acc=0.826
2026-04-26 19:22:21,280 INFO Regime epoch  7/50 — tr=0.6250 va=1.2424 acc=0.827
2026-04-26 19:22:21,770 INFO Regime epoch  8/50 — tr=0.6248 va=1.2376 acc=0.827
2026-04-26 19:22:22,258 INFO Regime epoch  9/50 — tr=0.6243 va=1.2380 acc=0.827
2026-04-26 19:22:22,797 INFO Regime epoch 10/50 — tr=0.6243 va=1.2411 acc=0.828 per_class={'TRENDING': 0.81, 'RANGING': 0.768, 'CONSOLIDATING': 0.839, 'VOLATILE': 0.892}
2026-04-26 19:22:23,282 INFO Regime epoch 11/50 — tr=0.6238 va=1.2418 acc=0.832
2026-04-26 19:22:23,787 INFO Regime epoch 12/50 — tr=0.6236 va=1.2377 acc=0.831
2026-04-26 19:22:24,307 INFO Regime epoch 13/50 — tr=0.6234 va=1.2331 acc=0.829
2026-04-26 19:22:24,816 INFO Regime epoch 14/50 — tr=0.6232 va=1.2378 acc=0.832
2026-04-26 19:22:25,373 INFO Regime epoch 15/50 — tr=0.6229 va=1.2380 acc=0.830 per_class={'TRENDING': 0.811, 'RANGING': 0.771, 'CONSOLIDATING': 0.838, 'VOLATILE': 0.898}
2026-04-26 19:22:25,876 INFO Regime epoch 16/50 — tr=0.6228 va=1.2379 acc=0.834
2026-04-26 19:22:26,382 INFO Regime epoch 17/50 — tr=0.6225 va=1.2359 acc=0.832
2026-04-26 19:22:26,907 INFO Regime epoch 18/50 — tr=0.6224 va=1.2339 acc=0.831
2026-04-26 19:22:27,406 INFO Regime epoch 19/50 — tr=0.6222 va=1.2320 acc=0.831
2026-04-26 19:22:27,951 INFO Regime epoch 20/50 — tr=0.6220 va=1.2299 acc=0.832 per_class={'TRENDING': 0.816, 'RANGING': 0.77, 'CONSOLIDATING': 0.851, 'VOLATILE': 0.889}
2026-04-26 19:22:28,467 INFO Regime epoch 21/50 — tr=0.6220 va=1.2355 acc=0.833
2026-04-26 19:22:29,005 INFO Regime epoch 22/50 — tr=0.6216 va=1.2317 acc=0.832
2026-04-26 19:22:29,533 INFO Regime epoch 23/50 — tr=0.6215 va=1.2305 acc=0.835
2026-04-26 19:22:30,026 INFO Regime epoch 24/50 — tr=0.6215 va=1.2288 acc=0.833
2026-04-26 19:22:30,552 INFO Regime epoch 25/50 — tr=0.6213 va=1.2304 acc=0.834 per_class={'TRENDING': 0.816, 'RANGING': 0.771, 'CONSOLIDATING': 0.847, 'VOLATILE': 0.897}
2026-04-26 19:22:31,088 INFO Regime epoch 26/50 — tr=0.6212 va=1.2266 acc=0.832
2026-04-26 19:22:31,608 INFO Regime epoch 27/50 — tr=0.6213 va=1.2358 acc=0.835
2026-04-26 19:22:32,120 INFO Regime epoch 28/50 — tr=0.6211 va=1.2320 acc=0.833
2026-04-26 19:22:32,606 INFO Regime epoch 29/50 — tr=0.6210 va=1.2280 acc=0.832
2026-04-26 19:22:33,138 INFO Regime epoch 30/50 — tr=0.6204 va=1.2257 acc=0.834 per_class={'TRENDING': 0.817, 'RANGING': 0.772, 'CONSOLIDATING': 0.856, 'VOLATILE': 0.888}
2026-04-26 19:22:33,666 INFO Regime epoch 31/50 — tr=0.6209 va=1.2318 acc=0.835
2026-04-26 19:22:34,184 INFO Regime epoch 32/50 — tr=0.6207 va=1.2283 acc=0.834
2026-04-26 19:22:34,682 INFO Regime epoch 33/50 — tr=0.6207 va=1.2250 acc=0.833
2026-04-26 19:22:35,180 INFO Regime epoch 34/50 — tr=0.6203 va=1.2288 acc=0.836
2026-04-26 19:22:35,727 INFO Regime epoch 35/50 — tr=0.6202 va=1.2269 acc=0.835 per_class={'TRENDING': 0.817, 'RANGING': 0.773, 'CONSOLIDATING': 0.854, 'VOLATILE': 0.894}
2026-04-26 19:22:36,225 INFO Regime epoch 36/50 — tr=0.6203 va=1.2293 acc=0.834
2026-04-26 19:22:36,721 INFO Regime epoch 37/50 — tr=0.6203 va=1.2272 acc=0.836
2026-04-26 19:22:37,212 INFO Regime epoch 38/50 — tr=0.6203 va=1.2289 acc=0.836
2026-04-26 19:22:37,716 INFO Regime epoch 39/50 — tr=0.6202 va=1.2262 acc=0.834
2026-04-26 19:22:38,266 INFO Regime epoch 40/50 — tr=0.6203 va=1.2289 acc=0.835 per_class={'TRENDING': 0.815, 'RANGING': 0.772, 'CONSOLIDATING': 0.855, 'VOLATILE': 0.898}
2026-04-26 19:22:38,817 INFO Regime epoch 41/50 — tr=0.6199 va=1.2259 acc=0.836
2026-04-26 19:22:39,331 INFO Regime epoch 42/50 — tr=0.6202 va=1.2274 acc=0.834
2026-04-26 19:22:39,831 INFO Regime epoch 43/50 — tr=0.6200 va=1.2297 acc=0.835
2026-04-26 19:22:39,831 INFO Regime early stop at epoch 43 (no_improve=10)
2026-04-26 19:22:39,872 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 19:22:39,872 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 19:22:39,875 INFO Regime phase LTF train: 22.3s
2026-04-26 19:22:40,008 INFO Regime LTF complete: acc=0.833, n=401471 per_class={'TRENDING': 0.812, 'RANGING': 0.776, 'CONSOLIDATING': 0.847, 'VOLATILE': 0.899}
2026-04-26 19:22:40,011 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 19:22:40,512 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 19:22:40,516 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 9.355291913830783, 'RANGING': 6.096303199751476, 'CONSOLIDATING': 5.598885793871866, 'VOLATILE': 6.771351107094442}
2026-04-26 19:22:40,525 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 29965, 'mean': 2.149062259083323e-05, 'mean_over_std': 0.011155601434349563}, 'RANGING': {'n': 19623, 'mean': -3.254493608872846e-06, 'mean_over_std': -0.001761415548292494}, 'CONSOLIDATING': {'n': 10050, 'mean': -3.039756158146958e-06, 'mean_over_std': -0.001792488091618151}, 'VOLATILE': {'n': 14985, 'mean': 1.8102764458593193e-05, 'mean_over_std': 0.006974696291378742}}
2026-04-26 19:22:40,525 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 26145, 'mean': 2.874435605138053e-05, 'mean_over_std': 0.01595450632691917}, 'RANGING': {'n': 7858, 'mean': 1.7829776785521727e-05, 'mean_over_std': 0.011851561804467463}, 'CONSOLIDATING': {'n': 8543, 'mean': 7.845231961250924e-06, 'mean_over_std': 0.004915839521549481}, 'VOLATILE': {'n': 13636, 'mean': 5.480114267306668e-06, 'mean_over_std': 0.0020378119917389375}}
2026-04-26 19:22:40,526 INFO Regime retrain total: 211.6s (504761 samples)
2026-04-26 19:22:40,528 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 19:22:40,528 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 19:22:40,528 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 19:22:40,529 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-26 19:22:40,529 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-26 19:22:40,529 INFO Retrain complete. Total wall-clock: 211.6s
  DONE  Retrain regime [full-data retrain]
  START Retrain quality [full-data retrain]
2026-04-26 19:22:41,916 INFO retrain environment: KAGGLE
2026-04-26 19:22:43,539 INFO Device: CUDA (2 GPU(s))
2026-04-26 19:22:43,550 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 19:22:43,550 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 19:22:43,550 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 19:22:43,550 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 19:22:43,551 INFO === QualityScorer retrain ===
2026-04-26 19:22:43,694 INFO NumExpr defaulting to 4 threads.
2026-04-26 19:22:43,893 INFO QualityScorer: CUDA available — using GPU
2026-04-26 19:22:44,273 INFO Quality phase label creation: 0.4s (5982 trades)
2026-04-26 19:22:44,632 INFO QualityScorer: 5982 samples, EV stats={'mean': 0.455624520778656, 'std': 1.2912683486938477, 'n_pos': 3573, 'n_neg': 2409}, device=cuda
2026-04-26 19:22:44,633 INFO QualityScorer: normalised win labels by median_win=0.940 — EV range now [-1, +3]
2026-04-26 19:22:44,829 INFO QualityScorer: DataParallel across 2 GPUs
2026-04-26 19:22:44,829 INFO QualityScorer: cold start
2026-04-26 19:22:44,829 ERROR QualityScorer.train failed: cannot access local variable 'tr_ds' where it is not associated with a value
Traceback (most recent call last):
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1343, in <module>
    main()
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1316, in main
    result = retrain_quality(dry)
             ^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1012, in retrain_quality
    result = model.train(JOURNAL_PATH)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../models/quality_scorer.py", line 437, in train
    _bs      = min(512, len(tr_ds)) if DEVICE.type == "cuda" else min(256, len(tr_ds))
                            ^^^^^
UnboundLocalError: cannot access local variable 'tr_ds' where it is not associated with a value
  WARN  Retrain quality failed (exit 1) — continuing
  START Retrain rl [full-data retrain]
2026-04-26 19:22:45,526 INFO retrain environment: KAGGLE
2026-04-26 19:22:47,133 INFO Device: CUDA (2 GPU(s))
2026-04-26 19:22:47,144 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 19:22:47,144 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 19:22:47,144 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 19:22:47,144 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 19:22:47,145 INFO === RLAgent (PPO) retrain ===
2026-04-26 19:22:47,147 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260426_192247
2026-04-26 19:22:47,303 INFO RL phase episode loading: 0.2s (5982 episodes)
2026-04-26 19:22:48.151749: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777231368.176224   68044 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777231368.184456   68044 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777231368.204234   68044 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777231368.204259   68044 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777231368.204262   68044 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777231368.204264   68044 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-26 19:22:52,632 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-26 19:22:55,562 INFO RLAgent: cold start — building new PPO policy
2026-04-26 19:24:52,973 INFO RLAgent: retrain complete, 5982 episodes
2026-04-26 19:24:52,974 INFO RL phase PPO train: 125.7s | total: 125.8s
2026-04-26 19:24:52,994 INFO Retrain complete. Total wall-clock: 125.8s
  DONE  Retrain rl [full-data retrain]

=== Round 3: Backtest on last 3yr (post-retrain evaluation) ===
  START Round 3 - Post-retrain backtest (last 3yr)
2026-04-26 19:24:55,106 INFO === STEP 6: BACKTEST (round3) ===
2026-04-26 19:24:55,107 INFO BT_WINDOW=round3 — post-retrain eval: 2022-08-05 → 2025-08-05 (last 3yr)
2026-04-26 19:24:55,108 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-04-26 19:24:55,108 INFO Round 3 — running backtest: 2022-08-05 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-26 19:24:57,426 WARNING QualityScorer unavailable (weights missing or load failed)
2026-04-26 19:28:09,363 INFO Round 3 backtest — 6050 trades | avg WR=59.1% | avg PF=2.71 | avg Sharpe=1.42
2026-04-26 19:28:09,363 INFO   ml_trader: 6050 trades | WR=59.1% | PF=2.71 | Return=205116876073010104107008.0% | DD=12.6% | Sharpe=1.42
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 6050
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (6050 rows)
2026-04-26 19:28:11,671 INFO Round 3: wrote 6050 journal entries (total in file: 12032)
  DONE  Round 3 - Post-retrain backtest (last 3yr)
  Saved Round 3 result → round3_summary.json

======================================================================
  BLIND BACKTEST PIPELINE COMPLETE
======================================================================
  Round 1 (val window)          trades=?  WR=0.0%  PF=0.000  Sharpe=0.000
  Round 2 (blind test)          trades=?  WR=0.0%  PF=0.000  Sharpe=0.000
  Round 3 (last 3yr)            trades=?  WR=0.0%  PF=0.000  Sharpe=0.000


WARNING: GITHUB_TOKEN not set — skipping GitHub push