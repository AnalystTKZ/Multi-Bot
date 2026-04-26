All scripts and inputs verified.

=== Phase 0-5: Data preparation ===
  SKIP  Step 0 - Resample
  SKIP  Step 1 - Inventory
  SKIP  Step 2 - Cleaning
  SKIP  Step 3 - Alignment
  SKIP  Step 4 - Features
  START Step 5 - Split
2026-04-26 17:58:08,100 INFO Loading feature-engineered data...
2026-04-26 17:58:08,693 INFO Loaded 221743 rows, 202 features
2026-04-26 17:58:08,694 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-26 17:58:08,697 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-26 17:58:08,697 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-26 17:58:08,697 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-26 17:58:08,697 INFO No leakage confirmed: train < val < test timestamps

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
2026-04-26 17:58:11,008 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-26 17:58:11,008 INFO --- Training regime ---
2026-04-26 17:58:11,009 INFO Running retrain --model regime
2026-04-26 17:58:11,180 INFO retrain environment: KAGGLE
2026-04-26 17:58:12,751 INFO Device: CUDA (2 GPU(s))
2026-04-26 17:58:12,761 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 17:58:12,761 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 17:58:12,761 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 17:58:12,763 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 17:58:12,764 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-26 17:58:12,929 INFO NumExpr defaulting to 4 threads.
2026-04-26 17:58:13,167 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 17:58:13,167 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 17:58:13,168 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 17:58:13,168 INFO Regime phase macro_correlations: 0.0s
2026-04-26 17:58:13,168 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-26 17:58:13,204 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 17:58:13,205 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:58:13,230 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:58:13,243 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:58:13,265 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:58:13,278 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:58:13,299 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:58:13,312 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:58:13,333 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:58:13,347 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:58:13,367 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:58:13,380 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:58:13,406 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:58:13,422 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:58:13,440 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:58:13,453 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:58:13,474 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:58:13,487 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:58:13,507 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:58:13,521 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:58:13,542 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:58:13,557 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:58:13,593 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:58:14,672 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 17:58:36,840 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 17:58:36,841 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 23.2s
2026-04-26 17:58:36,845 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 17:58:46,473 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 17:58:46,477 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 9.6s
2026-04-26 17:58:46,479 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 17:58:53,684 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 17:58:53,688 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 7.2s
2026-04-26 17:58:53,688 INFO Regime phase GMM HTF total: 40.1s
2026-04-26 17:58:53,689 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 18:00:01,547 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 18:00:01,550 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 67.9s
2026-04-26 18:00:01,550 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 18:00:31,600 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 18:00:31,601 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 30.1s
2026-04-26 18:00:31,601 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 18:00:52,700 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 18:00:52,701 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 21.1s
2026-04-26 18:00:52,701 INFO Regime phase GMM LTF total: 119.0s
2026-04-26 18:00:52,799 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-26 18:00:52,801 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:00:52,802 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:00:52,803 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:00:52,804 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:00:52,808 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:00:52,809 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:00:52,813 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:00:52,815 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:00:52,819 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:00:52,820 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:00:52,822 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:00:52,950 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:52,990 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:52,991 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:52,991 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:52,998 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:52,999 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:00:53,431 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-26 18:00:53,433 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-26 18:00:53,606 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:53,638 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:53,639 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:53,639 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:53,647 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:53,648 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:00:54,005 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-26 18:00:54,006 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-26 18:00:54,190 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:54,226 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:54,227 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:54,227 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:54,234 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:54,235 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:00:54,594 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-26 18:00:54,595 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-26 18:00:54,766 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:54,801 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:54,802 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:54,802 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:54,810 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:54,811 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:00:55,172 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-26 18:00:55,173 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-26 18:00:55,357 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:55,391 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:55,392 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:55,392 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:55,399 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:55,400 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:00:55,747 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-26 18:00:55,748 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-26 18:00:55,913 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:55,945 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:55,946 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:55,947 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:55,954 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:55,955 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:00:56,310 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-26 18:00:56,311 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-26 18:00:56,456 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:00:56,482 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:00:56,483 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:00:56,483 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:00:56,490 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:00:56,491 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:00:56,836 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-26 18:00:56,837 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-26 18:00:56,997 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:57,031 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:57,031 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:57,032 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:57,040 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:57,041 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:00:57,404 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-26 18:00:57,405 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-26 18:00:57,568 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:57,600 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:57,600 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:57,601 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:57,609 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:57,610 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:00:57,962 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-26 18:00:57,963 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-26 18:00:58,138 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:58,172 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:58,173 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:58,173 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:58,181 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:00:58,182 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:00:58,540 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-26 18:00:58,541 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-26 18:00:58,804 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:00:58,859 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:00:58,860 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:00:58,860 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:00:58,870 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:00:58,871 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:00:59,636 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 18:00:59,637 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-26 18:00:59,791 INFO Regime phase HTF dataset build: 7.0s (103290 samples)
2026-04-26 18:00:59,792 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=38114 dropped=65176 classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728})
2026-04-26 18:00:59,801 INFO RegimeClassifier[mode=htf_bias]: 38114 samples, classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728}, device=cuda
2026-04-26 18:00:59,801 INFO RegimeClassifier: sample weights — mean=0.708  ambiguous(<0.4)=0.0%
2026-04-26 18:01:00,077 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-04-26 18:01:00,077 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 18:01:04,547 INFO Regime epoch  1/50 — tr=0.6926 va=2.0328 acc=0.532 per_class={'BIAS_UP': 0.948, 'BIAS_DOWN': 0.221, 'BIAS_NEUTRAL': 0.012}
2026-04-26 18:01:04,625 INFO Regime epoch  2/50 — tr=0.6887 va=2.0038 acc=0.626
2026-04-26 18:01:04,701 INFO Regime epoch  3/50 — tr=0.6797 va=1.9789 acc=0.687
2026-04-26 18:01:04,770 INFO Regime epoch  4/50 — tr=0.6692 va=1.9365 acc=0.737
2026-04-26 18:01:04,840 INFO Regime epoch  5/50 — tr=0.6530 va=1.8702 acc=0.791 per_class={'BIAS_UP': 0.955, 'BIAS_DOWN': 0.81, 'BIAS_NEUTRAL': 0.32}
2026-04-26 18:01:04,905 INFO Regime epoch  6/50 — tr=0.6316 va=1.7838 acc=0.839
2026-04-26 18:01:04,974 INFO Regime epoch  7/50 — tr=0.6092 va=1.6885 acc=0.878
2026-04-26 18:01:05,046 INFO Regime epoch  8/50 — tr=0.5873 va=1.5952 acc=0.901
2026-04-26 18:01:05,112 INFO Regime epoch  9/50 — tr=0.5676 va=1.5151 acc=0.917
2026-04-26 18:01:05,182 INFO Regime epoch 10/50 — tr=0.5518 va=1.4478 acc=0.926 per_class={'BIAS_UP': 0.986, 'BIAS_DOWN': 0.967, 'BIAS_NEUTRAL': 0.687}
2026-04-26 18:01:05,248 INFO Regime epoch 11/50 — tr=0.5371 va=1.3944 acc=0.930
2026-04-26 18:01:05,315 INFO Regime epoch 12/50 — tr=0.5279 va=1.3564 acc=0.933
2026-04-26 18:01:05,383 INFO Regime epoch 13/50 — tr=0.5201 va=1.3229 acc=0.936
2026-04-26 18:01:05,449 INFO Regime epoch 14/50 — tr=0.5143 va=1.2953 acc=0.940
2026-04-26 18:01:05,518 INFO Regime epoch 15/50 — tr=0.5097 va=1.2754 acc=0.942 per_class={'BIAS_UP': 0.996, 'BIAS_DOWN': 0.991, 'BIAS_NEUTRAL': 0.705}
2026-04-26 18:01:05,584 INFO Regime epoch 16/50 — tr=0.5057 va=1.2557 acc=0.944
2026-04-26 18:01:05,649 INFO Regime epoch 17/50 — tr=0.5033 va=1.2365 acc=0.946
2026-04-26 18:01:05,718 INFO Regime epoch 18/50 — tr=0.5002 va=1.2247 acc=0.948
2026-04-26 18:01:05,782 INFO Regime epoch 19/50 — tr=0.4981 va=1.2134 acc=0.950
2026-04-26 18:01:05,855 INFO Regime epoch 20/50 — tr=0.4964 va=1.2023 acc=0.952 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.995, 'BIAS_NEUTRAL': 0.745}
2026-04-26 18:01:05,920 INFO Regime epoch 21/50 — tr=0.4936 va=1.1945 acc=0.953
2026-04-26 18:01:05,990 INFO Regime epoch 22/50 — tr=0.4932 va=1.1864 acc=0.954
2026-04-26 18:01:06,060 INFO Regime epoch 23/50 — tr=0.4916 va=1.1768 acc=0.956
2026-04-26 18:01:06,125 INFO Regime epoch 24/50 — tr=0.4907 va=1.1723 acc=0.957
2026-04-26 18:01:06,196 INFO Regime epoch 25/50 — tr=0.4887 va=1.1675 acc=0.958 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.995, 'BIAS_NEUTRAL': 0.78}
2026-04-26 18:01:06,263 INFO Regime epoch 26/50 — tr=0.4877 va=1.1609 acc=0.959
2026-04-26 18:01:06,334 INFO Regime epoch 27/50 — tr=0.4867 va=1.1522 acc=0.961
2026-04-26 18:01:06,405 INFO Regime epoch 28/50 — tr=0.4864 va=1.1474 acc=0.962
2026-04-26 18:01:06,476 INFO Regime epoch 29/50 — tr=0.4856 va=1.1452 acc=0.961
2026-04-26 18:01:06,545 INFO Regime epoch 30/50 — tr=0.4849 va=1.1385 acc=0.963 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.996, 'BIAS_NEUTRAL': 0.807}
2026-04-26 18:01:06,611 INFO Regime epoch 31/50 — tr=0.4839 va=1.1372 acc=0.964
2026-04-26 18:01:06,675 INFO Regime epoch 32/50 — tr=0.4830 va=1.1364 acc=0.964
2026-04-26 18:01:06,746 INFO Regime epoch 33/50 — tr=0.4826 va=1.1352 acc=0.965
2026-04-26 18:01:06,810 INFO Regime epoch 34/50 — tr=0.4820 va=1.1307 acc=0.966
2026-04-26 18:01:06,884 INFO Regime epoch 35/50 — tr=0.4818 va=1.1286 acc=0.967 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.997, 'BIAS_NEUTRAL': 0.823}
2026-04-26 18:01:06,950 INFO Regime epoch 36/50 — tr=0.4815 va=1.1248 acc=0.968
2026-04-26 18:01:07,014 INFO Regime epoch 37/50 — tr=0.4809 va=1.1255 acc=0.968
2026-04-26 18:01:07,081 INFO Regime epoch 38/50 — tr=0.4813 va=1.1228 acc=0.968
2026-04-26 18:01:07,146 INFO Regime epoch 39/50 — tr=0.4809 va=1.1214 acc=0.969
2026-04-26 18:01:07,221 INFO Regime epoch 40/50 — tr=0.4803 va=1.1205 acc=0.969 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.997, 'BIAS_NEUTRAL': 0.834}
2026-04-26 18:01:07,286 INFO Regime epoch 41/50 — tr=0.4802 va=1.1204 acc=0.969
2026-04-26 18:01:07,356 INFO Regime epoch 42/50 — tr=0.4802 va=1.1203 acc=0.969
2026-04-26 18:01:07,424 INFO Regime epoch 43/50 — tr=0.4803 va=1.1199 acc=0.969
2026-04-26 18:01:07,491 INFO Regime epoch 44/50 — tr=0.4798 va=1.1189 acc=0.969
2026-04-26 18:01:07,565 INFO Regime epoch 45/50 — tr=0.4795 va=1.1203 acc=0.969 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.997, 'BIAS_NEUTRAL': 0.834}
2026-04-26 18:01:07,637 INFO Regime epoch 46/50 — tr=0.4799 va=1.1194 acc=0.969
2026-04-26 18:01:07,708 INFO Regime epoch 47/50 — tr=0.4801 va=1.1189 acc=0.969
2026-04-26 18:01:07,775 INFO Regime epoch 48/50 — tr=0.4801 va=1.1166 acc=0.969
2026-04-26 18:01:07,847 INFO Regime epoch 49/50 — tr=0.4798 va=1.1179 acc=0.969
2026-04-26 18:01:07,918 INFO Regime epoch 50/50 — tr=0.4798 va=1.1213 acc=0.969 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.997, 'BIAS_NEUTRAL': 0.832}
2026-04-26 18:01:07,927 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 18:01:07,928 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 18:01:07,928 INFO Regime phase HTF train: 8.1s
2026-04-26 18:01:08,057 INFO Regime HTF complete: acc=0.969, n=103290 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.997, 'BIAS_NEUTRAL': 0.836}
2026-04-26 18:01:08,059 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:01:08,207 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 18:01:08,214 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 11.41578947368421, 'BIAS_DOWN': 10.635761589403973, 'BIAS_NEUTRAL': 17.960468521229867}
2026-04-26 18:01:08,218 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 4338, 'mean': 0.00011110466582543088, 'mean_over_std': 0.027632581210502927}, 'BIAS_DOWN': {'n': 3212, 'mean': 6.063430222660252e-05, 'mean_over_std': 0.013898489590377157}, 'BIAS_NEUTRAL': {'n': 12266, 'mean': 1.2151554902814046e-05, 'mean_over_std': 0.0031957000513162357}}
2026-04-26 18:01:08,219 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 3651, 'mean': 0.0002852242108836959, 'mean_over_std': 0.07015017232491755}, 'BIAS_DOWN': {'n': 2644, 'mean': -0.00014900749618981248, 'mean_over_std': -0.03333037041298237}, 'BIAS_NEUTRAL': {'n': 1376, 'mean': 9.595797941249082e-05, 'mean_over_std': 0.02492411099521803}}
2026-04-26 18:01:08,219 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-26 18:01:08,221 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:08,222 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:08,224 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:08,226 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:08,227 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:08,229 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:08,230 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:08,232 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:08,233 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:08,235 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:08,237 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:01:08,248 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:08,253 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:08,254 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:08,254 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:08,254 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:08,257 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:08,865 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-26 18:01:08,867 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-26 18:01:08,998 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:09,000 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:09,001 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:09,002 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:09,002 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:09,004 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:09,572 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-26 18:01:09,576 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-26 18:01:09,706 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:09,709 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:09,710 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:09,710 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:09,710 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:09,712 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:10,275 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-26 18:01:10,278 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-26 18:01:10,409 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:10,411 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:10,412 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:10,412 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:10,413 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:10,415 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:10,966 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-26 18:01:10,969 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-26 18:01:11,096 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:11,099 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:11,100 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:11,100 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:11,100 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:11,102 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:11,665 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-26 18:01:11,667 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-26 18:01:11,801 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:11,804 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:11,804 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:11,805 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:11,805 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:11,807 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:12,364 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-26 18:01:12,367 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-26 18:01:12,500 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:01:12,502 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:01:12,502 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:01:12,503 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:01:12,503 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:01:12,505 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:13,081 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-26 18:01:13,084 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-26 18:01:13,235 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:13,237 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:13,238 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:13,238 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:13,239 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:13,241 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:13,829 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-26 18:01:13,831 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-26 18:01:13,963 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:13,965 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:13,966 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:13,966 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:13,966 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:13,968 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:14,527 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-26 18:01:14,530 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-26 18:01:14,665 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:14,668 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:14,668 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:14,669 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:14,669 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:14,671 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:15,235 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-26 18:01:15,237 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-26 18:01:15,381 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:01:15,385 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:01:15,386 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:01:15,386 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:01:15,387 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:01:15,390 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:01:16,629 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 18:01:16,634 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-26 18:01:16,920 INFO Regime phase LTF dataset build: 8.7s (401471 samples)
2026-04-26 18:01:16,923 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=299185 dropped=102286 classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127})
2026-04-26 18:01:16,980 INFO RegimeClassifier[mode=ltf_behaviour]: 299185 samples, classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127}, device=cuda
2026-04-26 18:01:16,981 INFO RegimeClassifier: sample weights — mean=0.693  ambiguous(<0.4)=0.0%
2026-04-26 18:01:16,983 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-04-26 18:01:16,983 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 18:01:17,549 INFO Regime epoch  1/50 — tr=0.9018 va=2.2771 acc=0.334 per_class={'TRENDING': 0.492, 'RANGING': 0.017, 'CONSOLIDATING': 0.16, 'VOLATILE': 0.326}
2026-04-26 18:01:18,048 INFO Regime epoch  2/50 — tr=0.8719 va=2.1837 acc=0.437
2026-04-26 18:01:18,534 INFO Regime epoch  3/50 — tr=0.8267 va=2.0025 acc=0.513
2026-04-26 18:01:19,027 INFO Regime epoch  4/50 — tr=0.7775 va=1.7932 acc=0.576
2026-04-26 18:01:19,582 INFO Regime epoch  5/50 — tr=0.7385 va=1.6270 acc=0.617 per_class={'TRENDING': 0.498, 'RANGING': 0.581, 'CONSOLIDATING': 0.479, 'VOLATILE': 0.953}
2026-04-26 18:01:20,060 INFO Regime epoch  6/50 — tr=0.7131 va=1.5274 acc=0.645
2026-04-26 18:01:20,535 INFO Regime epoch  7/50 — tr=0.6972 va=1.4786 acc=0.665
2026-04-26 18:01:21,026 INFO Regime epoch  8/50 — tr=0.6850 va=1.4540 acc=0.682
2026-04-26 18:01:21,530 INFO Regime epoch  9/50 — tr=0.6760 va=1.4327 acc=0.697
2026-04-26 18:01:22,077 INFO Regime epoch 10/50 — tr=0.6692 va=1.4142 acc=0.714 per_class={'TRENDING': 0.596, 'RANGING': 0.721, 'CONSOLIDATING': 0.717, 'VOLATILE': 0.932}
2026-04-26 18:01:22,578 INFO Regime epoch 11/50 — tr=0.6639 va=1.3961 acc=0.726
2026-04-26 18:01:23,089 INFO Regime epoch 12/50 — tr=0.6593 va=1.3796 acc=0.738
2026-04-26 18:01:23,609 INFO Regime epoch 13/50 — tr=0.6556 va=1.3672 acc=0.747
2026-04-26 18:01:24,093 INFO Regime epoch 14/50 — tr=0.6527 va=1.3564 acc=0.755
2026-04-26 18:01:24,625 INFO Regime epoch 15/50 — tr=0.6498 va=1.3411 acc=0.763 per_class={'TRENDING': 0.699, 'RANGING': 0.753, 'CONSOLIDATING': 0.723, 'VOLATILE': 0.918}
2026-04-26 18:01:25,124 INFO Regime epoch 16/50 — tr=0.6477 va=1.3345 acc=0.768
2026-04-26 18:01:25,617 INFO Regime epoch 17/50 — tr=0.6453 va=1.3245 acc=0.774
2026-04-26 18:01:26,098 INFO Regime epoch 18/50 — tr=0.6436 va=1.3160 acc=0.779
2026-04-26 18:01:26,603 INFO Regime epoch 19/50 — tr=0.6420 va=1.3140 acc=0.783
2026-04-26 18:01:27,124 INFO Regime epoch 20/50 — tr=0.6404 va=1.3035 acc=0.788 per_class={'TRENDING': 0.748, 'RANGING': 0.763, 'CONSOLIDATING': 0.742, 'VOLATILE': 0.908}
2026-04-26 18:01:27,619 INFO Regime epoch 21/50 — tr=0.6391 va=1.3011 acc=0.789
2026-04-26 18:01:28,127 INFO Regime epoch 22/50 — tr=0.6379 va=1.2928 acc=0.793
2026-04-26 18:01:28,612 INFO Regime epoch 23/50 — tr=0.6367 va=1.2921 acc=0.797
2026-04-26 18:01:29,107 INFO Regime epoch 24/50 — tr=0.6358 va=1.2890 acc=0.798
2026-04-26 18:01:29,646 INFO Regime epoch 25/50 — tr=0.6346 va=1.2828 acc=0.799 per_class={'TRENDING': 0.764, 'RANGING': 0.765, 'CONSOLIDATING': 0.761, 'VOLATILE': 0.909}
2026-04-26 18:01:30,156 INFO Regime epoch 26/50 — tr=0.6338 va=1.2781 acc=0.803
2026-04-26 18:01:30,632 INFO Regime epoch 27/50 — tr=0.6328 va=1.2747 acc=0.804
2026-04-26 18:01:31,108 INFO Regime epoch 28/50 — tr=0.6319 va=1.2715 acc=0.806
2026-04-26 18:01:31,627 INFO Regime epoch 29/50 — tr=0.6315 va=1.2692 acc=0.806
2026-04-26 18:01:32,158 INFO Regime epoch 30/50 — tr=0.6307 va=1.2695 acc=0.809 per_class={'TRENDING': 0.778, 'RANGING': 0.771, 'CONSOLIDATING': 0.775, 'VOLATILE': 0.91}
2026-04-26 18:01:32,646 INFO Regime epoch 31/50 — tr=0.6303 va=1.2660 acc=0.812
2026-04-26 18:01:33,223 INFO Regime epoch 32/50 — tr=0.6294 va=1.2601 acc=0.813
2026-04-26 18:01:33,726 INFO Regime epoch 33/50 — tr=0.6290 va=1.2618 acc=0.815
2026-04-26 18:01:34,237 INFO Regime epoch 34/50 — tr=0.6288 va=1.2616 acc=0.813
2026-04-26 18:01:34,788 INFO Regime epoch 35/50 — tr=0.6282 va=1.2597 acc=0.813 per_class={'TRENDING': 0.783, 'RANGING': 0.777, 'CONSOLIDATING': 0.79, 'VOLATILE': 0.907}
2026-04-26 18:01:35,264 INFO Regime epoch 36/50 — tr=0.6279 va=1.2583 acc=0.817
2026-04-26 18:01:35,774 INFO Regime epoch 37/50 — tr=0.6277 va=1.2566 acc=0.817
2026-04-26 18:01:36,279 INFO Regime epoch 38/50 — tr=0.6274 va=1.2541 acc=0.815
2026-04-26 18:01:36,786 INFO Regime epoch 39/50 — tr=0.6270 va=1.2544 acc=0.818
2026-04-26 18:01:37,313 INFO Regime epoch 40/50 — tr=0.6268 va=1.2504 acc=0.816 per_class={'TRENDING': 0.786, 'RANGING': 0.773, 'CONSOLIDATING': 0.814, 'VOLATILE': 0.898}
2026-04-26 18:01:37,824 INFO Regime epoch 41/50 — tr=0.6267 va=1.2526 acc=0.819
2026-04-26 18:01:38,324 INFO Regime epoch 42/50 — tr=0.6264 va=1.2523 acc=0.819
2026-04-26 18:01:38,834 INFO Regime epoch 43/50 — tr=0.6264 va=1.2530 acc=0.820
2026-04-26 18:01:39,342 INFO Regime epoch 44/50 — tr=0.6263 va=1.2489 acc=0.818
2026-04-26 18:01:39,877 INFO Regime epoch 45/50 — tr=0.6267 va=1.2520 acc=0.820 per_class={'TRENDING': 0.792, 'RANGING': 0.774, 'CONSOLIDATING': 0.811, 'VOLATILE': 0.905}
2026-04-26 18:01:40,397 INFO Regime epoch 46/50 — tr=0.6262 va=1.2500 acc=0.820
2026-04-26 18:01:40,896 INFO Regime epoch 47/50 — tr=0.6264 va=1.2504 acc=0.819
2026-04-26 18:01:41,404 INFO Regime epoch 48/50 — tr=0.6262 va=1.2530 acc=0.820
2026-04-26 18:01:41,887 INFO Regime epoch 49/50 — tr=0.6263 va=1.2500 acc=0.818
2026-04-26 18:01:42,414 INFO Regime epoch 50/50 — tr=0.6260 va=1.2488 acc=0.818 per_class={'TRENDING': 0.789, 'RANGING': 0.777, 'CONSOLIDATING': 0.81, 'VOLATILE': 0.902}
2026-04-26 18:01:42,454 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 18:01:42,455 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 18:01:42,457 INFO Regime phase LTF train: 25.5s
2026-04-26 18:01:42,581 INFO Regime LTF complete: acc=0.818, n=401471 per_class={'TRENDING': 0.789, 'RANGING': 0.777, 'CONSOLIDATING': 0.81, 'VOLATILE': 0.902}
2026-04-26 18:01:42,585 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:01:43,071 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 18:01:43,076 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 9.355291913830783, 'RANGING': 6.096303199751476, 'CONSOLIDATING': 5.598885793871866, 'VOLATILE': 6.771351107094442}
2026-04-26 18:01:43,083 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 29965, 'mean': 2.149062259083323e-05, 'mean_over_std': 0.011155601434349563}, 'RANGING': {'n': 19623, 'mean': -3.254493608872846e-06, 'mean_over_std': -0.001761415548292494}, 'CONSOLIDATING': {'n': 10050, 'mean': -3.039756158146958e-06, 'mean_over_std': -0.001792488091618151}, 'VOLATILE': {'n': 14985, 'mean': 1.8102764458593193e-05, 'mean_over_std': 0.006974696291378742}}
2026-04-26 18:01:43,084 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 26145, 'mean': 2.874435605138053e-05, 'mean_over_std': 0.01595450632691917}, 'RANGING': {'n': 7858, 'mean': 1.7829776785521727e-05, 'mean_over_std': 0.011851561804467463}, 'CONSOLIDATING': {'n': 8543, 'mean': 7.845231961250924e-06, 'mean_over_std': 0.004915839521549481}, 'VOLATILE': {'n': 13636, 'mean': 5.480114267306668e-06, 'mean_over_std': 0.0020378119917389375}}
2026-04-26 18:01:43,085 INFO Regime retrain total: 210.3s (504761 samples)
2026-04-26 18:01:43,104 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 18:01:43,105 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 18:01:43,105 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 18:01:43,105 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-26 18:01:43,105 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-26 18:01:43,105 INFO Retrain complete. Total wall-clock: 210.3s
2026-04-26 18:01:45,330 INFO Model regime: SUCCESS
2026-04-26 18:01:45,330 INFO --- Training gru ---
2026-04-26 18:01:45,330 INFO Running retrain --model gru
2026-04-26 18:01:45,584 INFO retrain environment: KAGGLE
2026-04-26 18:01:47,193 INFO Device: CUDA (2 GPU(s))
2026-04-26 18:01:47,204 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 18:01:47,204 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 18:01:47,204 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 18:01:47,204 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 18:01:47,205 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-26 18:01:47,351 INFO NumExpr defaulting to 4 threads.
2026-04-26 18:01:47,543 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 18:01:47,543 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 18:01:47,543 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 18:01:47,544 INFO GRU phase macro_correlations: 0.0s
2026-04-26 18:01:47,544 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-26 18:01:47,544 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260426_180147
2026-04-26 18:01:47,547 WARNING WeightsManifest: no manifest at /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json — assuming compatible (run a full retrain to generate one)
2026-04-26 18:01:47,690 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:47,708 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:47,723 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:47,729 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:47,731 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 18:01:47,731 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 18:01:47,731 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 18:01:47,731 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 18:01:47,732 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:47,808 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-26 18:01:47,810 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:48,023 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-26 18:01:48,052 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:48,317 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:48,440 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:48,538 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:48,737 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:48,756 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:48,769 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:48,776 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:48,777 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:48,846 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-26 18:01:48,848 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:49,058 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-26 18:01:49,073 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:49,330 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:49,453 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:49,547 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:49,734 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:49,756 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:49,770 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:49,777 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:49,778 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:49,852 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-26 18:01:49,854 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:50,072 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-26 18:01:50,086 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:50,345 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:50,466 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:50,558 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:50,742 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:50,761 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:50,775 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:50,781 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:50,782 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:50,855 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-26 18:01:50,857 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:51,081 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-26 18:01:51,104 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:51,363 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:51,490 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:51,586 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:51,768 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:51,787 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:51,801 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:51,808 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:51,809 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:51,884 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-26 18:01:51,886 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:52,109 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-26 18:01:52,124 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:52,386 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:52,508 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:52,605 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:52,787 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:52,807 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:52,831 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:52,845 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:52,846 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:52,928 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-26 18:01:52,930 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:53,176 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-26 18:01:53,191 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:53,451 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:53,578 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:53,668 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:53,833 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:01:53,849 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:01:53,862 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:01:53,867 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 18:01:53,868 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:53,943 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-26 18:01:53,945 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:54,168 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-26 18:01:54,181 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:54,429 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:54,554 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:54,643 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:54,820 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:54,839 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:54,853 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:54,860 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:54,861 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:54,938 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-26 18:01:54,940 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:55,169 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-26 18:01:55,186 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:55,442 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:55,569 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:55,663 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:55,852 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:55,870 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:55,883 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:55,889 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:55,890 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:55,961 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-26 18:01:55,963 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:56,180 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-26 18:01:56,195 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:56,456 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:56,594 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:56,690 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:56,872 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:56,891 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:56,904 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:56,911 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 18:01:56,911 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:56,981 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-26 18:01:56,982 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:57,200 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-26 18:01:57,215 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:57,479 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:57,606 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:57,698 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 18:01:57,982 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:01:58,007 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:01:58,023 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:01:58,032 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 18:01:58,033 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:01:58,182 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 18:01:58,186 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:01:58,654 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 18:01:58,700 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:01:59,204 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:01:59,396 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:01:59,520 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 18:01:59,637 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-26 18:01:59,892 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-04-26 18:01:59,892 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-04-26 18:01:59,892 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-26 18:01:59,892 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-26 18:02:46,166 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-26 18:02:46,166 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-26 18:02:47,485 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-26 18:02:51,508 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=5)
2026-04-26 18:03:05,045 INFO train_multi TF=ALL epoch 1/50 train=0.8860 val=0.8770
2026-04-26 18:03:05,054 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:03:05,054 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:03:05,054 INFO train_multi TF=ALL: new best val=0.8770 — saved
2026-04-26 18:03:16,725 INFO train_multi TF=ALL epoch 2/50 train=0.8673 val=0.8474
2026-04-26 18:03:16,729 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:03:16,730 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:03:16,730 INFO train_multi TF=ALL: new best val=0.8474 — saved
2026-04-26 18:03:28,314 INFO train_multi TF=ALL epoch 3/50 train=0.7680 val=0.6885
2026-04-26 18:03:28,319 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:03:28,319 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:03:28,319 INFO train_multi TF=ALL: new best val=0.6885 — saved
2026-04-26 18:03:39,903 INFO train_multi TF=ALL epoch 4/50 train=0.6909 val=0.6878
2026-04-26 18:03:39,907 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:03:39,908 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:03:39,908 INFO train_multi TF=ALL: new best val=0.6878 — saved
2026-04-26 18:03:51,393 INFO train_multi TF=ALL epoch 5/50 train=0.6901 val=0.6878
2026-04-26 18:03:51,397 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:03:51,397 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:03:51,397 INFO train_multi TF=ALL: new best val=0.6878 — saved
2026-04-26 18:04:02,945 INFO train_multi TF=ALL epoch 6/50 train=0.6895 val=0.6878
2026-04-26 18:04:02,949 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:04:02,949 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:04:02,950 INFO train_multi TF=ALL: new best val=0.6878 — saved
2026-04-26 18:04:14,479 INFO train_multi TF=ALL epoch 7/50 train=0.6892 val=0.6878
2026-04-26 18:04:26,050 INFO train_multi TF=ALL epoch 8/50 train=0.6889 val=0.6877
2026-04-26 18:04:26,054 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:04:26,054 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:04:26,054 INFO train_multi TF=ALL: new best val=0.6877 — saved
2026-04-26 18:04:37,652 INFO train_multi TF=ALL epoch 9/50 train=0.6887 val=0.6876
2026-04-26 18:04:37,656 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:04:37,656 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:04:37,656 INFO train_multi TF=ALL: new best val=0.6876 — saved
2026-04-26 18:04:49,134 INFO train_multi TF=ALL epoch 10/50 train=0.6884 val=0.6869
2026-04-26 18:04:49,138 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:04:49,138 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:04:49,138 INFO train_multi TF=ALL: new best val=0.6869 — saved
2026-04-26 18:05:00,725 INFO train_multi TF=ALL epoch 11/50 train=0.6869 val=0.6849
2026-04-26 18:05:00,729 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:05:00,729 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:05:00,729 INFO train_multi TF=ALL: new best val=0.6849 — saved
2026-04-26 18:05:12,242 INFO train_multi TF=ALL epoch 12/50 train=0.6817 val=0.6766
2026-04-26 18:05:12,246 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:05:12,246 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:05:12,246 INFO train_multi TF=ALL: new best val=0.6766 — saved
2026-04-26 18:05:23,789 INFO train_multi TF=ALL epoch 13/50 train=0.6718 val=0.6649
2026-04-26 18:05:23,794 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:05:23,794 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:05:23,794 INFO train_multi TF=ALL: new best val=0.6649 — saved
2026-04-26 18:05:35,363 INFO train_multi TF=ALL epoch 14/50 train=0.6592 val=0.6514
2026-04-26 18:05:35,367 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:05:35,367 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:05:35,367 INFO train_multi TF=ALL: new best val=0.6514 — saved
2026-04-26 18:05:46,839 INFO train_multi TF=ALL epoch 15/50 train=0.6479 val=0.6399
2026-04-26 18:05:46,843 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:05:46,844 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:05:46,844 INFO train_multi TF=ALL: new best val=0.6399 — saved
2026-04-26 18:05:58,368 INFO train_multi TF=ALL epoch 16/50 train=0.6392 val=0.6331
2026-04-26 18:05:58,372 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:05:58,372 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:05:58,372 INFO train_multi TF=ALL: new best val=0.6331 — saved
2026-04-26 18:06:09,967 INFO train_multi TF=ALL epoch 17/50 train=0.6341 val=0.6294
2026-04-26 18:06:09,972 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:06:09,972 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:06:09,972 INFO train_multi TF=ALL: new best val=0.6294 — saved
2026-04-26 18:06:21,533 INFO train_multi TF=ALL epoch 18/50 train=0.6299 val=0.6266
2026-04-26 18:06:21,538 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:06:21,538 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:06:21,538 INFO train_multi TF=ALL: new best val=0.6266 — saved
2026-04-26 18:06:33,206 INFO train_multi TF=ALL epoch 19/50 train=0.6270 val=0.6209
2026-04-26 18:06:33,210 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:06:33,210 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:06:33,210 INFO train_multi TF=ALL: new best val=0.6209 — saved
2026-04-26 18:06:44,819 INFO train_multi TF=ALL epoch 20/50 train=0.6237 val=0.6213
2026-04-26 18:06:56,406 INFO train_multi TF=ALL epoch 21/50 train=0.6212 val=0.6216
2026-04-26 18:07:07,913 INFO train_multi TF=ALL epoch 22/50 train=0.6185 val=0.6203
2026-04-26 18:07:07,918 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:07:07,918 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:07:07,918 INFO train_multi TF=ALL: new best val=0.6203 — saved
2026-04-26 18:07:19,449 INFO train_multi TF=ALL epoch 23/50 train=0.6165 val=0.6186
2026-04-26 18:07:19,454 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:07:19,454 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:07:19,454 INFO train_multi TF=ALL: new best val=0.6186 — saved
2026-04-26 18:07:31,081 INFO train_multi TF=ALL epoch 24/50 train=0.6148 val=0.6197
2026-04-26 18:07:42,708 INFO train_multi TF=ALL epoch 25/50 train=0.6127 val=0.6172
2026-04-26 18:07:42,712 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:07:42,712 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:07:42,712 INFO train_multi TF=ALL: new best val=0.6172 — saved
2026-04-26 18:07:54,308 INFO train_multi TF=ALL epoch 26/50 train=0.6108 val=0.6161
2026-04-26 18:07:54,312 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:07:54,312 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:07:54,312 INFO train_multi TF=ALL: new best val=0.6161 — saved
2026-04-26 18:08:05,795 INFO train_multi TF=ALL epoch 27/50 train=0.6090 val=0.6147
2026-04-26 18:08:05,799 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:08:05,799 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:08:05,799 INFO train_multi TF=ALL: new best val=0.6147 — saved
2026-04-26 18:08:17,409 INFO train_multi TF=ALL epoch 28/50 train=0.6077 val=0.6189
2026-04-26 18:08:28,925 INFO train_multi TF=ALL epoch 29/50 train=0.6058 val=0.6141
2026-04-26 18:08:28,929 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:08:28,929 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:08:28,930 INFO train_multi TF=ALL: new best val=0.6141 — saved
2026-04-26 18:08:40,533 INFO train_multi TF=ALL epoch 30/50 train=0.6045 val=0.6147
2026-04-26 18:08:51,997 INFO train_multi TF=ALL epoch 31/50 train=0.6034 val=0.6146
2026-04-26 18:09:03,585 INFO train_multi TF=ALL epoch 32/50 train=0.6017 val=0.6112
2026-04-26 18:09:03,590 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:09:03,590 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:09:03,590 INFO train_multi TF=ALL: new best val=0.6112 — saved
2026-04-26 18:09:15,172 INFO train_multi TF=ALL epoch 33/50 train=0.6003 val=0.6142
2026-04-26 18:09:26,709 INFO train_multi TF=ALL epoch 34/50 train=0.5987 val=0.6122
2026-04-26 18:09:38,225 INFO train_multi TF=ALL epoch 35/50 train=0.5973 val=0.6116
2026-04-26 18:09:49,700 INFO train_multi TF=ALL epoch 36/50 train=0.5963 val=0.6106
2026-04-26 18:09:49,704 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:09:49,704 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:09:49,704 INFO train_multi TF=ALL: new best val=0.6106 — saved
2026-04-26 18:10:01,264 INFO train_multi TF=ALL epoch 37/50 train=0.5951 val=0.6115
2026-04-26 18:10:12,730 INFO train_multi TF=ALL epoch 38/50 train=0.5939 val=0.6129
2026-04-26 18:10:24,312 INFO train_multi TF=ALL epoch 39/50 train=0.5929 val=0.6122
2026-04-26 18:10:35,953 INFO train_multi TF=ALL epoch 40/50 train=0.5919 val=0.6104
2026-04-26 18:10:35,958 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:10:35,958 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:10:35,958 INFO train_multi TF=ALL: new best val=0.6104 — saved
2026-04-26 18:10:47,561 INFO train_multi TF=ALL epoch 41/50 train=0.5905 val=0.6122
2026-04-26 18:10:59,058 INFO train_multi TF=ALL epoch 42/50 train=0.5900 val=0.6142
2026-04-26 18:11:10,611 INFO train_multi TF=ALL epoch 43/50 train=0.5885 val=0.6099
2026-04-26 18:11:10,615 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 18:11:10,615 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:11:10,615 INFO train_multi TF=ALL: new best val=0.6099 — saved
2026-04-26 18:11:22,143 INFO train_multi TF=ALL epoch 44/50 train=0.5874 val=0.6101
2026-04-26 18:11:33,748 INFO train_multi TF=ALL epoch 45/50 train=0.5857 val=0.6111
2026-04-26 18:11:45,264 INFO train_multi TF=ALL epoch 46/50 train=0.5850 val=0.6126
2026-04-26 18:11:56,735 INFO train_multi TF=ALL epoch 47/50 train=0.5835 val=0.6121
2026-04-26 18:12:08,196 INFO train_multi TF=ALL epoch 48/50 train=0.5828 val=0.6127
2026-04-26 18:12:08,196 INFO train_multi TF=ALL early stop at epoch 48
2026-04-26 18:12:08,333 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-26 18:12:08,333 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-26 18:12:08,334 INFO Retrain complete. Total wall-clock: 621.1s
2026-04-26 18:12:10,229 INFO Model gru: SUCCESS
2026-04-26 18:12:10,229 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 18:12:10,230 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 18:12:10,230 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 18:12:10,230 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-04-26 18:12:10,230 INFO   [DEFERRED] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip (expected after Round 1)
2026-04-26 18:12:10,230 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-04-26 18:12:10,230 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer', 'rl_ppo']
2026-04-26 18:12:10,231 INFO Saved 5 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  START Round 1 - Backtest (val)
2026-04-26 18:12:10,911 INFO === STEP 6: BACKTEST (round1) ===
2026-04-26 18:12:10,912 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-04-26 18:12:10,912 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-26 18:12:10,912 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
2026-04-26 18:12:13,209 WARNING QualityScorer unavailable (weights missing or load failed)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260426_181212.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)                3   0.0%   0.00   -3.0%  0.0%  0.0%   3.0% -1932.34
  gate_diagnostics: bars=468696 no_signal=2 quality_block=0 session_skip=317895 density=0 pm_reject=0 daily_skip=150796 cooldown=0
  no_signal_reasons: weak_gru_direction=2

Calibration Summary:
  all          [N/A] Insufficient data: 3 samples
  ml_trader    [N/A] Insufficient data: 3 samples
2026-04-26 18:14:19,045 INFO Round 1 backtest — 3 trades | avg WR=0.0% | avg PF=0.00 | avg Sharpe=-1932.35
2026-04-26 18:14:19,045 INFO   ml_trader: 3 trades | WR=0.0% | PF=0.00 | Return=-3.0% | DD=3.0% | Sharpe=-1932.35
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 3
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (3 rows)
2026-04-26 18:14:19,264 INFO Round 1: wrote 3 journal entries (total in file: 3)
  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 3 entries

=== Round 1 → Retrain Quality + RL ===
  START Round 1 - Quality+RL retrain
  DONE  Round 1 - Quality+RL retrain

=== Round 2: BLIND backtest on test window (unseen 2yr) ===
  START Round 2 - Blind backtest (test)
2026-04-26 18:14:19,474 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-26 18:14:19,474 INFO Journal entries: 3
2026-04-26 18:14:19,474 WARNING Journal has only 3 entries (need 50) — backtest produced too few trades. Skipping Quality+RL training. Check step6 logs.
2026-04-26 18:14:19,950 INFO === STEP 6: BACKTEST (round2) ===
2026-04-26 18:14:19,951 INFO BT_WINDOW=round2 — BLIND backtest: 2023-08-07 → 2025-08-05 (test set)
2026-04-26 18:14:19,951 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-26 18:14:19,951 INFO Round 2 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-26 18:14:22,260 WARNING QualityScorer unavailable (weights missing or load failed)
2026-04-26 18:16:28,978 INFO Round 2 backtest — 8 trades | avg WR=12.5% | avg PF=0.07 | avg Sharpe=-22.50
2026-04-26 18:16:28,978 INFO   ml_trader: 8 trades | WR=12.5% | PF=0.07 | Return=-5.3% | DD=5.3% | Sharpe=-22.50
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 8
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (8 rows)
2026-04-26 18:16:29,198 INFO Round 2: wrote 8 journal entries (total in file: 11)
  DONE  Round 2 - Blind backtest (test)
  Saved Round 2 result → round2_summary.json
  Journal after Round 2: 11 entries

=== Round 2 → Retrain Quality + RL (Round 1+2 journal) ===
  START Round 2 - Quality+RL retrain
  DONE  Round 2 - Quality+RL retrain

=== Round 3: Incremental retrain ALL models on full data ===
  START Retrain gru [full-data retrain]
2026-04-26 18:16:29,414 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-26 18:16:29,414 INFO Journal entries: 11
2026-04-26 18:16:29,414 WARNING Journal has only 11 entries (need 50) — backtest produced too few trades. Skipping Quality+RL training. Check step6 logs.
2026-04-26 18:16:29,602 INFO retrain environment: KAGGLE
2026-04-26 18:16:31,204 INFO Device: CUDA (2 GPU(s))
2026-04-26 18:16:31,215 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 18:16:31,216 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 18:16:31,216 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 18:16:31,216 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 18:16:31,217 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-26 18:16:31,359 INFO NumExpr defaulting to 4 threads.
2026-04-26 18:16:31,552 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 18:16:31,552 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 18:16:31,552 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 18:16:31,795 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-26 18:16:31,796 INFO GRU phase macro_correlations: 0.0s
2026-04-26 18:16:31,796 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-26 18:16:31,797 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260426_181631
2026-04-26 18:16:31,800 INFO GRU feature contract unchanged (input_size=74) — incremental retrain