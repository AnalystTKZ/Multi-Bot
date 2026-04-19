  Cleared done-check: training_summary.json
  Cleared done-check: training_7b_summary.json
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

=== Running pipeline ===
  SKIP  Step 0 - Resample
  SKIP  Step 1 - Inventory
  SKIP  Step 2 - Cleaning
  SKIP  Step 3 - Alignment
  SKIP  Step 4 - Features
  START Step 5 - Split
2026-04-19 02:35:58,207 INFO Loading feature-engineered data...
2026-04-19 02:35:58,697 INFO Loaded 221743 rows, 202 features
2026-04-19 02:35:58,698 INFO Train: 155220 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:35:58,698 INFO Val:   33261 bars (2022-08-16 → 2024-03-05)
2026-04-19 02:35:58,698 INFO Test:  33262 bars (2024-03-05 → 2025-08-05)
2026-04-19 02:35:58,698 INFO No leakage confirmed: train < val < test timestamps

=== SPLIT COMPLETE (no shuffling, time-based) ===
  Train:      155,220 bars  2016-01-04 → 2022-08-16
  Validation:  33,261 bars  2022-08-16 → 2024-03-05
  Test:        33,262 bars  2024-03-05 → 2025-08-05
  Features: 202
  Leakage check: PASS
  DONE  Step 5 - Split
  START Step 7a - GRU+Regime
2026-04-19 02:36:01,210 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-19 02:36:01,210 INFO --- Training regime ---
2026-04-19 02:36:01,210 INFO Running retrain --model regime
2026-04-19 02:36:01,387 INFO retrain environment: KAGGLE
2026-04-19 02:36:03,076 INFO Device: CUDA (2 GPU(s))
2026-04-19 02:36:03,087 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 02:36:03,088 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 02:36:03,089 INFO === RegimeClassifier retrain (dual-TF cascade: 4H bias + 1H structure) ===
2026-04-19 02:36:03,234 INFO NumExpr defaulting to 4 threads.
2026-04-19 02:36:03,443 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-19 02:36:03,443 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 02:36:03,443 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 02:36:03,646 INFO Split boundaries loaded — train≤2022-08-16  val≤2024-03-05  test≤2025-08-05
2026-04-19 02:36:03,648 INFO Loaded AUDUSD/1H split=train: 38804 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:03,727 INFO Loaded EURGBP/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:03,802 INFO Loaded EURJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:03,877 INFO Loaded EURUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:03,952 INFO Loaded GBPJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:04,028 INFO Loaded GBPUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:04,098 INFO Loaded NZDUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:04,175 INFO Loaded USDCAD/1H split=train: 38808 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:04,254 INFO Loaded USDCHF/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:04,331 INFO Loaded USDJPY/1H split=train: 38811 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:04,421 INFO Loaded XAUUSD/1H split=train: 80734 bars (2009-03-15 → 2022-08-16)
2026-04-19 02:36:04,487 INFO Regime: fitting per-group GMMs on 4H data (dollar / cross / gold)...
2026-04-19 02:36:04,503 INFO Loaded AUDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:04,519 INFO Loaded EURGBP/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:04,535 INFO Loaded EURJPY/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:04,551 INFO Loaded EURUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:04,566 INFO Loaded GBPJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:04,580 INFO Loaded GBPUSD/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:04,595 INFO Loaded NZDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:04,610 INFO Loaded USDCAD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:04,625 INFO Loaded USDCHF/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:04,641 INFO Loaded USDJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:04,659 INFO Loaded XAUUSD/4H split=train: 21467 bars (2009-03-15 → 2022-08-16)
2026-04-19 02:36:05,442 INFO GMM fit: timeframe=4H → n_bar=50
2026-04-19 02:36:12,356 INFO GMM fitted on 69344 samples — cluster→regime: {0: 2, 1: 1, 2: 3, 3: 0} dist: {'TRENDING_UP': 1, 'TRENDING_DOWN': 1, 'RANGING': 1, 'VOLATILE': 1}
2026-04-19 02:36:12,358 INFO Regime: GMM 'dollar' fitted on 7 4H dfs (n_bar=50)
2026-04-19 02:36:12,358 INFO GMM fit: timeframe=4H → n_bar=50
2026-04-19 02:36:14,834 INFO GMM fitted on 29719 samples — cluster→regime: {0: 2, 1: 2, 2: 3, 3: 0} dist: {'TRENDING_UP': 1, 'TRENDING_DOWN': 0, 'RANGING': 2, 'VOLATILE': 1}
2026-04-19 02:36:14,834 INFO Regime: GMM 'cross' fitted on 3 4H dfs (n_bar=50)
2026-04-19 02:36:14,834 INFO GMM fit: timeframe=4H → n_bar=50
2026-04-19 02:36:15,477 INFO GMM fitted on 10708 samples — cluster→regime: {0: 2, 1: 2, 2: 0, 3: 2} dist: {'TRENDING_UP': 1, 'TRENDING_DOWN': 0, 'RANGING': 3, 'VOLATILE': 0}
2026-04-19 02:36:15,478 INFO Regime: GMM 'gold' fitted on 1 4H dfs (n_bar=50)
2026-04-19 02:36:15,583 INFO Regime: training 4H bias classifier...
2026-04-19 02:36:15,585 INFO Loaded AUDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:15,586 INFO Loaded EURGBP/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:15,587 INFO Loaded EURJPY/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:15,588 INFO Loaded EURUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:15,590 INFO Loaded GBPJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:15,591 INFO Loaded GBPUSD/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:15,592 INFO Loaded NZDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:15,593 INFO Loaded USDCAD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:15,594 INFO Loaded USDCHF/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:15,595 INFO Loaded USDJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:15,597 INFO Loaded XAUUSD/4H split=train: 21467 bars (2009-03-15 → 2022-08-16)
2026-04-19 02:36:15,726 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:15,767 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:15,768 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:15,768 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:15,778 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:15,779 INFO Loaded AUDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:16,243 INFO Rule labels [4H]: {'TRENDING_UP': 1833, 'TRENDING_DOWN': 1829, 'RANGING': 4293, 'VOLATILE': 2002}  ambiguous(conf<0.4)=5746 (total=9957)  short_runs_zeroed=4432
2026-04-19 02:36:16,244 INFO Regime[4H]: collected AUDUSD — 9907 samples (group=dollar)
2026-04-19 02:36:16,423 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:16,460 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:16,460 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:16,461 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:16,470 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:16,471 INFO Loaded EURGBP/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:16,887 INFO Rule labels [4H]: {'TRENDING_UP': 1702, 'TRENDING_DOWN': 1928, 'RANGING': 4297, 'VOLATILE': 2030}  ambiguous(conf<0.4)=5770 (total=9957)  short_runs_zeroed=4585
2026-04-19 02:36:16,888 INFO Regime[4H]: collected EURGBP — 9907 samples (group=cross)
2026-04-19 02:36:17,060 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:17,096 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:17,097 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:17,098 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:17,106 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:17,107 INFO Loaded EURJPY/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:17,521 INFO Rule labels [4H]: {'TRENDING_UP': 2037, 'TRENDING_DOWN': 1677, 'RANGING': 4228, 'VOLATILE': 2015}  ambiguous(conf<0.4)=5748 (total=9957)  short_runs_zeroed=4550
2026-04-19 02:36:17,522 INFO Regime[4H]: collected EURJPY — 9907 samples (group=cross)
2026-04-19 02:36:17,690 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:17,727 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:17,727 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:17,728 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:17,737 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:17,738 INFO Loaded EURUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:18,150 INFO Rule labels [4H]: {'TRENDING_UP': 1509, 'TRENDING_DOWN': 2039, 'RANGING': 4405, 'VOLATILE': 2004}  ambiguous(conf<0.4)=5805 (total=9957)  short_runs_zeroed=4449
2026-04-19 02:36:18,151 INFO Regime[4H]: collected EURUSD — 9907 samples (group=dollar)
2026-04-19 02:36:18,325 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:18,362 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:18,362 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:18,363 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:18,371 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:18,372 INFO Loaded GBPJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:18,788 INFO Rule labels [4H]: {'TRENDING_UP': 1903, 'TRENDING_DOWN': 1665, 'RANGING': 4358, 'VOLATILE': 2032}  ambiguous(conf<0.4)=5547 (total=9958)  short_runs_zeroed=4143
2026-04-19 02:36:18,789 INFO Regime[4H]: collected GBPJPY — 9908 samples (group=cross)
2026-04-19 02:36:18,961 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:18,994 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:18,995 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:18,996 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:19,004 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:19,005 INFO Loaded GBPUSD/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:19,427 INFO Rule labels [4H]: {'TRENDING_UP': 1740, 'TRENDING_DOWN': 1940, 'RANGING': 4257, 'VOLATILE': 2021}  ambiguous(conf<0.4)=5492 (total=9958)  short_runs_zeroed=4111
2026-04-19 02:36:19,428 INFO Regime[4H]: collected GBPUSD — 9908 samples (group=dollar)
2026-04-19 02:36:19,583 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-19 02:36:19,612 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-19 02:36:19,613 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-19 02:36:19,613 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-19 02:36:19,621 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-19 02:36:19,622 INFO Loaded NZDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:20,033 INFO Rule labels [4H]: {'TRENDING_UP': 1770, 'TRENDING_DOWN': 1892, 'RANGING': 4281, 'VOLATILE': 2014}  ambiguous(conf<0.4)=5760 (total=9957)  short_runs_zeroed=4413
2026-04-19 02:36:20,034 INFO Regime[4H]: collected NZDUSD — 9907 samples (group=dollar)
2026-04-19 02:36:20,215 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:20,249 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:20,250 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:20,250 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:20,259 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:20,260 INFO Loaded USDCAD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:20,709 INFO Rule labels [4H]: {'TRENDING_UP': 1656, 'TRENDING_DOWN': 2068, 'RANGING': 4207, 'VOLATILE': 2026}  ambiguous(conf<0.4)=5447 (total=9957)  short_runs_zeroed=4274
2026-04-19 02:36:20,711 INFO Regime[4H]: collected USDCAD — 9907 samples (group=dollar)
2026-04-19 02:36:20,880 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:20,914 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:20,914 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:20,915 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:20,923 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:20,924 INFO Loaded USDCHF/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:21,332 INFO Rule labels [4H]: {'TRENDING_UP': 1953, 'TRENDING_DOWN': 1781, 'RANGING': 4205, 'VOLATILE': 2018}  ambiguous(conf<0.4)=5407 (total=9957)  short_runs_zeroed=4074
2026-04-19 02:36:21,333 INFO Regime[4H]: collected USDCHF — 9907 samples (group=dollar)
2026-04-19 02:36:21,509 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:21,544 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:21,545 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:21,545 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:21,553 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:21,555 INFO Loaded USDJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:21,955 INFO Rule labels [4H]: {'TRENDING_UP': 2196, 'TRENDING_DOWN': 1484, 'RANGING': 4247, 'VOLATILE': 2031}  ambiguous(conf<0.4)=5540 (total=9958)  short_runs_zeroed=4404
2026-04-19 02:36:21,956 INFO Regime[4H]: collected USDJPY — 9908 samples (group=dollar)
2026-04-19 02:36:22,223 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-19 02:36:22,282 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-19 02:36:22,283 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-19 02:36:22,283 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-19 02:36:22,295 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-19 02:36:22,296 INFO Loaded XAUUSD/4H split=train: 21467 bars (2009-03-15 → 2022-08-16)
2026-04-19 02:36:23,137 INFO Rule labels [4H]: {'TRENDING_UP': 4484, 'TRENDING_DOWN': 3949, 'RANGING': 8738, 'VOLATILE': 4296}  ambiguous(conf<0.4)=11512 (total=21467)  short_runs_zeroed=8860
2026-04-19 02:36:23,139 INFO Regime[4H]: collected XAUUSD — 21417 samples (group=gold)
2026-04-19 02:36:23,305 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_4h.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_4h.pkl_20260419_023623
2026-04-19 02:36:23,506 INFO RegimeClassifier loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_4h.pkl (device=cuda, features=31)
2026-04-19 02:36:23,525 INFO RegimeClassifier: 120490 samples, classes={'TRENDING_UP': 22783, 'TRENDING_DOWN': 22252, 'RANGING': 50966, 'VOLATILE': 24489}, device=cuda
2026-04-19 02:36:23,525 INFO RegimeClassifier: sample weights — mean=0.342  ambiguous(<0.4)=56.2%
2026-04-19 02:36:23,525 INFO RegimeClassifier: warm start from existing weights
2026-04-19 02:36:23,526 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-19 02:36:25,985 INFO Regime epoch  1/50 — tr=0.3250 va=1.0413 acc=0.471 per_class={'TRENDING_UP': 0.642, 'TRENDING_DOWN': 0.535, 'RANGING': 0.159, 'VOLATILE': 0.837}
2026-04-19 02:36:26,183 INFO Regime epoch  2/50 — tr=0.3245 va=1.0443 acc=0.468
2026-04-19 02:36:26,374 INFO Regime epoch  3/50 — tr=0.3248 va=1.0461 acc=0.468
2026-04-19 02:36:26,568 INFO Regime epoch  4/50 — tr=0.3250 va=1.0496 acc=0.465
2026-04-19 02:36:26,777 INFO Regime epoch  5/50 — tr=0.3246 va=1.0538 acc=0.463 per_class={'TRENDING_UP': 0.619, 'TRENDING_DOWN': 0.505, 'RANGING': 0.159, 'VOLATILE': 0.849}
2026-04-19 02:36:26,978 INFO Regime epoch  6/50 — tr=0.3243 va=1.0525 acc=0.463
2026-04-19 02:36:27,174 INFO Regime epoch  7/50 — tr=0.3240 va=1.0631 acc=0.455
2026-04-19 02:36:27,365 INFO Regime epoch  8/50 — tr=0.3235 va=1.0631 acc=0.454
2026-04-19 02:36:27,559 INFO Regime epoch  9/50 — tr=0.3232 va=1.0650 acc=0.451
2026-04-19 02:36:27,768 INFO Regime epoch 10/50 — tr=0.3230 va=1.0658 acc=0.450 per_class={'TRENDING_UP': 0.593, 'TRENDING_DOWN': 0.465, 'RANGING': 0.154, 'VOLATILE': 0.859}
2026-04-19 02:36:27,962 INFO Regime epoch 11/50 — tr=0.3227 va=1.0659 acc=0.449
2026-04-19 02:36:27,962 INFO Regime early stop at epoch 11 (no_improve=10)
2026-04-19 02:36:27,977 WARNING RegimeClassifier accuracy 0.47 < 0.65 threshold
2026-04-19 02:36:27,980 INFO RegimeClassifier saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_4h.pkl
2026-04-19 02:36:27,980 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_4h.pkl
2026-04-19 02:36:28,103 INFO Regime 4H complete: acc=0.471, n=120490
2026-04-19 02:36:28,105 INFO Loaded XAUUSD/4H split=train: 21467 bars (2009-03-15 → 2022-08-16)
2026-04-19 02:36:28,271 INFO Rule labels [4H]: {'TRENDING_UP': 4484, 'TRENDING_DOWN': 3949, 'RANGING': 8738, 'VOLATILE': 4296}  ambiguous(conf<0.4)=11512 (total=21467)  short_runs_zeroed=8860
2026-04-19 02:36:28,273 INFO Regime[4H] persistence (avg bars/run) on XAUUSD 4H:
{0: 9.6017130620985, 1: 9.84788029925187, 2: 10.977386934673367, 3: 9.990697674418605}
2026-04-19 02:36:28,275 INFO Regime[4H] return separation on XAUUSD 4H:
{0: 7.975048601084771e-05, 1: 1.4941597824152149e-05, 2: 2.2194974086795702e-05, 3: 4.7707963375047856e-05}
2026-04-19 02:36:28,275 INFO Regime: training 1H structure classifier...
2026-04-19 02:36:28,276 INFO Loaded AUDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:28,277 INFO Loaded EURGBP/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:28,278 INFO Loaded EURJPY/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:28,279 INFO Loaded EURUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:28,280 INFO Loaded GBPJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:28,281 INFO Loaded GBPUSD/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:28,282 INFO Loaded NZDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:28,283 INFO Loaded USDCAD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:28,284 INFO Loaded USDCHF/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:28,284 INFO Loaded USDJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:28,286 INFO Loaded XAUUSD/4H split=train: 21467 bars (2009-03-15 → 2022-08-16)
2026-04-19 02:36:28,292 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:28,294 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:28,295 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:28,296 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:28,296 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:28,302 INFO Loaded AUDUSD/1H split=train: 38804 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:28,858 INFO Rule labels [4H]: {'TRENDING_UP': 1833, 'TRENDING_DOWN': 1829, 'RANGING': 4293, 'VOLATILE': 2002}  ambiguous(conf<0.4)=5746 (total=9957)  short_runs_zeroed=4432
2026-04-19 02:36:28,863 INFO Regime[1H]: collected AUDUSD — 38754 samples (group=dollar)
2026-04-19 02:36:29,004 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:29,007 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:29,007 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:29,008 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:29,008 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:29,010 INFO Loaded EURGBP/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:29,491 INFO Rule labels [4H]: {'TRENDING_UP': 1702, 'TRENDING_DOWN': 1928, 'RANGING': 4297, 'VOLATILE': 2030}  ambiguous(conf<0.4)=5770 (total=9957)  short_runs_zeroed=4585
2026-04-19 02:36:29,495 INFO Regime[1H]: collected EURGBP — 38756 samples (group=cross)
2026-04-19 02:36:29,631 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:29,633 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:29,634 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:29,635 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:29,635 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:29,637 INFO Loaded EURJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:30,096 INFO Rule labels [4H]: {'TRENDING_UP': 2037, 'TRENDING_DOWN': 1677, 'RANGING': 4228, 'VOLATILE': 2015}  ambiguous(conf<0.4)=5748 (total=9957)  short_runs_zeroed=4550
2026-04-19 02:36:30,101 INFO Regime[1H]: collected EURJPY — 38756 samples (group=cross)
2026-04-19 02:36:30,254 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:30,257 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:30,258 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:30,258 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:30,258 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:30,260 INFO Loaded EURUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:30,752 INFO Rule labels [4H]: {'TRENDING_UP': 1509, 'TRENDING_DOWN': 2039, 'RANGING': 4405, 'VOLATILE': 2004}  ambiguous(conf<0.4)=5805 (total=9957)  short_runs_zeroed=4449
2026-04-19 02:36:30,756 INFO Regime[1H]: collected EURUSD — 38757 samples (group=dollar)
2026-04-19 02:36:30,889 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:30,891 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:30,892 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:30,892 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:30,893 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:30,895 INFO Loaded GBPJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:31,353 INFO Rule labels [4H]: {'TRENDING_UP': 1903, 'TRENDING_DOWN': 1665, 'RANGING': 4358, 'VOLATILE': 2032}  ambiguous(conf<0.4)=5547 (total=9958)  short_runs_zeroed=4143
2026-04-19 02:36:31,357 INFO Regime[1H]: collected GBPJPY — 38756 samples (group=cross)
2026-04-19 02:36:31,487 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:31,489 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:31,490 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:31,490 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:31,491 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:31,493 INFO Loaded GBPUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:31,953 INFO Rule labels [4H]: {'TRENDING_UP': 1740, 'TRENDING_DOWN': 1940, 'RANGING': 4257, 'VOLATILE': 2021}  ambiguous(conf<0.4)=5492 (total=9958)  short_runs_zeroed=4111
2026-04-19 02:36:31,957 INFO Regime[1H]: collected GBPUSD — 38757 samples (group=dollar)
2026-04-19 02:36:32,085 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-19 02:36:32,087 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-19 02:36:32,087 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-19 02:36:32,088 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-19 02:36:32,088 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-19 02:36:32,089 INFO Loaded NZDUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:32,561 INFO Rule labels [4H]: {'TRENDING_UP': 1770, 'TRENDING_DOWN': 1892, 'RANGING': 4281, 'VOLATILE': 2014}  ambiguous(conf<0.4)=5760 (total=9957)  short_runs_zeroed=4413
2026-04-19 02:36:32,566 INFO Regime[1H]: collected NZDUSD — 38757 samples (group=dollar)
2026-04-19 02:36:32,700 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:32,702 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:32,703 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:32,703 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:32,703 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:32,705 INFO Loaded USDCAD/1H split=train: 38808 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:33,161 INFO Rule labels [4H]: {'TRENDING_UP': 1656, 'TRENDING_DOWN': 2068, 'RANGING': 4207, 'VOLATILE': 2026}  ambiguous(conf<0.4)=5447 (total=9957)  short_runs_zeroed=4274
2026-04-19 02:36:33,165 INFO Regime[1H]: collected USDCAD — 38758 samples (group=dollar)
2026-04-19 02:36:33,302 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:33,305 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:33,305 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:33,306 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:33,306 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:33,308 INFO Loaded USDCHF/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:33,788 INFO Rule labels [4H]: {'TRENDING_UP': 1953, 'TRENDING_DOWN': 1781, 'RANGING': 4205, 'VOLATILE': 2018}  ambiguous(conf<0.4)=5407 (total=9957)  short_runs_zeroed=4074
2026-04-19 02:36:33,792 INFO Regime[1H]: collected USDCHF — 38756 samples (group=dollar)
2026-04-19 02:36:33,922 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:33,926 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:33,926 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:33,927 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:33,927 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:33,929 INFO Loaded USDJPY/1H split=train: 38811 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:34,401 INFO Rule labels [4H]: {'TRENDING_UP': 2196, 'TRENDING_DOWN': 1484, 'RANGING': 4247, 'VOLATILE': 2031}  ambiguous(conf<0.4)=5540 (total=9958)  short_runs_zeroed=4404
2026-04-19 02:36:34,405 INFO Regime[1H]: collected USDJPY — 38761 samples (group=dollar)
2026-04-19 02:36:34,546 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-19 02:36:34,550 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-19 02:36:34,551 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-19 02:36:34,551 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-19 02:36:34,551 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-19 02:36:34,555 INFO Loaded XAUUSD/1H split=train: 80734 bars (2009-03-15 → 2022-08-16)
2026-04-19 02:36:35,479 INFO Rule labels [4H]: {'TRENDING_UP': 4484, 'TRENDING_DOWN': 3949, 'RANGING': 8738, 'VOLATILE': 4296}  ambiguous(conf<0.4)=11512 (total=21467)  short_runs_zeroed=8860
2026-04-19 02:36:35,487 INFO Regime[1H]: collected XAUUSD — 80684 samples (group=gold)
2026-04-19 02:36:35,789 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_1h.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_1h.pkl_20260419_023635
2026-04-19 02:36:35,793 INFO RegimeClassifier loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_1h.pkl (device=cuda, features=15)
2026-04-19 02:36:35,860 INFO RegimeClassifier: 468252 samples, classes={'TRENDING_UP': 88030, 'TRENDING_DOWN': 85841, 'RANGING': 199234, 'VOLATILE': 95147}, device=cuda
2026-04-19 02:36:35,861 INFO RegimeClassifier: sample weights — mean=0.341  ambiguous(<0.4)=56.4%
2026-04-19 02:36:35,861 INFO RegimeClassifier: warm start from existing weights
2026-04-19 02:36:35,862 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-19 02:36:36,717 INFO Regime epoch  1/50 — tr=0.3352 va=1.0976 acc=0.393 per_class={'TRENDING_UP': 0.488, 'TRENDING_DOWN': 0.458, 'RANGING': 0.102, 'VOLATILE': 0.811}
2026-04-19 02:36:37,486 INFO Regime epoch  2/50 — tr=0.3350 va=1.1016 acc=0.390
2026-04-19 02:36:38,239 INFO Regime epoch  3/50 — tr=0.3346 va=1.1145 acc=0.383
2026-04-19 02:36:39,002 INFO Regime epoch  4/50 — tr=0.3344 va=1.1237 acc=0.378
2026-04-19 02:36:39,809 INFO Regime epoch  5/50 — tr=0.3340 va=1.1308 acc=0.375 per_class={'TRENDING_UP': 0.439, 'TRENDING_DOWN': 0.387, 'RANGING': 0.101, 'VOLATILE': 0.834}
2026-04-19 02:36:40,592 INFO Regime epoch  6/50 — tr=0.3338 va=1.1291 acc=0.373
2026-04-19 02:36:41,325 INFO Regime epoch  7/50 — tr=0.3336 va=1.1322 acc=0.370
2026-04-19 02:36:42,048 INFO Regime epoch  8/50 — tr=0.3334 va=1.1433 acc=0.365
2026-04-19 02:36:42,780 INFO Regime epoch  9/50 — tr=0.3331 va=1.1388 acc=0.366
2026-04-19 02:36:43,574 INFO Regime epoch 10/50 — tr=0.3329 va=1.1432 acc=0.365 per_class={'TRENDING_UP': 0.417, 'TRENDING_DOWN': 0.345, 'RANGING': 0.104, 'VOLATILE': 0.842}
2026-04-19 02:36:44,305 INFO Regime epoch 11/50 — tr=0.3328 va=1.1415 acc=0.365
2026-04-19 02:36:44,305 INFO Regime early stop at epoch 11 (no_improve=10)
2026-04-19 02:36:44,357 WARNING RegimeClassifier accuracy 0.39 < 0.65 threshold
2026-04-19 02:36:44,360 INFO RegimeClassifier saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_1h.pkl
2026-04-19 02:36:44,360 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_1h.pkl
2026-04-19 02:36:44,486 INFO Regime 1H complete: acc=0.393, n=468252
2026-04-19 02:36:44,490 INFO Loaded XAUUSD/1H split=train: 80734 bars (2009-03-15 → 2022-08-16)
2026-04-19 02:36:44,995 INFO Rule labels [1H]: {'TRENDING_UP': 13359, 'TRENDING_DOWN': 10647, 'RANGING': 39552, 'VOLATILE': 17176}  ambiguous(conf<0.4)=76861 (total=80734)  short_runs_zeroed=74080
2026-04-19 02:36:45,000 INFO Regime[1H] persistence (avg bars/run) on XAUUSD 1H:
{0: 7.690846286701209, 1: 6.99080761654629, 2: 11.59542656112577, 3: 6.886928628708901}
2026-04-19 02:36:45,003 INFO Regime[1H] return separation on XAUUSD 1H:
{0: 3.527477809596854e-05, 1: -1.6049963815312037e-05, 2: 6.410596059137787e-06, 3: 1.504434637567949e-05}
2026-04-19 02:36:45,015 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-19 02:36:45,015 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 02:36:45,016 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 02:36:45,016 INFO === VectorStore: building similarity indices ===
2026-04-19 02:36:45,016 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-19 02:36:45,016 INFO Retrain complete.
2026-04-19 02:36:46,134 INFO Model regime: SUCCESS
2026-04-19 02:36:46,134 INFO --- Training gru ---
2026-04-19 02:36:46,134 INFO Running retrain --model gru
2026-04-19 02:36:46,415 INFO retrain environment: KAGGLE
2026-04-19 02:36:48,124 INFO Device: CUDA (2 GPU(s))
2026-04-19 02:36:48,135 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 02:36:48,135 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 02:36:48,136 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-19 02:36:48,270 INFO NumExpr defaulting to 4 threads.
2026-04-19 02:36:48,459 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-19 02:36:48,459 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 02:36:48,459 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 02:36:48,745 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-19 02:36:48,968 INFO Split boundaries loaded — train≤2022-08-16  val≤2024-03-05  test≤2025-08-05
2026-04-19 02:36:48,970 INFO Loaded AUDUSD/1H split=train: 38804 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:49,048 INFO Loaded EURGBP/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:49,130 INFO Loaded EURJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:49,208 INFO Loaded EURUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:49,285 INFO Loaded GBPJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:49,359 INFO Loaded GBPUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:49,430 INFO Loaded NZDUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:49,504 INFO Loaded USDCAD/1H split=train: 38808 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:49,580 INFO Loaded USDCHF/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:49,654 INFO Loaded USDJPY/1H split=train: 38811 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:49,744 INFO Loaded XAUUSD/1H split=train: 80734 bars (2009-03-15 → 2022-08-16)
2026-04-19 02:36:49,807 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-19 02:36:49,808 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260419_023649
2026-04-19 02:36:49,811 INFO GRU feature contract unchanged (input_size=72) — incremental retrain
2026-04-19 02:36:49,932 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:49,933 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:49,948 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:49,962 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:49,964 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-19 02:36:49,964 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 02:36:49,964 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 02:36:49,965 INFO Loaded AUDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:50,044 INFO Rule labels [4H]: {'TRENDING_UP': 1833, 'TRENDING_DOWN': 1829, 'RANGING': 4293, 'VOLATILE': 2002}  ambiguous(conf<0.4)=5746 (total=9957)  short_runs_zeroed=4432
2026-04-19 02:36:50,046 INFO Loaded AUDUSD/1H split=train: 38804 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:50,077 INFO Loaded AUDUSD/5M split=train: 465551 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:50,396 INFO Loaded AUDUSD/15M split=train: 155205 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:50,552 INFO Loaded AUDUSD/1H split=train: 38804 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:50,652 INFO Loaded AUDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:50,858 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:50,859 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:50,875 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:50,882 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:50,883 INFO Loaded EURGBP/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:50,964 INFO Rule labels [4H]: {'TRENDING_UP': 1702, 'TRENDING_DOWN': 1928, 'RANGING': 4297, 'VOLATILE': 2030}  ambiguous(conf<0.4)=5770 (total=9957)  short_runs_zeroed=4585
2026-04-19 02:36:50,966 INFO Loaded EURGBP/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:50,984 INFO Loaded EURGBP/5M split=train: 465522 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:51,264 INFO Loaded EURGBP/15M split=train: 155214 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:51,392 INFO Loaded EURGBP/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:51,487 INFO Loaded EURGBP/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:51,678 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:51,679 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:51,696 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:51,704 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:51,705 INFO Loaded EURJPY/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:51,789 INFO Rule labels [4H]: {'TRENDING_UP': 2037, 'TRENDING_DOWN': 1677, 'RANGING': 4228, 'VOLATILE': 2015}  ambiguous(conf<0.4)=5748 (total=9957)  short_runs_zeroed=4550
2026-04-19 02:36:51,791 INFO Loaded EURJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:51,808 INFO Loaded EURJPY/5M split=train: 465569 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:52,089 INFO Loaded EURJPY/15M split=train: 155217 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:52,217 INFO Loaded EURJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:52,314 INFO Loaded EURJPY/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:52,503 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:52,504 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:52,520 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:52,528 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:52,529 INFO Loaded EURUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:52,613 INFO Rule labels [4H]: {'TRENDING_UP': 1509, 'TRENDING_DOWN': 2039, 'RANGING': 4405, 'VOLATILE': 2004}  ambiguous(conf<0.4)=5805 (total=9957)  short_runs_zeroed=4449
2026-04-19 02:36:52,615 INFO Loaded EURUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:52,639 INFO Loaded EURUSD/5M split=train: 465631 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:52,933 INFO Loaded EURUSD/15M split=train: 155220 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:53,065 INFO Loaded EURUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:53,168 INFO Loaded EURUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:53,353 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:53,354 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:53,371 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:53,379 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:53,379 INFO Loaded GBPJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:53,458 INFO Rule labels [4H]: {'TRENDING_UP': 1903, 'TRENDING_DOWN': 1665, 'RANGING': 4358, 'VOLATILE': 2032}  ambiguous(conf<0.4)=5547 (total=9958)  short_runs_zeroed=4143
2026-04-19 02:36:53,460 INFO Loaded GBPJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:53,478 INFO Loaded GBPJPY/5M split=train: 465412 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:53,773 INFO Loaded GBPJPY/15M split=train: 155212 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:53,904 INFO Loaded GBPJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:54,004 INFO Loaded GBPJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:54,192 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:54,193 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:54,209 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:54,216 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:54,217 INFO Loaded GBPUSD/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:54,300 INFO Rule labels [4H]: {'TRENDING_UP': 1740, 'TRENDING_DOWN': 1940, 'RANGING': 4257, 'VOLATILE': 2021}  ambiguous(conf<0.4)=5492 (total=9958)  short_runs_zeroed=4111
2026-04-19 02:36:54,302 INFO Loaded GBPUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:54,320 INFO Loaded GBPUSD/5M split=train: 465597 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:54,609 INFO Loaded GBPUSD/15M split=train: 155220 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:54,738 INFO Loaded GBPUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:54,833 INFO Loaded GBPUSD/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:54,992 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-19 02:36:54,993 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-19 02:36:55,007 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-19 02:36:55,014 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-19 02:36:55,015 INFO Loaded NZDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:55,097 INFO Rule labels [4H]: {'TRENDING_UP': 1770, 'TRENDING_DOWN': 1892, 'RANGING': 4281, 'VOLATILE': 2014}  ambiguous(conf<0.4)=5760 (total=9957)  short_runs_zeroed=4413
2026-04-19 02:36:55,099 INFO Loaded NZDUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:55,114 INFO Loaded NZDUSD/5M split=train: 465546 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:55,399 INFO Loaded NZDUSD/15M split=train: 155219 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:55,532 INFO Loaded NZDUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:55,633 INFO Loaded NZDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:55,816 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:55,817 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:55,833 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:55,841 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:55,842 INFO Loaded USDCAD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:55,931 INFO Rule labels [4H]: {'TRENDING_UP': 1656, 'TRENDING_DOWN': 2068, 'RANGING': 4207, 'VOLATILE': 2026}  ambiguous(conf<0.4)=5447 (total=9957)  short_runs_zeroed=4274
2026-04-19 02:36:55,933 INFO Loaded USDCAD/1H split=train: 38808 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:55,954 INFO Loaded USDCAD/5M split=train: 465582 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:56,234 INFO Loaded USDCAD/15M split=train: 155222 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:56,365 INFO Loaded USDCAD/1H split=train: 38808 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:56,461 INFO Loaded USDCAD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:56,650 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:56,651 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:56,667 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:56,675 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:56,676 INFO Loaded USDCHF/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:56,756 INFO Rule labels [4H]: {'TRENDING_UP': 1953, 'TRENDING_DOWN': 1781, 'RANGING': 4205, 'VOLATILE': 2018}  ambiguous(conf<0.4)=5407 (total=9957)  short_runs_zeroed=4074
2026-04-19 02:36:56,758 INFO Loaded USDCHF/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:56,776 INFO Loaded USDCHF/5M split=train: 465478 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:57,076 INFO Loaded USDCHF/15M split=train: 155208 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:57,215 INFO Loaded USDCHF/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:57,318 INFO Loaded USDCHF/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:57,507 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:57,508 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:57,525 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:57,533 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 02:36:57,534 INFO Loaded USDJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:57,613 INFO Rule labels [4H]: {'TRENDING_UP': 2196, 'TRENDING_DOWN': 1484, 'RANGING': 4247, 'VOLATILE': 2031}  ambiguous(conf<0.4)=5540 (total=9958)  short_runs_zeroed=4404
2026-04-19 02:36:57,614 INFO Loaded USDJPY/1H split=train: 38811 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:57,632 INFO Loaded USDJPY/5M split=train: 465705 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:57,920 INFO Loaded USDJPY/15M split=train: 155241 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:58,052 INFO Loaded USDJPY/1H split=train: 38811 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:58,155 INFO Loaded USDJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 02:36:58,449 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-19 02:36:58,450 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-19 02:36:58,468 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-19 02:36:58,479 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-19 02:36:58,480 INFO Loaded XAUUSD/4H split=train: 21467 bars (2009-03-15 → 2022-08-16)
2026-04-19 02:36:58,637 INFO Rule labels [4H]: {'TRENDING_UP': 4484, 'TRENDING_DOWN': 3949, 'RANGING': 8738, 'VOLATILE': 4296}  ambiguous(conf<0.4)=11512 (total=21467)  short_runs_zeroed=8860
2026-04-19 02:36:58,641 INFO Loaded XAUUSD/1H split=train: 80734 bars (2009-03-15 → 2022-08-16)
2026-04-19 02:36:58,691 INFO Loaded XAUUSD/5M split=train: 955298 bars (2009-03-15 → 2022-08-16)
2026-04-19 02:36:59,263 INFO Loaded XAUUSD/15M split=train: 319506 bars (2009-03-15 → 2022-08-16)
2026-04-19 02:36:59,477 INFO Loaded XAUUSD/1H split=train: 80734 bars (2009-03-15 → 2022-08-16)
2026-04-19 02:36:59,614 INFO Loaded XAUUSD/4H split=train: 21467 bars (2009-03-15 → 2022-08-16)
2026-04-19 02:36:59,731 INFO train_multi: 44 segments, ~7083076 total bars
2026-04-19 02:36:59,731 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-19 02:36:59,731 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-19 02:41:21,971 INFO train_multi TF=ALL: 7081756 sequences across 44 segments
2026-04-19 02:41:21,971 INFO train_multi TF=ALL: estimated peak RAM = 10368 MB (train=479965 val=120014 n_feat=72 seq_len=30)
2026-04-19 02:41:23,252 INFO train_multi TF=ALL: train=479965 val=120014 (5191 MB tensors)
2026-04-19 02:41:43,817 INFO train_multi TF=ALL epoch 1/50 train=0.5951 val=0.6180
2026-04-19 02:41:43,821 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 02:41:43,821 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 02:41:43,821 INFO train_multi TF=ALL: new best val=0.6180 — saved
2026-04-19 02:41:58,122 INFO train_multi TF=ALL epoch 2/50 train=0.5947 val=0.6185
2026-04-19 02:42:12,643 INFO train_multi TF=ALL epoch 3/50 train=0.5946 val=0.6190
2026-04-19 02:42:27,023 INFO train_multi TF=ALL epoch 4/50 train=0.5947 val=0.6193
2026-04-19 02:42:41,629 INFO train_multi TF=ALL epoch 5/50 train=0.5944 val=0.6190
2026-04-19 02:42:56,014 INFO train_multi TF=ALL epoch 6/50 train=0.5940 val=0.6194
2026-04-19 02:42:56,014 INFO train_multi TF=ALL early stop at epoch 6
2026-04-19 02:42:56,148 INFO === VectorStore: building similarity indices ===
2026-04-19 02:42:56,149 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-19 02:42:56,149 INFO Retrain complete.
2026-04-19 02:42:58,080 INFO Model gru: SUCCESS
2026-04-19 02:42:58,080 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 02:42:58,080 WARNING   [MISSING] regime_classifier → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_classifier.pkl
2026-04-19 02:42:58,081 INFO   [OK] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-19 02:42:58,081 WARNING   [MISSING] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-19 02:42:58,081 WARNING Missing weights: ['regime_classifier', 'rl_ppo'] — run retrain_incremental.py for each
2026-04-19 02:42:58,082 INFO Saved 18 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime
  START Step 6 - Backtest
2026-04-19 02:42:58,625 INFO === STEP 6: BACKTEST + REINFORCED TRAINING (3 rounds) ===
2026-04-19 02:42:58,788 INFO Backtest date range: 2021-01-01 → 2024-03-05 (reinforcement loop, test set protected; set BT_START_FLOOR env to change)
2026-04-19 02:42:58,790 INFO Cleared existing journal for fresh reinforced training run
2026-04-19 02:42:58,790 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-19 02:42:58,790 INFO Round 1 — running backtest: 2021-01-01 → 2024-03-05 (ml_trader, shared ML cache)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260419_024259.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)             2530  45.6%   2.04 1501.9% 45.6% 12.4%   2.7%    3.72

Calibration Summary:
  all          [OK] Calibration OK — p_win correlates with actual win rate.
  ml_trader    [OK] Calibration OK — p_win correlates with actual win rate.
2026-04-19 02:49:28,408 INFO Round 1 backtest — 2530 trades | avg WR=45.6% | avg PF=2.04 | avg Sharpe=3.72
2026-04-19 02:49:28,408 INFO   ml_trader: 2530 trades | WR=45.6% | PF=2.04 | Return=1501.9% | DD=2.7% | Sharpe=3.72
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 2530
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2530 rows)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  AUDUSD        313 trades   282 days   1.11/day
  EURGBP        160 trades   138 days   1.16/day
  EURJPY        115 trades    94 days   1.22/day
  EURUSD         47 trades    47 days   1.00/day
  GBPJPY         88 trades    69 days   1.27/day
  GBPUSD        717 trades   684 days   1.05/day
  NZDUSD        185 trades   162 days   1.14/day
  USDCAD        221 trades   193 days   1.15/day
  USDCHF        190 trades   166 days   1.15/day
  USDJPY        409 trades   382 days   1.07/day
  XAUUSD         85 trades    64 days   1.33/day
  ✓  All symbols within normal range.
2026-04-19 02:49:29,509 INFO Round 1: wrote 2530 journal entries (total in file: 2530)
2026-04-19 02:49:29,511 INFO Round 1 — retraining regime...
2026-04-19 02:50:08,950 INFO Retrain regime: OK
2026-04-19 02:50:08,966 INFO Round 1 — retraining quality...
2026-04-19 02:50:15,357 INFO Retrain quality: OK
2026-04-19 02:50:15,373 INFO Round 1 — retraining rl...
2026-04-19 02:51:21,956 INFO Retrain rl: OK
2026-04-19 02:51:21,973 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-19 02:51:21,973 INFO Round 2 — running backtest: 2021-01-01 → 2024-03-05 (ml_trader, shared ML cache)
2026-04-19 02:58:01,358 INFO Round 2 backtest — 2519 trades | avg WR=45.9% | avg PF=2.06 | avg Sharpe=3.76
2026-04-19 02:58:01,358 INFO   ml_trader: 2519 trades | WR=45.9% | PF=2.06 | Return=1554.8% | DD=2.7% | Sharpe=3.76
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 2519
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2519 rows)
2026-04-19 02:58:02,458 INFO Round 2: wrote 2519 journal entries (total in file: 5049)
2026-04-19 02:58:02,461 INFO Round 2 — retraining regime...
2026-04-19 02:58:42,225 INFO Retrain regime: OK
2026-04-19 02:58:42,243 INFO Round 2 — retraining quality...
2026-04-19 02:58:54,173 INFO Retrain quality: OK
2026-04-19 02:58:54,190 INFO Round 2 — retraining rl...
2026-04-19 03:00:33,399 INFO Retrain rl: OK
2026-04-19 03:00:33,417 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-04-19 03:00:33,417 INFO Round 3 — running backtest: 2021-01-01 → 2024-03-05 (ml_trader, shared ML cache)
2026-04-19 03:07:12,331 INFO Round 3 backtest — 2584 trades | avg WR=45.5% | avg PF=2.06 | avg Sharpe=3.72
2026-04-19 03:07:12,331 INFO   ml_trader: 2584 trades | WR=45.5% | PF=2.06 | Return=1531.0% | DD=2.7% | Sharpe=3.72
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 2584
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2584 rows)
2026-04-19 03:07:13,431 INFO Round 3: wrote 2584 journal entries (total in file: 7633)
2026-04-19 03:07:13,433 INFO Round 3 (final): retraining after last backtest...
2026-04-19 03:07:13,433 INFO Round 3 — retraining regime...
2026-04-19 03:07:53,084 INFO Retrain regime: OK
2026-04-19 03:07:53,102 INFO Round 3 — retraining quality...
2026-04-19 03:08:01,301 INFO Retrain quality: OK
2026-04-19 03:08:01,317 INFO Round 3 — retraining rl...
2026-04-19 03:10:17,306 INFO Retrain rl: OK
2026-04-19 03:10:17,324 INFO Improvement round 1 → 3: WR -0.1% | PF +0.019 | Sharpe -0.004
2026-04-19 03:10:17,472 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-19 03:10:17,481 INFO Journal entries: 7633
2026-04-19 03:10:17,481 INFO --- Training quality ---
2026-04-19 03:10:17,481 INFO Running retrain --model quality
  DONE  Step 6 - Backtest
  START Step 7b - Quality+RL
2026-04-19 03:10:17,825 INFO retrain environment: KAGGLE
2026-04-19 03:10:19,531 INFO Device: CUDA (2 GPU(s))
2026-04-19 03:10:19,543 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 03:10:19,543 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 03:10:19,544 INFO === QualityScorer retrain ===
2026-04-19 03:10:19,676 INFO NumExpr defaulting to 4 threads.
2026-04-19 03:10:19,866 INFO QualityScorer: CUDA available — using GPU
2026-04-19 03:10:20,071 INFO QualityScorer loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (device=cuda)
2026-04-19 03:10:20,371 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/quality_scorer.pkl_20260419_031020
2026-04-19 03:10:20,651 INFO QualityScorer: 7620 samples, EV stats={'mean': 0.08802572637796402, 'std': 1.2874306440353394, 'n_pos': 3474, 'n_neg': 4146}, device=cuda
2026-04-19 03:10:20,652 INFO QualityScorer: normalised win labels by median_win=1.108 — EV range now [-1, +3]
2026-04-19 03:10:20,652 INFO QualityScorer: warm start from existing weights
2026-04-19 03:10:20,653 INFO QualityScorer: pos_weight=1.16 (n_pos=2816 n_neg=3280)
2026-04-19 03:10:22,331 INFO Quality epoch   1/100 — va_huber=0.6169
2026-04-19 03:10:22,476 INFO Quality epoch   2/100 — va_huber=0.6163
2026-04-19 03:10:22,610 INFO Quality epoch   3/100 — va_huber=0.6169
2026-04-19 03:10:22,742 INFO Quality epoch   4/100 — va_huber=0.6170
2026-04-19 03:10:22,881 INFO Quality epoch   5/100 — va_huber=0.6170
2026-04-19 03:10:23,688 INFO Quality epoch  11/100 — va_huber=0.6166
2026-04-19 03:10:25,062 INFO Quality epoch  21/100 — va_huber=0.6160
2026-04-19 03:10:26,428 INFO Quality epoch  31/100 — va_huber=0.6151
2026-04-19 03:10:27,779 INFO Quality epoch  41/100 — va_huber=0.6152
2026-04-19 03:10:28,191 INFO Quality early stop at epoch 44
2026-04-19 03:10:28,214 INFO QualityScorer EV model: MAE=1.023 dir_acc=0.616 n_val=1524
2026-04-19 03:10:28,218 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-19 03:10:28,294 INFO Retrain complete.
2026-04-19 03:10:29,177 INFO Model quality: SUCCESS
2026-04-19 03:10:29,177 INFO --- Training rl ---
2026-04-19 03:10:29,178 INFO Running retrain --model rl
2026-04-19 03:10:29,368 INFO retrain environment: KAGGLE
2026-04-19 03:10:31,113 INFO Device: CUDA (2 GPU(s))
2026-04-19 03:10:31,124 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 03:10:31,125 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 03:10:31,125 INFO === RLAgent (PPO) retrain ===
2026-04-19 03:10:31,128 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260419_031031
2026-04-19 03:10:31.948783: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1776568231.971032   26745 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1776568231.978988   26745 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1776568231.998468   26745 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1776568231.998512   26745 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1776568231.998515   26745 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1776568231.998517   26745 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-19 03:10:36,399 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-19 03:10:39,191 INFO RLAgent: PPO model loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-19 03:10:39,381 INFO RLAgent: warm start — fine-tuning existing PPO policy (lr=6.00e-05)
2026-04-19 03:12:50,611 INFO RLAgent: retrain complete, 7633 episodes
2026-04-19 03:12:50,611 INFO Retrain complete.
  DONE  Step 7b - Quality+RL

=== Pipeline complete ===
2026-04-19 03:12:52,160 INFO Model rl: SUCCESS
2026-04-19 03:12:52,161 INFO Step 7b complete — summary: /kaggle/working/Multi-Bot/trading-system/ml_training/metrics/training_7b_summary.json