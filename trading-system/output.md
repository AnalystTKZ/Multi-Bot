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
2026-04-19 04:38:50,659 INFO Loading feature-engineered data...
2026-04-19 04:38:51,154 INFO Loaded 221743 rows, 202 features
2026-04-19 04:38:51,155 INFO Train: 155220 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:38:51,155 INFO Val:   33261 bars (2022-08-16 → 2024-03-05)
2026-04-19 04:38:51,155 INFO Test:  33262 bars (2024-03-05 → 2025-08-05)
2026-04-19 04:38:51,155 INFO No leakage confirmed: train < val < test timestamps

=== SPLIT COMPLETE (no shuffling, time-based) ===
  Train:      155,220 bars  2016-01-04 → 2022-08-16
  Validation:  33,261 bars  2022-08-16 → 2024-03-05
  Test:        33,262 bars  2024-03-05 → 2025-08-05
  Features: 202
  Leakage check: PASS
  DONE  Step 5 - Split
  START Step 7a - GRU+Regime
2026-04-19 04:38:53,647 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-19 04:38:53,647 INFO --- Training regime ---
2026-04-19 04:38:53,647 INFO Running retrain --model regime
2026-04-19 04:38:53,828 INFO retrain environment: KAGGLE
2026-04-19 04:38:55,527 INFO Device: CUDA (2 GPU(s))
2026-04-19 04:38:55,543 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 04:38:55,544 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 04:38:55,545 INFO === RegimeClassifier retrain (dual-TF cascade: 4H bias + 1H structure) ===
2026-04-19 04:38:55,708 INFO NumExpr defaulting to 4 threads.
2026-04-19 04:38:55,937 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-19 04:38:55,938 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 04:38:55,938 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 04:38:56,165 INFO Split boundaries loaded — train≤2022-08-16  val≤2024-03-05  test≤2025-08-05
2026-04-19 04:38:56,168 INFO Loaded AUDUSD/1H split=train: 38804 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:38:56,253 INFO Loaded EURGBP/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:38:56,328 INFO Loaded EURJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:38:56,404 INFO Loaded EURUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:38:56,480 INFO Loaded GBPJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:38:56,558 INFO Loaded GBPUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:38:56,631 INFO Loaded NZDUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:38:56,704 INFO Loaded USDCAD/1H split=train: 38808 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:38:56,778 INFO Loaded USDCHF/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:38:56,854 INFO Loaded USDJPY/1H split=train: 38811 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:38:56,947 INFO Loaded XAUUSD/1H split=train: 80734 bars (2009-03-15 → 2022-08-16)
2026-04-19 04:38:57,010 INFO Regime: fitting per-group GMMs on 4H data (dollar / cross / gold)...
2026-04-19 04:38:57,025 INFO Loaded AUDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:38:57,040 INFO Loaded EURGBP/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:38:57,055 INFO Loaded EURJPY/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:38:57,069 INFO Loaded EURUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:38:57,084 INFO Loaded GBPJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:38:57,098 INFO Loaded GBPUSD/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:38:57,111 INFO Loaded NZDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:38:57,125 INFO Loaded USDCAD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:38:57,140 INFO Loaded USDCHF/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:38:57,156 INFO Loaded USDJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:38:57,174 INFO Loaded XAUUSD/4H split=train: 21467 bars (2009-03-15 → 2022-08-16)
2026-04-19 04:38:57,989 INFO GMM fit: timeframe=4H → n_bar=50
2026-04-19 04:39:24,384 INFO GMM fitted on 69344 samples — cluster→regime: {2: 3, 4: 0, 1: 1, 3: 4, 0: 2} dist: {'TRENDING_UP': 1, 'TRENDING_DOWN': 1, 'RANGING': 1, 'VOLATILE': 1, 'CONSOLIDATION': 1}
2026-04-19 04:39:24,385 INFO Regime: GMM 'dollar' fitted on 7 4H dfs (n_bar=50)
2026-04-19 04:39:24,385 INFO GMM fit: timeframe=4H → n_bar=50
2026-04-19 04:39:36,088 INFO GMM fitted on 29719 samples — cluster→regime: {0: 3, 4: 0, 2: 1, 1: 4, 3: 2} dist: {'TRENDING_UP': 1, 'TRENDING_DOWN': 1, 'RANGING': 1, 'VOLATILE': 1, 'CONSOLIDATION': 1}
2026-04-19 04:39:36,092 INFO Regime: GMM 'cross' fitted on 3 4H dfs (n_bar=50)
2026-04-19 04:39:36,092 INFO GMM fit: timeframe=4H → n_bar=50
2026-04-19 04:39:42,927 INFO GMM fitted on 10708 samples — cluster→regime: {1: 3, 2: 0, 4: 1, 0: 4, 3: 2} dist: {'TRENDING_UP': 1, 'TRENDING_DOWN': 1, 'RANGING': 1, 'VOLATILE': 1, 'CONSOLIDATION': 1}
2026-04-19 04:39:42,928 INFO Regime: GMM 'gold' fitted on 1 4H dfs (n_bar=50)
2026-04-19 04:39:43,028 INFO Regime: training 4H bias classifier...
2026-04-19 04:39:43,030 INFO Loaded AUDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:39:43,031 INFO Loaded EURGBP/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:39:43,032 INFO Loaded EURJPY/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:39:43,033 INFO Loaded EURUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:39:43,035 INFO Loaded GBPJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:39:43,036 INFO Loaded GBPUSD/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:39:43,037 INFO Loaded NZDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:39:43,038 INFO Loaded USDCAD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:39:43,039 INFO Loaded USDCHF/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:39:43,040 INFO Loaded USDJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:39:43,041 INFO Loaded XAUUSD/4H split=train: 21467 bars (2009-03-15 → 2022-08-16)
2026-04-19 04:39:43,170 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:43,211 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:43,212 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:43,212 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:43,222 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:43,223 INFO Loaded AUDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:39:43,688 INFO Rule labels [4H]: {'TRENDING_UP': 1833, 'TRENDING_DOWN': 1829, 'RANGING': 2945, 'VOLATILE': 2002, 'CONSOLIDATION': 1348}  ambiguous(conf<0.4)=6364 (total=9957)  short_runs_zeroed=5389
2026-04-19 04:39:43,690 INFO Regime[4H]: collected AUDUSD — 9907 samples (group=dollar)
2026-04-19 04:39:43,861 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:43,897 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:43,898 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:43,898 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:43,907 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:43,908 INFO Loaded EURGBP/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:39:44,344 INFO Rule labels [4H]: {'TRENDING_UP': 1702, 'TRENDING_DOWN': 1928, 'RANGING': 3011, 'VOLATILE': 2030, 'CONSOLIDATION': 1286}  ambiguous(conf<0.4)=6241 (total=9957)  short_runs_zeroed=5356
2026-04-19 04:39:44,345 INFO Regime[4H]: collected EURGBP — 9907 samples (group=cross)
2026-04-19 04:39:44,519 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:44,554 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:44,555 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:44,555 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:44,564 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:44,565 INFO Loaded EURJPY/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:39:44,983 INFO Rule labels [4H]: {'TRENDING_UP': 2037, 'TRENDING_DOWN': 1677, 'RANGING': 2817, 'VOLATILE': 2015, 'CONSOLIDATION': 1411}  ambiguous(conf<0.4)=6152 (total=9957)  short_runs_zeroed=5275
2026-04-19 04:39:44,984 INFO Regime[4H]: collected EURJPY — 9907 samples (group=cross)
2026-04-19 04:39:45,147 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:45,187 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:45,188 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:45,188 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:45,197 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:45,198 INFO Loaded EURUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:39:45,625 INFO Rule labels [4H]: {'TRENDING_UP': 1509, 'TRENDING_DOWN': 2039, 'RANGING': 2977, 'VOLATILE': 2004, 'CONSOLIDATION': 1428}  ambiguous(conf<0.4)=6567 (total=9957)  short_runs_zeroed=5840
2026-04-19 04:39:45,626 INFO Regime[4H]: collected EURUSD — 9907 samples (group=dollar)
2026-04-19 04:39:45,797 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:45,835 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:45,835 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:45,836 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:45,845 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:45,846 INFO Loaded GBPJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:39:46,270 INFO Rule labels [4H]: {'TRENDING_UP': 1903, 'TRENDING_DOWN': 1665, 'RANGING': 3018, 'VOLATILE': 2032, 'CONSOLIDATION': 1340}  ambiguous(conf<0.4)=6061 (total=9958)  short_runs_zeroed=4947
2026-04-19 04:39:46,271 INFO Regime[4H]: collected GBPJPY — 9908 samples (group=cross)
2026-04-19 04:39:46,442 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:46,476 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:46,476 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:46,477 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:46,486 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:46,487 INFO Loaded GBPUSD/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:39:46,903 INFO Rule labels [4H]: {'TRENDING_UP': 1740, 'TRENDING_DOWN': 1940, 'RANGING': 2949, 'VOLATILE': 2021, 'CONSOLIDATION': 1308}  ambiguous(conf<0.4)=6144 (total=9958)  short_runs_zeroed=5186
2026-04-19 04:39:46,905 INFO Regime[4H]: collected GBPUSD — 9908 samples (group=dollar)
2026-04-19 04:39:47,055 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-19 04:39:47,084 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-19 04:39:47,084 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-19 04:39:47,085 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-19 04:39:47,092 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-19 04:39:47,093 INFO Loaded NZDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:39:47,513 INFO Rule labels [4H]: {'TRENDING_UP': 1770, 'TRENDING_DOWN': 1892, 'RANGING': 2939, 'VOLATILE': 2014, 'CONSOLIDATION': 1342}  ambiguous(conf<0.4)=6173 (total=9957)  short_runs_zeroed=5185
2026-04-19 04:39:47,515 INFO Regime[4H]: collected NZDUSD — 9907 samples (group=dollar)
2026-04-19 04:39:47,679 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:47,714 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:47,714 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:47,715 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:47,724 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:47,725 INFO Loaded USDCAD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:39:48,144 INFO Rule labels [4H]: {'TRENDING_UP': 1656, 'TRENDING_DOWN': 2068, 'RANGING': 2799, 'VOLATILE': 2026, 'CONSOLIDATION': 1408}  ambiguous(conf<0.4)=5998 (total=9957)  short_runs_zeroed=5184
2026-04-19 04:39:48,145 INFO Regime[4H]: collected USDCAD — 9907 samples (group=dollar)
2026-04-19 04:39:48,314 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:48,347 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:48,348 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:48,348 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:48,357 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:48,358 INFO Loaded USDCHF/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:39:48,787 INFO Rule labels [4H]: {'TRENDING_UP': 1953, 'TRENDING_DOWN': 1781, 'RANGING': 2878, 'VOLATILE': 2018, 'CONSOLIDATION': 1327}  ambiguous(conf<0.4)=6098 (total=9957)  short_runs_zeroed=5260
2026-04-19 04:39:48,789 INFO Regime[4H]: collected USDCHF — 9907 samples (group=dollar)
2026-04-19 04:39:48,963 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:48,997 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:48,998 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:48,998 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:49,007 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:39:49,008 INFO Loaded USDJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:39:49,431 INFO Rule labels [4H]: {'TRENDING_UP': 2196, 'TRENDING_DOWN': 1484, 'RANGING': 2821, 'VOLATILE': 2031, 'CONSOLIDATION': 1426}  ambiguous(conf<0.4)=6104 (total=9958)  short_runs_zeroed=5301
2026-04-19 04:39:49,432 INFO Regime[4H]: collected USDJPY — 9908 samples (group=dollar)
2026-04-19 04:39:49,697 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-19 04:39:49,758 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-19 04:39:49,759 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-19 04:39:49,759 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-19 04:39:49,772 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-19 04:39:49,773 INFO Loaded XAUUSD/4H split=train: 21467 bars (2009-03-15 → 2022-08-16)
2026-04-19 04:39:50,627 INFO Rule labels [4H]: {'TRENDING_UP': 4484, 'TRENDING_DOWN': 3949, 'RANGING': 5847, 'VOLATILE': 4296, 'CONSOLIDATION': 2891}  ambiguous(conf<0.4)=12923 (total=21467)  short_runs_zeroed=11105
2026-04-19 04:39:50,629 INFO Regime[4H]: collected XAUUSD — 21417 samples (group=gold)
2026-04-19 04:39:50,791 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_4h.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_4h.pkl_20260419_043950
2026-04-19 04:39:50,987 INFO RegimeClassifier loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_4h.pkl (device=cuda, features=31)
2026-04-19 04:39:51,005 INFO RegimeClassifier: 120490 samples, classes={'TRENDING_UP': 22783, 'TRENDING_DOWN': 22252, 'RANGING': 34451, 'VOLATILE': 24489, 'CONSOLIDATION': 16515}, device=cuda
2026-04-19 04:39:51,006 INFO RegimeClassifier: sample weights — mean=0.314  ambiguous(<0.4)=62.0%
2026-04-19 04:39:51,006 WARNING RegimeClassifier: class count changed 4→5, resetting
2026-04-19 04:39:51,008 INFO RegimeClassifier: cold start (no existing weights)
2026-04-19 04:39:51,009 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-19 04:39:53,459 INFO Regime epoch  1/50 — tr=0.5298 va=1.6061 acc=0.297 per_class={'TRENDING_UP': 0.413, 'TRENDING_DOWN': 0.07, 'RANGING': 0.661, 'VOLATILE': 0.061, 'CONSOLIDATION': 0.017}
2026-04-19 04:39:53,654 INFO Regime epoch  2/50 — tr=0.5191 va=1.5744 acc=0.331
2026-04-19 04:39:53,852 INFO Regime epoch  3/50 — tr=0.5011 va=1.5369 acc=0.384
2026-04-19 04:39:54,059 INFO Regime epoch  4/50 — tr=0.4732 va=1.5032 acc=0.383
2026-04-19 04:39:54,277 INFO Regime epoch  5/50 — tr=0.4412 va=1.4768 acc=0.376 per_class={'TRENDING_UP': 0.382, 'TRENDING_DOWN': 0.272, 'RANGING': 0.276, 'VOLATILE': 0.818, 'CONSOLIDATION': 0.005}
2026-04-19 04:39:54,473 INFO Regime epoch  6/50 — tr=0.4146 va=1.4444 acc=0.383
2026-04-19 04:39:54,679 INFO Regime epoch  7/50 — tr=0.3936 va=1.4029 acc=0.402
2026-04-19 04:39:54,870 INFO Regime epoch  8/50 — tr=0.3762 va=1.3625 acc=0.414
2026-04-19 04:39:55,062 INFO Regime epoch  9/50 — tr=0.3652 va=1.3334 acc=0.418
2026-04-19 04:39:55,276 INFO Regime epoch 10/50 — tr=0.3563 va=1.3203 acc=0.416 per_class={'TRENDING_UP': 0.581, 'TRENDING_DOWN': 0.437, 'RANGING': 0.007, 'VOLATILE': 0.806, 'CONSOLIDATION': 0.341}
2026-04-19 04:39:55,483 INFO Regime epoch 11/50 — tr=0.3488 va=1.3041 acc=0.419
2026-04-19 04:39:55,681 INFO Regime epoch 12/50 — tr=0.3429 va=1.2854 acc=0.424
2026-04-19 04:39:55,873 INFO Regime epoch 13/50 — tr=0.3395 va=1.2825 acc=0.422
2026-04-19 04:39:56,068 INFO Regime epoch 14/50 — tr=0.3364 va=1.2788 acc=0.422
2026-04-19 04:39:56,280 INFO Regime epoch 15/50 — tr=0.3337 va=1.2767 acc=0.421 per_class={'TRENDING_UP': 0.586, 'TRENDING_DOWN': 0.421, 'RANGING': 0.002, 'VOLATILE': 0.826, 'CONSOLIDATION': 0.372}
2026-04-19 04:39:56,478 INFO Regime epoch 16/50 — tr=0.3309 va=1.2802 acc=0.418
2026-04-19 04:39:56,673 INFO Regime epoch 17/50 — tr=0.3289 va=1.2750 acc=0.421
2026-04-19 04:39:56,883 INFO Regime epoch 18/50 — tr=0.3272 va=1.2747 acc=0.421
2026-04-19 04:39:57,082 INFO Regime epoch 19/50 — tr=0.3246 va=1.2758 acc=0.421
2026-04-19 04:39:57,299 INFO Regime epoch 20/50 — tr=0.3234 va=1.2704 acc=0.425 per_class={'TRENDING_UP': 0.586, 'TRENDING_DOWN': 0.422, 'RANGING': 0.002, 'VOLATILE': 0.841, 'CONSOLIDATION': 0.381}
2026-04-19 04:39:57,507 INFO Regime epoch 21/50 — tr=0.3223 va=1.2746 acc=0.423
2026-04-19 04:39:57,699 INFO Regime epoch 22/50 — tr=0.3207 va=1.2780 acc=0.422
2026-04-19 04:39:57,891 INFO Regime epoch 23/50 — tr=0.3205 va=1.2715 acc=0.427
2026-04-19 04:39:58,086 INFO Regime epoch 24/50 — tr=0.3188 va=1.2747 acc=0.424
2026-04-19 04:39:58,301 INFO Regime epoch 25/50 — tr=0.3177 va=1.2695 acc=0.427 per_class={'TRENDING_UP': 0.587, 'TRENDING_DOWN': 0.423, 'RANGING': 0.003, 'VOLATILE': 0.85, 'CONSOLIDATION': 0.378}
2026-04-19 04:39:58,511 INFO Regime epoch 26/50 — tr=0.3173 va=1.2730 acc=0.426
2026-04-19 04:39:58,715 INFO Regime epoch 27/50 — tr=0.3163 va=1.2740 acc=0.426
2026-04-19 04:39:58,923 INFO Regime epoch 28/50 — tr=0.3161 va=1.2770 acc=0.424
2026-04-19 04:39:59,119 INFO Regime epoch 29/50 — tr=0.3150 va=1.2800 acc=0.423
2026-04-19 04:39:59,346 INFO Regime epoch 30/50 — tr=0.3155 va=1.2750 acc=0.427 per_class={'TRENDING_UP': 0.582, 'TRENDING_DOWN': 0.422, 'RANGING': 0.003, 'VOLATILE': 0.854, 'CONSOLIDATION': 0.375}
2026-04-19 04:39:59,543 INFO Regime epoch 31/50 — tr=0.3140 va=1.2806 acc=0.422
2026-04-19 04:39:59,733 INFO Regime epoch 32/50 — tr=0.3143 va=1.2788 acc=0.424
2026-04-19 04:39:59,943 INFO Regime epoch 33/50 — tr=0.3135 va=1.2781 acc=0.425
2026-04-19 04:40:00,148 INFO Regime epoch 34/50 — tr=0.3131 va=1.2814 acc=0.423
2026-04-19 04:40:00,356 INFO Regime epoch 35/50 — tr=0.3128 va=1.2788 acc=0.424 per_class={'TRENDING_UP': 0.576, 'TRENDING_DOWN': 0.41, 'RANGING': 0.003, 'VOLATILE': 0.858, 'CONSOLIDATION': 0.376}
2026-04-19 04:40:00,356 INFO Regime early stop at epoch 35 (no_improve=10)
2026-04-19 04:40:00,370 WARNING RegimeClassifier accuracy 0.43 < 0.65 threshold
2026-04-19 04:40:00,373 INFO RegimeClassifier saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_4h.pkl
2026-04-19 04:40:00,373 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_4h.pkl
2026-04-19 04:40:00,529 INFO Regime 4H complete: acc=0.427, n=120490
2026-04-19 04:40:00,531 INFO Loaded XAUUSD/4H split=train: 21467 bars (2009-03-15 → 2022-08-16)
2026-04-19 04:40:00,699 INFO Rule labels [4H]: {'TRENDING_UP': 4484, 'TRENDING_DOWN': 3949, 'RANGING': 5847, 'VOLATILE': 4296, 'CONSOLIDATION': 2891}  ambiguous(conf<0.4)=12923 (total=21467)  short_runs_zeroed=11105
2026-04-19 04:40:00,702 INFO Regime[4H] persistence (avg bars/run) on XAUUSD 4H:
{0: 9.6017130620985, 1: 9.84788029925187, 2: 6.63677639046538, 3: 9.990697674418605, 4: 5.793587174348698}
2026-04-19 04:40:00,704 INFO Regime[4H] return separation on XAUUSD 4H:
{0: 7.975048601084771e-05, 1: 1.4941597824152149e-05, 2: 6.315567302833706e-05, 3: 4.7707963375047856e-05, 4: -6.067603169562378e-05}
2026-04-19 04:40:00,704 INFO Regime: training 1H structure classifier...
2026-04-19 04:40:00,705 INFO Loaded AUDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:00,706 INFO Loaded EURGBP/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:00,707 INFO Loaded EURJPY/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:00,708 INFO Loaded EURUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:00,709 INFO Loaded GBPJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:00,710 INFO Loaded GBPUSD/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:00,710 INFO Loaded NZDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:00,711 INFO Loaded USDCAD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:00,712 INFO Loaded USDCHF/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:00,713 INFO Loaded USDJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:00,714 INFO Loaded XAUUSD/4H split=train: 21467 bars (2009-03-15 → 2022-08-16)
2026-04-19 04:40:00,720 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:00,722 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:00,723 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:00,724 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:00,724 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:00,727 INFO Loaded AUDUSD/1H split=train: 38804 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:01,234 INFO Rule labels [4H]: {'TRENDING_UP': 1833, 'TRENDING_DOWN': 1829, 'RANGING': 2945, 'VOLATILE': 2002, 'CONSOLIDATION': 1348}  ambiguous(conf<0.4)=6364 (total=9957)  short_runs_zeroed=5389
2026-04-19 04:40:01,239 INFO Regime[1H]: collected AUDUSD — 38754 samples (group=dollar)
2026-04-19 04:40:01,370 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:01,373 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:01,373 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:01,374 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:01,374 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:01,376 INFO Loaded EURGBP/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:01,837 INFO Rule labels [4H]: {'TRENDING_UP': 1702, 'TRENDING_DOWN': 1928, 'RANGING': 3011, 'VOLATILE': 2030, 'CONSOLIDATION': 1286}  ambiguous(conf<0.4)=6241 (total=9957)  short_runs_zeroed=5356
2026-04-19 04:40:01,841 INFO Regime[1H]: collected EURGBP — 38756 samples (group=cross)
2026-04-19 04:40:01,974 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:01,976 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:01,977 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:01,978 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:01,978 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:01,980 INFO Loaded EURJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:02,442 INFO Rule labels [4H]: {'TRENDING_UP': 2037, 'TRENDING_DOWN': 1677, 'RANGING': 2817, 'VOLATILE': 2015, 'CONSOLIDATION': 1411}  ambiguous(conf<0.4)=6152 (total=9957)  short_runs_zeroed=5275
2026-04-19 04:40:02,447 INFO Regime[1H]: collected EURJPY — 38756 samples (group=cross)
2026-04-19 04:40:02,576 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:02,579 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:02,579 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:02,580 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:02,580 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:02,582 INFO Loaded EURUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:03,048 INFO Rule labels [4H]: {'TRENDING_UP': 1509, 'TRENDING_DOWN': 2039, 'RANGING': 2977, 'VOLATILE': 2004, 'CONSOLIDATION': 1428}  ambiguous(conf<0.4)=6567 (total=9957)  short_runs_zeroed=5840
2026-04-19 04:40:03,053 INFO Regime[1H]: collected EURUSD — 38757 samples (group=dollar)
2026-04-19 04:40:03,184 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:03,186 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:03,187 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:03,187 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:03,188 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:03,190 INFO Loaded GBPJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:03,665 INFO Rule labels [4H]: {'TRENDING_UP': 1903, 'TRENDING_DOWN': 1665, 'RANGING': 3018, 'VOLATILE': 2032, 'CONSOLIDATION': 1340}  ambiguous(conf<0.4)=6061 (total=9958)  short_runs_zeroed=4947
2026-04-19 04:40:03,669 INFO Regime[1H]: collected GBPJPY — 38756 samples (group=cross)
2026-04-19 04:40:03,795 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:03,799 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:03,800 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:03,800 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:03,801 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:03,803 INFO Loaded GBPUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:04,271 INFO Rule labels [4H]: {'TRENDING_UP': 1740, 'TRENDING_DOWN': 1940, 'RANGING': 2949, 'VOLATILE': 2021, 'CONSOLIDATION': 1308}  ambiguous(conf<0.4)=6144 (total=9958)  short_runs_zeroed=5186
2026-04-19 04:40:04,275 INFO Regime[1H]: collected GBPUSD — 38757 samples (group=dollar)
2026-04-19 04:40:04,405 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-19 04:40:04,407 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-19 04:40:04,407 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-19 04:40:04,408 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-19 04:40:04,408 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-19 04:40:04,410 INFO Loaded NZDUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:04,873 INFO Rule labels [4H]: {'TRENDING_UP': 1770, 'TRENDING_DOWN': 1892, 'RANGING': 2939, 'VOLATILE': 2014, 'CONSOLIDATION': 1342}  ambiguous(conf<0.4)=6173 (total=9957)  short_runs_zeroed=5185
2026-04-19 04:40:04,877 INFO Regime[1H]: collected NZDUSD — 38757 samples (group=dollar)
2026-04-19 04:40:05,010 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:05,012 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:05,013 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:05,013 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:05,014 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:05,016 INFO Loaded USDCAD/1H split=train: 38808 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:05,497 INFO Rule labels [4H]: {'TRENDING_UP': 1656, 'TRENDING_DOWN': 2068, 'RANGING': 2799, 'VOLATILE': 2026, 'CONSOLIDATION': 1408}  ambiguous(conf<0.4)=5998 (total=9957)  short_runs_zeroed=5184
2026-04-19 04:40:05,502 INFO Regime[1H]: collected USDCAD — 38758 samples (group=dollar)
2026-04-19 04:40:05,634 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:05,637 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:05,637 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:05,638 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:05,638 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:05,640 INFO Loaded USDCHF/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:06,128 INFO Rule labels [4H]: {'TRENDING_UP': 1953, 'TRENDING_DOWN': 1781, 'RANGING': 2878, 'VOLATILE': 2018, 'CONSOLIDATION': 1327}  ambiguous(conf<0.4)=6098 (total=9957)  short_runs_zeroed=5260
2026-04-19 04:40:06,132 INFO Regime[1H]: collected USDCHF — 38756 samples (group=dollar)
2026-04-19 04:40:06,261 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:06,264 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:06,265 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:06,266 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:06,266 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:06,268 INFO Loaded USDJPY/1H split=train: 38811 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:06,741 INFO Rule labels [4H]: {'TRENDING_UP': 2196, 'TRENDING_DOWN': 1484, 'RANGING': 2821, 'VOLATILE': 2031, 'CONSOLIDATION': 1426}  ambiguous(conf<0.4)=6104 (total=9958)  short_runs_zeroed=5301
2026-04-19 04:40:06,746 INFO Regime[1H]: collected USDJPY — 38761 samples (group=dollar)
2026-04-19 04:40:06,885 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-19 04:40:06,888 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-19 04:40:06,890 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-19 04:40:06,890 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-19 04:40:06,890 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-19 04:40:06,894 INFO Loaded XAUUSD/1H split=train: 80734 bars (2009-03-15 → 2022-08-16)
2026-04-19 04:40:07,826 INFO Rule labels [4H]: {'TRENDING_UP': 4484, 'TRENDING_DOWN': 3949, 'RANGING': 5847, 'VOLATILE': 4296, 'CONSOLIDATION': 2891}  ambiguous(conf<0.4)=12923 (total=21467)  short_runs_zeroed=11105
2026-04-19 04:40:07,834 INFO Regime[1H]: collected XAUUSD — 80684 samples (group=gold)
2026-04-19 04:40:08,126 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_1h.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_1h.pkl_20260419_044008
2026-04-19 04:40:08,130 INFO RegimeClassifier loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_1h.pkl (device=cuda, features=15)
2026-04-19 04:40:08,196 INFO RegimeClassifier: 468252 samples, classes={'TRENDING_UP': 88030, 'TRENDING_DOWN': 85841, 'RANGING': 135377, 'VOLATILE': 95147, 'CONSOLIDATION': 63857}, device=cuda
2026-04-19 04:40:08,197 INFO RegimeClassifier: sample weights — mean=0.314  ambiguous(<0.4)=62.1%
2026-04-19 04:40:08,197 WARNING RegimeClassifier: class count changed 4→5, resetting
2026-04-19 04:40:08,199 INFO RegimeClassifier: cold start (no existing weights)
2026-04-19 04:40:08,199 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-19 04:40:09,045 INFO Regime epoch  1/50 — tr=0.4808 va=1.5991 acc=0.269 per_class={'TRENDING_UP': 0.696, 'TRENDING_DOWN': 0.157, 'RANGING': 0.088, 'VOLATILE': 0.124, 'CONSOLIDATION': 0.334}
2026-04-19 04:40:09,803 INFO Regime epoch  2/50 — tr=0.4502 va=1.4966 acc=0.356
2026-04-19 04:40:10,592 INFO Regime epoch  3/50 — tr=0.4148 va=1.4489 acc=0.362
2026-04-19 04:40:11,353 INFO Regime epoch  4/50 — tr=0.3869 va=1.4095 acc=0.372
2026-04-19 04:40:12,191 INFO Regime epoch  5/50 — tr=0.3706 va=1.3733 acc=0.373 per_class={'TRENDING_UP': 0.528, 'TRENDING_DOWN': 0.414, 'RANGING': 0.0, 'VOLATILE': 0.763, 'CONSOLIDATION': 0.232}
2026-04-19 04:40:12,954 INFO Regime epoch  6/50 — tr=0.3600 va=1.3419 acc=0.372
2026-04-19 04:40:13,720 INFO Regime epoch  7/50 — tr=0.3516 va=1.3339 acc=0.366
2026-04-19 04:40:14,465 INFO Regime epoch  8/50 — tr=0.3461 va=1.3256 acc=0.361
2026-04-19 04:40:15,206 INFO Regime epoch  9/50 — tr=0.3418 va=1.3147 acc=0.363
2026-04-19 04:40:16,014 INFO Regime epoch 10/50 — tr=0.3389 va=1.3208 acc=0.359 per_class={'TRENDING_UP': 0.459, 'TRENDING_DOWN': 0.375, 'RANGING': 0.0, 'VOLATILE': 0.797, 'CONSOLIDATION': 0.24}
2026-04-19 04:40:16,776 INFO Regime epoch 11/50 — tr=0.3359 va=1.3192 acc=0.357
2026-04-19 04:40:17,543 INFO Regime epoch 12/50 — tr=0.3342 va=1.3137 acc=0.360
2026-04-19 04:40:18,302 INFO Regime epoch 13/50 — tr=0.3325 va=1.3189 acc=0.359
2026-04-19 04:40:19,075 INFO Regime epoch 14/50 — tr=0.3310 va=1.3221 acc=0.358
2026-04-19 04:40:19,877 INFO Regime epoch 15/50 — tr=0.3301 va=1.3141 acc=0.361 per_class={'TRENDING_UP': 0.464, 'TRENDING_DOWN': 0.372, 'RANGING': 0.0, 'VOLATILE': 0.793, 'CONSOLIDATION': 0.253}
2026-04-19 04:40:20,671 INFO Regime epoch 16/50 — tr=0.3289 va=1.3176 acc=0.359
2026-04-19 04:40:21,428 INFO Regime epoch 17/50 — tr=0.3278 va=1.3171 acc=0.359
2026-04-19 04:40:22,175 INFO Regime epoch 18/50 — tr=0.3273 va=1.3179 acc=0.358
2026-04-19 04:40:22,935 INFO Regime epoch 19/50 — tr=0.3266 va=1.3217 acc=0.356
2026-04-19 04:40:23,750 INFO Regime epoch 20/50 — tr=0.3259 va=1.3222 acc=0.355 per_class={'TRENDING_UP': 0.434, 'TRENDING_DOWN': 0.349, 'RANGING': 0.0, 'VOLATILE': 0.811, 'CONSOLIDATION': 0.262}
2026-04-19 04:40:24,496 INFO Regime epoch 21/50 — tr=0.3253 va=1.3200 acc=0.357
2026-04-19 04:40:25,245 INFO Regime epoch 22/50 — tr=0.3250 va=1.3168 acc=0.356
2026-04-19 04:40:25,246 INFO Regime early stop at epoch 22 (no_improve=10)
2026-04-19 04:40:25,296 WARNING RegimeClassifier accuracy 0.36 < 0.65 threshold
2026-04-19 04:40:25,299 INFO RegimeClassifier saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_1h.pkl
2026-04-19 04:40:25,299 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_1h.pkl
2026-04-19 04:40:25,436 INFO Regime 1H complete: acc=0.360, n=468252
2026-04-19 04:40:25,440 INFO Loaded XAUUSD/1H split=train: 80734 bars (2009-03-15 → 2022-08-16)
2026-04-19 04:40:25,979 INFO Rule labels [1H]: {'TRENDING_UP': 13359, 'TRENDING_DOWN': 10647, 'RANGING': 26355, 'VOLATILE': 17176, 'CONSOLIDATION': 13197}  ambiguous(conf<0.4)=79972 (total=80734)  short_runs_zeroed=79972
2026-04-19 04:40:25,984 INFO Regime[1H] persistence (avg bars/run) on XAUUSD 1H:
{0: 7.690846286701209, 1: 6.99080761654629, 2: 6.159149333956532, 3: 6.886928628708901, 4: 5.788157894736842}
2026-04-19 04:40:25,988 INFO Regime[1H] return separation on XAUUSD 1H:
{0: 3.527477809596854e-05, 1: -1.6049963815312037e-05, 2: 7.135431078521511e-06, 3: 1.504434637567949e-05, 4: 4.963068065210527e-06}
2026-04-19 04:40:26,000 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-19 04:40:26,000 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 04:40:26,000 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 04:40:26,000 INFO === VectorStore: building similarity indices ===
2026-04-19 04:40:26,000 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-19 04:40:26,001 INFO Retrain complete.
2026-04-19 04:40:27,052 INFO Model regime: SUCCESS
2026-04-19 04:40:27,052 INFO --- Training gru ---
2026-04-19 04:40:27,052 INFO Running retrain --model gru
2026-04-19 04:40:27,340 INFO retrain environment: KAGGLE
2026-04-19 04:40:29,041 INFO Device: CUDA (2 GPU(s))
2026-04-19 04:40:29,052 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 04:40:29,052 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 04:40:29,053 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-19 04:40:29,186 INFO NumExpr defaulting to 4 threads.
2026-04-19 04:40:29,378 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-19 04:40:29,379 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 04:40:29,379 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 04:40:29,381 WARNING GRULSTMPredictor: stale weights detected (gru feature contract changed: added=['regime_1h_4', 'regime_4h_4']; count 72→74) — deleting /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt so retrain starts fresh
2026-04-19 04:40:29,382 INFO Deleted stale weights (gru feature contract changed: added=['regime_1h_4', 'regime_4h_4']; count 72→74): /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 04:40:29,590 INFO Split boundaries loaded — train≤2022-08-16  val≤2024-03-05  test≤2025-08-05
2026-04-19 04:40:29,592 INFO Loaded AUDUSD/1H split=train: 38804 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:29,673 INFO Loaded EURGBP/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:29,750 INFO Loaded EURJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:29,824 INFO Loaded EURUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:29,902 INFO Loaded GBPJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:29,979 INFO Loaded GBPUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:30,051 INFO Loaded NZDUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:30,137 INFO Loaded USDCAD/1H split=train: 38808 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:30,228 INFO Loaded USDCHF/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:30,306 INFO Loaded USDJPY/1H split=train: 38811 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:30,416 INFO Loaded XAUUSD/1H split=train: 80734 bars (2009-03-15 → 2022-08-16)
2026-04-19 04:40:30,502 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-19 04:40:30,503 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260419_044030
2026-04-19 04:40:30,503 INFO GRU weights stale (gru feature contract changed: added=['regime_1h_4', 'regime_4h_4']; count 72→74) — deleting for full retrain
2026-04-19 04:40:30,621 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:30,621 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:30,643 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:30,650 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:30,651 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-19 04:40:30,652 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 04:40:30,652 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 04:40:30,653 INFO Loaded AUDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:30,743 INFO Rule labels [4H]: {'TRENDING_UP': 1833, 'TRENDING_DOWN': 1829, 'RANGING': 2945, 'VOLATILE': 2002, 'CONSOLIDATION': 1348}  ambiguous(conf<0.4)=6364 (total=9957)  short_runs_zeroed=5389
2026-04-19 04:40:30,745 INFO Loaded AUDUSD/1H split=train: 38804 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:30,776 INFO Loaded AUDUSD/5M split=train: 465551 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:31,072 INFO Loaded AUDUSD/15M split=train: 155205 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:31,198 INFO Loaded AUDUSD/1H split=train: 38804 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:31,291 INFO Loaded AUDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:31,489 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:31,490 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:31,505 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:31,512 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:31,513 INFO Loaded EURGBP/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:31,607 INFO Rule labels [4H]: {'TRENDING_UP': 1702, 'TRENDING_DOWN': 1928, 'RANGING': 3011, 'VOLATILE': 2030, 'CONSOLIDATION': 1286}  ambiguous(conf<0.4)=6241 (total=9957)  short_runs_zeroed=5356
2026-04-19 04:40:31,609 INFO Loaded EURGBP/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:31,630 INFO Loaded EURGBP/5M split=train: 465522 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:31,929 INFO Loaded EURGBP/15M split=train: 155214 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:32,055 INFO Loaded EURGBP/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:32,152 INFO Loaded EURGBP/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:32,336 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:32,336 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:32,352 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:32,360 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:32,361 INFO Loaded EURJPY/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:32,452 INFO Rule labels [4H]: {'TRENDING_UP': 2037, 'TRENDING_DOWN': 1677, 'RANGING': 2817, 'VOLATILE': 2015, 'CONSOLIDATION': 1411}  ambiguous(conf<0.4)=6152 (total=9957)  short_runs_zeroed=5275
2026-04-19 04:40:32,454 INFO Loaded EURJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:32,471 INFO Loaded EURJPY/5M split=train: 465569 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:32,759 INFO Loaded EURJPY/15M split=train: 155217 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:32,894 INFO Loaded EURJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:32,993 INFO Loaded EURJPY/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:33,173 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:33,174 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:33,189 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:33,197 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:33,198 INFO Loaded EURUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:33,294 INFO Rule labels [4H]: {'TRENDING_UP': 1509, 'TRENDING_DOWN': 2039, 'RANGING': 2977, 'VOLATILE': 2004, 'CONSOLIDATION': 1428}  ambiguous(conf<0.4)=6567 (total=9957)  short_runs_zeroed=5840
2026-04-19 04:40:33,296 INFO Loaded EURUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:33,322 INFO Loaded EURUSD/5M split=train: 465631 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:33,611 INFO Loaded EURUSD/15M split=train: 155220 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:33,742 INFO Loaded EURUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:33,840 INFO Loaded EURUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:34,024 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:34,025 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:34,042 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:34,049 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:34,050 INFO Loaded GBPJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:34,141 INFO Rule labels [4H]: {'TRENDING_UP': 1903, 'TRENDING_DOWN': 1665, 'RANGING': 3018, 'VOLATILE': 2032, 'CONSOLIDATION': 1340}  ambiguous(conf<0.4)=6061 (total=9958)  short_runs_zeroed=4947
2026-04-19 04:40:34,143 INFO Loaded GBPJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:34,160 INFO Loaded GBPJPY/5M split=train: 465412 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:34,433 INFO Loaded GBPJPY/15M split=train: 155212 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:34,565 INFO Loaded GBPJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:34,662 INFO Loaded GBPJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:34,844 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:34,845 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:34,861 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:34,869 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:34,870 INFO Loaded GBPUSD/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:34,961 INFO Rule labels [4H]: {'TRENDING_UP': 1740, 'TRENDING_DOWN': 1940, 'RANGING': 2949, 'VOLATILE': 2021, 'CONSOLIDATION': 1308}  ambiguous(conf<0.4)=6144 (total=9958)  short_runs_zeroed=5186
2026-04-19 04:40:34,963 INFO Loaded GBPUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:34,980 INFO Loaded GBPUSD/5M split=train: 465597 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:35,265 INFO Loaded GBPUSD/15M split=train: 155220 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:35,402 INFO Loaded GBPUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:35,515 INFO Loaded GBPUSD/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:35,687 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-19 04:40:35,688 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-19 04:40:35,703 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-19 04:40:35,710 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-19 04:40:35,711 INFO Loaded NZDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:35,804 INFO Rule labels [4H]: {'TRENDING_UP': 1770, 'TRENDING_DOWN': 1892, 'RANGING': 2939, 'VOLATILE': 2014, 'CONSOLIDATION': 1342}  ambiguous(conf<0.4)=6173 (total=9957)  short_runs_zeroed=5185
2026-04-19 04:40:35,806 INFO Loaded NZDUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:35,820 INFO Loaded NZDUSD/5M split=train: 465546 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:36,089 INFO Loaded NZDUSD/15M split=train: 155219 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:36,219 INFO Loaded NZDUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:36,316 INFO Loaded NZDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:36,492 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:36,493 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:36,508 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:36,515 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:36,516 INFO Loaded USDCAD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:36,611 INFO Rule labels [4H]: {'TRENDING_UP': 1656, 'TRENDING_DOWN': 2068, 'RANGING': 2799, 'VOLATILE': 2026, 'CONSOLIDATION': 1408}  ambiguous(conf<0.4)=5998 (total=9957)  short_runs_zeroed=5184
2026-04-19 04:40:36,613 INFO Loaded USDCAD/1H split=train: 38808 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:36,634 INFO Loaded USDCAD/5M split=train: 465582 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:36,921 INFO Loaded USDCAD/15M split=train: 155222 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:37,051 INFO Loaded USDCAD/1H split=train: 38808 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:37,151 INFO Loaded USDCAD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:37,337 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:37,338 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:37,354 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:37,361 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:37,362 INFO Loaded USDCHF/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:37,455 INFO Rule labels [4H]: {'TRENDING_UP': 1953, 'TRENDING_DOWN': 1781, 'RANGING': 2878, 'VOLATILE': 2018, 'CONSOLIDATION': 1327}  ambiguous(conf<0.4)=6098 (total=9957)  short_runs_zeroed=5260
2026-04-19 04:40:37,457 INFO Loaded USDCHF/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:37,475 INFO Loaded USDCHF/5M split=train: 465478 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:37,754 INFO Loaded USDCHF/15M split=train: 155208 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:37,888 INFO Loaded USDCHF/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:37,990 INFO Loaded USDCHF/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:38,176 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:38,177 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:38,193 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:38,200 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 04:40:38,201 INFO Loaded USDJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:38,292 INFO Rule labels [4H]: {'TRENDING_UP': 2196, 'TRENDING_DOWN': 1484, 'RANGING': 2821, 'VOLATILE': 2031, 'CONSOLIDATION': 1426}  ambiguous(conf<0.4)=6104 (total=9958)  short_runs_zeroed=5301
2026-04-19 04:40:38,294 INFO Loaded USDJPY/1H split=train: 38811 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:38,314 INFO Loaded USDJPY/5M split=train: 465705 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:38,601 INFO Loaded USDJPY/15M split=train: 155241 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:38,733 INFO Loaded USDJPY/1H split=train: 38811 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:38,830 INFO Loaded USDJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-19 04:40:39,115 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-19 04:40:39,117 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-19 04:40:39,134 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-19 04:40:39,145 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-19 04:40:39,146 INFO Loaded XAUUSD/4H split=train: 21467 bars (2009-03-15 → 2022-08-16)
2026-04-19 04:40:39,310 INFO Rule labels [4H]: {'TRENDING_UP': 4484, 'TRENDING_DOWN': 3949, 'RANGING': 5847, 'VOLATILE': 4296, 'CONSOLIDATION': 2891}  ambiguous(conf<0.4)=12923 (total=21467)  short_runs_zeroed=11105
2026-04-19 04:40:39,314 INFO Loaded XAUUSD/1H split=train: 80734 bars (2009-03-15 → 2022-08-16)
2026-04-19 04:40:39,359 INFO Loaded XAUUSD/5M split=train: 955298 bars (2009-03-15 → 2022-08-16)
2026-04-19 04:40:39,906 INFO Loaded XAUUSD/15M split=train: 319506 bars (2009-03-15 → 2022-08-16)
2026-04-19 04:40:40,112 INFO Loaded XAUUSD/1H split=train: 80734 bars (2009-03-15 → 2022-08-16)
2026-04-19 04:40:40,248 INFO Loaded XAUUSD/4H split=train: 21467 bars (2009-03-15 → 2022-08-16)
2026-04-19 04:40:40,364 INFO train_multi: 44 segments, ~7083076 total bars
2026-04-19 04:40:40,595 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-04-19 04:40:40,595 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-04-19 04:40:40,595 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-19 04:40:40,595 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-19 04:44:57,335 INFO train_multi TF=ALL: 7081756 sequences across 44 segments
2026-04-19 04:44:57,335 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479965 val=120014 n_feat=74 seq_len=30)
2026-04-19 04:44:58,643 INFO train_multi TF=ALL: train=479965 val=120014 (5335 MB tensors)
2026-04-19 04:45:18,982 INFO train_multi TF=ALL epoch 1/50 train=0.8196 val=0.7679
2026-04-19 04:45:18,986 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 04:45:18,986 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 04:45:18,986 INFO train_multi TF=ALL: new best val=0.7679 — saved
2026-04-19 04:45:33,565 INFO train_multi TF=ALL epoch 2/50 train=0.7050 val=0.6891
2026-04-19 04:45:33,568 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 04:45:33,569 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 04:45:33,569 INFO train_multi TF=ALL: new best val=0.6891 — saved
2026-04-19 04:45:48,107 INFO train_multi TF=ALL epoch 3/50 train=0.6895 val=0.6891
2026-04-19 04:46:02,545 INFO train_multi TF=ALL epoch 4/50 train=0.6890 val=0.6894
2026-04-19 04:46:17,007 INFO train_multi TF=ALL epoch 5/50 train=0.6881 val=0.6904
2026-04-19 04:46:31,450 INFO train_multi TF=ALL epoch 6/50 train=0.6867 val=0.6876
2026-04-19 04:46:31,454 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 04:46:31,454 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 04:46:31,454 INFO train_multi TF=ALL: new best val=0.6876 — saved
2026-04-19 04:46:45,983 INFO train_multi TF=ALL epoch 7/50 train=0.6835 val=0.6830
2026-04-19 04:46:45,986 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 04:46:45,986 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 04:46:45,986 INFO train_multi TF=ALL: new best val=0.6830 — saved
2026-04-19 04:47:00,508 INFO train_multi TF=ALL epoch 8/50 train=0.6747 val=0.6713
2026-04-19 04:47:00,511 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 04:47:00,511 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 04:47:00,511 INFO train_multi TF=ALL: new best val=0.6713 — saved
2026-04-19 04:47:14,970 INFO train_multi TF=ALL epoch 9/50 train=0.6632 val=0.6591
2026-04-19 04:47:14,973 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 04:47:14,973 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 04:47:14,974 INFO train_multi TF=ALL: new best val=0.6591 — saved
2026-04-19 04:47:29,338 INFO train_multi TF=ALL epoch 10/50 train=0.6527 val=0.6478
2026-04-19 04:47:29,341 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 04:47:29,341 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 04:47:29,341 INFO train_multi TF=ALL: new best val=0.6478 — saved
2026-04-19 04:47:43,815 INFO train_multi TF=ALL epoch 11/50 train=0.6446 val=0.6392
2026-04-19 04:47:43,818 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 04:47:43,818 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 04:47:43,818 INFO train_multi TF=ALL: new best val=0.6392 — saved
2026-04-19 04:47:58,366 INFO train_multi TF=ALL epoch 12/50 train=0.6380 val=0.6354
2026-04-19 04:47:58,369 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 04:47:58,369 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 04:47:58,369 INFO train_multi TF=ALL: new best val=0.6354 — saved
2026-04-19 04:48:12,917 INFO train_multi TF=ALL epoch 13/50 train=0.6333 val=0.6316
2026-04-19 04:48:12,920 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 04:48:12,920 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 04:48:12,920 INFO train_multi TF=ALL: new best val=0.6316 — saved
2026-04-19 04:48:27,229 INFO train_multi TF=ALL epoch 14/50 train=0.6281 val=0.6282
2026-04-19 04:48:27,232 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 04:48:27,233 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 04:48:27,233 INFO train_multi TF=ALL: new best val=0.6282 — saved
2026-04-19 04:48:41,627 INFO train_multi TF=ALL epoch 15/50 train=0.6240 val=0.6266
2026-04-19 04:48:41,631 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 04:48:41,631 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 04:48:41,631 INFO train_multi TF=ALL: new best val=0.6266 — saved
2026-04-19 04:48:56,399 INFO train_multi TF=ALL epoch 16/50 train=0.6206 val=0.6245
2026-04-19 04:48:56,403 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 04:48:56,403 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 04:48:56,403 INFO train_multi TF=ALL: new best val=0.6245 — saved
2026-04-19 04:49:11,020 INFO train_multi TF=ALL epoch 17/50 train=0.6172 val=0.6226
2026-04-19 04:49:11,024 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 04:49:11,024 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 04:49:11,024 INFO train_multi TF=ALL: new best val=0.6226 — saved
2026-04-19 04:49:25,678 INFO train_multi TF=ALL epoch 18/50 train=0.6138 val=0.6243
2026-04-19 04:49:40,120 INFO train_multi TF=ALL epoch 19/50 train=0.6110 val=0.6241
2026-04-19 04:49:54,677 INFO train_multi TF=ALL epoch 20/50 train=0.6085 val=0.6220
2026-04-19 04:49:54,681 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 04:49:54,681 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 04:49:54,681 INFO train_multi TF=ALL: new best val=0.6220 — saved
2026-04-19 04:50:09,207 INFO train_multi TF=ALL epoch 21/50 train=0.6060 val=0.6178
2026-04-19 04:50:09,210 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 04:50:09,210 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 04:50:09,210 INFO train_multi TF=ALL: new best val=0.6178 — saved
2026-04-19 04:50:23,836 INFO train_multi TF=ALL epoch 22/50 train=0.6034 val=0.6219
2026-04-19 04:50:38,595 INFO train_multi TF=ALL epoch 23/50 train=0.6011 val=0.6222
2026-04-19 04:50:53,241 INFO train_multi TF=ALL epoch 24/50 train=0.5990 val=0.6255
2026-04-19 04:51:07,833 INFO train_multi TF=ALL epoch 25/50 train=0.5969 val=0.6174
2026-04-19 04:51:07,836 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 04:51:07,836 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 04:51:07,836 INFO train_multi TF=ALL: new best val=0.6174 — saved
2026-04-19 04:51:22,790 INFO train_multi TF=ALL epoch 26/50 train=0.5950 val=0.6209
2026-04-19 04:51:37,247 INFO train_multi TF=ALL epoch 27/50 train=0.5925 val=0.6199
2026-04-19 04:51:52,013 INFO train_multi TF=ALL epoch 28/50 train=0.5906 val=0.6194
2026-04-19 04:52:06,620 INFO train_multi TF=ALL epoch 29/50 train=0.5882 val=0.6186
2026-04-19 04:52:21,263 INFO train_multi TF=ALL epoch 30/50 train=0.5854 val=0.6179
2026-04-19 04:52:21,263 INFO train_multi TF=ALL early stop at epoch 30
2026-04-19 04:52:21,396 INFO === VectorStore: building similarity indices ===
2026-04-19 04:52:21,396 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-19 04:52:21,396 INFO Retrain complete.
2026-04-19 04:52:23,297 INFO Model gru: SUCCESS
2026-04-19 04:52:23,297 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 04:52:23,297 WARNING   [MISSING] regime_classifier → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_classifier.pkl
2026-04-19 04:52:23,297 INFO   [OK] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-19 04:52:23,297 WARNING   [MISSING] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-19 04:52:23,297 WARNING Missing weights: ['regime_classifier', 'rl_ppo'] — run retrain_incremental.py for each
2026-04-19 04:52:23,298 INFO Saved 18 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime
  START Step 6 - Backtest
2026-04-19 04:52:23,847 INFO === STEP 6: BACKTEST + REINFORCED TRAINING (3 rounds) ===
2026-04-19 04:52:24,007 INFO Backtest date range: 2021-01-01 → 2024-03-05 (reinforcement loop, test set protected; set BT_START_FLOOR env to change)
2026-04-19 04:52:24,009 INFO Cleared existing journal for fresh reinforced training run
2026-04-19 04:52:24,009 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-19 04:52:24,009 INFO Round 1 — running backtest: 2021-01-01 → 2024-03-05 (ml_trader, shared ML cache)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260419_045224.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)             2621  48.3%   2.18 2180.3% 48.3% 13.8%   3.6%    3.83

Calibration Summary:
  all          [OK] Calibration OK — p_win correlates with actual win rate.
  ml_trader    [OK] Calibration OK — p_win correlates with actual win rate.
2026-04-19 04:58:47,296 INFO Round 1 backtest — 2621 trades | avg WR=48.3% | avg PF=2.18 | avg Sharpe=3.83
2026-04-19 04:58:47,296 INFO   ml_trader: 2621 trades | WR=48.3% | PF=2.18 | Return=2180.3% | DD=3.6% | Sharpe=3.83
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 2621
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2621 rows)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  AUDUSD        326 trades   296 days   1.10/day
  EURGBP        164 trades   138 days   1.19/day
  EURJPY        122 trades   100 days   1.22/day
  EURUSD         32 trades    32 days   1.00/day
  GBPJPY         92 trades    75 days   1.23/day
  GBPUSD        730 trades   691 days   1.06/day
  NZDUSD        206 trades   181 days   1.14/day
  USDCAD        227 trades   198 days   1.15/day
  USDCHF        204 trades   174 days   1.17/day
  USDJPY        424 trades   395 days   1.07/day
  XAUUSD         94 trades    76 days   1.24/day
  ✓  All symbols within normal range.
2026-04-19 04:58:48,460 INFO Round 1: wrote 2621 journal entries (total in file: 2621)
2026-04-19 04:58:48,463 INFO Round 1 — retraining regime...
2026-04-19 05:00:05,768 INFO Retrain regime: OK
2026-04-19 05:00:05,784 INFO Round 1 — retraining quality...
2026-04-19 05:00:13,503 INFO Retrain quality: OK
2026-04-19 05:00:13,519 INFO Round 1 — retraining rl...
2026-04-19 05:01:08,045 INFO Retrain rl: OK
2026-04-19 05:01:08,061 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-19 05:01:08,062 INFO Round 2 — running backtest: 2021-01-01 → 2024-03-05 (ml_trader, shared ML cache)
2026-04-19 05:07:32,671 INFO Round 2 backtest — 2597 trades | avg WR=48.7% | avg PF=2.20 | avg Sharpe=3.87
2026-04-19 05:07:32,671 INFO   ml_trader: 2597 trades | WR=48.7% | PF=2.20 | Return=2287.6% | DD=3.5% | Sharpe=3.87
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 2597
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2597 rows)
2026-04-19 05:07:33,815 INFO Round 2: wrote 2597 journal entries (total in file: 5218)
2026-04-19 05:07:33,817 INFO Round 2 — retraining regime...
2026-04-19 05:08:49,164 INFO Retrain regime: OK
2026-04-19 05:08:49,181 INFO Round 2 — retraining quality...
2026-04-19 05:09:00,485 INFO Retrain quality: OK
2026-04-19 05:09:00,509 INFO Round 2 — retraining rl...
2026-04-19 05:10:37,748 INFO Retrain rl: OK
2026-04-19 05:10:37,764 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-04-19 05:10:37,765 INFO Round 3 — running backtest: 2021-01-01 → 2024-03-05 (ml_trader, shared ML cache)
2026-04-19 05:16:54,919 INFO Round 3 backtest — 2611 trades | avg WR=48.5% | avg PF=2.20 | avg Sharpe=3.85
2026-04-19 05:16:54,919 INFO   ml_trader: 2611 trades | WR=48.5% | PF=2.20 | Return=2280.4% | DD=3.5% | Sharpe=3.85
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 2611
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2611 rows)
2026-04-19 05:16:56,078 INFO Round 3: wrote 2611 journal entries (total in file: 7829)
2026-04-19 05:16:56,080 INFO Round 3 (final): retraining after last backtest...
2026-04-19 05:16:56,080 INFO Round 3 — retraining regime...
2026-04-19 05:18:14,012 INFO Retrain regime: OK
2026-04-19 05:18:14,029 INFO Round 3 — retraining quality...
2026-04-19 05:18:31,233 INFO Retrain quality: OK
2026-04-19 05:18:31,249 INFO Round 3 — retraining rl...
2026-04-19 05:20:54,059 INFO Retrain rl: OK
2026-04-19 05:20:54,076 INFO Improvement round 1 → 3: WR +0.3% | PF +0.016 | Sharpe +0.022
2026-04-19 05:20:54,224 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-19 05:20:54,232 INFO Journal entries: 7829
2026-04-19 05:20:54,233 INFO --- Training quality ---
2026-04-19 05:20:54,233 INFO Running retrain --model quality
  DONE  Step 6 - Backtest
  START Step 7b - Quality+RL
2026-04-19 05:20:54,580 INFO retrain environment: KAGGLE
2026-04-19 05:20:56,334 INFO Device: CUDA (2 GPU(s))
2026-04-19 05:20:56,346 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 05:20:56,346 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 05:20:56,347 INFO === QualityScorer retrain ===
2026-04-19 05:20:56,479 INFO NumExpr defaulting to 4 threads.
2026-04-19 05:20:56,672 INFO QualityScorer: CUDA available — using GPU
2026-04-19 05:20:56,874 INFO QualityScorer loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (device=cuda)
2026-04-19 05:20:57,146 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/quality_scorer.pkl_20260419_052057
2026-04-19 05:20:57,392 INFO QualityScorer: 7811 samples, EV stats={'mean': 0.13820305466651917, 'std': 1.2757495641708374, 'n_pos': 3778, 'n_neg': 4033}, device=cuda
2026-04-19 05:20:57,393 INFO QualityScorer: normalised win labels by median_win=1.012 — EV range now [-1, +3]
2026-04-19 05:20:57,393 INFO QualityScorer: warm start from existing weights
2026-04-19 05:20:57,394 INFO QualityScorer: pos_weight=1.04 (n_pos=3056 n_neg=3192)
2026-04-19 05:20:59,038 INFO Quality epoch   1/100 — va_huber=0.6290
2026-04-19 05:20:59,194 INFO Quality epoch   2/100 — va_huber=0.6292
2026-04-19 05:20:59,347 INFO Quality epoch   3/100 — va_huber=0.6287
2026-04-19 05:20:59,494 INFO Quality epoch   4/100 — va_huber=0.6291
2026-04-19 05:20:59,639 INFO Quality epoch   5/100 — va_huber=0.6292
2026-04-19 05:21:00,524 INFO Quality epoch  11/100 — va_huber=0.6288
2026-04-19 05:21:01,932 INFO Quality epoch  21/100 — va_huber=0.6284
2026-04-19 05:21:03,361 INFO Quality epoch  31/100 — va_huber=0.6279
2026-04-19 05:21:04,774 INFO Quality epoch  41/100 — va_huber=0.6279
2026-04-19 05:21:05,669 INFO Quality early stop at epoch 46
2026-04-19 05:21:05,693 INFO QualityScorer EV model: MAE=1.070 dir_acc=0.605 n_val=1563
2026-04-19 05:21:05,696 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-19 05:21:05,770 INFO Retrain complete.
2026-04-19 05:21:06,633 INFO Model quality: SUCCESS
2026-04-19 05:21:06,633 INFO --- Training rl ---
2026-04-19 05:21:06,634 INFO Running retrain --model rl
2026-04-19 05:21:06,899 INFO retrain environment: KAGGLE
2026-04-19 05:21:08,608 INFO Device: CUDA (2 GPU(s))
2026-04-19 05:21:08,619 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 05:21:08,619 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 05:21:08,620 INFO === RLAgent (PPO) retrain ===
2026-04-19 05:21:08,623 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260419_052108
2026-04-19 05:21:09.449927: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1776576069.472406  108959 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1776576069.480006  108959 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1776576069.499260  108959 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1776576069.499288  108959 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1776576069.499290  108959 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1776576069.499293  108959 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-19 05:21:13,860 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-19 05:21:16,618 INFO RLAgent: PPO model loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-19 05:21:16,809 INFO RLAgent: warm start — fine-tuning existing PPO policy (lr=6.00e-05)
2026-04-19 05:23:33,058 INFO RLAgent: retrain complete, 7829 episodes
2026-04-19 05:23:33,058 INFO Retrain complete.
2026-04-19 05:23:34,572 INFO Model rl: SUCCESS
2026-04-19 05:23:34,572 INFO Step 7b complete — summary: /kaggle/working/Multi-Bot/trading-system/ml_training/metrics/training_7b_summary.json
INFO  === STEP 8: PUSH TRAINING OUTPUTS TO GITHUB ===
INFO  Repo:   AnalystTKZ/Multi-Bot
INFO  Branch: main
INFO  Root:   /kaggle/working/remote/Multi-Bot
INFO  Pulling latest from origin/main ...
  DONE  Step 7b - Quality+RL
