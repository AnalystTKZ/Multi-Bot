All scripts and inputs verified.

=== Running pipeline ===
  SKIP  Step 0 - Resample
  SKIP  Step 1 - Inventory
  SKIP  Step 2 - Cleaning
  SKIP  Step 3 - Alignment
  SKIP  Step 4 - Features
  START Step 5 - Split
2026-04-18 20:42:46,660 INFO Loading feature-engineered data...
2026-04-18 20:42:47,139 INFO Loaded 221743 rows, 202 features
2026-04-18 20:42:47,139 INFO Train: 155220 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:42:47,139 INFO Val:   33261 bars (2022-08-16 → 2024-03-05)
2026-04-18 20:42:47,139 INFO Test:  33262 bars (2024-03-05 → 2025-08-05)
2026-04-18 20:42:47,140 INFO No leakage confirmed: train < val < test timestamps

=== SPLIT COMPLETE (no shuffling, time-based) ===
  Train:      155,220 bars  2016-01-04 → 2022-08-16
  Validation:  33,261 bars  2022-08-16 → 2024-03-05
  Test:        33,262 bars  2024-03-05 → 2025-08-05
  Features: 202
  Leakage check: PASS
  DONE  Step 5 - Split
  START Step 7a - GRU+Regime
2026-04-18 20:42:49,603 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-18 20:42:49,603 INFO --- Training regime ---
2026-04-18 20:42:49,604 INFO Running retrain --model regime
2026-04-18 20:42:49,696 INFO retrain environment: KAGGLE
2026-04-18 20:42:51,362 INFO Device: CUDA (2 GPU(s))
2026-04-18 20:42:51,374 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-18 20:42:51,374 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-18 20:42:51,375 INFO === RegimeClassifier retrain (dual-TF cascade: 4H bias + 1H structure) ===
2026-04-18 20:42:51,518 INFO NumExpr defaulting to 4 threads.
2026-04-18 20:42:51,727 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-18 20:42:51,727 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-18 20:42:51,727 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-18 20:42:51,916 INFO Split boundaries loaded — train≤2022-08-16  val≤2024-03-05  test≤2025-08-05
2026-04-18 20:42:51,918 INFO Loaded AUDUSD/1H split=train: 38804 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:42:51,995 INFO Loaded EURGBP/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:42:52,067 INFO Loaded EURJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:42:52,136 INFO Loaded EURUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:42:52,209 INFO Loaded GBPJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:42:52,280 INFO Loaded GBPUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:42:52,347 INFO Loaded NZDUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:42:52,417 INFO Loaded USDCAD/1H split=train: 38808 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:42:52,537 INFO Loaded USDCHF/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:42:52,634 INFO Loaded USDJPY/1H split=train: 38811 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:42:52,743 INFO Loaded XAUUSD/1H split=train: 80734 bars (2009-03-15 → 2022-08-16)
2026-04-18 20:42:52,808 INFO Regime: fitting per-group GMMs on 4H data (dollar / cross / gold)...
2026-04-18 20:42:52,822 INFO Loaded AUDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:42:52,837 INFO Loaded EURGBP/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:42:52,852 INFO Loaded EURJPY/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:42:52,870 INFO Loaded EURUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:42:52,885 INFO Loaded GBPJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:42:52,900 INFO Loaded GBPUSD/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:42:53,014 INFO Loaded NZDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:42:53,028 INFO Loaded USDCAD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:42:53,042 INFO Loaded USDCHF/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:42:53,056 INFO Loaded USDJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:42:53,073 INFO Loaded XAUUSD/4H split=train: 21467 bars (2009-03-15 → 2022-08-16)
2026-04-18 20:42:53,855 INFO GMM fit: timeframe=4H → n_bar=50
2026-04-18 20:42:59,938 INFO GMM fitted on 69344 samples — cluster→regime: {0: 2, 1: 1, 2: 3, 3: 0} dist: {'TRENDING_UP': 1, 'TRENDING_DOWN': 1, 'RANGING': 1, 'VOLATILE': 1}
2026-04-18 20:42:59,942 INFO Regime: GMM 'dollar' fitted on 7 4H dfs (n_bar=50)
2026-04-18 20:42:59,943 INFO GMM fit: timeframe=4H → n_bar=50
2026-04-18 20:43:01,987 INFO GMM fitted on 29719 samples — cluster→regime: {0: 2, 1: 2, 2: 3, 3: 0} dist: {'TRENDING_UP': 1, 'TRENDING_DOWN': 0, 'RANGING': 2, 'VOLATILE': 1}
2026-04-18 20:43:01,987 INFO Regime: GMM 'cross' fitted on 3 4H dfs (n_bar=50)
2026-04-18 20:43:01,987 INFO GMM fit: timeframe=4H → n_bar=50
2026-04-18 20:43:02,489 INFO GMM fitted on 10708 samples — cluster→regime: {0: 2, 1: 2, 2: 0, 3: 2} dist: {'TRENDING_UP': 1, 'TRENDING_DOWN': 0, 'RANGING': 3, 'VOLATILE': 0}
2026-04-18 20:43:02,489 INFO Regime: GMM 'gold' fitted on 1 4H dfs (n_bar=50)
2026-04-18 20:43:02,589 INFO Regime: training 4H bias classifier...
2026-04-18 20:43:02,591 INFO Loaded AUDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:02,592 INFO Loaded EURGBP/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:02,593 INFO Loaded EURJPY/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:02,594 INFO Loaded EURUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:02,596 INFO Loaded GBPJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:02,597 INFO Loaded GBPUSD/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:02,598 INFO Loaded NZDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:02,599 INFO Loaded USDCAD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:02,600 INFO Loaded USDCHF/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:02,602 INFO Loaded USDJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:02,604 INFO Loaded XAUUSD/4H split=train: 21467 bars (2009-03-15 → 2022-08-16)
2026-04-18 20:43:02,728 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:02,770 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:02,771 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:02,772 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:02,780 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:02,781 INFO Loaded AUDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:03,219 INFO Rule labels [4H]: {'TRENDING_UP': 1833, 'TRENDING_DOWN': 1829, 'RANGING': 4293, 'VOLATILE': 2002}  ambiguous(conf<0.4)=2412 (total=9957)
2026-04-18 20:43:03,220 INFO Regime[4H]: collected AUDUSD — 9907 samples (group=dollar)
2026-04-18 20:43:03,388 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:03,422 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:03,423 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:03,423 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:03,431 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:03,432 INFO Loaded EURGBP/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:03,823 INFO Rule labels [4H]: {'TRENDING_UP': 1702, 'TRENDING_DOWN': 1928, 'RANGING': 4297, 'VOLATILE': 2030}  ambiguous(conf<0.4)=2606 (total=9957)
2026-04-18 20:43:03,824 INFO Regime[4H]: collected EURGBP — 9907 samples (group=cross)
2026-04-18 20:43:04,002 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:04,038 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:04,038 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:04,039 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:04,048 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:04,049 INFO Loaded EURJPY/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:04,431 INFO Rule labels [4H]: {'TRENDING_UP': 2037, 'TRENDING_DOWN': 1677, 'RANGING': 4228, 'VOLATILE': 2015}  ambiguous(conf<0.4)=2417 (total=9957)
2026-04-18 20:43:04,432 INFO Regime[4H]: collected EURJPY — 9907 samples (group=cross)
2026-04-18 20:43:04,594 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:04,629 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:04,630 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:04,631 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:04,640 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:04,641 INFO Loaded EURUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:05,032 INFO Rule labels [4H]: {'TRENDING_UP': 1509, 'TRENDING_DOWN': 2039, 'RANGING': 4405, 'VOLATILE': 2004}  ambiguous(conf<0.4)=2510 (total=9957)
2026-04-18 20:43:05,033 INFO Regime[4H]: collected EURUSD — 9907 samples (group=dollar)
2026-04-18 20:43:05,200 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:05,236 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:05,236 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:05,237 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:05,245 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:05,246 INFO Loaded GBPJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:05,634 INFO Rule labels [4H]: {'TRENDING_UP': 1903, 'TRENDING_DOWN': 1665, 'RANGING': 4358, 'VOLATILE': 2032}  ambiguous(conf<0.4)=2532 (total=9958)
2026-04-18 20:43:05,636 INFO Regime[4H]: collected GBPJPY — 9908 samples (group=cross)
2026-04-18 20:43:05,802 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:05,835 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:05,836 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:05,837 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:05,845 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:05,846 INFO Loaded GBPUSD/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:06,237 INFO Rule labels [4H]: {'TRENDING_UP': 1740, 'TRENDING_DOWN': 1940, 'RANGING': 4257, 'VOLATILE': 2021}  ambiguous(conf<0.4)=2464 (total=9958)
2026-04-18 20:43:06,238 INFO Regime[4H]: collected GBPUSD — 9908 samples (group=dollar)
2026-04-18 20:43:06,400 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-18 20:43:06,436 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-18 20:43:06,437 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-18 20:43:06,437 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-18 20:43:06,445 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-18 20:43:06,446 INFO Loaded NZDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:06,832 INFO Rule labels [4H]: {'TRENDING_UP': 1770, 'TRENDING_DOWN': 1892, 'RANGING': 4281, 'VOLATILE': 2014}  ambiguous(conf<0.4)=2604 (total=9957)
2026-04-18 20:43:06,833 INFO Regime[4H]: collected NZDUSD — 9907 samples (group=dollar)
2026-04-18 20:43:06,994 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:07,025 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:07,026 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:07,027 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:07,035 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:07,036 INFO Loaded USDCAD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:07,417 INFO Rule labels [4H]: {'TRENDING_UP': 1656, 'TRENDING_DOWN': 2068, 'RANGING': 4207, 'VOLATILE': 2026}  ambiguous(conf<0.4)=2378 (total=9957)
2026-04-18 20:43:07,418 INFO Regime[4H]: collected USDCAD — 9907 samples (group=dollar)
2026-04-18 20:43:07,584 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:07,616 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:07,616 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:07,617 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:07,626 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:07,627 INFO Loaded USDCHF/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:08,010 INFO Rule labels [4H]: {'TRENDING_UP': 1953, 'TRENDING_DOWN': 1781, 'RANGING': 4205, 'VOLATILE': 2018}  ambiguous(conf<0.4)=2349 (total=9957)
2026-04-18 20:43:08,012 INFO Regime[4H]: collected USDCHF — 9907 samples (group=dollar)
2026-04-18 20:43:08,178 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:08,211 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:08,212 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:08,212 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:08,220 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:08,221 INFO Loaded USDJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:08,604 INFO Rule labels [4H]: {'TRENDING_UP': 2196, 'TRENDING_DOWN': 1484, 'RANGING': 4247, 'VOLATILE': 2031}  ambiguous(conf<0.4)=2394 (total=9958)
2026-04-18 20:43:08,605 INFO Regime[4H]: collected USDJPY — 9908 samples (group=dollar)
2026-04-18 20:43:08,869 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-18 20:43:08,921 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-18 20:43:08,922 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-18 20:43:08,923 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-18 20:43:08,933 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-18 20:43:08,935 INFO Loaded XAUUSD/4H split=train: 21467 bars (2009-03-15 → 2022-08-16)
2026-04-18 20:43:09,709 INFO Rule labels [4H]: {'TRENDING_UP': 4484, 'TRENDING_DOWN': 3949, 'RANGING': 8738, 'VOLATILE': 4296}  ambiguous(conf<0.4)=4817 (total=21467)
2026-04-18 20:43:09,710 INFO Regime[4H]: collected XAUUSD — 21417 samples (group=gold)
2026-04-18 20:43:09,873 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_4h.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_4h.pkl_20260418_204309
2026-04-18 20:43:10,074 INFO RegimeClassifier loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_4h.pkl (device=cuda, features=31)
2026-04-18 20:43:10,094 INFO RegimeClassifier: 120490 samples, classes={'TRENDING_UP': 22783, 'TRENDING_DOWN': 22252, 'RANGING': 50966, 'VOLATILE': 24489}, device=cuda
2026-04-18 20:43:10,094 INFO RegimeClassifier: sample weights — mean=0.563  ambiguous(<0.4)=24.4%
2026-04-18 20:43:10,094 INFO RegimeClassifier: warm start from existing weights
2026-04-18 20:43:10,094 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-18 20:43:12,493 INFO Regime epoch  1/50 — tr=0.6136 va=1.0388 acc=0.475 per_class={'TRENDING_UP': 0.643, 'TRENDING_DOWN': 0.535, 'RANGING': 0.168, 'VOLATILE': 0.839}
2026-04-18 20:43:12,701 INFO Regime epoch  2/50 — tr=0.6133 va=1.0402 acc=0.473
2026-04-18 20:43:12,897 INFO Regime epoch  3/50 — tr=0.6133 va=1.0377 acc=0.477
2026-04-18 20:43:13,097 INFO Regime epoch  4/50 — tr=0.6128 va=1.0430 acc=0.471
2026-04-18 20:43:13,313 INFO Regime epoch  5/50 — tr=0.6131 va=1.0416 acc=0.471 per_class={'TRENDING_UP': 0.636, 'TRENDING_DOWN': 0.534, 'RANGING': 0.162, 'VOLATILE': 0.839}
2026-04-18 20:43:13,516 INFO Regime epoch  6/50 — tr=0.6134 va=1.0409 acc=0.473
2026-04-18 20:43:13,727 INFO Regime epoch  7/50 — tr=0.6128 va=1.0415 acc=0.472
2026-04-18 20:43:13,942 INFO Regime epoch  8/50 — tr=0.6115 va=1.0384 acc=0.474
2026-04-18 20:43:14,146 INFO Regime epoch  9/50 — tr=0.6112 va=1.0453 acc=0.470
2026-04-18 20:43:14,354 INFO Regime epoch 10/50 — tr=0.6112 va=1.0409 acc=0.472 per_class={'TRENDING_UP': 0.641, 'TRENDING_DOWN': 0.538, 'RANGING': 0.16, 'VOLATILE': 0.838}
2026-04-18 20:43:14,546 INFO Regime epoch 11/50 — tr=0.6110 va=1.0450 acc=0.469
2026-04-18 20:43:14,748 INFO Regime epoch 12/50 — tr=0.6106 va=1.0432 acc=0.470
2026-04-18 20:43:14,957 INFO Regime epoch 13/50 — tr=0.6103 va=1.0404 acc=0.472
2026-04-18 20:43:14,957 INFO Regime early stop at epoch 13 (no_improve=10)
2026-04-18 20:43:14,971 WARNING RegimeClassifier accuracy 0.48 < 0.65 threshold
2026-04-18 20:43:14,974 INFO RegimeClassifier saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_4h.pkl
2026-04-18 20:43:14,975 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_4h.pkl
2026-04-18 20:43:15,096 INFO Regime 4H complete: acc=0.477, n=120490
2026-04-18 20:43:15,098 INFO Loaded XAUUSD/4H split=train: 21467 bars (2009-03-15 → 2022-08-16)
2026-04-18 20:43:15,232 INFO Rule labels [4H]: {'TRENDING_UP': 4484, 'TRENDING_DOWN': 3949, 'RANGING': 8738, 'VOLATILE': 4296}  ambiguous(conf<0.4)=4817 (total=21467)
2026-04-18 20:43:15,235 INFO Regime[4H] persistence (avg bars/run) on XAUUSD 4H:
{0: 9.6017130620985, 1: 9.84788029925187, 2: 10.977386934673367, 3: 9.990697674418605}
2026-04-18 20:43:15,237 INFO Regime[4H] return separation on XAUUSD 4H:
{0: 7.975048601084771e-05, 1: 1.4941597824152149e-05, 2: 2.2194974086795702e-05, 3: 4.7707963375047856e-05}
2026-04-18 20:43:15,237 INFO Regime: training 1H structure classifier...
2026-04-18 20:43:15,238 INFO Loaded AUDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:15,239 INFO Loaded EURGBP/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:15,240 INFO Loaded EURJPY/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:15,241 INFO Loaded EURUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:15,241 INFO Loaded GBPJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:15,242 INFO Loaded GBPUSD/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:15,243 INFO Loaded NZDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:15,244 INFO Loaded USDCAD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:15,245 INFO Loaded USDCHF/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:15,246 INFO Loaded USDJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:15,247 INFO Loaded XAUUSD/4H split=train: 21467 bars (2009-03-15 → 2022-08-16)
2026-04-18 20:43:15,253 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:15,255 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:15,256 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:15,256 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:15,257 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:15,260 INFO Loaded AUDUSD/1H split=train: 38804 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:15,755 INFO Rule labels [4H]: {'TRENDING_UP': 1833, 'TRENDING_DOWN': 1829, 'RANGING': 4293, 'VOLATILE': 2002}  ambiguous(conf<0.4)=2412 (total=9957)
2026-04-18 20:43:15,759 INFO Regime[1H]: collected AUDUSD — 38754 samples (group=dollar)
2026-04-18 20:43:15,889 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:15,892 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:15,893 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:15,893 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:15,893 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:15,895 INFO Loaded EURGBP/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:16,330 INFO Rule labels [4H]: {'TRENDING_UP': 1702, 'TRENDING_DOWN': 1928, 'RANGING': 4297, 'VOLATILE': 2030}  ambiguous(conf<0.4)=2606 (total=9957)
2026-04-18 20:43:16,336 INFO Regime[1H]: collected EURGBP — 38756 samples (group=cross)
2026-04-18 20:43:16,485 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:16,488 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:16,489 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:16,489 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:16,489 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:16,492 INFO Loaded EURJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:16,920 INFO Rule labels [4H]: {'TRENDING_UP': 2037, 'TRENDING_DOWN': 1677, 'RANGING': 4228, 'VOLATILE': 2015}  ambiguous(conf<0.4)=2417 (total=9957)
2026-04-18 20:43:16,924 INFO Regime[1H]: collected EURJPY — 38756 samples (group=cross)
2026-04-18 20:43:17,052 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:17,054 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:17,055 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:17,055 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:17,055 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:17,057 INFO Loaded EURUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:17,488 INFO Rule labels [4H]: {'TRENDING_UP': 1509, 'TRENDING_DOWN': 2039, 'RANGING': 4405, 'VOLATILE': 2004}  ambiguous(conf<0.4)=2510 (total=9957)
2026-04-18 20:43:17,492 INFO Regime[1H]: collected EURUSD — 38757 samples (group=dollar)
2026-04-18 20:43:17,619 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:17,621 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:17,622 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:17,622 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:17,622 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:17,625 INFO Loaded GBPJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:18,063 INFO Rule labels [4H]: {'TRENDING_UP': 1903, 'TRENDING_DOWN': 1665, 'RANGING': 4358, 'VOLATILE': 2032}  ambiguous(conf<0.4)=2532 (total=9958)
2026-04-18 20:43:18,067 INFO Regime[1H]: collected GBPJPY — 38756 samples (group=cross)
2026-04-18 20:43:18,197 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:18,200 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:18,201 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:18,201 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:18,201 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:18,203 INFO Loaded GBPUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:18,636 INFO Rule labels [4H]: {'TRENDING_UP': 1740, 'TRENDING_DOWN': 1940, 'RANGING': 4257, 'VOLATILE': 2021}  ambiguous(conf<0.4)=2464 (total=9958)
2026-04-18 20:43:18,640 INFO Regime[1H]: collected GBPUSD — 38757 samples (group=dollar)
2026-04-18 20:43:18,771 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-18 20:43:18,773 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-18 20:43:18,774 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-18 20:43:18,774 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-18 20:43:18,774 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-18 20:43:18,776 INFO Loaded NZDUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:19,219 INFO Rule labels [4H]: {'TRENDING_UP': 1770, 'TRENDING_DOWN': 1892, 'RANGING': 4281, 'VOLATILE': 2014}  ambiguous(conf<0.4)=2604 (total=9957)
2026-04-18 20:43:19,224 INFO Regime[1H]: collected NZDUSD — 38757 samples (group=dollar)
2026-04-18 20:43:19,350 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:19,353 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:19,353 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:19,354 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:19,354 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:19,356 INFO Loaded USDCAD/1H split=train: 38808 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:19,809 INFO Rule labels [4H]: {'TRENDING_UP': 1656, 'TRENDING_DOWN': 2068, 'RANGING': 4207, 'VOLATILE': 2026}  ambiguous(conf<0.4)=2378 (total=9957)
2026-04-18 20:43:19,814 INFO Regime[1H]: collected USDCAD — 38758 samples (group=dollar)
2026-04-18 20:43:19,942 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:19,944 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:19,945 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:19,946 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:19,946 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:19,948 INFO Loaded USDCHF/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:20,375 INFO Rule labels [4H]: {'TRENDING_UP': 1953, 'TRENDING_DOWN': 1781, 'RANGING': 4205, 'VOLATILE': 2018}  ambiguous(conf<0.4)=2349 (total=9957)
2026-04-18 20:43:20,379 INFO Regime[1H]: collected USDCHF — 38756 samples (group=dollar)
2026-04-18 20:43:20,509 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:20,511 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:20,512 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:20,512 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:20,512 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:20,514 INFO Loaded USDJPY/1H split=train: 38811 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:20,954 INFO Rule labels [4H]: {'TRENDING_UP': 2196, 'TRENDING_DOWN': 1484, 'RANGING': 4247, 'VOLATILE': 2031}  ambiguous(conf<0.4)=2394 (total=9958)
2026-04-18 20:43:20,958 INFO Regime[1H]: collected USDJPY — 38761 samples (group=dollar)
2026-04-18 20:43:21,097 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-18 20:43:21,101 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-18 20:43:21,102 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-18 20:43:21,103 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-18 20:43:21,103 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-18 20:43:21,106 INFO Loaded XAUUSD/1H split=train: 80734 bars (2009-03-15 → 2022-08-16)
2026-04-18 20:43:21,980 INFO Rule labels [4H]: {'TRENDING_UP': 4484, 'TRENDING_DOWN': 3949, 'RANGING': 8738, 'VOLATILE': 4296}  ambiguous(conf<0.4)=4817 (total=21467)
2026-04-18 20:43:21,988 INFO Regime[1H]: collected XAUUSD — 80684 samples (group=gold)
2026-04-18 20:43:22,276 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_1h.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_1h.pkl_20260418_204322
2026-04-18 20:43:22,280 INFO RegimeClassifier loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_1h.pkl (device=cuda, features=15)
2026-04-18 20:43:22,351 INFO RegimeClassifier: 468252 samples, classes={'TRENDING_UP': 88030, 'TRENDING_DOWN': 85841, 'RANGING': 199234, 'VOLATILE': 95147}, device=cuda
2026-04-18 20:43:22,352 INFO RegimeClassifier: sample weights — mean=0.562  ambiguous(<0.4)=24.8%
2026-04-18 20:43:22,352 INFO RegimeClassifier: warm start from existing weights
2026-04-18 20:43:22,352 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-18 20:43:23,206 INFO Regime epoch  1/50 — tr=0.6268 va=1.0934 acc=0.394 per_class={'TRENDING_UP': 0.498, 'TRENDING_DOWN': 0.461, 'RANGING': 0.1, 'VOLATILE': 0.807}
2026-04-18 20:43:23,963 INFO Regime epoch  2/50 — tr=0.6271 va=1.0913 acc=0.395
2026-04-18 20:43:24,712 INFO Regime epoch  3/50 — tr=0.6269 va=1.0961 acc=0.393
2026-04-18 20:43:25,475 INFO Regime epoch  4/50 — tr=0.6268 va=1.0926 acc=0.394
2026-04-18 20:43:26,314 INFO Regime epoch  5/50 — tr=0.6266 va=1.0932 acc=0.394 per_class={'TRENDING_UP': 0.495, 'TRENDING_DOWN': 0.47, 'RANGING': 0.098, 'VOLATILE': 0.806}
2026-04-18 20:43:27,097 INFO Regime epoch  6/50 — tr=0.6265 va=1.0961 acc=0.393
2026-04-18 20:43:27,847 INFO Regime epoch  7/50 — tr=0.6263 va=1.0963 acc=0.394
2026-04-18 20:43:28,645 INFO Regime epoch  8/50 — tr=0.6262 va=1.1010 acc=0.392
2026-04-18 20:43:29,428 INFO Regime epoch  9/50 — tr=0.6262 va=1.0971 acc=0.392
2026-04-18 20:43:30,261 INFO Regime epoch 10/50 — tr=0.6260 va=1.0989 acc=0.392 per_class={'TRENDING_UP': 0.49, 'TRENDING_DOWN': 0.452, 'RANGING': 0.102, 'VOLATILE': 0.809}
2026-04-18 20:43:31,041 INFO Regime epoch 11/50 — tr=0.6258 va=1.0950 acc=0.394
2026-04-18 20:43:31,801 INFO Regime epoch 12/50 — tr=0.6256 va=1.0978 acc=0.393
2026-04-18 20:43:31,801 INFO Regime early stop at epoch 12 (no_improve=10)
2026-04-18 20:43:31,854 WARNING RegimeClassifier accuracy 0.39 < 0.65 threshold
2026-04-18 20:43:31,857 INFO RegimeClassifier saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_1h.pkl
2026-04-18 20:43:31,857 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_1h.pkl
2026-04-18 20:43:31,981 INFO Regime 1H complete: acc=0.395, n=468252
2026-04-18 20:43:31,985 INFO Loaded XAUUSD/1H split=train: 80734 bars (2009-03-15 → 2022-08-16)
2026-04-18 20:43:32,455 INFO Rule labels [1H]: {'TRENDING_UP': 13359, 'TRENDING_DOWN': 10647, 'RANGING': 39552, 'VOLATILE': 17176}  ambiguous(conf<0.4)=22905 (total=80734)
2026-04-18 20:43:32,460 INFO Regime[1H] persistence (avg bars/run) on XAUUSD 1H:
{0: 7.690846286701209, 1: 6.99080761654629, 2: 11.59542656112577, 3: 6.886928628708901}
2026-04-18 20:43:32,463 INFO Regime[1H] return separation on XAUUSD 1H:
{0: 3.527477809596854e-05, 1: -1.6049963815312037e-05, 2: 6.410596059137787e-06, 3: 1.504434637567949e-05}
2026-04-18 20:43:32,474 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-18 20:43:32,474 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-18 20:43:32,474 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-18 20:43:32,475 INFO === VectorStore: building similarity indices ===
2026-04-18 20:43:32,475 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-18 20:43:32,475 INFO Retrain complete.
2026-04-18 20:43:33,518 INFO Model regime: SUCCESS
2026-04-18 20:43:33,518 INFO --- Training gru ---
2026-04-18 20:43:33,518 INFO Running retrain --model gru
2026-04-18 20:43:33,609 INFO retrain environment: KAGGLE
2026-04-18 20:43:35,271 INFO Device: CUDA (2 GPU(s))
2026-04-18 20:43:35,280 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-18 20:43:35,280 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-18 20:43:35,281 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-18 20:43:35,411 INFO NumExpr defaulting to 4 threads.
2026-04-18 20:43:35,602 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-18 20:43:35,602 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-18 20:43:35,602 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-18 20:43:35,843 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-18 20:43:36,054 INFO Split boundaries loaded — train≤2022-08-16  val≤2024-03-05  test≤2025-08-05
2026-04-18 20:43:36,056 INFO Loaded AUDUSD/1H split=train: 38804 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:36,137 INFO Loaded EURGBP/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:36,224 INFO Loaded EURJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:36,296 INFO Loaded EURUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:36,382 INFO Loaded GBPJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:36,467 INFO Loaded GBPUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:36,538 INFO Loaded NZDUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:36,618 INFO Loaded USDCAD/1H split=train: 38808 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:36,693 INFO Loaded USDCHF/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:36,767 INFO Loaded USDJPY/1H split=train: 38811 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:36,853 INFO Loaded XAUUSD/1H split=train: 80734 bars (2009-03-15 → 2022-08-16)
2026-04-18 20:43:36,916 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-18 20:43:36,917 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260418_204336
2026-04-18 20:43:36,920 INFO GRU feature contract unchanged (input_size=72) — incremental retrain
2026-04-18 20:43:37,037 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:37,038 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:37,060 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:37,067 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:37,069 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-18 20:43:37,069 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-18 20:43:37,069 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-18 20:43:37,070 INFO Loaded AUDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:37,145 INFO Rule labels [4H]: {'TRENDING_UP': 1833, 'TRENDING_DOWN': 1829, 'RANGING': 4293, 'VOLATILE': 2002}  ambiguous(conf<0.4)=2412 (total=9957)
2026-04-18 20:43:37,147 INFO Loaded AUDUSD/1H split=train: 38804 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:37,177 INFO Loaded AUDUSD/5M split=train: 465551 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:37,460 INFO Loaded AUDUSD/15M split=train: 155205 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:37,585 INFO Loaded AUDUSD/1H split=train: 38804 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:37,678 INFO Loaded AUDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:37,873 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:37,874 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:37,889 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:37,897 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:37,897 INFO Loaded EURGBP/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:37,976 INFO Rule labels [4H]: {'TRENDING_UP': 1702, 'TRENDING_DOWN': 1928, 'RANGING': 4297, 'VOLATILE': 2030}  ambiguous(conf<0.4)=2606 (total=9957)
2026-04-18 20:43:37,977 INFO Loaded EURGBP/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:37,997 INFO Loaded EURGBP/5M split=train: 465522 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:38,274 INFO Loaded EURGBP/15M split=train: 155214 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:38,399 INFO Loaded EURGBP/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:38,491 INFO Loaded EURGBP/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:38,673 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:38,674 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:38,688 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:38,696 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:38,697 INFO Loaded EURJPY/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:38,772 INFO Rule labels [4H]: {'TRENDING_UP': 2037, 'TRENDING_DOWN': 1677, 'RANGING': 4228, 'VOLATILE': 2015}  ambiguous(conf<0.4)=2417 (total=9957)
2026-04-18 20:43:38,774 INFO Loaded EURJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:38,791 INFO Loaded EURJPY/5M split=train: 465569 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:39,066 INFO Loaded EURJPY/15M split=train: 155217 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:39,192 INFO Loaded EURJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:39,284 INFO Loaded EURJPY/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:39,465 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:39,466 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:39,481 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:39,488 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:39,489 INFO Loaded EURUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:39,561 INFO Rule labels [4H]: {'TRENDING_UP': 1509, 'TRENDING_DOWN': 2039, 'RANGING': 4405, 'VOLATILE': 2004}  ambiguous(conf<0.4)=2510 (total=9957)
2026-04-18 20:43:39,563 INFO Loaded EURUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:39,587 INFO Loaded EURUSD/5M split=train: 465631 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:39,882 INFO Loaded EURUSD/15M split=train: 155220 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:40,017 INFO Loaded EURUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:40,114 INFO Loaded EURUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:40,290 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:40,291 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:40,306 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:40,314 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:40,315 INFO Loaded GBPJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:40,387 INFO Rule labels [4H]: {'TRENDING_UP': 1903, 'TRENDING_DOWN': 1665, 'RANGING': 4358, 'VOLATILE': 2032}  ambiguous(conf<0.4)=2532 (total=9958)
2026-04-18 20:43:40,389 INFO Loaded GBPJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:40,406 INFO Loaded GBPJPY/5M split=train: 465412 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:40,685 INFO Loaded GBPJPY/15M split=train: 155212 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:40,815 INFO Loaded GBPJPY/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:40,914 INFO Loaded GBPJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:41,091 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:41,092 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:41,107 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:41,114 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:41,115 INFO Loaded GBPUSD/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:41,185 INFO Rule labels [4H]: {'TRENDING_UP': 1740, 'TRENDING_DOWN': 1940, 'RANGING': 4257, 'VOLATILE': 2021}  ambiguous(conf<0.4)=2464 (total=9958)
2026-04-18 20:43:41,187 INFO Loaded GBPUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:41,204 INFO Loaded GBPUSD/5M split=train: 465597 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:41,481 INFO Loaded GBPUSD/15M split=train: 155220 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:41,612 INFO Loaded GBPUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:41,706 INFO Loaded GBPUSD/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:41,864 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-18 20:43:41,864 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-18 20:43:41,880 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-18 20:43:41,886 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-18 20:43:41,887 INFO Loaded NZDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:41,968 INFO Rule labels [4H]: {'TRENDING_UP': 1770, 'TRENDING_DOWN': 1892, 'RANGING': 4281, 'VOLATILE': 2014}  ambiguous(conf<0.4)=2604 (total=9957)
2026-04-18 20:43:41,970 INFO Loaded NZDUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:41,985 INFO Loaded NZDUSD/5M split=train: 465546 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:42,262 INFO Loaded NZDUSD/15M split=train: 155219 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:42,387 INFO Loaded NZDUSD/1H split=train: 38807 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:42,480 INFO Loaded NZDUSD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:42,656 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:42,656 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:42,671 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:42,678 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:42,679 INFO Loaded USDCAD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:42,754 INFO Rule labels [4H]: {'TRENDING_UP': 1656, 'TRENDING_DOWN': 2068, 'RANGING': 4207, 'VOLATILE': 2026}  ambiguous(conf<0.4)=2378 (total=9957)
2026-04-18 20:43:42,756 INFO Loaded USDCAD/1H split=train: 38808 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:42,773 INFO Loaded USDCAD/5M split=train: 465582 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:43,062 INFO Loaded USDCAD/15M split=train: 155222 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:43,194 INFO Loaded USDCAD/1H split=train: 38808 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:43,291 INFO Loaded USDCAD/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:43,469 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:43,470 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:43,485 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:43,491 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:43,492 INFO Loaded USDCHF/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:43,572 INFO Rule labels [4H]: {'TRENDING_UP': 1953, 'TRENDING_DOWN': 1781, 'RANGING': 4205, 'VOLATILE': 2018}  ambiguous(conf<0.4)=2349 (total=9957)
2026-04-18 20:43:43,574 INFO Loaded USDCHF/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:43,591 INFO Loaded USDCHF/5M split=train: 465478 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:43,874 INFO Loaded USDCHF/15M split=train: 155208 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:44,001 INFO Loaded USDCHF/1H split=train: 38806 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:44,098 INFO Loaded USDCHF/4H split=train: 9957 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:44,276 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:44,277 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:44,292 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:44,300 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 20:43:44,301 INFO Loaded USDJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:44,373 INFO Rule labels [4H]: {'TRENDING_UP': 2196, 'TRENDING_DOWN': 1484, 'RANGING': 4247, 'VOLATILE': 2031}  ambiguous(conf<0.4)=2394 (total=9958)
2026-04-18 20:43:44,375 INFO Loaded USDJPY/1H split=train: 38811 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:44,393 INFO Loaded USDJPY/5M split=train: 465705 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:44,675 INFO Loaded USDJPY/15M split=train: 155241 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:44,807 INFO Loaded USDJPY/1H split=train: 38811 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:44,904 INFO Loaded USDJPY/4H split=train: 9958 bars (2016-01-04 → 2022-08-16)
2026-04-18 20:43:45,191 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-18 20:43:45,192 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-18 20:43:45,210 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-18 20:43:45,220 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-18 20:43:45,221 INFO Loaded XAUUSD/4H split=train: 21467 bars (2009-03-15 → 2022-08-16)
2026-04-18 20:43:45,365 INFO Rule labels [4H]: {'TRENDING_UP': 4484, 'TRENDING_DOWN': 3949, 'RANGING': 8738, 'VOLATILE': 4296}  ambiguous(conf<0.4)=4817 (total=21467)
2026-04-18 20:43:45,368 INFO Loaded XAUUSD/1H split=train: 80734 bars (2009-03-15 → 2022-08-16)
2026-04-18 20:43:45,416 INFO Loaded XAUUSD/5M split=train: 955298 bars (2009-03-15 → 2022-08-16)
2026-04-18 20:43:45,949 INFO Loaded XAUUSD/15M split=train: 319506 bars (2009-03-15 → 2022-08-16)
2026-04-18 20:43:46,158 INFO Loaded XAUUSD/1H split=train: 80734 bars (2009-03-15 → 2022-08-16)
2026-04-18 20:43:46,284 INFO Loaded XAUUSD/4H split=train: 21467 bars (2009-03-15 → 2022-08-16)
2026-04-18 20:43:46,417 INFO train_multi: 44 segments, ~7083076 total bars
2026-04-18 20:43:46,417 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-18 20:43:46,417 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-18 20:48:05,472 INFO train_multi TF=ALL: 7081756 sequences across 44 segments
2026-04-18 20:48:05,473 INFO train_multi TF=ALL: estimated peak RAM = 10368 MB (train=479965 val=120014 n_feat=72 seq_len=30)
2026-04-18 20:48:06,759 INFO train_multi TF=ALL: train=479965 val=120014 (5191 MB tensors)
2026-04-18 20:48:26,892 INFO train_multi TF=ALL epoch 1/50 train=0.5955 val=0.6184
2026-04-18 20:48:26,896 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-18 20:48:26,896 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-18 20:48:26,896 INFO train_multi TF=ALL: new best val=0.6184 — saved
2026-04-18 20:48:41,326 INFO train_multi TF=ALL epoch 2/50 train=0.5950 val=0.6190
2026-04-18 20:48:55,673 INFO train_multi TF=ALL epoch 3/50 train=0.5948 val=0.6188
2026-04-18 20:49:09,957 INFO train_multi TF=ALL epoch 4/50 train=0.5945 val=0.6190
2026-04-18 20:49:24,308 INFO train_multi TF=ALL epoch 5/50 train=0.5948 val=0.6187
2026-04-18 20:49:38,574 INFO train_multi TF=ALL epoch 6/50 train=0.5943 val=0.6185
2026-04-18 20:49:38,574 INFO train_multi TF=ALL early stop at epoch 6
2026-04-18 20:49:38,703 INFO === VectorStore: building similarity indices ===
2026-04-18 20:49:38,704 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-18 20:49:38,704 INFO Retrain complete.
2026-04-18 20:49:40,577 INFO Model gru: SUCCESS
2026-04-18 20:49:40,577 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-18 20:49:40,577 WARNING   [MISSING] regime_classifier → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_classifier.pkl
2026-04-18 20:49:40,578 INFO   [OK] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-18 20:49:40,578 WARNING   [MISSING] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-18 20:49:40,578 WARNING Missing weights: ['regime_classifier', 'rl_ppo'] — run retrain_incremental.py for each
2026-04-18 20:49:40,579 INFO Saved 18 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime
  START Step 6 - Backtest
2026-04-18 20:49:41,105 INFO === STEP 6: BACKTEST + REINFORCED TRAINING (3 rounds) ===
2026-04-18 20:49:41,264 INFO Backtest date range: 2021-01-01 → 2024-03-05 (reinforcement loop, test set protected; set BT_START_FLOOR env to change)
2026-04-18 20:49:41,266 INFO Cleared existing journal for fresh reinforced training run
2026-04-18 20:49:41,266 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-18 20:49:41,267 INFO Round 1 — running backtest: 2021-01-01 → 2024-03-05 (ml_trader, shared ML cache)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260418_204941.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)             2571  45.4%   2.08 1549.2% 45.4% 13.0%   2.8%    3.77

Calibration Summary:
  all          [OK] Calibration OK — p_win correlates with actual win rate.
  ml_trader    [OK] Calibration OK — p_win correlates with actual win rate.
2026-04-18 20:55:50,316 INFO Round 1 backtest — 2571 trades | avg WR=45.4% | avg PF=2.08 | avg Sharpe=3.77
2026-04-18 20:55:50,316 INFO   ml_trader: 2571 trades | WR=45.4% | PF=2.08 | Return=1549.2% | DD=2.8% | Sharpe=3.77
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 2571
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2571 rows)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  AUDUSD        321 trades   292 days   1.10/day
  EURGBP        161 trades   139 days   1.16/day
  EURJPY        116 trades    96 days   1.21/day
  EURUSD         44 trades    44 days   1.00/day
  GBPJPY         97 trades    79 days   1.23/day
  GBPUSD        717 trades   682 days   1.05/day
  NZDUSD        192 trades   165 days   1.16/day
  USDCAD        218 trades   191 days   1.14/day
  USDCHF        194 trades   166 days   1.17/day
  USDJPY        422 trades   392 days   1.08/day
  XAUUSD         89 trades    69 days   1.29/day
  ✓  All symbols within normal range.
2026-04-18 20:55:51,402 INFO Round 1: wrote 2571 journal entries (total in file: 2571)
2026-04-18 20:55:51,404 INFO Round 1 — retraining regime...
2026-04-18 20:56:33,897 INFO Retrain regime: OK
2026-04-18 20:56:33,915 INFO Round 1 — retraining quality...
2026-04-18 20:56:39,506 INFO Retrain quality: OK
2026-04-18 20:56:39,523 INFO Round 1 — retraining rl...
2026-04-18 20:58:01,246 INFO Retrain rl: OK
2026-04-18 20:58:01,263 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-18 20:58:01,263 INFO Round 2 — running backtest: 2021-01-01 → 2024-03-05 (ml_trader, shared ML cache)
2026-04-18 21:04:06,752 INFO Round 2 backtest — 2610 trades | avg WR=44.8% | avg PF=2.04 | avg Sharpe=3.70
2026-04-18 21:04:06,752 INFO   ml_trader: 2610 trades | WR=44.8% | PF=2.04 | Return=1458.3% | DD=3.6% | Sharpe=3.70
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 2610
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2610 rows)
2026-04-18 21:04:07,845 INFO Round 2: wrote 2610 journal entries (total in file: 5181)
2026-04-18 21:04:07,847 INFO Round 2 — retraining regime...
2026-04-18 21:04:50,692 INFO Retrain regime: OK
2026-04-18 21:04:50,710 INFO Round 2 — retraining quality...
2026-04-18 21:04:57,924 INFO Retrain quality: OK
2026-04-18 21:04:57,941 INFO Round 2 — retraining rl...
2026-04-18 21:07:25,768 INFO Retrain rl: OK
2026-04-18 21:07:25,785 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-04-18 21:07:25,786 INFO Round 3 — running backtest: 2021-01-01 → 2024-03-05 (ml_trader, shared ML cache)
2026-04-18 21:13:32,132 INFO Round 3 backtest — 2586 trades | avg WR=45.3% | avg PF=2.05 | avg Sharpe=3.71
2026-04-18 21:13:32,132 INFO   ml_trader: 2586 trades | WR=45.3% | PF=2.05 | Return=1490.8% | DD=2.8% | Sharpe=3.71
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 2586
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2586 rows)
2026-04-18 21:13:33,205 INFO Round 3: wrote 2586 journal entries (total in file: 7767)
2026-04-18 21:13:33,207 INFO Round 3 (final): retraining after last backtest...
2026-04-18 21:13:33,207 INFO Round 3 — retraining regime...
2026-04-18 21:14:18,546 INFO Retrain regime: OK
2026-04-18 21:14:18,563 INFO Round 3 — retraining quality...
2026-04-18 21:14:28,106 INFO Retrain quality: OK
2026-04-18 21:14:28,122 INFO Round 3 — retraining rl...
2026-04-18 21:18:10,416 INFO Retrain rl: OK
2026-04-18 21:18:10,433 INFO Improvement round 1 → 3: WR -0.1% | PF -0.038 | Sharpe -0.057
2026-04-18 21:18:10,581 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-18 21:18:10,589 INFO Journal entries: 7767
2026-04-18 21:18:10,589 INFO --- Training quality ---
2026-04-18 21:18:10,590 INFO Running retrain --model quality
  DONE  Step 6 - Backtest
  START Step 7b - Quality+RL
2026-04-18 21:18:10,681 INFO retrain environment: KAGGLE
2026-04-18 21:18:12,357 INFO Device: CUDA (2 GPU(s))
2026-04-18 21:18:12,368 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-18 21:18:12,369 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-18 21:18:12,370 INFO === QualityScorer retrain ===
2026-04-18 21:18:12,504 INFO NumExpr defaulting to 4 threads.
2026-04-18 21:18:12,698 INFO QualityScorer: CUDA available — using GPU
2026-04-18 21:18:12,914 INFO QualityScorer loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (device=cuda)
2026-04-18 21:18:13,143 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/quality_scorer.pkl_20260418_211813
2026-04-18 21:18:13,353 INFO QualityScorer: 5264 samples, EV stats={'mean': -0.35576558113098145, 'std': 1.354196310043335, 'n_pos': 1003, 'n_neg': 4261}, device=cuda
2026-04-18 21:18:13,353 INFO QualityScorer: normalised win labels by median_win=2.000 — EV range now [-1, +3]
2026-04-18 21:18:13,353 INFO QualityScorer: warm start from existing weights
2026-04-18 21:18:13,354 INFO QualityScorer: pos_weight=4.15 (n_pos=817 n_neg=3394)
2026-04-18 21:18:14,967 INFO Quality epoch   1/100 — va_huber=0.8380
2026-04-18 21:18:15,071 INFO Quality epoch   2/100 — va_huber=0.8394
2026-04-18 21:18:15,174 INFO Quality epoch   3/100 — va_huber=0.8360
2026-04-18 21:18:15,277 INFO Quality epoch   4/100 — va_huber=0.8386
2026-04-18 21:18:15,379 INFO Quality epoch   5/100 — va_huber=0.8379
2026-04-18 21:18:15,993 INFO Quality epoch  11/100 — va_huber=0.8351
2026-04-18 21:18:17,099 INFO Quality epoch  21/100 — va_huber=0.8335
2026-04-18 21:18:18,098 INFO Quality epoch  31/100 — va_huber=0.8327
2026-04-18 21:18:19,095 INFO Quality epoch  41/100 — va_huber=0.8329
2026-04-18 21:18:19,682 INFO Quality early stop at epoch 47
2026-04-18 21:18:19,701 INFO QualityScorer EV model: MAE=0.933 dir_acc=0.615 n_val=1053
2026-04-18 21:18:19,704 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-18 21:18:19,770 INFO Retrain complete.
2026-04-18 21:18:20,579 INFO Model quality: SUCCESS
2026-04-18 21:18:20,580 INFO --- Training rl ---
2026-04-18 21:18:20,580 INFO Running retrain --model rl
2026-04-18 21:18:20,671 INFO retrain environment: KAGGLE
2026-04-18 21:18:22,348 INFO Device: CUDA (2 GPU(s))
2026-04-18 21:18:22,360 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-18 21:18:22,360 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-18 21:18:22,361 INFO === RLAgent (PPO) retrain ===
2026-04-18 21:18:22,364 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260418_211822
2026-04-18 21:18:23.009680: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1776547103.032094  100691 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1776547103.039680  100691 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1776547103.059327  100691 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1776547103.059378  100691 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1776547103.059385  100691 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1776547103.059389  100691 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-18 21:18:27,370 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-18 21:18:29,027 ERROR RLAgent.load failed: [Errno 21] Is a directory: '/kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model'
/usr/local/lib/python3.12/dist-packages/stable_baselines3/common/on_policy_algorithm.py:150: UserWarning: You are trying to run PPO on the GPU, but it is primarily intended to run on the CPU when not using a CNN policy (you are using ActorCriticPolicy which should be a MlpPolicy). See https://github.com/DLR-RM/stable-baselines3/issues/1245 for more info. You can pass `device='cpu'` or `export CUDA_VISIBLE_DEVICES=` to force using the CPU.Note: The model will train, but the GPU utilization will be poor and the training might take longer than on CPU.
  warnings.warn(
2026-04-18 21:18:30,663 INFO RLAgent: cold start — building new PPO policy
2026-04-18 21:22:02,818 INFO RLAgent: retrain complete, 7767 episodes
2026-04-18 21:22:02,818 INFO Retrain complete.
  DONE  Step 7b - Quality+RL

=== Pipeline complete ===
Weights saved to: /kaggle/working/outputs/processed_data/weights
Logs saved to: /kaggle/working/outputs/processed_data/logs