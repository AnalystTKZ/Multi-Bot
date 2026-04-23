
All scripts and inputs verified.

=== Phase 0-5: Data preparation ===
  SKIP  Step 0 - Resample
  SKIP  Step 1 - Inventory
  SKIP  Step 2 - Cleaning
  SKIP  Step 3 - Alignment
  SKIP  Step 4 - Features
  START Step 5 - Split
2026-04-19 15:28:37,113 INFO Loading feature-engineered data...
2026-04-19 15:28:37,591 INFO Loaded 221743 rows, 202 features
2026-04-19 15:28:37,592 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-19 15:28:37,592 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-19 15:28:37,592 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-19 15:28:37,593 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-19 15:28:37,593 INFO No leakage confirmed: train < val < test timestamps

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
2026-04-19 15:28:39,938 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-19 15:28:39,938 INFO --- Training regime ---
2026-04-19 15:28:39,938 INFO Running retrain --model regime
2026-04-19 15:28:40,136 INFO retrain environment: KAGGLE
2026-04-19 15:28:41,936 INFO Device: CUDA (2 GPU(s))
2026-04-19 15:28:41,948 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 15:28:41,948 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 15:28:41,949 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-19 15:28:42,101 INFO NumExpr defaulting to 4 threads.
2026-04-19 15:28:42,311 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-19 15:28:42,312 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 15:28:42,312 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 15:28:42,525 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-19 15:28:42,528 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:42,610 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:42,690 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:42,768 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:42,842 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:42,919 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:42,992 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:43,067 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:43,140 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:43,211 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:43,303 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:28:43,366 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-19 15:28:43,384 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:43,386 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:43,403 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:43,404 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:43,421 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:43,423 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:43,439 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:43,441 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:43,458 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:43,462 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:43,479 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:43,483 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:43,497 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:43,500 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:43,515 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:43,519 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:43,533 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:43,537 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:43,552 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:43,556 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:28:43,577 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:28:43,584 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:28:44,384 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-19 15:29:07,431 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-19 15:29:07,433 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias)
2026-04-19 15:29:07,433 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-19 15:29:18,057 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-19 15:29:18,064 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias)
2026-04-19 15:29:18,064 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-19 15:29:25,675 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-19 15:29:25,676 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias)
2026-04-19 15:29:25,676 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-19 15:30:39,704 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-19 15:30:39,707 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour)
2026-04-19 15:30:39,707 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-19 15:31:10,452 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-19 15:31:10,453 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour)
2026-04-19 15:31:10,454 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-19 15:31:32,113 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-19 15:31:32,116 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour)
2026-04-19 15:31:32,234 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-19 15:31:32,236 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:31:32,237 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:31:32,239 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:31:32,240 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:31:32,241 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:31:32,242 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:31:32,242 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:31:32,243 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:31:32,244 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:31:32,245 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:31:32,246 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:31:32,377 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:32,422 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:32,423 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:32,423 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:32,433 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:32,433 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:31:35,184 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-19 15:31:35,185 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-19 15:31:35,375 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:35,409 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:35,410 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:35,411 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:35,420 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:35,421 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:31:38,097 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-19 15:31:38,098 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-19 15:31:38,278 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:38,314 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:38,315 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:38,315 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:38,325 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:38,326 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:31:41,048 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-19 15:31:41,049 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-19 15:31:41,219 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:41,254 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:41,255 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:41,255 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:41,264 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:41,265 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:31:43,946 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-19 15:31:43,948 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-19 15:31:44,125 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:44,161 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:44,162 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:44,162 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:44,171 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:44,172 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:31:46,881 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-19 15:31:46,882 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-19 15:31:47,065 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:47,100 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:47,101 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:47,102 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:47,111 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:47,112 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:31:49,855 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-19 15:31:49,856 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-19 15:31:50,011 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:31:50,041 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:31:50,041 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:31:50,042 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:31:50,051 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:31:50,052 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:31:52,772 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-19 15:31:52,773 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-19 15:31:52,954 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:52,989 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:52,990 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:52,991 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:53,000 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:53,001 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:31:55,661 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-19 15:31:55,662 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-19 15:31:55,831 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:55,865 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:55,866 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:55,867 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:55,875 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:55,876 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:31:58,541 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-19 15:31:58,542 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-19 15:31:58,719 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:58,755 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:58,755 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:58,756 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:58,765 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:31:58,766 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:32:01,514 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-19 15:32:01,516 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-19 15:32:01,804 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:32:01,868 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:32:01,870 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:32:01,870 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:32:01,881 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:32:01,883 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:32:08,165 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-19 15:32:08,167 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-19 15:32:08,366 INFO RegimeClassifier[mode=htf_bias]: 103290 samples, classes={'BIAS_UP': 18622, 'BIAS_DOWN': 18286, 'BIAS_NEUTRAL': 66382}, device=cuda
2026-04-19 15:32:08,367 INFO RegimeClassifier: sample weights — mean=0.360  ambiguous(<0.4)=69.5%
2026-04-19 15:32:08,557 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-04-19 15:32:08,558 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-19 15:32:10,930 INFO Regime epoch  1/50 — tr=0.7063 va=1.2997 acc=0.516 per_class={'BIAS_UP': 0.334, 'BIAS_DOWN': 0.042, 'BIAS_NEUTRAL': 0.701}
2026-04-19 15:32:11,124 INFO Regime epoch  2/50 — tr=0.6912 va=1.2799 acc=0.524
2026-04-19 15:32:11,294 INFO Regime epoch  3/50 — tr=0.6676 va=1.2406 acc=0.511
2026-04-19 15:32:11,465 INFO Regime epoch  4/50 — tr=0.6366 va=1.1744 acc=0.485
2026-04-19 15:32:11,649 INFO Regime epoch  5/50 — tr=0.5980 va=1.0984 acc=0.483 per_class={'BIAS_UP': 0.808, 'BIAS_DOWN': 0.922, 'BIAS_NEUTRAL': 0.255}
2026-04-19 15:32:11,823 INFO Regime epoch  6/50 — tr=0.5636 va=1.0285 acc=0.490
2026-04-19 15:32:12,003 INFO Regime epoch  7/50 — tr=0.5373 va=0.9743 acc=0.497
2026-04-19 15:32:12,183 INFO Regime epoch  8/50 — tr=0.5195 va=0.9370 acc=0.508
2026-04-19 15:32:12,351 INFO Regime epoch  9/50 — tr=0.5098 va=0.9158 acc=0.518
2026-04-19 15:32:12,536 INFO Regime epoch 10/50 — tr=0.5028 va=0.9066 acc=0.530 per_class={'BIAS_UP': 0.987, 'BIAS_DOWN': 0.996, 'BIAS_NEUTRAL': 0.248}
2026-04-19 15:32:12,709 INFO Regime epoch 11/50 — tr=0.4969 va=0.9006 acc=0.532
2026-04-19 15:32:12,877 INFO Regime epoch 12/50 — tr=0.4924 va=0.8934 acc=0.536
2026-04-19 15:32:13,052 INFO Regime epoch 13/50 — tr=0.4884 va=0.8951 acc=0.538
2026-04-19 15:32:13,226 INFO Regime epoch 14/50 — tr=0.4848 va=0.8926 acc=0.543
2026-04-19 15:32:13,419 INFO Regime epoch 15/50 — tr=0.4827 va=0.8921 acc=0.545 per_class={'BIAS_UP': 0.996, 'BIAS_DOWN': 0.997, 'BIAS_NEUTRAL': 0.269}
2026-04-19 15:32:13,594 INFO Regime epoch 16/50 — tr=0.4800 va=0.8902 acc=0.548
2026-04-19 15:32:13,764 INFO Regime epoch 17/50 — tr=0.4782 va=0.8878 acc=0.553
2026-04-19 15:32:13,936 INFO Regime epoch 18/50 — tr=0.4764 va=0.8855 acc=0.554
2026-04-19 15:32:14,114 INFO Regime epoch 19/50 — tr=0.4749 va=0.8853 acc=0.557
2026-04-19 15:32:14,310 INFO Regime epoch 20/50 — tr=0.4740 va=0.8854 acc=0.560 per_class={'BIAS_UP': 0.997, 'BIAS_DOWN': 0.996, 'BIAS_NEUTRAL': 0.293}
2026-04-19 15:32:14,494 INFO Regime epoch 21/50 — tr=0.4726 va=0.8837 acc=0.562
2026-04-19 15:32:14,675 INFO Regime epoch 22/50 — tr=0.4715 va=0.8851 acc=0.563
2026-04-19 15:32:14,858 INFO Regime epoch 23/50 — tr=0.4705 va=0.8823 acc=0.562
2026-04-19 15:32:15,025 INFO Regime epoch 24/50 — tr=0.4701 va=0.8817 acc=0.567
2026-04-19 15:32:15,214 INFO Regime epoch 25/50 — tr=0.4690 va=0.8820 acc=0.570 per_class={'BIAS_UP': 0.997, 'BIAS_DOWN': 0.997, 'BIAS_NEUTRAL': 0.308}
2026-04-19 15:32:15,387 INFO Regime epoch 26/50 — tr=0.4684 va=0.8821 acc=0.573
2026-04-19 15:32:15,560 INFO Regime epoch 27/50 — tr=0.4683 va=0.8807 acc=0.571
2026-04-19 15:32:15,738 INFO Regime epoch 28/50 — tr=0.4678 va=0.8794 acc=0.572
2026-04-19 15:32:15,915 INFO Regime epoch 29/50 — tr=0.4676 va=0.8814 acc=0.579
2026-04-19 15:32:16,113 INFO Regime epoch 30/50 — tr=0.4666 va=0.8792 acc=0.575 per_class={'BIAS_UP': 0.998, 'BIAS_DOWN': 0.997, 'BIAS_NEUTRAL': 0.317}
2026-04-19 15:32:16,292 INFO Regime epoch 31/50 — tr=0.4665 va=0.8802 acc=0.579
2026-04-19 15:32:16,459 INFO Regime epoch 32/50 — tr=0.4663 va=0.8802 acc=0.579
2026-04-19 15:32:16,630 INFO Regime epoch 33/50 — tr=0.4658 va=0.8795 acc=0.580
2026-04-19 15:32:16,798 INFO Regime epoch 34/50 — tr=0.4656 va=0.8785 acc=0.577
2026-04-19 15:32:16,984 INFO Regime epoch 35/50 — tr=0.4656 va=0.8786 acc=0.577 per_class={'BIAS_UP': 0.998, 'BIAS_DOWN': 0.997, 'BIAS_NEUTRAL': 0.32}
2026-04-19 15:32:17,153 INFO Regime epoch 36/50 — tr=0.4652 va=0.8796 acc=0.580
2026-04-19 15:32:17,320 INFO Regime epoch 37/50 — tr=0.4655 va=0.8780 acc=0.579
2026-04-19 15:32:17,490 INFO Regime epoch 38/50 — tr=0.4647 va=0.8784 acc=0.579
2026-04-19 15:32:17,661 INFO Regime epoch 39/50 — tr=0.4650 va=0.8777 acc=0.578
2026-04-19 15:32:17,852 INFO Regime epoch 40/50 — tr=0.4647 va=0.8786 acc=0.583 per_class={'BIAS_UP': 0.998, 'BIAS_DOWN': 0.997, 'BIAS_NEUTRAL': 0.33}
2026-04-19 15:32:18,018 INFO Regime epoch 41/50 — tr=0.4650 va=0.8791 acc=0.583
2026-04-19 15:32:18,185 INFO Regime epoch 42/50 — tr=0.4649 va=0.8778 acc=0.584
2026-04-19 15:32:18,357 INFO Regime epoch 43/50 — tr=0.4645 va=0.8778 acc=0.583
2026-04-19 15:32:18,525 INFO Regime epoch 44/50 — tr=0.4645 va=0.8774 acc=0.582
2026-04-19 15:32:18,709 INFO Regime epoch 45/50 — tr=0.4645 va=0.8764 acc=0.582 per_class={'BIAS_UP': 0.998, 'BIAS_DOWN': 0.997, 'BIAS_NEUTRAL': 0.328}
2026-04-19 15:32:18,883 INFO Regime epoch 46/50 — tr=0.4644 va=0.8784 acc=0.583
2026-04-19 15:32:19,053 INFO Regime epoch 47/50 — tr=0.4644 va=0.8771 acc=0.581
2026-04-19 15:32:19,235 INFO Regime epoch 48/50 — tr=0.4642 va=0.8782 acc=0.583
2026-04-19 15:32:19,412 INFO Regime epoch 49/50 — tr=0.4646 va=0.8775 acc=0.584
2026-04-19 15:32:19,612 INFO Regime epoch 50/50 — tr=0.4647 va=0.8778 acc=0.584 per_class={'BIAS_UP': 0.998, 'BIAS_DOWN': 0.997, 'BIAS_NEUTRAL': 0.331}
2026-04-19 15:32:19,626 WARNING RegimeClassifier accuracy 0.58 < 0.65 threshold
2026-04-19 15:32:19,629 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-19 15:32:19,629 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-19 15:32:19,756 INFO Regime HTF complete: acc=0.582, n=103290
2026-04-19 15:32:19,758 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:32:19,915 INFO Rule labels LTF_BEHAVIOUR [4H]: {'TRENDING': 8970, 'RANGING': 4592, 'CONSOLIDATING': 2247, 'VOLATILE': 4008}  ambiguous=4539 (total=19817)  short_runs_zeroed=570
2026-04-19 15:32:19,917 INFO Regime[4H] persistence (avg bars/run) on XAUUSD 4H:
{0: 12.121621621621621, 1: 6.4858757062146895, 2: 5.575682382133995, 3: 10.329896907216495}
2026-04-19 15:32:19,919 INFO Regime[4H] return separation on XAUUSD 4H:
{0: 5.9502411776840314e-05, 1: 5.664222849552333e-05, 2: -6.138442068010657e-05, 3: 4.239843820627722e-05}
2026-04-19 15:32:19,919 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-19 15:32:19,921 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:32:19,923 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:32:19,924 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:32:19,926 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:32:19,927 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:32:19,929 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:32:19,930 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:32:19,932 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:32:19,934 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:32:19,935 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:32:19,938 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:32:19,949 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:19,951 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:19,952 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:19,952 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:19,952 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:19,954 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:32:29,827 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-19 15:32:29,830 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-19 15:32:29,968 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:29,970 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:29,972 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:29,972 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:29,973 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:29,975 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:32:39,755 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-19 15:32:39,758 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-19 15:32:39,898 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:39,900 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:39,901 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:39,902 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:39,902 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:39,904 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:32:49,726 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-19 15:32:49,729 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-19 15:32:49,862 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:49,866 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:49,867 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:49,867 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:49,868 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:49,869 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:32:59,668 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-19 15:32:59,671 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-19 15:32:59,818 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:59,820 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:59,821 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:59,822 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:59,822 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:32:59,824 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:33:09,744 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-19 15:33:09,747 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-19 15:33:09,884 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:33:09,887 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:33:09,888 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:33:09,888 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:33:09,889 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:33:09,891 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:33:19,700 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-19 15:33:19,703 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-19 15:33:19,847 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:33:19,849 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:33:19,849 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:33:19,850 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:33:19,850 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:33:19,852 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:33:29,432 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-19 15:33:29,435 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-19 15:33:29,569 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:33:29,572 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:33:29,572 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:33:29,573 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:33:29,573 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:33:29,575 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:33:39,169 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-19 15:33:39,172 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-19 15:33:39,313 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:33:39,315 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:33:39,316 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:33:39,317 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:33:39,317 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:33:39,319 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:33:48,983 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-19 15:33:48,986 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-19 15:33:49,136 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:33:49,138 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:33:49,139 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:33:49,139 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:33:49,140 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:33:49,142 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:33:58,824 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-19 15:33:58,827 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-19 15:33:58,979 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:33:58,983 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:33:58,984 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:33:58,985 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:33:58,985 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:33:58,988 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:34:21,180 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-19 15:34:21,186 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-19 15:34:21,533 INFO RegimeClassifier[mode=ltf_behaviour]: 401471 samples, classes={'TRENDING': 160094, 'RANGING': 105286, 'CONSOLIDATING': 53524, 'VOLATILE': 82567}, device=cuda
2026-04-19 15:34:21,534 INFO RegimeClassifier: sample weights — mean=0.505  ambiguous(<0.4)=33.0%
2026-04-19 15:34:21,537 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-04-19 15:34:21,537 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-19 15:34:22,233 INFO Regime epoch  1/50 — tr=0.9684 va=1.6841 acc=0.315 per_class={'TRENDING': 0.217, 'RANGING': 0.377, 'CONSOLIDATING': 0.2, 'VOLATILE': 0.508}
2026-04-19 15:34:22,869 INFO Regime epoch  2/50 — tr=0.9358 va=1.6208 acc=0.399
2026-04-19 15:34:23,520 INFO Regime epoch  3/50 — tr=0.8885 va=1.5627 acc=0.428
2026-04-19 15:34:24,165 INFO Regime epoch  4/50 — tr=0.8425 va=1.5091 acc=0.442
2026-04-19 15:34:24,864 INFO Regime epoch  5/50 — tr=0.8083 va=1.4839 acc=0.450 per_class={'TRENDING': 0.36, 'RANGING': 0.062, 'CONSOLIDATING': 0.789, 'VOLATILE': 0.906}
2026-04-19 15:34:25,517 INFO Regime epoch  6/50 — tr=0.7854 va=1.4743 acc=0.457
2026-04-19 15:34:26,147 INFO Regime epoch  7/50 — tr=0.7710 va=1.4745 acc=0.460
2026-04-19 15:34:26,772 INFO Regime epoch  8/50 — tr=0.7608 va=1.4723 acc=0.470
2026-04-19 15:34:27,406 INFO Regime epoch  9/50 — tr=0.7530 va=1.4640 acc=0.486
2026-04-19 15:34:28,072 INFO Regime epoch 10/50 — tr=0.7471 va=1.4516 acc=0.503 per_class={'TRENDING': 0.477, 'RANGING': 0.05, 'CONSOLIDATING': 0.847, 'VOLATILE': 0.916}
2026-04-19 15:34:28,725 INFO Regime epoch 11/50 — tr=0.7415 va=1.4376 acc=0.521
2026-04-19 15:34:29,387 INFO Regime epoch 12/50 — tr=0.7368 va=1.4253 acc=0.537
2026-04-19 15:34:30,030 INFO Regime epoch 13/50 — tr=0.7331 va=1.4103 acc=0.546
2026-04-19 15:34:30,680 INFO Regime epoch 14/50 — tr=0.7294 va=1.3990 acc=0.557
2026-04-19 15:34:31,409 INFO Regime epoch 15/50 — tr=0.7262 va=1.3901 acc=0.566 per_class={'TRENDING': 0.617, 'RANGING': 0.059, 'CONSOLIDATING': 0.882, 'VOLATILE': 0.914}
2026-04-19 15:34:32,058 INFO Regime epoch 16/50 — tr=0.7232 va=1.3834 acc=0.571
2026-04-19 15:34:32,683 INFO Regime epoch 17/50 — tr=0.7210 va=1.3780 acc=0.576
2026-04-19 15:34:33,326 INFO Regime epoch 18/50 — tr=0.7187 va=1.3711 acc=0.581
2026-04-19 15:34:33,986 INFO Regime epoch 19/50 — tr=0.7170 va=1.3667 acc=0.583
2026-04-19 15:34:34,665 INFO Regime epoch 20/50 — tr=0.7154 va=1.3639 acc=0.587 per_class={'TRENDING': 0.653, 'RANGING': 0.064, 'CONSOLIDATING': 0.918, 'VOLATILE': 0.916}
2026-04-19 15:34:35,315 INFO Regime epoch 21/50 — tr=0.7137 va=1.3577 acc=0.589
2026-04-19 15:34:35,951 INFO Regime epoch 22/50 — tr=0.7124 va=1.3563 acc=0.592
2026-04-19 15:34:36,594 INFO Regime epoch 23/50 — tr=0.7112 va=1.3546 acc=0.595
2026-04-19 15:34:37,232 INFO Regime epoch 24/50 — tr=0.7106 va=1.3526 acc=0.597
2026-04-19 15:34:37,943 INFO Regime epoch 25/50 — tr=0.7096 va=1.3486 acc=0.598 per_class={'TRENDING': 0.671, 'RANGING': 0.066, 'CONSOLIDATING': 0.951, 'VOLATILE': 0.912}
2026-04-19 15:34:38,612 INFO Regime epoch 26/50 — tr=0.7089 va=1.3476 acc=0.599
2026-04-19 15:34:39,248 INFO Regime epoch 27/50 — tr=0.7080 va=1.3460 acc=0.598
2026-04-19 15:34:39,872 INFO Regime epoch 28/50 — tr=0.7074 va=1.3428 acc=0.601
2026-04-19 15:34:40,508 INFO Regime epoch 29/50 — tr=0.7066 va=1.3435 acc=0.601
2026-04-19 15:34:41,238 INFO Regime epoch 30/50 — tr=0.7063 va=1.3420 acc=0.603 per_class={'TRENDING': 0.676, 'RANGING': 0.073, 'CONSOLIDATING': 0.953, 'VOLATILE': 0.917}
2026-04-19 15:34:41,864 INFO Regime epoch 31/50 — tr=0.7063 va=1.3410 acc=0.603
2026-04-19 15:34:42,498 INFO Regime epoch 32/50 — tr=0.7057 va=1.3426 acc=0.604
2026-04-19 15:34:43,119 INFO Regime epoch 33/50 — tr=0.7052 va=1.3397 acc=0.603
2026-04-19 15:34:43,738 INFO Regime epoch 34/50 — tr=0.7049 va=1.3385 acc=0.606
2026-04-19 15:34:44,432 INFO Regime epoch 35/50 — tr=0.7048 va=1.3391 acc=0.604 per_class={'TRENDING': 0.677, 'RANGING': 0.073, 'CONSOLIDATING': 0.953, 'VOLATILE': 0.919}
2026-04-19 15:34:45,087 INFO Regime epoch 36/50 — tr=0.7044 va=1.3379 acc=0.606
2026-04-19 15:34:45,716 INFO Regime epoch 37/50 — tr=0.7040 va=1.3332 acc=0.609
2026-04-19 15:34:46,396 INFO Regime epoch 38/50 — tr=0.7041 va=1.3351 acc=0.608
2026-04-19 15:34:47,038 INFO Regime epoch 39/50 — tr=0.7037 va=1.3354 acc=0.608
2026-04-19 15:34:47,732 INFO Regime epoch 40/50 — tr=0.7037 va=1.3360 acc=0.607 per_class={'TRENDING': 0.686, 'RANGING': 0.072, 'CONSOLIDATING': 0.954, 'VOLATILE': 0.917}
2026-04-19 15:34:48,358 INFO Regime epoch 41/50 — tr=0.7037 va=1.3330 acc=0.610
2026-04-19 15:34:49,011 INFO Regime epoch 42/50 — tr=0.7033 va=1.3366 acc=0.609
2026-04-19 15:34:49,653 INFO Regime epoch 43/50 — tr=0.7034 va=1.3321 acc=0.610
2026-04-19 15:34:50,364 INFO Regime epoch 44/50 — tr=0.7036 va=1.3337 acc=0.608
2026-04-19 15:34:51,083 INFO Regime epoch 45/50 — tr=0.7033 va=1.3335 acc=0.608 per_class={'TRENDING': 0.69, 'RANGING': 0.071, 'CONSOLIDATING': 0.959, 'VOLATILE': 0.911}
2026-04-19 15:34:51,721 INFO Regime epoch 46/50 — tr=0.7034 va=1.3328 acc=0.611
2026-04-19 15:34:52,370 INFO Regime epoch 47/50 — tr=0.7032 va=1.3343 acc=0.608
2026-04-19 15:34:53,004 INFO Regime epoch 48/50 — tr=0.7032 va=1.3343 acc=0.608
2026-04-19 15:34:53,632 INFO Regime epoch 49/50 — tr=0.7032 va=1.3351 acc=0.609
2026-04-19 15:34:54,308 INFO Regime epoch 50/50 — tr=0.7032 va=1.3360 acc=0.607 per_class={'TRENDING': 0.685, 'RANGING': 0.074, 'CONSOLIDATING': 0.953, 'VOLATILE': 0.919}
2026-04-19 15:34:54,352 WARNING RegimeClassifier accuracy 0.61 < 0.65 threshold
2026-04-19 15:34:54,355 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-19 15:34:54,355 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-19 15:34:54,488 INFO Regime LTF complete: acc=0.610, n=401471
2026-04-19 15:34:54,491 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:34:54,946 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-19 15:34:54,951 INFO Regime[1H] persistence (avg bars/run) on XAUUSD 1H:
{0: 9.355291913830783, 1: 6.096303199751476, 2: 5.598885793871866, 3: 6.771351107094442}
2026-04-19 15:34:54,954 INFO Regime[1H] return separation on XAUUSD 1H:
{0: 2.149062259083323e-05, 1: -3.254493608872846e-06, 2: -3.039756158146958e-06, 3: 1.8102764458593193e-05}
2026-04-19 15:34:54,966 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-19 15:34:54,966 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 15:34:54,966 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 15:34:54,966 INFO === VectorStore: building similarity indices ===
2026-04-19 15:34:54,967 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-19 15:34:54,967 INFO Retrain complete.
2026-04-19 15:34:56,070 INFO Model regime: SUCCESS
2026-04-19 15:34:56,071 INFO --- Training gru ---
2026-04-19 15:34:56,071 INFO Running retrain --model gru
2026-04-19 15:34:56,893 INFO retrain environment: KAGGLE
2026-04-19 15:34:58,573 INFO Device: CUDA (2 GPU(s))
2026-04-19 15:34:58,584 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 15:34:58,584 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 15:34:58,585 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-19 15:34:58,721 INFO NumExpr defaulting to 4 threads.
2026-04-19 15:34:58,914 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-19 15:34:58,914 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 15:34:58,914 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 15:34:58,917 WARNING GRULSTMPredictor: stale weights detected (regime_4h feature contract changed: added=['autocorr_lag1', 'efficiency_ratio', 'hurst_proxy']; count 31→34) — deleting /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt so retrain starts fresh
2026-04-19 15:34:58,918 INFO Deleted stale weights (regime_4h feature contract changed: added=['autocorr_lag1', 'efficiency_ratio', 'hurst_proxy']; count 31→34): /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 15:34:59,138 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-19 15:34:59,140 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:34:59,220 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:34:59,298 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:34:59,368 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:34:59,441 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:34:59,511 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:34:59,578 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:34:59,650 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:34:59,720 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:34:59,792 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:34:59,881 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:34:59,940 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-19 15:34:59,940 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260419_153459
2026-04-19 15:34:59,941 INFO GRU weights stale (regime_4h feature contract changed: added=['autocorr_lag1', 'efficiency_ratio', 'hurst_proxy']; count 31→34) — deleting for full retrain
2026-04-19 15:35:00,060 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:00,060 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:00,077 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:00,084 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:00,086 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-19 15:35:00,086 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 15:35:00,086 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 15:35:00,087 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:00,160 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-19 15:35:00,161 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:00,389 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-19 15:35:00,419 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:00,714 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:00,841 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:00,956 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:01,167 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:01,168 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:01,184 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:01,196 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:01,197 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:01,274 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-19 15:35:01,275 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:01,504 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-19 15:35:01,520 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:01,783 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:01,908 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:02,008 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:02,226 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:02,227 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:02,243 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:02,251 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:02,252 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:02,321 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-19 15:35:02,323 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:02,547 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-19 15:35:02,563 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:02,836 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:02,970 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:03,074 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:03,269 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:03,270 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:03,285 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:03,293 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:03,294 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:03,364 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-19 15:35:03,366 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:03,591 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-19 15:35:03,612 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:03,876 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:04,006 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:04,113 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:04,308 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:04,309 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:04,327 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:04,334 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:04,335 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:04,408 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-19 15:35:04,410 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:04,633 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-19 15:35:04,648 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:04,914 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:05,042 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:05,144 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:05,331 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:05,332 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:05,347 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:05,355 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:05,356 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:05,424 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-19 15:35:05,426 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:05,654 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-19 15:35:05,669 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:05,940 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:06,072 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:06,175 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:06,342 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:35:06,343 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:35:06,358 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:35:06,365 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:35:06,366 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:06,436 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-19 15:35:06,438 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:06,660 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-19 15:35:06,672 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:06,930 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:07,054 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:07,150 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:07,334 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:07,335 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:07,350 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:07,358 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:07,358 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:07,428 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-19 15:35:07,430 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:07,652 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-19 15:35:07,669 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:07,938 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:08,068 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:08,171 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:08,358 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:08,359 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:08,374 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:08,381 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:08,382 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:08,452 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-19 15:35:08,454 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:08,676 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-19 15:35:08,691 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:08,954 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:09,085 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:09,185 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:09,373 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:09,374 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:09,391 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:09,398 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:35:09,399 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:09,469 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-19 15:35:09,471 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:09,693 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-19 15:35:09,707 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:09,969 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:10,100 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:10,203 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:35:10,494 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:35:10,496 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:35:10,515 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:35:10,526 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:35:10,527 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:35:10,676 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-19 15:35:10,679 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:35:11,192 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-19 15:35:11,237 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:35:11,756 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:35:11,953 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:35:12,086 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:35:12,201 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-19 15:35:12,433 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-04-19 15:35:12,433 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-04-19 15:35:12,433 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-19 15:35:12,433 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-19 15:39:41,381 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-19 15:39:41,381 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-19 15:39:42,709 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-19 15:40:06,209 INFO train_multi TF=ALL epoch 1/50 train=0.8748 val=0.8334
2026-04-19 15:40:06,215 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 15:40:06,215 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 15:40:06,215 INFO train_multi TF=ALL: new best val=0.8334 — saved
2026-04-19 15:40:23,663 INFO train_multi TF=ALL epoch 2/50 train=0.7184 val=0.6879
2026-04-19 15:40:23,667 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 15:40:23,668 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 15:40:23,668 INFO train_multi TF=ALL: new best val=0.6879 — saved
2026-04-19 15:40:41,168 INFO train_multi TF=ALL epoch 3/50 train=0.6901 val=0.6879
2026-04-19 15:40:58,670 INFO train_multi TF=ALL epoch 4/50 train=0.6893 val=0.6879
2026-04-19 15:40:58,674 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 15:40:58,674 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 15:40:58,674 INFO train_multi TF=ALL: new best val=0.6879 — saved
2026-04-19 15:41:16,420 INFO train_multi TF=ALL epoch 5/50 train=0.6891 val=0.6880
2026-04-19 15:41:33,897 INFO train_multi TF=ALL epoch 6/50 train=0.6888 val=0.6878
2026-04-19 15:41:33,902 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 15:41:33,902 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 15:41:33,902 INFO train_multi TF=ALL: new best val=0.6878 — saved
2026-04-19 15:41:51,654 INFO train_multi TF=ALL epoch 7/50 train=0.6887 val=0.6875
2026-04-19 15:41:51,658 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 15:41:51,658 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 15:41:51,658 INFO train_multi TF=ALL: new best val=0.6875 — saved
2026-04-19 15:42:09,339 INFO train_multi TF=ALL epoch 8/50 train=0.6876 val=0.6865
2026-04-19 15:42:09,344 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 15:42:09,344 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 15:42:09,344 INFO train_multi TF=ALL: new best val=0.6865 — saved
2026-04-19 15:42:27,057 INFO train_multi TF=ALL epoch 9/50 train=0.6829 val=0.6773
2026-04-19 15:42:27,062 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 15:42:27,062 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 15:42:27,062 INFO train_multi TF=ALL: new best val=0.6773 — saved
2026-04-19 15:42:45,096 INFO train_multi TF=ALL epoch 10/50 train=0.6727 val=0.6658
2026-04-19 15:42:45,100 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 15:42:45,100 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 15:42:45,101 INFO train_multi TF=ALL: new best val=0.6658 — saved
2026-04-19 15:43:02,835 INFO train_multi TF=ALL epoch 11/50 train=0.6618 val=0.6511
2026-04-19 15:43:02,839 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 15:43:02,840 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 15:43:02,840 INFO train_multi TF=ALL: new best val=0.6511 — saved
2026-04-19 15:43:20,546 INFO train_multi TF=ALL epoch 12/50 train=0.6499 val=0.6406
2026-04-19 15:43:20,550 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 15:43:20,551 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 15:43:20,551 INFO train_multi TF=ALL: new best val=0.6406 — saved
2026-04-19 15:43:38,369 INFO train_multi TF=ALL epoch 13/50 train=0.6408 val=0.6347
2026-04-19 15:43:38,373 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 15:43:38,373 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 15:43:38,373 INFO train_multi TF=ALL: new best val=0.6347 — saved
2026-04-19 15:43:55,988 INFO train_multi TF=ALL epoch 14/50 train=0.6343 val=0.6281
2026-04-19 15:43:55,993 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 15:43:55,993 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 15:43:55,993 INFO train_multi TF=ALL: new best val=0.6281 — saved
2026-04-19 15:44:13,698 INFO train_multi TF=ALL epoch 15/50 train=0.6294 val=0.6267
2026-04-19 15:44:13,702 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 15:44:13,703 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 15:44:13,703 INFO train_multi TF=ALL: new best val=0.6267 — saved
2026-04-19 15:44:31,634 INFO train_multi TF=ALL epoch 16/50 train=0.6260 val=0.6245
2026-04-19 15:44:31,638 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 15:44:31,638 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 15:44:31,638 INFO train_multi TF=ALL: new best val=0.6245 — saved
2026-04-19 15:44:49,319 INFO train_multi TF=ALL epoch 17/50 train=0.6230 val=0.6219
2026-04-19 15:44:49,324 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 15:44:49,324 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 15:44:49,324 INFO train_multi TF=ALL: new best val=0.6219 — saved
2026-04-19 15:45:06,809 INFO train_multi TF=ALL epoch 18/50 train=0.6196 val=0.6218
2026-04-19 15:45:06,813 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 15:45:06,813 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 15:45:06,813 INFO train_multi TF=ALL: new best val=0.6218 — saved
2026-04-19 15:45:24,525 INFO train_multi TF=ALL epoch 19/50 train=0.6169 val=0.6191
2026-04-19 15:45:24,530 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 15:45:24,530 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 15:45:24,530 INFO train_multi TF=ALL: new best val=0.6191 — saved
2026-04-19 15:45:42,358 INFO train_multi TF=ALL epoch 20/50 train=0.6139 val=0.6170
2026-04-19 15:45:42,362 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 15:45:42,362 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 15:45:42,362 INFO train_multi TF=ALL: new best val=0.6170 — saved
2026-04-19 15:46:00,201 INFO train_multi TF=ALL epoch 21/50 train=0.6118 val=0.6154
2026-04-19 15:46:00,205 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 15:46:00,205 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 15:46:00,206 INFO train_multi TF=ALL: new best val=0.6154 — saved
2026-04-19 15:46:18,327 INFO train_multi TF=ALL epoch 22/50 train=0.6098 val=0.6155
2026-04-19 15:46:35,916 INFO train_multi TF=ALL epoch 23/50 train=0.6081 val=0.6127
2026-04-19 15:46:35,920 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 15:46:35,920 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 15:46:35,920 INFO train_multi TF=ALL: new best val=0.6127 — saved
2026-04-19 15:46:53,297 INFO train_multi TF=ALL epoch 24/50 train=0.6060 val=0.6170
2026-04-19 15:47:10,695 INFO train_multi TF=ALL epoch 25/50 train=0.6043 val=0.6170
2026-04-19 15:47:28,010 INFO train_multi TF=ALL epoch 26/50 train=0.6027 val=0.6149
2026-04-19 15:47:45,355 INFO train_multi TF=ALL epoch 27/50 train=0.6012 val=0.6139
2026-04-19 15:48:02,874 INFO train_multi TF=ALL epoch 28/50 train=0.5997 val=0.6141
2026-04-19 15:48:02,875 INFO train_multi TF=ALL early stop at epoch 28
2026-04-19 15:48:03,008 INFO === VectorStore: building similarity indices ===
2026-04-19 15:48:03,009 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-19 15:48:03,009 INFO Retrain complete.
2026-04-19 15:48:04,927 INFO Model gru: SUCCESS
2026-04-19 15:48:04,927 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 15:48:04,928 WARNING   [MISSING] regime_classifier → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_classifier.pkl
2026-04-19 15:48:04,928 INFO   [OK] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-19 15:48:04,928 WARNING   [MISSING] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-19 15:48:04,928 WARNING Missing weights: ['regime_classifier', 'rl_ppo'] — run retrain_incremental.py for each
2026-04-19 15:48:04,929 INFO Saved 18 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  Cleared journal for fresh Round 1 run
  START Round 1 - Backtest (val)
2026-04-19 15:48:05,460 INFO === STEP 6: BACKTEST + REINFORCED TRAINING (3 rounds, window=round1) ===
2026-04-19 15:48:05,461 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-04-19 15:48:05,461 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-19 15:48:05,461 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)

======================================================================
  BACKTEST + REINFORCED TRAINING COMPLETE  (0 rounds)
======================================================================
  Round     Trades       WR      PF   Sharpe
  ------------------------------------------

  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 0 entries

=== Round 1 → Retrain Quality + RL ===
  START Round 1 - Quality+RL retrain