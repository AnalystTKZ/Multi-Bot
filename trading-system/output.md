Cleared done-check: training_summary.json
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

=== Phase 0-5: Data preparation ===
  SKIP  Step 0 - Resample
  SKIP  Step 1 - Inventory
  SKIP  Step 2 - Cleaning
  SKIP  Step 3 - Alignment
  SKIP  Step 4 - Features
  START Step 5 - Split
2026-04-19 07:22:15,838 INFO Loading feature-engineered data...
2026-04-19 07:22:16,427 INFO Loaded 221743 rows, 202 features
2026-04-19 07:22:16,428 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-19 07:22:16,430 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-19 07:22:16,430 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-19 07:22:16,430 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-19 07:22:16,430 INFO No leakage confirmed: train < val < test timestamps

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
2026-04-19 07:22:18,724 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-19 07:22:18,724 INFO --- Training regime ---
2026-04-19 07:22:18,725 INFO Running retrain --model regime
2026-04-19 07:22:18,937 INFO retrain environment: KAGGLE
2026-04-19 07:22:20,553 INFO Device: CUDA (2 GPU(s))
2026-04-19 07:22:20,563 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 07:22:20,564 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 07:22:20,565 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-19 07:22:20,706 INFO NumExpr defaulting to 4 threads.
2026-04-19 07:22:20,908 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-19 07:22:20,908 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 07:22:20,908 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 07:22:21,116 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-19 07:22:21,118 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:21,204 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:21,283 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:21,364 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:21,433 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:21,513 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:21,594 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:21,668 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:21,750 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:21,823 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:21,917 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:22:21,980 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-19 07:22:21,996 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:21,997 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:22,013 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:22,014 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:22,031 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:22,032 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:22,048 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:22,050 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:22,066 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:22,069 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:22,083 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:22,086 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:22,101 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:22,103 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:22,118 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:22,122 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:22,137 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:22,140 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:22,155 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:22,158 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:22:22,176 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:22:22,183 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:22:23,236 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-19 07:22:44,666 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-19 07:22:44,667 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias)
2026-04-19 07:22:44,668 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-19 07:22:54,952 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-19 07:22:54,953 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias)
2026-04-19 07:22:54,953 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-19 07:23:02,970 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-19 07:23:02,974 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias)
2026-04-19 07:23:02,974 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-19 07:24:12,335 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-19 07:24:12,337 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour)
2026-04-19 07:24:12,337 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-19 07:24:42,745 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-19 07:24:42,749 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour)
2026-04-19 07:24:42,749 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-19 07:25:04,645 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-19 07:25:04,646 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour)
2026-04-19 07:25:04,741 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-19 07:25:04,743 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:04,744 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:04,745 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:04,746 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:04,747 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:04,748 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:04,749 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:04,750 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:04,751 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:04,752 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:04,753 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:25:04,877 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:04,918 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:04,919 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:04,919 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:04,928 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:04,929 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:05,319 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2439, 'BIAS_DOWN': 2349, 'BIAS_NEUTRAL': 3614}  ambiguous=4742 (total=8402)  short_runs_zeroed=1455
2026-04-19 07:25:05,320 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-19 07:25:05,492 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:05,525 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:05,526 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:05,526 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:05,535 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:05,535 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:05,881 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2503, 'BIAS_DOWN': 2397, 'BIAS_NEUTRAL': 3502}  ambiguous=4774 (total=8402)  short_runs_zeroed=1656
2026-04-19 07:25:05,882 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-19 07:25:06,049 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:06,085 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:06,085 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:06,086 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:06,094 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:06,095 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:06,450 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2482, 'BIAS_DOWN': 2360, 'BIAS_NEUTRAL': 3560}  ambiguous=4903 (total=8402)  short_runs_zeroed=1643
2026-04-19 07:25:06,451 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-19 07:25:06,611 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:06,644 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:06,645 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:06,645 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:06,654 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:06,654 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:07,005 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2460, 'BIAS_DOWN': 2376, 'BIAS_NEUTRAL': 3566}  ambiguous=4785 (total=8402)  short_runs_zeroed=1533
2026-04-19 07:25:07,006 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-19 07:25:07,171 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:07,205 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:07,206 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:07,206 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:07,215 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:07,216 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:07,558 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2348, 'BIAS_DOWN': 2535, 'BIAS_NEUTRAL': 3520}  ambiguous=4533 (total=8403)  short_runs_zeroed=1276
2026-04-19 07:25:07,559 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-19 07:25:07,719 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:07,751 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:07,752 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:07,752 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:07,761 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:07,762 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:08,110 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2440, 'BIAS_DOWN': 2280, 'BIAS_NEUTRAL': 3683}  ambiguous=4754 (total=8403)  short_runs_zeroed=1350
2026-04-19 07:25:08,111 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-19 07:25:08,258 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:25:08,285 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:25:08,286 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:25:08,286 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:25:08,294 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:25:08,294 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:08,645 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2526, 'BIAS_DOWN': 2331, 'BIAS_NEUTRAL': 3545}  ambiguous=4840 (total=8402)  short_runs_zeroed=1601
2026-04-19 07:25:08,646 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-19 07:25:08,800 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:08,832 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:08,833 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:08,833 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:08,841 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:08,842 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:09,193 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2250, 'BIAS_DOWN': 2599, 'BIAS_NEUTRAL': 3553}  ambiguous=4646 (total=8402)  short_runs_zeroed=1418
2026-04-19 07:25:09,194 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-19 07:25:09,356 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:09,389 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:09,390 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:09,390 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:09,399 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:09,400 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:09,734 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2214, 'BIAS_DOWN': 2634, 'BIAS_NEUTRAL': 3554}  ambiguous=4734 (total=8402)  short_runs_zeroed=1496
2026-04-19 07:25:09,735 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-19 07:25:09,905 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:09,939 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:09,940 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:09,940 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:09,949 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:09,950 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:10,302 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2278, 'BIAS_DOWN': 2515, 'BIAS_NEUTRAL': 3610}  ambiguous=4849 (total=8403)  short_runs_zeroed=1563
2026-04-19 07:25:10,303 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-19 07:25:10,561 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:25:10,620 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:25:10,621 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:25:10,622 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:25:10,632 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:25:10,633 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:25:11,374 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 6629, 'BIAS_DOWN': 5241, 'BIAS_NEUTRAL': 7947}  ambiguous=10212 (total=19817)  short_runs_zeroed=2878
2026-04-19 07:25:11,376 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-19 07:25:11,567 INFO RegimeClassifier[mode=htf_bias]: 103290 samples, classes={'BIAS_UP': 30569, 'BIAS_DOWN': 29617, 'BIAS_NEUTRAL': 43104}, device=cuda
2026-04-19 07:25:11,568 INFO RegimeClassifier: sample weights — mean=0.355  ambiguous(<0.4)=56.4%
2026-04-19 07:25:11,832 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-04-19 07:25:11,833 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-19 07:25:16,350 INFO Regime epoch  1/50 — tr=0.4309 va=1.1156 acc=0.387 per_class={'BIAS_UP': 0.061, 'BIAS_DOWN': 0.482, 'BIAS_NEUTRAL': 0.604}
2026-04-19 07:25:16,515 INFO Regime epoch  2/50 — tr=0.4221 va=1.1114 acc=0.416
2026-04-19 07:25:16,680 INFO Regime epoch  3/50 — tr=0.4021 va=1.0643 acc=0.505
2026-04-19 07:25:16,846 INFO Regime epoch  4/50 — tr=0.3765 va=0.9859 acc=0.629
2026-04-19 07:25:17,023 INFO Regime epoch  5/50 — tr=0.3436 va=0.8970 acc=0.681 per_class={'BIAS_UP': 0.932, 'BIAS_DOWN': 0.886, 'BIAS_NEUTRAL': 0.335}
2026-04-19 07:25:17,194 INFO Regime epoch  6/50 — tr=0.3111 va=0.8037 acc=0.697
2026-04-19 07:25:17,363 INFO Regime epoch  7/50 — tr=0.2848 va=0.7292 acc=0.698
2026-04-19 07:25:17,525 INFO Regime epoch  8/50 — tr=0.2714 va=0.6828 acc=0.697
2026-04-19 07:25:17,686 INFO Regime epoch  9/50 — tr=0.2639 va=0.6609 acc=0.697
2026-04-19 07:25:17,866 INFO Regime epoch 10/50 — tr=0.2597 va=0.6526 acc=0.693 per_class={'BIAS_UP': 0.998, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.234}
2026-04-19 07:25:18,040 INFO Regime epoch 11/50 — tr=0.2571 va=0.6482 acc=0.697
2026-04-19 07:25:18,211 INFO Regime epoch 12/50 — tr=0.2544 va=0.6466 acc=0.695
2026-04-19 07:25:18,380 INFO Regime epoch 13/50 — tr=0.2526 va=0.6462 acc=0.691
2026-04-19 07:25:18,554 INFO Regime epoch 14/50 — tr=0.2509 va=0.6453 acc=0.691
2026-04-19 07:25:18,741 INFO Regime epoch 15/50 — tr=0.2497 va=0.6447 acc=0.691 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.228}
2026-04-19 07:25:18,911 INFO Regime epoch 16/50 — tr=0.2484 va=0.6436 acc=0.693
2026-04-19 07:25:19,075 INFO Regime epoch 17/50 — tr=0.2474 va=0.6427 acc=0.690
2026-04-19 07:25:19,241 INFO Regime epoch 18/50 — tr=0.2466 va=0.6412 acc=0.691
2026-04-19 07:25:19,408 INFO Regime epoch 19/50 — tr=0.2457 va=0.6406 acc=0.691
2026-04-19 07:25:19,595 INFO Regime epoch 20/50 — tr=0.2452 va=0.6392 acc=0.696 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.24}
2026-04-19 07:25:19,757 INFO Regime epoch 21/50 — tr=0.2445 va=0.6380 acc=0.699
2026-04-19 07:25:19,930 INFO Regime epoch 22/50 — tr=0.2440 va=0.6387 acc=0.696
2026-04-19 07:25:20,100 INFO Regime epoch 23/50 — tr=0.2435 va=0.6393 acc=0.697
2026-04-19 07:25:20,267 INFO Regime epoch 24/50 — tr=0.2430 va=0.6353 acc=0.699
2026-04-19 07:25:20,454 INFO Regime epoch 25/50 — tr=0.2426 va=0.6366 acc=0.699 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.248}
2026-04-19 07:25:20,617 INFO Regime epoch 26/50 — tr=0.2423 va=0.6358 acc=0.700
2026-04-19 07:25:20,785 INFO Regime epoch 27/50 — tr=0.2420 va=0.6339 acc=0.700
2026-04-19 07:25:20,957 INFO Regime epoch 28/50 — tr=0.2417 va=0.6336 acc=0.699
2026-04-19 07:25:21,122 INFO Regime epoch 29/50 — tr=0.2414 va=0.6336 acc=0.704
2026-04-19 07:25:21,304 INFO Regime epoch 30/50 — tr=0.2412 va=0.6341 acc=0.703 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.257}
2026-04-19 07:25:21,482 INFO Regime epoch 31/50 — tr=0.2410 va=0.6326 acc=0.704
2026-04-19 07:25:21,671 INFO Regime epoch 32/50 — tr=0.2407 va=0.6336 acc=0.704
2026-04-19 07:25:21,834 INFO Regime epoch 33/50 — tr=0.2404 va=0.6328 acc=0.708
2026-04-19 07:25:21,997 INFO Regime epoch 34/50 — tr=0.2404 va=0.6314 acc=0.705
2026-04-19 07:25:22,187 INFO Regime epoch 35/50 — tr=0.2401 va=0.6316 acc=0.702 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.256}
2026-04-19 07:25:22,352 INFO Regime epoch 36/50 — tr=0.2403 va=0.6319 acc=0.705
2026-04-19 07:25:22,517 INFO Regime epoch 37/50 — tr=0.2400 va=0.6309 acc=0.706
2026-04-19 07:25:22,693 INFO Regime epoch 38/50 — tr=0.2400 va=0.6308 acc=0.708
2026-04-19 07:25:22,886 INFO Regime epoch 39/50 — tr=0.2401 va=0.6307 acc=0.707
2026-04-19 07:25:23,085 INFO Regime epoch 40/50 — tr=0.2398 va=0.6303 acc=0.706 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.265}
2026-04-19 07:25:23,262 INFO Regime epoch 41/50 — tr=0.2399 va=0.6307 acc=0.708
2026-04-19 07:25:23,432 INFO Regime epoch 42/50 — tr=0.2397 va=0.6307 acc=0.707
2026-04-19 07:25:23,599 INFO Regime epoch 43/50 — tr=0.2397 va=0.6305 acc=0.709
2026-04-19 07:25:23,765 INFO Regime epoch 44/50 — tr=0.2397 va=0.6310 acc=0.707
2026-04-19 07:25:23,947 INFO Regime epoch 45/50 — tr=0.2396 va=0.6315 acc=0.707 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.267}
2026-04-19 07:25:24,110 INFO Regime epoch 46/50 — tr=0.2397 va=0.6323 acc=0.709
2026-04-19 07:25:24,269 INFO Regime epoch 47/50 — tr=0.2396 va=0.6307 acc=0.709
2026-04-19 07:25:24,439 INFO Regime epoch 48/50 — tr=0.2395 va=0.6309 acc=0.705
2026-04-19 07:25:24,615 INFO Regime epoch 49/50 — tr=0.2396 va=0.6303 acc=0.704
2026-04-19 07:25:24,798 INFO Regime epoch 50/50 — tr=0.2396 va=0.6310 acc=0.707 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.267}
2026-04-19 07:25:24,798 INFO Regime early stop at epoch 50 (no_improve=10)
2026-04-19 07:25:24,814 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-19 07:25:24,814 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-19 07:25:24,939 INFO Regime HTF complete: acc=0.706, n=103290
2026-04-19 07:25:24,940 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:25:25,106 INFO Rule labels LTF_BEHAVIOUR [4H]: {'TRENDING': 8970, 'RANGING': 4592, 'CONSOLIDATING': 2247, 'VOLATILE': 4008}  ambiguous=8734 (total=19817)  short_runs_zeroed=5706
2026-04-19 07:25:25,111 INFO Regime[4H] persistence (avg bars/run) on XAUUSD 4H:
{0: 12.121621621621621, 1: 6.4858757062146895, 2: 5.575682382133995, 3: 10.329896907216495}
2026-04-19 07:25:25,112 INFO Regime[4H] return separation on XAUUSD 4H:
{0: 5.9502411776840314e-05, 1: 5.664222849552333e-05, 2: -6.138442068010657e-05, 3: 4.239843820627722e-05}
2026-04-19 07:25:25,113 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-19 07:25:25,114 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:25,116 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:25,117 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:25,119 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:25,120 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:25,122 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:25,123 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:25,125 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:25,126 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:25,128 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:25,131 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:25:25,141 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:25,143 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:25,144 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:25,144 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:25,145 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:25,147 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:25,740 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=29662 (total=32738)  short_runs_zeroed=27182
2026-04-19 07:25:25,742 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-19 07:25:25,878 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:25,880 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:25,881 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:25,882 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:25,882 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:25,884 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:26,442 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=29621 (total=32738)  short_runs_zeroed=27005
2026-04-19 07:25:26,445 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-19 07:25:26,578 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:26,580 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:26,581 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:26,581 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:26,581 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:26,583 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:27,134 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=28865 (total=32740)  short_runs_zeroed=26033
2026-04-19 07:25:27,137 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-19 07:25:27,268 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:27,270 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:27,271 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:27,271 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:27,271 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:27,273 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:27,823 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=29322 (total=32739)  short_runs_zeroed=26905
2026-04-19 07:25:27,825 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-19 07:25:27,953 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:27,955 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:27,956 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:27,956 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:27,957 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:27,958 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:28,533 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=29072 (total=32740)  short_runs_zeroed=26611
2026-04-19 07:25:28,535 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-19 07:25:28,666 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:28,668 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:28,669 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:28,669 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:28,669 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:28,671 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:29,211 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=29510 (total=32739)  short_runs_zeroed=26826
2026-04-19 07:25:29,214 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-19 07:25:29,345 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:25:29,347 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:25:29,347 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:25:29,348 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:25:29,348 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:25:29,349 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:29,891 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=29239 (total=32739)  short_runs_zeroed=26684
2026-04-19 07:25:29,894 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-19 07:25:30,022 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:30,024 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:30,024 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:30,025 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:30,025 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:30,027 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:30,555 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=29460 (total=32740)  short_runs_zeroed=26820
2026-04-19 07:25:30,558 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-19 07:25:30,693 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:30,695 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:30,696 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:30,697 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:30,697 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:30,699 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:31,241 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=29430 (total=32741)  short_runs_zeroed=26597
2026-04-19 07:25:31,243 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-19 07:25:31,375 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:31,377 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:31,378 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:31,378 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:31,378 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:25:31,380 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:25:31,925 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=28427 (total=32743)  short_runs_zeroed=25756
2026-04-19 07:25:31,928 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-19 07:25:32,067 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:25:32,070 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:25:32,071 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:25:32,072 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:25:32,072 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:25:32,075 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:25:33,276 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=66883 (total=74624)  short_runs_zeroed=60365
2026-04-19 07:25:33,282 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-19 07:25:33,619 INFO RegimeClassifier[mode=ltf_behaviour]: 401471 samples, classes={'TRENDING': 160094, 'RANGING': 105286, 'CONSOLIDATING': 53524, 'VOLATILE': 82567}, device=cuda
2026-04-19 07:25:33,620 INFO RegimeClassifier: sample weights — mean=0.096  ambiguous(<0.4)=89.4%
2026-04-19 07:25:33,622 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-04-19 07:25:33,623 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-19 07:25:34,332 INFO Regime epoch  1/50 — tr=0.1282 va=1.4711 acc=0.190 per_class={'TRENDING': 0.164, 'RANGING': 0.284, 'CONSOLIDATING': 0.064, 'VOLATILE': 0.205}
2026-04-19 07:25:34,981 INFO Regime epoch  2/50 — tr=0.1187 va=1.3602 acc=0.261
2026-04-19 07:25:35,609 INFO Regime epoch  3/50 — tr=0.1046 va=1.2581 acc=0.331
2026-04-19 07:25:36,253 INFO Regime epoch  4/50 — tr=0.0914 va=1.1820 acc=0.347
2026-04-19 07:25:36,929 INFO Regime epoch  5/50 — tr=0.0840 va=1.1397 acc=0.365 per_class={'TRENDING': 0.142, 'RANGING': 0.003, 'CONSOLIDATING': 0.862, 'VOLATILE': 0.949}
2026-04-19 07:25:37,549 INFO Regime epoch  6/50 — tr=0.0802 va=1.1214 acc=0.375
2026-04-19 07:25:38,178 INFO Regime epoch  7/50 — tr=0.0776 va=1.1108 acc=0.380
2026-04-19 07:25:38,818 INFO Regime epoch  8/50 — tr=0.0758 va=1.1058 acc=0.386
2026-04-19 07:25:39,445 INFO Regime epoch  9/50 — tr=0.0742 va=1.1066 acc=0.385
2026-04-19 07:25:40,133 INFO Regime epoch 10/50 — tr=0.0731 va=1.1025 acc=0.385 per_class={'TRENDING': 0.154, 'RANGING': 0.026, 'CONSOLIDATING': 0.928, 'VOLATILE': 0.952}
2026-04-19 07:25:40,753 INFO Regime epoch 11/50 — tr=0.0720 va=1.1009 acc=0.390
2026-04-19 07:25:41,386 INFO Regime epoch 12/50 — tr=0.0711 va=1.0992 acc=0.396
2026-04-19 07:25:42,018 INFO Regime epoch 13/50 — tr=0.0705 va=1.0993 acc=0.398
2026-04-19 07:25:42,634 INFO Regime epoch 14/50 — tr=0.0702 va=1.0916 acc=0.398
2026-04-19 07:25:43,329 INFO Regime epoch 15/50 — tr=0.0697 va=1.0896 acc=0.400 per_class={'TRENDING': 0.185, 'RANGING': 0.029, 'CONSOLIDATING': 0.942, 'VOLATILE': 0.949}
2026-04-19 07:25:43,943 INFO Regime epoch 16/50 — tr=0.0691 va=1.0882 acc=0.405
2026-04-19 07:25:44,553 INFO Regime epoch 17/50 — tr=0.0689 va=1.0831 acc=0.406
2026-04-19 07:25:45,175 INFO Regime epoch 18/50 — tr=0.0685 va=1.0833 acc=0.409
2026-04-19 07:25:45,780 INFO Regime epoch 19/50 — tr=0.0682 va=1.0806 acc=0.415
2026-04-19 07:25:46,461 INFO Regime epoch 20/50 — tr=0.0679 va=1.0762 acc=0.421 per_class={'TRENDING': 0.241, 'RANGING': 0.025, 'CONSOLIDATING': 0.952, 'VOLATILE': 0.943}
2026-04-19 07:25:47,106 INFO Regime epoch 21/50 — tr=0.0676 va=1.0745 acc=0.421
2026-04-19 07:25:47,744 INFO Regime epoch 22/50 — tr=0.0676 va=1.0731 acc=0.422
2026-04-19 07:25:48,383 INFO Regime epoch 23/50 — tr=0.0671 va=1.0686 acc=0.431
2026-04-19 07:25:49,019 INFO Regime epoch 24/50 — tr=0.0670 va=1.0684 acc=0.432
2026-04-19 07:25:49,667 INFO Regime epoch 25/50 — tr=0.0669 va=1.0682 acc=0.435 per_class={'TRENDING': 0.267, 'RANGING': 0.035, 'CONSOLIDATING': 0.958, 'VOLATILE': 0.939}
2026-04-19 07:25:50,282 INFO Regime epoch 26/50 — tr=0.0668 va=1.0646 acc=0.442
2026-04-19 07:25:50,905 INFO Regime epoch 27/50 — tr=0.0666 va=1.0647 acc=0.444
2026-04-19 07:25:51,553 INFO Regime epoch 28/50 — tr=0.0664 va=1.0617 acc=0.442
2026-04-19 07:25:52,195 INFO Regime epoch 29/50 — tr=0.0663 va=1.0608 acc=0.448
2026-04-19 07:25:52,879 INFO Regime epoch 30/50 — tr=0.0662 va=1.0605 acc=0.451 per_class={'TRENDING': 0.308, 'RANGING': 0.04, 'CONSOLIDATING': 0.962, 'VOLATILE': 0.932}
2026-04-19 07:25:53,518 INFO Regime epoch 31/50 — tr=0.0662 va=1.0608 acc=0.452
2026-04-19 07:25:54,162 INFO Regime epoch 32/50 — tr=0.0660 va=1.0563 acc=0.452
2026-04-19 07:25:54,794 INFO Regime epoch 33/50 — tr=0.0658 va=1.0560 acc=0.457
2026-04-19 07:25:55,419 INFO Regime epoch 34/50 — tr=0.0657 va=1.0559 acc=0.460
2026-04-19 07:25:56,093 INFO Regime epoch 35/50 — tr=0.0657 va=1.0535 acc=0.460 per_class={'TRENDING': 0.325, 'RANGING': 0.049, 'CONSOLIDATING': 0.963, 'VOLATILE': 0.931}
2026-04-19 07:25:56,709 INFO Regime epoch 36/50 — tr=0.0656 va=1.0544 acc=0.459
2026-04-19 07:25:57,343 INFO Regime epoch 37/50 — tr=0.0657 va=1.0529 acc=0.462
2026-04-19 07:25:57,951 INFO Regime epoch 38/50 — tr=0.0657 va=1.0528 acc=0.463
2026-04-19 07:25:58,577 INFO Regime epoch 39/50 — tr=0.0655 va=1.0526 acc=0.463
2026-04-19 07:25:59,248 INFO Regime epoch 40/50 — tr=0.0655 va=1.0519 acc=0.466 per_class={'TRENDING': 0.335, 'RANGING': 0.053, 'CONSOLIDATING': 0.964, 'VOLATILE': 0.931}
2026-04-19 07:25:59,852 INFO Regime epoch 41/50 — tr=0.0655 va=1.0513 acc=0.467
2026-04-19 07:26:00,473 INFO Regime epoch 42/50 — tr=0.0655 va=1.0512 acc=0.467
2026-04-19 07:26:01,097 INFO Regime epoch 43/50 — tr=0.0654 va=1.0492 acc=0.468
2026-04-19 07:26:01,770 INFO Regime epoch 44/50 — tr=0.0654 va=1.0505 acc=0.466
2026-04-19 07:26:02,451 INFO Regime epoch 45/50 — tr=0.0654 va=1.0513 acc=0.467 per_class={'TRENDING': 0.34, 'RANGING': 0.053, 'CONSOLIDATING': 0.967, 'VOLATILE': 0.926}
2026-04-19 07:26:03,052 INFO Regime epoch 46/50 — tr=0.0655 va=1.0490 acc=0.467
2026-04-19 07:26:03,659 INFO Regime epoch 47/50 — tr=0.0655 va=1.0519 acc=0.467
2026-04-19 07:26:04,273 INFO Regime epoch 48/50 — tr=0.0654 va=1.0515 acc=0.468
2026-04-19 07:26:04,881 INFO Regime epoch 49/50 — tr=0.0653 va=1.0514 acc=0.467
2026-04-19 07:26:05,574 INFO Regime epoch 50/50 — tr=0.0654 va=1.0514 acc=0.471 per_class={'TRENDING': 0.353, 'RANGING': 0.049, 'CONSOLIDATING': 0.97, 'VOLATILE': 0.92}
2026-04-19 07:26:05,615 WARNING RegimeClassifier accuracy 0.47 < 0.65 threshold
2026-04-19 07:26:05,618 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-19 07:26:05,618 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-19 07:26:05,743 INFO Regime LTF complete: acc=0.467, n=401471
2026-04-19 07:26:05,747 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:26:06,223 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=66883 (total=74624)  short_runs_zeroed=60365
2026-04-19 07:26:06,227 INFO Regime[1H] persistence (avg bars/run) on XAUUSD 1H:
{0: 9.355291913830783, 1: 6.096303199751476, 2: 5.598885793871866, 3: 6.771351107094442}
2026-04-19 07:26:06,230 INFO Regime[1H] return separation on XAUUSD 1H:
{0: 2.149062259083323e-05, 1: -3.254493608872846e-06, 2: -3.039756158146958e-06, 3: 1.8102764458593193e-05}
2026-04-19 07:26:06,242 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-19 07:26:06,242 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 07:26:06,242 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 07:26:06,242 INFO === VectorStore: building similarity indices ===
2026-04-19 07:26:06,242 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-19 07:26:06,243 INFO Retrain complete.
2026-04-19 07:26:08,390 INFO Model regime: SUCCESS
2026-04-19 07:26:08,390 INFO --- Training gru ---
2026-04-19 07:26:08,391 INFO Running retrain --model gru
2026-04-19 07:26:08,678 INFO retrain environment: KAGGLE
2026-04-19 07:26:10,325 INFO Device: CUDA (2 GPU(s))
2026-04-19 07:26:10,336 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 07:26:10,336 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 07:26:10,337 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-19 07:26:10,471 INFO NumExpr defaulting to 4 threads.
2026-04-19 07:26:10,662 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-19 07:26:10,662 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 07:26:10,662 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 07:26:10,665 WARNING GRULSTMPredictor: stale weights detected (gru feature contract changed: added=['htf_bias_conf', 'htf_bias_down', 'htf_bias_neutral', 'htf_bias_up', 'htf_ltf_align', 'htf_regime_dur', 'ltf_conf', 'ltf_consolidating', 'ltf_ranging', 'ltf_regime_dur', 'ltf_trending', 'ltf_volatile']; removed=['regime_1h_0', 'regime_1h_1', 'regime_1h_2', 'regime_1h_3', 'regime_1h_4', 'regime_1h_conf', 'regime_4h_0', 'regime_4h_1', 'regime_4h_2', 'regime_4h_3', 'regime_4h_4', 'regime_4h_conf']) — deleting /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt so retrain starts fresh
2026-04-19 07:26:10,665 INFO Deleted stale weights (gru feature contract changed: added=['htf_bias_conf', 'htf_bias_down', 'htf_bias_neutral', 'htf_bias_up', 'htf_ltf_align', 'htf_regime_dur', 'ltf_conf', 'ltf_consolidating', 'ltf_ranging', 'ltf_regime_dur', 'ltf_trending', 'ltf_volatile']; removed=['regime_1h_0', 'regime_1h_1', 'regime_1h_2', 'regime_1h_3', 'regime_1h_4', 'regime_1h_conf', 'regime_4h_0', 'regime_4h_1', 'regime_4h_2', 'regime_4h_3', 'regime_4h_4', 'regime_4h_conf']): /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 07:26:10,867 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-19 07:26:10,869 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:10,945 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:11,020 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:11,098 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:11,177 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:11,259 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:11,346 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:11,423 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:11,501 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:11,585 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:11,674 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:26:11,735 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-19 07:26:11,736 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260419_072611
2026-04-19 07:26:11,736 INFO GRU weights stale (gru feature contract changed: added=['htf_bias_conf', 'htf_bias_down', 'htf_bias_neutral', 'htf_bias_up', 'htf_ltf_align', 'htf_regime_dur', 'ltf_conf', 'ltf_consolidating', 'ltf_ranging', 'ltf_regime_dur', 'ltf_trending', 'ltf_volatile']; removed=['regime_1h_0', 'regime_1h_1', 'regime_1h_2', 'regime_1h_3', 'regime_1h_4', 'regime_1h_conf', 'regime_4h_0', 'regime_4h_1', 'regime_4h_2', 'regime_4h_3', 'regime_4h_4', 'regime_4h_conf']) — deleting for full retrain
2026-04-19 07:26:11,852 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:11,853 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:11,874 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:11,881 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:11,882 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-19 07:26:11,882 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 07:26:11,882 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 07:26:11,883 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:11,955 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2439, 'BIAS_DOWN': 2349, 'BIAS_NEUTRAL': 3614}  ambiguous=4742 (total=8402)  short_runs_zeroed=1455
2026-04-19 07:26:11,957 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:12,184 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=29662 (total=32738)  short_runs_zeroed=27182
2026-04-19 07:26:12,212 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:12,482 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:12,604 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:12,695 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:12,885 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:12,886 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:12,901 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:12,908 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:12,909 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:12,978 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2503, 'BIAS_DOWN': 2397, 'BIAS_NEUTRAL': 3502}  ambiguous=4774 (total=8402)  short_runs_zeroed=1656
2026-04-19 07:26:12,980 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:13,210 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=29621 (total=32738)  short_runs_zeroed=27005
2026-04-19 07:26:13,225 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:13,478 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:13,599 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:13,685 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:13,863 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:13,864 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:13,879 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:13,886 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:13,887 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:13,957 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2482, 'BIAS_DOWN': 2360, 'BIAS_NEUTRAL': 3560}  ambiguous=4903 (total=8402)  short_runs_zeroed=1643
2026-04-19 07:26:13,959 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:14,176 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=28865 (total=32740)  short_runs_zeroed=26033
2026-04-19 07:26:14,190 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:14,446 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:14,567 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:14,657 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:14,838 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:14,839 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:14,854 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:14,862 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:14,863 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:14,931 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2460, 'BIAS_DOWN': 2376, 'BIAS_NEUTRAL': 3566}  ambiguous=4785 (total=8402)  short_runs_zeroed=1533
2026-04-19 07:26:14,932 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:15,147 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=29322 (total=32739)  short_runs_zeroed=26905
2026-04-19 07:26:15,167 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:15,422 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:15,540 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:15,631 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:15,805 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:15,806 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:15,821 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:15,830 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:15,830 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:15,901 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2348, 'BIAS_DOWN': 2535, 'BIAS_NEUTRAL': 3520}  ambiguous=4533 (total=8403)  short_runs_zeroed=1276
2026-04-19 07:26:15,903 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:16,122 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=29072 (total=32740)  short_runs_zeroed=26611
2026-04-19 07:26:16,137 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:16,399 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:16,521 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:16,611 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:16,785 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:16,786 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:16,800 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:16,808 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:16,809 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:16,876 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2440, 'BIAS_DOWN': 2280, 'BIAS_NEUTRAL': 3683}  ambiguous=4754 (total=8403)  short_runs_zeroed=1350
2026-04-19 07:26:16,878 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:17,083 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=29510 (total=32739)  short_runs_zeroed=26826
2026-04-19 07:26:17,097 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:17,351 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:17,471 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:17,558 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:17,711 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:26:17,712 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:26:17,727 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:26:17,734 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:26:17,735 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:17,807 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2526, 'BIAS_DOWN': 2331, 'BIAS_NEUTRAL': 3545}  ambiguous=4840 (total=8402)  short_runs_zeroed=1601
2026-04-19 07:26:17,809 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:18,017 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=29239 (total=32739)  short_runs_zeroed=26684
2026-04-19 07:26:18,029 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:18,278 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:18,398 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:18,488 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:18,657 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:18,658 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:18,672 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:18,679 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:18,680 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:18,748 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2250, 'BIAS_DOWN': 2599, 'BIAS_NEUTRAL': 3553}  ambiguous=4646 (total=8402)  short_runs_zeroed=1418
2026-04-19 07:26:18,749 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:18,959 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=29460 (total=32740)  short_runs_zeroed=26820
2026-04-19 07:26:18,975 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:19,238 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:19,361 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:19,449 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:19,619 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:19,620 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:19,635 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:19,642 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:19,643 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:19,712 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2214, 'BIAS_DOWN': 2634, 'BIAS_NEUTRAL': 3554}  ambiguous=4734 (total=8402)  short_runs_zeroed=1496
2026-04-19 07:26:19,714 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:19,931 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=29430 (total=32741)  short_runs_zeroed=26597
2026-04-19 07:26:19,946 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:20,209 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:20,336 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:20,429 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:20,608 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:20,609 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:20,624 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:20,632 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:26:20,632 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:20,702 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2278, 'BIAS_DOWN': 2515, 'BIAS_NEUTRAL': 3610}  ambiguous=4849 (total=8403)  short_runs_zeroed=1563
2026-04-19 07:26:20,704 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:20,926 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=28427 (total=32743)  short_runs_zeroed=25756
2026-04-19 07:26:20,941 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:21,198 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:21,329 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:21,425 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:26:21,722 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:26:21,723 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:26:21,740 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:26:21,750 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:26:21,751 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:26:21,890 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 6629, 'BIAS_DOWN': 5241, 'BIAS_NEUTRAL': 7947}  ambiguous=10212 (total=19817)  short_runs_zeroed=2878
2026-04-19 07:26:21,893 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:26:22,367 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=66883 (total=74624)  short_runs_zeroed=60365
2026-04-19 07:26:22,411 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:26:22,909 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:26:23,099 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:26:23,229 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:26:23,345 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-19 07:26:23,592 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-04-19 07:26:23,592 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-04-19 07:26:23,592 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-19 07:26:23,592 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-19 07:30:42,593 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-19 07:30:42,594 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-19 07:30:43,900 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-19 07:31:04,005 INFO train_multi TF=ALL epoch 1/50 train=0.8674 val=0.8254
2026-04-19 07:31:04,012 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 07:31:04,012 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 07:31:04,012 INFO train_multi TF=ALL: new best val=0.8254 — saved
2026-04-19 07:31:17,967 INFO train_multi TF=ALL epoch 2/50 train=0.7281 val=0.6891
2026-04-19 07:31:17,970 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 07:31:17,970 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 07:31:17,970 INFO train_multi TF=ALL: new best val=0.6891 — saved
2026-04-19 07:31:31,892 INFO train_multi TF=ALL epoch 3/50 train=0.6891 val=0.6904
2026-04-19 07:31:45,622 INFO train_multi TF=ALL epoch 4/50 train=0.6882 val=0.6933
2026-04-19 07:31:59,618 INFO train_multi TF=ALL epoch 5/50 train=0.6872 val=0.6912
2026-04-19 07:32:13,527 INFO train_multi TF=ALL epoch 6/50 train=0.6859 val=0.6889
2026-04-19 07:32:13,530 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 07:32:13,530 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 07:32:13,530 INFO train_multi TF=ALL: new best val=0.6889 — saved
2026-04-19 07:32:27,561 INFO train_multi TF=ALL epoch 7/50 train=0.6835 val=0.6860
2026-04-19 07:32:27,564 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 07:32:27,564 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 07:32:27,564 INFO train_multi TF=ALL: new best val=0.6860 — saved
2026-04-19 07:32:41,511 INFO train_multi TF=ALL epoch 8/50 train=0.6753 val=0.6706
2026-04-19 07:32:41,515 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 07:32:41,515 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 07:32:41,515 INFO train_multi TF=ALL: new best val=0.6706 — saved
2026-04-19 07:32:55,493 INFO train_multi TF=ALL epoch 9/50 train=0.6604 val=0.6593
2026-04-19 07:32:55,496 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 07:32:55,496 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 07:32:55,497 INFO train_multi TF=ALL: new best val=0.6593 — saved
2026-04-19 07:33:09,420 INFO train_multi TF=ALL epoch 10/50 train=0.6493 val=0.6465
2026-04-19 07:33:09,423 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 07:33:09,423 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 07:33:09,423 INFO train_multi TF=ALL: new best val=0.6465 — saved
2026-04-19 07:33:23,288 INFO train_multi TF=ALL epoch 11/50 train=0.6406 val=0.6409
2026-04-19 07:33:23,291 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 07:33:23,291 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 07:33:23,291 INFO train_multi TF=ALL: new best val=0.6409 — saved
2026-04-19 07:33:37,098 INFO train_multi TF=ALL epoch 12/50 train=0.6336 val=0.6345
2026-04-19 07:33:37,101 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 07:33:37,101 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 07:33:37,102 INFO train_multi TF=ALL: new best val=0.6345 — saved
2026-04-19 07:33:50,906 INFO train_multi TF=ALL epoch 13/50 train=0.6290 val=0.6295
2026-04-19 07:33:50,909 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 07:33:50,909 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 07:33:50,909 INFO train_multi TF=ALL: new best val=0.6295 — saved
2026-04-19 07:34:04,780 INFO train_multi TF=ALL epoch 14/50 train=0.6251 val=0.6288
2026-04-19 07:34:04,783 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 07:34:04,783 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 07:34:04,783 INFO train_multi TF=ALL: new best val=0.6288 — saved
2026-04-19 07:34:18,625 INFO train_multi TF=ALL epoch 15/50 train=0.6220 val=0.6275
2026-04-19 07:34:18,628 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 07:34:18,628 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 07:34:18,628 INFO train_multi TF=ALL: new best val=0.6275 — saved
2026-04-19 07:34:32,520 INFO train_multi TF=ALL epoch 16/50 train=0.6182 val=0.6237
2026-04-19 07:34:32,524 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 07:34:32,524 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 07:34:32,524 INFO train_multi TF=ALL: new best val=0.6237 — saved
2026-04-19 07:34:46,360 INFO train_multi TF=ALL epoch 17/50 train=0.6157 val=0.6248
2026-04-19 07:35:00,152 INFO train_multi TF=ALL epoch 18/50 train=0.6132 val=0.6231
2026-04-19 07:35:00,155 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 07:35:00,155 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 07:35:00,155 INFO train_multi TF=ALL: new best val=0.6231 — saved
2026-04-19 07:35:14,294 INFO train_multi TF=ALL epoch 19/50 train=0.6103 val=0.6190
2026-04-19 07:35:14,297 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 07:35:14,297 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 07:35:14,297 INFO train_multi TF=ALL: new best val=0.6190 — saved
2026-04-19 07:35:28,142 INFO train_multi TF=ALL epoch 20/50 train=0.6078 val=0.6227
2026-04-19 07:35:41,971 INFO train_multi TF=ALL epoch 21/50 train=0.6055 val=0.6231
2026-04-19 07:35:55,755 INFO train_multi TF=ALL epoch 22/50 train=0.6035 val=0.6167
2026-04-19 07:35:55,759 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 07:35:55,759 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 07:35:55,759 INFO train_multi TF=ALL: new best val=0.6167 — saved
2026-04-19 07:36:09,555 INFO train_multi TF=ALL epoch 23/50 train=0.6016 val=0.6169
2026-04-19 07:36:23,491 INFO train_multi TF=ALL epoch 24/50 train=0.5987 val=0.6184
2026-04-19 07:36:37,382 INFO train_multi TF=ALL epoch 25/50 train=0.5963 val=0.6163
2026-04-19 07:36:37,385 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 07:36:37,385 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 07:36:37,385 INFO train_multi TF=ALL: new best val=0.6163 — saved
2026-04-19 07:36:51,193 INFO train_multi TF=ALL epoch 26/50 train=0.5948 val=0.6215
2026-04-19 07:37:05,017 INFO train_multi TF=ALL epoch 27/50 train=0.5923 val=0.6188
2026-04-19 07:37:18,872 INFO train_multi TF=ALL epoch 28/50 train=0.5903 val=0.6237
2026-04-19 07:37:32,777 INFO train_multi TF=ALL epoch 29/50 train=0.5879 val=0.6243
2026-04-19 07:37:46,629 INFO train_multi TF=ALL epoch 30/50 train=0.5857 val=0.6247
2026-04-19 07:37:46,629 INFO train_multi TF=ALL early stop at epoch 30
2026-04-19 07:37:46,760 INFO === VectorStore: building similarity indices ===
2026-04-19 07:37:46,760 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-19 07:37:46,761 INFO Retrain complete.
2026-04-19 07:37:48,668 INFO Model gru: SUCCESS
2026-04-19 07:37:48,668 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 07:37:48,668 WARNING   [MISSING] regime_classifier → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_classifier.pkl
2026-04-19 07:37:48,668 INFO   [OK] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-19 07:37:48,668 WARNING   [MISSING] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-19 07:37:48,668 WARNING Missing weights: ['regime_classifier', 'rl_ppo'] — run retrain_incremental.py for each
2026-04-19 07:37:48,669 INFO Saved 18 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  Cleared journal for fresh Round 1 run
  START Round 1 - Backtest (val)
2026-04-19 07:37:49,190 INFO === STEP 6: BACKTEST + REINFORCED TRAINING (3 rounds, window=round1) ===
2026-04-19 07:37:49,191 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-04-19 07:37:49,192 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-19 07:37:49,192 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)

======================================================================
  BACKTEST + REINFORCED TRAINING COMPLETE  (0 rounds)
======================================================================
  Round     Trades       WR      PF   Sharpe
  ------------------------------------------

  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 0 entries