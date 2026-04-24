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
2026-04-24 06:08:13,321 INFO Loading feature-engineered data...
2026-04-24 06:08:13,992 INFO Loaded 221743 rows, 202 features
2026-04-24 06:08:13,994 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-24 06:08:13,996 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-24 06:08:13,996 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-24 06:08:13,996 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-24 06:08:13,996 INFO No leakage confirmed: train < val < test timestamps

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
2026-04-24 06:08:16,689 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-24 06:08:16,689 INFO --- Training regime ---
2026-04-24 06:08:16,690 INFO Running retrain --model regime
2026-04-24 06:08:16,915 INFO retrain environment: KAGGLE
2026-04-24 06:08:19,005 INFO Device: CUDA (2 GPU(s))
2026-04-24 06:08:19,016 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 06:08:19,017 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 06:08:19,018 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-24 06:08:19,185 INFO NumExpr defaulting to 4 threads.
2026-04-24 06:08:19,434 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-24 06:08:19,434 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 06:08:19,434 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 06:08:19,698 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-24 06:08:19,700 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:19,806 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:19,894 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:19,981 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,067 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,155 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,239 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,329 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,414 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,503 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,606 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 06:08:20,675 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-24 06:08:20,694 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,696 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,714 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,716 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,736 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,738 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,756 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,759 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,776 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,780 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,797 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,800 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,815 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,818 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,835 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,839 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,857 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,861 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,878 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,882 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:08:20,902 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 06:08:20,910 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 06:08:22,196 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-24 06:08:47,202 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-24 06:08:47,204 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias)
2026-04-24 06:08:47,204 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-24 06:08:58,790 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-24 06:08:58,794 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias)
2026-04-24 06:08:58,798 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-24 06:09:07,774 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-24 06:09:07,775 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias)
2026-04-24 06:09:07,775 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-24 06:10:26,294 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-24 06:10:26,298 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour)
2026-04-24 06:10:26,298 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-24 06:11:00,435 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-24 06:11:00,440 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour)
2026-04-24 06:11:00,440 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-24 06:11:24,561 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-24 06:11:24,562 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour)
2026-04-24 06:11:24,667 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-24 06:11:24,669 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:11:24,670 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:11:24,672 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:11:24,673 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:11:24,674 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:11:24,675 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:11:24,676 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:11:24,678 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:11:24,679 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:11:24,680 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:11:24,682 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 06:11:24,810 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:24,852 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:24,853 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:24,854 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:24,866 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:24,867 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:11:27,936 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-24 06:11:27,937 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-24 06:11:28,123 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:28,159 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:28,160 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:28,160 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:28,171 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:28,172 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:11:31,107 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-24 06:11:31,108 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-24 06:11:31,293 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:31,331 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:31,332 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:31,333 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:31,342 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:31,343 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:11:34,297 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-24 06:11:34,298 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-24 06:11:34,483 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:34,522 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:34,523 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:34,523 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:34,533 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:34,534 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:11:37,386 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-24 06:11:37,387 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-24 06:11:37,557 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:37,596 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:37,596 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:37,597 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:37,608 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:37,609 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:11:40,354 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-24 06:11:40,356 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-24 06:11:40,535 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:40,570 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:40,571 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:40,571 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:40,580 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:40,581 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:11:43,396 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-24 06:11:43,397 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-24 06:11:43,554 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-24 06:11:43,583 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-24 06:11:43,584 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-24 06:11:43,584 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-24 06:11:43,593 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-24 06:11:43,594 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:11:46,443 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-24 06:11:46,445 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-24 06:11:46,617 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:46,653 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:46,654 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:46,654 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:46,663 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:46,664 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:11:49,463 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-24 06:11:49,464 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-24 06:11:49,638 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:49,672 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:49,673 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:49,674 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:49,683 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:49,684 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:11:52,415 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-24 06:11:52,416 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-24 06:11:52,594 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:52,631 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:52,632 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:52,632 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:52,642 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:11:52,643 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:11:55,447 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-24 06:11:55,448 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-24 06:11:55,714 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-24 06:11:55,773 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-24 06:11:55,774 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-24 06:11:55,775 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-24 06:11:55,786 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-24 06:11:55,787 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 06:12:02,250 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-24 06:12:02,252 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-24 06:12:02,434 INFO RegimeClassifier[mode=htf_bias]: 103290 samples, classes={'BIAS_UP': 18622, 'BIAS_DOWN': 18286, 'BIAS_NEUTRAL': 66382}, device=cuda
2026-04-24 06:12:02,434 INFO RegimeClassifier: sample weights — mean=0.360  ambiguous(<0.4)=69.5%
2026-04-24 06:12:02,719 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-04-24 06:12:02,719 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-24 06:12:07,766 INFO Regime epoch  1/50 — tr=0.5252 va=1.3167 acc=0.365 per_class={'BIAS_UP': 0.387, 'BIAS_DOWN': 0.403, 'BIAS_NEUTRAL': 0.347}
2026-04-24 06:12:07,954 INFO Regime epoch  2/50 — tr=0.5157 va=1.3148 acc=0.310
2026-04-24 06:12:08,143 INFO Regime epoch  3/50 — tr=0.4986 va=1.2803 acc=0.343
2026-04-24 06:12:08,326 INFO Regime epoch  4/50 — tr=0.4685 va=1.2110 acc=0.413
2026-04-24 06:12:08,521 INFO Regime epoch  5/50 — tr=0.4356 va=1.1248 acc=0.461 per_class={'BIAS_UP': 0.959, 'BIAS_DOWN': 0.937, 'BIAS_NEUTRAL': 0.163}
2026-04-24 06:12:08,691 INFO Regime epoch  6/50 — tr=0.4015 va=1.0380 acc=0.482
2026-04-24 06:12:08,861 INFO Regime epoch  7/50 — tr=0.3743 va=0.9699 acc=0.485
2026-04-24 06:12:09,039 INFO Regime epoch  8/50 — tr=0.3576 va=0.9298 acc=0.491
2026-04-24 06:12:09,218 INFO Regime epoch  9/50 — tr=0.3483 va=0.9081 acc=0.502
2026-04-24 06:12:09,406 INFO Regime epoch 10/50 — tr=0.3422 va=0.8974 acc=0.505 per_class={'BIAS_UP': 0.988, 'BIAS_DOWN': 0.997, 'BIAS_NEUTRAL': 0.208}
2026-04-24 06:12:09,593 INFO Regime epoch 11/50 — tr=0.3371 va=0.8925 acc=0.511
2026-04-24 06:12:09,768 INFO Regime epoch 12/50 — tr=0.3332 va=0.8898 acc=0.513
2026-04-24 06:12:09,941 INFO Regime epoch 13/50 — tr=0.3303 va=0.8891 acc=0.514
2026-04-24 06:12:10,119 INFO Regime epoch 14/50 — tr=0.3273 va=0.8859 acc=0.519
2026-04-24 06:12:10,303 INFO Regime epoch 15/50 — tr=0.3253 va=0.8840 acc=0.520 per_class={'BIAS_UP': 0.992, 'BIAS_DOWN': 0.998, 'BIAS_NEUTRAL': 0.23}
2026-04-24 06:12:10,481 INFO Regime epoch 16/50 — tr=0.3229 va=0.8804 acc=0.526
2026-04-24 06:12:10,662 INFO Regime epoch 17/50 — tr=0.3210 va=0.8804 acc=0.530
2026-04-24 06:12:10,838 INFO Regime epoch 18/50 — tr=0.3199 va=0.8776 acc=0.534
2026-04-24 06:12:11,019 INFO Regime epoch 19/50 — tr=0.3185 va=0.8776 acc=0.533
2026-04-24 06:12:11,214 INFO Regime epoch 20/50 — tr=0.3176 va=0.8754 acc=0.537 per_class={'BIAS_UP': 0.997, 'BIAS_DOWN': 0.998, 'BIAS_NEUTRAL': 0.255}
2026-04-24 06:12:11,388 INFO Regime epoch 21/50 — tr=0.3165 va=0.8748 acc=0.539
2026-04-24 06:12:11,559 INFO Regime epoch 22/50 — tr=0.3160 va=0.8751 acc=0.539
2026-04-24 06:12:11,744 INFO Regime epoch 23/50 — tr=0.3148 va=0.8762 acc=0.540
2026-04-24 06:12:11,916 INFO Regime epoch 24/50 — tr=0.3141 va=0.8753 acc=0.542
2026-04-24 06:12:12,103 INFO Regime epoch 25/50 — tr=0.3134 va=0.8744 acc=0.544 per_class={'BIAS_UP': 0.998, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.267}
2026-04-24 06:12:12,274 INFO Regime epoch 26/50 — tr=0.3128 va=0.8736 acc=0.544
2026-04-24 06:12:12,445 INFO Regime epoch 27/50 — tr=0.3124 va=0.8730 acc=0.547
2026-04-24 06:12:12,616 INFO Regime epoch 28/50 — tr=0.3121 va=0.8715 acc=0.549
2026-04-24 06:12:12,791 INFO Regime epoch 29/50 — tr=0.3114 va=0.8726 acc=0.549
2026-04-24 06:12:12,976 INFO Regime epoch 30/50 — tr=0.3113 va=0.8726 acc=0.548 per_class={'BIAS_UP': 0.998, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.272}
2026-04-24 06:12:13,149 INFO Regime epoch 31/50 — tr=0.3107 va=0.8726 acc=0.546
2026-04-24 06:12:13,318 INFO Regime epoch 32/50 — tr=0.3108 va=0.8726 acc=0.547
2026-04-24 06:12:13,491 INFO Regime epoch 33/50 — tr=0.3106 va=0.8714 acc=0.551
2026-04-24 06:12:13,666 INFO Regime epoch 34/50 — tr=0.3103 va=0.8730 acc=0.549
2026-04-24 06:12:13,856 INFO Regime epoch 35/50 — tr=0.3101 va=0.8734 acc=0.547 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.271}
2026-04-24 06:12:14,030 INFO Regime epoch 36/50 — tr=0.3102 va=0.8708 acc=0.547
2026-04-24 06:12:14,199 INFO Regime epoch 37/50 — tr=0.3098 va=0.8722 acc=0.552
2026-04-24 06:12:14,368 INFO Regime epoch 38/50 — tr=0.3102 va=0.8714 acc=0.547
2026-04-24 06:12:14,540 INFO Regime epoch 39/50 — tr=0.3099 va=0.8705 acc=0.551
2026-04-24 06:12:14,725 INFO Regime epoch 40/50 — tr=0.3097 va=0.8724 acc=0.550 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.275}
2026-04-24 06:12:14,894 INFO Regime epoch 41/50 — tr=0.3097 va=0.8725 acc=0.551
2026-04-24 06:12:15,068 INFO Regime epoch 42/50 — tr=0.3093 va=0.8720 acc=0.549
2026-04-24 06:12:15,237 INFO Regime epoch 43/50 — tr=0.3094 va=0.8712 acc=0.549
2026-04-24 06:12:15,418 INFO Regime epoch 44/50 — tr=0.3095 va=0.8709 acc=0.551
2026-04-24 06:12:15,604 INFO Regime epoch 45/50 — tr=0.3093 va=0.8711 acc=0.549 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.274}
2026-04-24 06:12:15,787 INFO Regime epoch 46/50 — tr=0.3094 va=0.8721 acc=0.552
2026-04-24 06:12:15,969 INFO Regime epoch 47/50 — tr=0.3092 va=0.8723 acc=0.548
2026-04-24 06:12:16,156 INFO Regime epoch 48/50 — tr=0.3094 va=0.8732 acc=0.549
2026-04-24 06:12:16,325 INFO Regime epoch 49/50 — tr=0.3092 va=0.8712 acc=0.549
2026-04-24 06:12:16,326 INFO Regime early stop at epoch 49 (no_improve=10)
2026-04-24 06:12:16,339 WARNING RegimeClassifier accuracy 0.55 < 0.65 threshold
2026-04-24 06:12:16,343 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-24 06:12:16,344 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-24 06:12:16,471 INFO Regime HTF complete: acc=0.551, n=103290
2026-04-24 06:12:16,473 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 06:12:16,634 INFO Rule labels LTF_BEHAVIOUR [4H]: {'TRENDING': 8970, 'RANGING': 4592, 'CONSOLIDATING': 2247, 'VOLATILE': 4008}  ambiguous=4539 (total=19817)  short_runs_zeroed=570
2026-04-24 06:12:16,640 INFO Regime[4H] persistence (avg bars/run) on XAUUSD 4H:
{0: 12.121621621621621, 1: 6.4858757062146895, 2: 5.575682382133995, 3: 10.329896907216495}
2026-04-24 06:12:16,641 INFO Regime[4H] return separation on XAUUSD 4H:
{0: 5.9502411776840314e-05, 1: 5.664222849552333e-05, 2: -6.138442068010657e-05, 3: 4.239843820627722e-05}
2026-04-24 06:12:16,642 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-24 06:12:16,643 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:12:16,645 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:12:16,647 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:12:16,648 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:12:16,650 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:12:16,652 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:12:16,653 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:12:16,655 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:12:16,656 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:12:16,658 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:12:16,661 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 06:12:16,670 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:16,672 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:16,673 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:16,673 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:16,673 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:16,675 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:12:26,721 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-24 06:12:26,723 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-24 06:12:26,861 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:26,863 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:26,864 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:26,865 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:26,865 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:26,867 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:12:36,829 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-24 06:12:36,832 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-24 06:12:36,974 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:36,976 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:36,977 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:36,978 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:36,978 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:36,980 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:12:46,957 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-24 06:12:46,960 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-24 06:12:47,097 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:47,102 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:47,102 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:47,103 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:47,103 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:47,105 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:12:57,198 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-24 06:12:57,201 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-24 06:12:57,369 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:57,371 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:57,372 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:57,373 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:57,373 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:12:57,375 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:13:08,034 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-24 06:13:08,037 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-24 06:13:08,177 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:13:08,181 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:13:08,182 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:13:08,183 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:13:08,183 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:13:08,185 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:13:18,655 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-24 06:13:18,658 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-24 06:13:18,794 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-24 06:13:18,795 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-24 06:13:18,796 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-24 06:13:18,796 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-24 06:13:18,797 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-24 06:13:18,798 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:13:29,052 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-24 06:13:29,055 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-24 06:13:29,197 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:13:29,199 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:13:29,200 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:13:29,200 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:13:29,200 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:13:29,202 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:13:39,679 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-24 06:13:39,682 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-24 06:13:39,826 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:13:39,829 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:13:39,830 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:13:39,830 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:13:39,830 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:13:39,832 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:13:50,925 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-24 06:13:50,928 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-24 06:13:51,089 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:13:51,091 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:13:51,092 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:13:51,093 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:13:51,093 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:13:51,096 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:14:02,794 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-24 06:14:02,798 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-24 06:14:02,986 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-24 06:14:02,990 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-24 06:14:02,991 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-24 06:14:02,992 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-24 06:14:02,992 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-24 06:14:02,996 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 06:14:30,154 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-24 06:14:30,161 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-24 06:14:30,620 INFO RegimeClassifier[mode=ltf_behaviour]: 401471 samples, classes={'TRENDING': 160094, 'RANGING': 105286, 'CONSOLIDATING': 53524, 'VOLATILE': 82567}, device=cuda
2026-04-24 06:14:30,621 INFO RegimeClassifier: sample weights — mean=0.505  ambiguous(<0.4)=33.0%
2026-04-24 06:14:30,623 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-04-24 06:14:30,624 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-24 06:14:31,582 INFO Regime epoch  1/50 — tr=0.8362 va=2.1081 acc=0.313 per_class={'TRENDING': 0.43, 'RANGING': 0.146, 'CONSOLIDATING': 0.547, 'VOLATILE': 0.141}
2026-04-24 06:14:32,483 INFO Regime epoch  2/50 — tr=0.7916 va=1.9947 acc=0.421
2026-04-24 06:14:33,372 INFO Regime epoch  3/50 — tr=0.7323 va=1.9222 acc=0.445
2026-04-24 06:14:34,218 INFO Regime epoch  4/50 — tr=0.6808 va=1.8790 acc=0.460
2026-04-24 06:14:35,184 INFO Regime epoch  5/50 — tr=0.6485 va=1.8458 acc=0.481 per_class={'TRENDING': 0.374, 'RANGING': 0.091, 'CONSOLIDATING': 0.84, 'VOLATILE': 0.961}
2026-04-24 06:14:36,046 INFO Regime epoch  6/50 — tr=0.6303 va=1.8269 acc=0.499
2026-04-24 06:14:36,920 INFO Regime epoch  7/50 — tr=0.6176 va=1.8056 acc=0.516
2026-04-24 06:14:37,852 INFO Regime epoch  8/50 — tr=0.6085 va=1.7935 acc=0.533
2026-04-24 06:14:38,720 INFO Regime epoch  9/50 — tr=0.6014 va=1.7794 acc=0.548
2026-04-24 06:14:39,643 INFO Regime epoch 10/50 — tr=0.5953 va=1.7582 acc=0.571 per_class={'TRENDING': 0.526, 'RANGING': 0.181, 'CONSOLIDATING': 0.871, 'VOLATILE': 0.969}
2026-04-24 06:14:40,559 INFO Regime epoch 11/50 — tr=0.5904 va=1.7394 acc=0.587
2026-04-24 06:14:41,416 INFO Regime epoch 12/50 — tr=0.5863 va=1.7212 acc=0.601
2026-04-24 06:14:42,317 INFO Regime epoch 13/50 — tr=0.5827 va=1.6998 acc=0.616
2026-04-24 06:14:43,196 INFO Regime epoch 14/50 — tr=0.5795 va=1.6874 acc=0.627
2026-04-24 06:14:44,173 INFO Regime epoch 15/50 — tr=0.5768 va=1.6775 acc=0.634 per_class={'TRENDING': 0.647, 'RANGING': 0.226, 'CONSOLIDATING': 0.904, 'VOLATILE': 0.962}
2026-04-24 06:14:45,066 INFO Regime epoch 16/50 — tr=0.5741 va=1.6632 acc=0.643
2026-04-24 06:14:45,927 INFO Regime epoch 17/50 — tr=0.5722 va=1.6537 acc=0.651
2026-04-24 06:14:46,771 INFO Regime epoch 18/50 — tr=0.5702 va=1.6410 acc=0.658
2026-04-24 06:14:47,676 INFO Regime epoch 19/50 — tr=0.5684 va=1.6388 acc=0.660
2026-04-24 06:14:48,624 INFO Regime epoch 20/50 — tr=0.5669 va=1.6278 acc=0.666 per_class={'TRENDING': 0.691, 'RANGING': 0.272, 'CONSOLIDATING': 0.928, 'VOLATILE': 0.955}
2026-04-24 06:14:49,477 INFO Regime epoch 21/50 — tr=0.5656 va=1.6235 acc=0.670
2026-04-24 06:14:50,305 INFO Regime epoch 22/50 — tr=0.5645 va=1.6174 acc=0.674
2026-04-24 06:14:51,159 INFO Regime epoch 23/50 — tr=0.5630 va=1.6126 acc=0.678
2026-04-24 06:14:52,003 INFO Regime epoch 24/50 — tr=0.5623 va=1.6085 acc=0.679
2026-04-24 06:14:52,922 INFO Regime epoch 25/50 — tr=0.5617 va=1.6091 acc=0.678 per_class={'TRENDING': 0.711, 'RANGING': 0.281, 'CONSOLIDATING': 0.946, 'VOLATILE': 0.952}
2026-04-24 06:14:53,777 INFO Regime epoch 26/50 — tr=0.5609 va=1.6029 acc=0.684
2026-04-24 06:14:54,642 INFO Regime epoch 27/50 — tr=0.5597 va=1.5990 acc=0.687
2026-04-24 06:14:55,499 INFO Regime epoch 28/50 — tr=0.5597 va=1.5972 acc=0.687
2026-04-24 06:14:56,360 INFO Regime epoch 29/50 — tr=0.5587 va=1.5983 acc=0.687
2026-04-24 06:14:57,299 INFO Regime epoch 30/50 — tr=0.5584 va=1.5924 acc=0.691 per_class={'TRENDING': 0.734, 'RANGING': 0.297, 'CONSOLIDATING': 0.952, 'VOLATILE': 0.944}
2026-04-24 06:14:58,160 INFO Regime epoch 31/50 — tr=0.5580 va=1.5934 acc=0.691
2026-04-24 06:14:59,010 INFO Regime epoch 32/50 — tr=0.5577 va=1.5891 acc=0.692
2026-04-24 06:14:59,874 INFO Regime epoch 33/50 — tr=0.5572 va=1.5902 acc=0.692
2026-04-24 06:15:00,736 INFO Regime epoch 34/50 — tr=0.5572 va=1.5898 acc=0.692
2026-04-24 06:15:01,683 INFO Regime epoch 35/50 — tr=0.5568 va=1.5856 acc=0.694 per_class={'TRENDING': 0.739, 'RANGING': 0.301, 'CONSOLIDATING': 0.954, 'VOLATILE': 0.943}
2026-04-24 06:15:02,544 INFO Regime epoch 36/50 — tr=0.5567 va=1.5880 acc=0.692
2026-04-24 06:15:03,403 INFO Regime epoch 37/50 — tr=0.5564 va=1.5841 acc=0.695
2026-04-24 06:15:04,293 INFO Regime epoch 38/50 — tr=0.5560 va=1.5864 acc=0.694
2026-04-24 06:15:05,165 INFO Regime epoch 39/50 — tr=0.5559 va=1.5855 acc=0.694
2026-04-24 06:15:06,095 INFO Regime epoch 40/50 — tr=0.5557 va=1.5818 acc=0.698 per_class={'TRENDING': 0.749, 'RANGING': 0.301, 'CONSOLIDATING': 0.956, 'VOLATILE': 0.94}
2026-04-24 06:15:06,955 INFO Regime epoch 41/50 — tr=0.5556 va=1.5867 acc=0.695
2026-04-24 06:15:07,850 INFO Regime epoch 42/50 — tr=0.5556 va=1.5857 acc=0.694
2026-04-24 06:15:08,725 INFO Regime epoch 43/50 — tr=0.5556 va=1.5820 acc=0.695
2026-04-24 06:15:09,597 INFO Regime epoch 44/50 — tr=0.5554 va=1.5819 acc=0.698
2026-04-24 06:15:10,508 INFO Regime epoch 45/50 — tr=0.5557 va=1.5858 acc=0.695 per_class={'TRENDING': 0.744, 'RANGING': 0.297, 'CONSOLIDATING': 0.955, 'VOLATILE': 0.944}
2026-04-24 06:15:11,372 INFO Regime epoch 46/50 — tr=0.5558 va=1.5854 acc=0.694
2026-04-24 06:15:12,239 INFO Regime epoch 47/50 — tr=0.5554 va=1.5811 acc=0.697
2026-04-24 06:15:13,104 INFO Regime epoch 48/50 — tr=0.5557 va=1.5849 acc=0.695
2026-04-24 06:15:13,970 INFO Regime epoch 49/50 — tr=0.5555 va=1.5851 acc=0.696
2026-04-24 06:15:14,870 INFO Regime epoch 50/50 — tr=0.5553 va=1.5825 acc=0.696 per_class={'TRENDING': 0.745, 'RANGING': 0.302, 'CONSOLIDATING': 0.955, 'VOLATILE': 0.941}
2026-04-24 06:15:14,928 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-24 06:15:14,928 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-24 06:15:15,098 INFO Regime LTF complete: acc=0.697, n=401471
2026-04-24 06:15:15,103 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 06:15:15,710 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-24 06:15:15,716 INFO Regime[1H] persistence (avg bars/run) on XAUUSD 1H:
{0: 9.355291913830783, 1: 6.096303199751476, 2: 5.598885793871866, 3: 6.771351107094442}
2026-04-24 06:15:15,720 INFO Regime[1H] return separation on XAUUSD 1H:
{0: 2.149062259083323e-05, 1: -3.254493608872846e-06, 2: -3.039756158146958e-06, 3: 1.8102764458593193e-05}
2026-04-24 06:15:15,736 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-24 06:15:15,737 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 06:15:15,737 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 06:15:15,737 INFO === VectorStore: building similarity indices ===
2026-04-24 06:15:15,742 INFO Loading faiss with AVX512 support.
2026-04-24 06:15:15,773 INFO Successfully loaded faiss with AVX512 support.
2026-04-24 06:15:15,782 INFO VectorStore: CPU FAISS index built (dim=45)
2026-04-24 06:15:15,782 INFO VectorStore: CPU FAISS index built (dim=53)
2026-04-24 06:15:15,782 INFO VectorStore: CPU FAISS index built (dim=64)
2026-04-24 06:15:15,879 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-24 06:15:15,886 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:15:23,469 WARNING VectorStore trade_patterns failed for AUDUSD: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 06:15:23,476 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:15:23,479 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:15:23,480 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:15:23,481 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:15:23,481 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:16:08,909 WARNING VectorStore market_structures failed for AUDUSD: name 'np' is not defined
2026-04-24 06:16:15,308 WARNING VectorStore regime_embeddings failed for AUDUSD: name 'np' is not defined
2026-04-24 06:16:15,477 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:16:22,487 WARNING VectorStore trade_patterns failed for EURGBP: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 06:16:22,494 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:16:22,496 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:16:22,497 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:16:22,497 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:16:22,498 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:17:06,473 WARNING VectorStore market_structures failed for EURGBP: name 'np' is not defined
2026-04-24 06:17:13,233 WARNING VectorStore regime_embeddings failed for EURGBP: name 'np' is not defined
2026-04-24 06:17:13,409 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:17:20,703 WARNING VectorStore trade_patterns failed for EURJPY: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 06:17:20,710 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:17:20,712 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:17:20,713 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:17:20,714 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:17:20,714 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:18:06,073 WARNING VectorStore market_structures failed for EURJPY: name 'np' is not defined
2026-04-24 06:18:12,777 WARNING VectorStore regime_embeddings failed for EURJPY: name 'np' is not defined
2026-04-24 06:18:12,946 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:18:20,123 WARNING VectorStore trade_patterns failed for EURUSD: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 06:18:20,130 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:18:20,132 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:18:20,133 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:18:20,134 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:18:20,134 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:19:06,224 WARNING VectorStore market_structures failed for EURUSD: name 'np' is not defined
2026-04-24 06:19:12,698 WARNING VectorStore regime_embeddings failed for EURUSD: name 'np' is not defined
2026-04-24 06:19:12,868 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:19:19,944 WARNING VectorStore trade_patterns failed for GBPJPY: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 06:19:19,950 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:19:19,952 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:19:19,953 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:19:19,954 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:19:19,954 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:20:05,282 WARNING VectorStore market_structures failed for GBPJPY: name 'np' is not defined
2026-04-24 06:20:11,866 WARNING VectorStore regime_embeddings failed for GBPJPY: name 'np' is not defined
2026-04-24 06:20:12,034 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:20:19,069 WARNING VectorStore trade_patterns failed for GBPUSD: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 06:20:19,076 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:20:19,078 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:20:19,079 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:20:19,080 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:20:19,080 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:21:04,577 WARNING VectorStore market_structures failed for GBPUSD: name 'np' is not defined
2026-04-24 06:21:11,025 WARNING VectorStore regime_embeddings failed for GBPUSD: name 'np' is not defined
2026-04-24 06:21:11,198 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:21:18,300 WARNING VectorStore trade_patterns failed for NZDUSD: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 06:21:18,304 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-24 06:21:18,306 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-24 06:21:18,307 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-24 06:21:18,307 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-24 06:21:18,308 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-24 06:22:03,620 WARNING VectorStore market_structures failed for NZDUSD: name 'np' is not defined
2026-04-24 06:22:10,054 WARNING VectorStore regime_embeddings failed for NZDUSD: name 'np' is not defined
2026-04-24 06:22:10,225 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:22:17,213 WARNING VectorStore trade_patterns failed for USDCAD: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 06:22:17,219 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:22:17,222 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:22:17,223 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:22:17,223 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:22:17,224 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:23:02,628 WARNING VectorStore market_structures failed for USDCAD: name 'np' is not defined
2026-04-24 06:23:09,244 WARNING VectorStore regime_embeddings failed for USDCAD: name 'np' is not defined
2026-04-24 06:23:09,416 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:23:16,787 WARNING VectorStore trade_patterns failed for USDCHF: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 06:23:16,793 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:23:16,795 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:23:16,796 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:23:16,797 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:23:16,797 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:24:03,068 WARNING VectorStore market_structures failed for USDCHF: name 'np' is not defined
2026-04-24 06:24:09,563 WARNING VectorStore regime_embeddings failed for USDCHF: name 'np' is not defined
2026-04-24 06:24:09,734 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:24:16,958 WARNING VectorStore trade_patterns failed for USDJPY: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 06:24:16,965 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:24:16,967 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:24:16,968 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:24:16,968 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:24:16,969 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:25:01,850 WARNING VectorStore market_structures failed for USDJPY: name 'np' is not defined
2026-04-24 06:25:08,368 WARNING VectorStore regime_embeddings failed for USDJPY: name 'np' is not defined
2026-04-24 06:25:08,545 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-24 06:25:23,701 WARNING VectorStore trade_patterns failed for XAUUSD: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 06:25:23,711 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-24 06:25:23,715 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-24 06:25:23,716 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-24 06:25:23,717 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-24 06:25:23,717 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-24 06:27:07,496 WARNING VectorStore market_structures failed for XAUUSD: name 'np' is not defined
2026-04-24 06:27:21,838 WARNING VectorStore regime_embeddings failed for XAUUSD: name 'np' is not defined
2026-04-24 06:27:22,003 INFO VectorStore: saved 0 total vectors to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/vector_store
2026-04-24 06:27:22,003 INFO VectorStore saved: {'trade_patterns': 0, 'market_structures': 0, 'regime_embeddings': 0}
2026-04-24 06:27:22,025 INFO Retrain complete.
2026-04-24 06:27:25,031 INFO Model regime: SUCCESS
2026-04-24 06:27:25,031 INFO --- Training gru ---
2026-04-24 06:27:25,031 INFO Running retrain --model gru
2026-04-24 06:27:25,553 INFO retrain environment: KAGGLE
2026-04-24 06:27:27,495 INFO Device: CUDA (2 GPU(s))
2026-04-24 06:27:27,506 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 06:27:27,506 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 06:27:27,507 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-24 06:27:27,656 INFO NumExpr defaulting to 4 threads.
2026-04-24 06:27:27,874 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-24 06:27:27,874 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 06:27:27,874 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 06:27:28,122 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-24 06:27:28,379 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-24 06:27:28,380 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:28,465 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:28,549 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:28,639 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:28,718 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:28,799 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:28,881 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:28,968 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:29,051 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:29,134 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:29,239 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 06:27:29,313 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-24 06:27:29,314 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260424_062729
2026-04-24 06:27:29,317 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-04-24 06:27:29,452 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:29,453 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:29,469 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:29,487 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:29,489 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-24 06:27:29,489 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 06:27:29,489 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 06:27:29,490 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:29,578 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-24 06:27:29,580 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:29,845 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-24 06:27:29,878 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:30,212 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:30,365 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:30,482 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:30,723 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:30,724 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:30,741 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:30,750 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:30,751 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:30,836 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-24 06:27:30,838 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:31,111 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-24 06:27:31,128 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:31,444 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:31,599 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:31,711 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:31,950 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:31,951 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:31,972 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:31,982 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:31,983 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:32,071 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-24 06:27:32,074 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:32,317 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-24 06:27:32,333 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:32,644 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:32,784 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:32,903 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:33,148 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:33,149 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:33,166 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:33,175 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:33,176 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:33,263 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-24 06:27:33,265 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:33,523 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-24 06:27:33,547 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:33,867 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:34,017 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:34,132 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:34,360 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:34,361 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:34,379 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:34,388 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:34,389 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:34,472 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-24 06:27:34,474 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:34,733 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-24 06:27:34,749 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:35,061 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:35,218 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:35,328 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:35,551 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:35,552 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:35,569 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:35,579 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:35,580 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:35,668 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-24 06:27:35,670 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:35,937 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-24 06:27:35,953 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:36,268 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:36,429 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:36,544 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:36,741 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-24 06:27:36,742 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-24 06:27:36,760 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-24 06:27:36,769 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-24 06:27:36,770 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:36,853 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-24 06:27:36,855 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:37,125 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-24 06:27:37,138 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:37,456 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:37,600 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:37,713 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:37,920 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:37,921 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:37,940 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:37,950 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:37,951 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:38,034 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-24 06:27:38,036 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:38,309 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-24 06:27:38,329 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:38,643 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:38,783 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:38,907 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:39,117 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:39,118 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:39,137 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:39,147 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:39,148 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:39,232 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-24 06:27:39,234 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:39,505 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-24 06:27:39,520 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:39,830 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:39,975 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:40,109 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:40,313 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:40,314 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:40,336 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:40,346 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:27:40,347 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:40,438 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-24 06:27:40,440 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:40,706 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-24 06:27:40,721 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:41,037 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:41,182 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:41,303 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:27:41,620 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-24 06:27:41,621 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-24 06:27:41,641 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-24 06:27:41,653 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-24 06:27:41,654 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 06:27:41,828 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-24 06:27:41,832 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 06:27:42,388 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-24 06:27:42,437 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-24 06:27:43,064 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-24 06:27:43,305 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 06:27:43,450 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 06:27:43,594 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-24 06:27:43,594 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-24 06:27:43,594 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-24 06:32:41,999 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-24 06:32:42,000 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-24 06:32:43,396 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-24 06:33:10,923 INFO train_multi TF=ALL epoch 1/50 train=0.6036 val=0.6134
2026-04-24 06:33:10,928 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-24 06:33:10,929 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-24 06:33:10,929 INFO train_multi TF=ALL: new best val=0.6134 — saved
2026-04-24 06:33:31,409 INFO train_multi TF=ALL epoch 2/50 train=0.6036 val=0.6137
2026-04-24 06:33:50,556 INFO train_multi TF=ALL epoch 3/50 train=0.6033 val=0.6136
2026-04-24 06:34:10,419 INFO train_multi TF=ALL epoch 4/50 train=0.6034 val=0.6135
2026-04-24 06:34:30,156 INFO train_multi TF=ALL epoch 5/50 train=0.6030 val=0.6140
2026-04-24 06:34:50,171 INFO train_multi TF=ALL epoch 6/50 train=0.6026 val=0.6144
2026-04-24 06:34:50,171 INFO train_multi TF=ALL early stop at epoch 6
2026-04-24 06:34:50,323 INFO === VectorStore: building similarity indices ===
2026-04-24 06:34:50,328 INFO Loading faiss with AVX512 support.
2026-04-24 06:34:50,351 INFO Successfully loaded faiss with AVX512 support.
2026-04-24 06:34:50,357 INFO VectorStore: CPU FAISS index built (dim=45)
2026-04-24 06:34:50,357 INFO VectorStore: CPU FAISS index built (dim=53)
2026-04-24 06:34:50,357 INFO VectorStore: CPU FAISS index built (dim=64)
2026-04-24 06:34:50,371 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-24 06:34:50,378 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:34:56,815 WARNING VectorStore trade_patterns failed for AUDUSD: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 06:34:56,822 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:34:56,824 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:34:56,825 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:34:56,826 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:34:56,826 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:35:40,853 WARNING VectorStore market_structures failed for AUDUSD: name 'np' is not defined
2026-04-24 06:35:47,086 WARNING VectorStore regime_embeddings failed for AUDUSD: name 'np' is not defined
2026-04-24 06:35:47,225 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:35:53,863 WARNING VectorStore trade_patterns failed for EURGBP: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 06:35:53,869 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:35:53,871 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:35:53,872 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:35:53,873 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:35:53,873 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:36:39,209 WARNING VectorStore market_structures failed for EURGBP: name 'np' is not defined
2026-04-24 06:36:45,329 WARNING VectorStore regime_embeddings failed for EURGBP: name 'np' is not defined
2026-04-24 06:36:45,470 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:36:52,012 WARNING VectorStore trade_patterns failed for EURJPY: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 06:36:52,019 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:36:52,021 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:36:52,022 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:36:52,023 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:36:52,023 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:37:36,641 WARNING VectorStore market_structures failed for EURJPY: name 'np' is not defined
2026-04-24 06:37:42,875 WARNING VectorStore regime_embeddings failed for EURJPY: name 'np' is not defined
2026-04-24 06:37:43,014 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:37:49,905 WARNING VectorStore trade_patterns failed for EURUSD: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 06:37:49,912 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:37:49,914 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:37:49,915 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:37:49,916 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:37:49,916 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:38:33,865 WARNING VectorStore market_structures failed for EURUSD: name 'np' is not defined
2026-04-24 06:38:40,091 WARNING VectorStore regime_embeddings failed for EURUSD: name 'np' is not defined
2026-04-24 06:38:40,213 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:38:46,761 WARNING VectorStore trade_patterns failed for GBPJPY: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 06:38:46,768 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:38:46,770 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:38:46,771 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:38:46,772 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:38:46,772 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:39:31,119 WARNING VectorStore market_structures failed for GBPJPY: name 'np' is not defined
2026-04-24 06:39:37,097 WARNING VectorStore regime_embeddings failed for GBPJPY: name 'np' is not defined
2026-04-24 06:39:37,227 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:39:44,166 WARNING VectorStore trade_patterns failed for GBPUSD: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 06:39:44,172 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:39:44,175 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:39:44,175 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:39:44,176 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:39:44,176 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:40:28,081 WARNING VectorStore market_structures failed for GBPUSD: name 'np' is not defined
2026-04-24 06:40:34,193 WARNING VectorStore regime_embeddings failed for GBPUSD: name 'np' is not defined
2026-04-24 06:40:34,324 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:40:40,973 WARNING VectorStore trade_patterns failed for NZDUSD: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 06:40:40,978 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-24 06:40:40,980 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-24 06:40:40,981 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-24 06:40:40,982 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-24 06:40:40,982 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-24 06:41:24,800 WARNING VectorStore market_structures failed for NZDUSD: name 'np' is not defined
2026-04-24 06:41:30,998 WARNING VectorStore regime_embeddings failed for NZDUSD: name 'np' is not defined
2026-04-24 06:41:31,139 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:41:37,904 WARNING VectorStore trade_patterns failed for USDCAD: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 06:41:37,911 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:41:37,913 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:41:37,914 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:41:37,915 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:41:37,915 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:42:21,726 WARNING VectorStore market_structures failed for USDCAD: name 'np' is not defined
2026-04-24 06:42:27,767 WARNING VectorStore regime_embeddings failed for USDCAD: name 'np' is not defined
2026-04-24 06:42:27,897 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:42:34,421 WARNING VectorStore trade_patterns failed for USDCHF: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 06:42:34,427 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:42:34,429 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:42:34,430 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:42:34,431 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:42:34,431 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:43:18,804 WARNING VectorStore market_structures failed for USDCHF: name 'np' is not defined
2026-04-24 06:43:25,116 WARNING VectorStore regime_embeddings failed for USDCHF: name 'np' is not defined
2026-04-24 06:43:25,251 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-24 06:43:31,768 WARNING VectorStore trade_patterns failed for USDJPY: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 06:43:31,775 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:43:31,777 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:43:31,778 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:43:31,778 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:43:31,779 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 06:44:16,246 WARNING VectorStore market_structures failed for USDJPY: name 'np' is not defined
2026-04-24 06:44:22,263 WARNING VectorStore regime_embeddings failed for USDJPY: name 'np' is not defined
2026-04-24 06:44:22,385 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-24 06:44:36,028 WARNING VectorStore trade_patterns failed for XAUUSD: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 06:44:36,038 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-24 06:44:36,042 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-24 06:44:36,044 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-24 06:44:36,044 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-24 06:44:36,044 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-24 06:46:18,403 WARNING VectorStore market_structures failed for XAUUSD: name 'np' is not defined
2026-04-24 06:46:32,516 WARNING VectorStore regime_embeddings failed for XAUUSD: name 'np' is not defined
2026-04-24 06:46:32,643 INFO VectorStore: saved 0 total vectors to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/vector_store
2026-04-24 06:46:32,643 INFO VectorStore saved: {'trade_patterns': 0, 'market_structures': 0, 'regime_embeddings': 0}
2026-04-24 06:46:32,648 INFO Retrain complete.
2026-04-24 06:46:34,870 INFO Model gru: SUCCESS
2026-04-24 06:46:34,870 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-24 06:46:34,870 WARNING   [MISSING] regime_classifier → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_classifier.pkl
2026-04-24 06:46:34,870 INFO   [OK] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-24 06:46:34,870 WARNING   [MISSING] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-24 06:46:34,870 WARNING Missing weights: ['regime_classifier', 'rl_ppo'] — run retrain_incremental.py for each
2026-04-24 06:46:34,871 INFO Saved 18 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  Cleared journal for fresh Round 1 run
  START Round 1 - Backtest (val)
2026-04-24 06:46:35,589 INFO === STEP 6: BACKTEST + REINFORCED TRAINING (3 rounds, window=round1) ===
2026-04-24 06:46:35,590 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-04-24 06:46:35,590 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-24 06:46:35,590 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260424_064636.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)             2309  53.8%   2.97 2272.1% 53.8% 16.7%   2.5%    5.11
  gate_diagnostics: bars=466496 no_signal=56678 quality_block=0 session_skip=190468 density=4515 pm_reject=26570 daily_skip=173700 cooldown=12256

Calibration Summary:
  all          [OK] Calibration OK — p_win correlates with actual win rate.
  ml_trader    [OK] Calibration OK — p_win correlates with actual win rate.
2026-04-24 06:53:35,352 INFO Round 1 backtest — 2309 trades | avg WR=53.8% | avg PF=2.97 | avg Sharpe=5.11
2026-04-24 06:53:35,352 INFO   ml_trader: 2309 trades | WR=53.8% | PF=2.97 | Return=2272.1% | DD=2.5% | Sharpe=5.11
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 2309
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2309 rows)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  AUDUSD        219 trades   183 days   1.20/day
  EURGBP        204 trades   163 days   1.25/day
  EURJPY        170 trades   129 days   1.32/day
  EURUSD        328 trades   269 days   1.22/day
  GBPJPY        165 trades   129 days   1.28/day
  GBPUSD        248 trades   207 days   1.20/day
  NZDUSD        153 trades   118 days   1.30/day
  USDCAD        212 trades   167 days   1.27/day
  USDCHF        199 trades   158 days   1.26/day
  USDJPY        255 trades   207 days   1.23/day
  XAUUSD        156 trades   124 days   1.26/day
  ✓  All symbols within normal range.
2026-04-24 06:53:36,568 INFO Round 1: wrote 2309 journal entries (total in file: 2309)
2026-04-24 06:53:36,571 INFO Round 1 — retraining regime...
2026-04-24 07:13:12,186 INFO Retrain regime: OK
2026-04-24 07:13:12,211 INFO Round 1 — retraining quality...
2026-04-24 07:13:19,709 INFO Retrain quality: OK
2026-04-24 07:13:19,729 INFO Round 1 — retraining rl...
2026-04-24 07:14:38,396 INFO Retrain rl: OK
2026-04-24 07:14:38,419 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-24 07:14:38,420 INFO Round 2 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
2026-04-24 07:21:55,733 INFO Round 2 backtest — 2245 trades | avg WR=54.6% | avg PF=3.09 | avg Sharpe=5.27
2026-04-24 07:21:55,733 INFO   ml_trader: 2245 trades | WR=54.6% | PF=3.09 | Return=2461.1% | DD=2.0% | Sharpe=5.27
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 2245
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2245 rows)
2026-04-24 07:21:56,906 INFO Round 2: wrote 2245 journal entries (total in file: 4554)
2026-04-24 07:21:56,909 INFO Round 2 — retraining regime...
2026-04-24 07:41:07,157 INFO Retrain regime: OK
2026-04-24 07:41:07,180 INFO Round 2 — retraining quality...
2026-04-24 07:41:15,366 INFO Retrain quality: OK
2026-04-24 07:41:15,391 INFO Round 2 — retraining rl...
2026-04-24 07:43:09,178 INFO Retrain rl: OK
2026-04-24 07:43:09,206 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-04-24 07:43:09,207 INFO Round 3 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
2026-04-24 07:50:01,239 INFO Round 3 backtest — 2300 trades | avg WR=54.3% | avg PF=3.14 | avg Sharpe=5.29
2026-04-24 07:50:01,240 INFO   ml_trader: 2300 trades | WR=54.3% | PF=3.14 | Return=2445.9% | DD=2.3% | Sharpe=5.29
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 2300
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2300 rows)
2026-04-24 07:50:02,283 INFO Round 3: wrote 2300 journal entries (total in file: 6854)
2026-04-24 07:50:02,285 INFO Round 3 (final): retraining after last backtest...
2026-04-24 07:50:02,285 INFO Round 3 — retraining regime...
2026-04-24 08:05:59,970 INFO Retrain regime: OK
2026-04-24 08:05:59,989 INFO Round 3 — retraining quality...
2026-04-24 08:06:07,576 INFO Retrain quality: OK
2026-04-24 08:06:07,593 INFO Round 3 — retraining rl...
2026-04-24 08:08:15,861 INFO Retrain rl: OK
2026-04-24 08:08:15,879 INFO Improvement round 1 → 3: WR +0.5% | PF +0.174 | Sharpe +0.179
2026-04-24 08:08:16,031 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-24 08:08:16,038 INFO Journal entries: 6854
2026-04-24 08:08:16,038 INFO --- Training quality ---
2026-04-24 08:08:16,038 INFO Running retrain --model quality
  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 6854 entries

=== Round 1 → Retrain Quality + RL ===
  START Round 1 - Quality+RL retrain
2026-04-24 08:08:16,412 INFO retrain environment: KAGGLE
2026-04-24 08:08:18,158 INFO Device: CUDA (2 GPU(s))
2026-04-24 08:08:18,171 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 08:08:18,171 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 08:08:18,172 INFO === QualityScorer retrain ===
2026-04-24 08:08:18,308 INFO NumExpr defaulting to 4 threads.
2026-04-24 08:08:18,506 INFO QualityScorer: CUDA available — using GPU
2026-04-24 08:08:18,709 INFO QualityScorer loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (device=cuda)
2026-04-24 08:08:18,949 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/quality_scorer.pkl_20260424_080818
2026-04-24 08:08:19,174 INFO QualityScorer: 6854 samples, EV stats={'mean': 0.24809513986110687, 'std': 1.2488632202148438, 'n_pos': 3714, 'n_neg': 3140}, device=cuda
2026-04-24 08:08:19,175 INFO QualityScorer: normalised win labels by median_win=0.844 — EV range now [-1, +3]
2026-04-24 08:08:19,175 INFO QualityScorer: warm start from existing weights
2026-04-24 08:08:19,175 INFO QualityScorer: pos_weight=1.00 (n_pos=2936 n_neg=2547)
2026-04-24 08:08:20,865 INFO Quality epoch   1/100 — va_huber=0.7616
2026-04-24 08:08:21,006 INFO Quality epoch   2/100 — va_huber=0.7623
2026-04-24 08:08:21,134 INFO Quality epoch   3/100 — va_huber=0.7624
2026-04-24 08:08:21,259 INFO Quality epoch   4/100 — va_huber=0.7622
2026-04-24 08:08:21,394 INFO Quality epoch   5/100 — va_huber=0.7618
2026-04-24 08:08:22,154 INFO Quality epoch  11/100 — va_huber=0.7643
2026-04-24 08:08:22,154 INFO Quality early stop at epoch 11
2026-04-24 08:08:22,174 INFO QualityScorer EV model: MAE=1.237 dir_acc=0.576 n_val=1371
2026-04-24 08:08:22,177 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-24 08:08:22,249 INFO Retrain complete.
2026-04-24 08:08:23,124 INFO Model quality: SUCCESS
2026-04-24 08:08:23,124 INFO --- Training rl ---
2026-04-24 08:08:23,124 INFO Running retrain --model rl
2026-04-24 08:08:23,335 INFO retrain environment: KAGGLE
2026-04-24 08:08:25,030 INFO Device: CUDA (2 GPU(s))
2026-04-24 08:08:25,041 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 08:08:25,041 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 08:08:25,042 INFO === RLAgent (PPO) retrain ===
2026-04-24 08:08:25,044 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260424_080825
2026-04-24 08:08:25.903545: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777018105.926148   47365 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777018105.934389   47365 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777018105.954187   47365 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777018105.954211   47365 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777018105.954214   47365 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777018105.954216   47365 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-24 08:08:30,375 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-24 08:08:33,210 INFO RLAgent: PPO model loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-24 08:08:33,385 INFO RLAgent: warm start — fine-tuning existing PPO policy (lr=6.00e-05)
2026-04-24 08:10:29,803 INFO RLAgent: retrain complete, 6854 episodes
2026-04-24 08:10:29,803 INFO Retrain complete.
2026-04-24 08:10:31,408 INFO Model rl: SUCCESS
2026-04-24 08:10:31,409 INFO Step 7b complete — summary: /kaggle/working/Multi-Bot/trading-system/ml_training/metrics/training_7b_summary.json
  DONE  Round 1 - Quality+RL retrain

=== Round 2: BLIND backtest on test window (unseen 2yr) ===
  START Round 2 - Blind backtest (test)
2026-04-24 08:10:31,901 INFO === STEP 6: BACKTEST + REINFORCED TRAINING (3 rounds, window=round2) ===
2026-04-24 08:10:31,901 INFO BT_WINDOW=round2 — BLIND backtest: 2023-08-07 → 2025-08-05 (test set)
2026-04-24 08:10:31,903 INFO Cleared existing journal for fresh reinforced training run
2026-04-24 08:10:31,903 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-24 08:10:31,903 INFO Round 1 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-24 08:16:58,800 INFO Round 1 backtest — 2187 trades | avg WR=50.9% | avg PF=2.45 | avg Sharpe=4.83
2026-04-24 08:16:58,800 INFO   ml_trader: 2187 trades | WR=50.9% | PF=2.45 | Return=1191.1% | DD=2.5% | Sharpe=4.83
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 2187
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2187 rows)
2026-04-24 08:16:59,787 INFO Round 1: wrote 2187 journal entries (total in file: 2187)
2026-04-24 08:16:59,789 INFO Round 1 — retraining regime...
2026-04-24 08:32:43,581 INFO Retrain regime: OK
2026-04-24 08:32:43,599 INFO Round 1 — retraining quality...
2026-04-24 08:32:49,717 INFO Retrain quality: OK
2026-04-24 08:32:49,733 INFO Round 1 — retraining rl...
2026-04-24 08:33:36,084 INFO Retrain rl: OK
2026-04-24 08:33:36,101 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-24 08:33:36,101 INFO Round 2 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-24 08:39:45,730 INFO Round 2 backtest — 2149 trades | avg WR=51.1% | avg PF=2.49 | avg Sharpe=4.90
2026-04-24 08:39:45,730 INFO   ml_trader: 2149 trades | WR=51.1% | PF=2.49 | Return=1201.4% | DD=2.9% | Sharpe=4.90
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 2149
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2149 rows)
2026-04-24 08:39:46,695 INFO Round 2: wrote 2149 journal entries (total in file: 4336)
2026-04-24 08:39:46,696 INFO Round 2 — retraining regime...
2026-04-24 08:55:45,424 INFO Retrain regime: OK
2026-04-24 08:55:45,442 INFO Round 2 — retraining quality...
2026-04-24 08:55:55,760 INFO Retrain quality: OK
2026-04-24 08:55:55,776 INFO Round 2 — retraining rl...
2026-04-24 08:57:17,769 INFO Retrain rl: OK
2026-04-24 08:57:17,787 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-04-24 08:57:17,787 INFO Round 3 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-24 09:03:31,678 INFO Round 3 backtest — 2159 trades | avg WR=51.4% | avg PF=2.52 | avg Sharpe=4.96
2026-04-24 09:03:31,678 INFO   ml_trader: 2159 trades | WR=51.4% | PF=2.52 | Return=1235.9% | DD=2.6% | Sharpe=4.96
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 2159
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2159 rows)
2026-04-24 09:03:32,653 INFO Round 3: wrote 2159 journal entries (total in file: 6495)
2026-04-24 09:03:32,655 INFO Round 3 (final): retraining after last backtest...
2026-04-24 09:03:32,655 INFO Round 3 — retraining regime...
2026-04-24 09:19:20,436 INFO Retrain regime: OK
2026-04-24 09:19:20,455 INFO Round 3 — retraining quality...
2026-04-24 09:19:34,003 INFO Retrain quality: OK
2026-04-24 09:19:34,020 INFO Round 3 — retraining rl...
2026-04-24 09:21:31,049 INFO Retrain rl: OK
2026-04-24 09:21:31,067 INFO Improvement round 1 → 3: WR +0.5% | PF +0.067 | Sharpe +0.128
2026-04-24 09:21:31,218 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-24 09:21:31,225 INFO Journal entries: 6495
2026-04-24 09:21:31,225 INFO --- Training quality ---
2026-04-24 09:21:31,226 INFO Running retrain --model quality
  DONE  Round 2 - Blind backtest (test)
  Saved Round 2 result → round2_summary.json
  Journal after Round 2: 6495 entries

=== Round 2 → Retrain Quality + RL (Round 1+2 journal) ===
  START Round 2 - Quality+RL retrain
2026-04-24 09:21:31,554 INFO retrain environment: KAGGLE
2026-04-24 09:21:33,258 INFO Device: CUDA (2 GPU(s))
2026-04-24 09:21:33,270 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 09:21:33,270 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 09:21:33,271 INFO === QualityScorer retrain ===
2026-04-24 09:21:33,408 INFO NumExpr defaulting to 4 threads.
2026-04-24 09:21:33,599 INFO QualityScorer: CUDA available — using GPU
2026-04-24 09:21:33,802 INFO QualityScorer loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (device=cuda)
2026-04-24 09:21:34,044 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/quality_scorer.pkl_20260424_092134
2026-04-24 09:21:34,268 INFO QualityScorer: 6495 samples, EV stats={'mean': 0.13979585468769073, 'std': 1.1955955028533936, 'n_pos': 3321, 'n_neg': 3174}, device=cuda
2026-04-24 09:21:34,269 INFO QualityScorer: normalised win labels by median_win=0.800 — EV range now [-1, +3]
2026-04-24 09:21:34,269 INFO QualityScorer: warm start from existing weights
2026-04-24 09:21:34,270 INFO QualityScorer: pos_weight=1.00 (n_pos=2658 n_neg=2538)
2026-04-24 09:21:35,973 INFO Quality epoch   1/100 — va_huber=0.7099
2026-04-24 09:21:36,103 INFO Quality epoch   2/100 — va_huber=0.7096
2026-04-24 09:21:36,228 INFO Quality epoch   3/100 — va_huber=0.7099
2026-04-24 09:21:36,353 INFO Quality epoch   4/100 — va_huber=0.7100
2026-04-24 09:21:36,478 INFO Quality epoch   5/100 — va_huber=0.7109
2026-04-24 09:21:37,255 INFO Quality epoch  11/100 — va_huber=0.7089
2026-04-24 09:21:38,545 INFO Quality epoch  21/100 — va_huber=0.7085
2026-04-24 09:21:39,767 INFO Quality epoch  31/100 — va_huber=0.7074
2026-04-24 09:21:40,975 INFO Quality epoch  41/100 — va_huber=0.7067
2026-04-24 09:21:42,214 INFO Quality epoch  51/100 — va_huber=0.7064
2026-04-24 09:21:43,418 INFO Quality epoch  61/100 — va_huber=0.7062
2026-04-24 09:21:44,826 INFO Quality epoch  71/100 — va_huber=0.7057
2026-04-24 09:21:44,945 INFO Quality early stop at epoch 72
2026-04-24 09:21:44,966 INFO QualityScorer EV model: MAE=1.178 dir_acc=0.584 n_val=1299
2026-04-24 09:21:44,970 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-24 09:21:45,048 INFO Retrain complete.
2026-04-24 09:21:45,880 INFO Model quality: SUCCESS
2026-04-24 09:21:45,881 INFO --- Training rl ---
2026-04-24 09:21:45,881 INFO Running retrain --model rl
2026-04-24 09:21:46,095 INFO retrain environment: KAGGLE
2026-04-24 09:21:47,815 INFO Device: CUDA (2 GPU(s))
2026-04-24 09:21:47,827 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 09:21:47,827 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 09:21:47,828 INFO === RLAgent (PPO) retrain ===
2026-04-24 09:21:47,831 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260424_092147
2026-04-24 09:21:48.660566: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777022508.683261   62017 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777022508.690789   62017 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777022508.710598   62017 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777022508.710625   62017 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777022508.710628   62017 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777022508.710630   62017 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-24 09:21:53,036 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-24 09:21:55,801 INFO RLAgent: PPO model loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-24 09:21:55,962 INFO RLAgent: warm start — fine-tuning existing PPO policy (lr=6.00e-05)
2026-04-24 09:23:44,051 INFO RLAgent: retrain complete, 6495 episodes
2026-04-24 09:23:44,052 INFO Retrain complete.
2026-04-24 09:23:45,597 INFO Model rl: SUCCESS
2026-04-24 09:23:45,598 INFO Step 7b complete — summary: /kaggle/working/Multi-Bot/trading-system/ml_training/metrics/training_7b_summary.json
  DONE  Round 2 - Quality+RL retrain

=== Round 3: Incremental retrain ALL models on full data ===
  START Retrain gru [full-data retrain]
2026-04-24 09:23:45,912 INFO retrain environment: KAGGLE
2026-04-24 09:23:47,661 INFO Device: CUDA (2 GPU(s))
2026-04-24 09:23:47,673 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 09:23:47,673 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 09:23:47,674 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-24 09:23:47,809 INFO NumExpr defaulting to 4 threads.
2026-04-24 09:23:48,002 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-24 09:23:48,002 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 09:23:48,002 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 09:23:48,237 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-24 09:23:48,468 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-24 09:23:48,470 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:48,545 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:48,620 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:48,692 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:48,764 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:48,839 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:48,908 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:48,979 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:49,051 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:49,125 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:49,214 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 09:23:49,273 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-24 09:23:49,275 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260424_092349
2026-04-24 09:23:49,279 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-04-24 09:23:49,400 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:49,401 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:49,417 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:49,425 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:49,426 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-24 09:23:49,426 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 09:23:49,426 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 09:23:49,427 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:49,505 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-24 09:23:49,507 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:49,737 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-24 09:23:49,764 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:50,046 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:50,173 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:50,268 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:50,463 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:50,464 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:50,479 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:50,487 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:50,487 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:50,559 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-24 09:23:50,561 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:50,778 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-24 09:23:50,793 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:51,059 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:51,191 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:51,289 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:51,480 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:51,481 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:51,498 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:51,506 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:51,507 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:51,582 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-24 09:23:51,584 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:51,801 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-24 09:23:51,816 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:52,078 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:52,207 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:52,299 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:52,485 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:52,486 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:52,502 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:52,509 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:52,510 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:52,580 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-24 09:23:52,582 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:52,807 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-24 09:23:52,829 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:53,095 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:53,226 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:53,324 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:53,515 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:53,516 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:53,535 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:53,544 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:53,545 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:53,622 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-24 09:23:53,624 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:53,848 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-24 09:23:53,863 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:54,130 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:54,272 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:54,383 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:54,577 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:54,578 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:54,593 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:54,603 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:54,604 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:54,675 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-24 09:23:54,677 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:54,910 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-24 09:23:54,925 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:55,203 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:55,335 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:55,434 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:55,596 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-24 09:23:55,597 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-24 09:23:55,612 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-24 09:23:55,620 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-24 09:23:55,621 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:55,693 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-24 09:23:55,695 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:55,927 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-24 09:23:55,939 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:56,200 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:56,325 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:56,421 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:56,603 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:56,604 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:56,620 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:56,627 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:56,628 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:56,704 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-24 09:23:56,706 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:56,953 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-24 09:23:56,972 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:57,249 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:57,392 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:57,494 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:57,670 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:57,670 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:57,685 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:57,693 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:57,694 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:57,764 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-24 09:23:57,766 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:58,007 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-24 09:23:58,021 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:58,292 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:58,420 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:58,515 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:58,697 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:58,698 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:58,715 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:58,723 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:23:58,724 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:58,798 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-24 09:23:58,800 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:59,030 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-24 09:23:59,045 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:59,306 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:59,442 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:59,548 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:23:59,836 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-24 09:23:59,837 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-24 09:23:59,854 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-24 09:23:59,864 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-24 09:23:59,865 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 09:24:00,014 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-24 09:24:00,017 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 09:24:00,505 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-24 09:24:00,550 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-24 09:24:01,113 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-24 09:24:01,322 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 09:24:01,465 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 09:24:01,587 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-24 09:24:01,587 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-24 09:24:01,587 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-24 09:28:28,810 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-24 09:28:28,810 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-24 09:28:30,127 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-24 09:28:53,646 INFO train_multi TF=ALL epoch 1/50 train=0.6034 val=0.6136
2026-04-24 09:28:53,650 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-24 09:28:53,650 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-24 09:28:53,651 INFO train_multi TF=ALL: new best val=0.6136 — saved
2026-04-24 09:29:10,943 INFO train_multi TF=ALL epoch 2/50 train=0.6033 val=0.6135
2026-04-24 09:29:10,947 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-24 09:29:10,948 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-24 09:29:10,948 INFO train_multi TF=ALL: new best val=0.6135 — saved
2026-04-24 09:29:28,174 INFO train_multi TF=ALL epoch 3/50 train=0.6033 val=0.6137
2026-04-24 09:29:45,488 INFO train_multi TF=ALL epoch 4/50 train=0.6030 val=0.6144
2026-04-24 09:30:03,055 INFO train_multi TF=ALL epoch 5/50 train=0.6026 val=0.6134
2026-04-24 09:30:03,060 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-24 09:30:03,060 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-24 09:30:03,060 INFO train_multi TF=ALL: new best val=0.6134 — saved
2026-04-24 09:30:20,643 INFO train_multi TF=ALL epoch 6/50 train=0.6023 val=0.6141
2026-04-24 09:30:38,096 INFO train_multi TF=ALL epoch 7/50 train=0.6024 val=0.6147
2026-04-24 09:30:55,661 INFO train_multi TF=ALL epoch 8/50 train=0.6022 val=0.6148
2026-04-24 09:31:13,145 INFO train_multi TF=ALL epoch 9/50 train=0.6019 val=0.6150
2026-04-24 09:31:30,976 INFO train_multi TF=ALL epoch 10/50 train=0.6015 val=0.6131
2026-04-24 09:31:30,980 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-24 09:31:30,981 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-24 09:31:30,981 INFO train_multi TF=ALL: new best val=0.6131 — saved
2026-04-24 09:31:48,514 INFO train_multi TF=ALL epoch 11/50 train=0.6013 val=0.6144
2026-04-24 09:32:05,778 INFO train_multi TF=ALL epoch 12/50 train=0.6008 val=0.6163
2026-04-24 09:32:23,425 INFO train_multi TF=ALL epoch 13/50 train=0.6005 val=0.6155
2026-04-24 09:32:40,716 INFO train_multi TF=ALL epoch 14/50 train=0.5998 val=0.6166
2026-04-24 09:32:58,105 INFO train_multi TF=ALL epoch 15/50 train=0.5992 val=0.6162
2026-04-24 09:32:58,105 INFO train_multi TF=ALL early stop at epoch 15
2026-04-24 09:32:58,248 INFO === VectorStore: building similarity indices ===
2026-04-24 09:32:58,252 INFO Loading faiss with AVX512 support.
2026-04-24 09:32:58,275 INFO Successfully loaded faiss with AVX512 support.
2026-04-24 09:32:58,279 INFO VectorStore: CPU FAISS index built (dim=45)
2026-04-24 09:32:58,279 INFO VectorStore: CPU FAISS index built (dim=53)
2026-04-24 09:32:58,280 INFO VectorStore: CPU FAISS index built (dim=64)
2026-04-24 09:32:58,293 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-24 09:32:58,299 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:33:04,285 WARNING VectorStore trade_patterns failed for AUDUSD: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 09:33:04,291 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:33:04,293 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:33:04,294 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:33:04,295 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:33:04,295 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:33:42,680 WARNING VectorStore market_structures failed for AUDUSD: name 'np' is not defined
2026-04-24 09:33:48,125 WARNING VectorStore regime_embeddings failed for AUDUSD: name 'np' is not defined
2026-04-24 09:33:48,240 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:33:54,197 WARNING VectorStore trade_patterns failed for EURGBP: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 09:33:54,203 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:33:54,206 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:33:54,207 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:33:54,207 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:33:54,208 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:34:32,524 WARNING VectorStore market_structures failed for EURGBP: name 'np' is not defined
2026-04-24 09:34:38,251 WARNING VectorStore regime_embeddings failed for EURGBP: name 'np' is not defined
2026-04-24 09:34:38,369 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:34:44,476 WARNING VectorStore trade_patterns failed for EURJPY: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 09:34:44,482 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:34:44,484 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:34:44,485 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:34:44,486 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:34:44,486 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:35:22,693 WARNING VectorStore market_structures failed for EURJPY: name 'np' is not defined
2026-04-24 09:35:28,224 WARNING VectorStore regime_embeddings failed for EURJPY: name 'np' is not defined
2026-04-24 09:35:28,339 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:35:34,198 WARNING VectorStore trade_patterns failed for EURUSD: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 09:35:34,205 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:35:34,208 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:35:34,209 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:35:34,209 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:35:34,209 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:36:12,653 WARNING VectorStore market_structures failed for EURUSD: name 'np' is not defined
2026-04-24 09:36:18,109 WARNING VectorStore regime_embeddings failed for EURUSD: name 'np' is not defined
2026-04-24 09:36:18,225 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:36:24,104 WARNING VectorStore trade_patterns failed for GBPJPY: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 09:36:24,110 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:36:24,113 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:36:24,114 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:36:24,114 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:36:24,114 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:37:02,932 WARNING VectorStore market_structures failed for GBPJPY: name 'np' is not defined
2026-04-24 09:37:08,468 WARNING VectorStore regime_embeddings failed for GBPJPY: name 'np' is not defined
2026-04-24 09:37:08,582 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:37:14,544 WARNING VectorStore trade_patterns failed for GBPUSD: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 09:37:14,550 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:37:14,553 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:37:14,554 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:37:14,554 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:37:14,555 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:37:52,537 WARNING VectorStore market_structures failed for GBPUSD: name 'np' is not defined
2026-04-24 09:37:58,036 WARNING VectorStore regime_embeddings failed for GBPUSD: name 'np' is not defined
2026-04-24 09:37:58,149 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:38:04,112 WARNING VectorStore trade_patterns failed for NZDUSD: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 09:38:04,117 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-24 09:38:04,119 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-24 09:38:04,120 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-24 09:38:04,120 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-24 09:38:04,120 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-24 09:38:42,226 WARNING VectorStore market_structures failed for NZDUSD: name 'np' is not defined
2026-04-24 09:38:47,733 WARNING VectorStore regime_embeddings failed for NZDUSD: name 'np' is not defined
2026-04-24 09:38:47,847 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:38:53,738 WARNING VectorStore trade_patterns failed for USDCAD: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 09:38:53,744 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:38:53,746 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:38:53,747 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:38:53,748 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:38:53,748 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:39:32,384 WARNING VectorStore market_structures failed for USDCAD: name 'np' is not defined
2026-04-24 09:39:38,019 WARNING VectorStore regime_embeddings failed for USDCAD: name 'np' is not defined
2026-04-24 09:39:38,131 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:39:44,143 WARNING VectorStore trade_patterns failed for USDCHF: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 09:39:44,149 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:39:44,151 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:39:44,152 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:39:44,152 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:39:44,153 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:40:22,491 WARNING VectorStore market_structures failed for USDCHF: name 'np' is not defined
2026-04-24 09:40:28,036 WARNING VectorStore regime_embeddings failed for USDCHF: name 'np' is not defined
2026-04-24 09:40:28,151 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:40:34,193 WARNING VectorStore trade_patterns failed for USDJPY: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 09:40:34,199 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:40:34,201 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:40:34,202 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:40:34,202 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:40:34,203 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:41:12,645 WARNING VectorStore market_structures failed for USDJPY: name 'np' is not defined
2026-04-24 09:41:18,136 WARNING VectorStore regime_embeddings failed for USDJPY: name 'np' is not defined
2026-04-24 09:41:18,257 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-24 09:41:31,054 WARNING VectorStore trade_patterns failed for XAUUSD: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 09:41:31,064 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-24 09:41:31,067 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-24 09:41:31,069 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-24 09:41:31,069 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-24 09:41:31,070 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-24 09:42:57,867 WARNING VectorStore market_structures failed for XAUUSD: name 'np' is not defined
2026-04-24 09:43:10,299 WARNING VectorStore regime_embeddings failed for XAUUSD: name 'np' is not defined
2026-04-24 09:43:10,414 INFO VectorStore: saved 0 total vectors to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/vector_store
2026-04-24 09:43:10,414 INFO VectorStore saved: {'trade_patterns': 0, 'market_structures': 0, 'regime_embeddings': 0}
2026-04-24 09:43:10,420 INFO Retrain complete.
  DONE  Retrain gru [full-data retrain]
  START Retrain regime [full-data retrain]
2026-04-24 09:43:12,973 INFO retrain environment: KAGGLE
2026-04-24 09:43:14,684 INFO Device: CUDA (2 GPU(s))
2026-04-24 09:43:14,695 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 09:43:14,695 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 09:43:14,696 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-24 09:43:14,841 INFO NumExpr defaulting to 4 threads.
2026-04-24 09:43:15,037 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-24 09:43:15,037 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 09:43:15,038 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 09:43:15,265 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-24 09:43:15,267 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:15,346 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:15,425 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:15,504 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:15,580 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:15,655 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:15,726 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:15,800 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:15,872 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:15,944 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:16,033 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 09:43:16,091 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-24 09:43:16,107 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:16,109 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:16,123 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:16,124 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:16,138 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:16,140 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:16,154 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:16,156 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:16,172 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:16,176 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:16,190 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:16,193 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:16,207 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:16,209 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:16,223 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:16,227 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:16,240 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:16,243 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:16,261 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:16,264 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:43:16,281 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 09:43:16,287 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 09:43:17,058 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-24 09:43:40,087 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-24 09:43:40,090 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias)
2026-04-24 09:43:40,090 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-24 09:43:49,673 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-24 09:43:49,675 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias)
2026-04-24 09:43:49,675 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-24 09:43:57,224 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-24 09:43:57,228 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias)
2026-04-24 09:43:57,228 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-24 09:45:06,755 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-24 09:45:06,758 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour)
2026-04-24 09:45:06,758 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-24 09:45:37,799 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-24 09:45:37,800 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour)
2026-04-24 09:45:37,800 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-24 09:45:59,323 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-24 09:45:59,324 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour)
2026-04-24 09:45:59,429 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-24 09:45:59,430 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:45:59,431 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:45:59,432 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:45:59,433 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:45:59,434 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:45:59,436 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:45:59,437 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:45:59,438 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:45:59,439 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:45:59,440 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:45:59,441 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 09:45:59,577 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:45:59,620 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:45:59,621 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:45:59,621 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:45:59,631 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:45:59,632 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:46:02,487 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-24 09:46:02,488 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-24 09:46:02,667 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:02,703 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:02,703 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:02,704 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:02,714 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:02,715 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:46:05,411 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-24 09:46:05,412 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-24 09:46:05,588 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:05,626 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:05,627 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:05,628 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:05,637 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:05,638 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:46:08,392 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-24 09:46:08,393 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-24 09:46:08,569 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:08,608 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:08,608 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:08,609 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:08,619 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:08,620 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:46:11,358 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-24 09:46:11,360 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-24 09:46:11,533 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:11,572 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:11,573 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:11,573 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:11,584 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:11,585 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:46:14,312 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-24 09:46:14,313 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-24 09:46:14,499 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:14,535 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:14,535 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:14,536 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:14,546 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:14,547 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:46:17,313 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-24 09:46:17,314 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-24 09:46:17,482 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-24 09:46:17,511 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-24 09:46:17,512 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-24 09:46:17,512 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-24 09:46:17,522 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-24 09:46:17,523 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:46:20,351 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-24 09:46:20,352 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-24 09:46:20,524 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:20,558 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:20,559 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:20,559 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:20,569 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:20,570 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:46:23,334 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-24 09:46:23,335 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-24 09:46:23,510 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:23,544 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:23,544 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:23,545 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:23,555 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:23,556 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:46:26,293 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-24 09:46:26,294 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-24 09:46:26,471 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:26,507 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:26,508 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:26,508 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:26,518 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:26,519 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:46:29,304 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-24 09:46:29,305 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-24 09:46:29,580 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-24 09:46:29,642 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-24 09:46:29,643 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-24 09:46:29,643 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-24 09:46:29,656 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-24 09:46:29,658 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 09:46:36,071 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-24 09:46:36,073 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-24 09:46:36,238 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260424_094636
2026-04-24 09:46:36,435 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=34, n_classes=3)
2026-04-24 09:46:36,453 INFO RegimeClassifier[mode=htf_bias]: 103290 samples, classes={'BIAS_UP': 18622, 'BIAS_DOWN': 18286, 'BIAS_NEUTRAL': 66382}, device=cuda
2026-04-24 09:46:36,454 INFO RegimeClassifier: sample weights — mean=0.360  ambiguous(<0.4)=69.5%
2026-04-24 09:46:36,454 INFO RegimeClassifier[mode=htf_bias]: warm start from existing weights
2026-04-24 09:46:36,454 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-24 09:46:38,788 INFO Regime epoch  1/50 — tr=0.3018 va=0.8509 acc=0.597 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.351}
2026-04-24 09:46:38,970 INFO Regime epoch  2/50 — tr=0.3019 va=0.8512 acc=0.595
2026-04-24 09:46:39,143 INFO Regime epoch  3/50 — tr=0.3019 va=0.8523 acc=0.597
2026-04-24 09:46:39,316 INFO Regime epoch  4/50 — tr=0.3018 va=0.8528 acc=0.596
2026-04-24 09:46:39,508 INFO Regime epoch  5/50 — tr=0.3016 va=0.8526 acc=0.598 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.352}
2026-04-24 09:46:39,683 INFO Regime epoch  6/50 — tr=0.3017 va=0.8513 acc=0.599
2026-04-24 09:46:39,869 INFO Regime epoch  7/50 — tr=0.3018 va=0.8529 acc=0.599
2026-04-24 09:46:40,048 INFO Regime epoch  8/50 — tr=0.3018 va=0.8505 acc=0.596
2026-04-24 09:46:40,231 INFO Regime epoch  9/50 — tr=0.3018 va=0.8516 acc=0.594
2026-04-24 09:46:40,420 INFO Regime epoch 10/50 — tr=0.3016 va=0.8505 acc=0.603 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.36}
2026-04-24 09:46:40,602 INFO Regime epoch 11/50 — tr=0.3016 va=0.8511 acc=0.605
2026-04-24 09:46:40,785 INFO Regime epoch 12/50 — tr=0.3017 va=0.8515 acc=0.605
2026-04-24 09:46:40,971 INFO Regime epoch 13/50 — tr=0.3016 va=0.8502 acc=0.604
2026-04-24 09:46:41,146 INFO Regime epoch 14/50 — tr=0.3016 va=0.8502 acc=0.602
2026-04-24 09:46:41,333 INFO Regime epoch 15/50 — tr=0.3014 va=0.8499 acc=0.602 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.359}
2026-04-24 09:46:41,517 INFO Regime epoch 16/50 — tr=0.3015 va=0.8499 acc=0.602
2026-04-24 09:46:41,697 INFO Regime epoch 17/50 — tr=0.3015 va=0.8492 acc=0.607
2026-04-24 09:46:41,877 INFO Regime epoch 18/50 — tr=0.3015 va=0.8484 acc=0.607
2026-04-24 09:46:42,055 INFO Regime epoch 19/50 — tr=0.3013 va=0.8489 acc=0.612
2026-04-24 09:46:42,259 INFO Regime epoch 20/50 — tr=0.3014 va=0.8483 acc=0.610 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.372}
2026-04-24 09:46:42,434 INFO Regime epoch 21/50 — tr=0.3014 va=0.8493 acc=0.608
2026-04-24 09:46:42,609 INFO Regime epoch 22/50 — tr=0.3014 va=0.8497 acc=0.610
2026-04-24 09:46:42,790 INFO Regime epoch 23/50 — tr=0.3015 va=0.8495 acc=0.610
2026-04-24 09:46:42,965 INFO Regime epoch 24/50 — tr=0.3013 va=0.8489 acc=0.611
2026-04-24 09:46:43,160 INFO Regime epoch 25/50 — tr=0.3012 va=0.8492 acc=0.612 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.374}
2026-04-24 09:46:43,340 INFO Regime epoch 26/50 — tr=0.3015 va=0.8485 acc=0.611
2026-04-24 09:46:43,517 INFO Regime epoch 27/50 — tr=0.3015 va=0.8465 acc=0.614
2026-04-24 09:46:43,689 INFO Regime epoch 28/50 — tr=0.3016 va=0.8487 acc=0.614
2026-04-24 09:46:43,875 INFO Regime epoch 29/50 — tr=0.3012 va=0.8475 acc=0.614
2026-04-24 09:46:44,069 INFO Regime epoch 30/50 — tr=0.3014 va=0.8494 acc=0.611 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.373}
2026-04-24 09:46:44,253 INFO Regime epoch 31/50 — tr=0.3014 va=0.8470 acc=0.613
2026-04-24 09:46:44,439 INFO Regime epoch 32/50 — tr=0.3014 va=0.8475 acc=0.612
2026-04-24 09:46:44,620 INFO Regime epoch 33/50 — tr=0.3012 va=0.8468 acc=0.615
2026-04-24 09:46:44,802 INFO Regime epoch 34/50 — tr=0.3013 va=0.8474 acc=0.614
2026-04-24 09:46:44,998 INFO Regime epoch 35/50 — tr=0.3013 va=0.8475 acc=0.614 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.379}
2026-04-24 09:46:45,172 INFO Regime epoch 36/50 — tr=0.3012 va=0.8475 acc=0.615
2026-04-24 09:46:45,352 INFO Regime epoch 37/50 — tr=0.3013 va=0.8472 acc=0.617
2026-04-24 09:46:45,353 INFO Regime early stop at epoch 37 (no_improve=10)
2026-04-24 09:46:45,367 WARNING RegimeClassifier accuracy 0.61 < 0.65 threshold
2026-04-24 09:46:45,370 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-24 09:46:45,370 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-24 09:46:45,502 INFO Regime HTF complete: acc=0.614, n=103290
2026-04-24 09:46:45,504 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 09:46:45,650 INFO Rule labels LTF_BEHAVIOUR [4H]: {'TRENDING': 8970, 'RANGING': 4592, 'CONSOLIDATING': 2247, 'VOLATILE': 4008}  ambiguous=4539 (total=19817)  short_runs_zeroed=570
2026-04-24 09:46:45,653 INFO Regime[4H] persistence (avg bars/run) on XAUUSD 4H:
{0: 12.121621621621621, 1: 6.4858757062146895, 2: 5.575682382133995, 3: 10.329896907216495}
2026-04-24 09:46:45,654 INFO Regime[4H] return separation on XAUUSD 4H:
{0: 5.9502411776840314e-05, 1: 5.664222849552333e-05, 2: -6.138442068010657e-05, 3: 4.239843820627722e-05}
2026-04-24 09:46:45,655 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-24 09:46:45,656 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:46:45,658 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:46:45,659 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:46:45,661 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:46:45,663 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:46:45,664 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:46:45,665 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:46:45,667 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:46:45,669 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:46:45,670 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:46:45,673 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 09:46:45,684 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:45,686 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:45,687 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:45,687 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:45,687 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:45,689 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:46:55,783 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-24 09:46:55,786 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-24 09:46:55,923 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:55,926 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:55,926 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:55,927 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:55,927 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:46:55,929 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:47:05,987 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-24 09:47:05,989 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-24 09:47:06,123 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:06,127 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:06,128 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:06,128 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:06,128 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:06,130 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:47:15,910 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-24 09:47:15,913 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-24 09:47:16,049 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:16,051 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:16,052 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:16,053 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:16,053 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:16,055 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:47:25,918 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-24 09:47:25,921 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-24 09:47:26,058 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:26,062 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:26,062 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:26,063 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:26,063 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:26,065 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:47:35,807 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-24 09:47:35,810 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-24 09:47:35,949 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:35,952 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:35,952 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:35,953 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:35,953 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:35,955 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:47:45,787 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-24 09:47:45,790 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-24 09:47:45,925 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-24 09:47:45,927 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-24 09:47:45,927 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-24 09:47:45,928 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-24 09:47:45,928 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-24 09:47:45,930 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:47:55,858 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-24 09:47:55,861 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-24 09:47:56,000 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:56,002 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:56,003 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:56,003 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:56,004 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:47:56,006 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:48:06,032 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-24 09:48:06,035 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-24 09:48:06,174 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:48:06,177 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:48:06,178 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:48:06,178 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:48:06,179 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:48:06,181 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:48:16,029 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-24 09:48:16,032 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-24 09:48:16,171 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:48:16,173 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:48:16,174 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:48:16,174 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:48:16,175 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:48:16,177 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:48:26,147 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-24 09:48:26,149 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-24 09:48:26,295 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-24 09:48:26,298 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-24 09:48:26,300 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-24 09:48:26,300 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-24 09:48:26,300 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-24 09:48:26,304 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 09:48:48,668 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-24 09:48:48,673 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-24 09:48:48,973 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260424_094848
2026-04-24 09:48:48,978 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=18, n_classes=4)
2026-04-24 09:48:49,033 INFO RegimeClassifier[mode=ltf_behaviour]: 401471 samples, classes={'TRENDING': 160094, 'RANGING': 105286, 'CONSOLIDATING': 53524, 'VOLATILE': 82567}, device=cuda
2026-04-24 09:48:49,034 INFO RegimeClassifier: sample weights — mean=0.505  ambiguous(<0.4)=33.0%
2026-04-24 09:48:49,034 INFO RegimeClassifier[mode=ltf_behaviour]: warm start from existing weights
2026-04-24 09:48:49,035 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-24 09:48:49,756 INFO Regime epoch  1/50 — tr=0.5504 va=1.5750 acc=0.701 per_class={'TRENDING': 0.762, 'RANGING': 0.294, 'CONSOLIDATING': 0.953, 'VOLATILE': 0.94}
2026-04-24 09:48:50,413 INFO Regime epoch  2/50 — tr=0.5504 va=1.5725 acc=0.702
2026-04-24 09:48:51,051 INFO Regime epoch  3/50 — tr=0.5504 va=1.5745 acc=0.701
2026-04-24 09:48:51,716 INFO Regime epoch  4/50 — tr=0.5503 va=1.5721 acc=0.700
2026-04-24 09:48:52,417 INFO Regime epoch  5/50 — tr=0.5502 va=1.5764 acc=0.701 per_class={'TRENDING': 0.765, 'RANGING': 0.289, 'CONSOLIDATING': 0.954, 'VOLATILE': 0.941}
2026-04-24 09:48:53,048 INFO Regime epoch  6/50 — tr=0.5502 va=1.5723 acc=0.701
2026-04-24 09:48:53,682 INFO Regime epoch  7/50 — tr=0.5501 va=1.5745 acc=0.700
2026-04-24 09:48:54,338 INFO Regime epoch  8/50 — tr=0.5502 va=1.5739 acc=0.700
2026-04-24 09:48:54,999 INFO Regime epoch  9/50 — tr=0.5499 va=1.5716 acc=0.700
2026-04-24 09:48:55,696 INFO Regime epoch 10/50 — tr=0.5501 va=1.5706 acc=0.701 per_class={'TRENDING': 0.763, 'RANGING': 0.294, 'CONSOLIDATING': 0.955, 'VOLATILE': 0.937}
2026-04-24 09:48:56,373 INFO Regime epoch 11/50 — tr=0.5500 va=1.5740 acc=0.702
2026-04-24 09:48:57,033 INFO Regime epoch 12/50 — tr=0.5498 va=1.5745 acc=0.701
2026-04-24 09:48:57,746 INFO Regime epoch 13/50 — tr=0.5501 va=1.5742 acc=0.701
2026-04-24 09:48:58,420 INFO Regime epoch 14/50 — tr=0.5498 va=1.5706 acc=0.702
2026-04-24 09:48:59,110 INFO Regime epoch 15/50 — tr=0.5499 va=1.5700 acc=0.702 per_class={'TRENDING': 0.766, 'RANGING': 0.296, 'CONSOLIDATING': 0.953, 'VOLATILE': 0.939}
2026-04-24 09:48:59,793 INFO Regime epoch 16/50 — tr=0.5495 va=1.5736 acc=0.703
2026-04-24 09:49:00,474 INFO Regime epoch 17/50 — tr=0.5496 va=1.5714 acc=0.700
2026-04-24 09:49:01,138 INFO Regime epoch 18/50 — tr=0.5498 va=1.5715 acc=0.702
2026-04-24 09:49:01,816 INFO Regime epoch 19/50 — tr=0.5495 va=1.5690 acc=0.703
2026-04-24 09:49:02,505 INFO Regime epoch 20/50 — tr=0.5494 va=1.5690 acc=0.702 per_class={'TRENDING': 0.764, 'RANGING': 0.297, 'CONSOLIDATING': 0.955, 'VOLATILE': 0.937}
2026-04-24 09:49:03,163 INFO Regime epoch 21/50 — tr=0.5495 va=1.5719 acc=0.701
2026-04-24 09:49:03,844 INFO Regime epoch 22/50 — tr=0.5495 va=1.5702 acc=0.701
2026-04-24 09:49:04,495 INFO Regime epoch 23/50 — tr=0.5494 va=1.5734 acc=0.701
2026-04-24 09:49:05,174 INFO Regime epoch 24/50 — tr=0.5494 va=1.5702 acc=0.702
2026-04-24 09:49:05,874 INFO Regime epoch 25/50 — tr=0.5493 va=1.5719 acc=0.702 per_class={'TRENDING': 0.767, 'RANGING': 0.29, 'CONSOLIDATING': 0.955, 'VOLATILE': 0.94}
2026-04-24 09:49:06,514 INFO Regime epoch 26/50 — tr=0.5494 va=1.5704 acc=0.703
2026-04-24 09:49:07,173 INFO Regime epoch 27/50 — tr=0.5494 va=1.5727 acc=0.702
2026-04-24 09:49:07,836 INFO Regime epoch 28/50 — tr=0.5494 va=1.5694 acc=0.703
2026-04-24 09:49:08,474 INFO Regime epoch 29/50 — tr=0.5492 va=1.5729 acc=0.702
2026-04-24 09:49:08,475 INFO Regime early stop at epoch 29 (no_improve=10)
2026-04-24 09:49:08,522 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-24 09:49:08,522 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-24 09:49:08,653 INFO Regime LTF complete: acc=0.703, n=401471
2026-04-24 09:49:08,656 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 09:49:09,120 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-24 09:49:09,125 INFO Regime[1H] persistence (avg bars/run) on XAUUSD 1H:
{0: 9.355291913830783, 1: 6.096303199751476, 2: 5.598885793871866, 3: 6.771351107094442}
2026-04-24 09:49:09,127 INFO Regime[1H] return separation on XAUUSD 1H:
{0: 2.149062259083323e-05, 1: -3.254493608872846e-06, 2: -3.039756158146958e-06, 3: 1.8102764458593193e-05}
2026-04-24 09:49:09,130 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-24 09:49:09,130 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 09:49:09,130 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 09:49:09,130 INFO === VectorStore: building similarity indices ===
2026-04-24 09:49:09,134 INFO Loading faiss with AVX512 support.
2026-04-24 09:49:09,157 INFO Successfully loaded faiss with AVX512 support.
2026-04-24 09:49:09,163 INFO VectorStore: CPU FAISS index built (dim=45)
2026-04-24 09:49:09,163 INFO VectorStore: CPU FAISS index built (dim=53)
2026-04-24 09:49:09,163 INFO VectorStore: CPU FAISS index built (dim=64)
2026-04-24 09:49:09,190 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-24 09:49:09,195 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:49:15,088 WARNING VectorStore trade_patterns failed for AUDUSD: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 09:49:15,095 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:49:15,097 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:49:15,098 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:49:15,098 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:49:15,099 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:49:53,229 WARNING VectorStore market_structures failed for AUDUSD: name 'np' is not defined
2026-04-24 09:49:58,750 WARNING VectorStore regime_embeddings failed for AUDUSD: name 'np' is not defined
2026-04-24 09:49:58,891 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:50:04,861 WARNING VectorStore trade_patterns failed for EURGBP: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 09:50:04,868 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:50:04,870 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:50:04,871 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:50:04,871 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:50:04,872 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:50:43,915 WARNING VectorStore market_structures failed for EURGBP: name 'np' is not defined
2026-04-24 09:50:49,483 WARNING VectorStore regime_embeddings failed for EURGBP: name 'np' is not defined
2026-04-24 09:50:49,629 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:50:55,477 WARNING VectorStore trade_patterns failed for EURJPY: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 09:50:55,484 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:50:55,486 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:50:55,487 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:50:55,488 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:50:55,488 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:51:33,825 WARNING VectorStore market_structures failed for EURJPY: name 'np' is not defined
2026-04-24 09:51:39,118 WARNING VectorStore regime_embeddings failed for EURJPY: name 'np' is not defined
2026-04-24 09:51:39,258 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:51:45,232 WARNING VectorStore trade_patterns failed for EURUSD: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 09:51:45,238 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:51:45,241 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:51:45,242 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:51:45,242 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:51:45,242 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:52:23,902 WARNING VectorStore market_structures failed for EURUSD: name 'np' is not defined
2026-04-24 09:52:29,450 WARNING VectorStore regime_embeddings failed for EURUSD: name 'np' is not defined
2026-04-24 09:52:29,593 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:52:35,634 WARNING VectorStore trade_patterns failed for GBPJPY: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 09:52:35,640 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:52:35,643 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:52:35,643 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:52:35,644 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:52:35,644 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:53:14,446 WARNING VectorStore market_structures failed for GBPJPY: name 'np' is not defined
2026-04-24 09:53:20,006 WARNING VectorStore regime_embeddings failed for GBPJPY: name 'np' is not defined
2026-04-24 09:53:20,150 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:53:26,104 WARNING VectorStore trade_patterns failed for GBPUSD: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 09:53:26,110 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:53:26,113 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:53:26,113 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:53:26,114 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:53:26,114 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:54:04,895 WARNING VectorStore market_structures failed for GBPUSD: name 'np' is not defined
2026-04-24 09:54:10,309 WARNING VectorStore regime_embeddings failed for GBPUSD: name 'np' is not defined
2026-04-24 09:54:10,452 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:54:16,284 WARNING VectorStore trade_patterns failed for NZDUSD: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 09:54:16,289 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-24 09:54:16,290 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-24 09:54:16,291 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-24 09:54:16,292 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-24 09:54:16,292 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-24 09:54:54,849 WARNING VectorStore market_structures failed for NZDUSD: name 'np' is not defined
2026-04-24 09:55:00,281 WARNING VectorStore regime_embeddings failed for NZDUSD: name 'np' is not defined
2026-04-24 09:55:00,426 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:55:06,441 WARNING VectorStore trade_patterns failed for USDCAD: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 09:55:06,447 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:55:06,449 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:55:06,450 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:55:06,451 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:55:06,451 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:55:44,787 WARNING VectorStore market_structures failed for USDCAD: name 'np' is not defined
2026-04-24 09:55:50,152 WARNING VectorStore regime_embeddings failed for USDCAD: name 'np' is not defined
2026-04-24 09:55:50,295 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:55:56,315 WARNING VectorStore trade_patterns failed for USDCHF: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 09:55:56,321 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:55:56,323 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:55:56,324 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:55:56,324 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:55:56,325 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:56:34,758 WARNING VectorStore market_structures failed for USDCHF: name 'np' is not defined
2026-04-24 09:56:40,366 WARNING VectorStore regime_embeddings failed for USDCHF: name 'np' is not defined
2026-04-24 09:56:40,510 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-24 09:56:46,469 WARNING VectorStore trade_patterns failed for USDJPY: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 09:56:46,476 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:56:46,478 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:56:46,479 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:56:46,480 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:56:46,480 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 09:57:25,009 WARNING VectorStore market_structures failed for USDJPY: name 'np' is not defined
2026-04-24 09:57:30,552 WARNING VectorStore regime_embeddings failed for USDJPY: name 'np' is not defined
2026-04-24 09:57:30,698 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-24 09:57:43,423 WARNING VectorStore trade_patterns failed for XAUUSD: VectorStore.add_batch: 'trade_patterns' expects dim=45, got 74
2026-04-24 09:57:43,432 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-24 09:57:43,436 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-24 09:57:43,437 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-24 09:57:43,437 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-24 09:57:43,438 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-24 09:59:09,615 WARNING VectorStore market_structures failed for XAUUSD: name 'np' is not defined
2026-04-24 09:59:21,768 WARNING VectorStore regime_embeddings failed for XAUUSD: name 'np' is not defined
2026-04-24 09:59:21,911 INFO VectorStore: saved 0 total vectors to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/vector_store
2026-04-24 09:59:21,911 INFO VectorStore saved: {'trade_patterns': 0, 'market_structures': 0, 'regime_embeddings': 0}
2026-04-24 09:59:21,923 INFO Retrain complete.
  DONE  Retrain regime [full-data retrain]
  START Retrain quality [full-data retrain]
2026-04-24 09:59:23,678 INFO retrain environment: KAGGLE
2026-04-24 09:59:25,404 INFO Device: CUDA (2 GPU(s))
2026-04-24 09:59:25,415 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 09:59:25,415 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 09:59:25,416 INFO === QualityScorer retrain ===
2026-04-24 09:59:25,552 INFO NumExpr defaulting to 4 threads.
2026-04-24 09:59:25,743 INFO QualityScorer: CUDA available — using GPU
2026-04-24 09:59:25,948 INFO QualityScorer loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (device=cuda)
2026-04-24 09:59:26,182 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/quality_scorer.pkl_20260424_095926
2026-04-24 09:59:26,400 INFO QualityScorer: 6495 samples, EV stats={'mean': 0.13979585468769073, 'std': 1.1955955028533936, 'n_pos': 3321, 'n_neg': 3174}, device=cuda
2026-04-24 09:59:26,401 INFO QualityScorer: normalised win labels by median_win=0.800 — EV range now [-1, +3]
2026-04-24 09:59:26,401 INFO QualityScorer: warm start from existing weights
2026-04-24 09:59:26,402 INFO QualityScorer: pos_weight=1.00 (n_pos=2658 n_neg=2538)
2026-04-24 09:59:28,080 INFO Quality epoch   1/100 — va_huber=0.7061
2026-04-24 09:59:28,206 INFO Quality epoch   2/100 — va_huber=0.7057
2026-04-24 09:59:28,331 INFO Quality epoch   3/100 — va_huber=0.7056
2026-04-24 09:59:28,451 INFO Quality epoch   4/100 — va_huber=0.7057
2026-04-24 09:59:28,573 INFO Quality epoch   5/100 — va_huber=0.7048
2026-04-24 09:59:29,324 INFO Quality epoch  11/100 — va_huber=0.7053
2026-04-24 09:59:30,057 INFO Quality early stop at epoch 17
2026-04-24 09:59:30,077 INFO QualityScorer EV model: MAE=1.178 dir_acc=0.583 n_val=1299
2026-04-24 09:59:30,080 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-24 09:59:30,149 INFO Retrain complete.
  DONE  Retrain quality [full-data retrain]
  START Retrain rl [full-data retrain]
2026-04-24 09:59:31,150 INFO retrain environment: KAGGLE
2026-04-24 09:59:32,817 INFO Device: CUDA (2 GPU(s))
2026-04-24 09:59:32,827 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 09:59:32,827 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 09:59:32,828 INFO === RLAgent (PPO) retrain ===
2026-04-24 09:59:32,830 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260424_095932
2026-04-24 09:59:33.662432: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777024773.685579  100948 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777024773.693392  100948 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777024773.713636  100948 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777024773.713683  100948 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777024773.713689  100948 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777024773.713693  100948 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-24 09:59:38,112 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-24 09:59:40,882 INFO RLAgent: PPO model loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-24 09:59:41,042 INFO RLAgent: warm start — fine-tuning existing PPO policy (lr=6.00e-05)
2026-04-24 10:01:29,663 INFO RLAgent: retrain complete, 6495 episodes
2026-04-24 10:01:29,663 INFO Retrain complete.
  DONE  Retrain rl [full-data retrain]

=== Round 3: Backtest on last 3yr (post-retrain evaluation) ===
  START Round 3 - Post-retrain backtest (last 3yr)
2026-04-24 10:01:31,660 INFO === STEP 6: BACKTEST + REINFORCED TRAINING (3 rounds, window=round3) ===
2026-04-24 10:01:31,661 INFO BT_WINDOW=round3 — post-retrain eval: 2022-08-05 → 2025-08-05 (last 3yr)
2026-04-24 10:01:31,662 INFO Cleared existing journal for fresh reinforced training run
2026-04-24 10:01:31,662 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-24 10:01:31,663 INFO Round 1 — running backtest: 2022-08-05 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-24 10:10:00,193 INFO Round 1 backtest — 2439 trades | avg WR=54.6% | avg PF=2.53 | avg Sharpe=4.68
2026-04-24 10:10:00,193 INFO   ml_trader: 2439 trades | WR=54.6% | PF=2.53 | Return=3826.3% | DD=2.8% | Sharpe=4.68
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 2439
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2439 rows)
2026-04-24 10:10:01,264 INFO Round 1: wrote 2439 journal entries (total in file: 2439)
2026-04-24 10:10:01,266 INFO Round 1 — retraining regime...
2026-04-24 10:25:43,031 INFO Retrain regime: OK
2026-04-24 10:25:43,049 INFO Round 1 — retraining quality...
2026-04-24 10:25:49,402 INFO Retrain quality: OK
2026-04-24 10:25:49,418 INFO Round 1 — retraining rl...
2026-04-24 10:26:41,476 INFO Retrain rl: OK
2026-04-24 10:26:41,494 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-24 10:26:41,494 INFO Round 2 — running backtest: 2022-08-05 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-24 10:35:23,133 INFO Round 2 backtest — 2422 trades | avg WR=54.0% | avg PF=2.49 | avg Sharpe=4.61
2026-04-24 10:35:23,133 INFO   ml_trader: 2422 trades | WR=54.0% | PF=2.49 | Return=3559.0% | DD=3.3% | Sharpe=4.61
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 2422
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2422 rows)
2026-04-24 10:35:24,205 INFO Round 2: wrote 2422 journal entries (total in file: 4861)
2026-04-24 10:35:24,207 INFO Round 2 — retraining regime...
2026-04-24 10:51:27,816 INFO Retrain regime: OK
2026-04-24 10:51:27,834 INFO Round 2 — retraining quality...
2026-04-24 10:51:34,688 INFO Retrain quality: OK
2026-04-24 10:51:34,706 INFO Round 2 — retraining rl...
2026-04-24 10:53:05,577 INFO Retrain rl: OK
2026-04-24 10:53:05,595 INFO ==========================