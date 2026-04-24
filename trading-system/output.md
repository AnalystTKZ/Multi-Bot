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
2026-04-24 00:48:53,420 INFO Loading feature-engineered data...
2026-04-24 00:48:53,961 INFO Loaded 221743 rows, 202 features
2026-04-24 00:48:53,961 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-24 00:48:53,962 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-24 00:48:53,962 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-24 00:48:53,962 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-24 00:48:53,962 INFO No leakage confirmed: train < val < test timestamps

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
2026-04-24 00:48:56,635 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-24 00:48:56,635 INFO --- Training regime ---
2026-04-24 00:48:56,636 INFO Running retrain --model regime
2026-04-24 00:48:56,865 INFO retrain environment: KAGGLE
2026-04-24 00:48:58,829 INFO Device: CUDA (2 GPU(s))
2026-04-24 00:48:58,844 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 00:48:58,844 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 00:48:58,845 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-24 00:48:59,015 INFO NumExpr defaulting to 4 threads.
2026-04-24 00:48:59,266 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-24 00:48:59,266 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 00:48:59,266 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 00:48:59,538 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-24 00:48:59,540 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:48:59,631 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:48:59,719 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:48:59,805 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:48:59,896 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:48:59,985 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,068 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,159 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,244 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,333 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,439 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:49:00,513 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-24 00:49:00,532 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,534 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,552 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,553 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,571 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,573 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,590 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,592 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,611 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,614 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,632 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,635 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,652 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,654 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,671 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,675 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,692 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,696 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,712 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,716 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:49:00,736 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:49:00,744 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:49:01,606 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-24 00:49:27,601 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-24 00:49:27,606 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias)
2026-04-24 00:49:27,606 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-24 00:49:39,175 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-24 00:49:39,177 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias)
2026-04-24 00:49:39,178 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-24 00:49:48,638 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-24 00:49:48,642 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias)
2026-04-24 00:49:48,642 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-24 00:51:10,079 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-24 00:51:10,083 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour)
2026-04-24 00:51:10,083 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-24 00:51:46,530 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-24 00:51:46,536 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour)
2026-04-24 00:51:46,536 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-24 00:52:12,244 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-24 00:52:12,245 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour)
2026-04-24 00:52:12,396 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-24 00:52:12,398 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:52:12,400 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:52:12,401 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:52:12,402 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:52:12,403 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:52:12,404 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:52:12,405 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:52:12,406 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:52:12,408 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:52:12,409 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:52:12,410 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:52:12,558 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:12,609 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:12,610 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:12,610 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:12,620 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:12,622 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:52:15,872 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-24 00:52:15,873 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-24 00:52:16,083 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:16,122 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:16,124 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:16,124 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:16,135 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:16,136 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:52:19,338 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-24 00:52:19,339 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-24 00:52:19,551 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:19,593 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:19,594 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:19,595 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:19,605 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:19,606 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:52:22,828 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-24 00:52:22,830 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-24 00:52:23,020 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:23,061 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:23,062 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:23,062 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:23,073 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:23,074 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:52:26,262 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-24 00:52:26,263 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-24 00:52:26,469 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:26,512 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:26,513 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:26,514 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:26,525 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:26,526 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:52:29,748 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-24 00:52:29,749 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-24 00:52:29,948 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:29,988 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:29,989 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:29,989 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:30,000 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:30,001 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:52:33,226 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-24 00:52:33,227 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-24 00:52:33,414 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:52:33,448 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:52:33,449 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:52:33,450 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:52:33,459 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:52:33,460 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:52:36,667 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-24 00:52:36,668 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-24 00:52:36,869 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:36,909 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:36,910 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:36,910 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:36,922 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:36,923 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:52:40,132 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-24 00:52:40,133 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-24 00:52:40,343 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:40,385 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:40,386 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:40,387 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:40,398 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:40,399 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:52:43,668 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-24 00:52:43,670 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-24 00:52:43,879 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:43,920 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:43,921 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:43,922 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:43,932 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:52:43,933 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:52:47,126 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-24 00:52:47,127 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-24 00:52:47,447 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:52:47,518 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:52:47,519 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:52:47,520 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:52:47,533 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:52:47,535 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:52:55,171 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-24 00:52:55,174 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-24 00:52:55,431 INFO RegimeClassifier[mode=htf_bias]: 103290 samples, classes={'BIAS_UP': 18622, 'BIAS_DOWN': 18286, 'BIAS_NEUTRAL': 66382}, device=cuda
2026-04-24 00:52:55,432 INFO RegimeClassifier: sample weights — mean=0.360  ambiguous(<0.4)=69.5%
2026-04-24 00:52:55,633 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-04-24 00:52:55,634 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-24 00:52:58,374 INFO Regime epoch  1/50 — tr=0.6864 va=1.3636 acc=0.277 per_class={'BIAS_UP': 0.47, 'BIAS_DOWN': 0.019, 'BIAS_NEUTRAL': 0.274}
2026-04-24 00:52:58,599 INFO Regime epoch  2/50 — tr=0.6764 va=1.3543 acc=0.292
2026-04-24 00:52:58,819 INFO Regime epoch  3/50 — tr=0.6575 va=1.3172 acc=0.327
2026-04-24 00:52:59,034 INFO Regime epoch  4/50 — tr=0.6286 va=1.2570 acc=0.394
2026-04-24 00:52:59,268 INFO Regime epoch  5/50 — tr=0.5974 va=1.1874 acc=0.447 per_class={'BIAS_UP': 0.801, 'BIAS_DOWN': 0.937, 'BIAS_NEUTRAL': 0.196}
2026-04-24 00:52:59,490 INFO Regime epoch  6/50 — tr=0.5661 va=1.0990 acc=0.482
2026-04-24 00:52:59,708 INFO Regime epoch  7/50 — tr=0.5402 va=1.0183 acc=0.493
2026-04-24 00:52:59,927 INFO Regime epoch  8/50 — tr=0.5234 va=0.9686 acc=0.495
2026-04-24 00:53:00,146 INFO Regime epoch  9/50 — tr=0.5115 va=0.9418 acc=0.500
2026-04-24 00:53:00,383 INFO Regime epoch 10/50 — tr=0.5046 va=0.9322 acc=0.500 per_class={'BIAS_UP': 0.989, 'BIAS_DOWN': 0.996, 'BIAS_NEUTRAL': 0.199}
2026-04-24 00:53:00,604 INFO Regime epoch 11/50 — tr=0.4975 va=0.9198 acc=0.512
2026-04-24 00:53:00,823 INFO Regime epoch 12/50 — tr=0.4926 va=0.9140 acc=0.522
2026-04-24 00:53:01,048 INFO Regime epoch 13/50 — tr=0.4881 va=0.9094 acc=0.518
2026-04-24 00:53:01,272 INFO Regime epoch 14/50 — tr=0.4856 va=0.9035 acc=0.524
2026-04-24 00:53:01,516 INFO Regime epoch 15/50 — tr=0.4829 va=0.9046 acc=0.524 per_class={'BIAS_UP': 0.994, 'BIAS_DOWN': 0.995, 'BIAS_NEUTRAL': 0.237}
2026-04-24 00:53:01,742 INFO Regime epoch 16/50 — tr=0.4800 va=0.9008 acc=0.534
2026-04-24 00:53:01,969 INFO Regime epoch 17/50 — tr=0.4780 va=0.8972 acc=0.540
2026-04-24 00:53:02,197 INFO Regime epoch 18/50 — tr=0.4766 va=0.8975 acc=0.545
2026-04-24 00:53:02,452 INFO Regime epoch 19/50 — tr=0.4748 va=0.8980 acc=0.544
2026-04-24 00:53:02,687 INFO Regime epoch 20/50 — tr=0.4734 va=0.8985 acc=0.545 per_class={'BIAS_UP': 0.992, 'BIAS_DOWN': 0.995, 'BIAS_NEUTRAL': 0.27}
2026-04-24 00:53:02,904 INFO Regime epoch 21/50 — tr=0.4728 va=0.8977 acc=0.549
2026-04-24 00:53:03,116 INFO Regime epoch 22/50 — tr=0.4711 va=0.8941 acc=0.554
2026-04-24 00:53:03,341 INFO Regime epoch 23/50 — tr=0.4705 va=0.8927 acc=0.555
2026-04-24 00:53:03,563 INFO Regime epoch 24/50 — tr=0.4698 va=0.8940 acc=0.552
2026-04-24 00:53:03,811 INFO Regime epoch 25/50 — tr=0.4694 va=0.8924 acc=0.564 per_class={'BIAS_UP': 0.993, 'BIAS_DOWN': 0.995, 'BIAS_NEUTRAL': 0.301}
2026-04-24 00:53:04,048 INFO Regime epoch 26/50 — tr=0.4689 va=0.8920 acc=0.566
2026-04-24 00:53:04,282 INFO Regime epoch 27/50 — tr=0.4680 va=0.8901 acc=0.566
2026-04-24 00:53:04,507 INFO Regime epoch 28/50 — tr=0.4677 va=0.8892 acc=0.562
2026-04-24 00:53:04,726 INFO Regime epoch 29/50 — tr=0.4672 va=0.8894 acc=0.565
2026-04-24 00:53:04,963 INFO Regime epoch 30/50 — tr=0.4669 va=0.8877 acc=0.569 per_class={'BIAS_UP': 0.995, 'BIAS_DOWN': 0.996, 'BIAS_NEUTRAL': 0.309}
2026-04-24 00:53:05,186 INFO Regime epoch 31/50 — tr=0.4668 va=0.8899 acc=0.573
2026-04-24 00:53:05,417 INFO Regime epoch 32/50 — tr=0.4660 va=0.8881 acc=0.571
2026-04-24 00:53:05,639 INFO Regime epoch 33/50 — tr=0.4659 va=0.8854 acc=0.572
2026-04-24 00:53:05,877 INFO Regime epoch 34/50 — tr=0.4659 va=0.8886 acc=0.573
2026-04-24 00:53:06,111 INFO Regime epoch 35/50 — tr=0.4658 va=0.8875 acc=0.572 per_class={'BIAS_UP': 0.995, 'BIAS_DOWN': 0.997, 'BIAS_NEUTRAL': 0.314}
2026-04-24 00:53:06,336 INFO Regime epoch 36/50 — tr=0.4656 va=0.8895 acc=0.575
2026-04-24 00:53:06,555 INFO Regime epoch 37/50 — tr=0.4657 va=0.8864 acc=0.574
2026-04-24 00:53:06,767 INFO Regime epoch 38/50 — tr=0.4656 va=0.8860 acc=0.577
2026-04-24 00:53:06,982 INFO Regime epoch 39/50 — tr=0.4652 va=0.8856 acc=0.579
2026-04-24 00:53:07,218 INFO Regime epoch 40/50 — tr=0.4647 va=0.8867 acc=0.579 per_class={'BIAS_UP': 0.994, 'BIAS_DOWN': 0.995, 'BIAS_NEUTRAL': 0.326}
2026-04-24 00:53:07,449 INFO Regime epoch 41/50 — tr=0.4648 va=0.8882 acc=0.581
2026-04-24 00:53:07,676 INFO Regime epoch 42/50 — tr=0.4647 va=0.8857 acc=0.579
2026-04-24 00:53:07,904 INFO Regime epoch 43/50 — tr=0.4649 va=0.8860 acc=0.579
2026-04-24 00:53:07,904 INFO Regime early stop at epoch 43 (no_improve=10)
2026-04-24 00:53:07,921 WARNING RegimeClassifier accuracy 0.57 < 0.65 threshold
2026-04-24 00:53:07,924 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-24 00:53:07,925 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-24 00:53:08,080 INFO Regime HTF complete: acc=0.572, n=103290
2026-04-24 00:53:08,082 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:53:08,271 INFO Rule labels LTF_BEHAVIOUR [4H]: {'TRENDING': 8970, 'RANGING': 4592, 'CONSOLIDATING': 2247, 'VOLATILE': 4008}  ambiguous=4539 (total=19817)  short_runs_zeroed=570
2026-04-24 00:53:08,275 INFO Regime[4H] persistence (avg bars/run) on XAUUSD 4H:
{0: 12.121621621621621, 1: 6.4858757062146895, 2: 5.575682382133995, 3: 10.329896907216495}
2026-04-24 00:53:08,277 INFO Regime[4H] return separation on XAUUSD 4H:
{0: 5.9502411776840314e-05, 1: 5.664222849552333e-05, 2: -6.138442068010657e-05, 3: 4.239843820627722e-05}
2026-04-24 00:53:08,278 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-24 00:53:08,280 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:53:08,282 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:53:08,284 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:53:08,286 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:53:08,288 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:53:08,290 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:53:08,291 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:53:08,293 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:53:08,295 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:53:08,298 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:53:08,302 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:53:08,314 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:08,316 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:08,317 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:08,318 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:08,318 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:08,320 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:53:20,010 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-24 00:53:20,013 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-24 00:53:20,173 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:20,175 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:20,176 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:20,176 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:20,177 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:20,179 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:53:31,674 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-24 00:53:31,677 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-24 00:53:31,834 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:31,837 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:31,838 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:31,838 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:31,839 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:31,841 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:53:43,314 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-24 00:53:43,317 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-24 00:53:43,474 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:43,476 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:43,477 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:43,478 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:43,478 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:43,480 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:53:55,092 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-24 00:53:55,096 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-24 00:53:55,255 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:55,258 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:55,259 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:55,259 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:55,259 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:53:55,262 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:54:06,777 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-24 00:54:06,780 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-24 00:54:06,939 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:54:06,942 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:54:06,943 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:54:06,943 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:54:06,944 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:54:06,946 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:54:18,481 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-24 00:54:18,484 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-24 00:54:18,651 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:54:18,653 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:54:18,654 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:54:18,654 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:54:18,655 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:54:18,657 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:54:30,344 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-24 00:54:30,348 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-24 00:54:30,510 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:54:30,513 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:54:30,514 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:54:30,515 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:54:30,515 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:54:30,517 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:54:41,961 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-24 00:54:41,964 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-24 00:54:42,130 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:54:42,132 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:54:42,133 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:54:42,134 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:54:42,134 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:54:42,137 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:54:53,362 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-24 00:54:53,365 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-24 00:54:53,518 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:54:53,522 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:54:53,523 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:54:53,523 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:54:53,524 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:54:53,526 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:55:04,860 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-24 00:55:04,863 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-24 00:55:05,018 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:55:05,024 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:55:05,025 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:55:05,026 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:55:05,026 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:55:05,029 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:55:30,574 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-24 00:55:30,580 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-24 00:55:30,971 INFO RegimeClassifier[mode=ltf_behaviour]: 401471 samples, classes={'TRENDING': 160094, 'RANGING': 105286, 'CONSOLIDATING': 53524, 'VOLATILE': 82567}, device=cuda
2026-04-24 00:55:30,972 INFO RegimeClassifier: sample weights — mean=0.505  ambiguous(<0.4)=33.0%
2026-04-24 00:55:30,974 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-04-24 00:55:30,975 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-24 00:55:31,773 INFO Regime epoch  1/50 — tr=0.9468 va=1.7277 acc=0.267 per_class={'TRENDING': 0.086, 'RANGING': 0.205, 'CONSOLIDATING': 0.08, 'VOLATILE': 0.836}
2026-04-24 00:55:32,523 INFO Regime epoch  2/50 — tr=0.9143 va=1.6524 acc=0.329
2026-04-24 00:55:33,221 INFO Regime epoch  3/50 — tr=0.8691 va=1.5717 acc=0.445
2026-04-24 00:55:33,922 INFO Regime epoch  4/50 — tr=0.8258 va=1.5039 acc=0.483
2026-04-24 00:55:34,687 INFO Regime epoch  5/50 — tr=0.7958 va=1.4672 acc=0.494 per_class={'TRENDING': 0.465, 'RANGING': 0.108, 'CONSOLIDATING': 0.731, 'VOLATILE': 0.895}
2026-04-24 00:55:35,408 INFO Regime epoch  6/50 — tr=0.7771 va=1.4586 acc=0.502
2026-04-24 00:55:36,132 INFO Regime epoch  7/50 — tr=0.7642 va=1.4549 acc=0.508
2026-04-24 00:55:36,867 INFO Regime epoch  8/50 — tr=0.7553 va=1.4497 acc=0.517
2026-04-24 00:55:37,600 INFO Regime epoch  9/50 — tr=0.7485 va=1.4437 acc=0.525
2026-04-24 00:55:38,372 INFO Regime epoch 10/50 — tr=0.7433 va=1.4364 acc=0.536 per_class={'TRENDING': 0.532, 'RANGING': 0.115, 'CONSOLIDATING': 0.801, 'VOLATILE': 0.918}
2026-04-24 00:55:39,084 INFO Regime epoch 11/50 — tr=0.7385 va=1.4258 acc=0.545
2026-04-24 00:55:39,798 INFO Regime epoch 12/50 — tr=0.7345 va=1.4130 acc=0.558
2026-04-24 00:55:40,503 INFO Regime epoch 13/50 — tr=0.7311 va=1.4045 acc=0.565
2026-04-24 00:55:41,240 INFO Regime epoch 14/50 — tr=0.7277 va=1.3940 acc=0.574
2026-04-24 00:55:42,023 INFO Regime epoch 15/50 — tr=0.7247 va=1.3874 acc=0.580 per_class={'TRENDING': 0.627, 'RANGING': 0.111, 'CONSOLIDATING': 0.857, 'VOLATILE': 0.915}
2026-04-24 00:55:42,811 INFO Regime epoch 16/50 — tr=0.7219 va=1.3804 acc=0.586
2026-04-24 00:55:43,526 INFO Regime epoch 17/50 — tr=0.7199 va=1.3750 acc=0.591
2026-04-24 00:55:44,230 INFO Regime epoch 18/50 — tr=0.7177 va=1.3680 acc=0.595
2026-04-24 00:55:44,912 INFO Regime epoch 19/50 — tr=0.7156 va=1.3604 acc=0.596
2026-04-24 00:55:45,649 INFO Regime epoch 20/50 — tr=0.7141 va=1.3596 acc=0.596 per_class={'TRENDING': 0.648, 'RANGING': 0.105, 'CONSOLIDATING': 0.917, 'VOLATILE': 0.918}
2026-04-24 00:55:46,334 INFO Regime epoch 21/50 — tr=0.7127 va=1.3553 acc=0.602
2026-04-24 00:55:47,067 INFO Regime epoch 22/50 — tr=0.7116 va=1.3545 acc=0.600
2026-04-24 00:55:47,750 INFO Regime epoch 23/50 — tr=0.7103 va=1.3492 acc=0.608
2026-04-24 00:55:48,478 INFO Regime epoch 24/50 — tr=0.7093 va=1.3446 acc=0.607
2026-04-24 00:55:49,253 INFO Regime epoch 25/50 — tr=0.7085 va=1.3431 acc=0.609 per_class={'TRENDING': 0.679, 'RANGING': 0.103, 'CONSOLIDATING': 0.934, 'VOLATILE': 0.914}
2026-04-24 00:55:49,968 INFO Regime epoch 26/50 — tr=0.7079 va=1.3434 acc=0.609
2026-04-24 00:55:50,670 INFO Regime epoch 27/50 — tr=0.7071 va=1.3393 acc=0.610
2026-04-24 00:55:51,398 INFO Regime epoch 28/50 — tr=0.7063 va=1.3377 acc=0.609
2026-04-24 00:55:52,137 INFO Regime epoch 29/50 — tr=0.7060 va=1.3401 acc=0.607
2026-04-24 00:55:52,947 INFO Regime epoch 30/50 — tr=0.7058 va=1.3362 acc=0.610 per_class={'TRENDING': 0.684, 'RANGING': 0.094, 'CONSOLIDATING': 0.944, 'VOLATILE': 0.915}
2026-04-24 00:55:53,647 INFO Regime epoch 31/50 — tr=0.7052 va=1.3347 acc=0.612
2026-04-24 00:55:54,386 INFO Regime epoch 32/50 — tr=0.7051 va=1.3343 acc=0.612
2026-04-24 00:55:55,109 INFO Regime epoch 33/50 — tr=0.7047 va=1.3345 acc=0.613
2026-04-24 00:55:55,838 INFO Regime epoch 34/50 — tr=0.7043 va=1.3319 acc=0.613
2026-04-24 00:55:56,604 INFO Regime epoch 35/50 — tr=0.7041 va=1.3324 acc=0.613 per_class={'TRENDING': 0.693, 'RANGING': 0.089, 'CONSOLIDATING': 0.949, 'VOLATILE': 0.914}
2026-04-24 00:55:57,311 INFO Regime epoch 36/50 — tr=0.7037 va=1.3297 acc=0.614
2026-04-24 00:55:58,037 INFO Regime epoch 37/50 — tr=0.7035 va=1.3322 acc=0.612
2026-04-24 00:55:58,786 INFO Regime epoch 38/50 — tr=0.7033 va=1.3293 acc=0.615
2026-04-24 00:55:59,548 INFO Regime epoch 39/50 — tr=0.7035 va=1.3294 acc=0.613
2026-04-24 00:56:00,321 INFO Regime epoch 40/50 — tr=0.7032 va=1.3311 acc=0.614 per_class={'TRENDING': 0.688, 'RANGING': 0.098, 'CONSOLIDATING': 0.941, 'VOLATILE': 0.921}
2026-04-24 00:56:01,017 INFO Regime epoch 41/50 — tr=0.7031 va=1.3318 acc=0.611
2026-04-24 00:56:01,766 INFO Regime epoch 42/50 — tr=0.7032 va=1.3296 acc=0.613
2026-04-24 00:56:02,555 INFO Regime epoch 43/50 — tr=0.7030 va=1.3288 acc=0.614
2026-04-24 00:56:03,274 INFO Regime epoch 44/50 — tr=0.7030 va=1.3315 acc=0.614
2026-04-24 00:56:04,093 INFO Regime epoch 45/50 — tr=0.7029 va=1.3265 acc=0.614 per_class={'TRENDING': 0.693, 'RANGING': 0.094, 'CONSOLIDATING': 0.95, 'VOLATILE': 0.914}
2026-04-24 00:56:04,827 INFO Regime epoch 46/50 — tr=0.7029 va=1.3303 acc=0.612
2026-04-24 00:56:05,558 INFO Regime epoch 47/50 — tr=0.7029 va=1.3299 acc=0.612
2026-04-24 00:56:06,281 INFO Regime epoch 48/50 — tr=0.7028 va=1.3269 acc=0.614
2026-04-24 00:56:07,030 INFO Regime epoch 49/50 — tr=0.7029 va=1.3283 acc=0.612
2026-04-24 00:56:07,783 INFO Regime epoch 50/50 — tr=0.7027 va=1.3305 acc=0.613 per_class={'TRENDING': 0.688, 'RANGING': 0.093, 'CONSOLIDATING': 0.946, 'VOLATILE': 0.919}
2026-04-24 00:56:07,833 WARNING RegimeClassifier accuracy 0.61 < 0.65 threshold
2026-04-24 00:56:07,836 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-24 00:56:07,836 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-24 00:56:07,982 INFO Regime LTF complete: acc=0.614, n=401471
2026-04-24 00:56:07,986 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:56:08,549 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-24 00:56:08,554 INFO Regime[1H] persistence (avg bars/run) on XAUUSD 1H:
{0: 9.355291913830783, 1: 6.096303199751476, 2: 5.598885793871866, 3: 6.771351107094442}
2026-04-24 00:56:08,557 INFO Regime[1H] return separation on XAUUSD 1H:
{0: 2.149062259083323e-05, 1: -3.254493608872846e-06, 2: -3.039756158146958e-06, 3: 1.8102764458593193e-05}
2026-04-24 00:56:08,570 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-24 00:56:08,570 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 00:56:08,570 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 00:56:08,571 INFO === VectorStore: building similarity indices ===
2026-04-24 00:56:08,571 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-24 00:56:08,571 INFO Retrain complete.
2026-04-24 00:56:09,728 INFO Model regime: SUCCESS
2026-04-24 00:56:09,728 INFO --- Training gru ---
2026-04-24 00:56:09,728 INFO Running retrain --model gru
2026-04-24 00:56:10,063 INFO retrain environment: KAGGLE
2026-04-24 00:56:11,909 INFO Device: CUDA (2 GPU(s))
2026-04-24 00:56:11,920 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 00:56:11,920 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 00:56:11,922 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-24 00:56:12,073 INFO NumExpr defaulting to 4 threads.
2026-04-24 00:56:12,302 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-24 00:56:12,302 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 00:56:12,302 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 00:56:12,559 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-24 00:56:12,817 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-24 00:56:12,819 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:12,908 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:12,996 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:13,078 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:13,164 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:13,251 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:13,332 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:13,413 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:13,498 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:13,580 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:13,683 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:56:13,761 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-24 00:56:13,763 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260424_005613
2026-04-24 00:56:13,767 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-04-24 00:56:13,897 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:13,898 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:13,915 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:13,923 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:13,925 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-24 00:56:13,925 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 00:56:13,925 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 00:56:13,926 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:14,022 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-24 00:56:14,024 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:14,311 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-24 00:56:14,342 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:14,651 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:14,799 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:14,913 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:15,150 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:15,151 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:15,169 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:15,178 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:15,179 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:15,265 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-24 00:56:15,267 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:15,539 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-24 00:56:15,556 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:15,850 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:15,990 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:16,104 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:16,325 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:16,326 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:16,344 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:16,353 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:16,354 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:16,443 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-24 00:56:16,445 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:16,715 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-24 00:56:16,732 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:17,027 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:17,167 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:17,274 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:17,487 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:17,488 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:17,506 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:17,514 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:17,515 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:17,602 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-24 00:56:17,604 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:17,873 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-24 00:56:17,896 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:18,204 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:18,348 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:18,459 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:18,671 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:18,672 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:18,690 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:18,700 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:18,701 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:18,791 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-24 00:56:18,793 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:19,060 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-24 00:56:19,076 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:19,368 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:19,506 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:19,612 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:19,816 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:19,816 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:19,832 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:19,841 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:19,842 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:19,926 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-24 00:56:19,928 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:20,191 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-24 00:56:20,206 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:20,493 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:20,628 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:20,732 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:20,914 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:56:20,915 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:56:20,931 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:56:20,940 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-24 00:56:20,941 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:21,027 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-24 00:56:21,029 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:21,305 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-24 00:56:21,318 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:21,595 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:21,735 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:21,844 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:22,042 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:22,043 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:22,060 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:22,069 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:22,070 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:22,165 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-24 00:56:22,168 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:22,473 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-24 00:56:22,492 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:22,783 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:22,923 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:23,030 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:23,234 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:23,235 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:23,252 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:23,260 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:23,261 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:23,346 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-24 00:56:23,348 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:23,619 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-24 00:56:23,635 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:23,926 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:24,076 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:24,189 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:24,406 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:24,407 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:24,426 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:24,434 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 00:56:24,435 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:24,523 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-24 00:56:24,525 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:24,798 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-24 00:56:24,814 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:25,122 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:25,275 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:25,395 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 00:56:25,728 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:56:25,729 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:56:25,752 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:56:25,765 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-24 00:56:25,766 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:56:25,949 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-24 00:56:25,953 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:56:26,563 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-24 00:56:26,612 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:56:27,231 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:56:27,455 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:56:27,592 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 00:56:27,719 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-24 00:56:27,719 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-24 00:56:27,720 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-24 01:01:16,446 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-24 01:01:16,447 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-24 01:01:17,788 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-24 01:01:42,101 INFO train_multi TF=ALL epoch 1/50 train=0.6045 val=0.6132
2026-04-24 01:01:42,106 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-24 01:01:42,107 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-24 01:01:42,107 INFO train_multi TF=ALL: new best val=0.6132 — saved
2026-04-24 01:02:00,516 INFO train_multi TF=ALL epoch 2/50 train=0.6043 val=0.6132
2026-04-24 01:02:18,847 INFO train_multi TF=ALL epoch 3/50 train=0.6040 val=0.6134
2026-04-24 01:02:36,659 INFO train_multi TF=ALL epoch 4/50 train=0.6041 val=0.6136
2026-04-24 01:02:54,713 INFO train_multi TF=ALL epoch 5/50 train=0.6040 val=0.6131
2026-04-24 01:02:54,718 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-24 01:02:54,718 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-24 01:02:54,718 INFO train_multi TF=ALL: new best val=0.6131 — saved
2026-04-24 01:03:13,959 INFO train_multi TF=ALL epoch 6/50 train=0.6035 val=0.6140
2026-04-24 01:03:34,468 INFO train_multi TF=ALL epoch 7/50 train=0.6036 val=0.6134
2026-04-24 01:03:53,564 INFO train_multi TF=ALL epoch 8/50 train=0.6032 val=0.6136
2026-04-24 01:04:12,027 INFO train_multi TF=ALL epoch 9/50 train=0.6029 val=0.6139
2026-04-24 01:04:30,256 INFO train_multi TF=ALL epoch 10/50 train=0.6026 val=0.6142
2026-04-24 01:04:30,256 INFO train_multi TF=ALL early stop at epoch 10
2026-04-24 01:04:30,406 INFO === VectorStore: building similarity indices ===
2026-04-24 01:04:30,406 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-24 01:04:30,406 INFO Retrain complete.
2026-04-24 01:04:32,439 INFO Model gru: SUCCESS
2026-04-24 01:04:32,439 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-24 01:04:32,439 WARNING   [MISSING] regime_classifier → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_classifier.pkl
2026-04-24 01:04:32,439 INFO   [OK] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-24 01:04:32,439 WARNING   [MISSING] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-24 01:04:32,439 WARNING Missing weights: ['regime_classifier', 'rl_ppo'] — run retrain_incremental.py for each
2026-04-24 01:04:32,440 INFO Saved 18 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  Cleared journal for fresh Round 1 run
  START Round 1 - Backtest (val)
2026-04-24 01:04:33,005 INFO === STEP 6: BACKTEST + REINFORCED TRAINING (3 rounds, window=round1) ===
2026-04-24 01:04:33,005 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-04-24 01:04:33,006 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-24 01:04:33,006 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260424_010433.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)             2196  55.2%   2.94 2346.3% 55.2% 16.4%   2.6%    5.20
  gate_diagnostics: bars=466496 no_signal=51899 quality_block=0 session_skip=189356 density=5018 pm_reject=27851 daily_skip=179236 cooldown=10940

Calibration Summary:
  all          [OK] Calibration OK — p_win correlates with actual win rate.
  ml_trader    [OK] Calibration OK — p_win correlates with actual win rate.
2026-04-24 01:10:52,464 INFO Round 1 backtest — 2196 trades | avg WR=55.2% | avg PF=2.94 | avg Sharpe=5.20
2026-04-24 01:10:52,464 INFO   ml_trader: 2196 trades | WR=55.2% | PF=2.94 | Return=2346.3% | DD=2.6% | Sharpe=5.20
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 2196
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2196 rows)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  AUDUSD        209 trades   177 days   1.18/day
  EURGBP        195 trades   157 days   1.24/day
  EURJPY        160 trades   129 days   1.24/day
  EURUSD        324 trades   285 days   1.14/day
  GBPJPY        142 trades   110 days   1.29/day
  GBPUSD        234 trades   197 days   1.19/day
  NZDUSD        144 trades   112 days   1.29/day
  USDCAD        207 trades   165 days   1.25/day
  USDCHF        192 trades   155 days   1.24/day
  USDJPY        242 trades   204 days   1.19/day
  XAUUSD        147 trades   114 days   1.29/day
  ✓  All symbols within normal range.

──────────────────────────────────────────────────────────────
CHECK 2: REGIME DISTRIBUTION  (% of trades, WR, avgEV)
──────────────────────────────────────────────────────────────
  BIAS_DOWN           574 trades   26.1%  WR=60.5%  avgEV=1724.400
  BIAS_NEUTRAL       1074 trades   48.9%  WR=51.4%  avgEV=1556.232
  BIAS_UP             548 trades   25.0%  WR=57.1%  avgEV=1734.951
  ⚠  Regimes never traded: ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'VOLATILE', 'CONSOLIDATION']

──────────────────────────────────────────────────────────────
CHECK 3: EV PREDICTED vs REALIZED RR
──────────────────────────────────────────────────────────────
  Pearson  = -0.0338   Spearman = -0.0432

  Bucket                  N     AvgEV     AvgRR   WinRate
  Q1 (low EV)           549   225.580     0.889     57.6%
  Q2                    548   859.494     0.677     52.9%
  Q3                    550  1819.307     0.792     56.5%
  Q4 (high EV)          549  3673.020     0.694     53.7%

  Top-20% EV trades: n=440  avgEV=3926.25  avgRR=0.738  WR=55.5%

  Per-regime EV↔RR correlation:
  Regime                N    Pearson   Spearman       WR     AvgEV
  BIAS_DOWN           574    -0.0382    -0.0215   60.5%  1724.400
  BIAS_NEUTRAL       1074    -0.0403    -0.0623   51.4%  1556.232
  BIAS_UP             548    -0.0284    -0.0322   57.1%  1734.951
  ⚠  EV↔RR Pearson=-0.034 < 0.1 — EV model weak, check training labels
  ⚠  EV↔RR Spearman=-0.043 < 0.15 — EV rankings don't predict outcomes
  ⚠  Non-monotonic bins: Q4 avg_rr=0.694 ≤ Q1 avg_rr=0.889 — EV not predictive
  ⚠  EV↔RR Spearman in BIAS_DOWN = -0.021 — EV useless in this regime
  ⚠  EV↔RR Spearman in BIAS_NEUTRAL = -0.062 — EV useless in this regime
  ⚠  EV↔RR Spearman in BIAS_UP = -0.032 — EV useless in this regime

──────────────────────────────────────────────────────────────
CHECK 4: GRU CONFIDENCE CALIBRATION  (p_bull/bear vs win rate)
──────────────────────────────────────────────────────────────
  ECE = 0.1505  (target < 0.10)
  Bin                   N   Midpoint    WinRate    Error
  [0.58-0.64]         561      0.612      0.471    0.141
  [0.64-0.71]         733      0.677      0.554    0.123
  [0.71-0.77]         488      0.742      0.596    0.146
  [0.77-0.84]         305      0.807      0.564    0.243
  [0.84-0.90]         109      0.871      0.725    0.146
  ⚠  Bin [0.77-0.84]: midpoint=0.81 win_rate=0.56 (err=0.24 > 0.15) — GRU miscalibrated
  ⚠  Win rate non-monotonic across confidence bins — GRU confidence unreliable

──────────────────────────────────────────────────────────────
CHECK 5: GRU ↔ EV MODEL CONSISTENCY  (direction × value agreement)
──────────────────────────────────────────────────────────────
  Pearson=-0.0685  Spearman=-0.0714  Agree=48%

  Quadrants  (conf_threshold=median, ev_threshold=median):
  high_conf + high_ev:   534  ← ideal
  high_conf + low_ev:    569  ← GRU overconfident
  low_conf  + high_ev:   565  ← EV optimistic
  low_conf  + low_ev:    528  ← correct abstention
  ⚠  GRU↔EV Pearson=-0.069 < 0.1 — direction model and EV model disagree (architecture misaligned?)
  ⚠  GRU and EV agree on only 48.4% of trades — models pulling in opposite directions

──────────────────────────────────────────────────────────────
SUMMARY — 11 flag(s):
  ⚠  Regimes never traded: ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'VOLATILE', 'CONSOLIDATION']
  ⚠  EV↔RR Pearson=-0.034 < 0.1 — EV model weak, check training labels
  ⚠  EV↔RR Spearman=-0.043 < 0.15 — EV rankings don't predict outcomes
  ⚠  Non-monotonic bins: Q4 avg_rr=0.694 ≤ Q1 avg_rr=0.889 — EV not predictive
  ⚠  EV↔RR Spearman in BIAS_DOWN = -0.021 — EV useless in this regime
  ⚠  EV↔RR Spearman in BIAS_NEUTRAL = -0.062 — EV useless in this regime
  ⚠  EV↔RR Spearman in BIAS_UP = -0.032 — EV useless in this regime
  ⚠  Bin [0.77-0.84]: midpoint=0.81 win_rate=0.56 (err=0.24 > 0.15) — GRU miscalibrated
  ⚠  Win rate non-monotonic across confidence bins — GRU confidence unreliable
  ⚠  GRU↔EV Pearson=-0.069 < 0.1 — direction model and EV model disagree (architecture misaligned?)
  ⚠  GRU and EV agree on only 48.4% of trades — models pulling in opposite directions
──────────────────────────────────────────────────────────────
2026-04-24 01:10:53,469 INFO Round 1: wrote 2196 journal entries (total in file: 2196)
2026-04-24 01:10:53,471 INFO Round 1 — retraining regime...
2026-04-24 01:16:53,665 INFO Retrain regime: OK
2026-04-24 01:16:53,683 INFO Round 1 — retraining quality...
2026-04-24 01:17:01,896 INFO Retrain quality: OK
2026-04-24 01:17:01,911 INFO Round 1 — retraining rl...
2026-04-24 01:18:03,526 INFO Retrain rl: OK
2026-04-24 01:18:03,544 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-24 01:18:03,544 INFO Round 2 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260424_011804.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)             2160  55.3%   3.07 2438.2% 55.3% 17.0%   2.2%    5.37
  gate_diagnostics: bars=466496 no_signal=51383 quality_block=0 session_skip=191435 density=4898 pm_reject=29945 daily_skip=176110 cooldown=10565

Calibration Summary:
  all          [OK] Calibration OK — p_win correlates with actual win rate.
  ml_trader    [OK] Calibration OK — p_win correlates with actual win rate.
2026-04-24 01:24:47,154 INFO Round 2 backtest — 2160 trades | avg WR=55.3% | avg PF=3.07 | avg Sharpe=5.37
2026-04-24 01:24:47,154 INFO   ml_trader: 2160 trades | WR=55.3% | PF=3.07 | Return=2438.2% | DD=2.2% | Sharpe=5.37
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 2160
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2160 rows)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  AUDUSD        208 trades   177 days   1.18/day
  EURGBP        190 trades   157 days   1.21/day
  EURJPY        156 trades   127 days   1.23/day
  EURUSD        327 trades   288 days   1.14/day
  GBPJPY        141 trades   109 days   1.29/day
  GBPUSD        240 trades   203 days   1.18/day
  NZDUSD        148 trades   117 days   1.26/day
  USDCAD        189 trades   148 days   1.28/day
  USDCHF        186 trades   151 days   1.23/day
  USDJPY        237 trades   200 days   1.19/day
  XAUUSD        138 trades   107 days   1.29/day
  ✓  All symbols within normal range.

──────────────────────────────────────────────────────────────
CHECK 2: REGIME DISTRIBUTION  (% of trades, WR, avgEV)
──────────────────────────────────────────────────────────────
  BIAS_DOWN           519 trades   24.0%  WR=58.4%  avgEV=1775.227
  BIAS_NEUTRAL       1126 trades   52.1%  WR=53.6%  avgEV=1561.671
  BIAS_UP             515 trades   23.8%  WR=55.9%  avgEV=1790.199
  ⚠  Regimes never traded: ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'VOLATILE', 'CONSOLIDATION']

──────────────────────────────────────────────────────────────
CHECK 3: EV PREDICTED vs REALIZED RR
──────────────────────────────────────────────────────────────
  Pearson  = -0.0521   Spearman = -0.0588

  Bucket                  N     AvgEV     AvgRR   WinRate
  Q1 (low EV)           540   252.951     0.921     58.7%
  Q2                    539   866.178     0.746     54.2%
  Q3                    540  1834.370     0.821     56.9%
  Q4 (high EV)          541  3711.116     0.635     51.6%

  Top-20% EV trades: n=432  avgEV=3969.028  avgRR=0.681  WR=53.0%

  Per-regime EV↔RR correlation:
  Regime                N    Pearson   Spearman       WR     AvgEV
  BIAS_DOWN           519    -0.0599    -0.0387   58.4%  1775.227
  BIAS_NEUTRAL       1126    -0.0508    -0.0668   53.6%  1561.671
  BIAS_UP             515    -0.0518    -0.0560   55.9%  1790.199
  ⚠  EV↔RR Pearson=-0.052 < 0.1 — EV model weak, check training labels
  ⚠  EV↔RR Spearman=-0.059 < 0.15 — EV rankings don't predict outcomes
  ⚠  Non-monotonic bins: Q4 avg_rr=0.635 ≤ Q1 avg_rr=0.921 — EV not predictive
  ⚠  EV↔RR Spearman in BIAS_DOWN = -0.039 — EV useless in this regime
  ⚠  EV↔RR Spearman in BIAS_NEUTRAL = -0.067 — EV useless in this regime
  ⚠  EV↔RR Spearman in BIAS_UP = -0.056 — EV useless in this regime

──────────────────────────────────────────────────────────────
CHECK 4: GRU CONFIDENCE CALIBRATION  (p_bull/bear vs win rate)
──────────────────────────────────────────────────────────────
  ECE = 0.1514  (target < 0.10)
  Bin                   N   Midpoint    WinRate    Error
  [0.58-0.64]         549      0.612      0.454    0.158
  [0.64-0.71]         685      0.676      0.565    0.111
  [0.71-0.77]         508      0.740      0.594    0.146
  [0.77-0.84]         301      0.804      0.581    0.223
  [0.84-0.90]         128      0.868      0.688    0.180
  ⚠  Bin [0.58-0.64]: midpoint=0.61 win_rate=0.45 (err=0.16 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.77-0.84]: midpoint=0.80 win_rate=0.58 (err=0.22 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.84-0.90]: midpoint=0.87 win_rate=0.69 (err=0.18 > 0.15) — GRU miscalibrated
  ⚠  Win rate non-monotonic across confidence bins — GRU confidence unreliable

──────────────────────────────────────────────────────────────
CHECK 5: GRU ↔ EV MODEL CONSISTENCY  (direction × value agreement)
──────────────────────────────────────────────────────────────
  Pearson=-0.0825  Spearman=-0.0842  Agree=48%

  Quadrants  (conf_threshold=median, ev_threshold=median):
  high_conf + high_ev:   515  ← ideal
  high_conf + low_ev:    566  ← GRU overconfident
  low_conf  + high_ev:   566  ← EV optimistic
  low_conf  + low_ev:    513  ← correct abstention
  ⚠  GRU↔EV Pearson=-0.083 < 0.1 — direction model and EV model disagree (architecture misaligned?)
  ⚠  GRU and EV agree on only 47.6% of trades — models pulling in opposite directions

──────────────────────────────────────────────────────────────
SUMMARY — 13 flag(s):
  ⚠  Regimes never traded: ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'VOLATILE', 'CONSOLIDATION']
  ⚠  EV↔RR Pearson=-0.052 < 0.1 — EV model weak, check training labels
  ⚠  EV↔RR Spearman=-0.059 < 0.15 — EV rankings don't predict outcomes
  ⚠  Non-monotonic bins: Q4 avg_rr=0.635 ≤ Q1 avg_rr=0.921 — EV not predictive
  ⚠  EV↔RR Spearman in BIAS_DOWN = -0.039 — EV useless in this regime
  ⚠  EV↔RR Spearman in BIAS_NEUTRAL = -0.067 — EV useless in this regime
  ⚠  EV↔RR Spearman in BIAS_UP = -0.056 — EV useless in this regime
  ⚠  Bin [0.58-0.64]: midpoint=0.61 win_rate=0.45 (err=0.16 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.77-0.84]: midpoint=0.80 win_rate=0.58 (err=0.22 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.84-0.90]: midpoint=0.87 win_rate=0.69 (err=0.18 > 0.15) — GRU miscalibrated
  ⚠  Win rate non-monotonic across confidence bins — GRU confidence unreliable
  ⚠  GRU↔EV Pearson=-0.083 < 0.1 — direction model and EV model disagree (architecture misaligned?)
  ⚠  GRU and EV agree on only 47.6% of trades — models pulling in opposite directions
──────────────────────────────────────────────────────────────
2026-04-24 01:24:48,153 INFO Round 2: wrote 2160 journal entries (total in file: 4356)
2026-04-24 01:24:48,155 INFO Round 2 — retraining regime...
2026-04-24 01:31:06,016 INFO Retrain regime: OK
2026-04-24 01:31:06,034 INFO Round 2 — retraining quality...
2026-04-24 01:31:13,634 INFO Retrain quality: OK
2026-04-24 01:31:13,650 INFO Round 2 — retraining rl...
2026-04-24 01:32:41,469 INFO Retrain rl: OK
2026-04-24 01:32:41,488 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-04-24 01:32:41,488 INFO Round 3 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260424_013241.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)             2166  56.1%   3.06 2636.6% 56.1% 17.1%   1.8%    5.35
  gate_diagnostics: bars=466496 no_signal=47893 quality_block=0 session_skip=189232 density=4838 pm_reject=29248 daily_skip=182782 cooldown=10337

Calibration Summary:
  all          [OK] Calibration OK — p_win correlates with actual win rate.
  ml_trader    [OK] Calibration OK — p_win correlates with actual win rate.
2026-04-24 01:38:43,764 INFO Round 3 backtest — 2166 trades | avg WR=56.1% | avg PF=3.06 | avg Sharpe=5.35
2026-04-24 01:38:43,764 INFO   ml_trader: 2166 trades | WR=56.1% | PF=3.06 | Return=2636.6% | DD=1.8% | Sharpe=5.35
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 2166
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2166 rows)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  AUDUSD        205 trades   171 days   1.20/day
  EURGBP        192 trades   156 days   1.23/day
  EURJPY        157 trades   131 days   1.20/day
  EURUSD        331 trades   294 days   1.13/day
  GBPJPY        147 trades   112 days   1.31/day
  GBPUSD        235 trades   198 days   1.19/day
  NZDUSD        150 trades   117 days   1.28/day
  USDCAD        190 trades   148 days   1.28/day
  USDCHF        190 trades   155 days   1.23/day
  USDJPY        234 trades   196 days   1.19/day
  XAUUSD        135 trades   106 days   1.27/day
  ✓  All symbols within normal range.

──────────────────────────────────────────────────────────────
CHECK 2: REGIME DISTRIBUTION  (% of trades, WR, avgEV)
──────────────────────────────────────────────────────────────
  BIAS_DOWN           517 trades   23.9%  WR=59.6%  avgEV=1649.651
  BIAS_NEUTRAL       1176 trades   54.3%  WR=54.1%  avgEV=1468.419
  BIAS_UP             473 trades   21.8%  WR=57.3%  avgEV=1671.241
  ⚠  Regimes never traded: ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'VOLATILE', 'CONSOLIDATION']

──────────────────────────────────────────────────────────────
CHECK 3: EV PREDICTED vs REALIZED RR
──────────────────────────────────────────────────────────────
  Pearson  = -0.0398   Spearman = -0.0510

  Bucket                  N     AvgEV     AvgRR   WinRate
  Q1 (low EV)           542   246.027     0.934     59.4%
  Q2                    541   832.587     0.748     54.3%
  Q3                    541  1710.338     0.829     57.5%
  Q4 (high EV)          542  3433.871     0.697     53.1%

  Top-20% EV trades: n=434  avgEV=3672.419  avgRR=0.723  WR=53.9%

  Per-regime EV↔RR correlation:
  Regime                N    Pearson   Spearman       WR     AvgEV
  BIAS_DOWN           517    -0.0764    -0.0593   59.6%  1649.651
  BIAS_NEUTRAL       1176    -0.0268    -0.0544   54.1%  1468.419
  BIAS_UP             473    -0.0380    -0.0345   57.3%  1671.241
  ⚠  EV↔RR Pearson=-0.040 < 0.1 — EV model weak, check training labels
  ⚠  EV↔RR Spearman=-0.051 < 0.15 — EV rankings don't predict outcomes
  ⚠  Non-monotonic bins: Q4 avg_rr=0.697 ≤ Q1 avg_rr=0.934 — EV not predictive
  ⚠  EV↔RR Spearman in BIAS_DOWN = -0.059 — EV useless in this regime
  ⚠  EV↔RR Spearman in BIAS_NEUTRAL = -0.054 — EV useless in this regime
  ⚠  EV↔RR Spearman in BIAS_UP = -0.035 — EV useless in this regime

──────────────────────────────────────────────────────────────
CHECK 4: GRU CONFIDENCE CALIBRATION  (p_bull/bear vs win rate)
──────────────────────────────────────────────────────────────
  ECE = 0.1445  (target < 0.10)
  Bin                   N   Midpoint    WinRate    Error
  [0.58-0.64]         529      0.612      0.452    0.160
  [0.64-0.71]         704      0.676      0.575    0.101
  [0.71-0.77]         512      0.740      0.602    0.138
  [0.77-0.84]         299      0.804      0.585    0.219
  [0.84-0.90]         129      0.868      0.705    0.163
  ⚠  Bin [0.58-0.64]: midpoint=0.61 win_rate=0.45 (err=0.16 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.77-0.84]: midpoint=0.80 win_rate=0.59 (err=0.22 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.84-0.90]: midpoint=0.87 win_rate=0.71 (err=0.16 > 0.15) — GRU miscalibrated
  ⚠  Win rate non-monotonic across confidence bins — GRU confidence unreliable

──────────────────────────────────────────────────────────────
CHECK 5: GRU ↔ EV MODEL CONSISTENCY  (direction × value agreement)
──────────────────────────────────────────────────────────────
  Pearson=-0.0878  Spearman=-0.0920  Agree=47%

  Quadrants  (conf_threshold=median, ev_threshold=median):
  high_conf + high_ev:   506  ← ideal
  high_conf + low_ev:    577  ← GRU overconfident
  low_conf  + high_ev:   577  ← EV optimistic
  low_conf  + low_ev:    506  ← correct abstention
  ⚠  GRU↔EV Pearson=-0.088 < 0.1 — direction model and EV model disagree (architecture misaligned?)
  ⚠  GRU and EV agree on only 46.7% of trades — models pulling in opposite directions

──────────────────────────────────────────────────────────────
SUMMARY — 13 flag(s):
  ⚠  Regimes never traded: ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'VOLATILE', 'CONSOLIDATION']
  ⚠  EV↔RR Pearson=-0.040 < 0.1 — EV model weak, check training labels
  ⚠  EV↔RR Spearman=-0.051 < 0.15 — EV rankings don't predict outcomes
  ⚠  Non-monotonic bins: Q4 avg_rr=0.697 ≤ Q1 avg_rr=0.934 — EV not predictive
  ⚠  EV↔RR Spearman in BIAS_DOWN = -0.059 — EV useless in this regime
  ⚠  EV↔RR Spearman in BIAS_NEUTRAL = -0.054 — EV useless in this regime
  ⚠  EV↔RR Spearman in BIAS_UP = -0.035 — EV useless in this regime
  ⚠  Bin [0.58-0.64]: midpoint=0.61 win_rate=0.45 (err=0.16 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.77-0.84]: midpoint=0.80 win_rate=0.59 (err=0.22 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.84-0.90]: midpoint=0.87 win_rate=0.71 (err=0.16 > 0.15) — GRU miscalibrated
  ⚠  Win rate non-monotonic across confidence bins — GRU confidence unreliable
  ⚠  GRU↔EV Pearson=-0.088 < 0.1 — direction model and EV model disagree (architecture misaligned?)
  ⚠  GRU and EV agree on only 46.7% of trades — models pulling in opposite directions
──────────────────────────────────────────────────────────────
2026-04-24 01:38:44,757 INFO Round 3: wrote 2166 journal entries (total in file: 6522)
2026-04-24 01:38:44,759 INFO Round 3 (final): retraining after last backtest...
2026-04-24 01:38:44,759 INFO Round 3 — retraining regime...
2026-04-24 01:44:28,830 INFO Retrain regime: OK
2026-04-24 01:44:28,848 INFO Round 3 — retraining quality...
2026-04-24 01:44:41,964 INFO Retrain quality: OK
2026-04-24 01:44:41,980 INFO Round 3 — retraining rl...
2026-04-24 01:46:46,913 INFO Retrain rl: OK
2026-04-24 01:46:46,932 INFO Improvement round 1 → 3: WR +0.9% | PF +0.115 | Sharpe +0.145
2026-04-24 01:46:47,099 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-24 01:46:47,107 INFO Journal entries: 6522
2026-04-24 01:46:47,107 INFO --- Training quality ---
2026-04-24 01:46:47,107 INFO Running retrain --model quality

======================================================================
  BACKTEST + REINFORCED TRAINING COMPLETE  (3 rounds)
======================================================================
  Round     Trades       WR      PF   Sharpe
  ------------------------------------------
  Round 1       2196     55.2%    2.941     5.203
  Round 2       2160     55.3%    3.067     5.369
  Round 3       2166     56.1%    3.056     5.348

  Net improvement (round 1 → 3):
    Win rate:      +0.9%
    Profit factor: +0.115
    Sharpe:        +0.145

  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 6522 entries

=== Round 1 → Retrain Quality + RL ===
  START Round 1 - Quality+RL retrain
2026-04-24 01:46:47,465 INFO retrain environment: KAGGLE
2026-04-24 01:46:49,210 INFO Device: CUDA (2 GPU(s))
2026-04-24 01:46:49,222 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 01:46:49,222 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 01:46:49,223 INFO === QualityScorer retrain ===
2026-04-24 01:46:49,370 INFO NumExpr defaulting to 4 threads.
2026-04-24 01:46:49,563 INFO QualityScorer: CUDA available — using GPU
2026-04-24 01:46:49,764 INFO QualityScorer loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (device=cuda)
2026-04-24 01:46:50,002 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/quality_scorer.pkl_20260424_014650
2026-04-24 01:46:50,217 INFO QualityScorer: 6520 samples, EV stats={'mean': 0.2623257637023926, 'std': 1.2281264066696167, 'n_pos': 3620, 'n_neg': 2900}, device=cuda
2026-04-24 01:46:50,217 INFO QualityScorer: normalised win labels by median_win=0.820 — EV range now [-1, +3]
2026-04-24 01:46:50,218 INFO QualityScorer: warm start from existing weights
2026-04-24 01:46:50,218 INFO QualityScorer: pos_weight=1.00 (n_pos=2881 n_neg=2335)
2026-04-24 01:46:51,904 INFO Quality epoch   1/100 — va_huber=0.7744
2026-04-24 01:46:52,031 INFO Quality epoch   2/100 — va_huber=0.7732
2026-04-24 01:46:52,174 INFO Quality epoch   3/100 — va_huber=0.7736
2026-04-24 01:46:52,315 INFO Quality epoch   4/100 — va_huber=0.7742
2026-04-24 01:46:52,485 INFO Quality epoch   5/100 — va_huber=0.7736
2026-04-24 01:46:53,267 INFO Quality epoch  11/100 — va_huber=0.7733
2026-04-24 01:46:54,551 INFO Quality epoch  21/100 — va_huber=0.7722
2026-04-24 01:46:55,836 INFO Quality epoch  31/100 — va_huber=0.7728
2026-04-24 01:46:57,092 INFO Quality epoch  41/100 — va_huber=0.7726
2026-04-24 01:46:57,478 INFO Quality early stop at epoch 44
2026-04-24 01:46:57,498 INFO QualityScorer EV model: MAE=1.244 dir_acc=0.595 n_val=1304
2026-04-24 01:46:57,502 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-24 01:46:57,570 INFO Retrain complete.
2026-04-24 01:46:58,460 INFO Model quality: SUCCESS
2026-04-24 01:46:58,460 INFO --- Training rl ---
2026-04-24 01:46:58,460 INFO Running retrain --model rl
2026-04-24 01:46:58,656 INFO retrain environment: KAGGLE
2026-04-24 01:47:00,413 INFO Device: CUDA (2 GPU(s))
2026-04-24 01:47:00,424 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 01:47:00,424 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 01:47:00,425 INFO === RLAgent (PPO) retrain ===
2026-04-24 01:47:00,428 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260424_014700
2026-04-24 01:47:01.272395: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1776995221.295393   82673 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1776995221.303385   82673 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1776995221.322779   82673 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1776995221.322832   82673 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1776995221.322838   82673 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1776995221.322842   82673 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-24 01:47:05,906 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-24 01:47:08,766 INFO RLAgent: PPO model loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-24 01:47:08,930 INFO RLAgent: warm start — fine-tuning existing PPO policy (lr=6.00e-05)
2026-04-24 01:49:08,219 INFO RLAgent: retrain complete, 6522 episodes
2026-04-24 01:49:08,220 INFO Retrain complete.
2026-04-24 01:49:09,780 INFO Model rl: SUCCESS
2026-04-24 01:49:09,781 INFO Step 7b complete — summary: /kaggle/working/Multi-Bot/trading-system/ml_training/metrics/training_7b_summary.json
  DONE  Round 1 - Quality+RL retrain

=== Round 2: BLIND backtest on test window (unseen 2yr) ===
  START Round 2 - Blind backtest (test)
2026-04-24 01:49:10,282 INFO === STEP 6: BACKTEST + REINFORCED TRAINING (3 rounds, window=round2) ===
2026-04-24 01:49:10,282 INFO BT_WINDOW=round2 — BLIND backtest: 2023-08-07 → 2025-08-05 (test set)
2026-04-24 01:49:10,284 INFO Cleared existing journal for fresh reinforced training run
2026-04-24 01:49:10,284 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-24 01:49:10,284 INFO Round 1 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260424_014910.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)             2254  51.7%   2.59 1365.5% 51.7% 13.5%   2.9%    4.95
  gate_diagnostics: bars=480221 no_signal=68325 quality_block=0 session_skip=205761 density=7000 pm_reject=46035 daily_skip=137673 cooldown=13173

Calibration Summary:
  all          [OK] Calibration OK — p_win correlates with actual win rate.
  ml_trader    [OK] Calibration OK — p_win correlates with actual win rate.
2026-04-24 01:55:40,116 INFO Round 1 backtest — 2254 trades | avg WR=51.7% | avg PF=2.59 | avg Sharpe=4.95
2026-04-24 01:55:40,116 INFO   ml_trader: 2254 trades | WR=51.7% | PF=2.59 | Return=1365.5% | DD=2.9% | Sharpe=4.95
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 2254
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2254 rows)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  AUDUSD        217 trades   175 days   1.24/day
  EURGBP        208 trades   158 days   1.32/day
  EURJPY        177 trades   130 days   1.36/day
  EURUSD        335 trades   290 days   1.16/day
  GBPJPY        164 trades   121 days   1.35/day
  GBPUSD        278 trades   239 days   1.16/day
  NZDUSD         48 trades    48 days   1.00/day
  USDCAD        212 trades   170 days   1.25/day
  USDCHF        203 trades   155 days   1.31/day
  USDJPY        255 trades   209 days   1.22/day
  XAUUSD        157 trades   120 days   1.31/day
  ✓  All symbols within normal range.

──────────────────────────────────────────────────────────────
CHECK 2: REGIME DISTRIBUTION  (% of trades, WR, avgEV)
──────────────────────────────────────────────────────────────
  BIAS_DOWN           399 trades   17.7%  WR=57.4%  avgEV=1654.781
  BIAS_NEUTRAL       1220 trades   54.1%  WR=50.2%  avgEV=1650.301
  BIAS_UP             635 trades   28.2%  WR=50.9%  avgEV=1805.409
  ⚠  Regimes never traded: ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'VOLATILE', 'CONSOLIDATION']

──────────────────────────────────────────────────────────────
CHECK 3: EV PREDICTED vs REALIZED RR
──────────────
2026-04-24 01:55:41,132 INFO Round 1: wrote 2254 journal entries (total in file: 2254)
2026-04-24 01:55:41,134 INFO Round 1 — retraining regime...
2026-04-24 02:01:38,194 INFO Retrain regime: OK
2026-04-24 02:01:38,213 INFO Round 1 — retraining quality...
2026-04-24 02:01:44,300 INFO Retrain quality: OK
2026-04-24 02:01:44,317 INFO Round 1 — retraining rl...
2026-04-24 02:02:34,429 INFO Retrain rl: OK
2026-04-24 02:02:34,447 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-24 02:02:34,447 INFO Round 2 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-24 02:09:00,345 INFO Round 2 backtest — 2269 trades | avg WR=52.1% | avg PF=2.64 | avg Sharpe=5.01
2026-04-24 02:09:00,345 INFO   ml_trader: 2269 trades | WR=52.1% | PF=2.64 | Return=1439.7% | DD=3.8% | Sharpe=5.01
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 2269
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2269 rows)
2026-04-24 02:09:01,368 INFO Round 2: wrote 2269 journal entries (total in file: 4523)
2026-04-24 02:09:01,370 INFO Round 2 — retraining regime...
2026-04-24 02:14:51,654 INFO Retrain regime: OK
2026-04-24 02:14:51,673 INFO Round 2 — retraining quality...
2026-04-24 02:14:58,302 INFO Retrain quality: OK
2026-04-24 02:14:58,319 INFO Round 2 — retraining rl...
2026-04-24 02:16:25,671 INFO Retrain rl: OK
2026-04-24 02:16:25,688 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-04-24 02:16:25,689 INFO Round 3 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-24 02:22:45,856 INFO Round 3 backtest — 2203 trades | avg WR=52.3% | avg PF=2.62 | avg Sharpe=5.03
2026-04-24 02:22:45,856 INFO   ml_trader: 2203 trades | WR=52.3% | PF=2.62 | Return=1376.0% | DD=2.5% | Sharpe=5.03
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 2203
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2203 rows)
2026-04-24 02:22:46,855 INFO Round 3: wrote 2203 journal entries (total in file: 6726)
2026-04-24 02:22:46,856 INFO Round 3 (final): retraining after last backtest...
2026-04-24 02:22:46,857 INFO Round 3 — retraining regime...
2026-04-24 02:28:30,008 INFO Retrain regime: OK
2026-04-24 02:28:30,025 INFO Round 3 — retraining quality...
2026-04-24 02:28:36,996 INFO Retrain quality: OK
2026-04-24 02:28:37,012 INFO Round 3 — retraining rl...
2026-04-24 02:30:41,143 INFO Retrain rl: OK
2026-04-24 02:30:41,162 INFO Improvement round 1 → 3: WR +0.6% | PF +0.023 | Sharpe +0.089
2026-04-24 02:30:41,318 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-24 02:30:41,326 INFO Journal entries: 6726
2026-04-24 02:30:41,326 INFO --- Training quality ---
2026-04-24 02:30:41,326 INFO Running retrain --model quality
  DONE  Round 2 - Blind backtest (test)
  Saved Round 2 result → round2_summary.json
  Journal after Round 2: 6726 entries

=== Round 2 → Retrain Quality + RL (Round 1+2 journal) ===
  START Round 2 - Quality+RL retrain
2026-04-24 02:30:41,671 INFO retrain environment: KAGGLE
2026-04-24 02:30:43,464 INFO Device: CUDA (2 GPU(s))
2026-04-24 02:30:43,476 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 02:30:43,476 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 02:30:43,477 INFO === QualityScorer retrain ===
2026-04-24 02:30:43,618 INFO NumExpr defaulting to 4 threads.
2026-04-24 02:30:43,811 INFO QualityScorer: CUDA available — using GPU
2026-04-24 02:30:44,019 INFO QualityScorer loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (device=cuda)
2026-04-24 02:30:44,251 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/quality_scorer.pkl_20260424_023044
2026-04-24 02:30:44,470 INFO QualityScorer: 6718 samples, EV stats={'mean': 0.14892378449440002, 'std': 1.187400221824646, 'n_pos': 3492, 'n_neg': 3226}, device=cuda
2026-04-24 02:30:44,470 INFO QualityScorer: normalised win labels by median_win=0.800 — EV range now [-1, +3]
2026-04-24 02:30:44,471 INFO QualityScorer: warm start from existing weights
2026-04-24 02:30:44,471 INFO QualityScorer: pos_weight=1.00 (n_pos=2790 n_neg=2584)
2026-04-24 02:30:46,149 INFO Quality epoch   1/100 — va_huber=0.7374
2026-04-24 02:30:46,282 INFO Quality epoch   2/100 — va_huber=0.7375
2026-04-24 02:30:46,406 INFO Quality epoch   3/100 — va_huber=0.7373
2026-04-24 02:30:46,532 INFO Quality epoch   4/100 — va_huber=0.7372
2026-04-24 02:30:46,659 INFO Quality epoch   5/100 — va_huber=0.7377
2026-04-24 02:30:47,410 INFO Quality epoch  11/100 — va_huber=0.7386
2026-04-24 02:30:47,790 INFO Quality early stop at epoch 14
2026-04-24 02:30:47,811 INFO QualityScorer EV model: MAE=1.214 dir_acc=0.569 n_val=1344
2026-04-24 02:30:47,814 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-24 02:30:47,886 INFO Retrain complete.
2026-04-24 02:30:48,766 INFO Model quality: SUCCESS
2026-04-24 02:30:48,767 INFO --- Training rl ---
2026-04-24 02:30:48,767 INFO Running retrain --model rl
2026-04-24 02:30:49,002 INFO retrain environment: KAGGLE
2026-04-24 02:30:50,702 INFO Device: CUDA (2 GPU(s))
2026-04-24 02:30:50,713 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 02:30:50,713 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 02:30:50,714 INFO === RLAgent (PPO) retrain ===
2026-04-24 02:30:50,717 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260424_023050
2026-04-24 02:30:51.561220: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1776997851.584473  101709 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1776997851.591898  101709 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1776997851.611399  101709 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1776997851.611424  101709 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1776997851.611427  101709 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1776997851.611429  101709 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-24 02:30:56,034 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-24 02:30:58,820 INFO RLAgent: PPO model loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-24 02:30:58,985 INFO RLAgent: warm start — fine-tuning existing PPO policy (lr=6.00e-05)
2026-04-24 02:32:53,751 INFO RLAgent: retrain complete, 6726 episodes
2026-04-24 02:32:53,751 INFO Retrain complete.
2026-04-24 02:32:55,379 INFO Model rl: SUCCESS
2026-04-24 02:32:55,380 INFO Step 7b complete — summary: /kaggle/working/Multi-Bot/trading-system/ml_training/metrics/training_7b_summary.json
  DONE  Round 2 - Quality+RL retrain

=== Round 3: Incremental retrain ALL models on full data ===
  START Retrain gru [full-data retrain]
2026-04-24 02:32:55,751 INFO retrain environment: KAGGLE
2026-04-24 02:32:57,491 INFO Device: CUDA (2 GPU(s))
2026-04-24 02:32:57,503 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 02:32:57,503 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 02:32:57,504 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-24 02:32:57,640 INFO NumExpr defaulting to 4 threads.
2026-04-24 02:32:57,835 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-24 02:32:57,836 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 02:32:57,836 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 02:32:58,073 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-24 02:32:58,313 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-24 02:32:58,314 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:32:58,394 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:32:58,475 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:32:58,557 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:32:58,636 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:32:58,710 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:32:58,783 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:32:58,854 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:32:58,928 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:32:59,004 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:32:59,092 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 02:32:59,151 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-24 02:32:59,153 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260424_023259
2026-04-24 02:32:59,156 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-04-24 02:32:59,276 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:32:59,277 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:32:59,293 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:32:59,306 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:32:59,308 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-24 02:32:59,308 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 02:32:59,308 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 02:32:59,309 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:32:59,388 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-24 02:32:59,390 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:32:59,612 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-24 02:32:59,641 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:32:59,919 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:00,049 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:00,148 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:00,353 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:00,353 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:00,369 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:00,377 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:00,377 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:00,454 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-24 02:33:00,457 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:00,677 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-24 02:33:00,692 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:00,953 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:01,087 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:01,185 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:01,379 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:01,380 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:01,397 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:01,405 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:01,405 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:01,482 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-24 02:33:01,484 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:01,709 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-24 02:33:01,725 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:01,997 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:02,138 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:02,250 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:02,483 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:02,484 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:02,499 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:02,507 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:02,507 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:02,577 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-24 02:33:02,579 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:02,804 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-24 02:33:02,826 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:03,091 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:03,219 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:03,318 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:03,504 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:03,505 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:03,522 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:03,530 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:03,531 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:03,602 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-24 02:33:03,604 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:03,830 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-24 02:33:03,845 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:04,116 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:04,251 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:04,351 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:04,536 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:04,537 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:04,554 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:04,563 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:04,564 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:04,636 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-24 02:33:04,638 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:04,878 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-24 02:33:04,894 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:05,166 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:05,297 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:05,394 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:05,565 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-24 02:33:05,565 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-24 02:33:05,581 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-24 02:33:05,588 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-24 02:33:05,589 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:05,660 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-24 02:33:05,661 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:05,891 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-24 02:33:05,904 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:06,166 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:06,293 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:06,390 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:06,570 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:06,571 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:06,587 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:06,596 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:06,596 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:06,666 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-24 02:33:06,668 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:06,891 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-24 02:33:06,909 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:07,184 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:07,315 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:07,418 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:07,603 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:07,604 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:07,620 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:07,628 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:07,629 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:07,700 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-24 02:33:07,702 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:07,922 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-24 02:33:07,937 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:08,202 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:08,332 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:08,429 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:08,618 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:08,619 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:08,637 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:08,646 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:33:08,647 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:08,720 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-24 02:33:08,722 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:08,949 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-24 02:33:08,964 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:09,231 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:09,361 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:09,463 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:33:09,749 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-24 02:33:09,750 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-24 02:33:09,768 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-24 02:33:09,781 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-24 02:33:09,782 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 02:33:09,931 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-24 02:33:09,934 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 02:33:10,432 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-24 02:33:10,478 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-24 02:33:11,004 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-24 02:33:11,201 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 02:33:11,334 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 02:33:11,450 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-24 02:33:11,450 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-24 02:33:11,450 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-24 02:37:35,841 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-24 02:37:35,841 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-24 02:37:37,153 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-24 02:38:01,225 INFO train_multi TF=ALL epoch 1/50 train=0.6037 val=0.6134
2026-04-24 02:38:01,230 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-24 02:38:01,230 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-24 02:38:01,231 INFO train_multi TF=ALL: new best val=0.6134 — saved
2026-04-24 02:38:19,032 INFO train_multi TF=ALL epoch 2/50 train=0.6035 val=0.6137
2026-04-24 02:38:36,899 INFO train_multi TF=ALL epoch 3/50 train=0.6032 val=0.6137
2026-04-24 02:38:54,770 INFO train_multi TF=ALL epoch 4/50 train=0.6031 val=0.6136
2026-04-24 02:39:12,570 INFO train_multi TF=ALL epoch 5/50 train=0.6029 val=0.6134
2026-04-24 02:39:30,159 INFO train_multi TF=ALL epoch 6/50 train=0.6028 val=0.6140
2026-04-24 02:39:30,159 INFO train_multi TF=ALL early stop at epoch 6
2026-04-24 02:39:30,304 INFO === VectorStore: building similarity indices ===
2026-04-24 02:39:30,304 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-24 02:39:30,305 INFO Retrain complete.
  DONE  Retrain gru [full-data retrain]
  START Retrain regime [full-data retrain]
2026-04-24 02:39:32,557 INFO retrain environment: KAGGLE
2026-04-24 02:39:34,242 INFO Device: CUDA (2 GPU(s))
2026-04-24 02:39:34,251 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 02:39:34,251 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 02:39:34,252 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-24 02:39:34,399 INFO NumExpr defaulting to 4 threads.
2026-04-24 02:39:34,598 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-24 02:39:34,598 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 02:39:34,598 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 02:39:34,812 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-24 02:39:34,815 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:34,893 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:34,968 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,042 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,114 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,188 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,258 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,333 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,407 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,482 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,577 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 02:39:35,636 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-24 02:39:35,651 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,653 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,668 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,670 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,685 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,687 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,701 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,704 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,720 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,724 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,740 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,743 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,758 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,760 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,775 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,778 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,792 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,796 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,812 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,815 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:39:35,833 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 02:39:35,840 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 02:39:36,578 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-24 02:39:58,842 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-24 02:39:58,847 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias)
2026-04-24 02:39:58,850 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-24 02:40:08,905 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-24 02:40:08,907 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias)
2026-04-24 02:40:08,907 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-24 02:40:16,492 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-24 02:40:16,494 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias)
2026-04-24 02:40:16,494 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-24 02:41:27,771 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-24 02:41:27,773 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour)
2026-04-24 02:41:27,773 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-24 02:41:59,970 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-24 02:41:59,974 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour)
2026-04-24 02:41:59,975 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-24 02:42:22,345 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-24 02:42:22,345 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour)
2026-04-24 02:42:22,489 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-24 02:42:22,491 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:42:22,492 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:42:22,492 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:42:22,493 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:42:22,494 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:42:22,495 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:42:22,496 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:42:22,497 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:42:22,498 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:42:22,499 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:42:22,500 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 02:42:22,631 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:22,672 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:22,673 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:22,674 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:22,682 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:22,683 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:42:25,543 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-24 02:42:25,544 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-24 02:42:25,727 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:25,763 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:25,764 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:25,765 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:25,774 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:25,775 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:42:28,561 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-24 02:42:28,562 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-24 02:42:28,741 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:28,776 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:28,777 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:28,778 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:28,787 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:28,788 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:42:31,552 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-24 02:42:31,554 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-24 02:42:31,732 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:31,770 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:31,771 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:31,771 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:31,780 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:31,781 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:42:34,581 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-24 02:42:34,583 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-24 02:42:34,758 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:34,794 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:34,795 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:34,795 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:34,804 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:34,805 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:42:37,647 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-24 02:42:37,649 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-24 02:42:37,824 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:37,858 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:37,858 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:37,859 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:37,868 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:37,869 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:42:40,663 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-24 02:42:40,664 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-24 02:42:40,826 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-24 02:42:40,854 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-24 02:42:40,855 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-24 02:42:40,855 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-24 02:42:40,864 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-24 02:42:40,865 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:42:43,680 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-24 02:42:43,682 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-24 02:42:43,862 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:43,896 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:43,897 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:43,897 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:43,907 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:43,908 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:42:46,751 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-24 02:42:46,753 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-24 02:42:46,929 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:46,964 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:46,964 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:46,965 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:46,974 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:46,975 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:42:49,780 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-24 02:42:49,781 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-24 02:42:49,960 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:49,998 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:49,998 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:49,999 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:50,009 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:42:50,010 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:42:52,862 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-24 02:42:52,863 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-24 02:42:53,139 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-24 02:42:53,199 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-24 02:42:53,200 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-24 02:42:53,200 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-24 02:42:53,211 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-24 02:42:53,212 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 02:42:59,774 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-24 02:42:59,776 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-24 02:42:59,953 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260424_024259
2026-04-24 02:43:00,152 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=34, n_classes=3)
2026-04-24 02:43:00,170 INFO RegimeClassifier[mode=htf_bias]: 103290 samples, classes={'BIAS_UP': 18622, 'BIAS_DOWN': 18286, 'BIAS_NEUTRAL': 66382}, device=cuda
2026-04-24 02:43:00,170 INFO RegimeClassifier: sample weights — mean=0.360  ambiguous(<0.4)=69.5%
2026-04-24 02:43:00,170 INFO RegimeClassifier[mode=htf_bias]: warm start from existing weights
2026-04-24 02:43:00,171 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-24 02:43:02,551 INFO Regime epoch  1/50 — tr=0.4549 va=0.8359 acc=0.705 per_class={'BIAS_UP': 0.996, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.526}
2026-04-24 02:43:02,736 INFO Regime epoch  2/50 — tr=0.4550 va=0.8375 acc=0.700
2026-04-24 02:43:02,914 INFO Regime epoch  3/50 — tr=0.4552 va=0.8361 acc=0.701
2026-04-24 02:43:03,094 INFO Regime epoch  4/50 — tr=0.4549 va=0.8360 acc=0.706
2026-04-24 02:43:03,301 INFO Regime epoch  5/50 — tr=0.4550 va=0.8365 acc=0.706 per_class={'BIAS_UP': 0.995, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.528}
2026-04-24 02:43:03,481 INFO Regime epoch  6/50 — tr=0.4550 va=0.8358 acc=0.705
2026-04-24 02:43:03,670 INFO Regime epoch  7/50 — tr=0.4551 va=0.8349 acc=0.703
2026-04-24 02:43:03,858 INFO Regime epoch  8/50 — tr=0.4550 va=0.8360 acc=0.704
2026-04-24 02:43:04,045 INFO Regime epoch  9/50 — tr=0.4549 va=0.8356 acc=0.706
2026-04-24 02:43:04,241 INFO Regime epoch 10/50 — tr=0.4547 va=0.8346 acc=0.704 per_class={'BIAS_UP': 0.996, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.525}
2026-04-24 02:43:04,433 INFO Regime epoch 11/50 — tr=0.4549 va=0.8348 acc=0.705
2026-04-24 02:43:04,622 INFO Regime epoch 12/50 — tr=0.4549 va=0.8360 acc=0.707
2026-04-24 02:43:04,806 INFO Regime epoch 13/50 — tr=0.4549 va=0.8353 acc=0.707
2026-04-24 02:43:04,980 INFO Regime epoch 14/50 — tr=0.4548 va=0.8352 acc=0.704
2026-04-24 02:43:05,174 INFO Regime epoch 15/50 — tr=0.4550 va=0.8353 acc=0.709 per_class={'BIAS_UP': 0.995, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.533}
2026-04-24 02:43:05,348 INFO Regime epoch 16/50 — tr=0.4548 va=0.8344 acc=0.707
2026-04-24 02:43:05,529 INFO Regime epoch 17/50 — tr=0.4548 va=0.8352 acc=0.707
2026-04-24 02:43:05,709 INFO Regime epoch 18/50 — tr=0.4549 va=0.8320 acc=0.708
2026-04-24 02:43:05,891 INFO Regime epoch 19/50 — tr=0.4547 va=0.8336 acc=0.707
2026-04-24 02:43:06,085 INFO Regime epoch 20/50 — tr=0.4548 va=0.8344 acc=0.707 per_class={'BIAS_UP': 0.996, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.529}
2026-04-24 02:43:06,261 INFO Regime epoch 21/50 — tr=0.4547 va=0.8341 acc=0.704
2026-04-24 02:43:06,444 INFO Regime epoch 22/50 — tr=0.4544 va=0.8328 acc=0.707
2026-04-24 02:43:06,616 INFO Regime epoch 23/50 — tr=0.4545 va=0.8351 acc=0.710
2026-04-24 02:43:06,799 INFO Regime epoch 24/50 — tr=0.4546 va=0.8330 acc=0.707
2026-04-24 02:43:06,997 INFO Regime epoch 25/50 — tr=0.4548 va=0.8344 acc=0.710 per_class={'BIAS_UP': 0.996, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.534}
2026-04-24 02:43:07,172 INFO Regime epoch 26/50 — tr=0.4546 va=0.8323 acc=0.709
2026-04-24 02:43:07,351 INFO Regime epoch 27/50 — tr=0.4545 va=0.8334 acc=0.708
2026-04-24 02:43:07,524 INFO Regime epoch 28/50 — tr=0.4547 va=0.8331 acc=0.709
2026-04-24 02:43:07,524 INFO Regime early stop at epoch 28 (no_improve=10)
2026-04-24 02:43:07,541 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-24 02:43:07,541 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-24 02:43:07,677 INFO Regime HTF complete: acc=0.708, n=103290
2026-04-24 02:43:07,679 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-24 02:43:07,832 INFO Rule labels LTF_BEHAVIOUR [4H]: {'TRENDING': 8970, 'RANGING': 4592, 'CONSOLIDATING': 2247, 'VOLATILE': 4008}  ambiguous=4539 (total=19817)  short_runs_zeroed=570
2026-04-24 02:43:07,835 INFO Regime[4H] persistence (avg bars/run) on XAUUSD 4H:
{0: 12.121621621621621, 1: 6.4858757062146895, 2: 5.575682382133995, 3: 10.329896907216495}
2026-04-24 02:43:07,836 INFO Regime[4H] return separation on XAUUSD 4H:
{0: 5.9502411776840314e-05, 1: 5.664222849552333e-05, 2: -6.138442068010657e-05, 3: 4.239843820627722e-05}
2026-04-24 02:43:07,837 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-24 02:43:07,839 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:43:07,840 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:43:07,842 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:43:07,843 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:43:07,845 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:43:07,847 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:43:07,848 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:43:07,849 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:43:07,851 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:43:07,853 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:43:07,855 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 02:43:07,865 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:07,867 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:07,868 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:07,869 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:07,869 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:07,871 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:43:18,081 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-24 02:43:18,084 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-24 02:43:18,223 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:18,226 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:18,228 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:18,228 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:18,228 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:18,230 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:43:28,254 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-24 02:43:28,257 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-24 02:43:28,393 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:28,395 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:28,396 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:28,397 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:28,397 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:28,399 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:43:38,532 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-24 02:43:38,535 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-24 02:43:38,673 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:38,675 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:38,676 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:38,677 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:38,677 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:38,679 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:43:48,852 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-24 02:43:48,855 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-24 02:43:48,997 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:48,999 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:49,000 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:49,000 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:49,001 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:49,003 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:43:59,373 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-24 02:43:59,376 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-24 02:43:59,522 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:59,524 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:59,525 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:59,526 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:59,526 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:43:59,528 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:44:09,840 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-24 02:44:09,844 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-24 02:44:09,982 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-24 02:44:09,984 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-24 02:44:09,984 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-24 02:44:09,985 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-24 02:44:09,985 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-24 02:44:09,986 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:44:20,216 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-24 02:44:20,219 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-24 02:44:20,363 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:44:20,365 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:44:20,366 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:44:20,366 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:44:20,367 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:44:20,369 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:44:30,478 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-24 02:44:30,481 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-24 02:44:30,622 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:44:30,625 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:44:30,625 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:44:30,626 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:44:30,626 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:44:30,628 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:44:40,659 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-24 02:44:40,662 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-24 02:44:40,806 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:44:40,808 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:44:40,809 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:44:40,810 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:44:40,810 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-24 02:44:40,812 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-24 02:44:51,015 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-24 02:44:51,018 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-24 02:44:51,171 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-24 02:44:51,178 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-24 02:44:51,180 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-24 02:44:51,180 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-24 02:44:51,180 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-24 02:44:51,184 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 02:45:14,208 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-24 02:45:14,214 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-24 02:45:14,523 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260424_024514
2026-04-24 02:45:14,530 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=18, n_classes=4)
2026-04-24 02:45:14,585 INFO RegimeClassifier[mode=ltf_behaviour]: 401471 samples, classes={'TRENDING': 160094, 'RANGING': 105286, 'CONSOLIDATING': 53524, 'VOLATILE': 82567}, device=cuda
2026-04-24 02:45:14,586 INFO RegimeClassifier: sample weights — mean=0.505  ambiguous(<0.4)=33.0%
2026-04-24 02:45:14,586 INFO RegimeClassifier[mode=ltf_behaviour]: warm start from existing weights
2026-04-24 02:45:14,587 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-24 02:45:15,343 INFO Regime epoch  1/50 — tr=0.6975 va=1.3193 acc=0.619 per_class={'TRENDING': 0.703, 'RANGING': 0.094, 'CONSOLIDATING': 0.946, 'VOLATILE': 0.918}
2026-04-24 02:45:16,011 INFO Regime epoch  2/50 — tr=0.6980 va=1.3198 acc=0.620
2026-04-24 02:45:16,675 INFO Regime epoch  3/50 — tr=0.6980 va=1.3191 acc=0.620
2026-04-24 02:45:17,327 INFO Regime epoch  4/50 — tr=0.6978 va=1.3193 acc=0.619
2026-04-24 02:45:18,055 INFO Regime epoch  5/50 — tr=0.6979 va=1.3188 acc=0.619 per_class={'TRENDING': 0.701, 'RANGING': 0.097, 'CONSOLIDATING': 0.948, 'VOLATILE': 0.918}
2026-04-24 02:45:18,727 INFO Regime epoch  6/50 — tr=0.6980 va=1.3147 acc=0.625
2026-04-24 02:45:19,411 INFO Regime epoch  7/50 — tr=0.6979 va=1.3159 acc=0.622
2026-04-24 02:45:20,098 INFO Regime epoch  8/50 — tr=0.6980 va=1.3206 acc=0.619
2026-04-24 02:45:20,778 INFO Regime epoch  9/50 — tr=0.6978 va=1.3183 acc=0.621
2026-04-24 02:45:21,502 INFO Regime epoch 10/50 — tr=0.6978 va=1.3162 acc=0.622 per_class={'TRENDING': 0.71, 'RANGING': 0.094, 'CONSOLIDATING': 0.949, 'VOLATILE': 0.918}
2026-04-24 02:45:22,189 INFO Regime epoch 11/50 — tr=0.6977 va=1.3156 acc=0.622
2026-04-24 02:45:22,892 INFO Regime epoch 12/50 — tr=0.6977 va=1.3155 acc=0.621
2026-04-24 02:45:23,555 INFO Regime epoch 13/50 — tr=0.6977 va=1.3184 acc=0.621
2026-04-24 02:45:24,207 INFO Regime epoch 14/50 — tr=0.6977 va=1.3157 acc=0.622
2026-04-24 02:45:24,917 INFO Regime epoch 15/50 — tr=0.6973 va=1.3195 acc=0.619 per_class={'TRENDING': 0.699, 'RANGING': 0.099, 'CONSOLIDATING': 0.941, 'VOLATILE': 0.923}
2026-04-24 02:45:25,652 INFO Regime epoch 16/50 — tr=0.6972 va=1.3157 acc=0.624
2026-04-24 02:45:25,652 INFO Regime early stop at epoch 16 (no_improve=10)
2026-04-24 02:45:25,703 WARNING RegimeClassifier accuracy 0.63 < 0.65 threshold
2026-04-24 02:45:25,706 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-24 02:45:25,706 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-24 02:45:25,847 INFO Regime LTF complete: acc=0.625, n=401471
2026-04-24 02:45:25,851 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-24 02:45:26,377 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-24 02:45:26,382 INFO Regime[1H] persistence (avg bars/run) on XAUUSD 1H:
{0: 9.355291913830783, 1: 6.096303199751476, 2: 5.598885793871866, 3: 6.771351107094442}
2026-04-24 02:45:26,385 INFO Regime[1H] return separation on XAUUSD 1H:
{0: 2.149062259083323e-05, 1: -3.254493608872846e-06, 2: -3.039756158146958e-06, 3: 1.8102764458593193e-05}
2026-04-24 02:45:26,387 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-24 02:45:26,387 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 02:45:26,388 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 02:45:26,388 INFO === VectorStore: building similarity indices ===
2026-04-24 02:45:26,388 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-24 02:45:26,388 INFO Retrain complete.
  DONE  Retrain regime [full-data retrain]
  START Retrain quality [full-data retrain]
2026-04-24 02:45:27,835 INFO retrain environment: KAGGLE
2026-04-24 02:45:29,574 INFO Device: CUDA (2 GPU(s))
2026-04-24 02:45:29,585 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 02:45:29,585 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 02:45:29,586 INFO === QualityScorer retrain ===
2026-04-24 02:45:29,722 INFO NumExpr defaulting to 4 threads.
2026-04-24 02:45:29,914 INFO QualityScorer: CUDA available — using GPU
2026-04-24 02:45:30,123 INFO QualityScorer loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (device=cuda)
2026-04-24 02:45:30,359 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/quality_scorer.pkl_20260424_024530
2026-04-24 02:45:30,585 INFO QualityScorer: 6718 samples, EV stats={'mean': 0.14892378449440002, 'std': 1.187400221824646, 'n_pos': 3492, 'n_neg': 3226}, device=cuda
2026-04-24 02:45:30,585 INFO QualityScorer: normalised win labels by median_win=0.800 — EV range now [-1, +3]
2026-04-24 02:45:30,586 INFO QualityScorer: warm start from existing weights
2026-04-24 02:45:30,586 INFO QualityScorer: pos_weight=1.00 (n_pos=2790 n_neg=2584)
2026-04-24 02:45:32,285 INFO Quality epoch   1/100 — va_huber=0.7376
2026-04-24 02:45:32,462 INFO Quality epoch   2/100 — va_huber=0.7381
2026-04-24 02:45:32,581 INFO Quality epoch   3/100 — va_huber=0.7380
2026-04-24 02:45:32,706 INFO Quality epoch   4/100 — va_huber=0.7381
2026-04-24 02:45:32,829 INFO Quality epoch   5/100 — va_huber=0.7382
2026-04-24 02:45:33,576 INFO Quality epoch  11/100 — va_huber=0.7389
2026-04-24 02:45:33,576 INFO Quality early stop at epoch 11
2026-04-24 02:45:33,598 INFO QualityScorer EV model: MAE=1.216 dir_acc=0.571 n_val=1344
2026-04-24 02:45:33,601 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-24 02:45:33,676 INFO Retrain complete.
  DONE  Retrain quality [full-data retrain]
  START Retrain rl [full-data retrain]
2026-04-24 02:45:34,742 INFO retrain environment: KAGGLE
2026-04-24 02:45:36,459 INFO Device: CUDA (2 GPU(s))
2026-04-24 02:45:36,470 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-24 02:45:36,470 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-24 02:45:36,471 INFO === RLAgent (PPO) retrain ===
2026-04-24 02:45:36,473 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260424_024536
2026-04-24 02:45:37.314816: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1776998737.337777  118834 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1776998737.345194  118834 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1776998737.364666  118834 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1776998737.364693  118834 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1776998737.364696  118834 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1776998737.364698  118834 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-24 02:45:41,750 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-24 02:45:44,603 INFO RLAgent: PPO model loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-24 02:45:44,774 INFO RLAgent: warm start — fine-tuning existing PPO policy (lr=6.00e-05)
2026-04-24 02:47:44,174 INFO RLAgent: retrain complete, 6726 episodes
2026-04-24 02:47:44,175 INFO Retrain complete.
  DONE  Retrain rl [full-data retrain]

=== Round 3: Backtest on last 3yr (post-retrain evaluation) ===
  START Round 3 - Post-retrain backtest (last 3yr)
2026-04-24 02:47:46,237 INFO === STEP 6: BACKTEST + REINFORCED TRAINING (3 rounds, window=round3) ===
2026-04-24 02:47:46,238 INFO BT_WINDOW=round3 — post-retrain eval: 2022-08-05 → 2025-08-05 (last 3yr)
2026-04-24 02:47:46,240 INFO Cleared existing journal for fresh reinforced training run
2026-04-24 02:47:46,240 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-24 02:47:46,240 INFO Round 1 — running backtest: 2022-08-05 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-24 02:56:34,214 INFO Round 1 backtest — 2452 trades | avg WR=55.8% | avg PF=2.58 | avg Sharpe=4.83
2026-04-24 02:56:34,215 INFO   ml_trader: 2452 trades | WR=55.8% | PF=2.58 | Return=3806.8% | DD=2.3% | Sharpe=4.83
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 2452
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2452 rows)
2026-04-24 02:56:35,288 INFO Round 1: wrote 2452 journal entries (total in file: 2452)
2026-04-24 02:56:35,289 INFO Round 1 — retraining regime...
2026-04-24 03:02:10,329 INFO Retrain regime: OK
2026-04-24 03:02:10,348 INFO Round 1 — retraining quality...
2026-04-24 03:02:16,487 INFO Retrain quality: OK
2026-04-24 03:02:16,504 INFO Round 1 — retraining rl...
2026-04-24 03:03:08,730 INFO Retrain rl: OK
2026-04-24 03:03:08,747 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-24 03:03:08,748 INFO Round 2 — running backtest: 2022-08-05 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-24 03:12:10,005 INFO Round 2 backtest — 2458 trades | avg WR=55.9% | avg PF=2.65 | avg Sharpe=4.90
2026-04-24 03:12:10,005 INFO   ml_trader: 2458 trades | WR=55.9% | PF=2.65 | Return=4014.8% | DD=2.3% | Sharpe=4.90
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 2458
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2458 rows)
2026-04-24 03:12:11,089 INFO Round 2: wrote 2458 journal entries (total in file: 4910)
2026-04-24 03:12:11,091 INFO Round 2 — retraining regime...
2026-04-24 03:18:03,712 INFO Retrain regime: OK
2026-04-24 03:18:03,729 INFO Round 2 — retraining quality...
2026-04-24 03:18:10,157 INFO Retrain quality: OK
2026-04-24 03:18:10,173 INFO Round 2 — retraining rl...
2026-04-24 03:19:41,261 INFO Retrain rl: OK
2026-04-24 03:19:41,278 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-04-24 03:19:41,278 INFO Round 3 — running backtest: 2022-08-05 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-24 03:28:33,598 INFO Round 3 backtest — 2455 trades | avg WR=55.8% | avg PF=2.64 | avg Sharpe=4.89
2026-04-24 03:28:33,598 INFO   ml_trader: 2455 trades | WR=55.8% | PF=2.64 | Return=3838.0% | DD=2.3% | Sharpe=4.89
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 2455
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2455 rows)
2026-04-24 03:28:34,674 INFO Round 3: wrote 2455 journal entries (total in file: 7365)
2026-04-24 03:28:34,675 INFO Round 3 (final): retraining after last backtest...
2026-04-24 03:28:34,676 INFO Round 3 — retraining regime...
2026-04-24 03:34:10,554 INFO Retrain regime: OK
2026-04-24 03:34:10,571 INFO Round 3 — retraining quality...
2026-04-24 03:34:23,939 INFO Retrain quality: OK
2026-04-24 03:34:23,955 INFO Round 3 — retraining rl...
2026-04-24 03:36:36,016 INFO Retrain rl: OK
2026-04-24 03:36:36,034 INFO Improvement round 1 → 3: WR +0.1% | PF +0.053 | Sharpe +0.069
  DONE  Round 3 - Post-retrain backtest (last 3yr)
  Saved Round 3 result → round3_summary.json

======================================================================
  BLIND BACKTEST PIPELINE COMPLETE
======================================================================
  Round 1 (val window)          trades=?  WR=0.0%  PF=0.000  Sharpe=0.000
  Round 2 (blind test)          trades=?  WR=0.0%  PF=0.000  Sharpe=0.000
  Round 3 (last 3yr)            trades=?  WR=0.0%  PF=0.000  Sharpe=0.000


=== STEP 8: Pushing outputs to GitHub ===
INFO  === STEP 8: PUSH TRAINING OUTPUTS TO GITHUB ===
INFO  Repo:   AnalystTKZ/Multi-Bot
INFO  Branch: main
INFO  Root:   /kaggle/working/remote/Multi-Bot
INFO  Pulling latest from origin/main ...
INFO  Collecting artifacts ...
INFO    copied: model.pt → trading-system/trading-engine/weights/gru_lstm/model.pt
INFO    copied: regime_4h.pkl → trading-system/trading-engine/weights/regime_4h.pkl
INFO    copied: regime_1h.pkl → trading-system/trading-engine/weights/regime_1h.pkl
INFO    copied: quality_scorer.pkl → trading-system/trading-engine/weights/quality_scorer.pkl
INFO    copied: model.zip → trading-system/trading-engine/weights/rl_ppo/model.zip
INFO    copied: macro_correlations.json → trading-system/trading-engine/weights/macro_correlations.json
INFO    copied: weights_manifest.json → trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
INFO    copied: training_summary.json → trading-system/ml_training/metrics/training_summary.json
INFO    copied: backtest_diagnostics.csv → trading-system/trading-engine/logs/backtest_diagnostics.csv
INFO    copied: trade_journal_detailed.jsonl → trading-system/trading-engine/logs/trade_journal_detailed.jsonl
INFO    copied: latest_summary.json → trading-system/backtesting/results/latest_summary.json
INFO    copied: backtest_20260424_020234.json → trading-system/trading-engine/backtest_results/backtest_20260424_020234.json
INFO    copied: backtest_20260424_024746.json → trading-system/trading-engine/backtest_results/backtest_20260424_024746.json
INFO    copied: backtest_20260418_123031.json → trading-system/trading-engine/backtest_results/backtest_20260418_123031.json
INFO    copied: backtest_20260424_011804.json → trading-system/trading-engine/backtest_results/backtest_20260424_011804.json
INFO    copied: backtest_20260419_025122.json → trading-system/trading-engine/backtest_results/backtest_20260419_025122.json
INFO    copied: backtest_20260424_031941.json → trading-system/trading-engine/backtest_results/backtest_20260424_031941.json
INFO    copied: backtest_20260419_024259.json → trading-system/trading-engine/backtest_results/backtest_20260419_024259.json
INFO    copied: backtest_20260418_124010.json → trading-system/trading-engine/backtest_results/backtest_20260418_124010.json
INFO    copied: backtest_20260424_030309.json → trading-system/trading-engine/backtest_results/backtest_20260424_030309.json
INFO    copied: backtest_20260419_050108.json → trading-system/trading-engine/backtest_results/backtest_20260419_050108.json
INFO    copied: backtest_20260424_014910.json → trading-system/trading-engine/backtest_results/backtest_20260424_014910.json
INFO    copied: backtest_20260424_021626.json → trading-system/trading-engine/backtest_results/backtest_20260424_021626.json
INFO    copied: backtest_20260419_045224.json → trading-system/trading-engine/backtest_results/backtest_20260419_045224.json
INFO    copied: backtest_20260424_013241.json → trading-system/trading-engine/backtest_results/backtest_20260424_013241.json
INFO    copied: backtest_20260424_010433.json → trading-system/trading-engine/backtest_results/backtest_20260424_010433.json
INFO    copied: backtest_20260418_125100.json → trading-system/trading-engine/backtest_results/backtest_20260418_125100.json
INFO    copied: backtest_20260419_051038.json → trading-system/trading-engine/backtest_results/backtest_20260419_051038.json
INFO    copied: backtest_20260419_030034.json → trading-system/trading-engine/backtest_results/backtest_20260419_030034.json
INFO  Collected 29 file(s)
INFO  Pushing to origin/main ...
=== GitHub push complete ===
INFO  === PUSH COMPLETE ===
