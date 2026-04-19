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
2026-04-19 14:42:37,863 INFO Loading feature-engineered data...
2026-04-19 14:42:38,510 INFO Loaded 221743 rows, 202 features
2026-04-19 14:42:38,512 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-19 14:42:38,514 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-19 14:42:38,515 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-19 14:42:38,515 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-19 14:42:38,515 INFO No leakage confirmed: train < val < test timestamps

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
2026-04-19 14:42:40,949 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-19 14:42:40,949 INFO --- Training regime ---
2026-04-19 14:42:40,949 INFO Running retrain --model regime
2026-04-19 14:42:41,229 INFO retrain environment: KAGGLE
2026-04-19 14:42:42,926 INFO Device: CUDA (2 GPU(s))
2026-04-19 14:42:42,937 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 14:42:42,937 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 14:42:42,938 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-19 14:42:43,085 INFO NumExpr defaulting to 4 threads.
2026-04-19 14:42:43,297 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-19 14:42:43,298 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 14:42:43,298 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 14:42:43,529 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-19 14:42:43,531 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:43,633 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:43,753 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:43,830 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:43,906 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:43,982 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,051 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,124 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,198 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,273 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,364 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 14:42:44,425 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-19 14:42:44,441 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,443 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,458 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,460 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,475 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,477 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,492 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,495 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,510 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,513 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,529 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,532 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,547 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,549 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,564 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,568 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,583 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,586 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,602 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,605 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:42:44,625 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 14:42:44,632 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 14:42:45,786 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-19 14:43:09,462 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-19 14:43:09,464 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias)
2026-04-19 14:43:09,464 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-19 14:43:20,045 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-19 14:43:20,049 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias)
2026-04-19 14:43:20,049 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-19 14:43:27,989 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-19 14:43:27,990 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias)
2026-04-19 14:43:27,990 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-19 14:44:37,830 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-19 14:44:37,833 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour)
2026-04-19 14:44:37,833 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-19 14:45:08,961 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-19 14:45:08,961 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour)
2026-04-19 14:45:08,962 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-19 14:45:31,261 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-19 14:45:31,262 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour)
2026-04-19 14:45:31,368 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-19 14:45:31,370 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:31,371 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:31,372 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:31,373 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:31,374 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:31,375 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:31,376 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:31,377 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:31,378 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:31,379 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:31,380 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 14:45:31,514 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:31,558 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:31,558 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:31,559 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:31,568 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:31,569 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:31,974 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2439, 'BIAS_DOWN': 2349, 'BIAS_NEUTRAL': 3614}  ambiguous=4099 (total=8402)  short_runs_zeroed=649
2026-04-19 14:45:31,975 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-19 14:45:32,164 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:32,201 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:32,202 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:32,202 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:32,211 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:32,212 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:32,579 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2503, 'BIAS_DOWN': 2397, 'BIAS_NEUTRAL': 3502}  ambiguous=4050 (total=8402)  short_runs_zeroed=744
2026-04-19 14:45:32,580 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-19 14:45:32,759 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:32,794 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:32,795 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:32,795 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:32,805 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:32,805 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:33,173 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2482, 'BIAS_DOWN': 2360, 'BIAS_NEUTRAL': 3560}  ambiguous=3926 (total=8402)  short_runs_zeroed=522
2026-04-19 14:45:33,174 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-19 14:45:33,346 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:33,382 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:33,382 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:33,383 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:33,393 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:33,394 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:33,756 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2460, 'BIAS_DOWN': 2376, 'BIAS_NEUTRAL': 3566}  ambiguous=3995 (total=8402)  short_runs_zeroed=585
2026-04-19 14:45:33,757 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-19 14:45:33,934 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:33,971 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:33,972 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:33,973 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:33,981 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:33,982 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:34,339 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2348, 'BIAS_DOWN': 2535, 'BIAS_NEUTRAL': 3520}  ambiguous=3831 (total=8403)  short_runs_zeroed=439
2026-04-19 14:45:34,340 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-19 14:45:34,510 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:34,544 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:34,544 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:34,545 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:34,554 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:34,555 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:34,911 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2440, 'BIAS_DOWN': 2280, 'BIAS_NEUTRAL': 3683}  ambiguous=4047 (total=8403)  short_runs_zeroed=508
2026-04-19 14:45:34,912 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-19 14:45:35,068 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-19 14:45:35,097 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-19 14:45:35,098 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-19 14:45:35,098 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-19 14:45:35,106 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-19 14:45:35,107 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:35,466 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2526, 'BIAS_DOWN': 2331, 'BIAS_NEUTRAL': 3545}  ambiguous=3965 (total=8402)  short_runs_zeroed=561
2026-04-19 14:45:35,467 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-19 14:45:35,641 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:35,676 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:35,677 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:35,677 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:35,687 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:35,688 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:36,053 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2250, 'BIAS_DOWN': 2599, 'BIAS_NEUTRAL': 3553}  ambiguous=3919 (total=8402)  short_runs_zeroed=547
2026-04-19 14:45:36,054 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-19 14:45:36,224 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:36,258 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:36,259 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:36,259 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:36,268 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:36,270 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:36,633 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2214, 'BIAS_DOWN': 2634, 'BIAS_NEUTRAL': 3554}  ambiguous=3934 (total=8402)  short_runs_zeroed=549
2026-04-19 14:45:36,634 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-19 14:45:36,809 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:36,845 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:36,845 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:36,846 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:36,855 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:36,856 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:37,218 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2278, 'BIAS_DOWN': 2515, 'BIAS_NEUTRAL': 3610}  ambiguous=3986 (total=8403)  short_runs_zeroed=516
2026-04-19 14:45:37,219 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-19 14:45:37,490 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-19 14:45:37,551 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-19 14:45:37,552 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-19 14:45:37,552 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-19 14:45:37,564 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-19 14:45:37,565 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 14:45:38,340 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 6629, 'BIAS_DOWN': 5241, 'BIAS_NEUTRAL': 7947}  ambiguous=8645 (total=19817)  short_runs_zeroed=980
2026-04-19 14:45:38,341 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-19 14:45:38,549 INFO RegimeClassifier[mode=htf_bias]: 103290 samples, classes={'BIAS_UP': 30569, 'BIAS_DOWN': 29617, 'BIAS_NEUTRAL': 43104}, device=cuda
2026-04-19 14:45:38,549 INFO RegimeClassifier: sample weights — mean=0.413  ambiguous(<0.4)=47.0%
2026-04-19 14:45:38,841 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-04-19 14:45:38,842 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-19 14:45:43,851 INFO Regime epoch  1/50 — tr=0.5587 va=1.1117 acc=0.470 per_class={'BIAS_UP': 0.021, 'BIAS_DOWN': 0.773, 'BIAS_NEUTRAL': 0.658}
2026-04-19 14:45:44,027 INFO Regime epoch  2/50 — tr=0.5515 va=1.0652 acc=0.470
2026-04-19 14:45:44,209 INFO Regime epoch  3/50 — tr=0.5345 va=1.0055 acc=0.516
2026-04-19 14:45:44,391 INFO Regime epoch  4/50 — tr=0.5090 va=0.9274 acc=0.623
2026-04-19 14:45:44,580 INFO Regime epoch  5/50 — tr=0.4794 va=0.8484 acc=0.662 per_class={'BIAS_UP': 0.937, 'BIAS_DOWN': 0.972, 'BIAS_NEUTRAL': 0.227}
2026-04-19 14:45:44,759 INFO Regime epoch  6/50 — tr=0.4548 va=0.7763 acc=0.684
2026-04-19 14:45:44,942 INFO Regime epoch  7/50 — tr=0.4367 va=0.7232 acc=0.699
2026-04-19 14:45:45,117 INFO Regime epoch  8/50 — tr=0.4261 va=0.6927 acc=0.708
2026-04-19 14:45:45,289 INFO Regime epoch  9/50 — tr=0.4199 va=0.6767 acc=0.711
2026-04-19 14:45:45,482 INFO Regime epoch 10/50 — tr=0.4150 va=0.6681 acc=0.717 per_class={'BIAS_UP': 0.997, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.296}
2026-04-19 14:45:45,653 INFO Regime epoch 11/50 — tr=0.4112 va=0.6642 acc=0.717
2026-04-19 14:45:45,821 INFO Regime epoch 12/50 — tr=0.4082 va=0.6591 acc=0.720
2026-04-19 14:45:46,003 INFO Regime epoch 13/50 — tr=0.4059 va=0.6572 acc=0.718
2026-04-19 14:45:46,180 INFO Regime epoch 14/50 — tr=0.4038 va=0.6577 acc=0.721
2026-04-19 14:45:46,365 INFO Regime epoch 15/50 — tr=0.4020 va=0.6556 acc=0.724 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.31}
2026-04-19 14:45:46,553 INFO Regime epoch 16/50 — tr=0.4008 va=0.6544 acc=0.730
2026-04-19 14:45:46,721 INFO Regime epoch 17/50 — tr=0.3995 va=0.6507 acc=0.726
2026-04-19 14:45:46,888 INFO Regime epoch 18/50 — tr=0.3982 va=0.6507 acc=0.729
2026-04-19 14:45:47,054 INFO Regime epoch 19/50 — tr=0.3975 va=0.6501 acc=0.731
2026-04-19 14:45:47,242 INFO Regime epoch 20/50 — tr=0.3963 va=0.6486 acc=0.731 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.329}
2026-04-19 14:45:47,424 INFO Regime epoch 21/50 — tr=0.3958 va=0.6497 acc=0.731
2026-04-19 14:45:47,591 INFO Regime epoch 22/50 — tr=0.3951 va=0.6478 acc=0.730
2026-04-19 14:45:47,764 INFO Regime epoch 23/50 — tr=0.3945 va=0.6488 acc=0.735
2026-04-19 14:45:47,937 INFO Regime epoch 24/50 — tr=0.3941 va=0.6471 acc=0.733
2026-04-19 14:45:48,123 INFO Regime epoch 25/50 — tr=0.3937 va=0.6488 acc=0.736 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.339}
2026-04-19 14:45:48,303 INFO Regime epoch 26/50 — tr=0.3932 va=0.6460 acc=0.735
2026-04-19 14:45:48,479 INFO Regime epoch 27/50 — tr=0.3927 va=0.6479 acc=0.737
2026-04-19 14:45:48,658 INFO Regime epoch 28/50 — tr=0.3924 va=0.6458 acc=0.736
2026-04-19 14:45:48,835 INFO Regime epoch 29/50 — tr=0.3922 va=0.6462 acc=0.735
2026-04-19 14:45:49,029 INFO Regime epoch 30/50 — tr=0.3920 va=0.6463 acc=0.736 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.339}
2026-04-19 14:45:49,202 INFO Regime epoch 31/50 — tr=0.3920 va=0.6466 acc=0.736
2026-04-19 14:45:49,371 INFO Regime epoch 32/50 — tr=0.3915 va=0.6449 acc=0.736
2026-04-19 14:45:49,541 INFO Regime epoch 33/50 — tr=0.3914 va=0.6480 acc=0.737
2026-04-19 14:45:49,711 INFO Regime epoch 34/50 — tr=0.3914 va=0.6459 acc=0.735
2026-04-19 14:45:49,894 INFO Regime epoch 35/50 — tr=0.3909 va=0.6459 acc=0.735 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.338}
2026-04-19 14:45:50,066 INFO Regime epoch 36/50 — tr=0.3910 va=0.6444 acc=0.737
2026-04-19 14:45:50,235 INFO Regime epoch 37/50 — tr=0.3910 va=0.6453 acc=0.736
2026-04-19 14:45:50,405 INFO Regime epoch 38/50 — tr=0.3908 va=0.6463 acc=0.737
2026-04-19 14:45:50,576 INFO Regime epoch 39/50 — tr=0.3909 va=0.6466 acc=0.736
2026-04-19 14:45:50,761 INFO Regime epoch 40/50 — tr=0.3907 va=0.6451 acc=0.736 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.34}
2026-04-19 14:45:50,953 INFO Regime epoch 41/50 — tr=0.3905 va=0.6458 acc=0.737
2026-04-19 14:45:51,129 INFO Regime epoch 42/50 — tr=0.3907 va=0.6462 acc=0.737
2026-04-19 14:45:51,306 INFO Regime epoch 43/50 — tr=0.3905 va=0.6459 acc=0.737
2026-04-19 14:45:51,482 INFO Regime epoch 44/50 — tr=0.3904 va=0.6453 acc=0.737
2026-04-19 14:45:51,666 INFO Regime epoch 45/50 — tr=0.3906 va=0.6452 acc=0.736 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.341}
2026-04-19 14:45:51,846 INFO Regime epoch 46/50 — tr=0.3905 va=0.6459 acc=0.735
2026-04-19 14:45:51,846 INFO Regime early stop at epoch 46 (no_improve=10)
2026-04-19 14:45:51,863 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-19 14:45:51,863 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-19 14:45:51,991 INFO Regime HTF complete: acc=0.737, n=103290
2026-04-19 14:45:51,992 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 14:45:52,155 INFO Rule labels LTF_BEHAVIOUR [4H]: {'TRENDING': 8970, 'RANGING': 4592, 'CONSOLIDATING': 2247, 'VOLATILE': 4008}  ambiguous=4132 (total=19817)  short_runs_zeroed=570
2026-04-19 14:45:52,163 INFO Regime[4H] persistence (avg bars/run) on XAUUSD 4H:
{0: 12.121621621621621, 1: 6.4858757062146895, 2: 5.575682382133995, 3: 10.329896907216495}
2026-04-19 14:45:52,164 INFO Regime[4H] return separation on XAUUSD 4H:
{0: 5.9502411776840314e-05, 1: 5.664222849552333e-05, 2: -6.138442068010657e-05, 3: 4.239843820627722e-05}
2026-04-19 14:45:52,165 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-19 14:45:52,166 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:52,168 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:52,169 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:52,171 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:52,173 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:52,174 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:52,175 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:52,177 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:52,179 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:52,180 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:52,183 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 14:45:52,193 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:52,195 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:52,196 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:52,196 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:52,197 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:52,199 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:52,796 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=10885 (total=32738)  short_runs_zeroed=4986
2026-04-19 14:45:52,798 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-19 14:45:52,952 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:52,954 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:52,955 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:52,955 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:52,956 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:52,958 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:53,549 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=10483 (total=32738)  short_runs_zeroed=4347
2026-04-19 14:45:53,552 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-19 14:45:53,691 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:53,693 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:53,694 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:53,694 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:53,694 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:53,696 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:54,241 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10628 (total=32740)  short_runs_zeroed=4399
2026-04-19 14:45:54,244 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-19 14:45:54,382 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:54,385 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:54,385 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:54,386 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:54,386 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:54,388 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:54,935 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10080 (total=32739)  short_runs_zeroed=3955
2026-04-19 14:45:54,938 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-19 14:45:55,076 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:55,079 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:55,080 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:55,080 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:55,080 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:55,082 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:55,641 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10370 (total=32740)  short_runs_zeroed=4397
2026-04-19 14:45:55,643 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-19 14:45:55,784 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:55,787 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:55,787 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:55,788 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:55,788 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:55,790 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:56,357 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=9824 (total=32739)  short_runs_zeroed=3724
2026-04-19 14:45:56,360 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-19 14:45:56,500 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-19 14:45:56,501 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-19 14:45:56,502 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-19 14:45:56,502 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-19 14:45:56,503 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-19 14:45:56,504 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:57,045 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10686 (total=32739)  short_runs_zeroed=4898
2026-04-19 14:45:57,048 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-19 14:45:57,180 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:57,182 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:57,183 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:57,184 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:57,184 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:57,186 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:57,750 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=9919 (total=32740)  short_runs_zeroed=3880
2026-04-19 14:45:57,753 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-19 14:45:57,890 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:57,894 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:57,895 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:57,895 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:57,895 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:57,897 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:58,462 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=10596 (total=32741)  short_runs_zeroed=3896
2026-04-19 14:45:58,465 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-19 14:45:58,603 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:58,607 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:58,608 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:58,608 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:58,608 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:45:58,611 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:45:59,167 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10093 (total=32743)  short_runs_zeroed=4275
2026-04-19 14:45:59,170 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-19 14:45:59,320 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-19 14:45:59,323 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-19 14:45:59,325 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-19 14:45:59,325 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-19 14:45:59,325 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-19 14:45:59,329 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 14:46:00,560 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=23184 (total=74624)  short_runs_zeroed=9134
2026-04-19 14:46:00,565 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-19 14:46:00,935 INFO RegimeClassifier[mode=ltf_behaviour]: 401471 samples, classes={'TRENDING': 160094, 'RANGING': 105286, 'CONSOLIDATING': 53524, 'VOLATILE': 82567}, device=cuda
2026-04-19 14:46:00,936 INFO RegimeClassifier: sample weights — mean=0.532  ambiguous(<0.4)=31.6%
2026-04-19 14:46:00,939 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-04-19 14:46:00,940 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-19 14:46:01,690 INFO Regime epoch  1/50 — tr=0.8628 va=1.4609 acc=0.203 per_class={'TRENDING': 0.279, 'RANGING': 0.145, 'CONSOLIDATING': 0.231, 'VOLATILE': 0.11}
2026-04-19 14:46:02,339 INFO Regime epoch  2/50 — tr=0.8185 va=1.3330 acc=0.334
2026-04-19 14:46:02,983 INFO Regime epoch  3/50 — tr=0.7568 va=1.2037 acc=0.432
2026-04-19 14:46:03,630 INFO Regime epoch  4/50 — tr=0.7007 va=1.1249 acc=0.463
2026-04-19 14:46:04,323 INFO Regime epoch  5/50 — tr=0.6664 va=1.0828 acc=0.473 per_class={'TRENDING': 0.457, 'RANGING': 0.004, 'CONSOLIDATING': 0.797, 'VOLATILE': 0.903}
2026-04-19 14:46:04,970 INFO Regime epoch  6/50 — tr=0.6475 va=1.0633 acc=0.485
2026-04-19 14:46:05,645 INFO Regime epoch  7/50 — tr=0.6345 va=1.0535 acc=0.494
2026-04-19 14:46:06,287 INFO Regime epoch  8/50 — tr=0.6257 va=1.0478 acc=0.504
2026-04-19 14:46:06,940 INFO Regime epoch  9/50 — tr=0.6185 va=1.0371 acc=0.518
2026-04-19 14:46:07,641 INFO Regime epoch 10/50 — tr=0.6123 va=1.0228 acc=0.528 per_class={'TRENDING': 0.575, 'RANGING': 0.003, 'CONSOLIDATING': 0.84, 'VOLATILE': 0.908}
2026-04-19 14:46:08,297 INFO Regime epoch 11/50 — tr=0.6070 va=1.0094 acc=0.539
2026-04-19 14:46:08,974 INFO Regime epoch 12/50 — tr=0.6027 va=0.9985 acc=0.545
2026-04-19 14:46:09,635 INFO Regime epoch 13/50 — tr=0.5989 va=0.9891 acc=0.551
2026-04-19 14:46:10,311 INFO Regime epoch 14/50 — tr=0.5959 va=0.9816 acc=0.555
2026-04-19 14:46:11,028 INFO Regime epoch 15/50 — tr=0.5933 va=0.9763 acc=0.559 per_class={'TRENDING': 0.633, 'RANGING': 0.005, 'CONSOLIDATING': 0.888, 'VOLATILE': 0.917}
2026-04-19 14:46:11,702 INFO Regime epoch 16/50 — tr=0.5909 va=0.9707 acc=0.562
2026-04-19 14:46:12,347 INFO Regime epoch 17/50 — tr=0.5883 va=0.9642 acc=0.564
2026-04-19 14:46:12,996 INFO Regime epoch 18/50 — tr=0.5866 va=0.9626 acc=0.565
2026-04-19 14:46:13,637 INFO Regime epoch 19/50 — tr=0.5847 va=0.9586 acc=0.572
2026-04-19 14:46:14,329 INFO Regime epoch 20/50 — tr=0.5832 va=0.9545 acc=0.573 per_class={'TRENDING': 0.653, 'RANGING': 0.002, 'CONSOLIDATING': 0.924, 'VOLATILE': 0.923}
2026-04-19 14:46:14,999 INFO Regime epoch 21/50 — tr=0.5819 va=0.9479 acc=0.575
2026-04-19 14:46:15,649 INFO Regime epoch 22/50 — tr=0.5808 va=0.9471 acc=0.576
2026-04-19 14:46:16,275 INFO Regime epoch 23/50 — tr=0.5798 va=0.9459 acc=0.577
2026-04-19 14:46:16,922 INFO Regime epoch 24/50 — tr=0.5786 va=0.9438 acc=0.580
2026-04-19 14:46:17,627 INFO Regime epoch 25/50 — tr=0.5780 va=0.9405 acc=0.580 per_class={'TRENDING': 0.662, 'RANGING': 0.0, 'CONSOLIDATING': 0.959, 'VOLATILE': 0.92}
2026-04-19 14:46:18,280 INFO Regime epoch 26/50 — tr=0.5771 va=0.9422 acc=0.582
2026-04-19 14:46:18,929 INFO Regime epoch 27/50 — tr=0.5767 va=0.9365 acc=0.583
2026-04-19 14:46:19,586 INFO Regime epoch 28/50 — tr=0.5760 va=0.9351 acc=0.583
2026-04-19 14:46:20,230 INFO Regime epoch 29/50 — tr=0.5755 va=0.9367 acc=0.580
2026-04-19 14:46:20,975 INFO Regime epoch 30/50 — tr=0.5752 va=0.9363 acc=0.582 per_class={'TRENDING': 0.665, 'RANGING': 0.0, 'CONSOLIDATING': 0.955, 'VOLATILE': 0.925}
2026-04-19 14:46:21,610 INFO Regime epoch 31/50 — tr=0.5748 va=0.9339 acc=0.583
2026-04-19 14:46:22,282 INFO Regime epoch 32/50 — tr=0.5744 va=0.9331 acc=0.583
2026-04-19 14:46:22,956 INFO Regime epoch 33/50 — tr=0.5739 va=0.9328 acc=0.583
2026-04-19 14:46:23,645 INFO Regime epoch 34/50 — tr=0.5738 va=0.9336 acc=0.586
2026-04-19 14:46:24,367 INFO Regime epoch 35/50 — tr=0.5735 va=0.9314 acc=0.583 per_class={'TRENDING': 0.668, 'RANGING': 0.0, 'CONSOLIDATING': 0.963, 'VOLATILE': 0.921}
2026-04-19 14:46:25,029 INFO Regime epoch 36/50 — tr=0.5733 va=0.9288 acc=0.586
2026-04-19 14:46:25,697 INFO Regime epoch 37/50 — tr=0.5733 va=0.9304 acc=0.584
2026-04-19 14:46:26,389 INFO Regime epoch 38/50 — tr=0.5730 va=0.9325 acc=0.581
2026-04-19 14:46:27,051 INFO Regime epoch 39/50 — tr=0.5728 va=0.9287 acc=0.584
2026-04-19 14:46:27,781 INFO Regime epoch 40/50 — tr=0.5727 va=0.9278 acc=0.587 per_class={'TRENDING': 0.679, 'RANGING': 0.0, 'CONSOLIDATING': 0.966, 'VOLATILE': 0.916}
2026-04-19 14:46:28,432 INFO Regime epoch 41/50 — tr=0.5727 va=0.9308 acc=0.582
2026-04-19 14:46:29,108 INFO Regime epoch 42/50 — tr=0.5724 va=0.9271 acc=0.585
2026-04-19 14:46:29,766 INFO Regime epoch 43/50 — tr=0.5724 va=0.9267 acc=0.586
2026-04-19 14:46:30,431 INFO Regime epoch 44/50 — tr=0.5724 va=0.9277 acc=0.586
2026-04-19 14:46:31,199 INFO Regime epoch 45/50 — tr=0.5721 va=0.9283 acc=0.584 per_class={'TRENDING': 0.671, 'RANGING': 0.0, 'CONSOLIDATING': 0.963, 'VOLATILE': 0.921}
2026-04-19 14:46:31,871 INFO Regime epoch 46/50 — tr=0.5723 va=0.9294 acc=0.583
2026-04-19 14:46:32,531 INFO Regime epoch 47/50 — tr=0.5722 va=0.9269 acc=0.584
2026-04-19 14:46:33,187 INFO Regime epoch 48/50 — tr=0.5722 va=0.9280 acc=0.586
2026-04-19 14:46:33,891 INFO Regime epoch 49/50 — tr=0.5721 va=0.9271 acc=0.585
2026-04-19 14:46:34,621 INFO Regime epoch 50/50 — tr=0.5723 va=0.9261 acc=0.585 per_class={'TRENDING': 0.673, 'RANGING': 0.0, 'CONSOLIDATING': 0.965, 'VOLATILE': 0.918}
2026-04-19 14:46:34,667 WARNING RegimeClassifier accuracy 0.58 < 0.65 threshold
2026-04-19 14:46:34,669 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-19 14:46:34,670 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-19 14:46:34,803 INFO Regime LTF complete: acc=0.585, n=401471
2026-04-19 14:46:34,807 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 14:46:35,314 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=23184 (total=74624)  short_runs_zeroed=9134
2026-04-19 14:46:35,318 INFO Regime[1H] persistence (avg bars/run) on XAUUSD 1H:
{0: 9.355291913830783, 1: 6.096303199751476, 2: 5.598885793871866, 3: 6.771351107094442}
2026-04-19 14:46:35,321 INFO Regime[1H] return separation on XAUUSD 1H:
{0: 2.149062259083323e-05, 1: -3.254493608872846e-06, 2: -3.039756158146958e-06, 3: 1.8102764458593193e-05}
2026-04-19 14:46:35,334 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-19 14:46:35,334 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 14:46:35,334 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 14:46:35,334 INFO === VectorStore: building similarity indices ===
2026-04-19 14:46:35,335 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-19 14:46:35,335 INFO Retrain complete.
2026-04-19 14:46:37,958 INFO Model regime: SUCCESS
2026-04-19 14:46:37,958 INFO --- Training gru ---
2026-04-19 14:46:37,959 INFO Running retrain --model gru
2026-04-19 14:46:38,412 INFO retrain environment: KAGGLE
2026-04-19 14:46:40,128 INFO Device: CUDA (2 GPU(s))
2026-04-19 14:46:40,139 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 14:46:40,139 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 14:46:40,140 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-19 14:46:40,275 INFO NumExpr defaulting to 4 threads.
2026-04-19 14:46:40,466 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-19 14:46:40,466 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 14:46:40,466 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 14:46:40,756 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-19 14:46:41,013 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-19 14:46:41,014 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:41,090 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:41,163 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:41,237 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:41,308 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:41,379 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:41,448 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:41,517 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:41,587 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:41,660 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:41,749 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 14:46:41,811 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-19 14:46:41,813 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260419_144641
2026-04-19 14:46:41,816 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-04-19 14:46:41,938 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:41,939 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:41,954 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:41,961 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:41,963 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-19 14:46:41,963 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 14:46:41,963 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 14:46:41,964 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:42,040 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2439, 'BIAS_DOWN': 2349, 'BIAS_NEUTRAL': 3614}  ambiguous=4099 (total=8402)  short_runs_zeroed=649
2026-04-19 14:46:42,042 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:42,266 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=10885 (total=32738)  short_runs_zeroed=4986
2026-04-19 14:46:42,295 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:42,561 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:42,701 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:42,801 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:43,007 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:43,008 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:43,023 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:43,030 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:43,031 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:43,103 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2503, 'BIAS_DOWN': 2397, 'BIAS_NEUTRAL': 3502}  ambiguous=4050 (total=8402)  short_runs_zeroed=744
2026-04-19 14:46:43,105 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:43,334 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=10483 (total=32738)  short_runs_zeroed=4347
2026-04-19 14:46:43,350 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:43,615 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:43,740 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:43,835 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:44,025 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:44,026 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:44,042 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:44,049 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:44,050 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:44,125 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2482, 'BIAS_DOWN': 2360, 'BIAS_NEUTRAL': 3560}  ambiguous=3926 (total=8402)  short_runs_zeroed=522
2026-04-19 14:46:44,127 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:44,355 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10628 (total=32740)  short_runs_zeroed=4399
2026-04-19 14:46:44,370 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:44,636 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:44,764 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:44,861 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:45,061 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:45,062 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:45,079 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:45,088 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:45,088 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:45,167 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2460, 'BIAS_DOWN': 2376, 'BIAS_NEUTRAL': 3566}  ambiguous=3995 (total=8402)  short_runs_zeroed=585
2026-04-19 14:46:45,169 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:45,397 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10080 (total=32739)  short_runs_zeroed=3955
2026-04-19 14:46:45,421 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:45,704 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:45,838 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:45,939 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:46,144 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:46,145 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:46,164 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:46,173 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:46,174 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:46,253 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2348, 'BIAS_DOWN': 2535, 'BIAS_NEUTRAL': 3520}  ambiguous=3831 (total=8403)  short_runs_zeroed=439
2026-04-19 14:46:46,255 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:46,481 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10370 (total=32740)  short_runs_zeroed=4397
2026-04-19 14:46:46,497 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:46,767 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:46,895 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:46,994 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:47,183 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:47,184 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:47,201 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:47,209 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:47,210 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:47,283 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2440, 'BIAS_DOWN': 2280, 'BIAS_NEUTRAL': 3683}  ambiguous=4047 (total=8403)  short_runs_zeroed=508
2026-04-19 14:46:47,285 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:47,511 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=9824 (total=32739)  short_runs_zeroed=3724
2026-04-19 14:46:47,526 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:47,799 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:47,928 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:48,030 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:48,195 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-19 14:46:48,196 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-19 14:46:48,210 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-19 14:46:48,217 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-19 14:46:48,218 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:48,295 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2526, 'BIAS_DOWN': 2331, 'BIAS_NEUTRAL': 3545}  ambiguous=3965 (total=8402)  short_runs_zeroed=561
2026-04-19 14:46:48,296 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:48,521 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10686 (total=32739)  short_runs_zeroed=4898
2026-04-19 14:46:48,533 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:48,796 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:48,919 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:49,015 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:49,196 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:49,197 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:49,213 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:49,221 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:49,222 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:49,296 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2250, 'BIAS_DOWN': 2599, 'BIAS_NEUTRAL': 3553}  ambiguous=3919 (total=8402)  short_runs_zeroed=547
2026-04-19 14:46:49,298 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:49,527 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=9919 (total=32740)  short_runs_zeroed=3880
2026-04-19 14:46:49,544 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:49,819 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:49,952 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:50,055 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:50,248 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:50,249 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:50,265 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:50,273 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:50,274 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:50,345 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2214, 'BIAS_DOWN': 2634, 'BIAS_NEUTRAL': 3554}  ambiguous=3934 (total=8402)  short_runs_zeroed=549
2026-04-19 14:46:50,346 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:50,572 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=10596 (total=32741)  short_runs_zeroed=3896
2026-04-19 14:46:50,586 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:50,852 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:51,004 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:51,104 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:51,293 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:51,294 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:51,310 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:51,317 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 14:46:51,318 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:51,394 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2278, 'BIAS_DOWN': 2515, 'BIAS_NEUTRAL': 3610}  ambiguous=3986 (total=8403)  short_runs_zeroed=516
2026-04-19 14:46:51,396 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:51,629 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10093 (total=32743)  short_runs_zeroed=4275
2026-04-19 14:46:51,644 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:51,912 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:52,042 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:52,144 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 14:46:52,437 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-19 14:46:52,438 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-19 14:46:52,456 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-19 14:46:52,467 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-19 14:46:52,468 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 14:46:52,617 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 6629, 'BIAS_DOWN': 5241, 'BIAS_NEUTRAL': 7947}  ambiguous=8645 (total=19817)  short_runs_zeroed=980
2026-04-19 14:46:52,621 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 14:46:53,150 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=23184 (total=74624)  short_runs_zeroed=9134
2026-04-19 14:46:53,198 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-19 14:46:53,735 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-19 14:46:53,932 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 14:46:54,064 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 14:46:54,180 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-19 14:46:54,180 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-19 14:46:54,180 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-19 14:51:33,910 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-19 14:51:33,910 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-19 14:51:35,252 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-19 14:51:59,717 INFO train_multi TF=ALL epoch 1/50 train=0.5922 val=0.6121
2026-04-19 14:51:59,722 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 14:51:59,722 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 14:51:59,722 INFO train_multi TF=ALL: new best val=0.6121 — saved
2026-04-19 14:52:17,608 INFO train_multi TF=ALL epoch 2/50 train=0.5915 val=0.6121
2026-04-19 14:52:17,613 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 14:52:17,613 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 14:52:17,613 INFO train_multi TF=ALL: new best val=0.6121 — saved
2026-04-19 14:52:35,777 INFO train_multi TF=ALL epoch 3/50 train=0.5916 val=0.6124
2026-04-19 14:52:53,828 INFO train_multi TF=ALL epoch 4/50 train=0.5913 val=0.6122
2026-04-19 14:53:11,798 INFO train_multi TF=ALL epoch 5/50 train=0.5911 val=0.6129
2026-04-19 14:53:29,544 INFO train_multi TF=ALL epoch 6/50 train=0.5907 val=0.6127
2026-04-19 14:53:47,584 INFO train_multi TF=ALL epoch 7/50 train=0.5909 val=0.6122
2026-04-19 14:53:47,584 INFO train_multi TF=ALL early stop at epoch 7
2026-04-19 14:53:47,724 INFO === VectorStore: building similarity indices ===
2026-04-19 14:53:47,725 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-19 14:53:47,725 INFO Retrain complete.
2026-04-19 14:53:49,690 INFO Model gru: SUCCESS
2026-04-19 14:53:49,690 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 14:53:49,691 WARNING   [MISSING] regime_classifier → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_classifier.pkl
2026-04-19 14:53:49,691 INFO   [OK] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-19 14:53:49,691 WARNING   [MISSING] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-19 14:53:49,691 WARNING Missing weights: ['regime_classifier', 'rl_ppo'] — run retrain_incremental.py for each
2026-04-19 14:53:49,692 INFO Saved 18 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  Cleared journal for fresh Round 1 run
  START Round 1 - Backtest (val)
2026-04-19 14:53:50,231 INFO === STEP 6: BACKTEST + REINFORCED TRAINING (3 rounds, window=round1) ===
2026-04-19 14:53:50,232 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-04-19 14:53:50,232 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-19 14:53:50,232 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)

======================================================================
  BACKTEST + REINFORCED TRAINING COMPLETE  (0 rounds)
======================================================================
  Round     Trades       WR      PF   Sharpe
  ------------------------------------------

  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 0 entries