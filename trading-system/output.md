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
2026-04-19 07:52:28,271 INFO Loading feature-engineered data...
2026-04-19 07:52:28,871 INFO Loaded 221743 rows, 202 features
2026-04-19 07:52:28,872 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-19 07:52:28,874 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-19 07:52:28,874 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-19 07:52:28,874 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-19 07:52:28,874 INFO No leakage confirmed: train < val < test timestamps

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
2026-04-19 07:52:31,209 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-19 07:52:31,209 INFO --- Training regime ---
2026-04-19 07:52:31,209 INFO Running retrain --model regime
2026-04-19 07:52:31,389 INFO retrain environment: KAGGLE
2026-04-19 07:52:33,032 INFO Device: CUDA (2 GPU(s))
2026-04-19 07:52:33,043 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 07:52:33,043 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 07:52:33,044 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-19 07:52:33,186 INFO NumExpr defaulting to 4 threads.
2026-04-19 07:52:33,388 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-19 07:52:33,388 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 07:52:33,389 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 07:52:33,602 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-19 07:52:33,604 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:33,689 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:33,765 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:33,841 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:33,915 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:33,991 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,061 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,131 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,202 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,274 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,363 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:52:34,425 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-19 07:52:34,441 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,443 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,457 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,459 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,474 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,476 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,492 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,494 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,510 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,514 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,530 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,534 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,554 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,556 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,571 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,574 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,590 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,593 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,608 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,612 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:52:34,630 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:52:34,637 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:52:35,752 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-19 07:52:57,270 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-19 07:52:57,275 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias)
2026-04-19 07:52:57,275 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-19 07:53:06,943 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-19 07:53:06,947 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias)
2026-04-19 07:53:06,947 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-19 07:53:14,311 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-19 07:53:14,315 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias)
2026-04-19 07:53:14,315 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-19 07:54:23,920 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-19 07:54:23,924 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour)
2026-04-19 07:54:23,924 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-19 07:54:55,525 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-19 07:54:55,533 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour)
2026-04-19 07:54:55,533 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-19 07:55:17,630 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-19 07:55:17,633 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour)
2026-04-19 07:55:17,730 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-19 07:55:17,732 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:17,733 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:17,734 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:17,735 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:17,736 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:17,737 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:17,738 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:17,739 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:17,740 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:17,741 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:17,743 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:55:17,874 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:17,917 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:17,917 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:17,918 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:17,926 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:17,927 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:18,309 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2439, 'BIAS_DOWN': 2349, 'BIAS_NEUTRAL': 3614}  ambiguous=4099 (total=8402)  short_runs_zeroed=649
2026-04-19 07:55:18,310 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-19 07:55:18,486 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:18,519 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:18,520 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:18,520 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:18,529 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:18,530 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:18,883 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2503, 'BIAS_DOWN': 2397, 'BIAS_NEUTRAL': 3502}  ambiguous=4050 (total=8402)  short_runs_zeroed=744
2026-04-19 07:55:18,884 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-19 07:55:19,081 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:19,117 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:19,118 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:19,118 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:19,128 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:19,129 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:19,482 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2482, 'BIAS_DOWN': 2360, 'BIAS_NEUTRAL': 3560}  ambiguous=3926 (total=8402)  short_runs_zeroed=522
2026-04-19 07:55:19,484 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-19 07:55:19,644 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:19,682 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:19,683 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:19,683 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:19,693 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:19,694 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:20,042 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2460, 'BIAS_DOWN': 2376, 'BIAS_NEUTRAL': 3566}  ambiguous=3995 (total=8402)  short_runs_zeroed=585
2026-04-19 07:55:20,043 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-19 07:55:20,213 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:20,250 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:20,251 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:20,252 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:20,261 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:20,262 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:20,617 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2348, 'BIAS_DOWN': 2535, 'BIAS_NEUTRAL': 3520}  ambiguous=3831 (total=8403)  short_runs_zeroed=439
2026-04-19 07:55:20,618 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-19 07:55:20,787 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:20,822 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:20,823 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:20,823 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:20,833 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:20,834 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:21,192 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2440, 'BIAS_DOWN': 2280, 'BIAS_NEUTRAL': 3683}  ambiguous=4047 (total=8403)  short_runs_zeroed=508
2026-04-19 07:55:21,193 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-19 07:55:21,340 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:55:21,369 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:55:21,369 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:55:21,370 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:55:21,379 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:55:21,380 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:21,734 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2526, 'BIAS_DOWN': 2331, 'BIAS_NEUTRAL': 3545}  ambiguous=3965 (total=8402)  short_runs_zeroed=561
2026-04-19 07:55:21,735 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-19 07:55:21,900 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:21,934 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:21,935 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:21,935 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:21,945 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:21,946 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:22,289 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2250, 'BIAS_DOWN': 2599, 'BIAS_NEUTRAL': 3553}  ambiguous=3919 (total=8402)  short_runs_zeroed=547
2026-04-19 07:55:22,290 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-19 07:55:22,456 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:22,491 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:22,492 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:22,493 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:22,502 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:22,503 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:22,856 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2214, 'BIAS_DOWN': 2634, 'BIAS_NEUTRAL': 3554}  ambiguous=3934 (total=8402)  short_runs_zeroed=549
2026-04-19 07:55:22,857 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-19 07:55:23,025 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:23,063 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:23,063 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:23,064 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:23,073 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:23,074 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:23,425 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2278, 'BIAS_DOWN': 2515, 'BIAS_NEUTRAL': 3610}  ambiguous=3986 (total=8403)  short_runs_zeroed=516
2026-04-19 07:55:23,426 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-19 07:55:23,695 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:55:23,751 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:55:23,752 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:55:23,753 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:55:23,765 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:55:23,766 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:55:24,506 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 6629, 'BIAS_DOWN': 5241, 'BIAS_NEUTRAL': 7947}  ambiguous=8645 (total=19817)  short_runs_zeroed=980
2026-04-19 07:55:24,508 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-19 07:55:24,681 INFO RegimeClassifier[mode=htf_bias]: 103290 samples, classes={'BIAS_UP': 30569, 'BIAS_DOWN': 29617, 'BIAS_NEUTRAL': 43104}, device=cuda
2026-04-19 07:55:24,681 INFO RegimeClassifier: sample weights — mean=0.413  ambiguous(<0.4)=47.0%
2026-04-19 07:55:24,931 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-04-19 07:55:24,932 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-19 07:55:29,274 INFO Regime epoch  1/50 — tr=0.4880 va=1.1133 acc=0.403 per_class={'BIAS_UP': 0.214, 'BIAS_DOWN': 0.123, 'BIAS_NEUTRAL': 0.746}
2026-04-19 07:55:29,446 INFO Regime epoch  2/50 — tr=0.4782 va=1.0884 acc=0.454
2026-04-19 07:55:29,625 INFO Regime epoch  3/50 — tr=0.4594 va=1.0295 acc=0.537
2026-04-19 07:55:29,800 INFO Regime epoch  4/50 — tr=0.4285 va=0.9328 acc=0.656
2026-04-19 07:55:29,989 INFO Regime epoch  5/50 — tr=0.3960 va=0.8411 acc=0.698 per_class={'BIAS_UP': 0.964, 'BIAS_DOWN': 0.928, 'BIAS_NEUTRAL': 0.322}
2026-04-19 07:55:30,155 INFO Regime epoch  6/50 — tr=0.3646 va=0.7648 acc=0.701
2026-04-19 07:55:30,323 INFO Regime epoch  7/50 — tr=0.3413 va=0.7067 acc=0.694
2026-04-19 07:55:30,495 INFO Regime epoch  8/50 — tr=0.3285 va=0.6761 acc=0.689
2026-04-19 07:55:30,665 INFO Regime epoch  9/50 — tr=0.3215 va=0.6610 acc=0.689
2026-04-19 07:55:30,859 INFO Regime epoch 10/50 — tr=0.3167 va=0.6588 acc=0.686 per_class={'BIAS_UP': 0.997, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.218}
2026-04-19 07:55:31,035 INFO Regime epoch 11/50 — tr=0.3140 va=0.6571 acc=0.684
2026-04-19 07:55:31,201 INFO Regime epoch 12/50 — tr=0.3114 va=0.6589 acc=0.681
2026-04-19 07:55:31,371 INFO Regime epoch 13/50 — tr=0.3097 va=0.6566 acc=0.680
2026-04-19 07:55:31,552 INFO Regime epoch 14/50 — tr=0.3079 va=0.6560 acc=0.682
2026-04-19 07:55:31,738 INFO Regime epoch 15/50 — tr=0.3060 va=0.6573 acc=0.679 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.199}
2026-04-19 07:55:31,903 INFO Regime epoch 16/50 — tr=0.3046 va=0.6566 acc=0.678
2026-04-19 07:55:32,069 INFO Regime epoch 17/50 — tr=0.3031 va=0.6549 acc=0.677
2026-04-19 07:55:32,235 INFO Regime epoch 18/50 — tr=0.3023 va=0.6532 acc=0.676
2026-04-19 07:55:32,404 INFO Regime epoch 19/50 — tr=0.3017 va=0.6526 acc=0.679
2026-04-19 07:55:32,586 INFO Regime epoch 20/50 — tr=0.3006 va=0.6549 acc=0.680 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.2}
2026-04-19 07:55:32,751 INFO Regime epoch 21/50 — tr=0.3001 va=0.6540 acc=0.678
2026-04-19 07:55:32,924 INFO Regime epoch 22/50 — tr=0.2992 va=0.6537 acc=0.680
2026-04-19 07:55:33,097 INFO Regime epoch 23/50 — tr=0.2989 va=0.6535 acc=0.680
2026-04-19 07:55:33,263 INFO Regime epoch 24/50 — tr=0.2985 va=0.6531 acc=0.678
2026-04-19 07:55:33,455 INFO Regime epoch 25/50 — tr=0.2980 va=0.6531 acc=0.679 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.198}
2026-04-19 07:55:33,632 INFO Regime epoch 26/50 — tr=0.2974 va=0.6511 acc=0.680
2026-04-19 07:55:33,805 INFO Regime epoch 27/50 — tr=0.2970 va=0.6512 acc=0.678
2026-04-19 07:55:33,979 INFO Regime epoch 28/50 — tr=0.2970 va=0.6510 acc=0.679
2026-04-19 07:55:34,152 INFO Regime epoch 29/50 — tr=0.2964 va=0.6510 acc=0.678
2026-04-19 07:55:34,341 INFO Regime epoch 30/50 — tr=0.2963 va=0.6497 acc=0.678 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.195}
2026-04-19 07:55:34,522 INFO Regime epoch 31/50 — tr=0.2961 va=0.6498 acc=0.678
2026-04-19 07:55:34,702 INFO Regime epoch 32/50 — tr=0.2960 va=0.6499 acc=0.677
2026-04-19 07:55:34,872 INFO Regime epoch 33/50 — tr=0.2959 va=0.6490 acc=0.679
2026-04-19 07:55:35,044 INFO Regime epoch 34/50 — tr=0.2955 va=0.6496 acc=0.680
2026-04-19 07:55:35,227 INFO Regime epoch 35/50 — tr=0.2954 va=0.6495 acc=0.678 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.196}
2026-04-19 07:55:35,392 INFO Regime epoch 36/50 — tr=0.2951 va=0.6499 acc=0.678
2026-04-19 07:55:35,559 INFO Regime epoch 37/50 — tr=0.2951 va=0.6497 acc=0.679
2026-04-19 07:55:35,729 INFO Regime epoch 38/50 — tr=0.2949 va=0.6499 acc=0.679
2026-04-19 07:55:35,901 INFO Regime epoch 39/50 — tr=0.2948 va=0.6498 acc=0.680
2026-04-19 07:55:36,086 INFO Regime epoch 40/50 — tr=0.2949 va=0.6486 acc=0.679 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.196}
2026-04-19 07:55:36,262 INFO Regime epoch 41/50 — tr=0.2951 va=0.6495 acc=0.679
2026-04-19 07:55:36,434 INFO Regime epoch 42/50 — tr=0.2948 va=0.6490 acc=0.678
2026-04-19 07:55:36,602 INFO Regime epoch 43/50 — tr=0.2948 va=0.6498 acc=0.677
2026-04-19 07:55:36,779 INFO Regime epoch 44/50 — tr=0.2946 va=0.6498 acc=0.680
2026-04-19 07:55:36,963 INFO Regime epoch 45/50 — tr=0.2947 va=0.6496 acc=0.678 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.195}
2026-04-19 07:55:37,142 INFO Regime epoch 46/50 — tr=0.2948 va=0.6495 acc=0.679
2026-04-19 07:55:37,324 INFO Regime epoch 47/50 — tr=0.2947 va=0.6490 acc=0.679
2026-04-19 07:55:37,505 INFO Regime epoch 48/50 — tr=0.2947 va=0.6485 acc=0.677
2026-04-19 07:55:37,681 INFO Regime epoch 49/50 — tr=0.2947 va=0.6511 acc=0.680
2026-04-19 07:55:37,872 INFO Regime epoch 50/50 — tr=0.2946 va=0.6505 acc=0.681 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.202}
2026-04-19 07:55:37,889 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-19 07:55:37,889 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-19 07:55:38,014 INFO Regime HTF complete: acc=0.677, n=103290
2026-04-19 07:55:38,016 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:55:38,172 INFO Rule labels LTF_BEHAVIOUR [4H]: {'TRENDING': 8970, 'RANGING': 4592, 'CONSOLIDATING': 2247, 'VOLATILE': 4008}  ambiguous=4132 (total=19817)  short_runs_zeroed=570
2026-04-19 07:55:38,178 INFO Regime[4H] persistence (avg bars/run) on XAUUSD 4H:
{0: 12.121621621621621, 1: 6.4858757062146895, 2: 5.575682382133995, 3: 10.329896907216495}
2026-04-19 07:55:38,180 INFO Regime[4H] return separation on XAUUSD 4H:
{0: 5.9502411776840314e-05, 1: 5.664222849552333e-05, 2: -6.138442068010657e-05, 3: 4.239843820627722e-05}
2026-04-19 07:55:38,180 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-19 07:55:38,182 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:38,183 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:38,185 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:38,187 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:38,188 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:38,190 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:38,191 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:38,193 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:38,194 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:38,195 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:38,198 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:55:38,209 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:38,211 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:38,212 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:38,212 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:38,212 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:38,215 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:38,821 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=10885 (total=32738)  short_runs_zeroed=4986
2026-04-19 07:55:38,823 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-19 07:55:38,958 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:38,961 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:38,962 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:38,962 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:38,962 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:38,965 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:39,551 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=10483 (total=32738)  short_runs_zeroed=4347
2026-04-19 07:55:39,554 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-19 07:55:39,686 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:39,688 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:39,689 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:39,689 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:39,689 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:39,692 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:40,243 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10628 (total=32740)  short_runs_zeroed=4399
2026-04-19 07:55:40,246 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-19 07:55:40,383 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:40,386 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:40,386 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:40,387 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:40,387 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:40,389 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:40,952 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10080 (total=32739)  short_runs_zeroed=3955
2026-04-19 07:55:40,955 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-19 07:55:41,087 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:41,089 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:41,090 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:41,090 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:41,090 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:41,092 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:41,646 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10370 (total=32740)  short_runs_zeroed=4397
2026-04-19 07:55:41,649 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-19 07:55:41,777 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:41,779 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:41,780 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:41,780 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:41,781 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:41,782 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:42,331 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=9824 (total=32739)  short_runs_zeroed=3724
2026-04-19 07:55:42,334 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-19 07:55:42,460 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:55:42,461 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:55:42,462 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:55:42,462 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:55:42,463 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:55:42,464 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:43,002 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10686 (total=32739)  short_runs_zeroed=4898
2026-04-19 07:55:43,005 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-19 07:55:43,142 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:43,144 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:43,145 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:43,146 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:43,146 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:43,148 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:43,697 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=9919 (total=32740)  short_runs_zeroed=3880
2026-04-19 07:55:43,700 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-19 07:55:43,832 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:43,834 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:43,835 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:43,836 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:43,836 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:43,838 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:44,378 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=10596 (total=32741)  short_runs_zeroed=3896
2026-04-19 07:55:44,381 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-19 07:55:44,515 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:44,517 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:44,518 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:44,518 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:44,519 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:55:44,521 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:55:45,070 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10093 (total=32743)  short_runs_zeroed=4275
2026-04-19 07:55:45,073 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-19 07:55:45,214 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:55:45,218 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:55:45,219 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:55:45,220 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:55:45,220 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:55:45,223 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:55:46,396 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=23184 (total=74624)  short_runs_zeroed=9134
2026-04-19 07:55:46,401 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-19 07:55:46,741 INFO RegimeClassifier[mode=ltf_behaviour]: 401471 samples, classes={'TRENDING': 160094, 'RANGING': 105286, 'CONSOLIDATING': 53524, 'VOLATILE': 82567}, device=cuda
2026-04-19 07:55:46,742 INFO RegimeClassifier: sample weights — mean=0.532  ambiguous(<0.4)=31.6%
2026-04-19 07:55:46,744 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-04-19 07:55:46,744 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-19 07:55:47,461 INFO Regime epoch  1/50 — tr=0.7519 va=1.3420 acc=0.335 per_class={'TRENDING': 0.397, 'RANGING': 0.352, 'CONSOLIDATING': 0.175, 'VOLATILE': 0.297}
2026-04-19 07:55:48,117 INFO Regime epoch  2/50 — tr=0.7118 va=1.2408 acc=0.441
2026-04-19 07:55:48,782 INFO Regime epoch  3/50 — tr=0.6579 va=1.1541 acc=0.450
2026-04-19 07:55:49,467 INFO Regime epoch  4/50 — tr=0.6093 va=1.0916 acc=0.452
2026-04-19 07:55:50,142 INFO Regime epoch  5/50 — tr=0.5794 va=1.0645 acc=0.454 per_class={'TRENDING': 0.397, 'RANGING': 0.006, 'CONSOLIDATING': 0.825, 'VOLATILE': 0.907}
2026-04-19 07:55:50,781 INFO Regime epoch  6/50 — tr=0.5613 va=1.0512 acc=0.464
2026-04-19 07:55:51,412 INFO Regime epoch  7/50 — tr=0.5497 va=1.0454 acc=0.477
2026-04-19 07:55:52,041 INFO Regime epoch  8/50 — tr=0.5406 va=1.0377 acc=0.486
2026-04-19 07:55:52,667 INFO Regime epoch  9/50 — tr=0.5339 va=1.0286 acc=0.498
2026-04-19 07:55:53,369 INFO Regime epoch 10/50 — tr=0.5282 va=1.0170 acc=0.508 per_class={'TRENDING': 0.525, 'RANGING': 0.006, 'CONSOLIDATING': 0.837, 'VOLATILE': 0.911}
2026-04-19 07:55:54,019 INFO Regime epoch 11/50 — tr=0.5229 va=1.0065 acc=0.519
2026-04-19 07:55:54,681 INFO Regime epoch 12/50 — tr=0.5190 va=0.9929 acc=0.530
2026-04-19 07:55:55,315 INFO Regime epoch 13/50 — tr=0.5153 va=0.9855 acc=0.537
2026-04-19 07:55:55,977 INFO Regime epoch 14/50 — tr=0.5117 va=0.9770 acc=0.544
2026-04-19 07:55:56,666 INFO Regime epoch 15/50 — tr=0.5089 va=0.9715 acc=0.550 per_class={'TRENDING': 0.614, 'RANGING': 0.005, 'CONSOLIDATING': 0.883, 'VOLATILE': 0.914}
2026-04-19 07:55:57,310 INFO Regime epoch 16/50 — tr=0.5067 va=0.9659 acc=0.555
2026-04-19 07:55:57,936 INFO Regime epoch 17/50 — tr=0.5037 va=0.9593 acc=0.558
2026-04-19 07:55:58,552 INFO Regime epoch 18/50 — tr=0.5018 va=0.9529 acc=0.564
2026-04-19 07:55:59,209 INFO Regime epoch 19/50 — tr=0.4998 va=0.9512 acc=0.566
2026-04-19 07:55:59,895 INFO Regime epoch 20/50 — tr=0.4982 va=0.9444 acc=0.569 per_class={'TRENDING': 0.641, 'RANGING': 0.003, 'CONSOLIDATING': 0.933, 'VOLATILE': 0.918}
2026-04-19 07:56:00,532 INFO Regime epoch 21/50 — tr=0.4966 va=0.9446 acc=0.573
2026-04-19 07:56:01,178 INFO Regime epoch 22/50 — tr=0.4954 va=0.9388 acc=0.575
2026-04-19 07:56:01,843 INFO Regime epoch 23/50 — tr=0.4942 va=0.9384 acc=0.573
2026-04-19 07:56:02,478 INFO Regime epoch 24/50 — tr=0.4931 va=0.9320 acc=0.574
2026-04-19 07:56:03,162 INFO Regime epoch 25/50 — tr=0.4922 va=0.9319 acc=0.575 per_class={'TRENDING': 0.651, 'RANGING': 0.001, 'CONSOLIDATING': 0.951, 'VOLATILE': 0.922}
2026-04-19 07:56:03,796 INFO Regime epoch 26/50 — tr=0.4914 va=0.9293 acc=0.576
2026-04-19 07:56:04,417 INFO Regime epoch 27/50 — tr=0.4908 va=0.9288 acc=0.577
2026-04-19 07:56:05,039 INFO Regime epoch 28/50 — tr=0.4902 va=0.9250 acc=0.578
2026-04-19 07:56:05,664 INFO Regime epoch 29/50 — tr=0.4896 va=0.9247 acc=0.579
2026-04-19 07:56:06,367 INFO Regime epoch 30/50 — tr=0.4891 va=0.9254 acc=0.577 per_class={'TRENDING': 0.652, 'RANGING': 0.0, 'CONSOLIDATING': 0.957, 'VOLATILE': 0.926}
2026-04-19 07:56:07,007 INFO Regime epoch 31/50 — tr=0.4888 va=0.9226 acc=0.578
2026-04-19 07:56:07,647 INFO Regime epoch 32/50 — tr=0.4883 va=0.9224 acc=0.578
2026-04-19 07:56:08,285 INFO Regime epoch 33/50 — tr=0.4876 va=0.9186 acc=0.583
2026-04-19 07:56:08,937 INFO Regime epoch 34/50 — tr=0.4875 va=0.9166 acc=0.582
2026-04-19 07:56:09,652 INFO Regime epoch 35/50 — tr=0.4874 va=0.9207 acc=0.580 per_class={'TRENDING': 0.659, 'RANGING': 0.0, 'CONSOLIDATING': 0.962, 'VOLATILE': 0.922}
2026-04-19 07:56:10,283 INFO Regime epoch 36/50 — tr=0.4873 va=0.9174 acc=0.583
2026-04-19 07:56:10,931 INFO Regime epoch 37/50 — tr=0.4870 va=0.9215 acc=0.580
2026-04-19 07:56:11,581 INFO Regime epoch 38/50 — tr=0.4870 va=0.9197 acc=0.580
2026-04-19 07:56:12,205 INFO Regime epoch 39/50 — tr=0.4865 va=0.9182 acc=0.580
2026-04-19 07:56:12,886 INFO Regime epoch 40/50 — tr=0.4865 va=0.9177 acc=0.581 per_class={'TRENDING': 0.664, 'RANGING': 0.0, 'CONSOLIDATING': 0.963, 'VOLATILE': 0.921}
2026-04-19 07:56:13,566 INFO Regime epoch 41/50 — tr=0.4864 va=0.9178 acc=0.583
2026-04-19 07:56:14,199 INFO Regime epoch 42/50 — tr=0.4864 va=0.9202 acc=0.580
2026-04-19 07:56:14,845 INFO Regime epoch 43/50 — tr=0.4862 va=0.9220 acc=0.579
2026-04-19 07:56:15,472 INFO Regime epoch 44/50 — tr=0.4863 va=0.9176 acc=0.583
2026-04-19 07:56:15,472 INFO Regime early stop at epoch 44 (no_improve=10)
2026-04-19 07:56:15,516 WARNING RegimeClassifier accuracy 0.58 < 0.65 threshold
2026-04-19 07:56:15,519 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-19 07:56:15,519 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-19 07:56:15,646 INFO Regime LTF complete: acc=0.582, n=401471
2026-04-19 07:56:15,649 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:56:16,167 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=23184 (total=74624)  short_runs_zeroed=9134
2026-04-19 07:56:16,171 INFO Regime[1H] persistence (avg bars/run) on XAUUSD 1H:
{0: 9.355291913830783, 1: 6.096303199751476, 2: 5.598885793871866, 3: 6.771351107094442}
2026-04-19 07:56:16,175 INFO Regime[1H] return separation on XAUUSD 1H:
{0: 2.149062259083323e-05, 1: -3.254493608872846e-06, 2: -3.039756158146958e-06, 3: 1.8102764458593193e-05}
2026-04-19 07:56:16,189 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-19 07:56:16,190 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 07:56:16,190 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 07:56:16,190 INFO === VectorStore: building similarity indices ===
2026-04-19 07:56:16,190 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-19 07:56:16,191 INFO Retrain complete.
2026-04-19 07:56:18,338 INFO Model regime: SUCCESS
2026-04-19 07:56:18,338 INFO --- Training gru ---
2026-04-19 07:56:18,338 INFO Running retrain --model gru
2026-04-19 07:56:18,878 INFO retrain environment: KAGGLE
2026-04-19 07:56:20,583 INFO Device: CUDA (2 GPU(s))
2026-04-19 07:56:20,595 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 07:56:20,595 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 07:56:20,596 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-19 07:56:20,733 INFO NumExpr defaulting to 4 threads.
2026-04-19 07:56:20,928 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-19 07:56:20,928 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 07:56:20,928 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 07:56:21,136 WARNING GRULSTMPredictor: shape mismatch loading weights — deleting stale /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt and retraining from scratch. Detail: Error(s) in loading state_dict for _MultiHeadGRULSTM:
	Missing key(s) in state_dict: "gru.weight_ih_l1", "gru.weight_hh_l1", "gru.bias_ih_l1", "gru.bias_hh_l1", "lstm.weight_ih_l1", "lstm.weight_hh_l1", "lstm.bias_ih_l1", "lstm.bias_hh_l1". 
2026-04-19 07:56:21,137 INFO Deleted stale weights (shape mismatch: Error(s) in loading state_dict for _MultiHeadGRULSTM:
	Missing key(s) in state_dict: "gru.weight_ih_l1", "gru.weight_hh_l1", "gru.bias_ih_l1", "gru.bias_hh_l1", "lstm.weight_ih_l1", "lstm.weight_hh_l1", "lstm.bias_ih_l1", "lstm.bias_hh_l1". ): /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 07:56:21,366 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-19 07:56:21,367 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:21,444 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:21,520 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:21,596 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:21,671 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:21,741 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:21,810 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:21,884 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:21,958 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:22,030 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:22,121 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:56:22,185 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-19 07:56:22,186 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260419_075622
2026-04-19 07:56:22,303 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:22,303 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:22,318 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:22,325 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:22,326 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-19 07:56:22,326 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 07:56:22,327 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 07:56:22,327 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:22,400 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2439, 'BIAS_DOWN': 2349, 'BIAS_NEUTRAL': 3614}  ambiguous=4099 (total=8402)  short_runs_zeroed=649
2026-04-19 07:56:22,402 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:22,631 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=10885 (total=32738)  short_runs_zeroed=4986
2026-04-19 07:56:22,664 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:22,933 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:23,058 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:23,152 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:23,345 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:23,346 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:23,362 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:23,370 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:23,371 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:23,446 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2503, 'BIAS_DOWN': 2397, 'BIAS_NEUTRAL': 3502}  ambiguous=4050 (total=8402)  short_runs_zeroed=744
2026-04-19 07:56:23,448 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:23,700 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=10483 (total=32738)  short_runs_zeroed=4347
2026-04-19 07:56:23,715 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:23,967 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:24,090 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:24,182 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:24,361 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:24,362 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:24,378 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:24,385 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:24,386 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:24,459 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2482, 'BIAS_DOWN': 2360, 'BIAS_NEUTRAL': 3560}  ambiguous=3926 (total=8402)  short_runs_zeroed=522
2026-04-19 07:56:24,461 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:24,698 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10628 (total=32740)  short_runs_zeroed=4399
2026-04-19 07:56:24,712 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:24,970 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:25,089 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:25,180 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:25,364 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:25,365 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:25,380 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:25,388 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:25,388 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:25,461 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2460, 'BIAS_DOWN': 2376, 'BIAS_NEUTRAL': 3566}  ambiguous=3995 (total=8402)  short_runs_zeroed=585
2026-04-19 07:56:25,463 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:25,696 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10080 (total=32739)  short_runs_zeroed=3955
2026-04-19 07:56:25,717 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:25,984 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:26,111 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:26,210 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:26,389 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:26,390 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:26,407 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:26,415 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:26,416 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:26,489 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2348, 'BIAS_DOWN': 2535, 'BIAS_NEUTRAL': 3520}  ambiguous=3831 (total=8403)  short_runs_zeroed=439
2026-04-19 07:56:26,491 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:26,717 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10370 (total=32740)  short_runs_zeroed=4397
2026-04-19 07:56:26,732 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:26,983 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:27,105 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:27,198 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:27,373 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:27,374 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:27,390 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:27,397 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:27,398 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:27,473 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2440, 'BIAS_DOWN': 2280, 'BIAS_NEUTRAL': 3683}  ambiguous=4047 (total=8403)  short_runs_zeroed=508
2026-04-19 07:56:27,475 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:27,701 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=9824 (total=32739)  short_runs_zeroed=3724
2026-04-19 07:56:27,716 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:27,973 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:28,096 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:28,186 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:28,346 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:56:28,347 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:56:28,365 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:56:28,372 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-19 07:56:28,373 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:28,449 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2526, 'BIAS_DOWN': 2331, 'BIAS_NEUTRAL': 3545}  ambiguous=3965 (total=8402)  short_runs_zeroed=561
2026-04-19 07:56:28,450 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:28,678 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10686 (total=32739)  short_runs_zeroed=4898
2026-04-19 07:56:28,691 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:28,957 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:29,091 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:29,180 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:29,353 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:29,353 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:29,368 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:29,376 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:29,377 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:29,452 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2250, 'BIAS_DOWN': 2599, 'BIAS_NEUTRAL': 3553}  ambiguous=3919 (total=8402)  short_runs_zeroed=547
2026-04-19 07:56:29,454 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:29,692 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=9919 (total=32740)  short_runs_zeroed=3880
2026-04-19 07:56:29,708 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:29,971 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:30,099 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:30,193 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:30,370 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:30,371 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:30,386 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:30,394 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:30,394 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:30,467 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2214, 'BIAS_DOWN': 2634, 'BIAS_NEUTRAL': 3554}  ambiguous=3934 (total=8402)  short_runs_zeroed=549
2026-04-19 07:56:30,469 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:30,691 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=10596 (total=32741)  short_runs_zeroed=3896
2026-04-19 07:56:30,706 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:30,963 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:31,084 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:31,174 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:31,348 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:31,349 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:31,366 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:31,373 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 07:56:31,374 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:31,448 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2278, 'BIAS_DOWN': 2515, 'BIAS_NEUTRAL': 3610}  ambiguous=3986 (total=8403)  short_runs_zeroed=516
2026-04-19 07:56:31,450 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:31,682 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10093 (total=32743)  short_runs_zeroed=4275
2026-04-19 07:56:31,698 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:31,960 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:32,086 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:32,186 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 07:56:32,462 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:56:32,463 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:56:32,481 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:56:32,490 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-19 07:56:32,492 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:56:32,643 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 6629, 'BIAS_DOWN': 5241, 'BIAS_NEUTRAL': 7947}  ambiguous=8645 (total=19817)  short_runs_zeroed=980
2026-04-19 07:56:32,646 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:56:33,146 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=23184 (total=74624)  short_runs_zeroed=9134
2026-04-19 07:56:33,190 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:56:33,717 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:56:33,913 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:56:34,042 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 07:56:34,155 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-19 07:56:34,211 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-04-19 07:56:34,211 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-04-19 07:56:34,211 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-19 07:56:34,212 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-19 08:00:58,760 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-19 08:00:58,761 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-19 08:01:00,109 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-19 08:01:24,509 INFO train_multi TF=ALL epoch 1/50 train=0.8576 val=0.8289
2026-04-19 08:01:24,513 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 08:01:24,513 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 08:01:24,513 INFO train_multi TF=ALL: new best val=0.8289 — saved
2026-04-19 08:01:42,500 INFO train_multi TF=ALL epoch 2/50 train=0.7303 val=0.6879
2026-04-19 08:01:42,505 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 08:01:42,505 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 08:01:42,505 INFO train_multi TF=ALL: new best val=0.6879 — saved
2026-04-19 08:02:00,531 INFO train_multi TF=ALL epoch 3/50 train=0.6893 val=0.6880
2026-04-19 08:02:18,575 INFO train_multi TF=ALL epoch 4/50 train=0.6888 val=0.6884
2026-04-19 08:02:36,342 INFO train_multi TF=ALL epoch 5/50 train=0.6887 val=0.6883
2026-04-19 08:02:54,173 INFO train_multi TF=ALL epoch 6/50 train=0.6885 val=0.6881
2026-04-19 08:03:12,072 INFO train_multi TF=ALL epoch 7/50 train=0.6879 val=0.6873
2026-04-19 08:03:12,076 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 08:03:12,076 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 08:03:12,076 INFO train_multi TF=ALL: new best val=0.6873 — saved
2026-04-19 08:03:30,052 INFO train_multi TF=ALL epoch 8/50 train=0.6862 val=0.6846
2026-04-19 08:03:30,057 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 08:03:30,057 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 08:03:30,057 INFO train_multi TF=ALL: new best val=0.6846 — saved
2026-04-19 08:03:47,983 INFO train_multi TF=ALL epoch 9/50 train=0.6812 val=0.6766
2026-04-19 08:03:47,988 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 08:03:47,988 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 08:03:47,988 INFO train_multi TF=ALL: new best val=0.6766 — saved
2026-04-19 08:04:05,802 INFO train_multi TF=ALL epoch 10/50 train=0.6736 val=0.6674
2026-04-19 08:04:05,806 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 08:04:05,806 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 08:04:05,807 INFO train_multi TF=ALL: new best val=0.6674 — saved
2026-04-19 08:04:23,731 INFO train_multi TF=ALL epoch 11/50 train=0.6642 val=0.6577
2026-04-19 08:04:23,735 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 08:04:23,735 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 08:04:23,735 INFO train_multi TF=ALL: new best val=0.6577 — saved
2026-04-19 08:04:41,855 INFO train_multi TF=ALL epoch 12/50 train=0.6537 val=0.6454
2026-04-19 08:04:41,860 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 08:04:41,860 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 08:04:41,860 INFO train_multi TF=ALL: new best val=0.6454 — saved
2026-04-19 08:04:59,872 INFO train_multi TF=ALL epoch 13/50 train=0.6444 val=0.6394
2026-04-19 08:04:59,876 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 08:04:59,876 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 08:04:59,877 INFO train_multi TF=ALL: new best val=0.6394 — saved
2026-04-19 08:05:18,207 INFO train_multi TF=ALL epoch 14/50 train=0.6371 val=0.6346
2026-04-19 08:05:18,211 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 08:05:18,211 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 08:05:18,211 INFO train_multi TF=ALL: new best val=0.6346 — saved
2026-04-19 08:05:36,368 INFO train_multi TF=ALL epoch 15/50 train=0.6314 val=0.6269
2026-04-19 08:05:36,372 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 08:05:36,372 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 08:05:36,372 INFO train_multi TF=ALL: new best val=0.6269 — saved
2026-04-19 08:05:54,691 INFO train_multi TF=ALL epoch 16/50 train=0.6269 val=0.6245
2026-04-19 08:05:54,696 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 08:05:54,696 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 08:05:54,696 INFO train_multi TF=ALL: new best val=0.6245 — saved
2026-04-19 08:06:12,861 INFO train_multi TF=ALL epoch 17/50 train=0.6230 val=0.6214
2026-04-19 08:06:12,866 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 08:06:12,866 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 08:06:12,866 INFO train_multi TF=ALL: new best val=0.6214 — saved
2026-04-19 08:06:31,031 INFO train_multi TF=ALL epoch 18/50 train=0.6199 val=0.6220
2026-04-19 08:06:49,410 INFO train_multi TF=ALL epoch 19/50 train=0.6171 val=0.6209
2026-04-19 08:06:49,414 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 08:06:49,414 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 08:06:49,414 INFO train_multi TF=ALL: new best val=0.6209 — saved
2026-04-19 08:07:07,470 INFO train_multi TF=ALL epoch 20/50 train=0.6147 val=0.6206
2026-04-19 08:07:07,475 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 08:07:07,475 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 08:07:07,475 INFO train_multi TF=ALL: new best val=0.6206 — saved
2026-04-19 08:07:25,641 INFO train_multi TF=ALL epoch 21/50 train=0.6123 val=0.6172
2026-04-19 08:07:25,645 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 08:07:25,645 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 08:07:25,646 INFO train_multi TF=ALL: new best val=0.6172 — saved
2026-04-19 08:07:43,574 INFO train_multi TF=ALL epoch 22/50 train=0.6102 val=0.6168
2026-04-19 08:07:43,579 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 08:07:43,579 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 08:07:43,579 INFO train_multi TF=ALL: new best val=0.6168 — saved
2026-04-19 08:08:01,642 INFO train_multi TF=ALL epoch 23/50 train=0.6080 val=0.6127
2026-04-19 08:08:01,647 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 08:08:01,647 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 08:08:01,647 INFO train_multi TF=ALL: new best val=0.6127 — saved
2026-04-19 08:08:19,573 INFO train_multi TF=ALL epoch 24/50 train=0.6066 val=0.6170
2026-04-19 08:08:37,440 INFO train_multi TF=ALL epoch 25/50 train=0.6045 val=0.6140
2026-04-19 08:08:55,761 INFO train_multi TF=ALL epoch 26/50 train=0.6030 val=0.6143
2026-04-19 08:09:14,065 INFO train_multi TF=ALL epoch 27/50 train=0.6010 val=0.6125
2026-04-19 08:09:14,070 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 08:09:14,070 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 08:09:14,070 INFO train_multi TF=ALL: new best val=0.6125 — saved
2026-04-19 08:09:31,523 INFO train_multi TF=ALL epoch 28/50 train=0.5993 val=0.6125
2026-04-19 08:09:49,384 INFO train_multi TF=ALL epoch 29/50 train=0.5980 val=0.6140
2026-04-19 08:10:07,230 INFO train_multi TF=ALL epoch 30/50 train=0.5966 val=0.6145
2026-04-19 08:10:25,402 INFO train_multi TF=ALL epoch 31/50 train=0.5944 val=0.6123
2026-04-19 08:10:25,407 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-19 08:10:25,407 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 08:10:25,407 INFO train_multi TF=ALL: new best val=0.6123 — saved
2026-04-19 08:10:43,179 INFO train_multi TF=ALL epoch 32/50 train=0.5934 val=0.6130
2026-04-19 08:11:01,216 INFO train_multi TF=ALL epoch 33/50 train=0.5918 val=0.6138
2026-04-19 08:11:19,161 INFO train_multi TF=ALL epoch 34/50 train=0.5903 val=0.6162
2026-04-19 08:11:37,047 INFO train_multi TF=ALL epoch 35/50 train=0.5888 val=0.6171
2026-04-19 08:11:55,003 INFO train_multi TF=ALL epoch 36/50 train=0.5870 val=0.6128
2026-04-19 08:11:55,003 INFO train_multi TF=ALL early stop at epoch 36
2026-04-19 08:11:55,137 INFO === VectorStore: building similarity indices ===
2026-04-19 08:11:55,137 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-19 08:11:55,138 INFO Retrain complete.
2026-04-19 08:11:57,089 INFO Model gru: SUCCESS
2026-04-19 08:11:57,089 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-19 08:11:57,089 WARNING   [MISSING] regime_classifier → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_classifier.pkl
2026-04-19 08:11:57,089 INFO   [OK] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-19 08:11:57,089 WARNING   [MISSING] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-19 08:11:57,089 WARNING Missing weights: ['regime_classifier', 'rl_ppo'] — run retrain_incremental.py for each
2026-04-19 08:11:57,091 INFO Saved 18 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  Cleared journal for fresh Round 1 run
  START Round 1 - Backtest (val)
2026-04-19 08:11:57,622 INFO === STEP 6: BACKTEST + REINFORCED TRAINING (3 rounds, window=round1) ===
2026-04-19 08:11:57,623 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-04-19 08:11:57,624 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-19 08:11:57,624 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)

======================================================================
  BACKTEST + REINFORCED TRAINING COMPLETE  (0 rounds)
======================================================================
  Round     Trades       WR      PF   Sharpe
  ------------------------------------------

  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 0 entries
