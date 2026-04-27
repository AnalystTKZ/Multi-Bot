 Cleared done-check: training_summary.json
Environment : KAGGLE
  base      -> /kaggle/working/Multi-Bot/trading-system
  data      -> /kaggle/working/Multi-Bot/trading-system/training_data
  processed -> /kaggle/working/Multi-Bot/trading-system/processed_data
  ml_train  -> /kaggle/working/Multi-Bot/trading-system/ml_training
  weights   -> /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
  output    -> /kaggle/working
  kaggle/input -> /kaggle/input
    dataset: datasets  (has training_data=False, processed_data=False)
  WARNING: optional file missing (macro features reduced): /kaggle/working/Multi-Bot/trading-system/training_data/indices/VIX_1d.csv
  WARNING: optional file missing (macro features reduced): /kaggle/working/Multi-Bot/trading-system/training_data/fundamental/macro_releases.csv

All scripts and inputs verified.

=== Phase 0-5: Data preparation ===
  SKIP  Step 0 - Resample
  SKIP  Step 1 - Inventory
  SKIP  Step 2 - Cleaning
  SKIP  Step 3 - Alignment
  SKIP  Step 4 - Features
  START Step 5 - Split
2026-04-27 05:34:02,564 INFO Loading feature-engineered data...
2026-04-27 05:34:03,198 INFO Loaded 221743 rows, 202 features
2026-04-27 05:34:03,199 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-27 05:34:03,202 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-27 05:34:03,202 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-27 05:34:03,202 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-27 05:34:03,202 INFO No leakage confirmed: train < val < test timestamps

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
2026-04-27 05:34:05,587 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-27 05:34:05,587 INFO --- Training regime ---
2026-04-27 05:34:05,587 INFO Running retrain --model regime
2026-04-27 05:34:05,780 INFO retrain environment: KAGGLE
2026-04-27 05:34:07,381 INFO Device: CUDA (2 GPU(s))
2026-04-27 05:34:07,392 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 05:34:07,392 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 05:34:07,392 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 05:34:07,394 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 05:34:07,394 INFO Retrain data split: train
2026-04-27 05:34:07,395 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-27 05:34:07,567 INFO NumExpr defaulting to 4 threads.
2026-04-27 05:34:07,780 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-27 05:34:07,780 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 05:34:07,780 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 05:34:07,780 INFO Regime phase macro_correlations: 0.0s
2026-04-27 05:34:07,780 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-27 05:34:07,817 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-27 05:34:07,818 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:34:07,843 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:34:07,857 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:34:07,879 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:34:07,892 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:34:07,915 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:34:07,928 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:34:07,950 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:34:07,964 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:34:07,986 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:34:07,999 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:34:08,019 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:34:08,032 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:34:08,049 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:34:08,062 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:34:08,082 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:34:08,095 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:34:08,116 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:34:08,130 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:34:08,151 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:34:08,168 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 05:34:08,205 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 05:34:09,349 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 05:34:31,945 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 05:34:31,946 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 23.7s
2026-04-27 05:34:31,946 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 05:34:41,785 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 05:34:41,786 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 9.8s
2026-04-27 05:34:41,786 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 05:34:49,372 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 05:34:49,376 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 7.6s
2026-04-27 05:34:49,376 INFO Regime phase GMM HTF total: 41.2s
2026-04-27 05:34:49,376 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 05:35:58,446 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 05:35:58,449 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 69.1s
2026-04-27 05:35:58,452 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 05:36:28,559 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 05:36:28,566 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 30.1s
2026-04-27 05:36:28,566 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 05:36:50,169 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 05:36:50,169 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 21.6s
2026-04-27 05:36:50,169 INFO Regime phase GMM LTF total: 120.8s
2026-04-27 05:36:50,268 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-27 05:36:50,270 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:36:50,271 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:36:50,272 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:36:50,273 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:36:50,274 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:36:50,275 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:36:50,276 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:36:50,277 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:36:50,278 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:36:50,279 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:36:50,281 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 05:36:50,404 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:50,443 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:50,443 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:50,444 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:50,451 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:50,452 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:36:50,865 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-27 05:36:50,866 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-27 05:36:51,047 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:51,081 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:51,081 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:51,082 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:51,089 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:51,090 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:36:51,462 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-27 05:36:51,464 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-27 05:36:51,649 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:51,684 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:51,684 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:51,685 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:51,693 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:51,694 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:36:52,056 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-27 05:36:52,057 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-27 05:36:52,235 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:52,272 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:52,273 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:52,273 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:52,281 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:52,282 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:36:52,647 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-27 05:36:52,648 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-27 05:36:52,833 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:52,869 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:52,870 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:52,870 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:52,878 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:52,879 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:36:53,245 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-27 05:36:53,246 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-27 05:36:53,414 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:53,447 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:53,448 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:53,448 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:53,456 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:53,457 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:36:53,837 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-27 05:36:53,838 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-27 05:36:53,990 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 05:36:54,017 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-27 05:36:54,017 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 05:36:54,018 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 05:36:54,024 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 05:36:54,025 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:36:54,409 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-27 05:36:54,410 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-27 05:36:54,580 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:54,611 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:54,612 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:54,613 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:54,621 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:54,622 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:36:54,986 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-27 05:36:54,988 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-27 05:36:55,150 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:55,182 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:55,183 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:55,183 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:55,192 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:55,193 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:36:55,560 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-27 05:36:55,561 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-27 05:36:55,727 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:55,772 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:55,773 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:55,773 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:55,782 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:36:55,783 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:36:56,171 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-27 05:36:56,172 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-27 05:36:56,444 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 05:36:56,503 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-27 05:36:56,504 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 05:36:56,504 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 05:36:56,514 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 05:36:56,516 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 05:36:57,315 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 05:36:57,317 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-27 05:36:57,471 INFO Regime phase HTF dataset build: 7.2s (103290 samples)
2026-04-27 05:36:57,472 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=38114 dropped=65176 classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728})
2026-04-27 05:36:57,481 INFO RegimeClassifier[mode=htf_bias]: 38114 samples, classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728}, device=cuda
2026-04-27 05:36:57,482 INFO RegimeClassifier: sample weights — mean=0.708  ambiguous(<0.4)=0.0%
2026-04-27 05:36:57,772 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-04-27 05:36:57,773 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-27 05:37:02,515 INFO Regime epoch  1/50 — tr=0.7517 va=2.1440 acc=0.342 per_class={'BIAS_UP': 0.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.006}
2026-04-27 05:37:02,582 INFO Regime epoch  2/50 — tr=0.7455 va=2.0768 acc=0.351
2026-04-27 05:37:02,647 INFO Regime epoch  3/50 — tr=0.7380 va=2.0580 acc=0.356
2026-04-27 05:37:02,713 INFO Regime epoch  4/50 — tr=0.7224 va=2.0250 acc=0.375
2026-04-27 05:37:02,784 INFO Regime epoch  5/50 — tr=0.6995 va=1.9680 acc=0.400 per_class={'BIAS_UP': 0.009, 'BIAS_DOWN': 0.941, 'BIAS_NEUTRAL': 0.414}
2026-04-27 05:37:02,849 INFO Regime epoch  6/50 — tr=0.6708 va=1.8907 acc=0.442
2026-04-27 05:37:02,919 INFO Regime epoch  7/50 — tr=0.6449 va=1.8019 acc=0.566
2026-04-27 05:37:02,989 INFO Regime epoch  8/50 — tr=0.6159 va=1.7116 acc=0.678
2026-04-27 05:37:03,064 INFO Regime epoch  9/50 — tr=0.5924 va=1.6206 acc=0.777
2026-04-27 05:37:03,137 INFO Regime epoch 10/50 — tr=0.5715 va=1.5527 acc=0.845 per_class={'BIAS_UP': 0.826, 'BIAS_DOWN': 0.974, 'BIAS_NEUTRAL': 0.649}
2026-04-27 05:37:03,202 INFO Regime epoch 11/50 — tr=0.5555 va=1.5006 acc=0.885
2026-04-27 05:37:03,269 INFO Regime epoch 12/50 — tr=0.5430 va=1.4502 acc=0.905
2026-04-27 05:37:03,335 INFO Regime epoch 13/50 — tr=0.5327 va=1.4075 acc=0.919
2026-04-27 05:37:03,401 INFO Regime epoch 14/50 — tr=0.5254 va=1.3718 acc=0.928
2026-04-27 05:37:03,473 INFO Regime epoch 15/50 — tr=0.5195 va=1.3431 acc=0.932 per_class={'BIAS_UP': 0.98, 'BIAS_DOWN': 0.99, 'BIAS_NEUTRAL': 0.694}
2026-04-27 05:37:03,539 INFO Regime epoch 16/50 — tr=0.5134 va=1.3210 acc=0.935
2026-04-27 05:37:03,604 INFO Regime epoch 17/50 — tr=0.5100 va=1.2979 acc=0.938
2026-04-27 05:37:03,670 INFO Regime epoch 18/50 — tr=0.5070 va=1.2832 acc=0.941
2026-04-27 05:37:03,736 INFO Regime epoch 19/50 — tr=0.5045 va=1.2680 acc=0.943
2026-04-27 05:37:03,806 INFO Regime epoch 20/50 — tr=0.5018 va=1.2576 acc=0.945 per_class={'BIAS_UP': 0.99, 'BIAS_DOWN': 0.995, 'BIAS_NEUTRAL': 0.73}
2026-04-27 05:37:03,871 INFO Regime epoch 21/50 — tr=0.4996 va=1.2437 acc=0.948
2026-04-27 05:37:03,936 INFO Regime epoch 22/50 — tr=0.4986 va=1.2292 acc=0.952
2026-04-27 05:37:04,002 INFO Regime epoch 23/50 — tr=0.4960 va=1.2235 acc=0.953
2026-04-27 05:37:04,067 INFO Regime epoch 24/50 — tr=0.4950 va=1.2141 acc=0.955
2026-04-27 05:37:04,139 INFO Regime epoch 25/50 — tr=0.4936 va=1.2093 acc=0.956 per_class={'BIAS_UP': 0.996, 'BIAS_DOWN': 0.995, 'BIAS_NEUTRAL': 0.774}
2026-04-27 05:37:04,204 INFO Regime epoch 26/50 — tr=0.4919 va=1.2031 acc=0.958
2026-04-27 05:37:04,270 INFO Regime epoch 27/50 — tr=0.4913 va=1.2001 acc=0.958
2026-04-27 05:37:04,335 INFO Regime epoch 28/50 — tr=0.4909 va=1.1923 acc=0.960
2026-04-27 05:37:04,401 INFO Regime epoch 29/50 — tr=0.4898 va=1.1854 acc=0.962
2026-04-27 05:37:04,470 INFO Regime epoch 30/50 — tr=0.4886 va=1.1793 acc=0.963 per_class={'BIAS_UP': 0.998, 'BIAS_DOWN': 0.997, 'BIAS_NEUTRAL': 0.808}
2026-04-27 05:37:04,535 INFO Regime epoch 31/50 — tr=0.4885 va=1.1770 acc=0.964
2026-04-27 05:37:04,600 INFO Regime epoch 32/50 — tr=0.4875 va=1.1775 acc=0.964
2026-04-27 05:37:04,666 INFO Regime epoch 33/50 — tr=0.4874 va=1.1767 acc=0.964
2026-04-27 05:37:04,733 INFO Regime epoch 34/50 — tr=0.4867 va=1.1724 acc=0.965
2026-04-27 05:37:04,804 INFO Regime epoch 35/50 — tr=0.4857 va=1.1695 acc=0.965 per_class={'BIAS_UP': 0.998, 'BIAS_DOWN': 0.998, 'BIAS_NEUTRAL': 0.816}
2026-04-27 05:37:04,870 INFO Regime epoch 36/50 — tr=0.4854 va=1.1671 acc=0.966
2026-04-27 05:37:04,936 INFO Regime epoch 37/50 — tr=0.4857 va=1.1638 acc=0.967
2026-04-27 05:37:05,001 INFO Regime epoch 38/50 — tr=0.4848 va=1.1626 acc=0.967
2026-04-27 05:37:05,066 INFO Regime epoch 39/50 — tr=0.4842 va=1.1606 acc=0.967
2026-04-27 05:37:05,137 INFO Regime epoch 40/50 — tr=0.4846 va=1.1598 acc=0.968 per_class={'BIAS_UP': 0.998, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.826}
2026-04-27 05:37:05,203 INFO Regime epoch 41/50 — tr=0.4846 va=1.1587 acc=0.968
2026-04-27 05:37:05,275 INFO Regime epoch 42/50 — tr=0.4837 va=1.1569 acc=0.968
2026-04-27 05:37:05,340 INFO Regime epoch 43/50 — tr=0.4846 va=1.1564 acc=0.968
2026-04-27 05:37:05,407 INFO Regime epoch 44/50 — tr=0.4841 va=1.1568 acc=0.968
2026-04-27 05:37:05,484 INFO Regime epoch 45/50 — tr=0.4838 va=1.1588 acc=0.968 per_class={'BIAS_UP': 0.998, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.826}
2026-04-27 05:37:05,549 INFO Regime epoch 46/50 — tr=0.4836 va=1.1568 acc=0.968
2026-04-27 05:37:05,615 INFO Regime epoch 47/50 — tr=0.4839 va=1.1598 acc=0.968
2026-04-27 05:37:05,679 INFO Regime epoch 48/50 — tr=0.4836 va=1.1584 acc=0.968
2026-04-27 05:37:05,753 INFO Regime epoch 49/50 — tr=0.4832 va=1.1576 acc=0.968
2026-04-27 05:37:05,843 INFO Regime epoch 50/50 — tr=0.4839 va=1.1550 acc=0.968 per_class={'BIAS_UP': 0.998, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.829}
2026-04-27 05:37:05,853 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 05:37:05,853 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 05:37:05,854 INFO Regime phase HTF train: 8.4s
2026-04-27 05:37:05,978 INFO Regime HTF complete: acc=0.968, n=103290 per_class={'BIAS_UP': 0.998, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.829}
2026-04-27 05:37:05,980 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 05:37:06,129 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 05:37:06,141 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 11.41578947368421, 'BIAS_DOWN': 10.635761589403973, 'BIAS_NEUTRAL': 17.960468521229867}
2026-04-27 05:37:06,145 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 4338, 'mean': 0.00011110466582543088, 'mean_over_std': 0.027632581210502927}, 'BIAS_DOWN': {'n': 3212, 'mean': 6.063430222660252e-05, 'mean_over_std': 0.013898489590377157}, 'BIAS_NEUTRAL': {'n': 12266, 'mean': 1.2151554902814046e-05, 'mean_over_std': 0.0031957000513162357}}
2026-04-27 05:37:06,146 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 3651, 'mean': 0.0002852242108836959, 'mean_over_std': 0.07015017232491755}, 'BIAS_DOWN': {'n': 2644, 'mean': -0.00014900749618981248, 'mean_over_std': -0.03333037041298237}, 'BIAS_NEUTRAL': {'n': 1376, 'mean': 9.595797941249082e-05, 'mean_over_std': 0.02492411099521803}}
2026-04-27 05:37:06,146 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-27 05:37:06,148 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:06,149 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:06,151 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:06,152 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:06,154 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:06,156 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:06,157 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:06,158 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:06,160 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:06,161 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:06,164 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 05:37:06,174 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:06,178 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:06,178 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:06,179 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:06,179 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:06,182 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:06,782 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-27 05:37:06,785 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-27 05:37:06,915 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:06,917 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:06,918 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:06,919 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:06,919 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:06,921 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:07,499 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-27 05:37:07,501 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-27 05:37:07,637 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:07,639 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:07,640 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:07,641 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:07,641 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:07,643 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:08,244 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-27 05:37:08,247 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-27 05:37:08,379 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:08,381 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:08,382 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:08,382 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:08,383 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:08,385 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:08,961 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-27 05:37:08,964 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-27 05:37:09,097 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:09,099 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:09,100 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:09,100 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:09,101 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:09,103 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:09,681 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-27 05:37:09,684 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-27 05:37:09,820 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:09,822 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:09,823 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:09,823 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:09,824 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:09,826 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:10,414 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-27 05:37:10,417 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-27 05:37:10,547 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 05:37:10,549 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-27 05:37:10,550 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 05:37:10,550 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 05:37:10,550 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 05:37:10,552 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:11,138 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-27 05:37:11,141 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-27 05:37:11,276 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:11,278 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:11,279 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:11,280 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:11,280 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:11,282 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:11,861 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-27 05:37:11,863 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-27 05:37:11,994 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:11,996 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:11,997 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:11,998 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:11,998 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:12,000 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:12,578 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-27 05:37:12,581 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-27 05:37:12,714 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:12,716 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:12,717 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:12,717 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:12,717 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:12,720 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:13,287 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-27 05:37:13,289 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-27 05:37:13,438 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 05:37:13,442 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-27 05:37:13,443 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 05:37:13,443 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 05:37:13,444 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 05:37:13,447 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 05:37:14,700 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 05:37:14,705 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-27 05:37:14,998 INFO Regime phase LTF dataset build: 8.9s (401471 samples)
2026-04-27 05:37:15,001 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=299185 dropped=102286 classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127})
2026-04-27 05:37:15,059 INFO RegimeClassifier[mode=ltf_behaviour]: 299185 samples, classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127}, device=cuda
2026-04-27 05:37:15,060 INFO RegimeClassifier: sample weights — mean=0.693  ambiguous(<0.4)=0.0%
2026-04-27 05:37:15,062 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-04-27 05:37:15,062 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-27 05:37:15,648 INFO Regime epoch  1/50 — tr=0.8965 va=2.1560 acc=0.284 per_class={'TRENDING': 0.11, 'RANGING': 0.121, 'CONSOLIDATING': 0.37, 'VOLATILE': 0.657}
2026-04-27 05:37:16,158 INFO Regime epoch  2/50 — tr=0.8687 va=2.0615 acc=0.378
2026-04-27 05:37:16,638 INFO Regime epoch  3/50 — tr=0.8263 va=1.9388 acc=0.471
2026-04-27 05:37:17,119 INFO Regime epoch  4/50 — tr=0.7767 va=1.7947 acc=0.559
2026-04-27 05:37:17,648 INFO Regime epoch  5/50 — tr=0.7366 va=1.6661 acc=0.611 per_class={'TRENDING': 0.445, 'RANGING': 0.401, 'CONSOLIDATING': 0.747, 'VOLATILE': 0.964}
2026-04-27 05:37:18,147 INFO Regime epoch  6/50 — tr=0.7106 va=1.5759 acc=0.639
2026-04-27 05:37:18,644 INFO Regime epoch  7/50 — tr=0.6941 va=1.5181 acc=0.663
2026-04-27 05:37:19,139 INFO Regime epoch  8/50 — tr=0.6828 va=1.4854 acc=0.682
2026-04-27 05:37:19,653 INFO Regime epoch  9/50 — tr=0.6747 va=1.4631 acc=0.698
2026-04-27 05:37:20,180 INFO Regime epoch 10/50 — tr=0.6678 va=1.4316 acc=0.711 per_class={'TRENDING': 0.587, 'RANGING': 0.659, 'CONSOLIDATING': 0.747, 'VOLATILE': 0.954}
2026-04-27 05:37:20,691 INFO Regime epoch 11/50 — tr=0.6629 va=1.4135 acc=0.729
2026-04-27 05:37:21,182 INFO Regime epoch 12/50 — tr=0.6589 va=1.3969 acc=0.737
2026-04-27 05:37:21,674 INFO Regime epoch 13/50 — tr=0.6549 va=1.3737 acc=0.751
2026-04-27 05:37:22,171 INFO Regime epoch 14/50 — tr=0.6520 va=1.3635 acc=0.758
2026-04-27 05:37:22,728 INFO Regime epoch 15/50 — tr=0.6495 va=1.3503 acc=0.767 per_class={'TRENDING': 0.696, 'RANGING': 0.714, 'CONSOLIDATING': 0.764, 'VOLATILE': 0.934}
2026-04-27 05:37:23,233 INFO Regime epoch 16/50 — tr=0.6470 va=1.3398 acc=0.775
2026-04-27 05:37:23,719 INFO Regime epoch 17/50 — tr=0.6450 va=1.3235 acc=0.780
2026-04-27 05:37:24,219 INFO Regime epoch 18/50 — tr=0.6428 va=1.3202 acc=0.787
2026-04-27 05:37:24,714 INFO Regime epoch 19/50 — tr=0.6413 va=1.3108 acc=0.789
2026-04-27 05:37:25,249 INFO Regime epoch 20/50 — tr=0.6398 va=1.3010 acc=0.795 per_class={'TRENDING': 0.753, 'RANGING': 0.747, 'CONSOLIDATING': 0.774, 'VOLATILE': 0.919}
2026-04-27 05:37:25,776 INFO Regime epoch 21/50 — tr=0.6385 va=1.2999 acc=0.797
2026-04-27 05:37:26,302 INFO Regime epoch 22/50 — tr=0.6374 va=1.2869 acc=0.799
2026-04-27 05:37:26,805 INFO Regime epoch 23/50 — tr=0.6365 va=1.2824 acc=0.804
2026-04-27 05:37:27,285 INFO Regime epoch 24/50 — tr=0.6353 va=1.2783 acc=0.804
2026-04-27 05:37:27,812 INFO Regime epoch 25/50 — tr=0.6343 va=1.2775 acc=0.807 per_class={'TRENDING': 0.769, 'RANGING': 0.748, 'CONSOLIDATING': 0.805, 'VOLATILE': 0.915}
2026-04-27 05:37:28,309 INFO Regime epoch 26/50 — tr=0.6336 va=1.2738 acc=0.808
2026-04-27 05:37:28,793 INFO Regime epoch 27/50 — tr=0.6323 va=1.2704 acc=0.812
2026-04-27 05:37:29,282 INFO Regime epoch 28/50 — tr=0.6318 va=1.2698 acc=0.812
2026-04-27 05:37:29,771 INFO Regime epoch 29/50 — tr=0.6312 va=1.2645 acc=0.812
2026-04-27 05:37:30,314 INFO Regime epoch 30/50 — tr=0.6305 va=1.2589 acc=0.813 per_class={'TRENDING': 0.778, 'RANGING': 0.761, 'CONSOLIDATING': 0.811, 'VOLATILE': 0.913}
2026-04-27 05:37:30,811 INFO Regime epoch 31/50 — tr=0.6302 va=1.2612 acc=0.815
2026-04-27 05:37:31,291 INFO Regime epoch 32/50 — tr=0.6294 va=1.2576 acc=0.817
2026-04-27 05:37:31,775 INFO Regime epoch 33/50 — tr=0.6290 va=1.2546 acc=0.815
2026-04-27 05:37:32,267 INFO Regime epoch 34/50 — tr=0.6286 va=1.2571 acc=0.819
2026-04-27 05:37:32,806 INFO Regime epoch 35/50 — tr=0.6283 va=1.2563 acc=0.821 per_class={'TRENDING': 0.792, 'RANGING': 0.756, 'CONSOLIDATING': 0.836, 'VOLATILE': 0.903}
2026-04-27 05:37:33,294 INFO Regime epoch 36/50 — tr=0.6279 va=1.2522 acc=0.817
2026-04-27 05:37:33,778 INFO Regime epoch 37/50 — tr=0.6278 va=1.2535 acc=0.820
2026-04-27 05:37:34,266 INFO Regime epoch 38/50 — tr=0.6276 va=1.2496 acc=0.818
2026-04-27 05:37:34,785 INFO Regime epoch 39/50 — tr=0.6272 va=1.2538 acc=0.821
2026-04-27 05:37:35,316 INFO Regime epoch 40/50 — tr=0.6271 va=1.2538 acc=0.820 per_class={'TRENDING': 0.788, 'RANGING': 0.763, 'CONSOLIDATING': 0.82, 'VOLATILE': 0.914}
2026-04-27 05:37:35,850 INFO Regime epoch 41/50 — tr=0.6269 va=1.2510 acc=0.819
2026-04-27 05:37:36,364 INFO Regime epoch 42/50 — tr=0.6268 va=1.2540 acc=0.821
2026-04-27 05:37:36,855 INFO Regime epoch 43/50 — tr=0.6269 va=1.2525 acc=0.821
2026-04-27 05:37:37,356 INFO Regime epoch 44/50 — tr=0.6264 va=1.2488 acc=0.822
2026-04-27 05:37:37,891 INFO Regime epoch 45/50 — tr=0.6267 va=1.2510 acc=0.824 per_class={'TRENDING': 0.796, 'RANGING': 0.763, 'CONSOLIDATING': 0.836, 'VOLATILE': 0.904}
2026-04-27 05:37:38,389 INFO Regime epoch 46/50 — tr=0.6267 va=1.2526 acc=0.823
2026-04-27 05:37:38,869 INFO Regime epoch 47/50 — tr=0.6266 va=1.2473 acc=0.821
2026-04-27 05:37:39,361 INFO Regime epoch 48/50 — tr=0.6267 va=1.2501 acc=0.824
2026-04-27 05:37:39,852 INFO Regime epoch 49/50 — tr=0.6265 va=1.2537 acc=0.821
2026-04-27 05:37:40,387 INFO Regime epoch 50/50 — tr=0.6268 va=1.2522 acc=0.822 per_class={'TRENDING': 0.793, 'RANGING': 0.76, 'CONSOLIDATING': 0.83, 'VOLATILE': 0.91}
2026-04-27 05:37:40,426 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 05:37:40,426 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 05:37:40,429 INFO Regime phase LTF train: 25.4s
2026-04-27 05:37:40,554 INFO Regime LTF complete: acc=0.821, n=401471 per_class={'TRENDING': 0.789, 'RANGING': 0.767, 'CONSOLIDATING': 0.827, 'VOLATILE': 0.91}
2026-04-27 05:37:40,557 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 05:37:41,061 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 05:37:41,066 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 9.355291913830783, 'RANGING': 6.096303199751476, 'CONSOLIDATING': 5.598885793871866, 'VOLATILE': 6.771351107094442}
2026-04-27 05:37:41,075 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 29965, 'mean': 2.149062259083323e-05, 'mean_over_std': 0.011155601434349563}, 'RANGING': {'n': 19623, 'mean': -3.254493608872846e-06, 'mean_over_std': -0.001761415548292494}, 'CONSOLIDATING': {'n': 10050, 'mean': -3.039756158146958e-06, 'mean_over_std': -0.001792488091618151}, 'VOLATILE': {'n': 14985, 'mean': 1.8102764458593193e-05, 'mean_over_std': 0.006974696291378742}}
2026-04-27 05:37:41,076 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 26145, 'mean': 2.874435605138053e-05, 'mean_over_std': 0.01595450632691917}, 'RANGING': {'n': 7858, 'mean': 1.7829776785521727e-05, 'mean_over_std': 0.011851561804467463}, 'CONSOLIDATING': {'n': 8543, 'mean': 7.845231961250924e-06, 'mean_over_std': 0.004915839521549481}, 'VOLATILE': {'n': 13636, 'mean': 5.480114267306668e-06, 'mean_over_std': 0.0020378119917389375}}
2026-04-27 05:37:41,076 INFO Regime retrain total: 213.7s (504761 samples)
2026-04-27 05:37:41,092 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-27 05:37:41,092 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 05:37:41,092 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 05:37:41,092 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-27 05:37:41,093 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-27 05:37:41,093 INFO Retrain complete. Total wall-clock: 213.7s
2026-04-27 05:37:43,602 INFO Model regime: SUCCESS
2026-04-27 05:37:43,602 INFO --- Training gru ---
2026-04-27 05:37:43,603 INFO Running retrain --model gru
2026-04-27 05:37:43,840 INFO retrain environment: KAGGLE
2026-04-27 05:37:45,443 INFO Device: CUDA (2 GPU(s))
2026-04-27 05:37:45,454 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 05:37:45,455 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 05:37:45,455 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 05:37:45,455 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 05:37:45,456 INFO Retrain data split: train
2026-04-27 05:37:45,457 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-27 05:37:45,620 INFO NumExpr defaulting to 4 threads.
2026-04-27 05:37:45,831 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-27 05:37:45,832 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 05:37:45,832 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 05:37:45,832 INFO GRU phase macro_correlations: 0.0s
2026-04-27 05:37:45,832 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-27 05:37:45,832 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260427_053745
2026-04-27 05:37:45,835 WARNING WeightsManifest: no manifest at /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json — assuming compatible (run a full retrain to generate one)
2026-04-27 05:37:45,980 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:46,000 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:46,014 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:46,021 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:46,022 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-27 05:37:46,022 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 05:37:46,022 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 05:37:46,023 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-27 05:37:46,024 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:46,108 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-27 05:37:46,109 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:46,332 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-27 05:37:46,363 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:46,629 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:46,748 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:46,838 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:47,029 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:47,046 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:47,060 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:47,066 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:47,067 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:47,149 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-27 05:37:47,151 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:47,396 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-27 05:37:47,412 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:47,669 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:47,789 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:47,878 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:48,061 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:48,081 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:48,095 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:48,102 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:48,103 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:48,179 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-27 05:37:48,180 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:48,410 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-27 05:37:48,426 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:48,701 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:48,834 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:48,929 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:49,111 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:49,130 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:49,144 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:49,151 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:49,151 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:49,228 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-27 05:37:49,229 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:49,452 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-27 05:37:49,474 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:49,738 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:49,860 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:49,953 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:50,135 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:50,155 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:50,169 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:50,176 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:50,177 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:50,257 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-27 05:37:50,259 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:50,492 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-27 05:37:50,506 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:50,767 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:50,895 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:50,995 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:51,178 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:51,196 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:51,210 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:51,216 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:51,217 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:51,302 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-27 05:37:51,304 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:51,542 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-27 05:37:51,558 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:51,820 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:51,946 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:52,042 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:52,212 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 05:37:52,229 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 05:37:52,242 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 05:37:52,250 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 05:37:52,251 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:52,336 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-27 05:37:52,338 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:52,576 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-27 05:37:52,588 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:52,841 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:52,962 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:53,055 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:53,229 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:53,247 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:53,260 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:53,267 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:53,268 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:53,347 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-27 05:37:53,348 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:53,584 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-27 05:37:53,602 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:53,857 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:53,979 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:54,070 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:54,242 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:54,259 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:54,274 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:54,281 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:54,281 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:54,360 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-27 05:37:54,361 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:54,593 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-27 05:37:54,608 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:54,874 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:54,997 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:55,091 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:55,267 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:55,286 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:55,300 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:55,307 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:37:55,308 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:55,383 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-27 05:37:55,386 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:55,609 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-27 05:37:55,624 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:55,918 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:56,042 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:56,130 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:37:56,425 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 05:37:56,454 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 05:37:56,472 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 05:37:56,483 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 05:37:56,484 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 05:37:56,669 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 05:37:56,672 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 05:37:57,193 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 05:37:57,239 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-27 05:37:57,775 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-27 05:37:57,962 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 05:37:58,087 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 05:37:58,195 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-27 05:37:58,458 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-04-27 05:37:58,458 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-04-27 05:37:58,459 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-27 05:37:58,459 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-27 05:38:45,248 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-27 05:38:45,248 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-27 05:38:46,582 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-27 05:38:50,667 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=18, min_epochs=22)
2026-04-27 05:39:04,591 INFO train_multi TF=ALL epoch 1/50 train=0.8877 val=0.8816
2026-04-27 05:39:04,601 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:39:04,601 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:39:04,602 INFO train_multi TF=ALL: new best val=0.8816 — saved
2026-04-27 05:39:16,487 INFO train_multi TF=ALL epoch 2/50 train=0.8719 val=0.8521
2026-04-27 05:39:16,491 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:39:16,491 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:39:16,491 INFO train_multi TF=ALL: new best val=0.8521 — saved
2026-04-27 05:39:28,306 INFO train_multi TF=ALL epoch 3/50 train=0.7729 val=0.6888
2026-04-27 05:39:28,311 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:39:28,311 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:39:28,311 INFO train_multi TF=ALL: new best val=0.6888 — saved
2026-04-27 05:39:40,231 INFO train_multi TF=ALL epoch 4/50 train=0.6906 val=0.6880
2026-04-27 05:39:40,235 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:39:40,236 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:39:40,236 INFO train_multi TF=ALL: new best val=0.6880 — saved
2026-04-27 05:39:52,154 INFO train_multi TF=ALL epoch 5/50 train=0.6900 val=0.6880
2026-04-27 05:39:52,158 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:39:52,158 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:39:52,158 INFO train_multi TF=ALL: new best val=0.6880 — saved
2026-04-27 05:40:04,119 INFO train_multi TF=ALL epoch 6/50 train=0.6895 val=0.6880
2026-04-27 05:40:04,124 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:40:04,124 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:40:04,124 INFO train_multi TF=ALL: new best val=0.6880 — saved
2026-04-27 05:40:16,039 INFO train_multi TF=ALL epoch 7/50 train=0.6891 val=0.6879
2026-04-27 05:40:16,043 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:40:16,043 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:40:16,043 INFO train_multi TF=ALL: new best val=0.6879 — saved
2026-04-27 05:40:27,881 INFO train_multi TF=ALL epoch 8/50 train=0.6889 val=0.6879
2026-04-27 05:40:39,711 INFO train_multi TF=ALL epoch 9/50 train=0.6888 val=0.6879
2026-04-27 05:40:39,715 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:40:39,715 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:40:39,715 INFO train_multi TF=ALL: new best val=0.6879 — saved
2026-04-27 05:40:51,557 INFO train_multi TF=ALL epoch 10/50 train=0.6887 val=0.6878
2026-04-27 05:40:51,561 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:40:51,561 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:40:51,561 INFO train_multi TF=ALL: new best val=0.6878 — saved
2026-04-27 05:41:03,547 INFO train_multi TF=ALL epoch 11/50 train=0.6886 val=0.6878
2026-04-27 05:41:03,551 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:41:03,551 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:41:03,551 INFO train_multi TF=ALL: new best val=0.6878 — saved
2026-04-27 05:41:15,382 INFO train_multi TF=ALL epoch 12/50 train=0.6883 val=0.6877
2026-04-27 05:41:15,387 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:41:15,387 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:41:15,387 INFO train_multi TF=ALL: new best val=0.6877 — saved
2026-04-27 05:41:27,342 INFO train_multi TF=ALL epoch 13/50 train=0.6877 val=0.6869
2026-04-27 05:41:27,346 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:41:27,346 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:41:27,346 INFO train_multi TF=ALL: new best val=0.6869 — saved
2026-04-27 05:41:39,237 INFO train_multi TF=ALL epoch 14/50 train=0.6844 val=0.6794
2026-04-27 05:41:39,241 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:41:39,241 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:41:39,241 INFO train_multi TF=ALL: new best val=0.6794 — saved
2026-04-27 05:41:51,175 INFO train_multi TF=ALL epoch 15/50 train=0.6742 val=0.6664
2026-04-27 05:41:51,179 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:41:51,180 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:41:51,180 INFO train_multi TF=ALL: new best val=0.6664 — saved
2026-04-27 05:42:03,195 INFO train_multi TF=ALL epoch 16/50 train=0.6583 val=0.6462
2026-04-27 05:42:03,200 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:42:03,200 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:42:03,200 INFO train_multi TF=ALL: new best val=0.6462 — saved
2026-04-27 05:42:15,187 INFO train_multi TF=ALL epoch 17/50 train=0.6440 val=0.6334
2026-04-27 05:42:15,192 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:42:15,192 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:42:15,192 INFO train_multi TF=ALL: new best val=0.6334 — saved
2026-04-27 05:42:27,258 INFO train_multi TF=ALL epoch 18/50 train=0.6353 val=0.6287
2026-04-27 05:42:27,263 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:42:27,263 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:42:27,263 INFO train_multi TF=ALL: new best val=0.6287 — saved
2026-04-27 05:42:39,190 INFO train_multi TF=ALL epoch 19/50 train=0.6287 val=0.6265
2026-04-27 05:42:39,195 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:42:39,195 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:42:39,195 INFO train_multi TF=ALL: new best val=0.6265 — saved
2026-04-27 05:42:51,222 INFO train_multi TF=ALL epoch 20/50 train=0.6241 val=0.6227
2026-04-27 05:42:51,227 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:42:51,227 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:42:51,227 INFO train_multi TF=ALL: new best val=0.6227 — saved
2026-04-27 05:43:03,251 INFO train_multi TF=ALL epoch 21/50 train=0.6203 val=0.6211
2026-04-27 05:43:03,256 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:43:03,256 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:43:03,256 INFO train_multi TF=ALL: new best val=0.6211 — saved
2026-04-27 05:43:15,225 INFO train_multi TF=ALL epoch 22/50 train=0.6174 val=0.6195
2026-04-27 05:43:15,229 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:43:15,230 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:43:15,230 INFO train_multi TF=ALL: new best val=0.6195 — saved
2026-04-27 05:43:27,273 INFO train_multi TF=ALL epoch 23/50 train=0.6144 val=0.6179
2026-04-27 05:43:27,278 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:43:27,278 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:43:27,278 INFO train_multi TF=ALL: new best val=0.6179 — saved
2026-04-27 05:43:39,339 INFO train_multi TF=ALL epoch 24/50 train=0.6123 val=0.6183
2026-04-27 05:43:51,347 INFO train_multi TF=ALL epoch 25/50 train=0.6100 val=0.6159
2026-04-27 05:43:51,352 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:43:51,352 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:43:51,352 INFO train_multi TF=ALL: new best val=0.6159 — saved
2026-04-27 05:44:03,383 INFO train_multi TF=ALL epoch 26/50 train=0.6084 val=0.6167
2026-04-27 05:44:15,363 INFO train_multi TF=ALL epoch 27/50 train=0.6068 val=0.6176
2026-04-27 05:44:27,454 INFO train_multi TF=ALL epoch 28/50 train=0.6052 val=0.6162
2026-04-27 05:44:39,399 INFO train_multi TF=ALL epoch 29/50 train=0.6035 val=0.6172
2026-04-27 05:44:51,349 INFO train_multi TF=ALL epoch 30/50 train=0.6019 val=0.6173
2026-04-27 05:45:03,285 INFO train_multi TF=ALL epoch 31/50 train=0.6003 val=0.6144
2026-04-27 05:45:03,289 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:45:03,289 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:45:03,289 INFO train_multi TF=ALL: new best val=0.6144 — saved
2026-04-27 05:45:15,270 INFO train_multi TF=ALL epoch 32/50 train=0.5991 val=0.6109
2026-04-27 05:45:15,275 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:45:15,275 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:45:15,275 INFO train_multi TF=ALL: new best val=0.6109 — saved
2026-04-27 05:45:27,249 INFO train_multi TF=ALL epoch 33/50 train=0.5980 val=0.6116
2026-04-27 05:45:39,199 INFO train_multi TF=ALL epoch 34/50 train=0.5967 val=0.6145
2026-04-27 05:45:51,110 INFO train_multi TF=ALL epoch 35/50 train=0.5957 val=0.6087
2026-04-27 05:45:51,114 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:45:51,114 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:45:51,114 INFO train_multi TF=ALL: new best val=0.6087 — saved
2026-04-27 05:46:03,165 INFO train_multi TF=ALL epoch 36/50 train=0.5943 val=0.6133
2026-04-27 05:46:15,083 INFO train_multi TF=ALL epoch 37/50 train=0.5932 val=0.6096
2026-04-27 05:46:27,025 INFO train_multi TF=ALL epoch 38/50 train=0.5921 val=0.6136
2026-04-27 05:46:38,930 INFO train_multi TF=ALL epoch 39/50 train=0.5908 val=0.6117
2026-04-27 05:46:50,860 INFO train_multi TF=ALL epoch 40/50 train=0.5898 val=0.6117
2026-04-27 05:47:02,852 INFO train_multi TF=ALL epoch 41/50 train=0.5882 val=0.6142
2026-04-27 05:47:14,835 INFO train_multi TF=ALL epoch 42/50 train=0.5870 val=0.6139
2026-04-27 05:47:26,823 INFO train_multi TF=ALL epoch 43/50 train=0.5859 val=0.6130
2026-04-27 05:47:38,763 INFO train_multi TF=ALL epoch 44/50 train=0.5849 val=0.6123
2026-04-27 05:47:50,636 INFO train_multi TF=ALL epoch 45/50 train=0.5838 val=0.6130
2026-04-27 05:48:02,678 INFO train_multi TF=ALL epoch 46/50 train=0.5823 val=0.6113
2026-04-27 05:48:14,624 INFO train_multi TF=ALL epoch 47/50 train=0.5812 val=0.6123
2026-04-27 05:48:26,575 INFO train_multi TF=ALL epoch 48/50 train=0.5800 val=0.6172
2026-04-27 05:48:38,563 INFO train_multi TF=ALL epoch 49/50 train=0.5787 val=0.6129
2026-04-27 05:48:50,703 INFO train_multi TF=ALL epoch 50/50 train=0.5779 val=0.6149
2026-04-27 05:48:50,840 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-27 05:48:50,840 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-27 05:48:50,840 INFO Retrain complete. Total wall-clock: 665.4s
2026-04-27 05:48:52,777 INFO Model gru: SUCCESS
2026-04-27 05:48:52,777 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:48:52,777 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 05:48:52,777 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 05:48:52,777 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-04-27 05:48:52,777 INFO   [DEFERRED] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip (expected after Round 1)
2026-04-27 05:48:52,777 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-04-27 05:48:52,778 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer', 'rl_ppo']
2026-04-27 05:48:52,779 INFO Saved 33 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  Cleared journal for fresh Round 1 run
  START Round 1 - Backtest (val)
2026-04-27 05:48:53,406 INFO === STEP 6: BACKTEST (round1) ===
2026-04-27 05:48:53,407 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-04-27 05:48:53,408 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-27 05:48:53,408 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
2026-04-27 05:48:55,836 WARNING QualityScorer unavailable (weights missing or load failed)
2026-04-27 05:51:02,770 WARNING ml_trader: portfolio drawdown 8.1% after trade exit — halting all trading

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260427_054855.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)               22  13.6%   0.52   -5.1% 13.6%  0.0%   8.1%   -3.94
  gate_diagnostics: bars=5289 no_signal=19 quality_block=0 session_skip=1872 density=0 pm_reject=0 daily_skip=3366 cooldown=10 daily_halt_events=5 enforce_daily_halt=True
  no_signal_reasons: weak_gru_direction=9, htf_bias_conflict=6, trend_pullback_conflict=4

Calibration Summary:
  all          [OK] Calibration OK — p_win correlates with actual win rate.
  ml_trader    [OK] Calibration OK — p_win correlates with actual win rate.
2026-04-27 05:51:03,389 INFO Round 1 backtest — 22 trades | avg WR=13.6% | avg PF=0.52 | avg Sharpe=-3.94
2026-04-27 05:51:03,389 INFO   ml_trader: 22 trades | WR=13.6% | PF=0.52 | Return=-5.1% | DD=8.1% | Sharpe=-3.94
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 22
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (22 rows)
2026-04-27 05:51:03,616 INFO Round 1: wrote 22 journal entries (total in file: 22)
  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 22 entries

=== Round 1 → Quality + RL retrain skipped (validation journal is not train data) ===

=== Round 2: BLIND backtest on test window (unseen 2yr) ===
  START Round 2 - Blind backtest (test)
2026-04-27 05:51:04,220 INFO === STEP 6: BACKTEST (round2) ===
2026-04-27 05:51:04,221 INFO BT_WINDOW=round2 — BLIND backtest: 2023-08-07 → 2025-08-05 (test set)
2026-04-27 05:51:04,221 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-27 05:51:04,221 INFO Round 2 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-27 05:51:06,632 WARNING QualityScorer unavailable (weights missing or load failed)
2026-04-27 05:53:15,047 WARNING ml_trader: portfolio drawdown 8.1% after trade exit — halting all trading
2026-04-27 05:53:15,660 INFO Round 2 backtest — 27 trades | avg WR=22.2% | avg PF=1.16 | avg Sharpe=0.87
2026-04-27 05:53:15,660 INFO   ml_trader: 27 trades | WR=22.2% | PF=1.16 | Return=1.9% | DD=8.1% | Sharpe=0.87
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 27
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (27 rows)
2026-04-27 05:53:15,939 INFO Round 2: wrote 27 journal entries (total in file: 49)
  DONE  Round 2 - Blind backtest (test)
  Saved Round 2 result → round2_summary.json
  Journal after Round 2: 49 entries

=== Round 2 → Quality + RL retrain skipped (blind-test journal is not train data) ===

=== Round 3: Incremental retrain on train split only ===
  START Retrain gru [train-split retrain]
2026-04-27 05:53:16,271 INFO retrain environment: KAGGLE
2026-04-27 05:53:17,875 INFO Device: CUDA (2 GPU(s))
2026-04-27 05:53:17,885 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 05:53:17,885 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 05:53:17,885 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 05:53:17,886 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 05:53:17,886 INFO Retrain data split: train
2026-04-27 05:53:17,887 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-27 05:53:18,031 INFO NumExpr defaulting to 4 threads.
2026-04-27 05:53:18,225 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-27 05:53:18,225 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 05:53:18,225 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 05:53:18,466 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-27 05:53:18,466 INFO GRU phase macro_correlations: 0.0s
2026-04-27 05:53:18,466 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-27 05:53:18,468 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260427_055318
2026-04-27 05:53:18,471 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-04-27 05:53:18,616 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:18,636 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:18,649 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:18,656 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:18,657 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-27 05:53:18,658 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 05:53:18,658 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 05:53:18,658 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-27 05:53:18,659 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:18,742 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-27 05:53:18,743 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:18,993 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-27 05:53:19,024 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:19,296 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:19,422 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:19,519 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:19,718 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:19,737 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:19,751 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:19,758 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:19,759 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:19,840 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-27 05:53:19,842 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:20,090 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-27 05:53:20,105 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:20,375 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:20,504 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:20,600 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:20,779 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:20,799 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:20,815 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:20,823 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:20,823 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:20,911 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-27 05:53:20,913 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:21,169 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-27 05:53:21,186 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:21,452 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:21,577 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:21,673 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:21,855 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:21,875 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:21,888 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:21,895 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:21,896 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:21,979 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-27 05:53:21,981 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:22,233 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-27 05:53:22,256 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:22,525 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:22,652 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:22,748 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:22,924 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:22,943 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:22,958 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:22,965 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:22,966 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:23,048 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-27 05:53:23,050 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:23,316 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-27 05:53:23,332 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:23,611 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:23,738 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:23,832 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:24,013 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:24,031 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:24,046 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:24,053 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:24,053 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:24,136 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-27 05:53:24,138 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:24,387 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-27 05:53:24,404 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:24,684 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:24,817 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:24,915 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:25,078 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 05:53:25,095 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 05:53:25,107 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 05:53:25,114 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 05:53:25,114 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:25,195 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-27 05:53:25,196 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:25,440 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-27 05:53:25,456 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:25,741 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:25,887 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:25,987 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:26,165 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:26,184 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:26,198 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:26,205 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:26,206 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:26,287 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-27 05:53:26,289 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:26,527 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-27 05:53:26,545 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:26,799 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:26,927 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:27,021 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:27,217 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:27,236 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:27,250 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:27,257 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:27,258 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:27,340 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-27 05:53:27,341 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:27,589 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-27 05:53:27,605 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:27,877 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:28,003 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:28,097 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:28,284 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:28,305 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:28,319 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:28,326 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 05:53:28,327 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:28,413 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-27 05:53:28,415 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:28,666 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-27 05:53:28,683 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:28,944 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:29,073 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:29,174 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:53:29,458 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 05:53:29,485 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 05:53:29,502 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 05:53:29,514 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 05:53:29,515 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 05:53:29,686 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 05:53:29,689 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 05:53:30,214 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 05:53:30,260 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-27 05:53:30,785 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-27 05:53:30,985 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 05:53:31,117 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 05:53:31,231 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-27 05:53:31,231 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-27 05:53:31,231 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-27 05:54:20,285 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-27 05:54:20,285 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-27 05:54:21,621 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-27 05:54:25,821 INFO train_multi TF=ALL: warm-start detected — using CosineAnnealingLR (lr=3e-05, patience=12)
2026-04-27 05:54:39,381 INFO train_multi TF=ALL epoch 1/50 train=0.5931 val=0.6110
2026-04-27 05:54:39,385 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:54:39,385 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:54:39,385 INFO train_multi TF=ALL: new best val=0.6110 — saved
2026-04-27 05:54:51,326 INFO train_multi TF=ALL epoch 2/50 train=0.5931 val=0.6109
2026-04-27 05:54:51,330 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:54:51,330 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:54:51,330 INFO train_multi TF=ALL: new best val=0.6109 — saved
2026-04-27 05:55:03,353 INFO train_multi TF=ALL epoch 3/50 train=0.5924 val=0.6105
2026-04-27 05:55:03,358 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 05:55:03,358 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 05:55:03,358 INFO train_multi TF=ALL: new best val=0.6105 — saved
2026-04-27 05:55:15,284 INFO train_multi TF=ALL epoch 4/50 train=0.5925 val=0.6106
2026-04-27 05:55:27,295 INFO train_multi TF=ALL epoch 5/50 train=0.5922 val=0.6111
2026-04-27 05:55:39,150 INFO train_multi TF=ALL epoch 6/50 train=0.5920 val=0.6113
2026-04-27 05:55:51,129 INFO train_multi TF=ALL epoch 7/50 train=0.5920 val=0.6112
2026-04-27 05:56:03,082 INFO train_multi TF=ALL epoch 8/50 train=0.5917 val=0.6115
2026-04-27 05:56:15,011 INFO train_multi TF=ALL epoch 9/50 train=0.5916 val=0.6120
2026-04-27 05:56:27,100 INFO train_multi TF=ALL epoch 10/50 train=0.5915 val=0.6118
2026-04-27 05:56:39,131 INFO train_multi TF=ALL epoch 11/50 train=0.5917 val=0.6121
2026-04-27 05:56:51,106 INFO train_multi TF=ALL epoch 12/50 train=0.5911 val=0.6116
2026-04-27 05:57:03,245 INFO train_multi TF=ALL epoch 13/50 train=0.5911 val=0.6119
2026-04-27 05:57:15,222 INFO train_multi TF=ALL epoch 14/50 train=0.5907 val=0.6113
2026-04-27 05:57:27,189 INFO train_multi TF=ALL epoch 15/50 train=0.5905 val=0.6114
2026-04-27 05:57:27,190 INFO train_multi TF=ALL early stop at epoch 15
2026-04-27 05:57:27,325 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-27 05:57:27,326 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-27 05:57:27,326 INFO Retrain complete. Total wall-clock: 249.4s
  DONE  Retrain gru [train-split retrain]
  START Retrain regime [train-split retrain]
2026-04-27 05:57:29,426 INFO retrain environment: KAGGLE
2026-04-27 05:57:31,012 INFO Device: CUDA (2 GPU(s))
2026-04-27 05:57:31,021 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 05:57:31,021 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 05:57:31,021 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 05:57:31,022 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 05:57:31,022 INFO Retrain data split: train
2026-04-27 05:57:31,023 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-27 05:57:31,167 INFO NumExpr defaulting to 4 threads.
2026-04-27 05:57:31,361 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-27 05:57:31,361 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 05:57:31,361 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 05:57:31,362 INFO Regime phase macro_correlations: 0.0s
2026-04-27 05:57:31,362 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-27 05:57:31,398 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-27 05:57:31,399 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:57:31,426 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:57:31,440 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:57:31,461 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:57:31,476 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:57:31,498 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:57:31,512 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:57:31,534 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:57:31,548 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:57:31,570 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:57:31,584 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:57:31,606 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:57:31,620 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:57:31,640 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:57:31,655 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:57:31,678 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:57:31,692 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:57:31,714 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:57:31,729 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:57:31,751 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 05:57:31,768 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 05:57:31,806 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 05:57:32,590 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 05:57:56,215 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 05:57:56,217 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 24.4s
2026-04-27 05:57:56,217 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 05:58:06,488 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 05:58:06,493 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 10.3s
2026-04-27 05:58:06,493 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 05:58:14,476 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 05:58:14,480 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 8.0s
2026-04-27 05:58:14,480 INFO Regime phase GMM HTF total: 42.7s
2026-04-27 05:58:14,481 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 05:59:26,288 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 05:59:26,291 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 71.8s
2026-04-27 05:59:26,292 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 05:59:58,217 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 05:59:58,221 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 31.9s
2026-04-27 05:59:58,222 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 06:00:20,497 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 06:00:20,498 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 22.3s
2026-04-27 06:00:20,498 INFO Regime phase GMM LTF total: 126.0s
2026-04-27 06:00:20,601 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-27 06:00:20,603 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:20,604 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:20,605 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:20,607 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:20,607 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:20,609 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:20,610 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:20,611 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:20,612 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:20,613 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:20,615 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 06:00:20,737 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:20,779 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:20,780 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:20,781 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:20,790 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:20,791 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:21,212 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-27 06:00:21,213 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-27 06:00:21,390 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:21,429 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:21,430 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:21,430 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:21,438 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:21,439 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:21,809 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-27 06:00:21,811 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-27 06:00:21,996 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:22,031 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:22,032 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:22,033 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:22,041 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:22,041 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:22,410 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-27 06:00:22,411 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-27 06:00:22,580 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:22,612 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:22,613 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:22,613 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:22,621 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:22,622 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:22,988 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-27 06:00:22,989 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-27 06:00:23,175 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:23,209 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:23,210 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:23,211 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:23,218 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:23,219 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:23,593 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-27 06:00:23,595 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-27 06:00:23,761 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:23,795 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:23,796 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:23,796 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:23,804 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:23,805 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:24,174 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-27 06:00:24,175 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-27 06:00:24,334 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 06:00:24,362 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-27 06:00:24,363 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 06:00:24,364 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 06:00:24,371 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 06:00:24,372 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:24,745 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-27 06:00:24,746 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-27 06:00:24,920 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:24,953 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:24,954 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:24,954 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:24,962 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:24,963 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:25,338 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-27 06:00:25,339 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-27 06:00:25,519 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:25,554 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:25,555 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:25,555 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:25,563 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:25,564 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:25,975 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-27 06:00:25,976 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-27 06:00:26,155 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:26,189 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:26,190 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:26,190 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:26,198 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:26,198 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:26,562 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-27 06:00:26,563 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-27 06:00:26,835 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 06:00:26,891 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-27 06:00:26,893 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 06:00:26,893 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 06:00:26,903 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 06:00:26,905 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 06:00:27,710 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 06:00:27,712 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-27 06:00:27,879 INFO Regime phase HTF dataset build: 7.3s (103290 samples)
2026-04-27 06:00:27,879 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260427_060027
2026-04-27 06:00:28,081 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=34, n_classes=3)
2026-04-27 06:00:28,082 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=38114 dropped=65176 classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728})
2026-04-27 06:00:28,092 INFO RegimeClassifier[mode=htf_bias]: 38114 samples, classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728}, device=cuda
2026-04-27 06:00:28,093 INFO RegimeClassifier: sample weights — mean=0.708  ambiguous(<0.4)=0.0%
2026-04-27 06:00:28,093 INFO RegimeClassifier[mode=htf_bias]: warm start from existing weights
2026-04-27 06:00:28,093 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-27 06:00:30,334 INFO Regime epoch  1/50 — tr=0.4837 va=1.1539 acc=0.968 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.83}
2026-04-27 06:00:30,403 INFO Regime epoch  2/50 — tr=0.4833 va=1.1559 acc=0.968
2026-04-27 06:00:30,471 INFO Regime epoch  3/50 — tr=0.4831 va=1.1539 acc=0.968
2026-04-27 06:00:30,539 INFO Regime epoch  4/50 — tr=0.4837 va=1.1550 acc=0.968
2026-04-27 06:00:30,617 INFO Regime epoch  5/50 — tr=0.4829 va=1.1512 acc=0.968 per_class={'BIAS_UP': 0.998, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.826}
2026-04-27 06:00:30,691 INFO Regime epoch  6/50 — tr=0.4832 va=1.1504 acc=0.968
2026-04-27 06:00:30,765 INFO Regime epoch  7/50 — tr=0.4822 va=1.1486 acc=0.969
2026-04-27 06:00:30,837 INFO Regime epoch  8/50 — tr=0.4819 va=1.1476 acc=0.969
2026-04-27 06:00:30,905 INFO Regime epoch  9/50 — tr=0.4815 va=1.1418 acc=0.969
2026-04-27 06:00:30,981 INFO Regime epoch 10/50 — tr=0.4813 va=1.1414 acc=0.970 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.836}
2026-04-27 06:00:31,049 INFO Regime epoch 11/50 — tr=0.4805 va=1.1374 acc=0.970
2026-04-27 06:00:31,118 INFO Regime epoch 12/50 — tr=0.4797 va=1.1332 acc=0.971
2026-04-27 06:00:31,186 INFO Regime epoch 13/50 — tr=0.4792 va=1.1340 acc=0.972
2026-04-27 06:00:31,254 INFO Regime epoch 14/50 — tr=0.4792 va=1.1304 acc=0.972
2026-04-27 06:00:31,326 INFO Regime epoch 15/50 — tr=0.4783 va=1.1266 acc=0.972 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.846}
2026-04-27 06:00:31,394 INFO Regime epoch 16/50 — tr=0.4773 va=1.1260 acc=0.972
2026-04-27 06:00:31,466 INFO Regime epoch 17/50 — tr=0.4777 va=1.1218 acc=0.973
2026-04-27 06:00:31,540 INFO Regime epoch 18/50 — tr=0.4774 va=1.1186 acc=0.974
2026-04-27 06:00:31,613 INFO Regime epoch 19/50 — tr=0.4771 va=1.1157 acc=0.974
2026-04-27 06:00:31,686 INFO Regime epoch 20/50 — tr=0.4766 va=1.1148 acc=0.974 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.858}
2026-04-27 06:00:31,760 INFO Regime epoch 21/50 — tr=0.4756 va=1.1127 acc=0.974
2026-04-27 06:00:31,837 INFO Regime epoch 22/50 — tr=0.4757 va=1.1119 acc=0.975
2026-04-27 06:00:31,910 INFO Regime epoch 23/50 — tr=0.4752 va=1.1107 acc=0.975
2026-04-27 06:00:31,984 INFO Regime epoch 24/50 — tr=0.4752 va=1.1068 acc=0.975
2026-04-27 06:00:32,060 INFO Regime epoch 25/50 — tr=0.4752 va=1.1063 acc=0.975 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.864}
2026-04-27 06:00:32,134 INFO Regime epoch 26/50 — tr=0.4747 va=1.1041 acc=0.976
2026-04-27 06:00:32,209 INFO Regime epoch 27/50 — tr=0.4742 va=1.1043 acc=0.976
2026-04-27 06:00:32,282 INFO Regime epoch 28/50 — tr=0.4743 va=1.1012 acc=0.977
2026-04-27 06:00:32,351 INFO Regime epoch 29/50 — tr=0.4738 va=1.1004 acc=0.977
2026-04-27 06:00:32,424 INFO Regime epoch 30/50 — tr=0.4734 va=1.1001 acc=0.977 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.872}
2026-04-27 06:00:32,496 INFO Regime epoch 31/50 — tr=0.4739 va=1.0973 acc=0.978
2026-04-27 06:00:32,563 INFO Regime epoch 32/50 — tr=0.4739 va=1.0958 acc=0.978
2026-04-27 06:00:32,634 INFO Regime epoch 33/50 — tr=0.4736 va=1.0957 acc=0.978
2026-04-27 06:00:32,705 INFO Regime epoch 34/50 — tr=0.4735 va=1.0955 acc=0.978
2026-04-27 06:00:32,781 INFO Regime epoch 35/50 — tr=0.4732 va=1.0935 acc=0.978 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.877}
2026-04-27 06:00:32,850 INFO Regime epoch 36/50 — tr=0.4727 va=1.0930 acc=0.979
2026-04-27 06:00:32,919 INFO Regime epoch 37/50 — tr=0.4726 va=1.0932 acc=0.978
2026-04-27 06:00:32,987 INFO Regime epoch 38/50 — tr=0.4727 va=1.0918 acc=0.978
2026-04-27 06:00:33,058 INFO Regime epoch 39/50 — tr=0.4734 va=1.0904 acc=0.979
2026-04-27 06:00:33,134 INFO Regime epoch 40/50 — tr=0.4728 va=1.0891 acc=0.979 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.884}
2026-04-27 06:00:33,203 INFO Regime epoch 41/50 — tr=0.4724 va=1.0912 acc=0.979
2026-04-27 06:00:33,269 INFO Regime epoch 42/50 — tr=0.4725 va=1.0899 acc=0.979
2026-04-27 06:00:33,337 INFO Regime epoch 43/50 — tr=0.4729 va=1.0907 acc=0.979
2026-04-27 06:00:33,405 INFO Regime epoch 44/50 — tr=0.4724 va=1.0923 acc=0.979
2026-04-27 06:00:33,479 INFO Regime epoch 45/50 — tr=0.4724 va=1.0916 acc=0.979 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.883}
2026-04-27 06:00:33,548 INFO Regime epoch 46/50 — tr=0.4724 va=1.0919 acc=0.979
2026-04-27 06:00:33,619 INFO Regime epoch 47/50 — tr=0.4728 va=1.0933 acc=0.978
2026-04-27 06:00:33,684 INFO Regime epoch 48/50 — tr=0.4724 va=1.0917 acc=0.978
2026-04-27 06:00:33,751 INFO Regime epoch 49/50 — tr=0.4721 va=1.0934 acc=0.979
2026-04-27 06:00:33,825 INFO Regime epoch 50/50 — tr=0.4724 va=1.0930 acc=0.978 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.88}
2026-04-27 06:00:33,825 INFO Regime early stop at epoch 50 (no_improve=10)
2026-04-27 06:00:33,833 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 06:00:33,833 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 06:00:33,834 INFO Regime phase HTF train: 5.8s
2026-04-27 06:00:33,955 INFO Regime HTF complete: acc=0.979, n=103290 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.884}
2026-04-27 06:00:33,957 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 06:00:34,113 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 06:00:34,116 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 11.41578947368421, 'BIAS_DOWN': 10.635761589403973, 'BIAS_NEUTRAL': 17.960468521229867}
2026-04-27 06:00:34,120 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 4338, 'mean': 0.00011110466582543088, 'mean_over_std': 0.027632581210502927}, 'BIAS_DOWN': {'n': 3212, 'mean': 6.063430222660252e-05, 'mean_over_std': 0.013898489590377157}, 'BIAS_NEUTRAL': {'n': 12266, 'mean': 1.2151554902814046e-05, 'mean_over_std': 0.0031957000513162357}}
2026-04-27 06:00:34,120 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 3651, 'mean': 0.0002852242108836959, 'mean_over_std': 0.07015017232491755}, 'BIAS_DOWN': {'n': 2644, 'mean': -0.00014900749618981248, 'mean_over_std': -0.03333037041298237}, 'BIAS_NEUTRAL': {'n': 1376, 'mean': 9.595797941249082e-05, 'mean_over_std': 0.02492411099521803}}
2026-04-27 06:00:34,121 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-27 06:00:34,123 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:34,125 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:34,126 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:34,128 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:34,129 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:34,131 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:34,132 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:34,134 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:34,136 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:34,137 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:34,140 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 06:00:34,151 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:34,155 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:34,155 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:34,156 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:34,156 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:34,158 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:34,786 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-27 06:00:34,789 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-27 06:00:34,925 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:34,927 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:34,928 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:34,928 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:34,929 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:34,931 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:35,549 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-27 06:00:35,552 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-27 06:00:35,688 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:35,690 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:35,691 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:35,691 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:35,692 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:35,694 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:36,323 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-27 06:00:36,326 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-27 06:00:36,460 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:36,462 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:36,463 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:36,463 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:36,464 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:36,466 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:37,053 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-27 06:00:37,056 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-27 06:00:37,200 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:37,202 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:37,203 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:37,203 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:37,204 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:37,206 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:37,801 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-27 06:00:37,804 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-27 06:00:37,941 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:37,944 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:37,944 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:37,945 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:37,945 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:37,947 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:38,538 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-27 06:00:38,541 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-27 06:00:38,671 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 06:00:38,673 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-27 06:00:38,674 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 06:00:38,674 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 06:00:38,674 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 06:00:38,676 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:39,259 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-27 06:00:39,262 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-27 06:00:39,397 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:39,399 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:39,400 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:39,400 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:39,401 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:39,403 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:40,006 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-27 06:00:40,009 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-27 06:00:40,140 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:40,145 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:40,146 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:40,146 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:40,146 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:40,148 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:40,729 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-27 06:00:40,732 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-27 06:00:40,870 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:40,872 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:40,873 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:40,874 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:40,874 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 06:00:40,876 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 06:00:41,479 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-27 06:00:41,482 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-27 06:00:41,636 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 06:00:41,639 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-27 06:00:41,641 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 06:00:41,641 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 06:00:41,641 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 06:00:41,645 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 06:00:43,018 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 06:00:43,023 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-27 06:00:43,311 INFO Regime phase LTF dataset build: 9.2s (401471 samples)
2026-04-27 06:00:43,312 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260427_060043
2026-04-27 06:00:43,317 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=18, n_classes=4)
2026-04-27 06:00:43,320 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=299185 dropped=102286 classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127})
2026-04-27 06:00:43,379 INFO RegimeClassifier[mode=ltf_behaviour]: 299185 samples, classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127}, device=cuda
2026-04-27 06:00:43,380 INFO RegimeClassifier: sample weights — mean=0.693  ambiguous(<0.4)=0.0%
2026-04-27 06:00:43,380 INFO RegimeClassifier[mode=ltf_behaviour]: warm start from existing weights
2026-04-27 06:00:43,380 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-27 06:00:43,974 INFO Regime epoch  1/50 — tr=0.6266 va=1.2482 acc=0.821 per_class={'TRENDING': 0.79, 'RANGING': 0.764, 'CONSOLIDATING': 0.829, 'VOLATILE': 0.91}
2026-04-27 06:00:44,471 INFO Regime epoch  2/50 — tr=0.6267 va=1.2501 acc=0.820
2026-04-27 06:00:44,996 INFO Regime epoch  3/50 — tr=0.6266 va=1.2513 acc=0.821
2026-04-27 06:00:45,509 INFO Regime epoch  4/50 — tr=0.6263 va=1.2488 acc=0.820
2026-04-27 06:00:46,096 INFO Regime epoch  5/50 — tr=0.6261 va=1.2501 acc=0.823 per_class={'TRENDING': 0.795, 'RANGING': 0.765, 'CONSOLIDATING': 0.821, 'VOLATILE': 0.912}
2026-04-27 06:00:46,612 INFO Regime epoch  6/50 — tr=0.6263 va=1.2476 acc=0.824
2026-04-27 06:00:47,123 INFO Regime epoch  7/50 — tr=0.6261 va=1.2459 acc=0.820
2026-04-27 06:00:47,628 INFO Regime epoch  8/50 — tr=0.6260 va=1.2467 acc=0.823
2026-04-27 06:00:48,127 INFO Regime epoch  9/50 — tr=0.6256 va=1.2459 acc=0.825
2026-04-27 06:00:48,689 INFO Regime epoch 10/50 — tr=0.6254 va=1.2459 acc=0.825 per_class={'TRENDING': 0.795, 'RANGING': 0.765, 'CONSOLIDATING': 0.834, 'VOLATILE': 0.911}
2026-04-27 06:00:49,207 INFO Regime epoch 11/50 — tr=0.6251 va=1.2401 acc=0.822
2026-04-27 06:00:49,732 INFO Regime epoch 12/50 — tr=0.6250 va=1.2415 acc=0.826
2026-04-27 06:00:50,235 INFO Regime epoch 13/50 — tr=0.6247 va=1.2374 acc=0.827
2026-04-27 06:00:50,731 INFO Regime epoch 14/50 — tr=0.6244 va=1.2415 acc=0.828
2026-04-27 06:00:51,287 INFO Regime epoch 15/50 — tr=0.6241 va=1.2414 acc=0.827 per_class={'TRENDING': 0.799, 'RANGING': 0.763, 'CONSOLIDATING': 0.843, 'VOLATILE': 0.906}
2026-04-27 06:00:51,792 INFO Regime epoch 16/50 — tr=0.6238 va=1.2421 acc=0.828
2026-04-27 06:00:52,307 INFO Regime epoch 17/50 — tr=0.6239 va=1.2401 acc=0.826
2026-04-27 06:00:52,826 INFO Regime epoch 18/50 — tr=0.6233 va=1.2357 acc=0.826
2026-04-27 06:00:53,338 INFO Regime epoch 19/50 — tr=0.6232 va=1.2424 acc=0.829
2026-04-27 06:00:53,871 INFO Regime epoch 20/50 — tr=0.6232 va=1.2373 acc=0.829 per_class={'TRENDING': 0.802, 'RANGING': 0.766, 'CONSOLIDATING': 0.846, 'VOLATILE': 0.904}
2026-04-27 06:00:54,369 INFO Regime epoch 21/50 — tr=0.6227 va=1.2368 acc=0.828
2026-04-27 06:00:54,863 INFO Regime epoch 22/50 — tr=0.6229 va=1.2331 acc=0.831
2026-04-27 06:00:55,357 INFO Regime epoch 23/50 — tr=0.6225 va=1.2396 acc=0.830
2026-04-27 06:00:55,891 INFO Regime epoch 24/50 — tr=0.6227 va=1.2338 acc=0.828
2026-04-27 06:00:56,417 INFO Regime epoch 25/50 — tr=0.6220 va=1.2354 acc=0.829 per_class={'TRENDING': 0.801, 'RANGING': 0.766, 'CONSOLIDATING': 0.851, 'VOLATILE': 0.906}
2026-04-27 06:00:56,914 INFO Regime epoch 26/50 — tr=0.6221 va=1.2333 acc=0.831
2026-04-27 06:00:57,437 INFO Regime epoch 27/50 — tr=0.6220 va=1.2351 acc=0.832
2026-04-27 06:00:57,947 INFO Regime epoch 28/50 — tr=0.6219 va=1.2333 acc=0.833
2026-04-27 06:00:58,456 INFO Regime epoch 29/50 — tr=0.6216 va=1.2374 acc=0.832
2026-04-27 06:00:59,004 INFO Regime epoch 30/50 — tr=0.6218 va=1.2356 acc=0.830 per_class={'TRENDING': 0.803, 'RANGING': 0.766, 'CONSOLIDATING': 0.845, 'VOLATILE': 0.911}
2026-04-27 06:00:59,512 INFO Regime epoch 31/50 — tr=0.6216 va=1.2337 acc=0.831
2026-04-27 06:01:00,003 INFO Regime epoch 32/50 — tr=0.6217 va=1.2320 acc=0.829
2026-04-27 06:01:00,487 INFO Regime epoch 33/50 — tr=0.6217 va=1.2349 acc=0.831
2026-04-27 06:01:00,993 INFO Regime epoch 34/50 — tr=0.6215 va=1.2340 acc=0.832
2026-04-27 06:01:01,531 INFO Regime epoch 35/50 — tr=0.6216 va=1.2357 acc=0.833 per_class={'TRENDING': 0.808, 'RANGING': 0.768, 'CONSOLIDATING': 0.849, 'VOLATILE': 0.907}
2026-04-27 06:01:02,071 INFO Regime epoch 36/50 — tr=0.6215 va=1.2322 acc=0.832
2026-04-27 06:01:02,604 INFO Regime epoch 37/50 — tr=0.6213 va=1.2313 acc=0.830
2026-04-27 06:01:03,096 INFO Regime epoch 38/50 — tr=0.6212 va=1.2293 acc=0.832
2026-04-27 06:01:03,617 INFO Regime epoch 39/50 — tr=0.6215 va=1.2312 acc=0.832
2026-04-27 06:01:04,177 INFO Regime epoch 40/50 — tr=0.6213 va=1.2295 acc=0.833 per_class={'TRENDING': 0.808, 'RANGING': 0.765, 'CONSOLIDATING': 0.859, 'VOLATILE': 0.902}
2026-04-27 06:01:04,713 INFO Regime epoch 41/50 — tr=0.6213 va=1.2355 acc=0.833
2026-04-27 06:01:05,218 INFO Regime epoch 42/50 — tr=0.6212 va=1.2326 acc=0.832
2026-04-27 06:01:05,760 INFO Regime epoch 43/50 — tr=0.6212 va=1.2290 acc=0.833
2026-04-27 06:01:06,282 INFO Regime epoch 44/50 — tr=0.6212 va=1.2299 acc=0.832
2026-04-27 06:01:06,824 INFO Regime epoch 45/50 — tr=0.6210 va=1.2325 acc=0.834 per_class={'TRENDING': 0.81, 'RANGING': 0.765, 'CONSOLIDATING': 0.86, 'VOLATILE': 0.901}
2026-04-27 06:01:07,336 INFO Regime epoch 46/50 — tr=0.6213 va=1.2312 acc=0.831
2026-04-27 06:01:07,843 INFO Regime epoch 47/50 — tr=0.6210 va=1.2338 acc=0.834
2026-04-27 06:01:08,339 INFO Regime epoch 48/50 — tr=0.6211 va=1.2342 acc=0.831
2026-04-27 06:01:08,855 INFO Regime epoch 49/50 — tr=0.6211 va=1.2317 acc=0.832
2026-04-27 06:01:09,406 INFO Regime epoch 50/50 — tr=0.6210 va=1.2290 acc=0.833 per_class={'TRENDING': 0.809, 'RANGING': 0.77, 'CONSOLIDATING': 0.853, 'VOLATILE': 0.904}
2026-04-27 06:01:09,446 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 06:01:09,447 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 06:01:09,449 INFO Regime phase LTF train: 26.1s
2026-04-27 06:01:09,574 INFO Regime LTF complete: acc=0.833, n=401471 per_class={'TRENDING': 0.809, 'RANGING': 0.77, 'CONSOLIDATING': 0.853, 'VOLATILE': 0.904}
2026-04-27 06:01:09,577 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 06:01:10,088 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 06:01:10,092 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 9.355291913830783, 'RANGING': 6.096303199751476, 'CONSOLIDATING': 5.598885793871866, 'VOLATILE': 6.771351107094442}
2026-04-27 06:01:10,099 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 29965, 'mean': 2.149062259083323e-05, 'mean_over_std': 0.011155601434349563}, 'RANGING': {'n': 19623, 'mean': -3.254493608872846e-06, 'mean_over_std': -0.001761415548292494}, 'CONSOLIDATING': {'n': 10050, 'mean': -3.039756158146958e-06, 'mean_over_std': -0.001792488091618151}, 'VOLATILE': {'n': 14985, 'mean': 1.8102764458593193e-05, 'mean_over_std': 0.006974696291378742}}
2026-04-27 06:01:10,100 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 26145, 'mean': 2.874435605138053e-05, 'mean_over_std': 0.01595450632691917}, 'RANGING': {'n': 7858, 'mean': 1.7829776785521727e-05, 'mean_over_std': 0.011851561804467463}, 'CONSOLIDATING': {'n': 8543, 'mean': 7.845231961250924e-06, 'mean_over_std': 0.004915839521549481}, 'VOLATILE': {'n': 13636, 'mean': 5.480114267306668e-06, 'mean_over_std': 0.0020378119917389375}}
2026-04-27 06:01:10,100 INFO Regime retrain total: 219.1s (504761 samples)
2026-04-27 06:01:10,102 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-27 06:01:10,102 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 06:01:10,102 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 06:01:10,102 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-27 06:01:10,103 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-27 06:01:10,103 INFO Retrain complete. Total wall-clock: 219.1s
  DONE  Retrain regime [train-split retrain]
  START Retrain quality [train-split retrain]
2026-04-27 06:01:11,402 INFO retrain environment: KAGGLE
2026-04-27 06:01:12,998 INFO Device: CUDA (2 GPU(s))
2026-04-27 06:01:13,008 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 06:01:13,008 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 06:01:13,008 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 06:01:13,008 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 06:01:13,008 INFO Retrain data split: train
2026-04-27 06:01:13,009 INFO === QualityScorer retrain ===
2026-04-27 06:01:13,154 INFO NumExpr defaulting to 4 threads.
2026-04-27 06:01:13,341 INFO QualityScorer: CUDA available — using GPU
2026-04-27 06:01:13,342 INFO QualityScorer: skipped 49 journal records outside allowed splits ['live', 'paper', 'production', 'train']
2026-04-27 06:01:13,344 INFO Quality phase label creation: 0.0s (0 trades)
2026-04-27 06:01:13,344 INFO Retrain complete. Total wall-clock: 0.3s
  WARN  Retrain quality failed (exit 1) — continuing
  START Retrain rl [train-split retrain]
2026-04-27 06:01:13,913 INFO retrain environment: KAGGLE
2026-04-27 06:01:15,554 INFO Device: CUDA (2 GPU(s))
2026-04-27 06:01:15,565 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 06:01:15,566 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 06:01:15,566 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 06:01:15,566 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 06:01:15,566 INFO Retrain data split: train
2026-04-27 06:01:15,567 INFO === RLAgent (PPO) retrain ===
2026-04-27 06:01:15,574 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260427_060115
2026-04-27 06:01:15,576 INFO RLAgent: skipped 49 journal records outside allowed splits ['live', 'paper', 'production', 'train']
2026-04-27 06:01:15,576 INFO RL phase episode loading: 0.0s (0 episodes)
2026-04-27 06:01:19.201552: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777269679.431068   84712 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777269679.503472   84712 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777269680.071405   84712 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777269680.071446   84712 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777269680.071449   84712 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777269680.071452   84712 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-27 06:01:34,231 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-27 06:01:36,287 INFO RLAgent: skipped 49 journal records outside allowed splits ['live', 'paper', 'production', 'train']
2026-04-27 06:01:36,287 WARNING RLAgent.retrain: only 0 episodes — skipping
2026-04-27 06:01:36,287 INFO RL phase PPO train: 20.7s | total: 20.7s
2026-04-27 06:01:36,288 INFO Retrain complete. Total wall-clock: 20.7s
  WARN  Retrain rl failed (exit 1) — continuing

=== Round 3: Backtest on last 3yr (post-retrain evaluation) ===
  START Round 3 - Post-retrain backtest (last 3yr)
2026-04-27 06:01:38,170 INFO === STEP 6: BACKTEST (round3) ===
2026-04-27 06:01:38,171 INFO BT_WINDOW=round3 — post-retrain eval: 2022-08-05 → 2025-08-05 (last 3yr)
2026-04-27 06:01:38,171 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-04-27 06:01:38,172 INFO Round 3 — running backtest: 2022-08-05 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-27 06:01:40,525 WARNING QualityScorer unavailable (weights missing or load failed)
2026-04-27 06:04:41,275 WARNING ml_trader: portfolio drawdown 8.1% after trade exit — halting all trading
2026-04-27 06:04:41,945 INFO Round 3 backtest — 66 trades | avg WR=30.3% | avg PF=1.21 | avg Sharpe=1.23
2026-04-27 06:04:41,945 INFO   ml_trader: 66 trades | WR=30.3% | PF=1.21 | Return=8.2% | DD=8.1% | Sharpe=1.23
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 66
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (66 rows)
  DONE  Round 3 - Post-retrain backtest (last 3yr)
  Saved Round 3 result → round3_summary.json

======================================================================
  BLIND BACKTEST PIPELINE COMPLETE
======================================================================
  Round 1 (val window)          trades=22  WR=13.6%  PF=0.521  Sharpe=-3.943
  Round 2 (blind test)          trades=27  WR=22.2%  PF=1.159  Sharpe=0.870
  Round 3 (last 3yr)            trades=66  WR=30.3%  PF=1.214  Sharpe=1.229


WARNING: GITHUB_TOKEN not set — skipping GitHub push
2026-04-27 06:04:42,185 INFO Round 3: wrote 66 journal entries (total in file: 115)