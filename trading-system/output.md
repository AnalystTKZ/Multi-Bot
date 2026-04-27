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
2026-04-27 07:15:49,313 INFO Loading feature-engineered data...
2026-04-27 07:15:49,945 INFO Loaded 221743 rows, 202 features
2026-04-27 07:15:49,946 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-27 07:15:49,948 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-27 07:15:49,949 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-27 07:15:49,949 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-27 07:15:49,949 INFO No leakage confirmed: train < val < test timestamps

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
2026-04-27 07:15:52,391 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-27 07:15:52,391 INFO --- Training regime ---
2026-04-27 07:15:52,391 INFO Running retrain --model regime
2026-04-27 07:15:52,583 INFO retrain environment: KAGGLE
2026-04-27 07:15:54,242 INFO Device: CUDA (2 GPU(s))
2026-04-27 07:15:54,254 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 07:15:54,254 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 07:15:54,254 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 07:15:54,258 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 07:15:54,258 INFO Retrain data split: train
2026-04-27 07:15:54,260 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-27 07:15:54,421 INFO NumExpr defaulting to 4 threads.
2026-04-27 07:15:54,638 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-27 07:15:54,638 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 07:15:54,638 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 07:15:54,639 INFO Regime phase macro_correlations: 0.0s
2026-04-27 07:15:54,639 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-27 07:15:54,676 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-27 07:15:54,677 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:15:54,706 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:15:54,721 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:15:54,746 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:15:54,762 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:15:54,787 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:15:54,803 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:15:54,828 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:15:54,845 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:15:54,868 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:15:54,884 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:15:54,906 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:15:54,920 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:15:54,939 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:15:54,955 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:15:54,977 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:15:54,993 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:15:55,015 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:15:55,030 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:15:55,053 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:15:55,070 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:15:55,108 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:15:56,320 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 07:16:19,961 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 07:16:19,963 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 24.9s
2026-04-27 07:16:19,963 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 07:16:30,237 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 07:16:30,238 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 10.3s
2026-04-27 07:16:30,238 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 07:16:38,409 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 07:16:38,413 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 8.2s
2026-04-27 07:16:38,413 INFO Regime phase GMM HTF total: 43.3s
2026-04-27 07:16:38,413 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 07:17:53,291 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 07:17:53,294 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 74.9s
2026-04-27 07:17:53,294 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 07:18:26,759 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 07:18:26,759 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 33.5s
2026-04-27 07:18:26,760 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 07:18:50,163 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 07:18:50,165 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 23.4s
2026-04-27 07:18:50,165 INFO Regime phase GMM LTF total: 131.8s
2026-04-27 07:18:50,272 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-27 07:18:50,274 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:18:50,275 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:18:50,276 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:18:50,277 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:18:50,278 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:18:50,279 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:18:50,280 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:18:50,281 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:18:50,282 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:18:50,283 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:18:50,285 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:18:50,413 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:50,457 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:50,458 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:50,458 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:50,467 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:50,468 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:18:50,897 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-27 07:18:50,898 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-27 07:18:51,075 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:51,108 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:51,109 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:51,109 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:51,118 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:51,120 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:18:51,504 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-27 07:18:51,506 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-27 07:18:51,695 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:51,730 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:51,731 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:51,731 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:51,740 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:51,741 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:18:52,140 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-27 07:18:52,142 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-27 07:18:52,326 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:52,361 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:52,362 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:52,362 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:52,371 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:52,372 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:18:52,752 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-27 07:18:52,753 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-27 07:18:52,944 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:52,980 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:52,981 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:52,981 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:52,990 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:52,990 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:18:53,382 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-27 07:18:53,383 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-27 07:18:53,552 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:53,586 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:53,586 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:53,587 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:53,595 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:53,596 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:18:53,982 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-27 07:18:53,983 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-27 07:18:54,147 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:18:54,177 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:18:54,178 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:18:54,178 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:18:54,187 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:18:54,188 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:18:54,577 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-27 07:18:54,579 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-27 07:18:54,763 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:54,797 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:54,798 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:54,799 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:54,806 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:54,807 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:18:55,199 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-27 07:18:55,200 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-27 07:18:55,379 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:55,415 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:55,416 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:55,417 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:55,425 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:55,426 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:18:55,843 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-27 07:18:55,845 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-27 07:18:56,018 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:56,054 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:56,054 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:56,055 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:56,063 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:18:56,064 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:18:56,456 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-27 07:18:56,458 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-27 07:18:56,733 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:18:56,795 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:18:56,796 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:18:56,797 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:18:56,808 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:18:56,809 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:18:57,628 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 07:18:57,630 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-27 07:18:57,794 INFO Regime phase HTF dataset build: 7.5s (103290 samples)
2026-04-27 07:18:57,795 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=38114 dropped=65176 classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728})
2026-04-27 07:18:57,804 INFO RegimeClassifier[mode=htf_bias]: 38114 samples, classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728}, device=cuda
2026-04-27 07:18:57,805 INFO RegimeClassifier: sample weights — mean=0.708  ambiguous(<0.4)=0.0%
2026-04-27 07:18:58,076 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-04-27 07:18:58,077 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-27 07:19:02,713 INFO Regime epoch  1/50 — tr=0.7329 va=2.1808 acc=0.319 per_class={'BIAS_UP': 0.017, 'BIAS_DOWN': 0.912, 'BIAS_NEUTRAL': 0.001}
2026-04-27 07:19:02,782 INFO Regime epoch  2/50 — tr=0.7269 va=2.0649 acc=0.350
2026-04-27 07:19:02,849 INFO Regime epoch  3/50 — tr=0.7172 va=2.0065 acc=0.419
2026-04-27 07:19:02,917 INFO Regime epoch  4/50 — tr=0.7044 va=1.9513 acc=0.485
2026-04-27 07:19:02,991 INFO Regime epoch  5/50 — tr=0.6854 va=1.8752 acc=0.566 per_class={'BIAS_UP': 0.398, 'BIAS_DOWN': 0.805, 'BIAS_NEUTRAL': 0.563}
2026-04-27 07:19:03,057 INFO Regime epoch  6/50 — tr=0.6612 va=1.7710 acc=0.673
2026-04-27 07:19:03,124 INFO Regime epoch  7/50 — tr=0.6357 va=1.6650 acc=0.773
2026-04-27 07:19:03,195 INFO Regime epoch  8/50 — tr=0.6106 va=1.5606 acc=0.852
2026-04-27 07:19:03,264 INFO Regime epoch  9/50 — tr=0.5878 va=1.4773 acc=0.894
2026-04-27 07:19:03,340 INFO Regime epoch 10/50 — tr=0.5693 va=1.4113 acc=0.918 per_class={'BIAS_UP': 0.97, 'BIAS_DOWN': 0.958, 'BIAS_NEUTRAL': 0.705}
2026-04-27 07:19:03,411 INFO Regime epoch 11/50 — tr=0.5539 va=1.3571 acc=0.927
2026-04-27 07:19:03,483 INFO Regime epoch 12/50 — tr=0.5417 va=1.3177 acc=0.934
2026-04-27 07:19:03,557 INFO Regime epoch 13/50 — tr=0.5314 va=1.2855 acc=0.940
2026-04-27 07:19:03,634 INFO Regime epoch 14/50 — tr=0.5238 va=1.2525 acc=0.945
2026-04-27 07:19:03,717 INFO Regime epoch 15/50 — tr=0.5187 va=1.2313 acc=0.948 per_class={'BIAS_UP': 0.996, 'BIAS_DOWN': 0.988, 'BIAS_NEUTRAL': 0.744}
2026-04-27 07:19:03,786 INFO Regime epoch 16/50 — tr=0.5145 va=1.2121 acc=0.951
2026-04-27 07:19:03,856 INFO Regime epoch 17/50 — tr=0.5095 va=1.1976 acc=0.953
2026-04-27 07:19:03,924 INFO Regime epoch 18/50 — tr=0.5069 va=1.1878 acc=0.956
2026-04-27 07:19:03,992 INFO Regime epoch 19/50 — tr=0.5037 va=1.1801 acc=0.957
2026-04-27 07:19:04,066 INFO Regime epoch 20/50 — tr=0.5011 va=1.1709 acc=0.958 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 0.995, 'BIAS_NEUTRAL': 0.776}
2026-04-27 07:19:04,134 INFO Regime epoch 21/50 — tr=0.4995 va=1.1639 acc=0.959
2026-04-27 07:19:04,208 INFO Regime epoch 22/50 — tr=0.4980 va=1.1553 acc=0.960
2026-04-27 07:19:04,280 INFO Regime epoch 23/50 — tr=0.4958 va=1.1522 acc=0.961
2026-04-27 07:19:04,348 INFO Regime epoch 24/50 — tr=0.4950 va=1.1484 acc=0.962
2026-04-27 07:19:04,424 INFO Regime epoch 25/50 — tr=0.4930 va=1.1449 acc=0.962 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 0.998, 'BIAS_NEUTRAL': 0.793}
2026-04-27 07:19:04,497 INFO Regime epoch 26/50 — tr=0.4920 va=1.1421 acc=0.962
2026-04-27 07:19:04,564 INFO Regime epoch 27/50 — tr=0.4911 va=1.1383 acc=0.963
2026-04-27 07:19:04,636 INFO Regime epoch 28/50 — tr=0.4896 va=1.1327 acc=0.964
2026-04-27 07:19:04,705 INFO Regime epoch 29/50 — tr=0.4888 va=1.1313 acc=0.964
2026-04-27 07:19:04,779 INFO Regime epoch 30/50 — tr=0.4876 va=1.1281 acc=0.965 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.806}
2026-04-27 07:19:04,847 INFO Regime epoch 31/50 — tr=0.4876 va=1.1253 acc=0.965
2026-04-27 07:19:04,913 INFO Regime epoch 32/50 — tr=0.4867 va=1.1229 acc=0.966
2026-04-27 07:19:04,986 INFO Regime epoch 33/50 — tr=0.4867 va=1.1218 acc=0.965
2026-04-27 07:19:05,054 INFO Regime epoch 34/50 — tr=0.4860 va=1.1185 acc=0.966
2026-04-27 07:19:05,128 INFO Regime epoch 35/50 — tr=0.4856 va=1.1173 acc=0.966 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.813}
2026-04-27 07:19:05,199 INFO Regime epoch 36/50 — tr=0.4847 va=1.1128 acc=0.967
2026-04-27 07:19:05,269 INFO Regime epoch 37/50 — tr=0.4843 va=1.1111 acc=0.967
2026-04-27 07:19:05,337 INFO Regime epoch 38/50 — tr=0.4844 va=1.1082 acc=0.968
2026-04-27 07:19:05,411 INFO Regime epoch 39/50 — tr=0.4841 va=1.1070 acc=0.969
2026-04-27 07:19:05,490 INFO Regime epoch 40/50 — tr=0.4836 va=1.1053 acc=0.970 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.832}
2026-04-27 07:19:05,561 INFO Regime epoch 41/50 — tr=0.4841 va=1.1045 acc=0.969
2026-04-27 07:19:05,631 INFO Regime epoch 42/50 — tr=0.4830 va=1.1060 acc=0.969
2026-04-27 07:19:05,712 INFO Regime epoch 43/50 — tr=0.4833 va=1.1037 acc=0.969
2026-04-27 07:19:05,801 INFO Regime epoch 44/50 — tr=0.4835 va=1.1075 acc=0.969
2026-04-27 07:19:05,875 INFO Regime epoch 45/50 — tr=0.4832 va=1.1060 acc=0.969 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.826}
2026-04-27 07:19:05,943 INFO Regime epoch 46/50 — tr=0.4830 va=1.1055 acc=0.969
2026-04-27 07:19:06,013 INFO Regime epoch 47/50 — tr=0.4827 va=1.1063 acc=0.969
2026-04-27 07:19:06,088 INFO Regime epoch 48/50 — tr=0.4835 va=1.1077 acc=0.968
2026-04-27 07:19:06,160 INFO Regime epoch 49/50 — tr=0.4834 va=1.1076 acc=0.968
2026-04-27 07:19:06,236 INFO Regime epoch 50/50 — tr=0.4828 va=1.1072 acc=0.969 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.825}
2026-04-27 07:19:06,246 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 07:19:06,247 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 07:19:06,247 INFO Regime phase HTF train: 8.5s
2026-04-27 07:19:06,376 INFO Regime HTF complete: acc=0.969, n=103290 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.83}
2026-04-27 07:19:06,378 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:19:06,538 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 07:19:06,546 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 11.41578947368421, 'BIAS_DOWN': 10.635761589403973, 'BIAS_NEUTRAL': 17.960468521229867}
2026-04-27 07:19:06,550 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 4338, 'mean': 0.00011110466582543088, 'mean_over_std': 0.027632581210502927}, 'BIAS_DOWN': {'n': 3212, 'mean': 6.063430222660252e-05, 'mean_over_std': 0.013898489590377157}, 'BIAS_NEUTRAL': {'n': 12266, 'mean': 1.2151554902814046e-05, 'mean_over_std': 0.0031957000513162357}}
2026-04-27 07:19:06,550 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 3651, 'mean': 0.0002852242108836959, 'mean_over_std': 0.07015017232491755}, 'BIAS_DOWN': {'n': 2644, 'mean': -0.00014900749618981248, 'mean_over_std': -0.03333037041298237}, 'BIAS_NEUTRAL': {'n': 1376, 'mean': 9.595797941249082e-05, 'mean_over_std': 0.02492411099521803}}
2026-04-27 07:19:06,554 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-27 07:19:06,555 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:06,557 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:06,559 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:06,560 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:06,562 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:06,564 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:06,565 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:06,566 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:06,568 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:06,570 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:06,573 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:19:06,585 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:06,589 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:06,589 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:06,590 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:06,590 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:06,592 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:07,231 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-27 07:19:07,234 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-27 07:19:07,373 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:07,375 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:07,376 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:07,376 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:07,377 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:07,379 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:07,997 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-27 07:19:08,000 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-27 07:19:08,135 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:08,137 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:08,138 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:08,139 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:08,139 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:08,141 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:08,759 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-27 07:19:08,762 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-27 07:19:08,904 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:08,907 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:08,908 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:08,908 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:08,908 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:08,910 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:09,523 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-27 07:19:09,526 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-27 07:19:09,659 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:09,663 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:09,664 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:09,664 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:09,665 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:09,667 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:10,284 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-27 07:19:10,287 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-27 07:19:10,423 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:10,425 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:10,426 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:10,427 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:10,427 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:10,429 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:11,035 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-27 07:19:11,038 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-27 07:19:11,173 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:19:11,175 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:19:11,176 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:19:11,176 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:19:11,176 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:19:11,178 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:11,793 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-27 07:19:11,795 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-27 07:19:11,935 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:11,938 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:11,938 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:11,939 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:11,939 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:11,941 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:12,563 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-27 07:19:12,566 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-27 07:19:12,701 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:12,703 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:12,704 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:12,705 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:12,705 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:12,707 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:13,335 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-27 07:19:13,338 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-27 07:19:13,477 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:13,480 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:13,480 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:13,481 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:13,481 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:13,483 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:14,089 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-27 07:19:14,092 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-27 07:19:14,243 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:19:14,246 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:19:14,248 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:19:14,248 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:19:14,249 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:19:14,252 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:19:15,594 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 07:19:15,601 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-27 07:19:15,918 INFO Regime phase LTF dataset build: 9.4s (401471 samples)
2026-04-27 07:19:15,921 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=299185 dropped=102286 classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127})
2026-04-27 07:19:15,987 INFO RegimeClassifier[mode=ltf_behaviour]: 299185 samples, classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127}, device=cuda
2026-04-27 07:19:15,988 INFO RegimeClassifier: sample weights — mean=0.693  ambiguous(<0.4)=0.0%
2026-04-27 07:19:15,990 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-04-27 07:19:15,991 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-27 07:19:16,570 INFO Regime epoch  1/50 — tr=0.8993 va=2.3731 acc=0.374 per_class={'TRENDING': 0.331, 'RANGING': 0.005, 'CONSOLIDATING': 0.211, 'VOLATILE': 0.774}
2026-04-27 07:19:17,090 INFO Regime epoch  2/50 — tr=0.8680 va=2.2640 acc=0.443
2026-04-27 07:19:17,649 INFO Regime epoch  3/50 — tr=0.8242 va=2.0999 acc=0.516
2026-04-27 07:19:18,165 INFO Regime epoch  4/50 — tr=0.7766 va=1.8932 acc=0.565
2026-04-27 07:19:18,704 INFO Regime epoch  5/50 — tr=0.7399 va=1.7026 acc=0.610 per_class={'TRENDING': 0.475, 'RANGING': 0.329, 'CONSOLIDATING': 0.731, 'VOLATILE': 0.954}
2026-04-27 07:19:19,200 INFO Regime epoch  6/50 — tr=0.7146 va=1.5898 acc=0.649
2026-04-27 07:19:19,704 INFO Regime epoch  7/50 — tr=0.6980 va=1.5321 acc=0.670
2026-04-27 07:19:20,207 INFO Regime epoch  8/50 — tr=0.6867 va=1.4945 acc=0.686
2026-04-27 07:19:20,726 INFO Regime epoch  9/50 — tr=0.6775 va=1.4711 acc=0.698
2026-04-27 07:19:21,267 INFO Regime epoch 10/50 — tr=0.6713 va=1.4494 acc=0.713 per_class={'TRENDING': 0.609, 'RANGING': 0.669, 'CONSOLIDATING': 0.712, 'VOLATILE': 0.937}
2026-04-27 07:19:21,786 INFO Regime epoch 11/50 — tr=0.6655 va=1.4394 acc=0.726
2026-04-27 07:19:22,308 INFO Regime epoch 12/50 — tr=0.6615 va=1.4121 acc=0.732
2026-04-27 07:19:22,808 INFO Regime epoch 13/50 — tr=0.6578 va=1.3991 acc=0.737
2026-04-27 07:19:23,326 INFO Regime epoch 14/50 — tr=0.6546 va=1.3902 acc=0.745
2026-04-27 07:19:23,882 INFO Regime epoch 15/50 — tr=0.6521 va=1.3753 acc=0.754 per_class={'TRENDING': 0.688, 'RANGING': 0.71, 'CONSOLIDATING': 0.731, 'VOLATILE': 0.918}
2026-04-27 07:19:24,397 INFO Regime epoch 16/50 — tr=0.6499 va=1.3658 acc=0.758
2026-04-27 07:19:24,892 INFO Regime epoch 17/50 — tr=0.6478 va=1.3555 acc=0.763
2026-04-27 07:19:25,394 INFO Regime epoch 18/50 — tr=0.6458 va=1.3444 acc=0.768
2026-04-27 07:19:25,927 INFO Regime epoch 19/50 — tr=0.6439 va=1.3394 acc=0.774
2026-04-27 07:19:26,496 INFO Regime epoch 20/50 — tr=0.6424 va=1.3318 acc=0.777 per_class={'TRENDING': 0.731, 'RANGING': 0.727, 'CONSOLIDATING': 0.749, 'VOLATILE': 0.912}
2026-04-27 07:19:26,990 INFO Regime epoch 21/50 — tr=0.6407 va=1.3222 acc=0.783
2026-04-27 07:19:27,516 INFO Regime epoch 22/50 — tr=0.6393 va=1.3160 acc=0.784
2026-04-27 07:19:28,037 INFO Regime epoch 23/50 — tr=0.6379 va=1.3134 acc=0.789
2026-04-27 07:19:28,544 INFO Regime epoch 24/50 — tr=0.6368 va=1.3060 acc=0.792
2026-04-27 07:19:29,092 INFO Regime epoch 25/50 — tr=0.6354 va=1.2983 acc=0.794 per_class={'TRENDING': 0.758, 'RANGING': 0.742, 'CONSOLIDATING': 0.766, 'VOLATILE': 0.91}
2026-04-27 07:19:29,607 INFO Regime epoch 26/50 — tr=0.6340 va=1.2985 acc=0.798
2026-04-27 07:19:30,111 INFO Regime epoch 27/50 — tr=0.6335 va=1.2903 acc=0.801
2026-04-27 07:19:30,631 INFO Regime epoch 28/50 — tr=0.6320 va=1.2876 acc=0.806
2026-04-27 07:19:31,123 INFO Regime epoch 29/50 — tr=0.6314 va=1.2766 acc=0.805
2026-04-27 07:19:31,655 INFO Regime epoch 30/50 — tr=0.6307 va=1.2775 acc=0.811 per_class={'TRENDING': 0.789, 'RANGING': 0.75, 'CONSOLIDATING': 0.792, 'VOLATILE': 0.902}
2026-04-27 07:19:32,177 INFO Regime epoch 31/50 — tr=0.6297 va=1.2718 acc=0.810
2026-04-27 07:19:32,688 INFO Regime epoch 32/50 — tr=0.6290 va=1.2727 acc=0.813
2026-04-27 07:19:33,225 INFO Regime epoch 33/50 — tr=0.6285 va=1.2667 acc=0.814
2026-04-27 07:19:33,722 INFO Regime epoch 34/50 — tr=0.6279 va=1.2701 acc=0.815
2026-04-27 07:19:34,258 INFO Regime epoch 35/50 — tr=0.6275 va=1.2644 acc=0.816 per_class={'TRENDING': 0.791, 'RANGING': 0.754, 'CONSOLIDATING': 0.809, 'VOLATILE': 0.906}
2026-04-27 07:19:34,747 INFO Regime epoch 36/50 — tr=0.6273 va=1.2623 acc=0.818
2026-04-27 07:19:35,230 INFO Regime epoch 37/50 — tr=0.6269 va=1.2627 acc=0.820
2026-04-27 07:19:35,768 INFO Regime epoch 38/50 — tr=0.6263 va=1.2602 acc=0.820
2026-04-27 07:19:36,282 INFO Regime epoch 39/50 — tr=0.6263 va=1.2622 acc=0.821
2026-04-27 07:19:36,832 INFO Regime epoch 40/50 — tr=0.6261 va=1.2608 acc=0.819 per_class={'TRENDING': 0.793, 'RANGING': 0.754, 'CONSOLIDATING': 0.822, 'VOLATILE': 0.905}
2026-04-27 07:19:37,330 INFO Regime epoch 41/50 — tr=0.6261 va=1.2562 acc=0.823
2026-04-27 07:19:37,856 INFO Regime epoch 42/50 — tr=0.6261 va=1.2588 acc=0.823
2026-04-27 07:19:38,373 INFO Regime epoch 43/50 — tr=0.6256 va=1.2629 acc=0.824
2026-04-27 07:19:38,876 INFO Regime epoch 44/50 — tr=0.6254 va=1.2558 acc=0.822
2026-04-27 07:19:39,441 INFO Regime epoch 45/50 — tr=0.6258 va=1.2593 acc=0.825 per_class={'TRENDING': 0.805, 'RANGING': 0.752, 'CONSOLIDATING': 0.827, 'VOLATILE': 0.904}
2026-04-27 07:19:39,937 INFO Regime epoch 46/50 — tr=0.6254 va=1.2597 acc=0.826
2026-04-27 07:19:40,445 INFO Regime epoch 47/50 — tr=0.6250 va=1.2567 acc=0.822
2026-04-27 07:19:40,961 INFO Regime epoch 48/50 — tr=0.6252 va=1.2572 acc=0.825
2026-04-27 07:19:41,463 INFO Regime epoch 49/50 — tr=0.6256 va=1.2616 acc=0.825
2026-04-27 07:19:42,022 INFO Regime epoch 50/50 — tr=0.6256 va=1.2556 acc=0.822 per_class={'TRENDING': 0.798, 'RANGING': 0.756, 'CONSOLIDATING': 0.829, 'VOLATILE': 0.902}
2026-04-27 07:19:42,065 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 07:19:42,065 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 07:19:42,067 INFO Regime phase LTF train: 26.1s
2026-04-27 07:19:42,203 INFO Regime LTF complete: acc=0.822, n=401471 per_class={'TRENDING': 0.798, 'RANGING': 0.756, 'CONSOLIDATING': 0.829, 'VOLATILE': 0.902}
2026-04-27 07:19:42,206 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:19:42,744 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 07:19:42,749 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 9.355291913830783, 'RANGING': 6.096303199751476, 'CONSOLIDATING': 5.598885793871866, 'VOLATILE': 6.771351107094442}
2026-04-27 07:19:42,758 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 29965, 'mean': 2.149062259083323e-05, 'mean_over_std': 0.011155601434349563}, 'RANGING': {'n': 19623, 'mean': -3.254493608872846e-06, 'mean_over_std': -0.001761415548292494}, 'CONSOLIDATING': {'n': 10050, 'mean': -3.039756158146958e-06, 'mean_over_std': -0.001792488091618151}, 'VOLATILE': {'n': 14985, 'mean': 1.8102764458593193e-05, 'mean_over_std': 0.006974696291378742}}
2026-04-27 07:19:42,759 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 26145, 'mean': 2.874435605138053e-05, 'mean_over_std': 0.01595450632691917}, 'RANGING': {'n': 7858, 'mean': 1.7829776785521727e-05, 'mean_over_std': 0.011851561804467463}, 'CONSOLIDATING': {'n': 8543, 'mean': 7.845231961250924e-06, 'mean_over_std': 0.004915839521549481}, 'VOLATILE': {'n': 13636, 'mean': 5.480114267306668e-06, 'mean_over_std': 0.0020378119917389375}}
2026-04-27 07:19:42,762 INFO Regime retrain total: 228.5s (504761 samples)
2026-04-27 07:19:42,779 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-27 07:19:42,779 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 07:19:42,779 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 07:19:42,779 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-27 07:19:42,779 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-27 07:19:42,780 INFO Retrain complete. Total wall-clock: 228.5s
2026-04-27 07:19:45,002 INFO Model regime: SUCCESS
2026-04-27 07:19:45,003 INFO --- Training gru ---
2026-04-27 07:19:45,003 INFO Running retrain --model gru
2026-04-27 07:19:45,331 INFO retrain environment: KAGGLE
2026-04-27 07:19:47,028 INFO Device: CUDA (2 GPU(s))
2026-04-27 07:19:47,039 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 07:19:47,039 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 07:19:47,039 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 07:19:47,040 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 07:19:47,040 INFO Retrain data split: train
2026-04-27 07:19:47,041 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-27 07:19:47,189 INFO NumExpr defaulting to 4 threads.
2026-04-27 07:19:47,388 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-27 07:19:47,388 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 07:19:47,388 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 07:19:47,388 INFO GRU phase macro_correlations: 0.0s
2026-04-27 07:19:47,388 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-27 07:19:47,389 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260427_071947
2026-04-27 07:19:47,391 WARNING WeightsManifest: no manifest at /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json — assuming compatible (run a full retrain to generate one)
2026-04-27 07:19:47,536 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:47,556 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:47,571 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:47,578 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:47,579 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-27 07:19:47,579 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 07:19:47,579 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 07:19:47,580 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-27 07:19:47,581 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:47,666 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-27 07:19:47,668 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:47,919 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-27 07:19:47,948 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:48,224 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:48,352 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:48,450 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:48,644 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:48,663 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:48,678 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:48,684 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:48,685 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:48,763 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-27 07:19:48,765 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:49,009 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-27 07:19:49,024 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:49,293 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:49,421 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:49,517 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:49,703 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:49,723 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:49,737 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:49,745 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:49,746 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:49,826 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-27 07:19:49,828 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:50,062 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-27 07:19:50,077 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:50,347 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:50,477 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:50,574 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:50,758 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:50,780 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:50,795 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:50,803 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:50,804 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:50,878 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-27 07:19:50,880 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:51,123 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-27 07:19:51,146 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:51,419 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:51,549 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:51,649 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:51,834 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:51,856 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:51,873 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:51,880 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:51,882 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:51,968 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-27 07:19:51,970 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:52,224 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-27 07:19:52,239 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:52,511 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:52,639 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:52,731 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:52,917 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:52,936 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:52,951 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:52,958 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:52,959 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:53,037 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-27 07:19:53,039 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:53,281 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-27 07:19:53,297 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:53,569 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:53,697 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:53,798 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:53,960 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:19:53,978 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:19:53,992 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:19:53,998 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:19:53,999 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:54,077 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-27 07:19:54,078 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:54,324 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-27 07:19:54,337 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:54,600 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:54,727 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:54,825 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:55,004 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:55,023 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:55,038 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:55,046 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:55,046 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:55,124 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-27 07:19:55,126 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:55,369 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-27 07:19:55,388 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:55,650 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:55,804 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:55,898 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:56,086 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:56,105 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:56,121 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:56,128 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:56,129 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:56,205 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-27 07:19:56,206 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:56,445 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-27 07:19:56,460 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:56,727 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:56,857 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:56,953 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:57,145 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:57,166 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:57,181 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:57,188 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:19:57,189 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:57,269 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-27 07:19:57,271 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:57,516 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-27 07:19:57,531 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:57,812 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:57,938 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:58,037 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:19:58,328 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:19:58,353 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:19:58,369 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:19:58,379 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:19:58,380 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:19:58,541 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 07:19:58,544 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:19:59,072 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 07:19:59,120 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:19:59,636 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:19:59,833 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:19:59,969 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:20:00,088 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-27 07:20:00,355 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-04-27 07:20:00,355 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-04-27 07:20:00,356 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-27 07:20:00,356 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-27 07:20:49,518 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-27 07:20:49,518 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-27 07:20:50,844 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-27 07:20:55,052 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=18, min_epochs=22)
2026-04-27 07:21:09,203 INFO train_multi TF=ALL epoch 1/50 train=0.8584 val=0.8518
2026-04-27 07:21:09,212 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:21:09,212 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:21:09,212 INFO train_multi TF=ALL: new best val=0.8518 — saved
2026-04-27 07:21:21,326 INFO train_multi TF=ALL epoch 2/50 train=0.8381 val=0.8055
2026-04-27 07:21:21,330 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:21:21,331 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:21:21,331 INFO train_multi TF=ALL: new best val=0.8055 — saved
2026-04-27 07:21:33,462 INFO train_multi TF=ALL epoch 3/50 train=0.7285 val=0.6879
2026-04-27 07:21:33,466 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:21:33,466 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:21:33,466 INFO train_multi TF=ALL: new best val=0.6879 — saved
2026-04-27 07:21:45,505 INFO train_multi TF=ALL epoch 4/50 train=0.6915 val=0.6878
2026-04-27 07:21:45,509 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:21:45,509 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:21:45,509 INFO train_multi TF=ALL: new best val=0.6878 — saved
2026-04-27 07:21:57,541 INFO train_multi TF=ALL epoch 5/50 train=0.6900 val=0.6883
2026-04-27 07:22:09,677 INFO train_multi TF=ALL epoch 6/50 train=0.6896 val=0.6882
2026-04-27 07:22:21,838 INFO train_multi TF=ALL epoch 7/50 train=0.6894 val=0.6883
2026-04-27 07:22:34,013 INFO train_multi TF=ALL epoch 8/50 train=0.6892 val=0.6878
2026-04-27 07:22:34,018 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:22:34,018 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:22:34,018 INFO train_multi TF=ALL: new best val=0.6878 — saved
2026-04-27 07:22:46,030 INFO train_multi TF=ALL epoch 9/50 train=0.6890 val=0.6876
2026-04-27 07:22:46,035 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:22:46,035 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:22:46,035 INFO train_multi TF=ALL: new best val=0.6876 — saved
2026-04-27 07:22:58,006 INFO train_multi TF=ALL epoch 10/50 train=0.6881 val=0.6867
2026-04-27 07:22:58,011 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:22:58,011 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:22:58,011 INFO train_multi TF=ALL: new best val=0.6867 — saved
2026-04-27 07:23:10,262 INFO train_multi TF=ALL epoch 11/50 train=0.6857 val=0.6839
2026-04-27 07:23:10,266 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:23:10,267 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:23:10,267 INFO train_multi TF=ALL: new best val=0.6839 — saved
2026-04-27 07:23:22,413 INFO train_multi TF=ALL epoch 12/50 train=0.6797 val=0.6759
2026-04-27 07:23:22,417 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:23:22,418 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:23:22,418 INFO train_multi TF=ALL: new best val=0.6759 — saved
2026-04-27 07:23:34,542 INFO train_multi TF=ALL epoch 13/50 train=0.6709 val=0.6647
2026-04-27 07:23:34,546 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:23:34,546 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:23:34,546 INFO train_multi TF=ALL: new best val=0.6647 — saved
2026-04-27 07:23:46,659 INFO train_multi TF=ALL epoch 14/50 train=0.6580 val=0.6532
2026-04-27 07:23:46,663 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:23:46,663 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:23:46,663 INFO train_multi TF=ALL: new best val=0.6532 — saved
2026-04-27 07:23:58,724 INFO train_multi TF=ALL epoch 15/50 train=0.6474 val=0.6376
2026-04-27 07:23:58,729 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:23:58,729 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:23:58,729 INFO train_multi TF=ALL: new best val=0.6376 — saved
2026-04-27 07:24:10,901 INFO train_multi TF=ALL epoch 16/50 train=0.6390 val=0.6315
2026-04-27 07:24:10,905 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:24:10,905 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:24:10,905 INFO train_multi TF=ALL: new best val=0.6315 — saved
2026-04-27 07:24:23,113 INFO train_multi TF=ALL epoch 17/50 train=0.6334 val=0.6284
2026-04-27 07:24:23,118 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:24:23,119 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:24:23,119 INFO train_multi TF=ALL: new best val=0.6284 — saved
2026-04-27 07:24:35,250 INFO train_multi TF=ALL epoch 18/50 train=0.6288 val=0.6246
2026-04-27 07:24:35,255 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:24:35,255 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:24:35,255 INFO train_multi TF=ALL: new best val=0.6246 — saved
2026-04-27 07:24:47,437 INFO train_multi TF=ALL epoch 19/50 train=0.6257 val=0.6240
2026-04-27 07:24:47,442 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:24:47,442 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:24:47,442 INFO train_multi TF=ALL: new best val=0.6240 — saved
2026-04-27 07:24:59,471 INFO train_multi TF=ALL epoch 20/50 train=0.6225 val=0.6188
2026-04-27 07:24:59,476 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:24:59,476 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:24:59,476 INFO train_multi TF=ALL: new best val=0.6188 — saved
2026-04-27 07:25:11,701 INFO train_multi TF=ALL epoch 21/50 train=0.6195 val=0.6195
2026-04-27 07:25:23,943 INFO train_multi TF=ALL epoch 22/50 train=0.6175 val=0.6213
2026-04-27 07:25:36,113 INFO train_multi TF=ALL epoch 23/50 train=0.6153 val=0.6180
2026-04-27 07:25:36,117 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:25:36,117 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:25:36,117 INFO train_multi TF=ALL: new best val=0.6180 — saved
2026-04-27 07:25:48,278 INFO train_multi TF=ALL epoch 24/50 train=0.6133 val=0.6180
2026-04-27 07:25:48,282 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:25:48,283 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:25:48,283 INFO train_multi TF=ALL: new best val=0.6180 — saved
2026-04-27 07:26:00,429 INFO train_multi TF=ALL epoch 25/50 train=0.6110 val=0.6137
2026-04-27 07:26:00,434 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:26:00,434 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:26:00,434 INFO train_multi TF=ALL: new best val=0.6137 — saved
2026-04-27 07:26:12,567 INFO train_multi TF=ALL epoch 26/50 train=0.6092 val=0.6147
2026-04-27 07:26:24,662 INFO train_multi TF=ALL epoch 27/50 train=0.6072 val=0.6133
2026-04-27 07:26:24,666 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:26:24,666 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:26:24,666 INFO train_multi TF=ALL: new best val=0.6133 — saved
2026-04-27 07:26:36,881 INFO train_multi TF=ALL epoch 28/50 train=0.6055 val=0.6115
2026-04-27 07:26:36,886 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:26:36,886 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:26:36,886 INFO train_multi TF=ALL: new best val=0.6115 — saved
2026-04-27 07:26:48,965 INFO train_multi TF=ALL epoch 29/50 train=0.6037 val=0.6141
2026-04-27 07:27:01,085 INFO train_multi TF=ALL epoch 30/50 train=0.6019 val=0.6130
2026-04-27 07:27:13,274 INFO train_multi TF=ALL epoch 31/50 train=0.6007 val=0.6146
2026-04-27 07:27:25,373 INFO train_multi TF=ALL epoch 32/50 train=0.5997 val=0.6110
2026-04-27 07:27:25,377 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:27:25,377 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:27:25,377 INFO train_multi TF=ALL: new best val=0.6110 — saved
2026-04-27 07:27:37,563 INFO train_multi TF=ALL epoch 33/50 train=0.5979 val=0.6170
2026-04-27 07:27:49,587 INFO train_multi TF=ALL epoch 34/50 train=0.5966 val=0.6124
2026-04-27 07:28:01,647 INFO train_multi TF=ALL epoch 35/50 train=0.5954 val=0.6141
2026-04-27 07:28:13,781 INFO train_multi TF=ALL epoch 36/50 train=0.5940 val=0.6131
2026-04-27 07:28:25,977 INFO train_multi TF=ALL epoch 37/50 train=0.5927 val=0.6153
2026-04-27 07:28:38,049 INFO train_multi TF=ALL epoch 38/50 train=0.5915 val=0.6123
2026-04-27 07:28:50,040 INFO train_multi TF=ALL epoch 39/50 train=0.5905 val=0.6149
2026-04-27 07:29:02,091 INFO train_multi TF=ALL epoch 40/50 train=0.5891 val=0.6129
2026-04-27 07:29:14,176 INFO train_multi TF=ALL epoch 41/50 train=0.5878 val=0.6141
2026-04-27 07:29:26,277 INFO train_multi TF=ALL epoch 42/50 train=0.5864 val=0.6154
2026-04-27 07:29:38,256 INFO train_multi TF=ALL epoch 43/50 train=0.5852 val=0.6204
2026-04-27 07:29:50,208 INFO train_multi TF=ALL epoch 44/50 train=0.5847 val=0.6144
2026-04-27 07:30:02,306 INFO train_multi TF=ALL epoch 45/50 train=0.5828 val=0.6108
2026-04-27 07:30:02,311 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:30:02,311 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:30:02,311 INFO train_multi TF=ALL: new best val=0.6108 — saved
2026-04-27 07:30:14,450 INFO train_multi TF=ALL epoch 46/50 train=0.5816 val=0.6141
2026-04-27 07:30:26,483 INFO train_multi TF=ALL epoch 47/50 train=0.5810 val=0.6132
2026-04-27 07:30:38,625 INFO train_multi TF=ALL epoch 48/50 train=0.5792 val=0.6125
2026-04-27 07:30:50,619 INFO train_multi TF=ALL epoch 49/50 train=0.5786 val=0.6117
2026-04-27 07:31:02,650 INFO train_multi TF=ALL epoch 50/50 train=0.5772 val=0.6159
2026-04-27 07:31:02,802 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-27 07:31:02,803 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-27 07:31:02,803 INFO Retrain complete. Total wall-clock: 675.8s
2026-04-27 07:31:04,810 INFO Model gru: SUCCESS
2026-04-27 07:31:04,810 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:31:04,810 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 07:31:04,811 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 07:31:04,811 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-04-27 07:31:04,811 INFO   [DEFERRED] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip (expected after Round 1)
2026-04-27 07:31:04,811 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-04-27 07:31:04,811 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer', 'rl_ppo']
2026-04-27 07:31:04,812 INFO Saved 5 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  START Round 1 - Backtest (val)
2026-04-27 07:31:05,603 INFO === STEP 6: BACKTEST (round1) ===
2026-04-27 07:31:05,604 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-04-27 07:31:05,604 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-27 07:31:05,604 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
2026-04-27 07:31:08,055 WARNING QualityScorer unavailable (weights missing or load failed)
2026-04-27 07:33:18,472 WARNING ml_trader: portfolio drawdown 8.2% after trade exit — halting all trading

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260427_073107.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)               20  15.0%   0.46   -6.4% 15.0%  0.0%   8.2%   -5.04
  gate_diagnostics: bars=4278 no_signal=18 quality_block=0 session_skip=1564 density=0 pm_reject=13 daily_skip=2647 cooldown=16 daily_halt_events=4 enforce_daily_halt=True
  no_signal_reasons: weak_gru_direction=11, htf_bias_conflict=6, trend_pullback_conflict=1

Calibration Summary:
  all          [OK] Too few populated bins for calibration check
  ml_trader    [OK] Too few populated bins for calibration check
2026-04-27 07:33:19,133 INFO Round 1 backtest — 20 trades | avg WR=15.0% | avg PF=0.46 | avg Sharpe=-5.04
2026-04-27 07:33:19,133 INFO   ml_trader: 20 trades | WR=15.0% | PF=0.46 | Return=-6.4% | DD=8.2% | Sharpe=-5.04
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 20
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (20 rows)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  AUDUSD          4 trades     4 days   1.00/day
  EURGBP          2 trades     2 days   1.00/day
  EURJPY          2 trades     2 days   1.00/day
  EURUSD          3 trades     3 days   1.00/day
  GBPJPY          2 trades     2 days   1.00/day
  NZDUSD          1 trades     1 days   1.00/day
  USDCAD          2 trades     2 days   1.00/day
  USDCHF          1 trades     1 days   1.00/day
  USDJPY          2 trades     2 days   1.00/day
  XAUUSD          1 trades     1 days   1.00/day
  ✓  All symbols within normal range.

──────────────────────────────────────────────────────────────
CHECK 2: REGIME DISTRIBUTION  (% of trades, WR, avgEV)
──────────────────────────────────────────────────────────────
  BIAS_DOWN             6 trades   30.0%  WR=16.7%  avgEV=0.000
  BIAS_NEUTRAL          9 trades   45.0%  WR=11.1%  avgEV=0.000
  BIAS_UP               5 trades   25.0%  WR=20.0%  avgEV=0.000
  ⚠  Regimes never traded: ['CONSOLIDATING', 'RANGING', 'TRENDING', 'VOLATILE']

──────────────────────────────────────────────────────────────
CHECK 3: EV PREDICTED vs REALIZED RR
────────────────────────────────────────  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 20 entries

=== Round 1 → Quality + RL retrain skipped (validation journal is not train data) ===

=== Round 2: BLIND backtest on test window (unseen 2yr) ===
  START Round 2 - Blind backtest (test)
2026-04-27 07:33:19,408 INFO Round 1: wrote 20 journal entries (total in file: 20)
2026-04-27 07:33:20,108 INFO === STEP 6: BACKTEST (round2) ===
2026-04-27 07:33:20,109 INFO BT_WINDOW=round2 — BLIND backtest: 2023-08-07 → 2025-08-05 (test set)
2026-04-27 07:33:20,109 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-27 07:33:20,109 INFO Round 2 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-27 07:33:22,543 WARNING QualityScorer unavailable (weights missing or load failed)
2026-04-27 07:35:36,944 WARNING ml_trader: portfolio drawdown 8.6% after trade exit — halting all trading
2026-04-27 07:35:37,611 INFO Round 2 backtest — 48 trades | avg WR=29.2% | avg PF=1.14 | avg Sharpe=0.85
2026-04-27 07:35:37,611 INFO   ml_trader: 48 trades | WR=29.2% | PF=1.14 | Return=3.5% | DD=8.6% | Sharpe=0.85
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 48
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (48 rows)
2026-04-27 07:35:37,896 INFO Round 2: wrote 48 journal entries (total in file: 68)
  DONE  Round 2 - Blind backtest (test)
  Saved Round 2 result → round2_summary.json
  Journal after Round 2: 68 entries

=== Round 2 → Quality + RL retrain skipped (blind-test journal is not train data) ===

=== Round 3: Incremental retrain on train split only ===
  START Retrain gru [train-split retrain]
2026-04-27 07:35:38,303 INFO retrain environment: KAGGLE
2026-04-27 07:35:39,949 INFO Device: CUDA (2 GPU(s))
2026-04-27 07:35:39,959 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 07:35:39,959 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 07:35:39,959 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 07:35:39,960 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 07:35:39,960 INFO Retrain data split: train
2026-04-27 07:35:39,961 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-27 07:35:40,115 INFO NumExpr defaulting to 4 threads.
2026-04-27 07:35:40,316 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-27 07:35:40,316 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 07:35:40,316 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 07:35:40,552 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-27 07:35:40,552 INFO GRU phase macro_correlations: 0.0s
2026-04-27 07:35:40,552 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-27 07:35:40,554 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260427_073540
2026-04-27 07:35:40,558 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-04-27 07:35:40,706 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:40,727 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:40,742 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:40,750 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:40,751 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-27 07:35:40,752 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 07:35:40,752 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 07:35:40,752 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-27 07:35:40,753 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:40,841 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-27 07:35:40,843 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:41,095 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-27 07:35:41,127 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:41,409 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:41,534 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:41,635 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:41,839 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:41,860 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:41,876 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:41,884 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:41,885 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:41,977 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-27 07:35:41,979 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:42,239 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-27 07:35:42,255 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:42,528 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:42,659 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:42,758 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:42,964 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:42,985 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:43,000 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:43,008 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:43,009 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:43,094 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-27 07:35:43,095 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:43,361 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-27 07:35:43,380 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:43,670 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:43,800 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:43,902 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:44,106 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:44,125 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:44,141 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:44,148 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:44,149 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:44,237 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-27 07:35:44,239 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:44,488 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-27 07:35:44,512 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:44,789 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:44,921 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:45,027 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:45,226 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:45,249 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:45,265 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:45,273 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:45,274 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:45,364 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-27 07:35:45,367 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:45,641 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-27 07:35:45,656 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:45,953 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:46,084 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:46,185 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:46,371 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:46,391 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:46,405 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:46,413 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:46,413 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:46,500 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-27 07:35:46,501 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:46,762 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-27 07:35:46,777 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:47,049 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:47,181 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:47,281 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:47,450 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:35:47,468 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:35:47,482 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:35:47,489 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:35:47,490 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:47,575 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-27 07:35:47,576 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:47,831 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-27 07:35:47,843 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:48,112 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:48,249 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:48,348 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:48,531 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:48,551 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:48,566 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:48,573 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:48,574 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:48,661 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-27 07:35:48,663 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:48,913 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-27 07:35:48,931 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:49,201 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:49,338 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:49,440 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:49,633 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:49,651 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:49,665 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:49,672 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:49,673 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:49,754 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-27 07:35:49,756 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:50,006 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-27 07:35:50,022 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:50,299 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:50,432 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:50,533 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:50,722 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:50,743 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:50,758 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:50,765 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:35:50,766 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:50,848 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-27 07:35:50,849 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:51,105 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-27 07:35:51,120 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:51,400 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:51,534 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:51,639 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:35:51,936 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:35:51,964 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:35:51,982 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:35:51,994 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:35:51,995 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:35:52,179 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 07:35:52,183 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:35:52,721 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 07:35:52,771 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:35:53,291 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:35:53,494 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:35:53,625 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:35:53,746 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-27 07:35:53,747 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-27 07:35:53,747 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-27 07:36:43,960 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-27 07:36:43,961 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-27 07:36:45,299 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-27 07:36:49,536 INFO train_multi TF=ALL: warm-start detected — using CosineAnnealingLR (lr=3e-05, patience=12)
2026-04-27 07:37:03,301 INFO train_multi TF=ALL epoch 1/50 train=0.5802 val=0.6119
2026-04-27 07:37:03,306 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:37:03,306 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:37:03,306 INFO train_multi TF=ALL: new best val=0.6119 — saved
2026-04-27 07:37:15,328 INFO train_multi TF=ALL epoch 2/50 train=0.5798 val=0.6118
2026-04-27 07:37:15,332 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-27 07:37:15,332 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-27 07:37:15,332 INFO train_multi TF=ALL: new best val=0.6118 — saved
2026-04-27 07:37:27,498 INFO train_multi TF=ALL epoch 3/50 train=0.5791 val=0.6125
2026-04-27 07:37:39,716 INFO train_multi TF=ALL epoch 4/50 train=0.5793 val=0.6124
2026-04-27 07:37:51,803 INFO train_multi TF=ALL epoch 5/50 train=0.5791 val=0.6135
2026-04-27 07:38:04,004 INFO train_multi TF=ALL epoch 6/50 train=0.5789 val=0.6126
2026-04-27 07:38:16,113 INFO train_multi TF=ALL epoch 7/50 train=0.5789 val=0.6131
2026-04-27 07:38:28,147 INFO train_multi TF=ALL epoch 8/50 train=0.5787 val=0.6127
2026-04-27 07:38:40,409 INFO train_multi TF=ALL epoch 9/50 train=0.5787 val=0.6134
2026-04-27 07:38:52,661 INFO train_multi TF=ALL epoch 10/50 train=0.5784 val=0.6127
2026-04-27 07:39:04,885 INFO train_multi TF=ALL epoch 11/50 train=0.5783 val=0.6126
2026-04-27 07:39:17,085 INFO train_multi TF=ALL epoch 12/50 train=0.5782 val=0.6134
2026-04-27 07:39:29,221 INFO train_multi TF=ALL epoch 13/50 train=0.5781 val=0.6137
2026-04-27 07:39:41,310 INFO train_multi TF=ALL epoch 14/50 train=0.5779 val=0.6126
2026-04-27 07:39:41,310 INFO train_multi TF=ALL early stop at epoch 14
2026-04-27 07:39:41,469 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-27 07:39:41,469 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-27 07:39:41,469 INFO Retrain complete. Total wall-clock: 241.5s
  DONE  Retrain gru [train-split retrain]
  START Retrain regime [train-split retrain]
2026-04-27 07:39:43,766 INFO retrain environment: KAGGLE
2026-04-27 07:39:45,407 INFO Device: CUDA (2 GPU(s))
2026-04-27 07:39:45,416 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 07:39:45,416 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 07:39:45,416 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 07:39:45,417 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 07:39:45,417 INFO Retrain data split: train
2026-04-27 07:39:45,418 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-27 07:39:45,574 INFO NumExpr defaulting to 4 threads.
2026-04-27 07:39:45,804 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-27 07:39:45,804 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 07:39:45,804 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 07:39:45,804 INFO Regime phase macro_correlations: 0.0s
2026-04-27 07:39:45,804 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-27 07:39:45,843 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-27 07:39:45,844 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:39:45,873 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:39:45,888 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:39:45,912 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:39:45,927 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:39:45,951 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:39:45,967 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:39:45,993 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:39:46,008 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:39:46,032 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:39:46,047 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:39:46,069 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:39:46,083 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:39:46,102 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:39:46,116 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:39:46,138 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:39:46,153 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:39:46,176 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:39:46,191 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:39:46,214 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:39:46,232 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:39:46,275 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:39:47,085 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 07:40:11,561 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 07:40:11,562 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 25.3s
2026-04-27 07:40:11,562 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 07:40:22,580 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 07:40:22,584 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 11.0s
2026-04-27 07:40:22,585 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-27 07:40:30,796 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-27 07:40:30,800 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 8.2s
2026-04-27 07:40:30,800 INFO Regime phase GMM HTF total: 44.5s
2026-04-27 07:40:30,800 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 07:41:44,301 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 07:41:44,304 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 73.5s
2026-04-27 07:41:44,304 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 07:42:17,102 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 07:42:17,106 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 32.8s
2026-04-27 07:42:17,106 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-27 07:42:40,452 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-27 07:42:40,453 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 23.3s
2026-04-27 07:42:40,453 INFO Regime phase GMM LTF total: 129.7s
2026-04-27 07:42:40,561 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-27 07:42:40,562 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:40,564 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:40,565 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:40,566 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:40,567 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:40,569 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:40,570 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:40,571 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:40,572 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:40,573 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:40,575 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:42:40,703 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:40,748 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:40,749 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:40,749 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:40,760 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:40,761 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:41,182 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-27 07:42:41,184 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-27 07:42:41,363 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:41,402 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:41,403 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:41,403 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:41,412 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:41,413 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:41,801 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-27 07:42:41,803 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-27 07:42:42,008 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:42,045 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:42,046 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:42,047 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:42,055 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:42,057 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:42,459 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-27 07:42:42,460 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-27 07:42:42,645 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:42,680 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:42,681 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:42,681 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:42,690 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:42,691 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:43,083 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-27 07:42:43,084 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-27 07:42:43,259 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:43,295 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:43,296 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:43,296 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:43,304 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:43,306 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:43,687 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-27 07:42:43,689 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-27 07:42:43,861 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:43,896 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:43,897 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:43,897 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:43,905 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:43,906 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:44,306 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-27 07:42:44,307 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-27 07:42:44,463 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:42:44,491 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:42:44,492 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:42:44,492 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:42:44,499 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:42:44,500 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:44,882 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-27 07:42:44,883 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-27 07:42:45,055 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:45,090 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:45,091 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:45,092 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:45,100 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:45,101 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:45,501 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-27 07:42:45,502 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-27 07:42:45,683 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:45,727 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:45,728 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:45,729 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:45,739 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:45,740 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:46,140 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-27 07:42:46,142 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-27 07:42:46,316 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:46,352 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:46,353 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:46,353 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:46,361 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:46,362 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:46,753 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-27 07:42:46,754 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-27 07:42:47,036 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:42:47,097 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:42:47,098 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:42:47,099 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:42:47,110 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:42:47,111 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:42:47,901 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 07:42:47,903 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-27 07:42:48,075 INFO Regime phase HTF dataset build: 7.5s (103290 samples)
2026-04-27 07:42:48,076 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260427_074248
2026-04-27 07:42:48,277 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=34, n_classes=3)
2026-04-27 07:42:48,278 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=38114 dropped=65176 classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728})
2026-04-27 07:42:48,288 INFO RegimeClassifier[mode=htf_bias]: 38114 samples, classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728}, device=cuda
2026-04-27 07:42:48,289 INFO RegimeClassifier: sample weights — mean=0.708  ambiguous(<0.4)=0.0%
2026-04-27 07:42:48,289 INFO RegimeClassifier[mode=htf_bias]: warm start from existing weights
2026-04-27 07:42:48,289 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-27 07:42:50,519 INFO Regime epoch  1/50 — tr=0.4828 va=1.1066 acc=0.969 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.826}
2026-04-27 07:42:50,593 INFO Regime epoch  2/50 — tr=0.4837 va=1.1025 acc=0.969
2026-04-27 07:42:50,664 INFO Regime epoch  3/50 — tr=0.4834 va=1.1056 acc=0.969
2026-04-27 07:42:50,736 INFO Regime epoch  4/50 — tr=0.4831 va=1.1022 acc=0.969
2026-04-27 07:42:50,817 INFO Regime epoch  5/50 — tr=0.4830 va=1.0997 acc=0.970 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.833}
2026-04-27 07:42:50,888 INFO Regime epoch  6/50 — tr=0.4831 va=1.0989 acc=0.970
2026-04-27 07:42:50,958 INFO Regime epoch  7/50 — tr=0.4811 va=1.0985 acc=0.970
2026-04-27 07:42:51,028 INFO Regime epoch  8/50 — tr=0.4818 va=1.0972 acc=0.970
2026-04-27 07:42:51,096 INFO Regime epoch  9/50 — tr=0.4815 va=1.0950 acc=0.970
2026-04-27 07:42:51,172 INFO Regime epoch 10/50 — tr=0.4807 va=1.0937 acc=0.970 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.834}
2026-04-27 07:42:51,242 INFO Regime epoch 11/50 — tr=0.4801 va=1.0938 acc=0.970
2026-04-27 07:42:51,310 INFO Regime epoch 12/50 — tr=0.4796 va=1.0949 acc=0.970
2026-04-27 07:42:51,379 INFO Regime epoch 13/50 — tr=0.4786 va=1.0939 acc=0.971
2026-04-27 07:42:51,446 INFO Regime epoch 14/50 — tr=0.4786 va=1.0906 acc=0.972
2026-04-27 07:42:51,521 INFO Regime epoch 15/50 — tr=0.4784 va=1.0893 acc=0.971 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.84}
2026-04-27 07:42:51,590 INFO Regime epoch 16/50 — tr=0.4779 va=1.0879 acc=0.972
2026-04-27 07:42:51,661 INFO Regime epoch 17/50 — tr=0.4778 va=1.0878 acc=0.972
2026-04-27 07:42:51,733 INFO Regime epoch 18/50 — tr=0.4771 va=1.0858 acc=0.973
2026-04-27 07:42:51,814 INFO Regime epoch 19/50 — tr=0.4766 va=1.0839 acc=0.973
2026-04-27 07:42:51,898 INFO Regime epoch 20/50 — tr=0.4756 va=1.0821 acc=0.973 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.85}
2026-04-27 07:42:51,975 INFO Regime epoch 21/50 — tr=0.4758 va=1.0791 acc=0.974
2026-04-27 07:42:52,051 INFO Regime epoch 22/50 — tr=0.4758 va=1.0808 acc=0.974
2026-04-27 07:42:52,119 INFO Regime epoch 23/50 — tr=0.4750 va=1.0776 acc=0.975
2026-04-27 07:42:52,192 INFO Regime epoch 24/50 — tr=0.4753 va=1.0773 acc=0.975
2026-04-27 07:42:52,267 INFO Regime epoch 25/50 — tr=0.4751 va=1.0785 acc=0.974 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.854}
2026-04-27 07:42:52,337 INFO Regime epoch 26/50 — tr=0.4746 va=1.0766 acc=0.975
2026-04-27 07:42:52,408 INFO Regime epoch 27/50 — tr=0.4743 va=1.0742 acc=0.975
2026-04-27 07:42:52,481 INFO Regime epoch 28/50 — tr=0.4744 va=1.0739 acc=0.976
2026-04-27 07:42:52,552 INFO Regime epoch 29/50 — tr=0.4742 va=1.0738 acc=0.977
2026-04-27 07:42:52,626 INFO Regime epoch 30/50 — tr=0.4738 va=1.0722 acc=0.977 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.87}
2026-04-27 07:42:52,696 INFO Regime epoch 31/50 — tr=0.4738 va=1.0714 acc=0.976
2026-04-27 07:42:52,774 INFO Regime epoch 32/50 — tr=0.4736 va=1.0681 acc=0.978
2026-04-27 07:42:52,849 INFO Regime epoch 33/50 — tr=0.4735 va=1.0669 acc=0.978
2026-04-27 07:42:52,925 INFO Regime epoch 34/50 — tr=0.4732 va=1.0683 acc=0.978
2026-04-27 07:42:53,003 INFO Regime epoch 35/50 — tr=0.4730 va=1.0678 acc=0.977 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.872}
2026-04-27 07:42:53,071 INFO Regime epoch 36/50 — tr=0.4730 va=1.0662 acc=0.978
2026-04-27 07:42:53,143 INFO Regime epoch 37/50 — tr=0.4733 va=1.0668 acc=0.978
2026-04-27 07:42:53,216 INFO Regime epoch 38/50 — tr=0.4726 va=1.0654 acc=0.978
2026-04-27 07:42:53,284 INFO Regime epoch 39/50 — tr=0.4731 va=1.0670 acc=0.978
2026-04-27 07:42:53,359 INFO Regime epoch 40/50 — tr=0.4729 va=1.0653 acc=0.979 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.881}
2026-04-27 07:42:53,427 INFO Regime epoch 41/50 — tr=0.4729 va=1.0669 acc=0.979
2026-04-27 07:42:53,495 INFO Regime epoch 42/50 — tr=0.4728 va=1.0654 acc=0.979
2026-04-27 07:42:53,565 INFO Regime epoch 43/50 — tr=0.4727 va=1.0656 acc=0.979
2026-04-27 07:42:53,632 INFO Regime epoch 44/50 — tr=0.4728 va=1.0664 acc=0.979
2026-04-27 07:42:53,712 INFO Regime epoch 45/50 — tr=0.4724 va=1.0664 acc=0.979 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.881}
2026-04-27 07:42:53,784 INFO Regime epoch 46/50 — tr=0.4728 va=1.0645 acc=0.979
2026-04-27 07:42:53,855 INFO Regime epoch 47/50 — tr=0.4728 va=1.0671 acc=0.978
2026-04-27 07:42:53,929 INFO Regime epoch 48/50 — tr=0.4726 va=1.0663 acc=0.979
2026-04-27 07:42:54,004 INFO Regime epoch 49/50 — tr=0.4729 va=1.0665 acc=0.979
2026-04-27 07:42:54,078 INFO Regime epoch 50/50 — tr=0.4727 va=1.0672 acc=0.978 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.878}
2026-04-27 07:42:54,087 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 07:42:54,087 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-27 07:42:54,088 INFO Regime phase HTF train: 5.8s
2026-04-27 07:42:54,224 INFO Regime HTF complete: acc=0.979, n=103290 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.883}
2026-04-27 07:42:54,226 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:42:54,385 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-27 07:42:54,388 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 11.41578947368421, 'BIAS_DOWN': 10.635761589403973, 'BIAS_NEUTRAL': 17.960468521229867}
2026-04-27 07:42:54,393 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 4338, 'mean': 0.00011110466582543088, 'mean_over_std': 0.027632581210502927}, 'BIAS_DOWN': {'n': 3212, 'mean': 6.063430222660252e-05, 'mean_over_std': 0.013898489590377157}, 'BIAS_NEUTRAL': {'n': 12266, 'mean': 1.2151554902814046e-05, 'mean_over_std': 0.0031957000513162357}}
2026-04-27 07:42:54,393 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 3651, 'mean': 0.0002852242108836959, 'mean_over_std': 0.07015017232491755}, 'BIAS_DOWN': {'n': 2644, 'mean': -0.00014900749618981248, 'mean_over_std': -0.03333037041298237}, 'BIAS_NEUTRAL': {'n': 1376, 'mean': 9.595797941249082e-05, 'mean_over_std': 0.02492411099521803}}
2026-04-27 07:42:54,397 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-27 07:42:54,399 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:54,400 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:54,402 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:54,404 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:54,405 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:54,407 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:54,408 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:54,410 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:54,412 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:54,413 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:54,416 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:42:54,426 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:54,431 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:54,432 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:54,432 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:54,433 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:54,435 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:55,088 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-27 07:42:55,091 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-27 07:42:55,236 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:55,238 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:55,239 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:55,240 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:55,240 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:55,242 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:55,894 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-27 07:42:55,897 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-27 07:42:56,037 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:56,039 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:56,040 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:56,041 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:56,041 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:56,043 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:56,657 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-27 07:42:56,660 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-27 07:42:56,797 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:56,800 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:56,800 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:56,801 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:56,801 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:56,803 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:57,409 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-27 07:42:57,412 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-27 07:42:57,548 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:57,550 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:57,551 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:57,552 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:57,552 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:57,554 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:58,160 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-27 07:42:58,163 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-27 07:42:58,299 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:58,302 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:58,303 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:58,303 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:58,303 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:58,305 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:58,929 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-27 07:42:58,932 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-27 07:42:59,071 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:42:59,073 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:42:59,073 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:42:59,074 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:42:59,074 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-27 07:42:59,076 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:42:59,692 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-27 07:42:59,695 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-27 07:42:59,836 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:59,838 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:59,839 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:59,839 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:59,840 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:42:59,842 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:43:00,464 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-27 07:43:00,467 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-27 07:43:00,599 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:43:00,601 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:43:00,602 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:43:00,603 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:43:00,603 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:43:00,605 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:43:01,227 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-27 07:43:01,230 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-27 07:43:01,367 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:43:01,370 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:43:01,371 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:43:01,371 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:43:01,371 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-27 07:43:01,373 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-27 07:43:01,989 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-27 07:43:01,992 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-27 07:43:02,145 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:43:02,149 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:43:02,150 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:43:02,151 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:43:02,151 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-27 07:43:02,155 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:43:03,473 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 07:43:03,479 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-27 07:43:03,786 INFO Regime phase LTF dataset build: 9.4s (401471 samples)
2026-04-27 07:43:03,787 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260427_074303
2026-04-27 07:43:03,792 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=18, n_classes=4)
2026-04-27 07:43:03,795 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=299185 dropped=102286 classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127})
2026-04-27 07:43:03,853 INFO RegimeClassifier[mode=ltf_behaviour]: 299185 samples, classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127}, device=cuda
2026-04-27 07:43:03,854 INFO RegimeClassifier: sample weights — mean=0.693  ambiguous(<0.4)=0.0%
2026-04-27 07:43:03,854 INFO RegimeClassifier[mode=ltf_behaviour]: warm start from existing weights
2026-04-27 07:43:03,854 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-27 07:43:04,445 INFO Regime epoch  1/50 — tr=0.6254 va=1.2572 acc=0.823 per_class={'TRENDING': 0.8, 'RANGING': 0.751, 'CONSOLIDATING': 0.837, 'VOLATILE': 0.899}
2026-04-27 07:43:04,961 INFO Regime epoch  2/50 — tr=0.6255 va=1.2575 acc=0.824
2026-04-27 07:43:05,483 INFO Regime epoch  3/50 — tr=0.6252 va=1.2588 acc=0.824
2026-04-27 07:43:06,049 INFO Regime epoch  4/50 — tr=0.6254 va=1.2564 acc=0.823
2026-04-27 07:43:06,641 INFO Regime epoch  5/50 — tr=0.6250 va=1.2537 acc=0.825 per_class={'TRENDING': 0.806, 'RANGING': 0.751, 'CONSOLIDATING': 0.848, 'VOLATILE': 0.891}
2026-04-27 07:43:07,178 INFO Regime epoch  6/50 — tr=0.6253 va=1.2524 acc=0.824
2026-04-27 07:43:07,710 INFO Regime epoch  7/50 — tr=0.6247 va=1.2545 acc=0.825
2026-04-27 07:43:08,238 INFO Regime epoch  8/50 — tr=0.6243 va=1.2540 acc=0.825
2026-04-27 07:43:08,772 INFO Regime epoch  9/50 — tr=0.6242 va=1.2511 acc=0.830
2026-04-27 07:43:09,342 INFO Regime epoch 10/50 — tr=0.6237 va=1.2533 acc=0.828 per_class={'TRENDING': 0.806, 'RANGING': 0.747, 'CONSOLIDATING': 0.856, 'VOLATILE': 0.898}
2026-04-27 07:43:09,850 INFO Regime epoch 11/50 — tr=0.6236 va=1.2489 acc=0.827
2026-04-27 07:43:10,380 INFO Regime epoch 12/50 — tr=0.6233 va=1.2521 acc=0.832
2026-04-27 07:43:10,895 INFO Regime epoch 13/50 — tr=0.6229 va=1.2450 acc=0.829
2026-04-27 07:43:11,416 INFO Regime epoch 14/50 — tr=0.6228 va=1.2462 acc=0.831
2026-04-27 07:43:11,980 INFO Regime epoch 15/50 — tr=0.6225 va=1.2430 acc=0.831 per_class={'TRENDING': 0.809, 'RANGING': 0.759, 'CONSOLIDATING': 0.853, 'VOLATILE': 0.899}
2026-04-27 07:43:12,507 INFO Regime epoch 16/50 — tr=0.6224 va=1.2425 acc=0.833
2026-04-27 07:43:13,056 INFO Regime epoch 17/50 — tr=0.6219 va=1.2370 acc=0.832
2026-04-27 07:43:13,619 INFO Regime epoch 18/50 — tr=0.6218 va=1.2426 acc=0.832
2026-04-27 07:43:14,169 INFO Regime epoch 19/50 — tr=0.6219 va=1.2438 acc=0.834
2026-04-27 07:43:14,733 INFO Regime epoch 20/50 — tr=0.6215 va=1.2386 acc=0.832 per_class={'TRENDING': 0.807, 'RANGING': 0.757, 'CONSOLIDATING': 0.863, 'VOLATILE': 0.902}
2026-04-27 07:43:15,270 INFO Regime epoch 21/50 — tr=0.6212 va=1.2390 acc=0.835
2026-04-27 07:43:15,838 INFO Regime epoch 22/50 — tr=0.6209 va=1.2359 acc=0.833
2026-04-27 07:43:16,346 INFO Regime epoch 23/50 — tr=0.6210 va=1.2341 acc=0.834
2026-04-27 07:43:16,878 INFO Regime epoch 24/50 — tr=0.6207 va=1.2332 acc=0.834
2026-04-27 07:43:17,440 INFO Regime epoch 25/50 — tr=0.6207 va=1.2390 acc=0.833 per_class={'TRENDING': 0.812, 'RANGING': 0.757, 'CONSOLIDATING': 0.864, 'VOLATILE': 0.899}
2026-04-27 07:43:17,978 INFO Regime epoch 26/50 — tr=0.6203 va=1.2384 acc=0.835
2026-04-27 07:43:18,506 INFO Regime epoch 27/50 — tr=0.6203 va=1.2325 acc=0.834
2026-04-27 07:43:19,035 INFO Regime epoch 28/50 — tr=0.6203 va=1.2353 acc=0.836
2026-04-27 07:43:19,548 INFO Regime epoch 29/50 — tr=0.6200 va=1.2345 acc=0.836
2026-04-27 07:43:20,112 INFO Regime epoch 30/50 — tr=0.6200 va=1.2336 acc=0.833 per_class={'TRENDING': 0.81, 'RANGING': 0.758, 'CONSOLIDATING': 0.868, 'VOLATILE': 0.9}
2026-04-27 07:43:20,619 INFO Regime epoch 31/50 — tr=0.6197 va=1.2294 acc=0.835
2026-04-27 07:43:21,137 INFO Regime epoch 32/50 — tr=0.6197 va=1.2325 acc=0.837
2026-04-27 07:43:21,651 INFO Regime epoch 33/50 — tr=0.6196 va=1.2333 acc=0.836
2026-04-27 07:43:22,192 INFO Regime epoch 34/50 — tr=0.6196 va=1.2346 acc=0.837
2026-04-27 07:43:22,744 INFO Regime epoch 35/50 — tr=0.6195 va=1.2315 acc=0.834 per_class={'TRENDING': 0.812, 'RANGING': 0.76, 'CONSOLIDATING': 0.872, 'VOLATILE': 0.897}
2026-04-27 07:43:23,259 INFO Regime epoch 36/50 — tr=0.6196 va=1.2321 acc=0.834
2026-04-27 07:43:23,790 INFO Regime epoch 37/50 — tr=0.6197 va=1.2329 acc=0.837
2026-04-27 07:43:24,325 INFO Regime epoch 38/50 — tr=0.6193 va=1.2285 acc=0.836
2026-04-27 07:43:24,875 INFO Regime epoch 39/50 — tr=0.6195 va=1.2352 acc=0.838
2026-04-27 07:43:25,455 INFO Regime epoch 40/50 — tr=0.6194 va=1.2263 acc=0.837 per_class={'TRENDING': 0.817, 'RANGING': 0.761, 'CONSOLIDATING': 0.872, 'VOLATILE': 0.895}
2026-04-27 07:43:26,004 INFO Regime epoch 41/50 — tr=0.6196 va=1.2303 acc=0.835
2026-04-27 07:43:26,511 INFO Regime epoch 42/50 — tr=0.6194 va=1.2306 acc=0.837
2026-04-27 07:43:27,011 INFO Regime epoch 43/50 — tr=0.6192 va=1.2296 acc=0.837
2026-04-27 07:43:27,531 INFO Regime epoch 44/50 — tr=0.6192 va=1.2292 acc=0.835
2026-04-27 07:43:28,098 INFO Regime epoch 45/50 — tr=0.6193 va=1.2281 acc=0.836 per_class={'TRENDING': 0.817, 'RANGING': 0.759, 'CONSOLIDATING': 0.874, 'VOLATILE': 0.894}
2026-04-27 07:43:28,617 INFO Regime epoch 46/50 — tr=0.6193 va=1.2338 acc=0.840
2026-04-27 07:43:29,135 INFO Regime epoch 47/50 — tr=0.6194 va=1.2345 acc=0.838
2026-04-27 07:43:29,636 INFO Regime epoch 48/50 — tr=0.6193 va=1.2294 acc=0.836
2026-04-27 07:43:30,158 INFO Regime epoch 49/50 — tr=0.6190 va=1.2296 acc=0.835
2026-04-27 07:43:30,706 INFO Regime epoch 50/50 — tr=0.6192 va=1.2309 acc=0.837 per_class={'TRENDING': 0.818, 'RANGING': 0.757, 'CONSOLIDATING': 0.874, 'VOLATILE': 0.896}
2026-04-27 07:43:30,706 INFO Regime early stop at epoch 50 (no_improve=10)
2026-04-27 07:43:30,747 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 07:43:30,747 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-27 07:43:30,749 INFO Regime phase LTF train: 27.0s
2026-04-27 07:43:30,879 INFO Regime LTF complete: acc=0.837, n=401471 per_class={'TRENDING': 0.817, 'RANGING': 0.761, 'CONSOLIDATING': 0.872, 'VOLATILE': 0.895}
2026-04-27 07:43:30,883 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-27 07:43:31,387 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-27 07:43:31,392 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 9.355291913830783, 'RANGING': 6.096303199751476, 'CONSOLIDATING': 5.598885793871866, 'VOLATILE': 6.771351107094442}
2026-04-27 07:43:31,400 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 29965, 'mean': 2.149062259083323e-05, 'mean_over_std': 0.011155601434349563}, 'RANGING': {'n': 19623, 'mean': -3.254493608872846e-06, 'mean_over_std': -0.001761415548292494}, 'CONSOLIDATING': {'n': 10050, 'mean': -3.039756158146958e-06, 'mean_over_std': -0.001792488091618151}, 'VOLATILE': {'n': 14985, 'mean': 1.8102764458593193e-05, 'mean_over_std': 0.006974696291378742}}
2026-04-27 07:43:31,400 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 26145, 'mean': 2.874435605138053e-05, 'mean_over_std': 0.01595450632691917}, 'RANGING': {'n': 7858, 'mean': 1.7829776785521727e-05, 'mean_over_std': 0.011851561804467463}, 'CONSOLIDATING': {'n': 8543, 'mean': 7.845231961250924e-06, 'mean_over_std': 0.004915839521549481}, 'VOLATILE': {'n': 13636, 'mean': 5.480114267306668e-06, 'mean_over_std': 0.0020378119917389375}}
2026-04-27 07:43:31,404 INFO Regime retrain total: 226.0s (504761 samples)
2026-04-27 07:43:31,409 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-27 07:43:31,409 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 07:43:31,409 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 07:43:31,409 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-27 07:43:31,410 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-27 07:43:31,410 INFO Retrain complete. Total wall-clock: 226.0s
  DONE  Retrain regime [train-split retrain]
  START Retrain quality [train-split retrain]
2026-04-27 07:43:32,887 INFO retrain environment: KAGGLE
2026-04-27 07:43:34,552 INFO Device: CUDA (2 GPU(s))
2026-04-27 07:43:34,564 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 07:43:34,564 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 07:43:34,564 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 07:43:34,564 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 07:43:34,564 INFO Retrain data split: train
2026-04-27 07:43:34,565 INFO === QualityScorer retrain ===
2026-04-27 07:43:34,718 INFO NumExpr defaulting to 4 threads.
2026-04-27 07:43:34,914 INFO QualityScorer: CUDA available — using GPU
2026-04-27 07:43:34,916 INFO QualityScorer: skipped 68 journal records outside allowed splits ['live', 'paper', 'production', 'train']
2026-04-27 07:43:34,917 INFO Quality phase label creation: 0.0s (0 trades)
2026-04-27 07:43:34,921 INFO Retrain complete. Total wall-clock: 0.4s
  WARN  Retrain quality failed (exit 1) — continuing
  START Retrain rl [train-split retrain]
2026-04-27 07:43:35,535 INFO retrain environment: KAGGLE
2026-04-27 07:43:37,257 INFO Device: CUDA (2 GPU(s))
2026-04-27 07:43:37,268 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-27 07:43:37,268 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-27 07:43:37,268 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-27 07:43:37,268 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-27 07:43:37,268 INFO Retrain data split: train
2026-04-27 07:43:37,269 INFO === RLAgent (PPO) retrain ===
2026-04-27 07:43:37,276 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260427_074337
2026-04-27 07:43:37,278 INFO RLAgent: skipped 68 journal records outside allowed splits ['live', 'paper', 'production', 'train']
2026-04-27 07:43:37,278 INFO RL phase episode loading: 0.0s (0 episodes)
2026-04-27 07:43:40.665158: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777275820.905181   83667 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777275820.966110   83667 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777275821.509853   83667 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777275821.509899   83667 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777275821.509905   83667 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777275821.509908   83667 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-04-27 07:43:55,859 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-04-27 07:43:57,918 INFO RLAgent: skipped 68 journal records outside allowed splits ['live', 'paper', 'production', 'train']
2026-04-27 07:43:57,918 WARNING RLAgent.retrain: only 0 episodes — skipping
2026-04-27 07:43:57,919 INFO RL phase PPO train: 20.6s | total: 20.6s
2026-04-27 07:43:57,922 INFO Retrain complete. Total wall-clock: 20.7s
  WARN  Retrain rl failed (exit 1) — continuing

=== Round 3: Backtest on last 3yr (post-retrain evaluation) ===
  START Round 3 - Post-retrain backtest (last 3yr)
2026-04-27 07:43:59,887 INFO === STEP 6: BACKTEST (round3) ===
2026-04-27 07:43:59,888 INFO BT_WINDOW=round3 — post-retrain eval: 2022-08-05 → 2025-08-05 (last 3yr)
2026-04-27 07:43:59,888 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-04-27 07:43:59,888 INFO Round 3 — running backtest: 2022-08-05 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-27 07:44:02,433 WARNING QualityScorer unavailable (weights missing or load failed)
2026-04-27 07:47:08,841 WARNING ml_trader: portfolio drawdown 8.4% after trade exit — halting all trading
2026-04-27 07:47:09,558 INFO Round 3 backtest — 70 trades | avg WR=31.4% | avg PF=1.07 | avg Sharpe=0.46
2026-04-27 07:47:09,558 INFO   ml_trader: 70 trades | WR=31.4% | PF=1.07 | Return=3.1% | DD=8.4% | Sharpe=0.46
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 70
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (70 rows)
  DONE  Round 3 - Post-retrain backtest (last 3yr)
  Saved Round 3 result → round3_summary.json

======================================================================
  BLIND BACKTEST PIPELINE COMPLETE
======================================================================
  Round 1 (val window)          trades=20  WR=15.0%  PF=0.460  Sharpe=-5.043
  Round 2 (blind test)          trades=48  WR=29.2%  PF=1.139  Sharpe=0.854
  Round 3 (last 3yr)            trades=70  WR=31.4%  PF=1.073  Sharpe=0.458


WARNING: GITHUB_TOKEN not set — skipping GitHub push
2026-04-27 07:47:09,852 INFO Round 3: wrote 70 journal entries (total in file: 138)