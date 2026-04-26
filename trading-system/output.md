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
2026-04-26 16:57:56,590 INFO Loading feature-engineered data...
2026-04-26 16:57:57,232 INFO Loaded 221743 rows, 202 features
2026-04-26 16:57:57,234 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-26 16:57:57,236 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-26 16:57:57,236 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-26 16:57:57,236 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-26 16:57:57,237 INFO No leakage confirmed: train < val < test timestamps

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
2026-04-26 16:57:59,640 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-26 16:57:59,640 INFO --- Training regime ---
2026-04-26 16:57:59,640 INFO Running retrain --model regime
2026-04-26 16:57:59,819 INFO retrain environment: KAGGLE
2026-04-26 16:58:01,454 INFO Device: CUDA (2 GPU(s))
2026-04-26 16:58:01,465 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 16:58:01,465 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 16:58:01,465 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 16:58:01,467 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 16:58:01,468 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-26 16:58:01,638 INFO NumExpr defaulting to 4 threads.
2026-04-26 16:58:01,877 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 16:58:01,878 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 16:58:01,878 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 16:58:01,878 INFO Regime phase macro_correlations: 0.0s
2026-04-26 16:58:01,878 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-26 16:58:01,920 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 16:58:01,921 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 16:58:01,949 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 16:58:01,963 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 16:58:01,985 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 16:58:01,999 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 16:58:02,022 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 16:58:02,036 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 16:58:02,058 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 16:58:02,072 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 16:58:02,094 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 16:58:02,109 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 16:58:02,129 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 16:58:02,143 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 16:58:02,162 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 16:58:02,176 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 16:58:02,199 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 16:58:02,213 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 16:58:02,236 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 16:58:02,254 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 16:58:02,281 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 16:58:02,301 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 16:58:02,344 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 16:58:03,503 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 16:58:27,011 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 16:58:27,016 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 24.7s
2026-04-26 16:58:27,019 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 16:58:37,549 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 16:58:37,555 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 10.5s
2026-04-26 16:58:37,558 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 16:58:44,993 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 16:58:44,994 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 7.4s
2026-04-26 16:58:44,994 INFO Regime phase GMM HTF total: 42.6s
2026-04-26 16:58:44,997 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 16:59:56,322 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 16:59:56,325 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 71.3s
2026-04-26 16:59:56,325 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 17:00:27,308 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 17:00:27,310 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 31.0s
2026-04-26 17:00:27,310 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 17:00:49,500 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 17:00:49,502 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 22.2s
2026-04-26 17:00:49,502 INFO Regime phase GMM LTF total: 124.5s
2026-04-26 17:00:49,603 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-26 17:00:49,605 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:00:49,606 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:00:49,607 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:00:49,609 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:00:49,610 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:00:49,611 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:00:49,612 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:00:49,613 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:00:49,614 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:00:49,616 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:00:49,617 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:00:49,741 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:49,782 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:49,783 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:49,783 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:49,792 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:49,793 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:00:50,215 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-26 17:00:50,216 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-26 17:00:50,393 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:50,427 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:50,428 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:50,428 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:50,436 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:50,437 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:00:50,805 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-26 17:00:50,806 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-26 17:00:50,992 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:51,029 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:51,030 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:51,030 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:51,038 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:51,039 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:00:51,412 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-26 17:00:51,414 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-26 17:00:51,582 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:51,618 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:51,618 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:51,619 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:51,627 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:51,628 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:00:52,029 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-26 17:00:52,030 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-26 17:00:52,198 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:52,233 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:52,233 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:52,234 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:52,242 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:52,243 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:00:52,616 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-26 17:00:52,617 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-26 17:00:52,780 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:52,813 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:52,814 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:52,814 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:52,822 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:52,823 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:00:53,193 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-26 17:00:53,194 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-26 17:00:53,344 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:00:53,370 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:00:53,371 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:00:53,371 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:00:53,378 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:00:53,379 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:00:53,741 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-26 17:00:53,742 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-26 17:00:53,911 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:53,948 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:53,949 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:53,949 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:53,957 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:53,958 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:00:54,329 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-26 17:00:54,330 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-26 17:00:54,498 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:54,529 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:54,530 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:54,531 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:54,539 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:54,540 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:00:54,907 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-26 17:00:54,908 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-26 17:00:55,073 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:55,108 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:55,108 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:55,109 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:55,117 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:00:55,118 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:00:55,490 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-26 17:00:55,492 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-26 17:00:55,752 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:00:55,810 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:00:55,811 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:00:55,811 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:00:55,822 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:00:55,823 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:00:56,607 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-26 17:00:56,609 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-26 17:00:56,768 INFO Regime phase HTF dataset build: 7.2s (103290 samples)
2026-04-26 17:00:56,785 INFO RegimeClassifier[mode=htf_bias]: 103290 samples, classes={'BIAS_UP': 18622, 'BIAS_DOWN': 18286, 'BIAS_NEUTRAL': 66382}, device=cuda
2026-04-26 17:00:56,786 INFO RegimeClassifier: sample weights — mean=0.360  ambiguous(<0.4)=69.5%
2026-04-26 17:00:57,059 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-04-26 17:00:57,059 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 17:01:01,886 INFO Regime epoch  1/50 — tr=0.3339 va=2.2895 acc=0.237 per_class={'BIAS_UP': 0.162, 'BIAS_DOWN': 0.589, 'BIAS_NEUTRAL': 0.175}
2026-04-26 17:01:02,078 INFO Regime epoch  2/50 — tr=0.3299 va=2.2293 acc=0.351
2026-04-26 17:01:02,259 INFO Regime epoch  3/50 — tr=0.3219 va=2.1570 acc=0.434
2026-04-26 17:01:02,436 INFO Regime epoch  4/50 — tr=0.3106 va=2.0658 acc=0.525
2026-04-26 17:01:02,641 INFO Regime epoch  5/50 — tr=0.2974 va=1.9817 acc=0.585 per_class={'BIAS_UP': 0.795, 'BIAS_DOWN': 0.931, 'BIAS_NEUTRAL': 0.422}
2026-04-26 17:01:02,823 INFO Regime epoch  6/50 — tr=0.2865 va=1.9201 acc=0.593
2026-04-26 17:01:03,008 INFO Regime epoch  7/50 — tr=0.2776 va=1.8904 acc=0.583
2026-04-26 17:01:03,178 INFO Regime epoch  8/50 — tr=0.2714 va=1.8630 acc=0.587
2026-04-26 17:01:03,360 INFO Regime epoch  9/50 — tr=0.2668 va=1.8384 acc=0.596
2026-04-26 17:01:03,558 INFO Regime epoch 10/50 — tr=0.2632 va=1.8316 acc=0.601 per_class={'BIAS_UP': 0.977, 'BIAS_DOWN': 0.982, 'BIAS_NEUTRAL': 0.369}
2026-04-26 17:01:03,740 INFO Regime epoch 11/50 — tr=0.2595 va=1.8242 acc=0.608
2026-04-26 17:01:03,915 INFO Regime epoch 12/50 — tr=0.2573 va=1.8178 acc=0.613
2026-04-26 17:01:04,105 INFO Regime epoch 13/50 — tr=0.2553 va=1.8143 acc=0.612
2026-04-26 17:01:04,284 INFO Regime epoch 14/50 — tr=0.2538 va=1.8121 acc=0.621
2026-04-26 17:01:04,479 INFO Regime epoch 15/50 — tr=0.2525 va=1.8091 acc=0.625 per_class={'BIAS_UP': 0.979, 'BIAS_DOWN': 0.988, 'BIAS_NEUTRAL': 0.407}
2026-04-26 17:01:04,650 INFO Regime epoch 16/50 — tr=0.2514 va=1.8086 acc=0.625
2026-04-26 17:01:04,838 INFO Regime epoch 17/50 — tr=0.2504 va=1.8058 acc=0.631
2026-04-26 17:01:05,025 INFO Regime epoch 18/50 — tr=0.2496 va=1.8058 acc=0.629
2026-04-26 17:01:05,196 INFO Regime epoch 19/50 — tr=0.2489 va=1.8096 acc=0.629
2026-04-26 17:01:05,386 INFO Regime epoch 20/50 — tr=0.2483 va=1.8046 acc=0.636 per_class={'BIAS_UP': 0.979, 'BIAS_DOWN': 0.988, 'BIAS_NEUTRAL': 0.424}
2026-04-26 17:01:05,572 INFO Regime epoch 21/50 — tr=0.2476 va=1.8029 acc=0.638
2026-04-26 17:01:05,744 INFO Regime epoch 22/50 — tr=0.2473 va=1.8036 acc=0.643
2026-04-26 17:01:05,920 INFO Regime epoch 23/50 — tr=0.2468 va=1.8026 acc=0.640
2026-04-26 17:01:06,110 INFO Regime epoch 24/50 — tr=0.2464 va=1.8021 acc=0.640
2026-04-26 17:01:06,305 INFO Regime epoch 25/50 — tr=0.2461 va=1.8008 acc=0.645 per_class={'BIAS_UP': 0.982, 'BIAS_DOWN': 0.988, 'BIAS_NEUTRAL': 0.437}
2026-04-26 17:01:06,485 INFO Regime epoch 26/50 — tr=0.2457 va=1.8012 acc=0.643
2026-04-26 17:01:06,655 INFO Regime epoch 27/50 — tr=0.2455 va=1.7992 acc=0.645
2026-04-26 17:01:06,830 INFO Regime epoch 28/50 — tr=0.2453 va=1.7968 acc=0.647
2026-04-26 17:01:07,018 INFO Regime epoch 29/50 — tr=0.2452 va=1.7964 acc=0.650
2026-04-26 17:01:07,219 INFO Regime epoch 30/50 — tr=0.2450 va=1.7977 acc=0.651 per_class={'BIAS_UP': 0.98, 'BIAS_DOWN': 0.988, 'BIAS_NEUTRAL': 0.447}
2026-04-26 17:01:07,391 INFO Regime epoch 31/50 — tr=0.2448 va=1.7975 acc=0.648
2026-04-26 17:01:07,567 INFO Regime epoch 32/50 — tr=0.2447 va=1.7955 acc=0.653
2026-04-26 17:01:07,742 INFO Regime epoch 33/50 — tr=0.2446 va=1.7960 acc=0.652
2026-04-26 17:01:07,925 INFO Regime epoch 34/50 — tr=0.2444 va=1.7940 acc=0.653
2026-04-26 17:01:08,115 INFO Regime epoch 35/50 — tr=0.2444 va=1.7949 acc=0.651 per_class={'BIAS_UP': 0.983, 'BIAS_DOWN': 0.99, 'BIAS_NEUTRAL': 0.447}
2026-04-26 17:01:08,302 INFO Regime epoch 36/50 — tr=0.2444 va=1.7924 acc=0.655
2026-04-26 17:01:08,488 INFO Regime epoch 37/50 — tr=0.2444 va=1.7933 acc=0.655
2026-04-26 17:01:08,659 INFO Regime epoch 38/50 — tr=0.2443 va=1.7932 acc=0.656
2026-04-26 17:01:08,849 INFO Regime epoch 39/50 — tr=0.2441 va=1.7952 acc=0.652
2026-04-26 17:01:09,055 INFO Regime epoch 40/50 — tr=0.2442 va=1.7929 acc=0.656 per_class={'BIAS_UP': 0.98, 'BIAS_DOWN': 0.99, 'BIAS_NEUTRAL': 0.456}
2026-04-26 17:01:09,227 INFO Regime epoch 41/50 — tr=0.2440 va=1.7908 acc=0.659
2026-04-26 17:01:09,418 INFO Regime epoch 42/50 — tr=0.2441 va=1.7931 acc=0.654
2026-04-26 17:01:09,603 INFO Regime epoch 43/50 — tr=0.2440 va=1.7920 acc=0.656
2026-04-26 17:01:09,777 INFO Regime epoch 44/50 — tr=0.2440 va=1.7915 acc=0.659
2026-04-26 17:01:09,976 INFO Regime epoch 45/50 — tr=0.2440 va=1.7917 acc=0.656 per_class={'BIAS_UP': 0.982, 'BIAS_DOWN': 0.989, 'BIAS_NEUTRAL': 0.455}
2026-04-26 17:01:10,151 INFO Regime epoch 46/50 — tr=0.2440 va=1.7905 acc=0.659
2026-04-26 17:01:10,332 INFO Regime epoch 47/50 — tr=0.2438 va=1.7900 acc=0.658
2026-04-26 17:01:10,504 INFO Regime epoch 48/50 — tr=0.2439 va=1.7933 acc=0.654
2026-04-26 17:01:10,672 INFO Regime epoch 49/50 — tr=0.2439 va=1.7932 acc=0.654
2026-04-26 17:01:10,857 INFO Regime epoch 50/50 — tr=0.2438 va=1.7911 acc=0.658 per_class={'BIAS_UP': 0.981, 'BIAS_DOWN': 0.989, 'BIAS_NEUTRAL': 0.458}
2026-04-26 17:01:10,874 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 17:01:10,874 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 17:01:10,875 INFO Regime phase HTF train: 14.1s
2026-04-26 17:01:10,999 INFO Regime HTF complete: acc=0.658, n=103290
2026-04-26 17:01:11,000 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:01:11,171 INFO Rule labels LTF_BEHAVIOUR [4H]: {'TRENDING': 8970, 'RANGING': 4592, 'CONSOLIDATING': 2247, 'VOLATILE': 4008}  ambiguous=4539 (total=19817)  short_runs_zeroed=570
2026-04-26 17:01:11,178 INFO Regime[4H] persistence (avg bars/run) on XAUUSD 4H:
{0: 12.121621621621621, 1: 6.4858757062146895, 2: 5.575682382133995, 3: 10.329896907216495}
2026-04-26 17:01:11,180 INFO Regime[4H] return separation on XAUUSD 4H:
{0: 5.9502411776840314e-05, 1: 5.664222849552333e-05, 2: -6.138442068010657e-05, 3: 4.239843820627722e-05}
2026-04-26 17:01:11,180 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-26 17:01:11,182 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:01:11,184 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:01:11,185 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:01:11,187 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:01:11,189 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:01:11,190 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:01:11,192 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:01:11,193 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:01:11,195 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:01:11,196 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:01:11,199 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:01:11,209 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:11,211 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:11,212 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:11,212 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:11,213 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:11,214 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:01:11,882 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-26 17:01:11,885 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-26 17:01:12,014 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:12,017 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:12,018 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:12,018 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:12,018 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:12,020 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:01:12,620 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-26 17:01:12,623 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-26 17:01:12,753 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:12,757 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:12,758 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:12,758 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:12,758 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:12,760 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:01:13,356 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-26 17:01:13,359 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-26 17:01:13,492 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:13,496 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:13,497 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:13,497 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:13,497 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:13,499 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:01:14,098 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-26 17:01:14,100 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-26 17:01:14,232 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:14,235 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:14,235 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:14,236 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:14,236 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:14,238 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:01:14,836 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-26 17:01:14,839 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-26 17:01:14,977 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:14,980 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:14,981 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:14,981 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:14,981 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:14,984 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:01:15,569 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-26 17:01:15,572 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-26 17:01:15,699 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:01:15,701 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:01:15,702 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:01:15,702 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:01:15,702 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:01:15,704 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:01:16,302 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-26 17:01:16,305 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-26 17:01:16,443 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:16,445 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:16,446 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:16,447 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:16,447 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:16,449 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:01:17,038 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-26 17:01:17,041 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-26 17:01:17,178 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:17,181 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:17,181 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:17,182 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:17,182 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:17,184 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:01:17,768 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-26 17:01:17,771 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-26 17:01:17,907 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:17,909 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:17,910 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:17,911 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:17,911 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:17,913 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:01:18,506 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-26 17:01:18,509 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-26 17:01:18,654 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:01:18,658 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:01:18,659 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:01:18,659 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:01:18,660 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:01:18,663 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:01:19,951 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-26 17:01:19,956 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-26 17:01:20,247 INFO Regime phase LTF dataset build: 9.1s (401471 samples)
2026-04-26 17:01:20,303 INFO RegimeClassifier[mode=ltf_behaviour]: 401471 samples, classes={'TRENDING': 160094, 'RANGING': 105286, 'CONSOLIDATING': 53524, 'VOLATILE': 82567}, device=cuda
2026-04-26 17:01:20,304 INFO RegimeClassifier: sample weights — mean=0.505  ambiguous(<0.4)=33.0%
2026-04-26 17:01:20,306 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-04-26 17:01:20,306 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 17:01:21,044 INFO Regime epoch  1/50 — tr=0.6516 va=2.0664 acc=0.312 per_class={'TRENDING': 0.214, 'RANGING': 0.258, 'CONSOLIDATING': 0.198, 'VOLATILE': 0.657}
2026-04-26 17:01:21,712 INFO Regime epoch  2/50 — tr=0.6245 va=1.9927 acc=0.391
2026-04-26 17:01:22,392 INFO Regime epoch  3/50 — tr=0.5908 va=1.9160 acc=0.470
2026-04-26 17:01:23,090 INFO Regime epoch  4/50 — tr=0.5624 va=1.8459 acc=0.504
2026-04-26 17:01:23,815 INFO Regime epoch  5/50 — tr=0.5433 va=1.7993 acc=0.532 per_class={'TRENDING': 0.494, 'RANGING': 0.155, 'CONSOLIDATING': 0.765, 'VOLATILE': 0.943}
2026-04-26 17:01:24,478 INFO Regime epoch  6/50 — tr=0.5309 va=1.7704 acc=0.559
2026-04-26 17:01:25,132 INFO Regime epoch  7/50 — tr=0.5216 va=1.7590 acc=0.580
2026-04-26 17:01:25,790 INFO Regime epoch  8/50 — tr=0.5146 va=1.7481 acc=0.596
2026-04-26 17:01:26,457 INFO Regime epoch  9/50 — tr=0.5089 va=1.7377 acc=0.619
2026-04-26 17:01:27,173 INFO Regime epoch 10/50 — tr=0.5040 va=1.7225 acc=0.636 per_class={'TRENDING': 0.688, 'RANGING': 0.254, 'CONSOLIDATING': 0.787, 'VOLATILE': 0.932}
2026-04-26 17:01:27,821 INFO Regime epoch 11/50 — tr=0.5000 va=1.7059 acc=0.648
2026-04-26 17:01:28,491 INFO Regime epoch 12/50 — tr=0.4969 va=1.6927 acc=0.659
2026-04-26 17:01:29,131 INFO Regime epoch 13/50 — tr=0.4941 va=1.6812 acc=0.668
2026-04-26 17:01:29,828 INFO Regime epoch 14/50 — tr=0.4918 va=1.6665 acc=0.674
2026-04-26 17:01:30,547 INFO Regime epoch 15/50 — tr=0.4898 va=1.6567 acc=0.678 per_class={'TRENDING': 0.774, 'RANGING': 0.26, 'CONSOLIDATING': 0.845, 'VOLATILE': 0.921}
2026-04-26 17:01:31,240 INFO Regime epoch 16/50 — tr=0.4882 va=1.6473 acc=0.682
2026-04-26 17:01:31,932 INFO Regime epoch 17/50 — tr=0.4867 va=1.6390 acc=0.688
2026-04-26 17:01:32,597 INFO Regime epoch 18/50 — tr=0.4855 va=1.6291 acc=0.692
2026-04-26 17:01:33,234 INFO Regime epoch 19/50 — tr=0.4844 va=1.6261 acc=0.694
2026-04-26 17:01:33,943 INFO Regime epoch 20/50 — tr=0.4832 va=1.6193 acc=0.696 per_class={'TRENDING': 0.79, 'RANGING': 0.282, 'CONSOLIDATING': 0.882, 'VOLATILE': 0.924}
2026-04-26 17:01:34,607 INFO Regime epoch 21/50 — tr=0.4824 va=1.6119 acc=0.699
2026-04-26 17:01:35,264 INFO Regime epoch 22/50 — tr=0.4814 va=1.6070 acc=0.702
2026-04-26 17:01:35,924 INFO Regime epoch 23/50 — tr=0.4806 va=1.6029 acc=0.704
2026-04-26 17:01:36,602 INFO Regime epoch 24/50 — tr=0.4801 va=1.5993 acc=0.707
2026-04-26 17:01:37,316 INFO Regime epoch 25/50 — tr=0.4792 va=1.5942 acc=0.710 per_class={'TRENDING': 0.803, 'RANGING': 0.307, 'CONSOLIDATING': 0.92, 'VOLATILE': 0.909}
2026-04-26 17:01:37,977 INFO Regime epoch 26/50 — tr=0.4790 va=1.5933 acc=0.711
2026-04-26 17:01:38,656 INFO Regime epoch 27/50 — tr=0.4783 va=1.5923 acc=0.713
2026-04-26 17:01:39,333 INFO Regime epoch 28/50 — tr=0.4779 va=1.5890 acc=0.714
2026-04-26 17:01:39,988 INFO Regime epoch 29/50 — tr=0.4777 va=1.5858 acc=0.716
2026-04-26 17:01:40,689 INFO Regime epoch 30/50 — tr=0.4772 va=1.5870 acc=0.715 per_class={'TRENDING': 0.81, 'RANGING': 0.315, 'CONSOLIDATING': 0.921, 'VOLATILE': 0.911}
2026-04-26 17:01:41,341 INFO Regime epoch 31/50 — tr=0.4769 va=1.5840 acc=0.717
2026-04-26 17:01:42,038 INFO Regime epoch 32/50 — tr=0.4767 va=1.5822 acc=0.718
2026-04-26 17:01:42,715 INFO Regime epoch 33/50 — tr=0.4767 va=1.5821 acc=0.719
2026-04-26 17:01:43,362 INFO Regime epoch 34/50 — tr=0.4763 va=1.5811 acc=0.719
2026-04-26 17:01:44,052 INFO Regime epoch 35/50 — tr=0.4762 va=1.5816 acc=0.718 per_class={'TRENDING': 0.814, 'RANGING': 0.319, 'CONSOLIDATING': 0.927, 'VOLATILE': 0.911}
2026-04-26 17:01:44,707 INFO Regime epoch 36/50 — tr=0.4760 va=1.5797 acc=0.720
2026-04-26 17:01:45,358 INFO Regime epoch 37/50 — tr=0.4760 va=1.5817 acc=0.719
2026-04-26 17:01:46,034 INFO Regime epoch 38/50 — tr=0.4758 va=1.5784 acc=0.721
2026-04-26 17:01:46,709 INFO Regime epoch 39/50 — tr=0.4756 va=1.5788 acc=0.721
2026-04-26 17:01:47,427 INFO Regime epoch 40/50 — tr=0.4756 va=1.5778 acc=0.722 per_class={'TRENDING': 0.818, 'RANGING': 0.326, 'CONSOLIDATING': 0.932, 'VOLATILE': 0.906}
2026-04-26 17:01:48,082 INFO Regime epoch 41/50 — tr=0.4755 va=1.5790 acc=0.721
2026-04-26 17:01:48,731 INFO Regime epoch 42/50 — tr=0.4754 va=1.5789 acc=0.721
2026-04-26 17:01:49,382 INFO Regime epoch 43/50 — tr=0.4752 va=1.5767 acc=0.722
2026-04-26 17:01:50,021 INFO Regime epoch 44/50 — tr=0.4753 va=1.5784 acc=0.722
2026-04-26 17:01:50,726 INFO Regime epoch 45/50 — tr=0.4754 va=1.5768 acc=0.722 per_class={'TRENDING': 0.823, 'RANGING': 0.321, 'CONSOLIDATING': 0.935, 'VOLATILE': 0.902}
2026-04-26 17:01:51,367 INFO Regime epoch 46/50 — tr=0.4753 va=1.5800 acc=0.721
2026-04-26 17:01:52,046 INFO Regime epoch 47/50 — tr=0.4751 va=1.5787 acc=0.721
2026-04-26 17:01:52,687 INFO Regime epoch 48/50 — tr=0.4752 va=1.5804 acc=0.721
2026-04-26 17:01:53,331 INFO Regime epoch 49/50 — tr=0.4753 va=1.5757 acc=0.723
2026-04-26 17:01:54,040 INFO Regime epoch 50/50 — tr=0.4753 va=1.5777 acc=0.722 per_class={'TRENDING': 0.819, 'RANGING': 0.323, 'CONSOLIDATING': 0.936, 'VOLATILE': 0.904}
2026-04-26 17:01:54,090 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 17:01:54,090 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 17:01:54,093 INFO Regime phase LTF train: 33.8s
2026-04-26 17:01:54,229 INFO Regime LTF complete: acc=0.723, n=401471
2026-04-26 17:01:54,232 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:01:54,751 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-26 17:01:54,756 INFO Regime[1H] persistence (avg bars/run) on XAUUSD 1H:
{0: 9.355291913830783, 1: 6.096303199751476, 2: 5.598885793871866, 3: 6.771351107094442}
2026-04-26 17:01:54,759 INFO Regime[1H] return separation on XAUUSD 1H:
{0: 2.149062259083323e-05, 1: -3.254493608872846e-06, 2: -3.039756158146958e-06, 3: 1.8102764458593193e-05}
2026-04-26 17:01:54,759 INFO Regime retrain total: 233.3s (504761 samples)
2026-04-26 17:01:54,773 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 17:01:54,773 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 17:01:54,773 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 17:01:54,774 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-26 17:01:54,774 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-26 17:01:54,774 INFO Retrain complete. Total wall-clock: 233.3s
2026-04-26 17:01:57,121 INFO Model regime: SUCCESS
2026-04-26 17:01:57,121 INFO --- Training gru ---
2026-04-26 17:01:57,121 INFO Running retrain --model gru
2026-04-26 17:01:57,421 INFO retrain environment: KAGGLE
2026-04-26 17:01:59,060 INFO Device: CUDA (2 GPU(s))
2026-04-26 17:01:59,071 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 17:01:59,071 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 17:01:59,071 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 17:01:59,071 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 17:01:59,072 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-26 17:01:59,217 INFO NumExpr defaulting to 4 threads.
2026-04-26 17:01:59,410 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 17:01:59,411 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 17:01:59,411 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 17:01:59,411 INFO GRU phase macro_correlations: 0.0s
2026-04-26 17:01:59,411 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-26 17:01:59,412 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260426_170159
2026-04-26 17:01:59,414 WARNING WeightsManifest: no manifest at /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json — assuming compatible (run a full retrain to generate one)
2026-04-26 17:01:59,555 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:59,575 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:59,589 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:59,596 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:01:59,597 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 17:01:59,598 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 17:01:59,598 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 17:01:59,598 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 17:01:59,599 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:01:59,675 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5739 (total=8402)  short_runs_zeroed=591
2026-04-26 17:01:59,677 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:01:59,945 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=11257 (total=32738)  short_runs_zeroed=4986
2026-04-26 17:01:59,978 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:00,259 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:00,383 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:00,481 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:00,680 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:00,700 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:00,714 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:00,720 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:00,721 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:00,795 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=6111 (total=8402)  short_runs_zeroed=726
2026-04-26 17:02:00,797 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:01,022 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=11216 (total=32738)  short_runs_zeroed=4347
2026-04-26 17:02:01,037 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:01,301 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:01,425 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:01,523 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:01,718 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:01,738 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:01,752 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:01,759 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:01,760 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:01,843 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5865 (total=8402)  short_runs_zeroed=675
2026-04-26 17:02:01,845 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:02,090 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10993 (total=32740)  short_runs_zeroed=4399
2026-04-26 17:02:02,107 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:02,380 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:02,504 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:02,599 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:02,788 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:02,807 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:02,821 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:02,828 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:02,829 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:02,908 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5868 (total=8402)  short_runs_zeroed=681
2026-04-26 17:02:02,909 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:03,146 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10567 (total=32739)  short_runs_zeroed=3955
2026-04-26 17:02:03,170 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:03,435 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:03,561 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:03,656 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:03,844 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:03,866 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:03,882 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:03,890 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:03,891 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:03,969 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5761 (total=8403)  short_runs_zeroed=577
2026-04-26 17:02:03,971 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:04,204 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10785 (total=32740)  short_runs_zeroed=4397
2026-04-26 17:02:04,221 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:04,500 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:04,630 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:04,729 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:04,914 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:04,935 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:04,950 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:04,957 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:04,958 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:05,035 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5885 (total=8403)  short_runs_zeroed=774
2026-04-26 17:02:05,037 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:05,275 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=10340 (total=32739)  short_runs_zeroed=3724
2026-04-26 17:02:05,291 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:05,560 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:05,687 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:05,785 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:05,961 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:02:05,978 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:02:05,991 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:02:05,997 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:02:05,998 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:06,073 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5882 (total=8402)  short_runs_zeroed=629
2026-04-26 17:02:06,074 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:06,303 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10968 (total=32739)  short_runs_zeroed=4898
2026-04-26 17:02:06,315 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:06,573 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:06,696 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:06,791 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:06,968 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:06,986 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:07,001 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:07,007 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:07,008 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:07,081 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5725 (total=8402)  short_runs_zeroed=615
2026-04-26 17:02:07,083 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:07,312 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=10378 (total=32740)  short_runs_zeroed=3880
2026-04-26 17:02:07,332 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:07,592 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:07,720 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:07,822 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:08,006 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:08,024 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:08,039 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:08,045 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:08,046 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:08,119 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5801 (total=8402)  short_runs_zeroed=616
2026-04-26 17:02:08,120 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:08,348 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=11177 (total=32741)  short_runs_zeroed=3896
2026-04-26 17:02:08,363 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:08,627 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:08,758 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:08,856 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:09,037 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:09,057 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:09,071 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:09,079 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:02:09,080 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:09,154 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5863 (total=8403)  short_runs_zeroed=589
2026-04-26 17:02:09,156 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:09,388 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10444 (total=32743)  short_runs_zeroed=4275
2026-04-26 17:02:09,403 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:09,672 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:09,800 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:09,895 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:02:10,186 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:02:10,211 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:02:10,227 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:02:10,237 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:02:10,238 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:02:10,384 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=13279 (total=19817)  short_runs_zeroed=1542
2026-04-26 17:02:10,387 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:02:10,879 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=24073 (total=74624)  short_runs_zeroed=9134
2026-04-26 17:02:10,924 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:02:11,429 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:02:11,629 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:02:11,755 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:02:11,891 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-26 17:02:12,150 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-04-26 17:02:12,151 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-04-26 17:02:12,151 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-26 17:02:12,151 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-26 17:02:59,604 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-26 17:02:59,604 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-26 17:03:00,939 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-26 17:03:05,071 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=5)
2026-04-26 17:03:18,901 INFO train_multi TF=ALL epoch 1/50 train=0.8838 val=0.8756
2026-04-26 17:03:18,909 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 17:03:18,910 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 17:03:18,910 INFO train_multi TF=ALL: new best val=0.8756 — saved
2026-04-26 17:03:30,623 INFO train_multi TF=ALL epoch 2/50 train=0.8624 val=0.8304
2026-04-26 17:03:30,627 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 17:03:30,628 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 17:03:30,628 INFO train_multi TF=ALL: new best val=0.8304 — saved
2026-04-26 17:03:42,344 INFO train_multi TF=ALL epoch 3/50 train=0.7441 val=0.6879
2026-04-26 17:03:42,349 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 17:03:42,349 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 17:03:42,349 INFO train_multi TF=ALL: new best val=0.6879 — saved
2026-04-26 17:03:54,089 INFO train_multi TF=ALL epoch 4/50 train=0.6923 val=0.6878
2026-04-26 17:03:54,093 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 17:03:54,093 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 17:03:54,093 INFO train_multi TF=ALL: new best val=0.6878 — saved
2026-04-26 17:04:05,809 INFO train_multi TF=ALL epoch 5/50 train=0.6908 val=0.6881
2026-04-26 17:04:17,503 INFO train_multi TF=ALL epoch 6/50 train=0.6901 val=0.6887
2026-04-26 17:04:29,227 INFO train_multi TF=ALL epoch 7/50 train=0.6899 val=0.6883
2026-04-26 17:04:40,988 INFO train_multi TF=ALL epoch 8/50 train=0.6895 val=0.6883
2026-04-26 17:04:52,827 INFO train_multi TF=ALL epoch 9/50 train=0.6892 val=0.6875
2026-04-26 17:04:52,831 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 17:04:52,831 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 17:04:52,831 INFO train_multi TF=ALL: new best val=0.6875 — saved
2026-04-26 17:05:04,608 INFO train_multi TF=ALL epoch 10/50 train=0.6885 val=0.6866
2026-04-26 17:05:04,613 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 17:05:04,613 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 17:05:04,613 INFO train_multi TF=ALL: new best val=0.6866 — saved
2026-04-26 17:05:16,317 INFO train_multi TF=ALL epoch 11/50 train=0.6871 val=0.6853
2026-04-26 17:05:16,322 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 17:05:16,322 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 17:05:16,322 INFO train_multi TF=ALL: new best val=0.6853 — saved
2026-04-26 17:05:28,012 INFO train_multi TF=ALL epoch 12/50 train=0.6823 val=0.6785
2026-04-26 17:05:28,016 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 17:05:28,016 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 17:05:28,016 INFO train_multi TF=ALL: new best val=0.6785 — saved
2026-04-26 17:05:39,700 INFO train_multi TF=ALL epoch 13/50 train=0.6731 val=0.6636
2026-04-26 17:05:39,705 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 17:05:39,705 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 17:05:39,705 INFO train_multi TF=ALL: new best val=0.6636 — saved
2026-04-26 17:05:51,440 INFO train_multi TF=ALL epoch 14/50 train=0.6613 val=0.6489
2026-04-26 17:05:51,445 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 17:05:51,445 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 17:05:51,445 INFO train_multi TF=ALL: new best val=0.6489 — saved
2026-04-26 17:06:03,200 INFO train_multi TF=ALL epoch 15/50 train=0.6499 val=0.6374
2026-04-26 17:06:03,205 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 17:06:03,205 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 17:06:03,205 INFO train_multi TF=ALL: new best val=0.6374 — saved
2026-04-26 17:06:14,869 INFO train_multi TF=ALL epoch 16/50 train=0.6415 val=0.6314
2026-04-26 17:06:14,874 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 17:06:14,874 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 17:06:14,874 INFO train_multi TF=ALL: new best val=0.6314 — saved
2026-04-26 17:06:26,632 INFO train_multi TF=ALL epoch 17/50 train=0.6350 val=0.6278
2026-04-26 17:06:26,636 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 17:06:26,636 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 17:06:26,636 INFO train_multi TF=ALL: new best val=0.6278 — saved
2026-04-26 17:06:38,248 INFO train_multi TF=ALL epoch 18/50 train=0.6304 val=0.6253
2026-04-26 17:06:38,253 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 17:06:38,253 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 17:06:38,253 INFO train_multi TF=ALL: new best val=0.6253 — saved
2026-04-26 17:06:49,902 INFO train_multi TF=ALL epoch 19/50 train=0.6263 val=0.6215
2026-04-26 17:06:49,907 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 17:06:49,907 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 17:06:49,907 INFO train_multi TF=ALL: new best val=0.6215 — saved
2026-04-26 17:07:01,645 INFO train_multi TF=ALL epoch 20/50 train=0.6232 val=0.6183
2026-04-26 17:07:01,650 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 17:07:01,650 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 17:07:01,650 INFO train_multi TF=ALL: new best val=0.6183 — saved
2026-04-26 17:07:13,456 INFO train_multi TF=ALL epoch 21/50 train=0.6199 val=0.6217
2026-04-26 17:07:25,217 INFO train_multi TF=ALL epoch 22/50 train=0.6176 val=0.6207
2026-04-26 17:07:36,918 INFO train_multi TF=ALL epoch 23/50 train=0.6157 val=0.6189
2026-04-26 17:07:48,602 INFO train_multi TF=ALL epoch 24/50 train=0.6139 val=0.6187
2026-04-26 17:08:00,265 INFO train_multi TF=ALL epoch 25/50 train=0.6124 val=0.6212
2026-04-26 17:08:00,265 INFO train_multi TF=ALL early stop at epoch 25
2026-04-26 17:08:00,399 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-26 17:08:00,400 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-26 17:08:00,400 INFO Retrain complete. Total wall-clock: 361.3s
2026-04-26 17:08:02,289 INFO Model gru: SUCCESS
2026-04-26 17:08:02,289 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 17:08:02,289 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 17:08:02,289 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 17:08:02,289 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-04-26 17:08:02,290 INFO   [DEFERRED] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip (expected after Round 1)
2026-04-26 17:08:02,290 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-04-26 17:08:02,290 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer', 'rl_ppo']
2026-04-26 17:08:02,291 INFO Saved 5 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  START Round 1 - Backtest (val)
2026-04-26 17:08:03,098 INFO === STEP 6: BACKTEST (round1) ===
2026-04-26 17:08:03,099 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-04-26 17:08:03,099 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-26 17:08:03,099 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
2026-04-26 17:08:05,384 WARNING QualityScorer unavailable (weights missing or load failed)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260426_170805.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)                3   0.0%   0.00   -3.0%  0.0%  0.0%   3.0% -1932.54
  gate_diagnostics: bars=468696 no_signal=3 quality_block=0 session_skip=317895 density=0 pm_reject=0 daily_skip=150795 cooldown=0

Calibration Summary:
  all          [N/A] Insufficient data: 3 samples
  ml_trader    [N/A] Insufficient data: 3 samples
2026-04-26 17:10:12,839 INFO Round 1 backtest — 3 trades | avg WR=0.0% | avg PF=0.00 | avg Sharpe=-1932.54
2026-04-26 17:10:12,839 INFO   ml_trader: 3 trades | WR=0.0% | PF=0.00 | Return=-3.0% | DD=3.0% | Sharpe=-1932.54
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 3
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (3 rows)
2026-04-26 17:10:13,058 INFO Round 1: wrote 3 journal entries (total in file: 3)
  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 3 entries

=== Round 1 → Retrain Quality + RL ===
  START Round 1 - Quality+RL retrain
2026-04-26 17:10:13,351 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-26 17:10:13,351 INFO Journal entries: 3
2026-04-26 17:10:13,351 WARNING Journal has only 3 entries (need 50) — backtest produced too few trades. Skipping Quality+RL training. Check step6 logs.
  DONE  Round 1 - Quality+RL retrain

=== Round 2: BLIND backtest on test window (unseen 2yr) ===
  START Round 2 - Blind backtest (test)
2026-04-26 17:10:13,845 INFO === STEP 6: BACKTEST (round2) ===
2026-04-26 17:10:13,845 INFO BT_WINDOW=round2 — BLIND backtest: 2023-08-07 → 2025-08-05 (test set)
2026-04-26 17:10:13,845 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-26 17:10:13,846 INFO Round 2 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-26 17:10:16,158 WARNING QualityScorer unavailable (weights missing or load failed)
2026-04-26 17:12:24,836 INFO Round 2 backtest — 5 trades | avg WR=20.0% | avg PF=0.63 | avg Sharpe=-3.39
2026-04-26 17:12:24,836 INFO   ml_trader: 5 trades | WR=20.0% | PF=0.63 | Return=-1.5% | DD=3.0% | Sharpe=-3.39
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 5
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (5 rows)
2026-04-26 17:12:25,054 INFO Round 2: wrote 5 journal entries (total in file: 8)
  DONE  Round 2 - Blind backtest (test)
  Saved Round 2 result → round2_summary.json
  Journal after Round 2: 8 entries

=== Round 2 → Retrain Quality + RL (Round 1+2 journal) ===
  START Round 2 - Quality+RL retrain
2026-04-26 17:12:25,336 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-26 17:12:25,336 INFO Journal entries: 8
2026-04-26 17:12:25,336 WARNING Journal has only 8 entries (need 50) — backtest produced too few trades. Skipping Quality+RL training. Check step6 logs.
2026-04-26 17:12:25,529 INFO retrain environment: KAGGLE
  DONE  Round 2 - Quality+RL retrain

=== Round 3: Incremental retrain ALL models on full data ===
  START Retrain gru [full-data retrain]
2026-04-26 17:12:27,141 INFO Device: CUDA (2 GPU(s))
2026-04-26 17:12:27,152 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 17:12:27,153 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 17:12:27,153 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 17:12:27,153 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 17:12:27,154 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===