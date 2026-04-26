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
2026-04-26 17:37:38,113 INFO Loading feature-engineered data...
2026-04-26 17:37:38,755 INFO Loaded 221743 rows, 202 features
2026-04-26 17:37:38,756 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-26 17:37:38,759 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-26 17:37:38,759 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-26 17:37:38,759 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-26 17:37:38,759 INFO No leakage confirmed: train < val < test timestamps

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
2026-04-26 17:37:41,178 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-26 17:37:41,178 INFO --- Training regime ---
2026-04-26 17:37:41,179 INFO Running retrain --model regime
2026-04-26 17:37:41,383 INFO retrain environment: KAGGLE
2026-04-26 17:37:42,988 INFO Device: CUDA (2 GPU(s))
2026-04-26 17:37:42,999 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 17:37:42,999 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 17:37:42,999 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 17:37:43,001 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 17:37:43,002 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-26 17:37:43,154 INFO NumExpr defaulting to 4 threads.
2026-04-26 17:37:43,366 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 17:37:43,366 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 17:37:43,366 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 17:37:43,367 INFO Regime phase macro_correlations: 0.0s
2026-04-26 17:37:43,367 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-26 17:37:43,403 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 17:37:43,404 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:37:43,431 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:37:43,445 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:37:43,466 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:37:43,480 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:37:43,502 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:37:43,515 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:37:43,538 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:37:43,552 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:37:43,574 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:37:43,587 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:37:43,607 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:37:43,620 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:37:43,638 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:37:43,651 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:37:43,671 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:37:43,686 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:37:43,707 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:37:43,722 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:37:43,744 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:37:43,760 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:37:43,797 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:37:44,913 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 17:38:07,105 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 17:38:07,110 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias) in 23.3s
2026-04-26 17:38:07,110 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 17:38:16,738 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 17:38:16,740 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias) in 9.6s
2026-04-26 17:38:16,740 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-26 17:38:24,009 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-26 17:38:24,010 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias) in 7.3s
2026-04-26 17:38:24,010 INFO Regime phase GMM HTF total: 40.2s
2026-04-26 17:38:24,011 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 17:39:31,970 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 17:39:31,973 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour) in 68.0s
2026-04-26 17:39:31,973 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 17:40:01,827 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 17:40:01,828 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour) in 29.9s
2026-04-26 17:40:01,828 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-26 17:40:23,283 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-26 17:40:23,285 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour) in 21.5s
2026-04-26 17:40:23,285 INFO Regime phase GMM LTF total: 119.3s
2026-04-26 17:40:23,383 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-26 17:40:23,384 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:23,385 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:23,386 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:23,387 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:23,388 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:23,389 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:23,390 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:23,391 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:23,392 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:23,393 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:23,394 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:40:23,513 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:23,553 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:23,554 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:23,554 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:23,561 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:23,562 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:23,958 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-26 17:40:23,959 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-26 17:40:24,129 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:24,164 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:24,164 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:24,165 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:24,173 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:24,174 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:24,546 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-26 17:40:24,547 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-26 17:40:24,729 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:24,763 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:24,764 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:24,765 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:24,772 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:24,773 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:25,142 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-26 17:40:25,143 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-26 17:40:25,325 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:25,368 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:25,369 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:25,369 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:25,377 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:25,377 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:25,742 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-26 17:40:25,743 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-26 17:40:25,916 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:25,953 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:25,954 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:25,954 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:25,962 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:25,963 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:26,333 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-26 17:40:26,334 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-26 17:40:26,497 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:26,529 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:26,530 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:26,530 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:26,539 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:26,540 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:26,917 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-26 17:40:26,918 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-26 17:40:27,068 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:40:27,097 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:40:27,098 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:40:27,098 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:40:27,105 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:40:27,106 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:27,478 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-26 17:40:27,479 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-26 17:40:27,647 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:27,679 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:27,680 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:27,680 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:27,688 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:27,689 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:28,058 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-26 17:40:28,059 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-26 17:40:28,222 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:28,255 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:28,256 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:28,256 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:28,264 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:28,265 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:28,626 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-26 17:40:28,627 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-26 17:40:28,794 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:28,828 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:28,829 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:28,829 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:28,837 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:28,838 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:29,197 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-26 17:40:29,198 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-26 17:40:29,462 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:40:29,517 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:40:29,518 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:40:29,519 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:40:29,529 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:40:29,530 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:40:30,334 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 17:40:30,336 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-26 17:40:30,489 INFO Regime phase HTF dataset build: 7.1s (103290 samples)
2026-04-26 17:40:30,490 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=38114 dropped=65176 classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728})
2026-04-26 17:40:30,501 INFO RegimeClassifier[mode=htf_bias]: 38114 samples, classes={'BIAS_UP': 15218, 'BIAS_DOWN': 15168, 'BIAS_NEUTRAL': 7728}, device=cuda
2026-04-26 17:40:30,501 INFO RegimeClassifier: sample weights — mean=0.708  ambiguous(<0.4)=0.0%
2026-04-26 17:40:30,803 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-04-26 17:40:30,804 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 17:40:35,669 INFO Regime epoch  1/50 — tr=0.7634 va=1.9297 acc=0.180 per_class={'BIAS_UP': 0.003, 'BIAS_DOWN': 0.011, 'BIAS_NEUTRAL': 0.972}
2026-04-26 17:40:35,737 INFO Regime epoch  2/50 — tr=0.7586 va=1.8893 acc=0.209
2026-04-26 17:40:35,800 INFO Regime epoch  3/50 — tr=0.7487 va=1.8291 acc=0.247
2026-04-26 17:40:35,869 INFO Regime epoch  4/50 — tr=0.7323 va=1.7660 acc=0.307
2026-04-26 17:40:35,944 INFO Regime epoch  5/50 — tr=0.7108 va=1.6973 acc=0.429 per_class={'BIAS_UP': 0.358, 'BIAS_DOWN': 0.261, 'BIAS_NEUTRAL': 0.937}
2026-04-26 17:40:36,014 INFO Regime epoch  6/50 — tr=0.6844 va=1.6287 acc=0.605
2026-04-26 17:40:36,081 INFO Regime epoch  7/50 — tr=0.6568 va=1.5635 acc=0.724
2026-04-26 17:40:36,144 INFO Regime epoch  8/50 — tr=0.6272 va=1.5000 acc=0.797
2026-04-26 17:40:36,210 INFO Regime epoch  9/50 — tr=0.6013 va=1.4433 acc=0.854
2026-04-26 17:40:36,278 INFO Regime epoch 10/50 — tr=0.5780 va=1.3891 acc=0.898 per_class={'BIAS_UP': 0.92, 'BIAS_DOWN': 0.921, 'BIAS_NEUTRAL': 0.794}
2026-04-26 17:40:36,341 INFO Regime epoch 11/50 — tr=0.5603 va=1.3465 acc=0.920
2026-04-26 17:40:36,408 INFO Regime epoch 12/50 — tr=0.5460 va=1.3129 acc=0.931
2026-04-26 17:40:36,480 INFO Regime epoch 13/50 — tr=0.5349 va=1.2817 acc=0.940
2026-04-26 17:40:36,548 INFO Regime epoch 14/50 — tr=0.5264 va=1.2585 acc=0.946
2026-04-26 17:40:36,620 INFO Regime epoch 15/50 — tr=0.5204 va=1.2384 acc=0.950 per_class={'BIAS_UP': 0.987, 'BIAS_DOWN': 0.987, 'BIAS_NEUTRAL': 0.782}
2026-04-26 17:40:36,693 INFO Regime epoch 16/50 — tr=0.5146 va=1.2245 acc=0.954
2026-04-26 17:40:36,762 INFO Regime epoch 17/50 — tr=0.5109 va=1.2111 acc=0.956
2026-04-26 17:40:36,831 INFO Regime epoch 18/50 — tr=0.5073 va=1.2016 acc=0.957
2026-04-26 17:40:36,898 INFO Regime epoch 19/50 — tr=0.5045 va=1.1967 acc=0.959
2026-04-26 17:40:36,965 INFO Regime epoch 20/50 — tr=0.5025 va=1.1869 acc=0.960 per_class={'BIAS_UP': 0.994, 'BIAS_DOWN': 0.995, 'BIAS_NEUTRAL': 0.802}
2026-04-26 17:40:37,035 INFO Regime epoch 21/50 — tr=0.5002 va=1.1816 acc=0.961
2026-04-26 17:40:37,105 INFO Regime epoch 22/50 — tr=0.4984 va=1.1757 acc=0.961
2026-04-26 17:40:37,172 INFO Regime epoch 23/50 — tr=0.4969 va=1.1738 acc=0.962
2026-04-26 17:40:37,241 INFO Regime epoch 24/50 — tr=0.4956 va=1.1697 acc=0.963
2026-04-26 17:40:37,311 INFO Regime epoch 25/50 — tr=0.4939 va=1.1635 acc=0.964 per_class={'BIAS_UP': 0.996, 'BIAS_DOWN': 0.997, 'BIAS_NEUTRAL': 0.816}
2026-04-26 17:40:37,374 INFO Regime epoch 26/50 — tr=0.4935 va=1.1594 acc=0.965
2026-04-26 17:40:37,437 INFO Regime epoch 27/50 — tr=0.4919 va=1.1566 acc=0.965
2026-04-26 17:40:37,501 INFO Regime epoch 28/50 — tr=0.4907 va=1.1484 acc=0.967
2026-04-26 17:40:37,571 INFO Regime epoch 29/50 — tr=0.4895 va=1.1464 acc=0.967
2026-04-26 17:40:37,653 INFO Regime epoch 30/50 — tr=0.4890 va=1.1449 acc=0.967 per_class={'BIAS_UP': 0.998, 'BIAS_DOWN': 0.998, 'BIAS_NEUTRAL': 0.829}
2026-04-26 17:40:37,723 INFO Regime epoch 31/50 — tr=0.4882 va=1.1434 acc=0.968
2026-04-26 17:40:37,791 INFO Regime epoch 32/50 — tr=0.4875 va=1.1408 acc=0.969
2026-04-26 17:40:37,860 INFO Regime epoch 33/50 — tr=0.4867 va=1.1409 acc=0.968
2026-04-26 17:40:37,927 INFO Regime epoch 34/50 — tr=0.4861 va=1.1378 acc=0.969
2026-04-26 17:40:38,000 INFO Regime epoch 35/50 — tr=0.4852 va=1.1342 acc=0.969 per_class={'BIAS_UP': 0.998, 'BIAS_DOWN': 0.998, 'BIAS_NEUTRAL': 0.839}
2026-04-26 17:40:38,070 INFO Regime epoch 36/50 — tr=0.4854 va=1.1336 acc=0.970
2026-04-26 17:40:38,138 INFO Regime epoch 37/50 — tr=0.4852 va=1.1331 acc=0.970
2026-04-26 17:40:38,208 INFO Regime epoch 38/50 — tr=0.4846 va=1.1339 acc=0.970
2026-04-26 17:40:38,270 INFO Regime epoch 39/50 — tr=0.4844 va=1.1340 acc=0.971
2026-04-26 17:40:38,340 INFO Regime epoch 40/50 — tr=0.4838 va=1.1347 acc=0.971 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.844}
2026-04-26 17:40:38,403 INFO Regime epoch 41/50 — tr=0.4844 va=1.1303 acc=0.970
2026-04-26 17:40:38,468 INFO Regime epoch 42/50 — tr=0.4834 va=1.1287 acc=0.970
2026-04-26 17:40:38,535 INFO Regime epoch 43/50 — tr=0.4838 va=1.1304 acc=0.971
2026-04-26 17:40:38,603 INFO Regime epoch 44/50 — tr=0.4836 va=1.1327 acc=0.971
2026-04-26 17:40:38,672 INFO Regime epoch 45/50 — tr=0.4838 va=1.1302 acc=0.971 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.845}
2026-04-26 17:40:38,738 INFO Regime epoch 46/50 — tr=0.4833 va=1.1298 acc=0.972
2026-04-26 17:40:38,802 INFO Regime epoch 47/50 — tr=0.4828 va=1.1293 acc=0.971
2026-04-26 17:40:38,867 INFO Regime epoch 48/50 — tr=0.4830 va=1.1292 acc=0.972
2026-04-26 17:40:38,934 INFO Regime epoch 49/50 — tr=0.4832 va=1.1321 acc=0.972
2026-04-26 17:40:39,005 INFO Regime epoch 50/50 — tr=0.4833 va=1.1301 acc=0.972 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.85}
2026-04-26 17:40:39,015 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 17:40:39,015 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 17:40:39,016 INFO Regime phase HTF train: 8.5s
2026-04-26 17:40:39,137 INFO Regime HTF complete: acc=0.970, n=103290 per_class={'BIAS_UP': 0.999, 'BIAS_DOWN': 0.999, 'BIAS_NEUTRAL': 0.842}
2026-04-26 17:40:39,138 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:40:39,288 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 17:40:39,297 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 11.41578947368421, 'BIAS_DOWN': 10.635761589403973, 'BIAS_NEUTRAL': 17.960468521229867}
2026-04-26 17:40:39,301 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 4338, 'mean': 0.00011110466582543088, 'mean_over_std': 0.027632581210502927}, 'BIAS_DOWN': {'n': 3212, 'mean': 6.063430222660252e-05, 'mean_over_std': 0.013898489590377157}, 'BIAS_NEUTRAL': {'n': 12266, 'mean': 1.2151554902814046e-05, 'mean_over_std': 0.0031957000513162357}}
2026-04-26 17:40:39,301 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 3651, 'mean': 0.0002852242108836959, 'mean_over_std': 0.07015017232491755}, 'BIAS_DOWN': {'n': 2644, 'mean': -0.00014900749618981248, 'mean_over_std': -0.03333037041298237}, 'BIAS_NEUTRAL': {'n': 1376, 'mean': 9.595797941249082e-05, 'mean_over_std': 0.02492411099521803}}
2026-04-26 17:40:39,302 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-26 17:40:39,304 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:39,306 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:39,307 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:39,309 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:39,311 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:39,312 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:39,313 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:39,315 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:39,317 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:39,318 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:39,321 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:40:39,331 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:39,336 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:39,336 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:39,337 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:39,337 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:39,340 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:39,975 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-26 17:40:39,978 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-26 17:40:40,113 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:40,116 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:40,117 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:40,117 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:40,117 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:40,119 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:40,695 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-26 17:40:40,699 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-26 17:40:40,829 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:40,831 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:40,832 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:40,832 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:40,833 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:40,835 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:41,397 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-26 17:40:41,400 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-26 17:40:41,537 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:41,539 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:41,540 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:41,540 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:41,541 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:41,543 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:42,094 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-26 17:40:42,097 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-26 17:40:42,226 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:42,228 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:42,229 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:42,229 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:42,230 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:42,231 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:42,794 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-26 17:40:42,797 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-26 17:40:42,929 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:42,931 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:42,932 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:42,932 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:42,933 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:42,935 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:43,500 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-26 17:40:43,503 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-26 17:40:43,634 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:40:43,635 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:40:43,636 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:40:43,636 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:40:43,637 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:40:43,638 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:44,213 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-26 17:40:44,216 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-26 17:40:44,351 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:44,354 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:44,354 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:44,355 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:44,355 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:44,357 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:44,927 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-26 17:40:44,930 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-26 17:40:45,082 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:45,084 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:45,085 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:45,085 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:45,086 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:45,088 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:45,693 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-26 17:40:45,696 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-26 17:40:45,833 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:45,835 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:45,836 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:45,837 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:45,837 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:40:45,839 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:40:46,424 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-26 17:40:46,427 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-26 17:40:46,570 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:40:46,574 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:40:46,575 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:40:46,575 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:40:46,576 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:40:46,579 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:40:47,852 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 17:40:47,858 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-26 17:40:48,151 INFO Regime phase LTF dataset build: 8.8s (401471 samples)
2026-04-26 17:40:48,154 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=299185 dropped=102286 classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127})
2026-04-26 17:40:48,217 INFO RegimeClassifier[mode=ltf_behaviour]: 299185 samples, classes={'TRENDING': 139297, 'RANGING': 40562, 'CONSOLIDATING': 45199, 'VOLATILE': 74127}, device=cuda
2026-04-26 17:40:48,218 INFO RegimeClassifier: sample weights — mean=0.693  ambiguous(<0.4)=0.0%
2026-04-26 17:40:48,221 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-04-26 17:40:48,221 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-26 17:40:48,792 INFO Regime epoch  1/50 — tr=0.8686 va=2.0188 acc=0.418 per_class={'TRENDING': 0.145, 'RANGING': 0.155, 'CONSOLIDATING': 0.774, 'VOLATILE': 0.869}
2026-04-26 17:40:49,288 INFO Regime epoch  2/50 — tr=0.8409 va=1.9172 acc=0.495
2026-04-26 17:40:49,807 INFO Regime epoch  3/50 — tr=0.7994 va=1.7930 acc=0.555
2026-04-26 17:40:50,295 INFO Regime epoch  4/50 — tr=0.7572 va=1.6687 acc=0.600
2026-04-26 17:40:50,822 INFO Regime epoch  5/50 — tr=0.7262 va=1.5635 acc=0.635 per_class={'TRENDING': 0.476, 'RANGING': 0.555, 'CONSOLIDATING': 0.732, 'VOLATILE': 0.923}
2026-04-26 17:40:51,296 INFO Regime epoch  6/50 — tr=0.7054 va=1.5055 acc=0.657
2026-04-26 17:40:51,787 INFO Regime epoch  7/50 — tr=0.6914 va=1.4705 acc=0.679
2026-04-26 17:40:52,267 INFO Regime epoch  8/50 — tr=0.6811 va=1.4452 acc=0.690
2026-04-26 17:40:52,760 INFO Regime epoch  9/50 — tr=0.6735 va=1.4319 acc=0.706
2026-04-26 17:40:53,307 INFO Regime epoch 10/50 — tr=0.6677 va=1.4195 acc=0.717 per_class={'TRENDING': 0.61, 'RANGING': 0.695, 'CONSOLIDATING': 0.715, 'VOLATILE': 0.936}
2026-04-26 17:40:53,801 INFO Regime epoch 11/50 — tr=0.6626 va=1.3946 acc=0.727
2026-04-26 17:40:54,316 INFO Regime epoch 12/50 — tr=0.6586 va=1.3900 acc=0.737
2026-04-26 17:40:54,792 INFO Regime epoch 13/50 — tr=0.6552 va=1.3713 acc=0.748
2026-04-26 17:40:55,310 INFO Regime epoch 14/50 — tr=0.6519 va=1.3648 acc=0.755
2026-04-26 17:40:55,842 INFO Regime epoch 15/50 — tr=0.6495 va=1.3518 acc=0.760 per_class={'TRENDING': 0.692, 'RANGING': 0.723, 'CONSOLIDATING': 0.735, 'VOLATILE': 0.925}
2026-04-26 17:40:56,317 INFO Regime epoch 16/50 — tr=0.6476 va=1.3451 acc=0.768
2026-04-26 17:40:56,808 INFO Regime epoch 17/50 — tr=0.6450 va=1.3296 acc=0.771
2026-04-26 17:40:57,319 INFO Regime epoch 18/50 — tr=0.6436 va=1.3209 acc=0.775
2026-04-26 17:40:57,801 INFO Regime epoch 19/50 — tr=0.6418 va=1.3188 acc=0.780
2026-04-26 17:40:58,356 INFO Regime epoch 20/50 — tr=0.6400 va=1.3076 acc=0.784 per_class={'TRENDING': 0.74, 'RANGING': 0.748, 'CONSOLIDATING': 0.744, 'VOLATILE': 0.915}
2026-04-26 17:40:58,863 INFO Regime epoch 21/50 — tr=0.6388 va=1.3015 acc=0.789
2026-04-26 17:40:59,353 INFO Regime epoch 22/50 — tr=0.6373 va=1.2941 acc=0.791
2026-04-26 17:40:59,852 INFO Regime epoch 23/50 — tr=0.6360 va=1.2898 acc=0.795
2026-04-26 17:41:00,339 INFO Regime epoch 24/50 — tr=0.6354 va=1.2887 acc=0.799
2026-04-26 17:41:00,866 INFO Regime epoch 25/50 — tr=0.6336 va=1.2805 acc=0.800 per_class={'TRENDING': 0.767, 'RANGING': 0.759, 'CONSOLIDATING': 0.769, 'VOLATILE': 0.907}
2026-04-26 17:41:01,341 INFO Regime epoch 26/50 — tr=0.6331 va=1.2758 acc=0.802
2026-04-26 17:41:01,864 INFO Regime epoch 27/50 — tr=0.6322 va=1.2751 acc=0.804
2026-04-26 17:41:02,352 INFO Regime epoch 28/50 — tr=0.6312 va=1.2689 acc=0.806
2026-04-26 17:41:02,864 INFO Regime epoch 29/50 — tr=0.6303 va=1.2634 acc=0.808
2026-04-26 17:41:03,404 INFO Regime epoch 30/50 — tr=0.6299 va=1.2601 acc=0.808 per_class={'TRENDING': 0.775, 'RANGING': 0.767, 'CONSOLIDATING': 0.79, 'VOLATILE': 0.907}
2026-04-26 17:41:03,904 INFO Regime epoch 31/50 — tr=0.6291 va=1.2581 acc=0.809
2026-04-26 17:41:04,402 INFO Regime epoch 32/50 — tr=0.6283 va=1.2534 acc=0.812
2026-04-26 17:41:04,907 INFO Regime epoch 33/50 — tr=0.6276 va=1.2542 acc=0.815
2026-04-26 17:41:05,429 INFO Regime epoch 34/50 — tr=0.6275 va=1.2514 acc=0.815
2026-04-26 17:41:05,958 INFO Regime epoch 35/50 — tr=0.6269 va=1.2491 acc=0.815 per_class={'TRENDING': 0.783, 'RANGING': 0.77, 'CONSOLIDATING': 0.811, 'VOLATILE': 0.905}
2026-04-26 17:41:06,451 INFO Regime epoch 36/50 — tr=0.6266 va=1.2461 acc=0.818
2026-04-26 17:41:06,954 INFO Regime epoch 37/50 — tr=0.6263 va=1.2481 acc=0.818
2026-04-26 17:41:07,446 INFO Regime epoch 38/50 — tr=0.6258 va=1.2494 acc=0.821
2026-04-26 17:41:07,940 INFO Regime epoch 39/50 — tr=0.6259 va=1.2444 acc=0.820
2026-04-26 17:41:08,465 INFO Regime epoch 40/50 — tr=0.6254 va=1.2422 acc=0.820 per_class={'TRENDING': 0.792, 'RANGING': 0.777, 'CONSOLIDATING': 0.817, 'VOLATILE': 0.901}
2026-04-26 17:41:08,960 INFO Regime epoch 41/50 — tr=0.6252 va=1.2432 acc=0.822
2026-04-26 17:41:09,445 INFO Regime epoch 42/50 — tr=0.6252 va=1.2414 acc=0.820
2026-04-26 17:41:09,939 INFO Regime epoch 43/50 — tr=0.6253 va=1.2470 acc=0.822
2026-04-26 17:41:10,446 INFO Regime epoch 44/50 — tr=0.6250 va=1.2396 acc=0.820
2026-04-26 17:41:11,015 INFO Regime epoch 45/50 — tr=0.6251 va=1.2436 acc=0.822 per_class={'TRENDING': 0.793, 'RANGING': 0.774, 'CONSOLIDATING': 0.819, 'VOLATILE': 0.908}
2026-04-26 17:41:11,502 INFO Regime epoch 46/50 — tr=0.6247 va=1.2444 acc=0.822
2026-04-26 17:41:12,023 INFO Regime epoch 47/50 — tr=0.6246 va=1.2429 acc=0.823
2026-04-26 17:41:12,506 INFO Regime epoch 48/50 — tr=0.6244 va=1.2435 acc=0.822
2026-04-26 17:41:13,000 INFO Regime epoch 49/50 — tr=0.6248 va=1.2441 acc=0.822
2026-04-26 17:41:13,524 INFO Regime epoch 50/50 — tr=0.6251 va=1.2442 acc=0.822 per_class={'TRENDING': 0.794, 'RANGING': 0.769, 'CONSOLIDATING': 0.826, 'VOLATILE': 0.904}
2026-04-26 17:41:13,562 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 17:41:13,562 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 17:41:13,564 INFO Regime phase LTF train: 25.4s
2026-04-26 17:41:13,692 INFO Regime LTF complete: acc=0.820, n=401471 per_class={'TRENDING': 0.79, 'RANGING': 0.772, 'CONSOLIDATING': 0.825, 'VOLATILE': 0.901}
2026-04-26 17:41:13,696 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:41:14,205 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 17:41:14,209 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 9.355291913830783, 'RANGING': 6.096303199751476, 'CONSOLIDATING': 5.598885793871866, 'VOLATILE': 6.771351107094442}
2026-04-26 17:41:14,216 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 29965, 'mean': 2.149062259083323e-05, 'mean_over_std': 0.011155601434349563}, 'RANGING': {'n': 19623, 'mean': -3.254493608872846e-06, 'mean_over_std': -0.001761415548292494}, 'CONSOLIDATING': {'n': 10050, 'mean': -3.039756158146958e-06, 'mean_over_std': -0.001792488091618151}, 'VOLATILE': {'n': 14985, 'mean': 1.8102764458593193e-05, 'mean_over_std': 0.006974696291378742}}
2026-04-26 17:41:14,217 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 26145, 'mean': 2.874435605138053e-05, 'mean_over_std': 0.01595450632691917}, 'RANGING': {'n': 7858, 'mean': 1.7829776785521727e-05, 'mean_over_std': 0.011851561804467463}, 'CONSOLIDATING': {'n': 8543, 'mean': 7.845231961250924e-06, 'mean_over_std': 0.004915839521549481}, 'VOLATILE': {'n': 13636, 'mean': 5.480114267306668e-06, 'mean_over_std': 0.0020378119917389375}}
2026-04-26 17:41:14,217 INFO Regime retrain total: 211.2s (504761 samples)
2026-04-26 17:41:14,230 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 17:41:14,230 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 17:41:14,230 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 17:41:14,230 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-26 17:41:14,230 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-26 17:41:14,230 INFO Retrain complete. Total wall-clock: 211.2s
2026-04-26 17:41:16,743 INFO Model regime: SUCCESS
2026-04-26 17:41:16,743 INFO --- Training gru ---
2026-04-26 17:41:16,744 INFO Running retrain --model gru
2026-04-26 17:41:16,970 INFO retrain environment: KAGGLE
2026-04-26 17:41:18,548 INFO Device: CUDA (2 GPU(s))
2026-04-26 17:41:18,557 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 17:41:18,557 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 17:41:18,557 INFO cuDNN benchmark=True, TF32 matmul=True
2026-04-26 17:41:18,560 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-04-26 17:41:18,561 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-26 17:41:18,702 INFO NumExpr defaulting to 4 threads.
2026-04-26 17:41:18,893 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-26 17:41:18,893 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 17:41:18,893 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 17:41:18,893 INFO GRU phase macro_correlations: 0.0s
2026-04-26 17:41:18,893 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-26 17:41:18,894 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260426_174118
2026-04-26 17:41:18,896 WARNING WeightsManifest: no manifest at /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json — assuming compatible (run a full retrain to generate one)
2026-04-26 17:41:19,037 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:19,057 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:19,069 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:19,076 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:19,077 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-26 17:41:19,077 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-26 17:41:19,077 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-26 17:41:19,078 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-26 17:41:19,079 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:19,153 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1465, 'BIAS_DOWN': 1510, 'BIAS_NEUTRAL': 5427}  ambiguous=5175 (total=8402)  short_runs_zeroed=591
2026-04-26 17:41:19,155 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:19,375 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=9161 (total=32738)  short_runs_zeroed=4986
2026-04-26 17:41:19,404 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:19,662 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:19,791 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:19,888 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:20,086 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:20,104 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:20,117 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:20,123 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:20,124 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:20,197 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1511, 'BIAS_DOWN': 1252, 'BIAS_NEUTRAL': 5639}  ambiguous=5684 (total=8402)  short_runs_zeroed=726
2026-04-26 17:41:20,199 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:20,427 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=8659 (total=32738)  short_runs_zeroed=4347
2026-04-26 17:41:20,444 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:20,699 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:20,828 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:20,919 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:21,096 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:21,116 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:21,131 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:21,137 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:21,138 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:21,210 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1437, 'BIAS_DOWN': 1457, 'BIAS_NEUTRAL': 5508}  ambiguous=5360 (total=8402)  short_runs_zeroed=675
2026-04-26 17:41:21,211 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:21,431 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=8676 (total=32740)  short_runs_zeroed=4399
2026-04-26 17:41:21,447 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:21,721 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:21,860 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:21,959 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:22,147 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:22,166 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:22,179 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:22,186 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:22,186 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:22,264 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1523, 'BIAS_DOWN': 1428, 'BIAS_NEUTRAL': 5451}  ambiguous=5266 (total=8402)  short_runs_zeroed=681
2026-04-26 17:41:22,266 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:22,500 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=8202 (total=32739)  short_runs_zeroed=3955
2026-04-26 17:41:22,522 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:22,794 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:22,921 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:23,020 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:23,198 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:23,217 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:23,231 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:23,238 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:23,239 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:23,314 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1384, 'BIAS_DOWN': 1626, 'BIAS_NEUTRAL': 5393}  ambiguous=5134 (total=8403)  short_runs_zeroed=577
2026-04-26 17:41:23,316 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:23,539 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=8551 (total=32740)  short_runs_zeroed=4397
2026-04-26 17:41:23,554 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:23,815 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:23,943 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:24,035 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:24,209 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:24,227 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:24,240 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:24,246 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:24,247 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:24,322 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1547, 'BIAS_DOWN': 1440, 'BIAS_NEUTRAL': 5416}  ambiguous=5307 (total=8403)  short_runs_zeroed=774
2026-04-26 17:41:24,324 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:24,554 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=8016 (total=32739)  short_runs_zeroed=3724
2026-04-26 17:41:24,569 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:24,849 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:24,972 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:25,081 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:25,238 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:41:25,254 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:41:25,267 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:41:25,274 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-26 17:41:25,275 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:25,378 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1483, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 5465}  ambiguous=5312 (total=8402)  short_runs_zeroed=629
2026-04-26 17:41:25,379 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:25,594 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=8604 (total=32739)  short_runs_zeroed=4898
2026-04-26 17:41:25,606 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:25,862 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:25,982 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:26,078 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:26,250 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:26,268 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:26,281 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:26,288 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:26,288 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:26,365 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1285, 'BIAS_DOWN': 1704, 'BIAS_NEUTRAL': 5413}  ambiguous=5199 (total=8402)  short_runs_zeroed=615
2026-04-26 17:41:26,367 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:26,593 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=7757 (total=32740)  short_runs_zeroed=3880
2026-04-26 17:41:26,611 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:26,870 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:26,996 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:27,087 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:27,262 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:27,280 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:27,295 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:27,301 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:27,302 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:27,378 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1344, 'BIAS_DOWN': 1614, 'BIAS_NEUTRAL': 5444}  ambiguous=5368 (total=8402)  short_runs_zeroed=616
2026-04-26 17:41:27,379 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:27,604 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=8293 (total=32741)  short_runs_zeroed=3896
2026-04-26 17:41:27,618 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:27,890 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:28,014 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:28,114 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:28,296 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:28,316 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:28,331 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:28,337 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-26 17:41:28,338 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:28,412 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 1305, 'BIAS_DOWN': 1589, 'BIAS_NEUTRAL': 5509}  ambiguous=5275 (total=8403)  short_runs_zeroed=589
2026-04-26 17:41:28,414 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:28,643 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=8252 (total=32743)  short_runs_zeroed=4275
2026-04-26 17:41:28,658 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:28,926 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:29,052 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:29,146 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-26 17:41:29,424 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:41:29,449 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:41:29,464 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:41:29,473 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-26 17:41:29,474 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:41:29,628 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 4338, 'BIAS_DOWN': 3212, 'BIAS_NEUTRAL': 12267}  ambiguous=12096 (total=19817)  short_runs_zeroed=1542
2026-04-26 17:41:29,631 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:41:30,158 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=18441 (total=74624)  short_runs_zeroed=9134
2026-04-26 17:41:30,203 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:41:30,715 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:41:30,914 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:41:31,046 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-26 17:41:31,163 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-26 17:41:31,427 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-04-26 17:41:31,427 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-04-26 17:41:31,427 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-26 17:41:31,427 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-26 17:42:18,853 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-26 17:42:18,853 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-26 17:42:20,178 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
2026-04-26 17:42:24,226 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=5)
2026-04-26 17:42:37,987 INFO train_multi TF=ALL epoch 1/50 train=0.8979 val=0.8901
2026-04-26 17:42:37,995 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 17:42:37,996 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 17:42:37,996 INFO train_multi TF=ALL: new best val=0.8901 — saved
2026-04-26 17:42:49,609 INFO train_multi TF=ALL epoch 2/50 train=0.8804 val=0.8570
2026-04-26 17:42:49,614 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 17:42:49,614 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 17:42:49,614 INFO train_multi TF=ALL: new best val=0.8570 — saved
2026-04-26 17:43:01,280 INFO train_multi TF=ALL epoch 3/50 train=0.7674 val=0.6896
2026-04-26 17:43:01,284 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 17:43:01,284 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 17:43:01,284 INFO train_multi TF=ALL: new best val=0.6896 — saved
2026-04-26 17:43:12,861 INFO train_multi TF=ALL epoch 4/50 train=0.6921 val=0.6878
2026-04-26 17:43:12,865 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-26 17:43:12,865 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 17:43:12,865 INFO train_multi TF=ALL: new best val=0.6878 — saved
2026-04-26 17:43:24,575 INFO train_multi TF=ALL epoch 5/50 train=0.6904 val=0.6880
2026-04-26 17:43:36,314 INFO train_multi TF=ALL epoch 6/50 train=0.6896 val=0.6883
2026-04-26 17:43:47,923 INFO train_multi TF=ALL epoch 7/50 train=0.6894 val=0.6883
2026-04-26 17:43:59,560 INFO train_multi TF=ALL epoch 8/50 train=0.6891 val=0.6888
2026-04-26 17:44:11,181 INFO train_multi TF=ALL epoch 9/50 train=0.6888 val=0.6880
2026-04-26 17:44:11,181 INFO train_multi TF=ALL early stop at epoch 9
2026-04-26 17:44:11,316 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-04-26 17:44:11,316 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-26 17:44:11,316 INFO Retrain complete. Total wall-clock: 172.8s
2026-04-26 17:44:13,218 INFO Model gru: SUCCESS
2026-04-26 17:44:13,219 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-26 17:44:13,219 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-26 17:44:13,219 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-26 17:44:13,219 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-04-26 17:44:13,219 INFO   [DEFERRED] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip (expected after Round 1)
2026-04-26 17:44:13,219 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-04-26 17:44:13,219 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer', 'rl_ppo']
2026-04-26 17:44:13,220 INFO Saved 5 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  START Round 1 - Backtest (val)
2026-04-26 17:44:13,817 INFO === STEP 6: BACKTEST (round1) ===
2026-04-26 17:44:13,818 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-04-26 17:44:13,818 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-26 17:44:13,818 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
2026-04-26 17:44:16,171 WARNING QualityScorer unavailable (weights missing or load failed)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260426_174415.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)                0   0.0%   0.00    0.0%  0.0%  0.0%   0.0%    0.00
  gate_diagnostics: bars=468696 no_signal=150801 quality_block=0 session_skip=317895 density=0 pm_reject=0 daily_skip=0 cooldown=0

Calibration Summary:
  all          [N/A] No outcome data yet
2026-04-26 17:46:27,808 INFO Round 1 backtest — 0 trades | avg WR=0.0% | avg PF=0.00 | avg Sharpe=0.00
2026-04-26 17:46:27,808 INFO   ml_trader: 0 trades | WR=0.0% | PF=0.00 | Return=0.0% | DD=0.0% | Sharpe=0.00
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 0
ERROR  trade_log is empty.
2026-04-26 17:46:28,026 WARNING Round 1: trade_log is empty — nothing to journal
2026-04-26 17:46:28,026 WARNING Round 1: no trades to journal

======================================================================
  BACKTEST COMPLETE  (round 1 / window=round1)
======================================================================
  Round     Trades       WR      PF   Sharpe
  ------------------------------------------
  Round 1          0      0.0%    0.000     0.000

  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 0 entries

=== Round 1 → Retrain Quality + RL ===
  START Round 1 - Quality+RL retrain
2026-04-26 17:46:28,234 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-26 17:46:28,234 WARNING Journal missing or empty at /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/trade_journal_detailed.jsonl — backtest produced no trades yet. Skipping Quality+RL training (will train after first successful backtest).
  DONE  Round 1 - Quality+RL retrain

=== Round 2: BLIND backtest on test window (unseen 2yr) ===
  START Round 2 - Blind backtest (test)
2026-04-26 17:46:28,723 INFO === STEP 6: BACKTEST (round2) ===
2026-04-26 17:46:28,724 INFO BT_WINDOW=round2 — BLIND backtest: 2023-08-07 → 2025-08-05 (test set)
2026-04-26 17:46:28,724 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-26 17:46:28,724 INFO Round 2 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
2026-04-26 17:46:31,051 WARNING QualityScorer unavailable (weights missing or load failed)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260426_174630.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)                0   0.0%   0.00    0.0%  0.0%  0.0%   0.0%    0.00
  gate_diagnostics: bars=482221 no_signal=120986 quality_block=0 session_skip=361235 density=0 pm_reject=0 daily_skip=0 cooldown=0

Calibration Summary:
  all          [N/A] No outcome data yet
2026-04-26 17:48:42,364 INFO Round 2 backtest — 0 trades | avg WR=0.0% | avg PF=0.00 | avg Sharpe=0.00
2026-04-26 17:48:42,364 INFO   ml_trader: 0 trades | WR=0.0% | PF=0.00 | Return=0.0% | DD=0.0% | Sharpe=0.00
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 0
ERROR  trade_log is empty.
2026-04-26 17:48:42,582 WARNING Round 2: trade_log is empty — nothing to journal
2026-04-26 17:48:42,582 WARNING Round 2: no trades to journal

======================================================================
  BACKTEST COMPLETE  (round 2 / window=round2)
======================================================================
  Round     Trades       WR      PF   Sharpe
  ------------------------------------------
  Round 2          0      0.0%    0.000     0.000

  DONE  Round 2 - Blind backtest (test)
  Saved Round 2 result → round2_summary.json
  Journal after Round 2: 0 entries

=== Round 2 → Retrain Quality + RL (Round 1+2 journal) ===
  START Round 2 - Quality+RL retrain
2026-04-26 17:48:42,796 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-26 17:48:42,796 WARNING Journal missing or empty at /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/trade_journal_detailed.jsonl — backtest produced no trades yet. Skipping Quality+RL training (will train after first successful backtest).
2026-04-26 17:48:42,983 INFO retrain environment: KAGGLE
  DONE  Round 2 - Quality+RL retrain