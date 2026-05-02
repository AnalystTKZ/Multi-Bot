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
2026-05-02 15:24:44,800 INFO Loading feature-engineered data...
2026-05-02 15:24:45,426 INFO Loaded 221743 rows, 202 features
2026-05-02 15:24:45,427 INFO Data span: 2016-01-04 -> 2025-08-05  (9.6 years)
2026-05-02 15:24:45,429 INFO Fold 000 train 2016-01-04 -> 2018-01-03 (47088 bars), val 2018-01-04 -> 2019-01-03 (23448 bars)
2026-05-02 15:24:45,429 INFO Fold 001 train 2018-01-04 -> 2020-01-03 (46825 bars), val 2020-01-06 -> 2020-12-31 (23259 bars)
2026-05-02 15:24:45,430 INFO Fold 002 train 2020-01-06 -> 2022-01-03 (46766 bars), val 2022-01-04 -> 2023-01-03 (23588 bars)
2026-05-02 15:24:45,430 INFO No leakage confirmed: every fold ends before final 2-year blind test

=== SPLIT COMPLETE (ROLLING CALENDAR, no shuffling) ===
  Folds:            3 rolling folds (2y train + 1y val, step=2y)
  Selected:   fold_002 for train.parquet / validation.parquet aliases
  Train:       46,766 bars  2020-01-06 -> 2022-01-03
  Validation:  23,588 bars  2022-01-04 -> 2023-01-03
  Test:        46,792 bars  2023-08-07 -> 2025-08-05  <- Blind / Round 2
  Features:   202
  Leakage check: PASS
  DONE  Step 5 - Split

  Data split (rolling_calendar):
    train          46766 bars  2020-01-06 → 2022-01-03
    validation     23588 bars  2022-01-04 → 2023-01-03
    test           46792 bars  2023-08-07 → 2025-08-05

=== Phase 7a: Train GRU + Regime (train set only) ===
  START Step 7a - GRU+Regime
2026-05-02 15:24:49,134 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-05-02 15:24:49,134 INFO --- Training regime ---
2026-05-02 15:24:49,135 INFO Running retrain --model regime
2026-05-02 15:24:49,330 INFO retrain environment: KAGGLE
2026-05-02 15:24:51,135 INFO Device: CUDA (2 GPU(s))
2026-05-02 15:24:51,146 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-02 15:24:51,146 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-02 15:24:51,146 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-02 15:24:51,148 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-02 15:24:51,148 INFO Retrain data split: train
2026-05-02 15:24:51,148 INFO Retrain rolling fold selector: latest
2026-05-02 15:24:51,149 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 5-score behaviour) ===
2026-05-02 15:24:51,328 INFO NumExpr defaulting to 4 threads.
2026-05-02 15:24:51,568 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-02 15:24:51,568 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-02 15:24:51,568 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-02 15:24:51,568 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-02 15:24:51,630 INFO Regime rolling folds selected: ['fold_000', 'fold_001', 'fold_002']
2026-05-02 15:24:51,630 INFO === Regime rolling fold 1/3: fold_000 ===
2026-05-02 15:24:51,630 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-05-02 15:24:51,669 INFO Split boundaries loaded fold=fold_000/3 — train 2016-01-04→2018-01-03  val 2018-01-04→2019-01-03  test 2023-08-07→2025-08-05
2026-05-02 15:24:51,670 INFO Loaded AUDUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:24:51,690 INFO Loaded EURGBP/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:24:51,706 INFO Loaded EURJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:24:51,724 INFO Loaded EURUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:24:51,743 INFO Loaded GBPJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:24:51,760 INFO Loaded GBPUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:24:51,777 INFO Loaded NZDUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:24:51,794 INFO Loaded USDCAD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:24:51,810 INFO Loaded USDCHF/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:24:51,828 INFO Loaded USDJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:24:51,849 INFO Loaded XAUUSD/4H split=train fold=fold_000: 3204 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:24:51,991 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:52,037 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:52,059 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:52,059 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:52,068 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:52,069 INFO Loaded AUDUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:24:52,283 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 92, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 2876}  ambiguous=1700 (total=3023) horizon=12
2026-05-02 15:24:52,285 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected AUDUSD — 2973 samples (group=dollar) labels={'BIAS_UP': 92, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 2826} clean={'BIAS_UP': 92, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 1152}
2026-05-02 15:24:52,473 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:52,513 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:52,533 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:52,534 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:52,542 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:52,544 INFO Loaded EURGBP/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:24:52,771 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 156, 'BIAS_DOWN': 71, 'BIAS_NEUTRAL': 2796}  ambiguous=1710 (total=3023) horizon=12
2026-05-02 15:24:52,773 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected EURGBP — 2973 samples (group=cross) labels={'BIAS_UP': 156, 'BIAS_DOWN': 71, 'BIAS_NEUTRAL': 2746} clean={'BIAS_UP': 156, 'BIAS_DOWN': 71, 'BIAS_NEUTRAL': 1071}
2026-05-02 15:24:52,990 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:53,038 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:53,059 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:53,059 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:53,068 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:53,069 INFO Loaded EURJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:24:53,297 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 95, 'BIAS_DOWN': 63, 'BIAS_NEUTRAL': 2865}  ambiguous=1742 (total=3023) horizon=12
2026-05-02 15:24:53,298 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected EURJPY — 2973 samples (group=cross) labels={'BIAS_UP': 95, 'BIAS_DOWN': 63, 'BIAS_NEUTRAL': 2815} clean={'BIAS_UP': 95, 'BIAS_DOWN': 63, 'BIAS_NEUTRAL': 1099}
2026-05-02 15:24:53,490 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:53,532 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:53,552 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:53,552 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:53,560 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:53,561 INFO Loaded EURUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:24:53,764 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 98, 'BIAS_DOWN': 57, 'BIAS_NEUTRAL': 2868}  ambiguous=1742 (total=3023) horizon=12
2026-05-02 15:24:53,766 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected EURUSD — 2973 samples (group=dollar) labels={'BIAS_UP': 98, 'BIAS_DOWN': 57, 'BIAS_NEUTRAL': 2818} clean={'BIAS_UP': 98, 'BIAS_DOWN': 57, 'BIAS_NEUTRAL': 1105}
2026-05-02 15:24:53,938 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:53,978 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:53,998 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:53,998 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:54,006 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:54,007 INFO Loaded GBPJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:24:54,209 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 164, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2758}  ambiguous=1723 (total=3023) horizon=12
2026-05-02 15:24:54,211 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected GBPJPY — 2973 samples (group=cross) labels={'BIAS_UP': 164, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2708} clean={'BIAS_UP': 164, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 1019}
2026-05-02 15:24:54,384 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:54,426 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:54,448 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:54,449 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:54,457 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:54,459 INFO Loaded GBPUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:24:54,676 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 79, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2843}  ambiguous=1759 (total=3023) horizon=12
2026-05-02 15:24:54,677 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected GBPUSD — 2973 samples (group=dollar) labels={'BIAS_UP': 79, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2793} clean={'BIAS_UP': 79, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 1073}
2026-05-02 15:24:54,836 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:24:54,870 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:24:54,890 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:24:54,890 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:24:54,898 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:24:54,899 INFO Loaded NZDUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:24:55,113 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 54, 'BIAS_DOWN': 54, 'BIAS_NEUTRAL': 2915}  ambiguous=1779 (total=3023) horizon=12
2026-05-02 15:24:55,115 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected NZDUSD — 2973 samples (group=dollar) labels={'BIAS_UP': 54, 'BIAS_DOWN': 54, 'BIAS_NEUTRAL': 2865} clean={'BIAS_UP': 54, 'BIAS_DOWN': 54, 'BIAS_NEUTRAL': 1117}
2026-05-02 15:24:55,297 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:55,335 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:55,370 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:55,370 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:55,381 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:55,382 INFO Loaded USDCAD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:24:55,606 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 89, 'BIAS_DOWN': 133, 'BIAS_NEUTRAL': 2801}  ambiguous=1770 (total=3023) horizon=12
2026-05-02 15:24:55,608 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected USDCAD — 2973 samples (group=dollar) labels={'BIAS_UP': 89, 'BIAS_DOWN': 133, 'BIAS_NEUTRAL': 2751} clean={'BIAS_UP': 89, 'BIAS_DOWN': 133, 'BIAS_NEUTRAL': 1016}
2026-05-02 15:24:55,788 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:55,826 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:55,846 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:55,846 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:55,855 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:55,856 INFO Loaded USDCHF/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:24:56,055 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 64, 'BIAS_DOWN': 52, 'BIAS_NEUTRAL': 2907}  ambiguous=1741 (total=3023) horizon=12
2026-05-02 15:24:56,056 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected USDCHF — 2973 samples (group=dollar) labels={'BIAS_UP': 64, 'BIAS_DOWN': 52, 'BIAS_NEUTRAL': 2857} clean={'BIAS_UP': 64, 'BIAS_DOWN': 52, 'BIAS_NEUTRAL': 1148}
2026-05-02 15:24:56,229 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:56,267 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:56,287 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:56,287 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:56,295 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:56,296 INFO Loaded USDJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:24:56,499 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 102, 'BIAS_DOWN': 78, 'BIAS_NEUTRAL': 2843}  ambiguous=1762 (total=3023) horizon=12
2026-05-02 15:24:56,501 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected USDJPY — 2973 samples (group=dollar) labels={'BIAS_UP': 102, 'BIAS_DOWN': 78, 'BIAS_NEUTRAL': 2793} clean={'BIAS_UP': 102, 'BIAS_DOWN': 78, 'BIAS_NEUTRAL': 1058}
2026-05-02 15:24:56,795 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:24:56,861 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:24:56,886 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:24:56,887 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:24:56,898 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:24:56,899 INFO Loaded XAUUSD/4H split=train fold=fold_000: 3204 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:24:57,121 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 2845}  ambiguous=1801 (total=3204) horizon=12
2026-05-02 15:24:57,123 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected XAUUSD — 3154 samples (group=gold) labels={'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 2795} clean={'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 1029}
2026-05-02 15:24:57,199 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 415, 'BIAS_DOWN': 235, 'BIAS_NEUTRAL': 8269}, 'dollar': {'BIAS_UP': 578, 'BIAS_DOWN': 530, 'BIAS_NEUTRAL': 19703}, 'gold': {'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 2795}}
2026-05-02 15:24:57,199 INFO Regime[4H mode=htf_bias] label distribution by year: {2016: {'BIAS_UP': 485, 'BIAS_DOWN': 511, 'BIAS_NEUTRAL': 15101}, 2017: {'BIAS_UP': 717, 'BIAS_DOWN': 401, 'BIAS_NEUTRAL': 15515}, 2018: {'BIAS_UP': 3, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 151}}
2026-05-02 15:24:57,276 INFO Loaded AUDUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:24:57,277 INFO Loaded EURGBP/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:24:57,278 INFO Loaded EURJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:24:57,279 INFO Loaded EURUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:24:57,280 INFO Loaded GBPJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:24:57,281 INFO Loaded GBPUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:24:57,282 INFO Loaded NZDUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:24:57,282 INFO Loaded USDCAD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:24:57,283 INFO Loaded USDCHF/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:24:57,284 INFO Loaded USDJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:24:57,285 INFO Loaded XAUUSD/4H split=val fold=fold_000: 1600 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:24:57,291 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:57,294 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:57,295 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:57,295 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:57,295 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:57,296 INFO Loaded AUDUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:24:57,477 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 5, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1448}  ambiguous=896 (total=1506) horizon=12
2026-05-02 15:24:57,478 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected AUDUSD — 1456 samples (group=dollar) labels={'BIAS_UP': 5, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1398} clean={'BIAS_UP': 5, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 531}
2026-05-02 15:24:57,561 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:57,564 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:57,565 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:57,565 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:57,566 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:57,567 INFO Loaded EURGBP/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:24:57,747 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 12, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1453}  ambiguous=868 (total=1506) horizon=12
2026-05-02 15:24:57,749 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected EURGBP — 1456 samples (group=cross) labels={'BIAS_UP': 12, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1403} clean={'BIAS_UP': 12, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 575}
2026-05-02 15:24:57,831 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:57,834 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:57,834 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:57,835 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:57,835 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:57,836 INFO Loaded EURJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:24:58,009 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 20, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1445}  ambiguous=874 (total=1506) horizon=12
2026-05-02 15:24:58,011 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected EURJPY — 1456 samples (group=cross) labels={'BIAS_UP': 20, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1395} clean={'BIAS_UP': 20, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 555}
2026-05-02 15:24:58,086 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:58,088 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:58,089 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:58,090 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:58,090 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:58,091 INFO Loaded EURUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:24:58,271 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 24, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 1415}  ambiguous=876 (total=1506) horizon=12
2026-05-02 15:24:58,272 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected EURUSD — 1456 samples (group=dollar) labels={'BIAS_UP': 24, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 1365} clean={'BIAS_UP': 24, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 522}
2026-05-02 15:24:58,357 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:58,360 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:58,361 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:58,361 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:58,362 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:58,363 INFO Loaded GBPJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:24:58,542 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 27, 'BIAS_DOWN': 37, 'BIAS_NEUTRAL': 1442}  ambiguous=926 (total=1506) horizon=12
2026-05-02 15:24:58,544 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected GBPJPY — 1456 samples (group=cross) labels={'BIAS_UP': 27, 'BIAS_DOWN': 37, 'BIAS_NEUTRAL': 1392} clean={'BIAS_UP': 27, 'BIAS_DOWN': 37, 'BIAS_NEUTRAL': 506}
2026-05-02 15:24:58,628 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:58,631 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:58,632 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:58,632 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:58,633 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:58,634 INFO Loaded GBPUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:24:58,823 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 50, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1371}  ambiguous=874 (total=1506) horizon=12
2026-05-02 15:24:58,825 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected GBPUSD — 1456 samples (group=dollar) labels={'BIAS_UP': 50, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1321} clean={'BIAS_UP': 50, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 476}
2026-05-02 15:24:58,908 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:24:58,909 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:24:58,910 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:24:58,911 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:24:58,911 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:24:58,912 INFO Loaded NZDUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:24:59,098 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 23, 'BIAS_DOWN': 80, 'BIAS_NEUTRAL': 1403}  ambiguous=896 (total=1506) horizon=12
2026-05-02 15:24:59,099 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected NZDUSD — 1456 samples (group=dollar) labels={'BIAS_UP': 23, 'BIAS_DOWN': 80, 'BIAS_NEUTRAL': 1353} clean={'BIAS_UP': 23, 'BIAS_DOWN': 80, 'BIAS_NEUTRAL': 482}
2026-05-02 15:24:59,186 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:59,188 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:59,189 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:59,189 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:59,190 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:59,191 INFO Loaded USDCAD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:24:59,372 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 48, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 1445}  ambiguous=907 (total=1506) horizon=12
2026-05-02 15:24:59,373 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected USDCAD — 1456 samples (group=dollar) labels={'BIAS_UP': 48, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 1395} clean={'BIAS_UP': 48, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 522}
2026-05-02 15:24:59,458 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:59,461 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:59,462 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:59,462 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:59,463 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:59,464 INFO Loaded USDCHF/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:24:59,675 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 68, 'BIAS_DOWN': 45, 'BIAS_NEUTRAL': 1393}  ambiguous=848 (total=1506) horizon=12
2026-05-02 15:24:59,677 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected USDCHF — 1456 samples (group=dollar) labels={'BIAS_UP': 68, 'BIAS_DOWN': 45, 'BIAS_NEUTRAL': 1343} clean={'BIAS_UP': 68, 'BIAS_DOWN': 45, 'BIAS_NEUTRAL': 530}
2026-05-02 15:24:59,768 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:59,770 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:59,771 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:59,772 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:59,772 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:24:59,773 INFO Loaded USDJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:24:59,956 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 58, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1418}  ambiguous=888 (total=1506) horizon=12
2026-05-02 15:24:59,957 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected USDJPY — 1456 samples (group=dollar) labels={'BIAS_UP': 58, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1368} clean={'BIAS_UP': 58, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 510}
2026-05-02 15:25:00,053 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:25:00,056 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:25:00,058 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:25:00,058 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:25:00,059 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:25:00,060 INFO Loaded XAUUSD/4H split=val fold=fold_000: 1600 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:25:00,253 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 18, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 1547}  ambiguous=851 (total=1600) horizon=12
2026-05-02 15:25:00,255 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected XAUUSD — 1550 samples (group=gold) labels={'BIAS_UP': 18, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 1497} clean={'BIAS_UP': 18, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 677}
2026-05-02 15:25:00,333 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 59, 'BIAS_DOWN': 119, 'BIAS_NEUTRAL': 4190}, 'dollar': {'BIAS_UP': 276, 'BIAS_DOWN': 373, 'BIAS_NEUTRAL': 9543}, 'gold': {'BIAS_UP': 18, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 1497}}
2026-05-02 15:25:00,333 INFO Regime[4H mode=htf_bias] label distribution by year: {2018: {'BIAS_UP': 352, 'BIAS_DOWN': 521, 'BIAS_NEUTRAL': 15083}, 2019: {'BIAS_UP': 1, 'BIAS_DOWN': 6, 'BIAS_NEUTRAL': 147}}
2026-05-02 15:25:00,410 INFO Regime phase HTF dataset build fold=fold_000: 8.8s (train=32884 val=16110)
2026-05-02 15:25:00,411 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=14004 dropped=18880 classes={'BIAS_UP': 1205, 'BIAS_DOWN': 912, 'BIAS_NEUTRAL': 11887})
2026-05-02 15:25:00,413 INFO RegimeClassifier[mode=htf_bias]: 14004 samples, classes={'BIAS_UP': 1205, 'BIAS_DOWN': 912, 'BIAS_NEUTRAL': 11887}, device=cuda
2026-05-02 15:25:00,413 INFO RegimeClassifier[mode=htf_bias]: undersample class BIAS_NEUTRAL: 11887 → 10944
2026-05-02 15:25:00,414 INFO RegimeClassifier[mode=htf_bias]: after undersampling: 13061 samples classes={'BIAS_UP': 1205, 'BIAS_DOWN': 912, 'BIAS_NEUTRAL': 10944}
2026-05-02 15:25:00,414 INFO RegimeClassifier: sample weights — mean=0.680  ambiguous(<0.4)=0.0%
2026-05-02 15:25:00,670 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-05-02 15:25:00,671 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-05-02 15:25:05,003 INFO Regime epoch  1/50 — tr=1.0995 va=0.6132 acc=0.945 bal=0.333 per_class={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 1.0}
2026-05-02 15:25:05,047 INFO Regime epoch  2/50 — tr=1.1006 va=0.6433 acc=0.945 bal=0.333
2026-05-02 15:25:05,088 INFO Regime epoch  3/50 — tr=1.0899 va=0.6459 acc=0.945 bal=0.333
2026-05-02 15:25:05,133 INFO Regime epoch  4/50 — tr=1.0917 va=0.6428 acc=0.945 bal=0.333
2026-05-02 15:25:05,179 INFO Regime epoch  5/50 — tr=1.0739 va=0.6401 acc=0.945 bal=0.333 per_class={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 1.0}
2026-05-02 15:25:05,224 INFO Regime epoch  6/50 — tr=1.0604 va=0.6366 acc=0.945 bal=0.333
2026-05-02 15:25:05,268 INFO Regime epoch  7/50 — tr=1.0369 va=0.6323 acc=0.945 bal=0.333
2026-05-02 15:25:05,313 INFO Regime epoch  8/50 — tr=1.0084 va=0.6275 acc=0.945 bal=0.333
2026-05-02 15:25:05,358 INFO Regime epoch  9/50 — tr=0.9853 va=0.6212 acc=0.945 bal=0.333
2026-05-02 15:25:05,401 INFO Regime epoch 10/50 — tr=0.9463 va=0.6144 acc=0.945 bal=0.333 per_class={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 1.0}
2026-05-02 15:25:05,443 INFO Regime epoch 11/50 — tr=0.9107 va=0.6082 acc=0.945 bal=0.333
2026-05-02 15:25:05,443 INFO Regime early stop at epoch 11 (no_improve=10)
2026-05-02 15:25:06,166 INFO RegimeClassifier[mode=htf_bias] selected HTF decision policy threshold=0.400 margin=0.200 policy_accuracy=0.945 policy_balanced=0.333
2026-05-02 15:25:06,179 INFO RegimeClassifier[mode=htf_bias] validation precision={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 0.945} recall={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 1.0} f1={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 0.972} confusion=[[0, 0, 353], [0, 0, 527], [0, 0, 15230]]
2026-05-02 15:25:06,180 INFO Regime phase HTF train fold=fold_000: 5.8s
Traceback (most recent call last):
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1707, in <module>
    main()
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1666, in main
    result = retrain_regime(dry)
             ^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1209, in retrain_regime
    raise RuntimeError(f"Regime HTF training failed fold={fold_key}: {res_4h['error']}")
RuntimeError: Regime HTF training failed fold=fold_000: Regime prediction distribution collapsed: pred_share={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 1.0}, max_pred_share=100.0%, collapsed_classes=['BIAS_UP', 'BIAS_DOWN']. Refusing to save misleading regime weights.

=== TRAINING COMPLETE ===
  regime: FAILED: exit 1
2026-05-02 15:25:08,480 ERROR retrain regime failed (exit 1)
2026-05-02 15:25:08,480 ERROR Model regime failed: exit 1
2026-05-02 15:25:08,480 WARNING   [MISSING] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 15:25:08,481 WARNING   [MISSING] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-02 15:25:08,481 WARNING   [MISSING] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-02 15:25:08,481 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-05-02 15:25:08,481 INFO   [DEFERRED] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip (expected after Round 1)
2026-05-02 15:25:08,481 WARNING Missing required weights: ['gru_lstm', 'regime_htf', 'regime_ltf'] — run retrain_incremental.py for each
2026-05-02 15:25:08,481 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer', 'rl_ppo']
2026-05-02 15:25:08,481 WARNING No retrain_history.jsonl found
2026-05-02 15:25:08,481 ERROR Step 7a failed; required training/artifacts missing: ['gru_lstm', 'regime', 'regime_htf', 'regime_ltf']
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/kaggle/working/Multi-Bot/trading-system/kaggle_train.py in <module>
    296 
    297 print("\n=== Phase 7a: Train GRU + Regime (train set only) ===")
--> 298 run_step(
    299     "Step 7a - GRU+Regime",
    300     "step7_train.py",

/kaggle/working/Multi-Bot/trading-system/kaggle_train.py in run_step(name, script, done_check, extra_env)
    186     )
    187     if result.returncode != 0:
--> 188         raise RuntimeError(f"{name} FAILED (exit {result.returncode})")
    189     print(f"  DONE  {name}")
    190 

RuntimeError: Step 7a - GRU+Regime FAILED (exit 1)