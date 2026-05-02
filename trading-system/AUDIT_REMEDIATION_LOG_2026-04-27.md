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
2026-05-02 15:05:09,439 INFO Loading feature-engineered data...
2026-05-02 15:05:10,067 INFO Loaded 221743 rows, 202 features
2026-05-02 15:05:10,068 INFO Data span: 2016-01-04 -> 2025-08-05  (9.6 years)
2026-05-02 15:05:10,071 INFO Fold 000 train 2016-01-04 -> 2018-01-03 (47088 bars), val 2018-01-04 -> 2019-01-03 (23448 bars)
2026-05-02 15:05:10,071 INFO Fold 001 train 2018-01-04 -> 2020-01-03 (46825 bars), val 2020-01-06 -> 2020-12-31 (23259 bars)
2026-05-02 15:05:10,071 INFO Fold 002 train 2020-01-06 -> 2022-01-03 (46766 bars), val 2022-01-04 -> 2023-01-03 (23588 bars)
2026-05-02 15:05:10,072 INFO No leakage confirmed: every fold ends before final 2-year blind test

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
2026-05-02 15:05:13,506 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-05-02 15:05:13,506 INFO --- Training regime ---
2026-05-02 15:05:13,506 INFO Running retrain --model regime
2026-05-02 15:05:13,691 INFO retrain environment: KAGGLE
2026-05-02 15:05:15,343 INFO Device: CUDA (2 GPU(s))
2026-05-02 15:05:15,354 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-02 15:05:15,354 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-02 15:05:15,354 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-02 15:05:15,356 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-02 15:05:15,356 INFO Retrain data split: train
2026-05-02 15:05:15,356 INFO Retrain rolling fold selector: latest
2026-05-02 15:05:15,357 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 5-score behaviour) ===
2026-05-02 15:05:15,521 INFO NumExpr defaulting to 4 threads.
2026-05-02 15:05:15,738 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-02 15:05:15,738 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-02 15:05:15,738 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-02 15:05:15,738 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-02 15:05:15,791 INFO Regime rolling folds selected: ['fold_000', 'fold_001', 'fold_002']
2026-05-02 15:05:15,791 INFO === Regime rolling fold 1/3: fold_000 ===
2026-05-02 15:05:15,791 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-05-02 15:05:15,829 INFO Split boundaries loaded fold=fold_000/3 — train 2016-01-04→2018-01-03  val 2018-01-04→2019-01-03  test 2023-08-07→2025-08-05
2026-05-02 15:05:15,830 INFO Loaded AUDUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:05:15,845 INFO Loaded EURGBP/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:05:15,860 INFO Loaded EURJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:05:15,875 INFO Loaded EURUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:05:15,892 INFO Loaded GBPJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:05:15,919 INFO Loaded GBPUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:05:15,943 INFO Loaded NZDUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:05:15,968 INFO Loaded USDCAD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:05:15,984 INFO Loaded USDCHF/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:05:15,999 INFO Loaded USDJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:05:16,016 INFO Loaded XAUUSD/4H split=train fold=fold_000: 3204 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:05:16,157 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:16,200 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:16,225 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:16,225 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:16,233 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:16,234 INFO Loaded AUDUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:05:16,461 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 92, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 2876}  ambiguous=1700 (total=3023) horizon=12
2026-05-02 15:05:16,463 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected AUDUSD — 2973 samples (group=dollar) labels={'BIAS_UP': 92, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 2826} clean={'BIAS_UP': 92, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 1152}
2026-05-02 15:05:16,627 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:16,662 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:16,681 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:16,681 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:16,689 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:16,690 INFO Loaded EURGBP/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:05:16,880 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 156, 'BIAS_DOWN': 71, 'BIAS_NEUTRAL': 2796}  ambiguous=1710 (total=3023) horizon=12
2026-05-02 15:05:16,881 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected EURGBP — 2973 samples (group=cross) labels={'BIAS_UP': 156, 'BIAS_DOWN': 71, 'BIAS_NEUTRAL': 2746} clean={'BIAS_UP': 156, 'BIAS_DOWN': 71, 'BIAS_NEUTRAL': 1071}
2026-05-02 15:05:17,051 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:17,090 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:17,109 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:17,109 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:17,116 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:17,117 INFO Loaded EURJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:05:17,303 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 95, 'BIAS_DOWN': 63, 'BIAS_NEUTRAL': 2865}  ambiguous=1742 (total=3023) horizon=12
2026-05-02 15:05:17,304 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected EURJPY — 2973 samples (group=cross) labels={'BIAS_UP': 95, 'BIAS_DOWN': 63, 'BIAS_NEUTRAL': 2815} clean={'BIAS_UP': 95, 'BIAS_DOWN': 63, 'BIAS_NEUTRAL': 1099}
2026-05-02 15:05:17,460 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:17,498 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:17,516 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:17,517 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:17,524 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:17,525 INFO Loaded EURUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:05:17,710 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 98, 'BIAS_DOWN': 57, 'BIAS_NEUTRAL': 2868}  ambiguous=1742 (total=3023) horizon=12
2026-05-02 15:05:17,712 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected EURUSD — 2973 samples (group=dollar) labels={'BIAS_UP': 98, 'BIAS_DOWN': 57, 'BIAS_NEUTRAL': 2818} clean={'BIAS_UP': 98, 'BIAS_DOWN': 57, 'BIAS_NEUTRAL': 1105}
2026-05-02 15:05:17,860 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:17,895 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:17,914 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:17,914 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:17,926 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:17,927 INFO Loaded GBPJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:05:18,135 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 164, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2758}  ambiguous=1723 (total=3023) horizon=12
2026-05-02 15:05:18,137 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected GBPJPY — 2973 samples (group=cross) labels={'BIAS_UP': 164, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2708} clean={'BIAS_UP': 164, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 1019}
2026-05-02 15:05:18,285 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:18,319 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:18,338 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:18,339 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:18,346 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:18,347 INFO Loaded GBPUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:05:18,541 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 79, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2843}  ambiguous=1759 (total=3023) horizon=12
2026-05-02 15:05:18,543 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected GBPUSD — 2973 samples (group=dollar) labels={'BIAS_UP': 79, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2793} clean={'BIAS_UP': 79, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 1073}
2026-05-02 15:05:18,676 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:05:18,706 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:05:18,723 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:05:18,724 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:05:18,730 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:05:18,731 INFO Loaded NZDUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:05:18,915 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 54, 'BIAS_DOWN': 54, 'BIAS_NEUTRAL': 2915}  ambiguous=1779 (total=3023) horizon=12
2026-05-02 15:05:18,917 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected NZDUSD — 2973 samples (group=dollar) labels={'BIAS_UP': 54, 'BIAS_DOWN': 54, 'BIAS_NEUTRAL': 2865} clean={'BIAS_UP': 54, 'BIAS_DOWN': 54, 'BIAS_NEUTRAL': 1117}
2026-05-02 15:05:19,066 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:19,101 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:19,119 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:19,120 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:19,126 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:19,127 INFO Loaded USDCAD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:05:19,308 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 89, 'BIAS_DOWN': 133, 'BIAS_NEUTRAL': 2801}  ambiguous=1770 (total=3023) horizon=12
2026-05-02 15:05:19,310 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected USDCAD — 2973 samples (group=dollar) labels={'BIAS_UP': 89, 'BIAS_DOWN': 133, 'BIAS_NEUTRAL': 2751} clean={'BIAS_UP': 89, 'BIAS_DOWN': 133, 'BIAS_NEUTRAL': 1016}
2026-05-02 15:05:19,472 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:19,507 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:19,527 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:19,527 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:19,535 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:19,535 INFO Loaded USDCHF/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:05:19,734 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 64, 'BIAS_DOWN': 52, 'BIAS_NEUTRAL': 2907}  ambiguous=1741 (total=3023) horizon=12
2026-05-02 15:05:19,736 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected USDCHF — 2973 samples (group=dollar) labels={'BIAS_UP': 64, 'BIAS_DOWN': 52, 'BIAS_NEUTRAL': 2857} clean={'BIAS_UP': 64, 'BIAS_DOWN': 52, 'BIAS_NEUTRAL': 1148}
2026-05-02 15:05:19,890 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:19,925 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:19,944 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:19,945 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:19,952 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:19,953 INFO Loaded USDJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:05:20,134 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 102, 'BIAS_DOWN': 78, 'BIAS_NEUTRAL': 2843}  ambiguous=1762 (total=3023) horizon=12
2026-05-02 15:05:20,136 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected USDJPY — 2973 samples (group=dollar) labels={'BIAS_UP': 102, 'BIAS_DOWN': 78, 'BIAS_NEUTRAL': 2793} clean={'BIAS_UP': 102, 'BIAS_DOWN': 78, 'BIAS_NEUTRAL': 1058}
2026-05-02 15:05:20,402 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:05:20,467 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:05:20,493 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:05:20,494 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:05:20,505 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:05:20,506 INFO Loaded XAUUSD/4H split=train fold=fold_000: 3204 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:05:20,711 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 2845}  ambiguous=1801 (total=3204) horizon=12
2026-05-02 15:05:20,712 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected XAUUSD — 3154 samples (group=gold) labels={'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 2795} clean={'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 1029}
2026-05-02 15:05:20,772 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 415, 'BIAS_DOWN': 235, 'BIAS_NEUTRAL': 8269}, 'dollar': {'BIAS_UP': 578, 'BIAS_DOWN': 530, 'BIAS_NEUTRAL': 19703}, 'gold': {'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 2795}}
2026-05-02 15:05:20,772 INFO Regime[4H mode=htf_bias] label distribution by year: {2016: {'BIAS_UP': 485, 'BIAS_DOWN': 511, 'BIAS_NEUTRAL': 15101}, 2017: {'BIAS_UP': 717, 'BIAS_DOWN': 401, 'BIAS_NEUTRAL': 15515}, 2018: {'BIAS_UP': 3, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 151}}
2026-05-02 15:05:20,814 INFO Loaded AUDUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:05:20,815 INFO Loaded EURGBP/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:05:20,816 INFO Loaded EURJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:05:20,817 INFO Loaded EURUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:05:20,818 INFO Loaded GBPJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:05:20,819 INFO Loaded GBPUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:05:20,819 INFO Loaded NZDUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:05:20,820 INFO Loaded USDCAD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:05:20,821 INFO Loaded USDCHF/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:05:20,822 INFO Loaded USDJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:05:20,823 INFO Loaded XAUUSD/4H split=val fold=fold_000: 1600 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:05:20,831 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:20,833 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:20,834 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:20,834 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:20,834 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:20,835 INFO Loaded AUDUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:05:20,999 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 5, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1448}  ambiguous=896 (total=1506) horizon=12
2026-05-02 15:05:21,001 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected AUDUSD — 1456 samples (group=dollar) labels={'BIAS_UP': 5, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1398} clean={'BIAS_UP': 5, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 531}
2026-05-02 15:05:21,074 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:21,077 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:21,077 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:21,078 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:21,078 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:21,079 INFO Loaded EURGBP/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:05:21,244 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 12, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1453}  ambiguous=868 (total=1506) horizon=12
2026-05-02 15:05:21,246 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected EURGBP — 1456 samples (group=cross) labels={'BIAS_UP': 12, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1403} clean={'BIAS_UP': 12, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 575}
2026-05-02 15:05:21,313 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:21,315 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:21,316 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:21,316 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:21,317 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:21,317 INFO Loaded EURJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:05:21,482 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 20, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1445}  ambiguous=874 (total=1506) horizon=12
2026-05-02 15:05:21,484 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected EURJPY — 1456 samples (group=cross) labels={'BIAS_UP': 20, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1395} clean={'BIAS_UP': 20, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 555}
2026-05-02 15:05:21,549 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:21,551 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:21,552 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:21,553 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:21,553 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:21,554 INFO Loaded EURUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:05:21,720 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 24, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 1415}  ambiguous=876 (total=1506) horizon=12
2026-05-02 15:05:21,722 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected EURUSD — 1456 samples (group=dollar) labels={'BIAS_UP': 24, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 1365} clean={'BIAS_UP': 24, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 522}
2026-05-02 15:05:21,785 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:21,788 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:21,788 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:21,789 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:21,789 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:21,790 INFO Loaded GBPJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:05:21,956 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 27, 'BIAS_DOWN': 37, 'BIAS_NEUTRAL': 1442}  ambiguous=926 (total=1506) horizon=12
2026-05-02 15:05:21,957 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected GBPJPY — 1456 samples (group=cross) labels={'BIAS_UP': 27, 'BIAS_DOWN': 37, 'BIAS_NEUTRAL': 1392} clean={'BIAS_UP': 27, 'BIAS_DOWN': 37, 'BIAS_NEUTRAL': 506}
2026-05-02 15:05:22,021 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:22,023 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:22,024 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:22,024 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:22,025 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:22,026 INFO Loaded GBPUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:05:22,186 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 50, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1371}  ambiguous=874 (total=1506) horizon=12
2026-05-02 15:05:22,187 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected GBPUSD — 1456 samples (group=dollar) labels={'BIAS_UP': 50, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1321} clean={'BIAS_UP': 50, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 476}
2026-05-02 15:05:22,246 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:05:22,248 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:05:22,249 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:05:22,249 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:05:22,250 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:05:22,250 INFO Loaded NZDUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:05:22,413 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 23, 'BIAS_DOWN': 80, 'BIAS_NEUTRAL': 1403}  ambiguous=896 (total=1506) horizon=12
2026-05-02 15:05:22,414 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected NZDUSD — 1456 samples (group=dollar) labels={'BIAS_UP': 23, 'BIAS_DOWN': 80, 'BIAS_NEUTRAL': 1353} clean={'BIAS_UP': 23, 'BIAS_DOWN': 80, 'BIAS_NEUTRAL': 482}
2026-05-02 15:05:22,476 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:22,478 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:22,479 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:22,480 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:22,480 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:22,481 INFO Loaded USDCAD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:05:22,647 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 48, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 1445}  ambiguous=907 (total=1506) horizon=12
2026-05-02 15:05:22,649 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected USDCAD — 1456 samples (group=dollar) labels={'BIAS_UP': 48, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 1395} clean={'BIAS_UP': 48, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 522}
2026-05-02 15:05:22,713 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:22,716 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:22,716 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:22,717 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:22,717 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:22,718 INFO Loaded USDCHF/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:05:22,883 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 68, 'BIAS_DOWN': 45, 'BIAS_NEUTRAL': 1393}  ambiguous=848 (total=1506) horizon=12
2026-05-02 15:05:22,885 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected USDCHF — 1456 samples (group=dollar) labels={'BIAS_UP': 68, 'BIAS_DOWN': 45, 'BIAS_NEUTRAL': 1343} clean={'BIAS_UP': 68, 'BIAS_DOWN': 45, 'BIAS_NEUTRAL': 530}
2026-05-02 15:05:22,946 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:22,949 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:22,949 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:22,950 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:22,950 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:05:22,951 INFO Loaded USDJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:05:23,184 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 58, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1418}  ambiguous=888 (total=1506) horizon=12
2026-05-02 15:05:23,186 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected USDJPY — 1456 samples (group=dollar) labels={'BIAS_UP': 58, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1368} clean={'BIAS_UP': 58, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 510}
2026-05-02 15:05:23,257 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:05:23,261 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:05:23,262 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:05:23,262 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:05:23,263 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:05:23,264 INFO Loaded XAUUSD/4H split=val fold=fold_000: 1600 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:05:23,450 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 18, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 1547}  ambiguous=851 (total=1600) horizon=12
2026-05-02 15:05:23,451 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected XAUUSD — 1550 samples (group=gold) labels={'BIAS_UP': 18, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 1497} clean={'BIAS_UP': 18, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 677}
2026-05-02 15:05:23,510 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 59, 'BIAS_DOWN': 119, 'BIAS_NEUTRAL': 4190}, 'dollar': {'BIAS_UP': 276, 'BIAS_DOWN': 373, 'BIAS_NEUTRAL': 9543}, 'gold': {'BIAS_UP': 18, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 1497}}
2026-05-02 15:05:23,511 INFO Regime[4H mode=htf_bias] label distribution by year: {2018: {'BIAS_UP': 352, 'BIAS_DOWN': 521, 'BIAS_NEUTRAL': 15083}, 2019: {'BIAS_UP': 1, 'BIAS_DOWN': 6, 'BIAS_NEUTRAL': 147}}
2026-05-02 15:05:23,552 INFO Regime phase HTF dataset build fold=fold_000: 7.8s (train=32884 val=16110)
2026-05-02 15:05:23,552 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=14004 dropped=18880 classes={'BIAS_UP': 1205, 'BIAS_DOWN': 912, 'BIAS_NEUTRAL': 11887})
2026-05-02 15:05:23,554 INFO RegimeClassifier[mode=htf_bias]: 14004 samples, classes={'BIAS_UP': 1205, 'BIAS_DOWN': 912, 'BIAS_NEUTRAL': 11887}, device=cuda
2026-05-02 15:05:23,554 INFO RegimeClassifier[mode=htf_bias]: undersample class BIAS_NEUTRAL: 11887 → 2736
2026-05-02 15:05:23,555 INFO RegimeClassifier[mode=htf_bias]: after undersampling: 4853 samples classes={'BIAS_UP': 1205, 'BIAS_DOWN': 912, 'BIAS_NEUTRAL': 2736}
2026-05-02 15:05:23,555 INFO RegimeClassifier: sample weights — mean=0.730  ambiguous(<0.4)=0.0%
2026-05-02 15:05:23,843 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-05-02 15:05:23,843 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-05-02 15:05:28,371 INFO Regime epoch  1/50 — tr=1.3015 va=0.6751 acc=0.263 bal=0.263 per_class={'BIAS_UP': 0.521, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 0.267}
2026-05-02 15:05:28,395 INFO Regime epoch  2/50 — tr=1.2753 va=0.6680 acc=0.379 bal=0.286
2026-05-02 15:05:28,417 INFO Regime epoch  3/50 — tr=1.2972 va=0.6654 acc=0.451 bal=0.315
2026-05-02 15:05:28,439 INFO Regime epoch  4/50 — tr=1.2834 va=0.6604 acc=0.521 bal=0.332
2026-05-02 15:05:28,461 INFO Regime epoch  5/50 — tr=1.2876 va=0.6543 acc=0.568 bal=0.346 per_class={'BIAS_UP': 0.34, 'BIAS_DOWN': 0.11, 'BIAS_NEUTRAL': 0.589}
2026-05-02 15:05:28,483 INFO Regime epoch  6/50 — tr=1.2782 va=0.6466 acc=0.614 bal=0.354
2026-05-02 15:05:28,506 INFO Regime epoch  7/50 — tr=1.2636 va=0.6396 acc=0.642 bal=0.373
2026-05-02 15:05:28,529 INFO Regime epoch  8/50 — tr=1.2602 va=0.6317 acc=0.670 bal=0.392
2026-05-02 15:05:28,550 INFO Regime epoch  9/50 — tr=1.2417 va=0.6250 acc=0.686 bal=0.412
2026-05-02 15:05:28,572 INFO Regime epoch 10/50 — tr=1.2002 va=0.6184 acc=0.699 bal=0.446 per_class={'BIAS_UP': 0.467, 'BIAS_DOWN': 0.148, 'BIAS_NEUTRAL': 0.724}
2026-05-02 15:05:28,594 INFO Regime epoch 11/50 — tr=1.1972 va=0.6118 acc=0.713 bal=0.494
2026-05-02 15:05:28,617 INFO Regime epoch 12/50 — tr=1.1722 va=0.6061 acc=0.719 bal=0.544
2026-05-02 15:05:28,639 INFO Regime epoch 13/50 — tr=1.1692 va=0.6015 acc=0.720 bal=0.591
2026-05-02 15:05:28,661 INFO Regime epoch 14/50 — tr=1.1570 va=0.5960 acc=0.724 bal=0.640
2026-05-02 15:05:28,689 INFO Regime epoch 15/50 — tr=1.1252 va=0.5907 acc=0.731 bal=0.673 per_class={'BIAS_UP': 0.802, 'BIAS_DOWN': 0.48, 'BIAS_NEUTRAL': 0.738}
2026-05-02 15:05:28,711 INFO Regime epoch 16/50 — tr=1.1124 va=0.5847 acc=0.739 bal=0.705
2026-05-02 15:05:28,733 INFO Regime epoch 17/50 — tr=1.0922 va=0.5804 acc=0.741 bal=0.733
2026-05-02 15:05:28,755 INFO Regime epoch 18/50 — tr=1.0698 va=0.5757 acc=0.742 bal=0.753
2026-05-02 15:05:28,777 INFO Regime epoch 19/50 — tr=1.0550 va=0.5715 acc=0.742 bal=0.773
2026-05-02 15:05:28,801 INFO Regime epoch 20/50 — tr=1.0379 va=0.5679 acc=0.741 bal=0.795 per_class={'BIAS_UP': 0.901, 'BIAS_DOWN': 0.748, 'BIAS_NEUTRAL': 0.737}
2026-05-02 15:05:28,824 INFO Regime epoch 21/50 — tr=1.0224 va=0.5633 acc=0.742 bal=0.809
2026-05-02 15:05:28,846 INFO Regime epoch 22/50 — tr=1.0003 va=0.5603 acc=0.738 bal=0.820
2026-05-02 15:05:28,867 INFO Regime epoch 23/50 — tr=1.0048 va=0.5575 acc=0.735 bal=0.829
2026-05-02 15:05:28,889 INFO Regime epoch 24/50 — tr=0.9965 va=0.5535 acc=0.737 bal=0.841
2026-05-02 15:05:28,912 INFO Regime epoch 25/50 — tr=0.9750 va=0.5510 acc=0.735 bal=0.851 per_class={'BIAS_UP': 0.926, 'BIAS_DOWN': 0.901, 'BIAS_NEUTRAL': 0.724}
2026-05-02 15:05:28,934 INFO Regime epoch 26/50 — tr=0.9692 va=0.5484 acc=0.732 bal=0.856
2026-05-02 15:05:28,956 INFO Regime epoch 27/50 — tr=0.9577 va=0.5468 acc=0.728 bal=0.860
2026-05-02 15:05:28,980 INFO Regime epoch 28/50 — tr=0.9504 va=0.5438 acc=0.730 bal=0.861
2026-05-02 15:05:29,005 INFO Regime epoch 29/50 — tr=0.9364 va=0.5411 acc=0.730 bal=0.865
2026-05-02 15:05:29,030 INFO Regime epoch 30/50 — tr=0.9198 va=0.5376 acc=0.731 bal=0.867 per_class={'BIAS_UP': 0.941, 'BIAS_DOWN': 0.941, 'BIAS_NEUTRAL': 0.719}
2026-05-02 15:05:29,054 INFO Regime epoch 31/50 — tr=0.9174 va=0.5357 acc=0.732 bal=0.871
2026-05-02 15:05:29,078 INFO Regime epoch 32/50 — tr=0.9176 va=0.5334 acc=0.730 bal=0.871
2026-05-02 15:05:29,103 INFO Regime epoch 33/50 — tr=0.9148 va=0.5310 acc=0.732 bal=0.874
2026-05-02 15:05:29,125 INFO Regime epoch 34/50 — tr=0.9077 va=0.5299 acc=0.731 bal=0.875
2026-05-02 15:05:29,149 INFO Regime epoch 35/50 — tr=0.8969 va=0.5281 acc=0.731 bal=0.877 per_class={'BIAS_UP': 0.946, 'BIAS_DOWN': 0.966, 'BIAS_NEUTRAL': 0.718}
2026-05-02 15:05:29,173 INFO Regime epoch 36/50 — tr=0.9048 va=0.5286 acc=0.728 bal=0.877
2026-05-02 15:05:29,197 INFO Regime epoch 37/50 — tr=0.9029 va=0.5289 acc=0.725 bal=0.880
2026-05-02 15:05:29,221 INFO Regime epoch 38/50 — tr=0.8967 va=0.5285 acc=0.724 bal=0.879
2026-05-02 15:05:29,245 INFO Regime epoch 39/50 — tr=0.8888 va=0.5279 acc=0.723 bal=0.880
2026-05-02 15:05:29,268 INFO Regime epoch 40/50 — tr=0.8869 va=0.5270 acc=0.723 bal=0.880 per_class={'BIAS_UP': 0.958, 'BIAS_DOWN': 0.973, 'BIAS_NEUTRAL': 0.709}
2026-05-02 15:05:29,293 INFO Regime epoch 41/50 — tr=0.8961 va=0.5270 acc=0.722 bal=0.879
2026-05-02 15:05:29,316 INFO Regime epoch 42/50 — tr=0.8786 va=0.5248 acc=0.726 bal=0.881
2026-05-02 15:05:29,343 INFO Regime epoch 43/50 — tr=0.8847 va=0.5253 acc=0.724 bal=0.880
2026-05-02 15:05:29,366 INFO Regime epoch 44/50 — tr=0.8823 va=0.5255 acc=0.725 bal=0.880
2026-05-02 15:05:29,389 INFO Regime epoch 45/50 — tr=0.8859 va=0.5250 acc=0.725 bal=0.881 per_class={'BIAS_UP': 0.958, 'BIAS_DOWN': 0.975, 'BIAS_NEUTRAL': 0.711}
2026-05-02 15:05:29,411 INFO Regime epoch 46/50 — tr=0.8757 va=0.5274 acc=0.721 bal=0.881
2026-05-02 15:05:29,435 INFO Regime epoch 47/50 — tr=0.8978 va=0.5277 acc=0.720 bal=0.881
2026-05-02 15:05:29,457 INFO Regime epoch 48/50 — tr=0.8834 va=0.5263 acc=0.722 bal=0.882
2026-05-02 15:05:29,481 INFO Regime epoch 49/50 — tr=0.8790 va=0.5253 acc=0.724 bal=0.882
2026-05-02 15:05:29,504 INFO Regime epoch 50/50 — tr=0.8878 va=0.5264 acc=0.722 bal=0.881 per_class={'BIAS_UP': 0.96, 'BIAS_DOWN': 0.975, 'BIAS_NEUTRAL': 0.708}
2026-05-02 15:05:29,525 INFO RegimeClassifier[mode=htf_bias] validation precision={'BIAS_UP': 0.134, 'BIAS_DOWN': 0.185, 'BIAS_NEUTRAL': 0.998} recall={'BIAS_UP': 0.963, 'BIAS_DOWN': 0.975, 'BIAS_NEUTRAL': 0.707} f1={'BIAS_UP': 0.236, 'BIAS_DOWN': 0.311, 'BIAS_NEUTRAL': 0.828} confusion=[[340, 0, 13], [0, 514, 13], [2192, 2268, 10770]]
2026-05-02 15:05:29,526 INFO Regime phase HTF train fold=fold_000: 6.0s
Traceback (most recent call last):
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1707, in <module>
    main()
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1666, in main
    result = retrain_regime(dry)
             ^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1209, in retrain_regime
    raise RuntimeError(f"Regime HTF training failed fold={fold_key}: {res_4h['error']}")
RuntimeError: Regime HTF training failed fold=fold_000: Regime HTF directional validation below acceptance floor: precision={'BIAS_UP': 0.134, 'BIAS_DOWN': 0.185, 'BIAS_NEUTRAL': 0.998} min_directional_precision=0.300 f1={'BIAS_UP': 0.236, 'BIAS_DOWN': 0.311, 'BIAS_NEUTRAL': 0.828} min_directional_f1=0.300 weak_precision=['BIAS_UP', 'BIAS_DOWN'] weak_f1=['BIAS_UP']. Refusing to save directional-bias weights that flood neutral bars.

=== TRAINING COMPLETE ===
  regime: FAILED: exit 1
2026-05-02 15:05:31,778 ERROR retrain regime failed (exit 1)
2026-05-02 15:05:31,778 ERROR Model regime failed: exit 1
2026-05-02 15:05:31,779 WARNING   [MISSING] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 15:05:31,779 WARNING   [MISSING] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-02 15:05:31,779 WARNING   [MISSING] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-02 15:05:31,779 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-05-02 15:05:31,779 INFO   [DEFERRED] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip (expected after Round 1)
2026-05-02 15:05:31,779 WARNING Missing required weights: ['gru_lstm', 'regime_htf', 'regime_ltf'] — run retrain_incremental.py for each
2026-05-02 15:05:31,779 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer', 'rl_ppo']
2026-05-02 15:05:31,779 WARNING No retrain_history.jsonl found
2026-05-02 15:05:31,780 ERROR Step 7a failed; required training/artifacts missing: ['gru_lstm', 'regime', 'regime_htf', 'regime_ltf']
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