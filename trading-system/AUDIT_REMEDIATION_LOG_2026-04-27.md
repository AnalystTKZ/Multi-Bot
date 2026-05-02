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
2026-05-02 08:57:02,356 INFO Loading feature-engineered data...
2026-05-02 08:57:02,979 INFO Loaded 221743 rows, 202 features
2026-05-02 08:57:02,980 INFO Data span: 2016-01-04 -> 2025-08-05  (9.6 years)
2026-05-02 08:57:02,982 INFO Fold 000 train 2016-01-04 -> 2018-01-03 (47088 bars), val 2018-01-04 -> 2019-01-03 (23448 bars)
2026-05-02 08:57:02,983 INFO Fold 001 train 2018-01-04 -> 2020-01-03 (46825 bars), val 2020-01-06 -> 2020-12-31 (23259 bars)
2026-05-02 08:57:02,983 INFO Fold 002 train 2020-01-06 -> 2022-01-03 (46766 bars), val 2022-01-04 -> 2023-01-03 (23588 bars)
2026-05-02 08:57:02,983 INFO No leakage confirmed: every fold ends before final 2-year blind test

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
2026-05-02 08:57:06,457 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-05-02 08:57:06,457 INFO --- Training regime ---
2026-05-02 08:57:06,458 INFO Running retrain --model regime
2026-05-02 08:57:06,634 INFO retrain environment: KAGGLE
2026-05-02 08:57:08,215 INFO Device: CUDA (2 GPU(s))
2026-05-02 08:57:08,227 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-02 08:57:08,227 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-02 08:57:08,227 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-02 08:57:08,230 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-02 08:57:08,231 INFO Retrain data split: train
2026-05-02 08:57:08,231 INFO Retrain rolling fold selector: latest
2026-05-02 08:57:08,232 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 5-score behaviour) ===
2026-05-02 08:57:08,392 INFO NumExpr defaulting to 4 threads.
2026-05-02 08:57:08,606 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-02 08:57:08,606 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-02 08:57:08,607 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-02 08:57:08,607 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-02 08:57:08,661 INFO Regime rolling folds selected: ['fold_000', 'fold_001', 'fold_002']
2026-05-02 08:57:08,661 INFO === Regime rolling fold 1/3: fold_000 ===
2026-05-02 08:57:08,661 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-05-02 08:57:08,698 INFO Split boundaries loaded fold=fold_000/3 — train 2016-01-04→2018-01-03  val 2018-01-04→2019-01-03  test 2023-08-07→2025-08-05
2026-05-02 08:57:08,699 INFO Loaded AUDUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:08,714 INFO Loaded EURGBP/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:08,729 INFO Loaded EURJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:08,743 INFO Loaded EURUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:08,757 INFO Loaded GBPJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:08,772 INFO Loaded GBPUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:08,787 INFO Loaded NZDUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:08,814 INFO Loaded USDCAD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:08,838 INFO Loaded USDCHF/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:08,862 INFO Loaded USDJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:08,892 INFO Loaded XAUUSD/4H split=train fold=fold_000: 3204 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:09,023 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:09,066 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:09,086 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:09,086 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:09,093 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:09,094 INFO Loaded AUDUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:09,284 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 208, 'BIAS_DOWN': 170, 'BIAS_NEUTRAL': 2645}  ambiguous=2195 (total=3023) horizon=12
2026-05-02 08:57:09,286 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected AUDUSD — 2973 samples (group=dollar) labels={'BIAS_UP': 208, 'BIAS_DOWN': 170, 'BIAS_NEUTRAL': 2595} clean={'BIAS_UP': 208, 'BIAS_DOWN': 170, 'BIAS_NEUTRAL': 442}
2026-05-02 08:57:09,445 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:09,477 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:09,495 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:09,496 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:09,503 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:09,504 INFO Loaded EURGBP/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:09,677 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 304, 'BIAS_DOWN': 141, 'BIAS_NEUTRAL': 2578}  ambiguous=2143 (total=3023) horizon=12
2026-05-02 08:57:09,679 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected EURGBP — 2973 samples (group=cross) labels={'BIAS_UP': 304, 'BIAS_DOWN': 141, 'BIAS_NEUTRAL': 2528} clean={'BIAS_UP': 304, 'BIAS_DOWN': 141, 'BIAS_NEUTRAL': 424}
2026-05-02 08:57:09,846 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:09,884 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:09,902 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:09,903 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:09,910 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:09,910 INFO Loaded EURJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:10,085 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 201, 'BIAS_DOWN': 169, 'BIAS_NEUTRAL': 2653}  ambiguous=2220 (total=3023) horizon=12
2026-05-02 08:57:10,086 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected EURJPY — 2973 samples (group=cross) labels={'BIAS_UP': 201, 'BIAS_DOWN': 169, 'BIAS_NEUTRAL': 2603} clean={'BIAS_UP': 201, 'BIAS_DOWN': 169, 'BIAS_NEUTRAL': 422}
2026-05-02 08:57:10,247 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:10,283 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:10,302 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:10,302 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:10,309 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:10,310 INFO Loaded EURUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:10,504 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 215, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 2661}  ambiguous=2269 (total=3023) horizon=12
2026-05-02 08:57:10,507 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected EURUSD — 2973 samples (group=dollar) labels={'BIAS_UP': 215, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 2611} clean={'BIAS_UP': 215, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 382}
2026-05-02 08:57:10,678 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:10,712 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:10,730 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:10,731 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:10,737 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:10,738 INFO Loaded GBPJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:10,943 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 301, 'BIAS_DOWN': 252, 'BIAS_NEUTRAL': 2470}  ambiguous=2067 (total=3023) horizon=12
2026-05-02 08:57:10,944 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected GBPJPY — 2973 samples (group=cross) labels={'BIAS_UP': 301, 'BIAS_DOWN': 252, 'BIAS_NEUTRAL': 2420} clean={'BIAS_UP': 301, 'BIAS_DOWN': 252, 'BIAS_NEUTRAL': 394}
2026-05-02 08:57:11,107 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:11,145 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:11,164 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:11,164 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:11,171 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:11,172 INFO Loaded GBPUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:11,350 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 206, 'BIAS_DOWN': 207, 'BIAS_NEUTRAL': 2610}  ambiguous=2211 (total=3023) horizon=12
2026-05-02 08:57:11,351 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected GBPUSD — 2973 samples (group=dollar) labels={'BIAS_UP': 206, 'BIAS_DOWN': 207, 'BIAS_NEUTRAL': 2560} clean={'BIAS_UP': 206, 'BIAS_DOWN': 207, 'BIAS_NEUTRAL': 393}
2026-05-02 08:57:11,486 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:11,516 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:11,533 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:11,533 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:11,540 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:11,541 INFO Loaded NZDUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:11,719 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 163, 'BIAS_DOWN': 174, 'BIAS_NEUTRAL': 2686}  ambiguous=2255 (total=3023) horizon=12
2026-05-02 08:57:11,721 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected NZDUSD — 2973 samples (group=dollar) labels={'BIAS_UP': 163, 'BIAS_DOWN': 174, 'BIAS_NEUTRAL': 2636} clean={'BIAS_UP': 163, 'BIAS_DOWN': 174, 'BIAS_NEUTRAL': 422}
2026-05-02 08:57:11,871 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:11,905 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:11,923 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:11,923 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:11,930 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:11,931 INFO Loaded USDCAD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:12,110 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 172, 'BIAS_DOWN': 311, 'BIAS_NEUTRAL': 2540}  ambiguous=2152 (total=3023) horizon=12
2026-05-02 08:57:12,111 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected USDCAD — 2973 samples (group=dollar) labels={'BIAS_UP': 172, 'BIAS_DOWN': 311, 'BIAS_NEUTRAL': 2490} clean={'BIAS_UP': 172, 'BIAS_DOWN': 311, 'BIAS_NEUTRAL': 385}
2026-05-02 08:57:12,262 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:12,298 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:12,317 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:12,317 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:12,326 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:12,327 INFO Loaded USDCHF/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:12,524 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 174, 'BIAS_DOWN': 174, 'BIAS_NEUTRAL': 2675}  ambiguous=2210 (total=3023) horizon=12
2026-05-02 08:57:12,525 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected USDCHF — 2973 samples (group=dollar) labels={'BIAS_UP': 174, 'BIAS_DOWN': 174, 'BIAS_NEUTRAL': 2625} clean={'BIAS_UP': 174, 'BIAS_DOWN': 174, 'BIAS_NEUTRAL': 453}
2026-05-02 08:57:12,679 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:12,715 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:12,736 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:12,736 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:12,744 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:12,745 INFO Loaded USDJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:12,944 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 234, 'BIAS_DOWN': 210, 'BIAS_NEUTRAL': 2579}  ambiguous=2164 (total=3023) horizon=12
2026-05-02 08:57:12,945 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected USDJPY — 2973 samples (group=dollar) labels={'BIAS_UP': 234, 'BIAS_DOWN': 210, 'BIAS_NEUTRAL': 2529} clean={'BIAS_UP': 234, 'BIAS_DOWN': 210, 'BIAS_NEUTRAL': 402}
2026-05-02 08:57:13,204 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:13,265 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:13,288 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:13,289 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:13,298 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:13,299 INFO Loaded XAUUSD/4H split=train fold=fold_000: 3204 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:13,503 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 350, 'BIAS_DOWN': 271, 'BIAS_NEUTRAL': 2583}  ambiguous=2166 (total=3204) horizon=12
2026-05-02 08:57:13,505 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected XAUUSD — 3154 samples (group=gold) labels={'BIAS_UP': 350, 'BIAS_DOWN': 271, 'BIAS_NEUTRAL': 2533} clean={'BIAS_UP': 350, 'BIAS_DOWN': 271, 'BIAS_NEUTRAL': 409}
2026-05-02 08:57:13,564 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 806, 'BIAS_DOWN': 562, 'BIAS_NEUTRAL': 7551}, 'dollar': {'BIAS_UP': 1372, 'BIAS_DOWN': 1393, 'BIAS_NEUTRAL': 18046}, 'gold': {'BIAS_UP': 350, 'BIAS_DOWN': 271, 'BIAS_NEUTRAL': 2533}}
2026-05-02 08:57:13,565 INFO Regime[4H mode=htf_bias] label distribution by year: {2016: {'BIAS_UP': 1083, 'BIAS_DOWN': 1204, 'BIAS_NEUTRAL': 13810}, 2017: {'BIAS_UP': 1441, 'BIAS_DOWN': 1022, 'BIAS_NEUTRAL': 14170}, 2018: {'BIAS_UP': 4, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 150}}
2026-05-02 08:57:13,606 INFO Loaded AUDUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:13,607 INFO Loaded EURGBP/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:13,608 INFO Loaded EURJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:13,609 INFO Loaded EURUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:13,609 INFO Loaded GBPJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:13,610 INFO Loaded GBPUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:13,611 INFO Loaded NZDUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:13,612 INFO Loaded USDCAD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:13,612 INFO Loaded USDCHF/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:13,613 INFO Loaded USDJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:13,614 INFO Loaded XAUUSD/4H split=val fold=fold_000: 1600 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:13,620 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:13,622 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:13,623 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:13,623 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:13,624 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:13,624 INFO Loaded AUDUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:13,789 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 36, 'BIAS_DOWN': 142, 'BIAS_NEUTRAL': 1328}  ambiguous=1099 (total=1506) horizon=12
2026-05-02 08:57:13,790 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected AUDUSD — 1456 samples (group=dollar) labels={'BIAS_UP': 36, 'BIAS_DOWN': 142, 'BIAS_NEUTRAL': 1278} clean={'BIAS_UP': 36, 'BIAS_DOWN': 142, 'BIAS_NEUTRAL': 219}
2026-05-02 08:57:13,857 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:13,859 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:13,860 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:13,860 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:13,861 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:13,862 INFO Loaded EURGBP/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:14,131 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 56, 'BIAS_DOWN': 79, 'BIAS_NEUTRAL': 1371}  ambiguous=1120 (total=1506) horizon=12
2026-05-02 08:57:14,132 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected EURGBP — 1456 samples (group=cross) labels={'BIAS_UP': 56, 'BIAS_DOWN': 79, 'BIAS_NEUTRAL': 1321} clean={'BIAS_UP': 56, 'BIAS_DOWN': 79, 'BIAS_NEUTRAL': 245}
2026-05-02 08:57:14,199 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:14,201 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:14,202 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:14,202 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:14,202 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:14,203 INFO Loaded EURJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:14,365 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 55, 'BIAS_DOWN': 119, 'BIAS_NEUTRAL': 1332}  ambiguous=1099 (total=1506) horizon=12
2026-05-02 08:57:14,367 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected EURJPY — 1456 samples (group=cross) labels={'BIAS_UP': 55, 'BIAS_DOWN': 119, 'BIAS_NEUTRAL': 1282} clean={'BIAS_UP': 55, 'BIAS_DOWN': 119, 'BIAS_NEUTRAL': 227}
2026-05-02 08:57:14,435 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:14,437 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:14,438 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:14,438 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:14,438 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:14,439 INFO Loaded EURUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:14,602 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 56, 'BIAS_DOWN': 129, 'BIAS_NEUTRAL': 1321}  ambiguous=1132 (total=1506) horizon=12
2026-05-02 08:57:14,603 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected EURUSD — 1456 samples (group=dollar) labels={'BIAS_UP': 56, 'BIAS_DOWN': 129, 'BIAS_NEUTRAL': 1271} clean={'BIAS_UP': 56, 'BIAS_DOWN': 129, 'BIAS_NEUTRAL': 180}
2026-05-02 08:57:14,667 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:14,669 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:14,670 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:14,670 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:14,671 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:14,671 INFO Loaded GBPJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:14,837 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 78, 'BIAS_DOWN': 149, 'BIAS_NEUTRAL': 1279}  ambiguous=1067 (total=1506) horizon=12
2026-05-02 08:57:14,838 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected GBPJPY — 1456 samples (group=cross) labels={'BIAS_UP': 78, 'BIAS_DOWN': 149, 'BIAS_NEUTRAL': 1229} clean={'BIAS_UP': 78, 'BIAS_DOWN': 149, 'BIAS_NEUTRAL': 208}
2026-05-02 08:57:14,901 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:14,903 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:14,904 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:14,905 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:14,905 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:14,906 INFO Loaded GBPUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:15,069 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 94, 'BIAS_DOWN': 144, 'BIAS_NEUTRAL': 1268}  ambiguous=1106 (total=1506) horizon=12
2026-05-02 08:57:15,070 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected GBPUSD — 1456 samples (group=dollar) labels={'BIAS_UP': 94, 'BIAS_DOWN': 144, 'BIAS_NEUTRAL': 1218} clean={'BIAS_UP': 94, 'BIAS_DOWN': 144, 'BIAS_NEUTRAL': 158}
2026-05-02 08:57:15,134 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:15,135 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:15,136 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:15,136 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:15,136 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:15,137 INFO Loaded NZDUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:15,300 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 64, 'BIAS_DOWN': 164, 'BIAS_NEUTRAL': 1278}  ambiguous=1102 (total=1506) horizon=12
2026-05-02 08:57:15,301 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected NZDUSD — 1456 samples (group=dollar) labels={'BIAS_UP': 64, 'BIAS_DOWN': 164, 'BIAS_NEUTRAL': 1228} clean={'BIAS_UP': 64, 'BIAS_DOWN': 164, 'BIAS_NEUTRAL': 172}
2026-05-02 08:57:15,366 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:15,368 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:15,369 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:15,370 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:15,370 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:15,371 INFO Loaded USDCAD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:15,538 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 115, 'BIAS_DOWN': 57, 'BIAS_NEUTRAL': 1334}  ambiguous=1126 (total=1506) horizon=12
2026-05-02 08:57:15,540 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected USDCAD — 1456 samples (group=dollar) labels={'BIAS_UP': 115, 'BIAS_DOWN': 57, 'BIAS_NEUTRAL': 1284} clean={'BIAS_UP': 115, 'BIAS_DOWN': 57, 'BIAS_NEUTRAL': 203}
2026-05-02 08:57:15,604 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:15,606 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:15,607 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:15,607 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:15,608 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:15,609 INFO Loaded USDCHF/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:15,770 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 124, 'BIAS_DOWN': 88, 'BIAS_NEUTRAL': 1294}  ambiguous=1080 (total=1506) horizon=12
2026-05-02 08:57:15,771 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected USDCHF — 1456 samples (group=dollar) labels={'BIAS_UP': 124, 'BIAS_DOWN': 88, 'BIAS_NEUTRAL': 1244} clean={'BIAS_UP': 124, 'BIAS_DOWN': 88, 'BIAS_NEUTRAL': 205}
2026-05-02 08:57:15,836 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:15,839 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:15,839 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:15,840 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:15,840 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:15,841 INFO Loaded USDJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:16,002 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 119, 'BIAS_DOWN': 62, 'BIAS_NEUTRAL': 1325}  ambiguous=1119 (total=1506) horizon=12
2026-05-02 08:57:16,003 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected USDJPY — 1456 samples (group=dollar) labels={'BIAS_UP': 119, 'BIAS_DOWN': 62, 'BIAS_NEUTRAL': 1275} clean={'BIAS_UP': 119, 'BIAS_DOWN': 62, 'BIAS_NEUTRAL': 203}
2026-05-02 08:57:16,075 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:16,079 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:16,080 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:16,080 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:16,081 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:16,082 INFO Loaded XAUUSD/4H split=val fold=fold_000: 1600 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:16,253 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 75, 'BIAS_DOWN': 92, 'BIAS_NEUTRAL': 1433}  ambiguous=1130 (total=1600) horizon=12
2026-05-02 08:57:16,254 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected XAUUSD — 1550 samples (group=gold) labels={'BIAS_UP': 75, 'BIAS_DOWN': 92, 'BIAS_NEUTRAL': 1383} clean={'BIAS_UP': 75, 'BIAS_DOWN': 92, 'BIAS_NEUTRAL': 297}
2026-05-02 08:57:16,312 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 189, 'BIAS_DOWN': 347, 'BIAS_NEUTRAL': 3832}, 'dollar': {'BIAS_UP': 608, 'BIAS_DOWN': 786, 'BIAS_NEUTRAL': 8798}, 'gold': {'BIAS_UP': 75, 'BIAS_DOWN': 92, 'BIAS_NEUTRAL': 1383}}
2026-05-02 08:57:16,312 INFO Regime[4H mode=htf_bias] label distribution by year: {2018: {'BIAS_UP': 871, 'BIAS_DOWN': 1218, 'BIAS_NEUTRAL': 13867}, 2019: {'BIAS_UP': 1, 'BIAS_DOWN': 7, 'BIAS_NEUTRAL': 146}}
2026-05-02 08:57:16,356 INFO Regime phase HTF dataset build fold=fold_000: 7.7s (train=32884 val=16110)
2026-05-02 08:57:16,357 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=9282 dropped=23602 classes={'BIAS_UP': 2528, 'BIAS_DOWN': 2226, 'BIAS_NEUTRAL': 4528})
2026-05-02 08:57:16,358 INFO RegimeClassifier[mode=htf_bias]: 9282 samples, classes={'BIAS_UP': 2528, 'BIAS_DOWN': 2226, 'BIAS_NEUTRAL': 4528}, device=cuda
2026-05-02 08:57:16,358 INFO RegimeClassifier[mode=htf_bias]: undersample class BIAS_NEUTRAL: 4528 → 4452
2026-05-02 08:57:16,359 INFO RegimeClassifier[mode=htf_bias]: after undersampling: 9206 samples classes={'BIAS_UP': 2528, 'BIAS_DOWN': 2226, 'BIAS_NEUTRAL': 4452}
2026-05-02 08:57:16,359 INFO RegimeClassifier: sample weights — mean=0.747  ambiguous(<0.4)=0.0%
2026-05-02 08:57:16,625 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-05-02 08:57:16,626 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-05-02 08:57:21,025 INFO Regime epoch  1/50 — tr=1.1395 va=0.8044 acc=0.080 bal=0.360 per_class={'BIAS_UP': 0.079, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.0}
2026-05-02 08:57:21,056 INFO Regime epoch  2/50 — tr=1.1360 va=0.7239 acc=0.090 bal=0.423
2026-05-02 08:57:21,086 INFO Regime epoch  3/50 — tr=1.1516 va=0.6899 acc=0.109 bal=0.452
2026-05-02 08:57:21,115 INFO Regime epoch  4/50 — tr=1.1321 va=0.6725 acc=0.163 bal=0.473
2026-05-02 08:57:21,144 INFO Regime epoch  5/50 — tr=1.1289 va=0.6647 acc=0.203 bal=0.484 per_class={'BIAS_UP': 0.682, 'BIAS_DOWN': 0.634, 'BIAS_NEUTRAL': 0.135}
2026-05-02 08:57:21,173 INFO Regime epoch  6/50 — tr=1.1160 va=0.6611 acc=0.230 bal=0.506
2026-05-02 08:57:21,202 INFO Regime epoch  7/50 — tr=1.0983 va=0.6585 acc=0.253 bal=0.535
2026-05-02 08:57:21,232 INFO Regime epoch  8/50 — tr=1.0817 va=0.6562 acc=0.273 bal=0.577
2026-05-02 08:57:21,262 INFO Regime epoch  9/50 — tr=1.0555 va=0.6531 acc=0.289 bal=0.619
2026-05-02 08:57:21,290 INFO Regime epoch 10/50 — tr=1.0356 va=0.6498 acc=0.304 bal=0.656 per_class={'BIAS_UP': 0.891, 'BIAS_DOWN': 0.856, 'BIAS_NEUTRAL': 0.22}
2026-05-02 08:57:21,320 INFO Regime epoch 11/50 — tr=1.0203 va=0.6458 acc=0.316 bal=0.680
2026-05-02 08:57:21,350 INFO Regime epoch 12/50 — tr=0.9807 va=0.6408 acc=0.329 bal=0.699
2026-05-02 08:57:21,380 INFO Regime epoch 13/50 — tr=0.9710 va=0.6363 acc=0.342 bal=0.711
2026-05-02 08:57:21,410 INFO Regime epoch 14/50 — tr=0.9523 va=0.6299 acc=0.361 bal=0.720
2026-05-02 08:57:21,442 INFO Regime epoch 15/50 — tr=0.9299 va=0.6231 acc=0.379 bal=0.729 per_class={'BIAS_UP': 0.928, 'BIAS_DOWN': 0.966, 'BIAS_NEUTRAL': 0.293}
2026-05-02 08:57:21,474 INFO Regime epoch 16/50 — tr=0.9117 va=0.6174 acc=0.393 bal=0.736
2026-05-02 08:57:21,504 INFO Regime epoch 17/50 — tr=0.8940 va=0.6117 acc=0.407 bal=0.744
2026-05-02 08:57:21,535 INFO Regime epoch 18/50 — tr=0.8809 va=0.6058 acc=0.423 bal=0.752
2026-05-02 08:57:21,567 INFO Regime epoch 19/50 — tr=0.8630 va=0.6005 acc=0.435 bal=0.759
2026-05-02 08:57:21,599 INFO Regime epoch 20/50 — tr=0.8522 va=0.5953 acc=0.448 bal=0.764 per_class={'BIAS_UP': 0.94, 'BIAS_DOWN': 0.982, 'BIAS_NEUTRAL': 0.371}
2026-05-02 08:57:21,629 INFO Regime epoch 21/50 — tr=0.8394 va=0.5902 acc=0.459 bal=0.769
2026-05-02 08:57:21,661 INFO Regime epoch 22/50 — tr=0.8352 va=0.5849 acc=0.470 bal=0.774
2026-05-02 08:57:21,690 INFO Regime epoch 23/50 — tr=0.8265 va=0.5802 acc=0.479 bal=0.776
2026-05-02 08:57:21,719 INFO Regime epoch 24/50 — tr=0.8229 va=0.5778 acc=0.483 bal=0.780
2026-05-02 08:57:21,749 INFO Regime epoch 25/50 — tr=0.8149 va=0.5741 acc=0.490 bal=0.782 per_class={'BIAS_UP': 0.947, 'BIAS_DOWN': 0.982, 'BIAS_NEUTRAL': 0.418}
2026-05-02 08:57:21,778 INFO Regime epoch 26/50 — tr=0.8079 va=0.5717 acc=0.493 bal=0.784
2026-05-02 08:57:21,809 INFO Regime epoch 27/50 — tr=0.8031 va=0.5686 acc=0.497 bal=0.787
2026-05-02 08:57:21,839 INFO Regime epoch 28/50 — tr=0.7980 va=0.5653 acc=0.504 bal=0.789
2026-05-02 08:57:21,870 INFO Regime epoch 29/50 — tr=0.7917 va=0.5633 acc=0.507 bal=0.791
2026-05-02 08:57:21,902 INFO Regime epoch 30/50 — tr=0.7809 va=0.5615 acc=0.511 bal=0.793 per_class={'BIAS_UP': 0.952, 'BIAS_DOWN': 0.984, 'BIAS_NEUTRAL': 0.443}
2026-05-02 08:57:21,931 INFO Regime epoch 31/50 — tr=0.7915 va=0.5604 acc=0.512 bal=0.794
2026-05-02 08:57:21,961 INFO Regime epoch 32/50 — tr=0.7808 va=0.5591 acc=0.514 bal=0.795
2026-05-02 08:57:21,991 INFO Regime epoch 33/50 — tr=0.7758 va=0.5568 acc=0.521 bal=0.797
2026-05-02 08:57:22,020 INFO Regime epoch 34/50 — tr=0.7784 va=0.5559 acc=0.521 bal=0.798
2026-05-02 08:57:22,049 INFO Regime epoch 35/50 — tr=0.7761 va=0.5545 acc=0.524 bal=0.799 per_class={'BIAS_UP': 0.954, 'BIAS_DOWN': 0.985, 'BIAS_NEUTRAL': 0.457}
2026-05-02 08:57:22,077 INFO Regime epoch 36/50 — tr=0.7702 va=0.5538 acc=0.524 bal=0.799
2026-05-02 08:57:22,107 INFO Regime epoch 37/50 — tr=0.7712 va=0.5530 acc=0.524 bal=0.799
2026-05-02 08:57:22,138 INFO Regime epoch 38/50 — tr=0.7739 va=0.5505 acc=0.530 bal=0.801
2026-05-02 08:57:22,168 INFO Regime epoch 39/50 — tr=0.7741 va=0.5500 acc=0.530 bal=0.801
2026-05-02 08:57:22,197 INFO Regime epoch 40/50 — tr=0.7692 va=0.5498 acc=0.530 bal=0.801 per_class={'BIAS_UP': 0.953, 'BIAS_DOWN': 0.986, 'BIAS_NEUTRAL': 0.464}
2026-05-02 08:57:22,227 INFO Regime epoch 41/50 — tr=0.7695 va=0.5490 acc=0.532 bal=0.801
2026-05-02 08:57:22,256 INFO Regime epoch 42/50 — tr=0.7649 va=0.5486 acc=0.532 bal=0.802
2026-05-02 08:57:22,285 INFO Regime epoch 43/50 — tr=0.7648 va=0.5490 acc=0.531 bal=0.802
2026-05-02 08:57:22,313 INFO Regime epoch 44/50 — tr=0.7613 va=0.5492 acc=0.530 bal=0.801
2026-05-02 08:57:22,343 INFO Regime epoch 45/50 — tr=0.7682 va=0.5497 acc=0.529 bal=0.802 per_class={'BIAS_UP': 0.958, 'BIAS_DOWN': 0.985, 'BIAS_NEUTRAL': 0.463}
2026-05-02 08:57:22,374 INFO Regime epoch 46/50 — tr=0.7650 va=0.5495 acc=0.530 bal=0.802
2026-05-02 08:57:22,404 INFO Regime epoch 47/50 — tr=0.7660 va=0.5499 acc=0.529 bal=0.802
2026-05-02 08:57:22,436 INFO Regime epoch 48/50 — tr=0.7665 va=0.5498 acc=0.529 bal=0.802
2026-05-02 08:57:22,466 INFO Regime epoch 49/50 — tr=0.7691 va=0.5496 acc=0.529 bal=0.802
2026-05-02 08:57:22,498 INFO Regime epoch 50/50 — tr=0.7623 va=0.5499 acc=0.528 bal=0.802 per_class={'BIAS_UP': 0.958, 'BIAS_DOWN': 0.986, 'BIAS_NEUTRAL': 0.461}
2026-05-02 08:57:22,519 INFO RegimeClassifier[mode=htf_bias] validation precision={'BIAS_UP': 0.231, 'BIAS_DOWN': 0.203, 'BIAS_NEUTRAL': 0.992} recall={'BIAS_UP': 0.959, 'BIAS_DOWN': 0.985, 'BIAS_NEUTRAL': 0.463} f1={'BIAS_UP': 0.373, 'BIAS_DOWN': 0.336, 'BIAS_NEUTRAL': 0.631} confusion=[[836, 1, 35], [0, 1207, 18], [2780, 4743, 6490]]
2026-05-02 08:57:22,524 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-02 08:57:22,524 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-02 08:57:22,524 INFO Regime phase HTF train fold=fold_000: 6.2s
2026-05-02 08:57:22,627 INFO Regime HTF complete fold=fold_000: acc=0.530, train=32884 val=16110 per_class={'BIAS_UP': 0.959, 'BIAS_DOWN': 0.985, 'BIAS_NEUTRAL': 0.463}
2026-05-02 08:57:22,629 INFO Loaded XAUUSD/4H split=train fold=fold_000: 3204 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:22,735 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 350, 'BIAS_DOWN': 271, 'BIAS_NEUTRAL': 2583}  ambiguous=2166 (total=3204) horizon=12
2026-05-02 08:57:22,742 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 5.384615384615385, 'BIAS_DOWN': 4.839285714285714, 'BIAS_NEUTRAL': 21.172131147540984}
2026-05-02 08:57:22,745 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 350, 'mean': 0.0011729262330679962, 'mean_over_std': 0.3838722939899478}, 'BIAS_DOWN': {'n': 271, 'mean': -0.0011149048370300268, 'mean_over_std': -0.3830594372267871}, 'BIAS_NEUTRAL': {'n': 2582, 'mean': 4.228327959398133e-05, 'mean_over_std': 0.01196518773704054}}
2026-05-02 08:57:22,745 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 350, 'mean': 0.0011729262330679962, 'mean_over_std': 0.3838722939899478}, 'BIAS_DOWN': {'n': 271, 'mean': -0.0011149048370300268, 'mean_over_std': -0.3830594372267871}, 'BIAS_NEUTRAL': {'n': 417, 'mean': 0.0001236439934333611, 'mean_over_std': 0.043497130931069335}}
2026-05-02 08:57:22,754 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-02 08:57:22,756 INFO Loaded AUDUSD/1H split=train fold=fold_000: 11773 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:22,757 INFO Loaded EURGBP/1H split=train fold=fold_000: 11773 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:22,758 INFO Loaded EURJPY/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:22,760 INFO Loaded EURUSD/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:22,761 INFO Loaded GBPJPY/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:22,763 INFO Loaded GBPUSD/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:22,764 INFO Loaded NZDUSD/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:22,766 INFO Loaded USDCAD/1H split=train fold=fold_000: 11773 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:22,767 INFO Loaded USDCHF/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:22,769 INFO Loaded USDJPY/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:22,772 INFO Loaded XAUUSD/1H split=train fold=fold_000: 11914 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:22,784 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:22,790 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:22,791 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:22,792 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:22,792 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:22,796 INFO Loaded AUDUSD/1H split=train fold=fold_000: 11773 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:23,109 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected AUDUSD — 11723 samples (group=dollar) score_means={'trend_score': 0.4834, 'range_score': 0.2374, 'chop_score': 0.4688, 'volatility_percentile': 0.3652, 'consolidation_score': 0.2}
2026-05-02 08:57:23,223 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:23,229 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:23,231 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:23,232 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:23,232 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:23,234 INFO Loaded EURGBP/1H split=train fold=fold_000: 11773 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:23,518 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected EURGBP — 11723 samples (group=cross) score_means={'trend_score': 0.497, 'range_score': 0.2358, 'chop_score': 0.4623, 'volatility_percentile': 0.3783, 'consolidation_score': 0.1905}
2026-05-02 08:57:23,627 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:23,630 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:23,632 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:23,632 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:23,633 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:23,634 INFO Loaded EURJPY/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:23,938 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected EURJPY — 11722 samples (group=cross) score_means={'trend_score': 0.4873, 'range_score': 0.2384, 'chop_score': 0.4674, 'volatility_percentile': 0.3763, 'consolidation_score': 0.1925}
2026-05-02 08:57:24,048 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:24,050 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:24,052 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:24,052 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:24,053 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:24,054 INFO Loaded EURUSD/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:24,357 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected EURUSD — 11722 samples (group=dollar) score_means={'trend_score': 0.4931, 'range_score': 0.2373, 'chop_score': 0.464, 'volatility_percentile': 0.3716, 'consolidation_score': 0.1896}
2026-05-02 08:57:24,468 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:24,471 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:24,472 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:24,473 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:24,473 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:24,475 INFO Loaded GBPJPY/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:24,780 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected GBPJPY — 11722 samples (group=cross) score_means={'trend_score': 0.5009, 'range_score': 0.2311, 'chop_score': 0.4571, 'volatility_percentile': 0.3758, 'consolidation_score': 0.1946}
2026-05-02 08:57:24,892 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:24,894 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:24,896 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:24,896 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:24,897 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:24,899 INFO Loaded GBPUSD/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:25,198 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected GBPUSD — 11722 samples (group=dollar) score_means={'trend_score': 0.5037, 'range_score': 0.2323, 'chop_score': 0.4563, 'volatility_percentile': 0.3792, 'consolidation_score': 0.186}
2026-05-02 08:57:25,304 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:25,306 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:25,306 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:25,307 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:25,307 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:25,308 INFO Loaded NZDUSD/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:25,605 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected NZDUSD — 11722 samples (group=dollar) score_means={'trend_score': 0.4841, 'range_score': 0.2391, 'chop_score': 0.4687, 'volatility_percentile': 0.3726, 'consolidation_score': 0.1911}
2026-05-02 08:57:25,712 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:25,715 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:25,715 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:25,716 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:25,716 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:25,718 INFO Loaded USDCAD/1H split=train fold=fold_000: 11773 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:26,017 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected USDCAD — 11723 samples (group=dollar) score_means={'trend_score': 0.4974, 'range_score': 0.2331, 'chop_score': 0.4561, 'volatility_percentile': 0.3775, 'consolidation_score': 0.1896}
2026-05-02 08:57:26,125 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:26,128 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:26,129 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:26,130 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:26,130 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:26,132 INFO Loaded USDCHF/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:26,420 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected USDCHF — 11722 samples (group=dollar) score_means={'trend_score': 0.4674, 'range_score': 0.2504, 'chop_score': 0.4822, 'volatility_percentile': 0.3731, 'consolidation_score': 0.1894}
2026-05-02 08:57:26,524 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:26,527 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:26,527 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:26,528 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:26,528 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:26,530 INFO Loaded USDJPY/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:26,829 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected USDJPY — 11722 samples (group=dollar) score_means={'trend_score': 0.4991, 'range_score': 0.231, 'chop_score': 0.4562, 'volatility_percentile': 0.3679, 'consolidation_score': 0.1984}
2026-05-02 08:57:26,950 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:26,953 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:26,955 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:26,955 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:26,955 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:26,958 INFO Loaded XAUUSD/1H split=train fold=fold_000: 11914 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:27,268 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected XAUUSD — 11864 samples (group=gold) score_means={'trend_score': 0.5075, 'range_score': 0.2273, 'chop_score': 0.4518, 'volatility_percentile': 0.3694, 'consolidation_score': 0.1952}
2026-05-02 08:57:27,373 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4951, 'range_score': 0.2351, 'chop_score': 0.4622, 'volatility_percentile': 0.3768, 'consolidation_score': 0.1925}, 'dollar': {'trend_score': 0.4897, 'range_score': 0.2372, 'chop_score': 0.4646, 'volatility_percentile': 0.3724, 'consolidation_score': 0.192}, 'gold': {'trend_score': 0.5075, 'range_score': 0.2273, 'chop_score': 0.4518, 'volatility_percentile': 0.3694, 'consolidation_score': 0.1952}}
2026-05-02 08:57:27,373 INFO Regime[1H mode=ltf_behaviour] score means by year: {2016: {'trend_score': 0.4914, 'range_score': 0.2348, 'chop_score': 0.4627, 'volatility_percentile': 0.375, 'consolidation_score': 0.192}, 2017: {'trend_score': 0.4941, 'range_score': 0.2364, 'chop_score': 0.463, 'volatility_percentile': 0.3716, 'consolidation_score': 0.1934}, 2018: {'trend_score': 0.51, 'range_score': 0.2569, 'chop_score': 0.4423, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1324}}
2026-05-02 08:57:27,456 INFO Loaded AUDUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:27,458 INFO Loaded EURGBP/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:27,459 INFO Loaded EURJPY/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:27,460 INFO Loaded EURUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:27,462 INFO Loaded GBPJPY/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:27,463 INFO Loaded GBPUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:27,464 INFO Loaded NZDUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:27,465 INFO Loaded USDCAD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:27,466 INFO Loaded USDCHF/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:27,467 INFO Loaded USDJPY/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:27,469 INFO Loaded XAUUSD/1H split=val fold=fold_000: 6034 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:27,475 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:27,477 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:27,478 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:27,479 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:27,479 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:27,480 INFO Loaded AUDUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:27,695 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected AUDUSD — 5812 samples (group=dollar) score_means={'trend_score': 0.484, 'range_score': 0.2467, 'chop_score': 0.4726, 'volatility_percentile': 0.3956, 'consolidation_score': 0.1777}
2026-05-02 08:57:27,803 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:27,806 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:27,807 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:27,807 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:27,807 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:27,809 INFO Loaded EURGBP/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:28,024 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected EURGBP — 5812 samples (group=cross) score_means={'trend_score': 0.4626, 'range_score': 0.2497, 'chop_score': 0.4853, 'volatility_percentile': 0.3975, 'consolidation_score': 0.1692}
2026-05-02 08:57:28,132 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:28,135 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:28,135 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:28,136 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:28,136 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:28,138 INFO Loaded EURJPY/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:28,346 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected EURJPY — 5812 samples (group=cross) score_means={'trend_score': 0.4749, 'range_score': 0.2394, 'chop_score': 0.474, 'volatility_percentile': 0.3878, 'consolidation_score': 0.1827}
2026-05-02 08:57:28,454 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:28,457 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:28,457 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:28,458 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:28,458 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:28,460 INFO Loaded EURUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:28,668 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected EURUSD — 5812 samples (group=dollar) score_means={'trend_score': 0.4993, 'range_score': 0.2343, 'chop_score': 0.4572, 'volatility_percentile': 0.389, 'consolidation_score': 0.1807}
2026-05-02 08:57:28,780 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:28,784 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:28,785 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:28,785 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:28,785 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:28,787 INFO Loaded GBPJPY/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:29,008 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected GBPJPY — 5812 samples (group=cross) score_means={'trend_score': 0.4888, 'range_score': 0.2412, 'chop_score': 0.4689, 'volatility_percentile': 0.3963, 'consolidation_score': 0.1732}
2026-05-02 08:57:29,114 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:29,117 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:29,117 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:29,118 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:29,118 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:29,120 INFO Loaded GBPUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:29,325 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected GBPUSD — 5812 samples (group=dollar) score_means={'trend_score': 0.5007, 'range_score': 0.2339, 'chop_score': 0.4559, 'volatility_percentile': 0.3971, 'consolidation_score': 0.1718}
2026-05-02 08:57:29,427 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:29,429 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:29,429 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:29,430 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:29,430 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:29,431 INFO Loaded NZDUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:29,641 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected NZDUSD — 5812 samples (group=dollar) score_means={'trend_score': 0.4931, 'range_score': 0.2353, 'chop_score': 0.4587, 'volatility_percentile': 0.3902, 'consolidation_score': 0.1824}
2026-05-02 08:57:29,746 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:29,748 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:29,749 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:29,749 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:29,750 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:29,751 INFO Loaded USDCAD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:29,962 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected USDCAD — 5812 samples (group=dollar) score_means={'trend_score': 0.4808, 'range_score': 0.2476, 'chop_score': 0.4717, 'volatility_percentile': 0.3857, 'consolidation_score': 0.1768}
2026-05-02 08:57:30,069 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:30,072 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:30,073 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:30,073 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:30,073 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:30,075 INFO Loaded USDCHF/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:30,287 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected USDCHF — 5812 samples (group=dollar) score_means={'trend_score': 0.4799, 'range_score': 0.2431, 'chop_score': 0.4697, 'volatility_percentile': 0.3907, 'consolidation_score': 0.1794}
2026-05-02 08:57:30,394 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:30,396 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:30,397 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:30,397 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:30,397 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:30,399 INFO Loaded USDJPY/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:30,630 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected USDJPY — 5812 samples (group=dollar) score_means={'trend_score': 0.4943, 'range_score': 0.2334, 'chop_score': 0.4614, 'volatility_percentile': 0.3872, 'consolidation_score': 0.1806}
2026-05-02 08:57:30,745 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:30,749 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:30,750 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:30,751 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:30,751 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:30,753 INFO Loaded XAUUSD/1H split=val fold=fold_000: 6034 bars (2018-01-04 → 2019-01-03)
2026-05-02 08:57:31,013 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected XAUUSD — 5984 samples (group=gold) score_means={'trend_score': 0.4716, 'range_score': 0.2479, 'chop_score': 0.4761, 'volatility_percentile': 0.3856, 'consolidation_score': 0.1812}
2026-05-02 08:57:31,115 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4754, 'range_score': 0.2434, 'chop_score': 0.476, 'volatility_percentile': 0.3939, 'consolidation_score': 0.175}, 'dollar': {'trend_score': 0.4903, 'range_score': 0.2392, 'chop_score': 0.4639, 'volatility_percentile': 0.3908, 'consolidation_score': 0.1785}, 'gold': {'trend_score': 0.4716, 'range_score': 0.2479, 'chop_score': 0.4761, 'volatility_percentile': 0.3856, 'consolidation_score': 0.1812}}
2026-05-02 08:57:31,115 INFO Regime[1H mode=ltf_behaviour] score means by year: {2018: {'trend_score': 0.4841, 'range_score': 0.2416, 'chop_score': 0.4687, 'volatility_percentile': 0.3892, 'consolidation_score': 0.1792}, 2019: {'trend_score': 0.5315, 'range_score': 0.1889, 'chop_score': 0.4263, 'volatility_percentile': 0.5999, 'consolidation_score': 0.0339}}
2026-05-02 08:57:31,194 INFO Regime phase LTF dataset build fold=fold_000: 8.4s (train=129087 val=64104)
2026-05-02 08:57:31,211 INFO RegimeClassifier[mode=ltf_behaviour]: cold start score head
2026-05-02 08:57:31,212 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-02 08:57:31,520 INFO Regime score epoch  1/50 — tr=0.0732 va=0.0621 mae={'trend_score': 0.1651, 'range_score': 0.2482, 'chop_score': 0.1458, 'volatility_percentile': 0.1768, 'consolidation_score': 0.3022}
2026-05-02 08:57:31,800 INFO Regime score epoch  2/50 — tr=0.0666 va=0.0538
2026-05-02 08:57:32,063 INFO Regime score epoch  3/50 — tr=0.0558 va=0.0424
2026-05-02 08:57:32,334 INFO Regime score epoch  4/50 — tr=0.0430 va=0.0310
2026-05-02 08:57:32,608 INFO Regime score epoch  5/50 — tr=0.0316 va=0.0222 mae={'trend_score': 0.0763, 'range_score': 0.1633, 'chop_score': 0.0697, 'volatility_percentile': 0.0809, 'consolidation_score': 0.2029}
2026-05-02 08:57:32,883 INFO Regime score epoch  6/50 — tr=0.0238 va=0.0164
2026-05-02 08:57:33,148 INFO Regime score epoch  7/50 — tr=0.0188 va=0.0120
2026-05-02 08:57:33,420 INFO Regime score epoch  8/50 — tr=0.0155 va=0.0090
2026-05-02 08:57:33,685 INFO Regime score epoch  9/50 — tr=0.0134 va=0.0073
2026-05-02 08:57:33,948 INFO Regime score epoch 10/50 — tr=0.0120 va=0.0061 mae={'trend_score': 0.0611, 'range_score': 0.0662, 'chop_score': 0.056, 'volatility_percentile': 0.0372, 'consolidation_score': 0.0877}
2026-05-02 08:57:34,211 INFO Regime score epoch 11/50 — tr=0.0110 va=0.0054
2026-05-02 08:57:34,476 INFO Regime score epoch 12/50 — tr=0.0103 va=0.0049
2026-05-02 08:57:34,746 INFO Regime score epoch 13/50 — tr=0.0097 va=0.0046
2026-05-02 08:57:35,028 INFO Regime score epoch 14/50 — tr=0.0093 va=0.0043
2026-05-02 08:57:35,306 INFO Regime score epoch 15/50 — tr=0.0090 va=0.0041 mae={'trend_score': 0.055, 'range_score': 0.0557, 'chop_score': 0.0518, 'volatility_percentile': 0.0313, 'consolidation_score': 0.0585}
2026-05-02 08:57:35,590 INFO Regime score epoch 16/50 — tr=0.0087 va=0.0040
2026-05-02 08:57:35,857 INFO Regime score epoch 17/50 — tr=0.0084 va=0.0038
2026-05-02 08:57:36,138 INFO Regime score epoch 18/50 — tr=0.0082 va=0.0037
2026-05-02 08:57:36,409 INFO Regime score epoch 19/50 — tr=0.0080 va=0.0036
2026-05-02 08:57:36,666 INFO Regime score epoch 20/50 — tr=0.0078 va=0.0035 mae={'trend_score': 0.0502, 'range_score': 0.054, 'chop_score': 0.0497, 'volatility_percentile': 0.0291, 'consolidation_score': 0.0469}
2026-05-02 08:57:36,925 INFO Regime score epoch 21/50 — tr=0.0077 va=0.0034
2026-05-02 08:57:37,196 INFO Regime score epoch 22/50 — tr=0.0076 va=0.0033
2026-05-02 08:57:37,471 INFO Regime score epoch 23/50 — tr=0.0074 va=0.0033
2026-05-02 08:57:37,740 INFO Regime score epoch 24/50 — tr=0.0073 va=0.0032
2026-05-02 08:57:38,013 INFO Regime score epoch 25/50 — tr=0.0072 va=0.0031 mae={'trend_score': 0.0474, 'range_score': 0.0522, 'chop_score': 0.0488, 'volatility_percentile': 0.0281, 'consolidation_score': 0.0415}
2026-05-02 08:57:38,276 INFO Regime score epoch 26/50 — tr=0.0071 va=0.0031
2026-05-02 08:57:38,544 INFO Regime score epoch 27/50 — tr=0.0070 va=0.0030
2026-05-02 08:57:38,811 INFO Regime score epoch 28/50 — tr=0.0070 va=0.0030
2026-05-02 08:57:39,087 INFO Regime score epoch 29/50 — tr=0.0069 va=0.0030
2026-05-02 08:57:39,348 INFO Regime score epoch 30/50 — tr=0.0068 va=0.0029 mae={'trend_score': 0.0454, 'range_score': 0.0511, 'chop_score': 0.0477, 'volatility_percentile': 0.0268, 'consolidation_score': 0.039}
2026-05-02 08:57:39,613 INFO Regime score epoch 31/50 — tr=0.0067 va=0.0029
2026-05-02 08:57:39,889 INFO Regime score epoch 32/50 — tr=0.0067 va=0.0029
2026-05-02 08:57:40,148 INFO Regime score epoch 33/50 — tr=0.0067 va=0.0028
2026-05-02 08:57:40,416 INFO Regime score epoch 34/50 — tr=0.0066 va=0.0028
2026-05-02 08:57:40,686 INFO Regime score epoch 35/50 — tr=0.0066 va=0.0028 mae={'trend_score': 0.044, 'range_score': 0.0495, 'chop_score': 0.0468, 'volatility_percentile': 0.0265, 'consolidation_score': 0.0364}
2026-05-02 08:57:40,985 INFO Regime score epoch 36/50 — tr=0.0065 va=0.0028
2026-05-02 08:57:41,253 INFO Regime score epoch 37/50 — tr=0.0065 va=0.0027
2026-05-02 08:57:41,518 INFO Regime score epoch 38/50 — tr=0.0065 va=0.0027
2026-05-02 08:57:41,787 INFO Regime score epoch 39/50 — tr=0.0065 va=0.0027
2026-05-02 08:57:42,056 INFO Regime score epoch 40/50 — tr=0.0064 va=0.0027 mae={'trend_score': 0.043, 'range_score': 0.0492, 'chop_score': 0.0462, 'volatility_percentile': 0.026, 'consolidation_score': 0.0358}
2026-05-02 08:57:42,318 INFO Regime score epoch 41/50 — tr=0.0064 va=0.0027
2026-05-02 08:57:42,583 INFO Regime score epoch 42/50 — tr=0.0064 va=0.0027
2026-05-02 08:57:42,854 INFO Regime score epoch 43/50 — tr=0.0064 va=0.0027
2026-05-02 08:57:43,124 INFO Regime score epoch 44/50 — tr=0.0064 va=0.0027
2026-05-02 08:57:43,390 INFO Regime score epoch 45/50 — tr=0.0064 va=0.0027 mae={'trend_score': 0.0429, 'range_score': 0.0493, 'chop_score': 0.0461, 'volatility_percentile': 0.0256, 'consolidation_score': 0.0363}
2026-05-02 08:57:43,662 INFO Regime score epoch 46/50 — tr=0.0064 va=0.0027
2026-05-02 08:57:43,947 INFO Regime score epoch 47/50 — tr=0.0064 va=0.0027
2026-05-02 08:57:44,226 INFO Regime score epoch 48/50 — tr=0.0064 va=0.0027
2026-05-02 08:57:44,493 INFO Regime score epoch 49/50 — tr=0.0064 va=0.0027
2026-05-02 08:57:44,756 INFO Regime score epoch 50/50 — tr=0.0064 va=0.0027 mae={'trend_score': 0.0425, 'range_score': 0.0492, 'chop_score': 0.0457, 'volatility_percentile': 0.0263, 'consolidation_score': 0.035}
2026-05-02 08:57:44,798 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0425, 'range_score': 0.0492, 'chop_score': 0.0457, 'volatility_percentile': 0.0263, 'consolidation_score': 0.035} mse={'trend_score': 0.00288, 'range_score': 0.00379, 'chop_score': 0.00324, 'volatility_percentile': 0.00128, 'consolidation_score': 0.00217} corr={'trend_score': 0.9707, 'range_score': 0.9097, 'chop_score': 0.9567, 'volatility_percentile': 0.9861, 'consolidation_score': 0.9753} pred_std={'trend_score': 0.2052, 'range_score': 0.1404, 'chop_score': 0.1744, 'volatility_percentile': 0.2059, 'consolidation_score': 0.2038} target_std={'trend_score': 0.2203, 'range_score': 0.1457, 'chop_score': 0.1926, 'volatility_percentile': 0.2123, 'consolidation_score': 0.2089}
2026-05-02 08:57:44,803 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-02 08:57:44,804 INFO Regime phase LTF train fold=fold_000: 13.6s
2026-05-02 08:57:44,905 INFO Regime LTF complete fold=fold_000: score_accuracy=0.960, train=129087 val=64104 mae={'trend_score': 0.0425, 'range_score': 0.0492, 'chop_score': 0.0457, 'volatility_percentile': 0.0263, 'consolidation_score': 0.035}
2026-05-02 08:57:44,907 INFO Loaded XAUUSD/1H split=train fold=fold_000: 11914 bars (2016-01-04 → 2018-01-03)
2026-05-02 08:57:45,039 INFO Regime[1H mode=ltf_behaviour fold=fold_000] LTF score diagnostics on XAUUSD:
{'trend_score': {'mean': 0.507, 'q10': 0.2031, 'q50': 0.5032, 'q90': 0.8121}, 'range_score': {'mean': 0.2284, 'q10': 0.0527, 'q50': 0.2, 'q90': 0.4305}, 'chop_score': {'mean': 0.4525, 'q10': 0.2007, 'q50': 0.4407, 'q90': 0.7194}, 'volatility_percentile': {'mean': 0.3694, 'q10': 0.0827, 'q50': 0.3584, 'q90': 0.6692}, 'consolidation_score': {'mean': 0.1944, 'q10': 0.0, 'q50': 0.1206, 'q90': 0.5428}}
2026-05-02 08:57:45,048 INFO === Regime rolling fold 2/3: fold_001 ===
2026-05-02 08:57:45,048 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-05-02 08:57:45,048 INFO Split boundaries loaded fold=fold_001/3 — train 2018-01-04→2020-01-03  val 2020-01-06→2020-12-31  test 2023-08-07→2025-08-05
2026-05-02 08:57:45,049 INFO Loaded AUDUSD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-02 08:57:45,050 INFO Loaded EURGBP/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-02 08:57:45,051 INFO Loaded EURJPY/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-02 08:57:45,051 INFO Loaded EURUSD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-02 08:57:45,052 INFO Loaded GBPJPY/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
2026-05-02 08:57:45,053 INFO Loaded GBPUSD/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
2026-05-02 08:57:45,054 INFO Loaded NZDUSD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-02 08:57:45,054 INFO Loaded USDCAD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-02 08:57:45,055 INFO Loaded USDCHF/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-02 08:57:45,056 INFO Loaded USDJPY/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
2026-05-02 08:57:45,057 INFO Loaded XAUUSD/4H split=train fold=fold_001: 3193 bars (2018-01-04 → 2020-01-03)
2026-05-02 08:57:45,062 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:45,064 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:45,064 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:45,065 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:45,065 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:45,066 INFO Loaded AUDUSD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-02 08:57:45,245 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 124, 'BIAS_DOWN': 254, 'BIAS_NEUTRAL': 2628}  ambiguous=2177 (total=3006) horizon=12
2026-05-02 08:57:45,246 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected AUDUSD — 2956 samples (group=dollar) labels={'BIAS_UP': 124, 'BIAS_DOWN': 254, 'BIAS_NEUTRAL': 2578} clean={'BIAS_UP': 124, 'BIAS_DOWN': 254, 'BIAS_NEUTRAL': 441}
2026-05-02 08:57:45,350 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:45,352 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:45,353 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:45,353 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:45,353 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:45,354 INFO Loaded EURGBP/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-02 08:57:45,541 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 135, 'BIAS_DOWN': 195, 'BIAS_NEUTRAL': 2676}  ambiguous=2149 (total=3006) horizon=12
2026-05-02 08:57:45,542 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected EURGBP — 2956 samples (group=cross) labels={'BIAS_UP': 135, 'BIAS_DOWN': 195, 'BIAS_NEUTRAL': 2626} clean={'BIAS_UP': 135, 'BIAS_DOWN': 195, 'BIAS_NEUTRAL': 521}
2026-05-02 08:57:45,647 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:45,649 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:45,650 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:45,650 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:45,651 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:45,652 INFO Loaded EURJPY/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-02 08:57:45,840 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 105, 'BIAS_DOWN': 224, 'BIAS_NEUTRAL': 2677}  ambiguous=2200 (total=3006) horizon=12
2026-05-02 08:57:45,841 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected EURJPY — 2956 samples (group=cross) labels={'BIAS_UP': 105, 'BIAS_DOWN': 224, 'BIAS_NEUTRAL': 2627} clean={'BIAS_UP': 105, 'BIAS_DOWN': 224, 'BIAS_NEUTRAL': 471}
2026-05-02 08:57:45,948 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:45,951 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:45,951 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:45,952 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:45,952 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:45,953 INFO Loaded EURUSD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-02 08:57:46,140 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 104, 'BIAS_DOWN': 185, 'BIAS_NEUTRAL': 2717}  ambiguous=2308 (total=3006) horizon=12
2026-05-02 08:57:46,141 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected EURUSD — 2956 samples (group=dollar) labels={'BIAS_UP': 104, 'BIAS_DOWN': 185, 'BIAS_NEUTRAL': 2667} clean={'BIAS_UP': 104, 'BIAS_DOWN': 185, 'BIAS_NEUTRAL': 400}
2026-05-02 08:57:46,246 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:46,249 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:46,249 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:46,250 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:46,250 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:46,251 INFO Loaded GBPJPY/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
2026-05-02 08:57:46,437 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 203, 'BIAS_DOWN': 272, 'BIAS_NEUTRAL': 2532}  ambiguous=2087 (total=3007) horizon=12
2026-05-02 08:57:46,439 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected GBPJPY — 2957 samples (group=cross) labels={'BIAS_UP': 203, 'BIAS_DOWN': 272, 'BIAS_NEUTRAL': 2482} clean={'BIAS_UP': 203, 'BIAS_DOWN': 272, 'BIAS_NEUTRAL': 441}
2026-05-02 08:57:46,546 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:46,548 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:46,549 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:46,549 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:46,549 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:46,550 INFO Loaded GBPUSD/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
2026-05-02 08:57:46,730 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 184, 'BIAS_DOWN': 241, 'BIAS_NEUTRAL': 2582}  ambiguous=2191 (total=3007) horizon=12
2026-05-02 08:57:46,732 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected GBPUSD — 2957 samples (group=dollar) labels={'BIAS_UP': 184, 'BIAS_DOWN': 241, 'BIAS_NEUTRAL': 2532} clean={'BIAS_UP': 184, 'BIAS_DOWN': 241, 'BIAS_NEUTRAL': 387}
2026-05-02 08:57:46,834 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:46,836 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:46,837 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:46,837 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:46,837 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:46,838 INFO Loaded NZDUSD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-02 08:57:47,022 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 153, 'BIAS_DOWN': 281, 'BIAS_NEUTRAL': 2572}  ambiguous=2215 (total=3006) horizon=12
2026-05-02 08:57:47,024 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected NZDUSD — 2956 samples (group=dollar) labels={'BIAS_UP': 153, 'BIAS_DOWN': 281, 'BIAS_NEUTRAL': 2522} clean={'BIAS_UP': 153, 'BIAS_DOWN': 281, 'BIAS_NEUTRAL': 353}
2026-05-02 08:57:47,128 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:47,130 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:47,131 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:47,131 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:47,132 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:47,133 INFO Loaded USDCAD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-02 08:57:47,317 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 185, 'BIAS_DOWN': 158, 'BIAS_NEUTRAL': 2663}  ambiguous=2249 (total=3006) horizon=12
2026-05-02 08:57:47,318 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected USDCAD — 2956 samples (group=dollar) labels={'BIAS_UP': 185, 'BIAS_DOWN': 158, 'BIAS_NEUTRAL': 2613} clean={'BIAS_UP': 185, 'BIAS_DOWN': 158, 'BIAS_NEUTRAL': 409}
2026-05-02 08:57:47,423 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:47,425 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:47,426 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:47,426 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:47,427 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:47,428 INFO Loaded USDCHF/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-02 08:57:47,610 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 211, 'BIAS_DOWN': 148, 'BIAS_NEUTRAL': 2647}  ambiguous=2181 (total=3006) horizon=12
2026-05-02 08:57:47,611 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected USDCHF — 2956 samples (group=dollar) labels={'BIAS_UP': 211, 'BIAS_DOWN': 148, 'BIAS_NEUTRAL': 2597} clean={'BIAS_UP': 211, 'BIAS_DOWN': 148, 'BIAS_NEUTRAL': 457}
2026-05-02 08:57:47,718 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:47,720 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:47,721 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:47,721 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:47,722 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:47,722 INFO Loaded USDJPY/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
2026-05-02 08:57:47,914 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 182, 'BIAS_DOWN': 144, 'BIAS_NEUTRAL': 2681}  ambiguous=2215 (total=3007) horizon=12
2026-05-02 08:57:47,915 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected USDJPY — 2957 samples (group=dollar) labels={'BIAS_UP': 182, 'BIAS_DOWN': 144, 'BIAS_NEUTRAL': 2631} clean={'BIAS_UP': 182, 'BIAS_DOWN': 144, 'BIAS_NEUTRAL': 463}
2026-05-02 08:57:48,031 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:48,034 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:48,035 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:48,036 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:48,036 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:48,037 INFO Loaded XAUUSD/4H split=train fold=fold_001: 3193 bars (2018-01-04 → 2020-01-03)
2026-05-02 08:57:48,239 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 215, 'BIAS_DOWN': 113, 'BIAS_NEUTRAL': 2865}  ambiguous=2302 (total=3193) horizon=12
2026-05-02 08:57:48,241 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected XAUUSD — 3143 samples (group=gold) labels={'BIAS_UP': 215, 'BIAS_DOWN': 113, 'BIAS_NEUTRAL': 2815} clean={'BIAS_UP': 215, 'BIAS_DOWN': 113, 'BIAS_NEUTRAL': 557}
2026-05-02 08:57:48,342 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 443, 'BIAS_DOWN': 691, 'BIAS_NEUTRAL': 7735}, 'dollar': {'BIAS_UP': 1143, 'BIAS_DOWN': 1411, 'BIAS_NEUTRAL': 18140}, 'gold': {'BIAS_UP': 215, 'BIAS_DOWN': 113, 'BIAS_NEUTRAL': 2815}}
2026-05-02 08:57:48,342 INFO Regime[4H mode=htf_bias] label distribution by year: {2018: {'BIAS_UP': 876, 'BIAS_DOWN': 1219, 'BIAS_NEUTRAL': 13861}, 2019: {'BIAS_UP': 924, 'BIAS_DOWN': 995, 'BIAS_NEUTRAL': 14688}, 2020: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 141}}
2026-05-02 08:57:48,418 INFO Loaded AUDUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-02 08:57:48,419 INFO Loaded EURGBP/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-02 08:57:48,420 INFO Loaded EURJPY/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-02 08:57:48,421 INFO Loaded EURUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-02 08:57:48,422 INFO Loaded GBPJPY/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-02 08:57:48,423 INFO Loaded GBPUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-02 08:57:48,423 INFO Loaded NZDUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-02 08:57:48,424 INFO Loaded USDCAD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-02 08:57:48,425 INFO Loaded USDCHF/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-02 08:57:48,426 INFO Loaded USDJPY/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-02 08:57:48,427 INFO Loaded XAUUSD/4H split=val fold=fold_001: 1581 bars (2020-01-06 → 2020-12-31)
2026-05-02 08:57:48,433 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:48,435 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:48,436 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:48,437 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:48,437 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:48,438 INFO Loaded AUDUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-02 08:57:48,603 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 106, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1299}  ambiguous=1069 (total=1490) horizon=12
2026-05-02 08:57:48,604 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected AUDUSD — 1440 samples (group=dollar) labels={'BIAS_UP': 106, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1249} clean={'BIAS_UP': 106, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 216}
2026-05-02 08:57:48,709 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:48,712 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:48,713 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:48,713 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:48,713 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:48,714 INFO Loaded EURGBP/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-02 08:57:48,887 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 124, 'BIAS_DOWN': 36, 'BIAS_NEUTRAL': 1330}  ambiguous=1118 (total=1490) horizon=12
2026-05-02 08:57:48,889 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected EURGBP — 1440 samples (group=cross) labels={'BIAS_UP': 124, 'BIAS_DOWN': 36, 'BIAS_NEUTRAL': 1280} clean={'BIAS_UP': 124, 'BIAS_DOWN': 36, 'BIAS_NEUTRAL': 198}
2026-05-02 08:57:48,993 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:48,995 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:48,996 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:48,996 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:48,996 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:48,997 INFO Loaded EURJPY/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-02 08:57:49,161 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 97, 'BIAS_DOWN': 50, 'BIAS_NEUTRAL': 1343}  ambiguous=1113 (total=1490) horizon=12
2026-05-02 08:57:49,162 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected EURJPY — 1440 samples (group=cross) labels={'BIAS_UP': 97, 'BIAS_DOWN': 50, 'BIAS_NEUTRAL': 1293} clean={'BIAS_UP': 97, 'BIAS_DOWN': 50, 'BIAS_NEUTRAL': 226}
2026-05-02 08:57:49,265 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:49,267 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:49,268 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:49,268 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:49,269 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:49,270 INFO Loaded EURUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-02 08:57:49,431 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 172, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 1251}  ambiguous=1060 (total=1490) horizon=12
2026-05-02 08:57:49,432 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected EURUSD — 1440 samples (group=dollar) labels={'BIAS_UP': 172, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 1201} clean={'BIAS_UP': 172, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 182}
2026-05-02 08:57:49,536 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:49,538 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:49,539 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:49,540 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:49,540 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:49,541 INFO Loaded GBPJPY/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-02 08:57:49,706 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 71, 'BIAS_DOWN': 100, 'BIAS_NEUTRAL': 1319}  ambiguous=1087 (total=1490) horizon=12
2026-05-02 08:57:49,708 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected GBPJPY — 1440 samples (group=cross) labels={'BIAS_UP': 71, 'BIAS_DOWN': 100, 'BIAS_NEUTRAL': 1269} clean={'BIAS_UP': 71, 'BIAS_DOWN': 100, 'BIAS_NEUTRAL': 215}
2026-05-02 08:57:49,813 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:49,818 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:49,819 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:49,819 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:49,820 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:49,821 INFO Loaded GBPUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-02 08:57:49,995 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 77, 'BIAS_DOWN': 58, 'BIAS_NEUTRAL': 1355}  ambiguous=1164 (total=1490) horizon=12
2026-05-02 08:57:49,996 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected GBPUSD — 1440 samples (group=dollar) labels={'BIAS_UP': 77, 'BIAS_DOWN': 58, 'BIAS_NEUTRAL': 1305} clean={'BIAS_UP': 77, 'BIAS_DOWN': 58, 'BIAS_NEUTRAL': 187}
2026-05-02 08:57:50,100 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:50,101 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:50,102 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:50,102 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:50,103 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:50,104 INFO Loaded NZDUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-02 08:57:50,269 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 129, 'BIAS_DOWN': 64, 'BIAS_NEUTRAL': 1297}  ambiguous=1085 (total=1490) horizon=12
2026-05-02 08:57:50,270 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected NZDUSD — 1440 samples (group=dollar) labels={'BIAS_UP': 129, 'BIAS_DOWN': 64, 'BIAS_NEUTRAL': 1247} clean={'BIAS_UP': 129, 'BIAS_DOWN': 64, 'BIAS_NEUTRAL': 208}
2026-05-02 08:57:50,374 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:50,377 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:50,377 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:50,378 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:50,378 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:50,379 INFO Loaded USDCAD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-02 08:57:50,551 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 106, 'BIAS_DOWN': 116, 'BIAS_NEUTRAL': 1268}  ambiguous=1037 (total=1490) horizon=12
2026-05-02 08:57:50,552 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected USDCAD — 1440 samples (group=dollar) labels={'BIAS_UP': 106, 'BIAS_DOWN': 116, 'BIAS_NEUTRAL': 1218} clean={'BIAS_UP': 106, 'BIAS_DOWN': 116, 'BIAS_NEUTRAL': 219}
2026-05-02 08:57:50,658 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:50,660 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:50,661 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:50,661 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:50,662 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:50,663 INFO Loaded USDCHF/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-02 08:57:50,855 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 28, 'BIAS_DOWN': 131, 'BIAS_NEUTRAL': 1331}  ambiguous=1081 (total=1490) horizon=12
2026-05-02 08:57:50,857 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected USDCHF — 1440 samples (group=dollar) labels={'BIAS_UP': 28, 'BIAS_DOWN': 131, 'BIAS_NEUTRAL': 1281} clean={'BIAS_UP': 28, 'BIAS_DOWN': 131, 'BIAS_NEUTRAL': 246}
2026-05-02 08:57:50,965 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:50,967 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:50,968 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:50,968 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:50,969 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:50,969 INFO Loaded USDJPY/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-02 08:57:51,133 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 37, 'BIAS_DOWN': 102, 'BIAS_NEUTRAL': 1351}  ambiguous=1119 (total=1490) horizon=12
2026-05-02 08:57:51,135 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected USDJPY — 1440 samples (group=dollar) labels={'BIAS_UP': 37, 'BIAS_DOWN': 102, 'BIAS_NEUTRAL': 1301} clean={'BIAS_UP': 37, 'BIAS_DOWN': 102, 'BIAS_NEUTRAL': 227}
2026-05-02 08:57:51,249 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:51,253 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:51,254 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:51,254 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:51,255 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:51,256 INFO Loaded XAUUSD/4H split=val fold=fold_001: 1581 bars (2020-01-06 → 2020-12-31)
2026-05-02 08:57:51,441 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 118, 'BIAS_DOWN': 34, 'BIAS_NEUTRAL': 1429}  ambiguous=1167 (total=1581) horizon=12
2026-05-02 08:57:51,442 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected XAUUSD — 1531 samples (group=gold) labels={'BIAS_UP': 118, 'BIAS_DOWN': 34, 'BIAS_NEUTRAL': 1379} clean={'BIAS_UP': 118, 'BIAS_DOWN': 34, 'BIAS_NEUTRAL': 247}
2026-05-02 08:57:51,546 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 292, 'BIAS_DOWN': 186, 'BIAS_NEUTRAL': 3842}, 'dollar': {'BIAS_UP': 655, 'BIAS_DOWN': 623, 'BIAS_NEUTRAL': 8802}, 'gold': {'BIAS_UP': 118, 'BIAS_DOWN': 34, 'BIAS_NEUTRAL': 1379}}
2026-05-02 08:57:51,546 INFO Regime[4H mode=htf_bias] label distribution by year: {2020: {'BIAS_UP': 1065, 'BIAS_DOWN': 843, 'BIAS_NEUTRAL': 14023}}
2026-05-02 08:57:51,629 INFO Regime phase HTF dataset build fold=fold_001: 6.6s (train=32706 val=15931)
2026-05-02 08:57:51,636 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=17, n_classes=3)
2026-05-02 08:57:51,637 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=8916 dropped=23790 classes={'BIAS_UP': 1801, 'BIAS_DOWN': 2215, 'BIAS_NEUTRAL': 4900})
2026-05-02 08:57:51,638 INFO RegimeClassifier[mode=htf_bias]: 8916 samples, classes={'BIAS_UP': 1801, 'BIAS_DOWN': 2215, 'BIAS_NEUTRAL': 4900}, device=cuda
2026-05-02 08:57:51,639 INFO RegimeClassifier[mode=htf_bias]: undersample class BIAS_NEUTRAL: 4900 → 3602
2026-05-02 08:57:51,639 INFO RegimeClassifier[mode=htf_bias]: after undersampling: 7618 samples classes={'BIAS_UP': 1801, 'BIAS_DOWN': 2215, 'BIAS_NEUTRAL': 3602}
2026-05-02 08:57:51,639 INFO RegimeClassifier: sample weights — mean=0.751  ambiguous(<0.4)=0.0%
2026-05-02 08:57:51,641 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-05-02 08:57:51,641 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-05-02 08:57:51,669 INFO Regime epoch  1/50 — tr=1.3448 va=0.6702 acc=0.478 bal=0.401 per_class={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.703, 'BIAS_NEUTRAL': 0.501}
2026-05-02 08:57:51,693 INFO Regime epoch  2/50 — tr=1.3390 va=0.6770 acc=0.387 bal=0.301
2026-05-02 08:57:51,714 INFO Regime epoch  3/50 — tr=1.3411 va=0.6860 acc=0.288 bal=0.229
2026-05-02 08:57:51,735 INFO Regime epoch  4/50 — tr=1.3317 va=0.6943 acc=0.212 bal=0.184
2026-05-02 08:57:51,735 INFO Regime: balanced_acc degraded 0.401→0.184 at epoch 4 — reverting to epoch-1 checkpoint to prevent collapse
2026-05-02 08:57:51,755 INFO RegimeClassifier[mode=htf_bias] validation precision={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.086, 'BIAS_NEUTRAL': 0.909} recall={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.703, 'BIAS_NEUTRAL': 0.501} f1={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.153, 'BIAS_NEUTRAL': 0.646} confusion=[[0, 404, 661], [210, 593, 40], [1066, 5934, 7023]]
2026-05-02 08:57:51,756 INFO Regime phase HTF train fold=fold_001: 0.1s
Traceback (most recent call last):
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1668, in <module>
    main()
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1627, in main
    result = retrain_regime(dry)
             ^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1170, in retrain_regime
    raise RuntimeError(f"Regime HTF training failed fold={fold_key}: {res_4h['error']}")
RuntimeError: Regime HTF training failed fold=fold_001: Regime validation below acceptance floor: accuracy=0.478 min_overall=0.363 per_class={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.703, 'BIAS_NEUTRAL': 0.501} min_class=0.100 weak_classes=['BIAS_UP']. Refusing to save misleading regime weights.
2026-05-02 08:57:54,093 ERROR retrain regime failed (exit 1)
2026-05-02 08:57:54,093 ERROR Model regime failed: exit 1
2026-05-02 08:57:54,094 WARNING Regime training failed but old regime_htf.pkl + regime_ltf.pkl are still present — continuing pipeline with existing weights. Fix the regime classifier and rerun: python scripts/retrain_incremental.py --model regime
2026-05-02 08:57:54,094 INFO --- Training gru ---
2026-05-02 08:57:54,094 INFO Running retrain --model gru
2026-05-02 08:57:54,327 INFO retrain environment: KAGGLE
2026-05-02 08:57:55,891 INFO Device: CUDA (2 GPU(s))
2026-05-02 08:57:55,903 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-02 08:57:55,903 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-02 08:57:55,903 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-02 08:57:55,903 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-02 08:57:55,903 INFO Retrain data split: train
2026-05-02 08:57:55,904 INFO Retrain rolling fold selector: latest
2026-05-02 08:57:55,905 INFO === GRU-LSTM retrain (timeframes: ['15M']) ===
2026-05-02 08:57:56,053 INFO NumExpr defaulting to 4 threads.
2026-05-02 08:57:56,242 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-02 08:57:56,242 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-02 08:57:56,243 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-02 08:57:56,243 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['15M']
2026-05-02 08:57:56,243 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260502_085756
2026-05-02 08:57:56,246 WARNING WeightsManifest: no manifest at /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json — assuming compatible (run a full retrain to generate one)
2026-05-02 08:57:56,397 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:56,416 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:56,430 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:56,437 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:56,479 INFO Split boundaries loaded fold=fold_002/3 — train 2020-01-06→2022-01-03  val 2022-01-04→2023-01-03  test 2023-08-07→2025-08-05
2026-05-02 08:57:56,482 INFO Loaded AUDUSD/15M split=train fold=latest: 46766 bars (2020-01-06 → 2022-01-03)
2026-05-02 08:57:56,676 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:56,695 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:56,708 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:56,715 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:56,748 INFO Loaded EURGBP/15M split=train fold=latest: 46759 bars (2020-01-06 → 2022-01-03)
2026-05-02 08:57:56,935 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:56,954 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:56,967 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:56,974 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:57,009 INFO Loaded EURJPY/15M split=train fold=latest: 46767 bars (2020-01-06 → 2022-01-03)
2026-05-02 08:57:57,199 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:57,217 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:57,231 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:57,237 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:57,271 INFO Loaded EURUSD/15M split=train fold=latest: 46766 bars (2020-01-06 → 2022-01-03)
2026-05-02 08:57:57,473 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:57,492 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:57,505 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:57,512 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:57,548 INFO Loaded GBPJPY/15M split=train fold=latest: 46765 bars (2020-01-06 → 2022-01-03)
2026-05-02 08:57:57,729 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:57,748 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:57,762 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:57,769 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:57,806 INFO Loaded GBPUSD/15M split=train fold=latest: 46764 bars (2020-01-06 → 2022-01-03)
2026-05-02 08:57:57,974 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:57,991 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:58,004 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:58,010 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-02 08:57:58,038 INFO Loaded NZDUSD/15M split=train fold=latest: 46766 bars (2020-01-06 → 2022-01-03)
2026-05-02 08:57:58,213 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:58,231 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:58,244 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:58,251 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:58,286 INFO Loaded USDCAD/15M split=train fold=latest: 46767 bars (2020-01-06 → 2022-01-03)
2026-05-02 08:57:58,460 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:58,478 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:58,492 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:58,499 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:58,534 INFO Loaded USDCHF/15M split=train fold=latest: 46763 bars (2020-01-06 → 2022-01-03)
2026-05-02 08:57:58,713 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:58,732 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:58,746 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:58,753 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 08:57:58,790 INFO Loaded USDJPY/15M split=train fold=latest: 46768 bars (2020-01-06 → 2022-01-03)
2026-05-02 08:57:59,092 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:59,118 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:59,133 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:59,143 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-02 08:57:59,205 INFO Loaded XAUUSD/15M split=train fold=latest: 47096 bars (2020-01-06 → 2022-01-03)
2026-05-02 08:57:59,309 INFO train_multi: 11 segments, ~448607 total bars
2026-05-02 08:57:59,557 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-05-02 08:57:59,557 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-05-02 08:57:59,557 INFO train_multi: training ALL 11 segments across TFs ['15M'] in one combined pass
2026-05-02 08:57:59,557 INFO train_multi: building combined dataset for TF=ALL (11 segments)
2026-05-02 08:58:03,841 WARNING train_multi: segment XAUUSD/ALL failed: _build_sequence_df: HTF ATR warmup produced non-finite values
2026-05-02 08:58:03,841 INFO train_multi TF=ALL: 407626 sequences across 10 segments
2026-05-02 08:58:03,841 INFO train_multi TF=ALL: estimated peak RAM = 5772 MB (train=326096 val=81530 n_feat=59 seq_len=30)
2026-05-02 08:58:04,632 INFO train_multi TF=ALL: train=326096 val=81530 (2891 MB tensors)
2026-05-02 08:58:08,477 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=18, min_epochs=22)
2026-05-02 08:58:19,733 INFO train_multi TF=ALL epoch 1/50 train=0.8499 val=0.8457
2026-05-02 08:58:19,744 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 08:58:19,744 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 08:58:19,745 INFO train_multi TF=ALL: new best val=0.8457 — saved
2026-05-02 08:58:28,534 INFO train_multi TF=ALL epoch 2/50 train=0.8408 val=0.8328
2026-05-02 08:58:28,539 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 08:58:28,539 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 08:58:28,539 INFO train_multi TF=ALL: new best val=0.8328 — saved
2026-05-02 08:58:37,305 INFO train_multi TF=ALL epoch 3/50 train=0.8175 val=0.7808
2026-05-02 08:58:37,310 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 08:58:37,310 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 08:58:37,310 INFO train_multi TF=ALL: new best val=0.7808 — saved
2026-05-02 08:58:46,034 INFO train_multi TF=ALL epoch 4/50 train=0.7124 val=0.6871
2026-05-02 08:58:46,039 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 08:58:46,039 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 08:58:46,039 INFO train_multi TF=ALL: new best val=0.6871 — saved
2026-05-02 08:58:54,900 INFO train_multi TF=ALL epoch 5/50 train=0.6904 val=0.6870
2026-05-02 08:58:54,905 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 08:58:54,905 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 08:58:54,905 INFO train_multi TF=ALL: new best val=0.6870 — saved
2026-05-02 08:59:03,705 INFO train_multi TF=ALL epoch 6/50 train=0.6894 val=0.6870
2026-05-02 08:59:03,710 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 08:59:03,710 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 08:59:03,710 INFO train_multi TF=ALL: new best val=0.6870 — saved
2026-05-02 08:59:12,374 INFO train_multi TF=ALL epoch 7/50 train=0.6885 val=0.6870
2026-05-02 08:59:12,379 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 08:59:12,379 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 08:59:12,379 INFO train_multi TF=ALL: new best val=0.6870 — saved
2026-05-02 08:59:21,155 INFO train_multi TF=ALL epoch 8/50 train=0.6884 val=0.6870
2026-05-02 08:59:21,159 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 08:59:21,159 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 08:59:21,159 INFO train_multi TF=ALL: new best val=0.6870 — saved
2026-05-02 08:59:29,859 INFO train_multi TF=ALL epoch 9/50 train=0.6882 val=0.6870
2026-05-02 08:59:38,482 INFO train_multi TF=ALL epoch 10/50 train=0.6881 val=0.6870
2026-05-02 08:59:38,487 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 08:59:38,488 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 08:59:38,488 INFO train_multi TF=ALL: new best val=0.6870 — saved
2026-05-02 08:59:47,189 INFO train_multi TF=ALL epoch 11/50 train=0.6879 val=0.6870
2026-05-02 08:59:47,194 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 08:59:47,194 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 08:59:47,194 INFO train_multi TF=ALL: new best val=0.6870 — saved
2026-05-02 08:59:56,115 INFO train_multi TF=ALL epoch 12/50 train=0.6877 val=0.6869
2026-05-02 08:59:56,120 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 08:59:56,120 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 08:59:56,120 INFO train_multi TF=ALL: new best val=0.6869 — saved
2026-05-02 09:00:04,952 INFO train_multi TF=ALL epoch 13/50 train=0.6874 val=0.6867
2026-05-02 09:00:04,957 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:00:04,957 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:00:04,957 INFO train_multi TF=ALL: new best val=0.6867 — saved
2026-05-02 09:00:13,664 INFO train_multi TF=ALL epoch 14/50 train=0.6856 val=0.6860
2026-05-02 09:00:13,669 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:00:13,669 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:00:13,669 INFO train_multi TF=ALL: new best val=0.6860 — saved
2026-05-02 09:00:22,395 INFO train_multi TF=ALL epoch 15/50 train=0.6832 val=0.6832
2026-05-02 09:00:22,400 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:00:22,400 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:00:22,400 INFO train_multi TF=ALL: new best val=0.6832 — saved
2026-05-02 09:00:31,031 INFO train_multi TF=ALL epoch 16/50 train=0.6816 val=0.6802
2026-05-02 09:00:31,036 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:00:31,036 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:00:31,036 INFO train_multi TF=ALL: new best val=0.6802 — saved
2026-05-02 09:00:39,840 INFO train_multi TF=ALL epoch 17/50 train=0.6802 val=0.6775
2026-05-02 09:00:39,846 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:00:39,846 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:00:39,846 INFO train_multi TF=ALL: new best val=0.6775 — saved
2026-05-02 09:00:48,575 INFO train_multi TF=ALL epoch 18/50 train=0.6787 val=0.6760
2026-05-02 09:00:48,580 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:00:48,580 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:00:48,580 INFO train_multi TF=ALL: new best val=0.6760 — saved
2026-05-02 09:00:57,397 INFO train_multi TF=ALL epoch 19/50 train=0.6768 val=0.6719
2026-05-02 09:00:57,401 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:00:57,402 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:00:57,402 INFO train_multi TF=ALL: new best val=0.6719 — saved
2026-05-02 09:01:06,173 INFO train_multi TF=ALL epoch 20/50 train=0.6718 val=0.6698
2026-05-02 09:01:06,178 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:01:06,178 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:01:06,178 INFO train_multi TF=ALL: new best val=0.6698 — saved
2026-05-02 09:01:14,804 INFO train_multi TF=ALL epoch 21/50 train=0.6676 val=0.6627
2026-05-02 09:01:14,809 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:01:14,809 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:01:14,809 INFO train_multi TF=ALL: new best val=0.6627 — saved
2026-05-02 09:01:23,593 INFO train_multi TF=ALL epoch 22/50 train=0.6608 val=0.6566
2026-05-02 09:01:23,598 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:01:23,598 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:01:23,598 INFO train_multi TF=ALL: new best val=0.6566 — saved
2026-05-02 09:01:32,577 INFO train_multi TF=ALL epoch 23/50 train=0.6499 val=0.6340
2026-05-02 09:01:32,582 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:01:32,582 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:01:32,582 INFO train_multi TF=ALL: new best val=0.6340 — saved
2026-05-02 09:01:41,450 INFO train_multi TF=ALL epoch 24/50 train=0.6313 val=0.6102
2026-05-02 09:01:41,455 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:01:41,455 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:01:41,455 INFO train_multi TF=ALL: new best val=0.6102 — saved
2026-05-02 09:01:50,163 INFO train_multi TF=ALL epoch 25/50 train=0.6119 val=0.5960
2026-05-02 09:01:50,168 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:01:50,168 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:01:50,168 INFO train_multi TF=ALL: new best val=0.5960 — saved
2026-05-02 09:01:59,152 INFO train_multi TF=ALL epoch 26/50 train=0.6008 val=0.5894
2026-05-02 09:01:59,157 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:01:59,157 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:01:59,157 INFO train_multi TF=ALL: new best val=0.5894 — saved
2026-05-02 09:02:08,087 INFO train_multi TF=ALL epoch 27/50 train=0.5922 val=0.5807
2026-05-02 09:02:08,092 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:02:08,092 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:02:08,092 INFO train_multi TF=ALL: new best val=0.5807 — saved
2026-05-02 09:02:17,050 INFO train_multi TF=ALL epoch 28/50 train=0.5858 val=0.5782
2026-05-02 09:02:17,055 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:02:17,055 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:02:17,055 INFO train_multi TF=ALL: new best val=0.5782 — saved
2026-05-02 09:02:26,108 INFO train_multi TF=ALL epoch 29/50 train=0.5808 val=0.5749
2026-05-02 09:02:26,113 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:02:26,113 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:02:26,113 INFO train_multi TF=ALL: new best val=0.5749 — saved
2026-05-02 09:02:35,024 INFO train_multi TF=ALL epoch 30/50 train=0.5762 val=0.5698
2026-05-02 09:02:35,029 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:02:35,029 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:02:35,029 INFO train_multi TF=ALL: new best val=0.5698 — saved
2026-05-02 09:02:43,852 INFO train_multi TF=ALL epoch 31/50 train=0.5727 val=0.5660
2026-05-02 09:02:43,857 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:02:43,857 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:02:43,857 INFO train_multi TF=ALL: new best val=0.5660 — saved
2026-05-02 09:02:52,657 INFO train_multi TF=ALL epoch 32/50 train=0.5693 val=0.5665
2026-05-02 09:03:01,333 INFO train_multi TF=ALL epoch 33/50 train=0.5672 val=0.5635
2026-05-02 09:03:01,338 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:03:01,338 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:03:01,338 INFO train_multi TF=ALL: new best val=0.5635 — saved
2026-05-02 09:03:10,234 INFO train_multi TF=ALL epoch 34/50 train=0.5640 val=0.5663
2026-05-02 09:03:19,000 INFO train_multi TF=ALL epoch 35/50 train=0.5615 val=0.5634
2026-05-02 09:03:19,008 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:03:19,008 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:03:19,008 INFO train_multi TF=ALL: new best val=0.5634 — saved
2026-05-02 09:03:27,888 INFO train_multi TF=ALL epoch 36/50 train=0.5600 val=0.5619
2026-05-02 09:03:27,893 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:03:27,893 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:03:27,893 INFO train_multi TF=ALL: new best val=0.5619 — saved
2026-05-02 09:03:36,788 INFO train_multi TF=ALL epoch 37/50 train=0.5578 val=0.5611
2026-05-02 09:03:36,793 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:03:36,793 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:03:36,794 INFO train_multi TF=ALL: new best val=0.5611 — saved
2026-05-02 09:03:45,585 INFO train_multi TF=ALL epoch 38/50 train=0.5557 val=0.5590
2026-05-02 09:03:45,591 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:03:45,591 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:03:45,591 INFO train_multi TF=ALL: new best val=0.5590 — saved
2026-05-02 09:03:54,492 INFO train_multi TF=ALL epoch 39/50 train=0.5534 val=0.5580
2026-05-02 09:03:54,497 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:03:54,497 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:03:54,497 INFO train_multi TF=ALL: new best val=0.5580 — saved
2026-05-02 09:04:03,220 INFO train_multi TF=ALL epoch 40/50 train=0.5524 val=0.5575
2026-05-02 09:04:03,225 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:04:03,225 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:04:03,225 INFO train_multi TF=ALL: new best val=0.5575 — saved
2026-05-02 09:04:11,982 INFO train_multi TF=ALL epoch 41/50 train=0.5498 val=0.5569
2026-05-02 09:04:11,986 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:04:11,987 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:04:11,987 INFO train_multi TF=ALL: new best val=0.5569 — saved
2026-05-02 09:04:20,775 INFO train_multi TF=ALL epoch 42/50 train=0.5491 val=0.5568
2026-05-02 09:04:20,781 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:04:20,781 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:04:20,781 INFO train_multi TF=ALL: new best val=0.5568 — saved
2026-05-02 09:04:30,064 INFO train_multi TF=ALL epoch 43/50 train=0.5477 val=0.5546
2026-05-02 09:04:30,069 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:04:30,070 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:04:30,070 INFO train_multi TF=ALL: new best val=0.5546 — saved
2026-05-02 09:04:39,907 INFO train_multi TF=ALL epoch 44/50 train=0.5454 val=0.5546
2026-05-02 09:04:49,714 INFO train_multi TF=ALL epoch 45/50 train=0.5438 val=0.5554
2026-05-02 09:04:58,887 INFO train_multi TF=ALL epoch 46/50 train=0.5425 val=0.5540
2026-05-02 09:04:58,892 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:04:58,892 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:04:58,892 INFO train_multi TF=ALL: new best val=0.5540 — saved
2026-05-02 09:05:07,601 INFO train_multi TF=ALL epoch 47/50 train=0.5408 val=0.5533
2026-05-02 09:05:07,606 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-02 09:05:07,606 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:05:07,606 INFO train_multi TF=ALL: new best val=0.5533 — saved
2026-05-02 09:05:16,389 INFO train_multi TF=ALL epoch 48/50 train=0.5392 val=0.5590
2026-05-02 09:05:25,094 INFO train_multi TF=ALL epoch 49/50 train=0.5376 val=0.5536
2026-05-02 09:05:33,932 INFO train_multi TF=ALL epoch 50/50 train=0.5363 val=0.5561
2026-05-02 09:05:34,091 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-02 09:05:34,092 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-02 09:05:34,092 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-02 09:05:34,092 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-05-02 09:05:34,092 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-05-02 09:05:34,092 INFO Retrain complete. Total wall-clock: 458.2s
2026-05-02 09:05:35,947 INFO Model gru: SUCCESS
2026-05-02 09:05:35,948 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 09:05:35,948 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-02 09:05:35,948 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-02 09:05:35,948 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-05-02 09:05:35,948 INFO   [DEFERRED] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip (expected after Round 1)
2026-05-02 09:05:35,948 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-05-02 09:05:35,948 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer', 'rl_ppo']
2026-05-02 09:05:35,950 INFO Saved 23 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: FAILED: exit 1
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Clean Quality/RL source: Backtest on train window ===
  START Train-window backtest for Quality/RL labels
2026-05-02 09:05:36,678 INFO === STEP 6: BACKTEST (train) ===
2026-05-02 09:05:36,679 INFO BT_WINDOW=train — train-window backtest: 2020-01-06 → 2022-01-03 (clean Quality/RL labels)
2026-05-02 09:05:36,680 INFO Cleared existing journal for fresh train run
2026-05-02 09:05:36,680 INFO ================================================================
  ROUND 0 / 3
================================================================
2026-05-02 09:05:36,680 INFO Round 0 — running backtest: 2020-01-06 → 2022-01-03 (ml_trader, shared ML cache)
2026-05-02 09:05:38,996 ERROR QualityScorer load failed: QualityScorer unavailable; train quality_scorer.pkl before backtesting with ML
Traceback (most recent call last):
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 3641, in <module>
    main()
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 3410, in main
    raise RuntimeError("QualityScorer unavailable; train quality_scorer.pkl before backtesting with ML")
RuntimeError: QualityScorer unavailable; train quality_scorer.pkl before backtesting with ML
2026-05-02 09:05:39,502 ERROR Backtest failed (rc=1) — check trading-engine/logs/backtest_*.log
2026-05-02 09:05:39,503 ERROR Round 0 backtest failed: backtest exited 1
