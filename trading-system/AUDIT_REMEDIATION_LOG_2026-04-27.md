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
2026-05-02 15:49:46,777 INFO Loading feature-engineered data...
2026-05-02 15:49:47,438 INFO Loaded 221743 rows, 202 features
2026-05-02 15:49:47,439 INFO Data span: 2016-01-04 -> 2025-08-05  (9.6 years)
2026-05-02 15:49:47,442 INFO Fold 000 train 2016-01-04 -> 2018-01-03 (47088 bars), val 2018-01-04 -> 2019-01-03 (23448 bars)
2026-05-02 15:49:47,442 INFO Fold 001 train 2018-01-04 -> 2020-01-03 (46825 bars), val 2020-01-06 -> 2020-12-31 (23259 bars)
2026-05-02 15:49:47,443 INFO Fold 002 train 2020-01-06 -> 2022-01-03 (46766 bars), val 2022-01-04 -> 2023-01-03 (23588 bars)
2026-05-02 15:49:47,443 INFO No leakage confirmed: every fold ends before final 2-year blind test

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
2026-05-02 15:49:50,898 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-05-02 15:49:50,898 INFO --- Training regime ---
2026-05-02 15:49:50,898 INFO Running retrain --model regime
2026-05-02 15:49:51,079 INFO retrain environment: KAGGLE
2026-05-02 15:49:52,704 INFO Device: CUDA (2 GPU(s))
2026-05-02 15:49:52,715 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-02 15:49:52,715 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-02 15:49:52,715 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-02 15:49:52,719 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-02 15:49:52,719 INFO Retrain data split: train
2026-05-02 15:49:52,719 INFO Retrain rolling fold selector: latest
2026-05-02 15:49:52,720 INFO === RegimeClassifier retrain (hierarchical: HTF 2-score bias + LTF 5-score behaviour) ===
2026-05-02 15:49:52,892 INFO NumExpr defaulting to 4 threads.
2026-05-02 15:49:53,126 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-02 15:49:53,126 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-02 15:49:53,126 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-02 15:49:53,126 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-02 15:49:53,208 INFO Regime rolling folds selected: ['fold_000', 'fold_001', 'fold_002']
2026-05-02 15:49:53,208 INFO === Regime rolling fold 1/3: fold_000 ===
2026-05-02 15:49:53,208 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-02 15:49:53,247 INFO Split boundaries loaded fold=fold_000/3 — train 2016-01-04→2018-01-03  val 2018-01-04→2019-01-03  test 2023-08-07→2025-08-05
2026-05-02 15:49:53,248 INFO Loaded AUDUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:49:53,263 INFO Loaded EURGBP/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:49:53,280 INFO Loaded EURJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:49:53,295 INFO Loaded EURUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:49:53,311 INFO Loaded GBPJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:49:53,326 INFO Loaded GBPUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:49:53,341 INFO Loaded NZDUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:49:53,355 INFO Loaded USDCAD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:49:53,370 INFO Loaded USDCHF/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:49:53,384 INFO Loaded USDJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:49:53,402 INFO Loaded XAUUSD/4H split=train fold=fold_000: 3204 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:49:53,529 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:53,573 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:53,591 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:53,591 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:53,599 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:53,600 INFO Loaded AUDUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:49:53,795 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 92, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 2876}  ambiguous=1700 (total=3023) horizon=12
2026-05-02 15:49:53,798 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected AUDUSD — 2973 samples (group=dollar) score_means={'bias_up_score': 0.0309, 'bias_down_score': 0.0185} labels={'BIAS_UP': 92, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 2826} clean={'BIAS_UP': 92, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 1152}
2026-05-02 15:49:53,966 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:53,999 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:54,017 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:54,017 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:54,024 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:54,025 INFO Loaded EURGBP/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:49:54,206 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 156, 'BIAS_DOWN': 71, 'BIAS_NEUTRAL': 2796}  ambiguous=1710 (total=3023) horizon=12
2026-05-02 15:49:54,209 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected EURGBP — 2973 samples (group=cross) score_means={'bias_up_score': 0.0525, 'bias_down_score': 0.0239} labels={'BIAS_UP': 156, 'BIAS_DOWN': 71, 'BIAS_NEUTRAL': 2746} clean={'BIAS_UP': 156, 'BIAS_DOWN': 71, 'BIAS_NEUTRAL': 1071}
2026-05-02 15:49:54,376 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:54,416 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:54,435 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:54,435 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:54,442 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:54,443 INFO Loaded EURJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:49:54,630 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 95, 'BIAS_DOWN': 63, 'BIAS_NEUTRAL': 2865}  ambiguous=1742 (total=3023) horizon=12
2026-05-02 15:49:54,634 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected EURJPY — 2973 samples (group=cross) score_means={'bias_up_score': 0.032, 'bias_down_score': 0.0212} labels={'BIAS_UP': 95, 'BIAS_DOWN': 63, 'BIAS_NEUTRAL': 2815} clean={'BIAS_UP': 95, 'BIAS_DOWN': 63, 'BIAS_NEUTRAL': 1099}
2026-05-02 15:49:54,787 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:54,823 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:54,841 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:54,841 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:54,849 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:54,849 INFO Loaded EURUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:49:55,038 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 98, 'BIAS_DOWN': 57, 'BIAS_NEUTRAL': 2868}  ambiguous=1742 (total=3023) horizon=12
2026-05-02 15:49:55,041 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected EURUSD — 2973 samples (group=dollar) score_means={'bias_up_score': 0.033, 'bias_down_score': 0.0192} labels={'BIAS_UP': 98, 'BIAS_DOWN': 57, 'BIAS_NEUTRAL': 2818} clean={'BIAS_UP': 98, 'BIAS_DOWN': 57, 'BIAS_NEUTRAL': 1105}
2026-05-02 15:49:55,190 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:55,225 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:55,244 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:55,244 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:55,251 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:55,252 INFO Loaded GBPJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:49:55,438 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 164, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2758}  ambiguous=1723 (total=3023) horizon=12
2026-05-02 15:49:55,441 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected GBPJPY — 2973 samples (group=cross) score_means={'bias_up_score': 0.0552, 'bias_down_score': 0.034} labels={'BIAS_UP': 164, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2708} clean={'BIAS_UP': 164, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 1019}
2026-05-02 15:49:55,584 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:55,617 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:55,635 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:55,635 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:55,643 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:55,644 INFO Loaded GBPUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:49:55,826 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 79, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2843}  ambiguous=1759 (total=3023) horizon=12
2026-05-02 15:49:55,829 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected GBPUSD — 2973 samples (group=dollar) score_means={'bias_up_score': 0.0266, 'bias_down_score': 0.034} labels={'BIAS_UP': 79, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2793} clean={'BIAS_UP': 79, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 1073}
2026-05-02 15:49:55,962 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:49:55,991 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:49:56,007 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:49:56,007 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:49:56,013 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:49:56,014 INFO Loaded NZDUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:49:56,200 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 54, 'BIAS_DOWN': 54, 'BIAS_NEUTRAL': 2915}  ambiguous=1779 (total=3023) horizon=12
2026-05-02 15:49:56,203 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected NZDUSD — 2973 samples (group=dollar) score_means={'bias_up_score': 0.0182, 'bias_down_score': 0.0182} labels={'BIAS_UP': 54, 'BIAS_DOWN': 54, 'BIAS_NEUTRAL': 2865} clean={'BIAS_UP': 54, 'BIAS_DOWN': 54, 'BIAS_NEUTRAL': 1117}
2026-05-02 15:49:56,357 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:56,393 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:56,414 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:56,414 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:56,422 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:56,423 INFO Loaded USDCAD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:49:56,626 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 89, 'BIAS_DOWN': 133, 'BIAS_NEUTRAL': 2801}  ambiguous=1770 (total=3023) horizon=12
2026-05-02 15:49:56,629 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected USDCAD — 2973 samples (group=dollar) score_means={'bias_up_score': 0.0299, 'bias_down_score': 0.0447} labels={'BIAS_UP': 89, 'BIAS_DOWN': 133, 'BIAS_NEUTRAL': 2751} clean={'BIAS_UP': 89, 'BIAS_DOWN': 133, 'BIAS_NEUTRAL': 1016}
2026-05-02 15:49:56,792 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:56,828 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:56,847 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:56,847 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:56,855 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:56,856 INFO Loaded USDCHF/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:49:57,071 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 64, 'BIAS_DOWN': 52, 'BIAS_NEUTRAL': 2907}  ambiguous=1741 (total=3023) horizon=12
2026-05-02 15:49:57,074 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected USDCHF — 2973 samples (group=dollar) score_means={'bias_up_score': 0.0215, 'bias_down_score': 0.0175} labels={'BIAS_UP': 64, 'BIAS_DOWN': 52, 'BIAS_NEUTRAL': 2857} clean={'BIAS_UP': 64, 'BIAS_DOWN': 52, 'BIAS_NEUTRAL': 1148}
2026-05-02 15:49:57,245 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:57,295 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:57,323 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:57,324 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:57,336 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:57,337 INFO Loaded USDJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:49:57,543 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 102, 'BIAS_DOWN': 78, 'BIAS_NEUTRAL': 2843}  ambiguous=1762 (total=3023) horizon=12
2026-05-02 15:49:57,546 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected USDJPY — 2973 samples (group=dollar) score_means={'bias_up_score': 0.0343, 'bias_down_score': 0.0262} labels={'BIAS_UP': 102, 'BIAS_DOWN': 78, 'BIAS_NEUTRAL': 2793} clean={'BIAS_UP': 102, 'BIAS_DOWN': 78, 'BIAS_NEUTRAL': 1058}
2026-05-02 15:49:57,809 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:49:57,872 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:49:57,896 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:49:57,896 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:49:57,906 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:49:57,907 INFO Loaded XAUUSD/4H split=train fold=fold_000: 3204 bars (2016-01-04 → 2018-01-03)
2026-05-02 15:49:58,110 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 2845}  ambiguous=1801 (total=3204) horizon=12
2026-05-02 15:49:58,114 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected XAUUSD — 3154 samples (group=gold) score_means={'bias_up_score': 0.0672, 'bias_down_score': 0.0466} labels={'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 2795} clean={'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 1029}
2026-05-02 15:49:58,174 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 415, 'BIAS_DOWN': 235, 'BIAS_NEUTRAL': 8269}, 'dollar': {'BIAS_UP': 578, 'BIAS_DOWN': 530, 'BIAS_NEUTRAL': 19703}, 'gold': {'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 2795}}
2026-05-02 15:49:58,174 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0465, 'bias_down_score': 0.0263}, 'dollar': {'bias_up_score': 0.0278, 'bias_down_score': 0.0255}, 'gold': {'bias_up_score': 0.0672, 'bias_down_score': 0.0466}}
2026-05-02 15:49:58,174 INFO Regime[4H mode=htf_bias] label distribution by year: {2016: {'BIAS_UP': 485, 'BIAS_DOWN': 511, 'BIAS_NEUTRAL': 15101}, 2017: {'BIAS_UP': 717, 'BIAS_DOWN': 401, 'BIAS_NEUTRAL': 15515}, 2018: {'BIAS_UP': 3, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 151}}
2026-05-02 15:49:58,174 INFO Regime[4H mode=htf_bias] score means by year: {2016: {'bias_up_score': 0.0301, 'bias_down_score': 0.0317}, 2017: {'bias_up_score': 0.0431, 'bias_down_score': 0.0241}, 2018: {'bias_up_score': 0.0195, 'bias_down_score': 0.0}}
2026-05-02 15:49:58,218 INFO Loaded AUDUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:49:58,220 INFO Loaded EURGBP/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:49:58,221 INFO Loaded EURJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:49:58,222 INFO Loaded EURUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:49:58,223 INFO Loaded GBPJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:49:58,224 INFO Loaded GBPUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:49:58,225 INFO Loaded NZDUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:49:58,226 INFO Loaded USDCAD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:49:58,228 INFO Loaded USDCHF/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:49:58,229 INFO Loaded USDJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:49:58,230 INFO Loaded XAUUSD/4H split=val fold=fold_000: 1600 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:49:58,237 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:58,240 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:58,241 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:58,241 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:58,241 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:58,243 INFO Loaded AUDUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:49:58,472 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 5, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1448}  ambiguous=896 (total=1506) horizon=12
2026-05-02 15:49:58,475 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected AUDUSD — 1456 samples (group=dollar) score_means={'bias_up_score': 0.0034, 'bias_down_score': 0.0364} labels={'BIAS_UP': 5, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1398} clean={'BIAS_UP': 5, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 531}
2026-05-02 15:49:58,541 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:58,543 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:58,544 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:58,544 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:58,544 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:58,545 INFO Loaded EURGBP/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:49:58,707 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 12, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1453}  ambiguous=868 (total=1506) horizon=12
2026-05-02 15:49:58,710 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected EURGBP — 1456 samples (group=cross) score_means={'bias_up_score': 0.0082, 'bias_down_score': 0.0282} labels={'BIAS_UP': 12, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1403} clean={'BIAS_UP': 12, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 575}
2026-05-02 15:49:58,773 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:58,775 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:58,776 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:58,776 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:58,776 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:58,777 INFO Loaded EURJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:49:58,945 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 20, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1445}  ambiguous=874 (total=1506) horizon=12
2026-05-02 15:49:58,948 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected EURJPY — 1456 samples (group=cross) score_means={'bias_up_score': 0.0137, 'bias_down_score': 0.0282} labels={'BIAS_UP': 20, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1395} clean={'BIAS_UP': 20, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 555}
2026-05-02 15:49:59,012 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:59,015 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:59,015 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:59,016 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:59,016 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:59,017 INFO Loaded EURUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:49:59,184 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 24, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 1415}  ambiguous=876 (total=1506) horizon=12
2026-05-02 15:49:59,187 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected EURUSD — 1456 samples (group=dollar) score_means={'bias_up_score': 0.0165, 'bias_down_score': 0.046} labels={'BIAS_UP': 24, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 1365} clean={'BIAS_UP': 24, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 522}
2026-05-02 15:49:59,252 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:59,254 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:59,255 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:59,256 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:59,256 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:59,257 INFO Loaded GBPJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:49:59,429 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 27, 'BIAS_DOWN': 37, 'BIAS_NEUTRAL': 1442}  ambiguous=926 (total=1506) horizon=12
2026-05-02 15:49:59,431 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected GBPJPY — 1456 samples (group=cross) score_means={'bias_up_score': 0.0185, 'bias_down_score': 0.0254} labels={'BIAS_UP': 27, 'BIAS_DOWN': 37, 'BIAS_NEUTRAL': 1392} clean={'BIAS_UP': 27, 'BIAS_DOWN': 37, 'BIAS_NEUTRAL': 506}
2026-05-02 15:49:59,494 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:59,496 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:59,497 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:59,497 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:59,498 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:59,499 INFO Loaded GBPUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:49:59,662 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 50, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1371}  ambiguous=874 (total=1506) horizon=12
2026-05-02 15:49:59,664 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected GBPUSD — 1456 samples (group=dollar) score_means={'bias_up_score': 0.0343, 'bias_down_score': 0.0584} labels={'BIAS_UP': 50, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1321} clean={'BIAS_UP': 50, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 476}
2026-05-02 15:49:59,727 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:49:59,729 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:49:59,730 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:49:59,730 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:49:59,730 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-02 15:49:59,731 INFO Loaded NZDUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:49:59,898 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 23, 'BIAS_DOWN': 80, 'BIAS_NEUTRAL': 1403}  ambiguous=896 (total=1506) horizon=12
2026-05-02 15:49:59,901 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected NZDUSD — 1456 samples (group=dollar) score_means={'bias_up_score': 0.0158, 'bias_down_score': 0.0549} labels={'BIAS_UP': 23, 'BIAS_DOWN': 80, 'BIAS_NEUTRAL': 1353} clean={'BIAS_UP': 23, 'BIAS_DOWN': 80, 'BIAS_NEUTRAL': 482}
2026-05-02 15:49:59,964 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:59,967 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:59,967 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:59,968 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:59,968 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:49:59,969 INFO Loaded USDCAD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:50:00,137 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 48, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 1445}  ambiguous=907 (total=1506) horizon=12
2026-05-02 15:50:00,139 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected USDCAD — 1456 samples (group=dollar) score_means={'bias_up_score': 0.033, 'bias_down_score': 0.0089} labels={'BIAS_UP': 48, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 1395} clean={'BIAS_UP': 48, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 522}
2026-05-02 15:50:00,207 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:50:00,209 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:50:00,210 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:50:00,210 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:50:00,210 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:50:00,211 INFO Loaded USDCHF/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:50:00,373 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 68, 'BIAS_DOWN': 45, 'BIAS_NEUTRAL': 1393}  ambiguous=848 (total=1506) horizon=12
2026-05-02 15:50:00,376 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected USDCHF — 1456 samples (group=dollar) score_means={'bias_up_score': 0.0467, 'bias_down_score': 0.0309} labels={'BIAS_UP': 68, 'BIAS_DOWN': 45, 'BIAS_NEUTRAL': 1343} clean={'BIAS_UP': 68, 'BIAS_DOWN': 45, 'BIAS_NEUTRAL': 530}
2026-05-02 15:50:00,440 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:50:00,443 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:50:00,443 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:50:00,444 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:50:00,444 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 15:50:00,445 INFO Loaded USDJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:50:00,604 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 58, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1418}  ambiguous=888 (total=1506) horizon=12
2026-05-02 15:50:00,607 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected USDJPY — 1456 samples (group=dollar) score_means={'bias_up_score': 0.0398, 'bias_down_score': 0.0206} labels={'BIAS_UP': 58, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1368} clean={'BIAS_UP': 58, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 510}
2026-05-02 15:50:00,673 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:50:00,678 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:50:00,679 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:50:00,679 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:50:00,680 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-02 15:50:00,681 INFO Loaded XAUUSD/4H split=val fold=fold_000: 1600 bars (2018-01-04 → 2019-01-03)
2026-05-02 15:50:00,859 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 18, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 1547}  ambiguous=851 (total=1600) horizon=12
2026-05-02 15:50:00,861 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected XAUUSD — 1550 samples (group=gold) score_means={'bias_up_score': 0.0116, 'bias_down_score': 0.0226} labels={'BIAS_UP': 18, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 1497} clean={'BIAS_UP': 18, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 677}
2026-05-02 15:50:00,921 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 59, 'BIAS_DOWN': 119, 'BIAS_NEUTRAL': 4190}, 'dollar': {'BIAS_UP': 276, 'BIAS_DOWN': 373, 'BIAS_NEUTRAL': 9543}, 'gold': {'BIAS_UP': 18, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 1497}}
2026-05-02 15:50:00,921 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0135, 'bias_down_score': 0.0272}, 'dollar': {'bias_up_score': 0.0271, 'bias_down_score': 0.0366}, 'gold': {'bias_up_score': 0.0116, 'bias_down_score': 0.0226}}
2026-05-02 15:50:00,921 INFO Regime[4H mode=htf_bias] label distribution by year: {2018: {'BIAS_UP': 352, 'BIAS_DOWN': 521, 'BIAS_NEUTRAL': 15083}, 2019: {'BIAS_UP': 1, 'BIAS_DOWN': 6, 'BIAS_NEUTRAL': 147}}
2026-05-02 15:50:00,921 INFO Regime[4H mode=htf_bias] score means by year: {2018: {'bias_up_score': 0.0221, 'bias_down_score': 0.0327}, 2019: {'bias_up_score': 0.0065, 'bias_down_score': 0.039}}
2026-05-02 15:50:00,964 INFO Regime phase HTF dataset build fold=fold_000: 7.8s (train=32884 val=16110)
2026-05-02 15:50:00,968 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=32884 val=16110 train_labels={'BIAS_UP': 1205, 'BIAS_DOWN': 912, 'BIAS_NEUTRAL': 30767} val_labels={'BIAS_UP': 353, 'BIAS_DOWN': 527, 'BIAS_NEUTRAL': 15230}
2026-05-02 15:50:01,246 INFO RegimeClassifier[mode=htf_bias]: cold start HTF score head
2026-05-02 15:50:01,247 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-02 15:50:01,247 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 20.0, 'bias_down_score': 20.0}
2026-05-02 15:50:06,562 INFO Regime HTF score epoch  1/50 — tr=2.1329 va=1.0608 acc=0.945 bal=0.333 threshold=0.35 margin=0.15 recall={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 1.0} precision={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 0.945}
2026-05-02 15:50:07,553 INFO Regime HTF score epoch  2/50 — tr=2.0278 va=1.0502 bal=0.333
2026-05-02 15:50:08,505 INFO Regime HTF score epoch  3/50 — tr=1.9650 va=1.0190 bal=0.333
2026-05-02 15:50:09,461 INFO Regime HTF score epoch  4/50 — tr=1.7854 va=0.9616 bal=0.349
2026-05-02 15:50:10,437 INFO Regime HTF score epoch  5/50 — tr=1.5397 va=0.8831 acc=0.858 bal=0.673 threshold=0.65 margin=0.00 recall={'BIAS_UP': 0.439, 'BIAS_DOWN': 0.708, 'BIAS_NEUTRAL': 0.873} precision={'BIAS_UP': 0.284, 'BIAS_DOWN': 0.194, 'BIAS_NEUTRAL': 0.974}
2026-05-02 15:50:11,404 INFO Regime HTF score epoch  6/50 — tr=1.3134 va=0.8051 bal=0.719
2026-05-02 15:50:12,363 INFO Regime HTF score epoch  7/50 — tr=1.0863 va=0.7737 bal=0.742
2026-05-02 15:50:13,321 INFO Regime HTF score epoch  8/50 — tr=0.9362 va=0.7617 bal=0.843
2026-05-02 15:50:14,308 INFO Regime HTF score epoch  9/50 — tr=0.8478 va=0.7418 bal=0.865
2026-05-02 15:50:15,286 INFO Regime HTF score epoch 10/50 — tr=0.7850 va=0.7301 acc=0.808 bal=0.881 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.898, 'BIAS_DOWN': 0.945, 'BIAS_NEUTRAL': 0.801} precision={'BIAS_UP': 0.22, 'BIAS_DOWN': 0.207, 'BIAS_NEUTRAL': 0.995}
2026-05-02 15:50:16,265 INFO Regime HTF score epoch 11/50 — tr=0.7477 va=0.7122 bal=0.889
2026-05-02 15:50:17,265 INFO Regime HTF score epoch 12/50 — tr=0.7166 va=0.7010 bal=0.892
2026-05-02 15:50:18,265 INFO Regime HTF score epoch 13/50 — tr=0.6938 va=0.6809 bal=0.896
2026-05-02 15:50:19,246 INFO Regime HTF score epoch 14/50 — tr=0.6738 va=0.6663 bal=0.897
2026-05-02 15:50:20,239 INFO Regime HTF score epoch 15/50 — tr=0.6522 va=0.6524 acc=0.814 bal=0.897 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.946, 'BIAS_DOWN': 0.939, 'BIAS_NEUTRAL': 0.806} precision={'BIAS_UP': 0.208, 'BIAS_DOWN': 0.228, 'BIAS_NEUTRAL': 0.996}
2026-05-02 15:50:21,210 INFO Regime HTF score epoch 16/50 — tr=0.6293 va=0.6448 bal=0.904
2026-05-02 15:50:22,173 INFO Regime HTF score epoch 17/50 — tr=0.6168 va=0.6376 bal=0.908
2026-05-02 15:50:23,137 INFO Regime HTF score epoch 18/50 — tr=0.6017 va=0.6225 bal=0.906
2026-05-02 15:50:24,095 INFO Regime HTF score epoch 19/50 — tr=0.5908 va=0.6144 bal=0.908
2026-05-02 15:50:25,056 INFO Regime HTF score epoch 20/50 — tr=0.5790 va=0.6031 acc=0.817 bal=0.909 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.949, 'BIAS_DOWN': 0.97, 'BIAS_NEUTRAL': 0.808} precision={'BIAS_UP': 0.219, 'BIAS_DOWN': 0.229, 'BIAS_NEUTRAL': 0.997}
2026-05-02 15:50:26,011 INFO Regime HTF score epoch 21/50 — tr=0.5686 va=0.5888 bal=0.910
2026-05-02 15:50:26,961 INFO Regime HTF score epoch 22/50 — tr=0.5650 va=0.5861 bal=0.911
2026-05-02 15:50:27,966 INFO Regime HTF score epoch 23/50 — tr=0.5553 va=0.5734 bal=0.909
2026-05-02 15:50:28,949 INFO Regime HTF score epoch 24/50 — tr=0.5511 va=0.5620 bal=0.911
2026-05-02 15:50:29,906 INFO Regime HTF score epoch 25/50 — tr=0.5358 va=0.5548 acc=0.828 bal=0.912 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.943, 'BIAS_DOWN': 0.973, 'BIAS_NEUTRAL': 0.82} precision={'BIAS_UP': 0.231, 'BIAS_DOWN': 0.239, 'BIAS_NEUTRAL': 0.997}
2026-05-02 15:50:30,876 INFO Regime HTF score epoch 26/50 — tr=0.5268 va=0.5473 bal=0.914
2026-05-02 15:50:31,836 INFO Regime HTF score epoch 27/50 — tr=0.5281 va=0.5501 bal=0.917
2026-05-02 15:50:32,804 INFO Regime HTF score epoch 28/50 — tr=0.5145 va=0.5385 bal=0.912
2026-05-02 15:50:33,754 INFO Regime HTF score epoch 29/50 — tr=0.5185 va=0.5332 bal=0.914
2026-05-02 15:50:34,714 INFO Regime HTF score epoch 30/50 — tr=0.5130 va=0.5260 acc=0.833 bal=0.913 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.958, 'BIAS_DOWN': 0.956, 'BIAS_NEUTRAL': 0.826} precision={'BIAS_UP': 0.226, 'BIAS_DOWN': 0.251, 'BIAS_NEUTRAL': 0.997}
2026-05-02 15:50:35,657 INFO Regime HTF score epoch 31/50 — tr=0.5063 va=0.5241 bal=0.914
2026-05-02 15:50:36,616 INFO Regime HTF score epoch 32/50 — tr=0.5093 va=0.5283 bal=0.917
2026-05-02 15:50:37,605 INFO Regime HTF score epoch 33/50 — tr=0.4990 va=0.5280 bal=0.918
2026-05-02 15:50:38,564 INFO Regime HTF score epoch 34/50 — tr=0.4881 va=0.5258 bal=0.918
2026-05-02 15:50:39,523 INFO Regime HTF score epoch 35/50 — tr=0.4968 va=0.5226 acc=0.828 bal=0.917 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.955, 'BIAS_DOWN': 0.975, 'BIAS_NEUTRAL': 0.82} precision={'BIAS_UP': 0.228, 'BIAS_DOWN': 0.244, 'BIAS_NEUTRAL': 0.998}
2026-05-02 15:50:40,469 INFO Regime HTF score epoch 36/50 — tr=0.4905 va=0.5240 bal=0.918
2026-05-02 15:50:41,423 INFO Regime HTF score epoch 37/50 — tr=0.4867 va=0.5208 bal=0.918
2026-05-02 15:50:42,375 INFO Regime HTF score epoch 38/50 — tr=0.4895 va=0.5161 bal=0.919
2026-05-02 15:50:43,337 INFO Regime HTF score epoch 39/50 — tr=0.4854 va=0.5106 bal=0.918
2026-05-02 15:50:44,283 INFO Regime HTF score epoch 40/50 — tr=0.4928 va=0.5121 acc=0.832 bal=0.919 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.96, 'BIAS_DOWN': 0.972, 'BIAS_NEUTRAL': 0.824} precision={'BIAS_UP': 0.229, 'BIAS_DOWN': 0.25, 'BIAS_NEUTRAL': 0.998}
2026-05-02 15:50:45,240 INFO Regime HTF score epoch 41/50 — tr=0.4825 va=0.5094 bal=0.918
2026-05-02 15:50:46,202 INFO Regime HTF score epoch 42/50 — tr=0.4846 va=0.5096 bal=0.919
2026-05-02 15:50:47,164 INFO Regime HTF score epoch 43/50 — tr=0.4860 va=0.5033 bal=0.916
2026-05-02 15:50:48,163 INFO Regime HTF score epoch 44/50 — tr=0.4844 va=0.5045 bal=0.918
2026-05-02 15:50:49,118 INFO Regime HTF score epoch 45/50 — tr=0.4778 va=0.5080 acc=0.832 bal=0.918 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.963, 'BIAS_DOWN': 0.966, 'BIAS_NEUTRAL': 0.825} precision={'BIAS_UP': 0.228, 'BIAS_DOWN': 0.251, 'BIAS_NEUTRAL': 0.998}
2026-05-02 15:50:50,078 INFO Regime HTF score epoch 46/50 — tr=0.4852 va=0.5107 bal=0.920
2026-05-02 15:50:50,078 INFO Regime HTF score early stop at epoch 46
2026-05-02 15:50:50,958 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.850 margin=0.000 precision={'BIAS_UP': 0.237, 'BIAS_DOWN': 0.24, 'BIAS_NEUTRAL': 0.998} recall={'BIAS_UP': 0.952, 'BIAS_DOWN': 0.983, 'BIAS_NEUTRAL': 0.821} f1={'BIAS_UP': 0.379, 'BIAS_DOWN': 0.386, 'BIAS_NEUTRAL': 0.901} confusion=[[336, 0, 17], [0, 518, 9], [1083, 1640, 12507]] score_mae={'bias_up_score': 0.2128, 'bias_down_score': 0.2466} pred_share={'BIAS_UP': 0.0881, 'BIAS_DOWN': 0.134, 'BIAS_NEUTRAL': 0.778}
2026-05-02 15:50:50,960 INFO Regime phase HTF train fold=fold_000: 50.0s
Traceback (most recent call last):
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1745, in <module>
    main()
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1704, in main
    result = retrain_regime(dry)
             ^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1239, in retrain_regime
    raise RuntimeError(f"Regime HTF training failed fold={fold_key}: {res_4h['error']}")
RuntimeError: Regime HTF training failed fold=fold_000: Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.237, 'BIAS_DOWN': 0.24, 'BIAS_NEUTRAL': 0.998} min_precision=0.300 recall={'BIAS_UP': 0.952, 'BIAS_DOWN': 0.983, 'BIAS_NEUTRAL': 0.821} min_recall=0.100 f1={'BIAS_UP': 0.379, 'BIAS_DOWN': 0.386, 'BIAS_NEUTRAL': 0.901} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_UP', 'BIAS_DOWN'] weak_recall=[] weak_f1=[] weak_neutral=False. Refusing to save directional-bias score weights.

=== TRAINING COMPLETE ===
  regime: FAILED: exit 1
2026-05-02 15:50:53,328 ERROR retrain regime failed (exit 1)
2026-05-02 15:50:53,328 ERROR Model regime failed: exit 1
2026-05-02 15:50:53,329 WARNING   [MISSING] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 15:50:53,329 WARNING   [MISSING] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-02 15:50:53,329 WARNING   [MISSING] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-02 15:50:53,329 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-05-02 15:50:53,329 INFO   [DEFERRED] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip (expected after Round 1)
2026-05-02 15:50:53,329 WARNING Missing required weights: ['gru_lstm', 'regime_htf', 'regime_ltf'] — run retrain_incremental.py for each
2026-05-02 15:50:53,329 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer', 'rl_ppo']
2026-05-02 15:50:53,329 WARNING No retrain_history.jsonl found
2026-05-02 15:50:53,329 ERROR Step 7a failed; required training/artifacts missing: ['gru_lstm', 'regime', 'regime_htf', 'regime_ltf']
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
