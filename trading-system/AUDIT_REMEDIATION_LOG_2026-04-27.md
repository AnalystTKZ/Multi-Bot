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
2026-05-03 08:35:04,658 INFO Loading feature-engineered data...
2026-05-03 08:35:05,134 INFO Loaded 221743 rows, 202 features
2026-05-03 08:35:05,135 INFO Data span: 2016-01-04 -> 2025-08-05  (9.6 years)
2026-05-03 08:35:05,135 INFO Fold 000 train 2016-01-04 -> 2018-01-03 (47088 bars), val 2018-01-04 -> 2019-01-03 (23448 bars)
2026-05-03 08:35:05,136 INFO Fold 001 train 2018-01-04 -> 2020-01-03 (46825 bars), val 2020-01-06 -> 2020-12-31 (23259 bars)
2026-05-03 08:35:05,136 INFO Fold 002 train 2020-01-06 -> 2022-01-03 (46766 bars), val 2022-01-04 -> 2023-01-03 (23588 bars)
2026-05-03 08:35:05,136 INFO No leakage confirmed: every fold ends before final 2-year blind test

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
2026-05-03 08:35:08,516 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-05-03 08:35:08,516 INFO --- Training regime ---
2026-05-03 08:35:08,516 INFO Running retrain --model regime
2026-05-03 08:35:08,708 INFO retrain environment: KAGGLE
2026-05-03 08:35:10,444 INFO Device: CUDA (2 GPU(s))
2026-05-03 08:35:10,456 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-03 08:35:10,456 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-03 08:35:10,456 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-03 08:35:10,459 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-03 08:35:10,459 INFO Retrain data split: train
2026-05-03 08:35:10,459 INFO Retrain rolling fold selector: latest
2026-05-03 08:35:10,460 INFO === RegimeClassifier retrain (hierarchical: HTF 2-score bias + LTF 5-score behaviour) ===
2026-05-03 08:35:10,633 INFO NumExpr defaulting to 4 threads.
2026-05-03 08:35:10,852 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-03 08:35:10,852 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-03 08:35:10,852 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-03 08:35:10,852 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-03 08:35:10,907 INFO Regime rolling folds selected: ['fold_000', 'fold_001', 'fold_002']
2026-05-03 08:35:10,908 INFO === Regime rolling fold 1/3: fold_000 ===
2026-05-03 08:35:10,908 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-03 08:35:10,944 INFO Split boundaries loaded fold=fold_000/3 — train 2016-01-04→2018-01-03  val 2018-01-04→2019-01-03  test 2023-08-07→2025-08-05
2026-05-03 08:35:10,945 INFO Loaded AUDUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-03 08:35:10,960 INFO Loaded EURGBP/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-03 08:35:10,975 INFO Loaded EURJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-03 08:35:10,990 INFO Loaded EURUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-03 08:35:11,006 INFO Loaded GBPJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-03 08:35:11,021 INFO Loaded GBPUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-03 08:35:11,035 INFO Loaded NZDUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-03 08:35:11,049 INFO Loaded USDCAD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-03 08:35:11,065 INFO Loaded USDCHF/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-03 08:35:11,080 INFO Loaded USDJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-03 08:35:11,098 INFO Loaded XAUUSD/4H split=train fold=fold_000: 3204 bars (2016-01-04 → 2018-01-03)
2026-05-03 08:35:11,232 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:11,277 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:11,295 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:11,296 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:11,303 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:11,304 INFO Loaded AUDUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:11,520 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 92, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 2876}  ambiguous=1700 (total=3023) horizon=12
2026-05-03 08:35:11,523 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected AUDUSD — 2973 samples (group=dollar) score_means={'bias_up_score': 0.0309, 'bias_down_score': 0.0185} labels={'BIAS_UP': 92, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 2826} clean={'BIAS_UP': 92, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 1152}
2026-05-03 08:35:11,691 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:11,727 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:11,746 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:11,747 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:11,754 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:11,755 INFO Loaded EURGBP/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:11,962 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 156, 'BIAS_DOWN': 71, 'BIAS_NEUTRAL': 2796}  ambiguous=1710 (total=3023) horizon=12
2026-05-03 08:35:11,964 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected EURGBP — 2973 samples (group=cross) score_means={'bias_up_score': 0.0525, 'bias_down_score': 0.0239} labels={'BIAS_UP': 156, 'BIAS_DOWN': 71, 'BIAS_NEUTRAL': 2746} clean={'BIAS_UP': 156, 'BIAS_DOWN': 71, 'BIAS_NEUTRAL': 1071}
2026-05-03 08:35:12,139 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:12,176 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:12,195 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:12,195 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:12,202 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:12,203 INFO Loaded EURJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:12,416 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 95, 'BIAS_DOWN': 63, 'BIAS_NEUTRAL': 2865}  ambiguous=1742 (total=3023) horizon=12
2026-05-03 08:35:12,420 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected EURJPY — 2973 samples (group=cross) score_means={'bias_up_score': 0.032, 'bias_down_score': 0.0212} labels={'BIAS_UP': 95, 'BIAS_DOWN': 63, 'BIAS_NEUTRAL': 2815} clean={'BIAS_UP': 95, 'BIAS_DOWN': 63, 'BIAS_NEUTRAL': 1099}
2026-05-03 08:35:12,586 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:12,622 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:12,639 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:12,640 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:12,647 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:12,648 INFO Loaded EURUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:12,858 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 98, 'BIAS_DOWN': 57, 'BIAS_NEUTRAL': 2868}  ambiguous=1742 (total=3023) horizon=12
2026-05-03 08:35:12,861 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected EURUSD — 2973 samples (group=dollar) score_means={'bias_up_score': 0.033, 'bias_down_score': 0.0192} labels={'BIAS_UP': 98, 'BIAS_DOWN': 57, 'BIAS_NEUTRAL': 2818} clean={'BIAS_UP': 98, 'BIAS_DOWN': 57, 'BIAS_NEUTRAL': 1105}
2026-05-03 08:35:13,015 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:13,052 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:13,072 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:13,072 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:13,080 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:13,081 INFO Loaded GBPJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:13,283 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 164, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2758}  ambiguous=1723 (total=3023) horizon=12
2026-05-03 08:35:13,285 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected GBPJPY — 2973 samples (group=cross) score_means={'bias_up_score': 0.0552, 'bias_down_score': 0.034} labels={'BIAS_UP': 164, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2708} clean={'BIAS_UP': 164, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 1019}
2026-05-03 08:35:13,431 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:13,465 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:13,483 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:13,483 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:13,491 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:13,492 INFO Loaded GBPUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:13,705 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 79, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2843}  ambiguous=1759 (total=3023) horizon=12
2026-05-03 08:35:13,708 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected GBPUSD — 2973 samples (group=dollar) score_means={'bias_up_score': 0.0266, 'bias_down_score': 0.034} labels={'BIAS_UP': 79, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2793} clean={'BIAS_UP': 79, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 1073}
2026-05-03 08:35:13,881 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:13,924 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:13,944 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:13,944 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:13,952 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:13,953 INFO Loaded NZDUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:14,174 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 54, 'BIAS_DOWN': 54, 'BIAS_NEUTRAL': 2915}  ambiguous=1779 (total=3023) horizon=12
2026-05-03 08:35:14,177 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected NZDUSD — 2973 samples (group=dollar) score_means={'bias_up_score': 0.0182, 'bias_down_score': 0.0182} labels={'BIAS_UP': 54, 'BIAS_DOWN': 54, 'BIAS_NEUTRAL': 2865} clean={'BIAS_UP': 54, 'BIAS_DOWN': 54, 'BIAS_NEUTRAL': 1117}
2026-05-03 08:35:14,329 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:14,364 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:14,382 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:14,383 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:14,390 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:14,391 INFO Loaded USDCAD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:14,589 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 89, 'BIAS_DOWN': 133, 'BIAS_NEUTRAL': 2801}  ambiguous=1770 (total=3023) horizon=12
2026-05-03 08:35:14,591 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected USDCAD — 2973 samples (group=dollar) score_means={'bias_up_score': 0.0299, 'bias_down_score': 0.0447} labels={'BIAS_UP': 89, 'BIAS_DOWN': 133, 'BIAS_NEUTRAL': 2751} clean={'BIAS_UP': 89, 'BIAS_DOWN': 133, 'BIAS_NEUTRAL': 1016}
2026-05-03 08:35:14,736 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:14,768 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:14,785 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:14,785 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:14,792 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:14,793 INFO Loaded USDCHF/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:14,999 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 64, 'BIAS_DOWN': 52, 'BIAS_NEUTRAL': 2907}  ambiguous=1741 (total=3023) horizon=12
2026-05-03 08:35:15,001 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected USDCHF — 2973 samples (group=dollar) score_means={'bias_up_score': 0.0215, 'bias_down_score': 0.0175} labels={'BIAS_UP': 64, 'BIAS_DOWN': 52, 'BIAS_NEUTRAL': 2857} clean={'BIAS_UP': 64, 'BIAS_DOWN': 52, 'BIAS_NEUTRAL': 1148}
2026-05-03 08:35:15,152 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:15,186 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:15,205 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:15,206 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:15,212 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:15,213 INFO Loaded USDJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:15,420 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 102, 'BIAS_DOWN': 78, 'BIAS_NEUTRAL': 2843}  ambiguous=1762 (total=3023) horizon=12
2026-05-03 08:35:15,423 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected USDJPY — 2973 samples (group=dollar) score_means={'bias_up_score': 0.0343, 'bias_down_score': 0.0262} labels={'BIAS_UP': 102, 'BIAS_DOWN': 78, 'BIAS_NEUTRAL': 2793} clean={'BIAS_UP': 102, 'BIAS_DOWN': 78, 'BIAS_NEUTRAL': 1058}
2026-05-03 08:35:15,695 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:15,763 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:15,789 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:15,790 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:15,800 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:15,801 INFO Loaded XAUUSD/4H split=train fold=fold_000: 3204 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:16,039 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 2845}  ambiguous=1801 (total=3204) horizon=12
2026-05-03 08:35:16,042 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected XAUUSD — 3154 samples (group=gold) score_means={'bias_up_score': 0.0672, 'bias_down_score': 0.0466} labels={'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 2795} clean={'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 1029}
2026-05-03 08:35:16,109 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 415, 'BIAS_DOWN': 235, 'BIAS_NEUTRAL': 8269}, 'dollar': {'BIAS_UP': 578, 'BIAS_DOWN': 530, 'BIAS_NEUTRAL': 19703}, 'gold': {'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 2795}}
2026-05-03 08:35:16,109 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0465, 'bias_down_score': 0.0263}, 'dollar': {'bias_up_score': 0.0278, 'bias_down_score': 0.0255}, 'gold': {'bias_up_score': 0.0672, 'bias_down_score': 0.0466}}
2026-05-03 08:35:16,109 INFO Regime[4H mode=htf_bias] label distribution by year: {2016: {'BIAS_UP': 485, 'BIAS_DOWN': 511, 'BIAS_NEUTRAL': 15101}, 2017: {'BIAS_UP': 717, 'BIAS_DOWN': 401, 'BIAS_NEUTRAL': 15515}, 2018: {'BIAS_UP': 3, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 151}}
2026-05-03 08:35:16,110 INFO Regime[4H mode=htf_bias] score means by year: {2016: {'bias_up_score': 0.0301, 'bias_down_score': 0.0317}, 2017: {'bias_up_score': 0.0431, 'bias_down_score': 0.0241}, 2018: {'bias_up_score': 0.0195, 'bias_down_score': 0.0}}
2026-05-03 08:35:16,155 INFO Loaded AUDUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-03 08:35:16,156 INFO Loaded EURGBP/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-03 08:35:16,157 INFO Loaded EURJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-03 08:35:16,158 INFO Loaded EURUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-03 08:35:16,159 INFO Loaded GBPJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-03 08:35:16,160 INFO Loaded GBPUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-03 08:35:16,161 INFO Loaded NZDUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-03 08:35:16,162 INFO Loaded USDCAD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-03 08:35:16,163 INFO Loaded USDCHF/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-03 08:35:16,164 INFO Loaded USDJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-03 08:35:16,165 INFO Loaded XAUUSD/4H split=val fold=fold_000: 1600 bars (2018-01-04 → 2019-01-03)
2026-05-03 08:35:16,171 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:16,173 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:16,174 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:16,174 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:16,174 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:16,175 INFO Loaded AUDUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:16,374 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 5, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1448}  ambiguous=896 (total=1506) horizon=12
2026-05-03 08:35:16,377 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected AUDUSD — 1456 samples (group=dollar) score_means={'bias_up_score': 0.0034, 'bias_down_score': 0.0364} labels={'BIAS_UP': 5, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1398} clean={'BIAS_UP': 5, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 531}
2026-05-03 08:35:16,448 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:16,450 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:16,451 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:16,451 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:16,452 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:16,453 INFO Loaded EURGBP/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:16,656 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 12, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1453}  ambiguous=868 (total=1506) horizon=12
2026-05-03 08:35:16,658 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected EURGBP — 1456 samples (group=cross) score_means={'bias_up_score': 0.0082, 'bias_down_score': 0.0282} labels={'BIAS_UP': 12, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1403} clean={'BIAS_UP': 12, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 575}
2026-05-03 08:35:16,729 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:16,731 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:16,732 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:16,732 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:16,733 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:16,734 INFO Loaded EURJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:16,932 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 20, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1445}  ambiguous=874 (total=1506) horizon=12
2026-05-03 08:35:16,935 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected EURJPY — 1456 samples (group=cross) score_means={'bias_up_score': 0.0137, 'bias_down_score': 0.0282} labels={'BIAS_UP': 20, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1395} clean={'BIAS_UP': 20, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 555}
2026-05-03 08:35:17,005 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:17,008 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:17,008 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:17,009 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:17,009 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:17,010 INFO Loaded EURUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:17,211 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 24, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 1415}  ambiguous=876 (total=1506) horizon=12
2026-05-03 08:35:17,214 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected EURUSD — 1456 samples (group=dollar) score_means={'bias_up_score': 0.0165, 'bias_down_score': 0.046} labels={'BIAS_UP': 24, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 1365} clean={'BIAS_UP': 24, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 522}
2026-05-03 08:35:17,282 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:17,284 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:17,285 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:17,286 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:17,286 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:17,287 INFO Loaded GBPJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:17,474 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 27, 'BIAS_DOWN': 37, 'BIAS_NEUTRAL': 1442}  ambiguous=926 (total=1506) horizon=12
2026-05-03 08:35:17,476 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected GBPJPY — 1456 samples (group=cross) score_means={'bias_up_score': 0.0185, 'bias_down_score': 0.0254} labels={'BIAS_UP': 27, 'BIAS_DOWN': 37, 'BIAS_NEUTRAL': 1392} clean={'BIAS_UP': 27, 'BIAS_DOWN': 37, 'BIAS_NEUTRAL': 506}
2026-05-03 08:35:17,539 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:17,542 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:17,542 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:17,543 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:17,543 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:17,544 INFO Loaded GBPUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:17,731 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 50, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1371}  ambiguous=874 (total=1506) horizon=12
2026-05-03 08:35:17,733 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected GBPUSD — 1456 samples (group=dollar) score_means={'bias_up_score': 0.0343, 'bias_down_score': 0.0584} labels={'BIAS_UP': 50, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1321} clean={'BIAS_UP': 50, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 476}
2026-05-03 08:35:17,795 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:17,797 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:17,798 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:17,798 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:17,798 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:17,799 INFO Loaded NZDUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:17,981 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 23, 'BIAS_DOWN': 80, 'BIAS_NEUTRAL': 1403}  ambiguous=896 (total=1506) horizon=12
2026-05-03 08:35:17,984 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected NZDUSD — 1456 samples (group=dollar) score_means={'bias_up_score': 0.0158, 'bias_down_score': 0.0549} labels={'BIAS_UP': 23, 'BIAS_DOWN': 80, 'BIAS_NEUTRAL': 1353} clean={'BIAS_UP': 23, 'BIAS_DOWN': 80, 'BIAS_NEUTRAL': 482}
2026-05-03 08:35:18,044 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:18,047 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:18,048 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:18,048 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:18,048 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:18,049 INFO Loaded USDCAD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:18,248 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 48, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 1445}  ambiguous=907 (total=1506) horizon=12
2026-05-03 08:35:18,250 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected USDCAD — 1456 samples (group=dollar) score_means={'bias_up_score': 0.033, 'bias_down_score': 0.0089} labels={'BIAS_UP': 48, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 1395} clean={'BIAS_UP': 48, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 522}
2026-05-03 08:35:18,320 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:18,322 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:18,323 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:18,323 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:18,323 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:18,325 INFO Loaded USDCHF/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:18,507 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 68, 'BIAS_DOWN': 45, 'BIAS_NEUTRAL': 1393}  ambiguous=848 (total=1506) horizon=12
2026-05-03 08:35:18,510 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected USDCHF — 1456 samples (group=dollar) score_means={'bias_up_score': 0.0467, 'bias_down_score': 0.0309} labels={'BIAS_UP': 68, 'BIAS_DOWN': 45, 'BIAS_NEUTRAL': 1343} clean={'BIAS_UP': 68, 'BIAS_DOWN': 45, 'BIAS_NEUTRAL': 530}
2026-05-03 08:35:18,572 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:18,574 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:18,575 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:18,576 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:18,576 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:18,577 INFO Loaded USDJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:18,761 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 58, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1418}  ambiguous=888 (total=1506) horizon=12
2026-05-03 08:35:18,764 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected USDJPY — 1456 samples (group=dollar) score_means={'bias_up_score': 0.0398, 'bias_down_score': 0.0206} labels={'BIAS_UP': 58, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1368} clean={'BIAS_UP': 58, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 510}
2026-05-03 08:35:18,831 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:18,838 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:18,839 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:18,840 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:18,840 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:18,841 INFO Loaded XAUUSD/4H split=val fold=fold_000: 1600 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:19,040 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 18, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 1547}  ambiguous=851 (total=1600) horizon=12
2026-05-03 08:35:19,043 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected XAUUSD — 1550 samples (group=gold) score_means={'bias_up_score': 0.0116, 'bias_down_score': 0.0226} labels={'BIAS_UP': 18, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 1497} clean={'BIAS_UP': 18, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 677}
2026-05-03 08:35:19,104 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 59, 'BIAS_DOWN': 119, 'BIAS_NEUTRAL': 4190}, 'dollar': {'BIAS_UP': 276, 'BIAS_DOWN': 373, 'BIAS_NEUTRAL': 9543}, 'gold': {'BIAS_UP': 18, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 1497}}
2026-05-03 08:35:19,104 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0135, 'bias_down_score': 0.0272}, 'dollar': {'bias_up_score': 0.0271, 'bias_down_score': 0.0366}, 'gold': {'bias_up_score': 0.0116, 'bias_down_score': 0.0226}}
2026-05-03 08:35:19,104 INFO Regime[4H mode=htf_bias] label distribution by year: {2018: {'BIAS_UP': 352, 'BIAS_DOWN': 521, 'BIAS_NEUTRAL': 15083}, 2019: {'BIAS_UP': 1, 'BIAS_DOWN': 6, 'BIAS_NEUTRAL': 147}}
2026-05-03 08:35:19,104 INFO Regime[4H mode=htf_bias] score means by year: {2018: {'bias_up_score': 0.0221, 'bias_down_score': 0.0327}, 2019: {'bias_up_score': 0.0065, 'bias_down_score': 0.039}}
2026-05-03 08:35:19,147 INFO Regime phase HTF dataset build fold=fold_000: 8.2s (train=32884 val=16110)
2026-05-03 08:35:19,148 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260503_083519
2026-05-03 08:35:19,343 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=24, n_classes=2)
2026-05-03 08:35:19,343 INFO Regime 4H/htf_bias warm start enabled from existing weights
2026-05-03 08:35:19,347 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=32884 val=16110 train_labels={'BIAS_UP': 1205, 'BIAS_DOWN': 912, 'BIAS_NEUTRAL': 30767} val_labels={'BIAS_UP': 353, 'BIAS_DOWN': 527, 'BIAS_NEUTRAL': 15230}
2026-05-03 08:35:19,347 INFO RegimeClassifier[mode=htf_bias]: warm start HTF score head
2026-05-03 08:35:19,348 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-03 08:35:19,348 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 20.0, 'bias_down_score': 20.0}
2026-05-03 08:35:22,368 INFO Regime HTF score epoch  1/50 — tr=0.4703 va=0.4378 acc=0.855 bal=0.915 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.918, 'BIAS_DOWN': 0.977, 'BIAS_NEUTRAL': 0.849} precision={'BIAS_UP': 0.279, 'BIAS_DOWN': 0.261, 'BIAS_NEUTRAL': 0.997}
2026-05-03 08:35:23,333 INFO Regime HTF score epoch  2/50 — tr=0.4662 va=0.4310 bal=0.908
2026-05-03 08:35:24,326 INFO Regime HTF score epoch  3/50 — tr=0.4696 va=0.4349 bal=0.913
2026-05-03 08:35:25,284 INFO Regime HTF score epoch  4/50 — tr=0.4629 va=0.4317 bal=0.912
2026-05-03 08:35:26,258 INFO Regime HTF score epoch  5/50 — tr=0.4590 va=0.4187 acc=0.861 bal=0.900 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.89, 'BIAS_DOWN': 0.954, 'BIAS_NEUTRAL': 0.858} precision={'BIAS_UP': 0.282, 'BIAS_DOWN': 0.269, 'BIAS_NEUTRAL': 0.995}
2026-05-03 08:35:27,240 INFO Regime HTF score epoch  6/50 — tr=0.4582 va=0.4201 bal=0.902
2026-05-03 08:35:28,221 INFO Regime HTF score epoch  7/50 — tr=0.4532 va=0.4410 bal=0.920
2026-05-03 08:35:29,216 INFO Regime HTF score epoch  8/50 — tr=0.4542 va=0.4467 bal=0.921
2026-05-03 08:35:30,200 INFO Regime HTF score epoch  9/50 — tr=0.4475 va=0.4491 bal=0.922
2026-05-03 08:35:31,182 INFO Regime HTF score epoch 10/50 — tr=0.4448 va=0.4524 acc=0.841 bal=0.923 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.949, 'BIAS_DOWN': 0.985, 'BIAS_NEUTRAL': 0.834} precision={'BIAS_UP': 0.252, 'BIAS_DOWN': 0.252, 'BIAS_NEUTRAL': 0.998}
2026-05-03 08:35:31,182 INFO Regime HTF score early stop at epoch 10
2026-05-03 08:35:32,096 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.850 margin=0.000 precision={'BIAS_UP': 0.279, 'BIAS_DOWN': 0.261, 'BIAS_NEUTRAL': 0.997} recall={'BIAS_UP': 0.918, 'BIAS_DOWN': 0.977, 'BIAS_NEUTRAL': 0.849} f1={'BIAS_UP': 0.428, 'BIAS_DOWN': 0.412, 'BIAS_NEUTRAL': 0.917} confusion=[[324, 0, 29], [0, 515, 12], [836, 1457, 12937]] score_mae={'bias_up_score': 0.1598, 'bias_down_score': 0.2102} pred_share={'BIAS_UP': 0.072, 'BIAS_DOWN': 0.1224, 'BIAS_NEUTRAL': 0.8056}
2026-05-03 08:35:32,097 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.279, 'BIAS_DOWN': 0.261, 'BIAS_NEUTRAL': 0.997} min_precision=0.300 recall={'BIAS_UP': 0.918, 'BIAS_DOWN': 0.977, 'BIAS_NEUTRAL': 0.849} min_recall=0.100 f1={'BIAS_UP': 0.428, 'BIAS_DOWN': 0.412, 'BIAS_NEUTRAL': 0.917} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_UP', 'BIAS_DOWN'] weak_recall=[] weak_f1=[] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-03 08:35:32,101 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-03 08:35:32,101 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-03 08:35:32,101 INFO Regime phase HTF train fold=fold_000: 12.8s
2026-05-03 08:35:32,200 INFO Regime HTF complete fold=fold_000: acc=0.855 bal=0.915 train=32884 val=16110 per_class={'BIAS_UP': 0.918, 'BIAS_DOWN': 0.977, 'BIAS_NEUTRAL': 0.849} precision={'BIAS_UP': 0.279, 'BIAS_DOWN': 0.261, 'BIAS_NEUTRAL': 0.997} threshold=0.850 margin=0.000
2026-05-03 08:35:32,201 INFO Loaded XAUUSD/4H split=train fold=fold_000: 3204 bars (2016-01-04 → 2018-01-03)
2026-05-03 08:35:32,297 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 2845}  ambiguous=1801 (total=3204) horizon=12
2026-05-03 08:35:32,300 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 4.1568627450980395, 'BIAS_DOWN': 3.972972972972973, 'BIAS_NEUTRAL': 31.96629213483146}
2026-05-03 08:35:32,302 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 212, 'mean': 0.0011890919998414733, 'mean_over_std': 0.3850896851317838}, 'BIAS_DOWN': {'n': 147, 'mean': -0.0013091755049482925, 'mean_over_std': -0.3942426410778961}, 'BIAS_NEUTRAL': {'n': 2844, 'mean': 5.552802176910261e-05, 'mean_over_std': 0.015959039476050135}}
2026-05-03 08:35:32,303 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 212, 'mean': 0.0011890919998414733, 'mean_over_std': 0.3850896851317838}, 'BIAS_DOWN': {'n': 147, 'mean': -0.0013091755049482925, 'mean_over_std': -0.3942426410778961}, 'BIAS_NEUTRAL': {'n': 1044, 'mean': 0.00010149382300531956, 'mean_over_std': 0.037249435653574005}}
2026-05-03 08:35:32,306 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-03 08:35:32,307 INFO Loaded AUDUSD/1H split=train fold=fold_000: 11773 bars (2016-01-04 → 2018-01-03)
2026-05-03 08:35:32,310 INFO Loaded EURGBP/1H split=train fold=fold_000: 11773 bars (2016-01-04 → 2018-01-03)
2026-05-03 08:35:32,311 INFO Loaded EURJPY/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-03 08:35:32,312 INFO Loaded EURUSD/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-03 08:35:32,314 INFO Loaded GBPJPY/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-03 08:35:32,315 INFO Loaded GBPUSD/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-03 08:35:32,316 INFO Loaded NZDUSD/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-03 08:35:32,317 INFO Loaded USDCAD/1H split=train fold=fold_000: 11773 bars (2016-01-04 → 2018-01-03)
2026-05-03 08:35:32,318 INFO Loaded USDCHF/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-03 08:35:32,320 INFO Loaded USDJPY/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-03 08:35:32,322 INFO Loaded XAUUSD/1H split=train fold=fold_000: 11914 bars (2016-01-04 → 2018-01-03)
2026-05-03 08:35:32,333 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:32,337 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:32,339 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:32,340 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:32,340 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:32,343 INFO Loaded AUDUSD/1H split=train fold=fold_000: 11773 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:32,660 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected AUDUSD — 11723 samples (group=dollar) score_means={'trend_score': 0.4834, 'range_score': 0.2374, 'chop_score': 0.4688, 'volatility_percentile': 0.3652, 'consolidation_score': 0.2}
2026-05-03 08:35:32,769 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:32,773 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:32,775 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:32,776 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:32,776 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:32,778 INFO Loaded EURGBP/1H split=train fold=fold_000: 11773 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:33,095 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected EURGBP — 11723 samples (group=cross) score_means={'trend_score': 0.497, 'range_score': 0.2358, 'chop_score': 0.4623, 'volatility_percentile': 0.3783, 'consolidation_score': 0.1905}
2026-05-03 08:35:33,201 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:33,206 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:33,207 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:33,207 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:33,207 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:33,209 INFO Loaded EURJPY/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:33,523 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected EURJPY — 11722 samples (group=cross) score_means={'trend_score': 0.4873, 'range_score': 0.2384, 'chop_score': 0.4674, 'volatility_percentile': 0.3763, 'consolidation_score': 0.1925}
2026-05-03 08:35:33,624 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:33,626 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:33,627 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:33,627 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:33,628 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:33,629 INFO Loaded EURUSD/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:33,941 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected EURUSD — 11722 samples (group=dollar) score_means={'trend_score': 0.4931, 'range_score': 0.2373, 'chop_score': 0.464, 'volatility_percentile': 0.3716, 'consolidation_score': 0.1896}
2026-05-03 08:35:34,044 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:34,046 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:34,047 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:34,047 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:34,048 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:34,049 INFO Loaded GBPJPY/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:34,361 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected GBPJPY — 11722 samples (group=cross) score_means={'trend_score': 0.5009, 'range_score': 0.2311, 'chop_score': 0.4571, 'volatility_percentile': 0.3758, 'consolidation_score': 0.1946}
2026-05-03 08:35:34,463 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:34,466 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:34,467 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:34,468 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:34,468 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:34,470 INFO Loaded GBPUSD/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:34,780 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected GBPUSD — 11722 samples (group=dollar) score_means={'trend_score': 0.5037, 'range_score': 0.2323, 'chop_score': 0.4563, 'volatility_percentile': 0.3792, 'consolidation_score': 0.186}
2026-05-03 08:35:34,880 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:34,882 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:34,883 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:34,883 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:34,883 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:34,885 INFO Loaded NZDUSD/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:35,193 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected NZDUSD — 11722 samples (group=dollar) score_means={'trend_score': 0.4841, 'range_score': 0.2391, 'chop_score': 0.4687, 'volatility_percentile': 0.3726, 'consolidation_score': 0.1911}
2026-05-03 08:35:35,295 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:35,298 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:35,299 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:35,299 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:35,299 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:35,301 INFO Loaded USDCAD/1H split=train fold=fold_000: 11773 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:35,616 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected USDCAD — 11723 samples (group=dollar) score_means={'trend_score': 0.4974, 'range_score': 0.2331, 'chop_score': 0.4561, 'volatility_percentile': 0.3775, 'consolidation_score': 0.1896}
2026-05-03 08:35:35,720 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:35,724 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:35,725 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:35,725 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:35,726 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:35,727 INFO Loaded USDCHF/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:36,045 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected USDCHF — 11722 samples (group=dollar) score_means={'trend_score': 0.4674, 'range_score': 0.2504, 'chop_score': 0.4822, 'volatility_percentile': 0.3731, 'consolidation_score': 0.1894}
2026-05-03 08:35:36,148 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:36,150 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:36,151 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:36,151 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:36,152 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:36,153 INFO Loaded USDJPY/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:36,469 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected USDJPY — 11722 samples (group=dollar) score_means={'trend_score': 0.4991, 'range_score': 0.231, 'chop_score': 0.4562, 'volatility_percentile': 0.3679, 'consolidation_score': 0.1984}
2026-05-03 08:35:36,581 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:36,585 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:36,586 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:36,586 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:36,587 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:36,589 INFO Loaded XAUUSD/1H split=train fold=fold_000: 11914 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:36,922 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected XAUUSD — 11864 samples (group=gold) score_means={'trend_score': 0.5075, 'range_score': 0.2273, 'chop_score': 0.4518, 'volatility_percentile': 0.3694, 'consolidation_score': 0.1952}
2026-05-03 08:35:37,024 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4951, 'range_score': 0.2351, 'chop_score': 0.4622, 'volatility_percentile': 0.3768, 'consolidation_score': 0.1925}, 'dollar': {'trend_score': 0.4897, 'range_score': 0.2372, 'chop_score': 0.4646, 'volatility_percentile': 0.3724, 'consolidation_score': 0.192}, 'gold': {'trend_score': 0.5075, 'range_score': 0.2273, 'chop_score': 0.4518, 'volatility_percentile': 0.3694, 'consolidation_score': 0.1952}}
2026-05-03 08:35:37,024 INFO Regime[1H mode=ltf_behaviour] score means by year: {2016: {'trend_score': 0.4914, 'range_score': 0.2348, 'chop_score': 0.4627, 'volatility_percentile': 0.375, 'consolidation_score': 0.192}, 2017: {'trend_score': 0.4941, 'range_score': 0.2364, 'chop_score': 0.463, 'volatility_percentile': 0.3716, 'consolidation_score': 0.1934}, 2018: {'trend_score': 0.51, 'range_score': 0.2569, 'chop_score': 0.4423, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1324}}
2026-05-03 08:35:37,103 INFO Loaded AUDUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-03 08:35:37,104 INFO Loaded EURGBP/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-03 08:35:37,105 INFO Loaded EURJPY/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-03 08:35:37,106 INFO Loaded EURUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-03 08:35:37,108 INFO Loaded GBPJPY/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-03 08:35:37,109 INFO Loaded GBPUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-03 08:35:37,110 INFO Loaded NZDUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-03 08:35:37,112 INFO Loaded USDCAD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-03 08:35:37,113 INFO Loaded USDCHF/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-03 08:35:37,114 INFO Loaded USDJPY/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-03 08:35:37,116 INFO Loaded XAUUSD/1H split=val fold=fold_000: 6034 bars (2018-01-04 → 2019-01-03)
2026-05-03 08:35:37,121 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:37,123 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:37,124 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:37,124 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:37,125 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:37,126 INFO Loaded AUDUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:37,355 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected AUDUSD — 5812 samples (group=dollar) score_means={'trend_score': 0.484, 'range_score': 0.2467, 'chop_score': 0.4726, 'volatility_percentile': 0.3956, 'consolidation_score': 0.1777}
2026-05-03 08:35:37,457 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:37,460 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:37,460 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:37,461 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:37,461 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:37,463 INFO Loaded EURGBP/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:37,688 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected EURGBP — 5812 samples (group=cross) score_means={'trend_score': 0.4626, 'range_score': 0.2497, 'chop_score': 0.4853, 'volatility_percentile': 0.3975, 'consolidation_score': 0.1692}
2026-05-03 08:35:37,789 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:37,791 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:37,792 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:37,792 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:37,793 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:37,794 INFO Loaded EURJPY/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:38,022 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected EURJPY — 5812 samples (group=cross) score_means={'trend_score': 0.4749, 'range_score': 0.2394, 'chop_score': 0.474, 'volatility_percentile': 0.3878, 'consolidation_score': 0.1827}
2026-05-03 08:35:38,125 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:38,128 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:38,128 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:38,129 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:38,129 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:38,131 INFO Loaded EURUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:38,362 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected EURUSD — 5812 samples (group=dollar) score_means={'trend_score': 0.4993, 'range_score': 0.2343, 'chop_score': 0.4572, 'volatility_percentile': 0.389, 'consolidation_score': 0.1807}
2026-05-03 08:35:38,466 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:38,469 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:38,469 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:38,470 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:38,470 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:38,472 INFO Loaded GBPJPY/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:38,697 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected GBPJPY — 5812 samples (group=cross) score_means={'trend_score': 0.4888, 'range_score': 0.2412, 'chop_score': 0.4689, 'volatility_percentile': 0.3963, 'consolidation_score': 0.1732}
2026-05-03 08:35:38,799 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:38,802 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:38,802 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:38,803 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:38,803 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:38,805 INFO Loaded GBPUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:39,033 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected GBPUSD — 5812 samples (group=dollar) score_means={'trend_score': 0.5007, 'range_score': 0.2339, 'chop_score': 0.4559, 'volatility_percentile': 0.3971, 'consolidation_score': 0.1718}
2026-05-03 08:35:39,135 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:39,136 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:39,137 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:39,137 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:39,138 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:39,139 INFO Loaded NZDUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:39,370 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected NZDUSD — 5812 samples (group=dollar) score_means={'trend_score': 0.4931, 'range_score': 0.2353, 'chop_score': 0.4587, 'volatility_percentile': 0.3902, 'consolidation_score': 0.1824}
2026-05-03 08:35:39,472 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:39,474 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:39,475 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:39,475 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:39,475 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:39,477 INFO Loaded USDCAD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:39,701 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected USDCAD — 5812 samples (group=dollar) score_means={'trend_score': 0.4808, 'range_score': 0.2476, 'chop_score': 0.4717, 'volatility_percentile': 0.3857, 'consolidation_score': 0.1768}
2026-05-03 08:35:39,804 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:39,806 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:39,807 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:39,807 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:39,808 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:39,809 INFO Loaded USDCHF/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:40,059 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected USDCHF — 5812 samples (group=dollar) score_means={'trend_score': 0.4799, 'range_score': 0.2431, 'chop_score': 0.4697, 'volatility_percentile': 0.3907, 'consolidation_score': 0.1794}
2026-05-03 08:35:40,163 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:40,166 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:40,167 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:40,167 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:40,167 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:40,169 INFO Loaded USDJPY/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:40,451 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected USDJPY — 5812 samples (group=dollar) score_means={'trend_score': 0.4943, 'range_score': 0.2334, 'chop_score': 0.4614, 'volatility_percentile': 0.3872, 'consolidation_score': 0.1806}
2026-05-03 08:35:40,569 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:40,572 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:40,574 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:40,574 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:40,574 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:40,577 INFO Loaded XAUUSD/1H split=val fold=fold_000: 6034 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:40,831 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected XAUUSD — 5984 samples (group=gold) score_means={'trend_score': 0.4716, 'range_score': 0.2479, 'chop_score': 0.4761, 'volatility_percentile': 0.3856, 'consolidation_score': 0.1812}
2026-05-03 08:35:40,930 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4754, 'range_score': 0.2434, 'chop_score': 0.476, 'volatility_percentile': 0.3939, 'consolidation_score': 0.175}, 'dollar': {'trend_score': 0.4903, 'range_score': 0.2392, 'chop_score': 0.4639, 'volatility_percentile': 0.3908, 'consolidation_score': 0.1785}, 'gold': {'trend_score': 0.4716, 'range_score': 0.2479, 'chop_score': 0.4761, 'volatility_percentile': 0.3856, 'consolidation_score': 0.1812}}
2026-05-03 08:35:40,931 INFO Regime[1H mode=ltf_behaviour] score means by year: {2018: {'trend_score': 0.4841, 'range_score': 0.2416, 'chop_score': 0.4687, 'volatility_percentile': 0.3892, 'consolidation_score': 0.1792}, 2019: {'trend_score': 0.5315, 'range_score': 0.1889, 'chop_score': 0.4263, 'volatility_percentile': 0.5999, 'consolidation_score': 0.0339}}
2026-05-03 08:35:41,006 INFO Regime phase LTF dataset build fold=fold_000: 8.7s (train=129087 val=64104)
2026-05-03 08:35:41,007 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260503_083541
2026-05-03 08:35:41,011 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=31, n_classes=5)
2026-05-03 08:35:41,012 INFO Regime 1H/ltf_behaviour warm start enabled from existing weights
2026-05-03 08:35:41,029 INFO RegimeClassifier[mode=ltf_behaviour]: warm start score head
2026-05-03 08:35:41,029 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-03 08:35:41,313 INFO Regime score epoch  1/50 — tr=0.0042 va=0.0013 mae={'trend_score': 0.025, 'range_score': 0.0384, 'chop_score': 0.027, 'volatility_percentile': 0.0167, 'consolidation_score': 0.0237}
2026-05-03 08:35:41,568 INFO Regime score epoch  2/50 — tr=0.0042 va=0.0012
2026-05-03 08:35:41,821 INFO Regime score epoch  3/50 — tr=0.0042 va=0.0012
2026-05-03 08:35:42,089 INFO Regime score epoch  4/50 — tr=0.0042 va=0.0012
2026-05-03 08:35:42,351 INFO Regime score epoch  5/50 — tr=0.0042 va=0.0012 mae={'trend_score': 0.0236, 'range_score': 0.0384, 'chop_score': 0.0261, 'volatility_percentile': 0.0163, 'consolidation_score': 0.0236}
2026-05-03 08:35:42,615 INFO Regime score epoch  6/50 — tr=0.0042 va=0.0012
2026-05-03 08:35:42,868 INFO Regime score epoch  7/50 — tr=0.0041 va=0.0012
2026-05-03 08:35:43,122 INFO Regime score epoch  8/50 — tr=0.0041 va=0.0012
2026-05-03 08:35:43,390 INFO Regime score epoch  9/50 — tr=0.0041 va=0.0012
2026-05-03 08:35:43,652 INFO Regime score epoch 10/50 — tr=0.0041 va=0.0012 mae={'trend_score': 0.0232, 'range_score': 0.0378, 'chop_score': 0.0253, 'volatility_percentile': 0.0163, 'consolidation_score': 0.0236}
2026-05-03 08:35:43,906 INFO Regime score epoch 11/50 — tr=0.0041 va=0.0012
2026-05-03 08:35:44,167 INFO Regime score epoch 12/50 — tr=0.0041 va=0.0012
2026-05-03 08:35:44,432 INFO Regime score epoch 13/50 — tr=0.0041 va=0.0012
2026-05-03 08:35:44,694 INFO Regime score epoch 14/50 — tr=0.0040 va=0.0011
2026-05-03 08:35:44,945 INFO Regime score epoch 15/50 — tr=0.0040 va=0.0011 mae={'trend_score': 0.0226, 'range_score': 0.0376, 'chop_score': 0.0243, 'volatility_percentile': 0.016, 'consolidation_score': 0.0231}
2026-05-03 08:35:45,197 INFO Regime score epoch 16/50 — tr=0.0040 va=0.0011
2026-05-03 08:35:45,452 INFO Regime score epoch 17/50 — tr=0.0040 va=0.0011
2026-05-03 08:35:45,717 INFO Regime score epoch 18/50 — tr=0.0040 va=0.0011
2026-05-03 08:35:45,974 INFO Regime score epoch 19/50 — tr=0.0039 va=0.0011
2026-05-03 08:35:46,241 INFO Regime score epoch 20/50 — tr=0.0039 va=0.0011 mae={'trend_score': 0.0221, 'range_score': 0.0369, 'chop_score': 0.0239, 'volatility_percentile': 0.0156, 'consolidation_score': 0.0227}
2026-05-03 08:35:46,496 INFO Regime score epoch 21/50 — tr=0.0039 va=0.0011
2026-05-03 08:35:46,760 INFO Regime score epoch 22/50 — tr=0.0039 va=0.0011
2026-05-03 08:35:47,025 INFO Regime score epoch 23/50 — tr=0.0039 va=0.0011
2026-05-03 08:35:47,288 INFO Regime score epoch 24/50 — tr=0.0039 va=0.0011
2026-05-03 08:35:47,543 INFO Regime score epoch 25/50 — tr=0.0039 va=0.0011 mae={'trend_score': 0.0214, 'range_score': 0.0361, 'chop_score': 0.0231, 'volatility_percentile': 0.0155, 'consolidation_score': 0.0223}
2026-05-03 08:35:47,802 INFO Regime score epoch 26/50 — tr=0.0039 va=0.0011
2026-05-03 08:35:48,064 INFO Regime score epoch 27/50 — tr=0.0039 va=0.0011
2026-05-03 08:35:48,319 INFO Regime score epoch 28/50 — tr=0.0039 va=0.0011
2026-05-03 08:35:48,580 INFO Regime score epoch 29/50 — tr=0.0038 va=0.0011
2026-05-03 08:35:48,841 INFO Regime score epoch 30/50 — tr=0.0038 va=0.0011 mae={'trend_score': 0.0213, 'range_score': 0.0363, 'chop_score': 0.0233, 'volatility_percentile': 0.0154, 'consolidation_score': 0.0222}
2026-05-03 08:35:49,105 INFO Regime score epoch 31/50 — tr=0.0038 va=0.0010
2026-05-03 08:35:49,376 INFO Regime score epoch 32/50 — tr=0.0038 va=0.0010
2026-05-03 08:35:49,631 INFO Regime score epoch 33/50 — tr=0.0038 va=0.0010
2026-05-03 08:35:49,894 INFO Regime score epoch 34/50 — tr=0.0038 va=0.0010
2026-05-03 08:35:50,148 INFO Regime score epoch 35/50 — tr=0.0038 va=0.0010 mae={'trend_score': 0.021, 'range_score': 0.0358, 'chop_score': 0.0227, 'volatility_percentile': 0.0153, 'consolidation_score': 0.0218}
2026-05-03 08:35:50,429 INFO Regime score epoch 36/50 — tr=0.0038 va=0.0010
2026-05-03 08:35:50,678 INFO Regime score epoch 37/50 — tr=0.0038 va=0.0010
2026-05-03 08:35:50,929 INFO Regime score epoch 38/50 — tr=0.0038 va=0.0010
2026-05-03 08:35:51,181 INFO Regime score epoch 39/50 — tr=0.0038 va=0.0010
2026-05-03 08:35:51,440 INFO Regime score epoch 40/50 — tr=0.0038 va=0.0010 mae={'trend_score': 0.0208, 'range_score': 0.0361, 'chop_score': 0.0224, 'volatility_percentile': 0.0152, 'consolidation_score': 0.0215}
2026-05-03 08:35:51,697 INFO Regime score epoch 41/50 — tr=0.0038 va=0.0010
2026-05-03 08:35:51,962 INFO Regime score epoch 42/50 — tr=0.0038 va=0.0010
2026-05-03 08:35:52,220 INFO Regime score epoch 43/50 — tr=0.0038 va=0.0010
2026-05-03 08:35:52,478 INFO Regime score epoch 44/50 — tr=0.0038 va=0.0010
2026-05-03 08:35:52,736 INFO Regime score epoch 45/50 — tr=0.0038 va=0.0010 mae={'trend_score': 0.0207, 'range_score': 0.0358, 'chop_score': 0.0223, 'volatility_percentile': 0.0153, 'consolidation_score': 0.0216}
2026-05-03 08:35:52,988 INFO Regime score epoch 46/50 — tr=0.0038 va=0.0010
2026-05-03 08:35:53,240 INFO Regime score epoch 47/50 — tr=0.0038 va=0.0010
2026-05-03 08:35:53,494 INFO Regime score epoch 48/50 — tr=0.0038 va=0.0010
2026-05-03 08:35:53,745 INFO Regime score epoch 49/50 — tr=0.0038 va=0.0010
2026-05-03 08:35:54,003 INFO Regime score epoch 50/50 — tr=0.0038 va=0.0010 mae={'trend_score': 0.0207, 'range_score': 0.036, 'chop_score': 0.0227, 'volatility_percentile': 0.0152, 'consolidation_score': 0.0219}
2026-05-03 08:35:54,042 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0208, 'range_score': 0.0355, 'chop_score': 0.0221, 'volatility_percentile': 0.0152, 'consolidation_score': 0.0215} mse={'trend_score': 0.00075, 'range_score': 0.00206, 'chop_score': 0.00079, 'volatility_percentile': 0.00042, 'consolidation_score': 0.00105} corr={'trend_score': 0.9923, 'range_score': 0.9515, 'chop_score': 0.9902, 'volatility_percentile': 0.9954, 'consolidation_score': 0.988} pred_std={'trend_score': 0.2167, 'range_score': 0.1331, 'chop_score': 0.1827, 'volatility_percentile': 0.2125, 'consolidation_score': 0.2063} target_std={'trend_score': 0.2203, 'range_score': 0.1457, 'chop_score': 0.1926, 'volatility_percentile': 0.2123, 'consolidation_score': 0.2089}
2026-05-03 08:35:54,047 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-03 08:35:54,047 INFO Regime phase LTF train fold=fold_000: 13.0s
2026-05-03 08:35:54,146 INFO Regime LTF complete fold=fold_000: score_accuracy=0.977, train=129087 val=64104 mae={'trend_score': 0.0208, 'range_score': 0.0355, 'chop_score': 0.0221, 'volatility_percentile': 0.0152, 'consolidation_score': 0.0215}
2026-05-03 08:35:54,149 INFO Loaded XAUUSD/1H split=train fold=fold_000: 11914 bars (2016-01-04 → 2018-01-03)
2026-05-03 08:35:54,288 INFO Regime[1H mode=ltf_behaviour fold=fold_000] LTF score diagnostics on XAUUSD:
{'trend_score': {'mean': 0.507, 'q10': 0.2031, 'q50': 0.5032, 'q90': 0.8121}, 'range_score': {'mean': 0.2284, 'q10': 0.0527, 'q50': 0.2, 'q90': 0.4305}, 'chop_score': {'mean': 0.4525, 'q10': 0.2007, 'q50': 0.4407, 'q90': 0.7194}, 'volatility_percentile': {'mean': 0.3694, 'q10': 0.0827, 'q50': 0.3584, 'q90': 0.6692}, 'consolidation_score': {'mean': 0.1944, 'q10': 0.0, 'q50': 0.1206, 'q90': 0.5428}}
2026-05-03 08:35:54,292 INFO === Regime rolling fold 2/3: fold_001 ===
2026-05-03 08:35:54,292 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-03 08:35:54,292 INFO Split boundaries loaded fold=fold_001/3 — train 2018-01-04→2020-01-03  val 2020-01-06→2020-12-31  test 2023-08-07→2025-08-05
2026-05-03 08:35:54,293 INFO Loaded AUDUSD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-03 08:35:54,294 INFO Loaded EURGBP/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-03 08:35:54,295 INFO Loaded EURJPY/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-03 08:35:54,295 INFO Loaded EURUSD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-03 08:35:54,296 INFO Loaded GBPJPY/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
2026-05-03 08:35:54,297 INFO Loaded GBPUSD/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
2026-05-03 08:35:54,298 INFO Loaded NZDUSD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-03 08:35:54,299 INFO Loaded USDCAD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-03 08:35:54,299 INFO Loaded USDCHF/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-03 08:35:54,300 INFO Loaded USDJPY/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
2026-05-03 08:35:54,301 INFO Loaded XAUUSD/4H split=train fold=fold_001: 3193 bars (2018-01-04 → 2020-01-03)
2026-05-03 08:35:54,306 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:54,308 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:54,309 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:54,309 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:54,310 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:54,310 INFO Loaded AUDUSD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:54,517 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 34, 'BIAS_DOWN': 105, 'BIAS_NEUTRAL': 2867}  ambiguous=1757 (total=3006) horizon=12
2026-05-03 08:35:54,520 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected AUDUSD — 2956 samples (group=dollar) score_means={'bias_up_score': 0.0115, 'bias_down_score': 0.0355} labels={'BIAS_UP': 34, 'BIAS_DOWN': 105, 'BIAS_NEUTRAL': 2817} clean={'BIAS_UP': 34, 'BIAS_DOWN': 105, 'BIAS_NEUTRAL': 1089}
2026-05-03 08:35:54,625 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:54,630 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:54,632 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:54,632 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:54,632 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:54,633 INFO Loaded EURGBP/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:54,842 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 49, 'BIAS_DOWN': 75, 'BIAS_NEUTRAL': 2882}  ambiguous=1672 (total=3006) horizon=12
2026-05-03 08:35:54,845 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected EURGBP — 2956 samples (group=cross) score_means={'bias_up_score': 0.0166, 'bias_down_score': 0.0254} labels={'BIAS_UP': 49, 'BIAS_DOWN': 75, 'BIAS_NEUTRAL': 2832} clean={'BIAS_UP': 49, 'BIAS_DOWN': 75, 'BIAS_NEUTRAL': 1200}
2026-05-03 08:35:54,947 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:54,949 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:54,950 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:54,951 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:54,951 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:54,952 INFO Loaded EURJPY/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:55,156 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 45, 'BIAS_DOWN': 69, 'BIAS_NEUTRAL': 2892}  ambiguous=1719 (total=3006) horizon=12
2026-05-03 08:35:55,159 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected EURJPY — 2956 samples (group=cross) score_means={'bias_up_score': 0.0152, 'bias_down_score': 0.0233} labels={'BIAS_UP': 45, 'BIAS_DOWN': 69, 'BIAS_NEUTRAL': 2842} clean={'BIAS_UP': 45, 'BIAS_DOWN': 69, 'BIAS_NEUTRAL': 1157}
2026-05-03 08:35:55,263 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:55,265 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:55,267 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:55,267 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:55,268 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:55,269 INFO Loaded EURUSD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:55,475 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 35, 'BIAS_DOWN': 83, 'BIAS_NEUTRAL': 2888}  ambiguous=1761 (total=3006) horizon=12
2026-05-03 08:35:55,478 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected EURUSD — 2956 samples (group=dollar) score_means={'bias_up_score': 0.0118, 'bias_down_score': 0.0281} labels={'BIAS_UP': 35, 'BIAS_DOWN': 83, 'BIAS_NEUTRAL': 2838} clean={'BIAS_UP': 35, 'BIAS_DOWN': 83, 'BIAS_NEUTRAL': 1110}
2026-05-03 08:35:55,580 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:55,582 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:55,583 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:55,583 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:55,583 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:55,584 INFO Loaded GBPJPY/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:55,787 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 76, 'BIAS_DOWN': 93, 'BIAS_NEUTRAL': 2838}  ambiguous=1772 (total=3007) horizon=12
2026-05-03 08:35:55,790 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected GBPJPY — 2957 samples (group=cross) score_means={'bias_up_score': 0.0257, 'bias_down_score': 0.0315} labels={'BIAS_UP': 76, 'BIAS_DOWN': 93, 'BIAS_NEUTRAL': 2788} clean={'BIAS_UP': 76, 'BIAS_DOWN': 93, 'BIAS_NEUTRAL': 1056}
2026-05-03 08:35:55,893 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:55,896 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:55,896 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:55,897 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:55,897 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:55,898 INFO Loaded GBPUSD/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:56,106 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 84, 'BIAS_DOWN': 126, 'BIAS_NEUTRAL': 2797}  ambiguous=1784 (total=3007) horizon=12
2026-05-03 08:35:56,109 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected GBPUSD — 2957 samples (group=dollar) score_means={'bias_up_score': 0.0284, 'bias_down_score': 0.0426} labels={'BIAS_UP': 84, 'BIAS_DOWN': 126, 'BIAS_NEUTRAL': 2747} clean={'BIAS_UP': 84, 'BIAS_DOWN': 126, 'BIAS_NEUTRAL': 992}
2026-05-03 08:35:56,211 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:56,212 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:56,213 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:56,213 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:56,214 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:56,215 INFO Loaded NZDUSD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:56,419 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 61, 'BIAS_DOWN': 121, 'BIAS_NEUTRAL': 2824}  ambiguous=1784 (total=3006) horizon=12
2026-05-03 08:35:56,422 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected NZDUSD — 2956 samples (group=dollar) score_means={'bias_up_score': 0.0206, 'bias_down_score': 0.0409} labels={'BIAS_UP': 61, 'BIAS_DOWN': 121, 'BIAS_NEUTRAL': 2774} clean={'BIAS_UP': 61, 'BIAS_DOWN': 121, 'BIAS_NEUTRAL': 1015}
2026-05-03 08:35:56,522 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:56,524 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:56,525 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:56,525 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:56,525 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:56,526 INFO Loaded USDCAD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:56,740 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 56, 'BIAS_DOWN': 44, 'BIAS_NEUTRAL': 2906}  ambiguous=1797 (total=3006) horizon=12
2026-05-03 08:35:56,742 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected USDCAD — 2956 samples (group=dollar) score_means={'bias_up_score': 0.0189, 'bias_down_score': 0.0149} labels={'BIAS_UP': 56, 'BIAS_DOWN': 44, 'BIAS_NEUTRAL': 2856} clean={'BIAS_UP': 56, 'BIAS_DOWN': 44, 'BIAS_NEUTRAL': 1093}
2026-05-03 08:35:56,845 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:56,848 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:56,849 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:56,849 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:56,850 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:56,851 INFO Loaded USDCHF/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:57,053 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 111, 'BIAS_DOWN': 61, 'BIAS_NEUTRAL': 2834}  ambiguous=1701 (total=3006) horizon=12
2026-05-03 08:35:57,056 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected USDCHF — 2956 samples (group=dollar) score_means={'bias_up_score': 0.0376, 'bias_down_score': 0.0206} labels={'BIAS_UP': 111, 'BIAS_DOWN': 61, 'BIAS_NEUTRAL': 2784} clean={'BIAS_UP': 111, 'BIAS_DOWN': 61, 'BIAS_NEUTRAL': 1118}
2026-05-03 08:35:57,159 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:57,163 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:57,164 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:57,164 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:57,164 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:57,165 INFO Loaded USDJPY/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:57,371 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 78, 'BIAS_DOWN': 75, 'BIAS_NEUTRAL': 2854}  ambiguous=1708 (total=3007) horizon=12
2026-05-03 08:35:57,374 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected USDJPY — 2957 samples (group=dollar) score_means={'bias_up_score': 0.0264, 'bias_down_score': 0.0254} labels={'BIAS_UP': 78, 'BIAS_DOWN': 75, 'BIAS_NEUTRAL': 2804} clean={'BIAS_UP': 78, 'BIAS_DOWN': 75, 'BIAS_NEUTRAL': 1126}
2026-05-03 08:35:57,485 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:57,489 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:57,490 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:57,491 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:57,491 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:35:57,492 INFO Loaded XAUUSD/4H split=train fold=fold_001: 3193 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:57,711 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 91, 'BIAS_DOWN': 40, 'BIAS_NEUTRAL': 3062}  ambiguous=1810 (total=3193) horizon=12
2026-05-03 08:35:57,714 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected XAUUSD — 3143 samples (group=gold) score_means={'bias_up_score': 0.029, 'bias_down_score': 0.0127} labels={'BIAS_UP': 91, 'BIAS_DOWN': 40, 'BIAS_NEUTRAL': 3012} clean={'BIAS_UP': 91, 'BIAS_DOWN': 40, 'BIAS_NEUTRAL': 1233}
2026-05-03 08:35:57,812 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 170, 'BIAS_DOWN': 237, 'BIAS_NEUTRAL': 8462}, 'dollar': {'BIAS_UP': 459, 'BIAS_DOWN': 615, 'BIAS_NEUTRAL': 19620}, 'gold': {'BIAS_UP': 91, 'BIAS_DOWN': 40, 'BIAS_NEUTRAL': 3012}}
2026-05-03 08:35:57,812 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0192, 'bias_down_score': 0.0267}, 'dollar': {'bias_up_score': 0.0222, 'bias_down_score': 0.0297}, 'gold': {'bias_up_score': 0.029, 'bias_down_score': 0.0127}}
2026-05-03 08:35:57,812 INFO Regime[4H mode=htf_bias] label distribution by year: {2018: {'BIAS_UP': 354, 'BIAS_DOWN': 523, 'BIAS_NEUTRAL': 15079}, 2019: {'BIAS_UP': 365, 'BIAS_DOWN': 368, 'BIAS_NEUTRAL': 15874}, 2020: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 141}}
2026-05-03 08:35:57,812 INFO Regime[4H mode=htf_bias] score means by year: {2018: {'bias_up_score': 0.0222, 'bias_down_score': 0.0328}, 2019: {'bias_up_score': 0.022, 'bias_down_score': 0.0222}, 2020: {'bias_up_score': 0.007, 'bias_down_score': 0.007}}
2026-05-03 08:35:57,892 INFO Loaded AUDUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-03 08:35:57,893 INFO Loaded EURGBP/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-03 08:35:57,894 INFO Loaded EURJPY/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-03 08:35:57,895 INFO Loaded EURUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-03 08:35:57,896 INFO Loaded GBPJPY/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-03 08:35:57,897 INFO Loaded GBPUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-03 08:35:57,897 INFO Loaded NZDUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-03 08:35:57,898 INFO Loaded USDCAD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-03 08:35:57,899 INFO Loaded USDCHF/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-03 08:35:57,900 INFO Loaded USDJPY/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-03 08:35:57,901 INFO Loaded XAUUSD/4H split=val fold=fold_001: 1581 bars (2020-01-06 → 2020-12-31)
2026-05-03 08:35:57,906 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:57,909 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:57,909 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:57,910 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:57,910 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:57,911 INFO Loaded AUDUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:58,105 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 26, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1411}  ambiguous=832 (total=1490) horizon=12
2026-05-03 08:35:58,107 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected AUDUSD — 1440 samples (group=dollar) score_means={'bias_up_score': 0.0181, 'bias_down_score': 0.0368} labels={'BIAS_UP': 26, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1361} clean={'BIAS_UP': 26, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 550}
2026-05-03 08:35:58,215 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:58,219 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:58,220 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:58,220 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:58,221 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:58,222 INFO Loaded EURGBP/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:58,405 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 62, 'BIAS_DOWN': 3, 'BIAS_NEUTRAL': 1425}  ambiguous=865 (total=1490) horizon=12
2026-05-03 08:35:58,407 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected EURGBP — 1440 samples (group=cross) score_means={'bias_up_score': 0.0431, 'bias_down_score': 0.0021} labels={'BIAS_UP': 62, 'BIAS_DOWN': 3, 'BIAS_NEUTRAL': 1375} clean={'BIAS_UP': 62, 'BIAS_DOWN': 3, 'BIAS_NEUTRAL': 536}
2026-05-03 08:35:58,508 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:58,512 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:58,513 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:58,513 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:58,513 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:58,514 INFO Loaded EURJPY/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:58,700 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 42, 'BIAS_DOWN': 12, 'BIAS_NEUTRAL': 1436}  ambiguous=928 (total=1490) horizon=12
2026-05-03 08:35:58,703 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected EURJPY — 1440 samples (group=cross) score_means={'bias_up_score': 0.0292, 'bias_down_score': 0.0083} labels={'BIAS_UP': 42, 'BIAS_DOWN': 12, 'BIAS_NEUTRAL': 1386} clean={'BIAS_UP': 42, 'BIAS_DOWN': 12, 'BIAS_NEUTRAL': 491}
2026-05-03 08:35:58,803 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:58,806 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:58,806 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:58,807 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:58,807 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:58,808 INFO Loaded EURUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:58,992 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 89, 'BIAS_DOWN': 33, 'BIAS_NEUTRAL': 1368}  ambiguous=880 (total=1490) horizon=12
2026-05-03 08:35:58,994 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected EURUSD — 1440 samples (group=dollar) score_means={'bias_up_score': 0.0618, 'bias_down_score': 0.0229} labels={'BIAS_UP': 89, 'BIAS_DOWN': 33, 'BIAS_NEUTRAL': 1318} clean={'BIAS_UP': 89, 'BIAS_DOWN': 33, 'BIAS_NEUTRAL': 462}
2026-05-03 08:35:59,095 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:59,098 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:59,098 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:59,099 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:59,099 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:59,100 INFO Loaded GBPJPY/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:59,290 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 26, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 1418}  ambiguous=910 (total=1490) horizon=12
2026-05-03 08:35:59,293 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected GBPJPY — 1440 samples (group=cross) score_means={'bias_up_score': 0.0181, 'bias_down_score': 0.0319} labels={'BIAS_UP': 26, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 1368} clean={'BIAS_UP': 26, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 476}
2026-05-03 08:35:59,393 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:59,395 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:59,396 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:59,396 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:59,397 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:59,398 INFO Loaded GBPUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:59,584 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 39, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 1429}  ambiguous=909 (total=1490) horizon=12
2026-05-03 08:35:59,586 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected GBPUSD — 1440 samples (group=dollar) score_means={'bias_up_score': 0.0271, 'bias_down_score': 0.0153} labels={'BIAS_UP': 39, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 1379} clean={'BIAS_UP': 39, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 493}
2026-05-03 08:35:59,685 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:59,687 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:59,688 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:59,688 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:59,688 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:35:59,690 INFO Loaded NZDUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:35:59,873 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 47, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1402}  ambiguous=817 (total=1490) horizon=12
2026-05-03 08:35:59,876 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected NZDUSD — 1440 samples (group=dollar) score_means={'bias_up_score': 0.0326, 'bias_down_score': 0.0285} labels={'BIAS_UP': 47, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1352} clean={'BIAS_UP': 47, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 561}
2026-05-03 08:35:59,988 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:59,991 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:59,992 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:59,992 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:59,992 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:35:59,993 INFO Loaded USDCAD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:00,185 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 61, 'BIAS_DOWN': 59, 'BIAS_NEUTRAL': 1370}  ambiguous=800 (total=1490) horizon=12
2026-05-03 08:36:00,187 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected USDCAD — 1440 samples (group=dollar) score_means={'bias_up_score': 0.0424, 'bias_down_score': 0.041} labels={'BIAS_UP': 61, 'BIAS_DOWN': 59, 'BIAS_NEUTRAL': 1320} clean={'BIAS_UP': 61, 'BIAS_DOWN': 59, 'BIAS_NEUTRAL': 551}
2026-05-03 08:36:00,312 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:00,316 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:00,317 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:00,317 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:00,318 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:00,319 INFO Loaded USDCHF/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:00,508 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 11, 'BIAS_DOWN': 76, 'BIAS_NEUTRAL': 1403}  ambiguous=838 (total=1490) horizon=12
2026-05-03 08:36:00,511 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected USDCHF — 1440 samples (group=dollar) score_means={'bias_up_score': 0.0076, 'bias_down_score': 0.0528} labels={'BIAS_UP': 11, 'BIAS_DOWN': 76, 'BIAS_NEUTRAL': 1353} clean={'BIAS_UP': 11, 'BIAS_DOWN': 76, 'BIAS_NEUTRAL': 539}
2026-05-03 08:36:00,614 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:00,618 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:00,618 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:00,619 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:00,619 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:00,620 INFO Loaded USDJPY/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:00,806 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 6, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1431}  ambiguous=877 (total=1490) horizon=12
2026-05-03 08:36:00,808 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected USDJPY — 1440 samples (group=dollar) score_means={'bias_up_score': 0.0042, 'bias_down_score': 0.0368} labels={'BIAS_UP': 6, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1381} clean={'BIAS_UP': 6, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 537}
2026-05-03 08:36:00,922 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:36:00,925 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:36:00,927 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:36:00,927 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:36:00,927 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:36:00,929 INFO Loaded XAUUSD/4H split=val fold=fold_001: 1581 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:01,143 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 76, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 1497}  ambiguous=916 (total=1581) horizon=12
2026-05-03 08:36:01,145 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected XAUUSD — 1531 samples (group=gold) score_means={'bias_up_score': 0.0496, 'bias_down_score': 0.0052} labels={'BIAS_UP': 76, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 1447} clean={'BIAS_UP': 76, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 553}
2026-05-03 08:36:01,242 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 130, 'BIAS_DOWN': 61, 'BIAS_NEUTRAL': 4129}, 'dollar': {'BIAS_UP': 279, 'BIAS_DOWN': 337, 'BIAS_NEUTRAL': 9464}, 'gold': {'BIAS_UP': 76, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 1447}}
2026-05-03 08:36:01,242 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0301, 'bias_down_score': 0.0141}, 'dollar': {'bias_up_score': 0.0277, 'bias_down_score': 0.0334}, 'gold': {'bias_up_score': 0.0496, 'bias_down_score': 0.0052}}
2026-05-03 08:36:01,242 INFO Regime[4H mode=htf_bias] label distribution by year: {2020: {'BIAS_UP': 485, 'BIAS_DOWN': 406, 'BIAS_NEUTRAL': 15040}}
2026-05-03 08:36:01,243 INFO Regime[4H mode=htf_bias] score means by year: {2020: {'bias_up_score': 0.0304, 'bias_down_score': 0.0255}}
2026-05-03 08:36:01,320 INFO Regime phase HTF dataset build fold=fold_001: 7.0s (train=32706 val=15931)
2026-05-03 08:36:01,324 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=24, n_classes=2)
2026-05-03 08:36:01,324 INFO Regime 4H/htf_bias warm start enabled from existing weights
2026-05-03 08:36:01,329 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=32706 val=15931 train_labels={'BIAS_UP': 720, 'BIAS_DOWN': 892, 'BIAS_NEUTRAL': 31094} val_labels={'BIAS_UP': 485, 'BIAS_DOWN': 406, 'BIAS_NEUTRAL': 15040}
2026-05-03 08:36:01,329 INFO RegimeClassifier[mode=htf_bias]: warm start HTF score head
2026-05-03 08:36:01,329 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-03 08:36:01,329 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 20.0, 'bias_down_score': 20.0}
2026-05-03 08:36:02,313 INFO Regime HTF score epoch  1/50 — tr=0.3893 va=0.4088 acc=0.869 bal=0.911 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.934, 'BIAS_DOWN': 0.933, 'BIAS_NEUTRAL': 0.866} precision={'BIAS_UP': 0.299, 'BIAS_DOWN': 0.283, 'BIAS_NEUTRAL': 0.995}
2026-05-03 08:36:03,241 INFO Regime HTF score epoch  2/50 — tr=0.3923 va=0.4142 bal=0.918
2026-05-03 08:36:04,196 INFO Regime HTF score epoch  3/50 — tr=0.3862 va=0.4157 bal=0.918
2026-05-03 08:36:05,130 INFO Regime HTF score epoch  4/50 — tr=0.3813 va=0.4156 bal=0.919
2026-05-03 08:36:06,063 INFO Regime HTF score epoch  5/50 — tr=0.3865 va=0.4134 acc=0.868 bal=0.919 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.963, 'BIAS_DOWN': 0.931, 'BIAS_NEUTRAL': 0.863} precision={'BIAS_UP': 0.295, 'BIAS_DOWN': 0.287, 'BIAS_NEUTRAL': 0.996}
2026-05-03 08:36:07,008 INFO Regime HTF score epoch  6/50 — tr=0.3805 va=0.4115 bal=0.918
2026-05-03 08:36:07,968 INFO Regime HTF score epoch  7/50 — tr=0.3849 va=0.4115 bal=0.918
2026-05-03 08:36:08,904 INFO Regime HTF score epoch  8/50 — tr=0.3796 va=0.4095 bal=0.918
2026-05-03 08:36:09,891 INFO Regime HTF score epoch  9/50 — tr=0.3789 va=0.4099 bal=0.918
2026-05-03 08:36:10,882 INFO Regime HTF score epoch 10/50 — tr=0.3816 va=0.4075 acc=0.868 bal=0.918 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.963, 'BIAS_DOWN': 0.929, 'BIAS_NEUTRAL': 0.863} precision={'BIAS_UP': 0.294, 'BIAS_DOWN': 0.287, 'BIAS_NEUTRAL': 0.996}
2026-05-03 08:36:11,844 INFO Regime HTF score epoch 11/50 — tr=0.3758 va=0.4052 bal=0.918
2026-05-03 08:36:12,789 INFO Regime HTF score epoch 12/50 — tr=0.3709 va=0.4018 bal=0.917
2026-05-03 08:36:13,739 INFO Regime HTF score epoch 13/50 — tr=0.3748 va=0.3993 bal=0.916
2026-05-03 08:36:14,681 INFO Regime HTF score epoch 14/50 — tr=0.3732 va=0.3999 bal=0.919
2026-05-03 08:36:15,627 INFO Regime HTF score epoch 15/50 — tr=0.3705 va=0.4002 acc=0.869 bal=0.920 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.965, 'BIAS_DOWN': 0.931, 'BIAS_NEUTRAL': 0.864} precision={'BIAS_UP': 0.295, 'BIAS_DOWN': 0.289, 'BIAS_NEUTRAL': 0.997}
2026-05-03 08:36:16,586 INFO Regime HTF score epoch 16/50 — tr=0.3579 va=0.3988 bal=0.919
2026-05-03 08:36:17,551 INFO Regime HTF score epoch 17/50 — tr=0.3636 va=0.3979 bal=0.920
2026-05-03 08:36:18,514 INFO Regime HTF score epoch 18/50 — tr=0.3597 va=0.3935 bal=0.918
2026-05-03 08:36:19,447 INFO Regime HTF score epoch 19/50 — tr=0.3602 va=0.3906 bal=0.914
2026-05-03 08:36:20,433 INFO Regime HTF score epoch 20/50 — tr=0.3621 va=0.3922 acc=0.870 bal=0.918 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.961, 'BIAS_DOWN': 0.929, 'BIAS_NEUTRAL': 0.865} precision={'BIAS_UP': 0.297, 'BIAS_DOWN': 0.29, 'BIAS_NEUTRAL': 0.996}
2026-05-03 08:36:21,384 INFO Regime HTF score epoch 21/50 — tr=0.3614 va=0.3941 bal=0.922
2026-05-03 08:36:22,314 INFO Regime HTF score epoch 22/50 — tr=0.3579 va=0.3935 bal=0.922
2026-05-03 08:36:23,270 INFO Regime HTF score epoch 23/50 — tr=0.3515 va=0.3915 bal=0.922
2026-05-03 08:36:24,216 INFO Regime HTF score epoch 24/50 — tr=0.3560 va=0.3889 bal=0.919
2026-05-03 08:36:25,161 INFO Regime HTF score epoch 25/50 — tr=0.3473 va=0.3872 acc=0.870 bal=0.919 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.963, 'BIAS_DOWN': 0.929, 'BIAS_NEUTRAL': 0.865} precision={'BIAS_UP': 0.297, 'BIAS_DOWN': 0.291, 'BIAS_NEUTRAL': 0.996}
2026-05-03 08:36:26,150 INFO Regime HTF score epoch 26/50 — tr=0.3474 va=0.3840 bal=0.916
2026-05-03 08:36:27,103 INFO Regime HTF score epoch 27/50 — tr=0.3533 va=0.3824 bal=0.916
2026-05-03 08:36:28,041 INFO Regime HTF score epoch 28/50 — tr=0.3486 va=0.3821 bal=0.916
2026-05-03 08:36:28,994 INFO Regime HTF score epoch 29/50 — tr=0.3452 va=0.3810 bal=0.915
2026-05-03 08:36:29,966 INFO Regime HTF score epoch 30/50 — tr=0.3488 va=0.3787 acc=0.872 bal=0.913 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.946, 'BIAS_DOWN': 0.924, 'BIAS_NEUTRAL': 0.868} precision={'BIAS_UP': 0.299, 'BIAS_DOWN': 0.294, 'BIAS_NEUTRAL': 0.996}
2026-05-03 08:36:30,946 INFO Regime HTF score epoch 31/50 — tr=0.3460 va=0.3825 bal=0.917
2026-05-03 08:36:30,946 INFO Regime HTF score early stop at epoch 31
2026-05-03 08:36:31,837 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.850 margin=0.000 precision={'BIAS_UP': 0.297, 'BIAS_DOWN': 0.291, 'BIAS_NEUTRAL': 0.997} recall={'BIAS_UP': 0.963, 'BIAS_DOWN': 0.938, 'BIAS_NEUTRAL': 0.865} f1={'BIAS_UP': 0.453, 'BIAS_DOWN': 0.444, 'BIAS_NEUTRAL': 0.926} confusion=[[467, 0, 18], [0, 381, 25], [1108, 928, 13004]] score_mae={'bias_up_score': 0.1701, 'bias_down_score': 0.1432} pred_share={'BIAS_UP': 0.0989, 'BIAS_DOWN': 0.0822, 'BIAS_NEUTRAL': 0.819}
2026-05-03 08:36:31,838 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.297, 'BIAS_DOWN': 0.291, 'BIAS_NEUTRAL': 0.997} min_precision=0.300 recall={'BIAS_UP': 0.963, 'BIAS_DOWN': 0.938, 'BIAS_NEUTRAL': 0.865} min_recall=0.100 f1={'BIAS_UP': 0.453, 'BIAS_DOWN': 0.444, 'BIAS_NEUTRAL': 0.926} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_UP', 'BIAS_DOWN'] weak_recall=[] weak_f1=[] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-03 08:36:31,841 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-03 08:36:31,841 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-03 08:36:31,841 INFO Regime phase HTF train fold=fold_001: 30.5s
2026-05-03 08:36:31,940 INFO Regime HTF complete fold=fold_001: acc=0.870 bal=0.922 train=32706 val=15931 per_class={'BIAS_UP': 0.963, 'BIAS_DOWN': 0.938, 'BIAS_NEUTRAL': 0.865} precision={'BIAS_UP': 0.297, 'BIAS_DOWN': 0.291, 'BIAS_NEUTRAL': 0.997} threshold=0.850 margin=0.000
2026-05-03 08:36:31,941 INFO Loaded XAUUSD/4H split=train fold=fold_001: 3193 bars (2018-01-04 → 2020-01-03)
2026-05-03 08:36:32,034 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 91, 'BIAS_DOWN': 40, 'BIAS_NEUTRAL': 3062}  ambiguous=1810 (total=3193) horizon=12
2026-05-03 08:36:32,036 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 4.136363636363637, 'BIAS_DOWN': 3.076923076923077, 'BIAS_NEUTRAL': 85.05555555555556}
2026-05-03 08:36:32,040 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 91, 'mean': 0.0012718413994960778, 'mean_over_std': 0.5225173433708362}, 'BIAS_DOWN': {'n': 40, 'mean': -0.0007135081911815277, 'mean_over_std': -0.3923435533984593}, 'BIAS_NEUTRAL': {'n': 3061, 'mean': 3.036271341687858e-05, 'mean_over_std': 0.010967519850120955}}
2026-05-03 08:36:32,040 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 91, 'mean': 0.0012718413994960778, 'mean_over_std': 0.5225173433708362}, 'BIAS_DOWN': {'n': 40, 'mean': -0.0007135081911815277, 'mean_over_std': -0.3923435533984593}, 'BIAS_NEUTRAL': {'n': 1252, 'mean': 3.51836161561549e-05, 'mean_over_std': 0.016057793936327037}}
2026-05-03 08:36:32,043 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-03 08:36:32,045 INFO Loaded AUDUSD/1H split=train fold=fold_001: 11705 bars (2018-01-04 → 2020-01-03)
2026-05-03 08:36:32,046 INFO Loaded EURGBP/1H split=train fold=fold_001: 11707 bars (2018-01-04 → 2020-01-03)
2026-05-03 08:36:32,047 INFO Loaded EURJPY/1H split=train fold=fold_001: 11708 bars (2018-01-04 → 2020-01-03)
2026-05-03 08:36:32,049 INFO Loaded EURUSD/1H split=train fold=fold_001: 11707 bars (2018-01-04 → 2020-01-03)
2026-05-03 08:36:32,050 INFO Loaded GBPJPY/1H split=train fold=fold_001: 11708 bars (2018-01-04 → 2020-01-03)
2026-05-03 08:36:32,051 INFO Loaded GBPUSD/1H split=train fold=fold_001: 11708 bars (2018-01-04 → 2020-01-03)
2026-05-03 08:36:32,052 INFO Loaded NZDUSD/1H split=train fold=fold_001: 11707 bars (2018-01-04 → 2020-01-03)
2026-05-03 08:36:32,054 INFO Loaded USDCAD/1H split=train fold=fold_001: 11707 bars (2018-01-04 → 2020-01-03)
2026-05-03 08:36:32,055 INFO Loaded USDCHF/1H split=train fold=fold_001: 11709 bars (2018-01-04 → 2020-01-03)
2026-05-03 08:36:32,056 INFO Loaded USDJPY/1H split=train fold=fold_001: 11711 bars (2018-01-04 → 2020-01-03)
2026-05-03 08:36:32,058 INFO Loaded XAUUSD/1H split=train fold=fold_001: 11927 bars (2018-01-04 → 2020-01-03)
2026-05-03 08:36:32,063 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:32,065 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:32,066 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:32,066 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:32,066 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:32,068 INFO Loaded AUDUSD/1H split=train fold=fold_001: 11705 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:32,387 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected AUDUSD — 11655 samples (group=dollar) score_means={'trend_score': 0.4956, 'range_score': 0.2359, 'chop_score': 0.4612, 'volatility_percentile': 0.3808, 'consolidation_score': 0.1864}
2026-05-03 08:36:32,493 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:32,497 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:32,499 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:32,500 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:32,500 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:32,502 INFO Loaded EURGBP/1H split=train fold=fold_001: 11707 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:32,825 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected EURGBP — 11657 samples (group=cross) score_means={'trend_score': 0.4573, 'range_score': 0.2481, 'chop_score': 0.4857, 'volatility_percentile': 0.3934, 'consolidation_score': 0.1799}
2026-05-03 08:36:32,927 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:32,931 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:32,932 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:32,932 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:32,933 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:32,934 INFO Loaded EURJPY/1H split=train fold=fold_001: 11708 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:33,244 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected EURJPY — 11658 samples (group=cross) score_means={'trend_score': 0.4805, 'range_score': 0.2366, 'chop_score': 0.4707, 'volatility_percentile': 0.3744, 'consolidation_score': 0.1928}
2026-05-03 08:36:33,347 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:33,350 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:33,352 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:33,352 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:33,353 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:33,354 INFO Loaded EURUSD/1H split=train fold=fold_001: 11707 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:33,669 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected EURUSD — 11657 samples (group=dollar) score_means={'trend_score': 0.4875, 'range_score': 0.2372, 'chop_score': 0.4621, 'volatility_percentile': 0.378, 'consolidation_score': 0.1876}
2026-05-03 08:36:33,772 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:33,774 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:33,775 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:33,775 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:33,775 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:33,777 INFO Loaded GBPJPY/1H split=train fold=fold_001: 11708 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:34,097 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected GBPJPY — 11658 samples (group=cross) score_means={'trend_score': 0.4891, 'range_score': 0.2383, 'chop_score': 0.4697, 'volatility_percentile': 0.39, 'consolidation_score': 0.184}
2026-05-03 08:36:34,199 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:34,203 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:34,204 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:34,204 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:34,204 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:34,206 INFO Loaded GBPUSD/1H split=train fold=fold_001: 11708 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:34,523 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected GBPUSD — 11658 samples (group=dollar) score_means={'trend_score': 0.4959, 'range_score': 0.2313, 'chop_score': 0.4576, 'volatility_percentile': 0.3919, 'consolidation_score': 0.1801}
2026-05-03 08:36:34,624 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:36:34,625 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:36:34,626 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:36:34,626 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:36:34,627 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:36:34,628 INFO Loaded NZDUSD/1H split=train fold=fold_001: 11707 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:34,948 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected NZDUSD — 11657 samples (group=dollar) score_means={'trend_score': 0.4999, 'range_score': 0.2321, 'chop_score': 0.4537, 'volatility_percentile': 0.3824, 'consolidation_score': 0.1829}
2026-05-03 08:36:35,051 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:35,053 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:35,054 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:35,054 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:35,054 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:35,056 INFO Loaded USDCAD/1H split=train fold=fold_001: 11707 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:35,371 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected USDCAD — 11657 samples (group=dollar) score_means={'trend_score': 0.4841, 'range_score': 0.2445, 'chop_score': 0.47, 'volatility_percentile': 0.3763, 'consolidation_score': 0.186}
2026-05-03 08:36:35,475 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:35,477 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:35,478 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:35,478 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:35,478 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:35,480 INFO Loaded USDCHF/1H split=train fold=fold_001: 11709 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:35,804 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected USDCHF — 11659 samples (group=dollar) score_means={'trend_score': 0.4748, 'range_score': 0.2414, 'chop_score': 0.47, 'volatility_percentile': 0.3832, 'consolidation_score': 0.1861}
2026-05-03 08:36:35,907 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:35,911 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:35,912 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:35,912 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:35,913 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:35,915 INFO Loaded USDJPY/1H split=train fold=fold_001: 11711 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:36,247 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected USDJPY — 11661 samples (group=dollar) score_means={'trend_score': 0.4818, 'range_score': 0.237, 'chop_score': 0.4712, 'volatility_percentile': 0.3725, 'consolidation_score': 0.1975}
2026-05-03 08:36:36,360 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:36:36,363 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:36:36,365 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:36:36,365 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:36:36,365 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:36:36,368 INFO Loaded XAUUSD/1H split=train fold=fold_001: 11927 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:36,704 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected XAUUSD — 11877 samples (group=gold) score_means={'trend_score': 0.475, 'range_score': 0.244, 'chop_score': 0.473, 'volatility_percentile': 0.3803, 'consolidation_score': 0.1878}
2026-05-03 08:36:36,805 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4756, 'range_score': 0.241, 'chop_score': 0.4754, 'volatility_percentile': 0.3859, 'consolidation_score': 0.1855}, 'dollar': {'trend_score': 0.4885, 'range_score': 0.237, 'chop_score': 0.4637, 'volatility_percentile': 0.3808, 'consolidation_score': 0.1867}, 'gold': {'trend_score': 0.475, 'range_score': 0.244, 'chop_score': 0.473, 'volatility_percentile': 0.3803, 'consolidation_score': 0.1878}}
2026-05-03 08:36:36,805 INFO Regime[1H mode=ltf_behaviour] score means by year: {2018: {'trend_score': 0.4841, 'range_score': 0.2416, 'chop_score': 0.4687, 'volatility_percentile': 0.3892, 'consolidation_score': 0.1792}, 2019: {'trend_score': 0.4829, 'range_score': 0.2363, 'chop_score': 0.4672, 'volatility_percentile': 0.3736, 'consolidation_score': 0.1949}, 2020: {'trend_score': 0.5462, 'range_score': 0.1954, 'chop_score': 0.4095, 'volatility_percentile': 0.5807, 'consolidation_score': 0.0319}}
2026-05-03 08:36:36,884 INFO Loaded AUDUSD/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
2026-05-03 08:36:36,885 INFO Loaded EURGBP/1H split=val fold=fold_001: 5814 bars (2020-01-06 → 2020-12-31)
2026-05-03 08:36:36,887 INFO Loaded EURJPY/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
2026-05-03 08:36:36,888 INFO Loaded EURUSD/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
2026-05-03 08:36:36,889 INFO Loaded GBPJPY/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
2026-05-03 08:36:36,890 INFO Loaded GBPUSD/1H split=val fold=fold_001: 5814 bars (2020-01-06 → 2020-12-31)
2026-05-03 08:36:36,891 INFO Loaded NZDUSD/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
2026-05-03 08:36:36,892 INFO Loaded USDCAD/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
2026-05-03 08:36:36,894 INFO Loaded USDCHF/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
2026-05-03 08:36:36,895 INFO Loaded USDJPY/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
2026-05-03 08:36:36,897 INFO Loaded XAUUSD/1H split=val fold=fold_001: 5855 bars (2020-01-06 → 2020-12-31)
2026-05-03 08:36:36,902 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:36,905 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:36,905 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:36,906 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:36,906 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:36,907 INFO Loaded AUDUSD/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:37,139 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected AUDUSD — 5765 samples (group=dollar) score_means={'trend_score': 0.4839, 'range_score': 0.2326, 'chop_score': 0.469, 'volatility_percentile': 0.3782, 'consolidation_score': 0.1933}
2026-05-03 08:36:37,243 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:37,246 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:37,246 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:37,247 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:37,247 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:37,249 INFO Loaded EURGBP/1H split=val fold=fold_001: 5814 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:37,483 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected EURGBP — 5764 samples (group=cross) score_means={'trend_score': 0.4874, 'range_score': 0.2329, 'chop_score': 0.4636, 'volatility_percentile': 0.3931, 'consolidation_score': 0.1787}
2026-05-03 08:36:37,587 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:37,589 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:37,590 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:37,590 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:37,590 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:37,592 INFO Loaded EURJPY/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:37,829 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected EURJPY — 5765 samples (group=cross) score_means={'trend_score': 0.4913, 'range_score': 0.2317, 'chop_score': 0.469, 'volatility_percentile': 0.3781, 'consolidation_score': 0.1938}
2026-05-03 08:36:37,931 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:37,933 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:37,934 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:37,934 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:37,935 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:37,936 INFO Loaded EURUSD/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:38,175 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected EURUSD — 5765 samples (group=dollar) score_means={'trend_score': 0.4992, 'range_score': 0.2236, 'chop_score': 0.4521, 'volatility_percentile': 0.3866, 'consolidation_score': 0.1844}
2026-05-03 08:36:38,282 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:38,284 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:38,285 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:38,285 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:38,285 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:38,287 INFO Loaded GBPJPY/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:38,521 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected GBPJPY — 5765 samples (group=cross) score_means={'trend_score': 0.4675, 'range_score': 0.2397, 'chop_score': 0.484, 'volatility_percentile': 0.3868, 'consolidation_score': 0.1954}
2026-05-03 08:36:38,623 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:38,625 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:38,626 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:38,626 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:38,626 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:38,628 INFO Loaded GBPUSD/1H split=val fold=fold_001: 5814 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:38,866 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected GBPUSD — 5764 samples (group=dollar) score_means={'trend_score': 0.4894, 'range_score': 0.2257, 'chop_score': 0.4574, 'volatility_percentile': 0.3874, 'consolidation_score': 0.1784}
2026-05-03 08:36:38,969 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:36:38,971 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:36:38,971 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:36:38,972 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:36:38,972 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:36:38,974 INFO Loaded NZDUSD/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:39,208 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected NZDUSD — 5765 samples (group=dollar) score_means={'trend_score': 0.4801, 'range_score': 0.2329, 'chop_score': 0.4674, 'volatility_percentile': 0.3741, 'consolidation_score': 0.1922}
2026-05-03 08:36:39,312 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:39,314 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:39,315 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:39,315 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:39,316 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:39,317 INFO Loaded USDCAD/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:39,556 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected USDCAD — 5765 samples (group=dollar) score_means={'trend_score': 0.4734, 'range_score': 0.2423, 'chop_score': 0.4776, 'volatility_percentile': 0.3777, 'consolidation_score': 0.1886}
2026-05-03 08:36:39,658 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:39,661 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:39,662 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:39,663 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:39,663 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:39,664 INFO Loaded USDCHF/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:39,912 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected USDCHF — 5765 samples (group=dollar) score_means={'trend_score': 0.4788, 'range_score': 0.2389, 'chop_score': 0.4664, 'volatility_percentile': 0.3821, 'consolidation_score': 0.1903}
2026-05-03 08:36:40,020 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:40,022 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:40,023 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:40,023 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:40,024 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:40,025 INFO Loaded USDJPY/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:40,275 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected USDJPY — 5765 samples (group=dollar) score_means={'trend_score': 0.4839, 'range_score': 0.2352, 'chop_score': 0.4713, 'volatility_percentile': 0.3706, 'consolidation_score': 0.1995}
2026-05-03 08:36:40,397 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:36:40,400 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:36:40,401 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:36:40,402 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:36:40,402 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:36:40,404 INFO Loaded XAUUSD/1H split=val fold=fold_001: 5855 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:40,652 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected XAUUSD — 5805 samples (group=gold) score_means={'trend_score': 0.4836, 'range_score': 0.2372, 'chop_score': 0.4777, 'volatility_percentile': 0.3611, 'consolidation_score': 0.2086}
2026-05-03 08:36:40,749 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4821, 'range_score': 0.2348, 'chop_score': 0.4722, 'volatility_percentile': 0.386, 'consolidation_score': 0.1893}, 'dollar': {'trend_score': 0.4841, 'range_score': 0.233, 'chop_score': 0.4659, 'volatility_percentile': 0.3795, 'consolidation_score': 0.1895}, 'gold': {'trend_score': 0.4836, 'range_score': 0.2372, 'chop_score': 0.4777, 'volatility_percentile': 0.3611, 'consolidation_score': 0.2086}}
2026-05-03 08:36:40,749 INFO Regime[1H mode=ltf_behaviour] score means by year: {2020: {'trend_score': 0.4835, 'range_score': 0.2339, 'chop_score': 0.4687, 'volatility_percentile': 0.3796, 'consolidation_score': 0.1912}}
2026-05-03 08:36:40,825 INFO Regime phase LTF dataset build fold=fold_001: 8.8s (train=128454 val=63453)
2026-05-03 08:36:40,830 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=31, n_classes=5)
2026-05-03 08:36:40,830 INFO Regime 1H/ltf_behaviour warm start enabled from existing weights
2026-05-03 08:36:40,847 INFO RegimeClassifier[mode=ltf_behaviour]: warm start score head
2026-05-03 08:36:40,847 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-03 08:36:41,123 INFO Regime score epoch  1/50 — tr=0.0038 va=0.0011 mae={'trend_score': 0.0231, 'range_score': 0.0352, 'chop_score': 0.0228, 'volatility_percentile': 0.0172, 'consolidation_score': 0.0244}
2026-05-03 08:36:41,378 INFO Regime score epoch  2/50 — tr=0.0038 va=0.0011
2026-05-03 08:36:41,639 INFO Regime score epoch  3/50 — tr=0.0038 va=0.0011
2026-05-03 08:36:41,889 INFO Regime score epoch  4/50 — tr=0.0038 va=0.0011
2026-05-03 08:36:42,148 INFO Regime score epoch  5/50 — tr=0.0038 va=0.0011 mae={'trend_score': 0.0211, 'range_score': 0.0353, 'chop_score': 0.0221, 'volatility_percentile': 0.016, 'consolidation_score': 0.0233}
2026-05-03 08:36:42,415 INFO Regime score epoch  6/50 — tr=0.0037 va=0.0011
2026-05-03 08:36:42,687 INFO Regime score epoch  7/50 — tr=0.0038 va=0.0011
2026-05-03 08:36:42,972 INFO Regime score epoch  8/50 — tr=0.0038 va=0.0010
2026-05-03 08:36:43,238 INFO Regime score epoch  9/50 — tr=0.0037 va=0.0010
2026-05-03 08:36:43,489 INFO Regime score epoch 10/50 — tr=0.0037 va=0.0010 mae={'trend_score': 0.0207, 'range_score': 0.0352, 'chop_score': 0.0214, 'volatility_percentile': 0.0153, 'consolidation_score': 0.0231}
2026-05-03 08:36:43,744 INFO Regime score epoch 11/50 — tr=0.0037 va=0.0010
2026-05-03 08:36:44,016 INFO Regime score epoch 12/50 — tr=0.0037 va=0.0010
2026-05-03 08:36:44,277 INFO Regime score epoch 13/50 — tr=0.0037 va=0.0010
2026-05-03 08:36:44,531 INFO Regime score epoch 14/50 — tr=0.0037 va=0.0010
2026-05-03 08:36:44,790 INFO Regime score epoch 15/50 — tr=0.0037 va=0.0010 mae={'trend_score': 0.0198, 'range_score': 0.0347, 'chop_score': 0.021, 'volatility_percentile': 0.0154, 'consolidation_score': 0.0233}
2026-05-03 08:36:45,050 INFO Regime score epoch 16/50 — tr=0.0037 va=0.0010
2026-05-03 08:36:45,309 INFO Regime score epoch 17/50 — tr=0.0036 va=0.0010
2026-05-03 08:36:45,574 INFO Regime score epoch 18/50 — tr=0.0036 va=0.0010
2026-05-03 08:36:45,843 INFO Regime score epoch 19/50 — tr=0.0036 va=0.0010
2026-05-03 08:36:46,110 INFO Regime score epoch 20/50 — tr=0.0036 va=0.0010 mae={'trend_score': 0.0196, 'range_score': 0.0345, 'chop_score': 0.0208, 'volatility_percentile': 0.0152, 'consolidation_score': 0.0223}
2026-05-03 08:36:46,380 INFO Regime score epoch 21/50 — tr=0.0036 va=0.0010
2026-05-03 08:36:46,649 INFO Regime score epoch 22/50 — tr=0.0036 va=0.0010
2026-05-03 08:36:46,909 INFO Regime score epoch 23/50 — tr=0.0036 va=0.0010
2026-05-03 08:36:47,173 INFO Regime score epoch 24/50 — tr=0.0036 va=0.0010
2026-05-03 08:36:47,440 INFO Regime score epoch 25/50 — tr=0.0036 va=0.0010 mae={'trend_score': 0.0193, 'range_score': 0.034, 'chop_score': 0.0205, 'volatility_percentile': 0.0148, 'consolidation_score': 0.0222}
2026-05-03 08:36:47,696 INFO Regime score epoch 26/50 — tr=0.0036 va=0.0010
2026-05-03 08:36:47,971 INFO Regime score epoch 27/50 — tr=0.0036 va=0.0010
2026-05-03 08:36:48,230 INFO Regime score epoch 28/50 — tr=0.0036 va=0.0010
2026-05-03 08:36:48,493 INFO Regime score epoch 29/50 — tr=0.0035 va=0.0009
2026-05-03 08:36:48,755 INFO Regime score epoch 30/50 — tr=0.0035 va=0.0009 mae={'trend_score': 0.0189, 'range_score': 0.0337, 'chop_score': 0.0205, 'volatility_percentile': 0.0147, 'consolidation_score': 0.0225}
2026-05-03 08:36:49,028 INFO Regime score epoch 31/50 — tr=0.0035 va=0.0009
2026-05-03 08:36:49,287 INFO Regime score epoch 32/50 — tr=0.0035 va=0.0010
2026-05-03 08:36:49,547 INFO Regime score epoch 33/50 — tr=0.0035 va=0.0009
2026-05-03 08:36:49,806 INFO Regime score epoch 34/50 — tr=0.0035 va=0.0009
2026-05-03 08:36:50,071 INFO Regime score epoch 35/50 — tr=0.0035 va=0.0009 mae={'trend_score': 0.0189, 'range_score': 0.0338, 'chop_score': 0.0203, 'volatility_percentile': 0.0147, 'consolidation_score': 0.0223}
2026-05-03 08:36:50,366 INFO Regime score epoch 36/50 — tr=0.0035 va=0.0009
2026-05-03 08:36:50,621 INFO Regime score epoch 37/50 — tr=0.0035 va=0.0009
2026-05-03 08:36:50,880 INFO Regime score epoch 38/50 — tr=0.0035 va=0.0009
2026-05-03 08:36:51,147 INFO Regime score epoch 39/50 — tr=0.0035 va=0.0009
2026-05-03 08:36:51,413 INFO Regime score epoch 40/50 — tr=0.0035 va=0.0009 mae={'trend_score': 0.0189, 'range_score': 0.0336, 'chop_score': 0.02, 'volatility_percentile': 0.0147, 'consolidation_score': 0.0222}
2026-05-03 08:36:51,679 INFO Regime score epoch 41/50 — tr=0.0035 va=0.0009
2026-05-03 08:36:51,938 INFO Regime score epoch 42/50 — tr=0.0035 va=0.0009
2026-05-03 08:36:52,201 INFO Regime score epoch 43/50 — tr=0.0035 va=0.0009
2026-05-03 08:36:52,463 INFO Regime score epoch 44/50 — tr=0.0035 va=0.0009
2026-05-03 08:36:52,727 INFO Regime score epoch 45/50 — tr=0.0035 va=0.0009 mae={'trend_score': 0.0186, 'range_score': 0.0338, 'chop_score': 0.0201, 'volatility_percentile': 0.0147, 'consolidation_score': 0.0223}
2026-05-03 08:36:52,997 INFO Regime score epoch 46/50 — tr=0.0035 va=0.0009
2026-05-03 08:36:53,256 INFO Regime score epoch 47/50 — tr=0.0035 va=0.0009
2026-05-03 08:36:53,532 INFO Regime score epoch 48/50 — tr=0.0035 va=0.0009
2026-05-03 08:36:53,801 INFO Regime score epoch 49/50 — tr=0.0035 va=0.0009
2026-05-03 08:36:54,083 INFO Regime score epoch 50/50 — tr=0.0035 va=0.0009 mae={'trend_score': 0.0186, 'range_score': 0.0336, 'chop_score': 0.0202, 'volatility_percentile': 0.0148, 'consolidation_score': 0.0222}
2026-05-03 08:36:54,126 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0186, 'range_score': 0.0336, 'chop_score': 0.02, 'volatility_percentile': 0.0147, 'consolidation_score': 0.0221} mse={'trend_score': 0.00059, 'range_score': 0.00184, 'chop_score': 0.00065, 'volatility_percentile': 0.00039, 'consolidation_score': 0.00119} corr={'trend_score': 0.9942, 'range_score': 0.9579, 'chop_score': 0.9922, 'volatility_percentile': 0.9962, 'consolidation_score': 0.988} pred_std={'trend_score': 0.2215, 'range_score': 0.1323, 'chop_score': 0.1849, 'volatility_percentile': 0.226, 'consolidation_score': 0.2145} target_std={'trend_score': 0.2246, 'range_score': 0.1462, 'chop_score': 0.1945, 'volatility_percentile': 0.2261, 'consolidation_score': 0.2201}
2026-05-03 08:36:54,131 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-03 08:36:54,132 INFO Regime phase LTF train fold=fold_001: 13.3s
2026-05-03 08:36:54,238 INFO Regime LTF complete fold=fold_001: score_accuracy=0.978, train=128454 val=63453 mae={'trend_score': 0.0186, 'range_score': 0.0336, 'chop_score': 0.02, 'volatility_percentile': 0.0147, 'consolidation_score': 0.0221}
2026-05-03 08:36:54,240 INFO Loaded XAUUSD/1H split=train fold=fold_001: 11927 bars (2018-01-04 → 2020-01-03)
2026-05-03 08:36:54,383 INFO Regime[1H mode=ltf_behaviour fold=fold_001] LTF score diagnostics on XAUUSD:
{'trend_score': {'mean': 0.4746, 'q10': 0.1632, 'q50': 0.4665, 'q90': 0.7938}, 'range_score': {'mean': 0.245, 'q10': 0.0601, 'q50': 0.2291, 'q90': 0.4465}, 'chop_score': {'mean': 0.4737, 'q10': 0.2152, 'q50': 0.4751, 'q90': 0.7334}, 'volatility_percentile': {'mean': 0.3804, 'q10': 0.0861, 'q50': 0.3727, 'q90': 0.6763}, 'consolidation_score': {'mean': 0.187, 'q10': 0.0, 'q50': 0.1137, 'q90': 0.5247}}
2026-05-03 08:36:54,386 INFO === Regime rolling fold 3/3: fold_002 ===
2026-05-03 08:36:54,386 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-03 08:36:54,387 INFO Split boundaries loaded fold=fold_002/3 — train 2020-01-06→2022-01-03  val 2022-01-04→2023-01-03  test 2023-08-07→2025-08-05
2026-05-03 08:36:54,388 INFO Loaded AUDUSD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:36:54,389 INFO Loaded EURGBP/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:36:54,390 INFO Loaded EURJPY/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:36:54,390 INFO Loaded EURUSD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:36:54,391 INFO Loaded GBPJPY/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:36:54,392 INFO Loaded GBPUSD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:36:54,393 INFO Loaded NZDUSD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:36:54,394 INFO Loaded USDCAD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:36:54,395 INFO Loaded USDCHF/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:36:54,396 INFO Loaded USDJPY/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:36:54,397 INFO Loaded XAUUSD/4H split=train fold=fold_002: 3180 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:36:54,402 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:54,404 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:54,405 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:54,405 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:54,406 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:54,406 INFO Loaded AUDUSD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:54,630 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 58, 'BIAS_DOWN': 109, 'BIAS_NEUTRAL': 2829}  ambiguous=1636 (total=2996) horizon=12
2026-05-03 08:36:54,632 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected AUDUSD — 2946 samples (group=dollar) score_means={'bias_up_score': 0.0197, 'bias_down_score': 0.037} labels={'BIAS_UP': 58, 'BIAS_DOWN': 109, 'BIAS_NEUTRAL': 2779} clean={'BIAS_UP': 58, 'BIAS_DOWN': 109, 'BIAS_NEUTRAL': 1164}
2026-05-03 08:36:54,741 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:54,744 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:54,745 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:54,745 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:54,746 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:54,746 INFO Loaded EURGBP/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:54,961 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 63, 'BIAS_DOWN': 32, 'BIAS_NEUTRAL': 2901}  ambiguous=1717 (total=2996) horizon=12
2026-05-03 08:36:54,964 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected EURGBP — 2946 samples (group=cross) score_means={'bias_up_score': 0.0214, 'bias_down_score': 0.0109} labels={'BIAS_UP': 63, 'BIAS_DOWN': 32, 'BIAS_NEUTRAL': 2851} clean={'BIAS_UP': 63, 'BIAS_DOWN': 32, 'BIAS_NEUTRAL': 1160}
2026-05-03 08:36:55,068 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:55,073 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:55,073 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:55,074 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:55,074 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:55,075 INFO Loaded EURJPY/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:55,290 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 82, 'BIAS_DOWN': 58, 'BIAS_NEUTRAL': 2856}  ambiguous=1784 (total=2996) horizon=12
2026-05-03 08:36:55,293 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected EURJPY — 2946 samples (group=cross) score_means={'bias_up_score': 0.0278, 'bias_down_score': 0.0197} labels={'BIAS_UP': 82, 'BIAS_DOWN': 58, 'BIAS_NEUTRAL': 2806} clean={'BIAS_UP': 82, 'BIAS_DOWN': 58, 'BIAS_NEUTRAL': 1055}
2026-05-03 08:36:55,396 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:55,399 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:55,399 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:55,400 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:55,400 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:55,401 INFO Loaded EURUSD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:55,613 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 125, 'BIAS_DOWN': 105, 'BIAS_NEUTRAL': 2766}  ambiguous=1697 (total=2996) horizon=12
2026-05-03 08:36:55,615 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected EURUSD — 2946 samples (group=dollar) score_means={'bias_up_score': 0.0424, 'bias_down_score': 0.0356} labels={'BIAS_UP': 125, 'BIAS_DOWN': 105, 'BIAS_NEUTRAL': 2716} clean={'BIAS_UP': 125, 'BIAS_DOWN': 105, 'BIAS_NEUTRAL': 1043}
2026-05-03 08:36:55,720 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:55,723 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:55,724 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:55,724 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:55,724 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:55,725 INFO Loaded GBPJPY/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:55,935 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 112, 'BIAS_DOWN': 61, 'BIAS_NEUTRAL': 2823}  ambiguous=1763 (total=2996) horizon=12
2026-05-03 08:36:55,938 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected GBPJPY — 2946 samples (group=cross) score_means={'bias_up_score': 0.038, 'bias_down_score': 0.0207} labels={'BIAS_UP': 112, 'BIAS_DOWN': 61, 'BIAS_NEUTRAL': 2773} clean={'BIAS_UP': 112, 'BIAS_DOWN': 61, 'BIAS_NEUTRAL': 1028}
2026-05-03 08:36:56,045 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:56,047 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:56,048 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:56,048 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:56,049 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:56,049 INFO Loaded GBPUSD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:56,256 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 71, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 2879}  ambiguous=1724 (total=2996) horizon=12
2026-05-03 08:36:56,259 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected GBPUSD — 2946 samples (group=dollar) score_means={'bias_up_score': 0.0241, 'bias_down_score': 0.0156} labels={'BIAS_UP': 71, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 2829} clean={'BIAS_UP': 71, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 1128}
2026-05-03 08:36:56,362 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:36:56,363 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:36:56,364 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:36:56,364 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:36:56,365 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:36:56,365 INFO Loaded NZDUSD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:56,575 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 73, 'BIAS_DOWN': 89, 'BIAS_NEUTRAL': 2834}  ambiguous=1663 (total=2996) horizon=12
2026-05-03 08:36:56,578 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected NZDUSD — 2946 samples (group=dollar) score_means={'bias_up_score': 0.0248, 'bias_down_score': 0.0302} labels={'BIAS_UP': 73, 'BIAS_DOWN': 89, 'BIAS_NEUTRAL': 2784} clean={'BIAS_UP': 73, 'BIAS_DOWN': 89, 'BIAS_NEUTRAL': 1147}
2026-05-03 08:36:56,678 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:56,681 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:56,682 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:56,682 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:56,682 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:56,683 INFO Loaded USDCAD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:56,892 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 108, 'BIAS_DOWN': 102, 'BIAS_NEUTRAL': 2786}  ambiguous=1605 (total=2996) horizon=12
2026-05-03 08:36:56,895 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected USDCAD — 2946 samples (group=dollar) score_means={'bias_up_score': 0.0367, 'bias_down_score': 0.0346} labels={'BIAS_UP': 108, 'BIAS_DOWN': 102, 'BIAS_NEUTRAL': 2736} clean={'BIAS_UP': 108, 'BIAS_DOWN': 102, 'BIAS_NEUTRAL': 1162}
2026-05-03 08:36:56,999 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:57,002 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:57,002 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:57,003 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:57,003 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:57,004 INFO Loaded USDCHF/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:57,223 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 74, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 2831}  ambiguous=1654 (total=2996) horizon=12
2026-05-03 08:36:57,226 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected USDCHF — 2946 samples (group=dollar) score_means={'bias_up_score': 0.0251, 'bias_down_score': 0.0309} labels={'BIAS_UP': 74, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 2781} clean={'BIAS_UP': 74, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 1151}
2026-05-03 08:36:57,331 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:57,333 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:57,334 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:57,334 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:57,335 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:57,336 INFO Loaded USDJPY/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:57,549 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 70, 'BIAS_DOWN': 56, 'BIAS_NEUTRAL': 2870}  ambiguous=1792 (total=2996) horizon=12
2026-05-03 08:36:57,552 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected USDJPY — 2946 samples (group=dollar) score_means={'bias_up_score': 0.0238, 'bias_down_score': 0.019} labels={'BIAS_UP': 70, 'BIAS_DOWN': 56, 'BIAS_NEUTRAL': 2820} clean={'BIAS_UP': 70, 'BIAS_DOWN': 56, 'BIAS_NEUTRAL': 1061}
2026-05-03 08:36:57,664 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:36:57,668 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:36:57,669 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:36:57,670 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:36:57,670 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:36:57,671 INFO Loaded XAUUSD/4H split=train fold=fold_002: 3180 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:57,898 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 100, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 3045}  ambiguous=1873 (total=3180) horizon=12
2026-05-03 08:36:57,901 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected XAUUSD — 3130 samples (group=gold) score_means={'bias_up_score': 0.0319, 'bias_down_score': 0.0112} labels={'BIAS_UP': 100, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 2995} clean={'BIAS_UP': 100, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 1144}
2026-05-03 08:36:58,001 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 257, 'BIAS_DOWN': 151, 'BIAS_NEUTRAL': 8430}, 'dollar': {'BIAS_UP': 579, 'BIAS_DOWN': 598, 'BIAS_NEUTRAL': 19445}, 'gold': {'BIAS_UP': 100, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 2995}}
2026-05-03 08:36:58,001 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0291, 'bias_down_score': 0.0171}, 'dollar': {'bias_up_score': 0.0281, 'bias_down_score': 0.029}, 'gold': {'bias_up_score': 0.0319, 'bias_down_score': 0.0112}}
2026-05-03 08:36:58,001 INFO Regime[4H mode=htf_bias] label distribution by year: {2020: {'BIAS_UP': 484, 'BIAS_DOWN': 407, 'BIAS_NEUTRAL': 15040}, 2021: {'BIAS_UP': 452, 'BIAS_DOWN': 377, 'BIAS_NEUTRAL': 15762}, 2022: {'BIAS_UP': 0, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 68}}
2026-05-03 08:36:58,002 INFO Regime[4H mode=htf_bias] score means by year: {2020: {'bias_up_score': 0.0304, 'bias_down_score': 0.0255}, 2021: {'bias_up_score': 0.0272, 'bias_down_score': 0.0227}, 2022: {'bias_up_score': 0.0, 'bias_down_score': 0.0}}
2026-05-03 08:36:58,080 INFO Loaded AUDUSD/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
2026-05-03 08:36:58,081 INFO Loaded EURGBP/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
2026-05-03 08:36:58,082 INFO Loaded EURJPY/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
2026-05-03 08:36:58,083 INFO Loaded EURUSD/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
2026-05-03 08:36:58,084 INFO Loaded GBPJPY/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
2026-05-03 08:36:58,085 INFO Loaded GBPUSD/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
2026-05-03 08:36:58,085 INFO Loaded NZDUSD/4H split=val fold=fold_002: 1235 bars (2022-01-04 → 2022-10-28)
2026-05-03 08:36:58,086 INFO Loaded USDCAD/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
2026-05-03 08:36:58,087 INFO Loaded USDCHF/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
2026-05-03 08:36:58,088 INFO Loaded USDJPY/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
2026-05-03 08:36:58,089 INFO Loaded XAUUSD/4H split=val fold=fold_002: 1596 bars (2022-01-04 → 2023-01-03)
2026-05-03 08:36:58,095 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:58,097 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:58,098 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:58,098 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:58,099 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:58,100 INFO Loaded AUDUSD/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:58,300 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 10, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 1476}  ambiguous=876 (total=1511) horizon=12
2026-05-03 08:36:58,303 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected AUDUSD — 1461 samples (group=dollar) score_means={'bias_up_score': 0.0068, 'bias_down_score': 0.0171} labels={'BIAS_UP': 10, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 1426} clean={'BIAS_UP': 10, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 583}
2026-05-03 08:36:58,412 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:58,416 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:58,417 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:58,417 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:58,417 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:58,418 INFO Loaded EURGBP/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:58,617 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 36, 'BIAS_DOWN': 14, 'BIAS_NEUTRAL': 1461}  ambiguous=814 (total=1511) horizon=12
2026-05-03 08:36:58,619 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected EURGBP — 1461 samples (group=cross) score_means={'bias_up_score': 0.0246, 'bias_down_score': 0.0096} labels={'BIAS_UP': 36, 'BIAS_DOWN': 14, 'BIAS_NEUTRAL': 1411} clean={'BIAS_UP': 36, 'BIAS_DOWN': 14, 'BIAS_NEUTRAL': 614}
2026-05-03 08:36:58,723 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:58,725 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:58,727 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:58,727 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:58,728 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:58,729 INFO Loaded EURJPY/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:58,915 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 92, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 1406}  ambiguous=879 (total=1511) horizon=12
2026-05-03 08:36:58,917 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected EURJPY — 1461 samples (group=cross) score_means={'bias_up_score': 0.063, 'bias_down_score': 0.0089} labels={'BIAS_UP': 92, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 1356} clean={'BIAS_UP': 92, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 498}
2026-05-03 08:36:59,022 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:59,026 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:59,027 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:59,027 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:59,028 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:59,029 INFO Loaded EURUSD/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:59,236 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 11, 'BIAS_DOWN': 64, 'BIAS_NEUTRAL': 1436}  ambiguous=853 (total=1511) horizon=12
2026-05-03 08:36:59,239 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected EURUSD — 1461 samples (group=dollar) score_means={'bias_up_score': 0.0075, 'bias_down_score': 0.0438} labels={'BIAS_UP': 11, 'BIAS_DOWN': 64, 'BIAS_NEUTRAL': 1386} clean={'BIAS_UP': 11, 'BIAS_DOWN': 64, 'BIAS_NEUTRAL': 566}
2026-05-03 08:36:59,346 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:59,348 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:59,349 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:59,349 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:59,350 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:59,351 INFO Loaded GBPJPY/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:59,540 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 75, 'BIAS_DOWN': 26, 'BIAS_NEUTRAL': 1410}  ambiguous=856 (total=1511) horizon=12
2026-05-03 08:36:59,543 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected GBPJPY — 1461 samples (group=cross) score_means={'bias_up_score': 0.0513, 'bias_down_score': 0.0178} labels={'BIAS_UP': 75, 'BIAS_DOWN': 26, 'BIAS_NEUTRAL': 1360} clean={'BIAS_UP': 75, 'BIAS_DOWN': 26, 'BIAS_NEUTRAL': 522}
2026-05-03 08:36:59,645 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:59,647 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:59,648 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:59,649 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:59,649 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:36:59,650 INFO Loaded GBPUSD/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:36:59,837 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 4, 'BIAS_DOWN': 70, 'BIAS_NEUTRAL': 1437}  ambiguous=862 (total=1511) horizon=12
2026-05-03 08:36:59,840 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected GBPUSD — 1461 samples (group=dollar) score_means={'bias_up_score': 0.0027, 'bias_down_score': 0.0479} labels={'BIAS_UP': 4, 'BIAS_DOWN': 70, 'BIAS_NEUTRAL': 1387} clean={'BIAS_UP': 4, 'BIAS_DOWN': 70, 'BIAS_NEUTRAL': 555}
2026-05-03 08:36:59,948 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:36:59,950 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:36:59,951 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:36:59,951 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:36:59,951 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:36:59,952 INFO Loaded NZDUSD/4H split=val fold=fold_002: 1235 bars (2022-01-04 → 2022-10-28)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:00,139 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 2, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 1166}  ambiguous=696 (total=1235) horizon=12
2026-05-03 08:37:00,141 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected NZDUSD — 1185 samples (group=dollar) score_means={'bias_up_score': 0.0017, 'bias_down_score': 0.0565} labels={'BIAS_UP': 2, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 1116} clean={'BIAS_UP': 2, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 452}
2026-05-03 08:37:00,246 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:00,249 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:00,250 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:00,250 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:00,250 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:00,251 INFO Loaded USDCAD/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:00,460 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 27, 'BIAS_DOWN': 14, 'BIAS_NEUTRAL': 1470}  ambiguous=814 (total=1511) horizon=12
2026-05-03 08:37:00,463 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected USDCAD — 1461 samples (group=dollar) score_means={'bias_up_score': 0.0185, 'bias_down_score': 0.0096} labels={'BIAS_UP': 27, 'BIAS_DOWN': 14, 'BIAS_NEUTRAL': 1420} clean={'BIAS_UP': 27, 'BIAS_DOWN': 14, 'BIAS_NEUTRAL': 637}
2026-05-03 08:37:00,566 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:00,568 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:00,569 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:00,569 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:00,570 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:00,571 INFO Loaded USDCHF/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:00,756 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 103, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 1383}  ambiguous=907 (total=1511) horizon=12
2026-05-03 08:37:00,759 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected USDCHF — 1461 samples (group=dollar) score_means={'bias_up_score': 0.0705, 'bias_down_score': 0.0171} labels={'BIAS_UP': 103, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 1333} clean={'BIAS_UP': 103, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 461}
2026-05-03 08:37:00,861 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:00,863 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:00,864 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:00,864 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:00,865 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:00,866 INFO Loaded USDJPY/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:01,060 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 143, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 1346}  ambiguous=890 (total=1511) horizon=12
2026-05-03 08:37:01,062 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected USDJPY — 1461 samples (group=dollar) score_means={'bias_up_score': 0.0979, 'bias_down_score': 0.0151} labels={'BIAS_UP': 143, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 1296} clean={'BIAS_UP': 143, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 440}
2026-05-03 08:37:01,175 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:37:01,178 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:37:01,179 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:37:01,180 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:37:01,180 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:37:01,181 INFO Loaded XAUUSD/4H split=val fold=fold_002: 1596 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:01,393 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 38, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1528}  ambiguous=938 (total=1596) horizon=12
2026-05-03 08:37:01,396 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected XAUUSD — 1546 samples (group=gold) score_means={'bias_up_score': 0.0246, 'bias_down_score': 0.0194} labels={'BIAS_UP': 38, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1478} clean={'BIAS_UP': 38, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 568}
2026-05-03 08:37:01,493 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 203, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 4127}, 'dollar': {'BIAS_UP': 300, 'BIAS_DOWN': 287, 'BIAS_NEUTRAL': 9364}, 'gold': {'BIAS_UP': 38, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1478}}
2026-05-03 08:37:01,493 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0463, 'bias_down_score': 0.0121}, 'dollar': {'bias_up_score': 0.0301, 'bias_down_score': 0.0288}, 'gold': {'bias_up_score': 0.0246, 'bias_down_score': 0.0194}}
2026-05-03 08:37:01,493 INFO Regime[4H mode=htf_bias] label distribution by year: {2022: {'BIAS_UP': 541, 'BIAS_DOWN': 370, 'BIAS_NEUTRAL': 14853}, 2023: {'BIAS_UP': 0, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 116}}
2026-05-03 08:37:01,493 INFO Regime[4H mode=htf_bias] score means by year: {2022: {'bias_up_score': 0.0343, 'bias_down_score': 0.0235}, 2023: {'bias_up_score': 0.0, 'bias_down_score': 0.0}}
2026-05-03 08:37:01,572 INFO Regime phase HTF dataset build fold=fold_002: 7.2s (train=32590 val=15880)
2026-05-03 08:37:01,576 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=24, n_classes=2)
2026-05-03 08:37:01,576 INFO Regime 4H/htf_bias warm start enabled from existing weights
2026-05-03 08:37:01,580 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=32590 val=15880 train_labels={'BIAS_UP': 936, 'BIAS_DOWN': 784, 'BIAS_NEUTRAL': 30870} val_labels={'BIAS_UP': 541, 'BIAS_DOWN': 370, 'BIAS_NEUTRAL': 14969}
2026-05-03 08:37:01,580 INFO RegimeClassifier[mode=htf_bias]: warm start HTF score head
2026-05-03 08:37:01,580 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-03 08:37:01,581 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 20.0, 'bias_down_score': 20.0}
2026-05-03 08:37:02,519 INFO Regime HTF score epoch  1/50 — tr=0.3638 va=0.4342 acc=0.851 bal=0.914 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.957, 'BIAS_DOWN': 0.941, 'BIAS_NEUTRAL': 0.845} precision={'BIAS_UP': 0.323, 'BIAS_DOWN': 0.22, 'BIAS_NEUTRAL': 0.996}
2026-05-03 08:37:03,471 INFO Regime HTF score epoch  2/50 — tr=0.3650 va=0.4367 bal=0.918
2026-05-03 08:37:04,409 INFO Regime HTF score epoch  3/50 — tr=0.3624 va=0.4392 bal=0.919
2026-05-03 08:37:05,351 INFO Regime HTF score epoch  4/50 — tr=0.3654 va=0.4422 bal=0.920
2026-05-03 08:37:06,284 INFO Regime HTF score epoch  5/50 — tr=0.3688 va=0.4500 acc=0.843 bal=0.923 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.965, 'BIAS_DOWN': 0.97, 'BIAS_NEUTRAL': 0.835} precision={'BIAS_UP': 0.315, 'BIAS_DOWN': 0.212, 'BIAS_NEUTRAL': 0.998}
2026-05-03 08:37:07,214 INFO Regime HTF score epoch  6/50 — tr=0.3626 va=0.4506 bal=0.923
2026-05-03 08:37:08,146 INFO Regime HTF score epoch  7/50 — tr=0.3535 va=0.4502 bal=0.923
2026-05-03 08:37:09,080 INFO Regime HTF score epoch  8/50 — tr=0.3511 va=0.4502 bal=0.924
2026-05-03 08:37:10,023 INFO Regime HTF score epoch  9/50 — tr=0.3522 va=0.4508 bal=0.923
2026-05-03 08:37:11,010 INFO Regime HTF score epoch 10/50 — tr=0.3529 va=0.4516 acc=0.840 bal=0.922 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.965, 'BIAS_DOWN': 0.97, 'BIAS_NEUTRAL': 0.832} precision={'BIAS_UP': 0.31, 'BIAS_DOWN': 0.21, 'BIAS_NEUTRAL': 0.998}
2026-05-03 08:37:11,949 INFO Regime HTF score epoch 11/50 — tr=0.3493 va=0.4482 bal=0.923
2026-05-03 08:37:12,880 INFO Regime HTF score epoch 12/50 — tr=0.3469 va=0.4487 bal=0.923
2026-05-03 08:37:13,829 INFO Regime HTF score epoch 13/50 — tr=0.3446 va=0.4471 bal=0.922
2026-05-03 08:37:14,776 INFO Regime HTF score epoch 14/50 — tr=0.3358 va=0.4459 bal=0.922
2026-05-03 08:37:15,711 INFO Regime HTF score epoch 15/50 — tr=0.3379 va=0.4430 acc=0.842 bal=0.922 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.967, 'BIAS_DOWN': 0.965, 'BIAS_NEUTRAL': 0.835} precision={'BIAS_UP': 0.312, 'BIAS_DOWN': 0.213, 'BIAS_NEUTRAL': 0.998}
2026-05-03 08:37:16,645 INFO Regime HTF score epoch 16/50 — tr=0.3382 va=0.4433 bal=0.921
2026-05-03 08:37:16,646 INFO Regime HTF score early stop at epoch 16
2026-05-03 08:37:17,518 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.850 margin=0.000 precision={'BIAS_UP': 0.31, 'BIAS_DOWN': 0.214, 'BIAS_NEUTRAL': 0.998} recall={'BIAS_UP': 0.967, 'BIAS_DOWN': 0.97, 'BIAS_NEUTRAL': 0.834} f1={'BIAS_UP': 0.47, 'BIAS_DOWN': 0.35, 'BIAS_NEUTRAL': 0.909} confusion=[[523, 0, 18], [0, 359, 11], [1162, 1321, 12486]] score_mae={'bias_up_score': 0.1654, 'bias_down_score': 0.1831} pred_share={'BIAS_UP': 0.1061, 'BIAS_DOWN': 0.1058, 'BIAS_NEUTRAL': 0.7881}
2026-05-03 08:37:17,519 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.31, 'BIAS_DOWN': 0.214, 'BIAS_NEUTRAL': 0.998} min_precision=0.300 recall={'BIAS_UP': 0.967, 'BIAS_DOWN': 0.97, 'BIAS_NEUTRAL': 0.834} min_recall=0.100 f1={'BIAS_UP': 0.47, 'BIAS_DOWN': 0.35, 'BIAS_NEUTRAL': 0.909} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_DOWN'] weak_recall=[] weak_f1=[] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-03 08:37:17,522 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-03 08:37:17,522 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-03 08:37:17,523 INFO Regime phase HTF train fold=fold_002: 15.9s
2026-05-03 08:37:17,619 INFO Regime HTF complete fold=fold_002: acc=0.842 bal=0.924 train=32590 val=15880 per_class={'BIAS_UP': 0.967, 'BIAS_DOWN': 0.97, 'BIAS_NEUTRAL': 0.834} precision={'BIAS_UP': 0.31, 'BIAS_DOWN': 0.214, 'BIAS_NEUTRAL': 0.998} threshold=0.850 margin=0.000
2026-05-03 08:37:17,621 INFO Loaded XAUUSD/4H split=train fold=fold_002: 3180 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:37:17,715 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 100, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 3045}  ambiguous=1873 (total=3180) horizon=12
2026-05-03 08:37:17,717 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 3.8461538461538463, 'BIAS_DOWN': 2.9166666666666665, 'BIAS_NEUTRAL': 78.07692307692308}
2026-05-03 08:37:17,719 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 100, 'mean': 0.002066267653568764, 'mean_over_std': 0.5821881003488305}, 'BIAS_DOWN': {'n': 35, 'mean': -0.002374414438759463, 'mean_over_std': -0.4293108000037002}, 'BIAS_NEUTRAL': {'n': 3044, 'mean': 1.3199249771542425e-05, 'mean_over_std': 0.0031224494254571203}}
2026-05-03 08:37:17,720 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 100, 'mean': 0.002066267653568764, 'mean_over_std': 0.5821881003488305}, 'BIAS_DOWN': {'n': 35, 'mean': -0.002374414438759463, 'mean_over_std': -0.4293108000037002}, 'BIAS_NEUTRAL': {'n': 1172, 'mean': 1.3513765578608547e-05, 'mean_over_std': 0.004104063760480685}}
2026-05-03 08:37:17,723 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-03 08:37:17,725 INFO Loaded AUDUSD/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:37:17,726 INFO Loaded EURGBP/1H split=train fold=fold_002: 11690 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:37:17,727 INFO Loaded EURJPY/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:37:17,729 INFO Loaded EURUSD/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:37:17,730 INFO Loaded GBPJPY/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:37:17,731 INFO Loaded GBPUSD/1H split=train fold=fold_002: 11691 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:37:17,732 INFO Loaded NZDUSD/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:37:17,734 INFO Loaded USDCAD/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:37:17,735 INFO Loaded USDCHF/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:37:17,736 INFO Loaded USDJPY/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:37:17,738 INFO Loaded XAUUSD/1H split=train fold=fold_002: 11775 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:37:17,743 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:17,745 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:17,746 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:17,747 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:17,747 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:17,748 INFO Loaded AUDUSD/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:18,082 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected AUDUSD — 11642 samples (group=dollar) score_means={'trend_score': 0.4899, 'range_score': 0.2319, 'chop_score': 0.4643, 'volatility_percentile': 0.382, 'consolidation_score': 0.1914}
2026-05-03 08:37:18,192 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:18,196 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:18,198 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:18,199 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:18,199 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:18,201 INFO Loaded EURGBP/1H split=train fold=fold_002: 11690 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:18,524 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected EURGBP — 11640 samples (group=cross) score_means={'trend_score': 0.473, 'range_score': 0.2414, 'chop_score': 0.4735, 'volatility_percentile': 0.3812, 'consolidation_score': 0.1876}
2026-05-03 08:37:18,627 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:18,632 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:18,632 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:18,633 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:18,633 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:18,635 INFO Loaded EURJPY/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:18,959 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected EURJPY — 11642 samples (group=cross) score_means={'trend_score': 0.4884, 'range_score': 0.2345, 'chop_score': 0.4693, 'volatility_percentile': 0.3824, 'consolidation_score': 0.1897}
2026-05-03 08:37:19,061 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:19,064 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:19,064 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:19,065 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:19,065 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:19,067 INFO Loaded EURUSD/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:19,391 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected EURUSD — 11642 samples (group=dollar) score_means={'trend_score': 0.4896, 'range_score': 0.2324, 'chop_score': 0.4597, 'volatility_percentile': 0.3849, 'consolidation_score': 0.1841}
2026-05-03 08:37:19,494 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:19,497 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:19,498 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:19,498 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:19,499 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:19,501 INFO Loaded GBPJPY/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:19,817 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected GBPJPY — 11642 samples (group=cross) score_means={'trend_score': 0.4783, 'range_score': 0.2365, 'chop_score': 0.4744, 'volatility_percentile': 0.3783, 'consolidation_score': 0.1953}
2026-05-03 08:37:19,925 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:19,927 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:19,928 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:19,929 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:19,929 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:19,931 INFO Loaded GBPUSD/1H split=train fold=fold_002: 11691 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:20,279 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected GBPUSD — 11641 samples (group=dollar) score_means={'trend_score': 0.4904, 'range_score': 0.231, 'chop_score': 0.4614, 'volatility_percentile': 0.3769, 'consolidation_score': 0.1885}
2026-05-03 08:37:20,392 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:37:20,394 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:37:20,394 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:37:20,395 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:37:20,395 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:37:20,396 INFO Loaded NZDUSD/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:20,711 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected NZDUSD — 11642 samples (group=dollar) score_means={'trend_score': 0.4792, 'range_score': 0.2346, 'chop_score': 0.4664, 'volatility_percentile': 0.378, 'consolidation_score': 0.1905}
2026-05-03 08:37:20,813 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:20,815 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:20,816 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:20,817 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:20,817 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:20,819 INFO Loaded USDCAD/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:21,152 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected USDCAD — 11642 samples (group=dollar) score_means={'trend_score': 0.4835, 'range_score': 0.2384, 'chop_score': 0.4682, 'volatility_percentile': 0.3817, 'consolidation_score': 0.1872}
2026-05-03 08:37:21,256 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:21,258 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:21,259 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:21,260 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:21,260 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:21,262 INFO Loaded USDCHF/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:21,583 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected USDCHF — 11642 samples (group=dollar) score_means={'trend_score': 0.4766, 'range_score': 0.2426, 'chop_score': 0.4695, 'volatility_percentile': 0.3844, 'consolidation_score': 0.1868}
2026-05-03 08:37:21,685 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:21,687 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:21,688 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:21,689 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:21,689 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:21,691 INFO Loaded USDJPY/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:22,016 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected USDJPY — 11642 samples (group=dollar) score_means={'trend_score': 0.4905, 'range_score': 0.2324, 'chop_score': 0.4655, 'volatility_percentile': 0.3784, 'consolidation_score': 0.1968}
2026-05-03 08:37:22,133 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:37:22,137 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:37:22,138 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:37:22,139 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:37:22,139 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:37:22,142 INFO Loaded XAUUSD/1H split=train fold=fold_002: 11775 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:22,487 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected XAUUSD — 11725 samples (group=gold) score_means={'trend_score': 0.4817, 'range_score': 0.2418, 'chop_score': 0.4772, 'volatility_percentile': 0.3667, 'consolidation_score': 0.1995}
2026-05-03 08:37:22,589 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4799, 'range_score': 0.2375, 'chop_score': 0.4724, 'volatility_percentile': 0.3806, 'consolidation_score': 0.1909}, 'dollar': {'trend_score': 0.4857, 'range_score': 0.2348, 'chop_score': 0.465, 'volatility_percentile': 0.3809, 'consolidation_score': 0.1893}, 'gold': {'trend_score': 0.4817, 'range_score': 0.2418, 'chop_score': 0.4772, 'volatility_percentile': 0.3667, 'consolidation_score': 0.1995}}
2026-05-03 08:37:22,589 INFO Regime[1H mode=ltf_behaviour] score means by year: {2020: {'trend_score': 0.4835, 'range_score': 0.2339, 'chop_score': 0.4687, 'volatility_percentile': 0.3796, 'consolidation_score': 0.1912}, 2021: {'trend_score': 0.484, 'range_score': 0.2383, 'chop_score': 0.4676, 'volatility_percentile': 0.3789, 'consolidation_score': 0.1908}, 2022: {'trend_score': 0.4753, 'range_score': 0.2581, 'chop_score': 0.4747, 'volatility_percentile': 0.5171, 'consolidation_score': 0.0424}}
2026-05-03 08:37:22,668 INFO Loaded AUDUSD/1H split=val fold=fold_002: 5894 bars (2022-01-04 → 2023-01-03)
2026-05-03 08:37:22,669 INFO Loaded EURGBP/1H split=val fold=fold_002: 5897 bars (2022-01-04 → 2023-01-03)
2026-05-03 08:37:22,670 INFO Loaded EURJPY/1H split=val fold=fold_002: 5893 bars (2022-01-04 → 2023-01-03)
2026-05-03 08:37:22,672 INFO Loaded EURUSD/1H split=val fold=fold_002: 5897 bars (2022-01-04 → 2023-01-03)
2026-05-03 08:37:22,673 INFO Loaded GBPJPY/1H split=val fold=fold_002: 5895 bars (2022-01-04 → 2023-01-03)
2026-05-03 08:37:22,674 INFO Loaded GBPUSD/1H split=val fold=fold_002: 5895 bars (2022-01-04 → 2023-01-03)
2026-05-03 08:37:22,676 INFO Loaded NZDUSD/1H split=val fold=fold_002: 4820 bars (2022-01-04 → 2022-10-28)
2026-05-03 08:37:22,677 INFO Loaded USDCAD/1H split=val fold=fold_002: 5897 bars (2022-01-04 → 2023-01-03)
2026-05-03 08:37:22,678 INFO Loaded USDCHF/1H split=val fold=fold_002: 5894 bars (2022-01-04 → 2023-01-03)
2026-05-03 08:37:22,680 INFO Loaded USDJPY/1H split=val fold=fold_002: 5895 bars (2022-01-04 → 2023-01-03)
2026-05-03 08:37:22,681 INFO Loaded XAUUSD/1H split=val fold=fold_002: 5914 bars (2022-01-04 → 2023-01-03)
2026-05-03 08:37:22,687 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:22,689 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:22,690 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:22,690 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:22,691 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:22,692 INFO Loaded AUDUSD/1H split=val fold=fold_002: 5894 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:22,938 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected AUDUSD — 5844 samples (group=dollar) score_means={'trend_score': 0.4823, 'range_score': 0.2353, 'chop_score': 0.4656, 'volatility_percentile': 0.3905, 'consolidation_score': 0.1791}
2026-05-03 08:37:23,040 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:23,043 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:23,043 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:23,044 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:23,044 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:23,046 INFO Loaded EURGBP/1H split=val fold=fold_002: 5897 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:23,291 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected EURGBP — 5847 samples (group=cross) score_means={'trend_score': 0.4487, 'range_score': 0.2519, 'chop_score': 0.4897, 'volatility_percentile': 0.3945, 'consolidation_score': 0.1768}
2026-05-03 08:37:23,392 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:23,394 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:23,395 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:23,395 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:23,396 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:23,397 INFO Loaded EURJPY/1H split=val fold=fold_002: 5893 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:23,635 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected EURJPY — 5843 samples (group=cross) score_means={'trend_score': 0.5036, 'range_score': 0.2299, 'chop_score': 0.4561, 'volatility_percentile': 0.4037, 'consolidation_score': 0.1685}
2026-05-03 08:37:23,735 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:23,737 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:23,738 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:23,738 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:23,739 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:23,740 INFO Loaded EURUSD/1H split=val fold=fold_002: 5897 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:23,988 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected EURUSD — 5847 samples (group=dollar) score_means={'trend_score': 0.4803, 'range_score': 0.2444, 'chop_score': 0.47, 'volatility_percentile': 0.3951, 'consolidation_score': 0.1781}
2026-05-03 08:37:24,092 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:24,094 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:24,095 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:24,095 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:24,095 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:24,097 INFO Loaded GBPJPY/1H split=val fold=fold_002: 5895 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:24,350 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected GBPJPY — 5845 samples (group=cross) score_means={'trend_score': 0.4766, 'range_score': 0.2379, 'chop_score': 0.4728, 'volatility_percentile': 0.3937, 'consolidation_score': 0.1772}
2026-05-03 08:37:24,453 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:24,455 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:24,456 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:24,456 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:24,456 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:24,458 INFO Loaded GBPUSD/1H split=val fold=fold_002: 5895 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:24,708 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected GBPUSD — 5845 samples (group=dollar) score_means={'trend_score': 0.4678, 'range_score': 0.246, 'chop_score': 0.476, 'volatility_percentile': 0.3971, 'consolidation_score': 0.179}
2026-05-03 08:37:24,811 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:37:24,812 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:37:24,813 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:37:24,813 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:37:24,814 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:37:24,815 INFO Loaded NZDUSD/1H split=val fold=fold_002: 4820 bars (2022-01-04 → 2022-10-28)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:25,048 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected NZDUSD — 4770 samples (group=dollar) score_means={'trend_score': 0.4804, 'range_score': 0.2356, 'chop_score': 0.4646, 'volatility_percentile': 0.4152, 'consolidation_score': 0.1616}
2026-05-03 08:37:25,155 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:25,158 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:25,158 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:25,159 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:25,159 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:25,161 INFO Loaded USDCAD/1H split=val fold=fold_002: 5897 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:25,411 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected USDCAD — 5847 samples (group=dollar) score_means={'trend_score': 0.4792, 'range_score': 0.2417, 'chop_score': 0.4731, 'volatility_percentile': 0.3881, 'consolidation_score': 0.1864}
2026-05-03 08:37:25,513 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:25,515 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:25,516 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:25,516 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:25,516 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:25,518 INFO Loaded USDCHF/1H split=val fold=fold_002: 5894 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:25,753 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected USDCHF — 5844 samples (group=dollar) score_means={'trend_score': 0.4698, 'range_score': 0.2432, 'chop_score': 0.4693, 'volatility_percentile': 0.3966, 'consolidation_score': 0.172}
2026-05-03 08:37:25,856 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:25,858 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:25,859 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:25,859 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:25,859 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:25,861 INFO Loaded USDJPY/1H split=val fold=fold_002: 5895 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:26,099 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected USDJPY — 5845 samples (group=dollar) score_means={'trend_score': 0.5188, 'range_score': 0.2217, 'chop_score': 0.4472, 'volatility_percentile': 0.398, 'consolidation_score': 0.1782}
2026-05-03 08:37:26,212 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:37:26,215 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:37:26,216 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:37:26,217 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:37:26,217 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:37:26,220 INFO Loaded XAUUSD/1H split=val fold=fold_002: 5914 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:26,479 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected XAUUSD — 5864 samples (group=gold) score_means={'trend_score': 0.4904, 'range_score': 0.2349, 'chop_score': 0.465, 'volatility_percentile': 0.3828, 'consolidation_score': 0.1876}
2026-05-03 08:37:26,576 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4763, 'range_score': 0.2399, 'chop_score': 0.4729, 'volatility_percentile': 0.3973, 'consolidation_score': 0.1742}, 'dollar': {'trend_score': 0.4827, 'range_score': 0.2383, 'chop_score': 0.4666, 'volatility_percentile': 0.3968, 'consolidation_score': 0.1767}, 'gold': {'trend_score': 0.4904, 'range_score': 0.2349, 'chop_score': 0.465, 'volatility_percentile': 0.3828, 'consolidation_score': 0.1876}}
2026-05-03 08:37:26,577 INFO Regime[1H mode=ltf_behaviour] score means by year: {2022: {'trend_score': 0.4816, 'range_score': 0.2386, 'chop_score': 0.4682, 'volatility_percentile': 0.3949, 'consolidation_score': 0.1773}, 2023: {'trend_score': 0.4952, 'range_score': 0.2222, 'chop_score': 0.462, 'volatility_percentile': 0.4876, 'consolidation_score': 0.1447}}
2026-05-03 08:37:26,651 INFO Regime phase LTF dataset build fold=fold_002: 8.9s (train=128142 val=63241)
2026-05-03 08:37:26,656 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=31, n_classes=5)
2026-05-03 08:37:26,656 INFO Regime 1H/ltf_behaviour warm start enabled from existing weights
2026-05-03 08:37:26,674 INFO RegimeClassifier[mode=ltf_behaviour]: warm start score head
2026-05-03 08:37:26,674 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-03 08:37:26,943 INFO Regime score epoch  1/50 — tr=0.0036 va=0.0009 mae={'trend_score': 0.0201, 'range_score': 0.0343, 'chop_score': 0.021, 'volatility_percentile': 0.0144, 'consolidation_score': 0.0208}
2026-05-03 08:37:27,217 INFO Regime score epoch  2/50 — tr=0.0036 va=0.0009
2026-05-03 08:37:27,469 INFO Regime score epoch  3/50 — tr=0.0036 va=0.0009
2026-05-03 08:37:27,728 INFO Regime score epoch  4/50 — tr=0.0035 va=0.0009
2026-05-03 08:37:28,001 INFO Regime score epoch  5/50 — tr=0.0036 va=0.0009 mae={'trend_score': 0.0189, 'range_score': 0.0334, 'chop_score': 0.0204, 'volatility_percentile': 0.0142, 'consolidation_score': 0.0208}
2026-05-03 08:37:28,258 INFO Regime score epoch  6/50 — tr=0.0035 va=0.0009
2026-05-03 08:37:28,516 INFO Regime score epoch  7/50 — tr=0.0035 va=0.0009
2026-05-03 08:37:28,773 INFO Regime score epoch  8/50 — tr=0.0035 va=0.0009
2026-05-03 08:37:29,050 INFO Regime score epoch  9/50 — tr=0.0035 va=0.0009
2026-05-03 08:37:29,313 INFO Regime score epoch 10/50 — tr=0.0035 va=0.0009 mae={'trend_score': 0.0184, 'range_score': 0.0331, 'chop_score': 0.0202, 'volatility_percentile': 0.0145, 'consolidation_score': 0.0206}
2026-05-03 08:37:29,574 INFO Regime score epoch 11/50 — tr=0.0035 va=0.0009
2026-05-03 08:37:29,836 INFO Regime score epoch 12/50 — tr=0.0035 va=0.0009
2026-05-03 08:37:30,102 INFO Regime score epoch 13/50 — tr=0.0035 va=0.0009
2026-05-03 08:37:30,387 INFO Regime score epoch 14/50 — tr=0.0035 va=0.0009
2026-05-03 08:37:30,641 INFO Regime score epoch 15/50 — tr=0.0035 va=0.0009 mae={'trend_score': 0.0182, 'range_score': 0.0332, 'chop_score': 0.0203, 'volatility_percentile': 0.0146, 'consolidation_score': 0.0204}
2026-05-03 08:37:30,900 INFO Regime score epoch 16/50 — tr=0.0035 va=0.0009
2026-05-03 08:37:31,160 INFO Regime score epoch 17/50 — tr=0.0035 va=0.0009
2026-05-03 08:37:31,430 INFO Regime score epoch 18/50 — tr=0.0035 va=0.0009
2026-05-03 08:37:31,696 INFO Regime score epoch 19/50 — tr=0.0034 va=0.0009
2026-05-03 08:37:31,947 INFO Regime score epoch 20/50 — tr=0.0034 va=0.0009 mae={'trend_score': 0.0179, 'range_score': 0.0327, 'chop_score': 0.0199, 'volatility_percentile': 0.0144, 'consolidation_score': 0.0201}
2026-05-03 08:37:32,211 INFO Regime score epoch 21/50 — tr=0.0034 va=0.0008
2026-05-03 08:37:32,466 INFO Regime score epoch 22/50 — tr=0.0034 va=0.0008
2026-05-03 08:37:32,725 INFO Regime score epoch 23/50 — tr=0.0034 va=0.0008
2026-05-03 08:37:32,985 INFO Regime score epoch 24/50 — tr=0.0034 va=0.0008
2026-05-03 08:37:33,238 INFO Regime score epoch 25/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.0176, 'range_score': 0.0325, 'chop_score': 0.0198, 'volatility_percentile': 0.0143, 'consolidation_score': 0.0201}
2026-05-03 08:37:33,499 INFO Regime score epoch 26/50 — tr=0.0034 va=0.0008
2026-05-03 08:37:33,750 INFO Regime score epoch 27/50 — tr=0.0034 va=0.0008
2026-05-03 08:37:34,010 INFO Regime score epoch 28/50 — tr=0.0034 va=0.0008
2026-05-03 08:37:34,266 INFO Regime score epoch 29/50 — tr=0.0034 va=0.0008
2026-05-03 08:37:34,526 INFO Regime score epoch 30/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.0177, 'range_score': 0.0327, 'chop_score': 0.0198, 'volatility_percentile': 0.0144, 'consolidation_score': 0.0201}
2026-05-03 08:37:34,784 INFO Regime score epoch 31/50 — tr=0.0034 va=0.0008
2026-05-03 08:37:35,043 INFO Regime score epoch 32/50 — tr=0.0034 va=0.0008
2026-05-03 08:37:35,296 INFO Regime score epoch 33/50 — tr=0.0034 va=0.0008
2026-05-03 08:37:35,550 INFO Regime score epoch 34/50 — tr=0.0034 va=0.0008
2026-05-03 08:37:35,808 INFO Regime score epoch 35/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.0174, 'range_score': 0.0324, 'chop_score': 0.0194, 'volatility_percentile': 0.0141, 'consolidation_score': 0.02}
2026-05-03 08:37:36,070 INFO Regime score epoch 36/50 — tr=0.0034 va=0.0008
2026-05-03 08:37:36,338 INFO Regime score epoch 37/50 — tr=0.0034 va=0.0008
2026-05-03 08:37:36,592 INFO Regime score epoch 38/50 — tr=0.0034 va=0.0008
2026-05-03 08:37:36,840 INFO Regime score epoch 39/50 — tr=0.0034 va=0.0008
2026-05-03 08:37:37,089 INFO Regime score epoch 40/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.0174, 'range_score': 0.0324, 'chop_score': 0.0195, 'volatility_percentile': 0.014, 'consolidation_score': 0.0197}
2026-05-03 08:37:37,338 INFO Regime score epoch 41/50 — tr=0.0034 va=0.0008
2026-05-03 08:37:37,595 INFO Regime score epoch 42/50 — tr=0.0034 va=0.0008
2026-05-03 08:37:37,860 INFO Regime score epoch 43/50 — tr=0.0034 va=0.0008
2026-05-03 08:37:38,123 INFO Regime score epoch 44/50 — tr=0.0034 va=0.0008
2026-05-03 08:37:38,377 INFO Regime score epoch 45/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.0175, 'range_score': 0.0323, 'chop_score': 0.0193, 'volatility_percentile': 0.0139, 'consolidation_score': 0.0198}
2026-05-03 08:37:38,628 INFO Regime score epoch 46/50 — tr=0.0034 va=0.0008
2026-05-03 08:37:38,884 INFO Regime score epoch 47/50 — tr=0.0034 va=0.0008
2026-05-03 08:37:39,151 INFO Regime score epoch 48/50 — tr=0.0034 va=0.0008
2026-05-03 08:37:39,409 INFO Regime score epoch 49/50 — tr=0.0034 va=0.0008
2026-05-03 08:37:39,668 INFO Regime score epoch 50/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.0174, 'range_score': 0.0326, 'chop_score': 0.0195, 'volatility_percentile': 0.014, 'consolidation_score': 0.0198}
2026-05-03 08:37:39,705 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0174, 'range_score': 0.0321, 'chop_score': 0.0193, 'volatility_percentile': 0.014, 'consolidation_score': 0.0198} mse={'trend_score': 0.00054, 'range_score': 0.00171, 'chop_score': 0.00061, 'volatility_percentile': 0.00035, 'consolidation_score': 0.00089} corr={'trend_score': 0.9948, 'range_score': 0.9611, 'chop_score': 0.9925, 'volatility_percentile': 0.9964, 'consolidation_score': 0.9899} pred_std={'trend_score': 0.2232, 'range_score': 0.1332, 'chop_score': 0.1836, 'volatility_percentile': 0.2209, 'consolidation_score': 0.2065} target_std={'trend_score': 0.2249, 'range_score': 0.1469, 'chop_score': 0.1924, 'volatility_percentile': 0.2182, 'consolidation_score': 0.2098}
2026-05-03 08:37:39,710 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-03 08:37:39,710 INFO Regime phase LTF train fold=fold_002: 13.1s
2026-05-03 08:37:39,810 INFO Regime LTF complete fold=fold_002: score_accuracy=0.980, train=128142 val=63241 mae={'trend_score': 0.0174, 'range_score': 0.0321, 'chop_score': 0.0193, 'volatility_percentile': 0.014, 'consolidation_score': 0.0198}
2026-05-03 08:37:39,812 INFO Loaded XAUUSD/1H split=train fold=fold_002: 11775 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:37:39,964 INFO Regime[1H mode=ltf_behaviour fold=fold_002] LTF score diagnostics on XAUUSD:
{'trend_score': {'mean': 0.4815, 'q10': 0.1762, 'q50': 0.4731, 'q90': 0.7982}, 'range_score': {'mean': 0.2428, 'q10': 0.0514, 'q50': 0.2289, 'q90': 0.4497}, 'chop_score': {'mean': 0.4776, 'q10': 0.2185, 'q50': 0.4756, 'q90': 0.7418}, 'volatility_percentile': {'mean': 0.3668, 'q10': 0.0788, 'q50': 0.3514, 'q90': 0.6764}, 'consolidation_score': {'mean': 0.1986, 'q10': 0.0, 'q50': 0.1291, 'q90': 0.5425}}
2026-05-03 08:37:39,967 INFO Regime retrain total: 149.5s (722582 train+val samples)
2026-05-03 08:37:39,984 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-03 08:37:39,984 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-03 08:37:39,984 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-03 08:37:39,984 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-05-03 08:37:39,985 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-05-03 08:37:39,985 INFO Retrain complete. Total wall-clock: 149.5s
2026-05-03 08:37:40,955 INFO Model regime: SUCCESS
2026-05-03 08:37:40,955 INFO --- Training gru ---
2026-05-03 08:37:40,955 INFO Running retrain --model gru
2026-05-03 08:37:41,406 INFO retrain environment: KAGGLE
2026-05-03 08:37:42,992 INFO Device: CUDA (2 GPU(s))
2026-05-03 08:37:43,003 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-03 08:37:43,004 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-03 08:37:43,004 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-03 08:37:43,004 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-03 08:37:43,004 INFO Retrain data split: train
2026-05-03 08:37:43,004 INFO Retrain rolling fold selector: latest
2026-05-03 08:37:43,005 INFO === GRU-LSTM retrain (timeframes: ['15M']) ===
2026-05-03 08:37:43,148 INFO NumExpr defaulting to 4 threads.
2026-05-03 08:37:43,335 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-03 08:37:43,335 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-03 08:37:43,335 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-03 08:37:43,581 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-05-03 08:37:43,581 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['15M']
2026-05-03 08:37:43,583 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260503_083743
2026-05-03 08:37:43,586 INFO GRU feature contract unchanged (input_size=71) — incremental retrain
2026-05-03 08:37:43,586 INFO GRU warm start enabled from existing weights: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 08:37:43,734 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:43,753 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:43,767 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:43,774 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:43,816 INFO Split boundaries loaded fold=fold_002/3 — train 2020-01-06→2022-01-03  val 2022-01-04→2023-01-03  test 2023-08-07→2025-08-05
2026-05-03 08:37:43,819 INFO Loaded AUDUSD/15M split=train fold=latest: 46766 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:37:44,038 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:44,056 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:44,070 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:44,077 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:44,112 INFO Loaded EURGBP/15M split=train fold=latest: 46759 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:37:44,314 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:44,335 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:44,349 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:44,355 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:44,390 INFO Loaded EURJPY/15M split=train fold=latest: 46767 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:37:44,594 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:44,612 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:44,625 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:44,641 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:44,682 INFO Loaded EURUSD/15M split=train fold=latest: 46766 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:37:44,899 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:44,919 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:44,933 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:44,940 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:44,975 INFO Loaded GBPJPY/15M split=train fold=latest: 46765 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:37:45,173 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:45,191 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:45,204 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:45,211 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:45,246 INFO Loaded GBPUSD/15M split=train fold=latest: 46764 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:37:45,429 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:37:45,445 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:37:45,457 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:37:45,464 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 08:37:45,492 INFO Loaded NZDUSD/15M split=train fold=latest: 46766 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:37:45,680 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:45,697 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:45,710 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:45,716 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:45,749 INFO Loaded USDCAD/15M split=train fold=latest: 46767 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:37:45,946 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:45,964 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:45,977 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:45,983 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:46,017 INFO Loaded USDCHF/15M split=train fold=latest: 46763 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:37:46,220 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:46,239 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:46,253 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:46,259 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 08:37:46,295 INFO Loaded USDJPY/15M split=train fold=latest: 46768 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:37:46,614 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:37:46,638 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:37:46,653 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:37:46,662 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 08:37:46,723 INFO Loaded XAUUSD/15M split=train fold=latest: 47096 bars (2020-01-06 → 2022-01-03)
2026-05-03 08:37:46,847 INFO train_multi: 11 segments, ~500055 total bars
2026-05-03 08:37:46,848 INFO train_multi: training ALL 11 segments across TFs ['15M'] in one combined pass
2026-05-03 08:37:46,848 INFO train_multi: building combined dataset for TF=ALL (11 segments)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:37:52,467 INFO train_multi TF=ALL: 499725 sequences across 11 segments
2026-05-03 08:37:52,467 INFO train_multi TF=ALL: estimated peak RAM = 8515 MB (train=399775 val=99950 n_feat=71 seq_len=30)
2026-05-03 08:37:53,530 INFO train_multi TF=ALL: train=399775 val=99950 (4264 MB tensors)
2026-05-03 08:37:57,482 INFO train_multi TF=ALL: warm-start detected — using CosineAnnealingLR (lr=3e-05, patience=12)
2026-05-03 08:38:10,723 INFO train_multi TF=ALL epoch 1/50 train=0.6518 val=0.6489 dir_acc=0.634 dir_n=99950
2026-05-03 08:38:10,728 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 08:38:10,728 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 08:38:10,728 INFO train_multi TF=ALL: new best val=0.6489 — saved
2026-05-03 08:38:21,857 INFO train_multi TF=ALL epoch 2/50 train=0.6516 val=0.6487 dir_acc=0.634 dir_n=99950
2026-05-03 08:38:21,862 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 08:38:21,862 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 08:38:21,862 INFO train_multi TF=ALL: new best val=0.6487 — saved
2026-05-03 08:38:32,957 INFO train_multi TF=ALL epoch 3/50 train=0.6516 val=0.6487 dir_acc=0.634 dir_n=99950
2026-05-03 08:38:32,962 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 08:38:32,962 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 08:38:32,962 INFO train_multi TF=ALL: new best val=0.6487 — saved
2026-05-03 08:38:44,058 INFO train_multi TF=ALL epoch 4/50 train=0.6515 val=0.6487 dir_acc=0.634 dir_n=99950
2026-05-03 08:38:44,064 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 08:38:44,064 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 08:38:44,064 INFO train_multi TF=ALL: new best val=0.6487 — saved
2026-05-03 08:38:55,030 INFO train_multi TF=ALL epoch 5/50 train=0.6514 val=0.6489 dir_acc=0.634 dir_n=99950
2026-05-03 08:39:06,051 INFO train_multi TF=ALL epoch 6/50 train=0.6514 val=0.6487 dir_acc=0.634 dir_n=99950
2026-05-03 08:39:06,056 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 08:39:06,056 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 08:39:06,056 INFO train_multi TF=ALL: new best val=0.6487 — saved
2026-05-03 08:39:16,930 INFO train_multi TF=ALL epoch 7/50 train=0.6511 val=0.6488 dir_acc=0.634 dir_n=99950
2026-05-03 08:39:27,978 INFO train_multi TF=ALL epoch 8/50 train=0.6512 val=0.6488 dir_acc=0.634 dir_n=99950
2026-05-03 08:39:38,901 INFO train_multi TF=ALL epoch 9/50 train=0.6512 val=0.6487 dir_acc=0.634 dir_n=99950
2026-05-03 08:39:38,906 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 08:39:38,906 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 08:39:38,906 INFO train_multi TF=ALL: new best val=0.6487 — saved
2026-05-03 08:39:49,988 INFO train_multi TF=ALL epoch 10/50 train=0.6513 val=0.6487 dir_acc=0.634 dir_n=99950
2026-05-03 08:40:00,993 INFO train_multi TF=ALL epoch 11/50 train=0.6511 val=0.6487 dir_acc=0.634 dir_n=99950
2026-05-03 08:40:12,011 INFO train_multi TF=ALL epoch 12/50 train=0.6511 val=0.6486 dir_acc=0.635 dir_n=99950
2026-05-03 08:40:12,016 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 08:40:12,016 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 08:40:12,016 INFO train_multi TF=ALL: new best val=0.6486 — saved
2026-05-03 08:40:23,009 INFO train_multi TF=ALL epoch 13/50 train=0.6509 val=0.6487 dir_acc=0.635 dir_n=99950
2026-05-03 08:40:34,027 INFO train_multi TF=ALL epoch 14/50 train=0.6508 val=0.6487 dir_acc=0.634 dir_n=99950
2026-05-03 08:40:45,049 INFO train_multi TF=ALL epoch 15/50 train=0.6508 val=0.6486 dir_acc=0.635 dir_n=99950
2026-05-03 08:40:55,965 INFO train_multi TF=ALL epoch 16/50 train=0.6509 val=0.6486 dir_acc=0.634 dir_n=99950
2026-05-03 08:40:55,971 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 08:40:55,971 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 08:40:55,971 INFO train_multi TF=ALL: new best val=0.6486 — saved
2026-05-03 08:41:07,038 INFO train_multi TF=ALL epoch 17/50 train=0.6508 val=0.6486 dir_acc=0.634 dir_n=99950
2026-05-03 08:41:18,658 INFO train_multi TF=ALL epoch 18/50 train=0.6509 val=0.6486 dir_acc=0.635 dir_n=99950
2026-05-03 08:41:30,415 INFO train_multi TF=ALL epoch 19/50 train=0.6505 val=0.6487 dir_acc=0.635 dir_n=99950
2026-05-03 08:41:42,257 INFO train_multi TF=ALL epoch 20/50 train=0.6506 val=0.6486 dir_acc=0.635 dir_n=99950
2026-05-03 08:41:42,262 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 08:41:42,263 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 08:41:42,263 INFO train_multi TF=ALL: new best val=0.6486 — saved
2026-05-03 08:41:53,460 INFO train_multi TF=ALL epoch 21/50 train=0.6508 val=0.6486 dir_acc=0.635 dir_n=99950
2026-05-03 08:42:04,430 INFO train_multi TF=ALL epoch 22/50 train=0.6504 val=0.6485 dir_acc=0.635 dir_n=99950
2026-05-03 08:42:04,435 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 08:42:04,435 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 08:42:04,435 INFO train_multi TF=ALL: new best val=0.6485 — saved
2026-05-03 08:42:15,502 INFO train_multi TF=ALL epoch 23/50 train=0.6506 val=0.6486 dir_acc=0.634 dir_n=99950
2026-05-03 08:42:26,457 INFO train_multi TF=ALL epoch 24/50 train=0.6506 val=0.6486 dir_acc=0.635 dir_n=99950
2026-05-03 08:42:37,424 INFO train_multi TF=ALL epoch 25/50 train=0.6506 val=0.6485 dir_acc=0.635 dir_n=99950
2026-05-03 08:42:37,429 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 08:42:37,429 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 08:42:37,429 INFO train_multi TF=ALL: new best val=0.6485 — saved
2026-05-03 08:42:48,547 INFO train_multi TF=ALL epoch 26/50 train=0.6504 val=0.6485 dir_acc=0.635 dir_n=99950
2026-05-03 08:42:48,552 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 08:42:48,552 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 08:42:48,552 INFO train_multi TF=ALL: new best val=0.6485 — saved
2026-05-03 08:42:59,584 INFO train_multi TF=ALL epoch 27/50 train=0.6504 val=0.6485 dir_acc=0.635 dir_n=99950
2026-05-03 08:43:10,598 INFO train_multi TF=ALL epoch 28/50 train=0.6503 val=0.6485 dir_acc=0.635 dir_n=99950
2026-05-03 08:43:10,603 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 08:43:10,603 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 08:43:10,603 INFO train_multi TF=ALL: new best val=0.6485 — saved
2026-05-03 08:43:21,490 INFO train_multi TF=ALL epoch 29/50 train=0.6502 val=0.6485 dir_acc=0.634 dir_n=99950
2026-05-03 08:43:32,491 INFO train_multi TF=ALL epoch 30/50 train=0.6504 val=0.6485 dir_acc=0.635 dir_n=99950
2026-05-03 08:43:43,395 INFO train_multi TF=ALL epoch 31/50 train=0.6503 val=0.6485 dir_acc=0.635 dir_n=99950
2026-05-03 08:43:54,375 INFO train_multi TF=ALL epoch 32/50 train=0.6501 val=0.6485 dir_acc=0.635 dir_n=99950
2026-05-03 08:44:05,321 INFO train_multi TF=ALL epoch 33/50 train=0.6502 val=0.6485 dir_acc=0.635 dir_n=99950
2026-05-03 08:44:16,273 INFO train_multi TF=ALL epoch 34/50 train=0.6504 val=0.6485 dir_acc=0.635 dir_n=99950
2026-05-03 08:44:27,379 INFO train_multi TF=ALL epoch 35/50 train=0.6502 val=0.6485 dir_acc=0.635 dir_n=99950
2026-05-03 08:44:38,446 INFO train_multi TF=ALL epoch 36/50 train=0.6502 val=0.6485 dir_acc=0.635 dir_n=99950
2026-05-03 08:44:38,451 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 08:44:38,451 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 08:44:38,451 INFO train_multi TF=ALL: new best val=0.6485 — saved
2026-05-03 08:44:49,497 INFO train_multi TF=ALL epoch 37/50 train=0.6503 val=0.6485 dir_acc=0.635 dir_n=99950
2026-05-03 08:45:00,574 INFO train_multi TF=ALL epoch 38/50 train=0.6502 val=0.6485 dir_acc=0.635 dir_n=99950
2026-05-03 08:45:11,685 INFO train_multi TF=ALL epoch 39/50 train=0.6503 val=0.6485 dir_acc=0.635 dir_n=99950
2026-05-03 08:45:22,747 INFO train_multi TF=ALL epoch 40/50 train=0.6503 val=0.6484 dir_acc=0.635 dir_n=99950
2026-05-03 08:45:22,752 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 08:45:22,752 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 08:45:22,752 INFO train_multi TF=ALL: new best val=0.6484 — saved
2026-05-03 08:45:33,689 INFO train_multi TF=ALL epoch 41/50 train=0.6501 val=0.6485 dir_acc=0.635 dir_n=99950
2026-05-03 08:45:44,696 INFO train_multi TF=ALL epoch 42/50 train=0.6501 val=0.6484 dir_acc=0.635 dir_n=99950
2026-05-03 08:45:55,826 INFO train_multi TF=ALL epoch 43/50 train=0.6502 val=0.6484 dir_acc=0.635 dir_n=99950
2026-05-03 08:46:06,883 INFO train_multi TF=ALL epoch 44/50 train=0.6501 val=0.6484 dir_acc=0.635 dir_n=99950
2026-05-03 08:46:17,944 INFO train_multi TF=ALL epoch 45/50 train=0.6502 val=0.6484 dir_acc=0.635 dir_n=99950
2026-05-03 08:46:29,102 INFO train_multi TF=ALL epoch 46/50 train=0.6503 val=0.6484 dir_acc=0.635 dir_n=99950
2026-05-03 08:46:40,081 INFO train_multi TF=ALL epoch 47/50 train=0.6502 val=0.6484 dir_acc=0.635 dir_n=99950
2026-05-03 08:46:51,159 INFO train_multi TF=ALL epoch 48/50 train=0.6503 val=0.6484 dir_acc=0.635 dir_n=99950
2026-05-03 08:47:02,132 INFO train_multi TF=ALL epoch 49/50 train=0.6503 val=0.6484 dir_acc=0.635 dir_n=99950
2026-05-03 08:47:13,124 INFO train_multi TF=ALL epoch 50/50 train=0.6501 val=0.6484 dir_acc=0.635 dir_n=99950
2026-05-03 08:47:13,267 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-03 08:47:13,267 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-03 08:47:13,267 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-03 08:47:13,267 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-05-03 08:47:13,268 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-05-03 08:47:13,268 INFO Retrain complete. Total wall-clock: 570.3s
2026-05-03 08:47:15,062 INFO Model gru: SUCCESS
2026-05-03 08:47:15,062 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 08:47:15,063 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-03 08:47:15,063 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-03 08:47:15,063 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-05-03 08:47:15,063 INFO   [DORMANT] rl_ppo — RL_ENABLED=false, skipping
2026-05-03 08:47:15,063 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-05-03 08:47:15,063 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer']
2026-05-03 08:47:15,066 INFO Saved 18 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Clean Quality/RL source: Backtest on train window ===
  START Train-window backtest for Quality/RL labels
2026-05-03 08:47:15,838 INFO === STEP 6: BACKTEST (train) ===
2026-05-03 08:47:15,839 INFO BT_WINDOW=train — train-window backtest: 2020-01-06 → 2022-01-03 (clean Quality/RL labels)
2026-05-03 08:47:15,839 INFO ================================================================
  ROUND 0 / 3
================================================================
2026-05-03 08:47:15,839 INFO Round 0 — running backtest: 2020-01-06 → 2022-01-03 (ml_trader, shared ML cache)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:49:30,481 ERROR ML cache: sequence feature build failed for USDJPY: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
2026-05-03 08:49:30,481 ERROR _precompute_ml_cache failed for USDJPY: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
2026-05-03 08:49:31,053 ERROR ML cache: sequence feature build failed for GBPUSD: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
2026-05-03 08:49:31,059 ERROR _precompute_ml_cache failed for GBPUSD: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:49:31,543 ERROR ML cache: sequence feature build failed for AUDUSD: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
2026-05-03 08:49:31,543 ERROR _precompute_ml_cache failed for AUDUSD: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:49:31,812 ERROR ML cache: sequence feature build failed for EURUSD: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
2026-05-03 08:49:31,819 ERROR _precompute_ml_cache failed for EURUSD: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:49:38,294 ERROR ML cache: sequence feature build failed for NZDUSD: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
2026-05-03 08:49:38,299 ERROR _precompute_ml_cache failed for NZDUSD: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:49:39,257 ERROR ML cache: sequence feature build failed for USDCAD: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
2026-05-03 08:49:39,257 ERROR _precompute_ml_cache failed for USDCAD: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:49:39,901 ERROR ML cache: sequence feature build failed for USDCHF: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
2026-05-03 08:49:39,905 ERROR _precompute_ml_cache failed for USDCHF: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:49:40,197 ERROR ML cache: sequence feature build failed for EURGBP: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
2026-05-03 08:49:40,199 ERROR _precompute_ml_cache failed for EURGBP: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:49:45,717 ERROR ML cache: sequence feature build failed for XAUUSD: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
2026-05-03 08:49:45,717 ERROR _precompute_ml_cache failed for XAUUSD: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
2026-05-03 08:49:45,831 ERROR ML cache: sequence feature build failed for GBPJPY: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
2026-05-03 08:49:45,832 ERROR _precompute_ml_cache failed for GBPJPY: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
2026-05-03 08:49:45,906 ERROR ML cache: sequence feature build failed for EURJPY: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
2026-05-03 08:49:45,906 ERROR _precompute_ml_cache failed for EURJPY: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
Traceback (most recent call last):
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 2344, in _backtest_trader
    _, c = fut.result()
           ^^^^^^^^^^^^
  File "/usr/lib/python3.12/concurrent/futures/_base.py", line 449, in result
    return self.__get_result()
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/concurrent/futures/_base.py", line 401, in __get_result
    raise self._exception
  File "/usr/lib/python3.12/concurrent/futures/thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 2327, in _build_cache_sym
    return sym, _precompute_ml_cache(
                ^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 1567, in _precompute_ml_cache
    feat_df = fe._build_sequence_df(df, htf, symbol=symbol)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py", line 724, in _build_sequence_df
    out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
                             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py", line 700, in _htf_series
    raise ValueError(
ValueError: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 3804, in <module>
    main()
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 3618, in main
    result = _backtest_trader("ml_trader", symbols, pm, bt_start, bt_end,
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 2348, in _backtest_trader
    raise RuntimeError(f"ML cache build failed for {sym}: {exc}") from exc
RuntimeError: ML cache build failed for USDJPY: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
2026-05-03 08:49:46,560 ERROR Backtest failed (rc=1) — check trading-engine/logs/backtest_*.log
2026-05-03 08:49:46,561 ERROR Round 0 backtest failed: backtest exited 1
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/kaggle/working/Multi-Bot/trading-system/kaggle_train.py in <module>
    309 
    310 print("\n=== Clean Quality/RL source: Backtest on train window ===")
--> 311 run_step(
    312     "Train-window backtest for Quality/RL labels",
    313     "step6_backtest.py",

/kaggle/working/Multi-Bot/trading-system/kaggle_train.py in run_step(name, script, done_check, extra_env)
    186     )
    187     if result.returncode != 0:
--> 188         raise RuntimeError(f"{name} FAILED (exit {result.returncode})")
    189     print(f"  DONE  {name}")
    190 

RuntimeError: Train-window backtest for Quality/RL labels FAILED (exit 1)
