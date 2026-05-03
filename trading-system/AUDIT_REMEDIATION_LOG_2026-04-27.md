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
2026-05-03 07:46:39,127 INFO Loading feature-engineered data...
2026-05-03 07:46:39,865 INFO Loaded 221743 rows, 202 features
2026-05-03 07:46:39,867 INFO Data span: 2016-01-04 -> 2025-08-05  (9.6 years)
2026-05-03 07:46:39,872 INFO Fold 000 train 2016-01-04 -> 2018-01-03 (47088 bars), val 2018-01-04 -> 2019-01-03 (23448 bars)
2026-05-03 07:46:39,872 INFO Fold 001 train 2018-01-04 -> 2020-01-03 (46825 bars), val 2020-01-06 -> 2020-12-31 (23259 bars)
2026-05-03 07:46:39,872 INFO Fold 002 train 2020-01-06 -> 2022-01-03 (46766 bars), val 2022-01-04 -> 2023-01-03 (23588 bars)
2026-05-03 07:46:39,873 INFO No leakage confirmed: every fold ends before final 2-year blind test

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
2026-05-03 07:46:43,426 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-05-03 07:46:43,426 INFO --- Training regime ---
2026-05-03 07:46:43,426 INFO Running retrain --model regime
2026-05-03 07:46:43,635 INFO retrain environment: KAGGLE
2026-05-03 07:46:45,207 INFO Device: CUDA (2 GPU(s))
2026-05-03 07:46:45,218 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-03 07:46:45,218 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-03 07:46:45,218 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-03 07:46:45,221 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-03 07:46:45,221 INFO Retrain data split: train
2026-05-03 07:46:45,222 INFO Retrain rolling fold selector: latest
2026-05-03 07:46:45,222 INFO === RegimeClassifier retrain (hierarchical: HTF 2-score bias + LTF 5-score behaviour) ===
2026-05-03 07:46:45,393 INFO NumExpr defaulting to 4 threads.
2026-05-03 07:46:45,605 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-03 07:46:45,605 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-03 07:46:45,605 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-03 07:46:45,605 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-03 07:46:45,657 INFO Regime rolling folds selected: ['fold_000', 'fold_001', 'fold_002']
2026-05-03 07:46:45,657 INFO === Regime rolling fold 1/3: fold_000 ===
2026-05-03 07:46:45,657 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-03 07:46:45,697 INFO Split boundaries loaded fold=fold_000/3 — train 2016-01-04→2018-01-03  val 2018-01-04→2019-01-03  test 2023-08-07→2025-08-05
2026-05-03 07:46:45,698 INFO Loaded AUDUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-03 07:46:45,714 INFO Loaded EURGBP/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-03 07:46:45,728 INFO Loaded EURJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-03 07:46:45,743 INFO Loaded EURUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-03 07:46:45,759 INFO Loaded GBPJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-03 07:46:45,773 INFO Loaded GBPUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-03 07:46:45,788 INFO Loaded NZDUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-03 07:46:45,802 INFO Loaded USDCAD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-03 07:46:45,817 INFO Loaded USDCHF/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-03 07:46:45,832 INFO Loaded USDJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-03 07:46:45,850 INFO Loaded XAUUSD/4H split=train fold=fold_000: 3204 bars (2016-01-04 → 2018-01-03)
2026-05-03 07:46:45,981 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:46,027 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:46,044 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:46,044 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:46,051 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:46,052 INFO Loaded AUDUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:46:46,280 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 92, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 2876}  ambiguous=1700 (total=3023) horizon=12
2026-05-03 07:46:46,283 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected AUDUSD — 2973 samples (group=dollar) score_means={'bias_up_score': 0.0309, 'bias_down_score': 0.0185} labels={'BIAS_UP': 92, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 2826} clean={'BIAS_UP': 92, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 1152}
2026-05-03 07:46:46,447 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:46,479 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:46,498 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:46,498 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:46,506 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:46,506 INFO Loaded EURGBP/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:46:46,712 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 156, 'BIAS_DOWN': 71, 'BIAS_NEUTRAL': 2796}  ambiguous=1710 (total=3023) horizon=12
2026-05-03 07:46:46,714 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected EURGBP — 2973 samples (group=cross) score_means={'bias_up_score': 0.0525, 'bias_down_score': 0.0239} labels={'BIAS_UP': 156, 'BIAS_DOWN': 71, 'BIAS_NEUTRAL': 2746} clean={'BIAS_UP': 156, 'BIAS_DOWN': 71, 'BIAS_NEUTRAL': 1071}
2026-05-03 07:46:46,882 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:46,920 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:46,939 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:46,940 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:46,947 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:46,948 INFO Loaded EURJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:46:47,153 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 95, 'BIAS_DOWN': 63, 'BIAS_NEUTRAL': 2865}  ambiguous=1742 (total=3023) horizon=12
2026-05-03 07:46:47,155 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected EURJPY — 2973 samples (group=cross) score_means={'bias_up_score': 0.032, 'bias_down_score': 0.0212} labels={'BIAS_UP': 95, 'BIAS_DOWN': 63, 'BIAS_NEUTRAL': 2815} clean={'BIAS_UP': 95, 'BIAS_DOWN': 63, 'BIAS_NEUTRAL': 1099}
2026-05-03 07:46:47,314 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:47,351 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:47,370 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:47,370 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:47,377 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:47,378 INFO Loaded EURUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:46:47,590 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 98, 'BIAS_DOWN': 57, 'BIAS_NEUTRAL': 2868}  ambiguous=1742 (total=3023) horizon=12
2026-05-03 07:46:47,593 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected EURUSD — 2973 samples (group=dollar) score_means={'bias_up_score': 0.033, 'bias_down_score': 0.0192} labels={'BIAS_UP': 98, 'BIAS_DOWN': 57, 'BIAS_NEUTRAL': 2818} clean={'BIAS_UP': 98, 'BIAS_DOWN': 57, 'BIAS_NEUTRAL': 1105}
2026-05-03 07:46:47,760 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:47,796 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:47,822 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:47,822 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:47,829 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:47,830 INFO Loaded GBPJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:46:48,043 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 164, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2758}  ambiguous=1723 (total=3023) horizon=12
2026-05-03 07:46:48,046 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected GBPJPY — 2973 samples (group=cross) score_means={'bias_up_score': 0.0552, 'bias_down_score': 0.034} labels={'BIAS_UP': 164, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2708} clean={'BIAS_UP': 164, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 1019}
2026-05-03 07:46:48,211 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:48,247 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:48,266 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:48,266 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:48,274 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:48,275 INFO Loaded GBPUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:46:48,483 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 79, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2843}  ambiguous=1759 (total=3023) horizon=12
2026-05-03 07:46:48,486 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected GBPUSD — 2973 samples (group=dollar) score_means={'bias_up_score': 0.0266, 'bias_down_score': 0.034} labels={'BIAS_UP': 79, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2793} clean={'BIAS_UP': 79, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 1073}
2026-05-03 07:46:48,623 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:46:48,656 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:46:48,677 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:46:48,677 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:46:48,684 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:46:48,685 INFO Loaded NZDUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:46:48,911 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 54, 'BIAS_DOWN': 54, 'BIAS_NEUTRAL': 2915}  ambiguous=1779 (total=3023) horizon=12
2026-05-03 07:46:48,915 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected NZDUSD — 2973 samples (group=dollar) score_means={'bias_up_score': 0.0182, 'bias_down_score': 0.0182} labels={'BIAS_UP': 54, 'BIAS_DOWN': 54, 'BIAS_NEUTRAL': 2865} clean={'BIAS_UP': 54, 'BIAS_DOWN': 54, 'BIAS_NEUTRAL': 1117}
2026-05-03 07:46:49,073 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:49,109 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:49,127 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:49,128 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:49,135 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:49,135 INFO Loaded USDCAD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:46:49,340 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 89, 'BIAS_DOWN': 133, 'BIAS_NEUTRAL': 2801}  ambiguous=1770 (total=3023) horizon=12
2026-05-03 07:46:49,343 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected USDCAD — 2973 samples (group=dollar) score_means={'bias_up_score': 0.0299, 'bias_down_score': 0.0447} labels={'BIAS_UP': 89, 'BIAS_DOWN': 133, 'BIAS_NEUTRAL': 2751} clean={'BIAS_UP': 89, 'BIAS_DOWN': 133, 'BIAS_NEUTRAL': 1016}
2026-05-03 07:46:49,493 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:49,525 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:49,543 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:49,544 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:49,550 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:49,551 INFO Loaded USDCHF/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:46:49,760 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 64, 'BIAS_DOWN': 52, 'BIAS_NEUTRAL': 2907}  ambiguous=1741 (total=3023) horizon=12
2026-05-03 07:46:49,763 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected USDCHF — 2973 samples (group=dollar) score_means={'bias_up_score': 0.0215, 'bias_down_score': 0.0175} labels={'BIAS_UP': 64, 'BIAS_DOWN': 52, 'BIAS_NEUTRAL': 2857} clean={'BIAS_UP': 64, 'BIAS_DOWN': 52, 'BIAS_NEUTRAL': 1148}
2026-05-03 07:46:49,918 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:49,962 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:49,983 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:49,983 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:49,991 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:49,991 INFO Loaded USDJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:46:50,217 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 102, 'BIAS_DOWN': 78, 'BIAS_NEUTRAL': 2843}  ambiguous=1762 (total=3023) horizon=12
2026-05-03 07:46:50,220 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected USDJPY — 2973 samples (group=dollar) score_means={'bias_up_score': 0.0343, 'bias_down_score': 0.0262} labels={'BIAS_UP': 102, 'BIAS_DOWN': 78, 'BIAS_NEUTRAL': 2793} clean={'BIAS_UP': 102, 'BIAS_DOWN': 78, 'BIAS_NEUTRAL': 1058}
2026-05-03 07:46:50,500 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:46:50,564 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:46:50,587 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:46:50,588 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:46:50,598 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:46:50,599 INFO Loaded XAUUSD/4H split=train fold=fold_000: 3204 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:46:50,828 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 2845}  ambiguous=1801 (total=3204) horizon=12
2026-05-03 07:46:50,831 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected XAUUSD — 3154 samples (group=gold) score_means={'bias_up_score': 0.0672, 'bias_down_score': 0.0466} labels={'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 2795} clean={'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 1029}
2026-05-03 07:46:50,892 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 415, 'BIAS_DOWN': 235, 'BIAS_NEUTRAL': 8269}, 'dollar': {'BIAS_UP': 578, 'BIAS_DOWN': 530, 'BIAS_NEUTRAL': 19703}, 'gold': {'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 2795}}
2026-05-03 07:46:50,893 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0465, 'bias_down_score': 0.0263}, 'dollar': {'bias_up_score': 0.0278, 'bias_down_score': 0.0255}, 'gold': {'bias_up_score': 0.0672, 'bias_down_score': 0.0466}}
2026-05-03 07:46:50,893 INFO Regime[4H mode=htf_bias] label distribution by year: {2016: {'BIAS_UP': 485, 'BIAS_DOWN': 511, 'BIAS_NEUTRAL': 15101}, 2017: {'BIAS_UP': 717, 'BIAS_DOWN': 401, 'BIAS_NEUTRAL': 15515}, 2018: {'BIAS_UP': 3, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 151}}
2026-05-03 07:46:50,893 INFO Regime[4H mode=htf_bias] score means by year: {2016: {'bias_up_score': 0.0301, 'bias_down_score': 0.0317}, 2017: {'bias_up_score': 0.0431, 'bias_down_score': 0.0241}, 2018: {'bias_up_score': 0.0195, 'bias_down_score': 0.0}}
2026-05-03 07:46:50,937 INFO Loaded AUDUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-03 07:46:50,938 INFO Loaded EURGBP/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-03 07:46:50,939 INFO Loaded EURJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-03 07:46:50,940 INFO Loaded EURUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-03 07:46:50,940 INFO Loaded GBPJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-03 07:46:50,941 INFO Loaded GBPUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-03 07:46:50,942 INFO Loaded NZDUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-03 07:46:50,943 INFO Loaded USDCAD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-03 07:46:50,943 INFO Loaded USDCHF/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-03 07:46:50,944 INFO Loaded USDJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-03 07:46:50,945 INFO Loaded XAUUSD/4H split=val fold=fold_000: 1600 bars (2018-01-04 → 2019-01-03)
2026-05-03 07:46:50,951 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:50,953 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:50,954 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:50,954 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:50,955 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:50,955 INFO Loaded AUDUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:46:51,151 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 5, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1448}  ambiguous=896 (total=1506) horizon=12
2026-05-03 07:46:51,153 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected AUDUSD — 1456 samples (group=dollar) score_means={'bias_up_score': 0.0034, 'bias_down_score': 0.0364} labels={'BIAS_UP': 5, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1398} clean={'BIAS_UP': 5, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 531}
2026-05-03 07:46:51,219 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:51,221 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:51,222 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:51,222 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:51,222 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:51,223 INFO Loaded EURGBP/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:46:51,408 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 12, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1453}  ambiguous=868 (total=1506) horizon=12
2026-05-03 07:46:51,410 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected EURGBP — 1456 samples (group=cross) score_means={'bias_up_score': 0.0082, 'bias_down_score': 0.0282} labels={'BIAS_UP': 12, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1403} clean={'BIAS_UP': 12, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 575}
2026-05-03 07:46:51,474 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:51,476 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:51,477 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:51,477 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:51,477 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:51,478 INFO Loaded EURJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:46:51,661 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 20, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1445}  ambiguous=874 (total=1506) horizon=12
2026-05-03 07:46:51,664 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected EURJPY — 1456 samples (group=cross) score_means={'bias_up_score': 0.0137, 'bias_down_score': 0.0282} labels={'BIAS_UP': 20, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1395} clean={'BIAS_UP': 20, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 555}
2026-05-03 07:46:51,726 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:51,728 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:51,729 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:51,730 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:51,730 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:51,731 INFO Loaded EURUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:46:51,919 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 24, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 1415}  ambiguous=876 (total=1506) horizon=12
2026-05-03 07:46:51,922 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected EURUSD — 1456 samples (group=dollar) score_means={'bias_up_score': 0.0165, 'bias_down_score': 0.046} labels={'BIAS_UP': 24, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 1365} clean={'BIAS_UP': 24, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 522}
2026-05-03 07:46:51,984 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:51,986 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:51,987 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:51,988 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:51,988 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:51,989 INFO Loaded GBPJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:46:52,175 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 27, 'BIAS_DOWN': 37, 'BIAS_NEUTRAL': 1442}  ambiguous=926 (total=1506) horizon=12
2026-05-03 07:46:52,177 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected GBPJPY — 1456 samples (group=cross) score_means={'bias_up_score': 0.0185, 'bias_down_score': 0.0254} labels={'BIAS_UP': 27, 'BIAS_DOWN': 37, 'BIAS_NEUTRAL': 1392} clean={'BIAS_UP': 27, 'BIAS_DOWN': 37, 'BIAS_NEUTRAL': 506}
2026-05-03 07:46:52,241 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:52,243 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:52,244 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:52,244 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:52,245 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:52,246 INFO Loaded GBPUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:46:52,429 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 50, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1371}  ambiguous=874 (total=1506) horizon=12
2026-05-03 07:46:52,432 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected GBPUSD — 1456 samples (group=dollar) score_means={'bias_up_score': 0.0343, 'bias_down_score': 0.0584} labels={'BIAS_UP': 50, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1321} clean={'BIAS_UP': 50, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 476}
2026-05-03 07:46:52,495 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:46:52,496 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:46:52,497 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:46:52,497 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:46:52,497 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:46:52,498 INFO Loaded NZDUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:46:52,682 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 23, 'BIAS_DOWN': 80, 'BIAS_NEUTRAL': 1403}  ambiguous=896 (total=1506) horizon=12
2026-05-03 07:46:52,685 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected NZDUSD — 1456 samples (group=dollar) score_means={'bias_up_score': 0.0158, 'bias_down_score': 0.0549} labels={'BIAS_UP': 23, 'BIAS_DOWN': 80, 'BIAS_NEUTRAL': 1353} clean={'BIAS_UP': 23, 'BIAS_DOWN': 80, 'BIAS_NEUTRAL': 482}
2026-05-03 07:46:52,747 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:52,750 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:52,751 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:52,751 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:52,752 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:52,753 INFO Loaded USDCAD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:46:53,014 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 48, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 1445}  ambiguous=907 (total=1506) horizon=12
2026-05-03 07:46:53,016 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected USDCAD — 1456 samples (group=dollar) score_means={'bias_up_score': 0.033, 'bias_down_score': 0.0089} labels={'BIAS_UP': 48, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 1395} clean={'BIAS_UP': 48, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 522}
2026-05-03 07:46:53,082 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:53,085 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:53,086 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:53,086 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:53,086 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:53,087 INFO Loaded USDCHF/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:46:53,278 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 68, 'BIAS_DOWN': 45, 'BIAS_NEUTRAL': 1393}  ambiguous=848 (total=1506) horizon=12
2026-05-03 07:46:53,281 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected USDCHF — 1456 samples (group=dollar) score_means={'bias_up_score': 0.0467, 'bias_down_score': 0.0309} labels={'BIAS_UP': 68, 'BIAS_DOWN': 45, 'BIAS_NEUTRAL': 1343} clean={'BIAS_UP': 68, 'BIAS_DOWN': 45, 'BIAS_NEUTRAL': 530}
2026-05-03 07:46:53,344 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:53,346 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:53,347 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:53,347 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:53,347 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:46:53,348 INFO Loaded USDJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:46:53,530 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 58, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1418}  ambiguous=888 (total=1506) horizon=12
2026-05-03 07:46:53,532 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected USDJPY — 1456 samples (group=dollar) score_means={'bias_up_score': 0.0398, 'bias_down_score': 0.0206} labels={'BIAS_UP': 58, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1368} clean={'BIAS_UP': 58, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 510}
2026-05-03 07:46:53,605 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:46:53,609 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:46:53,610 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:46:53,610 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:46:53,611 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:46:53,612 INFO Loaded XAUUSD/4H split=val fold=fold_000: 1600 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:46:53,818 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 18, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 1547}  ambiguous=851 (total=1600) horizon=12
2026-05-03 07:46:53,821 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected XAUUSD — 1550 samples (group=gold) score_means={'bias_up_score': 0.0116, 'bias_down_score': 0.0226} labels={'BIAS_UP': 18, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 1497} clean={'BIAS_UP': 18, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 677}
2026-05-03 07:46:53,884 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 59, 'BIAS_DOWN': 119, 'BIAS_NEUTRAL': 4190}, 'dollar': {'BIAS_UP': 276, 'BIAS_DOWN': 373, 'BIAS_NEUTRAL': 9543}, 'gold': {'BIAS_UP': 18, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 1497}}
2026-05-03 07:46:53,884 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0135, 'bias_down_score': 0.0272}, 'dollar': {'bias_up_score': 0.0271, 'bias_down_score': 0.0366}, 'gold': {'bias_up_score': 0.0116, 'bias_down_score': 0.0226}}
2026-05-03 07:46:53,884 INFO Regime[4H mode=htf_bias] label distribution by year: {2018: {'BIAS_UP': 352, 'BIAS_DOWN': 521, 'BIAS_NEUTRAL': 15083}, 2019: {'BIAS_UP': 1, 'BIAS_DOWN': 6, 'BIAS_NEUTRAL': 147}}
2026-05-03 07:46:53,884 INFO Regime[4H mode=htf_bias] score means by year: {2018: {'bias_up_score': 0.0221, 'bias_down_score': 0.0327}, 2019: {'bias_up_score': 0.0065, 'bias_down_score': 0.039}}
2026-05-03 07:46:53,932 INFO Regime phase HTF dataset build fold=fold_000: 8.3s (train=32884 val=16110)
2026-05-03 07:46:53,933 INFO Regime 4H/htf_bias cold start: no existing weights found
2026-05-03 07:46:53,937 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=32884 val=16110 train_labels={'BIAS_UP': 1205, 'BIAS_DOWN': 912, 'BIAS_NEUTRAL': 30767} val_labels={'BIAS_UP': 353, 'BIAS_DOWN': 527, 'BIAS_NEUTRAL': 15230}
2026-05-03 07:46:54,292 INFO RegimeClassifier[mode=htf_bias]: cold start HTF score head
2026-05-03 07:46:54,292 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-03 07:46:54,293 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 20.0, 'bias_down_score': 20.0}
2026-05-03 07:47:00,953 INFO Regime HTF score epoch  1/50 — tr=1.8016 va=1.0335 acc=0.945 bal=0.333 threshold=0.35 margin=0.30 recall={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 1.0} precision={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 0.945}
2026-05-03 07:47:01,933 INFO Regime HTF score epoch  2/50 — tr=1.7572 va=1.0245 bal=0.349
2026-05-03 07:47:02,960 INFO Regime HTF score epoch  3/50 — tr=1.6922 va=0.9928 bal=0.611
2026-05-03 07:47:03,993 INFO Regime HTF score epoch  4/50 — tr=1.5458 va=0.9452 bal=0.713
2026-05-03 07:47:04,959 INFO Regime HTF score epoch  5/50 — tr=1.3930 va=0.8912 acc=0.873 bal=0.724 threshold=0.70 margin=0.30 recall={'BIAS_UP': 0.72, 'BIAS_DOWN': 0.565, 'BIAS_NEUTRAL': 0.887} precision={'BIAS_UP': 0.194, 'BIAS_DOWN': 0.309, 'BIAS_NEUTRAL': 0.976}
2026-05-03 07:47:05,975 INFO Regime HTF score epoch  6/50 — tr=1.1987 va=0.8506 bal=0.793
2026-05-03 07:47:06,941 INFO Regime HTF score epoch  7/50 — tr=1.0367 va=0.8409 bal=0.842
2026-05-03 07:47:07,940 INFO Regime HTF score epoch  8/50 — tr=0.9312 va=0.8417 bal=0.872
2026-05-03 07:47:08,934 INFO Regime HTF score epoch  9/50 — tr=0.8655 va=0.8232 bal=0.877
2026-05-03 07:47:09,892 INFO Regime HTF score epoch 10/50 — tr=0.8219 va=0.7896 acc=0.800 bal=0.882 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.929, 'BIAS_DOWN': 0.926, 'BIAS_NEUTRAL': 0.792} precision={'BIAS_UP': 0.196, 'BIAS_DOWN': 0.212, 'BIAS_NEUTRAL': 0.995}
2026-05-03 07:47:10,894 INFO Regime HTF score epoch 11/50 — tr=0.7868 va=0.7780 bal=0.886
2026-05-03 07:47:11,867 INFO Regime HTF score epoch 12/50 — tr=0.7581 va=0.7493 bal=0.887
2026-05-03 07:47:12,828 INFO Regime HTF score epoch 13/50 — tr=0.7278 va=0.7271 bal=0.888
2026-05-03 07:47:13,805 INFO Regime HTF score epoch 14/50 — tr=0.7053 va=0.7211 bal=0.891
2026-05-03 07:47:14,789 INFO Regime HTF score epoch 15/50 — tr=0.6880 va=0.7098 acc=0.795 bal=0.894 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.941, 'BIAS_DOWN': 0.954, 'BIAS_NEUTRAL': 0.786} precision={'BIAS_UP': 0.199, 'BIAS_DOWN': 0.208, 'BIAS_NEUTRAL': 0.996}
2026-05-03 07:47:15,752 INFO Regime HTF score epoch 16/50 — tr=0.6730 va=0.6874 bal=0.893
2026-05-03 07:47:16,716 INFO Regime HTF score epoch 17/50 — tr=0.6432 va=0.6778 bal=0.895
2026-05-03 07:47:17,739 INFO Regime HTF score epoch 18/50 — tr=0.6321 va=0.6597 bal=0.895
2026-05-03 07:47:18,773 INFO Regime HTF score epoch 19/50 — tr=0.6233 va=0.6490 bal=0.896
2026-05-03 07:47:19,799 INFO Regime HTF score epoch 20/50 — tr=0.6072 va=0.6368 acc=0.803 bal=0.900 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.946, 'BIAS_DOWN': 0.958, 'BIAS_NEUTRAL': 0.795} precision={'BIAS_UP': 0.208, 'BIAS_DOWN': 0.214, 'BIAS_NEUTRAL': 0.997}
2026-05-03 07:47:20,836 INFO Regime HTF score epoch 21/50 — tr=0.5951 va=0.6148 bal=0.899
2026-05-03 07:47:21,809 INFO Regime HTF score epoch 22/50 — tr=0.5877 va=0.6038 bal=0.901
2026-05-03 07:47:22,827 INFO Regime HTF score epoch 23/50 — tr=0.5786 va=0.6045 bal=0.901
2026-05-03 07:47:23,875 INFO Regime HTF score epoch 24/50 — tr=0.5679 va=0.6015 bal=0.903
2026-05-03 07:47:24,914 INFO Regime HTF score epoch 25/50 — tr=0.5579 va=0.5966 acc=0.809 bal=0.909 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.963, 'BIAS_DOWN': 0.964, 'BIAS_NEUTRAL': 0.801} precision={'BIAS_UP': 0.21, 'BIAS_DOWN': 0.224, 'BIAS_NEUTRAL': 0.997}
2026-05-03 07:47:25,890 INFO Regime HTF score epoch 26/50 — tr=0.5511 va=0.5951 bal=0.909
2026-05-03 07:47:26,851 INFO Regime HTF score epoch 27/50 — tr=0.5492 va=0.5797 bal=0.910
2026-05-03 07:47:27,853 INFO Regime HTF score epoch 28/50 — tr=0.5418 va=0.5647 bal=0.907
2026-05-03 07:47:28,814 INFO Regime HTF score epoch 29/50 — tr=0.5395 va=0.5691 bal=0.909
2026-05-03 07:47:29,827 INFO Regime HTF score epoch 30/50 — tr=0.5286 va=0.5643 acc=0.816 bal=0.911 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.958, 'BIAS_DOWN': 0.968, 'BIAS_NEUTRAL': 0.807} precision={'BIAS_UP': 0.219, 'BIAS_DOWN': 0.227, 'BIAS_NEUTRAL': 0.997}
2026-05-03 07:47:30,815 INFO Regime HTF score epoch 31/50 — tr=0.5231 va=0.5518 bal=0.908
2026-05-03 07:47:31,769 INFO Regime HTF score epoch 32/50 — tr=0.5128 va=0.5451 bal=0.909
2026-05-03 07:47:32,723 INFO Regime HTF score epoch 33/50 — tr=0.5200 va=0.5475 bal=0.910
2026-05-03 07:47:33,701 INFO Regime HTF score epoch 34/50 — tr=0.5088 va=0.5522 bal=0.909
2026-05-03 07:47:34,650 INFO Regime HTF score epoch 35/50 — tr=0.5077 va=0.5452 acc=0.820 bal=0.911 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.952, 'BIAS_DOWN': 0.968, 'BIAS_NEUTRAL': 0.812} precision={'BIAS_UP': 0.223, 'BIAS_DOWN': 0.232, 'BIAS_NEUTRAL': 0.997}
2026-05-03 07:47:35,604 INFO Regime HTF score epoch 36/50 — tr=0.5066 va=0.5306 bal=0.910
2026-05-03 07:47:36,596 INFO Regime HTF score epoch 37/50 — tr=0.5025 va=0.5385 bal=0.911
2026-05-03 07:47:37,574 INFO Regime HTF score epoch 38/50 — tr=0.5050 va=0.5385 bal=0.910
2026-05-03 07:47:38,559 INFO Regime HTF score epoch 39/50 — tr=0.4989 va=0.5171 bal=0.910
2026-05-03 07:47:39,553 INFO Regime HTF score epoch 40/50 — tr=0.5011 va=0.5210 acc=0.827 bal=0.910 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.943, 'BIAS_DOWN': 0.968, 'BIAS_NEUTRAL': 0.82} precision={'BIAS_UP': 0.234, 'BIAS_DOWN': 0.236, 'BIAS_NEUTRAL': 0.997}
2026-05-03 07:47:40,601 INFO Regime HTF score epoch 41/50 — tr=0.4998 va=0.5220 bal=0.911
2026-05-03 07:47:41,616 INFO Regime HTF score epoch 42/50 — tr=0.4995 va=0.5175 bal=0.911
2026-05-03 07:47:42,563 INFO Regime HTF score epoch 43/50 — tr=0.4954 va=0.5241 bal=0.911
2026-05-03 07:47:43,530 INFO Regime HTF score epoch 44/50 — tr=0.5013 va=0.5177 bal=0.910
2026-05-03 07:47:44,537 INFO Regime HTF score epoch 45/50 — tr=0.4916 va=0.5309 acc=0.822 bal=0.912 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.955, 'BIAS_DOWN': 0.968, 'BIAS_NEUTRAL': 0.813} precision={'BIAS_UP': 0.225, 'BIAS_DOWN': 0.233, 'BIAS_NEUTRAL': 0.997}
2026-05-03 07:47:45,520 INFO Regime HTF score epoch 46/50 — tr=0.4946 va=0.5274 bal=0.913
2026-05-03 07:47:46,504 INFO Regime HTF score epoch 47/50 — tr=0.4939 va=0.5362 bal=0.914
2026-05-03 07:47:47,451 INFO Regime HTF score epoch 48/50 — tr=0.4918 va=0.5215 bal=0.910
2026-05-03 07:47:47,451 INFO Regime HTF score early stop at epoch 48
2026-05-03 07:47:48,332 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.850 margin=0.000 precision={'BIAS_UP': 0.234, 'BIAS_DOWN': 0.236, 'BIAS_NEUTRAL': 0.997} recall={'BIAS_UP': 0.943, 'BIAS_DOWN': 0.968, 'BIAS_NEUTRAL': 0.82} f1={'BIAS_UP': 0.375, 'BIAS_DOWN': 0.379, 'BIAS_NEUTRAL': 0.9} confusion=[[333, 0, 20], [0, 510, 17], [1091, 1653, 12486]] score_mae={'bias_up_score': 0.2101, 'bias_down_score': 0.2598} pred_share={'BIAS_UP': 0.0884, 'BIAS_DOWN': 0.1343, 'BIAS_NEUTRAL': 0.7773}
2026-05-03 07:47:48,333 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.234, 'BIAS_DOWN': 0.236, 'BIAS_NEUTRAL': 0.997} min_precision=0.300 recall={'BIAS_UP': 0.943, 'BIAS_DOWN': 0.968, 'BIAS_NEUTRAL': 0.82} min_recall=0.100 f1={'BIAS_UP': 0.375, 'BIAS_DOWN': 0.379, 'BIAS_NEUTRAL': 0.9} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_UP', 'BIAS_DOWN'] weak_recall=[] weak_f1=[] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-03 07:47:48,338 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-03 07:47:48,338 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-03 07:47:48,339 INFO Regime phase HTF train fold=fold_000: 54.4s
2026-05-03 07:47:48,437 INFO Regime HTF complete fold=fold_000: acc=0.827 bal=0.910 train=32884 val=16110 per_class={'BIAS_UP': 0.943, 'BIAS_DOWN': 0.968, 'BIAS_NEUTRAL': 0.82} precision={'BIAS_UP': 0.234, 'BIAS_DOWN': 0.236, 'BIAS_NEUTRAL': 0.997} threshold=0.850 margin=0.000
2026-05-03 07:47:48,439 INFO Loaded XAUUSD/4H split=train fold=fold_000: 3204 bars (2016-01-04 → 2018-01-03)
2026-05-03 07:47:48,535 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 2845}  ambiguous=1801 (total=3204) horizon=12
2026-05-03 07:47:48,547 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 4.1568627450980395, 'BIAS_DOWN': 3.972972972972973, 'BIAS_NEUTRAL': 31.96629213483146}
2026-05-03 07:47:48,550 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 212, 'mean': 0.0011890919998414733, 'mean_over_std': 0.3850896851317838}, 'BIAS_DOWN': {'n': 147, 'mean': -0.0013091755049482925, 'mean_over_std': -0.3942426410778961}, 'BIAS_NEUTRAL': {'n': 2844, 'mean': 5.552802176910261e-05, 'mean_over_std': 0.015959039476050135}}
2026-05-03 07:47:48,550 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 212, 'mean': 0.0011890919998414733, 'mean_over_std': 0.3850896851317838}, 'BIAS_DOWN': {'n': 147, 'mean': -0.0013091755049482925, 'mean_over_std': -0.3942426410778961}, 'BIAS_NEUTRAL': {'n': 1044, 'mean': 0.00010149382300531956, 'mean_over_std': 0.037249435653574005}}
2026-05-03 07:47:48,554 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-03 07:47:48,555 INFO Loaded AUDUSD/1H split=train fold=fold_000: 11773 bars (2016-01-04 → 2018-01-03)
2026-05-03 07:47:48,556 INFO Loaded EURGBP/1H split=train fold=fold_000: 11773 bars (2016-01-04 → 2018-01-03)
2026-05-03 07:47:48,558 INFO Loaded EURJPY/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-03 07:47:48,559 INFO Loaded EURUSD/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-03 07:47:48,560 INFO Loaded GBPJPY/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-03 07:47:48,562 INFO Loaded GBPUSD/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-03 07:47:48,563 INFO Loaded NZDUSD/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-03 07:47:48,565 INFO Loaded USDCAD/1H split=train fold=fold_000: 11773 bars (2016-01-04 → 2018-01-03)
2026-05-03 07:47:48,566 INFO Loaded USDCHF/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-03 07:47:48,567 INFO Loaded USDJPY/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-03 07:47:48,570 INFO Loaded XAUUSD/1H split=train fold=fold_000: 11914 bars (2016-01-04 → 2018-01-03)
2026-05-03 07:47:48,581 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:48,587 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:48,588 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:48,589 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:48,589 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:48,592 INFO Loaded AUDUSD/1H split=train fold=fold_000: 11773 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:47:48,917 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected AUDUSD — 11723 samples (group=dollar) score_means={'trend_score': 0.4834, 'range_score': 0.2374, 'chop_score': 0.4688, 'volatility_percentile': 0.3652, 'consolidation_score': 0.2}
2026-05-03 07:47:49,025 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:49,029 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:49,031 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:49,031 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:49,031 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:49,033 INFO Loaded EURGBP/1H split=train fold=fold_000: 11773 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:47:49,346 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected EURGBP — 11723 samples (group=cross) score_means={'trend_score': 0.497, 'range_score': 0.2358, 'chop_score': 0.4623, 'volatility_percentile': 0.3783, 'consolidation_score': 0.1905}
2026-05-03 07:47:49,448 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:49,453 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:49,453 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:49,454 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:49,454 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:49,456 INFO Loaded EURJPY/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:47:49,777 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected EURJPY — 11722 samples (group=cross) score_means={'trend_score': 0.4873, 'range_score': 0.2384, 'chop_score': 0.4674, 'volatility_percentile': 0.3763, 'consolidation_score': 0.1925}
2026-05-03 07:47:49,879 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:49,881 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:49,882 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:49,883 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:49,883 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:49,885 INFO Loaded EURUSD/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:47:50,211 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected EURUSD — 11722 samples (group=dollar) score_means={'trend_score': 0.4931, 'range_score': 0.2373, 'chop_score': 0.464, 'volatility_percentile': 0.3716, 'consolidation_score': 0.1896}
2026-05-03 07:47:50,336 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:50,338 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:50,340 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:50,340 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:50,340 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:50,342 INFO Loaded GBPJPY/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:47:50,662 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected GBPJPY — 11722 samples (group=cross) score_means={'trend_score': 0.5009, 'range_score': 0.2311, 'chop_score': 0.4571, 'volatility_percentile': 0.3758, 'consolidation_score': 0.1946}
2026-05-03 07:47:50,764 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:50,767 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:50,768 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:50,769 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:50,769 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:50,771 INFO Loaded GBPUSD/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:47:51,098 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected GBPUSD — 11722 samples (group=dollar) score_means={'trend_score': 0.5037, 'range_score': 0.2323, 'chop_score': 0.4563, 'volatility_percentile': 0.3792, 'consolidation_score': 0.186}
2026-05-03 07:47:51,205 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:47:51,206 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:47:51,207 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:47:51,207 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:47:51,208 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:47:51,209 INFO Loaded NZDUSD/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:47:51,534 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected NZDUSD — 11722 samples (group=dollar) score_means={'trend_score': 0.4841, 'range_score': 0.2391, 'chop_score': 0.4687, 'volatility_percentile': 0.3726, 'consolidation_score': 0.1911}
2026-05-03 07:47:51,640 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:51,642 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:51,643 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:51,643 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:51,644 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:51,645 INFO Loaded USDCAD/1H split=train fold=fold_000: 11773 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:47:51,966 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected USDCAD — 11723 samples (group=dollar) score_means={'trend_score': 0.4974, 'range_score': 0.2331, 'chop_score': 0.4561, 'volatility_percentile': 0.3775, 'consolidation_score': 0.1896}
2026-05-03 07:47:52,072 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:52,077 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:52,079 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:52,080 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:52,080 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:52,082 INFO Loaded USDCHF/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:47:52,408 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected USDCHF — 11722 samples (group=dollar) score_means={'trend_score': 0.4674, 'range_score': 0.2504, 'chop_score': 0.4822, 'volatility_percentile': 0.3731, 'consolidation_score': 0.1894}
2026-05-03 07:47:52,514 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:52,516 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:52,517 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:52,517 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:52,518 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:52,519 INFO Loaded USDJPY/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:47:52,847 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected USDJPY — 11722 samples (group=dollar) score_means={'trend_score': 0.4991, 'range_score': 0.231, 'chop_score': 0.4562, 'volatility_percentile': 0.3679, 'consolidation_score': 0.1984}
2026-05-03 07:47:52,961 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:47:52,964 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:47:52,966 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:47:52,966 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:47:52,966 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:47:52,969 INFO Loaded XAUUSD/1H split=train fold=fold_000: 11914 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:47:53,318 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected XAUUSD — 11864 samples (group=gold) score_means={'trend_score': 0.5075, 'range_score': 0.2273, 'chop_score': 0.4518, 'volatility_percentile': 0.3694, 'consolidation_score': 0.1952}
2026-05-03 07:47:53,423 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4951, 'range_score': 0.2351, 'chop_score': 0.4622, 'volatility_percentile': 0.3768, 'consolidation_score': 0.1925}, 'dollar': {'trend_score': 0.4897, 'range_score': 0.2372, 'chop_score': 0.4646, 'volatility_percentile': 0.3724, 'consolidation_score': 0.192}, 'gold': {'trend_score': 0.5075, 'range_score': 0.2273, 'chop_score': 0.4518, 'volatility_percentile': 0.3694, 'consolidation_score': 0.1952}}
2026-05-03 07:47:53,423 INFO Regime[1H mode=ltf_behaviour] score means by year: {2016: {'trend_score': 0.4914, 'range_score': 0.2348, 'chop_score': 0.4627, 'volatility_percentile': 0.375, 'consolidation_score': 0.192}, 2017: {'trend_score': 0.4941, 'range_score': 0.2364, 'chop_score': 0.463, 'volatility_percentile': 0.3716, 'consolidation_score': 0.1934}, 2018: {'trend_score': 0.51, 'range_score': 0.2569, 'chop_score': 0.4423, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1324}}
2026-05-03 07:47:53,508 INFO Loaded AUDUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-03 07:47:53,509 INFO Loaded EURGBP/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-03 07:47:53,511 INFO Loaded EURJPY/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-03 07:47:53,512 INFO Loaded EURUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-03 07:47:53,513 INFO Loaded GBPJPY/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-03 07:47:53,515 INFO Loaded GBPUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-03 07:47:53,516 INFO Loaded NZDUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-03 07:47:53,517 INFO Loaded USDCAD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-03 07:47:53,518 INFO Loaded USDCHF/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-03 07:47:53,520 INFO Loaded USDJPY/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-03 07:47:53,522 INFO Loaded XAUUSD/1H split=val fold=fold_000: 6034 bars (2018-01-04 → 2019-01-03)
2026-05-03 07:47:53,527 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:53,529 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:53,530 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:53,530 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:53,530 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:53,532 INFO Loaded AUDUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:47:53,773 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected AUDUSD — 5812 samples (group=dollar) score_means={'trend_score': 0.484, 'range_score': 0.2467, 'chop_score': 0.4726, 'volatility_percentile': 0.3956, 'consolidation_score': 0.1777}
2026-05-03 07:47:53,878 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:53,880 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:53,881 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:53,881 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:53,882 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:53,883 INFO Loaded EURGBP/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:47:54,122 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected EURGBP — 5812 samples (group=cross) score_means={'trend_score': 0.4626, 'range_score': 0.2497, 'chop_score': 0.4853, 'volatility_percentile': 0.3975, 'consolidation_score': 0.1692}
2026-05-03 07:47:54,226 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:54,228 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:54,229 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:54,229 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:54,229 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:54,231 INFO Loaded EURJPY/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:47:54,463 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected EURJPY — 5812 samples (group=cross) score_means={'trend_score': 0.4749, 'range_score': 0.2394, 'chop_score': 0.474, 'volatility_percentile': 0.3878, 'consolidation_score': 0.1827}
2026-05-03 07:47:54,567 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:54,570 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:54,570 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:54,571 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:54,571 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:54,573 INFO Loaded EURUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:47:54,816 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected EURUSD — 5812 samples (group=dollar) score_means={'trend_score': 0.4993, 'range_score': 0.2343, 'chop_score': 0.4572, 'volatility_percentile': 0.389, 'consolidation_score': 0.1807}
2026-05-03 07:47:54,922 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:54,924 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:54,925 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:54,926 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:54,926 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:54,927 INFO Loaded GBPJPY/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:47:55,166 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected GBPJPY — 5812 samples (group=cross) score_means={'trend_score': 0.4888, 'range_score': 0.2412, 'chop_score': 0.4689, 'volatility_percentile': 0.3963, 'consolidation_score': 0.1732}
2026-05-03 07:47:55,272 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:55,274 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:55,275 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:55,275 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:55,275 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:55,277 INFO Loaded GBPUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:47:55,513 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected GBPUSD — 5812 samples (group=dollar) score_means={'trend_score': 0.5007, 'range_score': 0.2339, 'chop_score': 0.4559, 'volatility_percentile': 0.3971, 'consolidation_score': 0.1718}
2026-05-03 07:47:55,617 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:47:55,619 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:47:55,619 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:47:55,620 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:47:55,620 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:47:55,621 INFO Loaded NZDUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:47:55,856 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected NZDUSD — 5812 samples (group=dollar) score_means={'trend_score': 0.4931, 'range_score': 0.2353, 'chop_score': 0.4587, 'volatility_percentile': 0.3902, 'consolidation_score': 0.1824}
2026-05-03 07:47:55,958 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:55,960 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:55,961 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:55,961 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:55,961 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:55,963 INFO Loaded USDCAD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:47:56,196 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected USDCAD — 5812 samples (group=dollar) score_means={'trend_score': 0.4808, 'range_score': 0.2476, 'chop_score': 0.4717, 'volatility_percentile': 0.3857, 'consolidation_score': 0.1768}
2026-05-03 07:47:56,300 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:56,303 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:56,303 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:56,304 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:56,304 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:56,306 INFO Loaded USDCHF/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:47:56,537 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected USDCHF — 5812 samples (group=dollar) score_means={'trend_score': 0.4799, 'range_score': 0.2431, 'chop_score': 0.4697, 'volatility_percentile': 0.3907, 'consolidation_score': 0.1794}
2026-05-03 07:47:56,642 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:56,645 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:56,646 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:56,646 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:56,646 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:47:56,648 INFO Loaded USDJPY/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:47:56,885 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected USDJPY — 5812 samples (group=dollar) score_means={'trend_score': 0.4943, 'range_score': 0.2334, 'chop_score': 0.4614, 'volatility_percentile': 0.3872, 'consolidation_score': 0.1806}
2026-05-03 07:47:56,997 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:47:57,000 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:47:57,002 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:47:57,002 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:47:57,002 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:47:57,004 INFO Loaded XAUUSD/1H split=val fold=fold_000: 6034 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:47:57,257 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected XAUUSD — 5984 samples (group=gold) score_means={'trend_score': 0.4716, 'range_score': 0.2479, 'chop_score': 0.4761, 'volatility_percentile': 0.3856, 'consolidation_score': 0.1812}
2026-05-03 07:47:57,357 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4754, 'range_score': 0.2434, 'chop_score': 0.476, 'volatility_percentile': 0.3939, 'consolidation_score': 0.175}, 'dollar': {'trend_score': 0.4903, 'range_score': 0.2392, 'chop_score': 0.4639, 'volatility_percentile': 0.3908, 'consolidation_score': 0.1785}, 'gold': {'trend_score': 0.4716, 'range_score': 0.2479, 'chop_score': 0.4761, 'volatility_percentile': 0.3856, 'consolidation_score': 0.1812}}
2026-05-03 07:47:57,357 INFO Regime[1H mode=ltf_behaviour] score means by year: {2018: {'trend_score': 0.4841, 'range_score': 0.2416, 'chop_score': 0.4687, 'volatility_percentile': 0.3892, 'consolidation_score': 0.1792}, 2019: {'trend_score': 0.5315, 'range_score': 0.1889, 'chop_score': 0.4263, 'volatility_percentile': 0.5999, 'consolidation_score': 0.0339}}
2026-05-03 07:47:57,434 INFO Regime phase LTF dataset build fold=fold_000: 8.9s (train=129087 val=64104)
2026-05-03 07:47:57,435 INFO Regime 1H/ltf_behaviour cold start: no existing weights found
2026-05-03 07:47:57,455 INFO RegimeClassifier[mode=ltf_behaviour]: cold start score head
2026-05-03 07:47:57,455 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-03 07:47:57,798 INFO Regime score epoch  1/50 — tr=0.0953 va=0.0750 mae={'trend_score': 0.1644, 'range_score': 0.255, 'chop_score': 0.1723, 'volatility_percentile': 0.2097, 'consolidation_score': 0.3448}
2026-05-03 07:47:58,077 INFO Regime score epoch  2/50 — tr=0.0872 va=0.0671
2026-05-03 07:47:58,345 INFO Regime score epoch  3/50 — tr=0.0736 va=0.0540
2026-05-03 07:47:58,605 INFO Regime score epoch  4/50 — tr=0.0568 va=0.0406
2026-05-03 07:47:58,864 INFO Regime score epoch  5/50 — tr=0.0420 va=0.0302 mae={'trend_score': 0.079, 'range_score': 0.1678, 'chop_score': 0.0868, 'volatility_percentile': 0.0802, 'consolidation_score': 0.2662}
2026-05-03 07:47:59,134 INFO Regime score epoch  6/50 — tr=0.0314 va=0.0226
2026-05-03 07:47:59,391 INFO Regime score epoch  7/50 — tr=0.0241 va=0.0167
2026-05-03 07:47:59,649 INFO Regime score epoch  8/50 — tr=0.0193 va=0.0126
2026-05-03 07:47:59,919 INFO Regime score epoch  9/50 — tr=0.0161 va=0.0098
2026-05-03 07:48:00,179 INFO Regime score epoch 10/50 — tr=0.0140 va=0.0079 mae={'trend_score': 0.0624, 'range_score': 0.07, 'chop_score': 0.0593, 'volatility_percentile': 0.0356, 'consolidation_score': 0.1156}
2026-05-03 07:48:00,468 INFO Regime score epoch 11/50 — tr=0.0126 va=0.0067
2026-05-03 07:48:00,717 INFO Regime score epoch 12/50 — tr=0.0115 va=0.0059
2026-05-03 07:48:00,972 INFO Regime score epoch 13/50 — tr=0.0108 va=0.0053
2026-05-03 07:48:01,227 INFO Regime score epoch 14/50 — tr=0.0101 va=0.0050
2026-05-03 07:48:01,485 INFO Regime score epoch 15/50 — tr=0.0097 va=0.0047 mae={'trend_score': 0.0545, 'range_score': 0.0587, 'chop_score': 0.0524, 'volatility_percentile': 0.0302, 'consolidation_score': 0.072}
2026-05-03 07:48:01,770 INFO Regime score epoch 16/50 — tr=0.0093 va=0.0043
2026-05-03 07:48:02,024 INFO Regime score epoch 17/50 — tr=0.0090 va=0.0042
2026-05-03 07:48:02,282 INFO Regime score epoch 18/50 — tr=0.0086 va=0.0040
2026-05-03 07:48:02,539 INFO Regime score epoch 19/50 — tr=0.0084 va=0.0038
2026-05-03 07:48:02,786 INFO Regime score epoch 20/50 — tr=0.0082 va=0.0037 mae={'trend_score': 0.0499, 'range_score': 0.0562, 'chop_score': 0.0492, 'volatility_percentile': 0.0292, 'consolidation_score': 0.0538}
2026-05-03 07:48:03,037 INFO Regime score epoch 21/50 — tr=0.0080 va=0.0035
2026-05-03 07:48:03,288 INFO Regime score epoch 22/50 — tr=0.0078 va=0.0035
2026-05-03 07:48:03,537 INFO Regime score epoch 23/50 — tr=0.0077 va=0.0034
2026-05-03 07:48:03,788 INFO Regime score epoch 24/50 — tr=0.0075 va=0.0033
2026-05-03 07:48:04,050 INFO Regime score epoch 25/50 — tr=0.0074 va=0.0032 mae={'trend_score': 0.0469, 'range_score': 0.0545, 'chop_score': 0.0475, 'volatility_percentile': 0.0275, 'consolidation_score': 0.0453}
2026-05-03 07:48:04,303 INFO Regime score epoch 26/50 — tr=0.0073 va=0.0032
2026-05-03 07:48:04,562 INFO Regime score epoch 27/50 — tr=0.0072 va=0.0031
2026-05-03 07:48:04,812 INFO Regime score epoch 28/50 — tr=0.0071 va=0.0030
2026-05-03 07:48:05,072 INFO Regime score epoch 29/50 — tr=0.0070 va=0.0030
2026-05-03 07:48:05,331 INFO Regime score epoch 30/50 — tr=0.0069 va=0.0029 mae={'trend_score': 0.045, 'range_score': 0.0524, 'chop_score': 0.046, 'volatility_percentile': 0.027, 'consolidation_score': 0.0406}
2026-05-03 07:48:05,593 INFO Regime score epoch 31/50 — tr=0.0069 va=0.0029
2026-05-03 07:48:05,847 INFO Regime score epoch 32/50 — tr=0.0068 va=0.0029
2026-05-03 07:48:06,104 INFO Regime score epoch 33/50 — tr=0.0068 va=0.0028
2026-05-03 07:48:06,370 INFO Regime score epoch 34/50 — tr=0.0067 va=0.0028
2026-05-03 07:48:06,633 INFO Regime score epoch 35/50 — tr=0.0067 va=0.0028 mae={'trend_score': 0.0436, 'range_score': 0.0522, 'chop_score': 0.0453, 'volatility_percentile': 0.0265, 'consolidation_score': 0.0385}
2026-05-03 07:48:06,901 INFO Regime score epoch 36/50 — tr=0.0066 va=0.0028
2026-05-03 07:48:07,160 INFO Regime score epoch 37/50 — tr=0.0066 va=0.0027
2026-05-03 07:48:07,423 INFO Regime score epoch 38/50 — tr=0.0066 va=0.0028
2026-05-03 07:48:07,699 INFO Regime score epoch 39/50 — tr=0.0066 va=0.0027
2026-05-03 07:48:07,969 INFO Regime score epoch 40/50 — tr=0.0065 va=0.0027 mae={'trend_score': 0.0429, 'range_score': 0.0514, 'chop_score': 0.045, 'volatility_percentile': 0.0256, 'consolidation_score': 0.0376}
2026-05-03 07:48:08,230 INFO Regime score epoch 41/50 — tr=0.0065 va=0.0027
2026-05-03 07:48:08,491 INFO Regime score epoch 42/50 — tr=0.0065 va=0.0027
2026-05-03 07:48:08,745 INFO Regime score epoch 43/50 — tr=0.0065 va=0.0027
2026-05-03 07:48:09,004 INFO Regime score epoch 44/50 — tr=0.0065 va=0.0027
2026-05-03 07:48:09,257 INFO Regime score epoch 45/50 — tr=0.0065 va=0.0027 mae={'trend_score': 0.0425, 'range_score': 0.0509, 'chop_score': 0.0444, 'volatility_percentile': 0.0254, 'consolidation_score': 0.0372}
2026-05-03 07:48:09,513 INFO Regime score epoch 46/50 — tr=0.0064 va=0.0027
2026-05-03 07:48:09,769 INFO Regime score epoch 47/50 — tr=0.0064 va=0.0027
2026-05-03 07:48:10,040 INFO Regime score epoch 48/50 — tr=0.0064 va=0.0027
2026-05-03 07:48:10,336 INFO Regime score epoch 49/50 — tr=0.0065 va=0.0027
2026-05-03 07:48:10,592 INFO Regime score epoch 50/50 — tr=0.0064 va=0.0027 mae={'trend_score': 0.0425, 'range_score': 0.0508, 'chop_score': 0.0444, 'volatility_percentile': 0.0257, 'consolidation_score': 0.0367}
2026-05-03 07:48:10,632 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0424, 'range_score': 0.0506, 'chop_score': 0.0442, 'volatility_percentile': 0.0258, 'consolidation_score': 0.0367} mse={'trend_score': 0.00279, 'range_score': 0.00396, 'chop_score': 0.00301, 'volatility_percentile': 0.00125, 'consolidation_score': 0.00225} corr={'trend_score': 0.9718, 'range_score': 0.9067, 'chop_score': 0.96, 'volatility_percentile': 0.9864, 'consolidation_score': 0.9759} pred_std={'trend_score': 0.2044, 'range_score': 0.1402, 'chop_score': 0.1749, 'volatility_percentile': 0.2059, 'consolidation_score': 0.2014} target_std={'trend_score': 0.2203, 'range_score': 0.1457, 'chop_score': 0.1926, 'volatility_percentile': 0.2123, 'consolidation_score': 0.2089}
2026-05-03 07:48:10,637 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-03 07:48:10,637 INFO Regime phase LTF train fold=fold_000: 13.2s
2026-05-03 07:48:10,736 INFO Regime LTF complete fold=fold_000: score_accuracy=0.960, train=129087 val=64104 mae={'trend_score': 0.0424, 'range_score': 0.0506, 'chop_score': 0.0442, 'volatility_percentile': 0.0258, 'consolidation_score': 0.0367}
2026-05-03 07:48:10,739 INFO Loaded XAUUSD/1H split=train fold=fold_000: 11914 bars (2016-01-04 → 2018-01-03)
2026-05-03 07:48:10,880 INFO Regime[1H mode=ltf_behaviour fold=fold_000] LTF score diagnostics on XAUUSD:
{'trend_score': {'mean': 0.507, 'q10': 0.2031, 'q50': 0.5032, 'q90': 0.8121}, 'range_score': {'mean': 0.2284, 'q10': 0.0527, 'q50': 0.2, 'q90': 0.4305}, 'chop_score': {'mean': 0.4525, 'q10': 0.2007, 'q50': 0.4407, 'q90': 0.7194}, 'volatility_percentile': {'mean': 0.3694, 'q10': 0.0827, 'q50': 0.3584, 'q90': 0.6692}, 'consolidation_score': {'mean': 0.1944, 'q10': 0.0, 'q50': 0.1206, 'q90': 0.5428}}
2026-05-03 07:48:10,883 INFO === Regime rolling fold 2/3: fold_001 ===
2026-05-03 07:48:10,884 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-03 07:48:10,884 INFO Split boundaries loaded fold=fold_001/3 — train 2018-01-04→2020-01-03  val 2020-01-06→2020-12-31  test 2023-08-07→2025-08-05
2026-05-03 07:48:10,885 INFO Loaded AUDUSD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-03 07:48:10,886 INFO Loaded EURGBP/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-03 07:48:10,886 INFO Loaded EURJPY/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-03 07:48:10,887 INFO Loaded EURUSD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-03 07:48:10,888 INFO Loaded GBPJPY/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
2026-05-03 07:48:10,889 INFO Loaded GBPUSD/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
2026-05-03 07:48:10,890 INFO Loaded NZDUSD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-03 07:48:10,890 INFO Loaded USDCAD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-03 07:48:10,891 INFO Loaded USDCHF/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-03 07:48:10,892 INFO Loaded USDJPY/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
2026-05-03 07:48:10,893 INFO Loaded XAUUSD/4H split=train fold=fold_001: 3193 bars (2018-01-04 → 2020-01-03)
2026-05-03 07:48:10,898 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:10,901 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:10,901 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:10,902 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:10,902 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:10,903 INFO Loaded AUDUSD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:48:11,117 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 34, 'BIAS_DOWN': 105, 'BIAS_NEUTRAL': 2867}  ambiguous=1757 (total=3006) horizon=12
2026-05-03 07:48:11,120 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected AUDUSD — 2956 samples (group=dollar) score_means={'bias_up_score': 0.0115, 'bias_down_score': 0.0355} labels={'BIAS_UP': 34, 'BIAS_DOWN': 105, 'BIAS_NEUTRAL': 2817} clean={'BIAS_UP': 34, 'BIAS_DOWN': 105, 'BIAS_NEUTRAL': 1089}
2026-05-03 07:48:11,231 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:11,234 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:11,235 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:11,235 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:11,236 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:11,237 INFO Loaded EURGBP/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:48:11,446 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 49, 'BIAS_DOWN': 75, 'BIAS_NEUTRAL': 2882}  ambiguous=1672 (total=3006) horizon=12
2026-05-03 07:48:11,450 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected EURGBP — 2956 samples (group=cross) score_means={'bias_up_score': 0.0166, 'bias_down_score': 0.0254} labels={'BIAS_UP': 49, 'BIAS_DOWN': 75, 'BIAS_NEUTRAL': 2832} clean={'BIAS_UP': 49, 'BIAS_DOWN': 75, 'BIAS_NEUTRAL': 1200}
2026-05-03 07:48:11,556 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:11,561 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:11,561 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:11,562 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:11,562 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:11,563 INFO Loaded EURJPY/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:48:11,778 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 45, 'BIAS_DOWN': 69, 'BIAS_NEUTRAL': 2892}  ambiguous=1719 (total=3006) horizon=12
2026-05-03 07:48:11,780 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected EURJPY — 2956 samples (group=cross) score_means={'bias_up_score': 0.0152, 'bias_down_score': 0.0233} labels={'BIAS_UP': 45, 'BIAS_DOWN': 69, 'BIAS_NEUTRAL': 2842} clean={'BIAS_UP': 45, 'BIAS_DOWN': 69, 'BIAS_NEUTRAL': 1157}
2026-05-03 07:48:11,887 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:11,889 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:11,890 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:11,890 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:11,891 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:11,891 INFO Loaded EURUSD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:48:12,107 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 35, 'BIAS_DOWN': 83, 'BIAS_NEUTRAL': 2888}  ambiguous=1761 (total=3006) horizon=12
2026-05-03 07:48:12,110 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected EURUSD — 2956 samples (group=dollar) score_means={'bias_up_score': 0.0118, 'bias_down_score': 0.0281} labels={'BIAS_UP': 35, 'BIAS_DOWN': 83, 'BIAS_NEUTRAL': 2838} clean={'BIAS_UP': 35, 'BIAS_DOWN': 83, 'BIAS_NEUTRAL': 1110}
2026-05-03 07:48:12,212 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:12,215 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:12,216 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:12,216 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:12,216 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:12,217 INFO Loaded GBPJPY/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:48:12,427 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 76, 'BIAS_DOWN': 93, 'BIAS_NEUTRAL': 2838}  ambiguous=1772 (total=3007) horizon=12
2026-05-03 07:48:12,430 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected GBPJPY — 2957 samples (group=cross) score_means={'bias_up_score': 0.0257, 'bias_down_score': 0.0315} labels={'BIAS_UP': 76, 'BIAS_DOWN': 93, 'BIAS_NEUTRAL': 2788} clean={'BIAS_UP': 76, 'BIAS_DOWN': 93, 'BIAS_NEUTRAL': 1056}
2026-05-03 07:48:12,534 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:12,537 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:12,537 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:12,538 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:12,538 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:12,539 INFO Loaded GBPUSD/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:48:12,745 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 84, 'BIAS_DOWN': 126, 'BIAS_NEUTRAL': 2797}  ambiguous=1784 (total=3007) horizon=12
2026-05-03 07:48:12,747 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected GBPUSD — 2957 samples (group=dollar) score_means={'bias_up_score': 0.0284, 'bias_down_score': 0.0426} labels={'BIAS_UP': 84, 'BIAS_DOWN': 126, 'BIAS_NEUTRAL': 2747} clean={'BIAS_UP': 84, 'BIAS_DOWN': 126, 'BIAS_NEUTRAL': 992}
2026-05-03 07:48:12,848 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:48:12,850 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:48:12,850 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:48:12,851 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:48:12,851 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:48:12,852 INFO Loaded NZDUSD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:48:13,060 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 61, 'BIAS_DOWN': 121, 'BIAS_NEUTRAL': 2824}  ambiguous=1784 (total=3006) horizon=12
2026-05-03 07:48:13,063 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected NZDUSD — 2956 samples (group=dollar) score_means={'bias_up_score': 0.0206, 'bias_down_score': 0.0409} labels={'BIAS_UP': 61, 'BIAS_DOWN': 121, 'BIAS_NEUTRAL': 2774} clean={'BIAS_UP': 61, 'BIAS_DOWN': 121, 'BIAS_NEUTRAL': 1015}
2026-05-03 07:48:13,177 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:13,179 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:13,180 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:13,181 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:13,181 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:13,182 INFO Loaded USDCAD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:48:13,405 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 56, 'BIAS_DOWN': 44, 'BIAS_NEUTRAL': 2906}  ambiguous=1797 (total=3006) horizon=12
2026-05-03 07:48:13,409 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected USDCAD — 2956 samples (group=dollar) score_means={'bias_up_score': 0.0189, 'bias_down_score': 0.0149} labels={'BIAS_UP': 56, 'BIAS_DOWN': 44, 'BIAS_NEUTRAL': 2856} clean={'BIAS_UP': 56, 'BIAS_DOWN': 44, 'BIAS_NEUTRAL': 1093}
2026-05-03 07:48:13,522 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:13,524 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:13,525 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:13,525 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:13,526 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:13,527 INFO Loaded USDCHF/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:48:13,767 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 111, 'BIAS_DOWN': 61, 'BIAS_NEUTRAL': 2834}  ambiguous=1701 (total=3006) horizon=12
2026-05-03 07:48:13,770 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected USDCHF — 2956 samples (group=dollar) score_means={'bias_up_score': 0.0376, 'bias_down_score': 0.0206} labels={'BIAS_UP': 111, 'BIAS_DOWN': 61, 'BIAS_NEUTRAL': 2784} clean={'BIAS_UP': 111, 'BIAS_DOWN': 61, 'BIAS_NEUTRAL': 1118}
2026-05-03 07:48:13,880 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:13,884 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:13,885 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:13,885 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:13,886 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:13,887 INFO Loaded USDJPY/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:48:14,120 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 78, 'BIAS_DOWN': 75, 'BIAS_NEUTRAL': 2854}  ambiguous=1708 (total=3007) horizon=12
2026-05-03 07:48:14,123 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected USDJPY — 2957 samples (group=dollar) score_means={'bias_up_score': 0.0264, 'bias_down_score': 0.0254} labels={'BIAS_UP': 78, 'BIAS_DOWN': 75, 'BIAS_NEUTRAL': 2804} clean={'BIAS_UP': 78, 'BIAS_DOWN': 75, 'BIAS_NEUTRAL': 1126}
2026-05-03 07:48:14,238 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:48:14,242 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:48:14,243 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:48:14,244 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:48:14,244 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:48:14,245 INFO Loaded XAUUSD/4H split=train fold=fold_001: 3193 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:48:14,467 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 91, 'BIAS_DOWN': 40, 'BIAS_NEUTRAL': 3062}  ambiguous=1810 (total=3193) horizon=12
2026-05-03 07:48:14,470 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected XAUUSD — 3143 samples (group=gold) score_means={'bias_up_score': 0.029, 'bias_down_score': 0.0127} labels={'BIAS_UP': 91, 'BIAS_DOWN': 40, 'BIAS_NEUTRAL': 3012} clean={'BIAS_UP': 91, 'BIAS_DOWN': 40, 'BIAS_NEUTRAL': 1233}
2026-05-03 07:48:14,571 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 170, 'BIAS_DOWN': 237, 'BIAS_NEUTRAL': 8462}, 'dollar': {'BIAS_UP': 459, 'BIAS_DOWN': 615, 'BIAS_NEUTRAL': 19620}, 'gold': {'BIAS_UP': 91, 'BIAS_DOWN': 40, 'BIAS_NEUTRAL': 3012}}
2026-05-03 07:48:14,571 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0192, 'bias_down_score': 0.0267}, 'dollar': {'bias_up_score': 0.0222, 'bias_down_score': 0.0297}, 'gold': {'bias_up_score': 0.029, 'bias_down_score': 0.0127}}
2026-05-03 07:48:14,571 INFO Regime[4H mode=htf_bias] label distribution by year: {2018: {'BIAS_UP': 354, 'BIAS_DOWN': 523, 'BIAS_NEUTRAL': 15079}, 2019: {'BIAS_UP': 365, 'BIAS_DOWN': 368, 'BIAS_NEUTRAL': 15874}, 2020: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 141}}
2026-05-03 07:48:14,571 INFO Regime[4H mode=htf_bias] score means by year: {2018: {'bias_up_score': 0.0222, 'bias_down_score': 0.0328}, 2019: {'bias_up_score': 0.022, 'bias_down_score': 0.0222}, 2020: {'bias_up_score': 0.007, 'bias_down_score': 0.007}}
2026-05-03 07:48:14,646 INFO Loaded AUDUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-03 07:48:14,647 INFO Loaded EURGBP/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-03 07:48:14,648 INFO Loaded EURJPY/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-03 07:48:14,649 INFO Loaded EURUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-03 07:48:14,650 INFO Loaded GBPJPY/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-03 07:48:14,650 INFO Loaded GBPUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-03 07:48:14,651 INFO Loaded NZDUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-03 07:48:14,652 INFO Loaded USDCAD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-03 07:48:14,653 INFO Loaded USDCHF/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-03 07:48:14,653 INFO Loaded USDJPY/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-03 07:48:14,654 INFO Loaded XAUUSD/4H split=val fold=fold_001: 1581 bars (2020-01-06 → 2020-12-31)
2026-05-03 07:48:14,660 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:14,662 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:14,663 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:14,663 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:14,663 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:14,664 INFO Loaded AUDUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:48:14,852 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 26, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1411}  ambiguous=832 (total=1490) horizon=12
2026-05-03 07:48:14,854 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected AUDUSD — 1440 samples (group=dollar) score_means={'bias_up_score': 0.0181, 'bias_down_score': 0.0368} labels={'BIAS_UP': 26, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1361} clean={'BIAS_UP': 26, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 550}
2026-05-03 07:48:14,965 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:14,969 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:14,970 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:14,970 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:14,971 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:14,972 INFO Loaded EURGBP/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:48:15,158 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 62, 'BIAS_DOWN': 3, 'BIAS_NEUTRAL': 1425}  ambiguous=865 (total=1490) horizon=12
2026-05-03 07:48:15,160 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected EURGBP — 1440 samples (group=cross) score_means={'bias_up_score': 0.0431, 'bias_down_score': 0.0021} labels={'BIAS_UP': 62, 'BIAS_DOWN': 3, 'BIAS_NEUTRAL': 1375} clean={'BIAS_UP': 62, 'BIAS_DOWN': 3, 'BIAS_NEUTRAL': 536}
2026-05-03 07:48:15,263 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:15,267 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:15,268 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:15,268 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:15,268 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:15,269 INFO Loaded EURJPY/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:48:15,454 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 42, 'BIAS_DOWN': 12, 'BIAS_NEUTRAL': 1436}  ambiguous=928 (total=1490) horizon=12
2026-05-03 07:48:15,456 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected EURJPY — 1440 samples (group=cross) score_means={'bias_up_score': 0.0292, 'bias_down_score': 0.0083} labels={'BIAS_UP': 42, 'BIAS_DOWN': 12, 'BIAS_NEUTRAL': 1386} clean={'BIAS_UP': 42, 'BIAS_DOWN': 12, 'BIAS_NEUTRAL': 491}
2026-05-03 07:48:15,560 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:15,562 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:15,563 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:15,563 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:15,563 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:15,564 INFO Loaded EURUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:48:15,754 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 89, 'BIAS_DOWN': 33, 'BIAS_NEUTRAL': 1368}  ambiguous=880 (total=1490) horizon=12
2026-05-03 07:48:15,756 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected EURUSD — 1440 samples (group=dollar) score_means={'bias_up_score': 0.0618, 'bias_down_score': 0.0229} labels={'BIAS_UP': 89, 'BIAS_DOWN': 33, 'BIAS_NEUTRAL': 1318} clean={'BIAS_UP': 89, 'BIAS_DOWN': 33, 'BIAS_NEUTRAL': 462}
2026-05-03 07:48:15,859 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:15,861 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:15,862 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:15,862 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:15,863 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:15,864 INFO Loaded GBPJPY/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:48:16,049 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 26, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 1418}  ambiguous=910 (total=1490) horizon=12
2026-05-03 07:48:16,052 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected GBPJPY — 1440 samples (group=cross) score_means={'bias_up_score': 0.0181, 'bias_down_score': 0.0319} labels={'BIAS_UP': 26, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 1368} clean={'BIAS_UP': 26, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 476}
2026-05-03 07:48:16,156 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:16,158 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:16,159 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:16,159 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:16,160 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:16,160 INFO Loaded GBPUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:48:16,353 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 39, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 1429}  ambiguous=909 (total=1490) horizon=12
2026-05-03 07:48:16,356 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected GBPUSD — 1440 samples (group=dollar) score_means={'bias_up_score': 0.0271, 'bias_down_score': 0.0153} labels={'BIAS_UP': 39, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 1379} clean={'BIAS_UP': 39, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 493}
2026-05-03 07:48:16,459 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:48:16,461 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:48:16,461 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:48:16,462 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:48:16,462 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:48:16,463 INFO Loaded NZDUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:48:16,648 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 47, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1402}  ambiguous=817 (total=1490) horizon=12
2026-05-03 07:48:16,651 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected NZDUSD — 1440 samples (group=dollar) score_means={'bias_up_score': 0.0326, 'bias_down_score': 0.0285} labels={'BIAS_UP': 47, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1352} clean={'BIAS_UP': 47, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 561}
2026-05-03 07:48:16,754 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:16,756 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:16,757 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:16,757 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:16,758 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:16,758 INFO Loaded USDCAD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:48:16,948 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 61, 'BIAS_DOWN': 59, 'BIAS_NEUTRAL': 1370}  ambiguous=800 (total=1490) horizon=12
2026-05-03 07:48:16,950 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected USDCAD — 1440 samples (group=dollar) score_means={'bias_up_score': 0.0424, 'bias_down_score': 0.041} labels={'BIAS_UP': 61, 'BIAS_DOWN': 59, 'BIAS_NEUTRAL': 1320} clean={'BIAS_UP': 61, 'BIAS_DOWN': 59, 'BIAS_NEUTRAL': 551}
2026-05-03 07:48:17,055 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:17,057 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:17,058 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:17,058 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:17,058 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:17,059 INFO Loaded USDCHF/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:48:17,248 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 11, 'BIAS_DOWN': 76, 'BIAS_NEUTRAL': 1403}  ambiguous=838 (total=1490) horizon=12
2026-05-03 07:48:17,251 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected USDCHF — 1440 samples (group=dollar) score_means={'bias_up_score': 0.0076, 'bias_down_score': 0.0528} labels={'BIAS_UP': 11, 'BIAS_DOWN': 76, 'BIAS_NEUTRAL': 1353} clean={'BIAS_UP': 11, 'BIAS_DOWN': 76, 'BIAS_NEUTRAL': 539}
2026-05-03 07:48:17,353 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:17,355 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:17,356 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:17,356 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:17,356 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:48:17,357 INFO Loaded USDJPY/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:48:17,545 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 6, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1431}  ambiguous=877 (total=1490) horizon=12
2026-05-03 07:48:17,547 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected USDJPY — 1440 samples (group=dollar) score_means={'bias_up_score': 0.0042, 'bias_down_score': 0.0368} labels={'BIAS_UP': 6, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1381} clean={'BIAS_UP': 6, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 537}
2026-05-03 07:48:17,659 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:48:17,663 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:48:17,664 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:48:17,665 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:48:17,665 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:48:17,666 INFO Loaded XAUUSD/4H split=val fold=fold_001: 1581 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:48:17,865 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 76, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 1497}  ambiguous=916 (total=1581) horizon=12
2026-05-03 07:48:17,868 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected XAUUSD — 1531 samples (group=gold) score_means={'bias_up_score': 0.0496, 'bias_down_score': 0.0052} labels={'BIAS_UP': 76, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 1447} clean={'BIAS_UP': 76, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 553}
2026-05-03 07:48:17,966 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 130, 'BIAS_DOWN': 61, 'BIAS_NEUTRAL': 4129}, 'dollar': {'BIAS_UP': 279, 'BIAS_DOWN': 337, 'BIAS_NEUTRAL': 9464}, 'gold': {'BIAS_UP': 76, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 1447}}
2026-05-03 07:48:17,966 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0301, 'bias_down_score': 0.0141}, 'dollar': {'bias_up_score': 0.0277, 'bias_down_score': 0.0334}, 'gold': {'bias_up_score': 0.0496, 'bias_down_score': 0.0052}}
2026-05-03 07:48:17,966 INFO Regime[4H mode=htf_bias] label distribution by year: {2020: {'BIAS_UP': 485, 'BIAS_DOWN': 406, 'BIAS_NEUTRAL': 15040}}
2026-05-03 07:48:17,966 INFO Regime[4H mode=htf_bias] score means by year: {2020: {'bias_up_score': 0.0304, 'bias_down_score': 0.0255}}
2026-05-03 07:48:18,041 INFO Regime phase HTF dataset build fold=fold_001: 7.2s (train=32706 val=15931)
2026-05-03 07:48:18,051 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=24, n_classes=2)
2026-05-03 07:48:18,051 INFO Regime 4H/htf_bias warm start enabled from existing weights
2026-05-03 07:48:18,056 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=32706 val=15931 train_labels={'BIAS_UP': 720, 'BIAS_DOWN': 892, 'BIAS_NEUTRAL': 31094} val_labels={'BIAS_UP': 485, 'BIAS_DOWN': 406, 'BIAS_NEUTRAL': 15040}
2026-05-03 07:48:18,056 INFO RegimeClassifier[mode=htf_bias]: warm start HTF score head
2026-05-03 07:48:18,056 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-03 07:48:18,056 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 20.0, 'bias_down_score': 20.0}
2026-05-03 07:48:19,012 INFO Regime HTF score epoch  1/50 — tr=0.4765 va=0.4847 acc=0.849 bal=0.921 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.971, 'BIAS_DOWN': 0.948, 'BIAS_NEUTRAL': 0.843} precision={'BIAS_UP': 0.261, 'BIAS_DOWN': 0.272, 'BIAS_NEUTRAL': 0.997}
2026-05-03 07:48:20,028 INFO Regime HTF score epoch  2/50 — tr=0.4754 va=0.4947 bal=0.924
2026-05-03 07:48:21,022 INFO Regime HTF score epoch  3/50 — tr=0.4769 va=0.4970 bal=0.925
2026-05-03 07:48:21,975 INFO Regime HTF score epoch  4/50 — tr=0.4765 va=0.4982 bal=0.926
2026-05-03 07:48:23,004 INFO Regime HTF score epoch  5/50 — tr=0.4726 va=0.4933 acc=0.845 bal=0.925 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.975, 'BIAS_DOWN': 0.963, 'BIAS_NEUTRAL': 0.837} precision={'BIAS_UP': 0.255, 'BIAS_DOWN': 0.269, 'BIAS_NEUTRAL': 0.998}
2026-05-03 07:48:24,008 INFO Regime HTF score epoch  6/50 — tr=0.4627 va=0.4876 bal=0.925
2026-05-03 07:48:25,020 INFO Regime HTF score epoch  7/50 — tr=0.4615 va=0.4820 bal=0.924
2026-05-03 07:48:26,055 INFO Regime HTF score epoch  8/50 — tr=0.4600 va=0.4747 bal=0.920
2026-05-03 07:48:27,026 INFO Regime HTF score epoch  9/50 — tr=0.4514 va=0.4730 bal=0.922
2026-05-03 07:48:27,984 INFO Regime HTF score epoch 10/50 — tr=0.4450 va=0.4682 acc=0.857 bal=0.922 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.965, 'BIAS_DOWN': 0.951, 'BIAS_NEUTRAL': 0.851} precision={'BIAS_UP': 0.278, 'BIAS_DOWN': 0.272, 'BIAS_NEUTRAL': 0.997}
2026-05-03 07:48:28,945 INFO Regime HTF score epoch 11/50 — tr=0.4416 va=0.4619 bal=0.920
2026-05-03 07:48:29,907 INFO Regime HTF score epoch 12/50 — tr=0.4381 va=0.4567 bal=0.918
2026-05-03 07:48:30,900 INFO Regime HTF score epoch 13/50 — tr=0.4343 va=0.4566 bal=0.917
2026-05-03 07:48:31,927 INFO Regime HTF score epoch 14/50 — tr=0.4276 va=0.4532 bal=0.917
2026-05-03 07:48:32,895 INFO Regime HTF score epoch 15/50 — tr=0.4282 va=0.4534 acc=0.860 bal=0.916 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.965, 'BIAS_DOWN': 0.929, 'BIAS_NEUTRAL': 0.854} precision={'BIAS_UP': 0.279, 'BIAS_DOWN': 0.278, 'BIAS_NEUTRAL': 0.996}
2026-05-03 07:48:33,867 INFO Regime HTF score epoch 16/50 — tr=0.4249 va=0.4513 bal=0.918
2026-05-03 07:48:34,884 INFO Regime HTF score epoch 17/50 — tr=0.4237 va=0.4487 bal=0.917
2026-05-03 07:48:35,837 INFO Regime HTF score epoch 18/50 — tr=0.4175 va=0.4451 bal=0.916
2026-05-03 07:48:36,790 INFO Regime HTF score epoch 19/50 — tr=0.4220 va=0.4405 bal=0.914
2026-05-03 07:48:37,742 INFO Regime HTF score epoch 20/50 — tr=0.4146 va=0.4399 acc=0.863 bal=0.916 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.963, 'BIAS_DOWN': 0.926, 'BIAS_NEUTRAL': 0.858} precision={'BIAS_UP': 0.284, 'BIAS_DOWN': 0.281, 'BIAS_NEUTRAL': 0.996}
2026-05-03 07:48:38,690 INFO Regime HTF score epoch 21/50 — tr=0.4099 va=0.4382 bal=0.916
2026-05-03 07:48:39,653 INFO Regime HTF score epoch 22/50 — tr=0.4126 va=0.4346 bal=0.915
2026-05-03 07:48:40,668 INFO Regime HTF score epoch 23/50 — tr=0.4120 va=0.4348 bal=0.916
2026-05-03 07:48:41,624 INFO Regime HTF score epoch 24/50 — tr=0.4064 va=0.4316 bal=0.917
2026-05-03 07:48:42,576 INFO Regime HTF score epoch 25/50 — tr=0.3949 va=0.4303 acc=0.865 bal=0.917 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.959, 'BIAS_DOWN': 0.931, 'BIAS_NEUTRAL': 0.86} precision={'BIAS_UP': 0.288, 'BIAS_DOWN': 0.283, 'BIAS_NEUTRAL': 0.996}
2026-05-03 07:48:43,541 INFO Regime HTF score epoch 26/50 — tr=0.4048 va=0.4287 bal=0.917
2026-05-03 07:48:44,518 INFO Regime HTF score epoch 27/50 — tr=0.4016 va=0.4267 bal=0.915
2026-05-03 07:48:45,469 INFO Regime HTF score epoch 28/50 — tr=0.4000 va=0.4264 bal=0.918
2026-05-03 07:48:46,435 INFO Regime HTF score epoch 29/50 — tr=0.3966 va=0.4258 bal=0.918
2026-05-03 07:48:47,408 INFO Regime HTF score epoch 30/50 — tr=0.3955 va=0.4252 acc=0.865 bal=0.918 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.963, 'BIAS_DOWN': 0.931, 'BIAS_NEUTRAL': 0.86} precision={'BIAS_UP': 0.289, 'BIAS_DOWN': 0.284, 'BIAS_NEUTRAL': 0.996}
2026-05-03 07:48:48,373 INFO Regime HTF score epoch 31/50 — tr=0.4015 va=0.4260 bal=0.918
2026-05-03 07:48:49,342 INFO Regime HTF score epoch 32/50 — tr=0.3924 va=0.4262 bal=0.919
2026-05-03 07:48:50,326 INFO Regime HTF score epoch 33/50 — tr=0.3945 va=0.4235 bal=0.919
2026-05-03 07:48:51,302 INFO Regime HTF score epoch 34/50 — tr=0.3920 va=0.4213 bal=0.918
2026-05-03 07:48:52,319 INFO Regime HTF score epoch 35/50 — tr=0.3888 va=0.4203 acc=0.866 bal=0.918 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.963, 'BIAS_DOWN': 0.931, 'BIAS_NEUTRAL': 0.861} precision={'BIAS_UP': 0.291, 'BIAS_DOWN': 0.285, 'BIAS_NEUTRAL': 0.996}
2026-05-03 07:48:53,276 INFO Regime HTF score epoch 36/50 — tr=0.3906 va=0.4201 bal=0.918
2026-05-03 07:48:54,288 INFO Regime HTF score epoch 37/50 — tr=0.3907 va=0.4196 bal=0.918
2026-05-03 07:48:55,264 INFO Regime HTF score epoch 38/50 — tr=0.3960 va=0.4186 bal=0.918
2026-05-03 07:48:56,259 INFO Regime HTF score epoch 39/50 — tr=0.3873 va=0.4192 bal=0.918
2026-05-03 07:48:57,210 INFO Regime HTF score epoch 40/50 — tr=0.3839 va=0.4202 acc=0.865 bal=0.920 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.965, 'BIAS_DOWN': 0.933, 'BIAS_NEUTRAL': 0.86} precision={'BIAS_UP': 0.289, 'BIAS_DOWN': 0.285, 'BIAS_NEUTRAL': 0.997}
2026-05-03 07:48:58,245 INFO Regime HTF score epoch 41/50 — tr=0.3836 va=0.4183 bal=0.918
2026-05-03 07:48:59,213 INFO Regime HTF score epoch 42/50 — tr=0.3909 va=0.4165 bal=0.916
2026-05-03 07:49:00,203 INFO Regime HTF score epoch 43/50 — tr=0.3908 va=0.4155 bal=0.916
2026-05-03 07:49:01,192 INFO Regime HTF score epoch 44/50 — tr=0.3856 va=0.4158 bal=0.916
2026-05-03 07:49:02,154 INFO Regime HTF score epoch 45/50 — tr=0.3885 va=0.4166 acc=0.867 bal=0.918 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.963, 'BIAS_DOWN': 0.929, 'BIAS_NEUTRAL': 0.862} precision={'BIAS_UP': 0.291, 'BIAS_DOWN': 0.286, 'BIAS_NEUTRAL': 0.996}
2026-05-03 07:49:03,120 INFO Regime HTF score epoch 46/50 — tr=0.3904 va=0.4177 bal=0.918
2026-05-03 07:49:04,139 INFO Regime HTF score epoch 47/50 — tr=0.3866 va=0.4185 bal=0.919
2026-05-03 07:49:05,093 INFO Regime HTF score epoch 48/50 — tr=0.3860 va=0.4169 bal=0.918
2026-05-03 07:49:05,093 INFO Regime HTF score early stop at epoch 48
2026-05-03 07:49:05,982 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.850 margin=0.000 precision={'BIAS_UP': 0.289, 'BIAS_DOWN': 0.285, 'BIAS_NEUTRAL': 0.997} recall={'BIAS_UP': 0.965, 'BIAS_DOWN': 0.933, 'BIAS_NEUTRAL': 0.86} f1={'BIAS_UP': 0.445, 'BIAS_DOWN': 0.437, 'BIAS_NEUTRAL': 0.923} confusion=[[468, 0, 17], [0, 379, 27], [1152, 949, 12939]] score_mae={'bias_up_score': 0.1954, 'bias_down_score': 0.1661} pred_share={'BIAS_UP': 0.1017, 'BIAS_DOWN': 0.0834, 'BIAS_NEUTRAL': 0.815}
2026-05-03 07:49:05,983 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.289, 'BIAS_DOWN': 0.285, 'BIAS_NEUTRAL': 0.997} min_precision=0.300 recall={'BIAS_UP': 0.965, 'BIAS_DOWN': 0.933, 'BIAS_NEUTRAL': 0.86} min_recall=0.100 f1={'BIAS_UP': 0.445, 'BIAS_DOWN': 0.437, 'BIAS_NEUTRAL': 0.923} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_UP', 'BIAS_DOWN'] weak_recall=[] weak_f1=[] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-03 07:49:05,986 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-03 07:49:05,986 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-03 07:49:05,986 INFO Regime phase HTF train fold=fold_001: 47.9s
2026-05-03 07:49:06,092 INFO Regime HTF complete fold=fold_001: acc=0.865 bal=0.920 train=32706 val=15931 per_class={'BIAS_UP': 0.965, 'BIAS_DOWN': 0.933, 'BIAS_NEUTRAL': 0.86} precision={'BIAS_UP': 0.289, 'BIAS_DOWN': 0.285, 'BIAS_NEUTRAL': 0.997} threshold=0.850 margin=0.000
2026-05-03 07:49:06,094 INFO Loaded XAUUSD/4H split=train fold=fold_001: 3193 bars (2018-01-04 → 2020-01-03)
2026-05-03 07:49:06,190 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 91, 'BIAS_DOWN': 40, 'BIAS_NEUTRAL': 3062}  ambiguous=1810 (total=3193) horizon=12
2026-05-03 07:49:06,192 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 4.136363636363637, 'BIAS_DOWN': 3.076923076923077, 'BIAS_NEUTRAL': 85.05555555555556}
2026-05-03 07:49:06,195 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 91, 'mean': 0.0012718413994960778, 'mean_over_std': 0.5225173433708362}, 'BIAS_DOWN': {'n': 40, 'mean': -0.0007135081911815277, 'mean_over_std': -0.3923435533984593}, 'BIAS_NEUTRAL': {'n': 3061, 'mean': 3.036271341687858e-05, 'mean_over_std': 0.010967519850120955}}
2026-05-03 07:49:06,195 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 91, 'mean': 0.0012718413994960778, 'mean_over_std': 0.5225173433708362}, 'BIAS_DOWN': {'n': 40, 'mean': -0.0007135081911815277, 'mean_over_std': -0.3923435533984593}, 'BIAS_NEUTRAL': {'n': 1252, 'mean': 3.51836161561549e-05, 'mean_over_std': 0.016057793936327037}}
2026-05-03 07:49:06,199 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-03 07:49:06,200 INFO Loaded AUDUSD/1H split=train fold=fold_001: 11705 bars (2018-01-04 → 2020-01-03)
2026-05-03 07:49:06,202 INFO Loaded EURGBP/1H split=train fold=fold_001: 11707 bars (2018-01-04 → 2020-01-03)
2026-05-03 07:49:06,203 INFO Loaded EURJPY/1H split=train fold=fold_001: 11708 bars (2018-01-04 → 2020-01-03)
2026-05-03 07:49:06,204 INFO Loaded EURUSD/1H split=train fold=fold_001: 11707 bars (2018-01-04 → 2020-01-03)
2026-05-03 07:49:06,206 INFO Loaded GBPJPY/1H split=train fold=fold_001: 11708 bars (2018-01-04 → 2020-01-03)
2026-05-03 07:49:06,207 INFO Loaded GBPUSD/1H split=train fold=fold_001: 11708 bars (2018-01-04 → 2020-01-03)
2026-05-03 07:49:06,208 INFO Loaded NZDUSD/1H split=train fold=fold_001: 11707 bars (2018-01-04 → 2020-01-03)
2026-05-03 07:49:06,210 INFO Loaded USDCAD/1H split=train fold=fold_001: 11707 bars (2018-01-04 → 2020-01-03)
2026-05-03 07:49:06,211 INFO Loaded USDCHF/1H split=train fold=fold_001: 11709 bars (2018-01-04 → 2020-01-03)
2026-05-03 07:49:06,212 INFO Loaded USDJPY/1H split=train fold=fold_001: 11711 bars (2018-01-04 → 2020-01-03)
2026-05-03 07:49:06,214 INFO Loaded XAUUSD/1H split=train fold=fold_001: 11927 bars (2018-01-04 → 2020-01-03)
2026-05-03 07:49:06,220 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:06,222 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:06,223 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:06,223 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:06,223 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:06,225 INFO Loaded AUDUSD/1H split=train fold=fold_001: 11705 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:06,559 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected AUDUSD — 11655 samples (group=dollar) score_means={'trend_score': 0.4956, 'range_score': 0.2359, 'chop_score': 0.4612, 'volatility_percentile': 0.3808, 'consolidation_score': 0.1864}
2026-05-03 07:49:06,685 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:06,689 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:06,690 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:06,690 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:06,691 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:06,693 INFO Loaded EURGBP/1H split=train fold=fold_001: 11707 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:07,017 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected EURGBP — 11657 samples (group=cross) score_means={'trend_score': 0.4573, 'range_score': 0.2481, 'chop_score': 0.4857, 'volatility_percentile': 0.3934, 'consolidation_score': 0.1799}
2026-05-03 07:49:07,145 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:07,148 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:07,150 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:07,151 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:07,151 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:07,153 INFO Loaded EURJPY/1H split=train fold=fold_001: 11708 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:07,489 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected EURJPY — 11658 samples (group=cross) score_means={'trend_score': 0.4805, 'range_score': 0.2366, 'chop_score': 0.4707, 'volatility_percentile': 0.3744, 'consolidation_score': 0.1928}
2026-05-03 07:49:07,604 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:07,608 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:07,609 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:07,609 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:07,610 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:07,611 INFO Loaded EURUSD/1H split=train fold=fold_001: 11707 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:07,944 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected EURUSD — 11657 samples (group=dollar) score_means={'trend_score': 0.4875, 'range_score': 0.2372, 'chop_score': 0.4621, 'volatility_percentile': 0.378, 'consolidation_score': 0.1876}
2026-05-03 07:49:08,074 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:08,076 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:08,077 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:08,077 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:08,078 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:08,079 INFO Loaded GBPJPY/1H split=train fold=fold_001: 11708 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:08,411 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected GBPJPY — 11658 samples (group=cross) score_means={'trend_score': 0.4891, 'range_score': 0.2383, 'chop_score': 0.4697, 'volatility_percentile': 0.39, 'consolidation_score': 0.184}
2026-05-03 07:49:08,526 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:08,529 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:08,530 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:08,530 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:08,530 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:08,532 INFO Loaded GBPUSD/1H split=train fold=fold_001: 11708 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:08,865 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected GBPUSD — 11658 samples (group=dollar) score_means={'trend_score': 0.4959, 'range_score': 0.2313, 'chop_score': 0.4576, 'volatility_percentile': 0.3919, 'consolidation_score': 0.1801}
2026-05-03 07:49:08,982 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:08,984 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:08,985 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:08,985 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:08,985 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:08,987 INFO Loaded NZDUSD/1H split=train fold=fold_001: 11707 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:09,309 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected NZDUSD — 11657 samples (group=dollar) score_means={'trend_score': 0.4999, 'range_score': 0.2321, 'chop_score': 0.4537, 'volatility_percentile': 0.3824, 'consolidation_score': 0.1829}
2026-05-03 07:49:09,425 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:09,427 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:09,428 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:09,428 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:09,428 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:09,430 INFO Loaded USDCAD/1H split=train fold=fold_001: 11707 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:09,757 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected USDCAD — 11657 samples (group=dollar) score_means={'trend_score': 0.4841, 'range_score': 0.2445, 'chop_score': 0.47, 'volatility_percentile': 0.3763, 'consolidation_score': 0.186}
2026-05-03 07:49:09,864 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:09,867 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:09,868 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:09,868 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:09,868 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:09,870 INFO Loaded USDCHF/1H split=train fold=fold_001: 11709 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:10,195 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected USDCHF — 11659 samples (group=dollar) score_means={'trend_score': 0.4748, 'range_score': 0.2414, 'chop_score': 0.47, 'volatility_percentile': 0.3832, 'consolidation_score': 0.1861}
2026-05-03 07:49:10,331 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:10,333 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:10,334 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:10,335 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:10,335 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:10,337 INFO Loaded USDJPY/1H split=train fold=fold_001: 11711 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:10,658 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected USDJPY — 11661 samples (group=dollar) score_means={'trend_score': 0.4818, 'range_score': 0.237, 'chop_score': 0.4712, 'volatility_percentile': 0.3725, 'consolidation_score': 0.1975}
2026-05-03 07:49:10,793 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:10,801 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:10,804 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:10,805 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:10,805 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:10,808 INFO Loaded XAUUSD/1H split=train fold=fold_001: 11927 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:11,167 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected XAUUSD — 11877 samples (group=gold) score_means={'trend_score': 0.475, 'range_score': 0.244, 'chop_score': 0.473, 'volatility_percentile': 0.3803, 'consolidation_score': 0.1878}
2026-05-03 07:49:11,290 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4756, 'range_score': 0.241, 'chop_score': 0.4754, 'volatility_percentile': 0.3859, 'consolidation_score': 0.1855}, 'dollar': {'trend_score': 0.4885, 'range_score': 0.237, 'chop_score': 0.4637, 'volatility_percentile': 0.3808, 'consolidation_score': 0.1867}, 'gold': {'trend_score': 0.475, 'range_score': 0.244, 'chop_score': 0.473, 'volatility_percentile': 0.3803, 'consolidation_score': 0.1878}}
2026-05-03 07:49:11,291 INFO Regime[1H mode=ltf_behaviour] score means by year: {2018: {'trend_score': 0.4841, 'range_score': 0.2416, 'chop_score': 0.4687, 'volatility_percentile': 0.3892, 'consolidation_score': 0.1792}, 2019: {'trend_score': 0.4829, 'range_score': 0.2363, 'chop_score': 0.4672, 'volatility_percentile': 0.3736, 'consolidation_score': 0.1949}, 2020: {'trend_score': 0.5462, 'range_score': 0.1954, 'chop_score': 0.4095, 'volatility_percentile': 0.5807, 'consolidation_score': 0.0319}}
2026-05-03 07:49:11,406 INFO Loaded AUDUSD/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
2026-05-03 07:49:11,407 INFO Loaded EURGBP/1H split=val fold=fold_001: 5814 bars (2020-01-06 → 2020-12-31)
2026-05-03 07:49:11,409 INFO Loaded EURJPY/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
2026-05-03 07:49:11,410 INFO Loaded EURUSD/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
2026-05-03 07:49:11,411 INFO Loaded GBPJPY/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
2026-05-03 07:49:11,413 INFO Loaded GBPUSD/1H split=val fold=fold_001: 5814 bars (2020-01-06 → 2020-12-31)
2026-05-03 07:49:11,414 INFO Loaded NZDUSD/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
2026-05-03 07:49:11,415 INFO Loaded USDCAD/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
2026-05-03 07:49:11,416 INFO Loaded USDCHF/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
2026-05-03 07:49:11,417 INFO Loaded USDJPY/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
2026-05-03 07:49:11,419 INFO Loaded XAUUSD/1H split=val fold=fold_001: 5855 bars (2020-01-06 → 2020-12-31)
2026-05-03 07:49:11,425 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:11,427 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:11,428 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:11,428 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:11,428 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:11,430 INFO Loaded AUDUSD/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:11,665 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected AUDUSD — 5765 samples (group=dollar) score_means={'trend_score': 0.4839, 'range_score': 0.2326, 'chop_score': 0.469, 'volatility_percentile': 0.3782, 'consolidation_score': 0.1933}
2026-05-03 07:49:11,789 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:11,791 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:11,792 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:11,792 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:11,793 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:11,794 INFO Loaded EURGBP/1H split=val fold=fold_001: 5814 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:12,030 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected EURGBP — 5764 samples (group=cross) score_means={'trend_score': 0.4874, 'range_score': 0.2329, 'chop_score': 0.4636, 'volatility_percentile': 0.3931, 'consolidation_score': 0.1787}
2026-05-03 07:49:12,144 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:12,146 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:12,147 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:12,148 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:12,148 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:12,150 INFO Loaded EURJPY/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:12,392 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected EURJPY — 5765 samples (group=cross) score_means={'trend_score': 0.4913, 'range_score': 0.2317, 'chop_score': 0.469, 'volatility_percentile': 0.3781, 'consolidation_score': 0.1938}
2026-05-03 07:49:12,517 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:12,520 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:12,520 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:12,521 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:12,521 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:12,523 INFO Loaded EURUSD/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:12,764 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected EURUSD — 5765 samples (group=dollar) score_means={'trend_score': 0.4992, 'range_score': 0.2236, 'chop_score': 0.4521, 'volatility_percentile': 0.3866, 'consolidation_score': 0.1844}
2026-05-03 07:49:12,880 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:12,882 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:12,883 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:12,884 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:12,884 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:12,885 INFO Loaded GBPJPY/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:13,125 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected GBPJPY — 5765 samples (group=cross) score_means={'trend_score': 0.4675, 'range_score': 0.2397, 'chop_score': 0.484, 'volatility_percentile': 0.3868, 'consolidation_score': 0.1954}
2026-05-03 07:49:13,248 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:13,251 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:13,251 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:13,252 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:13,252 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:13,254 INFO Loaded GBPUSD/1H split=val fold=fold_001: 5814 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:13,507 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected GBPUSD — 5764 samples (group=dollar) score_means={'trend_score': 0.4894, 'range_score': 0.2257, 'chop_score': 0.4574, 'volatility_percentile': 0.3874, 'consolidation_score': 0.1784}
2026-05-03 07:49:13,622 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:13,624 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:13,625 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:13,625 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:13,625 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:13,627 INFO Loaded NZDUSD/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:13,867 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected NZDUSD — 5765 samples (group=dollar) score_means={'trend_score': 0.4801, 'range_score': 0.2329, 'chop_score': 0.4674, 'volatility_percentile': 0.3741, 'consolidation_score': 0.1922}
2026-05-03 07:49:13,984 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:13,987 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:13,987 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:13,988 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:13,988 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:13,990 INFO Loaded USDCAD/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:14,228 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected USDCAD — 5765 samples (group=dollar) score_means={'trend_score': 0.4734, 'range_score': 0.2423, 'chop_score': 0.4776, 'volatility_percentile': 0.3777, 'consolidation_score': 0.1886}
2026-05-03 07:49:14,347 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:14,349 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:14,350 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:14,350 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:14,351 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:14,352 INFO Loaded USDCHF/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:14,597 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected USDCHF — 5765 samples (group=dollar) score_means={'trend_score': 0.4788, 'range_score': 0.2389, 'chop_score': 0.4664, 'volatility_percentile': 0.3821, 'consolidation_score': 0.1903}
2026-05-03 07:49:14,715 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:14,718 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:14,718 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:14,719 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:14,719 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:14,721 INFO Loaded USDJPY/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:14,962 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected USDJPY — 5765 samples (group=dollar) score_means={'trend_score': 0.4839, 'range_score': 0.2352, 'chop_score': 0.4713, 'volatility_percentile': 0.3706, 'consolidation_score': 0.1995}
2026-05-03 07:49:15,091 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:15,094 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:15,096 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:15,096 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:15,096 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:15,099 INFO Loaded XAUUSD/1H split=val fold=fold_001: 5855 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:15,356 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected XAUUSD — 5805 samples (group=gold) score_means={'trend_score': 0.4836, 'range_score': 0.2372, 'chop_score': 0.4777, 'volatility_percentile': 0.3611, 'consolidation_score': 0.2086}
2026-05-03 07:49:15,471 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4821, 'range_score': 0.2348, 'chop_score': 0.4722, 'volatility_percentile': 0.386, 'consolidation_score': 0.1893}, 'dollar': {'trend_score': 0.4841, 'range_score': 0.233, 'chop_score': 0.4659, 'volatility_percentile': 0.3795, 'consolidation_score': 0.1895}, 'gold': {'trend_score': 0.4836, 'range_score': 0.2372, 'chop_score': 0.4777, 'volatility_percentile': 0.3611, 'consolidation_score': 0.2086}}
2026-05-03 07:49:15,471 INFO Regime[1H mode=ltf_behaviour] score means by year: {2020: {'trend_score': 0.4835, 'range_score': 0.2339, 'chop_score': 0.4687, 'volatility_percentile': 0.3796, 'consolidation_score': 0.1912}}
2026-05-03 07:49:15,584 INFO Regime phase LTF dataset build fold=fold_001: 9.4s (train=128454 val=63453)
2026-05-03 07:49:15,589 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=31, n_classes=5)
2026-05-03 07:49:15,589 INFO Regime 1H/ltf_behaviour warm start enabled from existing weights
2026-05-03 07:49:15,607 INFO RegimeClassifier[mode=ltf_behaviour]: warm start score head
2026-05-03 07:49:15,607 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-03 07:49:15,892 INFO Regime score epoch  1/50 — tr=0.0065 va=0.0027 mae={'trend_score': 0.0433, 'range_score': 0.0489, 'chop_score': 0.0437, 'volatility_percentile': 0.0247, 'consolidation_score': 0.0405}
2026-05-03 07:49:16,171 INFO Regime score epoch  2/50 — tr=0.0064 va=0.0027
2026-05-03 07:49:16,439 INFO Regime score epoch  3/50 — tr=0.0064 va=0.0027
2026-05-03 07:49:16,713 INFO Regime score epoch  4/50 — tr=0.0064 va=0.0027
2026-05-03 07:49:16,969 INFO Regime score epoch  5/50 — tr=0.0063 va=0.0026 mae={'trend_score': 0.0418, 'range_score': 0.0488, 'chop_score': 0.0434, 'volatility_percentile': 0.0249, 'consolidation_score': 0.0378}
2026-05-03 07:49:17,237 INFO Regime score epoch  6/50 — tr=0.0063 va=0.0026
2026-05-03 07:49:17,506 INFO Regime score epoch  7/50 — tr=0.0062 va=0.0025
2026-05-03 07:49:17,770 INFO Regime score epoch  8/50 — tr=0.0061 va=0.0025
2026-05-03 07:49:18,032 INFO Regime score epoch  9/50 — tr=0.0060 va=0.0024
2026-05-03 07:49:18,301 INFO Regime score epoch 10/50 — tr=0.0060 va=0.0024 mae={'trend_score': 0.0394, 'range_score': 0.0474, 'chop_score': 0.0419, 'volatility_percentile': 0.0234, 'consolidation_score': 0.0345}
2026-05-03 07:49:18,560 INFO Regime score epoch 11/50 — tr=0.0059 va=0.0023
2026-05-03 07:49:18,827 INFO Regime score epoch 12/50 — tr=0.0058 va=0.0023
2026-05-03 07:49:19,087 INFO Regime score epoch 13/50 — tr=0.0057 va=0.0023
2026-05-03 07:49:19,344 INFO Regime score epoch 14/50 — tr=0.0057 va=0.0022
2026-05-03 07:49:19,601 INFO Regime score epoch 15/50 — tr=0.0056 va=0.0022 mae={'trend_score': 0.0371, 'range_score': 0.0452, 'chop_score': 0.04, 'volatility_percentile': 0.0218, 'consolidation_score': 0.0331}
2026-05-03 07:49:19,879 INFO Regime score epoch 16/50 — tr=0.0056 va=0.0022
2026-05-03 07:49:20,152 INFO Regime score epoch 17/50 — tr=0.0055 va=0.0021
2026-05-03 07:49:20,452 INFO Regime score epoch 18/50 — tr=0.0054 va=0.0021
2026-05-03 07:49:20,716 INFO Regime score epoch 19/50 — tr=0.0054 va=0.0021
2026-05-03 07:49:20,981 INFO Regime score epoch 20/50 — tr=0.0054 va=0.0020 mae={'trend_score': 0.035, 'range_score': 0.044, 'chop_score': 0.0379, 'volatility_percentile': 0.021, 'consolidation_score': 0.0325}
2026-05-03 07:49:21,267 INFO Regime score epoch 21/50 — tr=0.0053 va=0.0020
2026-05-03 07:49:21,530 INFO Regime score epoch 22/50 — tr=0.0053 va=0.0020
2026-05-03 07:49:21,805 INFO Regime score epoch 23/50 — tr=0.0052 va=0.0019
2026-05-03 07:49:22,060 INFO Regime score epoch 24/50 — tr=0.0052 va=0.0019
2026-05-03 07:49:22,319 INFO Regime score epoch 25/50 — tr=0.0051 va=0.0019 mae={'trend_score': 0.0334, 'range_score': 0.0434, 'chop_score': 0.036, 'volatility_percentile': 0.02, 'consolidation_score': 0.0307}
2026-05-03 07:49:22,576 INFO Regime score epoch 26/50 — tr=0.0051 va=0.0019
2026-05-03 07:49:22,836 INFO Regime score epoch 27/50 — tr=0.0051 va=0.0019
2026-05-03 07:49:23,107 INFO Regime score epoch 28/50 — tr=0.0051 va=0.0019
2026-05-03 07:49:23,365 INFO Regime score epoch 29/50 — tr=0.0050 va=0.0018
2026-05-03 07:49:23,622 INFO Regime score epoch 30/50 — tr=0.0050 va=0.0018 mae={'trend_score': 0.0321, 'range_score': 0.0429, 'chop_score': 0.0346, 'volatility_percentile': 0.0196, 'consolidation_score': 0.03}
2026-05-03 07:49:23,887 INFO Regime score epoch 31/50 — tr=0.0050 va=0.0018
2026-05-03 07:49:24,149 INFO Regime score epoch 32/50 — tr=0.0050 va=0.0018
2026-05-03 07:49:24,400 INFO Regime score epoch 33/50 — tr=0.0049 va=0.0018
2026-05-03 07:49:24,669 INFO Regime score epoch 34/50 — tr=0.0049 va=0.0018
2026-05-03 07:49:24,928 INFO Regime score epoch 35/50 — tr=0.0049 va=0.0018 mae={'trend_score': 0.0312, 'range_score': 0.0425, 'chop_score': 0.0337, 'volatility_percentile': 0.0193, 'consolidation_score': 0.0297}
2026-05-03 07:49:25,199 INFO Regime score epoch 36/50 — tr=0.0049 va=0.0017
2026-05-03 07:49:25,456 INFO Regime score epoch 37/50 — tr=0.0049 va=0.0017
2026-05-03 07:49:25,731 INFO Regime score epoch 38/50 — tr=0.0049 va=0.0017
2026-05-03 07:49:25,999 INFO Regime score epoch 39/50 — tr=0.0049 va=0.0017
2026-05-03 07:49:26,266 INFO Regime score epoch 40/50 — tr=0.0049 va=0.0017 mae={'trend_score': 0.0306, 'range_score': 0.0425, 'chop_score': 0.0332, 'volatility_percentile': 0.0189, 'consolidation_score': 0.0297}
2026-05-03 07:49:26,531 INFO Regime score epoch 41/50 — tr=0.0049 va=0.0017
2026-05-03 07:49:26,790 INFO Regime score epoch 42/50 — tr=0.0049 va=0.0017
2026-05-03 07:49:27,042 INFO Regime score epoch 43/50 — tr=0.0049 va=0.0017
2026-05-03 07:49:27,302 INFO Regime score epoch 44/50 — tr=0.0048 va=0.0017
2026-05-03 07:49:27,559 INFO Regime score epoch 45/50 — tr=0.0048 va=0.0017 mae={'trend_score': 0.0306, 'range_score': 0.0421, 'chop_score': 0.033, 'volatility_percentile': 0.0189, 'consolidation_score': 0.0303}
2026-05-03 07:49:27,825 INFO Regime score epoch 46/50 — tr=0.0049 va=0.0017
2026-05-03 07:49:28,084 INFO Regime score epoch 47/50 — tr=0.0048 va=0.0017
2026-05-03 07:49:28,349 INFO Regime score epoch 48/50 — tr=0.0048 va=0.0017
2026-05-03 07:49:28,616 INFO Regime score epoch 49/50 — tr=0.0048 va=0.0017
2026-05-03 07:49:28,882 INFO Regime score epoch 50/50 — tr=0.0048 va=0.0017 mae={'trend_score': 0.0305, 'range_score': 0.0423, 'chop_score': 0.0331, 'volatility_percentile': 0.0188, 'consolidation_score': 0.0298}
2026-05-03 07:49:28,921 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0304, 'range_score': 0.042, 'chop_score': 0.033, 'volatility_percentile': 0.019, 'consolidation_score': 0.0293} mse={'trend_score': 0.0015, 'range_score': 0.00279, 'chop_score': 0.00171, 'volatility_percentile': 0.00071, 'consolidation_score': 0.00182} corr={'trend_score': 0.9853, 'range_score': 0.9339, 'chop_score': 0.9785, 'volatility_percentile': 0.9931, 'consolidation_score': 0.981} pred_std={'trend_score': 0.2162, 'range_score': 0.1365, 'chop_score': 0.1803, 'volatility_percentile': 0.2223, 'consolidation_score': 0.214} target_std={'trend_score': 0.2246, 'range_score': 0.1462, 'chop_score': 0.1945, 'volatility_percentile': 0.2261, 'consolidation_score': 0.2201}
2026-05-03 07:49:28,925 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-03 07:49:28,926 INFO Regime phase LTF train fold=fold_001: 13.3s
2026-05-03 07:49:29,042 INFO Regime LTF complete fold=fold_001: score_accuracy=0.969, train=128454 val=63453 mae={'trend_score': 0.0304, 'range_score': 0.042, 'chop_score': 0.033, 'volatility_percentile': 0.019, 'consolidation_score': 0.0293}
2026-05-03 07:49:29,045 INFO Loaded XAUUSD/1H split=train fold=fold_001: 11927 bars (2018-01-04 → 2020-01-03)
2026-05-03 07:49:29,185 INFO Regime[1H mode=ltf_behaviour fold=fold_001] LTF score diagnostics on XAUUSD:
{'trend_score': {'mean': 0.4746, 'q10': 0.1632, 'q50': 0.4665, 'q90': 0.7938}, 'range_score': {'mean': 0.245, 'q10': 0.0601, 'q50': 0.2291, 'q90': 0.4465}, 'chop_score': {'mean': 0.4737, 'q10': 0.2152, 'q50': 0.4751, 'q90': 0.7334}, 'volatility_percentile': {'mean': 0.3804, 'q10': 0.0861, 'q50': 0.3727, 'q90': 0.6763}, 'consolidation_score': {'mean': 0.187, 'q10': 0.0, 'q50': 0.1137, 'q90': 0.5247}}
2026-05-03 07:49:29,189 INFO === Regime rolling fold 3/3: fold_002 ===
2026-05-03 07:49:29,189 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-03 07:49:29,190 INFO Split boundaries loaded fold=fold_002/3 — train 2020-01-06→2022-01-03  val 2022-01-04→2023-01-03  test 2023-08-07→2025-08-05
2026-05-03 07:49:29,191 INFO Loaded AUDUSD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:49:29,191 INFO Loaded EURGBP/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:49:29,192 INFO Loaded EURJPY/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:49:29,193 INFO Loaded EURUSD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:49:29,194 INFO Loaded GBPJPY/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:49:29,195 INFO Loaded GBPUSD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:49:29,195 INFO Loaded NZDUSD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:49:29,196 INFO Loaded USDCAD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:49:29,197 INFO Loaded USDCHF/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:49:29,198 INFO Loaded USDJPY/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:49:29,199 INFO Loaded XAUUSD/4H split=train fold=fold_002: 3180 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:49:29,205 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:29,207 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:29,208 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:29,208 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:29,209 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:29,209 INFO Loaded AUDUSD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:29,424 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 58, 'BIAS_DOWN': 109, 'BIAS_NEUTRAL': 2829}  ambiguous=1636 (total=2996) horizon=12
2026-05-03 07:49:29,427 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected AUDUSD — 2946 samples (group=dollar) score_means={'bias_up_score': 0.0197, 'bias_down_score': 0.037} labels={'BIAS_UP': 58, 'BIAS_DOWN': 109, 'BIAS_NEUTRAL': 2779} clean={'BIAS_UP': 58, 'BIAS_DOWN': 109, 'BIAS_NEUTRAL': 1164}
2026-05-03 07:49:29,552 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:29,556 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:29,556 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:29,557 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:29,557 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:29,558 INFO Loaded EURGBP/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:29,771 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 63, 'BIAS_DOWN': 32, 'BIAS_NEUTRAL': 2901}  ambiguous=1717 (total=2996) horizon=12
2026-05-03 07:49:29,774 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected EURGBP — 2946 samples (group=cross) score_means={'bias_up_score': 0.0214, 'bias_down_score': 0.0109} labels={'BIAS_UP': 63, 'BIAS_DOWN': 32, 'BIAS_NEUTRAL': 2851} clean={'BIAS_UP': 63, 'BIAS_DOWN': 32, 'BIAS_NEUTRAL': 1160}
2026-05-03 07:49:29,898 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:29,901 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:29,901 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:29,902 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:29,902 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:29,903 INFO Loaded EURJPY/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:30,119 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 82, 'BIAS_DOWN': 58, 'BIAS_NEUTRAL': 2856}  ambiguous=1784 (total=2996) horizon=12
2026-05-03 07:49:30,122 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected EURJPY — 2946 samples (group=cross) score_means={'bias_up_score': 0.0278, 'bias_down_score': 0.0197} labels={'BIAS_UP': 82, 'BIAS_DOWN': 58, 'BIAS_NEUTRAL': 2806} clean={'BIAS_UP': 82, 'BIAS_DOWN': 58, 'BIAS_NEUTRAL': 1055}
2026-05-03 07:49:30,242 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:30,246 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:30,247 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:30,247 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:30,248 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:30,249 INFO Loaded EURUSD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:30,481 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 125, 'BIAS_DOWN': 105, 'BIAS_NEUTRAL': 2766}  ambiguous=1697 (total=2996) horizon=12
2026-05-03 07:49:30,484 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected EURUSD — 2946 samples (group=dollar) score_means={'bias_up_score': 0.0424, 'bias_down_score': 0.0356} labels={'BIAS_UP': 125, 'BIAS_DOWN': 105, 'BIAS_NEUTRAL': 2716} clean={'BIAS_UP': 125, 'BIAS_DOWN': 105, 'BIAS_NEUTRAL': 1043}
2026-05-03 07:49:30,605 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:30,610 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:30,611 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:30,611 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:30,611 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:30,612 INFO Loaded GBPJPY/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:30,823 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 112, 'BIAS_DOWN': 61, 'BIAS_NEUTRAL': 2823}  ambiguous=1763 (total=2996) horizon=12
2026-05-03 07:49:30,826 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected GBPJPY — 2946 samples (group=cross) score_means={'bias_up_score': 0.038, 'bias_down_score': 0.0207} labels={'BIAS_UP': 112, 'BIAS_DOWN': 61, 'BIAS_NEUTRAL': 2773} clean={'BIAS_UP': 112, 'BIAS_DOWN': 61, 'BIAS_NEUTRAL': 1028}
2026-05-03 07:49:30,943 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:30,946 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:30,946 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:30,947 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:30,947 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:30,948 INFO Loaded GBPUSD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:31,158 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 71, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 2879}  ambiguous=1724 (total=2996) horizon=12
2026-05-03 07:49:31,161 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected GBPUSD — 2946 samples (group=dollar) score_means={'bias_up_score': 0.0241, 'bias_down_score': 0.0156} labels={'BIAS_UP': 71, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 2829} clean={'BIAS_UP': 71, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 1128}
2026-05-03 07:49:31,275 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:31,277 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:31,278 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:31,278 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:31,278 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:31,279 INFO Loaded NZDUSD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:31,488 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 73, 'BIAS_DOWN': 89, 'BIAS_NEUTRAL': 2834}  ambiguous=1663 (total=2996) horizon=12
2026-05-03 07:49:31,492 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected NZDUSD — 2946 samples (group=dollar) score_means={'bias_up_score': 0.0248, 'bias_down_score': 0.0302} labels={'BIAS_UP': 73, 'BIAS_DOWN': 89, 'BIAS_NEUTRAL': 2784} clean={'BIAS_UP': 73, 'BIAS_DOWN': 89, 'BIAS_NEUTRAL': 1147}
2026-05-03 07:49:31,616 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:31,618 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:31,619 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:31,619 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:31,619 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:31,620 INFO Loaded USDCAD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:31,835 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 108, 'BIAS_DOWN': 102, 'BIAS_NEUTRAL': 2786}  ambiguous=1605 (total=2996) horizon=12
2026-05-03 07:49:31,838 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected USDCAD — 2946 samples (group=dollar) score_means={'bias_up_score': 0.0367, 'bias_down_score': 0.0346} labels={'BIAS_UP': 108, 'BIAS_DOWN': 102, 'BIAS_NEUTRAL': 2736} clean={'BIAS_UP': 108, 'BIAS_DOWN': 102, 'BIAS_NEUTRAL': 1162}
2026-05-03 07:49:31,950 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:31,952 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:31,953 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:31,953 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:31,954 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:31,955 INFO Loaded USDCHF/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:32,166 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 74, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 2831}  ambiguous=1654 (total=2996) horizon=12
2026-05-03 07:49:32,169 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected USDCHF — 2946 samples (group=dollar) score_means={'bias_up_score': 0.0251, 'bias_down_score': 0.0309} labels={'BIAS_UP': 74, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 2781} clean={'BIAS_UP': 74, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 1151}
2026-05-03 07:49:32,287 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:32,289 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:32,290 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:32,290 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:32,290 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:32,292 INFO Loaded USDJPY/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:32,507 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 70, 'BIAS_DOWN': 56, 'BIAS_NEUTRAL': 2870}  ambiguous=1792 (total=2996) horizon=12
2026-05-03 07:49:32,510 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected USDJPY — 2946 samples (group=dollar) score_means={'bias_up_score': 0.0238, 'bias_down_score': 0.019} labels={'BIAS_UP': 70, 'BIAS_DOWN': 56, 'BIAS_NEUTRAL': 2820} clean={'BIAS_UP': 70, 'BIAS_DOWN': 56, 'BIAS_NEUTRAL': 1061}
2026-05-03 07:49:32,634 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:32,641 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:32,642 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:32,643 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:32,643 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:32,644 INFO Loaded XAUUSD/4H split=train fold=fold_002: 3180 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:32,877 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 100, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 3045}  ambiguous=1873 (total=3180) horizon=12
2026-05-03 07:49:32,880 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected XAUUSD — 3130 samples (group=gold) score_means={'bias_up_score': 0.0319, 'bias_down_score': 0.0112} labels={'BIAS_UP': 100, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 2995} clean={'BIAS_UP': 100, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 1144}
2026-05-03 07:49:32,986 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 257, 'BIAS_DOWN': 151, 'BIAS_NEUTRAL': 8430}, 'dollar': {'BIAS_UP': 579, 'BIAS_DOWN': 598, 'BIAS_NEUTRAL': 19445}, 'gold': {'BIAS_UP': 100, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 2995}}
2026-05-03 07:49:32,986 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0291, 'bias_down_score': 0.0171}, 'dollar': {'bias_up_score': 0.0281, 'bias_down_score': 0.029}, 'gold': {'bias_up_score': 0.0319, 'bias_down_score': 0.0112}}
2026-05-03 07:49:32,986 INFO Regime[4H mode=htf_bias] label distribution by year: {2020: {'BIAS_UP': 484, 'BIAS_DOWN': 407, 'BIAS_NEUTRAL': 15040}, 2021: {'BIAS_UP': 452, 'BIAS_DOWN': 377, 'BIAS_NEUTRAL': 15762}, 2022: {'BIAS_UP': 0, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 68}}
2026-05-03 07:49:32,986 INFO Regime[4H mode=htf_bias] score means by year: {2020: {'bias_up_score': 0.0304, 'bias_down_score': 0.0255}, 2021: {'bias_up_score': 0.0272, 'bias_down_score': 0.0227}, 2022: {'bias_up_score': 0.0, 'bias_down_score': 0.0}}
2026-05-03 07:49:33,092 INFO Loaded AUDUSD/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
2026-05-03 07:49:33,093 INFO Loaded EURGBP/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
2026-05-03 07:49:33,094 INFO Loaded EURJPY/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
2026-05-03 07:49:33,095 INFO Loaded EURUSD/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
2026-05-03 07:49:33,095 INFO Loaded GBPJPY/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
2026-05-03 07:49:33,096 INFO Loaded GBPUSD/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
2026-05-03 07:49:33,097 INFO Loaded NZDUSD/4H split=val fold=fold_002: 1235 bars (2022-01-04 → 2022-10-28)
2026-05-03 07:49:33,098 INFO Loaded USDCAD/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
2026-05-03 07:49:33,098 INFO Loaded USDCHF/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
2026-05-03 07:49:33,099 INFO Loaded USDJPY/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
2026-05-03 07:49:33,100 INFO Loaded XAUUSD/4H split=val fold=fold_002: 1596 bars (2022-01-04 → 2023-01-03)
2026-05-03 07:49:33,106 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:33,109 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:33,109 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:33,110 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:33,110 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:33,111 INFO Loaded AUDUSD/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:33,302 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 10, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 1476}  ambiguous=876 (total=1511) horizon=12
2026-05-03 07:49:33,305 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected AUDUSD — 1461 samples (group=dollar) score_means={'bias_up_score': 0.0068, 'bias_down_score': 0.0171} labels={'BIAS_UP': 10, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 1426} clean={'BIAS_UP': 10, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 583}
2026-05-03 07:49:33,421 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:33,425 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:33,425 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:33,426 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:33,426 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:33,427 INFO Loaded EURGBP/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:33,623 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 36, 'BIAS_DOWN': 14, 'BIAS_NEUTRAL': 1461}  ambiguous=814 (total=1511) horizon=12
2026-05-03 07:49:33,625 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected EURGBP — 1461 samples (group=cross) score_means={'bias_up_score': 0.0246, 'bias_down_score': 0.0096} labels={'BIAS_UP': 36, 'BIAS_DOWN': 14, 'BIAS_NEUTRAL': 1411} clean={'BIAS_UP': 36, 'BIAS_DOWN': 14, 'BIAS_NEUTRAL': 614}
2026-05-03 07:49:33,742 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:33,746 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:33,747 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:33,747 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:33,748 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:33,749 INFO Loaded EURJPY/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:33,941 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 92, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 1406}  ambiguous=879 (total=1511) horizon=12
2026-05-03 07:49:33,944 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected EURJPY — 1461 samples (group=cross) score_means={'bias_up_score': 0.063, 'bias_down_score': 0.0089} labels={'BIAS_UP': 92, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 1356} clean={'BIAS_UP': 92, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 498}
2026-05-03 07:49:34,063 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:34,065 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:34,066 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:34,067 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:34,067 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:34,068 INFO Loaded EURUSD/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:34,260 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 11, 'BIAS_DOWN': 64, 'BIAS_NEUTRAL': 1436}  ambiguous=853 (total=1511) horizon=12
2026-05-03 07:49:34,263 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected EURUSD — 1461 samples (group=dollar) score_means={'bias_up_score': 0.0075, 'bias_down_score': 0.0438} labels={'BIAS_UP': 11, 'BIAS_DOWN': 64, 'BIAS_NEUTRAL': 1386} clean={'BIAS_UP': 11, 'BIAS_DOWN': 64, 'BIAS_NEUTRAL': 566}
2026-05-03 07:49:34,381 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:34,384 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:34,384 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:34,385 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:34,385 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:34,386 INFO Loaded GBPJPY/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:34,584 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 75, 'BIAS_DOWN': 26, 'BIAS_NEUTRAL': 1410}  ambiguous=856 (total=1511) horizon=12
2026-05-03 07:49:34,587 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected GBPJPY — 1461 samples (group=cross) score_means={'bias_up_score': 0.0513, 'bias_down_score': 0.0178} labels={'BIAS_UP': 75, 'BIAS_DOWN': 26, 'BIAS_NEUTRAL': 1360} clean={'BIAS_UP': 75, 'BIAS_DOWN': 26, 'BIAS_NEUTRAL': 522}
2026-05-03 07:49:34,705 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:34,707 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:34,708 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:34,709 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:34,709 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:34,710 INFO Loaded GBPUSD/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:34,909 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 4, 'BIAS_DOWN': 70, 'BIAS_NEUTRAL': 1437}  ambiguous=862 (total=1511) horizon=12
2026-05-03 07:49:34,912 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected GBPUSD — 1461 samples (group=dollar) score_means={'bias_up_score': 0.0027, 'bias_down_score': 0.0479} labels={'BIAS_UP': 4, 'BIAS_DOWN': 70, 'BIAS_NEUTRAL': 1387} clean={'BIAS_UP': 4, 'BIAS_DOWN': 70, 'BIAS_NEUTRAL': 555}
2026-05-03 07:49:35,023 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:35,025 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:35,025 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:35,026 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:35,026 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:35,027 INFO Loaded NZDUSD/4H split=val fold=fold_002: 1235 bars (2022-01-04 → 2022-10-28)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:35,210 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 2, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 1166}  ambiguous=696 (total=1235) horizon=12
2026-05-03 07:49:35,213 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected NZDUSD — 1185 samples (group=dollar) score_means={'bias_up_score': 0.0017, 'bias_down_score': 0.0565} labels={'BIAS_UP': 2, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 1116} clean={'BIAS_UP': 2, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 452}
2026-05-03 07:49:35,330 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:35,332 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:35,333 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:35,333 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:35,333 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:35,334 INFO Loaded USDCAD/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:35,529 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 27, 'BIAS_DOWN': 14, 'BIAS_NEUTRAL': 1470}  ambiguous=814 (total=1511) horizon=12
2026-05-03 07:49:35,532 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected USDCAD — 1461 samples (group=dollar) score_means={'bias_up_score': 0.0185, 'bias_down_score': 0.0096} labels={'BIAS_UP': 27, 'BIAS_DOWN': 14, 'BIAS_NEUTRAL': 1420} clean={'BIAS_UP': 27, 'BIAS_DOWN': 14, 'BIAS_NEUTRAL': 637}
2026-05-03 07:49:35,648 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:35,650 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:35,651 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:35,651 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:35,652 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:35,653 INFO Loaded USDCHF/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:35,843 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 103, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 1383}  ambiguous=907 (total=1511) horizon=12
2026-05-03 07:49:35,845 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected USDCHF — 1461 samples (group=dollar) score_means={'bias_up_score': 0.0705, 'bias_down_score': 0.0171} labels={'BIAS_UP': 103, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 1333} clean={'BIAS_UP': 103, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 461}
2026-05-03 07:49:35,957 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:35,960 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:35,960 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:35,961 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:35,961 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:35,962 INFO Loaded USDJPY/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:36,150 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 143, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 1346}  ambiguous=890 (total=1511) horizon=12
2026-05-03 07:49:36,153 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected USDJPY — 1461 samples (group=dollar) score_means={'bias_up_score': 0.0979, 'bias_down_score': 0.0151} labels={'BIAS_UP': 143, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 1296} clean={'BIAS_UP': 143, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 440}
2026-05-03 07:49:36,287 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:36,290 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:36,291 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:36,292 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:36,292 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:36,293 INFO Loaded XAUUSD/4H split=val fold=fold_002: 1596 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:36,505 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 38, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1528}  ambiguous=938 (total=1596) horizon=12
2026-05-03 07:49:36,507 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected XAUUSD — 1546 samples (group=gold) score_means={'bias_up_score': 0.0246, 'bias_down_score': 0.0194} labels={'BIAS_UP': 38, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1478} clean={'BIAS_UP': 38, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 568}
2026-05-03 07:49:36,616 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 203, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 4127}, 'dollar': {'BIAS_UP': 300, 'BIAS_DOWN': 287, 'BIAS_NEUTRAL': 9364}, 'gold': {'BIAS_UP': 38, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1478}}
2026-05-03 07:49:36,616 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0463, 'bias_down_score': 0.0121}, 'dollar': {'bias_up_score': 0.0301, 'bias_down_score': 0.0288}, 'gold': {'bias_up_score': 0.0246, 'bias_down_score': 0.0194}}
2026-05-03 07:49:36,616 INFO Regime[4H mode=htf_bias] label distribution by year: {2022: {'BIAS_UP': 541, 'BIAS_DOWN': 370, 'BIAS_NEUTRAL': 14853}, 2023: {'BIAS_UP': 0, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 116}}
2026-05-03 07:49:36,616 INFO Regime[4H mode=htf_bias] score means by year: {2022: {'bias_up_score': 0.0343, 'bias_down_score': 0.0235}, 2023: {'bias_up_score': 0.0, 'bias_down_score': 0.0}}
2026-05-03 07:49:36,717 INFO Regime phase HTF dataset build fold=fold_002: 7.5s (train=32590 val=15880)
2026-05-03 07:49:36,721 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=24, n_classes=2)
2026-05-03 07:49:36,721 INFO Regime 4H/htf_bias warm start enabled from existing weights
2026-05-03 07:49:36,726 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=32590 val=15880 train_labels={'BIAS_UP': 936, 'BIAS_DOWN': 784, 'BIAS_NEUTRAL': 30870} val_labels={'BIAS_UP': 541, 'BIAS_DOWN': 370, 'BIAS_NEUTRAL': 14969}
2026-05-03 07:49:36,726 INFO RegimeClassifier[mode=htf_bias]: warm start HTF score head
2026-05-03 07:49:36,726 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-03 07:49:36,726 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 20.0, 'bias_down_score': 20.0}
2026-05-03 07:49:37,753 INFO Regime HTF score epoch  1/50 — tr=0.3950 va=0.4662 acc=0.846 bal=0.914 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.959, 'BIAS_DOWN': 0.943, 'BIAS_NEUTRAL': 0.84} precision={'BIAS_UP': 0.316, 'BIAS_DOWN': 0.214, 'BIAS_NEUTRAL': 0.997}
2026-05-03 07:49:38,742 INFO Regime HTF score epoch  2/50 — tr=0.3923 va=0.4703 bal=0.917
2026-05-03 07:49:39,696 INFO Regime HTF score epoch  3/50 — tr=0.3953 va=0.4724 bal=0.920
2026-05-03 07:49:40,770 INFO Regime HTF score epoch  4/50 — tr=0.3958 va=0.4723 bal=0.921
2026-05-03 07:49:41,730 INFO Regime HTF score epoch  5/50 — tr=0.3932 va=0.4752 acc=0.841 bal=0.922 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.959, 'BIAS_DOWN': 0.973, 'BIAS_NEUTRAL': 0.834} precision={'BIAS_UP': 0.317, 'BIAS_DOWN': 0.208, 'BIAS_NEUTRAL': 0.997}
2026-05-03 07:49:42,755 INFO Regime HTF score epoch  6/50 — tr=0.3882 va=0.4783 bal=0.922
2026-05-03 07:49:43,720 INFO Regime HTF score epoch  7/50 — tr=0.3876 va=0.4783 bal=0.921
2026-05-03 07:49:44,796 INFO Regime HTF score epoch  8/50 — tr=0.3851 va=0.4780 bal=0.921
2026-05-03 07:49:45,746 INFO Regime HTF score epoch  9/50 — tr=0.3853 va=0.4757 bal=0.921
2026-05-03 07:49:46,713 INFO Regime HTF score epoch 10/50 — tr=0.3793 va=0.4804 acc=0.836 bal=0.923 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.965, 'BIAS_DOWN': 0.976, 'BIAS_NEUTRAL': 0.828} precision={'BIAS_UP': 0.306, 'BIAS_DOWN': 0.207, 'BIAS_NEUTRAL': 0.998}
2026-05-03 07:49:46,713 INFO Regime HTF score early stop at epoch 10
2026-05-03 07:49:47,615 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.850 margin=0.000 precision={'BIAS_UP': 0.316, 'BIAS_DOWN': 0.214, 'BIAS_NEUTRAL': 0.997} recall={'BIAS_UP': 0.959, 'BIAS_DOWN': 0.943, 'BIAS_NEUTRAL': 0.84} f1={'BIAS_UP': 0.475, 'BIAS_DOWN': 0.349, 'BIAS_NEUTRAL': 0.911} confusion=[[519, 0, 22], [0, 349, 21], [1123, 1279, 12567]] score_mae={'bias_up_score': 0.1935, 'bias_down_score': 0.202} pred_share={'BIAS_UP': 0.1034, 'BIAS_DOWN': 0.1025, 'BIAS_NEUTRAL': 0.7941}
2026-05-03 07:49:47,617 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.316, 'BIAS_DOWN': 0.214, 'BIAS_NEUTRAL': 0.997} min_precision=0.300 recall={'BIAS_UP': 0.959, 'BIAS_DOWN': 0.943, 'BIAS_NEUTRAL': 0.84} min_recall=0.100 f1={'BIAS_UP': 0.475, 'BIAS_DOWN': 0.349, 'BIAS_NEUTRAL': 0.911} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_DOWN'] weak_recall=[] weak_f1=[] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-03 07:49:47,620 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-03 07:49:47,620 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-03 07:49:47,620 INFO Regime phase HTF train fold=fold_002: 10.9s
2026-05-03 07:49:47,737 INFO Regime HTF complete fold=fold_002: acc=0.846 bal=0.914 train=32590 val=15880 per_class={'BIAS_UP': 0.959, 'BIAS_DOWN': 0.943, 'BIAS_NEUTRAL': 0.84} precision={'BIAS_UP': 0.316, 'BIAS_DOWN': 0.214, 'BIAS_NEUTRAL': 0.997} threshold=0.850 margin=0.000
2026-05-03 07:49:47,738 INFO Loaded XAUUSD/4H split=train fold=fold_002: 3180 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:49:47,841 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 100, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 3045}  ambiguous=1873 (total=3180) horizon=12
2026-05-03 07:49:47,843 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 3.8461538461538463, 'BIAS_DOWN': 2.9166666666666665, 'BIAS_NEUTRAL': 78.07692307692308}
2026-05-03 07:49:47,846 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 100, 'mean': 0.002066267653568764, 'mean_over_std': 0.5821881003488305}, 'BIAS_DOWN': {'n': 35, 'mean': -0.002374414438759463, 'mean_over_std': -0.4293108000037002}, 'BIAS_NEUTRAL': {'n': 3044, 'mean': 1.3199249771542425e-05, 'mean_over_std': 0.0031224494254571203}}
2026-05-03 07:49:47,846 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 100, 'mean': 0.002066267653568764, 'mean_over_std': 0.5821881003488305}, 'BIAS_DOWN': {'n': 35, 'mean': -0.002374414438759463, 'mean_over_std': -0.4293108000037002}, 'BIAS_NEUTRAL': {'n': 1172, 'mean': 1.3513765578608547e-05, 'mean_over_std': 0.004104063760480685}}
2026-05-03 07:49:47,850 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-03 07:49:47,852 INFO Loaded AUDUSD/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:49:47,853 INFO Loaded EURGBP/1H split=train fold=fold_002: 11690 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:49:47,855 INFO Loaded EURJPY/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:49:47,857 INFO Loaded EURUSD/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:49:47,858 INFO Loaded GBPJPY/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:49:47,860 INFO Loaded GBPUSD/1H split=train fold=fold_002: 11691 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:49:47,861 INFO Loaded NZDUSD/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:49:47,863 INFO Loaded USDCAD/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:49:47,864 INFO Loaded USDCHF/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:49:47,865 INFO Loaded USDJPY/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:49:47,867 INFO Loaded XAUUSD/1H split=train fold=fold_002: 11775 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:49:47,873 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:47,876 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:47,876 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:47,877 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:47,877 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:47,879 INFO Loaded AUDUSD/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:48,220 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected AUDUSD — 11642 samples (group=dollar) score_means={'trend_score': 0.4899, 'range_score': 0.2319, 'chop_score': 0.4643, 'volatility_percentile': 0.382, 'consolidation_score': 0.1914}
2026-05-03 07:49:48,348 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:48,351 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:48,353 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:48,354 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:48,354 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:48,356 INFO Loaded EURGBP/1H split=train fold=fold_002: 11690 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:48,698 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected EURGBP — 11640 samples (group=cross) score_means={'trend_score': 0.473, 'range_score': 0.2414, 'chop_score': 0.4735, 'volatility_percentile': 0.3812, 'consolidation_score': 0.1876}
2026-05-03 07:49:48,827 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:48,830 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:48,832 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:48,832 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:48,833 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:48,834 INFO Loaded EURJPY/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:49,173 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected EURJPY — 11642 samples (group=cross) score_means={'trend_score': 0.4884, 'range_score': 0.2345, 'chop_score': 0.4693, 'volatility_percentile': 0.3824, 'consolidation_score': 0.1897}
2026-05-03 07:49:49,297 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:49,301 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:49,302 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:49,302 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:49,302 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:49,304 INFO Loaded EURUSD/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:49,643 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected EURUSD — 11642 samples (group=dollar) score_means={'trend_score': 0.4896, 'range_score': 0.2324, 'chop_score': 0.4597, 'volatility_percentile': 0.3849, 'consolidation_score': 0.1841}
2026-05-03 07:49:49,763 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:49,765 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:49,766 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:49,767 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:49,767 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:49,769 INFO Loaded GBPJPY/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:50,119 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected GBPJPY — 11642 samples (group=cross) score_means={'trend_score': 0.4783, 'range_score': 0.2365, 'chop_score': 0.4744, 'volatility_percentile': 0.3783, 'consolidation_score': 0.1953}
2026-05-03 07:49:50,236 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:50,239 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:50,241 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:50,241 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:50,241 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:50,244 INFO Loaded GBPUSD/1H split=train fold=fold_002: 11691 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:50,611 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected GBPUSD — 11641 samples (group=dollar) score_means={'trend_score': 0.4904, 'range_score': 0.231, 'chop_score': 0.4614, 'volatility_percentile': 0.3769, 'consolidation_score': 0.1885}
2026-05-03 07:49:50,725 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:50,727 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:50,727 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:50,728 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:50,728 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:50,730 INFO Loaded NZDUSD/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:51,075 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected NZDUSD — 11642 samples (group=dollar) score_means={'trend_score': 0.4792, 'range_score': 0.2346, 'chop_score': 0.4664, 'volatility_percentile': 0.378, 'consolidation_score': 0.1905}
2026-05-03 07:49:51,196 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:51,198 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:51,199 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:51,199 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:51,199 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:51,201 INFO Loaded USDCAD/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:51,546 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected USDCAD — 11642 samples (group=dollar) score_means={'trend_score': 0.4835, 'range_score': 0.2384, 'chop_score': 0.4682, 'volatility_percentile': 0.3817, 'consolidation_score': 0.1872}
2026-05-03 07:49:51,673 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:51,677 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:51,678 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:51,679 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:51,679 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:51,681 INFO Loaded USDCHF/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:52,034 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected USDCHF — 11642 samples (group=dollar) score_means={'trend_score': 0.4766, 'range_score': 0.2426, 'chop_score': 0.4695, 'volatility_percentile': 0.3844, 'consolidation_score': 0.1868}
2026-05-03 07:49:52,150 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:52,153 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:52,154 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:52,155 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:52,155 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:52,157 INFO Loaded USDJPY/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:52,488 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected USDJPY — 11642 samples (group=dollar) score_means={'trend_score': 0.4905, 'range_score': 0.2324, 'chop_score': 0.4655, 'volatility_percentile': 0.3784, 'consolidation_score': 0.1968}
2026-05-03 07:49:52,619 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:52,622 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:52,623 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:52,624 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:52,624 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:52,627 INFO Loaded XAUUSD/1H split=train fold=fold_002: 11775 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:52,983 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected XAUUSD — 11725 samples (group=gold) score_means={'trend_score': 0.4817, 'range_score': 0.2418, 'chop_score': 0.4772, 'volatility_percentile': 0.3667, 'consolidation_score': 0.1995}
2026-05-03 07:49:53,107 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4799, 'range_score': 0.2375, 'chop_score': 0.4724, 'volatility_percentile': 0.3806, 'consolidation_score': 0.1909}, 'dollar': {'trend_score': 0.4857, 'range_score': 0.2348, 'chop_score': 0.465, 'volatility_percentile': 0.3809, 'consolidation_score': 0.1893}, 'gold': {'trend_score': 0.4817, 'range_score': 0.2418, 'chop_score': 0.4772, 'volatility_percentile': 0.3667, 'consolidation_score': 0.1995}}
2026-05-03 07:49:53,107 INFO Regime[1H mode=ltf_behaviour] score means by year: {2020: {'trend_score': 0.4835, 'range_score': 0.2339, 'chop_score': 0.4687, 'volatility_percentile': 0.3796, 'consolidation_score': 0.1912}, 2021: {'trend_score': 0.484, 'range_score': 0.2383, 'chop_score': 0.4676, 'volatility_percentile': 0.3789, 'consolidation_score': 0.1908}, 2022: {'trend_score': 0.4753, 'range_score': 0.2581, 'chop_score': 0.4747, 'volatility_percentile': 0.5171, 'consolidation_score': 0.0424}}
2026-05-03 07:49:53,225 INFO Loaded AUDUSD/1H split=val fold=fold_002: 5894 bars (2022-01-04 → 2023-01-03)
2026-05-03 07:49:53,226 INFO Loaded EURGBP/1H split=val fold=fold_002: 5897 bars (2022-01-04 → 2023-01-03)
2026-05-03 07:49:53,228 INFO Loaded EURJPY/1H split=val fold=fold_002: 5893 bars (2022-01-04 → 2023-01-03)
2026-05-03 07:49:53,229 INFO Loaded EURUSD/1H split=val fold=fold_002: 5897 bars (2022-01-04 → 2023-01-03)
2026-05-03 07:49:53,231 INFO Loaded GBPJPY/1H split=val fold=fold_002: 5895 bars (2022-01-04 → 2023-01-03)
2026-05-03 07:49:53,232 INFO Loaded GBPUSD/1H split=val fold=fold_002: 5895 bars (2022-01-04 → 2023-01-03)
2026-05-03 07:49:53,233 INFO Loaded NZDUSD/1H split=val fold=fold_002: 4820 bars (2022-01-04 → 2022-10-28)
2026-05-03 07:49:53,234 INFO Loaded USDCAD/1H split=val fold=fold_002: 5897 bars (2022-01-04 → 2023-01-03)
2026-05-03 07:49:53,235 INFO Loaded USDCHF/1H split=val fold=fold_002: 5894 bars (2022-01-04 → 2023-01-03)
2026-05-03 07:49:53,237 INFO Loaded USDJPY/1H split=val fold=fold_002: 5895 bars (2022-01-04 → 2023-01-03)
2026-05-03 07:49:53,239 INFO Loaded XAUUSD/1H split=val fold=fold_002: 5914 bars (2022-01-04 → 2023-01-03)
2026-05-03 07:49:53,245 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:53,247 INFO Loaded AUDUSD/15M split=all fold=latest: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:53,248 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:53,248 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:53,248 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:53,250 INFO Loaded AUDUSD/1H split=val fold=fold_002: 5894 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:53,500 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected AUDUSD — 5844 samples (group=dollar) score_means={'trend_score': 0.4823, 'range_score': 0.2353, 'chop_score': 0.4656, 'volatility_percentile': 0.3905, 'consolidation_score': 0.1791}
2026-05-03 07:49:53,613 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:53,616 INFO Loaded EURGBP/15M split=all fold=latest: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:53,617 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:53,617 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:53,617 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:53,619 INFO Loaded EURGBP/1H split=val fold=fold_002: 5897 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:53,869 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected EURGBP — 5847 samples (group=cross) score_means={'trend_score': 0.4487, 'range_score': 0.2519, 'chop_score': 0.4897, 'volatility_percentile': 0.3945, 'consolidation_score': 0.1768}
2026-05-03 07:49:53,979 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:53,981 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:53,982 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:53,983 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:53,983 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:53,985 INFO Loaded EURJPY/1H split=val fold=fold_002: 5893 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:54,228 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected EURJPY — 5843 samples (group=cross) score_means={'trend_score': 0.5036, 'range_score': 0.2299, 'chop_score': 0.4561, 'volatility_percentile': 0.4037, 'consolidation_score': 0.1685}
2026-05-03 07:49:54,335 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:54,337 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:54,338 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:54,338 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:54,338 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:54,340 INFO Loaded EURUSD/1H split=val fold=fold_002: 5897 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:54,587 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected EURUSD — 5847 samples (group=dollar) score_means={'trend_score': 0.4803, 'range_score': 0.2444, 'chop_score': 0.47, 'volatility_percentile': 0.3951, 'consolidation_score': 0.1781}
2026-05-03 07:49:54,700 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:54,703 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:54,704 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:54,704 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:54,704 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:54,706 INFO Loaded GBPJPY/1H split=val fold=fold_002: 5895 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:54,952 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected GBPJPY — 5845 samples (group=cross) score_means={'trend_score': 0.4766, 'range_score': 0.2379, 'chop_score': 0.4728, 'volatility_percentile': 0.3937, 'consolidation_score': 0.1772}
2026-05-03 07:49:55,060 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:55,062 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:55,063 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:55,063 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:55,063 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:55,065 INFO Loaded GBPUSD/1H split=val fold=fold_002: 5895 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:55,305 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected GBPUSD — 5845 samples (group=dollar) score_means={'trend_score': 0.4678, 'range_score': 0.246, 'chop_score': 0.476, 'volatility_percentile': 0.3971, 'consolidation_score': 0.179}
2026-05-03 07:49:55,411 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:55,413 INFO Loaded NZDUSD/15M split=all fold=latest: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:55,413 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:55,414 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:55,414 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:49:55,415 INFO Loaded NZDUSD/1H split=val fold=fold_002: 4820 bars (2022-01-04 → 2022-10-28)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:55,637 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected NZDUSD — 4770 samples (group=dollar) score_means={'trend_score': 0.4804, 'range_score': 0.2356, 'chop_score': 0.4646, 'volatility_percentile': 0.4152, 'consolidation_score': 0.1616}
2026-05-03 07:49:55,752 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:55,754 INFO Loaded USDCAD/15M split=all fold=latest: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:55,755 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:55,756 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:55,756 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:55,758 INFO Loaded USDCAD/1H split=val fold=fold_002: 5897 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:56,015 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected USDCAD — 5847 samples (group=dollar) score_means={'trend_score': 0.4792, 'range_score': 0.2417, 'chop_score': 0.4731, 'volatility_percentile': 0.3881, 'consolidation_score': 0.1864}
2026-05-03 07:49:56,131 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:56,134 INFO Loaded USDCHF/15M split=all fold=latest: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:56,134 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:56,135 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:56,135 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:56,137 INFO Loaded USDCHF/1H split=val fold=fold_002: 5894 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:56,382 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected USDCHF — 5844 samples (group=dollar) score_means={'trend_score': 0.4698, 'range_score': 0.2432, 'chop_score': 0.4693, 'volatility_percentile': 0.3966, 'consolidation_score': 0.172}
2026-05-03 07:49:56,500 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:56,502 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:56,503 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:56,503 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:56,503 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:49:56,505 INFO Loaded USDJPY/1H split=val fold=fold_002: 5895 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:56,753 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected USDJPY — 5845 samples (group=dollar) score_means={'trend_score': 0.5188, 'range_score': 0.2217, 'chop_score': 0.4472, 'volatility_percentile': 0.398, 'consolidation_score': 0.1782}
2026-05-03 07:49:56,878 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:56,882 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:56,883 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:56,883 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:56,884 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:49:56,886 INFO Loaded XAUUSD/1H split=val fold=fold_002: 5914 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 07:49:57,146 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected XAUUSD — 5864 samples (group=gold) score_means={'trend_score': 0.4904, 'range_score': 0.2349, 'chop_score': 0.465, 'volatility_percentile': 0.3828, 'consolidation_score': 0.1876}
2026-05-03 07:49:57,259 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4763, 'range_score': 0.2399, 'chop_score': 0.4729, 'volatility_percentile': 0.3973, 'consolidation_score': 0.1742}, 'dollar': {'trend_score': 0.4827, 'range_score': 0.2383, 'chop_score': 0.4666, 'volatility_percentile': 0.3968, 'consolidation_score': 0.1767}, 'gold': {'trend_score': 0.4904, 'range_score': 0.2349, 'chop_score': 0.465, 'volatility_percentile': 0.3828, 'consolidation_score': 0.1876}}
2026-05-03 07:49:57,259 INFO Regime[1H mode=ltf_behaviour] score means by year: {2022: {'trend_score': 0.4816, 'range_score': 0.2386, 'chop_score': 0.4682, 'volatility_percentile': 0.3949, 'consolidation_score': 0.1773}, 2023: {'trend_score': 0.4952, 'range_score': 0.2222, 'chop_score': 0.462, 'volatility_percentile': 0.4876, 'consolidation_score': 0.1447}}
2026-05-03 07:49:57,363 INFO Regime phase LTF dataset build fold=fold_002: 9.5s (train=128142 val=63241)
2026-05-03 07:49:57,367 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=31, n_classes=5)
2026-05-03 07:49:57,368 INFO Regime 1H/ltf_behaviour warm start enabled from existing weights
2026-05-03 07:49:57,386 INFO RegimeClassifier[mode=ltf_behaviour]: warm start score head
2026-05-03 07:49:57,386 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-03 07:49:57,661 INFO Regime score epoch  1/50 — tr=0.0049 va=0.0017 mae={'trend_score': 0.0315, 'range_score': 0.0433, 'chop_score': 0.034, 'volatility_percentile': 0.0193, 'consolidation_score': 0.0274}
2026-05-03 07:49:57,914 INFO Regime score epoch  2/50 — tr=0.0049 va=0.0017
2026-05-03 07:49:58,166 INFO Regime score epoch  3/50 — tr=0.0049 va=0.0017
2026-05-03 07:49:58,416 INFO Regime score epoch  4/50 — tr=0.0049 va=0.0017
2026-05-03 07:49:58,669 INFO Regime score epoch  5/50 — tr=0.0049 va=0.0017 mae={'trend_score': 0.0304, 'range_score': 0.0424, 'chop_score': 0.0333, 'volatility_percentile': 0.0189, 'consolidation_score': 0.0278}
2026-05-03 07:49:58,926 INFO Regime score epoch  6/50 — tr=0.0049 va=0.0016
2026-05-03 07:49:59,175 INFO Regime score epoch  7/50 — tr=0.0048 va=0.0016
2026-05-03 07:49:59,426 INFO Regime score epoch  8/50 — tr=0.0048 va=0.0016
2026-05-03 07:49:59,680 INFO Regime score epoch  9/50 — tr=0.0048 va=0.0016
2026-05-03 07:49:59,942 INFO Regime score epoch 10/50 — tr=0.0047 va=0.0016 mae={'trend_score': 0.0286, 'range_score': 0.0415, 'chop_score': 0.0318, 'volatility_percentile': 0.018, 'consolidation_score': 0.0267}
2026-05-03 07:50:00,198 INFO Regime score epoch 11/50 — tr=0.0047 va=0.0015
2026-05-03 07:50:00,478 INFO Regime score epoch 12/50 — tr=0.0047 va=0.0015
2026-05-03 07:50:00,737 INFO Regime score epoch 13/50 — tr=0.0046 va=0.0015
2026-05-03 07:50:01,007 INFO Regime score epoch 14/50 — tr=0.0046 va=0.0015
2026-05-03 07:50:01,270 INFO Regime score epoch 15/50 — tr=0.0046 va=0.0015 mae={'trend_score': 0.0272, 'range_score': 0.0407, 'chop_score': 0.0303, 'volatility_percentile': 0.0176, 'consolidation_score': 0.0261}
2026-05-03 07:50:01,534 INFO Regime score epoch 16/50 — tr=0.0045 va=0.0014
2026-05-03 07:50:01,802 INFO Regime score epoch 17/50 — tr=0.0045 va=0.0014
2026-05-03 07:50:02,075 INFO Regime score epoch 18/50 — tr=0.0045 va=0.0014
2026-05-03 07:50:02,339 INFO Regime score epoch 19/50 — tr=0.0045 va=0.0014
2026-05-03 07:50:02,597 INFO Regime score epoch 20/50 — tr=0.0045 va=0.0014 mae={'trend_score': 0.0258, 'range_score': 0.0397, 'chop_score': 0.0288, 'volatility_percentile': 0.0172, 'consolidation_score': 0.0251}
2026-05-03 07:50:02,857 INFO Regime score epoch 21/50 — tr=0.0044 va=0.0014
2026-05-03 07:50:03,108 INFO Regime score epoch 22/50 — tr=0.0044 va=0.0014
2026-05-03 07:50:03,389 INFO Regime score epoch 23/50 — tr=0.0044 va=0.0013
2026-05-03 07:50:03,654 INFO Regime score epoch 24/50 — tr=0.0044 va=0.0013
2026-05-03 07:50:03,919 INFO Regime score epoch 25/50 — tr=0.0043 va=0.0013 mae={'trend_score': 0.025, 'range_score': 0.0396, 'chop_score': 0.028, 'volatility_percentile': 0.0168, 'consolidation_score': 0.0251}
2026-05-03 07:50:04,188 INFO Regime score epoch 26/50 — tr=0.0043 va=0.0013
2026-05-03 07:50:04,448 INFO Regime score epoch 27/50 — tr=0.0043 va=0.0013
2026-05-03 07:50:04,750 INFO Regime score epoch 28/50 — tr=0.0043 va=0.0013
2026-05-03 07:50:05,019 INFO Regime score epoch 29/50 — tr=0.0043 va=0.0013
2026-05-03 07:50:05,296 INFO Regime score epoch 30/50 — tr=0.0043 va=0.0013 mae={'trend_score': 0.0246, 'range_score': 0.0386, 'chop_score': 0.0274, 'volatility_percentile': 0.0166, 'consolidation_score': 0.0247}
2026-05-03 07:50:05,564 INFO Regime score epoch 31/50 — tr=0.0043 va=0.0013
2026-05-03 07:50:05,842 INFO Regime score epoch 32/50 — tr=0.0043 va=0.0013
2026-05-03 07:50:06,119 INFO Regime score epoch 33/50 — tr=0.0042 va=0.0013
2026-05-03 07:50:06,395 INFO Regime score epoch 34/50 — tr=0.0042 va=0.0013
2026-05-03 07:50:06,677 INFO Regime score epoch 35/50 — tr=0.0042 va=0.0013 mae={'trend_score': 0.0241, 'range_score': 0.0385, 'chop_score': 0.0268, 'volatility_percentile': 0.0163, 'consolidation_score': 0.0242}
2026-05-03 07:50:06,943 INFO Regime score epoch 36/50 — tr=0.0042 va=0.0013
2026-05-03 07:50:07,210 INFO Regime score epoch 37/50 — tr=0.0042 va=0.0013
2026-05-03 07:50:07,489 INFO Regime score epoch 38/50 — tr=0.0042 va=0.0012
2026-05-03 07:50:07,757 INFO Regime score epoch 39/50 — tr=0.0042 va=0.0013
2026-05-03 07:50:08,038 INFO Regime score epoch 40/50 — tr=0.0042 va=0.0012 mae={'trend_score': 0.0241, 'range_score': 0.0382, 'chop_score': 0.0267, 'volatility_percentile': 0.0163, 'consolidation_score': 0.0239}
2026-05-03 07:50:08,309 INFO Regime score epoch 41/50 — tr=0.0042 va=0.0012
2026-05-03 07:50:08,579 INFO Regime score epoch 42/50 — tr=0.0042 va=0.0013
2026-05-03 07:50:08,867 INFO Regime score epoch 43/50 — tr=0.0042 va=0.0012
2026-05-03 07:50:09,144 INFO Regime score epoch 44/50 — tr=0.0042 va=0.0012
2026-05-03 07:50:09,408 INFO Regime score epoch 45/50 — tr=0.0042 va=0.0012 mae={'trend_score': 0.0238, 'range_score': 0.0383, 'chop_score': 0.0265, 'volatility_percentile': 0.0163, 'consolidation_score': 0.0237}
2026-05-03 07:50:09,687 INFO Regime score epoch 46/50 — tr=0.0042 va=0.0012
2026-05-03 07:50:09,970 INFO Regime score epoch 47/50 — tr=0.0042 va=0.0012
2026-05-03 07:50:10,241 INFO Regime score epoch 48/50 — tr=0.0042 va=0.0012
2026-05-03 07:50:10,538 INFO Regime score epoch 49/50 — tr=0.0042 va=0.0012
2026-05-03 07:50:10,799 INFO Regime score epoch 50/50 — tr=0.0042 va=0.0012 mae={'trend_score': 0.0238, 'range_score': 0.0381, 'chop_score': 0.0264, 'volatility_percentile': 0.0163, 'consolidation_score': 0.024}
2026-05-03 07:50:10,839 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0238, 'range_score': 0.0383, 'chop_score': 0.0265, 'volatility_percentile': 0.0163, 'consolidation_score': 0.0237} mse={'trend_score': 0.00094, 'range_score': 0.00235, 'chop_score': 0.00114, 'volatility_percentile': 0.00049, 'consolidation_score': 0.00124} corr={'trend_score': 0.9907, 'range_score': 0.945, 'chop_score': 0.9854, 'volatility_percentile': 0.9948, 'consolidation_score': 0.9858} pred_std={'trend_score': 0.2201, 'range_score': 0.1371, 'chop_score': 0.1817, 'volatility_percentile': 0.2171, 'consolidation_score': 0.2065} target_std={'trend_score': 0.2249, 'range_score': 0.1469, 'chop_score': 0.1924, 'volatility_percentile': 0.2182, 'consolidation_score': 0.2098}
2026-05-03 07:50:10,845 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-03 07:50:10,846 INFO Regime phase LTF train fold=fold_002: 13.5s
2026-05-03 07:50:10,954 INFO Regime LTF complete fold=fold_002: score_accuracy=0.974, train=128142 val=63241 mae={'trend_score': 0.0238, 'range_score': 0.0383, 'chop_score': 0.0265, 'volatility_percentile': 0.0163, 'consolidation_score': 0.0237}
2026-05-03 07:50:10,957 INFO Loaded XAUUSD/1H split=train fold=fold_002: 11775 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:50:11,102 INFO Regime[1H mode=ltf_behaviour fold=fold_002] LTF score diagnostics on XAUUSD:
{'trend_score': {'mean': 0.4815, 'q10': 0.1762, 'q50': 0.4731, 'q90': 0.7982}, 'range_score': {'mean': 0.2428, 'q10': 0.0514, 'q50': 0.2289, 'q90': 0.4497}, 'chop_score': {'mean': 0.4776, 'q10': 0.2185, 'q50': 0.4756, 'q90': 0.7418}, 'volatility_percentile': {'mean': 0.3668, 'q10': 0.0788, 'q50': 0.3514, 'q90': 0.6764}, 'consolidation_score': {'mean': 0.1986, 'q10': 0.0, 'q50': 0.1291, 'q90': 0.5425}}
2026-05-03 07:50:11,105 INFO Regime retrain total: 205.9s (722582 train+val samples)
2026-05-03 07:50:11,124 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-03 07:50:11,124 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-03 07:50:11,124 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-03 07:50:11,124 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-05-03 07:50:11,125 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-05-03 07:50:11,125 INFO Retrain complete. Total wall-clock: 205.9s
2026-05-03 07:50:16,015 INFO Model regime: SUCCESS
2026-05-03 07:50:16,015 INFO --- Training gru ---
2026-05-03 07:50:16,016 INFO Running retrain --model gru
2026-05-03 07:50:16,450 INFO retrain environment: KAGGLE
2026-05-03 07:50:18,142 INFO Device: CUDA (2 GPU(s))
2026-05-03 07:50:18,154 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-03 07:50:18,154 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-03 07:50:18,155 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-03 07:50:18,157 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-03 07:50:18,157 INFO Retrain data split: train
2026-05-03 07:50:18,157 INFO Retrain rolling fold selector: latest
2026-05-03 07:50:18,158 INFO === GRU-LSTM retrain (timeframes: ['15M']) ===
2026-05-03 07:50:18,305 INFO NumExpr defaulting to 4 threads.
2026-05-03 07:50:18,514 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-03 07:50:18,514 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-03 07:50:18,514 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-03 07:50:18,514 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['15M']
2026-05-03 07:50:18,515 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260503_075018
2026-05-03 07:50:18,517 WARNING WeightsManifest: no manifest at /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json — assuming compatible (run a full retrain to generate one)
2026-05-03 07:50:18,517 INFO GRU cold start: no compatible existing weights found
2026-05-03 07:50:18,680 INFO Loaded AUDUSD/5M split=all fold=latest: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:18,704 INFO Loaded AUDUSD/1H split=all fold=latest: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:18,721 INFO Loaded AUDUSD/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:18,729 INFO Loaded AUDUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:18,781 INFO Split boundaries loaded fold=fold_002/3 — train 2020-01-06→2022-01-03  val 2022-01-04→2023-01-03  test 2023-08-07→2025-08-05
2026-05-03 07:50:18,784 INFO Loaded AUDUSD/15M split=train fold=latest: 46766 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:50:19,041 INFO Loaded EURGBP/5M split=all fold=latest: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:19,061 INFO Loaded EURGBP/1H split=all fold=latest: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:19,077 INFO Loaded EURGBP/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:19,086 INFO Loaded EURGBP/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:19,124 INFO Loaded EURGBP/15M split=train fold=latest: 46759 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:50:19,354 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:19,374 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:19,389 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:19,396 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:19,434 INFO Loaded EURJPY/15M split=train fold=latest: 46767 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:50:19,662 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:19,682 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:19,697 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:19,704 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:19,740 INFO Loaded EURUSD/15M split=train fold=latest: 46766 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:50:19,987 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:20,008 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:20,024 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:20,032 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:20,071 INFO Loaded GBPJPY/15M split=train fold=latest: 46765 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:50:20,304 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:20,325 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:20,345 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:20,352 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:20,391 INFO Loaded GBPUSD/15M split=train fold=latest: 46764 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:50:20,584 INFO Loaded NZDUSD/5M split=all fold=latest: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:50:20,601 INFO Loaded NZDUSD/1H split=all fold=latest: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:50:20,616 INFO Loaded NZDUSD/4H split=all fold=latest: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:50:20,622 INFO Loaded NZDUSD/1D split=all fold=latest: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-03 07:50:20,654 INFO Loaded NZDUSD/15M split=train fold=latest: 46766 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:50:20,870 INFO Loaded USDCAD/5M split=all fold=latest: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:20,892 INFO Loaded USDCAD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:20,907 INFO Loaded USDCAD/4H split=all fold=latest: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:20,915 INFO Loaded USDCAD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:20,952 INFO Loaded USDCAD/15M split=train fold=latest: 46767 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:50:21,165 INFO Loaded USDCHF/5M split=all fold=latest: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:21,184 INFO Loaded USDCHF/1H split=all fold=latest: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:21,198 INFO Loaded USDCHF/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:21,206 INFO Loaded USDCHF/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:21,243 INFO Loaded USDCHF/15M split=train fold=latest: 46763 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:50:21,466 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:21,486 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:21,500 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:21,507 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-03 07:50:21,546 INFO Loaded USDJPY/15M split=train fold=latest: 46768 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:50:21,887 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:50:21,912 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:50:21,929 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:50:21,939 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-03 07:50:22,004 INFO Loaded XAUUSD/15M split=train fold=latest: 47096 bars (2020-01-06 → 2022-01-03)
2026-05-03 07:50:22,139 INFO train_multi: 11 segments, ~500055 total bars
2026-05-03 07:50:22,435 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-05-03 07:50:22,436 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-05-03 07:50:22,436 INFO train_multi: training ALL 11 segments across TFs ['15M'] in one combined pass
2026-05-03 07:50:22,436 INFO train_multi: building combined dataset for TF=ALL (11 segments)
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
2026-05-03 07:50:28,426 INFO train_multi TF=ALL: 499725 sequences across 11 segments
2026-05-03 07:50:28,426 INFO train_multi TF=ALL: estimated peak RAM = 8515 MB (train=399775 val=99950 n_feat=71 seq_len=30)
2026-05-03 07:50:29,509 INFO train_multi TF=ALL: train=399775 val=99950 (4264 MB tensors)
2026-05-03 07:50:33,622 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=18, min_epochs=22)
2026-05-03 07:50:48,205 INFO train_multi TF=ALL epoch 1/50 train=0.8423 val=0.8331 dir_acc=0.496 dir_n=99950
2026-05-03 07:50:48,213 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:50:48,213 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:50:48,213 INFO train_multi TF=ALL: new best val=0.8331 — saved
2026-05-03 07:51:00,415 INFO train_multi TF=ALL epoch 2/50 train=0.8130 val=0.7777 dir_acc=0.496 dir_n=99950
2026-05-03 07:51:00,420 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:51:00,421 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:51:00,421 INFO train_multi TF=ALL: new best val=0.7777 — saved
2026-05-03 07:51:12,526 INFO train_multi TF=ALL epoch 3/50 train=0.7283 val=0.7004 dir_acc=0.496 dir_n=99950
2026-05-03 07:51:12,532 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:51:12,532 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:51:12,532 INFO train_multi TF=ALL: new best val=0.7004 — saved
2026-05-03 07:51:24,311 INFO train_multi TF=ALL epoch 4/50 train=0.7069 val=0.7001 dir_acc=0.504 dir_n=99950
2026-05-03 07:51:24,316 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:51:24,317 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:51:24,317 INFO train_multi TF=ALL: new best val=0.7001 — saved
2026-05-03 07:51:35,932 INFO train_multi TF=ALL epoch 5/50 train=0.7059 val=0.6998 dir_acc=0.504 dir_n=99950
2026-05-03 07:51:35,937 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:51:35,937 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:51:35,938 INFO train_multi TF=ALL: new best val=0.6998 — saved
2026-05-03 07:51:47,777 INFO train_multi TF=ALL epoch 6/50 train=0.7051 val=0.6993 dir_acc=0.495 dir_n=99950
2026-05-03 07:51:47,782 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:51:47,782 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:51:47,782 INFO train_multi TF=ALL: new best val=0.6993 — saved
2026-05-03 07:51:59,197 INFO train_multi TF=ALL epoch 7/50 train=0.7043 val=0.6987 dir_acc=0.504 dir_n=99950
2026-05-03 07:51:59,202 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:51:59,202 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:51:59,202 INFO train_multi TF=ALL: new best val=0.6987 — saved
2026-05-03 07:52:11,041 INFO train_multi TF=ALL epoch 8/50 train=0.7033 val=0.6981 dir_acc=0.503 dir_n=99950
2026-05-03 07:52:11,046 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:52:11,046 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:52:11,046 INFO train_multi TF=ALL: new best val=0.6981 — saved
2026-05-03 07:52:22,733 INFO train_multi TF=ALL epoch 9/50 train=0.7025 val=0.6978 dir_acc=0.496 dir_n=99950
2026-05-03 07:52:22,738 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:52:22,738 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:52:22,738 INFO train_multi TF=ALL: new best val=0.6978 — saved
2026-05-03 07:52:34,291 INFO train_multi TF=ALL epoch 10/50 train=0.7020 val=0.6976 dir_acc=0.500 dir_n=99950
2026-05-03 07:52:34,296 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:52:34,296 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:52:34,296 INFO train_multi TF=ALL: new best val=0.6976 — saved
2026-05-03 07:52:45,636 INFO train_multi TF=ALL epoch 11/50 train=0.7015 val=0.6975 dir_acc=0.497 dir_n=99950
2026-05-03 07:52:45,641 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:52:45,641 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:52:45,641 INFO train_multi TF=ALL: new best val=0.6975 — saved
2026-05-03 07:52:57,343 INFO train_multi TF=ALL epoch 12/50 train=0.7013 val=0.6973 dir_acc=0.500 dir_n=99950
2026-05-03 07:52:57,348 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:52:57,348 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:52:57,348 INFO train_multi TF=ALL: new best val=0.6973 — saved
2026-05-03 07:53:09,022 INFO train_multi TF=ALL epoch 13/50 train=0.7011 val=0.6972 dir_acc=0.500 dir_n=99950
2026-05-03 07:53:09,027 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:53:09,027 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:53:09,027 INFO train_multi TF=ALL: new best val=0.6972 — saved
2026-05-03 07:53:20,608 INFO train_multi TF=ALL epoch 14/50 train=0.7008 val=0.6971 dir_acc=0.501 dir_n=99950
2026-05-03 07:53:20,613 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:53:20,613 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:53:20,613 INFO train_multi TF=ALL: new best val=0.6971 — saved
2026-05-03 07:53:32,050 INFO train_multi TF=ALL epoch 15/50 train=0.7005 val=0.6969 dir_acc=0.503 dir_n=99950
2026-05-03 07:53:32,056 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:53:32,056 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:53:32,056 INFO train_multi TF=ALL: new best val=0.6969 — saved
2026-05-03 07:53:43,859 INFO train_multi TF=ALL epoch 16/50 train=0.7004 val=0.6966 dir_acc=0.503 dir_n=99950
2026-05-03 07:53:43,864 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:53:43,864 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:53:43,864 INFO train_multi TF=ALL: new best val=0.6966 — saved
2026-05-03 07:53:55,608 INFO train_multi TF=ALL epoch 17/50 train=0.7000 val=0.6963 dir_acc=0.513 dir_n=99950
2026-05-03 07:53:55,612 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:53:55,613 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:53:55,613 INFO train_multi TF=ALL: new best val=0.6963 — saved
2026-05-03 07:54:06,945 INFO train_multi TF=ALL epoch 18/50 train=0.6994 val=0.6957 dir_acc=0.517 dir_n=99950
2026-05-03 07:54:06,949 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:54:06,950 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:54:06,950 INFO train_multi TF=ALL: new best val=0.6957 — saved
2026-05-03 07:54:18,027 INFO train_multi TF=ALL epoch 19/50 train=0.6988 val=0.6951 dir_acc=0.522 dir_n=99950
2026-05-03 07:54:18,032 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:54:18,033 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:54:18,033 INFO train_multi TF=ALL: new best val=0.6951 — saved
2026-05-03 07:54:29,003 INFO train_multi TF=ALL epoch 20/50 train=0.6982 val=0.6948 dir_acc=0.522 dir_n=99950
2026-05-03 07:54:29,008 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:54:29,009 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:54:29,009 INFO train_multi TF=ALL: new best val=0.6948 — saved
2026-05-03 07:54:39,941 INFO train_multi TF=ALL epoch 21/50 train=0.6979 val=0.6942 dir_acc=0.526 dir_n=99950
2026-05-03 07:54:39,946 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:54:39,946 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:54:39,946 INFO train_multi TF=ALL: new best val=0.6942 — saved
2026-05-03 07:54:50,967 INFO train_multi TF=ALL epoch 22/50 train=0.6975 val=0.6939 dir_acc=0.529 dir_n=99950
2026-05-03 07:54:50,972 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:54:50,972 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:54:50,973 INFO train_multi TF=ALL: new best val=0.6939 — saved
2026-05-03 07:55:01,869 INFO train_multi TF=ALL epoch 23/50 train=0.6970 val=0.6939 dir_acc=0.528 dir_n=99950
2026-05-03 07:55:01,874 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:55:01,874 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:55:01,874 INFO train_multi TF=ALL: new best val=0.6939 — saved
2026-05-03 07:55:12,855 INFO train_multi TF=ALL epoch 24/50 train=0.6961 val=0.6925 dir_acc=0.535 dir_n=99950
2026-05-03 07:55:12,860 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:55:12,860 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:55:12,860 INFO train_multi TF=ALL: new best val=0.6925 — saved
2026-05-03 07:55:23,968 INFO train_multi TF=ALL epoch 25/50 train=0.6938 val=0.6864 dir_acc=0.561 dir_n=99950
2026-05-03 07:55:23,973 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:55:23,973 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:55:23,973 INFO train_multi TF=ALL: new best val=0.6864 — saved
2026-05-03 07:55:35,171 INFO train_multi TF=ALL epoch 26/50 train=0.6879 val=0.6764 dir_acc=0.586 dir_n=99950
2026-05-03 07:55:35,176 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:55:35,176 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:55:35,176 INFO train_multi TF=ALL: new best val=0.6764 — saved
2026-05-03 07:55:46,125 INFO train_multi TF=ALL epoch 27/50 train=0.6808 val=0.6715 dir_acc=0.596 dir_n=99950
2026-05-03 07:55:46,130 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:55:46,130 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:55:46,130 INFO train_multi TF=ALL: new best val=0.6715 — saved
2026-05-03 07:55:57,250 INFO train_multi TF=ALL epoch 28/50 train=0.6756 val=0.6657 dir_acc=0.609 dir_n=99950
2026-05-03 07:55:57,255 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:55:57,255 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:55:57,256 INFO train_multi TF=ALL: new best val=0.6657 — saved
2026-05-03 07:56:08,179 INFO train_multi TF=ALL epoch 29/50 train=0.6719 val=0.6628 dir_acc=0.615 dir_n=99950
2026-05-03 07:56:08,184 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:56:08,184 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:56:08,184 INFO train_multi TF=ALL: new best val=0.6628 — saved
2026-05-03 07:56:19,186 INFO train_multi TF=ALL epoch 30/50 train=0.6689 val=0.6584 dir_acc=0.622 dir_n=99950
2026-05-03 07:56:19,191 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:56:19,191 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:56:19,191 INFO train_multi TF=ALL: new best val=0.6584 — saved
2026-05-03 07:56:30,311 INFO train_multi TF=ALL epoch 31/50 train=0.6671 val=0.6566 dir_acc=0.626 dir_n=99950
2026-05-03 07:56:30,318 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:56:30,318 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:56:30,318 INFO train_multi TF=ALL: new best val=0.6566 — saved
2026-05-03 07:56:41,360 INFO train_multi TF=ALL epoch 32/50 train=0.6652 val=0.6557 dir_acc=0.627 dir_n=99950
2026-05-03 07:56:41,365 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:56:41,365 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:56:41,365 INFO train_multi TF=ALL: new best val=0.6557 — saved
2026-05-03 07:56:52,429 INFO train_multi TF=ALL epoch 33/50 train=0.6639 val=0.6544 dir_acc=0.628 dir_n=99950
2026-05-03 07:56:52,434 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:56:52,434 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:56:52,434 INFO train_multi TF=ALL: new best val=0.6544 — saved
2026-05-03 07:57:03,345 INFO train_multi TF=ALL epoch 34/50 train=0.6623 val=0.6541 dir_acc=0.629 dir_n=99950
2026-05-03 07:57:03,350 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:57:03,350 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:57:03,350 INFO train_multi TF=ALL: new best val=0.6541 — saved
2026-05-03 07:57:14,428 INFO train_multi TF=ALL epoch 35/50 train=0.6615 val=0.6536 dir_acc=0.627 dir_n=99950
2026-05-03 07:57:14,433 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:57:14,433 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:57:14,433 INFO train_multi TF=ALL: new best val=0.6536 — saved
2026-05-03 07:57:25,514 INFO train_multi TF=ALL epoch 36/50 train=0.6604 val=0.6528 dir_acc=0.628 dir_n=99950
2026-05-03 07:57:25,519 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:57:25,519 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:57:25,519 INFO train_multi TF=ALL: new best val=0.6528 — saved
2026-05-03 07:57:36,475 INFO train_multi TF=ALL epoch 37/50 train=0.6593 val=0.6521 dir_acc=0.629 dir_n=99950
2026-05-03 07:57:36,480 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:57:36,480 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:57:36,480 INFO train_multi TF=ALL: new best val=0.6521 — saved
2026-05-03 07:57:47,448 INFO train_multi TF=ALL epoch 38/50 train=0.6592 val=0.6537 dir_acc=0.627 dir_n=99950
2026-05-03 07:57:58,429 INFO train_multi TF=ALL epoch 39/50 train=0.6585 val=0.6524 dir_acc=0.629 dir_n=99950
2026-05-03 07:58:09,390 INFO train_multi TF=ALL epoch 40/50 train=0.6574 val=0.6514 dir_acc=0.630 dir_n=99950
2026-05-03 07:58:09,395 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:58:09,395 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:58:09,395 INFO train_multi TF=ALL: new best val=0.6514 — saved
2026-05-03 07:58:20,304 INFO train_multi TF=ALL epoch 41/50 train=0.6570 val=0.6515 dir_acc=0.630 dir_n=99950
2026-05-03 07:58:31,435 INFO train_multi TF=ALL epoch 42/50 train=0.6564 val=0.6520 dir_acc=0.630 dir_n=99950
2026-05-03 07:58:42,495 INFO train_multi TF=ALL epoch 43/50 train=0.6558 val=0.6503 dir_acc=0.632 dir_n=99950
2026-05-03 07:58:42,500 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:58:42,500 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:58:42,500 INFO train_multi TF=ALL: new best val=0.6503 — saved
2026-05-03 07:58:53,468 INFO train_multi TF=ALL epoch 44/50 train=0.6556 val=0.6505 dir_acc=0.633 dir_n=99950
2026-05-03 07:59:04,458 INFO train_multi TF=ALL epoch 45/50 train=0.6551 val=0.6504 dir_acc=0.632 dir_n=99950
2026-05-03 07:59:15,356 INFO train_multi TF=ALL epoch 46/50 train=0.6546 val=0.6504 dir_acc=0.632 dir_n=99950
2026-05-03 07:59:26,267 INFO train_multi TF=ALL epoch 47/50 train=0.6541 val=0.6495 dir_acc=0.633 dir_n=99950
2026-05-03 07:59:26,273 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:59:26,273 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:59:26,273 INFO train_multi TF=ALL: new best val=0.6495 — saved
2026-05-03 07:59:37,339 INFO train_multi TF=ALL epoch 48/50 train=0.6535 val=0.6499 dir_acc=0.633 dir_n=99950
2026-05-03 07:59:48,366 INFO train_multi TF=ALL epoch 49/50 train=0.6531 val=0.6509 dir_acc=0.630 dir_n=99950
2026-05-03 07:59:59,287 INFO train_multi TF=ALL epoch 50/50 train=0.6529 val=0.6491 dir_acc=0.633 dir_n=99950
2026-05-03 07:59:59,292 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-03 07:59:59,292 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 07:59:59,292 INFO train_multi TF=ALL: new best val=0.6491 — saved
2026-05-03 07:59:59,433 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-03 07:59:59,433 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-03 07:59:59,433 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-03 07:59:59,433 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-05-03 07:59:59,434 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-05-03 07:59:59,434 INFO Retrain complete. Total wall-clock: 581.3s
2026-05-03 08:00:01,321 INFO Model gru: SUCCESS
2026-05-03 08:00:01,321 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-03 08:00:01,321 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-03 08:00:01,322 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-03 08:00:01,322 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-05-03 08:00:01,322 INFO   [DORMANT] rl_ppo — RL_ENABLED=false, skipping
2026-05-03 08:00:01,322 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-05-03 08:00:01,322 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer']
2026-05-03 08:00:01,323 INFO Saved 9 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Clean Quality/RL source: Backtest on train window ===
  START Train-window backtest for Quality/RL labels
2026-05-03 08:00:02,113 INFO === STEP 6: BACKTEST (train) ===
2026-05-03 08:00:02,114 INFO BT_WINDOW=train — train-window backtest: 2020-01-06 → 2022-01-03 (clean Quality/RL labels)
2026-05-03 08:00:02,114 INFO ================================================================
  ROUND 0 / 3
================================================================
2026-05-03 08:00:02,114 INFO Round 0 — running backtest: 2020-01-06 → 2022-01-03 (ml_trader, shared ML cache)
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
2026-05-03 08:02:11,288 ERROR _precompute_ml_cache failed for GBPUSD: ML cache model LTF score frame has gaps for GBPUSD
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:02:11,522 ERROR _precompute_ml_cache failed for USDJPY: ML cache model LTF score frame has gaps for USDJPY
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:02:11,718 ERROR _precompute_ml_cache failed for AUDUSD: ML cache model LTF score frame has gaps for AUDUSD
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:02:12,047 ERROR _precompute_ml_cache failed for EURUSD: ML cache model LTF score frame has gaps for EURUSD
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
2026-05-03 08:02:14,341 ERROR _precompute_ml_cache failed for NZDUSD: ML cache model LTF score frame has gaps for NZDUSD
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:02:14,562 ERROR _precompute_ml_cache failed for USDCAD: ML cache model LTF score frame has gaps for USDCAD
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:02:14,955 ERROR _precompute_ml_cache failed for USDCHF: ML cache model LTF score frame has gaps for USDCHF
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-03 08:02:15,230 ERROR _precompute_ml_cache failed for EURGBP: ML cache model LTF score frame has gaps for EURGBP
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
2026-05-03 08:02:16,769 ERROR _precompute_ml_cache failed for EURJPY: ML cache model LTF score frame has gaps for EURJPY
2026-05-03 08:02:16,887 ERROR _precompute_ml_cache failed for GBPJPY: ML cache model LTF score frame has gaps for GBPJPY
2026-05-03 08:02:16,969 ERROR _precompute_ml_cache failed for XAUUSD: ML cache model LTF score frame has gaps for XAUUSD
Traceback (most recent call last):
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 2305, in _backtest_trader
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
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 2288, in _build_cache_sym
    return sym, _precompute_ml_cache(
                ^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 1497, in _precompute_ml_cache
    raise RuntimeError(f"ML cache model LTF score frame has gaps for {symbol}")
RuntimeError: ML cache model LTF score frame has gaps for GBPUSD

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 3765, in <module>
    main()
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 3579, in main
    result = _backtest_trader("ml_trader", symbols, pm, bt_start, bt_end,
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 2309, in _backtest_trader
    raise RuntimeError(f"ML cache build failed for {sym}: {exc}") from exc
RuntimeError: ML cache build failed for GBPUSD: ML cache model LTF score frame has gaps for GBPUSD
2026-05-03 08:02:17,597 ERROR Backtest failed (rc=1) — check trading-engine/logs/backtest_*.log
2026-05-03 08:02:17,597 ERROR Round 0 backtest failed: backtest exited 1
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