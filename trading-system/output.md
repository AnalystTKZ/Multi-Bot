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
2026-05-09 07:41:19,479 INFO Loading feature-engineered data...
2026-05-09 07:41:19,970 INFO Loaded 221743 rows, 202 features
2026-05-09 07:41:19,971 INFO Data span: 2016-01-04 -> 2025-08-05  (9.6 years)
2026-05-09 07:41:19,971 INFO Fold 000 train 2016-01-04 -> 2018-01-03 (47088 bars), val 2018-01-04 -> 2019-01-03 (23448 bars)
2026-05-09 07:41:19,972 INFO Fold 001 train 2016-01-04 -> 2019-01-03 (70536 bars), val 2019-01-04 -> 2020-01-03 (23377 bars)
2026-05-09 07:41:19,972 INFO Fold 002 train 2016-01-04 -> 2020-01-03 (93913 bars), val 2020-01-06 -> 2020-12-31 (23259 bars)
2026-05-09 07:41:19,972 INFO Fold 003 train 2016-01-04 -> 2020-12-31 (117172 bars), val 2021-01-04 -> 2022-01-03 (23507 bars)
2026-05-09 07:41:19,972 INFO Fold 004 train 2016-01-04 -> 2022-01-03 (140679 bars), val 2022-01-04 -> 2023-01-03 (23588 bars)
2026-05-09 07:41:19,973 INFO Fold 005 train 2016-01-04 -> 2022-08-05 (154539 bars), val 2022-08-05 -> 2023-08-04 (20412 bars)
2026-05-09 07:41:19,973 INFO No leakage confirmed: every expanding fold ends before final 2-year blind test

=== SPLIT COMPLETE (EXPANDING CALENDAR, no shuffling) ===
  Folds:            6 expanding folds (min 2y train + 1y val, step=1y)
  Selected:   fold_005 for train.parquet / validation.parquet aliases
  Train:      154,539 bars  2016-01-04 -> 2022-08-05
  Validation:  20,412 bars  2022-08-05 -> 2023-08-04
  Test:        46,792 bars  2023-08-07 -> 2025-08-05  <- Blind / Round 2
  Features:   202
  Leakage check: PASS
  DONE  Step 5 - Split

  Data split (expanding_calendar):
    train         154539 bars  2016-01-04 → 2022-08-05
    validation     20412 bars  2022-08-05 → 2023-08-04
    test           46792 bars  2023-08-07 → 2025-08-05

=== Phase 7a: Train GRU + Regime (train set only) ===
  START Step 7a - GRU+Regime
2026-05-09 07:41:28,937 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-05-09 07:41:28,937 INFO --- Training regime ---
2026-05-09 07:41:28,938 INFO Running retrain --model regime
2026-05-09 07:41:29,113 INFO retrain environment: KAGGLE
2026-05-09 07:41:30,681 INFO Device: CUDA (2 GPU(s))
2026-05-09 07:41:30,693 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-09 07:41:30,693 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-09 07:41:30,693 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-09 07:41:30,693 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-09 07:41:30,693 INFO Retrain data split: train
2026-05-09 07:41:30,693 INFO Retrain rolling fold selector: latest
2026-05-09 07:41:30,694 INFO === RegimeClassifier retrain (hierarchical: HTF 2-score bias + LTF 5-score behaviour) ===
2026-05-09 07:41:30,864 INFO NumExpr defaulting to 4 threads.
2026-05-09 07:41:31,077 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-09 07:41:31,077 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-09 07:41:31,077 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-09 07:41:31,077 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-09 07:41:31,131 INFO Regime rolling folds selected: ['fold_005']
2026-05-09 07:41:31,132 INFO === Regime rolling fold 1/1: fold_005 ===
2026-05-09 07:41:31,132 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-09 07:41:31,169 INFO Split boundaries loaded fold=fold_005/6 — train 2016-01-04→2022-08-05  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-09 07:41:31,170 INFO Loaded XAUUSD/4H split=train fold=fold_005: 10521 bars (2016-01-04 → 2022-08-05)
2026-05-09 07:41:31,184 INFO Loaded EURUSD/4H split=train fold=fold_005: 9913 bars (2016-01-04 → 2022-08-05)
2026-05-09 07:41:31,198 INFO Loaded USDJPY/4H split=train fold=fold_005: 9914 bars (2016-01-04 → 2022-08-05)
2026-05-09 07:41:31,213 INFO Loaded EURJPY/4H split=train fold=fold_005: 9913 bars (2016-01-04 → 2022-08-05)
2026-05-09 07:41:31,227 INFO Loaded GBPJPY/4H split=train fold=fold_005: 9914 bars (2016-01-04 → 2022-08-05)
2026-05-09 07:41:31,240 INFO Loaded GBPUSD/4H split=train fold=fold_005: 9914 bars (2016-01-04 → 2022-08-05)
2026-05-09 07:41:31,474 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:41:31,542 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:41:31,564 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:41:31,565 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:41:31,574 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:41:31,575 INFO Loaded XAUUSD/4H split=train fold=fold_005: 10521 bars (2016-01-04 → 2022-08-05)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:41:31,945 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 450, 'BIAS_DOWN': 240, 'BIAS_NEUTRAL': 9831}  ambiguous=6000 (total=10521) horizon=12
2026-05-09 07:41:31,950 INFO Regime[4H mode=htf_bias split=train fold=fold_005]: collected XAUUSD — 10471 samples (group=gold) score_means={'bias_up_score': 0.043, 'bias_down_score': 0.0229} labels={'BIAS_UP': 450, 'BIAS_DOWN': 240, 'BIAS_NEUTRAL': 9781} clean={'BIAS_UP': 450, 'BIAS_DOWN': 240, 'BIAS_NEUTRAL': 3816}
2026-05-09 07:41:32,124 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:32,157 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:32,183 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:32,183 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:32,190 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:32,191 INFO Loaded EURUSD/4H split=train fold=fold_005: 9913 bars (2016-01-04 → 2022-08-05)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:41:32,487 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 257, 'BIAS_DOWN': 298, 'BIAS_NEUTRAL': 9358}  ambiguous=5696 (total=9913) horizon=12
2026-05-09 07:41:32,491 INFO Regime[4H mode=htf_bias split=train fold=fold_005]: collected EURUSD — 9863 samples (group=dollar) score_means={'bias_up_score': 0.0261, 'bias_down_score': 0.0302} labels={'BIAS_UP': 257, 'BIAS_DOWN': 298, 'BIAS_NEUTRAL': 9308} clean={'BIAS_UP': 257, 'BIAS_DOWN': 298, 'BIAS_NEUTRAL': 3641}
2026-05-09 07:41:32,655 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:32,691 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:32,710 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:32,710 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:32,717 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:32,718 INFO Loaded USDJPY/4H split=train fold=fold_005: 9914 bars (2016-01-04 → 2022-08-05)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:41:33,025 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 364, 'BIAS_DOWN': 219, 'BIAS_NEUTRAL': 9331}  ambiguous=5765 (total=9914) horizon=12
2026-05-09 07:41:33,030 INFO Regime[4H mode=htf_bias split=train fold=fold_005]: collected USDJPY — 9864 samples (group=dollar) score_means={'bias_up_score': 0.0369, 'bias_down_score': 0.0222} labels={'BIAS_UP': 364, 'BIAS_DOWN': 219, 'BIAS_NEUTRAL': 9281} clean={'BIAS_UP': 364, 'BIAS_DOWN': 219, 'BIAS_NEUTRAL': 3543}
2026-05-09 07:41:33,177 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:33,211 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:33,230 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:33,231 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:33,237 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:33,238 INFO Loaded EURJPY/4H split=train fold=fold_005: 9913 bars (2016-01-04 → 2022-08-05)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:41:33,544 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 298, 'BIAS_DOWN': 198, 'BIAS_NEUTRAL': 9417}  ambiguous=5736 (total=9913) horizon=12
2026-05-09 07:41:33,549 INFO Regime[4H mode=htf_bias split=train fold=fold_005]: collected EURJPY — 9863 samples (group=cross) score_means={'bias_up_score': 0.0302, 'bias_down_score': 0.0201} labels={'BIAS_UP': 298, 'BIAS_DOWN': 198, 'BIAS_NEUTRAL': 9367} clean={'BIAS_UP': 298, 'BIAS_DOWN': 198, 'BIAS_NEUTRAL': 3657}
2026-05-09 07:41:33,694 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:33,729 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:33,749 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:33,749 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:33,756 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:33,757 INFO Loaded GBPJPY/4H split=train fold=fold_005: 9914 bars (2016-01-04 → 2022-08-05)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:41:34,060 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 429, 'BIAS_DOWN': 260, 'BIAS_NEUTRAL': 9225}  ambiguous=5757 (total=9914) horizon=12
2026-05-09 07:41:34,065 INFO Regime[4H mode=htf_bias split=train fold=fold_005]: collected GBPJPY — 9864 samples (group=cross) score_means={'bias_up_score': 0.0435, 'bias_down_score': 0.0264} labels={'BIAS_UP': 429, 'BIAS_DOWN': 260, 'BIAS_NEUTRAL': 9175} clean={'BIAS_UP': 429, 'BIAS_DOWN': 260, 'BIAS_NEUTRAL': 3452}
2026-05-09 07:41:34,209 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:34,241 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:34,258 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:34,259 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:34,266 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:34,267 INFO Loaded GBPUSD/4H split=train fold=fold_005: 9914 bars (2016-01-04 → 2022-08-05)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:41:34,565 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 243, 'BIAS_DOWN': 322, 'BIAS_NEUTRAL': 9349}  ambiguous=5751 (total=9914) horizon=12
2026-05-09 07:41:34,569 INFO Regime[4H mode=htf_bias split=train fold=fold_005]: collected GBPUSD — 9864 samples (group=dollar) score_means={'bias_up_score': 0.0246, 'bias_down_score': 0.0326} labels={'BIAS_UP': 243, 'BIAS_DOWN': 322, 'BIAS_NEUTRAL': 9299} clean={'BIAS_UP': 243, 'BIAS_DOWN': 322, 'BIAS_NEUTRAL': 3587}
2026-05-09 07:41:34,627 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 727, 'BIAS_DOWN': 458, 'BIAS_NEUTRAL': 18542}, 'dollar': {'BIAS_UP': 864, 'BIAS_DOWN': 839, 'BIAS_NEUTRAL': 27888}, 'gold': {'BIAS_UP': 450, 'BIAS_DOWN': 240, 'BIAS_NEUTRAL': 9781}}
2026-05-09 07:41:34,627 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0369, 'bias_down_score': 0.0232}, 'dollar': {'bias_up_score': 0.0292, 'bias_down_score': 0.0284}, 'gold': {'bias_up_score': 0.043, 'bias_down_score': 0.0229}}
2026-05-09 07:41:34,628 INFO Regime[4H mode=htf_bias] label distribution by year: {2016: {'BIAS_UP': 286, 'BIAS_DOWN': 342, 'BIAS_NEUTRAL': 8194}, 2017: {'BIAS_UP': 461, 'BIAS_DOWN': 205, 'BIAS_NEUTRAL': 8447}, 2018: {'BIAS_UP': 214, 'BIAS_DOWN': 288, 'BIAS_NEUTRAL': 8628}, 2019: {'BIAS_UP': 210, 'BIAS_DOWN': 192, 'BIAS_NEUTRAL': 8700}, 2020: {'BIAS_UP': 298, 'BIAS_DOWN': 181, 'BIAS_NEUTRAL': 8632}, 2021: {'BIAS_UP': 295, 'BIAS_DOWN': 186, 'BIAS_NEUTRAL': 8610}, 2022: {'BIAS_UP': 277, 'BIAS_DOWN': 143, 'BIAS_NEUTRAL': 5000}}
2026-05-09 07:41:34,628 INFO Regime[4H mode=htf_bias] score means by year: {2016: {'bias_up_score': 0.0324, 'bias_down_score': 0.0388}, 2017: {'bias_up_score': 0.0506, 'bias_down_score': 0.0225}, 2018: {'bias_up_score': 0.0234, 'bias_down_score': 0.0315}, 2019: {'bias_up_score': 0.0231, 'bias_down_score': 0.0211}, 2020: {'bias_up_score': 0.0327, 'bias_down_score': 0.0199}, 2021: {'bias_up_score': 0.0324, 'bias_down_score': 0.0205}, 2022: {'bias_up_score': 0.0511, 'bias_down_score': 0.0264}}
2026-05-09 07:41:34,671 INFO Loaded XAUUSD/4H split=val fold=fold_005: 1581 bars (2022-08-05 → 2023-08-04)
2026-05-09 07:41:34,672 INFO Loaded EURUSD/4H split=val fold=fold_005: 1491 bars (2022-08-05 → 2023-08-04)
2026-05-09 07:41:34,673 INFO Loaded USDJPY/4H split=val fold=fold_005: 1489 bars (2022-08-05 → 2023-08-04)
2026-05-09 07:41:34,674 INFO Loaded EURJPY/4H split=val fold=fold_005: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-09 07:41:34,674 INFO Loaded GBPJPY/4H split=val fold=fold_005: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-09 07:41:34,675 INFO Loaded GBPUSD/4H split=val fold=fold_005: 1488 bars (2022-08-05 → 2023-08-04)
2026-05-09 07:41:34,692 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:41:34,695 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:41:34,696 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:41:34,697 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:41:34,697 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:41:34,698 INFO Loaded XAUUSD/4H split=val fold=fold_005: 1581 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:41:34,900 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 39, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1512}  ambiguous=936 (total=1581) horizon=12
2026-05-09 07:41:34,903 INFO Regime[4H mode=htf_bias split=val fold=fold_005]: collected XAUUSD — 1531 samples (group=gold) score_means={'bias_up_score': 0.0255, 'bias_down_score': 0.0196} labels={'BIAS_UP': 39, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1462} clean={'BIAS_UP': 39, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 555}
2026-05-09 07:41:34,970 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:34,973 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:34,974 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:34,974 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:34,974 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:34,976 INFO Loaded EURUSD/4H split=val fold=fold_005: 1491 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:41:35,154 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 35, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1403}  ambiguous=861 (total=1491) horizon=12
2026-05-09 07:41:35,157 INFO Regime[4H mode=htf_bias split=val fold=fold_005]: collected EURUSD — 1441 samples (group=dollar) score_means={'bias_up_score': 0.0243, 'bias_down_score': 0.0368} labels={'BIAS_UP': 35, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1353} clean={'BIAS_UP': 35, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 532}
2026-05-09 07:41:35,219 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:35,221 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:35,222 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:35,222 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:35,223 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:35,224 INFO Loaded USDJPY/4H split=val fold=fold_005: 1489 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:41:35,402 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 102, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 1362}  ambiguous=886 (total=1489) horizon=12
2026-05-09 07:41:35,405 INFO Regime[4H mode=htf_bias split=val fold=fold_005]: collected USDJPY — 1439 samples (group=dollar) score_means={'bias_up_score': 0.0709, 'bias_down_score': 0.0174} labels={'BIAS_UP': 102, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 1312} clean={'BIAS_UP': 102, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 454}
2026-05-09 07:41:35,466 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:35,469 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:35,470 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:35,470 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:35,470 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:35,471 INFO Loaded EURJPY/4H split=val fold=fold_005: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:41:35,650 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 46, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1447}  ambiguous=915 (total=1494) horizon=12
2026-05-09 07:41:35,653 INFO Regime[4H mode=htf_bias split=val fold=fold_005]: collected EURJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0319, 'bias_down_score': 0.0007} labels={'BIAS_UP': 46, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1397} clean={'BIAS_UP': 46, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 509}
2026-05-09 07:41:35,715 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:35,717 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:35,718 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:35,719 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:35,719 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:35,720 INFO Loaded GBPJPY/4H split=val fold=fold_005: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:41:35,908 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 29, 'BIAS_DOWN': 21, 'BIAS_NEUTRAL': 1444}  ambiguous=861 (total=1494) horizon=12
2026-05-09 07:41:35,911 INFO Regime[4H mode=htf_bias split=val fold=fold_005]: collected GBPJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0201, 'bias_down_score': 0.0145} labels={'BIAS_UP': 29, 'BIAS_DOWN': 21, 'BIAS_NEUTRAL': 1394} clean={'BIAS_UP': 29, 'BIAS_DOWN': 21, 'BIAS_NEUTRAL': 561}
2026-05-09 07:41:35,974 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:35,977 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:35,978 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:35,978 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:35,979 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:41:35,980 INFO Loaded GBPUSD/4H split=val fold=fold_005: 1488 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:41:36,169 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 26, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 1440}  ambiguous=885 (total=1488) horizon=12
2026-05-09 07:41:36,171 INFO Regime[4H mode=htf_bias split=val fold=fold_005]: collected GBPUSD — 1438 samples (group=dollar) score_means={'bias_up_score': 0.0181, 'bias_down_score': 0.0153} labels={'BIAS_UP': 26, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 1390} clean={'BIAS_UP': 26, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 543}
2026-05-09 07:41:36,230 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 75, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 2791}, 'dollar': {'BIAS_UP': 163, 'BIAS_DOWN': 100, 'BIAS_NEUTRAL': 4055}, 'gold': {'BIAS_UP': 39, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1462}}
2026-05-09 07:41:36,230 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.026, 'bias_down_score': 0.0076}, 'dollar': {'bias_up_score': 0.0377, 'bias_down_score': 0.0232}, 'gold': {'bias_up_score': 0.0255, 'bias_down_score': 0.0196}}
2026-05-09 07:41:36,230 INFO Regime[4H mode=htf_bias] label distribution by year: {2022: {'BIAS_UP': 91, 'BIAS_DOWN': 81, 'BIAS_NEUTRAL': 3229}, 2023: {'BIAS_UP': 186, 'BIAS_DOWN': 71, 'BIAS_NEUTRAL': 5079}}
2026-05-09 07:41:36,230 INFO Regime[4H mode=htf_bias] score means by year: {2022: {'bias_up_score': 0.0268, 'bias_down_score': 0.0238}, 2023: {'bias_up_score': 0.0349, 'bias_down_score': 0.0133}}
2026-05-09 07:41:36,272 INFO Regime phase HTF dataset build fold=fold_005: 5.1s (train=59789 val=8737)
2026-05-09 07:41:36,272 INFO Regime 4H/htf_bias cold start: no existing weights found
2026-05-09 07:41:36,277 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=59789 val=8737 train_labels={'BIAS_UP': 2041, 'BIAS_DOWN': 1537, 'BIAS_NEUTRAL': 56211} val_labels={'BIAS_UP': 277, 'BIAS_DOWN': 152, 'BIAS_NEUTRAL': 8308}
2026-05-09 07:41:36,464 INFO RegimeClassifier[mode=htf_bias]: cold start HTF score head
2026-05-09 07:41:36,464 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-09 07:41:36,465 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 20.0, 'bias_down_score': 20.0}
2026-05-09 07:41:39,023 INFO Regime HTF score epoch  1/50 — tr=1.7840 va=1.0369 acc=0.951 bal=0.333 threshold=0.35 margin=0.30 recall={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 1.0} precision={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 0.951}
2026-05-09 07:41:39,614 INFO Regime HTF score epoch  2/50 — tr=1.7106 va=1.0064 bal=0.333
2026-05-09 07:41:40,198 INFO Regime HTF score epoch  3/50 — tr=1.5370 va=0.9474 bal=0.608
2026-05-09 07:41:40,795 INFO Regime HTF score epoch  4/50 — tr=1.3029 va=0.8877 bal=0.653
2026-05-09 07:41:41,380 INFO Regime HTF score epoch  5/50 — tr=1.0673 va=0.8769 acc=0.845 bal=0.731 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.801, 'BIAS_DOWN': 0.539, 'BIAS_NEUTRAL': 0.852} precision={'BIAS_UP': 0.197, 'BIAS_DOWN': 0.201, 'BIAS_NEUTRAL': 0.983}
2026-05-09 07:41:41,978 INFO Regime HTF score epoch  6/50 — tr=0.9131 va=0.8818 bal=0.823
2026-05-09 07:41:42,593 INFO Regime HTF score epoch  7/50 — tr=0.8247 va=0.8554 bal=0.846
2026-05-09 07:41:43,181 INFO Regime HTF score epoch  8/50 — tr=0.7691 va=0.8106 bal=0.854
2026-05-09 07:41:43,764 INFO Regime HTF score epoch  9/50 — tr=0.7155 va=0.7761 bal=0.861
2026-05-09 07:41:44,356 INFO Regime HTF score epoch 10/50 — tr=0.6743 va=0.7390 acc=0.797 bal=0.873 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.906, 'BIAS_DOWN': 0.921, 'BIAS_NEUTRAL': 0.791} precision={'BIAS_UP': 0.187, 'BIAS_DOWN': 0.179, 'BIAS_NEUTRAL': 0.994}
2026-05-09 07:41:44,946 INFO Regime HTF score epoch 11/50 — tr=0.6387 va=0.7067 bal=0.880
2026-05-09 07:41:45,538 INFO Regime HTF score epoch 12/50 — tr=0.6116 va=0.6807 bal=0.881
2026-05-09 07:41:46,134 INFO Regime HTF score epoch 13/50 — tr=0.5817 va=0.6577 bal=0.887
2026-05-09 07:41:46,733 INFO Regime HTF score epoch 14/50 — tr=0.5592 va=0.6372 bal=0.894
2026-05-09 07:41:47,323 INFO Regime HTF score epoch 15/50 — tr=0.5415 va=0.6203 acc=0.815 bal=0.892 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.921, 'BIAS_DOWN': 0.947, 'BIAS_NEUTRAL': 0.809} precision={'BIAS_UP': 0.204, 'BIAS_DOWN': 0.196, 'BIAS_NEUTRAL': 0.996}
2026-05-09 07:41:47,919 INFO Regime HTF score epoch 16/50 — tr=0.5260 va=0.6033 bal=0.897
2026-05-09 07:41:48,506 INFO Regime HTF score epoch 17/50 — tr=0.5038 va=0.5850 bal=0.893
2026-05-09 07:41:49,106 INFO Regime HTF score epoch 18/50 — tr=0.4901 va=0.5770 bal=0.901
2026-05-09 07:41:49,708 INFO Regime HTF score epoch 19/50 — tr=0.4752 va=0.5549 bal=0.903
2026-05-09 07:41:50,302 INFO Regime HTF score epoch 20/50 — tr=0.4652 va=0.5515 acc=0.827 bal=0.909 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.946, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 0.82} precision={'BIAS_UP': 0.218, 'BIAS_DOWN': 0.208, 'BIAS_NEUTRAL': 0.997}
2026-05-09 07:41:50,902 INFO Regime HTF score epoch 21/50 — tr=0.4517 va=0.5435 bal=0.908
2026-05-09 07:41:51,496 INFO Regime HTF score epoch 22/50 — tr=0.4450 va=0.5343 bal=0.905
2026-05-09 07:41:52,120 INFO Regime HTF score epoch 23/50 — tr=0.4332 va=0.5203 bal=0.904
2026-05-09 07:41:52,723 INFO Regime HTF score epoch 24/50 — tr=0.4268 va=0.5131 bal=0.905
2026-05-09 07:41:53,316 INFO Regime HTF score epoch 25/50 — tr=0.4207 va=0.5056 acc=0.835 bal=0.904 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.935, 'BIAS_DOWN': 0.947, 'BIAS_NEUTRAL': 0.83} precision={'BIAS_UP': 0.228, 'BIAS_DOWN': 0.211, 'BIAS_NEUTRAL': 0.996}
2026-05-09 07:41:53,906 INFO Regime HTF score epoch 26/50 — tr=0.4159 va=0.5034 bal=0.910
2026-05-09 07:41:54,529 INFO Regime HTF score epoch 27/50 — tr=0.4075 va=0.4909 bal=0.909
2026-05-09 07:41:55,124 INFO Regime HTF score epoch 28/50 — tr=0.4025 va=0.4828 bal=0.908
2026-05-09 07:41:55,711 INFO Regime HTF score epoch 29/50 — tr=0.3993 va=0.4881 bal=0.907
2026-05-09 07:41:56,320 INFO Regime HTF score epoch 30/50 — tr=0.3938 va=0.4836 acc=0.839 bal=0.909 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.946, 'BIAS_DOWN': 0.947, 'BIAS_NEUTRAL': 0.833} precision={'BIAS_UP': 0.233, 'BIAS_DOWN': 0.215, 'BIAS_NEUTRAL': 0.997}
2026-05-09 07:41:56,927 INFO Regime HTF score epoch 31/50 — tr=0.3908 va=0.4808 bal=0.909
2026-05-09 07:41:57,526 INFO Regime HTF score epoch 32/50 — tr=0.3876 va=0.4723 bal=0.906
2026-05-09 07:41:58,128 INFO Regime HTF score epoch 33/50 — tr=0.3856 va=0.4719 bal=0.906
2026-05-09 07:41:58,731 INFO Regime HTF score epoch 34/50 — tr=0.3788 va=0.4742 bal=0.907
2026-05-09 07:41:58,731 INFO Regime HTF score early stop at epoch 34
2026-05-09 07:41:59,220 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.850 margin=0.000 precision={'BIAS_UP': 0.227, 'BIAS_DOWN': 0.214, 'BIAS_NEUTRAL': 0.997} recall={'BIAS_UP': 0.949, 'BIAS_DOWN': 0.954, 'BIAS_NEUTRAL': 0.828} f1={'BIAS_UP': 0.366, 'BIAS_DOWN': 0.35, 'BIAS_NEUTRAL': 0.905} confusion=[[263, 0, 14], [0, 145, 7], [898, 531, 6879]] score_mae={'bias_up_score': 0.239, 'bias_down_score': 0.1675} pred_share={'BIAS_UP': 0.1329, 'BIAS_DOWN': 0.0774, 'BIAS_NEUTRAL': 0.7897}
2026-05-09 07:41:59,221 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.227, 'BIAS_DOWN': 0.214, 'BIAS_NEUTRAL': 0.997} min_precision=0.300 recall={'BIAS_UP': 0.949, 'BIAS_DOWN': 0.954, 'BIAS_NEUTRAL': 0.828} min_recall=0.100 f1={'BIAS_UP': 0.366, 'BIAS_DOWN': 0.35, 'BIAS_NEUTRAL': 0.905} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_UP', 'BIAS_DOWN'] weak_recall=[] weak_f1=[] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-09 07:41:59,225 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-09 07:41:59,225 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-09 07:41:59,226 INFO Regime phase HTF train fold=fold_005: 23.0s
2026-05-09 07:41:59,328 INFO Regime HTF complete fold=fold_005: acc=0.834 bal=0.910 train=59789 val=8737 per_class={'BIAS_UP': 0.949, 'BIAS_DOWN': 0.954, 'BIAS_NEUTRAL': 0.828} precision={'BIAS_UP': 0.227, 'BIAS_DOWN': 0.214, 'BIAS_NEUTRAL': 0.997} threshold=0.850 margin=0.000
2026-05-09 07:41:59,329 INFO Loaded GBPUSD/4H split=train fold=fold_005: 9914 bars (2016-01-04 → 2022-08-05)
2026-05-09 07:41:59,487 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 243, 'BIAS_DOWN': 322, 'BIAS_NEUTRAL': 9349}  ambiguous=5751 (total=9914) horizon=12
2026-05-09 07:41:59,489 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on GBPUSD 4H:
{'BIAS_UP': 3.1153846153846154, 'BIAS_DOWN': 4.075949367088608, 'BIAS_NEUTRAL': 59.17088607594937}
2026-05-09 07:41:59,493 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (all labels):
{'BIAS_UP': {'n': 243, 'mean': 0.0009196585951238863, 'mean_over_std': 0.4475780372764997}, 'BIAS_DOWN': {'n': 322, 'mean': -0.0012795894690640242, 'mean_over_std': -0.44099378219368485}, 'BIAS_NEUTRAL': {'n': 9348, 'mean': 2.580192436230557e-06, 'mean_over_std': 0.0010304408742778319}}
2026-05-09 07:41:59,494 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 243, 'mean': 0.0009196585951238863, 'mean_over_std': 0.4475780372764997}, 'BIAS_DOWN': {'n': 322, 'mean': -0.0012795894690640242, 'mean_over_std': -0.44099378219368485}, 'BIAS_NEUTRAL': {'n': 3598, 'mean': 1.415804339687073e-05, 'mean_over_std': 0.006961900611690859}}
2026-05-09 07:41:59,498 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-09 07:41:59,501 INFO Loaded XAUUSD/1H split=train fold=fold_005: 39110 bars (2016-01-04 → 2022-08-05)
2026-05-09 07:41:59,503 INFO Loaded EURUSD/1H split=train fold=fold_005: 38636 bars (2016-01-04 → 2022-08-05)
2026-05-09 07:41:59,504 INFO Loaded USDJPY/1H split=train fold=fold_005: 38640 bars (2016-01-04 → 2022-08-05)
2026-05-09 07:41:59,506 INFO Loaded EURJPY/1H split=train fold=fold_005: 38635 bars (2016-01-04 → 2022-08-05)
2026-05-09 07:41:59,508 INFO Loaded GBPJPY/1H split=train fold=fold_005: 38635 bars (2016-01-04 → 2022-08-05)
2026-05-09 07:41:59,510 INFO Loaded GBPUSD/1H split=train fold=fold_005: 38636 bars (2016-01-04 → 2022-08-05)
2026-05-09 07:41:59,527 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:41:59,536 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:41:59,539 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:41:59,540 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:41:59,541 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:41:59,547 INFO Loaded XAUUSD/1H split=train fold=fold_005: 39110 bars (2016-01-04 → 2022-08-05)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:42:00,373 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_005]: collected XAUUSD — 39060 samples (group=gold) score_means={'trend_score': 0.4874, 'range_score': 0.2376, 'chop_score': 0.4677, 'volatility_percentile': 0.3777, 'consolidation_score': 0.1917}
2026-05-09 07:42:00,480 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:00,482 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:00,483 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:00,483 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:00,484 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:00,486 INFO Loaded EURUSD/1H split=train fold=fold_005: 38636 bars (2016-01-04 → 2022-08-05)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:42:01,233 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_005]: collected EURUSD — 38586 samples (group=dollar) score_means={'trend_score': 0.4904, 'range_score': 0.2354, 'chop_score': 0.462, 'volatility_percentile': 0.382, 'consolidation_score': 0.1857}
2026-05-09 07:42:01,334 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:01,336 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:01,337 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:01,337 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:01,338 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:01,340 INFO Loaded USDJPY/1H split=train fold=fold_005: 38640 bars (2016-01-04 → 2022-08-05)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:42:02,083 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_005]: collected USDJPY — 38590 samples (group=dollar) score_means={'trend_score': 0.4931, 'range_score': 0.232, 'chop_score': 0.4628, 'volatility_percentile': 0.3801, 'consolidation_score': 0.1927}
2026-05-09 07:42:02,189 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:02,192 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:02,192 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:02,193 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:02,193 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:02,195 INFO Loaded EURJPY/1H split=train fold=fold_005: 38635 bars (2016-01-04 → 2022-08-05)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:42:02,922 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_005]: collected EURJPY — 38585 samples (group=cross) score_means={'trend_score': 0.487, 'range_score': 0.2359, 'chop_score': 0.4681, 'volatility_percentile': 0.3843, 'consolidation_score': 0.1875}
2026-05-09 07:42:03,022 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:03,024 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:03,025 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:03,025 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:03,026 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:03,028 INFO Loaded GBPJPY/1H split=train fold=fold_005: 38635 bars (2016-01-04 → 2022-08-05)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:42:03,737 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_005]: collected GBPJPY — 38585 samples (group=cross) score_means={'trend_score': 0.4897, 'range_score': 0.2346, 'chop_score': 0.4667, 'volatility_percentile': 0.3856, 'consolidation_score': 0.1886}
2026-05-09 07:42:03,836 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:03,838 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:03,839 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:03,839 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:03,840 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:03,842 INFO Loaded GBPUSD/1H split=train fold=fold_005: 38636 bars (2016-01-04 → 2022-08-05)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:42:04,549 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_005]: collected GBPUSD — 38586 samples (group=dollar) score_means={'trend_score': 0.4935, 'range_score': 0.2331, 'chop_score': 0.4609, 'volatility_percentile': 0.3848, 'consolidation_score': 0.1848}
2026-05-09 07:42:04,650 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4884, 'range_score': 0.2353, 'chop_score': 0.4674, 'volatility_percentile': 0.385, 'consolidation_score': 0.1881}, 'dollar': {'trend_score': 0.4924, 'range_score': 0.2335, 'chop_score': 0.4619, 'volatility_percentile': 0.3823, 'consolidation_score': 0.1877}, 'gold': {'trend_score': 0.4874, 'range_score': 0.2376, 'chop_score': 0.4677, 'volatility_percentile': 0.3777, 'consolidation_score': 0.1917}}
2026-05-09 07:42:04,650 INFO Regime[1H mode=ltf_behaviour] score means by year: {2016: {'trend_score': 0.495, 'range_score': 0.233, 'chop_score': 0.4607, 'volatility_percentile': 0.3776, 'consolidation_score': 0.192}, 2017: {'trend_score': 0.5017, 'range_score': 0.2328, 'chop_score': 0.4574, 'volatility_percentile': 0.3689, 'consolidation_score': 0.1942}, 2018: {'trend_score': 0.4886, 'range_score': 0.237, 'chop_score': 0.4655, 'volatility_percentile': 0.3947, 'consolidation_score': 0.1785}, 2019: {'trend_score': 0.4817, 'range_score': 0.236, 'chop_score': 0.4691, 'volatility_percentile': 0.3725, 'consolidation_score': 0.198}, 2020: {'trend_score': 0.4874, 'range_score': 0.2311, 'chop_score': 0.4674, 'volatility_percentile': 0.3848, 'consolidation_score': 0.1919}, 2021: {'trend_score': 0.4872, 'range_score': 0.2372, 'chop_score': 0.4672, 'volatility_percentile': 0.377, 'consolidation_score': 0.192}, 2022: {'trend_score': 0.4893, 'range_score': 0.2373, 'chop_score': 0.4662, 'volatility_percentile': 0.4141, 'consolidation_score': 0.1626}}
2026-05-09 07:42:04,731 INFO Loaded XAUUSD/1H split=val fold=fold_005: 5167 bars (2022-08-05 → 2023-08-04)
2026-05-09 07:42:04,733 INFO Loaded EURUSD/1H split=val fold=fold_005: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-09 07:42:04,734 INFO Loaded USDJPY/1H split=val fold=fold_005: 5096 bars (2022-08-05 → 2023-08-04)
2026-05-09 07:42:04,735 INFO Loaded EURJPY/1H split=val fold=fold_005: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-09 07:42:04,737 INFO Loaded GBPJPY/1H split=val fold=fold_005: 5093 bars (2022-08-05 → 2023-08-04)
2026-05-09 07:42:04,738 INFO Loaded GBPUSD/1H split=val fold=fold_005: 5090 bars (2022-08-05 → 2023-08-04)
2026-05-09 07:42:04,748 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:42:04,751 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:42:04,752 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:42:04,753 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:42:04,753 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:42:04,755 INFO Loaded XAUUSD/1H split=val fold=fold_005: 5167 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:42:05,016 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_005]: collected XAUUSD — 5117 samples (group=gold) score_means={'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}
2026-05-09 07:42:05,122 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:05,126 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:05,127 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:05,127 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:05,128 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:05,129 INFO Loaded EURUSD/1H split=val fold=fold_005: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:42:05,353 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_005]: collected EURUSD — 5053 samples (group=dollar) score_means={'trend_score': 0.4883, 'range_score': 0.2413, 'chop_score': 0.463, 'volatility_percentile': 0.3837, 'consolidation_score': 0.1793}
2026-05-09 07:42:05,455 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:05,457 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:05,458 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:05,459 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:05,459 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:05,462 INFO Loaded USDJPY/1H split=val fold=fold_005: 5096 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:42:05,692 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_005]: collected USDJPY — 5046 samples (group=dollar) score_means={'trend_score': 0.5085, 'range_score': 0.2259, 'chop_score': 0.4506, 'volatility_percentile': 0.3809, 'consolidation_score': 0.1888}
2026-05-09 07:42:05,811 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:05,814 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:05,814 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:05,815 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:05,815 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:05,817 INFO Loaded EURJPY/1H split=val fold=fold_005: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:42:06,053 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_005]: collected EURJPY — 5053 samples (group=cross) score_means={'trend_score': 0.5023, 'range_score': 0.2261, 'chop_score': 0.4524, 'volatility_percentile': 0.3837, 'consolidation_score': 0.18}
2026-05-09 07:42:06,156 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:06,158 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:06,159 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:06,159 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:06,160 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:06,161 INFO Loaded GBPJPY/1H split=val fold=fold_005: 5093 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:42:06,390 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_005]: collected GBPJPY — 5043 samples (group=cross) score_means={'trend_score': 0.4743, 'range_score': 0.2372, 'chop_score': 0.4716, 'volatility_percentile': 0.3777, 'consolidation_score': 0.1917}
2026-05-09 07:42:06,493 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:06,495 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:06,496 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:06,496 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:06,497 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:06,498 INFO Loaded GBPUSD/1H split=val fold=fold_005: 5090 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:42:06,728 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_005]: collected GBPUSD — 5040 samples (group=dollar) score_means={'trend_score': 0.4843, 'range_score': 0.2371, 'chop_score': 0.4614, 'volatility_percentile': 0.377, 'consolidation_score': 0.1835}
2026-05-09 07:42:06,830 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2317, 'chop_score': 0.462, 'volatility_percentile': 0.3807, 'consolidation_score': 0.1858}, 'dollar': {'trend_score': 0.4937, 'range_score': 0.2348, 'chop_score': 0.4583, 'volatility_percentile': 0.3806, 'consolidation_score': 0.1839}, 'gold': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}}
2026-05-09 07:42:06,830 INFO Regime[1H mode=ltf_behaviour] score means by year: {2022: {'trend_score': 0.4905, 'range_score': 0.2364, 'chop_score': 0.4625, 'volatility_percentile': 0.385, 'consolidation_score': 0.1797}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-09 07:42:06,917 INFO Regime phase LTF dataset build fold=fold_005: 7.4s (train=231992 val=30352)
2026-05-09 07:42:06,917 INFO Regime 1H/ltf_behaviour cold start: no existing weights found
2026-05-09 07:42:06,940 INFO RegimeClassifier[mode=ltf_behaviour]: cold start score head
2026-05-09 07:42:06,941 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-09 07:42:07,407 INFO Regime score epoch  1/50 — tr=0.0798 va=0.0672 mae={'trend_score': 0.1561, 'range_score': 0.2432, 'chop_score': 0.1676, 'volatility_percentile': 0.1747, 'consolidation_score': 0.3359}
2026-05-09 07:42:07,828 INFO Regime score epoch  2/50 — tr=0.0680 va=0.0534
2026-05-09 07:42:08,254 INFO Regime score epoch  3/50 — tr=0.0515 va=0.0377
2026-05-09 07:42:08,680 INFO Regime score epoch  4/50 — tr=0.0361 va=0.0259
2026-05-09 07:42:09,101 INFO Regime score epoch  5/50 — tr=0.0257 va=0.0175 mae={'trend_score': 0.0692, 'range_score': 0.1192, 'chop_score': 0.0654, 'volatility_percentile': 0.0592, 'consolidation_score': 0.1971}
2026-05-09 07:42:09,519 INFO Regime score epoch  6/50 — tr=0.0190 va=0.0117
2026-05-09 07:42:09,936 INFO Regime score epoch  7/50 — tr=0.0150 va=0.0083
2026-05-09 07:42:10,344 INFO Regime score epoch  8/50 — tr=0.0125 va=0.0063
2026-05-09 07:42:10,768 INFO Regime score epoch  9/50 — tr=0.0109 va=0.0052
2026-05-09 07:42:11,187 INFO Regime score epoch 10/50 — tr=0.0099 va=0.0046 mae={'trend_score': 0.0552, 'range_score': 0.0587, 'chop_score': 0.0535, 'volatility_percentile': 0.0308, 'consolidation_score': 0.0688}
2026-05-09 07:42:11,613 INFO Regime score epoch 11/50 — tr=0.0091 va=0.0042
2026-05-09 07:42:12,045 INFO Regime score epoch 12/50 — tr=0.0085 va=0.0038
2026-05-09 07:42:12,497 INFO Regime score epoch 13/50 — tr=0.0081 va=0.0036
2026-05-09 07:42:12,922 INFO Regime score epoch 14/50 — tr=0.0078 va=0.0034
2026-05-09 07:42:13,330 INFO Regime score epoch 15/50 — tr=0.0074 va=0.0032 mae={'trend_score': 0.0469, 'range_score': 0.0533, 'chop_score': 0.0474, 'volatility_percentile': 0.0276, 'consolidation_score': 0.0456}
2026-05-09 07:42:13,753 INFO Regime score epoch 16/50 — tr=0.0072 va=0.0030
2026-05-09 07:42:14,178 INFO Regime score epoch 17/50 — tr=0.0070 va=0.0029
2026-05-09 07:42:14,605 INFO Regime score epoch 18/50 — tr=0.0068 va=0.0028
2026-05-09 07:42:15,016 INFO Regime score epoch 19/50 — tr=0.0066 va=0.0027
2026-05-09 07:42:15,442 INFO Regime score epoch 20/50 — tr=0.0064 va=0.0026 mae={'trend_score': 0.041, 'range_score': 0.0492, 'chop_score': 0.0427, 'volatility_percentile': 0.0256, 'consolidation_score': 0.0374}
2026-05-09 07:42:15,855 INFO Regime score epoch 21/50 — tr=0.0063 va=0.0025
2026-05-09 07:42:16,264 INFO Regime score epoch 22/50 — tr=0.0061 va=0.0024
2026-05-09 07:42:16,691 INFO Regime score epoch 23/50 — tr=0.0060 va=0.0023
2026-05-09 07:42:17,107 INFO Regime score epoch 24/50 — tr=0.0059 va=0.0023
2026-05-09 07:42:17,519 INFO Regime score epoch 25/50 — tr=0.0058 va=0.0022 mae={'trend_score': 0.0365, 'range_score': 0.0466, 'chop_score': 0.0384, 'volatility_percentile': 0.0237, 'consolidation_score': 0.0345}
2026-05-09 07:42:17,952 INFO Regime score epoch 26/50 — tr=0.0057 va=0.0021
2026-05-09 07:42:18,382 INFO Regime score epoch 27/50 — tr=0.0057 va=0.0021
2026-05-09 07:42:18,819 INFO Regime score epoch 28/50 — tr=0.0056 va=0.0021
2026-05-09 07:42:19,237 INFO Regime score epoch 29/50 — tr=0.0055 va=0.0020
2026-05-09 07:42:19,657 INFO Regime score epoch 30/50 — tr=0.0054 va=0.0020 mae={'trend_score': 0.0334, 'range_score': 0.0448, 'chop_score': 0.0351, 'volatility_percentile': 0.0229, 'consolidation_score': 0.0326}
2026-05-09 07:42:20,073 INFO Regime score epoch 31/50 — tr=0.0054 va=0.0019
2026-05-09 07:42:20,483 INFO Regime score epoch 32/50 — tr=0.0053 va=0.0019
2026-05-09 07:42:20,918 INFO Regime score epoch 33/50 — tr=0.0053 va=0.0019
2026-05-09 07:42:21,332 INFO Regime score epoch 34/50 — tr=0.0052 va=0.0018
2026-05-09 07:42:21,764 INFO Regime score epoch 35/50 — tr=0.0052 va=0.0018 mae={'trend_score': 0.0311, 'range_score': 0.0435, 'chop_score': 0.0332, 'volatility_percentile': 0.0216, 'consolidation_score': 0.0312}
2026-05-09 07:42:22,215 INFO Regime score epoch 36/50 — tr=0.0052 va=0.0018
2026-05-09 07:42:22,623 INFO Regime score epoch 37/50 — tr=0.0052 va=0.0018
2026-05-09 07:42:23,055 INFO Regime score epoch 38/50 — tr=0.0051 va=0.0018
2026-05-09 07:42:23,480 INFO Regime score epoch 39/50 — tr=0.0051 va=0.0018
2026-05-09 07:42:23,902 INFO Regime score epoch 40/50 — tr=0.0051 va=0.0018 mae={'trend_score': 0.03, 'range_score': 0.0433, 'chop_score': 0.0324, 'volatility_percentile': 0.0209, 'consolidation_score': 0.0307}
2026-05-09 07:42:24,335 INFO Regime score epoch 41/50 — tr=0.0051 va=0.0017
2026-05-09 07:42:24,761 INFO Regime score epoch 42/50 — tr=0.0051 va=0.0017
2026-05-09 07:42:25,190 INFO Regime score epoch 43/50 — tr=0.0051 va=0.0017
2026-05-09 07:42:25,612 INFO Regime score epoch 44/50 — tr=0.0051 va=0.0017
2026-05-09 07:42:26,037 INFO Regime score epoch 45/50 — tr=0.0050 va=0.0017 mae={'trend_score': 0.0294, 'range_score': 0.0429, 'chop_score': 0.0318, 'volatility_percentile': 0.0206, 'consolidation_score': 0.0308}
2026-05-09 07:42:26,457 INFO Regime score epoch 46/50 — tr=0.0051 va=0.0017
2026-05-09 07:42:26,887 INFO Regime score epoch 47/50 — tr=0.0050 va=0.0017
2026-05-09 07:42:27,324 INFO Regime score epoch 48/50 — tr=0.0050 va=0.0017
2026-05-09 07:42:27,754 INFO Regime score epoch 49/50 — tr=0.0050 va=0.0017
2026-05-09 07:42:28,170 INFO Regime score epoch 50/50 — tr=0.0051 va=0.0017 mae={'trend_score': 0.0295, 'range_score': 0.0429, 'chop_score': 0.0319, 'volatility_percentile': 0.0207, 'consolidation_score': 0.0305}
2026-05-09 07:42:28,190 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0295, 'range_score': 0.0429, 'chop_score': 0.0319, 'volatility_percentile': 0.0207, 'consolidation_score': 0.0305} mse={'trend_score': 0.00143, 'range_score': 0.00289, 'chop_score': 0.00164, 'volatility_percentile': 0.00081, 'consolidation_score': 0.00181} corr={'trend_score': 0.9855, 'range_score': 0.9288, 'chop_score': 0.9773, 'volatility_percentile': 0.9913, 'consolidation_score': 0.981} pred_std={'trend_score': 0.2162, 'range_score': 0.1381, 'chop_score': 0.1781, 'volatility_percentile': 0.2151, 'consolidation_score': 0.2154} target_std={'trend_score': 0.2219, 'range_score': 0.1433, 'chop_score': 0.1888, 'volatility_percentile': 0.2165, 'consolidation_score': 0.2179}
2026-05-09 07:42:28,195 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-09 07:42:28,195 INFO Regime phase LTF train fold=fold_005: 21.3s
2026-05-09 07:42:28,291 INFO Regime LTF complete fold=fold_005: score_accuracy=0.969, train=231992 val=30352 mae={'trend_score': 0.0295, 'range_score': 0.0429, 'chop_score': 0.0319, 'volatility_percentile': 0.0207, 'consolidation_score': 0.0305}
2026-05-09 07:42:28,293 INFO Loaded GBPUSD/1H split=train fold=fold_005: 38636 bars (2016-01-04 → 2022-08-05)
2026-05-09 07:42:28,601 INFO Regime[1H mode=ltf_behaviour fold=fold_005] LTF score diagnostics on GBPUSD:
{'trend_score': {'mean': 0.4934, 'q10': 0.1938, 'q50': 0.4888, 'q90': 0.8011}, 'range_score': {'mean': 0.2333, 'q10': 0.0513, 'q50': 0.2121, 'q90': 0.4341}, 'chop_score': {'mean': 0.4611, 'q10': 0.2146, 'q50': 0.4499, 'q90': 0.7255}, 'volatility_percentile': {'mean': 0.3848, 'q10': 0.1022, 'q50': 0.3725, 'q90': 0.6848}, 'consolidation_score': {'mean': 0.1845, 'q10': 0.0, 'q50': 0.1183, 'q90': 0.5021}}
2026-05-09 07:42:28,605 INFO Regime retrain total: 57.9s (330870 train+val samples)
2026-05-09 07:42:28,608 INFO Retrain complete. Total wall-clock: 57.9s
2026-05-09 07:42:29,518 INFO Model regime: SUCCESS
2026-05-09 07:42:29,518 INFO --- Training gru ---
2026-05-09 07:42:29,518 INFO Running retrain --model gru
2026-05-09 07:42:29,773 INFO retrain environment: KAGGLE
2026-05-09 07:42:31,333 INFO Device: CUDA (2 GPU(s))
2026-05-09 07:42:31,344 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-09 07:42:31,344 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-09 07:42:31,344 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-09 07:42:31,347 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-09 07:42:31,347 INFO Retrain data split: train
2026-05-09 07:42:31,348 INFO Retrain rolling fold selector: latest
2026-05-09 07:42:31,348 INFO === GRU-LSTM retrain (timeframes: ['15M']) ===
2026-05-09 07:42:31,496 INFO NumExpr defaulting to 4 threads.
2026-05-09 07:42:31,681 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-09 07:42:31,681 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-09 07:42:31,681 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-09 07:42:31,682 INFO GRU multi-symbol training (Kaggle mode): 6 symbols × ['15M']
2026-05-09 07:42:31,682 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260509_074231
2026-05-09 07:42:31,685 WARNING WeightsManifest: no manifest at /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json — assuming compatible (run a full retrain to generate one)
2026-05-09 07:42:31,685 INFO GRU cold start: no compatible existing weights found
2026-05-09 07:42:31,959 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:42:31,987 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:42:32,002 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:42:32,011 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:42:32,097 INFO Split boundaries loaded fold=fold_005/6 — train 2016-01-04→2022-08-05  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-09 07:42:32,105 INFO Loaded XAUUSD/15M split=train fold=latest: 155779 bars (2016-01-04 → 2022-08-05)
2026-05-09 07:42:32,399 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:32,417 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:32,430 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:32,436 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:32,477 INFO Loaded EURUSD/15M split=train fold=latest: 154539 bars (2016-01-04 → 2022-08-05)
2026-05-09 07:42:32,752 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:32,783 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:32,796 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:32,803 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:32,844 INFO Loaded USDJPY/15M split=train fold=latest: 154560 bars (2016-01-04 → 2022-08-05)
2026-05-09 07:42:33,118 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:33,136 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:33,150 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:33,156 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:33,192 INFO Loaded EURJPY/15M split=train fold=latest: 154536 bars (2016-01-04 → 2022-08-05)
2026-05-09 07:42:33,460 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:33,479 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:33,492 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:33,499 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:33,535 INFO Loaded GBPJPY/15M split=train fold=latest: 154532 bars (2016-01-04 → 2022-08-05)
2026-05-09 07:42:33,795 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:33,813 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:33,826 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:33,833 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:42:33,869 INFO Loaded GBPUSD/15M split=train fold=latest: 154539 bars (2016-01-04 → 2022-08-05)
2026-05-09 07:42:34,035 INFO train_multi: 6 segments, ~901665 total bars
2026-05-09 07:42:34,249 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-05-09 07:42:34,249 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-05-09 07:42:34,250 INFO train_multi: training ALL 6 segments across TFs ['15M'] in one combined pass
2026-05-09 07:42:34,250 INFO train_multi: building combined dataset for TF=ALL (6 segments)
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
2026-05-09 07:42:42,370 INFO train_multi TF=ALL: 901485 sequences across 6 segments
2026-05-09 07:42:42,370 INFO train_multi TF=ALL: estimated peak RAM = 10224 MB (train=479997 val=120001 n_feat=71 seq_len=30)
2026-05-09 07:42:43,609 INFO train_multi TF=ALL: train=479997 val=120001 (5119 MB tensors)
2026-05-09 07:42:47,631 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=18, min_epochs=22)
2026-05-09 07:43:03,142 INFO train_multi TF=ALL epoch 1/50 train=0.8177 val=0.8072 dir_acc=0.501 dir_n=120001
2026-05-09 07:43:03,147 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:43:03,148 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:43:03,148 INFO train_multi TF=ALL: new best val=0.8072 — saved
2026-05-09 07:43:16,302 INFO train_multi TF=ALL epoch 2/50 train=0.7934 val=0.7597 dir_acc=0.501 dir_n=120001
2026-05-09 07:43:16,307 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:43:16,307 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:43:16,307 INFO train_multi TF=ALL: new best val=0.7597 — saved
2026-05-09 07:43:29,398 INFO train_multi TF=ALL epoch 3/50 train=0.7261 val=0.7082 dir_acc=0.499 dir_n=120001
2026-05-09 07:43:29,403 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:43:29,403 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:43:29,403 INFO train_multi TF=ALL: new best val=0.7082 — saved
2026-05-09 07:43:42,611 INFO train_multi TF=ALL epoch 4/50 train=0.7137 val=0.7078 dir_acc=0.501 dir_n=120001
2026-05-09 07:43:42,616 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:43:42,616 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:43:42,616 INFO train_multi TF=ALL: new best val=0.7078 — saved
2026-05-09 07:43:55,728 INFO train_multi TF=ALL epoch 5/50 train=0.7126 val=0.7076 dir_acc=0.500 dir_n=120001
2026-05-09 07:43:55,733 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:43:55,733 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:43:55,733 INFO train_multi TF=ALL: new best val=0.7076 — saved
2026-05-09 07:44:08,744 INFO train_multi TF=ALL epoch 6/50 train=0.7117 val=0.7071 dir_acc=0.501 dir_n=120001
2026-05-09 07:44:08,749 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:44:08,749 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:44:08,749 INFO train_multi TF=ALL: new best val=0.7071 — saved
2026-05-09 07:44:21,728 INFO train_multi TF=ALL epoch 7/50 train=0.7107 val=0.7068 dir_acc=0.501 dir_n=120001
2026-05-09 07:44:21,733 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:44:21,733 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:44:21,733 INFO train_multi TF=ALL: new best val=0.7068 — saved
2026-05-09 07:44:34,857 INFO train_multi TF=ALL epoch 8/50 train=0.7102 val=0.7066 dir_acc=0.503 dir_n=120001
2026-05-09 07:44:34,861 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:44:34,862 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:44:34,862 INFO train_multi TF=ALL: new best val=0.7066 — saved
2026-05-09 07:44:47,758 INFO train_multi TF=ALL epoch 9/50 train=0.7098 val=0.7065 dir_acc=0.504 dir_n=120001
2026-05-09 07:44:47,763 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:44:47,763 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:44:47,763 INFO train_multi TF=ALL: new best val=0.7065 — saved
2026-05-09 07:45:00,890 INFO train_multi TF=ALL epoch 10/50 train=0.7095 val=0.7064 dir_acc=0.501 dir_n=120001
2026-05-09 07:45:00,895 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:45:00,895 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:45:00,895 INFO train_multi TF=ALL: new best val=0.7064 — saved
2026-05-09 07:45:13,948 INFO train_multi TF=ALL epoch 11/50 train=0.7093 val=0.7063 dir_acc=0.501 dir_n=120001
2026-05-09 07:45:13,953 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:45:13,954 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:45:13,954 INFO train_multi TF=ALL: new best val=0.7063 — saved
2026-05-09 07:45:26,937 INFO train_multi TF=ALL epoch 12/50 train=0.7090 val=0.7063 dir_acc=0.503 dir_n=120001
2026-05-09 07:45:26,942 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:45:26,942 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:45:26,942 INFO train_multi TF=ALL: new best val=0.7063 — saved
2026-05-09 07:45:40,008 INFO train_multi TF=ALL epoch 13/50 train=0.7088 val=0.7061 dir_acc=0.505 dir_n=120001
2026-05-09 07:45:40,013 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:45:40,013 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:45:40,013 INFO train_multi TF=ALL: new best val=0.7061 — saved
2026-05-09 07:45:53,151 INFO train_multi TF=ALL epoch 14/50 train=0.7085 val=0.7060 dir_acc=0.508 dir_n=120001
2026-05-09 07:45:53,156 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:45:53,156 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:45:53,156 INFO train_multi TF=ALL: new best val=0.7060 — saved
2026-05-09 07:46:06,356 INFO train_multi TF=ALL epoch 15/50 train=0.7081 val=0.7057 dir_acc=0.510 dir_n=120001
2026-05-09 07:46:06,361 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:46:06,361 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:46:06,361 INFO train_multi TF=ALL: new best val=0.7057 — saved
2026-05-09 07:46:19,364 INFO train_multi TF=ALL epoch 16/50 train=0.7072 val=0.7053 dir_acc=0.514 dir_n=120001
2026-05-09 07:46:19,369 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:46:19,369 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:46:19,369 INFO train_multi TF=ALL: new best val=0.7053 — saved
2026-05-09 07:46:32,559 INFO train_multi TF=ALL epoch 17/50 train=0.7069 val=0.7049 dir_acc=0.516 dir_n=120001
2026-05-09 07:46:32,564 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:46:32,564 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:46:32,564 INFO train_multi TF=ALL: new best val=0.7049 — saved
2026-05-09 07:46:45,653 INFO train_multi TF=ALL epoch 18/50 train=0.7064 val=0.7046 dir_acc=0.519 dir_n=120001
2026-05-09 07:46:45,657 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:46:45,658 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:46:45,658 INFO train_multi TF=ALL: new best val=0.7046 — saved
2026-05-09 07:46:58,648 INFO train_multi TF=ALL epoch 19/50 train=0.7059 val=0.7040 dir_acc=0.521 dir_n=120001
2026-05-09 07:46:58,653 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:46:58,653 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:46:58,653 INFO train_multi TF=ALL: new best val=0.7040 — saved
2026-05-09 07:47:11,588 INFO train_multi TF=ALL epoch 20/50 train=0.7056 val=0.7036 dir_acc=0.525 dir_n=120001
2026-05-09 07:47:11,593 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:47:11,593 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:47:11,593 INFO train_multi TF=ALL: new best val=0.7036 — saved
2026-05-09 07:47:24,742 INFO train_multi TF=ALL epoch 21/50 train=0.7052 val=0.7032 dir_acc=0.528 dir_n=120001
2026-05-09 07:47:24,747 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:47:24,748 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:47:24,748 INFO train_multi TF=ALL: new best val=0.7032 — saved
2026-05-09 07:47:37,897 INFO train_multi TF=ALL epoch 22/50 train=0.7047 val=0.7031 dir_acc=0.527 dir_n=120001
2026-05-09 07:47:37,902 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:47:37,902 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:47:37,902 INFO train_multi TF=ALL: new best val=0.7031 — saved
2026-05-09 07:47:50,927 INFO train_multi TF=ALL epoch 23/50 train=0.7038 val=0.7014 dir_acc=0.536 dir_n=120001
2026-05-09 07:47:50,933 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:47:50,933 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:47:50,933 INFO train_multi TF=ALL: new best val=0.7014 — saved
2026-05-09 07:48:04,116 INFO train_multi TF=ALL epoch 24/50 train=0.7020 val=0.6963 dir_acc=0.556 dir_n=120001
2026-05-09 07:48:04,121 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:48:04,121 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:48:04,121 INFO train_multi TF=ALL: new best val=0.6963 — saved
2026-05-09 07:48:17,153 INFO train_multi TF=ALL epoch 25/50 train=0.6955 val=0.6851 dir_acc=0.585 dir_n=120001
2026-05-09 07:48:17,157 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:48:17,158 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:48:17,158 INFO train_multi TF=ALL: new best val=0.6851 — saved
2026-05-09 07:48:30,224 INFO train_multi TF=ALL epoch 26/50 train=0.6876 val=0.6785 dir_acc=0.602 dir_n=120001
2026-05-09 07:48:30,229 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:48:30,229 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:48:30,229 INFO train_multi TF=ALL: new best val=0.6785 — saved
2026-05-09 07:48:43,324 INFO train_multi TF=ALL epoch 27/50 train=0.6826 val=0.6770 dir_acc=0.602 dir_n=120001
2026-05-09 07:48:43,329 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:48:43,329 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:48:43,329 INFO train_multi TF=ALL: new best val=0.6770 — saved
2026-05-09 07:48:56,563 INFO train_multi TF=ALL epoch 28/50 train=0.6794 val=0.6727 dir_acc=0.610 dir_n=120001
2026-05-09 07:48:56,568 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:48:56,568 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:48:56,568 INFO train_multi TF=ALL: new best val=0.6727 — saved
2026-05-09 07:49:09,694 INFO train_multi TF=ALL epoch 29/50 train=0.6760 val=0.6694 dir_acc=0.616 dir_n=120001
2026-05-09 07:49:09,699 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:49:09,699 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:49:09,699 INFO train_multi TF=ALL: new best val=0.6694 — saved
2026-05-09 07:49:22,715 INFO train_multi TF=ALL epoch 30/50 train=0.6739 val=0.6672 dir_acc=0.619 dir_n=120001
2026-05-09 07:49:22,720 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:49:22,720 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:49:22,720 INFO train_multi TF=ALL: new best val=0.6672 — saved
2026-05-09 07:49:35,725 INFO train_multi TF=ALL epoch 31/50 train=0.6719 val=0.6657 dir_acc=0.624 dir_n=120001
2026-05-09 07:49:35,730 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:49:35,730 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:49:35,730 INFO train_multi TF=ALL: new best val=0.6657 — saved
2026-05-09 07:49:48,754 INFO train_multi TF=ALL epoch 32/50 train=0.6705 val=0.6667 dir_acc=0.620 dir_n=120001
2026-05-09 07:50:01,749 INFO train_multi TF=ALL epoch 33/50 train=0.6694 val=0.6655 dir_acc=0.623 dir_n=120001
2026-05-09 07:50:01,754 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:50:01,754 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:50:01,754 INFO train_multi TF=ALL: new best val=0.6655 — saved
2026-05-09 07:50:14,798 INFO train_multi TF=ALL epoch 34/50 train=0.6686 val=0.6635 dir_acc=0.626 dir_n=120001
2026-05-09 07:50:14,803 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:50:14,803 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:50:14,803 INFO train_multi TF=ALL: new best val=0.6635 — saved
2026-05-09 07:50:27,903 INFO train_multi TF=ALL epoch 35/50 train=0.6676 val=0.6627 dir_acc=0.628 dir_n=120001
2026-05-09 07:50:27,908 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:50:27,908 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:50:27,908 INFO train_multi TF=ALL: new best val=0.6627 — saved
2026-05-09 07:50:41,107 INFO train_multi TF=ALL epoch 36/50 train=0.6672 val=0.6632 dir_acc=0.626 dir_n=120001
2026-05-09 07:50:54,184 INFO train_multi TF=ALL epoch 37/50 train=0.6663 val=0.6638 dir_acc=0.626 dir_n=120001
2026-05-09 07:51:07,332 INFO train_multi TF=ALL epoch 38/50 train=0.6659 val=0.6633 dir_acc=0.626 dir_n=120001
2026-05-09 07:51:20,409 INFO train_multi TF=ALL epoch 39/50 train=0.6652 val=0.6630 dir_acc=0.627 dir_n=120001
2026-05-09 07:51:33,580 INFO train_multi TF=ALL epoch 40/50 train=0.6650 val=0.6616 dir_acc=0.629 dir_n=120001
2026-05-09 07:51:33,585 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:51:33,585 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:51:33,585 INFO train_multi TF=ALL: new best val=0.6616 — saved
2026-05-09 07:51:46,582 INFO train_multi TF=ALL epoch 41/50 train=0.6640 val=0.6616 dir_acc=0.629 dir_n=120001
2026-05-09 07:51:46,587 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:51:46,587 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:51:46,587 INFO train_multi TF=ALL: new best val=0.6616 — saved
2026-05-09 07:51:59,856 INFO train_multi TF=ALL epoch 42/50 train=0.6636 val=0.6611 dir_acc=0.629 dir_n=120001
2026-05-09 07:51:59,861 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:51:59,861 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:51:59,861 INFO train_multi TF=ALL: new best val=0.6611 — saved
2026-05-09 07:52:13,116 INFO train_multi TF=ALL epoch 43/50 train=0.6634 val=0.6620 dir_acc=0.628 dir_n=120001
2026-05-09 07:52:26,365 INFO train_multi TF=ALL epoch 44/50 train=0.6629 val=0.6615 dir_acc=0.628 dir_n=120001
2026-05-09 07:52:39,427 INFO train_multi TF=ALL epoch 45/50 train=0.6627 val=0.6611 dir_acc=0.630 dir_n=120001
2026-05-09 07:52:52,596 INFO train_multi TF=ALL epoch 46/50 train=0.6624 val=0.6608 dir_acc=0.630 dir_n=120001
2026-05-09 07:52:52,600 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:52:52,601 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:52:52,601 INFO train_multi TF=ALL: new best val=0.6608 — saved
2026-05-09 07:53:05,705 INFO train_multi TF=ALL epoch 47/50 train=0.6618 val=0.6607 dir_acc=0.629 dir_n=120001
2026-05-09 07:53:05,711 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:53:05,711 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:53:05,711 INFO train_multi TF=ALL: new best val=0.6607 — saved
2026-05-09 07:53:18,786 INFO train_multi TF=ALL epoch 48/50 train=0.6617 val=0.6601 dir_acc=0.630 dir_n=120001
2026-05-09 07:53:18,791 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:53:18,791 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:53:18,791 INFO train_multi TF=ALL: new best val=0.6601 — saved
2026-05-09 07:53:32,137 INFO train_multi TF=ALL epoch 49/50 train=0.6611 val=0.6602 dir_acc=0.631 dir_n=120001
2026-05-09 07:53:45,129 INFO train_multi TF=ALL epoch 50/50 train=0.6612 val=0.6614 dir_acc=0.628 dir_n=120001
2026-05-09 07:53:45,257 INFO Retrain complete. Total wall-clock: 673.9s
2026-05-09 07:53:47,027 INFO Model gru: SUCCESS
2026-05-09 07:53:47,027 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:53:47,027 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-09 07:53:47,027 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-09 07:53:47,027 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-05-09 07:53:47,027 INFO   [DORMANT] rl_ppo — RL_ENABLED=false, skipping
2026-05-09 07:53:47,028 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-05-09 07:53:47,028 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer']
2026-05-09 07:53:47,029 INFO Saved 10 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Clean Quality/RL source: Backtest on train window ===
  START Train-window backtest for Quality/RL labels
2026-05-09 07:53:47,847 INFO === STEP 6: BACKTEST (train) ===
2026-05-09 07:53:47,848 INFO BT_WINDOW=train — train-window backtest: 2016-01-04 → 2022-08-05 (clean Quality/RL labels)
2026-05-09 07:53:47,848 INFO ================================================================
  ROUND 0 / 3
================================================================
2026-05-09 07:53:47,849 INFO Round 0 — running backtest: 2016-01-04 → 2022-08-05 (ml_trader, shared ML cache)
2026-05-09 07:57:17,817 ERROR _precompute_ml_cache failed for EURJPY: ML cache alignment left gaps in regime_htf for EURJPY
2026-05-09 07:57:17,921 ERROR _precompute_ml_cache failed for USDJPY: ML cache alignment left gaps in regime_htf for USDJPY
2026-05-09 07:57:18,092 ERROR _precompute_ml_cache failed for EURUSD: ML cache alignment left gaps in regime_htf for EURUSD
2026-05-09 07:57:23,527 ERROR _precompute_ml_cache failed for GBPJPY: ML cache alignment left gaps in regime_htf for GBPJPY
2026-05-09 07:57:23,623 ERROR _precompute_ml_cache failed for GBPUSD: ML cache alignment left gaps in regime_htf for GBPUSD
2026-05-09 07:57:23,743 WARNING _build_sequence_df: HTF frame 5M filled 357 warmup/alignment gaps with 0.000
2026-05-09 07:57:23,801 WARNING _build_sequence_df: HTF frame 5M filled 357 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:753: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:759: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:767: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
Traceback (most recent call last):
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 2378, in _backtest_trader
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
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 2361, in _build_cache_sym
    return sym, _precompute_ml_cache(
                ^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 1500, in _precompute_ml_cache
    _regime_htf_series = _align_complete(_r4h, df.index, "regime_htf", int)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 1492, in _align_complete
    raise RuntimeError(f"ML cache alignment left gaps in {name} for {symbol}")
RuntimeError: ML cache alignment left gaps in regime_htf for EURJPY

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 3860, in <module>
    main()
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 3674, in main
    result = _backtest_trader("ml_trader", symbols, pm, bt_start, bt_end,
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 2382, in _backtest_trader
    raise RuntimeError(f"ML cache build failed for {sym}: {exc}") from exc
RuntimeError: ML cache build failed for EURJPY: ML cache alignment left gaps in regime_htf for EURJPY
2026-05-09 07:57:27,951 ERROR Backtest failed (rc=1) — check trading-engine/logs/backtest_*.log
2026-05-09 07:57:27,951 ERROR Round 0 backtest failed: backtest exited 1
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/kaggle/working/Multi-Bot/trading-system/kaggle_train.py in <module>
    340 
    341 print("\n=== Clean Quality/RL source: Backtest on train window ===")
--> 342 run_step(
    343     "Train-window backtest for Quality/RL labels",
    344     "step6_backtest.py",

/kaggle/working/Multi-Bot/trading-system/kaggle_train.py in run_step(name, script, done_check, extra_env)
    198     )
    199     if result.returncode != 0:
--> 200         raise RuntimeError(f"{name} FAILED (exit {result.returncode})")
    201     print(f"  DONE  {name}")
    202 

RuntimeError: Train-window backtest for Quality/RL labels FAILED (exit 1)