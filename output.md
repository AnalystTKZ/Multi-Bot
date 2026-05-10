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
2026-05-10 01:43:19,937 INFO Loading feature-engineered data...
2026-05-10 01:43:20,668 INFO Loaded 221743 rows, 202 features
2026-05-10 01:43:20,670 INFO Data span: 2016-01-04 -> 2025-08-05  (9.6 years)
2026-05-10 01:43:20,675 INFO Fold 000 train 2016-01-04 -> 2018-01-03 (47088 bars), val 2018-01-04 -> 2019-01-03 (23448 bars)
2026-05-10 01:43:20,676 INFO Fold 001 train 2016-01-04 -> 2019-01-03 (70536 bars), val 2019-01-04 -> 2020-01-03 (23377 bars)
2026-05-10 01:43:20,676 INFO Fold 002 train 2016-01-04 -> 2020-01-03 (93913 bars), val 2020-01-06 -> 2020-12-31 (23259 bars)
2026-05-10 01:43:20,676 INFO Fold 003 train 2016-01-04 -> 2020-12-31 (117172 bars), val 2021-01-04 -> 2022-01-03 (23507 bars)
2026-05-10 01:43:20,677 INFO Fold 004 train 2016-01-04 -> 2022-01-03 (140679 bars), val 2022-01-04 -> 2023-01-03 (23588 bars)
2026-05-10 01:43:20,677 INFO Fold 005 train 2016-01-04 -> 2022-08-05 (154539 bars), val 2022-08-05 -> 2023-08-04 (20412 bars)
2026-05-10 01:43:20,677 INFO No leakage confirmed: train/train_tail/internal folds end before final 2-year blind test

=== SPLIT COMPLETE (EXPANDING CALENDAR, no shuffling) ===
  Folds:            6 expanding folds (min 2y train + 1y val, step=1y)
  Selected:   fold_005 for internal validation alias
  Train:      174,951 bars  2016-01-04 -> 2023-08-04  <- model fitting
  TrainTail:   44,000 bars  2021-08-05 -> 2023-08-04  <- Round 1 seen backtest
  Validation:  20,412 bars  2022-08-05 -> 2023-08-04  <- internal only
  Test:        46,792 bars  2023-08-07 -> 2025-08-05  <- Blind / Round 2
  Features:   202
  Leakage check: PASS
  DONE  Step 5 - Split

  Data split (expanding_calendar):
    train         174951 bars  2016-01-04 → 2023-08-04
    validation     20412 bars  2022-08-05 → 2023-08-04
    train_tail     44000 bars  2021-08-05 → 2023-08-04
    test           46792 bars  2023-08-07 → 2025-08-05

=== Phase 7a: Train GRU + Regime (train set only) ===
  START Step 7a - GRU+Regime
2026-05-10 01:43:30,537 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-05-10 01:43:30,537 INFO --- Training regime ---
2026-05-10 01:43:30,538 INFO Running retrain --model regime
2026-05-10 01:43:30,730 INFO retrain environment: KAGGLE
2026-05-10 01:43:32,435 INFO Device: CUDA (2 GPU(s))
2026-05-10 01:43:32,446 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-10 01:43:32,446 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-10 01:43:32,446 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-10 01:43:32,449 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-10 01:43:32,449 INFO Retrain data split: train
2026-05-10 01:43:32,449 INFO Retrain rolling fold selector: latest
2026-05-10 01:43:32,450 INFO === RegimeClassifier retrain (hierarchical: HTF 2-score bias + LTF 5-score behaviour) ===
2026-05-10 01:43:32,631 INFO NumExpr defaulting to 4 threads.
2026-05-10 01:43:32,850 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-10 01:43:32,850 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-10 01:43:32,850 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-10 01:43:32,850 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-10 01:43:32,905 INFO Regime rolling folds selected: [None]
2026-05-10 01:43:32,906 INFO === Regime rolling fold 1/1: train_all ===
2026-05-10 01:43:32,906 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-10 01:43:32,949 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-10 01:43:32,950 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
2026-05-10 01:43:32,967 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
2026-05-10 01:43:32,982 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
2026-05-10 01:43:32,997 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
2026-05-10 01:43:33,012 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
2026-05-10 01:43:33,027 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-10 01:43:33,283 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-10 01:43:33,379 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-10 01:43:33,405 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-10 01:43:33,406 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-10 01:43:33,417 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-10 01:43:33,418 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 01:43:33,885 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 492, 'BIAS_DOWN': 270, 'BIAS_NEUTRAL': 11340}  ambiguous=6929 (total=12102) horizon=12
2026-05-10 01:43:33,890 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected XAUUSD — 12052 samples (group=gold) score_means={'bias_up_score': 0.0408, 'bias_down_score': 0.0224} labels={'BIAS_UP': 492, 'BIAS_DOWN': 270, 'BIAS_NEUTRAL': 11290} clean={'BIAS_UP': 492, 'BIAS_DOWN': 270, 'BIAS_NEUTRAL': 4396}
2026-05-10 01:43:34,070 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:34,108 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:34,126 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:34,126 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:34,133 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:34,134 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 01:43:34,502 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 294, 'BIAS_DOWN': 349, 'BIAS_NEUTRAL': 10761}  ambiguous=6552 (total=11404) horizon=12
2026-05-10 01:43:34,507 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURUSD — 11354 samples (group=dollar) score_means={'bias_up_score': 0.0259, 'bias_down_score': 0.0307} labels={'BIAS_UP': 294, 'BIAS_DOWN': 349, 'BIAS_NEUTRAL': 10711} clean={'BIAS_UP': 294, 'BIAS_DOWN': 349, 'BIAS_NEUTRAL': 4188}
2026-05-10 01:43:34,672 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:34,709 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:34,729 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:34,729 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:34,738 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:34,739 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 01:43:35,101 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 465, 'BIAS_DOWN': 243, 'BIAS_NEUTRAL': 10695}  ambiguous=6644 (total=11403) horizon=12
2026-05-10 01:43:35,106 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected USDJPY — 11353 samples (group=dollar) score_means={'bias_up_score': 0.041, 'bias_down_score': 0.0214} labels={'BIAS_UP': 465, 'BIAS_DOWN': 243, 'BIAS_NEUTRAL': 10645} clean={'BIAS_UP': 465, 'BIAS_DOWN': 243, 'BIAS_NEUTRAL': 4028}
2026-05-10 01:43:35,257 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:35,294 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:35,316 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:35,316 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:35,324 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:35,325 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 01:43:35,677 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 345, 'BIAS_DOWN': 200, 'BIAS_NEUTRAL': 10862}  ambiguous=6647 (total=11407) horizon=12
2026-05-10 01:43:35,682 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURJPY — 11357 samples (group=cross) score_means={'bias_up_score': 0.0304, 'bias_down_score': 0.0176} labels={'BIAS_UP': 345, 'BIAS_DOWN': 200, 'BIAS_NEUTRAL': 10812} clean={'BIAS_UP': 345, 'BIAS_DOWN': 200, 'BIAS_NEUTRAL': 4191}
2026-05-10 01:43:35,832 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:35,868 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:35,888 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:35,889 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:35,895 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:35,897 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 01:43:36,243 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 458, 'BIAS_DOWN': 281, 'BIAS_NEUTRAL': 10669}  ambiguous=6611 (total=11408) horizon=12
2026-05-10 01:43:36,249 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPJPY — 11358 samples (group=cross) score_means={'bias_up_score': 0.0403, 'bias_down_score': 0.0247} labels={'BIAS_UP': 458, 'BIAS_DOWN': 281, 'BIAS_NEUTRAL': 10619} clean={'BIAS_UP': 458, 'BIAS_DOWN': 281, 'BIAS_NEUTRAL': 4042}
2026-05-10 01:43:36,402 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:36,435 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:36,455 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:36,455 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:36,462 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:36,463 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 01:43:36,816 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 269, 'BIAS_DOWN': 344, 'BIAS_NEUTRAL': 10789}  ambiguous=6630 (total=11402) horizon=12
2026-05-10 01:43:36,822 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPUSD — 11352 samples (group=dollar) score_means={'bias_up_score': 0.0237, 'bias_down_score': 0.0303} labels={'BIAS_UP': 269, 'BIAS_DOWN': 344, 'BIAS_NEUTRAL': 10739} clean={'BIAS_UP': 269, 'BIAS_DOWN': 344, 'BIAS_NEUTRAL': 4148}
2026-05-10 01:43:36,882 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 803, 'BIAS_DOWN': 481, 'BIAS_NEUTRAL': 21431}, 'dollar': {'BIAS_UP': 1028, 'BIAS_DOWN': 936, 'BIAS_NEUTRAL': 32095}, 'gold': {'BIAS_UP': 492, 'BIAS_DOWN': 270, 'BIAS_NEUTRAL': 11290}}
2026-05-10 01:43:36,883 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0354, 'bias_down_score': 0.0212}, 'dollar': {'bias_up_score': 0.0302, 'bias_down_score': 0.0275}, 'gold': {'bias_up_score': 0.0408, 'bias_down_score': 0.0224}}
2026-05-10 01:43:36,883 INFO Regime[4H mode=htf_bias] label distribution by year: {2016: {'BIAS_UP': 286, 'BIAS_DOWN': 340, 'BIAS_NEUTRAL': 8196}, 2017: {'BIAS_UP': 461, 'BIAS_DOWN': 205, 'BIAS_NEUTRAL': 8447}, 2018: {'BIAS_UP': 213, 'BIAS_DOWN': 288, 'BIAS_NEUTRAL': 8629}, 2019: {'BIAS_UP': 210, 'BIAS_DOWN': 192, 'BIAS_NEUTRAL': 8700}, 2020: {'BIAS_UP': 298, 'BIAS_DOWN': 180, 'BIAS_NEUTRAL': 8633}, 2021: {'BIAS_UP': 294, 'BIAS_DOWN': 186, 'BIAS_NEUTRAL': 8611}, 2022: {'BIAS_UP': 370, 'BIAS_DOWN': 225, 'BIAS_NEUTRAL': 8526}, 2023: {'BIAS_UP': 191, 'BIAS_DOWN': 71, 'BIAS_NEUTRAL': 5074}}
2026-05-10 01:43:36,883 INFO Regime[4H mode=htf_bias] score means by year: {2016: {'bias_up_score': 0.0324, 'bias_down_score': 0.0385}, 2017: {'bias_up_score': 0.0506, 'bias_down_score': 0.0225}, 2018: {'bias_up_score': 0.0233, 'bias_down_score': 0.0315}, 2019: {'bias_up_score': 0.0231, 'bias_down_score': 0.0211}, 2020: {'bias_up_score': 0.0327, 'bias_down_score': 0.0198}, 2021: {'bias_up_score': 0.0323, 'bias_down_score': 0.0205}, 2022: {'bias_up_score': 0.0406, 'bias_down_score': 0.0247}, 2023: {'bias_up_score': 0.0358, 'bias_down_score': 0.0133}}
2026-05-10 01:43:36,926 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
2026-05-10 01:43:36,927 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
2026-05-10 01:43:36,928 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
2026-05-10 01:43:36,929 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-10 01:43:36,929 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-10 01:43:36,930 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
2026-05-10 01:43:36,946 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-10 01:43:36,950 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-10 01:43:36,951 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-10 01:43:36,951 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-10 01:43:36,951 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-10 01:43:36,952 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 01:43:37,167 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 39, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1512}  ambiguous=936 (total=1581) horizon=12
2026-05-10 01:43:37,170 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected XAUUSD — 1531 samples (group=gold) score_means={'bias_up_score': 0.0255, 'bias_down_score': 0.0196} labels={'BIAS_UP': 39, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1462} clean={'BIAS_UP': 39, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 555}
2026-05-10 01:43:37,241 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:37,244 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:37,244 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:37,245 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:37,245 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:37,247 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 01:43:37,448 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 35, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1403}  ambiguous=861 (total=1491) horizon=12
2026-05-10 01:43:37,451 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURUSD — 1441 samples (group=dollar) score_means={'bias_up_score': 0.0243, 'bias_down_score': 0.0368} labels={'BIAS_UP': 35, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1353} clean={'BIAS_UP': 35, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 532}
2026-05-10 01:43:37,517 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:37,520 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:37,520 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:37,521 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:37,521 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:37,522 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 01:43:37,718 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 102, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 1362}  ambiguous=886 (total=1489) horizon=12
2026-05-10 01:43:37,721 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected USDJPY — 1439 samples (group=dollar) score_means={'bias_up_score': 0.0709, 'bias_down_score': 0.0174} labels={'BIAS_UP': 102, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 1312} clean={'BIAS_UP': 102, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 454}
2026-05-10 01:43:37,805 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:37,808 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:37,809 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:37,809 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:37,810 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:37,811 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 01:43:38,013 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 46, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1447}  ambiguous=915 (total=1494) horizon=12
2026-05-10 01:43:38,016 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0319, 'bias_down_score': 0.0007} labels={'BIAS_UP': 46, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1397} clean={'BIAS_UP': 46, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 509}
2026-05-10 01:43:38,083 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:38,085 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:38,086 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:38,087 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:38,087 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:38,088 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 01:43:38,282 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 29, 'BIAS_DOWN': 21, 'BIAS_NEUTRAL': 1444}  ambiguous=861 (total=1494) horizon=12
2026-05-10 01:43:38,284 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0201, 'bias_down_score': 0.0145} labels={'BIAS_UP': 29, 'BIAS_DOWN': 21, 'BIAS_NEUTRAL': 1394} clean={'BIAS_UP': 29, 'BIAS_DOWN': 21, 'BIAS_NEUTRAL': 561}
2026-05-10 01:43:38,351 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:38,353 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:38,354 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:38,355 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:38,355 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 01:43:38,356 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 01:43:38,561 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 26, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 1440}  ambiguous=885 (total=1488) horizon=12
2026-05-10 01:43:38,564 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPUSD — 1438 samples (group=dollar) score_means={'bias_up_score': 0.0181, 'bias_down_score': 0.0153} labels={'BIAS_UP': 26, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 1390} clean={'BIAS_UP': 26, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 543}
2026-05-10 01:43:38,624 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 75, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 2791}, 'dollar': {'BIAS_UP': 163, 'BIAS_DOWN': 100, 'BIAS_NEUTRAL': 4055}, 'gold': {'BIAS_UP': 39, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1462}}
2026-05-10 01:43:38,624 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.026, 'bias_down_score': 0.0076}, 'dollar': {'bias_up_score': 0.0377, 'bias_down_score': 0.0232}, 'gold': {'bias_up_score': 0.0255, 'bias_down_score': 0.0196}}
2026-05-10 01:43:38,624 INFO Regime[4H mode=htf_bias] label distribution by year: {2022: {'BIAS_UP': 91, 'BIAS_DOWN': 81, 'BIAS_NEUTRAL': 3229}, 2023: {'BIAS_UP': 186, 'BIAS_DOWN': 71, 'BIAS_NEUTRAL': 5079}}
2026-05-10 01:43:38,624 INFO Regime[4H mode=htf_bias] score means by year: {2022: {'bias_up_score': 0.0268, 'bias_down_score': 0.0238}, 2023: {'bias_up_score': 0.0349, 'bias_down_score': 0.0133}}
2026-05-10 01:43:38,665 INFO Regime phase HTF dataset build fold=train_all: 5.8s (train=68826 val=8737)
2026-05-10 01:43:38,666 INFO Regime 4H/htf_bias cold start: no existing weights found
2026-05-10 01:43:38,671 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=68826 val=8737 train_labels={'BIAS_UP': 2323, 'BIAS_DOWN': 1687, 'BIAS_NEUTRAL': 64816} val_labels={'BIAS_UP': 277, 'BIAS_DOWN': 152, 'BIAS_NEUTRAL': 8308}
2026-05-10 01:43:39,028 INFO RegimeClassifier[mode=htf_bias]: cold start HTF score head
2026-05-10 01:43:39,028 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-10 01:43:39,029 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 8.0, 'bias_down_score': 8.0}
2026-05-10 01:43:47,589 INFO Regime HTF score epoch  1/50 — tr=1.1385 va=0.8511 acc=0.899 bal=0.373 threshold=0.60 margin=0.05 recall={'BIAS_UP': 0.04, 'BIAS_DOWN': 0.138, 'BIAS_NEUTRAL': 0.941} precision={'BIAS_UP': 0.059, 'BIAS_DOWN': 0.063, 'BIAS_NEUTRAL': 0.952}
2026-05-10 01:43:48,260 INFO Regime HTF score epoch  2/50 — tr=1.0999 va=0.8307 bal=0.356
2026-05-10 01:43:48,898 INFO Regime HTF score epoch  3/50 — tr=1.0254 va=0.8034 bal=0.514
2026-05-10 01:43:49,525 INFO Regime HTF score epoch  4/50 — tr=0.9290 va=0.7774 bal=0.736
2026-05-10 01:43:50,146 INFO Regime HTF score epoch  5/50 — tr=0.8354 va=0.7602 acc=0.825 bal=0.771 threshold=0.75 margin=0.30 recall={'BIAS_UP': 0.852, 'BIAS_DOWN': 0.632, 'BIAS_NEUTRAL': 0.828} precision={'BIAS_UP': 0.187, 'BIAS_DOWN': 0.193, 'BIAS_NEUTRAL': 0.986}
2026-05-10 01:43:50,806 INFO Regime HTF score epoch  6/50 — tr=0.7573 va=0.7377 bal=0.837
2026-05-10 01:43:51,436 INFO Regime HTF score epoch  7/50 — tr=0.6934 va=0.7015 bal=0.719
2026-05-10 01:43:52,059 INFO Regime HTF score epoch  8/50 — tr=0.6400 va=0.6635 bal=0.744
2026-05-10 01:43:52,702 INFO Regime HTF score epoch  9/50 — tr=0.5915 va=0.6205 bal=0.743
2026-05-10 01:43:53,359 INFO Regime HTF score epoch 10/50 — tr=0.5462 va=0.5815 acc=0.839 bal=0.816 threshold=0.80 margin=0.00 recall={'BIAS_UP': 0.877, 'BIAS_DOWN': 0.73, 'BIAS_NEUTRAL': 0.839} precision={'BIAS_UP': 0.216, 'BIAS_DOWN': 0.197, 'BIAS_NEUTRAL': 0.989}
2026-05-10 01:43:54,038 INFO Regime HTF score epoch 11/50 — tr=0.5112 va=0.5483 bal=0.827
2026-05-10 01:43:54,680 INFO Regime HTF score epoch 12/50 — tr=0.4799 va=0.5144 bal=0.874
2026-05-10 01:43:55,350 INFO Regime HTF score epoch 13/50 — tr=0.4524 va=0.4908 bal=0.884
2026-05-10 01:43:56,032 INFO Regime HTF score epoch 14/50 — tr=0.4303 va=0.4654 bal=0.892
2026-05-10 01:43:56,657 INFO Regime HTF score epoch 15/50 — tr=0.4085 va=0.4446 acc=0.827 bal=0.896 threshold=0.75 margin=0.00 recall={'BIAS_UP': 0.931, 'BIAS_DOWN': 0.934, 'BIAS_NEUTRAL': 0.821} precision={'BIAS_UP': 0.218, 'BIAS_DOWN': 0.203, 'BIAS_NEUTRAL': 0.996}
2026-05-10 01:43:57,281 INFO Regime HTF score epoch 16/50 — tr=0.3891 va=0.4251 bal=0.898
2026-05-10 01:43:57,918 INFO Regime HTF score epoch 17/50 — tr=0.3734 va=0.4124 bal=0.866
2026-05-10 01:43:58,558 INFO Regime HTF score epoch 18/50 — tr=0.3601 va=0.3995 bal=0.900
2026-05-10 01:43:59,189 INFO Regime HTF score epoch 19/50 — tr=0.3480 va=0.3881 bal=0.908
2026-05-10 01:43:59,819 INFO Regime HTF score epoch 20/50 — tr=0.3405 va=0.3808 acc=0.837 bal=0.909 threshold=0.75 margin=0.00 recall={'BIAS_UP': 0.942, 'BIAS_DOWN': 0.954, 'BIAS_NEUTRAL': 0.831} precision={'BIAS_UP': 0.229, 'BIAS_DOWN': 0.218, 'BIAS_NEUTRAL': 0.997}
2026-05-10 01:44:00,458 INFO Regime HTF score epoch 21/50 — tr=0.3280 va=0.3703 bal=0.911
2026-05-10 01:44:01,082 INFO Regime HTF score epoch 22/50 — tr=0.3199 va=0.3586 bal=0.911
2026-05-10 01:44:01,729 INFO Regime HTF score epoch 23/50 — tr=0.3119 va=0.3477 bal=0.901
2026-05-10 01:44:02,368 INFO Regime HTF score epoch 24/50 — tr=0.3057 va=0.3456 bal=0.902
2026-05-10 01:44:03,026 INFO Regime HTF score epoch 25/50 — tr=0.3012 va=0.3407 acc=0.849 bal=0.908 threshold=0.75 margin=0.00 recall={'BIAS_UP': 0.939, 'BIAS_DOWN': 0.941, 'BIAS_NEUTRAL': 0.844} precision={'BIAS_UP': 0.244, 'BIAS_DOWN': 0.226, 'BIAS_NEUTRAL': 0.996}
2026-05-10 01:44:03,705 INFO Regime HTF score epoch 26/50 — tr=0.2990 va=0.3376 bal=0.910
2026-05-10 01:44:04,360 INFO Regime HTF score epoch 27/50 — tr=0.2917 va=0.3343 bal=0.912
2026-05-10 01:44:05,011 INFO Regime HTF score epoch 28/50 — tr=0.2880 va=0.3316 bal=0.914
2026-05-10 01:44:05,655 INFO Regime HTF score epoch 29/50 — tr=0.2825 va=0.3306 bal=0.917
2026-05-10 01:44:06,289 INFO Regime HTF score epoch 30/50 — tr=0.2794 va=0.3259 acc=0.851 bal=0.914 threshold=0.75 margin=0.00 recall={'BIAS_UP': 0.942, 'BIAS_DOWN': 0.954, 'BIAS_NEUTRAL': 0.846} precision={'BIAS_UP': 0.249, 'BIAS_DOWN': 0.228, 'BIAS_NEUTRAL': 0.997}
2026-05-10 01:44:06,911 INFO Regime HTF score epoch 31/50 — tr=0.2773 va=0.3244 bal=0.914
2026-05-10 01:44:07,531 INFO Regime HTF score epoch 32/50 — tr=0.2721 va=0.3185 bal=0.913
2026-05-10 01:44:08,162 INFO Regime HTF score epoch 33/50 — tr=0.2731 va=0.3185 bal=0.914
2026-05-10 01:44:08,791 INFO Regime HTF score epoch 34/50 — tr=0.2708 va=0.3218 bal=0.919
2026-05-10 01:44:09,420 INFO Regime HTF score epoch 35/50 — tr=0.2706 va=0.3191 acc=0.850 bal=0.916 threshold=0.75 margin=0.00 recall={'BIAS_UP': 0.949, 'BIAS_DOWN': 0.954, 'BIAS_NEUTRAL': 0.845} precision={'BIAS_UP': 0.25, 'BIAS_DOWN': 0.226, 'BIAS_NEUTRAL': 0.997}
2026-05-10 01:44:10,051 INFO Regime HTF score epoch 36/50 — tr=0.2661 va=0.3158 bal=0.916
2026-05-10 01:44:10,720 INFO Regime HTF score epoch 37/50 — tr=0.2670 va=0.3174 bal=0.920
2026-05-10 01:44:11,386 INFO Regime HTF score epoch 38/50 — tr=0.2680 va=0.3186 bal=0.920
2026-05-10 01:44:12,024 INFO Regime HTF score epoch 39/50 — tr=0.2615 va=0.3138 bal=0.919
2026-05-10 01:44:12,644 INFO Regime HTF score epoch 40/50 — tr=0.2632 va=0.3129 acc=0.853 bal=0.919 threshold=0.75 margin=0.00 recall={'BIAS_UP': 0.957, 'BIAS_DOWN': 0.954, 'BIAS_NEUTRAL': 0.847} precision={'BIAS_UP': 0.253, 'BIAS_DOWN': 0.229, 'BIAS_NEUTRAL': 0.997}
2026-05-10 01:44:13,309 INFO Regime HTF score epoch 41/50 — tr=0.2626 va=0.3099 bal=0.915
2026-05-10 01:44:14,016 INFO Regime HTF score epoch 42/50 — tr=0.2625 va=0.3129 bal=0.919
2026-05-10 01:44:14,675 INFO Regime HTF score epoch 43/50 — tr=0.2602 va=0.3140 bal=0.921
2026-05-10 01:44:15,306 INFO Regime HTF score epoch 44/50 — tr=0.2586 va=0.3130 bal=0.921
2026-05-10 01:44:15,937 INFO Regime HTF score epoch 45/50 — tr=0.2623 va=0.3107 acc=0.867 bal=0.881 threshold=0.80 margin=0.00 recall={'BIAS_UP': 0.895, 'BIAS_DOWN': 0.882, 'BIAS_NEUTRAL': 0.865} precision={'BIAS_UP': 0.263, 'BIAS_DOWN': 0.24, 'BIAS_NEUTRAL': 0.994}
2026-05-10 01:44:16,587 INFO Regime HTF score epoch 46/50 — tr=0.2597 va=0.3105 bal=0.882
2026-05-10 01:44:17,246 INFO Regime HTF score epoch 47/50 — tr=0.2617 va=0.3118 bal=0.919
2026-05-10 01:44:17,896 INFO Regime HTF score epoch 48/50 — tr=0.2604 va=0.3132 bal=0.921
2026-05-10 01:44:18,543 INFO Regime HTF score epoch 49/50 — tr=0.2602 va=0.3099 bal=0.917
2026-05-10 01:44:19,213 INFO Regime HTF score epoch 50/50 — tr=0.2610 va=0.3111 acc=0.853 bal=0.919 threshold=0.75 margin=0.00 recall={'BIAS_UP': 0.957, 'BIAS_DOWN': 0.954, 'BIAS_NEUTRAL': 0.848} precision={'BIAS_UP': 0.254, 'BIAS_DOWN': 0.229, 'BIAS_NEUTRAL': 0.997}
2026-05-10 01:44:19,753 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.750 margin=0.000 precision={'BIAS_UP': 0.253, 'BIAS_DOWN': 0.229, 'BIAS_NEUTRAL': 0.997} recall={'BIAS_UP': 0.957, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 0.847} f1={'BIAS_UP': 0.4, 'BIAS_DOWN': 0.37, 'BIAS_NEUTRAL': 0.916} confusion=[[265, 0, 12], [0, 146, 6], [783, 491, 7034]] score_mae={'bias_up_score': 0.1569, 'bias_down_score': 0.1057} pred_share={'BIAS_UP': 0.1199, 'BIAS_DOWN': 0.0729, 'BIAS_NEUTRAL': 0.8071}
2026-05-10 01:44:19,754 ERROR Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.253, 'BIAS_DOWN': 0.229, 'BIAS_NEUTRAL': 0.997} min_precision=0.500 recall={'BIAS_UP': 0.957, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 0.847} min_recall=0.100 f1={'BIAS_UP': 0.4, 'BIAS_DOWN': 0.37, 'BIAS_NEUTRAL': 0.916} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_UP', 'BIAS_DOWN'] weak_recall=[] weak_f1=[] weak_neutral=False. Refusing to save HTF weights because directional precision failed.
2026-05-10 01:44:19,754 INFO Regime phase HTF train fold=train_all: 41.1s
Traceback (most recent call last):
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1625, in <module>
    main()
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1594, in main
    result = retrain_regime(dry)
             ^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1303, in retrain_regime
    raise RuntimeError(f"Regime HTF training failed fold={fold_key}: {res_4h['error']}")
RuntimeError: Regime HTF training failed fold=train_all: Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.253, 'BIAS_DOWN': 0.229, 'BIAS_NEUTRAL': 0.997} min_precision=0.500 recall={'BIAS_UP': 0.957, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 0.847} min_recall=0.100 f1={'BIAS_UP': 0.4, 'BIAS_DOWN': 0.37, 'BIAS_NEUTRAL': 0.916} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_UP', 'BIAS_DOWN'] weak_recall=[] weak_f1=[] weak_neutral=False.

=== TRAINING COMPLETE ===
  regime: FAILED: exit 1
2026-05-10 01:44:24,217 ERROR retrain regime failed (exit 1)
2026-05-10 01:44:24,217 ERROR Model regime failed: exit 1
2026-05-10 01:44:24,218 WARNING   [MISSING] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 01:44:24,218 WARNING   [MISSING] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-10 01:44:24,218 WARNING   [MISSING] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-10 01:44:24,218 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-05-10 01:44:24,218 INFO   [DORMANT] rl_ppo — RL_ENABLED=false, skipping
2026-05-10 01:44:24,218 WARNING Missing required weights: ['gru_lstm', 'regime_htf', 'regime_ltf'] — run retrain_incremental.py for each
2026-05-10 01:44:24,218 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer']
2026-05-10 01:44:24,218 WARNING No retrain_history.jsonl found
2026-05-10 01:44:24,218 ERROR Step 7a failed; required training/artifacts missing: ['gru_lstm', 'regime', 'regime_htf', 'regime_ltf']
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/kaggle/working/Multi-Bot/trading-system/kaggle_train.py in <module>
    337 
    338 print("\n=== Phase 7a: Train GRU + Regime (train set only) ===")
--> 339 run_step(
    340     "Step 7a - GRU+Regime",
    341     "step7_train.py",

/kaggle/working/Multi-Bot/trading-system/kaggle_train.py in run_step(name, script, done_check, extra_env)
    207     )
    208     if result.returncode != 0:
--> 209         raise RuntimeError(f"{name} FAILED (exit {result.returncode})")
    210     print(f"  DONE  {name}")
    211 