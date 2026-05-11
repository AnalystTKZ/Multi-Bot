Cleared done-check: training_summary.json
Environment : KAGGLE
  base      -> /kaggle/working/Multi-Bot/trading-system
  data      -> /kaggle/input/datasets/tybobo/ml-dataset/training_data
  processed -> /kaggle/input/datasets/tybobo/ml-dataset/processed_data
  ml_train  -> /kaggle/working/Multi-Bot/trading-system/ml_training
  weights   -> /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
  output    -> /kaggle/working
  kaggle/input -> /kaggle/input
    dataset: datasets  (has training_data=False, processed_data=False)

All scripts and inputs verified.

=== Phase 0-5: Data preparation ===
  SKIP  Step 0 - Resample
  SKIP  Step 1 - Inventory
  SKIP  Step 2 - Cleaning
  SKIP  Step 3 - Alignment
  SKIP  Step 4 - Features
  START Step 5 - Split
2026-05-11 09:19:28,206 INFO Loading feature-engineered data...
2026-05-11 09:19:28,958 INFO Loaded 221743 rows, 202 features
2026-05-11 09:19:28,960 INFO Data span: 2016-01-04 -> 2025-08-05  (9.6 years)
2026-05-11 09:19:28,965 INFO Fold 000 train 2016-01-04 -> 2018-01-03 (47088 bars), val 2018-01-04 -> 2019-01-03 (23448 bars)
2026-05-11 09:19:28,966 INFO Fold 001 train 2016-01-04 -> 2019-01-03 (70536 bars), val 2019-01-04 -> 2020-01-03 (23377 bars)
2026-05-11 09:19:28,966 INFO Fold 002 train 2016-01-04 -> 2020-01-03 (93913 bars), val 2020-01-06 -> 2020-12-31 (23259 bars)
2026-05-11 09:19:28,966 INFO Fold 003 train 2016-01-04 -> 2020-12-31 (117172 bars), val 2021-01-04 -> 2022-01-03 (23507 bars)
2026-05-11 09:19:28,967 INFO Fold 004 train 2016-01-04 -> 2022-01-03 (140679 bars), val 2022-01-04 -> 2023-01-03 (23588 bars)
2026-05-11 09:19:28,967 INFO Fold 005 train 2016-01-04 -> 2022-08-05 (154539 bars), val 2022-08-05 -> 2023-08-04 (20412 bars)
2026-05-11 09:19:28,967 INFO No leakage confirmed: train/train_tail/internal folds end before final 2-year blind test

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
2026-05-11 09:19:38,370 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-05-11 09:19:38,370 INFO --- Training gru ---
2026-05-11 09:19:38,370 INFO Running retrain --model gru
2026-05-11 09:19:38,598 INFO retrain environment: KAGGLE
2026-05-11 09:19:40,162 INFO Device: CUDA (2 GPU(s))
2026-05-11 09:19:40,173 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 09:19:40,173 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 09:19:40,173 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 09:19:40,181 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 09:19:40,181 INFO Retrain data split: train
2026-05-11 09:19:40,181 INFO Retrain rolling fold selector: latest
2026-05-11 09:19:40,182 INFO === GRU-LSTM retrain (timeframes: ['15M']) ===
2026-05-11 09:19:40,345 INFO NumExpr defaulting to 4 threads.
2026-05-11 09:19:40,560 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-11 09:19:40,560 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 09:19:40,560 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 09:19:40,562 WARNING GRULSTMPredictor: stale weights detected (regime_4h feature contract changed: added=['body_direction_20', 'breakout_close_strength', 'candle_body_atr', 'candle_close_location', 'candle_range_atr', 'range_close_position_20', 'trend_body_pressure_20', 'wick_rejection_20']; count 28→36) — deleting /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt so retrain starts fresh
2026-05-11 09:19:40,563 INFO Deleted stale weights (regime_4h feature contract changed: added=['body_direction_20', 'breakout_close_strength', 'candle_body_atr', 'candle_close_location', 'candle_range_atr', 'range_close_position_20', 'trend_body_pressure_20', 'wick_rejection_20']; count 28→36): /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:19:40,563 INFO GRU multi-symbol training (Kaggle mode): 6 symbols × ['15M']
2026-05-11 09:19:40,564 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260511_091940
2026-05-11 09:19:40,564 INFO GRU weights stale (regime_4h feature contract changed: added=['body_direction_20', 'breakout_close_strength', 'candle_body_atr', 'candle_close_location', 'candle_range_atr', 'range_close_position_20', 'trend_body_pressure_20', 'wick_rejection_20']; count 28→36) — deleting for full retrain
2026-05-11 09:19:40,565 INFO Deleted stale GRU artifact for cold start: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/r_isotonic_long.pkl
2026-05-11 09:19:40,565 INFO Deleted stale GRU artifact for cold start: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/r_isotonic_short.pkl
2026-05-11 09:19:40,565 INFO GRU warm start disabled by default; set GRU_ALLOW_WARM_START=1 to reuse compatible weights
2026-05-11 09:19:40,827 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:19:40,855 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:19:40,872 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:19:40,883 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:19:40,962 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 09:19:40,969 INFO Loaded XAUUSD/15M split=train fold=latest: 176438 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:19:41,576 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:19:41,598 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:19:41,616 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:19:41,625 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:19:41,671 INFO Loaded EURUSD/15M split=train fold=latest: 174951 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:19:42,264 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:19:42,290 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:19:42,312 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:19:42,325 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:19:42,393 INFO Loaded USDJPY/15M split=train fold=latest: 174944 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:19:42,945 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:19:42,967 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:19:42,982 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:19:42,991 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:19:43,032 INFO Loaded EURJPY/15M split=train fold=latest: 174948 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:19:43,565 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:19:43,586 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:19:43,603 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:19:43,613 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:19:43,655 INFO Loaded GBPJPY/15M split=train fold=latest: 174902 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:19:44,200 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:19:44,220 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:19:44,234 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:19:44,244 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:19:44,284 INFO Loaded GBPUSD/15M split=train fold=latest: 174899 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:19:44,706 INFO train_multi: 6 segments, ~971854 total bars
2026-05-11 09:19:45,243 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-05-11 09:19:45,243 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-05-11 09:19:45,243 INFO train_multi: training ALL 6 segments across TFs ['15M'] in one combined pass
2026-05-11 09:19:45,243 INFO train_multi: building combined dataset for TF=ALL (6 segments)
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
2026-05-11 09:19:54,673 INFO train_multi TF=ALL: 971494 sequences across 6 segments
2026-05-11 09:19:54,673 INFO train_multi TF=ALL: estimated peak RAM = 21312 MB (train=419996 calib=60000 val=120002 n_feat=74 seq_len=60)
2026-05-11 09:19:54,673 WARNING train_multi TF=ALL: trimming to fit RAM budget — new train=394144 calib=56306 val=112612 (20000 MB est)
2026-05-11 09:19:56,992 INFO train_multi TF=ALL: train=394144 calib=56306 val=112612 (10009 MB tensors)
2026-05-11 09:20:03,972 INFO train_multi TF=ALL: structural bar weighting — 252452 structural bars (64.1%) weight=15.0 structural_only=0
2026-05-11 09:20:07,580 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=25, min_epochs=45)
2026-05-11 09:20:26,677 INFO train_multi TF=ALL epoch 1/100 train=2.3268 val=2.3304 r_mae=0.966 pos_r_acc=0.545 side_acc=0.493 r_n=161888
2026-05-11 09:20:26,690 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:20:26,690 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:20:26,690 INFO train_multi TF=ALL: new best val=2.3304 r_mae=0.9655 — saved
2026-05-11 09:20:26,694 INFO train_multi TF=ALL: new best r_mae=0.9655 — saved rmae checkpoint
2026-05-11 09:20:42,319 INFO train_multi TF=ALL epoch 2/100 train=2.3270 val=2.3299 r_mae=0.965 pos_r_acc=0.545 side_acc=0.493 r_n=161888
2026-05-11 09:20:42,325 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:20:42,325 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:20:42,325 INFO train_multi TF=ALL: new best val=2.3299 r_mae=0.9651 — saved
2026-05-11 09:20:42,329 INFO train_multi TF=ALL: new best r_mae=0.9651 — saved rmae checkpoint
2026-05-11 09:20:57,908 INFO train_multi TF=ALL epoch 3/100 train=2.3260 val=2.3294 r_mae=0.965 pos_r_acc=0.545 side_acc=0.494 r_n=161888
2026-05-11 09:20:57,913 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:20:57,913 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:20:57,914 INFO train_multi TF=ALL: new best val=2.3294 r_mae=0.9648 — saved
2026-05-11 09:20:57,918 INFO train_multi TF=ALL: new best r_mae=0.9648 — saved rmae checkpoint
2026-05-11 09:21:13,551 INFO train_multi TF=ALL epoch 4/100 train=2.3257 val=2.3287 r_mae=0.964 pos_r_acc=0.545 side_acc=0.510 r_n=161888
2026-05-11 09:21:13,556 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:21:13,556 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:21:13,557 INFO train_multi TF=ALL: new best val=2.3287 r_mae=0.9644 — saved
2026-05-11 09:21:13,561 INFO train_multi TF=ALL: new best r_mae=0.9644 — saved rmae checkpoint
2026-05-11 09:21:29,070 INFO train_multi TF=ALL epoch 5/100 train=2.3240 val=2.3265 r_mae=0.963 pos_r_acc=0.545 side_acc=0.517 r_n=161888
2026-05-11 09:21:29,075 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:21:29,075 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:21:29,075 INFO train_multi TF=ALL: new best val=2.3265 r_mae=0.9633 — saved
2026-05-11 09:21:29,079 INFO train_multi TF=ALL: new best r_mae=0.9633 — saved rmae checkpoint
2026-05-11 09:21:44,638 INFO train_multi TF=ALL epoch 6/100 train=2.3210 val=2.3255 r_mae=0.962 pos_r_acc=0.545 side_acc=0.518 r_n=161888
2026-05-11 09:21:44,643 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:21:44,643 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:21:44,643 INFO train_multi TF=ALL: new best val=2.3255 r_mae=0.9625 — saved
2026-05-11 09:21:44,647 INFO train_multi TF=ALL: new best r_mae=0.9625 — saved rmae checkpoint
2026-05-11 09:22:00,370 INFO train_multi TF=ALL epoch 7/100 train=2.3195 val=2.3243 r_mae=0.962 pos_r_acc=0.545 side_acc=0.518 r_n=161888
2026-05-11 09:22:00,376 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:22:00,376 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:22:00,376 INFO train_multi TF=ALL: new best val=2.3243 r_mae=0.9622 — saved
2026-05-11 09:22:00,380 INFO train_multi TF=ALL: new best r_mae=0.9622 — saved rmae checkpoint
2026-05-11 09:22:16,076 INFO train_multi TF=ALL epoch 8/100 train=2.3183 val=2.3230 r_mae=0.962 pos_r_acc=0.546 side_acc=0.523 r_n=161888
2026-05-11 09:22:16,081 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:22:16,081 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:22:16,081 INFO train_multi TF=ALL: new best val=2.3230 r_mae=0.9615 — saved
2026-05-11 09:22:16,085 INFO train_multi TF=ALL: new best r_mae=0.9615 — saved rmae checkpoint
2026-05-11 09:22:31,634 INFO train_multi TF=ALL epoch 9/100 train=2.3167 val=2.3215 r_mae=0.961 pos_r_acc=0.547 side_acc=0.522 r_n=161888
2026-05-11 09:22:31,639 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:22:31,639 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:22:31,639 INFO train_multi TF=ALL: new best val=2.3215 r_mae=0.9610 — saved
2026-05-11 09:22:31,643 INFO train_multi TF=ALL: new best r_mae=0.9610 — saved rmae checkpoint
2026-05-11 09:22:47,124 INFO train_multi TF=ALL epoch 10/100 train=2.3149 val=2.3204 r_mae=0.960 pos_r_acc=0.549 side_acc=0.526 r_n=161888
2026-05-11 09:22:47,129 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:22:47,129 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:22:47,130 INFO train_multi TF=ALL: new best val=2.3204 r_mae=0.9598 — saved
2026-05-11 09:22:47,134 INFO train_multi TF=ALL: new best r_mae=0.9598 — saved rmae checkpoint
2026-05-11 09:23:02,680 INFO train_multi TF=ALL epoch 11/100 train=2.3124 val=2.3160 r_mae=0.959 pos_r_acc=0.551 side_acc=0.530 r_n=161888
2026-05-11 09:23:02,685 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:23:02,685 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:23:02,685 INFO train_multi TF=ALL: new best val=2.3160 r_mae=0.9592 — saved
2026-05-11 09:23:02,690 INFO train_multi TF=ALL: new best r_mae=0.9592 — saved rmae checkpoint
2026-05-11 09:23:18,243 INFO train_multi TF=ALL epoch 12/100 train=2.3097 val=2.3125 r_mae=0.958 pos_r_acc=0.552 side_acc=0.535 r_n=161888
2026-05-11 09:23:18,254 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:23:18,255 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:23:18,255 INFO train_multi TF=ALL: new best val=2.3125 r_mae=0.9577 — saved
2026-05-11 09:23:18,259 INFO train_multi TF=ALL: new best r_mae=0.9577 — saved rmae checkpoint
2026-05-11 09:23:33,832 INFO train_multi TF=ALL epoch 13/100 train=2.3066 val=2.3039 r_mae=0.954 pos_r_acc=0.557 side_acc=0.540 r_n=161888
2026-05-11 09:23:33,837 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:23:33,837 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:23:33,837 INFO train_multi TF=ALL: new best val=2.3039 r_mae=0.9544 — saved
2026-05-11 09:23:33,841 INFO train_multi TF=ALL: new best r_mae=0.9544 — saved rmae checkpoint
2026-05-11 09:23:49,341 INFO train_multi TF=ALL epoch 14/100 train=2.2945 val=2.2914 r_mae=0.946 pos_r_acc=0.570 side_acc=0.545 r_n=161888
2026-05-11 09:23:49,346 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:23:49,346 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:23:49,346 INFO train_multi TF=ALL: new best val=2.2914 r_mae=0.9463 — saved
2026-05-11 09:23:49,351 INFO train_multi TF=ALL: new best r_mae=0.9463 — saved rmae checkpoint
2026-05-11 09:24:04,962 INFO train_multi TF=ALL epoch 15/100 train=2.2826 val=2.2821 r_mae=0.943 pos_r_acc=0.575 side_acc=0.550 r_n=161888
2026-05-11 09:24:04,973 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:24:04,973 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:24:04,973 INFO train_multi TF=ALL: new best val=2.2821 r_mae=0.9434 — saved
2026-05-11 09:24:04,977 INFO train_multi TF=ALL: new best r_mae=0.9434 — saved rmae checkpoint
2026-05-11 09:24:20,586 INFO train_multi TF=ALL epoch 16/100 train=2.2731 val=2.2722 r_mae=0.938 pos_r_acc=0.581 side_acc=0.553 r_n=161888
2026-05-11 09:24:20,590 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:24:20,591 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:24:20,591 INFO train_multi TF=ALL: new best val=2.2722 r_mae=0.9377 — saved
2026-05-11 09:24:20,595 INFO train_multi TF=ALL: new best r_mae=0.9377 — saved rmae checkpoint
2026-05-11 09:24:36,252 INFO train_multi TF=ALL epoch 17/100 train=2.2644 val=2.2690 r_mae=0.938 pos_r_acc=0.582 side_acc=0.556 r_n=161888
2026-05-11 09:24:36,257 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:24:36,257 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:24:36,257 INFO train_multi TF=ALL: new best val=2.2690 r_mae=0.9375 — saved
2026-05-11 09:24:36,261 INFO train_multi TF=ALL: new best r_mae=0.9375 — saved rmae checkpoint
2026-05-11 09:24:51,785 INFO train_multi TF=ALL epoch 18/100 train=2.2597 val=2.2654 r_mae=0.935 pos_r_acc=0.584 side_acc=0.556 r_n=161888
2026-05-11 09:24:51,790 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:24:51,790 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:24:51,790 INFO train_multi TF=ALL: new best val=2.2654 r_mae=0.9347 — saved
2026-05-11 09:24:51,794 INFO train_multi TF=ALL: new best r_mae=0.9347 — saved rmae checkpoint
2026-05-11 09:25:07,316 INFO train_multi TF=ALL epoch 19/100 train=2.2537 val=2.2607 r_mae=0.935 pos_r_acc=0.588 side_acc=0.557 r_n=161888
2026-05-11 09:25:07,321 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:25:07,321 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:25:07,321 INFO train_multi TF=ALL: new best val=2.2607 r_mae=0.9351 — saved
2026-05-11 09:25:22,929 INFO train_multi TF=ALL epoch 20/100 train=2.2502 val=2.2621 r_mae=0.932 pos_r_acc=0.583 side_acc=0.557 r_n=161888
2026-05-11 09:25:22,933 INFO train_multi TF=ALL: new best r_mae=0.9319 — saved rmae checkpoint
2026-05-11 09:25:38,635 INFO train_multi TF=ALL epoch 21/100 train=2.2460 val=2.2573 r_mae=0.929 pos_r_acc=0.588 side_acc=0.558 r_n=161888
2026-05-11 09:25:38,640 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:25:38,640 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:25:38,640 INFO train_multi TF=ALL: new best val=2.2573 r_mae=0.9290 — saved
2026-05-11 09:25:38,644 INFO train_multi TF=ALL: new best r_mae=0.9290 — saved rmae checkpoint
2026-05-11 09:25:54,259 INFO train_multi TF=ALL epoch 22/100 train=2.2423 val=2.2510 r_mae=0.926 pos_r_acc=0.595 side_acc=0.562 r_n=161888
2026-05-11 09:25:54,264 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:25:54,264 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:25:54,264 INFO train_multi TF=ALL: new best val=2.2510 r_mae=0.9264 — saved
2026-05-11 09:25:54,268 INFO train_multi TF=ALL: new best r_mae=0.9264 — saved rmae checkpoint
2026-05-11 09:26:09,826 INFO train_multi TF=ALL epoch 23/100 train=2.2384 val=2.2459 r_mae=0.927 pos_r_acc=0.595 side_acc=0.564 r_n=161888
2026-05-11 09:26:09,830 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:26:09,831 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:26:09,831 INFO train_multi TF=ALL: new best val=2.2459 r_mae=0.9272 — saved
2026-05-11 09:26:25,459 INFO train_multi TF=ALL epoch 24/100 train=2.2361 val=2.2472 r_mae=0.926 pos_r_acc=0.594 side_acc=0.562 r_n=161888
2026-05-11 09:26:25,463 INFO train_multi TF=ALL: new best r_mae=0.9258 — saved rmae checkpoint
2026-05-11 09:26:41,041 INFO train_multi TF=ALL epoch 25/100 train=2.2311 val=2.2407 r_mae=0.922 pos_r_acc=0.599 side_acc=0.567 r_n=161888
2026-05-11 09:26:41,046 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:26:41,047 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:26:41,047 INFO train_multi TF=ALL: new best val=2.2407 r_mae=0.9219 — saved
2026-05-11 09:26:41,051 INFO train_multi TF=ALL: new best r_mae=0.9219 — saved rmae checkpoint
2026-05-11 09:26:56,852 INFO train_multi TF=ALL epoch 26/100 train=2.2292 val=2.2428 r_mae=0.923 pos_r_acc=0.596 side_acc=0.562 r_n=161888
2026-05-11 09:27:12,396 INFO train_multi TF=ALL epoch 27/100 train=2.2249 val=2.2357 r_mae=0.920 pos_r_acc=0.600 side_acc=0.567 r_n=161888
2026-05-11 09:27:12,401 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:27:12,401 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:27:12,401 INFO train_multi TF=ALL: new best val=2.2357 r_mae=0.9197 — saved
2026-05-11 09:27:12,405 INFO train_multi TF=ALL: new best r_mae=0.9197 — saved rmae checkpoint
2026-05-11 09:27:27,942 INFO train_multi TF=ALL epoch 28/100 train=2.2212 val=2.2355 r_mae=0.920 pos_r_acc=0.601 side_acc=0.569 r_n=161888
2026-05-11 09:27:27,948 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:27:27,948 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:27:27,948 INFO train_multi TF=ALL: new best val=2.2355 r_mae=0.9205 — saved
2026-05-11 09:27:43,963 INFO train_multi TF=ALL epoch 29/100 train=2.2146 val=2.2289 r_mae=0.917 pos_r_acc=0.604 side_acc=0.569 r_n=161888
2026-05-11 09:27:43,969 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:27:43,969 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:27:43,969 INFO train_multi TF=ALL: new best val=2.2289 r_mae=0.9168 — saved
2026-05-11 09:27:43,973 INFO train_multi TF=ALL: new best r_mae=0.9168 — saved rmae checkpoint
2026-05-11 09:28:00,020 INFO train_multi TF=ALL epoch 30/100 train=2.2120 val=2.2262 r_mae=0.913 pos_r_acc=0.607 side_acc=0.573 r_n=161888
2026-05-11 09:28:00,025 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:28:00,025 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:28:00,026 INFO train_multi TF=ALL: new best val=2.2262 r_mae=0.9133 — saved
2026-05-11 09:28:00,035 INFO train_multi TF=ALL: new best r_mae=0.9133 — saved rmae checkpoint
2026-05-11 09:28:15,972 INFO train_multi TF=ALL epoch 31/100 train=2.2055 val=2.2204 r_mae=0.915 pos_r_acc=0.606 side_acc=0.575 r_n=161888
2026-05-11 09:28:15,977 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:28:15,977 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:28:15,977 INFO train_multi TF=ALL: new best val=2.2204 r_mae=0.9146 — saved
2026-05-11 09:28:32,159 INFO train_multi TF=ALL epoch 32/100 train=2.1982 val=2.2145 r_mae=0.910 pos_r_acc=0.609 side_acc=0.580 r_n=161888
2026-05-11 09:28:32,164 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:28:32,165 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:28:32,165 INFO train_multi TF=ALL: new best val=2.2145 r_mae=0.9100 — saved
2026-05-11 09:28:32,169 INFO train_multi TF=ALL: new best r_mae=0.9100 — saved rmae checkpoint
2026-05-11 09:28:48,347 INFO train_multi TF=ALL epoch 33/100 train=2.1874 val=2.2056 r_mae=0.905 pos_r_acc=0.613 side_acc=0.585 r_n=161888
2026-05-11 09:28:48,352 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:28:48,352 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:28:48,352 INFO train_multi TF=ALL: new best val=2.2056 r_mae=0.9054 — saved
2026-05-11 09:28:48,356 INFO train_multi TF=ALL: new best r_mae=0.9054 — saved rmae checkpoint
2026-05-11 09:29:04,778 INFO train_multi TF=ALL epoch 34/100 train=2.1783 val=2.1924 r_mae=0.904 pos_r_acc=0.614 side_acc=0.596 r_n=161888
2026-05-11 09:29:04,784 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:29:04,784 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:29:04,784 INFO train_multi TF=ALL: new best val=2.1924 r_mae=0.9040 — saved
2026-05-11 09:29:04,788 INFO train_multi TF=ALL: new best r_mae=0.9040 — saved rmae checkpoint
2026-05-11 09:29:21,011 INFO train_multi TF=ALL epoch 35/100 train=2.1635 val=2.1720 r_mae=0.893 pos_r_acc=0.622 side_acc=0.605 r_n=161888
2026-05-11 09:29:21,017 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:29:21,017 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:29:21,017 INFO train_multi TF=ALL: new best val=2.1720 r_mae=0.8935 — saved
2026-05-11 09:29:21,021 INFO train_multi TF=ALL: new best r_mae=0.8935 — saved rmae checkpoint
2026-05-11 09:29:37,560 INFO train_multi TF=ALL epoch 36/100 train=2.1383 val=2.1435 r_mae=0.884 pos_r_acc=0.630 side_acc=0.620 r_n=161888
2026-05-11 09:29:37,565 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:29:37,565 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:29:37,565 INFO train_multi TF=ALL: new best val=2.1435 r_mae=0.8845 — saved
2026-05-11 09:29:37,570 INFO train_multi TF=ALL: new best r_mae=0.8845 — saved rmae checkpoint
2026-05-11 09:29:53,971 INFO train_multi TF=ALL epoch 37/100 train=2.1108 val=2.1251 r_mae=0.871 pos_r_acc=0.641 side_acc=0.623 r_n=161888
2026-05-11 09:29:53,977 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:29:53,977 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:29:53,977 INFO train_multi TF=ALL: new best val=2.1251 r_mae=0.8708 — saved
2026-05-11 09:29:53,982 INFO train_multi TF=ALL: new best r_mae=0.8708 — saved rmae checkpoint
2026-05-11 09:30:10,447 INFO train_multi TF=ALL epoch 38/100 train=2.0859 val=2.0952 r_mae=0.857 pos_r_acc=0.653 side_acc=0.632 r_n=161888
2026-05-11 09:30:10,452 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:30:10,452 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:30:10,452 INFO train_multi TF=ALL: new best val=2.0952 r_mae=0.8566 — saved
2026-05-11 09:30:10,456 INFO train_multi TF=ALL: new best r_mae=0.8566 — saved rmae checkpoint
2026-05-11 09:30:26,711 INFO train_multi TF=ALL epoch 39/100 train=2.0612 val=2.0827 r_mae=0.844 pos_r_acc=0.659 side_acc=0.636 r_n=161888
2026-05-11 09:30:26,716 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:30:26,716 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:30:26,716 INFO train_multi TF=ALL: new best val=2.0827 r_mae=0.8443 — saved
2026-05-11 09:30:26,721 INFO train_multi TF=ALL: new best r_mae=0.8443 — saved rmae checkpoint
2026-05-11 09:30:43,204 INFO train_multi TF=ALL epoch 40/100 train=2.0482 val=2.0737 r_mae=0.840 pos_r_acc=0.660 side_acc=0.639 r_n=161888
2026-05-11 09:30:43,209 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:30:43,209 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:30:43,210 INFO train_multi TF=ALL: new best val=2.0737 r_mae=0.8396 — saved
2026-05-11 09:30:43,214 INFO train_multi TF=ALL: new best r_mae=0.8396 — saved rmae checkpoint
2026-05-11 09:31:00,256 INFO train_multi TF=ALL epoch 41/100 train=2.0347 val=2.0677 r_mae=0.832 pos_r_acc=0.659 side_acc=0.640 r_n=161888
2026-05-11 09:31:00,262 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:31:00,262 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:31:00,262 INFO train_multi TF=ALL: new best val=2.0677 r_mae=0.8316 — saved
2026-05-11 09:31:00,267 INFO train_multi TF=ALL: new best r_mae=0.8316 — saved rmae checkpoint
2026-05-11 09:31:17,814 INFO train_multi TF=ALL epoch 42/100 train=2.0200 val=2.0574 r_mae=0.829 pos_r_acc=0.665 side_acc=0.642 r_n=161888
2026-05-11 09:31:17,821 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:31:17,821 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:31:17,821 INFO train_multi TF=ALL: new best val=2.0574 r_mae=0.8286 — saved
2026-05-11 09:31:17,827 INFO train_multi TF=ALL: new best r_mae=0.8286 — saved rmae checkpoint
2026-05-11 09:31:34,395 INFO train_multi TF=ALL epoch 43/100 train=2.0107 val=2.0446 r_mae=0.820 pos_r_acc=0.668 side_acc=0.648 r_n=161888
2026-05-11 09:31:34,401 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:31:34,402 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:31:34,402 INFO train_multi TF=ALL: new best val=2.0446 r_mae=0.8199 — saved
2026-05-11 09:31:34,407 INFO train_multi TF=ALL: new best r_mae=0.8199 — saved rmae checkpoint
2026-05-11 09:31:50,674 INFO train_multi TF=ALL epoch 44/100 train=1.9989 val=2.0428 r_mae=0.816 pos_r_acc=0.670 side_acc=0.647 r_n=161888
2026-05-11 09:31:50,680 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:31:50,680 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:31:50,680 INFO train_multi TF=ALL: new best val=2.0428 r_mae=0.8161 — saved
2026-05-11 09:31:50,685 INFO train_multi TF=ALL: new best r_mae=0.8161 — saved rmae checkpoint
2026-05-11 09:32:06,911 INFO train_multi TF=ALL epoch 45/100 train=1.9857 val=2.0284 r_mae=0.817 pos_r_acc=0.670 side_acc=0.655 r_n=161888
2026-05-11 09:32:06,917 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:32:06,917 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:32:06,917 INFO train_multi TF=ALL: new best val=2.0284 r_mae=0.8172 — saved
2026-05-11 09:32:23,414 INFO train_multi TF=ALL epoch 46/100 train=1.9790 val=2.0266 r_mae=0.815 pos_r_acc=0.671 side_acc=0.655 r_n=161888
2026-05-11 09:32:23,419 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:32:23,419 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:32:23,419 INFO train_multi TF=ALL: new best val=2.0266 r_mae=0.8150 — saved
2026-05-11 09:32:23,423 INFO train_multi TF=ALL: new best r_mae=0.8150 — saved rmae checkpoint
2026-05-11 09:32:39,827 INFO train_multi TF=ALL epoch 47/100 train=1.9714 val=2.0231 r_mae=0.808 pos_r_acc=0.673 side_acc=0.655 r_n=161888
2026-05-11 09:32:39,833 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:32:39,833 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:32:39,833 INFO train_multi TF=ALL: new best val=2.0231 r_mae=0.8077 — saved
2026-05-11 09:32:39,837 INFO train_multi TF=ALL: new best r_mae=0.8077 — saved rmae checkpoint
2026-05-11 09:32:56,176 INFO train_multi TF=ALL epoch 48/100 train=1.9605 val=2.0101 r_mae=0.804 pos_r_acc=0.674 side_acc=0.662 r_n=161888
2026-05-11 09:32:56,181 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:32:56,181 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:32:56,181 INFO train_multi TF=ALL: new best val=2.0101 r_mae=0.8040 — saved
2026-05-11 09:32:56,185 INFO train_multi TF=ALL: new best r_mae=0.8040 — saved rmae checkpoint
2026-05-11 09:33:12,477 INFO train_multi TF=ALL epoch 49/100 train=1.9477 val=2.0060 r_mae=0.804 pos_r_acc=0.674 side_acc=0.665 r_n=161888
2026-05-11 09:33:12,482 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:33:12,483 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:33:12,483 INFO train_multi TF=ALL: new best val=2.0060 r_mae=0.8042 — saved
2026-05-11 09:33:28,718 INFO train_multi TF=ALL epoch 50/100 train=1.9382 val=2.0019 r_mae=0.801 pos_r_acc=0.675 side_acc=0.665 r_n=161888
2026-05-11 09:33:28,723 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:33:28,723 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:33:28,723 INFO train_multi TF=ALL: new best val=2.0019 r_mae=0.8010 — saved
2026-05-11 09:33:28,727 INFO train_multi TF=ALL: new best r_mae=0.8010 — saved rmae checkpoint
2026-05-11 09:33:44,941 INFO train_multi TF=ALL epoch 51/100 train=1.9296 val=1.9953 r_mae=0.800 pos_r_acc=0.674 side_acc=0.671 r_n=161888
2026-05-11 09:33:44,946 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:33:44,946 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:33:44,946 INFO train_multi TF=ALL: new best val=1.9953 r_mae=0.7998 — saved
2026-05-11 09:33:44,950 INFO train_multi TF=ALL: new best r_mae=0.7998 — saved rmae checkpoint
2026-05-11 09:34:01,161 INFO train_multi TF=ALL epoch 52/100 train=1.9212 val=1.9868 r_mae=0.801 pos_r_acc=0.676 side_acc=0.673 r_n=161888
2026-05-11 09:34:01,166 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:34:01,166 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:34:01,166 INFO train_multi TF=ALL: new best val=1.9868 r_mae=0.8008 — saved
2026-05-11 09:34:17,564 INFO train_multi TF=ALL epoch 53/100 train=1.9131 val=1.9915 r_mae=0.797 pos_r_acc=0.676 side_acc=0.669 r_n=161888
2026-05-11 09:34:17,568 INFO train_multi TF=ALL: new best r_mae=0.7968 — saved rmae checkpoint
2026-05-11 09:34:34,116 INFO train_multi TF=ALL epoch 54/100 train=1.9054 val=1.9825 r_mae=0.793 pos_r_acc=0.679 side_acc=0.671 r_n=161888
2026-05-11 09:34:34,121 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:34:34,121 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:34:34,121 INFO train_multi TF=ALL: new best val=1.9825 r_mae=0.7927 — saved
2026-05-11 09:34:34,126 INFO train_multi TF=ALL: new best r_mae=0.7927 — saved rmae checkpoint
2026-05-11 09:34:50,469 INFO train_multi TF=ALL epoch 55/100 train=1.8933 val=1.9753 r_mae=0.787 pos_r_acc=0.680 side_acc=0.677 r_n=161888
2026-05-11 09:34:50,474 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:34:50,474 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:34:50,475 INFO train_multi TF=ALL: new best val=1.9753 r_mae=0.7875 — saved
2026-05-11 09:34:50,479 INFO train_multi TF=ALL: new best r_mae=0.7875 — saved rmae checkpoint
2026-05-11 09:35:06,717 INFO train_multi TF=ALL epoch 56/100 train=1.8884 val=1.9697 r_mae=0.789 pos_r_acc=0.680 side_acc=0.679 r_n=161888
2026-05-11 09:35:06,722 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:35:06,722 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:35:06,722 INFO train_multi TF=ALL: new best val=1.9697 r_mae=0.7888 — saved
2026-05-11 09:35:22,951 INFO train_multi TF=ALL epoch 57/100 train=1.8842 val=1.9681 r_mae=0.797 pos_r_acc=0.679 side_acc=0.677 r_n=161888
2026-05-11 09:35:22,956 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:35:22,956 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:35:22,956 INFO train_multi TF=ALL: new best val=1.9681 r_mae=0.7973 — saved
2026-05-11 09:35:39,445 INFO train_multi TF=ALL epoch 58/100 train=1.8762 val=1.9743 r_mae=0.793 pos_r_acc=0.676 side_acc=0.680 r_n=161888
2026-05-11 09:35:55,690 INFO train_multi TF=ALL epoch 59/100 train=1.8690 val=1.9658 r_mae=0.789 pos_r_acc=0.680 side_acc=0.679 r_n=161888
2026-05-11 09:35:55,695 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:35:55,696 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:35:55,696 INFO train_multi TF=ALL: new best val=1.9658 r_mae=0.7889 — saved
2026-05-11 09:36:12,046 INFO train_multi TF=ALL epoch 60/100 train=1.8573 val=1.9676 r_mae=0.788 pos_r_acc=0.680 side_acc=0.681 r_n=161888
2026-05-11 09:36:28,195 INFO train_multi TF=ALL epoch 61/100 train=1.8510 val=1.9617 r_mae=0.782 pos_r_acc=0.682 side_acc=0.685 r_n=161888
2026-05-11 09:36:28,200 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:36:28,200 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:36:28,200 INFO train_multi TF=ALL: new best val=1.9617 r_mae=0.7816 — saved
2026-05-11 09:36:28,204 INFO train_multi TF=ALL: new best r_mae=0.7816 — saved rmae checkpoint
2026-05-11 09:36:44,721 INFO train_multi TF=ALL epoch 62/100 train=1.8471 val=1.9637 r_mae=0.785 pos_r_acc=0.680 side_acc=0.681 r_n=161888
2026-05-11 09:37:01,077 INFO train_multi TF=ALL epoch 63/100 train=1.8390 val=1.9557 r_mae=0.778 pos_r_acc=0.683 side_acc=0.686 r_n=161888
2026-05-11 09:37:01,082 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:37:01,082 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:37:01,082 INFO train_multi TF=ALL: new best val=1.9557 r_mae=0.7783 — saved
2026-05-11 09:37:01,086 INFO train_multi TF=ALL: new best r_mae=0.7783 — saved rmae checkpoint
2026-05-11 09:37:17,320 INFO train_multi TF=ALL epoch 64/100 train=1.8293 val=1.9541 r_mae=0.782 pos_r_acc=0.681 side_acc=0.688 r_n=161888
2026-05-11 09:37:17,326 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:37:17,326 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:37:17,326 INFO train_multi TF=ALL: new best val=1.9541 r_mae=0.7818 — saved
2026-05-11 09:37:33,519 INFO train_multi TF=ALL epoch 65/100 train=1.8223 val=1.9485 r_mae=0.781 pos_r_acc=0.682 side_acc=0.692 r_n=161888
2026-05-11 09:37:33,524 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:37:33,525 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:37:33,525 INFO train_multi TF=ALL: new best val=1.9485 r_mae=0.7811 — saved
2026-05-11 09:37:49,701 INFO train_multi TF=ALL epoch 66/100 train=1.8144 val=1.9508 r_mae=0.782 pos_r_acc=0.683 side_acc=0.688 r_n=161888
2026-05-11 09:38:05,947 INFO train_multi TF=ALL epoch 67/100 train=1.8099 val=1.9495 r_mae=0.784 pos_r_acc=0.678 side_acc=0.691 r_n=161888
2026-05-11 09:38:22,161 INFO train_multi TF=ALL epoch 68/100 train=1.7983 val=1.9435 r_mae=0.784 pos_r_acc=0.681 side_acc=0.694 r_n=161888
2026-05-11 09:38:22,167 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:38:22,167 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:38:22,167 INFO train_multi TF=ALL: new best val=1.9435 r_mae=0.7837 — saved
2026-05-11 09:38:38,406 INFO train_multi TF=ALL epoch 69/100 train=1.7959 val=1.9390 r_mae=0.786 pos_r_acc=0.679 side_acc=0.700 r_n=161888
2026-05-11 09:38:38,411 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:38:38,411 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:38:38,411 INFO train_multi TF=ALL: new best val=1.9390 r_mae=0.7862 — saved
2026-05-11 09:38:54,676 INFO train_multi TF=ALL epoch 70/100 train=1.7885 val=1.9284 r_mae=0.781 pos_r_acc=0.681 side_acc=0.704 r_n=161888
2026-05-11 09:38:54,682 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:38:54,682 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:38:54,682 INFO train_multi TF=ALL: new best val=1.9284 r_mae=0.7806 — saved
2026-05-11 09:39:10,750 INFO train_multi TF=ALL epoch 71/100 train=1.7771 val=1.9399 r_mae=0.781 pos_r_acc=0.683 side_acc=0.702 r_n=161888
2026-05-11 09:39:26,878 INFO train_multi TF=ALL epoch 72/100 train=1.7706 val=1.9210 r_mae=0.783 pos_r_acc=0.681 side_acc=0.707 r_n=161888
2026-05-11 09:39:26,883 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:39:26,883 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:39:26,883 INFO train_multi TF=ALL: new best val=1.9210 r_mae=0.7833 — saved
2026-05-11 09:39:43,158 INFO train_multi TF=ALL epoch 73/100 train=1.7655 val=1.9202 r_mae=0.784 pos_r_acc=0.680 side_acc=0.710 r_n=161888
2026-05-11 09:39:43,171 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:39:43,171 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:39:43,171 INFO train_multi TF=ALL: new best val=1.9202 r_mae=0.7843 — saved
2026-05-11 09:39:59,559 INFO train_multi TF=ALL epoch 74/100 train=1.7569 val=1.9283 r_mae=0.778 pos_r_acc=0.684 side_acc=0.705 r_n=161888
2026-05-11 09:40:15,820 INFO train_multi TF=ALL epoch 75/100 train=1.7502 val=1.9262 r_mae=0.777 pos_r_acc=0.681 side_acc=0.713 r_n=161888
2026-05-11 09:40:15,825 INFO train_multi TF=ALL: new best r_mae=0.7769 — saved rmae checkpoint
2026-05-11 09:40:31,991 INFO train_multi TF=ALL epoch 76/100 train=1.7455 val=1.9233 r_mae=0.778 pos_r_acc=0.680 side_acc=0.714 r_n=161888
2026-05-11 09:40:48,147 INFO train_multi TF=ALL epoch 77/100 train=1.7362 val=1.9298 r_mae=0.781 pos_r_acc=0.681 side_acc=0.711 r_n=161888
2026-05-11 09:41:04,407 INFO train_multi TF=ALL epoch 78/100 train=1.7300 val=1.9232 r_mae=0.786 pos_r_acc=0.679 side_acc=0.711 r_n=161888
2026-05-11 09:41:20,537 INFO train_multi TF=ALL epoch 79/100 train=1.7225 val=1.9389 r_mae=0.782 pos_r_acc=0.678 side_acc=0.709 r_n=161888
2026-05-11 09:41:36,716 INFO train_multi TF=ALL epoch 80/100 train=1.7202 val=1.9284 r_mae=0.780 pos_r_acc=0.680 side_acc=0.714 r_n=161888
2026-05-11 09:41:52,925 INFO train_multi TF=ALL epoch 81/100 train=1.7148 val=1.9454 r_mae=0.784 pos_r_acc=0.674 side_acc=0.713 r_n=161888
2026-05-11 09:42:09,077 INFO train_multi TF=ALL epoch 82/100 train=1.7064 val=1.9267 r_mae=0.781 pos_r_acc=0.678 side_acc=0.720 r_n=161888
2026-05-11 09:42:25,328 INFO train_multi TF=ALL epoch 83/100 train=1.7005 val=1.9360 r_mae=0.783 pos_r_acc=0.676 side_acc=0.719 r_n=161888
2026-05-11 09:42:41,695 INFO train_multi TF=ALL epoch 84/100 train=1.6928 val=1.9280 r_mae=0.783 pos_r_acc=0.678 side_acc=0.719 r_n=161888
2026-05-11 09:42:57,925 INFO train_multi TF=ALL epoch 85/100 train=1.6845 val=1.9410 r_mae=0.779 pos_r_acc=0.678 side_acc=0.717 r_n=161888
2026-05-11 09:43:14,248 INFO train_multi TF=ALL epoch 86/100 train=1.6816 val=1.9308 r_mae=0.783 pos_r_acc=0.677 side_acc=0.721 r_n=161888
2026-05-11 09:43:30,442 INFO train_multi TF=ALL epoch 87/100 train=1.6729 val=1.9751 r_mae=0.791 pos_r_acc=0.670 side_acc=0.707 r_n=161888
2026-05-11 09:43:46,706 INFO train_multi TF=ALL epoch 88/100 train=1.6676 val=1.9389 r_mae=0.782 pos_r_acc=0.677 side_acc=0.720 r_n=161888
2026-05-11 09:44:02,955 INFO train_multi TF=ALL epoch 89/100 train=1.6592 val=1.9436 r_mae=0.783 pos_r_acc=0.676 side_acc=0.721 r_n=161888
2026-05-11 09:44:19,124 INFO train_multi TF=ALL epoch 90/100 train=1.6528 val=1.9431 r_mae=0.785 pos_r_acc=0.675 side_acc=0.721 r_n=161888
2026-05-11 09:44:35,392 INFO train_multi TF=ALL epoch 91/100 train=1.6497 val=1.9557 r_mae=0.784 pos_r_acc=0.675 side_acc=0.720 r_n=161888
2026-05-11 09:44:51,652 INFO train_multi TF=ALL epoch 92/100 train=1.6426 val=1.9509 r_mae=0.782 pos_r_acc=0.676 side_acc=0.722 r_n=161888
2026-05-11 09:45:07,852 INFO train_multi TF=ALL epoch 93/100 train=1.6361 val=1.9517 r_mae=0.782 pos_r_acc=0.675 side_acc=0.719 r_n=161888
2026-05-11 09:45:23,935 INFO train_multi TF=ALL epoch 94/100 train=1.6294 val=1.9455 r_mae=0.784 pos_r_acc=0.675 side_acc=0.727 r_n=161888
2026-05-11 09:45:40,148 INFO train_multi TF=ALL epoch 95/100 train=1.6259 val=1.9513 r_mae=0.787 pos_r_acc=0.674 side_acc=0.724 r_n=161888
2026-05-11 09:45:56,397 INFO train_multi TF=ALL epoch 96/100 train=1.6203 val=1.9603 r_mae=0.782 pos_r_acc=0.676 side_acc=0.726 r_n=161888
2026-05-11 09:46:12,610 INFO train_multi TF=ALL epoch 97/100 train=1.6184 val=1.9633 r_mae=0.788 pos_r_acc=0.673 side_acc=0.721 r_n=161888
2026-05-11 09:46:28,855 INFO train_multi TF=ALL epoch 98/100 train=1.6117 val=1.9632 r_mae=0.781 pos_r_acc=0.675 side_acc=0.723 r_n=161888
2026-05-11 09:46:45,010 INFO train_multi TF=ALL epoch 99/100 train=1.6079 val=1.9726 r_mae=0.790 pos_r_acc=0.670 side_acc=0.723 r_n=161888
2026-05-11 09:47:01,280 INFO train_multi TF=ALL epoch 100/100 train=1.6011 val=1.9709 r_mae=0.786 pos_r_acc=0.674 side_acc=0.723 r_n=161888
2026-05-11 09:47:01,280 INFO train_multi TF=ALL early stop at epoch 100
2026-05-11 09:47:01,298 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:47:01,298 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:47:01,298 INFO train_multi TF=ALL: promoted r_mae checkpoint (0.7769 < primary 0.7843) — overwriting model.pt
2026-05-11 09:47:02,803 WARNING fit_r_isotonic: long calibrator increases MAE (cal=0.7906 >= raw=0.7833) — skipping
2026-05-11 09:47:02,816 WARNING fit_r_isotonic: short calibrator increases MAE (cal=0.8010 >= raw=0.7966) — skipping
2026-05-11 09:47:02,816 INFO fit_r_isotonic: saved side-R calibrators stats={'long': {'n': 40072, 'raw_mae': 0.7833163126756391, 'calibrated_mae': 0.790569066034469, 'skipped': 'calibrator_hurts'}, 'short': {'n': 41197, 'raw_mae': 0.796601655670924, 'calibrated_mae': 0.8009588598652648, 'skipped': 'calibrator_hurts'}}
2026-05-11 09:47:02,965 WARNING GRU validation R-MAE above floor for TF=ALL: best_val_r_mae=0.777 max=0.750. Keeping saved best weights so the pipeline can progress.
2026-05-11 09:47:02,979 INFO GRU R threshold XAUUSD/buy: q25_pos=0.535 q50_pos=1.667 pos_rate=46.0% (n=119259 n_pos=54899)
2026-05-11 09:47:02,986 INFO GRU R threshold XAUUSD/sell: q25_pos=0.528 q50_pos=1.667 pos_rate=45.5% (n=118087 n_pos=53734)
2026-05-11 09:47:02,993 INFO GRU R threshold EURUSD/buy: q25_pos=0.541 q50_pos=1.667 pos_rate=45.6% (n=118751 n_pos=54196)
2026-05-11 09:47:03,000 INFO GRU R threshold EURUSD/sell: q25_pos=0.545 q50_pos=1.667 pos_rate=45.6% (n=117679 n_pos=53721)
2026-05-11 09:47:03,006 INFO GRU R threshold USDJPY/buy: q25_pos=0.551 q50_pos=1.667 pos_rate=46.2% (n=118691 n_pos=54870)
2026-05-11 09:47:03,013 INFO GRU R threshold USDJPY/sell: q25_pos=0.542 q50_pos=1.667 pos_rate=44.6% (n=116030 n_pos=51793)
2026-05-11 09:47:03,019 INFO GRU R threshold EURJPY/buy: q25_pos=0.530 q50_pos=1.416 pos_rate=46.1% (n=118651 n_pos=54654)
2026-05-11 09:47:03,026 INFO GRU R threshold EURJPY/sell: q25_pos=0.525 q50_pos=1.667 pos_rate=44.7% (n=117320 n_pos=52399)
2026-05-11 09:47:03,032 INFO GRU R threshold GBPJPY/buy: q25_pos=0.517 q50_pos=1.443 pos_rate=45.9% (n=118277 n_pos=54278)
2026-05-11 09:47:03,038 INFO GRU R threshold GBPJPY/sell: q25_pos=0.521 q50_pos=1.667 pos_rate=45.3% (n=116299 n_pos=52702)
2026-05-11 09:47:03,044 INFO GRU R threshold GBPUSD/buy: q25_pos=0.524 q50_pos=1.667 pos_rate=45.8% (n=117764 n_pos=53894)
2026-05-11 09:47:03,050 INFO GRU R threshold GBPUSD/sell: q25_pos=0.533 q50_pos=1.667 pos_rate=45.8% (n=117404 n_pos=53785)
2026-05-11 09:47:03,051 INFO GRU per-symbol R thresholds saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/symbol_r_thresholds.json (6 symbols)
2026-05-11 09:47:03,096 INFO Retrain complete. Total wall-clock: 1642.9s
2026-05-11 09:47:09,326 INFO Model gru: SUCCESS
2026-05-11 09:47:09,327 INFO --- Training regime ---
2026-05-11 09:47:09,327 INFO Running retrain --model regime
2026-05-11 09:47:09,947 INFO retrain environment: KAGGLE
2026-05-11 09:47:11,725 INFO Device: CUDA (2 GPU(s))
2026-05-11 09:47:11,736 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 09:47:11,737 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 09:47:11,737 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 09:47:11,737 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 09:47:11,737 INFO Retrain data split: train
2026-05-11 09:47:11,737 INFO Retrain rolling fold selector: latest
2026-05-11 09:47:11,738 INFO === RegimeClassifier retrain (hierarchical: HTF 2-score bias + LTF 5-score behaviour) ===
2026-05-11 09:47:12,033 INFO NumExpr defaulting to 4 threads.
2026-05-11 09:47:12,294 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-11 09:47:12,295 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 09:47:12,295 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 09:47:12,295 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-11 09:47:12,359 INFO Regime rolling folds selected: [None]
2026-05-11 09:47:12,359 INFO === Regime rolling fold 1/1: train_all ===
2026-05-11 09:47:12,360 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-11 09:47:12,404 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 09:47:12,406 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
2026-05-11 09:47:12,423 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
2026-05-11 09:47:12,441 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
2026-05-11 09:47:12,471 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
2026-05-11 09:47:12,507 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
2026-05-11 09:47:12,532 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 09:47:12,782 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:47:12,855 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:47:12,883 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:47:12,883 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:47:12,897 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:47:12,898 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:47:13,326 INFO macro_cache: using training data dir /kaggle/working/Multi-Bot/trading-system/training_data
2026-05-11 09:47:13,467 INFO macro_cache loaded 19 series: {'asx200': '2842 bars 2015-01-02→2026-03-27', 'cac40': '2876 bars 2015-01-02→2026-03-27', 'dax': '2851 bars 2015-01-02→2026-03-27', 'djia': '2825 bars 2015-01-02→2026-03-27', 'dxy': '2826 bars 2015-01-02→2026-03-27', 'eurostoxx': '2823 bars 2015-01-05→2026-03-27', 'ftse': '2839 bars 2015-01-02→2026-03-27', 'gold_fut': '2824 bars 2015-01-02→2026-03-27', 'hsi': '2764 bars 2015-01-02→2026-03-27', 'nasdaq': '2825 bars 2015-01-02→2026-03-27', 'nikkei': '2744 bars 2015-01-05→2026-03-27', 'oil_fut': '2825 bars 2015-01-02→2026-03-27', 'spx': '2825 bars 2015-01-02→2026-03-27', 'us10y': '2824 bars 2015-01-02→2026-03-27', 'us30y': '2824 bars 2015-01-02→2026-03-27', 'us3m': '2824 bars 2015-01-02→2026-03-27', 'vix': '2825 bars 2015-01-02→2026-03-27', 'us10y_fred': '2607 bars 2016-03-28→2026-03-24', 'us2y_fred': '2607 bars 2016-03-28→2026-03-24'}
2026-05-11 09:47:13,721 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1209, 'BIAS_DOWN': 568, 'BIAS_NEUTRAL': 10325}  ambiguous=3935 (total=12102) horizon=84
2026-05-11 09:47:13,726 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected XAUUSD — 12052 samples (group=gold) score_means={'bias_up_score': 0.1003, 'bias_down_score': 0.0471} labels={'BIAS_UP': 1209, 'BIAS_DOWN': 568, 'BIAS_NEUTRAL': 10275} clean={'BIAS_UP': 1209, 'BIAS_DOWN': 568, 'BIAS_NEUTRAL': 6348}
2026-05-11 09:47:13,913 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:13,957 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:13,977 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:13,978 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:13,987 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:13,989 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:47:14,610 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 722, 'BIAS_DOWN': 567, 'BIAS_NEUTRAL': 10115}  ambiguous=3689 (total=11404) horizon=84
2026-05-11 09:47:14,615 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURUSD — 11354 samples (group=dollar) score_means={'bias_up_score': 0.0636, 'bias_down_score': 0.0499} labels={'BIAS_UP': 722, 'BIAS_DOWN': 567, 'BIAS_NEUTRAL': 10065} clean={'BIAS_UP': 722, 'BIAS_DOWN': 567, 'BIAS_NEUTRAL': 6394}
2026-05-11 09:47:14,783 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:14,821 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:14,841 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:14,842 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:14,851 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:14,852 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:47:15,457 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 872, 'BIAS_DOWN': 463, 'BIAS_NEUTRAL': 10068}  ambiguous=3827 (total=11403) horizon=84
2026-05-11 09:47:15,462 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected USDJPY — 11353 samples (group=dollar) score_means={'bias_up_score': 0.0768, 'bias_down_score': 0.0408} labels={'BIAS_UP': 872, 'BIAS_DOWN': 463, 'BIAS_NEUTRAL': 10018} clean={'BIAS_UP': 872, 'BIAS_DOWN': 463, 'BIAS_NEUTRAL': 6201}
2026-05-11 09:47:15,624 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:15,663 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:15,686 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:15,686 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:15,695 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:15,697 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:47:16,302 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 712, 'BIAS_DOWN': 556, 'BIAS_NEUTRAL': 10139}  ambiguous=3816 (total=11407) horizon=84
2026-05-11 09:47:16,307 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURJPY — 11357 samples (group=cross) score_means={'bias_up_score': 0.0627, 'bias_down_score': 0.049} labels={'BIAS_UP': 712, 'BIAS_DOWN': 556, 'BIAS_NEUTRAL': 10089} clean={'BIAS_UP': 712, 'BIAS_DOWN': 556, 'BIAS_NEUTRAL': 6279}
2026-05-11 09:47:16,473 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:16,511 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:16,532 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:16,532 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:16,543 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:16,544 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:47:17,164 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 873, 'BIAS_DOWN': 633, 'BIAS_NEUTRAL': 9902}  ambiguous=4022 (total=11408) horizon=84
2026-05-11 09:47:17,169 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPJPY — 11358 samples (group=cross) score_means={'bias_up_score': 0.0769, 'bias_down_score': 0.0557} labels={'BIAS_UP': 873, 'BIAS_DOWN': 633, 'BIAS_NEUTRAL': 9852} clean={'BIAS_UP': 873, 'BIAS_DOWN': 633, 'BIAS_NEUTRAL': 5852}
2026-05-11 09:47:17,333 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:17,369 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:17,391 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:17,391 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:17,400 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:17,401 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:47:18,010 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 546, 'BIAS_DOWN': 754, 'BIAS_NEUTRAL': 10102}  ambiguous=3944 (total=11402) horizon=84
2026-05-11 09:47:18,016 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPUSD — 11352 samples (group=dollar) score_means={'bias_up_score': 0.0481, 'bias_down_score': 0.0651} labels={'BIAS_UP': 546, 'BIAS_DOWN': 739, 'BIAS_NEUTRAL': 10067} clean={'BIAS_UP': 546, 'BIAS_DOWN': 739, 'BIAS_NEUTRAL': 6149}
2026-05-11 09:47:18,087 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 1585, 'BIAS_DOWN': 1189, 'BIAS_NEUTRAL': 19941}, 'dollar': {'BIAS_UP': 2140, 'BIAS_DOWN': 1769, 'BIAS_NEUTRAL': 30150}, 'gold': {'BIAS_UP': 1209, 'BIAS_DOWN': 568, 'BIAS_NEUTRAL': 10275}}
2026-05-11 09:47:18,088 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0698, 'bias_down_score': 0.0523}, 'dollar': {'bias_up_score': 0.0628, 'bias_down_score': 0.0519}, 'gold': {'bias_up_score': 0.1003, 'bias_down_score': 0.0471}}
2026-05-11 09:47:18,088 INFO Regime[4H mode=htf_bias] label distribution by year: {2016: {'BIAS_UP': 525, 'BIAS_DOWN': 617, 'BIAS_NEUTRAL': 7680}, 2017: {'BIAS_UP': 776, 'BIAS_DOWN': 315, 'BIAS_NEUTRAL': 8022}, 2018: {'BIAS_UP': 453, 'BIAS_DOWN': 753, 'BIAS_NEUTRAL': 7924}, 2019: {'BIAS_UP': 427, 'BIAS_DOWN': 481, 'BIAS_NEUTRAL': 8194}, 2020: {'BIAS_UP': 721, 'BIAS_DOWN': 181, 'BIAS_NEUTRAL': 8209}, 2021: {'BIAS_UP': 768, 'BIAS_DOWN': 506, 'BIAS_NEUTRAL': 7817}, 2022: {'BIAS_UP': 703, 'BIAS_DOWN': 561, 'BIAS_NEUTRAL': 7857}, 2023: {'BIAS_UP': 561, 'BIAS_DOWN': 112, 'BIAS_NEUTRAL': 4663}}
2026-05-11 09:47:18,088 INFO Regime[4H mode=htf_bias] score means by year: {2016: {'bias_up_score': 0.0595, 'bias_down_score': 0.0699}, 2017: {'bias_up_score': 0.0852, 'bias_down_score': 0.0346}, 2018: {'bias_up_score': 0.0496, 'bias_down_score': 0.0825}, 2019: {'bias_up_score': 0.0469, 'bias_down_score': 0.0528}, 2020: {'bias_up_score': 0.0791, 'bias_down_score': 0.0199}, 2021: {'bias_up_score': 0.0845, 'bias_down_score': 0.0557}, 2022: {'bias_up_score': 0.0771, 'bias_down_score': 0.0615}, 2023: {'bias_up_score': 0.1051, 'bias_down_score': 0.021}}
2026-05-11 09:47:18,157 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
2026-05-11 09:47:18,158 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
2026-05-11 09:47:18,159 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
2026-05-11 09:47:18,160 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 09:47:18,161 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 09:47:18,161 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
2026-05-11 09:47:18,178 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:47:18,182 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:47:18,183 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:47:18,184 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:47:18,184 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:47:18,185 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:47:18,553 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 166, 'BIAS_DOWN': 96, 'BIAS_NEUTRAL': 1319}  ambiguous=536 (total=1581) horizon=84
2026-05-11 09:47:18,556 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected XAUUSD — 1531 samples (group=gold) score_means={'bias_up_score': 0.1084, 'bias_down_score': 0.0627} labels={'BIAS_UP': 166, 'BIAS_DOWN': 96, 'BIAS_NEUTRAL': 1269} clean={'BIAS_UP': 166, 'BIAS_DOWN': 96, 'BIAS_NEUTRAL': 754}
2026-05-11 09:47:18,636 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:18,639 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:18,639 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:18,640 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:18,640 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:18,641 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:47:18,987 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 141, 'BIAS_DOWN': 68, 'BIAS_NEUTRAL': 1282}  ambiguous=504 (total=1491) horizon=84
2026-05-11 09:47:18,990 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURUSD — 1441 samples (group=dollar) score_means={'bias_up_score': 0.0978, 'bias_down_score': 0.0472} labels={'BIAS_UP': 141, 'BIAS_DOWN': 68, 'BIAS_NEUTRAL': 1232} clean={'BIAS_UP': 141, 'BIAS_DOWN': 68, 'BIAS_NEUTRAL': 757}
2026-05-11 09:47:19,067 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:19,069 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:19,070 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:19,070 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:19,070 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:19,071 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:47:19,427 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 173, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1231}  ambiguous=584 (total=1489) horizon=84
2026-05-11 09:47:19,430 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected USDJPY — 1439 samples (group=dollar) score_means={'bias_up_score': 0.1202, 'bias_down_score': 0.0591} labels={'BIAS_UP': 173, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1181} clean={'BIAS_UP': 173, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 621}
2026-05-11 09:47:19,508 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:19,510 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:19,511 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:19,511 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:19,511 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:19,512 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:47:19,873 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 125, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 1364}  ambiguous=540 (total=1494) horizon=84
2026-05-11 09:47:19,876 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0866, 'bias_down_score': 0.0035} labels={'BIAS_UP': 125, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 1314} clean={'BIAS_UP': 125, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 780}
2026-05-11 09:47:19,953 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:19,955 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:19,956 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:19,956 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:19,957 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:19,958 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:47:20,301 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 134, 'BIAS_DOWN': 11, 'BIAS_NEUTRAL': 1349}  ambiguous=512 (total=1494) horizon=84
2026-05-11 09:47:20,304 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0928, 'bias_down_score': 0.0069} labels={'BIAS_UP': 134, 'BIAS_DOWN': 10, 'BIAS_NEUTRAL': 1300} clean={'BIAS_UP': 134, 'BIAS_DOWN': 10, 'BIAS_NEUTRAL': 807}
2026-05-11 09:47:20,382 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:20,384 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:20,385 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:20,385 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:20,385 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:20,386 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:47:20,733 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 93, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 1304}  ambiguous=544 (total=1488) horizon=84
2026-05-11 09:47:20,736 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPUSD — 1438 samples (group=dollar) score_means={'bias_up_score': 0.0647, 'bias_down_score': 0.0633} labels={'BIAS_UP': 93, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 1254} clean={'BIAS_UP': 93, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 738}
2026-05-11 09:47:20,806 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 259, 'BIAS_DOWN': 15, 'BIAS_NEUTRAL': 2614}, 'dollar': {'BIAS_UP': 407, 'BIAS_DOWN': 244, 'BIAS_NEUTRAL': 3667}, 'gold': {'BIAS_UP': 166, 'BIAS_DOWN': 96, 'BIAS_NEUTRAL': 1269}}
2026-05-11 09:47:20,806 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0897, 'bias_down_score': 0.0052}, 'dollar': {'bias_up_score': 0.0943, 'bias_down_score': 0.0565}, 'gold': {'bias_up_score': 0.1084, 'bias_down_score': 0.0627}}
2026-05-11 09:47:20,806 INFO Regime[4H mode=htf_bias] label distribution by year: {2022: {'BIAS_UP': 276, 'BIAS_DOWN': 248, 'BIAS_NEUTRAL': 2877}, 2023: {'BIAS_UP': 556, 'BIAS_DOWN': 107, 'BIAS_NEUTRAL': 4673}}
2026-05-11 09:47:20,806 INFO Regime[4H mode=htf_bias] score means by year: {2022: {'bias_up_score': 0.0812, 'bias_down_score': 0.0729}, 2023: {'bias_up_score': 0.1042, 'bias_down_score': 0.0201}}
2026-05-11 09:47:20,869 INFO Regime phase HTF dataset build fold=train_all: 8.5s (train=68826 val=8737)
2026-05-11 09:47:20,870 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260511_094720
2026-05-11 09:47:20,874 ERROR RegimeClassifier.load failed: RegimeClassifier.load: feature contract mismatch for /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl. saved=['adx_14_base', 'ema_stack_score', 'mtf_1d_adx', 'mtf_1d_ema_stack', 'mtf_1d_atr_ratio', 'efficiency_ratio', 'plus_di', 'minus_di', 'ema_50_slope', 'ema_200_slope', 'ema_50_dist_atr', 'ema_200_dist_atr', 'atr_percentile_500', 'rolling_range_percentile', 'hh_hl_structure', 'lh_ll_structure', 'external_trend_direction', 'external_structure_score', 'internal_structure_state', 'swing_sequence_score', 'bars_since_mss', 'bars_since_last_mss', 'bars_since_last_bos', 'directional_bars_20', 'mss_bull_bars_ago', 'mss_bear_bars_ago', 'symbol_group_code', 'macro_vix_level'] expected=['adx_14_base', 'ema_stack_score', 'mtf_1d_adx', 'mtf_1d_ema_stack', 'mtf_1d_atr_ratio', 'efficiency_ratio', 'plus_di', 'minus_di', 'ema_50_slope', 'ema_200_slope', 'ema_50_dist_atr', 'ema_200_dist_atr', 'atr_percentile_500', 'rolling_range_percentile', 'hh_hl_structure', 'lh_ll_structure', 'external_trend_direction', 'external_structure_score', 'internal_structure_state', 'swing_sequence_score', 'bars_since_mss', 'bars_since_last_mss', 'bars_since_last_bos', 'directional_bars_20', 'candle_body_atr', 'candle_range_atr', 'candle_close_location', 'body_direction_20', 'wick_rejection_20', 'trend_body_pressure_20', 'range_close_position_20', 'breakout_close_strength', 'mss_bull_bars_ago', 'mss_bear_bars_ago', 'symbol_group_code', 'macro_vix_level']. Delete the stale file and retrain regime weights.
2026-05-11 09:47:20,874 WARNING Regime 4H/htf_bias existing weights are unusable (RegimeClassifier.load: feature contract mismatch for /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl. saved=['adx_14_base', 'ema_stack_score', 'mtf_1d_adx', 'mtf_1d_ema_stack', 'mtf_1d_atr_ratio', 'efficiency_ratio', 'plus_di', 'minus_di', 'ema_50_slope', 'ema_200_slope', 'ema_50_dist_atr', 'ema_200_dist_atr', 'atr_percentile_500', 'rolling_range_percentile', 'hh_hl_structure', 'lh_ll_structure', 'external_trend_direction', 'external_structure_score', 'internal_structure_state', 'swing_sequence_score', 'bars_since_mss', 'bars_since_last_mss', 'bars_since_last_bos', 'directional_bars_20', 'mss_bull_bars_ago', 'mss_bear_bars_ago', 'symbol_group_code', 'macro_vix_level'] expected=['adx_14_base', 'ema_stack_score', 'mtf_1d_adx', 'mtf_1d_ema_stack', 'mtf_1d_atr_ratio', 'efficiency_ratio', 'plus_di', 'minus_di', 'ema_50_slope', 'ema_200_slope', 'ema_50_dist_atr', 'ema_200_dist_atr', 'atr_percentile_500', 'rolling_range_percentile', 'hh_hl_structure', 'lh_ll_structure', 'external_trend_direction', 'external_structure_score', 'internal_structure_state', 'swing_sequence_score', 'bars_since_mss', 'bars_since_last_mss', 'bars_since_last_bos', 'directional_bars_20', 'candle_body_atr', 'candle_range_atr', 'candle_close_location', 'body_direction_20', 'wick_rejection_20', 'trend_body_pressure_20', 'range_close_position_20', 'breakout_close_strength', 'mss_bull_bars_ago', 'mss_bear_bars_ago', 'symbol_group_code', 'macro_vix_level']. Delete the stale file and retrain regime weights.); cold starting
2026-05-11 09:47:20,881 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=68826 val=8737 train_labels={'BIAS_UP': 4934, 'BIAS_DOWN': 3526, 'BIAS_NEUTRAL': 60366} val_labels={'BIAS_UP': 832, 'BIAS_DOWN': 355, 'BIAS_NEUTRAL': 7550}
2026-05-11 09:47:21,092 INFO RegimeClassifier[mode=htf_bias]: cold start HTF score head
2026-05-11 09:47:21,093 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-11 09:47:21,093 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 12.949, 'bias_down_score': 18.52}
2026-05-11 09:47:21,097 INFO RegimeClassifier[mode=htf_bias]: HTF balanced sampler — dir=8460 neutral=60366 dir_weight=5 => dir_frac_per_epoch≈41.2%
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../models/regime_classifier.py:2519: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  scheduler.step()
2026-05-11 09:47:24,823 INFO Regime HTF score epoch  1/50 — tr=12.7208 va=2.7810 acc=0.864 bal=0.333 threshold=0.60 margin=0.10 recall={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 1.0} precision={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 0.864}
2026-05-11 09:47:26,197 INFO Regime HTF score epoch  2/50 — tr=12.2639 va=2.6906 bal=0.333
2026-05-11 09:47:27,599 INFO Regime HTF score epoch  3/50 — tr=11.5391 va=2.4615 bal=0.333
2026-05-11 09:47:28,968 INFO Regime HTF score epoch  4/50 — tr=10.0482 va=2.0737 bal=0.333
2026-05-11 09:47:30,358 INFO Regime HTF score epoch  5/50 — tr=8.2374 va=1.5812 acc=0.836 bal=0.404 threshold=0.40 margin=0.15 recall={'BIAS_UP': 0.145, 'BIAS_DOWN': 0.121, 'BIAS_NEUTRAL': 0.945} precision={'BIAS_UP': 0.289, 'BIAS_DOWN': 0.272, 'BIAS_NEUTRAL': 0.875}
2026-05-11 09:47:31,768 INFO Regime HTF score epoch  6/50 — tr=6.1413 va=1.1751 bal=0.407
2026-05-11 09:47:33,219 INFO Regime HTF score epoch  7/50 — tr=4.7051 va=0.9518 bal=0.397
2026-05-11 09:47:34,601 INFO Regime HTF score epoch  8/50 — tr=3.6129 va=0.8048 bal=0.409
2026-05-11 09:47:35,999 INFO Regime HTF score epoch  9/50 — tr=2.7256 va=0.7174 bal=0.351
2026-05-11 09:47:37,368 INFO Regime HTF score epoch 10/50 — tr=2.0627 va=0.6806 acc=0.848 bal=0.388 threshold=0.97 margin=0.15 recall={'BIAS_UP': 0.04, 'BIAS_DOWN': 0.155, 'BIAS_NEUTRAL': 0.97} precision={'BIAS_UP': 0.359, 'BIAS_DOWN': 0.249, 'BIAS_NEUTRAL': 0.87}
2026-05-11 09:47:38,766 INFO Regime HTF score epoch 11/50 — tr=1.6342 va=0.6853 bal=0.421
2026-05-11 09:47:40,138 INFO Regime HTF score epoch 12/50 — tr=1.3778 va=0.7094 bal=0.380
2026-05-11 09:47:41,583 INFO Regime HTF score epoch 13/50 — tr=1.2718 va=0.7209 bal=0.404
2026-05-11 09:47:41,583 INFO Regime HTF score early stop at epoch 13
2026-05-11 09:47:42,827 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.400 margin=0.150 precision={'BIAS_UP': 0.289, 'BIAS_DOWN': 0.272, 'BIAS_NEUTRAL': 0.875} recall={'BIAS_UP': 0.145, 'BIAS_DOWN': 0.121, 'BIAS_NEUTRAL': 0.945} f1={'BIAS_UP': 0.193, 'BIAS_DOWN': 0.168, 'BIAS_NEUTRAL': 0.909} confusion=[[121, 0, 711], [0, 43, 312], [298, 115, 7137]] score_mae={'bias_up_score': 0.1684, 'bias_down_score': 0.101} pred_share={'BIAS_UP': 0.048, 'BIAS_DOWN': 0.0181, 'BIAS_NEUTRAL': 0.934}
2026-05-11 09:47:42,828 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.289, 'BIAS_DOWN': 0.272, 'BIAS_NEUTRAL': 0.875} min_precision=0.500 recall={'BIAS_UP': 0.145, 'BIAS_DOWN': 0.121, 'BIAS_NEUTRAL': 0.945} min_recall=0.100 f1={'BIAS_UP': 0.193, 'BIAS_DOWN': 0.168, 'BIAS_NEUTRAL': 0.909} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_UP', 'BIAS_DOWN'] weak_recall=[] weak_f1=[] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-11 09:47:42,832 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 09:47:42,832 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 09:47:42,833 INFO Regime phase HTF train fold=train_all: 22.0s
2026-05-11 09:47:42,944 INFO Regime HTF complete fold=train_all: acc=0.836 bal=0.404 train=68826 val=8737 per_class={'BIAS_UP': 0.145, 'BIAS_DOWN': 0.121, 'BIAS_NEUTRAL': 0.945} precision={'BIAS_UP': 0.289, 'BIAS_DOWN': 0.272, 'BIAS_NEUTRAL': 0.875} threshold=0.400 margin=0.150
2026-05-11 09:47:42,946 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 09:47:43,132 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 546, 'BIAS_DOWN': 754, 'BIAS_NEUTRAL': 10102}  ambiguous=3944 (total=11402) horizon=84
2026-05-11 09:47:43,149 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on GBPUSD 4H:
{'BIAS_UP': 4.403225806451613, 'BIAS_DOWN': 5.755725190839694, 'BIAS_NEUTRAL': 39.4609375}
2026-05-11 09:47:43,152 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (all labels):
{'BIAS_UP': {'n': 546, 'mean': 0.00040267113448389793, 'mean_over_std': 0.18208067865763405}, 'BIAS_DOWN': {'n': 754, 'mean': -0.00047099607164125245, 'mean_over_std': -0.19477795734555267}, 'BIAS_NEUTRAL': {'n': 10101, 'mean': 2.6464098295242517e-06, 'mean_over_std': 0.0010012545608535832}}
2026-05-11 09:47:43,153 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 546, 'mean': 0.00040267113448389793, 'mean_over_std': 0.18208067865763405}, 'BIAS_DOWN': {'n': 754, 'mean': -0.00047099607164125245, 'mean_over_std': -0.19477795734555267}, 'BIAS_NEUTRAL': {'n': 6158, 'mean': 2.1496848003307296e-05, 'mean_over_std': 0.009079001472003705}}
2026-05-11 09:47:43,157 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-11 09:47:43,159 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
2026-05-11 09:47:43,161 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
2026-05-11 09:47:43,163 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
2026-05-11 09:47:43,165 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
2026-05-11 09:47:43,167 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
2026-05-11 09:47:43,169 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 09:47:43,189 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:47:43,197 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:47:43,200 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:47:43,201 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:47:43,201 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:47:43,208 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:47:44,098 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected XAUUSD — 44227 samples (group=gold) score_means={'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}
2026-05-11 09:47:44,215 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:44,217 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:44,218 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:44,218 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:44,219 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:44,221 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:47:45,041 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURUSD — 43689 samples (group=dollar) score_means={'trend_score': 0.49, 'range_score': 0.236, 'chop_score': 0.4622, 'volatility_percentile': 0.3812, 'consolidation_score': 0.1862}
2026-05-11 09:47:45,157 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:45,159 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:45,160 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:45,160 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:45,161 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:45,163 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:47:45,992 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected USDJPY — 43686 samples (group=dollar) score_means={'trend_score': 0.4949, 'range_score': 0.2312, 'chop_score': 0.4613, 'volatility_percentile': 0.38, 'consolidation_score': 0.1931}
2026-05-11 09:47:46,108 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:46,110 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:46,111 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:46,112 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:46,112 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:46,114 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:47:46,945 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURJPY — 43688 samples (group=cross) score_means={'trend_score': 0.4888, 'range_score': 0.2345, 'chop_score': 0.4662, 'volatility_percentile': 0.3831, 'consolidation_score': 0.1882}
2026-05-11 09:47:47,062 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:47,064 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:47,065 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:47,065 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:47,066 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:47,068 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:47:47,896 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPJPY — 43678 samples (group=cross) score_means={'trend_score': 0.4878, 'range_score': 0.2348, 'chop_score': 0.4673, 'volatility_percentile': 0.3839, 'consolidation_score': 0.1904}
2026-05-11 09:47:48,013 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:48,016 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:48,016 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:48,017 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:48,017 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:48,020 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:47:48,834 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPUSD — 43676 samples (group=dollar) score_means={'trend_score': 0.4923, 'range_score': 0.2335, 'chop_score': 0.4611, 'volatility_percentile': 0.3832, 'consolidation_score': 0.1856}
2026-05-11 09:47:48,957 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2346, 'chop_score': 0.4667, 'volatility_percentile': 0.3835, 'consolidation_score': 0.1893}, 'dollar': {'trend_score': 0.4924, 'range_score': 0.2335, 'chop_score': 0.4615, 'volatility_percentile': 0.3815, 'consolidation_score': 0.1883}, 'gold': {'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}}
2026-05-11 09:47:48,957 INFO Regime[1H mode=ltf_behaviour] score means by year: {2016: {'trend_score': 0.495, 'range_score': 0.233, 'chop_score': 0.4607, 'volatility_percentile': 0.3776, 'consolidation_score': 0.192}, 2017: {'trend_score': 0.5017, 'range_score': 0.2328, 'chop_score': 0.4574, 'volatility_percentile': 0.3689, 'consolidation_score': 0.1942}, 2018: {'trend_score': 0.4886, 'range_score': 0.237, 'chop_score': 0.4655, 'volatility_percentile': 0.3947, 'consolidation_score': 0.1785}, 2019: {'trend_score': 0.4817, 'range_score': 0.236, 'chop_score': 0.4691, 'volatility_percentile': 0.3725, 'consolidation_score': 0.198}, 2020: {'trend_score': 0.4874, 'range_score': 0.2311, 'chop_score': 0.4674, 'volatility_percentile': 0.3848, 'consolidation_score': 0.1919}, 2021: {'trend_score': 0.4872, 'range_score': 0.2372, 'chop_score': 0.4672, 'volatility_percentile': 0.377, 'consolidation_score': 0.192}, 2022: {'trend_score': 0.4893, 'range_score': 0.2357, 'chop_score': 0.4647, 'volatility_percentile': 0.3962, 'consolidation_score': 0.1784}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 09:47:49,075 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
2026-05-11 09:47:49,076 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 09:47:49,078 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
2026-05-11 09:47:49,079 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 09:47:49,080 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
2026-05-11 09:47:49,082 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
2026-05-11 09:47:49,091 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:47:49,094 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:47:49,095 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:47:49,096 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:47:49,096 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:47:49,098 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:47:49,360 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected XAUUSD — 5117 samples (group=gold) score_means={'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}
2026-05-11 09:47:49,484 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:49,488 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:49,489 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:49,489 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:49,490 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:49,491 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:47:49,733 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURUSD — 5053 samples (group=dollar) score_means={'trend_score': 0.4883, 'range_score': 0.2413, 'chop_score': 0.463, 'volatility_percentile': 0.3837, 'consolidation_score': 0.1793}
2026-05-11 09:47:49,848 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:49,850 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:49,851 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:49,851 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:49,852 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:49,853 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:47:50,085 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected USDJPY — 5046 samples (group=dollar) score_means={'trend_score': 0.5085, 'range_score': 0.2259, 'chop_score': 0.4506, 'volatility_percentile': 0.3809, 'consolidation_score': 0.1888}
2026-05-11 09:47:50,204 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:50,206 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:50,207 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:50,208 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:50,208 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:50,211 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:47:50,454 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURJPY — 5053 samples (group=cross) score_means={'trend_score': 0.5023, 'range_score': 0.2261, 'chop_score': 0.4524, 'volatility_percentile': 0.3837, 'consolidation_score': 0.18}
2026-05-11 09:47:50,578 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:50,580 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:50,581 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:50,582 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:50,582 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:50,584 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:47:50,829 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPJPY — 5043 samples (group=cross) score_means={'trend_score': 0.4743, 'range_score': 0.2372, 'chop_score': 0.4716, 'volatility_percentile': 0.3777, 'consolidation_score': 0.1917}
2026-05-11 09:47:50,948 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:50,951 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:50,951 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:50,952 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:50,952 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:47:50,955 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:47:51,212 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPUSD — 5040 samples (group=dollar) score_means={'trend_score': 0.4843, 'range_score': 0.2371, 'chop_score': 0.4614, 'volatility_percentile': 0.377, 'consolidation_score': 0.1835}
2026-05-11 09:47:51,338 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2317, 'chop_score': 0.462, 'volatility_percentile': 0.3807, 'consolidation_score': 0.1858}, 'dollar': {'trend_score': 0.4937, 'range_score': 0.2348, 'chop_score': 0.4583, 'volatility_percentile': 0.3806, 'consolidation_score': 0.1839}, 'gold': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}}
2026-05-11 09:47:51,338 INFO Regime[1H mode=ltf_behaviour] score means by year: {2022: {'trend_score': 0.4905, 'range_score': 0.2364, 'chop_score': 0.4625, 'volatility_percentile': 0.385, 'consolidation_score': 0.1797}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 09:47:51,455 INFO Regime phase LTF dataset build fold=train_all: 8.3s (train=262644 val=30352)
2026-05-11 09:47:51,456 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260511_094751
2026-05-11 09:47:51,460 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=31, n_classes=5)
2026-05-11 09:47:51,460 INFO Regime 1H/ltf_behaviour warm start enabled from existing weights
2026-05-11 09:47:51,484 INFO RegimeClassifier[mode=ltf_behaviour]: warm start score head
2026-05-11 09:47:51,484 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-11 09:47:52,029 INFO Regime score epoch  1/50 — tr=0.0038 va=0.0010 mae={'trend_score': 0.0205, 'range_score': 0.0347, 'chop_score': 0.0226, 'volatility_percentile': 0.0156, 'consolidation_score': 0.0221}
2026-05-11 09:47:52,543 INFO Regime score epoch  2/50 — tr=0.0038 va=0.0010
2026-05-11 09:47:53,075 INFO Regime score epoch  3/50 — tr=0.0038 va=0.0010
2026-05-11 09:47:53,593 INFO Regime score epoch  4/50 — tr=0.0038 va=0.0010
2026-05-11 09:47:54,139 INFO Regime score epoch  5/50 — tr=0.0038 va=0.0010 mae={'trend_score': 0.0203, 'range_score': 0.0343, 'chop_score': 0.0223, 'volatility_percentile': 0.0154, 'consolidation_score': 0.0216}
2026-05-11 09:47:54,665 INFO Regime score epoch  6/50 — tr=0.0038 va=0.0010
2026-05-11 09:47:55,176 INFO Regime score epoch  7/50 — tr=0.0038 va=0.0010
2026-05-11 09:47:55,695 INFO Regime score epoch  8/50 — tr=0.0037 va=0.0010
2026-05-11 09:47:56,226 INFO Regime score epoch  9/50 — tr=0.0037 va=0.0010
2026-05-11 09:47:56,740 INFO Regime score epoch 10/50 — tr=0.0037 va=0.0009 mae={'trend_score': 0.0196, 'range_score': 0.0335, 'chop_score': 0.0217, 'volatility_percentile': 0.0152, 'consolidation_score': 0.0211}
2026-05-11 09:47:57,245 INFO Regime score epoch 11/50 — tr=0.0037 va=0.0009
2026-05-11 09:47:57,765 INFO Regime score epoch 12/50 — tr=0.0037 va=0.0009
2026-05-11 09:47:58,293 INFO Regime score epoch 13/50 — tr=0.0036 va=0.0009
2026-05-11 09:47:58,822 INFO Regime score epoch 14/50 — tr=0.0036 va=0.0009
2026-05-11 09:47:59,328 INFO Regime score epoch 15/50 — tr=0.0036 va=0.0009 mae={'trend_score': 0.019, 'range_score': 0.033, 'chop_score': 0.021, 'volatility_percentile': 0.0146, 'consolidation_score': 0.0204}
2026-05-11 09:47:59,834 INFO Regime score epoch 16/50 — tr=0.0036 va=0.0009
2026-05-11 09:48:00,348 INFO Regime score epoch 17/50 — tr=0.0036 va=0.0009
2026-05-11 09:48:00,856 INFO Regime score epoch 18/50 — tr=0.0036 va=0.0009
2026-05-11 09:48:01,413 INFO Regime score epoch 19/50 — tr=0.0035 va=0.0009
2026-05-11 09:48:01,971 INFO Regime score epoch 20/50 — tr=0.0035 va=0.0009 mae={'trend_score': 0.0184, 'range_score': 0.0324, 'chop_score': 0.0205, 'volatility_percentile': 0.0146, 'consolidation_score': 0.0202}
2026-05-11 09:48:02,538 INFO Regime score epoch 21/50 — tr=0.0035 va=0.0009
2026-05-11 09:48:03,052 INFO Regime score epoch 22/50 — tr=0.0035 va=0.0009
2026-05-11 09:48:03,564 INFO Regime score epoch 23/50 — tr=0.0035 va=0.0009
2026-05-11 09:48:04,084 INFO Regime score epoch 24/50 — tr=0.0035 va=0.0009
2026-05-11 09:48:04,609 INFO Regime score epoch 25/50 — tr=0.0035 va=0.0008 mae={'trend_score': 0.018, 'range_score': 0.0322, 'chop_score': 0.0201, 'volatility_percentile': 0.0144, 'consolidation_score': 0.02}
2026-05-11 09:48:05,123 INFO Regime score epoch 26/50 — tr=0.0035 va=0.0008
2026-05-11 09:48:05,641 INFO Regime score epoch 27/50 — tr=0.0035 va=0.0008
2026-05-11 09:48:06,187 INFO Regime score epoch 28/50 — tr=0.0035 va=0.0008
2026-05-11 09:48:06,704 INFO Regime score epoch 29/50 — tr=0.0034 va=0.0008
2026-05-11 09:48:07,238 INFO Regime score epoch 30/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.0178, 'range_score': 0.032, 'chop_score': 0.02, 'volatility_percentile': 0.0143, 'consolidation_score': 0.0195}
2026-05-11 09:48:07,767 INFO Regime score epoch 31/50 — tr=0.0034 va=0.0008
2026-05-11 09:48:08,283 INFO Regime score epoch 32/50 — tr=0.0034 va=0.0008
2026-05-11 09:48:08,834 INFO Regime score epoch 33/50 — tr=0.0034 va=0.0008
2026-05-11 09:48:09,361 INFO Regime score epoch 34/50 — tr=0.0034 va=0.0008
2026-05-11 09:48:09,892 INFO Regime score epoch 35/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.0176, 'range_score': 0.0315, 'chop_score': 0.0199, 'volatility_percentile': 0.0141, 'consolidation_score': 0.0196}
2026-05-11 09:48:10,404 INFO Regime score epoch 36/50 — tr=0.0034 va=0.0008
2026-05-11 09:48:10,932 INFO Regime score epoch 37/50 — tr=0.0034 va=0.0008
2026-05-11 09:48:11,495 INFO Regime score epoch 38/50 — tr=0.0034 va=0.0008
2026-05-11 09:48:12,012 INFO Regime score epoch 39/50 — tr=0.0034 va=0.0008
2026-05-11 09:48:12,526 INFO Regime score epoch 40/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.0179, 'range_score': 0.0316, 'chop_score': 0.0195, 'volatility_percentile': 0.0144, 'consolidation_score': 0.0196}
2026-05-11 09:48:13,052 INFO Regime score epoch 41/50 — tr=0.0034 va=0.0008
2026-05-11 09:48:13,560 INFO Regime score epoch 42/50 — tr=0.0034 va=0.0008
2026-05-11 09:48:14,073 INFO Regime score epoch 43/50 — tr=0.0034 va=0.0008
2026-05-11 09:48:14,591 INFO Regime score epoch 44/50 — tr=0.0034 va=0.0008
2026-05-11 09:48:15,121 INFO Regime score epoch 45/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.0172, 'range_score': 0.0317, 'chop_score': 0.0198, 'volatility_percentile': 0.014, 'consolidation_score': 0.0196}
2026-05-11 09:48:15,645 INFO Regime score epoch 46/50 — tr=0.0034 va=0.0008
2026-05-11 09:48:16,184 INFO Regime score epoch 47/50 — tr=0.0034 va=0.0008
2026-05-11 09:48:16,691 INFO Regime score epoch 48/50 — tr=0.0034 va=0.0008
2026-05-11 09:48:17,202 INFO Regime score epoch 49/50 — tr=0.0034 va=0.0008
2026-05-11 09:48:17,717 INFO Regime score epoch 50/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.0173, 'range_score': 0.0317, 'chop_score': 0.0196, 'volatility_percentile': 0.0146, 'consolidation_score': 0.0203}
2026-05-11 09:48:17,739 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0174, 'range_score': 0.0316, 'chop_score': 0.0194, 'volatility_percentile': 0.0141, 'consolidation_score': 0.0195} mse={'trend_score': 0.00053, 'range_score': 0.00166, 'chop_score': 0.00061, 'volatility_percentile': 0.00036, 'consolidation_score': 0.00087} corr={'trend_score': 0.9947, 'range_score': 0.9597, 'chop_score': 0.992, 'volatility_percentile': 0.9963, 'consolidation_score': 0.9909} pred_std={'trend_score': 0.221, 'range_score': 0.1315, 'chop_score': 0.1809, 'volatility_percentile': 0.218, 'consolidation_score': 0.2148} target_std={'trend_score': 0.2219, 'range_score': 0.1433, 'chop_score': 0.1888, 'volatility_percentile': 0.2165, 'consolidation_score': 0.2179}
2026-05-11 09:48:18,079 INFO RegimeClassifier[mode=ltf_behaviour] train group diagnostics={'dollar': {'n': 16632, 'score_mae': {'trend_score': 0.0169, 'range_score': 0.0314, 'chop_score': 0.0194, 'volatility_percentile': 0.0137, 'consolidation_score': 0.0199}, 'target_score_mean': {'trend_score': 0.491, 'range_score': 0.2347, 'chop_score': 0.4613, 'volatility_percentile': 0.3824, 'consolidation_score': 0.186}, 'pred_score_mean': {'trend_score': 0.4937, 'range_score': 0.2327, 'chop_score': 0.4584, 'volatility_percentile': 0.3795, 'consolidation_score': 0.1839}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3602, 52, 0, 3, 0, 0, 122], [5, 98, 0, 0, 0, 3, 4], [0, 0, 193, 11, 39, 0, 217], [2, 0, 5, 551, 29, 0, 102], [0, 0, 40, 21, 3044, 0, 211], [0, 19, 0, 0, 7, 57, 45], [169, 12, 70, 42, 46, 4, 7807]]}, 'gold': {'n': 8420, 'score_mae': {'trend_score': 0.0168, 'range_score': 0.032, 'chop_score': 0.0197, 'volatility_percentile': 0.0144, 'consolidation_score': 0.0202}, 'target_score_mean': {'trend_score': 0.4883, 'range_score': 0.2355, 'chop_score': 0.4652, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}, 'pred_score_mean': {'trend_score': 0.4918, 'range_score': 0.2332, 'chop_score': 0.4617, 'volatility_percentile': 0.3736, 'consolidation_score': 0.1895}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1802, 28, 0, 0, 0, 0, 55], [4, 51, 0, 0, 0, 0, 1], [0, 0, 91, 11, 24, 0, 118], [1, 0, 2, 336, 16, 0, 61], [0, 0, 20, 26, 1541, 0, 117], [0, 14, 0, 0, 5, 40, 22], [77, 4, 51, 19, 32, 0, 3851]]}, 'jpy': {'n': 24948, 'score_mae': {'trend_score': 0.0169, 'range_score': 0.0315, 'chop_score': 0.0193, 'volatility_percentile': 0.0145, 'consolidation_score': 0.0197}, 'target_score_mean': {'trend_score': 0.4904, 'range_score': 0.2338, 'chop_score': 0.4653, 'volatility_percentile': 0.3826, 'consolidation_score': 0.1905}, 'pred_score_mean': {'trend_score': 0.4929, 'range_score': 0.2316, 'chop_score': 0.4618, 'volatility_percentile': 0.3793, 'consolidation_score': 0.1876}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[5359, 110, 0, 1, 0, 0, 176], [10, 170, 0, 0, 0, 1, 6], [0, 0, 235, 18, 74, 0, 320], [4, 0, 3, 1075, 60, 0, 172], [0, 0, 42, 70, 4644, 0, 359], [0, 33, 0, 0, 16, 90, 84], [237, 13, 97, 80, 101, 2, 11286]]}}
2026-05-11 09:48:18,254 INFO RegimeClassifier[mode=ltf_behaviour] validation group diagnostics={'dollar': {'n': 10093, 'score_mae': {'trend_score': 0.0177, 'range_score': 0.0322, 'chop_score': 0.0195, 'volatility_percentile': 0.0137, 'consolidation_score': 0.0191}, 'target_score_mean': {'trend_score': 0.4863, 'range_score': 0.2392, 'chop_score': 0.4622, 'volatility_percentile': 0.3804, 'consolidation_score': 0.1814}, 'pred_score_mean': {'trend_score': 0.4891, 'range_score': 0.2356, 'chop_score': 0.4603, 'volatility_percentile': 0.3773, 'consolidation_score': 0.1801}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[2304, 21, 0, 1, 0, 0, 86], [4, 45, 0, 0, 0, 3, 1], [0, 0, 114, 7, 39, 0, 156], [0, 0, 1, 339, 21, 0, 62], [0, 0, 25, 29, 1893, 0, 103], [0, 12, 0, 0, 2, 36, 27], [76, 6, 37, 42, 39, 3, 4559]]}, 'gold': {'n': 5117, 'score_mae': {'trend_score': 0.0165, 'range_score': 0.0307, 'chop_score': 0.0193, 'volatility_percentile': 0.0143, 'consolidation_score': 0.0199}, 'target_score_mean': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}, 'pred_score_mean': {'trend_score': 0.4994, 'range_score': 0.2302, 'chop_score': 0.4544, 'volatility_percentile': 0.3782, 'consolidation_score': 0.1807}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1113, 12, 0, 0, 0, 0, 42], [3, 30, 0, 0, 0, 1, 1], [0, 0, 66, 3, 12, 0, 90], [0, 0, 2, 225, 7, 0, 21], [0, 0, 13, 12, 791, 0, 71], [0, 6, 0, 0, 4, 23, 17], [57, 2, 30, 27, 22, 0, 2414]]}, 'jpy': {'n': 15142, 'score_mae': {'trend_score': 0.0176, 'range_score': 0.0314, 'chop_score': 0.0193, 'volatility_percentile': 0.0144, 'consolidation_score': 0.0196}, 'target_score_mean': {'trend_score': 0.495, 'range_score': 0.2297, 'chop_score': 0.4582, 'volatility_percentile': 0.3808, 'consolidation_score': 0.1868}, 'pred_score_mean': {'trend_score': 0.4966, 'range_score': 0.2271, 'chop_score': 0.4561, 'volatility_percentile': 0.3776, 'consolidation_score': 0.1841}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3354, 49, 0, 1, 0, 0, 114], [7, 100, 0, 0, 0, 3, 5], [0, 0, 148, 14, 43, 0, 179], [4, 0, 3, 688, 29, 0, 103], [0, 0, 35, 32, 2551, 0, 199], [0, 18, 0, 0, 7, 54, 43], [126, 10, 65, 42, 62, 3, 7051]]}}
2026-05-11 09:48:18,260 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-11 09:48:18,260 INFO Regime phase LTF train fold=train_all: 26.8s
2026-05-11 09:48:18,377 INFO Regime LTF complete fold=train_all: score_accuracy=0.980, train=262644 val=30352 mae={'trend_score': 0.0174, 'range_score': 0.0316, 'chop_score': 0.0194, 'volatility_percentile': 0.0141, 'consolidation_score': 0.0195}
2026-05-11 09:48:18,380 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 09:48:18,743 INFO Regime[1H mode=ltf_behaviour fold=latest] LTF score diagnostics on GBPUSD:
{'trend_score': {'mean': 0.4921, 'q10': 0.1918, 'q50': 0.4877, 'q90': 0.8002}, 'range_score': {'mean': 0.2337, 'q10': 0.0513, 'q50': 0.2125, 'q90': 0.4349}, 'chop_score': {'mean': 0.4612, 'q10': 0.2148, 'q50': 0.4503, 'q90': 0.7255}, 'volatility_percentile': {'mean': 0.3832, 'q10': 0.1018, 'q50': 0.3704, 'q90': 0.6823}, 'consolidation_score': {'mean': 0.1854, 'q10': 0.0, 'q50': 0.1185, 'q90': 0.506}}
2026-05-11 09:48:18,747 INFO Regime retrain total: 67.0s (370559 train+val samples)
2026-05-11 09:48:18,751 INFO Retrain complete. Total wall-clock: 67.0s
2026-05-11 09:48:19,835 INFO Model regime: SUCCESS
2026-05-11 09:48:19,835 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:48:19,835 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 09:48:19,835 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-11 09:48:19,835 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-05-11 09:48:19,836 INFO   [DORMANT] rl_ppo — RL_ENABLED=false, skipping
2026-05-11 09:48:19,836 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-05-11 09:48:19,836 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer']
2026-05-11 09:48:19,861 INFO Saved 90 retrain records to metrics/

=== TRAINING COMPLETE ===
  gru: SUCCESS
  regime: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Clean Quality/RL source: Backtest on train window ===
  START Train-window backtest for Quality/RL labels
2026-05-11 09:48:20,617 INFO === STEP 6: BACKTEST (train) ===
2026-05-11 09:48:20,619 INFO BT_WINDOW=train — train-window backtest: 2016-01-04 → 2023-08-04 (clean Quality/RL labels)
2026-05-11 09:48:20,619 INFO Cleared existing journal for fresh train run
2026-05-11 09:48:20,620 INFO ================================================================
  ROUND 0 / 3
================================================================
2026-05-11 09:48:20,620 INFO Round 0 — running backtest: 2016-01-04 → 2023-08-04 (ml_trader, shared ML cache)
2026-05-11 09:52:19,203 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for EURJPY with 2
2026-05-11 09:52:19,232 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for EURJPY with 0.3333333333333333
2026-05-11 09:52:19,387 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for EURUSD with 2
2026-05-11 09:52:19,417 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for EURUSD with 0.3333333333333333
2026-05-11 09:52:19,684 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for USDJPY with 2
2026-05-11 09:52:19,705 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for USDJPY with 0.3333333333333333
2026-05-11 09:52:19,839 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for EURJPY with 2
2026-05-11 09:52:19,855 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for EURJPY with 0.25
2026-05-11 09:52:19,887 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for EURJPY: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 09:52:20,141 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for EURUSD with 2
2026-05-11 09:52:20,166 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for EURUSD with 0.25
2026-05-11 09:52:20,206 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for EURUSD: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 09:52:20,402 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for USDJPY with 2
2026-05-11 09:52:20,419 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for USDJPY with 0.25
2026-05-11 09:52:20,448 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for USDJPY: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 09:52:23,781 WARNING ML cache score overlay filled 4 warmup/alignment gaps for EURJPY
2026-05-11 09:52:26,527 WARNING ML cache score overlay filled 4 warmup/alignment gaps for USDJPY
2026-05-11 09:52:26,878 WARNING ML cache score overlay filled 4 warmup/alignment gaps for EURUSD
2026-05-11 09:52:33,532 WARNING _build_sequence_df: HTF frame 5M filled 357 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:812: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 09:52:35,578 WARNING _build_sequence_df: HTF frame 5M filled 357 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:814: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:818: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:822: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:826: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:830: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:834: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:836: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 09:52:39,515 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:812: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 09:52:39,603 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:814: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 09:52:39,643 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
2026-05-11 09:52:39,649 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:812: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:812: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 09:52:39,680 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:818: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 09:52:39,713 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 09:52:39,746 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:822: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 09:52:39,773 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:826: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 09:52:39,826 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
2026-05-11 09:52:39,831 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
2026-05-11 09:52:39,832 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:814: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:814: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 09:52:39,866 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:830: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 09:52:39,905 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
2026-05-11 09:52:39,922 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:834: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:818: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 09:52:39,939 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:818: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 09:52:39,958 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:836: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 09:52:39,992 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 09:52:40,010 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 09:52:40,064 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:822: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 09:52:40,118 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
2026-05-11 09:52:40,155 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:826: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:822: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 09:52:40,221 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
2026-05-11 09:52:40,232 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 09:52:40,237 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:826: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 09:52:40,311 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
2026-05-11 09:52:40,344 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:830: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 09:52:40,417 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
2026-05-11 09:52:40,427 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:834: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:830: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 09:52:40,490 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
2026-05-11 09:52:40,494 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:834: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:836: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 09:52:40,543 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:836: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 09:52:40,756 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 09:52:40,800 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 09:52:57,422 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for GBPJPY with 2
2026-05-11 09:52:57,437 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for GBPJPY with 0.3333333333333333
2026-05-11 09:52:57,573 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for GBPUSD with 2
2026-05-11 09:52:57,589 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for GBPUSD with 0.3333333333333333
2026-05-11 09:52:57,747 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for GBPJPY with 2
2026-05-11 09:52:57,771 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for GBPJPY with 0.25
2026-05-11 09:52:57,797 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for GBPJPY: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 09:52:57,965 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for GBPUSD with 2
2026-05-11 09:52:57,979 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for GBPUSD with 0.25
2026-05-11 09:52:57,995 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for GBPUSD: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 09:52:58,431 WARNING ML cache score overlay filled 4 warmup/alignment gaps for GBPJPY
2026-05-11 09:52:59,512 WARNING ML cache score overlay filled 4 warmup/alignment gaps for GBPUSD
2026-05-11 09:53:07,351 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:812: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 09:53:07,399 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:812: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 09:53:07,436 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:814: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 09:53:07,467 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:818: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 09:53:07,500 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 09:53:07,521 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:814: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 09:53:07,540 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:822: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 09:53:07,563 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:826: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 09:53:07,579 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:818: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 09:53:07,599 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 09:53:07,631 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 09:53:07,635 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:830: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 09:53:07,653 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:834: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 09:53:07,682 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:822: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 09:53:07,686 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:836: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 09:53:07,701 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:826: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 09:53:07,722 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 09:53:07,747 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:830: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 09:53:07,781 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:834: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 09:53:07,784 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 09:53:07,800 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:836: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 09:53:07,840 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260511_094822.json
Trader                                   Trades      WR    PF*  Return*    ExpR   TP1%   TP2%     DD*  Sharpe*  Sortino     IC  Verdict
-----------------------------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)                9  44.4%   1.44    2.2%   0.246 44.4% 11.1%   3.8%     2.49     0.25  0.000     FAIL
  FAILED rules: min_trades, t_stat_above_1_5
  monthly R: 2018-05=-1.00  2019-05=-1.00  2020-07=-1.00  2021-10=-1.00  2022-04=+2.88  2022-06=-1.00
  MonteCarlo P95 DD=4.0%  P10 equity=10,221  t=0.47 (p=0.651)  Sharpe CI=[2.49, 2.49]  streak=4
  gate_diagnostics: bars=1049680 no_signal=766634 quality_block=0 session_skip=283037 density=0 pm_reject=0 daily_skip=0 cooldown=0 daily_halt_events=0 enforce_daily_halt=False
  no_signal_reasons: no_trade_uncertain=288334, weak_gru_direction=199572, gru_expected_r_below_threshold=142644, no_trade_chop=69436, no_trade_extreme_vol=56844, htf_low_regime_confidence=7129

Calibration Summary:
  all          [N/A] Insufficient data: 9 samples
  ml_trader    [N/A] Insufficient data: 9 samples
2026-05-11 09:56:00,153 INFO Round 0 backtest — 9 trades | avg WR=44.4% | avg PF=1.44 | avg Sharpe=2.49
2026-05-11 09:56:00,154 INFO   ml_trader: 9 trades | WR=44.4% | fixed PF=1.44 | Return=2.2% | ExpR=0.246 | DD=3.8% | Sharpe=2.49
2026-05-11 09:56:00,154 INFO   ml_trader gate_diagnostics: bars=1049680 no_signal=766634 quality_block=0 session_skip=283037 density=0 pm_reject=0
2026-05-11 09:56:00,154 INFO   ml_trader no_signal_reasons: {'no_trade_uncertain': 288334, 'gru_expected_r_below_threshold': 142644, 'weak_gru_direction': 199572, 'no_trade_extreme_vol': 56844, 'no_trade_chop': 69436, 'htf_low_regime_confidence': 7129, 'tradeability_direction_conflict': 1726, 'wait_pullback': 746, 'trend_structure_missing': 203}
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_0.json
INFO  Total trades: 9
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (9 rows)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  GBPJPY          1 trades     1 days   1.00/day
  GBPUSD          3 trades     3 days   1.00/day
  USDJPY          3 trades     2 days   1.50/day
  XAUUSD          2 trades     1 days   2.00/day  [OVERTRADE]
  ⚠  XAUUSD: 2.00/day (>1.5)

──────────────────────────────────────────────────────────────
CHECK 2: REGIME DISTRIBUTION  (% of trades, WR, avgEV)
──────────────────────────────────────────────────────────────
  BIAS_DOWN             2 trades   22.2%  WR=0.0%  avgEV=0.000
  BIAS_UP               7 trades   77.8%  WR=57.1%  avgEV=0.000
  ⚠  BIAS_UP = 78% of trades — regime collapse?
  ⚠  Regimes never traded: ['BIAS_NEUTRAL', 'CONSOLIDATING', 'RANGING', 'TRENDING', 'VOLATILE']

──────────────────────────────────────────────────────────────
CHECK 3: EV PREDICTED vs REALIZED RR
──────────────────────────────────────────────────────────────
  Insufficient trades (9) for EV calibration

──────────────────────────────────────────────────────────────
CHECK 4: GRU CONFIDENCE CALIBRATION  (p_bull/bear vs win rate)
──────────────────────────────────────────────────────────────
  Insufficient trades (9) for calibration

──────────────────────────────────────────────────────────────
CHECK 5: GRU ↔ EV MODEL CONSISTENCY  (direction × value agreement)
──────────────────────────────────────────────────────────────
  Insufficient trades (9) for GRU↔EV check

──────────────────────────────────────────────────────────────
SUMMARY — 3 flag(s):
  ⚠  XAUUSD: 2.00/day (>1.5)
  ⚠  BIAS_UP = 78% of trades — regime collapse?
  ⚠  Regimes never traded: ['BIAS_NEUTRAL', 'CONSOLIDATING', 'RANGING', 'TRENDING', 'VOLATILE']
──────────────────────────────────────────────────────────────

======================================================================
  BACKTEST COMPLETE  (round 0 / window=train)
======================================================================
  Round     Trades       WR     PF*  Sharpe*
  ------------------------------------------
  Round 0          9     44.4%    1.442     2.489

  DONE  Train-window backtest for Quality/RL labels
  Saved Train Quality/RL source result → train_quality_rl_source_summary.json
  Train-label journal entries: 9

=== Train Quality + RL on train-only journal ===
  START Train-only Quality+RL retrain
2026-05-11 09:56:00,379 INFO Round 0: wrote 9 journal entries (total in file: 9)
2026-05-11 09:56:00,927 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-05-11 09:56:00,928 INFO Journal entries: 9 total, 9 allowed for training (['live', 'paper', 'production', 'train'])
2026-05-11 09:56:00,929 WARNING Journal has only 9 allowed entries (need 50) — not enough clean Quality/RL training data. Check step6 logs or collect live/paper data.
  DONE  Train-only Quality+RL retrain
  Archived journal → trade_journal_train_only.jsonl
  Archived journal CSV → trade_journal_train_only.csv

=== Round 1: Backtest on train-tail window (latest 2yr inside training data) ===
  Cleared journal for fresh Round 1 run
  START Round 1 - Backtest (train-tail)
2026-05-11 09:56:01,510 INFO === STEP 6: BACKTEST (round1) ===
2026-05-11 09:56:01,513 INFO BT_WINDOW=round1 — train-tail backtest: 2021-08-05 → 2023-08-04 (seen training data; test set protected)
2026-05-11 09:56:01,513 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-05-11 09:56:01,513 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
2026-05-11 09:56:01,513 INFO Round 1 — QualityScorer hard gate disabled; set BACKTEST_ENABLE_QUALITY_GATE=1 to enable it
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
2026-05-11 09:57:17,082 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
2026-05-11 09:57:17,130 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:812: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:812: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 09:57:17,437 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:814: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 09:57:17,446 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:814: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:818: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:818: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:822: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:822: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:826: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:826: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:830: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:830: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:834: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 09:57:17,640 WARNING _build_sequence_df: HTF frame 5M filled 461 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:812: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:834: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:836: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:836: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 09:57:17,697 WARNING _build_sequence_df: HTF frame 5M filled 461 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:814: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 09:57:17,740 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:818: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:812: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:822: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:826: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 09:57:17,809 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:814: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:830: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:818: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:834: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:836: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:822: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:826: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:830: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:834: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:836: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
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
2026-05-11 09:57:26,648 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:812: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 09:57:26,672 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:814: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:818: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:822: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:826: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 09:57:26,725 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:812: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:830: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:834: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:836: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 09:57:26,776 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:814: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:818: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:822: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:826: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:830: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:834: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:836: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260511_095603.json
Trader                                   Trades      WR    PF*  Return*    ExpR   TP1%   TP2%     DD*  Sharpe*  Sortino     IC  Verdict
-----------------------------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)                4  50.0%   1.44    0.9%   0.220 50.0%  0.0%   1.0%     2.45     0.22  0.000     FAIL
  FAILED rules: min_trades, t_stat_above_1_5
  monthly R: 2021-10=-1.00  2022-04=+2.88  2022-06=-1.00
  MonteCarlo P95 DD=2.0%  P10 equity=10,088  t=0.00 (p=1.000)  Sharpe CI=[2.45, 2.45]  streak=1
  gate_diagnostics: bars=263960 no_signal=189326 quality_block=0 session_skip=74630 density=0 pm_reject=0 daily_skip=0 cooldown=0 daily_halt_events=0 enforce_daily_halt=False
  no_signal_reasons: no_trade_uncertain=69131, weak_gru_direction=49173, gru_expected_r_below_threshold=35014, no_trade_chop=17464, no_trade_extreme_vol=15292, htf_low_regime_confidence=2104

Calibration Summary:
  all          [N/A] Insufficient data: 4 samples
  ml_trader    [N/A] Insufficient data: 4 samples
2026-05-11 09:58:10,032 INFO Round 1 backtest — 4 trades | avg WR=50.0% | avg PF=1.44 | avg Sharpe=2.45
2026-05-11 09:58:10,032 INFO   ml_trader: 4 trades | WR=50.0% | fixed PF=1.44 | Return=0.9% | ExpR=0.220 | DD=1.0% | Sharpe=2.45
2026-05-11 09:58:10,032 INFO   ml_trader gate_diagnostics: bars=263960 no_signal=189326 quality_block=0 session_skip=74630 density=0 pm_reject=0
2026-05-11 09:58:10,032 INFO   ml_trader no_signal_reasons: {'no_trade_uncertain': 69131, 'weak_gru_direction': 49173, 'gru_expected_r_below_threshold': 35014, 'no_trade_chop': 17464, 'no_trade_extreme_vol': 15292, 'tradeability_direction_conflict': 625, 'htf_low_regime_confidence': 2104, 'wait_pullback': 398, 'trend_structure_missing': 125}
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 4
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (4 rows)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  GBPJPY          1 trades     1 days   1.00/day
  USDJPY          3 trades     2 days   1.50/day
  ✓  All symbols within normal range.

──────────────────────────────────────────────────────────────
CHECK 2: REGIME DISTRIBUTION  (% of trades, WR, avgEV)
──────────────────────────────────────────────────────────────
  BIAS_UP               4 trades  100.0%  WR=50.0%  avgEV=0.000
  ⚠  BIAS_UP = 100% of trades — regime collapse?
  ⚠  Regimes never traded: ['BIAS_DOWN', 'BIAS_NEUTRAL', 'CONSOLIDATING', 'RANGING', 'TRENDING', 'VOLATILE']

──────────────────────────────────────────────────────────────
CHECK 3: EV PREDICTED vs REALIZED RR
──────────────────────────────────────────────────────────────
  Insufficient trades (4) for EV calibration

──────────────────────────────────────────────────────────────
CHECK 4: GRU CONFIDENCE CALIBRATION  (p_bull/bear vs win rate)
──────────────────────────────────────────────────────────────
  Insufficient trades (4) for calibration

──────────────────────────────────────────────────────────────
CHECK 5: GRU ↔ EV MODEL CONSISTENCY  (direction × value agreement)
──────────────────────────────────────────────────────────────
  Insufficient trades (4) for GRU↔EV check

──────────────────────────────────────────────────────────────
SUMMARY — 2 flag(s):
  ⚠  BIAS_UP = 100% of trades — regime collapse?
  ⚠  Regimes never traded: ['BIAS_DOWN', 'BIAS_NEUTRAL', 'CONSOLIDATING', 'RANGING', 'TRENDING', 'VOLATILE']
──────────────────────────────────────────────────────────────

======================================================================
  BACKTEST COMPLETE  (round 1 / window=round1)
======================================================================
  Round     Trades       WR     PF*  Sharpe*
  ------------------------------------------
  Round 1          4     50.0%    1.439     2.454

  DONE  Round 1 - Backtest (train-tail)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 4 entries

  SKIP  Round 1 Quality+RL retrain — train-tail journal kept evaluation-only

  QualityScorer trade count: R0=9 R1=4 combined=13 (floor=50)
  QualityScorer: 13 combined trades < 50 minimum — gate disabled

=== Pre-Round 2: Incremental retrain (GRU + Regime) ===
  START Retrain gru [pre-R2 retrain]
2026-05-11 09:58:10,253 INFO Round 1: wrote 4 journal entries (total in file: 4)
2026-05-11 09:58:10,882 INFO retrain environment: KAGGLE
2026-05-11 09:58:12,581 INFO Device: CUDA (2 GPU(s))
2026-05-11 09:58:12,593 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 09:58:12,593 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 09:58:12,593 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 09:58:12,595 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 09:58:12,595 INFO Retrain data split: train
2026-05-11 09:58:12,595 INFO Retrain rolling fold selector: latest
2026-05-11 09:58:12,596 INFO === GRU-LSTM retrain (timeframes: ['15M']) ===
2026-05-11 09:58:12,741 INFO NumExpr defaulting to 4 threads.
2026-05-11 09:58:12,947 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-11 09:58:12,948 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 09:58:12,948 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 09:58:13,201 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-05-11 09:58:13,201 INFO GRU multi-symbol training (Kaggle mode): 6 symbols × ['15M']
2026-05-11 09:58:13,204 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260511_095813
2026-05-11 09:58:13,209 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-05-11 09:58:13,209 INFO Deleted stale GRU artifact for cold start: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:58:13,209 INFO GRU warm start disabled by default; set GRU_ALLOW_WARM_START=1 to reuse compatible weights
2026-05-11 09:58:13,474 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:58:13,504 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:58:13,521 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:58:13,533 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 09:58:13,609 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 09:58:13,615 INFO Loaded XAUUSD/15M split=train fold=latest: 176438 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:58:14,191 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:58:14,211 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:58:14,227 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:58:14,236 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:58:14,279 INFO Loaded EURUSD/15M split=train fold=latest: 174951 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:58:14,853 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:58:14,874 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:58:14,890 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:58:14,898 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:58:14,940 INFO Loaded USDJPY/15M split=train fold=latest: 174944 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:58:15,494 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:58:15,516 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:58:15,532 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:58:15,541 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:58:15,582 INFO Loaded EURJPY/15M split=train fold=latest: 174948 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:58:16,138 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:58:16,159 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:58:16,175 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:58:16,184 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:58:16,226 INFO Loaded GBPJPY/15M split=train fold=latest: 174902 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:58:16,768 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:58:16,788 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:58:16,805 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:58:16,813 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 09:58:16,853 INFO Loaded GBPUSD/15M split=train fold=latest: 174899 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 09:58:17,304 INFO train_multi: 6 segments, ~971854 total bars
2026-05-11 09:58:17,312 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-05-11 09:58:17,312 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-05-11 09:58:17,312 INFO train_multi: training ALL 6 segments across TFs ['15M'] in one combined pass
2026-05-11 09:58:17,312 INFO train_multi: building combined dataset for TF=ALL (6 segments)
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
2026-05-11 09:58:26,869 INFO train_multi TF=ALL: 971494 sequences across 6 segments
2026-05-11 09:58:26,870 INFO train_multi TF=ALL: estimated peak RAM = 21312 MB (train=419996 calib=60000 val=120002 n_feat=74 seq_len=60)
2026-05-11 09:58:26,870 WARNING train_multi TF=ALL: trimming to fit RAM budget — new train=394144 calib=56306 val=112612 (20000 MB est)
2026-05-11 09:58:29,163 INFO train_multi TF=ALL: train=394144 calib=56306 val=112612 (10009 MB tensors)
2026-05-11 09:58:35,803 INFO train_multi TF=ALL: structural bar weighting — 252452 structural bars (64.1%) weight=15.0 structural_only=0
2026-05-11 09:58:36,863 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=25, min_epochs=45)
2026-05-11 09:58:55,600 INFO train_multi TF=ALL epoch 1/100 train=2.3377 val=2.3420 r_mae=0.977 pos_r_acc=0.455 side_acc=0.507 r_n=161888
2026-05-11 09:58:55,606 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:58:55,606 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:58:55,606 INFO train_multi TF=ALL: new best val=2.3420 r_mae=0.9769 — saved
2026-05-11 09:58:55,611 INFO train_multi TF=ALL: new best r_mae=0.9769 — saved rmae checkpoint
2026-05-11 09:59:12,017 INFO train_multi TF=ALL epoch 2/100 train=2.3350 val=2.3373 r_mae=0.973 pos_r_acc=0.505 side_acc=0.507 r_n=161888
2026-05-11 09:59:12,023 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:59:12,023 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:59:12,023 INFO train_multi TF=ALL: new best val=2.3373 r_mae=0.9732 — saved
2026-05-11 09:59:12,027 INFO train_multi TF=ALL: new best r_mae=0.9732 — saved rmae checkpoint
2026-05-11 09:59:28,319 INFO train_multi TF=ALL epoch 3/100 train=2.3312 val=2.3302 r_mae=0.967 pos_r_acc=0.545 side_acc=0.493 r_n=161888
2026-05-11 09:59:28,325 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:59:28,325 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:59:28,325 INFO train_multi TF=ALL: new best val=2.3302 r_mae=0.9666 — saved
2026-05-11 09:59:28,329 INFO train_multi TF=ALL: new best r_mae=0.9666 — saved rmae checkpoint
2026-05-11 09:59:44,657 INFO train_multi TF=ALL epoch 4/100 train=2.3293 val=2.3288 r_mae=0.966 pos_r_acc=0.545 side_acc=0.516 r_n=161888
2026-05-11 09:59:44,662 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 09:59:44,662 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 09:59:44,663 INFO train_multi TF=ALL: new best val=2.3288 r_mae=0.9662 — saved
2026-05-11 09:59:44,667 INFO train_multi TF=ALL: new best r_mae=0.9662 — saved rmae checkpoint
2026-05-11 10:00:01,092 INFO train_multi TF=ALL epoch 5/100 train=2.3279 val=2.3272 r_mae=0.966 pos_r_acc=0.545 side_acc=0.527 r_n=161888
2026-05-11 10:00:01,097 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:00:01,097 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:00:01,097 INFO train_multi TF=ALL: new best val=2.3272 r_mae=0.9659 — saved
2026-05-11 10:00:01,102 INFO train_multi TF=ALL: new best r_mae=0.9659 — saved rmae checkpoint
2026-05-11 10:00:17,311 INFO train_multi TF=ALL epoch 6/100 train=2.3266 val=2.3245 r_mae=0.966 pos_r_acc=0.545 side_acc=0.526 r_n=161888
2026-05-11 10:00:17,316 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:00:17,316 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:00:17,316 INFO train_multi TF=ALL: new best val=2.3245 r_mae=0.9659 — saved
2026-05-11 10:00:33,779 INFO train_multi TF=ALL epoch 7/100 train=2.3246 val=2.3220 r_mae=0.966 pos_r_acc=0.546 side_acc=0.526 r_n=161888
2026-05-11 10:00:33,785 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:00:33,785 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:00:33,785 INFO train_multi TF=ALL: new best val=2.3220 r_mae=0.9658 — saved
2026-05-11 10:00:33,790 INFO train_multi TF=ALL: new best r_mae=0.9658 — saved rmae checkpoint
2026-05-11 10:00:50,411 INFO train_multi TF=ALL epoch 8/100 train=2.3206 val=2.3207 r_mae=0.965 pos_r_acc=0.546 side_acc=0.522 r_n=161888
2026-05-11 10:00:50,416 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:00:50,416 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:00:50,416 INFO train_multi TF=ALL: new best val=2.3207 r_mae=0.9646 — saved
2026-05-11 10:00:50,421 INFO train_multi TF=ALL: new best r_mae=0.9646 — saved rmae checkpoint
2026-05-11 10:01:06,822 INFO train_multi TF=ALL epoch 9/100 train=2.3184 val=2.3188 r_mae=0.964 pos_r_acc=0.545 side_acc=0.526 r_n=161888
2026-05-11 10:01:06,828 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:01:06,828 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:01:06,828 INFO train_multi TF=ALL: new best val=2.3188 r_mae=0.9640 — saved
2026-05-11 10:01:06,832 INFO train_multi TF=ALL: new best r_mae=0.9640 — saved rmae checkpoint
2026-05-11 10:01:23,337 INFO train_multi TF=ALL epoch 10/100 train=2.3162 val=2.3180 r_mae=0.963 pos_r_acc=0.546 side_acc=0.528 r_n=161888
2026-05-11 10:01:23,342 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:01:23,342 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:01:23,342 INFO train_multi TF=ALL: new best val=2.3180 r_mae=0.9633 — saved
2026-05-11 10:01:23,347 INFO train_multi TF=ALL: new best r_mae=0.9633 — saved rmae checkpoint
2026-05-11 10:01:39,650 INFO train_multi TF=ALL epoch 11/100 train=2.3145 val=2.3163 r_mae=0.962 pos_r_acc=0.546 side_acc=0.530 r_n=161888
2026-05-11 10:01:39,655 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:01:39,655 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:01:39,655 INFO train_multi TF=ALL: new best val=2.3163 r_mae=0.9622 — saved
2026-05-11 10:01:39,660 INFO train_multi TF=ALL: new best r_mae=0.9622 — saved rmae checkpoint
2026-05-11 10:01:56,194 INFO train_multi TF=ALL epoch 12/100 train=2.3127 val=2.3144 r_mae=0.962 pos_r_acc=0.548 side_acc=0.530 r_n=161888
2026-05-11 10:01:56,199 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:01:56,199 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:01:56,199 INFO train_multi TF=ALL: new best val=2.3144 r_mae=0.9615 — saved
2026-05-11 10:01:56,203 INFO train_multi TF=ALL: new best r_mae=0.9615 — saved rmae checkpoint
2026-05-11 10:02:12,694 INFO train_multi TF=ALL epoch 13/100 train=2.3099 val=2.3126 r_mae=0.961 pos_r_acc=0.549 side_acc=0.533 r_n=161888
2026-05-11 10:02:12,700 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:02:12,700 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:02:12,700 INFO train_multi TF=ALL: new best val=2.3126 r_mae=0.9606 — saved
2026-05-11 10:02:12,704 INFO train_multi TF=ALL: new best r_mae=0.9606 — saved rmae checkpoint
2026-05-11 10:02:29,050 INFO train_multi TF=ALL epoch 14/100 train=2.3073 val=2.3106 r_mae=0.960 pos_r_acc=0.552 side_acc=0.537 r_n=161888
2026-05-11 10:02:29,056 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:02:29,056 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:02:29,056 INFO train_multi TF=ALL: new best val=2.3106 r_mae=0.9596 — saved
2026-05-11 10:02:29,061 INFO train_multi TF=ALL: new best r_mae=0.9596 — saved rmae checkpoint
2026-05-11 10:02:45,520 INFO train_multi TF=ALL epoch 15/100 train=2.3028 val=2.3036 r_mae=0.956 pos_r_acc=0.559 side_acc=0.543 r_n=161888
2026-05-11 10:02:45,526 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:02:45,526 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:02:45,526 INFO train_multi TF=ALL: new best val=2.3036 r_mae=0.9556 — saved
2026-05-11 10:02:45,530 INFO train_multi TF=ALL: new best r_mae=0.9556 — saved rmae checkpoint
2026-05-11 10:03:02,023 INFO train_multi TF=ALL epoch 16/100 train=2.2949 val=2.2892 r_mae=0.950 pos_r_acc=0.570 side_acc=0.549 r_n=161888
2026-05-11 10:03:02,028 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:03:02,029 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:03:02,029 INFO train_multi TF=ALL: new best val=2.2892 r_mae=0.9504 — saved
2026-05-11 10:03:02,034 INFO train_multi TF=ALL: new best r_mae=0.9504 — saved rmae checkpoint
2026-05-11 10:03:18,369 INFO train_multi TF=ALL epoch 17/100 train=2.2828 val=2.2758 r_mae=0.944 pos_r_acc=0.580 side_acc=0.553 r_n=161888
2026-05-11 10:03:18,374 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:03:18,374 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:03:18,375 INFO train_multi TF=ALL: new best val=2.2758 r_mae=0.9443 — saved
2026-05-11 10:03:18,379 INFO train_multi TF=ALL: new best r_mae=0.9443 — saved rmae checkpoint
2026-05-11 10:03:34,807 INFO train_multi TF=ALL epoch 18/100 train=2.2721 val=2.2684 r_mae=0.938 pos_r_acc=0.583 side_acc=0.551 r_n=161888
2026-05-11 10:03:34,812 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:03:34,812 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:03:34,812 INFO train_multi TF=ALL: new best val=2.2684 r_mae=0.9384 — saved
2026-05-11 10:03:34,816 INFO train_multi TF=ALL: new best r_mae=0.9384 — saved rmae checkpoint
2026-05-11 10:03:51,267 INFO train_multi TF=ALL epoch 19/100 train=2.2654 val=2.2604 r_mae=0.935 pos_r_acc=0.587 side_acc=0.556 r_n=161888
2026-05-11 10:03:51,272 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:03:51,272 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:03:51,272 INFO train_multi TF=ALL: new best val=2.2604 r_mae=0.9347 — saved
2026-05-11 10:03:51,277 INFO train_multi TF=ALL: new best r_mae=0.9347 — saved rmae checkpoint
2026-05-11 10:04:07,637 INFO train_multi TF=ALL epoch 20/100 train=2.2583 val=2.2598 r_mae=0.936 pos_r_acc=0.585 side_acc=0.552 r_n=161888
2026-05-11 10:04:07,642 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:04:07,642 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:04:07,642 INFO train_multi TF=ALL: new best val=2.2598 r_mae=0.9365 — saved
2026-05-11 10:04:23,977 INFO train_multi TF=ALL epoch 21/100 train=2.2527 val=2.2545 r_mae=0.930 pos_r_acc=0.590 side_acc=0.559 r_n=161888
2026-05-11 10:04:23,982 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:04:23,982 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:04:23,982 INFO train_multi TF=ALL: new best val=2.2545 r_mae=0.9297 — saved
2026-05-11 10:04:23,987 INFO train_multi TF=ALL: new best r_mae=0.9297 — saved rmae checkpoint
2026-05-11 10:04:40,310 INFO train_multi TF=ALL epoch 22/100 train=2.2480 val=2.2519 r_mae=0.931 pos_r_acc=0.591 side_acc=0.557 r_n=161888
2026-05-11 10:04:40,315 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:04:40,315 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:04:40,315 INFO train_multi TF=ALL: new best val=2.2519 r_mae=0.9308 — saved
2026-05-11 10:04:56,819 INFO train_multi TF=ALL epoch 23/100 train=2.2434 val=2.2457 r_mae=0.928 pos_r_acc=0.593 side_acc=0.557 r_n=161888
2026-05-11 10:04:56,824 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:04:56,824 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:04:56,825 INFO train_multi TF=ALL: new best val=2.2457 r_mae=0.9279 — saved
2026-05-11 10:04:56,829 INFO train_multi TF=ALL: new best r_mae=0.9279 — saved rmae checkpoint
2026-05-11 10:05:13,258 INFO train_multi TF=ALL epoch 24/100 train=2.2369 val=2.2398 r_mae=0.924 pos_r_acc=0.594 side_acc=0.565 r_n=161888
2026-05-11 10:05:13,263 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:05:13,264 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:05:13,264 INFO train_multi TF=ALL: new best val=2.2398 r_mae=0.9238 — saved
2026-05-11 10:05:13,268 INFO train_multi TF=ALL: new best r_mae=0.9238 — saved rmae checkpoint
2026-05-11 10:05:29,607 INFO train_multi TF=ALL epoch 25/100 train=2.2317 val=2.2359 r_mae=0.921 pos_r_acc=0.596 side_acc=0.568 r_n=161888
2026-05-11 10:05:29,612 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:05:29,612 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:05:29,612 INFO train_multi TF=ALL: new best val=2.2359 r_mae=0.9215 — saved
2026-05-11 10:05:29,616 INFO train_multi TF=ALL: new best r_mae=0.9215 — saved rmae checkpoint
2026-05-11 10:05:46,120 INFO train_multi TF=ALL epoch 26/100 train=2.2281 val=2.2320 r_mae=0.918 pos_r_acc=0.601 side_acc=0.570 r_n=161888
2026-05-11 10:05:46,125 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:05:46,125 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:05:46,125 INFO train_multi TF=ALL: new best val=2.2320 r_mae=0.9183 — saved
2026-05-11 10:05:46,129 INFO train_multi TF=ALL: new best r_mae=0.9183 — saved rmae checkpoint
2026-05-11 10:06:02,629 INFO train_multi TF=ALL epoch 27/100 train=2.2229 val=2.2294 r_mae=0.918 pos_r_acc=0.601 side_acc=0.571 r_n=161888
2026-05-11 10:06:02,634 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:06:02,634 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:06:02,634 INFO train_multi TF=ALL: new best val=2.2294 r_mae=0.9184 — saved
2026-05-11 10:06:19,066 INFO train_multi TF=ALL epoch 28/100 train=2.2161 val=2.2250 r_mae=0.915 pos_r_acc=0.602 side_acc=0.573 r_n=161888
2026-05-11 10:06:19,076 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:06:19,076 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:06:19,076 INFO train_multi TF=ALL: new best val=2.2250 r_mae=0.9151 — saved
2026-05-11 10:06:19,080 INFO train_multi TF=ALL: new best r_mae=0.9151 — saved rmae checkpoint
2026-05-11 10:06:35,479 INFO train_multi TF=ALL epoch 29/100 train=2.2094 val=2.2185 r_mae=0.915 pos_r_acc=0.607 side_acc=0.577 r_n=161888
2026-05-11 10:06:35,484 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:06:35,484 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:06:35,484 INFO train_multi TF=ALL: new best val=2.2185 r_mae=0.9154 — saved
2026-05-11 10:06:51,909 INFO train_multi TF=ALL epoch 30/100 train=2.2034 val=2.2214 r_mae=0.913 pos_r_acc=0.603 side_acc=0.578 r_n=161888
2026-05-11 10:06:51,920 INFO train_multi TF=ALL: new best r_mae=0.9128 — saved rmae checkpoint
2026-05-11 10:07:08,240 INFO train_multi TF=ALL epoch 31/100 train=2.1945 val=2.2025 r_mae=0.907 pos_r_acc=0.612 side_acc=0.588 r_n=161888
2026-05-11 10:07:08,245 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:07:08,245 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:07:08,245 INFO train_multi TF=ALL: new best val=2.2025 r_mae=0.9066 — saved
2026-05-11 10:07:08,250 INFO train_multi TF=ALL: new best r_mae=0.9066 — saved rmae checkpoint
2026-05-11 10:07:24,683 INFO train_multi TF=ALL epoch 32/100 train=2.1820 val=2.1918 r_mae=0.906 pos_r_acc=0.613 side_acc=0.598 r_n=161888
2026-05-11 10:07:24,688 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:07:24,688 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:07:24,688 INFO train_multi TF=ALL: new best val=2.1918 r_mae=0.9058 — saved
2026-05-11 10:07:24,693 INFO train_multi TF=ALL: new best r_mae=0.9058 — saved rmae checkpoint
2026-05-11 10:07:41,060 INFO train_multi TF=ALL epoch 33/100 train=2.1712 val=2.1703 r_mae=0.898 pos_r_acc=0.621 side_acc=0.608 r_n=161888
2026-05-11 10:07:41,065 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:07:41,065 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:07:41,065 INFO train_multi TF=ALL: new best val=2.1703 r_mae=0.8976 — saved
2026-05-11 10:07:41,069 INFO train_multi TF=ALL: new best r_mae=0.8976 — saved rmae checkpoint
2026-05-11 10:07:57,520 INFO train_multi TF=ALL epoch 34/100 train=2.1557 val=2.1544 r_mae=0.894 pos_r_acc=0.625 side_acc=0.614 r_n=161888
2026-05-11 10:07:57,526 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:07:57,526 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:07:57,526 INFO train_multi TF=ALL: new best val=2.1544 r_mae=0.8938 — saved
2026-05-11 10:07:57,530 INFO train_multi TF=ALL: new best r_mae=0.8938 — saved rmae checkpoint
2026-05-11 10:08:13,891 INFO train_multi TF=ALL epoch 35/100 train=2.1333 val=2.1350 r_mae=0.882 pos_r_acc=0.633 side_acc=0.621 r_n=161888
2026-05-11 10:08:13,897 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:08:13,897 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:08:13,897 INFO train_multi TF=ALL: new best val=2.1350 r_mae=0.8817 — saved
2026-05-11 10:08:13,901 INFO train_multi TF=ALL: new best r_mae=0.8817 — saved rmae checkpoint
2026-05-11 10:08:30,267 INFO train_multi TF=ALL epoch 36/100 train=2.1138 val=2.1135 r_mae=0.874 pos_r_acc=0.641 side_acc=0.633 r_n=161888
2026-05-11 10:08:30,272 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:08:30,273 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:08:30,273 INFO train_multi TF=ALL: new best val=2.1135 r_mae=0.8740 — saved
2026-05-11 10:08:30,277 INFO train_multi TF=ALL: new best r_mae=0.8740 — saved rmae checkpoint
2026-05-11 10:08:46,633 INFO train_multi TF=ALL epoch 37/100 train=2.0890 val=2.0943 r_mae=0.863 pos_r_acc=0.647 side_acc=0.637 r_n=161888
2026-05-11 10:08:46,640 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:08:46,640 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:08:46,640 INFO train_multi TF=ALL: new best val=2.0943 r_mae=0.8626 — saved
2026-05-11 10:08:46,644 INFO train_multi TF=ALL: new best r_mae=0.8626 — saved rmae checkpoint
2026-05-11 10:09:03,085 INFO train_multi TF=ALL epoch 38/100 train=2.0708 val=2.0815 r_mae=0.852 pos_r_acc=0.652 side_acc=0.641 r_n=161888
2026-05-11 10:09:03,090 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:09:03,090 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:09:03,091 INFO train_multi TF=ALL: new best val=2.0815 r_mae=0.8519 — saved
2026-05-11 10:09:03,095 INFO train_multi TF=ALL: new best r_mae=0.8519 — saved rmae checkpoint
2026-05-11 10:09:19,380 INFO train_multi TF=ALL epoch 39/100 train=2.0507 val=2.0721 r_mae=0.844 pos_r_acc=0.658 side_acc=0.641 r_n=161888
2026-05-11 10:09:19,385 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:09:19,385 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:09:19,385 INFO train_multi TF=ALL: new best val=2.0721 r_mae=0.8442 — saved
2026-05-11 10:09:19,390 INFO train_multi TF=ALL: new best r_mae=0.8442 — saved rmae checkpoint
2026-05-11 10:09:35,904 INFO train_multi TF=ALL epoch 40/100 train=2.0425 val=2.0668 r_mae=0.844 pos_r_acc=0.657 side_acc=0.641 r_n=161888
2026-05-11 10:09:35,909 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:09:35,909 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:09:35,909 INFO train_multi TF=ALL: new best val=2.0668 r_mae=0.8445 — saved
2026-05-11 10:09:52,344 INFO train_multi TF=ALL epoch 41/100 train=2.0262 val=2.0615 r_mae=0.840 pos_r_acc=0.658 side_acc=0.645 r_n=161888
2026-05-11 10:09:52,354 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:09:52,355 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:09:52,355 INFO train_multi TF=ALL: new best val=2.0615 r_mae=0.8397 — saved
2026-05-11 10:09:52,359 INFO train_multi TF=ALL: new best r_mae=0.8397 — saved rmae checkpoint
2026-05-11 10:10:08,700 INFO train_multi TF=ALL epoch 42/100 train=2.0168 val=2.0522 r_mae=0.830 pos_r_acc=0.663 side_acc=0.647 r_n=161888
2026-05-11 10:10:08,705 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:10:08,705 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:10:08,705 INFO train_multi TF=ALL: new best val=2.0522 r_mae=0.8302 — saved
2026-05-11 10:10:08,710 INFO train_multi TF=ALL: new best r_mae=0.8302 — saved rmae checkpoint
2026-05-11 10:10:25,036 INFO train_multi TF=ALL epoch 43/100 train=2.0082 val=2.0522 r_mae=0.830 pos_r_acc=0.663 side_acc=0.648 r_n=161888
2026-05-11 10:10:41,442 INFO train_multi TF=ALL epoch 44/100 train=1.9968 val=2.0379 r_mae=0.823 pos_r_acc=0.668 side_acc=0.651 r_n=161888
2026-05-11 10:10:41,447 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:10:41,447 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:10:41,447 INFO train_multi TF=ALL: new best val=2.0379 r_mae=0.8228 — saved
2026-05-11 10:10:41,452 INFO train_multi TF=ALL: new best r_mae=0.8228 — saved rmae checkpoint
2026-05-11 10:10:58,035 INFO train_multi TF=ALL epoch 45/100 train=1.9938 val=2.0357 r_mae=0.821 pos_r_acc=0.666 side_acc=0.654 r_n=161888
2026-05-11 10:10:58,041 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:10:58,041 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:10:58,041 INFO train_multi TF=ALL: new best val=2.0357 r_mae=0.8215 — saved
2026-05-11 10:10:58,046 INFO train_multi TF=ALL: new best r_mae=0.8215 — saved rmae checkpoint
2026-05-11 10:11:14,378 INFO train_multi TF=ALL epoch 46/100 train=1.9837 val=2.0297 r_mae=0.821 pos_r_acc=0.669 side_acc=0.657 r_n=161888
2026-05-11 10:11:14,383 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:11:14,383 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:11:14,383 INFO train_multi TF=ALL: new best val=2.0297 r_mae=0.8210 — saved
2026-05-11 10:11:14,388 INFO train_multi TF=ALL: new best r_mae=0.8210 — saved rmae checkpoint
2026-05-11 10:11:30,726 INFO train_multi TF=ALL epoch 47/100 train=1.9724 val=2.0231 r_mae=0.813 pos_r_acc=0.670 side_acc=0.659 r_n=161888
2026-05-11 10:11:30,731 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:11:30,731 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:11:30,731 INFO train_multi TF=ALL: new best val=2.0231 r_mae=0.8127 — saved
2026-05-11 10:11:30,735 INFO train_multi TF=ALL: new best r_mae=0.8127 — saved rmae checkpoint
2026-05-11 10:11:46,858 INFO train_multi TF=ALL epoch 48/100 train=1.9641 val=2.0208 r_mae=0.812 pos_r_acc=0.669 side_acc=0.662 r_n=161888
2026-05-11 10:11:46,864 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:11:46,864 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:11:46,864 INFO train_multi TF=ALL: new best val=2.0208 r_mae=0.8124 — saved
2026-05-11 10:11:46,868 INFO train_multi TF=ALL: new best r_mae=0.8124 — saved rmae checkpoint
2026-05-11 10:12:03,190 INFO train_multi TF=ALL epoch 49/100 train=1.9548 val=2.0273 r_mae=0.809 pos_r_acc=0.667 side_acc=0.659 r_n=161888
2026-05-11 10:12:03,195 INFO train_multi TF=ALL: new best r_mae=0.8093 — saved rmae checkpoint
2026-05-11 10:12:19,580 INFO train_multi TF=ALL epoch 50/100 train=1.9498 val=2.0100 r_mae=0.803 pos_r_acc=0.675 side_acc=0.665 r_n=161888
2026-05-11 10:12:19,591 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:12:19,591 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:12:19,591 INFO train_multi TF=ALL: new best val=2.0100 r_mae=0.8028 — saved
2026-05-11 10:12:19,595 INFO train_multi TF=ALL: new best r_mae=0.8028 — saved rmae checkpoint
2026-05-11 10:12:35,890 INFO train_multi TF=ALL epoch 51/100 train=1.9403 val=2.0179 r_mae=0.812 pos_r_acc=0.668 side_acc=0.661 r_n=161888
2026-05-11 10:12:52,286 INFO train_multi TF=ALL epoch 52/100 train=1.9330 val=1.9931 r_mae=0.804 pos_r_acc=0.674 side_acc=0.671 r_n=161888
2026-05-11 10:12:52,291 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:12:52,291 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:12:52,292 INFO train_multi TF=ALL: new best val=1.9931 r_mae=0.8035 — saved
2026-05-11 10:13:08,492 INFO train_multi TF=ALL epoch 53/100 train=1.9249 val=2.0019 r_mae=0.799 pos_r_acc=0.676 side_acc=0.665 r_n=161888
2026-05-11 10:13:08,503 INFO train_multi TF=ALL: new best r_mae=0.7987 — saved rmae checkpoint
2026-05-11 10:13:24,826 INFO train_multi TF=ALL epoch 54/100 train=1.9193 val=1.9877 r_mae=0.793 pos_r_acc=0.679 side_acc=0.670 r_n=161888
2026-05-11 10:13:24,832 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:13:24,832 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:13:24,832 INFO train_multi TF=ALL: new best val=1.9877 r_mae=0.7930 — saved
2026-05-11 10:13:24,836 INFO train_multi TF=ALL: new best r_mae=0.7930 — saved rmae checkpoint
2026-05-11 10:13:41,088 INFO train_multi TF=ALL epoch 55/100 train=1.9129 val=1.9871 r_mae=0.796 pos_r_acc=0.678 side_acc=0.671 r_n=161888
2026-05-11 10:13:41,099 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:13:41,099 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:13:41,099 INFO train_multi TF=ALL: new best val=1.9871 r_mae=0.7958 — saved
2026-05-11 10:13:57,639 INFO train_multi TF=ALL epoch 56/100 train=1.9068 val=1.9848 r_mae=0.795 pos_r_acc=0.676 side_acc=0.675 r_n=161888
2026-05-11 10:13:57,644 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:13:57,644 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:13:57,644 INFO train_multi TF=ALL: new best val=1.9848 r_mae=0.7945 — saved
2026-05-11 10:14:14,094 INFO train_multi TF=ALL epoch 57/100 train=1.8951 val=1.9666 r_mae=0.790 pos_r_acc=0.683 side_acc=0.677 r_n=161888
2026-05-11 10:14:14,099 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:14:14,099 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:14:14,099 INFO train_multi TF=ALL: new best val=1.9666 r_mae=0.7902 — saved
2026-05-11 10:14:14,103 INFO train_multi TF=ALL: new best r_mae=0.7902 — saved rmae checkpoint
2026-05-11 10:14:30,496 INFO train_multi TF=ALL epoch 58/100 train=1.8866 val=1.9610 r_mae=0.792 pos_r_acc=0.678 side_acc=0.684 r_n=161888
2026-05-11 10:14:30,501 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:14:30,501 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:14:30,502 INFO train_multi TF=ALL: new best val=1.9610 r_mae=0.7923 — saved
2026-05-11 10:14:47,005 INFO train_multi TF=ALL epoch 59/100 train=1.8778 val=1.9554 r_mae=0.786 pos_r_acc=0.680 side_acc=0.684 r_n=161888
2026-05-11 10:14:47,011 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:14:47,011 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:14:47,011 INFO train_multi TF=ALL: new best val=1.9554 r_mae=0.7856 — saved
2026-05-11 10:14:47,015 INFO train_multi TF=ALL: new best r_mae=0.7856 — saved rmae checkpoint
2026-05-11 10:15:03,413 INFO train_multi TF=ALL epoch 60/100 train=1.8703 val=1.9587 r_mae=0.788 pos_r_acc=0.685 side_acc=0.682 r_n=161888
2026-05-11 10:15:19,773 INFO train_multi TF=ALL epoch 61/100 train=1.8585 val=1.9446 r_mae=0.783 pos_r_acc=0.683 side_acc=0.690 r_n=161888
2026-05-11 10:15:19,778 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:15:19,778 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:15:19,778 INFO train_multi TF=ALL: new best val=1.9446 r_mae=0.7828 — saved
2026-05-11 10:15:19,783 INFO train_multi TF=ALL: new best r_mae=0.7828 — saved rmae checkpoint
2026-05-11 10:15:36,173 INFO train_multi TF=ALL epoch 62/100 train=1.8531 val=1.9444 r_mae=0.781 pos_r_acc=0.683 side_acc=0.692 r_n=161888
2026-05-11 10:15:36,179 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:15:36,179 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:15:36,179 INFO train_multi TF=ALL: new best val=1.9444 r_mae=0.7813 — saved
2026-05-11 10:15:36,183 INFO train_multi TF=ALL: new best r_mae=0.7813 — saved rmae checkpoint
2026-05-11 10:15:52,636 INFO train_multi TF=ALL epoch 63/100 train=1.8442 val=1.9439 r_mae=0.781 pos_r_acc=0.684 side_acc=0.692 r_n=161888
2026-05-11 10:15:52,641 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:15:52,641 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:15:52,641 INFO train_multi TF=ALL: new best val=1.9439 r_mae=0.7808 — saved
2026-05-11 10:15:52,646 INFO train_multi TF=ALL: new best r_mae=0.7808 — saved rmae checkpoint
2026-05-11 10:16:09,057 INFO train_multi TF=ALL epoch 64/100 train=1.8394 val=1.9446 r_mae=0.786 pos_r_acc=0.678 side_acc=0.694 r_n=161888
2026-05-11 10:16:25,407 INFO train_multi TF=ALL epoch 65/100 train=1.8311 val=1.9193 r_mae=0.780 pos_r_acc=0.684 side_acc=0.702 r_n=161888
2026-05-11 10:16:25,413 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:16:25,413 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:16:25,413 INFO train_multi TF=ALL: new best val=1.9193 r_mae=0.7796 — saved
2026-05-11 10:16:25,417 INFO train_multi TF=ALL: new best r_mae=0.7796 — saved rmae checkpoint
2026-05-11 10:16:41,658 INFO train_multi TF=ALL epoch 66/100 train=1.8316 val=1.9281 r_mae=0.781 pos_r_acc=0.684 side_acc=0.700 r_n=161888
2026-05-11 10:16:58,086 INFO train_multi TF=ALL epoch 67/100 train=1.8139 val=1.9182 r_mae=0.779 pos_r_acc=0.684 side_acc=0.704 r_n=161888
2026-05-11 10:16:58,091 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:16:58,091 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:16:58,091 INFO train_multi TF=ALL: new best val=1.9182 r_mae=0.7786 — saved
2026-05-11 10:16:58,096 INFO train_multi TF=ALL: new best r_mae=0.7786 — saved rmae checkpoint
2026-05-11 10:17:14,421 INFO train_multi TF=ALL epoch 68/100 train=1.8060 val=1.9051 r_mae=0.782 pos_r_acc=0.686 side_acc=0.710 r_n=161888
2026-05-11 10:17:14,427 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:17:14,427 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:17:14,427 INFO train_multi TF=ALL: new best val=1.9051 r_mae=0.7822 — saved
2026-05-11 10:17:30,757 INFO train_multi TF=ALL epoch 69/100 train=1.7991 val=1.8949 r_mae=0.776 pos_r_acc=0.687 side_acc=0.715 r_n=161888
2026-05-11 10:17:30,763 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:17:30,763 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:17:30,763 INFO train_multi TF=ALL: new best val=1.8949 r_mae=0.7764 — saved
2026-05-11 10:17:30,772 INFO train_multi TF=ALL: new best r_mae=0.7764 — saved rmae checkpoint
2026-05-11 10:17:47,221 INFO train_multi TF=ALL epoch 70/100 train=1.7928 val=1.9176 r_mae=0.783 pos_r_acc=0.678 side_acc=0.713 r_n=161888
2026-05-11 10:18:03,586 INFO train_multi TF=ALL epoch 71/100 train=1.7875 val=1.8942 r_mae=0.774 pos_r_acc=0.686 side_acc=0.718 r_n=161888
2026-05-11 10:18:03,591 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:18:03,591 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:18:03,591 INFO train_multi TF=ALL: new best val=1.8942 r_mae=0.7740 — saved
2026-05-11 10:18:03,595 INFO train_multi TF=ALL: new best r_mae=0.7740 — saved rmae checkpoint
2026-05-11 10:18:19,957 INFO train_multi TF=ALL epoch 72/100 train=1.7816 val=1.9052 r_mae=0.779 pos_r_acc=0.685 side_acc=0.714 r_n=161888
2026-05-11 10:18:36,388 INFO train_multi TF=ALL epoch 73/100 train=1.7722 val=1.9004 r_mae=0.775 pos_r_acc=0.684 side_acc=0.718 r_n=161888
2026-05-11 10:18:52,781 INFO train_multi TF=ALL epoch 74/100 train=1.7651 val=1.8913 r_mae=0.775 pos_r_acc=0.686 side_acc=0.721 r_n=161888
2026-05-11 10:18:52,786 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:18:52,786 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:18:52,786 INFO train_multi TF=ALL: new best val=1.8913 r_mae=0.7755 — saved
2026-05-11 10:19:09,128 INFO train_multi TF=ALL epoch 75/100 train=1.7581 val=1.9060 r_mae=0.778 pos_r_acc=0.683 side_acc=0.719 r_n=161888
2026-05-11 10:19:25,503 INFO train_multi TF=ALL epoch 76/100 train=1.7553 val=1.8929 r_mae=0.778 pos_r_acc=0.684 side_acc=0.720 r_n=161888
2026-05-11 10:19:42,032 INFO train_multi TF=ALL epoch 77/100 train=1.7453 val=1.8920 r_mae=0.775 pos_r_acc=0.685 side_acc=0.725 r_n=161888
2026-05-11 10:19:58,442 INFO train_multi TF=ALL epoch 78/100 train=1.7375 val=1.9056 r_mae=0.776 pos_r_acc=0.682 side_acc=0.721 r_n=161888
2026-05-11 10:20:14,773 INFO train_multi TF=ALL epoch 79/100 train=1.7324 val=1.8984 r_mae=0.777 pos_r_acc=0.681 side_acc=0.726 r_n=161888
2026-05-11 10:20:31,042 INFO train_multi TF=ALL epoch 80/100 train=1.7255 val=1.8908 r_mae=0.777 pos_r_acc=0.683 side_acc=0.726 r_n=161888
2026-05-11 10:20:31,047 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:20:31,047 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:20:31,047 INFO train_multi TF=ALL: new best val=1.8908 r_mae=0.7772 — saved
2026-05-11 10:20:47,365 INFO train_multi TF=ALL epoch 81/100 train=1.7155 val=1.8951 r_mae=0.774 pos_r_acc=0.684 side_acc=0.726 r_n=161888
2026-05-11 10:21:03,843 INFO train_multi TF=ALL epoch 82/100 train=1.7095 val=1.9039 r_mae=0.777 pos_r_acc=0.681 side_acc=0.727 r_n=161888
2026-05-11 10:21:20,222 INFO train_multi TF=ALL epoch 83/100 train=1.6997 val=1.8972 r_mae=0.774 pos_r_acc=0.683 side_acc=0.729 r_n=161888
2026-05-11 10:21:36,593 INFO train_multi TF=ALL epoch 84/100 train=1.6989 val=1.8962 r_mae=0.784 pos_r_acc=0.680 side_acc=0.732 r_n=161888
2026-05-11 10:21:52,964 INFO train_multi TF=ALL epoch 85/100 train=1.6932 val=1.9061 r_mae=0.778 pos_r_acc=0.678 side_acc=0.729 r_n=161888
2026-05-11 10:22:09,239 INFO train_multi TF=ALL epoch 86/100 train=1.6891 val=1.8942 r_mae=0.783 pos_r_acc=0.680 side_acc=0.732 r_n=161888
2026-05-11 10:22:25,492 INFO train_multi TF=ALL epoch 87/100 train=1.6802 val=1.9045 r_mae=0.778 pos_r_acc=0.680 side_acc=0.730 r_n=161888
2026-05-11 10:22:41,807 INFO train_multi TF=ALL epoch 88/100 train=1.6727 val=1.8978 r_mae=0.777 pos_r_acc=0.682 side_acc=0.733 r_n=161888
2026-05-11 10:22:58,128 INFO train_multi TF=ALL epoch 89/100 train=1.6687 val=1.9194 r_mae=0.776 pos_r_acc=0.681 side_acc=0.729 r_n=161888
2026-05-11 10:23:14,515 INFO train_multi TF=ALL epoch 90/100 train=1.6621 val=1.9078 r_mae=0.779 pos_r_acc=0.680 side_acc=0.728 r_n=161888
2026-05-11 10:23:30,826 INFO train_multi TF=ALL epoch 91/100 train=1.6516 val=1.9346 r_mae=0.776 pos_r_acc=0.679 side_acc=0.726 r_n=161888
2026-05-11 10:23:47,236 INFO train_multi TF=ALL epoch 92/100 train=1.6486 val=1.9104 r_mae=0.779 pos_r_acc=0.679 side_acc=0.734 r_n=161888
2026-05-11 10:24:03,637 INFO train_multi TF=ALL epoch 93/100 train=1.6429 val=1.9327 r_mae=0.783 pos_r_acc=0.676 side_acc=0.728 r_n=161888
2026-05-11 10:24:19,936 INFO train_multi TF=ALL epoch 94/100 train=1.6347 val=1.9348 r_mae=0.784 pos_r_acc=0.675 side_acc=0.730 r_n=161888
2026-05-11 10:24:36,272 INFO train_multi TF=ALL epoch 95/100 train=1.6289 val=1.9300 r_mae=0.784 pos_r_acc=0.676 side_acc=0.730 r_n=161888
2026-05-11 10:24:52,627 INFO train_multi TF=ALL epoch 96/100 train=1.6242 val=1.9226 r_mae=0.784 pos_r_acc=0.676 side_acc=0.732 r_n=161888
2026-05-11 10:25:08,911 INFO train_multi TF=ALL epoch 97/100 train=1.6157 val=1.9430 r_mae=0.783 pos_r_acc=0.676 side_acc=0.728 r_n=161888
2026-05-11 10:25:25,267 INFO train_multi TF=ALL epoch 98/100 train=1.6133 val=1.9397 r_mae=0.785 pos_r_acc=0.675 side_acc=0.732 r_n=161888
2026-05-11 10:25:41,647 INFO train_multi TF=ALL epoch 99/100 train=1.6115 val=1.9376 r_mae=0.787 pos_r_acc=0.673 side_acc=0.733 r_n=161888
2026-05-11 10:25:57,538 INFO train_multi TF=ALL epoch 100/100 train=1.6054 val=1.9355 r_mae=0.785 pos_r_acc=0.675 side_acc=0.733 r_n=161888
2026-05-11 10:25:57,550 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:25:57,550 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:25:57,550 INFO train_multi TF=ALL: promoted r_mae checkpoint (0.7740 < primary 0.7772) — overwriting model.pt
2026-05-11 10:25:58,764 WARNING fit_r_isotonic: long calibrator increases MAE (cal=0.7803 >= raw=0.7798) — skipping
2026-05-11 10:25:58,775 INFO fit_r_isotonic: saved side-R calibrators stats={'long': {'n': 40072, 'raw_mae': 0.7798319787329362, 'calibrated_mae': 0.7803478199204986, 'skipped': 'calibrator_hurts'}, 'short': {'n': 41197, 'raw_mae': 0.7966220527750292, 'calibrated_mae': 0.79296753020312}}
2026-05-11 10:25:58,896 WARNING GRU validation R-MAE above floor for TF=ALL: best_val_r_mae=0.774 max=0.750. Keeping saved best weights so the pipeline can progress.
2026-05-11 10:25:58,909 INFO GRU R threshold XAUUSD/buy: q25_pos=0.535 q50_pos=1.667 pos_rate=46.0% (n=119259 n_pos=54899)
2026-05-11 10:25:58,915 INFO GRU R threshold XAUUSD/sell: q25_pos=0.528 q50_pos=1.667 pos_rate=45.5% (n=118087 n_pos=53734)
2026-05-11 10:25:58,921 INFO GRU R threshold EURUSD/buy: q25_pos=0.541 q50_pos=1.667 pos_rate=45.6% (n=118751 n_pos=54196)
2026-05-11 10:25:58,927 INFO GRU R threshold EURUSD/sell: q25_pos=0.545 q50_pos=1.667 pos_rate=45.6% (n=117679 n_pos=53721)
2026-05-11 10:25:58,932 INFO GRU R threshold USDJPY/buy: q25_pos=0.551 q50_pos=1.667 pos_rate=46.2% (n=118691 n_pos=54870)
2026-05-11 10:25:58,937 INFO GRU R threshold USDJPY/sell: q25_pos=0.542 q50_pos=1.667 pos_rate=44.6% (n=116030 n_pos=51793)
2026-05-11 10:25:58,943 INFO GRU R threshold EURJPY/buy: q25_pos=0.530 q50_pos=1.416 pos_rate=46.1% (n=118651 n_pos=54654)
2026-05-11 10:25:58,950 INFO GRU R threshold EURJPY/sell: q25_pos=0.525 q50_pos=1.667 pos_rate=44.7% (n=117320 n_pos=52399)
2026-05-11 10:25:58,956 INFO GRU R threshold GBPJPY/buy: q25_pos=0.517 q50_pos=1.443 pos_rate=45.9% (n=118277 n_pos=54278)
2026-05-11 10:25:58,962 INFO GRU R threshold GBPJPY/sell: q25_pos=0.521 q50_pos=1.667 pos_rate=45.3% (n=116299 n_pos=52702)
2026-05-11 10:25:58,968 INFO GRU R threshold GBPUSD/buy: q25_pos=0.524 q50_pos=1.667 pos_rate=45.8% (n=117764 n_pos=53894)
2026-05-11 10:25:58,974 INFO GRU R threshold GBPUSD/sell: q25_pos=0.533 q50_pos=1.667 pos_rate=45.8% (n=117404 n_pos=53785)
2026-05-11 10:25:58,975 INFO GRU per-symbol R thresholds saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/symbol_r_thresholds.json (6 symbols)
2026-05-11 10:25:58,990 INFO Retrain complete. Total wall-clock: 1666.4s
  DONE  Retrain gru [pre-R2 retrain]
  START Retrain regime [pre-R2 retrain]
2026-05-11 10:26:02,535 INFO retrain environment: KAGGLE
2026-05-11 10:26:04,134 INFO Device: CUDA (2 GPU(s))
2026-05-11 10:26:04,143 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 10:26:04,143 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 10:26:04,143 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 10:26:04,144 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 10:26:04,144 INFO Retrain data split: train
2026-05-11 10:26:04,144 INFO Retrain rolling fold selector: latest
2026-05-11 10:26:04,145 INFO === RegimeClassifier retrain (hierarchical: HTF 2-score bias + LTF 5-score behaviour) ===
2026-05-11 10:26:04,289 INFO NumExpr defaulting to 4 threads.
2026-05-11 10:26:04,478 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-11 10:26:04,479 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 10:26:04,479 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 10:26:04,479 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-11 10:26:04,530 INFO Regime rolling folds selected: [None]
2026-05-11 10:26:04,530 INFO === Regime rolling fold 1/1: train_all ===
2026-05-11 10:26:04,530 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-11 10:26:04,572 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 10:26:04,573 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:26:04,590 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:26:04,606 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:26:04,622 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:26:04,640 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:26:04,656 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:26:04,891 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:26:04,961 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:26:04,985 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:26:04,986 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:26:04,997 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:26:04,998 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:26:05,416 INFO macro_cache: using training data dir /kaggle/working/Multi-Bot/trading-system/training_data
2026-05-11 10:26:05,525 INFO macro_cache loaded 19 series: {'asx200': '2842 bars 2015-01-02→2026-03-27', 'cac40': '2876 bars 2015-01-02→2026-03-27', 'dax': '2851 bars 2015-01-02→2026-03-27', 'djia': '2825 bars 2015-01-02→2026-03-27', 'dxy': '2826 bars 2015-01-02→2026-03-27', 'eurostoxx': '2823 bars 2015-01-05→2026-03-27', 'ftse': '2839 bars 2015-01-02→2026-03-27', 'gold_fut': '2824 bars 2015-01-02→2026-03-27', 'hsi': '2764 bars 2015-01-02→2026-03-27', 'nasdaq': '2825 bars 2015-01-02→2026-03-27', 'nikkei': '2744 bars 2015-01-05→2026-03-27', 'oil_fut': '2825 bars 2015-01-02→2026-03-27', 'spx': '2825 bars 2015-01-02→2026-03-27', 'us10y': '2824 bars 2015-01-02→2026-03-27', 'us30y': '2824 bars 2015-01-02→2026-03-27', 'us3m': '2824 bars 2015-01-02→2026-03-27', 'vix': '2825 bars 2015-01-02→2026-03-27', 'us10y_fred': '2607 bars 2016-03-28→2026-03-24', 'us2y_fred': '2607 bars 2016-03-28→2026-03-24'}
2026-05-11 10:26:05,758 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1209, 'BIAS_DOWN': 568, 'BIAS_NEUTRAL': 10325}  ambiguous=3935 (total=12102) horizon=84
2026-05-11 10:26:05,763 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected XAUUSD — 12052 samples (group=gold) score_means={'bias_up_score': 0.1003, 'bias_down_score': 0.0471} labels={'BIAS_UP': 1209, 'BIAS_DOWN': 568, 'BIAS_NEUTRAL': 10275} clean={'BIAS_UP': 1209, 'BIAS_DOWN': 568, 'BIAS_NEUTRAL': 6348}
2026-05-11 10:26:05,928 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:05,967 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:05,986 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:05,987 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:05,995 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:05,996 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:26:06,574 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 722, 'BIAS_DOWN': 567, 'BIAS_NEUTRAL': 10115}  ambiguous=3689 (total=11404) horizon=84
2026-05-11 10:26:06,579 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURUSD — 11354 samples (group=dollar) score_means={'bias_up_score': 0.0636, 'bias_down_score': 0.0499} labels={'BIAS_UP': 722, 'BIAS_DOWN': 567, 'BIAS_NEUTRAL': 10065} clean={'BIAS_UP': 722, 'BIAS_DOWN': 567, 'BIAS_NEUTRAL': 6394}
2026-05-11 10:26:06,733 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:06,769 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:06,789 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:06,790 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:06,798 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:06,799 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:26:07,387 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 872, 'BIAS_DOWN': 463, 'BIAS_NEUTRAL': 10068}  ambiguous=3827 (total=11403) horizon=84
2026-05-11 10:26:07,391 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected USDJPY — 11353 samples (group=dollar) score_means={'bias_up_score': 0.0768, 'bias_down_score': 0.0408} labels={'BIAS_UP': 872, 'BIAS_DOWN': 463, 'BIAS_NEUTRAL': 10018} clean={'BIAS_UP': 872, 'BIAS_DOWN': 463, 'BIAS_NEUTRAL': 6201}
2026-05-11 10:26:07,550 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:07,586 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:07,609 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:07,609 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:07,618 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:07,619 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:26:08,210 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 712, 'BIAS_DOWN': 556, 'BIAS_NEUTRAL': 10139}  ambiguous=3816 (total=11407) horizon=84
2026-05-11 10:26:08,216 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURJPY — 11357 samples (group=cross) score_means={'bias_up_score': 0.0627, 'bias_down_score': 0.049} labels={'BIAS_UP': 712, 'BIAS_DOWN': 556, 'BIAS_NEUTRAL': 10089} clean={'BIAS_UP': 712, 'BIAS_DOWN': 556, 'BIAS_NEUTRAL': 6279}
2026-05-11 10:26:08,367 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:08,404 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:08,427 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:08,428 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:08,439 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:08,440 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:26:09,047 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 873, 'BIAS_DOWN': 633, 'BIAS_NEUTRAL': 9902}  ambiguous=4022 (total=11408) horizon=84
2026-05-11 10:26:09,053 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPJPY — 11358 samples (group=cross) score_means={'bias_up_score': 0.0769, 'bias_down_score': 0.0557} labels={'BIAS_UP': 873, 'BIAS_DOWN': 633, 'BIAS_NEUTRAL': 9852} clean={'BIAS_UP': 873, 'BIAS_DOWN': 633, 'BIAS_NEUTRAL': 5852}
2026-05-11 10:26:09,211 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:09,245 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:09,265 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:09,265 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:09,274 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:09,275 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:26:09,859 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 546, 'BIAS_DOWN': 754, 'BIAS_NEUTRAL': 10102}  ambiguous=3944 (total=11402) horizon=84
2026-05-11 10:26:09,864 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPUSD — 11352 samples (group=dollar) score_means={'bias_up_score': 0.0481, 'bias_down_score': 0.0651} labels={'BIAS_UP': 546, 'BIAS_DOWN': 739, 'BIAS_NEUTRAL': 10067} clean={'BIAS_UP': 546, 'BIAS_DOWN': 739, 'BIAS_NEUTRAL': 6149}
2026-05-11 10:26:09,928 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 1585, 'BIAS_DOWN': 1189, 'BIAS_NEUTRAL': 19941}, 'dollar': {'BIAS_UP': 2140, 'BIAS_DOWN': 1769, 'BIAS_NEUTRAL': 30150}, 'gold': {'BIAS_UP': 1209, 'BIAS_DOWN': 568, 'BIAS_NEUTRAL': 10275}}
2026-05-11 10:26:09,929 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0698, 'bias_down_score': 0.0523}, 'dollar': {'bias_up_score': 0.0628, 'bias_down_score': 0.0519}, 'gold': {'bias_up_score': 0.1003, 'bias_down_score': 0.0471}}
2026-05-11 10:26:09,929 INFO Regime[4H mode=htf_bias] label distribution by year: {2016: {'BIAS_UP': 525, 'BIAS_DOWN': 617, 'BIAS_NEUTRAL': 7680}, 2017: {'BIAS_UP': 776, 'BIAS_DOWN': 315, 'BIAS_NEUTRAL': 8022}, 2018: {'BIAS_UP': 453, 'BIAS_DOWN': 753, 'BIAS_NEUTRAL': 7924}, 2019: {'BIAS_UP': 427, 'BIAS_DOWN': 481, 'BIAS_NEUTRAL': 8194}, 2020: {'BIAS_UP': 721, 'BIAS_DOWN': 181, 'BIAS_NEUTRAL': 8209}, 2021: {'BIAS_UP': 768, 'BIAS_DOWN': 506, 'BIAS_NEUTRAL': 7817}, 2022: {'BIAS_UP': 703, 'BIAS_DOWN': 561, 'BIAS_NEUTRAL': 7857}, 2023: {'BIAS_UP': 561, 'BIAS_DOWN': 112, 'BIAS_NEUTRAL': 4663}}
2026-05-11 10:26:09,929 INFO Regime[4H mode=htf_bias] score means by year: {2016: {'bias_up_score': 0.0595, 'bias_down_score': 0.0699}, 2017: {'bias_up_score': 0.0852, 'bias_down_score': 0.0346}, 2018: {'bias_up_score': 0.0496, 'bias_down_score': 0.0825}, 2019: {'bias_up_score': 0.0469, 'bias_down_score': 0.0528}, 2020: {'bias_up_score': 0.0791, 'bias_down_score': 0.0199}, 2021: {'bias_up_score': 0.0845, 'bias_down_score': 0.0557}, 2022: {'bias_up_score': 0.0771, 'bias_down_score': 0.0615}, 2023: {'bias_up_score': 0.1051, 'bias_down_score': 0.021}}
2026-05-11 10:26:09,982 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
2026-05-11 10:26:09,983 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
2026-05-11 10:26:09,984 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
2026-05-11 10:26:09,985 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 10:26:09,985 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 10:26:09,986 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
2026-05-11 10:26:10,003 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:26:10,007 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:26:10,008 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:26:10,008 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:26:10,008 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:26:10,009 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:26:10,375 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 166, 'BIAS_DOWN': 96, 'BIAS_NEUTRAL': 1319}  ambiguous=536 (total=1581) horizon=84
2026-05-11 10:26:10,378 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected XAUUSD — 1531 samples (group=gold) score_means={'bias_up_score': 0.1084, 'bias_down_score': 0.0627} labels={'BIAS_UP': 166, 'BIAS_DOWN': 96, 'BIAS_NEUTRAL': 1269} clean={'BIAS_UP': 166, 'BIAS_DOWN': 96, 'BIAS_NEUTRAL': 754}
2026-05-11 10:26:10,453 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:10,455 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:10,456 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:10,457 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:10,457 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:10,458 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:26:10,794 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 141, 'BIAS_DOWN': 68, 'BIAS_NEUTRAL': 1282}  ambiguous=504 (total=1491) horizon=84
2026-05-11 10:26:10,796 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURUSD — 1441 samples (group=dollar) score_means={'bias_up_score': 0.0978, 'bias_down_score': 0.0472} labels={'BIAS_UP': 141, 'BIAS_DOWN': 68, 'BIAS_NEUTRAL': 1232} clean={'BIAS_UP': 141, 'BIAS_DOWN': 68, 'BIAS_NEUTRAL': 757}
2026-05-11 10:26:10,867 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:10,869 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:10,870 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:10,870 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:10,871 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:10,872 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:26:11,227 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 173, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1231}  ambiguous=584 (total=1489) horizon=84
2026-05-11 10:26:11,230 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected USDJPY — 1439 samples (group=dollar) score_means={'bias_up_score': 0.1202, 'bias_down_score': 0.0591} labels={'BIAS_UP': 173, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1181} clean={'BIAS_UP': 173, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 621}
2026-05-11 10:26:11,313 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:11,315 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:11,316 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:11,316 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:11,317 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:11,318 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:26:11,676 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 125, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 1364}  ambiguous=540 (total=1494) horizon=84
2026-05-11 10:26:11,679 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0866, 'bias_down_score': 0.0035} labels={'BIAS_UP': 125, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 1314} clean={'BIAS_UP': 125, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 780}
2026-05-11 10:26:11,750 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:11,752 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:11,753 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:11,753 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:11,753 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:11,754 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:26:12,101 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 134, 'BIAS_DOWN': 11, 'BIAS_NEUTRAL': 1349}  ambiguous=512 (total=1494) horizon=84
2026-05-11 10:26:12,104 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0928, 'bias_down_score': 0.0069} labels={'BIAS_UP': 134, 'BIAS_DOWN': 10, 'BIAS_NEUTRAL': 1300} clean={'BIAS_UP': 134, 'BIAS_DOWN': 10, 'BIAS_NEUTRAL': 807}
2026-05-11 10:26:12,172 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:12,174 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:12,175 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:12,175 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:12,176 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:12,177 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:26:12,509 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 93, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 1304}  ambiguous=544 (total=1488) horizon=84
2026-05-11 10:26:12,512 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPUSD — 1438 samples (group=dollar) score_means={'bias_up_score': 0.0647, 'bias_down_score': 0.0633} labels={'BIAS_UP': 93, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 1254} clean={'BIAS_UP': 93, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 738}
2026-05-11 10:26:12,584 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 259, 'BIAS_DOWN': 15, 'BIAS_NEUTRAL': 2614}, 'dollar': {'BIAS_UP': 407, 'BIAS_DOWN': 244, 'BIAS_NEUTRAL': 3667}, 'gold': {'BIAS_UP': 166, 'BIAS_DOWN': 96, 'BIAS_NEUTRAL': 1269}}
2026-05-11 10:26:12,584 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0897, 'bias_down_score': 0.0052}, 'dollar': {'bias_up_score': 0.0943, 'bias_down_score': 0.0565}, 'gold': {'bias_up_score': 0.1084, 'bias_down_score': 0.0627}}
2026-05-11 10:26:12,584 INFO Regime[4H mode=htf_bias] label distribution by year: {2022: {'BIAS_UP': 276, 'BIAS_DOWN': 248, 'BIAS_NEUTRAL': 2877}, 2023: {'BIAS_UP': 556, 'BIAS_DOWN': 107, 'BIAS_NEUTRAL': 4673}}
2026-05-11 10:26:12,584 INFO Regime[4H mode=htf_bias] score means by year: {2022: {'bias_up_score': 0.0812, 'bias_down_score': 0.0729}, 2023: {'bias_up_score': 0.1042, 'bias_down_score': 0.0201}}
2026-05-11 10:26:12,657 INFO Regime phase HTF dataset build fold=train_all: 8.1s (train=68826 val=8737)
2026-05-11 10:26:12,658 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260511_102612
2026-05-11 10:26:12,866 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=36, n_classes=2)
2026-05-11 10:26:12,866 INFO Regime 4H/htf_bias warm start enabled from existing weights
2026-05-11 10:26:12,873 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=68826 val=8737 train_labels={'BIAS_UP': 4934, 'BIAS_DOWN': 3526, 'BIAS_NEUTRAL': 60366} val_labels={'BIAS_UP': 832, 'BIAS_DOWN': 355, 'BIAS_NEUTRAL': 7550}
2026-05-11 10:26:12,873 INFO RegimeClassifier[mode=htf_bias]: warm start HTF score head
2026-05-11 10:26:12,873 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-11 10:26:12,873 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 12.949, 'bias_down_score': 18.52}
2026-05-11 10:26:12,877 INFO RegimeClassifier[mode=htf_bias]: HTF balanced sampler — dir=8460 neutral=60366 dir_weight=5 => dir_frac_per_epoch≈41.2%
2026-05-11 10:26:16,326 INFO Regime HTF score epoch  1/50 — tr=7.0159 va=1.6610 acc=0.836 bal=0.411 threshold=0.35 margin=0.15 recall={'BIAS_UP': 0.159, 'BIAS_DOWN': 0.13, 'BIAS_NEUTRAL': 0.943} precision={'BIAS_UP': 0.298, 'BIAS_DOWN': 0.284, 'BIAS_NEUTRAL': 0.876}
2026-05-11 10:26:17,716 INFO Regime HTF score epoch  2/50 — tr=6.9546 va=1.6488 bal=0.412
2026-05-11 10:26:19,111 INFO Regime HTF score epoch  3/50 — tr=6.8543 va=1.6146 bal=0.429
2026-05-11 10:26:20,517 INFO Regime HTF score epoch  4/50 — tr=6.6770 va=1.5557 bal=0.416
2026-05-11 10:26:21,956 INFO Regime HTF score epoch  5/50 — tr=6.3931 va=1.4718 acc=0.831 bal=0.428 threshold=0.35 margin=0.40 recall={'BIAS_UP': 0.171, 'BIAS_DOWN': 0.18, 'BIAS_NEUTRAL': 0.934} precision={'BIAS_UP': 0.299, 'BIAS_DOWN': 0.281, 'BIAS_NEUTRAL': 0.878}
2026-05-11 10:26:23,357 INFO Regime HTF score epoch  6/50 — tr=6.0922 va=1.3850 bal=0.429
2026-05-11 10:26:24,788 INFO Regime HTF score epoch  7/50 — tr=5.6339 va=1.2887 bal=0.392
2026-05-11 10:26:26,192 INFO Regime HTF score epoch  8/50 — tr=5.2311 va=1.2032 bal=0.397
2026-05-11 10:26:27,589 INFO Regime HTF score epoch  9/50 — tr=4.8059 va=1.1189 bal=0.382
2026-05-11 10:26:28,996 INFO Regime HTF score epoch 10/50 — tr=4.4471 va=1.0485 acc=0.851 bal=0.377 threshold=0.80 margin=0.15 recall={'BIAS_UP': 0.05, 'BIAS_DOWN': 0.107, 'BIAS_NEUTRAL': 0.974} precision={'BIAS_UP': 0.316, 'BIAS_DOWN': 0.262, 'BIAS_NEUTRAL': 0.869}
2026-05-11 10:26:30,451 INFO Regime HTF score epoch 11/50 — tr=4.1529 va=0.9925 bal=0.418
2026-05-11 10:26:31,904 INFO Regime HTF score epoch 12/50 — tr=3.8789 va=0.9320 bal=0.375
2026-05-11 10:26:33,322 INFO Regime HTF score epoch 13/50 — tr=3.5572 va=0.8854 bal=0.373
2026-05-11 10:26:33,323 INFO Regime HTF score early stop at epoch 13
2026-05-11 10:26:34,574 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.350 margin=0.400 precision={'BIAS_UP': 0.299, 'BIAS_DOWN': 0.281, 'BIAS_NEUTRAL': 0.878} recall={'BIAS_UP': 0.171, 'BIAS_DOWN': 0.18, 'BIAS_NEUTRAL': 0.934} f1={'BIAS_UP': 0.217, 'BIAS_DOWN': 0.22, 'BIAS_NEUTRAL': 0.905} confusion=[[142, 0, 690], [0, 64, 291], [333, 164, 7053]] score_mae={'bias_up_score': 0.1704, 'bias_down_score': 0.1041} pred_share={'BIAS_UP': 0.0544, 'BIAS_DOWN': 0.0261, 'BIAS_NEUTRAL': 0.9195}
2026-05-11 10:26:34,575 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.299, 'BIAS_DOWN': 0.281, 'BIAS_NEUTRAL': 0.878} min_precision=0.500 recall={'BIAS_UP': 0.171, 'BIAS_DOWN': 0.18, 'BIAS_NEUTRAL': 0.934} min_recall=0.100 f1={'BIAS_UP': 0.217, 'BIAS_DOWN': 0.22, 'BIAS_NEUTRAL': 0.905} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_UP', 'BIAS_DOWN'] weak_recall=[] weak_f1=[] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-11 10:26:34,579 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 10:26:34,579 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 10:26:34,580 INFO Regime phase HTF train fold=train_all: 21.7s
2026-05-11 10:26:34,698 INFO Regime HTF complete fold=train_all: acc=0.831 bal=0.428 train=68826 val=8737 per_class={'BIAS_UP': 0.171, 'BIAS_DOWN': 0.18, 'BIAS_NEUTRAL': 0.934} precision={'BIAS_UP': 0.299, 'BIAS_DOWN': 0.281, 'BIAS_NEUTRAL': 0.878} threshold=0.350 margin=0.400
2026-05-11 10:26:34,699 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:26:34,916 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 546, 'BIAS_DOWN': 754, 'BIAS_NEUTRAL': 10102}  ambiguous=3944 (total=11402) horizon=84
2026-05-11 10:26:34,919 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on GBPUSD 4H:
{'BIAS_UP': 4.403225806451613, 'BIAS_DOWN': 5.755725190839694, 'BIAS_NEUTRAL': 39.4609375}
2026-05-11 10:26:34,924 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (all labels):
{'BIAS_UP': {'n': 546, 'mean': 0.00040267113448389793, 'mean_over_std': 0.18208067865763405}, 'BIAS_DOWN': {'n': 754, 'mean': -0.00047099607164125245, 'mean_over_std': -0.19477795734555267}, 'BIAS_NEUTRAL': {'n': 10101, 'mean': 2.6464098295242517e-06, 'mean_over_std': 0.0010012545608535832}}
2026-05-11 10:26:34,924 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 546, 'mean': 0.00040267113448389793, 'mean_over_std': 0.18208067865763405}, 'BIAS_DOWN': {'n': 754, 'mean': -0.00047099607164125245, 'mean_over_std': -0.19477795734555267}, 'BIAS_NEUTRAL': {'n': 6158, 'mean': 2.1496848003307296e-05, 'mean_over_std': 0.009079001472003705}}
2026-05-11 10:26:34,928 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-11 10:26:34,931 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:26:34,934 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:26:34,936 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:26:34,938 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:26:34,940 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:26:34,942 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:26:34,961 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:26:34,969 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:26:34,972 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:26:34,973 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:26:34,973 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:26:34,979 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:26:35,870 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected XAUUSD — 44227 samples (group=gold) score_means={'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}
2026-05-11 10:26:35,992 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:35,995 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:35,996 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:35,996 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:35,996 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:35,999 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:26:36,822 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURUSD — 43689 samples (group=dollar) score_means={'trend_score': 0.49, 'range_score': 0.236, 'chop_score': 0.4622, 'volatility_percentile': 0.3812, 'consolidation_score': 0.1862}
2026-05-11 10:26:36,940 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:36,942 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:36,943 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:36,943 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:36,944 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:36,946 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:26:37,762 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected USDJPY — 43686 samples (group=dollar) score_means={'trend_score': 0.4949, 'range_score': 0.2312, 'chop_score': 0.4613, 'volatility_percentile': 0.38, 'consolidation_score': 0.1931}
2026-05-11 10:26:37,879 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:37,882 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:37,882 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:37,883 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:37,883 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:37,885 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:26:38,709 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURJPY — 43688 samples (group=cross) score_means={'trend_score': 0.4888, 'range_score': 0.2345, 'chop_score': 0.4662, 'volatility_percentile': 0.3831, 'consolidation_score': 0.1882}
2026-05-11 10:26:38,833 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:38,835 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:38,836 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:38,836 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:38,837 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:38,839 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:26:39,668 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPJPY — 43678 samples (group=cross) score_means={'trend_score': 0.4878, 'range_score': 0.2348, 'chop_score': 0.4673, 'volatility_percentile': 0.3839, 'consolidation_score': 0.1904}
2026-05-11 10:26:39,789 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:39,792 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:39,793 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:39,793 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:39,793 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:39,796 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:26:40,638 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPUSD — 43676 samples (group=dollar) score_means={'trend_score': 0.4923, 'range_score': 0.2335, 'chop_score': 0.4611, 'volatility_percentile': 0.3832, 'consolidation_score': 0.1856}
2026-05-11 10:26:40,765 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2346, 'chop_score': 0.4667, 'volatility_percentile': 0.3835, 'consolidation_score': 0.1893}, 'dollar': {'trend_score': 0.4924, 'range_score': 0.2335, 'chop_score': 0.4615, 'volatility_percentile': 0.3815, 'consolidation_score': 0.1883}, 'gold': {'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}}
2026-05-11 10:26:40,765 INFO Regime[1H mode=ltf_behaviour] score means by year: {2016: {'trend_score': 0.495, 'range_score': 0.233, 'chop_score': 0.4607, 'volatility_percentile': 0.3776, 'consolidation_score': 0.192}, 2017: {'trend_score': 0.5017, 'range_score': 0.2328, 'chop_score': 0.4574, 'volatility_percentile': 0.3689, 'consolidation_score': 0.1942}, 2018: {'trend_score': 0.4886, 'range_score': 0.237, 'chop_score': 0.4655, 'volatility_percentile': 0.3947, 'consolidation_score': 0.1785}, 2019: {'trend_score': 0.4817, 'range_score': 0.236, 'chop_score': 0.4691, 'volatility_percentile': 0.3725, 'consolidation_score': 0.198}, 2020: {'trend_score': 0.4874, 'range_score': 0.2311, 'chop_score': 0.4674, 'volatility_percentile': 0.3848, 'consolidation_score': 0.1919}, 2021: {'trend_score': 0.4872, 'range_score': 0.2372, 'chop_score': 0.4672, 'volatility_percentile': 0.377, 'consolidation_score': 0.192}, 2022: {'trend_score': 0.4893, 'range_score': 0.2357, 'chop_score': 0.4647, 'volatility_percentile': 0.3962, 'consolidation_score': 0.1784}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 10:26:40,884 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
2026-05-11 10:26:40,886 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 10:26:40,887 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
2026-05-11 10:26:40,888 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 10:26:40,889 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
2026-05-11 10:26:40,891 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
2026-05-11 10:26:40,900 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:26:40,903 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:26:40,904 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:26:40,905 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:26:40,905 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:26:40,907 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:26:41,187 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected XAUUSD — 5117 samples (group=gold) score_means={'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}
2026-05-11 10:26:41,319 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:41,324 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:41,326 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:41,326 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:41,326 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:41,328 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:26:41,572 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURUSD — 5053 samples (group=dollar) score_means={'trend_score': 0.4883, 'range_score': 0.2413, 'chop_score': 0.463, 'volatility_percentile': 0.3837, 'consolidation_score': 0.1793}
2026-05-11 10:26:41,691 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:41,693 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:41,694 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:41,695 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:41,695 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:41,697 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:26:41,940 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected USDJPY — 5046 samples (group=dollar) score_means={'trend_score': 0.5085, 'range_score': 0.2259, 'chop_score': 0.4506, 'volatility_percentile': 0.3809, 'consolidation_score': 0.1888}
2026-05-11 10:26:42,059 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:42,061 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:42,062 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:42,062 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:42,063 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:42,066 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:26:42,311 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURJPY — 5053 samples (group=cross) score_means={'trend_score': 0.5023, 'range_score': 0.2261, 'chop_score': 0.4524, 'volatility_percentile': 0.3837, 'consolidation_score': 0.18}
2026-05-11 10:26:42,436 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:42,441 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:42,442 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:42,442 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:42,443 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:42,444 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:26:42,699 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPJPY — 5043 samples (group=cross) score_means={'trend_score': 0.4743, 'range_score': 0.2372, 'chop_score': 0.4716, 'volatility_percentile': 0.3777, 'consolidation_score': 0.1917}
2026-05-11 10:26:42,821 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:42,823 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:42,824 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:42,825 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:42,825 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:26:42,827 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:26:43,063 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPUSD — 5040 samples (group=dollar) score_means={'trend_score': 0.4843, 'range_score': 0.2371, 'chop_score': 0.4614, 'volatility_percentile': 0.377, 'consolidation_score': 0.1835}
2026-05-11 10:26:43,183 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2317, 'chop_score': 0.462, 'volatility_percentile': 0.3807, 'consolidation_score': 0.1858}, 'dollar': {'trend_score': 0.4937, 'range_score': 0.2348, 'chop_score': 0.4583, 'volatility_percentile': 0.3806, 'consolidation_score': 0.1839}, 'gold': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}}
2026-05-11 10:26:43,183 INFO Regime[1H mode=ltf_behaviour] score means by year: {2022: {'trend_score': 0.4905, 'range_score': 0.2364, 'chop_score': 0.4625, 'volatility_percentile': 0.385, 'consolidation_score': 0.1797}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 10:26:43,296 INFO Regime phase LTF dataset build fold=train_all: 8.4s (train=262644 val=30352)
2026-05-11 10:26:43,297 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260511_102643
2026-05-11 10:26:43,302 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=31, n_classes=5)
2026-05-11 10:26:43,302 INFO Regime 1H/ltf_behaviour warm start enabled from existing weights
2026-05-11 10:26:43,326 INFO RegimeClassifier[mode=ltf_behaviour]: warm start score head
2026-05-11 10:26:43,327 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-11 10:26:43,846 INFO Regime score epoch  1/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.0175, 'range_score': 0.0316, 'chop_score': 0.0196, 'volatility_percentile': 0.0144, 'consolidation_score': 0.0197}
2026-05-11 10:26:44,365 INFO Regime score epoch  2/50 — tr=0.0034 va=0.0008
2026-05-11 10:26:44,877 INFO Regime score epoch  3/50 — tr=0.0034 va=0.0008
2026-05-11 10:26:45,403 INFO Regime score epoch  4/50 — tr=0.0034 va=0.0008
2026-05-11 10:26:45,911 INFO Regime score epoch  5/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.0175, 'range_score': 0.0317, 'chop_score': 0.0192, 'volatility_percentile': 0.0144, 'consolidation_score': 0.0195}
2026-05-11 10:26:46,436 INFO Regime score epoch  6/50 — tr=0.0034 va=0.0008
2026-05-11 10:26:46,962 INFO Regime score epoch  7/50 — tr=0.0034 va=0.0008
2026-05-11 10:26:47,491 INFO Regime score epoch  8/50 — tr=0.0034 va=0.0008
2026-05-11 10:26:47,994 INFO Regime score epoch  9/50 — tr=0.0033 va=0.0008
2026-05-11 10:26:48,530 INFO Regime score epoch 10/50 — tr=0.0033 va=0.0008 mae={'trend_score': 0.0171, 'range_score': 0.0312, 'chop_score': 0.0192, 'volatility_percentile': 0.0142, 'consolidation_score': 0.0194}
2026-05-11 10:26:49,040 INFO Regime score epoch 11/50 — tr=0.0033 va=0.0008
2026-05-11 10:26:49,558 INFO Regime score epoch 12/50 — tr=0.0033 va=0.0008
2026-05-11 10:26:50,081 INFO Regime score epoch 13/50 — tr=0.0033 va=0.0008
2026-05-11 10:26:50,599 INFO Regime score epoch 14/50 — tr=0.0033 va=0.0008
2026-05-11 10:26:51,147 INFO Regime score epoch 15/50 — tr=0.0033 va=0.0008 mae={'trend_score': 0.0166, 'range_score': 0.0308, 'chop_score': 0.019, 'volatility_percentile': 0.0137, 'consolidation_score': 0.0191}
2026-05-11 10:26:51,689 INFO Regime score epoch 16/50 — tr=0.0033 va=0.0008
2026-05-11 10:26:52,195 INFO Regime score epoch 17/50 — tr=0.0033 va=0.0008
2026-05-11 10:26:52,712 INFO Regime score epoch 18/50 — tr=0.0033 va=0.0008
2026-05-11 10:26:53,225 INFO Regime score epoch 19/50 — tr=0.0032 va=0.0008
2026-05-11 10:26:53,754 INFO Regime score epoch 20/50 — tr=0.0032 va=0.0008 mae={'trend_score': 0.0164, 'range_score': 0.0305, 'chop_score': 0.0184, 'volatility_percentile': 0.0133, 'consolidation_score': 0.0191}
2026-05-11 10:26:54,270 INFO Regime score epoch 21/50 — tr=0.0032 va=0.0008
2026-05-11 10:26:54,802 INFO Regime score epoch 22/50 — tr=0.0032 va=0.0008
2026-05-11 10:26:55,322 INFO Regime score epoch 23/50 — tr=0.0032 va=0.0008
2026-05-11 10:26:55,849 INFO Regime score epoch 24/50 — tr=0.0032 va=0.0008
2026-05-11 10:26:56,370 INFO Regime score epoch 25/50 — tr=0.0032 va=0.0007 mae={'trend_score': 0.0161, 'range_score': 0.0304, 'chop_score': 0.0185, 'volatility_percentile': 0.0138, 'consolidation_score': 0.0187}
2026-05-11 10:26:56,893 INFO Regime score epoch 26/50 — tr=0.0032 va=0.0007
2026-05-11 10:26:57,424 INFO Regime score epoch 27/50 — tr=0.0032 va=0.0007
2026-05-11 10:26:57,933 INFO Regime score epoch 28/50 — tr=0.0032 va=0.0007
2026-05-11 10:26:58,452 INFO Regime score epoch 29/50 — tr=0.0032 va=0.0007
2026-05-11 10:26:58,969 INFO Regime score epoch 30/50 — tr=0.0032 va=0.0007 mae={'trend_score': 0.0159, 'range_score': 0.0303, 'chop_score': 0.0186, 'volatility_percentile': 0.0134, 'consolidation_score': 0.0192}
2026-05-11 10:26:59,500 INFO Regime score epoch 31/50 — tr=0.0032 va=0.0007
2026-05-11 10:27:00,019 INFO Regime score epoch 32/50 — tr=0.0032 va=0.0007
2026-05-11 10:27:00,522 INFO Regime score epoch 33/50 — tr=0.0032 va=0.0007
2026-05-11 10:27:01,038 INFO Regime score epoch 34/50 — tr=0.0032 va=0.0007
2026-05-11 10:27:01,578 INFO Regime score epoch 35/50 — tr=0.0032 va=0.0007 mae={'trend_score': 0.0159, 'range_score': 0.0302, 'chop_score': 0.0183, 'volatility_percentile': 0.014, 'consolidation_score': 0.019}
2026-05-11 10:27:02,091 INFO Regime score epoch 36/50 — tr=0.0032 va=0.0007
2026-05-11 10:27:02,630 INFO Regime score epoch 37/50 — tr=0.0032 va=0.0007
2026-05-11 10:27:03,167 INFO Regime score epoch 38/50 — tr=0.0032 va=0.0007
2026-05-11 10:27:03,684 INFO Regime score epoch 39/50 — tr=0.0032 va=0.0007
2026-05-11 10:27:04,189 INFO Regime score epoch 40/50 — tr=0.0032 va=0.0007 mae={'trend_score': 0.016, 'range_score': 0.03, 'chop_score': 0.0182, 'volatility_percentile': 0.0133, 'consolidation_score': 0.0185}
2026-05-11 10:27:04,707 INFO Regime score epoch 41/50 — tr=0.0032 va=0.0007
2026-05-11 10:27:05,211 INFO Regime score epoch 42/50 — tr=0.0032 va=0.0007
2026-05-11 10:27:05,724 INFO Regime score epoch 43/50 — tr=0.0031 va=0.0007
2026-05-11 10:27:06,235 INFO Regime score epoch 44/50 — tr=0.0032 va=0.0007
2026-05-11 10:27:06,775 INFO Regime score epoch 45/50 — tr=0.0032 va=0.0007 mae={'trend_score': 0.0161, 'range_score': 0.0299, 'chop_score': 0.018, 'volatility_percentile': 0.0129, 'consolidation_score': 0.0185}
2026-05-11 10:27:07,308 INFO Regime score epoch 46/50 — tr=0.0032 va=0.0007
2026-05-11 10:27:07,840 INFO Regime score epoch 47/50 — tr=0.0032 va=0.0007
2026-05-11 10:27:08,341 INFO Regime score epoch 48/50 — tr=0.0031 va=0.0007
2026-05-11 10:27:08,872 INFO Regime score epoch 49/50 — tr=0.0032 va=0.0007
2026-05-11 10:27:09,385 INFO Regime score epoch 50/50 — tr=0.0032 va=0.0007 mae={'trend_score': 0.0158, 'range_score': 0.0301, 'chop_score': 0.018, 'volatility_percentile': 0.0131, 'consolidation_score': 0.0185}
2026-05-11 10:27:09,405 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0159, 'range_score': 0.0301, 'chop_score': 0.0179, 'volatility_percentile': 0.0131, 'consolidation_score': 0.0185} mse={'trend_score': 0.00044, 'range_score': 0.0015, 'chop_score': 0.00052, 'volatility_percentile': 0.00033, 'consolidation_score': 0.00078} corr={'trend_score': 0.9956, 'range_score': 0.9636, 'chop_score': 0.993, 'volatility_percentile': 0.9967, 'consolidation_score': 0.9918} pred_std={'trend_score': 0.2221, 'range_score': 0.1327, 'chop_score': 0.1832, 'volatility_percentile': 0.2201, 'consolidation_score': 0.2156} target_std={'trend_score': 0.2219, 'range_score': 0.1433, 'chop_score': 0.1888, 'volatility_percentile': 0.2165, 'consolidation_score': 0.2179}
2026-05-11 10:27:09,719 INFO RegimeClassifier[mode=ltf_behaviour] train group diagnostics={'dollar': {'n': 16632, 'score_mae': {'trend_score': 0.0153, 'range_score': 0.03, 'chop_score': 0.0177, 'volatility_percentile': 0.0126, 'consolidation_score': 0.0189}, 'target_score_mean': {'trend_score': 0.491, 'range_score': 0.2347, 'chop_score': 0.4613, 'volatility_percentile': 0.3824, 'consolidation_score': 0.186}, 'pred_score_mean': {'trend_score': 0.49, 'range_score': 0.2344, 'chop_score': 0.4627, 'volatility_percentile': 0.3808, 'consolidation_score': 0.1838}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3551, 75, 0, 4, 0, 0, 149], [3, 99, 0, 0, 0, 4, 4], [0, 0, 190, 10, 65, 0, 195], [1, 0, 4, 567, 40, 0, 77], [0, 0, 20, 15, 3142, 1, 138], [0, 17, 0, 0, 7, 69, 35], [105, 12, 63, 54, 76, 6, 7834]]}, 'gold': {'n': 8420, 'score_mae': {'trend_score': 0.0153, 'range_score': 0.0307, 'chop_score': 0.018, 'volatility_percentile': 0.0133, 'consolidation_score': 0.0193}, 'target_score_mean': {'trend_score': 0.4883, 'range_score': 0.2355, 'chop_score': 0.4652, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}, 'pred_score_mean': {'trend_score': 0.4879, 'range_score': 0.2351, 'chop_score': 0.4661, 'volatility_percentile': 0.3747, 'consolidation_score': 0.1897}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1785, 38, 0, 0, 0, 0, 62], [3, 51, 0, 0, 0, 1, 1], [0, 0, 95, 12, 32, 0, 105], [0, 0, 2, 348, 21, 0, 45], [0, 0, 14, 17, 1601, 0, 72], [0, 12, 0, 0, 4, 47, 18], [51, 4, 44, 19, 58, 0, 3858]]}, 'jpy': {'n': 24948, 'score_mae': {'trend_score': 0.0154, 'range_score': 0.0301, 'chop_score': 0.0176, 'volatility_percentile': 0.0136, 'consolidation_score': 0.0187}, 'target_score_mean': {'trend_score': 0.4904, 'range_score': 0.2338, 'chop_score': 0.4653, 'volatility_percentile': 0.3826, 'consolidation_score': 0.1905}, 'pred_score_mean': {'trend_score': 0.489, 'range_score': 0.2334, 'chop_score': 0.4662, 'volatility_percentile': 0.3804, 'consolidation_score': 0.1878}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[5296, 145, 0, 3, 0, 0, 202], [5, 169, 0, 0, 0, 5, 8], [0, 0, 236, 18, 102, 0, 291], [2, 0, 3, 1102, 82, 0, 125], [0, 0, 26, 45, 4825, 0, 219], [0, 30, 0, 0, 15, 103, 75], [163, 11, 88, 79, 160, 7, 11308]]}}
2026-05-11 10:27:09,888 INFO RegimeClassifier[mode=ltf_behaviour] validation group diagnostics={'dollar': {'n': 10093, 'score_mae': {'trend_score': 0.0162, 'range_score': 0.0305, 'chop_score': 0.0181, 'volatility_percentile': 0.0125, 'consolidation_score': 0.0182}, 'target_score_mean': {'trend_score': 0.4863, 'range_score': 0.2392, 'chop_score': 0.4622, 'volatility_percentile': 0.3804, 'consolidation_score': 0.1814}, 'pred_score_mean': {'trend_score': 0.4852, 'range_score': 0.2374, 'chop_score': 0.4646, 'volatility_percentile': 0.3784, 'consolidation_score': 0.1799}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[2289, 27, 0, 2, 0, 0, 94], [3, 46, 0, 0, 0, 3, 1], [0, 0, 102, 7, 56, 0, 151], [0, 0, 1, 338, 32, 0, 52], [0, 0, 13, 15, 1958, 0, 64], [0, 10, 0, 0, 3, 38, 26], [47, 5, 29, 43, 66, 3, 4569]]}, 'gold': {'n': 5117, 'score_mae': {'trend_score': 0.0152, 'range_score': 0.0294, 'chop_score': 0.0179, 'volatility_percentile': 0.0132, 'consolidation_score': 0.0192}, 'target_score_mean': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}, 'pred_score_mean': {'trend_score': 0.4957, 'range_score': 0.2317, 'chop_score': 0.4586, 'volatility_percentile': 0.3792, 'consolidation_score': 0.1809}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1107, 14, 0, 0, 0, 0, 46], [3, 29, 0, 0, 0, 2, 1], [0, 0, 61, 3, 18, 0, 89], [0, 0, 2, 231, 10, 0, 12], [0, 0, 6, 8, 830, 0, 43], [0, 6, 0, 0, 4, 26, 14], [44, 2, 25, 27, 39, 1, 2414]]}, 'jpy': {'n': 15142, 'score_mae': {'trend_score': 0.0159, 'range_score': 0.0301, 'chop_score': 0.0178, 'volatility_percentile': 0.0135, 'consolidation_score': 0.0185}, 'target_score_mean': {'trend_score': 0.495, 'range_score': 0.2297, 'chop_score': 0.4582, 'volatility_percentile': 0.3808, 'consolidation_score': 0.1868}, 'pred_score_mean': {'trend_score': 0.4929, 'range_score': 0.2287, 'chop_score': 0.4602, 'volatility_percentile': 0.3786, 'consolidation_score': 0.1844}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3325, 63, 0, 1, 0, 0, 129], [4, 98, 0, 0, 0, 7, 6], [0, 0, 139, 13, 59, 0, 173], [2, 0, 3, 704, 43, 0, 75], [0, 0, 19, 23, 2647, 0, 128], [0, 15, 0, 0, 8, 62, 37], [88, 9, 61, 47, 92, 8, 7054]]}}
2026-05-11 10:27:09,894 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-11 10:27:09,894 INFO Regime phase LTF train fold=train_all: 26.6s
2026-05-11 10:27:10,013 INFO Regime LTF complete fold=train_all: score_accuracy=0.981, train=262644 val=30352 mae={'trend_score': 0.0159, 'range_score': 0.0301, 'chop_score': 0.0179, 'volatility_percentile': 0.0131, 'consolidation_score': 0.0185}
2026-05-11 10:27:10,015 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:27:10,378 INFO Regime[1H mode=ltf_behaviour fold=latest] LTF score diagnostics on GBPUSD:
{'trend_score': {'mean': 0.4921, 'q10': 0.1918, 'q50': 0.4877, 'q90': 0.8002}, 'range_score': {'mean': 0.2337, 'q10': 0.0513, 'q50': 0.2125, 'q90': 0.4349}, 'chop_score': {'mean': 0.4612, 'q10': 0.2148, 'q50': 0.4503, 'q90': 0.7255}, 'volatility_percentile': {'mean': 0.3832, 'q10': 0.1018, 'q50': 0.3704, 'q90': 0.6823}, 'consolidation_score': {'mean': 0.1854, 'q10': 0.0, 'q50': 0.1185, 'q90': 0.506}}
2026-05-11 10:27:10,382 INFO Regime retrain total: 66.2s (370559 train+val samples)
2026-05-11 10:27:10,386 INFO Retrain complete. Total wall-clock: 66.2s
  DONE  Retrain regime [pre-R2 retrain]

=== Round 2: BLIND backtest on test window (unseen 2yr) ===
  START Round 2 - Blind backtest (test)
2026-05-11 10:27:12,159 INFO === STEP 6: BACKTEST (round2) ===
2026-05-11 10:27:12,161 INFO BT_WINDOW=round2 — BLIND backtest: 2023-08-07 → 2025-08-05 (test set)
2026-05-11 10:27:12,161 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-05-11 10:27:12,161 INFO Round 2 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
2026-05-11 10:27:12,162 INFO Round 2 — QualityScorer hard gate disabled; set BACKTEST_ENABLE_QUALITY_GATE=1 to enable it
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
2026-05-11 10:28:30,631 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
2026-05-11 10:28:30,696 WARNING _build_sequence_df: HTF frame 5M filled 317 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:812: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:812: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 10:28:31,135 WARNING _build_sequence_df: HTF frame 5M filled 317 warmup/alignment gaps with 0.000
2026-05-11 10:28:31,136 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:814: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:814: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:818: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:818: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 10:28:31,188 WARNING _build_sequence_df: HTF frame 5M filled 341 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:812: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:822: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:822: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:826: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:826: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 10:28:31,295 WARNING _build_sequence_df: HTF frame 5M filled 341 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:830: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:814: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:830: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:834: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:834: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:836: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:836: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:818: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:822: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:826: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 10:28:31,481 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:830: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:812: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:834: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:836: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 10:28:31,535 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:814: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:818: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:822: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:826: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:830: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:834: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:836: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
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
2026-05-11 10:28:40,126 WARNING _build_sequence_df: HTF frame 5M filled 317 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:812: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 10:28:40,277 WARNING _build_sequence_df: HTF frame 5M filled 317 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:814: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:818: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:822: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:826: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:830: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:834: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:836: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 10:28:40,362 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:812: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 10:28:40,396 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:814: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:818: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:822: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:826: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:830: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:834: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:836: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260511_102714.json
Trader                                   Trades      WR    PF*  Return*    ExpR   TP1%   TP2%     DD*  Sharpe*  Sortino     IC  Verdict
-----------------------------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)                4   0.0%   0.00   -4.0%  -1.001  0.0%  0.0%   4.0% -15890380336.99    -1.00  0.000     FAIL
  FAILED rules: min_trades, positive_expectancy, profit_factor_min_1_25, sharpe_positive, sortino_positive, win_rate_above_breakeven, t_stat_above_1_5, sharpe_ci_positive
  monthly R: 2024-07=-1.00  2024-11=-1.00  2025-04=-2.00
  MonteCarlo P95 DD=4.0%  P10 equity=9,600  t=0.00 (p=1.000)  Sharpe CI=[-15890380336.99, -15890380336.99]  streak=4
  gate_diagnostics: bars=280782 no_signal=206137 quality_block=0 session_skip=74641 density=0 pm_reject=0 daily_skip=0 cooldown=0 daily_halt_events=0 enforce_daily_halt=False
  no_signal_reasons: no_trade_uncertain=71542, weak_gru_direction=54575, gru_expected_r_below_threshold=45064, no_trade_chop=17258, no_trade_extreme_vol=13991, htf_low_regime_confidence=3007

Calibration Summary:
  all          [N/A] Insufficient data: 4 samples
  ml_trader    [N/A] Insufficient data: 4 samples
2026-05-11 10:29:26,014 INFO Round 2 backtest — 4 trades | avg WR=0.0% | avg PF=0.00 | avg Sharpe=-15890380336.99
2026-05-11 10:29:26,014 INFO   ml_trader: 4 trades | WR=0.0% | fixed PF=0.00 | Return=-4.0% | ExpR=-1.001 | DD=4.0% | Sharpe=-15890380336.99
2026-05-11 10:29:26,014 INFO   ml_trader gate_diagnostics: bars=280782 no_signal=206137 quality_block=0 session_skip=74641 density=0 pm_reject=0
2026-05-11 10:29:26,014 INFO   ml_trader no_signal_reasons: {'no_trade_chop': 17258, 'no_trade_uncertain': 71542, 'weak_gru_direction': 54575, 'gru_expected_r_below_threshold': 45064, 'htf_low_regime_confidence': 3007, 'no_trade_extreme_vol': 13991, 'tradeability_direction_conflict': 396, 'wait_pullback': 251, 'trend_structure_missing': 53}
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 4
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (4 rows)
2026-05-11 10:29:26,234 INFO Round 2: wrote 4 journal entries (total in file: 8)
  DONE  Round 2 - Blind backtest (test)
  Saved Round 2 result → round2_summary.json
  Journal after Round 2: 8 entries

  SKIP  Round 2 Quality+RL retrain — blind test journal kept untouched

=== Round 3: Incremental retrain ===
  START Retrain gru [train-split retrain]
2026-05-11 10:29:26,795 INFO retrain environment: KAGGLE
2026-05-11 10:29:28,399 INFO Device: CUDA (2 GPU(s))
2026-05-11 10:29:28,412 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 10:29:28,412 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 10:29:28,412 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 10:29:28,412 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 10:29:28,412 INFO Retrain data split: train
2026-05-11 10:29:28,413 INFO Retrain rolling fold selector: latest
2026-05-11 10:29:28,413 INFO === GRU-LSTM retrain (timeframes: ['15M']) ===
2026-05-11 10:29:28,558 INFO NumExpr defaulting to 4 threads.
2026-05-11 10:29:28,747 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-11 10:29:28,747 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 10:29:28,747 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 10:29:29,632 INFO GRULSTMPredictor: loaded short R isotonic calibrator from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/r_isotonic_short.pkl
2026-05-11 10:29:29,632 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-05-11 10:29:29,632 INFO GRU multi-symbol training (Kaggle mode): 6 symbols × ['15M']
2026-05-11 10:29:29,635 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260511_102929
2026-05-11 10:29:29,640 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-05-11 10:29:29,641 INFO Deleted stale GRU artifact for cold start: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:29:29,641 INFO Deleted stale GRU artifact for cold start: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/r_isotonic_short.pkl
2026-05-11 10:29:29,641 INFO GRU warm start disabled by default; set GRU_ALLOW_WARM_START=1 to reuse compatible weights
2026-05-11 10:29:29,903 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:29:29,936 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:29:29,953 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:29:29,965 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:29:30,041 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 10:29:30,047 INFO Loaded XAUUSD/15M split=train fold=latest: 176438 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:29:30,628 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:29:30,649 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:29:30,665 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:29:30,673 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:29:30,716 INFO Loaded EURUSD/15M split=train fold=latest: 174951 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:29:31,324 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:29:31,349 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:29:31,369 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:29:31,379 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:29:31,420 INFO Loaded USDJPY/15M split=train fold=latest: 174944 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:29:31,976 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:29:31,998 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:29:32,014 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:29:32,022 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:29:32,061 INFO Loaded EURJPY/15M split=train fold=latest: 174948 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:29:32,599 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:29:32,620 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:29:32,637 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:29:32,647 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:29:32,687 INFO Loaded GBPJPY/15M split=train fold=latest: 174902 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:29:33,241 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:29:33,261 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:29:33,275 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:29:33,284 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:29:33,324 INFO Loaded GBPUSD/15M split=train fold=latest: 174899 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:29:33,771 INFO train_multi: 6 segments, ~971854 total bars
2026-05-11 10:29:33,778 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-05-11 10:29:33,778 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-05-11 10:29:33,778 INFO train_multi: training ALL 6 segments across TFs ['15M'] in one combined pass
2026-05-11 10:29:33,779 INFO train_multi: building combined dataset for TF=ALL (6 segments)
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
2026-05-11 10:29:43,298 INFO train_multi TF=ALL: 971494 sequences across 6 segments
2026-05-11 10:29:43,298 INFO train_multi TF=ALL: estimated peak RAM = 21312 MB (train=419996 calib=60000 val=120002 n_feat=74 seq_len=60)
2026-05-11 10:29:43,298 WARNING train_multi TF=ALL: trimming to fit RAM budget — new train=394144 calib=56306 val=112612 (20000 MB est)
2026-05-11 10:29:45,654 INFO train_multi TF=ALL: train=394144 calib=56306 val=112612 (10009 MB tensors)
2026-05-11 10:29:52,492 INFO train_multi TF=ALL: structural bar weighting — 252452 structural bars (64.1%) weight=15.0 structural_only=0
2026-05-11 10:29:53,609 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=25, min_epochs=45)
2026-05-11 10:30:11,692 INFO train_multi TF=ALL epoch 1/100 train=2.3406 val=2.3444 r_mae=0.979 pos_r_acc=0.455 side_acc=0.507 r_n=161888
2026-05-11 10:30:11,696 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:30:11,697 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:30:11,697 INFO train_multi TF=ALL: new best val=2.3444 r_mae=0.9787 — saved
2026-05-11 10:30:11,701 INFO train_multi TF=ALL: new best r_mae=0.9787 — saved rmae checkpoint
2026-05-11 10:30:27,649 INFO train_multi TF=ALL epoch 2/100 train=2.3364 val=2.3386 r_mae=0.974 pos_r_acc=0.495 side_acc=0.507 r_n=161888
2026-05-11 10:30:27,654 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:30:27,654 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:30:27,654 INFO train_multi TF=ALL: new best val=2.3386 r_mae=0.9741 — saved
2026-05-11 10:30:27,659 INFO train_multi TF=ALL: new best r_mae=0.9741 — saved rmae checkpoint
2026-05-11 10:30:43,570 INFO train_multi TF=ALL epoch 3/100 train=2.3311 val=2.3309 r_mae=0.967 pos_r_acc=0.545 side_acc=0.494 r_n=161888
2026-05-11 10:30:43,575 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:30:43,575 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:30:43,575 INFO train_multi TF=ALL: new best val=2.3309 r_mae=0.9667 — saved
2026-05-11 10:30:43,580 INFO train_multi TF=ALL: new best r_mae=0.9667 — saved rmae checkpoint
2026-05-11 10:30:59,728 INFO train_multi TF=ALL epoch 4/100 train=2.3290 val=2.3299 r_mae=0.966 pos_r_acc=0.545 side_acc=0.494 r_n=161888
2026-05-11 10:30:59,734 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:30:59,734 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:30:59,734 INFO train_multi TF=ALL: new best val=2.3299 r_mae=0.9662 — saved
2026-05-11 10:30:59,738 INFO train_multi TF=ALL: new best r_mae=0.9662 — saved rmae checkpoint
2026-05-11 10:31:15,775 INFO train_multi TF=ALL epoch 5/100 train=2.3282 val=2.3285 r_mae=0.966 pos_r_acc=0.545 side_acc=0.498 r_n=161888
2026-05-11 10:31:15,780 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:31:15,780 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:31:15,780 INFO train_multi TF=ALL: new best val=2.3285 r_mae=0.9657 — saved
2026-05-11 10:31:15,785 INFO train_multi TF=ALL: new best r_mae=0.9657 — saved rmae checkpoint
2026-05-11 10:31:31,923 INFO train_multi TF=ALL epoch 6/100 train=2.3268 val=2.3259 r_mae=0.965 pos_r_acc=0.545 side_acc=0.522 r_n=161888
2026-05-11 10:31:31,929 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:31:31,929 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:31:31,929 INFO train_multi TF=ALL: new best val=2.3259 r_mae=0.9653 — saved
2026-05-11 10:31:31,934 INFO train_multi TF=ALL: new best r_mae=0.9653 — saved rmae checkpoint
2026-05-11 10:31:48,517 INFO train_multi TF=ALL epoch 7/100 train=2.3237 val=2.3222 r_mae=0.965 pos_r_acc=0.545 side_acc=0.520 r_n=161888
2026-05-11 10:31:48,522 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:31:48,522 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:31:48,522 INFO train_multi TF=ALL: new best val=2.3222 r_mae=0.9647 — saved
2026-05-11 10:31:48,527 INFO train_multi TF=ALL: new best r_mae=0.9647 — saved rmae checkpoint
2026-05-11 10:32:05,335 INFO train_multi TF=ALL epoch 8/100 train=2.3201 val=2.3203 r_mae=0.964 pos_r_acc=0.545 side_acc=0.521 r_n=161888
2026-05-11 10:32:05,346 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:32:05,346 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:32:05,346 INFO train_multi TF=ALL: new best val=2.3203 r_mae=0.9638 — saved
2026-05-11 10:32:05,350 INFO train_multi TF=ALL: new best r_mae=0.9638 — saved rmae checkpoint
2026-05-11 10:32:22,173 INFO train_multi TF=ALL epoch 9/100 train=2.3174 val=2.3180 r_mae=0.963 pos_r_acc=0.545 side_acc=0.524 r_n=161888
2026-05-11 10:32:22,178 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:32:22,178 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:32:22,179 INFO train_multi TF=ALL: new best val=2.3180 r_mae=0.9631 — saved
2026-05-11 10:32:22,183 INFO train_multi TF=ALL: new best r_mae=0.9631 — saved rmae checkpoint
2026-05-11 10:32:38,739 INFO train_multi TF=ALL epoch 10/100 train=2.3163 val=2.3160 r_mae=0.963 pos_r_acc=0.545 side_acc=0.528 r_n=161888
2026-05-11 10:32:38,744 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:32:38,744 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:32:38,744 INFO train_multi TF=ALL: new best val=2.3160 r_mae=0.9626 — saved
2026-05-11 10:32:38,749 INFO train_multi TF=ALL: new best r_mae=0.9626 — saved rmae checkpoint
2026-05-11 10:32:54,799 INFO train_multi TF=ALL epoch 11/100 train=2.3135 val=2.3155 r_mae=0.962 pos_r_acc=0.547 side_acc=0.528 r_n=161888
2026-05-11 10:32:54,809 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:32:54,809 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:32:54,809 INFO train_multi TF=ALL: new best val=2.3155 r_mae=0.9617 — saved
2026-05-11 10:32:54,813 INFO train_multi TF=ALL: new best r_mae=0.9617 — saved rmae checkpoint
2026-05-11 10:33:10,745 INFO train_multi TF=ALL epoch 12/100 train=2.3118 val=2.3135 r_mae=0.961 pos_r_acc=0.548 side_acc=0.532 r_n=161888
2026-05-11 10:33:10,750 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:33:10,751 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:33:10,751 INFO train_multi TF=ALL: new best val=2.3135 r_mae=0.9605 — saved
2026-05-11 10:33:10,755 INFO train_multi TF=ALL: new best r_mae=0.9605 — saved rmae checkpoint
2026-05-11 10:33:26,711 INFO train_multi TF=ALL epoch 13/100 train=2.3077 val=2.3117 r_mae=0.959 pos_r_acc=0.551 side_acc=0.534 r_n=161888
2026-05-11 10:33:26,717 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:33:26,717 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:33:26,717 INFO train_multi TF=ALL: new best val=2.3117 r_mae=0.9587 — saved
2026-05-11 10:33:26,721 INFO train_multi TF=ALL: new best r_mae=0.9587 — saved rmae checkpoint
2026-05-11 10:33:42,541 INFO train_multi TF=ALL epoch 14/100 train=2.3024 val=2.3034 r_mae=0.954 pos_r_acc=0.561 side_acc=0.537 r_n=161888
2026-05-11 10:33:42,546 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:33:42,546 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:33:42,546 INFO train_multi TF=ALL: new best val=2.3034 r_mae=0.9538 — saved
2026-05-11 10:33:42,550 INFO train_multi TF=ALL: new best r_mae=0.9538 — saved rmae checkpoint
2026-05-11 10:33:58,383 INFO train_multi TF=ALL epoch 15/100 train=2.2940 val=2.2909 r_mae=0.949 pos_r_acc=0.571 side_acc=0.544 r_n=161888
2026-05-11 10:33:58,393 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:33:58,393 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:33:58,393 INFO train_multi TF=ALL: new best val=2.2909 r_mae=0.9486 — saved
2026-05-11 10:33:58,398 INFO train_multi TF=ALL: new best r_mae=0.9486 — saved rmae checkpoint
2026-05-11 10:34:14,378 INFO train_multi TF=ALL epoch 16/100 train=2.2843 val=2.2782 r_mae=0.944 pos_r_acc=0.579 side_acc=0.547 r_n=161888
2026-05-11 10:34:14,383 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:34:14,383 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:34:14,383 INFO train_multi TF=ALL: new best val=2.2782 r_mae=0.9438 — saved
2026-05-11 10:34:14,388 INFO train_multi TF=ALL: new best r_mae=0.9438 — saved rmae checkpoint
2026-05-11 10:34:30,370 INFO train_multi TF=ALL epoch 17/100 train=2.2746 val=2.2677 r_mae=0.940 pos_r_acc=0.582 side_acc=0.558 r_n=161888
2026-05-11 10:34:30,375 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:34:30,375 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:34:30,375 INFO train_multi TF=ALL: new best val=2.2677 r_mae=0.9400 — saved
2026-05-11 10:34:30,379 INFO train_multi TF=ALL: new best r_mae=0.9400 — saved rmae checkpoint
2026-05-11 10:34:46,392 INFO train_multi TF=ALL epoch 18/100 train=2.2635 val=2.2607 r_mae=0.934 pos_r_acc=0.588 side_acc=0.557 r_n=161888
2026-05-11 10:34:46,398 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:34:46,398 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:34:46,398 INFO train_multi TF=ALL: new best val=2.2607 r_mae=0.9340 — saved
2026-05-11 10:34:46,402 INFO train_multi TF=ALL: new best r_mae=0.9340 — saved rmae checkpoint
2026-05-11 10:35:02,231 INFO train_multi TF=ALL epoch 19/100 train=2.2597 val=2.2585 r_mae=0.932 pos_r_acc=0.589 side_acc=0.557 r_n=161888
2026-05-11 10:35:02,236 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:35:02,236 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:35:02,236 INFO train_multi TF=ALL: new best val=2.2585 r_mae=0.9325 — saved
2026-05-11 10:35:02,240 INFO train_multi TF=ALL: new best r_mae=0.9325 — saved rmae checkpoint
2026-05-11 10:35:18,642 INFO train_multi TF=ALL epoch 20/100 train=2.2524 val=2.2529 r_mae=0.930 pos_r_acc=0.591 side_acc=0.562 r_n=161888
2026-05-11 10:35:18,648 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:35:18,648 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:35:18,648 INFO train_multi TF=ALL: new best val=2.2529 r_mae=0.9297 — saved
2026-05-11 10:35:18,652 INFO train_multi TF=ALL: new best r_mae=0.9297 — saved rmae checkpoint
2026-05-11 10:35:34,803 INFO train_multi TF=ALL epoch 21/100 train=2.2494 val=2.2495 r_mae=0.929 pos_r_acc=0.591 side_acc=0.564 r_n=161888
2026-05-11 10:35:34,815 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:35:34,815 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:35:34,815 INFO train_multi TF=ALL: new best val=2.2495 r_mae=0.9287 — saved
2026-05-11 10:35:34,820 INFO train_multi TF=ALL: new best r_mae=0.9287 — saved rmae checkpoint
2026-05-11 10:35:50,759 INFO train_multi TF=ALL epoch 22/100 train=2.2418 val=2.2495 r_mae=0.928 pos_r_acc=0.590 side_acc=0.560 r_n=161888
2026-05-11 10:35:50,764 INFO train_multi TF=ALL: new best r_mae=0.9275 — saved rmae checkpoint
2026-05-11 10:36:06,686 INFO train_multi TF=ALL epoch 23/100 train=2.2402 val=2.2425 r_mae=0.926 pos_r_acc=0.596 side_acc=0.566 r_n=161888
2026-05-11 10:36:06,691 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:36:06,691 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:36:06,691 INFO train_multi TF=ALL: new best val=2.2425 r_mae=0.9260 — saved
2026-05-11 10:36:06,695 INFO train_multi TF=ALL: new best r_mae=0.9260 — saved rmae checkpoint
2026-05-11 10:36:22,722 INFO train_multi TF=ALL epoch 24/100 train=2.2358 val=2.2438 r_mae=0.923 pos_r_acc=0.593 side_acc=0.565 r_n=161888
2026-05-11 10:36:22,727 INFO train_multi TF=ALL: new best r_mae=0.9228 — saved rmae checkpoint
2026-05-11 10:36:38,592 INFO train_multi TF=ALL epoch 25/100 train=2.2331 val=2.2403 r_mae=0.921 pos_r_acc=0.596 side_acc=0.567 r_n=161888
2026-05-11 10:36:38,597 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:36:38,597 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:36:38,597 INFO train_multi TF=ALL: new best val=2.2403 r_mae=0.9210 — saved
2026-05-11 10:36:38,601 INFO train_multi TF=ALL: new best r_mae=0.9210 — saved rmae checkpoint
2026-05-11 10:36:54,673 INFO train_multi TF=ALL epoch 26/100 train=2.2285 val=2.2342 r_mae=0.921 pos_r_acc=0.599 side_acc=0.569 r_n=161888
2026-05-11 10:36:54,678 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:36:54,678 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:36:54,678 INFO train_multi TF=ALL: new best val=2.2342 r_mae=0.9206 — saved
2026-05-11 10:36:54,682 INFO train_multi TF=ALL: new best r_mae=0.9206 — saved rmae checkpoint
2026-05-11 10:37:10,557 INFO train_multi TF=ALL epoch 27/100 train=2.2215 val=2.2318 r_mae=0.919 pos_r_acc=0.601 side_acc=0.571 r_n=161888
2026-05-11 10:37:10,564 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:37:10,564 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:37:10,564 INFO train_multi TF=ALL: new best val=2.2318 r_mae=0.9191 — saved
2026-05-11 10:37:10,571 INFO train_multi TF=ALL: new best r_mae=0.9191 — saved rmae checkpoint
2026-05-11 10:37:26,591 INFO train_multi TF=ALL epoch 28/100 train=2.2171 val=2.2262 r_mae=0.914 pos_r_acc=0.604 side_acc=0.576 r_n=161888
2026-05-11 10:37:26,596 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:37:26,596 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:37:26,597 INFO train_multi TF=ALL: new best val=2.2262 r_mae=0.9143 — saved
2026-05-11 10:37:26,601 INFO train_multi TF=ALL: new best r_mae=0.9143 — saved rmae checkpoint
2026-05-11 10:37:42,638 INFO train_multi TF=ALL epoch 29/100 train=2.2117 val=2.2186 r_mae=0.914 pos_r_acc=0.604 side_acc=0.581 r_n=161888
2026-05-11 10:37:42,648 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:37:42,648 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:37:42,648 INFO train_multi TF=ALL: new best val=2.2186 r_mae=0.9141 — saved
2026-05-11 10:37:42,653 INFO train_multi TF=ALL: new best r_mae=0.9141 — saved rmae checkpoint
2026-05-11 10:37:58,720 INFO train_multi TF=ALL epoch 30/100 train=2.2017 val=2.2097 r_mae=0.908 pos_r_acc=0.609 side_acc=0.585 r_n=161888
2026-05-11 10:37:58,725 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:37:58,725 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:37:58,725 INFO train_multi TF=ALL: new best val=2.2097 r_mae=0.9084 — saved
2026-05-11 10:37:58,729 INFO train_multi TF=ALL: new best r_mae=0.9084 — saved rmae checkpoint
2026-05-11 10:38:14,739 INFO train_multi TF=ALL epoch 31/100 train=2.1910 val=2.2012 r_mae=0.906 pos_r_acc=0.613 side_acc=0.594 r_n=161888
2026-05-11 10:38:14,744 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:38:14,744 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:38:14,744 INFO train_multi TF=ALL: new best val=2.2012 r_mae=0.9059 — saved
2026-05-11 10:38:14,748 INFO train_multi TF=ALL: new best r_mae=0.9059 — saved rmae checkpoint
2026-05-11 10:38:30,877 INFO train_multi TF=ALL epoch 32/100 train=2.1809 val=2.1843 r_mae=0.905 pos_r_acc=0.615 side_acc=0.605 r_n=161888
2026-05-11 10:38:30,883 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:38:30,883 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:38:30,883 INFO train_multi TF=ALL: new best val=2.1843 r_mae=0.9048 — saved
2026-05-11 10:38:30,887 INFO train_multi TF=ALL: new best r_mae=0.9048 — saved rmae checkpoint
2026-05-11 10:38:47,712 INFO train_multi TF=ALL epoch 33/100 train=2.1646 val=2.1679 r_mae=0.899 pos_r_acc=0.623 side_acc=0.614 r_n=161888
2026-05-11 10:38:47,718 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:38:47,718 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:38:47,718 INFO train_multi TF=ALL: new best val=2.1679 r_mae=0.8986 — saved
2026-05-11 10:38:47,723 INFO train_multi TF=ALL: new best r_mae=0.8986 — saved rmae checkpoint
2026-05-11 10:39:04,564 INFO train_multi TF=ALL epoch 34/100 train=2.1463 val=2.1488 r_mae=0.888 pos_r_acc=0.629 side_acc=0.621 r_n=161888
2026-05-11 10:39:04,569 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:39:04,570 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:39:04,570 INFO train_multi TF=ALL: new best val=2.1488 r_mae=0.8881 — saved
2026-05-11 10:39:04,574 INFO train_multi TF=ALL: new best r_mae=0.8881 — saved rmae checkpoint
2026-05-11 10:39:21,272 INFO train_multi TF=ALL epoch 35/100 train=2.1222 val=2.1248 r_mae=0.881 pos_r_acc=0.636 side_acc=0.631 r_n=161888
2026-05-11 10:39:21,278 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:39:21,279 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:39:21,279 INFO train_multi TF=ALL: new best val=2.1248 r_mae=0.8805 — saved
2026-05-11 10:39:21,285 INFO train_multi TF=ALL: new best r_mae=0.8805 — saved rmae checkpoint
2026-05-11 10:39:37,896 INFO train_multi TF=ALL epoch 36/100 train=2.1041 val=2.0993 r_mae=0.867 pos_r_acc=0.647 side_acc=0.636 r_n=161888
2026-05-11 10:39:37,901 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:39:37,901 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:39:37,901 INFO train_multi TF=ALL: new best val=2.0993 r_mae=0.8667 — saved
2026-05-11 10:39:37,906 INFO train_multi TF=ALL: new best r_mae=0.8667 — saved rmae checkpoint
2026-05-11 10:39:53,994 INFO train_multi TF=ALL epoch 37/100 train=2.0779 val=2.0846 r_mae=0.852 pos_r_acc=0.651 side_acc=0.642 r_n=161888
2026-05-11 10:39:54,000 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:39:54,000 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:39:54,000 INFO train_multi TF=ALL: new best val=2.0846 r_mae=0.8521 — saved
2026-05-11 10:39:54,004 INFO train_multi TF=ALL: new best r_mae=0.8521 — saved rmae checkpoint
2026-05-11 10:40:10,003 INFO train_multi TF=ALL epoch 38/100 train=2.0562 val=2.0696 r_mae=0.850 pos_r_acc=0.656 side_acc=0.644 r_n=161888
2026-05-11 10:40:10,008 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:40:10,008 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:40:10,008 INFO train_multi TF=ALL: new best val=2.0696 r_mae=0.8500 — saved
2026-05-11 10:40:10,013 INFO train_multi TF=ALL: new best r_mae=0.8500 — saved rmae checkpoint
2026-05-11 10:40:25,993 INFO train_multi TF=ALL epoch 39/100 train=2.0455 val=2.0614 r_mae=0.839 pos_r_acc=0.663 side_acc=0.644 r_n=161888
2026-05-11 10:40:26,003 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:40:26,003 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:40:26,003 INFO train_multi TF=ALL: new best val=2.0614 r_mae=0.8389 — saved
2026-05-11 10:40:26,008 INFO train_multi TF=ALL: new best r_mae=0.8389 — saved rmae checkpoint
2026-05-11 10:40:42,098 INFO train_multi TF=ALL epoch 40/100 train=2.0327 val=2.0536 r_mae=0.835 pos_r_acc=0.663 side_acc=0.647 r_n=161888
2026-05-11 10:40:42,108 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:40:42,109 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:40:42,109 INFO train_multi TF=ALL: new best val=2.0536 r_mae=0.8348 — saved
2026-05-11 10:40:42,113 INFO train_multi TF=ALL: new best r_mae=0.8348 — saved rmae checkpoint
2026-05-11 10:40:58,076 INFO train_multi TF=ALL epoch 41/100 train=2.0183 val=2.0461 r_mae=0.830 pos_r_acc=0.665 side_acc=0.650 r_n=161888
2026-05-11 10:40:58,086 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:40:58,087 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:40:58,087 INFO train_multi TF=ALL: new best val=2.0461 r_mae=0.8305 — saved
2026-05-11 10:40:58,091 INFO train_multi TF=ALL: new best r_mae=0.8305 — saved rmae checkpoint
2026-05-11 10:41:14,057 INFO train_multi TF=ALL epoch 42/100 train=2.0120 val=2.0421 r_mae=0.827 pos_r_acc=0.667 side_acc=0.650 r_n=161888
2026-05-11 10:41:14,062 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:41:14,063 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:41:14,063 INFO train_multi TF=ALL: new best val=2.0421 r_mae=0.8273 — saved
2026-05-11 10:41:14,067 INFO train_multi TF=ALL: new best r_mae=0.8273 — saved rmae checkpoint
2026-05-11 10:41:30,123 INFO train_multi TF=ALL epoch 43/100 train=2.0028 val=2.0356 r_mae=0.825 pos_r_acc=0.665 side_acc=0.655 r_n=161888
2026-05-11 10:41:30,129 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:41:30,129 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:41:30,129 INFO train_multi TF=ALL: new best val=2.0356 r_mae=0.8249 — saved
2026-05-11 10:41:30,133 INFO train_multi TF=ALL: new best r_mae=0.8249 — saved rmae checkpoint
2026-05-11 10:41:46,197 INFO train_multi TF=ALL epoch 44/100 train=1.9890 val=2.0370 r_mae=0.815 pos_r_acc=0.670 side_acc=0.653 r_n=161888
2026-05-11 10:41:46,202 INFO train_multi TF=ALL: new best r_mae=0.8151 — saved rmae checkpoint
2026-05-11 10:42:02,473 INFO train_multi TF=ALL epoch 45/100 train=1.9809 val=2.0223 r_mae=0.809 pos_r_acc=0.670 side_acc=0.659 r_n=161888
2026-05-11 10:42:02,478 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:42:02,478 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:42:02,478 INFO train_multi TF=ALL: new best val=2.0223 r_mae=0.8090 — saved
2026-05-11 10:42:02,483 INFO train_multi TF=ALL: new best r_mae=0.8090 — saved rmae checkpoint
2026-05-11 10:42:18,593 INFO train_multi TF=ALL epoch 46/100 train=1.9730 val=2.0126 r_mae=0.812 pos_r_acc=0.673 side_acc=0.662 r_n=161888
2026-05-11 10:42:18,599 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:42:18,599 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:42:18,599 INFO train_multi TF=ALL: new best val=2.0126 r_mae=0.8118 — saved
2026-05-11 10:42:34,619 INFO train_multi TF=ALL epoch 47/100 train=1.9631 val=2.0049 r_mae=0.809 pos_r_acc=0.674 side_acc=0.665 r_n=161888
2026-05-11 10:42:34,625 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:42:34,625 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:42:34,625 INFO train_multi TF=ALL: new best val=2.0049 r_mae=0.8086 — saved
2026-05-11 10:42:34,630 INFO train_multi TF=ALL: new best r_mae=0.8086 — saved rmae checkpoint
2026-05-11 10:42:50,611 INFO train_multi TF=ALL epoch 48/100 train=1.9529 val=2.0074 r_mae=0.800 pos_r_acc=0.677 side_acc=0.662 r_n=161888
2026-05-11 10:42:50,615 INFO train_multi TF=ALL: new best r_mae=0.8004 — saved rmae checkpoint
2026-05-11 10:43:07,204 INFO train_multi TF=ALL epoch 49/100 train=1.9480 val=2.0004 r_mae=0.806 pos_r_acc=0.675 side_acc=0.665 r_n=161888
2026-05-11 10:43:07,215 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:43:07,215 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:43:07,215 INFO train_multi TF=ALL: new best val=2.0004 r_mae=0.8064 — saved
2026-05-11 10:43:23,955 INFO train_multi TF=ALL epoch 50/100 train=1.9386 val=1.9989 r_mae=0.801 pos_r_acc=0.675 side_acc=0.666 r_n=161888
2026-05-11 10:43:23,966 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:43:23,966 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:43:23,966 INFO train_multi TF=ALL: new best val=1.9989 r_mae=0.8013 — saved
2026-05-11 10:43:40,629 INFO train_multi TF=ALL epoch 51/100 train=1.9329 val=1.9904 r_mae=0.792 pos_r_acc=0.678 side_acc=0.671 r_n=161888
2026-05-11 10:43:40,635 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:43:40,635 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:43:40,635 INFO train_multi TF=ALL: new best val=1.9904 r_mae=0.7922 — saved
2026-05-11 10:43:40,640 INFO train_multi TF=ALL: new best r_mae=0.7922 — saved rmae checkpoint
2026-05-11 10:43:57,384 INFO train_multi TF=ALL epoch 52/100 train=1.9248 val=1.9948 r_mae=0.799 pos_r_acc=0.672 side_acc=0.672 r_n=161888
2026-05-11 10:44:14,044 INFO train_multi TF=ALL epoch 53/100 train=1.9157 val=1.9780 r_mae=0.792 pos_r_acc=0.678 side_acc=0.677 r_n=161888
2026-05-11 10:44:14,050 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:44:14,050 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:44:14,050 INFO train_multi TF=ALL: new best val=1.9780 r_mae=0.7923 — saved
2026-05-11 10:44:30,749 INFO train_multi TF=ALL epoch 54/100 train=1.9085 val=1.9679 r_mae=0.796 pos_r_acc=0.677 side_acc=0.680 r_n=161888
2026-05-11 10:44:30,755 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:44:30,755 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:44:30,755 INFO train_multi TF=ALL: new best val=1.9679 r_mae=0.7959 — saved
2026-05-11 10:44:47,396 INFO train_multi TF=ALL epoch 55/100 train=1.9031 val=1.9716 r_mae=0.792 pos_r_acc=0.680 side_acc=0.677 r_n=161888
2026-05-11 10:44:47,401 INFO train_multi TF=ALL: new best r_mae=0.7917 — saved rmae checkpoint
2026-05-11 10:45:04,096 INFO train_multi TF=ALL epoch 56/100 train=1.8943 val=1.9675 r_mae=0.791 pos_r_acc=0.677 side_acc=0.681 r_n=161888
2026-05-11 10:45:04,103 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:45:04,103 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:45:04,103 INFO train_multi TF=ALL: new best val=1.9675 r_mae=0.7906 — saved
2026-05-11 10:45:04,107 INFO train_multi TF=ALL: new best r_mae=0.7906 — saved rmae checkpoint
2026-05-11 10:45:20,157 INFO train_multi TF=ALL epoch 57/100 train=1.8861 val=1.9566 r_mae=0.787 pos_r_acc=0.679 side_acc=0.682 r_n=161888
2026-05-11 10:45:20,162 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:45:20,162 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:45:20,162 INFO train_multi TF=ALL: new best val=1.9566 r_mae=0.7870 — saved
2026-05-11 10:45:20,167 INFO train_multi TF=ALL: new best r_mae=0.7870 — saved rmae checkpoint
2026-05-11 10:45:36,111 INFO train_multi TF=ALL epoch 58/100 train=1.8794 val=1.9641 r_mae=0.794 pos_r_acc=0.677 side_acc=0.682 r_n=161888
2026-05-11 10:45:51,967 INFO train_multi TF=ALL epoch 59/100 train=1.8710 val=1.9484 r_mae=0.789 pos_r_acc=0.678 side_acc=0.688 r_n=161888
2026-05-11 10:45:51,973 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:45:51,973 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:45:51,973 INFO train_multi TF=ALL: new best val=1.9484 r_mae=0.7887 — saved
2026-05-11 10:46:07,831 INFO train_multi TF=ALL epoch 60/100 train=1.8659 val=1.9458 r_mae=0.785 pos_r_acc=0.682 side_acc=0.688 r_n=161888
2026-05-11 10:46:07,837 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:46:07,837 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:46:07,837 INFO train_multi TF=ALL: new best val=1.9458 r_mae=0.7852 — saved
2026-05-11 10:46:07,842 INFO train_multi TF=ALL: new best r_mae=0.7852 — saved rmae checkpoint
2026-05-11 10:46:23,820 INFO train_multi TF=ALL epoch 61/100 train=1.8574 val=1.9442 r_mae=0.788 pos_r_acc=0.680 side_acc=0.689 r_n=161888
2026-05-11 10:46:23,831 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:46:23,831 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:46:23,831 INFO train_multi TF=ALL: new best val=1.9442 r_mae=0.7877 — saved
2026-05-11 10:46:39,678 INFO train_multi TF=ALL epoch 62/100 train=1.8479 val=1.9379 r_mae=0.784 pos_r_acc=0.681 side_acc=0.692 r_n=161888
2026-05-11 10:46:39,684 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:46:39,684 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:46:39,684 INFO train_multi TF=ALL: new best val=1.9379 r_mae=0.7842 — saved
2026-05-11 10:46:39,688 INFO train_multi TF=ALL: new best r_mae=0.7842 — saved rmae checkpoint
2026-05-11 10:46:55,694 INFO train_multi TF=ALL epoch 63/100 train=1.8409 val=1.9287 r_mae=0.778 pos_r_acc=0.682 side_acc=0.698 r_n=161888
2026-05-11 10:46:55,699 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:46:55,699 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:46:55,699 INFO train_multi TF=ALL: new best val=1.9287 r_mae=0.7780 — saved
2026-05-11 10:46:55,703 INFO train_multi TF=ALL: new best r_mae=0.7780 — saved rmae checkpoint
2026-05-11 10:47:11,734 INFO train_multi TF=ALL epoch 64/100 train=1.8333 val=1.9314 r_mae=0.781 pos_r_acc=0.679 side_acc=0.699 r_n=161888
2026-05-11 10:47:27,653 INFO train_multi TF=ALL epoch 65/100 train=1.8247 val=1.9210 r_mae=0.780 pos_r_acc=0.683 side_acc=0.702 r_n=161888
2026-05-11 10:47:27,659 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:47:27,659 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:47:27,659 INFO train_multi TF=ALL: new best val=1.9210 r_mae=0.7803 — saved
2026-05-11 10:47:43,468 INFO train_multi TF=ALL epoch 66/100 train=1.8240 val=1.9148 r_mae=0.784 pos_r_acc=0.681 side_acc=0.706 r_n=161888
2026-05-11 10:47:43,473 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:47:43,473 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:47:43,473 INFO train_multi TF=ALL: new best val=1.9148 r_mae=0.7843 — saved
2026-05-11 10:47:59,424 INFO train_multi TF=ALL epoch 67/100 train=1.8094 val=1.9109 r_mae=0.781 pos_r_acc=0.682 side_acc=0.706 r_n=161888
2026-05-11 10:47:59,435 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:47:59,435 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:47:59,435 INFO train_multi TF=ALL: new best val=1.9109 r_mae=0.7812 — saved
2026-05-11 10:48:15,648 INFO train_multi TF=ALL epoch 68/100 train=1.8043 val=1.9083 r_mae=0.780 pos_r_acc=0.683 side_acc=0.708 r_n=161888
2026-05-11 10:48:15,660 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:48:15,660 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:48:15,660 INFO train_multi TF=ALL: new best val=1.9083 r_mae=0.7800 — saved
2026-05-11 10:48:31,769 INFO train_multi TF=ALL epoch 69/100 train=1.7945 val=1.9115 r_mae=0.782 pos_r_acc=0.680 side_acc=0.709 r_n=161888
2026-05-11 10:48:47,669 INFO train_multi TF=ALL epoch 70/100 train=1.7891 val=1.8958 r_mae=0.782 pos_r_acc=0.684 side_acc=0.714 r_n=161888
2026-05-11 10:48:47,680 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:48:47,680 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:48:47,680 INFO train_multi TF=ALL: new best val=1.8958 r_mae=0.7821 — saved
2026-05-11 10:49:04,123 INFO train_multi TF=ALL epoch 71/100 train=1.7795 val=1.8956 r_mae=0.776 pos_r_acc=0.685 side_acc=0.716 r_n=161888
2026-05-11 10:49:04,127 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:49:04,128 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:49:04,128 INFO train_multi TF=ALL: new best val=1.8956 r_mae=0.7756 — saved
2026-05-11 10:49:04,132 INFO train_multi TF=ALL: new best r_mae=0.7756 — saved rmae checkpoint
2026-05-11 10:49:19,945 INFO train_multi TF=ALL epoch 72/100 train=1.7774 val=1.8975 r_mae=0.772 pos_r_acc=0.685 side_acc=0.719 r_n=161888
2026-05-11 10:49:19,950 INFO train_multi TF=ALL: new best r_mae=0.7721 — saved rmae checkpoint
2026-05-11 10:49:35,762 INFO train_multi TF=ALL epoch 73/100 train=1.7707 val=1.8900 r_mae=0.772 pos_r_acc=0.686 side_acc=0.722 r_n=161888
2026-05-11 10:49:35,767 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:49:35,768 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:49:35,768 INFO train_multi TF=ALL: new best val=1.8900 r_mae=0.7724 — saved
2026-05-11 10:49:51,562 INFO train_multi TF=ALL epoch 74/100 train=1.7636 val=1.8835 r_mae=0.776 pos_r_acc=0.684 side_acc=0.725 r_n=161888
2026-05-11 10:49:51,567 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:49:51,568 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:49:51,568 INFO train_multi TF=ALL: new best val=1.8835 r_mae=0.7760 — saved
2026-05-11 10:50:07,296 INFO train_multi TF=ALL epoch 75/100 train=1.7592 val=1.8934 r_mae=0.773 pos_r_acc=0.686 side_acc=0.723 r_n=161888
2026-05-11 10:50:23,141 INFO train_multi TF=ALL epoch 76/100 train=1.7512 val=1.8842 r_mae=0.772 pos_r_acc=0.685 side_acc=0.728 r_n=161888
2026-05-11 10:50:38,910 INFO train_multi TF=ALL epoch 77/100 train=1.7498 val=1.8848 r_mae=0.772 pos_r_acc=0.684 side_acc=0.729 r_n=161888
2026-05-11 10:50:38,915 INFO train_multi TF=ALL: new best r_mae=0.7721 — saved rmae checkpoint
2026-05-11 10:50:54,679 INFO train_multi TF=ALL epoch 78/100 train=1.7341 val=1.8941 r_mae=0.780 pos_r_acc=0.678 side_acc=0.730 r_n=161888
2026-05-11 10:51:10,304 INFO train_multi TF=ALL epoch 79/100 train=1.7319 val=1.8864 r_mae=0.777 pos_r_acc=0.683 side_acc=0.730 r_n=161888
2026-05-11 10:51:26,122 INFO train_multi TF=ALL epoch 80/100 train=1.7270 val=1.8826 r_mae=0.774 pos_r_acc=0.683 side_acc=0.733 r_n=161888
2026-05-11 10:51:26,133 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:51:26,133 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:51:26,134 INFO train_multi TF=ALL: new best val=1.8826 r_mae=0.7740 — saved
2026-05-11 10:51:41,853 INFO train_multi TF=ALL epoch 81/100 train=1.7198 val=1.9026 r_mae=0.780 pos_r_acc=0.680 side_acc=0.725 r_n=161888
2026-05-11 10:51:57,633 INFO train_multi TF=ALL epoch 82/100 train=1.7159 val=1.8723 r_mae=0.778 pos_r_acc=0.683 side_acc=0.735 r_n=161888
2026-05-11 10:51:57,638 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:51:57,638 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:51:57,638 INFO train_multi TF=ALL: new best val=1.8723 r_mae=0.7784 — saved
2026-05-11 10:52:13,516 INFO train_multi TF=ALL epoch 83/100 train=1.7068 val=1.8887 r_mae=0.776 pos_r_acc=0.682 side_acc=0.733 r_n=161888
2026-05-11 10:52:29,555 INFO train_multi TF=ALL epoch 84/100 train=1.7000 val=1.8994 r_mae=0.777 pos_r_acc=0.681 side_acc=0.729 r_n=161888
2026-05-11 10:52:45,342 INFO train_multi TF=ALL epoch 85/100 train=1.6975 val=1.8902 r_mae=0.773 pos_r_acc=0.684 side_acc=0.734 r_n=161888
2026-05-11 10:53:01,001 INFO train_multi TF=ALL epoch 86/100 train=1.6931 val=1.8964 r_mae=0.775 pos_r_acc=0.681 side_acc=0.735 r_n=161888
2026-05-11 10:53:16,724 INFO train_multi TF=ALL epoch 87/100 train=1.6855 val=1.8936 r_mae=0.780 pos_r_acc=0.680 side_acc=0.734 r_n=161888
2026-05-11 10:53:32,497 INFO train_multi TF=ALL epoch 88/100 train=1.6762 val=1.8935 r_mae=0.776 pos_r_acc=0.681 side_acc=0.735 r_n=161888
2026-05-11 10:53:48,339 INFO train_multi TF=ALL epoch 89/100 train=1.6694 val=1.8917 r_mae=0.781 pos_r_acc=0.681 side_acc=0.735 r_n=161888
2026-05-11 10:54:04,610 INFO train_multi TF=ALL epoch 90/100 train=1.6701 val=1.9035 r_mae=0.779 pos_r_acc=0.679 side_acc=0.737 r_n=161888
2026-05-11 10:54:20,905 INFO train_multi TF=ALL epoch 91/100 train=1.6623 val=1.8991 r_mae=0.781 pos_r_acc=0.680 side_acc=0.735 r_n=161888
2026-05-11 10:54:36,958 INFO train_multi TF=ALL epoch 92/100 train=1.6575 val=1.9072 r_mae=0.776 pos_r_acc=0.680 side_acc=0.735 r_n=161888
2026-05-11 10:54:53,269 INFO train_multi TF=ALL epoch 93/100 train=1.6530 val=1.9090 r_mae=0.781 pos_r_acc=0.678 side_acc=0.735 r_n=161888
2026-05-11 10:55:09,322 INFO train_multi TF=ALL epoch 94/100 train=1.6444 val=1.9073 r_mae=0.783 pos_r_acc=0.678 side_acc=0.735 r_n=161888
2026-05-11 10:55:25,741 INFO train_multi TF=ALL epoch 95/100 train=1.6405 val=1.9066 r_mae=0.787 pos_r_acc=0.675 side_acc=0.736 r_n=161888
2026-05-11 10:55:42,422 INFO train_multi TF=ALL epoch 96/100 train=1.6343 val=1.9122 r_mae=0.783 pos_r_acc=0.676 side_acc=0.735 r_n=161888
2026-05-11 10:55:58,844 INFO train_multi TF=ALL epoch 97/100 train=1.6282 val=1.9297 r_mae=0.780 pos_r_acc=0.677 side_acc=0.734 r_n=161888
2026-05-11 10:56:15,299 INFO train_multi TF=ALL epoch 98/100 train=1.6218 val=1.9332 r_mae=0.783 pos_r_acc=0.675 side_acc=0.735 r_n=161888
2026-05-11 10:56:31,532 INFO train_multi TF=ALL epoch 99/100 train=1.6174 val=1.9466 r_mae=0.784 pos_r_acc=0.674 side_acc=0.733 r_n=161888
2026-05-11 10:56:47,808 INFO train_multi TF=ALL epoch 100/100 train=1.6126 val=1.9333 r_mae=0.782 pos_r_acc=0.674 side_acc=0.734 r_n=161888
2026-05-11 10:56:47,826 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 10:56:47,826 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 10:56:47,826 INFO train_multi TF=ALL: promoted r_mae checkpoint (0.7721 < primary 0.7784) — overwriting model.pt
2026-05-11 10:56:48,232 WARNING fit_r_isotonic: long calibrator increases MAE (cal=0.7889 >= raw=0.7806) — skipping
2026-05-11 10:56:48,246 WARNING fit_r_isotonic: short calibrator increases MAE (cal=0.7993 >= raw=0.7985) — skipping
2026-05-11 10:56:48,246 INFO fit_r_isotonic: saved side-R calibrators stats={'long': {'n': 40072, 'raw_mae': 0.7806036282433984, 'calibrated_mae': 0.788896683184944, 'skipped': 'calibrator_hurts'}, 'short': {'n': 41197, 'raw_mae': 0.7985118625309208, 'calibrated_mae': 0.7993050308993832, 'skipped': 'calibrator_hurts'}}
2026-05-11 10:56:48,390 WARNING GRU validation R-MAE above floor for TF=ALL: best_val_r_mae=0.772 max=0.750. Keeping saved best weights so the pipeline can progress.
2026-05-11 10:56:48,404 INFO GRU R threshold XAUUSD/buy: q25_pos=0.535 q50_pos=1.667 pos_rate=46.0% (n=119259 n_pos=54899)
2026-05-11 10:56:48,411 INFO GRU R threshold XAUUSD/sell: q25_pos=0.528 q50_pos=1.667 pos_rate=45.5% (n=118087 n_pos=53734)
2026-05-11 10:56:48,418 INFO GRU R threshold EURUSD/buy: q25_pos=0.541 q50_pos=1.667 pos_rate=45.6% (n=118751 n_pos=54196)
2026-05-11 10:56:48,424 INFO GRU R threshold EURUSD/sell: q25_pos=0.545 q50_pos=1.667 pos_rate=45.6% (n=117679 n_pos=53721)
2026-05-11 10:56:48,431 INFO GRU R threshold USDJPY/buy: q25_pos=0.551 q50_pos=1.667 pos_rate=46.2% (n=118691 n_pos=54870)
2026-05-11 10:56:48,437 INFO GRU R threshold USDJPY/sell: q25_pos=0.542 q50_pos=1.667 pos_rate=44.6% (n=116030 n_pos=51793)
2026-05-11 10:56:48,444 INFO GRU R threshold EURJPY/buy: q25_pos=0.530 q50_pos=1.416 pos_rate=46.1% (n=118651 n_pos=54654)
2026-05-11 10:56:48,450 INFO GRU R threshold EURJPY/sell: q25_pos=0.525 q50_pos=1.667 pos_rate=44.7% (n=117320 n_pos=52399)
2026-05-11 10:56:48,456 INFO GRU R threshold GBPJPY/buy: q25_pos=0.517 q50_pos=1.443 pos_rate=45.9% (n=118277 n_pos=54278)
2026-05-11 10:56:48,462 INFO GRU R threshold GBPJPY/sell: q25_pos=0.521 q50_pos=1.667 pos_rate=45.3% (n=116299 n_pos=52702)
2026-05-11 10:56:48,469 INFO GRU R threshold GBPUSD/buy: q25_pos=0.524 q50_pos=1.667 pos_rate=45.8% (n=117764 n_pos=53894)
2026-05-11 10:56:48,475 INFO GRU R threshold GBPUSD/sell: q25_pos=0.533 q50_pos=1.667 pos_rate=45.8% (n=117404 n_pos=53785)
2026-05-11 10:56:48,476 INFO GRU per-symbol R thresholds saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/symbol_r_thresholds.json (6 symbols)
2026-05-11 10:56:48,492 INFO Retrain complete. Total wall-clock: 1640.1s
  DONE  Retrain gru [train-split retrain]
  START Retrain regime [train-split retrain]
2026-05-11 10:56:52,203 INFO retrain environment: KAGGLE
2026-05-11 10:56:53,814 INFO Device: CUDA (2 GPU(s))
2026-05-11 10:56:53,822 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 10:56:53,823 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 10:56:53,823 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 10:56:53,823 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 10:56:53,823 INFO Retrain data split: train
2026-05-11 10:56:53,823 INFO Retrain rolling fold selector: latest
2026-05-11 10:56:53,824 INFO === RegimeClassifier retrain (hierarchical: HTF 2-score bias + LTF 5-score behaviour) ===
2026-05-11 10:56:53,975 INFO NumExpr defaulting to 4 threads.
2026-05-11 10:56:54,174 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-11 10:56:54,175 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 10:56:54,175 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 10:56:54,175 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-11 10:56:54,233 INFO Regime rolling folds selected: [None]
2026-05-11 10:56:54,234 INFO === Regime rolling fold 1/1: train_all ===
2026-05-11 10:56:54,234 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-11 10:56:54,277 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 10:56:54,278 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:56:54,296 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:56:54,313 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:56:54,330 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:56:54,351 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:56:54,368 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:56:54,614 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:56:54,687 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:56:54,712 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:56:54,713 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:56:54,725 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:56:54,726 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:56:55,160 INFO macro_cache: using training data dir /kaggle/working/Multi-Bot/trading-system/training_data
2026-05-11 10:56:55,274 INFO macro_cache loaded 19 series: {'asx200': '2842 bars 2015-01-02→2026-03-27', 'cac40': '2876 bars 2015-01-02→2026-03-27', 'dax': '2851 bars 2015-01-02→2026-03-27', 'djia': '2825 bars 2015-01-02→2026-03-27', 'dxy': '2826 bars 2015-01-02→2026-03-27', 'eurostoxx': '2823 bars 2015-01-05→2026-03-27', 'ftse': '2839 bars 2015-01-02→2026-03-27', 'gold_fut': '2824 bars 2015-01-02→2026-03-27', 'hsi': '2764 bars 2015-01-02→2026-03-27', 'nasdaq': '2825 bars 2015-01-02→2026-03-27', 'nikkei': '2744 bars 2015-01-05→2026-03-27', 'oil_fut': '2825 bars 2015-01-02→2026-03-27', 'spx': '2825 bars 2015-01-02→2026-03-27', 'us10y': '2824 bars 2015-01-02→2026-03-27', 'us30y': '2824 bars 2015-01-02→2026-03-27', 'us3m': '2824 bars 2015-01-02→2026-03-27', 'vix': '2825 bars 2015-01-02→2026-03-27', 'us10y_fred': '2607 bars 2016-03-28→2026-03-24', 'us2y_fred': '2607 bars 2016-03-28→2026-03-24'}
2026-05-11 10:56:55,531 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1209, 'BIAS_DOWN': 568, 'BIAS_NEUTRAL': 10325}  ambiguous=3935 (total=12102) horizon=84
2026-05-11 10:56:55,536 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected XAUUSD — 12052 samples (group=gold) score_means={'bias_up_score': 0.1003, 'bias_down_score': 0.0471} labels={'BIAS_UP': 1209, 'BIAS_DOWN': 568, 'BIAS_NEUTRAL': 10275} clean={'BIAS_UP': 1209, 'BIAS_DOWN': 568, 'BIAS_NEUTRAL': 6348}
2026-05-11 10:56:55,721 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:55,762 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:55,783 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:55,783 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:55,792 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:55,794 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:56:56,428 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 722, 'BIAS_DOWN': 567, 'BIAS_NEUTRAL': 10115}  ambiguous=3689 (total=11404) horizon=84
2026-05-11 10:56:56,434 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURUSD — 11354 samples (group=dollar) score_means={'bias_up_score': 0.0636, 'bias_down_score': 0.0499} labels={'BIAS_UP': 722, 'BIAS_DOWN': 567, 'BIAS_NEUTRAL': 10065} clean={'BIAS_UP': 722, 'BIAS_DOWN': 567, 'BIAS_NEUTRAL': 6394}
2026-05-11 10:56:56,600 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:56,640 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:56,660 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:56,661 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:56,670 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:56,671 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:56:57,292 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 872, 'BIAS_DOWN': 463, 'BIAS_NEUTRAL': 10068}  ambiguous=3827 (total=11403) horizon=84
2026-05-11 10:56:57,297 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected USDJPY — 11353 samples (group=dollar) score_means={'bias_up_score': 0.0768, 'bias_down_score': 0.0408} labels={'BIAS_UP': 872, 'BIAS_DOWN': 463, 'BIAS_NEUTRAL': 10018} clean={'BIAS_UP': 872, 'BIAS_DOWN': 463, 'BIAS_NEUTRAL': 6201}
2026-05-11 10:56:57,454 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:57,491 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:57,515 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:57,516 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:57,525 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:57,526 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:56:58,132 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 712, 'BIAS_DOWN': 556, 'BIAS_NEUTRAL': 10139}  ambiguous=3816 (total=11407) horizon=84
2026-05-11 10:56:58,137 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURJPY — 11357 samples (group=cross) score_means={'bias_up_score': 0.0627, 'bias_down_score': 0.049} labels={'BIAS_UP': 712, 'BIAS_DOWN': 556, 'BIAS_NEUTRAL': 10089} clean={'BIAS_UP': 712, 'BIAS_DOWN': 556, 'BIAS_NEUTRAL': 6279}
2026-05-11 10:56:58,294 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:58,332 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:58,354 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:58,355 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:58,367 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:58,369 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:56:58,976 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 873, 'BIAS_DOWN': 633, 'BIAS_NEUTRAL': 9902}  ambiguous=4022 (total=11408) horizon=84
2026-05-11 10:56:58,982 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPJPY — 11358 samples (group=cross) score_means={'bias_up_score': 0.0769, 'bias_down_score': 0.0557} labels={'BIAS_UP': 873, 'BIAS_DOWN': 633, 'BIAS_NEUTRAL': 9852} clean={'BIAS_UP': 873, 'BIAS_DOWN': 633, 'BIAS_NEUTRAL': 5852}
2026-05-11 10:56:59,141 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:59,178 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:59,197 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:59,198 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:59,207 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:56:59,208 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:56:59,843 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 546, 'BIAS_DOWN': 754, 'BIAS_NEUTRAL': 10102}  ambiguous=3944 (total=11402) horizon=84
2026-05-11 10:56:59,848 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPUSD — 11352 samples (group=dollar) score_means={'bias_up_score': 0.0481, 'bias_down_score': 0.0651} labels={'BIAS_UP': 546, 'BIAS_DOWN': 739, 'BIAS_NEUTRAL': 10067} clean={'BIAS_UP': 546, 'BIAS_DOWN': 739, 'BIAS_NEUTRAL': 6149}
2026-05-11 10:56:59,913 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 1585, 'BIAS_DOWN': 1189, 'BIAS_NEUTRAL': 19941}, 'dollar': {'BIAS_UP': 2140, 'BIAS_DOWN': 1769, 'BIAS_NEUTRAL': 30150}, 'gold': {'BIAS_UP': 1209, 'BIAS_DOWN': 568, 'BIAS_NEUTRAL': 10275}}
2026-05-11 10:56:59,914 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0698, 'bias_down_score': 0.0523}, 'dollar': {'bias_up_score': 0.0628, 'bias_down_score': 0.0519}, 'gold': {'bias_up_score': 0.1003, 'bias_down_score': 0.0471}}
2026-05-11 10:56:59,914 INFO Regime[4H mode=htf_bias] label distribution by year: {2016: {'BIAS_UP': 525, 'BIAS_DOWN': 617, 'BIAS_NEUTRAL': 7680}, 2017: {'BIAS_UP': 776, 'BIAS_DOWN': 315, 'BIAS_NEUTRAL': 8022}, 2018: {'BIAS_UP': 453, 'BIAS_DOWN': 753, 'BIAS_NEUTRAL': 7924}, 2019: {'BIAS_UP': 427, 'BIAS_DOWN': 481, 'BIAS_NEUTRAL': 8194}, 2020: {'BIAS_UP': 721, 'BIAS_DOWN': 181, 'BIAS_NEUTRAL': 8209}, 2021: {'BIAS_UP': 768, 'BIAS_DOWN': 506, 'BIAS_NEUTRAL': 7817}, 2022: {'BIAS_UP': 703, 'BIAS_DOWN': 561, 'BIAS_NEUTRAL': 7857}, 2023: {'BIAS_UP': 561, 'BIAS_DOWN': 112, 'BIAS_NEUTRAL': 4663}}
2026-05-11 10:56:59,914 INFO Regime[4H mode=htf_bias] score means by year: {2016: {'bias_up_score': 0.0595, 'bias_down_score': 0.0699}, 2017: {'bias_up_score': 0.0852, 'bias_down_score': 0.0346}, 2018: {'bias_up_score': 0.0496, 'bias_down_score': 0.0825}, 2019: {'bias_up_score': 0.0469, 'bias_down_score': 0.0528}, 2020: {'bias_up_score': 0.0791, 'bias_down_score': 0.0199}, 2021: {'bias_up_score': 0.0845, 'bias_down_score': 0.0557}, 2022: {'bias_up_score': 0.0771, 'bias_down_score': 0.0615}, 2023: {'bias_up_score': 0.1051, 'bias_down_score': 0.021}}
2026-05-11 10:56:59,960 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
2026-05-11 10:56:59,961 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
2026-05-11 10:56:59,962 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
2026-05-11 10:56:59,963 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 10:56:59,964 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 10:56:59,964 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
2026-05-11 10:56:59,982 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:56:59,985 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:56:59,987 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:56:59,987 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:56:59,987 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:56:59,988 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:57:00,372 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 166, 'BIAS_DOWN': 96, 'BIAS_NEUTRAL': 1319}  ambiguous=536 (total=1581) horizon=84
2026-05-11 10:57:00,375 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected XAUUSD — 1531 samples (group=gold) score_means={'bias_up_score': 0.1084, 'bias_down_score': 0.0627} labels={'BIAS_UP': 166, 'BIAS_DOWN': 96, 'BIAS_NEUTRAL': 1269} clean={'BIAS_UP': 166, 'BIAS_DOWN': 96, 'BIAS_NEUTRAL': 754}
2026-05-11 10:57:00,448 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:00,451 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:00,451 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:00,452 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:00,452 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:00,453 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:57:00,816 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 141, 'BIAS_DOWN': 68, 'BIAS_NEUTRAL': 1282}  ambiguous=504 (total=1491) horizon=84
2026-05-11 10:57:00,819 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURUSD — 1441 samples (group=dollar) score_means={'bias_up_score': 0.0978, 'bias_down_score': 0.0472} labels={'BIAS_UP': 141, 'BIAS_DOWN': 68, 'BIAS_NEUTRAL': 1232} clean={'BIAS_UP': 141, 'BIAS_DOWN': 68, 'BIAS_NEUTRAL': 757}
2026-05-11 10:57:00,885 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:00,888 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:00,888 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:00,889 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:00,889 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:00,891 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:57:01,259 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 173, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1231}  ambiguous=584 (total=1489) horizon=84
2026-05-11 10:57:01,261 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected USDJPY — 1439 samples (group=dollar) score_means={'bias_up_score': 0.1202, 'bias_down_score': 0.0591} labels={'BIAS_UP': 173, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1181} clean={'BIAS_UP': 173, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 621}
2026-05-11 10:57:01,354 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:01,357 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:01,357 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:01,358 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:01,358 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:01,360 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:57:01,718 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 125, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 1364}  ambiguous=540 (total=1494) horizon=84
2026-05-11 10:57:01,721 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0866, 'bias_down_score': 0.0035} labels={'BIAS_UP': 125, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 1314} clean={'BIAS_UP': 125, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 780}
2026-05-11 10:57:01,787 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:01,790 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:01,790 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:01,791 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:01,791 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:01,792 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:57:02,147 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 134, 'BIAS_DOWN': 11, 'BIAS_NEUTRAL': 1349}  ambiguous=512 (total=1494) horizon=84
2026-05-11 10:57:02,150 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0928, 'bias_down_score': 0.0069} labels={'BIAS_UP': 134, 'BIAS_DOWN': 10, 'BIAS_NEUTRAL': 1300} clean={'BIAS_UP': 134, 'BIAS_DOWN': 10, 'BIAS_NEUTRAL': 807}
2026-05-11 10:57:02,217 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:02,219 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:02,220 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:02,221 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:02,221 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:02,222 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:57:02,567 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 93, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 1304}  ambiguous=544 (total=1488) horizon=84
2026-05-11 10:57:02,569 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPUSD — 1438 samples (group=dollar) score_means={'bias_up_score': 0.0647, 'bias_down_score': 0.0633} labels={'BIAS_UP': 93, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 1254} clean={'BIAS_UP': 93, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 738}
2026-05-11 10:57:02,632 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 259, 'BIAS_DOWN': 15, 'BIAS_NEUTRAL': 2614}, 'dollar': {'BIAS_UP': 407, 'BIAS_DOWN': 244, 'BIAS_NEUTRAL': 3667}, 'gold': {'BIAS_UP': 166, 'BIAS_DOWN': 96, 'BIAS_NEUTRAL': 1269}}
2026-05-11 10:57:02,633 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0897, 'bias_down_score': 0.0052}, 'dollar': {'bias_up_score': 0.0943, 'bias_down_score': 0.0565}, 'gold': {'bias_up_score': 0.1084, 'bias_down_score': 0.0627}}
2026-05-11 10:57:02,633 INFO Regime[4H mode=htf_bias] label distribution by year: {2022: {'BIAS_UP': 276, 'BIAS_DOWN': 248, 'BIAS_NEUTRAL': 2877}, 2023: {'BIAS_UP': 556, 'BIAS_DOWN': 107, 'BIAS_NEUTRAL': 4673}}
2026-05-11 10:57:02,633 INFO Regime[4H mode=htf_bias] score means by year: {2022: {'bias_up_score': 0.0812, 'bias_down_score': 0.0729}, 2023: {'bias_up_score': 0.1042, 'bias_down_score': 0.0201}}
2026-05-11 10:57:02,675 INFO Regime phase HTF dataset build fold=train_all: 8.4s (train=68826 val=8737)
2026-05-11 10:57:02,676 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260511_105702
2026-05-11 10:57:02,878 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=36, n_classes=2)
2026-05-11 10:57:02,878 INFO Regime 4H/htf_bias warm start enabled from existing weights
2026-05-11 10:57:02,885 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=68826 val=8737 train_labels={'BIAS_UP': 4934, 'BIAS_DOWN': 3526, 'BIAS_NEUTRAL': 60366} val_labels={'BIAS_UP': 832, 'BIAS_DOWN': 355, 'BIAS_NEUTRAL': 7550}
2026-05-11 10:57:02,886 INFO RegimeClassifier[mode=htf_bias]: warm start HTF score head
2026-05-11 10:57:02,886 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-11 10:57:02,886 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 12.949, 'bias_down_score': 18.52}
2026-05-11 10:57:02,890 INFO RegimeClassifier[mode=htf_bias]: HTF balanced sampler — dir=8460 neutral=60366 dir_weight=5 => dir_frac_per_epoch≈41.2%
2026-05-11 10:57:06,481 INFO Regime HTF score epoch  1/50 — tr=6.1762 va=1.4881 acc=0.834 bal=0.425 threshold=0.35 margin=0.40 recall={'BIAS_UP': 0.165, 'BIAS_DOWN': 0.172, 'BIAS_NEUTRAL': 0.938} precision={'BIAS_UP': 0.302, 'BIAS_DOWN': 0.292, 'BIAS_NEUTRAL': 0.878}
2026-05-11 10:57:07,892 INFO Regime HTF score epoch  2/50 — tr=6.1269 va=1.4776 bal=0.427
2026-05-11 10:57:09,348 INFO Regime HTF score epoch  3/50 — tr=6.0747 va=1.4445 bal=0.408
2026-05-11 10:57:10,765 INFO Regime HTF score epoch  4/50 — tr=5.9322 va=1.3952 bal=0.426
2026-05-11 10:57:12,229 INFO Regime HTF score epoch  5/50 — tr=5.6604 va=1.3258 acc=0.844 bal=0.390 threshold=0.60 margin=0.15 recall={'BIAS_UP': 0.091, 'BIAS_DOWN': 0.118, 'BIAS_NEUTRAL': 0.962} precision={'BIAS_UP': 0.297, 'BIAS_DOWN': 0.276, 'BIAS_NEUTRAL': 0.872}
2026-05-11 10:57:13,641 INFO Regime HTF score epoch  6/50 — tr=5.3844 va=1.2600 bal=0.398
2026-05-11 10:57:15,082 INFO Regime HTF score epoch  7/50 — tr=5.0838 va=1.1805 bal=0.400
2026-05-11 10:57:16,483 INFO Regime HTF score epoch  8/50 — tr=4.7762 va=1.1072 bal=0.383
2026-05-11 10:57:17,886 INFO Regime HTF score epoch  9/50 — tr=4.4390 va=1.0433 bal=0.379
2026-05-11 10:57:19,324 INFO Regime HTF score epoch 10/50 — tr=4.1272 va=0.9830 acc=0.845 bal=0.392 threshold=0.80 margin=0.15 recall={'BIAS_UP': 0.069, 'BIAS_DOWN': 0.144, 'BIAS_NEUTRAL': 0.964} precision={'BIAS_UP': 0.311, 'BIAS_DOWN': 0.255, 'BIAS_NEUTRAL': 0.871}
2026-05-11 10:57:19,324 INFO Regime HTF score early stop at epoch 10
2026-05-11 10:57:20,588 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.350 margin=0.400 precision={'BIAS_UP': 0.299, 'BIAS_DOWN': 0.294, 'BIAS_NEUTRAL': 0.878} recall={'BIAS_UP': 0.167, 'BIAS_DOWN': 0.177, 'BIAS_NEUTRAL': 0.937} f1={'BIAS_UP': 0.214, 'BIAS_DOWN': 0.221, 'BIAS_NEUTRAL': 0.906} confusion=[[139, 0, 693], [0, 63, 292], [326, 151, 7073]] score_mae={'bias_up_score': 0.1703, 'bias_down_score': 0.1033} pred_share={'BIAS_UP': 0.0532, 'BIAS_DOWN': 0.0245, 'BIAS_NEUTRAL': 0.9223}
2026-05-11 10:57:20,589 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.299, 'BIAS_DOWN': 0.294, 'BIAS_NEUTRAL': 0.878} min_precision=0.500 recall={'BIAS_UP': 0.167, 'BIAS_DOWN': 0.177, 'BIAS_NEUTRAL': 0.937} min_recall=0.100 f1={'BIAS_UP': 0.214, 'BIAS_DOWN': 0.221, 'BIAS_NEUTRAL': 0.906} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_UP', 'BIAS_DOWN'] weak_recall=[] weak_f1=[] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-11 10:57:20,592 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 10:57:20,592 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 10:57:20,593 INFO Regime phase HTF train fold=train_all: 17.7s
2026-05-11 10:57:20,695 INFO Regime HTF complete fold=train_all: acc=0.833 bal=0.427 train=68826 val=8737 per_class={'BIAS_UP': 0.167, 'BIAS_DOWN': 0.177, 'BIAS_NEUTRAL': 0.937} precision={'BIAS_UP': 0.299, 'BIAS_DOWN': 0.294, 'BIAS_NEUTRAL': 0.878} threshold=0.350 margin=0.400
2026-05-11 10:57:20,696 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:57:20,897 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 546, 'BIAS_DOWN': 754, 'BIAS_NEUTRAL': 10102}  ambiguous=3944 (total=11402) horizon=84
2026-05-11 10:57:20,900 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on GBPUSD 4H:
{'BIAS_UP': 4.403225806451613, 'BIAS_DOWN': 5.755725190839694, 'BIAS_NEUTRAL': 39.4609375}
2026-05-11 10:57:20,903 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (all labels):
{'BIAS_UP': {'n': 546, 'mean': 0.00040267113448389793, 'mean_over_std': 0.18208067865763405}, 'BIAS_DOWN': {'n': 754, 'mean': -0.00047099607164125245, 'mean_over_std': -0.19477795734555267}, 'BIAS_NEUTRAL': {'n': 10101, 'mean': 2.6464098295242517e-06, 'mean_over_std': 0.0010012545608535832}}
2026-05-11 10:57:20,904 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 546, 'mean': 0.00040267113448389793, 'mean_over_std': 0.18208067865763405}, 'BIAS_DOWN': {'n': 754, 'mean': -0.00047099607164125245, 'mean_over_std': -0.19477795734555267}, 'BIAS_NEUTRAL': {'n': 6158, 'mean': 2.1496848003307296e-05, 'mean_over_std': 0.009079001472003705}}
2026-05-11 10:57:20,908 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-11 10:57:20,910 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:57:20,912 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:57:20,914 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:57:20,916 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:57:20,918 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:57:20,920 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:57:20,939 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:57:20,948 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:57:20,951 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:57:20,952 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:57:20,952 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:57:20,961 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:57:21,856 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected XAUUSD — 44227 samples (group=gold) score_means={'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}
2026-05-11 10:57:21,962 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:21,964 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:21,965 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:21,966 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:21,966 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:21,968 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:57:22,786 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURUSD — 43689 samples (group=dollar) score_means={'trend_score': 0.49, 'range_score': 0.236, 'chop_score': 0.4622, 'volatility_percentile': 0.3812, 'consolidation_score': 0.1862}
2026-05-11 10:57:22,896 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:22,899 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:22,900 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:22,900 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:22,900 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:22,903 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:57:23,745 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected USDJPY — 43686 samples (group=dollar) score_means={'trend_score': 0.4949, 'range_score': 0.2312, 'chop_score': 0.4613, 'volatility_percentile': 0.38, 'consolidation_score': 0.1931}
2026-05-11 10:57:23,850 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:23,852 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:23,853 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:23,853 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:23,854 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:23,856 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:57:24,677 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURJPY — 43688 samples (group=cross) score_means={'trend_score': 0.4888, 'range_score': 0.2345, 'chop_score': 0.4662, 'volatility_percentile': 0.3831, 'consolidation_score': 0.1882}
2026-05-11 10:57:24,785 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:24,788 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:24,788 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:24,789 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:24,789 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:24,791 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:57:25,622 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPJPY — 43678 samples (group=cross) score_means={'trend_score': 0.4878, 'range_score': 0.2348, 'chop_score': 0.4673, 'volatility_percentile': 0.3839, 'consolidation_score': 0.1904}
2026-05-11 10:57:25,730 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:25,733 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:25,733 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:25,734 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:25,734 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:25,737 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:57:26,568 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPUSD — 43676 samples (group=dollar) score_means={'trend_score': 0.4923, 'range_score': 0.2335, 'chop_score': 0.4611, 'volatility_percentile': 0.3832, 'consolidation_score': 0.1856}
2026-05-11 10:57:26,679 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2346, 'chop_score': 0.4667, 'volatility_percentile': 0.3835, 'consolidation_score': 0.1893}, 'dollar': {'trend_score': 0.4924, 'range_score': 0.2335, 'chop_score': 0.4615, 'volatility_percentile': 0.3815, 'consolidation_score': 0.1883}, 'gold': {'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}}
2026-05-11 10:57:26,680 INFO Regime[1H mode=ltf_behaviour] score means by year: {2016: {'trend_score': 0.495, 'range_score': 0.233, 'chop_score': 0.4607, 'volatility_percentile': 0.3776, 'consolidation_score': 0.192}, 2017: {'trend_score': 0.5017, 'range_score': 0.2328, 'chop_score': 0.4574, 'volatility_percentile': 0.3689, 'consolidation_score': 0.1942}, 2018: {'trend_score': 0.4886, 'range_score': 0.237, 'chop_score': 0.4655, 'volatility_percentile': 0.3947, 'consolidation_score': 0.1785}, 2019: {'trend_score': 0.4817, 'range_score': 0.236, 'chop_score': 0.4691, 'volatility_percentile': 0.3725, 'consolidation_score': 0.198}, 2020: {'trend_score': 0.4874, 'range_score': 0.2311, 'chop_score': 0.4674, 'volatility_percentile': 0.3848, 'consolidation_score': 0.1919}, 2021: {'trend_score': 0.4872, 'range_score': 0.2372, 'chop_score': 0.4672, 'volatility_percentile': 0.377, 'consolidation_score': 0.192}, 2022: {'trend_score': 0.4893, 'range_score': 0.2357, 'chop_score': 0.4647, 'volatility_percentile': 0.3962, 'consolidation_score': 0.1784}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 10:57:26,771 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
2026-05-11 10:57:26,773 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 10:57:26,774 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
2026-05-11 10:57:26,776 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 10:57:26,777 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
2026-05-11 10:57:26,778 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
2026-05-11 10:57:26,788 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:57:26,792 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:57:26,793 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:57:26,793 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:57:26,794 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 10:57:26,796 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:57:27,072 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected XAUUSD — 5117 samples (group=gold) score_means={'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}
2026-05-11 10:57:27,185 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:27,188 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:27,189 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:27,190 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:27,190 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:27,192 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:57:27,442 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURUSD — 5053 samples (group=dollar) score_means={'trend_score': 0.4883, 'range_score': 0.2413, 'chop_score': 0.463, 'volatility_percentile': 0.3837, 'consolidation_score': 0.1793}
2026-05-11 10:57:27,550 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:27,552 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:27,553 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:27,553 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:27,554 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:27,557 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:57:27,809 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected USDJPY — 5046 samples (group=dollar) score_means={'trend_score': 0.5085, 'range_score': 0.2259, 'chop_score': 0.4506, 'volatility_percentile': 0.3809, 'consolidation_score': 0.1888}
2026-05-11 10:57:27,922 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:27,924 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:27,925 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:27,926 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:27,926 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:27,928 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:57:28,172 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURJPY — 5053 samples (group=cross) score_means={'trend_score': 0.5023, 'range_score': 0.2261, 'chop_score': 0.4524, 'volatility_percentile': 0.3837, 'consolidation_score': 0.18}
2026-05-11 10:57:28,277 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:28,279 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:28,280 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:28,280 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:28,281 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:28,282 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:57:28,525 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPJPY — 5043 samples (group=cross) score_means={'trend_score': 0.4743, 'range_score': 0.2372, 'chop_score': 0.4716, 'volatility_percentile': 0.3777, 'consolidation_score': 0.1917}
2026-05-11 10:57:28,632 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:28,635 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:28,635 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:28,636 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:28,636 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 10:57:28,638 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 10:57:28,880 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPUSD — 5040 samples (group=dollar) score_means={'trend_score': 0.4843, 'range_score': 0.2371, 'chop_score': 0.4614, 'volatility_percentile': 0.377, 'consolidation_score': 0.1835}
2026-05-11 10:57:28,983 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2317, 'chop_score': 0.462, 'volatility_percentile': 0.3807, 'consolidation_score': 0.1858}, 'dollar': {'trend_score': 0.4937, 'range_score': 0.2348, 'chop_score': 0.4583, 'volatility_percentile': 0.3806, 'consolidation_score': 0.1839}, 'gold': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}}
2026-05-11 10:57:28,983 INFO Regime[1H mode=ltf_behaviour] score means by year: {2022: {'trend_score': 0.4905, 'range_score': 0.2364, 'chop_score': 0.4625, 'volatility_percentile': 0.385, 'consolidation_score': 0.1797}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 10:57:29,066 INFO Regime phase LTF dataset build fold=train_all: 8.2s (train=262644 val=30352)
2026-05-11 10:57:29,067 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260511_105729
2026-05-11 10:57:29,071 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=31, n_classes=5)
2026-05-11 10:57:29,071 INFO Regime 1H/ltf_behaviour warm start enabled from existing weights
2026-05-11 10:57:29,096 INFO RegimeClassifier[mode=ltf_behaviour]: warm start score head
2026-05-11 10:57:29,096 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-11 10:57:29,652 INFO Regime score epoch  1/50 — tr=0.0032 va=0.0007 mae={'trend_score': 0.0158, 'range_score': 0.0302, 'chop_score': 0.0183, 'volatility_percentile': 0.013, 'consolidation_score': 0.0184}
2026-05-11 10:57:30,173 INFO Regime score epoch  2/50 — tr=0.0032 va=0.0007
2026-05-11 10:57:30,687 INFO Regime score epoch  3/50 — tr=0.0032 va=0.0007
2026-05-11 10:57:31,225 INFO Regime score epoch  4/50 — tr=0.0032 va=0.0007
2026-05-11 10:57:31,783 INFO Regime score epoch  5/50 — tr=0.0031 va=0.0007 mae={'trend_score': 0.0156, 'range_score': 0.0301, 'chop_score': 0.0181, 'volatility_percentile': 0.013, 'consolidation_score': 0.0184}
2026-05-11 10:57:32,300 INFO Regime score epoch  6/50 — tr=0.0031 va=0.0007
2026-05-11 10:57:32,818 INFO Regime score epoch  7/50 — tr=0.0032 va=0.0007
2026-05-11 10:57:33,367 INFO Regime score epoch  8/50 — tr=0.0031 va=0.0007
2026-05-11 10:57:33,869 INFO Regime score epoch  9/50 — tr=0.0031 va=0.0007
2026-05-11 10:57:34,367 INFO Regime score epoch 10/50 — tr=0.0031 va=0.0007 mae={'trend_score': 0.0156, 'range_score': 0.0299, 'chop_score': 0.0179, 'volatility_percentile': 0.0136, 'consolidation_score': 0.0186}
2026-05-11 10:57:34,886 INFO Regime score epoch 11/50 — tr=0.0031 va=0.0007
2026-05-11 10:57:35,396 INFO Regime score epoch 12/50 — tr=0.0031 va=0.0007
2026-05-11 10:57:35,915 INFO Regime score epoch 13/50 — tr=0.0031 va=0.0007
2026-05-11 10:57:36,438 INFO Regime score epoch 14/50 — tr=0.0031 va=0.0007
2026-05-11 10:57:36,966 INFO Regime score epoch 15/50 — tr=0.0031 va=0.0007 mae={'trend_score': 0.0152, 'range_score': 0.0298, 'chop_score': 0.0176, 'volatility_percentile': 0.0128, 'consolidation_score': 0.0179}
2026-05-11 10:57:37,484 INFO Regime score epoch 16/50 — tr=0.0031 va=0.0007
2026-05-11 10:57:38,001 INFO Regime score epoch 17/50 — tr=0.0031 va=0.0007
2026-05-11 10:57:38,523 INFO Regime score epoch 18/50 — tr=0.0031 va=0.0007
2026-05-11 10:57:39,057 INFO Regime score epoch 19/50 — tr=0.0031 va=0.0007
2026-05-11 10:57:39,571 INFO Regime score epoch 20/50 — tr=0.0031 va=0.0007 mae={'trend_score': 0.0154, 'range_score': 0.0295, 'chop_score': 0.0176, 'volatility_percentile': 0.0126, 'consolidation_score': 0.018}
2026-05-11 10:57:40,061 INFO Regime score epoch 21/50 — tr=0.0031 va=0.0007
2026-05-11 10:57:40,545 INFO Regime score epoch 22/50 — tr=0.0031 va=0.0007
2026-05-11 10:57:41,051 INFO Regime score epoch 23/50 — tr=0.0030 va=0.0007
2026-05-11 10:57:41,598 INFO Regime score epoch 24/50 — tr=0.0030 va=0.0007
2026-05-11 10:57:42,121 INFO Regime score epoch 25/50 — tr=0.0030 va=0.0007 mae={'trend_score': 0.0154, 'range_score': 0.0293, 'chop_score': 0.0179, 'volatility_percentile': 0.0125, 'consolidation_score': 0.0178}
2026-05-11 10:57:42,627 INFO Regime score epoch 26/50 — tr=0.0030 va=0.0007
2026-05-11 10:57:43,133 INFO Regime score epoch 27/50 — tr=0.0030 va=0.0007
2026-05-11 10:57:43,633 INFO Regime score epoch 28/50 — tr=0.0030 va=0.0007
2026-05-11 10:57:44,138 INFO Regime score epoch 29/50 — tr=0.0030 va=0.0007
2026-05-11 10:57:44,653 INFO Regime score epoch 30/50 — tr=0.0030 va=0.0007 mae={'trend_score': 0.0152, 'range_score': 0.0292, 'chop_score': 0.0174, 'volatility_percentile': 0.0132, 'consolidation_score': 0.018}
2026-05-11 10:57:45,165 INFO Regime score epoch 31/50 — tr=0.0030 va=0.0007
2026-05-11 10:57:45,678 INFO Regime score epoch 32/50 — tr=0.0030 va=0.0007
2026-05-11 10:57:46,185 INFO Regime score epoch 33/50 — tr=0.0030 va=0.0007
2026-05-11 10:57:46,688 INFO Regime score epoch 34/50 — tr=0.0030 va=0.0007
2026-05-11 10:57:47,199 INFO Regime score epoch 35/50 — tr=0.0030 va=0.0007 mae={'trend_score': 0.0152, 'range_score': 0.0292, 'chop_score': 0.0173, 'volatility_percentile': 0.0124, 'consolidation_score': 0.0173}
2026-05-11 10:57:47,706 INFO Regime score epoch 36/50 — tr=0.0030 va=0.0007
2026-05-11 10:57:48,196 INFO Regime score epoch 37/50 — tr=0.0030 va=0.0007
2026-05-11 10:57:48,700 INFO Regime score epoch 38/50 — tr=0.0030 va=0.0007
2026-05-11 10:57:49,227 INFO Regime score epoch 39/50 — tr=0.0030 va=0.0007
2026-05-11 10:57:49,747 INFO Regime score epoch 40/50 — tr=0.0030 va=0.0007 mae={'trend_score': 0.0155, 'range_score': 0.0289, 'chop_score': 0.0173, 'volatility_percentile': 0.012, 'consolidation_score': 0.0174}
2026-05-11 10:57:50,259 INFO Regime score epoch 41/50 — tr=0.0030 va=0.0007
2026-05-11 10:57:50,766 INFO Regime score epoch 42/50 — tr=0.0030 va=0.0007
2026-05-11 10:57:51,308 INFO Regime score epoch 43/50 — tr=0.0030 va=0.0007
2026-05-11 10:57:51,874 INFO Regime score epoch 44/50 — tr=0.0030 va=0.0007
2026-05-11 10:57:52,390 INFO Regime score epoch 45/50 — tr=0.0030 va=0.0007 mae={'trend_score': 0.0152, 'range_score': 0.0291, 'chop_score': 0.0173, 'volatility_percentile': 0.0122, 'consolidation_score': 0.0172}
2026-05-11 10:57:52,926 INFO Regime score epoch 46/50 — tr=0.0030 va=0.0007
2026-05-11 10:57:53,453 INFO Regime score epoch 47/50 — tr=0.0030 va=0.0007
2026-05-11 10:57:53,977 INFO Regime score epoch 48/50 — tr=0.0030 va=0.0007
2026-05-11 10:57:54,503 INFO Regime score epoch 49/50 — tr=0.0030 va=0.0007
2026-05-11 10:57:55,016 INFO Regime score epoch 50/50 — tr=0.0030 va=0.0007 mae={'trend_score': 0.0154, 'range_score': 0.029, 'chop_score': 0.0174, 'volatility_percentile': 0.0128, 'consolidation_score': 0.0176}
2026-05-11 10:57:55,037 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0151, 'range_score': 0.0288, 'chop_score': 0.0172, 'volatility_percentile': 0.012, 'consolidation_score': 0.0174} mse={'trend_score': 0.0004, 'range_score': 0.00138, 'chop_score': 0.00049, 'volatility_percentile': 0.00029, 'consolidation_score': 0.00072} corr={'trend_score': 0.9959, 'range_score': 0.9672, 'chop_score': 0.9936, 'volatility_percentile': 0.9971, 'consolidation_score': 0.9929} pred_std={'trend_score': 0.2214, 'range_score': 0.1314, 'chop_score': 0.1819, 'volatility_percentile': 0.2201, 'consolidation_score': 0.2123} target_std={'trend_score': 0.2219, 'range_score': 0.1433, 'chop_score': 0.1888, 'volatility_percentile': 0.2165, 'consolidation_score': 0.2179}
2026-05-11 10:57:55,381 INFO RegimeClassifier[mode=ltf_behaviour] train group diagnostics={'dollar': {'n': 16632, 'score_mae': {'trend_score': 0.0147, 'range_score': 0.0287, 'chop_score': 0.0171, 'volatility_percentile': 0.0115, 'consolidation_score': 0.0178}, 'target_score_mean': {'trend_score': 0.491, 'range_score': 0.2347, 'chop_score': 0.4613, 'volatility_percentile': 0.3824, 'consolidation_score': 0.186}, 'pred_score_mean': {'trend_score': 0.4926, 'range_score': 0.2324, 'chop_score': 0.4604, 'volatility_percentile': 0.3817, 'consolidation_score': 0.1802}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3583, 62, 0, 4, 0, 0, 130], [3, 98, 0, 0, 0, 6, 3], [0, 0, 189, 12, 54, 0, 205], [1, 0, 3, 564, 29, 0, 92], [0, 0, 22, 18, 3105, 1, 170], [0, 16, 0, 0, 6, 76, 30], [126, 13, 49, 42, 44, 8, 7868]]}, 'gold': {'n': 8420, 'score_mae': {'trend_score': 0.0146, 'range_score': 0.0293, 'chop_score': 0.0174, 'volatility_percentile': 0.0121, 'consolidation_score': 0.0179}, 'target_score_mean': {'trend_score': 0.4883, 'range_score': 0.2355, 'chop_score': 0.4652, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}, 'pred_score_mean': {'trend_score': 0.4904, 'range_score': 0.2334, 'chop_score': 0.4641, 'volatility_percentile': 0.3755, 'consolidation_score': 0.1862}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1789, 35, 0, 0, 0, 0, 61], [4, 49, 0, 0, 0, 2, 1], [0, 0, 100, 11, 24, 0, 109], [1, 0, 2, 343, 14, 0, 56], [0, 0, 17, 18, 1591, 0, 78], [0, 10, 0, 0, 4, 49, 18], [57, 4, 37, 12, 38, 2, 3884]]}, 'jpy': {'n': 24948, 'score_mae': {'trend_score': 0.0147, 'range_score': 0.0289, 'chop_score': 0.0171, 'volatility_percentile': 0.0125, 'consolidation_score': 0.0177}, 'target_score_mean': {'trend_score': 0.4904, 'range_score': 0.2338, 'chop_score': 0.4653, 'volatility_percentile': 0.3826, 'consolidation_score': 0.1905}, 'pred_score_mean': {'trend_score': 0.4917, 'range_score': 0.2315, 'chop_score': 0.4641, 'volatility_percentile': 0.3812, 'consolidation_score': 0.1842}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[5338, 120, 0, 3, 0, 0, 185], [5, 169, 0, 0, 0, 6, 7], [0, 0, 244, 18, 84, 0, 301], [4, 0, 3, 1073, 65, 0, 169], [0, 0, 27, 49, 4768, 0, 271], [0, 28, 0, 0, 12, 113, 70], [188, 12, 73, 63, 105, 10, 11365]]}}
2026-05-11 10:57:55,563 INFO RegimeClassifier[mode=ltf_behaviour] validation group diagnostics={'dollar': {'n': 10093, 'score_mae': {'trend_score': 0.0154, 'range_score': 0.0291, 'chop_score': 0.0173, 'volatility_percentile': 0.0113, 'consolidation_score': 0.017}, 'target_score_mean': {'trend_score': 0.4863, 'range_score': 0.2392, 'chop_score': 0.4622, 'volatility_percentile': 0.3804, 'consolidation_score': 0.1814}, 'pred_score_mean': {'trend_score': 0.4878, 'range_score': 0.2356, 'chop_score': 0.4624, 'volatility_percentile': 0.3793, 'consolidation_score': 0.1763}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[2301, 22, 0, 1, 0, 0, 88], [3, 47, 0, 0, 0, 3, 0], [0, 0, 106, 7, 49, 0, 154], [0, 0, 0, 338, 24, 0, 61], [0, 0, 16, 16, 1941, 0, 77], [0, 8, 0, 0, 3, 42, 24], [54, 5, 26, 33, 39, 6, 4599]]}, 'gold': {'n': 5117, 'score_mae': {'trend_score': 0.0144, 'range_score': 0.0282, 'chop_score': 0.0171, 'volatility_percentile': 0.0119, 'consolidation_score': 0.0177}, 'target_score_mean': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}, 'pred_score_mean': {'trend_score': 0.4983, 'range_score': 0.2299, 'chop_score': 0.4565, 'volatility_percentile': 0.3799, 'consolidation_score': 0.1777}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1113, 14, 0, 0, 0, 0, 40], [3, 29, 0, 0, 0, 2, 1], [0, 0, 63, 3, 13, 0, 92], [0, 0, 2, 224, 8, 0, 21], [0, 0, 6, 10, 817, 1, 53], [0, 5, 0, 0, 3, 32, 10], [44, 2, 23, 18, 20, 2, 2443]]}, 'jpy': {'n': 15142, 'score_mae': {'trend_score': 0.0152, 'range_score': 0.0289, 'chop_score': 0.0171, 'volatility_percentile': 0.0124, 'consolidation_score': 0.0176}, 'target_score_mean': {'trend_score': 0.495, 'range_score': 0.2297, 'chop_score': 0.4582, 'volatility_percentile': 0.3808, 'consolidation_score': 0.1868}, 'pred_score_mean': {'trend_score': 0.4956, 'range_score': 0.2269, 'chop_score': 0.4582, 'volatility_percentile': 0.3794, 'consolidation_score': 0.1808}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3353, 47, 0, 1, 0, 0, 117], [3, 101, 0, 0, 0, 7, 4], [0, 0, 142, 13, 46, 0, 183], [2, 0, 2, 695, 33, 0, 95], [0, 0, 26, 31, 2608, 0, 152], [0, 14, 0, 0, 7, 68, 33], [112, 8, 45, 33, 57, 13, 7091]]}}
2026-05-11 10:57:55,568 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-11 10:57:55,569 INFO Regime phase LTF train fold=train_all: 26.5s
2026-05-11 10:57:55,670 INFO Regime LTF complete fold=train_all: score_accuracy=0.982, train=262644 val=30352 mae={'trend_score': 0.0151, 'range_score': 0.0288, 'chop_score': 0.0172, 'volatility_percentile': 0.012, 'consolidation_score': 0.0174}
2026-05-11 10:57:55,673 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 10:57:56,026 INFO Regime[1H mode=ltf_behaviour fold=latest] LTF score diagnostics on GBPUSD:
{'trend_score': {'mean': 0.4921, 'q10': 0.1918, 'q50': 0.4877, 'q90': 0.8002}, 'range_score': {'mean': 0.2337, 'q10': 0.0513, 'q50': 0.2125, 'q90': 0.4349}, 'chop_score': {'mean': 0.4612, 'q10': 0.2148, 'q50': 0.4503, 'q90': 0.7255}, 'volatility_percentile': {'mean': 0.3832, 'q10': 0.1018, 'q50': 0.3704, 'q90': 0.6823}, 'consolidation_score': {'mean': 0.1854, 'q10': 0.0, 'q50': 0.1185, 'q90': 0.506}}
2026-05-11 10:57:56,030 INFO Regime retrain total: 62.2s (370559 train+val samples)
2026-05-11 10:57:56,034 INFO Retrain complete. Total wall-clock: 62.2s
  DONE  Retrain regime [train-split retrain]
  SKIP  Quality/RL incremental retrain — clean train-only weights retained

=== Round 3: Backtest on last 3yr (post-retrain evaluation) ===
  START Round 3 - Post-retrain backtest (last 3yr)
2026-05-11 10:57:57,628 INFO === STEP 6: BACKTEST (round3) ===
2026-05-11 10:57:57,630 INFO BT_WINDOW=round3 — post-retrain eval: 2022-08-05 → 2025-08-05 (last 3yr)
2026-05-11 10:57:57,630 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-05-11 10:57:57,630 INFO Round 3 — running backtest: 2022-08-05 → 2025-08-05 (ml_trader, shared ML cache)
2026-05-11 10:57:57,631 INFO Round 3 — QualityScorer hard gate disabled; set BACKTEST_ENABLE_QUALITY_GATE=1 to enable it
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
2026-05-11 10:59:46,360 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:812: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 10:59:46,824 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:812: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 10:59:46,928 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:814: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:818: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 10:59:47,242 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:814: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:822: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:818: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:826: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 10:59:47,333 WARNING _build_sequence_df: HTF frame 5M filled 461 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:812: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:830: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:822: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:826: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:834: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:836: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:830: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 10:59:47,446 WARNING _build_sequence_df: HTF frame 5M filled 461 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:814: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:834: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:836: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:818: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 10:59:47,540 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:812: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:822: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:826: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:830: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:834: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 10:59:47,646 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:836: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:814: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:818: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:822: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:826: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:830: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:834: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:836: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
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
2026-05-11 10:59:59,815 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:812: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 10:59:59,851 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:814: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:818: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:822: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:826: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 10:59:59,962 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:830: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:812: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:834: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:836: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 11:00:00,000 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:814: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:818: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:822: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:826: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:830: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:834: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:836: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 11:01:08,678 INFO Round 3 backtest — 4 trades | avg WR=0.0% | avg PF=0.00 | avg Sharpe=-15890380336.99
2026-05-11 11:01:08,678 INFO   ml_trader: 4 trades | WR=0.0% | fixed PF=0.00 | Return=-4.0% | ExpR=-1.001 | DD=4.0% | Sharpe=-15890380336.99
2026-05-11 11:01:08,678 INFO   ml_trader gate_diagnostics: bars=403523 no_signal=291690 quality_block=0 session_skip=111829 density=0 pm_reject=0
2026-05-11 11:01:08,678 INFO   ml_trader no_signal_reasons: {'no_trade_chop': 25003, 'weak_gru_direction': 75818, 'no_trade_uncertain': 101657, 'gru_expected_r_below_threshold': 65774, 'no_trade_extreme_vol': 18961, 'tradeability_direction_conflict': 823, 'htf_low_regime_confidence': 3213, 'wait_pullback': 346, 'trend_structure_missing': 95}
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 4
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (4 rows)
  DONE  Round 3 - Post-retrain backtest (last 3yr)
  Saved Round 3 result → round3_summary.json
  Journal after Round 3: 12 entries

  SKIP  Round 3 Quality+RL retrain — evaluation journals not used for fitting

======================================================================
  BLIND BACKTEST PIPELINE COMPLETE
======================================================================
  Round 1 (train-tail window)   trades=4  WR=50.0%  PF=1.439  Sharpe=2.454
  Round 2 (blind test)          trades=4  WR=0.0%  PF=0.000  Sharpe=-15890380336.988
  Round 3 (last 3yr)            trades=4  WR=0.0%  PF=0.000  Sharpe=-15890380336.988


WARNING: GITHUB_TOKEN not set — skipping GitHub push