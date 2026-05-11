  Cleared done-check: training_summary.json
  Cleared done-check: training_7b_train_summary.json
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
2026-05-11 17:25:09,319 INFO Loading feature-engineered data...
2026-05-11 17:25:09,969 INFO Loaded 221743 rows, 202 features
2026-05-11 17:25:09,970 INFO Data span: 2016-01-04 -> 2025-08-05  (9.6 years)
2026-05-11 17:25:09,972 INFO Fold 000 train 2016-01-04 -> 2018-01-03 (47088 bars), val 2018-01-04 -> 2019-01-03 (23448 bars)
2026-05-11 17:25:09,973 INFO Fold 001 train 2016-01-04 -> 2019-01-03 (70536 bars), val 2019-01-04 -> 2020-01-03 (23377 bars)
2026-05-11 17:25:09,973 INFO Fold 002 train 2016-01-04 -> 2020-01-03 (93913 bars), val 2020-01-06 -> 2020-12-31 (23259 bars)
2026-05-11 17:25:09,973 INFO Fold 003 train 2016-01-04 -> 2020-12-31 (117172 bars), val 2021-01-04 -> 2022-01-03 (23507 bars)
2026-05-11 17:25:09,973 INFO Fold 004 train 2016-01-04 -> 2022-01-03 (140679 bars), val 2022-01-04 -> 2023-01-03 (23588 bars)
2026-05-11 17:25:09,974 INFO Fold 005 train 2016-01-04 -> 2022-08-05 (154539 bars), val 2022-08-05 -> 2023-08-04 (20412 bars)
2026-05-11 17:25:09,974 INFO No leakage confirmed: train/train_tail/internal folds end before final 2-year blind test

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
2026-05-11 17:25:19,707 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-05-11 17:25:19,708 INFO --- Training gru ---
2026-05-11 17:25:19,708 INFO Running retrain --model gru
2026-05-11 17:25:19,916 INFO retrain environment: KAGGLE
2026-05-11 17:25:21,542 INFO Device: CUDA (2 GPU(s))
2026-05-11 17:25:21,553 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 17:25:21,553 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 17:25:21,553 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 17:25:21,555 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 17:25:21,555 INFO Retrain data split: train
2026-05-11 17:25:21,555 INFO Retrain rolling fold selector: latest
2026-05-11 17:25:21,556 INFO === GRU-LSTM retrain (timeframes: ['15M']) ===
2026-05-11 17:25:21,717 INFO NumExpr defaulting to 4 threads.
2026-05-11 17:25:21,940 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-11 17:25:21,940 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 17:25:21,940 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 17:25:22,317 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-05-11 17:25:22,318 INFO GRU multi-symbol training (Kaggle mode): 6 symbols × ['15M']
2026-05-11 17:25:22,320 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260511_172522
2026-05-11 17:25:22,325 INFO GRU feature contract unchanged (input_size=94) — incremental retrain
2026-05-11 17:25:22,325 INFO Deleted stale GRU artifact for cold start: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:25:22,326 INFO GRU warm start disabled by default; set GRU_ALLOW_WARM_START=1 to reuse compatible weights
2026-05-11 17:25:22,597 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 17:25:22,626 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 17:25:22,642 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 17:25:22,653 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 17:25:22,730 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 17:25:22,737 INFO Loaded XAUUSD/15M split=train fold=latest: 176438 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:25:23,377 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:25:23,396 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:25:23,424 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:25:23,432 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:25:23,480 INFO Loaded EURUSD/15M split=train fold=latest: 174951 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:25:24,068 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:25:24,090 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:25:24,105 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:25:24,113 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:25:24,153 INFO Loaded USDJPY/15M split=train fold=latest: 174944 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:25:24,759 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:25:24,779 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:25:24,796 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:25:24,804 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:25:24,845 INFO Loaded EURJPY/15M split=train fold=latest: 174948 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:25:25,413 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:25:25,436 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:25:25,452 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:25:25,461 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:25:25,504 INFO Loaded GBPJPY/15M split=train fold=latest: 174902 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:25:26,066 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:25:26,086 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:25:26,102 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:25:26,111 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:25:26,150 INFO Loaded GBPUSD/15M split=train fold=latest: 174899 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:25:26,601 INFO train_multi: 6 segments, ~971854 total bars
2026-05-11 17:25:26,609 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-05-11 17:25:26,609 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-05-11 17:25:26,609 INFO train_multi: training ALL 6 segments across TFs ['15M'] in one combined pass
2026-05-11 17:25:26,609 INFO train_multi: building combined dataset for TF=ALL (6 segments)
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
2026-05-11 17:25:39,487 INFO train_multi TF=ALL: 971494 sequences across 6 segments
2026-05-11 17:25:39,487 INFO train_multi TF=ALL: estimated peak RAM = 27072 MB (train=419996 calib=60000 val=120002 n_feat=94 seq_len=60)
2026-05-11 17:25:39,488 WARNING train_multi TF=ALL: trimming to fit RAM budget — new train=310283 calib=44326 val=88652 (20000 MB est)
2026-05-11 17:25:41,882 INFO train_multi TF=ALL: train=310283 calib=44326 val=88652 (10007 MB tensors)
2026-05-11 17:25:49,094 INFO train_multi TF=ALL: structural bar weighting — 199279 structural bars (64.2%) weight=15.0 structural_only=0
2026-05-11 17:25:52,291 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=25, min_epochs=45)
2026-05-11 17:26:08,694 INFO train_multi TF=ALL epoch 1/100 train=2.3347 val=2.3346 r_mae=0.967 pos_r_acc=0.545 side_acc=0.510 r_n=127469
2026-05-11 17:26:08,700 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:26:08,700 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:26:08,700 INFO train_multi TF=ALL: new best val=2.3346 r_mae=0.9667 — saved
2026-05-11 17:26:08,704 INFO train_multi TF=ALL: new best r_mae=0.9667 — saved rmae checkpoint
2026-05-11 17:26:22,107 INFO train_multi TF=ALL epoch 2/100 train=2.3327 val=2.3337 r_mae=0.966 pos_r_acc=0.545 side_acc=0.510 r_n=127469
2026-05-11 17:26:22,112 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:26:22,112 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:26:22,113 INFO train_multi TF=ALL: new best val=2.3337 r_mae=0.9662 — saved
2026-05-11 17:26:22,117 INFO train_multi TF=ALL: new best r_mae=0.9662 — saved rmae checkpoint
2026-05-11 17:26:35,789 INFO train_multi TF=ALL epoch 3/100 train=2.3318 val=2.3329 r_mae=0.966 pos_r_acc=0.545 side_acc=0.510 r_n=127469
2026-05-11 17:26:35,794 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:26:35,795 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:26:35,795 INFO train_multi TF=ALL: new best val=2.3329 r_mae=0.9655 — saved
2026-05-11 17:26:35,799 INFO train_multi TF=ALL: new best r_mae=0.9655 — saved rmae checkpoint
2026-05-11 17:26:49,500 INFO train_multi TF=ALL epoch 4/100 train=2.3306 val=2.3324 r_mae=0.965 pos_r_acc=0.545 side_acc=0.513 r_n=127469
2026-05-11 17:26:49,505 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:26:49,505 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:26:49,505 INFO train_multi TF=ALL: new best val=2.3324 r_mae=0.9650 — saved
2026-05-11 17:26:49,510 INFO train_multi TF=ALL: new best r_mae=0.9650 — saved rmae checkpoint
2026-05-11 17:27:03,409 INFO train_multi TF=ALL epoch 5/100 train=2.3305 val=2.3323 r_mae=0.965 pos_r_acc=0.545 side_acc=0.516 r_n=127469
2026-05-11 17:27:03,414 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:27:03,414 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:27:03,414 INFO train_multi TF=ALL: new best val=2.3323 r_mae=0.9647 — saved
2026-05-11 17:27:03,419 INFO train_multi TF=ALL: new best r_mae=0.9647 — saved rmae checkpoint
2026-05-11 17:27:16,965 INFO train_multi TF=ALL epoch 6/100 train=2.3305 val=2.3320 r_mae=0.965 pos_r_acc=0.545 side_acc=0.519 r_n=127469
2026-05-11 17:27:16,970 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:27:16,970 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:27:16,970 INFO train_multi TF=ALL: new best val=2.3320 r_mae=0.9647 — saved
2026-05-11 17:27:16,974 INFO train_multi TF=ALL: new best r_mae=0.9647 — saved rmae checkpoint
2026-05-11 17:27:30,558 INFO train_multi TF=ALL epoch 7/100 train=2.3298 val=2.3316 r_mae=0.965 pos_r_acc=0.545 side_acc=0.524 r_n=127469
2026-05-11 17:27:30,563 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:27:30,564 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:27:30,564 INFO train_multi TF=ALL: new best val=2.3316 r_mae=0.9646 — saved
2026-05-11 17:27:30,568 INFO train_multi TF=ALL: new best r_mae=0.9646 — saved rmae checkpoint
2026-05-11 17:27:44,047 INFO train_multi TF=ALL epoch 8/100 train=2.3293 val=2.3305 r_mae=0.964 pos_r_acc=0.545 side_acc=0.528 r_n=127469
2026-05-11 17:27:44,053 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:27:44,053 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:27:44,053 INFO train_multi TF=ALL: new best val=2.3305 r_mae=0.9643 — saved
2026-05-11 17:27:44,057 INFO train_multi TF=ALL: new best r_mae=0.9643 — saved rmae checkpoint
2026-05-11 17:27:57,652 INFO train_multi TF=ALL epoch 9/100 train=2.3282 val=2.3295 r_mae=0.964 pos_r_acc=0.545 side_acc=0.520 r_n=127469
2026-05-11 17:27:57,657 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:27:57,657 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:27:57,657 INFO train_multi TF=ALL: new best val=2.3295 r_mae=0.9636 — saved
2026-05-11 17:27:57,662 INFO train_multi TF=ALL: new best r_mae=0.9636 — saved rmae checkpoint
2026-05-11 17:28:11,297 INFO train_multi TF=ALL epoch 10/100 train=2.3260 val=2.3285 r_mae=0.963 pos_r_acc=0.545 side_acc=0.521 r_n=127469
2026-05-11 17:28:11,302 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:28:11,302 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:28:11,302 INFO train_multi TF=ALL: new best val=2.3285 r_mae=0.9627 — saved
2026-05-11 17:28:11,307 INFO train_multi TF=ALL: new best r_mae=0.9627 — saved rmae checkpoint
2026-05-11 17:28:24,824 INFO train_multi TF=ALL epoch 11/100 train=2.3234 val=2.3252 r_mae=0.962 pos_r_acc=0.545 side_acc=0.527 r_n=127469
2026-05-11 17:28:24,829 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:28:24,829 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:28:24,829 INFO train_multi TF=ALL: new best val=2.3252 r_mae=0.9619 — saved
2026-05-11 17:28:24,834 INFO train_multi TF=ALL: new best r_mae=0.9619 — saved rmae checkpoint
2026-05-11 17:28:38,653 INFO train_multi TF=ALL epoch 12/100 train=2.3213 val=2.3230 r_mae=0.961 pos_r_acc=0.546 side_acc=0.528 r_n=127469
2026-05-11 17:28:38,659 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:28:38,659 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:28:38,659 INFO train_multi TF=ALL: new best val=2.3230 r_mae=0.9615 — saved
2026-05-11 17:28:38,663 INFO train_multi TF=ALL: new best r_mae=0.9615 — saved rmae checkpoint
2026-05-11 17:28:52,204 INFO train_multi TF=ALL epoch 13/100 train=2.3183 val=2.3202 r_mae=0.961 pos_r_acc=0.546 side_acc=0.531 r_n=127469
2026-05-11 17:28:52,215 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:28:52,215 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:28:52,215 INFO train_multi TF=ALL: new best val=2.3202 r_mae=0.9612 — saved
2026-05-11 17:28:52,220 INFO train_multi TF=ALL: new best r_mae=0.9612 — saved rmae checkpoint
2026-05-11 17:29:05,965 INFO train_multi TF=ALL epoch 14/100 train=2.3147 val=2.3173 r_mae=0.961 pos_r_acc=0.547 side_acc=0.535 r_n=127469
2026-05-11 17:29:05,970 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:29:05,970 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:29:05,970 INFO train_multi TF=ALL: new best val=2.3173 r_mae=0.9606 — saved
2026-05-11 17:29:05,974 INFO train_multi TF=ALL: new best r_mae=0.9606 — saved rmae checkpoint
2026-05-11 17:29:19,400 INFO train_multi TF=ALL epoch 15/100 train=2.3132 val=2.3168 r_mae=0.960 pos_r_acc=0.547 side_acc=0.535 r_n=127469
2026-05-11 17:29:19,406 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:29:19,406 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:29:19,406 INFO train_multi TF=ALL: new best val=2.3168 r_mae=0.9603 — saved
2026-05-11 17:29:19,411 INFO train_multi TF=ALL: new best r_mae=0.9603 — saved rmae checkpoint
2026-05-11 17:29:32,883 INFO train_multi TF=ALL epoch 16/100 train=2.3114 val=2.3155 r_mae=0.959 pos_r_acc=0.549 side_acc=0.536 r_n=127469
2026-05-11 17:29:32,893 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:29:32,893 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:29:32,893 INFO train_multi TF=ALL: new best val=2.3155 r_mae=0.9594 — saved
2026-05-11 17:29:32,898 INFO train_multi TF=ALL: new best r_mae=0.9594 — saved rmae checkpoint
2026-05-11 17:29:46,549 INFO train_multi TF=ALL epoch 17/100 train=2.3098 val=2.3139 r_mae=0.959 pos_r_acc=0.551 side_acc=0.539 r_n=127469
2026-05-11 17:29:46,555 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:29:46,555 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:29:46,555 INFO train_multi TF=ALL: new best val=2.3139 r_mae=0.9589 — saved
2026-05-11 17:29:46,559 INFO train_multi TF=ALL: new best r_mae=0.9589 — saved rmae checkpoint
2026-05-11 17:30:00,289 INFO train_multi TF=ALL epoch 18/100 train=2.3076 val=2.3135 r_mae=0.959 pos_r_acc=0.552 side_acc=0.537 r_n=127469
2026-05-11 17:30:00,295 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:30:00,296 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:30:00,296 INFO train_multi TF=ALL: new best val=2.3135 r_mae=0.9586 — saved
2026-05-11 17:30:00,301 INFO train_multi TF=ALL: new best r_mae=0.9586 — saved rmae checkpoint
2026-05-11 17:30:13,758 INFO train_multi TF=ALL epoch 19/100 train=2.3047 val=2.3127 r_mae=0.957 pos_r_acc=0.553 side_acc=0.538 r_n=127469
2026-05-11 17:30:13,763 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:30:13,763 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:30:13,763 INFO train_multi TF=ALL: new best val=2.3127 r_mae=0.9574 — saved
2026-05-11 17:30:13,767 INFO train_multi TF=ALL: new best r_mae=0.9574 — saved rmae checkpoint
2026-05-11 17:30:27,426 INFO train_multi TF=ALL epoch 20/100 train=2.3025 val=2.3109 r_mae=0.956 pos_r_acc=0.554 side_acc=0.539 r_n=127469
2026-05-11 17:30:27,432 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:30:27,432 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:30:27,432 INFO train_multi TF=ALL: new best val=2.3109 r_mae=0.9564 — saved
2026-05-11 17:30:27,436 INFO train_multi TF=ALL: new best r_mae=0.9564 — saved rmae checkpoint
2026-05-11 17:30:40,901 INFO train_multi TF=ALL epoch 21/100 train=2.2999 val=2.3090 r_mae=0.956 pos_r_acc=0.555 side_acc=0.541 r_n=127469
2026-05-11 17:30:40,907 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:30:40,907 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:30:40,907 INFO train_multi TF=ALL: new best val=2.3090 r_mae=0.9558 — saved
2026-05-11 17:30:40,912 INFO train_multi TF=ALL: new best r_mae=0.9558 — saved rmae checkpoint
2026-05-11 17:30:54,322 INFO train_multi TF=ALL epoch 22/100 train=2.2976 val=2.3079 r_mae=0.956 pos_r_acc=0.556 side_acc=0.543 r_n=127469
2026-05-11 17:30:54,332 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:30:54,332 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:30:54,332 INFO train_multi TF=ALL: new best val=2.3079 r_mae=0.9559 — saved
2026-05-11 17:31:07,728 INFO train_multi TF=ALL epoch 23/100 train=2.2935 val=2.3108 r_mae=0.955 pos_r_acc=0.558 side_acc=0.535 r_n=127469
2026-05-11 17:31:07,733 INFO train_multi TF=ALL: new best r_mae=0.9548 — saved rmae checkpoint
2026-05-11 17:31:21,400 INFO train_multi TF=ALL epoch 24/100 train=2.2906 val=2.3010 r_mae=0.951 pos_r_acc=0.561 side_acc=0.541 r_n=127469
2026-05-11 17:31:21,406 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:31:21,406 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:31:21,406 INFO train_multi TF=ALL: new best val=2.3010 r_mae=0.9514 — saved
2026-05-11 17:31:21,410 INFO train_multi TF=ALL: new best r_mae=0.9514 — saved rmae checkpoint
2026-05-11 17:31:34,764 INFO train_multi TF=ALL epoch 25/100 train=2.2832 val=2.2949 r_mae=0.947 pos_r_acc=0.567 side_acc=0.544 r_n=127469
2026-05-11 17:31:34,774 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:31:34,774 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:31:34,774 INFO train_multi TF=ALL: new best val=2.2949 r_mae=0.9465 — saved
2026-05-11 17:31:34,778 INFO train_multi TF=ALL: new best r_mae=0.9465 — saved rmae checkpoint
2026-05-11 17:31:48,518 INFO train_multi TF=ALL epoch 26/100 train=2.2744 val=2.2857 r_mae=0.946 pos_r_acc=0.572 side_acc=0.550 r_n=127469
2026-05-11 17:31:48,524 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:31:48,524 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:31:48,524 INFO train_multi TF=ALL: new best val=2.2857 r_mae=0.9464 — saved
2026-05-11 17:31:48,528 INFO train_multi TF=ALL: new best r_mae=0.9464 — saved rmae checkpoint
2026-05-11 17:32:02,073 INFO train_multi TF=ALL epoch 27/100 train=2.2646 val=2.2797 r_mae=0.939 pos_r_acc=0.578 side_acc=0.556 r_n=127469
2026-05-11 17:32:02,078 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:32:02,078 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:32:02,078 INFO train_multi TF=ALL: new best val=2.2797 r_mae=0.9392 — saved
2026-05-11 17:32:02,082 INFO train_multi TF=ALL: new best r_mae=0.9392 — saved rmae checkpoint
2026-05-11 17:32:16,077 INFO train_multi TF=ALL epoch 28/100 train=2.2537 val=2.2736 r_mae=0.933 pos_r_acc=0.580 side_acc=0.555 r_n=127469
2026-05-11 17:32:16,086 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:32:16,086 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:32:16,086 INFO train_multi TF=ALL: new best val=2.2736 r_mae=0.9334 — saved
2026-05-11 17:32:16,090 INFO train_multi TF=ALL: new best r_mae=0.9334 — saved rmae checkpoint
2026-05-11 17:32:29,593 INFO train_multi TF=ALL epoch 29/100 train=2.2481 val=2.2736 r_mae=0.929 pos_r_acc=0.585 side_acc=0.557 r_n=127469
2026-05-11 17:32:29,599 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:32:29,599 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:32:29,599 INFO train_multi TF=ALL: new best val=2.2736 r_mae=0.9293 — saved
2026-05-11 17:32:29,604 INFO train_multi TF=ALL: new best r_mae=0.9293 — saved rmae checkpoint
2026-05-11 17:32:43,101 INFO train_multi TF=ALL epoch 30/100 train=2.2409 val=2.2636 r_mae=0.930 pos_r_acc=0.585 side_acc=0.562 r_n=127469
2026-05-11 17:32:43,106 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:32:43,106 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:32:43,106 INFO train_multi TF=ALL: new best val=2.2636 r_mae=0.9301 — saved
2026-05-11 17:32:56,670 INFO train_multi TF=ALL epoch 31/100 train=2.2295 val=2.2606 r_mae=0.927 pos_r_acc=0.591 side_acc=0.561 r_n=127469
2026-05-11 17:32:56,675 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:32:56,675 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:32:56,675 INFO train_multi TF=ALL: new best val=2.2606 r_mae=0.9266 — saved
2026-05-11 17:32:56,680 INFO train_multi TF=ALL: new best r_mae=0.9266 — saved rmae checkpoint
2026-05-11 17:33:10,193 INFO train_multi TF=ALL epoch 32/100 train=2.2234 val=2.2720 r_mae=0.926 pos_r_acc=0.586 side_acc=0.554 r_n=127469
2026-05-11 17:33:10,198 INFO train_multi TF=ALL: new best r_mae=0.9255 — saved rmae checkpoint
2026-05-11 17:33:23,753 INFO train_multi TF=ALL epoch 33/100 train=2.2171 val=2.2594 r_mae=0.922 pos_r_acc=0.591 side_acc=0.565 r_n=127469
2026-05-11 17:33:23,763 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:33:23,763 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:33:23,763 INFO train_multi TF=ALL: new best val=2.2594 r_mae=0.9216 — saved
2026-05-11 17:33:23,767 INFO train_multi TF=ALL: new best r_mae=0.9216 — saved rmae checkpoint
2026-05-11 17:33:37,251 INFO train_multi TF=ALL epoch 34/100 train=2.2095 val=2.2552 r_mae=0.922 pos_r_acc=0.592 side_acc=0.565 r_n=127469
2026-05-11 17:33:37,256 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:33:37,256 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:33:37,257 INFO train_multi TF=ALL: new best val=2.2552 r_mae=0.9222 — saved
2026-05-11 17:33:50,863 INFO train_multi TF=ALL epoch 35/100 train=2.2053 val=2.2513 r_mae=0.920 pos_r_acc=0.591 side_acc=0.571 r_n=127469
2026-05-11 17:33:50,869 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:33:50,869 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:33:50,869 INFO train_multi TF=ALL: new best val=2.2513 r_mae=0.9201 — saved
2026-05-11 17:33:50,873 INFO train_multi TF=ALL: new best r_mae=0.9201 — saved rmae checkpoint
2026-05-11 17:34:04,246 INFO train_multi TF=ALL epoch 36/100 train=2.1979 val=2.2550 r_mae=0.915 pos_r_acc=0.593 side_acc=0.573 r_n=127469
2026-05-11 17:34:04,251 INFO train_multi TF=ALL: new best r_mae=0.9149 — saved rmae checkpoint
2026-05-11 17:34:17,699 INFO train_multi TF=ALL epoch 37/100 train=2.1886 val=2.2463 r_mae=0.912 pos_r_acc=0.596 side_acc=0.577 r_n=127469
2026-05-11 17:34:17,705 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:34:17,705 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:34:17,705 INFO train_multi TF=ALL: new best val=2.2463 r_mae=0.9123 — saved
2026-05-11 17:34:17,710 INFO train_multi TF=ALL: new best r_mae=0.9123 — saved rmae checkpoint
2026-05-11 17:34:31,358 INFO train_multi TF=ALL epoch 38/100 train=2.1795 val=2.2370 r_mae=0.914 pos_r_acc=0.596 side_acc=0.581 r_n=127469
2026-05-11 17:34:31,368 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:34:31,368 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:34:31,368 INFO train_multi TF=ALL: new best val=2.2370 r_mae=0.9141 — saved
2026-05-11 17:34:45,181 INFO train_multi TF=ALL epoch 39/100 train=2.1662 val=2.2322 r_mae=0.908 pos_r_acc=0.599 side_acc=0.586 r_n=127469
2026-05-11 17:34:45,186 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:34:45,186 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:34:45,186 INFO train_multi TF=ALL: new best val=2.2322 r_mae=0.9076 — saved
2026-05-11 17:34:45,190 INFO train_multi TF=ALL: new best r_mae=0.9076 — saved rmae checkpoint
2026-05-11 17:34:58,932 INFO train_multi TF=ALL epoch 40/100 train=2.1572 val=2.2224 r_mae=0.905 pos_r_acc=0.603 side_acc=0.591 r_n=127469
2026-05-11 17:34:58,937 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:34:58,938 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:34:58,938 INFO train_multi TF=ALL: new best val=2.2224 r_mae=0.9048 — saved
2026-05-11 17:34:58,942 INFO train_multi TF=ALL: new best r_mae=0.9048 — saved rmae checkpoint
2026-05-11 17:35:12,406 INFO train_multi TF=ALL epoch 41/100 train=2.1444 val=2.2118 r_mae=0.900 pos_r_acc=0.609 side_acc=0.592 r_n=127469
2026-05-11 17:35:12,411 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:35:12,411 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:35:12,411 INFO train_multi TF=ALL: new best val=2.2118 r_mae=0.8996 — saved
2026-05-11 17:35:12,415 INFO train_multi TF=ALL: new best r_mae=0.8996 — saved rmae checkpoint
2026-05-11 17:35:25,850 INFO train_multi TF=ALL epoch 42/100 train=2.1329 val=2.1996 r_mae=0.896 pos_r_acc=0.613 side_acc=0.599 r_n=127469
2026-05-11 17:35:25,856 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:35:25,857 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:35:25,857 INFO train_multi TF=ALL: new best val=2.1996 r_mae=0.8961 — saved
2026-05-11 17:35:25,861 INFO train_multi TF=ALL: new best r_mae=0.8961 — saved rmae checkpoint
2026-05-11 17:35:39,308 INFO train_multi TF=ALL epoch 43/100 train=2.1203 val=2.1904 r_mae=0.885 pos_r_acc=0.623 side_acc=0.603 r_n=127469
2026-05-11 17:35:39,319 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:35:39,319 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:35:39,319 INFO train_multi TF=ALL: new best val=2.1904 r_mae=0.8853 — saved
2026-05-11 17:35:39,323 INFO train_multi TF=ALL: new best r_mae=0.8853 — saved rmae checkpoint
2026-05-11 17:35:52,794 INFO train_multi TF=ALL epoch 44/100 train=2.0996 val=2.1762 r_mae=0.881 pos_r_acc=0.623 side_acc=0.605 r_n=127469
2026-05-11 17:35:52,799 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:35:52,799 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:35:52,800 INFO train_multi TF=ALL: new best val=2.1762 r_mae=0.8809 — saved
2026-05-11 17:35:52,804 INFO train_multi TF=ALL: new best r_mae=0.8809 — saved rmae checkpoint
2026-05-11 17:36:06,408 INFO train_multi TF=ALL epoch 45/100 train=2.0846 val=2.1617 r_mae=0.872 pos_r_acc=0.632 side_acc=0.612 r_n=127469
2026-05-11 17:36:06,414 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:36:06,414 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:36:06,414 INFO train_multi TF=ALL: new best val=2.1617 r_mae=0.8722 — saved
2026-05-11 17:36:06,418 INFO train_multi TF=ALL: new best r_mae=0.8722 — saved rmae checkpoint
2026-05-11 17:36:20,061 INFO train_multi TF=ALL epoch 46/100 train=2.0685 val=2.1474 r_mae=0.861 pos_r_acc=0.637 side_acc=0.619 r_n=127469
2026-05-11 17:36:20,071 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:36:20,071 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:36:20,071 INFO train_multi TF=ALL: new best val=2.1474 r_mae=0.8610 — saved
2026-05-11 17:36:20,075 INFO train_multi TF=ALL: new best r_mae=0.8610 — saved rmae checkpoint
2026-05-11 17:36:33,588 INFO train_multi TF=ALL epoch 47/100 train=2.0567 val=2.1422 r_mae=0.855 pos_r_acc=0.643 side_acc=0.616 r_n=127469
2026-05-11 17:36:33,598 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:36:33,598 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:36:33,598 INFO train_multi TF=ALL: new best val=2.1422 r_mae=0.8551 — saved
2026-05-11 17:36:33,603 INFO train_multi TF=ALL: new best r_mae=0.8551 — saved rmae checkpoint
2026-05-11 17:36:47,326 INFO train_multi TF=ALL epoch 48/100 train=2.0420 val=2.1669 r_mae=0.857 pos_r_acc=0.630 side_acc=0.613 r_n=127469
2026-05-11 17:37:00,985 INFO train_multi TF=ALL epoch 49/100 train=2.0315 val=2.1131 r_mae=0.850 pos_r_acc=0.651 side_acc=0.627 r_n=127469
2026-05-11 17:37:00,990 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:37:00,991 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:37:00,991 INFO train_multi TF=ALL: new best val=2.1131 r_mae=0.8498 — saved
2026-05-11 17:37:00,995 INFO train_multi TF=ALL: new best r_mae=0.8498 — saved rmae checkpoint
2026-05-11 17:37:14,438 INFO train_multi TF=ALL epoch 50/100 train=2.0210 val=2.1139 r_mae=0.843 pos_r_acc=0.648 side_acc=0.628 r_n=127469
2026-05-11 17:37:14,443 INFO train_multi TF=ALL: new best r_mae=0.8426 — saved rmae checkpoint
2026-05-11 17:37:27,846 INFO train_multi TF=ALL epoch 51/100 train=2.0052 val=2.1076 r_mae=0.838 pos_r_acc=0.654 side_acc=0.629 r_n=127469
2026-05-11 17:37:27,856 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:37:27,856 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:37:27,856 INFO train_multi TF=ALL: new best val=2.1076 r_mae=0.8376 — saved
2026-05-11 17:37:27,861 INFO train_multi TF=ALL: new best r_mae=0.8376 — saved rmae checkpoint
2026-05-11 17:37:41,610 INFO train_multi TF=ALL epoch 52/100 train=1.9949 val=2.1061 r_mae=0.836 pos_r_acc=0.652 side_acc=0.634 r_n=127469
2026-05-11 17:37:41,615 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:37:41,615 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:37:41,616 INFO train_multi TF=ALL: new best val=2.1061 r_mae=0.8359 — saved
2026-05-11 17:37:41,620 INFO train_multi TF=ALL: new best r_mae=0.8359 — saved rmae checkpoint
2026-05-11 17:37:55,162 INFO train_multi TF=ALL epoch 53/100 train=1.9847 val=2.1086 r_mae=0.828 pos_r_acc=0.655 side_acc=0.630 r_n=127469
2026-05-11 17:37:55,167 INFO train_multi TF=ALL: new best r_mae=0.8284 — saved rmae checkpoint
2026-05-11 17:38:08,704 INFO train_multi TF=ALL epoch 54/100 train=1.9725 val=2.1053 r_mae=0.829 pos_r_acc=0.656 side_acc=0.632 r_n=127469
2026-05-11 17:38:08,709 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:38:08,709 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:38:08,709 INFO train_multi TF=ALL: new best val=2.1053 r_mae=0.8290 — saved
2026-05-11 17:38:22,071 INFO train_multi TF=ALL epoch 55/100 train=1.9648 val=2.0936 r_mae=0.826 pos_r_acc=0.655 side_acc=0.642 r_n=127469
2026-05-11 17:38:22,082 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:38:22,082 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:38:22,082 INFO train_multi TF=ALL: new best val=2.0936 r_mae=0.8263 — saved
2026-05-11 17:38:22,086 INFO train_multi TF=ALL: new best r_mae=0.8263 — saved rmae checkpoint
2026-05-11 17:38:35,406 INFO train_multi TF=ALL epoch 56/100 train=1.9512 val=2.0874 r_mae=0.821 pos_r_acc=0.660 side_acc=0.642 r_n=127469
2026-05-11 17:38:35,412 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:38:35,412 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:38:35,412 INFO train_multi TF=ALL: new best val=2.0874 r_mae=0.8211 — saved
2026-05-11 17:38:35,416 INFO train_multi TF=ALL: new best r_mae=0.8211 — saved rmae checkpoint
2026-05-11 17:38:48,812 INFO train_multi TF=ALL epoch 57/100 train=1.9464 val=2.0896 r_mae=0.823 pos_r_acc=0.655 side_acc=0.638 r_n=127469
2026-05-11 17:39:02,287 INFO train_multi TF=ALL epoch 58/100 train=1.9347 val=2.1034 r_mae=0.821 pos_r_acc=0.654 side_acc=0.640 r_n=127469
2026-05-11 17:39:02,292 INFO train_multi TF=ALL: new best r_mae=0.8208 — saved rmae checkpoint
2026-05-11 17:39:15,557 INFO train_multi TF=ALL epoch 59/100 train=1.9281 val=2.0873 r_mae=0.824 pos_r_acc=0.655 side_acc=0.642 r_n=127469
2026-05-11 17:39:15,562 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:39:15,563 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:39:15,563 INFO train_multi TF=ALL: new best val=2.0873 r_mae=0.8243 — saved
2026-05-11 17:39:28,910 INFO train_multi TF=ALL epoch 60/100 train=1.9202 val=2.0967 r_mae=0.823 pos_r_acc=0.654 side_acc=0.636 r_n=127469
2026-05-11 17:39:42,495 INFO train_multi TF=ALL epoch 61/100 train=1.9124 val=2.0853 r_mae=0.818 pos_r_acc=0.661 side_acc=0.639 r_n=127469
2026-05-11 17:39:42,501 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:39:42,501 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:39:42,501 INFO train_multi TF=ALL: new best val=2.0853 r_mae=0.8184 — saved
2026-05-11 17:39:42,505 INFO train_multi TF=ALL: new best r_mae=0.8184 — saved rmae checkpoint
2026-05-11 17:39:56,156 INFO train_multi TF=ALL epoch 62/100 train=1.9027 val=2.0880 r_mae=0.819 pos_r_acc=0.657 side_acc=0.643 r_n=127469
2026-05-11 17:40:09,537 INFO train_multi TF=ALL epoch 63/100 train=1.8941 val=2.1179 r_mae=0.814 pos_r_acc=0.658 side_acc=0.634 r_n=127469
2026-05-11 17:40:09,542 INFO train_multi TF=ALL: new best r_mae=0.8142 — saved rmae checkpoint
2026-05-11 17:40:23,138 INFO train_multi TF=ALL epoch 64/100 train=1.8889 val=2.1229 r_mae=0.813 pos_r_acc=0.658 side_acc=0.636 r_n=127469
2026-05-11 17:40:23,143 INFO train_multi TF=ALL: new best r_mae=0.8128 — saved rmae checkpoint
2026-05-11 17:40:36,490 INFO train_multi TF=ALL epoch 65/100 train=1.8770 val=2.1015 r_mae=0.817 pos_r_acc=0.654 side_acc=0.641 r_n=127469
2026-05-11 17:40:50,040 INFO train_multi TF=ALL epoch 66/100 train=1.8699 val=2.0872 r_mae=0.818 pos_r_acc=0.657 side_acc=0.647 r_n=127469
2026-05-11 17:41:03,590 INFO train_multi TF=ALL epoch 67/100 train=1.8652 val=2.1076 r_mae=0.818 pos_r_acc=0.653 side_acc=0.641 r_n=127469
2026-05-11 17:41:17,117 INFO train_multi TF=ALL epoch 68/100 train=1.8582 val=2.0857 r_mae=0.816 pos_r_acc=0.658 side_acc=0.648 r_n=127469
2026-05-11 17:41:30,742 INFO train_multi TF=ALL epoch 69/100 train=1.8465 val=2.0963 r_mae=0.815 pos_r_acc=0.660 side_acc=0.642 r_n=127469
2026-05-11 17:41:44,322 INFO train_multi TF=ALL epoch 70/100 train=1.8406 val=2.0872 r_mae=0.816 pos_r_acc=0.656 side_acc=0.650 r_n=127469
2026-05-11 17:41:57,947 INFO train_multi TF=ALL epoch 71/100 train=1.8334 val=2.1180 r_mae=0.820 pos_r_acc=0.652 side_acc=0.642 r_n=127469
2026-05-11 17:42:11,225 INFO train_multi TF=ALL epoch 72/100 train=1.8252 val=2.0983 r_mae=0.810 pos_r_acc=0.659 side_acc=0.649 r_n=127469
2026-05-11 17:42:11,230 INFO train_multi TF=ALL: new best r_mae=0.8100 — saved rmae checkpoint
2026-05-11 17:42:24,719 INFO train_multi TF=ALL epoch 73/100 train=1.8179 val=2.1405 r_mae=0.819 pos_r_acc=0.651 side_acc=0.644 r_n=127469
2026-05-11 17:42:38,309 INFO train_multi TF=ALL epoch 74/100 train=1.8090 val=2.1172 r_mae=0.819 pos_r_acc=0.653 side_acc=0.646 r_n=127469
2026-05-11 17:42:51,743 INFO train_multi TF=ALL epoch 75/100 train=1.8050 val=2.1168 r_mae=0.816 pos_r_acc=0.654 side_acc=0.649 r_n=127469
2026-05-11 17:43:05,440 INFO train_multi TF=ALL epoch 76/100 train=1.7939 val=2.1254 r_mae=0.810 pos_r_acc=0.656 side_acc=0.650 r_n=127469
2026-05-11 17:43:18,904 INFO train_multi TF=ALL epoch 77/100 train=1.7875 val=2.1152 r_mae=0.811 pos_r_acc=0.658 side_acc=0.648 r_n=127469
2026-05-11 17:43:32,652 INFO train_multi TF=ALL epoch 78/100 train=1.7843 val=2.1287 r_mae=0.811 pos_r_acc=0.659 side_acc=0.643 r_n=127469
2026-05-11 17:43:46,182 INFO train_multi TF=ALL epoch 79/100 train=1.7789 val=2.1171 r_mae=0.812 pos_r_acc=0.656 side_acc=0.650 r_n=127469
2026-05-11 17:43:59,555 INFO train_multi TF=ALL epoch 80/100 train=1.7640 val=2.1198 r_mae=0.814 pos_r_acc=0.656 side_acc=0.645 r_n=127469
2026-05-11 17:44:13,075 INFO train_multi TF=ALL epoch 81/100 train=1.7564 val=2.1348 r_mae=0.812 pos_r_acc=0.655 side_acc=0.650 r_n=127469
2026-05-11 17:44:26,901 INFO train_multi TF=ALL epoch 82/100 train=1.7573 val=2.1201 r_mae=0.812 pos_r_acc=0.657 side_acc=0.650 r_n=127469
2026-05-11 17:44:40,395 INFO train_multi TF=ALL epoch 83/100 train=1.7534 val=2.1254 r_mae=0.812 pos_r_acc=0.657 side_acc=0.651 r_n=127469
2026-05-11 17:44:54,317 INFO train_multi TF=ALL epoch 84/100 train=1.7417 val=2.1422 r_mae=0.810 pos_r_acc=0.658 side_acc=0.643 r_n=127469
2026-05-11 17:44:54,326 INFO train_multi TF=ALL: new best r_mae=0.8099 — saved rmae checkpoint
2026-05-11 17:45:08,197 INFO train_multi TF=ALL epoch 85/100 train=1.7378 val=2.1329 r_mae=0.815 pos_r_acc=0.653 side_acc=0.655 r_n=127469
2026-05-11 17:45:21,997 INFO train_multi TF=ALL epoch 86/100 train=1.7291 val=2.1478 r_mae=0.820 pos_r_acc=0.651 side_acc=0.651 r_n=127469
2026-05-11 17:45:35,479 INFO train_multi TF=ALL epoch 87/100 train=1.7213 val=2.1448 r_mae=0.813 pos_r_acc=0.653 side_acc=0.657 r_n=127469
2026-05-11 17:45:49,217 INFO train_multi TF=ALL epoch 88/100 train=1.7176 val=2.1416 r_mae=0.814 pos_r_acc=0.654 side_acc=0.652 r_n=127469
2026-05-11 17:46:02,885 INFO train_multi TF=ALL epoch 89/100 train=1.7114 val=2.1492 r_mae=0.816 pos_r_acc=0.653 side_acc=0.650 r_n=127469
2026-05-11 17:46:16,652 INFO train_multi TF=ALL epoch 90/100 train=1.7041 val=2.1560 r_mae=0.821 pos_r_acc=0.649 side_acc=0.653 r_n=127469
2026-05-11 17:46:30,235 INFO train_multi TF=ALL epoch 91/100 train=1.6939 val=2.1547 r_mae=0.817 pos_r_acc=0.653 side_acc=0.654 r_n=127469
2026-05-11 17:46:44,119 INFO train_multi TF=ALL epoch 92/100 train=1.6920 val=2.1570 r_mae=0.819 pos_r_acc=0.652 side_acc=0.653 r_n=127469
2026-05-11 17:46:57,651 INFO train_multi TF=ALL epoch 93/100 train=1.6848 val=2.1422 r_mae=0.815 pos_r_acc=0.654 side_acc=0.659 r_n=127469
2026-05-11 17:47:11,302 INFO train_multi TF=ALL epoch 94/100 train=1.6808 val=2.1615 r_mae=0.817 pos_r_acc=0.653 side_acc=0.651 r_n=127469
2026-05-11 17:47:24,975 INFO train_multi TF=ALL epoch 95/100 train=1.6811 val=2.1487 r_mae=0.815 pos_r_acc=0.653 side_acc=0.653 r_n=127469
2026-05-11 17:47:38,821 INFO train_multi TF=ALL epoch 96/100 train=1.6684 val=2.1526 r_mae=0.817 pos_r_acc=0.653 side_acc=0.657 r_n=127469
2026-05-11 17:47:52,343 INFO train_multi TF=ALL epoch 97/100 train=1.6633 val=2.1691 r_mae=0.823 pos_r_acc=0.649 side_acc=0.655 r_n=127469
2026-05-11 17:48:06,117 INFO train_multi TF=ALL epoch 98/100 train=1.6599 val=2.1675 r_mae=0.818 pos_r_acc=0.653 side_acc=0.652 r_n=127469
2026-05-11 17:48:19,602 INFO train_multi TF=ALL epoch 99/100 train=1.6525 val=2.1602 r_mae=0.820 pos_r_acc=0.651 side_acc=0.662 r_n=127469
2026-05-11 17:48:33,137 INFO train_multi TF=ALL epoch 100/100 train=1.6468 val=2.1660 r_mae=0.817 pos_r_acc=0.653 side_acc=0.661 r_n=127469
2026-05-11 17:48:33,149 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 17:48:33,149 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:48:33,149 INFO train_multi TF=ALL: promoted r_mae checkpoint (0.8099 < primary 0.8184) — overwriting model.pt
2026-05-11 17:48:34,498 WARNING fit_r_isotonic: long calibrator increases MAE (cal=0.8306 >= raw=0.8119) — skipping
2026-05-11 17:48:34,508 WARNING fit_r_isotonic: short calibrator increases MAE (cal=0.8463 >= raw=0.8265) — skipping
2026-05-11 17:48:34,508 INFO fit_r_isotonic: saved side-R calibrators stats={'long': {'n': 31689, 'raw_mae': 0.8118784220800579, 'calibrated_mae': 0.8306095433351995, 'skipped': 'calibrator_hurts'}, 'short': {'n': 32408, 'raw_mae': 0.8265499169846091, 'calibrated_mae': 0.8462538787702982, 'skipped': 'calibrator_hurts'}}
2026-05-11 17:48:34,673 WARNING GRU validation R-MAE above floor for TF=ALL: best_val_r_mae=0.810 max=0.750. Keeping saved best weights so the pipeline can progress.
2026-05-11 17:48:34,688 INFO GRU R threshold XAUUSD/buy: q25_pos=0.535 q50_pos=1.667 pos_rate=46.0% (n=119259 n_pos=54899)
2026-05-11 17:48:34,694 INFO GRU R threshold XAUUSD/sell: q25_pos=0.528 q50_pos=1.667 pos_rate=45.5% (n=118087 n_pos=53734)
2026-05-11 17:48:34,701 INFO GRU R threshold EURUSD/buy: q25_pos=0.541 q50_pos=1.667 pos_rate=45.6% (n=118751 n_pos=54196)
2026-05-11 17:48:34,709 INFO GRU R threshold EURUSD/sell: q25_pos=0.545 q50_pos=1.667 pos_rate=45.6% (n=117679 n_pos=53721)
2026-05-11 17:48:34,715 INFO GRU R threshold USDJPY/buy: q25_pos=0.551 q50_pos=1.667 pos_rate=46.2% (n=118691 n_pos=54870)
2026-05-11 17:48:34,721 INFO GRU R threshold USDJPY/sell: q25_pos=0.542 q50_pos=1.667 pos_rate=44.6% (n=116030 n_pos=51793)
2026-05-11 17:48:34,728 INFO GRU R threshold EURJPY/buy: q25_pos=0.530 q50_pos=1.416 pos_rate=46.1% (n=118651 n_pos=54654)
2026-05-11 17:48:34,734 INFO GRU R threshold EURJPY/sell: q25_pos=0.525 q50_pos=1.667 pos_rate=44.7% (n=117320 n_pos=52399)
2026-05-11 17:48:34,741 INFO GRU R threshold GBPJPY/buy: q25_pos=0.517 q50_pos=1.443 pos_rate=45.9% (n=118277 n_pos=54278)
2026-05-11 17:48:34,747 INFO GRU R threshold GBPJPY/sell: q25_pos=0.521 q50_pos=1.667 pos_rate=45.3% (n=116299 n_pos=52702)
2026-05-11 17:48:34,753 INFO GRU R threshold GBPUSD/buy: q25_pos=0.524 q50_pos=1.667 pos_rate=45.8% (n=117764 n_pos=53894)
2026-05-11 17:48:34,759 INFO GRU R threshold GBPUSD/sell: q25_pos=0.533 q50_pos=1.667 pos_rate=45.8% (n=117404 n_pos=53785)
2026-05-11 17:48:34,760 INFO GRU per-symbol R thresholds saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/symbol_r_thresholds.json (6 symbols)
2026-05-11 17:48:34,827 INFO Retrain complete. Total wall-clock: 1393.3s
2026-05-11 17:48:39,564 INFO Model gru: SUCCESS
2026-05-11 17:48:39,564 INFO --- Training regime ---
2026-05-11 17:48:39,564 INFO Running retrain --model regime
2026-05-11 17:48:39,998 INFO retrain environment: KAGGLE
2026-05-11 17:48:41,710 INFO Device: CUDA (2 GPU(s))
2026-05-11 17:48:41,723 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 17:48:41,723 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 17:48:41,723 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 17:48:41,724 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 17:48:41,724 INFO Retrain data split: train
2026-05-11 17:48:41,724 INFO Retrain rolling fold selector: latest
2026-05-11 17:48:41,725 INFO === RegimeClassifier retrain (hierarchical: HTF 2-score bias + LTF 5-score behaviour) ===
2026-05-11 17:48:41,980 INFO NumExpr defaulting to 4 threads.
2026-05-11 17:48:42,207 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-11 17:48:42,207 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 17:48:42,207 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 17:48:42,208 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-11 17:48:42,268 INFO Regime rolling folds selected: [None]
2026-05-11 17:48:42,268 INFO === Regime rolling fold 1/1: train_all ===
2026-05-11 17:48:42,268 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-11 17:48:42,309 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 17:48:42,310 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
2026-05-11 17:48:42,328 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
2026-05-11 17:48:42,344 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
2026-05-11 17:48:42,360 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
2026-05-11 17:48:42,378 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
2026-05-11 17:48:42,394 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 17:48:42,646 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 17:48:42,717 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 17:48:42,745 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 17:48:42,746 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 17:48:42,757 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 17:48:42,758 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:48:43,578 INFO macro_cache: using training data dir /kaggle/working/Multi-Bot/trading-system/training_data
2026-05-11 17:48:43,703 INFO macro_cache loaded 19 series: {'asx200': '2842 bars 2015-01-02→2026-03-27', 'cac40': '2876 bars 2015-01-02→2026-03-27', 'dax': '2851 bars 2015-01-02→2026-03-27', 'djia': '2825 bars 2015-01-02→2026-03-27', 'dxy': '2826 bars 2015-01-02→2026-03-27', 'eurostoxx': '2823 bars 2015-01-05→2026-03-27', 'ftse': '2839 bars 2015-01-02→2026-03-27', 'gold_fut': '2824 bars 2015-01-02→2026-03-27', 'hsi': '2764 bars 2015-01-02→2026-03-27', 'nasdaq': '2825 bars 2015-01-02→2026-03-27', 'nikkei': '2744 bars 2015-01-05→2026-03-27', 'oil_fut': '2825 bars 2015-01-02→2026-03-27', 'spx': '2825 bars 2015-01-02→2026-03-27', 'us10y': '2824 bars 2015-01-02→2026-03-27', 'us30y': '2824 bars 2015-01-02→2026-03-27', 'us3m': '2824 bars 2015-01-02→2026-03-27', 'vix': '2825 bars 2015-01-02→2026-03-27', 'us10y_fred': '2607 bars 2016-03-28→2026-03-24', 'us2y_fred': '2607 bars 2016-03-28→2026-03-24'}
2026-05-11 17:48:43,963 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1142, 'BIAS_DOWN': 522, 'BIAS_NEUTRAL': 10438}  ambiguous=4182 (total=12102) horizon=84
2026-05-11 17:48:43,968 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected XAUUSD — 12052 samples (group=gold) score_means={'bias_up_score': 0.0948, 'bias_down_score': 0.0433} labels={'BIAS_UP': 1142, 'BIAS_DOWN': 522, 'BIAS_NEUTRAL': 10388} clean={'BIAS_UP': 1142, 'BIAS_DOWN': 522, 'BIAS_NEUTRAL': 6216}
2026-05-11 17:48:44,148 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:44,186 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:44,206 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:44,206 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:44,214 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:44,216 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:48:45,255 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 690, 'BIAS_DOWN': 540, 'BIAS_NEUTRAL': 10174}  ambiguous=3886 (total=11404) horizon=84
2026-05-11 17:48:45,260 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURUSD — 11354 samples (group=dollar) score_means={'bias_up_score': 0.0608, 'bias_down_score': 0.0476} labels={'BIAS_UP': 690, 'BIAS_DOWN': 540, 'BIAS_NEUTRAL': 10124} clean={'BIAS_UP': 690, 'BIAS_DOWN': 540, 'BIAS_NEUTRAL': 6257}
2026-05-11 17:48:45,429 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:45,467 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:45,488 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:45,489 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:45,497 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:45,499 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:48:46,512 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 826, 'BIAS_DOWN': 423, 'BIAS_NEUTRAL': 10154}  ambiguous=4036 (total=11403) horizon=84
2026-05-11 17:48:46,518 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected USDJPY — 11353 samples (group=dollar) score_means={'bias_up_score': 0.0728, 'bias_down_score': 0.0373} labels={'BIAS_UP': 826, 'BIAS_DOWN': 423, 'BIAS_NEUTRAL': 10104} clean={'BIAS_UP': 826, 'BIAS_DOWN': 423, 'BIAS_NEUTRAL': 6078}
2026-05-11 17:48:46,682 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:46,719 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:46,743 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:46,743 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:46,751 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:46,752 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:48:47,724 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 681, 'BIAS_DOWN': 527, 'BIAS_NEUTRAL': 10199}  ambiguous=4044 (total=11407) horizon=84
2026-05-11 17:48:47,729 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURJPY — 11357 samples (group=cross) score_means={'bias_up_score': 0.06, 'bias_down_score': 0.0464} labels={'BIAS_UP': 681, 'BIAS_DOWN': 527, 'BIAS_NEUTRAL': 10149} clean={'BIAS_UP': 681, 'BIAS_DOWN': 527, 'BIAS_NEUTRAL': 6111}
2026-05-11 17:48:47,890 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:47,927 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:47,949 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:47,949 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:47,959 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:47,960 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:48:48,932 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 839, 'BIAS_DOWN': 579, 'BIAS_NEUTRAL': 9990}  ambiguous=4240 (total=11408) horizon=84
2026-05-11 17:48:48,938 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPJPY — 11358 samples (group=cross) score_means={'bias_up_score': 0.0739, 'bias_down_score': 0.051} labels={'BIAS_UP': 839, 'BIAS_DOWN': 579, 'BIAS_NEUTRAL': 9940} clean={'BIAS_UP': 839, 'BIAS_DOWN': 579, 'BIAS_NEUTRAL': 5723}
2026-05-11 17:48:49,095 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:49,130 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:49,149 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:49,149 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:49,157 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:49,158 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:48:50,141 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 502, 'BIAS_DOWN': 720, 'BIAS_NEUTRAL': 10180}  ambiguous=4113 (total=11402) horizon=84
2026-05-11 17:48:50,147 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPUSD — 11352 samples (group=dollar) score_means={'bias_up_score': 0.0442, 'bias_down_score': 0.0623} labels={'BIAS_UP': 502, 'BIAS_DOWN': 707, 'BIAS_NEUTRAL': 10143} clean={'BIAS_UP': 502, 'BIAS_DOWN': 707, 'BIAS_NEUTRAL': 6056}
2026-05-11 17:48:50,215 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 1520, 'BIAS_DOWN': 1106, 'BIAS_NEUTRAL': 20089}, 'dollar': {'BIAS_UP': 2018, 'BIAS_DOWN': 1670, 'BIAS_NEUTRAL': 30371}, 'gold': {'BIAS_UP': 1142, 'BIAS_DOWN': 522, 'BIAS_NEUTRAL': 10388}}
2026-05-11 17:48:50,215 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0669, 'bias_down_score': 0.0487}, 'dollar': {'bias_up_score': 0.0593, 'bias_down_score': 0.049}, 'gold': {'bias_up_score': 0.0948, 'bias_down_score': 0.0433}}
2026-05-11 17:48:50,215 INFO Regime[4H mode=htf_bias] label distribution by year: {2016: {'BIAS_UP': 491, 'BIAS_DOWN': 576, 'BIAS_NEUTRAL': 7755}, 2017: {'BIAS_UP': 734, 'BIAS_DOWN': 286, 'BIAS_NEUTRAL': 8093}, 2018: {'BIAS_UP': 427, 'BIAS_DOWN': 714, 'BIAS_NEUTRAL': 7989}, 2019: {'BIAS_UP': 410, 'BIAS_DOWN': 447, 'BIAS_NEUTRAL': 8245}, 2020: {'BIAS_UP': 694, 'BIAS_DOWN': 174, 'BIAS_NEUTRAL': 8243}, 2021: {'BIAS_UP': 722, 'BIAS_DOWN': 473, 'BIAS_NEUTRAL': 7896}, 2022: {'BIAS_UP': 667, 'BIAS_DOWN': 519, 'BIAS_NEUTRAL': 7935}, 2023: {'BIAS_UP': 535, 'BIAS_DOWN': 109, 'BIAS_NEUTRAL': 4692}}
2026-05-11 17:48:50,216 INFO Regime[4H mode=htf_bias] score means by year: {2016: {'bias_up_score': 0.0557, 'bias_down_score': 0.0653}, 2017: {'bias_up_score': 0.0805, 'bias_down_score': 0.0314}, 2018: {'bias_up_score': 0.0468, 'bias_down_score': 0.0782}, 2019: {'bias_up_score': 0.045, 'bias_down_score': 0.0491}, 2020: {'bias_up_score': 0.0762, 'bias_down_score': 0.0191}, 2021: {'bias_up_score': 0.0794, 'bias_down_score': 0.052}, 2022: {'bias_up_score': 0.0731, 'bias_down_score': 0.0569}, 2023: {'bias_up_score': 0.1003, 'bias_down_score': 0.0204}}
2026-05-11 17:48:50,273 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
2026-05-11 17:48:50,274 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
2026-05-11 17:48:50,275 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
2026-05-11 17:48:50,276 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 17:48:50,277 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 17:48:50,277 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
2026-05-11 17:48:50,295 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 17:48:50,298 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 17:48:50,299 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 17:48:50,300 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 17:48:50,300 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 17:48:50,301 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:48:50,882 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 157, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1339}  ambiguous=566 (total=1581) horizon=84
2026-05-11 17:48:50,885 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected XAUUSD — 1531 samples (group=gold) score_means={'bias_up_score': 0.1025, 'bias_down_score': 0.0555} labels={'BIAS_UP': 157, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1289} clean={'BIAS_UP': 157, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 744}
2026-05-11 17:48:50,960 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:50,962 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:50,963 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:50,964 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:50,964 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:50,965 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:48:51,529 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 135, 'BIAS_DOWN': 66, 'BIAS_NEUTRAL': 1290}  ambiguous=531 (total=1491) horizon=84
2026-05-11 17:48:51,532 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURUSD — 1441 samples (group=dollar) score_means={'bias_up_score': 0.0937, 'bias_down_score': 0.0458} labels={'BIAS_UP': 135, 'BIAS_DOWN': 66, 'BIAS_NEUTRAL': 1240} clean={'BIAS_UP': 135, 'BIAS_DOWN': 66, 'BIAS_NEUTRAL': 739}
2026-05-11 17:48:51,605 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:51,608 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:51,608 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:51,609 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:51,609 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:51,610 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:48:52,174 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 164, 'BIAS_DOWN': 77, 'BIAS_NEUTRAL': 1248}  ambiguous=616 (total=1489) horizon=84
2026-05-11 17:48:52,177 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected USDJPY — 1439 samples (group=dollar) score_means={'bias_up_score': 0.114, 'bias_down_score': 0.0535} labels={'BIAS_UP': 164, 'BIAS_DOWN': 77, 'BIAS_NEUTRAL': 1198} clean={'BIAS_UP': 164, 'BIAS_DOWN': 77, 'BIAS_NEUTRAL': 608}
2026-05-11 17:48:52,249 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:52,252 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:52,253 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:52,253 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:52,253 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:52,254 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:48:52,822 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 123, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 1366}  ambiguous=582 (total=1494) horizon=84
2026-05-11 17:48:52,825 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0852, 'bias_down_score': 0.0035} labels={'BIAS_UP': 123, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 1316} clean={'BIAS_UP': 123, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 741}
2026-05-11 17:48:52,897 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:52,900 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:52,901 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:52,901 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:52,901 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:52,903 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:48:53,457 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 129, 'BIAS_DOWN': 9, 'BIAS_NEUTRAL': 1356}  ambiguous=551 (total=1494) horizon=84
2026-05-11 17:48:53,460 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0893, 'bias_down_score': 0.0055} labels={'BIAS_UP': 129, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 1307} clean={'BIAS_UP': 129, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 775}
2026-05-11 17:48:53,537 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:53,539 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:53,540 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:53,541 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:53,541 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:48:53,542 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:48:54,099 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 81, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 1316}  ambiguous=560 (total=1488) horizon=84
2026-05-11 17:48:54,102 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPUSD — 1438 samples (group=dollar) score_means={'bias_up_score': 0.0563, 'bias_down_score': 0.0633} labels={'BIAS_UP': 81, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 1266} clean={'BIAS_UP': 81, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 735}
2026-05-11 17:48:54,169 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 252, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 2623}, 'dollar': {'BIAS_UP': 380, 'BIAS_DOWN': 234, 'BIAS_NEUTRAL': 3704}, 'gold': {'BIAS_UP': 157, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1289}}
2026-05-11 17:48:54,169 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0873, 'bias_down_score': 0.0045}, 'dollar': {'bias_up_score': 0.088, 'bias_down_score': 0.0542}, 'gold': {'bias_up_score': 0.1025, 'bias_down_score': 0.0555}}
2026-05-11 17:48:54,169 INFO Regime[4H mode=htf_bias] label distribution by year: {2022: {'BIAS_UP': 258, 'BIAS_DOWN': 228, 'BIAS_NEUTRAL': 2915}, 2023: {'BIAS_UP': 531, 'BIAS_DOWN': 104, 'BIAS_NEUTRAL': 4701}}
2026-05-11 17:48:54,169 INFO Regime[4H mode=htf_bias] score means by year: {2022: {'bias_up_score': 0.0759, 'bias_down_score': 0.067}, 2023: {'bias_up_score': 0.0995, 'bias_down_score': 0.0195}}
2026-05-11 17:48:54,224 INFO Regime phase HTF dataset build fold=train_all: 12.0s (train=68826 val=8737)
2026-05-11 17:48:54,225 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260511_174854
2026-05-11 17:48:54,440 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=51, n_classes=2)
2026-05-11 17:48:54,440 INFO Regime 4H/htf_bias warm start enabled from existing weights
2026-05-11 17:48:54,455 INFO RegimeClassifier[mode=htf_bias]: HTF clean-label fit filter kept train=44419/68826 val=5463/8737 at conf>=0.40 train_counts={'BIAS_UP': 4680, 'BIAS_DOWN': 3298, 'BIAS_NEUTRAL': 36441} val_counts={'BIAS_UP': 789, 'BIAS_DOWN': 332, 'BIAS_NEUTRAL': 4342}
2026-05-11 17:48:54,455 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=44419 val=5463 train_labels={'BIAS_UP': 4680, 'BIAS_DOWN': 3298, 'BIAS_NEUTRAL': 36441} val_labels={'BIAS_UP': 789, 'BIAS_DOWN': 332, 'BIAS_NEUTRAL': 4342}
2026-05-11 17:48:54,457 INFO RegimeClassifier[mode=htf_bias]: warm start HTF score head
2026-05-11 17:48:54,457 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-11 17:48:54,457 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 8.491, 'bias_down_score': 12.0}
2026-05-11 17:48:54,461 INFO RegimeClassifier[mode=htf_bias]: HTF balanced sampler — dir=7978 neutral=36441 dir_weight=3 => dir_frac_per_epoch≈47.2%
2026-05-11 17:48:58,129 INFO Regime HTF score epoch  1/50 — tr=2.6762 va=1.0323 acc=0.782 bal=0.665 threshold=0.40 margin=0.15 recall={'BIAS_UP': 0.497, 'BIAS_DOWN': 0.654, 'BIAS_NEUTRAL': 0.843} precision={'BIAS_UP': 0.543, 'BIAS_DOWN': 0.382, 'BIAS_NEUTRAL': 0.877}
2026-05-11 17:48:59,463 INFO Regime HTF score epoch  2/50 — tr=2.6642 va=1.0326 bal=0.663
2026-05-11 17:49:00,892 INFO Regime HTF score epoch  3/50 — tr=2.6495 va=1.0173 bal=0.680
2026-05-11 17:49:02,305 INFO Regime HTF score epoch  4/50 — tr=2.6681 va=0.9923 bal=0.641
2026-05-11 17:49:03,668 INFO Regime HTF score epoch  5/50 — tr=2.5753 va=0.9662 acc=0.784 bal=0.656 threshold=0.45 margin=0.15 recall={'BIAS_UP': 0.482, 'BIAS_DOWN': 0.636, 'BIAS_NEUTRAL': 0.85} precision={'BIAS_UP': 0.548, 'BIAS_DOWN': 0.384, 'BIAS_NEUTRAL': 0.874}
2026-05-11 17:49:05,131 INFO Regime HTF score epoch  6/50 — tr=2.5406 va=0.9373 bal=0.653
2026-05-11 17:49:06,499 INFO Regime HTF score epoch  7/50 — tr=2.4476 va=0.9058 bal=0.647
2026-05-11 17:49:07,965 INFO Regime HTF score epoch  8/50 — tr=2.3504 va=0.8834 bal=0.664
2026-05-11 17:49:09,433 INFO Regime HTF score epoch  9/50 — tr=2.2703 va=0.8513 bal=0.561
2026-05-11 17:49:10,826 INFO Regime HTF score epoch 10/50 — tr=2.1958 va=0.8266 acc=0.785 bal=0.645 threshold=0.55 margin=0.15 recall={'BIAS_UP': 0.504, 'BIAS_DOWN': 0.578, 'BIAS_NEUTRAL': 0.852} precision={'BIAS_UP': 0.549, 'BIAS_DOWN': 0.378, 'BIAS_NEUTRAL': 0.874}
2026-05-11 17:49:12,295 INFO Regime HTF score epoch 11/50 — tr=2.0937 va=0.8028 bal=0.622
2026-05-11 17:49:12,296 INFO Regime HTF score early stop at epoch 11
2026-05-11 17:49:13,666 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.400 margin=0.150 precision={'BIAS_UP': 0.539, 'BIAS_DOWN': 0.381, 'BIAS_NEUTRAL': 0.881} recall={'BIAS_UP': 0.512, 'BIAS_DOWN': 0.693, 'BIAS_NEUTRAL': 0.834} f1={'BIAS_UP': 0.525, 'BIAS_DOWN': 0.491, 'BIAS_NEUTRAL': 0.857} confusion=[[404, 0, 385], [0, 230, 102], [346, 374, 3622]] score_mae={'bias_up_score': 0.2045, 'bias_down_score': 0.1345} pred_share={'BIAS_UP': 0.1373, 'BIAS_DOWN': 0.1106, 'BIAS_NEUTRAL': 0.7522}
2026-05-11 17:49:13,667 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.539, 'BIAS_DOWN': 0.381, 'BIAS_NEUTRAL': 0.881} min_precision=0.500 recall={'BIAS_UP': 0.512, 'BIAS_DOWN': 0.693, 'BIAS_NEUTRAL': 0.834} min_recall=0.150 f1={'BIAS_UP': 0.525, 'BIAS_DOWN': 0.491, 'BIAS_NEUTRAL': 0.857} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_DOWN'] weak_recall=[] weak_f1=[] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-11 17:49:13,671 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 17:49:13,671 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 17:49:13,672 INFO Regime phase HTF train fold=train_all: 19.2s
2026-05-11 17:49:13,781 INFO Regime HTF complete fold=train_all: acc=0.779 bal=0.680 train=68826 val=8737 per_class={'BIAS_UP': 0.512, 'BIAS_DOWN': 0.693, 'BIAS_NEUTRAL': 0.834} precision={'BIAS_UP': 0.539, 'BIAS_DOWN': 0.381, 'BIAS_NEUTRAL': 0.881} threshold=0.400 margin=0.150
2026-05-11 17:49:13,782 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 17:49:13,982 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 502, 'BIAS_DOWN': 720, 'BIAS_NEUTRAL': 10180}  ambiguous=4113 (total=11402) horizon=84
2026-05-11 17:49:13,988 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on GBPUSD 4H:
{'BIAS_UP': 4.482142857142857, 'BIAS_DOWN': 5.669291338582677, 'BIAS_NEUTRAL': 42.416666666666664}
2026-05-11 17:49:13,992 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (all labels):
{'BIAS_UP': {'n': 502, 'mean': 0.00043281055222645396, 'mean_over_std': 0.19321420748230153}, 'BIAS_DOWN': {'n': 720, 'mean': -0.00046308823082979826, 'mean_over_std': -0.19538165843530478}, 'BIAS_NEUTRAL': {'n': 10179, 'mean': 7.477567618138561e-07, 'mean_over_std': 0.0002829536380249001}}
2026-05-11 17:49:13,992 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 502, 'mean': 0.00043281055222645396, 'mean_over_std': 0.19321420748230153}, 'BIAS_DOWN': {'n': 720, 'mean': -0.00046308823082979826, 'mean_over_std': -0.19538165843530478}, 'BIAS_NEUTRAL': {'n': 6067, 'mean': 9.596616495197703e-06, 'mean_over_std': 0.004013656697571348}}
2026-05-11 17:49:14,001 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-11 17:49:14,004 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
2026-05-11 17:49:14,006 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
2026-05-11 17:49:14,008 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
2026-05-11 17:49:14,010 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
2026-05-11 17:49:14,012 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
2026-05-11 17:49:14,014 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 17:49:14,030 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 17:49:14,038 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 17:49:14,041 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 17:49:14,042 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 17:49:14,042 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 17:49:14,047 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:49:15,208 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected XAUUSD — 44227 samples (group=gold) score_means={'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}
2026-05-11 17:49:15,322 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:15,324 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:15,325 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:15,325 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:15,326 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:15,328 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:49:16,362 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURUSD — 43689 samples (group=dollar) score_means={'trend_score': 0.49, 'range_score': 0.236, 'chop_score': 0.4622, 'volatility_percentile': 0.3812, 'consolidation_score': 0.1862}
2026-05-11 17:49:16,474 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:16,476 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:16,477 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:16,478 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:16,478 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:16,481 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:49:17,507 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected USDJPY — 43686 samples (group=dollar) score_means={'trend_score': 0.4949, 'range_score': 0.2312, 'chop_score': 0.4613, 'volatility_percentile': 0.38, 'consolidation_score': 0.1931}
2026-05-11 17:49:17,620 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:17,622 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:17,623 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:17,624 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:17,624 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:17,626 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:49:18,665 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURJPY — 43688 samples (group=cross) score_means={'trend_score': 0.4888, 'range_score': 0.2345, 'chop_score': 0.4662, 'volatility_percentile': 0.3831, 'consolidation_score': 0.1882}
2026-05-11 17:49:18,784 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:18,786 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:18,787 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:18,787 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:18,788 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:18,790 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:49:19,849 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPJPY — 43678 samples (group=cross) score_means={'trend_score': 0.4878, 'range_score': 0.2348, 'chop_score': 0.4673, 'volatility_percentile': 0.3839, 'consolidation_score': 0.1904}
2026-05-11 17:49:19,960 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:19,963 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:19,963 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:19,964 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:19,964 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:19,967 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:49:21,008 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPUSD — 43676 samples (group=dollar) score_means={'trend_score': 0.4923, 'range_score': 0.2335, 'chop_score': 0.4611, 'volatility_percentile': 0.3832, 'consolidation_score': 0.1856}
2026-05-11 17:49:21,135 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2346, 'chop_score': 0.4667, 'volatility_percentile': 0.3835, 'consolidation_score': 0.1893}, 'dollar': {'trend_score': 0.4924, 'range_score': 0.2335, 'chop_score': 0.4615, 'volatility_percentile': 0.3815, 'consolidation_score': 0.1883}, 'gold': {'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}}
2026-05-11 17:49:21,135 INFO Regime[1H mode=ltf_behaviour] score means by year: {2016: {'trend_score': 0.495, 'range_score': 0.233, 'chop_score': 0.4607, 'volatility_percentile': 0.3776, 'consolidation_score': 0.192}, 2017: {'trend_score': 0.5017, 'range_score': 0.2328, 'chop_score': 0.4574, 'volatility_percentile': 0.3689, 'consolidation_score': 0.1942}, 2018: {'trend_score': 0.4886, 'range_score': 0.237, 'chop_score': 0.4655, 'volatility_percentile': 0.3947, 'consolidation_score': 0.1785}, 2019: {'trend_score': 0.4817, 'range_score': 0.236, 'chop_score': 0.4691, 'volatility_percentile': 0.3725, 'consolidation_score': 0.198}, 2020: {'trend_score': 0.4874, 'range_score': 0.2311, 'chop_score': 0.4674, 'volatility_percentile': 0.3848, 'consolidation_score': 0.1919}, 2021: {'trend_score': 0.4872, 'range_score': 0.2372, 'chop_score': 0.4672, 'volatility_percentile': 0.377, 'consolidation_score': 0.192}, 2022: {'trend_score': 0.4893, 'range_score': 0.2357, 'chop_score': 0.4647, 'volatility_percentile': 0.3962, 'consolidation_score': 0.1784}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 17:49:21,245 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
2026-05-11 17:49:21,247 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 17:49:21,248 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
2026-05-11 17:49:21,249 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 17:49:21,251 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
2026-05-11 17:49:21,252 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
2026-05-11 17:49:21,261 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 17:49:21,265 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 17:49:21,266 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 17:49:21,266 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 17:49:21,267 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 17:49:21,269 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:49:21,612 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected XAUUSD — 5117 samples (group=gold) score_means={'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}
2026-05-11 17:49:21,723 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:21,726 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:21,726 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:21,727 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:21,727 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:21,729 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:49:22,059 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURUSD — 5053 samples (group=dollar) score_means={'trend_score': 0.4883, 'range_score': 0.2413, 'chop_score': 0.463, 'volatility_percentile': 0.3837, 'consolidation_score': 0.1793}
2026-05-11 17:49:22,175 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:22,177 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:22,178 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:22,178 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:22,179 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:22,180 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:49:22,510 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected USDJPY — 5046 samples (group=dollar) score_means={'trend_score': 0.5085, 'range_score': 0.2259, 'chop_score': 0.4506, 'volatility_percentile': 0.3809, 'consolidation_score': 0.1888}
2026-05-11 17:49:22,622 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:22,624 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:22,625 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:22,626 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:22,626 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:22,628 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:49:22,958 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURJPY — 5053 samples (group=cross) score_means={'trend_score': 0.5023, 'range_score': 0.2261, 'chop_score': 0.4524, 'volatility_percentile': 0.3837, 'consolidation_score': 0.18}
2026-05-11 17:49:23,070 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:23,072 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:23,073 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:23,073 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:23,074 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:23,075 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:49:23,399 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPJPY — 5043 samples (group=cross) score_means={'trend_score': 0.4743, 'range_score': 0.2372, 'chop_score': 0.4716, 'volatility_percentile': 0.3777, 'consolidation_score': 0.1917}
2026-05-11 17:49:23,511 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:23,513 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:23,514 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:23,514 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:23,515 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 17:49:23,516 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 17:49:23,843 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPUSD — 5040 samples (group=dollar) score_means={'trend_score': 0.4843, 'range_score': 0.2371, 'chop_score': 0.4614, 'volatility_percentile': 0.377, 'consolidation_score': 0.1835}
2026-05-11 17:49:23,953 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2317, 'chop_score': 0.462, 'volatility_percentile': 0.3807, 'consolidation_score': 0.1858}, 'dollar': {'trend_score': 0.4937, 'range_score': 0.2348, 'chop_score': 0.4583, 'volatility_percentile': 0.3806, 'consolidation_score': 0.1839}, 'gold': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}}
2026-05-11 17:49:23,954 INFO Regime[1H mode=ltf_behaviour] score means by year: {2022: {'trend_score': 0.4905, 'range_score': 0.2364, 'chop_score': 0.4625, 'volatility_percentile': 0.385, 'consolidation_score': 0.1797}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 17:49:24,046 INFO Regime phase LTF dataset build fold=train_all: 10.0s (train=262644 val=30352)
2026-05-11 17:49:24,047 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260511_174924
2026-05-11 17:49:24,052 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=53, n_classes=5)
2026-05-11 17:49:24,052 INFO Regime 1H/ltf_behaviour warm start enabled from existing weights
2026-05-11 17:49:24,086 INFO RegimeClassifier[mode=ltf_behaviour]: warm start score head
2026-05-11 17:49:24,086 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-11 17:49:24,671 INFO Regime score epoch  1/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.0175, 'range_score': 0.0318, 'chop_score': 0.0191, 'volatility_percentile': 0.0153, 'consolidation_score': 0.0199}
2026-05-11 17:49:25,188 INFO Regime score epoch  2/50 — tr=0.0034 va=0.0008
2026-05-11 17:49:25,708 INFO Regime score epoch  3/50 — tr=0.0034 va=0.0008
2026-05-11 17:49:26,222 INFO Regime score epoch  4/50 — tr=0.0034 va=0.0008
2026-05-11 17:49:26,728 INFO Regime score epoch  5/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.0172, 'range_score': 0.0316, 'chop_score': 0.019, 'volatility_percentile': 0.0146, 'consolidation_score': 0.0195}
2026-05-11 17:49:27,254 INFO Regime score epoch  6/50 — tr=0.0034 va=0.0008
2026-05-11 17:49:27,771 INFO Regime score epoch  7/50 — tr=0.0034 va=0.0008
2026-05-11 17:49:28,291 INFO Regime score epoch  8/50 — tr=0.0034 va=0.0008
2026-05-11 17:49:28,813 INFO Regime score epoch  9/50 — tr=0.0034 va=0.0008
2026-05-11 17:49:29,321 INFO Regime score epoch 10/50 — tr=0.0033 va=0.0008 mae={'trend_score': 0.0172, 'range_score': 0.0314, 'chop_score': 0.0189, 'volatility_percentile': 0.0145, 'consolidation_score': 0.0197}
2026-05-11 17:49:29,828 INFO Regime score epoch 11/50 — tr=0.0033 va=0.0008
2026-05-11 17:49:30,354 INFO Regime score epoch 12/50 — tr=0.0033 va=0.0008
2026-05-11 17:49:30,881 INFO Regime score epoch 13/50 — tr=0.0033 va=0.0008
2026-05-11 17:49:31,402 INFO Regime score epoch 14/50 — tr=0.0033 va=0.0008
2026-05-11 17:49:31,936 INFO Regime score epoch 15/50 — tr=0.0033 va=0.0008 mae={'trend_score': 0.0167, 'range_score': 0.0312, 'chop_score': 0.0184, 'volatility_percentile': 0.0143, 'consolidation_score': 0.0191}
2026-05-11 17:49:32,454 INFO Regime score epoch 16/50 — tr=0.0033 va=0.0008
2026-05-11 17:49:32,965 INFO Regime score epoch 17/50 — tr=0.0033 va=0.0008
2026-05-11 17:49:33,491 INFO Regime score epoch 18/50 — tr=0.0033 va=0.0008
2026-05-11 17:49:34,029 INFO Regime score epoch 19/50 — tr=0.0033 va=0.0008
2026-05-11 17:49:34,599 INFO Regime score epoch 20/50 — tr=0.0032 va=0.0008 mae={'trend_score': 0.0165, 'range_score': 0.0309, 'chop_score': 0.0185, 'volatility_percentile': 0.0139, 'consolidation_score': 0.0188}
2026-05-11 17:49:35,139 INFO Regime score epoch 21/50 — tr=0.0032 va=0.0008
2026-05-11 17:49:35,651 INFO Regime score epoch 22/50 — tr=0.0032 va=0.0008
2026-05-11 17:49:36,154 INFO Regime score epoch 23/50 — tr=0.0032 va=0.0008
2026-05-11 17:49:36,656 INFO Regime score epoch 24/50 — tr=0.0032 va=0.0008
2026-05-11 17:49:37,180 INFO Regime score epoch 25/50 — tr=0.0032 va=0.0008 mae={'trend_score': 0.016, 'range_score': 0.0307, 'chop_score': 0.0185, 'volatility_percentile': 0.0135, 'consolidation_score': 0.0189}
2026-05-11 17:49:37,693 INFO Regime score epoch 26/50 — tr=0.0032 va=0.0007
2026-05-11 17:49:38,206 INFO Regime score epoch 27/50 — tr=0.0032 va=0.0007
2026-05-11 17:49:38,730 INFO Regime score epoch 28/50 — tr=0.0032 va=0.0008
2026-05-11 17:49:39,270 INFO Regime score epoch 29/50 — tr=0.0032 va=0.0007
2026-05-11 17:49:39,790 INFO Regime score epoch 30/50 — tr=0.0032 va=0.0007 mae={'trend_score': 0.0166, 'range_score': 0.0304, 'chop_score': 0.018, 'volatility_percentile': 0.0138, 'consolidation_score': 0.0188}
2026-05-11 17:49:40,315 INFO Regime score epoch 31/50 — tr=0.0032 va=0.0007
2026-05-11 17:49:40,841 INFO Regime score epoch 32/50 — tr=0.0032 va=0.0007
2026-05-11 17:49:41,366 INFO Regime score epoch 33/50 — tr=0.0032 va=0.0007
2026-05-11 17:49:41,893 INFO Regime score epoch 34/50 — tr=0.0032 va=0.0007
2026-05-11 17:49:42,410 INFO Regime score epoch 35/50 — tr=0.0032 va=0.0007 mae={'trend_score': 0.016, 'range_score': 0.0302, 'chop_score': 0.0178, 'volatility_percentile': 0.0135, 'consolidation_score': 0.0187}
2026-05-11 17:49:42,931 INFO Regime score epoch 36/50 — tr=0.0032 va=0.0007
2026-05-11 17:49:43,462 INFO Regime score epoch 37/50 — tr=0.0032 va=0.0007
2026-05-11 17:49:43,984 INFO Regime score epoch 38/50 — tr=0.0032 va=0.0007
2026-05-11 17:49:44,502 INFO Regime score epoch 39/50 — tr=0.0031 va=0.0007
2026-05-11 17:49:45,049 INFO Regime score epoch 40/50 — tr=0.0032 va=0.0007 mae={'trend_score': 0.0162, 'range_score': 0.0301, 'chop_score': 0.0177, 'volatility_percentile': 0.0129, 'consolidation_score': 0.0186}
2026-05-11 17:49:45,562 INFO Regime score epoch 41/50 — tr=0.0031 va=0.0007
2026-05-11 17:49:46,085 INFO Regime score epoch 42/50 — tr=0.0031 va=0.0007
2026-05-11 17:49:46,596 INFO Regime score epoch 43/50 — tr=0.0031 va=0.0007
2026-05-11 17:49:47,124 INFO Regime score epoch 44/50 — tr=0.0031 va=0.0007
2026-05-11 17:49:47,640 INFO Regime score epoch 45/50 — tr=0.0031 va=0.0007 mae={'trend_score': 0.0158, 'range_score': 0.0303, 'chop_score': 0.0178, 'volatility_percentile': 0.0133, 'consolidation_score': 0.0186}
2026-05-11 17:49:48,163 INFO Regime score epoch 46/50 — tr=0.0031 va=0.0007
2026-05-11 17:49:48,686 INFO Regime score epoch 47/50 — tr=0.0031 va=0.0007
2026-05-11 17:49:49,204 INFO Regime score epoch 48/50 — tr=0.0031 va=0.0007
2026-05-11 17:49:49,734 INFO Regime score epoch 49/50 — tr=0.0031 va=0.0007
2026-05-11 17:49:50,252 INFO Regime score epoch 50/50 — tr=0.0031 va=0.0007 mae={'trend_score': 0.0162, 'range_score': 0.0301, 'chop_score': 0.0178, 'volatility_percentile': 0.0131, 'consolidation_score': 0.0186}
2026-05-11 17:49:50,274 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.016, 'range_score': 0.0302, 'chop_score': 0.0177, 'volatility_percentile': 0.0133, 'consolidation_score': 0.0188} mse={'trend_score': 0.00045, 'range_score': 0.00152, 'chop_score': 0.00051, 'volatility_percentile': 0.00034, 'consolidation_score': 0.0008} corr={'trend_score': 0.9955, 'range_score': 0.963, 'chop_score': 0.9931, 'volatility_percentile': 0.9965, 'consolidation_score': 0.9916} pred_std={'trend_score': 0.2226, 'range_score': 0.1324, 'chop_score': 0.1831, 'volatility_percentile': 0.2182, 'consolidation_score': 0.2144} target_std={'trend_score': 0.2219, 'range_score': 0.1433, 'chop_score': 0.1888, 'volatility_percentile': 0.2165, 'consolidation_score': 0.2179}
2026-05-11 17:49:50,603 INFO RegimeClassifier[mode=ltf_behaviour] train group diagnostics={'dollar': {'n': 16632, 'score_mae': {'trend_score': 0.0156, 'range_score': 0.0301, 'chop_score': 0.0176, 'volatility_percentile': 0.013, 'consolidation_score': 0.0191}, 'target_score_mean': {'trend_score': 0.491, 'range_score': 0.2347, 'chop_score': 0.4613, 'volatility_percentile': 0.3824, 'consolidation_score': 0.186}, 'pred_score_mean': {'trend_score': 0.4914, 'range_score': 0.2339, 'chop_score': 0.4609, 'volatility_percentile': 0.3794, 'consolidation_score': 0.1842}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3580, 51, 0, 4, 0, 0, 144], [4, 98, 0, 0, 0, 3, 5], [0, 0, 187, 10, 58, 0, 205], [2, 0, 3, 562, 34, 0, 88], [0, 0, 21, 16, 3118, 0, 161], [0, 18, 0, 0, 8, 60, 42], [135, 13, 58, 44, 69, 5, 7826]]}, 'gold': {'n': 8420, 'score_mae': {'trend_score': 0.0155, 'range_score': 0.0307, 'chop_score': 0.0178, 'volatility_percentile': 0.0135, 'consolidation_score': 0.0195}, 'target_score_mean': {'trend_score': 0.4883, 'range_score': 0.2355, 'chop_score': 0.4652, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}, 'pred_score_mean': {'trend_score': 0.4892, 'range_score': 0.2347, 'chop_score': 0.4645, 'volatility_percentile': 0.3735, 'consolidation_score': 0.1899}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1789, 28, 0, 0, 0, 0, 68], [4, 51, 0, 0, 0, 0, 1], [0, 0, 93, 10, 27, 0, 114], [1, 0, 2, 349, 20, 0, 44], [0, 0, 13, 18, 1598, 0, 75], [0, 14, 0, 0, 4, 43, 20], [65, 3, 42, 14, 46, 0, 3864]]}, 'jpy': {'n': 24948, 'score_mae': {'trend_score': 0.0155, 'range_score': 0.0301, 'chop_score': 0.0174, 'volatility_percentile': 0.0139, 'consolidation_score': 0.0191}, 'target_score_mean': {'trend_score': 0.4904, 'range_score': 0.2338, 'chop_score': 0.4653, 'volatility_percentile': 0.3826, 'consolidation_score': 0.1905}, 'pred_score_mean': {'trend_score': 0.4905, 'range_score': 0.233, 'chop_score': 0.4644, 'volatility_percentile': 0.379, 'consolidation_score': 0.1879}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[5353, 101, 0, 3, 0, 0, 189], [8, 169, 0, 0, 0, 2, 8], [0, 0, 236, 18, 91, 0, 302], [3, 0, 3, 1085, 73, 0, 150], [0, 0, 29, 43, 4787, 0, 256], [0, 30, 0, 0, 15, 94, 84], [188, 13, 86, 76, 127, 5, 11321]]}}
2026-05-11 17:49:50,790 INFO RegimeClassifier[mode=ltf_behaviour] validation group diagnostics={'dollar': {'n': 10093, 'score_mae': {'trend_score': 0.0162, 'range_score': 0.0306, 'chop_score': 0.0178, 'volatility_percentile': 0.0128, 'consolidation_score': 0.0183}, 'target_score_mean': {'trend_score': 0.4863, 'range_score': 0.2392, 'chop_score': 0.4622, 'volatility_percentile': 0.3804, 'consolidation_score': 0.1814}, 'pred_score_mean': {'trend_score': 0.487, 'range_score': 0.2365, 'chop_score': 0.4628, 'volatility_percentile': 0.3773, 'consolidation_score': 0.1801}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[2299, 22, 0, 1, 0, 0, 90], [4, 46, 0, 0, 0, 3, 0], [0, 0, 103, 7, 50, 0, 156], [0, 0, 1, 342, 26, 0, 54], [0, 0, 16, 13, 1936, 0, 85], [0, 11, 0, 0, 3, 36, 27], [58, 6, 29, 37, 45, 3, 4584]]}, 'gold': {'n': 5117, 'score_mae': {'trend_score': 0.0152, 'range_score': 0.0294, 'chop_score': 0.0177, 'volatility_percentile': 0.0131, 'consolidation_score': 0.0194}, 'target_score_mean': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}, 'pred_score_mean': {'trend_score': 0.4974, 'range_score': 0.2309, 'chop_score': 0.4567, 'volatility_percentile': 0.3781, 'consolidation_score': 0.1814}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1115, 11, 0, 0, 0, 0, 41], [3, 30, 0, 0, 0, 1, 1], [0, 0, 62, 3, 16, 0, 90], [0, 0, 2, 223, 8, 0, 22], [0, 0, 6, 9, 824, 0, 48], [0, 6, 0, 0, 4, 23, 17], [51, 2, 27, 19, 25, 1, 2427]]}, 'jpy': {'n': 15142, 'score_mae': {'trend_score': 0.0161, 'range_score': 0.0302, 'chop_score': 0.0175, 'volatility_percentile': 0.0138, 'consolidation_score': 0.0189}, 'target_score_mean': {'trend_score': 0.495, 'range_score': 0.2297, 'chop_score': 0.4582, 'volatility_percentile': 0.3808, 'consolidation_score': 0.1868}, 'pred_score_mean': {'trend_score': 0.4946, 'range_score': 0.228, 'chop_score': 0.4584, 'volatility_percentile': 0.3774, 'consolidation_score': 0.1844}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3351, 41, 0, 1, 0, 0, 125], [4, 103, 0, 0, 0, 4, 4], [0, 0, 141, 13, 51, 0, 179], [3, 0, 1, 698, 41, 0, 84], [0, 0, 22, 27, 2613, 0, 155], [0, 17, 0, 0, 9, 56, 40], [105, 11, 53, 35, 81, 3, 7071]]}}
2026-05-11 17:49:50,797 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-11 17:49:50,797 INFO Regime phase LTF train fold=train_all: 26.7s
2026-05-11 17:49:50,906 INFO Regime LTF complete fold=train_all: score_accuracy=0.981, train=262644 val=30352 mae={'trend_score': 0.016, 'range_score': 0.0302, 'chop_score': 0.0177, 'volatility_percentile': 0.0133, 'consolidation_score': 0.0188}
2026-05-11 17:49:50,909 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 17:49:51,288 INFO Regime[1H mode=ltf_behaviour fold=latest] LTF score diagnostics on GBPUSD:
{'trend_score': {'mean': 0.4921, 'q10': 0.1918, 'q50': 0.4877, 'q90': 0.8002}, 'range_score': {'mean': 0.2337, 'q10': 0.0513, 'q50': 0.2125, 'q90': 0.4349}, 'chop_score': {'mean': 0.4612, 'q10': 0.2148, 'q50': 0.4503, 'q90': 0.7255}, 'volatility_percentile': {'mean': 0.3832, 'q10': 0.1018, 'q50': 0.3704, 'q90': 0.6823}, 'consolidation_score': {'mean': 0.1854, 'q10': 0.0, 'q50': 0.1185, 'q90': 0.506}}
2026-05-11 17:49:51,296 INFO Regime retrain total: 69.6s (370559 train+val samples)
2026-05-11 17:49:51,304 INFO Retrain complete. Total wall-clock: 69.6s
2026-05-11 17:49:52,330 INFO Model regime: SUCCESS
2026-05-11 17:49:52,331 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 17:49:52,331 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 17:49:52,331 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-11 17:49:52,331 INFO   [OK] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-05-11 17:49:52,331 INFO   [DORMANT] rl_ppo — RL_ENABLED=false, skipping
2026-05-11 17:49:52,331 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-05-11 17:49:52,340 INFO Saved 23 retrain records to metrics/

=== TRAINING COMPLETE ===
  gru: SUCCESS
  regime: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Clean Quality/RL source: Backtest on train window ===
  START Train-window backtest for Quality/RL labels
2026-05-11 17:49:52,997 INFO === STEP 6: BACKTEST (train) ===
2026-05-11 17:49:52,998 INFO BT_WINDOW=train — train-window backtest: 2016-01-04 → 2023-08-04 (clean Quality/RL labels)
2026-05-11 17:49:52,999 INFO Cleared existing journal for fresh train run
2026-05-11 17:49:52,999 INFO ================================================================
  ROUND 0 / 3
================================================================
2026-05-11 17:49:52,999 INFO Round 0 — running backtest: 2016-01-04 → 2023-08-04 (ml_trader, shared ML cache)
2026-05-11 17:54:07,233 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for EURJPY with 2
2026-05-11 17:54:07,248 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for EURJPY with 0.3333333333333333
2026-05-11 17:54:07,406 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for USDJPY with 2
2026-05-11 17:54:07,435 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for USDJPY with 0.3333333333333333
2026-05-11 17:54:07,599 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for EURUSD with 2
2026-05-11 17:54:07,620 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for EURUSD with 0.3333333333333333
2026-05-11 17:54:07,718 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for EURJPY with 2
2026-05-11 17:54:07,744 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for EURJPY with 0.25
2026-05-11 17:54:07,782 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for EURJPY: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 17:54:08,024 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for USDJPY with 2
2026-05-11 17:54:08,046 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for USDJPY with 0.25
2026-05-11 17:54:08,091 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for USDJPY: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 17:54:08,371 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for EURUSD with 2
2026-05-11 17:54:08,393 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for EURUSD with 0.25
2026-05-11 17:54:08,419 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for EURUSD: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 17:54:12,377 WARNING ML cache score overlay filled 4 warmup/alignment gaps for EURJPY
2026-05-11 17:54:13,808 WARNING ML cache score overlay filled 4 warmup/alignment gaps for USDJPY
2026-05-11 17:54:14,272 WARNING ML cache score overlay filled 4 warmup/alignment gaps for EURUSD
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
2026-05-11 17:54:23,665 WARNING _build_sequence_df: HTF frame 5M filled 357 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 17:54:25,645 WARNING _build_sequence_df: HTF frame 5M filled 357 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
2026-05-11 17:54:28,671 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 17:54:28,697 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 17:54:28,792 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 17:54:28,827 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
2026-05-11 17:54:28,832 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 17:54:28,885 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 17:54:28,910 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 17:54:28,939 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 17:54:28,978 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 17:54:28,980 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 17:54:29,009 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 17:54:29,040 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 17:54:29,059 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 17:54:29,097 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 17:54:29,112 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
2026-05-11 17:54:29,113 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 17:54:29,154 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 17:54:29,184 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
2026-05-11 17:54:29,195 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 17:54:29,216 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 17:54:29,241 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 17:54:29,261 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 17:54:29,301 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 17:54:29,351 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 17:54:29,395 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
2026-05-11 17:54:29,424 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 17:54:29,511 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 17:54:29,587 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 17:54:29,639 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 17:54:29,673 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 17:54:29,723 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 17:54:29,812 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 17:54:30,049 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 17:54:49,928 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for GBPUSD with 2
2026-05-11 17:54:49,944 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for GBPUSD with 0.3333333333333333
2026-05-11 17:54:50,071 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for GBPJPY with 2
2026-05-11 17:54:50,086 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for GBPJPY with 0.3333333333333333
2026-05-11 17:54:50,251 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for GBPUSD with 2
2026-05-11 17:54:50,280 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for GBPUSD with 0.25
2026-05-11 17:54:50,311 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for GBPUSD: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 17:54:50,483 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for GBPJPY with 2
2026-05-11 17:54:50,498 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for GBPJPY with 0.25
2026-05-11 17:54:50,514 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for GBPJPY: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 17:54:50,964 WARNING ML cache score overlay filled 4 warmup/alignment gaps for GBPUSD
2026-05-11 17:54:52,625 WARNING ML cache score overlay filled 4 warmup/alignment gaps for GBPJPY
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
2026-05-11 17:55:00,492 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
2026-05-11 17:55:00,559 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 17:55:00,591 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 17:55:00,614 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 17:55:00,636 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 17:55:00,654 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 17:55:00,676 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 17:55:00,694 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 17:55:00,712 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 17:55:00,730 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 17:55:00,747 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 17:55:00,789 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 17:55:00,816 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 17:55:00,855 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 17:55:00,883 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 17:55:00,908 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 17:55:00,925 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 17:55:00,954 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 17:55:00,978 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 17:55:00,998 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 17:55:01,017 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 17:55:01,079 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 17:58:00,137 WARNING ml_trader: portfolio drawdown 100.0% after trade exit — halting all trading

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260511_174955.json
Trader                                   Trades      WR    PF*  Return*    ExpR   TP1%   TP2%     DD*  Sharpe*  Sortino     IC  Verdict
-----------------------------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)              463  20.5%   0.73 -100.0%  -0.216 20.5%  3.9% 100.0%    -2.13    -0.22 -0.080     FAIL
  FAILED rules: positive_expectancy, profit_factor_min_1_25, drawdown_below_20pct, sharpe_positive, sortino_positive, win_rate_above_breakeven, mc_p10_not_ruin, sharpe_ci_positive
  monthly R: 2021-11=-2.81  2021-12=-1.00  2022-01=-7.01  2022-02=-2.00  2022-03=-3.00  2022-04=-2.29
  MonteCarlo P95 DD=117.3%  P10 equity=-3  t=-2.89 (p=0.004)  Sharpe CI=[-3.87, -0.49]  streak=24
  gate_diagnostics: bars=888285 no_signal=652237 quality_block=0 session_skip=235583 density=2 pm_reject=0 daily_skip=0 cooldown=0 daily_halt_events=0 enforce_daily_halt=False
  no_signal_reasons: no_trade_uncertain=226037, weak_gru_direction=163462, no_trade_chop=75202, gru_expected_r_below_threshold=64336, no_trade_extreme_vol=56763, wait_pullback=33033

Calibration Summary:
  all          [WARN] Only 1 populated bin(s) — cannot assess calibration monotonicity
  ml_trader    [WARN] Only 1 populated bin(s) — cannot assess calibration monotonicity
2026-05-11 17:58:01,794 INFO Round 0 backtest — 463 trades | avg WR=20.5% | avg PF=0.73 | avg Sharpe=-2.13
2026-05-11 17:58:01,794 INFO   ml_trader: 463 trades | WR=20.5% | fixed PF=0.73 | Return=-100.0% | ExpR=-0.216 | DD=100.0% | Sharpe=-2.13
2026-05-11 17:58:01,794 INFO   ml_trader gate_diagnostics: bars=888285 no_signal=652237 quality_block=0 session_skip=235583 density=2 pm_reject=0
2026-05-11 17:58:01,794 INFO   ml_trader no_signal_reasons: {'wait_pullback': 33033, 'trend_structure_missing': 14530, 'no_trade_uncertain': 226037, 'weak_gru_direction': 163462, 'no_trade_extreme_vol': 56763, 'gru_expected_r_below_threshold': 64336, 'no_trade_chop': 75202, 'htf_low_regime_confidence': 11173, 'tradeability_direction_conflict': 7676, 'expected_r_below_threshold': 25}
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_0.json
INFO  Total trades: 463
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (463 rows)
2026-05-11 17:58:02,389 INFO Round 0: wrote 463 journal entries (total in file: 463)
  DONE  Train-window backtest for Quality/RL labels
  Saved Train Quality/RL source result → train_quality_rl_source_summary.json
  Train-label journal entries: 463

=== Train Quality + RL on train-only journal ===
  START Train-only Quality+RL retrain
2026-05-11 17:58:02,741 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-05-11 17:58:02,758 INFO Journal entries: 463 total, 463 allowed for training (['live', 'paper', 'production', 'train'])
2026-05-11 17:58:02,758 INFO --- Training quality ---
2026-05-11 17:58:02,759 INFO Running retrain --model quality with JOURNAL_ALLOWED_SPLITS=train,live,paper,production
2026-05-11 17:58:02,960 INFO retrain environment: KAGGLE
2026-05-11 17:58:04,705 INFO Device: CUDA (2 GPU(s))
2026-05-11 17:58:04,717 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 17:58:04,717 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 17:58:04,717 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 17:58:04,718 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 17:58:04,718 INFO Retrain data split: train
2026-05-11 17:58:04,718 INFO Retrain rolling fold selector: latest
2026-05-11 17:58:04,719 INFO === QualityScorer retrain ===
2026-05-11 17:58:04,877 INFO NumExpr defaulting to 4 threads.
2026-05-11 17:58:05,088 INFO QualityScorer: CUDA available — using GPU
2026-05-11 17:58:05,304 INFO QualityScorer loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (device=cuda)
2026-05-11 17:58:05,374 INFO QualityScorer: group EV smoothing applied to 428/463 rows (blend=30% group, min_group=10)
2026-05-11 17:58:05,376 INFO Quality phase label creation: 0.1s (463 trades)
2026-05-11 17:58:05,377 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/quality_scorer.pkl_20260511_175805
2026-05-11 17:58:05,444 INFO QualityScorer: group EV smoothing applied to 428/463 rows (blend=30% group, min_group=10)
2026-05-11 17:58:05,448 INFO QualityScorer: 463 samples, EV stats={'mean': -0.4837709963321686, 'std': 0.7922940850257874, 'n_pos': 95, 'n_neg': 368}, device=cuda
2026-05-11 17:58:05,448 INFO QualityScorer: warm start from existing weights
2026-05-11 17:58:05,448 INFO QualityScorer: pos_weight=3.62 (n_pos=80 n_neg=290)
2026-05-11 17:58:07,756 INFO Quality epoch   1/100 — va_huber=0.5009
2026-05-11 17:58:07,797 INFO Quality epoch   2/100 — va_huber=0.5046
2026-05-11 17:58:07,818 INFO Quality epoch   3/100 — va_huber=0.5076
2026-05-11 17:58:07,839 INFO Quality epoch   4/100 — va_huber=0.5109
2026-05-11 17:58:07,859 INFO Quality epoch   5/100 — va_huber=0.5143
2026-05-11 17:58:08,176 INFO Quality epoch  11/100 — va_huber=0.5372
2026-05-11 17:58:08,176 INFO Quality early stop at epoch 11
2026-05-11 17:58:08,185 INFO QualityScorer EV model: MAE=0.914 dir_acc=0.161 n_val=93
2026-05-11 17:58:08,189 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-05-11 17:58:08,239 INFO Quality phase train: 2.9s | total: 3.5s
2026-05-11 17:58:08,248 INFO Retrain complete. Total wall-clock: 3.5s
2026-05-11 17:58:09,300 INFO Model quality: SUCCESS
2026-05-11 17:58:09,300 INFO --- Training rl ---
2026-05-11 17:58:09,301 INFO Running retrain --model rl with JOURNAL_ALLOWED_SPLITS=train,live,paper,production
2026-05-11 17:58:09,505 INFO retrain environment: KAGGLE
2026-05-11 17:58:11,195 INFO Device: CUDA (2 GPU(s))
2026-05-11 17:58:11,207 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 17:58:11,207 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 17:58:11,207 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 17:58:11,208 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 17:58:11,208 INFO Retrain data split: train
2026-05-11 17:58:11,208 INFO Retrain rolling fold selector: latest
2026-05-11 17:58:11,209 INFO === RLAgent (PPO) retrain ===
2026-05-11 17:58:11,367 INFO NumExpr defaulting to 4 threads.
2026-05-11 17:58:11,571 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260511_175811
2026-05-11 17:58:15.118137: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1778522295.363666   77265 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1778522295.435488   77265 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1778522295.980443   77265 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1778522295.980485   77265 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1778522295.980489   77265 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1778522295.980491   77265 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-05-11 17:58:33,824 INFO RLAgent: PPO model loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-05-11 17:58:33,853 INFO RL phase episode loading: 0.0s (463 episodes)
2026-05-11 17:58:33,877 INFO RLAgent: warm start — fine-tuning existing PPO policy (lr=6.00e-05)
2026-05-11 17:58:43,699 INFO RLAgent: retrain complete, 463 episodes
2026-05-11 17:58:43,700 INFO RL phase PPO train: 9.8s | total: 32.5s
2026-05-11 17:58:43,710 INFO Retrain complete. Total wall-clock: 32.5s
2026-05-11 17:58:45,522 INFO Model rl: SUCCESS
2026-05-11 17:58:45,522 INFO Step 7b complete — summary: /kaggle/working/Multi-Bot/trading-system/ml_training/metrics/training_7b_summary.json
  DONE  Train-only Quality+RL retrain
  Archived journal → trade_journal_train_only.jsonl
  Archived journal CSV → trade_journal_train_only.csv

=== Round 1: Backtest on train-tail window (latest 2yr inside training data) ===
  Cleared journal for fresh Round 1 run
  START Round 1 - Backtest (train-tail)
2026-05-11 17:58:46,041 INFO === STEP 6: BACKTEST (round1) ===
2026-05-11 17:58:46,042 INFO BT_WINDOW=round1 — train-tail backtest: 2021-08-05 → 2023-08-04 (seen training data; test set protected)
2026-05-11 17:58:46,043 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-05-11 17:58:46,043 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
2026-05-11 17:58:46,043 INFO Round 1 — QualityScorer hard gate disabled; set BACKTEST_ENABLE_QUALITY_GATE=1 to enable it
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
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
2026-05-11 18:00:08,624 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
2026-05-11 18:00:08,969 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
2026-05-11 18:00:08,977 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
2026-05-11 18:00:09,059 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 18:00:09,154 WARNING _build_sequence_df: HTF frame 5M filled 461 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 18:00:09,224 WARNING _build_sequence_df: HTF frame 5M filled 461 warmup/alignment gaps with 0.000
2026-05-11 18:00:09,224 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 18:00:09,328 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
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
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
2026-05-11 18:00:20,406 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 18:00:20,740 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 18:00:20,848 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 18:00:20,874 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 18:01:14,773 INFO Round 1 backtest — 115 trades | avg WR=21.7% | avg PF=0.70 | avg Sharpe=-2.43
2026-05-11 18:01:14,774 INFO   ml_trader: 115 trades | WR=21.7% | fixed PF=0.70 | Return=-26.6% | ExpR=-0.231 | DD=34.8% | Sharpe=-2.43
2026-05-11 18:01:14,774 INFO   ml_trader gate_diagnostics: bars=263960 no_signal=189214 quality_block=0 session_skip=74630 density=1 pm_reject=0
2026-05-11 18:01:14,774 INFO   ml_trader no_signal_reasons: {'no_trade_uncertain': 63244, 'weak_gru_direction': 47236, 'trend_structure_missing': 4029, 'no_trade_chop': 21601, 'gru_expected_r_below_threshold': 18825, 'wait_pullback': 10216, 'no_trade_extreme_vol': 17960, 'tradeability_direction_conflict': 2725, 'htf_low_regime_confidence': 3369, 'expected_r_below_threshold': 9}
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 115
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (115 rows)
2026-05-11 18:01:15,122 INFO Round 1: wrote 115 journal entries (total in file: 115)
  DONE  Round 1 - Backtest (train-tail)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 115 entries

  SKIP  Round 1 Quality+RL retrain — train-tail journal kept evaluation-only

  QualityScorer trade count: R0=463 R1=115 combined=578 (floor=50)
  Combined R0+R1 journal → trade_journal_qs_combined.jsonl (578 trades)

=== QualityScorer: 578 combined trades ≥ 50 — training and activating ===
  START Retrain quality [R0+R1 combined journal]
2026-05-11 18:01:15,570 INFO retrain environment: KAGGLE
2026-05-11 18:01:17,258 INFO Device: CUDA (2 GPU(s))
2026-05-11 18:01:17,270 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 18:01:17,270 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 18:01:17,270 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 18:01:17,270 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 18:01:17,270 INFO Retrain data split: train
2026-05-11 18:01:17,270 INFO Retrain rolling fold selector: latest
2026-05-11 18:01:17,271 INFO === QualityScorer retrain ===
2026-05-11 18:01:17,418 INFO NumExpr defaulting to 4 threads.
2026-05-11 18:01:17,625 INFO QualityScorer: CUDA available — using GPU
2026-05-11 18:01:17,835 INFO QualityScorer loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (device=cuda)
2026-05-11 18:01:17,861 INFO QualityScorer: skipped 115 journal records outside allowed splits ['combined_eval', 'live', 'paper', 'production', 'test', 'train', 'validation']
2026-05-11 18:01:17,907 INFO QualityScorer: group EV smoothing applied to 428/463 rows (blend=30% group, min_group=10)
2026-05-11 18:01:17,910 INFO Quality phase label creation: 0.1s (463 trades)
2026-05-11 18:01:17,910 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/quality_scorer.pkl_20260511_180117
2026-05-11 18:01:17,933 INFO QualityScorer: skipped 115 journal records outside allowed splits ['combined_eval', 'live', 'paper', 'production', 'test', 'train', 'validation']
2026-05-11 18:01:17,976 INFO QualityScorer: group EV smoothing applied to 428/463 rows (blend=30% group, min_group=10)
2026-05-11 18:01:17,979 INFO QualityScorer: 463 samples, EV stats={'mean': -0.4837709963321686, 'std': 0.7922940850257874, 'n_pos': 95, 'n_neg': 368}, device=cuda
2026-05-11 18:01:17,979 INFO QualityScorer: warm start from existing weights
2026-05-11 18:01:17,979 INFO QualityScorer: pos_weight=3.62 (n_pos=80 n_neg=290)
2026-05-11 18:01:20,297 INFO Quality epoch   1/100 — va_huber=0.5040
2026-05-11 18:01:20,333 INFO Quality epoch   2/100 — va_huber=0.5064
2026-05-11 18:01:20,355 INFO Quality epoch   3/100 — va_huber=0.5102
2026-05-11 18:01:20,376 INFO Quality epoch   4/100 — va_huber=0.5140
2026-05-11 18:01:20,397 INFO Quality epoch   5/100 — va_huber=0.5184
2026-05-11 18:01:20,709 INFO Quality epoch  11/100 — va_huber=0.5444
2026-05-11 18:01:20,709 INFO Quality early stop at epoch 11
2026-05-11 18:01:20,717 INFO QualityScorer EV model: MAE=0.919 dir_acc=0.161 n_val=93
2026-05-11 18:01:20,722 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-05-11 18:01:20,770 INFO Quality phase train: 2.9s | total: 3.5s
2026-05-11 18:01:20,780 INFO Retrain complete. Total wall-clock: 3.5s
  DONE  Retrain quality [R0+R1 combined journal]
  START Retrain win_loss [R0+R1 combined journal]
2026-05-11 18:01:22,085 INFO retrain environment: KAGGLE
2026-05-11 18:01:23,748 INFO Device: CUDA (2 GPU(s))
2026-05-11 18:01:23,764 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 18:01:23,765 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 18:01:23,765 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 18:01:23,765 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 18:01:23,765 INFO Retrain data split: train
2026-05-11 18:01:23,765 INFO Retrain rolling fold selector: latest
2026-05-11 18:01:23,766 INFO === Win/loss ANN classifier retrain ===
2026-05-11 18:01:23,810 WARNING WinLoss ANN retrain failed (non-fatal): Only 0 labeled trades after outcome filtering — need ≥30
2026-05-11 18:01:23,810 INFO Retrain complete. Total wall-clock: 0.0s
  DONE  Retrain win_loss [R0+R1 combined journal]
  QualityScorer trained — gate ACTIVE for Round 2+

=== Pre-Round 2: Ensemble models (RF + K-Means, trained on processed data) ===
  START Retrain rf [pre-R2 ensemble retrain]
2026-05-11 18:01:24,396 INFO retrain environment: KAGGLE
2026-05-11 18:01:26,106 INFO Device: CUDA (2 GPU(s))
2026-05-11 18:01:26,117 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 18:01:26,117 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 18:01:26,117 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 18:01:26,117 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 18:01:26,117 INFO Retrain data split: train
2026-05-11 18:01:26,117 INFO Retrain rolling fold selector: latest
2026-05-11 18:01:26,119 INFO === RF direction classifier retrain ===
2026-05-11 18:01:26,269 INFO NumExpr defaulting to 4 threads.
2026-05-11 18:01:26,597 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 18:01:26,606 INFO Loaded XAUUSD/15M split=train fold=latest: 176438 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:01:26,840 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:01:26,868 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:01:26,886 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:01:26,898 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:01:29,470 INFO Loaded EURUSD/15M split=train fold=latest: 174951 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:01:29,569 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:29,589 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:29,604 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:29,612 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:01:31,818 INFO Loaded USDJPY/15M split=train fold=latest: 174944 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:01:31,913 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:31,936 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:31,953 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:31,961 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:01:34,111 INFO Loaded EURJPY/15M split=train fold=latest: 174948 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:01:34,219 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:34,257 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:34,281 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:34,290 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:01:36,451 INFO Loaded GBPJPY/15M split=train fold=latest: 174902 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:01:36,539 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:36,560 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:36,577 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:36,585 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:01:38,638 INFO Loaded GBPUSD/15M split=train fold=latest: 174899 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:01:38,792 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:38,812 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:38,828 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:38,836 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:01:40,879 WARNING RF direction retrain failed (non-fatal): No feature data available for RF direction training
2026-05-11 18:01:40,879 INFO Retrain complete. Total wall-clock: 14.8s
  DONE  Retrain rf [pre-R2 ensemble retrain]
  START Retrain kmeans [pre-R2 ensemble retrain]
2026-05-11 18:01:41,635 INFO retrain environment: KAGGLE
2026-05-11 18:01:43,351 INFO Device: CUDA (2 GPU(s))
2026-05-11 18:01:43,362 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 18:01:43,363 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 18:01:43,363 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 18:01:43,363 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 18:01:43,363 INFO Retrain data split: train
2026-05-11 18:01:43,363 INFO Retrain rolling fold selector: latest
2026-05-11 18:01:43,364 INFO === K-Means regime retrain ===
2026-05-11 18:01:43,517 INFO NumExpr defaulting to 4 threads.
2026-05-11 18:01:43,758 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:01:43,798 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:01:43,799 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:01:43,811 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:01:43,816 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-11 18:01:43,816 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 18:01:43,816 INFO   GPU 1: Tesla T4 (15.6 GB)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:01:45,246 INFO KMeans: loaded XAUUSD 4H (27183 bars)
2026-05-11 18:01:45,265 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:45,286 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:45,286 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:45,294 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:01:46,148 INFO KMeans: loaded EURUSD 4H (15258 bars)
2026-05-11 18:01:46,165 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:46,188 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:46,188 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:46,197 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:01:47,058 INFO KMeans: loaded USDJPY 4H (15254 bars)
2026-05-11 18:01:47,075 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:47,096 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:47,097 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:47,105 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:01:47,945 INFO KMeans: loaded EURJPY 4H (15257 bars)
2026-05-11 18:01:47,962 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:47,983 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:47,983 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:47,992 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:01:48,834 INFO KMeans: loaded GBPJPY 4H (15259 bars)
2026-05-11 18:01:48,850 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:48,869 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:48,870 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:48,878 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:01:49,729 INFO KMeans: loaded GBPUSD 4H (15256 bars)
2026-05-11 18:01:49,731 INFO KMeans: total samples=103467
2026-05-11 18:01:50,895 INFO KMeansRegime train: N=103467 k=8
Traceback (most recent call last):
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1910, in <module>
    main()
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1887, in main
    result = retrain_kmeans(dry)
             ^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/retrain_incremental.py", line 1713, in retrain_kmeans
    result = km.train(X_all, n_clusters=n_clusters)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../models/kmeans_regime.py", line 172, in train
    km = KMeans(
         ^^^^^^^
TypeError: KMeans.__init__() got an unexpected keyword argument 'n_jobs'
  WARN  Retrain kmeans failed (exit 1) — continuing

=== Pre-Round 2: Incremental retrain (GRU + Regime) ===
  START Retrain gru [pre-R2 retrain]
2026-05-11 18:01:51,767 INFO retrain environment: KAGGLE
2026-05-11 18:01:53,433 INFO Device: CUDA (2 GPU(s))
2026-05-11 18:01:53,444 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 18:01:53,445 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 18:01:53,445 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 18:01:53,445 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 18:01:53,445 INFO Retrain data split: train
2026-05-11 18:01:53,445 INFO Retrain rolling fold selector: latest
2026-05-11 18:01:53,446 INFO === GRU-LSTM retrain (timeframes: ['15M']) ===
2026-05-11 18:01:53,597 INFO NumExpr defaulting to 4 threads.
2026-05-11 18:01:53,797 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-11 18:01:53,797 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 18:01:53,797 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 18:01:54,044 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-05-11 18:01:54,044 INFO GRU multi-symbol training (Kaggle mode): 6 symbols × ['15M']
2026-05-11 18:01:54,047 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260511_180154
2026-05-11 18:01:54,051 INFO GRU feature contract unchanged (input_size=94) — incremental retrain
2026-05-11 18:01:54,052 INFO Deleted stale GRU artifact for cold start: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:01:54,052 INFO GRU warm start disabled by default; set GRU_ALLOW_WARM_START=1 to reuse compatible weights
2026-05-11 18:01:54,341 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:01:54,371 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:01:54,389 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:01:54,400 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:01:54,483 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 18:01:54,489 INFO Loaded XAUUSD/15M split=train fold=latest: 176438 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:01:55,138 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:55,172 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:55,187 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:55,195 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:55,241 INFO Loaded EURUSD/15M split=train fold=latest: 174951 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:01:55,830 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:55,850 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:55,866 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:55,874 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:55,914 INFO Loaded USDJPY/15M split=train fold=latest: 174944 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:01:56,486 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:56,508 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:56,523 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:56,532 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:56,573 INFO Loaded EURJPY/15M split=train fold=latest: 174948 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:01:57,134 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:57,154 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:57,170 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:57,178 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:57,221 INFO Loaded GBPJPY/15M split=train fold=latest: 174902 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:01:57,779 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:57,799 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:57,814 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:57,822 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:01:57,863 INFO Loaded GBPUSD/15M split=train fold=latest: 174899 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:01:58,322 INFO train_multi: 6 segments, ~971854 total bars
2026-05-11 18:01:58,330 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-05-11 18:01:58,330 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-05-11 18:01:58,330 INFO train_multi: training ALL 6 segments across TFs ['15M'] in one combined pass
2026-05-11 18:01:58,330 INFO train_multi: building combined dataset for TF=ALL (6 segments)
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
2026-05-11 18:02:11,111 INFO train_multi TF=ALL: 971494 sequences across 6 segments
2026-05-11 18:02:11,111 INFO train_multi TF=ALL: estimated peak RAM = 27072 MB (train=419996 calib=60000 val=120002 n_feat=94 seq_len=60)
2026-05-11 18:02:11,111 WARNING train_multi TF=ALL: trimming to fit RAM budget — new train=310283 calib=44326 val=88652 (20000 MB est)
2026-05-11 18:02:13,506 INFO train_multi TF=ALL: train=310283 calib=44326 val=88652 (10007 MB tensors)
2026-05-11 18:02:20,392 INFO train_multi TF=ALL: structural bar weighting — 199279 structural bars (64.2%) weight=15.0 structural_only=0
2026-05-11 18:02:21,434 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=25, min_epochs=45)
2026-05-11 18:02:37,165 INFO train_multi TF=ALL epoch 1/100 train=2.3338 val=2.3364 r_mae=0.969 pos_r_acc=0.545 side_acc=0.490 r_n=127469
2026-05-11 18:02:37,171 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:02:37,171 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:02:37,171 INFO train_multi TF=ALL: new best val=2.3364 r_mae=0.9692 — saved
2026-05-11 18:02:37,175 INFO train_multi TF=ALL: new best r_mae=0.9692 — saved rmae checkpoint
2026-05-11 18:02:50,515 INFO train_multi TF=ALL epoch 2/100 train=2.3332 val=2.3354 r_mae=0.968 pos_r_acc=0.545 side_acc=0.490 r_n=127469
2026-05-11 18:02:50,521 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:02:50,521 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:02:50,521 INFO train_multi TF=ALL: new best val=2.3354 r_mae=0.9681 — saved
2026-05-11 18:02:50,526 INFO train_multi TF=ALL: new best r_mae=0.9681 — saved rmae checkpoint
2026-05-11 18:03:03,866 INFO train_multi TF=ALL epoch 3/100 train=2.3315 val=2.3342 r_mae=0.967 pos_r_acc=0.545 side_acc=0.490 r_n=127469
2026-05-11 18:03:03,871 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:03:03,871 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:03:03,871 INFO train_multi TF=ALL: new best val=2.3342 r_mae=0.9668 — saved
2026-05-11 18:03:03,876 INFO train_multi TF=ALL: new best r_mae=0.9668 — saved rmae checkpoint
2026-05-11 18:03:17,283 INFO train_multi TF=ALL epoch 4/100 train=2.3314 val=2.3333 r_mae=0.966 pos_r_acc=0.545 side_acc=0.490 r_n=127469
2026-05-11 18:03:17,289 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:03:17,289 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:03:17,289 INFO train_multi TF=ALL: new best val=2.3333 r_mae=0.9658 — saved
2026-05-11 18:03:17,293 INFO train_multi TF=ALL: new best r_mae=0.9658 — saved rmae checkpoint
2026-05-11 18:03:30,856 INFO train_multi TF=ALL epoch 5/100 train=2.3310 val=2.3330 r_mae=0.965 pos_r_acc=0.545 side_acc=0.490 r_n=127469
2026-05-11 18:03:30,861 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:03:30,861 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:03:30,861 INFO train_multi TF=ALL: new best val=2.3330 r_mae=0.9653 — saved
2026-05-11 18:03:30,866 INFO train_multi TF=ALL: new best r_mae=0.9653 — saved rmae checkpoint
2026-05-11 18:03:44,275 INFO train_multi TF=ALL epoch 6/100 train=2.3305 val=2.3327 r_mae=0.965 pos_r_acc=0.545 side_acc=0.490 r_n=127469
2026-05-11 18:03:44,281 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:03:44,281 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:03:44,281 INFO train_multi TF=ALL: new best val=2.3327 r_mae=0.9652 — saved
2026-05-11 18:03:44,285 INFO train_multi TF=ALL: new best r_mae=0.9652 — saved rmae checkpoint
2026-05-11 18:03:57,677 INFO train_multi TF=ALL epoch 7/100 train=2.3303 val=2.3323 r_mae=0.965 pos_r_acc=0.545 side_acc=0.493 r_n=127469
2026-05-11 18:03:57,682 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:03:57,682 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:03:57,682 INFO train_multi TF=ALL: new best val=2.3323 r_mae=0.9651 — saved
2026-05-11 18:03:57,687 INFO train_multi TF=ALL: new best r_mae=0.9651 — saved rmae checkpoint
2026-05-11 18:04:11,158 INFO train_multi TF=ALL epoch 8/100 train=2.3299 val=2.3314 r_mae=0.965 pos_r_acc=0.545 side_acc=0.532 r_n=127469
2026-05-11 18:04:11,168 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:04:11,168 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:04:11,168 INFO train_multi TF=ALL: new best val=2.3314 r_mae=0.9651 — saved
2026-05-11 18:04:11,173 INFO train_multi TF=ALL: new best r_mae=0.9651 — saved rmae checkpoint
2026-05-11 18:04:24,908 INFO train_multi TF=ALL epoch 9/100 train=2.3298 val=2.3304 r_mae=0.965 pos_r_acc=0.545 side_acc=0.523 r_n=127469
2026-05-11 18:04:24,913 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:04:24,913 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:04:24,913 INFO train_multi TF=ALL: new best val=2.3304 r_mae=0.9648 — saved
2026-05-11 18:04:24,918 INFO train_multi TF=ALL: new best r_mae=0.9648 — saved rmae checkpoint
2026-05-11 18:04:38,788 INFO train_multi TF=ALL epoch 10/100 train=2.3281 val=2.3280 r_mae=0.965 pos_r_acc=0.545 side_acc=0.527 r_n=127469
2026-05-11 18:04:38,801 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:04:38,801 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:04:38,801 INFO train_multi TF=ALL: new best val=2.3280 r_mae=0.9646 — saved
2026-05-11 18:04:38,805 INFO train_multi TF=ALL: new best r_mae=0.9646 — saved rmae checkpoint
2026-05-11 18:04:52,547 INFO train_multi TF=ALL epoch 11/100 train=2.3247 val=2.3250 r_mae=0.964 pos_r_acc=0.545 side_acc=0.526 r_n=127469
2026-05-11 18:04:52,553 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:04:52,553 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:04:52,553 INFO train_multi TF=ALL: new best val=2.3250 r_mae=0.9640 — saved
2026-05-11 18:04:52,558 INFO train_multi TF=ALL: new best r_mae=0.9640 — saved rmae checkpoint
2026-05-11 18:05:06,271 INFO train_multi TF=ALL epoch 12/100 train=2.3226 val=2.3223 r_mae=0.964 pos_r_acc=0.544 side_acc=0.531 r_n=127469
2026-05-11 18:05:06,277 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:05:06,277 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:05:06,277 INFO train_multi TF=ALL: new best val=2.3223 r_mae=0.9636 — saved
2026-05-11 18:05:06,287 INFO train_multi TF=ALL: new best r_mae=0.9636 — saved rmae checkpoint
2026-05-11 18:05:19,963 INFO train_multi TF=ALL epoch 13/100 train=2.3185 val=2.3214 r_mae=0.963 pos_r_acc=0.545 side_acc=0.532 r_n=127469
2026-05-11 18:05:19,968 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:05:19,968 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:05:19,969 INFO train_multi TF=ALL: new best val=2.3214 r_mae=0.9629 — saved
2026-05-11 18:05:19,973 INFO train_multi TF=ALL: new best r_mae=0.9629 — saved rmae checkpoint
2026-05-11 18:05:33,842 INFO train_multi TF=ALL epoch 14/100 train=2.3166 val=2.3191 r_mae=0.962 pos_r_acc=0.547 side_acc=0.534 r_n=127469
2026-05-11 18:05:33,847 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:05:33,847 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:05:33,847 INFO train_multi TF=ALL: new best val=2.3191 r_mae=0.9622 — saved
2026-05-11 18:05:33,851 INFO train_multi TF=ALL: new best r_mae=0.9622 — saved rmae checkpoint
2026-05-11 18:05:47,627 INFO train_multi TF=ALL epoch 15/100 train=2.3143 val=2.3182 r_mae=0.961 pos_r_acc=0.550 side_acc=0.535 r_n=127469
2026-05-11 18:05:47,633 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:05:47,633 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:05:47,633 INFO train_multi TF=ALL: new best val=2.3182 r_mae=0.9615 — saved
2026-05-11 18:05:47,637 INFO train_multi TF=ALL: new best r_mae=0.9615 — saved rmae checkpoint
2026-05-11 18:06:01,554 INFO train_multi TF=ALL epoch 16/100 train=2.3111 val=2.3159 r_mae=0.960 pos_r_acc=0.551 side_acc=0.537 r_n=127469
2026-05-11 18:06:01,560 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:06:01,560 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:06:01,560 INFO train_multi TF=ALL: new best val=2.3159 r_mae=0.9602 — saved
2026-05-11 18:06:01,565 INFO train_multi TF=ALL: new best r_mae=0.9602 — saved rmae checkpoint
2026-05-11 18:06:15,149 INFO train_multi TF=ALL epoch 17/100 train=2.3075 val=2.3147 r_mae=0.959 pos_r_acc=0.550 side_acc=0.540 r_n=127469
2026-05-11 18:06:15,155 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:06:15,155 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:06:15,155 INFO train_multi TF=ALL: new best val=2.3147 r_mae=0.9594 — saved
2026-05-11 18:06:15,160 INFO train_multi TF=ALL: new best r_mae=0.9594 — saved rmae checkpoint
2026-05-11 18:06:28,873 INFO train_multi TF=ALL epoch 18/100 train=2.3050 val=2.3126 r_mae=0.958 pos_r_acc=0.552 side_acc=0.540 r_n=127469
2026-05-11 18:06:28,879 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:06:28,879 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:06:28,879 INFO train_multi TF=ALL: new best val=2.3126 r_mae=0.9581 — saved
2026-05-11 18:06:28,883 INFO train_multi TF=ALL: new best r_mae=0.9581 — saved rmae checkpoint
2026-05-11 18:06:42,658 INFO train_multi TF=ALL epoch 19/100 train=2.3026 val=2.3124 r_mae=0.956 pos_r_acc=0.555 side_acc=0.539 r_n=127469
2026-05-11 18:06:42,665 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:06:42,665 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:06:42,665 INFO train_multi TF=ALL: new best val=2.3124 r_mae=0.9564 — saved
2026-05-11 18:06:42,670 INFO train_multi TF=ALL: new best r_mae=0.9564 — saved rmae checkpoint
2026-05-11 18:06:56,243 INFO train_multi TF=ALL epoch 20/100 train=2.2986 val=2.3084 r_mae=0.954 pos_r_acc=0.559 side_acc=0.543 r_n=127469
2026-05-11 18:06:56,248 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:06:56,248 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:06:56,248 INFO train_multi TF=ALL: new best val=2.3084 r_mae=0.9538 — saved
2026-05-11 18:06:56,253 INFO train_multi TF=ALL: new best r_mae=0.9538 — saved rmae checkpoint
2026-05-11 18:07:09,948 INFO train_multi TF=ALL epoch 21/100 train=2.2952 val=2.3000 r_mae=0.952 pos_r_acc=0.564 side_acc=0.545 r_n=127469
2026-05-11 18:07:09,959 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:07:09,959 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:07:09,959 INFO train_multi TF=ALL: new best val=2.3000 r_mae=0.9518 — saved
2026-05-11 18:07:09,963 INFO train_multi TF=ALL: new best r_mae=0.9518 — saved rmae checkpoint
2026-05-11 18:07:23,716 INFO train_multi TF=ALL epoch 22/100 train=2.2864 val=2.2964 r_mae=0.947 pos_r_acc=0.568 side_acc=0.547 r_n=127469
2026-05-11 18:07:23,722 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:07:23,722 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:07:23,722 INFO train_multi TF=ALL: new best val=2.2964 r_mae=0.9474 — saved
2026-05-11 18:07:23,727 INFO train_multi TF=ALL: new best r_mae=0.9474 — saved rmae checkpoint
2026-05-11 18:07:37,373 INFO train_multi TF=ALL epoch 23/100 train=2.2810 val=2.2886 r_mae=0.945 pos_r_acc=0.573 side_acc=0.548 r_n=127469
2026-05-11 18:07:37,378 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:07:37,378 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:07:37,378 INFO train_multi TF=ALL: new best val=2.2886 r_mae=0.9454 — saved
2026-05-11 18:07:37,383 INFO train_multi TF=ALL: new best r_mae=0.9454 — saved rmae checkpoint
2026-05-11 18:07:51,055 INFO train_multi TF=ALL epoch 24/100 train=2.2729 val=2.2864 r_mae=0.942 pos_r_acc=0.578 side_acc=0.546 r_n=127469
2026-05-11 18:07:51,061 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:07:51,061 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:07:51,061 INFO train_multi TF=ALL: new best val=2.2864 r_mae=0.9415 — saved
2026-05-11 18:07:51,065 INFO train_multi TF=ALL: new best r_mae=0.9415 — saved rmae checkpoint
2026-05-11 18:08:04,619 INFO train_multi TF=ALL epoch 25/100 train=2.2668 val=2.2783 r_mae=0.939 pos_r_acc=0.579 side_acc=0.555 r_n=127469
2026-05-11 18:08:04,632 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:08:04,632 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:08:04,632 INFO train_multi TF=ALL: new best val=2.2783 r_mae=0.9391 — saved
2026-05-11 18:08:04,638 INFO train_multi TF=ALL: new best r_mae=0.9391 — saved rmae checkpoint
2026-05-11 18:08:18,127 INFO train_multi TF=ALL epoch 26/100 train=2.2598 val=2.2761 r_mae=0.938 pos_r_acc=0.583 side_acc=0.554 r_n=127469
2026-05-11 18:08:18,133 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:08:18,133 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:08:18,133 INFO train_multi TF=ALL: new best val=2.2761 r_mae=0.9380 — saved
2026-05-11 18:08:18,138 INFO train_multi TF=ALL: new best r_mae=0.9380 — saved rmae checkpoint
2026-05-11 18:08:31,853 INFO train_multi TF=ALL epoch 27/100 train=2.2534 val=2.2734 r_mae=0.934 pos_r_acc=0.580 side_acc=0.556 r_n=127469
2026-05-11 18:08:31,859 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:08:31,859 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:08:31,859 INFO train_multi TF=ALL: new best val=2.2734 r_mae=0.9344 — saved
2026-05-11 18:08:31,863 INFO train_multi TF=ALL: new best r_mae=0.9344 — saved rmae checkpoint
2026-05-11 18:08:45,603 INFO train_multi TF=ALL epoch 28/100 train=2.2474 val=2.2740 r_mae=0.931 pos_r_acc=0.583 side_acc=0.559 r_n=127469
2026-05-11 18:08:45,608 INFO train_multi TF=ALL: new best r_mae=0.9313 — saved rmae checkpoint
2026-05-11 18:08:59,041 INFO train_multi TF=ALL epoch 29/100 train=2.2386 val=2.2708 r_mae=0.930 pos_r_acc=0.585 side_acc=0.557 r_n=127469
2026-05-11 18:08:59,046 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:08:59,046 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:08:59,046 INFO train_multi TF=ALL: new best val=2.2708 r_mae=0.9295 — saved
2026-05-11 18:08:59,056 INFO train_multi TF=ALL: new best r_mae=0.9295 — saved rmae checkpoint
2026-05-11 18:09:12,782 INFO train_multi TF=ALL epoch 30/100 train=2.2332 val=2.2684 r_mae=0.929 pos_r_acc=0.587 side_acc=0.558 r_n=127469
2026-05-11 18:09:12,788 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:09:12,788 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:09:12,788 INFO train_multi TF=ALL: new best val=2.2684 r_mae=0.9294 — saved
2026-05-11 18:09:12,792 INFO train_multi TF=ALL: new best r_mae=0.9294 — saved rmae checkpoint
2026-05-11 18:09:26,873 INFO train_multi TF=ALL epoch 31/100 train=2.2263 val=2.2660 r_mae=0.929 pos_r_acc=0.584 side_acc=0.556 r_n=127469
2026-05-11 18:09:26,878 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:09:26,878 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:09:26,879 INFO train_multi TF=ALL: new best val=2.2660 r_mae=0.9293 — saved
2026-05-11 18:09:26,883 INFO train_multi TF=ALL: new best r_mae=0.9293 — saved rmae checkpoint
2026-05-11 18:09:40,612 INFO train_multi TF=ALL epoch 32/100 train=2.2196 val=2.2638 r_mae=0.926 pos_r_acc=0.587 side_acc=0.561 r_n=127469
2026-05-11 18:09:40,618 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:09:40,618 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:09:40,618 INFO train_multi TF=ALL: new best val=2.2638 r_mae=0.9260 — saved
2026-05-11 18:09:40,623 INFO train_multi TF=ALL: new best r_mae=0.9260 — saved rmae checkpoint
2026-05-11 18:09:54,507 INFO train_multi TF=ALL epoch 33/100 train=2.2129 val=2.2566 r_mae=0.927 pos_r_acc=0.589 side_acc=0.565 r_n=127469
2026-05-11 18:09:54,512 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:09:54,512 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:09:54,513 INFO train_multi TF=ALL: new best val=2.2566 r_mae=0.9268 — saved
2026-05-11 18:10:08,033 INFO train_multi TF=ALL epoch 34/100 train=2.2077 val=2.2571 r_mae=0.921 pos_r_acc=0.589 side_acc=0.569 r_n=127469
2026-05-11 18:10:08,038 INFO train_multi TF=ALL: new best r_mae=0.9210 — saved rmae checkpoint
2026-05-11 18:10:21,447 INFO train_multi TF=ALL epoch 35/100 train=2.2001 val=2.2565 r_mae=0.921 pos_r_acc=0.589 side_acc=0.563 r_n=127469
2026-05-11 18:10:21,453 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:10:21,453 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:10:21,453 INFO train_multi TF=ALL: new best val=2.2565 r_mae=0.9213 — saved
2026-05-11 18:10:35,123 INFO train_multi TF=ALL epoch 36/100 train=2.1935 val=2.2644 r_mae=0.915 pos_r_acc=0.588 side_acc=0.569 r_n=127469
2026-05-11 18:10:35,128 INFO train_multi TF=ALL: new best r_mae=0.9152 — saved rmae checkpoint
2026-05-11 18:10:49,054 INFO train_multi TF=ALL epoch 37/100 train=2.1892 val=2.2660 r_mae=0.917 pos_r_acc=0.587 side_acc=0.566 r_n=127469
2026-05-11 18:11:02,841 INFO train_multi TF=ALL epoch 38/100 train=2.1772 val=2.2567 r_mae=0.914 pos_r_acc=0.590 side_acc=0.573 r_n=127469
2026-05-11 18:11:02,846 INFO train_multi TF=ALL: new best r_mae=0.9142 — saved rmae checkpoint
2026-05-11 18:11:16,432 INFO train_multi TF=ALL epoch 39/100 train=2.1717 val=2.2556 r_mae=0.915 pos_r_acc=0.589 side_acc=0.574 r_n=127469
2026-05-11 18:11:16,437 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:11:16,438 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:11:16,438 INFO train_multi TF=ALL: new best val=2.2556 r_mae=0.9146 — saved
2026-05-11 18:11:29,842 INFO train_multi TF=ALL epoch 40/100 train=2.1560 val=2.2574 r_mae=0.910 pos_r_acc=0.590 side_acc=0.579 r_n=127469
2026-05-11 18:11:29,846 INFO train_multi TF=ALL: new best r_mae=0.9097 — saved rmae checkpoint
2026-05-11 18:11:43,460 INFO train_multi TF=ALL epoch 41/100 train=2.1513 val=2.2312 r_mae=0.906 pos_r_acc=0.599 side_acc=0.585 r_n=127469
2026-05-11 18:11:43,465 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:11:43,465 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:11:43,465 INFO train_multi TF=ALL: new best val=2.2312 r_mae=0.9065 — saved
2026-05-11 18:11:43,470 INFO train_multi TF=ALL: new best r_mae=0.9065 — saved rmae checkpoint
2026-05-11 18:11:56,956 INFO train_multi TF=ALL epoch 42/100 train=2.1323 val=2.2399 r_mae=0.905 pos_r_acc=0.597 side_acc=0.581 r_n=127469
2026-05-11 18:11:56,961 INFO train_multi TF=ALL: new best r_mae=0.9050 — saved rmae checkpoint
2026-05-11 18:12:10,590 INFO train_multi TF=ALL epoch 43/100 train=2.1235 val=2.2204 r_mae=0.898 pos_r_acc=0.607 side_acc=0.592 r_n=127469
2026-05-11 18:12:10,596 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:12:10,596 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:12:10,596 INFO train_multi TF=ALL: new best val=2.2204 r_mae=0.8982 — saved
2026-05-11 18:12:10,600 INFO train_multi TF=ALL: new best r_mae=0.8982 — saved rmae checkpoint
2026-05-11 18:12:24,030 INFO train_multi TF=ALL epoch 44/100 train=2.1101 val=2.2039 r_mae=0.891 pos_r_acc=0.613 side_acc=0.598 r_n=127469
2026-05-11 18:12:24,035 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:12:24,035 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:12:24,035 INFO train_multi TF=ALL: new best val=2.2039 r_mae=0.8906 — saved
2026-05-11 18:12:24,040 INFO train_multi TF=ALL: new best r_mae=0.8906 — saved rmae checkpoint
2026-05-11 18:12:37,540 INFO train_multi TF=ALL epoch 45/100 train=2.0921 val=2.1962 r_mae=0.886 pos_r_acc=0.616 side_acc=0.601 r_n=127469
2026-05-11 18:12:37,547 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:12:37,547 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:12:37,547 INFO train_multi TF=ALL: new best val=2.1962 r_mae=0.8856 — saved
2026-05-11 18:12:37,552 INFO train_multi TF=ALL: new best r_mae=0.8856 — saved rmae checkpoint
2026-05-11 18:12:50,988 INFO train_multi TF=ALL epoch 46/100 train=2.0822 val=2.1762 r_mae=0.881 pos_r_acc=0.622 side_acc=0.608 r_n=127469
2026-05-11 18:12:50,993 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:12:50,994 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:12:50,994 INFO train_multi TF=ALL: new best val=2.1762 r_mae=0.8805 — saved
2026-05-11 18:12:50,998 INFO train_multi TF=ALL: new best r_mae=0.8805 — saved rmae checkpoint
2026-05-11 18:13:04,480 INFO train_multi TF=ALL epoch 47/100 train=2.0647 val=2.1705 r_mae=0.871 pos_r_acc=0.626 side_acc=0.613 r_n=127469
2026-05-11 18:13:04,485 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:13:04,485 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:13:04,485 INFO train_multi TF=ALL: new best val=2.1705 r_mae=0.8712 — saved
2026-05-11 18:13:04,490 INFO train_multi TF=ALL: new best r_mae=0.8712 — saved rmae checkpoint
2026-05-11 18:13:17,916 INFO train_multi TF=ALL epoch 48/100 train=2.0532 val=2.1854 r_mae=0.870 pos_r_acc=0.624 side_acc=0.606 r_n=127469
2026-05-11 18:13:17,921 INFO train_multi TF=ALL: new best r_mae=0.8696 — saved rmae checkpoint
2026-05-11 18:13:31,344 INFO train_multi TF=ALL epoch 49/100 train=2.0407 val=2.1617 r_mae=0.867 pos_r_acc=0.630 side_acc=0.614 r_n=127469
2026-05-11 18:13:31,349 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:13:31,349 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:13:31,349 INFO train_multi TF=ALL: new best val=2.1617 r_mae=0.8671 — saved
2026-05-11 18:13:31,354 INFO train_multi TF=ALL: new best r_mae=0.8671 — saved rmae checkpoint
2026-05-11 18:13:44,896 INFO train_multi TF=ALL epoch 50/100 train=2.0304 val=2.1563 r_mae=0.856 pos_r_acc=0.636 side_acc=0.617 r_n=127469
2026-05-11 18:13:44,901 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:13:44,901 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:13:44,901 INFO train_multi TF=ALL: new best val=2.1563 r_mae=0.8561 — saved
2026-05-11 18:13:44,906 INFO train_multi TF=ALL: new best r_mae=0.8561 — saved rmae checkpoint
2026-05-11 18:13:58,340 INFO train_multi TF=ALL epoch 51/100 train=2.0187 val=2.1585 r_mae=0.855 pos_r_acc=0.634 side_acc=0.618 r_n=127469
2026-05-11 18:13:58,345 INFO train_multi TF=ALL: new best r_mae=0.8555 — saved rmae checkpoint
2026-05-11 18:14:11,751 INFO train_multi TF=ALL epoch 52/100 train=2.0072 val=2.1584 r_mae=0.851 pos_r_acc=0.640 side_acc=0.617 r_n=127469
2026-05-11 18:14:11,757 INFO train_multi TF=ALL: new best r_mae=0.8505 — saved rmae checkpoint
2026-05-11 18:14:25,294 INFO train_multi TF=ALL epoch 53/100 train=1.9949 val=2.1398 r_mae=0.849 pos_r_acc=0.641 side_acc=0.621 r_n=127469
2026-05-11 18:14:25,300 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:14:25,300 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:14:25,300 INFO train_multi TF=ALL: new best val=2.1398 r_mae=0.8491 — saved
2026-05-11 18:14:25,305 INFO train_multi TF=ALL: new best r_mae=0.8491 — saved rmae checkpoint
2026-05-11 18:14:38,860 INFO train_multi TF=ALL epoch 54/100 train=1.9855 val=2.1402 r_mae=0.847 pos_r_acc=0.644 side_acc=0.623 r_n=127469
2026-05-11 18:14:38,865 INFO train_multi TF=ALL: new best r_mae=0.8474 — saved rmae checkpoint
2026-05-11 18:14:52,425 INFO train_multi TF=ALL epoch 55/100 train=1.9839 val=2.1591 r_mae=0.842 pos_r_acc=0.642 side_acc=0.622 r_n=127469
2026-05-11 18:14:52,430 INFO train_multi TF=ALL: new best r_mae=0.8419 — saved rmae checkpoint
2026-05-11 18:15:06,069 INFO train_multi TF=ALL epoch 56/100 train=1.9760 val=2.1503 r_mae=0.840 pos_r_acc=0.642 side_acc=0.621 r_n=127469
2026-05-11 18:15:06,074 INFO train_multi TF=ALL: new best r_mae=0.8400 — saved rmae checkpoint
2026-05-11 18:15:19,586 INFO train_multi TF=ALL epoch 57/100 train=1.9631 val=2.1516 r_mae=0.840 pos_r_acc=0.647 side_acc=0.620 r_n=127469
2026-05-11 18:15:19,591 INFO train_multi TF=ALL: new best r_mae=0.8399 — saved rmae checkpoint
2026-05-11 18:15:33,011 INFO train_multi TF=ALL epoch 58/100 train=1.9545 val=2.1372 r_mae=0.838 pos_r_acc=0.646 side_acc=0.626 r_n=127469
2026-05-11 18:15:33,017 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:15:33,017 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:15:33,017 INFO train_multi TF=ALL: new best val=2.1372 r_mae=0.8382 — saved
2026-05-11 18:15:33,022 INFO train_multi TF=ALL: new best r_mae=0.8382 — saved rmae checkpoint
2026-05-11 18:15:46,672 INFO train_multi TF=ALL epoch 59/100 train=1.9454 val=2.1400 r_mae=0.833 pos_r_acc=0.649 side_acc=0.628 r_n=127469
2026-05-11 18:15:46,677 INFO train_multi TF=ALL: new best r_mae=0.8333 — saved rmae checkpoint
2026-05-11 18:16:00,301 INFO train_multi TF=ALL epoch 60/100 train=1.9392 val=2.1595 r_mae=0.838 pos_r_acc=0.643 side_acc=0.620 r_n=127469
2026-05-11 18:16:13,861 INFO train_multi TF=ALL epoch 61/100 train=1.9283 val=2.1416 r_mae=0.834 pos_r_acc=0.649 side_acc=0.629 r_n=127469
2026-05-11 18:16:27,405 INFO train_multi TF=ALL epoch 62/100 train=1.9225 val=2.1472 r_mae=0.833 pos_r_acc=0.647 side_acc=0.627 r_n=127469
2026-05-11 18:16:27,410 INFO train_multi TF=ALL: new best r_mae=0.8326 — saved rmae checkpoint
2026-05-11 18:16:40,943 INFO train_multi TF=ALL epoch 63/100 train=1.9108 val=2.1518 r_mae=0.833 pos_r_acc=0.646 side_acc=0.628 r_n=127469
2026-05-11 18:16:54,470 INFO train_multi TF=ALL epoch 64/100 train=1.9070 val=2.1555 r_mae=0.830 pos_r_acc=0.650 side_acc=0.623 r_n=127469
2026-05-11 18:16:54,475 INFO train_multi TF=ALL: new best r_mae=0.8302 — saved rmae checkpoint
2026-05-11 18:17:08,035 INFO train_multi TF=ALL epoch 65/100 train=1.8987 val=2.1720 r_mae=0.833 pos_r_acc=0.643 side_acc=0.631 r_n=127469
2026-05-11 18:17:21,627 INFO train_multi TF=ALL epoch 66/100 train=1.8924 val=2.1381 r_mae=0.834 pos_r_acc=0.649 side_acc=0.631 r_n=127469
2026-05-11 18:17:35,136 INFO train_multi TF=ALL epoch 67/100 train=1.8801 val=2.1674 r_mae=0.828 pos_r_acc=0.649 side_acc=0.629 r_n=127469
2026-05-11 18:17:35,141 INFO train_multi TF=ALL: new best r_mae=0.8275 — saved rmae checkpoint
2026-05-11 18:17:48,630 INFO train_multi TF=ALL epoch 68/100 train=1.8757 val=2.1540 r_mae=0.826 pos_r_acc=0.651 side_acc=0.631 r_n=127469
2026-05-11 18:17:48,635 INFO train_multi TF=ALL: new best r_mae=0.8262 — saved rmae checkpoint
2026-05-11 18:18:02,097 INFO train_multi TF=ALL epoch 69/100 train=1.8681 val=2.1581 r_mae=0.831 pos_r_acc=0.648 side_acc=0.630 r_n=127469
2026-05-11 18:18:15,480 INFO train_multi TF=ALL epoch 70/100 train=1.8597 val=2.1751 r_mae=0.830 pos_r_acc=0.648 side_acc=0.629 r_n=127469
2026-05-11 18:18:28,836 INFO train_multi TF=ALL epoch 71/100 train=1.8555 val=2.1710 r_mae=0.829 pos_r_acc=0.647 side_acc=0.627 r_n=127469
2026-05-11 18:18:42,203 INFO train_multi TF=ALL epoch 72/100 train=1.8443 val=2.1708 r_mae=0.827 pos_r_acc=0.649 side_acc=0.629 r_n=127469
2026-05-11 18:18:55,697 INFO train_multi TF=ALL epoch 73/100 train=1.8428 val=2.1707 r_mae=0.826 pos_r_acc=0.649 side_acc=0.631 r_n=127469
2026-05-11 18:18:55,702 INFO train_multi TF=ALL: new best r_mae=0.8257 — saved rmae checkpoint
2026-05-11 18:19:09,080 INFO train_multi TF=ALL epoch 74/100 train=1.8351 val=2.1650 r_mae=0.826 pos_r_acc=0.649 side_acc=0.634 r_n=127469
2026-05-11 18:19:22,438 INFO train_multi TF=ALL epoch 75/100 train=1.8250 val=2.1632 r_mae=0.825 pos_r_acc=0.650 side_acc=0.632 r_n=127469
2026-05-11 18:19:22,443 INFO train_multi TF=ALL: new best r_mae=0.8248 — saved rmae checkpoint
2026-05-11 18:19:35,884 INFO train_multi TF=ALL epoch 76/100 train=1.8186 val=2.1916 r_mae=0.834 pos_r_acc=0.642 side_acc=0.631 r_n=127469
2026-05-11 18:19:49,313 INFO train_multi TF=ALL epoch 77/100 train=1.8155 val=2.1627 r_mae=0.828 pos_r_acc=0.649 side_acc=0.630 r_n=127469
2026-05-11 18:20:02,775 INFO train_multi TF=ALL epoch 78/100 train=1.8064 val=2.1781 r_mae=0.823 pos_r_acc=0.651 side_acc=0.631 r_n=127469
2026-05-11 18:20:02,780 INFO train_multi TF=ALL: new best r_mae=0.8231 — saved rmae checkpoint
2026-05-11 18:20:16,013 INFO train_multi TF=ALL epoch 79/100 train=1.7979 val=2.2110 r_mae=0.830 pos_r_acc=0.645 side_acc=0.626 r_n=127469
2026-05-11 18:20:29,208 INFO train_multi TF=ALL epoch 80/100 train=1.7928 val=2.1755 r_mae=0.825 pos_r_acc=0.650 side_acc=0.633 r_n=127469
2026-05-11 18:20:42,751 INFO train_multi TF=ALL epoch 81/100 train=1.7824 val=2.1829 r_mae=0.827 pos_r_acc=0.647 side_acc=0.632 r_n=127469
2026-05-11 18:20:56,300 INFO train_multi TF=ALL epoch 82/100 train=1.7809 val=2.2045 r_mae=0.833 pos_r_acc=0.643 side_acc=0.628 r_n=127469
2026-05-11 18:21:10,037 INFO train_multi TF=ALL epoch 83/100 train=1.7726 val=2.2047 r_mae=0.824 pos_r_acc=0.650 side_acc=0.633 r_n=127469
2026-05-11 18:21:23,644 INFO train_multi TF=ALL epoch 84/100 train=1.7625 val=2.2164 r_mae=0.828 pos_r_acc=0.646 side_acc=0.627 r_n=127469
2026-05-11 18:21:37,255 INFO train_multi TF=ALL epoch 85/100 train=1.7588 val=2.2132 r_mae=0.833 pos_r_acc=0.643 side_acc=0.633 r_n=127469
2026-05-11 18:21:50,720 INFO train_multi TF=ALL epoch 86/100 train=1.7507 val=2.2031 r_mae=0.827 pos_r_acc=0.648 side_acc=0.635 r_n=127469
2026-05-11 18:22:04,120 INFO train_multi TF=ALL epoch 87/100 train=1.7482 val=2.2137 r_mae=0.829 pos_r_acc=0.646 side_acc=0.634 r_n=127469
2026-05-11 18:22:17,571 INFO train_multi TF=ALL epoch 88/100 train=1.7338 val=2.2361 r_mae=0.827 pos_r_acc=0.647 side_acc=0.627 r_n=127469
2026-05-11 18:22:30,883 INFO train_multi TF=ALL epoch 89/100 train=1.7328 val=2.2468 r_mae=0.833 pos_r_acc=0.643 side_acc=0.628 r_n=127469
2026-05-11 18:22:44,337 INFO train_multi TF=ALL epoch 90/100 train=1.7256 val=2.2173 r_mae=0.828 pos_r_acc=0.646 side_acc=0.634 r_n=127469
2026-05-11 18:22:57,811 INFO train_multi TF=ALL epoch 91/100 train=1.7224 val=2.2271 r_mae=0.837 pos_r_acc=0.640 side_acc=0.629 r_n=127469
2026-05-11 18:23:11,247 INFO train_multi TF=ALL epoch 92/100 train=1.7189 val=2.2267 r_mae=0.832 pos_r_acc=0.643 side_acc=0.630 r_n=127469
2026-05-11 18:23:24,749 INFO train_multi TF=ALL epoch 93/100 train=1.7100 val=2.2480 r_mae=0.832 pos_r_acc=0.643 side_acc=0.628 r_n=127469
2026-05-11 18:23:38,231 INFO train_multi TF=ALL epoch 94/100 train=1.7048 val=2.2287 r_mae=0.832 pos_r_acc=0.644 side_acc=0.631 r_n=127469
2026-05-11 18:23:51,524 INFO train_multi TF=ALL epoch 95/100 train=1.6990 val=2.2389 r_mae=0.830 pos_r_acc=0.646 side_acc=0.632 r_n=127469
2026-05-11 18:24:05,057 INFO train_multi TF=ALL epoch 96/100 train=1.6944 val=2.2272 r_mae=0.831 pos_r_acc=0.645 side_acc=0.633 r_n=127469
2026-05-11 18:24:18,459 INFO train_multi TF=ALL epoch 97/100 train=1.6901 val=2.2378 r_mae=0.835 pos_r_acc=0.642 side_acc=0.631 r_n=127469
2026-05-11 18:24:31,799 INFO train_multi TF=ALL epoch 98/100 train=1.6786 val=2.2368 r_mae=0.829 pos_r_acc=0.645 side_acc=0.637 r_n=127469
2026-05-11 18:24:45,269 INFO train_multi TF=ALL epoch 99/100 train=1.6756 val=2.2469 r_mae=0.829 pos_r_acc=0.644 side_acc=0.636 r_n=127469
2026-05-11 18:24:59,015 INFO train_multi TF=ALL epoch 100/100 train=1.6744 val=2.2557 r_mae=0.834 pos_r_acc=0.643 side_acc=0.631 r_n=127469
2026-05-11 18:24:59,027 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:24:59,027 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:24:59,027 INFO train_multi TF=ALL: promoted r_mae checkpoint (0.8231 < primary 0.8382) — overwriting model.pt
2026-05-11 18:25:00,264 WARNING fit_r_isotonic: long calibrator increases MAE (cal=0.8475 >= raw=0.8261) — skipping
2026-05-11 18:25:00,274 WARNING fit_r_isotonic: short calibrator increases MAE (cal=0.8555 >= raw=0.8392) — skipping
2026-05-11 18:25:00,275 INFO fit_r_isotonic: saved side-R calibrators stats={'long': {'n': 31689, 'raw_mae': 0.8260690771046747, 'calibrated_mae': 0.8475125494790529, 'skipped': 'calibrator_hurts'}, 'short': {'n': 32408, 'raw_mae': 0.8391701203241061, 'calibrated_mae': 0.8555243114009503, 'skipped': 'calibrator_hurts'}}
2026-05-11 18:25:00,424 WARNING GRU validation R-MAE above floor for TF=ALL: best_val_r_mae=0.823 max=0.750. Keeping saved best weights so the pipeline can progress.
2026-05-11 18:25:00,438 INFO GRU R threshold XAUUSD/buy: q25_pos=0.535 q50_pos=1.667 pos_rate=46.0% (n=119259 n_pos=54899)
2026-05-11 18:25:00,445 INFO GRU R threshold XAUUSD/sell: q25_pos=0.528 q50_pos=1.667 pos_rate=45.5% (n=118087 n_pos=53734)
2026-05-11 18:25:00,451 INFO GRU R threshold EURUSD/buy: q25_pos=0.541 q50_pos=1.667 pos_rate=45.6% (n=118751 n_pos=54196)
2026-05-11 18:25:00,457 INFO GRU R threshold EURUSD/sell: q25_pos=0.545 q50_pos=1.667 pos_rate=45.6% (n=117679 n_pos=53721)
2026-05-11 18:25:00,463 INFO GRU R threshold USDJPY/buy: q25_pos=0.551 q50_pos=1.667 pos_rate=46.2% (n=118691 n_pos=54870)
2026-05-11 18:25:00,470 INFO GRU R threshold USDJPY/sell: q25_pos=0.542 q50_pos=1.667 pos_rate=44.6% (n=116030 n_pos=51793)
2026-05-11 18:25:00,476 INFO GRU R threshold EURJPY/buy: q25_pos=0.530 q50_pos=1.416 pos_rate=46.1% (n=118651 n_pos=54654)
2026-05-11 18:25:00,483 INFO GRU R threshold EURJPY/sell: q25_pos=0.525 q50_pos=1.667 pos_rate=44.7% (n=117320 n_pos=52399)
2026-05-11 18:25:00,490 INFO GRU R threshold GBPJPY/buy: q25_pos=0.517 q50_pos=1.443 pos_rate=45.9% (n=118277 n_pos=54278)
2026-05-11 18:25:00,496 INFO GRU R threshold GBPJPY/sell: q25_pos=0.521 q50_pos=1.667 pos_rate=45.3% (n=116299 n_pos=52702)
2026-05-11 18:25:00,502 INFO GRU R threshold GBPUSD/buy: q25_pos=0.524 q50_pos=1.667 pos_rate=45.8% (n=117764 n_pos=53894)
2026-05-11 18:25:00,510 INFO GRU R threshold GBPUSD/sell: q25_pos=0.533 q50_pos=1.667 pos_rate=45.8% (n=117404 n_pos=53785)
2026-05-11 18:25:00,510 INFO GRU per-symbol R thresholds saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/symbol_r_thresholds.json (6 symbols)
2026-05-11 18:25:00,529 INFO Retrain complete. Total wall-clock: 1387.1s
  DONE  Retrain gru [pre-R2 retrain]
  START Retrain regime [pre-R2 retrain]
2026-05-11 18:25:04,129 INFO retrain environment: KAGGLE
2026-05-11 18:25:05,857 INFO Device: CUDA (2 GPU(s))
2026-05-11 18:25:05,866 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 18:25:05,866 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 18:25:05,866 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 18:25:05,866 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 18:25:05,867 INFO Retrain data split: train
2026-05-11 18:25:05,867 INFO Retrain rolling fold selector: latest
2026-05-11 18:25:05,868 INFO === RegimeClassifier retrain (hierarchical: HTF 2-score bias + LTF 5-score behaviour) ===
2026-05-11 18:25:06,017 INFO NumExpr defaulting to 4 threads.
2026-05-11 18:25:06,226 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-11 18:25:06,227 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 18:25:06,227 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 18:25:06,227 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-11 18:25:06,291 INFO Regime rolling folds selected: [None]
2026-05-11 18:25:06,291 INFO === Regime rolling fold 1/1: train_all ===
2026-05-11 18:25:06,291 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-11 18:25:06,335 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 18:25:06,336 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:25:06,353 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:25:06,369 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:25:06,386 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:25:06,405 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:25:06,422 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:25:06,667 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:25:06,741 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:25:06,767 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:25:06,768 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:25:06,779 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:25:06,780 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:25:07,607 INFO macro_cache: using training data dir /kaggle/working/Multi-Bot/trading-system/training_data
2026-05-11 18:25:07,728 INFO macro_cache loaded 19 series: {'asx200': '2842 bars 2015-01-02→2026-03-27', 'cac40': '2876 bars 2015-01-02→2026-03-27', 'dax': '2851 bars 2015-01-02→2026-03-27', 'djia': '2825 bars 2015-01-02→2026-03-27', 'dxy': '2826 bars 2015-01-02→2026-03-27', 'eurostoxx': '2823 bars 2015-01-05→2026-03-27', 'ftse': '2839 bars 2015-01-02→2026-03-27', 'gold_fut': '2824 bars 2015-01-02→2026-03-27', 'hsi': '2764 bars 2015-01-02→2026-03-27', 'nasdaq': '2825 bars 2015-01-02→2026-03-27', 'nikkei': '2744 bars 2015-01-05→2026-03-27', 'oil_fut': '2825 bars 2015-01-02→2026-03-27', 'spx': '2825 bars 2015-01-02→2026-03-27', 'us10y': '2824 bars 2015-01-02→2026-03-27', 'us30y': '2824 bars 2015-01-02→2026-03-27', 'us3m': '2824 bars 2015-01-02→2026-03-27', 'vix': '2825 bars 2015-01-02→2026-03-27', 'us10y_fred': '2607 bars 2016-03-28→2026-03-24', 'us2y_fred': '2607 bars 2016-03-28→2026-03-24'}
2026-05-11 18:25:07,987 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1142, 'BIAS_DOWN': 522, 'BIAS_NEUTRAL': 10438}  ambiguous=4182 (total=12102) horizon=84
2026-05-11 18:25:07,992 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected XAUUSD — 12052 samples (group=gold) score_means={'bias_up_score': 0.0948, 'bias_down_score': 0.0433} labels={'BIAS_UP': 1142, 'BIAS_DOWN': 522, 'BIAS_NEUTRAL': 10388} clean={'BIAS_UP': 1142, 'BIAS_DOWN': 522, 'BIAS_NEUTRAL': 6216}
2026-05-11 18:25:08,172 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:08,210 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:08,231 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:08,231 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:08,240 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:08,242 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:25:09,254 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 690, 'BIAS_DOWN': 540, 'BIAS_NEUTRAL': 10174}  ambiguous=3886 (total=11404) horizon=84
2026-05-11 18:25:09,260 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURUSD — 11354 samples (group=dollar) score_means={'bias_up_score': 0.0608, 'bias_down_score': 0.0476} labels={'BIAS_UP': 690, 'BIAS_DOWN': 540, 'BIAS_NEUTRAL': 10124} clean={'BIAS_UP': 690, 'BIAS_DOWN': 540, 'BIAS_NEUTRAL': 6257}
2026-05-11 18:25:09,425 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:09,463 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:09,485 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:09,485 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:09,494 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:09,495 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:25:10,513 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 826, 'BIAS_DOWN': 423, 'BIAS_NEUTRAL': 10154}  ambiguous=4036 (total=11403) horizon=84
2026-05-11 18:25:10,519 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected USDJPY — 11353 samples (group=dollar) score_means={'bias_up_score': 0.0728, 'bias_down_score': 0.0373} labels={'BIAS_UP': 826, 'BIAS_DOWN': 423, 'BIAS_NEUTRAL': 10104} clean={'BIAS_UP': 826, 'BIAS_DOWN': 423, 'BIAS_NEUTRAL': 6078}
2026-05-11 18:25:10,694 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:10,734 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:10,756 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:10,756 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:10,765 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:10,766 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:25:11,774 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 681, 'BIAS_DOWN': 527, 'BIAS_NEUTRAL': 10199}  ambiguous=4044 (total=11407) horizon=84
2026-05-11 18:25:11,780 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURJPY — 11357 samples (group=cross) score_means={'bias_up_score': 0.06, 'bias_down_score': 0.0464} labels={'BIAS_UP': 681, 'BIAS_DOWN': 527, 'BIAS_NEUTRAL': 10149} clean={'BIAS_UP': 681, 'BIAS_DOWN': 527, 'BIAS_NEUTRAL': 6111}
2026-05-11 18:25:11,938 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:11,975 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:11,997 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:11,997 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:12,008 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:12,009 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:25:13,006 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 839, 'BIAS_DOWN': 579, 'BIAS_NEUTRAL': 9990}  ambiguous=4240 (total=11408) horizon=84
2026-05-11 18:25:13,012 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPJPY — 11358 samples (group=cross) score_means={'bias_up_score': 0.0739, 'bias_down_score': 0.051} labels={'BIAS_UP': 839, 'BIAS_DOWN': 579, 'BIAS_NEUTRAL': 9940} clean={'BIAS_UP': 839, 'BIAS_DOWN': 579, 'BIAS_NEUTRAL': 5723}
2026-05-11 18:25:13,176 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:13,211 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:13,232 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:13,232 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:13,241 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:13,242 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:25:14,223 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 502, 'BIAS_DOWN': 720, 'BIAS_NEUTRAL': 10180}  ambiguous=4113 (total=11402) horizon=84
2026-05-11 18:25:14,230 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPUSD — 11352 samples (group=dollar) score_means={'bias_up_score': 0.0442, 'bias_down_score': 0.0623} labels={'BIAS_UP': 502, 'BIAS_DOWN': 707, 'BIAS_NEUTRAL': 10143} clean={'BIAS_UP': 502, 'BIAS_DOWN': 707, 'BIAS_NEUTRAL': 6056}
2026-05-11 18:25:14,304 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 1520, 'BIAS_DOWN': 1106, 'BIAS_NEUTRAL': 20089}, 'dollar': {'BIAS_UP': 2018, 'BIAS_DOWN': 1670, 'BIAS_NEUTRAL': 30371}, 'gold': {'BIAS_UP': 1142, 'BIAS_DOWN': 522, 'BIAS_NEUTRAL': 10388}}
2026-05-11 18:25:14,304 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0669, 'bias_down_score': 0.0487}, 'dollar': {'bias_up_score': 0.0593, 'bias_down_score': 0.049}, 'gold': {'bias_up_score': 0.0948, 'bias_down_score': 0.0433}}
2026-05-11 18:25:14,304 INFO Regime[4H mode=htf_bias] label distribution by year: {2016: {'BIAS_UP': 491, 'BIAS_DOWN': 576, 'BIAS_NEUTRAL': 7755}, 2017: {'BIAS_UP': 734, 'BIAS_DOWN': 286, 'BIAS_NEUTRAL': 8093}, 2018: {'BIAS_UP': 427, 'BIAS_DOWN': 714, 'BIAS_NEUTRAL': 7989}, 2019: {'BIAS_UP': 410, 'BIAS_DOWN': 447, 'BIAS_NEUTRAL': 8245}, 2020: {'BIAS_UP': 694, 'BIAS_DOWN': 174, 'BIAS_NEUTRAL': 8243}, 2021: {'BIAS_UP': 722, 'BIAS_DOWN': 473, 'BIAS_NEUTRAL': 7896}, 2022: {'BIAS_UP': 667, 'BIAS_DOWN': 519, 'BIAS_NEUTRAL': 7935}, 2023: {'BIAS_UP': 535, 'BIAS_DOWN': 109, 'BIAS_NEUTRAL': 4692}}
2026-05-11 18:25:14,304 INFO Regime[4H mode=htf_bias] score means by year: {2016: {'bias_up_score': 0.0557, 'bias_down_score': 0.0653}, 2017: {'bias_up_score': 0.0805, 'bias_down_score': 0.0314}, 2018: {'bias_up_score': 0.0468, 'bias_down_score': 0.0782}, 2019: {'bias_up_score': 0.045, 'bias_down_score': 0.0491}, 2020: {'bias_up_score': 0.0762, 'bias_down_score': 0.0191}, 2021: {'bias_up_score': 0.0794, 'bias_down_score': 0.052}, 2022: {'bias_up_score': 0.0731, 'bias_down_score': 0.0569}, 2023: {'bias_up_score': 0.1003, 'bias_down_score': 0.0204}}
2026-05-11 18:25:14,359 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
2026-05-11 18:25:14,360 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
2026-05-11 18:25:14,361 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
2026-05-11 18:25:14,362 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 18:25:14,363 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 18:25:14,364 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
2026-05-11 18:25:14,381 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:25:14,385 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:25:14,386 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:25:14,386 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:25:14,387 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:25:14,388 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:25:15,012 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 157, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1339}  ambiguous=566 (total=1581) horizon=84
2026-05-11 18:25:15,015 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected XAUUSD — 1531 samples (group=gold) score_means={'bias_up_score': 0.1025, 'bias_down_score': 0.0555} labels={'BIAS_UP': 157, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1289} clean={'BIAS_UP': 157, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 744}
2026-05-11 18:25:15,094 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:15,096 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:15,097 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:15,098 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:15,098 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:15,099 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:25:15,671 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 135, 'BIAS_DOWN': 66, 'BIAS_NEUTRAL': 1290}  ambiguous=531 (total=1491) horizon=84
2026-05-11 18:25:15,673 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURUSD — 1441 samples (group=dollar) score_means={'bias_up_score': 0.0937, 'bias_down_score': 0.0458} labels={'BIAS_UP': 135, 'BIAS_DOWN': 66, 'BIAS_NEUTRAL': 1240} clean={'BIAS_UP': 135, 'BIAS_DOWN': 66, 'BIAS_NEUTRAL': 739}
2026-05-11 18:25:15,748 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:15,751 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:15,752 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:15,752 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:15,752 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:15,753 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:25:16,322 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 164, 'BIAS_DOWN': 77, 'BIAS_NEUTRAL': 1248}  ambiguous=616 (total=1489) horizon=84
2026-05-11 18:25:16,325 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected USDJPY — 1439 samples (group=dollar) score_means={'bias_up_score': 0.114, 'bias_down_score': 0.0535} labels={'BIAS_UP': 164, 'BIAS_DOWN': 77, 'BIAS_NEUTRAL': 1198} clean={'BIAS_UP': 164, 'BIAS_DOWN': 77, 'BIAS_NEUTRAL': 608}
2026-05-11 18:25:16,400 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:16,402 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:16,403 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:16,403 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:16,404 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:16,405 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:25:16,971 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 123, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 1366}  ambiguous=582 (total=1494) horizon=84
2026-05-11 18:25:16,973 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0852, 'bias_down_score': 0.0035} labels={'BIAS_UP': 123, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 1316} clean={'BIAS_UP': 123, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 741}
2026-05-11 18:25:17,047 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:17,049 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:17,050 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:17,051 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:17,051 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:17,052 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:25:17,615 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 129, 'BIAS_DOWN': 9, 'BIAS_NEUTRAL': 1356}  ambiguous=551 (total=1494) horizon=84
2026-05-11 18:25:17,618 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0893, 'bias_down_score': 0.0055} labels={'BIAS_UP': 129, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 1307} clean={'BIAS_UP': 129, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 775}
2026-05-11 18:25:17,693 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:17,695 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:17,696 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:17,696 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:17,696 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:17,697 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:25:18,255 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 81, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 1316}  ambiguous=560 (total=1488) horizon=84
2026-05-11 18:25:18,258 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPUSD — 1438 samples (group=dollar) score_means={'bias_up_score': 0.0563, 'bias_down_score': 0.0633} labels={'BIAS_UP': 81, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 1266} clean={'BIAS_UP': 81, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 735}
2026-05-11 18:25:18,330 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 252, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 2623}, 'dollar': {'BIAS_UP': 380, 'BIAS_DOWN': 234, 'BIAS_NEUTRAL': 3704}, 'gold': {'BIAS_UP': 157, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1289}}
2026-05-11 18:25:18,330 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0873, 'bias_down_score': 0.0045}, 'dollar': {'bias_up_score': 0.088, 'bias_down_score': 0.0542}, 'gold': {'bias_up_score': 0.1025, 'bias_down_score': 0.0555}}
2026-05-11 18:25:18,330 INFO Regime[4H mode=htf_bias] label distribution by year: {2022: {'BIAS_UP': 258, 'BIAS_DOWN': 228, 'BIAS_NEUTRAL': 2915}, 2023: {'BIAS_UP': 531, 'BIAS_DOWN': 104, 'BIAS_NEUTRAL': 4701}}
2026-05-11 18:25:18,330 INFO Regime[4H mode=htf_bias] score means by year: {2022: {'bias_up_score': 0.0759, 'bias_down_score': 0.067}, 2023: {'bias_up_score': 0.0995, 'bias_down_score': 0.0195}}
2026-05-11 18:25:18,383 INFO Regime phase HTF dataset build fold=train_all: 12.1s (train=68826 val=8737)
2026-05-11 18:25:18,384 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260511_182518
2026-05-11 18:25:18,588 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=51, n_classes=2)
2026-05-11 18:25:18,588 INFO Regime 4H/htf_bias warm start enabled from existing weights
2026-05-11 18:25:18,602 INFO RegimeClassifier[mode=htf_bias]: HTF clean-label fit filter kept train=44419/68826 val=5463/8737 at conf>=0.40 train_counts={'BIAS_UP': 4680, 'BIAS_DOWN': 3298, 'BIAS_NEUTRAL': 36441} val_counts={'BIAS_UP': 789, 'BIAS_DOWN': 332, 'BIAS_NEUTRAL': 4342}
2026-05-11 18:25:18,603 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=44419 val=5463 train_labels={'BIAS_UP': 4680, 'BIAS_DOWN': 3298, 'BIAS_NEUTRAL': 36441} val_labels={'BIAS_UP': 789, 'BIAS_DOWN': 332, 'BIAS_NEUTRAL': 4342}
2026-05-11 18:25:18,604 INFO RegimeClassifier[mode=htf_bias]: warm start HTF score head
2026-05-11 18:25:18,604 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-11 18:25:18,604 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 8.491, 'bias_down_score': 12.0}
2026-05-11 18:25:18,608 INFO RegimeClassifier[mode=htf_bias]: HTF balanced sampler — dir=7978 neutral=36441 dir_weight=3 => dir_frac_per_epoch≈47.2%
2026-05-11 18:25:22,154 INFO Regime HTF score epoch  1/50 — tr=2.6646 va=1.0212 acc=0.781 bal=0.674 threshold=0.40 margin=0.15 recall={'BIAS_UP': 0.512, 'BIAS_DOWN': 0.672, 'BIAS_NEUTRAL': 0.838} precision={'BIAS_UP': 0.539, 'BIAS_DOWN': 0.384, 'BIAS_NEUTRAL': 0.88}
2026-05-11 18:25:23,584 INFO Regime HTF score epoch  2/50 — tr=2.6335 va=1.0175 bal=0.620
2026-05-11 18:25:24,972 INFO Regime HTF score epoch  3/50 — tr=2.5898 va=1.0004 bal=0.637
2026-05-11 18:25:26,327 INFO Regime HTF score epoch  4/50 — tr=2.5794 va=0.9813 bal=0.648
2026-05-11 18:25:27,681 INFO Regime HTF score epoch  5/50 — tr=2.5459 va=0.9525 acc=0.786 bal=0.641 threshold=0.35 margin=0.45 recall={'BIAS_UP': 0.484, 'BIAS_DOWN': 0.581, 'BIAS_NEUTRAL': 0.857} precision={'BIAS_UP': 0.545, 'BIAS_DOWN': 0.39, 'BIAS_NEUTRAL': 0.872}
2026-05-11 18:25:29,097 INFO Regime HTF score epoch  6/50 — tr=2.4647 va=0.9261 bal=0.631
2026-05-11 18:25:30,436 INFO Regime HTF score epoch  7/50 — tr=2.3980 va=0.8963 bal=0.653
2026-05-11 18:25:31,891 INFO Regime HTF score epoch  8/50 — tr=2.3411 va=0.8679 bal=0.617
2026-05-11 18:25:33,332 INFO Regime HTF score epoch  9/50 — tr=2.2420 va=0.8425 bal=0.639
2026-05-11 18:25:34,826 INFO Regime HTF score epoch 10/50 — tr=2.1885 va=0.8188 acc=0.790 bal=0.603 threshold=0.60 margin=0.15 recall={'BIAS_UP': 0.417, 'BIAS_DOWN': 0.512, 'BIAS_NEUTRAL': 0.879} precision={'BIAS_UP': 0.56, 'BIAS_DOWN': 0.39, 'BIAS_NEUTRAL': 0.86}
2026-05-11 18:25:34,826 INFO Regime HTF score early stop at epoch 10
2026-05-11 18:25:36,118 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.400 margin=0.150 precision={'BIAS_UP': 0.539, 'BIAS_DOWN': 0.384, 'BIAS_NEUTRAL': 0.88} recall={'BIAS_UP': 0.512, 'BIAS_DOWN': 0.672, 'BIAS_NEUTRAL': 0.838} f1={'BIAS_UP': 0.525, 'BIAS_DOWN': 0.488, 'BIAS_NEUTRAL': 0.859} confusion=[[404, 0, 385], [0, 223, 109], [346, 358, 3638]] score_mae={'bias_up_score': 0.2049, 'bias_down_score': 0.1334} pred_share={'BIAS_UP': 0.1373, 'BIAS_DOWN': 0.1064, 'BIAS_NEUTRAL': 0.7564}
2026-05-11 18:25:36,120 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.539, 'BIAS_DOWN': 0.384, 'BIAS_NEUTRAL': 0.88} min_precision=0.500 recall={'BIAS_UP': 0.512, 'BIAS_DOWN': 0.672, 'BIAS_NEUTRAL': 0.838} min_recall=0.150 f1={'BIAS_UP': 0.525, 'BIAS_DOWN': 0.488, 'BIAS_NEUTRAL': 0.859} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_DOWN'] weak_recall=[] weak_f1=[] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-11 18:25:36,123 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 18:25:36,124 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 18:25:36,124 INFO Regime phase HTF train fold=train_all: 17.5s
2026-05-11 18:25:36,241 INFO Regime HTF complete fold=train_all: acc=0.781 bal=0.674 train=68826 val=8737 per_class={'BIAS_UP': 0.512, 'BIAS_DOWN': 0.672, 'BIAS_NEUTRAL': 0.838} precision={'BIAS_UP': 0.539, 'BIAS_DOWN': 0.384, 'BIAS_NEUTRAL': 0.88} threshold=0.400 margin=0.150
2026-05-11 18:25:36,242 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:25:36,445 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 502, 'BIAS_DOWN': 720, 'BIAS_NEUTRAL': 10180}  ambiguous=4113 (total=11402) horizon=84
2026-05-11 18:25:36,448 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on GBPUSD 4H:
{'BIAS_UP': 4.482142857142857, 'BIAS_DOWN': 5.669291338582677, 'BIAS_NEUTRAL': 42.416666666666664}
2026-05-11 18:25:36,452 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (all labels):
{'BIAS_UP': {'n': 502, 'mean': 0.00043281055222645396, 'mean_over_std': 0.19321420748230153}, 'BIAS_DOWN': {'n': 720, 'mean': -0.00046308823082979826, 'mean_over_std': -0.19538165843530478}, 'BIAS_NEUTRAL': {'n': 10179, 'mean': 7.477567618138561e-07, 'mean_over_std': 0.0002829536380249001}}
2026-05-11 18:25:36,452 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 502, 'mean': 0.00043281055222645396, 'mean_over_std': 0.19321420748230153}, 'BIAS_DOWN': {'n': 720, 'mean': -0.00046308823082979826, 'mean_over_std': -0.19538165843530478}, 'BIAS_NEUTRAL': {'n': 6067, 'mean': 9.596616495197703e-06, 'mean_over_std': 0.004013656697571348}}
2026-05-11 18:25:36,457 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-11 18:25:36,459 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:25:36,461 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:25:36,463 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:25:36,465 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:25:36,467 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:25:36,469 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:25:36,486 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:25:36,494 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:25:36,497 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:25:36,498 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:25:36,498 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:25:36,503 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:25:37,672 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected XAUUSD — 44227 samples (group=gold) score_means={'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}
2026-05-11 18:25:37,793 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:37,795 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:37,796 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:37,796 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:37,797 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:37,800 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:25:38,871 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURUSD — 43689 samples (group=dollar) score_means={'trend_score': 0.49, 'range_score': 0.236, 'chop_score': 0.4622, 'volatility_percentile': 0.3812, 'consolidation_score': 0.1862}
2026-05-11 18:25:38,992 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:38,994 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:38,995 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:38,996 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:38,996 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:38,999 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:25:40,079 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected USDJPY — 43686 samples (group=dollar) score_means={'trend_score': 0.4949, 'range_score': 0.2312, 'chop_score': 0.4613, 'volatility_percentile': 0.38, 'consolidation_score': 0.1931}
2026-05-11 18:25:40,196 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:40,199 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:40,200 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:40,200 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:40,201 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:40,203 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:25:41,281 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURJPY — 43688 samples (group=cross) score_means={'trend_score': 0.4888, 'range_score': 0.2345, 'chop_score': 0.4662, 'volatility_percentile': 0.3831, 'consolidation_score': 0.1882}
2026-05-11 18:25:41,398 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:41,401 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:41,402 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:41,402 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:41,403 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:41,405 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:25:42,474 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPJPY — 43678 samples (group=cross) score_means={'trend_score': 0.4878, 'range_score': 0.2348, 'chop_score': 0.4673, 'volatility_percentile': 0.3839, 'consolidation_score': 0.1904}
2026-05-11 18:25:42,594 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:42,596 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:42,597 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:42,598 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:42,598 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:42,601 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:25:43,678 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPUSD — 43676 samples (group=dollar) score_means={'trend_score': 0.4923, 'range_score': 0.2335, 'chop_score': 0.4611, 'volatility_percentile': 0.3832, 'consolidation_score': 0.1856}
2026-05-11 18:25:43,804 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2346, 'chop_score': 0.4667, 'volatility_percentile': 0.3835, 'consolidation_score': 0.1893}, 'dollar': {'trend_score': 0.4924, 'range_score': 0.2335, 'chop_score': 0.4615, 'volatility_percentile': 0.3815, 'consolidation_score': 0.1883}, 'gold': {'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}}
2026-05-11 18:25:43,804 INFO Regime[1H mode=ltf_behaviour] score means by year: {2016: {'trend_score': 0.495, 'range_score': 0.233, 'chop_score': 0.4607, 'volatility_percentile': 0.3776, 'consolidation_score': 0.192}, 2017: {'trend_score': 0.5017, 'range_score': 0.2328, 'chop_score': 0.4574, 'volatility_percentile': 0.3689, 'consolidation_score': 0.1942}, 2018: {'trend_score': 0.4886, 'range_score': 0.237, 'chop_score': 0.4655, 'volatility_percentile': 0.3947, 'consolidation_score': 0.1785}, 2019: {'trend_score': 0.4817, 'range_score': 0.236, 'chop_score': 0.4691, 'volatility_percentile': 0.3725, 'consolidation_score': 0.198}, 2020: {'trend_score': 0.4874, 'range_score': 0.2311, 'chop_score': 0.4674, 'volatility_percentile': 0.3848, 'consolidation_score': 0.1919}, 2021: {'trend_score': 0.4872, 'range_score': 0.2372, 'chop_score': 0.4672, 'volatility_percentile': 0.377, 'consolidation_score': 0.192}, 2022: {'trend_score': 0.4893, 'range_score': 0.2357, 'chop_score': 0.4647, 'volatility_percentile': 0.3962, 'consolidation_score': 0.1784}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 18:25:43,908 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
2026-05-11 18:25:43,910 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 18:25:43,911 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
2026-05-11 18:25:43,912 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 18:25:43,914 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
2026-05-11 18:25:43,915 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
2026-05-11 18:25:43,925 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:25:43,928 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:25:43,929 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:25:43,930 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:25:43,930 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:25:43,932 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:25:44,289 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected XAUUSD — 5117 samples (group=gold) score_means={'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}
2026-05-11 18:25:44,406 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:44,409 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:44,410 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:44,410 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:44,411 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:44,412 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:25:44,782 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURUSD — 5053 samples (group=dollar) score_means={'trend_score': 0.4883, 'range_score': 0.2413, 'chop_score': 0.463, 'volatility_percentile': 0.3837, 'consolidation_score': 0.1793}
2026-05-11 18:25:44,908 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:44,911 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:44,912 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:44,912 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:44,913 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:44,914 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:25:45,246 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected USDJPY — 5046 samples (group=dollar) score_means={'trend_score': 0.5085, 'range_score': 0.2259, 'chop_score': 0.4506, 'volatility_percentile': 0.3809, 'consolidation_score': 0.1888}
2026-05-11 18:25:45,359 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:45,361 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:45,362 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:45,362 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:45,363 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:45,364 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:25:45,700 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURJPY — 5053 samples (group=cross) score_means={'trend_score': 0.5023, 'range_score': 0.2261, 'chop_score': 0.4524, 'volatility_percentile': 0.3837, 'consolidation_score': 0.18}
2026-05-11 18:25:45,809 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:45,811 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:45,812 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:45,812 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:45,812 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:45,814 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:25:46,141 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPJPY — 5043 samples (group=cross) score_means={'trend_score': 0.4743, 'range_score': 0.2372, 'chop_score': 0.4716, 'volatility_percentile': 0.3777, 'consolidation_score': 0.1917}
2026-05-11 18:25:46,260 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:46,262 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:46,263 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:46,263 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:46,264 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:25:46,265 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:25:46,589 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPUSD — 5040 samples (group=dollar) score_means={'trend_score': 0.4843, 'range_score': 0.2371, 'chop_score': 0.4614, 'volatility_percentile': 0.377, 'consolidation_score': 0.1835}
2026-05-11 18:25:46,697 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2317, 'chop_score': 0.462, 'volatility_percentile': 0.3807, 'consolidation_score': 0.1858}, 'dollar': {'trend_score': 0.4937, 'range_score': 0.2348, 'chop_score': 0.4583, 'volatility_percentile': 0.3806, 'consolidation_score': 0.1839}, 'gold': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}}
2026-05-11 18:25:46,697 INFO Regime[1H mode=ltf_behaviour] score means by year: {2022: {'trend_score': 0.4905, 'range_score': 0.2364, 'chop_score': 0.4625, 'volatility_percentile': 0.385, 'consolidation_score': 0.1797}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 18:25:46,789 INFO Regime phase LTF dataset build fold=train_all: 10.3s (train=262644 val=30352)
2026-05-11 18:25:46,789 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260511_182546
2026-05-11 18:25:46,795 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=53, n_classes=5)
2026-05-11 18:25:46,795 INFO Regime 1H/ltf_behaviour warm start enabled from existing weights
2026-05-11 18:25:46,829 INFO RegimeClassifier[mode=ltf_behaviour]: warm start score head
2026-05-11 18:25:46,829 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-11 18:25:47,370 INFO Regime score epoch  1/50 — tr=0.0032 va=0.0007 mae={'trend_score': 0.0158, 'range_score': 0.0302, 'chop_score': 0.0178, 'volatility_percentile': 0.0136, 'consolidation_score': 0.0187}
2026-05-11 18:25:47,896 INFO Regime score epoch  2/50 — tr=0.0031 va=0.0007
2026-05-11 18:25:48,441 INFO Regime score epoch  3/50 — tr=0.0031 va=0.0007
2026-05-11 18:25:48,970 INFO Regime score epoch  4/50 — tr=0.0031 va=0.0007
2026-05-11 18:25:49,514 INFO Regime score epoch  5/50 — tr=0.0031 va=0.0007 mae={'trend_score': 0.0161, 'range_score': 0.03, 'chop_score': 0.018, 'volatility_percentile': 0.013, 'consolidation_score': 0.0188}
2026-05-11 18:25:50,037 INFO Regime score epoch  6/50 — tr=0.0031 va=0.0007
2026-05-11 18:25:50,560 INFO Regime score epoch  7/50 — tr=0.0031 va=0.0007
2026-05-11 18:25:51,079 INFO Regime score epoch  8/50 — tr=0.0031 va=0.0007
2026-05-11 18:25:51,605 INFO Regime score epoch  9/50 — tr=0.0031 va=0.0007
2026-05-11 18:25:52,137 INFO Regime score epoch 10/50 — tr=0.0031 va=0.0007 mae={'trend_score': 0.016, 'range_score': 0.0298, 'chop_score': 0.0175, 'volatility_percentile': 0.0133, 'consolidation_score': 0.0184}
2026-05-11 18:25:52,661 INFO Regime score epoch 11/50 — tr=0.0031 va=0.0007
2026-05-11 18:25:53,184 INFO Regime score epoch 12/50 — tr=0.0031 va=0.0007
2026-05-11 18:25:53,695 INFO Regime score epoch 13/50 — tr=0.0031 va=0.0007
2026-05-11 18:25:54,204 INFO Regime score epoch 14/50 — tr=0.0031 va=0.0007
2026-05-11 18:25:54,772 INFO Regime score epoch 15/50 — tr=0.0031 va=0.0007 mae={'trend_score': 0.0155, 'range_score': 0.03, 'chop_score': 0.0178, 'volatility_percentile': 0.0128, 'consolidation_score': 0.0184}
2026-05-11 18:25:55,319 INFO Regime score epoch 16/50 — tr=0.0031 va=0.0007
2026-05-11 18:25:55,860 INFO Regime score epoch 17/50 — tr=0.0031 va=0.0007
2026-05-11 18:25:56,382 INFO Regime score epoch 18/50 — tr=0.0031 va=0.0007
2026-05-11 18:25:56,922 INFO Regime score epoch 19/50 — tr=0.0031 va=0.0007
2026-05-11 18:25:57,466 INFO Regime score epoch 20/50 — tr=0.0031 va=0.0007 mae={'trend_score': 0.0153, 'range_score': 0.0295, 'chop_score': 0.0175, 'volatility_percentile': 0.0125, 'consolidation_score': 0.0183}
2026-05-11 18:25:58,006 INFO Regime score epoch 21/50 — tr=0.0031 va=0.0007
2026-05-11 18:25:58,538 INFO Regime score epoch 22/50 — tr=0.0030 va=0.0007
2026-05-11 18:25:59,053 INFO Regime score epoch 23/50 — tr=0.0030 va=0.0007
2026-05-11 18:25:59,578 INFO Regime score epoch 24/50 — tr=0.0030 va=0.0007
2026-05-11 18:26:00,098 INFO Regime score epoch 25/50 — tr=0.0030 va=0.0007 mae={'trend_score': 0.0151, 'range_score': 0.0292, 'chop_score': 0.0172, 'volatility_percentile': 0.013, 'consolidation_score': 0.018}
2026-05-11 18:26:00,662 INFO Regime score epoch 26/50 — tr=0.0030 va=0.0007
2026-05-11 18:26:01,213 INFO Regime score epoch 27/50 — tr=0.0030 va=0.0007
2026-05-11 18:26:01,747 INFO Regime score epoch 28/50 — tr=0.0030 va=0.0007
2026-05-11 18:26:02,273 INFO Regime score epoch 29/50 — tr=0.0030 va=0.0007
2026-05-11 18:26:02,807 INFO Regime score epoch 30/50 — tr=0.0030 va=0.0007 mae={'trend_score': 0.0151, 'range_score': 0.0292, 'chop_score': 0.0173, 'volatility_percentile': 0.0124, 'consolidation_score': 0.0179}
2026-05-11 18:26:03,336 INFO Regime score epoch 31/50 — tr=0.0030 va=0.0007
2026-05-11 18:26:03,860 INFO Regime score epoch 32/50 — tr=0.0030 va=0.0007
2026-05-11 18:26:04,388 INFO Regime score epoch 33/50 — tr=0.0030 va=0.0007
2026-05-11 18:26:04,942 INFO Regime score epoch 34/50 — tr=0.0030 va=0.0007
2026-05-11 18:26:05,500 INFO Regime score epoch 35/50 — tr=0.0030 va=0.0007 mae={'trend_score': 0.0156, 'range_score': 0.0292, 'chop_score': 0.0176, 'volatility_percentile': 0.0124, 'consolidation_score': 0.0178}
2026-05-11 18:26:06,034 INFO Regime score epoch 36/50 — tr=0.0030 va=0.0007
2026-05-11 18:26:06,583 INFO Regime score epoch 37/50 — tr=0.0030 va=0.0007
2026-05-11 18:26:07,114 INFO Regime score epoch 38/50 — tr=0.0030 va=0.0007
2026-05-11 18:26:07,654 INFO Regime score epoch 39/50 — tr=0.0030 va=0.0007
2026-05-11 18:26:08,196 INFO Regime score epoch 40/50 — tr=0.0030 va=0.0007 mae={'trend_score': 0.015, 'range_score': 0.0292, 'chop_score': 0.0174, 'volatility_percentile': 0.0121, 'consolidation_score': 0.0179}
2026-05-11 18:26:08,197 INFO Regime score early stop at epoch 40
2026-05-11 18:26:08,218 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0149, 'range_score': 0.0291, 'chop_score': 0.0171, 'volatility_percentile': 0.0128, 'consolidation_score': 0.0176} mse={'trend_score': 0.00039, 'range_score': 0.00143, 'chop_score': 0.00048, 'volatility_percentile': 0.00033, 'consolidation_score': 0.00071} corr={'trend_score': 0.996, 'range_score': 0.9662, 'chop_score': 0.9936, 'volatility_percentile': 0.9969, 'consolidation_score': 0.9926} pred_std={'trend_score': 0.2211, 'range_score': 0.1309, 'chop_score': 0.1824, 'volatility_percentile': 0.2191, 'consolidation_score': 0.2143} target_std={'trend_score': 0.2219, 'range_score': 0.1433, 'chop_score': 0.1888, 'volatility_percentile': 0.2165, 'consolidation_score': 0.2179}
2026-05-11 18:26:08,556 INFO RegimeClassifier[mode=ltf_behaviour] train group diagnostics={'dollar': {'n': 16632, 'score_mae': {'trend_score': 0.0145, 'range_score': 0.0291, 'chop_score': 0.0171, 'volatility_percentile': 0.0125, 'consolidation_score': 0.0179}, 'target_score_mean': {'trend_score': 0.491, 'range_score': 0.2347, 'chop_score': 0.4613, 'volatility_percentile': 0.3824, 'consolidation_score': 0.186}, 'pred_score_mean': {'trend_score': 0.4916, 'range_score': 0.2328, 'chop_score': 0.4603, 'volatility_percentile': 0.377, 'consolidation_score': 0.1847}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3578, 53, 0, 3, 0, 0, 145], [5, 94, 0, 0, 0, 6, 5], [0, 0, 182, 14, 48, 0, 216], [2, 0, 2, 581, 27, 0, 77], [0, 0, 21, 17, 3114, 1, 163], [0, 16, 0, 0, 7, 71, 34], [119, 12, 41, 50, 53, 6, 7869]]}, 'gold': {'n': 8420, 'score_mae': {'trend_score': 0.0145, 'range_score': 0.0296, 'chop_score': 0.0172, 'volatility_percentile': 0.013, 'consolidation_score': 0.0183}, 'target_score_mean': {'trend_score': 0.4883, 'range_score': 0.2355, 'chop_score': 0.4652, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}, 'pred_score_mean': {'trend_score': 0.4893, 'range_score': 0.2338, 'chop_score': 0.4641, 'volatility_percentile': 0.3711, 'consolidation_score': 0.1905}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1787, 30, 0, 0, 0, 0, 68], [4, 49, 0, 0, 0, 1, 2], [0, 0, 95, 12, 24, 0, 113], [2, 0, 2, 359, 14, 0, 39], [0, 0, 14, 18, 1598, 0, 74], [0, 12, 0, 0, 4, 47, 18], [56, 3, 29, 15, 41, 0, 3890]]}, 'jpy': {'n': 24948, 'score_mae': {'trend_score': 0.0144, 'range_score': 0.0291, 'chop_score': 0.0169, 'volatility_percentile': 0.0133, 'consolidation_score': 0.018}, 'target_score_mean': {'trend_score': 0.4904, 'range_score': 0.2338, 'chop_score': 0.4653, 'volatility_percentile': 0.3826, 'consolidation_score': 0.1905}, 'pred_score_mean': {'trend_score': 0.4906, 'range_score': 0.232, 'chop_score': 0.464, 'volatility_percentile': 0.3768, 'consolidation_score': 0.1884}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[5326, 109, 0, 2, 0, 0, 209], [8, 167, 0, 0, 0, 5, 7], [0, 0, 234, 20, 81, 0, 312], [3, 0, 2, 1113, 68, 0, 128], [0, 0, 30, 44, 4797, 0, 244], [0, 28, 0, 0, 14, 105, 76], [166, 9, 66, 81, 116, 9, 11369]]}}
2026-05-11 18:26:08,756 INFO RegimeClassifier[mode=ltf_behaviour] validation group diagnostics={'dollar': {'n': 10093, 'score_mae': {'trend_score': 0.0152, 'range_score': 0.0295, 'chop_score': 0.0174, 'volatility_percentile': 0.0123, 'consolidation_score': 0.0172}, 'target_score_mean': {'trend_score': 0.4863, 'range_score': 0.2392, 'chop_score': 0.4622, 'volatility_percentile': 0.3804, 'consolidation_score': 0.1814}, 'pred_score_mean': {'trend_score': 0.4871, 'range_score': 0.2356, 'chop_score': 0.4622, 'volatility_percentile': 0.3748, 'consolidation_score': 0.1805}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[2299, 21, 0, 1, 0, 0, 91], [4, 46, 0, 0, 0, 3, 0], [0, 0, 103, 8, 42, 0, 163], [0, 0, 0, 351, 24, 0, 48], [0, 0, 13, 16, 1931, 0, 90], [0, 9, 0, 0, 3, 39, 26], [48, 6, 23, 43, 40, 5, 4597]]}, 'gold': {'n': 5117, 'score_mae': {'trend_score': 0.0141, 'range_score': 0.0284, 'chop_score': 0.0172, 'volatility_percentile': 0.0127, 'consolidation_score': 0.0182}, 'target_score_mean': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}, 'pred_score_mean': {'trend_score': 0.4974, 'range_score': 0.23, 'chop_score': 0.4564, 'volatility_percentile': 0.3754, 'consolidation_score': 0.1819}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1114, 11, 0, 0, 0, 0, 42], [3, 29, 0, 0, 0, 2, 1], [0, 0, 61, 3, 14, 0, 93], [0, 0, 2, 225, 8, 0, 20], [0, 0, 4, 10, 824, 0, 49], [0, 6, 0, 0, 3, 28, 13], [49, 2, 19, 22, 21, 1, 2438]]}, 'jpy': {'n': 15142, 'score_mae': {'trend_score': 0.015, 'range_score': 0.0292, 'chop_score': 0.017, 'volatility_percentile': 0.0132, 'consolidation_score': 0.0177}, 'target_score_mean': {'trend_score': 0.495, 'range_score': 0.2297, 'chop_score': 0.4582, 'volatility_percentile': 0.3808, 'consolidation_score': 0.1868}, 'pred_score_mean': {'trend_score': 0.4947, 'range_score': 0.2272, 'chop_score': 0.4581, 'volatility_percentile': 0.3752, 'consolidation_score': 0.1849}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3339, 46, 0, 1, 0, 0, 132], [6, 96, 0, 0, 0, 7, 6], [0, 0, 136, 15, 46, 0, 187], [3, 0, 2, 715, 32, 0, 75], [0, 0, 21, 27, 2626, 0, 143], [0, 14, 0, 0, 9, 63, 36], [95, 8, 39, 42, 65, 9, 7101]]}}
2026-05-11 18:26:08,763 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-11 18:26:08,764 INFO Regime phase LTF train fold=train_all: 22.0s
2026-05-11 18:26:08,882 INFO Regime LTF complete fold=train_all: score_accuracy=0.982, train=262644 val=30352 mae={'trend_score': 0.0149, 'range_score': 0.0291, 'chop_score': 0.0171, 'volatility_percentile': 0.0128, 'consolidation_score': 0.0176}
2026-05-11 18:26:08,884 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:26:09,289 INFO Regime[1H mode=ltf_behaviour fold=latest] LTF score diagnostics on GBPUSD:
{'trend_score': {'mean': 0.4921, 'q10': 0.1918, 'q50': 0.4877, 'q90': 0.8002}, 'range_score': {'mean': 0.2337, 'q10': 0.0513, 'q50': 0.2125, 'q90': 0.4349}, 'chop_score': {'mean': 0.4612, 'q10': 0.2148, 'q50': 0.4503, 'q90': 0.7255}, 'volatility_percentile': {'mean': 0.3832, 'q10': 0.1018, 'q50': 0.3704, 'q90': 0.6823}, 'consolidation_score': {'mean': 0.1854, 'q10': 0.0, 'q50': 0.1185, 'q90': 0.506}}
2026-05-11 18:26:09,294 INFO Regime retrain total: 63.4s (370559 train+val samples)
2026-05-11 18:26:09,299 INFO Retrain complete. Total wall-clock: 63.4s
  DONE  Retrain regime [pre-R2 retrain]

=== Round 2: BLIND backtest on test window (unseen 2yr) ===
  START Round 2 - Blind backtest (test)
2026-05-11 18:26:10,867 INFO === STEP 6: BACKTEST (round2) ===
2026-05-11 18:26:10,869 INFO BT_WINDOW=round2 — BLIND backtest: 2023-08-07 → 2025-08-05 (test set)
2026-05-11 18:26:10,869 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-05-11 18:26:10,869 INFO Round 2 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
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
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
2026-05-11 18:27:36,239 WARNING _build_sequence_df: HTF frame 5M filled 317 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 18:27:36,476 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 18:27:36,681 WARNING _build_sequence_df: HTF frame 5M filled 317 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 18:27:36,727 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 18:27:36,917 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 18:27:36,965 WARNING _build_sequence_df: HTF frame 5M filled 341 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 18:27:36,987 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 18:27:37,050 WARNING _build_sequence_df: HTF frame 5M filled 341 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
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
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
2026-05-11 18:27:48,621 WARNING _build_sequence_df: HTF frame 5M filled 317 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 18:27:48,643 WARNING _build_sequence_df: HTF frame 5M filled 317 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 18:27:48,664 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 18:27:48,712 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 18:28:46,302 INFO Round 2 backtest — 0 trades | avg WR=0.0% | avg PF=0.00 | avg Sharpe=0.00
2026-05-11 18:28:46,302 INFO   ml_trader: 0 trades | WR=0.0% | fixed PF=0.00 | Return=0.0% | ExpR=0.000 | DD=0.0% | Sharpe=0.00
2026-05-11 18:28:46,302 INFO   ml_trader gate_diagnostics: bars=280782 no_signal=206025 quality_block=115 session_skip=74641 density=1 pm_reject=0
2026-05-11 18:28:46,302 INFO   ml_trader no_signal_reasons: {'no_trade_chop': 25903, 'no_trade_uncertain': 78844, 'gru_expected_r_below_threshold': 10253, 'trend_structure_missing': 4615, 'weak_gru_direction': 48526, 'wait_pullback': 11025, 'tradeability_direction_conflict': 2994, 'htf_low_regime_confidence': 4245, 'no_trade_extreme_vol': 19611, 'expected_r_below_threshold': 9}
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 0
ERROR  trade_log is empty.
2026-05-11 18:28:46,518 WARNING Round 2: trade_log is empty — nothing to journal
2026-05-11 18:28:46,519 WARNING Round 2: no trades to journal
  DONE  Round 2 - Blind backtest (test)
  Saved Round 2 result → round2_summary.json
  Journal after Round 2: 115 entries

  SKIP  Round 2 Quality+RL retrain — blind test journal kept untouched

=== Round 3: Incremental retrain ===
  START Retrain gru [train-split retrain]
2026-05-11 18:28:46,964 INFO retrain environment: KAGGLE
2026-05-11 18:28:48,653 INFO Device: CUDA (2 GPU(s))
2026-05-11 18:28:48,665 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 18:28:48,665 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 18:28:48,665 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 18:28:48,669 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 18:28:48,669 INFO Retrain data split: train
2026-05-11 18:28:48,669 INFO Retrain rolling fold selector: latest
2026-05-11 18:28:48,670 INFO === GRU-LSTM retrain (timeframes: ['15M']) ===
2026-05-11 18:28:48,821 INFO NumExpr defaulting to 4 threads.
2026-05-11 18:28:49,026 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-11 18:28:49,026 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 18:28:49,026 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 18:28:49,278 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-05-11 18:28:49,279 INFO GRU multi-symbol training (Kaggle mode): 6 symbols × ['15M']
2026-05-11 18:28:49,282 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260511_182849
2026-05-11 18:28:49,286 INFO GRU feature contract unchanged (input_size=94) — incremental retrain
2026-05-11 18:28:49,287 INFO Deleted stale GRU artifact for cold start: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:28:49,287 INFO GRU warm start disabled by default; set GRU_ALLOW_WARM_START=1 to reuse compatible weights
2026-05-11 18:28:49,568 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:28:49,599 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:28:49,617 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:28:49,628 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:28:49,712 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 18:28:49,719 INFO Loaded XAUUSD/15M split=train fold=latest: 176438 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:28:50,342 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:28:50,363 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:28:50,390 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:28:50,398 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:28:50,448 INFO Loaded EURUSD/15M split=train fold=latest: 174951 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:28:51,056 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:28:51,079 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:28:51,095 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:28:51,104 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:28:51,146 INFO Loaded USDJPY/15M split=train fold=latest: 174944 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:28:51,734 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:28:51,757 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:28:51,773 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:28:51,782 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:28:51,824 INFO Loaded EURJPY/15M split=train fold=latest: 174948 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:28:52,390 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:28:52,413 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:28:52,432 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:28:52,441 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:28:52,486 INFO Loaded GBPJPY/15M split=train fold=latest: 174902 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:28:53,078 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:28:53,100 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:28:53,115 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:28:53,123 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:28:53,164 INFO Loaded GBPUSD/15M split=train fold=latest: 174899 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:28:53,627 INFO train_multi: 6 segments, ~971854 total bars
2026-05-11 18:28:53,635 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-05-11 18:28:53,635 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-05-11 18:28:53,636 INFO train_multi: training ALL 6 segments across TFs ['15M'] in one combined pass
2026-05-11 18:28:53,636 INFO train_multi: building combined dataset for TF=ALL (6 segments)
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
2026-05-11 18:29:06,708 INFO train_multi TF=ALL: 971494 sequences across 6 segments
2026-05-11 18:29:06,708 INFO train_multi TF=ALL: estimated peak RAM = 27072 MB (train=419996 calib=60000 val=120002 n_feat=94 seq_len=60)
2026-05-11 18:29:06,708 WARNING train_multi TF=ALL: trimming to fit RAM budget — new train=310283 calib=44326 val=88652 (20000 MB est)
2026-05-11 18:29:09,100 INFO train_multi TF=ALL: train=310283 calib=44326 val=88652 (10007 MB tensors)
2026-05-11 18:29:16,031 INFO train_multi TF=ALL: structural bar weighting — 199279 structural bars (64.2%) weight=15.0 structural_only=0
2026-05-11 18:29:17,073 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=25, min_epochs=45)
2026-05-11 18:29:32,821 INFO train_multi TF=ALL epoch 1/100 train=2.3472 val=2.3490 r_mae=0.980 pos_r_acc=0.455 side_acc=0.510 r_n=127469
2026-05-11 18:29:32,827 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:29:32,827 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:29:32,827 INFO train_multi TF=ALL: new best val=2.3490 r_mae=0.9801 — saved
2026-05-11 18:29:32,832 INFO train_multi TF=ALL: new best r_mae=0.9801 — saved rmae checkpoint
2026-05-11 18:29:46,497 INFO train_multi TF=ALL epoch 2/100 train=2.3433 val=2.3456 r_mae=0.977 pos_r_acc=0.455 side_acc=0.510 r_n=127469
2026-05-11 18:29:46,503 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:29:46,503 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:29:46,503 INFO train_multi TF=ALL: new best val=2.3456 r_mae=0.9773 — saved
2026-05-11 18:29:46,508 INFO train_multi TF=ALL: new best r_mae=0.9773 — saved rmae checkpoint
2026-05-11 18:30:00,319 INFO train_multi TF=ALL epoch 3/100 train=2.3398 val=2.3414 r_mae=0.974 pos_r_acc=0.493 side_acc=0.486 r_n=127469
2026-05-11 18:30:00,324 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:30:00,324 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:30:00,325 INFO train_multi TF=ALL: new best val=2.3414 r_mae=0.9738 — saved
2026-05-11 18:30:00,329 INFO train_multi TF=ALL: new best r_mae=0.9738 — saved rmae checkpoint
2026-05-11 18:30:13,948 INFO train_multi TF=ALL epoch 4/100 train=2.3362 val=2.3359 r_mae=0.968 pos_r_acc=0.545 side_acc=0.490 r_n=127469
2026-05-11 18:30:13,954 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:30:13,954 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:30:13,955 INFO train_multi TF=ALL: new best val=2.3359 r_mae=0.9682 — saved
2026-05-11 18:30:13,959 INFO train_multi TF=ALL: new best r_mae=0.9682 — saved rmae checkpoint
2026-05-11 18:30:27,412 INFO train_multi TF=ALL epoch 5/100 train=2.3341 val=2.3338 r_mae=0.967 pos_r_acc=0.545 side_acc=0.490 r_n=127469
2026-05-11 18:30:27,417 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:30:27,418 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:30:27,418 INFO train_multi TF=ALL: new best val=2.3338 r_mae=0.9666 — saved
2026-05-11 18:30:27,422 INFO train_multi TF=ALL: new best r_mae=0.9666 — saved rmae checkpoint
2026-05-11 18:30:40,991 INFO train_multi TF=ALL epoch 6/100 train=2.3331 val=2.3333 r_mae=0.966 pos_r_acc=0.545 side_acc=0.491 r_n=127469
2026-05-11 18:30:40,996 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:30:40,996 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:30:40,997 INFO train_multi TF=ALL: new best val=2.3333 r_mae=0.9663 — saved
2026-05-11 18:30:41,001 INFO train_multi TF=ALL: new best r_mae=0.9663 — saved rmae checkpoint
2026-05-11 18:30:54,479 INFO train_multi TF=ALL epoch 7/100 train=2.3325 val=2.3328 r_mae=0.966 pos_r_acc=0.545 side_acc=0.494 r_n=127469
2026-05-11 18:30:54,486 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:30:54,486 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:30:54,486 INFO train_multi TF=ALL: new best val=2.3328 r_mae=0.9662 — saved
2026-05-11 18:30:54,491 INFO train_multi TF=ALL: new best r_mae=0.9662 — saved rmae checkpoint
2026-05-11 18:31:08,243 INFO train_multi TF=ALL epoch 8/100 train=2.3325 val=2.3319 r_mae=0.966 pos_r_acc=0.545 side_acc=0.498 r_n=127469
2026-05-11 18:31:08,248 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:31:08,248 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:31:08,248 INFO train_multi TF=ALL: new best val=2.3319 r_mae=0.9660 — saved
2026-05-11 18:31:08,257 INFO train_multi TF=ALL: new best r_mae=0.9660 — saved rmae checkpoint
2026-05-11 18:31:22,031 INFO train_multi TF=ALL epoch 9/100 train=2.3314 val=2.3302 r_mae=0.966 pos_r_acc=0.545 side_acc=0.510 r_n=127469
2026-05-11 18:31:22,037 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:31:22,037 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:31:22,037 INFO train_multi TF=ALL: new best val=2.3302 r_mae=0.9659 — saved
2026-05-11 18:31:22,042 INFO train_multi TF=ALL: new best r_mae=0.9659 — saved rmae checkpoint
2026-05-11 18:31:35,791 INFO train_multi TF=ALL epoch 10/100 train=2.3308 val=2.3290 r_mae=0.967 pos_r_acc=0.545 side_acc=0.505 r_n=127469
2026-05-11 18:31:35,803 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:31:35,803 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:31:35,803 INFO train_multi TF=ALL: new best val=2.3290 r_mae=0.9666 — saved
2026-05-11 18:31:49,490 INFO train_multi TF=ALL epoch 11/100 train=2.3307 val=2.3289 r_mae=0.967 pos_r_acc=0.545 side_acc=0.503 r_n=127469
2026-05-11 18:31:49,495 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:31:49,495 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:31:49,496 INFO train_multi TF=ALL: new best val=2.3289 r_mae=0.9668 — saved
2026-05-11 18:32:03,296 INFO train_multi TF=ALL epoch 12/100 train=2.3300 val=2.3287 r_mae=0.967 pos_r_acc=0.545 side_acc=0.506 r_n=127469
2026-05-11 18:32:03,302 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:32:03,302 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:32:03,302 INFO train_multi TF=ALL: new best val=2.3287 r_mae=0.9672 — saved
2026-05-11 18:32:17,392 INFO train_multi TF=ALL epoch 13/100 train=2.3290 val=2.3278 r_mae=0.967 pos_r_acc=0.545 side_acc=0.513 r_n=127469
2026-05-11 18:32:17,398 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:32:17,398 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:32:17,398 INFO train_multi TF=ALL: new best val=2.3278 r_mae=0.9671 — saved
2026-05-11 18:32:31,259 INFO train_multi TF=ALL epoch 14/100 train=2.3284 val=2.3271 r_mae=0.967 pos_r_acc=0.545 side_acc=0.515 r_n=127469
2026-05-11 18:32:31,264 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:32:31,264 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:32:31,264 INFO train_multi TF=ALL: new best val=2.3271 r_mae=0.9668 — saved
2026-05-11 18:32:45,554 INFO train_multi TF=ALL epoch 15/100 train=2.3264 val=2.3250 r_mae=0.965 pos_r_acc=0.545 side_acc=0.523 r_n=127469
2026-05-11 18:32:45,560 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:32:45,560 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:32:45,560 INFO train_multi TF=ALL: new best val=2.3250 r_mae=0.9653 — saved
2026-05-11 18:32:45,564 INFO train_multi TF=ALL: new best r_mae=0.9653 — saved rmae checkpoint
2026-05-11 18:32:59,595 INFO train_multi TF=ALL epoch 16/100 train=2.3232 val=2.3224 r_mae=0.964 pos_r_acc=0.544 side_acc=0.525 r_n=127469
2026-05-11 18:32:59,608 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:32:59,608 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:32:59,608 INFO train_multi TF=ALL: new best val=2.3224 r_mae=0.9645 — saved
2026-05-11 18:32:59,613 INFO train_multi TF=ALL: new best r_mae=0.9645 — saved rmae checkpoint
2026-05-11 18:33:13,589 INFO train_multi TF=ALL epoch 17/100 train=2.3208 val=2.3209 r_mae=0.964 pos_r_acc=0.546 side_acc=0.534 r_n=127469
2026-05-11 18:33:13,595 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:33:13,595 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:33:13,595 INFO train_multi TF=ALL: new best val=2.3209 r_mae=0.9635 — saved
2026-05-11 18:33:13,600 INFO train_multi TF=ALL: new best r_mae=0.9635 — saved rmae checkpoint
2026-05-11 18:33:27,549 INFO train_multi TF=ALL epoch 18/100 train=2.3178 val=2.3183 r_mae=0.962 pos_r_acc=0.546 side_acc=0.537 r_n=127469
2026-05-11 18:33:27,555 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:33:27,555 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:33:27,555 INFO train_multi TF=ALL: new best val=2.3183 r_mae=0.9624 — saved
2026-05-11 18:33:27,560 INFO train_multi TF=ALL: new best r_mae=0.9624 — saved rmae checkpoint
2026-05-11 18:33:41,395 INFO train_multi TF=ALL epoch 19/100 train=2.3149 val=2.3180 r_mae=0.962 pos_r_acc=0.547 side_acc=0.535 r_n=127469
2026-05-11 18:33:41,401 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:33:41,401 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:33:41,401 INFO train_multi TF=ALL: new best val=2.3180 r_mae=0.9617 — saved
2026-05-11 18:33:41,411 INFO train_multi TF=ALL: new best r_mae=0.9617 — saved rmae checkpoint
2026-05-11 18:33:55,544 INFO train_multi TF=ALL epoch 20/100 train=2.3116 val=2.3160 r_mae=0.960 pos_r_acc=0.547 side_acc=0.535 r_n=127469
2026-05-11 18:33:55,549 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:33:55,549 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:33:55,549 INFO train_multi TF=ALL: new best val=2.3160 r_mae=0.9600 — saved
2026-05-11 18:33:55,554 INFO train_multi TF=ALL: new best r_mae=0.9600 — saved rmae checkpoint
2026-05-11 18:34:09,711 INFO train_multi TF=ALL epoch 21/100 train=2.3079 val=2.3153 r_mae=0.960 pos_r_acc=0.549 side_acc=0.534 r_n=127469
2026-05-11 18:34:09,716 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:34:09,716 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:34:09,716 INFO train_multi TF=ALL: new best val=2.3153 r_mae=0.9595 — saved
2026-05-11 18:34:09,720 INFO train_multi TF=ALL: new best r_mae=0.9595 — saved rmae checkpoint
2026-05-11 18:34:23,674 INFO train_multi TF=ALL epoch 22/100 train=2.3051 val=2.3097 r_mae=0.957 pos_r_acc=0.552 side_acc=0.545 r_n=127469
2026-05-11 18:34:23,680 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:34:23,680 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:34:23,680 INFO train_multi TF=ALL: new best val=2.3097 r_mae=0.9575 — saved
2026-05-11 18:34:23,685 INFO train_multi TF=ALL: new best r_mae=0.9575 — saved rmae checkpoint
2026-05-11 18:34:37,589 INFO train_multi TF=ALL epoch 23/100 train=2.3016 val=2.3088 r_mae=0.956 pos_r_acc=0.556 side_acc=0.542 r_n=127469
2026-05-11 18:34:37,595 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:34:37,595 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:34:37,595 INFO train_multi TF=ALL: new best val=2.3088 r_mae=0.9564 — saved
2026-05-11 18:34:37,600 INFO train_multi TF=ALL: new best r_mae=0.9564 — saved rmae checkpoint
2026-05-11 18:34:51,314 INFO train_multi TF=ALL epoch 24/100 train=2.2957 val=2.3019 r_mae=0.953 pos_r_acc=0.560 side_acc=0.545 r_n=127469
2026-05-11 18:34:51,320 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:34:51,320 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:34:51,320 INFO train_multi TF=ALL: new best val=2.3019 r_mae=0.9535 — saved
2026-05-11 18:34:51,325 INFO train_multi TF=ALL: new best r_mae=0.9535 — saved rmae checkpoint
2026-05-11 18:35:04,899 INFO train_multi TF=ALL epoch 25/100 train=2.2910 val=2.2990 r_mae=0.951 pos_r_acc=0.562 side_acc=0.545 r_n=127469
2026-05-11 18:35:04,904 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:35:04,904 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:35:04,904 INFO train_multi TF=ALL: new best val=2.2990 r_mae=0.9507 — saved
2026-05-11 18:35:04,909 INFO train_multi TF=ALL: new best r_mae=0.9507 — saved rmae checkpoint
2026-05-11 18:35:18,396 INFO train_multi TF=ALL epoch 26/100 train=2.2817 val=2.2870 r_mae=0.948 pos_r_acc=0.568 side_acc=0.553 r_n=127469
2026-05-11 18:35:18,402 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:35:18,402 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:35:18,402 INFO train_multi TF=ALL: new best val=2.2870 r_mae=0.9476 — saved
2026-05-11 18:35:18,406 INFO train_multi TF=ALL: new best r_mae=0.9476 — saved rmae checkpoint
2026-05-11 18:35:32,004 INFO train_multi TF=ALL epoch 27/100 train=2.2729 val=2.2764 r_mae=0.942 pos_r_acc=0.577 side_acc=0.560 r_n=127469
2026-05-11 18:35:32,009 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:35:32,010 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:35:32,010 INFO train_multi TF=ALL: new best val=2.2764 r_mae=0.9418 — saved
2026-05-11 18:35:32,014 INFO train_multi TF=ALL: new best r_mae=0.9418 — saved rmae checkpoint
2026-05-11 18:35:45,816 INFO train_multi TF=ALL epoch 28/100 train=2.2644 val=2.2763 r_mae=0.938 pos_r_acc=0.574 side_acc=0.556 r_n=127469
2026-05-11 18:35:45,821 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:35:45,822 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:35:45,822 INFO train_multi TF=ALL: new best val=2.2763 r_mae=0.9381 — saved
2026-05-11 18:35:45,826 INFO train_multi TF=ALL: new best r_mae=0.9381 — saved rmae checkpoint
2026-05-11 18:35:59,652 INFO train_multi TF=ALL epoch 29/100 train=2.2530 val=2.2686 r_mae=0.933 pos_r_acc=0.582 side_acc=0.566 r_n=127469
2026-05-11 18:35:59,658 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:35:59,658 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:35:59,658 INFO train_multi TF=ALL: new best val=2.2686 r_mae=0.9328 — saved
2026-05-11 18:35:59,662 INFO train_multi TF=ALL: new best r_mae=0.9328 — saved rmae checkpoint
2026-05-11 18:36:13,314 INFO train_multi TF=ALL epoch 30/100 train=2.2461 val=2.2686 r_mae=0.934 pos_r_acc=0.580 side_acc=0.560 r_n=127469
2026-05-11 18:36:13,320 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:36:13,320 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:36:13,320 INFO train_multi TF=ALL: new best val=2.2686 r_mae=0.9344 — saved
2026-05-11 18:36:27,079 INFO train_multi TF=ALL epoch 31/100 train=2.2427 val=2.2658 r_mae=0.932 pos_r_acc=0.581 side_acc=0.564 r_n=127469
2026-05-11 18:36:27,085 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:36:27,085 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:36:27,085 INFO train_multi TF=ALL: new best val=2.2658 r_mae=0.9323 — saved
2026-05-11 18:36:27,089 INFO train_multi TF=ALL: new best r_mae=0.9323 — saved rmae checkpoint
2026-05-11 18:36:40,932 INFO train_multi TF=ALL epoch 32/100 train=2.2362 val=2.2601 r_mae=0.931 pos_r_acc=0.585 side_acc=0.566 r_n=127469
2026-05-11 18:36:40,938 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:36:40,938 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:36:40,938 INFO train_multi TF=ALL: new best val=2.2601 r_mae=0.9312 — saved
2026-05-11 18:36:40,942 INFO train_multi TF=ALL: new best r_mae=0.9312 — saved rmae checkpoint
2026-05-11 18:36:54,803 INFO train_multi TF=ALL epoch 33/100 train=2.2317 val=2.2638 r_mae=0.928 pos_r_acc=0.583 side_acc=0.566 r_n=127469
2026-05-11 18:36:54,808 INFO train_multi TF=ALL: new best r_mae=0.9282 — saved rmae checkpoint
2026-05-11 18:37:08,555 INFO train_multi TF=ALL epoch 34/100 train=2.2231 val=2.2566 r_mae=0.928 pos_r_acc=0.589 side_acc=0.568 r_n=127469
2026-05-11 18:37:08,560 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:37:08,561 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:37:08,561 INFO train_multi TF=ALL: new best val=2.2566 r_mae=0.9279 — saved
2026-05-11 18:37:08,566 INFO train_multi TF=ALL: new best r_mae=0.9279 — saved rmae checkpoint
2026-05-11 18:37:22,081 INFO train_multi TF=ALL epoch 35/100 train=2.2153 val=2.2669 r_mae=0.928 pos_r_acc=0.583 side_acc=0.559 r_n=127469
2026-05-11 18:37:22,086 INFO train_multi TF=ALL: new best r_mae=0.9276 — saved rmae checkpoint
2026-05-11 18:37:35,795 INFO train_multi TF=ALL epoch 36/100 train=2.2129 val=2.2577 r_mae=0.923 pos_r_acc=0.586 side_acc=0.566 r_n=127469
2026-05-11 18:37:35,800 INFO train_multi TF=ALL: new best r_mae=0.9228 — saved rmae checkpoint
2026-05-11 18:37:49,407 INFO train_multi TF=ALL epoch 37/100 train=2.2057 val=2.2596 r_mae=0.924 pos_r_acc=0.586 side_acc=0.564 r_n=127469
2026-05-11 18:38:03,060 INFO train_multi TF=ALL epoch 38/100 train=2.1993 val=2.2608 r_mae=0.920 pos_r_acc=0.585 side_acc=0.568 r_n=127469
2026-05-11 18:38:03,065 INFO train_multi TF=ALL: new best r_mae=0.9199 — saved rmae checkpoint
2026-05-11 18:38:16,689 INFO train_multi TF=ALL epoch 39/100 train=2.1943 val=2.2610 r_mae=0.919 pos_r_acc=0.588 side_acc=0.567 r_n=127469
2026-05-11 18:38:16,694 INFO train_multi TF=ALL: new best r_mae=0.9193 — saved rmae checkpoint
2026-05-11 18:38:30,480 INFO train_multi TF=ALL epoch 40/100 train=2.1869 val=2.2529 r_mae=0.918 pos_r_acc=0.587 side_acc=0.572 r_n=127469
2026-05-11 18:38:30,486 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:38:30,486 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:38:30,486 INFO train_multi TF=ALL: new best val=2.2529 r_mae=0.9185 — saved
2026-05-11 18:38:30,491 INFO train_multi TF=ALL: new best r_mae=0.9185 — saved rmae checkpoint
2026-05-11 18:38:44,169 INFO train_multi TF=ALL epoch 41/100 train=2.1813 val=2.2716 r_mae=0.921 pos_r_acc=0.583 side_acc=0.565 r_n=127469
2026-05-11 18:38:57,989 INFO train_multi TF=ALL epoch 42/100 train=2.1741 val=2.2877 r_mae=0.919 pos_r_acc=0.579 side_acc=0.563 r_n=127469
2026-05-11 18:39:11,719 INFO train_multi TF=ALL epoch 43/100 train=2.1722 val=2.2536 r_mae=0.913 pos_r_acc=0.590 side_acc=0.577 r_n=127469
2026-05-11 18:39:11,730 INFO train_multi TF=ALL: new best r_mae=0.9134 — saved rmae checkpoint
2026-05-11 18:39:25,688 INFO train_multi TF=ALL epoch 44/100 train=2.1542 val=2.2537 r_mae=0.911 pos_r_acc=0.595 side_acc=0.574 r_n=127469
2026-05-11 18:39:25,693 INFO train_multi TF=ALL: new best r_mae=0.9108 — saved rmae checkpoint
2026-05-11 18:39:39,322 INFO train_multi TF=ALL epoch 45/100 train=2.1454 val=2.2371 r_mae=0.908 pos_r_acc=0.595 side_acc=0.584 r_n=127469
2026-05-11 18:39:39,329 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:39:39,329 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:39:39,329 INFO train_multi TF=ALL: new best val=2.2371 r_mae=0.9082 — saved
2026-05-11 18:39:39,333 INFO train_multi TF=ALL: new best r_mae=0.9082 — saved rmae checkpoint
2026-05-11 18:39:53,033 INFO train_multi TF=ALL epoch 46/100 train=2.1376 val=2.2309 r_mae=0.902 pos_r_acc=0.602 side_acc=0.587 r_n=127469
2026-05-11 18:39:53,039 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:39:53,039 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:39:53,039 INFO train_multi TF=ALL: new best val=2.2309 r_mae=0.9022 — saved
2026-05-11 18:39:53,043 INFO train_multi TF=ALL: new best r_mae=0.9022 — saved rmae checkpoint
2026-05-11 18:40:06,785 INFO train_multi TF=ALL epoch 47/100 train=2.1260 val=2.2242 r_mae=0.894 pos_r_acc=0.609 side_acc=0.597 r_n=127469
2026-05-11 18:40:06,790 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:40:06,790 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:40:06,790 INFO train_multi TF=ALL: new best val=2.2242 r_mae=0.8943 — saved
2026-05-11 18:40:06,795 INFO train_multi TF=ALL: new best r_mae=0.8943 — saved rmae checkpoint
2026-05-11 18:40:20,539 INFO train_multi TF=ALL epoch 48/100 train=2.1150 val=2.2109 r_mae=0.890 pos_r_acc=0.613 side_acc=0.596 r_n=127469
2026-05-11 18:40:20,545 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:40:20,545 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:40:20,545 INFO train_multi TF=ALL: new best val=2.2109 r_mae=0.8898 — saved
2026-05-11 18:40:20,550 INFO train_multi TF=ALL: new best r_mae=0.8898 — saved rmae checkpoint
2026-05-11 18:40:34,287 INFO train_multi TF=ALL epoch 49/100 train=2.0975 val=2.1992 r_mae=0.885 pos_r_acc=0.618 side_acc=0.601 r_n=127469
2026-05-11 18:40:34,293 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:40:34,293 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:40:34,293 INFO train_multi TF=ALL: new best val=2.1992 r_mae=0.8853 — saved
2026-05-11 18:40:34,298 INFO train_multi TF=ALL: new best r_mae=0.8853 — saved rmae checkpoint
2026-05-11 18:40:47,968 INFO train_multi TF=ALL epoch 50/100 train=2.0872 val=2.1919 r_mae=0.879 pos_r_acc=0.623 side_acc=0.600 r_n=127469
2026-05-11 18:40:47,974 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:40:47,974 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:40:47,974 INFO train_multi TF=ALL: new best val=2.1919 r_mae=0.8788 — saved
2026-05-11 18:40:47,978 INFO train_multi TF=ALL: new best r_mae=0.8788 — saved rmae checkpoint
2026-05-11 18:41:01,927 INFO train_multi TF=ALL epoch 51/100 train=2.0716 val=2.1933 r_mae=0.872 pos_r_acc=0.624 side_acc=0.606 r_n=127469
2026-05-11 18:41:01,932 INFO train_multi TF=ALL: new best r_mae=0.8722 — saved rmae checkpoint
2026-05-11 18:41:15,692 INFO train_multi TF=ALL epoch 52/100 train=2.0613 val=2.1643 r_mae=0.868 pos_r_acc=0.631 side_acc=0.612 r_n=127469
2026-05-11 18:41:15,697 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:41:15,697 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:41:15,698 INFO train_multi TF=ALL: new best val=2.1643 r_mae=0.8678 — saved
2026-05-11 18:41:15,702 INFO train_multi TF=ALL: new best r_mae=0.8678 — saved rmae checkpoint
2026-05-11 18:41:29,489 INFO train_multi TF=ALL epoch 53/100 train=2.0464 val=2.1520 r_mae=0.862 pos_r_acc=0.637 side_acc=0.617 r_n=127469
2026-05-11 18:41:29,495 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:41:29,495 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:41:29,495 INFO train_multi TF=ALL: new best val=2.1520 r_mae=0.8619 — saved
2026-05-11 18:41:29,500 INFO train_multi TF=ALL: new best r_mae=0.8619 — saved rmae checkpoint
2026-05-11 18:41:43,309 INFO train_multi TF=ALL epoch 54/100 train=2.0356 val=2.1617 r_mae=0.862 pos_r_acc=0.633 side_acc=0.611 r_n=127469
2026-05-11 18:41:43,315 INFO train_multi TF=ALL: new best r_mae=0.8618 — saved rmae checkpoint
2026-05-11 18:41:57,285 INFO train_multi TF=ALL epoch 55/100 train=2.0266 val=2.1487 r_mae=0.850 pos_r_acc=0.644 side_acc=0.617 r_n=127469
2026-05-11 18:41:57,291 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:41:57,291 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:41:57,291 INFO train_multi TF=ALL: new best val=2.1487 r_mae=0.8503 — saved
2026-05-11 18:41:57,296 INFO train_multi TF=ALL: new best r_mae=0.8503 — saved rmae checkpoint
2026-05-11 18:42:11,121 INFO train_multi TF=ALL epoch 56/100 train=2.0132 val=2.1450 r_mae=0.850 pos_r_acc=0.641 side_acc=0.617 r_n=127469
2026-05-11 18:42:11,126 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:42:11,126 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:42:11,127 INFO train_multi TF=ALL: new best val=2.1450 r_mae=0.8504 — saved
2026-05-11 18:42:24,679 INFO train_multi TF=ALL epoch 57/100 train=2.0012 val=2.1306 r_mae=0.843 pos_r_acc=0.650 side_acc=0.622 r_n=127469
2026-05-11 18:42:24,685 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:42:24,685 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:42:24,685 INFO train_multi TF=ALL: new best val=2.1306 r_mae=0.8430 — saved
2026-05-11 18:42:24,689 INFO train_multi TF=ALL: new best r_mae=0.8430 — saved rmae checkpoint
2026-05-11 18:42:38,393 INFO train_multi TF=ALL epoch 58/100 train=1.9878 val=2.1353 r_mae=0.837 pos_r_acc=0.649 side_acc=0.624 r_n=127469
2026-05-11 18:42:38,398 INFO train_multi TF=ALL: new best r_mae=0.8365 — saved rmae checkpoint
2026-05-11 18:42:52,032 INFO train_multi TF=ALL epoch 59/100 train=1.9784 val=2.1470 r_mae=0.837 pos_r_acc=0.648 side_acc=0.620 r_n=127469
2026-05-11 18:43:06,119 INFO train_multi TF=ALL epoch 60/100 train=1.9674 val=2.1278 r_mae=0.832 pos_r_acc=0.650 side_acc=0.625 r_n=127469
2026-05-11 18:43:06,130 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:43:06,130 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:43:06,130 INFO train_multi TF=ALL: new best val=2.1278 r_mae=0.8320 — saved
2026-05-11 18:43:06,135 INFO train_multi TF=ALL: new best r_mae=0.8320 — saved rmae checkpoint
2026-05-11 18:43:19,740 INFO train_multi TF=ALL epoch 61/100 train=1.9570 val=2.1314 r_mae=0.832 pos_r_acc=0.649 side_acc=0.629 r_n=127469
2026-05-11 18:43:33,549 INFO train_multi TF=ALL epoch 62/100 train=1.9550 val=2.1269 r_mae=0.830 pos_r_acc=0.653 side_acc=0.629 r_n=127469
2026-05-11 18:43:33,555 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:43:33,555 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:43:33,555 INFO train_multi TF=ALL: new best val=2.1269 r_mae=0.8297 — saved
2026-05-11 18:43:33,560 INFO train_multi TF=ALL: new best r_mae=0.8297 — saved rmae checkpoint
2026-05-11 18:43:47,387 INFO train_multi TF=ALL epoch 63/100 train=1.9462 val=2.1594 r_mae=0.832 pos_r_acc=0.648 side_acc=0.621 r_n=127469
2026-05-11 18:44:01,293 INFO train_multi TF=ALL epoch 64/100 train=1.9377 val=2.1487 r_mae=0.831 pos_r_acc=0.648 side_acc=0.621 r_n=127469
2026-05-11 18:44:15,004 INFO train_multi TF=ALL epoch 65/100 train=1.9315 val=2.1184 r_mae=0.825 pos_r_acc=0.655 side_acc=0.627 r_n=127469
2026-05-11 18:44:15,014 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:44:15,014 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:44:15,014 INFO train_multi TF=ALL: new best val=2.1184 r_mae=0.8253 — saved
2026-05-11 18:44:15,019 INFO train_multi TF=ALL: new best r_mae=0.8253 — saved rmae checkpoint
2026-05-11 18:44:28,845 INFO train_multi TF=ALL epoch 66/100 train=1.9194 val=2.1355 r_mae=0.822 pos_r_acc=0.654 side_acc=0.626 r_n=127469
2026-05-11 18:44:28,855 INFO train_multi TF=ALL: new best r_mae=0.8221 — saved rmae checkpoint
2026-05-11 18:44:42,726 INFO train_multi TF=ALL epoch 67/100 train=1.9115 val=2.1409 r_mae=0.819 pos_r_acc=0.656 side_acc=0.625 r_n=127469
2026-05-11 18:44:42,732 INFO train_multi TF=ALL: new best r_mae=0.8186 — saved rmae checkpoint
2026-05-11 18:44:56,695 INFO train_multi TF=ALL epoch 68/100 train=1.9082 val=2.1331 r_mae=0.822 pos_r_acc=0.653 side_acc=0.632 r_n=127469
2026-05-11 18:45:10,498 INFO train_multi TF=ALL epoch 69/100 train=1.9007 val=2.1344 r_mae=0.820 pos_r_acc=0.655 side_acc=0.634 r_n=127469
2026-05-11 18:45:24,303 INFO train_multi TF=ALL epoch 70/100 train=1.8893 val=2.1339 r_mae=0.824 pos_r_acc=0.654 side_acc=0.629 r_n=127469
2026-05-11 18:45:38,122 INFO train_multi TF=ALL epoch 71/100 train=1.8774 val=2.1453 r_mae=0.826 pos_r_acc=0.650 side_acc=0.631 r_n=127469
2026-05-11 18:45:52,072 INFO train_multi TF=ALL epoch 72/100 train=1.8749 val=2.1387 r_mae=0.822 pos_r_acc=0.655 side_acc=0.627 r_n=127469
2026-05-11 18:46:05,951 INFO train_multi TF=ALL epoch 73/100 train=1.8627 val=2.1742 r_mae=0.822 pos_r_acc=0.650 side_acc=0.625 r_n=127469
2026-05-11 18:46:19,724 INFO train_multi TF=ALL epoch 74/100 train=1.8610 val=2.1446 r_mae=0.821 pos_r_acc=0.654 side_acc=0.632 r_n=127469
2026-05-11 18:46:33,789 INFO train_multi TF=ALL epoch 75/100 train=1.8588 val=2.1380 r_mae=0.827 pos_r_acc=0.652 side_acc=0.631 r_n=127469
2026-05-11 18:46:47,952 INFO train_multi TF=ALL epoch 76/100 train=1.8425 val=2.1617 r_mae=0.822 pos_r_acc=0.651 side_acc=0.628 r_n=127469
2026-05-11 18:47:01,830 INFO train_multi TF=ALL epoch 77/100 train=1.8363 val=2.1724 r_mae=0.824 pos_r_acc=0.652 side_acc=0.627 r_n=127469
2026-05-11 18:47:15,758 INFO train_multi TF=ALL epoch 78/100 train=1.8364 val=2.1631 r_mae=0.820 pos_r_acc=0.653 side_acc=0.630 r_n=127469
2026-05-11 18:47:29,555 INFO train_multi TF=ALL epoch 79/100 train=1.8233 val=2.1671 r_mae=0.824 pos_r_acc=0.651 side_acc=0.633 r_n=127469
2026-05-11 18:47:43,715 INFO train_multi TF=ALL epoch 80/100 train=1.8163 val=2.1802 r_mae=0.824 pos_r_acc=0.649 side_acc=0.635 r_n=127469
2026-05-11 18:47:57,497 INFO train_multi TF=ALL epoch 81/100 train=1.8046 val=2.1735 r_mae=0.826 pos_r_acc=0.648 side_acc=0.632 r_n=127469
2026-05-11 18:48:11,288 INFO train_multi TF=ALL epoch 82/100 train=1.8022 val=2.1948 r_mae=0.827 pos_r_acc=0.648 side_acc=0.626 r_n=127469
2026-05-11 18:48:25,311 INFO train_multi TF=ALL epoch 83/100 train=1.8010 val=2.1975 r_mae=0.821 pos_r_acc=0.650 side_acc=0.630 r_n=127469
2026-05-11 18:48:39,326 INFO train_multi TF=ALL epoch 84/100 train=1.7901 val=2.2137 r_mae=0.828 pos_r_acc=0.647 side_acc=0.628 r_n=127469
2026-05-11 18:48:53,268 INFO train_multi TF=ALL epoch 85/100 train=1.7891 val=2.2009 r_mae=0.830 pos_r_acc=0.645 side_acc=0.627 r_n=127469
2026-05-11 18:49:07,649 INFO train_multi TF=ALL epoch 86/100 train=1.7771 val=2.2033 r_mae=0.819 pos_r_acc=0.653 side_acc=0.631 r_n=127469
2026-05-11 18:49:21,355 INFO train_multi TF=ALL epoch 87/100 train=1.7779 val=2.1887 r_mae=0.826 pos_r_acc=0.649 side_acc=0.633 r_n=127469
2026-05-11 18:49:35,351 INFO train_multi TF=ALL epoch 88/100 train=1.7663 val=2.1985 r_mae=0.822 pos_r_acc=0.652 side_acc=0.635 r_n=127469
2026-05-11 18:49:49,008 INFO train_multi TF=ALL epoch 89/100 train=1.7633 val=2.2198 r_mae=0.830 pos_r_acc=0.647 side_acc=0.631 r_n=127469
2026-05-11 18:50:02,899 INFO train_multi TF=ALL epoch 90/100 train=1.7571 val=2.2062 r_mae=0.826 pos_r_acc=0.648 side_acc=0.633 r_n=127469
2026-05-11 18:50:16,768 INFO train_multi TF=ALL epoch 91/100 train=1.7504 val=2.2174 r_mae=0.827 pos_r_acc=0.649 side_acc=0.629 r_n=127469
2026-05-11 18:50:30,666 INFO train_multi TF=ALL epoch 92/100 train=1.7442 val=2.2398 r_mae=0.836 pos_r_acc=0.644 side_acc=0.627 r_n=127469
2026-05-11 18:50:30,666 INFO train_multi TF=ALL early stop at epoch 92
2026-05-11 18:50:30,683 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 18:50:30,683 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 18:50:30,683 INFO train_multi TF=ALL: promoted r_mae checkpoint (0.8186 < primary 0.8253) — overwriting model.pt
2026-05-11 18:50:31,923 WARNING fit_r_isotonic: long calibrator increases MAE (cal=0.8416 >= raw=0.8371) — skipping
2026-05-11 18:50:31,934 WARNING fit_r_isotonic: short calibrator increases MAE (cal=0.8490 >= raw=0.8336) — skipping
2026-05-11 18:50:31,934 INFO fit_r_isotonic: saved side-R calibrators stats={'long': {'n': 31689, 'raw_mae': 0.8370500477289611, 'calibrated_mae': 0.8416123220925649, 'skipped': 'calibrator_hurts'}, 'short': {'n': 32408, 'raw_mae': 0.8336063040849621, 'calibrated_mae': 0.8489838081583733, 'skipped': 'calibrator_hurts'}}
2026-05-11 18:50:32,077 WARNING GRU validation R-MAE above floor for TF=ALL: best_val_r_mae=0.819 max=0.750. Keeping saved best weights so the pipeline can progress.
2026-05-11 18:50:32,092 INFO GRU R threshold XAUUSD/buy: q25_pos=0.535 q50_pos=1.667 pos_rate=46.0% (n=119259 n_pos=54899)
2026-05-11 18:50:32,098 INFO GRU R threshold XAUUSD/sell: q25_pos=0.528 q50_pos=1.667 pos_rate=45.5% (n=118087 n_pos=53734)
2026-05-11 18:50:32,105 INFO GRU R threshold EURUSD/buy: q25_pos=0.541 q50_pos=1.667 pos_rate=45.6% (n=118751 n_pos=54196)
2026-05-11 18:50:32,112 INFO GRU R threshold EURUSD/sell: q25_pos=0.545 q50_pos=1.667 pos_rate=45.6% (n=117679 n_pos=53721)
2026-05-11 18:50:32,118 INFO GRU R threshold USDJPY/buy: q25_pos=0.551 q50_pos=1.667 pos_rate=46.2% (n=118691 n_pos=54870)
2026-05-11 18:50:32,125 INFO GRU R threshold USDJPY/sell: q25_pos=0.542 q50_pos=1.667 pos_rate=44.6% (n=116030 n_pos=51793)
2026-05-11 18:50:32,132 INFO GRU R threshold EURJPY/buy: q25_pos=0.530 q50_pos=1.416 pos_rate=46.1% (n=118651 n_pos=54654)
2026-05-11 18:50:32,139 INFO GRU R threshold EURJPY/sell: q25_pos=0.525 q50_pos=1.667 pos_rate=44.7% (n=117320 n_pos=52399)
2026-05-11 18:50:32,146 INFO GRU R threshold GBPJPY/buy: q25_pos=0.517 q50_pos=1.443 pos_rate=45.9% (n=118277 n_pos=54278)
2026-05-11 18:50:32,153 INFO GRU R threshold GBPJPY/sell: q25_pos=0.521 q50_pos=1.667 pos_rate=45.3% (n=116299 n_pos=52702)
2026-05-11 18:50:32,160 INFO GRU R threshold GBPUSD/buy: q25_pos=0.524 q50_pos=1.667 pos_rate=45.8% (n=117764 n_pos=53894)
2026-05-11 18:50:32,167 INFO GRU R threshold GBPUSD/sell: q25_pos=0.533 q50_pos=1.667 pos_rate=45.8% (n=117404 n_pos=53785)
2026-05-11 18:50:32,167 INFO GRU per-symbol R thresholds saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/symbol_r_thresholds.json (6 symbols)
2026-05-11 18:50:32,187 INFO Retrain complete. Total wall-clock: 1303.5s
  DONE  Retrain gru [train-split retrain]
  START Retrain regime [train-split retrain]
2026-05-11 18:50:35,844 INFO retrain environment: KAGGLE
2026-05-11 18:50:37,524 INFO Device: CUDA (2 GPU(s))
2026-05-11 18:50:37,533 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 18:50:37,533 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 18:50:37,533 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 18:50:37,534 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 18:50:37,534 INFO Retrain data split: train
2026-05-11 18:50:37,534 INFO Retrain rolling fold selector: latest
2026-05-11 18:50:37,535 INFO === RegimeClassifier retrain (hierarchical: HTF 2-score bias + LTF 5-score behaviour) ===
2026-05-11 18:50:37,686 INFO NumExpr defaulting to 4 threads.
2026-05-11 18:50:37,899 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-11 18:50:37,899 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 18:50:37,899 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 18:50:37,900 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-11 18:50:37,964 INFO Regime rolling folds selected: [None]
2026-05-11 18:50:37,964 INFO === Regime rolling fold 1/1: train_all ===
2026-05-11 18:50:37,964 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-11 18:50:38,009 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 18:50:38,010 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:50:38,028 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:50:38,045 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:50:38,063 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:50:38,080 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:50:38,097 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:50:38,352 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:50:38,428 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:50:38,455 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:50:38,455 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:50:38,467 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:50:38,468 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:50:39,331 INFO macro_cache: using training data dir /kaggle/working/Multi-Bot/trading-system/training_data
2026-05-11 18:50:39,458 INFO macro_cache loaded 19 series: {'asx200': '2842 bars 2015-01-02→2026-03-27', 'cac40': '2876 bars 2015-01-02→2026-03-27', 'dax': '2851 bars 2015-01-02→2026-03-27', 'djia': '2825 bars 2015-01-02→2026-03-27', 'dxy': '2826 bars 2015-01-02→2026-03-27', 'eurostoxx': '2823 bars 2015-01-05→2026-03-27', 'ftse': '2839 bars 2015-01-02→2026-03-27', 'gold_fut': '2824 bars 2015-01-02→2026-03-27', 'hsi': '2764 bars 2015-01-02→2026-03-27', 'nasdaq': '2825 bars 2015-01-02→2026-03-27', 'nikkei': '2744 bars 2015-01-05→2026-03-27', 'oil_fut': '2825 bars 2015-01-02→2026-03-27', 'spx': '2825 bars 2015-01-02→2026-03-27', 'us10y': '2824 bars 2015-01-02→2026-03-27', 'us30y': '2824 bars 2015-01-02→2026-03-27', 'us3m': '2824 bars 2015-01-02→2026-03-27', 'vix': '2825 bars 2015-01-02→2026-03-27', 'us10y_fred': '2607 bars 2016-03-28→2026-03-24', 'us2y_fred': '2607 bars 2016-03-28→2026-03-24'}
2026-05-11 18:50:39,736 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1142, 'BIAS_DOWN': 522, 'BIAS_NEUTRAL': 10438}  ambiguous=4182 (total=12102) horizon=84
2026-05-11 18:50:39,742 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected XAUUSD — 12052 samples (group=gold) score_means={'bias_up_score': 0.0948, 'bias_down_score': 0.0433} labels={'BIAS_UP': 1142, 'BIAS_DOWN': 522, 'BIAS_NEUTRAL': 10388} clean={'BIAS_UP': 1142, 'BIAS_DOWN': 522, 'BIAS_NEUTRAL': 6216}
2026-05-11 18:50:39,927 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:39,966 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:39,986 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:39,986 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:39,994 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:39,996 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:50:41,049 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 690, 'BIAS_DOWN': 540, 'BIAS_NEUTRAL': 10174}  ambiguous=3886 (total=11404) horizon=84
2026-05-11 18:50:41,056 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURUSD — 11354 samples (group=dollar) score_means={'bias_up_score': 0.0608, 'bias_down_score': 0.0476} labels={'BIAS_UP': 690, 'BIAS_DOWN': 540, 'BIAS_NEUTRAL': 10124} clean={'BIAS_UP': 690, 'BIAS_DOWN': 540, 'BIAS_NEUTRAL': 6257}
2026-05-11 18:50:41,227 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:41,265 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:41,286 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:41,287 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:41,295 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:41,297 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:50:42,334 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 826, 'BIAS_DOWN': 423, 'BIAS_NEUTRAL': 10154}  ambiguous=4036 (total=11403) horizon=84
2026-05-11 18:50:42,340 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected USDJPY — 11353 samples (group=dollar) score_means={'bias_up_score': 0.0728, 'bias_down_score': 0.0373} labels={'BIAS_UP': 826, 'BIAS_DOWN': 423, 'BIAS_NEUTRAL': 10104} clean={'BIAS_UP': 826, 'BIAS_DOWN': 423, 'BIAS_NEUTRAL': 6078}
2026-05-11 18:50:42,509 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:42,547 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:42,569 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:42,570 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:42,578 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:42,580 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:50:43,603 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 681, 'BIAS_DOWN': 527, 'BIAS_NEUTRAL': 10199}  ambiguous=4044 (total=11407) horizon=84
2026-05-11 18:50:43,608 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURJPY — 11357 samples (group=cross) score_means={'bias_up_score': 0.06, 'bias_down_score': 0.0464} labels={'BIAS_UP': 681, 'BIAS_DOWN': 527, 'BIAS_NEUTRAL': 10149} clean={'BIAS_UP': 681, 'BIAS_DOWN': 527, 'BIAS_NEUTRAL': 6111}
2026-05-11 18:50:43,772 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:43,811 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:43,833 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:43,833 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:43,843 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:43,844 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:50:44,915 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 839, 'BIAS_DOWN': 579, 'BIAS_NEUTRAL': 9990}  ambiguous=4240 (total=11408) horizon=84
2026-05-11 18:50:44,920 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPJPY — 11358 samples (group=cross) score_means={'bias_up_score': 0.0739, 'bias_down_score': 0.051} labels={'BIAS_UP': 839, 'BIAS_DOWN': 579, 'BIAS_NEUTRAL': 9940} clean={'BIAS_UP': 839, 'BIAS_DOWN': 579, 'BIAS_NEUTRAL': 5723}
2026-05-11 18:50:45,083 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:45,121 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:45,142 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:45,142 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:45,151 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:45,152 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:50:46,178 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 502, 'BIAS_DOWN': 720, 'BIAS_NEUTRAL': 10180}  ambiguous=4113 (total=11402) horizon=84
2026-05-11 18:50:46,185 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPUSD — 11352 samples (group=dollar) score_means={'bias_up_score': 0.0442, 'bias_down_score': 0.0623} labels={'BIAS_UP': 502, 'BIAS_DOWN': 707, 'BIAS_NEUTRAL': 10143} clean={'BIAS_UP': 502, 'BIAS_DOWN': 707, 'BIAS_NEUTRAL': 6056}
2026-05-11 18:50:46,261 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 1520, 'BIAS_DOWN': 1106, 'BIAS_NEUTRAL': 20089}, 'dollar': {'BIAS_UP': 2018, 'BIAS_DOWN': 1670, 'BIAS_NEUTRAL': 30371}, 'gold': {'BIAS_UP': 1142, 'BIAS_DOWN': 522, 'BIAS_NEUTRAL': 10388}}
2026-05-11 18:50:46,261 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0669, 'bias_down_score': 0.0487}, 'dollar': {'bias_up_score': 0.0593, 'bias_down_score': 0.049}, 'gold': {'bias_up_score': 0.0948, 'bias_down_score': 0.0433}}
2026-05-11 18:50:46,261 INFO Regime[4H mode=htf_bias] label distribution by year: {2016: {'BIAS_UP': 491, 'BIAS_DOWN': 576, 'BIAS_NEUTRAL': 7755}, 2017: {'BIAS_UP': 734, 'BIAS_DOWN': 286, 'BIAS_NEUTRAL': 8093}, 2018: {'BIAS_UP': 427, 'BIAS_DOWN': 714, 'BIAS_NEUTRAL': 7989}, 2019: {'BIAS_UP': 410, 'BIAS_DOWN': 447, 'BIAS_NEUTRAL': 8245}, 2020: {'BIAS_UP': 694, 'BIAS_DOWN': 174, 'BIAS_NEUTRAL': 8243}, 2021: {'BIAS_UP': 722, 'BIAS_DOWN': 473, 'BIAS_NEUTRAL': 7896}, 2022: {'BIAS_UP': 667, 'BIAS_DOWN': 519, 'BIAS_NEUTRAL': 7935}, 2023: {'BIAS_UP': 535, 'BIAS_DOWN': 109, 'BIAS_NEUTRAL': 4692}}
2026-05-11 18:50:46,261 INFO Regime[4H mode=htf_bias] score means by year: {2016: {'bias_up_score': 0.0557, 'bias_down_score': 0.0653}, 2017: {'bias_up_score': 0.0805, 'bias_down_score': 0.0314}, 2018: {'bias_up_score': 0.0468, 'bias_down_score': 0.0782}, 2019: {'bias_up_score': 0.045, 'bias_down_score': 0.0491}, 2020: {'bias_up_score': 0.0762, 'bias_down_score': 0.0191}, 2021: {'bias_up_score': 0.0794, 'bias_down_score': 0.052}, 2022: {'bias_up_score': 0.0731, 'bias_down_score': 0.0569}, 2023: {'bias_up_score': 0.1003, 'bias_down_score': 0.0204}}
2026-05-11 18:50:46,322 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
2026-05-11 18:50:46,323 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
2026-05-11 18:50:46,324 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
2026-05-11 18:50:46,325 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 18:50:46,326 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 18:50:46,326 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
2026-05-11 18:50:46,342 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:50:46,346 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:50:46,347 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:50:46,348 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:50:46,348 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:50:46,349 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:50:46,956 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 157, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1339}  ambiguous=566 (total=1581) horizon=84
2026-05-11 18:50:46,959 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected XAUUSD — 1531 samples (group=gold) score_means={'bias_up_score': 0.1025, 'bias_down_score': 0.0555} labels={'BIAS_UP': 157, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1289} clean={'BIAS_UP': 157, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 744}
2026-05-11 18:50:47,035 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:47,038 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:47,038 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:47,039 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:47,039 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:47,040 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:50:47,633 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 135, 'BIAS_DOWN': 66, 'BIAS_NEUTRAL': 1290}  ambiguous=531 (total=1491) horizon=84
2026-05-11 18:50:47,636 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURUSD — 1441 samples (group=dollar) score_means={'bias_up_score': 0.0937, 'bias_down_score': 0.0458} labels={'BIAS_UP': 135, 'BIAS_DOWN': 66, 'BIAS_NEUTRAL': 1240} clean={'BIAS_UP': 135, 'BIAS_DOWN': 66, 'BIAS_NEUTRAL': 739}
2026-05-11 18:50:47,709 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:47,711 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:47,712 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:47,712 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:47,712 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:47,713 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:50:48,298 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 164, 'BIAS_DOWN': 77, 'BIAS_NEUTRAL': 1248}  ambiguous=616 (total=1489) horizon=84
2026-05-11 18:50:48,301 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected USDJPY — 1439 samples (group=dollar) score_means={'bias_up_score': 0.114, 'bias_down_score': 0.0535} labels={'BIAS_UP': 164, 'BIAS_DOWN': 77, 'BIAS_NEUTRAL': 1198} clean={'BIAS_UP': 164, 'BIAS_DOWN': 77, 'BIAS_NEUTRAL': 608}
2026-05-11 18:50:48,375 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:48,377 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:48,378 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:48,378 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:48,379 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:48,380 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:50:48,960 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 123, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 1366}  ambiguous=582 (total=1494) horizon=84
2026-05-11 18:50:48,963 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0852, 'bias_down_score': 0.0035} labels={'BIAS_UP': 123, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 1316} clean={'BIAS_UP': 123, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 741}
2026-05-11 18:50:49,037 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:49,039 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:49,040 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:49,040 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:49,040 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:49,041 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:50:49,638 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 129, 'BIAS_DOWN': 9, 'BIAS_NEUTRAL': 1356}  ambiguous=551 (total=1494) horizon=84
2026-05-11 18:50:49,641 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0893, 'bias_down_score': 0.0055} labels={'BIAS_UP': 129, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 1307} clean={'BIAS_UP': 129, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 775}
2026-05-11 18:50:49,715 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:49,717 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:49,718 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:49,719 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:49,719 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:50:49,720 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:50:50,298 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 81, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 1316}  ambiguous=560 (total=1488) horizon=84
2026-05-11 18:50:50,301 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPUSD — 1438 samples (group=dollar) score_means={'bias_up_score': 0.0563, 'bias_down_score': 0.0633} labels={'BIAS_UP': 81, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 1266} clean={'BIAS_UP': 81, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 735}
2026-05-11 18:50:50,373 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 252, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 2623}, 'dollar': {'BIAS_UP': 380, 'BIAS_DOWN': 234, 'BIAS_NEUTRAL': 3704}, 'gold': {'BIAS_UP': 157, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1289}}
2026-05-11 18:50:50,373 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0873, 'bias_down_score': 0.0045}, 'dollar': {'bias_up_score': 0.088, 'bias_down_score': 0.0542}, 'gold': {'bias_up_score': 0.1025, 'bias_down_score': 0.0555}}
2026-05-11 18:50:50,373 INFO Regime[4H mode=htf_bias] label distribution by year: {2022: {'BIAS_UP': 258, 'BIAS_DOWN': 228, 'BIAS_NEUTRAL': 2915}, 2023: {'BIAS_UP': 531, 'BIAS_DOWN': 104, 'BIAS_NEUTRAL': 4701}}
2026-05-11 18:50:50,373 INFO Regime[4H mode=htf_bias] score means by year: {2022: {'bias_up_score': 0.0759, 'bias_down_score': 0.067}, 2023: {'bias_up_score': 0.0995, 'bias_down_score': 0.0195}}
2026-05-11 18:50:50,433 INFO Regime phase HTF dataset build fold=train_all: 12.5s (train=68826 val=8737)
2026-05-11 18:50:50,433 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260511_185050
2026-05-11 18:50:50,637 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=51, n_classes=2)
2026-05-11 18:50:50,638 INFO Regime 4H/htf_bias warm start enabled from existing weights
2026-05-11 18:50:50,652 INFO RegimeClassifier[mode=htf_bias]: HTF clean-label fit filter kept train=44419/68826 val=5463/8737 at conf>=0.40 train_counts={'BIAS_UP': 4680, 'BIAS_DOWN': 3298, 'BIAS_NEUTRAL': 36441} val_counts={'BIAS_UP': 789, 'BIAS_DOWN': 332, 'BIAS_NEUTRAL': 4342}
2026-05-11 18:50:50,652 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=44419 val=5463 train_labels={'BIAS_UP': 4680, 'BIAS_DOWN': 3298, 'BIAS_NEUTRAL': 36441} val_labels={'BIAS_UP': 789, 'BIAS_DOWN': 332, 'BIAS_NEUTRAL': 4342}
2026-05-11 18:50:50,653 INFO RegimeClassifier[mode=htf_bias]: warm start HTF score head
2026-05-11 18:50:50,653 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-11 18:50:50,654 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 8.491, 'bias_down_score': 12.0}
2026-05-11 18:50:50,658 INFO RegimeClassifier[mode=htf_bias]: HTF balanced sampler — dir=7978 neutral=36441 dir_weight=3 => dir_frac_per_epoch≈47.2%
2026-05-11 18:50:54,252 INFO Regime HTF score epoch  1/50 — tr=2.6604 va=1.0186 acc=0.781 bal=0.676 threshold=0.40 margin=0.15 recall={'BIAS_UP': 0.517, 'BIAS_DOWN': 0.675, 'BIAS_NEUTRAL': 0.837} precision={'BIAS_UP': 0.54, 'BIAS_DOWN': 0.384, 'BIAS_NEUTRAL': 0.881}
2026-05-11 18:50:55,784 INFO Regime HTF score epoch  2/50 — tr=2.6281 va=1.0174 bal=0.622
2026-05-11 18:50:57,177 INFO Regime HTF score epoch  3/50 — tr=2.5990 va=0.9984 bal=0.636
2026-05-11 18:50:58,634 INFO Regime HTF score epoch  4/50 — tr=2.5681 va=0.9779 bal=0.647
2026-05-11 18:51:00,065 INFO Regime HTF score epoch  5/50 — tr=2.5128 va=0.9525 acc=0.786 bal=0.644 threshold=0.35 margin=0.45 recall={'BIAS_UP': 0.48, 'BIAS_DOWN': 0.596, 'BIAS_NEUTRAL': 0.856} precision={'BIAS_UP': 0.551, 'BIAS_DOWN': 0.385, 'BIAS_NEUTRAL': 0.872}
2026-05-11 18:51:01,457 INFO Regime HTF score epoch  6/50 — tr=2.4786 va=0.9244 bal=0.631
2026-05-11 18:51:02,852 INFO Regime HTF score epoch  7/50 — tr=2.4181 va=0.8974 bal=0.656
2026-05-11 18:51:04,268 INFO Regime HTF score epoch  8/50 — tr=2.3274 va=0.8674 bal=0.651
2026-05-11 18:51:05,752 INFO Regime HTF score epoch  9/50 — tr=2.2637 va=0.8393 bal=0.642
2026-05-11 18:51:07,158 INFO Regime HTF score epoch 10/50 — tr=2.1783 va=0.8130 acc=0.790 bal=0.607 threshold=0.60 margin=0.15 recall={'BIAS_UP': 0.442, 'BIAS_DOWN': 0.503, 'BIAS_NEUTRAL': 0.875} precision={'BIAS_UP': 0.56, 'BIAS_DOWN': 0.385, 'BIAS_NEUTRAL': 0.863}
2026-05-11 18:51:07,158 INFO Regime HTF score early stop at epoch 10
2026-05-11 18:51:08,445 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.400 margin=0.150 precision={'BIAS_UP': 0.54, 'BIAS_DOWN': 0.384, 'BIAS_NEUTRAL': 0.881} recall={'BIAS_UP': 0.517, 'BIAS_DOWN': 0.675, 'BIAS_NEUTRAL': 0.837} f1={'BIAS_UP': 0.528, 'BIAS_DOWN': 0.489, 'BIAS_NEUTRAL': 0.859} confusion=[[408, 0, 381], [0, 224, 108], [348, 360, 3634]] score_mae={'bias_up_score': 0.2052, 'bias_down_score': 0.1335} pred_share={'BIAS_UP': 0.1384, 'BIAS_DOWN': 0.1069, 'BIAS_NEUTRAL': 0.7547}
2026-05-11 18:51:08,446 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.54, 'BIAS_DOWN': 0.384, 'BIAS_NEUTRAL': 0.881} min_precision=0.500 recall={'BIAS_UP': 0.517, 'BIAS_DOWN': 0.675, 'BIAS_NEUTRAL': 0.837} min_recall=0.150 f1={'BIAS_UP': 0.528, 'BIAS_DOWN': 0.489, 'BIAS_NEUTRAL': 0.859} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_DOWN'] weak_recall=[] weak_f1=[] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-11 18:51:08,451 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 18:51:08,451 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 18:51:08,452 INFO Regime phase HTF train fold=train_all: 17.8s
2026-05-11 18:51:08,570 INFO Regime HTF complete fold=train_all: acc=0.781 bal=0.676 train=68826 val=8737 per_class={'BIAS_UP': 0.517, 'BIAS_DOWN': 0.675, 'BIAS_NEUTRAL': 0.837} precision={'BIAS_UP': 0.54, 'BIAS_DOWN': 0.384, 'BIAS_NEUTRAL': 0.881} threshold=0.400 margin=0.150
2026-05-11 18:51:08,571 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:51:08,780 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 502, 'BIAS_DOWN': 720, 'BIAS_NEUTRAL': 10180}  ambiguous=4113 (total=11402) horizon=84
2026-05-11 18:51:08,782 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on GBPUSD 4H:
{'BIAS_UP': 4.482142857142857, 'BIAS_DOWN': 5.669291338582677, 'BIAS_NEUTRAL': 42.416666666666664}
2026-05-11 18:51:08,786 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (all labels):
{'BIAS_UP': {'n': 502, 'mean': 0.00043281055222645396, 'mean_over_std': 0.19321420748230153}, 'BIAS_DOWN': {'n': 720, 'mean': -0.00046308823082979826, 'mean_over_std': -0.19538165843530478}, 'BIAS_NEUTRAL': {'n': 10179, 'mean': 7.477567618138561e-07, 'mean_over_std': 0.0002829536380249001}}
2026-05-11 18:51:08,787 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 502, 'mean': 0.00043281055222645396, 'mean_over_std': 0.19321420748230153}, 'BIAS_DOWN': {'n': 720, 'mean': -0.00046308823082979826, 'mean_over_std': -0.19538165843530478}, 'BIAS_NEUTRAL': {'n': 6067, 'mean': 9.596616495197703e-06, 'mean_over_std': 0.004013656697571348}}
2026-05-11 18:51:08,792 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-11 18:51:08,794 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:51:08,796 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:51:08,798 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:51:08,800 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:51:08,802 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:51:08,804 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:51:08,820 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:51:08,828 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:51:08,831 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:51:08,831 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:51:08,832 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:51:08,836 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:51:10,009 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected XAUUSD — 44227 samples (group=gold) score_means={'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}
2026-05-11 18:51:10,125 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:10,127 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:10,128 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:10,129 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:10,129 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:10,132 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:51:11,231 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURUSD — 43689 samples (group=dollar) score_means={'trend_score': 0.49, 'range_score': 0.236, 'chop_score': 0.4622, 'volatility_percentile': 0.3812, 'consolidation_score': 0.1862}
2026-05-11 18:51:11,354 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:11,356 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:11,357 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:11,358 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:11,358 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:11,362 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:51:12,469 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected USDJPY — 43686 samples (group=dollar) score_means={'trend_score': 0.4949, 'range_score': 0.2312, 'chop_score': 0.4613, 'volatility_percentile': 0.38, 'consolidation_score': 0.1931}
2026-05-11 18:51:12,590 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:12,592 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:12,593 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:12,594 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:12,594 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:12,596 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:51:13,665 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURJPY — 43688 samples (group=cross) score_means={'trend_score': 0.4888, 'range_score': 0.2345, 'chop_score': 0.4662, 'volatility_percentile': 0.3831, 'consolidation_score': 0.1882}
2026-05-11 18:51:13,779 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:13,781 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:13,782 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:13,783 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:13,783 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:13,785 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:51:14,861 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPJPY — 43678 samples (group=cross) score_means={'trend_score': 0.4878, 'range_score': 0.2348, 'chop_score': 0.4673, 'volatility_percentile': 0.3839, 'consolidation_score': 0.1904}
2026-05-11 18:51:14,981 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:14,983 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:14,984 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:14,984 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:14,984 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:14,987 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:51:16,050 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPUSD — 43676 samples (group=dollar) score_means={'trend_score': 0.4923, 'range_score': 0.2335, 'chop_score': 0.4611, 'volatility_percentile': 0.3832, 'consolidation_score': 0.1856}
2026-05-11 18:51:16,182 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2346, 'chop_score': 0.4667, 'volatility_percentile': 0.3835, 'consolidation_score': 0.1893}, 'dollar': {'trend_score': 0.4924, 'range_score': 0.2335, 'chop_score': 0.4615, 'volatility_percentile': 0.3815, 'consolidation_score': 0.1883}, 'gold': {'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}}
2026-05-11 18:51:16,182 INFO Regime[1H mode=ltf_behaviour] score means by year: {2016: {'trend_score': 0.495, 'range_score': 0.233, 'chop_score': 0.4607, 'volatility_percentile': 0.3776, 'consolidation_score': 0.192}, 2017: {'trend_score': 0.5017, 'range_score': 0.2328, 'chop_score': 0.4574, 'volatility_percentile': 0.3689, 'consolidation_score': 0.1942}, 2018: {'trend_score': 0.4886, 'range_score': 0.237, 'chop_score': 0.4655, 'volatility_percentile': 0.3947, 'consolidation_score': 0.1785}, 2019: {'trend_score': 0.4817, 'range_score': 0.236, 'chop_score': 0.4691, 'volatility_percentile': 0.3725, 'consolidation_score': 0.198}, 2020: {'trend_score': 0.4874, 'range_score': 0.2311, 'chop_score': 0.4674, 'volatility_percentile': 0.3848, 'consolidation_score': 0.1919}, 2021: {'trend_score': 0.4872, 'range_score': 0.2372, 'chop_score': 0.4672, 'volatility_percentile': 0.377, 'consolidation_score': 0.192}, 2022: {'trend_score': 0.4893, 'range_score': 0.2357, 'chop_score': 0.4647, 'volatility_percentile': 0.3962, 'consolidation_score': 0.1784}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 18:51:16,298 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
2026-05-11 18:51:16,299 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 18:51:16,301 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
2026-05-11 18:51:16,302 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 18:51:16,303 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
2026-05-11 18:51:16,305 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
2026-05-11 18:51:16,314 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:51:16,318 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:51:16,319 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:51:16,319 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:51:16,320 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 18:51:16,322 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:51:16,684 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected XAUUSD — 5117 samples (group=gold) score_means={'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}
2026-05-11 18:51:16,804 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:16,808 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:16,809 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:16,810 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:16,810 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:16,812 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:51:17,154 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURUSD — 5053 samples (group=dollar) score_means={'trend_score': 0.4883, 'range_score': 0.2413, 'chop_score': 0.463, 'volatility_percentile': 0.3837, 'consolidation_score': 0.1793}
2026-05-11 18:51:17,273 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:17,275 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:17,276 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:17,277 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:17,277 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:17,279 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:51:17,619 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected USDJPY — 5046 samples (group=dollar) score_means={'trend_score': 0.5085, 'range_score': 0.2259, 'chop_score': 0.4506, 'volatility_percentile': 0.3809, 'consolidation_score': 0.1888}
2026-05-11 18:51:17,735 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:17,737 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:17,738 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:17,739 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:17,739 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:17,741 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:51:18,083 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURJPY — 5053 samples (group=cross) score_means={'trend_score': 0.5023, 'range_score': 0.2261, 'chop_score': 0.4524, 'volatility_percentile': 0.3837, 'consolidation_score': 0.18}
2026-05-11 18:51:18,199 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:18,201 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:18,202 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:18,202 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:18,203 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:18,204 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:51:18,543 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPJPY — 5043 samples (group=cross) score_means={'trend_score': 0.4743, 'range_score': 0.2372, 'chop_score': 0.4716, 'volatility_percentile': 0.3777, 'consolidation_score': 0.1917}
2026-05-11 18:51:18,661 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:18,663 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:18,664 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:18,664 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:18,665 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 18:51:18,667 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 18:51:19,010 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPUSD — 5040 samples (group=dollar) score_means={'trend_score': 0.4843, 'range_score': 0.2371, 'chop_score': 0.4614, 'volatility_percentile': 0.377, 'consolidation_score': 0.1835}
2026-05-11 18:51:19,126 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2317, 'chop_score': 0.462, 'volatility_percentile': 0.3807, 'consolidation_score': 0.1858}, 'dollar': {'trend_score': 0.4937, 'range_score': 0.2348, 'chop_score': 0.4583, 'volatility_percentile': 0.3806, 'consolidation_score': 0.1839}, 'gold': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}}
2026-05-11 18:51:19,127 INFO Regime[1H mode=ltf_behaviour] score means by year: {2022: {'trend_score': 0.4905, 'range_score': 0.2364, 'chop_score': 0.4625, 'volatility_percentile': 0.385, 'consolidation_score': 0.1797}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 18:51:19,239 INFO Regime phase LTF dataset build fold=train_all: 10.4s (train=262644 val=30352)
2026-05-11 18:51:19,240 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260511_185119
2026-05-11 18:51:19,245 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=53, n_classes=5)
2026-05-11 18:51:19,245 INFO Regime 1H/ltf_behaviour warm start enabled from existing weights
2026-05-11 18:51:19,280 INFO RegimeClassifier[mode=ltf_behaviour]: warm start score head
2026-05-11 18:51:19,280 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-11 18:51:19,850 INFO Regime score epoch  1/50 — tr=0.0030 va=0.0007 mae={'trend_score': 0.0151, 'range_score': 0.0293, 'chop_score': 0.0173, 'volatility_percentile': 0.0126, 'consolidation_score': 0.0177}
2026-05-11 18:51:20,413 INFO Regime score epoch  2/50 — tr=0.0030 va=0.0007
2026-05-11 18:51:20,945 INFO Regime score epoch  3/50 — tr=0.0030 va=0.0007
2026-05-11 18:51:21,470 INFO Regime score epoch  4/50 — tr=0.0030 va=0.0007
2026-05-11 18:51:22,009 INFO Regime score epoch  5/50 — tr=0.0030 va=0.0007 mae={'trend_score': 0.0152, 'range_score': 0.029, 'chop_score': 0.0175, 'volatility_percentile': 0.0125, 'consolidation_score': 0.018}
2026-05-11 18:51:22,547 INFO Regime score epoch  6/50 — tr=0.0030 va=0.0007
2026-05-11 18:51:23,070 INFO Regime score epoch  7/50 — tr=0.0030 va=0.0007
2026-05-11 18:51:23,611 INFO Regime score epoch  8/50 — tr=0.0030 va=0.0007
2026-05-11 18:51:24,142 INFO Regime score epoch  9/50 — tr=0.0030 va=0.0007
2026-05-11 18:51:24,715 INFO Regime score epoch 10/50 — tr=0.0030 va=0.0007 mae={'trend_score': 0.0153, 'range_score': 0.029, 'chop_score': 0.017, 'volatility_percentile': 0.0135, 'consolidation_score': 0.0179}
2026-05-11 18:51:25,249 INFO Regime score epoch 11/50 — tr=0.0030 va=0.0007
2026-05-11 18:51:25,799 INFO Regime score epoch 12/50 — tr=0.0030 va=0.0007
2026-05-11 18:51:26,328 INFO Regime score epoch 13/50 — tr=0.0030 va=0.0007
2026-05-11 18:51:26,854 INFO Regime score epoch 14/50 — tr=0.0030 va=0.0007
2026-05-11 18:51:27,373 INFO Regime score epoch 15/50 — tr=0.0030 va=0.0007 mae={'trend_score': 0.015, 'range_score': 0.0288, 'chop_score': 0.0171, 'volatility_percentile': 0.0122, 'consolidation_score': 0.0177}
2026-05-11 18:51:27,893 INFO Regime score epoch 16/50 — tr=0.0030 va=0.0007
2026-05-11 18:51:28,408 INFO Regime score epoch 17/50 — tr=0.0030 va=0.0007
2026-05-11 18:51:28,408 INFO Regime score early stop at epoch 17
2026-05-11 18:51:28,429 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.015, 'range_score': 0.0288, 'chop_score': 0.0169, 'volatility_percentile': 0.0121, 'consolidation_score': 0.0177} mse={'trend_score': 0.00039, 'range_score': 0.00139, 'chop_score': 0.00047, 'volatility_percentile': 0.0003, 'consolidation_score': 0.00073} corr={'trend_score': 0.996, 'range_score': 0.967, 'chop_score': 0.9937, 'volatility_percentile': 0.997, 'consolidation_score': 0.9926} pred_std={'trend_score': 0.2204, 'range_score': 0.1317, 'chop_score': 0.1827, 'volatility_percentile': 0.2188, 'consolidation_score': 0.213} target_std={'trend_score': 0.2219, 'range_score': 0.1433, 'chop_score': 0.1888, 'volatility_percentile': 0.2165, 'consolidation_score': 0.2179}
2026-05-11 18:51:28,777 INFO RegimeClassifier[mode=ltf_behaviour] train group diagnostics={'dollar': {'n': 16632, 'score_mae': {'trend_score': 0.0146, 'range_score': 0.0287, 'chop_score': 0.0169, 'volatility_percentile': 0.0117, 'consolidation_score': 0.0181}, 'target_score_mean': {'trend_score': 0.491, 'range_score': 0.2347, 'chop_score': 0.4613, 'volatility_percentile': 0.3824, 'consolidation_score': 0.186}, 'pred_score_mean': {'trend_score': 0.4918, 'range_score': 0.2325, 'chop_score': 0.4605, 'volatility_percentile': 0.3797, 'consolidation_score': 0.182}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3583, 50, 0, 3, 0, 0, 143], [4, 95, 0, 0, 0, 6, 5], [0, 0, 188, 11, 56, 0, 205], [2, 0, 3, 574, 26, 0, 84], [0, 0, 17, 20, 3119, 1, 159], [0, 16, 0, 0, 7, 71, 34], [123, 12, 41, 38, 62, 7, 7867]]}, 'gold': {'n': 8420, 'score_mae': {'trend_score': 0.0146, 'range_score': 0.0292, 'chop_score': 0.017, 'volatility_percentile': 0.0122, 'consolidation_score': 0.0183}, 'target_score_mean': {'trend_score': 0.4883, 'range_score': 0.2355, 'chop_score': 0.4652, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}, 'pred_score_mean': {'trend_score': 0.4893, 'range_score': 0.2333, 'chop_score': 0.4643, 'volatility_percentile': 0.3737, 'consolidation_score': 0.1877}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1786, 30, 0, 0, 0, 0, 69], [4, 49, 0, 0, 0, 1, 2], [0, 0, 95, 10, 26, 0, 113], [2, 0, 2, 355, 11, 0, 46], [0, 0, 13, 17, 1597, 0, 77], [0, 11, 0, 0, 4, 47, 19], [53, 2, 33, 11, 42, 0, 3893]]}, 'jpy': {'n': 24948, 'score_mae': {'trend_score': 0.0145, 'range_score': 0.0287, 'chop_score': 0.0167, 'volatility_percentile': 0.0126, 'consolidation_score': 0.0181}, 'target_score_mean': {'trend_score': 0.4904, 'range_score': 0.2338, 'chop_score': 0.4653, 'volatility_percentile': 0.3826, 'consolidation_score': 0.1905}, 'pred_score_mean': {'trend_score': 0.4907, 'range_score': 0.2316, 'chop_score': 0.4642, 'volatility_percentile': 0.3794, 'consolidation_score': 0.1858}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[5331, 98, 0, 2, 0, 0, 215], [7, 168, 0, 0, 0, 5, 7], [0, 0, 238, 18, 91, 0, 300], [3, 0, 3, 1094, 65, 0, 149], [0, 0, 24, 47, 4795, 0, 249], [0, 28, 0, 0, 14, 106, 75], [165, 10, 72, 67, 109, 10, 11383]]}}
2026-05-11 18:51:28,984 INFO RegimeClassifier[mode=ltf_behaviour] validation group diagnostics={'dollar': {'n': 10093, 'score_mae': {'trend_score': 0.0153, 'range_score': 0.0292, 'chop_score': 0.0171, 'volatility_percentile': 0.0115, 'consolidation_score': 0.0173}, 'target_score_mean': {'trend_score': 0.4863, 'range_score': 0.2392, 'chop_score': 0.4622, 'volatility_percentile': 0.3804, 'consolidation_score': 0.1814}, 'pred_score_mean': {'trend_score': 0.4874, 'range_score': 0.2352, 'chop_score': 0.4624, 'volatility_percentile': 0.3775, 'consolidation_score': 0.1777}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[2299, 21, 0, 1, 0, 0, 91], [3, 47, 0, 0, 0, 3, 0], [0, 0, 106, 7, 49, 0, 154], [0, 0, 2, 347, 22, 0, 52], [0, 0, 12, 16, 1936, 0, 86], [0, 9, 0, 0, 3, 40, 25], [48, 6, 24, 30, 40, 5, 4609]]}, 'gold': {'n': 5117, 'score_mae': {'trend_score': 0.0142, 'range_score': 0.0281, 'chop_score': 0.017, 'volatility_percentile': 0.0119, 'consolidation_score': 0.0181}, 'target_score_mean': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}, 'pred_score_mean': {'trend_score': 0.4974, 'range_score': 0.2295, 'chop_score': 0.4566, 'volatility_percentile': 0.3782, 'consolidation_score': 0.1792}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1112, 10, 0, 0, 0, 0, 45], [3, 29, 0, 0, 0, 2, 1], [0, 0, 63, 3, 16, 0, 89], [0, 0, 2, 224, 8, 0, 21], [0, 0, 4, 10, 827, 1, 45], [0, 5, 0, 0, 3, 29, 13], [44, 1, 21, 16, 22, 1, 2447]]}, 'jpy': {'n': 15142, 'score_mae': {'trend_score': 0.0151, 'range_score': 0.0289, 'chop_score': 0.0168, 'volatility_percentile': 0.0126, 'consolidation_score': 0.0178}, 'target_score_mean': {'trend_score': 0.495, 'range_score': 0.2297, 'chop_score': 0.4582, 'volatility_percentile': 0.3808, 'consolidation_score': 0.1868}, 'pred_score_mean': {'trend_score': 0.4949, 'range_score': 0.2267, 'chop_score': 0.4582, 'volatility_percentile': 0.3777, 'consolidation_score': 0.1823}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3337, 39, 0, 1, 0, 0, 141], [6, 97, 0, 0, 0, 7, 5], [0, 0, 137, 15, 47, 0, 185], [3, 0, 2, 709, 28, 0, 85], [0, 0, 19, 29, 2625, 0, 144], [0, 14, 0, 0, 9, 64, 35], [97, 8, 42, 33, 61, 12, 7106]]}}
2026-05-11 18:51:28,991 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-11 18:51:28,991 INFO Regime phase LTF train fold=train_all: 9.7s
2026-05-11 18:51:29,111 INFO Regime LTF complete fold=train_all: score_accuracy=0.982, train=262644 val=30352 mae={'trend_score': 0.015, 'range_score': 0.0288, 'chop_score': 0.0169, 'volatility_percentile': 0.0121, 'consolidation_score': 0.0177}
2026-05-11 18:51:29,113 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 18:51:29,512 INFO Regime[1H mode=ltf_behaviour fold=latest] LTF score diagnostics on GBPUSD:
{'trend_score': {'mean': 0.4921, 'q10': 0.1918, 'q50': 0.4877, 'q90': 0.8002}, 'range_score': {'mean': 0.2337, 'q10': 0.0513, 'q50': 0.2125, 'q90': 0.4349}, 'chop_score': {'mean': 0.4612, 'q10': 0.2148, 'q50': 0.4503, 'q90': 0.7255}, 'volatility_percentile': {'mean': 0.3832, 'q10': 0.1018, 'q50': 0.3704, 'q90': 0.6823}, 'consolidation_score': {'mean': 0.1854, 'q10': 0.0, 'q50': 0.1185, 'q90': 0.506}}
2026-05-11 18:51:29,517 INFO Regime retrain total: 52.0s (370559 train+val samples)
2026-05-11 18:51:29,522 INFO Retrain complete. Total wall-clock: 52.0s
  DONE  Retrain regime [train-split retrain]
  SKIP  Quality/RL incremental retrain — clean train-only weights retained

=== Round 3: Backtest on last 3yr (post-retrain evaluation) ===
  START Round 3 - Post-retrain backtest (last 3yr)
2026-05-11 18:51:31,063 INFO === STEP 6: BACKTEST (round3) ===
2026-05-11 18:51:31,064 INFO BT_WINDOW=round3 — post-retrain eval: 2022-08-05 → 2025-08-05 (last 3yr)
2026-05-11 18:51:31,065 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-05-11 18:51:31,065 INFO Round 3 — running backtest: 2022-08-05 → 2025-08-05 (ml_trader, shared ML cache)
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
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
2026-05-11 18:53:28,957 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
2026-05-11 18:53:28,996 WARNING _build_sequence_df: HTF frame 5M filled 461 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 18:53:29,448 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
2026-05-11 18:53:29,518 WARNING _build_sequence_df: HTF frame 5M filled 461 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 18:53:30,095 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 18:53:30,166 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 18:53:30,206 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 18:53:30,298 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
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
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
2026-05-11 18:53:45,208 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 18:53:45,360 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 18:53:45,473 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 18:53:45,518 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 18:55:06,477 INFO Round 3 backtest — 4 trades | avg WR=25.0% | avg PF=0.39 | avg Sharpe=-6.68
2026-05-11 18:55:06,477 INFO   ml_trader: 4 trades | WR=25.0% | fixed PF=0.39 | Return=-1.8% | ExpR=-0.458 | DD=2.0% | Sharpe=-6.68
2026-05-11 18:55:06,477 INFO   ml_trader gate_diagnostics: bars=403523 no_signal=291536 quality_block=154 session_skip=111829 density=0 pm_reject=0
2026-05-11 18:55:06,477 INFO   ml_trader no_signal_reasons: {'no_trade_chop': 35614, 'no_trade_uncertain': 110730, 'weak_gru_direction': 73847, 'gru_expected_r_below_threshold': 11864, 'trend_structure_missing': 5950, 'no_trade_extreme_vol': 27564, 'wait_pullback': 15997, 'tradeability_direction_conflict': 4172, 'htf_low_regime_confidence': 5772, 'expected_r_below_threshold': 26}
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 4
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (4 rows)
  DONE  Round 3 - Post-retrain backtest (last 3yr)
  Saved Round 3 result → round3_summary.json
  Journal after Round 3: 119 entries

  SKIP  Round 3 Quality+RL retrain — evaluation journals not used for fitting

======================================================================
  BLIND BACKTEST PIPELINE COMPLETE
======================================================================
  Round 1 (train-tail window)   trades=115  WR=21.7%  PF=0.705  Sharpe=-2.431
  Round 2 (blind test)          trades=0  WR=0.0%  PF=0.000  Sharpe=0.000
  Round 3 (last 3yr)            trades=4  WR=25.0%  PF=0.391  Sharpe=-6.681


WARNING: GITHUB_TOKEN not set — skipping GitHub push
2026-05-11 18:55:06,699 INFO Round 3: wrote 4 journal entries (total in file: 119)