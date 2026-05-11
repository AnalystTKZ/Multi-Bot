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
2026-05-11 00:56:17,797 INFO Loading feature-engineered data...
2026-05-11 00:56:18,454 INFO Loaded 221743 rows, 202 features
2026-05-11 00:56:18,456 INFO Data span: 2016-01-04 -> 2025-08-05  (9.6 years)
2026-05-11 00:56:18,458 INFO Fold 000 train 2016-01-04 -> 2018-01-03 (47088 bars), val 2018-01-04 -> 2019-01-03 (23448 bars)
2026-05-11 00:56:18,459 INFO Fold 001 train 2016-01-04 -> 2019-01-03 (70536 bars), val 2019-01-04 -> 2020-01-03 (23377 bars)
2026-05-11 00:56:18,459 INFO Fold 002 train 2016-01-04 -> 2020-01-03 (93913 bars), val 2020-01-06 -> 2020-12-31 (23259 bars)
2026-05-11 00:56:18,459 INFO Fold 003 train 2016-01-04 -> 2020-12-31 (117172 bars), val 2021-01-04 -> 2022-01-03 (23507 bars)
2026-05-11 00:56:18,459 INFO Fold 004 train 2016-01-04 -> 2022-01-03 (140679 bars), val 2022-01-04 -> 2023-01-03 (23588 bars)
2026-05-11 00:56:18,460 INFO Fold 005 train 2016-01-04 -> 2022-08-05 (154539 bars), val 2022-08-05 -> 2023-08-04 (20412 bars)
2026-05-11 00:56:18,460 INFO No leakage confirmed: train/train_tail/internal folds end before final 2-year blind test

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
2026-05-11 00:56:27,886 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-05-11 00:56:27,886 INFO --- Training gru ---
2026-05-11 00:56:27,887 INFO Running retrain --model gru
2026-05-11 00:56:28,078 INFO retrain environment: KAGGLE
2026-05-11 00:56:29,711 INFO Device: CUDA (2 GPU(s))
2026-05-11 00:56:29,722 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 00:56:29,722 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 00:56:29,722 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 00:56:29,724 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 00:56:29,724 INFO Retrain data split: train
2026-05-11 00:56:29,724 INFO Retrain rolling fold selector: latest
2026-05-11 00:56:29,725 INFO === GRU-LSTM retrain (timeframes: ['15M']) ===
2026-05-11 00:56:29,905 INFO NumExpr defaulting to 4 threads.
2026-05-11 00:56:30,121 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-11 00:56:30,121 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 00:56:30,121 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 00:56:30,122 INFO GRU multi-symbol training (Kaggle mode): 6 symbols × ['15M']
2026-05-11 00:56:30,122 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260511_005630
2026-05-11 00:56:30,124 WARNING WeightsManifest: no manifest at /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json — assuming compatible (run a full retrain to generate one)
2026-05-11 00:56:30,125 INFO GRU cold start: no compatible existing weights found
2026-05-11 00:56:30,388 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 00:56:30,416 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 00:56:30,433 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 00:56:30,442 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 00:56:30,519 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 00:56:30,525 INFO Loaded XAUUSD/15M split=train fold=latest: 176438 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 00:56:31,084 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 00:56:31,102 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 00:56:31,117 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 00:56:31,124 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 00:56:31,170 INFO Loaded EURUSD/15M split=train fold=latest: 174951 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 00:56:31,780 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 00:56:31,801 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 00:56:31,816 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 00:56:31,823 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 00:56:31,863 INFO Loaded USDJPY/15M split=train fold=latest: 174944 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 00:56:32,404 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 00:56:32,429 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 00:56:32,444 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 00:56:32,452 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 00:56:32,493 INFO Loaded EURJPY/15M split=train fold=latest: 174948 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 00:56:33,013 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 00:56:33,034 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 00:56:33,050 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 00:56:33,058 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 00:56:33,097 INFO Loaded GBPJPY/15M split=train fold=latest: 174902 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 00:56:33,627 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 00:56:33,647 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 00:56:33,661 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 00:56:33,669 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 00:56:33,707 INFO Loaded GBPUSD/15M split=train fold=latest: 174899 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 00:56:34,137 INFO train_multi: 6 segments, ~971854 total bars
2026-05-11 00:56:34,530 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-05-11 00:56:34,530 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-05-11 00:56:34,530 INFO train_multi: training ALL 6 segments across TFs ['15M'] in one combined pass
2026-05-11 00:56:34,531 INFO train_multi: building combined dataset for TF=ALL (6 segments)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 00:56:44,098 INFO train_multi TF=ALL: 971494 sequences across 6 segments
2026-05-11 00:56:44,098 INFO train_multi TF=ALL: estimated peak RAM = 21312 MB (train=479996 val=120002 n_feat=74 seq_len=60)
2026-05-11 00:56:44,098 WARNING train_multi TF=ALL: trimming to fit RAM budget — new train=450450 val=112612 (20000 MB est)
2026-05-11 00:56:46,435 INFO train_multi TF=ALL: train=450450 val=112612 (10009 MB tensors)
2026-05-11 00:56:52,994 INFO train_multi TF=ALL: structural bar weighting — 288225 structural bars (64.0%) weight=15.0 structural_only=0
2026-05-11 00:56:56,006 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=25, min_epochs=45)
2026-05-11 00:57:16,617 INFO train_multi TF=ALL epoch 1/100 train=2.3417 val=2.3456 r_mae=0.974 pos_r_acc=0.505 side_acc=0.493 r_n=161888
2026-05-11 00:57:16,627 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 00:57:16,628 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 00:57:16,628 INFO train_multi TF=ALL: new best val=2.3456 — saved
2026-05-11 00:57:34,526 INFO train_multi TF=ALL epoch 2/100 train=2.3390 val=2.3418 r_mae=0.971 pos_r_acc=0.505 side_acc=0.493 r_n=161888
2026-05-11 00:57:34,531 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 00:57:34,531 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 00:57:34,531 INFO train_multi TF=ALL: new best val=2.3418 — saved
2026-05-11 00:57:52,439 INFO train_multi TF=ALL epoch 3/100 train=2.3355 val=2.3360 r_mae=0.966 pos_r_acc=0.545 side_acc=0.493 r_n=161888
2026-05-11 00:57:52,444 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 00:57:52,444 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 00:57:52,444 INFO train_multi TF=ALL: new best val=2.3360 — saved
2026-05-11 00:58:10,461 INFO train_multi TF=ALL epoch 4/100 train=2.3334 val=2.3350 r_mae=0.965 pos_r_acc=0.545 side_acc=0.493 r_n=161888
2026-05-11 00:58:10,467 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 00:58:10,467 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 00:58:10,467 INFO train_multi TF=ALL: new best val=2.3350 — saved
2026-05-11 00:58:28,334 INFO train_multi TF=ALL epoch 5/100 train=2.3321 val=2.3330 r_mae=0.965 pos_r_acc=0.545 side_acc=0.497 r_n=161888
2026-05-11 00:58:28,339 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 00:58:28,339 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 00:58:28,339 INFO train_multi TF=ALL: new best val=2.3330 — saved
2026-05-11 00:58:46,390 INFO train_multi TF=ALL epoch 6/100 train=2.3278 val=2.3270 r_mae=0.963 pos_r_acc=0.547 side_acc=0.520 r_n=161888
2026-05-11 00:58:46,400 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 00:58:46,400 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 00:58:46,400 INFO train_multi TF=ALL: new best val=2.3270 — saved
2026-05-11 00:59:04,409 INFO train_multi TF=ALL epoch 7/100 train=2.3230 val=2.3249 r_mae=0.963 pos_r_acc=0.547 side_acc=0.521 r_n=161888
2026-05-11 00:59:04,414 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 00:59:04,414 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 00:59:04,414 INFO train_multi TF=ALL: new best val=2.3249 — saved
2026-05-11 00:59:22,342 INFO train_multi TF=ALL epoch 8/100 train=2.3193 val=2.3225 r_mae=0.962 pos_r_acc=0.546 side_acc=0.527 r_n=161888
2026-05-11 00:59:22,348 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 00:59:22,348 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 00:59:22,348 INFO train_multi TF=ALL: new best val=2.3225 — saved
2026-05-11 00:59:40,237 INFO train_multi TF=ALL epoch 9/100 train=2.3167 val=2.3208 r_mae=0.961 pos_r_acc=0.547 side_acc=0.528 r_n=161888
2026-05-11 00:59:40,242 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 00:59:40,242 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 00:59:40,242 INFO train_multi TF=ALL: new best val=2.3208 — saved
2026-05-11 00:59:58,125 INFO train_multi TF=ALL epoch 10/100 train=2.3137 val=2.3185 r_mae=0.959 pos_r_acc=0.548 side_acc=0.533 r_n=161888
2026-05-11 00:59:58,130 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 00:59:58,131 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 00:59:58,131 INFO train_multi TF=ALL: new best val=2.3185 — saved
2026-05-11 01:00:16,006 INFO train_multi TF=ALL epoch 11/100 train=2.3104 val=2.3141 r_mae=0.958 pos_r_acc=0.553 side_acc=0.536 r_n=161888
2026-05-11 01:00:16,011 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:00:16,011 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:00:16,011 INFO train_multi TF=ALL: new best val=2.3141 — saved
2026-05-11 01:00:33,935 INFO train_multi TF=ALL epoch 12/100 train=2.3038 val=2.2991 r_mae=0.950 pos_r_acc=0.568 side_acc=0.549 r_n=161888
2026-05-11 01:00:33,940 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:00:33,940 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:00:33,940 INFO train_multi TF=ALL: new best val=2.2991 — saved
2026-05-11 01:00:51,903 INFO train_multi TF=ALL epoch 13/100 train=2.2926 val=2.2858 r_mae=0.944 pos_r_acc=0.576 side_acc=0.549 r_n=161888
2026-05-11 01:00:51,913 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:00:51,913 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:00:51,913 INFO train_multi TF=ALL: new best val=2.2858 — saved
2026-05-11 01:01:09,853 INFO train_multi TF=ALL epoch 14/100 train=2.2799 val=2.2736 r_mae=0.938 pos_r_acc=0.583 side_acc=0.555 r_n=161888
2026-05-11 01:01:09,863 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:01:09,863 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:01:09,863 INFO train_multi TF=ALL: new best val=2.2736 — saved
2026-05-11 01:01:27,928 INFO train_multi TF=ALL epoch 15/100 train=2.2738 val=2.2676 r_mae=0.936 pos_r_acc=0.585 side_acc=0.558 r_n=161888
2026-05-11 01:01:27,934 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:01:27,934 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:01:27,934 INFO train_multi TF=ALL: new best val=2.2676 — saved
2026-05-11 01:01:45,783 INFO train_multi TF=ALL epoch 16/100 train=2.2682 val=2.2644 r_mae=0.933 pos_r_acc=0.585 side_acc=0.560 r_n=161888
2026-05-11 01:01:45,788 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:01:45,788 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:01:45,788 INFO train_multi TF=ALL: new best val=2.2644 — saved
2026-05-11 01:02:03,749 INFO train_multi TF=ALL epoch 17/100 train=2.2612 val=2.2587 r_mae=0.931 pos_r_acc=0.590 side_acc=0.560 r_n=161888
2026-05-11 01:02:03,753 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:02:03,753 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:02:03,754 INFO train_multi TF=ALL: new best val=2.2587 — saved
2026-05-11 01:02:21,633 INFO train_multi TF=ALL epoch 18/100 train=2.2565 val=2.2555 r_mae=0.931 pos_r_acc=0.591 side_acc=0.562 r_n=161888
2026-05-11 01:02:21,638 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:02:21,638 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:02:21,638 INFO train_multi TF=ALL: new best val=2.2555 — saved
2026-05-11 01:02:39,569 INFO train_multi TF=ALL epoch 19/100 train=2.2507 val=2.2500 r_mae=0.926 pos_r_acc=0.593 side_acc=0.567 r_n=161888
2026-05-11 01:02:39,574 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:02:39,574 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:02:39,574 INFO train_multi TF=ALL: new best val=2.2500 — saved
2026-05-11 01:02:57,386 INFO train_multi TF=ALL epoch 20/100 train=2.2461 val=2.2465 r_mae=0.924 pos_r_acc=0.596 side_acc=0.564 r_n=161888
2026-05-11 01:02:57,391 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:02:57,391 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:02:57,391 INFO train_multi TF=ALL: new best val=2.2465 — saved
2026-05-11 01:03:15,254 INFO train_multi TF=ALL epoch 21/100 train=2.2411 val=2.2407 r_mae=0.922 pos_r_acc=0.598 side_acc=0.566 r_n=161888
2026-05-11 01:03:15,259 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:03:15,259 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:03:15,259 INFO train_multi TF=ALL: new best val=2.2407 — saved
2026-05-11 01:03:33,166 INFO train_multi TF=ALL epoch 22/100 train=2.2391 val=2.2396 r_mae=0.920 pos_r_acc=0.600 side_acc=0.567 r_n=161888
2026-05-11 01:03:33,170 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:03:33,171 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:03:33,171 INFO train_multi TF=ALL: new best val=2.2396 — saved
2026-05-11 01:03:51,315 INFO train_multi TF=ALL epoch 23/100 train=2.2309 val=2.2358 r_mae=0.917 pos_r_acc=0.599 side_acc=0.570 r_n=161888
2026-05-11 01:03:51,320 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:03:51,320 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:03:51,320 INFO train_multi TF=ALL: new best val=2.2358 — saved
2026-05-11 01:04:09,029 INFO train_multi TF=ALL epoch 24/100 train=2.2279 val=2.2327 r_mae=0.919 pos_r_acc=0.601 side_acc=0.571 r_n=161888
2026-05-11 01:04:09,035 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:04:09,035 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:04:09,035 INFO train_multi TF=ALL: new best val=2.2327 — saved
2026-05-11 01:04:26,996 INFO train_multi TF=ALL epoch 25/100 train=2.2242 val=2.2276 r_mae=0.918 pos_r_acc=0.604 side_acc=0.575 r_n=161888
2026-05-11 01:04:27,001 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:04:27,001 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:04:27,001 INFO train_multi TF=ALL: new best val=2.2276 — saved
2026-05-11 01:04:44,968 INFO train_multi TF=ALL epoch 26/100 train=2.2181 val=2.2250 r_mae=0.915 pos_r_acc=0.603 side_acc=0.579 r_n=161888
2026-05-11 01:04:44,974 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:04:44,974 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:04:44,974 INFO train_multi TF=ALL: new best val=2.2250 — saved
2026-05-11 01:05:03,235 INFO train_multi TF=ALL epoch 27/100 train=2.2135 val=2.2171 r_mae=0.911 pos_r_acc=0.609 side_acc=0.584 r_n=161888
2026-05-11 01:05:03,240 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:05:03,240 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:05:03,241 INFO train_multi TF=ALL: new best val=2.2171 — saved
2026-05-11 01:05:21,309 INFO train_multi TF=ALL epoch 28/100 train=2.2051 val=2.2164 r_mae=0.910 pos_r_acc=0.606 side_acc=0.585 r_n=161888
2026-05-11 01:05:21,315 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:05:21,315 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:05:21,315 INFO train_multi TF=ALL: new best val=2.2164 — saved
2026-05-11 01:05:39,563 INFO train_multi TF=ALL epoch 29/100 train=2.1944 val=2.1939 r_mae=0.901 pos_r_acc=0.617 side_acc=0.603 r_n=161888
2026-05-11 01:05:39,568 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:05:39,568 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:05:39,568 INFO train_multi TF=ALL: new best val=2.1939 — saved
2026-05-11 01:05:57,838 INFO train_multi TF=ALL epoch 30/100 train=2.1836 val=2.1830 r_mae=0.900 pos_r_acc=0.615 side_acc=0.607 r_n=161888
2026-05-11 01:05:57,843 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:05:57,844 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:05:57,844 INFO train_multi TF=ALL: new best val=2.1830 — saved
2026-05-11 01:06:15,882 INFO train_multi TF=ALL epoch 31/100 train=2.1684 val=2.1617 r_mae=0.893 pos_r_acc=0.624 side_acc=0.618 r_n=161888
2026-05-11 01:06:15,887 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:06:15,887 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:06:15,887 INFO train_multi TF=ALL: new best val=2.1617 — saved
2026-05-11 01:06:33,962 INFO train_multi TF=ALL epoch 32/100 train=2.1499 val=2.1443 r_mae=0.886 pos_r_acc=0.633 side_acc=0.623 r_n=161888
2026-05-11 01:06:33,967 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:06:33,967 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:06:33,967 INFO train_multi TF=ALL: new best val=2.1443 — saved
2026-05-11 01:06:51,982 INFO train_multi TF=ALL epoch 33/100 train=2.1260 val=2.1209 r_mae=0.874 pos_r_acc=0.637 side_acc=0.632 r_n=161888
2026-05-11 01:06:51,988 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:06:51,988 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:06:51,988 INFO train_multi TF=ALL: new best val=2.1209 — saved
2026-05-11 01:07:10,153 INFO train_multi TF=ALL epoch 34/100 train=2.1030 val=2.1096 r_mae=0.862 pos_r_acc=0.648 side_acc=0.631 r_n=161888
2026-05-11 01:07:10,158 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:07:10,158 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:07:10,158 INFO train_multi TF=ALL: new best val=2.1096 — saved
2026-05-11 01:07:28,198 INFO train_multi TF=ALL epoch 35/100 train=2.0810 val=2.0903 r_mae=0.848 pos_r_acc=0.653 side_acc=0.637 r_n=161888
2026-05-11 01:07:28,208 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:07:28,208 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:07:28,208 INFO train_multi TF=ALL: new best val=2.0903 — saved
2026-05-11 01:07:46,357 INFO train_multi TF=ALL epoch 36/100 train=2.0711 val=2.0726 r_mae=0.846 pos_r_acc=0.659 side_acc=0.644 r_n=161888
2026-05-11 01:07:46,362 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:07:46,363 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:07:46,363 INFO train_multi TF=ALL: new best val=2.0726 — saved
2026-05-11 01:08:04,385 INFO train_multi TF=ALL epoch 37/100 train=2.0519 val=2.0831 r_mae=0.845 pos_r_acc=0.650 side_acc=0.639 r_n=161888
2026-05-11 01:08:22,485 INFO train_multi TF=ALL epoch 38/100 train=2.0419 val=2.0636 r_mae=0.833 pos_r_acc=0.660 side_acc=0.645 r_n=161888
2026-05-11 01:08:22,496 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:08:22,496 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:08:22,496 INFO train_multi TF=ALL: new best val=2.0636 — saved
2026-05-11 01:08:40,493 INFO train_multi TF=ALL epoch 39/100 train=2.0307 val=2.0612 r_mae=0.834 pos_r_acc=0.658 side_acc=0.647 r_n=161888
2026-05-11 01:08:40,503 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:08:40,503 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:08:40,504 INFO train_multi TF=ALL: new best val=2.0612 — saved
2026-05-11 01:08:58,379 INFO train_multi TF=ALL epoch 40/100 train=2.0185 val=2.0479 r_mae=0.819 pos_r_acc=0.669 side_acc=0.650 r_n=161888
2026-05-11 01:08:58,384 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:08:58,384 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:08:58,384 INFO train_multi TF=ALL: new best val=2.0479 — saved
2026-05-11 01:09:16,308 INFO train_multi TF=ALL epoch 41/100 train=2.0101 val=2.0418 r_mae=0.824 pos_r_acc=0.667 side_acc=0.652 r_n=161888
2026-05-11 01:09:16,313 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:09:16,313 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:09:16,313 INFO train_multi TF=ALL: new best val=2.0418 — saved
2026-05-11 01:09:34,320 INFO train_multi TF=ALL epoch 42/100 train=2.0006 val=2.0363 r_mae=0.819 pos_r_acc=0.670 side_acc=0.654 r_n=161888
2026-05-11 01:09:34,331 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:09:34,331 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:09:34,331 INFO train_multi TF=ALL: new best val=2.0363 — saved
2026-05-11 01:09:52,176 INFO train_multi TF=ALL epoch 43/100 train=1.9902 val=2.0405 r_mae=0.816 pos_r_acc=0.669 side_acc=0.652 r_n=161888
2026-05-11 01:10:10,035 INFO train_multi TF=ALL epoch 44/100 train=1.9812 val=2.0268 r_mae=0.815 pos_r_acc=0.670 side_acc=0.657 r_n=161888
2026-05-11 01:10:10,040 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:10:10,041 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:10:10,041 INFO train_multi TF=ALL: new best val=2.0268 — saved
2026-05-11 01:10:27,940 INFO train_multi TF=ALL epoch 45/100 train=1.9733 val=2.0222 r_mae=0.816 pos_r_acc=0.671 side_acc=0.662 r_n=161888
2026-05-11 01:10:27,952 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:10:27,952 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:10:27,952 INFO train_multi TF=ALL: new best val=2.0222 — saved
2026-05-11 01:10:45,812 INFO train_multi TF=ALL epoch 46/100 train=1.9598 val=2.0205 r_mae=0.805 pos_r_acc=0.672 side_acc=0.660 r_n=161888
2026-05-11 01:10:45,818 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:10:45,818 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:10:45,818 INFO train_multi TF=ALL: new best val=2.0205 — saved
2026-05-11 01:11:03,662 INFO train_multi TF=ALL epoch 47/100 train=1.9540 val=2.0017 r_mae=0.805 pos_r_acc=0.672 side_acc=0.667 r_n=161888
2026-05-11 01:11:03,667 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:11:03,667 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:11:03,667 INFO train_multi TF=ALL: new best val=2.0017 — saved
2026-05-11 01:11:21,563 INFO train_multi TF=ALL epoch 48/100 train=1.9476 val=2.0128 r_mae=0.807 pos_r_acc=0.669 side_acc=0.663 r_n=161888
2026-05-11 01:11:39,696 INFO train_multi TF=ALL epoch 49/100 train=1.9411 val=1.9920 r_mae=0.796 pos_r_acc=0.678 side_acc=0.671 r_n=161888
2026-05-11 01:11:39,706 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:11:39,706 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:11:39,706 INFO train_multi TF=ALL: new best val=1.9920 — saved
2026-05-11 01:11:57,684 INFO train_multi TF=ALL epoch 50/100 train=1.9275 val=1.9894 r_mae=0.795 pos_r_acc=0.674 side_acc=0.673 r_n=161888
2026-05-11 01:11:57,693 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:11:57,693 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:11:57,694 INFO train_multi TF=ALL: new best val=1.9894 — saved
2026-05-11 01:12:15,911 INFO train_multi TF=ALL epoch 51/100 train=1.9216 val=1.9838 r_mae=0.802 pos_r_acc=0.673 side_acc=0.674 r_n=161888
2026-05-11 01:12:15,921 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:12:15,921 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:12:15,921 INFO train_multi TF=ALL: new best val=1.9838 — saved
2026-05-11 01:12:33,897 INFO train_multi TF=ALL epoch 52/100 train=1.9104 val=1.9723 r_mae=0.789 pos_r_acc=0.681 side_acc=0.675 r_n=161888
2026-05-11 01:12:33,902 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:12:33,902 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:12:33,903 INFO train_multi TF=ALL: new best val=1.9723 — saved
2026-05-11 01:12:51,883 INFO train_multi TF=ALL epoch 53/100 train=1.9012 val=1.9661 r_mae=0.788 pos_r_acc=0.681 side_acc=0.680 r_n=161888
2026-05-11 01:12:51,888 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:12:51,888 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:12:51,888 INFO train_multi TF=ALL: new best val=1.9661 — saved
2026-05-11 01:13:09,959 INFO train_multi TF=ALL epoch 54/100 train=1.8965 val=1.9683 r_mae=0.783 pos_r_acc=0.683 side_acc=0.676 r_n=161888
2026-05-11 01:13:28,021 INFO train_multi TF=ALL epoch 55/100 train=1.8894 val=1.9507 r_mae=0.789 pos_r_acc=0.682 side_acc=0.685 r_n=161888
2026-05-11 01:13:28,031 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:13:28,031 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:13:28,031 INFO train_multi TF=ALL: new best val=1.9507 — saved
2026-05-11 01:13:46,341 INFO train_multi TF=ALL epoch 56/100 train=1.8803 val=1.9572 r_mae=0.787 pos_r_acc=0.680 side_acc=0.684 r_n=161888
2026-05-11 01:14:04,450 INFO train_multi TF=ALL epoch 57/100 train=1.8715 val=1.9523 r_mae=0.790 pos_r_acc=0.677 side_acc=0.687 r_n=161888
2026-05-11 01:14:22,717 INFO train_multi TF=ALL epoch 58/100 train=1.8606 val=1.9340 r_mae=0.781 pos_r_acc=0.685 side_acc=0.694 r_n=161888
2026-05-11 01:14:22,722 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:14:22,722 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:14:22,722 INFO train_multi TF=ALL: new best val=1.9340 — saved
2026-05-11 01:14:40,911 INFO train_multi TF=ALL epoch 59/100 train=1.8574 val=1.9245 r_mae=0.784 pos_r_acc=0.684 side_acc=0.698 r_n=161888
2026-05-11 01:14:40,921 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:14:40,921 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:14:40,922 INFO train_multi TF=ALL: new best val=1.9245 — saved
2026-05-11 01:14:58,887 INFO train_multi TF=ALL epoch 60/100 train=1.8539 val=1.9323 r_mae=0.780 pos_r_acc=0.686 side_acc=0.693 r_n=161888
2026-05-11 01:15:16,996 INFO train_multi TF=ALL epoch 61/100 train=1.8421 val=1.9159 r_mae=0.775 pos_r_acc=0.688 side_acc=0.700 r_n=161888
2026-05-11 01:15:17,001 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:15:17,002 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:15:17,002 INFO train_multi TF=ALL: new best val=1.9159 — saved
2026-05-11 01:15:35,008 INFO train_multi TF=ALL epoch 62/100 train=1.8354 val=1.9099 r_mae=0.782 pos_r_acc=0.685 side_acc=0.704 r_n=161888
2026-05-11 01:15:35,013 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:15:35,013 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:15:35,014 INFO train_multi TF=ALL: new best val=1.9099 — saved
2026-05-11 01:15:53,077 INFO train_multi TF=ALL epoch 63/100 train=1.8311 val=1.9061 r_mae=0.775 pos_r_acc=0.688 side_acc=0.707 r_n=161888
2026-05-11 01:15:53,082 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:15:53,082 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:15:53,082 INFO train_multi TF=ALL: new best val=1.9061 — saved
2026-05-11 01:16:10,986 INFO train_multi TF=ALL epoch 64/100 train=1.8211 val=1.9008 r_mae=0.779 pos_r_acc=0.684 side_acc=0.711 r_n=161888
2026-05-11 01:16:10,991 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:16:10,991 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:16:10,991 INFO train_multi TF=ALL: new best val=1.9008 — saved
2026-05-11 01:16:28,940 INFO train_multi TF=ALL epoch 65/100 train=1.8144 val=1.8973 r_mae=0.780 pos_r_acc=0.685 side_acc=0.713 r_n=161888
2026-05-11 01:16:28,945 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:16:28,945 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:16:28,945 INFO train_multi TF=ALL: new best val=1.8973 — saved
2026-05-11 01:16:47,281 INFO train_multi TF=ALL epoch 66/100 train=1.8089 val=1.8985 r_mae=0.770 pos_r_acc=0.689 side_acc=0.716 r_n=161888
2026-05-11 01:17:05,500 INFO train_multi TF=ALL epoch 67/100 train=1.8013 val=1.8845 r_mae=0.777 pos_r_acc=0.688 side_acc=0.717 r_n=161888
2026-05-11 01:17:05,510 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:17:05,510 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:17:05,510 INFO train_multi TF=ALL: new best val=1.8845 — saved
2026-05-11 01:17:23,619 INFO train_multi TF=ALL epoch 68/100 train=1.7970 val=1.8856 r_mae=0.785 pos_r_acc=0.682 side_acc=0.723 r_n=161888
2026-05-11 01:17:41,519 INFO train_multi TF=ALL epoch 69/100 train=1.7894 val=1.8809 r_mae=0.774 pos_r_acc=0.688 side_acc=0.724 r_n=161888
2026-05-11 01:17:41,524 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:17:41,524 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:17:41,524 INFO train_multi TF=ALL: new best val=1.8809 — saved
2026-05-11 01:17:59,517 INFO train_multi TF=ALL epoch 70/100 train=1.7782 val=1.8717 r_mae=0.772 pos_r_acc=0.691 side_acc=0.727 r_n=161888
2026-05-11 01:17:59,523 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:17:59,523 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:17:59,523 INFO train_multi TF=ALL: new best val=1.8717 — saved
2026-05-11 01:18:17,462 INFO train_multi TF=ALL epoch 71/100 train=1.7735 val=1.8721 r_mae=0.774 pos_r_acc=0.688 side_acc=0.728 r_n=161888
2026-05-11 01:18:35,381 INFO train_multi TF=ALL epoch 72/100 train=1.7666 val=1.8773 r_mae=0.772 pos_r_acc=0.687 side_acc=0.729 r_n=161888
2026-05-11 01:18:53,642 INFO train_multi TF=ALL epoch 73/100 train=1.7642 val=1.8554 r_mae=0.771 pos_r_acc=0.692 side_acc=0.736 r_n=161888
2026-05-11 01:18:53,648 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:18:53,648 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:18:53,648 INFO train_multi TF=ALL: new best val=1.8554 — saved
2026-05-11 01:19:11,411 INFO train_multi TF=ALL epoch 74/100 train=1.7565 val=1.8756 r_mae=0.770 pos_r_acc=0.687 side_acc=0.732 r_n=161888
2026-05-11 01:19:29,277 INFO train_multi TF=ALL epoch 75/100 train=1.7553 val=1.8607 r_mae=0.773 pos_r_acc=0.688 side_acc=0.736 r_n=161888
2026-05-11 01:19:47,250 INFO train_multi TF=ALL epoch 76/100 train=1.7445 val=1.8810 r_mae=0.774 pos_r_acc=0.684 side_acc=0.734 r_n=161888
2026-05-11 01:20:05,100 INFO train_multi TF=ALL epoch 77/100 train=1.7402 val=1.8588 r_mae=0.774 pos_r_acc=0.687 side_acc=0.738 r_n=161888
2026-05-11 01:20:22,894 INFO train_multi TF=ALL epoch 78/100 train=1.7330 val=1.8524 r_mae=0.770 pos_r_acc=0.691 side_acc=0.739 r_n=161888
2026-05-11 01:20:22,900 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:20:22,900 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:20:22,900 INFO train_multi TF=ALL: new best val=1.8524 — saved
2026-05-11 01:20:40,717 INFO train_multi TF=ALL epoch 79/100 train=1.7276 val=1.8701 r_mae=0.774 pos_r_acc=0.687 side_acc=0.736 r_n=161888
2026-05-11 01:20:58,598 INFO train_multi TF=ALL epoch 80/100 train=1.7220 val=1.8646 r_mae=0.774 pos_r_acc=0.686 side_acc=0.740 r_n=161888
2026-05-11 01:21:16,489 INFO train_multi TF=ALL epoch 81/100 train=1.7149 val=1.8637 r_mae=0.769 pos_r_acc=0.687 side_acc=0.738 r_n=161888
2026-05-11 01:21:34,557 INFO train_multi TF=ALL epoch 82/100 train=1.7106 val=1.8623 r_mae=0.770 pos_r_acc=0.688 side_acc=0.741 r_n=161888
2026-05-11 01:21:52,547 INFO train_multi TF=ALL epoch 83/100 train=1.7076 val=1.8772 r_mae=0.772 pos_r_acc=0.685 side_acc=0.738 r_n=161888
2026-05-11 01:22:10,729 INFO train_multi TF=ALL epoch 84/100 train=1.6970 val=1.8659 r_mae=0.775 pos_r_acc=0.685 side_acc=0.741 r_n=161888
2026-05-11 01:22:28,699 INFO train_multi TF=ALL epoch 85/100 train=1.6911 val=1.8711 r_mae=0.776 pos_r_acc=0.683 side_acc=0.740 r_n=161888
2026-05-11 01:22:46,820 INFO train_multi TF=ALL epoch 86/100 train=1.6865 val=1.8748 r_mae=0.773 pos_r_acc=0.684 side_acc=0.738 r_n=161888
2026-05-11 01:23:04,965 INFO train_multi TF=ALL epoch 87/100 train=1.6824 val=1.8716 r_mae=0.772 pos_r_acc=0.685 side_acc=0.743 r_n=161888
2026-05-11 01:23:22,961 INFO train_multi TF=ALL epoch 88/100 train=1.6741 val=1.8809 r_mae=0.777 pos_r_acc=0.682 side_acc=0.742 r_n=161888
2026-05-11 01:23:40,873 INFO train_multi TF=ALL epoch 89/100 train=1.6722 val=1.8758 r_mae=0.773 pos_r_acc=0.683 side_acc=0.743 r_n=161888
2026-05-11 01:23:58,876 INFO train_multi TF=ALL epoch 90/100 train=1.6656 val=1.8779 r_mae=0.771 pos_r_acc=0.684 side_acc=0.744 r_n=161888
2026-05-11 01:24:16,925 INFO train_multi TF=ALL epoch 91/100 train=1.6580 val=1.8736 r_mae=0.774 pos_r_acc=0.684 side_acc=0.744 r_n=161888
2026-05-11 01:24:35,047 INFO train_multi TF=ALL epoch 92/100 train=1.6523 val=1.8777 r_mae=0.776 pos_r_acc=0.682 side_acc=0.743 r_n=161888
2026-05-11 01:24:53,301 INFO train_multi TF=ALL epoch 93/100 train=1.6435 val=1.8911 r_mae=0.774 pos_r_acc=0.684 side_acc=0.740 r_n=161888
2026-05-11 01:25:11,338 INFO train_multi TF=ALL epoch 94/100 train=1.6424 val=1.8814 r_mae=0.778 pos_r_acc=0.681 side_acc=0.744 r_n=161888
2026-05-11 01:25:29,354 INFO train_multi TF=ALL epoch 95/100 train=1.6376 val=1.8899 r_mae=0.771 pos_r_acc=0.685 side_acc=0.744 r_n=161888
2026-05-11 01:25:47,262 INFO train_multi TF=ALL epoch 96/100 train=1.6307 val=1.8963 r_mae=0.777 pos_r_acc=0.682 side_acc=0.743 r_n=161888
2026-05-11 01:26:05,239 INFO train_multi TF=ALL epoch 97/100 train=1.6243 val=1.9037 r_mae=0.776 pos_r_acc=0.681 side_acc=0.739 r_n=161888
2026-05-11 01:26:23,266 INFO train_multi TF=ALL epoch 98/100 train=1.6218 val=1.8858 r_mae=0.778 pos_r_acc=0.682 side_acc=0.744 r_n=161888
2026-05-11 01:26:41,335 INFO train_multi TF=ALL epoch 99/100 train=1.6121 val=1.9016 r_mae=0.777 pos_r_acc=0.681 side_acc=0.742 r_n=161888
2026-05-11 01:26:59,295 INFO train_multi TF=ALL epoch 100/100 train=1.6111 val=1.9094 r_mae=0.780 pos_r_acc=0.680 side_acc=0.740 r_n=161888
2026-05-11 01:27:00,928 INFO fit_r_isotonic: saved side-R calibrators stats={'long': {'n': 81464, 'raw_mae': 0.7853183090405267, 'calibrated_mae': 0.7900878220364091}, 'short': {'n': 80424, 'raw_mae': 0.7680274550479984, 'calibrated_mae': 0.7837425997990446}}
2026-05-11 01:27:01,063 WARNING GRU validation R-MAE above floor for TF=ALL: best_val_r_mae=0.770 max=0.750. Keeping saved best weights so the pipeline can progress.
2026-05-11 01:27:01,076 INFO GRU R threshold XAUUSD/buy: q25_pos=0.535 q50_pos=1.667 pos_rate=46.0% (n=119259 n_pos=54899)
2026-05-11 01:27:01,083 INFO GRU R threshold XAUUSD/sell: q25_pos=0.528 q50_pos=1.667 pos_rate=45.5% (n=118087 n_pos=53734)
2026-05-11 01:27:01,089 INFO GRU R threshold EURUSD/buy: q25_pos=0.541 q50_pos=1.667 pos_rate=45.6% (n=118751 n_pos=54196)
2026-05-11 01:27:01,095 INFO GRU R threshold EURUSD/sell: q25_pos=0.545 q50_pos=1.667 pos_rate=45.6% (n=117679 n_pos=53721)
2026-05-11 01:27:01,101 INFO GRU R threshold USDJPY/buy: q25_pos=0.551 q50_pos=1.667 pos_rate=46.2% (n=118691 n_pos=54870)
2026-05-11 01:27:01,107 INFO GRU R threshold USDJPY/sell: q25_pos=0.542 q50_pos=1.667 pos_rate=44.6% (n=116030 n_pos=51793)
2026-05-11 01:27:01,113 INFO GRU R threshold EURJPY/buy: q25_pos=0.530 q50_pos=1.416 pos_rate=46.1% (n=118651 n_pos=54654)
2026-05-11 01:27:01,118 INFO GRU R threshold EURJPY/sell: q25_pos=0.525 q50_pos=1.667 pos_rate=44.7% (n=117320 n_pos=52399)
2026-05-11 01:27:01,124 INFO GRU R threshold GBPJPY/buy: q25_pos=0.517 q50_pos=1.443 pos_rate=45.9% (n=118277 n_pos=54278)
2026-05-11 01:27:01,130 INFO GRU R threshold GBPJPY/sell: q25_pos=0.521 q50_pos=1.667 pos_rate=45.3% (n=116299 n_pos=52702)
2026-05-11 01:27:01,137 INFO GRU R threshold GBPUSD/buy: q25_pos=0.524 q50_pos=1.667 pos_rate=45.8% (n=117764 n_pos=53894)
2026-05-11 01:27:01,143 INFO GRU R threshold GBPUSD/sell: q25_pos=0.533 q50_pos=1.667 pos_rate=45.8% (n=117404 n_pos=53785)
2026-05-11 01:27:01,144 INFO GRU per-symbol R thresholds saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/symbol_r_thresholds.json (6 symbols)
2026-05-11 01:27:01,180 INFO Retrain complete. Total wall-clock: 1831.5s
2026-05-11 01:27:05,764 INFO Model gru: SUCCESS
2026-05-11 01:27:05,765 INFO --- Training regime ---
2026-05-11 01:27:05,765 INFO Running retrain --model regime
2026-05-11 01:27:06,195 INFO retrain environment: KAGGLE
2026-05-11 01:27:07,861 INFO Device: CUDA (2 GPU(s))
2026-05-11 01:27:07,873 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 01:27:07,873 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 01:27:07,873 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 01:27:07,873 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 01:27:07,873 INFO Retrain data split: train
2026-05-11 01:27:07,874 INFO Retrain rolling fold selector: latest
2026-05-11 01:27:07,875 INFO === RegimeClassifier retrain (hierarchical: HTF 2-score bias + LTF 5-score behaviour) ===
2026-05-11 01:27:08,115 INFO NumExpr defaulting to 4 threads.
2026-05-11 01:27:08,341 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-11 01:27:08,341 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 01:27:08,341 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 01:27:08,342 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-11 01:27:08,401 INFO Regime rolling folds selected: [None]
2026-05-11 01:27:08,401 INFO === Regime rolling fold 1/1: train_all ===
2026-05-11 01:27:08,401 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-11 01:27:08,445 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 01:27:08,447 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:27:08,464 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:27:08,481 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:27:08,498 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:27:08,517 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:27:08,532 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:27:08,779 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:27:08,849 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:27:08,873 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:27:08,874 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:27:08,885 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:27:08,886 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:27:09,492 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 550, 'BIAS_DOWN': 255, 'BIAS_NEUTRAL': 11297}  ambiguous=7029 (total=12102) horizon=36
2026-05-11 01:27:09,500 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected XAUUSD — 12052 samples (group=gold) score_means={'bias_up_score': 0.0456, 'bias_down_score': 0.0212} labels={'BIAS_UP': 550, 'BIAS_DOWN': 255, 'BIAS_NEUTRAL': 11247} clean={'BIAS_UP': 550, 'BIAS_DOWN': 255, 'BIAS_NEUTRAL': 4250}
2026-05-11 01:27:09,675 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:09,713 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:09,732 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:09,732 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:09,740 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:09,742 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:27:10,316 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 299, 'BIAS_DOWN': 288, 'BIAS_NEUTRAL': 10817}  ambiguous=6558 (total=11404) horizon=36
2026-05-11 01:27:10,321 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURUSD — 11354 samples (group=dollar) score_means={'bias_up_score': 0.0263, 'bias_down_score': 0.0254} labels={'BIAS_UP': 299, 'BIAS_DOWN': 288, 'BIAS_NEUTRAL': 10767} clean={'BIAS_UP': 299, 'BIAS_DOWN': 288, 'BIAS_NEUTRAL': 4231}
2026-05-11 01:27:10,484 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:10,520 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:10,541 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:10,541 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:10,549 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:10,550 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:27:11,076 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 431, 'BIAS_DOWN': 164, 'BIAS_NEUTRAL': 10808}  ambiguous=6695 (total=11403) horizon=36
2026-05-11 01:27:11,081 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected USDJPY — 11353 samples (group=dollar) score_means={'bias_up_score': 0.038, 'bias_down_score': 0.0144} labels={'BIAS_UP': 431, 'BIAS_DOWN': 164, 'BIAS_NEUTRAL': 10758} clean={'BIAS_UP': 431, 'BIAS_DOWN': 164, 'BIAS_NEUTRAL': 4092}
2026-05-11 01:27:11,231 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:11,266 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:11,289 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:11,289 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:11,297 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:11,298 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:27:11,856 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 313, 'BIAS_DOWN': 162, 'BIAS_NEUTRAL': 10932}  ambiguous=6806 (total=11407) horizon=36
2026-05-11 01:27:11,862 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURJPY — 11357 samples (group=cross) score_means={'bias_up_score': 0.0276, 'bias_down_score': 0.0143} labels={'BIAS_UP': 313, 'BIAS_DOWN': 162, 'BIAS_NEUTRAL': 10882} clean={'BIAS_UP': 313, 'BIAS_DOWN': 162, 'BIAS_NEUTRAL': 4094}
2026-05-11 01:27:12,016 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:12,052 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:12,073 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:12,074 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:12,083 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:12,084 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:27:12,613 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 467, 'BIAS_DOWN': 285, 'BIAS_NEUTRAL': 10656}  ambiguous=6835 (total=11408) horizon=36
2026-05-11 01:27:12,618 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPJPY — 11358 samples (group=cross) score_means={'bias_up_score': 0.0411, 'bias_down_score': 0.0251} labels={'BIAS_UP': 467, 'BIAS_DOWN': 285, 'BIAS_NEUTRAL': 10606} clean={'BIAS_UP': 467, 'BIAS_DOWN': 285, 'BIAS_NEUTRAL': 3818}
2026-05-11 01:27:12,773 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:12,811 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:12,830 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:12,831 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:12,839 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:12,840 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:27:13,361 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 255, 'BIAS_DOWN': 300, 'BIAS_NEUTRAL': 10847}  ambiguous=6860 (total=11402) horizon=36
2026-05-11 01:27:13,366 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPUSD — 11352 samples (group=dollar) score_means={'bias_up_score': 0.0225, 'bias_down_score': 0.0264} labels={'BIAS_UP': 255, 'BIAS_DOWN': 300, 'BIAS_NEUTRAL': 10797} clean={'BIAS_UP': 255, 'BIAS_DOWN': 300, 'BIAS_NEUTRAL': 3987}
2026-05-11 01:27:13,437 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 780, 'BIAS_DOWN': 447, 'BIAS_NEUTRAL': 21488}, 'dollar': {'BIAS_UP': 985, 'BIAS_DOWN': 752, 'BIAS_NEUTRAL': 32322}, 'gold': {'BIAS_UP': 550, 'BIAS_DOWN': 255, 'BIAS_NEUTRAL': 11247}}
2026-05-11 01:27:13,437 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0343, 'bias_down_score': 0.0197}, 'dollar': {'bias_up_score': 0.0289, 'bias_down_score': 0.0221}, 'gold': {'bias_up_score': 0.0456, 'bias_down_score': 0.0212}}
2026-05-11 01:27:13,437 INFO Regime[4H mode=htf_bias] label distribution by year: {2016: {'BIAS_UP': 274, 'BIAS_DOWN': 351, 'BIAS_NEUTRAL': 8197}, 2017: {'BIAS_UP': 495, 'BIAS_DOWN': 167, 'BIAS_NEUTRAL': 8451}, 2018: {'BIAS_UP': 208, 'BIAS_DOWN': 250, 'BIAS_NEUTRAL': 8672}, 2019: {'BIAS_UP': 189, 'BIAS_DOWN': 174, 'BIAS_NEUTRAL': 8739}, 2020: {'BIAS_UP': 285, 'BIAS_DOWN': 131, 'BIAS_NEUTRAL': 8695}, 2021: {'BIAS_UP': 311, 'BIAS_DOWN': 163, 'BIAS_NEUTRAL': 8617}, 2022: {'BIAS_UP': 340, 'BIAS_DOWN': 162, 'BIAS_NEUTRAL': 8619}, 2023: {'BIAS_UP': 213, 'BIAS_DOWN': 56, 'BIAS_NEUTRAL': 5067}}
2026-05-11 01:27:13,437 INFO Regime[4H mode=htf_bias] score means by year: {2016: {'bias_up_score': 0.0311, 'bias_down_score': 0.0398}, 2017: {'bias_up_score': 0.0543, 'bias_down_score': 0.0183}, 2018: {'bias_up_score': 0.0228, 'bias_down_score': 0.0274}, 2019: {'bias_up_score': 0.0208, 'bias_down_score': 0.0191}, 2020: {'bias_up_score': 0.0313, 'bias_down_score': 0.0144}, 2021: {'bias_up_score': 0.0342, 'bias_down_score': 0.0179}, 2022: {'bias_up_score': 0.0373, 'bias_down_score': 0.0178}, 2023: {'bias_up_score': 0.0399, 'bias_down_score': 0.0105}}
2026-05-11 01:27:13,492 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
2026-05-11 01:27:13,493 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
2026-05-11 01:27:13,494 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
2026-05-11 01:27:13,495 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 01:27:13,495 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 01:27:13,496 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
2026-05-11 01:27:13,512 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:27:13,516 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:27:13,517 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:27:13,518 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:27:13,518 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:27:13,519 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:27:13,848 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 53, 'BIAS_DOWN': 23, 'BIAS_NEUTRAL': 1505}  ambiguous=929 (total=1581) horizon=36
2026-05-11 01:27:13,850 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected XAUUSD — 1531 samples (group=gold) score_means={'bias_up_score': 0.0346, 'bias_down_score': 0.015} labels={'BIAS_UP': 53, 'BIAS_DOWN': 23, 'BIAS_NEUTRAL': 1455} clean={'BIAS_UP': 53, 'BIAS_DOWN': 23, 'BIAS_NEUTRAL': 558}
2026-05-11 01:27:13,923 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:13,925 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:13,926 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:13,927 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:13,927 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:13,928 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:27:14,242 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 18, 'BIAS_DOWN': 38, 'BIAS_NEUTRAL': 1435}  ambiguous=844 (total=1491) horizon=36
2026-05-11 01:27:14,245 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURUSD — 1441 samples (group=dollar) score_means={'bias_up_score': 0.0125, 'bias_down_score': 0.0264} labels={'BIAS_UP': 18, 'BIAS_DOWN': 38, 'BIAS_NEUTRAL': 1385} clean={'BIAS_UP': 18, 'BIAS_DOWN': 38, 'BIAS_NEUTRAL': 577}
2026-05-11 01:27:14,315 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:14,317 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:14,318 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:14,319 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:14,319 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:14,320 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:27:14,625 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 95, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 1386}  ambiguous=905 (total=1489) horizon=36
2026-05-11 01:27:14,628 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected USDJPY — 1439 samples (group=dollar) score_means={'bias_up_score': 0.066, 'bias_down_score': 0.0056} labels={'BIAS_UP': 95, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 1336} clean={'BIAS_UP': 95, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 478}
2026-05-11 01:27:14,696 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:14,699 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:14,699 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:14,700 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:14,700 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:14,701 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:27:15,007 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 52, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 1442}  ambiguous=913 (total=1494) horizon=36
2026-05-11 01:27:15,009 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.036, 'bias_down_score': 0.0} labels={'BIAS_UP': 52, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 1392} clean={'BIAS_UP': 52, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 510}
2026-05-11 01:27:15,077 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:15,079 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:15,080 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:15,081 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:15,081 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:15,082 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:27:15,384 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 40, 'BIAS_DOWN': 3, 'BIAS_NEUTRAL': 1451}  ambiguous=884 (total=1494) horizon=36
2026-05-11 01:27:15,386 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0277, 'bias_down_score': 0.0021} labels={'BIAS_UP': 40, 'BIAS_DOWN': 3, 'BIAS_NEUTRAL': 1401} clean={'BIAS_UP': 40, 'BIAS_DOWN': 3, 'BIAS_NEUTRAL': 542}
2026-05-11 01:27:15,454 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:15,457 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:15,457 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:15,458 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:15,458 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:27:15,459 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:27:15,770 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 18, 'BIAS_DOWN': 29, 'BIAS_NEUTRAL': 1441}  ambiguous=896 (total=1488) horizon=36
2026-05-11 01:27:15,773 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPUSD — 1438 samples (group=dollar) score_means={'bias_up_score': 0.0125, 'bias_down_score': 0.0202} labels={'BIAS_UP': 18, 'BIAS_DOWN': 29, 'BIAS_NEUTRAL': 1391} clean={'BIAS_UP': 18, 'BIAS_DOWN': 29, 'BIAS_NEUTRAL': 533}
2026-05-11 01:27:15,838 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 92, 'BIAS_DOWN': 3, 'BIAS_NEUTRAL': 2793}, 'dollar': {'BIAS_UP': 131, 'BIAS_DOWN': 75, 'BIAS_NEUTRAL': 4112}, 'gold': {'BIAS_UP': 53, 'BIAS_DOWN': 23, 'BIAS_NEUTRAL': 1455}}
2026-05-11 01:27:15,838 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0319, 'bias_down_score': 0.001}, 'dollar': {'bias_up_score': 0.0303, 'bias_down_score': 0.0174}, 'gold': {'bias_up_score': 0.0346, 'bias_down_score': 0.015}}
2026-05-11 01:27:15,838 INFO Regime[4H mode=htf_bias] label distribution by year: {2022: {'BIAS_UP': 72, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 3283}, 2023: {'BIAS_UP': 204, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 5077}}
2026-05-11 01:27:15,838 INFO Regime[4H mode=htf_bias] score means by year: {2022: {'bias_up_score': 0.0212, 'bias_down_score': 0.0135}, 2023: {'bias_up_score': 0.0382, 'bias_down_score': 0.0103}}
2026-05-11 01:27:15,882 INFO Regime phase HTF dataset build fold=train_all: 7.5s (train=68826 val=8737)
2026-05-11 01:27:15,882 INFO Regime 4H/htf_bias cold start: no existing weights found
2026-05-11 01:27:15,888 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=68826 val=8737 train_labels={'BIAS_UP': 2315, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 65057} val_labels={'BIAS_UP': 276, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 8360}
2026-05-11 01:27:16,093 INFO RegimeClassifier[mode=htf_bias]: cold start HTF score head
2026-05-11 01:27:16,094 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-11 01:27:16,094 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 28.73, 'bias_down_score': 30.0}
2026-05-11 01:27:16,098 INFO RegimeClassifier[mode=htf_bias]: HTF balanced sampler — dir=3769 neutral=65057 dir_weight=15 => dir_frac_per_epoch≈46.5%
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../models/regime_classifier.py:2349: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  scheduler.step()
2026-05-11 01:27:19,987 INFO Regime HTF score epoch  1/50 — tr=38.4610 va=2.2569 acc=0.957 bal=0.333 threshold=0.60 margin=0.10 recall={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 1.0} precision={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 0.957}
2026-05-11 01:27:21,616 INFO Regime HTF score epoch  2/50 — tr=37.5162 va=2.1739 bal=0.333
2026-05-11 01:27:23,247 INFO Regime HTF score epoch  3/50 — tr=35.3394 va=1.9961 bal=0.333
2026-05-11 01:27:24,876 INFO Regime HTF score epoch  4/50 — tr=31.1785 va=1.6959 bal=0.333
2026-05-11 01:27:26,504 INFO Regime HTF score epoch  5/50 — tr=25.7030 va=1.3087 acc=0.954 bal=0.336 threshold=0.40 margin=0.00 recall={'BIAS_UP': 0.011, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 0.996} precision={'BIAS_UP': 0.125, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 0.957}
2026-05-11 01:27:28,113 INFO Regime HTF score epoch  6/50 — tr=19.6462 va=0.9427 bal=0.518
2026-05-11 01:27:29,808 INFO Regime HTF score epoch  7/50 — tr=14.9478 va=0.7436 bal=0.606
2026-05-11 01:27:31,448 INFO Regime HTF score epoch  8/50 — tr=11.1464 va=0.6001 bal=0.626
2026-05-11 01:27:33,078 INFO Regime HTF score epoch  9/50 — tr=8.0430 va=0.5095 bal=0.573
2026-05-11 01:27:34,712 INFO Regime HTF score epoch 10/50 — tr=5.6200 va=0.4514 acc=0.884 bal=0.716 threshold=0.75 margin=0.00 recall={'BIAS_UP': 0.609, 'BIAS_DOWN': 0.644, 'BIAS_NEUTRAL': 0.896} precision={'BIAS_UP': 0.274, 'BIAS_DOWN': 0.132, 'BIAS_NEUTRAL': 0.981}
2026-05-11 01:27:36,350 INFO Regime HTF score epoch 11/50 — tr=3.7350 va=0.4289 bal=0.705
2026-05-11 01:27:37,980 INFO Regime HTF score epoch 12/50 — tr=2.4875 va=0.4415 bal=0.678
2026-05-11 01:27:39,685 INFO Regime HTF score epoch 13/50 — tr=1.6143 va=0.4756 bal=0.601
2026-05-11 01:27:41,341 INFO Regime HTF score epoch 14/50 — tr=1.0178 va=0.5208 bal=0.502
2026-05-11 01:27:43,021 INFO Regime HTF score epoch 15/50 — tr=0.6805 va=0.5894 acc=0.902 bal=0.627 threshold=0.99 margin=0.00 recall={'BIAS_UP': 0.344, 'BIAS_DOWN': 0.614, 'BIAS_NEUTRAL': 0.924} precision={'BIAS_UP': 0.23, 'BIAS_DOWN': 0.162, 'BIAS_NEUTRAL': 0.972}
2026-05-11 01:27:44,675 INFO Regime HTF score epoch 16/50 — tr=0.5083 va=0.6418 bal=0.725
2026-05-11 01:27:46,340 INFO Regime HTF score epoch 17/50 — tr=0.4254 va=0.6678 bal=0.776
2026-05-11 01:27:47,986 INFO Regime HTF score epoch 18/50 — tr=0.3825 va=0.6785 bal=0.801
2026-05-11 01:27:49,642 INFO Regime HTF score epoch 19/50 — tr=0.3648 va=0.7006 bal=0.860
2026-05-11 01:27:51,304 INFO Regime HTF score epoch 20/50 — tr=0.3545 va=0.7037 acc=0.871 bal=0.868 threshold=0.99 margin=0.00 recall={'BIAS_UP': 0.862, 'BIAS_DOWN': 0.871, 'BIAS_NEUTRAL': 0.872} precision={'BIAS_UP': 0.279, 'BIAS_DOWN': 0.162, 'BIAS_NEUTRAL': 0.993}
2026-05-11 01:27:52,984 INFO Regime HTF score epoch 21/50 — tr=0.3360 va=0.7033 bal=0.872
2026-05-11 01:27:54,640 INFO Regime HTF score epoch 22/50 — tr=0.3341 va=0.7128 bal=0.873
2026-05-11 01:27:56,280 INFO Regime HTF score epoch 23/50 — tr=0.3280 va=0.7031 bal=0.877
2026-05-11 01:27:57,923 INFO Regime HTF score epoch 24/50 — tr=0.3194 va=0.7097 bal=0.882
2026-05-11 01:27:59,576 INFO Regime HTF score epoch 25/50 — tr=0.3217 va=0.7053 acc=0.869 bal=0.879 threshold=0.99 margin=0.00 recall={'BIAS_UP': 0.888, 'BIAS_DOWN': 0.881, 'BIAS_NEUTRAL': 0.868} precision={'BIAS_UP': 0.274, 'BIAS_DOWN': 0.165, 'BIAS_NEUTRAL': 0.994}
2026-05-11 01:28:01,258 INFO Regime HTF score epoch 26/50 — tr=0.3221 va=0.7034 bal=0.884
2026-05-11 01:28:02,914 INFO Regime HTF score epoch 27/50 — tr=0.3077 va=0.7148 bal=0.887
2026-05-11 01:28:04,600 INFO Regime HTF score epoch 28/50 — tr=0.3185 va=0.7057 bal=0.890
2026-05-11 01:28:06,267 INFO Regime HTF score epoch 29/50 — tr=0.3122 va=0.7108 bal=0.889
2026-05-11 01:28:07,933 INFO Regime HTF score epoch 30/50 — tr=0.3095 va=0.7198 acc=0.866 bal=0.892 threshold=0.99 margin=0.00 recall={'BIAS_UP': 0.909, 'BIAS_DOWN': 0.901, 'BIAS_NEUTRAL': 0.864} precision={'BIAS_UP': 0.274, 'BIAS_DOWN': 0.163, 'BIAS_NEUTRAL': 0.995}
2026-05-11 01:28:09,616 INFO Regime HTF score epoch 31/50 — tr=0.3054 va=0.7058 bal=0.888
2026-05-11 01:28:11,300 INFO Regime HTF score epoch 32/50 — tr=0.3060 va=0.7090 bal=0.885
2026-05-11 01:28:12,960 INFO Regime HTF score epoch 33/50 — tr=0.2998 va=0.7118 bal=0.889
2026-05-11 01:28:14,624 INFO Regime HTF score epoch 34/50 — tr=0.2968 va=0.7091 bal=0.892
2026-05-11 01:28:16,284 INFO Regime HTF score epoch 35/50 — tr=0.2958 va=0.7140 acc=0.868 bal=0.892 threshold=0.99 margin=0.00 recall={'BIAS_UP': 0.909, 'BIAS_DOWN': 0.901, 'BIAS_NEUTRAL': 0.866} precision={'BIAS_UP': 0.277, 'BIAS_DOWN': 0.164, 'BIAS_NEUTRAL': 0.995}
2026-05-11 01:28:17,940 INFO Regime HTF score epoch 36/50 — tr=0.3001 va=0.7095 bal=0.893
2026-05-11 01:28:19,581 INFO Regime HTF score epoch 37/50 — tr=0.2891 va=0.7089 bal=0.893
2026-05-11 01:28:21,243 INFO Regime HTF score epoch 38/50 — tr=0.2899 va=0.7082 bal=0.894
2026-05-11 01:28:22,879 INFO Regime HTF score epoch 39/50 — tr=0.2840 va=0.7023 bal=0.890
2026-05-11 01:28:24,526 INFO Regime HTF score epoch 40/50 — tr=0.2971 va=0.7025 acc=0.871 bal=0.890 threshold=0.99 margin=0.00 recall={'BIAS_UP': 0.909, 'BIAS_DOWN': 0.891, 'BIAS_NEUTRAL': 0.87} precision={'BIAS_UP': 0.279, 'BIAS_DOWN': 0.17, 'BIAS_NEUTRAL': 0.995}
2026-05-11 01:28:26,161 INFO Regime HTF score epoch 41/50 — tr=0.2843 va=0.6943 bal=0.886
2026-05-11 01:28:27,813 INFO Regime HTF score epoch 42/50 — tr=0.2960 va=0.7072 bal=0.897
2026-05-11 01:28:29,473 INFO Regime HTF score epoch 43/50 — tr=0.2925 va=0.7133 bal=0.896
2026-05-11 01:28:31,164 INFO Regime HTF score epoch 44/50 — tr=0.2864 va=0.7098 bal=0.897
2026-05-11 01:28:32,798 INFO Regime HTF score epoch 45/50 — tr=0.2922 va=0.6972 acc=0.872 bal=0.891 threshold=0.99 margin=0.00 recall={'BIAS_UP': 0.902, 'BIAS_DOWN': 0.901, 'BIAS_NEUTRAL': 0.871} precision={'BIAS_UP': 0.279, 'BIAS_DOWN': 0.173, 'BIAS_NEUTRAL': 0.995}
2026-05-11 01:28:34,448 INFO Regime HTF score epoch 46/50 — tr=0.2880 va=0.7015 bal=0.892
2026-05-11 01:28:36,098 INFO Regime HTF score epoch 47/50 — tr=0.2933 va=0.6920 bal=0.888
2026-05-11 01:28:37,763 INFO Regime HTF score epoch 48/50 — tr=0.2830 va=0.6987 bal=0.893
2026-05-11 01:28:39,414 INFO Regime HTF score epoch 49/50 — tr=0.2903 va=0.7004 bal=0.892
2026-05-11 01:28:41,094 INFO Regime HTF score epoch 50/50 — tr=0.2900 va=0.7002 acc=0.872 bal=0.892 threshold=0.99 margin=0.00 recall={'BIAS_UP': 0.906, 'BIAS_DOWN': 0.901, 'BIAS_NEUTRAL': 0.87} precision={'BIAS_UP': 0.279, 'BIAS_DOWN': 0.171, 'BIAS_NEUTRAL': 0.995}
2026-05-11 01:28:42,639 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.990 margin=0.000 precision={'BIAS_UP': 0.279, 'BIAS_DOWN': 0.174, 'BIAS_NEUTRAL': 0.995} recall={'BIAS_UP': 0.909, 'BIAS_DOWN': 0.881, 'BIAS_NEUTRAL': 0.872} f1={'BIAS_UP': 0.427, 'BIAS_DOWN': 0.291, 'BIAS_NEUTRAL': 0.929} confusion=[[251, 0, 25], [0, 89, 12], [648, 422, 7290]] score_mae={'bias_up_score': 0.1882, 'bias_down_score': 0.1235} pred_share={'BIAS_UP': 0.1029, 'BIAS_DOWN': 0.0585, 'BIAS_NEUTRAL': 0.8386}
2026-05-11 01:28:42,640 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.279, 'BIAS_DOWN': 0.174, 'BIAS_NEUTRAL': 0.995} min_precision=0.500 recall={'BIAS_UP': 0.909, 'BIAS_DOWN': 0.881, 'BIAS_NEUTRAL': 0.872} min_recall=0.100 f1={'BIAS_UP': 0.427, 'BIAS_DOWN': 0.291, 'BIAS_NEUTRAL': 0.929} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_UP', 'BIAS_DOWN'] weak_recall=[] weak_f1=[] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-11 01:28:42,643 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 01:28:42,644 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 01:28:42,644 INFO Regime phase HTF train fold=train_all: 86.8s
2026-05-11 01:28:42,754 INFO Regime HTF complete fold=train_all: acc=0.873 bal=0.887 train=68826 val=8737 per_class={'BIAS_UP': 0.909, 'BIAS_DOWN': 0.881, 'BIAS_NEUTRAL': 0.872} precision={'BIAS_UP': 0.279, 'BIAS_DOWN': 0.174, 'BIAS_NEUTRAL': 0.995} threshold=0.990 margin=0.000
2026-05-11 01:28:42,755 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:28:42,913 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 255, 'BIAS_DOWN': 300, 'BIAS_NEUTRAL': 10847}  ambiguous=6860 (total=11402) horizon=36
2026-05-11 01:28:42,920 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on GBPUSD 4H:
{'BIAS_UP': 3.8636363636363638, 'BIAS_DOWN': 3.5294117647058822, 'BIAS_NEUTRAL': 71.36184210526316}
2026-05-11 01:28:42,924 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (all labels):
{'BIAS_UP': {'n': 255, 'mean': 0.0006942435718027296, 'mean_over_std': 0.3535182817741541}, 'BIAS_DOWN': {'n': 300, 'mean': -0.0007615786091842384, 'mean_over_std': -0.2905559984734911}, 'BIAS_NEUTRAL': {'n': 10846, 'mean': -5.264589798607426e-06, 'mean_over_std': -0.002007943010728463}}
2026-05-11 01:28:42,925 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 255, 'mean': 0.0006942435718027296, 'mean_over_std': 0.3535182817741541}, 'BIAS_DOWN': {'n': 300, 'mean': -0.0007615786091842384, 'mean_over_std': -0.2905559984734911}, 'BIAS_NEUTRAL': {'n': 3987, 'mean': -1.7489525578049586e-05, 'mean_over_std': -0.007235149795477735}}
2026-05-11 01:28:42,931 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-11 01:28:42,934 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:28:42,936 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:28:42,937 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:28:42,939 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:28:42,941 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:28:42,943 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:28:42,962 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:28:42,971 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:28:42,974 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:28:42,975 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:28:42,975 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:28:42,982 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:28:43,880 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected XAUUSD — 44227 samples (group=gold) score_means={'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}
2026-05-11 01:28:43,988 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:43,990 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:43,991 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:43,991 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:43,992 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:43,994 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:28:44,818 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURUSD — 43689 samples (group=dollar) score_means={'trend_score': 0.49, 'range_score': 0.236, 'chop_score': 0.4622, 'volatility_percentile': 0.3812, 'consolidation_score': 0.1862}
2026-05-11 01:28:44,926 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:44,928 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:44,929 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:44,929 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:44,930 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:44,932 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:28:45,759 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected USDJPY — 43686 samples (group=dollar) score_means={'trend_score': 0.4949, 'range_score': 0.2312, 'chop_score': 0.4613, 'volatility_percentile': 0.38, 'consolidation_score': 0.1931}
2026-05-11 01:28:45,868 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:45,870 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:45,871 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:45,871 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:45,872 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:45,874 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:28:46,696 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURJPY — 43688 samples (group=cross) score_means={'trend_score': 0.4888, 'range_score': 0.2345, 'chop_score': 0.4662, 'volatility_percentile': 0.3831, 'consolidation_score': 0.1882}
2026-05-11 01:28:46,810 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:46,812 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:46,813 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:46,813 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:46,814 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:46,816 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:28:47,640 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPJPY — 43678 samples (group=cross) score_means={'trend_score': 0.4878, 'range_score': 0.2348, 'chop_score': 0.4673, 'volatility_percentile': 0.3839, 'consolidation_score': 0.1904}
2026-05-11 01:28:47,754 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:47,756 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:47,757 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:47,758 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:47,758 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:47,761 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:28:48,608 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPUSD — 43676 samples (group=dollar) score_means={'trend_score': 0.4923, 'range_score': 0.2335, 'chop_score': 0.4611, 'volatility_percentile': 0.3832, 'consolidation_score': 0.1856}
2026-05-11 01:28:48,727 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2346, 'chop_score': 0.4667, 'volatility_percentile': 0.3835, 'consolidation_score': 0.1893}, 'dollar': {'trend_score': 0.4924, 'range_score': 0.2335, 'chop_score': 0.4615, 'volatility_percentile': 0.3815, 'consolidation_score': 0.1883}, 'gold': {'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}}
2026-05-11 01:28:48,727 INFO Regime[1H mode=ltf_behaviour] score means by year: {2016: {'trend_score': 0.495, 'range_score': 0.233, 'chop_score': 0.4607, 'volatility_percentile': 0.3776, 'consolidation_score': 0.192}, 2017: {'trend_score': 0.5017, 'range_score': 0.2328, 'chop_score': 0.4574, 'volatility_percentile': 0.3689, 'consolidation_score': 0.1942}, 2018: {'trend_score': 0.4886, 'range_score': 0.237, 'chop_score': 0.4655, 'volatility_percentile': 0.3947, 'consolidation_score': 0.1785}, 2019: {'trend_score': 0.4817, 'range_score': 0.236, 'chop_score': 0.4691, 'volatility_percentile': 0.3725, 'consolidation_score': 0.198}, 2020: {'trend_score': 0.4874, 'range_score': 0.2311, 'chop_score': 0.4674, 'volatility_percentile': 0.3848, 'consolidation_score': 0.1919}, 2021: {'trend_score': 0.4872, 'range_score': 0.2372, 'chop_score': 0.4672, 'volatility_percentile': 0.377, 'consolidation_score': 0.192}, 2022: {'trend_score': 0.4893, 'range_score': 0.2357, 'chop_score': 0.4647, 'volatility_percentile': 0.3962, 'consolidation_score': 0.1784}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 01:28:48,826 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
2026-05-11 01:28:48,828 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 01:28:48,829 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
2026-05-11 01:28:48,830 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 01:28:48,832 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
2026-05-11 01:28:48,833 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
2026-05-11 01:28:48,843 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:28:48,846 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:28:48,847 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:28:48,848 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:28:48,848 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:28:48,850 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:28:49,112 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected XAUUSD — 5117 samples (group=gold) score_means={'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}
2026-05-11 01:28:49,231 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:49,235 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:49,235 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:49,236 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:49,236 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:49,238 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:28:49,492 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURUSD — 5053 samples (group=dollar) score_means={'trend_score': 0.4883, 'range_score': 0.2413, 'chop_score': 0.463, 'volatility_percentile': 0.3837, 'consolidation_score': 0.1793}
2026-05-11 01:28:49,602 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:49,604 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:49,605 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:49,606 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:49,606 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:49,609 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:28:49,874 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected USDJPY — 5046 samples (group=dollar) score_means={'trend_score': 0.5085, 'range_score': 0.2259, 'chop_score': 0.4506, 'volatility_percentile': 0.3809, 'consolidation_score': 0.1888}
2026-05-11 01:28:49,987 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:49,990 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:49,990 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:49,991 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:49,991 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:49,993 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:28:50,226 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURJPY — 5053 samples (group=cross) score_means={'trend_score': 0.5023, 'range_score': 0.2261, 'chop_score': 0.4524, 'volatility_percentile': 0.3837, 'consolidation_score': 0.18}
2026-05-11 01:28:50,334 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:50,336 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:50,337 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:50,337 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:50,338 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:50,339 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:28:50,580 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPJPY — 5043 samples (group=cross) score_means={'trend_score': 0.4743, 'range_score': 0.2372, 'chop_score': 0.4716, 'volatility_percentile': 0.3777, 'consolidation_score': 0.1917}
2026-05-11 01:28:50,690 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:50,693 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:50,693 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:50,694 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:50,694 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:28:50,696 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:28:50,939 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPUSD — 5040 samples (group=dollar) score_means={'trend_score': 0.4843, 'range_score': 0.2371, 'chop_score': 0.4614, 'volatility_percentile': 0.377, 'consolidation_score': 0.1835}
2026-05-11 01:28:51,043 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2317, 'chop_score': 0.462, 'volatility_percentile': 0.3807, 'consolidation_score': 0.1858}, 'dollar': {'trend_score': 0.4937, 'range_score': 0.2348, 'chop_score': 0.4583, 'volatility_percentile': 0.3806, 'consolidation_score': 0.1839}, 'gold': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}}
2026-05-11 01:28:51,043 INFO Regime[1H mode=ltf_behaviour] score means by year: {2022: {'trend_score': 0.4905, 'range_score': 0.2364, 'chop_score': 0.4625, 'volatility_percentile': 0.385, 'consolidation_score': 0.1797}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 01:28:51,123 INFO Regime phase LTF dataset build fold=train_all: 8.2s (train=262644 val=30352)
2026-05-11 01:28:51,123 INFO Regime 1H/ltf_behaviour cold start: no existing weights found
2026-05-11 01:28:51,149 INFO RegimeClassifier[mode=ltf_behaviour]: cold start score head
2026-05-11 01:28:51,149 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-11 01:28:51,659 INFO Regime score epoch  1/50 — tr=0.0946 va=0.0782 mae={'trend_score': 0.1709, 'range_score': 0.2942, 'chop_score': 0.1651, 'volatility_percentile': 0.1843, 'consolidation_score': 0.3577}
2026-05-11 01:28:52,158 INFO Regime score epoch  2/50 — tr=0.0806 va=0.0616
2026-05-11 01:28:52,660 INFO Regime score epoch  3/50 — tr=0.0603 va=0.0421
2026-05-11 01:28:53,151 INFO Regime score epoch  4/50 — tr=0.0411 va=0.0288
2026-05-11 01:28:53,666 INFO Regime score epoch  5/50 — tr=0.0281 va=0.0189 mae={'trend_score': 0.0726, 'range_score': 0.1393, 'chop_score': 0.0641, 'volatility_percentile': 0.0545, 'consolidation_score': 0.1999}
2026-05-11 01:28:54,181 INFO Regime score epoch  6/50 — tr=0.0198 va=0.0123
2026-05-11 01:28:54,697 INFO Regime score epoch  7/50 — tr=0.0150 va=0.0085
2026-05-11 01:28:55,207 INFO Regime score epoch  8/50 — tr=0.0123 va=0.0066
2026-05-11 01:28:55,720 INFO Regime score epoch  9/50 — tr=0.0107 va=0.0054
2026-05-11 01:28:56,220 INFO Regime score epoch 10/50 — tr=0.0097 va=0.0048 mae={'trend_score': 0.0568, 'range_score': 0.0589, 'chop_score': 0.0521, 'volatility_percentile': 0.0321, 'consolidation_score': 0.0707}
2026-05-11 01:28:56,738 INFO Regime score epoch 11/50 — tr=0.0090 va=0.0043
2026-05-11 01:28:57,238 INFO Regime score epoch 12/50 — tr=0.0085 va=0.0039
2026-05-11 01:28:57,755 INFO Regime score epoch 13/50 — tr=0.0080 va=0.0037
2026-05-11 01:28:58,282 INFO Regime score epoch 14/50 — tr=0.0077 va=0.0034
2026-05-11 01:28:58,789 INFO Regime score epoch 15/50 — tr=0.0074 va=0.0032 mae={'trend_score': 0.0474, 'range_score': 0.0523, 'chop_score': 0.0486, 'volatility_percentile': 0.0274, 'consolidation_score': 0.0446}
2026-05-11 01:28:59,288 INFO Regime score epoch 16/50 — tr=0.0072 va=0.0032
2026-05-11 01:28:59,811 INFO Regime score epoch 17/50 — tr=0.0070 va=0.0030
2026-05-11 01:29:00,317 INFO Regime score epoch 18/50 — tr=0.0068 va=0.0029
2026-05-11 01:29:00,819 INFO Regime score epoch 19/50 — tr=0.0066 va=0.0028
2026-05-11 01:29:01,324 INFO Regime score epoch 20/50 — tr=0.0064 va=0.0027 mae={'trend_score': 0.0422, 'range_score': 0.049, 'chop_score': 0.0452, 'volatility_percentile': 0.0248, 'consolidation_score': 0.0376}
2026-05-11 01:29:01,822 INFO Regime score epoch 21/50 — tr=0.0063 va=0.0026
2026-05-11 01:29:02,321 INFO Regime score epoch 22/50 — tr=0.0061 va=0.0025
2026-05-11 01:29:02,851 INFO Regime score epoch 23/50 — tr=0.0060 va=0.0024
2026-05-11 01:29:03,382 INFO Regime score epoch 24/50 — tr=0.0059 va=0.0023
2026-05-11 01:29:03,889 INFO Regime score epoch 25/50 — tr=0.0058 va=0.0023 mae={'trend_score': 0.0373, 'range_score': 0.0461, 'chop_score': 0.0409, 'volatility_percentile': 0.0229, 'consolidation_score': 0.0335}
2026-05-11 01:29:04,400 INFO Regime score epoch 26/50 — tr=0.0057 va=0.0022
2026-05-11 01:29:04,908 INFO Regime score epoch 27/50 — tr=0.0057 va=0.0022
2026-05-11 01:29:05,414 INFO Regime score epoch 28/50 — tr=0.0056 va=0.0021
2026-05-11 01:29:05,929 INFO Regime score epoch 29/50 — tr=0.0055 va=0.0020
2026-05-11 01:29:06,463 INFO Regime score epoch 30/50 — tr=0.0054 va=0.0020 mae={'trend_score': 0.034, 'range_score': 0.0445, 'chop_score': 0.037, 'volatility_percentile': 0.0219, 'consolidation_score': 0.0311}
2026-05-11 01:29:06,982 INFO Regime score epoch 31/50 — tr=0.0054 va=0.0019
2026-05-11 01:29:07,504 INFO Regime score epoch 32/50 — tr=0.0053 va=0.0019
2026-05-11 01:29:08,023 INFO Regime score epoch 33/50 — tr=0.0053 va=0.0019
2026-05-11 01:29:08,563 INFO Regime score epoch 34/50 — tr=0.0052 va=0.0018
2026-05-11 01:29:09,111 INFO Regime score epoch 35/50 — tr=0.0052 va=0.0018 mae={'trend_score': 0.0317, 'range_score': 0.0436, 'chop_score': 0.0342, 'volatility_percentile': 0.0206, 'consolidation_score': 0.0301}
2026-05-11 01:29:09,649 INFO Regime score epoch 36/50 — tr=0.0052 va=0.0018
2026-05-11 01:29:10,200 INFO Regime score epoch 37/50 — tr=0.0051 va=0.0018
2026-05-11 01:29:10,715 INFO Regime score epoch 38/50 — tr=0.0051 va=0.0017
2026-05-11 01:29:11,250 INFO Regime score epoch 39/50 — tr=0.0051 va=0.0017
2026-05-11 01:29:11,782 INFO Regime score epoch 40/50 — tr=0.0051 va=0.0017 mae={'trend_score': 0.03, 'range_score': 0.0424, 'chop_score': 0.0325, 'volatility_percentile': 0.0195, 'consolidation_score': 0.0305}
2026-05-11 01:29:12,313 INFO Regime score epoch 41/50 — tr=0.0051 va=0.0017
2026-05-11 01:29:12,831 INFO Regime score epoch 42/50 — tr=0.0050 va=0.0017
2026-05-11 01:29:13,349 INFO Regime score epoch 43/50 — tr=0.0050 va=0.0017
2026-05-11 01:29:13,878 INFO Regime score epoch 44/50 — tr=0.0050 va=0.0017
2026-05-11 01:29:14,405 INFO Regime score epoch 45/50 — tr=0.0050 va=0.0017 mae={'trend_score': 0.0298, 'range_score': 0.0422, 'chop_score': 0.0319, 'volatility_percentile': 0.0196, 'consolidation_score': 0.0298}
2026-05-11 01:29:14,954 INFO Regime score epoch 46/50 — tr=0.0050 va=0.0017
2026-05-11 01:29:15,481 INFO Regime score epoch 47/50 — tr=0.0050 va=0.0017
2026-05-11 01:29:16,000 INFO Regime score epoch 48/50 — tr=0.0050 va=0.0017
2026-05-11 01:29:16,501 INFO Regime score epoch 49/50 — tr=0.0050 va=0.0017
2026-05-11 01:29:17,010 INFO Regime score epoch 50/50 — tr=0.0050 va=0.0017 mae={'trend_score': 0.0296, 'range_score': 0.0425, 'chop_score': 0.0318, 'volatility_percentile': 0.019, 'consolidation_score': 0.0292}
2026-05-11 01:29:17,031 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0296, 'range_score': 0.0425, 'chop_score': 0.0318, 'volatility_percentile': 0.019, 'consolidation_score': 0.0292} mse={'trend_score': 0.00145, 'range_score': 0.00291, 'chop_score': 0.00163, 'volatility_percentile': 0.0007, 'consolidation_score': 0.00162} corr={'trend_score': 0.9852, 'range_score': 0.9297, 'chop_score': 0.9777, 'volatility_percentile': 0.9925, 'consolidation_score': 0.9831} pred_std={'trend_score': 0.2162, 'range_score': 0.1421, 'chop_score': 0.1767, 'volatility_percentile': 0.2147, 'consolidation_score': 0.2136} target_std={'trend_score': 0.2219, 'range_score': 0.1433, 'chop_score': 0.1888, 'volatility_percentile': 0.2165, 'consolidation_score': 0.2179}
2026-05-11 01:29:17,360 INFO RegimeClassifier[mode=ltf_behaviour] train group diagnostics={'dollar': {'n': 16632, 'score_mae': {'trend_score': 0.029, 'range_score': 0.0417, 'chop_score': 0.0317, 'volatility_percentile': 0.0191, 'consolidation_score': 0.0292}, 'target_score_mean': {'trend_score': 0.491, 'range_score': 0.2347, 'chop_score': 0.4613, 'volatility_percentile': 0.3824, 'consolidation_score': 0.186}, 'pred_score_mean': {'trend_score': 0.4905, 'range_score': 0.2405, 'chop_score': 0.4606, 'volatility_percentile': 0.3837, 'consolidation_score': 0.1903}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3536, 26, 0, 1, 0, 0, 216], [16, 87, 0, 0, 0, 0, 7], [0, 0, 223, 9, 52, 0, 176], [2, 0, 15, 524, 43, 0, 105], [0, 0, 110, 32, 2950, 0, 224], [0, 30, 0, 0, 8, 0, 90], [213, 15, 335, 87, 109, 0, 7391]]}, 'gold': {'n': 8420, 'score_mae': {'trend_score': 0.0284, 'range_score': 0.0424, 'chop_score': 0.0316, 'volatility_percentile': 0.0196, 'consolidation_score': 0.0298}, 'target_score_mean': {'trend_score': 0.4883, 'range_score': 0.2355, 'chop_score': 0.4652, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}, 'pred_score_mean': {'trend_score': 0.4892, 'range_score': 0.2411, 'chop_score': 0.4636, 'volatility_percentile': 0.3791, 'consolidation_score': 0.1954}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1783, 17, 0, 0, 0, 0, 85], [6, 47, 0, 0, 0, 0, 3], [0, 0, 124, 8, 28, 0, 84], [1, 0, 11, 327, 24, 0, 53], [0, 0, 53, 31, 1488, 0, 132], [0, 25, 0, 0, 7, 0, 49], [103, 5, 176, 59, 63, 0, 3628]]}, 'jpy': {'n': 24948, 'score_mae': {'trend_score': 0.0288, 'range_score': 0.0417, 'chop_score': 0.0313, 'volatility_percentile': 0.0192, 'consolidation_score': 0.0293}, 'target_score_mean': {'trend_score': 0.4904, 'range_score': 0.2338, 'chop_score': 0.4653, 'volatility_percentile': 0.3826, 'consolidation_score': 0.1905}, 'pred_score_mean': {'trend_score': 0.4896, 'range_score': 0.2401, 'chop_score': 0.4639, 'volatility_percentile': 0.3839, 'consolidation_score': 0.1942}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[5272, 53, 0, 6, 0, 0, 315], [36, 141, 0, 0, 0, 0, 10], [0, 0, 309, 16, 69, 0, 253], [3, 0, 22, 1046, 95, 0, 148], [0, 0, 165, 95, 4457, 0, 398], [0, 59, 0, 0, 12, 0, 152], [314, 14, 491, 134, 218, 0, 10645]]}}
2026-05-11 01:29:17,541 INFO RegimeClassifier[mode=ltf_behaviour] validation group diagnostics={'dollar': {'n': 10093, 'score_mae': {'trend_score': 0.0298, 'range_score': 0.044, 'chop_score': 0.0318, 'volatility_percentile': 0.0189, 'consolidation_score': 0.0286}, 'target_score_mean': {'trend_score': 0.4863, 'range_score': 0.2392, 'chop_score': 0.4622, 'volatility_percentile': 0.3804, 'consolidation_score': 0.1814}, 'pred_score_mean': {'trend_score': 0.4868, 'range_score': 0.2437, 'chop_score': 0.462, 'volatility_percentile': 0.3823, 'consolidation_score': 0.1873}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[2251, 11, 0, 0, 0, 0, 150], [8, 43, 0, 0, 0, 0, 2], [0, 0, 146, 8, 45, 0, 117], [0, 0, 10, 324, 31, 0, 58], [0, 0, 75, 38, 1830, 0, 107], [0, 23, 0, 0, 3, 0, 51], [100, 5, 215, 65, 80, 0, 4297]]}, 'gold': {'n': 5117, 'score_mae': {'trend_score': 0.0285, 'range_score': 0.0421, 'chop_score': 0.0316, 'volatility_percentile': 0.0194, 'consolidation_score': 0.0301}, 'target_score_mean': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}, 'pred_score_mean': {'trend_score': 0.4953, 'range_score': 0.2415, 'chop_score': 0.4576, 'volatility_percentile': 0.3846, 'consolidation_score': 0.1878}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1090, 6, 0, 0, 0, 0, 71], [6, 26, 0, 0, 0, 0, 3], [0, 0, 98, 1, 11, 0, 61], [1, 0, 4, 213, 13, 0, 24], [0, 0, 33, 12, 768, 0, 74], [0, 14, 0, 0, 2, 0, 34], [68, 2, 120, 40, 44, 0, 2278]]}, 'jpy': {'n': 15142, 'score_mae': {'trend_score': 0.0299, 'range_score': 0.0417, 'chop_score': 0.0319, 'volatility_percentile': 0.019, 'consolidation_score': 0.0293}, 'target_score_mean': {'trend_score': 0.495, 'range_score': 0.2297, 'chop_score': 0.4582, 'volatility_percentile': 0.3808, 'consolidation_score': 0.1868}, 'pred_score_mean': {'trend_score': 0.4927, 'range_score': 0.2359, 'chop_score': 0.4584, 'volatility_percentile': 0.3827, 'consolidation_score': 0.1919}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3287, 36, 0, 2, 0, 0, 193], [25, 83, 0, 0, 0, 0, 7], [0, 0, 177, 9, 63, 0, 135], [2, 0, 22, 643, 50, 0, 110], [0, 0, 100, 44, 2450, 0, 223], [0, 29, 0, 0, 8, 0, 85], [169, 10, 280, 73, 146, 0, 6681]]}}
2026-05-11 01:29:17,546 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-11 01:29:17,547 INFO Regime phase LTF train fold=train_all: 26.4s
2026-05-11 01:29:17,657 INFO Regime LTF complete fold=train_all: score_accuracy=0.970, train=262644 val=30352 mae={'trend_score': 0.0296, 'range_score': 0.0425, 'chop_score': 0.0318, 'volatility_percentile': 0.019, 'consolidation_score': 0.0292}
2026-05-11 01:29:17,660 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:29:18,028 INFO Regime[1H mode=ltf_behaviour fold=latest] LTF score diagnostics on GBPUSD:
{'trend_score': {'mean': 0.4921, 'q10': 0.1918, 'q50': 0.4877, 'q90': 0.8002}, 'range_score': {'mean': 0.2337, 'q10': 0.0513, 'q50': 0.2125, 'q90': 0.4349}, 'chop_score': {'mean': 0.4612, 'q10': 0.2148, 'q50': 0.4503, 'q90': 0.7255}, 'volatility_percentile': {'mean': 0.3832, 'q10': 0.1018, 'q50': 0.3704, 'q90': 0.6823}, 'consolidation_score': {'mean': 0.1854, 'q10': 0.0, 'q50': 0.1185, 'q90': 0.506}}
2026-05-11 01:29:18,035 INFO Regime retrain total: 130.2s (370559 train+val samples)
2026-05-11 01:29:18,041 INFO Retrain complete. Total wall-clock: 130.2s
2026-05-11 01:29:19,082 INFO Model regime: SUCCESS
2026-05-11 01:29:19,082 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:29:19,082 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 01:29:19,082 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-11 01:29:19,082 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-05-11 01:29:19,082 INFO   [DORMANT] rl_ppo — RL_ENABLED=false, skipping
2026-05-11 01:29:19,082 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-05-11 01:29:19,083 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer']
2026-05-11 01:29:19,097 INFO Saved 58 retrain records to metrics/

=== TRAINING COMPLETE ===
  gru: SUCCESS
  regime: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Clean Quality/RL source: Backtest on train window ===
  START Train-window backtest for Quality/RL labels
2026-05-11 01:29:19,771 INFO === STEP 6: BACKTEST (train) ===
2026-05-11 01:29:19,773 INFO BT_WINDOW=train — train-window backtest: 2016-01-04 → 2023-08-04 (clean Quality/RL labels)
2026-05-11 01:29:19,773 INFO Cleared existing journal for fresh train run
2026-05-11 01:29:19,774 INFO ================================================================
  ROUND 0 / 3
================================================================
2026-05-11 01:29:19,774 INFO Round 0 — running backtest: 2016-01-04 → 2023-08-04 (ml_trader, shared ML cache)
2026-05-11 01:33:19,714 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for EURJPY with 2
2026-05-11 01:33:19,730 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for EURJPY with 0.3333333333333333
2026-05-11 01:33:19,897 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for EURUSD with 2
2026-05-11 01:33:19,927 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for EURUSD with 0.3333333333333333
2026-05-11 01:33:20,076 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for USDJPY with 2
2026-05-11 01:33:20,102 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for USDJPY with 0.3333333333333333
2026-05-11 01:33:20,218 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for EURJPY with 2
2026-05-11 01:33:20,244 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for EURJPY with 0.25
2026-05-11 01:33:20,275 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for EURJPY: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 01:33:20,605 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for EURUSD with 2
2026-05-11 01:33:20,622 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for EURUSD with 0.25
2026-05-11 01:33:20,667 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for EURUSD: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 01:33:20,850 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for USDJPY with 2
2026-05-11 01:33:20,866 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for USDJPY with 0.25
2026-05-11 01:33:20,908 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for USDJPY: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 01:33:21,529 WARNING ML cache score overlay filled 4 warmup/alignment gaps for EURJPY
2026-05-11 01:33:24,996 WARNING ML cache score overlay filled 4 warmup/alignment gaps for EURUSD
2026-05-11 01:33:25,373 WARNING ML cache score overlay filled 4 warmup/alignment gaps for USDJPY
2026-05-11 01:33:39,434 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 01:33:40,162 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 01:33:40,231 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 01:33:40,277 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 01:33:40,321 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 01:33:40,346 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 01:33:40,382 WARNING _build_sequence_df: HTF frame 5M filled 357 warmup/alignment gaps with 0.000
2026-05-11 01:33:40,397 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 01:33:40,430 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 01:33:40,448 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 01:33:40,473 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 01:33:40,508 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 01:33:40,513 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 01:33:40,556 WARNING _build_sequence_df: HTF frame 5M filled 357 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 01:33:40,588 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 01:33:40,657 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 01:33:40,661 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 01:33:40,685 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 01:33:40,751 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 01:33:40,755 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 01:33:40,826 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 01:33:40,845 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 01:33:40,872 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 01:33:40,934 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 01:33:40,951 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 01:33:40,989 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
2026-05-11 01:33:41,003 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 01:33:41,029 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 01:33:41,066 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 01:33:41,071 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 01:33:41,095 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 01:33:41,120 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 01:33:41,157 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 01:33:41,194 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 01:33:41,243 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 01:33:57,889 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for GBPJPY with 2
2026-05-11 01:33:57,904 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for GBPJPY with 0.3333333333333333
2026-05-11 01:33:58,043 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for GBPUSD with 2
2026-05-11 01:33:58,069 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for GBPUSD with 0.3333333333333333
2026-05-11 01:33:58,234 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for GBPJPY with 2
2026-05-11 01:33:58,247 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for GBPJPY with 0.25
2026-05-11 01:33:58,272 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for GBPJPY: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 01:33:58,444 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for GBPUSD with 2
2026-05-11 01:33:58,459 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for GBPUSD with 0.25
2026-05-11 01:33:58,476 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for GBPUSD: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 01:33:58,944 WARNING ML cache score overlay filled 4 warmup/alignment gaps for GBPJPY
2026-05-11 01:33:59,490 WARNING ML cache score overlay filled 4 warmup/alignment gaps for GBPUSD
2026-05-11 01:34:07,842 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 01:34:07,861 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 01:34:07,921 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 01:34:07,963 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 01:34:07,967 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 01:34:08,001 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 01:34:08,021 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 01:34:08,038 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 01:34:08,067 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
2026-05-11 01:34:08,068 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 01:34:08,098 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 01:34:08,114 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 01:34:08,143 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 01:34:08,147 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 01:34:08,165 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 01:34:08,197 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
2026-05-11 01:34:08,197 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 01:34:08,219 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 01:34:08,237 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 01:34:08,255 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 01:34:08,277 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 01:34:08,305 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260511_012921.json
Trader                                   Trades      WR    PF*  Return*    ExpR   TP1%   TP2%     DD*  Sharpe*  Sortino     IC  Verdict
-----------------------------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)              576  23.8%   0.86  -63.0%  -0.109 23.8%  6.8%  96.3%    -1.05    -0.11 -0.077     FAIL
  FAILED rules: positive_expectancy, profit_factor_min_1_25, drawdown_below_20pct, sharpe_positive, sortino_positive, win_rate_above_breakeven, mc_p10_not_ruin, sharpe_ci_positive
  monthly R: 2022-11=+0.27  2022-12=-6.01  2023-02=-2.18  2023-05=+5.82  2023-06=+9.65  2023-07=-1.42
  MonteCarlo P95 DD=90.9%  P10 equity=3,698  t=-1.58 (p=0.114)  Sharpe CI=[-2.63, 0.17]  streak=33
  gate_diagnostics: bars=1049680 no_signal=486478 quality_block=0 session_skip=562433 density=193 pm_reject=0 daily_skip=0 cooldown=0 daily_halt_events=0 enforce_daily_halt=False
  no_signal_reasons: gru_expected_r_below_threshold=245927, weak_gru_direction=114854, no_trade_uncertain=79878, no_trade_extreme_vol=23683, no_trade_chop=18057, wait_pullback=2899

Calibration Summary:
  all          [OK] Too few populated bins for calibration check
  ml_trader    [OK] Too few populated bins for calibration check
2026-05-11 01:36:36,793 INFO Round 0 backtest — 576 trades | avg WR=23.8% | avg PF=0.86 | avg Sharpe=-1.05
2026-05-11 01:36:36,793 INFO   ml_trader: 576 trades | WR=23.8% | fixed PF=0.86 | Return=-63.0% | ExpR=-0.109 | DD=96.3% | Sharpe=-1.05
2026-05-11 01:36:36,793 INFO   ml_trader gate_diagnostics: bars=1049680 no_signal=486478 quality_block=0 session_skip=562433 density=193 pm_reject=0
2026-05-11 01:36:36,793 INFO   ml_trader no_signal_reasons: {'no_trade_uncertain': 79878, 'weak_gru_direction': 114854, 'gru_expected_r_below_threshold': 245927, 'no_trade_extreme_vol': 23683, 'wait_pullback': 2899, 'expected_r_below_threshold': 36, 'tradeability_direction_conflict': 1144, 'no_trade_chop': 18057}
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_0.json
INFO  Total trades: 576
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (576 rows)
2026-05-11 01:36:37,443 INFO Round 0: wrote 576 journal entries (total in file: 576)
  DONE  Train-window backtest for Quality/RL labels
  Saved Train Quality/RL source result → train_quality_rl_source_summary.json
  Train-label journal entries: 576

=== Train Quality + RL on train-only journal ===
  START Train-only Quality+RL retrain
2026-05-11 01:36:37,709 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-05-11 01:36:37,731 INFO Journal entries: 576 total, 576 allowed for training (['live', 'paper', 'production', 'train'])
2026-05-11 01:36:37,731 INFO --- Training quality ---
2026-05-11 01:36:37,732 INFO Running retrain --model quality with JOURNAL_ALLOWED_SPLITS=train,live,paper,production
2026-05-11 01:36:37,928 INFO retrain environment: KAGGLE
2026-05-11 01:36:39,639 INFO Device: CUDA (2 GPU(s))
2026-05-11 01:36:39,651 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 01:36:39,651 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 01:36:39,651 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 01:36:39,654 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 01:36:39,654 INFO Retrain data split: train
2026-05-11 01:36:39,654 INFO Retrain rolling fold selector: latest
2026-05-11 01:36:39,655 INFO === QualityScorer retrain ===
2026-05-11 01:36:39,837 INFO NumExpr defaulting to 4 threads.
2026-05-11 01:36:40,046 INFO QualityScorer: CUDA available — using GPU
2026-05-11 01:36:40,129 INFO QualityScorer: group EV smoothing applied to 551/576 rows (blend=30% group, min_group=10)
2026-05-11 01:36:40,132 INFO Quality phase label creation: 0.1s (576 trades)
2026-05-11 01:36:40,209 INFO QualityScorer: group EV smoothing applied to 551/576 rows (blend=30% group, min_group=10)
2026-05-11 01:36:40,212 INFO QualityScorer: 576 samples, EV stats={'mean': -0.3911701440811157, 'std': 0.8455881476402283, 'n_pos': 137, 'n_neg': 439}, device=cuda
2026-05-11 01:36:40,426 INFO QualityScorer: DataParallel across 2 GPUs
2026-05-11 01:36:40,426 INFO QualityScorer: cold start
2026-05-11 01:36:40,427 INFO QualityScorer: pos_weight=3.47 (n_pos=103 n_neg=357)
2026-05-11 01:36:42,832 INFO Quality epoch   1/100 — va_huber=0.8565
2026-05-11 01:36:42,873 INFO Quality epoch   2/100 — va_huber=0.8521
2026-05-11 01:36:42,896 INFO Quality epoch   3/100 — va_huber=0.8495
2026-05-11 01:36:42,918 INFO Quality epoch   4/100 — va_huber=0.8472
2026-05-11 01:36:42,948 INFO Quality epoch   5/100 — va_huber=0.8449
2026-05-11 01:36:43,073 INFO Quality epoch  11/100 — va_huber=0.8321
2026-05-11 01:36:43,282 INFO Quality epoch  21/100 — va_huber=0.8071
2026-05-11 01:36:43,500 INFO Quality epoch  31/100 — va_huber=0.7979
2026-05-11 01:36:43,713 INFO Quality epoch  41/100 — va_huber=0.7936
2026-05-11 01:36:43,855 INFO Quality early stop at epoch 48
2026-05-11 01:36:43,864 INFO QualityScorer EV model: MAE=0.846 dir_acc=0.578 n_val=116
2026-05-11 01:36:43,868 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-05-11 01:36:43,926 INFO Quality phase train: 3.8s | total: 4.3s
2026-05-11 01:36:43,935 INFO Retrain complete. Total wall-clock: 4.3s
2026-05-11 01:36:45,024 INFO Model quality: SUCCESS
2026-05-11 01:36:45,024 INFO --- Training rl ---
2026-05-11 01:36:45,024 INFO Running retrain --model rl with JOURNAL_ALLOWED_SPLITS=train,live,paper,production
2026-05-11 01:36:45,225 INFO retrain environment: KAGGLE
2026-05-11 01:36:46,856 INFO Device: CUDA (2 GPU(s))
2026-05-11 01:36:46,866 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 01:36:46,866 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 01:36:46,866 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 01:36:46,867 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 01:36:46,867 INFO Retrain data split: train
2026-05-11 01:36:46,867 INFO Retrain rolling fold selector: latest
2026-05-11 01:36:46,869 INFO === RLAgent (PPO) retrain ===
2026-05-11 01:36:47,018 INFO NumExpr defaulting to 4 threads.
2026-05-11 01:36:47,214 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260511_013647
2026-05-11 01:36:47,239 INFO RL phase episode loading: 0.0s (576 episodes)
2026-05-11 01:36:50.961323: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1778463411.207735  108517 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1778463411.276677  108517 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1778463411.866187  108517 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1778463411.866246  108517 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1778463411.866250  108517 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1778463411.866253  108517 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-05-11 01:37:09,640 INFO RLAgent: cold start — building new PPO policy
2026-05-11 01:37:21,751 INFO RLAgent: retrain complete, 576 episodes
2026-05-11 01:37:21,752 INFO RL phase PPO train: 34.5s | total: 34.9s
2026-05-11 01:37:21,763 INFO Retrain complete. Total wall-clock: 34.9s
2026-05-11 01:37:23,464 INFO Model rl: SUCCESS
2026-05-11 01:37:23,465 INFO Step 7b complete — summary: /kaggle/working/Multi-Bot/trading-system/ml_training/metrics/training_7b_summary.json
  DONE  Train-only Quality+RL retrain
  Archived journal → trade_journal_train_only.jsonl
  Archived journal CSV → trade_journal_train_only.csv

=== Round 1: Backtest on train-tail window (latest 2yr inside training data) ===
  Cleared journal for fresh Round 1 run
  START Round 1 - Backtest (train-tail)
2026-05-11 01:37:23,982 INFO === STEP 6: BACKTEST (round1) ===
2026-05-11 01:37:23,983 INFO BT_WINDOW=round1 — train-tail backtest: 2021-08-05 → 2023-08-04 (seen training data; test set protected)
2026-05-11 01:37:23,983 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-05-11 01:37:23,983 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
2026-05-11 01:37:23,984 INFO Round 1 — QualityScorer hard gate disabled; set BACKTEST_ENABLE_QUALITY_GATE=1 to enable it
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:38:40,829 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 01:38:41,339 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 01:38:41,431 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 01:38:41,491 WARNING _build_sequence_df: HTF frame 5M filled 461 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 01:38:41,517 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 01:38:41,564 WARNING _build_sequence_df: HTF frame 5M filled 461 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 01:38:41,581 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 01:38:41,692 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
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
2026-05-11 01:38:50,611 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 01:38:50,637 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 01:38:50,718 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 01:38:50,756 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 01:39:28,674 INFO Round 1 backtest — 149 trades | avg WR=28.9% | avg PF=1.18 | avg Sharpe=1.07
2026-05-11 01:39:28,674 INFO   ml_trader: 149 trades | WR=28.9% | fixed PF=1.18 | Return=18.6% | ExpR=0.125 | DD=18.6% | Sharpe=1.07
2026-05-11 01:39:28,674 INFO   ml_trader gate_diagnostics: bars=263960 no_signal=117908 quality_block=0 session_skip=145857 density=46 pm_reject=0
2026-05-11 01:39:28,674 INFO   ml_trader no_signal_reasons: {'no_trade_uncertain': 18828, 'gru_expected_r_below_threshold': 59562, 'weak_gru_direction': 27334, 'no_trade_chop': 4658, 'no_trade_extreme_vol': 6401, 'tradeability_direction_conflict': 298, 'expected_r_below_threshold': 9, 'wait_pullback': 818}
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 149
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (149 rows)
2026-05-11 01:39:28,993 INFO Round 1: wrote 149 journal entries (total in file: 149)
  DONE  Round 1 - Backtest (train-tail)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 149 entries

  SKIP  Round 1 Quality+RL retrain — train-tail journal kept evaluation-only

=== Round 2: BLIND backtest on test window (unseen 2yr) ===
  START Round 2 - Blind backtest (test)
2026-05-11 01:39:29,692 INFO === STEP 6: BACKTEST (round2) ===
2026-05-11 01:39:29,694 INFO BT_WINDOW=round2 — BLIND backtest: 2023-08-07 → 2025-08-05 (test set)
2026-05-11 01:39:29,694 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-05-11 01:39:29,694 INFO Round 2 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
2026-05-11 01:39:29,694 INFO Round 2 — QualityScorer hard gate disabled; set BACKTEST_ENABLE_QUALITY_GATE=1 to enable it
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
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
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
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
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:40:48,966 WARNING _build_sequence_df: HTF frame 5M filled 317 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 01:40:49,695 WARNING _build_sequence_df: HTF frame 5M filled 317 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 01:40:49,973 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 01:40:50,043 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 01:40:50,144 WARNING _build_sequence_df: HTF frame 5M filled 341 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 01:40:50,218 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 01:40:50,225 WARNING _build_sequence_df: HTF frame 5M filled 341 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 01:40:50,288 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
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
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:40:59,051 WARNING _build_sequence_df: HTF frame 5M filled 317 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 01:40:59,180 WARNING _build_sequence_df: HTF frame 5M filled 317 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 01:40:59,265 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 01:40:59,295 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 01:41:39,495 INFO Round 2 backtest — 223 trades | avg WR=26.9% | avg PF=1.01 | avg Sharpe=0.05
2026-05-11 01:41:39,496 INFO   ml_trader: 223 trades | WR=26.9% | fixed PF=1.01 | Return=1.1% | ExpR=0.005 | DD=34.8% | Sharpe=0.05
2026-05-11 01:41:39,496 INFO   ml_trader gate_diagnostics: bars=280782 no_signal=131366 quality_block=0 session_skip=149122 density=71 pm_reject=0
2026-05-11 01:41:39,496 INFO   ml_trader no_signal_reasons: {'weak_gru_direction': 30723, 'no_trade_uncertain': 22858, 'gru_expected_r_below_threshold': 64846, 'no_trade_chop': 5116, 'tradeability_direction_conflict': 323, 'wait_pullback': 1053, 'no_trade_extreme_vol': 6439, 'expected_r_below_threshold': 8}
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 223
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (223 rows)
2026-05-11 01:41:39,936 INFO Round 2: wrote 223 journal entries (total in file: 372)
  DONE  Round 2 - Blind backtest (test)
  Saved Round 2 result → round2_summary.json
  Journal after Round 2: 372 entries

  SKIP  Round 2 Quality+RL retrain — blind test journal kept untouched

=== Round 3: Incremental retrain ===
  START Retrain gru [train-split retrain]
2026-05-11 01:41:40,279 INFO retrain environment: KAGGLE
2026-05-11 01:41:41,922 INFO Device: CUDA (2 GPU(s))
2026-05-11 01:41:41,934 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 01:41:41,934 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 01:41:41,934 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 01:41:41,934 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 01:41:41,934 INFO Retrain data split: train
2026-05-11 01:41:41,934 INFO Retrain rolling fold selector: latest
2026-05-11 01:41:41,935 INFO === GRU-LSTM retrain (timeframes: ['15M']) ===
2026-05-11 01:41:42,080 INFO NumExpr defaulting to 4 threads.
2026-05-11 01:41:42,277 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-11 01:41:42,277 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 01:41:42,277 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 01:41:43,178 INFO GRULSTMPredictor: loaded long R isotonic calibrator from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/r_isotonic_long.pkl
2026-05-11 01:41:43,179 INFO GRULSTMPredictor: loaded short R isotonic calibrator from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/r_isotonic_short.pkl
2026-05-11 01:41:43,179 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-05-11 01:41:43,179 INFO GRU multi-symbol training (Kaggle mode): 6 symbols × ['15M']
2026-05-11 01:41:43,181 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260511_014143
2026-05-11 01:41:43,185 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-05-11 01:41:43,186 INFO GRU warm start enabled from existing weights: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:41:43,448 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:41:43,478 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:41:43,496 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:41:43,507 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:41:43,583 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 01:41:43,589 INFO Loaded XAUUSD/15M split=train fold=latest: 176438 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:41:44,170 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:41:44,190 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:41:44,204 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:41:44,211 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:41:44,251 INFO Loaded EURUSD/15M split=train fold=latest: 174951 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:41:44,833 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:41:44,853 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:41:44,869 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:41:44,877 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:41:44,916 INFO Loaded USDJPY/15M split=train fold=latest: 174944 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:41:45,478 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:41:45,500 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:41:45,514 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:41:45,523 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:41:45,562 INFO Loaded EURJPY/15M split=train fold=latest: 174948 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:41:46,111 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:41:46,131 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:41:46,146 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:41:46,154 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:41:46,196 INFO Loaded GBPJPY/15M split=train fold=latest: 174902 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:41:46,757 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:41:46,776 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:41:46,790 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:41:46,798 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:41:46,837 INFO Loaded GBPUSD/15M split=train fold=latest: 174899 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:41:47,304 INFO train_multi: 6 segments, ~971854 total bars
2026-05-11 01:41:47,304 INFO train_multi: training ALL 6 segments across TFs ['15M'] in one combined pass
2026-05-11 01:41:47,304 INFO train_multi: building combined dataset for TF=ALL (6 segments)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:41:56,991 INFO train_multi TF=ALL: 971494 sequences across 6 segments
2026-05-11 01:41:56,991 INFO train_multi TF=ALL: estimated peak RAM = 21312 MB (train=479996 val=120002 n_feat=74 seq_len=60)
2026-05-11 01:41:56,992 WARNING train_multi TF=ALL: trimming to fit RAM budget — new train=450450 val=112612 (20000 MB est)
2026-05-11 01:41:59,314 INFO train_multi TF=ALL: train=450450 val=112612 (10009 MB tensors)
2026-05-11 01:42:05,547 INFO train_multi TF=ALL: structural bar weighting — 288225 structural bars (64.0%) weight=15.0 structural_only=0
2026-05-11 01:42:06,614 INFO train_multi TF=ALL: warm-start detected — using CosineAnnealingLR (lr=3e-05, patience=12)
2026-05-11 01:42:27,306 INFO train_multi TF=ALL epoch 1/100 train=1.7200 val=1.8565 r_mae=0.768 pos_r_acc=0.689 side_acc=0.740 r_n=161888
2026-05-11 01:42:27,312 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 01:42:27,313 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 01:42:27,313 INFO train_multi TF=ALL: new best val=1.8565 — saved
2026-05-11 01:42:45,905 INFO train_multi TF=ALL epoch 2/100 train=1.7158 val=1.8592 r_mae=0.770 pos_r_acc=0.688 side_acc=0.739 r_n=161888
2026-05-11 01:43:04,208 INFO train_multi TF=ALL epoch 3/100 train=1.7130 val=1.8621 r_mae=0.771 pos_r_acc=0.688 side_acc=0.739 r_n=161888
2026-05-11 01:43:22,743 INFO train_multi TF=ALL epoch 4/100 train=1.7099 val=1.8624 r_mae=0.771 pos_r_acc=0.688 side_acc=0.739 r_n=161888
2026-05-11 01:43:41,244 INFO train_multi TF=ALL epoch 5/100 train=1.7103 val=1.8639 r_mae=0.773 pos_r_acc=0.687 side_acc=0.738 r_n=161888
2026-05-11 01:43:59,607 INFO train_multi TF=ALL epoch 6/100 train=1.7105 val=1.8631 r_mae=0.770 pos_r_acc=0.688 side_acc=0.739 r_n=161888
2026-05-11 01:44:17,998 INFO train_multi TF=ALL epoch 7/100 train=1.7087 val=1.8618 r_mae=0.771 pos_r_acc=0.688 side_acc=0.739 r_n=161888
2026-05-11 01:44:36,356 INFO train_multi TF=ALL epoch 8/100 train=1.7083 val=1.8620 r_mae=0.770 pos_r_acc=0.688 side_acc=0.739 r_n=161888
2026-05-11 01:44:54,809 INFO train_multi TF=ALL epoch 9/100 train=1.7072 val=1.8643 r_mae=0.772 pos_r_acc=0.687 side_acc=0.739 r_n=161888
2026-05-11 01:45:13,003 INFO train_multi TF=ALL epoch 10/100 train=1.7064 val=1.8647 r_mae=0.771 pos_r_acc=0.687 side_acc=0.739 r_n=161888
2026-05-11 01:45:31,325 INFO train_multi TF=ALL epoch 11/100 train=1.7037 val=1.8634 r_mae=0.769 pos_r_acc=0.688 side_acc=0.740 r_n=161888
2026-05-11 01:45:49,389 INFO train_multi TF=ALL epoch 12/100 train=1.7057 val=1.8640 r_mae=0.770 pos_r_acc=0.688 side_acc=0.739 r_n=161888
2026-05-11 01:46:07,726 INFO train_multi TF=ALL epoch 13/100 train=1.7026 val=1.8652 r_mae=0.770 pos_r_acc=0.687 side_acc=0.739 r_n=161888
2026-05-11 01:46:07,726 INFO train_multi TF=ALL early stop at epoch 13
2026-05-11 01:46:08,376 INFO fit_r_isotonic: saved side-R calibrators stats={'long': {'n': 81464, 'raw_mae': 0.7747004362309899, 'calibrated_mae': 0.7671767397169019}, 'short': {'n': 80424, 'raw_mae': 0.7619543980331152, 'calibrated_mae': 0.7639185182316801}}
2026-05-11 01:46:08,532 WARNING GRU validation R-MAE above floor for TF=ALL: best_val_r_mae=0.768 max=0.750. Keeping saved best weights so the pipeline can progress.
2026-05-11 01:46:08,545 INFO GRU R threshold XAUUSD/buy: q25_pos=0.535 q50_pos=1.667 pos_rate=46.0% (n=119259 n_pos=54899)
2026-05-11 01:46:08,552 INFO GRU R threshold XAUUSD/sell: q25_pos=0.528 q50_pos=1.667 pos_rate=45.5% (n=118087 n_pos=53734)
2026-05-11 01:46:08,558 INFO GRU R threshold EURUSD/buy: q25_pos=0.541 q50_pos=1.667 pos_rate=45.6% (n=118751 n_pos=54196)
2026-05-11 01:46:08,564 INFO GRU R threshold EURUSD/sell: q25_pos=0.545 q50_pos=1.667 pos_rate=45.6% (n=117679 n_pos=53721)
2026-05-11 01:46:08,570 INFO GRU R threshold USDJPY/buy: q25_pos=0.551 q50_pos=1.667 pos_rate=46.2% (n=118691 n_pos=54870)
2026-05-11 01:46:08,577 INFO GRU R threshold USDJPY/sell: q25_pos=0.542 q50_pos=1.667 pos_rate=44.6% (n=116030 n_pos=51793)
2026-05-11 01:46:08,583 INFO GRU R threshold EURJPY/buy: q25_pos=0.530 q50_pos=1.416 pos_rate=46.1% (n=118651 n_pos=54654)
2026-05-11 01:46:08,589 INFO GRU R threshold EURJPY/sell: q25_pos=0.525 q50_pos=1.667 pos_rate=44.7% (n=117320 n_pos=52399)
2026-05-11 01:46:08,595 INFO GRU R threshold GBPJPY/buy: q25_pos=0.517 q50_pos=1.443 pos_rate=45.9% (n=118277 n_pos=54278)
2026-05-11 01:46:08,601 INFO GRU R threshold GBPJPY/sell: q25_pos=0.521 q50_pos=1.667 pos_rate=45.3% (n=116299 n_pos=52702)
2026-05-11 01:46:08,607 INFO GRU R threshold GBPUSD/buy: q25_pos=0.524 q50_pos=1.667 pos_rate=45.8% (n=117764 n_pos=53894)
2026-05-11 01:46:08,613 INFO GRU R threshold GBPUSD/sell: q25_pos=0.533 q50_pos=1.667 pos_rate=45.8% (n=117404 n_pos=53785)
2026-05-11 01:46:08,614 INFO GRU per-symbol R thresholds saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/symbol_r_thresholds.json (6 symbols)
2026-05-11 01:46:08,635 INFO Retrain complete. Total wall-clock: 266.7s
  DONE  Retrain gru [train-split retrain]
  START Retrain regime [train-split retrain]
2026-05-11 01:46:11,769 INFO retrain environment: KAGGLE
2026-05-11 01:46:13,386 INFO Device: CUDA (2 GPU(s))
2026-05-11 01:46:13,395 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 01:46:13,395 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 01:46:13,395 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 01:46:13,398 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 01:46:13,398 INFO Retrain data split: train
2026-05-11 01:46:13,398 INFO Retrain rolling fold selector: latest
2026-05-11 01:46:13,399 INFO === RegimeClassifier retrain (hierarchical: HTF 2-score bias + LTF 5-score behaviour) ===
2026-05-11 01:46:13,546 INFO NumExpr defaulting to 4 threads.
2026-05-11 01:46:13,759 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-11 01:46:13,759 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 01:46:13,759 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 01:46:13,760 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-11 01:46:13,818 INFO Regime rolling folds selected: [None]
2026-05-11 01:46:13,818 INFO === Regime rolling fold 1/1: train_all ===
2026-05-11 01:46:13,818 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-11 01:46:13,860 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 01:46:13,861 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:46:13,877 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:46:13,894 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:46:13,911 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:46:13,928 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:46:13,943 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:46:14,196 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:46:14,269 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:46:14,294 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:46:14,295 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:46:14,306 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:46:14,307 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:46:14,892 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 550, 'BIAS_DOWN': 255, 'BIAS_NEUTRAL': 11297}  ambiguous=7029 (total=12102) horizon=36
2026-05-11 01:46:14,897 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected XAUUSD — 12052 samples (group=gold) score_means={'bias_up_score': 0.0456, 'bias_down_score': 0.0212} labels={'BIAS_UP': 550, 'BIAS_DOWN': 255, 'BIAS_NEUTRAL': 11247} clean={'BIAS_UP': 550, 'BIAS_DOWN': 255, 'BIAS_NEUTRAL': 4250}
2026-05-11 01:46:15,074 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:15,110 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:15,128 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:15,129 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:15,136 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:15,138 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:46:15,668 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 299, 'BIAS_DOWN': 288, 'BIAS_NEUTRAL': 10817}  ambiguous=6558 (total=11404) horizon=36
2026-05-11 01:46:15,673 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURUSD — 11354 samples (group=dollar) score_means={'bias_up_score': 0.0263, 'bias_down_score': 0.0254} labels={'BIAS_UP': 299, 'BIAS_DOWN': 288, 'BIAS_NEUTRAL': 10767} clean={'BIAS_UP': 299, 'BIAS_DOWN': 288, 'BIAS_NEUTRAL': 4231}
2026-05-11 01:46:15,833 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:15,868 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:15,888 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:15,888 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:15,897 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:15,898 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:46:16,439 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 431, 'BIAS_DOWN': 164, 'BIAS_NEUTRAL': 10808}  ambiguous=6695 (total=11403) horizon=36
2026-05-11 01:46:16,444 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected USDJPY — 11353 samples (group=dollar) score_means={'bias_up_score': 0.038, 'bias_down_score': 0.0144} labels={'BIAS_UP': 431, 'BIAS_DOWN': 164, 'BIAS_NEUTRAL': 10758} clean={'BIAS_UP': 431, 'BIAS_DOWN': 164, 'BIAS_NEUTRAL': 4092}
2026-05-11 01:46:16,603 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:16,638 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:16,660 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:16,660 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:16,668 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:16,669 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:46:17,198 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 313, 'BIAS_DOWN': 162, 'BIAS_NEUTRAL': 10932}  ambiguous=6806 (total=11407) horizon=36
2026-05-11 01:46:17,203 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURJPY — 11357 samples (group=cross) score_means={'bias_up_score': 0.0276, 'bias_down_score': 0.0143} labels={'BIAS_UP': 313, 'BIAS_DOWN': 162, 'BIAS_NEUTRAL': 10882} clean={'BIAS_UP': 313, 'BIAS_DOWN': 162, 'BIAS_NEUTRAL': 4094}
2026-05-11 01:46:17,357 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:17,393 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:17,413 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:17,413 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:17,422 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:17,423 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:46:17,948 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 467, 'BIAS_DOWN': 285, 'BIAS_NEUTRAL': 10656}  ambiguous=6835 (total=11408) horizon=36
2026-05-11 01:46:17,953 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPJPY — 11358 samples (group=cross) score_means={'bias_up_score': 0.0411, 'bias_down_score': 0.0251} labels={'BIAS_UP': 467, 'BIAS_DOWN': 285, 'BIAS_NEUTRAL': 10606} clean={'BIAS_UP': 467, 'BIAS_DOWN': 285, 'BIAS_NEUTRAL': 3818}
2026-05-11 01:46:18,107 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:18,141 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:18,160 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:18,160 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:18,172 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:18,173 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:46:18,700 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 255, 'BIAS_DOWN': 300, 'BIAS_NEUTRAL': 10847}  ambiguous=6860 (total=11402) horizon=36
2026-05-11 01:46:18,705 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPUSD — 11352 samples (group=dollar) score_means={'bias_up_score': 0.0225, 'bias_down_score': 0.0264} labels={'BIAS_UP': 255, 'BIAS_DOWN': 300, 'BIAS_NEUTRAL': 10797} clean={'BIAS_UP': 255, 'BIAS_DOWN': 300, 'BIAS_NEUTRAL': 3987}
2026-05-11 01:46:18,775 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 780, 'BIAS_DOWN': 447, 'BIAS_NEUTRAL': 21488}, 'dollar': {'BIAS_UP': 985, 'BIAS_DOWN': 752, 'BIAS_NEUTRAL': 32322}, 'gold': {'BIAS_UP': 550, 'BIAS_DOWN': 255, 'BIAS_NEUTRAL': 11247}}
2026-05-11 01:46:18,775 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0343, 'bias_down_score': 0.0197}, 'dollar': {'bias_up_score': 0.0289, 'bias_down_score': 0.0221}, 'gold': {'bias_up_score': 0.0456, 'bias_down_score': 0.0212}}
2026-05-11 01:46:18,776 INFO Regime[4H mode=htf_bias] label distribution by year: {2016: {'BIAS_UP': 274, 'BIAS_DOWN': 351, 'BIAS_NEUTRAL': 8197}, 2017: {'BIAS_UP': 495, 'BIAS_DOWN': 167, 'BIAS_NEUTRAL': 8451}, 2018: {'BIAS_UP': 208, 'BIAS_DOWN': 250, 'BIAS_NEUTRAL': 8672}, 2019: {'BIAS_UP': 189, 'BIAS_DOWN': 174, 'BIAS_NEUTRAL': 8739}, 2020: {'BIAS_UP': 285, 'BIAS_DOWN': 131, 'BIAS_NEUTRAL': 8695}, 2021: {'BIAS_UP': 311, 'BIAS_DOWN': 163, 'BIAS_NEUTRAL': 8617}, 2022: {'BIAS_UP': 340, 'BIAS_DOWN': 162, 'BIAS_NEUTRAL': 8619}, 2023: {'BIAS_UP': 213, 'BIAS_DOWN': 56, 'BIAS_NEUTRAL': 5067}}
2026-05-11 01:46:18,776 INFO Regime[4H mode=htf_bias] score means by year: {2016: {'bias_up_score': 0.0311, 'bias_down_score': 0.0398}, 2017: {'bias_up_score': 0.0543, 'bias_down_score': 0.0183}, 2018: {'bias_up_score': 0.0228, 'bias_down_score': 0.0274}, 2019: {'bias_up_score': 0.0208, 'bias_down_score': 0.0191}, 2020: {'bias_up_score': 0.0313, 'bias_down_score': 0.0144}, 2021: {'bias_up_score': 0.0342, 'bias_down_score': 0.0179}, 2022: {'bias_up_score': 0.0373, 'bias_down_score': 0.0178}, 2023: {'bias_up_score': 0.0399, 'bias_down_score': 0.0105}}
2026-05-11 01:46:18,823 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
2026-05-11 01:46:18,824 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
2026-05-11 01:46:18,824 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
2026-05-11 01:46:18,825 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 01:46:18,826 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 01:46:18,827 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
2026-05-11 01:46:18,843 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:46:18,847 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:46:18,848 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:46:18,848 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:46:18,849 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:46:18,850 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:46:19,180 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 53, 'BIAS_DOWN': 23, 'BIAS_NEUTRAL': 1505}  ambiguous=929 (total=1581) horizon=36
2026-05-11 01:46:19,183 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected XAUUSD — 1531 samples (group=gold) score_means={'bias_up_score': 0.0346, 'bias_down_score': 0.015} labels={'BIAS_UP': 53, 'BIAS_DOWN': 23, 'BIAS_NEUTRAL': 1455} clean={'BIAS_UP': 53, 'BIAS_DOWN': 23, 'BIAS_NEUTRAL': 558}
2026-05-11 01:46:19,258 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:19,260 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:19,261 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:19,262 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:19,262 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:19,263 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:46:19,588 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 18, 'BIAS_DOWN': 38, 'BIAS_NEUTRAL': 1435}  ambiguous=844 (total=1491) horizon=36
2026-05-11 01:46:19,591 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURUSD — 1441 samples (group=dollar) score_means={'bias_up_score': 0.0125, 'bias_down_score': 0.0264} labels={'BIAS_UP': 18, 'BIAS_DOWN': 38, 'BIAS_NEUTRAL': 1385} clean={'BIAS_UP': 18, 'BIAS_DOWN': 38, 'BIAS_NEUTRAL': 577}
2026-05-11 01:46:19,667 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:19,669 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:19,670 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:19,671 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:19,671 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:19,672 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:46:20,010 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 95, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 1386}  ambiguous=905 (total=1489) horizon=36
2026-05-11 01:46:20,013 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected USDJPY — 1439 samples (group=dollar) score_means={'bias_up_score': 0.066, 'bias_down_score': 0.0056} labels={'BIAS_UP': 95, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 1336} clean={'BIAS_UP': 95, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 478}
2026-05-11 01:46:20,087 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:20,089 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:20,090 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:20,091 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:20,091 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:20,092 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:46:20,404 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 52, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 1442}  ambiguous=913 (total=1494) horizon=36
2026-05-11 01:46:20,407 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.036, 'bias_down_score': 0.0} labels={'BIAS_UP': 52, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 1392} clean={'BIAS_UP': 52, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 510}
2026-05-11 01:46:20,484 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:20,486 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:20,487 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:20,487 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:20,488 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:20,489 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:46:20,800 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 40, 'BIAS_DOWN': 3, 'BIAS_NEUTRAL': 1451}  ambiguous=884 (total=1494) horizon=36
2026-05-11 01:46:20,803 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0277, 'bias_down_score': 0.0021} labels={'BIAS_UP': 40, 'BIAS_DOWN': 3, 'BIAS_NEUTRAL': 1401} clean={'BIAS_UP': 40, 'BIAS_DOWN': 3, 'BIAS_NEUTRAL': 542}
2026-05-11 01:46:20,875 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:20,877 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:20,878 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:20,878 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:20,878 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:20,879 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:46:21,178 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 18, 'BIAS_DOWN': 29, 'BIAS_NEUTRAL': 1441}  ambiguous=896 (total=1488) horizon=36
2026-05-11 01:46:21,181 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPUSD — 1438 samples (group=dollar) score_means={'bias_up_score': 0.0125, 'bias_down_score': 0.0202} labels={'BIAS_UP': 18, 'BIAS_DOWN': 29, 'BIAS_NEUTRAL': 1391} clean={'BIAS_UP': 18, 'BIAS_DOWN': 29, 'BIAS_NEUTRAL': 533}
2026-05-11 01:46:21,246 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 92, 'BIAS_DOWN': 3, 'BIAS_NEUTRAL': 2793}, 'dollar': {'BIAS_UP': 131, 'BIAS_DOWN': 75, 'BIAS_NEUTRAL': 4112}, 'gold': {'BIAS_UP': 53, 'BIAS_DOWN': 23, 'BIAS_NEUTRAL': 1455}}
2026-05-11 01:46:21,246 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0319, 'bias_down_score': 0.001}, 'dollar': {'bias_up_score': 0.0303, 'bias_down_score': 0.0174}, 'gold': {'bias_up_score': 0.0346, 'bias_down_score': 0.015}}
2026-05-11 01:46:21,246 INFO Regime[4H mode=htf_bias] label distribution by year: {2022: {'BIAS_UP': 72, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 3283}, 2023: {'BIAS_UP': 204, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 5077}}
2026-05-11 01:46:21,246 INFO Regime[4H mode=htf_bias] score means by year: {2022: {'bias_up_score': 0.0212, 'bias_down_score': 0.0135}, 2023: {'bias_up_score': 0.0382, 'bias_down_score': 0.0103}}
2026-05-11 01:46:21,291 INFO Regime phase HTF dataset build fold=train_all: 7.5s (train=68826 val=8737)
2026-05-11 01:46:21,291 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260511_014621
2026-05-11 01:46:21,492 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=28, n_classes=2)
2026-05-11 01:46:21,492 INFO Regime 4H/htf_bias warm start enabled from existing weights
2026-05-11 01:46:21,498 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=68826 val=8737 train_labels={'BIAS_UP': 2315, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 65057} val_labels={'BIAS_UP': 276, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 8360}
2026-05-11 01:46:21,498 INFO RegimeClassifier[mode=htf_bias]: warm start HTF score head
2026-05-11 01:46:21,498 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-11 01:46:21,499 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 28.73, 'bias_down_score': 30.0}
2026-05-11 01:46:21,502 INFO RegimeClassifier[mode=htf_bias]: HTF balanced sampler — dir=3769 neutral=65057 dir_weight=15 => dir_frac_per_epoch≈46.5%
2026-05-11 01:46:25,305 INFO Regime HTF score epoch  1/50 — tr=0.2963 va=0.7060 acc=0.871 bal=0.896 threshold=0.99 margin=0.00 recall={'BIAS_UP': 0.917, 'BIAS_DOWN': 0.901, 'BIAS_NEUTRAL': 0.869} precision={'BIAS_UP': 0.28, 'BIAS_DOWN': 0.171, 'BIAS_NEUTRAL': 0.995}
2026-05-11 01:46:26,965 INFO Regime HTF score epoch  2/50 — tr=0.2899 va=0.7064 bal=0.896
2026-05-11 01:46:28,626 INFO Regime HTF score epoch  3/50 — tr=0.2795 va=0.7139 bal=0.896
2026-05-11 01:46:30,325 INFO Regime HTF score epoch  4/50 — tr=0.2791 va=0.7144 bal=0.896
2026-05-11 01:46:31,982 INFO Regime HTF score epoch  5/50 — tr=0.2904 va=0.7109 acc=0.869 bal=0.894 threshold=0.99 margin=0.00 recall={'BIAS_UP': 0.913, 'BIAS_DOWN': 0.901, 'BIAS_NEUTRAL': 0.867} precision={'BIAS_UP': 0.278, 'BIAS_DOWN': 0.166, 'BIAS_NEUTRAL': 0.995}
2026-05-11 01:46:33,622 INFO Regime HTF score epoch  6/50 — tr=0.2812 va=0.7192 bal=0.898
2026-05-11 01:46:35,251 INFO Regime HTF score epoch  7/50 — tr=0.2825 va=0.7184 bal=0.896
2026-05-11 01:46:36,904 INFO Regime HTF score epoch  8/50 — tr=0.2860 va=0.7141 bal=0.896
2026-05-11 01:46:38,552 INFO Regime HTF score epoch  9/50 — tr=0.2764 va=0.7092 bal=0.895
2026-05-11 01:46:40,239 INFO Regime HTF score epoch 10/50 — tr=0.2831 va=0.7140 acc=0.869 bal=0.898 threshold=0.99 margin=0.00 recall={'BIAS_UP': 0.928, 'BIAS_DOWN': 0.901, 'BIAS_NEUTRAL': 0.867} precision={'BIAS_UP': 0.277, 'BIAS_DOWN': 0.169, 'BIAS_NEUTRAL': 0.996}
2026-05-11 01:46:41,903 INFO Regime HTF score epoch 11/50 — tr=0.2740 va=0.7187 bal=0.899
2026-05-11 01:46:43,554 INFO Regime HTF score epoch 12/50 — tr=0.2757 va=0.7192 bal=0.895
2026-05-11 01:46:45,202 INFO Regime HTF score epoch 13/50 — tr=0.2769 va=0.7124 bal=0.895
2026-05-11 01:46:46,825 INFO Regime HTF score epoch 14/50 — tr=0.2739 va=0.7020 bal=0.886
2026-05-11 01:46:48,459 INFO Regime HTF score epoch 15/50 — tr=0.2695 va=0.7064 acc=0.870 bal=0.895 threshold=0.99 margin=0.00 recall={'BIAS_UP': 0.917, 'BIAS_DOWN': 0.901, 'BIAS_NEUTRAL': 0.868} precision={'BIAS_UP': 0.279, 'BIAS_DOWN': 0.17, 'BIAS_NEUTRAL': 0.995}
2026-05-11 01:46:50,144 INFO Regime HTF score epoch 16/50 — tr=0.2699 va=0.7104 bal=0.895
2026-05-11 01:46:50,144 INFO Regime HTF score early stop at epoch 16
2026-05-11 01:46:51,636 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.990 margin=0.000 precision={'BIAS_UP': 0.275, 'BIAS_DOWN': 0.172, 'BIAS_NEUTRAL': 0.996} recall={'BIAS_UP': 0.92, 'BIAS_DOWN': 0.901, 'BIAS_NEUTRAL': 0.867} f1={'BIAS_UP': 0.423, 'BIAS_DOWN': 0.289, 'BIAS_NEUTRAL': 0.927} confusion=[[254, 0, 22], [0, 91, 10], [671, 437, 7252]] score_mae={'bias_up_score': 0.1869, 'bias_down_score': 0.1245} pred_share={'BIAS_UP': 0.1059, 'BIAS_DOWN': 0.0604, 'BIAS_NEUTRAL': 0.8337}
2026-05-11 01:46:51,637 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.275, 'BIAS_DOWN': 0.172, 'BIAS_NEUTRAL': 0.996} min_precision=0.500 recall={'BIAS_UP': 0.92, 'BIAS_DOWN': 0.901, 'BIAS_NEUTRAL': 0.867} min_recall=0.100 f1={'BIAS_UP': 0.423, 'BIAS_DOWN': 0.289, 'BIAS_NEUTRAL': 0.927} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_UP', 'BIAS_DOWN'] weak_recall=[] weak_f1=[] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-11 01:46:51,640 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 01:46:51,641 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 01:46:51,641 INFO Regime phase HTF train fold=train_all: 30.1s
2026-05-11 01:46:51,750 INFO Regime HTF complete fold=train_all: acc=0.870 bal=0.896 train=68826 val=8737 per_class={'BIAS_UP': 0.92, 'BIAS_DOWN': 0.901, 'BIAS_NEUTRAL': 0.867} precision={'BIAS_UP': 0.275, 'BIAS_DOWN': 0.172, 'BIAS_NEUTRAL': 0.996} threshold=0.990 margin=0.000
2026-05-11 01:46:51,752 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:46:51,930 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 255, 'BIAS_DOWN': 300, 'BIAS_NEUTRAL': 10847}  ambiguous=6860 (total=11402) horizon=36
2026-05-11 01:46:51,933 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on GBPUSD 4H:
{'BIAS_UP': 3.8636363636363638, 'BIAS_DOWN': 3.5294117647058822, 'BIAS_NEUTRAL': 71.36184210526316}
2026-05-11 01:46:51,937 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (all labels):
{'BIAS_UP': {'n': 255, 'mean': 0.0006942435718027296, 'mean_over_std': 0.3535182817741541}, 'BIAS_DOWN': {'n': 300, 'mean': -0.0007615786091842384, 'mean_over_std': -0.2905559984734911}, 'BIAS_NEUTRAL': {'n': 10846, 'mean': -5.264589798607426e-06, 'mean_over_std': -0.002007943010728463}}
2026-05-11 01:46:51,937 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 255, 'mean': 0.0006942435718027296, 'mean_over_std': 0.3535182817741541}, 'BIAS_DOWN': {'n': 300, 'mean': -0.0007615786091842384, 'mean_over_std': -0.2905559984734911}, 'BIAS_NEUTRAL': {'n': 3987, 'mean': -1.7489525578049586e-05, 'mean_over_std': -0.007235149795477735}}
2026-05-11 01:46:51,943 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-11 01:46:51,946 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:46:51,948 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:46:51,949 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:46:51,952 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:46:51,954 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:46:51,956 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:46:51,976 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:46:51,984 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:46:51,986 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:46:51,987 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:46:51,987 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:46:51,995 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:46:52,918 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected XAUUSD — 44227 samples (group=gold) score_means={'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}
2026-05-11 01:46:53,031 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:53,033 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:53,034 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:53,034 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:53,035 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:53,037 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:46:53,878 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURUSD — 43689 samples (group=dollar) score_means={'trend_score': 0.49, 'range_score': 0.236, 'chop_score': 0.4622, 'volatility_percentile': 0.3812, 'consolidation_score': 0.1862}
2026-05-11 01:46:53,993 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:53,995 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:53,996 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:53,997 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:53,997 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:54,000 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:46:54,851 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected USDJPY — 43686 samples (group=dollar) score_means={'trend_score': 0.4949, 'range_score': 0.2312, 'chop_score': 0.4613, 'volatility_percentile': 0.38, 'consolidation_score': 0.1931}
2026-05-11 01:46:54,971 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:54,974 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:54,975 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:54,975 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:54,975 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:54,978 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:46:55,843 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURJPY — 43688 samples (group=cross) score_means={'trend_score': 0.4888, 'range_score': 0.2345, 'chop_score': 0.4662, 'volatility_percentile': 0.3831, 'consolidation_score': 0.1882}
2026-05-11 01:46:55,957 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:55,959 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:55,960 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:55,960 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:55,960 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:55,963 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:46:56,818 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPJPY — 43678 samples (group=cross) score_means={'trend_score': 0.4878, 'range_score': 0.2348, 'chop_score': 0.4673, 'volatility_percentile': 0.3839, 'consolidation_score': 0.1904}
2026-05-11 01:46:56,939 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:56,942 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:56,943 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:56,943 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:56,943 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:56,946 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:46:57,796 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPUSD — 43676 samples (group=dollar) score_means={'trend_score': 0.4923, 'range_score': 0.2335, 'chop_score': 0.4611, 'volatility_percentile': 0.3832, 'consolidation_score': 0.1856}
2026-05-11 01:46:57,910 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2346, 'chop_score': 0.4667, 'volatility_percentile': 0.3835, 'consolidation_score': 0.1893}, 'dollar': {'trend_score': 0.4924, 'range_score': 0.2335, 'chop_score': 0.4615, 'volatility_percentile': 0.3815, 'consolidation_score': 0.1883}, 'gold': {'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}}
2026-05-11 01:46:57,911 INFO Regime[1H mode=ltf_behaviour] score means by year: {2016: {'trend_score': 0.495, 'range_score': 0.233, 'chop_score': 0.4607, 'volatility_percentile': 0.3776, 'consolidation_score': 0.192}, 2017: {'trend_score': 0.5017, 'range_score': 0.2328, 'chop_score': 0.4574, 'volatility_percentile': 0.3689, 'consolidation_score': 0.1942}, 2018: {'trend_score': 0.4886, 'range_score': 0.237, 'chop_score': 0.4655, 'volatility_percentile': 0.3947, 'consolidation_score': 0.1785}, 2019: {'trend_score': 0.4817, 'range_score': 0.236, 'chop_score': 0.4691, 'volatility_percentile': 0.3725, 'consolidation_score': 0.198}, 2020: {'trend_score': 0.4874, 'range_score': 0.2311, 'chop_score': 0.4674, 'volatility_percentile': 0.3848, 'consolidation_score': 0.1919}, 2021: {'trend_score': 0.4872, 'range_score': 0.2372, 'chop_score': 0.4672, 'volatility_percentile': 0.377, 'consolidation_score': 0.192}, 2022: {'trend_score': 0.4893, 'range_score': 0.2357, 'chop_score': 0.4647, 'volatility_percentile': 0.3962, 'consolidation_score': 0.1784}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 01:46:58,008 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
2026-05-11 01:46:58,010 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 01:46:58,011 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
2026-05-11 01:46:58,012 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 01:46:58,013 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
2026-05-11 01:46:58,015 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
2026-05-11 01:46:58,024 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:46:58,027 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:46:58,028 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:46:58,029 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:46:58,029 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 01:46:58,031 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:46:58,294 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected XAUUSD — 5117 samples (group=gold) score_means={'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}
2026-05-11 01:46:58,407 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:58,409 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:58,410 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:58,411 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:58,411 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:58,413 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:46:58,655 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURUSD — 5053 samples (group=dollar) score_means={'trend_score': 0.4883, 'range_score': 0.2413, 'chop_score': 0.463, 'volatility_percentile': 0.3837, 'consolidation_score': 0.1793}
2026-05-11 01:46:58,765 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:58,768 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:58,768 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:58,769 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:58,769 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:58,771 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:46:59,009 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected USDJPY — 5046 samples (group=dollar) score_means={'trend_score': 0.5085, 'range_score': 0.2259, 'chop_score': 0.4506, 'volatility_percentile': 0.3809, 'consolidation_score': 0.1888}
2026-05-11 01:46:59,124 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:59,127 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:59,128 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:59,128 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:59,128 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:59,130 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:46:59,371 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURJPY — 5053 samples (group=cross) score_means={'trend_score': 0.5023, 'range_score': 0.2261, 'chop_score': 0.4524, 'volatility_percentile': 0.3837, 'consolidation_score': 0.18}
2026-05-11 01:46:59,484 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:59,487 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:59,487 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:59,488 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:59,488 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:59,490 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:46:59,733 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPJPY — 5043 samples (group=cross) score_means={'trend_score': 0.4743, 'range_score': 0.2372, 'chop_score': 0.4716, 'volatility_percentile': 0.3777, 'consolidation_score': 0.1917}
2026-05-11 01:46:59,872 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:59,875 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:59,875 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:59,876 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:59,876 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 01:46:59,878 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:47:00,122 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPUSD — 5040 samples (group=dollar) score_means={'trend_score': 0.4843, 'range_score': 0.2371, 'chop_score': 0.4614, 'volatility_percentile': 0.377, 'consolidation_score': 0.1835}
2026-05-11 01:47:00,232 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2317, 'chop_score': 0.462, 'volatility_percentile': 0.3807, 'consolidation_score': 0.1858}, 'dollar': {'trend_score': 0.4937, 'range_score': 0.2348, 'chop_score': 0.4583, 'volatility_percentile': 0.3806, 'consolidation_score': 0.1839}, 'gold': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}}
2026-05-11 01:47:00,232 INFO Regime[1H mode=ltf_behaviour] score means by year: {2022: {'trend_score': 0.4905, 'range_score': 0.2364, 'chop_score': 0.4625, 'volatility_percentile': 0.385, 'consolidation_score': 0.1797}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 01:47:00,319 INFO Regime phase LTF dataset build fold=train_all: 8.4s (train=262644 val=30352)
2026-05-11 01:47:00,319 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260511_014700
2026-05-11 01:47:00,324 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=31, n_classes=5)
2026-05-11 01:47:00,325 INFO Regime 1H/ltf_behaviour warm start enabled from existing weights
2026-05-11 01:47:00,348 INFO RegimeClassifier[mode=ltf_behaviour]: warm start score head
2026-05-11 01:47:00,348 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-11 01:47:00,886 INFO Regime score epoch  1/50 — tr=0.0050 va=0.0017 mae={'trend_score': 0.0296, 'range_score': 0.0423, 'chop_score': 0.032, 'volatility_percentile': 0.0196, 'consolidation_score': 0.0306}
2026-05-11 01:47:01,390 INFO Regime score epoch  2/50 — tr=0.0050 va=0.0017
2026-05-11 01:47:01,895 INFO Regime score epoch  3/50 — tr=0.0050 va=0.0017
2026-05-11 01:47:02,428 INFO Regime score epoch  4/50 — tr=0.0050 va=0.0017
2026-05-11 01:47:02,949 INFO Regime score epoch  5/50 — tr=0.0049 va=0.0016 mae={'trend_score': 0.0294, 'range_score': 0.0426, 'chop_score': 0.0314, 'volatility_percentile': 0.0195, 'consolidation_score': 0.0286}
2026-05-11 01:47:03,471 INFO Regime score epoch  6/50 — tr=0.0049 va=0.0016
2026-05-11 01:47:03,991 INFO Regime score epoch  7/50 — tr=0.0048 va=0.0016
2026-05-11 01:47:04,515 INFO Regime score epoch  8/50 — tr=0.0048 va=0.0015
2026-05-11 01:47:05,024 INFO Regime score epoch  9/50 — tr=0.0047 va=0.0015
2026-05-11 01:47:05,531 INFO Regime score epoch 10/50 — tr=0.0047 va=0.0015 mae={'trend_score': 0.0272, 'range_score': 0.0407, 'chop_score': 0.0289, 'volatility_percentile': 0.0185, 'consolidation_score': 0.0262}
2026-05-11 01:47:06,064 INFO Regime score epoch 11/50 — tr=0.0046 va=0.0014
2026-05-11 01:47:06,592 INFO Regime score epoch 12/50 — tr=0.0046 va=0.0014
2026-05-11 01:47:07,113 INFO Regime score epoch 13/50 — tr=0.0045 va=0.0014
2026-05-11 01:47:07,649 INFO Regime score epoch 14/50 — tr=0.0045 va=0.0014
2026-05-11 01:47:08,164 INFO Regime score epoch 15/50 — tr=0.0044 va=0.0013 mae={'trend_score': 0.0252, 'range_score': 0.0396, 'chop_score': 0.0271, 'volatility_percentile': 0.0172, 'consolidation_score': 0.0251}
2026-05-11 01:47:08,686 INFO Regime score epoch 16/50 — tr=0.0044 va=0.0013
2026-05-11 01:47:09,213 INFO Regime score epoch 17/50 — tr=0.0043 va=0.0013
2026-05-11 01:47:09,746 INFO Regime score epoch 18/50 — tr=0.0043 va=0.0013
2026-05-11 01:47:10,298 INFO Regime score epoch 19/50 — tr=0.0043 va=0.0012
2026-05-11 01:47:10,813 INFO Regime score epoch 20/50 — tr=0.0042 va=0.0012 mae={'trend_score': 0.0237, 'range_score': 0.0381, 'chop_score': 0.0256, 'volatility_percentile': 0.0172, 'consolidation_score': 0.0234}
2026-05-11 01:47:11,328 INFO Regime score epoch 21/50 — tr=0.0042 va=0.0012
2026-05-11 01:47:11,834 INFO Regime score epoch 22/50 — tr=0.0042 va=0.0012
2026-05-11 01:47:12,371 INFO Regime score epoch 23/50 — tr=0.0041 va=0.0012
2026-05-11 01:47:12,895 INFO Regime score epoch 24/50 — tr=0.0041 va=0.0012
2026-05-11 01:47:13,430 INFO Regime score epoch 25/50 — tr=0.0041 va=0.0011 mae={'trend_score': 0.0223, 'range_score': 0.0374, 'chop_score': 0.0248, 'volatility_percentile': 0.0167, 'consolidation_score': 0.0227}
2026-05-11 01:47:13,957 INFO Regime score epoch 26/50 — tr=0.0041 va=0.0011
2026-05-11 01:47:14,484 INFO Regime score epoch 27/50 — tr=0.0041 va=0.0011
2026-05-11 01:47:15,000 INFO Regime score epoch 28/50 — tr=0.0040 va=0.0011
2026-05-11 01:47:15,516 INFO Regime score epoch 29/50 — tr=0.0040 va=0.0011
2026-05-11 01:47:16,018 INFO Regime score epoch 30/50 — tr=0.0040 va=0.0011 mae={'trend_score': 0.0214, 'range_score': 0.0371, 'chop_score': 0.0244, 'volatility_percentile': 0.0166, 'consolidation_score': 0.0225}
2026-05-11 01:47:16,529 INFO Regime score epoch 31/50 — tr=0.0040 va=0.0011
2026-05-11 01:47:17,048 INFO Regime score epoch 32/50 — tr=0.0040 va=0.0011
2026-05-11 01:47:17,559 INFO Regime score epoch 33/50 — tr=0.0040 va=0.0011
2026-05-11 01:47:18,071 INFO Regime score epoch 34/50 — tr=0.0039 va=0.0011
2026-05-11 01:47:18,588 INFO Regime score epoch 35/50 — tr=0.0039 va=0.0011 mae={'trend_score': 0.0211, 'range_score': 0.0364, 'chop_score': 0.0237, 'volatility_percentile': 0.0164, 'consolidation_score': 0.0218}
2026-05-11 01:47:19,106 INFO Regime score epoch 36/50 — tr=0.0039 va=0.0011
2026-05-11 01:47:19,622 INFO Regime score epoch 37/50 — tr=0.0039 va=0.0011
2026-05-11 01:47:20,166 INFO Regime score epoch 38/50 — tr=0.0039 va=0.0011
2026-05-11 01:47:20,690 INFO Regime score epoch 39/50 — tr=0.0039 va=0.0011
2026-05-11 01:47:21,193 INFO Regime score epoch 40/50 — tr=0.0039 va=0.0010 mae={'trend_score': 0.0206, 'range_score': 0.0357, 'chop_score': 0.0233, 'volatility_percentile': 0.0162, 'consolidation_score': 0.0217}
2026-05-11 01:47:21,694 INFO Regime score epoch 41/50 — tr=0.0039 va=0.0011
2026-05-11 01:47:22,218 INFO Regime score epoch 42/50 — tr=0.0039 va=0.0010
2026-05-11 01:47:22,730 INFO Regime score epoch 43/50 — tr=0.0039 va=0.0010
2026-05-11 01:47:23,231 INFO Regime score epoch 44/50 — tr=0.0039 va=0.0010
2026-05-11 01:47:23,729 INFO Regime score epoch 45/50 — tr=0.0039 va=0.0011 mae={'trend_score': 0.0207, 'range_score': 0.036, 'chop_score': 0.0231, 'volatility_percentile': 0.0162, 'consolidation_score': 0.0224}
2026-05-11 01:47:24,228 INFO Regime score epoch 46/50 — tr=0.0039 va=0.0010
2026-05-11 01:47:24,740 INFO Regime score epoch 47/50 — tr=0.0039 va=0.0010
2026-05-11 01:47:25,255 INFO Regime score epoch 48/50 — tr=0.0039 va=0.0010
2026-05-11 01:47:25,763 INFO Regime score epoch 49/50 — tr=0.0039 va=0.0011
2026-05-11 01:47:26,272 INFO Regime score epoch 50/50 — tr=0.0039 va=0.0011 mae={'trend_score': 0.0208, 'range_score': 0.0358, 'chop_score': 0.0232, 'volatility_percentile': 0.0165, 'consolidation_score': 0.0231}
2026-05-11 01:47:26,293 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0206, 'range_score': 0.0361, 'chop_score': 0.0231, 'volatility_percentile': 0.0162, 'consolidation_score': 0.0214} mse={'trend_score': 0.00073, 'range_score': 0.00213, 'chop_score': 0.00087, 'volatility_percentile': 0.00046, 'consolidation_score': 0.00102} corr={'trend_score': 0.9926, 'range_score': 0.947, 'chop_score': 0.9882, 'volatility_percentile': 0.9952, 'consolidation_score': 0.9892} pred_std={'trend_score': 0.2208, 'range_score': 0.1356, 'chop_score': 0.181, 'volatility_percentile': 0.2185, 'consolidation_score': 0.2143} target_std={'trend_score': 0.2219, 'range_score': 0.1433, 'chop_score': 0.1888, 'volatility_percentile': 0.2165, 'consolidation_score': 0.2179}
2026-05-11 01:47:26,604 INFO RegimeClassifier[mode=ltf_behaviour] train group diagnostics={'dollar': {'n': 16632, 'score_mae': {'trend_score': 0.0201, 'range_score': 0.0358, 'chop_score': 0.0228, 'volatility_percentile': 0.0159, 'consolidation_score': 0.0218}, 'target_score_mean': {'trend_score': 0.491, 'range_score': 0.2347, 'chop_score': 0.4613, 'volatility_percentile': 0.3824, 'consolidation_score': 0.186}, 'pred_score_mean': {'trend_score': 0.4911, 'range_score': 0.2379, 'chop_score': 0.4607, 'volatility_percentile': 0.3815, 'consolidation_score': 0.1838}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3567, 62, 0, 2, 0, 0, 148], [5, 101, 0, 0, 0, 0, 4], [0, 0, 226, 10, 47, 0, 177], [3, 0, 12, 548, 35, 0, 91], [0, 0, 50, 21, 3050, 0, 195], [0, 25, 0, 0, 7, 27, 69], [162, 13, 171, 54, 97, 0, 7653]]}, 'gold': {'n': 8420, 'score_mae': {'trend_score': 0.0199, 'range_score': 0.0365, 'chop_score': 0.0231, 'volatility_percentile': 0.0167, 'consolidation_score': 0.0222}, 'target_score_mean': {'trend_score': 0.4883, 'range_score': 0.2355, 'chop_score': 0.4652, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}, 'pred_score_mean': {'trend_score': 0.4892, 'range_score': 0.2387, 'chop_score': 0.464, 'volatility_percentile': 0.376, 'consolidation_score': 0.1893}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1784, 36, 0, 0, 0, 0, 65], [4, 49, 0, 0, 0, 0, 3], [0, 0, 119, 8, 24, 0, 93], [1, 0, 8, 331, 22, 0, 54], [0, 0, 29, 23, 1547, 0, 105], [0, 18, 0, 0, 7, 24, 32], [77, 4, 94, 24, 55, 0, 3780]]}, 'jpy': {'n': 24948, 'score_mae': {'trend_score': 0.0201, 'range_score': 0.0359, 'chop_score': 0.0227, 'volatility_percentile': 0.0165, 'consolidation_score': 0.0217}, 'target_score_mean': {'trend_score': 0.4904, 'range_score': 0.2338, 'chop_score': 0.4653, 'volatility_percentile': 0.3826, 'consolidation_score': 0.1905}, 'pred_score_mean': {'trend_score': 0.4902, 'range_score': 0.2374, 'chop_score': 0.464, 'volatility_percentile': 0.3816, 'consolidation_score': 0.1877}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[5313, 124, 0, 3, 0, 0, 206], [10, 168, 0, 0, 0, 0, 9], [0, 0, 291, 19, 68, 0, 269], [1, 0, 11, 1069, 85, 0, 148], [0, 0, 82, 59, 4643, 0, 331], [0, 46, 0, 0, 16, 48, 113], [226, 18, 275, 93, 178, 0, 11026]]}}
2026-05-11 01:47:26,779 INFO RegimeClassifier[mode=ltf_behaviour] validation group diagnostics={'dollar': {'n': 10093, 'score_mae': {'trend_score': 0.0209, 'range_score': 0.0372, 'chop_score': 0.0231, 'volatility_percentile': 0.0159, 'consolidation_score': 0.0209}, 'target_score_mean': {'trend_score': 0.4863, 'range_score': 0.2392, 'chop_score': 0.4622, 'volatility_percentile': 0.3804, 'consolidation_score': 0.1814}, 'pred_score_mean': {'trend_score': 0.4869, 'range_score': 0.2406, 'chop_score': 0.4622, 'volatility_percentile': 0.3795, 'consolidation_score': 0.1802}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[2295, 21, 0, 2, 0, 0, 94], [4, 48, 0, 0, 0, 0, 1], [0, 0, 146, 8, 45, 0, 117], [0, 0, 9, 328, 26, 0, 60], [0, 0, 34, 28, 1895, 0, 93], [0, 18, 0, 0, 3, 18, 38], [66, 3, 112, 50, 63, 0, 4468]]}, 'gold': {'n': 5117, 'score_mae': {'trend_score': 0.0197, 'range_score': 0.0353, 'chop_score': 0.0229, 'volatility_percentile': 0.0166, 'consolidation_score': 0.022}, 'target_score_mean': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}, 'pred_score_mean': {'trend_score': 0.4965, 'range_score': 0.237, 'chop_score': 0.4567, 'volatility_percentile': 0.381, 'consolidation_score': 0.181}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1099, 18, 0, 0, 0, 0, 50], [3, 30, 0, 0, 0, 0, 2], [0, 0, 88, 2, 11, 0, 70], [0, 0, 4, 221, 9, 0, 21], [0, 0, 22, 10, 792, 0, 63], [0, 7, 0, 0, 4, 13, 26], [45, 3, 70, 29, 33, 0, 2372]]}, 'jpy': {'n': 15142, 'score_mae': {'trend_score': 0.0207, 'range_score': 0.0357, 'chop_score': 0.0231, 'volatility_percentile': 0.0163, 'consolidation_score': 0.0215}, 'target_score_mean': {'trend_score': 0.495, 'range_score': 0.2297, 'chop_score': 0.4582, 'volatility_percentile': 0.3808, 'consolidation_score': 0.1868}, 'pred_score_mean': {'trend_score': 0.4938, 'range_score': 0.233, 'chop_score': 0.4582, 'volatility_percentile': 0.3801, 'consolidation_score': 0.1848}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3315, 71, 0, 2, 0, 0, 130], [5, 105, 0, 0, 0, 0, 5], [0, 0, 178, 12, 46, 0, 148], [3, 0, 17, 671, 43, 0, 93], [0, 0, 50, 26, 2555, 0, 186], [0, 24, 0, 0, 8, 30, 60], [115, 10, 157, 55, 119, 0, 6903]]}}
2026-05-11 01:47:26,784 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-11 01:47:26,785 INFO Regime phase LTF train fold=train_all: 26.5s
2026-05-11 01:47:26,897 INFO Regime LTF complete fold=train_all: score_accuracy=0.977, train=262644 val=30352 mae={'trend_score': 0.0206, 'range_score': 0.0361, 'chop_score': 0.0231, 'volatility_percentile': 0.0162, 'consolidation_score': 0.0214}
2026-05-11 01:47:26,900 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 01:47:27,271 INFO Regime[1H mode=ltf_behaviour fold=latest] LTF score diagnostics on GBPUSD:
{'trend_score': {'mean': 0.4921, 'q10': 0.1918, 'q50': 0.4877, 'q90': 0.8002}, 'range_score': {'mean': 0.2337, 'q10': 0.0513, 'q50': 0.2125, 'q90': 0.4349}, 'chop_score': {'mean': 0.4612, 'q10': 0.2148, 'q50': 0.4503, 'q90': 0.7255}, 'volatility_percentile': {'mean': 0.3832, 'q10': 0.1018, 'q50': 0.3704, 'q90': 0.6823}, 'consolidation_score': {'mean': 0.1854, 'q10': 0.0, 'q50': 0.1185, 'q90': 0.506}}
2026-05-11 01:47:27,277 INFO Regime retrain total: 73.9s (370559 train+val samples)
2026-05-11 01:47:27,283 INFO Retrain complete. Total wall-clock: 73.9s
  DONE  Retrain regime [train-split retrain]
  SKIP  Quality/RL incremental retrain — clean train-only weights retained

=== Round 3: Backtest on last 3yr (post-retrain evaluation) ===
  START Round 3 - Post-retrain backtest (last 3yr)
2026-05-11 01:47:28,820 INFO === STEP 6: BACKTEST (round3) ===
2026-05-11 01:47:28,822 INFO BT_WINDOW=round3 — post-retrain eval: 2022-08-05 → 2025-08-05 (last 3yr)
2026-05-11 01:47:28,822 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-05-11 01:47:28,822 INFO Round 3 — running backtest: 2022-08-05 → 2025-08-05 (ml_trader, shared ML cache)
2026-05-11 01:47:28,822 INFO Round 3 — QualityScorer hard gate disabled; set BACKTEST_ENABLE_QUALITY_GATE=1 to enable it
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
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 01:49:14,769 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 01:49:15,762 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 01:49:16,504 WARNING _build_sequence_df: HTF frame 5M filled 461 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 01:49:16,932 WARNING _build_sequence_df: HTF frame 5M filled 461 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 01:49:17,061 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 01:49:17,144 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 01:49:17,198 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 01:49:17,278 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
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
2026-05-11 01:49:28,128 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 01:49:28,574 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 01:49:28,924 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 01:49:28,960 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 01:50:25,400 INFO Round 3 backtest — 358 trades | avg WR=27.9% | avg PF=0.99 | avg Sharpe=-0.07
2026-05-11 01:50:25,400 INFO   ml_trader: 358 trades | WR=27.9% | fixed PF=0.99 | Return=-2.7% | ExpR=-0.007 | DD=34.2% | Sharpe=-0.07
2026-05-11 01:50:25,400 INFO   ml_trader gate_diagnostics: bars=403523 no_signal=182939 quality_block=0 session_skip=220090 density=136 pm_reject=0
2026-05-11 01:50:25,400 INFO   ml_trader no_signal_reasons: {'weak_gru_direction': 39395, 'gru_expected_r_below_threshold': 78429, 'no_trade_uncertain': 41435, 'no_trade_chop': 10090, 'no_trade_extreme_vol': 11118, 'wait_pullback': 1802, 'tradeability_direction_conflict': 656, 'expected_r_below_threshold': 14}
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 358
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (358 rows)
  DONE  Round 3 - Post-retrain backtest (last 3yr)
  Saved Round 3 result → round3_summary.json
  Journal after Round 3: 730 entries

  SKIP  Round 3 Quality+RL retrain — evaluation journals not used for fitting

======================================================================
  BLIND BACKTEST PIPELINE COMPLETE
======================================================================
  Round 1 (train-tail window)   trades=149  WR=28.9%  PF=1.175  Sharpe=1.073
  Round 2 (blind test)          trades=223  WR=26.9%  PF=1.007  Sharpe=0.046
  Round 3 (last 3yr)            trades=358  WR=27.9%  PF=0.990  Sharpe=-0.071


WARNING: GITHUB_TOKEN not set — skipping GitHub push
2026-05-11 01:50:25,911 INFO Round 3: wrote 358 journal entries (total in file: 730)