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
2026-05-11 13:39:35,922 INFO Loading feature-engineered data...
2026-05-11 13:39:36,531 INFO Loaded 221743 rows, 202 features
2026-05-11 13:39:36,533 INFO Data span: 2016-01-04 -> 2025-08-05  (9.6 years)
2026-05-11 13:39:36,535 INFO Fold 000 train 2016-01-04 -> 2018-01-03 (47088 bars), val 2018-01-04 -> 2019-01-03 (23448 bars)
2026-05-11 13:39:36,536 INFO Fold 001 train 2016-01-04 -> 2019-01-03 (70536 bars), val 2019-01-04 -> 2020-01-03 (23377 bars)
2026-05-11 13:39:36,536 INFO Fold 002 train 2016-01-04 -> 2020-01-03 (93913 bars), val 2020-01-06 -> 2020-12-31 (23259 bars)
2026-05-11 13:39:36,536 INFO Fold 003 train 2016-01-04 -> 2020-12-31 (117172 bars), val 2021-01-04 -> 2022-01-03 (23507 bars)
2026-05-11 13:39:36,537 INFO Fold 004 train 2016-01-04 -> 2022-01-03 (140679 bars), val 2022-01-04 -> 2023-01-03 (23588 bars)
2026-05-11 13:39:36,537 INFO Fold 005 train 2016-01-04 -> 2022-08-05 (154539 bars), val 2022-08-05 -> 2023-08-04 (20412 bars)
2026-05-11 13:39:36,537 INFO No leakage confirmed: train/train_tail/internal folds end before final 2-year blind test

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
2026-05-11 13:39:45,907 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-05-11 13:39:45,907 INFO --- Training gru ---
2026-05-11 13:39:45,907 INFO Running retrain --model gru
2026-05-11 13:39:46,092 INFO retrain environment: KAGGLE
2026-05-11 13:39:47,749 INFO Device: CUDA (2 GPU(s))
2026-05-11 13:39:47,760 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 13:39:47,760 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 13:39:47,760 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 13:39:47,764 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 13:39:47,764 INFO Retrain data split: train
2026-05-11 13:39:47,764 INFO Retrain rolling fold selector: latest
2026-05-11 13:39:47,765 INFO === GRU-LSTM retrain (timeframes: ['15M']) ===
2026-05-11 13:39:47,918 INFO NumExpr defaulting to 4 threads.
2026-05-11 13:39:48,145 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-11 13:39:48,145 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 13:39:48,146 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 13:39:48,146 INFO GRU multi-symbol training (Kaggle mode): 6 symbols × ['15M']
2026-05-11 13:39:48,146 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260511_133948
2026-05-11 13:39:48,149 WARNING WeightsManifest: no manifest at /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json — assuming compatible (run a full retrain to generate one)
2026-05-11 13:39:48,150 INFO GRU warm start disabled by default; set GRU_ALLOW_WARM_START=1 to reuse compatible weights
2026-05-11 13:39:48,429 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 13:39:48,465 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 13:39:48,482 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 13:39:48,492 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 13:39:48,562 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 13:39:48,567 INFO Loaded XAUUSD/15M split=train fold=latest: 176438 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 13:39:49,120 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 13:39:49,139 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 13:39:49,153 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 13:39:49,160 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 13:39:49,201 INFO Loaded EURUSD/15M split=train fold=latest: 174951 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 13:39:49,752 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 13:39:49,771 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 13:39:49,786 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 13:39:49,793 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 13:39:49,830 INFO Loaded USDJPY/15M split=train fold=latest: 174944 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 13:39:50,377 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 13:39:50,397 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 13:39:50,412 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 13:39:50,419 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 13:39:50,456 INFO Loaded EURJPY/15M split=train fold=latest: 174948 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 13:39:50,960 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 13:39:50,979 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 13:39:50,994 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 13:39:51,002 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 13:39:51,042 INFO Loaded GBPJPY/15M split=train fold=latest: 174902 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 13:39:51,554 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 13:39:51,572 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 13:39:51,585 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 13:39:51,592 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 13:39:51,629 INFO Loaded GBPUSD/15M split=train fold=latest: 174899 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 13:39:52,037 INFO train_multi: 6 segments, ~971854 total bars
2026-05-11 13:39:52,412 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-05-11 13:39:52,412 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-05-11 13:39:52,412 INFO train_multi: training ALL 6 segments across TFs ['15M'] in one combined pass
2026-05-11 13:39:52,412 INFO train_multi: building combined dataset for TF=ALL (6 segments)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 13:40:04,241 INFO train_multi TF=ALL: 971494 sequences across 6 segments
2026-05-11 13:40:04,241 INFO train_multi TF=ALL: estimated peak RAM = 27072 MB (train=419996 calib=60000 val=120002 n_feat=94 seq_len=60)
2026-05-11 13:40:04,241 WARNING train_multi TF=ALL: trimming to fit RAM budget — new train=310283 calib=44326 val=88652 (20000 MB est)
2026-05-11 13:40:06,618 INFO train_multi TF=ALL: train=310283 calib=44326 val=88652 (10007 MB tensors)
2026-05-11 13:40:13,572 INFO train_multi TF=ALL: structural bar weighting — 199279 structural bars (64.2%) weight=15.0 structural_only=0
2026-05-11 13:40:16,633 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=25, min_epochs=45)
2026-05-11 13:40:32,659 INFO train_multi TF=ALL epoch 1/100 train=2.3341 val=2.3369 r_mae=0.970 pos_r_acc=0.545 side_acc=0.510 r_n=127469
2026-05-11 13:40:32,669 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:40:32,670 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:40:32,670 INFO train_multi TF=ALL: new best val=2.3369 r_mae=0.9699 — saved
2026-05-11 13:40:32,673 INFO train_multi TF=ALL: new best r_mae=0.9699 — saved rmae checkpoint
2026-05-11 13:40:45,413 INFO train_multi TF=ALL epoch 2/100 train=2.3328 val=2.3357 r_mae=0.969 pos_r_acc=0.545 side_acc=0.510 r_n=127469
2026-05-11 13:40:45,418 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:40:45,418 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:40:45,419 INFO train_multi TF=ALL: new best val=2.3357 r_mae=0.9688 — saved
2026-05-11 13:40:45,423 INFO train_multi TF=ALL: new best r_mae=0.9688 — saved rmae checkpoint
2026-05-11 13:40:58,232 INFO train_multi TF=ALL epoch 3/100 train=2.3321 val=2.3344 r_mae=0.967 pos_r_acc=0.545 side_acc=0.510 r_n=127469
2026-05-11 13:40:58,238 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:40:58,238 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:40:58,238 INFO train_multi TF=ALL: new best val=2.3344 r_mae=0.9675 — saved
2026-05-11 13:40:58,243 INFO train_multi TF=ALL: new best r_mae=0.9675 — saved rmae checkpoint
2026-05-11 13:41:11,225 INFO train_multi TF=ALL epoch 4/100 train=2.3316 val=2.3334 r_mae=0.966 pos_r_acc=0.545 side_acc=0.510 r_n=127469
2026-05-11 13:41:11,230 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:41:11,230 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:41:11,230 INFO train_multi TF=ALL: new best val=2.3334 r_mae=0.9665 — saved
2026-05-11 13:41:11,234 INFO train_multi TF=ALL: new best r_mae=0.9665 — saved rmae checkpoint
2026-05-11 13:41:23,967 INFO train_multi TF=ALL epoch 5/100 train=2.3308 val=2.3326 r_mae=0.966 pos_r_acc=0.545 side_acc=0.510 r_n=127469
2026-05-11 13:41:23,972 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:41:23,972 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:41:23,972 INFO train_multi TF=ALL: new best val=2.3326 r_mae=0.9657 — saved
2026-05-11 13:41:23,976 INFO train_multi TF=ALL: new best r_mae=0.9657 — saved rmae checkpoint
2026-05-11 13:41:36,974 INFO train_multi TF=ALL epoch 6/100 train=2.3306 val=2.3320 r_mae=0.965 pos_r_acc=0.545 side_acc=0.518 r_n=127469
2026-05-11 13:41:36,980 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:41:36,980 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:41:36,980 INFO train_multi TF=ALL: new best val=2.3320 r_mae=0.9654 — saved
2026-05-11 13:41:36,985 INFO train_multi TF=ALL: new best r_mae=0.9654 — saved rmae checkpoint
2026-05-11 13:41:49,857 INFO train_multi TF=ALL epoch 7/100 train=2.3298 val=2.3305 r_mae=0.965 pos_r_acc=0.545 side_acc=0.524 r_n=127469
2026-05-11 13:41:49,868 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:41:49,868 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:41:49,868 INFO train_multi TF=ALL: new best val=2.3305 r_mae=0.9648 — saved
2026-05-11 13:41:49,873 INFO train_multi TF=ALL: new best r_mae=0.9648 — saved rmae checkpoint
2026-05-11 13:42:02,785 INFO train_multi TF=ALL epoch 8/100 train=2.3280 val=2.3270 r_mae=0.964 pos_r_acc=0.545 side_acc=0.530 r_n=127469
2026-05-11 13:42:02,790 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:42:02,790 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:42:02,790 INFO train_multi TF=ALL: new best val=2.3270 r_mae=0.9640 — saved
2026-05-11 13:42:02,794 INFO train_multi TF=ALL: new best r_mae=0.9640 — saved rmae checkpoint
2026-05-11 13:42:15,657 INFO train_multi TF=ALL epoch 9/100 train=2.3252 val=2.3255 r_mae=0.963 pos_r_acc=0.547 side_acc=0.523 r_n=127469
2026-05-11 13:42:15,663 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:42:15,663 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:42:15,663 INFO train_multi TF=ALL: new best val=2.3255 r_mae=0.9632 — saved
2026-05-11 13:42:15,667 INFO train_multi TF=ALL: new best r_mae=0.9632 — saved rmae checkpoint
2026-05-11 13:42:28,476 INFO train_multi TF=ALL epoch 10/100 train=2.3229 val=2.3247 r_mae=0.962 pos_r_acc=0.547 side_acc=0.523 r_n=127469
2026-05-11 13:42:28,481 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:42:28,482 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:42:28,482 INFO train_multi TF=ALL: new best val=2.3247 r_mae=0.9620 — saved
2026-05-11 13:42:28,486 INFO train_multi TF=ALL: new best r_mae=0.9620 — saved rmae checkpoint
2026-05-11 13:42:41,402 INFO train_multi TF=ALL epoch 11/100 train=2.3204 val=2.3246 r_mae=0.961 pos_r_acc=0.549 side_acc=0.523 r_n=127469
2026-05-11 13:42:41,407 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:42:41,407 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:42:41,407 INFO train_multi TF=ALL: new best val=2.3246 r_mae=0.9614 — saved
2026-05-11 13:42:41,411 INFO train_multi TF=ALL: new best r_mae=0.9614 — saved rmae checkpoint
2026-05-11 13:42:54,385 INFO train_multi TF=ALL epoch 12/100 train=2.3195 val=2.3228 r_mae=0.960 pos_r_acc=0.551 side_acc=0.528 r_n=127469
2026-05-11 13:42:54,390 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:42:54,390 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:42:54,391 INFO train_multi TF=ALL: new best val=2.3228 r_mae=0.9602 — saved
2026-05-11 13:42:54,395 INFO train_multi TF=ALL: new best r_mae=0.9602 — saved rmae checkpoint
2026-05-11 13:43:07,158 INFO train_multi TF=ALL epoch 13/100 train=2.3182 val=2.3205 r_mae=0.960 pos_r_acc=0.550 side_acc=0.527 r_n=127469
2026-05-11 13:43:07,168 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:43:07,168 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:43:07,168 INFO train_multi TF=ALL: new best val=2.3205 r_mae=0.9602 — saved
2026-05-11 13:43:07,173 INFO train_multi TF=ALL: new best r_mae=0.9602 — saved rmae checkpoint
2026-05-11 13:43:20,125 INFO train_multi TF=ALL epoch 14/100 train=2.3156 val=2.3191 r_mae=0.959 pos_r_acc=0.551 side_acc=0.529 r_n=127469
2026-05-11 13:43:20,130 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:43:20,130 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:43:20,130 INFO train_multi TF=ALL: new best val=2.3191 r_mae=0.9592 — saved
2026-05-11 13:43:20,134 INFO train_multi TF=ALL: new best r_mae=0.9592 — saved rmae checkpoint
2026-05-11 13:43:32,926 INFO train_multi TF=ALL epoch 15/100 train=2.3141 val=2.3173 r_mae=0.959 pos_r_acc=0.550 side_acc=0.531 r_n=127469
2026-05-11 13:43:32,931 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:43:32,931 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:43:32,931 INFO train_multi TF=ALL: new best val=2.3173 r_mae=0.9595 — saved
2026-05-11 13:43:45,843 INFO train_multi TF=ALL epoch 16/100 train=2.3129 val=2.3145 r_mae=0.959 pos_r_acc=0.552 side_acc=0.535 r_n=127469
2026-05-11 13:43:45,849 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:43:45,849 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:43:45,849 INFO train_multi TF=ALL: new best val=2.3145 r_mae=0.9586 — saved
2026-05-11 13:43:45,853 INFO train_multi TF=ALL: new best r_mae=0.9586 — saved rmae checkpoint
2026-05-11 13:43:58,679 INFO train_multi TF=ALL epoch 17/100 train=2.3102 val=2.3155 r_mae=0.958 pos_r_acc=0.553 side_acc=0.533 r_n=127469
2026-05-11 13:43:58,684 INFO train_multi TF=ALL: new best r_mae=0.9576 — saved rmae checkpoint
2026-05-11 13:44:11,423 INFO train_multi TF=ALL epoch 18/100 train=2.3093 val=2.3141 r_mae=0.957 pos_r_acc=0.553 side_acc=0.534 r_n=127469
2026-05-11 13:44:11,428 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:44:11,428 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:44:11,428 INFO train_multi TF=ALL: new best val=2.3141 r_mae=0.9568 — saved
2026-05-11 13:44:11,432 INFO train_multi TF=ALL: new best r_mae=0.9568 — saved rmae checkpoint
2026-05-11 13:44:24,338 INFO train_multi TF=ALL epoch 19/100 train=2.3066 val=2.3122 r_mae=0.957 pos_r_acc=0.555 side_acc=0.536 r_n=127469
2026-05-11 13:44:24,343 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:44:24,343 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:44:24,343 INFO train_multi TF=ALL: new best val=2.3122 r_mae=0.9569 — saved
2026-05-11 13:44:37,142 INFO train_multi TF=ALL epoch 20/100 train=2.3035 val=2.3145 r_mae=0.955 pos_r_acc=0.556 side_acc=0.535 r_n=127469
2026-05-11 13:44:37,147 INFO train_multi TF=ALL: new best r_mae=0.9547 — saved rmae checkpoint
2026-05-11 13:44:50,012 INFO train_multi TF=ALL epoch 21/100 train=2.3014 val=2.3109 r_mae=0.956 pos_r_acc=0.557 side_acc=0.540 r_n=127469
2026-05-11 13:44:50,018 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:44:50,018 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:44:50,018 INFO train_multi TF=ALL: new best val=2.3109 r_mae=0.9556 — saved
2026-05-11 13:45:02,809 INFO train_multi TF=ALL epoch 22/100 train=2.2979 val=2.3033 r_mae=0.951 pos_r_acc=0.564 side_acc=0.545 r_n=127469
2026-05-11 13:45:02,814 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:45:02,814 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:45:02,814 INFO train_multi TF=ALL: new best val=2.3033 r_mae=0.9512 — saved
2026-05-11 13:45:02,818 INFO train_multi TF=ALL: new best r_mae=0.9512 — saved rmae checkpoint
2026-05-11 13:45:15,674 INFO train_multi TF=ALL epoch 23/100 train=2.2904 val=2.2937 r_mae=0.947 pos_r_acc=0.570 side_acc=0.546 r_n=127469
2026-05-11 13:45:15,679 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:45:15,679 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:45:15,679 INFO train_multi TF=ALL: new best val=2.2937 r_mae=0.9468 — saved
2026-05-11 13:45:15,683 INFO train_multi TF=ALL: new best r_mae=0.9468 — saved rmae checkpoint
2026-05-11 13:45:28,499 INFO train_multi TF=ALL epoch 24/100 train=2.2818 val=2.2869 r_mae=0.945 pos_r_acc=0.571 side_acc=0.548 r_n=127469
2026-05-11 13:45:28,505 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:45:28,505 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:45:28,505 INFO train_multi TF=ALL: new best val=2.2869 r_mae=0.9446 — saved
2026-05-11 13:45:28,510 INFO train_multi TF=ALL: new best r_mae=0.9446 — saved rmae checkpoint
2026-05-11 13:45:41,631 INFO train_multi TF=ALL epoch 25/100 train=2.2718 val=2.2763 r_mae=0.941 pos_r_acc=0.581 side_acc=0.553 r_n=127469
2026-05-11 13:45:41,636 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:45:41,636 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:45:41,636 INFO train_multi TF=ALL: new best val=2.2763 r_mae=0.9412 — saved
2026-05-11 13:45:41,641 INFO train_multi TF=ALL: new best r_mae=0.9412 — saved rmae checkpoint
2026-05-11 13:45:54,587 INFO train_multi TF=ALL epoch 26/100 train=2.2636 val=2.2735 r_mae=0.938 pos_r_acc=0.583 side_acc=0.550 r_n=127469
2026-05-11 13:45:54,592 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:45:54,592 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:45:54,592 INFO train_multi TF=ALL: new best val=2.2735 r_mae=0.9380 — saved
2026-05-11 13:45:54,596 INFO train_multi TF=ALL: new best r_mae=0.9380 — saved rmae checkpoint
2026-05-11 13:46:07,497 INFO train_multi TF=ALL epoch 27/100 train=2.2547 val=2.2699 r_mae=0.936 pos_r_acc=0.584 side_acc=0.554 r_n=127469
2026-05-11 13:46:07,502 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:46:07,503 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:46:07,503 INFO train_multi TF=ALL: new best val=2.2699 r_mae=0.9355 — saved
2026-05-11 13:46:07,507 INFO train_multi TF=ALL: new best r_mae=0.9355 — saved rmae checkpoint
2026-05-11 13:46:20,339 INFO train_multi TF=ALL epoch 28/100 train=2.2486 val=2.2633 r_mae=0.932 pos_r_acc=0.585 side_acc=0.558 r_n=127469
2026-05-11 13:46:20,345 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:46:20,345 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:46:20,345 INFO train_multi TF=ALL: new best val=2.2633 r_mae=0.9322 — saved
2026-05-11 13:46:20,349 INFO train_multi TF=ALL: new best r_mae=0.9322 — saved rmae checkpoint
2026-05-11 13:46:33,097 INFO train_multi TF=ALL epoch 29/100 train=2.2433 val=2.2713 r_mae=0.931 pos_r_acc=0.586 side_acc=0.552 r_n=127469
2026-05-11 13:46:33,108 INFO train_multi TF=ALL: new best r_mae=0.9312 — saved rmae checkpoint
2026-05-11 13:46:46,161 INFO train_multi TF=ALL epoch 30/100 train=2.2382 val=2.2643 r_mae=0.930 pos_r_acc=0.586 side_acc=0.555 r_n=127469
2026-05-11 13:46:46,166 INFO train_multi TF=ALL: new best r_mae=0.9297 — saved rmae checkpoint
2026-05-11 13:46:59,025 INFO train_multi TF=ALL epoch 31/100 train=2.2288 val=2.2622 r_mae=0.929 pos_r_acc=0.587 side_acc=0.557 r_n=127469
2026-05-11 13:46:59,030 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:46:59,030 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:46:59,030 INFO train_multi TF=ALL: new best val=2.2622 r_mae=0.9288 — saved
2026-05-11 13:46:59,034 INFO train_multi TF=ALL: new best r_mae=0.9288 — saved rmae checkpoint
2026-05-11 13:47:11,984 INFO train_multi TF=ALL epoch 32/100 train=2.2262 val=2.2571 r_mae=0.926 pos_r_acc=0.589 side_acc=0.558 r_n=127469
2026-05-11 13:47:11,991 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:47:11,991 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:47:11,991 INFO train_multi TF=ALL: new best val=2.2571 r_mae=0.9262 — saved
2026-05-11 13:47:11,995 INFO train_multi TF=ALL: new best r_mae=0.9262 — saved rmae checkpoint
2026-05-11 13:47:24,856 INFO train_multi TF=ALL epoch 33/100 train=2.2193 val=2.2569 r_mae=0.927 pos_r_acc=0.588 side_acc=0.555 r_n=127469
2026-05-11 13:47:24,861 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:47:24,862 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:47:24,862 INFO train_multi TF=ALL: new best val=2.2569 r_mae=0.9268 — saved
2026-05-11 13:47:37,748 INFO train_multi TF=ALL epoch 34/100 train=2.2167 val=2.2515 r_mae=0.922 pos_r_acc=0.592 side_acc=0.562 r_n=127469
2026-05-11 13:47:37,753 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:47:37,753 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:47:37,754 INFO train_multi TF=ALL: new best val=2.2515 r_mae=0.9223 — saved
2026-05-11 13:47:37,758 INFO train_multi TF=ALL: new best r_mae=0.9223 — saved rmae checkpoint
2026-05-11 13:47:50,772 INFO train_multi TF=ALL epoch 35/100 train=2.2118 val=2.2504 r_mae=0.921 pos_r_acc=0.593 side_acc=0.565 r_n=127469
2026-05-11 13:47:50,777 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:47:50,778 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:47:50,778 INFO train_multi TF=ALL: new best val=2.2504 r_mae=0.9209 — saved
2026-05-11 13:47:50,782 INFO train_multi TF=ALL: new best r_mae=0.9209 — saved rmae checkpoint
2026-05-11 13:48:03,587 INFO train_multi TF=ALL epoch 36/100 train=2.2020 val=2.2523 r_mae=0.920 pos_r_acc=0.596 side_acc=0.561 r_n=127469
2026-05-11 13:48:03,592 INFO train_multi TF=ALL: new best r_mae=0.9200 — saved rmae checkpoint
2026-05-11 13:48:16,265 INFO train_multi TF=ALL epoch 37/100 train=2.1977 val=2.2576 r_mae=0.922 pos_r_acc=0.591 side_acc=0.563 r_n=127469
2026-05-11 13:48:29,008 INFO train_multi TF=ALL epoch 38/100 train=2.1927 val=2.2531 r_mae=0.917 pos_r_acc=0.593 side_acc=0.565 r_n=127469
2026-05-11 13:48:29,012 INFO train_multi TF=ALL: new best r_mae=0.9172 — saved rmae checkpoint
2026-05-11 13:48:42,043 INFO train_multi TF=ALL epoch 39/100 train=2.1870 val=2.2478 r_mae=0.921 pos_r_acc=0.594 side_acc=0.565 r_n=127469
2026-05-11 13:48:42,049 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:48:42,049 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:48:42,049 INFO train_multi TF=ALL: new best val=2.2478 r_mae=0.9210 — saved
2026-05-11 13:48:54,745 INFO train_multi TF=ALL epoch 40/100 train=2.1824 val=2.2522 r_mae=0.916 pos_r_acc=0.594 side_acc=0.567 r_n=127469
2026-05-11 13:48:54,749 INFO train_multi TF=ALL: new best r_mae=0.9162 — saved rmae checkpoint
2026-05-11 13:49:07,570 INFO train_multi TF=ALL epoch 41/100 train=2.1746 val=2.2611 r_mae=0.915 pos_r_acc=0.593 side_acc=0.564 r_n=127469
2026-05-11 13:49:07,574 INFO train_multi TF=ALL: new best r_mae=0.9151 — saved rmae checkpoint
2026-05-11 13:49:20,502 INFO train_multi TF=ALL epoch 42/100 train=2.1668 val=2.2523 r_mae=0.914 pos_r_acc=0.597 side_acc=0.569 r_n=127469
2026-05-11 13:49:20,507 INFO train_multi TF=ALL: new best r_mae=0.9135 — saved rmae checkpoint
2026-05-11 13:49:33,382 INFO train_multi TF=ALL epoch 43/100 train=2.1589 val=2.2456 r_mae=0.911 pos_r_acc=0.597 side_acc=0.577 r_n=127469
2026-05-11 13:49:33,387 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:49:33,387 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:49:33,387 INFO train_multi TF=ALL: new best val=2.2456 r_mae=0.9107 — saved
2026-05-11 13:49:33,392 INFO train_multi TF=ALL: new best r_mae=0.9107 — saved rmae checkpoint
2026-05-11 13:49:46,205 INFO train_multi TF=ALL epoch 44/100 train=2.1465 val=2.2411 r_mae=0.906 pos_r_acc=0.600 side_acc=0.577 r_n=127469
2026-05-11 13:49:46,210 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:49:46,210 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:49:46,210 INFO train_multi TF=ALL: new best val=2.2411 r_mae=0.9063 — saved
2026-05-11 13:49:46,214 INFO train_multi TF=ALL: new best r_mae=0.9063 — saved rmae checkpoint
2026-05-11 13:49:59,242 INFO train_multi TF=ALL epoch 45/100 train=2.1371 val=2.2307 r_mae=0.903 pos_r_acc=0.604 side_acc=0.582 r_n=127469
2026-05-11 13:49:59,247 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:49:59,247 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:49:59,247 INFO train_multi TF=ALL: new best val=2.2307 r_mae=0.9028 — saved
2026-05-11 13:49:59,251 INFO train_multi TF=ALL: new best r_mae=0.9028 — saved rmae checkpoint
2026-05-11 13:50:12,177 INFO train_multi TF=ALL epoch 46/100 train=2.1284 val=2.2309 r_mae=0.899 pos_r_acc=0.607 side_acc=0.582 r_n=127469
2026-05-11 13:50:12,182 INFO train_multi TF=ALL: new best r_mae=0.8992 — saved rmae checkpoint
2026-05-11 13:50:24,958 INFO train_multi TF=ALL epoch 47/100 train=2.1143 val=2.2306 r_mae=0.898 pos_r_acc=0.606 side_acc=0.585 r_n=127469
2026-05-11 13:50:24,963 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:50:24,963 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:50:24,963 INFO train_multi TF=ALL: new best val=2.2306 r_mae=0.8981 — saved
2026-05-11 13:50:24,968 INFO train_multi TF=ALL: new best r_mae=0.8981 — saved rmae checkpoint
2026-05-11 13:50:37,766 INFO train_multi TF=ALL epoch 48/100 train=2.1058 val=2.2184 r_mae=0.891 pos_r_acc=0.609 side_acc=0.591 r_n=127469
2026-05-11 13:50:37,771 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:50:37,771 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:50:37,771 INFO train_multi TF=ALL: new best val=2.2184 r_mae=0.8913 — saved
2026-05-11 13:50:37,775 INFO train_multi TF=ALL: new best r_mae=0.8913 — saved rmae checkpoint
2026-05-11 13:50:50,796 INFO train_multi TF=ALL epoch 49/100 train=2.0908 val=2.1967 r_mae=0.883 pos_r_acc=0.617 side_acc=0.603 r_n=127469
2026-05-11 13:50:50,801 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:50:50,801 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:50:50,801 INFO train_multi TF=ALL: new best val=2.1967 r_mae=0.8831 — saved
2026-05-11 13:50:50,805 INFO train_multi TF=ALL: new best r_mae=0.8831 — saved rmae checkpoint
2026-05-11 13:51:03,705 INFO train_multi TF=ALL epoch 50/100 train=2.0768 val=2.1939 r_mae=0.882 pos_r_acc=0.617 side_acc=0.605 r_n=127469
2026-05-11 13:51:03,711 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:51:03,711 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:51:03,711 INFO train_multi TF=ALL: new best val=2.1939 r_mae=0.8815 — saved
2026-05-11 13:51:03,715 INFO train_multi TF=ALL: new best r_mae=0.8815 — saved rmae checkpoint
2026-05-11 13:51:16,493 INFO train_multi TF=ALL epoch 51/100 train=2.0635 val=2.1735 r_mae=0.871 pos_r_acc=0.628 side_acc=0.606 r_n=127469
2026-05-11 13:51:16,498 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:51:16,498 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:51:16,498 INFO train_multi TF=ALL: new best val=2.1735 r_mae=0.8708 — saved
2026-05-11 13:51:16,502 INFO train_multi TF=ALL: new best r_mae=0.8708 — saved rmae checkpoint
2026-05-11 13:51:29,497 INFO train_multi TF=ALL epoch 52/100 train=2.0528 val=2.1745 r_mae=0.869 pos_r_acc=0.625 side_acc=0.608 r_n=127469
2026-05-11 13:51:29,501 INFO train_multi TF=ALL: new best r_mae=0.8693 — saved rmae checkpoint
2026-05-11 13:51:42,335 INFO train_multi TF=ALL epoch 53/100 train=2.0373 val=2.1687 r_mae=0.864 pos_r_acc=0.631 side_acc=0.607 r_n=127469
2026-05-11 13:51:42,340 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:51:42,340 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:51:42,340 INFO train_multi TF=ALL: new best val=2.1687 r_mae=0.8639 — saved
2026-05-11 13:51:42,344 INFO train_multi TF=ALL: new best r_mae=0.8639 — saved rmae checkpoint
2026-05-11 13:51:55,259 INFO train_multi TF=ALL epoch 54/100 train=2.0252 val=2.1634 r_mae=0.859 pos_r_acc=0.633 side_acc=0.611 r_n=127469
2026-05-11 13:51:55,264 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:51:55,264 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:51:55,264 INFO train_multi TF=ALL: new best val=2.1634 r_mae=0.8592 — saved
2026-05-11 13:51:55,268 INFO train_multi TF=ALL: new best r_mae=0.8592 — saved rmae checkpoint
2026-05-11 13:52:08,073 INFO train_multi TF=ALL epoch 55/100 train=2.0127 val=2.1678 r_mae=0.863 pos_r_acc=0.629 side_acc=0.611 r_n=127469
2026-05-11 13:52:20,963 INFO train_multi TF=ALL epoch 56/100 train=2.0049 val=2.1588 r_mae=0.851 pos_r_acc=0.636 side_acc=0.616 r_n=127469
2026-05-11 13:52:20,968 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:52:20,968 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:52:20,968 INFO train_multi TF=ALL: new best val=2.1588 r_mae=0.8513 — saved
2026-05-11 13:52:20,972 INFO train_multi TF=ALL: new best r_mae=0.8513 — saved rmae checkpoint
2026-05-11 13:52:33,930 INFO train_multi TF=ALL epoch 57/100 train=1.9945 val=2.1429 r_mae=0.847 pos_r_acc=0.641 side_acc=0.618 r_n=127469
2026-05-11 13:52:33,936 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:52:33,936 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:52:33,936 INFO train_multi TF=ALL: new best val=2.1429 r_mae=0.8468 — saved
2026-05-11 13:52:33,940 INFO train_multi TF=ALL: new best r_mae=0.8468 — saved rmae checkpoint
2026-05-11 13:52:47,008 INFO train_multi TF=ALL epoch 58/100 train=1.9806 val=2.1569 r_mae=0.838 pos_r_acc=0.643 side_acc=0.614 r_n=127469
2026-05-11 13:52:47,013 INFO train_multi TF=ALL: new best r_mae=0.8384 — saved rmae checkpoint
2026-05-11 13:52:59,890 INFO train_multi TF=ALL epoch 59/100 train=1.9722 val=2.1571 r_mae=0.838 pos_r_acc=0.644 side_acc=0.619 r_n=127469
2026-05-11 13:52:59,894 INFO train_multi TF=ALL: new best r_mae=0.8383 — saved rmae checkpoint
2026-05-11 13:53:13,141 INFO train_multi TF=ALL epoch 60/100 train=1.9612 val=2.1406 r_mae=0.839 pos_r_acc=0.646 side_acc=0.617 r_n=127469
2026-05-11 13:53:13,146 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 13:53:13,146 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 13:53:13,146 INFO train_multi TF=ALL: new best val=2.1406 r_mae=0.8393 — saved
2026-05-11 13:53:25,960 INFO train_multi TF=ALL epoch 61/100 train=1.9566 val=2.1429 r_mae=0.837 pos_r_acc=0.649 side_acc=0.619 r_n=127469
2026-05-11 13:53:25,964 INFO train_multi TF=ALL: new best r_mae=0.8369 — saved rmae checkpoint
2026-05-11 13:53:38,877 INFO train_multi TF=ALL epoch 62/100 train=1.9464 val=2.1596 r_mae=0.841 pos_r_acc=0.641 side_acc=0.614 r_n=127469
2026-05-11 13:53:51,841 INFO train_multi TF=ALL epoch 63/100 train=1.9375 val=2.1504 r_mae=0.828 pos_r_acc=0.649 side_acc=0.625 r_n=127469
2026-05-11 13:53:51,846 INFO train_multi TF=ALL: new best r_mae=0.8280 — saved rmae checkpoint
2026-05-11 13:54:04,619 INFO train_multi TF=ALL epoch 64/100 train=1.9291 val=2.1529 r_mae=0.832 pos_r_acc=0.647 side_acc=0.618 r_n=127469
2026-05-11 13:54:17,515 INFO train_multi TF=ALL epoch 65/100 train=1.9263 val=2.1532 r_mae=0.830 pos_r_acc=0.647 side_acc=0.622 r_n=127469
2026-05-11 13:54:30,396 INFO train_multi TF=ALL epoch 66/100 train=1.9134 val=2.1476 r_mae=0.826 pos_r_acc=0.652 side_acc=0.622 r_n=127469
2026-05-11 13:54:30,407 INFO train_multi TF=ALL: new best r_mae=0.8259 — saved rmae checkpoint
2026-05-11 13:54:43,341 INFO train_multi TF=ALL epoch 67/100 train=1.9054 val=2.1497 r_mae=0.829 pos_r_acc=0.649 side_acc=0.624 r_n=127469
2026-05-11 13:54:56,189 INFO train_multi TF=ALL epoch 68/100 train=1.8959 val=2.1539 r_mae=0.830 pos_r_acc=0.647 side_acc=0.625 r_n=127469
2026-05-11 13:55:09,168 INFO train_multi TF=ALL epoch 69/100 train=1.8917 val=2.1691 r_mae=0.830 pos_r_acc=0.647 side_acc=0.626 r_n=127469
2026-05-11 13:55:21,972 INFO train_multi TF=ALL epoch 70/100 train=1.8778 val=2.1647 r_mae=0.830 pos_r_acc=0.646 side_acc=0.624 r_n=127469
2026-05-11 13:55:35,034 INFO train_multi TF=ALL epoch 71/100 train=1.8727 val=2.1681 r_mae=0.830 pos_r_acc=0.645 side_acc=0.625 r_n=127469
2026-05-11 13:55:48,058 INFO train_multi TF=ALL epoch 72/100 train=1.8638 val=2.1630 r_mae=0.826 pos_r_acc=0.651 side_acc=0.624 r_n=127469
2026-05-11 13:55:48,063 INFO train_multi TF=ALL: new best r_mae=0.8256 — saved rmae checkpoint
2026-05-11 13:56:00,977 INFO train_multi TF=ALL epoch 73/100 train=1.8581 val=2.1653 r_mae=0.823 pos_r_acc=0.651 side_acc=0.625 r_n=127469
2026-05-11 13:56:00,981 INFO train_multi TF=ALL: new best r_mae=0.8228 — saved rmae checkpoint
2026-05-11 13:56:13,866 INFO train_multi TF=ALL epoch 74/100 train=1.8534 val=2.1772 r_mae=0.829 pos_r_acc=0.646 side_acc=0.628 r_n=127469
2026-05-11 13:56:26,916 INFO train_multi TF=ALL epoch 75/100 train=1.8484 val=2.1887 r_mae=0.830 pos_r_acc=0.646 side_acc=0.621 r_n=127469
2026-05-11 13:56:40,083 INFO train_multi TF=ALL epoch 76/100 train=1.8470 val=2.1734 r_mae=0.828 pos_r_acc=0.648 side_acc=0.625 r_n=127469
2026-05-11 13:56:53,033 INFO train_multi TF=ALL epoch 77/100 train=1.8340 val=2.1912 r_mae=0.827 pos_r_acc=0.648 side_acc=0.627 r_n=127469
2026-05-11 13:57:06,126 INFO train_multi TF=ALL epoch 78/100 train=1.8261 val=2.1948 r_mae=0.825 pos_r_acc=0.649 side_acc=0.627 r_n=127469
2026-05-11 13:57:19,305 INFO train_multi TF=ALL epoch 79/100 train=1.8196 val=2.1904 r_mae=0.828 pos_r_acc=0.646 side_acc=0.623 r_n=127469
2026-05-11 13:57:32,179 INFO train_multi TF=ALL epoch 80/100 train=1.8101 val=2.1769 r_mae=0.826 pos_r_acc=0.648 side_acc=0.630 r_n=127469
2026-05-11 13:57:44,953 INFO train_multi TF=ALL epoch 81/100 train=1.8054 val=2.1909 r_mae=0.825 pos_r_acc=0.649 side_acc=0.629 r_n=127469
2026-05-11 13:57:57,880 INFO train_multi TF=ALL epoch 82/100 train=1.7953 val=2.2209 r_mae=0.827 pos_r_acc=0.648 side_acc=0.622 r_n=127469
2026-05-11 13:58:10,817 INFO train_multi TF=ALL epoch 83/100 train=1.7907 val=2.2148 r_mae=0.830 pos_r_acc=0.646 side_acc=0.627 r_n=127469
2026-05-11 13:58:23,807 INFO train_multi TF=ALL epoch 84/100 train=1.7821 val=2.1993 r_mae=0.830 pos_r_acc=0.645 side_acc=0.628 r_n=127469
2026-05-11 13:58:36,873 INFO train_multi TF=ALL epoch 85/100 train=1.7787 val=2.2347 r_mae=0.831 pos_r_acc=0.644 side_acc=0.620 r_n=127469
2026-05-11 13:58:49,817 INFO train_multi TF=ALL epoch 86/100 train=1.7773 val=2.2138 r_mae=0.828 pos_r_acc=0.647 side_acc=0.629 r_n=127469
2026-05-11 13:59:02,826 INFO train_multi TF=ALL epoch 87/100 train=1.7650 val=2.2179 r_mae=0.832 pos_r_acc=0.645 side_acc=0.625 r_n=127469
2026-05-11 13:59:15,801 INFO train_multi TF=ALL epoch 88/100 train=1.7603 val=2.2234 r_mae=0.834 pos_r_acc=0.643 side_acc=0.622 r_n=127469
2026-05-11 13:59:28,908 INFO train_multi TF=ALL epoch 89/100 train=1.7553 val=2.2368 r_mae=0.828 pos_r_acc=0.646 side_acc=0.623 r_n=127469
2026-05-11 13:59:41,788 INFO train_multi TF=ALL epoch 90/100 train=1.7488 val=2.2373 r_mae=0.829 pos_r_acc=0.645 side_acc=0.628 r_n=127469
2026-05-11 13:59:54,720 INFO train_multi TF=ALL epoch 91/100 train=1.7423 val=2.2606 r_mae=0.839 pos_r_acc=0.639 side_acc=0.624 r_n=127469
2026-05-11 14:00:07,810 INFO train_multi TF=ALL epoch 92/100 train=1.7377 val=2.2401 r_mae=0.830 pos_r_acc=0.643 side_acc=0.627 r_n=127469
2026-05-11 14:00:20,725 INFO train_multi TF=ALL epoch 93/100 train=1.7271 val=2.2257 r_mae=0.830 pos_r_acc=0.645 side_acc=0.626 r_n=127469
2026-05-11 14:00:33,583 INFO train_multi TF=ALL epoch 94/100 train=1.7218 val=2.2336 r_mae=0.829 pos_r_acc=0.645 side_acc=0.633 r_n=127469
2026-05-11 14:00:46,604 INFO train_multi TF=ALL epoch 95/100 train=1.7253 val=2.2372 r_mae=0.828 pos_r_acc=0.644 side_acc=0.626 r_n=127469
2026-05-11 14:00:59,436 INFO train_multi TF=ALL epoch 96/100 train=1.7113 val=2.2565 r_mae=0.831 pos_r_acc=0.644 side_acc=0.630 r_n=127469
2026-05-11 14:01:12,344 INFO train_multi TF=ALL epoch 97/100 train=1.7068 val=2.2509 r_mae=0.833 pos_r_acc=0.641 side_acc=0.629 r_n=127469
2026-05-11 14:01:25,550 INFO train_multi TF=ALL epoch 98/100 train=1.7016 val=2.2505 r_mae=0.829 pos_r_acc=0.645 side_acc=0.628 r_n=127469
2026-05-11 14:01:25,550 INFO train_multi TF=ALL early stop at epoch 98
2026-05-11 14:01:25,562 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:01:25,562 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:01:25,562 INFO train_multi TF=ALL: promoted r_mae checkpoint (0.8228 < primary 0.8393) — overwriting model.pt
2026-05-11 14:01:26,826 WARNING fit_r_isotonic: long calibrator increases MAE (cal=0.8428 >= raw=0.8280) — skipping
2026-05-11 14:01:26,835 WARNING fit_r_isotonic: short calibrator increases MAE (cal=0.8517 >= raw=0.8322) — skipping
2026-05-11 14:01:26,835 INFO fit_r_isotonic: saved side-R calibrators stats={'long': {'n': 31689, 'raw_mae': 0.8280427353715868, 'calibrated_mae': 0.8427682306913493, 'skipped': 'calibrator_hurts'}, 'short': {'n': 32408, 'raw_mae': 0.8322181784818156, 'calibrated_mae': 0.851712378685533, 'skipped': 'calibrator_hurts'}}
2026-05-11 14:01:26,963 WARNING GRU validation R-MAE above floor for TF=ALL: best_val_r_mae=0.823 max=0.750. Keeping saved best weights so the pipeline can progress.
2026-05-11 14:01:26,976 INFO GRU R threshold XAUUSD/buy: q25_pos=0.535 q50_pos=1.667 pos_rate=46.0% (n=119259 n_pos=54899)
2026-05-11 14:01:26,982 INFO GRU R threshold XAUUSD/sell: q25_pos=0.528 q50_pos=1.667 pos_rate=45.5% (n=118087 n_pos=53734)
2026-05-11 14:01:26,987 INFO GRU R threshold EURUSD/buy: q25_pos=0.541 q50_pos=1.667 pos_rate=45.6% (n=118751 n_pos=54196)
2026-05-11 14:01:26,993 INFO GRU R threshold EURUSD/sell: q25_pos=0.545 q50_pos=1.667 pos_rate=45.6% (n=117679 n_pos=53721)
2026-05-11 14:01:26,998 INFO GRU R threshold USDJPY/buy: q25_pos=0.551 q50_pos=1.667 pos_rate=46.2% (n=118691 n_pos=54870)
2026-05-11 14:01:27,005 INFO GRU R threshold USDJPY/sell: q25_pos=0.542 q50_pos=1.667 pos_rate=44.6% (n=116030 n_pos=51793)
2026-05-11 14:01:27,011 INFO GRU R threshold EURJPY/buy: q25_pos=0.530 q50_pos=1.416 pos_rate=46.1% (n=118651 n_pos=54654)
2026-05-11 14:01:27,016 INFO GRU R threshold EURJPY/sell: q25_pos=0.525 q50_pos=1.667 pos_rate=44.7% (n=117320 n_pos=52399)
2026-05-11 14:01:27,022 INFO GRU R threshold GBPJPY/buy: q25_pos=0.517 q50_pos=1.443 pos_rate=45.9% (n=118277 n_pos=54278)
2026-05-11 14:01:27,027 INFO GRU R threshold GBPJPY/sell: q25_pos=0.521 q50_pos=1.667 pos_rate=45.3% (n=116299 n_pos=52702)
2026-05-11 14:01:27,033 INFO GRU R threshold GBPUSD/buy: q25_pos=0.524 q50_pos=1.667 pos_rate=45.8% (n=117764 n_pos=53894)
2026-05-11 14:01:27,039 INFO GRU R threshold GBPUSD/sell: q25_pos=0.533 q50_pos=1.667 pos_rate=45.8% (n=117404 n_pos=53785)
2026-05-11 14:01:27,039 INFO GRU per-symbol R thresholds saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/symbol_r_thresholds.json (6 symbols)
2026-05-11 14:01:27,068 INFO Retrain complete. Total wall-clock: 1299.3s
2026-05-11 14:01:31,689 INFO Model gru: SUCCESS
2026-05-11 14:01:31,689 INFO --- Training regime ---
2026-05-11 14:01:31,689 INFO Running retrain --model regime
2026-05-11 14:01:32,171 INFO retrain environment: KAGGLE
2026-05-11 14:01:33,803 INFO Device: CUDA (2 GPU(s))
2026-05-11 14:01:33,814 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 14:01:33,814 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 14:01:33,814 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 14:01:33,817 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 14:01:33,817 INFO Retrain data split: train
2026-05-11 14:01:33,817 INFO Retrain rolling fold selector: latest
2026-05-11 14:01:33,818 INFO === RegimeClassifier retrain (hierarchical: HTF 2-score bias + LTF 5-score behaviour) ===
2026-05-11 14:01:34,056 INFO NumExpr defaulting to 4 threads.
2026-05-11 14:01:34,263 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-11 14:01:34,263 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 14:01:34,263 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 14:01:34,263 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-11 14:01:34,314 INFO Regime rolling folds selected: [None]
2026-05-11 14:01:34,314 INFO === Regime rolling fold 1/1: train_all ===
2026-05-11 14:01:34,314 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-11 14:01:34,353 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 14:01:34,354 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:01:34,371 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:01:34,388 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:01:34,405 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:01:34,423 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:01:34,460 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:01:34,726 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:01:34,794 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:01:34,819 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:01:34,820 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:01:34,830 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:01:34,831 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:01:35,573 INFO macro_cache: using training data dir /kaggle/working/Multi-Bot/trading-system/training_data
2026-05-11 14:01:35,690 INFO macro_cache loaded 19 series: {'asx200': '2842 bars 2015-01-02→2026-03-27', 'cac40': '2876 bars 2015-01-02→2026-03-27', 'dax': '2851 bars 2015-01-02→2026-03-27', 'djia': '2825 bars 2015-01-02→2026-03-27', 'dxy': '2826 bars 2015-01-02→2026-03-27', 'eurostoxx': '2823 bars 2015-01-05→2026-03-27', 'ftse': '2839 bars 2015-01-02→2026-03-27', 'gold_fut': '2824 bars 2015-01-02→2026-03-27', 'hsi': '2764 bars 2015-01-02→2026-03-27', 'nasdaq': '2825 bars 2015-01-02→2026-03-27', 'nikkei': '2744 bars 2015-01-05→2026-03-27', 'oil_fut': '2825 bars 2015-01-02→2026-03-27', 'spx': '2825 bars 2015-01-02→2026-03-27', 'us10y': '2824 bars 2015-01-02→2026-03-27', 'us30y': '2824 bars 2015-01-02→2026-03-27', 'us3m': '2824 bars 2015-01-02→2026-03-27', 'vix': '2825 bars 2015-01-02→2026-03-27', 'us10y_fred': '2607 bars 2016-03-28→2026-03-24', 'us2y_fred': '2607 bars 2016-03-28→2026-03-24'}
2026-05-11 14:01:35,928 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1142, 'BIAS_DOWN': 522, 'BIAS_NEUTRAL': 10438}  ambiguous=4182 (total=12102) horizon=84
2026-05-11 14:01:35,933 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected XAUUSD — 12052 samples (group=gold) score_means={'bias_up_score': 0.0948, 'bias_down_score': 0.0433} labels={'BIAS_UP': 1142, 'BIAS_DOWN': 522, 'BIAS_NEUTRAL': 10388} clean={'BIAS_UP': 1142, 'BIAS_DOWN': 522, 'BIAS_NEUTRAL': 6216}
2026-05-11 14:01:36,098 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:36,135 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:36,153 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:36,153 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:36,161 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:36,162 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:01:37,069 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 690, 'BIAS_DOWN': 540, 'BIAS_NEUTRAL': 10174}  ambiguous=3886 (total=11404) horizon=84
2026-05-11 14:01:37,074 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURUSD — 11354 samples (group=dollar) score_means={'bias_up_score': 0.0608, 'bias_down_score': 0.0476} labels={'BIAS_UP': 690, 'BIAS_DOWN': 540, 'BIAS_NEUTRAL': 10124} clean={'BIAS_UP': 690, 'BIAS_DOWN': 540, 'BIAS_NEUTRAL': 6257}
2026-05-11 14:01:37,232 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:37,266 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:37,286 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:37,286 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:37,294 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:37,296 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:01:38,236 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 826, 'BIAS_DOWN': 423, 'BIAS_NEUTRAL': 10154}  ambiguous=4036 (total=11403) horizon=84
2026-05-11 14:01:38,241 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected USDJPY — 11353 samples (group=dollar) score_means={'bias_up_score': 0.0728, 'bias_down_score': 0.0373} labels={'BIAS_UP': 826, 'BIAS_DOWN': 423, 'BIAS_NEUTRAL': 10104} clean={'BIAS_UP': 826, 'BIAS_DOWN': 423, 'BIAS_NEUTRAL': 6078}
2026-05-11 14:01:38,397 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:38,440 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:38,466 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:38,466 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:38,474 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:38,475 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:01:39,383 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 681, 'BIAS_DOWN': 527, 'BIAS_NEUTRAL': 10199}  ambiguous=4044 (total=11407) horizon=84
2026-05-11 14:01:39,388 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURJPY — 11357 samples (group=cross) score_means={'bias_up_score': 0.06, 'bias_down_score': 0.0464} labels={'BIAS_UP': 681, 'BIAS_DOWN': 527, 'BIAS_NEUTRAL': 10149} clean={'BIAS_UP': 681, 'BIAS_DOWN': 527, 'BIAS_NEUTRAL': 6111}
2026-05-11 14:01:39,541 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:39,576 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:39,596 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:39,597 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:39,605 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:39,606 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:01:40,503 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 839, 'BIAS_DOWN': 579, 'BIAS_NEUTRAL': 9990}  ambiguous=4240 (total=11408) horizon=84
2026-05-11 14:01:40,507 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPJPY — 11358 samples (group=cross) score_means={'bias_up_score': 0.0739, 'bias_down_score': 0.051} labels={'BIAS_UP': 839, 'BIAS_DOWN': 579, 'BIAS_NEUTRAL': 9940} clean={'BIAS_UP': 839, 'BIAS_DOWN': 579, 'BIAS_NEUTRAL': 5723}
2026-05-11 14:01:40,655 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:40,687 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:40,707 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:40,707 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:40,715 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:40,716 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:01:41,612 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 502, 'BIAS_DOWN': 720, 'BIAS_NEUTRAL': 10180}  ambiguous=4113 (total=11402) horizon=84
2026-05-11 14:01:41,618 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPUSD — 11352 samples (group=dollar) score_means={'bias_up_score': 0.0442, 'bias_down_score': 0.0623} labels={'BIAS_UP': 502, 'BIAS_DOWN': 707, 'BIAS_NEUTRAL': 10143} clean={'BIAS_UP': 502, 'BIAS_DOWN': 707, 'BIAS_NEUTRAL': 6056}
2026-05-11 14:01:41,692 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 1520, 'BIAS_DOWN': 1106, 'BIAS_NEUTRAL': 20089}, 'dollar': {'BIAS_UP': 2018, 'BIAS_DOWN': 1670, 'BIAS_NEUTRAL': 30371}, 'gold': {'BIAS_UP': 1142, 'BIAS_DOWN': 522, 'BIAS_NEUTRAL': 10388}}
2026-05-11 14:01:41,692 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0669, 'bias_down_score': 0.0487}, 'dollar': {'bias_up_score': 0.0593, 'bias_down_score': 0.049}, 'gold': {'bias_up_score': 0.0948, 'bias_down_score': 0.0433}}
2026-05-11 14:01:41,692 INFO Regime[4H mode=htf_bias] label distribution by year: {2016: {'BIAS_UP': 491, 'BIAS_DOWN': 576, 'BIAS_NEUTRAL': 7755}, 2017: {'BIAS_UP': 734, 'BIAS_DOWN': 286, 'BIAS_NEUTRAL': 8093}, 2018: {'BIAS_UP': 427, 'BIAS_DOWN': 714, 'BIAS_NEUTRAL': 7989}, 2019: {'BIAS_UP': 410, 'BIAS_DOWN': 447, 'BIAS_NEUTRAL': 8245}, 2020: {'BIAS_UP': 694, 'BIAS_DOWN': 174, 'BIAS_NEUTRAL': 8243}, 2021: {'BIAS_UP': 722, 'BIAS_DOWN': 473, 'BIAS_NEUTRAL': 7896}, 2022: {'BIAS_UP': 667, 'BIAS_DOWN': 519, 'BIAS_NEUTRAL': 7935}, 2023: {'BIAS_UP': 535, 'BIAS_DOWN': 109, 'BIAS_NEUTRAL': 4692}}
2026-05-11 14:01:41,693 INFO Regime[4H mode=htf_bias] score means by year: {2016: {'bias_up_score': 0.0557, 'bias_down_score': 0.0653}, 2017: {'bias_up_score': 0.0805, 'bias_down_score': 0.0314}, 2018: {'bias_up_score': 0.0468, 'bias_down_score': 0.0782}, 2019: {'bias_up_score': 0.045, 'bias_down_score': 0.0491}, 2020: {'bias_up_score': 0.0762, 'bias_down_score': 0.0191}, 2021: {'bias_up_score': 0.0794, 'bias_down_score': 0.052}, 2022: {'bias_up_score': 0.0731, 'bias_down_score': 0.0569}, 2023: {'bias_up_score': 0.1003, 'bias_down_score': 0.0204}}
2026-05-11 14:01:41,742 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
2026-05-11 14:01:41,743 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
2026-05-11 14:01:41,744 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
2026-05-11 14:01:41,745 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 14:01:41,746 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 14:01:41,747 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
2026-05-11 14:01:41,757 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:01:41,761 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:01:41,762 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:01:41,762 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:01:41,763 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:01:41,764 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:01:42,314 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 157, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1339}  ambiguous=566 (total=1581) horizon=84
2026-05-11 14:01:42,316 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected XAUUSD — 1531 samples (group=gold) score_means={'bias_up_score': 0.1025, 'bias_down_score': 0.0555} labels={'BIAS_UP': 157, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1289} clean={'BIAS_UP': 157, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 744}
2026-05-11 14:01:42,385 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:42,388 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:42,388 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:42,389 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:42,389 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:42,390 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:01:42,895 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 135, 'BIAS_DOWN': 66, 'BIAS_NEUTRAL': 1290}  ambiguous=531 (total=1491) horizon=84
2026-05-11 14:01:42,898 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURUSD — 1441 samples (group=dollar) score_means={'bias_up_score': 0.0937, 'bias_down_score': 0.0458} labels={'BIAS_UP': 135, 'BIAS_DOWN': 66, 'BIAS_NEUTRAL': 1240} clean={'BIAS_UP': 135, 'BIAS_DOWN': 66, 'BIAS_NEUTRAL': 739}
2026-05-11 14:01:42,963 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:42,965 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:42,966 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:42,966 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:42,966 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:42,968 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:01:43,476 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 164, 'BIAS_DOWN': 77, 'BIAS_NEUTRAL': 1248}  ambiguous=616 (total=1489) horizon=84
2026-05-11 14:01:43,479 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected USDJPY — 1439 samples (group=dollar) score_means={'bias_up_score': 0.114, 'bias_down_score': 0.0535} labels={'BIAS_UP': 164, 'BIAS_DOWN': 77, 'BIAS_NEUTRAL': 1198} clean={'BIAS_UP': 164, 'BIAS_DOWN': 77, 'BIAS_NEUTRAL': 608}
2026-05-11 14:01:43,548 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:43,550 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:43,551 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:43,551 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:43,552 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:43,553 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:01:44,071 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 123, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 1366}  ambiguous=582 (total=1494) horizon=84
2026-05-11 14:01:44,074 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0852, 'bias_down_score': 0.0035} labels={'BIAS_UP': 123, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 1316} clean={'BIAS_UP': 123, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 741}
2026-05-11 14:01:44,140 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:44,142 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:44,143 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:44,143 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:44,144 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:44,144 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:01:44,649 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 129, 'BIAS_DOWN': 9, 'BIAS_NEUTRAL': 1356}  ambiguous=551 (total=1494) horizon=84
2026-05-11 14:01:44,652 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0893, 'bias_down_score': 0.0055} labels={'BIAS_UP': 129, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 1307} clean={'BIAS_UP': 129, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 775}
2026-05-11 14:01:44,718 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:44,721 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:44,721 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:44,722 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:44,722 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:01:44,723 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:01:45,240 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 81, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 1316}  ambiguous=560 (total=1488) horizon=84
2026-05-11 14:01:45,242 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPUSD — 1438 samples (group=dollar) score_means={'bias_up_score': 0.0563, 'bias_down_score': 0.0633} labels={'BIAS_UP': 81, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 1266} clean={'BIAS_UP': 81, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 735}
2026-05-11 14:01:45,302 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 252, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 2623}, 'dollar': {'BIAS_UP': 380, 'BIAS_DOWN': 234, 'BIAS_NEUTRAL': 3704}, 'gold': {'BIAS_UP': 157, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1289}}
2026-05-11 14:01:45,303 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0873, 'bias_down_score': 0.0045}, 'dollar': {'bias_up_score': 0.088, 'bias_down_score': 0.0542}, 'gold': {'bias_up_score': 0.1025, 'bias_down_score': 0.0555}}
2026-05-11 14:01:45,303 INFO Regime[4H mode=htf_bias] label distribution by year: {2022: {'BIAS_UP': 258, 'BIAS_DOWN': 228, 'BIAS_NEUTRAL': 2915}, 2023: {'BIAS_UP': 531, 'BIAS_DOWN': 104, 'BIAS_NEUTRAL': 4701}}
2026-05-11 14:01:45,303 INFO Regime[4H mode=htf_bias] score means by year: {2022: {'bias_up_score': 0.0759, 'bias_down_score': 0.067}, 2023: {'bias_up_score': 0.0995, 'bias_down_score': 0.0195}}
2026-05-11 14:01:45,345 INFO Regime phase HTF dataset build fold=train_all: 11.0s (train=68826 val=8737)
2026-05-11 14:01:45,346 INFO Regime 4H/htf_bias cold start: no existing weights found
2026-05-11 14:01:45,359 INFO RegimeClassifier[mode=htf_bias]: HTF clean-label fit filter kept train=44419/68826 val=5463/8737 at conf>=0.40 train_counts={'BIAS_UP': 4680, 'BIAS_DOWN': 3298, 'BIAS_NEUTRAL': 36441} val_counts={'BIAS_UP': 789, 'BIAS_DOWN': 332, 'BIAS_NEUTRAL': 4342}
2026-05-11 14:01:45,359 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=44419 val=5463 train_labels={'BIAS_UP': 4680, 'BIAS_DOWN': 3298, 'BIAS_NEUTRAL': 36441} val_labels={'BIAS_UP': 789, 'BIAS_DOWN': 332, 'BIAS_NEUTRAL': 4342}
2026-05-11 14:01:45,560 INFO RegimeClassifier[mode=htf_bias]: cold start HTF score head
2026-05-11 14:01:45,560 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-11 14:01:45,560 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 8.491, 'bias_down_score': 12.0}
2026-05-11 14:01:45,564 INFO RegimeClassifier[mode=htf_bias]: HTF balanced sampler — dir=7978 neutral=36441 dir_weight=3 => dir_frac_per_epoch≈47.2%
2026-05-11 14:01:49,003 INFO Regime HTF score epoch  1/50 — tr=6.6117 va=2.3226 acc=0.795 bal=0.333 threshold=0.60 margin=0.10 recall={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 1.0} precision={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 0.795}
2026-05-11 14:01:50,307 INFO Regime HTF score epoch  2/50 — tr=6.5455 va=2.2902 bal=0.333
2026-05-11 14:01:51,655 INFO Regime HTF score epoch  3/50 — tr=6.1826 va=2.1701 bal=0.333
2026-05-11 14:01:53,017 INFO Regime HTF score epoch  4/50 — tr=5.6327 va=1.9462 bal=0.333
2026-05-11 14:01:54,329 INFO Regime HTF score epoch  5/50 — tr=4.8611 va=1.6112 acc=0.801 bal=0.355 threshold=0.35 margin=0.40 recall={'BIAS_UP': 0.035, 'BIAS_DOWN': 0.03, 'BIAS_NEUTRAL': 0.999} precision={'BIAS_UP': 0.933, 'BIAS_DOWN': 0.833, 'BIAS_NEUTRAL': 0.8}
2026-05-11 14:01:55,635 INFO Regime HTF score epoch  6/50 — tr=3.9811 va=1.2348 bal=0.413
2026-05-11 14:01:56,950 INFO Regime HTF score epoch  7/50 — tr=3.2988 va=0.9890 bal=0.417
2026-05-11 14:01:58,247 INFO Regime HTF score epoch  8/50 — tr=2.7422 va=0.8553 bal=0.365
2026-05-11 14:01:59,583 INFO Regime HTF score epoch  9/50 — tr=2.3059 va=0.7540 bal=0.368
2026-05-11 14:02:00,883 INFO Regime HTF score epoch 10/50 — tr=1.9672 va=0.6856 acc=0.801 bal=0.364 threshold=0.95 margin=0.15 recall={'BIAS_UP': 0.044, 'BIAS_DOWN': 0.051, 'BIAS_NEUTRAL': 0.996} precision={'BIAS_UP': 0.897, 'BIAS_DOWN': 0.567, 'BIAS_NEUTRAL': 0.802}
2026-05-11 14:02:02,177 INFO Regime HTF score epoch 11/50 — tr=1.7313 va=0.6415 bal=0.378
2026-05-11 14:02:03,488 INFO Regime HTF score epoch 12/50 — tr=1.5558 va=0.6089 bal=0.350
2026-05-11 14:02:04,795 INFO Regime HTF score epoch 13/50 — tr=1.4622 va=0.5939 bal=0.362
2026-05-11 14:02:04,795 INFO Regime HTF score early stop at epoch 13
2026-05-11 14:02:06,005 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.350 margin=0.400 precision={'BIAS_UP': 0.933, 'BIAS_DOWN': 0.833, 'BIAS_NEUTRAL': 0.8} recall={'BIAS_UP': 0.035, 'BIAS_DOWN': 0.03, 'BIAS_NEUTRAL': 0.999} f1={'BIAS_UP': 0.068, 'BIAS_DOWN': 0.058, 'BIAS_NEUTRAL': 0.889} confusion=[[28, 0, 761], [0, 10, 322], [2, 2, 4338]] score_mae={'bias_up_score': 0.2084, 'bias_down_score': 0.1194} pred_share={'BIAS_UP': 0.0055, 'BIAS_DOWN': 0.0022, 'BIAS_NEUTRAL': 0.9923}
2026-05-11 14:02:06,006 WARNING Regime HTF score prediction distribution collapsed: pred_share={'BIAS_UP': 0.0055, 'BIAS_DOWN': 0.0022, 'BIAS_NEUTRAL': 0.9923}, max_pred_share=99.2%, collapsed_classes=[]. Saving weights anyway so the pipeline can progress.
2026-05-11 14:02:06,006 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.933, 'BIAS_DOWN': 0.833, 'BIAS_NEUTRAL': 0.8} min_precision=0.500 recall={'BIAS_UP': 0.035, 'BIAS_DOWN': 0.03, 'BIAS_NEUTRAL': 0.999} min_recall=0.100 f1={'BIAS_UP': 0.068, 'BIAS_DOWN': 0.058, 'BIAS_NEUTRAL': 0.889} min_f1=0.150 min_neutral_recall=0.500 weak_precision=[] weak_recall=['BIAS_UP', 'BIAS_DOWN'] weak_f1=['BIAS_UP', 'BIAS_DOWN'] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-11 14:02:06,009 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 14:02:06,009 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 14:02:06,010 INFO Regime phase HTF train fold=train_all: 20.7s
2026-05-11 14:02:06,108 INFO Regime HTF complete fold=train_all: acc=0.801 bal=0.355 train=68826 val=8737 per_class={'BIAS_UP': 0.035, 'BIAS_DOWN': 0.03, 'BIAS_NEUTRAL': 0.999} precision={'BIAS_UP': 0.933, 'BIAS_DOWN': 0.833, 'BIAS_NEUTRAL': 0.8} threshold=0.350 margin=0.400
2026-05-11 14:02:06,110 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:02:06,294 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 502, 'BIAS_DOWN': 720, 'BIAS_NEUTRAL': 10180}  ambiguous=4113 (total=11402) horizon=84
2026-05-11 14:02:06,300 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on GBPUSD 4H:
{'BIAS_UP': 4.482142857142857, 'BIAS_DOWN': 5.669291338582677, 'BIAS_NEUTRAL': 42.416666666666664}
2026-05-11 14:02:06,303 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (all labels):
{'BIAS_UP': {'n': 502, 'mean': 0.00043281055222645396, 'mean_over_std': 0.19321420748230153}, 'BIAS_DOWN': {'n': 720, 'mean': -0.00046308823082979826, 'mean_over_std': -0.19538165843530478}, 'BIAS_NEUTRAL': {'n': 10179, 'mean': 7.477567618138561e-07, 'mean_over_std': 0.0002829536380249001}}
2026-05-11 14:02:06,304 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 502, 'mean': 0.00043281055222645396, 'mean_over_std': 0.19321420748230153}, 'BIAS_DOWN': {'n': 720, 'mean': -0.00046308823082979826, 'mean_over_std': -0.19538165843530478}, 'BIAS_NEUTRAL': {'n': 6067, 'mean': 9.596616495197703e-06, 'mean_over_std': 0.004013656697571348}}
2026-05-11 14:02:06,307 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-11 14:02:06,310 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:02:06,311 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:02:06,313 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:02:06,315 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:02:06,317 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:02:06,319 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:02:06,334 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:02:06,342 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:02:06,345 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:02:06,345 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:02:06,345 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:02:06,349 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:02:07,362 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected XAUUSD — 44227 samples (group=gold) score_means={'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}
2026-05-11 14:02:07,463 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:07,465 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:07,466 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:07,467 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:07,467 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:07,469 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:02:08,418 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURUSD — 43689 samples (group=dollar) score_means={'trend_score': 0.49, 'range_score': 0.236, 'chop_score': 0.4622, 'volatility_percentile': 0.3812, 'consolidation_score': 0.1862}
2026-05-11 14:02:08,528 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:08,530 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:08,531 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:08,531 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:08,531 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:08,534 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:02:09,470 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected USDJPY — 43686 samples (group=dollar) score_means={'trend_score': 0.4949, 'range_score': 0.2312, 'chop_score': 0.4613, 'volatility_percentile': 0.38, 'consolidation_score': 0.1931}
2026-05-11 14:02:09,571 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:09,573 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:09,574 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:09,574 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:09,574 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:09,576 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:02:10,519 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURJPY — 43688 samples (group=cross) score_means={'trend_score': 0.4888, 'range_score': 0.2345, 'chop_score': 0.4662, 'volatility_percentile': 0.3831, 'consolidation_score': 0.1882}
2026-05-11 14:02:10,621 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:10,624 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:10,625 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:10,625 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:10,625 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:10,627 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:02:11,553 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPJPY — 43678 samples (group=cross) score_means={'trend_score': 0.4878, 'range_score': 0.2348, 'chop_score': 0.4673, 'volatility_percentile': 0.3839, 'consolidation_score': 0.1904}
2026-05-11 14:02:11,654 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:11,657 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:11,657 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:11,658 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:11,658 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:11,661 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:02:12,605 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPUSD — 43676 samples (group=dollar) score_means={'trend_score': 0.4923, 'range_score': 0.2335, 'chop_score': 0.4611, 'volatility_percentile': 0.3832, 'consolidation_score': 0.1856}
2026-05-11 14:02:12,721 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2346, 'chop_score': 0.4667, 'volatility_percentile': 0.3835, 'consolidation_score': 0.1893}, 'dollar': {'trend_score': 0.4924, 'range_score': 0.2335, 'chop_score': 0.4615, 'volatility_percentile': 0.3815, 'consolidation_score': 0.1883}, 'gold': {'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}}
2026-05-11 14:02:12,722 INFO Regime[1H mode=ltf_behaviour] score means by year: {2016: {'trend_score': 0.495, 'range_score': 0.233, 'chop_score': 0.4607, 'volatility_percentile': 0.3776, 'consolidation_score': 0.192}, 2017: {'trend_score': 0.5017, 'range_score': 0.2328, 'chop_score': 0.4574, 'volatility_percentile': 0.3689, 'consolidation_score': 0.1942}, 2018: {'trend_score': 0.4886, 'range_score': 0.237, 'chop_score': 0.4655, 'volatility_percentile': 0.3947, 'consolidation_score': 0.1785}, 2019: {'trend_score': 0.4817, 'range_score': 0.236, 'chop_score': 0.4691, 'volatility_percentile': 0.3725, 'consolidation_score': 0.198}, 2020: {'trend_score': 0.4874, 'range_score': 0.2311, 'chop_score': 0.4674, 'volatility_percentile': 0.3848, 'consolidation_score': 0.1919}, 2021: {'trend_score': 0.4872, 'range_score': 0.2372, 'chop_score': 0.4672, 'volatility_percentile': 0.377, 'consolidation_score': 0.192}, 2022: {'trend_score': 0.4893, 'range_score': 0.2357, 'chop_score': 0.4647, 'volatility_percentile': 0.3962, 'consolidation_score': 0.1784}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 14:02:12,810 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
2026-05-11 14:02:12,812 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 14:02:12,813 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
2026-05-11 14:02:12,814 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 14:02:12,816 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
2026-05-11 14:02:12,817 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
2026-05-11 14:02:12,833 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:02:12,837 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:02:12,838 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:02:12,838 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:02:12,839 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:02:12,840 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:02:13,148 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected XAUUSD — 5117 samples (group=gold) score_means={'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}
2026-05-11 14:02:13,255 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:13,260 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:13,261 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:13,261 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:13,261 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:13,263 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:02:13,560 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURUSD — 5053 samples (group=dollar) score_means={'trend_score': 0.4883, 'range_score': 0.2413, 'chop_score': 0.463, 'volatility_percentile': 0.3837, 'consolidation_score': 0.1793}
2026-05-11 14:02:13,662 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:13,664 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:13,665 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:13,665 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:13,666 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:13,667 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:02:13,958 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected USDJPY — 5046 samples (group=dollar) score_means={'trend_score': 0.5085, 'range_score': 0.2259, 'chop_score': 0.4506, 'volatility_percentile': 0.3809, 'consolidation_score': 0.1888}
2026-05-11 14:02:14,060 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:14,063 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:14,064 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:14,064 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:14,064 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:14,066 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:02:14,359 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURJPY — 5053 samples (group=cross) score_means={'trend_score': 0.5023, 'range_score': 0.2261, 'chop_score': 0.4524, 'volatility_percentile': 0.3837, 'consolidation_score': 0.18}
2026-05-11 14:02:14,461 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:14,463 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:14,464 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:14,465 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:14,465 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:14,466 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:02:14,751 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPJPY — 5043 samples (group=cross) score_means={'trend_score': 0.4743, 'range_score': 0.2372, 'chop_score': 0.4716, 'volatility_percentile': 0.3777, 'consolidation_score': 0.1917}
2026-05-11 14:02:14,852 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:14,855 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:14,855 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:14,856 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:14,856 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:02:14,858 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:02:15,154 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPUSD — 5040 samples (group=dollar) score_means={'trend_score': 0.4843, 'range_score': 0.2371, 'chop_score': 0.4614, 'volatility_percentile': 0.377, 'consolidation_score': 0.1835}
2026-05-11 14:02:15,253 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2317, 'chop_score': 0.462, 'volatility_percentile': 0.3807, 'consolidation_score': 0.1858}, 'dollar': {'trend_score': 0.4937, 'range_score': 0.2348, 'chop_score': 0.4583, 'volatility_percentile': 0.3806, 'consolidation_score': 0.1839}, 'gold': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}}
2026-05-11 14:02:15,253 INFO Regime[1H mode=ltf_behaviour] score means by year: {2022: {'trend_score': 0.4905, 'range_score': 0.2364, 'chop_score': 0.4625, 'volatility_percentile': 0.385, 'consolidation_score': 0.1797}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 14:02:15,325 INFO Regime phase LTF dataset build fold=train_all: 9.0s (train=262644 val=30352)
2026-05-11 14:02:15,325 INFO Regime 1H/ltf_behaviour cold start: no existing weights found
2026-05-11 14:02:15,358 INFO RegimeClassifier[mode=ltf_behaviour]: cold start score head
2026-05-11 14:02:15,359 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-11 14:02:15,872 INFO Regime score epoch  1/50 — tr=0.0990 va=0.0782 mae={'trend_score': 0.1733, 'range_score': 0.2868, 'chop_score': 0.1619, 'volatility_percentile': 0.185, 'consolidation_score': 0.3665}
2026-05-11 14:02:16,360 INFO Regime score epoch  2/50 — tr=0.0841 va=0.0628
2026-05-11 14:02:16,849 INFO Regime score epoch  3/50 — tr=0.0634 va=0.0450
2026-05-11 14:02:17,328 INFO Regime score epoch  4/50 — tr=0.0436 va=0.0304
2026-05-11 14:02:17,819 INFO Regime score epoch  5/50 — tr=0.0296 va=0.0195 mae={'trend_score': 0.0694, 'range_score': 0.1414, 'chop_score': 0.0641, 'volatility_percentile': 0.0516, 'consolidation_score': 0.2094}
2026-05-11 14:02:18,317 INFO Regime score epoch  6/50 — tr=0.0207 va=0.0122
2026-05-11 14:02:18,832 INFO Regime score epoch  7/50 — tr=0.0154 va=0.0082
2026-05-11 14:02:19,321 INFO Regime score epoch  8/50 — tr=0.0123 va=0.0059
2026-05-11 14:02:19,824 INFO Regime score epoch  9/50 — tr=0.0105 va=0.0049
2026-05-11 14:02:20,336 INFO Regime score epoch 10/50 — tr=0.0094 va=0.0044 mae={'trend_score': 0.0535, 'range_score': 0.0569, 'chop_score': 0.0481, 'volatility_percentile': 0.032, 'consolidation_score': 0.0691}
2026-05-11 14:02:20,822 INFO Regime score epoch 11/50 — tr=0.0087 va=0.0039
2026-05-11 14:02:21,325 INFO Regime score epoch 12/50 — tr=0.0081 va=0.0036
2026-05-11 14:02:21,825 INFO Regime score epoch 13/50 — tr=0.0077 va=0.0033
2026-05-11 14:02:22,322 INFO Regime score epoch 14/50 — tr=0.0074 va=0.0031
2026-05-11 14:02:22,801 INFO Regime score epoch 15/50 — tr=0.0071 va=0.0030 mae={'trend_score': 0.0439, 'range_score': 0.0513, 'chop_score': 0.0431, 'volatility_percentile': 0.0289, 'consolidation_score': 0.0459}
2026-05-11 14:02:23,293 INFO Regime score epoch 16/50 — tr=0.0069 va=0.0028
2026-05-11 14:02:23,791 INFO Regime score epoch 17/50 — tr=0.0066 va=0.0027
2026-05-11 14:02:24,292 INFO Regime score epoch 18/50 — tr=0.0064 va=0.0026
2026-05-11 14:02:24,827 INFO Regime score epoch 19/50 — tr=0.0063 va=0.0025
2026-05-11 14:02:25,330 INFO Regime score epoch 20/50 — tr=0.0061 va=0.0024 mae={'trend_score': 0.0376, 'range_score': 0.0469, 'chop_score': 0.039, 'volatility_percentile': 0.0271, 'consolidation_score': 0.0373}
2026-05-11 14:02:25,816 INFO Regime score epoch 21/50 — tr=0.0060 va=0.0023
2026-05-11 14:02:26,303 INFO Regime score epoch 22/50 — tr=0.0059 va=0.0023
2026-05-11 14:02:26,787 INFO Regime score epoch 23/50 — tr=0.0058 va=0.0022
2026-05-11 14:02:27,279 INFO Regime score epoch 24/50 — tr=0.0057 va=0.0021
2026-05-11 14:02:27,778 INFO Regime score epoch 25/50 — tr=0.0056 va=0.0021 mae={'trend_score': 0.0332, 'range_score': 0.0447, 'chop_score': 0.0352, 'volatility_percentile': 0.0252, 'consolidation_score': 0.0336}
2026-05-11 14:02:28,290 INFO Regime score epoch 26/50 — tr=0.0055 va=0.0020
2026-05-11 14:02:28,821 INFO Regime score epoch 27/50 — tr=0.0054 va=0.0020
2026-05-11 14:02:29,303 INFO Regime score epoch 28/50 — tr=0.0054 va=0.0019
2026-05-11 14:02:29,823 INFO Regime score epoch 29/50 — tr=0.0053 va=0.0019
2026-05-11 14:02:30,328 INFO Regime score epoch 30/50 — tr=0.0053 va=0.0018 mae={'trend_score': 0.0302, 'range_score': 0.0427, 'chop_score': 0.0326, 'volatility_percentile': 0.0239, 'consolidation_score': 0.0315}
2026-05-11 14:02:30,819 INFO Regime score epoch 31/50 — tr=0.0052 va=0.0018
2026-05-11 14:02:31,302 INFO Regime score epoch 32/50 — tr=0.0052 va=0.0018
2026-05-11 14:02:31,797 INFO Regime score epoch 33/50 — tr=0.0051 va=0.0017
2026-05-11 14:02:32,301 INFO Regime score epoch 34/50 — tr=0.0051 va=0.0017
2026-05-11 14:02:32,787 INFO Regime score epoch 35/50 — tr=0.0051 va=0.0017 mae={'trend_score': 0.0286, 'range_score': 0.0424, 'chop_score': 0.0305, 'volatility_percentile': 0.0229, 'consolidation_score': 0.0301}
2026-05-11 14:02:33,288 INFO Regime score epoch 36/50 — tr=0.0050 va=0.0017
2026-05-11 14:02:33,795 INFO Regime score epoch 37/50 — tr=0.0050 va=0.0017
2026-05-11 14:02:34,308 INFO Regime score epoch 38/50 — tr=0.0050 va=0.0017
2026-05-11 14:02:34,815 INFO Regime score epoch 39/50 — tr=0.0050 va=0.0017
2026-05-11 14:02:35,315 INFO Regime score epoch 40/50 — tr=0.0050 va=0.0016 mae={'trend_score': 0.0276, 'range_score': 0.0416, 'chop_score': 0.0295, 'volatility_percentile': 0.0223, 'consolidation_score': 0.03}
2026-05-11 14:02:35,807 INFO Regime score epoch 41/50 — tr=0.0050 va=0.0016
2026-05-11 14:02:36,349 INFO Regime score epoch 42/50 — tr=0.0050 va=0.0016
2026-05-11 14:02:36,858 INFO Regime score epoch 43/50 — tr=0.0050 va=0.0016
2026-05-11 14:02:37,352 INFO Regime score epoch 44/50 — tr=0.0049 va=0.0016
2026-05-11 14:02:37,844 INFO Regime score epoch 45/50 — tr=0.0049 va=0.0016 mae={'trend_score': 0.0272, 'range_score': 0.0412, 'chop_score': 0.0291, 'volatility_percentile': 0.0222, 'consolidation_score': 0.0298}
2026-05-11 14:02:38,355 INFO Regime score epoch 46/50 — tr=0.0049 va=0.0016
2026-05-11 14:02:38,885 INFO Regime score epoch 47/50 — tr=0.0049 va=0.0016
2026-05-11 14:02:39,383 INFO Regime score epoch 48/50 — tr=0.0049 va=0.0016
2026-05-11 14:02:39,896 INFO Regime score epoch 49/50 — tr=0.0049 va=0.0016
2026-05-11 14:02:40,405 INFO Regime score epoch 50/50 — tr=0.0049 va=0.0016 mae={'trend_score': 0.0271, 'range_score': 0.0412, 'chop_score': 0.029, 'volatility_percentile': 0.022, 'consolidation_score': 0.0293}
2026-05-11 14:02:40,426 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0271, 'range_score': 0.0412, 'chop_score': 0.029, 'volatility_percentile': 0.022, 'consolidation_score': 0.0293} mse={'trend_score': 0.00122, 'range_score': 0.00273, 'chop_score': 0.00136, 'volatility_percentile': 0.00098, 'consolidation_score': 0.00172} corr={'trend_score': 0.9875, 'range_score': 0.9322, 'chop_score': 0.9814, 'volatility_percentile': 0.9895, 'consolidation_score': 0.9817} pred_std={'trend_score': 0.2188, 'range_score': 0.1393, 'chop_score': 0.179, 'volatility_percentile': 0.2144, 'consolidation_score': 0.2136} target_std={'trend_score': 0.2219, 'range_score': 0.1433, 'chop_score': 0.1888, 'volatility_percentile': 0.2165, 'consolidation_score': 0.2179}
2026-05-11 14:02:40,747 INFO RegimeClassifier[mode=ltf_behaviour] train group diagnostics={'dollar': {'n': 16632, 'score_mae': {'trend_score': 0.0258, 'range_score': 0.0407, 'chop_score': 0.0288, 'volatility_percentile': 0.0223, 'consolidation_score': 0.0294}, 'target_score_mean': {'trend_score': 0.491, 'range_score': 0.2347, 'chop_score': 0.4613, 'volatility_percentile': 0.3824, 'consolidation_score': 0.186}, 'pred_score_mean': {'trend_score': 0.491, 'range_score': 0.2379, 'chop_score': 0.4596, 'volatility_percentile': 0.3827, 'consolidation_score': 0.1872}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3581, 29, 0, 2, 0, 0, 167], [37, 69, 0, 0, 0, 0, 4], [0, 0, 234, 9, 45, 0, 172], [3, 0, 13, 527, 44, 0, 102], [0, 0, 93, 36, 2986, 0, 201], [0, 30, 0, 0, 8, 0, 90], [250, 12, 286, 88, 124, 0, 7390]]}, 'gold': {'n': 8420, 'score_mae': {'trend_score': 0.0258, 'range_score': 0.0413, 'chop_score': 0.0287, 'volatility_percentile': 0.0224, 'consolidation_score': 0.03}, 'target_score_mean': {'trend_score': 0.4883, 'range_score': 0.2355, 'chop_score': 0.4652, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}, 'pred_score_mean': {'trend_score': 0.4892, 'range_score': 0.2378, 'chop_score': 0.4629, 'volatility_percentile': 0.3779, 'consolidation_score': 0.1928}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1792, 21, 0, 0, 0, 0, 72], [15, 39, 0, 0, 0, 0, 2], [0, 0, 122, 7, 24, 0, 91], [1, 0, 9, 326, 24, 0, 56], [0, 0, 41, 34, 1507, 0, 122], [0, 26, 0, 0, 8, 0, 47], [127, 4, 140, 50, 65, 0, 3648]]}, 'jpy': {'n': 24948, 'score_mae': {'trend_score': 0.026, 'range_score': 0.0404, 'chop_score': 0.0286, 'volatility_percentile': 0.0224, 'consolidation_score': 0.0293}, 'target_score_mean': {'trend_score': 0.4904, 'range_score': 0.2338, 'chop_score': 0.4653, 'volatility_percentile': 0.3826, 'consolidation_score': 0.1905}, 'pred_score_mean': {'trend_score': 0.4897, 'range_score': 0.2371, 'chop_score': 0.4631, 'volatility_percentile': 0.3829, 'consolidation_score': 0.1917}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[5345, 56, 0, 4, 0, 0, 241], [65, 112, 0, 0, 0, 0, 10], [0, 0, 308, 23, 64, 0, 252], [2, 0, 22, 1059, 75, 0, 156], [0, 0, 135, 93, 4547, 0, 340], [0, 64, 0, 0, 14, 0, 145], [341, 14, 420, 130, 217, 0, 10694]]}}
2026-05-11 14:02:40,928 INFO RegimeClassifier[mode=ltf_behaviour] validation group diagnostics={'dollar': {'n': 10093, 'score_mae': {'trend_score': 0.0273, 'range_score': 0.0428, 'chop_score': 0.029, 'volatility_percentile': 0.0218, 'consolidation_score': 0.0286}, 'target_score_mean': {'trend_score': 0.4863, 'range_score': 0.2392, 'chop_score': 0.4622, 'volatility_percentile': 0.3804, 'consolidation_score': 0.1814}, 'pred_score_mean': {'trend_score': 0.4872, 'range_score': 0.2403, 'chop_score': 0.461, 'volatility_percentile': 0.381, 'consolidation_score': 0.184}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[2285, 8, 0, 0, 0, 0, 119], [20, 33, 0, 0, 0, 0, 0], [0, 0, 157, 8, 36, 0, 115], [1, 0, 10, 324, 32, 0, 56], [0, 0, 61, 39, 1849, 0, 101], [0, 23, 0, 0, 3, 0, 51], [131, 7, 182, 68, 93, 0, 4281]]}, 'gold': {'n': 5117, 'score_mae': {'trend_score': 0.026, 'range_score': 0.0401, 'chop_score': 0.0288, 'volatility_percentile': 0.0217, 'consolidation_score': 0.0304}, 'target_score_mean': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}, 'pred_score_mean': {'trend_score': 0.4965, 'range_score': 0.2376, 'chop_score': 0.4562, 'volatility_percentile': 0.3828, 'consolidation_score': 0.1852}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1114, 6, 0, 0, 0, 0, 47], [16, 18, 0, 0, 0, 0, 1], [0, 0, 91, 1, 16, 0, 63], [1, 0, 7, 214, 8, 0, 25], [0, 0, 31, 14, 785, 0, 57], [0, 14, 0, 0, 4, 0, 32], [79, 2, 116, 37, 46, 0, 2272]]}, 'jpy': {'n': 15142, 'score_mae': {'trend_score': 0.0274, 'range_score': 0.0405, 'chop_score': 0.029, 'volatility_percentile': 0.0223, 'consolidation_score': 0.0293}, 'target_score_mean': {'trend_score': 0.495, 'range_score': 0.2297, 'chop_score': 0.4582, 'volatility_percentile': 0.3808, 'consolidation_score': 0.1868}, 'pred_score_mean': {'trend_score': 0.4933, 'range_score': 0.2326, 'chop_score': 0.4578, 'volatility_percentile': 0.3817, 'consolidation_score': 0.1894}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3337, 29, 0, 2, 0, 0, 150], [39, 67, 0, 0, 0, 0, 9], [0, 0, 182, 12, 60, 0, 130], [2, 0, 22, 648, 47, 0, 108], [0, 0, 69, 38, 2515, 0, 195], [0, 33, 0, 0, 8, 0, 81], [184, 13, 240, 78, 156, 0, 6688]]}}
2026-05-11 14:02:40,934 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-11 14:02:40,934 INFO Regime phase LTF train fold=train_all: 25.6s
2026-05-11 14:02:41,034 INFO Regime LTF complete fold=train_all: score_accuracy=0.970, train=262644 val=30352 mae={'trend_score': 0.0271, 'range_score': 0.0412, 'chop_score': 0.029, 'volatility_percentile': 0.022, 'consolidation_score': 0.0293}
2026-05-11 14:02:41,036 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:02:41,378 INFO Regime[1H mode=ltf_behaviour fold=latest] LTF score diagnostics on GBPUSD:
{'trend_score': {'mean': 0.4921, 'q10': 0.1918, 'q50': 0.4877, 'q90': 0.8002}, 'range_score': {'mean': 0.2337, 'q10': 0.0513, 'q50': 0.2125, 'q90': 0.4349}, 'chop_score': {'mean': 0.4612, 'q10': 0.2148, 'q50': 0.4503, 'q90': 0.7255}, 'volatility_percentile': {'mean': 0.3832, 'q10': 0.1018, 'q50': 0.3704, 'q90': 0.6823}, 'consolidation_score': {'mean': 0.1854, 'q10': 0.0, 'q50': 0.1185, 'q90': 0.506}}
2026-05-11 14:02:41,381 INFO Regime retrain total: 67.6s (370559 train+val samples)
2026-05-11 14:02:41,385 INFO Retrain complete. Total wall-clock: 67.6s
2026-05-11 14:02:42,326 INFO Model regime: SUCCESS
2026-05-11 14:02:42,326 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:02:42,326 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 14:02:42,327 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-11 14:02:42,327 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-05-11 14:02:42,327 INFO   [DORMANT] rl_ppo — RL_ENABLED=false, skipping
2026-05-11 14:02:42,327 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-05-11 14:02:42,327 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer']
2026-05-11 14:02:42,329 INFO Saved 5 retrain records to metrics/

=== TRAINING COMPLETE ===
  gru: SUCCESS
  regime: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Clean Quality/RL source: Backtest on train window ===
  START Train-window backtest for Quality/RL labels
2026-05-11 14:02:42,936 INFO === STEP 6: BACKTEST (train) ===
2026-05-11 14:02:42,937 INFO BT_WINDOW=train — train-window backtest: 2016-01-04 → 2023-08-04 (clean Quality/RL labels)
2026-05-11 14:02:42,937 INFO ================================================================
  ROUND 0 / 3
================================================================
2026-05-11 14:02:42,937 INFO Round 0 — running backtest: 2016-01-04 → 2023-08-04 (ml_trader, shared ML cache)
2026-05-11 14:06:40,045 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for USDJPY with 2
2026-05-11 14:06:40,066 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for USDJPY with 0.3333333333333333
2026-05-11 14:06:40,231 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for EURUSD with 2
2026-05-11 14:06:40,255 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for EURUSD with 0.3333333333333333
2026-05-11 14:06:40,388 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for USDJPY with 2
2026-05-11 14:06:40,408 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for USDJPY with 0.25
2026-05-11 14:06:40,446 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for EURJPY with 2
2026-05-11 14:06:40,449 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for USDJPY: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 14:06:40,464 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for EURJPY with 0.3333333333333333
2026-05-11 14:06:40,688 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for EURUSD with 2
2026-05-11 14:06:40,704 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for EURUSD with 0.25
2026-05-11 14:06:40,734 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for EURUSD: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 14:06:40,954 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for EURJPY with 2
2026-05-11 14:06:40,979 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for EURJPY with 0.25
2026-05-11 14:06:41,011 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for EURJPY: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 14:06:44,747 WARNING ML cache score overlay filled 4 warmup/alignment gaps for USDJPY
2026-05-11 14:06:45,543 WARNING ML cache score overlay filled 4 warmup/alignment gaps for EURUSD
2026-05-11 14:06:47,230 WARNING ML cache score overlay filled 4 warmup/alignment gaps for EURJPY
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
2026-05-11 14:06:54,437 WARNING _build_sequence_df: HTF frame 5M filled 357 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 14:06:56,200 WARNING _build_sequence_df: HTF frame 5M filled 357 warmup/alignment gaps with 0.000
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
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
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
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
2026-05-11 14:06:59,673 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 14:06:59,759 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 14:06:59,789 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 14:06:59,834 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 14:06:59,860 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 14:06:59,894 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
2026-05-11 14:06:59,897 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 14:06:59,945 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 14:06:59,959 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 14:06:59,977 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 14:07:00,021 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
2026-05-11 14:07:00,023 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 14:07:00,046 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 14:07:00,067 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 14:07:00,120 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
2026-05-11 14:07:00,122 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 14:07:00,123 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 14:07:00,158 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 14:07:00,162 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 14:07:00,186 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 14:07:00,213 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 14:07:00,256 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 14:07:00,259 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 14:07:00,306 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
2026-05-11 14:07:00,309 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
2026-05-11 14:07:00,309 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 14:07:00,344 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 14:07:00,388 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 14:07:00,454 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 14:07:00,500 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 14:07:00,545 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 14:07:00,627 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 14:07:00,780 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 14:07:19,525 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for GBPJPY with 2
2026-05-11 14:07:19,540 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for GBPJPY with 0.3333333333333333
2026-05-11 14:07:19,683 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for GBPUSD with 2
2026-05-11 14:07:19,698 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for GBPUSD with 0.3333333333333333
2026-05-11 14:07:19,835 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for GBPJPY with 2
2026-05-11 14:07:19,860 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for GBPJPY with 0.25
2026-05-11 14:07:19,887 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for GBPJPY: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 14:07:20,050 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for GBPUSD with 2
2026-05-11 14:07:20,064 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for GBPUSD with 0.25
2026-05-11 14:07:20,079 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for GBPUSD: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 14:07:20,484 WARNING ML cache score overlay filled 4 warmup/alignment gaps for GBPJPY
2026-05-11 14:07:21,601 WARNING ML cache score overlay filled 4 warmup/alignment gaps for GBPUSD
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
2026-05-11 14:07:29,457 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 14:07:29,513 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 14:07:29,544 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 14:07:29,557 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 14:07:29,583 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 14:07:29,604 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 14:07:29,621 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 14:07:29,644 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 14:07:29,673 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 14:07:29,677 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 14:07:29,693 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 14:07:29,716 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 14:07:29,733 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 14:07:29,754 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 14:07:29,787 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 14:07:29,789 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 14:07:29,808 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 14:07:29,833 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 14:07:29,853 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 14:07:29,872 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 14:07:29,891 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 14:07:29,934 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260511_140245.json
Trader                                   Trades      WR    PF*  Return*    ExpR   TP1%   TP2%     DD*  Sharpe*  Sortino     IC  Verdict
-----------------------------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)              596  23.2%   0.79  -96.8%  -0.162 23.2%  4.0%  99.4%    -1.64    -0.16 -0.061     FAIL
  FAILED rules: positive_expectancy, profit_factor_min_1_25, drawdown_below_20pct, sharpe_positive, sortino_positive, win_rate_above_breakeven, mc_p10_not_ruin, sharpe_ci_positive
  monthly R: 2023-02=+1.85  2023-03=+0.28  2023-04=-8.62  2023-05=-8.32  2023-06=+5.65  2023-07=-10.01
  MonteCarlo P95 DD=115.9%  P10 equity=319  t=-2.52 (p=0.012)  Sharpe CI=[-3.11, -0.29]  streak=22
  gate_diagnostics: bars=1049680 no_signal=766040 quality_block=0 session_skip=283037 density=7 pm_reject=0 daily_skip=0 cooldown=0 daily_halt_events=0 enforce_daily_halt=False
  no_signal_reasons: no_trade_uncertain=295765, weak_gru_direction=191277, no_trade_chop=92932, no_trade_extreme_vol=73786, wait_pullback=55293, gru_expected_r_below_threshold=29816

Calibration Summary:
  all          [OK] Too few populated bins for calibration check
  ml_trader    [OK] Too few populated bins for calibration check
2026-05-11 14:10:37,465 INFO Round 0 backtest — 596 trades | avg WR=23.2% | avg PF=0.79 | avg Sharpe=-1.64
2026-05-11 14:10:37,465 INFO   ml_trader: 596 trades | WR=23.2% | fixed PF=0.79 | Return=-96.8% | ExpR=-0.162 | DD=99.4% | Sharpe=-1.64
2026-05-11 14:10:37,465 INFO   ml_trader gate_diagnostics: bars=1049680 no_signal=766040 quality_block=0 session_skip=283037 density=7 pm_reject=0
2026-05-11 14:10:37,465 INFO   ml_trader no_signal_reasons: {'wait_pullback': 55293, 'weak_gru_direction': 191277, 'trend_structure_missing': 22529, 'no_trade_extreme_vol': 73786, 'no_trade_uncertain': 295765, 'gru_expected_r_below_threshold': 29816, 'no_trade_chop': 92932, 'htf_low_regime_confidence': 3340, 'tradeability_direction_conflict': 1255, 'expected_r_below_threshold': 47}
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_0.json
INFO  Total trades: 596
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (596 rows)
2026-05-11 14:10:38,177 INFO Round 0: wrote 596 journal entries (total in file: 596)
  DONE  Train-window backtest for Quality/RL labels
  Saved Train Quality/RL source result → train_quality_rl_source_summary.json
  Train-label journal entries: 596

=== Train Quality + RL on train-only journal ===
  START Train-only Quality+RL retrain
2026-05-11 14:10:38,555 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-05-11 14:10:38,577 INFO Journal entries: 596 total, 596 allowed for training (['live', 'paper', 'production', 'train'])
2026-05-11 14:10:38,577 INFO --- Training quality ---
2026-05-11 14:10:38,577 INFO Running retrain --model quality with JOURNAL_ALLOWED_SPLITS=train,live,paper,production
2026-05-11 14:10:38,764 INFO retrain environment: KAGGLE
2026-05-11 14:10:40,360 INFO Device: CUDA (2 GPU(s))
2026-05-11 14:10:40,373 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 14:10:40,373 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 14:10:40,373 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 14:10:40,373 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 14:10:40,373 INFO Retrain data split: train
2026-05-11 14:10:40,373 INFO Retrain rolling fold selector: latest
2026-05-11 14:10:40,374 INFO === QualityScorer retrain ===
2026-05-11 14:10:40,523 INFO NumExpr defaulting to 4 threads.
2026-05-11 14:10:40,715 INFO QualityScorer: CUDA available — using GPU
2026-05-11 14:10:40,795 INFO QualityScorer: group EV smoothing applied to 564/596 rows (blend=30% group, min_group=10)
2026-05-11 14:10:40,798 INFO Quality phase label creation: 0.1s (596 trades)
2026-05-11 14:10:40,876 INFO QualityScorer: group EV smoothing applied to 564/596 rows (blend=30% group, min_group=10)
2026-05-11 14:10:40,879 INFO QualityScorer: 596 samples, EV stats={'mean': -0.452862411737442, 'std': 0.7471243143081665, 'n_pos': 138, 'n_neg': 458}, device=cuda
2026-05-11 14:10:41,080 INFO QualityScorer: DataParallel across 2 GPUs
2026-05-11 14:10:41,080 INFO QualityScorer: cold start
2026-05-11 14:10:41,081 INFO QualityScorer: pos_weight=3.45 (n_pos=107 n_neg=369)
2026-05-11 14:10:43,405 INFO Quality epoch   1/100 — va_huber=0.6719
2026-05-11 14:10:43,441 INFO Quality epoch   2/100 — va_huber=0.6725
2026-05-11 14:10:43,462 INFO Quality epoch   3/100 — va_huber=0.6731
2026-05-11 14:10:43,482 INFO Quality epoch   4/100 — va_huber=0.6734
2026-05-11 14:10:43,510 INFO Quality epoch   5/100 — va_huber=0.6740
2026-05-11 14:10:43,631 INFO Quality epoch  11/100 — va_huber=0.6781
2026-05-11 14:10:43,631 INFO Quality early stop at epoch 11
2026-05-11 14:10:43,639 INFO QualityScorer EV model: MAE=0.909 dir_acc=0.258 n_val=120
2026-05-11 14:10:43,643 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-05-11 14:10:43,689 INFO Quality phase train: 2.9s | total: 3.3s
2026-05-11 14:10:43,697 INFO Retrain complete. Total wall-clock: 3.3s
2026-05-11 14:10:44,715 INFO Model quality: SUCCESS
2026-05-11 14:10:44,715 INFO --- Training rl ---
2026-05-11 14:10:44,715 INFO Running retrain --model rl with JOURNAL_ALLOWED_SPLITS=train,live,paper,production
2026-05-11 14:10:44,912 INFO retrain environment: KAGGLE
2026-05-11 14:10:46,506 INFO Device: CUDA (2 GPU(s))
2026-05-11 14:10:46,517 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 14:10:46,517 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 14:10:46,517 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 14:10:46,517 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 14:10:46,517 INFO Retrain data split: train
2026-05-11 14:10:46,517 INFO Retrain rolling fold selector: latest
2026-05-11 14:10:46,518 INFO === RLAgent (PPO) retrain ===
2026-05-11 14:10:46,666 INFO NumExpr defaulting to 4 threads.
2026-05-11 14:10:46,858 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260511_141046
2026-05-11 14:10:46,884 INFO RL phase episode loading: 0.0s (596 episodes)
2026-05-11 14:10:50.450125: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1778508650.699138   75919 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1778508650.767839   75919 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1778508651.349507   75919 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1778508651.349558   75919 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1778508651.349565   75919 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1778508651.349569   75919 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-05-11 14:11:08,390 INFO RLAgent: cold start — building new PPO policy
2026-05-11 14:11:19,990 INFO RLAgent: retrain complete, 596 episodes
2026-05-11 14:11:19,991 INFO RL phase PPO train: 33.1s | total: 33.5s
2026-05-11 14:11:20,003 INFO Retrain complete. Total wall-clock: 33.5s
2026-05-11 14:11:21,601 INFO Model rl: SUCCESS
2026-05-11 14:11:21,601 INFO Step 7b complete — summary: /kaggle/working/Multi-Bot/trading-system/ml_training/metrics/training_7b_summary.json
  DONE  Train-only Quality+RL retrain
  Archived journal → trade_journal_train_only.jsonl
  Archived journal CSV → trade_journal_train_only.csv

=== Round 1: Backtest on train-tail window (latest 2yr inside training data) ===
  Cleared journal for fresh Round 1 run
  START Round 1 - Backtest (train-tail)
2026-05-11 14:11:22,100 INFO === STEP 6: BACKTEST (round1) ===
2026-05-11 14:11:22,102 INFO BT_WINDOW=round1 — train-tail backtest: 2021-08-05 → 2023-08-04 (seen training data; test set protected)
2026-05-11 14:11:22,103 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-05-11 14:11:22,103 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
2026-05-11 14:11:22,103 INFO Round 1 — QualityScorer hard gate disabled; set BACKTEST_ENABLE_QUALITY_GATE=1 to enable it
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
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
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
2026-05-11 14:12:38,620 WARNING _build_sequence_df: HTF frame 5M filled 461 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
2026-05-11 14:12:39,393 WARNING _build_sequence_df: HTF frame 5M filled 461 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
2026-05-11 14:12:39,499 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 14:12:39,586 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 14:12:39,661 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
2026-05-11 14:12:39,662 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 14:12:39,735 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 14:12:39,743 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
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
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
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
2026-05-11 14:12:50,315 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 14:12:50,341 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 14:12:50,358 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 14:12:50,391 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
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
2026-05-11 14:13:39,935 INFO Round 1 backtest — 155 trades | avg WR=25.8% | avg PF=0.94 | avg Sharpe=-0.41
2026-05-11 14:13:39,935 INFO   ml_trader: 155 trades | WR=25.8% | fixed PF=0.94 | Return=-6.7% | ExpR=-0.043 | DD=20.1% | Sharpe=-0.41
2026-05-11 14:13:39,935 INFO   ml_trader gate_diagnostics: bars=263960 no_signal=189173 quality_block=0 session_skip=74630 density=2 pm_reject=0
2026-05-11 14:13:39,935 INFO   ml_trader no_signal_reasons: {'no_trade_uncertain': 70727, 'trend_structure_missing': 5305, 'weak_gru_direction': 47181, 'no_trade_chop': 22989, 'gru_expected_r_below_threshold': 7185, 'no_trade_extreme_vol': 19655, 'wait_pullback': 14557, 'tradeability_direction_conflict': 469, 'htf_low_regime_confidence': 1088, 'expected_r_below_threshold': 17}
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 155
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (155 rows)
2026-05-11 14:13:40,304 INFO Round 1: wrote 155 journal entries (total in file: 155)
  DONE  Round 1 - Backtest (train-tail)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 155 entries

  SKIP  Round 1 Quality+RL retrain — train-tail journal kept evaluation-only

  QualityScorer trade count: R0=596 R1=155 combined=751 (floor=50)
  Combined R0+R1 journal → trade_journal_qs_combined.jsonl (751 trades)

=== QualityScorer: 751 combined trades ≥ 50 — training and activating ===
  START Retrain quality [R0+R1 combined journal]
2026-05-11 14:13:40,745 INFO retrain environment: KAGGLE
2026-05-11 14:13:42,361 INFO Device: CUDA (2 GPU(s))
2026-05-11 14:13:42,372 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 14:13:42,372 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 14:13:42,373 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 14:13:42,374 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 14:13:42,374 INFO Retrain data split: train
2026-05-11 14:13:42,374 INFO Retrain rolling fold selector: latest
2026-05-11 14:13:42,376 INFO === QualityScorer retrain ===
2026-05-11 14:13:42,519 INFO NumExpr defaulting to 4 threads.
2026-05-11 14:13:42,710 INFO QualityScorer: CUDA available — using GPU
2026-05-11 14:13:42,916 INFO QualityScorer loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (device=cuda)
2026-05-11 14:13:42,947 INFO QualityScorer: skipped 155 journal records outside allowed splits ['combined_eval', 'live', 'paper', 'production', 'test', 'train', 'validation']
2026-05-11 14:13:43,004 INFO QualityScorer: group EV smoothing applied to 564/596 rows (blend=30% group, min_group=10)
2026-05-11 14:13:43,007 INFO Quality phase label creation: 0.1s (596 trades)
2026-05-11 14:13:43,007 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/quality_scorer.pkl_20260511_141343
2026-05-11 14:13:43,038 INFO QualityScorer: skipped 155 journal records outside allowed splits ['combined_eval', 'live', 'paper', 'production', 'test', 'train', 'validation']
2026-05-11 14:13:43,092 INFO QualityScorer: group EV smoothing applied to 564/596 rows (blend=30% group, min_group=10)
2026-05-11 14:13:43,095 INFO QualityScorer: 596 samples, EV stats={'mean': -0.452862411737442, 'std': 0.7471243143081665, 'n_pos': 138, 'n_neg': 458}, device=cuda
2026-05-11 14:13:43,095 INFO QualityScorer: warm start from existing weights
2026-05-11 14:13:43,096 INFO QualityScorer: pos_weight=3.45 (n_pos=107 n_neg=369)
2026-05-11 14:13:45,308 INFO Quality epoch   1/100 — va_huber=0.6729
2026-05-11 14:13:45,345 INFO Quality epoch   2/100 — va_huber=0.6740
2026-05-11 14:13:45,365 INFO Quality epoch   3/100 — va_huber=0.6751
2026-05-11 14:13:45,556 INFO Quality epoch   4/100 — va_huber=0.6761
2026-05-11 14:13:45,577 INFO Quality epoch   5/100 — va_huber=0.6774
2026-05-11 14:13:45,699 INFO Quality epoch  11/100 — va_huber=0.6877
2026-05-11 14:13:45,699 INFO Quality early stop at epoch 11
2026-05-11 14:13:45,708 INFO QualityScorer EV model: MAE=0.913 dir_acc=0.258 n_val=120
2026-05-11 14:13:45,713 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-05-11 14:13:45,760 INFO Quality phase train: 2.8s | total: 3.4s
2026-05-11 14:13:45,771 INFO Retrain complete. Total wall-clock: 3.4s
  DONE  Retrain quality [R0+R1 combined journal]
  QualityScorer trained — gate ACTIVE for Round 2+

=== Pre-Round 2: Incremental retrain (GRU + Regime) ===
  START Retrain gru [pre-R2 retrain]
2026-05-11 14:13:46,945 INFO retrain environment: KAGGLE
2026-05-11 14:13:48,585 INFO Device: CUDA (2 GPU(s))
2026-05-11 14:13:48,594 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 14:13:48,594 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 14:13:48,594 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 14:13:48,597 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 14:13:48,597 INFO Retrain data split: train
2026-05-11 14:13:48,598 INFO Retrain rolling fold selector: latest
2026-05-11 14:13:48,599 INFO === GRU-LSTM retrain (timeframes: ['15M']) ===
2026-05-11 14:13:48,740 INFO NumExpr defaulting to 4 threads.
2026-05-11 14:13:48,930 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-11 14:13:48,930 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 14:13:48,930 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 14:13:49,172 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-05-11 14:13:49,173 INFO GRU multi-symbol training (Kaggle mode): 6 symbols × ['15M']
2026-05-11 14:13:49,175 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260511_141349
2026-05-11 14:13:49,179 INFO GRU feature contract unchanged (input_size=94) — incremental retrain
2026-05-11 14:13:49,180 INFO Deleted stale GRU artifact for cold start: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:13:49,180 INFO GRU warm start disabled by default; set GRU_ALLOW_WARM_START=1 to reuse compatible weights
2026-05-11 14:13:49,437 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:13:49,464 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:13:49,480 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:13:49,489 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:13:49,561 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 14:13:49,567 INFO Loaded XAUUSD/15M split=train fold=latest: 176438 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:13:50,124 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:13:50,144 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:13:50,161 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:13:50,170 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:13:50,213 INFO Loaded EURUSD/15M split=train fold=latest: 174951 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:13:50,764 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:13:50,785 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:13:50,799 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:13:50,807 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:13:50,845 INFO Loaded USDJPY/15M split=train fold=latest: 174944 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:13:51,373 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:13:51,394 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:13:51,409 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:13:51,416 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:13:51,453 INFO Loaded EURJPY/15M split=train fold=latest: 174948 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:13:51,971 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:13:51,991 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:13:52,005 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:13:52,013 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:13:52,053 INFO Loaded GBPJPY/15M split=train fold=latest: 174902 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:13:52,567 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:13:52,586 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:13:52,601 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:13:52,609 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:13:52,647 INFO Loaded GBPUSD/15M split=train fold=latest: 174899 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:13:53,066 INFO train_multi: 6 segments, ~971854 total bars
2026-05-11 14:13:53,073 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-05-11 14:13:53,073 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-05-11 14:13:53,073 INFO train_multi: training ALL 6 segments across TFs ['15M'] in one combined pass
2026-05-11 14:13:53,073 INFO train_multi: building combined dataset for TF=ALL (6 segments)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:14:05,097 INFO train_multi TF=ALL: 971494 sequences across 6 segments
2026-05-11 14:14:05,097 INFO train_multi TF=ALL: estimated peak RAM = 27072 MB (train=419996 calib=60000 val=120002 n_feat=94 seq_len=60)
2026-05-11 14:14:05,097 WARNING train_multi TF=ALL: trimming to fit RAM budget — new train=310283 calib=44326 val=88652 (20000 MB est)
2026-05-11 14:14:07,421 INFO train_multi TF=ALL: train=310283 calib=44326 val=88652 (10007 MB tensors)
2026-05-11 14:14:14,147 INFO train_multi TF=ALL: structural bar weighting — 199279 structural bars (64.2%) weight=15.0 structural_only=0
2026-05-11 14:14:15,138 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=25, min_epochs=45)
2026-05-11 14:14:30,335 INFO train_multi TF=ALL epoch 1/100 train=2.3387 val=2.3410 r_mae=0.973 pos_r_acc=0.507 side_acc=0.490 r_n=127469
2026-05-11 14:14:30,340 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:14:30,340 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:14:30,340 INFO train_multi TF=ALL: new best val=2.3410 r_mae=0.9732 — saved
2026-05-11 14:14:30,344 INFO train_multi TF=ALL: new best r_mae=0.9732 — saved rmae checkpoint
2026-05-11 14:14:43,546 INFO train_multi TF=ALL epoch 2/100 train=2.3372 val=2.3392 r_mae=0.972 pos_r_acc=0.507 side_acc=0.490 r_n=127469
2026-05-11 14:14:43,551 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:14:43,551 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:14:43,551 INFO train_multi TF=ALL: new best val=2.3392 r_mae=0.9718 — saved
2026-05-11 14:14:43,556 INFO train_multi TF=ALL: new best r_mae=0.9718 — saved rmae checkpoint
2026-05-11 14:14:56,824 INFO train_multi TF=ALL epoch 3/100 train=2.3351 val=2.3374 r_mae=0.970 pos_r_acc=0.545 side_acc=0.490 r_n=127469
2026-05-11 14:14:56,829 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:14:56,829 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:14:56,829 INFO train_multi TF=ALL: new best val=2.3374 r_mae=0.9703 — saved
2026-05-11 14:14:56,834 INFO train_multi TF=ALL: new best r_mae=0.9703 — saved rmae checkpoint
2026-05-11 14:15:10,202 INFO train_multi TF=ALL epoch 4/100 train=2.3338 val=2.3351 r_mae=0.968 pos_r_acc=0.545 side_acc=0.498 r_n=127469
2026-05-11 14:15:10,207 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:15:10,207 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:15:10,207 INFO train_multi TF=ALL: new best val=2.3351 r_mae=0.9682 — saved
2026-05-11 14:15:10,212 INFO train_multi TF=ALL: new best r_mae=0.9682 — saved rmae checkpoint
2026-05-11 14:15:23,215 INFO train_multi TF=ALL epoch 5/100 train=2.3326 val=2.3333 r_mae=0.966 pos_r_acc=0.545 side_acc=0.500 r_n=127469
2026-05-11 14:15:23,220 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:15:23,221 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:15:23,221 INFO train_multi TF=ALL: new best val=2.3333 r_mae=0.9662 — saved
2026-05-11 14:15:23,225 INFO train_multi TF=ALL: new best r_mae=0.9662 — saved rmae checkpoint
2026-05-11 14:15:36,508 INFO train_multi TF=ALL epoch 6/100 train=2.3319 val=2.3329 r_mae=0.966 pos_r_acc=0.545 side_acc=0.502 r_n=127469
2026-05-11 14:15:36,514 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:15:36,514 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:15:36,514 INFO train_multi TF=ALL: new best val=2.3329 r_mae=0.9660 — saved
2026-05-11 14:15:36,518 INFO train_multi TF=ALL: new best r_mae=0.9660 — saved rmae checkpoint
2026-05-11 14:15:49,753 INFO train_multi TF=ALL epoch 7/100 train=2.3308 val=2.3322 r_mae=0.966 pos_r_acc=0.545 side_acc=0.498 r_n=127469
2026-05-11 14:15:49,758 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:15:49,758 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:15:49,758 INFO train_multi TF=ALL: new best val=2.3322 r_mae=0.9657 — saved
2026-05-11 14:15:49,762 INFO train_multi TF=ALL: new best r_mae=0.9657 — saved rmae checkpoint
2026-05-11 14:16:02,855 INFO train_multi TF=ALL epoch 8/100 train=2.3303 val=2.3310 r_mae=0.965 pos_r_acc=0.545 side_acc=0.515 r_n=127469
2026-05-11 14:16:02,860 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:16:02,860 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:16:02,860 INFO train_multi TF=ALL: new best val=2.3310 r_mae=0.9653 — saved
2026-05-11 14:16:02,864 INFO train_multi TF=ALL: new best r_mae=0.9653 — saved rmae checkpoint
2026-05-11 14:16:16,159 INFO train_multi TF=ALL epoch 9/100 train=2.3288 val=2.3286 r_mae=0.964 pos_r_acc=0.545 side_acc=0.516 r_n=127469
2026-05-11 14:16:16,165 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:16:16,165 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:16:16,165 INFO train_multi TF=ALL: new best val=2.3286 r_mae=0.9644 — saved
2026-05-11 14:16:16,170 INFO train_multi TF=ALL: new best r_mae=0.9644 — saved rmae checkpoint
2026-05-11 14:16:29,245 INFO train_multi TF=ALL epoch 10/100 train=2.3255 val=2.3258 r_mae=0.964 pos_r_acc=0.546 side_acc=0.517 r_n=127469
2026-05-11 14:16:29,255 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:16:29,255 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:16:29,255 INFO train_multi TF=ALL: new best val=2.3258 r_mae=0.9635 — saved
2026-05-11 14:16:29,259 INFO train_multi TF=ALL: new best r_mae=0.9635 — saved rmae checkpoint
2026-05-11 14:16:42,520 INFO train_multi TF=ALL epoch 11/100 train=2.3227 val=2.3235 r_mae=0.963 pos_r_acc=0.545 side_acc=0.522 r_n=127469
2026-05-11 14:16:42,525 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:16:42,525 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:16:42,526 INFO train_multi TF=ALL: new best val=2.3235 r_mae=0.9633 — saved
2026-05-11 14:16:42,530 INFO train_multi TF=ALL: new best r_mae=0.9633 — saved rmae checkpoint
2026-05-11 14:16:55,839 INFO train_multi TF=ALL epoch 12/100 train=2.3217 val=2.3222 r_mae=0.963 pos_r_acc=0.545 side_acc=0.527 r_n=127469
2026-05-11 14:16:55,844 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:16:55,844 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:16:55,845 INFO train_multi TF=ALL: new best val=2.3222 r_mae=0.9632 — saved
2026-05-11 14:16:55,849 INFO train_multi TF=ALL: new best r_mae=0.9632 — saved rmae checkpoint
2026-05-11 14:17:08,897 INFO train_multi TF=ALL epoch 13/100 train=2.3207 val=2.3202 r_mae=0.962 pos_r_acc=0.546 side_acc=0.533 r_n=127469
2026-05-11 14:17:08,904 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:17:08,904 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:17:08,904 INFO train_multi TF=ALL: new best val=2.3202 r_mae=0.9625 — saved
2026-05-11 14:17:08,908 INFO train_multi TF=ALL: new best r_mae=0.9625 — saved rmae checkpoint
2026-05-11 14:17:22,282 INFO train_multi TF=ALL epoch 14/100 train=2.3182 val=2.3187 r_mae=0.961 pos_r_acc=0.547 side_acc=0.533 r_n=127469
2026-05-11 14:17:22,288 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:17:22,288 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:17:22,288 INFO train_multi TF=ALL: new best val=2.3187 r_mae=0.9614 — saved
2026-05-11 14:17:22,292 INFO train_multi TF=ALL: new best r_mae=0.9614 — saved rmae checkpoint
2026-05-11 14:17:35,777 INFO train_multi TF=ALL epoch 15/100 train=2.3154 val=2.3163 r_mae=0.961 pos_r_acc=0.548 side_acc=0.537 r_n=127469
2026-05-11 14:17:35,783 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:17:35,783 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:17:35,783 INFO train_multi TF=ALL: new best val=2.3163 r_mae=0.9607 — saved
2026-05-11 14:17:35,788 INFO train_multi TF=ALL: new best r_mae=0.9607 — saved rmae checkpoint
2026-05-11 14:17:49,714 INFO train_multi TF=ALL epoch 16/100 train=2.3122 val=2.3138 r_mae=0.959 pos_r_acc=0.551 side_acc=0.540 r_n=127469
2026-05-11 14:17:49,719 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:17:49,720 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:17:49,720 INFO train_multi TF=ALL: new best val=2.3138 r_mae=0.9589 — saved
2026-05-11 14:17:49,724 INFO train_multi TF=ALL: new best r_mae=0.9589 — saved rmae checkpoint
2026-05-11 14:18:02,998 INFO train_multi TF=ALL epoch 17/100 train=2.3105 val=2.3125 r_mae=0.958 pos_r_acc=0.552 side_acc=0.538 r_n=127469
2026-05-11 14:18:03,004 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:18:03,004 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:18:03,004 INFO train_multi TF=ALL: new best val=2.3125 r_mae=0.9582 — saved
2026-05-11 14:18:03,008 INFO train_multi TF=ALL: new best r_mae=0.9582 — saved rmae checkpoint
2026-05-11 14:18:16,534 INFO train_multi TF=ALL epoch 18/100 train=2.3086 val=2.3103 r_mae=0.957 pos_r_acc=0.556 side_acc=0.540 r_n=127469
2026-05-11 14:18:16,539 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:18:16,540 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:18:16,540 INFO train_multi TF=ALL: new best val=2.3103 r_mae=0.9574 — saved
2026-05-11 14:18:16,544 INFO train_multi TF=ALL: new best r_mae=0.9574 — saved rmae checkpoint
2026-05-11 14:18:29,864 INFO train_multi TF=ALL epoch 19/100 train=2.3043 val=2.3060 r_mae=0.954 pos_r_acc=0.560 side_acc=0.542 r_n=127469
2026-05-11 14:18:29,870 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:18:29,870 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:18:29,870 INFO train_multi TF=ALL: new best val=2.3060 r_mae=0.9545 — saved
2026-05-11 14:18:29,874 INFO train_multi TF=ALL: new best r_mae=0.9545 — saved rmae checkpoint
2026-05-11 14:18:43,573 INFO train_multi TF=ALL epoch 20/100 train=2.3002 val=2.2982 r_mae=0.952 pos_r_acc=0.563 side_acc=0.548 r_n=127469
2026-05-11 14:18:43,579 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:18:43,579 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:18:43,579 INFO train_multi TF=ALL: new best val=2.2982 r_mae=0.9522 — saved
2026-05-11 14:18:43,584 INFO train_multi TF=ALL: new best r_mae=0.9522 — saved rmae checkpoint
2026-05-11 14:18:57,095 INFO train_multi TF=ALL epoch 21/100 train=2.2919 val=2.2965 r_mae=0.948 pos_r_acc=0.567 side_acc=0.546 r_n=127469
2026-05-11 14:18:57,100 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:18:57,101 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:18:57,101 INFO train_multi TF=ALL: new best val=2.2965 r_mae=0.9483 — saved
2026-05-11 14:18:57,105 INFO train_multi TF=ALL: new best r_mae=0.9483 — saved rmae checkpoint
2026-05-11 14:19:10,491 INFO train_multi TF=ALL epoch 22/100 train=2.2848 val=2.2813 r_mae=0.945 pos_r_acc=0.574 side_acc=0.553 r_n=127469
2026-05-11 14:19:10,497 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:19:10,497 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:19:10,497 INFO train_multi TF=ALL: new best val=2.2813 r_mae=0.9452 — saved
2026-05-11 14:19:10,502 INFO train_multi TF=ALL: new best r_mae=0.9452 — saved rmae checkpoint
2026-05-11 14:19:23,942 INFO train_multi TF=ALL epoch 23/100 train=2.2767 val=2.2768 r_mae=0.938 pos_r_acc=0.579 side_acc=0.552 r_n=127469
2026-05-11 14:19:23,947 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:19:23,947 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:19:23,947 INFO train_multi TF=ALL: new best val=2.2768 r_mae=0.9384 — saved
2026-05-11 14:19:23,952 INFO train_multi TF=ALL: new best r_mae=0.9384 — saved rmae checkpoint
2026-05-11 14:19:37,089 INFO train_multi TF=ALL epoch 24/100 train=2.2660 val=2.2715 r_mae=0.937 pos_r_acc=0.580 side_acc=0.554 r_n=127469
2026-05-11 14:19:37,094 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:19:37,095 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:19:37,095 INFO train_multi TF=ALL: new best val=2.2715 r_mae=0.9368 — saved
2026-05-11 14:19:37,099 INFO train_multi TF=ALL: new best r_mae=0.9368 — saved rmae checkpoint
2026-05-11 14:19:50,326 INFO train_multi TF=ALL epoch 25/100 train=2.2598 val=2.2653 r_mae=0.934 pos_r_acc=0.585 side_acc=0.558 r_n=127469
2026-05-11 14:19:50,332 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:19:50,332 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:19:50,332 INFO train_multi TF=ALL: new best val=2.2653 r_mae=0.9337 — saved
2026-05-11 14:19:50,337 INFO train_multi TF=ALL: new best r_mae=0.9337 — saved rmae checkpoint
2026-05-11 14:20:03,677 INFO train_multi TF=ALL epoch 26/100 train=2.2556 val=2.2599 r_mae=0.931 pos_r_acc=0.588 side_acc=0.561 r_n=127469
2026-05-11 14:20:03,683 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:20:03,683 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:20:03,683 INFO train_multi TF=ALL: new best val=2.2599 r_mae=0.9311 — saved
2026-05-11 14:20:03,687 INFO train_multi TF=ALL: new best r_mae=0.9311 — saved rmae checkpoint
2026-05-11 14:20:16,832 INFO train_multi TF=ALL epoch 27/100 train=2.2479 val=2.2685 r_mae=0.932 pos_r_acc=0.580 side_acc=0.553 r_n=127469
2026-05-11 14:20:30,072 INFO train_multi TF=ALL epoch 28/100 train=2.2454 val=2.2560 r_mae=0.929 pos_r_acc=0.587 side_acc=0.562 r_n=127469
2026-05-11 14:20:30,083 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:20:30,084 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:20:30,084 INFO train_multi TF=ALL: new best val=2.2560 r_mae=0.9286 — saved
2026-05-11 14:20:30,088 INFO train_multi TF=ALL: new best r_mae=0.9286 — saved rmae checkpoint
2026-05-11 14:20:43,461 INFO train_multi TF=ALL epoch 29/100 train=2.2352 val=2.2555 r_mae=0.927 pos_r_acc=0.587 side_acc=0.564 r_n=127469
2026-05-11 14:20:43,466 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:20:43,466 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:20:43,466 INFO train_multi TF=ALL: new best val=2.2555 r_mae=0.9268 — saved
2026-05-11 14:20:43,471 INFO train_multi TF=ALL: new best r_mae=0.9268 — saved rmae checkpoint
2026-05-11 14:20:56,910 INFO train_multi TF=ALL epoch 30/100 train=2.2303 val=2.2529 r_mae=0.924 pos_r_acc=0.589 side_acc=0.565 r_n=127469
2026-05-11 14:20:56,916 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:20:56,916 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:20:56,916 INFO train_multi TF=ALL: new best val=2.2529 r_mae=0.9237 — saved
2026-05-11 14:20:56,920 INFO train_multi TF=ALL: new best r_mae=0.9237 — saved rmae checkpoint
2026-05-11 14:21:10,082 INFO train_multi TF=ALL epoch 31/100 train=2.2281 val=2.2451 r_mae=0.921 pos_r_acc=0.592 side_acc=0.570 r_n=127469
2026-05-11 14:21:10,087 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:21:10,087 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:21:10,087 INFO train_multi TF=ALL: new best val=2.2451 r_mae=0.9209 — saved
2026-05-11 14:21:10,091 INFO train_multi TF=ALL: new best r_mae=0.9209 — saved rmae checkpoint
2026-05-11 14:21:23,176 INFO train_multi TF=ALL epoch 32/100 train=2.2187 val=2.2425 r_mae=0.921 pos_r_acc=0.593 side_acc=0.574 r_n=127469
2026-05-11 14:21:23,182 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:21:23,182 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:21:23,182 INFO train_multi TF=ALL: new best val=2.2425 r_mae=0.9211 — saved
2026-05-11 14:21:36,465 INFO train_multi TF=ALL epoch 33/100 train=2.2161 val=2.2445 r_mae=0.921 pos_r_acc=0.592 side_acc=0.574 r_n=127469
2026-05-11 14:21:49,521 INFO train_multi TF=ALL epoch 34/100 train=2.2107 val=2.2448 r_mae=0.922 pos_r_acc=0.594 side_acc=0.571 r_n=127469
2026-05-11 14:22:02,509 INFO train_multi TF=ALL epoch 35/100 train=2.2001 val=2.2377 r_mae=0.917 pos_r_acc=0.593 side_acc=0.580 r_n=127469
2026-05-11 14:22:02,514 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:22:02,515 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:22:02,515 INFO train_multi TF=ALL: new best val=2.2377 r_mae=0.9172 — saved
2026-05-11 14:22:02,519 INFO train_multi TF=ALL: new best r_mae=0.9172 — saved rmae checkpoint
2026-05-11 14:22:15,747 INFO train_multi TF=ALL epoch 36/100 train=2.1918 val=2.2266 r_mae=0.913 pos_r_acc=0.601 side_acc=0.587 r_n=127469
2026-05-11 14:22:15,752 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:22:15,752 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:22:15,752 INFO train_multi TF=ALL: new best val=2.2266 r_mae=0.9133 — saved
2026-05-11 14:22:15,757 INFO train_multi TF=ALL: new best r_mae=0.9133 — saved rmae checkpoint
2026-05-11 14:22:28,766 INFO train_multi TF=ALL epoch 37/100 train=2.1824 val=2.2212 r_mae=0.909 pos_r_acc=0.605 side_acc=0.590 r_n=127469
2026-05-11 14:22:28,772 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:22:28,772 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:22:28,772 INFO train_multi TF=ALL: new best val=2.2212 r_mae=0.9093 — saved
2026-05-11 14:22:28,776 INFO train_multi TF=ALL: new best r_mae=0.9093 — saved rmae checkpoint
2026-05-11 14:22:42,001 INFO train_multi TF=ALL epoch 38/100 train=2.1726 val=2.2104 r_mae=0.907 pos_r_acc=0.607 side_acc=0.595 r_n=127469
2026-05-11 14:22:42,006 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:22:42,006 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:22:42,006 INFO train_multi TF=ALL: new best val=2.2104 r_mae=0.9070 — saved
2026-05-11 14:22:42,010 INFO train_multi TF=ALL: new best r_mae=0.9070 — saved rmae checkpoint
2026-05-11 14:22:55,070 INFO train_multi TF=ALL epoch 39/100 train=2.1676 val=2.2010 r_mae=0.902 pos_r_acc=0.610 side_acc=0.600 r_n=127469
2026-05-11 14:22:55,082 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:22:55,083 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:22:55,083 INFO train_multi TF=ALL: new best val=2.2010 r_mae=0.9021 — saved
2026-05-11 14:22:55,087 INFO train_multi TF=ALL: new best r_mae=0.9021 — saved rmae checkpoint
2026-05-11 14:23:08,095 INFO train_multi TF=ALL epoch 40/100 train=2.1527 val=2.1903 r_mae=0.898 pos_r_acc=0.613 side_acc=0.605 r_n=127469
2026-05-11 14:23:08,102 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:23:08,102 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:23:08,102 INFO train_multi TF=ALL: new best val=2.1903 r_mae=0.8978 — saved
2026-05-11 14:23:08,106 INFO train_multi TF=ALL: new best r_mae=0.8978 — saved rmae checkpoint
2026-05-11 14:23:21,281 INFO train_multi TF=ALL epoch 41/100 train=2.1397 val=2.1870 r_mae=0.891 pos_r_acc=0.619 side_acc=0.605 r_n=127469
2026-05-11 14:23:21,286 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:23:21,286 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:23:21,286 INFO train_multi TF=ALL: new best val=2.1870 r_mae=0.8905 — saved
2026-05-11 14:23:21,290 INFO train_multi TF=ALL: new best r_mae=0.8905 — saved rmae checkpoint
2026-05-11 14:23:34,653 INFO train_multi TF=ALL epoch 42/100 train=2.1219 val=2.1661 r_mae=0.884 pos_r_acc=0.627 side_acc=0.614 r_n=127469
2026-05-11 14:23:34,658 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:23:34,658 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:23:34,658 INFO train_multi TF=ALL: new best val=2.1661 r_mae=0.8844 — saved
2026-05-11 14:23:34,663 INFO train_multi TF=ALL: new best r_mae=0.8844 — saved rmae checkpoint
2026-05-11 14:23:48,003 INFO train_multi TF=ALL epoch 43/100 train=2.1063 val=2.1569 r_mae=0.875 pos_r_acc=0.632 side_acc=0.614 r_n=127469
2026-05-11 14:23:48,009 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:23:48,009 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:23:48,009 INFO train_multi TF=ALL: new best val=2.1569 r_mae=0.8749 — saved
2026-05-11 14:23:48,014 INFO train_multi TF=ALL: new best r_mae=0.8749 — saved rmae checkpoint
2026-05-11 14:24:01,286 INFO train_multi TF=ALL epoch 44/100 train=2.0961 val=2.1342 r_mae=0.871 pos_r_acc=0.638 side_acc=0.623 r_n=127469
2026-05-11 14:24:01,291 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:24:01,291 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:24:01,291 INFO train_multi TF=ALL: new best val=2.1342 r_mae=0.8713 — saved
2026-05-11 14:24:01,296 INFO train_multi TF=ALL: new best r_mae=0.8713 — saved rmae checkpoint
2026-05-11 14:24:14,748 INFO train_multi TF=ALL epoch 45/100 train=2.0737 val=2.1325 r_mae=0.864 pos_r_acc=0.639 side_acc=0.622 r_n=127469
2026-05-11 14:24:14,753 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:24:14,753 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:24:14,754 INFO train_multi TF=ALL: new best val=2.1325 r_mae=0.8638 — saved
2026-05-11 14:24:14,758 INFO train_multi TF=ALL: new best r_mae=0.8638 — saved rmae checkpoint
2026-05-11 14:24:27,957 INFO train_multi TF=ALL epoch 46/100 train=2.0620 val=2.1168 r_mae=0.854 pos_r_acc=0.644 side_acc=0.629 r_n=127469
2026-05-11 14:24:27,963 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:24:27,963 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:24:27,963 INFO train_multi TF=ALL: new best val=2.1168 r_mae=0.8538 — saved
2026-05-11 14:24:27,967 INFO train_multi TF=ALL: new best r_mae=0.8538 — saved rmae checkpoint
2026-05-11 14:24:41,088 INFO train_multi TF=ALL epoch 47/100 train=2.0459 val=2.1049 r_mae=0.846 pos_r_acc=0.651 side_acc=0.633 r_n=127469
2026-05-11 14:24:41,093 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:24:41,093 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:24:41,093 INFO train_multi TF=ALL: new best val=2.1049 r_mae=0.8461 — saved
2026-05-11 14:24:41,098 INFO train_multi TF=ALL: new best r_mae=0.8461 — saved rmae checkpoint
2026-05-11 14:24:53,974 INFO train_multi TF=ALL epoch 48/100 train=2.0346 val=2.0954 r_mae=0.841 pos_r_acc=0.654 side_acc=0.633 r_n=127469
2026-05-11 14:24:53,979 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:24:53,980 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:24:53,980 INFO train_multi TF=ALL: new best val=2.0954 r_mae=0.8415 — saved
2026-05-11 14:24:53,984 INFO train_multi TF=ALL: new best r_mae=0.8415 — saved rmae checkpoint
2026-05-11 14:25:06,821 INFO train_multi TF=ALL epoch 49/100 train=2.0206 val=2.0889 r_mae=0.839 pos_r_acc=0.656 side_acc=0.636 r_n=127469
2026-05-11 14:25:06,827 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:25:06,827 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:25:06,827 INFO train_multi TF=ALL: new best val=2.0889 r_mae=0.8385 — saved
2026-05-11 14:25:06,831 INFO train_multi TF=ALL: new best r_mae=0.8385 — saved rmae checkpoint
2026-05-11 14:25:19,822 INFO train_multi TF=ALL epoch 50/100 train=2.0141 val=2.0971 r_mae=0.837 pos_r_acc=0.652 side_acc=0.634 r_n=127469
2026-05-11 14:25:19,827 INFO train_multi TF=ALL: new best r_mae=0.8368 — saved rmae checkpoint
2026-05-11 14:25:32,828 INFO train_multi TF=ALL epoch 51/100 train=2.0025 val=2.0829 r_mae=0.832 pos_r_acc=0.657 side_acc=0.640 r_n=127469
2026-05-11 14:25:32,834 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:25:32,834 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:25:32,834 INFO train_multi TF=ALL: new best val=2.0829 r_mae=0.8320 — saved
2026-05-11 14:25:32,838 INFO train_multi TF=ALL: new best r_mae=0.8320 — saved rmae checkpoint
2026-05-11 14:25:45,873 INFO train_multi TF=ALL epoch 52/100 train=1.9920 val=2.0898 r_mae=0.825 pos_r_acc=0.660 side_acc=0.635 r_n=127469
2026-05-11 14:25:45,877 INFO train_multi TF=ALL: new best r_mae=0.8250 — saved rmae checkpoint
2026-05-11 14:25:58,835 INFO train_multi TF=ALL epoch 53/100 train=1.9857 val=2.0765 r_mae=0.828 pos_r_acc=0.659 side_acc=0.640 r_n=127469
2026-05-11 14:25:58,840 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:25:58,841 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:25:58,841 INFO train_multi TF=ALL: new best val=2.0765 r_mae=0.8283 — saved
2026-05-11 14:26:11,787 INFO train_multi TF=ALL epoch 54/100 train=1.9744 val=2.0729 r_mae=0.826 pos_r_acc=0.659 side_acc=0.642 r_n=127469
2026-05-11 14:26:11,793 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:26:11,793 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:26:11,793 INFO train_multi TF=ALL: new best val=2.0729 r_mae=0.8257 — saved
2026-05-11 14:26:24,735 INFO train_multi TF=ALL epoch 55/100 train=1.9662 val=2.0716 r_mae=0.820 pos_r_acc=0.661 side_acc=0.647 r_n=127469
2026-05-11 14:26:24,741 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:26:24,741 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:26:24,741 INFO train_multi TF=ALL: new best val=2.0716 r_mae=0.8204 — saved
2026-05-11 14:26:24,745 INFO train_multi TF=ALL: new best r_mae=0.8204 — saved rmae checkpoint
2026-05-11 14:26:37,666 INFO train_multi TF=ALL epoch 56/100 train=1.9538 val=2.0812 r_mae=0.821 pos_r_acc=0.660 side_acc=0.639 r_n=127469
2026-05-11 14:26:50,668 INFO train_multi TF=ALL epoch 57/100 train=1.9529 val=2.0748 r_mae=0.815 pos_r_acc=0.663 side_acc=0.642 r_n=127469
2026-05-11 14:26:50,673 INFO train_multi TF=ALL: new best r_mae=0.8149 — saved rmae checkpoint
2026-05-11 14:27:03,519 INFO train_multi TF=ALL epoch 58/100 train=1.9418 val=2.0645 r_mae=0.816 pos_r_acc=0.663 side_acc=0.648 r_n=127469
2026-05-11 14:27:03,525 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:27:03,525 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:27:03,525 INFO train_multi TF=ALL: new best val=2.0645 r_mae=0.8163 — saved
2026-05-11 14:27:16,599 INFO train_multi TF=ALL epoch 59/100 train=1.9329 val=2.0787 r_mae=0.817 pos_r_acc=0.660 side_acc=0.645 r_n=127469
2026-05-11 14:27:29,559 INFO train_multi TF=ALL epoch 60/100 train=1.9321 val=2.0740 r_mae=0.817 pos_r_acc=0.659 side_acc=0.644 r_n=127469
2026-05-11 14:27:42,645 INFO train_multi TF=ALL epoch 61/100 train=1.9167 val=2.0690 r_mae=0.813 pos_r_acc=0.660 side_acc=0.648 r_n=127469
2026-05-11 14:27:42,656 INFO train_multi TF=ALL: new best r_mae=0.8131 — saved rmae checkpoint
2026-05-11 14:27:55,647 INFO train_multi TF=ALL epoch 62/100 train=1.9169 val=2.0593 r_mae=0.815 pos_r_acc=0.664 side_acc=0.649 r_n=127469
2026-05-11 14:27:55,653 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:27:55,653 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:27:55,653 INFO train_multi TF=ALL: new best val=2.0593 r_mae=0.8147 — saved
2026-05-11 14:28:08,783 INFO train_multi TF=ALL epoch 63/100 train=1.9065 val=2.0750 r_mae=0.818 pos_r_acc=0.660 side_acc=0.646 r_n=127469
2026-05-11 14:28:21,853 INFO train_multi TF=ALL epoch 64/100 train=1.9046 val=2.0766 r_mae=0.811 pos_r_acc=0.662 side_acc=0.647 r_n=127469
2026-05-11 14:28:21,858 INFO train_multi TF=ALL: new best r_mae=0.8111 — saved rmae checkpoint
2026-05-11 14:28:34,915 INFO train_multi TF=ALL epoch 65/100 train=1.8955 val=2.0939 r_mae=0.815 pos_r_acc=0.657 side_acc=0.643 r_n=127469
2026-05-11 14:28:47,961 INFO train_multi TF=ALL epoch 66/100 train=1.8859 val=2.0774 r_mae=0.809 pos_r_acc=0.661 side_acc=0.649 r_n=127469
2026-05-11 14:28:47,965 INFO train_multi TF=ALL: new best r_mae=0.8091 — saved rmae checkpoint
2026-05-11 14:29:00,964 INFO train_multi TF=ALL epoch 67/100 train=1.8824 val=2.0848 r_mae=0.806 pos_r_acc=0.660 side_acc=0.652 r_n=127469
2026-05-11 14:29:00,968 INFO train_multi TF=ALL: new best r_mae=0.8058 — saved rmae checkpoint
2026-05-11 14:29:14,347 INFO train_multi TF=ALL epoch 68/100 train=1.8721 val=2.0864 r_mae=0.811 pos_r_acc=0.659 side_acc=0.647 r_n=127469
2026-05-11 14:29:27,548 INFO train_multi TF=ALL epoch 69/100 train=1.8693 val=2.0812 r_mae=0.810 pos_r_acc=0.657 side_acc=0.651 r_n=127469
2026-05-11 14:29:40,628 INFO train_multi TF=ALL epoch 70/100 train=1.8539 val=2.0983 r_mae=0.812 pos_r_acc=0.657 side_acc=0.646 r_n=127469
2026-05-11 14:29:53,533 INFO train_multi TF=ALL epoch 71/100 train=1.8485 val=2.0774 r_mae=0.811 pos_r_acc=0.658 side_acc=0.652 r_n=127469
2026-05-11 14:30:06,902 INFO train_multi TF=ALL epoch 72/100 train=1.8423 val=2.1016 r_mae=0.814 pos_r_acc=0.655 side_acc=0.644 r_n=127469
2026-05-11 14:30:20,137 INFO train_multi TF=ALL epoch 73/100 train=1.8374 val=2.0869 r_mae=0.811 pos_r_acc=0.660 side_acc=0.648 r_n=127469
2026-05-11 14:30:33,196 INFO train_multi TF=ALL epoch 74/100 train=1.8309 val=2.0888 r_mae=0.806 pos_r_acc=0.659 side_acc=0.650 r_n=127469
2026-05-11 14:30:46,299 INFO train_multi TF=ALL epoch 75/100 train=1.8252 val=2.0738 r_mae=0.812 pos_r_acc=0.659 side_acc=0.654 r_n=127469
2026-05-11 14:30:59,382 INFO train_multi TF=ALL epoch 76/100 train=1.8135 val=2.0840 r_mae=0.810 pos_r_acc=0.658 side_acc=0.650 r_n=127469
2026-05-11 14:31:12,473 INFO train_multi TF=ALL epoch 77/100 train=1.8037 val=2.0973 r_mae=0.809 pos_r_acc=0.660 side_acc=0.647 r_n=127469
2026-05-11 14:31:25,495 INFO train_multi TF=ALL epoch 78/100 train=1.8002 val=2.0830 r_mae=0.812 pos_r_acc=0.659 side_acc=0.651 r_n=127469
2026-05-11 14:31:38,621 INFO train_multi TF=ALL epoch 79/100 train=1.7963 val=2.0812 r_mae=0.806 pos_r_acc=0.660 side_acc=0.652 r_n=127469
2026-05-11 14:31:51,599 INFO train_multi TF=ALL epoch 80/100 train=1.7883 val=2.0927 r_mae=0.812 pos_r_acc=0.658 side_acc=0.651 r_n=127469
2026-05-11 14:32:04,659 INFO train_multi TF=ALL epoch 81/100 train=1.7794 val=2.0905 r_mae=0.810 pos_r_acc=0.658 side_acc=0.654 r_n=127469
2026-05-11 14:32:17,671 INFO train_multi TF=ALL epoch 82/100 train=1.7755 val=2.1194 r_mae=0.813 pos_r_acc=0.655 side_acc=0.647 r_n=127469
2026-05-11 14:32:30,640 INFO train_multi TF=ALL epoch 83/100 train=1.7664 val=2.0906 r_mae=0.807 pos_r_acc=0.660 side_acc=0.653 r_n=127469
2026-05-11 14:32:43,889 INFO train_multi TF=ALL epoch 84/100 train=1.7598 val=2.1090 r_mae=0.807 pos_r_acc=0.658 side_acc=0.654 r_n=127469
2026-05-11 14:32:56,829 INFO train_multi TF=ALL epoch 85/100 train=1.7532 val=2.1256 r_mae=0.812 pos_r_acc=0.654 side_acc=0.655 r_n=127469
2026-05-11 14:33:09,836 INFO train_multi TF=ALL epoch 86/100 train=1.7484 val=2.0971 r_mae=0.809 pos_r_acc=0.658 side_acc=0.655 r_n=127469
2026-05-11 14:33:23,114 INFO train_multi TF=ALL epoch 87/100 train=1.7410 val=2.1075 r_mae=0.811 pos_r_acc=0.657 side_acc=0.655 r_n=127469
2026-05-11 14:33:36,206 INFO train_multi TF=ALL epoch 88/100 train=1.7340 val=2.1125 r_mae=0.811 pos_r_acc=0.656 side_acc=0.654 r_n=127469
2026-05-11 14:33:49,210 INFO train_multi TF=ALL epoch 89/100 train=1.7316 val=2.1054 r_mae=0.813 pos_r_acc=0.656 side_acc=0.658 r_n=127469
2026-05-11 14:34:02,189 INFO train_multi TF=ALL epoch 90/100 train=1.7145 val=2.1108 r_mae=0.813 pos_r_acc=0.654 side_acc=0.659 r_n=127469
2026-05-11 14:34:15,196 INFO train_multi TF=ALL epoch 91/100 train=1.7150 val=2.1146 r_mae=0.815 pos_r_acc=0.654 side_acc=0.660 r_n=127469
2026-05-11 14:34:28,327 INFO train_multi TF=ALL epoch 92/100 train=1.7092 val=2.1253 r_mae=0.813 pos_r_acc=0.654 side_acc=0.660 r_n=127469
2026-05-11 14:34:28,327 INFO train_multi TF=ALL early stop at epoch 92
2026-05-11 14:34:28,344 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:34:28,344 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:34:28,344 INFO train_multi TF=ALL: promoted r_mae checkpoint (0.8058 < primary 0.8147) — overwriting model.pt
2026-05-11 14:34:29,481 WARNING fit_r_isotonic: long calibrator increases MAE (cal=0.8359 >= raw=0.8232) — skipping
2026-05-11 14:34:29,489 WARNING fit_r_isotonic: short calibrator increases MAE (cal=0.8419 >= raw=0.8285) — skipping
2026-05-11 14:34:29,489 INFO fit_r_isotonic: saved side-R calibrators stats={'long': {'n': 31689, 'raw_mae': 0.8231648828534721, 'calibrated_mae': 0.8359316565088101, 'skipped': 'calibrator_hurts'}, 'short': {'n': 32408, 'raw_mae': 0.8285213258856119, 'calibrated_mae': 0.8419481160688744, 'skipped': 'calibrator_hurts'}}
2026-05-11 14:34:29,612 WARNING GRU validation R-MAE above floor for TF=ALL: best_val_r_mae=0.806 max=0.750. Keeping saved best weights so the pipeline can progress.
2026-05-11 14:34:29,625 INFO GRU R threshold XAUUSD/buy: q25_pos=0.535 q50_pos=1.667 pos_rate=46.0% (n=119259 n_pos=54899)
2026-05-11 14:34:29,631 INFO GRU R threshold XAUUSD/sell: q25_pos=0.528 q50_pos=1.667 pos_rate=45.5% (n=118087 n_pos=53734)
2026-05-11 14:34:29,636 INFO GRU R threshold EURUSD/buy: q25_pos=0.541 q50_pos=1.667 pos_rate=45.6% (n=118751 n_pos=54196)
2026-05-11 14:34:29,642 INFO GRU R threshold EURUSD/sell: q25_pos=0.545 q50_pos=1.667 pos_rate=45.6% (n=117679 n_pos=53721)
2026-05-11 14:34:29,648 INFO GRU R threshold USDJPY/buy: q25_pos=0.551 q50_pos=1.667 pos_rate=46.2% (n=118691 n_pos=54870)
2026-05-11 14:34:29,653 INFO GRU R threshold USDJPY/sell: q25_pos=0.542 q50_pos=1.667 pos_rate=44.6% (n=116030 n_pos=51793)
2026-05-11 14:34:29,659 INFO GRU R threshold EURJPY/buy: q25_pos=0.530 q50_pos=1.416 pos_rate=46.1% (n=118651 n_pos=54654)
2026-05-11 14:34:29,664 INFO GRU R threshold EURJPY/sell: q25_pos=0.525 q50_pos=1.667 pos_rate=44.7% (n=117320 n_pos=52399)
2026-05-11 14:34:29,670 INFO GRU R threshold GBPJPY/buy: q25_pos=0.517 q50_pos=1.443 pos_rate=45.9% (n=118277 n_pos=54278)
2026-05-11 14:34:29,675 INFO GRU R threshold GBPJPY/sell: q25_pos=0.521 q50_pos=1.667 pos_rate=45.3% (n=116299 n_pos=52702)
2026-05-11 14:34:29,681 INFO GRU R threshold GBPUSD/buy: q25_pos=0.524 q50_pos=1.667 pos_rate=45.8% (n=117764 n_pos=53894)
2026-05-11 14:34:29,686 INFO GRU R threshold GBPUSD/sell: q25_pos=0.533 q50_pos=1.667 pos_rate=45.8% (n=117404 n_pos=53785)
2026-05-11 14:34:29,687 INFO GRU per-symbol R thresholds saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/symbol_r_thresholds.json (6 symbols)
2026-05-11 14:34:29,704 INFO Retrain complete. Total wall-clock: 1241.1s
  DONE  Retrain gru [pre-R2 retrain]
  START Retrain regime [pre-R2 retrain]
2026-05-11 14:34:33,071 INFO retrain environment: KAGGLE
2026-05-11 14:34:34,664 INFO Device: CUDA (2 GPU(s))
2026-05-11 14:34:34,673 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 14:34:34,673 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 14:34:34,673 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 14:34:34,673 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 14:34:34,673 INFO Retrain data split: train
2026-05-11 14:34:34,674 INFO Retrain rolling fold selector: latest
2026-05-11 14:34:34,675 INFO === RegimeClassifier retrain (hierarchical: HTF 2-score bias + LTF 5-score behaviour) ===
2026-05-11 14:34:34,829 INFO NumExpr defaulting to 4 threads.
2026-05-11 14:34:35,018 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-11 14:34:35,019 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 14:34:35,019 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 14:34:35,019 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-11 14:34:35,072 INFO Regime rolling folds selected: [None]
2026-05-11 14:34:35,072 INFO === Regime rolling fold 1/1: train_all ===
2026-05-11 14:34:35,072 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-11 14:34:35,112 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 14:34:35,113 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:34:35,129 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:34:35,143 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:34:35,158 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:34:35,174 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:34:35,189 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:34:35,420 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:34:35,489 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:34:35,512 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:34:35,513 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:34:35,523 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:34:35,524 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:34:36,250 INFO macro_cache: using training data dir /kaggle/working/Multi-Bot/trading-system/training_data
2026-05-11 14:34:36,355 INFO macro_cache loaded 19 series: {'asx200': '2842 bars 2015-01-02→2026-03-27', 'cac40': '2876 bars 2015-01-02→2026-03-27', 'dax': '2851 bars 2015-01-02→2026-03-27', 'djia': '2825 bars 2015-01-02→2026-03-27', 'dxy': '2826 bars 2015-01-02→2026-03-27', 'eurostoxx': '2823 bars 2015-01-05→2026-03-27', 'ftse': '2839 bars 2015-01-02→2026-03-27', 'gold_fut': '2824 bars 2015-01-02→2026-03-27', 'hsi': '2764 bars 2015-01-02→2026-03-27', 'nasdaq': '2825 bars 2015-01-02→2026-03-27', 'nikkei': '2744 bars 2015-01-05→2026-03-27', 'oil_fut': '2825 bars 2015-01-02→2026-03-27', 'spx': '2825 bars 2015-01-02→2026-03-27', 'us10y': '2824 bars 2015-01-02→2026-03-27', 'us30y': '2824 bars 2015-01-02→2026-03-27', 'us3m': '2824 bars 2015-01-02→2026-03-27', 'vix': '2825 bars 2015-01-02→2026-03-27', 'us10y_fred': '2607 bars 2016-03-28→2026-03-24', 'us2y_fred': '2607 bars 2016-03-28→2026-03-24'}
2026-05-11 14:34:36,590 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1142, 'BIAS_DOWN': 522, 'BIAS_NEUTRAL': 10438}  ambiguous=4182 (total=12102) horizon=84
2026-05-11 14:34:36,595 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected XAUUSD — 12052 samples (group=gold) score_means={'bias_up_score': 0.0948, 'bias_down_score': 0.0433} labels={'BIAS_UP': 1142, 'BIAS_DOWN': 522, 'BIAS_NEUTRAL': 10388} clean={'BIAS_UP': 1142, 'BIAS_DOWN': 522, 'BIAS_NEUTRAL': 6216}
2026-05-11 14:34:36,785 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:36,825 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:36,847 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:36,847 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:36,858 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:36,859 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:34:37,795 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 690, 'BIAS_DOWN': 540, 'BIAS_NEUTRAL': 10174}  ambiguous=3886 (total=11404) horizon=84
2026-05-11 14:34:37,800 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURUSD — 11354 samples (group=dollar) score_means={'bias_up_score': 0.0608, 'bias_down_score': 0.0476} labels={'BIAS_UP': 690, 'BIAS_DOWN': 540, 'BIAS_NEUTRAL': 10124} clean={'BIAS_UP': 690, 'BIAS_DOWN': 540, 'BIAS_NEUTRAL': 6257}
2026-05-11 14:34:37,954 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:37,989 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:38,008 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:38,009 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:38,018 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:38,019 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:34:38,953 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 826, 'BIAS_DOWN': 423, 'BIAS_NEUTRAL': 10154}  ambiguous=4036 (total=11403) horizon=84
2026-05-11 14:34:38,959 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected USDJPY — 11353 samples (group=dollar) score_means={'bias_up_score': 0.0728, 'bias_down_score': 0.0373} labels={'BIAS_UP': 826, 'BIAS_DOWN': 423, 'BIAS_NEUTRAL': 10104} clean={'BIAS_UP': 826, 'BIAS_DOWN': 423, 'BIAS_NEUTRAL': 6078}
2026-05-11 14:34:39,107 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:39,143 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:39,165 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:39,166 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:39,173 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:39,175 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:34:40,056 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 681, 'BIAS_DOWN': 527, 'BIAS_NEUTRAL': 10199}  ambiguous=4044 (total=11407) horizon=84
2026-05-11 14:34:40,061 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURJPY — 11357 samples (group=cross) score_means={'bias_up_score': 0.06, 'bias_down_score': 0.0464} labels={'BIAS_UP': 681, 'BIAS_DOWN': 527, 'BIAS_NEUTRAL': 10149} clean={'BIAS_UP': 681, 'BIAS_DOWN': 527, 'BIAS_NEUTRAL': 6111}
2026-05-11 14:34:40,216 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:40,251 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:40,271 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:40,271 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:40,280 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:40,281 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:34:41,157 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 839, 'BIAS_DOWN': 579, 'BIAS_NEUTRAL': 9990}  ambiguous=4240 (total=11408) horizon=84
2026-05-11 14:34:41,162 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPJPY — 11358 samples (group=cross) score_means={'bias_up_score': 0.0739, 'bias_down_score': 0.051} labels={'BIAS_UP': 839, 'BIAS_DOWN': 579, 'BIAS_NEUTRAL': 9940} clean={'BIAS_UP': 839, 'BIAS_DOWN': 579, 'BIAS_NEUTRAL': 5723}
2026-05-11 14:34:41,314 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:41,346 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:41,366 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:41,366 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:41,376 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:41,377 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:34:42,274 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 502, 'BIAS_DOWN': 720, 'BIAS_NEUTRAL': 10180}  ambiguous=4113 (total=11402) horizon=84
2026-05-11 14:34:42,279 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPUSD — 11352 samples (group=dollar) score_means={'bias_up_score': 0.0442, 'bias_down_score': 0.0623} labels={'BIAS_UP': 502, 'BIAS_DOWN': 707, 'BIAS_NEUTRAL': 10143} clean={'BIAS_UP': 502, 'BIAS_DOWN': 707, 'BIAS_NEUTRAL': 6056}
2026-05-11 14:34:42,343 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 1520, 'BIAS_DOWN': 1106, 'BIAS_NEUTRAL': 20089}, 'dollar': {'BIAS_UP': 2018, 'BIAS_DOWN': 1670, 'BIAS_NEUTRAL': 30371}, 'gold': {'BIAS_UP': 1142, 'BIAS_DOWN': 522, 'BIAS_NEUTRAL': 10388}}
2026-05-11 14:34:42,344 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0669, 'bias_down_score': 0.0487}, 'dollar': {'bias_up_score': 0.0593, 'bias_down_score': 0.049}, 'gold': {'bias_up_score': 0.0948, 'bias_down_score': 0.0433}}
2026-05-11 14:34:42,344 INFO Regime[4H mode=htf_bias] label distribution by year: {2016: {'BIAS_UP': 491, 'BIAS_DOWN': 576, 'BIAS_NEUTRAL': 7755}, 2017: {'BIAS_UP': 734, 'BIAS_DOWN': 286, 'BIAS_NEUTRAL': 8093}, 2018: {'BIAS_UP': 427, 'BIAS_DOWN': 714, 'BIAS_NEUTRAL': 7989}, 2019: {'BIAS_UP': 410, 'BIAS_DOWN': 447, 'BIAS_NEUTRAL': 8245}, 2020: {'BIAS_UP': 694, 'BIAS_DOWN': 174, 'BIAS_NEUTRAL': 8243}, 2021: {'BIAS_UP': 722, 'BIAS_DOWN': 473, 'BIAS_NEUTRAL': 7896}, 2022: {'BIAS_UP': 667, 'BIAS_DOWN': 519, 'BIAS_NEUTRAL': 7935}, 2023: {'BIAS_UP': 535, 'BIAS_DOWN': 109, 'BIAS_NEUTRAL': 4692}}
2026-05-11 14:34:42,344 INFO Regime[4H mode=htf_bias] score means by year: {2016: {'bias_up_score': 0.0557, 'bias_down_score': 0.0653}, 2017: {'bias_up_score': 0.0805, 'bias_down_score': 0.0314}, 2018: {'bias_up_score': 0.0468, 'bias_down_score': 0.0782}, 2019: {'bias_up_score': 0.045, 'bias_down_score': 0.0491}, 2020: {'bias_up_score': 0.0762, 'bias_down_score': 0.0191}, 2021: {'bias_up_score': 0.0794, 'bias_down_score': 0.052}, 2022: {'bias_up_score': 0.0731, 'bias_down_score': 0.0569}, 2023: {'bias_up_score': 0.1003, 'bias_down_score': 0.0204}}
2026-05-11 14:34:42,389 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
2026-05-11 14:34:42,390 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
2026-05-11 14:34:42,391 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
2026-05-11 14:34:42,392 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 14:34:42,392 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 14:34:42,393 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
2026-05-11 14:34:42,410 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:34:42,414 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:34:42,415 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:34:42,416 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:34:42,416 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:34:42,417 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:34:42,934 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 157, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1339}  ambiguous=566 (total=1581) horizon=84
2026-05-11 14:34:42,936 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected XAUUSD — 1531 samples (group=gold) score_means={'bias_up_score': 0.1025, 'bias_down_score': 0.0555} labels={'BIAS_UP': 157, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1289} clean={'BIAS_UP': 157, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 744}
2026-05-11 14:34:43,008 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:43,011 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:43,011 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:43,012 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:43,012 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:43,014 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:34:43,506 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 135, 'BIAS_DOWN': 66, 'BIAS_NEUTRAL': 1290}  ambiguous=531 (total=1491) horizon=84
2026-05-11 14:34:43,509 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURUSD — 1441 samples (group=dollar) score_means={'bias_up_score': 0.0937, 'bias_down_score': 0.0458} labels={'BIAS_UP': 135, 'BIAS_DOWN': 66, 'BIAS_NEUTRAL': 1240} clean={'BIAS_UP': 135, 'BIAS_DOWN': 66, 'BIAS_NEUTRAL': 739}
2026-05-11 14:34:43,575 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:43,578 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:43,578 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:43,579 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:43,579 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:43,580 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:34:44,080 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 164, 'BIAS_DOWN': 77, 'BIAS_NEUTRAL': 1248}  ambiguous=616 (total=1489) horizon=84
2026-05-11 14:34:44,083 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected USDJPY — 1439 samples (group=dollar) score_means={'bias_up_score': 0.114, 'bias_down_score': 0.0535} labels={'BIAS_UP': 164, 'BIAS_DOWN': 77, 'BIAS_NEUTRAL': 1198} clean={'BIAS_UP': 164, 'BIAS_DOWN': 77, 'BIAS_NEUTRAL': 608}
2026-05-11 14:34:44,149 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:44,152 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:44,153 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:44,153 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:44,153 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:44,154 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:34:44,760 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 123, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 1366}  ambiguous=582 (total=1494) horizon=84
2026-05-11 14:34:44,763 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0852, 'bias_down_score': 0.0035} labels={'BIAS_UP': 123, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 1316} clean={'BIAS_UP': 123, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 741}
2026-05-11 14:34:44,836 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:44,838 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:44,839 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:44,840 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:44,840 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:44,841 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:34:45,426 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 129, 'BIAS_DOWN': 9, 'BIAS_NEUTRAL': 1356}  ambiguous=551 (total=1494) horizon=84
2026-05-11 14:34:45,429 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0893, 'bias_down_score': 0.0055} labels={'BIAS_UP': 129, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 1307} clean={'BIAS_UP': 129, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 775}
2026-05-11 14:34:45,501 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:45,504 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:45,505 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:45,505 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:45,505 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:34:45,506 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:34:46,038 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 81, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 1316}  ambiguous=560 (total=1488) horizon=84
2026-05-11 14:34:46,041 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPUSD — 1438 samples (group=dollar) score_means={'bias_up_score': 0.0563, 'bias_down_score': 0.0633} labels={'BIAS_UP': 81, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 1266} clean={'BIAS_UP': 81, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 735}
2026-05-11 14:34:46,103 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 252, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 2623}, 'dollar': {'BIAS_UP': 380, 'BIAS_DOWN': 234, 'BIAS_NEUTRAL': 3704}, 'gold': {'BIAS_UP': 157, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1289}}
2026-05-11 14:34:46,104 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0873, 'bias_down_score': 0.0045}, 'dollar': {'bias_up_score': 0.088, 'bias_down_score': 0.0542}, 'gold': {'bias_up_score': 0.1025, 'bias_down_score': 0.0555}}
2026-05-11 14:34:46,104 INFO Regime[4H mode=htf_bias] label distribution by year: {2022: {'BIAS_UP': 258, 'BIAS_DOWN': 228, 'BIAS_NEUTRAL': 2915}, 2023: {'BIAS_UP': 531, 'BIAS_DOWN': 104, 'BIAS_NEUTRAL': 4701}}
2026-05-11 14:34:46,104 INFO Regime[4H mode=htf_bias] score means by year: {2022: {'bias_up_score': 0.0759, 'bias_down_score': 0.067}, 2023: {'bias_up_score': 0.0995, 'bias_down_score': 0.0195}}
2026-05-11 14:34:46,146 INFO Regime phase HTF dataset build fold=train_all: 11.1s (train=68826 val=8737)
2026-05-11 14:34:46,147 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260511_143446
2026-05-11 14:34:46,345 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=51, n_classes=2)
2026-05-11 14:34:46,345 INFO Regime 4H/htf_bias warm start enabled from existing weights
2026-05-11 14:34:46,358 INFO RegimeClassifier[mode=htf_bias]: HTF clean-label fit filter kept train=44419/68826 val=5463/8737 at conf>=0.40 train_counts={'BIAS_UP': 4680, 'BIAS_DOWN': 3298, 'BIAS_NEUTRAL': 36441} val_counts={'BIAS_UP': 789, 'BIAS_DOWN': 332, 'BIAS_NEUTRAL': 4342}
2026-05-11 14:34:46,359 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=44419 val=5463 train_labels={'BIAS_UP': 4680, 'BIAS_DOWN': 3298, 'BIAS_NEUTRAL': 36441} val_labels={'BIAS_UP': 789, 'BIAS_DOWN': 332, 'BIAS_NEUTRAL': 4342}
2026-05-11 14:34:46,360 INFO RegimeClassifier[mode=htf_bias]: warm start HTF score head
2026-05-11 14:34:46,360 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-11 14:34:46,360 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 8.491, 'bias_down_score': 12.0}
2026-05-11 14:34:46,364 INFO RegimeClassifier[mode=htf_bias]: HTF balanced sampler — dir=7978 neutral=36441 dir_weight=3 => dir_frac_per_epoch≈47.2%
2026-05-11 14:34:49,813 INFO Regime HTF score epoch  1/50 — tr=4.3759 va=1.6321 acc=0.801 bal=0.353 threshold=0.35 margin=0.40 recall={'BIAS_UP': 0.033, 'BIAS_DOWN': 0.027, 'BIAS_NEUTRAL': 1.0} precision={'BIAS_UP': 0.963, 'BIAS_DOWN': 0.9, 'BIAS_NEUTRAL': 0.8}
2026-05-11 14:34:51,076 INFO Regime HTF score epoch  2/50 — tr=4.3558 va=1.6271 bal=0.354
2026-05-11 14:34:52,334 INFO Regime HTF score epoch  3/50 — tr=4.2729 va=1.6055 bal=0.357
2026-05-11 14:34:53,598 INFO Regime HTF score epoch  4/50 — tr=4.2158 va=1.5546 bal=0.441
2026-05-11 14:34:54,864 INFO Regime HTF score epoch  5/50 — tr=4.0407 va=1.4908 acc=0.803 bal=0.433 threshold=0.35 margin=0.35 recall={'BIAS_UP': 0.127, 'BIAS_DOWN': 0.199, 'BIAS_NEUTRAL': 0.973} precision={'BIAS_UP': 0.629, 'BIAS_DOWN': 0.524, 'BIAS_NEUTRAL': 0.816}
2026-05-11 14:34:56,133 INFO Regime HTF score epoch  6/50 — tr=3.9127 va=1.4166 bal=0.429
2026-05-11 14:34:57,392 INFO Regime HTF score epoch  7/50 — tr=3.7157 va=1.3294 bal=0.435
2026-05-11 14:34:58,693 INFO Regime HTF score epoch  8/50 — tr=3.4995 va=1.2496 bal=0.355
2026-05-11 14:34:59,956 INFO Regime HTF score epoch  9/50 — tr=3.3221 va=1.1824 bal=0.353
2026-05-11 14:35:01,226 INFO Regime HTF score epoch 10/50 — tr=3.1131 va=1.1173 acc=0.802 bal=0.357 threshold=0.35 margin=0.75 recall={'BIAS_UP': 0.043, 'BIAS_DOWN': 0.03, 'BIAS_NEUTRAL': 0.999} precision={'BIAS_UP': 0.919, 'BIAS_DOWN': 0.833, 'BIAS_NEUTRAL': 0.801}
2026-05-11 14:35:02,487 INFO Regime HTF score epoch 11/50 — tr=2.9670 va=1.0646 bal=0.418
2026-05-11 14:35:03,747 INFO Regime HTF score epoch 12/50 — tr=2.7730 va=1.0162 bal=0.358
2026-05-11 14:35:05,012 INFO Regime HTF score epoch 13/50 — tr=2.6979 va=0.9800 bal=0.364
2026-05-11 14:35:06,287 INFO Regime HTF score epoch 14/50 — tr=2.5489 va=0.9417 bal=0.355
2026-05-11 14:35:07,559 INFO Regime HTF score epoch 15/50 — tr=2.4681 va=0.9076 acc=0.802 bal=0.357 threshold=0.35 margin=0.85 recall={'BIAS_UP': 0.039, 'BIAS_DOWN': 0.033, 'BIAS_NEUTRAL': 0.999} precision={'BIAS_UP': 1.0, 'BIAS_DOWN': 0.733, 'BIAS_NEUTRAL': 0.801}
2026-05-11 14:35:08,863 INFO Regime HTF score epoch 16/50 — tr=2.3389 va=0.8762 bal=0.353
2026-05-11 14:35:10,127 INFO Regime HTF score epoch 17/50 — tr=2.2640 va=0.8471 bal=0.358
2026-05-11 14:35:11,384 INFO Regime HTF score epoch 18/50 — tr=2.1970 va=0.8237 bal=0.363
2026-05-11 14:35:12,649 INFO Regime HTF score epoch 19/50 — tr=2.1079 va=0.7999 bal=0.353
2026-05-11 14:35:13,910 INFO Regime HTF score epoch 20/50 — tr=2.0317 va=0.7786 acc=0.801 bal=0.354 threshold=0.35 margin=0.90 recall={'BIAS_UP': 0.034, 'BIAS_DOWN': 0.027, 'BIAS_NEUTRAL': 0.999} precision={'BIAS_UP': 1.0, 'BIAS_DOWN': 0.75, 'BIAS_NEUTRAL': 0.8}
2026-05-11 14:35:13,910 INFO Regime HTF score early stop at epoch 20
2026-05-11 14:35:15,083 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.350 margin=0.800 precision={'BIAS_UP': 0.941, 'BIAS_DOWN': 0.917, 'BIAS_NEUTRAL': 0.801} recall={'BIAS_UP': 0.041, 'BIAS_DOWN': 0.033, 'BIAS_NEUTRAL': 0.999} f1={'BIAS_UP': 0.078, 'BIAS_DOWN': 0.064, 'BIAS_NEUTRAL': 0.889} confusion=[[32, 0, 757], [0, 11, 321], [2, 1, 4339]] score_mae={'bias_up_score': 0.2051, 'bias_down_score': 0.1333} pred_share={'BIAS_UP': 0.0062, 'BIAS_DOWN': 0.0022, 'BIAS_NEUTRAL': 0.9916}
2026-05-11 14:35:15,084 WARNING Regime HTF score prediction distribution collapsed: pred_share={'BIAS_UP': 0.0062, 'BIAS_DOWN': 0.0022, 'BIAS_NEUTRAL': 0.9916}, max_pred_share=99.2%, collapsed_classes=[]. Saving weights anyway so the pipeline can progress.
2026-05-11 14:35:15,085 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.941, 'BIAS_DOWN': 0.917, 'BIAS_NEUTRAL': 0.801} min_precision=0.500 recall={'BIAS_UP': 0.041, 'BIAS_DOWN': 0.033, 'BIAS_NEUTRAL': 0.999} min_recall=0.100 f1={'BIAS_UP': 0.078, 'BIAS_DOWN': 0.064, 'BIAS_NEUTRAL': 0.889} min_f1=0.150 min_neutral_recall=0.500 weak_precision=[] weak_recall=['BIAS_UP', 'BIAS_DOWN'] weak_f1=['BIAS_UP', 'BIAS_DOWN'] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-11 14:35:15,088 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 14:35:15,088 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 14:35:15,088 INFO Regime phase HTF train fold=train_all: 28.7s
2026-05-11 14:35:15,195 INFO Regime HTF complete fold=train_all: acc=0.802 bal=0.358 train=68826 val=8737 per_class={'BIAS_UP': 0.041, 'BIAS_DOWN': 0.033, 'BIAS_NEUTRAL': 0.999} precision={'BIAS_UP': 0.941, 'BIAS_DOWN': 0.917, 'BIAS_NEUTRAL': 0.801} threshold=0.350 margin=0.800
2026-05-11 14:35:15,197 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:35:15,379 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 502, 'BIAS_DOWN': 720, 'BIAS_NEUTRAL': 10180}  ambiguous=4113 (total=11402) horizon=84
2026-05-11 14:35:15,382 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on GBPUSD 4H:
{'BIAS_UP': 4.482142857142857, 'BIAS_DOWN': 5.669291338582677, 'BIAS_NEUTRAL': 42.416666666666664}
2026-05-11 14:35:15,385 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (all labels):
{'BIAS_UP': {'n': 502, 'mean': 0.00043281055222645396, 'mean_over_std': 0.19321420748230153}, 'BIAS_DOWN': {'n': 720, 'mean': -0.00046308823082979826, 'mean_over_std': -0.19538165843530478}, 'BIAS_NEUTRAL': {'n': 10179, 'mean': 7.477567618138561e-07, 'mean_over_std': 0.0002829536380249001}}
2026-05-11 14:35:15,386 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 502, 'mean': 0.00043281055222645396, 'mean_over_std': 0.19321420748230153}, 'BIAS_DOWN': {'n': 720, 'mean': -0.00046308823082979826, 'mean_over_std': -0.19538165843530478}, 'BIAS_NEUTRAL': {'n': 6067, 'mean': 9.596616495197703e-06, 'mean_over_std': 0.004013656697571348}}
2026-05-11 14:35:15,390 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-11 14:35:15,392 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:35:15,394 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:35:15,396 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:35:15,397 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:35:15,399 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:35:15,401 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:35:15,417 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:35:15,424 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:35:15,426 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:35:15,427 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:35:15,427 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:35:15,433 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:35:16,472 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected XAUUSD — 44227 samples (group=gold) score_means={'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}
2026-05-11 14:35:16,579 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:16,581 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:16,582 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:16,582 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:16,582 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:16,585 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:35:17,521 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURUSD — 43689 samples (group=dollar) score_means={'trend_score': 0.49, 'range_score': 0.236, 'chop_score': 0.4622, 'volatility_percentile': 0.3812, 'consolidation_score': 0.1862}
2026-05-11 14:35:17,628 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:17,630 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:17,631 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:17,632 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:17,632 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:17,634 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:35:18,609 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected USDJPY — 43686 samples (group=dollar) score_means={'trend_score': 0.4949, 'range_score': 0.2312, 'chop_score': 0.4613, 'volatility_percentile': 0.38, 'consolidation_score': 0.1931}
2026-05-11 14:35:18,714 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:18,716 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:18,717 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:18,718 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:18,718 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:18,720 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:35:19,663 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURJPY — 43688 samples (group=cross) score_means={'trend_score': 0.4888, 'range_score': 0.2345, 'chop_score': 0.4662, 'volatility_percentile': 0.3831, 'consolidation_score': 0.1882}
2026-05-11 14:35:19,771 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:19,773 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:19,774 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:19,774 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:19,774 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:19,777 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:35:20,715 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPJPY — 43678 samples (group=cross) score_means={'trend_score': 0.4878, 'range_score': 0.2348, 'chop_score': 0.4673, 'volatility_percentile': 0.3839, 'consolidation_score': 0.1904}
2026-05-11 14:35:20,822 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:20,825 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:20,826 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:20,826 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:20,826 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:20,829 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:35:21,772 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPUSD — 43676 samples (group=dollar) score_means={'trend_score': 0.4923, 'range_score': 0.2335, 'chop_score': 0.4611, 'volatility_percentile': 0.3832, 'consolidation_score': 0.1856}
2026-05-11 14:35:21,895 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2346, 'chop_score': 0.4667, 'volatility_percentile': 0.3835, 'consolidation_score': 0.1893}, 'dollar': {'trend_score': 0.4924, 'range_score': 0.2335, 'chop_score': 0.4615, 'volatility_percentile': 0.3815, 'consolidation_score': 0.1883}, 'gold': {'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}}
2026-05-11 14:35:21,895 INFO Regime[1H mode=ltf_behaviour] score means by year: {2016: {'trend_score': 0.495, 'range_score': 0.233, 'chop_score': 0.4607, 'volatility_percentile': 0.3776, 'consolidation_score': 0.192}, 2017: {'trend_score': 0.5017, 'range_score': 0.2328, 'chop_score': 0.4574, 'volatility_percentile': 0.3689, 'consolidation_score': 0.1942}, 2018: {'trend_score': 0.4886, 'range_score': 0.237, 'chop_score': 0.4655, 'volatility_percentile': 0.3947, 'consolidation_score': 0.1785}, 2019: {'trend_score': 0.4817, 'range_score': 0.236, 'chop_score': 0.4691, 'volatility_percentile': 0.3725, 'consolidation_score': 0.198}, 2020: {'trend_score': 0.4874, 'range_score': 0.2311, 'chop_score': 0.4674, 'volatility_percentile': 0.3848, 'consolidation_score': 0.1919}, 2021: {'trend_score': 0.4872, 'range_score': 0.2372, 'chop_score': 0.4672, 'volatility_percentile': 0.377, 'consolidation_score': 0.192}, 2022: {'trend_score': 0.4893, 'range_score': 0.2357, 'chop_score': 0.4647, 'volatility_percentile': 0.3962, 'consolidation_score': 0.1784}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 14:35:21,984 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
2026-05-11 14:35:21,986 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 14:35:21,987 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
2026-05-11 14:35:21,988 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 14:35:21,989 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
2026-05-11 14:35:21,990 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
2026-05-11 14:35:22,000 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:35:22,003 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:35:22,004 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:35:22,004 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:35:22,005 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:35:22,006 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:35:22,316 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected XAUUSD — 5117 samples (group=gold) score_means={'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}
2026-05-11 14:35:22,422 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:22,425 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:22,425 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:22,426 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:22,426 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:22,428 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:35:22,717 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURUSD — 5053 samples (group=dollar) score_means={'trend_score': 0.4883, 'range_score': 0.2413, 'chop_score': 0.463, 'volatility_percentile': 0.3837, 'consolidation_score': 0.1793}
2026-05-11 14:35:22,826 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:22,828 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:22,829 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:22,829 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:22,830 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:22,831 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:35:23,120 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected USDJPY — 5046 samples (group=dollar) score_means={'trend_score': 0.5085, 'range_score': 0.2259, 'chop_score': 0.4506, 'volatility_percentile': 0.3809, 'consolidation_score': 0.1888}
2026-05-11 14:35:23,227 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:23,230 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:23,231 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:23,231 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:23,231 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:23,233 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:35:23,523 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURJPY — 5053 samples (group=cross) score_means={'trend_score': 0.5023, 'range_score': 0.2261, 'chop_score': 0.4524, 'volatility_percentile': 0.3837, 'consolidation_score': 0.18}
2026-05-11 14:35:23,631 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:23,633 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:23,634 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:23,634 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:23,634 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:23,636 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:35:23,924 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPJPY — 5043 samples (group=cross) score_means={'trend_score': 0.4743, 'range_score': 0.2372, 'chop_score': 0.4716, 'volatility_percentile': 0.3777, 'consolidation_score': 0.1917}
2026-05-11 14:35:24,029 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:24,031 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:24,032 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:24,032 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:24,032 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:35:24,034 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:35:24,323 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPUSD — 5040 samples (group=dollar) score_means={'trend_score': 0.4843, 'range_score': 0.2371, 'chop_score': 0.4614, 'volatility_percentile': 0.377, 'consolidation_score': 0.1835}
2026-05-11 14:35:24,425 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2317, 'chop_score': 0.462, 'volatility_percentile': 0.3807, 'consolidation_score': 0.1858}, 'dollar': {'trend_score': 0.4937, 'range_score': 0.2348, 'chop_score': 0.4583, 'volatility_percentile': 0.3806, 'consolidation_score': 0.1839}, 'gold': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}}
2026-05-11 14:35:24,425 INFO Regime[1H mode=ltf_behaviour] score means by year: {2022: {'trend_score': 0.4905, 'range_score': 0.2364, 'chop_score': 0.4625, 'volatility_percentile': 0.385, 'consolidation_score': 0.1797}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 14:35:24,500 INFO Regime phase LTF dataset build fold=train_all: 9.1s (train=262644 val=30352)
2026-05-11 14:35:24,500 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260511_143524
2026-05-11 14:35:24,505 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=53, n_classes=5)
2026-05-11 14:35:24,505 INFO Regime 1H/ltf_behaviour warm start enabled from existing weights
2026-05-11 14:35:24,538 INFO RegimeClassifier[mode=ltf_behaviour]: warm start score head
2026-05-11 14:35:24,538 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-11 14:35:25,055 INFO Regime score epoch  1/50 — tr=0.0049 va=0.0016 mae={'trend_score': 0.0272, 'range_score': 0.0414, 'chop_score': 0.029, 'volatility_percentile': 0.0219, 'consolidation_score': 0.0293}
2026-05-11 14:35:25,555 INFO Regime score epoch  2/50 — tr=0.0049 va=0.0016
2026-05-11 14:35:26,044 INFO Regime score epoch  3/50 — tr=0.0049 va=0.0016
2026-05-11 14:35:26,534 INFO Regime score epoch  4/50 — tr=0.0049 va=0.0016
2026-05-11 14:35:27,033 INFO Regime score epoch  5/50 — tr=0.0048 va=0.0016 mae={'trend_score': 0.0267, 'range_score': 0.0407, 'chop_score': 0.0286, 'volatility_percentile': 0.0218, 'consolidation_score': 0.0288}
2026-05-11 14:35:27,545 INFO Regime score epoch  6/50 — tr=0.0048 va=0.0015
2026-05-11 14:35:28,025 INFO Regime score epoch  7/50 — tr=0.0048 va=0.0015
2026-05-11 14:35:28,543 INFO Regime score epoch  8/50 — tr=0.0047 va=0.0015
2026-05-11 14:35:29,029 INFO Regime score epoch  9/50 — tr=0.0047 va=0.0015
2026-05-11 14:35:29,529 INFO Regime score epoch 10/50 — tr=0.0046 va=0.0014 mae={'trend_score': 0.0253, 'range_score': 0.0393, 'chop_score': 0.027, 'volatility_percentile': 0.0203, 'consolidation_score': 0.0266}
2026-05-11 14:35:30,029 INFO Regime score epoch 11/50 — tr=0.0046 va=0.0014
2026-05-11 14:35:30,524 INFO Regime score epoch 12/50 — tr=0.0045 va=0.0014
2026-05-11 14:35:31,018 INFO Regime score epoch 13/50 — tr=0.0045 va=0.0013
2026-05-11 14:35:31,514 INFO Regime score epoch 14/50 — tr=0.0044 va=0.0013
2026-05-11 14:35:32,020 INFO Regime score epoch 15/50 — tr=0.0044 va=0.0013 mae={'trend_score': 0.0241, 'range_score': 0.0383, 'chop_score': 0.0253, 'volatility_percentile': 0.0194, 'consolidation_score': 0.0247}
2026-05-11 14:35:32,514 INFO Regime score epoch 16/50 — tr=0.0043 va=0.0013
2026-05-11 14:35:33,018 INFO Regime score epoch 17/50 — tr=0.0043 va=0.0012
2026-05-11 14:35:33,512 INFO Regime score epoch 18/50 — tr=0.0043 va=0.0012
2026-05-11 14:35:34,023 INFO Regime score epoch 19/50 — tr=0.0042 va=0.0012
2026-05-11 14:35:34,542 INFO Regime score epoch 20/50 — tr=0.0042 va=0.0012 mae={'trend_score': 0.0226, 'range_score': 0.0375, 'chop_score': 0.0245, 'volatility_percentile': 0.0182, 'consolidation_score': 0.024}
2026-05-11 14:35:35,039 INFO Regime score epoch 21/50 — tr=0.0042 va=0.0012
2026-05-11 14:35:35,560 INFO Regime score epoch 22/50 — tr=0.0041 va=0.0012
2026-05-11 14:35:36,063 INFO Regime score epoch 23/50 — tr=0.0041 va=0.0011
2026-05-11 14:35:36,571 INFO Regime score epoch 24/50 — tr=0.0041 va=0.0011
2026-05-11 14:35:37,068 INFO Regime score epoch 25/50 — tr=0.0041 va=0.0011 mae={'trend_score': 0.0216, 'range_score': 0.0364, 'chop_score': 0.0233, 'volatility_percentile': 0.0178, 'consolidation_score': 0.0239}
2026-05-11 14:35:37,549 INFO Regime score epoch 26/50 — tr=0.0040 va=0.0011
2026-05-11 14:35:38,051 INFO Regime score epoch 27/50 — tr=0.0040 va=0.0011
2026-05-11 14:35:38,593 INFO Regime score epoch 28/50 — tr=0.0040 va=0.0011
2026-05-11 14:35:39,074 INFO Regime score epoch 29/50 — tr=0.0040 va=0.0011
2026-05-11 14:35:39,563 INFO Regime score epoch 30/50 — tr=0.0040 va=0.0011 mae={'trend_score': 0.0208, 'range_score': 0.0359, 'chop_score': 0.0227, 'volatility_percentile': 0.0171, 'consolidation_score': 0.0224}
2026-05-11 14:35:40,050 INFO Regime score epoch 31/50 — tr=0.0039 va=0.0011
2026-05-11 14:35:40,535 INFO Regime score epoch 32/50 — tr=0.0039 va=0.0011
2026-05-11 14:35:41,022 INFO Regime score epoch 33/50 — tr=0.0039 va=0.0011
2026-05-11 14:35:41,506 INFO Regime score epoch 34/50 — tr=0.0039 va=0.0010
2026-05-11 14:35:42,013 INFO Regime score epoch 35/50 — tr=0.0039 va=0.0010 mae={'trend_score': 0.0207, 'range_score': 0.0356, 'chop_score': 0.0222, 'volatility_percentile': 0.0167, 'consolidation_score': 0.022}
2026-05-11 14:35:42,492 INFO Regime score epoch 36/50 — tr=0.0039 va=0.0011
2026-05-11 14:35:42,969 INFO Regime score epoch 37/50 — tr=0.0039 va=0.0010
2026-05-11 14:35:43,446 INFO Regime score epoch 38/50 — tr=0.0039 va=0.0010
2026-05-11 14:35:43,941 INFO Regime score epoch 39/50 — tr=0.0039 va=0.0010
2026-05-11 14:35:44,433 INFO Regime score epoch 40/50 — tr=0.0039 va=0.0010 mae={'trend_score': 0.0203, 'range_score': 0.0353, 'chop_score': 0.0219, 'volatility_percentile': 0.0169, 'consolidation_score': 0.0233}
2026-05-11 14:35:44,915 INFO Regime score epoch 41/50 — tr=0.0039 va=0.0010
2026-05-11 14:35:45,418 INFO Regime score epoch 42/50 — tr=0.0039 va=0.0011
2026-05-11 14:35:45,929 INFO Regime score epoch 43/50 — tr=0.0039 va=0.0010
2026-05-11 14:35:46,448 INFO Regime score epoch 44/50 — tr=0.0039 va=0.0010
2026-05-11 14:35:46,936 INFO Regime score epoch 45/50 — tr=0.0039 va=0.0010 mae={'trend_score': 0.0206, 'range_score': 0.0352, 'chop_score': 0.022, 'volatility_percentile': 0.0165, 'consolidation_score': 0.0216}
2026-05-11 14:35:47,442 INFO Regime score epoch 46/50 — tr=0.0039 va=0.0010
2026-05-11 14:35:47,937 INFO Regime score epoch 47/50 — tr=0.0039 va=0.0010
2026-05-11 14:35:48,467 INFO Regime score epoch 48/50 — tr=0.0039 va=0.0010
2026-05-11 14:35:48,973 INFO Regime score epoch 49/50 — tr=0.0039 va=0.0010
2026-05-11 14:35:49,461 INFO Regime score epoch 50/50 — tr=0.0039 va=0.0010 mae={'trend_score': 0.0204, 'range_score': 0.0353, 'chop_score': 0.022, 'volatility_percentile': 0.0167, 'consolidation_score': 0.0217}
2026-05-11 14:35:49,482 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0206, 'range_score': 0.0352, 'chop_score': 0.022, 'volatility_percentile': 0.0165, 'consolidation_score': 0.0216} mse={'trend_score': 0.00072, 'range_score': 0.00203, 'chop_score': 0.00079, 'volatility_percentile': 0.0005, 'consolidation_score': 0.00106} corr={'trend_score': 0.9927, 'range_score': 0.9494, 'chop_score': 0.9894, 'volatility_percentile': 0.9948, 'consolidation_score': 0.9888} pred_std={'trend_score': 0.2193, 'range_score': 0.1348, 'chop_score': 0.1802, 'volatility_percentile': 0.2161, 'consolidation_score': 0.2144} target_std={'trend_score': 0.2219, 'range_score': 0.1433, 'chop_score': 0.1888, 'volatility_percentile': 0.2165, 'consolidation_score': 0.2179}
2026-05-11 14:35:49,797 INFO RegimeClassifier[mode=ltf_behaviour] train group diagnostics={'dollar': {'n': 16632, 'score_mae': {'trend_score': 0.0197, 'range_score': 0.035, 'chop_score': 0.022, 'volatility_percentile': 0.0163, 'consolidation_score': 0.022}, 'target_score_mean': {'trend_score': 0.491, 'range_score': 0.2347, 'chop_score': 0.4613, 'volatility_percentile': 0.3824, 'consolidation_score': 0.186}, 'pred_score_mean': {'trend_score': 0.4931, 'range_score': 0.2364, 'chop_score': 0.4597, 'volatility_percentile': 0.3804, 'consolidation_score': 0.1842}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3569, 50, 0, 2, 0, 0, 158], [11, 96, 0, 0, 0, 0, 3], [0, 0, 224, 10, 42, 0, 184], [5, 0, 12, 542, 28, 0, 102], [0, 0, 69, 22, 3026, 0, 199], [0, 27, 0, 0, 8, 16, 77], [181, 12, 148, 47, 77, 0, 7685]]}, 'gold': {'n': 8420, 'score_mae': {'trend_score': 0.0196, 'range_score': 0.0356, 'chop_score': 0.0223, 'volatility_percentile': 0.0167, 'consolidation_score': 0.0224}, 'target_score_mean': {'trend_score': 0.4883, 'range_score': 0.2355, 'chop_score': 0.4652, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}, 'pred_score_mean': {'trend_score': 0.491, 'range_score': 0.2369, 'chop_score': 0.4628, 'volatility_percentile': 0.3748, 'consolidation_score': 0.1895}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1788, 31, 0, 0, 0, 0, 66], [4, 50, 0, 0, 0, 0, 2], [0, 0, 119, 8, 18, 0, 99], [1, 0, 8, 331, 17, 0, 59], [0, 0, 28, 24, 1539, 0, 113], [0, 22, 0, 0, 9, 11, 39], [79, 5, 88, 21, 39, 0, 3802]]}, 'jpy': {'n': 24948, 'score_mae': {'trend_score': 0.0198, 'range_score': 0.035, 'chop_score': 0.0219, 'volatility_percentile': 0.0169, 'consolidation_score': 0.0219}, 'target_score_mean': {'trend_score': 0.4904, 'range_score': 0.2338, 'chop_score': 0.4653, 'volatility_percentile': 0.3826, 'consolidation_score': 0.1905}, 'pred_score_mean': {'trend_score': 0.492, 'range_score': 0.2358, 'chop_score': 0.463, 'volatility_percentile': 0.3804, 'consolidation_score': 0.1878}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[5327, 111, 0, 4, 0, 0, 204], [22, 159, 0, 0, 0, 0, 6], [0, 0, 308, 20, 54, 0, 265], [5, 0, 11, 1066, 65, 0, 167], [0, 0, 77, 68, 4619, 0, 351], [0, 51, 0, 0, 17, 26, 129], [234, 17, 230, 77, 137, 0, 11121]]}}
2026-05-11 14:35:49,976 INFO RegimeClassifier[mode=ltf_behaviour] validation group diagnostics={'dollar': {'n': 10093, 'score_mae': {'trend_score': 0.0208, 'range_score': 0.0362, 'chop_score': 0.0221, 'volatility_percentile': 0.0161, 'consolidation_score': 0.0211}, 'target_score_mean': {'trend_score': 0.4863, 'range_score': 0.2392, 'chop_score': 0.4622, 'volatility_percentile': 0.3804, 'consolidation_score': 0.1814}, 'pred_score_mean': {'trend_score': 0.4892, 'range_score': 0.2386, 'chop_score': 0.4611, 'volatility_percentile': 0.3782, 'consolidation_score': 0.1805}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[2290, 20, 0, 3, 0, 0, 99], [6, 46, 0, 0, 0, 0, 1], [0, 0, 142, 7, 34, 0, 133], [1, 0, 6, 326, 26, 0, 64], [0, 0, 45, 30, 1877, 0, 98], [0, 21, 0, 0, 4, 9, 43], [82, 4, 94, 43, 54, 0, 4485]]}, 'gold': {'n': 5117, 'score_mae': {'trend_score': 0.0194, 'range_score': 0.034, 'chop_score': 0.022, 'volatility_percentile': 0.0163, 'consolidation_score': 0.0223}, 'target_score_mean': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}, 'pred_score_mean': {'trend_score': 0.4993, 'range_score': 0.2347, 'chop_score': 0.455, 'volatility_percentile': 0.3793, 'consolidation_score': 0.1811}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1106, 16, 0, 0, 0, 0, 45], [4, 30, 0, 0, 0, 0, 1], [0, 0, 85, 2, 12, 0, 72], [0, 0, 4, 212, 9, 0, 30], [0, 0, 21, 12, 796, 0, 58], [0, 13, 0, 0, 4, 4, 29], [56, 3, 68, 23, 24, 0, 2378]]}, 'jpy': {'n': 15142, 'score_mae': {'trend_score': 0.0208, 'range_score': 0.0349, 'chop_score': 0.022, 'volatility_percentile': 0.0168, 'consolidation_score': 0.0217}, 'target_score_mean': {'trend_score': 0.495, 'range_score': 0.2297, 'chop_score': 0.4582, 'volatility_percentile': 0.3808, 'consolidation_score': 0.1868}, 'pred_score_mean': {'trend_score': 0.496, 'range_score': 0.2312, 'chop_score': 0.4571, 'volatility_percentile': 0.3788, 'consolidation_score': 0.1846}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3320, 65, 0, 1, 0, 0, 132], [12, 98, 0, 0, 0, 0, 5], [0, 0, 183, 14, 39, 0, 148], [2, 0, 15, 667, 38, 0, 105], [0, 0, 50, 29, 2544, 0, 194], [0, 26, 0, 0, 10, 19, 67], [128, 18, 143, 45, 88, 0, 6937]]}}
2026-05-11 14:35:49,982 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-11 14:35:49,983 INFO Regime phase LTF train fold=train_all: 25.5s
2026-05-11 14:35:50,086 INFO Regime LTF complete fold=train_all: score_accuracy=0.977, train=262644 val=30352 mae={'trend_score': 0.0206, 'range_score': 0.0352, 'chop_score': 0.022, 'volatility_percentile': 0.0165, 'consolidation_score': 0.0216}
2026-05-11 14:35:50,088 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 14:35:50,432 INFO Regime[1H mode=ltf_behaviour fold=latest] LTF score diagnostics on GBPUSD:
{'trend_score': {'mean': 0.4921, 'q10': 0.1918, 'q50': 0.4877, 'q90': 0.8002}, 'range_score': {'mean': 0.2337, 'q10': 0.0513, 'q50': 0.2125, 'q90': 0.4349}, 'chop_score': {'mean': 0.4612, 'q10': 0.2148, 'q50': 0.4503, 'q90': 0.7255}, 'volatility_percentile': {'mean': 0.3832, 'q10': 0.1018, 'q50': 0.3704, 'q90': 0.6823}, 'consolidation_score': {'mean': 0.1854, 'q10': 0.0, 'q50': 0.1185, 'q90': 0.506}}
2026-05-11 14:35:50,437 INFO Regime retrain total: 75.8s (370559 train+val samples)
2026-05-11 14:35:50,442 INFO Retrain complete. Total wall-clock: 75.8s
  DONE  Retrain regime [pre-R2 retrain]

=== Round 2: BLIND backtest on test window (unseen 2yr) ===
  START Round 2 - Blind backtest (test)
2026-05-11 14:35:51,956 INFO === STEP 6: BACKTEST (round2) ===
2026-05-11 14:35:51,958 INFO BT_WINDOW=round2 — BLIND backtest: 2023-08-07 → 2025-08-05 (test set)
2026-05-11 14:35:51,958 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-05-11 14:35:51,958 INFO Round 2 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
2026-05-11 14:35:51,958 INFO Round 2 — QualityScorer hard gate disabled; set BACKTEST_ENABLE_QUALITY_GATE=1 to enable it
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
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
2026-05-11 14:37:11,839 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 14:37:12,798 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 14:37:12,992 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 14:37:13,395 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 14:37:13,841 WARNING _build_sequence_df: HTF frame 5M filled 317 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 14:37:13,892 WARNING _build_sequence_df: HTF frame 5M filled 317 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 14:37:13,936 WARNING _build_sequence_df: HTF frame 5M filled 341 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 14:37:14,000 WARNING _build_sequence_df: HTF frame 5M filled 341 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
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
2026-05-11 14:37:25,025 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 14:37:25,251 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
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
2026-05-11 14:37:25,358 WARNING _build_sequence_df: HTF frame 5M filled 317 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 14:37:25,395 WARNING _build_sequence_df: HTF frame 5M filled 317 warmup/alignment gaps with 0.000
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
2026-05-11 14:38:25,082 INFO Round 2 backtest — 169 trades | avg WR=24.9% | avg PF=0.98 | avg Sharpe=-0.15
2026-05-11 14:38:25,082 INFO   ml_trader: 169 trades | WR=24.9% | fixed PF=0.98 | Return=-2.8% | ExpR=-0.017 | DD=19.8% | Sharpe=-0.15
2026-05-11 14:38:25,082 INFO   ml_trader gate_diagnostics: bars=280782 no_signal=205970 quality_block=0 session_skip=74641 density=2 pm_reject=0
2026-05-11 14:38:25,082 INFO   ml_trader no_signal_reasons: {'no_trade_chop': 22919, 'no_trade_uncertain': 74759, 'weak_gru_direction': 55511, 'gru_expected_r_below_threshold': 11470, 'trend_structure_missing': 6747, 'wait_pullback': 15761, 'no_trade_extreme_vol': 18701, 'tradeability_direction_conflict': 91, 'expected_r_below_threshold': 11}
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 169
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (169 rows)
2026-05-11 14:38:25,473 INFO Round 2: wrote 169 journal entries (total in file: 324)
  DONE  Round 2 - Blind backtest (test)
  Saved Round 2 result → round2_summary.json
  Journal after Round 2: 324 entries

  SKIP  Round 2 Quality+RL retrain — blind test journal kept untouched

=== Round 3: Incremental retrain ===
  START Retrain gru [train-split retrain]
2026-05-11 14:38:25,994 INFO retrain environment: KAGGLE
2026-05-11 14:38:27,817 INFO Device: CUDA (2 GPU(s))
2026-05-11 14:38:27,829 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 14:38:27,829 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 14:38:27,829 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 14:38:27,831 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 14:38:27,831 INFO Retrain data split: train
2026-05-11 14:38:27,832 INFO Retrain rolling fold selector: latest
2026-05-11 14:38:27,833 INFO === GRU-LSTM retrain (timeframes: ['15M']) ===
2026-05-11 14:38:27,996 INFO NumExpr defaulting to 4 threads.
2026-05-11 14:38:28,233 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-11 14:38:28,233 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 14:38:28,233 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 14:38:28,505 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-05-11 14:38:28,505 INFO GRU multi-symbol training (Kaggle mode): 6 symbols × ['15M']
2026-05-11 14:38:28,508 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260511_143828
2026-05-11 14:38:28,513 INFO GRU feature contract unchanged (input_size=94) — incremental retrain
2026-05-11 14:38:28,513 INFO Deleted stale GRU artifact for cold start: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:38:28,514 INFO GRU warm start disabled by default; set GRU_ALLOW_WARM_START=1 to reuse compatible weights
2026-05-11 14:38:28,802 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:38:28,835 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:38:28,852 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:38:28,864 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 14:38:28,951 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 14:38:28,958 INFO Loaded XAUUSD/15M split=train fold=latest: 176438 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:38:29,594 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:38:29,627 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:38:29,644 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:38:29,652 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:38:29,708 INFO Loaded EURUSD/15M split=train fold=latest: 174951 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:38:30,340 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:38:30,363 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:38:30,380 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:38:30,389 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:38:30,433 INFO Loaded USDJPY/15M split=train fold=latest: 174944 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:38:31,045 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:38:31,072 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:38:31,090 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:38:31,099 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:38:31,145 INFO Loaded EURJPY/15M split=train fold=latest: 174948 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:38:31,737 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:38:31,760 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:38:31,779 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:38:31,790 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:38:31,839 INFO Loaded GBPJPY/15M split=train fold=latest: 174902 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:38:32,458 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:38:32,480 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:38:32,498 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:38:32,508 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 14:38:32,551 INFO Loaded GBPUSD/15M split=train fold=latest: 174899 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:38:33,051 INFO train_multi: 6 segments, ~971854 total bars
2026-05-11 14:38:33,058 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-05-11 14:38:33,059 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-05-11 14:38:33,059 INFO train_multi: training ALL 6 segments across TFs ['15M'] in one combined pass
2026-05-11 14:38:33,059 INFO train_multi: building combined dataset for TF=ALL (6 segments)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 14:38:47,043 INFO train_multi TF=ALL: 971494 sequences across 6 segments
2026-05-11 14:38:47,044 INFO train_multi TF=ALL: estimated peak RAM = 27072 MB (train=419996 calib=60000 val=120002 n_feat=94 seq_len=60)
2026-05-11 14:38:47,044 WARNING train_multi TF=ALL: trimming to fit RAM budget — new train=310283 calib=44326 val=88652 (20000 MB est)
2026-05-11 14:38:49,532 INFO train_multi TF=ALL: train=310283 calib=44326 val=88652 (10007 MB tensors)
2026-05-11 14:38:56,706 INFO train_multi TF=ALL: structural bar weighting — 199279 structural bars (64.2%) weight=15.0 structural_only=0
2026-05-11 14:38:57,832 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=25, min_epochs=45)
2026-05-11 14:39:14,624 INFO train_multi TF=ALL epoch 1/100 train=2.3333 val=2.3382 r_mae=0.969 pos_r_acc=0.545 side_acc=0.490 r_n=127469
2026-05-11 14:39:14,630 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:39:14,630 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:39:14,630 INFO train_multi TF=ALL: new best val=2.3382 r_mae=0.9694 — saved
2026-05-11 14:39:14,635 INFO train_multi TF=ALL: new best r_mae=0.9694 — saved rmae checkpoint
2026-05-11 14:39:28,971 INFO train_multi TF=ALL epoch 2/100 train=2.3330 val=2.3372 r_mae=0.969 pos_r_acc=0.545 side_acc=0.490 r_n=127469
2026-05-11 14:39:28,977 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:39:28,977 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:39:28,977 INFO train_multi TF=ALL: new best val=2.3372 r_mae=0.9686 — saved
2026-05-11 14:39:28,982 INFO train_multi TF=ALL: new best r_mae=0.9686 — saved rmae checkpoint
2026-05-11 14:39:43,247 INFO train_multi TF=ALL epoch 3/100 train=2.3327 val=2.3360 r_mae=0.968 pos_r_acc=0.545 side_acc=0.490 r_n=127469
2026-05-11 14:39:43,253 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:39:43,253 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:39:43,253 INFO train_multi TF=ALL: new best val=2.3360 r_mae=0.9676 — saved
2026-05-11 14:39:43,258 INFO train_multi TF=ALL: new best r_mae=0.9676 — saved rmae checkpoint
2026-05-11 14:39:57,709 INFO train_multi TF=ALL epoch 4/100 train=2.3317 val=2.3347 r_mae=0.967 pos_r_acc=0.545 side_acc=0.490 r_n=127469
2026-05-11 14:39:57,719 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:39:57,719 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:39:57,720 INFO train_multi TF=ALL: new best val=2.3347 r_mae=0.9665 — saved
2026-05-11 14:39:57,724 INFO train_multi TF=ALL: new best r_mae=0.9665 — saved rmae checkpoint
2026-05-11 14:40:12,090 INFO train_multi TF=ALL epoch 5/100 train=2.3308 val=2.3334 r_mae=0.966 pos_r_acc=0.545 side_acc=0.490 r_n=127469
2026-05-11 14:40:12,096 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:40:12,096 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:40:12,096 INFO train_multi TF=ALL: new best val=2.3334 r_mae=0.9656 — saved
2026-05-11 14:40:12,101 INFO train_multi TF=ALL: new best r_mae=0.9656 — saved rmae checkpoint
2026-05-11 14:40:26,181 INFO train_multi TF=ALL epoch 6/100 train=2.3308 val=2.3327 r_mae=0.965 pos_r_acc=0.545 side_acc=0.490 r_n=127469
2026-05-11 14:40:26,187 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:40:26,188 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:40:26,188 INFO train_multi TF=ALL: new best val=2.3327 r_mae=0.9652 — saved
2026-05-11 14:40:26,192 INFO train_multi TF=ALL: new best r_mae=0.9652 — saved rmae checkpoint
2026-05-11 14:40:40,009 INFO train_multi TF=ALL epoch 7/100 train=2.3301 val=2.3320 r_mae=0.965 pos_r_acc=0.545 side_acc=0.492 r_n=127469
2026-05-11 14:40:40,015 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:40:40,015 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:40:40,015 INFO train_multi TF=ALL: new best val=2.3320 r_mae=0.9650 — saved
2026-05-11 14:40:40,020 INFO train_multi TF=ALL: new best r_mae=0.9650 — saved rmae checkpoint
2026-05-11 14:40:53,959 INFO train_multi TF=ALL epoch 8/100 train=2.3291 val=2.3306 r_mae=0.964 pos_r_acc=0.545 side_acc=0.525 r_n=127469
2026-05-11 14:40:53,964 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:40:53,965 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:40:53,965 INFO train_multi TF=ALL: new best val=2.3306 r_mae=0.9645 — saved
2026-05-11 14:40:53,969 INFO train_multi TF=ALL: new best r_mae=0.9645 — saved rmae checkpoint
2026-05-11 14:41:07,870 INFO train_multi TF=ALL epoch 9/100 train=2.3274 val=2.3292 r_mae=0.963 pos_r_acc=0.546 side_acc=0.528 r_n=127469
2026-05-11 14:41:07,876 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:41:07,876 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:41:07,876 INFO train_multi TF=ALL: new best val=2.3292 r_mae=0.9634 — saved
2026-05-11 14:41:07,881 INFO train_multi TF=ALL: new best r_mae=0.9634 — saved rmae checkpoint
2026-05-11 14:41:22,238 INFO train_multi TF=ALL epoch 10/100 train=2.3250 val=2.3266 r_mae=0.963 pos_r_acc=0.545 side_acc=0.525 r_n=127469
2026-05-11 14:41:22,249 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:41:22,249 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:41:22,249 INFO train_multi TF=ALL: new best val=2.3266 r_mae=0.9628 — saved
2026-05-11 14:41:22,253 INFO train_multi TF=ALL: new best r_mae=0.9628 — saved rmae checkpoint
2026-05-11 14:41:36,464 INFO train_multi TF=ALL epoch 11/100 train=2.3223 val=2.3239 r_mae=0.963 pos_r_acc=0.546 side_acc=0.529 r_n=127469
2026-05-11 14:41:36,469 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:41:36,470 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:41:36,470 INFO train_multi TF=ALL: new best val=2.3239 r_mae=0.9626 — saved
2026-05-11 14:41:36,474 INFO train_multi TF=ALL: new best r_mae=0.9626 — saved rmae checkpoint
2026-05-11 14:41:50,602 INFO train_multi TF=ALL epoch 12/100 train=2.3206 val=2.3225 r_mae=0.962 pos_r_acc=0.548 side_acc=0.528 r_n=127469
2026-05-11 14:41:50,607 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:41:50,607 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:41:50,608 INFO train_multi TF=ALL: new best val=2.3225 r_mae=0.9621 — saved
2026-05-11 14:41:50,612 INFO train_multi TF=ALL: new best r_mae=0.9621 — saved rmae checkpoint
2026-05-11 14:42:04,919 INFO train_multi TF=ALL epoch 13/100 train=2.3186 val=2.3211 r_mae=0.962 pos_r_acc=0.548 side_acc=0.526 r_n=127469
2026-05-11 14:42:04,926 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:42:04,926 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:42:04,926 INFO train_multi TF=ALL: new best val=2.3211 r_mae=0.9617 — saved
2026-05-11 14:42:04,931 INFO train_multi TF=ALL: new best r_mae=0.9617 — saved rmae checkpoint
2026-05-11 14:42:20,436 INFO train_multi TF=ALL epoch 14/100 train=2.3171 val=2.3192 r_mae=0.961 pos_r_acc=0.549 side_acc=0.531 r_n=127469
2026-05-11 14:42:20,442 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:42:20,443 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:42:20,443 INFO train_multi TF=ALL: new best val=2.3192 r_mae=0.9606 — saved
2026-05-11 14:42:20,448 INFO train_multi TF=ALL: new best r_mae=0.9606 — saved rmae checkpoint
2026-05-11 14:42:36,544 INFO train_multi TF=ALL epoch 15/100 train=2.3155 val=2.3170 r_mae=0.960 pos_r_acc=0.548 side_acc=0.534 r_n=127469
2026-05-11 14:42:36,551 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:42:36,551 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:42:36,551 INFO train_multi TF=ALL: new best val=2.3170 r_mae=0.9602 — saved
2026-05-11 14:42:36,557 INFO train_multi TF=ALL: new best r_mae=0.9602 — saved rmae checkpoint
2026-05-11 14:42:52,633 INFO train_multi TF=ALL epoch 16/100 train=2.3135 val=2.3167 r_mae=0.959 pos_r_acc=0.549 side_acc=0.534 r_n=127469
2026-05-11 14:42:52,639 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:42:52,640 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:42:52,640 INFO train_multi TF=ALL: new best val=2.3167 r_mae=0.9588 — saved
2026-05-11 14:42:52,645 INFO train_multi TF=ALL: new best r_mae=0.9588 — saved rmae checkpoint
2026-05-11 14:43:08,709 INFO train_multi TF=ALL epoch 17/100 train=2.3107 val=2.3127 r_mae=0.958 pos_r_acc=0.552 side_acc=0.539 r_n=127469
2026-05-11 14:43:08,715 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:43:08,715 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:43:08,715 INFO train_multi TF=ALL: new best val=2.3127 r_mae=0.9579 — saved
2026-05-11 14:43:08,721 INFO train_multi TF=ALL: new best r_mae=0.9579 — saved rmae checkpoint
2026-05-11 14:43:24,871 INFO train_multi TF=ALL epoch 18/100 train=2.3078 val=2.3114 r_mae=0.957 pos_r_acc=0.554 side_acc=0.539 r_n=127469
2026-05-11 14:43:24,878 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:43:24,878 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:43:24,878 INFO train_multi TF=ALL: new best val=2.3114 r_mae=0.9572 — saved
2026-05-11 14:43:24,883 INFO train_multi TF=ALL: new best r_mae=0.9572 — saved rmae checkpoint
2026-05-11 14:43:41,134 INFO train_multi TF=ALL epoch 19/100 train=2.3047 val=2.3099 r_mae=0.956 pos_r_acc=0.558 side_acc=0.543 r_n=127469
2026-05-11 14:43:41,140 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:43:41,140 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:43:41,140 INFO train_multi TF=ALL: new best val=2.3099 r_mae=0.9559 — saved
2026-05-11 14:43:41,145 INFO train_multi TF=ALL: new best r_mae=0.9559 — saved rmae checkpoint
2026-05-11 14:43:56,687 INFO train_multi TF=ALL epoch 20/100 train=2.3016 val=2.3074 r_mae=0.953 pos_r_acc=0.562 side_acc=0.540 r_n=127469
2026-05-11 14:43:56,694 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:43:56,694 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:43:56,694 INFO train_multi TF=ALL: new best val=2.3074 r_mae=0.9527 — saved
2026-05-11 14:43:56,699 INFO train_multi TF=ALL: new best r_mae=0.9527 — saved rmae checkpoint
2026-05-11 14:44:11,938 INFO train_multi TF=ALL epoch 21/100 train=2.2972 val=2.3019 r_mae=0.950 pos_r_acc=0.566 side_acc=0.543 r_n=127469
2026-05-11 14:44:11,944 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:44:11,945 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:44:11,945 INFO train_multi TF=ALL: new best val=2.3019 r_mae=0.9500 — saved
2026-05-11 14:44:11,949 INFO train_multi TF=ALL: new best r_mae=0.9500 — saved rmae checkpoint
2026-05-11 14:44:27,239 INFO train_multi TF=ALL epoch 22/100 train=2.2908 val=2.2906 r_mae=0.947 pos_r_acc=0.571 side_acc=0.543 r_n=127469
2026-05-11 14:44:27,245 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:44:27,245 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:44:27,245 INFO train_multi TF=ALL: new best val=2.2906 r_mae=0.9469 — saved
2026-05-11 14:44:27,250 INFO train_multi TF=ALL: new best r_mae=0.9469 — saved rmae checkpoint
2026-05-11 14:44:42,700 INFO train_multi TF=ALL epoch 23/100 train=2.2810 val=2.2830 r_mae=0.943 pos_r_acc=0.574 side_acc=0.549 r_n=127469
2026-05-11 14:44:42,706 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:44:42,706 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:44:42,706 INFO train_multi TF=ALL: new best val=2.2830 r_mae=0.9428 — saved
2026-05-11 14:44:42,711 INFO train_multi TF=ALL: new best r_mae=0.9428 — saved rmae checkpoint
2026-05-11 14:44:57,878 INFO train_multi TF=ALL epoch 24/100 train=2.2737 val=2.2780 r_mae=0.938 pos_r_acc=0.581 side_acc=0.553 r_n=127469
2026-05-11 14:44:57,884 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:44:57,884 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:44:57,884 INFO train_multi TF=ALL: new best val=2.2780 r_mae=0.9384 — saved
2026-05-11 14:44:57,888 INFO train_multi TF=ALL: new best r_mae=0.9384 — saved rmae checkpoint
2026-05-11 14:45:12,339 INFO train_multi TF=ALL epoch 25/100 train=2.2657 val=2.2786 r_mae=0.934 pos_r_acc=0.581 side_acc=0.551 r_n=127469
2026-05-11 14:45:12,344 INFO train_multi TF=ALL: new best r_mae=0.9344 — saved rmae checkpoint
2026-05-11 14:45:27,573 INFO train_multi TF=ALL epoch 26/100 train=2.2569 val=2.2658 r_mae=0.932 pos_r_acc=0.587 side_acc=0.555 r_n=127469
2026-05-11 14:45:27,579 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:45:27,579 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:45:27,579 INFO train_multi TF=ALL: new best val=2.2658 r_mae=0.9323 — saved
2026-05-11 14:45:27,584 INFO train_multi TF=ALL: new best r_mae=0.9323 — saved rmae checkpoint
2026-05-11 14:45:42,916 INFO train_multi TF=ALL epoch 27/100 train=2.2483 val=2.2622 r_mae=0.928 pos_r_acc=0.589 side_acc=0.557 r_n=127469
2026-05-11 14:45:42,922 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:45:42,922 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:45:42,922 INFO train_multi TF=ALL: new best val=2.2622 r_mae=0.9283 — saved
2026-05-11 14:45:42,927 INFO train_multi TF=ALL: new best r_mae=0.9283 — saved rmae checkpoint
2026-05-11 14:45:58,126 INFO train_multi TF=ALL epoch 28/100 train=2.2409 val=2.2656 r_mae=0.928 pos_r_acc=0.585 side_acc=0.556 r_n=127469
2026-05-11 14:45:58,132 INFO train_multi TF=ALL: new best r_mae=0.9280 — saved rmae checkpoint
2026-05-11 14:46:13,387 INFO train_multi TF=ALL epoch 29/100 train=2.2327 val=2.2584 r_mae=0.926 pos_r_acc=0.588 side_acc=0.562 r_n=127469
2026-05-11 14:46:13,393 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:46:13,393 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:46:13,393 INFO train_multi TF=ALL: new best val=2.2584 r_mae=0.9259 — saved
2026-05-11 14:46:13,398 INFO train_multi TF=ALL: new best r_mae=0.9259 — saved rmae checkpoint
2026-05-11 14:46:28,631 INFO train_multi TF=ALL epoch 30/100 train=2.2285 val=2.2496 r_mae=0.924 pos_r_acc=0.593 side_acc=0.569 r_n=127469
2026-05-11 14:46:28,637 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:46:28,637 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:46:28,637 INFO train_multi TF=ALL: new best val=2.2496 r_mae=0.9243 — saved
2026-05-11 14:46:28,642 INFO train_multi TF=ALL: new best r_mae=0.9243 — saved rmae checkpoint
2026-05-11 14:46:44,020 INFO train_multi TF=ALL epoch 31/100 train=2.2191 val=2.2495 r_mae=0.924 pos_r_acc=0.593 side_acc=0.568 r_n=127469
2026-05-11 14:46:44,026 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:46:44,026 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:46:44,026 INFO train_multi TF=ALL: new best val=2.2495 r_mae=0.9241 — saved
2026-05-11 14:46:44,031 INFO train_multi TF=ALL: new best r_mae=0.9241 — saved rmae checkpoint
2026-05-11 14:46:59,223 INFO train_multi TF=ALL epoch 32/100 train=2.2163 val=2.2485 r_mae=0.921 pos_r_acc=0.594 side_acc=0.565 r_n=127469
2026-05-11 14:46:59,229 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:46:59,229 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:46:59,229 INFO train_multi TF=ALL: new best val=2.2485 r_mae=0.9214 — saved
2026-05-11 14:46:59,235 INFO train_multi TF=ALL: new best r_mae=0.9214 — saved rmae checkpoint
2026-05-11 14:47:14,777 INFO train_multi TF=ALL epoch 33/100 train=2.2064 val=2.2447 r_mae=0.917 pos_r_acc=0.593 side_acc=0.571 r_n=127469
2026-05-11 14:47:14,782 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:47:14,783 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:47:14,783 INFO train_multi TF=ALL: new best val=2.2447 r_mae=0.9172 — saved
2026-05-11 14:47:14,788 INFO train_multi TF=ALL: new best r_mae=0.9172 — saved rmae checkpoint
2026-05-11 14:47:29,966 INFO train_multi TF=ALL epoch 34/100 train=2.1975 val=2.2427 r_mae=0.916 pos_r_acc=0.598 side_acc=0.575 r_n=127469
2026-05-11 14:47:29,972 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:47:29,972 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:47:29,973 INFO train_multi TF=ALL: new best val=2.2427 r_mae=0.9156 — saved
2026-05-11 14:47:29,977 INFO train_multi TF=ALL: new best r_mae=0.9156 — saved rmae checkpoint
2026-05-11 14:47:45,207 INFO train_multi TF=ALL epoch 35/100 train=2.1897 val=2.2318 r_mae=0.914 pos_r_acc=0.600 side_acc=0.582 r_n=127469
2026-05-11 14:47:45,214 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:47:45,214 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:47:45,214 INFO train_multi TF=ALL: new best val=2.2318 r_mae=0.9141 — saved
2026-05-11 14:47:45,219 INFO train_multi TF=ALL: new best r_mae=0.9141 — saved rmae checkpoint
2026-05-11 14:48:00,295 INFO train_multi TF=ALL epoch 36/100 train=2.1823 val=2.2395 r_mae=0.913 pos_r_acc=0.596 side_acc=0.580 r_n=127469
2026-05-11 14:48:00,300 INFO train_multi TF=ALL: new best r_mae=0.9132 — saved rmae checkpoint
2026-05-11 14:48:15,493 INFO train_multi TF=ALL epoch 37/100 train=2.1718 val=2.2214 r_mae=0.908 pos_r_acc=0.604 side_acc=0.585 r_n=127469
2026-05-11 14:48:15,499 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:48:15,499 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:48:15,499 INFO train_multi TF=ALL: new best val=2.2214 r_mae=0.9076 — saved
2026-05-11 14:48:15,504 INFO train_multi TF=ALL: new best r_mae=0.9076 — saved rmae checkpoint
2026-05-11 14:48:30,746 INFO train_multi TF=ALL epoch 38/100 train=2.1627 val=2.2114 r_mae=0.904 pos_r_acc=0.606 side_acc=0.594 r_n=127469
2026-05-11 14:48:30,752 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:48:30,752 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:48:30,752 INFO train_multi TF=ALL: new best val=2.2114 r_mae=0.9038 — saved
2026-05-11 14:48:30,757 INFO train_multi TF=ALL: new best r_mae=0.9038 — saved rmae checkpoint
2026-05-11 14:48:46,127 INFO train_multi TF=ALL epoch 39/100 train=2.1512 val=2.1991 r_mae=0.898 pos_r_acc=0.612 side_acc=0.599 r_n=127469
2026-05-11 14:48:46,138 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:48:46,138 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:48:46,138 INFO train_multi TF=ALL: new best val=2.1991 r_mae=0.8984 — saved
2026-05-11 14:48:46,143 INFO train_multi TF=ALL: new best r_mae=0.8984 — saved rmae checkpoint
2026-05-11 14:49:01,547 INFO train_multi TF=ALL epoch 40/100 train=2.1362 val=2.1924 r_mae=0.894 pos_r_acc=0.616 side_acc=0.600 r_n=127469
2026-05-11 14:49:01,553 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:49:01,553 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:49:01,553 INFO train_multi TF=ALL: new best val=2.1924 r_mae=0.8942 — saved
2026-05-11 14:49:01,557 INFO train_multi TF=ALL: new best r_mae=0.8942 — saved rmae checkpoint
2026-05-11 14:49:16,540 INFO train_multi TF=ALL epoch 41/100 train=2.1209 val=2.1806 r_mae=0.890 pos_r_acc=0.621 side_acc=0.604 r_n=127469
2026-05-11 14:49:16,546 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:49:16,546 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:49:16,546 INFO train_multi TF=ALL: new best val=2.1806 r_mae=0.8899 — saved
2026-05-11 14:49:16,551 INFO train_multi TF=ALL: new best r_mae=0.8899 — saved rmae checkpoint
2026-05-11 14:49:31,806 INFO train_multi TF=ALL epoch 42/100 train=2.1053 val=2.1679 r_mae=0.878 pos_r_acc=0.629 side_acc=0.611 r_n=127469
2026-05-11 14:49:31,812 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:49:31,812 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:49:31,812 INFO train_multi TF=ALL: new best val=2.1679 r_mae=0.8776 — saved
2026-05-11 14:49:31,817 INFO train_multi TF=ALL: new best r_mae=0.8776 — saved rmae checkpoint
2026-05-11 14:49:46,945 INFO train_multi TF=ALL epoch 43/100 train=2.0883 val=2.1479 r_mae=0.872 pos_r_acc=0.636 side_acc=0.618 r_n=127469
2026-05-11 14:49:46,952 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:49:46,952 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:49:46,952 INFO train_multi TF=ALL: new best val=2.1479 r_mae=0.8717 — saved
2026-05-11 14:49:46,957 INFO train_multi TF=ALL: new best r_mae=0.8717 — saved rmae checkpoint
2026-05-11 14:50:02,522 INFO train_multi TF=ALL epoch 44/100 train=2.0758 val=2.1353 r_mae=0.867 pos_r_acc=0.638 side_acc=0.621 r_n=127469
2026-05-11 14:50:02,528 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:50:02,528 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:50:02,528 INFO train_multi TF=ALL: new best val=2.1353 r_mae=0.8671 — saved
2026-05-11 14:50:02,533 INFO train_multi TF=ALL: new best r_mae=0.8671 — saved rmae checkpoint
2026-05-11 14:50:17,629 INFO train_multi TF=ALL epoch 45/100 train=2.0593 val=2.1450 r_mae=0.855 pos_r_acc=0.639 side_acc=0.619 r_n=127469
2026-05-11 14:50:17,634 INFO train_multi TF=ALL: new best r_mae=0.8552 — saved rmae checkpoint
2026-05-11 14:50:32,967 INFO train_multi TF=ALL epoch 46/100 train=2.0469 val=2.1364 r_mae=0.859 pos_r_acc=0.638 side_acc=0.618 r_n=127469
2026-05-11 14:50:48,130 INFO train_multi TF=ALL epoch 47/100 train=2.0395 val=2.1172 r_mae=0.845 pos_r_acc=0.649 side_acc=0.627 r_n=127469
2026-05-11 14:50:48,136 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:50:48,136 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:50:48,136 INFO train_multi TF=ALL: new best val=2.1172 r_mae=0.8449 — saved
2026-05-11 14:50:48,141 INFO train_multi TF=ALL: new best r_mae=0.8449 — saved rmae checkpoint
2026-05-11 14:51:03,336 INFO train_multi TF=ALL epoch 48/100 train=2.0213 val=2.1098 r_mae=0.842 pos_r_acc=0.651 side_acc=0.628 r_n=127469
2026-05-11 14:51:03,342 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:51:03,342 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:51:03,342 INFO train_multi TF=ALL: new best val=2.1098 r_mae=0.8419 — saved
2026-05-11 14:51:03,347 INFO train_multi TF=ALL: new best r_mae=0.8419 — saved rmae checkpoint
2026-05-11 14:51:18,526 INFO train_multi TF=ALL epoch 49/100 train=2.0129 val=2.1209 r_mae=0.841 pos_r_acc=0.647 side_acc=0.626 r_n=127469
2026-05-11 14:51:18,531 INFO train_multi TF=ALL: new best r_mae=0.8411 — saved rmae checkpoint
2026-05-11 14:51:33,692 INFO train_multi TF=ALL epoch 50/100 train=2.0014 val=2.1112 r_mae=0.828 pos_r_acc=0.654 side_acc=0.630 r_n=127469
2026-05-11 14:51:33,698 INFO train_multi TF=ALL: new best r_mae=0.8278 — saved rmae checkpoint
2026-05-11 14:51:48,838 INFO train_multi TF=ALL epoch 51/100 train=1.9966 val=2.1081 r_mae=0.827 pos_r_acc=0.657 side_acc=0.629 r_n=127469
2026-05-11 14:51:48,844 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:51:48,844 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:51:48,845 INFO train_multi TF=ALL: new best val=2.1081 r_mae=0.8274 — saved
2026-05-11 14:51:48,849 INFO train_multi TF=ALL: new best r_mae=0.8274 — saved rmae checkpoint
2026-05-11 14:52:04,194 INFO train_multi TF=ALL epoch 52/100 train=1.9859 val=2.0961 r_mae=0.830 pos_r_acc=0.657 side_acc=0.631 r_n=127469
2026-05-11 14:52:04,205 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:52:04,205 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:52:04,206 INFO train_multi TF=ALL: new best val=2.0961 r_mae=0.8299 — saved
2026-05-11 14:52:19,439 INFO train_multi TF=ALL epoch 53/100 train=1.9761 val=2.1026 r_mae=0.826 pos_r_acc=0.657 side_acc=0.631 r_n=127469
2026-05-11 14:52:19,445 INFO train_multi TF=ALL: new best r_mae=0.8256 — saved rmae checkpoint
2026-05-11 14:52:34,491 INFO train_multi TF=ALL epoch 54/100 train=1.9723 val=2.1046 r_mae=0.823 pos_r_acc=0.658 side_acc=0.634 r_n=127469
2026-05-11 14:52:34,496 INFO train_multi TF=ALL: new best r_mae=0.8227 — saved rmae checkpoint
2026-05-11 14:52:49,661 INFO train_multi TF=ALL epoch 55/100 train=1.9642 val=2.1090 r_mae=0.824 pos_r_acc=0.654 side_acc=0.635 r_n=127469
2026-05-11 14:53:04,626 INFO train_multi TF=ALL epoch 56/100 train=1.9614 val=2.1134 r_mae=0.832 pos_r_acc=0.649 side_acc=0.627 r_n=127469
2026-05-11 14:53:18,362 INFO train_multi TF=ALL epoch 57/100 train=1.9529 val=2.1034 r_mae=0.819 pos_r_acc=0.658 side_acc=0.632 r_n=127469
2026-05-11 14:53:18,367 INFO train_multi TF=ALL: new best r_mae=0.8186 — saved rmae checkpoint
2026-05-11 14:53:33,518 INFO train_multi TF=ALL epoch 58/100 train=1.9453 val=2.0946 r_mae=0.824 pos_r_acc=0.657 side_acc=0.633 r_n=127469
2026-05-11 14:53:33,524 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 14:53:33,525 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 14:53:33,525 INFO train_multi TF=ALL: new best val=2.0946 r_mae=0.8240 — saved
2026-05-11 14:53:48,491 INFO train_multi TF=ALL epoch 59/100 train=1.9350 val=2.1132 r_mae=0.816 pos_r_acc=0.658 side_acc=0.629 r_n=127469
2026-05-11 14:53:48,497 INFO train_multi TF=ALL: new best r_mae=0.8165 — saved rmae checkpoint
2026-05-11 14:54:03,715 INFO train_multi TF=ALL epoch 60/100 train=1.9310 val=2.0989 r_mae=0.819 pos_r_acc=0.657 side_acc=0.639 r_n=127469
2026-05-11 14:54:18,954 INFO train_multi TF=ALL epoch 61/100 train=1.9193 val=2.1079 r_mae=0.820 pos_r_acc=0.654 side_acc=0.638 r_n=127469
2026-05-11 14:54:34,300 INFO train_multi TF=ALL epoch 62/100 train=1.9172 val=2.1079 r_mae=0.819 pos_r_acc=0.654 side_acc=0.637 r_n=127469
2026-05-11 14:54:49,407 INFO train_multi TF=ALL epoch 63/100 train=1.9100 val=2.1212 r_mae=0.815 pos_r_acc=0.654 side_acc=0.632 r_n=127469
2026-05-11 14:54:49,419 INFO train_multi TF=ALL: new best r_mae=0.8146 — saved rmae checkpoint
2026-05-11 14:55:04,560 INFO train_multi TF=ALL epoch 64/100 train=1.9002 val=2.1117 r_mae=0.828 pos_r_acc=0.649 side_acc=0.641 r_n=127469
2026-05-11 14:55:18,607 INFO train_multi TF=ALL epoch 65/100 train=1.8987 val=2.1046 r_mae=0.813 pos_r_acc=0.659 side_acc=0.637 r_n=127469
2026-05-11 14:55:18,612 INFO train_multi TF=ALL: new best r_mae=0.8133 — saved rmae checkpoint
2026-05-11 14:55:32,478 INFO train_multi TF=ALL epoch 66/100 train=1.8847 val=2.1061 r_mae=0.809 pos_r_acc=0.660 side_acc=0.638 r_n=127469
2026-05-11 14:55:32,483 INFO train_multi TF=ALL: new best r_mae=0.8088 — saved rmae checkpoint
2026-05-11 14:55:46,626 INFO train_multi TF=ALL epoch 67/100 train=1.8835 val=2.1191 r_mae=0.816 pos_r_acc=0.657 side_acc=0.640 r_n=127469
2026-05-11 14:56:00,729 INFO train_multi TF=ALL epoch 68/100 train=1.8799 val=2.1274 r_mae=0.821 pos_r_acc=0.649 side_acc=0.633 r_n=127469
2026-05-11 14:56:14,832 INFO train_multi TF=ALL epoch 69/100 train=1.8713 val=2.1046 r_mae=0.808 pos_r_acc=0.660 side_acc=0.642 r_n=127469
2026-05-11 14:56:14,841 INFO train_multi TF=ALL: new best r_mae=0.8078 — saved rmae checkpoint
2026-05-11 14:56:28,927 INFO train_multi TF=ALL epoch 70/100 train=1.8666 val=2.1032 r_mae=0.811 pos_r_acc=0.659 side_acc=0.640 r_n=127469
2026-05-11 14:56:43,051 INFO train_multi TF=ALL epoch 71/100 train=1.8583 val=2.1058 r_mae=0.811 pos_r_acc=0.658 side_acc=0.643 r_n=127469
2026-05-11 14:56:57,107 INFO train_multi TF=ALL epoch 72/100 train=1.8487 val=2.1036 r_mae=0.813 pos_r_acc=0.656 side_acc=0.642 r_n=127469
2026-05-11 14:57:11,155 INFO train_multi TF=ALL epoch 73/100 train=1.8448 val=2.1201 r_mae=0.816 pos_r_acc=0.655 side_acc=0.643 r_n=127469
2026-05-11 14:57:25,277 INFO train_multi TF=ALL epoch 74/100 train=1.8362 val=2.0992 r_mae=0.812 pos_r_acc=0.657 side_acc=0.646 r_n=127469
2026-05-11 14:57:39,437 INFO train_multi TF=ALL epoch 75/100 train=1.8246 val=2.1241 r_mae=0.812 pos_r_acc=0.655 side_acc=0.644 r_n=127469
2026-05-11 14:57:53,611 INFO train_multi TF=ALL epoch 76/100 train=1.8251 val=2.1253 r_mae=0.814 pos_r_acc=0.654 side_acc=0.641 r_n=127469
2026-05-11 14:58:07,002 INFO train_multi TF=ALL epoch 77/100 train=1.8137 val=2.1067 r_mae=0.813 pos_r_acc=0.657 side_acc=0.646 r_n=127469
2026-05-11 14:58:20,355 INFO train_multi TF=ALL epoch 78/100 train=1.8077 val=2.1399 r_mae=0.821 pos_r_acc=0.649 side_acc=0.640 r_n=127469
2026-05-11 14:58:34,257 INFO train_multi TF=ALL epoch 79/100 train=1.8062 val=2.1405 r_mae=0.818 pos_r_acc=0.650 side_acc=0.639 r_n=127469
2026-05-11 14:58:48,431 INFO train_multi TF=ALL epoch 80/100 train=1.7993 val=2.1209 r_mae=0.814 pos_r_acc=0.653 side_acc=0.649 r_n=127469
2026-05-11 14:59:02,419 INFO train_multi TF=ALL epoch 81/100 train=1.7953 val=2.1304 r_mae=0.814 pos_r_acc=0.655 side_acc=0.644 r_n=127469
2026-05-11 14:59:16,522 INFO train_multi TF=ALL epoch 82/100 train=1.7823 val=2.1165 r_mae=0.814 pos_r_acc=0.653 side_acc=0.649 r_n=127469
2026-05-11 14:59:30,470 INFO train_multi TF=ALL epoch 83/100 train=1.7749 val=2.1352 r_mae=0.816 pos_r_acc=0.653 side_acc=0.646 r_n=127469
2026-05-11 14:59:44,705 INFO train_multi TF=ALL epoch 84/100 train=1.7674 val=2.1359 r_mae=0.814 pos_r_acc=0.655 side_acc=0.648 r_n=127469
2026-05-11 14:59:58,627 INFO train_multi TF=ALL epoch 85/100 train=1.7658 val=2.1556 r_mae=0.818 pos_r_acc=0.652 side_acc=0.642 r_n=127469
2026-05-11 15:00:12,852 INFO train_multi TF=ALL epoch 86/100 train=1.7643 val=2.1625 r_mae=0.816 pos_r_acc=0.654 side_acc=0.642 r_n=127469
2026-05-11 15:00:26,893 INFO train_multi TF=ALL epoch 87/100 train=1.7511 val=2.1496 r_mae=0.817 pos_r_acc=0.653 side_acc=0.647 r_n=127469
2026-05-11 15:00:40,982 INFO train_multi TF=ALL epoch 88/100 train=1.7474 val=2.1603 r_mae=0.816 pos_r_acc=0.654 side_acc=0.642 r_n=127469
2026-05-11 15:00:54,998 INFO train_multi TF=ALL epoch 89/100 train=1.7427 val=2.1420 r_mae=0.815 pos_r_acc=0.653 side_acc=0.647 r_n=127469
2026-05-11 15:01:09,186 INFO train_multi TF=ALL epoch 90/100 train=1.7384 val=2.1575 r_mae=0.816 pos_r_acc=0.653 side_acc=0.642 r_n=127469
2026-05-11 15:01:23,370 INFO train_multi TF=ALL epoch 91/100 train=1.7322 val=2.1747 r_mae=0.818 pos_r_acc=0.652 side_acc=0.647 r_n=127469
2026-05-11 15:01:37,556 INFO train_multi TF=ALL epoch 92/100 train=1.7269 val=2.1667 r_mae=0.819 pos_r_acc=0.652 side_acc=0.642 r_n=127469
2026-05-11 15:01:51,726 INFO train_multi TF=ALL epoch 93/100 train=1.7181 val=2.1926 r_mae=0.817 pos_r_acc=0.651 side_acc=0.641 r_n=127469
2026-05-11 15:02:05,276 INFO train_multi TF=ALL epoch 94/100 train=1.7126 val=2.1633 r_mae=0.818 pos_r_acc=0.651 side_acc=0.647 r_n=127469
2026-05-11 15:02:05,276 INFO train_multi TF=ALL early stop at epoch 94
2026-05-11 15:02:05,288 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 15:02:05,288 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 15:02:05,289 INFO train_multi TF=ALL: promoted r_mae checkpoint (0.8078 < primary 0.8240) — overwriting model.pt
2026-05-11 15:02:06,481 WARNING fit_r_isotonic: long calibrator increases MAE (cal=0.8298 >= raw=0.8133) — skipping
2026-05-11 15:02:06,491 WARNING fit_r_isotonic: short calibrator increases MAE (cal=0.8388 >= raw=0.8324) — skipping
2026-05-11 15:02:06,491 INFO fit_r_isotonic: saved side-R calibrators stats={'long': {'n': 31689, 'raw_mae': 0.8132797861726101, 'calibrated_mae': 0.8298194027771826, 'skipped': 'calibrator_hurts'}, 'short': {'n': 32408, 'raw_mae': 0.8323969878864333, 'calibrated_mae': 0.8388431468680322, 'skipped': 'calibrator_hurts'}}
2026-05-11 15:02:06,625 WARNING GRU validation R-MAE above floor for TF=ALL: best_val_r_mae=0.808 max=0.750. Keeping saved best weights so the pipeline can progress.
2026-05-11 15:02:06,639 INFO GRU R threshold XAUUSD/buy: q25_pos=0.535 q50_pos=1.667 pos_rate=46.0% (n=119259 n_pos=54899)
2026-05-11 15:02:06,645 INFO GRU R threshold XAUUSD/sell: q25_pos=0.528 q50_pos=1.667 pos_rate=45.5% (n=118087 n_pos=53734)
2026-05-11 15:02:06,652 INFO GRU R threshold EURUSD/buy: q25_pos=0.541 q50_pos=1.667 pos_rate=45.6% (n=118751 n_pos=54196)
2026-05-11 15:02:06,658 INFO GRU R threshold EURUSD/sell: q25_pos=0.545 q50_pos=1.667 pos_rate=45.6% (n=117679 n_pos=53721)
2026-05-11 15:02:06,664 INFO GRU R threshold USDJPY/buy: q25_pos=0.551 q50_pos=1.667 pos_rate=46.2% (n=118691 n_pos=54870)
2026-05-11 15:02:06,669 INFO GRU R threshold USDJPY/sell: q25_pos=0.542 q50_pos=1.667 pos_rate=44.6% (n=116030 n_pos=51793)
2026-05-11 15:02:06,675 INFO GRU R threshold EURJPY/buy: q25_pos=0.530 q50_pos=1.416 pos_rate=46.1% (n=118651 n_pos=54654)
2026-05-11 15:02:06,682 INFO GRU R threshold EURJPY/sell: q25_pos=0.525 q50_pos=1.667 pos_rate=44.7% (n=117320 n_pos=52399)
2026-05-11 15:02:06,688 INFO GRU R threshold GBPJPY/buy: q25_pos=0.517 q50_pos=1.443 pos_rate=45.9% (n=118277 n_pos=54278)
2026-05-11 15:02:06,694 INFO GRU R threshold GBPJPY/sell: q25_pos=0.521 q50_pos=1.667 pos_rate=45.3% (n=116299 n_pos=52702)
2026-05-11 15:02:06,700 INFO GRU R threshold GBPUSD/buy: q25_pos=0.524 q50_pos=1.667 pos_rate=45.8% (n=117764 n_pos=53894)
2026-05-11 15:02:06,706 INFO GRU R threshold GBPUSD/sell: q25_pos=0.533 q50_pos=1.667 pos_rate=45.8% (n=117404 n_pos=53785)
2026-05-11 15:02:06,707 INFO GRU per-symbol R thresholds saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/symbol_r_thresholds.json (6 symbols)
2026-05-11 15:02:06,727 INFO Retrain complete. Total wall-clock: 1418.9s
  DONE  Retrain gru [train-split retrain]
  START Retrain regime [train-split retrain]
2026-05-11 15:02:10,262 INFO retrain environment: KAGGLE
2026-05-11 15:02:11,900 INFO Device: CUDA (2 GPU(s))
2026-05-11 15:02:11,908 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 15:02:11,908 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 15:02:11,908 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 15:02:11,909 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 15:02:11,909 INFO Retrain data split: train
2026-05-11 15:02:11,909 INFO Retrain rolling fold selector: latest
2026-05-11 15:02:11,910 INFO === RegimeClassifier retrain (hierarchical: HTF 2-score bias + LTF 5-score behaviour) ===
2026-05-11 15:02:12,063 INFO NumExpr defaulting to 4 threads.
2026-05-11 15:02:12,277 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-11 15:02:12,277 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 15:02:12,277 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 15:02:12,278 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-11 15:02:12,340 INFO Regime rolling folds selected: [None]
2026-05-11 15:02:12,340 INFO === Regime rolling fold 1/1: train_all ===
2026-05-11 15:02:12,340 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-11 15:02:12,382 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 15:02:12,384 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
2026-05-11 15:02:12,399 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
2026-05-11 15:02:12,416 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
2026-05-11 15:02:12,432 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
2026-05-11 15:02:12,450 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
2026-05-11 15:02:12,465 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 15:02:12,713 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 15:02:12,783 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 15:02:12,809 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 15:02:12,810 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 15:02:12,822 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 15:02:12,823 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 15:02:13,627 INFO macro_cache: using training data dir /kaggle/working/Multi-Bot/trading-system/training_data
2026-05-11 15:02:13,742 INFO macro_cache loaded 19 series: {'asx200': '2842 bars 2015-01-02→2026-03-27', 'cac40': '2876 bars 2015-01-02→2026-03-27', 'dax': '2851 bars 2015-01-02→2026-03-27', 'djia': '2825 bars 2015-01-02→2026-03-27', 'dxy': '2826 bars 2015-01-02→2026-03-27', 'eurostoxx': '2823 bars 2015-01-05→2026-03-27', 'ftse': '2839 bars 2015-01-02→2026-03-27', 'gold_fut': '2824 bars 2015-01-02→2026-03-27', 'hsi': '2764 bars 2015-01-02→2026-03-27', 'nasdaq': '2825 bars 2015-01-02→2026-03-27', 'nikkei': '2744 bars 2015-01-05→2026-03-27', 'oil_fut': '2825 bars 2015-01-02→2026-03-27', 'spx': '2825 bars 2015-01-02→2026-03-27', 'us10y': '2824 bars 2015-01-02→2026-03-27', 'us30y': '2824 bars 2015-01-02→2026-03-27', 'us3m': '2824 bars 2015-01-02→2026-03-27', 'vix': '2825 bars 2015-01-02→2026-03-27', 'us10y_fred': '2607 bars 2016-03-28→2026-03-24', 'us2y_fred': '2607 bars 2016-03-28→2026-03-24'}
2026-05-11 15:02:13,993 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1142, 'BIAS_DOWN': 522, 'BIAS_NEUTRAL': 10438}  ambiguous=4182 (total=12102) horizon=84
2026-05-11 15:02:13,998 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected XAUUSD — 12052 samples (group=gold) score_means={'bias_up_score': 0.0948, 'bias_down_score': 0.0433} labels={'BIAS_UP': 1142, 'BIAS_DOWN': 522, 'BIAS_NEUTRAL': 10388} clean={'BIAS_UP': 1142, 'BIAS_DOWN': 522, 'BIAS_NEUTRAL': 6216}
2026-05-11 15:02:14,173 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:14,210 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:14,229 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:14,230 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:14,238 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:14,240 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 15:02:15,212 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 690, 'BIAS_DOWN': 540, 'BIAS_NEUTRAL': 10174}  ambiguous=3886 (total=11404) horizon=84
2026-05-11 15:02:15,217 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURUSD — 11354 samples (group=dollar) score_means={'bias_up_score': 0.0608, 'bias_down_score': 0.0476} labels={'BIAS_UP': 690, 'BIAS_DOWN': 540, 'BIAS_NEUTRAL': 10124} clean={'BIAS_UP': 690, 'BIAS_DOWN': 540, 'BIAS_NEUTRAL': 6257}
2026-05-11 15:02:15,380 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:15,416 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:15,437 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:15,437 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:15,445 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:15,446 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 15:02:16,398 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 826, 'BIAS_DOWN': 423, 'BIAS_NEUTRAL': 10154}  ambiguous=4036 (total=11403) horizon=84
2026-05-11 15:02:16,403 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected USDJPY — 11353 samples (group=dollar) score_means={'bias_up_score': 0.0728, 'bias_down_score': 0.0373} labels={'BIAS_UP': 826, 'BIAS_DOWN': 423, 'BIAS_NEUTRAL': 10104} clean={'BIAS_UP': 826, 'BIAS_DOWN': 423, 'BIAS_NEUTRAL': 6078}
2026-05-11 15:02:16,560 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:16,595 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:16,617 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:16,618 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:16,626 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:16,627 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 15:02:17,597 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 681, 'BIAS_DOWN': 527, 'BIAS_NEUTRAL': 10199}  ambiguous=4044 (total=11407) horizon=84
2026-05-11 15:02:17,602 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURJPY — 11357 samples (group=cross) score_means={'bias_up_score': 0.06, 'bias_down_score': 0.0464} labels={'BIAS_UP': 681, 'BIAS_DOWN': 527, 'BIAS_NEUTRAL': 10149} clean={'BIAS_UP': 681, 'BIAS_DOWN': 527, 'BIAS_NEUTRAL': 6111}
2026-05-11 15:02:17,759 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:17,796 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:17,817 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:17,817 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:17,827 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:17,828 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 15:02:18,860 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 839, 'BIAS_DOWN': 579, 'BIAS_NEUTRAL': 9990}  ambiguous=4240 (total=11408) horizon=84
2026-05-11 15:02:18,866 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPJPY — 11358 samples (group=cross) score_means={'bias_up_score': 0.0739, 'bias_down_score': 0.051} labels={'BIAS_UP': 839, 'BIAS_DOWN': 579, 'BIAS_NEUTRAL': 9940} clean={'BIAS_UP': 839, 'BIAS_DOWN': 579, 'BIAS_NEUTRAL': 5723}
2026-05-11 15:02:19,034 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:19,069 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:19,090 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:19,090 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:19,099 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:19,100 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 15:02:20,069 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 502, 'BIAS_DOWN': 720, 'BIAS_NEUTRAL': 10180}  ambiguous=4113 (total=11402) horizon=84
2026-05-11 15:02:20,074 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPUSD — 11352 samples (group=dollar) score_means={'bias_up_score': 0.0442, 'bias_down_score': 0.0623} labels={'BIAS_UP': 502, 'BIAS_DOWN': 707, 'BIAS_NEUTRAL': 10143} clean={'BIAS_UP': 502, 'BIAS_DOWN': 707, 'BIAS_NEUTRAL': 6056}
2026-05-11 15:02:20,142 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 1520, 'BIAS_DOWN': 1106, 'BIAS_NEUTRAL': 20089}, 'dollar': {'BIAS_UP': 2018, 'BIAS_DOWN': 1670, 'BIAS_NEUTRAL': 30371}, 'gold': {'BIAS_UP': 1142, 'BIAS_DOWN': 522, 'BIAS_NEUTRAL': 10388}}
2026-05-11 15:02:20,142 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0669, 'bias_down_score': 0.0487}, 'dollar': {'bias_up_score': 0.0593, 'bias_down_score': 0.049}, 'gold': {'bias_up_score': 0.0948, 'bias_down_score': 0.0433}}
2026-05-11 15:02:20,142 INFO Regime[4H mode=htf_bias] label distribution by year: {2016: {'BIAS_UP': 491, 'BIAS_DOWN': 576, 'BIAS_NEUTRAL': 7755}, 2017: {'BIAS_UP': 734, 'BIAS_DOWN': 286, 'BIAS_NEUTRAL': 8093}, 2018: {'BIAS_UP': 427, 'BIAS_DOWN': 714, 'BIAS_NEUTRAL': 7989}, 2019: {'BIAS_UP': 410, 'BIAS_DOWN': 447, 'BIAS_NEUTRAL': 8245}, 2020: {'BIAS_UP': 694, 'BIAS_DOWN': 174, 'BIAS_NEUTRAL': 8243}, 2021: {'BIAS_UP': 722, 'BIAS_DOWN': 473, 'BIAS_NEUTRAL': 7896}, 2022: {'BIAS_UP': 667, 'BIAS_DOWN': 519, 'BIAS_NEUTRAL': 7935}, 2023: {'BIAS_UP': 535, 'BIAS_DOWN': 109, 'BIAS_NEUTRAL': 4692}}
2026-05-11 15:02:20,142 INFO Regime[4H mode=htf_bias] score means by year: {2016: {'bias_up_score': 0.0557, 'bias_down_score': 0.0653}, 2017: {'bias_up_score': 0.0805, 'bias_down_score': 0.0314}, 2018: {'bias_up_score': 0.0468, 'bias_down_score': 0.0782}, 2019: {'bias_up_score': 0.045, 'bias_down_score': 0.0491}, 2020: {'bias_up_score': 0.0762, 'bias_down_score': 0.0191}, 2021: {'bias_up_score': 0.0794, 'bias_down_score': 0.052}, 2022: {'bias_up_score': 0.0731, 'bias_down_score': 0.0569}, 2023: {'bias_up_score': 0.1003, 'bias_down_score': 0.0204}}
2026-05-11 15:02:20,193 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
2026-05-11 15:02:20,194 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
2026-05-11 15:02:20,195 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
2026-05-11 15:02:20,195 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 15:02:20,196 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 15:02:20,197 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
2026-05-11 15:02:20,213 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 15:02:20,216 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 15:02:20,217 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 15:02:20,218 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 15:02:20,218 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 15:02:20,219 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 15:02:20,808 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 157, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1339}  ambiguous=566 (total=1581) horizon=84
2026-05-11 15:02:20,812 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected XAUUSD — 1531 samples (group=gold) score_means={'bias_up_score': 0.1025, 'bias_down_score': 0.0555} labels={'BIAS_UP': 157, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1289} clean={'BIAS_UP': 157, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 744}
2026-05-11 15:02:20,915 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:20,918 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:20,919 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:20,920 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:20,920 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:20,921 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 15:02:21,589 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 135, 'BIAS_DOWN': 66, 'BIAS_NEUTRAL': 1290}  ambiguous=531 (total=1491) horizon=84
2026-05-11 15:02:21,591 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURUSD — 1441 samples (group=dollar) score_means={'bias_up_score': 0.0937, 'bias_down_score': 0.0458} labels={'BIAS_UP': 135, 'BIAS_DOWN': 66, 'BIAS_NEUTRAL': 1240} clean={'BIAS_UP': 135, 'BIAS_DOWN': 66, 'BIAS_NEUTRAL': 739}
2026-05-11 15:02:21,664 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:21,666 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:21,667 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:21,667 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:21,668 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:21,669 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 15:02:22,233 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 164, 'BIAS_DOWN': 77, 'BIAS_NEUTRAL': 1248}  ambiguous=616 (total=1489) horizon=84
2026-05-11 15:02:22,236 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected USDJPY — 1439 samples (group=dollar) score_means={'bias_up_score': 0.114, 'bias_down_score': 0.0535} labels={'BIAS_UP': 164, 'BIAS_DOWN': 77, 'BIAS_NEUTRAL': 1198} clean={'BIAS_UP': 164, 'BIAS_DOWN': 77, 'BIAS_NEUTRAL': 608}
2026-05-11 15:02:22,312 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:22,314 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:22,315 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:22,315 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:22,315 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:22,316 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 15:02:22,871 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 123, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 1366}  ambiguous=582 (total=1494) horizon=84
2026-05-11 15:02:22,874 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0852, 'bias_down_score': 0.0035} labels={'BIAS_UP': 123, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 1316} clean={'BIAS_UP': 123, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 741}
2026-05-11 15:02:22,955 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:22,957 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:22,958 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:22,958 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:22,959 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:22,960 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 15:02:23,532 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 129, 'BIAS_DOWN': 9, 'BIAS_NEUTRAL': 1356}  ambiguous=551 (total=1494) horizon=84
2026-05-11 15:02:23,535 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0893, 'bias_down_score': 0.0055} labels={'BIAS_UP': 129, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 1307} clean={'BIAS_UP': 129, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 775}
2026-05-11 15:02:23,619 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:23,621 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:23,622 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:23,623 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:23,623 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:23,624 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 15:02:24,260 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 81, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 1316}  ambiguous=560 (total=1488) horizon=84
2026-05-11 15:02:24,264 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPUSD — 1438 samples (group=dollar) score_means={'bias_up_score': 0.0563, 'bias_down_score': 0.0633} labels={'BIAS_UP': 81, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 1266} clean={'BIAS_UP': 81, 'BIAS_DOWN': 91, 'BIAS_NEUTRAL': 735}
2026-05-11 15:02:24,337 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 252, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 2623}, 'dollar': {'BIAS_UP': 380, 'BIAS_DOWN': 234, 'BIAS_NEUTRAL': 3704}, 'gold': {'BIAS_UP': 157, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1289}}
2026-05-11 15:02:24,337 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0873, 'bias_down_score': 0.0045}, 'dollar': {'bias_up_score': 0.088, 'bias_down_score': 0.0542}, 'gold': {'bias_up_score': 0.1025, 'bias_down_score': 0.0555}}
2026-05-11 15:02:24,337 INFO Regime[4H mode=htf_bias] label distribution by year: {2022: {'BIAS_UP': 258, 'BIAS_DOWN': 228, 'BIAS_NEUTRAL': 2915}, 2023: {'BIAS_UP': 531, 'BIAS_DOWN': 104, 'BIAS_NEUTRAL': 4701}}
2026-05-11 15:02:24,338 INFO Regime[4H mode=htf_bias] score means by year: {2022: {'bias_up_score': 0.0759, 'bias_down_score': 0.067}, 2023: {'bias_up_score': 0.0995, 'bias_down_score': 0.0195}}
2026-05-11 15:02:24,391 INFO Regime phase HTF dataset build fold=train_all: 12.1s (train=68826 val=8737)
2026-05-11 15:02:24,392 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260511_150224
2026-05-11 15:02:24,591 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=51, n_classes=2)
2026-05-11 15:02:24,591 INFO Regime 4H/htf_bias warm start enabled from existing weights
2026-05-11 15:02:24,606 INFO RegimeClassifier[mode=htf_bias]: HTF clean-label fit filter kept train=44419/68826 val=5463/8737 at conf>=0.40 train_counts={'BIAS_UP': 4680, 'BIAS_DOWN': 3298, 'BIAS_NEUTRAL': 36441} val_counts={'BIAS_UP': 789, 'BIAS_DOWN': 332, 'BIAS_NEUTRAL': 4342}
2026-05-11 15:02:24,607 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=44419 val=5463 train_labels={'BIAS_UP': 4680, 'BIAS_DOWN': 3298, 'BIAS_NEUTRAL': 36441} val_labels={'BIAS_UP': 789, 'BIAS_DOWN': 332, 'BIAS_NEUTRAL': 4342}
2026-05-11 15:02:24,608 INFO RegimeClassifier[mode=htf_bias]: warm start HTF score head
2026-05-11 15:02:24,608 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-11 15:02:24,608 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 8.491, 'bias_down_score': 12.0}
2026-05-11 15:02:24,613 INFO RegimeClassifier[mode=htf_bias]: HTF balanced sampler — dir=7978 neutral=36441 dir_weight=3 => dir_frac_per_epoch≈47.2%
2026-05-11 15:02:28,242 INFO Regime HTF score epoch  1/50 — tr=2.7533 va=1.0420 acc=0.802 bal=0.359 threshold=0.80 margin=0.15 recall={'BIAS_UP': 0.037, 'BIAS_DOWN': 0.042, 'BIAS_NEUTRAL': 0.999} precision={'BIAS_UP': 1.0, 'BIAS_DOWN': 0.824, 'BIAS_NEUTRAL': 0.801}
2026-05-11 15:02:29,683 INFO Regime HTF score epoch  2/50 — tr=2.7299 va=1.0392 bal=0.359
2026-05-11 15:02:31,062 INFO Regime HTF score epoch  3/50 — tr=2.6945 va=1.0291 bal=0.356
2026-05-11 15:02:32,471 INFO Regime HTF score epoch  4/50 — tr=2.6723 va=1.0132 bal=0.357
2026-05-11 15:02:33,880 INFO Regime HTF score epoch  5/50 — tr=2.6038 va=0.9837 acc=0.802 bal=0.363 threshold=0.35 margin=0.80 recall={'BIAS_UP': 0.049, 'BIAS_DOWN': 0.042, 'BIAS_NEUTRAL': 0.997} precision={'BIAS_UP': 0.929, 'BIAS_DOWN': 0.609, 'BIAS_NEUTRAL': 0.802}
2026-05-11 15:02:35,272 INFO Regime HTF score epoch  6/50 — tr=2.5682 va=0.9512 bal=0.355
2026-05-11 15:02:36,718 INFO Regime HTF score epoch  7/50 — tr=2.4994 va=0.9221 bal=0.355
2026-05-11 15:02:38,143 INFO Regime HTF score epoch  8/50 — tr=2.3934 va=0.8886 bal=0.354
2026-05-11 15:02:39,577 INFO Regime HTF score epoch  9/50 — tr=2.2843 va=0.8614 bal=0.357
2026-05-11 15:02:40,949 INFO Regime HTF score epoch 10/50 — tr=2.2269 va=0.8339 acc=0.802 bal=0.365 threshold=0.88 margin=0.15 recall={'BIAS_UP': 0.039, 'BIAS_DOWN': 0.057, 'BIAS_NEUTRAL': 0.997} precision={'BIAS_UP': 1.0, 'BIAS_DOWN': 0.594, 'BIAS_NEUTRAL': 0.802}
2026-05-11 15:02:42,342 INFO Regime HTF score epoch 11/50 — tr=2.1419 va=0.8106 bal=0.352
2026-05-11 15:02:42,342 INFO Regime HTF score early stop at epoch 11
2026-05-11 15:02:43,633 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.350 margin=0.800 precision={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.801} recall={'BIAS_UP': 0.037, 'BIAS_DOWN': 0.03, 'BIAS_NEUTRAL': 1.0} f1={'BIAS_UP': 0.071, 'BIAS_DOWN': 0.058, 'BIAS_NEUTRAL': 0.889} confusion=[[29, 0, 760], [0, 10, 322], [0, 0, 4342]] score_mae={'bias_up_score': 0.2048, 'bias_down_score': 0.1332} pred_share={'BIAS_UP': 0.0053, 'BIAS_DOWN': 0.0018, 'BIAS_NEUTRAL': 0.9929}
2026-05-11 15:02:43,634 WARNING Regime HTF score prediction distribution collapsed: pred_share={'BIAS_UP': 0.0053, 'BIAS_DOWN': 0.0018, 'BIAS_NEUTRAL': 0.9929}, max_pred_share=99.3%, collapsed_classes=['BIAS_DOWN']. Saving weights anyway so the pipeline can progress.
2026-05-11 15:02:43,635 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.801} min_precision=0.500 recall={'BIAS_UP': 0.037, 'BIAS_DOWN': 0.03, 'BIAS_NEUTRAL': 1.0} min_recall=0.100 f1={'BIAS_UP': 0.071, 'BIAS_DOWN': 0.058, 'BIAS_NEUTRAL': 0.889} min_f1=0.150 min_neutral_recall=0.500 weak_precision=[] weak_recall=['BIAS_UP', 'BIAS_DOWN'] weak_f1=['BIAS_UP', 'BIAS_DOWN'] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-11 15:02:43,638 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 15:02:43,638 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 15:02:43,639 INFO Regime phase HTF train fold=train_all: 19.0s
2026-05-11 15:02:43,753 INFO Regime HTF complete fold=train_all: acc=0.802 bal=0.356 train=68826 val=8737 per_class={'BIAS_UP': 0.037, 'BIAS_DOWN': 0.03, 'BIAS_NEUTRAL': 1.0} precision={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.801} threshold=0.350 margin=0.800
2026-05-11 15:02:43,754 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 15:02:43,957 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 502, 'BIAS_DOWN': 720, 'BIAS_NEUTRAL': 10180}  ambiguous=4113 (total=11402) horizon=84
2026-05-11 15:02:43,960 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on GBPUSD 4H:
{'BIAS_UP': 4.482142857142857, 'BIAS_DOWN': 5.669291338582677, 'BIAS_NEUTRAL': 42.416666666666664}
2026-05-11 15:02:43,964 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (all labels):
{'BIAS_UP': {'n': 502, 'mean': 0.00043281055222645396, 'mean_over_std': 0.19321420748230153}, 'BIAS_DOWN': {'n': 720, 'mean': -0.00046308823082979826, 'mean_over_std': -0.19538165843530478}, 'BIAS_NEUTRAL': {'n': 10179, 'mean': 7.477567618138561e-07, 'mean_over_std': 0.0002829536380249001}}
2026-05-11 15:02:43,964 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 502, 'mean': 0.00043281055222645396, 'mean_over_std': 0.19321420748230153}, 'BIAS_DOWN': {'n': 720, 'mean': -0.00046308823082979826, 'mean_over_std': -0.19538165843530478}, 'BIAS_NEUTRAL': {'n': 6067, 'mean': 9.596616495197703e-06, 'mean_over_std': 0.004013656697571348}}
2026-05-11 15:02:43,970 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-11 15:02:43,973 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
2026-05-11 15:02:43,975 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
2026-05-11 15:02:43,977 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
2026-05-11 15:02:43,979 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
2026-05-11 15:02:43,981 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
2026-05-11 15:02:43,983 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 15:02:43,999 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 15:02:44,006 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 15:02:44,008 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 15:02:44,009 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 15:02:44,010 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 15:02:44,016 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 15:02:45,187 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected XAUUSD — 44227 samples (group=gold) score_means={'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}
2026-05-11 15:02:45,305 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:45,308 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:45,308 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:45,309 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:45,309 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:45,312 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 15:02:46,374 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURUSD — 43689 samples (group=dollar) score_means={'trend_score': 0.49, 'range_score': 0.236, 'chop_score': 0.4622, 'volatility_percentile': 0.3812, 'consolidation_score': 0.1862}
2026-05-11 15:02:46,495 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:46,497 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:46,498 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:46,498 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:46,499 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:46,501 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 15:02:47,568 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected USDJPY — 43686 samples (group=dollar) score_means={'trend_score': 0.4949, 'range_score': 0.2312, 'chop_score': 0.4613, 'volatility_percentile': 0.38, 'consolidation_score': 0.1931}
2026-05-11 15:02:47,696 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:47,698 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:47,699 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:47,699 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:47,700 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:47,702 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 15:02:48,804 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURJPY — 43688 samples (group=cross) score_means={'trend_score': 0.4888, 'range_score': 0.2345, 'chop_score': 0.4662, 'volatility_percentile': 0.3831, 'consolidation_score': 0.1882}
2026-05-11 15:02:48,932 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:48,935 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:48,936 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:48,936 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:48,936 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:48,939 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 15:02:50,004 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPJPY — 43678 samples (group=cross) score_means={'trend_score': 0.4878, 'range_score': 0.2348, 'chop_score': 0.4673, 'volatility_percentile': 0.3839, 'consolidation_score': 0.1904}
2026-05-11 15:02:50,122 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:50,125 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:50,125 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:50,126 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:50,126 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:50,129 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 15:02:51,195 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPUSD — 43676 samples (group=dollar) score_means={'trend_score': 0.4923, 'range_score': 0.2335, 'chop_score': 0.4611, 'volatility_percentile': 0.3832, 'consolidation_score': 0.1856}
2026-05-11 15:02:51,328 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2346, 'chop_score': 0.4667, 'volatility_percentile': 0.3835, 'consolidation_score': 0.1893}, 'dollar': {'trend_score': 0.4924, 'range_score': 0.2335, 'chop_score': 0.4615, 'volatility_percentile': 0.3815, 'consolidation_score': 0.1883}, 'gold': {'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}}
2026-05-11 15:02:51,328 INFO Regime[1H mode=ltf_behaviour] score means by year: {2016: {'trend_score': 0.495, 'range_score': 0.233, 'chop_score': 0.4607, 'volatility_percentile': 0.3776, 'consolidation_score': 0.192}, 2017: {'trend_score': 0.5017, 'range_score': 0.2328, 'chop_score': 0.4574, 'volatility_percentile': 0.3689, 'consolidation_score': 0.1942}, 2018: {'trend_score': 0.4886, 'range_score': 0.237, 'chop_score': 0.4655, 'volatility_percentile': 0.3947, 'consolidation_score': 0.1785}, 2019: {'trend_score': 0.4817, 'range_score': 0.236, 'chop_score': 0.4691, 'volatility_percentile': 0.3725, 'consolidation_score': 0.198}, 2020: {'trend_score': 0.4874, 'range_score': 0.2311, 'chop_score': 0.4674, 'volatility_percentile': 0.3848, 'consolidation_score': 0.1919}, 2021: {'trend_score': 0.4872, 'range_score': 0.2372, 'chop_score': 0.4672, 'volatility_percentile': 0.377, 'consolidation_score': 0.192}, 2022: {'trend_score': 0.4893, 'range_score': 0.2357, 'chop_score': 0.4647, 'volatility_percentile': 0.3962, 'consolidation_score': 0.1784}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 15:02:51,443 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
2026-05-11 15:02:51,445 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 15:02:51,446 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
2026-05-11 15:02:51,448 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 15:02:51,449 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
2026-05-11 15:02:51,451 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
2026-05-11 15:02:51,460 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 15:02:51,464 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 15:02:51,465 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 15:02:51,465 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 15:02:51,466 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 15:02:51,468 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 15:02:51,847 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected XAUUSD — 5117 samples (group=gold) score_means={'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}
2026-05-11 15:02:51,977 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:51,981 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:51,982 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:51,983 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:51,983 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:51,985 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 15:02:52,312 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURUSD — 5053 samples (group=dollar) score_means={'trend_score': 0.4883, 'range_score': 0.2413, 'chop_score': 0.463, 'volatility_percentile': 0.3837, 'consolidation_score': 0.1793}
2026-05-11 15:02:52,431 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:52,434 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:52,434 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:52,435 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:52,435 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:52,437 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 15:02:52,768 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected USDJPY — 5046 samples (group=dollar) score_means={'trend_score': 0.5085, 'range_score': 0.2259, 'chop_score': 0.4506, 'volatility_percentile': 0.3809, 'consolidation_score': 0.1888}
2026-05-11 15:02:52,885 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:52,887 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:52,888 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:52,888 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:52,889 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:52,890 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 15:02:53,263 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURJPY — 5053 samples (group=cross) score_means={'trend_score': 0.5023, 'range_score': 0.2261, 'chop_score': 0.4524, 'volatility_percentile': 0.3837, 'consolidation_score': 0.18}
2026-05-11 15:02:53,381 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:53,383 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:53,384 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:53,385 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:53,385 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:53,387 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 15:02:53,719 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPJPY — 5043 samples (group=cross) score_means={'trend_score': 0.4743, 'range_score': 0.2372, 'chop_score': 0.4716, 'volatility_percentile': 0.3777, 'consolidation_score': 0.1917}
2026-05-11 15:02:53,835 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:53,837 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:53,838 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:53,839 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:53,839 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 15:02:53,841 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 15:02:54,179 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPUSD — 5040 samples (group=dollar) score_means={'trend_score': 0.4843, 'range_score': 0.2371, 'chop_score': 0.4614, 'volatility_percentile': 0.377, 'consolidation_score': 0.1835}
2026-05-11 15:02:54,307 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2317, 'chop_score': 0.462, 'volatility_percentile': 0.3807, 'consolidation_score': 0.1858}, 'dollar': {'trend_score': 0.4937, 'range_score': 0.2348, 'chop_score': 0.4583, 'volatility_percentile': 0.3806, 'consolidation_score': 0.1839}, 'gold': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}}
2026-05-11 15:02:54,307 INFO Regime[1H mode=ltf_behaviour] score means by year: {2022: {'trend_score': 0.4905, 'range_score': 0.2364, 'chop_score': 0.4625, 'volatility_percentile': 0.385, 'consolidation_score': 0.1797}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 15:02:54,430 INFO Regime phase LTF dataset build fold=train_all: 10.5s (train=262644 val=30352)
2026-05-11 15:02:54,430 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260511_150254
2026-05-11 15:02:54,435 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=53, n_classes=5)
2026-05-11 15:02:54,435 INFO Regime 1H/ltf_behaviour warm start enabled from existing weights
2026-05-11 15:02:54,469 INFO RegimeClassifier[mode=ltf_behaviour]: warm start score head
2026-05-11 15:02:54,469 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-11 15:02:55,027 INFO Regime score epoch  1/50 — tr=0.0038 va=0.0010 mae={'trend_score': 0.0203, 'range_score': 0.0354, 'chop_score': 0.022, 'volatility_percentile': 0.0166, 'consolidation_score': 0.0224}
2026-05-11 15:02:55,608 INFO Regime score epoch  2/50 — tr=0.0039 va=0.0010
2026-05-11 15:02:56,135 INFO Regime score epoch  3/50 — tr=0.0038 va=0.0010
2026-05-11 15:02:56,694 INFO Regime score epoch  4/50 — tr=0.0038 va=0.0010
2026-05-11 15:02:57,240 INFO Regime score epoch  5/50 — tr=0.0038 va=0.0010 mae={'trend_score': 0.0201, 'range_score': 0.0352, 'chop_score': 0.0221, 'volatility_percentile': 0.0165, 'consolidation_score': 0.0218}
2026-05-11 15:02:57,758 INFO Regime score epoch  6/50 — tr=0.0038 va=0.0010
2026-05-11 15:02:58,339 INFO Regime score epoch  7/50 — tr=0.0038 va=0.0010
2026-05-11 15:02:58,906 INFO Regime score epoch  8/50 — tr=0.0038 va=0.0010
2026-05-11 15:02:59,504 INFO Regime score epoch  9/50 — tr=0.0038 va=0.0010
2026-05-11 15:03:00,026 INFO Regime score epoch 10/50 — tr=0.0037 va=0.0010 mae={'trend_score': 0.0196, 'range_score': 0.0339, 'chop_score': 0.0213, 'volatility_percentile': 0.0162, 'consolidation_score': 0.021}
2026-05-11 15:03:00,583 INFO Regime score epoch 11/50 — tr=0.0037 va=0.0010
2026-05-11 15:03:01,130 INFO Regime score epoch 12/50 — tr=0.0037 va=0.0010
2026-05-11 15:03:01,646 INFO Regime score epoch 13/50 — tr=0.0037 va=0.0009
2026-05-11 15:03:02,210 INFO Regime score epoch 14/50 — tr=0.0037 va=0.0009
2026-05-11 15:03:02,742 INFO Regime score epoch 15/50 — tr=0.0036 va=0.0009 mae={'trend_score': 0.0191, 'range_score': 0.0336, 'chop_score': 0.0203, 'volatility_percentile': 0.0157, 'consolidation_score': 0.0204}
2026-05-11 15:03:03,343 INFO Regime score epoch 16/50 — tr=0.0036 va=0.0009
2026-05-11 15:03:03,856 INFO Regime score epoch 17/50 — tr=0.0036 va=0.0009
2026-05-11 15:03:04,449 INFO Regime score epoch 18/50 — tr=0.0036 va=0.0009
2026-05-11 15:03:04,968 INFO Regime score epoch 19/50 — tr=0.0036 va=0.0009
2026-05-11 15:03:05,528 INFO Regime score epoch 20/50 — tr=0.0036 va=0.0009 mae={'trend_score': 0.0188, 'range_score': 0.0329, 'chop_score': 0.0197, 'volatility_percentile': 0.0154, 'consolidation_score': 0.0203}
2026-05-11 15:03:06,068 INFO Regime score epoch 21/50 — tr=0.0035 va=0.0009
2026-05-11 15:03:06,582 INFO Regime score epoch 22/50 — tr=0.0035 va=0.0009
2026-05-11 15:03:07,137 INFO Regime score epoch 23/50 — tr=0.0035 va=0.0009
2026-05-11 15:03:07,643 INFO Regime score epoch 24/50 — tr=0.0035 va=0.0009
2026-05-11 15:03:08,237 INFO Regime score epoch 25/50 — tr=0.0035 va=0.0009 mae={'trend_score': 0.0182, 'range_score': 0.0322, 'chop_score': 0.0197, 'volatility_percentile': 0.0152, 'consolidation_score': 0.0202}
2026-05-11 15:03:08,789 INFO Regime score epoch 26/50 — tr=0.0035 va=0.0008
2026-05-11 15:03:09,377 INFO Regime score epoch 27/50 — tr=0.0035 va=0.0008
2026-05-11 15:03:09,916 INFO Regime score epoch 28/50 — tr=0.0035 va=0.0008
2026-05-11 15:03:10,446 INFO Regime score epoch 29/50 — tr=0.0035 va=0.0008
2026-05-11 15:03:11,005 INFO Regime score epoch 30/50 — tr=0.0035 va=0.0008 mae={'trend_score': 0.0178, 'range_score': 0.0322, 'chop_score': 0.0195, 'volatility_percentile': 0.0147, 'consolidation_score': 0.0199}
2026-05-11 15:03:11,536 INFO Regime score epoch 31/50 — tr=0.0034 va=0.0008
2026-05-11 15:03:12,111 INFO Regime score epoch 32/50 — tr=0.0034 va=0.0008
2026-05-11 15:03:12,654 INFO Regime score epoch 33/50 — tr=0.0034 va=0.0008
2026-05-11 15:03:13,236 INFO Regime score epoch 34/50 — tr=0.0034 va=0.0008
2026-05-11 15:03:13,770 INFO Regime score epoch 35/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.0175, 'range_score': 0.0318, 'chop_score': 0.019, 'volatility_percentile': 0.015, 'consolidation_score': 0.0198}
2026-05-11 15:03:14,341 INFO Regime score epoch 36/50 — tr=0.0034 va=0.0008
2026-05-11 15:03:14,857 INFO Regime score epoch 37/50 — tr=0.0034 va=0.0008
2026-05-11 15:03:15,399 INFO Regime score epoch 38/50 — tr=0.0034 va=0.0008
2026-05-11 15:03:15,953 INFO Regime score epoch 39/50 — tr=0.0034 va=0.0008
2026-05-11 15:03:16,480 INFO Regime score epoch 40/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.0175, 'range_score': 0.0318, 'chop_score': 0.0191, 'volatility_percentile': 0.0149, 'consolidation_score': 0.0195}
2026-05-11 15:03:17,061 INFO Regime score epoch 41/50 — tr=0.0034 va=0.0008
2026-05-11 15:03:17,585 INFO Regime score epoch 42/50 — tr=0.0034 va=0.0008
2026-05-11 15:03:18,161 INFO Regime score epoch 43/50 — tr=0.0034 va=0.0008
2026-05-11 15:03:18,747 INFO Regime score epoch 44/50 — tr=0.0034 va=0.0008
2026-05-11 15:03:19,311 INFO Regime score epoch 45/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.0174, 'range_score': 0.0318, 'chop_score': 0.019, 'volatility_percentile': 0.0153, 'consolidation_score': 0.0201}
2026-05-11 15:03:19,869 INFO Regime score epoch 46/50 — tr=0.0034 va=0.0008
2026-05-11 15:03:20,421 INFO Regime score epoch 47/50 — tr=0.0034 va=0.0008
2026-05-11 15:03:20,974 INFO Regime score epoch 48/50 — tr=0.0034 va=0.0008
2026-05-11 15:03:21,508 INFO Regime score epoch 49/50 — tr=0.0034 va=0.0008
2026-05-11 15:03:22,076 INFO Regime score epoch 50/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.0174, 'range_score': 0.032, 'chop_score': 0.0191, 'volatility_percentile': 0.0147, 'consolidation_score': 0.0195}
2026-05-11 15:03:22,097 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0175, 'range_score': 0.032, 'chop_score': 0.0188, 'volatility_percentile': 0.0145, 'consolidation_score': 0.0194} mse={'trend_score': 0.00054, 'range_score': 0.00169, 'chop_score': 0.00058, 'volatility_percentile': 0.00038, 'consolidation_score': 0.00087} corr={'trend_score': 0.9946, 'range_score': 0.9585, 'chop_score': 0.9922, 'volatility_percentile': 0.996, 'consolidation_score': 0.991} pred_std={'trend_score': 0.2193, 'range_score': 0.1324, 'chop_score': 0.1817, 'volatility_percentile': 0.2178, 'consolidation_score': 0.2129} target_std={'trend_score': 0.2219, 'range_score': 0.1433, 'chop_score': 0.1888, 'volatility_percentile': 0.2165, 'consolidation_score': 0.2179}
2026-05-11 15:03:22,424 INFO RegimeClassifier[mode=ltf_behaviour] train group diagnostics={'dollar': {'n': 16632, 'score_mae': {'trend_score': 0.017, 'range_score': 0.032, 'chop_score': 0.0189, 'volatility_percentile': 0.0141, 'consolidation_score': 0.0197}, 'target_score_mean': {'trend_score': 0.491, 'range_score': 0.2347, 'chop_score': 0.4613, 'volatility_percentile': 0.3824, 'consolidation_score': 0.186}, 'pred_score_mean': {'trend_score': 0.4918, 'range_score': 0.2354, 'chop_score': 0.4616, 'volatility_percentile': 0.3804, 'consolidation_score': 0.1829}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3550, 63, 0, 2, 0, 0, 164], [4, 100, 0, 0, 0, 1, 5], [0, 0, 194, 10, 54, 0, 202], [2, 0, 7, 535, 33, 0, 112], [0, 0, 31, 17, 3102, 0, 166], [0, 21, 0, 0, 7, 50, 50], [135, 12, 75, 41, 69, 3, 7815]]}, 'gold': {'n': 8420, 'score_mae': {'trend_score': 0.0168, 'range_score': 0.0327, 'chop_score': 0.019, 'volatility_percentile': 0.0147, 'consolidation_score': 0.02}, 'target_score_mean': {'trend_score': 0.4883, 'range_score': 0.2355, 'chop_score': 0.4652, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}, 'pred_score_mean': {'trend_score': 0.4898, 'range_score': 0.2362, 'chop_score': 0.4649, 'volatility_percentile': 0.3747, 'consolidation_score': 0.1885}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1780, 35, 0, 0, 0, 0, 70], [3, 52, 0, 0, 0, 0, 1], [0, 0, 94, 8, 23, 0, 119], [1, 0, 2, 330, 19, 0, 64], [0, 0, 17, 20, 1583, 0, 84], [0, 15, 0, 0, 6, 37, 23], [61, 4, 50, 13, 49, 0, 3857]]}, 'jpy': {'n': 24948, 'score_mae': {'trend_score': 0.0169, 'range_score': 0.032, 'chop_score': 0.0187, 'volatility_percentile': 0.015, 'consolidation_score': 0.0197}, 'target_score_mean': {'trend_score': 0.4904, 'range_score': 0.2338, 'chop_score': 0.4653, 'volatility_percentile': 0.3826, 'consolidation_score': 0.1905}, 'pred_score_mean': {'trend_score': 0.491, 'range_score': 0.2346, 'chop_score': 0.4648, 'volatility_percentile': 0.3802, 'consolidation_score': 0.1866}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[5299, 127, 0, 2, 0, 0, 218], [8, 170, 0, 0, 0, 1, 8], [0, 0, 255, 19, 77, 0, 296], [2, 0, 3, 1049, 67, 0, 193], [0, 0, 43, 52, 4731, 0, 289], [0, 38, 0, 0, 17, 78, 90], [187, 13, 113, 72, 132, 0, 11299]]}}
2026-05-11 15:03:22,608 INFO RegimeClassifier[mode=ltf_behaviour] validation group diagnostics={'dollar': {'n': 10093, 'score_mae': {'trend_score': 0.0178, 'range_score': 0.0327, 'chop_score': 0.019, 'volatility_percentile': 0.014, 'consolidation_score': 0.0189}, 'target_score_mean': {'trend_score': 0.4863, 'range_score': 0.2392, 'chop_score': 0.4622, 'volatility_percentile': 0.3804, 'consolidation_score': 0.1814}, 'pred_score_mean': {'trend_score': 0.4876, 'range_score': 0.2377, 'chop_score': 0.4631, 'volatility_percentile': 0.3782, 'consolidation_score': 0.179}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[2281, 24, 0, 2, 0, 0, 105], [3, 46, 0, 0, 0, 3, 1], [0, 0, 115, 7, 43, 0, 151], [0, 0, 4, 331, 25, 0, 63], [0, 0, 24, 19, 1923, 0, 84], [0, 13, 0, 0, 3, 32, 29], [52, 6, 45, 34, 52, 1, 4572]]}, 'gold': {'n': 5117, 'score_mae': {'trend_score': 0.0166, 'range_score': 0.0312, 'chop_score': 0.0189, 'volatility_percentile': 0.0144, 'consolidation_score': 0.02}, 'target_score_mean': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}, 'pred_score_mean': {'trend_score': 0.498, 'range_score': 0.233, 'chop_score': 0.457, 'volatility_percentile': 0.3793, 'consolidation_score': 0.18}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1103, 17, 0, 0, 0, 0, 47], [3, 30, 0, 0, 0, 1, 1], [0, 0, 65, 3, 15, 0, 88], [0, 0, 3, 215, 9, 0, 28], [0, 0, 8, 9, 819, 0, 51], [0, 6, 0, 0, 4, 19, 21], [44, 3, 34, 24, 28, 0, 2419]]}, 'jpy': {'n': 15142, 'score_mae': {'trend_score': 0.0176, 'range_score': 0.0319, 'chop_score': 0.0187, 'volatility_percentile': 0.0149, 'consolidation_score': 0.0195}, 'target_score_mean': {'trend_score': 0.495, 'range_score': 0.2297, 'chop_score': 0.4582, 'volatility_percentile': 0.3808, 'consolidation_score': 0.1868}, 'pred_score_mean': {'trend_score': 0.4951, 'range_score': 0.2297, 'chop_score': 0.4587, 'volatility_percentile': 0.3786, 'consolidation_score': 0.1833}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3313, 60, 0, 1, 0, 0, 144], [4, 102, 0, 0, 0, 2, 7], [0, 0, 147, 14, 48, 0, 175], [2, 0, 4, 667, 35, 0, 119], [0, 0, 26, 29, 2601, 0, 161], [0, 20, 0, 0, 10, 44, 48], [99, 12, 72, 35, 86, 0, 7055]]}}
2026-05-11 15:03:22,614 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-11 15:03:22,615 INFO Regime phase LTF train fold=train_all: 28.2s
2026-05-11 15:03:22,740 INFO Regime LTF complete fold=train_all: score_accuracy=0.980, train=262644 val=30352 mae={'trend_score': 0.0175, 'range_score': 0.032, 'chop_score': 0.0188, 'volatility_percentile': 0.0145, 'consolidation_score': 0.0194}
2026-05-11 15:03:22,743 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 15:03:23,161 INFO Regime[1H mode=ltf_behaviour fold=latest] LTF score diagnostics on GBPUSD:
{'trend_score': {'mean': 0.4921, 'q10': 0.1918, 'q50': 0.4877, 'q90': 0.8002}, 'range_score': {'mean': 0.2337, 'q10': 0.0513, 'q50': 0.2125, 'q90': 0.4349}, 'chop_score': {'mean': 0.4612, 'q10': 0.2148, 'q50': 0.4503, 'q90': 0.7255}, 'volatility_percentile': {'mean': 0.3832, 'q10': 0.1018, 'q50': 0.3704, 'q90': 0.6823}, 'consolidation_score': {'mean': 0.1854, 'q10': 0.0, 'q50': 0.1185, 'q90': 0.506}}
2026-05-11 15:03:23,167 INFO Regime retrain total: 71.3s (370559 train+val samples)
2026-05-11 15:03:23,173 INFO Retrain complete. Total wall-clock: 71.3s
  DONE  Retrain regime [train-split retrain]
  SKIP  Quality/RL incremental retrain — clean train-only weights retained

=== Round 3: Backtest on last 3yr (post-retrain evaluation) ===
  START Round 3 - Post-retrain backtest (last 3yr)
2026-05-11 15:03:24,784 INFO === STEP 6: BACKTEST (round3) ===
2026-05-11 15:03:24,786 INFO BT_WINDOW=round3 — post-retrain eval: 2022-08-05 → 2025-08-05 (last 3yr)
2026-05-11 15:03:24,786 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-05-11 15:03:24,786 INFO Round 3 — running backtest: 2022-08-05 → 2025-08-05 (ml_trader, shared ML cache)
2026-05-11 15:03:24,786 INFO Round 3 — QualityScorer hard gate disabled; set BACKTEST_ENABLE_QUALITY_GATE=1 to enable it
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
2026-05-11 15:05:22,840 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 15:05:23,654 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 15:05:23,741 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 15:05:24,207 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1003: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["bos_bear_flag"] = out["bos_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1013: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bull_open"] = out["fvg_bull"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1014: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["fvg_bear_open"] = out["fvg_bear"].astype(float)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1117: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1119: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1121: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1125: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 15:05:25,247 WARNING _build_sequence_df: HTF frame 5M filled 461 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1129: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 15:05:25,339 WARNING _build_sequence_df: HTF frame 5M filled 461 warmup/alignment gaps with 0.000
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
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1133: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 15:05:25,520 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 15:05:25,583 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
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
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
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
2026-05-11 15:05:39,954 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 15:05:40,434 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
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
2026-05-11 15:05:40,707 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:1111: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 15:05:40,744 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
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
2026-05-11 15:07:04,497 INFO Round 3 backtest — 266 trades | avg WR=22.2% | avg PF=0.84 | avg Sharpe=-1.18
2026-05-11 15:07:04,497 INFO   ml_trader: 266 trades | WR=22.2% | fixed PF=0.84 | Return=-33.3% | ExpR=-0.125 | DD=43.9% | Sharpe=-1.18
2026-05-11 15:07:04,497 INFO   ml_trader gate_diagnostics: bars=403523 no_signal=291424 quality_block=0 session_skip=111829 density=4 pm_reject=0
2026-05-11 15:07:04,497 INFO   ml_trader no_signal_reasons: {'no_trade_chop': 34317, 'no_trade_uncertain': 109309, 'weak_gru_direction': 72421, 'no_trade_extreme_vol': 27015, 'gru_expected_r_below_threshold': 15123, 'trend_structure_missing': 9412, 'wait_pullback': 23710, 'expected_r_below_threshold': 21, 'tradeability_direction_conflict': 96}
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 266
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (266 rows)
  DONE  Round 3 - Post-retrain backtest (last 3yr)
  Saved Round 3 result → round3_summary.json
  Journal after Round 3: 590 entries

  SKIP  Round 3 Quality+RL retrain — evaluation journals not used for fitting

======================================================================
  BLIND BACKTEST PIPELINE COMPLETE
======================================================================
  Round 1 (train-tail window)   trades=155  WR=25.8%  PF=0.942  Sharpe=-0.413
  Round 2 (blind test)          trades=169  WR=24.9%  PF=0.978  Sharpe=-0.152
  Round 3 (last 3yr)            trades=266  WR=22.2%  PF=0.839  Sharpe=-1.179


WARNING: GITHUB_TOKEN not set — skipping GitHub push
2026-05-11 15:07:04,959 INFO Round 3: wrote 266 journal entries (total in file: 590)