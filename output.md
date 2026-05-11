 Cleared done-check: training_summary.json
  Cleared done-check: training_7b_train_summary.json
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
2026-05-10 22:57:07,584 INFO Loading feature-engineered data...
2026-05-10 22:57:08,174 INFO Loaded 221743 rows, 202 features
2026-05-10 22:57:08,175 INFO Data span: 2016-01-04 -> 2025-08-05  (9.6 years)
2026-05-10 22:57:08,177 INFO Fold 000 train 2016-01-04 -> 2018-01-03 (47088 bars), val 2018-01-04 -> 2019-01-03 (23448 bars)
2026-05-10 22:57:08,178 INFO Fold 001 train 2016-01-04 -> 2019-01-03 (70536 bars), val 2019-01-04 -> 2020-01-03 (23377 bars)
2026-05-10 22:57:08,178 INFO Fold 002 train 2016-01-04 -> 2020-01-03 (93913 bars), val 2020-01-06 -> 2020-12-31 (23259 bars)
2026-05-10 22:57:08,178 INFO Fold 003 train 2016-01-04 -> 2020-12-31 (117172 bars), val 2021-01-04 -> 2022-01-03 (23507 bars)
2026-05-10 22:57:08,179 INFO Fold 004 train 2016-01-04 -> 2022-01-03 (140679 bars), val 2022-01-04 -> 2023-01-03 (23588 bars)
2026-05-10 22:57:08,179 INFO Fold 005 train 2016-01-04 -> 2022-08-05 (154539 bars), val 2022-08-05 -> 2023-08-04 (20412 bars)
2026-05-10 22:57:08,179 INFO No leakage confirmed: train/train_tail/internal folds end before final 2-year blind test

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
2026-05-10 22:57:17,719 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-05-10 22:57:17,719 INFO --- Training gru ---
2026-05-10 22:57:17,719 INFO Running retrain --model gru
2026-05-10 22:57:17,931 INFO retrain environment: KAGGLE
2026-05-10 22:57:19,514 INFO Device: CUDA (2 GPU(s))
2026-05-10 22:57:19,524 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-10 22:57:19,524 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-10 22:57:19,525 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-10 22:57:19,526 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-10 22:57:19,526 INFO Retrain data split: train
2026-05-10 22:57:19,526 INFO Retrain rolling fold selector: latest
2026-05-10 22:57:19,527 INFO === GRU-LSTM retrain (timeframes: ['15M']) ===
2026-05-10 22:57:19,681 INFO NumExpr defaulting to 4 threads.
2026-05-10 22:57:19,892 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-10 22:57:19,892 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-10 22:57:19,893 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-10 22:57:19,893 INFO GRU multi-symbol training (Kaggle mode): 6 symbols × ['15M']
2026-05-10 22:57:19,893 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260510_225719
2026-05-10 22:57:19,896 WARNING WeightsManifest: no manifest at /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json — assuming compatible (run a full retrain to generate one)
2026-05-10 22:57:19,896 INFO GRU cold start: no compatible existing weights found
2026-05-10 22:57:20,152 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-10 22:57:20,179 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-10 22:57:20,195 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-10 22:57:20,205 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-10 22:57:20,279 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-10 22:57:20,285 INFO Loaded XAUUSD/15M split=train fold=latest: 176438 bars (2016-01-04 → 2023-08-04)
2026-05-10 22:57:20,571 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-10 22:57:20,589 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-10 22:57:20,604 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-10 22:57:20,611 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 22:57:20,646 INFO Loaded EURUSD/15M split=train fold=latest: 174951 bars (2016-01-04 → 2023-08-04)
2026-05-10 22:57:20,928 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-10 22:57:20,947 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-10 22:57:20,961 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-10 22:57:20,979 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 22:57:21,022 INFO Loaded USDJPY/15M split=train fold=latest: 174944 bars (2016-01-04 → 2023-08-04)
2026-05-10 22:57:21,334 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-10 22:57:21,356 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-10 22:57:21,371 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-10 22:57:21,378 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 22:57:21,419 INFO Loaded EURJPY/15M split=train fold=latest: 174948 bars (2016-01-04 → 2023-08-04)
2026-05-10 22:57:21,717 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-10 22:57:21,737 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-10 22:57:21,751 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-10 22:57:21,758 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 22:57:21,796 INFO Loaded GBPJPY/15M split=train fold=latest: 174902 bars (2016-01-04 → 2023-08-04)
2026-05-10 22:57:22,080 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-10 22:57:22,101 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-10 22:57:22,119 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-10 22:57:22,129 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 22:57:22,171 INFO Loaded GBPUSD/15M split=train fold=latest: 174899 bars (2016-01-04 → 2023-08-04)
2026-05-10 22:57:22,353 INFO train_multi: 6 segments, ~936212 total bars
2026-05-10 22:57:22,676 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-05-10 22:57:22,676 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-05-10 22:57:22,676 INFO train_multi: training ALL 6 segments across TFs ['15M'] in one combined pass
2026-05-10 22:57:22,676 INFO train_multi: building combined dataset for TF=ALL (6 segments)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 22:57:31,517 INFO train_multi TF=ALL: 936032 sequences across 6 segments
2026-05-10 22:57:31,517 INFO train_multi TF=ALL: estimated peak RAM = 10224 MB (train=479995 val=120002 n_feat=71 seq_len=30)
2026-05-10 22:57:32,784 INFO train_multi TF=ALL: train=479995 val=120002 (5122 MB tensors)
2026-05-10 22:57:39,308 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=25, min_epochs=45)
2026-05-10 22:57:56,733 INFO train_multi TF=ALL epoch 1/100 train=2.3455 val=2.3463 r_mae=0.986 pos_r_acc=0.493 side_acc=0.495 r_n=240004
2026-05-10 22:57:56,742 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 22:57:56,742 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 22:57:56,743 INFO train_multi TF=ALL: new best val=2.3463 — saved
2026-05-10 22:58:11,971 INFO train_multi TF=ALL epoch 2/100 train=2.3432 val=2.3432 r_mae=0.984 pos_r_acc=0.520 side_acc=0.495 r_n=240004
2026-05-10 22:58:11,976 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 22:58:11,976 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 22:58:11,977 INFO train_multi TF=ALL: new best val=2.3432 — saved
2026-05-10 22:58:27,254 INFO train_multi TF=ALL epoch 3/100 train=2.3408 val=2.3397 r_mae=0.982 pos_r_acc=0.525 side_acc=0.518 r_n=240004
2026-05-10 22:58:27,259 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 22:58:27,260 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 22:58:27,260 INFO train_multi TF=ALL: new best val=2.3397 — saved
2026-05-10 22:58:42,614 INFO train_multi TF=ALL epoch 4/100 train=2.3383 val=2.3356 r_mae=0.980 pos_r_acc=0.525 side_acc=0.526 r_n=240004
2026-05-10 22:58:42,619 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 22:58:42,619 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 22:58:42,620 INFO train_multi TF=ALL: new best val=2.3356 — saved
2026-05-10 22:58:57,744 INFO train_multi TF=ALL epoch 5/100 train=2.3332 val=2.3291 r_mae=0.979 pos_r_acc=0.530 side_acc=0.529 r_n=240004
2026-05-10 22:58:57,748 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 22:58:57,749 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 22:58:57,749 INFO train_multi TF=ALL: new best val=2.3291 — saved
2026-05-10 22:59:13,149 INFO train_multi TF=ALL epoch 6/100 train=2.3276 val=2.3246 r_mae=0.978 pos_r_acc=0.534 side_acc=0.535 r_n=240004
2026-05-10 22:59:13,154 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 22:59:13,154 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 22:59:13,154 INFO train_multi TF=ALL: new best val=2.3246 — saved
2026-05-10 22:59:28,479 INFO train_multi TF=ALL epoch 7/100 train=2.3236 val=2.3221 r_mae=0.978 pos_r_acc=0.536 side_acc=0.537 r_n=240004
2026-05-10 22:59:28,484 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 22:59:28,485 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 22:59:28,485 INFO train_multi TF=ALL: new best val=2.3221 — saved
2026-05-10 22:59:43,834 INFO train_multi TF=ALL epoch 8/100 train=2.3213 val=2.3206 r_mae=0.977 pos_r_acc=0.536 side_acc=0.536 r_n=240004
2026-05-10 22:59:43,839 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 22:59:43,839 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 22:59:43,839 INFO train_multi TF=ALL: new best val=2.3206 — saved
2026-05-10 22:59:59,030 INFO train_multi TF=ALL epoch 9/100 train=2.3196 val=2.3195 r_mae=0.977 pos_r_acc=0.536 side_acc=0.536 r_n=240004
2026-05-10 22:59:59,036 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 22:59:59,036 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 22:59:59,036 INFO train_multi TF=ALL: new best val=2.3195 — saved
2026-05-10 23:00:14,396 INFO train_multi TF=ALL epoch 10/100 train=2.3178 val=2.3190 r_mae=0.977 pos_r_acc=0.535 side_acc=0.536 r_n=240004
2026-05-10 23:00:14,401 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:00:14,401 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:00:14,401 INFO train_multi TF=ALL: new best val=2.3190 — saved
2026-05-10 23:00:29,522 INFO train_multi TF=ALL epoch 11/100 train=2.3163 val=2.3180 r_mae=0.976 pos_r_acc=0.536 side_acc=0.536 r_n=240004
2026-05-10 23:00:29,527 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:00:29,527 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:00:29,527 INFO train_multi TF=ALL: new best val=2.3180 — saved
2026-05-10 23:00:44,888 INFO train_multi TF=ALL epoch 12/100 train=2.3151 val=2.3168 r_mae=0.977 pos_r_acc=0.536 side_acc=0.536 r_n=240004
2026-05-10 23:00:44,893 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:00:44,893 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:00:44,893 INFO train_multi TF=ALL: new best val=2.3168 — saved
2026-05-10 23:00:59,909 INFO train_multi TF=ALL epoch 13/100 train=2.3140 val=2.3165 r_mae=0.976 pos_r_acc=0.537 side_acc=0.540 r_n=240004
2026-05-10 23:00:59,914 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:00:59,914 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:00:59,914 INFO train_multi TF=ALL: new best val=2.3165 — saved
2026-05-10 23:01:15,109 INFO train_multi TF=ALL epoch 14/100 train=2.3120 val=2.3157 r_mae=0.975 pos_r_acc=0.537 side_acc=0.539 r_n=240004
2026-05-10 23:01:15,114 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:01:15,114 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:01:15,114 INFO train_multi TF=ALL: new best val=2.3157 — saved
2026-05-10 23:01:30,193 INFO train_multi TF=ALL epoch 15/100 train=2.3099 val=2.3155 r_mae=0.974 pos_r_acc=0.539 side_acc=0.542 r_n=240004
2026-05-10 23:01:30,199 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:01:30,199 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:01:30,199 INFO train_multi TF=ALL: new best val=2.3155 — saved
2026-05-10 23:01:45,538 INFO train_multi TF=ALL epoch 16/100 train=2.3079 val=2.3132 r_mae=0.975 pos_r_acc=0.539 side_acc=0.544 r_n=240004
2026-05-10 23:01:45,543 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:01:45,543 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:01:45,543 INFO train_multi TF=ALL: new best val=2.3132 — saved
2026-05-10 23:02:00,856 INFO train_multi TF=ALL epoch 17/100 train=2.3057 val=2.3112 r_mae=0.974 pos_r_acc=0.541 side_acc=0.542 r_n=240004
2026-05-10 23:02:00,861 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:02:00,861 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:02:00,861 INFO train_multi TF=ALL: new best val=2.3112 — saved
2026-05-10 23:02:16,215 INFO train_multi TF=ALL epoch 18/100 train=2.3032 val=2.3093 r_mae=0.973 pos_r_acc=0.541 side_acc=0.544 r_n=240004
2026-05-10 23:02:16,219 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:02:16,220 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:02:16,220 INFO train_multi TF=ALL: new best val=2.3093 — saved
2026-05-10 23:02:31,372 INFO train_multi TF=ALL epoch 19/100 train=2.2986 val=2.3035 r_mae=0.970 pos_r_acc=0.545 side_acc=0.549 r_n=240004
2026-05-10 23:02:31,377 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:02:31,377 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:02:31,377 INFO train_multi TF=ALL: new best val=2.3035 — saved
2026-05-10 23:02:46,505 INFO train_multi TF=ALL epoch 20/100 train=2.2923 val=2.2944 r_mae=0.967 pos_r_acc=0.554 side_acc=0.556 r_n=240004
2026-05-10 23:02:46,510 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:02:46,511 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:02:46,511 INFO train_multi TF=ALL: new best val=2.2944 — saved
2026-05-10 23:03:01,653 INFO train_multi TF=ALL epoch 21/100 train=2.2840 val=2.2808 r_mae=0.962 pos_r_acc=0.560 side_acc=0.564 r_n=240004
2026-05-10 23:03:01,658 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:03:01,658 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:03:01,658 INFO train_multi TF=ALL: new best val=2.2808 — saved
2026-05-10 23:03:16,916 INFO train_multi TF=ALL epoch 22/100 train=2.2707 val=2.2576 r_mae=0.952 pos_r_acc=0.570 side_acc=0.575 r_n=240004
2026-05-10 23:03:16,921 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:03:16,922 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:03:16,922 INFO train_multi TF=ALL: new best val=2.2576 — saved
2026-05-10 23:03:32,331 INFO train_multi TF=ALL epoch 23/100 train=2.2474 val=2.2162 r_mae=0.940 pos_r_acc=0.590 side_acc=0.596 r_n=240004
2026-05-10 23:03:32,336 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:03:32,336 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:03:32,336 INFO train_multi TF=ALL: new best val=2.2162 — saved
2026-05-10 23:03:47,626 INFO train_multi TF=ALL epoch 24/100 train=2.2152 val=2.1588 r_mae=0.918 pos_r_acc=0.616 side_acc=0.626 r_n=240004
2026-05-10 23:03:47,631 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:03:47,631 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:03:47,631 INFO train_multi TF=ALL: new best val=2.1588 — saved
2026-05-10 23:04:02,803 INFO train_multi TF=ALL epoch 25/100 train=2.1802 val=2.1118 r_mae=0.901 pos_r_acc=0.634 side_acc=0.645 r_n=240004
2026-05-10 23:04:02,809 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:04:02,809 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:04:02,809 INFO train_multi TF=ALL: new best val=2.1118 — saved
2026-05-10 23:04:17,943 INFO train_multi TF=ALL epoch 26/100 train=2.1434 val=2.0685 r_mae=0.883 pos_r_acc=0.649 side_acc=0.660 r_n=240004
2026-05-10 23:04:17,948 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:04:17,948 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:04:17,948 INFO train_multi TF=ALL: new best val=2.0685 — saved
2026-05-10 23:04:33,308 INFO train_multi TF=ALL epoch 27/100 train=2.1111 val=2.0335 r_mae=0.868 pos_r_acc=0.659 side_acc=0.670 r_n=240004
2026-05-10 23:04:33,313 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:04:33,313 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:04:33,313 INFO train_multi TF=ALL: new best val=2.0335 — saved
2026-05-10 23:04:48,576 INFO train_multi TF=ALL epoch 28/100 train=2.0836 val=2.0073 r_mae=0.858 pos_r_acc=0.668 side_acc=0.678 r_n=240004
2026-05-10 23:04:48,581 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:04:48,581 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:04:48,581 INFO train_multi TF=ALL: new best val=2.0073 — saved
2026-05-10 23:05:03,891 INFO train_multi TF=ALL epoch 29/100 train=2.0582 val=1.9861 r_mae=0.847 pos_r_acc=0.671 side_acc=0.682 r_n=240004
2026-05-10 23:05:03,897 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:05:03,897 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:05:03,897 INFO train_multi TF=ALL: new best val=1.9861 — saved
2026-05-10 23:05:19,157 INFO train_multi TF=ALL epoch 30/100 train=2.0356 val=1.9703 r_mae=0.840 pos_r_acc=0.675 side_acc=0.685 r_n=240004
2026-05-10 23:05:19,162 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:05:19,162 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:05:19,162 INFO train_multi TF=ALL: new best val=1.9703 — saved
2026-05-10 23:05:34,344 INFO train_multi TF=ALL epoch 31/100 train=2.0189 val=1.9559 r_mae=0.835 pos_r_acc=0.679 side_acc=0.690 r_n=240004
2026-05-10 23:05:34,349 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:05:34,349 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:05:34,349 INFO train_multi TF=ALL: new best val=1.9559 — saved
2026-05-10 23:05:49,583 INFO train_multi TF=ALL epoch 32/100 train=2.0034 val=1.9440 r_mae=0.830 pos_r_acc=0.682 side_acc=0.692 r_n=240004
2026-05-10 23:05:49,588 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:05:49,588 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:05:49,588 INFO train_multi TF=ALL: new best val=1.9440 — saved
2026-05-10 23:06:04,927 INFO train_multi TF=ALL epoch 33/100 train=1.9876 val=1.9342 r_mae=0.821 pos_r_acc=0.683 side_acc=0.693 r_n=240004
2026-05-10 23:06:04,932 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:06:04,932 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:06:04,932 INFO train_multi TF=ALL: new best val=1.9342 — saved
2026-05-10 23:06:20,368 INFO train_multi TF=ALL epoch 34/100 train=1.9777 val=1.9322 r_mae=0.821 pos_r_acc=0.681 side_acc=0.691 r_n=240004
2026-05-10 23:06:20,373 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:06:20,373 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:06:20,373 INFO train_multi TF=ALL: new best val=1.9322 — saved
2026-05-10 23:06:35,671 INFO train_multi TF=ALL epoch 35/100 train=1.9681 val=1.9308 r_mae=0.813 pos_r_acc=0.681 side_acc=0.692 r_n=240004
2026-05-10 23:06:35,676 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:06:35,676 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:06:35,676 INFO train_multi TF=ALL: new best val=1.9308 — saved
2026-05-10 23:06:50,981 INFO train_multi TF=ALL epoch 36/100 train=1.9554 val=1.9115 r_mae=0.807 pos_r_acc=0.687 side_acc=0.696 r_n=240004
2026-05-10 23:06:50,987 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:06:50,987 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:06:50,987 INFO train_multi TF=ALL: new best val=1.9115 — saved
2026-05-10 23:07:06,140 INFO train_multi TF=ALL epoch 37/100 train=1.9458 val=1.9058 r_mae=0.807 pos_r_acc=0.690 side_acc=0.699 r_n=240004
2026-05-10 23:07:06,145 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:07:06,145 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:07:06,146 INFO train_multi TF=ALL: new best val=1.9058 — saved
2026-05-10 23:07:21,355 INFO train_multi TF=ALL epoch 38/100 train=1.9394 val=1.9046 r_mae=0.800 pos_r_acc=0.688 side_acc=0.698 r_n=240004
2026-05-10 23:07:21,361 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:07:21,361 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:07:21,362 INFO train_multi TF=ALL: new best val=1.9046 — saved
2026-05-10 23:07:36,602 INFO train_multi TF=ALL epoch 39/100 train=1.9303 val=1.8993 r_mae=0.795 pos_r_acc=0.690 side_acc=0.700 r_n=240004
2026-05-10 23:07:36,607 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:07:36,608 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:07:36,608 INFO train_multi TF=ALL: new best val=1.8993 — saved
2026-05-10 23:07:52,089 INFO train_multi TF=ALL epoch 40/100 train=1.9234 val=1.8885 r_mae=0.794 pos_r_acc=0.692 side_acc=0.702 r_n=240004
2026-05-10 23:07:52,094 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:07:52,095 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:07:52,095 INFO train_multi TF=ALL: new best val=1.8885 — saved
2026-05-10 23:08:07,391 INFO train_multi TF=ALL epoch 41/100 train=1.9181 val=1.8876 r_mae=0.794 pos_r_acc=0.692 side_acc=0.703 r_n=240004
2026-05-10 23:08:07,396 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:08:07,396 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:08:07,396 INFO train_multi TF=ALL: new best val=1.8876 — saved
2026-05-10 23:08:22,800 INFO train_multi TF=ALL epoch 42/100 train=1.9117 val=1.8841 r_mae=0.792 pos_r_acc=0.693 side_acc=0.704 r_n=240004
2026-05-10 23:08:22,805 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:08:22,805 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:08:22,805 INFO train_multi TF=ALL: new best val=1.8841 — saved
2026-05-10 23:08:38,073 INFO train_multi TF=ALL epoch 43/100 train=1.9044 val=1.8780 r_mae=0.792 pos_r_acc=0.692 side_acc=0.705 r_n=240004
2026-05-10 23:08:38,077 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:08:38,078 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:08:38,078 INFO train_multi TF=ALL: new best val=1.8780 — saved
2026-05-10 23:08:53,335 INFO train_multi TF=ALL epoch 44/100 train=1.8991 val=1.8744 r_mae=0.791 pos_r_acc=0.695 side_acc=0.707 r_n=240004
2026-05-10 23:08:53,344 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:08:53,344 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:08:53,344 INFO train_multi TF=ALL: new best val=1.8744 — saved
2026-05-10 23:09:08,754 INFO train_multi TF=ALL epoch 45/100 train=1.8921 val=1.8755 r_mae=0.788 pos_r_acc=0.691 side_acc=0.706 r_n=240004
2026-05-10 23:09:24,269 INFO train_multi TF=ALL epoch 46/100 train=1.8872 val=1.8707 r_mae=0.783 pos_r_acc=0.694 side_acc=0.708 r_n=240004
2026-05-10 23:09:24,274 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:09:24,274 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:09:24,274 INFO train_multi TF=ALL: new best val=1.8707 — saved
2026-05-10 23:09:39,416 INFO train_multi TF=ALL epoch 47/100 train=1.8816 val=1.8651 r_mae=0.780 pos_r_acc=0.696 side_acc=0.707 r_n=240004
2026-05-10 23:09:39,421 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:09:39,421 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:09:39,421 INFO train_multi TF=ALL: new best val=1.8651 — saved
2026-05-10 23:09:54,854 INFO train_multi TF=ALL epoch 48/100 train=1.8762 val=1.8629 r_mae=0.776 pos_r_acc=0.698 side_acc=0.709 r_n=240004
2026-05-10 23:09:54,859 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:09:54,859 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:09:54,859 INFO train_multi TF=ALL: new best val=1.8629 — saved
2026-05-10 23:10:10,092 INFO train_multi TF=ALL epoch 49/100 train=1.8719 val=1.8651 r_mae=0.773 pos_r_acc=0.697 side_acc=0.708 r_n=240004
2026-05-10 23:10:25,453 INFO train_multi TF=ALL epoch 50/100 train=1.8651 val=1.8606 r_mae=0.782 pos_r_acc=0.695 side_acc=0.709 r_n=240004
2026-05-10 23:10:25,458 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:10:25,458 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:10:25,458 INFO train_multi TF=ALL: new best val=1.8606 — saved
2026-05-10 23:10:40,688 INFO train_multi TF=ALL epoch 51/100 train=1.8613 val=1.8524 r_mae=0.769 pos_r_acc=0.699 side_acc=0.711 r_n=240004
2026-05-10 23:10:40,693 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:10:40,693 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:10:40,693 INFO train_multi TF=ALL: new best val=1.8524 — saved
2026-05-10 23:10:56,006 INFO train_multi TF=ALL epoch 52/100 train=1.8566 val=1.8559 r_mae=0.761 pos_r_acc=0.701 side_acc=0.711 r_n=240004
2026-05-10 23:11:11,216 INFO train_multi TF=ALL epoch 53/100 train=1.8518 val=1.8546 r_mae=0.765 pos_r_acc=0.701 side_acc=0.711 r_n=240004
2026-05-10 23:11:26,491 INFO train_multi TF=ALL epoch 54/100 train=1.8482 val=1.8465 r_mae=0.776 pos_r_acc=0.697 side_acc=0.712 r_n=240004
2026-05-10 23:11:26,497 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:11:26,497 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:11:26,497 INFO train_multi TF=ALL: new best val=1.8465 — saved
2026-05-10 23:11:41,655 INFO train_multi TF=ALL epoch 55/100 train=1.8432 val=1.8557 r_mae=0.764 pos_r_acc=0.700 side_acc=0.710 r_n=240004
2026-05-10 23:11:56,914 INFO train_multi TF=ALL epoch 56/100 train=1.8399 val=1.8433 r_mae=0.767 pos_r_acc=0.701 side_acc=0.714 r_n=240004
2026-05-10 23:11:56,918 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:11:56,919 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:11:56,919 INFO train_multi TF=ALL: new best val=1.8433 — saved
2026-05-10 23:12:12,180 INFO train_multi TF=ALL epoch 57/100 train=1.8337 val=1.8523 r_mae=0.773 pos_r_acc=0.697 side_acc=0.712 r_n=240004
2026-05-10 23:12:27,446 INFO train_multi TF=ALL epoch 58/100 train=1.8305 val=1.8462 r_mae=0.766 pos_r_acc=0.698 side_acc=0.713 r_n=240004
2026-05-10 23:12:42,704 INFO train_multi TF=ALL epoch 59/100 train=1.8254 val=1.8341 r_mae=0.767 pos_r_acc=0.703 side_acc=0.716 r_n=240004
2026-05-10 23:12:42,709 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:12:42,709 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:12:42,709 INFO train_multi TF=ALL: new best val=1.8341 — saved
2026-05-10 23:12:57,955 INFO train_multi TF=ALL epoch 60/100 train=1.8226 val=1.8418 r_mae=0.768 pos_r_acc=0.700 side_acc=0.711 r_n=240004
2026-05-10 23:13:13,188 INFO train_multi TF=ALL epoch 61/100 train=1.8177 val=1.8437 r_mae=0.763 pos_r_acc=0.700 side_acc=0.712 r_n=240004
2026-05-10 23:13:28,261 INFO train_multi TF=ALL epoch 62/100 train=1.8148 val=1.8359 r_mae=0.760 pos_r_acc=0.702 side_acc=0.716 r_n=240004
2026-05-10 23:13:43,463 INFO train_multi TF=ALL epoch 63/100 train=1.8105 val=1.8288 r_mae=0.759 pos_r_acc=0.705 side_acc=0.717 r_n=240004
2026-05-10 23:13:43,468 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:13:43,468 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:13:43,468 INFO train_multi TF=ALL: new best val=1.8288 — saved
2026-05-10 23:13:58,670 INFO train_multi TF=ALL epoch 64/100 train=1.8045 val=1.8346 r_mae=0.753 pos_r_acc=0.705 side_acc=0.717 r_n=240004
2026-05-10 23:14:13,818 INFO train_multi TF=ALL epoch 65/100 train=1.8012 val=1.8296 r_mae=0.765 pos_r_acc=0.702 side_acc=0.716 r_n=240004
2026-05-10 23:14:28,949 INFO train_multi TF=ALL epoch 66/100 train=1.7965 val=1.8449 r_mae=0.761 pos_r_acc=0.701 side_acc=0.714 r_n=240004
2026-05-10 23:14:44,133 INFO train_multi TF=ALL epoch 67/100 train=1.7933 val=1.8372 r_mae=0.761 pos_r_acc=0.702 side_acc=0.714 r_n=240004
2026-05-10 23:14:59,337 INFO train_multi TF=ALL epoch 68/100 train=1.7892 val=1.8273 r_mae=0.760 pos_r_acc=0.705 side_acc=0.719 r_n=240004
2026-05-10 23:14:59,343 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:14:59,343 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:14:59,343 INFO train_multi TF=ALL: new best val=1.8273 — saved
2026-05-10 23:15:14,679 INFO train_multi TF=ALL epoch 69/100 train=1.7851 val=1.8377 r_mae=0.763 pos_r_acc=0.698 side_acc=0.715 r_n=240004
2026-05-10 23:15:30,050 INFO train_multi TF=ALL epoch 70/100 train=1.7806 val=1.8242 r_mae=0.757 pos_r_acc=0.705 side_acc=0.718 r_n=240004
2026-05-10 23:15:30,055 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:15:30,055 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:15:30,055 INFO train_multi TF=ALL: new best val=1.8242 — saved
2026-05-10 23:15:45,368 INFO train_multi TF=ALL epoch 71/100 train=1.7757 val=1.8481 r_mae=0.758 pos_r_acc=0.700 side_acc=0.712 r_n=240004
2026-05-10 23:16:00,599 INFO train_multi TF=ALL epoch 72/100 train=1.7733 val=1.8231 r_mae=0.750 pos_r_acc=0.708 side_acc=0.719 r_n=240004
2026-05-10 23:16:00,604 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:16:00,604 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:16:00,604 INFO train_multi TF=ALL: new best val=1.8231 — saved
2026-05-10 23:16:15,752 INFO train_multi TF=ALL epoch 73/100 train=1.7683 val=1.8399 r_mae=0.755 pos_r_acc=0.704 side_acc=0.715 r_n=240004
2026-05-10 23:16:30,953 INFO train_multi TF=ALL epoch 74/100 train=1.7631 val=1.8380 r_mae=0.740 pos_r_acc=0.709 side_acc=0.718 r_n=240004
2026-05-10 23:16:46,076 INFO train_multi TF=ALL epoch 75/100 train=1.7603 val=1.8346 r_mae=0.748 pos_r_acc=0.705 side_acc=0.718 r_n=240004
2026-05-10 23:17:01,237 INFO train_multi TF=ALL epoch 76/100 train=1.7549 val=1.8389 r_mae=0.757 pos_r_acc=0.701 side_acc=0.715 r_n=240004
2026-05-10 23:17:16,406 INFO train_multi TF=ALL epoch 77/100 train=1.7510 val=1.8269 r_mae=0.749 pos_r_acc=0.706 side_acc=0.718 r_n=240004
2026-05-10 23:17:31,512 INFO train_multi TF=ALL epoch 78/100 train=1.7456 val=1.8341 r_mae=0.752 pos_r_acc=0.703 side_acc=0.717 r_n=240004
2026-05-10 23:17:46,575 INFO train_multi TF=ALL epoch 79/100 train=1.7421 val=1.8323 r_mae=0.760 pos_r_acc=0.703 side_acc=0.716 r_n=240004
2026-05-10 23:18:01,711 INFO train_multi TF=ALL epoch 80/100 train=1.7394 val=1.8334 r_mae=0.752 pos_r_acc=0.705 side_acc=0.717 r_n=240004
2026-05-10 23:18:16,984 INFO train_multi TF=ALL epoch 81/100 train=1.7333 val=1.8408 r_mae=0.752 pos_r_acc=0.705 side_acc=0.714 r_n=240004
2026-05-10 23:18:32,269 INFO train_multi TF=ALL epoch 82/100 train=1.7288 val=1.8344 r_mae=0.752 pos_r_acc=0.705 side_acc=0.717 r_n=240004
2026-05-10 23:18:47,425 INFO train_multi TF=ALL epoch 83/100 train=1.7224 val=1.8348 r_mae=0.749 pos_r_acc=0.707 side_acc=0.718 r_n=240004
2026-05-10 23:19:02,581 INFO train_multi TF=ALL epoch 84/100 train=1.7185 val=1.8632 r_mae=0.753 pos_r_acc=0.702 side_acc=0.713 r_n=240004
2026-05-10 23:19:17,940 INFO train_multi TF=ALL epoch 85/100 train=1.7149 val=1.8447 r_mae=0.753 pos_r_acc=0.703 side_acc=0.716 r_n=240004
2026-05-10 23:19:33,149 INFO train_multi TF=ALL epoch 86/100 train=1.7111 val=1.8554 r_mae=0.753 pos_r_acc=0.702 side_acc=0.712 r_n=240004
2026-05-10 23:19:48,225 INFO train_multi TF=ALL epoch 87/100 train=1.7031 val=1.8579 r_mae=0.742 pos_r_acc=0.704 side_acc=0.716 r_n=240004
2026-05-10 23:20:03,413 INFO train_multi TF=ALL epoch 88/100 train=1.6978 val=1.8625 r_mae=0.754 pos_r_acc=0.701 side_acc=0.713 r_n=240004
2026-05-10 23:20:18,854 INFO train_multi TF=ALL epoch 89/100 train=1.6923 val=1.8554 r_mae=0.745 pos_r_acc=0.705 side_acc=0.716 r_n=240004
2026-05-10 23:20:33,991 INFO train_multi TF=ALL epoch 90/100 train=1.6879 val=1.8673 r_mae=0.750 pos_r_acc=0.702 side_acc=0.712 r_n=240004
2026-05-10 23:20:49,091 INFO train_multi TF=ALL epoch 91/100 train=1.6850 val=1.8655 r_mae=0.741 pos_r_acc=0.705 side_acc=0.715 r_n=240004
2026-05-10 23:21:04,349 INFO train_multi TF=ALL epoch 92/100 train=1.6777 val=1.8599 r_mae=0.745 pos_r_acc=0.703 side_acc=0.714 r_n=240004
2026-05-10 23:21:19,622 INFO train_multi TF=ALL epoch 93/100 train=1.6749 val=1.8675 r_mae=0.754 pos_r_acc=0.701 side_acc=0.713 r_n=240004
2026-05-10 23:21:34,933 INFO train_multi TF=ALL epoch 94/100 train=1.6706 val=1.8659 r_mae=0.751 pos_r_acc=0.702 side_acc=0.712 r_n=240004
2026-05-10 23:21:50,022 INFO train_multi TF=ALL epoch 95/100 train=1.6622 val=1.8776 r_mae=0.747 pos_r_acc=0.701 side_acc=0.710 r_n=240004
2026-05-10 23:22:05,172 INFO train_multi TF=ALL epoch 96/100 train=1.6555 val=1.8778 r_mae=0.748 pos_r_acc=0.701 side_acc=0.711 r_n=240004
2026-05-10 23:22:20,229 INFO train_multi TF=ALL epoch 97/100 train=1.6528 val=1.8861 r_mae=0.751 pos_r_acc=0.699 side_acc=0.710 r_n=240004
2026-05-10 23:22:20,229 INFO train_multi TF=ALL early stop at epoch 97
2026-05-10 23:22:20,347 INFO GRU R threshold XAUUSD/buy: q25=-1.000 q50=-0.127 (n=156482)
2026-05-10 23:22:20,351 INFO GRU R threshold XAUUSD/sell: q25=-1.000 q50=-0.140 (n=156482)
2026-05-10 23:22:20,355 INFO GRU R threshold EURUSD/buy: q25=-1.000 q50=-0.134 (n=154969)
2026-05-10 23:22:20,360 INFO GRU R threshold EURUSD/sell: q25=-1.000 q50=-0.142 (n=154969)
2026-05-10 23:22:20,364 INFO GRU R threshold USDJPY/buy: q25=-1.000 q50=-0.115 (n=155969)
2026-05-10 23:22:20,368 INFO GRU R threshold USDJPY/sell: q25=-1.000 q50=-0.173 (n=155969)
2026-05-10 23:22:20,372 INFO GRU R threshold EURJPY/buy: q25=-1.000 q50=-0.120 (n=156487)
2026-05-10 23:22:20,376 INFO GRU R threshold EURJPY/sell: q25=-1.000 q50=-0.174 (n=156487)
2026-05-10 23:22:20,380 INFO GRU R threshold GBPJPY/buy: q25=-1.000 q50=-0.137 (n=156862)
2026-05-10 23:22:20,384 INFO GRU R threshold GBPJPY/sell: q25=-1.000 q50=-0.153 (n=156862)
2026-05-10 23:22:20,388 INFO GRU R threshold GBPUSD/buy: q25=-1.000 q50=-0.139 (n=155443)
2026-05-10 23:22:20,392 INFO GRU R threshold GBPUSD/sell: q25=-1.000 q50=-0.133 (n=155443)
2026-05-10 23:22:20,392 INFO GRU per-symbol R thresholds saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/symbol_r_thresholds.json (6 symbols)
2026-05-10 23:22:20,417 INFO Retrain complete. Total wall-clock: 1500.9s
2026-05-10 23:22:23,237 INFO Model gru: SUCCESS
2026-05-10 23:22:23,237 INFO --- Training regime ---
2026-05-10 23:22:23,237 INFO Running retrain --model regime
2026-05-10 23:22:23,592 INFO retrain environment: KAGGLE
2026-05-10 23:22:25,180 INFO Device: CUDA (2 GPU(s))
2026-05-10 23:22:25,188 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-10 23:22:25,189 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-10 23:22:25,189 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-10 23:22:25,189 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-10 23:22:25,189 INFO Retrain data split: train
2026-05-10 23:22:25,189 INFO Retrain rolling fold selector: latest
2026-05-10 23:22:25,190 INFO === RegimeClassifier retrain (hierarchical: HTF 2-score bias + LTF 5-score behaviour) ===
2026-05-10 23:22:25,364 INFO NumExpr defaulting to 4 threads.
2026-05-10 23:22:25,559 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-10 23:22:25,559 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-10 23:22:25,559 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-10 23:22:25,559 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-10 23:22:25,609 INFO Regime rolling folds selected: [None]
2026-05-10 23:22:25,609 INFO === Regime rolling fold 1/1: train_all ===
2026-05-10 23:22:25,609 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-10 23:22:25,648 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-10 23:22:25,649 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:22:25,663 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:22:25,677 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:22:25,691 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:22:25,706 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:22:25,720 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:22:25,957 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:22:26,029 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:22:26,053 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:22:26,053 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:22:26,064 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:22:26,066 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:22:26,470 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 492, 'BIAS_DOWN': 270, 'BIAS_NEUTRAL': 11340}  ambiguous=6929 (total=12102) horizon=12
2026-05-10 23:22:26,475 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected XAUUSD — 12052 samples (group=gold) score_means={'bias_up_score': 0.0408, 'bias_down_score': 0.0224} labels={'BIAS_UP': 492, 'BIAS_DOWN': 270, 'BIAS_NEUTRAL': 11290} clean={'BIAS_UP': 492, 'BIAS_DOWN': 270, 'BIAS_NEUTRAL': 4396}
2026-05-10 23:22:26,634 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:26,675 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:26,692 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:26,692 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:26,699 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:26,701 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:22:27,056 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 294, 'BIAS_DOWN': 349, 'BIAS_NEUTRAL': 10761}  ambiguous=6552 (total=11404) horizon=12
2026-05-10 23:22:27,061 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURUSD — 11354 samples (group=dollar) score_means={'bias_up_score': 0.0259, 'bias_down_score': 0.0307} labels={'BIAS_UP': 294, 'BIAS_DOWN': 349, 'BIAS_NEUTRAL': 10711} clean={'BIAS_UP': 294, 'BIAS_DOWN': 349, 'BIAS_NEUTRAL': 4188}
2026-05-10 23:22:27,230 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:27,266 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:27,286 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:27,286 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:27,294 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:27,295 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:22:27,666 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 465, 'BIAS_DOWN': 243, 'BIAS_NEUTRAL': 10695}  ambiguous=6644 (total=11403) horizon=12
2026-05-10 23:22:27,671 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected USDJPY — 11353 samples (group=dollar) score_means={'bias_up_score': 0.041, 'bias_down_score': 0.0214} labels={'BIAS_UP': 465, 'BIAS_DOWN': 243, 'BIAS_NEUTRAL': 10645} clean={'BIAS_UP': 465, 'BIAS_DOWN': 243, 'BIAS_NEUTRAL': 4028}
2026-05-10 23:22:27,826 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:27,860 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:27,880 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:27,881 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:27,888 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:27,889 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:22:28,253 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 345, 'BIAS_DOWN': 200, 'BIAS_NEUTRAL': 10862}  ambiguous=6647 (total=11407) horizon=12
2026-05-10 23:22:28,258 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURJPY — 11357 samples (group=cross) score_means={'bias_up_score': 0.0304, 'bias_down_score': 0.0176} labels={'BIAS_UP': 345, 'BIAS_DOWN': 200, 'BIAS_NEUTRAL': 10812} clean={'BIAS_UP': 345, 'BIAS_DOWN': 200, 'BIAS_NEUTRAL': 4191}
2026-05-10 23:22:28,404 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:28,440 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:28,459 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:28,460 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:28,467 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:28,468 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:22:28,824 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 458, 'BIAS_DOWN': 281, 'BIAS_NEUTRAL': 10669}  ambiguous=6611 (total=11408) horizon=12
2026-05-10 23:22:28,829 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPJPY — 11358 samples (group=cross) score_means={'bias_up_score': 0.0403, 'bias_down_score': 0.0247} labels={'BIAS_UP': 458, 'BIAS_DOWN': 281, 'BIAS_NEUTRAL': 10619} clean={'BIAS_UP': 458, 'BIAS_DOWN': 281, 'BIAS_NEUTRAL': 4042}
2026-05-10 23:22:28,977 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:29,012 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:29,030 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:29,030 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:29,038 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:29,039 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:22:29,388 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 269, 'BIAS_DOWN': 344, 'BIAS_NEUTRAL': 10789}  ambiguous=6630 (total=11402) horizon=12
2026-05-10 23:22:29,393 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPUSD — 11352 samples (group=dollar) score_means={'bias_up_score': 0.0237, 'bias_down_score': 0.0303} labels={'BIAS_UP': 269, 'BIAS_DOWN': 344, 'BIAS_NEUTRAL': 10739} clean={'BIAS_UP': 269, 'BIAS_DOWN': 344, 'BIAS_NEUTRAL': 4148}
2026-05-10 23:22:29,454 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 803, 'BIAS_DOWN': 481, 'BIAS_NEUTRAL': 21431}, 'dollar': {'BIAS_UP': 1028, 'BIAS_DOWN': 936, 'BIAS_NEUTRAL': 32095}, 'gold': {'BIAS_UP': 492, 'BIAS_DOWN': 270, 'BIAS_NEUTRAL': 11290}}
2026-05-10 23:22:29,454 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0354, 'bias_down_score': 0.0212}, 'dollar': {'bias_up_score': 0.0302, 'bias_down_score': 0.0275}, 'gold': {'bias_up_score': 0.0408, 'bias_down_score': 0.0224}}
2026-05-10 23:22:29,454 INFO Regime[4H mode=htf_bias] label distribution by year: {2016: {'BIAS_UP': 286, 'BIAS_DOWN': 340, 'BIAS_NEUTRAL': 8196}, 2017: {'BIAS_UP': 461, 'BIAS_DOWN': 205, 'BIAS_NEUTRAL': 8447}, 2018: {'BIAS_UP': 213, 'BIAS_DOWN': 288, 'BIAS_NEUTRAL': 8629}, 2019: {'BIAS_UP': 210, 'BIAS_DOWN': 192, 'BIAS_NEUTRAL': 8700}, 2020: {'BIAS_UP': 298, 'BIAS_DOWN': 180, 'BIAS_NEUTRAL': 8633}, 2021: {'BIAS_UP': 294, 'BIAS_DOWN': 186, 'BIAS_NEUTRAL': 8611}, 2022: {'BIAS_UP': 370, 'BIAS_DOWN': 225, 'BIAS_NEUTRAL': 8526}, 2023: {'BIAS_UP': 191, 'BIAS_DOWN': 71, 'BIAS_NEUTRAL': 5074}}
2026-05-10 23:22:29,454 INFO Regime[4H mode=htf_bias] score means by year: {2016: {'bias_up_score': 0.0324, 'bias_down_score': 0.0385}, 2017: {'bias_up_score': 0.0506, 'bias_down_score': 0.0225}, 2018: {'bias_up_score': 0.0233, 'bias_down_score': 0.0315}, 2019: {'bias_up_score': 0.0231, 'bias_down_score': 0.0211}, 2020: {'bias_up_score': 0.0327, 'bias_down_score': 0.0198}, 2021: {'bias_up_score': 0.0323, 'bias_down_score': 0.0205}, 2022: {'bias_up_score': 0.0406, 'bias_down_score': 0.0247}, 2023: {'bias_up_score': 0.0358, 'bias_down_score': 0.0133}}
2026-05-10 23:22:29,495 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
2026-05-10 23:22:29,496 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
2026-05-10 23:22:29,497 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
2026-05-10 23:22:29,498 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-10 23:22:29,498 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-10 23:22:29,499 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
2026-05-10 23:22:29,516 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:22:29,519 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:22:29,520 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:22:29,521 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:22:29,521 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:22:29,522 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:22:29,748 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 39, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1512}  ambiguous=936 (total=1581) horizon=12
2026-05-10 23:22:29,750 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected XAUUSD — 1531 samples (group=gold) score_means={'bias_up_score': 0.0255, 'bias_down_score': 0.0196} labels={'BIAS_UP': 39, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1462} clean={'BIAS_UP': 39, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 555}
2026-05-10 23:22:29,818 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:29,820 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:29,821 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:29,822 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:29,822 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:29,823 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:22:30,034 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 35, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1403}  ambiguous=861 (total=1491) horizon=12
2026-05-10 23:22:30,036 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURUSD — 1441 samples (group=dollar) score_means={'bias_up_score': 0.0243, 'bias_down_score': 0.0368} labels={'BIAS_UP': 35, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1353} clean={'BIAS_UP': 35, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 532}
2026-05-10 23:22:30,100 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:30,103 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:30,103 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:30,104 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:30,104 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:30,105 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:22:30,310 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 102, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 1362}  ambiguous=886 (total=1489) horizon=12
2026-05-10 23:22:30,313 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected USDJPY — 1439 samples (group=dollar) score_means={'bias_up_score': 0.0709, 'bias_down_score': 0.0174} labels={'BIAS_UP': 102, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 1312} clean={'BIAS_UP': 102, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 454}
2026-05-10 23:22:30,375 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:30,377 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:30,378 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:30,378 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:30,378 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:30,379 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:22:30,584 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 46, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1447}  ambiguous=915 (total=1494) horizon=12
2026-05-10 23:22:30,586 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0319, 'bias_down_score': 0.0007} labels={'BIAS_UP': 46, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1397} clean={'BIAS_UP': 46, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 509}
2026-05-10 23:22:30,650 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:30,652 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:30,653 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:30,653 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:30,653 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:30,654 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:22:30,858 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 29, 'BIAS_DOWN': 21, 'BIAS_NEUTRAL': 1444}  ambiguous=861 (total=1494) horizon=12
2026-05-10 23:22:30,860 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0201, 'bias_down_score': 0.0145} labels={'BIAS_UP': 29, 'BIAS_DOWN': 21, 'BIAS_NEUTRAL': 1394} clean={'BIAS_UP': 29, 'BIAS_DOWN': 21, 'BIAS_NEUTRAL': 561}
2026-05-10 23:22:30,926 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:30,928 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:30,929 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:30,929 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:30,929 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:22:30,930 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:22:31,140 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 26, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 1440}  ambiguous=885 (total=1488) horizon=12
2026-05-10 23:22:31,142 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPUSD — 1438 samples (group=dollar) score_means={'bias_up_score': 0.0181, 'bias_down_score': 0.0153} labels={'BIAS_UP': 26, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 1390} clean={'BIAS_UP': 26, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 543}
2026-05-10 23:22:31,200 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 75, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 2791}, 'dollar': {'BIAS_UP': 163, 'BIAS_DOWN': 100, 'BIAS_NEUTRAL': 4055}, 'gold': {'BIAS_UP': 39, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1462}}
2026-05-10 23:22:31,200 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.026, 'bias_down_score': 0.0076}, 'dollar': {'bias_up_score': 0.0377, 'bias_down_score': 0.0232}, 'gold': {'bias_up_score': 0.0255, 'bias_down_score': 0.0196}}
2026-05-10 23:22:31,200 INFO Regime[4H mode=htf_bias] label distribution by year: {2022: {'BIAS_UP': 91, 'BIAS_DOWN': 81, 'BIAS_NEUTRAL': 3229}, 2023: {'BIAS_UP': 186, 'BIAS_DOWN': 71, 'BIAS_NEUTRAL': 5079}}
2026-05-10 23:22:31,200 INFO Regime[4H mode=htf_bias] score means by year: {2022: {'bias_up_score': 0.0268, 'bias_down_score': 0.0238}, 2023: {'bias_up_score': 0.0349, 'bias_down_score': 0.0133}}
2026-05-10 23:22:31,239 INFO Regime phase HTF dataset build fold=train_all: 5.6s (train=68826 val=8737)
2026-05-10 23:22:31,239 INFO Regime 4H/htf_bias cold start: no existing weights found
2026-05-10 23:22:31,245 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=68826 val=8737 train_labels={'BIAS_UP': 2323, 'BIAS_DOWN': 1687, 'BIAS_NEUTRAL': 64816} val_labels={'BIAS_UP': 277, 'BIAS_DOWN': 152, 'BIAS_NEUTRAL': 8308}
2026-05-10 23:22:31,434 INFO RegimeClassifier[mode=htf_bias]: cold start HTF score head
2026-05-10 23:22:31,435 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-10 23:22:31,435 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 28.628, 'bias_down_score': 30.0}
2026-05-10 23:22:31,438 INFO RegimeClassifier[mode=htf_bias]: HTF balanced sampler — dir=4010 neutral=64816 dir_weight=15 => dir_frac_per_epoch≈48.1%
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../models/regime_classifier.py:2288: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  scheduler.step()
2026-05-10 23:22:35,196 INFO Regime HTF score epoch  1/50 — tr=39.2403 va=2.5432 acc=0.951 bal=0.333 threshold=0.60 margin=0.10 recall={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 1.0} precision={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 0.951}
2026-05-10 23:22:36,832 INFO Regime HTF score epoch  2/50 — tr=38.2331 va=2.4253 bal=0.333
2026-05-10 23:22:38,494 INFO Regime HTF score epoch  3/50 — tr=36.0870 va=2.2217 bal=0.333
2026-05-10 23:22:40,127 INFO Regime HTF score epoch  4/50 — tr=32.3801 va=1.8975 bal=0.333
2026-05-10 23:22:41,773 INFO Regime HTF score epoch  5/50 — tr=27.4005 va=1.5046 acc=0.950 bal=0.333 threshold=0.60 margin=0.10 recall={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 0.999} precision={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 0.951}
2026-05-10 23:22:43,447 INFO Regime HTF score epoch  6/50 — tr=22.2172 va=1.1785 bal=0.372
2026-05-10 23:22:45,090 INFO Regime HTF score epoch  7/50 — tr=17.6247 va=0.9475 bal=0.497
2026-05-10 23:22:46,728 INFO Regime HTF score epoch  8/50 — tr=13.6367 va=0.7579 bal=0.459
2026-05-10 23:22:48,379 INFO Regime HTF score epoch  9/50 — tr=10.0613 va=0.6060 bal=0.430
2026-05-10 23:22:50,034 INFO Regime HTF score epoch 10/50 — tr=7.3012 va=0.4958 acc=0.941 bal=0.416 threshold=0.92 margin=0.00 recall={'BIAS_UP': 0.181, 'BIAS_DOWN': 0.086, 'BIAS_NEUTRAL': 0.982} precision={'BIAS_UP': 0.296, 'BIAS_DOWN': 0.283, 'BIAS_NEUTRAL': 0.957}
2026-05-10 23:22:51,690 INFO Regime HTF score epoch 11/50 — tr=5.0820 va=0.4256 bal=0.388
2026-05-10 23:22:53,406 INFO Regime HTF score epoch 12/50 — tr=3.4071 va=0.3973 bal=0.427
2026-05-10 23:22:55,065 INFO Regime HTF score epoch 13/50 — tr=2.1500 va=0.4113 bal=0.376
2026-05-10 23:22:56,805 INFO Regime HTF score epoch 14/50 — tr=1.3645 va=0.4577 bal=0.436
2026-05-10 23:22:58,466 INFO Regime HTF score epoch 15/50 — tr=0.8740 va=0.5312 acc=0.914 bal=0.556 threshold=0.99 margin=0.00 recall={'BIAS_UP': 0.43, 'BIAS_DOWN': 0.296, 'BIAS_NEUTRAL': 0.942} precision={'BIAS_UP': 0.264, 'BIAS_DOWN': 0.228, 'BIAS_NEUTRAL': 0.967}
2026-05-10 23:23:00,129 INFO Regime HTF score epoch 16/50 — tr=0.5654 va=0.5894 bal=0.677
2026-05-10 23:23:00,129 INFO Regime HTF score early stop at epoch 16
2026-05-10 23:23:01,626 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.700 margin=0.000 precision={'BIAS_UP': 0.327, 'BIAS_DOWN': 0.283, 'BIAS_NEUTRAL': 0.961} recall={'BIAS_UP': 0.318, 'BIAS_DOWN': 0.086, 'BIAS_NEUTRAL': 0.974} f1={'BIAS_UP': 0.322, 'BIAS_DOWN': 0.131, 'BIAS_NEUTRAL': 0.968} confusion=[[88, 0, 189], [0, 13, 139], [181, 33, 8094]] score_mae={'bias_up_score': 0.0936, 'bias_down_score': 0.0592} pred_share={'BIAS_UP': 0.0308, 'BIAS_DOWN': 0.0053, 'BIAS_NEUTRAL': 0.9639}
2026-05-10 23:23:01,627 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.327, 'BIAS_DOWN': 0.283, 'BIAS_NEUTRAL': 0.961} min_precision=0.500 recall={'BIAS_UP': 0.318, 'BIAS_DOWN': 0.086, 'BIAS_NEUTRAL': 0.974} min_recall=0.100 f1={'BIAS_UP': 0.322, 'BIAS_DOWN': 0.131, 'BIAS_NEUTRAL': 0.968} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_UP', 'BIAS_DOWN'] weak_recall=['BIAS_DOWN'] weak_f1=['BIAS_DOWN'] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-10 23:23:01,630 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-10 23:23:01,630 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-10 23:23:01,631 INFO Regime phase HTF train fold=train_all: 30.4s
2026-05-10 23:23:01,730 INFO Regime HTF complete fold=train_all: acc=0.938 bal=0.459 train=68826 val=8737 per_class={'BIAS_UP': 0.318, 'BIAS_DOWN': 0.086, 'BIAS_NEUTRAL': 0.974} precision={'BIAS_UP': 0.327, 'BIAS_DOWN': 0.283, 'BIAS_NEUTRAL': 0.961} threshold=0.700 margin=0.000
2026-05-10 23:23:01,732 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:23:01,884 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 269, 'BIAS_DOWN': 344, 'BIAS_NEUTRAL': 10789}  ambiguous=6630 (total=11402) horizon=12
2026-05-10 23:23:01,889 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on GBPUSD 4H:
{'BIAS_UP': 3.0568181818181817, 'BIAS_DOWN': 3.909090909090909, 'BIAS_NEUTRAL': 60.954802259887}
2026-05-10 23:23:01,893 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (all labels):
{'BIAS_UP': {'n': 269, 'mean': 0.0009352809682544664, 'mean_over_std': 0.46065867099997965}, 'BIAS_DOWN': {'n': 344, 'mean': -0.0013986222214448377, 'mean_over_std': -0.453155978736302}, 'BIAS_NEUTRAL': {'n': 10788, 'mean': 1.121563318643874e-05, 'mean_over_std': 0.0043231848821040425}}
2026-05-10 23:23:01,893 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 269, 'mean': 0.0009352809682544664, 'mean_over_std': 0.46065867099997965}, 'BIAS_DOWN': {'n': 344, 'mean': -0.0013986222214448377, 'mean_over_std': -0.453155978736302}, 'BIAS_NEUTRAL': {'n': 4159, 'mean': 1.3724894091827828e-05, 'mean_over_std': 0.006431864931044914}}
2026-05-10 23:23:01,901 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-10 23:23:01,903 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:23:01,905 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:23:01,907 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:23:01,908 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:23:01,910 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:23:01,912 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:23:01,930 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:23:01,938 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:23:01,941 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:23:01,941 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:23:01,942 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:23:01,947 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:23:02,821 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected XAUUSD — 44227 samples (group=gold) score_means={'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}
2026-05-10 23:23:02,922 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:02,924 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:02,925 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:02,926 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:02,926 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:02,928 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:23:03,711 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURUSD — 43689 samples (group=dollar) score_means={'trend_score': 0.49, 'range_score': 0.236, 'chop_score': 0.4622, 'volatility_percentile': 0.3812, 'consolidation_score': 0.1862}
2026-05-10 23:23:03,813 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:03,815 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:03,816 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:03,817 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:03,817 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:03,819 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:23:04,623 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected USDJPY — 43686 samples (group=dollar) score_means={'trend_score': 0.4949, 'range_score': 0.2312, 'chop_score': 0.4613, 'volatility_percentile': 0.38, 'consolidation_score': 0.1931}
2026-05-10 23:23:04,727 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:04,729 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:04,730 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:04,730 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:04,731 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:04,733 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:23:05,521 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURJPY — 43688 samples (group=cross) score_means={'trend_score': 0.4888, 'range_score': 0.2345, 'chop_score': 0.4662, 'volatility_percentile': 0.3831, 'consolidation_score': 0.1882}
2026-05-10 23:23:05,622 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:05,624 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:05,625 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:05,625 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:05,626 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:05,628 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:23:06,415 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPJPY — 43678 samples (group=cross) score_means={'trend_score': 0.4878, 'range_score': 0.2348, 'chop_score': 0.4673, 'volatility_percentile': 0.3839, 'consolidation_score': 0.1904}
2026-05-10 23:23:06,517 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:06,520 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:06,520 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:06,521 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:06,521 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:06,523 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:23:07,340 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPUSD — 43676 samples (group=dollar) score_means={'trend_score': 0.4923, 'range_score': 0.2335, 'chop_score': 0.4611, 'volatility_percentile': 0.3832, 'consolidation_score': 0.1856}
2026-05-10 23:23:07,448 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2346, 'chop_score': 0.4667, 'volatility_percentile': 0.3835, 'consolidation_score': 0.1893}, 'dollar': {'trend_score': 0.4924, 'range_score': 0.2335, 'chop_score': 0.4615, 'volatility_percentile': 0.3815, 'consolidation_score': 0.1883}, 'gold': {'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}}
2026-05-10 23:23:07,448 INFO Regime[1H mode=ltf_behaviour] score means by year: {2016: {'trend_score': 0.495, 'range_score': 0.233, 'chop_score': 0.4607, 'volatility_percentile': 0.3776, 'consolidation_score': 0.192}, 2017: {'trend_score': 0.5017, 'range_score': 0.2328, 'chop_score': 0.4574, 'volatility_percentile': 0.3689, 'consolidation_score': 0.1942}, 2018: {'trend_score': 0.4886, 'range_score': 0.237, 'chop_score': 0.4655, 'volatility_percentile': 0.3947, 'consolidation_score': 0.1785}, 2019: {'trend_score': 0.4817, 'range_score': 0.236, 'chop_score': 0.4691, 'volatility_percentile': 0.3725, 'consolidation_score': 0.198}, 2020: {'trend_score': 0.4874, 'range_score': 0.2311, 'chop_score': 0.4674, 'volatility_percentile': 0.3848, 'consolidation_score': 0.1919}, 2021: {'trend_score': 0.4872, 'range_score': 0.2372, 'chop_score': 0.4672, 'volatility_percentile': 0.377, 'consolidation_score': 0.192}, 2022: {'trend_score': 0.4893, 'range_score': 0.2357, 'chop_score': 0.4647, 'volatility_percentile': 0.3962, 'consolidation_score': 0.1784}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-10 23:23:07,531 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
2026-05-10 23:23:07,533 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-10 23:23:07,534 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
2026-05-10 23:23:07,535 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-10 23:23:07,536 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
2026-05-10 23:23:07,538 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
2026-05-10 23:23:07,547 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:23:07,550 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:23:07,551 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:23:07,552 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:23:07,552 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:23:07,554 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:23:07,796 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected XAUUSD — 5117 samples (group=gold) score_means={'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}
2026-05-10 23:23:07,901 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:07,904 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:07,904 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:07,905 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:07,905 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:07,907 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:23:08,128 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURUSD — 5053 samples (group=dollar) score_means={'trend_score': 0.4883, 'range_score': 0.2413, 'chop_score': 0.463, 'volatility_percentile': 0.3837, 'consolidation_score': 0.1793}
2026-05-10 23:23:08,231 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:08,233 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:08,234 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:08,234 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:08,234 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:08,236 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:23:08,469 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected USDJPY — 5046 samples (group=dollar) score_means={'trend_score': 0.5085, 'range_score': 0.2259, 'chop_score': 0.4506, 'volatility_percentile': 0.3809, 'consolidation_score': 0.1888}
2026-05-10 23:23:08,573 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:08,575 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:08,576 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:08,576 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:08,577 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:08,578 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:23:08,804 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURJPY — 5053 samples (group=cross) score_means={'trend_score': 0.5023, 'range_score': 0.2261, 'chop_score': 0.4524, 'volatility_percentile': 0.3837, 'consolidation_score': 0.18}
2026-05-10 23:23:08,910 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:08,913 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:08,913 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:08,914 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:08,914 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:08,916 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:23:09,137 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPJPY — 5043 samples (group=cross) score_means={'trend_score': 0.4743, 'range_score': 0.2372, 'chop_score': 0.4716, 'volatility_percentile': 0.3777, 'consolidation_score': 0.1917}
2026-05-10 23:23:09,242 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:09,244 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:09,245 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:09,245 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:09,245 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:23:09,247 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:23:09,470 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPUSD — 5040 samples (group=dollar) score_means={'trend_score': 0.4843, 'range_score': 0.2371, 'chop_score': 0.4614, 'volatility_percentile': 0.377, 'consolidation_score': 0.1835}
2026-05-10 23:23:09,569 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2317, 'chop_score': 0.462, 'volatility_percentile': 0.3807, 'consolidation_score': 0.1858}, 'dollar': {'trend_score': 0.4937, 'range_score': 0.2348, 'chop_score': 0.4583, 'volatility_percentile': 0.3806, 'consolidation_score': 0.1839}, 'gold': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}}
2026-05-10 23:23:09,569 INFO Regime[1H mode=ltf_behaviour] score means by year: {2022: {'trend_score': 0.4905, 'range_score': 0.2364, 'chop_score': 0.4625, 'volatility_percentile': 0.385, 'consolidation_score': 0.1797}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-10 23:23:09,643 INFO Regime phase LTF dataset build fold=train_all: 7.7s (train=262644 val=30352)
2026-05-10 23:23:09,644 INFO Regime 1H/ltf_behaviour cold start: no existing weights found
2026-05-10 23:23:09,668 INFO RegimeClassifier[mode=ltf_behaviour]: cold start score head
2026-05-10 23:23:09,669 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-10 23:23:10,172 INFO Regime score epoch  1/50 — tr=0.0866 va=0.0694 mae={'trend_score': 0.1743, 'range_score': 0.2493, 'chop_score': 0.1508, 'volatility_percentile': 0.1853, 'consolidation_score': 0.3398}
2026-05-10 23:23:10,687 INFO Regime score epoch  2/50 — tr=0.0721 va=0.0525
2026-05-10 23:23:11,182 INFO Regime score epoch  3/50 — tr=0.0527 va=0.0358
2026-05-10 23:23:11,658 INFO Regime score epoch  4/50 — tr=0.0358 va=0.0244
2026-05-10 23:23:12,184 INFO Regime score epoch  5/50 — tr=0.0247 va=0.0163 mae={'trend_score': 0.0685, 'range_score': 0.1142, 'chop_score': 0.0666, 'volatility_percentile': 0.0481, 'consolidation_score': 0.1903}
2026-05-10 23:23:12,668 INFO Regime score epoch  6/50 — tr=0.0179 va=0.0111
2026-05-10 23:23:13,147 INFO Regime score epoch  7/50 — tr=0.0139 va=0.0078
2026-05-10 23:23:13,632 INFO Regime score epoch  8/50 — tr=0.0116 va=0.0061
2026-05-10 23:23:14,111 INFO Regime score epoch  9/50 — tr=0.0102 va=0.0051
2026-05-10 23:23:14,619 INFO Regime score epoch 10/50 — tr=0.0092 va=0.0044 mae={'trend_score': 0.0531, 'range_score': 0.056, 'chop_score': 0.0517, 'volatility_percentile': 0.032, 'consolidation_score': 0.0673}
2026-05-10 23:23:15,108 INFO Regime score epoch 11/50 — tr=0.0086 va=0.0040
2026-05-10 23:23:15,598 INFO Regime score epoch 12/50 — tr=0.0081 va=0.0038
2026-05-10 23:23:16,083 INFO Regime score epoch 13/50 — tr=0.0077 va=0.0035
2026-05-10 23:23:16,582 INFO Regime score epoch 14/50 — tr=0.0074 va=0.0034
2026-05-10 23:23:17,081 INFO Regime score epoch 15/50 — tr=0.0072 va=0.0031 mae={'trend_score': 0.0457, 'range_score': 0.0515, 'chop_score': 0.0473, 'volatility_percentile': 0.0269, 'consolidation_score': 0.0457}
2026-05-10 23:23:17,588 INFO Regime score epoch 16/50 — tr=0.0069 va=0.0030
2026-05-10 23:23:18,070 INFO Regime score epoch 17/50 — tr=0.0067 va=0.0029
2026-05-10 23:23:18,564 INFO Regime score epoch 18/50 — tr=0.0065 va=0.0028
2026-05-10 23:23:19,044 INFO Regime score epoch 19/50 — tr=0.0063 va=0.0026
2026-05-10 23:23:19,540 INFO Regime score epoch 20/50 — tr=0.0062 va=0.0025 mae={'trend_score': 0.0402, 'range_score': 0.0477, 'chop_score': 0.0429, 'volatility_percentile': 0.0243, 'consolidation_score': 0.0381}
2026-05-10 23:23:20,035 INFO Regime score epoch 21/50 — tr=0.0060 va=0.0024
2026-05-10 23:23:20,543 INFO Regime score epoch 22/50 — tr=0.0059 va=0.0023
2026-05-10 23:23:21,026 INFO Regime score epoch 23/50 — tr=0.0058 va=0.0022
2026-05-10 23:23:21,523 INFO Regime score epoch 24/50 — tr=0.0056 va=0.0022
2026-05-10 23:23:22,025 INFO Regime score epoch 25/50 — tr=0.0055 va=0.0021 mae={'trend_score': 0.035, 'range_score': 0.0445, 'chop_score': 0.0375, 'volatility_percentile': 0.0224, 'consolidation_score': 0.0337}
2026-05-10 23:23:22,537 INFO Regime score epoch 26/50 — tr=0.0055 va=0.0020
2026-05-10 23:23:23,018 INFO Regime score epoch 27/50 — tr=0.0054 va=0.0020
2026-05-10 23:23:23,497 INFO Regime score epoch 28/50 — tr=0.0053 va=0.0020
2026-05-10 23:23:23,978 INFO Regime score epoch 29/50 — tr=0.0052 va=0.0019
2026-05-10 23:23:24,460 INFO Regime score epoch 30/50 — tr=0.0052 va=0.0018 mae={'trend_score': 0.0312, 'range_score': 0.0432, 'chop_score': 0.034, 'volatility_percentile': 0.0211, 'consolidation_score': 0.0319}
2026-05-10 23:23:24,947 INFO Regime score epoch 31/50 — tr=0.0051 va=0.0018
2026-05-10 23:23:25,440 INFO Regime score epoch 32/50 — tr=0.0051 va=0.0018
2026-05-10 23:23:25,922 INFO Regime score epoch 33/50 — tr=0.0050 va=0.0017
2026-05-10 23:23:26,412 INFO Regime score epoch 34/50 — tr=0.0050 va=0.0017
2026-05-10 23:23:26,926 INFO Regime score epoch 35/50 — tr=0.0050 va=0.0017 mae={'trend_score': 0.0291, 'range_score': 0.042, 'chop_score': 0.032, 'volatility_percentile': 0.0206, 'consolidation_score': 0.0305}
2026-05-10 23:23:27,428 INFO Regime score epoch 36/50 — tr=0.0049 va=0.0017
2026-05-10 23:23:27,923 INFO Regime score epoch 37/50 — tr=0.0049 va=0.0017
2026-05-10 23:23:28,435 INFO Regime score epoch 38/50 — tr=0.0049 va=0.0017
2026-05-10 23:23:28,930 INFO Regime score epoch 39/50 — tr=0.0049 va=0.0016
2026-05-10 23:23:29,441 INFO Regime score epoch 40/50 — tr=0.0049 va=0.0016 mae={'trend_score': 0.0278, 'range_score': 0.0414, 'chop_score': 0.031, 'volatility_percentile': 0.0197, 'consolidation_score': 0.0311}
2026-05-10 23:23:29,947 INFO Regime score epoch 41/50 — tr=0.0049 va=0.0016
2026-05-10 23:23:30,455 INFO Regime score epoch 42/50 — tr=0.0048 va=0.0016
2026-05-10 23:23:30,955 INFO Regime score epoch 43/50 — tr=0.0048 va=0.0016
2026-05-10 23:23:31,440 INFO Regime score epoch 44/50 — tr=0.0048 va=0.0016
2026-05-10 23:23:31,932 INFO Regime score epoch 45/50 — tr=0.0048 va=0.0016 mae={'trend_score': 0.0275, 'range_score': 0.041, 'chop_score': 0.0306, 'volatility_percentile': 0.0196, 'consolidation_score': 0.0294}
2026-05-10 23:23:32,461 INFO Regime score epoch 46/50 — tr=0.0048 va=0.0016
2026-05-10 23:23:32,970 INFO Regime score epoch 47/50 — tr=0.0048 va=0.0016
2026-05-10 23:23:33,457 INFO Regime score epoch 48/50 — tr=0.0048 va=0.0016
2026-05-10 23:23:33,956 INFO Regime score epoch 49/50 — tr=0.0048 va=0.0016
2026-05-10 23:23:34,455 INFO Regime score epoch 50/50 — tr=0.0048 va=0.0016 mae={'trend_score': 0.0274, 'range_score': 0.0415, 'chop_score': 0.0307, 'volatility_percentile': 0.0194, 'consolidation_score': 0.0301}
2026-05-10 23:23:34,476 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0275, 'range_score': 0.041, 'chop_score': 0.0306, 'volatility_percentile': 0.0196, 'consolidation_score': 0.0294} mse={'trend_score': 0.00126, 'range_score': 0.0027, 'chop_score': 0.00152, 'volatility_percentile': 0.00077, 'consolidation_score': 0.0017} corr={'trend_score': 0.9872, 'range_score': 0.9326, 'chop_score': 0.979, 'volatility_percentile': 0.9918, 'consolidation_score': 0.9821} pred_std={'trend_score': 0.2181, 'range_score': 0.1363, 'chop_score': 0.1782, 'volatility_percentile': 0.215, 'consolidation_score': 0.2139} target_std={'trend_score': 0.2219, 'range_score': 0.1433, 'chop_score': 0.1888, 'volatility_percentile': 0.2165, 'consolidation_score': 0.2179}
2026-05-10 23:23:34,792 INFO RegimeClassifier[mode=ltf_behaviour] train group diagnostics={'dollar': {'n': 16632, 'score_mae': {'trend_score': 0.0263, 'range_score': 0.0406, 'chop_score': 0.0304, 'volatility_percentile': 0.0198, 'consolidation_score': 0.0297}, 'target_score_mean': {'trend_score': 0.491, 'range_score': 0.2347, 'chop_score': 0.4613, 'volatility_percentile': 0.3824, 'consolidation_score': 0.186}, 'pred_score_mean': {'trend_score': 0.49, 'range_score': 0.239, 'chop_score': 0.4612, 'volatility_percentile': 0.3836, 'consolidation_score': 0.1892}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3532, 38, 0, 3, 0, 0, 206], [19, 87, 0, 0, 0, 0, 4], [0, 0, 214, 10, 49, 0, 187], [1, 0, 12, 544, 45, 0, 87], [0, 0, 68, 36, 3008, 0, 204], [0, 31, 0, 0, 8, 0, 89], [228, 18, 260, 118, 129, 0, 7397]]}, 'gold': {'n': 8420, 'score_mae': {'trend_score': 0.0259, 'range_score': 0.0412, 'chop_score': 0.0303, 'volatility_percentile': 0.0201, 'consolidation_score': 0.0301}, 'target_score_mean': {'trend_score': 0.4883, 'range_score': 0.2355, 'chop_score': 0.4652, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}, 'pred_score_mean': {'trend_score': 0.4885, 'range_score': 0.2394, 'chop_score': 0.4644, 'volatility_percentile': 0.3789, 'consolidation_score': 0.1944}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1778, 21, 0, 0, 0, 0, 86], [7, 46, 0, 0, 0, 0, 3], [0, 0, 122, 8, 24, 0, 90], [1, 0, 9, 330, 24, 0, 52], [0, 0, 39, 30, 1529, 0, 106], [0, 27, 0, 0, 9, 0, 45], [106, 7, 150, 68, 68, 0, 3635]]}, 'jpy': {'n': 24948, 'score_mae': {'trend_score': 0.0265, 'range_score': 0.0407, 'chop_score': 0.0299, 'volatility_percentile': 0.0198, 'consolidation_score': 0.0295}, 'target_score_mean': {'trend_score': 0.4904, 'range_score': 0.2338, 'chop_score': 0.4653, 'volatility_percentile': 0.3826, 'consolidation_score': 0.1905}, 'pred_score_mean': {'trend_score': 0.4889, 'range_score': 0.2386, 'chop_score': 0.4647, 'volatility_percentile': 0.3835, 'consolidation_score': 0.193}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[5265, 78, 0, 5, 0, 0, 298], [40, 137, 0, 0, 0, 0, 10], [0, 0, 287, 20, 71, 0, 269], [1, 0, 18, 1071, 84, 0, 140], [0, 0, 121, 84, 4570, 0, 340], [0, 64, 0, 0, 16, 0, 143], [318, 18, 389, 157, 249, 0, 10685]]}}
2026-05-10 23:23:34,966 INFO RegimeClassifier[mode=ltf_behaviour] validation group diagnostics={'dollar': {'n': 10093, 'score_mae': {'trend_score': 0.0274, 'range_score': 0.0423, 'chop_score': 0.0306, 'volatility_percentile': 0.0194, 'consolidation_score': 0.0288}, 'target_score_mean': {'trend_score': 0.4863, 'range_score': 0.2392, 'chop_score': 0.4622, 'volatility_percentile': 0.3804, 'consolidation_score': 0.1814}, 'pred_score_mean': {'trend_score': 0.4867, 'range_score': 0.2416, 'chop_score': 0.4619, 'volatility_percentile': 0.3821, 'consolidation_score': 0.1857}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[2262, 17, 0, 3, 0, 0, 130], [10, 43, 0, 0, 0, 0, 0], [0, 0, 149, 8, 44, 0, 115], [1, 0, 8, 339, 31, 0, 44], [0, 0, 51, 33, 1860, 0, 106], [0, 24, 0, 0, 3, 0, 50], [122, 10, 172, 76, 84, 0, 4298]]}, 'gold': {'n': 5117, 'score_mae': {'trend_score': 0.0264, 'range_score': 0.04, 'chop_score': 0.0303, 'volatility_percentile': 0.0196, 'consolidation_score': 0.0307}, 'target_score_mean': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}, 'pred_score_mean': {'trend_score': 0.4951, 'range_score': 0.2392, 'chop_score': 0.4577, 'volatility_percentile': 0.384, 'consolidation_score': 0.1868}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1084, 14, 0, 0, 0, 0, 69], [10, 23, 0, 0, 0, 0, 2], [0, 0, 89, 2, 13, 0, 67], [0, 0, 4, 214, 14, 0, 23], [0, 0, 20, 11, 795, 0, 61], [0, 14, 0, 0, 3, 0, 33], [68, 3, 106, 51, 50, 0, 2274]]}, 'jpy': {'n': 15142, 'score_mae': {'trend_score': 0.0279, 'range_score': 0.0406, 'chop_score': 0.0306, 'volatility_percentile': 0.0198, 'consolidation_score': 0.0294}, 'target_score_mean': {'trend_score': 0.495, 'range_score': 0.2297, 'chop_score': 0.4582, 'volatility_percentile': 0.3808, 'consolidation_score': 0.1868}, 'pred_score_mean': {'trend_score': 0.4926, 'range_score': 0.2343, 'chop_score': 0.4586, 'volatility_percentile': 0.3823, 'consolidation_score': 0.1899}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3316, 40, 0, 2, 0, 0, 160], [27, 81, 0, 0, 0, 0, 7], [0, 0, 177, 12, 56, 0, 139], [2, 0, 20, 663, 45, 0, 97], [0, 0, 76, 41, 2506, 0, 194], [0, 32, 0, 0, 10, 0, 80], [183, 17, 231, 91, 156, 0, 6681]]}}
2026-05-10 23:23:34,971 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-10 23:23:34,971 INFO Regime phase LTF train fold=train_all: 25.3s
2026-05-10 23:23:35,073 INFO Regime LTF complete fold=train_all: score_accuracy=0.970, train=262644 val=30352 mae={'trend_score': 0.0275, 'range_score': 0.041, 'chop_score': 0.0306, 'volatility_percentile': 0.0196, 'consolidation_score': 0.0294}
2026-05-10 23:23:35,075 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:23:35,417 INFO Regime[1H mode=ltf_behaviour fold=latest] LTF score diagnostics on GBPUSD:
{'trend_score': {'mean': 0.4921, 'q10': 0.1918, 'q50': 0.4877, 'q90': 0.8002}, 'range_score': {'mean': 0.2337, 'q10': 0.0513, 'q50': 0.2125, 'q90': 0.4349}, 'chop_score': {'mean': 0.4612, 'q10': 0.2148, 'q50': 0.4503, 'q90': 0.7255}, 'volatility_percentile': {'mean': 0.3832, 'q10': 0.1018, 'q50': 0.3704, 'q90': 0.6823}, 'consolidation_score': {'mean': 0.1854, 'q10': 0.0, 'q50': 0.1185, 'q90': 0.506}}
2026-05-10 23:23:35,426 INFO Regime retrain total: 70.2s (370559 train+val samples)
2026-05-10 23:23:35,434 INFO Retrain complete. Total wall-clock: 70.2s
2026-05-10 23:23:36,397 INFO Model regime: SUCCESS
2026-05-10 23:23:36,398 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:23:36,398 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-10 23:23:36,398 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-10 23:23:36,398 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-05-10 23:23:36,398 INFO   [DORMANT] rl_ppo — RL_ENABLED=false, skipping
2026-05-10 23:23:36,398 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-05-10 23:23:36,398 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer']
2026-05-10 23:23:36,410 INFO Saved 46 retrain records to metrics/

=== TRAINING COMPLETE ===
  gru: SUCCESS
  regime: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Clean Quality/RL source: Backtest on train window ===
  START Train-window backtest for Quality/RL labels
2026-05-10 23:23:37,033 INFO === STEP 6: BACKTEST (train) ===
2026-05-10 23:23:37,034 INFO BT_WINDOW=train — train-window backtest: 2016-01-04 → 2023-08-04 (clean Quality/RL labels)
2026-05-10 23:23:37,035 INFO Cleared existing journal for fresh train run
2026-05-10 23:23:37,035 INFO ================================================================
  ROUND 0 / 3
================================================================
2026-05-10 23:23:37,035 INFO Round 0 — running backtest: 2016-01-04 → 2023-08-04 (ml_trader, shared ML cache)
2026-05-10 23:27:30,141 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for EURUSD with 2
2026-05-10 23:27:30,155 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for EURUSD with 0.3333333333333333
2026-05-10 23:27:30,275 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for EURJPY with 2
2026-05-10 23:27:30,290 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for EURJPY with 0.3333333333333333
2026-05-10 23:27:30,417 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for USDJPY with 2
2026-05-10 23:27:30,442 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for USDJPY with 0.3333333333333333
2026-05-10 23:27:30,552 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for EURUSD with 2
2026-05-10 23:27:30,565 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for EURUSD with 0.25
2026-05-10 23:27:30,609 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for EURUSD: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-10 23:27:30,842 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for EURJPY with 2
2026-05-10 23:27:30,868 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for EURJPY with 0.25
2026-05-10 23:27:30,901 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for EURJPY: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-10 23:27:31,059 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for USDJPY with 2
2026-05-10 23:27:31,077 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for USDJPY with 0.25
2026-05-10 23:27:31,112 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for USDJPY: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-10 23:27:31,668 WARNING ML cache score overlay filled 4 warmup/alignment gaps for EURUSD
2026-05-10 23:27:35,422 WARNING ML cache score overlay filled 4 warmup/alignment gaps for USDJPY
2026-05-10 23:27:35,422 WARNING ML cache score overlay filled 4 warmup/alignment gaps for EURJPY
2026-05-10 23:27:45,906 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
2026-05-10 23:27:48,314 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
2026-05-10 23:27:49,075 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-10 23:27:49,897 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-10 23:27:49,947 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:757: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-10 23:27:49,974 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-10 23:27:50,015 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-10 23:27:50,044 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:765: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-10 23:27:50,071 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-10 23:27:50,106 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
2026-05-10 23:27:50,112 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:771: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-10 23:27:50,179 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-10 23:27:50,217 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
2026-05-10 23:27:50,235 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
2026-05-10 23:27:50,273 WARNING _build_sequence_df: HTF frame 5M filled 357 warmup/alignment gaps with 0.000
2026-05-10 23:27:50,305 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-10 23:27:50,347 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-10 23:27:50,367 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
2026-05-10 23:27:50,412 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:757: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-10 23:27:50,420 WARNING _build_sequence_df: HTF frame 5M filled 357 warmup/alignment gaps with 0.000
2026-05-10 23:27:50,439 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-10 23:27:50,461 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-10 23:27:50,490 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-10 23:27:50,529 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:765: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-10 23:27:50,546 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-10 23:27:50,568 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-10 23:27:50,602 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:771: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-10 23:27:50,624 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:757: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:757: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-10 23:27:50,658 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-10 23:27:50,711 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-10 23:27:50,770 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:765: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-10 23:27:50,799 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:765: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-10 23:27:50,826 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:771: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-10 23:27:50,865 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:771: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-10 23:27:51,035 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-10 23:28:03,078 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for GBPJPY with 2
2026-05-10 23:28:03,100 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for GBPJPY with 0.3333333333333333
2026-05-10 23:28:03,300 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for GBPJPY with 2
2026-05-10 23:28:03,315 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for GBPJPY with 0.25
2026-05-10 23:28:03,333 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for GBPJPY: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-10 23:28:03,503 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for GBPUSD with 2
2026-05-10 23:28:03,517 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for GBPUSD with 0.3333333333333333
2026-05-10 23:28:03,696 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for GBPUSD with 2
2026-05-10 23:28:03,709 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for GBPUSD with 0.25
2026-05-10 23:28:03,725 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for GBPUSD: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-10 23:28:03,967 WARNING ML cache score overlay filled 4 warmup/alignment gaps for GBPJPY
2026-05-10 23:28:08,270 WARNING ML cache score overlay filled 4 warmup/alignment gaps for GBPUSD
2026-05-10 23:28:08,812 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
2026-05-10 23:28:09,352 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
2026-05-10 23:28:09,864 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-10 23:28:10,300 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-10 23:28:10,721 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:757: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-10 23:28:10,935 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-10 23:28:11,354 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-10 23:28:11,647 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:765: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-10 23:28:11,814 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-10 23:28:12,000 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:771: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-10 23:28:12,785 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-10 23:28:12,898 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
2026-05-10 23:28:12,962 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
2026-05-10 23:28:12,996 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-10 23:28:13,017 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-10 23:28:13,146 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:757: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-10 23:28:13,173 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-10 23:28:13,211 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-10 23:28:13,244 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:765: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-10 23:28:13,272 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-10 23:28:13,298 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:771: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-10 23:28:13,379 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260510_232339.json
Trader                                   Trades      WR    PF*  Return*    ExpR   TP1%   TP2%     DD*  Sharpe*  Sortino     IC  Verdict
-----------------------------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)              343  25.1%   0.93  -17.3%  -0.051 25.1%  7.0%  61.0%    -0.47    -0.05  0.005     FAIL
  FAILED rules: positive_expectancy, profit_factor_min_1_25, drawdown_below_20pct, sharpe_positive, sortino_positive, win_rate_above_breakeven, t_stat_above_1_5, sharpe_ci_positive
  monthly R: 2022-09=+11.70  2022-10=-5.55  2022-11=-1.00  2023-05=-2.00  2023-06=+16.81  2023-07=+1.68
  MonteCarlo P95 DD=52.2%  P10 equity=8,266  t=-0.55 (p=0.583)  Sharpe CI=[-2.25, 1.27]  streak=40
  gate_diagnostics: bars=1049680 no_signal=486658 quality_block=0 session_skip=562433 density=246 pm_reject=0 daily_skip=0 cooldown=0 daily_halt_events=0 enforce_daily_halt=False
  no_signal_reasons: no_trade_uncertain=226617, weak_gru_direction=108996, no_trade_extreme_vol=61152, no_trade_chop=58911, gru_expected_r_below_threshold=28148, wait_pullback=2418

Calibration Summary:
  all          [OK] Too few populated bins for calibration check
  ml_trader    [OK] Too few populated bins for calibration check
2026-05-10 23:30:33,906 INFO Round 0 backtest — 343 trades | avg WR=25.1% | avg PF=0.93 | avg Sharpe=-0.47
2026-05-10 23:30:33,906 INFO   ml_trader: 343 trades | WR=25.1% | fixed PF=0.93 | Return=-17.3% | ExpR=-0.051 | DD=61.0% | Sharpe=-0.47
2026-05-10 23:30:33,906 INFO   ml_trader gate_diagnostics: bars=1049680 no_signal=486658 quality_block=0 session_skip=562433 density=246 pm_reject=0
2026-05-10 23:30:33,906 INFO   ml_trader no_signal_reasons: {'no_trade_uncertain': 226617, 'weak_gru_direction': 108996, 'gru_expected_r_below_threshold': 28148, 'no_trade_extreme_vol': 61152, 'no_trade_chop': 58911, 'wait_pullback': 2418, 'tradeability_direction_conflict': 391, 'expected_r_below_threshold': 25}
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_0.json
INFO  Total trades: 343
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (343 rows)
2026-05-10 23:30:34,387 INFO Round 0: wrote 343 journal entries (total in file: 343)
  DONE  Train-window backtest for Quality/RL labels
  Saved Train Quality/RL source result → train_quality_rl_source_summary.json
  Train-label journal entries: 343

=== Train Quality + RL on train-only journal ===
  START Train-only Quality+RL retrain
2026-05-10 23:30:34,601 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-05-10 23:30:34,614 INFO Journal entries: 343 total, 343 allowed for training (['live', 'paper', 'production', 'train'])
2026-05-10 23:30:34,614 INFO --- Training quality ---
2026-05-10 23:30:34,614 INFO Running retrain --model quality with JOURNAL_ALLOWED_SPLITS=train,live,paper,production
2026-05-10 23:30:34,792 INFO retrain environment: KAGGLE
2026-05-10 23:30:36,385 INFO Device: CUDA (2 GPU(s))
2026-05-10 23:30:36,396 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-10 23:30:36,396 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-10 23:30:36,396 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-10 23:30:36,398 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-10 23:30:36,398 INFO Retrain data split: train
2026-05-10 23:30:36,398 INFO Retrain rolling fold selector: latest
2026-05-10 23:30:36,399 INFO === QualityScorer retrain ===
2026-05-10 23:30:36,552 INFO NumExpr defaulting to 4 threads.
2026-05-10 23:30:36,742 INFO QualityScorer: CUDA available — using GPU
2026-05-10 23:30:36,792 INFO QualityScorer: group EV smoothing applied to 323/343 rows (blend=30% group, min_group=10)
2026-05-10 23:30:36,794 INFO Quality phase label creation: 0.1s (343 trades)
2026-05-10 23:30:36,842 INFO QualityScorer: group EV smoothing applied to 323/343 rows (blend=30% group, min_group=10)
2026-05-10 23:30:36,844 INFO QualityScorer: 343 samples, EV stats={'mean': -0.35544607043266296, 'std': 0.8774281740188599, 'n_pos': 86, 'n_neg': 257}, device=cuda
2026-05-10 23:30:37,047 INFO QualityScorer: DataParallel across 2 GPUs
2026-05-10 23:30:37,047 INFO QualityScorer: cold start
2026-05-10 23:30:37,047 INFO QualityScorer: pos_weight=3.42 (n_pos=62 n_neg=212)
2026-05-10 23:30:39,322 INFO Quality epoch   1/100 — va_huber=1.0879
2026-05-10 23:30:39,361 INFO Quality epoch   2/100 — va_huber=1.0874
2026-05-10 23:30:39,381 INFO Quality epoch   3/100 — va_huber=1.0850
2026-05-10 23:30:39,401 INFO Quality epoch   4/100 — va_huber=1.0836
2026-05-10 23:30:39,420 INFO Quality epoch   5/100 — va_huber=1.0814
2026-05-10 23:30:39,541 INFO Quality epoch  11/100 — va_huber=1.0626
2026-05-10 23:30:39,734 INFO Quality epoch  21/100 — va_huber=1.0231
2026-05-10 23:30:39,928 INFO Quality epoch  31/100 — va_huber=0.9618
2026-05-10 23:30:40,120 INFO Quality epoch  41/100 — va_huber=0.9281
2026-05-10 23:30:40,477 INFO Quality epoch  51/100 — va_huber=0.9362
2026-05-10 23:30:40,534 INFO Quality early stop at epoch 54
2026-05-10 23:30:40,542 INFO QualityScorer EV model: MAE=0.826 dir_acc=0.565 n_val=69
2026-05-10 23:30:40,546 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-05-10 23:30:40,592 INFO Quality phase train: 3.8s | total: 4.2s
2026-05-10 23:30:40,599 INFO Retrain complete. Total wall-clock: 4.2s
2026-05-10 23:30:41,612 INFO Model quality: SUCCESS
2026-05-10 23:30:41,612 INFO --- Training rl ---
2026-05-10 23:30:41,612 INFO Running retrain --model rl with JOURNAL_ALLOWED_SPLITS=train,live,paper,production
2026-05-10 23:30:41,794 INFO retrain environment: KAGGLE
2026-05-10 23:30:43,431 INFO Device: CUDA (2 GPU(s))
2026-05-10 23:30:43,442 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-10 23:30:43,443 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-10 23:30:43,443 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-10 23:30:43,443 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-10 23:30:43,443 INFO Retrain data split: train
2026-05-10 23:30:43,443 INFO Retrain rolling fold selector: latest
2026-05-10 23:30:43,444 INFO === RLAgent (PPO) retrain ===
2026-05-10 23:30:43,594 INFO NumExpr defaulting to 4 threads.
2026-05-10 23:30:43,784 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260510_233043
2026-05-10 23:30:43,799 INFO RL phase episode loading: 0.0s (343 episodes)
2026-05-10 23:30:46.420752: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1778455846.572776  110542 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1778455846.618039  110542 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1778455846.979283  110542 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1778455846.979319  110542 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1778455846.979322  110542 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1778455846.979325  110542 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-05-10 23:31:02,505 INFO RLAgent: cold start — building new PPO policy
2026-05-10 23:31:09,060 INFO RLAgent: retrain complete, 343 episodes
2026-05-10 23:31:09,060 INFO RL phase PPO train: 25.3s | total: 25.6s
2026-05-10 23:31:09,069 INFO Retrain complete. Total wall-clock: 25.6s
2026-05-10 23:31:10,654 INFO Model rl: SUCCESS
2026-05-10 23:31:10,655 INFO Step 7b complete — summary: /kaggle/working/Multi-Bot/trading-system/ml_training/metrics/training_7b_summary.json
  DONE  Train-only Quality+RL retrain
  Archived journal → trade_journal_train_only.jsonl
  Archived journal CSV → trade_journal_train_only.csv

=== Round 1: Backtest on train-tail window (latest 2yr inside training data) ===
  Cleared journal for fresh Round 1 run
  START Round 1 - Backtest (train-tail)
2026-05-10 23:31:11,179 INFO === STEP 6: BACKTEST (round1) ===
2026-05-10 23:31:11,180 INFO BT_WINDOW=round1 — train-tail backtest: 2021-08-05 → 2023-08-04 (seen training data; test set protected)
2026-05-10 23:31:11,180 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-05-10 23:31:11,180 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
2026-05-10 23:31:11,181 INFO Round 1 — QualityScorer hard gate disabled; set BACKTEST_ENABLE_QUALITY_GATE=1 to enable it
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
2026-05-10 23:32:24,418 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
2026-05-10 23:32:24,777 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
2026-05-10 23:32:24,847 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:757: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-10 23:32:24,910 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:765: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-10 23:32:24,974 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:771: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:757: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-10 23:32:25,057 WARNING _build_sequence_df: HTF frame 5M filled 461 warmup/alignment gaps with 0.000
2026-05-10 23:32:25,059 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:765: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:771: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-10 23:32:25,133 WARNING _build_sequence_df: HTF frame 5M filled 461 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:757: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:765: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:757: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:771: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:765: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:771: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
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
2026-05-10 23:32:32,099 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
2026-05-10 23:32:32,329 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:757: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:765: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-10 23:32:32,412 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:771: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-10 23:32:32,443 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:757: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:765: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:771: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-10 23:33:08,257 INFO Round 1 backtest — 120 trades | avg WR=30.8% | avg PF=1.22 | avg Sharpe=1.35
2026-05-10 23:33:08,257 INFO   ml_trader: 120 trades | WR=30.8% | fixed PF=1.22 | Return=18.6% | ExpR=0.155 | DD=21.5% | Sharpe=1.35
2026-05-10 23:33:08,257 INFO   ml_trader gate_diagnostics: bars=263960 no_signal=117887 quality_block=0 session_skip=145857 density=96 pm_reject=0
2026-05-10 23:33:08,257 INFO   ml_trader no_signal_reasons: {'no_trade_uncertain': 53785, 'no_trade_chop': 14418, 'weak_gru_direction': 26196, 'gru_expected_r_below_threshold': 6157, 'no_trade_extreme_vol': 16195, 'wait_pullback': 963, 'tradeability_direction_conflict': 160, 'expected_r_below_threshold': 13}
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 120
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (120 rows)
2026-05-10 23:33:08,551 INFO Round 1: wrote 120 journal entries (total in file: 120)
  DONE  Round 1 - Backtest (train-tail)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 120 entries

  SKIP  Round 1 Quality+RL retrain — train-tail journal kept evaluation-only

=== Round 2: BLIND backtest on test window (unseen 2yr) ===
  START Round 2 - Blind backtest (test)
2026-05-10 23:33:09,143 INFO === STEP 6: BACKTEST (round2) ===
2026-05-10 23:33:09,144 INFO BT_WINDOW=round2 — BLIND backtest: 2023-08-07 → 2025-08-05 (test set)
2026-05-10 23:33:09,144 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-05-10 23:33:09,144 INFO Round 2 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
2026-05-10 23:33:09,145 INFO Round 2 — QualityScorer hard gate disabled; set BACKTEST_ENABLE_QUALITY_GATE=1 to enable it
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
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
2026-05-10 23:34:25,319 WARNING _build_sequence_df: HTF frame 5M filled 341 warmup/alignment gaps with 0.000
2026-05-10 23:34:25,382 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
2026-05-10 23:34:25,713 WARNING _build_sequence_df: HTF frame 5M filled 341 warmup/alignment gaps with 0.000
2026-05-10 23:34:25,835 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:757: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:757: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:765: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-10 23:34:26,044 WARNING _build_sequence_df: HTF frame 5M filled 317 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:771: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:765: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-10 23:34:26,130 WARNING _build_sequence_df: HTF frame 5M filled 317 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:771: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-10 23:34:26,193 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:757: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:765: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-10 23:34:26,270 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:771: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:757: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:765: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:771: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-10 23:34:33,563 WARNING _build_sequence_df: HTF frame 5M filled 317 warmup/alignment gaps with 0.000
2026-05-10 23:34:33,688 WARNING _build_sequence_df: HTF frame 5M filled 317 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:757: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:765: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-10 23:34:33,784 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:771: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-10 23:34:33,812 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:757: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:765: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:771: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-10 23:35:13,017 INFO Round 2 backtest — 128 trades | avg WR=26.6% | avg PF=0.89 | avg Sharpe=-0.82
2026-05-10 23:35:13,017 INFO   ml_trader: 128 trades | WR=26.6% | fixed PF=0.89 | Return=-10.5% | ExpR=-0.082 | DD=23.7% | Sharpe=-0.82
2026-05-10 23:35:13,017 INFO   ml_trader gate_diagnostics: bars=280782 no_signal=131441 quality_block=0 session_skip=149122 density=91 pm_reject=0
2026-05-10 23:35:13,017 INFO   ml_trader no_signal_reasons: {'no_trade_chop': 15806, 'no_trade_uncertain': 61914, 'weak_gru_direction': 29220, 'gru_expected_r_below_threshold': 7810, 'no_trade_extreme_vol': 15634, 'tradeability_direction_conflict': 134, 'wait_pullback': 912, 'expected_r_below_threshold': 11}
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 128
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (128 rows)
2026-05-10 23:35:13,318 INFO Round 2: wrote 128 journal entries (total in file: 248)
  DONE  Round 2 - Blind backtest (test)
  Saved Round 2 result → round2_summary.json
  Journal after Round 2: 248 entries

  SKIP  Round 2 Quality+RL retrain — blind test journal kept untouched

=== Round 3: Incremental retrain ===
  START Retrain gru [train-split retrain]
2026-05-10 23:35:13,628 INFO retrain environment: KAGGLE
2026-05-10 23:35:15,206 INFO Device: CUDA (2 GPU(s))
2026-05-10 23:35:15,217 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-10 23:35:15,217 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-10 23:35:15,217 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-10 23:35:15,217 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-10 23:35:15,217 INFO Retrain data split: train
2026-05-10 23:35:15,217 INFO Retrain rolling fold selector: latest
2026-05-10 23:35:15,219 INFO === GRU-LSTM retrain (timeframes: ['15M']) ===
2026-05-10 23:35:15,360 INFO NumExpr defaulting to 4 threads.
2026-05-10 23:35:15,552 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-10 23:35:15,552 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-10 23:35:15,552 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-10 23:35:15,793 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-05-10 23:35:15,793 INFO GRU multi-symbol training (Kaggle mode): 6 symbols × ['15M']
2026-05-10 23:35:15,795 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260510_233515
2026-05-10 23:35:15,799 INFO GRU feature contract unchanged (input_size=71) — incremental retrain
2026-05-10 23:35:15,799 INFO GRU warm start enabled from existing weights: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:35:16,053 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:35:16,080 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:35:16,095 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:35:16,104 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:35:16,173 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-10 23:35:16,179 INFO Loaded XAUUSD/15M split=train fold=latest: 176438 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:35:16,480 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:35:16,498 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:35:16,512 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:35:16,518 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:35:16,556 INFO Loaded EURUSD/15M split=train fold=latest: 174951 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:35:16,843 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:35:16,862 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:35:16,887 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:35:16,894 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:35:16,939 INFO Loaded USDJPY/15M split=train fold=latest: 174944 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:35:17,229 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:35:17,248 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:35:17,262 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:35:17,269 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:35:17,306 INFO Loaded EURJPY/15M split=train fold=latest: 174948 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:35:17,580 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:35:17,599 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:35:17,612 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:35:17,619 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:35:17,658 INFO Loaded GBPJPY/15M split=train fold=latest: 174902 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:35:17,932 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:35:17,950 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:35:17,963 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:35:17,970 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:35:18,005 INFO Loaded GBPUSD/15M split=train fold=latest: 174899 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:35:18,181 INFO train_multi: 6 segments, ~936212 total bars
2026-05-10 23:35:18,182 INFO train_multi: training ALL 6 segments across TFs ['15M'] in one combined pass
2026-05-10 23:35:18,182 INFO train_multi: building combined dataset for TF=ALL (6 segments)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:35:27,152 INFO train_multi TF=ALL: 936032 sequences across 6 segments
2026-05-10 23:35:27,152 INFO train_multi TF=ALL: estimated peak RAM = 10224 MB (train=479995 val=120002 n_feat=71 seq_len=30)
2026-05-10 23:35:28,394 INFO train_multi TF=ALL: train=479995 val=120002 (5122 MB tensors)
2026-05-10 23:35:32,464 INFO train_multi TF=ALL: warm-start detected — using CosineAnnealingLR (lr=3e-05, patience=12)
2026-05-10 23:35:49,537 INFO train_multi TF=ALL epoch 1/100 train=1.7560 val=1.8249 r_mae=0.749 pos_r_acc=0.707 side_acc=0.719 r_n=240004
2026-05-10 23:35:49,543 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-10 23:35:49,543 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-10 23:35:49,543 INFO train_multi TF=ALL: new best val=1.8249 — saved
2026-05-10 23:36:04,807 INFO train_multi TF=ALL epoch 2/100 train=1.7548 val=1.8265 r_mae=0.749 pos_r_acc=0.707 side_acc=0.719 r_n=240004
2026-05-10 23:36:19,893 INFO train_multi TF=ALL epoch 3/100 train=1.7549 val=1.8266 r_mae=0.748 pos_r_acc=0.707 side_acc=0.719 r_n=240004
2026-05-10 23:36:35,003 INFO train_multi TF=ALL epoch 4/100 train=1.7524 val=1.8271 r_mae=0.747 pos_r_acc=0.708 side_acc=0.719 r_n=240004
2026-05-10 23:36:50,242 INFO train_multi TF=ALL epoch 5/100 train=1.7524 val=1.8281 r_mae=0.750 pos_r_acc=0.706 side_acc=0.718 r_n=240004
2026-05-10 23:37:05,491 INFO train_multi TF=ALL epoch 6/100 train=1.7516 val=1.8263 r_mae=0.749 pos_r_acc=0.707 side_acc=0.718 r_n=240004
2026-05-10 23:37:20,882 INFO train_multi TF=ALL epoch 7/100 train=1.7507 val=1.8275 r_mae=0.750 pos_r_acc=0.707 side_acc=0.719 r_n=240004
2026-05-10 23:37:36,075 INFO train_multi TF=ALL epoch 8/100 train=1.7502 val=1.8320 r_mae=0.749 pos_r_acc=0.706 side_acc=0.718 r_n=240004
2026-05-10 23:37:51,234 INFO train_multi TF=ALL epoch 9/100 train=1.7494 val=1.8261 r_mae=0.752 pos_r_acc=0.706 side_acc=0.719 r_n=240004
2026-05-10 23:38:06,461 INFO train_multi TF=ALL epoch 10/100 train=1.7484 val=1.8286 r_mae=0.749 pos_r_acc=0.706 side_acc=0.718 r_n=240004
2026-05-10 23:38:21,574 INFO train_multi TF=ALL epoch 11/100 train=1.7471 val=1.8293 r_mae=0.748 pos_r_acc=0.707 side_acc=0.719 r_n=240004
2026-05-10 23:38:36,796 INFO train_multi TF=ALL epoch 12/100 train=1.7469 val=1.8290 r_mae=0.748 pos_r_acc=0.707 side_acc=0.719 r_n=240004
2026-05-10 23:38:51,949 INFO train_multi TF=ALL epoch 13/100 train=1.7460 val=1.8300 r_mae=0.749 pos_r_acc=0.707 side_acc=0.718 r_n=240004
2026-05-10 23:38:51,949 INFO train_multi TF=ALL early stop at epoch 13
2026-05-10 23:38:52,083 INFO GRU R threshold XAUUSD/buy: q25=-1.000 q50=-0.127 (n=156482)
2026-05-10 23:38:52,088 INFO GRU R threshold XAUUSD/sell: q25=-1.000 q50=-0.140 (n=156482)
2026-05-10 23:38:52,093 INFO GRU R threshold EURUSD/buy: q25=-1.000 q50=-0.134 (n=154969)
2026-05-10 23:38:52,098 INFO GRU R threshold EURUSD/sell: q25=-1.000 q50=-0.142 (n=154969)
2026-05-10 23:38:52,103 INFO GRU R threshold USDJPY/buy: q25=-1.000 q50=-0.115 (n=155969)
2026-05-10 23:38:52,109 INFO GRU R threshold USDJPY/sell: q25=-1.000 q50=-0.173 (n=155969)
2026-05-10 23:38:52,115 INFO GRU R threshold EURJPY/buy: q25=-1.000 q50=-0.120 (n=156487)
2026-05-10 23:38:52,120 INFO GRU R threshold EURJPY/sell: q25=-1.000 q50=-0.174 (n=156487)
2026-05-10 23:38:52,124 INFO GRU R threshold GBPJPY/buy: q25=-1.000 q50=-0.137 (n=156862)
2026-05-10 23:38:52,130 INFO GRU R threshold GBPJPY/sell: q25=-1.000 q50=-0.153 (n=156862)
2026-05-10 23:38:52,134 INFO GRU R threshold GBPUSD/buy: q25=-1.000 q50=-0.139 (n=155443)
2026-05-10 23:38:52,139 INFO GRU R threshold GBPUSD/sell: q25=-1.000 q50=-0.133 (n=155443)
2026-05-10 23:38:52,140 INFO GRU per-symbol R thresholds saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/symbol_r_thresholds.json (6 symbols)
2026-05-10 23:38:52,164 INFO Retrain complete. Total wall-clock: 216.9s
  DONE  Retrain gru [train-split retrain]
  START Retrain regime [train-split retrain]
2026-05-10 23:38:54,178 INFO retrain environment: KAGGLE
2026-05-10 23:38:55,766 INFO Device: CUDA (2 GPU(s))
2026-05-10 23:38:55,775 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-10 23:38:55,775 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-10 23:38:55,775 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-10 23:38:55,775 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-10 23:38:55,775 INFO Retrain data split: train
2026-05-10 23:38:55,775 INFO Retrain rolling fold selector: latest
2026-05-10 23:38:55,776 INFO === RegimeClassifier retrain (hierarchical: HTF 2-score bias + LTF 5-score behaviour) ===
2026-05-10 23:38:55,920 INFO NumExpr defaulting to 4 threads.
2026-05-10 23:38:56,110 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-10 23:38:56,110 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-10 23:38:56,110 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-10 23:38:56,111 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-10 23:38:56,163 INFO Regime rolling folds selected: [None]
2026-05-10 23:38:56,163 INFO === Regime rolling fold 1/1: train_all ===
2026-05-10 23:38:56,163 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-10 23:38:56,203 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-10 23:38:56,204 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:38:56,219 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:38:56,233 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:38:56,248 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:38:56,262 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:38:56,276 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:38:56,512 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:38:56,581 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:38:56,604 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:38:56,604 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:38:56,614 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:38:56,615 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:38:57,007 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 492, 'BIAS_DOWN': 270, 'BIAS_NEUTRAL': 11340}  ambiguous=6929 (total=12102) horizon=12
2026-05-10 23:38:57,013 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected XAUUSD — 12052 samples (group=gold) score_means={'bias_up_score': 0.0408, 'bias_down_score': 0.0224} labels={'BIAS_UP': 492, 'BIAS_DOWN': 270, 'BIAS_NEUTRAL': 11290} clean={'BIAS_UP': 492, 'BIAS_DOWN': 270, 'BIAS_NEUTRAL': 4396}
2026-05-10 23:38:57,171 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:57,213 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:57,231 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:57,231 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:57,239 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:57,240 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:38:57,588 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 294, 'BIAS_DOWN': 349, 'BIAS_NEUTRAL': 10761}  ambiguous=6552 (total=11404) horizon=12
2026-05-10 23:38:57,593 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURUSD — 11354 samples (group=dollar) score_means={'bias_up_score': 0.0259, 'bias_down_score': 0.0307} labels={'BIAS_UP': 294, 'BIAS_DOWN': 349, 'BIAS_NEUTRAL': 10711} clean={'BIAS_UP': 294, 'BIAS_DOWN': 349, 'BIAS_NEUTRAL': 4188}
2026-05-10 23:38:57,759 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:57,793 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:57,811 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:57,811 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:57,818 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:57,819 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:38:58,168 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 465, 'BIAS_DOWN': 243, 'BIAS_NEUTRAL': 10695}  ambiguous=6644 (total=11403) horizon=12
2026-05-10 23:38:58,174 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected USDJPY — 11353 samples (group=dollar) score_means={'bias_up_score': 0.041, 'bias_down_score': 0.0214} labels={'BIAS_UP': 465, 'BIAS_DOWN': 243, 'BIAS_NEUTRAL': 10645} clean={'BIAS_UP': 465, 'BIAS_DOWN': 243, 'BIAS_NEUTRAL': 4028}
2026-05-10 23:38:58,337 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:58,371 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:58,389 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:58,390 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:58,396 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:58,397 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:38:58,738 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 345, 'BIAS_DOWN': 200, 'BIAS_NEUTRAL': 10862}  ambiguous=6647 (total=11407) horizon=12
2026-05-10 23:38:58,743 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURJPY — 11357 samples (group=cross) score_means={'bias_up_score': 0.0304, 'bias_down_score': 0.0176} labels={'BIAS_UP': 345, 'BIAS_DOWN': 200, 'BIAS_NEUTRAL': 10812} clean={'BIAS_UP': 345, 'BIAS_DOWN': 200, 'BIAS_NEUTRAL': 4191}
2026-05-10 23:38:58,888 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:58,924 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:58,943 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:58,944 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:58,950 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:58,951 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:38:59,299 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 458, 'BIAS_DOWN': 281, 'BIAS_NEUTRAL': 10669}  ambiguous=6611 (total=11408) horizon=12
2026-05-10 23:38:59,304 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPJPY — 11358 samples (group=cross) score_means={'bias_up_score': 0.0403, 'bias_down_score': 0.0247} labels={'BIAS_UP': 458, 'BIAS_DOWN': 281, 'BIAS_NEUTRAL': 10619} clean={'BIAS_UP': 458, 'BIAS_DOWN': 281, 'BIAS_NEUTRAL': 4042}
2026-05-10 23:38:59,464 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:59,497 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:59,515 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:59,515 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:59,522 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:38:59,523 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:38:59,871 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 269, 'BIAS_DOWN': 344, 'BIAS_NEUTRAL': 10789}  ambiguous=6630 (total=11402) horizon=12
2026-05-10 23:38:59,876 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPUSD — 11352 samples (group=dollar) score_means={'bias_up_score': 0.0237, 'bias_down_score': 0.0303} labels={'BIAS_UP': 269, 'BIAS_DOWN': 344, 'BIAS_NEUTRAL': 10739} clean={'BIAS_UP': 269, 'BIAS_DOWN': 344, 'BIAS_NEUTRAL': 4148}
2026-05-10 23:38:59,938 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 803, 'BIAS_DOWN': 481, 'BIAS_NEUTRAL': 21431}, 'dollar': {'BIAS_UP': 1028, 'BIAS_DOWN': 936, 'BIAS_NEUTRAL': 32095}, 'gold': {'BIAS_UP': 492, 'BIAS_DOWN': 270, 'BIAS_NEUTRAL': 11290}}
2026-05-10 23:38:59,938 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0354, 'bias_down_score': 0.0212}, 'dollar': {'bias_up_score': 0.0302, 'bias_down_score': 0.0275}, 'gold': {'bias_up_score': 0.0408, 'bias_down_score': 0.0224}}
2026-05-10 23:38:59,938 INFO Regime[4H mode=htf_bias] label distribution by year: {2016: {'BIAS_UP': 286, 'BIAS_DOWN': 340, 'BIAS_NEUTRAL': 8196}, 2017: {'BIAS_UP': 461, 'BIAS_DOWN': 205, 'BIAS_NEUTRAL': 8447}, 2018: {'BIAS_UP': 213, 'BIAS_DOWN': 288, 'BIAS_NEUTRAL': 8629}, 2019: {'BIAS_UP': 210, 'BIAS_DOWN': 192, 'BIAS_NEUTRAL': 8700}, 2020: {'BIAS_UP': 298, 'BIAS_DOWN': 180, 'BIAS_NEUTRAL': 8633}, 2021: {'BIAS_UP': 294, 'BIAS_DOWN': 186, 'BIAS_NEUTRAL': 8611}, 2022: {'BIAS_UP': 370, 'BIAS_DOWN': 225, 'BIAS_NEUTRAL': 8526}, 2023: {'BIAS_UP': 191, 'BIAS_DOWN': 71, 'BIAS_NEUTRAL': 5074}}
2026-05-10 23:38:59,938 INFO Regime[4H mode=htf_bias] score means by year: {2016: {'bias_up_score': 0.0324, 'bias_down_score': 0.0385}, 2017: {'bias_up_score': 0.0506, 'bias_down_score': 0.0225}, 2018: {'bias_up_score': 0.0233, 'bias_down_score': 0.0315}, 2019: {'bias_up_score': 0.0231, 'bias_down_score': 0.0211}, 2020: {'bias_up_score': 0.0327, 'bias_down_score': 0.0198}, 2021: {'bias_up_score': 0.0323, 'bias_down_score': 0.0205}, 2022: {'bias_up_score': 0.0406, 'bias_down_score': 0.0247}, 2023: {'bias_up_score': 0.0358, 'bias_down_score': 0.0133}}
2026-05-10 23:38:59,981 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
2026-05-10 23:38:59,982 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
2026-05-10 23:38:59,982 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
2026-05-10 23:38:59,983 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-10 23:38:59,984 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-10 23:38:59,985 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
2026-05-10 23:39:00,000 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:39:00,004 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:39:00,005 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:39:00,005 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:39:00,006 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:39:00,007 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:39:00,235 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 39, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1512}  ambiguous=936 (total=1581) horizon=12
2026-05-10 23:39:00,238 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected XAUUSD — 1531 samples (group=gold) score_means={'bias_up_score': 0.0255, 'bias_down_score': 0.0196} labels={'BIAS_UP': 39, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1462} clean={'BIAS_UP': 39, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 555}
2026-05-10 23:39:00,307 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:00,310 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:00,311 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:00,311 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:00,311 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:00,312 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:39:00,524 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 35, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1403}  ambiguous=861 (total=1491) horizon=12
2026-05-10 23:39:00,526 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURUSD — 1441 samples (group=dollar) score_means={'bias_up_score': 0.0243, 'bias_down_score': 0.0368} labels={'BIAS_UP': 35, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1353} clean={'BIAS_UP': 35, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 532}
2026-05-10 23:39:00,595 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:00,597 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:00,598 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:00,599 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:00,599 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:00,600 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:39:00,809 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 102, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 1362}  ambiguous=886 (total=1489) horizon=12
2026-05-10 23:39:00,812 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected USDJPY — 1439 samples (group=dollar) score_means={'bias_up_score': 0.0709, 'bias_down_score': 0.0174} labels={'BIAS_UP': 102, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 1312} clean={'BIAS_UP': 102, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 454}
2026-05-10 23:39:00,879 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:00,882 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:00,882 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:00,883 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:00,883 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:00,884 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:39:01,098 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 46, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1447}  ambiguous=915 (total=1494) horizon=12
2026-05-10 23:39:01,100 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0319, 'bias_down_score': 0.0007} labels={'BIAS_UP': 46, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1397} clean={'BIAS_UP': 46, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 509}
2026-05-10 23:39:01,168 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:01,170 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:01,171 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:01,172 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:01,172 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:01,173 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:39:01,375 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 29, 'BIAS_DOWN': 21, 'BIAS_NEUTRAL': 1444}  ambiguous=861 (total=1494) horizon=12
2026-05-10 23:39:01,378 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0201, 'bias_down_score': 0.0145} labels={'BIAS_UP': 29, 'BIAS_DOWN': 21, 'BIAS_NEUTRAL': 1394} clean={'BIAS_UP': 29, 'BIAS_DOWN': 21, 'BIAS_NEUTRAL': 561}
2026-05-10 23:39:01,444 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:01,446 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:01,447 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:01,447 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:01,447 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:01,448 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:39:01,652 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 26, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 1440}  ambiguous=885 (total=1488) horizon=12
2026-05-10 23:39:01,655 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPUSD — 1438 samples (group=dollar) score_means={'bias_up_score': 0.0181, 'bias_down_score': 0.0153} labels={'BIAS_UP': 26, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 1390} clean={'BIAS_UP': 26, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 543}
2026-05-10 23:39:01,715 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 75, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 2791}, 'dollar': {'BIAS_UP': 163, 'BIAS_DOWN': 100, 'BIAS_NEUTRAL': 4055}, 'gold': {'BIAS_UP': 39, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1462}}
2026-05-10 23:39:01,715 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.026, 'bias_down_score': 0.0076}, 'dollar': {'bias_up_score': 0.0377, 'bias_down_score': 0.0232}, 'gold': {'bias_up_score': 0.0255, 'bias_down_score': 0.0196}}
2026-05-10 23:39:01,715 INFO Regime[4H mode=htf_bias] label distribution by year: {2022: {'BIAS_UP': 91, 'BIAS_DOWN': 81, 'BIAS_NEUTRAL': 3229}, 2023: {'BIAS_UP': 186, 'BIAS_DOWN': 71, 'BIAS_NEUTRAL': 5079}}
2026-05-10 23:39:01,715 INFO Regime[4H mode=htf_bias] score means by year: {2022: {'bias_up_score': 0.0268, 'bias_down_score': 0.0238}, 2023: {'bias_up_score': 0.0349, 'bias_down_score': 0.0133}}
2026-05-10 23:39:01,757 INFO Regime phase HTF dataset build fold=train_all: 5.6s (train=68826 val=8737)
2026-05-10 23:39:01,757 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260510_233901
2026-05-10 23:39:01,958 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=25, n_classes=2)
2026-05-10 23:39:01,958 INFO Regime 4H/htf_bias warm start enabled from existing weights
2026-05-10 23:39:01,964 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=68826 val=8737 train_labels={'BIAS_UP': 2323, 'BIAS_DOWN': 1687, 'BIAS_NEUTRAL': 64816} val_labels={'BIAS_UP': 277, 'BIAS_DOWN': 152, 'BIAS_NEUTRAL': 8308}
2026-05-10 23:39:01,964 INFO RegimeClassifier[mode=htf_bias]: warm start HTF score head
2026-05-10 23:39:01,964 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-10 23:39:01,964 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 28.628, 'bias_down_score': 30.0}
2026-05-10 23:39:01,968 INFO RegimeClassifier[mode=htf_bias]: HTF balanced sampler — dir=4010 neutral=64816 dir_weight=15 => dir_frac_per_epoch≈48.1%
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../models/regime_classifier.py:2288: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  scheduler.step()
2026-05-10 23:39:05,671 INFO Regime HTF score epoch  1/50 — tr=11.6860 va=0.8031 acc=0.940 bal=0.432 threshold=0.70 margin=0.00 recall={'BIAS_UP': 0.271, 'BIAS_DOWN': 0.046, 'BIAS_NEUTRAL': 0.979} precision={'BIAS_UP': 0.323, 'BIAS_DOWN': 0.28, 'BIAS_NEUTRAL': 0.959}
2026-05-10 23:39:07,262 INFO Regime HTF score epoch  2/50 — tr=11.5443 va=0.8042 bal=0.428
2026-05-10 23:39:08,864 INFO Regime HTF score epoch  3/50 — tr=11.4202 va=0.7948 bal=0.461
2026-05-10 23:39:10,476 INFO Regime HTF score epoch  4/50 — tr=11.3652 va=0.7740 bal=0.463
2026-05-10 23:39:12,100 INFO Regime HTF score epoch  5/50 — tr=10.8790 va=0.7512 acc=0.941 bal=0.424 threshold=0.75 margin=0.00 recall={'BIAS_UP': 0.245, 'BIAS_DOWN': 0.046, 'BIAS_NEUTRAL': 0.981} precision={'BIAS_UP': 0.319, 'BIAS_DOWN': 0.292, 'BIAS_NEUTRAL': 0.958}
2026-05-10 23:39:13,724 INFO Regime HTF score epoch  6/50 — tr=10.4142 va=0.7184 bal=0.460
2026-05-10 23:39:15,321 INFO Regime HTF score epoch  7/50 — tr=9.8911 va=0.6832 bal=0.449
2026-05-10 23:39:16,925 INFO Regime HTF score epoch  8/50 — tr=9.1871 va=0.6476 bal=0.429
2026-05-10 23:39:18,547 INFO Regime HTF score epoch  9/50 — tr=8.6310 va=0.6118 bal=0.404
2026-05-10 23:39:20,163 INFO Regime HTF score epoch 10/50 — tr=7.9194 va=0.5786 acc=0.942 bal=0.423 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.22, 'BIAS_DOWN': 0.066, 'BIAS_NEUTRAL': 0.982} precision={'BIAS_UP': 0.324, 'BIAS_DOWN': 0.278, 'BIAS_NEUTRAL': 0.958}
2026-05-10 23:39:21,775 INFO Regime HTF score epoch 11/50 — tr=7.2922 va=0.5472 bal=0.405
2026-05-10 23:39:23,413 INFO Regime HTF score epoch 12/50 — tr=6.6781 va=0.5166 bal=0.423
2026-05-10 23:39:25,044 INFO Regime HTF score epoch 13/50 — tr=6.0183 va=0.4849 bal=0.413
2026-05-10 23:39:26,638 INFO Regime HTF score epoch 14/50 — tr=5.4810 va=0.4620 bal=0.393
2026-05-10 23:39:28,254 INFO Regime HTF score epoch 15/50 — tr=5.0894 va=0.4412 acc=0.941 bal=0.403 threshold=0.92 margin=0.00 recall={'BIAS_UP': 0.141, 'BIAS_DOWN': 0.086, 'BIAS_NEUTRAL': 0.984} precision={'BIAS_UP': 0.283, 'BIAS_DOWN': 0.255, 'BIAS_NEUTRAL': 0.956}
2026-05-10 23:39:28,254 INFO Regime HTF score early stop at epoch 15
2026-05-10 23:39:29,712 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.750 margin=0.000 precision={'BIAS_UP': 0.324, 'BIAS_DOWN': 0.295, 'BIAS_NEUTRAL': 0.96} recall={'BIAS_UP': 0.285, 'BIAS_DOWN': 0.086, 'BIAS_NEUTRAL': 0.976} f1={'BIAS_UP': 0.303, 'BIAS_DOWN': 0.133, 'BIAS_NEUTRAL': 0.968} confusion=[[79, 0, 198], [0, 13, 139], [165, 31, 8112]] score_mae={'bias_up_score': 0.0998, 'bias_down_score': 0.0623} pred_share={'BIAS_UP': 0.0279, 'BIAS_DOWN': 0.005, 'BIAS_NEUTRAL': 0.967}
2026-05-10 23:39:29,713 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.324, 'BIAS_DOWN': 0.295, 'BIAS_NEUTRAL': 0.96} min_precision=0.500 recall={'BIAS_UP': 0.285, 'BIAS_DOWN': 0.086, 'BIAS_NEUTRAL': 0.976} min_recall=0.100 f1={'BIAS_UP': 0.303, 'BIAS_DOWN': 0.133, 'BIAS_NEUTRAL': 0.968} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_UP', 'BIAS_DOWN'] weak_recall=['BIAS_DOWN'] weak_f1=['BIAS_DOWN'] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-10 23:39:29,716 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-10 23:39:29,717 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-10 23:39:29,717 INFO Regime phase HTF train fold=train_all: 27.8s
2026-05-10 23:39:29,820 INFO Regime HTF complete fold=train_all: acc=0.939 bal=0.449 train=68826 val=8737 per_class={'BIAS_UP': 0.285, 'BIAS_DOWN': 0.086, 'BIAS_NEUTRAL': 0.976} precision={'BIAS_UP': 0.324, 'BIAS_DOWN': 0.295, 'BIAS_NEUTRAL': 0.96} threshold=0.750 margin=0.000
2026-05-10 23:39:29,821 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:39:29,970 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 269, 'BIAS_DOWN': 344, 'BIAS_NEUTRAL': 10789}  ambiguous=6630 (total=11402) horizon=12
2026-05-10 23:39:29,973 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on GBPUSD 4H:
{'BIAS_UP': 3.0568181818181817, 'BIAS_DOWN': 3.909090909090909, 'BIAS_NEUTRAL': 60.954802259887}
2026-05-10 23:39:29,976 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (all labels):
{'BIAS_UP': {'n': 269, 'mean': 0.0009352809682544664, 'mean_over_std': 0.46065867099997965}, 'BIAS_DOWN': {'n': 344, 'mean': -0.0013986222214448377, 'mean_over_std': -0.453155978736302}, 'BIAS_NEUTRAL': {'n': 10788, 'mean': 1.121563318643874e-05, 'mean_over_std': 0.0043231848821040425}}
2026-05-10 23:39:29,977 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 269, 'mean': 0.0009352809682544664, 'mean_over_std': 0.46065867099997965}, 'BIAS_DOWN': {'n': 344, 'mean': -0.0013986222214448377, 'mean_over_std': -0.453155978736302}, 'BIAS_NEUTRAL': {'n': 4159, 'mean': 1.3724894091827828e-05, 'mean_over_std': 0.006431864931044914}}
2026-05-10 23:39:29,982 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-10 23:39:29,984 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:39:29,986 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:39:29,987 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:39:29,989 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:39:29,991 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:39:29,993 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:39:30,012 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:39:30,020 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:39:30,023 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:39:30,024 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:39:30,024 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:39:30,030 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:39:30,889 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected XAUUSD — 44227 samples (group=gold) score_means={'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}
2026-05-10 23:39:30,995 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:30,997 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:30,998 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:30,998 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:30,998 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:31,001 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:39:31,796 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURUSD — 43689 samples (group=dollar) score_means={'trend_score': 0.49, 'range_score': 0.236, 'chop_score': 0.4622, 'volatility_percentile': 0.3812, 'consolidation_score': 0.1862}
2026-05-10 23:39:31,910 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:31,913 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:31,914 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:31,914 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:31,914 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:31,917 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:39:32,737 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected USDJPY — 43686 samples (group=dollar) score_means={'trend_score': 0.4949, 'range_score': 0.2312, 'chop_score': 0.4613, 'volatility_percentile': 0.38, 'consolidation_score': 0.1931}
2026-05-10 23:39:32,844 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:32,847 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:32,848 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:32,848 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:32,848 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:32,851 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:39:33,648 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURJPY — 43688 samples (group=cross) score_means={'trend_score': 0.4888, 'range_score': 0.2345, 'chop_score': 0.4662, 'volatility_percentile': 0.3831, 'consolidation_score': 0.1882}
2026-05-10 23:39:33,755 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:33,757 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:33,758 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:33,759 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:33,759 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:33,761 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:39:34,565 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPJPY — 43678 samples (group=cross) score_means={'trend_score': 0.4878, 'range_score': 0.2348, 'chop_score': 0.4673, 'volatility_percentile': 0.3839, 'consolidation_score': 0.1904}
2026-05-10 23:39:34,672 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:34,675 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:34,675 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:34,676 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:34,676 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:34,678 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:39:35,472 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPUSD — 43676 samples (group=dollar) score_means={'trend_score': 0.4923, 'range_score': 0.2335, 'chop_score': 0.4611, 'volatility_percentile': 0.3832, 'consolidation_score': 0.1856}
2026-05-10 23:39:35,586 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2346, 'chop_score': 0.4667, 'volatility_percentile': 0.3835, 'consolidation_score': 0.1893}, 'dollar': {'trend_score': 0.4924, 'range_score': 0.2335, 'chop_score': 0.4615, 'volatility_percentile': 0.3815, 'consolidation_score': 0.1883}, 'gold': {'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}}
2026-05-10 23:39:35,586 INFO Regime[1H mode=ltf_behaviour] score means by year: {2016: {'trend_score': 0.495, 'range_score': 0.233, 'chop_score': 0.4607, 'volatility_percentile': 0.3776, 'consolidation_score': 0.192}, 2017: {'trend_score': 0.5017, 'range_score': 0.2328, 'chop_score': 0.4574, 'volatility_percentile': 0.3689, 'consolidation_score': 0.1942}, 2018: {'trend_score': 0.4886, 'range_score': 0.237, 'chop_score': 0.4655, 'volatility_percentile': 0.3947, 'consolidation_score': 0.1785}, 2019: {'trend_score': 0.4817, 'range_score': 0.236, 'chop_score': 0.4691, 'volatility_percentile': 0.3725, 'consolidation_score': 0.198}, 2020: {'trend_score': 0.4874, 'range_score': 0.2311, 'chop_score': 0.4674, 'volatility_percentile': 0.3848, 'consolidation_score': 0.1919}, 2021: {'trend_score': 0.4872, 'range_score': 0.2372, 'chop_score': 0.4672, 'volatility_percentile': 0.377, 'consolidation_score': 0.192}, 2022: {'trend_score': 0.4893, 'range_score': 0.2357, 'chop_score': 0.4647, 'volatility_percentile': 0.3962, 'consolidation_score': 0.1784}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-10 23:39:35,676 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
2026-05-10 23:39:35,677 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-10 23:39:35,679 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
2026-05-10 23:39:35,680 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-10 23:39:35,681 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
2026-05-10 23:39:35,683 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
2026-05-10 23:39:35,692 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:39:35,695 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:39:35,696 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:39:35,697 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:39:35,697 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-10 23:39:35,699 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:39:35,939 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected XAUUSD — 5117 samples (group=gold) score_means={'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}
2026-05-10 23:39:36,046 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:36,048 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:36,049 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:36,049 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:36,050 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:36,051 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:39:36,277 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURUSD — 5053 samples (group=dollar) score_means={'trend_score': 0.4883, 'range_score': 0.2413, 'chop_score': 0.463, 'volatility_percentile': 0.3837, 'consolidation_score': 0.1793}
2026-05-10 23:39:36,383 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:36,386 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:36,386 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:36,387 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:36,387 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:36,389 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:39:36,620 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected USDJPY — 5046 samples (group=dollar) score_means={'trend_score': 0.5085, 'range_score': 0.2259, 'chop_score': 0.4506, 'volatility_percentile': 0.3809, 'consolidation_score': 0.1888}
2026-05-10 23:39:36,725 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:36,728 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:36,729 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:36,729 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:36,729 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:36,731 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:39:36,957 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURJPY — 5053 samples (group=cross) score_means={'trend_score': 0.5023, 'range_score': 0.2261, 'chop_score': 0.4524, 'volatility_percentile': 0.3837, 'consolidation_score': 0.18}
2026-05-10 23:39:37,062 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:37,064 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:37,065 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:37,065 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:37,066 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:37,067 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:39:37,305 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPJPY — 5043 samples (group=cross) score_means={'trend_score': 0.4743, 'range_score': 0.2372, 'chop_score': 0.4716, 'volatility_percentile': 0.3777, 'consolidation_score': 0.1917}
2026-05-10 23:39:37,412 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:37,414 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:37,415 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:37,415 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:37,415 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-10 23:39:37,417 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:39:37,647 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPUSD — 5040 samples (group=dollar) score_means={'trend_score': 0.4843, 'range_score': 0.2371, 'chop_score': 0.4614, 'volatility_percentile': 0.377, 'consolidation_score': 0.1835}
2026-05-10 23:39:37,749 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2317, 'chop_score': 0.462, 'volatility_percentile': 0.3807, 'consolidation_score': 0.1858}, 'dollar': {'trend_score': 0.4937, 'range_score': 0.2348, 'chop_score': 0.4583, 'volatility_percentile': 0.3806, 'consolidation_score': 0.1839}, 'gold': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}}
2026-05-10 23:39:37,749 INFO Regime[1H mode=ltf_behaviour] score means by year: {2022: {'trend_score': 0.4905, 'range_score': 0.2364, 'chop_score': 0.4625, 'volatility_percentile': 0.385, 'consolidation_score': 0.1797}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-10 23:39:37,828 INFO Regime phase LTF dataset build fold=train_all: 7.8s (train=262644 val=30352)
2026-05-10 23:39:37,828 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260510_233937
2026-05-10 23:39:37,833 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=31, n_classes=5)
2026-05-10 23:39:37,833 INFO Regime 1H/ltf_behaviour warm start enabled from existing weights
2026-05-10 23:39:37,857 INFO RegimeClassifier[mode=ltf_behaviour]: warm start score head
2026-05-10 23:39:37,857 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-10 23:39:38,394 INFO Regime score epoch  1/50 — tr=0.0048 va=0.0016 mae={'trend_score': 0.0275, 'range_score': 0.0413, 'chop_score': 0.0305, 'volatility_percentile': 0.0195, 'consolidation_score': 0.0297}
2026-05-10 23:39:38,895 INFO Regime score epoch  2/50 — tr=0.0048 va=0.0016
2026-05-10 23:39:39,420 INFO Regime score epoch  3/50 — tr=0.0048 va=0.0016
2026-05-10 23:39:39,922 INFO Regime score epoch  4/50 — tr=0.0048 va=0.0016
2026-05-10 23:39:40,423 INFO Regime score epoch  5/50 — tr=0.0047 va=0.0016 mae={'trend_score': 0.0271, 'range_score': 0.0403, 'chop_score': 0.0299, 'volatility_percentile': 0.0193, 'consolidation_score': 0.0295}
2026-05-10 23:39:40,913 INFO Regime score epoch  6/50 — tr=0.0047 va=0.0015
2026-05-10 23:39:41,410 INFO Regime score epoch  7/50 — tr=0.0047 va=0.0015
2026-05-10 23:39:41,920 INFO Regime score epoch  8/50 — tr=0.0046 va=0.0015
2026-05-10 23:39:42,458 INFO Regime score epoch  9/50 — tr=0.0046 va=0.0015
2026-05-10 23:39:42,967 INFO Regime score epoch 10/50 — tr=0.0045 va=0.0014 mae={'trend_score': 0.025, 'range_score': 0.0392, 'chop_score': 0.0287, 'volatility_percentile': 0.0183, 'consolidation_score': 0.0274}
2026-05-10 23:39:43,472 INFO Regime score epoch 11/50 — tr=0.0044 va=0.0014
2026-05-10 23:39:43,964 INFO Regime score epoch 12/50 — tr=0.0044 va=0.0014
2026-05-10 23:39:44,459 INFO Regime score epoch 13/50 — tr=0.0044 va=0.0013
2026-05-10 23:39:44,953 INFO Regime score epoch 14/50 — tr=0.0043 va=0.0013
2026-05-10 23:39:45,458 INFO Regime score epoch 15/50 — tr=0.0043 va=0.0013 mae={'trend_score': 0.0235, 'range_score': 0.0381, 'chop_score': 0.0269, 'volatility_percentile': 0.0171, 'consolidation_score': 0.0256}
2026-05-10 23:39:45,955 INFO Regime score epoch 16/50 — tr=0.0042 va=0.0013
2026-05-10 23:39:46,448 INFO Regime score epoch 17/50 — tr=0.0042 va=0.0012
2026-05-10 23:39:46,965 INFO Regime score epoch 18/50 — tr=0.0042 va=0.0012
2026-05-10 23:39:47,460 INFO Regime score epoch 19/50 — tr=0.0041 va=0.0012
2026-05-10 23:39:47,962 INFO Regime score epoch 20/50 — tr=0.0041 va=0.0012 mae={'trend_score': 0.0223, 'range_score': 0.0368, 'chop_score': 0.025, 'volatility_percentile': 0.0167, 'consolidation_score': 0.0248}
2026-05-10 23:39:48,450 INFO Regime score epoch 21/50 — tr=0.0041 va=0.0012
2026-05-10 23:39:48,952 INFO Regime score epoch 22/50 — tr=0.0040 va=0.0011
2026-05-10 23:39:49,465 INFO Regime score epoch 23/50 — tr=0.0040 va=0.0011
2026-05-10 23:39:49,970 INFO Regime score epoch 24/50 — tr=0.0040 va=0.0011
2026-05-10 23:39:50,473 INFO Regime score epoch 25/50 — tr=0.0040 va=0.0011 mae={'trend_score': 0.0214, 'range_score': 0.0356, 'chop_score': 0.0233, 'volatility_percentile': 0.0165, 'consolidation_score': 0.0241}
2026-05-10 23:39:50,978 INFO Regime score epoch 26/50 — tr=0.0039 va=0.0011
2026-05-10 23:39:51,471 INFO Regime score epoch 27/50 — tr=0.0039 va=0.0011
2026-05-10 23:39:51,991 INFO Regime score epoch 28/50 — tr=0.0039 va=0.0011
2026-05-10 23:39:52,512 INFO Regime score epoch 29/50 — tr=0.0039 va=0.0011
2026-05-10 23:39:53,007 INFO Regime score epoch 30/50 — tr=0.0039 va=0.0011 mae={'trend_score': 0.0207, 'range_score': 0.0353, 'chop_score': 0.0222, 'volatility_percentile': 0.016, 'consolidation_score': 0.0234}
2026-05-10 23:39:53,513 INFO Regime score epoch 31/50 — tr=0.0039 va=0.0010
2026-05-10 23:39:54,019 INFO Regime score epoch 32/50 — tr=0.0039 va=0.0010
2026-05-10 23:39:54,529 INFO Regime score epoch 33/50 — tr=0.0038 va=0.0010
2026-05-10 23:39:55,033 INFO Regime score epoch 34/50 — tr=0.0038 va=0.0010
2026-05-10 23:39:55,526 INFO Regime score epoch 35/50 — tr=0.0038 va=0.0010 mae={'trend_score': 0.0202, 'range_score': 0.0348, 'chop_score': 0.0221, 'volatility_percentile': 0.0159, 'consolidation_score': 0.0241}
2026-05-10 23:39:56,024 INFO Regime score epoch 36/50 — tr=0.0038 va=0.0010
2026-05-10 23:39:56,520 INFO Regime score epoch 37/50 — tr=0.0038 va=0.0010
2026-05-10 23:39:57,057 INFO Regime score epoch 38/50 — tr=0.0038 va=0.0010
2026-05-10 23:39:57,571 INFO Regime score epoch 39/50 — tr=0.0038 va=0.0010
2026-05-10 23:39:58,093 INFO Regime score epoch 40/50 — tr=0.0038 va=0.0010 mae={'trend_score': 0.0199, 'range_score': 0.035, 'chop_score': 0.022, 'volatility_percentile': 0.0154, 'consolidation_score': 0.0228}
2026-05-10 23:39:58,580 INFO Regime score epoch 41/50 — tr=0.0038 va=0.0010
2026-05-10 23:39:59,070 INFO Regime score epoch 42/50 — tr=0.0038 va=0.0010
2026-05-10 23:39:59,556 INFO Regime score epoch 43/50 — tr=0.0038 va=0.0010
2026-05-10 23:40:00,053 INFO Regime score epoch 44/50 — tr=0.0038 va=0.0010
2026-05-10 23:40:00,556 INFO Regime score epoch 45/50 — tr=0.0038 va=0.0010 mae={'trend_score': 0.0198, 'range_score': 0.0346, 'chop_score': 0.022, 'volatility_percentile': 0.0156, 'consolidation_score': 0.0225}
2026-05-10 23:40:01,045 INFO Regime score epoch 46/50 — tr=0.0038 va=0.0010
2026-05-10 23:40:01,549 INFO Regime score epoch 47/50 — tr=0.0038 va=0.0010
2026-05-10 23:40:02,057 INFO Regime score epoch 48/50 — tr=0.0038 va=0.0010
2026-05-10 23:40:02,588 INFO Regime score epoch 49/50 — tr=0.0038 va=0.0010
2026-05-10 23:40:03,083 INFO Regime score epoch 50/50 — tr=0.0038 va=0.0010 mae={'trend_score': 0.0207, 'range_score': 0.0344, 'chop_score': 0.0221, 'volatility_percentile': 0.0157, 'consolidation_score': 0.0224}
2026-05-10 23:40:03,104 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0198, 'range_score': 0.0346, 'chop_score': 0.022, 'volatility_percentile': 0.0156, 'consolidation_score': 0.0225} mse={'trend_score': 0.00068, 'range_score': 0.00196, 'chop_score': 0.00079, 'volatility_percentile': 0.00044, 'consolidation_score': 0.00115} corr={'trend_score': 0.9932, 'range_score': 0.9516, 'chop_score': 0.9897, 'volatility_percentile': 0.9953, 'consolidation_score': 0.9879} pred_std={'trend_score': 0.2187, 'range_score': 0.132, 'chop_score': 0.1793, 'volatility_percentile': 0.2168, 'consolidation_score': 0.2141} target_std={'trend_score': 0.2219, 'range_score': 0.1433, 'chop_score': 0.1888, 'volatility_percentile': 0.2165, 'consolidation_score': 0.2179}
2026-05-10 23:40:03,422 INFO RegimeClassifier[mode=ltf_behaviour] train group diagnostics={'dollar': {'n': 16632, 'score_mae': {'trend_score': 0.0191, 'range_score': 0.0345, 'chop_score': 0.0221, 'volatility_percentile': 0.0155, 'consolidation_score': 0.0229}, 'target_score_mean': {'trend_score': 0.491, 'range_score': 0.2347, 'chop_score': 0.4613, 'volatility_percentile': 0.3824, 'consolidation_score': 0.186}, 'pred_score_mean': {'trend_score': 0.4896, 'range_score': 0.2367, 'chop_score': 0.4617, 'volatility_percentile': 0.3798, 'consolidation_score': 0.1863}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3543, 37, 0, 4, 0, 0, 195], [10, 95, 0, 0, 0, 0, 5], [0, 0, 200, 10, 44, 0, 206], [2, 0, 8, 549, 34, 0, 96], [0, 0, 48, 23, 3052, 0, 193], [0, 26, 0, 0, 9, 22, 71], [128, 10, 112, 62, 69, 0, 7769]]}, 'gold': {'n': 8420, 'score_mae': {'trend_score': 0.0187, 'range_score': 0.0351, 'chop_score': 0.0222, 'volatility_percentile': 0.0161, 'consolidation_score': 0.0234}, 'target_score_mean': {'trend_score': 0.4883, 'range_score': 0.2355, 'chop_score': 0.4652, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}, 'pred_score_mean': {'trend_score': 0.4878, 'range_score': 0.2372, 'chop_score': 0.4651, 'volatility_percentile': 0.3742, 'consolidation_score': 0.1912}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1774, 25, 0, 0, 0, 0, 86], [4, 51, 0, 0, 0, 0, 1], [0, 0, 103, 9, 21, 0, 111], [1, 0, 2, 339, 20, 0, 54], [0, 0, 21, 25, 1552, 0, 106], [0, 22, 0, 0, 7, 20, 32], [62, 4, 63, 28, 50, 0, 3827]]}, 'jpy': {'n': 24948, 'score_mae': {'trend_score': 0.0192, 'range_score': 0.0346, 'chop_score': 0.0219, 'volatility_percentile': 0.0161, 'consolidation_score': 0.0229}, 'target_score_mean': {'trend_score': 0.4904, 'range_score': 0.2338, 'chop_score': 0.4653, 'volatility_percentile': 0.3826, 'consolidation_score': 0.1905}, 'pred_score_mean': {'trend_score': 0.4887, 'range_score': 0.236, 'chop_score': 0.4653, 'volatility_percentile': 0.3795, 'consolidation_score': 0.1892}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[5270, 74, 0, 5, 0, 0, 297], [22, 155, 0, 0, 0, 0, 10], [0, 0, 265, 16, 70, 0, 296], [1, 0, 4, 1067, 74, 0, 168], [0, 0, 61, 66, 4641, 0, 347], [0, 48, 0, 0, 16, 41, 118], [183, 11, 188, 104, 136, 0, 11194]]}}
2026-05-10 23:40:03,598 INFO RegimeClassifier[mode=ltf_behaviour] validation group diagnostics={'dollar': {'n': 10093, 'score_mae': {'trend_score': 0.0199, 'range_score': 0.0354, 'chop_score': 0.0221, 'volatility_percentile': 0.0152, 'consolidation_score': 0.0219}, 'target_score_mean': {'trend_score': 0.4863, 'range_score': 0.2392, 'chop_score': 0.4622, 'volatility_percentile': 0.3804, 'consolidation_score': 0.1814}, 'pred_score_mean': {'trend_score': 0.4856, 'range_score': 0.2391, 'chop_score': 0.463, 'volatility_percentile': 0.3782, 'consolidation_score': 0.1823}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[2265, 16, 0, 3, 0, 0, 128], [5, 47, 0, 0, 0, 0, 1], [0, 0, 133, 7, 41, 0, 135], [0, 0, 4, 337, 25, 0, 57], [0, 0, 34, 26, 1895, 0, 95], [0, 19, 0, 0, 4, 15, 39], [58, 6, 76, 49, 48, 0, 4525]]}, 'gold': {'n': 5117, 'score_mae': {'trend_score': 0.0188, 'range_score': 0.0335, 'chop_score': 0.022, 'volatility_percentile': 0.0156, 'consolidation_score': 0.0234}, 'target_score_mean': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}, 'pred_score_mean': {'trend_score': 0.4955, 'range_score': 0.235, 'chop_score': 0.4578, 'volatility_percentile': 0.3791, 'consolidation_score': 0.1829}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1092, 11, 0, 0, 0, 0, 64], [3, 31, 0, 0, 0, 0, 1], [0, 0, 75, 3, 12, 0, 81], [0, 0, 3, 219, 10, 0, 23], [0, 0, 14, 10, 798, 0, 65], [0, 9, 0, 0, 4, 9, 28], [42, 2, 52, 31, 27, 0, 2398]]}, 'jpy': {'n': 15142, 'score_mae': {'trend_score': 0.0201, 'range_score': 0.0344, 'chop_score': 0.022, 'volatility_percentile': 0.0159, 'consolidation_score': 0.0226}, 'target_score_mean': {'trend_score': 0.495, 'range_score': 0.2297, 'chop_score': 0.4582, 'volatility_percentile': 0.3808, 'consolidation_score': 0.1868}, 'pred_score_mean': {'trend_score': 0.4927, 'range_score': 0.2312, 'chop_score': 0.4591, 'volatility_percentile': 0.3781, 'consolidation_score': 0.1856}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3304, 42, 0, 1, 0, 0, 171], [13, 94, 0, 0, 0, 0, 8], [0, 0, 161, 11, 45, 0, 167], [1, 0, 5, 678, 38, 0, 105], [0, 0, 38, 29, 2550, 0, 200], [0, 25, 0, 0, 10, 27, 60], [108, 9, 108, 54, 89, 0, 6991]]}}
2026-05-10 23:40:03,603 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-10 23:40:03,603 INFO Regime phase LTF train fold=train_all: 25.8s
2026-05-10 23:40:03,706 INFO Regime LTF complete fold=train_all: score_accuracy=0.977, train=262644 val=30352 mae={'trend_score': 0.0198, 'range_score': 0.0346, 'chop_score': 0.022, 'volatility_percentile': 0.0156, 'consolidation_score': 0.0225}
2026-05-10 23:40:03,708 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-10 23:40:04,060 INFO Regime[1H mode=ltf_behaviour fold=latest] LTF score diagnostics on GBPUSD:
{'trend_score': {'mean': 0.4921, 'q10': 0.1918, 'q50': 0.4877, 'q90': 0.8002}, 'range_score': {'mean': 0.2337, 'q10': 0.0513, 'q50': 0.2125, 'q90': 0.4349}, 'chop_score': {'mean': 0.4612, 'q10': 0.2148, 'q50': 0.4503, 'q90': 0.7255}, 'volatility_percentile': {'mean': 0.3832, 'q10': 0.1018, 'q50': 0.3704, 'q90': 0.6823}, 'consolidation_score': {'mean': 0.1854, 'q10': 0.0, 'q50': 0.1185, 'q90': 0.506}}
2026-05-10 23:40:04,066 INFO Regime retrain total: 68.3s (370559 train+val samples)
2026-05-10 23:40:04,071 INFO Retrain complete. Total wall-clock: 68.3s
  DONE  Retrain regime [train-split retrain]
  SKIP  Quality/RL incremental retrain — clean train-only weights retained

=== Round 3: Backtest on last 3yr (post-retrain evaluation) ===
  START Round 3 - Post-retrain backtest (last 3yr)
2026-05-10 23:40:05,513 INFO === STEP 6: BACKTEST (round3) ===
2026-05-10 23:40:05,514 INFO BT_WINDOW=round3 — post-retrain eval: 2022-08-05 → 2025-08-05 (last 3yr)
2026-05-10 23:40:05,514 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-05-10 23:40:05,514 INFO Round 3 — running backtest: 2022-08-05 → 2025-08-05 (ml_trader, shared ML cache)
2026-05-10 23:40:05,515 INFO Round 3 — QualityScorer hard gate disabled; set BACKTEST_ENABLE_QUALITY_GATE=1 to enable it
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-10 23:41:50,581 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
2026-05-10 23:41:51,376 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
2026-05-10 23:41:51,668 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
2026-05-10 23:41:52,102 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:757: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:765: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:757: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:771: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:765: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:771: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-10 23:41:52,378 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
2026-05-10 23:41:52,414 WARNING _build_sequence_df: HTF frame 5M filled 461 warmup/alignment gaps with 0.000
2026-05-10 23:41:52,442 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
2026-05-10 23:41:52,484 WARNING _build_sequence_df: HTF frame 5M filled 461 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:757: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:765: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:757: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:771: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:765: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:771: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
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
2026-05-10 23:42:02,368 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
2026-05-10 23:42:02,398 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:757: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-10 23:42:02,464 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:765: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:771: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-10 23:42:02,507 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:757: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:765: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:771: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-10 23:42:57,713 INFO Round 3 backtest — 157 trades | avg WR=28.7% | avg PF=1.10 | avg Sharpe=0.63
2026-05-10 23:42:57,714 INFO   ml_trader: 157 trades | WR=28.7% | fixed PF=1.10 | Return=11.1% | ExpR=0.071 | DD=23.0% | Sharpe=0.63
2026-05-10 23:42:57,714 INFO   ml_trader gate_diagnostics: bars=403523 no_signal=183170 quality_block=0 session_skip=220090 density=106 pm_reject=0
2026-05-10 23:42:57,714 INFO   ml_trader no_signal_reasons: {'no_trade_uncertain': 88905, 'weak_gru_direction': 38905, 'no_trade_chop': 22793, 'no_trade_extreme_vol': 22183, 'gru_expected_r_below_threshold': 9122, 'tradeability_direction_conflict': 175, 'wait_pullback': 1074, 'expected_r_below_threshold': 13}
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 157
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (157 rows)
  DONE  Round 3 - Post-retrain backtest (last 3yr)
  Saved Round 3 result → round3_summary.json
  Journal after Round 3: 405 entries

  SKIP  Round 3 Quality+RL retrain — evaluation journals not used for fitting

======================================================================
  BLIND BACKTEST PIPELINE COMPLETE
======================================================================
  Round 1 (train-tail window)   trades=120  WR=30.8%  PF=1.223  Sharpe=1.351
  Round 2 (blind test)          trades=128  WR=26.6%  PF=0.888  Sharpe=-0.823
  Round 3 (last 3yr)            trades=157  WR=28.7%  PF=1.099  Sharpe=0.633


WARNING: GITHUB_TOKEN not set — skipping GitHub push
2026-05-10 23:42:58,034 INFO Round 3: wrote 157 journal entries (total in file: 405)