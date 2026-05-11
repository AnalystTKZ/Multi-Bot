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
2026-05-11 02:48:12,878 INFO Loading feature-engineered data...
2026-05-11 02:48:13,496 INFO Loaded 221743 rows, 202 features
2026-05-11 02:48:13,498 INFO Data span: 2016-01-04 -> 2025-08-05  (9.6 years)
2026-05-11 02:48:13,500 INFO Fold 000 train 2016-01-04 -> 2018-01-03 (47088 bars), val 2018-01-04 -> 2019-01-03 (23448 bars)
2026-05-11 02:48:13,501 INFO Fold 001 train 2016-01-04 -> 2019-01-03 (70536 bars), val 2019-01-04 -> 2020-01-03 (23377 bars)
2026-05-11 02:48:13,501 INFO Fold 002 train 2016-01-04 -> 2020-01-03 (93913 bars), val 2020-01-06 -> 2020-12-31 (23259 bars)
2026-05-11 02:48:13,501 INFO Fold 003 train 2016-01-04 -> 2020-12-31 (117172 bars), val 2021-01-04 -> 2022-01-03 (23507 bars)
2026-05-11 02:48:13,502 INFO Fold 004 train 2016-01-04 -> 2022-01-03 (140679 bars), val 2022-01-04 -> 2023-01-03 (23588 bars)
2026-05-11 02:48:13,502 INFO Fold 005 train 2016-01-04 -> 2022-08-05 (154539 bars), val 2022-08-05 -> 2023-08-04 (20412 bars)
2026-05-11 02:48:13,502 INFO No leakage confirmed: train/train_tail/internal folds end before final 2-year blind test

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
2026-05-11 02:48:23,266 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-05-11 02:48:23,266 INFO --- Training gru ---
2026-05-11 02:48:23,266 INFO Running retrain --model gru
2026-05-11 02:48:23,463 INFO retrain environment: KAGGLE
2026-05-11 02:48:25,114 INFO Device: CUDA (2 GPU(s))
2026-05-11 02:48:25,125 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 02:48:25,125 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 02:48:25,125 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 02:48:25,127 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 02:48:25,127 INFO Retrain data split: train
2026-05-11 02:48:25,127 INFO Retrain rolling fold selector: latest
2026-05-11 02:48:25,129 INFO === GRU-LSTM retrain (timeframes: ['15M']) ===
2026-05-11 02:48:25,288 INFO NumExpr defaulting to 4 threads.
2026-05-11 02:48:25,514 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-11 02:48:25,514 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 02:48:25,514 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 02:48:26,550 INFO GRULSTMPredictor: loaded long R isotonic calibrator from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/r_isotonic_long.pkl
2026-05-11 02:48:26,550 INFO GRULSTMPredictor: loaded short R isotonic calibrator from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/r_isotonic_short.pkl
2026-05-11 02:48:26,550 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-05-11 02:48:26,550 INFO GRU multi-symbol training (Kaggle mode): 6 symbols × ['15M']
2026-05-11 02:48:26,552 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260511_024826
2026-05-11 02:48:26,557 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-05-11 02:48:26,557 INFO GRU warm start enabled from existing weights: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 02:48:26,826 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 02:48:26,856 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 02:48:26,891 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 02:48:26,901 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 02:48:26,979 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 02:48:26,985 INFO Loaded XAUUSD/15M split=train fold=latest: 176438 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:48:27,595 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:48:27,627 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:48:27,642 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:48:27,650 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:48:27,698 INFO Loaded EURUSD/15M split=train fold=latest: 174951 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:48:28,305 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:48:28,328 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:48:28,346 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:48:28,354 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:48:28,395 INFO Loaded USDJPY/15M split=train fold=latest: 174944 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:48:28,970 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:48:28,990 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:48:29,006 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:48:29,014 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:48:29,056 INFO Loaded EURJPY/15M split=train fold=latest: 174948 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:48:29,632 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:48:29,655 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:48:29,671 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:48:29,679 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:48:29,723 INFO Loaded GBPJPY/15M split=train fold=latest: 174902 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:48:30,291 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:48:30,311 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:48:30,327 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:48:30,335 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:48:30,377 INFO Loaded GBPUSD/15M split=train fold=latest: 174899 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:48:30,852 INFO train_multi: 6 segments, ~971854 total bars
2026-05-11 02:48:30,852 INFO train_multi: training ALL 6 segments across TFs ['15M'] in one combined pass
2026-05-11 02:48:30,852 INFO train_multi: building combined dataset for TF=ALL (6 segments)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:48:40,701 INFO train_multi TF=ALL: 971494 sequences across 6 segments
2026-05-11 02:48:40,701 INFO train_multi TF=ALL: estimated peak RAM = 21312 MB (train=419996 calib=60000 val=120002 n_feat=74 seq_len=60)
2026-05-11 02:48:40,701 WARNING train_multi TF=ALL: trimming to fit RAM budget — new train=394144 calib=56306 val=112612 (20000 MB est)
2026-05-11 02:48:43,110 INFO train_multi TF=ALL: train=394144 calib=56306 val=112612 (10009 MB tensors)
2026-05-11 02:48:50,268 INFO train_multi TF=ALL: structural bar weighting — 252452 structural bars (64.1%) weight=15.0 structural_only=0
2026-05-11 02:48:53,310 INFO train_multi TF=ALL: warm-start detected — using CosineAnnealingLR (lr=3e-05, patience=12)
2026-05-11 02:49:12,172 INFO train_multi TF=ALL epoch 1/100 train=1.7257 val=1.8537 r_mae=0.772 pos_r_acc=0.687 side_acc=0.740 r_n=161888
2026-05-11 02:49:12,179 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 02:49:12,179 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 02:49:12,179 INFO train_multi TF=ALL: new best val=1.8537 — saved
2026-05-11 02:49:28,255 INFO train_multi TF=ALL epoch 2/100 train=1.7295 val=1.8573 r_mae=0.772 pos_r_acc=0.687 side_acc=0.738 r_n=161888
2026-05-11 02:49:44,404 INFO train_multi TF=ALL epoch 3/100 train=1.7202 val=1.8616 r_mae=0.772 pos_r_acc=0.687 side_acc=0.738 r_n=161888
2026-05-11 02:50:00,568 INFO train_multi TF=ALL epoch 4/100 train=1.7213 val=1.8639 r_mae=0.773 pos_r_acc=0.687 side_acc=0.736 r_n=161888
2026-05-11 02:50:16,782 INFO train_multi TF=ALL epoch 5/100 train=1.7201 val=1.8605 r_mae=0.772 pos_r_acc=0.688 side_acc=0.738 r_n=161888
2026-05-11 02:50:32,905 INFO train_multi TF=ALL epoch 6/100 train=1.7181 val=1.8597 r_mae=0.772 pos_r_acc=0.688 side_acc=0.739 r_n=161888
2026-05-11 02:50:49,028 INFO train_multi TF=ALL epoch 7/100 train=1.7171 val=1.8629 r_mae=0.771 pos_r_acc=0.688 side_acc=0.738 r_n=161888
2026-05-11 02:51:05,236 INFO train_multi TF=ALL epoch 8/100 train=1.7190 val=1.8602 r_mae=0.774 pos_r_acc=0.686 side_acc=0.739 r_n=161888
2026-05-11 02:51:21,550 INFO train_multi TF=ALL epoch 9/100 train=1.7156 val=1.8591 r_mae=0.771 pos_r_acc=0.688 side_acc=0.739 r_n=161888
2026-05-11 02:51:37,758 INFO train_multi TF=ALL epoch 10/100 train=1.7141 val=1.8636 r_mae=0.772 pos_r_acc=0.687 side_acc=0.738 r_n=161888
2026-05-11 02:51:53,970 INFO train_multi TF=ALL epoch 11/100 train=1.7135 val=1.8655 r_mae=0.770 pos_r_acc=0.689 side_acc=0.736 r_n=161888
2026-05-11 02:52:10,324 INFO train_multi TF=ALL epoch 12/100 train=1.7133 val=1.8618 r_mae=0.772 pos_r_acc=0.687 side_acc=0.738 r_n=161888
2026-05-11 02:52:26,518 INFO train_multi TF=ALL epoch 13/100 train=1.7125 val=1.8615 r_mae=0.772 pos_r_acc=0.687 side_acc=0.739 r_n=161888
2026-05-11 02:52:26,518 INFO train_multi TF=ALL early stop at epoch 13
2026-05-11 02:52:26,927 INFO fit_r_isotonic: saved side-R calibrators stats={'long': {'n': 40072, 'raw_mae': 0.7426727622287902, 'calibrated_mae': 0.7191678399136859}, 'short': {'n': 41197, 'raw_mae': 0.7422866340389652, 'calibrated_mae': 0.7256542441591981}}
2026-05-11 02:52:27,083 WARNING GRU validation R-MAE above floor for TF=ALL: best_val_r_mae=0.772 max=0.750. Keeping saved best weights so the pipeline can progress.
2026-05-11 02:52:27,097 INFO GRU R threshold XAUUSD/buy: q25_pos=0.535 q50_pos=1.667 pos_rate=46.0% (n=119259 n_pos=54899)
2026-05-11 02:52:27,103 INFO GRU R threshold XAUUSD/sell: q25_pos=0.528 q50_pos=1.667 pos_rate=45.5% (n=118087 n_pos=53734)
2026-05-11 02:52:27,109 INFO GRU R threshold EURUSD/buy: q25_pos=0.541 q50_pos=1.667 pos_rate=45.6% (n=118751 n_pos=54196)
2026-05-11 02:52:27,115 INFO GRU R threshold EURUSD/sell: q25_pos=0.545 q50_pos=1.667 pos_rate=45.6% (n=117679 n_pos=53721)
2026-05-11 02:52:27,121 INFO GRU R threshold USDJPY/buy: q25_pos=0.551 q50_pos=1.667 pos_rate=46.2% (n=118691 n_pos=54870)
2026-05-11 02:52:27,127 INFO GRU R threshold USDJPY/sell: q25_pos=0.542 q50_pos=1.667 pos_rate=44.6% (n=116030 n_pos=51793)
2026-05-11 02:52:27,133 INFO GRU R threshold EURJPY/buy: q25_pos=0.530 q50_pos=1.416 pos_rate=46.1% (n=118651 n_pos=54654)
2026-05-11 02:52:27,139 INFO GRU R threshold EURJPY/sell: q25_pos=0.525 q50_pos=1.667 pos_rate=44.7% (n=117320 n_pos=52399)
2026-05-11 02:52:27,146 INFO GRU R threshold GBPJPY/buy: q25_pos=0.517 q50_pos=1.443 pos_rate=45.9% (n=118277 n_pos=54278)
2026-05-11 02:52:27,152 INFO GRU R threshold GBPJPY/sell: q25_pos=0.521 q50_pos=1.667 pos_rate=45.3% (n=116299 n_pos=52702)
2026-05-11 02:52:27,158 INFO GRU R threshold GBPUSD/buy: q25_pos=0.524 q50_pos=1.667 pos_rate=45.8% (n=117764 n_pos=53894)
2026-05-11 02:52:27,164 INFO GRU R threshold GBPUSD/sell: q25_pos=0.533 q50_pos=1.667 pos_rate=45.8% (n=117404 n_pos=53785)
2026-05-11 02:52:27,165 INFO GRU per-symbol R thresholds saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/symbol_r_thresholds.json (6 symbols)
2026-05-11 02:52:27,204 INFO Retrain complete. Total wall-clock: 242.1s
2026-05-11 02:52:31,694 INFO Model gru: SUCCESS
2026-05-11 02:52:31,694 INFO --- Training regime ---
2026-05-11 02:52:31,694 INFO Running retrain --model regime
2026-05-11 02:52:31,984 INFO retrain environment: KAGGLE
2026-05-11 02:52:33,639 INFO Device: CUDA (2 GPU(s))
2026-05-11 02:52:33,648 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 02:52:33,648 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 02:52:33,648 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 02:52:33,649 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 02:52:33,649 INFO Retrain data split: train
2026-05-11 02:52:33,649 INFO Retrain rolling fold selector: latest
2026-05-11 02:52:33,650 INFO === RegimeClassifier retrain (hierarchical: HTF 2-score bias + LTF 5-score behaviour) ===
2026-05-11 02:52:33,877 INFO NumExpr defaulting to 4 threads.
2026-05-11 02:52:34,101 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-11 02:52:34,101 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 02:52:34,101 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 02:52:34,102 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-11 02:52:34,169 INFO Regime rolling folds selected: [None]
2026-05-11 02:52:34,170 INFO === Regime rolling fold 1/1: train_all ===
2026-05-11 02:52:34,170 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-11 02:52:34,212 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 02:52:34,213 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
2026-05-11 02:52:34,229 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
2026-05-11 02:52:34,247 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
2026-05-11 02:52:34,264 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
2026-05-11 02:52:34,281 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
2026-05-11 02:52:34,298 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 02:52:34,552 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 02:52:34,625 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 02:52:34,651 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 02:52:34,651 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 02:52:34,662 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 02:52:34,663 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:52:35,278 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 550, 'BIAS_DOWN': 255, 'BIAS_NEUTRAL': 11297}  ambiguous=7029 (total=12102) horizon=36
2026-05-11 02:52:35,283 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected XAUUSD — 12052 samples (group=gold) score_means={'bias_up_score': 0.0456, 'bias_down_score': 0.0212} labels={'BIAS_UP': 550, 'BIAS_DOWN': 255, 'BIAS_NEUTRAL': 11247} clean={'BIAS_UP': 550, 'BIAS_DOWN': 255, 'BIAS_NEUTRAL': 4250}
2026-05-11 02:52:35,466 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:35,506 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:35,527 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:35,527 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:35,536 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:35,538 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:52:36,129 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 299, 'BIAS_DOWN': 288, 'BIAS_NEUTRAL': 10817}  ambiguous=6558 (total=11404) horizon=36
2026-05-11 02:52:36,135 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURUSD — 11354 samples (group=dollar) score_means={'bias_up_score': 0.0263, 'bias_down_score': 0.0254} labels={'BIAS_UP': 299, 'BIAS_DOWN': 288, 'BIAS_NEUTRAL': 10767} clean={'BIAS_UP': 299, 'BIAS_DOWN': 288, 'BIAS_NEUTRAL': 4231}
2026-05-11 02:52:36,316 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:36,354 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:36,375 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:36,375 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:36,384 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:36,385 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:52:36,931 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 431, 'BIAS_DOWN': 164, 'BIAS_NEUTRAL': 10808}  ambiguous=6695 (total=11403) horizon=36
2026-05-11 02:52:36,937 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected USDJPY — 11353 samples (group=dollar) score_means={'bias_up_score': 0.038, 'bias_down_score': 0.0144} labels={'BIAS_UP': 431, 'BIAS_DOWN': 164, 'BIAS_NEUTRAL': 10758} clean={'BIAS_UP': 431, 'BIAS_DOWN': 164, 'BIAS_NEUTRAL': 4092}
2026-05-11 02:52:37,103 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:37,142 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:37,164 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:37,165 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:37,174 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:37,175 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:52:37,733 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 313, 'BIAS_DOWN': 162, 'BIAS_NEUTRAL': 10932}  ambiguous=6806 (total=11407) horizon=36
2026-05-11 02:52:37,739 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURJPY — 11357 samples (group=cross) score_means={'bias_up_score': 0.0276, 'bias_down_score': 0.0143} labels={'BIAS_UP': 313, 'BIAS_DOWN': 162, 'BIAS_NEUTRAL': 10882} clean={'BIAS_UP': 313, 'BIAS_DOWN': 162, 'BIAS_NEUTRAL': 4094}
2026-05-11 02:52:37,899 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:37,939 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:37,961 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:37,961 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:37,970 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:37,972 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:52:38,521 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 467, 'BIAS_DOWN': 285, 'BIAS_NEUTRAL': 10656}  ambiguous=6835 (total=11408) horizon=36
2026-05-11 02:52:38,527 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPJPY — 11358 samples (group=cross) score_means={'bias_up_score': 0.0411, 'bias_down_score': 0.0251} labels={'BIAS_UP': 467, 'BIAS_DOWN': 285, 'BIAS_NEUTRAL': 10606} clean={'BIAS_UP': 467, 'BIAS_DOWN': 285, 'BIAS_NEUTRAL': 3818}
2026-05-11 02:52:38,692 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:38,728 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:38,748 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:38,748 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:38,757 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:38,758 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:52:39,302 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 255, 'BIAS_DOWN': 300, 'BIAS_NEUTRAL': 10847}  ambiguous=6860 (total=11402) horizon=36
2026-05-11 02:52:39,307 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPUSD — 11352 samples (group=dollar) score_means={'bias_up_score': 0.0225, 'bias_down_score': 0.0264} labels={'BIAS_UP': 255, 'BIAS_DOWN': 300, 'BIAS_NEUTRAL': 10797} clean={'BIAS_UP': 255, 'BIAS_DOWN': 300, 'BIAS_NEUTRAL': 3987}
2026-05-11 02:52:39,381 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 780, 'BIAS_DOWN': 447, 'BIAS_NEUTRAL': 21488}, 'dollar': {'BIAS_UP': 985, 'BIAS_DOWN': 752, 'BIAS_NEUTRAL': 32322}, 'gold': {'BIAS_UP': 550, 'BIAS_DOWN': 255, 'BIAS_NEUTRAL': 11247}}
2026-05-11 02:52:39,382 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0343, 'bias_down_score': 0.0197}, 'dollar': {'bias_up_score': 0.0289, 'bias_down_score': 0.0221}, 'gold': {'bias_up_score': 0.0456, 'bias_down_score': 0.0212}}
2026-05-11 02:52:39,382 INFO Regime[4H mode=htf_bias] label distribution by year: {2016: {'BIAS_UP': 274, 'BIAS_DOWN': 351, 'BIAS_NEUTRAL': 8197}, 2017: {'BIAS_UP': 495, 'BIAS_DOWN': 167, 'BIAS_NEUTRAL': 8451}, 2018: {'BIAS_UP': 208, 'BIAS_DOWN': 250, 'BIAS_NEUTRAL': 8672}, 2019: {'BIAS_UP': 189, 'BIAS_DOWN': 174, 'BIAS_NEUTRAL': 8739}, 2020: {'BIAS_UP': 285, 'BIAS_DOWN': 131, 'BIAS_NEUTRAL': 8695}, 2021: {'BIAS_UP': 311, 'BIAS_DOWN': 163, 'BIAS_NEUTRAL': 8617}, 2022: {'BIAS_UP': 340, 'BIAS_DOWN': 162, 'BIAS_NEUTRAL': 8619}, 2023: {'BIAS_UP': 213, 'BIAS_DOWN': 56, 'BIAS_NEUTRAL': 5067}}
2026-05-11 02:52:39,382 INFO Regime[4H mode=htf_bias] score means by year: {2016: {'bias_up_score': 0.0311, 'bias_down_score': 0.0398}, 2017: {'bias_up_score': 0.0543, 'bias_down_score': 0.0183}, 2018: {'bias_up_score': 0.0228, 'bias_down_score': 0.0274}, 2019: {'bias_up_score': 0.0208, 'bias_down_score': 0.0191}, 2020: {'bias_up_score': 0.0313, 'bias_down_score': 0.0144}, 2021: {'bias_up_score': 0.0342, 'bias_down_score': 0.0179}, 2022: {'bias_up_score': 0.0373, 'bias_down_score': 0.0178}, 2023: {'bias_up_score': 0.0399, 'bias_down_score': 0.0105}}
2026-05-11 02:52:39,444 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
2026-05-11 02:52:39,446 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
2026-05-11 02:52:39,446 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
2026-05-11 02:52:39,447 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 02:52:39,448 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 02:52:39,449 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
2026-05-11 02:52:39,466 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 02:52:39,470 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 02:52:39,471 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 02:52:39,472 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 02:52:39,472 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 02:52:39,473 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:52:39,809 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 53, 'BIAS_DOWN': 23, 'BIAS_NEUTRAL': 1505}  ambiguous=929 (total=1581) horizon=36
2026-05-11 02:52:39,812 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected XAUUSD — 1531 samples (group=gold) score_means={'bias_up_score': 0.0346, 'bias_down_score': 0.015} labels={'BIAS_UP': 53, 'BIAS_DOWN': 23, 'BIAS_NEUTRAL': 1455} clean={'BIAS_UP': 53, 'BIAS_DOWN': 23, 'BIAS_NEUTRAL': 558}
2026-05-11 02:52:39,892 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:39,895 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:39,896 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:39,896 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:39,897 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:39,898 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:52:40,228 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 18, 'BIAS_DOWN': 38, 'BIAS_NEUTRAL': 1435}  ambiguous=844 (total=1491) horizon=36
2026-05-11 02:52:40,231 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURUSD — 1441 samples (group=dollar) score_means={'bias_up_score': 0.0125, 'bias_down_score': 0.0264} labels={'BIAS_UP': 18, 'BIAS_DOWN': 38, 'BIAS_NEUTRAL': 1385} clean={'BIAS_UP': 18, 'BIAS_DOWN': 38, 'BIAS_NEUTRAL': 577}
2026-05-11 02:52:40,313 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:40,315 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:40,316 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:40,317 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:40,317 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:40,318 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:52:40,642 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 95, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 1386}  ambiguous=905 (total=1489) horizon=36
2026-05-11 02:52:40,645 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected USDJPY — 1439 samples (group=dollar) score_means={'bias_up_score': 0.066, 'bias_down_score': 0.0056} labels={'BIAS_UP': 95, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 1336} clean={'BIAS_UP': 95, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 478}
2026-05-11 02:52:40,721 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:40,724 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:40,725 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:40,725 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:40,725 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:40,726 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:52:41,036 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 52, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 1442}  ambiguous=913 (total=1494) horizon=36
2026-05-11 02:52:41,039 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.036, 'bias_down_score': 0.0} labels={'BIAS_UP': 52, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 1392} clean={'BIAS_UP': 52, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 510}
2026-05-11 02:52:41,112 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:41,114 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:41,115 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:41,115 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:41,116 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:41,117 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:52:41,442 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 40, 'BIAS_DOWN': 3, 'BIAS_NEUTRAL': 1451}  ambiguous=884 (total=1494) horizon=36
2026-05-11 02:52:41,444 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0277, 'bias_down_score': 0.0021} labels={'BIAS_UP': 40, 'BIAS_DOWN': 3, 'BIAS_NEUTRAL': 1401} clean={'BIAS_UP': 40, 'BIAS_DOWN': 3, 'BIAS_NEUTRAL': 542}
2026-05-11 02:52:41,535 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:41,538 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:41,539 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:41,539 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:41,539 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:52:41,541 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:52:41,861 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 18, 'BIAS_DOWN': 29, 'BIAS_NEUTRAL': 1441}  ambiguous=896 (total=1488) horizon=36
2026-05-11 02:52:41,864 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPUSD — 1438 samples (group=dollar) score_means={'bias_up_score': 0.0125, 'bias_down_score': 0.0202} labels={'BIAS_UP': 18, 'BIAS_DOWN': 29, 'BIAS_NEUTRAL': 1391} clean={'BIAS_UP': 18, 'BIAS_DOWN': 29, 'BIAS_NEUTRAL': 533}
2026-05-11 02:52:41,936 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 92, 'BIAS_DOWN': 3, 'BIAS_NEUTRAL': 2793}, 'dollar': {'BIAS_UP': 131, 'BIAS_DOWN': 75, 'BIAS_NEUTRAL': 4112}, 'gold': {'BIAS_UP': 53, 'BIAS_DOWN': 23, 'BIAS_NEUTRAL': 1455}}
2026-05-11 02:52:41,936 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0319, 'bias_down_score': 0.001}, 'dollar': {'bias_up_score': 0.0303, 'bias_down_score': 0.0174}, 'gold': {'bias_up_score': 0.0346, 'bias_down_score': 0.015}}
2026-05-11 02:52:41,936 INFO Regime[4H mode=htf_bias] label distribution by year: {2022: {'BIAS_UP': 72, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 3283}, 2023: {'BIAS_UP': 204, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 5077}}
2026-05-11 02:52:41,936 INFO Regime[4H mode=htf_bias] score means by year: {2022: {'bias_up_score': 0.0212, 'bias_down_score': 0.0135}, 2023: {'bias_up_score': 0.0382, 'bias_down_score': 0.0103}}
2026-05-11 02:52:41,995 INFO Regime phase HTF dataset build fold=train_all: 7.8s (train=68826 val=8737)
2026-05-11 02:52:41,995 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260511_025241
2026-05-11 02:52:42,209 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=28, n_classes=2)
2026-05-11 02:52:42,209 INFO Regime 4H/htf_bias warm start enabled from existing weights
2026-05-11 02:52:42,215 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=68826 val=8737 train_labels={'BIAS_UP': 2315, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 65057} val_labels={'BIAS_UP': 276, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 8360}
2026-05-11 02:52:42,216 INFO RegimeClassifier[mode=htf_bias]: warm start HTF score head
2026-05-11 02:52:42,216 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-11 02:52:42,216 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 28.73, 'bias_down_score': 30.0}
2026-05-11 02:52:42,220 INFO RegimeClassifier[mode=htf_bias]: HTF balanced sampler — dir=3769 neutral=65057 dir_weight=8 => dir_frac_per_epoch≈31.7%
2026-05-11 02:52:46,124 INFO Regime HTF score epoch  1/50 — tr=0.5811 va=0.9352 acc=0.824 bal=0.930 threshold=0.99 margin=0.00 recall={'BIAS_UP': 0.993, 'BIAS_DOWN': 0.98, 'BIAS_NEUTRAL': 0.817} precision={'BIAS_UP': 0.222, 'BIAS_DOWN': 0.147, 'BIAS_NEUTRAL': 0.999}
2026-05-11 02:52:47,833 INFO Regime HTF score epoch  2/50 — tr=0.5830 va=0.9681 bal=0.927
2026-05-11 02:52:49,561 INFO Regime HTF score epoch  3/50 — tr=0.5787 va=0.9615 bal=0.928
2026-05-11 02:52:51,310 INFO Regime HTF score epoch  4/50 — tr=0.5468 va=0.8982 bal=0.930
2026-05-11 02:52:53,059 INFO Regime HTF score epoch  5/50 — tr=0.5251 va=0.8592 acc=0.841 bal=0.929 threshold=0.99 margin=0.00 recall={'BIAS_UP': 0.982, 'BIAS_DOWN': 0.97, 'BIAS_NEUTRAL': 0.835} precision={'BIAS_UP': 0.238, 'BIAS_DOWN': 0.161, 'BIAS_NEUTRAL': 0.999}
2026-05-11 02:52:54,760 INFO Regime HTF score epoch  6/50 — tr=0.5107 va=0.7842 bal=0.912
2026-05-11 02:52:56,463 INFO Regime HTF score epoch  7/50 — tr=0.4840 va=0.7353 bal=0.893
2026-05-11 02:52:58,191 INFO Regime HTF score epoch  8/50 — tr=0.4667 va=0.6859 bal=0.873
2026-05-11 02:52:59,889 INFO Regime HTF score epoch  9/50 — tr=0.4665 va=0.6739 bal=0.860
2026-05-11 02:53:01,623 INFO Regime HTF score epoch 10/50 — tr=0.4648 va=0.6684 acc=0.878 bal=0.852 threshold=0.99 margin=0.00 recall={'BIAS_UP': 0.866, 'BIAS_DOWN': 0.812, 'BIAS_NEUTRAL': 0.88} precision={'BIAS_UP': 0.279, 'BIAS_DOWN': 0.174, 'BIAS_NEUTRAL': 0.992}
2026-05-11 02:53:03,342 INFO Regime HTF score epoch 11/50 — tr=0.4659 va=0.6596 bal=0.845
2026-05-11 02:53:05,058 INFO Regime HTF score epoch 12/50 — tr=0.4563 va=0.6698 bal=0.853
2026-05-11 02:53:06,754 INFO Regime HTF score epoch 13/50 — tr=0.4588 va=0.6735 bal=0.852
2026-05-11 02:53:08,488 INFO Regime HTF score epoch 14/50 — tr=0.4600 va=0.6691 bal=0.843
2026-05-11 02:53:10,203 INFO Regime HTF score epoch 15/50 — tr=0.4508 va=0.6700 acc=0.880 bal=0.844 threshold=0.99 margin=0.00 recall={'BIAS_UP': 0.87, 'BIAS_DOWN': 0.782, 'BIAS_NEUTRAL': 0.881} precision={'BIAS_UP': 0.281, 'BIAS_DOWN': 0.173, 'BIAS_NEUTRAL': 0.992}
2026-05-11 02:53:11,961 INFO Regime HTF score epoch 16/50 — tr=0.4517 va=0.6677 bal=0.846
2026-05-11 02:53:11,961 INFO Regime HTF score early stop at epoch 16
2026-05-11 02:53:13,547 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.990 margin=0.000 precision={'BIAS_UP': 0.283, 'BIAS_DOWN': 0.175, 'BIAS_NEUTRAL': 0.994} recall={'BIAS_UP': 0.891, 'BIAS_DOWN': 0.851, 'BIAS_NEUTRAL': 0.877} f1={'BIAS_UP': 0.429, 'BIAS_DOWN': 0.291, 'BIAS_NEUTRAL': 0.932} confusion=[[246, 0, 30], [0, 86, 15], [624, 405, 7331]] score_mae={'bias_up_score': 0.1838, 'bias_down_score': 0.1217} pred_share={'BIAS_UP': 0.0996, 'BIAS_DOWN': 0.0562, 'BIAS_NEUTRAL': 0.8442}
2026-05-11 02:53:13,548 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.283, 'BIAS_DOWN': 0.175, 'BIAS_NEUTRAL': 0.994} min_precision=0.500 recall={'BIAS_UP': 0.891, 'BIAS_DOWN': 0.851, 'BIAS_NEUTRAL': 0.877} min_recall=0.100 f1={'BIAS_UP': 0.429, 'BIAS_DOWN': 0.291, 'BIAS_NEUTRAL': 0.932} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_UP', 'BIAS_DOWN'] weak_recall=[] weak_f1=[] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-11 02:53:13,552 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 02:53:13,552 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 02:53:13,553 INFO Regime phase HTF train fold=train_all: 31.3s
2026-05-11 02:53:13,665 INFO Regime HTF complete fold=train_all: acc=0.877 bal=0.873 train=68826 val=8737 per_class={'BIAS_UP': 0.891, 'BIAS_DOWN': 0.851, 'BIAS_NEUTRAL': 0.877} precision={'BIAS_UP': 0.283, 'BIAS_DOWN': 0.175, 'BIAS_NEUTRAL': 0.994} threshold=0.990 margin=0.000
2026-05-11 02:53:13,666 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 02:53:13,831 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 255, 'BIAS_DOWN': 300, 'BIAS_NEUTRAL': 10847}  ambiguous=6860 (total=11402) horizon=36
2026-05-11 02:53:13,837 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on GBPUSD 4H:
{'BIAS_UP': 3.8636363636363638, 'BIAS_DOWN': 3.5294117647058822, 'BIAS_NEUTRAL': 71.36184210526316}
2026-05-11 02:53:13,840 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (all labels):
{'BIAS_UP': {'n': 255, 'mean': 0.0006942435718027296, 'mean_over_std': 0.3535182817741541}, 'BIAS_DOWN': {'n': 300, 'mean': -0.0007615786091842384, 'mean_over_std': -0.2905559984734911}, 'BIAS_NEUTRAL': {'n': 10846, 'mean': -5.264589798607426e-06, 'mean_over_std': -0.002007943010728463}}
2026-05-11 02:53:13,841 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 255, 'mean': 0.0006942435718027296, 'mean_over_std': 0.3535182817741541}, 'BIAS_DOWN': {'n': 300, 'mean': -0.0007615786091842384, 'mean_over_std': -0.2905559984734911}, 'BIAS_NEUTRAL': {'n': 3987, 'mean': -1.7489525578049586e-05, 'mean_over_std': -0.007235149795477735}}
2026-05-11 02:53:13,850 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-11 02:53:13,852 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
2026-05-11 02:53:13,854 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
2026-05-11 02:53:13,856 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
2026-05-11 02:53:13,858 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
2026-05-11 02:53:13,860 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
2026-05-11 02:53:13,862 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 02:53:13,882 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 02:53:13,890 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 02:53:13,893 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 02:53:13,894 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 02:53:13,894 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 02:53:13,900 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:53:14,807 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected XAUUSD — 44227 samples (group=gold) score_means={'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}
2026-05-11 02:53:14,927 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:14,929 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:14,930 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:14,931 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:14,931 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:14,934 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:53:15,783 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURUSD — 43689 samples (group=dollar) score_means={'trend_score': 0.49, 'range_score': 0.236, 'chop_score': 0.4622, 'volatility_percentile': 0.3812, 'consolidation_score': 0.1862}
2026-05-11 02:53:15,904 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:15,907 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:15,907 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:15,908 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:15,908 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:15,911 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:53:16,754 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected USDJPY — 43686 samples (group=dollar) score_means={'trend_score': 0.4949, 'range_score': 0.2312, 'chop_score': 0.4613, 'volatility_percentile': 0.38, 'consolidation_score': 0.1931}
2026-05-11 02:53:16,873 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:16,875 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:16,876 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:16,877 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:16,877 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:16,879 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:53:17,726 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURJPY — 43688 samples (group=cross) score_means={'trend_score': 0.4888, 'range_score': 0.2345, 'chop_score': 0.4662, 'volatility_percentile': 0.3831, 'consolidation_score': 0.1882}
2026-05-11 02:53:17,842 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:17,845 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:17,846 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:17,846 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:17,846 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:17,849 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:53:18,693 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPJPY — 43678 samples (group=cross) score_means={'trend_score': 0.4878, 'range_score': 0.2348, 'chop_score': 0.4673, 'volatility_percentile': 0.3839, 'consolidation_score': 0.1904}
2026-05-11 02:53:18,809 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:18,811 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:18,812 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:18,813 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:18,813 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:18,815 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:53:19,677 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPUSD — 43676 samples (group=dollar) score_means={'trend_score': 0.4923, 'range_score': 0.2335, 'chop_score': 0.4611, 'volatility_percentile': 0.3832, 'consolidation_score': 0.1856}
2026-05-11 02:53:19,795 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2346, 'chop_score': 0.4667, 'volatility_percentile': 0.3835, 'consolidation_score': 0.1893}, 'dollar': {'trend_score': 0.4924, 'range_score': 0.2335, 'chop_score': 0.4615, 'volatility_percentile': 0.3815, 'consolidation_score': 0.1883}, 'gold': {'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}}
2026-05-11 02:53:19,795 INFO Regime[1H mode=ltf_behaviour] score means by year: {2016: {'trend_score': 0.495, 'range_score': 0.233, 'chop_score': 0.4607, 'volatility_percentile': 0.3776, 'consolidation_score': 0.192}, 2017: {'trend_score': 0.5017, 'range_score': 0.2328, 'chop_score': 0.4574, 'volatility_percentile': 0.3689, 'consolidation_score': 0.1942}, 2018: {'trend_score': 0.4886, 'range_score': 0.237, 'chop_score': 0.4655, 'volatility_percentile': 0.3947, 'consolidation_score': 0.1785}, 2019: {'trend_score': 0.4817, 'range_score': 0.236, 'chop_score': 0.4691, 'volatility_percentile': 0.3725, 'consolidation_score': 0.198}, 2020: {'trend_score': 0.4874, 'range_score': 0.2311, 'chop_score': 0.4674, 'volatility_percentile': 0.3848, 'consolidation_score': 0.1919}, 2021: {'trend_score': 0.4872, 'range_score': 0.2372, 'chop_score': 0.4672, 'volatility_percentile': 0.377, 'consolidation_score': 0.192}, 2022: {'trend_score': 0.4893, 'range_score': 0.2357, 'chop_score': 0.4647, 'volatility_percentile': 0.3962, 'consolidation_score': 0.1784}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 02:53:19,909 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
2026-05-11 02:53:19,910 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 02:53:19,912 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
2026-05-11 02:53:19,913 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 02:53:19,914 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
2026-05-11 02:53:19,916 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
2026-05-11 02:53:19,925 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 02:53:19,929 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 02:53:19,930 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 02:53:19,930 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 02:53:19,931 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 02:53:19,933 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:53:20,196 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected XAUUSD — 5117 samples (group=gold) score_means={'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}
2026-05-11 02:53:20,312 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:20,316 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:20,317 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:20,317 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:20,318 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:20,319 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:53:20,567 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURUSD — 5053 samples (group=dollar) score_means={'trend_score': 0.4883, 'range_score': 0.2413, 'chop_score': 0.463, 'volatility_percentile': 0.3837, 'consolidation_score': 0.1793}
2026-05-11 02:53:20,686 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:20,688 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:20,689 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:20,690 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:20,690 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:20,691 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:53:20,941 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected USDJPY — 5046 samples (group=dollar) score_means={'trend_score': 0.5085, 'range_score': 0.2259, 'chop_score': 0.4506, 'volatility_percentile': 0.3809, 'consolidation_score': 0.1888}
2026-05-11 02:53:21,057 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:21,060 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:21,060 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:21,061 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:21,061 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:21,063 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:53:21,310 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURJPY — 5053 samples (group=cross) score_means={'trend_score': 0.5023, 'range_score': 0.2261, 'chop_score': 0.4524, 'volatility_percentile': 0.3837, 'consolidation_score': 0.18}
2026-05-11 02:53:21,433 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:21,435 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:21,436 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:21,436 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:21,437 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:21,438 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:53:21,713 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPJPY — 5043 samples (group=cross) score_means={'trend_score': 0.4743, 'range_score': 0.2372, 'chop_score': 0.4716, 'volatility_percentile': 0.3777, 'consolidation_score': 0.1917}
2026-05-11 02:53:21,831 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:21,834 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:21,834 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:21,835 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:21,835 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 02:53:21,837 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 02:53:22,084 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPUSD — 5040 samples (group=dollar) score_means={'trend_score': 0.4843, 'range_score': 0.2371, 'chop_score': 0.4614, 'volatility_percentile': 0.377, 'consolidation_score': 0.1835}
2026-05-11 02:53:22,201 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2317, 'chop_score': 0.462, 'volatility_percentile': 0.3807, 'consolidation_score': 0.1858}, 'dollar': {'trend_score': 0.4937, 'range_score': 0.2348, 'chop_score': 0.4583, 'volatility_percentile': 0.3806, 'consolidation_score': 0.1839}, 'gold': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}}
2026-05-11 02:53:22,201 INFO Regime[1H mode=ltf_behaviour] score means by year: {2022: {'trend_score': 0.4905, 'range_score': 0.2364, 'chop_score': 0.4625, 'volatility_percentile': 0.385, 'consolidation_score': 0.1797}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 02:53:22,307 INFO Regime phase LTF dataset build fold=train_all: 8.5s (train=262644 val=30352)
2026-05-11 02:53:22,308 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260511_025322
2026-05-11 02:53:22,313 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=31, n_classes=5)
2026-05-11 02:53:22,313 INFO Regime 1H/ltf_behaviour warm start enabled from existing weights
2026-05-11 02:53:22,337 INFO RegimeClassifier[mode=ltf_behaviour]: warm start score head
2026-05-11 02:53:22,337 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-11 02:53:22,883 INFO Regime score epoch  1/50 — tr=0.0039 va=0.0010 mae={'trend_score': 0.0206, 'range_score': 0.0357, 'chop_score': 0.0232, 'volatility_percentile': 0.0163, 'consolidation_score': 0.0213}
2026-05-11 02:53:23,383 INFO Regime score epoch  2/50 — tr=0.0039 va=0.0010
2026-05-11 02:53:23,890 INFO Regime score epoch  3/50 — tr=0.0039 va=0.0010
2026-05-11 02:53:24,397 INFO Regime score epoch  4/50 — tr=0.0039 va=0.0010
2026-05-11 02:53:24,903 INFO Regime score epoch  5/50 — tr=0.0039 va=0.0010 mae={'trend_score': 0.0207, 'range_score': 0.0353, 'chop_score': 0.0232, 'volatility_percentile': 0.0164, 'consolidation_score': 0.0213}
2026-05-11 02:53:25,406 INFO Regime score epoch  6/50 — tr=0.0039 va=0.0010
2026-05-11 02:53:25,921 INFO Regime score epoch  7/50 — tr=0.0038 va=0.0010
2026-05-11 02:53:26,436 INFO Regime score epoch  8/50 — tr=0.0038 va=0.0010
2026-05-11 02:53:26,935 INFO Regime score epoch  9/50 — tr=0.0038 va=0.0010
2026-05-11 02:53:27,439 INFO Regime score epoch 10/50 — tr=0.0038 va=0.0010 mae={'trend_score': 0.0195, 'range_score': 0.0349, 'chop_score': 0.0225, 'volatility_percentile': 0.0163, 'consolidation_score': 0.0222}
2026-05-11 02:53:27,934 INFO Regime score epoch 11/50 — tr=0.0038 va=0.0010
2026-05-11 02:53:28,423 INFO Regime score epoch 12/50 — tr=0.0037 va=0.0010
2026-05-11 02:53:28,925 INFO Regime score epoch 13/50 — tr=0.0037 va=0.0009
2026-05-11 02:53:29,435 INFO Regime score epoch 14/50 — tr=0.0037 va=0.0009
2026-05-11 02:53:29,951 INFO Regime score epoch 15/50 — tr=0.0037 va=0.0009 mae={'trend_score': 0.0187, 'range_score': 0.0341, 'chop_score': 0.0215, 'volatility_percentile': 0.0155, 'consolidation_score': 0.0206}
2026-05-11 02:53:30,458 INFO Regime score epoch 16/50 — tr=0.0036 va=0.0009
2026-05-11 02:53:30,963 INFO Regime score epoch 17/50 — tr=0.0036 va=0.0009
2026-05-11 02:53:31,469 INFO Regime score epoch 18/50 — tr=0.0036 va=0.0009
2026-05-11 02:53:31,998 INFO Regime score epoch 19/50 — tr=0.0036 va=0.0009
2026-05-11 02:53:32,505 INFO Regime score epoch 20/50 — tr=0.0036 va=0.0009 mae={'trend_score': 0.018, 'range_score': 0.0332, 'chop_score': 0.0212, 'volatility_percentile': 0.0153, 'consolidation_score': 0.0204}
2026-05-11 02:53:33,013 INFO Regime score epoch 21/50 — tr=0.0036 va=0.0009
2026-05-11 02:53:33,502 INFO Regime score epoch 22/50 — tr=0.0035 va=0.0009
2026-05-11 02:53:34,010 INFO Regime score epoch 23/50 — tr=0.0035 va=0.0009
2026-05-11 02:53:34,513 INFO Regime score epoch 24/50 — tr=0.0035 va=0.0009
2026-05-11 02:53:35,026 INFO Regime score epoch 25/50 — tr=0.0035 va=0.0009 mae={'trend_score': 0.0175, 'range_score': 0.0331, 'chop_score': 0.0204, 'volatility_percentile': 0.0147, 'consolidation_score': 0.0197}
2026-05-11 02:53:35,549 INFO Regime score epoch 26/50 — tr=0.0035 va=0.0008
2026-05-11 02:53:36,118 INFO Regime score epoch 27/50 — tr=0.0035 va=0.0009
2026-05-11 02:53:36,701 INFO Regime score epoch 28/50 — tr=0.0035 va=0.0009
2026-05-11 02:53:37,203 INFO Regime score epoch 29/50 — tr=0.0035 va=0.0009
2026-05-11 02:53:37,696 INFO Regime score epoch 30/50 — tr=0.0035 va=0.0008 mae={'trend_score': 0.0173, 'range_score': 0.0329, 'chop_score': 0.0203, 'volatility_percentile': 0.0145, 'consolidation_score': 0.0194}
2026-05-11 02:53:38,196 INFO Regime score epoch 31/50 — tr=0.0035 va=0.0008
2026-05-11 02:53:38,713 INFO Regime score epoch 32/50 — tr=0.0035 va=0.0008
2026-05-11 02:53:39,221 INFO Regime score epoch 33/50 — tr=0.0034 va=0.0008
2026-05-11 02:53:39,736 INFO Regime score epoch 34/50 — tr=0.0034 va=0.0008
2026-05-11 02:53:40,226 INFO Regime score epoch 35/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.017, 'range_score': 0.0325, 'chop_score': 0.0201, 'volatility_percentile': 0.0151, 'consolidation_score': 0.0197}
2026-05-11 02:53:40,727 INFO Regime score epoch 36/50 — tr=0.0034 va=0.0008
2026-05-11 02:53:41,247 INFO Regime score epoch 37/50 — tr=0.0034 va=0.0008
2026-05-11 02:53:41,773 INFO Regime score epoch 38/50 — tr=0.0034 va=0.0008
2026-05-11 02:53:42,275 INFO Regime score epoch 39/50 — tr=0.0034 va=0.0008
2026-05-11 02:53:42,781 INFO Regime score epoch 40/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.017, 'range_score': 0.0323, 'chop_score': 0.0201, 'volatility_percentile': 0.0144, 'consolidation_score': 0.019}
2026-05-11 02:53:43,291 INFO Regime score epoch 41/50 — tr=0.0034 va=0.0008
2026-05-11 02:53:43,787 INFO Regime score epoch 42/50 — tr=0.0034 va=0.0008
2026-05-11 02:53:44,298 INFO Regime score epoch 43/50 — tr=0.0034 va=0.0008
2026-05-11 02:53:44,799 INFO Regime score epoch 44/50 — tr=0.0034 va=0.0008
2026-05-11 02:53:45,314 INFO Regime score epoch 45/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.017, 'range_score': 0.0323, 'chop_score': 0.0199, 'volatility_percentile': 0.0145, 'consolidation_score': 0.0192}
2026-05-11 02:53:45,823 INFO Regime score epoch 46/50 — tr=0.0034 va=0.0008
2026-05-11 02:53:46,321 INFO Regime score epoch 47/50 — tr=0.0034 va=0.0008
2026-05-11 02:53:46,840 INFO Regime score epoch 48/50 — tr=0.0034 va=0.0008
2026-05-11 02:53:47,327 INFO Regime score epoch 49/50 — tr=0.0034 va=0.0008
2026-05-11 02:53:47,811 INFO Regime score epoch 50/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.017, 'range_score': 0.0322, 'chop_score': 0.0197, 'volatility_percentile': 0.0148, 'consolidation_score': 0.0197}
2026-05-11 02:53:47,831 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0169, 'range_score': 0.0322, 'chop_score': 0.0197, 'volatility_percentile': 0.0146, 'consolidation_score': 0.0192} mse={'trend_score': 0.00049, 'range_score': 0.00173, 'chop_score': 0.00063, 'volatility_percentile': 0.00038, 'consolidation_score': 0.00083} corr={'trend_score': 0.995, 'range_score': 0.9576, 'chop_score': 0.9915, 'volatility_percentile': 0.9961, 'consolidation_score': 0.9912} pred_std={'trend_score': 0.2209, 'range_score': 0.1325, 'chop_score': 0.1818, 'volatility_percentile': 0.2194, 'consolidation_score': 0.215} target_std={'trend_score': 0.2219, 'range_score': 0.1433, 'chop_score': 0.1888, 'volatility_percentile': 0.2165, 'consolidation_score': 0.2179}
2026-05-11 02:53:48,146 INFO RegimeClassifier[mode=ltf_behaviour] train group diagnostics={'dollar': {'n': 16632, 'score_mae': {'trend_score': 0.0164, 'range_score': 0.0321, 'chop_score': 0.0196, 'volatility_percentile': 0.0142, 'consolidation_score': 0.0196}, 'target_score_mean': {'trend_score': 0.491, 'range_score': 0.2347, 'chop_score': 0.4613, 'volatility_percentile': 0.3824, 'consolidation_score': 0.186}, 'pred_score_mean': {'trend_score': 0.4902, 'range_score': 0.2352, 'chop_score': 0.4616, 'volatility_percentile': 0.3799, 'consolidation_score': 0.1844}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3565, 58, 0, 2, 0, 0, 154], [4, 99, 0, 0, 0, 3, 4], [0, 0, 185, 10, 59, 0, 206], [2, 0, 7, 570, 33, 0, 77], [0, 0, 36, 18, 3065, 0, 197], [0, 21, 0, 0, 8, 55, 44], [126, 8, 81, 52, 70, 4, 7809]]}, 'gold': {'n': 8420, 'score_mae': {'trend_score': 0.0163, 'range_score': 0.0328, 'chop_score': 0.0199, 'volatility_percentile': 0.015, 'consolidation_score': 0.0199}, 'target_score_mean': {'trend_score': 0.4883, 'range_score': 0.2355, 'chop_score': 0.4652, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}, 'pred_score_mean': {'trend_score': 0.4881, 'range_score': 0.236, 'chop_score': 0.4649, 'volatility_percentile': 0.3739, 'consolidation_score': 0.1898}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1785, 34, 0, 0, 0, 0, 66], [4, 50, 0, 0, 0, 0, 2], [0, 0, 95, 11, 27, 0, 111], [0, 0, 2, 349, 20, 0, 45], [0, 0, 18, 22, 1568, 0, 96], [0, 14, 0, 0, 6, 38, 23], [53, 3, 53, 20, 53, 0, 3852]]}, 'jpy': {'n': 24948, 'score_mae': {'trend_score': 0.0164, 'range_score': 0.0322, 'chop_score': 0.0195, 'volatility_percentile': 0.015, 'consolidation_score': 0.0195}, 'target_score_mean': {'trend_score': 0.4904, 'range_score': 0.2338, 'chop_score': 0.4653, 'volatility_percentile': 0.3826, 'consolidation_score': 0.1905}, 'pred_score_mean': {'trend_score': 0.4893, 'range_score': 0.2344, 'chop_score': 0.4649, 'volatility_percentile': 0.3798, 'consolidation_score': 0.1881}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[5331, 113, 0, 3, 0, 0, 199], [7, 171, 0, 0, 0, 1, 8], [0, 0, 240, 19, 87, 0, 301], [2, 0, 3, 1098, 83, 0, 128], [0, 0, 38, 54, 4695, 0, 328], [0, 33, 0, 0, 17, 82, 91], [174, 11, 122, 89, 144, 1, 11275]]}}
2026-05-11 02:53:48,318 INFO RegimeClassifier[mode=ltf_behaviour] validation group diagnostics={'dollar': {'n': 10093, 'score_mae': {'trend_score': 0.0171, 'range_score': 0.0329, 'chop_score': 0.0199, 'volatility_percentile': 0.0142, 'consolidation_score': 0.0188}, 'target_score_mean': {'trend_score': 0.4863, 'range_score': 0.2392, 'chop_score': 0.4622, 'volatility_percentile': 0.3804, 'consolidation_score': 0.1814}, 'pred_score_mean': {'trend_score': 0.4857, 'range_score': 0.2378, 'chop_score': 0.4633, 'volatility_percentile': 0.3778, 'consolidation_score': 0.1804}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[2295, 24, 0, 1, 0, 0, 92], [3, 46, 0, 0, 0, 3, 1], [0, 0, 110, 7, 50, 0, 149], [0, 0, 3, 335, 32, 0, 53], [0, 0, 24, 25, 1905, 0, 96], [0, 11, 0, 0, 4, 33, 29], [58, 5, 51, 45, 48, 3, 4552]]}, 'gold': {'n': 5117, 'score_mae': {'trend_score': 0.0161, 'range_score': 0.0314, 'chop_score': 0.0196, 'volatility_percentile': 0.0148, 'consolidation_score': 0.0197}, 'target_score_mean': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}, 'pred_score_mean': {'trend_score': 0.4958, 'range_score': 0.2331, 'chop_score': 0.4573, 'volatility_percentile': 0.3788, 'consolidation_score': 0.1811}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1103, 14, 0, 0, 0, 0, 50], [3, 30, 0, 0, 0, 1, 1], [0, 0, 67, 3, 14, 0, 87], [0, 0, 3, 226, 10, 0, 16], [0, 0, 11, 10, 797, 0, 69], [0, 6, 0, 0, 4, 19, 21], [38, 3, 37, 26, 30, 0, 2418]]}, 'jpy': {'n': 15142, 'score_mae': {'trend_score': 0.017, 'range_score': 0.0321, 'chop_score': 0.0197, 'volatility_percentile': 0.0149, 'consolidation_score': 0.0193}, 'target_score_mean': {'trend_score': 0.495, 'range_score': 0.2297, 'chop_score': 0.4582, 'volatility_percentile': 0.3808, 'consolidation_score': 0.1868}, 'pred_score_mean': {'trend_score': 0.493, 'range_score': 0.2297, 'chop_score': 0.4591, 'volatility_percentile': 0.3782, 'consolidation_score': 0.1848}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3339, 53, 0, 1, 0, 0, 125], [5, 102, 0, 0, 0, 3, 5], [0, 0, 146, 14, 53, 0, 171], [2, 0, 4, 699, 41, 0, 81], [0, 0, 26, 31, 2576, 0, 184], [0, 19, 0, 0, 10, 45, 48], [94, 11, 81, 46, 89, 3, 7035]]}}
2026-05-11 02:53:48,324 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-11 02:53:48,324 INFO Regime phase LTF train fold=train_all: 26.0s
2026-05-11 02:53:48,442 INFO Regime LTF complete fold=train_all: score_accuracy=0.980, train=262644 val=30352 mae={'trend_score': 0.0169, 'range_score': 0.0322, 'chop_score': 0.0197, 'volatility_percentile': 0.0146, 'consolidation_score': 0.0192}
2026-05-11 02:53:48,445 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 02:53:48,835 INFO Regime[1H mode=ltf_behaviour fold=latest] LTF score diagnostics on GBPUSD:
{'trend_score': {'mean': 0.4921, 'q10': 0.1918, 'q50': 0.4877, 'q90': 0.8002}, 'range_score': {'mean': 0.2337, 'q10': 0.0513, 'q50': 0.2125, 'q90': 0.4349}, 'chop_score': {'mean': 0.4612, 'q10': 0.2148, 'q50': 0.4503, 'q90': 0.7255}, 'volatility_percentile': {'mean': 0.3832, 'q10': 0.1018, 'q50': 0.3704, 'q90': 0.6823}, 'consolidation_score': {'mean': 0.1854, 'q10': 0.0, 'q50': 0.1185, 'q90': 0.506}}
2026-05-11 02:53:48,845 INFO Regime retrain total: 75.2s (370559 train+val samples)
2026-05-11 02:53:48,854 INFO Retrain complete. Total wall-clock: 75.2s
2026-05-11 02:53:49,931 INFO Model regime: SUCCESS
2026-05-11 02:53:49,931 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 02:53:49,931 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 02:53:49,932 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-11 02:53:49,932 INFO   [OK] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-05-11 02:53:49,932 INFO   [DORMANT] rl_ppo — RL_ENABLED=false, skipping
2026-05-11 02:53:49,932 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-05-11 02:53:49,952 INFO Saved 70 retrain records to metrics/

=== TRAINING COMPLETE ===
  gru: SUCCESS
  regime: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Clean Quality/RL source: Backtest on train window ===
  START Train-window backtest for Quality/RL labels
2026-05-11 02:53:50,605 INFO === STEP 6: BACKTEST (train) ===
2026-05-11 02:53:50,607 INFO BT_WINDOW=train — train-window backtest: 2016-01-04 → 2023-08-04 (clean Quality/RL labels)
2026-05-11 02:53:50,607 INFO Cleared existing journal for fresh train run
2026-05-11 02:53:50,607 INFO ================================================================
  ROUND 0 / 3
================================================================
2026-05-11 02:53:50,608 INFO Round 0 — running backtest: 2016-01-04 → 2023-08-04 (ml_trader, shared ML cache)
2026-05-11 02:57:49,082 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for EURJPY with 2
2026-05-11 02:57:49,099 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for EURJPY with 0.3333333333333333
2026-05-11 02:57:49,228 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for EURUSD with 2
2026-05-11 02:57:49,254 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for EURUSD with 0.3333333333333333
2026-05-11 02:57:49,573 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for EURJPY with 2
2026-05-11 02:57:49,599 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for EURJPY with 0.25
2026-05-11 02:57:49,631 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for EURJPY: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 02:57:49,663 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for USDJPY with 2
2026-05-11 02:57:49,782 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for USDJPY with 0.3333333333333333
2026-05-11 02:57:49,948 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for EURUSD with 2
2026-05-11 02:57:49,963 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for EURUSD with 0.25
2026-05-11 02:57:50,000 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for EURUSD: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 02:57:50,526 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for USDJPY with 2
2026-05-11 02:57:50,549 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for USDJPY with 0.25
2026-05-11 02:57:50,576 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for USDJPY: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 02:57:51,003 WARNING ML cache score overlay filled 4 warmup/alignment gaps for EURJPY
2026-05-11 02:57:52,682 WARNING ML cache score overlay filled 4 warmup/alignment gaps for EURUSD
2026-05-11 02:57:59,190 WARNING ML cache score overlay filled 4 warmup/alignment gaps for USDJPY
2026-05-11 02:58:04,497 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 02:58:05,633 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 02:58:06,883 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 02:58:07,857 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 02:58:08,308 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 02:58:08,756 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 02:58:09,168 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 02:58:09,376 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 02:58:09,728 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 02:58:09,966 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 02:58:10,125 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
2026-05-11 02:58:10,181 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 02:58:10,256 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 02:58:10,487 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 02:58:10,537 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 02:58:10,562 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 02:58:10,579 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 02:58:10,612 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 02:58:10,643 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 02:58:10,671 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 02:58:10,694 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 02:58:10,741 WARNING _build_sequence_df: HTF frame 5M filled 357 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 02:58:10,777 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 02:58:10,825 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 02:58:10,875 WARNING _build_sequence_df: HTF frame 5M filled 357 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 02:58:10,960 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 02:58:11,035 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 02:58:11,112 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
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
2026-05-11 02:58:11,281 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 02:58:11,346 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 02:58:11,423 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 02:58:11,507 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 02:58:11,576 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 02:58:11,654 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 02:58:12,066 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 02:58:27,625 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for GBPUSD with 2
2026-05-11 02:58:27,638 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for GBPUSD with 0.3333333333333333
2026-05-11 02:58:27,782 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf for GBPJPY with 2
2026-05-11 02:58:27,795 WARNING ML cache alignment filled 16 warmup/alignment gaps in regime_htf_conf for GBPJPY with 0.3333333333333333
2026-05-11 02:58:27,982 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for GBPUSD with 2
2026-05-11 02:58:28,004 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for GBPUSD with 0.25
2026-05-11 02:58:28,032 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for GBPUSD: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 02:58:28,231 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf for GBPJPY with 2
2026-05-11 02:58:28,245 WARNING ML cache alignment filled 4 warmup/alignment gaps in regime_ltf_conf for GBPJPY with 0.25
2026-05-11 02:58:28,261 WARNING ML cache alignment LTF score frame filled 4 warmup/alignment gaps for GBPJPY: columns=['trend_score', 'range_score', 'chop_score', 'volatility_percentile', 'consolidation_score']
2026-05-11 02:58:28,700 WARNING ML cache score overlay filled 4 warmup/alignment gaps for GBPUSD
2026-05-11 02:58:29,735 WARNING ML cache score overlay filled 4 warmup/alignment gaps for GBPJPY
2026-05-11 02:58:37,715 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 02:58:37,777 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 02:58:37,798 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 02:58:37,820 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 02:58:37,842 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 02:58:37,865 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 02:58:37,881 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 02:58:37,904 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 02:58:37,933 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 02:58:37,939 WARNING _build_sequence_df: HTF frame 5M filled 1 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 02:58:37,959 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 02:58:37,984 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 02:58:38,001 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 02:58:38,024 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 02:58:38,057 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
2026-05-11 02:58:38,059 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 02:58:38,077 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 02:58:38,108 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-11 02:58:38,131 WARNING _build_sequence_df: HTF frame 4H filled 16 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 02:58:38,150 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-11 02:58:38,171 WARNING _build_sequence_df: HTF frame 1D filled 96 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 02:58:38,229 WARNING _build_sequence_df: HTF frame 1H filled 4 warmup/alignment gaps with 0.000

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260511_025352.json
Trader                                   Trades      WR    PF*  Return*    ExpR   TP1%   TP2%     DD*  Sharpe*  Sortino     IC  Verdict
-----------------------------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)               43  20.9%   0.85   -4.9%  -0.115 20.9% 11.6%  11.5%    -1.01    -0.11 -0.301     FAIL
  FAILED rules: positive_expectancy, profit_factor_min_1_25, sharpe_positive, sortino_positive, win_rate_above_breakeven, t_stat_above_1_5, sharpe_ci_positive
  monthly R: 2022-04=-0.10  2022-08=+3.36  2022-11=+1.32  2023-01=-2.00  2023-04=-2.00  2023-06=+0.82
  MonteCarlo P95 DD=17.9%  P10 equity=9,506  t=-0.42 (p=0.678)  Sharpe CI=[-8.52, 3.97]  streak=12
  gate_diagnostics: bars=1049680 no_signal=487203 quality_block=0 session_skip=562433 density=1 pm_reject=0 daily_skip=0 cooldown=0 daily_halt_events=0 enforce_daily_halt=False
  no_signal_reasons: gru_expected_r_below_threshold=197326, no_trade_uncertain=114438, weak_gru_direction=107435, no_trade_extreme_vol=32620, no_trade_chop=28883, wait_pullback=3990

Calibration Summary:
  all          [OK] Too few populated bins for calibration check
  ml_trader    [OK] Too few populated bins for calibration check
2026-05-11 03:01:01,226 INFO Round 0 backtest — 43 trades | avg WR=20.9% | avg PF=0.85 | avg Sharpe=-1.01
2026-05-11 03:01:01,226 INFO   ml_trader: 43 trades | WR=20.9% | fixed PF=0.85 | Return=-4.9% | ExpR=-0.115 | DD=11.5% | Sharpe=-1.01
2026-05-11 03:01:01,226 INFO   ml_trader gate_diagnostics: bars=1049680 no_signal=487203 quality_block=0 session_skip=562433 density=1 pm_reject=0
2026-05-11 03:01:01,226 INFO   ml_trader no_signal_reasons: {'no_trade_uncertain': 114438, 'weak_gru_direction': 107435, 'gru_expected_r_below_threshold': 197326, 'no_trade_extreme_vol': 32620, 'trend_structure_missing': 1101, 'wait_pullback': 3990, 'tradeability_direction_conflict': 1410, 'no_trade_chop': 28883}
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_0.json
INFO  Total trades: 43
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (43 rows)
2026-05-11 03:01:01,475 INFO Round 0: wrote 43 journal entries (total in file: 43)
  DONE  Train-window backtest for Quality/RL labels
  Saved Train Quality/RL source result → train_quality_rl_source_summary.json
  Train-label journal entries: 43

=== Train Quality + RL on train-only journal ===
  START Train-only Quality+RL retrain
  DONE  Train-only Quality+RL retrain
  Archived journal → trade_journal_train_only.jsonl
  Archived journal CSV → trade_journal_train_only.csv

=== Round 1: Backtest on train-tail window (latest 2yr inside training data) ===
  Cleared journal for fresh Round 1 run
  START Round 1 - Backtest (train-tail)
2026-05-11 03:01:01,743 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-05-11 03:01:01,745 INFO Journal entries: 43 total, 43 allowed for training (['live', 'paper', 'production', 'train'])
2026-05-11 03:01:01,745 WARNING Journal has only 43 allowed entries (need 50) — not enough clean Quality/RL training data. Check step6 logs or collect live/paper data.
2026-05-11 03:01:02,268 INFO === STEP 6: BACKTEST (round1) ===
2026-05-11 03:01:02,269 INFO BT_WINDOW=round1 — train-tail backtest: 2021-08-05 → 2023-08-04 (seen training data; test set protected)
2026-05-11 03:01:02,269 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-05-11 03:01:02,270 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
2026-05-11 03:01:02,270 INFO Round 1 — QualityScorer hard gate disabled; set BACKTEST_ENABLE_QUALITY_GATE=1 to enable it
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
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
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:02:17,746 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 03:02:18,078 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 03:02:18,369 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 03:02:18,394 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 03:02:18,609 WARNING _build_sequence_df: HTF frame 5M filled 461 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 03:02:18,652 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 03:02:18,670 WARNING _build_sequence_df: HTF frame 5M filled 461 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
2026-05-11 03:02:18,720 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
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
2026-05-11 03:02:27,638 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 03:02:27,663 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 03:02:27,704 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 03:02:27,753 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
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
2026-05-11 03:03:04,543 INFO Round 1 backtest — 18 trades | avg WR=22.2% | avg PF=0.74 | avg Sharpe=-1.96
2026-05-11 03:03:04,543 INFO   ml_trader: 18 trades | WR=22.2% | fixed PF=0.74 | Return=-3.6% | ExpR=-0.200 | DD=7.0% | Sharpe=-1.96
2026-05-11 03:03:04,544 INFO   ml_trader gate_diagnostics: bars=263960 no_signal=118085 quality_block=0 session_skip=145857 density=0 pm_reject=0
2026-05-11 03:03:04,544 INFO   ml_trader no_signal_reasons: {'no_trade_uncertain': 27063, 'gru_expected_r_below_threshold': 47678, 'weak_gru_direction': 25488, 'no_trade_chop': 7209, 'no_trade_extreme_vol': 8844, 'tradeability_direction_conflict': 384, 'wait_pullback': 1118, 'trend_structure_missing': 301}
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 18
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (18 rows)
2026-05-11 03:03:04,775 INFO Round 1: wrote 18 journal entries (total in file: 18)
  DONE  Round 1 - Backtest (train-tail)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 18 entries

  SKIP  Round 1 Quality+RL retrain — train-tail journal kept evaluation-only

=== Round 2: BLIND backtest on test window (unseen 2yr) ===
  START Round 2 - Blind backtest (test)
2026-05-11 03:03:05,417 INFO === STEP 6: BACKTEST (round2) ===
2026-05-11 03:03:05,418 INFO BT_WINDOW=round2 — BLIND backtest: 2023-08-07 → 2025-08-05 (test set)
2026-05-11 03:03:05,418 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-05-11 03:03:05,418 INFO Round 2 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
2026-05-11 03:03:05,419 INFO Round 2 — QualityScorer hard gate disabled; set BACKTEST_ENABLE_QUALITY_GATE=1 to enable it
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
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
2026-05-11 03:04:25,161 WARNING _build_sequence_df: HTF frame 5M filled 317 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 03:04:25,602 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
2026-05-11 03:04:25,609 WARNING _build_sequence_df: HTF frame 5M filled 317 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 03:04:25,648 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 03:04:25,699 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 03:04:25,733 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 03:04:25,956 WARNING _build_sequence_df: HTF frame 5M filled 341 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 03:04:26,005 WARNING _build_sequence_df: HTF frame 5M filled 341 warmup/alignment gaps with 0.000
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
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:04:34,677 WARNING _build_sequence_df: HTF frame 5M filled 317 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 03:04:34,699 WARNING _build_sequence_df: HTF frame 5M filled 317 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 03:04:34,730 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-11 03:04:34,787 WARNING _build_sequence_df: HTF frame 5M filled 321 warmup/alignment gaps with 0.000
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
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:774: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 03:05:13,865 INFO Round 2 backtest — 7 trades | avg WR=28.6% | avg PF=1.51 | avg Sharpe=2.49
2026-05-11 03:05:13,865 INFO   ml_trader: 7 trades | WR=28.6% | fixed PF=1.51 | Return=2.6% | ExpR=0.367 | DD=3.8% | Sharpe=2.49
2026-05-11 03:05:13,865 INFO   ml_trader gate_diagnostics: bars=280782 no_signal=131653 quality_block=0 session_skip=149122 density=0 pm_reject=0
2026-05-11 03:05:13,865 INFO   ml_trader no_signal_reasons: {'weak_gru_direction': 28689, 'no_trade_uncertain': 32220, 'gru_expected_r_below_threshold': 51807, 'no_trade_chop': 7977, 'tradeability_direction_conflict': 488, 'trend_structure_missing': 404, 'wait_pullback': 1460, 'no_trade_extreme_vol': 8608}
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 7
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (7 rows)
2026-05-11 03:05:14,089 INFO Round 2: wrote 7 journal entries (total in file: 25)
  DONE  Round 2 - Blind backtest (test)
  Saved Round 2 result → round2_summary.json
  Journal after Round 2: 25 entries

  SKIP  Round 2 Quality+RL retrain — blind test journal kept untouched

=== Round 3: Incremental retrain ===
  START Retrain gru [train-split retrain]
2026-05-11 03:05:14,412 INFO retrain environment: KAGGLE
2026-05-11 03:05:16,007 INFO Device: CUDA (2 GPU(s))
2026-05-11 03:05:16,019 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 03:05:16,019 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 03:05:16,019 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 03:05:16,020 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 03:05:16,020 INFO Retrain data split: train
2026-05-11 03:05:16,020 INFO Retrain rolling fold selector: latest
2026-05-11 03:05:16,021 INFO === GRU-LSTM retrain (timeframes: ['15M']) ===
2026-05-11 03:05:16,163 INFO NumExpr defaulting to 4 threads.
2026-05-11 03:05:16,353 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-11 03:05:16,354 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 03:05:16,354 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 03:05:17,226 INFO GRULSTMPredictor: loaded long R isotonic calibrator from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/r_isotonic_long.pkl
2026-05-11 03:05:17,226 INFO GRULSTMPredictor: loaded short R isotonic calibrator from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/r_isotonic_short.pkl
2026-05-11 03:05:17,226 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-05-11 03:05:17,226 INFO GRU multi-symbol training (Kaggle mode): 6 symbols × ['15M']
2026-05-11 03:05:17,228 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260511_030517
2026-05-11 03:05:17,233 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-05-11 03:05:17,233 INFO GRU warm start enabled from existing weights: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 03:05:17,494 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 03:05:17,522 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 03:05:17,542 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 03:05:17,553 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 03:05:17,629 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 03:05:17,635 INFO Loaded XAUUSD/15M split=train fold=latest: 176438 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:05:18,203 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:05:18,221 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:05:18,235 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:05:18,242 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:05:18,279 INFO Loaded EURUSD/15M split=train fold=latest: 174951 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:05:18,844 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:05:18,864 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:05:18,880 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:05:18,887 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:05:18,926 INFO Loaded USDJPY/15M split=train fold=latest: 174944 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:05:19,498 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:05:19,521 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:05:19,536 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:05:19,544 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:05:19,582 INFO Loaded EURJPY/15M split=train fold=latest: 174948 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:05:20,126 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:05:20,147 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:05:20,162 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:05:20,169 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:05:20,208 INFO Loaded GBPJPY/15M split=train fold=latest: 174902 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:05:20,750 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:05:20,770 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:05:20,784 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:05:20,792 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:05:20,830 INFO Loaded GBPUSD/15M split=train fold=latest: 174899 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:05:21,293 INFO train_multi: 6 segments, ~971854 total bars
2026-05-11 03:05:21,293 INFO train_multi: training ALL 6 segments across TFs ['15M'] in one combined pass
2026-05-11 03:05:21,293 INFO train_multi: building combined dataset for TF=ALL (6 segments)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:05:30,671 INFO train_multi TF=ALL: 971494 sequences across 6 segments
2026-05-11 03:05:30,671 INFO train_multi TF=ALL: estimated peak RAM = 21312 MB (train=419996 calib=60000 val=120002 n_feat=74 seq_len=60)
2026-05-11 03:05:30,672 WARNING train_multi TF=ALL: trimming to fit RAM budget — new train=394144 calib=56306 val=112612 (20000 MB est)
2026-05-11 03:05:32,998 INFO train_multi TF=ALL: train=394144 calib=56306 val=112612 (10009 MB tensors)
2026-05-11 03:05:39,665 INFO train_multi TF=ALL: structural bar weighting — 252452 structural bars (64.1%) weight=15.0 structural_only=0
2026-05-11 03:05:40,741 INFO train_multi TF=ALL: warm-start detected — using CosineAnnealingLR (lr=3e-05, patience=12)
2026-05-11 03:05:58,644 INFO train_multi TF=ALL epoch 1/100 train=1.7252 val=1.8543 r_mae=0.771 pos_r_acc=0.688 side_acc=0.740 r_n=161888
2026-05-11 03:05:58,649 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-11 03:05:58,649 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-11 03:05:58,649 INFO train_multi TF=ALL: new best val=1.8543 — saved
2026-05-11 03:06:14,437 INFO train_multi TF=ALL epoch 2/100 train=1.7239 val=1.8605 r_mae=0.772 pos_r_acc=0.687 side_acc=0.738 r_n=161888
2026-05-11 03:06:30,217 INFO train_multi TF=ALL epoch 3/100 train=1.7240 val=1.8585 r_mae=0.771 pos_r_acc=0.688 side_acc=0.739 r_n=161888
2026-05-11 03:06:45,957 INFO train_multi TF=ALL epoch 4/100 train=1.7183 val=1.8584 r_mae=0.771 pos_r_acc=0.688 side_acc=0.738 r_n=161888
2026-05-11 03:07:01,702 INFO train_multi TF=ALL epoch 5/100 train=1.7191 val=1.8589 r_mae=0.772 pos_r_acc=0.688 side_acc=0.738 r_n=161888
2026-05-11 03:07:17,375 INFO train_multi TF=ALL epoch 6/100 train=1.7147 val=1.8611 r_mae=0.773 pos_r_acc=0.687 side_acc=0.738 r_n=161888
2026-05-11 03:07:33,180 INFO train_multi TF=ALL epoch 7/100 train=1.7169 val=1.8625 r_mae=0.771 pos_r_acc=0.688 side_acc=0.738 r_n=161888
2026-05-11 03:07:48,834 INFO train_multi TF=ALL epoch 8/100 train=1.7154 val=1.8655 r_mae=0.771 pos_r_acc=0.687 side_acc=0.737 r_n=161888
2026-05-11 03:08:04,632 INFO train_multi TF=ALL epoch 9/100 train=1.7167 val=1.8599 r_mae=0.771 pos_r_acc=0.688 side_acc=0.738 r_n=161888
2026-05-11 03:08:20,338 INFO train_multi TF=ALL epoch 10/100 train=1.7138 val=1.8636 r_mae=0.770 pos_r_acc=0.688 side_acc=0.738 r_n=161888
2026-05-11 03:08:36,192 INFO train_multi TF=ALL epoch 11/100 train=1.7129 val=1.8634 r_mae=0.772 pos_r_acc=0.687 side_acc=0.739 r_n=161888
2026-05-11 03:08:51,891 INFO train_multi TF=ALL epoch 12/100 train=1.7127 val=1.8636 r_mae=0.773 pos_r_acc=0.686 side_acc=0.738 r_n=161888
2026-05-11 03:09:07,728 INFO train_multi TF=ALL epoch 13/100 train=1.7091 val=1.8605 r_mae=0.772 pos_r_acc=0.688 side_acc=0.739 r_n=161888
2026-05-11 03:09:07,728 INFO train_multi TF=ALL early stop at epoch 13
2026-05-11 03:09:08,135 INFO fit_r_isotonic: saved side-R calibrators stats={'long': {'n': 40072, 'raw_mae': 0.7364308495500853, 'calibrated_mae': 0.7195170021059585}, 'short': {'n': 41197, 'raw_mae': 0.7497485767548983, 'calibrated_mae': 0.7263203352525166}}
2026-05-11 03:09:08,276 WARNING GRU validation R-MAE above floor for TF=ALL: best_val_r_mae=0.771 max=0.750. Keeping saved best weights so the pipeline can progress.
2026-05-11 03:09:08,290 INFO GRU R threshold XAUUSD/buy: q25_pos=0.535 q50_pos=1.667 pos_rate=46.0% (n=119259 n_pos=54899)
2026-05-11 03:09:08,296 INFO GRU R threshold XAUUSD/sell: q25_pos=0.528 q50_pos=1.667 pos_rate=45.5% (n=118087 n_pos=53734)
2026-05-11 03:09:08,302 INFO GRU R threshold EURUSD/buy: q25_pos=0.541 q50_pos=1.667 pos_rate=45.6% (n=118751 n_pos=54196)
2026-05-11 03:09:08,308 INFO GRU R threshold EURUSD/sell: q25_pos=0.545 q50_pos=1.667 pos_rate=45.6% (n=117679 n_pos=53721)
2026-05-11 03:09:08,314 INFO GRU R threshold USDJPY/buy: q25_pos=0.551 q50_pos=1.667 pos_rate=46.2% (n=118691 n_pos=54870)
2026-05-11 03:09:08,320 INFO GRU R threshold USDJPY/sell: q25_pos=0.542 q50_pos=1.667 pos_rate=44.6% (n=116030 n_pos=51793)
2026-05-11 03:09:08,326 INFO GRU R threshold EURJPY/buy: q25_pos=0.530 q50_pos=1.416 pos_rate=46.1% (n=118651 n_pos=54654)
2026-05-11 03:09:08,332 INFO GRU R threshold EURJPY/sell: q25_pos=0.525 q50_pos=1.667 pos_rate=44.7% (n=117320 n_pos=52399)
2026-05-11 03:09:08,339 INFO GRU R threshold GBPJPY/buy: q25_pos=0.517 q50_pos=1.443 pos_rate=45.9% (n=118277 n_pos=54278)
2026-05-11 03:09:08,345 INFO GRU R threshold GBPJPY/sell: q25_pos=0.521 q50_pos=1.667 pos_rate=45.3% (n=116299 n_pos=52702)
2026-05-11 03:09:08,351 INFO GRU R threshold GBPUSD/buy: q25_pos=0.524 q50_pos=1.667 pos_rate=45.8% (n=117764 n_pos=53894)
2026-05-11 03:09:08,357 INFO GRU R threshold GBPUSD/sell: q25_pos=0.533 q50_pos=1.667 pos_rate=45.8% (n=117404 n_pos=53785)
2026-05-11 03:09:08,357 INFO GRU per-symbol R thresholds saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/symbol_r_thresholds.json (6 symbols)
2026-05-11 03:09:08,374 INFO Retrain complete. Total wall-clock: 232.4s
  DONE  Retrain gru [train-split retrain]
  START Retrain regime [train-split retrain]
2026-05-11 03:09:11,662 INFO retrain environment: KAGGLE
2026-05-11 03:09:13,240 INFO Device: CUDA (2 GPU(s))
2026-05-11 03:09:13,249 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 03:09:13,249 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 03:09:13,249 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-11 03:09:13,249 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-11 03:09:13,250 INFO Retrain data split: train
2026-05-11 03:09:13,250 INFO Retrain rolling fold selector: latest
2026-05-11 03:09:13,251 INFO === RegimeClassifier retrain (hierarchical: HTF 2-score bias + LTF 5-score behaviour) ===
2026-05-11 03:09:13,394 INFO NumExpr defaulting to 4 threads.
2026-05-11 03:09:13,585 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-11 03:09:13,585 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-11 03:09:13,585 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-11 03:09:13,586 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-11 03:09:13,638 INFO Regime rolling folds selected: [None]
2026-05-11 03:09:13,638 INFO === Regime rolling fold 1/1: train_all ===
2026-05-11 03:09:13,638 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-11 03:09:13,679 INFO Split boundaries loaded fold=train_all/6 — train 2016-01-04→2023-08-04  val 2022-08-05→2023-08-04  test 2023-08-07→2025-08-05
2026-05-11 03:09:13,680 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
2026-05-11 03:09:13,697 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
2026-05-11 03:09:13,712 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
2026-05-11 03:09:13,728 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
2026-05-11 03:09:13,744 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
2026-05-11 03:09:13,758 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 03:09:14,001 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 03:09:14,071 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 03:09:14,095 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 03:09:14,095 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 03:09:14,106 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 03:09:14,107 INFO Loaded XAUUSD/4H split=train fold=latest: 12102 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:09:14,677 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 550, 'BIAS_DOWN': 255, 'BIAS_NEUTRAL': 11297}  ambiguous=7029 (total=12102) horizon=36
2026-05-11 03:09:14,683 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected XAUUSD — 12052 samples (group=gold) score_means={'bias_up_score': 0.0456, 'bias_down_score': 0.0212} labels={'BIAS_UP': 550, 'BIAS_DOWN': 255, 'BIAS_NEUTRAL': 11247} clean={'BIAS_UP': 550, 'BIAS_DOWN': 255, 'BIAS_NEUTRAL': 4250}
2026-05-11 03:09:14,854 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:14,890 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:14,909 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:14,909 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:14,918 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:14,919 INFO Loaded EURUSD/4H split=train fold=latest: 11404 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:09:15,434 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 299, 'BIAS_DOWN': 288, 'BIAS_NEUTRAL': 10817}  ambiguous=6558 (total=11404) horizon=36
2026-05-11 03:09:15,439 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURUSD — 11354 samples (group=dollar) score_means={'bias_up_score': 0.0263, 'bias_down_score': 0.0254} labels={'BIAS_UP': 299, 'BIAS_DOWN': 288, 'BIAS_NEUTRAL': 10767} clean={'BIAS_UP': 299, 'BIAS_DOWN': 288, 'BIAS_NEUTRAL': 4231}
2026-05-11 03:09:15,595 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:15,630 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:15,650 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:15,650 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:15,658 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:15,659 INFO Loaded USDJPY/4H split=train fold=latest: 11403 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:09:16,194 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 431, 'BIAS_DOWN': 164, 'BIAS_NEUTRAL': 10808}  ambiguous=6695 (total=11403) horizon=36
2026-05-11 03:09:16,199 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected USDJPY — 11353 samples (group=dollar) score_means={'bias_up_score': 0.038, 'bias_down_score': 0.0144} labels={'BIAS_UP': 431, 'BIAS_DOWN': 164, 'BIAS_NEUTRAL': 10758} clean={'BIAS_UP': 431, 'BIAS_DOWN': 164, 'BIAS_NEUTRAL': 4092}
2026-05-11 03:09:16,349 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:16,384 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:16,405 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:16,406 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:16,413 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:16,414 INFO Loaded EURJPY/4H split=train fold=latest: 11407 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:09:16,941 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 313, 'BIAS_DOWN': 162, 'BIAS_NEUTRAL': 10932}  ambiguous=6806 (total=11407) horizon=36
2026-05-11 03:09:16,946 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected EURJPY — 11357 samples (group=cross) score_means={'bias_up_score': 0.0276, 'bias_down_score': 0.0143} labels={'BIAS_UP': 313, 'BIAS_DOWN': 162, 'BIAS_NEUTRAL': 10882} clean={'BIAS_UP': 313, 'BIAS_DOWN': 162, 'BIAS_NEUTRAL': 4094}
2026-05-11 03:09:17,101 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:17,137 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:17,159 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:17,159 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:17,167 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:17,168 INFO Loaded GBPJPY/4H split=train fold=latest: 11408 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:09:17,693 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 467, 'BIAS_DOWN': 285, 'BIAS_NEUTRAL': 10656}  ambiguous=6835 (total=11408) horizon=36
2026-05-11 03:09:17,698 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPJPY — 11358 samples (group=cross) score_means={'bias_up_score': 0.0411, 'bias_down_score': 0.0251} labels={'BIAS_UP': 467, 'BIAS_DOWN': 285, 'BIAS_NEUTRAL': 10606} clean={'BIAS_UP': 467, 'BIAS_DOWN': 285, 'BIAS_NEUTRAL': 3818}
2026-05-11 03:09:17,861 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:17,897 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:17,917 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:17,917 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:17,925 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:17,926 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:09:18,439 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 255, 'BIAS_DOWN': 300, 'BIAS_NEUTRAL': 10847}  ambiguous=6860 (total=11402) horizon=36
2026-05-11 03:09:18,444 INFO Regime[4H mode=htf_bias split=train fold=latest]: collected GBPUSD — 11352 samples (group=dollar) score_means={'bias_up_score': 0.0225, 'bias_down_score': 0.0264} labels={'BIAS_UP': 255, 'BIAS_DOWN': 300, 'BIAS_NEUTRAL': 10797} clean={'BIAS_UP': 255, 'BIAS_DOWN': 300, 'BIAS_NEUTRAL': 3987}
2026-05-11 03:09:18,505 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 780, 'BIAS_DOWN': 447, 'BIAS_NEUTRAL': 21488}, 'dollar': {'BIAS_UP': 985, 'BIAS_DOWN': 752, 'BIAS_NEUTRAL': 32322}, 'gold': {'BIAS_UP': 550, 'BIAS_DOWN': 255, 'BIAS_NEUTRAL': 11247}}
2026-05-11 03:09:18,506 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0343, 'bias_down_score': 0.0197}, 'dollar': {'bias_up_score': 0.0289, 'bias_down_score': 0.0221}, 'gold': {'bias_up_score': 0.0456, 'bias_down_score': 0.0212}}
2026-05-11 03:09:18,506 INFO Regime[4H mode=htf_bias] label distribution by year: {2016: {'BIAS_UP': 274, 'BIAS_DOWN': 351, 'BIAS_NEUTRAL': 8197}, 2017: {'BIAS_UP': 495, 'BIAS_DOWN': 167, 'BIAS_NEUTRAL': 8451}, 2018: {'BIAS_UP': 208, 'BIAS_DOWN': 250, 'BIAS_NEUTRAL': 8672}, 2019: {'BIAS_UP': 189, 'BIAS_DOWN': 174, 'BIAS_NEUTRAL': 8739}, 2020: {'BIAS_UP': 285, 'BIAS_DOWN': 131, 'BIAS_NEUTRAL': 8695}, 2021: {'BIAS_UP': 311, 'BIAS_DOWN': 163, 'BIAS_NEUTRAL': 8617}, 2022: {'BIAS_UP': 340, 'BIAS_DOWN': 162, 'BIAS_NEUTRAL': 8619}, 2023: {'BIAS_UP': 213, 'BIAS_DOWN': 56, 'BIAS_NEUTRAL': 5067}}
2026-05-11 03:09:18,506 INFO Regime[4H mode=htf_bias] score means by year: {2016: {'bias_up_score': 0.0311, 'bias_down_score': 0.0398}, 2017: {'bias_up_score': 0.0543, 'bias_down_score': 0.0183}, 2018: {'bias_up_score': 0.0228, 'bias_down_score': 0.0274}, 2019: {'bias_up_score': 0.0208, 'bias_down_score': 0.0191}, 2020: {'bias_up_score': 0.0313, 'bias_down_score': 0.0144}, 2021: {'bias_up_score': 0.0342, 'bias_down_score': 0.0179}, 2022: {'bias_up_score': 0.0373, 'bias_down_score': 0.0178}, 2023: {'bias_up_score': 0.0399, 'bias_down_score': 0.0105}}
2026-05-11 03:09:18,551 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
2026-05-11 03:09:18,552 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
2026-05-11 03:09:18,552 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
2026-05-11 03:09:18,553 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 03:09:18,554 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
2026-05-11 03:09:18,555 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
2026-05-11 03:09:18,572 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 03:09:18,575 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 03:09:18,576 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 03:09:18,577 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 03:09:18,577 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 03:09:18,578 INFO Loaded XAUUSD/4H split=val fold=latest: 1581 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:09:18,901 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 53, 'BIAS_DOWN': 23, 'BIAS_NEUTRAL': 1505}  ambiguous=929 (total=1581) horizon=36
2026-05-11 03:09:18,903 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected XAUUSD — 1531 samples (group=gold) score_means={'bias_up_score': 0.0346, 'bias_down_score': 0.015} labels={'BIAS_UP': 53, 'BIAS_DOWN': 23, 'BIAS_NEUTRAL': 1455} clean={'BIAS_UP': 53, 'BIAS_DOWN': 23, 'BIAS_NEUTRAL': 558}
2026-05-11 03:09:18,975 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:18,977 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:18,978 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:18,978 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:18,979 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:18,980 INFO Loaded EURUSD/4H split=val fold=latest: 1491 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:09:19,281 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 18, 'BIAS_DOWN': 38, 'BIAS_NEUTRAL': 1435}  ambiguous=844 (total=1491) horizon=36
2026-05-11 03:09:19,284 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURUSD — 1441 samples (group=dollar) score_means={'bias_up_score': 0.0125, 'bias_down_score': 0.0264} labels={'BIAS_UP': 18, 'BIAS_DOWN': 38, 'BIAS_NEUTRAL': 1385} clean={'BIAS_UP': 18, 'BIAS_DOWN': 38, 'BIAS_NEUTRAL': 577}
2026-05-11 03:09:19,357 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:19,359 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:19,360 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:19,361 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:19,361 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:19,362 INFO Loaded USDJPY/4H split=val fold=latest: 1489 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:09:19,662 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 95, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 1386}  ambiguous=905 (total=1489) horizon=36
2026-05-11 03:09:19,665 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected USDJPY — 1439 samples (group=dollar) score_means={'bias_up_score': 0.066, 'bias_down_score': 0.0056} labels={'BIAS_UP': 95, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 1336} clean={'BIAS_UP': 95, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 478}
2026-05-11 03:09:19,731 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:19,734 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:19,734 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:19,735 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:19,735 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:19,736 INFO Loaded EURJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:09:20,040 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 52, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 1442}  ambiguous=913 (total=1494) horizon=36
2026-05-11 03:09:20,043 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected EURJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.036, 'bias_down_score': 0.0} labels={'BIAS_UP': 52, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 1392} clean={'BIAS_UP': 52, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 510}
2026-05-11 03:09:20,108 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:20,110 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:20,111 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:20,111 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:20,112 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:20,113 INFO Loaded GBPJPY/4H split=val fold=latest: 1494 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:09:20,406 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 40, 'BIAS_DOWN': 3, 'BIAS_NEUTRAL': 1451}  ambiguous=884 (total=1494) horizon=36
2026-05-11 03:09:20,408 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPJPY — 1444 samples (group=cross) score_means={'bias_up_score': 0.0277, 'bias_down_score': 0.0021} labels={'BIAS_UP': 40, 'BIAS_DOWN': 3, 'BIAS_NEUTRAL': 1401} clean={'BIAS_UP': 40, 'BIAS_DOWN': 3, 'BIAS_NEUTRAL': 542}
2026-05-11 03:09:20,473 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:20,476 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:20,476 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:20,477 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:20,477 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:20,478 INFO Loaded GBPUSD/4H split=val fold=latest: 1488 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:09:20,781 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 18, 'BIAS_DOWN': 29, 'BIAS_NEUTRAL': 1441}  ambiguous=896 (total=1488) horizon=36
2026-05-11 03:09:20,783 INFO Regime[4H mode=htf_bias split=val fold=latest]: collected GBPUSD — 1438 samples (group=dollar) score_means={'bias_up_score': 0.0125, 'bias_down_score': 0.0202} labels={'BIAS_UP': 18, 'BIAS_DOWN': 29, 'BIAS_NEUTRAL': 1391} clean={'BIAS_UP': 18, 'BIAS_DOWN': 29, 'BIAS_NEUTRAL': 533}
2026-05-11 03:09:20,844 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 92, 'BIAS_DOWN': 3, 'BIAS_NEUTRAL': 2793}, 'dollar': {'BIAS_UP': 131, 'BIAS_DOWN': 75, 'BIAS_NEUTRAL': 4112}, 'gold': {'BIAS_UP': 53, 'BIAS_DOWN': 23, 'BIAS_NEUTRAL': 1455}}
2026-05-11 03:09:20,844 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0319, 'bias_down_score': 0.001}, 'dollar': {'bias_up_score': 0.0303, 'bias_down_score': 0.0174}, 'gold': {'bias_up_score': 0.0346, 'bias_down_score': 0.015}}
2026-05-11 03:09:20,844 INFO Regime[4H mode=htf_bias] label distribution by year: {2022: {'BIAS_UP': 72, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 3283}, 2023: {'BIAS_UP': 204, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 5077}}
2026-05-11 03:09:20,845 INFO Regime[4H mode=htf_bias] score means by year: {2022: {'bias_up_score': 0.0212, 'bias_down_score': 0.0135}, 2023: {'bias_up_score': 0.0382, 'bias_down_score': 0.0103}}
2026-05-11 03:09:20,886 INFO Regime phase HTF dataset build fold=train_all: 7.2s (train=68826 val=8737)
2026-05-11 03:09:20,887 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260511_030920
2026-05-11 03:09:21,090 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=28, n_classes=2)
2026-05-11 03:09:21,090 INFO Regime 4H/htf_bias warm start enabled from existing weights
2026-05-11 03:09:21,097 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=68826 val=8737 train_labels={'BIAS_UP': 2315, 'BIAS_DOWN': 1454, 'BIAS_NEUTRAL': 65057} val_labels={'BIAS_UP': 276, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 8360}
2026-05-11 03:09:21,097 INFO RegimeClassifier[mode=htf_bias]: warm start HTF score head
2026-05-11 03:09:21,097 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-11 03:09:21,097 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 28.73, 'bias_down_score': 30.0}
2026-05-11 03:09:21,101 INFO RegimeClassifier[mode=htf_bias]: HTF balanced sampler — dir=3769 neutral=65057 dir_weight=8 => dir_frac_per_epoch≈31.7%
2026-05-11 03:09:24,856 INFO Regime HTF score epoch  1/50 — tr=0.4745 va=0.6943 acc=0.876 bal=0.872 threshold=0.99 margin=0.00 recall={'BIAS_UP': 0.899, 'BIAS_DOWN': 0.842, 'BIAS_NEUTRAL': 0.875} precision={'BIAS_UP': 0.278, 'BIAS_DOWN': 0.176, 'BIAS_NEUTRAL': 0.994}
2026-05-11 03:09:26,502 INFO Regime HTF score epoch  2/50 — tr=0.4616 va=0.6828 bal=0.867
2026-05-11 03:09:28,157 INFO Regime HTF score epoch  3/50 — tr=0.4702 va=0.6686 bal=0.864
2026-05-11 03:09:29,806 INFO Regime HTF score epoch  4/50 — tr=0.4647 va=0.6788 bal=0.866
2026-05-11 03:09:31,427 INFO Regime HTF score epoch  5/50 — tr=0.4592 va=0.6808 acc=0.877 bal=0.862 threshold=0.99 margin=0.00 recall={'BIAS_UP': 0.877, 'BIAS_DOWN': 0.832, 'BIAS_NEUTRAL': 0.878} precision={'BIAS_UP': 0.278, 'BIAS_DOWN': 0.175, 'BIAS_NEUTRAL': 0.993}
2026-05-11 03:09:33,104 INFO Regime HTF score epoch  6/50 — tr=0.4655 va=0.6731 bal=0.854
2026-05-11 03:09:34,746 INFO Regime HTF score epoch  7/50 — tr=0.4642 va=0.6718 bal=0.857
2026-05-11 03:09:36,400 INFO Regime HTF score epoch  8/50 — tr=0.4690 va=0.6780 bal=0.862
2026-05-11 03:09:38,023 INFO Regime HTF score epoch  9/50 — tr=0.4494 va=0.6697 bal=0.850
2026-05-11 03:09:39,658 INFO Regime HTF score epoch 10/50 — tr=0.4650 va=0.6749 acc=0.879 bal=0.846 threshold=0.99 margin=0.00 recall={'BIAS_UP': 0.855, 'BIAS_DOWN': 0.802, 'BIAS_NEUTRAL': 0.88} precision={'BIAS_UP': 0.28, 'BIAS_DOWN': 0.171, 'BIAS_NEUTRAL': 0.992}
2026-05-11 03:09:39,658 INFO Regime HTF score early stop at epoch 10
2026-05-11 03:09:41,152 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.990 margin=0.000 precision={'BIAS_UP': 0.281, 'BIAS_DOWN': 0.176, 'BIAS_NEUTRAL': 0.994} recall={'BIAS_UP': 0.891, 'BIAS_DOWN': 0.832, 'BIAS_NEUTRAL': 0.878} f1={'BIAS_UP': 0.427, 'BIAS_DOWN': 0.291, 'BIAS_NEUTRAL': 0.932} confusion=[[246, 0, 30], [0, 84, 17], [630, 393, 7337]] score_mae={'bias_up_score': 0.1831, 'bias_down_score': 0.1199} pred_share={'BIAS_UP': 0.1003, 'BIAS_DOWN': 0.0546, 'BIAS_NEUTRAL': 0.8451}
2026-05-11 03:09:41,153 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.281, 'BIAS_DOWN': 0.176, 'BIAS_NEUTRAL': 0.994} min_precision=0.500 recall={'BIAS_UP': 0.891, 'BIAS_DOWN': 0.832, 'BIAS_NEUTRAL': 0.878} min_recall=0.100 f1={'BIAS_UP': 0.427, 'BIAS_DOWN': 0.291, 'BIAS_NEUTRAL': 0.932} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_UP', 'BIAS_DOWN'] weak_recall=[] weak_f1=[] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-11 03:09:41,156 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 03:09:41,156 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-11 03:09:41,157 INFO Regime phase HTF train fold=train_all: 20.1s
2026-05-11 03:09:41,260 INFO Regime HTF complete fold=train_all: acc=0.877 bal=0.867 train=68826 val=8737 per_class={'BIAS_UP': 0.891, 'BIAS_DOWN': 0.832, 'BIAS_NEUTRAL': 0.878} precision={'BIAS_UP': 0.281, 'BIAS_DOWN': 0.176, 'BIAS_NEUTRAL': 0.994} threshold=0.990 margin=0.000
2026-05-11 03:09:41,261 INFO Loaded GBPUSD/4H split=train fold=latest: 11402 bars (2016-01-04 → 2023-08-04)
2026-05-11 03:09:41,425 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 255, 'BIAS_DOWN': 300, 'BIAS_NEUTRAL': 10847}  ambiguous=6860 (total=11402) horizon=36
2026-05-11 03:09:41,427 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on GBPUSD 4H:
{'BIAS_UP': 3.8636363636363638, 'BIAS_DOWN': 3.5294117647058822, 'BIAS_NEUTRAL': 71.36184210526316}
2026-05-11 03:09:41,431 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (all labels):
{'BIAS_UP': {'n': 255, 'mean': 0.0006942435718027296, 'mean_over_std': 0.3535182817741541}, 'BIAS_DOWN': {'n': 300, 'mean': -0.0007615786091842384, 'mean_over_std': -0.2905559984734911}, 'BIAS_NEUTRAL': {'n': 10846, 'mean': -5.264589798607426e-06, 'mean_over_std': -0.002007943010728463}}
2026-05-11 03:09:41,431 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 255, 'mean': 0.0006942435718027296, 'mean_over_std': 0.3535182817741541}, 'BIAS_DOWN': {'n': 300, 'mean': -0.0007615786091842384, 'mean_over_std': -0.2905559984734911}, 'BIAS_NEUTRAL': {'n': 3987, 'mean': -1.7489525578049586e-05, 'mean_over_std': -0.007235149795477735}}
2026-05-11 03:09:41,435 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-11 03:09:41,438 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
2026-05-11 03:09:41,440 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
2026-05-11 03:09:41,442 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
2026-05-11 03:09:41,443 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
2026-05-11 03:09:41,445 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
2026-05-11 03:09:41,447 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 03:09:41,466 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 03:09:41,475 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 03:09:41,478 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 03:09:41,479 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 03:09:41,479 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 03:09:41,487 INFO Loaded XAUUSD/1H split=train fold=latest: 44277 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:09:42,379 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected XAUUSD — 44227 samples (group=gold) score_means={'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}
2026-05-11 03:09:42,484 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:42,486 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:42,487 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:42,487 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:42,487 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:42,490 INFO Loaded EURUSD/1H split=train fold=latest: 43739 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:09:43,310 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURUSD — 43689 samples (group=dollar) score_means={'trend_score': 0.49, 'range_score': 0.236, 'chop_score': 0.4622, 'volatility_percentile': 0.3812, 'consolidation_score': 0.1862}
2026-05-11 03:09:43,413 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:43,415 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:43,416 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:43,416 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:43,417 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:43,419 INFO Loaded USDJPY/1H split=train fold=latest: 43736 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:09:44,237 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected USDJPY — 43686 samples (group=dollar) score_means={'trend_score': 0.4949, 'range_score': 0.2312, 'chop_score': 0.4613, 'volatility_percentile': 0.38, 'consolidation_score': 0.1931}
2026-05-11 03:09:44,340 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:44,343 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:44,344 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:44,344 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:44,344 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:44,347 INFO Loaded EURJPY/1H split=train fold=latest: 43738 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:09:45,167 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected EURJPY — 43688 samples (group=cross) score_means={'trend_score': 0.4888, 'range_score': 0.2345, 'chop_score': 0.4662, 'volatility_percentile': 0.3831, 'consolidation_score': 0.1882}
2026-05-11 03:09:45,271 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:45,273 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:45,274 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:45,274 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:45,274 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:45,277 INFO Loaded GBPJPY/1H split=train fold=latest: 43728 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:09:46,088 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPJPY — 43678 samples (group=cross) score_means={'trend_score': 0.4878, 'range_score': 0.2348, 'chop_score': 0.4673, 'volatility_percentile': 0.3839, 'consolidation_score': 0.1904}
2026-05-11 03:09:46,195 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:46,197 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:46,198 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:46,198 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:46,199 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:46,201 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:09:47,026 INFO Regime[1H mode=ltf_behaviour split=train fold=latest]: collected GBPUSD — 43676 samples (group=dollar) score_means={'trend_score': 0.4923, 'range_score': 0.2335, 'chop_score': 0.4611, 'volatility_percentile': 0.3832, 'consolidation_score': 0.1856}
2026-05-11 03:09:47,149 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2346, 'chop_score': 0.4667, 'volatility_percentile': 0.3835, 'consolidation_score': 0.1893}, 'dollar': {'trend_score': 0.4924, 'range_score': 0.2335, 'chop_score': 0.4615, 'volatility_percentile': 0.3815, 'consolidation_score': 0.1883}, 'gold': {'trend_score': 0.4886, 'range_score': 0.2365, 'chop_score': 0.4663, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}}
2026-05-11 03:09:47,150 INFO Regime[1H mode=ltf_behaviour] score means by year: {2016: {'trend_score': 0.495, 'range_score': 0.233, 'chop_score': 0.4607, 'volatility_percentile': 0.3776, 'consolidation_score': 0.192}, 2017: {'trend_score': 0.5017, 'range_score': 0.2328, 'chop_score': 0.4574, 'volatility_percentile': 0.3689, 'consolidation_score': 0.1942}, 2018: {'trend_score': 0.4886, 'range_score': 0.237, 'chop_score': 0.4655, 'volatility_percentile': 0.3947, 'consolidation_score': 0.1785}, 2019: {'trend_score': 0.4817, 'range_score': 0.236, 'chop_score': 0.4691, 'volatility_percentile': 0.3725, 'consolidation_score': 0.198}, 2020: {'trend_score': 0.4874, 'range_score': 0.2311, 'chop_score': 0.4674, 'volatility_percentile': 0.3848, 'consolidation_score': 0.1919}, 2021: {'trend_score': 0.4872, 'range_score': 0.2372, 'chop_score': 0.4672, 'volatility_percentile': 0.377, 'consolidation_score': 0.192}, 2022: {'trend_score': 0.4893, 'range_score': 0.2357, 'chop_score': 0.4647, 'volatility_percentile': 0.3962, 'consolidation_score': 0.1784}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 03:09:47,250 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
2026-05-11 03:09:47,251 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 03:09:47,253 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
2026-05-11 03:09:47,254 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
2026-05-11 03:09:47,255 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
2026-05-11 03:09:47,257 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
2026-05-11 03:09:47,266 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-11 03:09:47,269 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-11 03:09:47,270 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-11 03:09:47,271 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-11 03:09:47,271 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-11 03:09:47,273 INFO Loaded XAUUSD/1H split=val fold=latest: 5167 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:09:47,520 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected XAUUSD — 5117 samples (group=gold) score_means={'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}
2026-05-11 03:09:47,626 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:47,628 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:47,629 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:47,629 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:47,629 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:47,631 INFO Loaded EURUSD/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:09:47,862 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURUSD — 5053 samples (group=dollar) score_means={'trend_score': 0.4883, 'range_score': 0.2413, 'chop_score': 0.463, 'volatility_percentile': 0.3837, 'consolidation_score': 0.1793}
2026-05-11 03:09:47,969 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:47,971 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:47,972 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:47,972 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:47,973 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:47,974 INFO Loaded USDJPY/1H split=val fold=latest: 5096 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:09:48,204 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected USDJPY — 5046 samples (group=dollar) score_means={'trend_score': 0.5085, 'range_score': 0.2259, 'chop_score': 0.4506, 'volatility_percentile': 0.3809, 'consolidation_score': 0.1888}
2026-05-11 03:09:48,309 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:48,311 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:48,312 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:48,312 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:48,313 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:48,314 INFO Loaded EURJPY/1H split=val fold=latest: 5103 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:09:48,547 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected EURJPY — 5053 samples (group=cross) score_means={'trend_score': 0.5023, 'range_score': 0.2261, 'chop_score': 0.4524, 'volatility_percentile': 0.3837, 'consolidation_score': 0.18}
2026-05-11 03:09:48,653 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:48,655 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:48,656 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:48,656 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:48,657 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:48,658 INFO Loaded GBPJPY/1H split=val fold=latest: 5093 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:09:48,895 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPJPY — 5043 samples (group=cross) score_means={'trend_score': 0.4743, 'range_score': 0.2372, 'chop_score': 0.4716, 'volatility_percentile': 0.3777, 'consolidation_score': 0.1917}
2026-05-11 03:09:49,015 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:49,017 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:49,018 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:49,018 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:49,018 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-11 03:09:49,020 INFO Loaded GBPUSD/1H split=val fold=latest: 5090 bars (2022-08-05 → 2023-08-04)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-11 03:09:49,258 INFO Regime[1H mode=ltf_behaviour split=val fold=latest]: collected GBPUSD — 5040 samples (group=dollar) score_means={'trend_score': 0.4843, 'range_score': 0.2371, 'chop_score': 0.4614, 'volatility_percentile': 0.377, 'consolidation_score': 0.1835}
2026-05-11 03:09:49,359 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4883, 'range_score': 0.2317, 'chop_score': 0.462, 'volatility_percentile': 0.3807, 'consolidation_score': 0.1858}, 'dollar': {'trend_score': 0.4937, 'range_score': 0.2348, 'chop_score': 0.4583, 'volatility_percentile': 0.3806, 'consolidation_score': 0.1839}, 'gold': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}}
2026-05-11 03:09:49,359 INFO Regime[1H mode=ltf_behaviour] score means by year: {2022: {'trend_score': 0.4905, 'range_score': 0.2364, 'chop_score': 0.4625, 'volatility_percentile': 0.385, 'consolidation_score': 0.1797}, 2023: {'trend_score': 0.4942, 'range_score': 0.2301, 'chop_score': 0.4564, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1883}}
2026-05-11 03:09:49,433 INFO Regime phase LTF dataset build fold=train_all: 8.0s (train=262644 val=30352)
2026-05-11 03:09:49,434 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260511_030949
2026-05-11 03:09:49,439 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=31, n_classes=5)
2026-05-11 03:09:49,439 INFO Regime 1H/ltf_behaviour warm start enabled from existing weights
2026-05-11 03:09:49,462 INFO RegimeClassifier[mode=ltf_behaviour]: warm start score head
2026-05-11 03:09:49,462 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-11 03:09:49,978 INFO Regime score epoch  1/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.0167, 'range_score': 0.0322, 'chop_score': 0.0198, 'volatility_percentile': 0.0146, 'consolidation_score': 0.0195}
2026-05-11 03:09:50,461 INFO Regime score epoch  2/50 — tr=0.0034 va=0.0008
2026-05-11 03:09:50,969 INFO Regime score epoch  3/50 — tr=0.0034 va=0.0008
2026-05-11 03:09:51,462 INFO Regime score epoch  4/50 — tr=0.0034 va=0.0008
2026-05-11 03:09:51,984 INFO Regime score epoch  5/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.0167, 'range_score': 0.0322, 'chop_score': 0.0197, 'volatility_percentile': 0.0148, 'consolidation_score': 0.019}
2026-05-11 03:09:52,466 INFO Regime score epoch  6/50 — tr=0.0034 va=0.0008
2026-05-11 03:09:52,951 INFO Regime score epoch  7/50 — tr=0.0034 va=0.0008
2026-05-11 03:09:53,457 INFO Regime score epoch  8/50 — tr=0.0034 va=0.0008
2026-05-11 03:09:53,965 INFO Regime score epoch  9/50 — tr=0.0034 va=0.0008
2026-05-11 03:09:54,457 INFO Regime score epoch 10/50 — tr=0.0034 va=0.0008 mae={'trend_score': 0.0167, 'range_score': 0.0318, 'chop_score': 0.0197, 'volatility_percentile': 0.0146, 'consolidation_score': 0.0192}
2026-05-11 03:09:54,947 INFO Regime score epoch 11/50 — tr=0.0033 va=0.0008
2026-05-11 03:09:55,443 INFO Regime score epoch 12/50 — tr=0.0033 va=0.0008
2026-05-11 03:09:55,933 INFO Regime score epoch 13/50 — tr=0.0033 va=0.0008
2026-05-11 03:09:56,419 INFO Regime score epoch 14/50 — tr=0.0033 va=0.0008
2026-05-11 03:09:56,906 INFO Regime score epoch 15/50 — tr=0.0033 va=0.0008 mae={'trend_score': 0.0159, 'range_score': 0.0314, 'chop_score': 0.0192, 'volatility_percentile': 0.0141, 'consolidation_score': 0.0189}
2026-05-11 03:09:57,403 INFO Regime score epoch 16/50 — tr=0.0033 va=0.0008
2026-05-11 03:09:57,892 INFO Regime score epoch 17/50 — tr=0.0033 va=0.0008
2026-05-11 03:09:58,375 INFO Regime score epoch 18/50 — tr=0.0033 va=0.0008
2026-05-11 03:09:58,862 INFO Regime score epoch 19/50 — tr=0.0033 va=0.0008
2026-05-11 03:09:59,349 INFO Regime score epoch 20/50 — tr=0.0033 va=0.0008 mae={'trend_score': 0.0164, 'range_score': 0.0312, 'chop_score': 0.0187, 'volatility_percentile': 0.0134, 'consolidation_score': 0.0184}
2026-05-11 03:09:59,833 INFO Regime score epoch 21/50 — tr=0.0032 va=0.0008
2026-05-11 03:10:00,313 INFO Regime score epoch 22/50 — tr=0.0032 va=0.0008
2026-05-11 03:10:00,807 INFO Regime score epoch 23/50 — tr=0.0032 va=0.0008
2026-05-11 03:10:01,309 INFO Regime score epoch 24/50 — tr=0.0032 va=0.0008
2026-05-11 03:10:01,836 INFO Regime score epoch 25/50 — tr=0.0032 va=0.0008 mae={'trend_score': 0.0157, 'range_score': 0.0311, 'chop_score': 0.0191, 'volatility_percentile': 0.0138, 'consolidation_score': 0.0183}
2026-05-11 03:10:02,335 INFO Regime score epoch 26/50 — tr=0.0032 va=0.0008
2026-05-11 03:10:02,835 INFO Regime score epoch 27/50 — tr=0.0032 va=0.0008
2026-05-11 03:10:03,330 INFO Regime score epoch 28/50 — tr=0.0032 va=0.0007
2026-05-11 03:10:03,836 INFO Regime score epoch 29/50 — tr=0.0032 va=0.0007
2026-05-11 03:10:04,340 INFO Regime score epoch 30/50 — tr=0.0032 va=0.0007 mae={'trend_score': 0.0158, 'range_score': 0.0308, 'chop_score': 0.0185, 'volatility_percentile': 0.0131, 'consolidation_score': 0.0184}
2026-05-11 03:10:04,835 INFO Regime score epoch 31/50 — tr=0.0032 va=0.0007
2026-05-11 03:10:05,334 INFO Regime score epoch 32/50 — tr=0.0032 va=0.0007
2026-05-11 03:10:05,844 INFO Regime score epoch 33/50 — tr=0.0032 va=0.0007
2026-05-11 03:10:06,339 INFO Regime score epoch 34/50 — tr=0.0032 va=0.0007
2026-05-11 03:10:06,851 INFO Regime score epoch 35/50 — tr=0.0032 va=0.0007 mae={'trend_score': 0.0155, 'range_score': 0.0307, 'chop_score': 0.0185, 'volatility_percentile': 0.014, 'consolidation_score': 0.0182}
2026-05-11 03:10:07,353 INFO Regime score epoch 36/50 — tr=0.0032 va=0.0007
2026-05-11 03:10:07,872 INFO Regime score epoch 37/50 — tr=0.0032 va=0.0007
2026-05-11 03:10:08,364 INFO Regime score epoch 38/50 — tr=0.0032 va=0.0007
2026-05-11 03:10:08,876 INFO Regime score epoch 39/50 — tr=0.0031 va=0.0007
2026-05-11 03:10:09,413 INFO Regime score epoch 40/50 — tr=0.0032 va=0.0007 mae={'trend_score': 0.0154, 'range_score': 0.0308, 'chop_score': 0.0184, 'volatility_percentile': 0.013, 'consolidation_score': 0.0179}
2026-05-11 03:10:09,908 INFO Regime score epoch 41/50 — tr=0.0032 va=0.0007
2026-05-11 03:10:10,410 INFO Regime score epoch 42/50 — tr=0.0032 va=0.0007
2026-05-11 03:10:10,906 INFO Regime score epoch 43/50 — tr=0.0032 va=0.0007
2026-05-11 03:10:11,415 INFO Regime score epoch 44/50 — tr=0.0032 va=0.0007
2026-05-11 03:10:11,954 INFO Regime score epoch 45/50 — tr=0.0032 va=0.0007 mae={'trend_score': 0.0154, 'range_score': 0.0306, 'chop_score': 0.0183, 'volatility_percentile': 0.0134, 'consolidation_score': 0.018}
2026-05-11 03:10:12,441 INFO Regime score epoch 46/50 — tr=0.0031 va=0.0007
2026-05-11 03:10:12,939 INFO Regime score epoch 47/50 — tr=0.0031 va=0.0007
2026-05-11 03:10:13,433 INFO Regime score epoch 48/50 — tr=0.0031 va=0.0007
2026-05-11 03:10:13,942 INFO Regime score epoch 49/50 — tr=0.0032 va=0.0007
2026-05-11 03:10:14,444 INFO Regime score epoch 50/50 — tr=0.0032 va=0.0007 mae={'trend_score': 0.0155, 'range_score': 0.0308, 'chop_score': 0.0184, 'volatility_percentile': 0.0134, 'consolidation_score': 0.018}
2026-05-11 03:10:14,465 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0154, 'range_score': 0.0306, 'chop_score': 0.0183, 'volatility_percentile': 0.0134, 'consolidation_score': 0.018} mse={'trend_score': 0.00042, 'range_score': 0.00157, 'chop_score': 0.00055, 'volatility_percentile': 0.00034, 'consolidation_score': 0.00074} corr={'trend_score': 0.9958, 'range_score': 0.9623, 'chop_score': 0.9927, 'volatility_percentile': 0.9967, 'consolidation_score': 0.9923} pred_std={'trend_score': 0.2207, 'range_score': 0.1315, 'chop_score': 0.182, 'volatility_percentile': 0.22, 'consolidation_score': 0.2147} target_std={'trend_score': 0.2219, 'range_score': 0.1433, 'chop_score': 0.1888, 'volatility_percentile': 0.2165, 'consolidation_score': 0.2179}
2026-05-11 03:10:14,783 INFO RegimeClassifier[mode=ltf_behaviour] train group diagnostics={'dollar': {'n': 16632, 'score_mae': {'trend_score': 0.0151, 'range_score': 0.0305, 'chop_score': 0.0182, 'volatility_percentile': 0.0129, 'consolidation_score': 0.0183}, 'target_score_mean': {'trend_score': 0.491, 'range_score': 0.2347, 'chop_score': 0.4613, 'volatility_percentile': 0.3824, 'consolidation_score': 0.186}, 'pred_score_mean': {'trend_score': 0.4927, 'range_score': 0.2324, 'chop_score': 0.4609, 'volatility_percentile': 0.3781, 'consolidation_score': 0.1837}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3593, 44, 0, 4, 0, 0, 138], [5, 98, 0, 0, 0, 4, 3], [0, 0, 179, 10, 55, 0, 216], [1, 0, 4, 569, 33, 0, 82], [0, 0, 29, 15, 3083, 1, 188], [0, 17, 0, 0, 8, 68, 35], [121, 12, 61, 45, 60, 6, 7845]]}, 'gold': {'n': 8420, 'score_mae': {'trend_score': 0.0151, 'range_score': 0.0311, 'chop_score': 0.0184, 'volatility_percentile': 0.0136, 'consolidation_score': 0.0186}, 'target_score_mean': {'trend_score': 0.4883, 'range_score': 0.2355, 'chop_score': 0.4652, 'volatility_percentile': 0.3772, 'consolidation_score': 0.1919}, 'pred_score_mean': {'trend_score': 0.4907, 'range_score': 0.2332, 'chop_score': 0.4642, 'volatility_percentile': 0.372, 'consolidation_score': 0.1896}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1799, 23, 0, 0, 0, 0, 63], [4, 50, 0, 0, 0, 0, 2], [0, 0, 94, 10, 24, 0, 116], [1, 0, 2, 344, 18, 0, 51], [0, 0, 16, 21, 1576, 0, 91], [0, 13, 0, 0, 4, 45, 19], [49, 3, 40, 15, 50, 0, 3877]]}, 'jpy': {'n': 24948, 'score_mae': {'trend_score': 0.015, 'range_score': 0.0305, 'chop_score': 0.0181, 'volatility_percentile': 0.0138, 'consolidation_score': 0.0182}, 'target_score_mean': {'trend_score': 0.4904, 'range_score': 0.2338, 'chop_score': 0.4653, 'volatility_percentile': 0.3826, 'consolidation_score': 0.1905}, 'pred_score_mean': {'trend_score': 0.4919, 'range_score': 0.2315, 'chop_score': 0.4643, 'volatility_percentile': 0.3778, 'consolidation_score': 0.1876}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[5368, 91, 0, 2, 0, 0, 185], [7, 168, 0, 0, 0, 5, 7], [0, 0, 227, 18, 86, 0, 316], [3, 0, 3, 1102, 67, 0, 139], [0, 0, 35, 56, 4717, 0, 307], [0, 31, 0, 0, 13, 101, 78], [169, 11, 83, 78, 121, 7, 11347]]}}
2026-05-11 03:10:14,957 INFO RegimeClassifier[mode=ltf_behaviour] validation group diagnostics={'dollar': {'n': 10093, 'score_mae': {'trend_score': 0.0158, 'range_score': 0.0311, 'chop_score': 0.0185, 'volatility_percentile': 0.0129, 'consolidation_score': 0.0175}, 'target_score_mean': {'trend_score': 0.4863, 'range_score': 0.2392, 'chop_score': 0.4622, 'volatility_percentile': 0.3804, 'consolidation_score': 0.1814}, 'pred_score_mean': {'trend_score': 0.4883, 'range_score': 0.2352, 'chop_score': 0.4626, 'volatility_percentile': 0.3759, 'consolidation_score': 0.1797}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[2307, 18, 0, 1, 0, 0, 86], [4, 45, 0, 0, 0, 3, 1], [0, 0, 102, 7, 45, 0, 162], [0, 0, 1, 339, 28, 0, 55], [0, 0, 21, 20, 1919, 0, 90], [0, 10, 0, 0, 3, 38, 26], [56, 6, 33, 41, 43, 4, 4579]]}, 'gold': {'n': 5117, 'score_mae': {'trend_score': 0.0147, 'range_score': 0.0298, 'chop_score': 0.0182, 'volatility_percentile': 0.0133, 'consolidation_score': 0.0185}, 'target_score_mean': {'trend_score': 0.4973, 'range_score': 0.2307, 'chop_score': 0.4563, 'volatility_percentile': 0.3818, 'consolidation_score': 0.1828}, 'pred_score_mean': {'trend_score': 0.4985, 'range_score': 0.2299, 'chop_score': 0.4565, 'volatility_percentile': 0.3767, 'consolidation_score': 0.1808}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[1112, 12, 0, 0, 0, 0, 43], [3, 30, 0, 0, 0, 1, 1], [0, 0, 61, 3, 14, 0, 93], [0, 0, 2, 225, 9, 0, 19], [0, 0, 10, 10, 808, 0, 59], [0, 6, 0, 0, 4, 26, 14], [44, 1, 23, 19, 29, 1, 2435]]}, 'jpy': {'n': 15142, 'score_mae': {'trend_score': 0.0155, 'range_score': 0.0305, 'chop_score': 0.0181, 'volatility_percentile': 0.0137, 'consolidation_score': 0.0181}, 'target_score_mean': {'trend_score': 0.495, 'range_score': 0.2297, 'chop_score': 0.4582, 'volatility_percentile': 0.3808, 'consolidation_score': 0.1868}, 'pred_score_mean': {'trend_score': 0.496, 'range_score': 0.2266, 'chop_score': 0.4582, 'volatility_percentile': 0.3762, 'consolidation_score': 0.1842}, 'trade_states': ['TRADEABLE_TREND', 'TRADEABLE_TREND_HIGH_VOL', 'RANGE', 'CONSOLIDATION', 'NO_TRADE_CHOP', 'NO_TRADE_EXTREME_VOL', 'NO_TRADE_UNCERTAIN'], 'confusion': [[3362, 38, 0, 1, 0, 0, 117], [6, 98, 0, 0, 0, 6, 5], [0, 0, 137, 16, 47, 0, 184], [3, 0, 3, 703, 33, 0, 85], [0, 0, 25, 31, 2596, 0, 165], [0, 17, 0, 0, 7, 60, 38], [101, 6, 54, 37, 68, 7, 7086]]}}
2026-05-11 03:10:14,962 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-11 03:10:14,962 INFO Regime phase LTF train fold=train_all: 25.5s
2026-05-11 03:10:15,066 INFO Regime LTF complete fold=train_all: score_accuracy=0.981, train=262644 val=30352 mae={'trend_score': 0.0154, 'range_score': 0.0306, 'chop_score': 0.0183, 'volatility_percentile': 0.0134, 'consolidation_score': 0.018}
2026-05-11 03:10:15,069 INFO Loaded GBPUSD/1H split=train fold=latest: 43726 bars (2016-01-04 → 2023-08-04)
2026-05-11 03:10:15,419 INFO Regime[1H mode=ltf_behaviour fold=latest] LTF score diagnostics on GBPUSD:
{'trend_score': {'mean': 0.4921, 'q10': 0.1918, 'q50': 0.4877, 'q90': 0.8002}, 'range_score': {'mean': 0.2337, 'q10': 0.0513, 'q50': 0.2125, 'q90': 0.4349}, 'chop_score': {'mean': 0.4612, 'q10': 0.2148, 'q50': 0.4503, 'q90': 0.7255}, 'volatility_percentile': {'mean': 0.3832, 'q10': 0.1018, 'q50': 0.3704, 'q90': 0.6823}, 'consolidation_score': {'mean': 0.1854, 'q10': 0.0, 'q50': 0.1185, 'q90': 0.506}}
2026-05-11 03:10:15,423 INFO Regime retrain total: 62.2s (370559 train+val samples)
2026-05-11 03:10:15,427 INFO Retrain complete. Total wall-clock: 62.2s
  DONE  Retrain regime [train-split retrain]
  SKIP  Quality/RL incremental retrain — clean train-only weights retained

=== Round 3: Backtest on last 3yr (post-retrain evaluation) ===
  START Round 3 - Post-retrain backtest (last 3yr)
2026-05-11 03:10:16,979 INFO === STEP 6: BACKTEST (round3) ===
2026-05-11 03:10:16,980 INFO BT_WINDOW=round3 — post-retrain eval: 2022-08-05 → 2025-08-05 (last 3yr)
2026-05-11 03:10:16,980 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-05-11 03:10:16,981 INFO Round 3 — running backtest: 2022-08-05 → 2025-08-05 (ml_trader, shared ML cache)
2026-05-11 03:10:16,981 INFO Round 3 — QualityScorer hard gate disabled; set BACKTEST_ENABLE_QUALITY_GATE=1 to enable it
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
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
2026-05-11 03:12:03,022 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 03:12:03,886 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-11 03:12:04,833 WARNING _build_sequence_df: HTF frame 5M filled 461 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
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
2026-05-11 03:12:05,396 WARNING _build_sequence_df: HTF frame 5M filled 461 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
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
2026-05-11 03:12:05,536 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:780: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-11 03:12:05,609 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
2026-05-11 03:12:05,654 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
2026-05-11 03:12:05,725 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
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
2026-05-11 03:12:17,354 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
2026-05-11 03:12:17,384 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_ema21_dist"] = _htf_series(df_5m, "5M",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_adx"]      = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:766: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:768: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-11 03:12:17,430 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:758: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
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
2026-05-11 03:12:17,481 WARNING _build_sequence_df: HTF frame 5M filled 453 warmup/alignment gaps with 0.000
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
2026-05-11 03:13:13,356 INFO Round 3 backtest — 16 trades | avg WR=25.0% | avg PF=1.23 | avg Sharpe=1.29
2026-05-11 03:13:13,357 INFO   ml_trader: 16 trades | WR=25.0% | fixed PF=1.23 | Return=2.7% | ExpR=0.170 | DD=4.7% | Sharpe=1.29
2026-05-11 03:13:13,357 INFO   ml_trader gate_diagnostics: bars=403523 no_signal=183417 quality_block=0 session_skip=220090 density=0 pm_reject=0
2026-05-11 03:13:13,357 INFO   ml_trader no_signal_reasons: {'weak_gru_direction': 39849, 'gru_expected_r_below_threshold': 69255, 'no_trade_uncertain': 47009, 'no_trade_chop': 11687, 'no_trade_extreme_vol': 12533, 'wait_pullback': 1892, 'trend_structure_missing': 527, 'tradeability_direction_conflict': 665}
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 16
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (16 rows)
  DONE  Round 3 - Post-retrain backtest (last 3yr)
  Saved Round 3 result → round3_summary.json
  Journal after Round 3: 41 entries

  SKIP  Round 3 Quality+RL retrain — evaluation journals not used for fitting

======================================================================
  BLIND BACKTEST PIPELINE COMPLETE
======================================================================
  Round 1 (train-tail window)   trades=18  WR=22.2%  PF=0.743  Sharpe=-1.963
  Round 2 (blind test)          trades=7  WR=28.6%  PF=1.514  Sharpe=2.486
  Round 3 (last 3yr)            trades=16  WR=25.0%  PF=1.227  Sharpe=1.287


WARNING: GITHUB_TOKEN not set — skipping GitHub push
2026-05-11 03:13:13,586 INFO Round 3: wrote 16 journal entries (total in file: 41)