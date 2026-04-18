=== KAGGLE RETRAIN START ===
Base      : /kaggle/working/Multi-Bot/trading-system
Processed : /kaggle/working/Multi-Bot/trading-system/processed_data
Weights   : /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
Output    : /kaggle/working

✓ Processed data verified

=== RUNNING RETRAIN ===

2026-04-18 10:48:43,428 INFO Loaded M1-resampled AUDUSD: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:48:43,786 INFO Saved AUDUSD: 234948 bars, 17 cols (2016-01-04 → 2026-02-27)
2026-04-18 10:48:43,845 INFO Loaded M1-resampled EURGBP: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:48:44,156 INFO Saved EURGBP: 234979 bars, 17 cols (2016-01-04 → 2026-02-27)
2026-04-18 10:48:44,217 INFO Loaded M1-resampled EURJPY: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:48:44,577 INFO Saved EURJPY: 234916 bars, 17 cols (2016-01-04 → 2026-02-27)
2026-04-18 10:48:44,640 INFO Loaded M1-resampled EURUSD: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:48:44,963 INFO Saved EURUSD: 235026 bars, 17 cols (2016-01-04 → 2026-02-27)
2026-04-18 10:48:45,021 INFO Loaded M1-resampled GBPJPY: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:48:45,366 INFO Saved GBPJPY: 234918 bars, 17 cols (2016-01-04 → 2026-02-27)
2026-04-18 10:48:45,428 INFO Loaded M1-resampled GBPUSD: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:48:45,765 INFO Saved GBPUSD: 234968 bars, 17 cols (2016-01-04 → 2026-02-27)
2026-04-18 10:48:45,814 INFO Loaded M1-resampled NZDUSD: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-18 10:48:46,085 INFO Saved NZDUSD: 174689 bars, 17 cols (2016-01-04 → 2025-08-05)
2026-04-18 10:48:46,143 INFO Loaded M1-resampled USDCAD: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:48:46,484 INFO Saved USDCAD: 234962 bars, 17 cols (2016-01-04 → 2026-02-27)
2026-04-18 10:48:46,549 INFO Loaded M1-resampled USDCHF: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:48:46,873 INFO Saved USDCHF: 234958 bars, 17 cols (2016-01-04 → 2026-02-27)
2026-04-18 10:48:46,932 INFO Loaded M1-resampled USDJPY: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:48:47,248 INFO Saved USDJPY: 234955 bars, 17 cols (2016-01-04 → 2026-02-27)
2026-04-18 10:48:47,328 INFO Loaded M1-resampled XAUUSD: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-18 10:48:47,887 INFO Saved XAUUSD: 401431 bars, 17 cols (2009-03-15 → 2026-03-20)

=== CLEANING COMPLETE ===
  AUDUSD: 234,948 bars  2016-01-04 → 2026-02-27  [ok]
  EURGBP: 234,979 bars  2016-01-04 → 2026-02-27  [ok]
  EURJPY: 234,916 bars  2016-01-04 → 2026-02-27  [ok]
  EURUSD: 235,026 bars  2016-01-04 → 2026-02-27  [ok]
  GBPJPY: 234,918 bars  2016-01-04 → 2026-02-27  [ok]
  GBPUSD: 234,968 bars  2016-01-04 → 2026-02-27  [ok]
  NZDUSD: 174,689 bars  2016-01-04 → 2025-08-05  [ok]
  USDCAD: 234,962 bars  2016-01-04 → 2026-02-27  [ok]
  USDCHF: 234,958 bars  2016-01-04 → 2026-02-27  [ok]
  USDJPY: 234,955 bars  2016-01-04 → 2026-02-27  [ok]
  XAUUSD: 401,431 bars  2009-03-15 → 2026-03-20  [ok]
  Total bars: 2,690,750
2026-04-18 10:48:48,519 INFO Loading anchor: EURUSD
2026-04-18 10:48:48,959 INFO Master index: 221743 bars (2016-01-04 → 2025-08-05) [clipped to common end 2025-08-05]
2026-04-18 10:48:49,046 INFO Added AUDUSD OHLCV columns
2026-04-18 10:48:49,115 INFO Added EURGBP OHLCV columns
2026-04-18 10:48:49,194 INFO Added EURJPY OHLCV columns
2026-04-18 10:48:49,267 INFO Added GBPJPY OHLCV columns
2026-04-18 10:48:49,334 INFO Added GBPUSD OHLCV columns
2026-04-18 10:48:49,403 INFO Added NZDUSD OHLCV columns
2026-04-18 10:48:49,472 INFO Added USDCAD OHLCV columns
2026-04-18 10:48:49,555 INFO Added USDCHF OHLCV columns
2026-04-18 10:48:49,651 INFO Added USDJPY OHLCV columns
2026-04-18 10:48:49,794 INFO Added XAUUSD OHLCV columns
2026-04-18 10:48:50,453 INFO Added context: ASX200
2026-04-18 10:48:51,114 INFO Added context: CAC40
2026-04-18 10:48:51,773 INFO Added context: DAX
2026-04-18 10:48:52,460 INFO Added context: DJIA
2026-04-18 10:48:53,167 INFO Added context: DXY
2026-04-18 10:48:53,831 INFO Added context: EUROSTOXX
2026-04-18 10:48:54,481 INFO Added context: FTSE
2026-04-18 10:48:55,134 INFO Added context: GOLD_FUT
2026-04-18 10:48:55,783 INFO Added context: HSI
2026-04-18 10:48:56,455 INFO Added context: NASDAQ
2026-04-18 10:48:57,123 INFO Added context: NIKKEI
2026-04-18 10:48:57,779 INFO Added context: OIL_FUT
2026-04-18 10:48:58,451 INFO Added context: SPX
2026-04-18 10:48:59,105 INFO Added context: US10Y
2026-04-18 10:48:59,767 INFO Added context: US30Y
2026-04-18 10:49:00,430 INFO Added context: US3M
2026-04-18 10:49:01,108 INFO Added context: VIX
2026-04-18 10:49:01,421 INFO Added fundamental: treasury_10yr → fund_us10y
2026-04-18 10:49:01,748 INFO Added fundamental: treasury_2yr → fund_us2y
2026-04-18 10:49:02,090 INFO Added fundamental: fed_funds_rate → fund_fedfunds
2026-04-18 10:49:02,445 INFO Added fundamental: vix → fund_vix
2026-04-18 10:49:02,799 INFO Aligned frame: 221743 bars, 105 features

=== ALIGNMENT COMPLETE ===
  Bars: 221,743  Features: 105
  Output: /kaggle/working/Multi-Bot/trading-system/processed_data/aligned_multi_asset.parquet
2026-04-18 10:49:04,312 INFO Loading aligned data...
2026-04-18 10:49:04,548 INFO Loaded: 221743 bars, 105 columns
2026-04-18 10:49:04,548 INFO Pass 1/6: technical indicators...
/kaggle/working/Multi-Bot/trading-system/trading-engine/indicators/market_structure.py:299: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  impulse_up = impulse_up.shift(1).infer_objects(copy=False).fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/indicators/market_structure.py:300: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  impulse_dn = impulse_dn.shift(1).infer_objects(copy=False).fillna(False)
2026-04-18 10:49:23,262 INFO Pass 2/6: SMC rolling features...
2026-04-18 10:49:23,329 INFO Pass 3/6: cross-asset features...
2026-04-18 10:49:23,387 INFO Pass 4/6: fundamental features...
2026-04-18 10:49:23,405 INFO Pass 5/6: session features...
2026-04-18 10:49:23,461 INFO Pass 6/6: RL-state features...
/kaggle/working/Multi-Bot/trading-system/pipeline/step4_features.py:178: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = series.values
/kaggle/working/Multi-Bot/trading-system/pipeline/step4_features.py:178: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = series.values
/kaggle/working/Multi-Bot/trading-system/pipeline/step4_features.py:178: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = series.values
2026-04-18 10:49:23,521 INFO Final cleanup...
2026-04-18 10:49:25,055 INFO Writing 221743 bars, 202 features...

=== FEATURE ENGINEERING COMPLETE ===
  Total bars: 221,743
  Total features: 202
  Output: /kaggle/working/Multi-Bot/trading-system/processed_data/feature_engineered.parquet
2026-04-18 10:49:27,845 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-18 10:49:27,845 INFO --- Training regime ---
2026-04-18 10:49:27,845 INFO Running retrain --model regime
2026-04-18 10:49:27,948 INFO retrain environment: KAGGLE
2026-04-18 10:49:29,838 INFO Device: CUDA (2 GPU(s))
2026-04-18 10:49:29,849 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-18 10:49:29,850 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-18 10:49:29,851 INFO === RegimeClassifier retrain (dual-TF cascade: 4H bias + 1H structure) ===
2026-04-18 10:49:30,017 INFO NumExpr defaulting to 4 threads.
2026-04-18 10:49:30,244 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-18 10:49:30,244 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-18 10:49:30,244 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-18 10:49:30,452 INFO Split boundaries loaded — train≤2023-01-06  val≤2024-08-27  test≤2026-02-27
2026-04-18 10:49:30,455 INFO Loaded AUDUSD/1H split=train: 41127 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:30,558 INFO Loaded EURGBP/1H split=train: 41130 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:30,647 INFO Loaded EURJPY/1H split=train: 41128 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:30,733 INFO Loaded EURUSD/1H split=train: 41131 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:30,819 INFO Loaded GBPJPY/1H split=train: 41130 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:30,905 INFO Loaded GBPUSD/1H split=train: 41129 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:30,986 INFO Loaded NZDUSD/1H split=train: 39991 bars (2016-01-04 → 2022-10-28)
2026-04-18 10:49:31,071 INFO Loaded USDCAD/1H split=train: 41132 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:31,159 INFO Loaded USDCHF/1H split=train: 41130 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:31,246 INFO Loaded USDJPY/1H split=train: 41133 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:31,351 INFO Loaded XAUUSD/1H split=train: 83044 bars (2009-03-15 → 2023-01-06)
2026-04-18 10:49:31,427 INFO Regime: fitting per-group GMMs on 4H data (dollar / cross / gold)...
2026-04-18 10:49:31,446 INFO Loaded AUDUSD/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:31,464 INFO Loaded EURGBP/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:31,483 INFO Loaded EURJPY/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:31,500 INFO Loaded EURUSD/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:31,518 INFO Loaded GBPJPY/4H split=train: 10553 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:31,535 INFO Loaded GBPUSD/4H split=train: 10553 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:31,553 INFO Loaded NZDUSD/4H split=train: 10260 bars (2016-01-04 → 2022-10-28)
2026-04-18 10:49:31,570 INFO Loaded USDCAD/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:31,587 INFO Loaded USDCHF/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:31,604 INFO Loaded USDJPY/4H split=train: 10553 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:31,625 INFO Loaded XAUUSD/4H split=train: 22089 bars (2009-03-15 → 2023-01-06)
2026-04-18 10:49:32,928 INFO GMM fit: timeframe=4H → n_bar=50
2026-04-18 10:49:39,769 INFO GMM fitted on 73217 samples — cluster→regime: {0: 2, 1: 1, 2: 3, 3: 0} dist: {'TRENDING_UP': 1, 'TRENDING_DOWN': 1, 'RANGING': 1, 'VOLATILE': 1}
2026-04-18 10:49:39,771 INFO Regime: GMM 'dollar' fitted on 7 4H dfs (n_bar=50)
2026-04-18 10:49:39,772 INFO GMM fit: timeframe=4H → n_bar=50
2026-04-18 10:49:42,201 INFO GMM fitted on 31504 samples — cluster→regime: {0: 0, 1: 2, 2: 3, 3: 2} dist: {'TRENDING_UP': 1, 'TRENDING_DOWN': 0, 'RANGING': 2, 'VOLATILE': 1}
2026-04-18 10:49:42,201 INFO Regime: GMM 'cross' fitted on 3 4H dfs (n_bar=50)
2026-04-18 10:49:42,201 INFO GMM fit: timeframe=4H → n_bar=50
2026-04-18 10:49:42,755 INFO GMM fitted on 11019 samples — cluster→regime: {0: 2, 1: 3, 2: 2, 3: 0} dist: {'TRENDING_UP': 1, 'TRENDING_DOWN': 0, 'RANGING': 2, 'VOLATILE': 1}
2026-04-18 10:49:42,755 INFO Regime: GMM 'gold' fitted on 1 4H dfs (n_bar=50)
2026-04-18 10:49:42,873 INFO Regime: training 4H bias classifier...
2026-04-18 10:49:42,875 INFO Loaded AUDUSD/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:42,876 INFO Loaded EURGBP/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:42,877 INFO Loaded EURJPY/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:42,878 INFO Loaded EURUSD/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:42,879 INFO Loaded GBPJPY/4H split=train: 10553 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:42,880 INFO Loaded GBPUSD/4H split=train: 10553 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:42,881 INFO Loaded NZDUSD/4H split=train: 10260 bars (2016-01-04 → 2022-10-28)
2026-04-18 10:49:42,882 INFO Loaded USDCAD/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:42,883 INFO Loaded USDCHF/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:42,884 INFO Loaded USDJPY/4H split=train: 10553 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:42,885 INFO Loaded XAUUSD/4H split=train: 22089 bars (2009-03-15 → 2023-01-06)
2026-04-18 10:49:43,022 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:43,067 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:43,068 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:43,069 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:43,077 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:43,079 INFO Loaded AUDUSD/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:43,627 INFO Rule labels [4H]: {'TRENDING_UP': 1930, 'TRENDING_DOWN': 1931, 'RANGING': 4580, 'VOLATILE': 2111}  ambiguous(conf<0.4)=2607 (total=10552)
2026-04-18 10:49:43,629 INFO Regime[4H]: collected AUDUSD — 10502 samples (group=dollar)
2026-04-18 10:49:43,837 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:43,876 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:43,877 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:43,878 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:43,887 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:43,888 INFO Loaded EURGBP/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:44,397 INFO Rule labels [4H]: {'TRENDING_UP': 1847, 'TRENDING_DOWN': 1998, 'RANGING': 4586, 'VOLATILE': 2121}  ambiguous(conf<0.4)=2734 (total=10552)
2026-04-18 10:49:44,399 INFO Regime[4H]: collected EURGBP — 10502 samples (group=cross)
2026-04-18 10:49:44,598 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:44,637 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:44,638 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:44,639 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:44,648 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:44,649 INFO Loaded EURJPY/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:45,143 INFO Rule labels [4H]: {'TRENDING_UP': 2157, 'TRENDING_DOWN': 1757, 'RANGING': 4507, 'VOLATILE': 2131}  ambiguous(conf<0.4)=2584 (total=10552)
2026-04-18 10:49:45,145 INFO Regime[4H]: collected EURJPY — 10502 samples (group=cross)
2026-04-18 10:49:45,337 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:45,379 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:45,380 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:45,381 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:45,390 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:45,391 INFO Loaded EURUSD/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:45,898 INFO Rule labels [4H]: {'TRENDING_UP': 1644, 'TRENDING_DOWN': 2108, 'RANGING': 4678, 'VOLATILE': 2122}  ambiguous(conf<0.4)=2719 (total=10552)
2026-04-18 10:49:45,900 INFO Regime[4H]: collected EURUSD — 10502 samples (group=dollar)
2026-04-18 10:49:46,111 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:46,152 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:46,153 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:46,153 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:46,162 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:46,163 INFO Loaded GBPJPY/4H split=train: 10553 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:46,657 INFO Rule labels [4H]: {'TRENDING_UP': 2031, 'TRENDING_DOWN': 1747, 'RANGING': 4653, 'VOLATILE': 2122}  ambiguous(conf<0.4)=2668 (total=10553)
2026-04-18 10:49:46,659 INFO Regime[4H]: collected GBPJPY — 10503 samples (group=cross)
2026-04-18 10:49:46,865 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:46,903 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:46,904 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:46,904 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:46,914 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:46,915 INFO Loaded GBPUSD/4H split=train: 10553 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:47,410 INFO Rule labels [4H]: {'TRENDING_UP': 1841, 'TRENDING_DOWN': 2017, 'RANGING': 4529, 'VOLATILE': 2166}  ambiguous(conf<0.4)=2643 (total=10553)
2026-04-18 10:49:47,411 INFO Regime[4H]: collected GBPUSD — 10503 samples (group=dollar)
2026-04-18 10:49:47,596 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-18 10:49:47,629 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-18 10:49:47,630 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-18 10:49:47,630 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-18 10:49:47,638 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-18 10:49:47,639 INFO Loaded NZDUSD/4H split=train: 10260 bars (2016-01-04 → 2022-10-28)
2026-04-18 10:49:48,115 INFO Rule labels [4H]: {'TRENDING_UP': 1780, 'TRENDING_DOWN': 1985, 'RANGING': 4419, 'VOLATILE': 2076}  ambiguous(conf<0.4)=2697 (total=10260)
2026-04-18 10:49:48,116 INFO Regime[4H]: collected NZDUSD — 10210 samples (group=dollar)
2026-04-18 10:49:48,309 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:48,345 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:48,346 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:48,347 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:48,356 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:48,357 INFO Loaded USDCAD/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:48,844 INFO Rule labels [4H]: {'TRENDING_UP': 1779, 'TRENDING_DOWN': 2128, 'RANGING': 4525, 'VOLATILE': 2120}  ambiguous(conf<0.4)=2545 (total=10552)
2026-04-18 10:49:48,846 INFO Regime[4H]: collected USDCAD — 10502 samples (group=dollar)
2026-04-18 10:49:49,044 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:49,081 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:49,082 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:49,083 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:49,092 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:49,093 INFO Loaded USDCHF/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:49,570 INFO Rule labels [4H]: {'TRENDING_UP': 2032, 'TRENDING_DOWN': 1900, 'RANGING': 4460, 'VOLATILE': 2160}  ambiguous(conf<0.4)=2487 (total=10552)
2026-04-18 10:49:49,571 INFO Regime[4H]: collected USDCHF — 10502 samples (group=dollar)
2026-04-18 10:49:49,770 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:49,810 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:49,811 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:49,812 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:49,821 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:49:49,822 INFO Loaded USDJPY/4H split=train: 10553 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:49:50,304 INFO Rule labels [4H]: {'TRENDING_UP': 2348, 'TRENDING_DOWN': 1607, 'RANGING': 4451, 'VOLATILE': 2147}  ambiguous(conf<0.4)=2524 (total=10553)
2026-04-18 10:49:50,305 INFO Regime[4H]: collected USDJPY — 10503 samples (group=dollar)
2026-04-18 10:49:50,606 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-18 10:49:50,671 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-18 10:49:50,672 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-18 10:49:50,672 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-18 10:49:50,684 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-18 10:49:50,685 INFO Loaded XAUUSD/4H split=train: 22089 bars (2009-03-15 → 2023-01-06)
2026-04-18 10:49:51,606 INFO Rule labels [4H]: {'TRENDING_UP': 4634, 'TRENDING_DOWN': 4090, 'RANGING': 8914, 'VOLATILE': 4451}  ambiguous(conf<0.4)=4901 (total=22089)
2026-04-18 10:49:51,608 INFO Regime[4H]: collected XAUUSD — 22039 samples (group=gold)
2026-04-18 10:49:51,818 INFO RegimeClassifier: 126770 samples, classes={'TRENDING_UP': 24023, 'TRENDING_DOWN': 23268, 'RANGING': 53752, 'VOLATILE': 25727}, device=cuda
2026-04-18 10:49:51,819 INFO RegimeClassifier: sample weights — mean=0.562  ambiguous(<0.4)=24.6%
2026-04-18 10:49:52,096 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-18 10:49:57,332 INFO Regime epoch  1/50 — tr=0.8693 va=1.4499 acc=0.269 per_class={'TRENDING_UP': 0.266, 'TRENDING_DOWN': 0.03, 'RANGING': 0.446, 'VOLATILE': 0.137}
2026-04-18 10:49:57,575 INFO Regime epoch  2/50 — tr=0.8539 va=1.3999 acc=0.313
2026-04-18 10:49:57,808 INFO Regime epoch  3/50 — tr=0.8270 va=1.3435 acc=0.382
2026-04-18 10:49:58,037 INFO Regime epoch  4/50 — tr=0.7899 va=1.2812 acc=0.409
2026-04-18 10:49:58,298 INFO Regime epoch  5/50 — tr=0.7486 va=1.2180 acc=0.437 per_class={'TRENDING_UP': 0.538, 'TRENDING_DOWN': 0.489, 'RANGING': 0.165, 'VOLATILE': 0.803}
2026-04-18 10:49:58,544 INFO Regime epoch  6/50 — tr=0.7144 va=1.1569 acc=0.462
2026-04-18 10:49:58,802 INFO Regime epoch  7/50 — tr=0.6915 va=1.1190 acc=0.476
2026-04-18 10:49:59,048 INFO Regime epoch  8/50 — tr=0.6753 va=1.0911 acc=0.487
2026-04-18 10:49:59,289 INFO Regime epoch  9/50 — tr=0.6652 va=1.0774 acc=0.490
2026-04-18 10:49:59,546 INFO Regime epoch 10/50 — tr=0.6565 va=1.0698 acc=0.490 per_class={'TRENDING_UP': 0.692, 'TRENDING_DOWN': 0.627, 'RANGING': 0.175, 'VOLATILE': 0.76}
2026-04-18 10:49:59,794 INFO Regime epoch 11/50 — tr=0.6497 va=1.0671 acc=0.487
2026-04-18 10:50:00,040 INFO Regime epoch 12/50 — tr=0.6442 va=1.0663 acc=0.488
2026-04-18 10:50:00,302 INFO Regime epoch 13/50 — tr=0.6404 va=1.0564 acc=0.491
2026-04-18 10:50:00,545 INFO Regime epoch 14/50 — tr=0.6371 va=1.0575 acc=0.491
2026-04-18 10:50:00,815 INFO Regime epoch 15/50 — tr=0.6332 va=1.0492 acc=0.494 per_class={'TRENDING_UP': 0.676, 'TRENDING_DOWN': 0.62, 'RANGING': 0.18, 'VOLATILE': 0.793}
2026-04-18 10:50:01,054 INFO Regime epoch 16/50 — tr=0.6314 va=1.0542 acc=0.487
2026-04-18 10:50:01,301 INFO Regime epoch 17/50 — tr=0.6287 va=1.0520 acc=0.486
2026-04-18 10:50:01,536 INFO Regime epoch 18/50 — tr=0.6260 va=1.0491 acc=0.488
2026-04-18 10:50:01,794 INFO Regime epoch 19/50 — tr=0.6248 va=1.0515 acc=0.485
2026-04-18 10:50:02,068 INFO Regime epoch 20/50 — tr=0.6230 va=1.0473 acc=0.488 per_class={'TRENDING_UP': 0.654, 'TRENDING_DOWN': 0.592, 'RANGING': 0.178, 'VOLATILE': 0.817}
2026-04-18 10:50:02,344 INFO Regime epoch 21/50 — tr=0.6215 va=1.0456 acc=0.488
2026-04-18 10:50:02,597 INFO Regime epoch 22/50 — tr=0.6203 va=1.0431 acc=0.488
2026-04-18 10:50:02,868 INFO Regime epoch 23/50 — tr=0.6196 va=1.0404 acc=0.491
2026-04-18 10:50:03,102 INFO Regime epoch 24/50 — tr=0.6179 va=1.0422 acc=0.488
2026-04-18 10:50:03,379 INFO Regime epoch 25/50 — tr=0.6170 va=1.0472 acc=0.484 per_class={'TRENDING_UP': 0.641, 'TRENDING_DOWN': 0.576, 'RANGING': 0.174, 'VOLATILE': 0.831}
2026-04-18 10:50:03,634 INFO Regime epoch 26/50 — tr=0.6166 va=1.0443 acc=0.485
2026-04-18 10:50:03,887 INFO Regime epoch 27/50 — tr=0.6159 va=1.0392 acc=0.487
2026-04-18 10:50:04,119 INFO Regime epoch 28/50 — tr=0.6151 va=1.0369 acc=0.488
2026-04-18 10:50:04,359 INFO Regime epoch 29/50 — tr=0.6146 va=1.0387 acc=0.487
2026-04-18 10:50:04,639 INFO Regime epoch 30/50 — tr=0.6144 va=1.0419 acc=0.484 per_class={'TRENDING_UP': 0.643, 'TRENDING_DOWN': 0.576, 'RANGING': 0.171, 'VOLATILE': 0.833}
2026-04-18 10:50:04,902 INFO Regime epoch 31/50 — tr=0.6130 va=1.0394 acc=0.485
2026-04-18 10:50:05,155 INFO Regime epoch 32/50 — tr=0.6129 va=1.0453 acc=0.482
2026-04-18 10:50:05,388 INFO Regime epoch 33/50 — tr=0.6121 va=1.0457 acc=0.482
2026-04-18 10:50:05,632 INFO Regime epoch 34/50 — tr=0.6122 va=1.0442 acc=0.482
2026-04-18 10:50:05,908 INFO Regime epoch 35/50 — tr=0.6114 va=1.0391 acc=0.485 per_class={'TRENDING_UP': 0.646, 'TRENDING_DOWN': 0.574, 'RANGING': 0.172, 'VOLATILE': 0.833}
2026-04-18 10:50:06,160 INFO Regime epoch 36/50 — tr=0.6115 va=1.0384 acc=0.485
2026-04-18 10:50:06,403 INFO Regime epoch 37/50 — tr=0.6108 va=1.0401 acc=0.486
2026-04-18 10:50:06,653 INFO Regime epoch 38/50 — tr=0.6107 va=1.0353 acc=0.488
2026-04-18 10:50:06,900 INFO Regime epoch 39/50 — tr=0.6102 va=1.0396 acc=0.485
2026-04-18 10:50:07,169 INFO Regime epoch 40/50 — tr=0.6110 va=1.0405 acc=0.484 per_class={'TRENDING_UP': 0.638, 'TRENDING_DOWN': 0.571, 'RANGING': 0.173, 'VOLATILE': 0.84}
2026-04-18 10:50:07,427 INFO Regime epoch 41/50 — tr=0.6111 va=1.0349 acc=0.487
2026-04-18 10:50:07,678 INFO Regime epoch 42/50 — tr=0.6104 va=1.0379 acc=0.485
2026-04-18 10:50:07,934 INFO Regime epoch 43/50 — tr=0.6105 va=1.0406 acc=0.483
2026-04-18 10:50:08,174 INFO Regime epoch 44/50 — tr=0.6104 va=1.0370 acc=0.486
2026-04-18 10:50:08,426 INFO Regime epoch 45/50 — tr=0.6107 va=1.0368 acc=0.487 per_class={'TRENDING_UP': 0.647, 'TRENDING_DOWN': 0.577, 'RANGING': 0.174, 'VOLATILE': 0.835}
2026-04-18 10:50:08,660 INFO Regime epoch 46/50 — tr=0.6100 va=1.0371 acc=0.486
2026-04-18 10:50:08,893 INFO Regime epoch 47/50 — tr=0.6097 va=1.0409 acc=0.483
2026-04-18 10:50:09,125 INFO Regime epoch 48/50 — tr=0.6098 va=1.0363 acc=0.486
2026-04-18 10:50:09,354 INFO Regime epoch 49/50 — tr=0.6100 va=1.0423 acc=0.482
2026-04-18 10:50:09,611 INFO Regime epoch 50/50 — tr=0.6101 va=1.0381 acc=0.485 per_class={'TRENDING_UP': 0.644, 'TRENDING_DOWN': 0.573, 'RANGING': 0.171, 'VOLATILE': 0.837}
2026-04-18 10:50:09,631 WARNING RegimeClassifier accuracy 0.49 < 0.65 threshold
2026-04-18 10:50:09,636 INFO RegimeClassifier saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_4h.pkl
2026-04-18 10:50:09,636 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_4h.pkl
2026-04-18 10:50:09,785 INFO Regime 4H complete: acc=0.487, n=126770
2026-04-18 10:50:09,787 INFO Loaded XAUUSD/4H split=train: 22089 bars (2009-03-15 → 2023-01-06)
2026-04-18 10:50:09,966 INFO Rule labels [4H]: {'TRENDING_UP': 4634, 'TRENDING_DOWN': 4090, 'RANGING': 8914, 'VOLATILE': 4451}  ambiguous(conf<0.4)=4901 (total=22089)
2026-04-18 10:50:09,972 INFO Regime[4H] persistence (avg bars/run) on XAUUSD 4H:
{0: 9.594202898550725, 1: 9.855421686746988, 2: 10.844282238442823, 3: 10.047404063205418}
2026-04-18 10:50:09,974 INFO Regime[4H] return separation on XAUUSD 4H:
{0: 8.866127244754663e-05, 1: 1.568307252883805e-05, 2: 2.1998476665049278e-05, 3: 4.4727359348779954e-05}
2026-04-18 10:50:09,974 INFO Regime: training 1H structure classifier...
2026-04-18 10:50:09,975 INFO Loaded AUDUSD/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:50:09,976 INFO Loaded EURGBP/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:50:09,977 INFO Loaded EURJPY/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:50:09,978 INFO Loaded EURUSD/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:50:09,979 INFO Loaded GBPJPY/4H split=train: 10553 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:50:09,980 INFO Loaded GBPUSD/4H split=train: 10553 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:50:09,981 INFO Loaded NZDUSD/4H split=train: 10260 bars (2016-01-04 → 2022-10-28)
2026-04-18 10:50:09,982 INFO Loaded USDCAD/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:50:09,983 INFO Loaded USDCHF/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:50:09,984 INFO Loaded USDJPY/4H split=train: 10553 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:50:09,986 INFO Loaded XAUUSD/4H split=train: 22089 bars (2009-03-15 → 2023-01-06)
2026-04-18 10:50:09,992 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:09,994 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:09,995 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:09,996 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:09,996 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:09,999 INFO Loaded AUDUSD/1H split=train: 41127 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:50:10,584 INFO Rule labels [4H]: {'TRENDING_UP': 1930, 'TRENDING_DOWN': 1931, 'RANGING': 4580, 'VOLATILE': 2111}  ambiguous(conf<0.4)=2607 (total=10552)
2026-04-18 10:50:10,589 INFO Regime[1H]: collected AUDUSD — 41077 samples (group=dollar)
2026-04-18 10:50:10,742 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:10,744 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:10,745 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:10,745 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:10,746 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:10,748 INFO Loaded EURGBP/1H split=train: 41130 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:50:11,282 INFO Rule labels [4H]: {'TRENDING_UP': 1847, 'TRENDING_DOWN': 1998, 'RANGING': 4586, 'VOLATILE': 2121}  ambiguous(conf<0.4)=2734 (total=10552)
2026-04-18 10:50:11,288 INFO Regime[1H]: collected EURGBP — 41080 samples (group=cross)
2026-04-18 10:50:11,428 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:11,430 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:11,431 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:11,431 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:11,431 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:11,434 INFO Loaded EURJPY/1H split=train: 41128 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:50:11,985 INFO Rule labels [4H]: {'TRENDING_UP': 2157, 'TRENDING_DOWN': 1757, 'RANGING': 4507, 'VOLATILE': 2131}  ambiguous(conf<0.4)=2584 (total=10552)
2026-04-18 10:50:11,990 INFO Regime[1H]: collected EURJPY — 41078 samples (group=cross)
2026-04-18 10:50:12,148 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:12,150 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:12,151 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:12,152 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:12,152 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:12,154 INFO Loaded EURUSD/1H split=train: 41131 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:50:12,731 INFO Rule labels [4H]: {'TRENDING_UP': 1644, 'TRENDING_DOWN': 2108, 'RANGING': 4678, 'VOLATILE': 2122}  ambiguous(conf<0.4)=2719 (total=10552)
2026-04-18 10:50:12,737 INFO Regime[1H]: collected EURUSD — 41081 samples (group=dollar)
2026-04-18 10:50:12,882 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:12,884 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:12,885 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:12,886 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:12,886 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:12,888 INFO Loaded GBPJPY/1H split=train: 41130 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:50:13,445 INFO Rule labels [4H]: {'TRENDING_UP': 2031, 'TRENDING_DOWN': 1747, 'RANGING': 4653, 'VOLATILE': 2122}  ambiguous(conf<0.4)=2668 (total=10553)
2026-04-18 10:50:13,450 INFO Regime[1H]: collected GBPJPY — 41080 samples (group=cross)
2026-04-18 10:50:13,604 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:13,607 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:13,608 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:13,608 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:13,609 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:13,611 INFO Loaded GBPUSD/1H split=train: 41129 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:50:14,163 INFO Rule labels [4H]: {'TRENDING_UP': 1841, 'TRENDING_DOWN': 2017, 'RANGING': 4529, 'VOLATILE': 2166}  ambiguous(conf<0.4)=2643 (total=10553)
2026-04-18 10:50:14,169 INFO Regime[1H]: collected GBPUSD — 41079 samples (group=dollar)
2026-04-18 10:50:14,319 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-18 10:50:14,321 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-18 10:50:14,322 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-18 10:50:14,322 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-18 10:50:14,322 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-18 10:50:14,324 INFO Loaded NZDUSD/1H split=train: 39991 bars (2016-01-04 → 2022-10-28)
2026-04-18 10:50:14,844 INFO Rule labels [4H]: {'TRENDING_UP': 1780, 'TRENDING_DOWN': 1985, 'RANGING': 4419, 'VOLATILE': 2076}  ambiguous(conf<0.4)=2697 (total=10260)
2026-04-18 10:50:14,849 INFO Regime[1H]: collected NZDUSD — 39941 samples (group=dollar)
2026-04-18 10:50:15,007 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:15,009 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:15,010 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:15,011 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:15,011 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:15,013 INFO Loaded USDCAD/1H split=train: 41132 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:50:15,558 INFO Rule labels [4H]: {'TRENDING_UP': 1779, 'TRENDING_DOWN': 2128, 'RANGING': 4525, 'VOLATILE': 2120}  ambiguous(conf<0.4)=2545 (total=10552)
2026-04-18 10:50:15,564 INFO Regime[1H]: collected USDCAD — 41082 samples (group=dollar)
2026-04-18 10:50:15,717 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:15,720 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:15,721 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:15,721 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:15,721 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:15,724 INFO Loaded USDCHF/1H split=train: 41130 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:50:16,278 INFO Rule labels [4H]: {'TRENDING_UP': 2032, 'TRENDING_DOWN': 1900, 'RANGING': 4460, 'VOLATILE': 2160}  ambiguous(conf<0.4)=2487 (total=10552)
2026-04-18 10:50:16,283 INFO Regime[1H]: collected USDCHF — 41080 samples (group=dollar)
2026-04-18 10:50:16,435 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:16,437 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:16,438 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:16,439 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:16,439 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:50:16,441 INFO Loaded USDJPY/1H split=train: 41133 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:50:16,984 INFO Rule labels [4H]: {'TRENDING_UP': 2348, 'TRENDING_DOWN': 1607, 'RANGING': 4451, 'VOLATILE': 2147}  ambiguous(conf<0.4)=2524 (total=10553)
2026-04-18 10:50:16,990 INFO Regime[1H]: collected USDJPY — 41083 samples (group=dollar)
2026-04-18 10:50:17,151 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-18 10:50:17,154 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-18 10:50:17,155 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-18 10:50:17,156 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-18 10:50:17,156 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-18 10:50:17,160 INFO Loaded XAUUSD/1H split=train: 83044 bars (2009-03-15 → 2023-01-06)
2026-04-18 10:50:18,195 INFO Rule labels [4H]: {'TRENDING_UP': 4634, 'TRENDING_DOWN': 4090, 'RANGING': 8914, 'VOLATILE': 4451}  ambiguous(conf<0.4)=4901 (total=22089)
2026-04-18 10:50:18,205 INFO Regime[1H]: collected XAUUSD — 82994 samples (group=gold)
2026-04-18 10:50:18,626 INFO RegimeClassifier: 492655 samples, classes={'TRENDING_UP': 92836, 'TRENDING_DOWN': 89750, 'RANGING': 210125, 'VOLATILE': 99944}, device=cuda
2026-04-18 10:50:18,628 INFO RegimeClassifier: sample weights — mean=0.560  ambiguous(<0.4)=24.9%
2026-04-18 10:50:18,630 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-18 10:50:19,654 INFO Regime epoch  1/50 — tr=0.8200 va=1.3332 acc=0.439 per_class={'TRENDING_UP': 0.6, 'TRENDING_DOWN': 0.315, 'RANGING': 0.401, 'VOLATILE': 0.449}
2026-04-18 10:50:20,570 INFO Regime epoch  2/50 — tr=0.7775 va=1.2734 acc=0.432
2026-04-18 10:50:21,491 INFO Regime epoch  3/50 — tr=0.7318 va=1.2215 acc=0.430
2026-04-18 10:50:22,414 INFO Regime epoch  4/50 — tr=0.6988 va=1.1544 acc=0.433
2026-04-18 10:50:23,441 INFO Regime epoch  5/50 — tr=0.6796 va=1.1228 acc=0.426 per_class={'TRENDING_UP': 0.636, 'TRENDING_DOWN': 0.501, 'RANGING': 0.121, 'VOLATILE': 0.736}
2026-04-18 10:50:24,417 INFO Regime epoch  6/50 — tr=0.6663 va=1.1045 acc=0.422
2026-04-18 10:50:25,319 INFO Regime epoch  7/50 — tr=0.6566 va=1.0857 acc=0.425
2026-04-18 10:50:26,284 INFO Regime epoch  8/50 — tr=0.6504 va=1.0801 acc=0.421
2026-04-18 10:50:27,241 INFO Regime epoch  9/50 — tr=0.6458 va=1.0786 acc=0.419
2026-04-18 10:50:28,236 INFO Regime epoch 10/50 — tr=0.6425 va=1.0789 acc=0.415 per_class={'TRENDING_UP': 0.571, 'TRENDING_DOWN': 0.504, 'RANGING': 0.11, 'VOLATILE': 0.769}
2026-04-18 10:50:29,138 INFO Regime epoch 11/50 — tr=0.6398 va=1.0815 acc=0.413
2026-04-18 10:50:30,036 INFO Regime epoch 12/50 — tr=0.6379 va=1.0786 acc=0.412
2026-04-18 10:50:31,012 INFO Regime epoch 13/50 — tr=0.6360 va=1.0778 acc=0.412
2026-04-18 10:50:31,922 INFO Regime epoch 14/50 — tr=0.6348 va=1.0775 acc=0.412
2026-04-18 10:50:33,000 INFO Regime epoch 15/50 — tr=0.6340 va=1.0841 acc=0.407 per_class={'TRENDING_UP': 0.545, 'TRENDING_DOWN': 0.477, 'RANGING': 0.104, 'VOLATILE': 0.793}
2026-04-18 10:50:33,882 INFO Regime epoch 16/50 — tr=0.6325 va=1.0838 acc=0.408
2026-04-18 10:50:34,775 INFO Regime epoch 17/50 — tr=0.6321 va=1.0867 acc=0.406
2026-04-18 10:50:35,729 INFO Regime epoch 18/50 — tr=0.6312 va=1.0822 acc=0.410
2026-04-18 10:50:36,689 INFO Regime epoch 19/50 — tr=0.6304 va=1.0781 acc=0.412
2026-04-18 10:50:37,651 INFO Regime epoch 20/50 — tr=0.6299 va=1.0751 acc=0.415 per_class={'TRENDING_UP': 0.554, 'TRENDING_DOWN': 0.503, 'RANGING': 0.108, 'VOLATILE': 0.791}
2026-04-18 10:50:38,585 INFO Regime epoch 21/50 — tr=0.6289 va=1.0743 acc=0.416
2026-04-18 10:50:39,496 INFO Regime epoch 22/50 — tr=0.6289 va=1.0756 acc=0.415
2026-04-18 10:50:40,440 INFO Regime epoch 23/50 — tr=0.6282 va=1.0740 acc=0.415
2026-04-18 10:50:41,370 INFO Regime epoch 24/50 — tr=0.6278 va=1.0726 acc=0.416
2026-04-18 10:50:42,402 INFO Regime epoch 25/50 — tr=0.6270 va=1.0742 acc=0.415 per_class={'TRENDING_UP': 0.546, 'TRENDING_DOWN': 0.509, 'RANGING': 0.109, 'VOLATILE': 0.794}
2026-04-18 10:50:43,337 INFO Regime epoch 26/50 — tr=0.6267 va=1.0796 acc=0.412
2026-04-18 10:50:44,303 INFO Regime epoch 27/50 — tr=0.6261 va=1.0734 acc=0.416
2026-04-18 10:50:45,237 INFO Regime epoch 28/50 — tr=0.6258 va=1.0742 acc=0.414
2026-04-18 10:50:46,175 INFO Regime epoch 29/50 — tr=0.6258 va=1.0721 acc=0.416
2026-04-18 10:50:47,200 INFO Regime epoch 30/50 — tr=0.6253 va=1.0715 acc=0.417 per_class={'TRENDING_UP': 0.537, 'TRENDING_DOWN': 0.508, 'RANGING': 0.117, 'VOLATILE': 0.798}
2026-04-18 10:50:48,138 INFO Regime epoch 31/50 — tr=0.6251 va=1.0704 acc=0.416
2026-04-18 10:50:49,106 INFO Regime epoch 32/50 — tr=0.6248 va=1.0718 acc=0.417
2026-04-18 10:50:50,074 INFO Regime epoch 33/50 — tr=0.6248 va=1.0731 acc=0.415
2026-04-18 10:50:51,010 INFO Regime epoch 34/50 — tr=0.6246 va=1.0712 acc=0.415
2026-04-18 10:50:51,979 INFO Regime epoch 35/50 — tr=0.6246 va=1.0737 acc=0.414 per_class={'TRENDING_UP': 0.54, 'TRENDING_DOWN': 0.49, 'RANGING': 0.114, 'VOLATILE': 0.801}
2026-04-18 10:50:52,920 INFO Regime epoch 36/50 — tr=0.6242 va=1.0654 acc=0.420
2026-04-18 10:50:53,838 INFO Regime epoch 37/50 — tr=0.6241 va=1.0722 acc=0.416
2026-04-18 10:50:54,741 INFO Regime epoch 38/50 — tr=0.6241 va=1.0705 acc=0.418
2026-04-18 10:50:55,630 INFO Regime epoch 39/50 — tr=0.6239 va=1.0711 acc=0.417
2026-04-18 10:50:56,634 INFO Regime epoch 40/50 — tr=0.6241 va=1.0666 acc=0.420 per_class={'TRENDING_UP': 0.549, 'TRENDING_DOWN': 0.514, 'RANGING': 0.118, 'VOLATILE': 0.794}
2026-04-18 10:50:57,555 INFO Regime epoch 41/50 — tr=0.6238 va=1.0667 acc=0.420
2026-04-18 10:50:58,521 INFO Regime epoch 42/50 — tr=0.6240 va=1.0684 acc=0.419
2026-04-18 10:50:59,445 INFO Regime epoch 43/50 — tr=0.6238 va=1.0673 acc=0.420
2026-04-18 10:51:00,355 INFO Regime epoch 44/50 — tr=0.6239 va=1.0732 acc=0.416
2026-04-18 10:51:01,387 INFO Regime epoch 45/50 — tr=0.6239 va=1.0704 acc=0.417 per_class={'TRENDING_UP': 0.545, 'TRENDING_DOWN': 0.496, 'RANGING': 0.118, 'VOLATILE': 0.797}
2026-04-18 10:51:02,352 INFO Regime epoch 46/50 — tr=0.6238 va=1.0701 acc=0.417
2026-04-18 10:51:02,352 INFO Regime early stop at epoch 46 (no_improve=10)
2026-04-18 10:51:02,415 WARNING RegimeClassifier accuracy 0.42 < 0.65 threshold
2026-04-18 10:51:02,418 INFO RegimeClassifier saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_1h.pkl
2026-04-18 10:51:02,418 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_1h.pkl
2026-04-18 10:51:02,566 INFO Regime 1H complete: acc=0.420, n=492655
2026-04-18 10:51:02,570 INFO Loaded XAUUSD/1H split=train: 83044 bars (2009-03-15 → 2023-01-06)
2026-04-18 10:51:03,166 INFO Rule labels [1H]: {'TRENDING_UP': 13699, 'TRENDING_DOWN': 11021, 'RANGING': 40665, 'VOLATILE': 17659}  ambiguous(conf<0.4)=23637 (total=83044)
2026-04-18 10:51:03,172 INFO Regime[1H] persistence (avg bars/run) on XAUUSD 1H:
{0: 7.665920537213206, 1: 7.024219247928617, 2: 11.592075256556443, 3: 6.881917381137958}
2026-04-18 10:51:03,176 INFO Regime[1H] return separation on XAUUSD 1H:
{0: 3.534735734906192e-05, 1: -1.4422020587816898e-05, 2: 6.460844726243422e-06, 3: 1.5880900631789552e-05}
2026-04-18 10:51:03,190 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-18 10:51:03,190 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-18 10:51:03,190 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-18 10:51:03,190 INFO === VectorStore: building similarity indices ===
2026-04-18 10:51:03,191 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-18 10:51:03,191 INFO Retrain complete.
2026-04-18 10:51:05,772 INFO Model regime: SUCCESS
2026-04-18 10:51:05,772 INFO --- Training gru ---
2026-04-18 10:51:05,772 INFO Running retrain --model gru
2026-04-18 10:51:05,878 INFO retrain environment: KAGGLE
2026-04-18 10:51:07,810 INFO Device: CUDA (2 GPU(s))
2026-04-18 10:51:07,821 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-18 10:51:07,821 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-18 10:51:07,822 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-18 10:51:07,981 INFO NumExpr defaulting to 4 threads.
2026-04-18 10:51:08,219 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-18 10:51:08,219 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-18 10:51:08,219 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-18 10:51:08,442 INFO Split boundaries loaded — train≤2023-01-06  val≤2024-08-27  test≤2026-02-27
2026-04-18 10:51:08,445 INFO Loaded AUDUSD/1H split=train: 41127 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:08,536 INFO Loaded EURGBP/1H split=train: 41130 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:08,628 INFO Loaded EURJPY/1H split=train: 41128 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:08,716 INFO Loaded EURUSD/1H split=train: 41131 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:08,803 INFO Loaded GBPJPY/1H split=train: 41130 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:08,892 INFO Loaded GBPUSD/1H split=train: 41129 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:08,979 INFO Loaded NZDUSD/1H split=train: 39991 bars (2016-01-04 → 2022-10-28)
2026-04-18 10:51:09,066 INFO Loaded USDCAD/1H split=train: 41132 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:09,157 INFO Loaded USDCHF/1H split=train: 41130 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:09,244 INFO Loaded USDJPY/1H split=train: 41133 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:09,350 INFO Loaded XAUUSD/1H split=train: 83044 bars (2009-03-15 → 2023-01-06)
2026-04-18 10:51:09,423 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-18 10:51:09,424 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260418_105109
2026-04-18 10:51:09,427 WARNING WeightsManifest: no manifest at /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json — assuming compatible (run a full retrain to generate one)
2026-04-18 10:51:09,565 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:09,565 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:09,582 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:09,590 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:09,591 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-18 10:51:09,592 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-18 10:51:09,592 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-18 10:51:09,593 INFO Loaded AUDUSD/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:09,693 INFO Rule labels [4H]: {'TRENDING_UP': 1930, 'TRENDING_DOWN': 1931, 'RANGING': 4580, 'VOLATILE': 2111}  ambiguous(conf<0.4)=2607 (total=10552)
2026-04-18 10:51:09,696 INFO Loaded AUDUSD/1H split=train: 41127 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:09,733 INFO Loaded AUDUSD/5M split=train: 493421 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:10,084 INFO Loaded AUDUSD/15M split=train: 164499 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:10,244 INFO Loaded AUDUSD/1H split=train: 41127 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:10,369 INFO Loaded AUDUSD/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:10,614 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:10,615 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:10,632 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:10,641 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:10,642 INFO Loaded EURGBP/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:10,740 INFO Rule labels [4H]: {'TRENDING_UP': 1847, 'TRENDING_DOWN': 1998, 'RANGING': 4586, 'VOLATILE': 2121}  ambiguous(conf<0.4)=2734 (total=10552)
2026-04-18 10:51:10,743 INFO Loaded EURGBP/1H split=train: 41130 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:10,765 INFO Loaded EURGBP/5M split=train: 493402 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:11,096 INFO Loaded EURGBP/15M split=train: 164511 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:11,257 INFO Loaded EURGBP/1H split=train: 41130 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:11,373 INFO Loaded EURGBP/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:11,602 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:11,603 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:11,623 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:11,631 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:11,632 INFO Loaded EURJPY/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:11,731 INFO Rule labels [4H]: {'TRENDING_UP': 2157, 'TRENDING_DOWN': 1757, 'RANGING': 4507, 'VOLATILE': 2131}  ambiguous(conf<0.4)=2584 (total=10552)
2026-04-18 10:51:11,733 INFO Loaded EURJPY/1H split=train: 41128 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:11,752 INFO Loaded EURJPY/5M split=train: 493414 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:12,085 INFO Loaded EURJPY/15M split=train: 164507 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:12,232 INFO Loaded EURJPY/1H split=train: 41128 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:12,359 INFO Loaded EURJPY/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:12,583 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:12,584 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:12,605 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:12,614 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:12,615 INFO Loaded EURUSD/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:12,732 INFO Rule labels [4H]: {'TRENDING_UP': 1644, 'TRENDING_DOWN': 2108, 'RANGING': 4678, 'VOLATILE': 2122}  ambiguous(conf<0.4)=2719 (total=10552)
2026-04-18 10:51:12,734 INFO Loaded EURUSD/1H split=train: 41131 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:12,762 INFO Loaded EURUSD/5M split=train: 493522 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:13,096 INFO Loaded EURUSD/15M split=train: 164518 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:13,252 INFO Loaded EURUSD/1H split=train: 41131 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:13,370 INFO Loaded EURUSD/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:13,586 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:13,587 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:13,605 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:13,613 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:13,614 INFO Loaded GBPJPY/4H split=train: 10553 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:13,711 INFO Rule labels [4H]: {'TRENDING_UP': 2031, 'TRENDING_DOWN': 1747, 'RANGING': 4653, 'VOLATILE': 2122}  ambiguous(conf<0.4)=2668 (total=10553)
2026-04-18 10:51:13,714 INFO Loaded GBPJPY/1H split=train: 41130 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:13,734 INFO Loaded GBPJPY/5M split=train: 493270 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:14,074 INFO Loaded GBPJPY/15M split=train: 164509 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:14,226 INFO Loaded GBPJPY/1H split=train: 41130 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:14,350 INFO Loaded GBPJPY/4H split=train: 10553 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:14,561 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:14,562 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:14,578 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:14,586 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:14,587 INFO Loaded GBPUSD/4H split=train: 10553 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:14,683 INFO Rule labels [4H]: {'TRENDING_UP': 1841, 'TRENDING_DOWN': 2017, 'RANGING': 4529, 'VOLATILE': 2166}  ambiguous(conf<0.4)=2643 (total=10553)
2026-04-18 10:51:14,686 INFO Loaded GBPUSD/1H split=train: 41129 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:14,707 INFO Loaded GBPUSD/5M split=train: 493457 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:15,049 INFO Loaded GBPUSD/15M split=train: 164510 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:15,202 INFO Loaded GBPUSD/1H split=train: 41129 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:15,327 INFO Loaded GBPUSD/4H split=train: 10553 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:15,510 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-18 10:51:15,511 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-18 10:51:15,526 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-18 10:51:15,533 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-18 10:51:15,534 INFO Loaded NZDUSD/4H split=train: 10260 bars (2016-01-04 → 2022-10-28)
2026-04-18 10:51:15,627 INFO Rule labels [4H]: {'TRENDING_UP': 1780, 'TRENDING_DOWN': 1985, 'RANGING': 4419, 'VOLATILE': 2076}  ambiguous(conf<0.4)=2697 (total=10260)
2026-04-18 10:51:15,629 INFO Loaded NZDUSD/1H split=train: 39991 bars (2016-01-04 → 2022-10-28)
2026-04-18 10:51:15,646 INFO Loaded NZDUSD/5M split=train: 479749 bars (2016-01-04 → 2022-10-28)
2026-04-18 10:51:15,991 INFO Loaded NZDUSD/15M split=train: 159956 bars (2016-01-04 → 2022-10-28)
2026-04-18 10:51:16,152 INFO Loaded NZDUSD/1H split=train: 39991 bars (2016-01-04 → 2022-10-28)
2026-04-18 10:51:16,275 INFO Loaded NZDUSD/4H split=train: 10260 bars (2016-01-04 → 2022-10-28)
2026-04-18 10:51:16,491 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:16,492 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:16,508 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:16,516 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:16,517 INFO Loaded USDCAD/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:16,612 INFO Rule labels [4H]: {'TRENDING_UP': 1779, 'TRENDING_DOWN': 2128, 'RANGING': 4525, 'VOLATILE': 2120}  ambiguous(conf<0.4)=2545 (total=10552)
2026-04-18 10:51:16,615 INFO Loaded USDCAD/1H split=train: 41132 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:16,638 INFO Loaded USDCAD/5M split=train: 493455 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:16,976 INFO Loaded USDCAD/15M split=train: 164520 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:17,119 INFO Loaded USDCAD/1H split=train: 41132 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:17,230 INFO Loaded USDCAD/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:17,441 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:17,442 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:17,459 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:17,466 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:17,468 INFO Loaded USDCHF/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:17,564 INFO Rule labels [4H]: {'TRENDING_UP': 2032, 'TRENDING_DOWN': 1900, 'RANGING': 4460, 'VOLATILE': 2160}  ambiguous(conf<0.4)=2487 (total=10552)
2026-04-18 10:51:17,567 INFO Loaded USDCHF/1H split=train: 41130 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:17,586 INFO Loaded USDCHF/5M split=train: 493287 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:17,937 INFO Loaded USDCHF/15M split=train: 164500 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:18,091 INFO Loaded USDCHF/1H split=train: 41130 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:18,205 INFO Loaded USDCHF/4H split=train: 10552 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:18,424 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:18,425 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:18,442 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:18,450 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-18 10:51:18,451 INFO Loaded USDJPY/4H split=train: 10553 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:18,545 INFO Rule labels [4H]: {'TRENDING_UP': 2348, 'TRENDING_DOWN': 1607, 'RANGING': 4451, 'VOLATILE': 2147}  ambiguous(conf<0.4)=2524 (total=10553)
2026-04-18 10:51:18,548 INFO Loaded USDJPY/1H split=train: 41133 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:18,570 INFO Loaded USDJPY/5M split=train: 493570 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:18,901 INFO Loaded USDJPY/15M split=train: 164531 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:19,048 INFO Loaded USDJPY/1H split=train: 41133 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:19,159 INFO Loaded USDJPY/4H split=train: 10553 bars (2016-01-04 → 2023-01-06)
2026-04-18 10:51:19,472 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-18 10:51:19,474 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-18 10:51:19,493 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-18 10:51:19,504 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-18 10:51:19,505 INFO Loaded XAUUSD/4H split=train: 22089 bars (2009-03-15 → 2023-01-06)
2026-04-18 10:51:19,680 INFO Rule labels [4H]: {'TRENDING_UP': 4634, 'TRENDING_DOWN': 4090, 'RANGING': 8914, 'VOLATILE': 4451}  ambiguous(conf<0.4)=4901 (total=22089)
2026-04-18 10:51:19,684 INFO Loaded XAUUSD/1H split=train: 83044 bars (2009-03-15 → 2023-01-06)
2026-04-18 10:51:19,736 INFO Loaded XAUUSD/5M split=train: 983008 bars (2009-03-15 → 2023-01-06)
2026-04-18 10:51:20,400 INFO Loaded XAUUSD/15M split=train: 328743 bars (2009-03-15 → 2023-01-06)
2026-04-18 10:51:20,640 INFO Loaded XAUUSD/1H split=train: 83044 bars (2009-03-15 → 2023-01-06)
2026-04-18 10:51:20,797 INFO Loaded XAUUSD/4H split=train: 22089 bars (2009-03-15 → 2023-01-06)
2026-04-18 10:51:20,936 INFO train_multi: 44 segments, ~7452801 total bars
2026-04-18 10:51:21,212 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-04-18 10:51:21,212 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-04-18 10:51:21,212 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-18 10:51:21,213 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-18 10:56:30,065 INFO train_multi TF=ALL: 7451481 sequences across 44 segments
2026-04-18 10:56:30,066 INFO train_multi TF=ALL: estimated peak RAM = 10368 MB (train=479960 val=120017 n_feat=72 seq_len=30)
2026-04-18 10:56:31,411 INFO train_multi TF=ALL: train=479960 val=120017 (5191 MB tensors)
2026-04-18 10:56:54,646 INFO train_multi TF=ALL epoch 1/50 train=0.8668 val=0.8403
2026-04-18 10:56:54,657 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-18 10:56:54,657 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-18 10:56:54,657 INFO train_multi TF=ALL: new best val=0.8403 — saved
2026-04-18 10:57:11,154 INFO train_multi TF=ALL epoch 2/50 train=0.7619 val=0.6887
2026-04-18 10:57:11,157 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-18 10:57:11,157 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-18 10:57:11,157 INFO train_multi TF=ALL: new best val=0.6887 — saved
2026-04-18 10:57:27,629 INFO train_multi TF=ALL epoch 3/50 train=0.6895 val=0.6889
2026-04-18 10:57:44,270 INFO train_multi TF=ALL epoch 4/50 train=0.6890 val=0.6894
2026-04-18 10:58:00,991 INFO train_multi TF=ALL epoch 5/50 train=0.6881 val=0.6908
2026-04-18 10:58:17,591 INFO train_multi TF=ALL epoch 6/50 train=0.6868 val=0.6887
2026-04-18 10:58:17,595 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-18 10:58:17,595 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-18 10:58:17,595 INFO train_multi TF=ALL: new best val=0.6887 — saved
2026-04-18 10:58:33,985 INFO train_multi TF=ALL epoch 7/50 train=0.6851 val=0.6868
2026-04-18 10:58:33,989 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-18 10:58:33,989 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-18 10:58:33,989 INFO train_multi TF=ALL: new best val=0.6868 — saved
2026-04-18 10:58:50,438 INFO train_multi TF=ALL epoch 8/50 train=0.6797 val=0.6764
2026-04-18 10:58:50,441 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-18 10:58:50,441 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-18 10:58:50,441 INFO train_multi TF=ALL: new best val=0.6764 — saved
2026-04-18 10:59:07,163 INFO train_multi TF=ALL epoch 9/50 train=0.6657 val=0.6601
2026-04-18 10:59:07,166 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-18 10:59:07,166 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-18 10:59:07,166 INFO train_multi TF=ALL: new best val=0.6601 — saved
2026-04-18 10:59:23,572 INFO train_multi TF=ALL epoch 10/50 train=0.6526 val=0.6481
2026-04-18 10:59:23,575 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-18 10:59:23,575 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-18 10:59:23,575 INFO train_multi TF=ALL: new best val=0.6481 — saved
2026-04-18 10:59:39,806 INFO train_multi TF=ALL epoch 11/50 train=0.6429 val=0.6420
2026-04-18 10:59:39,809 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-18 10:59:39,809 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-18 10:59:39,810 INFO train_multi TF=ALL: new best val=0.6420 — saved
2026-04-18 10:59:56,181 INFO train_multi TF=ALL epoch 12/50 train=0.6363 val=0.6349
2026-04-18 10:59:56,185 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-18 10:59:56,185 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-18 10:59:56,185 INFO train_multi TF=ALL: new best val=0.6349 — saved
2026-04-18 11:00:12,889 INFO train_multi TF=ALL epoch 13/50 train=0.6310 val=0.6299
2026-04-18 11:00:12,892 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-18 11:00:12,892 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-18 11:00:12,892 INFO train_multi TF=ALL: new best val=0.6299 — saved
2026-04-18 11:00:28,816 INFO train_multi TF=ALL epoch 14/50 train=0.6269 val=0.6302
2026-04-18 11:00:45,085 INFO train_multi TF=ALL epoch 15/50 train=0.6235 val=0.6278
2026-04-18 11:00:45,088 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-18 11:00:45,088 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-18 11:00:45,089 INFO train_multi TF=ALL: new best val=0.6278 — saved
2026-04-18 11:01:01,006 INFO train_multi TF=ALL epoch 16/50 train=0.6201 val=0.6249
2026-04-18 11:01:01,009 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-18 11:01:01,010 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-18 11:01:01,010 INFO train_multi TF=ALL: new best val=0.6249 — saved
2026-04-18 11:01:17,369 INFO train_multi TF=ALL epoch 17/50 train=0.6176 val=0.6239
2026-04-18 11:01:17,372 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-18 11:01:17,373 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-18 11:01:17,373 INFO train_multi TF=ALL: new best val=0.6239 — saved
2026-04-18 11:01:33,464 INFO train_multi TF=ALL epoch 18/50 train=0.6147 val=0.6255
2026-04-18 11:01:49,620 INFO train_multi TF=ALL epoch 19/50 train=0.6120 val=0.6213
2026-04-18 11:01:49,624 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-18 11:01:49,624 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-18 11:01:49,624 INFO train_multi TF=ALL: new best val=0.6213 — saved
2026-04-18 11:02:05,838 INFO train_multi TF=ALL epoch 20/50 train=0.6101 val=0.6200
2026-04-18 11:02:05,841 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-18 11:02:05,841 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-18 11:02:05,841 INFO train_multi TF=ALL: new best val=0.6200 — saved
2026-04-18 11:02:21,946 INFO train_multi TF=ALL epoch 21/50 train=0.6080 val=0.6185
2026-04-18 11:02:21,950 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-18 11:02:21,950 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-18 11:02:21,950 INFO train_multi TF=ALL: new best val=0.6185 — saved
2026-04-18 11:02:38,910 INFO train_multi TF=ALL epoch 22/50 train=0.6056 val=0.6203
2026-04-18 11:02:55,451 INFO train_multi TF=ALL epoch 23/50 train=0.6038 val=0.6190
2026-04-18 11:03:12,059 INFO train_multi TF=ALL epoch 24/50 train=0.6017 val=0.6182
2026-04-18 11:03:12,062 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-18 11:03:12,063 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-18 11:03:12,063 INFO train_multi TF=ALL: new best val=0.6182 — saved
2026-04-18 11:03:28,388 INFO train_multi TF=ALL epoch 25/50 train=0.6000 val=0.6180
2026-04-18 11:03:28,392 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-04-18 11:03:28,392 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-18 11:03:28,392 INFO train_multi TF=ALL: new best val=0.6180 — saved
2026-04-18 11:03:44,688 INFO train_multi TF=ALL epoch 26/50 train=0.5978 val=0.6239
2026-04-18 11:04:01,118 INFO train_multi TF=ALL epoch 27/50 train=0.5957 val=0.6233
2026-04-18 11:04:17,513 INFO train_multi TF=ALL epoch 28/50 train=0.5936 val=0.6202
2026-04-18 11:04:33,870 INFO train_multi TF=ALL epoch 29/50 train=0.5913 val=0.6228
2026-04-18 11:04:50,200 INFO train_multi TF=ALL epoch 30/50 train=0.5894 val=0.6272
2026-04-18 11:04:50,200 INFO train_multi TF=ALL early stop at epoch 30
2026-04-18 11:04:50,354 INFO === VectorStore: building similarity indices ===
2026-04-18 11:04:50,354 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-18 11:04:50,354 INFO Retrain complete.
2026-04-18 11:04:52,562 INFO Model gru: SUCCESS
2026-04-18 11:04:52,562 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-04-18 11:04:52,562 WARNING   [MISSING] regime_classifier → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_classifier.pkl
2026-04-18 11:04:52,562 WARNING   [MISSING] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-18 11:04:52,562 WARNING   [MISSING] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-04-18 11:04:52,563 WARNING Missing weights: ['regime_classifier', 'quality_scorer', 'rl_ppo'] — run retrain_incremental.py for each
2026-04-18 11:04:52,563 INFO Saved 5 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
2026-04-18 11:04:53,202 INFO === STEP 6: BACKTEST + REINFORCED TRAINING (3 rounds) ===
2026-04-18 11:04:53,389 INFO Backtest date range: 2021-01-01 → 2024-08-27 (reinforcement loop, test set protected; set BT_START_FLOOR env to change)
2026-04-18 11:04:53,389 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-04-18 11:04:53,389 INFO Round 1 — running backtest: 2021-01-01 → 2024-08-27 (ml_trader, shared ML cache)

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260418_110454.json
Trader                                   Trades      WR     PF   Return   TP1%   TP2%      DD  Sharpe
---------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)             3048  43.6%   2.00 1821.8% 43.6% 11.6%   3.3%    3.42

Calibration Summary:
  all          [OK] Calibration OK — p_win correlates with actual win rate.
  ml_trader    [OK] Calibration OK — p_win correlates with actual win rate.
2026-04-18 11:13:36,964 INFO Round 1 backtest — 3048 trades | avg WR=43.6% | avg PF=2.00 | avg Sharpe=3.42
2026-04-18 11:13:36,964 INFO   ml_trader: 3048 trades | WR=43.6% | PF=2.00 | Return=1821.8% | DD=3.3% | Sharpe=3.42
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 3048
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3045: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3046: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (3048 rows)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  AUDUSD        285 trades   260 days   1.10/day
  EURGBP        146 trades   129 days   1.13/day
  EURJPY        111 trades    92 days   1.21/day
  EURUSD        875 trades   837 days   1.04/day
  GBPJPY        100 trades    80 days   1.25/day
  GBPUSD        508 trades   478 days   1.06/day
  NZDUSD        181 trades   159 days   1.14/day
  USDCAD        192 trades   175 days   1.10/day
  USDCHF        174 trades   155 days   1.12/day
  USDJPY        382 trades   357 days   1.07/day
  XAUUSD         94 trades    77 days   1.22/day
  ✓  All symbols within normal range.
2026-04-18 11:13:38,435 INFO Round 1: wrote 3048 journal entries (total in file: 3048)
2026-04-18 11:13:38,438 INFO Round 1 — retraining regime...
2026-04-18 11:14:41,119 INFO Retrain regime: OK
2026-04-18 11:14:41,147 INFO Round 1 — retraining quality...
2026-04-18 11:14:48,457 INFO Retrain quality: OK
2026-04-18 11:14:48,477 INFO Round 1 — retraining rl...
2026-04-18 11:14:51,050 ERROR Retrain rl failed (rc=1):
2026-04-18 11:14:48,583 INFO retrain environment: KAGGLE
2026-04-18 11:14:50,447 INFO Device: CUDA (2 GPU(s))
2026-04-18 11:14:50,458 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-18 11:14:50,459 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-18 11:14:50,460 INFO === RLAgent (PPO) retrain ===
2026-04-18 11:14:50,465 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260418_111450
2026-04-18 11:14:50,632 WARNING RLAgent.retrain: stable-baselines3/gymnasium not available
2026-04-18 11:14:50,632 INFO Retrain complete.

2026-04-18 11:14:51,069 WARNING Round 1: some models failed to retrain — continuing anyway
2026-04-18 11:14:51,069 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-04-18 11:14:51,069 INFO Round 2 — running backtest: 2021-01-01 → 2024-08-27 (ml_trader, shared ML cache)
2026-04-18 11:23:21,590 INFO Round 2 backtest — 2880 trades | avg WR=45.1% | avg PF=2.01 | avg Sharpe=3.51
2026-04-18 11:23:21,590 INFO   ml_trader: 2880 trades | WR=45.1% | PF=2.01 | Return=1939.1% | DD=3.2% | Sharpe=3.51
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 2880
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2880 rows)
2026-04-18 11:23:23,024 INFO Round 2: wrote 2880 journal entries (total in file: 5928)
2026-04-18 11:23:23,027 INFO Round 2 — retraining regime...
2026-04-18 11:24:40,101 INFO Retrain regime: OK
2026-04-18 11:24:40,119 INFO Round 2 — retraining quality...
2026-04-18 11:24:49,688 INFO Retrain quality: OK
2026-04-18 11:24:49,706 INFO Round 2 — retraining rl...
2026-04-18 11:24:52,149 ERROR Retrain rl failed (rc=1):
2026-04-18 11:24:49,808 INFO retrain environment: KAGGLE
2026-04-18 11:24:51,648 INFO Device: CUDA (2 GPU(s))
2026-04-18 11:24:51,660 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-18 11:24:51,660 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-18 11:24:51,661 INFO === RLAgent (PPO) retrain ===
2026-04-18 11:24:51,662 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260418_112451
2026-04-18 11:24:51,694 WARNING RLAgent.retrain: stable-baselines3/gymnasium not available
2026-04-18 11:24:51,695 INFO Retrain complete.

2026-04-18 11:24:52,166 WARNING Round 2: some models failed to retrain — continuing anyway
2026-04-18 11:24:52,167 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-04-18 11:24:52,167 INFO Round 3 — running backtest: 2021-01-01 → 2024-08-27 (ml_trader, shared ML cache)
2026-04-18 11:33:20,015 INFO Round 3 backtest — 2926 trades | avg WR=44.0% | avg PF=1.90 | avg Sharpe=3.33
2026-04-18 11:33:20,016 INFO   ml_trader: 2926 trades | WR=44.0% | PF=1.90 | Return=1595.7% | DD=3.7% | Sharpe=3.33
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 2926
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (2926 rows)
2026-04-18 11:33:21,422 INFO Round 3: wrote 2926 journal entries (total in file: 8854)
2026-04-18 11:33:21,425 INFO Round 3 (final): retraining after last backtest...
2026-04-18 11:33:21,425 INFO Round 3 — retraining regime...
2026-04-18 11:34:23,177 INFO Retrain regime: OK
2026-04-18 11:34:23,202 INFO Round 3 — retraining quality...
2026-04-18 11:34:32,183 INFO Retrain quality: OK
2026-04-18 11:34:32,201 INFO Round 3 — retraining rl...
2026-04-18 11:34:34,671 ERROR Retrain rl failed (rc=1):
2026-04-18 11:34:32,304 INFO retrain environment: KAGGLE
2026-04-18 11:34:34,212 INFO Device: CUDA (2 GPU(s))
2026-04-18 11:34:34,224 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-18 11:34:34,224 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-18 11:34:34,225 INFO === RLAgent (PPO) retrain ===
2026-04-18 11:34:34,226 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260418_113434
2026-04-18 11:34:34,259 WARNING RLAgent.retrain: stable-baselines3/gymnasium not available
2026-04-18 11:34:34,259 INFO Retrain complete.

2026-04-18 11:34:34,690 INFO Improvement round 1 → 3: WR +0.4% | PF -0.105 | Sharpe -0.086
2026-04-18 11:34:34,868 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-04-18 11:34:34,878 INFO Journal entries: 8854
2026-04-18 11:34:34,878 INFO --- Training quality ---
2026-04-18 11:34:34,879 INFO Running retrain --model quality
2026-04-18 11:34:34,983 INFO retrain environment: KAGGLE
2026-04-18 11:34:36,917 INFO Device: CUDA (2 GPU(s))
2026-04-18 11:34:36,928 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-18 11:34:36,929 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-18 11:34:36,930 INFO === QualityScorer retrain ===
2026-04-18 11:34:37,084 INFO NumExpr defaulting to 4 threads.
2026-04-18 11:34:37,309 INFO QualityScorer: CUDA available — using GPU
2026-04-18 11:34:37,525 INFO QualityScorer loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (device=cuda)
2026-04-18 11:34:37,858 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/quality_scorer.pkl_20260418_113437
2026-04-18 11:34:38,166 INFO QualityScorer: 6009 samples, EV stats={'mean': -0.3565727174282074, 'std': 2.474141836166382, 'n_pos': 1072, 'n_neg': 4937}, device=cuda
2026-04-18 11:34:39,944 INFO Quality epoch   1/100 — va_huber=1.0660
2026-04-18 11:34:40,078 INFO Quality epoch   2/100 — va_huber=1.0627
2026-04-18 11:34:40,200 INFO Quality epoch   3/100 — va_huber=1.0640
2026-04-18 11:34:40,324 INFO Quality epoch   4/100 — va_huber=1.0643
2026-04-18 11:34:40,441 INFO Quality epoch   5/100 — va_huber=1.0624
2026-04-18 11:34:41,157 INFO Quality epoch  11/100 — va_huber=1.0626
2026-04-18 11:34:41,998 INFO Quality early stop at epoch 18
2026-04-18 11:34:42,018 INFO QualityScorer EV model: MAE=1.364 dir_acc=0.830 n_val=1202
2026-04-18 11:34:42,022 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-04-18 11:34:42,099 INFO Retrain complete.
2026-04-18 11:34:43,121 INFO Model quality: SUCCESS
2026-04-18 11:34:43,121 INFO --- Training rl ---
2026-04-18 11:34:43,121 INFO Running retrain --model rl
2026-04-18 11:34:43,221 INFO retrain environment: KAGGLE
2026-04-18 11:34:45,101 INFO Device: CUDA (2 GPU(s))
2026-04-18 11:34:45,112 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-18 11:34:45,112 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-18 11:34:45,113 INFO === RLAgent (PPO) retrain ===
2026-04-18 11:34:45,115 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260418_113445
2026-04-18 11:34:45,150 WARNING RLAgent.retrain: stable-baselines3/gymnasium not available
2026-04-18 11:34:45,151 INFO Retrain complete.

=== RETRAIN COMPLETE ===

Weights exported → /kaggle/working/trained_weights
2026-04-18 11:34:45,597 ERROR retrain rl failed (exit 1)
2026-04-18 11:34:45,597 ERROR Model rl failed: exit 1
2026-04-18 11:34:45,597 INFO Step 7b complete — summary: /kaggle/working/Multi-Bot/trading-system/ml_training/metrics/training_7b_summary.json
INFO  === STEP 8: PUSH TRAINING OUTPUTS TO GITHUB ===
INFO  Repo:   AnalystTKZ/Multi-Bot
INFO  Branch: main
INFO  Root:   /kaggle/working/Multi-Bot
INFO  Cloning AnalystTKZ/Multi-Bot ...
Traceback (most recent call last):
  File "/kaggle/working/Multi-Bot/trading-system/step8_push_to_github.py", line 206, in <module>
    main()
  File "/kaggle/working/Multi-Bot/trading-system/step8_push_to_github.py", line 154, in main
    _run(["git", "clone", "--depth=1", f"--branch={GITHUB_BRANCH}",
  File "/kaggle/working/Multi-Bot/trading-system/step8_push_to_github.py", line 111, in _run
    return subprocess.run(cmd, cwd=str(cwd), check=check, capture_output=True, text=True)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/subprocess.py", line 571, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['git', 'clone', '--depth=1', '--branch=main', 'https://ghp_jEcAzjsiNNcRDQnIV@github.com/AnalystTKZ/Multi-Bot.git', '/kaggle/working/Multi-Bot']' returned non-zero exit status 128.