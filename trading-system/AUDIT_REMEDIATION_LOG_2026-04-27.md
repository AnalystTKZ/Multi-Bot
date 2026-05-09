 (macro features reduced): /kaggle/working/Multi-Bot/trading-system/training_data/indices/VIX_1d.csv
  WARNING: optional file missing (macro features reduced): /kaggle/working/Multi-Bot/trading-system/training_data/fundamental/macro_releases.csv

All scripts and inputs verified.

=== Phase 0-5: Data preparation ===
  SKIP  Step 0 - Resample
  SKIP  Step 1 - Inventory
  SKIP  Step 2 - Cleaning
  SKIP  Step 3 - Alignment
  SKIP  Step 4 - Features
  START Step 5 - Split
2026-05-09 02:51:42,269 INFO Loading feature-engineered data...
2026-05-09 02:51:43,021 INFO Loaded 221743 rows, 202 features
2026-05-09 02:51:43,023 INFO Data span: 2016-01-04 -> 2025-08-05  (9.6 years)
2026-05-09 02:51:43,028 INFO Fold 000 train 2016-01-04 -> 2018-01-03 (47088 bars), val 2018-01-04 -> 2019-01-03 (23448 bars)
2026-05-09 02:51:43,028 INFO Fold 001 train 2018-01-04 -> 2020-01-03 (46825 bars), val 2020-01-06 -> 2020-12-31 (23259 bars)
2026-05-09 02:51:43,028 INFO Fold 002 train 2020-01-06 -> 2022-01-03 (46766 bars), val 2022-01-04 -> 2023-01-03 (23588 bars)
2026-05-09 02:51:43,029 INFO No leakage confirmed: every fold ends before final 2-year blind test

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
2026-05-09 02:51:46,577 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-05-09 02:51:46,577 INFO --- Training regime ---
2026-05-09 02:51:46,578 INFO Running retrain --model regime
2026-05-09 02:51:46,775 INFO retrain environment: KAGGLE
2026-05-09 02:51:48,458 INFO Device: CUDA (2 GPU(s))
2026-05-09 02:51:48,469 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-09 02:51:48,469 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-09 02:51:48,469 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-09 02:51:48,471 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-09 02:51:48,472 INFO Retrain data split: train
2026-05-09 02:51:48,472 INFO Retrain rolling fold selector: latest
2026-05-09 02:51:48,473 INFO === RegimeClassifier retrain (hierarchical: HTF 2-score bias + LTF 5-score behaviour) ===
2026-05-09 02:51:48,649 INFO NumExpr defaulting to 4 threads.
2026-05-09 02:51:48,913 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-09 02:51:48,913 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-09 02:51:48,913 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-09 02:51:48,914 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-09 02:51:48,999 INFO Regime rolling folds selected: ['fold_000', 'fold_001', 'fold_002']
2026-05-09 02:51:48,999 INFO === Regime rolling fold 1/3: fold_000 ===
2026-05-09 02:51:48,999 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-09 02:51:49,045 INFO Split boundaries loaded fold=fold_000/3 — train 2016-01-04→2018-01-03  val 2018-01-04→2019-01-03  test 2023-08-07→2025-08-05
2026-05-09 02:51:49,047 INFO Loaded XAUUSD/4H split=train fold=fold_000: 3204 bars (2016-01-04 → 2018-01-03)
2026-05-09 02:51:49,063 INFO Loaded EURUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-09 02:51:49,078 INFO Loaded USDJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-09 02:51:49,094 INFO Loaded EURJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-09 02:51:49,110 INFO Loaded GBPJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-09 02:51:49,126 INFO Loaded GBPUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-09 02:51:49,396 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:51:49,469 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:51:49,493 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:51:49,494 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:51:49,504 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:51:49,505 INFO Loaded XAUUSD/4H split=train fold=fold_000: 3204 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:51:49,800 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 2845}  ambiguous=1801 (total=3204) horizon=12
2026-05-09 02:51:49,803 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected XAUUSD — 3154 samples (group=gold) score_means={'bias_up_score': 0.0672, 'bias_down_score': 0.0466} labels={'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 2795} clean={'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 1029}
2026-05-09 02:51:49,983 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:50,026 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:50,058 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:50,058 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:50,066 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:50,067 INFO Loaded EURUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:51:50,285 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 98, 'BIAS_DOWN': 57, 'BIAS_NEUTRAL': 2868}  ambiguous=1742 (total=3023) horizon=12
2026-05-09 02:51:50,288 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected EURUSD — 2973 samples (group=dollar) score_means={'bias_up_score': 0.033, 'bias_down_score': 0.0192} labels={'BIAS_UP': 98, 'BIAS_DOWN': 57, 'BIAS_NEUTRAL': 2818} clean={'BIAS_UP': 98, 'BIAS_DOWN': 57, 'BIAS_NEUTRAL': 1105}
2026-05-09 02:51:50,477 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:50,514 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:50,535 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:50,535 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:50,543 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:50,544 INFO Loaded USDJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:51:50,762 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 102, 'BIAS_DOWN': 78, 'BIAS_NEUTRAL': 2843}  ambiguous=1762 (total=3023) horizon=12
2026-05-09 02:51:50,765 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected USDJPY — 2973 samples (group=dollar) score_means={'bias_up_score': 0.0343, 'bias_down_score': 0.0262} labels={'BIAS_UP': 102, 'BIAS_DOWN': 78, 'BIAS_NEUTRAL': 2793} clean={'BIAS_UP': 102, 'BIAS_DOWN': 78, 'BIAS_NEUTRAL': 1058}
2026-05-09 02:51:50,941 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:50,979 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:51,000 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:51,001 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:51,009 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:51,010 INFO Loaded EURJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:51:51,227 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 95, 'BIAS_DOWN': 63, 'BIAS_NEUTRAL': 2865}  ambiguous=1742 (total=3023) horizon=12
2026-05-09 02:51:51,231 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected EURJPY — 2973 samples (group=cross) score_means={'bias_up_score': 0.032, 'bias_down_score': 0.0212} labels={'BIAS_UP': 95, 'BIAS_DOWN': 63, 'BIAS_NEUTRAL': 2815} clean={'BIAS_UP': 95, 'BIAS_DOWN': 63, 'BIAS_NEUTRAL': 1099}
2026-05-09 02:51:51,401 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:51,439 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:51,460 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:51,460 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:51,468 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:51,469 INFO Loaded GBPJPY/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:51:51,690 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 164, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2758}  ambiguous=1723 (total=3023) horizon=12
2026-05-09 02:51:51,694 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected GBPJPY — 2973 samples (group=cross) score_means={'bias_up_score': 0.0552, 'bias_down_score': 0.034} labels={'BIAS_UP': 164, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2708} clean={'BIAS_UP': 164, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 1019}
2026-05-09 02:51:51,876 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:51,917 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:51,938 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:51,938 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:51,947 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:51,948 INFO Loaded GBPUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:51:52,179 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 79, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2843}  ambiguous=1759 (total=3023) horizon=12
2026-05-09 02:51:52,183 INFO Regime[4H mode=htf_bias split=train fold=fold_000]: collected GBPUSD — 2973 samples (group=dollar) score_means={'bias_up_score': 0.0266, 'bias_down_score': 0.034} labels={'BIAS_UP': 79, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2793} clean={'BIAS_UP': 79, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 1073}
2026-05-09 02:51:52,258 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 259, 'BIAS_DOWN': 164, 'BIAS_NEUTRAL': 5523}, 'dollar': {'BIAS_UP': 279, 'BIAS_DOWN': 236, 'BIAS_NEUTRAL': 8404}, 'gold': {'BIAS_UP': 212, 'BIAS_DOWN': 147, 'BIAS_NEUTRAL': 2795}}
2026-05-09 02:51:52,259 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0436, 'bias_down_score': 0.0276}, 'dollar': {'bias_up_score': 0.0313, 'bias_down_score': 0.0265}, 'gold': {'bias_up_score': 0.0672, 'bias_down_score': 0.0466}}
2026-05-09 02:51:52,259 INFO Regime[4H mode=htf_bias] label distribution by year: {2016: {'BIAS_UP': 286, 'BIAS_DOWN': 343, 'BIAS_NEUTRAL': 8193}, 2017: {'BIAS_UP': 462, 'BIAS_DOWN': 204, 'BIAS_NEUTRAL': 8447}, 2018: {'BIAS_UP': 2, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 82}}
2026-05-09 02:51:52,259 INFO Regime[4H mode=htf_bias] score means by year: {2016: {'bias_up_score': 0.0324, 'bias_down_score': 0.0389}, 2017: {'bias_up_score': 0.0507, 'bias_down_score': 0.0224}, 2018: {'bias_up_score': 0.0238, 'bias_down_score': 0.0}}
2026-05-09 02:51:52,329 INFO Loaded XAUUSD/4H split=val fold=fold_000: 1600 bars (2018-01-04 → 2019-01-03)
2026-05-09 02:51:52,330 INFO Loaded EURUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-09 02:51:52,330 INFO Loaded USDJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-09 02:51:52,331 INFO Loaded EURJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-09 02:51:52,332 INFO Loaded GBPJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-09 02:51:52,333 INFO Loaded GBPUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
2026-05-09 02:51:52,349 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:51:52,353 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:51:52,354 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:51:52,354 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:51:52,355 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:51:52,356 INFO Loaded XAUUSD/4H split=val fold=fold_000: 1600 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:51:52,565 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 18, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 1547}  ambiguous=851 (total=1600) horizon=12
2026-05-09 02:51:52,567 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected XAUUSD — 1550 samples (group=gold) score_means={'bias_up_score': 0.0116, 'bias_down_score': 0.0226} labels={'BIAS_UP': 18, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 1497} clean={'BIAS_UP': 18, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 677}
2026-05-09 02:51:52,654 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:52,659 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:52,660 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:52,660 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:52,661 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:52,662 INFO Loaded EURUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:51:52,859 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 24, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 1415}  ambiguous=876 (total=1506) horizon=12
2026-05-09 02:51:52,862 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected EURUSD — 1456 samples (group=dollar) score_means={'bias_up_score': 0.0165, 'bias_down_score': 0.046} labels={'BIAS_UP': 24, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 1365} clean={'BIAS_UP': 24, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 522}
2026-05-09 02:51:52,938 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:52,941 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:52,942 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:52,942 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:52,942 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:52,943 INFO Loaded USDJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:51:53,142 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 58, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1418}  ambiguous=888 (total=1506) horizon=12
2026-05-09 02:51:53,145 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected USDJPY — 1456 samples (group=dollar) score_means={'bias_up_score': 0.0398, 'bias_down_score': 0.0206} labels={'BIAS_UP': 58, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1368} clean={'BIAS_UP': 58, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 510}
2026-05-09 02:51:53,222 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:53,224 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:53,225 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:53,226 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:53,226 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:53,227 INFO Loaded EURJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:51:53,425 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 20, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1445}  ambiguous=874 (total=1506) horizon=12
2026-05-09 02:51:53,428 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected EURJPY — 1456 samples (group=cross) score_means={'bias_up_score': 0.0137, 'bias_down_score': 0.0282} labels={'BIAS_UP': 20, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 1395} clean={'BIAS_UP': 20, 'BIAS_DOWN': 41, 'BIAS_NEUTRAL': 555}
2026-05-09 02:51:53,512 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:53,515 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:53,516 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:53,516 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:53,516 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:53,517 INFO Loaded GBPJPY/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:51:53,714 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 27, 'BIAS_DOWN': 37, 'BIAS_NEUTRAL': 1442}  ambiguous=926 (total=1506) horizon=12
2026-05-09 02:51:53,717 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected GBPJPY — 1456 samples (group=cross) score_means={'bias_up_score': 0.0185, 'bias_down_score': 0.0254} labels={'BIAS_UP': 27, 'BIAS_DOWN': 37, 'BIAS_NEUTRAL': 1392} clean={'BIAS_UP': 27, 'BIAS_DOWN': 37, 'BIAS_NEUTRAL': 506}
2026-05-09 02:51:53,785 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:53,787 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:53,788 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:53,788 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:53,789 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:51:53,790 INFO Loaded GBPUSD/4H split=val fold=fold_000: 1506 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:51:53,984 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 50, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1371}  ambiguous=874 (total=1506) horizon=12
2026-05-09 02:51:53,986 INFO Regime[4H mode=htf_bias split=val fold=fold_000]: collected GBPUSD — 1456 samples (group=dollar) score_means={'bias_up_score': 0.0343, 'bias_down_score': 0.0584} labels={'BIAS_UP': 50, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 1321} clean={'BIAS_UP': 50, 'BIAS_DOWN': 85, 'BIAS_NEUTRAL': 476}
2026-05-09 02:51:54,052 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 47, 'BIAS_DOWN': 78, 'BIAS_NEUTRAL': 2787}, 'dollar': {'BIAS_UP': 132, 'BIAS_DOWN': 182, 'BIAS_NEUTRAL': 4054}, 'gold': {'BIAS_UP': 18, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 1497}}
2026-05-09 02:51:54,053 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0161, 'bias_down_score': 0.0268}, 'dollar': {'bias_up_score': 0.0302, 'bias_down_score': 0.0417}, 'gold': {'bias_up_score': 0.0116, 'bias_down_score': 0.0226}}
2026-05-09 02:51:54,053 INFO Regime[4H mode=htf_bias] label distribution by year: {2018: {'BIAS_UP': 196, 'BIAS_DOWN': 290, 'BIAS_NEUTRAL': 8260}, 2019: {'BIAS_UP': 1, 'BIAS_DOWN': 5, 'BIAS_NEUTRAL': 78}}
2026-05-09 02:51:54,053 INFO Regime[4H mode=htf_bias] score means by year: {2018: {'bias_up_score': 0.0224, 'bias_down_score': 0.0332}, 2019: {'bias_up_score': 0.0119, 'bias_down_score': 0.0595}}
2026-05-09 02:51:54,102 INFO Regime phase HTF dataset build fold=fold_000: 5.1s (train=18019 val=8830)
2026-05-09 02:51:54,103 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260509_025154
2026-05-09 02:51:54,479 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=24, n_classes=2)
2026-05-09 02:51:54,479 INFO Regime 4H/htf_bias warm start enabled from existing weights
2026-05-09 02:51:54,482 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=18019 val=8830 train_labels={'BIAS_UP': 750, 'BIAS_DOWN': 547, 'BIAS_NEUTRAL': 16722} val_labels={'BIAS_UP': 197, 'BIAS_DOWN': 295, 'BIAS_NEUTRAL': 8338}
2026-05-09 02:51:54,482 INFO RegimeClassifier[mode=htf_bias]: warm start HTF score head
2026-05-09 02:51:54,482 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-09 02:51:54,482 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 20.0, 'bias_down_score': 20.0}
2026-05-09 02:52:01,297 INFO Regime HTF score epoch  1/50 — tr=0.5008 va=0.4381 acc=0.855 bal=0.903 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.888, 'BIAS_DOWN': 0.969, 'BIAS_NEUTRAL': 0.85} precision={'BIAS_UP': 0.276, 'BIAS_DOWN': 0.265, 'BIAS_NEUTRAL': 0.996}
2026-05-09 02:52:01,842 INFO Regime HTF score epoch  2/50 — tr=0.4944 va=0.4312 bal=0.897
2026-05-09 02:52:02,415 INFO Regime HTF score epoch  3/50 — tr=0.5099 va=0.4288 bal=0.886
2026-05-09 02:52:02,968 INFO Regime HTF score epoch  4/50 — tr=0.5044 va=0.4268 bal=0.887
2026-05-09 02:52:03,554 INFO Regime HTF score epoch  5/50 — tr=0.4933 va=0.4253 acc=0.859 bal=0.886 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.843, 'BIAS_DOWN': 0.959, 'BIAS_NEUTRAL': 0.855} precision={'BIAS_UP': 0.282, 'BIAS_DOWN': 0.266, 'BIAS_NEUTRAL': 0.994}
2026-05-09 02:52:04,143 INFO Regime HTF score epoch  6/50 — tr=0.4919 va=0.4258 bal=0.887
2026-05-09 02:52:04,736 INFO Regime HTF score epoch  7/50 — tr=0.4840 va=0.4271 bal=0.892
2026-05-09 02:52:05,321 INFO Regime HTF score epoch  8/50 — tr=0.4831 va=0.4294 bal=0.901
2026-05-09 02:52:05,920 INFO Regime HTF score epoch  9/50 — tr=0.4788 va=0.4284 bal=0.903
2026-05-09 02:52:06,513 INFO Regime HTF score epoch 10/50 — tr=0.4702 va=0.4313 acc=0.853 bal=0.907 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.904, 'BIAS_DOWN': 0.969, 'BIAS_NEUTRAL': 0.848} precision={'BIAS_UP': 0.27, 'BIAS_DOWN': 0.266, 'BIAS_NEUTRAL': 0.996}
2026-05-09 02:52:07,092 INFO Regime HTF score epoch 11/50 — tr=0.4668 va=0.4325 bal=0.911
2026-05-09 02:52:07,685 INFO Regime HTF score epoch 12/50 — tr=0.4670 va=0.4362 bal=0.913
2026-05-09 02:52:08,271 INFO Regime HTF score epoch 13/50 — tr=0.4544 va=0.4368 bal=0.913
2026-05-09 02:52:08,827 INFO Regime HTF score epoch 14/50 — tr=0.4518 va=0.4400 bal=0.914
2026-05-09 02:52:09,425 INFO Regime HTF score epoch 15/50 — tr=0.4576 va=0.4402 acc=0.846 bal=0.916 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.924, 'BIAS_DOWN': 0.983, 'BIAS_NEUTRAL': 0.84} precision={'BIAS_UP': 0.26, 'BIAS_DOWN': 0.261, 'BIAS_NEUTRAL': 0.997}
2026-05-09 02:52:09,981 INFO Regime HTF score epoch 16/50 — tr=0.4601 va=0.4408 bal=0.915
2026-05-09 02:52:10,534 INFO Regime HTF score epoch 17/50 — tr=0.4414 va=0.4424 bal=0.918
2026-05-09 02:52:11,179 INFO Regime HTF score epoch 18/50 — tr=0.4444 va=0.4435 bal=0.918
2026-05-09 02:52:11,792 INFO Regime HTF score epoch 19/50 — tr=0.4445 va=0.4402 bal=0.918
2026-05-09 02:52:11,792 INFO Regime HTF score early stop at epoch 19
2026-05-09 02:52:12,339 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.850 margin=0.000 precision={'BIAS_UP': 0.268, 'BIAS_DOWN': 0.267, 'BIAS_NEUTRAL': 0.997} recall={'BIAS_UP': 0.904, 'BIAS_DOWN': 0.983, 'BIAS_NEUTRAL': 0.846} f1={'BIAS_UP': 0.414, 'BIAS_DOWN': 0.42, 'BIAS_NEUTRAL': 0.915} confusion=[[178, 0, 19], [0, 290, 5], [485, 797, 7056]] score_mae={'bias_up_score': 0.1448, 'bias_down_score': 0.2065} pred_share={'BIAS_UP': 0.0751, 'BIAS_DOWN': 0.1231, 'BIAS_NEUTRAL': 0.8018}
2026-05-09 02:52:12,341 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.268, 'BIAS_DOWN': 0.267, 'BIAS_NEUTRAL': 0.997} min_precision=0.300 recall={'BIAS_UP': 0.904, 'BIAS_DOWN': 0.983, 'BIAS_NEUTRAL': 0.846} min_recall=0.100 f1={'BIAS_UP': 0.414, 'BIAS_DOWN': 0.42, 'BIAS_NEUTRAL': 0.915} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_UP', 'BIAS_DOWN'] weak_recall=[] weak_f1=[] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-09 02:52:12,346 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-09 02:52:12,346 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-09 02:52:12,346 INFO Regime phase HTF train fold=fold_000: 17.9s
2026-05-09 02:52:12,466 INFO Regime HTF complete fold=fold_000: acc=0.852 bal=0.911 train=18019 val=8830 per_class={'BIAS_UP': 0.904, 'BIAS_DOWN': 0.983, 'BIAS_NEUTRAL': 0.846} precision={'BIAS_UP': 0.268, 'BIAS_DOWN': 0.267, 'BIAS_NEUTRAL': 0.997} threshold=0.850 margin=0.000
2026-05-09 02:52:12,468 INFO Loaded GBPUSD/4H split=train fold=fold_000: 3023 bars (2016-01-04 → 2018-01-03)
2026-05-09 02:52:12,570 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 79, 'BIAS_DOWN': 101, 'BIAS_NEUTRAL': 2843}  ambiguous=1759 (total=3023) horizon=12
2026-05-09 02:52:12,583 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on GBPUSD 4H:
{'BIAS_UP': 2.7241379310344827, 'BIAS_DOWN': 4.208333333333333, 'BIAS_NEUTRAL': 52.648148148148145}
2026-05-09 02:52:12,586 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (all labels):
{'BIAS_UP': {'n': 79, 'mean': 0.0006760971151340907, 'mean_over_std': 0.3060753727949538}, 'BIAS_DOWN': {'n': 101, 'mean': -0.0013596985315225921, 'mean_over_std': -0.5177272469970726}, 'BIAS_NEUTRAL': {'n': 2842, 'mean': 3.2626052441168133e-06, 'mean_over_std': 0.0010780520349318765}}
2026-05-09 02:52:12,587 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 79, 'mean': 0.0006760971151340907, 'mean_over_std': 0.3060753727949538}, 'BIAS_DOWN': {'n': 101, 'mean': -0.0013596985315225921, 'mean_over_std': -0.5177272469970726}, 'BIAS_NEUTRAL': {'n': 1084, 'mean': 2.970566261101014e-05, 'mean_over_std': 0.012922234592716483}}
2026-05-09 02:52:12,590 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-09 02:52:12,594 INFO Loaded XAUUSD/1H split=train fold=fold_000: 11914 bars (2016-01-04 → 2018-01-03)
2026-05-09 02:52:12,595 INFO Loaded EURUSD/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-09 02:52:12,596 INFO Loaded USDJPY/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-09 02:52:12,598 INFO Loaded EURJPY/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-09 02:52:12,599 INFO Loaded GBPJPY/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-09 02:52:12,601 INFO Loaded GBPUSD/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-09 02:52:12,619 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:12,627 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:12,630 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:12,631 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:12,632 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:12,637 INFO Loaded XAUUSD/1H split=train fold=fold_000: 11914 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:13,025 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected XAUUSD — 11864 samples (group=gold) score_means={'trend_score': 0.5075, 'range_score': 0.2273, 'chop_score': 0.4518, 'volatility_percentile': 0.3694, 'consolidation_score': 0.1952}
2026-05-09 02:52:13,158 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:13,163 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:13,165 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:13,165 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:13,166 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:13,168 INFO Loaded EURUSD/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:13,513 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected EURUSD — 11722 samples (group=dollar) score_means={'trend_score': 0.4931, 'range_score': 0.2373, 'chop_score': 0.464, 'volatility_percentile': 0.3716, 'consolidation_score': 0.1896}
2026-05-09 02:52:13,636 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:13,638 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:13,639 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:13,640 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:13,640 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:13,642 INFO Loaded USDJPY/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:13,986 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected USDJPY — 11722 samples (group=dollar) score_means={'trend_score': 0.4991, 'range_score': 0.231, 'chop_score': 0.4562, 'volatility_percentile': 0.3679, 'consolidation_score': 0.1984}
2026-05-09 02:52:14,117 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:14,119 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:14,120 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:14,121 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:14,121 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:14,123 INFO Loaded EURJPY/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:14,468 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected EURJPY — 11722 samples (group=cross) score_means={'trend_score': 0.4873, 'range_score': 0.2384, 'chop_score': 0.4674, 'volatility_percentile': 0.3763, 'consolidation_score': 0.1925}
2026-05-09 02:52:14,596 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:14,598 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:14,599 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:14,599 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:14,600 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:14,602 INFO Loaded GBPJPY/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:14,938 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected GBPJPY — 11722 samples (group=cross) score_means={'trend_score': 0.5009, 'range_score': 0.2311, 'chop_score': 0.4571, 'volatility_percentile': 0.3758, 'consolidation_score': 0.1946}
2026-05-09 02:52:15,065 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:15,068 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:15,068 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:15,069 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:15,069 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:15,071 INFO Loaded GBPUSD/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:15,411 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_000]: collected GBPUSD — 11722 samples (group=dollar) score_means={'trend_score': 0.5037, 'range_score': 0.2323, 'chop_score': 0.4563, 'volatility_percentile': 0.3792, 'consolidation_score': 0.186}
2026-05-09 02:52:15,532 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4941, 'range_score': 0.2347, 'chop_score': 0.4622, 'volatility_percentile': 0.3761, 'consolidation_score': 0.1936}, 'dollar': {'trend_score': 0.4986, 'range_score': 0.2335, 'chop_score': 0.4588, 'volatility_percentile': 0.3729, 'consolidation_score': 0.1913}, 'gold': {'trend_score': 0.5075, 'range_score': 0.2273, 'chop_score': 0.4518, 'volatility_percentile': 0.3694, 'consolidation_score': 0.1952}}
2026-05-09 02:52:15,532 INFO Regime[1H mode=ltf_behaviour] score means by year: {2016: {'trend_score': 0.495, 'range_score': 0.233, 'chop_score': 0.4607, 'volatility_percentile': 0.3776, 'consolidation_score': 0.192}, 2017: {'trend_score': 0.5017, 'range_score': 0.2328, 'chop_score': 0.4574, 'volatility_percentile': 0.3689, 'consolidation_score': 0.1942}, 2018: {'trend_score': 0.5603, 'range_score': 0.2296, 'chop_score': 0.3979, 'volatility_percentile': 0.395, 'consolidation_score': 0.1118}}
2026-05-09 02:52:15,645 INFO Loaded XAUUSD/1H split=val fold=fold_000: 6034 bars (2018-01-04 → 2019-01-03)
2026-05-09 02:52:15,646 INFO Loaded EURUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-09 02:52:15,648 INFO Loaded USDJPY/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-09 02:52:15,649 INFO Loaded EURJPY/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-09 02:52:15,650 INFO Loaded GBPJPY/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-09 02:52:15,652 INFO Loaded GBPUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
2026-05-09 02:52:15,668 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:15,671 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:15,672 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:15,673 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:15,673 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:15,675 INFO Loaded XAUUSD/1H split=val fold=fold_000: 6034 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:15,965 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected XAUUSD — 5984 samples (group=gold) score_means={'trend_score': 0.4716, 'range_score': 0.2479, 'chop_score': 0.4761, 'volatility_percentile': 0.3856, 'consolidation_score': 0.1812}
2026-05-09 02:52:16,082 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:16,087 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:16,089 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:16,090 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:16,090 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:16,092 INFO Loaded EURUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:16,350 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected EURUSD — 5812 samples (group=dollar) score_means={'trend_score': 0.4993, 'range_score': 0.2343, 'chop_score': 0.4572, 'volatility_percentile': 0.389, 'consolidation_score': 0.1807}
2026-05-09 02:52:16,478 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:16,481 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:16,482 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:16,482 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:16,482 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:16,484 INFO Loaded USDJPY/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:16,733 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected USDJPY — 5812 samples (group=dollar) score_means={'trend_score': 0.4943, 'range_score': 0.2334, 'chop_score': 0.4614, 'volatility_percentile': 0.3872, 'consolidation_score': 0.1806}
2026-05-09 02:52:16,861 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:16,864 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:16,865 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:16,865 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:16,866 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:16,868 INFO Loaded EURJPY/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:17,129 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected EURJPY — 5812 samples (group=cross) score_means={'trend_score': 0.4749, 'range_score': 0.2394, 'chop_score': 0.474, 'volatility_percentile': 0.3878, 'consolidation_score': 0.1827}
2026-05-09 02:52:17,259 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:17,262 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:17,263 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:17,263 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:17,263 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:17,265 INFO Loaded GBPJPY/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:17,521 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected GBPJPY — 5812 samples (group=cross) score_means={'trend_score': 0.4888, 'range_score': 0.2412, 'chop_score': 0.4689, 'volatility_percentile': 0.3963, 'consolidation_score': 0.1732}
2026-05-09 02:52:17,648 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:17,650 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:17,651 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:17,652 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:17,652 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:17,654 INFO Loaded GBPUSD/1H split=val fold=fold_000: 5862 bars (2018-01-04 → 2019-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:17,908 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_000]: collected GBPUSD — 5812 samples (group=dollar) score_means={'trend_score': 0.5007, 'range_score': 0.2339, 'chop_score': 0.4559, 'volatility_percentile': 0.3971, 'consolidation_score': 0.1718}
2026-05-09 02:52:18,032 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4818, 'range_score': 0.2403, 'chop_score': 0.4714, 'volatility_percentile': 0.3921, 'consolidation_score': 0.1779}, 'dollar': {'trend_score': 0.4981, 'range_score': 0.2338, 'chop_score': 0.4582, 'volatility_percentile': 0.3911, 'consolidation_score': 0.1777}, 'gold': {'trend_score': 0.4716, 'range_score': 0.2479, 'chop_score': 0.4761, 'volatility_percentile': 0.3856, 'consolidation_score': 0.1812}}
2026-05-09 02:52:18,032 INFO Regime[1H mode=ltf_behaviour] score means by year: {2018: {'trend_score': 0.4871, 'range_score': 0.2393, 'chop_score': 0.4666, 'volatility_percentile': 0.3883, 'consolidation_score': 0.1798}, 2019: {'trend_score': 0.6054, 'range_score': 0.1419, 'chop_score': 0.3633, 'volatility_percentile': 0.62, 'consolidation_score': 0.0268}}
2026-05-09 02:52:18,155 INFO Regime phase LTF dataset build fold=fold_000: 5.6s (train=70474 val=35044)
2026-05-09 02:52:18,156 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260509_025218
2026-05-09 02:52:18,161 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=31, n_classes=5)
2026-05-09 02:52:18,161 INFO Regime 1H/ltf_behaviour warm start enabled from existing weights
2026-05-09 02:52:18,171 INFO RegimeClassifier[mode=ltf_behaviour]: warm start score head
2026-05-09 02:52:18,172 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-09 02:52:18,445 INFO Regime score epoch  1/50 — tr=0.0043 va=0.0013 mae={'trend_score': 0.0256, 'range_score': 0.0386, 'chop_score': 0.0274, 'volatility_percentile': 0.0168, 'consolidation_score': 0.0237}
2026-05-09 02:52:18,608 INFO Regime score epoch  2/50 — tr=0.0043 va=0.0013
2026-05-09 02:52:18,765 INFO Regime score epoch  3/50 — tr=0.0043 va=0.0013
2026-05-09 02:52:18,941 INFO Regime score epoch  4/50 — tr=0.0043 va=0.0012
2026-05-09 02:52:19,101 INFO Regime score epoch  5/50 — tr=0.0042 va=0.0012 mae={'trend_score': 0.0244, 'range_score': 0.0388, 'chop_score': 0.0266, 'volatility_percentile': 0.0157, 'consolidation_score': 0.024}
2026-05-09 02:52:19,291 INFO Regime score epoch  6/50 — tr=0.0042 va=0.0013
2026-05-09 02:52:19,456 INFO Regime score epoch  7/50 — tr=0.0042 va=0.0012
2026-05-09 02:52:19,611 INFO Regime score epoch  8/50 — tr=0.0042 va=0.0012
2026-05-09 02:52:19,767 INFO Regime score epoch  9/50 — tr=0.0042 va=0.0012
2026-05-09 02:52:19,915 INFO Regime score epoch 10/50 — tr=0.0042 va=0.0012 mae={'trend_score': 0.0237, 'range_score': 0.0376, 'chop_score': 0.0258, 'volatility_percentile': 0.0161, 'consolidation_score': 0.0242}
2026-05-09 02:52:20,067 INFO Regime score epoch 11/50 — tr=0.0042 va=0.0012
2026-05-09 02:52:20,212 INFO Regime score epoch 12/50 — tr=0.0042 va=0.0012
2026-05-09 02:52:20,364 INFO Regime score epoch 13/50 — tr=0.0041 va=0.0012
2026-05-09 02:52:20,520 INFO Regime score epoch 14/50 — tr=0.0041 va=0.0012
2026-05-09 02:52:20,673 INFO Regime score epoch 15/50 — tr=0.0041 va=0.0012 mae={'trend_score': 0.0233, 'range_score': 0.0376, 'chop_score': 0.0254, 'volatility_percentile': 0.0162, 'consolidation_score': 0.0234}
2026-05-09 02:52:20,817 INFO Regime score epoch 16/50 — tr=0.0041 va=0.0012
2026-05-09 02:52:20,963 INFO Regime score epoch 17/50 — tr=0.0041 va=0.0012
2026-05-09 02:52:21,113 INFO Regime score epoch 18/50 — tr=0.0041 va=0.0012
2026-05-09 02:52:21,264 INFO Regime score epoch 19/50 — tr=0.0041 va=0.0012
2026-05-09 02:52:21,412 INFO Regime score epoch 20/50 — tr=0.0041 va=0.0012 mae={'trend_score': 0.0232, 'range_score': 0.0375, 'chop_score': 0.0251, 'volatility_percentile': 0.0161, 'consolidation_score': 0.0236}
2026-05-09 02:52:21,561 INFO Regime score epoch 21/50 — tr=0.0040 va=0.0012
2026-05-09 02:52:21,704 INFO Regime score epoch 22/50 — tr=0.0041 va=0.0011
2026-05-09 02:52:21,856 INFO Regime score epoch 23/50 — tr=0.0040 va=0.0012
2026-05-09 02:52:22,001 INFO Regime score epoch 24/50 — tr=0.0041 va=0.0011
2026-05-09 02:52:22,144 INFO Regime score epoch 25/50 — tr=0.0041 va=0.0011 mae={'trend_score': 0.0229, 'range_score': 0.0373, 'chop_score': 0.0249, 'volatility_percentile': 0.0158, 'consolidation_score': 0.0227}
2026-05-09 02:52:22,304 INFO Regime score epoch 26/50 — tr=0.0040 va=0.0012
2026-05-09 02:52:22,453 INFO Regime score epoch 27/50 — tr=0.0040 va=0.0011
2026-05-09 02:52:22,597 INFO Regime score epoch 28/50 — tr=0.0040 va=0.0011
2026-05-09 02:52:22,745 INFO Regime score epoch 29/50 — tr=0.0040 va=0.0011
2026-05-09 02:52:22,890 INFO Regime score epoch 30/50 — tr=0.0040 va=0.0011 mae={'trend_score': 0.0225, 'range_score': 0.0372, 'chop_score': 0.0245, 'volatility_percentile': 0.0159, 'consolidation_score': 0.0225}
2026-05-09 02:52:23,038 INFO Regime score epoch 31/50 — tr=0.0040 va=0.0011
2026-05-09 02:52:23,180 INFO Regime score epoch 32/50 — tr=0.0040 va=0.0011
2026-05-09 02:52:23,326 INFO Regime score epoch 33/50 — tr=0.0040 va=0.0011
2026-05-09 02:52:23,473 INFO Regime score epoch 34/50 — tr=0.0040 va=0.0011
2026-05-09 02:52:23,617 INFO Regime score epoch 35/50 — tr=0.0040 va=0.0011 mae={'trend_score': 0.0224, 'range_score': 0.0373, 'chop_score': 0.0244, 'volatility_percentile': 0.0159, 'consolidation_score': 0.0231}
2026-05-09 02:52:23,762 INFO Regime score epoch 36/50 — tr=0.0040 va=0.0011
2026-05-09 02:52:23,913 INFO Regime score epoch 37/50 — tr=0.0040 va=0.0011
2026-05-09 02:52:24,068 INFO Regime score epoch 38/50 — tr=0.0040 va=0.0011
2026-05-09 02:52:24,222 INFO Regime score epoch 39/50 — tr=0.0040 va=0.0011
2026-05-09 02:52:24,368 INFO Regime score epoch 40/50 — tr=0.0040 va=0.0011 mae={'trend_score': 0.0224, 'range_score': 0.0368, 'chop_score': 0.0242, 'volatility_percentile': 0.0157, 'consolidation_score': 0.0223}
2026-05-09 02:52:24,515 INFO Regime score epoch 41/50 — tr=0.0040 va=0.0011
2026-05-09 02:52:24,664 INFO Regime score epoch 42/50 — tr=0.0040 va=0.0011
2026-05-09 02:52:24,813 INFO Regime score epoch 43/50 — tr=0.0040 va=0.0011
2026-05-09 02:52:24,957 INFO Regime score epoch 44/50 — tr=0.0040 va=0.0011
2026-05-09 02:52:25,100 INFO Regime score epoch 45/50 — tr=0.0040 va=0.0011 mae={'trend_score': 0.0222, 'range_score': 0.0371, 'chop_score': 0.024, 'volatility_percentile': 0.0156, 'consolidation_score': 0.0223}
2026-05-09 02:52:25,243 INFO Regime score epoch 46/50 — tr=0.0040 va=0.0011
2026-05-09 02:52:25,390 INFO Regime score epoch 47/50 — tr=0.0040 va=0.0011
2026-05-09 02:52:25,541 INFO Regime score epoch 48/50 — tr=0.0040 va=0.0011
2026-05-09 02:52:25,684 INFO Regime score epoch 49/50 — tr=0.0040 va=0.0011
2026-05-09 02:52:25,828 INFO Regime score epoch 50/50 — tr=0.0040 va=0.0011 mae={'trend_score': 0.0223, 'range_score': 0.0372, 'chop_score': 0.0242, 'volatility_percentile': 0.0155, 'consolidation_score': 0.0226}
2026-05-09 02:52:25,828 INFO Regime score early stop at epoch 50
2026-05-09 02:52:25,852 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0222, 'range_score': 0.0366, 'chop_score': 0.0239, 'volatility_percentile': 0.0156, 'consolidation_score': 0.0226} mse={'trend_score': 0.00084, 'range_score': 0.00217, 'chop_score': 0.00092, 'volatility_percentile': 0.00046, 'consolidation_score': 0.00114} corr={'trend_score': 0.9914, 'range_score': 0.9477, 'chop_score': 0.9884, 'volatility_percentile': 0.995, 'consolidation_score': 0.9871} pred_std={'trend_score': 0.2159, 'range_score': 0.1333, 'chop_score': 0.1815, 'volatility_percentile': 0.2133, 'consolidation_score': 0.2083} target_std={'trend_score': 0.2203, 'range_score': 0.1446, 'chop_score': 0.1918, 'volatility_percentile': 0.2139, 'consolidation_score': 0.2101}
2026-05-09 02:52:25,857 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-09 02:52:25,857 INFO Regime phase LTF train fold=fold_000: 7.7s
2026-05-09 02:52:25,958 INFO Regime LTF complete fold=fold_000: score_accuracy=0.976, train=70474 val=35044 mae={'trend_score': 0.0222, 'range_score': 0.0366, 'chop_score': 0.0239, 'volatility_percentile': 0.0156, 'consolidation_score': 0.0226}
2026-05-09 02:52:25,959 INFO Loaded GBPUSD/1H split=train fold=fold_000: 11772 bars (2016-01-04 → 2018-01-03)
2026-05-09 02:52:26,105 INFO Regime[1H mode=ltf_behaviour fold=fold_000] LTF score diagnostics on GBPUSD:
{'trend_score': {'mean': 0.5032, 'q10': 0.2093, 'q50': 0.4988, 'q90': 0.8097}, 'range_score': {'mean': 0.2333, 'q10': 0.0513, 'q50': 0.2116, 'q90': 0.434}, 'chop_score': {'mean': 0.4568, 'q10': 0.2088, 'q50': 0.4474, 'q90': 0.7215}, 'volatility_percentile': {'mean': 0.3792, 'q10': 0.1013, 'q50': 0.3675, 'q90': 0.6776}, 'consolidation_score': {'mean': 0.1852, 'q10': 0.0, 'q50': 0.1183, 'q90': 0.5112}}
2026-05-09 02:52:26,108 INFO === Regime rolling fold 2/3: fold_001 ===
2026-05-09 02:52:26,108 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-09 02:52:26,109 INFO Split boundaries loaded fold=fold_001/3 — train 2018-01-04→2020-01-03  val 2020-01-06→2020-12-31  test 2023-08-07→2025-08-05
2026-05-09 02:52:26,110 INFO Loaded XAUUSD/4H split=train fold=fold_001: 3193 bars (2018-01-04 → 2020-01-03)
2026-05-09 02:52:26,111 INFO Loaded EURUSD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-09 02:52:26,111 INFO Loaded USDJPY/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
2026-05-09 02:52:26,112 INFO Loaded EURJPY/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
2026-05-09 02:52:26,113 INFO Loaded GBPJPY/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
2026-05-09 02:52:26,114 INFO Loaded GBPUSD/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
2026-05-09 02:52:26,133 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:26,139 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:26,140 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:26,141 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:26,141 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:26,142 INFO Loaded XAUUSD/4H split=train fold=fold_001: 3193 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:26,387 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 91, 'BIAS_DOWN': 40, 'BIAS_NEUTRAL': 3062}  ambiguous=1810 (total=3193) horizon=12
2026-05-09 02:52:26,390 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected XAUUSD — 3143 samples (group=gold) score_means={'bias_up_score': 0.029, 'bias_down_score': 0.0127} labels={'BIAS_UP': 91, 'BIAS_DOWN': 40, 'BIAS_NEUTRAL': 3012} clean={'BIAS_UP': 91, 'BIAS_DOWN': 40, 'BIAS_NEUTRAL': 1233}
2026-05-09 02:52:26,507 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:26,512 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:26,512 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:26,513 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:26,513 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:26,514 INFO Loaded EURUSD/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:26,729 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 35, 'BIAS_DOWN': 83, 'BIAS_NEUTRAL': 2888}  ambiguous=1761 (total=3006) horizon=12
2026-05-09 02:52:26,732 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected EURUSD — 2956 samples (group=dollar) score_means={'bias_up_score': 0.0118, 'bias_down_score': 0.0281} labels={'BIAS_UP': 35, 'BIAS_DOWN': 83, 'BIAS_NEUTRAL': 2838} clean={'BIAS_UP': 35, 'BIAS_DOWN': 83, 'BIAS_NEUTRAL': 1110}
2026-05-09 02:52:26,839 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:26,842 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:26,843 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:26,843 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:26,844 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:26,845 INFO Loaded USDJPY/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:27,067 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 78, 'BIAS_DOWN': 75, 'BIAS_NEUTRAL': 2854}  ambiguous=1708 (total=3007) horizon=12
2026-05-09 02:52:27,070 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected USDJPY — 2957 samples (group=dollar) score_means={'bias_up_score': 0.0264, 'bias_down_score': 0.0254} labels={'BIAS_UP': 78, 'BIAS_DOWN': 75, 'BIAS_NEUTRAL': 2804} clean={'BIAS_UP': 78, 'BIAS_DOWN': 75, 'BIAS_NEUTRAL': 1126}
2026-05-09 02:52:27,176 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:27,178 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:27,179 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:27,179 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:27,180 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:27,181 INFO Loaded EURJPY/4H split=train fold=fold_001: 3006 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:27,397 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 45, 'BIAS_DOWN': 69, 'BIAS_NEUTRAL': 2892}  ambiguous=1719 (total=3006) horizon=12
2026-05-09 02:52:27,400 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected EURJPY — 2956 samples (group=cross) score_means={'bias_up_score': 0.0152, 'bias_down_score': 0.0233} labels={'BIAS_UP': 45, 'BIAS_DOWN': 69, 'BIAS_NEUTRAL': 2842} clean={'BIAS_UP': 45, 'BIAS_DOWN': 69, 'BIAS_NEUTRAL': 1157}
2026-05-09 02:52:27,507 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:27,509 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:27,510 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:27,510 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:27,510 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:27,511 INFO Loaded GBPJPY/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:27,723 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 76, 'BIAS_DOWN': 93, 'BIAS_NEUTRAL': 2838}  ambiguous=1772 (total=3007) horizon=12
2026-05-09 02:52:27,726 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected GBPJPY — 2957 samples (group=cross) score_means={'bias_up_score': 0.0257, 'bias_down_score': 0.0315} labels={'BIAS_UP': 76, 'BIAS_DOWN': 93, 'BIAS_NEUTRAL': 2788} clean={'BIAS_UP': 76, 'BIAS_DOWN': 93, 'BIAS_NEUTRAL': 1056}
2026-05-09 02:52:27,830 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:27,832 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:27,833 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:27,834 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:27,834 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:27,835 INFO Loaded GBPUSD/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:28,052 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 84, 'BIAS_DOWN': 126, 'BIAS_NEUTRAL': 2797}  ambiguous=1784 (total=3007) horizon=12
2026-05-09 02:52:28,055 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected GBPUSD — 2957 samples (group=dollar) score_means={'bias_up_score': 0.0284, 'bias_down_score': 0.0426} labels={'BIAS_UP': 84, 'BIAS_DOWN': 126, 'BIAS_NEUTRAL': 2747} clean={'BIAS_UP': 84, 'BIAS_DOWN': 126, 'BIAS_NEUTRAL': 992}
2026-05-09 02:52:28,155 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 121, 'BIAS_DOWN': 162, 'BIAS_NEUTRAL': 5630}, 'dollar': {'BIAS_UP': 197, 'BIAS_DOWN': 284, 'BIAS_NEUTRAL': 8389}, 'gold': {'BIAS_UP': 91, 'BIAS_DOWN': 40, 'BIAS_NEUTRAL': 3012}}
2026-05-09 02:52:28,155 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0205, 'bias_down_score': 0.0274}, 'dollar': {'bias_up_score': 0.0222, 'bias_down_score': 0.032}, 'gold': {'bias_up_score': 0.029, 'bias_down_score': 0.0127}}
2026-05-09 02:52:28,155 INFO Regime[4H mode=htf_bias] label distribution by year: {2018: {'BIAS_UP': 197, 'BIAS_DOWN': 290, 'BIAS_NEUTRAL': 8259}, 2019: {'BIAS_UP': 211, 'BIAS_DOWN': 195, 'BIAS_NEUTRAL': 8696}, 2020: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 76}}
2026-05-09 02:52:28,155 INFO Regime[4H mode=htf_bias] score means by year: {2018: {'bias_up_score': 0.0225, 'bias_down_score': 0.0332}, 2019: {'bias_up_score': 0.0232, 'bias_down_score': 0.0214}, 2020: {'bias_up_score': 0.0128, 'bias_down_score': 0.0128}}
2026-05-09 02:52:28,237 INFO Loaded XAUUSD/4H split=val fold=fold_001: 1581 bars (2020-01-06 → 2020-12-31)
2026-05-09 02:52:28,238 INFO Loaded EURUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-09 02:52:28,239 INFO Loaded USDJPY/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-09 02:52:28,240 INFO Loaded EURJPY/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-09 02:52:28,240 INFO Loaded GBPJPY/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-09 02:52:28,241 INFO Loaded GBPUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
2026-05-09 02:52:28,258 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:28,262 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:28,263 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:28,263 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:28,263 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:28,264 INFO Loaded XAUUSD/4H split=val fold=fold_001: 1581 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:28,484 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 76, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 1497}  ambiguous=916 (total=1581) horizon=12
2026-05-09 02:52:28,487 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected XAUUSD — 1531 samples (group=gold) score_means={'bias_up_score': 0.0496, 'bias_down_score': 0.0052} labels={'BIAS_UP': 76, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 1447} clean={'BIAS_UP': 76, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 553}
2026-05-09 02:52:28,598 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:28,602 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:28,602 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:28,603 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:28,603 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:28,604 INFO Loaded EURUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:28,790 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 89, 'BIAS_DOWN': 33, 'BIAS_NEUTRAL': 1368}  ambiguous=880 (total=1490) horizon=12
2026-05-09 02:52:28,792 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected EURUSD — 1440 samples (group=dollar) score_means={'bias_up_score': 0.0618, 'bias_down_score': 0.0229} labels={'BIAS_UP': 89, 'BIAS_DOWN': 33, 'BIAS_NEUTRAL': 1318} clean={'BIAS_UP': 89, 'BIAS_DOWN': 33, 'BIAS_NEUTRAL': 462}
2026-05-09 02:52:28,901 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:28,905 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:28,905 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:28,906 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:28,906 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:28,907 INFO Loaded USDJPY/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:29,100 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 6, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1431}  ambiguous=877 (total=1490) horizon=12
2026-05-09 02:52:29,103 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected USDJPY — 1440 samples (group=dollar) score_means={'bias_up_score': 0.0042, 'bias_down_score': 0.0368} labels={'BIAS_UP': 6, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 1381} clean={'BIAS_UP': 6, 'BIAS_DOWN': 53, 'BIAS_NEUTRAL': 537}
2026-05-09 02:52:29,230 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:29,233 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:29,234 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:29,234 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:29,234 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:29,235 INFO Loaded EURJPY/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:29,433 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 42, 'BIAS_DOWN': 12, 'BIAS_NEUTRAL': 1436}  ambiguous=928 (total=1490) horizon=12
2026-05-09 02:52:29,436 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected EURJPY — 1440 samples (group=cross) score_means={'bias_up_score': 0.0292, 'bias_down_score': 0.0083} labels={'BIAS_UP': 42, 'BIAS_DOWN': 12, 'BIAS_NEUTRAL': 1386} clean={'BIAS_UP': 42, 'BIAS_DOWN': 12, 'BIAS_NEUTRAL': 491}
2026-05-09 02:52:29,543 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:29,545 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:29,546 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:29,546 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:29,547 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:29,548 INFO Loaded GBPJPY/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:29,744 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 26, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 1418}  ambiguous=910 (total=1490) horizon=12
2026-05-09 02:52:29,746 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected GBPJPY — 1440 samples (group=cross) score_means={'bias_up_score': 0.0181, 'bias_down_score': 0.0319} labels={'BIAS_UP': 26, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 1368} clean={'BIAS_UP': 26, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 476}
2026-05-09 02:52:29,852 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:29,854 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:29,855 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:29,855 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:29,856 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:29,857 INFO Loaded GBPUSD/4H split=val fold=fold_001: 1490 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:30,048 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 39, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 1429}  ambiguous=909 (total=1490) horizon=12
2026-05-09 02:52:30,050 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected GBPUSD — 1440 samples (group=dollar) score_means={'bias_up_score': 0.0271, 'bias_down_score': 0.0153} labels={'BIAS_UP': 39, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 1379} clean={'BIAS_UP': 39, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 493}
2026-05-09 02:52:30,150 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 68, 'BIAS_DOWN': 58, 'BIAS_NEUTRAL': 2754}, 'dollar': {'BIAS_UP': 134, 'BIAS_DOWN': 108, 'BIAS_NEUTRAL': 4078}, 'gold': {'BIAS_UP': 76, 'BIAS_DOWN': 8, 'BIAS_NEUTRAL': 1447}}
2026-05-09 02:52:30,150 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0236, 'bias_down_score': 0.0201}, 'dollar': {'bias_up_score': 0.031, 'bias_down_score': 0.025}, 'gold': {'bias_up_score': 0.0496, 'bias_down_score': 0.0052}}
2026-05-09 02:52:30,150 INFO Regime[4H mode=htf_bias] label distribution by year: {2020: {'BIAS_UP': 278, 'BIAS_DOWN': 174, 'BIAS_NEUTRAL': 8279}}
2026-05-09 02:52:30,150 INFO Regime[4H mode=htf_bias] score means by year: {2020: {'bias_up_score': 0.0318, 'bias_down_score': 0.0199}}
2026-05-09 02:52:30,230 INFO Regime phase HTF dataset build fold=fold_001: 4.1s (train=17926 val=8731)
2026-05-09 02:52:30,235 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=24, n_classes=2)
2026-05-09 02:52:30,235 INFO Regime 4H/htf_bias warm start enabled from existing weights
2026-05-09 02:52:30,238 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=17926 val=8731 train_labels={'BIAS_UP': 409, 'BIAS_DOWN': 486, 'BIAS_NEUTRAL': 17031} val_labels={'BIAS_UP': 278, 'BIAS_DOWN': 174, 'BIAS_NEUTRAL': 8279}
2026-05-09 02:52:30,238 INFO RegimeClassifier[mode=htf_bias]: warm start HTF score head
2026-05-09 02:52:30,238 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-09 02:52:30,238 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 20.0, 'bias_down_score': 20.0}
2026-05-09 02:52:30,779 INFO Regime HTF score epoch  1/50 — tr=0.4032 va=0.4141 acc=0.864 bal=0.922 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.971, 'BIAS_DOWN': 0.937, 'BIAS_NEUTRAL': 0.859} precision={'BIAS_UP': 0.296, 'BIAS_DOWN': 0.237, 'BIAS_NEUTRAL': 0.997}
2026-05-09 02:52:31,351 INFO Regime HTF score epoch  2/50 — tr=0.3981 va=0.4292 bal=0.925
2026-05-09 02:52:31,896 INFO Regime HTF score epoch  3/50 — tr=0.4015 va=0.4392 bal=0.928
2026-05-09 02:52:32,443 INFO Regime HTF score epoch  4/50 — tr=0.3948 va=0.4418 bal=0.926
2026-05-09 02:52:32,978 INFO Regime HTF score epoch  5/50 — tr=0.3922 va=0.4405 acc=0.853 bal=0.920 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.989, 'BIAS_DOWN': 0.925, 'BIAS_NEUTRAL': 0.846} precision={'BIAS_UP': 0.269, 'BIAS_DOWN': 0.235, 'BIAS_NEUTRAL': 0.998}
2026-05-09 02:52:33,516 INFO Regime HTF score epoch  6/50 — tr=0.3955 va=0.4398 bal=0.922
2026-05-09 02:52:34,051 INFO Regime HTF score epoch  7/50 — tr=0.3826 va=0.4344 bal=0.921
2026-05-09 02:52:34,623 INFO Regime HTF score epoch  8/50 — tr=0.3833 va=0.4291 bal=0.922
2026-05-09 02:52:35,155 INFO Regime HTF score epoch  9/50 — tr=0.3832 va=0.4246 bal=0.922
2026-05-09 02:52:35,692 INFO Regime HTF score epoch 10/50 — tr=0.3774 va=0.4190 acc=0.863 bal=0.921 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.986, 'BIAS_DOWN': 0.92, 'BIAS_NEUTRAL': 0.858} precision={'BIAS_UP': 0.291, 'BIAS_DOWN': 0.239, 'BIAS_NEUTRAL': 0.997}
2026-05-09 02:52:36,215 INFO Regime HTF score epoch 11/50 — tr=0.3799 va=0.4148 bal=0.920
2026-05-09 02:52:36,215 INFO Regime HTF score early stop at epoch 11
2026-05-09 02:52:36,703 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.850 margin=0.000 precision={'BIAS_UP': 0.272, 'BIAS_DOWN': 0.237, 'BIAS_NEUTRAL': 0.998} recall={'BIAS_UP': 0.989, 'BIAS_DOWN': 0.948, 'BIAS_NEUTRAL': 0.847} f1={'BIAS_UP': 0.426, 'BIAS_DOWN': 0.38, 'BIAS_NEUTRAL': 0.916} confusion=[[275, 0, 3], [0, 165, 9], [737, 530, 7012]] score_mae={'bias_up_score': 0.2038, 'bias_down_score': 0.1586} pred_share={'BIAS_UP': 0.1159, 'BIAS_DOWN': 0.0796, 'BIAS_NEUTRAL': 0.8045}
2026-05-09 02:52:36,704 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.272, 'BIAS_DOWN': 0.237, 'BIAS_NEUTRAL': 0.998} min_precision=0.300 recall={'BIAS_UP': 0.989, 'BIAS_DOWN': 0.948, 'BIAS_NEUTRAL': 0.847} min_recall=0.100 f1={'BIAS_UP': 0.426, 'BIAS_DOWN': 0.38, 'BIAS_NEUTRAL': 0.916} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_UP', 'BIAS_DOWN'] weak_recall=[] weak_f1=[] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-09 02:52:36,707 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-09 02:52:36,707 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-09 02:52:36,707 INFO Regime phase HTF train fold=fold_001: 6.5s
2026-05-09 02:52:36,809 INFO Regime HTF complete fold=fold_001: acc=0.854 bal=0.928 train=17926 val=8731 per_class={'BIAS_UP': 0.989, 'BIAS_DOWN': 0.948, 'BIAS_NEUTRAL': 0.847} precision={'BIAS_UP': 0.272, 'BIAS_DOWN': 0.237, 'BIAS_NEUTRAL': 0.998} threshold=0.850 margin=0.000
2026-05-09 02:52:36,811 INFO Loaded GBPUSD/4H split=train fold=fold_001: 3007 bars (2018-01-04 → 2020-01-03)
2026-05-09 02:52:36,906 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 84, 'BIAS_DOWN': 126, 'BIAS_NEUTRAL': 2797}  ambiguous=1784 (total=3007) horizon=12
2026-05-09 02:52:36,908 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on GBPUSD 4H:
{'BIAS_UP': 3.111111111111111, 'BIAS_DOWN': 4.5, 'BIAS_NEUTRAL': 49.94642857142857}
2026-05-09 02:52:36,911 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (all labels):
{'BIAS_UP': {'n': 84, 'mean': 0.0010512561557484462, 'mean_over_std': 0.48337911468599853}, 'BIAS_DOWN': {'n': 126, 'mean': -0.0008028232731602335, 'mean_over_std': -0.4685017842061616}, 'BIAS_NEUTRAL': {'n': 2796, 'mean': -5.339145816305192e-06, 'mean_over_std': -0.0024890728873315396}}
2026-05-09 02:52:36,911 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 84, 'mean': 0.0010512561557484462, 'mean_over_std': 0.48337911468599853}, 'BIAS_DOWN': {'n': 126, 'mean': -0.0008028232731602335, 'mean_over_std': -0.4685017842061616}, 'BIAS_NEUTRAL': {'n': 1013, 'mean': 7.359198623222808e-06, 'mean_over_std': 0.003910296078013891}}
2026-05-09 02:52:36,915 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-09 02:52:36,917 INFO Loaded XAUUSD/1H split=train fold=fold_001: 11927 bars (2018-01-04 → 2020-01-03)
2026-05-09 02:52:36,918 INFO Loaded EURUSD/1H split=train fold=fold_001: 11707 bars (2018-01-04 → 2020-01-03)
2026-05-09 02:52:36,919 INFO Loaded USDJPY/1H split=train fold=fold_001: 11711 bars (2018-01-04 → 2020-01-03)
2026-05-09 02:52:36,921 INFO Loaded EURJPY/1H split=train fold=fold_001: 11708 bars (2018-01-04 → 2020-01-03)
2026-05-09 02:52:36,922 INFO Loaded GBPJPY/1H split=train fold=fold_001: 11708 bars (2018-01-04 → 2020-01-03)
2026-05-09 02:52:36,924 INFO Loaded GBPUSD/1H split=train fold=fold_001: 11708 bars (2018-01-04 → 2020-01-03)
2026-05-09 02:52:36,940 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:36,943 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:36,945 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:36,945 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:36,945 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:36,947 INFO Loaded XAUUSD/1H split=train fold=fold_001: 11927 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:37,308 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected XAUUSD — 11877 samples (group=gold) score_means={'trend_score': 0.475, 'range_score': 0.244, 'chop_score': 0.473, 'volatility_percentile': 0.3803, 'consolidation_score': 0.1878}
2026-05-09 02:52:37,422 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:37,427 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:37,429 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:37,430 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:37,430 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:37,432 INFO Loaded EURUSD/1H split=train fold=fold_001: 11707 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:37,762 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected EURUSD — 11657 samples (group=dollar) score_means={'trend_score': 0.4875, 'range_score': 0.2372, 'chop_score': 0.4621, 'volatility_percentile': 0.378, 'consolidation_score': 0.1876}
2026-05-09 02:52:37,870 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:37,872 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:37,873 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:37,873 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:37,873 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:37,875 INFO Loaded USDJPY/1H split=train fold=fold_001: 11711 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:38,197 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected USDJPY — 11661 samples (group=dollar) score_means={'trend_score': 0.4818, 'range_score': 0.237, 'chop_score': 0.4712, 'volatility_percentile': 0.3725, 'consolidation_score': 0.1975}
2026-05-09 02:52:38,303 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:38,305 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:38,306 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:38,306 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:38,306 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:38,308 INFO Loaded EURJPY/1H split=train fold=fold_001: 11708 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:38,636 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected EURJPY — 11658 samples (group=cross) score_means={'trend_score': 0.4805, 'range_score': 0.2366, 'chop_score': 0.4707, 'volatility_percentile': 0.3744, 'consolidation_score': 0.1928}
2026-05-09 02:52:38,742 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:38,745 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:38,746 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:38,746 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:38,746 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:38,748 INFO Loaded GBPJPY/1H split=train fold=fold_001: 11708 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:39,082 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected GBPJPY — 11658 samples (group=cross) score_means={'trend_score': 0.4891, 'range_score': 0.2383, 'chop_score': 0.4697, 'volatility_percentile': 0.39, 'consolidation_score': 0.184}
2026-05-09 02:52:39,211 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:39,214 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:39,215 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:39,215 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:39,215 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:39,217 INFO Loaded GBPUSD/1H split=train fold=fold_001: 11708 bars (2018-01-04 → 2020-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:39,548 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected GBPUSD — 11658 samples (group=dollar) score_means={'trend_score': 0.4959, 'range_score': 0.2313, 'chop_score': 0.4576, 'volatility_percentile': 0.3919, 'consolidation_score': 0.1801}
2026-05-09 02:52:39,653 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4848, 'range_score': 0.2374, 'chop_score': 0.4702, 'volatility_percentile': 0.3822, 'consolidation_score': 0.1884}, 'dollar': {'trend_score': 0.4884, 'range_score': 0.2352, 'chop_score': 0.4636, 'volatility_percentile': 0.3808, 'consolidation_score': 0.1884}, 'gold': {'trend_score': 0.475, 'range_score': 0.244, 'chop_score': 0.473, 'volatility_percentile': 0.3803, 'consolidation_score': 0.1878}}
2026-05-09 02:52:39,653 INFO Regime[1H mode=ltf_behaviour] score means by year: {2018: {'trend_score': 0.4871, 'range_score': 0.2393, 'chop_score': 0.4666, 'volatility_percentile': 0.3883, 'consolidation_score': 0.1798}, 2019: {'trend_score': 0.4817, 'range_score': 0.236, 'chop_score': 0.4691, 'volatility_percentile': 0.3725, 'consolidation_score': 0.198}, 2020: {'trend_score': 0.6166, 'range_score': 0.1752, 'chop_score': 0.3624, 'volatility_percentile': 0.5885, 'consolidation_score': 0.0286}}
2026-05-09 02:52:39,753 INFO Loaded XAUUSD/1H split=val fold=fold_001: 5855 bars (2020-01-06 → 2020-12-31)
2026-05-09 02:52:39,754 INFO Loaded EURUSD/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
2026-05-09 02:52:39,756 INFO Loaded USDJPY/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
2026-05-09 02:52:39,757 INFO Loaded EURJPY/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
2026-05-09 02:52:39,758 INFO Loaded GBPJPY/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
2026-05-09 02:52:39,759 INFO Loaded GBPUSD/1H split=val fold=fold_001: 5814 bars (2020-01-06 → 2020-12-31)
2026-05-09 02:52:39,769 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:39,772 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:39,773 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:39,774 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:39,774 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:39,776 INFO Loaded XAUUSD/1H split=val fold=fold_001: 5855 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:40,036 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected XAUUSD — 5805 samples (group=gold) score_means={'trend_score': 0.4836, 'range_score': 0.2372, 'chop_score': 0.4777, 'volatility_percentile': 0.3611, 'consolidation_score': 0.2086}
2026-05-09 02:52:40,146 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:40,150 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:40,152 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:40,152 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:40,153 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:40,154 INFO Loaded EURUSD/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:40,400 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected EURUSD — 5765 samples (group=dollar) score_means={'trend_score': 0.4992, 'range_score': 0.2236, 'chop_score': 0.4521, 'volatility_percentile': 0.3866, 'consolidation_score': 0.1844}
2026-05-09 02:52:40,507 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:40,509 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:40,510 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:40,510 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:40,511 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:40,512 INFO Loaded USDJPY/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:40,756 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected USDJPY — 5765 samples (group=dollar) score_means={'trend_score': 0.4839, 'range_score': 0.2352, 'chop_score': 0.4713, 'volatility_percentile': 0.3706, 'consolidation_score': 0.1995}
2026-05-09 02:52:40,862 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:40,864 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:40,865 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:40,865 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:40,866 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:40,869 INFO Loaded EURJPY/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:41,112 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected EURJPY — 5765 samples (group=cross) score_means={'trend_score': 0.4913, 'range_score': 0.2317, 'chop_score': 0.469, 'volatility_percentile': 0.3781, 'consolidation_score': 0.1938}
2026-05-09 02:52:41,221 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:41,223 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:41,224 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:41,225 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:41,225 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:41,227 INFO Loaded GBPJPY/1H split=val fold=fold_001: 5815 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:41,471 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected GBPJPY — 5765 samples (group=cross) score_means={'trend_score': 0.4675, 'range_score': 0.2397, 'chop_score': 0.484, 'volatility_percentile': 0.3868, 'consolidation_score': 0.1954}
2026-05-09 02:52:41,577 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:41,579 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:41,580 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:41,581 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:41,581 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:41,583 INFO Loaded GBPUSD/1H split=val fold=fold_001: 5814 bars (2020-01-06 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:41,831 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected GBPUSD — 5764 samples (group=dollar) score_means={'trend_score': 0.4894, 'range_score': 0.2257, 'chop_score': 0.4574, 'volatility_percentile': 0.3874, 'consolidation_score': 0.1784}
2026-05-09 02:52:41,933 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4794, 'range_score': 0.2357, 'chop_score': 0.4765, 'volatility_percentile': 0.3825, 'consolidation_score': 0.1946}, 'dollar': {'trend_score': 0.4908, 'range_score': 0.2281, 'chop_score': 0.4603, 'volatility_percentile': 0.3815, 'consolidation_score': 0.1874}, 'gold': {'trend_score': 0.4836, 'range_score': 0.2372, 'chop_score': 0.4777, 'volatility_percentile': 0.3611, 'consolidation_score': 0.2086}}
2026-05-09 02:52:41,933 INFO Regime[1H mode=ltf_behaviour] score means by year: {2020: {'trend_score': 0.4858, 'range_score': 0.2322, 'chop_score': 0.4686, 'volatility_percentile': 0.3784, 'consolidation_score': 0.1934}}
2026-05-09 02:52:42,014 INFO Regime phase LTF dataset build fold=fold_001: 5.1s (train=70169 val=34629)
2026-05-09 02:52:42,019 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=31, n_classes=5)
2026-05-09 02:52:42,019 INFO Regime 1H/ltf_behaviour warm start enabled from existing weights
2026-05-09 02:52:42,029 INFO RegimeClassifier[mode=ltf_behaviour]: warm start score head
2026-05-09 02:52:42,029 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-09 02:52:42,190 INFO Regime score epoch  1/50 — tr=0.0041 va=0.0012 mae={'trend_score': 0.0251, 'range_score': 0.036, 'chop_score': 0.0249, 'volatility_percentile': 0.0165, 'consolidation_score': 0.0246}
2026-05-09 02:52:42,335 INFO Regime score epoch  2/50 — tr=0.0040 va=0.0012
2026-05-09 02:52:42,487 INFO Regime score epoch  3/50 — tr=0.0040 va=0.0012
2026-05-09 02:52:42,633 INFO Regime score epoch  4/50 — tr=0.0040 va=0.0012
2026-05-09 02:52:42,780 INFO Regime score epoch  5/50 — tr=0.0040 va=0.0011 mae={'trend_score': 0.0229, 'range_score': 0.0363, 'chop_score': 0.0233, 'volatility_percentile': 0.0165, 'consolidation_score': 0.0246}
2026-05-09 02:52:42,939 INFO Regime score epoch  6/50 — tr=0.0040 va=0.0011
2026-05-09 02:52:43,091 INFO Regime score epoch  7/50 — tr=0.0039 va=0.0011
2026-05-09 02:52:43,240 INFO Regime score epoch  8/50 — tr=0.0040 va=0.0011
2026-05-09 02:52:43,389 INFO Regime score epoch  9/50 — tr=0.0039 va=0.0011
2026-05-09 02:52:43,544 INFO Regime score epoch 10/50 — tr=0.0039 va=0.0011 mae={'trend_score': 0.0219, 'range_score': 0.0368, 'chop_score': 0.0234, 'volatility_percentile': 0.0157, 'consolidation_score': 0.024}
2026-05-09 02:52:43,700 INFO Regime score epoch 11/50 — tr=0.0039 va=0.0011
2026-05-09 02:52:43,850 INFO Regime score epoch 12/50 — tr=0.0039 va=0.0011
2026-05-09 02:52:43,998 INFO Regime score epoch 13/50 — tr=0.0039 va=0.0011
2026-05-09 02:52:44,156 INFO Regime score epoch 14/50 — tr=0.0039 va=0.0011
2026-05-09 02:52:44,302 INFO Regime score epoch 15/50 — tr=0.0039 va=0.0011 mae={'trend_score': 0.0213, 'range_score': 0.0365, 'chop_score': 0.0229, 'volatility_percentile': 0.0156, 'consolidation_score': 0.0242}
2026-05-09 02:52:44,451 INFO Regime score epoch 16/50 — tr=0.0039 va=0.0011
2026-05-09 02:52:44,602 INFO Regime score epoch 17/50 — tr=0.0039 va=0.0011
2026-05-09 02:52:44,751 INFO Regime score epoch 18/50 — tr=0.0039 va=0.0011
2026-05-09 02:52:44,905 INFO Regime score epoch 19/50 — tr=0.0039 va=0.0011
2026-05-09 02:52:45,050 INFO Regime score epoch 20/50 — tr=0.0039 va=0.0011 mae={'trend_score': 0.0216, 'range_score': 0.0356, 'chop_score': 0.023, 'volatility_percentile': 0.0157, 'consolidation_score': 0.0239}
2026-05-09 02:52:45,198 INFO Regime score epoch 21/50 — tr=0.0039 va=0.0011
2026-05-09 02:52:45,343 INFO Regime score epoch 22/50 — tr=0.0038 va=0.0011
2026-05-09 02:52:45,490 INFO Regime score epoch 23/50 — tr=0.0038 va=0.0011
2026-05-09 02:52:45,637 INFO Regime score epoch 24/50 — tr=0.0038 va=0.0011
2026-05-09 02:52:45,784 INFO Regime score epoch 25/50 — tr=0.0038 va=0.0011 mae={'trend_score': 0.0213, 'range_score': 0.0357, 'chop_score': 0.0226, 'volatility_percentile': 0.0156, 'consolidation_score': 0.0239}
2026-05-09 02:52:45,944 INFO Regime score epoch 26/50 — tr=0.0038 va=0.0011
2026-05-09 02:52:46,097 INFO Regime score epoch 27/50 — tr=0.0038 va=0.0011
2026-05-09 02:52:46,255 INFO Regime score epoch 28/50 — tr=0.0038 va=0.0011
2026-05-09 02:52:46,407 INFO Regime score epoch 29/50 — tr=0.0038 va=0.0011
2026-05-09 02:52:46,555 INFO Regime score epoch 30/50 — tr=0.0038 va=0.0011 mae={'trend_score': 0.021, 'range_score': 0.0358, 'chop_score': 0.0223, 'volatility_percentile': 0.0154, 'consolidation_score': 0.0233}
2026-05-09 02:52:46,705 INFO Regime score epoch 31/50 — tr=0.0038 va=0.0011
2026-05-09 02:52:46,851 INFO Regime score epoch 32/50 — tr=0.0038 va=0.0011
2026-05-09 02:52:47,009 INFO Regime score epoch 33/50 — tr=0.0038 va=0.0011
2026-05-09 02:52:47,168 INFO Regime score epoch 34/50 — tr=0.0038 va=0.0011
2026-05-09 02:52:47,318 INFO Regime score epoch 35/50 — tr=0.0038 va=0.0011 mae={'trend_score': 0.021, 'range_score': 0.0358, 'chop_score': 0.0224, 'volatility_percentile': 0.0157, 'consolidation_score': 0.0241}
2026-05-09 02:52:47,478 INFO Regime score epoch 36/50 — tr=0.0038 va=0.0011
2026-05-09 02:52:47,633 INFO Regime score epoch 37/50 — tr=0.0038 va=0.0011
2026-05-09 02:52:47,783 INFO Regime score epoch 38/50 — tr=0.0038 va=0.0011
2026-05-09 02:52:47,941 INFO Regime score epoch 39/50 — tr=0.0038 va=0.0011
2026-05-09 02:52:48,101 INFO Regime score epoch 40/50 — tr=0.0038 va=0.0011 mae={'trend_score': 0.0205, 'range_score': 0.0357, 'chop_score': 0.0217, 'volatility_percentile': 0.0157, 'consolidation_score': 0.0239}
2026-05-09 02:52:48,252 INFO Regime score epoch 41/50 — tr=0.0038 va=0.0011
2026-05-09 02:52:48,413 INFO Regime score epoch 42/50 — tr=0.0038 va=0.0011
2026-05-09 02:52:48,561 INFO Regime score epoch 43/50 — tr=0.0038 va=0.0011
2026-05-09 02:52:48,711 INFO Regime score epoch 44/50 — tr=0.0038 va=0.0011
2026-05-09 02:52:48,866 INFO Regime score epoch 45/50 — tr=0.0038 va=0.0011 mae={'trend_score': 0.0208, 'range_score': 0.0357, 'chop_score': 0.0223, 'volatility_percentile': 0.0156, 'consolidation_score': 0.0233}
2026-05-09 02:52:49,023 INFO Regime score epoch 46/50 — tr=0.0038 va=0.0011
2026-05-09 02:52:49,185 INFO Regime score epoch 47/50 — tr=0.0038 va=0.0011
2026-05-09 02:52:49,344 INFO Regime score epoch 48/50 — tr=0.0038 va=0.0011
2026-05-09 02:52:49,494 INFO Regime score epoch 49/50 — tr=0.0038 va=0.0011
2026-05-09 02:52:49,651 INFO Regime score epoch 50/50 — tr=0.0038 va=0.0011 mae={'trend_score': 0.0206, 'range_score': 0.036, 'chop_score': 0.022, 'volatility_percentile': 0.0158, 'consolidation_score': 0.0241}
2026-05-09 02:52:49,675 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0207, 'range_score': 0.0358, 'chop_score': 0.0217, 'volatility_percentile': 0.0154, 'consolidation_score': 0.0235} mse={'trend_score': 0.00072, 'range_score': 0.00204, 'chop_score': 0.00077, 'volatility_percentile': 0.00044, 'consolidation_score': 0.0013} corr={'trend_score': 0.9928, 'range_score': 0.9523, 'chop_score': 0.9906, 'volatility_percentile': 0.9958, 'consolidation_score': 0.9869} pred_std={'trend_score': 0.2203, 'range_score': 0.1323, 'chop_score': 0.1849, 'volatility_percentile': 0.2261, 'consolidation_score': 0.2168} target_std={'trend_score': 0.2234, 'range_score': 0.1453, 'chop_score': 0.1945, 'volatility_percentile': 0.2276, 'consolidation_score': 0.2218}
2026-05-09 02:52:49,680 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-09 02:52:49,681 INFO Regime phase LTF train fold=fold_001: 7.7s
2026-05-09 02:52:49,788 INFO Regime LTF complete fold=fold_001: score_accuracy=0.977, train=70169 val=34629 mae={'trend_score': 0.0207, 'range_score': 0.0358, 'chop_score': 0.0217, 'volatility_percentile': 0.0154, 'consolidation_score': 0.0235}
2026-05-09 02:52:49,790 INFO Loaded GBPUSD/1H split=train fold=fold_001: 11708 bars (2018-01-04 → 2020-01-03)
2026-05-09 02:52:49,932 INFO Regime[1H mode=ltf_behaviour fold=fold_001] LTF score diagnostics on GBPUSD:
{'trend_score': {'mean': 0.4957, 'q10': 0.1933, 'q50': 0.4914, 'q90': 0.8058}, 'range_score': {'mean': 0.2324, 'q10': 0.0513, 'q50': 0.2091, 'q90': 0.4337}, 'chop_score': {'mean': 0.4583, 'q10': 0.2133, 'q50': 0.4442, 'q90': 0.7243}, 'volatility_percentile': {'mean': 0.3919, 'q10': 0.1066, 'q50': 0.3763, 'q90': 0.6881}, 'consolidation_score': {'mean': 0.1793, 'q10': 0.0, 'q50': 0.1069, 'q90': 0.4936}}
2026-05-09 02:52:49,935 INFO === Regime rolling fold 3/3: fold_002 ===
2026-05-09 02:52:49,935 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-09 02:52:49,936 INFO Split boundaries loaded fold=fold_002/3 — train 2020-01-06→2022-01-03  val 2022-01-04→2023-01-03  test 2023-08-07→2025-08-05
2026-05-09 02:52:49,937 INFO Loaded XAUUSD/4H split=train fold=fold_002: 3180 bars (2020-01-06 → 2022-01-03)
2026-05-09 02:52:49,938 INFO Loaded EURUSD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-09 02:52:49,939 INFO Loaded USDJPY/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-09 02:52:49,940 INFO Loaded EURJPY/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-09 02:52:49,940 INFO Loaded GBPJPY/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-09 02:52:49,941 INFO Loaded GBPUSD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-09 02:52:49,951 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:49,954 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:49,955 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:49,956 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:49,956 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:49,957 INFO Loaded XAUUSD/4H split=train fold=fold_002: 3180 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:50,190 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 100, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 3045}  ambiguous=1873 (total=3180) horizon=12
2026-05-09 02:52:50,193 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected XAUUSD — 3130 samples (group=gold) score_means={'bias_up_score': 0.0319, 'bias_down_score': 0.0112} labels={'BIAS_UP': 100, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 2995} clean={'BIAS_UP': 100, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 1144}
2026-05-09 02:52:50,304 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:50,308 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:50,310 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:50,310 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:50,311 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:50,312 INFO Loaded EURUSD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:50,527 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 125, 'BIAS_DOWN': 105, 'BIAS_NEUTRAL': 2766}  ambiguous=1697 (total=2996) horizon=12
2026-05-09 02:52:50,530 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected EURUSD — 2946 samples (group=dollar) score_means={'bias_up_score': 0.0424, 'bias_down_score': 0.0356} labels={'BIAS_UP': 125, 'BIAS_DOWN': 105, 'BIAS_NEUTRAL': 2716} clean={'BIAS_UP': 125, 'BIAS_DOWN': 105, 'BIAS_NEUTRAL': 1043}
2026-05-09 02:52:50,636 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:50,638 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:50,639 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:50,640 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:50,640 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:50,641 INFO Loaded USDJPY/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:50,856 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 70, 'BIAS_DOWN': 56, 'BIAS_NEUTRAL': 2870}  ambiguous=1792 (total=2996) horizon=12
2026-05-09 02:52:50,860 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected USDJPY — 2946 samples (group=dollar) score_means={'bias_up_score': 0.0238, 'bias_down_score': 0.019} labels={'BIAS_UP': 70, 'BIAS_DOWN': 56, 'BIAS_NEUTRAL': 2820} clean={'BIAS_UP': 70, 'BIAS_DOWN': 56, 'BIAS_NEUTRAL': 1061}
2026-05-09 02:52:50,968 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:50,970 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:50,971 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:50,972 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:50,972 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:50,974 INFO Loaded EURJPY/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:51,192 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 82, 'BIAS_DOWN': 58, 'BIAS_NEUTRAL': 2856}  ambiguous=1784 (total=2996) horizon=12
2026-05-09 02:52:51,196 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected EURJPY — 2946 samples (group=cross) score_means={'bias_up_score': 0.0278, 'bias_down_score': 0.0197} labels={'BIAS_UP': 82, 'BIAS_DOWN': 58, 'BIAS_NEUTRAL': 2806} clean={'BIAS_UP': 82, 'BIAS_DOWN': 58, 'BIAS_NEUTRAL': 1055}
2026-05-09 02:52:51,302 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:51,304 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:51,305 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:51,305 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:51,306 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:51,307 INFO Loaded GBPJPY/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:51,523 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 112, 'BIAS_DOWN': 61, 'BIAS_NEUTRAL': 2823}  ambiguous=1763 (total=2996) horizon=12
2026-05-09 02:52:51,526 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected GBPJPY — 2946 samples (group=cross) score_means={'bias_up_score': 0.038, 'bias_down_score': 0.0207} labels={'BIAS_UP': 112, 'BIAS_DOWN': 61, 'BIAS_NEUTRAL': 2773} clean={'BIAS_UP': 112, 'BIAS_DOWN': 61, 'BIAS_NEUTRAL': 1028}
2026-05-09 02:52:51,634 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:51,637 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:51,637 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:51,638 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:51,638 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:51,639 INFO Loaded GBPUSD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:51,860 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 71, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 2879}  ambiguous=1724 (total=2996) horizon=12
2026-05-09 02:52:51,863 INFO Regime[4H mode=htf_bias split=train fold=fold_002]: collected GBPUSD — 2946 samples (group=dollar) score_means={'bias_up_score': 0.0241, 'bias_down_score': 0.0156} labels={'BIAS_UP': 71, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 2829} clean={'BIAS_UP': 71, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 1128}
2026-05-09 02:52:51,964 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 194, 'BIAS_DOWN': 119, 'BIAS_NEUTRAL': 5579}, 'dollar': {'BIAS_UP': 266, 'BIAS_DOWN': 207, 'BIAS_NEUTRAL': 8365}, 'gold': {'BIAS_UP': 100, 'BIAS_DOWN': 35, 'BIAS_NEUTRAL': 2995}}
2026-05-09 02:52:51,964 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0329, 'bias_down_score': 0.0202}, 'dollar': {'bias_up_score': 0.0301, 'bias_down_score': 0.0234}, 'gold': {'bias_up_score': 0.0319, 'bias_down_score': 0.0112}}
2026-05-09 02:52:51,964 INFO Regime[4H mode=htf_bias] label distribution by year: {2020: {'BIAS_UP': 278, 'BIAS_DOWN': 175, 'BIAS_NEUTRAL': 8278}, 2021: {'BIAS_UP': 282, 'BIAS_DOWN': 186, 'BIAS_NEUTRAL': 8623}, 2022: {'BIAS_UP': 0, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 38}}
2026-05-09 02:52:51,964 INFO Regime[4H mode=htf_bias] score means by year: {2020: {'bias_up_score': 0.0318, 'bias_down_score': 0.02}, 2021: {'bias_up_score': 0.031, 'bias_down_score': 0.0205}, 2022: {'bias_up_score': 0.0, 'bias_down_score': 0.0}}
2026-05-09 02:52:52,048 INFO Loaded XAUUSD/4H split=val fold=fold_002: 1596 bars (2022-01-04 → 2023-01-03)
2026-05-09 02:52:52,049 INFO Loaded EURUSD/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
2026-05-09 02:52:52,050 INFO Loaded USDJPY/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
2026-05-09 02:52:52,050 INFO Loaded EURJPY/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
2026-05-09 02:52:52,051 INFO Loaded GBPJPY/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
2026-05-09 02:52:52,052 INFO Loaded GBPUSD/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
2026-05-09 02:52:52,061 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:52,065 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:52,066 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:52,066 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:52,066 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:52:52,067 INFO Loaded XAUUSD/4H split=val fold=fold_002: 1596 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:52,286 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 38, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1528}  ambiguous=938 (total=1596) horizon=12
2026-05-09 02:52:52,288 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected XAUUSD — 1546 samples (group=gold) score_means={'bias_up_score': 0.0246, 'bias_down_score': 0.0194} labels={'BIAS_UP': 38, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1478} clean={'BIAS_UP': 38, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 568}
2026-05-09 02:52:52,399 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:52,403 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:52,405 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:52,405 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:52,406 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:52,407 INFO Loaded EURUSD/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:52,605 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 11, 'BIAS_DOWN': 64, 'BIAS_NEUTRAL': 1436}  ambiguous=853 (total=1511) horizon=12
2026-05-09 02:52:52,608 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected EURUSD — 1461 samples (group=dollar) score_means={'bias_up_score': 0.0075, 'bias_down_score': 0.0438} labels={'BIAS_UP': 11, 'BIAS_DOWN': 64, 'BIAS_NEUTRAL': 1386} clean={'BIAS_UP': 11, 'BIAS_DOWN': 64, 'BIAS_NEUTRAL': 566}
2026-05-09 02:52:52,716 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:52,718 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:52,719 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:52,719 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:52,720 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:52,721 INFO Loaded USDJPY/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:52,914 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 143, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 1346}  ambiguous=890 (total=1511) horizon=12
2026-05-09 02:52:52,917 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected USDJPY — 1461 samples (group=dollar) score_means={'bias_up_score': 0.0979, 'bias_down_score': 0.0151} labels={'BIAS_UP': 143, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 1296} clean={'BIAS_UP': 143, 'BIAS_DOWN': 22, 'BIAS_NEUTRAL': 440}
2026-05-09 02:52:53,024 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:53,027 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:53,027 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:53,028 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:53,028 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:53,029 INFO Loaded EURJPY/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:53,227 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 92, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 1406}  ambiguous=879 (total=1511) horizon=12
2026-05-09 02:52:53,229 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected EURJPY — 1461 samples (group=cross) score_means={'bias_up_score': 0.063, 'bias_down_score': 0.0089} labels={'BIAS_UP': 92, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 1356} clean={'BIAS_UP': 92, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 498}
2026-05-09 02:52:53,334 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:53,337 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:53,338 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:53,338 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:53,338 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:53,339 INFO Loaded GBPJPY/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:53,534 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 75, 'BIAS_DOWN': 26, 'BIAS_NEUTRAL': 1410}  ambiguous=856 (total=1511) horizon=12
2026-05-09 02:52:53,537 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected GBPJPY — 1461 samples (group=cross) score_means={'bias_up_score': 0.0513, 'bias_down_score': 0.0178} labels={'BIAS_UP': 75, 'BIAS_DOWN': 26, 'BIAS_NEUTRAL': 1360} clean={'BIAS_UP': 75, 'BIAS_DOWN': 26, 'BIAS_NEUTRAL': 522}
2026-05-09 02:52:53,644 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:53,647 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:53,647 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:53,648 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:53,648 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:52:53,649 INFO Loaded GBPUSD/4H split=val fold=fold_002: 1511 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:52:53,840 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 4, 'BIAS_DOWN': 70, 'BIAS_NEUTRAL': 1437}  ambiguous=862 (total=1511) horizon=12
2026-05-09 02:52:53,843 INFO Regime[4H mode=htf_bias split=val fold=fold_002]: collected GBPUSD — 1461 samples (group=dollar) score_means={'bias_up_score': 0.0027, 'bias_down_score': 0.0479} labels={'BIAS_UP': 4, 'BIAS_DOWN': 70, 'BIAS_NEUTRAL': 1387} clean={'BIAS_UP': 4, 'BIAS_DOWN': 70, 'BIAS_NEUTRAL': 555}
2026-05-09 02:52:53,941 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 167, 'BIAS_DOWN': 39, 'BIAS_NEUTRAL': 2716}, 'dollar': {'BIAS_UP': 158, 'BIAS_DOWN': 156, 'BIAS_NEUTRAL': 4069}, 'gold': {'BIAS_UP': 38, 'BIAS_DOWN': 30, 'BIAS_NEUTRAL': 1478}}
2026-05-09 02:52:53,941 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0572, 'bias_down_score': 0.0133}, 'dollar': {'bias_up_score': 0.036, 'bias_down_score': 0.0356}, 'gold': {'bias_up_score': 0.0246, 'bias_down_score': 0.0194}}
2026-05-09 02:52:53,942 INFO Regime[4H mode=htf_bias] label distribution by year: {2022: {'BIAS_UP': 363, 'BIAS_DOWN': 225, 'BIAS_NEUTRAL': 8195}, 2023: {'BIAS_UP': 0, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 68}}
2026-05-09 02:52:53,942 INFO Regime[4H mode=htf_bias] score means by year: {2022: {'bias_up_score': 0.0413, 'bias_down_score': 0.0256}, 2023: {'bias_up_score': 0.0, 'bias_down_score': 0.0}}
2026-05-09 02:52:54,021 INFO Regime phase HTF dataset build fold=fold_002: 4.1s (train=17860 val=8851)
2026-05-09 02:52:54,025 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=24, n_classes=2)
2026-05-09 02:52:54,026 INFO Regime 4H/htf_bias warm start enabled from existing weights
2026-05-09 02:52:54,028 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=17860 val=8851 train_labels={'BIAS_UP': 560, 'BIAS_DOWN': 361, 'BIAS_NEUTRAL': 16939} val_labels={'BIAS_UP': 363, 'BIAS_DOWN': 225, 'BIAS_NEUTRAL': 8263}
2026-05-09 02:52:54,028 INFO RegimeClassifier[mode=htf_bias]: warm start HTF score head
2026-05-09 02:52:54,028 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-09 02:52:54,029 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 20.0, 'bias_down_score': 20.0}
2026-05-09 02:52:54,585 INFO Regime HTF score epoch  1/50 — tr=0.3950 va=0.5123 acc=0.822 bal=0.920 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.981, 'BIAS_DOWN': 0.969, 'BIAS_NEUTRAL': 0.811} precision={'BIAS_UP': 0.33, 'BIAS_DOWN': 0.206, 'BIAS_NEUTRAL': 0.998}
2026-05-09 02:52:55,130 INFO Regime HTF score epoch  2/50 — tr=0.3929 va=0.5147 bal=0.923
2026-05-09 02:52:55,707 INFO Regime HTF score epoch  3/50 — tr=0.3929 va=0.5159 bal=0.924
2026-05-09 02:52:56,261 INFO Regime HTF score epoch  4/50 — tr=0.3888 va=0.5151 bal=0.924
2026-05-09 02:52:56,818 INFO Regime HTF score epoch  5/50 — tr=0.3837 va=0.5147 acc=0.823 bal=0.924 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.978, 'BIAS_DOWN': 0.982, 'BIAS_NEUTRAL': 0.812} precision={'BIAS_UP': 0.346, 'BIAS_DOWN': 0.2, 'BIAS_NEUTRAL': 0.998}
2026-05-09 02:52:57,396 INFO Regime HTF score epoch  6/50 — tr=0.3868 va=0.5136 bal=0.924
2026-05-09 02:52:57,979 INFO Regime HTF score epoch  7/50 — tr=0.3816 va=0.5089 bal=0.924
2026-05-09 02:52:58,569 INFO Regime HTF score epoch  8/50 — tr=0.3742 va=0.5092 bal=0.924
2026-05-09 02:52:59,185 INFO Regime HTF score epoch  9/50 — tr=0.3760 va=0.5051 bal=0.923
2026-05-09 02:52:59,779 INFO Regime HTF score epoch 10/50 — tr=0.3773 va=0.5021 acc=0.827 bal=0.924 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.972, 'BIAS_DOWN': 0.982, 'BIAS_NEUTRAL': 0.817} precision={'BIAS_UP': 0.354, 'BIAS_DOWN': 0.203, 'BIAS_NEUTRAL': 0.998}
2026-05-09 02:52:59,779 INFO Regime HTF score early stop at epoch 10
2026-05-09 02:53:00,323 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.850 margin=0.000 precision={'BIAS_UP': 0.33, 'BIAS_DOWN': 0.206, 'BIAS_NEUTRAL': 0.998} recall={'BIAS_UP': 0.981, 'BIAS_DOWN': 0.969, 'BIAS_NEUTRAL': 0.811} f1={'BIAS_UP': 0.494, 'BIAS_DOWN': 0.34, 'BIAS_NEUTRAL': 0.895} confusion=[[356, 0, 7], [0, 218, 7], [722, 839, 6702]] score_mae={'bias_up_score': 0.1923, 'bias_down_score': 0.2076} pred_share={'BIAS_UP': 0.1218, 'BIAS_DOWN': 0.1194, 'BIAS_NEUTRAL': 0.7588}
2026-05-09 02:53:00,324 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.33, 'BIAS_DOWN': 0.206, 'BIAS_NEUTRAL': 0.998} min_precision=0.300 recall={'BIAS_UP': 0.981, 'BIAS_DOWN': 0.969, 'BIAS_NEUTRAL': 0.811} min_recall=0.100 f1={'BIAS_UP': 0.494, 'BIAS_DOWN': 0.34, 'BIAS_NEUTRAL': 0.895} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_DOWN'] weak_recall=[] weak_f1=[] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-09 02:53:00,327 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-09 02:53:00,328 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-09 02:53:00,328 INFO Regime phase HTF train fold=fold_002: 6.3s
2026-05-09 02:53:00,430 INFO Regime HTF complete fold=fold_002: acc=0.822 bal=0.920 train=17860 val=8851 per_class={'BIAS_UP': 0.981, 'BIAS_DOWN': 0.969, 'BIAS_NEUTRAL': 0.811} precision={'BIAS_UP': 0.33, 'BIAS_DOWN': 0.206, 'BIAS_NEUTRAL': 0.998} threshold=0.850 margin=0.000
2026-05-09 02:53:00,432 INFO Loaded GBPUSD/4H split=train fold=fold_002: 2996 bars (2020-01-06 → 2022-01-03)
2026-05-09 02:53:00,532 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 71, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 2879}  ambiguous=1724 (total=2996) horizon=12
2026-05-09 02:53:00,534 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on GBPUSD 4H:
{'BIAS_UP': 3.736842105263158, 'BIAS_DOWN': 4.6, 'BIAS_NEUTRAL': 95.96666666666667}
2026-05-09 02:53:00,536 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (all labels):
{'BIAS_UP': {'n': 71, 'mean': 0.0010475634664007492, 'mean_over_std': 0.5956451009901603}, 'BIAS_DOWN': {'n': 46, 'mean': -0.002272462121201956, 'mean_over_std': -0.45169703431230573}, 'BIAS_NEUTRAL': {'n': 2878, 'mean': 2.2850386371321218e-05, 'mean_over_std': 0.010071421024711761}}
2026-05-09 02:53:00,537 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 71, 'mean': 0.0010475634664007492, 'mean_over_std': 0.5956451009901603}, 'BIAS_DOWN': {'n': 46, 'mean': -0.002272462121201956, 'mean_over_std': -0.45169703431230573}, 'BIAS_NEUTRAL': {'n': 1155, 'mean': -3.989525060506483e-06, 'mean_over_std': -0.0020758907147755464}}
2026-05-09 02:53:00,540 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-09 02:53:00,543 INFO Loaded XAUUSD/1H split=train fold=fold_002: 11775 bars (2020-01-06 → 2022-01-03)
2026-05-09 02:53:00,544 INFO Loaded EURUSD/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
2026-05-09 02:53:00,545 INFO Loaded USDJPY/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
2026-05-09 02:53:00,547 INFO Loaded EURJPY/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
2026-05-09 02:53:00,548 INFO Loaded GBPJPY/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
2026-05-09 02:53:00,549 INFO Loaded GBPUSD/1H split=train fold=fold_002: 11691 bars (2020-01-06 → 2022-01-03)
2026-05-09 02:53:00,559 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:53:00,562 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:53:00,563 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:53:00,564 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:53:00,564 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:53:00,566 INFO Loaded XAUUSD/1H split=train fold=fold_002: 11775 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:53:00,927 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected XAUUSD — 11725 samples (group=gold) score_means={'trend_score': 0.4817, 'range_score': 0.2418, 'chop_score': 0.4772, 'volatility_percentile': 0.3667, 'consolidation_score': 0.1995}
2026-05-09 02:53:01,043 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:01,047 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:01,049 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:01,050 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:01,050 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:01,052 INFO Loaded EURUSD/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:53:01,393 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected EURUSD — 11642 samples (group=dollar) score_means={'trend_score': 0.4896, 'range_score': 0.2324, 'chop_score': 0.4597, 'volatility_percentile': 0.3849, 'consolidation_score': 0.1841}
2026-05-09 02:53:01,509 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:01,511 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:01,512 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:01,512 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:01,512 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:01,514 INFO Loaded USDJPY/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:53:01,856 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected USDJPY — 11642 samples (group=dollar) score_means={'trend_score': 0.4905, 'range_score': 0.2324, 'chop_score': 0.4655, 'volatility_percentile': 0.3784, 'consolidation_score': 0.1968}
2026-05-09 02:53:01,964 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:01,967 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:01,967 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:01,968 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:01,968 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:01,971 INFO Loaded EURJPY/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:53:02,303 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected EURJPY — 11642 samples (group=cross) score_means={'trend_score': 0.4884, 'range_score': 0.2345, 'chop_score': 0.4693, 'volatility_percentile': 0.3824, 'consolidation_score': 0.1897}
2026-05-09 02:53:02,415 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:02,417 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:02,418 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:02,418 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:02,419 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:02,420 INFO Loaded GBPJPY/1H split=train fold=fold_002: 11692 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:53:02,754 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected GBPJPY — 11642 samples (group=cross) score_means={'trend_score': 0.4783, 'range_score': 0.2365, 'chop_score': 0.4744, 'volatility_percentile': 0.3783, 'consolidation_score': 0.1953}
2026-05-09 02:53:02,863 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:02,865 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:02,866 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:02,866 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:02,866 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:02,868 INFO Loaded GBPUSD/1H split=train fold=fold_002: 11691 bars (2020-01-06 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:53:03,199 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_002]: collected GBPUSD — 11641 samples (group=dollar) score_means={'trend_score': 0.4904, 'range_score': 0.231, 'chop_score': 0.4614, 'volatility_percentile': 0.3769, 'consolidation_score': 0.1885}
2026-05-09 02:53:03,301 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4834, 'range_score': 0.2355, 'chop_score': 0.4718, 'volatility_percentile': 0.3804, 'consolidation_score': 0.1925}, 'dollar': {'trend_score': 0.4902, 'range_score': 0.2319, 'chop_score': 0.4622, 'volatility_percentile': 0.3801, 'consolidation_score': 0.1898}, 'gold': {'trend_score': 0.4817, 'range_score': 0.2418, 'chop_score': 0.4772, 'volatility_percentile': 0.3667, 'consolidation_score': 0.1995}}
2026-05-09 02:53:03,301 INFO Regime[1H mode=ltf_behaviour] score means by year: {2020: {'trend_score': 0.4858, 'range_score': 0.2322, 'chop_score': 0.4686, 'volatility_percentile': 0.3784, 'consolidation_score': 0.1934}, 2021: {'trend_score': 0.4872, 'range_score': 0.2372, 'chop_score': 0.4672, 'volatility_percentile': 0.377, 'consolidation_score': 0.192}, 2022: {'trend_score': 0.4767, 'range_score': 0.272, 'chop_score': 0.483, 'volatility_percentile': 0.4806, 'consolidation_score': 0.0234}}
2026-05-09 02:53:03,384 INFO Loaded XAUUSD/1H split=val fold=fold_002: 5914 bars (2022-01-04 → 2023-01-03)
2026-05-09 02:53:03,386 INFO Loaded EURUSD/1H split=val fold=fold_002: 5897 bars (2022-01-04 → 2023-01-03)
2026-05-09 02:53:03,387 INFO Loaded USDJPY/1H split=val fold=fold_002: 5895 bars (2022-01-04 → 2023-01-03)
2026-05-09 02:53:03,388 INFO Loaded EURJPY/1H split=val fold=fold_002: 5893 bars (2022-01-04 → 2023-01-03)
2026-05-09 02:53:03,389 INFO Loaded GBPJPY/1H split=val fold=fold_002: 5895 bars (2022-01-04 → 2023-01-03)
2026-05-09 02:53:03,391 INFO Loaded GBPUSD/1H split=val fold=fold_002: 5895 bars (2022-01-04 → 2023-01-03)
2026-05-09 02:53:03,400 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:53:03,403 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:53:03,404 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:53:03,405 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:53:03,405 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:53:03,407 INFO Loaded XAUUSD/1H split=val fold=fold_002: 5914 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:53:03,677 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected XAUUSD — 5864 samples (group=gold) score_means={'trend_score': 0.4904, 'range_score': 0.2349, 'chop_score': 0.465, 'volatility_percentile': 0.3828, 'consolidation_score': 0.1876}
2026-05-09 02:53:03,788 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:03,791 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:03,792 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:03,792 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:03,793 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:03,794 INFO Loaded EURUSD/1H split=val fold=fold_002: 5897 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:53:04,052 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected EURUSD — 5847 samples (group=dollar) score_means={'trend_score': 0.4803, 'range_score': 0.2444, 'chop_score': 0.47, 'volatility_percentile': 0.3951, 'consolidation_score': 0.1781}
2026-05-09 02:53:04,158 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:04,160 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:04,161 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:04,161 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:04,162 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:04,164 INFO Loaded USDJPY/1H split=val fold=fold_002: 5895 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:53:04,407 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected USDJPY — 5845 samples (group=dollar) score_means={'trend_score': 0.5188, 'range_score': 0.2217, 'chop_score': 0.4472, 'volatility_percentile': 0.398, 'consolidation_score': 0.1782}
2026-05-09 02:53:04,518 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:04,522 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:04,523 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:04,523 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:04,524 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:04,525 INFO Loaded EURJPY/1H split=val fold=fold_002: 5893 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:53:04,775 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected EURJPY — 5843 samples (group=cross) score_means={'trend_score': 0.5036, 'range_score': 0.2299, 'chop_score': 0.4561, 'volatility_percentile': 0.4037, 'consolidation_score': 0.1685}
2026-05-09 02:53:04,882 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:04,885 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:04,885 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:04,886 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:04,886 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:04,888 INFO Loaded GBPJPY/1H split=val fold=fold_002: 5895 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:53:05,131 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected GBPJPY — 5845 samples (group=cross) score_means={'trend_score': 0.4766, 'range_score': 0.2379, 'chop_score': 0.4728, 'volatility_percentile': 0.3937, 'consolidation_score': 0.1772}
2026-05-09 02:53:05,239 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:05,241 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:05,242 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:05,242 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:05,243 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:05,244 INFO Loaded GBPUSD/1H split=val fold=fold_002: 5895 bars (2022-01-04 → 2023-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:53:05,491 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_002]: collected GBPUSD — 5845 samples (group=dollar) score_means={'trend_score': 0.4678, 'range_score': 0.246, 'chop_score': 0.476, 'volatility_percentile': 0.3971, 'consolidation_score': 0.179}
2026-05-09 02:53:05,591 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4901, 'range_score': 0.2339, 'chop_score': 0.4644, 'volatility_percentile': 0.3987, 'consolidation_score': 0.1729}, 'dollar': {'trend_score': 0.489, 'range_score': 0.2374, 'chop_score': 0.4644, 'volatility_percentile': 0.3968, 'consolidation_score': 0.1784}, 'gold': {'trend_score': 0.4904, 'range_score': 0.2349, 'chop_score': 0.465, 'volatility_percentile': 0.3828, 'consolidation_score': 0.1876}}
2026-05-09 02:53:05,591 INFO Regime[1H mode=ltf_behaviour] score means by year: {2022: {'trend_score': 0.489, 'range_score': 0.2362, 'chop_score': 0.4649, 'volatility_percentile': 0.3943, 'consolidation_score': 0.1786}, 2023: {'trend_score': 0.5684, 'range_score': 0.1853, 'chop_score': 0.415, 'volatility_percentile': 0.4956, 'consolidation_score': 0.1128}}
2026-05-09 02:53:05,673 INFO Regime phase LTF dataset build fold=fold_002: 5.1s (train=69934 val=35089)
2026-05-09 02:53:05,679 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=31, n_classes=5)
2026-05-09 02:53:05,679 INFO Regime 1H/ltf_behaviour warm start enabled from existing weights
2026-05-09 02:53:05,689 INFO RegimeClassifier[mode=ltf_behaviour]: warm start score head
2026-05-09 02:53:05,689 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-09 02:53:05,845 INFO Regime score epoch  1/50 — tr=0.0039 va=0.0011 mae={'trend_score': 0.0226, 'range_score': 0.0371, 'chop_score': 0.0242, 'volatility_percentile': 0.0157, 'consolidation_score': 0.0217}
2026-05-09 02:53:05,990 INFO Regime score epoch  2/50 — tr=0.0039 va=0.0011
2026-05-09 02:53:06,143 INFO Regime score epoch  3/50 — tr=0.0039 va=0.0011
2026-05-09 02:53:06,295 INFO Regime score epoch  4/50 — tr=0.0039 va=0.0011
2026-05-09 02:53:06,441 INFO Regime score epoch  5/50 — tr=0.0038 va=0.0011 mae={'trend_score': 0.0221, 'range_score': 0.0362, 'chop_score': 0.0237, 'volatility_percentile': 0.0153, 'consolidation_score': 0.0222}
2026-05-09 02:53:06,591 INFO Regime score epoch  6/50 — tr=0.0038 va=0.0010
2026-05-09 02:53:06,741 INFO Regime score epoch  7/50 — tr=0.0038 va=0.0010
2026-05-09 02:53:06,886 INFO Regime score epoch  8/50 — tr=0.0038 va=0.0010
2026-05-09 02:53:07,042 INFO Regime score epoch  9/50 — tr=0.0038 va=0.0010
2026-05-09 02:53:07,189 INFO Regime score epoch 10/50 — tr=0.0038 va=0.0010 mae={'trend_score': 0.0207, 'range_score': 0.0354, 'chop_score': 0.0226, 'volatility_percentile': 0.0161, 'consolidation_score': 0.0223}
2026-05-09 02:53:07,346 INFO Regime score epoch 11/50 — tr=0.0038 va=0.0010
2026-05-09 02:53:07,502 INFO Regime score epoch 12/50 — tr=0.0038 va=0.0010
2026-05-09 02:53:07,658 INFO Regime score epoch 13/50 — tr=0.0038 va=0.0010
2026-05-09 02:53:07,806 INFO Regime score epoch 14/50 — tr=0.0038 va=0.0010
2026-05-09 02:53:07,952 INFO Regime score epoch 15/50 — tr=0.0038 va=0.0010 mae={'trend_score': 0.0206, 'range_score': 0.0357, 'chop_score': 0.0221, 'volatility_percentile': 0.0154, 'consolidation_score': 0.021}
2026-05-09 02:53:08,105 INFO Regime score epoch 16/50 — tr=0.0038 va=0.0010
2026-05-09 02:53:08,260 INFO Regime score epoch 17/50 — tr=0.0038 va=0.0010
2026-05-09 02:53:08,407 INFO Regime score epoch 18/50 — tr=0.0037 va=0.0010
2026-05-09 02:53:08,561 INFO Regime score epoch 19/50 — tr=0.0037 va=0.0010
2026-05-09 02:53:08,705 INFO Regime score epoch 20/50 — tr=0.0038 va=0.0010 mae={'trend_score': 0.0204, 'range_score': 0.0351, 'chop_score': 0.0221, 'volatility_percentile': 0.0155, 'consolidation_score': 0.021}
2026-05-09 02:53:08,858 INFO Regime score epoch 21/50 — tr=0.0037 va=0.0010
2026-05-09 02:53:09,001 INFO Regime score epoch 22/50 — tr=0.0037 va=0.0010
2026-05-09 02:53:09,155 INFO Regime score epoch 23/50 — tr=0.0037 va=0.0010
2026-05-09 02:53:09,320 INFO Regime score epoch 24/50 — tr=0.0037 va=0.0010
2026-05-09 02:53:09,478 INFO Regime score epoch 25/50 — tr=0.0037 va=0.0010 mae={'trend_score': 0.0207, 'range_score': 0.0349, 'chop_score': 0.0223, 'volatility_percentile': 0.015, 'consolidation_score': 0.021}
2026-05-09 02:53:09,630 INFO Regime score epoch 26/50 — tr=0.0037 va=0.0010
2026-05-09 02:53:09,778 INFO Regime score epoch 27/50 — tr=0.0037 va=0.0010
2026-05-09 02:53:09,930 INFO Regime score epoch 28/50 — tr=0.0037 va=0.0010
2026-05-09 02:53:10,085 INFO Regime score epoch 29/50 — tr=0.0037 va=0.0010
2026-05-09 02:53:10,234 INFO Regime score epoch 30/50 — tr=0.0037 va=0.0010 mae={'trend_score': 0.0199, 'range_score': 0.035, 'chop_score': 0.0217, 'volatility_percentile': 0.0153, 'consolidation_score': 0.0209}
2026-05-09 02:53:10,384 INFO Regime score epoch 31/50 — tr=0.0037 va=0.0010
2026-05-09 02:53:10,533 INFO Regime score epoch 32/50 — tr=0.0037 va=0.0010
2026-05-09 02:53:10,678 INFO Regime score epoch 33/50 — tr=0.0037 va=0.0010
2026-05-09 02:53:10,823 INFO Regime score epoch 34/50 — tr=0.0037 va=0.0010
2026-05-09 02:53:10,969 INFO Regime score epoch 35/50 — tr=0.0037 va=0.0010 mae={'trend_score': 0.0199, 'range_score': 0.035, 'chop_score': 0.0218, 'volatility_percentile': 0.015, 'consolidation_score': 0.0207}
2026-05-09 02:53:11,115 INFO Regime score epoch 36/50 — tr=0.0037 va=0.0010
2026-05-09 02:53:11,267 INFO Regime score epoch 37/50 — tr=0.0037 va=0.0010
2026-05-09 02:53:11,421 INFO Regime score epoch 38/50 — tr=0.0037 va=0.0010
2026-05-09 02:53:11,574 INFO Regime score epoch 39/50 — tr=0.0037 va=0.0010
2026-05-09 02:53:11,729 INFO Regime score epoch 40/50 — tr=0.0037 va=0.0010 mae={'trend_score': 0.0198, 'range_score': 0.0352, 'chop_score': 0.0217, 'volatility_percentile': 0.0152, 'consolidation_score': 0.0208}
2026-05-09 02:53:11,729 INFO Regime score early stop at epoch 40
2026-05-09 02:53:11,754 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0198, 'range_score': 0.0346, 'chop_score': 0.0214, 'volatility_percentile': 0.0152, 'consolidation_score': 0.0206} mse={'trend_score': 0.00067, 'range_score': 0.00197, 'chop_score': 0.00074, 'volatility_percentile': 0.00041, 'consolidation_score': 0.00097} corr={'trend_score': 0.9935, 'range_score': 0.9542, 'chop_score': 0.9908, 'volatility_percentile': 0.9958, 'consolidation_score': 0.9892} pred_std={'trend_score': 0.2258, 'range_score': 0.1342, 'chop_score': 0.1861, 'volatility_percentile': 0.2214, 'consolidation_score': 0.2092} target_std={'trend_score': 0.2278, 'range_score': 0.1469, 'chop_score': 0.1946, 'volatility_percentile': 0.2199, 'consolidation_score': 0.2121}
2026-05-09 02:53:11,758 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-09 02:53:11,759 INFO Regime phase LTF train fold=fold_002: 6.1s
2026-05-09 02:53:11,861 INFO Regime LTF complete fold=fold_002: score_accuracy=0.978, train=69934 val=35089 mae={'trend_score': 0.0198, 'range_score': 0.0346, 'chop_score': 0.0214, 'volatility_percentile': 0.0152, 'consolidation_score': 0.0206}
2026-05-09 02:53:11,863 INFO Loaded GBPUSD/1H split=train fold=fold_002: 11691 bars (2020-01-06 → 2022-01-03)
2026-05-09 02:53:12,001 INFO Regime[1H mode=ltf_behaviour fold=fold_002] LTF score diagnostics on GBPUSD:
{'trend_score': {'mean': 0.49, 'q10': 0.193, 'q50': 0.4874, 'q90': 0.7863}, 'range_score': {'mean': 0.2321, 'q10': 0.0521, 'q50': 0.2116, 'q90': 0.4302}, 'chop_score': {'mean': 0.462, 'q10': 0.2201, 'q50': 0.4494, 'q90': 0.7227}, 'volatility_percentile': {'mean': 0.377, 'q10': 0.0976, 'q50': 0.3668, 'q90': 0.6772}, 'consolidation_score': {'mean': 0.1877, 'q10': 0.0, 'q50': 0.1276, 'q90': 0.4995}}
2026-05-09 02:53:12,004 INFO Regime retrain total: 83.5s (395556 train+val samples)
2026-05-09 02:53:12,007 INFO Retrain complete. Total wall-clock: 83.5s
2026-05-09 02:53:15,965 INFO Model regime: SUCCESS
2026-05-09 02:53:15,965 INFO --- Training gru ---
2026-05-09 02:53:15,966 INFO Running retrain --model gru
2026-05-09 02:53:16,271 INFO retrain environment: KAGGLE
2026-05-09 02:53:17,915 INFO Device: CUDA (2 GPU(s))
2026-05-09 02:53:17,926 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-09 02:53:17,926 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-09 02:53:17,926 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-09 02:53:17,927 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-09 02:53:17,927 INFO Retrain data split: train
2026-05-09 02:53:17,927 INFO Retrain rolling fold selector: latest
2026-05-09 02:53:17,928 INFO === GRU-LSTM retrain (timeframes: ['15M']) ===
2026-05-09 02:53:18,081 INFO NumExpr defaulting to 4 threads.
2026-05-09 02:53:18,276 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-09 02:53:18,276 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-09 02:53:18,276 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-09 02:53:18,608 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-05-09 02:53:18,609 INFO GRU multi-symbol training (Kaggle mode): 6 symbols × ['15M']
2026-05-09 02:53:18,610 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260509_025318
2026-05-09 02:53:18,614 INFO GRU feature contract unchanged (input_size=71) — incremental retrain
2026-05-09 02:53:18,614 INFO GRU warm start enabled from existing weights: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 02:53:18,898 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:53:18,925 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:53:18,941 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:53:18,952 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-09 02:53:19,023 INFO Split boundaries loaded fold=fold_002/3 — train 2020-01-06→2022-01-03  val 2022-01-04→2023-01-03  test 2023-08-07→2025-08-05
2026-05-09 02:53:19,027 INFO Loaded XAUUSD/15M split=train fold=latest: 47096 bars (2020-01-06 → 2022-01-03)
2026-05-09 02:53:19,289 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:19,309 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:19,323 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:19,329 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:19,365 INFO Loaded EURUSD/15M split=train fold=latest: 46766 bars (2020-01-06 → 2022-01-03)
2026-05-09 02:53:19,583 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:19,604 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:19,618 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:19,625 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:19,661 INFO Loaded USDJPY/15M split=train fold=latest: 46768 bars (2020-01-06 → 2022-01-03)
2026-05-09 02:53:19,881 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:19,903 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:19,918 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:19,925 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:19,962 INFO Loaded EURJPY/15M split=train fold=latest: 46767 bars (2020-01-06 → 2022-01-03)
2026-05-09 02:53:20,171 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:20,191 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:20,205 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:20,212 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:20,250 INFO Loaded GBPJPY/15M split=train fold=latest: 46765 bars (2020-01-06 → 2022-01-03)
2026-05-09 02:53:20,454 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:20,474 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:20,488 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:20,495 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 02:53:20,532 INFO Loaded GBPUSD/15M split=train fold=latest: 46764 bars (2020-01-06 → 2022-01-03)
2026-05-09 02:53:20,645 INFO train_multi: 6 segments, ~272746 total bars
2026-05-09 02:53:20,645 INFO train_multi: training ALL 6 segments across TFs ['15M'] in one combined pass
2026-05-09 02:53:20,645 INFO train_multi: building combined dataset for TF=ALL (6 segments)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:53:23,913 INFO train_multi TF=ALL: 272566 sequences across 6 segments
2026-05-09 02:53:23,913 INFO train_multi TF=ALL: estimated peak RAM = 4645 MB (train=218050 val=54516 n_feat=71 seq_len=30)
2026-05-09 02:53:24,544 INFO train_multi TF=ALL: train=218050 val=54516 (2326 MB tensors)
2026-05-09 02:53:27,067 INFO train_multi TF=ALL: warm-start detected — using CosineAnnealingLR (lr=3e-05, patience=12)
2026-05-09 02:53:35,873 INFO train_multi TF=ALL epoch 1/50 train=0.6511 val=0.6490 dir_acc=0.632 dir_n=54516
2026-05-09 02:53:35,878 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 02:53:35,879 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 02:53:35,879 INFO train_multi TF=ALL: new best val=0.6490 — saved
2026-05-09 02:53:42,071 INFO train_multi TF=ALL epoch 2/50 train=0.6513 val=0.6490 dir_acc=0.631 dir_n=54516
2026-05-09 02:53:42,076 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 02:53:42,076 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 02:53:42,076 INFO train_multi TF=ALL: new best val=0.6490 — saved
2026-05-09 02:53:48,086 INFO train_multi TF=ALL epoch 3/50 train=0.6512 val=0.6492 dir_acc=0.631 dir_n=54516
2026-05-09 02:53:54,210 INFO train_multi TF=ALL epoch 4/50 train=0.6511 val=0.6491 dir_acc=0.632 dir_n=54516
2026-05-09 02:54:00,291 INFO train_multi TF=ALL epoch 5/50 train=0.6511 val=0.6494 dir_acc=0.630 dir_n=54516
2026-05-09 02:54:06,255 INFO train_multi TF=ALL epoch 6/50 train=0.6512 val=0.6493 dir_acc=0.630 dir_n=54516
2026-05-09 02:54:12,322 INFO train_multi TF=ALL epoch 7/50 train=0.6511 val=0.6492 dir_acc=0.632 dir_n=54516
2026-05-09 02:54:18,297 INFO train_multi TF=ALL epoch 8/50 train=0.6506 val=0.6492 dir_acc=0.632 dir_n=54516
2026-05-09 02:54:24,361 INFO train_multi TF=ALL epoch 9/50 train=0.6506 val=0.6491 dir_acc=0.631 dir_n=54516
2026-05-09 02:54:30,594 INFO train_multi TF=ALL epoch 10/50 train=0.6508 val=0.6492 dir_acc=0.631 dir_n=54516
2026-05-09 02:54:36,596 INFO train_multi TF=ALL epoch 11/50 train=0.6507 val=0.6493 dir_acc=0.631 dir_n=54516
2026-05-09 02:54:42,671 INFO train_multi TF=ALL epoch 12/50 train=0.6507 val=0.6491 dir_acc=0.632 dir_n=54516
2026-05-09 02:54:48,736 INFO train_multi TF=ALL epoch 13/50 train=0.6504 val=0.6491 dir_acc=0.632 dir_n=54516
2026-05-09 02:54:54,779 INFO train_multi TF=ALL epoch 14/50 train=0.6502 val=0.6493 dir_acc=0.630 dir_n=54516
2026-05-09 02:54:54,779 INFO train_multi TF=ALL early stop at epoch 14
2026-05-09 02:54:54,913 INFO Retrain complete. Total wall-clock: 97.0s
2026-05-09 02:54:56,366 INFO Model gru: SUCCESS
2026-05-09 02:54:56,367 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 02:54:56,367 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-09 02:54:56,367 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-09 02:54:56,367 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-05-09 02:54:56,367 INFO   [DORMANT] rl_ppo — RL_ENABLED=false, skipping
2026-05-09 02:54:56,367 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-05-09 02:54:56,367 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer']
2026-05-09 02:54:56,370 INFO Saved 18 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Clean Quality/RL source: Backtest on train window ===
  START Train-window backtest for Quality/RL labels
2026-05-09 02:54:57,034 INFO === STEP 6: BACKTEST (train) ===
2026-05-09 02:54:57,035 INFO BT_WINDOW=train — train-window backtest: 2020-01-06 → 2022-01-03 (clean Quality/RL labels)
2026-05-09 02:54:57,036 INFO ================================================================
  ROUND 0 / 3
================================================================
2026-05-09 02:54:57,036 INFO Round 0 — running backtest: 2020-01-06 → 2022-01-03 (ml_trader, shared ML cache)
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
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:56:16,199 ERROR ML cache: sequence feature build failed for XAUUSD: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
2026-05-09 02:56:16,205 ERROR _precompute_ml_cache failed for XAUUSD: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:56:16,751 ERROR ML cache: sequence feature build failed for USDJPY: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
2026-05-09 02:56:16,751 ERROR _precompute_ml_cache failed for USDJPY: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
2026-05-09 02:56:17,139 ERROR ML cache: sequence feature build failed for EURUSD: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
2026-05-09 02:56:17,159 ERROR _precompute_ml_cache failed for EURUSD: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:56:17,832 ERROR ML cache: sequence feature build failed for EURJPY: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
2026-05-09 02:56:17,833 ERROR _precompute_ml_cache failed for EURJPY: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 02:56:21,522 ERROR ML cache: sequence feature build failed for GBPJPY: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
2026-05-09 02:56:21,523 ERROR _precompute_ml_cache failed for GBPJPY: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
2026-05-09 02:56:21,637 ERROR ML cache: sequence feature build failed for GBPUSD: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
2026-05-09 02:56:21,637 ERROR _precompute_ml_cache failed for GBPUSD: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
Traceback (most recent call last):
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 2358, in _backtest_trader
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
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 2341, in _build_cache_sym
    return sym, _precompute_ml_cache(
                ^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 1581, in _precompute_ml_cache
    feat_df = fe._build_sequence_df(df, htf, symbol=symbol)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py", line 724, in _build_sequence_df
    out["mtf_5m_rsi"]      = _htf_series(df_5m, "5M",
                             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py", line 700, in _htf_series
    raise ValueError(
ValueError: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 3840, in <module>
    main()
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 3654, in main
    result = _backtest_trader("ml_trader", symbols, pm, bt_start, bt_end,
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 2362, in _backtest_trader
    raise RuntimeError(f"ML cache build failed for {sym}: {exc}") from exc
RuntimeError: ML cache build failed for XAUUSD: _build_sequence_df: HTF frame 5M has non-finite warmup or alignment gaps
2026-05-09 02:56:22,333 ERROR Backtest failed (rc=1) — check trading-engine/logs/backtest_*.log
2026-05-09 02:56:22,333 ERROR Round 0 backtest failed: backtest exited 1
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