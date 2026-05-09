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
2026-05-09 07:08:34,961 INFO Loading feature-engineered data...
2026-05-09 07:08:35,594 INFO Loaded 221743 rows, 202 features
2026-05-09 07:08:35,596 INFO Data span: 2016-01-04 -> 2025-08-05  (9.6 years)
2026-05-09 07:08:35,598 INFO Fold 000 train 2016-01-04 -> 2018-01-03 (47088 bars), val 2018-01-04 -> 2019-01-03 (23448 bars)
2026-05-09 07:08:35,598 INFO Fold 001 train 2019-01-04 -> 2020-12-31 (46636 bars), val 2021-01-04 -> 2022-01-03 (23507 bars)
2026-05-09 07:08:35,599 INFO No leakage confirmed: every fold ends before final 2-year blind test

=== SPLIT COMPLETE (FIXED CALENDAR, no shuffling) ===
  Folds:            2 fixed folds (2y train + 1y val, step=3y)
  Selected:   fold_001 for train.parquet / validation.parquet aliases
  Train:       46,636 bars  2019-01-04 -> 2020-12-31
  Validation:  23,507 bars  2021-01-04 -> 2022-01-03
  Test:        46,792 bars  2023-08-07 -> 2025-08-05  <- Blind / Round 2
  Features:   202
  Leakage check: PASS
  DONE  Step 5 - Split

  Data split (fixed_calendar):
    train          46636 bars  2019-01-04 → 2020-12-31
    validation     23507 bars  2021-01-04 → 2022-01-03
    test           46792 bars  2023-08-07 → 2025-08-05

=== Phase 7a: Train GRU + Regime (train set only) ===
  START Step 7a - GRU+Regime
2026-05-09 07:08:38,363 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-05-09 07:08:38,363 INFO --- Training regime ---
2026-05-09 07:08:38,363 INFO Running retrain --model regime
2026-05-09 07:08:38,547 INFO retrain environment: KAGGLE
2026-05-09 07:08:40,183 INFO Device: CUDA (2 GPU(s))
2026-05-09 07:08:40,194 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-09 07:08:40,194 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-09 07:08:40,194 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-09 07:08:40,196 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-09 07:08:40,196 INFO Retrain data split: train
2026-05-09 07:08:40,196 INFO Retrain rolling fold selector: latest
2026-05-09 07:08:40,197 INFO === RegimeClassifier retrain (hierarchical: HTF 2-score bias + LTF 5-score behaviour) ===
2026-05-09 07:08:40,369 INFO NumExpr defaulting to 4 threads.
2026-05-09 07:08:40,587 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-09 07:08:40,587 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-09 07:08:40,587 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-09 07:08:40,588 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-09 07:08:40,643 INFO Regime rolling folds selected: ['fold_001']
2026-05-09 07:08:40,643 INFO === Regime rolling fold 1/1: fold_001 ===
2026-05-09 07:08:40,643 INFO Regime: training HTF bias score head (bias_up_score/bias_down_score)...
2026-05-09 07:08:40,684 INFO Split boundaries loaded fold=fold_001/2 — train 2019-01-04→2020-12-31  val 2021-01-04→2022-01-03  test 2023-08-07→2025-08-05
2026-05-09 07:08:40,685 INFO Loaded XAUUSD/4H split=train fold=fold_001: 3176 bars (2019-01-04 → 2020-12-31)
2026-05-09 07:08:40,700 INFO Loaded EURUSD/4H split=train fold=fold_001: 2990 bars (2019-01-04 → 2020-12-31)
2026-05-09 07:08:40,715 INFO Loaded USDJPY/4H split=train fold=fold_001: 2991 bars (2019-01-04 → 2020-12-31)
2026-05-09 07:08:40,730 INFO Loaded EURJPY/4H split=train fold=fold_001: 2990 bars (2019-01-04 → 2020-12-31)
2026-05-09 07:08:40,745 INFO Loaded GBPJPY/4H split=train fold=fold_001: 2991 bars (2019-01-04 → 2020-12-31)
2026-05-09 07:08:40,758 INFO Loaded GBPUSD/4H split=train fold=fold_001: 2991 bars (2019-01-04 → 2020-12-31)
2026-05-09 07:08:40,994 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:08:41,062 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:08:41,086 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:08:41,086 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:08:41,096 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:08:41,097 INFO Loaded XAUUSD/4H split=train fold=fold_001: 3176 bars (2019-01-04 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:08:41,347 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 162, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 3001}  ambiguous=1876 (total=3176) horizon=12
2026-05-09 07:08:41,350 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected XAUUSD — 3126 samples (group=gold) score_means={'bias_up_score': 0.0518, 'bias_down_score': 0.0042} labels={'BIAS_UP': 162, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 2951} clean={'BIAS_UP': 162, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 1099}
2026-05-09 07:08:41,505 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:41,538 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:41,557 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:41,558 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:41,565 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:41,566 INFO Loaded EURUSD/4H split=train fold=fold_001: 2990 bars (2019-01-04 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:08:41,795 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 100, 'BIAS_DOWN': 52, 'BIAS_NEUTRAL': 2838}  ambiguous=1763 (total=2990) horizon=12
2026-05-09 07:08:41,798 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected EURUSD — 2940 samples (group=dollar) score_means={'bias_up_score': 0.034, 'bias_down_score': 0.0177} labels={'BIAS_UP': 100, 'BIAS_DOWN': 52, 'BIAS_NEUTRAL': 2788} clean={'BIAS_UP': 100, 'BIAS_DOWN': 52, 'BIAS_NEUTRAL': 1064}
2026-05-09 07:08:41,967 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:42,001 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:42,022 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:42,022 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:42,031 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:42,032 INFO Loaded USDJPY/4H split=train fold=fold_001: 2991 bars (2019-01-04 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:08:42,258 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 31, 'BIAS_DOWN': 97, 'BIAS_NEUTRAL': 2863}  ambiguous=1697 (total=2991) horizon=12
2026-05-09 07:08:42,261 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected USDJPY — 2941 samples (group=dollar) score_means={'bias_up_score': 0.0105, 'bias_down_score': 0.033} labels={'BIAS_UP': 31, 'BIAS_DOWN': 97, 'BIAS_NEUTRAL': 2813} clean={'BIAS_UP': 31, 'BIAS_DOWN': 97, 'BIAS_NEUTRAL': 1144}
2026-05-09 07:08:42,414 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:42,448 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:42,468 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:42,468 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:42,475 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:42,476 INFO Loaded EURJPY/4H split=train fold=fold_001: 2990 bars (2019-01-04 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:08:42,680 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 67, 'BIAS_DOWN': 34, 'BIAS_NEUTRAL': 2889}  ambiguous=1781 (total=2990) horizon=12
2026-05-09 07:08:42,683 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected EURJPY — 2940 samples (group=cross) score_means={'bias_up_score': 0.0228, 'bias_down_score': 0.0116} labels={'BIAS_UP': 67, 'BIAS_DOWN': 34, 'BIAS_NEUTRAL': 2839} clean={'BIAS_UP': 67, 'BIAS_DOWN': 34, 'BIAS_NEUTRAL': 1085}
2026-05-09 07:08:42,827 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:42,862 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:42,883 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:42,883 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:42,890 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:42,891 INFO Loaded GBPJPY/4H split=train fold=fold_001: 2991 bars (2019-01-04 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:08:43,094 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 74, 'BIAS_DOWN': 100, 'BIAS_NEUTRAL': 2817}  ambiguous=1758 (total=2991) horizon=12
2026-05-09 07:08:43,097 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected GBPJPY — 2941 samples (group=cross) score_means={'bias_up_score': 0.0252, 'bias_down_score': 0.034} labels={'BIAS_UP': 74, 'BIAS_DOWN': 100, 'BIAS_NEUTRAL': 2767} clean={'BIAS_UP': 74, 'BIAS_DOWN': 100, 'BIAS_NEUTRAL': 1034}
2026-05-09 07:08:43,241 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:43,273 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:43,291 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:43,291 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:43,299 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:43,300 INFO Loaded GBPUSD/4H split=train fold=fold_001: 2991 bars (2019-01-04 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:08:43,505 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 73, 'BIAS_DOWN': 62, 'BIAS_NEUTRAL': 2856}  ambiguous=1819 (total=2991) horizon=12
2026-05-09 07:08:43,507 INFO Regime[4H mode=htf_bias split=train fold=fold_001]: collected GBPUSD — 2941 samples (group=dollar) score_means={'bias_up_score': 0.0248, 'bias_down_score': 0.0211} labels={'BIAS_UP': 73, 'BIAS_DOWN': 62, 'BIAS_NEUTRAL': 2806} clean={'BIAS_UP': 73, 'BIAS_DOWN': 62, 'BIAS_NEUTRAL': 1010}
2026-05-09 07:08:43,567 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 141, 'BIAS_DOWN': 134, 'BIAS_NEUTRAL': 5606}, 'dollar': {'BIAS_UP': 204, 'BIAS_DOWN': 211, 'BIAS_NEUTRAL': 8407}, 'gold': {'BIAS_UP': 162, 'BIAS_DOWN': 13, 'BIAS_NEUTRAL': 2951}}
2026-05-09 07:08:43,568 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.024, 'bias_down_score': 0.0228}, 'dollar': {'bias_up_score': 0.0231, 'bias_down_score': 0.0239}, 'gold': {'bias_up_score': 0.0518, 'bias_down_score': 0.0042}}
2026-05-09 07:08:43,568 INFO Regime[4H mode=htf_bias] label distribution by year: {2019: {'BIAS_UP': 209, 'BIAS_DOWN': 177, 'BIAS_NEUTRAL': 8332}, 2020: {'BIAS_UP': 298, 'BIAS_DOWN': 181, 'BIAS_NEUTRAL': 8632}}
2026-05-09 07:08:43,568 INFO Regime[4H mode=htf_bias] score means by year: {2019: {'bias_up_score': 0.024, 'bias_down_score': 0.0203}, 2020: {'bias_up_score': 0.0327, 'bias_down_score': 0.0199}}
2026-05-09 07:08:43,609 INFO Loaded XAUUSD/4H split=val fold=fold_001: 1597 bars (2021-01-04 → 2022-01-03)
2026-05-09 07:08:43,610 INFO Loaded EURUSD/4H split=val fold=fold_001: 1506 bars (2021-01-04 → 2022-01-03)
2026-05-09 07:08:43,611 INFO Loaded USDJPY/4H split=val fold=fold_001: 1506 bars (2021-01-04 → 2022-01-03)
2026-05-09 07:08:43,611 INFO Loaded EURJPY/4H split=val fold=fold_001: 1506 bars (2021-01-04 → 2022-01-03)
2026-05-09 07:08:43,612 INFO Loaded GBPJPY/4H split=val fold=fold_001: 1506 bars (2021-01-04 → 2022-01-03)
2026-05-09 07:08:43,613 INFO Loaded GBPUSD/4H split=val fold=fold_001: 1506 bars (2021-01-04 → 2022-01-03)
2026-05-09 07:08:43,622 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:08:43,626 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:08:43,627 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:08:43,627 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:08:43,628 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:08:43,628 INFO Loaded XAUUSD/4H split=val fold=fold_001: 1597 bars (2021-01-04 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:08:43,829 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 24, 'BIAS_DOWN': 28, 'BIAS_NEUTRAL': 1545}  ambiguous=955 (total=1597) horizon=12
2026-05-09 07:08:43,831 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected XAUUSD — 1547 samples (group=gold) score_means={'bias_up_score': 0.0155, 'bias_down_score': 0.0181} labels={'BIAS_UP': 24, 'BIAS_DOWN': 28, 'BIAS_NEUTRAL': 1495} clean={'BIAS_UP': 24, 'BIAS_DOWN': 28, 'BIAS_NEUTRAL': 574}
2026-05-09 07:08:43,898 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:43,903 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:43,903 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:43,904 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:43,904 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:43,905 INFO Loaded EURUSD/4H split=val fold=fold_001: 1506 bars (2021-01-04 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:08:44,091 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 34, 'BIAS_DOWN': 70, 'BIAS_NEUTRAL': 1402}  ambiguous=824 (total=1506) horizon=12
2026-05-09 07:08:44,093 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected EURUSD — 1456 samples (group=dollar) score_means={'bias_up_score': 0.0234, 'bias_down_score': 0.0481} labels={'BIAS_UP': 34, 'BIAS_DOWN': 70, 'BIAS_NEUTRAL': 1352} clean={'BIAS_UP': 34, 'BIAS_DOWN': 70, 'BIAS_NEUTRAL': 560}
2026-05-09 07:08:44,155 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:44,157 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:44,158 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:44,158 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:44,159 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:44,159 INFO Loaded USDJPY/4H split=val fold=fold_001: 1506 bars (2021-01-04 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:08:44,341 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 64, 'BIAS_DOWN': 4, 'BIAS_NEUTRAL': 1438}  ambiguous=919 (total=1506) horizon=12
2026-05-09 07:08:44,343 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected USDJPY — 1456 samples (group=dollar) score_means={'bias_up_score': 0.044, 'bias_down_score': 0.0027} labels={'BIAS_UP': 64, 'BIAS_DOWN': 4, 'BIAS_NEUTRAL': 1388} clean={'BIAS_UP': 64, 'BIAS_DOWN': 4, 'BIAS_NEUTRAL': 495}
2026-05-09 07:08:44,407 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:44,409 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:44,410 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:44,411 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:44,411 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:44,412 INFO Loaded EURJPY/4H split=val fold=fold_001: 1506 bars (2021-01-04 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:08:44,599 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 40, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 1420}  ambiguous=860 (total=1506) horizon=12
2026-05-09 07:08:44,602 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected EURJPY — 1456 samples (group=cross) score_means={'bias_up_score': 0.0275, 'bias_down_score': 0.0316} labels={'BIAS_UP': 40, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 1370} clean={'BIAS_UP': 40, 'BIAS_DOWN': 46, 'BIAS_NEUTRAL': 548}
2026-05-09 07:08:44,667 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:44,669 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:44,670 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:44,670 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:44,670 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:44,671 INFO Loaded GBPJPY/4H split=val fold=fold_001: 1506 bars (2021-01-04 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:08:44,857 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 86, 'BIAS_DOWN': 15, 'BIAS_NEUTRAL': 1405}  ambiguous=857 (total=1506) horizon=12
2026-05-09 07:08:44,859 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected GBPJPY — 1456 samples (group=cross) score_means={'bias_up_score': 0.0591, 'bias_down_score': 0.0103} labels={'BIAS_UP': 86, 'BIAS_DOWN': 15, 'BIAS_NEUTRAL': 1355} clean={'BIAS_UP': 86, 'BIAS_DOWN': 15, 'BIAS_NEUTRAL': 527}
2026-05-09 07:08:44,922 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:44,924 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:44,925 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:44,926 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:44,926 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:08:44,927 INFO Loaded GBPUSD/4H split=val fold=fold_001: 1506 bars (2021-01-04 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:08:45,157 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 32, 'BIAS_DOWN': 24, 'BIAS_NEUTRAL': 1450}  ambiguous=820 (total=1506) horizon=12
2026-05-09 07:08:45,160 INFO Regime[4H mode=htf_bias split=val fold=fold_001]: collected GBPUSD — 1456 samples (group=dollar) score_means={'bias_up_score': 0.022, 'bias_down_score': 0.0165} labels={'BIAS_UP': 32, 'BIAS_DOWN': 24, 'BIAS_NEUTRAL': 1400} clean={'BIAS_UP': 32, 'BIAS_DOWN': 24, 'BIAS_NEUTRAL': 606}
2026-05-09 07:08:45,227 INFO Regime[4H mode=htf_bias] label distribution by symbol group: {'cross': {'BIAS_UP': 126, 'BIAS_DOWN': 61, 'BIAS_NEUTRAL': 2725}, 'dollar': {'BIAS_UP': 130, 'BIAS_DOWN': 98, 'BIAS_NEUTRAL': 4140}, 'gold': {'BIAS_UP': 24, 'BIAS_DOWN': 28, 'BIAS_NEUTRAL': 1495}}
2026-05-09 07:08:45,227 INFO Regime[4H mode=htf_bias] score means by symbol group: {'cross': {'bias_up_score': 0.0433, 'bias_down_score': 0.0209}, 'dollar': {'bias_up_score': 0.0298, 'bias_down_score': 0.0224}, 'gold': {'bias_up_score': 0.0155, 'bias_down_score': 0.0181}}
2026-05-09 07:08:45,227 INFO Regime[4H mode=htf_bias] label distribution by year: {2021: {'BIAS_UP': 280, 'BIAS_DOWN': 187, 'BIAS_NEUTRAL': 8322}, 2022: {'BIAS_UP': 0, 'BIAS_DOWN': 0, 'BIAS_NEUTRAL': 38}}
2026-05-09 07:08:45,227 INFO Regime[4H mode=htf_bias] score means by year: {2021: {'bias_up_score': 0.0319, 'bias_down_score': 0.0213}, 2022: {'bias_up_score': 0.0, 'bias_down_score': 0.0}}
2026-05-09 07:08:45,274 INFO Regime phase HTF dataset build fold=fold_001: 4.6s (train=17829 val=8827)
2026-05-09 07:08:45,274 INFO Regime 4H/htf_bias cold start: no existing weights found
2026-05-09 07:08:45,277 INFO RegimeClassifier[mode=htf_bias]: HTF score samples train=17829 val=8827 train_labels={'BIAS_UP': 507, 'BIAS_DOWN': 358, 'BIAS_NEUTRAL': 16964} val_labels={'BIAS_UP': 280, 'BIAS_DOWN': 187, 'BIAS_NEUTRAL': 8360}
2026-05-09 07:08:45,553 INFO RegimeClassifier[mode=htf_bias]: cold start HTF score head
2026-05-09 07:08:45,553 INFO RegimeClassifier HTF score head: DataParallel across 2 GPUs
2026-05-09 07:08:45,554 INFO RegimeClassifier[mode=htf_bias]: HTF BCE pos_weight={'bias_up_score': 20.0, 'bias_down_score': 20.0}
2026-05-09 07:08:50,593 INFO Regime HTF score epoch  1/50 — tr=1.7788 va=1.0538 acc=0.947 bal=0.333 threshold=0.35 margin=0.25 recall={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 1.0} precision={'BIAS_UP': 0.0, 'BIAS_DOWN': 0.0, 'BIAS_NEUTRAL': 0.947}
2026-05-09 07:08:51,135 INFO Regime HTF score epoch  2/50 — tr=1.7286 va=1.0436 bal=0.333
2026-05-09 07:08:51,663 INFO Regime HTF score epoch  3/50 — tr=1.6785 va=1.0307 bal=0.462
2026-05-09 07:08:52,253 INFO Regime HTF score epoch  4/50 — tr=1.5821 va=1.0055 bal=0.375
2026-05-09 07:08:52,793 INFO Regime HTF score epoch  5/50 — tr=1.4635 va=0.9652 acc=0.900 bal=0.568 threshold=0.55 margin=0.15 recall={'BIAS_UP': 0.368, 'BIAS_DOWN': 0.406, 'BIAS_NEUTRAL': 0.928} precision={'BIAS_UP': 0.288, 'BIAS_DOWN': 0.181, 'BIAS_NEUTRAL': 0.964}
2026-05-09 07:08:53,338 INFO Regime HTF score epoch  6/50 — tr=1.3244 va=0.9154 bal=0.665
2026-05-09 07:08:53,881 INFO Regime HTF score epoch  7/50 — tr=1.1585 va=0.8697 bal=0.654
2026-05-09 07:08:54,420 INFO Regime HTF score epoch  8/50 — tr=1.0301 va=0.8449 bal=0.843
2026-05-09 07:08:54,956 INFO Regime HTF score epoch  9/50 — tr=0.9264 va=0.8388 bal=0.797
2026-05-09 07:08:55,502 INFO Regime HTF score epoch 10/50 — tr=0.8462 va=0.8384 acc=0.829 bal=0.790 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.821, 'BIAS_DOWN': 0.717, 'BIAS_NEUTRAL': 0.832} precision={'BIAS_UP': 0.239, 'BIAS_DOWN': 0.166, 'BIAS_NEUTRAL': 0.985}
2026-05-09 07:08:56,065 INFO Regime HTF score epoch 11/50 — tr=0.8016 va=0.8335 bal=0.833
2026-05-09 07:08:56,603 INFO Regime HTF score epoch 12/50 — tr=0.7543 va=0.8209 bal=0.850
2026-05-09 07:08:57,146 INFO Regime HTF score epoch 13/50 — tr=0.7245 va=0.8059 bal=0.849
2026-05-09 07:08:57,679 INFO Regime HTF score epoch 14/50 — tr=0.7063 va=0.7921 bal=0.847
2026-05-09 07:08:58,213 INFO Regime HTF score epoch 15/50 — tr=0.6871 va=0.7798 acc=0.804 bal=0.848 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.871, 'BIAS_DOWN': 0.872, 'BIAS_NEUTRAL': 0.8} precision={'BIAS_UP': 0.231, 'BIAS_DOWN': 0.159, 'BIAS_NEUTRAL': 0.991}
2026-05-09 07:08:58,744 INFO Regime HTF score epoch 16/50 — tr=0.6639 va=0.7689 bal=0.849
2026-05-09 07:08:59,273 INFO Regime HTF score epoch 17/50 — tr=0.6499 va=0.7620 bal=0.852
2026-05-09 07:08:59,812 INFO Regime HTF score epoch 18/50 — tr=0.6231 va=0.7507 bal=0.854
2026-05-09 07:09:00,346 INFO Regime HTF score epoch 19/50 — tr=0.6236 va=0.7418 bal=0.854
2026-05-09 07:09:00,889 INFO Regime HTF score epoch 20/50 — tr=0.6108 va=0.7313 acc=0.806 bal=0.858 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.879, 'BIAS_DOWN': 0.893, 'BIAS_NEUTRAL': 0.802} precision={'BIAS_UP': 0.238, 'BIAS_DOWN': 0.161, 'BIAS_NEUTRAL': 0.992}
2026-05-09 07:09:01,422 INFO Regime HTF score epoch 21/50 — tr=0.6024 va=0.7226 bal=0.858
2026-05-09 07:09:01,973 INFO Regime HTF score epoch 22/50 — tr=0.5837 va=0.7180 bal=0.859
2026-05-09 07:09:02,537 INFO Regime HTF score epoch 23/50 — tr=0.5727 va=0.7089 bal=0.860
2026-05-09 07:09:03,077 INFO Regime HTF score epoch 24/50 — tr=0.5683 va=0.7015 bal=0.860
2026-05-09 07:09:03,627 INFO Regime HTF score epoch 25/50 — tr=0.5568 va=0.6934 acc=0.810 bal=0.860 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.882, 'BIAS_DOWN': 0.893, 'BIAS_NEUTRAL': 0.806} precision={'BIAS_UP': 0.24, 'BIAS_DOWN': 0.166, 'BIAS_NEUTRAL': 0.992}
2026-05-09 07:09:04,158 INFO Regime HTF score epoch 26/50 — tr=0.5491 va=0.6883 bal=0.862
2026-05-09 07:09:04,694 INFO Regime HTF score epoch 27/50 — tr=0.5435 va=0.6837 bal=0.862
2026-05-09 07:09:05,228 INFO Regime HTF score epoch 28/50 — tr=0.5364 va=0.6803 bal=0.866
2026-05-09 07:09:05,756 INFO Regime HTF score epoch 29/50 — tr=0.5282 va=0.6753 bal=0.867
2026-05-09 07:09:06,342 INFO Regime HTF score epoch 30/50 — tr=0.5270 va=0.6728 acc=0.810 bal=0.867 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.886, 'BIAS_DOWN': 0.909, 'BIAS_NEUTRAL': 0.805} precision={'BIAS_UP': 0.241, 'BIAS_DOWN': 0.167, 'BIAS_NEUTRAL': 0.993}
2026-05-09 07:09:06,876 INFO Regime HTF score epoch 31/50 — tr=0.5177 va=0.6691 bal=0.867
2026-05-09 07:09:07,406 INFO Regime HTF score epoch 32/50 — tr=0.5135 va=0.6639 bal=0.866
2026-05-09 07:09:07,933 INFO Regime HTF score epoch 33/50 — tr=0.5150 va=0.6599 bal=0.866
2026-05-09 07:09:08,469 INFO Regime HTF score epoch 34/50 — tr=0.5084 va=0.6596 bal=0.868
2026-05-09 07:09:09,006 INFO Regime HTF score epoch 35/50 — tr=0.5098 va=0.6580 acc=0.812 bal=0.871 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.886, 'BIAS_DOWN': 0.92, 'BIAS_NEUTRAL': 0.807} precision={'BIAS_UP': 0.243, 'BIAS_DOWN': 0.17, 'BIAS_NEUTRAL': 0.993}
2026-05-09 07:09:09,542 INFO Regime HTF score epoch 36/50 — tr=0.5055 va=0.6543 bal=0.870
2026-05-09 07:09:10,073 INFO Regime HTF score epoch 37/50 — tr=0.4993 va=0.6523 bal=0.871
2026-05-09 07:09:10,604 INFO Regime HTF score epoch 38/50 — tr=0.5008 va=0.6499 bal=0.869
2026-05-09 07:09:11,144 INFO Regime HTF score epoch 39/50 — tr=0.4952 va=0.6448 bal=0.868
2026-05-09 07:09:11,687 INFO Regime HTF score epoch 40/50 — tr=0.4980 va=0.6473 acc=0.813 bal=0.871 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.889, 'BIAS_DOWN': 0.914, 'BIAS_NEUTRAL': 0.808} precision={'BIAS_UP': 0.243, 'BIAS_DOWN': 0.171, 'BIAS_NEUTRAL': 0.993}
2026-05-09 07:09:12,251 INFO Regime HTF score epoch 41/50 — tr=0.4939 va=0.6464 bal=0.871
2026-05-09 07:09:12,787 INFO Regime HTF score epoch 42/50 — tr=0.4997 va=0.6416 bal=0.869
2026-05-09 07:09:13,321 INFO Regime HTF score epoch 43/50 — tr=0.4989 va=0.6426 bal=0.870
2026-05-09 07:09:13,854 INFO Regime HTF score epoch 44/50 — tr=0.4941 va=0.6456 bal=0.870
2026-05-09 07:09:14,394 INFO Regime HTF score epoch 45/50 — tr=0.4860 va=0.6458 acc=0.812 bal=0.872 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.889, 'BIAS_DOWN': 0.92, 'BIAS_NEUTRAL': 0.807} precision={'BIAS_UP': 0.242, 'BIAS_DOWN': 0.172, 'BIAS_NEUTRAL': 0.993}
2026-05-09 07:09:14,924 INFO Regime HTF score epoch 46/50 — tr=0.4935 va=0.6453 bal=0.872
2026-05-09 07:09:15,454 INFO Regime HTF score epoch 47/50 — tr=0.4915 va=0.6456 bal=0.871
2026-05-09 07:09:15,984 INFO Regime HTF score epoch 48/50 — tr=0.5013 va=0.6447 bal=0.869
2026-05-09 07:09:16,524 INFO Regime HTF score epoch 49/50 — tr=0.4945 va=0.6446 bal=0.872
2026-05-09 07:09:17,059 INFO Regime HTF score epoch 50/50 — tr=0.4923 va=0.6453 acc=0.812 bal=0.872 threshold=0.85 margin=0.00 recall={'BIAS_UP': 0.889, 'BIAS_DOWN': 0.92, 'BIAS_NEUTRAL': 0.807} precision={'BIAS_UP': 0.243, 'BIAS_DOWN': 0.171, 'BIAS_NEUTRAL': 0.993}
2026-05-09 07:09:17,553 INFO RegimeClassifier[mode=htf_bias] HTF score validation threshold=0.850 margin=0.000 precision={'BIAS_UP': 0.241, 'BIAS_DOWN': 0.173, 'BIAS_NEUTRAL': 0.993} recall={'BIAS_UP': 0.889, 'BIAS_DOWN': 0.92, 'BIAS_NEUTRAL': 0.808} f1={'BIAS_UP': 0.379, 'BIAS_DOWN': 0.292, 'BIAS_NEUTRAL': 0.891} confusion=[[249, 0, 31], [0, 172, 15], [785, 820, 6755]] score_mae={'bias_up_score': 0.3047, 'bias_down_score': 0.3142} pred_share={'BIAS_UP': 0.1171, 'BIAS_DOWN': 0.1124, 'BIAS_NEUTRAL': 0.7705}
2026-05-09 07:09:17,554 WARNING Regime HTF directional score validation below acceptance floor: precision={'BIAS_UP': 0.241, 'BIAS_DOWN': 0.173, 'BIAS_NEUTRAL': 0.993} min_precision=0.300 recall={'BIAS_UP': 0.889, 'BIAS_DOWN': 0.92, 'BIAS_NEUTRAL': 0.808} min_recall=0.100 f1={'BIAS_UP': 0.379, 'BIAS_DOWN': 0.292, 'BIAS_NEUTRAL': 0.891} min_f1=0.150 min_neutral_recall=0.500 weak_precision=['BIAS_UP', 'BIAS_DOWN'] weak_recall=[] weak_f1=[] weak_neutral=False. Saving weights anyway so the pipeline can progress.
2026-05-09 07:09:17,560 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-09 07:09:17,560 INFO RegimeClassifier[4H] HTF score head saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-09 07:09:17,561 INFO Regime phase HTF train fold=fold_001: 32.3s
2026-05-09 07:09:17,660 INFO Regime HTF complete fold=fold_001: acc=0.813 bal=0.872 train=17829 val=8827 per_class={'BIAS_UP': 0.889, 'BIAS_DOWN': 0.92, 'BIAS_NEUTRAL': 0.808} precision={'BIAS_UP': 0.241, 'BIAS_DOWN': 0.173, 'BIAS_NEUTRAL': 0.993} threshold=0.850 margin=0.000
2026-05-09 07:09:17,661 INFO Loaded GBPUSD/4H split=train fold=fold_001: 2991 bars (2019-01-04 → 2020-12-31)
2026-05-09 07:09:17,756 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 73, 'BIAS_DOWN': 62, 'BIAS_NEUTRAL': 2856}  ambiguous=1819 (total=2991) horizon=12
2026-05-09 07:09:17,763 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on GBPUSD 4H:
{'BIAS_UP': 2.7037037037037037, 'BIAS_DOWN': 4.133333333333334, 'BIAS_NEUTRAL': 66.4186046511628}
2026-05-09 07:09:17,766 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (all labels):
{'BIAS_UP': {'n': 73, 'mean': 0.0011794172656542634, 'mean_over_std': 0.5078713211251215}, 'BIAS_DOWN': {'n': 62, 'mean': -0.001698431456718425, 'mean_over_std': -0.38468439923623815}, 'BIAS_NEUTRAL': {'n': 2855, 'mean': 3.648950989833485e-05, 'mean_over_std': 0.014949978381420216}}
2026-05-09 07:09:17,767 INFO Regime[4H mode=htf_bias] return separation on GBPUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 73, 'mean': 0.0011794172656542634, 'mean_over_std': 0.5078713211251215}, 'BIAS_DOWN': {'n': 62, 'mean': -0.001698431456718425, 'mean_over_std': -0.38468439923623815}, 'BIAS_NEUTRAL': {'n': 1037, 'mean': -5.415793101180411e-05, 'mean_over_std': -0.025939438294562082}}
2026-05-09 07:09:17,770 INFO Regime: training LTF behaviour score head (trend/range/chop/volatility/consolidation)...
2026-05-09 07:09:17,773 INFO Loaded XAUUSD/1H split=train fold=fold_001: 11754 bars (2019-01-04 → 2020-12-31)
2026-05-09 07:09:17,775 INFO Loaded EURUSD/1H split=train fold=fold_001: 11660 bars (2019-01-04 → 2020-12-31)
2026-05-09 07:09:17,776 INFO Loaded USDJPY/1H split=train fold=fold_001: 11664 bars (2019-01-04 → 2020-12-31)
2026-05-09 07:09:17,777 INFO Loaded EURJPY/1H split=train fold=fold_001: 11661 bars (2019-01-04 → 2020-12-31)
2026-05-09 07:09:17,779 INFO Loaded GBPJPY/1H split=train fold=fold_001: 11661 bars (2019-01-04 → 2020-12-31)
2026-05-09 07:09:17,780 INFO Loaded GBPUSD/1H split=train fold=fold_001: 11660 bars (2019-01-04 → 2020-12-31)
2026-05-09 07:09:17,799 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:09:17,807 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:09:17,810 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:09:17,810 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:09:17,811 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:09:17,815 INFO Loaded XAUUSD/1H split=train fold=fold_001: 11754 bars (2019-01-04 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:09:18,171 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected XAUUSD — 11704 samples (group=gold) score_means={'trend_score': 0.4822, 'range_score': 0.2383, 'chop_score': 0.4728, 'volatility_percentile': 0.3767, 'consolidation_score': 0.1952}
2026-05-09 07:09:18,277 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:18,282 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:18,284 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:18,284 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:18,285 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:18,287 INFO Loaded EURUSD/1H split=train fold=fold_001: 11660 bars (2019-01-04 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:09:18,605 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected EURUSD — 11610 samples (group=dollar) score_means={'trend_score': 0.4878, 'range_score': 0.2316, 'chop_score': 0.4592, 'volatility_percentile': 0.3778, 'consolidation_score': 0.189}
2026-05-09 07:09:18,709 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:18,711 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:18,712 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:18,712 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:18,712 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:18,714 INFO Loaded USDJPY/1H split=train fold=fold_001: 11664 bars (2019-01-04 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:09:19,045 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected USDJPY — 11614 samples (group=dollar) score_means={'trend_score': 0.477, 'range_score': 0.2382, 'chop_score': 0.4759, 'volatility_percentile': 0.3671, 'consolidation_score': 0.2033}
2026-05-09 07:09:19,150 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:19,152 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:19,153 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:19,153 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:19,154 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:19,155 INFO Loaded EURJPY/1H split=train fold=fold_001: 11661 bars (2019-01-04 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:09:19,469 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected EURJPY — 11611 samples (group=cross) score_means={'trend_score': 0.4891, 'range_score': 0.2329, 'chop_score': 0.4683, 'volatility_percentile': 0.3726, 'consolidation_score': 0.1958}
2026-05-09 07:09:19,573 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:19,575 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:19,576 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:19,576 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:19,577 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:19,578 INFO Loaded GBPJPY/1H split=train fold=fold_001: 11661 bars (2019-01-04 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:09:19,897 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected GBPJPY — 11611 samples (group=cross) score_means={'trend_score': 0.4782, 'range_score': 0.2377, 'chop_score': 0.478, 'volatility_percentile': 0.3851, 'consolidation_score': 0.1968}
2026-05-09 07:09:20,002 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:20,004 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:20,005 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:20,005 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:20,005 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:20,007 INFO Loaded GBPUSD/1H split=train fold=fold_001: 11660 bars (2019-01-04 → 2020-12-31)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:09:20,322 INFO Regime[1H mode=ltf_behaviour split=train fold=fold_001]: collected GBPUSD — 11610 samples (group=dollar) score_means={'trend_score': 0.4899, 'range_score': 0.227, 'chop_score': 0.4587, 'volatility_percentile': 0.3861, 'consolidation_score': 0.1853}
2026-05-09 07:09:20,421 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4836, 'range_score': 0.2353, 'chop_score': 0.4731, 'volatility_percentile': 0.3789, 'consolidation_score': 0.1963}, 'dollar': {'trend_score': 0.4849, 'range_score': 0.2323, 'chop_score': 0.4646, 'volatility_percentile': 0.377, 'consolidation_score': 0.1925}, 'gold': {'trend_score': 0.4822, 'range_score': 0.2383, 'chop_score': 0.4728, 'volatility_percentile': 0.3767, 'consolidation_score': 0.1952}}
2026-05-09 07:09:20,422 INFO Regime[1H mode=ltf_behaviour] score means by year: {2019: {'trend_score': 0.4805, 'range_score': 0.2375, 'chop_score': 0.4703, 'volatility_percentile': 0.3701, 'consolidation_score': 0.1966}, 2020: {'trend_score': 0.4874, 'range_score': 0.2311, 'chop_score': 0.4674, 'volatility_percentile': 0.3848, 'consolidation_score': 0.1919}}
2026-05-09 07:09:20,503 INFO Loaded XAUUSD/1H split=val fold=fold_001: 5914 bars (2021-01-04 → 2022-01-03)
2026-05-09 07:09:20,504 INFO Loaded EURUSD/1H split=val fold=fold_001: 5877 bars (2021-01-04 → 2022-01-03)
2026-05-09 07:09:20,506 INFO Loaded USDJPY/1H split=val fold=fold_001: 5877 bars (2021-01-04 → 2022-01-03)
2026-05-09 07:09:20,507 INFO Loaded EURJPY/1H split=val fold=fold_001: 5877 bars (2021-01-04 → 2022-01-03)
2026-05-09 07:09:20,508 INFO Loaded GBPJPY/1H split=val fold=fold_001: 5877 bars (2021-01-04 → 2022-01-03)
2026-05-09 07:09:20,509 INFO Loaded GBPUSD/1H split=val fold=fold_001: 5877 bars (2021-01-04 → 2022-01-03)
2026-05-09 07:09:20,519 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:09:20,522 INFO Loaded XAUUSD/15M split=all fold=latest: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:09:20,523 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:09:20,524 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:09:20,524 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:09:20,526 INFO Loaded XAUUSD/1H split=val fold=fold_001: 5914 bars (2021-01-04 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:09:20,783 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected XAUUSD — 5864 samples (group=gold) score_means={'trend_score': 0.4785, 'range_score': 0.2481, 'chop_score': 0.4778, 'volatility_percentile': 0.3664, 'consolidation_score': 0.1922}
2026-05-09 07:09:20,892 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:20,896 INFO Loaded EURUSD/15M split=all fold=latest: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:20,897 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:20,897 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:20,898 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:20,899 INFO Loaded EURUSD/1H split=val fold=fold_001: 5877 bars (2021-01-04 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:09:21,142 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected EURUSD — 5827 samples (group=dollar) score_means={'trend_score': 0.4811, 'range_score': 0.2409, 'chop_score': 0.4662, 'volatility_percentile': 0.3816, 'consolidation_score': 0.1843}
2026-05-09 07:09:21,244 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:21,246 INFO Loaded USDJPY/15M split=all fold=latest: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:21,247 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:21,248 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:21,248 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:21,250 INFO Loaded USDJPY/1H split=val fold=fold_001: 5877 bars (2021-01-04 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:09:21,503 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected USDJPY — 5827 samples (group=dollar) score_means={'trend_score': 0.4965, 'range_score': 0.2302, 'chop_score': 0.4597, 'volatility_percentile': 0.3816, 'consolidation_score': 0.1947}
2026-05-09 07:09:21,611 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:21,613 INFO Loaded EURJPY/15M split=all fold=latest: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:21,614 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:21,614 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:21,615 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:21,618 INFO Loaded EURJPY/1H split=val fold=fold_001: 5877 bars (2021-01-04 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:09:21,861 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected EURJPY — 5827 samples (group=cross) score_means={'trend_score': 0.4864, 'range_score': 0.2377, 'chop_score': 0.4687, 'volatility_percentile': 0.391, 'consolidation_score': 0.1812}
2026-05-09 07:09:21,968 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:21,970 INFO Loaded GBPJPY/15M split=all fold=latest: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:21,971 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:21,971 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:21,972 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:21,973 INFO Loaded GBPJPY/1H split=val fold=fold_001: 5877 bars (2021-01-04 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:09:22,241 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected GBPJPY — 5827 samples (group=cross) score_means={'trend_score': 0.489, 'range_score': 0.2354, 'chop_score': 0.4647, 'volatility_percentile': 0.3793, 'consolidation_score': 0.1845}
2026-05-09 07:09:22,347 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:22,350 INFO Loaded GBPUSD/15M split=all fold=latest: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:22,350 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:22,351 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:22,351 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:22,353 INFO Loaded GBPUSD/1H split=val fold=fold_001: 5877 bars (2021-01-04 → 2022-01-03)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:09:22,591 INFO Regime[1H mode=ltf_behaviour split=val fold=fold_001]: collected GBPUSD — 5827 samples (group=dollar) score_means={'trend_score': 0.4919, 'range_score': 0.2381, 'chop_score': 0.4654, 'volatility_percentile': 0.3725, 'consolidation_score': 0.188}
2026-05-09 07:09:22,689 INFO Regime[1H mode=ltf_behaviour] score means by symbol group: {'cross': {'trend_score': 0.4877, 'range_score': 0.2365, 'chop_score': 0.4667, 'volatility_percentile': 0.3851, 'consolidation_score': 0.1828}, 'dollar': {'trend_score': 0.4898, 'range_score': 0.2364, 'chop_score': 0.4638, 'volatility_percentile': 0.3786, 'consolidation_score': 0.189}, 'gold': {'trend_score': 0.4785, 'range_score': 0.2481, 'chop_score': 0.4778, 'volatility_percentile': 0.3664, 'consolidation_score': 0.1922}}
2026-05-09 07:09:22,689 INFO Regime[1H mode=ltf_behaviour] score means by year: {2021: {'trend_score': 0.4873, 'range_score': 0.2383, 'chop_score': 0.467, 'volatility_percentile': 0.3783, 'consolidation_score': 0.1882}, 2022: {'trend_score': 0.4767, 'range_score': 0.272, 'chop_score': 0.483, 'volatility_percentile': 0.4806, 'consolidation_score': 0.0234}}
2026-05-09 07:09:22,765 INFO Regime phase LTF dataset build fold=fold_001: 5.0s (train=69760 val=34999)
2026-05-09 07:09:22,765 INFO Regime 1H/ltf_behaviour cold start: no existing weights found
2026-05-09 07:09:22,777 INFO RegimeClassifier[mode=ltf_behaviour]: cold start score head
2026-05-09 07:09:22,777 INFO RegimeClassifier score head: DataParallel across 2 GPUs
2026-05-09 07:09:22,986 INFO Regime score epoch  1/50 — tr=0.0921 va=0.0822 mae={'trend_score': 0.186, 'range_score': 0.274, 'chop_score': 0.1745, 'volatility_percentile': 0.2274, 'consolidation_score': 0.3479}
2026-05-09 07:09:23,148 INFO Regime score epoch  2/50 — tr=0.0878 va=0.0751
2026-05-09 07:09:23,299 INFO Regime score epoch  3/50 — tr=0.0794 va=0.0651
2026-05-09 07:09:23,448 INFO Regime score epoch  4/50 — tr=0.0671 va=0.0522
2026-05-09 07:09:23,598 INFO Regime score epoch  5/50 — tr=0.0529 va=0.0397 mae={'trend_score': 0.1038, 'range_score': 0.2166, 'chop_score': 0.0894, 'volatility_percentile': 0.143, 'consolidation_score': 0.2612}
2026-05-09 07:09:23,753 INFO Regime score epoch  6/50 — tr=0.0404 va=0.0304
2026-05-09 07:09:23,903 INFO Regime score epoch  7/50 — tr=0.0313 va=0.0240
2026-05-09 07:09:24,050 INFO Regime score epoch  8/50 — tr=0.0252 va=0.0194
2026-05-09 07:09:24,202 INFO Regime score epoch  9/50 — tr=0.0212 va=0.0155
2026-05-09 07:09:24,350 INFO Regime score epoch 10/50 — tr=0.0184 va=0.0130 mae={'trend_score': 0.0703, 'range_score': 0.108, 'chop_score': 0.0644, 'volatility_percentile': 0.0529, 'consolidation_score': 0.1503}
2026-05-09 07:09:24,503 INFO Regime score epoch 11/50 — tr=0.0165 va=0.0110
2026-05-09 07:09:24,655 INFO Regime score epoch 12/50 — tr=0.0151 va=0.0095
2026-05-09 07:09:24,800 INFO Regime score epoch 13/50 — tr=0.0141 va=0.0087
2026-05-09 07:09:24,950 INFO Regime score epoch 14/50 — tr=0.0133 va=0.0078
2026-05-09 07:09:25,096 INFO Regime score epoch 15/50 — tr=0.0127 va=0.0072 mae={'trend_score': 0.0648, 'range_score': 0.072, 'chop_score': 0.0579, 'volatility_percentile': 0.0381, 'consolidation_score': 0.0995}
2026-05-09 07:09:25,242 INFO Regime score epoch 16/50 — tr=0.0121 va=0.0068
2026-05-09 07:09:25,385 INFO Regime score epoch 17/50 — tr=0.0117 va=0.0062
2026-05-09 07:09:25,531 INFO Regime score epoch 18/50 — tr=0.0113 va=0.0061
2026-05-09 07:09:25,680 INFO Regime score epoch 19/50 — tr=0.0110 va=0.0060
2026-05-09 07:09:25,824 INFO Regime score epoch 20/50 — tr=0.0106 va=0.0057 mae={'trend_score': 0.0606, 'range_score': 0.0641, 'chop_score': 0.0543, 'volatility_percentile': 0.0345, 'consolidation_score': 0.0811}
2026-05-09 07:09:25,979 INFO Regime score epoch 21/50 — tr=0.0104 va=0.0055
2026-05-09 07:09:26,129 INFO Regime score epoch 22/50 — tr=0.0102 va=0.0054
2026-05-09 07:09:26,277 INFO Regime score epoch 23/50 — tr=0.0099 va=0.0052
2026-05-09 07:09:26,428 INFO Regime score epoch 24/50 — tr=0.0098 va=0.0050
2026-05-09 07:09:26,584 INFO Regime score epoch 25/50 — tr=0.0096 va=0.0048 mae={'trend_score': 0.0573, 'range_score': 0.0608, 'chop_score': 0.0522, 'volatility_percentile': 0.0331, 'consolidation_score': 0.0691}
2026-05-09 07:09:26,742 INFO Regime score epoch 26/50 — tr=0.0095 va=0.0049
2026-05-09 07:09:26,894 INFO Regime score epoch 27/50 — tr=0.0094 va=0.0047
2026-05-09 07:09:27,063 INFO Regime score epoch 28/50 — tr=0.0092 va=0.0046
2026-05-09 07:09:27,231 INFO Regime score epoch 29/50 — tr=0.0091 va=0.0045
2026-05-09 07:09:27,380 INFO Regime score epoch 30/50 — tr=0.0090 va=0.0045 mae={'trend_score': 0.055, 'range_score': 0.06, 'chop_score': 0.0516, 'volatility_percentile': 0.0323, 'consolidation_score': 0.0644}
2026-05-09 07:09:27,529 INFO Regime score epoch 31/50 — tr=0.0090 va=0.0045
2026-05-09 07:09:27,679 INFO Regime score epoch 32/50 — tr=0.0089 va=0.0044
2026-05-09 07:09:27,823 INFO Regime score epoch 33/50 — tr=0.0088 va=0.0044
2026-05-09 07:09:27,968 INFO Regime score epoch 34/50 — tr=0.0088 va=0.0043
2026-05-09 07:09:28,114 INFO Regime score epoch 35/50 — tr=0.0087 va=0.0043 mae={'trend_score': 0.0532, 'range_score': 0.0595, 'chop_score': 0.0509, 'volatility_percentile': 0.0313, 'consolidation_score': 0.0607}
2026-05-09 07:09:28,266 INFO Regime score epoch 36/50 — tr=0.0087 va=0.0043
2026-05-09 07:09:28,415 INFO Regime score epoch 37/50 — tr=0.0086 va=0.0043
2026-05-09 07:09:28,566 INFO Regime score epoch 38/50 — tr=0.0086 va=0.0042
2026-05-09 07:09:28,721 INFO Regime score epoch 39/50 — tr=0.0085 va=0.0042
2026-05-09 07:09:28,868 INFO Regime score epoch 40/50 — tr=0.0086 va=0.0043 mae={'trend_score': 0.0531, 'range_score': 0.0606, 'chop_score': 0.0515, 'volatility_percentile': 0.0322, 'consolidation_score': 0.0596}
2026-05-09 07:09:29,020 INFO Regime score epoch 41/50 — tr=0.0085 va=0.0042
2026-05-09 07:09:29,168 INFO Regime score epoch 42/50 — tr=0.0085 va=0.0042
2026-05-09 07:09:29,314 INFO Regime score epoch 43/50 — tr=0.0085 va=0.0043
2026-05-09 07:09:29,464 INFO Regime score epoch 44/50 — tr=0.0085 va=0.0042
2026-05-09 07:09:29,614 INFO Regime score epoch 45/50 — tr=0.0085 va=0.0042 mae={'trend_score': 0.0528, 'range_score': 0.0586, 'chop_score': 0.0506, 'volatility_percentile': 0.0301, 'consolidation_score': 0.0593}
2026-05-09 07:09:29,760 INFO Regime score epoch 46/50 — tr=0.0084 va=0.0042
2026-05-09 07:09:29,913 INFO Regime score epoch 47/50 — tr=0.0085 va=0.0042
2026-05-09 07:09:30,072 INFO Regime score epoch 48/50 — tr=0.0084 va=0.0042
2026-05-09 07:09:30,226 INFO Regime score epoch 49/50 — tr=0.0084 va=0.0042
2026-05-09 07:09:30,377 INFO Regime score epoch 50/50 — tr=0.0085 va=0.0041 mae={'trend_score': 0.0524, 'range_score': 0.0585, 'chop_score': 0.0506, 'volatility_percentile': 0.0307, 'consolidation_score': 0.0577}
2026-05-09 07:09:30,401 INFO RegimeClassifier[mode=ltf_behaviour] score validation mae={'trend_score': 0.0524, 'range_score': 0.0585, 'chop_score': 0.0506, 'volatility_percentile': 0.0307, 'consolidation_score': 0.0577} mse={'trend_score': 0.00427, 'range_score': 0.00534, 'chop_score': 0.00396, 'volatility_percentile': 0.00169, 'consolidation_score': 0.00526} corr={'trend_score': 0.958, 'range_score': 0.8734, 'chop_score': 0.9463, 'volatility_percentile': 0.9819, 'consolidation_score': 0.9454} pred_std={'trend_score': 0.2038, 'range_score': 0.1432, 'chop_score': 0.1747, 'volatility_percentile': 0.2102, 'consolidation_score': 0.2006} target_std={'trend_score': 0.2242, 'range_score': 0.144, 'chop_score': 0.1929, 'volatility_percentile': 0.2133, 'consolidation_score': 0.2134}
2026-05-09 07:09:30,406 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-09 07:09:30,406 INFO Regime phase LTF train fold=fold_001: 7.6s
2026-05-09 07:09:30,505 INFO Regime LTF complete fold=fold_001: score_accuracy=0.950, train=69760 val=34999 mae={'trend_score': 0.0524, 'range_score': 0.0585, 'chop_score': 0.0506, 'volatility_percentile': 0.0307, 'consolidation_score': 0.0577}
2026-05-09 07:09:30,507 INFO Loaded GBPUSD/1H split=train fold=fold_001: 11660 bars (2019-01-04 → 2020-12-31)
2026-05-09 07:09:30,645 INFO Regime[1H mode=ltf_behaviour fold=fold_001] LTF score diagnostics on GBPUSD:
{'trend_score': {'mean': 0.4899, 'q10': 0.1862, 'q50': 0.4875, 'q90': 0.7995}, 'range_score': {'mean': 0.2279, 'q10': 0.0505, 'q50': 0.2043, 'q90': 0.4243}, 'chop_score': {'mean': 0.459, 'q10': 0.2172, 'q50': 0.4448, 'q90': 0.7214}, 'volatility_percentile': {'mean': 0.3861, 'q10': 0.0931, 'q50': 0.375, 'q90': 0.6919}, 'consolidation_score': {'mean': 0.1846, 'q10': 0.0, 'q50': 0.1125, 'q90': 0.5059}}
2026-05-09 07:09:30,648 INFO Regime retrain total: 50.5s (131415 train+val samples)
2026-05-09 07:09:30,652 INFO Retrain complete. Total wall-clock: 50.5s
2026-05-09 07:09:32,928 INFO Model regime: SUCCESS
2026-05-09 07:09:32,928 INFO --- Training gru ---
2026-05-09 07:09:32,929 INFO Running retrain --model gru
2026-05-09 07:09:33,152 INFO retrain environment: KAGGLE
2026-05-09 07:09:34,766 INFO Device: CUDA (2 GPU(s))
2026-05-09 07:09:34,777 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-09 07:09:34,778 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-09 07:09:34,778 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-09 07:09:34,778 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-09 07:09:34,778 INFO Retrain data split: train
2026-05-09 07:09:34,778 INFO Retrain rolling fold selector: latest
2026-05-09 07:09:34,779 INFO === GRU-LSTM retrain (timeframes: ['15M']) ===
2026-05-09 07:09:34,951 INFO NumExpr defaulting to 4 threads.
2026-05-09 07:09:35,154 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-09 07:09:35,154 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-09 07:09:35,154 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-09 07:09:35,155 INFO GRU multi-symbol training (Kaggle mode): 6 symbols × ['15M']
2026-05-09 07:09:35,155 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260509_070935
2026-05-09 07:09:35,158 INFO GRU cold start: no compatible existing weights found
2026-05-09 07:09:35,420 INFO Loaded XAUUSD/5M split=all fold=latest: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:09:35,449 INFO Loaded XAUUSD/1H split=all fold=latest: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:09:35,464 INFO Loaded XAUUSD/4H split=all fold=latest: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:09:35,474 INFO Loaded XAUUSD/1D split=all fold=latest: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-09 07:09:35,549 INFO Split boundaries loaded fold=fold_001/2 — train 2019-01-04→2020-12-31  val 2021-01-04→2022-01-03  test 2023-08-07→2025-08-05
2026-05-09 07:09:35,552 INFO Loaded XAUUSD/15M split=train fold=latest: 46996 bars (2019-01-04 → 2020-12-31)
2026-05-09 07:09:35,772 INFO Loaded EURUSD/5M split=all fold=latest: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:35,789 INFO Loaded EURUSD/1H split=all fold=latest: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:35,803 INFO Loaded EURUSD/4H split=all fold=latest: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:35,810 INFO Loaded EURUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:35,846 INFO Loaded EURUSD/15M split=train fold=latest: 46636 bars (2019-01-04 → 2020-12-31)
2026-05-09 07:09:36,048 INFO Loaded USDJPY/5M split=all fold=latest: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:36,067 INFO Loaded USDJPY/1H split=all fold=latest: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:36,081 INFO Loaded USDJPY/4H split=all fold=latest: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:36,088 INFO Loaded USDJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:36,125 INFO Loaded USDJPY/15M split=train fold=latest: 46656 bars (2019-01-04 → 2020-12-31)
2026-05-09 07:09:36,349 INFO Loaded EURJPY/5M split=all fold=latest: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:36,369 INFO Loaded EURJPY/1H split=all fold=latest: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:36,384 INFO Loaded EURJPY/4H split=all fold=latest: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:36,391 INFO Loaded EURJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:36,427 INFO Loaded EURJPY/15M split=train fold=latest: 46640 bars (2019-01-04 → 2020-12-31)
2026-05-09 07:09:36,629 INFO Loaded GBPJPY/5M split=all fold=latest: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:36,650 INFO Loaded GBPJPY/1H split=all fold=latest: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:36,664 INFO Loaded GBPJPY/4H split=all fold=latest: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:36,671 INFO Loaded GBPJPY/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:36,706 INFO Loaded GBPJPY/15M split=train fold=latest: 46638 bars (2019-01-04 → 2020-12-31)
2026-05-09 07:09:36,909 INFO Loaded GBPUSD/5M split=all fold=latest: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:36,928 INFO Loaded GBPUSD/1H split=all fold=latest: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:36,942 INFO Loaded GBPUSD/4H split=all fold=latest: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:36,949 INFO Loaded GBPUSD/1D split=all fold=latest: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-09 07:09:36,985 INFO Loaded GBPUSD/15M split=train fold=latest: 46637 bars (2019-01-04 → 2020-12-31)
2026-05-09 07:09:37,088 INFO train_multi: 6 segments, ~272001 total bars
2026-05-09 07:09:37,353 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-05-09 07:09:37,353 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-05-09 07:09:37,353 INFO train_multi: training ALL 6 segments across TFs ['15M'] in one combined pass
2026-05-09 07:09:37,353 INFO train_multi: building combined dataset for TF=ALL (6 segments)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:09:40,429 INFO train_multi TF=ALL: 271821 sequences across 6 segments
2026-05-09 07:09:40,429 INFO train_multi TF=ALL: estimated peak RAM = 4632 MB (train=217455 val=54366 n_feat=71 seq_len=30)
2026-05-09 07:09:41,051 INFO train_multi TF=ALL: train=217455 val=54366 (2319 MB tensors)
2026-05-09 07:09:43,613 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=18, min_epochs=22)
2026-05-09 07:09:51,589 INFO train_multi TF=ALL epoch 1/50 train=0.8528 val=0.8526 dir_acc=0.509 dir_n=54366
2026-05-09 07:09:51,599 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:09:51,599 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:09:51,599 INFO train_multi TF=ALL: new best val=0.8526 — saved
2026-05-09 07:09:57,598 INFO train_multi TF=ALL epoch 2/50 train=0.8509 val=0.8466 dir_acc=0.509 dir_n=54366
2026-05-09 07:09:57,603 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:09:57,603 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:09:57,603 INFO train_multi TF=ALL: new best val=0.8466 — saved
2026-05-09 07:10:03,544 INFO train_multi TF=ALL epoch 3/50 train=0.8351 val=0.8147 dir_acc=0.509 dir_n=54366
2026-05-09 07:10:03,549 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:10:03,549 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:10:03,549 INFO train_multi TF=ALL: new best val=0.8147 — saved
2026-05-09 07:10:09,487 INFO train_multi TF=ALL epoch 4/50 train=0.7833 val=0.7243 dir_acc=0.491 dir_n=54366
2026-05-09 07:10:09,492 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:10:09,492 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:10:09,493 INFO train_multi TF=ALL: new best val=0.7243 — saved
2026-05-09 07:10:15,512 INFO train_multi TF=ALL epoch 5/50 train=0.7170 val=0.7056 dir_acc=0.491 dir_n=54366
2026-05-09 07:10:15,517 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:10:15,517 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:10:15,517 INFO train_multi TF=ALL: new best val=0.7056 — saved
2026-05-09 07:10:21,428 INFO train_multi TF=ALL epoch 6/50 train=0.7130 val=0.7054 dir_acc=0.491 dir_n=54366
2026-05-09 07:10:21,433 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:10:21,433 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:10:21,433 INFO train_multi TF=ALL: new best val=0.7054 — saved
2026-05-09 07:10:27,467 INFO train_multi TF=ALL epoch 7/50 train=0.7121 val=0.7051 dir_acc=0.491 dir_n=54366
2026-05-09 07:10:27,471 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:10:27,471 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:10:27,471 INFO train_multi TF=ALL: new best val=0.7051 — saved
2026-05-09 07:10:33,467 INFO train_multi TF=ALL epoch 8/50 train=0.7115 val=0.7048 dir_acc=0.491 dir_n=54366
2026-05-09 07:10:33,472 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:10:33,473 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:10:33,473 INFO train_multi TF=ALL: new best val=0.7048 — saved
2026-05-09 07:10:39,353 INFO train_multi TF=ALL epoch 9/50 train=0.7106 val=0.7046 dir_acc=0.491 dir_n=54366
2026-05-09 07:10:39,358 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:10:39,358 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:10:39,358 INFO train_multi TF=ALL: new best val=0.7046 — saved
2026-05-09 07:10:45,393 INFO train_multi TF=ALL epoch 10/50 train=0.7100 val=0.7042 dir_acc=0.491 dir_n=54366
2026-05-09 07:10:45,398 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:10:45,399 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:10:45,399 INFO train_multi TF=ALL: new best val=0.7042 — saved
2026-05-09 07:10:51,341 INFO train_multi TF=ALL epoch 11/50 train=0.7092 val=0.7035 dir_acc=0.496 dir_n=54366
2026-05-09 07:10:51,348 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:10:51,348 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:10:51,348 INFO train_multi TF=ALL: new best val=0.7035 — saved
2026-05-09 07:10:57,273 INFO train_multi TF=ALL epoch 12/50 train=0.7083 val=0.7032 dir_acc=0.492 dir_n=54366
2026-05-09 07:10:57,278 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:10:57,278 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:10:57,278 INFO train_multi TF=ALL: new best val=0.7032 — saved
2026-05-09 07:11:03,267 INFO train_multi TF=ALL epoch 13/50 train=0.7080 val=0.7032 dir_acc=0.492 dir_n=54366
2026-05-09 07:11:03,272 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:11:03,272 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:11:03,272 INFO train_multi TF=ALL: new best val=0.7032 — saved
2026-05-09 07:11:09,307 INFO train_multi TF=ALL epoch 14/50 train=0.7077 val=0.7031 dir_acc=0.492 dir_n=54366
2026-05-09 07:11:09,312 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:11:09,312 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:11:09,312 INFO train_multi TF=ALL: new best val=0.7031 — saved
2026-05-09 07:11:15,200 INFO train_multi TF=ALL epoch 15/50 train=0.7074 val=0.7028 dir_acc=0.501 dir_n=54366
2026-05-09 07:11:15,206 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:11:15,206 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:11:15,207 INFO train_multi TF=ALL: new best val=0.7028 — saved
2026-05-09 07:11:21,159 INFO train_multi TF=ALL epoch 16/50 train=0.7070 val=0.7028 dir_acc=0.491 dir_n=54366
2026-05-09 07:11:27,172 INFO train_multi TF=ALL epoch 17/50 train=0.7069 val=0.7027 dir_acc=0.498 dir_n=54366
2026-05-09 07:11:27,179 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:11:27,179 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:11:27,179 INFO train_multi TF=ALL: new best val=0.7027 — saved
2026-05-09 07:11:33,201 INFO train_multi TF=ALL epoch 18/50 train=0.7067 val=0.7026 dir_acc=0.503 dir_n=54366
2026-05-09 07:11:33,207 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:11:33,207 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:11:33,207 INFO train_multi TF=ALL: new best val=0.7026 — saved
2026-05-09 07:11:39,195 INFO train_multi TF=ALL epoch 19/50 train=0.7064 val=0.7025 dir_acc=0.498 dir_n=54366
2026-05-09 07:11:39,200 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:11:39,200 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:11:39,200 INFO train_multi TF=ALL: new best val=0.7025 — saved
2026-05-09 07:11:45,105 INFO train_multi TF=ALL epoch 20/50 train=0.7062 val=0.7024 dir_acc=0.503 dir_n=54366
2026-05-09 07:11:45,110 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:11:45,111 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:11:45,111 INFO train_multi TF=ALL: new best val=0.7024 — saved
2026-05-09 07:11:51,041 INFO train_multi TF=ALL epoch 21/50 train=0.7060 val=0.7023 dir_acc=0.499 dir_n=54366
2026-05-09 07:11:51,047 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:11:51,047 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:11:51,047 INFO train_multi TF=ALL: new best val=0.7023 — saved
2026-05-09 07:11:56,981 INFO train_multi TF=ALL epoch 22/50 train=0.7058 val=0.7022 dir_acc=0.507 dir_n=54366
2026-05-09 07:11:56,986 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:11:56,986 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:11:56,987 INFO train_multi TF=ALL: new best val=0.7022 — saved
2026-05-09 07:12:02,885 INFO train_multi TF=ALL epoch 23/50 train=0.7052 val=0.7019 dir_acc=0.508 dir_n=54366
2026-05-09 07:12:02,890 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:12:02,890 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:12:02,890 INFO train_multi TF=ALL: new best val=0.7019 — saved
2026-05-09 07:12:08,709 INFO train_multi TF=ALL epoch 24/50 train=0.7050 val=0.7015 dir_acc=0.515 dir_n=54366
2026-05-09 07:12:08,714 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:12:08,714 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:12:08,714 INFO train_multi TF=ALL: new best val=0.7015 — saved
2026-05-09 07:12:14,648 INFO train_multi TF=ALL epoch 25/50 train=0.7047 val=0.7014 dir_acc=0.516 dir_n=54366
2026-05-09 07:12:14,652 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:12:14,653 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:12:14,653 INFO train_multi TF=ALL: new best val=0.7014 — saved
2026-05-09 07:12:20,505 INFO train_multi TF=ALL epoch 26/50 train=0.7043 val=0.7014 dir_acc=0.512 dir_n=54366
2026-05-09 07:12:20,511 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:12:20,511 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:12:20,511 INFO train_multi TF=ALL: new best val=0.7014 — saved
2026-05-09 07:12:26,479 INFO train_multi TF=ALL epoch 27/50 train=0.7039 val=0.7011 dir_acc=0.514 dir_n=54366
2026-05-09 07:12:26,484 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:12:26,484 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:12:26,484 INFO train_multi TF=ALL: new best val=0.7011 — saved
2026-05-09 07:12:32,472 INFO train_multi TF=ALL epoch 28/50 train=0.7035 val=0.7007 dir_acc=0.519 dir_n=54366
2026-05-09 07:12:32,477 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:12:32,477 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:12:32,477 INFO train_multi TF=ALL: new best val=0.7007 — saved
2026-05-09 07:12:38,424 INFO train_multi TF=ALL epoch 29/50 train=0.7034 val=0.7009 dir_acc=0.520 dir_n=54366
2026-05-09 07:12:44,504 INFO train_multi TF=ALL epoch 30/50 train=0.7030 val=0.7005 dir_acc=0.521 dir_n=54366
2026-05-09 07:12:44,509 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:12:44,509 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:12:44,509 INFO train_multi TF=ALL: new best val=0.7005 — saved
2026-05-09 07:12:50,426 INFO train_multi TF=ALL epoch 31/50 train=0.7027 val=0.7009 dir_acc=0.517 dir_n=54366
2026-05-09 07:12:56,414 INFO train_multi TF=ALL epoch 32/50 train=0.7025 val=0.7012 dir_acc=0.516 dir_n=54366
2026-05-09 07:13:02,463 INFO train_multi TF=ALL epoch 33/50 train=0.7023 val=0.7001 dir_acc=0.528 dir_n=54366
2026-05-09 07:13:02,468 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:13:02,468 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:13:02,468 INFO train_multi TF=ALL: new best val=0.7001 — saved
2026-05-09 07:13:08,414 INFO train_multi TF=ALL epoch 34/50 train=0.7019 val=0.6996 dir_acc=0.529 dir_n=54366
2026-05-09 07:13:08,419 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:13:08,419 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:13:08,419 INFO train_multi TF=ALL: new best val=0.6996 — saved
2026-05-09 07:13:14,317 INFO train_multi TF=ALL epoch 35/50 train=0.7012 val=0.7005 dir_acc=0.524 dir_n=54366
2026-05-09 07:13:20,123 INFO train_multi TF=ALL epoch 36/50 train=0.7007 val=0.6982 dir_acc=0.537 dir_n=54366
2026-05-09 07:13:20,128 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:13:20,128 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:13:20,128 INFO train_multi TF=ALL: new best val=0.6982 — saved
2026-05-09 07:13:26,075 INFO train_multi TF=ALL epoch 37/50 train=0.7001 val=0.6975 dir_acc=0.537 dir_n=54366
2026-05-09 07:13:26,080 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:13:26,080 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:13:26,080 INFO train_multi TF=ALL: new best val=0.6975 — saved
2026-05-09 07:13:32,051 INFO train_multi TF=ALL epoch 38/50 train=0.6991 val=0.7007 dir_acc=0.523 dir_n=54366
2026-05-09 07:13:37,986 INFO train_multi TF=ALL epoch 39/50 train=0.6981 val=0.6962 dir_acc=0.541 dir_n=54366
2026-05-09 07:13:37,992 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:13:37,992 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:13:37,992 INFO train_multi TF=ALL: new best val=0.6962 — saved
2026-05-09 07:13:44,053 INFO train_multi TF=ALL epoch 40/50 train=0.6960 val=0.6949 dir_acc=0.544 dir_n=54366
2026-05-09 07:13:44,057 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:13:44,058 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:13:44,058 INFO train_multi TF=ALL: new best val=0.6949 — saved
2026-05-09 07:13:49,951 INFO train_multi TF=ALL epoch 41/50 train=0.6936 val=0.6916 dir_acc=0.566 dir_n=54366
2026-05-09 07:13:49,957 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:13:49,957 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:13:49,957 INFO train_multi TF=ALL: new best val=0.6916 — saved
2026-05-09 07:13:56,010 INFO train_multi TF=ALL epoch 42/50 train=0.6905 val=0.6858 dir_acc=0.579 dir_n=54366
2026-05-09 07:13:56,015 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:13:56,015 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:13:56,015 INFO train_multi TF=ALL: new best val=0.6858 — saved
2026-05-09 07:14:01,996 INFO train_multi TF=ALL epoch 43/50 train=0.6868 val=0.6804 dir_acc=0.589 dir_n=54366
2026-05-09 07:14:02,001 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:14:02,001 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:14:02,001 INFO train_multi TF=ALL: new best val=0.6804 — saved
2026-05-09 07:14:08,016 INFO train_multi TF=ALL epoch 44/50 train=0.6832 val=0.6785 dir_acc=0.596 dir_n=54366
2026-05-09 07:14:08,021 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:14:08,021 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:14:08,021 INFO train_multi TF=ALL: new best val=0.6785 — saved
2026-05-09 07:14:13,983 INFO train_multi TF=ALL epoch 45/50 train=0.6812 val=0.6764 dir_acc=0.597 dir_n=54366
2026-05-09 07:14:13,989 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:14:13,989 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:14:13,989 INFO train_multi TF=ALL: new best val=0.6764 — saved
2026-05-09 07:14:19,876 INFO train_multi TF=ALL epoch 46/50 train=0.6790 val=0.6773 dir_acc=0.593 dir_n=54366
2026-05-09 07:14:25,808 INFO train_multi TF=ALL epoch 47/50 train=0.6769 val=0.6740 dir_acc=0.604 dir_n=54366
2026-05-09 07:14:25,813 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:14:25,814 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:14:25,814 INFO train_multi TF=ALL: new best val=0.6740 — saved
2026-05-09 07:14:31,745 INFO train_multi TF=ALL epoch 48/50 train=0.6759 val=0.6718 dir_acc=0.609 dir_n=54366
2026-05-09 07:14:31,750 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:14:31,750 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:14:31,750 INFO train_multi TF=ALL: new best val=0.6718 — saved
2026-05-09 07:14:37,841 INFO train_multi TF=ALL epoch 49/50 train=0.6742 val=0.6709 dir_acc=0.607 dir_n=54366
2026-05-09 07:14:37,845 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:14:37,846 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:14:37,846 INFO train_multi TF=ALL: new best val=0.6709 — saved
2026-05-09 07:14:43,822 INFO train_multi TF=ALL epoch 50/50 train=0.6732 val=0.6704 dir_acc=0.612 dir_n=54366
2026-05-09 07:14:43,827 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-09 07:14:43,827 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:14:43,827 INFO train_multi TF=ALL: new best val=0.6704 — saved
2026-05-09 07:14:43,960 INFO Retrain complete. Total wall-clock: 309.2s
2026-05-09 07:14:45,354 INFO Model gru: SUCCESS
2026-05-09 07:14:45,355 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-09 07:14:45,355 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-09 07:14:45,355 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-09 07:14:45,355 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-05-09 07:14:45,355 INFO   [DORMANT] rl_ppo — RL_ENABLED=false, skipping
2026-05-09 07:14:45,355 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-05-09 07:14:45,355 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer']
2026-05-09 07:14:45,356 INFO Saved 5 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Clean Quality/RL source: Backtest on train window ===
  START Train-window backtest for Quality/RL labels
2026-05-09 07:14:46,109 INFO === STEP 6: BACKTEST (train) ===
2026-05-09 07:14:46,110 INFO BT_WINDOW=train — train-window backtest: 2019-01-04 → 2020-12-31 (clean Quality/RL labels)
2026-05-09 07:14:46,110 INFO ================================================================
  ROUND 0 / 3
================================================================
2026-05-09 07:14:46,110 INFO Round 0 — running backtest: 2019-01-04 → 2020-12-31 (ml_trader, shared ML cache)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:16:03,738 WARNING _build_sequence_df: HTF frame 5M filled 397 warmup/alignment gaps with 0.000
2026-05-09 07:16:03,781 WARNING _build_sequence_df: HTF frame 5M filled 397 warmup/alignment gaps with 0.000
2026-05-09 07:16:04,202 WARNING _build_sequence_df: HTF frame 5M filled 397 warmup/alignment gaps with 0.000
2026-05-09 07:16:04,280 WARNING _build_sequence_df: HTF frame 5M filled 397 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:753: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:753: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:759: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:759: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
2026-05-09 07:16:04,537 WARNING _build_sequence_df: HTF frame 5M filled 397 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:767: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:767: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
2026-05-09 07:16:04,618 WARNING _build_sequence_df: HTF frame 5M filled 397 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:753: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
2026-05-09 07:16:04,687 WARNING _build_sequence_df: HTF frame 5M filled 356 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:759: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
2026-05-09 07:16:04,761 WARNING _build_sequence_df: HTF frame 5M filled 356 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:767: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
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
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:574: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bull = is_bearish_sh.shift(1).fillna(False) & bos_bull.fillna(False)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../indicators/market_structure.py:575: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  mss_bear = (is_bullish_sh | is_bullish_sl).shift(1).fillna(False) & bos_bear.fillna(False)
2026-05-09 07:16:12,300 WARNING _build_sequence_df: HTF frame 5M filled 396 warmup/alignment gaps with 0.000
2026-05-09 07:16:12,329 WARNING _build_sequence_df: HTF frame 5M filled 396 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:753: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema21_dist"] = _htf_series(df_1h, "1H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1h_ema50_dist"] = _htf_series(df_1h, "1H",
2026-05-09 07:16:12,387 WARNING _build_sequence_df: HTF frame 5M filled 395 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:759: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_ema21_ema50_diff"] = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:761: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_adx"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_4h_rsi"]      = _htf_series(df_4h, "4H",
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:767: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema21_dist"] = _htf_series(df_1d, "1D",
2026-05-09 07:16:12,439 WARNING _build_sequence_df: HTF frame 5M filled 395 warmup/alignment gaps with 0.000
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:769: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  out["mtf_1d_ema_stack"]  = _htf_series(df_1d, "1D",
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

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260509_071448.json
Trader                                   Trades      WR    PF*  Return*    ExpR   TP1%   TP2%     DD*  Sharpe*  Sortino     IC  Verdict
-----------------------------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)                0   0.0%   0.00    0.0%   0.000  0.0%  0.0%   0.0%     0.00     0.00  0.000     FAIL
  MonteCarlo P95 DD=0.0%  P10 equity=10,000  t=0.00 (p=1.000)  Sharpe CI=[0.00, 0.00]  streak=0
  gate_diagnostics: bars=279801 no_signal=131364 quality_block=0 session_skip=148437 density=0 pm_reject=0 daily_skip=0 cooldown=0 daily_halt_events=0 enforce_daily_halt=False
  no_signal_reasons: high_uncertainty=131364

Calibration Summary:
  all          [N/A] No outcome data yet
2026-05-09 07:16:47,689 INFO Round 0 backtest — 0 trades | avg WR=0.0% | avg PF=0.00 | avg Sharpe=0.00
2026-05-09 07:16:47,689 INFO   ml_trader: 0 trades | WR=0.0% | fixed PF=0.00 | Return=0.0% | ExpR=0.000 | DD=0.0% | Sharpe=0.00
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_0.json
INFO  Total trades: 0
ERROR  trade_log is empty.
2026-05-09 07:16:47,906 WARNING Round 0: trade_log is empty — nothing to journal
2026-05-09 07:16:47,906 WARNING Round 0: no trades to journal
2026-05-09 07:16:48,103 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-05-09 07:16:48,103 WARNING Journal missing or empty at /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/trade_journal_detailed.jsonl — backtest produced no trades yet. Skipping Quality+RL training (will train after first successful backtest).

======================================================================
  BACKTEST COMPLETE  (round 0 / window=train)
======================================================================
  Round     Trades       WR     PF*  Sharpe*
  ------------------------------------------
  Round 0          0      0.0%    0.000     0.000

  DONE  Train-window backtest for Quality/RL labels
  Saved Train Quality/RL source result → train_quality_rl_source_summary.json
  Train-label journal entries: 0

=== Train Quality + RL on train-only journal ===
  START Train-only Quality+RL retrain
  DONE  Train-only Quality+RL retrain

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  START Round 1 - Backtest (val)
2026-05-09 07:16:48,595 INFO === STEP 6: BACKTEST (round1) ===
2026-05-09 07:16:48,596 INFO BT_WINDOW=round1 — val-window backtest: 2021-01-04 → 2022-01-03 (test set protected)
2026-05-09 07:16:48,596 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-05-09 07:16:48,597 INFO Round 1 — running backtest: 2021-01-04 → 2022-01-03 (ml_trader, shared ML cache)
2026-05-09 07:16:50,937 ERROR QualityScorer load failed: QualityScorer unavailable; train quality_scorer.pkl before backtesting with ML
Traceback (most recent call last):
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 3840, in <module>
    main()
  File "/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/run_backtest.py", line 3604, in main
    raise RuntimeError("QualityScorer unavailable; train quality_scorer.pkl before backtesting with ML")
RuntimeError: QualityScorer unavailable; train quality_scorer.pkl before backtesting with ML
2026-05-09 07:16:51,470 ERROR Backtest failed (rc=1) — check trading-engine/logs/backtest_*.log
2026-05-09 07:16:51,470 ERROR Round 1 backtest failed: backtest exited 1